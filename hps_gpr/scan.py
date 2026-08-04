"""Scan driver and CSV writers."""

import json
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import joblib
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:
    import contextlib
    _threadpool_limits = contextlib.nullcontext  # type: ignore[assignment]

from .evaluation import (
    evaluate_single_dataset,
    evaluate_combined,
    active_datasets_for_mass,
    _dataset_visibility,
)
from .plotting import (
    make_mass_folder,
    ensure_dir,
    plot_full_range,
    plot_blind_window,
    plot_s_over_b,
)

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def union_scan_grid(
    datasets: Dict[str, "DatasetConfig"], step: float
) -> np.ndarray:
    """Generate mass scan grid covering all datasets.

    Args:
        datasets: Dictionary of dataset configurations
        step: Mass step size (GeV)

    Returns:
        Array of mass values
    """
    lo = min([d.m_low for d in datasets.values()])
    hi = max([d.m_high for d in datasets.values()])
    masses = np.arange(lo, hi + 0.5 * step, step)
    return np.round(masses, 3)


def _write_json(path: str, payload: dict) -> None:
    """Write a dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _gp_diagnostics_payload(pred) -> dict:
    """Build the nested, JSON-safe provenance block for one GP fit."""
    blind_train = getattr(pred, "blind_train", None)
    return {
        "kernel": str(getattr(pred, "kernel_str", "")),
        "length_scale": {
            "lower": _jfloat(getattr(pred, "ls_lo", float("nan"))),
            "upper": _jfloat(getattr(pred, "ls_hi", float("nan"))),
            "initial": _jfloat(getattr(pred, "ls_init", float("nan"))),
            "optimized": _jfloat(getattr(pred, "ls_opt", float("nan"))),
            "at_lower": bool(getattr(pred, "ls_at_lower", False)),
            "at_upper": bool(getattr(pred, "ls_at_upper", False)),
            "sigma_x": _jfloat(getattr(pred, "sigma_x", float("nan"))),
        },
        "constant": {
            "lower": _jfloat(getattr(pred, "const_lo", float("nan"))),
            "upper": _jfloat(getattr(pred, "const_hi", float("nan"))),
            "initial": _jfloat(getattr(pred, "const_init", float("nan"))),
            "optimized": _jfloat(getattr(pred, "const_opt", float("nan"))),
            "at_lower": bool(getattr(pred, "const_at_lower", False)),
            "at_upper": bool(getattr(pred, "const_at_upper", False)),
        },
        "training": {
            "domain_lo_GeV": _jfloat(getattr(pred, "train_domain_lo", float("nan"))),
            "domain_hi_GeV": _jfloat(getattr(pred, "train_domain_hi", float("nan"))),
            "exclude_lo_GeV": (
                _jfloat(blind_train[0]) if blind_train is not None else None
            ),
            "exclude_hi_GeV": (
                _jfloat(blind_train[1]) if blind_train is not None else None
            ),
            "n_full": int(getattr(pred, "n_full", 0)),
            "n_blind": int(getattr(pred, "n_blind", 0)),
            "n_train": int(getattr(pred, "n_train", 0)),
            "n_train_low": int(getattr(pred, "n_train_low", 0)),
            "n_train_high": int(getattr(pred, "n_train_high", 0)),
            "bin_width_median_GeV": _jfloat(
                getattr(pred, "bin_width_median", float("nan"))
            ),
        },
        "optimizer": {
            "restarts": int(getattr(pred, "optimizer_restarts", 0)),
            "log_marginal_likelihood": _jfloat(getattr(pred, "lml", float("nan"))),
        },
    }


def _density_diagnostics_payload(pred) -> dict:
    """Build the JSON-safe physical density-window provenance block."""
    return {
        "counts_per_GeV": _jfloat(
            getattr(pred, "integral_density", float("nan"))
        ),
        "nsigma": _jfloat(getattr(pred, "density_nsigma", float("nan"))),
        "window_lo_GeV": _jfloat(
            getattr(pred, "density_window_lo", float("nan"))
        ),
        "window_hi_GeV": _jfloat(
            getattr(pred, "density_window_hi", float("nan"))
        ),
        "window_width_GeV": _jfloat(
            getattr(pred, "density_window_width", float("nan"))
        ),
        "source_lo_GeV": _jfloat(
            getattr(pred, "density_source_lo", float("nan"))
        ),
        "source_hi_GeV": _jfloat(
            getattr(pred, "density_source_hi", float("nan"))
        ),
        "source_n_bins": int(getattr(pred, "density_source_n_bins", 0)),
        "source_bin_width_median_GeV": _jfloat(
            getattr(pred, "density_source_bin_width_median", float("nan"))
        ),
        "fully_covered": bool(
            getattr(pred, "density_window_fully_covered", False)
        ),
    }


def run_scan(
    datasets: Dict[str, "DatasetConfig"],
    config: "Config",
    mass_min: float = None,
    mass_max: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full mass scan.

    Uses active_datasets_for_mass() from evaluation.py, which supports the
    config-based edge guards (scan_require_two_sidebands, scan_edge_guard_nsigma).
    Supports joblib parallelization via config.scan_parallel / scan_n_workers.

    Args:
        datasets: Dictionary of enabled datasets
        config: Global configuration
        mass_min: Minimum mass to scan (optional)
        mass_max: Maximum mass to scan (optional)

    Returns:
        Tuple of (single-dataset DataFrame, combined DataFrame)
    """
    masses = union_scan_grid(datasets, config.mass_step_gev)
    if mass_min is not None:
        masses = masses[masses >= mass_min]
    if mass_max is not None:
        masses = masses[masses <= mass_max]

    scan_parallel = bool(getattr(config, "scan_parallel", False))
    n_workers = int(getattr(config, "scan_n_workers", 1) or 1)
    backend = str(getattr(config, "scan_parallel_backend", "loky"))
    threads_per_worker = int(getattr(config, "scan_threads_per_worker", 1) or 1)
    do_combined = bool(getattr(config, "do_combined", False))

    # scan_diagnostic_* is currently a no-op performance knob until a real
    # diagnostic writer consumes fit details.

    # Apply publication-style plotting defaults once for the full scan.
    if bool(getattr(config, "save_plots", False)):
        try:
            from .plotting import set_plot_style
            set_plot_style("paper")
        except Exception:
            pass

    def _process_one_mass(m: float) -> Tuple[List[dict], List[dict]]:
        """Process a single mass point. Returns (rows_single, rows_comb)."""
        rows_s: List[dict] = []
        rows_c: List[dict] = []

        print(f"[scan] testing mass hypothesis {float(m):.3f} GeV", flush=True)

        ds_here = active_datasets_for_mass(float(m), datasets, config)
        if not ds_here:
            return rows_s, rows_c

        mass_dir = (
            make_mass_folder(config.output_dir, float(m))
            if config.save_per_mass_folders
            else config.output_dir
        )

        preds_here = []
        ds_list_here = []

        for ds in ds_here:
            ds_dir = os.path.join(mass_dir, ds.key)
            ensure_dir(ds_dir)
            compute_obs = (_dataset_visibility(ds, config) == "observed")

            try:
                with _threadpool_limits(limits=int(threads_per_worker)):
                    res, pred, _ = evaluate_single_dataset(
                        ds, float(m), config,
                        do_extraction=True,
                        compute_observed=compute_obs,
                        return_fit_details=False,
                    )

                preds_here.append(pred)
                ds_list_here.append(ds)

                if config.save_plots and compute_obs:
                    try:
                        plot_full_range(
                            ds, float(m), pred,
                            os.path.join(ds_dir, "fit_full.png"),
                            A_show=res.A_up,
                            config=config,
                        )
                        plot_blind_window(
                            ds, float(m), pred,
                            os.path.join(ds_dir, "blind_fit.png"),
                            A_up=res.A_up,
                            A_hat=res.A_hat,
                            config=config,
                        )
                        plot_s_over_b(
                            ds, float(m), pred, res.A_up,
                            os.path.join(ds_dir, "s_over_b_ul.png"),
                            config=config,
                        )
                    except Exception as pe:
                        if config.debug_print:
                            print(f"[WARN] plot failure for {ds.key} @ {float(m):.4f} GeV: {pe}")

                if config.save_fit_json:
                    _write_json(
                        os.path.join(ds_dir, "numbers.json"),
                        {
                            "dataset": ds.key,
                            "mass_GeV": float(m),
                            "A_up": _jfloat(res.A_up),
                            "eps2_up": _jfloat(res.eps2_up),
                            "p0_analytic": _jfloat(res.p0_analytic),
                            "Z_analytic": _jfloat(res.Z_analytic),
                            "A_hat": _jfloat(res.A_hat),
                            "sigma_A": _jfloat(res.sigma_A),
                            "extract_success": bool(res.extract_success),
                            "sigma_val": _jfloat(pred.sigma_val),
                            "blind": [_jfloat(pred.blind[0]), _jfloat(pred.blind[1])],
                            "integral_density": _jfloat(pred.integral_density),
                            "density": _density_diagnostics_payload(pred),
                            "gp_diagnostics": _gp_diagnostics_payload(pred),
                            "cls_statistic": "tilde_q_mu",
                            "cls_calibration": str(config.cls_mode).lower().strip(),
                            "signal_model": str(getattr(config, "signal_model", "default")),
                            "global_method": "sidak_approx",
                            "visibility": "observed" if compute_obs else "expected_only",
                        },
                    )

                rows_s.append({
                    "dataset": ds.key,
                    "mass_GeV": float(res.mass),
                    "sigma_val": float(pred.sigma_val),
                    "blind_lo": float(pred.blind[0]),
                    "blind_hi": float(pred.blind[1]),
                    "integral_density": float(pred.integral_density),
                    "density_nsigma": float(
                        getattr(pred, "density_nsigma", float("nan"))
                    ),
                    "density_window_lo": float(
                        getattr(pred, "density_window_lo", float("nan"))
                    ),
                    "density_window_hi": float(
                        getattr(pred, "density_window_hi", float("nan"))
                    ),
                    "density_window_width": float(
                        getattr(pred, "density_window_width", float("nan"))
                    ),
                    "density_source_lo": float(
                        getattr(pred, "density_source_lo", float("nan"))
                    ),
                    "density_source_hi": float(
                        getattr(pred, "density_source_hi", float("nan"))
                    ),
                    "density_source_n_bins": int(
                        getattr(pred, "density_source_n_bins", 0)
                    ),
                    "density_source_bin_width_median": float(
                        getattr(
                            pred,
                            "density_source_bin_width_median",
                            float("nan"),
                        )
                    ),
                    "density_window_fully_covered": bool(
                        getattr(pred, "density_window_fully_covered", False)
                    ),
                    "A_up": float(res.A_up),
                    "eps2_up": float(res.eps2_up),
                    "p0_analytic": float(res.p0_analytic),
                    "Z_analytic": float(res.Z_analytic),
                    "A_hat": float(res.A_hat),
                    "sigma_A": float(res.sigma_A),
                    "extract_success": bool(res.extract_success),
                    "cls_statistic": "tilde_q_mu",
                    "cls_calibration": str(config.cls_mode).lower().strip(),
                    "signal_model": str(getattr(config, "signal_model", "default")),
                    "global_method": "sidak_approx",
                    "visibility": "observed" if compute_obs else "expected_only",
                    "kernel_str": str(getattr(pred, "kernel_str", "")),
                    "ls_lo": float(getattr(pred, "ls_lo", float("nan"))),
                    "ls_hi": float(getattr(pred, "ls_hi", float("nan"))),
                    "ls_init": float(getattr(pred, "ls_init", float("nan"))),
                    "ls_opt": float(getattr(pred, "ls_opt", float("nan"))),
                    "sigma_x": float(getattr(pred, "sigma_x", float("nan"))),
                    "const_opt": float(getattr(pred, "const_opt", float("nan"))),
                    "lml": float(getattr(pred, "lml", float("nan"))),
                    "n_train": int(getattr(pred, "n_train", 0)),
                    "n_train_low": int(getattr(pred, "n_train_low", 0)),
                    "n_train_high": int(getattr(pred, "n_train_high", 0)),
                    "n_full": int(getattr(pred, "n_full", 0)),
                    "n_blind": int(getattr(pred, "n_blind", 0)),
                    "train_domain_lo": float(
                        getattr(pred, "train_domain_lo", float("nan"))
                    ),
                    "train_domain_hi": float(
                        getattr(pred, "train_domain_hi", float("nan"))
                    ),
                    "bin_width_median": float(
                        getattr(pred, "bin_width_median", float("nan"))
                    ),
                    "const_init": float(getattr(pred, "const_init", float("nan"))),
                    "const_lo": float(getattr(pred, "const_lo", float("nan"))),
                    "const_hi": float(getattr(pred, "const_hi", float("nan"))),
                    "const_at_lower": bool(getattr(pred, "const_at_lower", False)),
                    "const_at_upper": bool(getattr(pred, "const_at_upper", False)),
                    "ls_at_lower": bool(getattr(pred, "ls_at_lower", False)),
                    "ls_at_upper": bool(getattr(pred, "ls_at_upper", False)),
                    "optimizer_restarts": int(
                        getattr(pred, "optimizer_restarts", 0)
                    ),
                })

            except Exception as e:
                try:
                    with open(os.path.join(ds_dir, "error.txt"), "w") as ef:
                        ef.write(str(e) + "\n")
                except Exception:
                    pass

                if config.debug_print:
                    print(f"[ERROR] {ds.key} @ {float(m):.4f} GeV: {e}")

                rows_s.append({
                    "dataset": ds.key,
                    "mass_GeV": float(m),
                    "sigma_val": float("nan"),
                    "blind_lo": float("nan"), "blind_hi": float("nan"),
                    "A_up": float("nan"), "eps2_up": float("nan"),
                    "p0_analytic": float("nan"), "Z_analytic": float("nan"),
                    "A_hat": float("nan"), "sigma_A": float("nan"),
                    "extract_success": False,
                    "visibility": "error",
                    "error": str(e),
                })

        # Combined fit (only when do_combined=True and >=2 datasets with data)
        if do_combined and len(ds_list_here) >= 2:
            all_obs = all(
                _dataset_visibility(ds, config) == "observed" for ds in ds_list_here
            )
            try:
                comb = evaluate_combined(float(m), ds_list_here, preds_here, config)

                if config.save_plots and all_obs:
                    cdir = os.path.join(mass_dir, "combined")
                    ensure_dir(cdir)
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.axis("off")
                    ax.text(0.05, 0.8, f"Combined @ {float(m):.4f} GeV",
                            fontsize=12, transform=ax.transAxes)
                    ax.text(0.05, 0.5, f"eps2_up = {comb.eps2_up:.3e}",
                            fontsize=12, transform=ax.transAxes)
                    ax.text(0.05, 0.2,
                            f"p0 = {comb.p0_analytic:.3e}   Z = {comb.Z_analytic:.2f}",
                            fontsize=12, transform=ax.transAxes)
                    plt.tight_layout()
                    plt.savefig(os.path.join(cdir, "combined_summary.png"), dpi=160)
                    plt.close(fig)

                if config.save_fit_json:
                    cdir = os.path.join(mass_dir, "combined")
                    ensure_dir(cdir)
                    _write_json(
                        os.path.join(cdir, "numbers.json"),
                        {
                            "mass_GeV": float(m),
                            "datasets": [d.key for d in ds_list_here],
                            "eps2_up": _jfloat(comb.eps2_up),
                            "p0_analytic": _jfloat(comb.p0_analytic),
                            "Z_analytic": _jfloat(comb.Z_analytic),
                            "gp_diagnostics_by_dataset": {
                                ds.key: _gp_diagnostics_payload(pred)
                                for ds, pred in zip(ds_list_here, preds_here)
                            },
                            "density_by_dataset": {
                                ds.key: _density_diagnostics_payload(pred)
                                for ds, pred in zip(ds_list_here, preds_here)
                            },
                            "cls_statistic": "tilde_q_mu",
                            "cls_calibration": str(config.cls_mode).lower().strip(),
                            "signal_model": str(getattr(config, "signal_model", "default")),
                            "global_method": "sidak_approx",
                        },
                    )

                rows_c.append({
                    "mass_GeV": float(comb.mass),
                    "datasets": "+".join([d.key for d in ds_list_here]),
                    "n_datasets": len(ds_list_here),
                    "eps2_up": float(comb.eps2_up),
                    "p0_analytic": float(comb.p0_analytic),
                    "Z_analytic": float(comb.Z_analytic),
                    "cls_statistic": "tilde_q_mu",
                    "cls_calibration": str(config.cls_mode).lower().strip(),
                    "signal_model": str(getattr(config, "signal_model", "default")),
                    "global_method": "sidak_approx",
                })

            except Exception as e:
                rows_c.append({
                    "mass_GeV": float(m),
                    "datasets": "+".join([d.key for d in ds_list_here]),
                    "n_datasets": len(ds_list_here),
                    "eps2_up": float("nan"),
                    "p0_analytic": float("nan"),
                    "Z_analytic": float("nan"),
                    "error": str(e),
                })
        return rows_s, rows_c

    # Run (parallel or sequential)
    if scan_parallel and n_workers > 1 and _HAVE_JOBLIB:
        results = joblib.Parallel(n_jobs=int(n_workers), backend=str(backend))(
            joblib.delayed(_process_one_mass)(float(m)) for m in masses
        )
    else:
        results = [_process_one_mass(float(m)) for m in masses]

    rows_single: List[dict] = []
    rows_comb: List[dict] = []
    for rs, rc in results:
        rows_single.extend(rs)
        rows_comb.extend(rc)

    df_single = pd.DataFrame(rows_single)
    if {"dataset", "mass_GeV"}.issubset(df_single.columns):
        df_single = df_single.sort_values(["dataset", "mass_GeV"]).reset_index(drop=True)

    if len(df_single) and {"ls_lo", "ls_hi", "ls_opt", "sigma_x", "sigma_val"}.issubset(df_single.columns):
        sx = df_single["sigma_x"].to_numpy(float)
        sg = df_single["sigma_val"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            df_single["ls_lo_over_sigma_x"] = df_single["ls_lo"].to_numpy(float) / sx
            df_single["ls_hi_over_sigma_x"] = df_single["ls_hi"].to_numpy(float) / sx
            df_single["ls_opt_over_sigma_x"] = df_single["ls_opt"].to_numpy(float) / sx
            df_single["ls_lo_over_sigma"] = df_single["ls_lo"].to_numpy(float) / sg
            df_single["ls_hi_over_sigma"] = df_single["ls_hi"].to_numpy(float) / sg
            df_single["ls_opt_over_sigma"] = df_single["ls_opt"].to_numpy(float) / sg

    df_comb = pd.DataFrame(rows_comb)
    if "mass_GeV" in df_comb.columns:
        df_comb = df_comb.sort_values(["mass_GeV"]).reset_index(drop=True)

    single_path = os.path.join(config.output_dir, "results_single.csv")
    comb_path = os.path.join(config.output_dir, "results_combined.csv")
    df_single.to_csv(single_path, index=False)
    df_comb.to_csv(comb_path, index=False)

    # Backward-compatible alias
    comb_alias = os.path.join(config.output_dir, "combined.csv")
    df_comb.to_csv(comb_alias, index=False)

    print("Wrote:", single_path)
    print("Wrote:", comb_path)
    print("Wrote:", comb_alias)

    return df_single, df_comb


def _jfloat(x) -> object:
    """Convert to float for JSON, replacing inf/nan with None."""
    import math
    v = float(x)
    return None if (math.isnan(v) or math.isinf(v)) else v
