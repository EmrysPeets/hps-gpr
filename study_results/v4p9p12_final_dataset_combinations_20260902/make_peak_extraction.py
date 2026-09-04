#!/usr/bin/env python3
"""Extract the three campaign contributions at the all-three local-p0 minimum."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMPAIGN = REPO / "study_results/v4p9p7_2016_support_combined_100toy_20260902"
sys.path.insert(0, str(CAMPAIGN))
from runtime_guard import activate_and_verify  # noqa: E402


activate_and_verify()
sys.path.insert(0, str(HERE / "runtime"))
sys.path.insert(0, str(HERE))

import run_final_combinations as workflow  # noqa: E402
from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.conversion import A_from_epsilon2  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import build_combined_components  # noqa: E402
from hps_gpr.statistics import (  # noqa: E402
    fit_A_profiled_gaussian_details,
    p0_profiled_gaussian_LRT,
    profiled_gaussian_fixed_poi_nll,
)
from hps_gpr.template import build_window_template_from_full  # noqa: E402
from piecewise_cached_solver import CachedPiecewiseBoundedLimit  # noqa: E402


CARD = HERE / "inputs" / "analysis_card.yaml"
STATES = HERE / "inputs" / "reviewed_gp_states.csv"
PROVENANCE = HERE / "inputs" / "analysis_input_provenance.json"
CURVES = HERE / "derived" / "final_dataset_result_curves.csv"
OUTPUT = HERE / "derived"
FIGURES = HERE / "figures"
KEYS = ("2015", "2016", "2021")
COLORS = {
    "gp": "#4C78A8",
    "background": "#E69F00",
    "shared": "#CC3311",
    "independent": "#009E73",
    "window": "#7A3E9D",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_status(fit: Dict[str, object]) -> Dict[str, object]:
    return {
        key: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else float(value)
            if isinstance(value, (int, float, np.integer, np.floating))
            else str(value)
        )
        for key, value in fit.items()
        if key in {"success", "nll", "A_hat", "sigma_A"}
    }


def require_unbounded_fit_nesting(
    fit: Dict[str, object],
    null: Dict[str, object],
    label: str,
) -> None:
    if not bool(fit.get("success", False)) or not bool(null.get("success", False)):
        raise RuntimeError(f"{label} extraction optimizer did not converge")
    nll_fit = float(fit.get("nll", float("nan")))
    nll_null = float(null.get("nll", float("nan")))
    tolerance = 1.0e-6 + 1.0e-8 * max(1.0, abs(nll_null - nll_fit))
    if not (
        np.isfinite(nll_fit)
        and np.isfinite(nll_null)
        and nll_fit <= nll_null + tolerance
    ):
        raise RuntimeError(f"{label} extraction violates likelihood nesting")


def split(vector: np.ndarray, predictions: Dict[str, object]) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    start = 0
    for key in KEYS:
        stop = start + len(predictions[key].obs)
        output[key] = np.asarray(vector[start:stop], dtype=float)
        start = stop
    if start != len(vector):
        raise RuntimeError("combined extraction vector did not split cleanly")
    return output


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.4,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    config = load_config(CARD)
    workflow.result_config = config
    workflow.validate_card(config)
    workflow.validate_input_provenance(PROVENANCE, CARD, STATES, config)
    workflow.validate_histogram_inputs(config)
    states_frame = workflow.load_states(STATES, config)
    states = workflow.state_map(states_frame)
    datasets = make_datasets(config)
    curves = pd.read_csv(CURVES)
    triple = curves[curves.scope_key == "all_2015_2016_2021"].copy()
    if len(triple) != 41:
        raise RuntimeError("all-three result is not the exact 50--90 MeV grid")
    peak = triple.loc[triple.p0_local_asymptotic.idxmin()]
    mass_mev = int(peak.mass_MeV)
    mass = mass_mev / 1000.0

    predictions, conditioned, conditioning, _ = workflow.reconstruct_predictions(
        mass, datasets, config, states
    )
    ds_here = [datasets[key] for key in KEYS]
    pred_here = [predictions[key] for key in KEYS]
    obs, bkg, _raw_cov, s_unit = build_combined_components(
        mass, ds_here, pred_here, config=config
    )
    cov = workflow.block_diagonal([conditioned[key] for key in KEYS])
    signal_scale = float(np.sum(s_unit))
    shared_template = s_unit / signal_scale
    shared_fit = fit_A_profiled_gaussian_details(
        obs, bkg, cov, shared_template, allow_negative=True
    )
    shared_null = profiled_gaussian_fixed_poi_nll(
        obs, bkg, cov, shared_template, A_fixed=0.0
    )
    require_unbounded_fit_nesting(shared_fit, shared_null, "shared")
    shared_eps2 = float(shared_fit["A_hat"]) / signal_scale
    shared_sigma_eps2 = float(shared_fit["sigma_A"]) / signal_scale

    p0, z_value, q0, p0_info = p0_profiled_gaussian_LRT(
        obs, bkg, cov, shared_template
    )
    if not bool(p0_info.get("ok", False)):
        raise RuntimeError("all-three p0 profile did not converge")
    p0_nll_alt = float(p0_info.get("nll_alt", float("nan")))
    p0_nll_null = float(p0_info.get("nll0", float("nan")))
    p0_nll_tolerance = 1.0e-6 + 1.0e-8 * max(
        1.0, abs(p0_nll_null - p0_nll_alt)
    )
    if not (
        np.isfinite(p0_nll_alt)
        and np.isfinite(p0_nll_null)
        and p0_nll_alt <= p0_nll_null + p0_nll_tolerance
    ):
        raise RuntimeError("all-three p0 extraction violates likelihood nesting")
    if not (np.isfinite(shared_eps2) and shared_eps2 > 0.0):
        raise RuntimeError(
            "the selected all-three excess does not have a positive interior shared fit"
        )
    solver = CachedPiecewiseBoundedLimit(
        bkg,
        cov,
        s_unit,
        alpha=float(config.cls_alpha),
        combined_mode=str(config.combined_mode),
    )
    limit = solver.limit(obs)
    closures = {
        "p0": (float(p0), float(peak.p0_local_asymptotic), 2.0e-10, 1.0e-300),
        "Z": (float(z_value), float(peak.Z_local_asymptotic), 2.0e-10, 1.0e-12),
        "eps2_90": (float(limit.eps2_90), float(peak.eps2_90), 2.0e-8, 1.0e-16),
        "shared_eps2_hat": (
            shared_eps2,
            float(peak.eps2_hat_bounded_for_p0),
            2.0e-9,
            1.0e-18,
        ),
        "shared_sigma_eps2": (
            shared_sigma_eps2,
            float(peak.sigma_eps2_hat_bounded_for_p0),
            2.0e-9,
            1.0e-18,
        ),
    }
    for name, (actual, expected, rtol, atol) in closures.items():
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            raise RuntimeError(
                f"extraction does not reproduce {name}: {actual} vs {expected}"
            )

    bfit_by_dataset = split(np.asarray(shared_fit["b_fit"], float), predictions)
    lambda_by_dataset = split(
        np.asarray(shared_fit["lambda_hat"], float), predictions
    )
    summary_rows: List[Dict[str, object]] = []
    plot_rows: List[Dict[str, object]] = []
    independent: Dict[str, Dict[str, object]] = {}

    for key, dataset, prediction in zip(KEYS, ds_here, pred_here):
        window_template, full_template = build_window_template_from_full(
            prediction.edges_full,
            prediction.blind_mask,
            mass,
            prediction.sigma_val,
            config=config,
        )
        k_factor = float(
            A_from_epsilon2(
                dataset, mass, 1.0, prediction.integral_density
            )
        )
        s_window = k_factor * np.asarray(window_template, dtype=float)
        scale_here = float(np.sum(s_window))
        signed_template = s_window / scale_here
        signed_fit = fit_A_profiled_gaussian_details(
            prediction.obs,
            prediction.mu,
            conditioned[key],
            signed_template,
            allow_negative=True,
        )
        signed_null = profiled_gaussian_fixed_poi_nll(
            prediction.obs,
            prediction.mu,
            conditioned[key],
            signed_template,
            A_fixed=0.0,
        )
        require_unbounded_fit_nesting(signed_fit, signed_null, f"independent {key}")
        independent_eps2 = float(signed_fit["A_hat"]) / scale_here
        independent_sigma_eps2 = float(signed_fit["sigma_A"]) / scale_here
        independent[key] = {
            "eps2_hat": independent_eps2,
            "sigma_eps2": independent_sigma_eps2,
            "fit": fit_status(signed_fit),
            "null": fit_status(signed_null),
        }

        shared_window_yield = shared_eps2 * scale_here
        shared_window_sigma = shared_sigma_eps2 * scale_here
        shared_full_yield = shared_eps2 * k_factor
        shared_full_sigma = shared_sigma_eps2 * k_factor
        independent_window_yield = float(signed_fit["A_hat"])
        independent_window_sigma = float(signed_fit["sigma_A"])
        independent_full_yield = independent_eps2 * k_factor
        independent_full_sigma = independent_sigma_eps2 * k_factor
        summary_rows.append(
            {
                "mass_MeV": mass_mev,
                "dataset": key,
                "shared_eps2_hat": shared_eps2,
                "shared_sigma_eps2": shared_sigma_eps2,
                "shared_fitted_window_yield": shared_window_yield,
                "shared_fitted_window_sigma": shared_window_sigma,
                "shared_full_template_yield": shared_full_yield,
                "shared_full_template_sigma": shared_full_sigma,
                "independent_signed_eps2_hat": independent_eps2,
                "independent_signed_sigma_eps2": independent_sigma_eps2,
                "independent_signed_fitted_window_yield": independent_window_yield,
                "independent_signed_fitted_window_sigma": independent_window_sigma,
                "independent_signed_full_template_yield": independent_full_yield,
                "independent_signed_full_template_sigma": independent_full_sigma,
                "signal_yield_per_eps2_fitted_window": scale_here,
                "signal_yield_per_eps2_full_template": k_factor,
                "post_selection_diagnostic": True,
            }
        )

        profiled_full = np.asarray(prediction.mu_full, dtype=float).copy()
        profiled_full[prediction.blind_mask] = bfit_by_dataset[key]
        shared_signal_full = shared_eps2 * k_factor * np.asarray(full_template)
        independent_signal_full = (
            independent_eps2 * k_factor * np.asarray(full_template)
        )
        shared_total = profiled_full + shared_signal_full
        independent_profiled_full = np.asarray(
            prediction.mu_full, dtype=float
        ).copy()
        independent_profiled_full[prediction.blind_mask] = np.asarray(
            signed_fit["b_fit"], dtype=float
        )
        independent_total = independent_profiled_full + independent_signal_full
        if not np.allclose(
            shared_total[prediction.blind_mask],
            lambda_by_dataset[key],
            rtol=2.0e-9,
            atol=2.0e-6,
        ):
            raise RuntimeError(f"shared extraction split failed for {key}")
        for index, center in enumerate(prediction.x_full):
            plot_rows.append(
                {
                    "mass_MeV": mass_mev,
                    "dataset": key,
                    "bin_center_GeV": float(center),
                    "bin_center_MeV": float(center * 1000.0),
                    "observed_events": float(prediction.y_full[index]),
                    "gp_mean_events": float(prediction.mu_full[index]),
                    "joint_profiled_background_events": float(profiled_full[index]),
                    "independent_profiled_background_events": float(
                        independent_profiled_full[index]
                    ),
                    "shared_signal_events": float(shared_signal_full[index]),
                    "shared_total_events": float(shared_total[index]),
                    "independent_signed_signal_events": float(
                        independent_signal_full[index]
                    ),
                    "independent_signed_total_events": float(
                        independent_total[index]
                    ),
                    "in_fitted_window": bool(prediction.blind_mask[index]),
                    "bin_width_MeV": float(
                        1000.0
                        * (prediction.edges_full[index + 1] - prediction.edges_full[index])
                    ),
                }
            )

    table = pd.DataFrame(summary_rows)
    plot_data = pd.DataFrame(plot_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "all_three_peak_extraction_table.csv", index=False)
    plot_data.to_csv(
        OUTPUT / "all_three_peak_extraction_plot_data.csv", index=False
    )

    style()
    fig = plt.figure(figsize=(12.2, 7.2))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.65, 1.0),
        hspace=0.27,
        wspace=0.25,
        left=0.07,
        right=0.985,
        top=0.78,
        bottom=0.10,
    )
    handles = labels = None
    for column, key in enumerate(KEYS):
        prediction = predictions[key]
        data = plot_data[plot_data.dataset.astype(str) == key].copy()
        sigma_mev = float(prediction.sigma_val * 1000.0)
        display = data[
            data.bin_center_MeV.between(
                mass_mev - 6.0 * sigma_mev,
                mass_mev + 6.0 * sigma_mev,
            )
        ]
        ax = fig.add_subplot(grid[0, column])
        pull_ax = fig.add_subplot(grid[1, column], sharex=ax)
        width = display.bin_width_MeV.to_numpy(float)
        x = display.bin_center_MeV.to_numpy(float)
        observed = display.observed_events.to_numpy(float) / width
        error = np.sqrt(np.clip(display.observed_events.to_numpy(float), 1.0, None)) / width
        ax.axvspan(
            float(prediction.blind[0] * 1000.0),
            float(prediction.blind[1] * 1000.0),
            color=COLORS["window"],
            alpha=0.08,
            label=r"$\pm2.25\sigma_m$ fit window",
        )
        ax.errorbar(
            x,
            observed,
            yerr=error,
            fmt="o",
            ms=2.8,
            color="black",
            elinewidth=0.6,
            label="Observed",
            zorder=5,
        )
        ax.plot(x, display.gp_mean_events / width, color=COLORS["gp"], lw=1.5, label="Frozen GP mean")
        ax.plot(
            x,
            display.joint_profiled_background_events / width,
            color=COLORS["background"],
            lw=1.6,
            ls="--",
            label="Joint-profiled background",
        )
        ax.plot(
            x,
            display.shared_total_events / width,
            color=COLORS["shared"],
            lw=2.0,
            label=r"Shared-$\epsilon^2$ best fit",
        )
        ax.plot(
            x,
            display.independent_signed_total_events / width,
            color=COLORS["independent"],
            lw=1.35,
            ls=":",
            label="Independent signed diagnostic",
        )
        ax.set_title(f"{key} — local display", loc="left", fontweight="semibold")
        ax.set_ylabel("Events / MeV")
        ax.tick_params(labelbottom=False)

        window = data[data.in_fitted_window.astype(bool)].copy()
        win_x = window.bin_center_MeV.to_numpy(float)
        obs_win = window.observed_events.to_numpy(float)
        bfit = window.joint_profiled_background_events.to_numpy(float)
        # conditioned[key] already has only the fitted-window dimensions.
        errors = np.sqrt(
            np.clip(obs_win, 1.0, None)
            + np.clip(np.diag(conditioned[key]), 0.0, None)
        )
        pulls = (obs_win - bfit) / errors
        shared_pull = window.shared_signal_events.to_numpy(float) / errors
        independent_pull = (
            window.independent_signed_signal_events.to_numpy(float) / errors
        )
        pull_ax.axhline(0.0, color="0.3", lw=0.8)
        pull_ax.errorbar(
            win_x,
            pulls,
            yerr=np.ones_like(pulls),
            fmt="o",
            ms=3.0,
            color="black",
            elinewidth=0.6,
        )
        pull_ax.plot(win_x, shared_pull, color=COLORS["shared"], lw=1.8)
        pull_ax.plot(
            win_x,
            independent_pull,
            color=COLORS["independent"],
            lw=1.3,
            ls=":",
        )
        pull_ax.set_ylabel("Marginal residual / $\sigma$")
        pull_ax.set_xlabel("Invariant mass (MeV)")
        pull_ax.set_xlim(
            float(prediction.blind[0] * 1000.0),
            float(prediction.blind[1] * 1000.0),
        )
        if column == 0:
            handles, labels = ax.get_legend_handles_labels()

    if handles is not None:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.885),
            ncol=3,
            fontsize=8.0,
        )
    fig.suptitle(
        (
            f"All-three signal extraction at {mass_mev} MeV — "
            f"local $Z={z_value:.2f}$ (not scan-corrected)"
        ),
        x=0.07,
        y=0.97,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.018,
        (
            "Shared fit uses one common coupling; green signed fits are post-selection "
            "campaign-level concordance diagnostics, not independent measurements. "
            "Lower panels show descriptive marginal residuals, not calibrated pulls."
        ),
        ha="center",
        fontsize=8.0,
        color="0.35",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "all_three_peak_extraction.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES / "all_three_peak_extraction.png", bbox_inches="tight", dpi=220
    )
    plt.close(fig)

    summary = {
        "schema_version": 1,
        "status": "computed",
        "selection": {
            "scope": "all_2015_2016_2021",
            "grid_MeV": [50, 90, 1],
            "rule": "argmin local asymptotic p0",
            "mass_MeV": mass_mev,
            "p0_local_asymptotic": float(p0),
            "Z_local_asymptotic": float(z_value),
            "q0_local_asymptotic": float(q0),
            "look_elsewhere_corrected": False,
        },
        "shared_fit": {
            "eps2_hat": shared_eps2,
            "sigma_eps2": shared_sigma_eps2,
            "A_hat_fitted_window_counts": float(shared_fit["A_hat"]),
            "sigma_A_fitted_window_counts": float(shared_fit["sigma_A"]),
            "fit": fit_status(shared_fit),
            "null": fit_status(shared_null),
            "p0_nll_alt": p0_nll_alt,
            "p0_nll_null": p0_nll_null,
            "p0_nll_nesting_tolerance": p0_nll_tolerance,
        },
        "independent_signed_diagnostics": independent,
        "covariance_conditioning": conditioning,
        "result_closure": closures,
        "inputs": {
            "curves": str(CURVES),
            "curves_sha256": sha256(CURVES),
            "card_sha256": sha256(CARD),
            "states_sha256": sha256(STATES),
            "provenance_sha256": sha256(PROVENANCE),
        },
        "claim_boundary": (
            "Mass selected by the minimum local all-three p0; no global p-value. "
            "Independent signed fits are post-selection concordance diagnostics, "
            "not three measurements."
        ),
    }
    (OUTPUT / "all_three_peak_extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
