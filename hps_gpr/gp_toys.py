"""GP-propagated full-range toy generation and scan orchestration."""

from __future__ import annotations

import copy
import os
from dataclasses import replace
from typing import List, Optional, Sequence, TYPE_CHECKING

import hist
import numpy as np

from .funcform_toys import (
    _augment_scan_table_metadata,
    _sanitize_toy_path_component,
    _write_toy_metadata_payload,
)
from .gpr import fit_gpr, make_kernel_for_dataset, predict_counts_mean_from_log_gpr
from .io import _build_model
from .plotting import ensure_dir
from .toy_backgrounds import (
    draw_full_background_toy,
    normalize_full_toy_bkg_mode,
    observed_total_count,
)

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def _source_model_name(full_toy_bkg_mode: str) -> str:
    """Return a compact provenance label for the GP toy source."""
    mode = normalize_full_toy_bkg_mode(full_toy_bkg_mode)
    suffix = "fixedtotal" if mode == "fixed_total_multinomial" else "poisson"
    return f"gp_propagated_mean_refit_{suffix}"


def _toy_seed(base_seed: int, toy_index: int) -> int:
    """Stable per-toy seed that does not depend on batching order."""
    return int((int(base_seed) + 1000003 * int(toy_index)) % (2**31 - 1))


def _gp_toy_output_dir(
    base_output_dir: str,
    dataset_key: str,
    source_label: str,
    toy_index: int,
) -> str:
    """Return the output directory for one GP-generated toy scan."""
    source_dir = _sanitize_toy_path_component(source_label)
    return os.path.join(
        str(base_output_dir),
        "toy_scans",
        str(dataset_key),
        source_dir,
        f"toy_{int(toy_index):04d}",
    )


def _hist_from_counts(edges: np.ndarray, counts: np.ndarray) -> hist.Hist:
    """Build an in-memory histogram compatible with the scan pipeline."""
    edges = np.asarray(edges, float).reshape(-1)
    counts = np.asarray(counts, float).reshape(-1)
    hout = hist.Hist(
        hist.axis.Variable(edges, label="Mass / GeV"),
        storage=hist.storage.Weight(),
    )
    view = hout.view()
    view.value[...] = counts
    view.variance[...] = np.clip(counts, 0.0, None)
    return hout


def build_gp_propagated_mean(
    ds: "DatasetConfig",
    config: "Config",
    *,
    rebin: Optional[int] = None,
    restarts: Optional[int] = None,
    optimize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a full-range GP and return (edges, observed_counts, propagated_mean)."""
    if rebin is None:
        rebin = int(config.neighborhood_rebin)
    if restarts is None:
        restarts = int(config.n_restarts)

    model = _build_model(
        ds,
        blind=(float(ds.m_low), float(ds.m_low)),
        rebin=int(rebin),
        config=config,
        mass=None,
    )
    x_full = np.asarray(model.histogram.axes[0].centers, float).reshape(-1)
    y_full = np.asarray(model.histogram.values(), float).reshape(-1)
    edges_full = np.asarray(model.histogram.axes[0].edges, float).reshape(-1)

    kernel = make_kernel_for_dataset(ds, config, mass=None)
    gpr = fit_gpr(
        x_full,
        y_full,
        config,
        restarts=int(restarts),
        kernel=kernel,
        optimize=bool(optimize),
    )
    mu_full = predict_counts_mean_from_log_gpr(gpr, x_full, config)
    return edges_full, y_full, np.asarray(mu_full, float).reshape(-1)


def run_gp_toy_scans(
    ds: "DatasetConfig",
    config: "Config",
    *,
    n_toys: int,
    base_output_dir: str,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    seed: Optional[int] = None,
    toy_indices: Optional[Sequence[int]] = None,
    save_plots: Optional[bool] = None,
    save_fit_json: Optional[bool] = None,
    save_per_mass_folders: Optional[bool] = None,
    scan_parallel: Optional[bool] = None,
    scan_n_workers: Optional[int] = None,
    scan_parallel_backend: Optional[str] = None,
    scan_threads_per_worker: Optional[int] = None,
) -> List[str]:
    """Run the mass scan once per GP-generated full-range background toy."""
    from .scan import run_scan

    if toy_indices is None:
        resolved_toy_indices = list(range(int(n_toys)))
    else:
        resolved_toy_indices = sorted({int(idx) for idx in toy_indices})
    if not resolved_toy_indices:
        return []

    base_seed = int(seed if seed is not None else getattr(config, "ul_bands_seed", config.cls_seed_base))
    full_toy_bkg_mode = normalize_full_toy_bkg_mode(
        getattr(config, "full_toy_bkg_mode", "poisson")
    )
    source_model = _source_model_name(full_toy_bkg_mode)

    edges_full, y_full_obs, mu_full = build_gp_propagated_mean(ds, config)
    total_full = observed_total_count(y_full_obs)

    written: List[str] = []
    for toy_index in resolved_toy_indices:
        rng = np.random.default_rng(_toy_seed(base_seed, int(toy_index)))
        toy_counts = draw_full_background_toy(
            mu_full,
            rng,
            mode=full_toy_bkg_mode,
            total_count=total_full,
        )
        toy_hist = _hist_from_counts(edges_full, toy_counts)
        toy_ds = replace(ds, hist_override=toy_hist)

        toy_cfg = copy.deepcopy(config)
        toy_cfg.output_dir = _gp_toy_output_dir(base_output_dir, ds.key, source_model, int(toy_index))
        toy_cfg.neighborhood_rebin = 1
        toy_cfg.save_plots = bool(
            getattr(config, "toy_scan_save_plots", False)
            if save_plots is None else save_plots
        )
        toy_cfg.save_fit_json = bool(
            getattr(config, "toy_scan_save_fit_json", False)
            if save_fit_json is None else save_fit_json
        )
        toy_cfg.save_per_mass_folders = bool(
            getattr(config, "toy_scan_save_per_mass_folders", False)
            if save_per_mass_folders is None else save_per_mass_folders
        )
        toy_cfg.scan_parallel = bool(
            getattr(config, "toy_scan_parallel", False)
            if scan_parallel is None else scan_parallel
        )
        toy_cfg.scan_n_workers = max(
            1,
            int(
                getattr(config, "toy_scan_n_workers", 1)
                if scan_n_workers is None else scan_n_workers
            ),
        )
        toy_cfg.scan_parallel_backend = str(
            getattr(config, "toy_scan_parallel_backend", "threading")
            if scan_parallel_backend is None else scan_parallel_backend
        )
        toy_cfg.scan_threads_per_worker = max(
            1,
            int(
                getattr(config, "toy_scan_threads_per_worker", 1)
                if scan_threads_per_worker is None else scan_threads_per_worker
            ),
        )
        toy_cfg.ensure_output_dir()

        df_single, df_comb = run_scan(
            {str(ds.key): toy_ds},
            toy_cfg,
            mass_min=mass_min,
            mass_max=mass_max,
        )

        toy_name = f"{source_model}_toy_{int(toy_index):04d}"
        df_single = _augment_scan_table_metadata(
            df_single,
            toy_index=int(toy_index),
            toy_name=toy_name,
            dataset=str(ds.key),
            source_model=source_model,
            source_label=source_model,
            source_root="",
            container="generated",
            function_tag=source_model,
        )
        df_comb = _augment_scan_table_metadata(
            df_comb,
            toy_index=int(toy_index),
            toy_name=toy_name,
            dataset=str(ds.key),
            source_model=source_model,
            source_label=source_model,
            source_root="",
            container="generated",
            function_tag=source_model,
        )

        single_path = os.path.join(toy_cfg.output_dir, "results_single.csv")
        comb_path = os.path.join(toy_cfg.output_dir, "results_combined.csv")
        alias_path = os.path.join(toy_cfg.output_dir, "combined.csv")

        df_single.to_csv(single_path, index=False)
        df_comb.to_csv(comb_path, index=False)
        df_comb.to_csv(alias_path, index=False)
        _write_toy_metadata_payload(
            toy_cfg.output_dir,
            {
                "dataset": str(ds.key),
                "toy_index": int(toy_index),
                "toy_name": toy_name,
                "function_tag": source_model,
                "source_model": source_model,
                "source_label": source_model,
                "source_root": "",
                "container": "generated",
                "output_dir": str(toy_cfg.output_dir),
                "toy_seed": int(_toy_seed(base_seed, int(toy_index))),
                "full_toy_bkg_mode": str(full_toy_bkg_mode),
                "full_input_total_count": int(total_full),
                "full_input_n_bins": int(max(0, len(edges_full) - 1)),
            },
        )
        written.append(toy_cfg.output_dir)

    return written
