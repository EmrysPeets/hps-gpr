#!/usr/bin/env python3
"""Exact-binning follow-up to HPS-GPR v4.2 Figures 61 and 62.

This study is intentionally local to the 65 MeV mass hypothesis.  It:

* reconstructs the authoritative native v4.2 fixed-GP extraction;
* refits all three GP backgrounds with exact, count-preserving 0.5 MeV bins;
* repeats the refit with exact 1.25 MeV bins as a coarsening stress test;
* recomputes standalone and simultaneous shared-epsilon-squared profile fits;
* computes physical-domain 68% profile-likelihood intervals;
* writes Figure-61-style, profiled-background, and corrected Figure-62 plots.

The 2015 and 2016 inputs have 0.05 MeV source bins, while the 2021 input has
0.125 MeV source bins.  Therefore 0.625 MeV is not an integer aggregation of
the 2015/2016 histogram bins (0.625 / 0.05 = 12.5).  This script records that
fact and never splits, weights, rounds, or truncates source-bin counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-v4p2-m065-followup")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import gp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq

from hps_gpr.config import load_config
from hps_gpr.conversion import A_from_epsilon2
from hps_gpr.dataset import DatasetConfig, make_datasets
from hps_gpr.evaluation import build_combined_components
from hps_gpr.gpr import (
    _extract_rbf_bounds_and_scale,
    compute_kernel_ls_bounds,
    fit_gpr,
    make_fixed_kernel,
    make_kernel_for_dataset,
    predict_counts_from_log_gpr,
    predict_counts_mean_from_log_gpr,
)
from hps_gpr.io import (
    BlindPrediction,
    _at_kernel_bound,
    _blind_pred_detail,
    _compute_integral_density,
    _extract_constant_bounds_and_value,
)
from hps_gpr.statistics import (
    fit_A_profiled_gaussian_details,
    p0_profiled_gaussian_LRT,
    profile_theta_given_A,
    profiled_gaussian_fixed_poi_nll,
)
from hps_gpr.template import build_window_template_from_full


MASS_GEV = 0.065
MASS_MEV = 65.0
DATASET_ORDER = ("2015", "2016", "2021")
LABELS = {
    "2015": "2015 100%",
    "2016": "2016 100%",
    "2021": "2021 10%",
}
COLORS = {
    "2015": "#0072B2",
    "2016": "#D55E00",
    "2021": "#009E73",
    "gp": "#365F91",
    "standalone": "#E69F00",
    "shared": "#B51F2E",
    "blind": "#8064A2",
    "ink": "#23262D",
    "grid": "#B8BEC8",
    "native": "#4C566A",
    "common": "#B51F2E",
}

CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
REVIEWED_STATES = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
ENRICHED_STATES = (
    REPO
    / "study_results"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "derived"
    / "observed_gp_states_v4p2_enriched.csv"
)
NATIVE_EXTRACTION_DIR = (
    REPO
    / "study_results"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "note_figures"
    / "extractions_m065"
)
NATIVE_FIT_SUMMARY = NATIVE_EXTRACTION_DIR / "observed_extraction_m065_fit_summary.csv"
NATIVE_PLOT_DATA = NATIVE_EXTRACTION_DIR / "observed_extraction_m065_plot_data.csv"
NATIVE_PROVENANCE = NATIVE_EXTRACTION_DIR / "observed_extraction_m065_provenance.json"

REFERENCE_DIR = HERE / "reference_v4p2"
TABLE_DIR = HERE / "tables"
FIGURE_DIR = HERE / "figures"
VALIDATION_PATH = HERE / "validation.json"
PROVENANCE_PATH = HERE / "provenance.json"
CAPTIONS_PATH = HERE / "CAPTIONS.md"
README_PATH = HERE / "README.md"

EXPECTED_CONFIG_SHA256 = "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
EXPECTED_REVIEWED_STATES_SHA256 = (
    "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9"
)

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "native_v4p2": {
        "target_width_GeV": None,
        "factors": {"2015": 5, "2016": 5, "2021": 5},
        "optimize": False,
        "description": "Authoritative v4.2 fixed-GP reference binning.",
    },
    "common_0p5MeV": {
        "target_width_GeV": 0.0005,
        "factors": {"2015": 10, "2016": 10, "2021": 4},
        "optimize": True,
        "description": (
            "Exact common 0.5 MeV bins; the finer of the equidistant "
            "0.5/0.75 MeV source-compatible choices and the nearest one "
            "that retains all three full supports."
        ),
    },
    "common_1p25MeV": {
        "target_width_GeV": 0.00125,
        "factors": {"2015": 25, "2016": 25, "2021": 10},
        "optimize": True,
        "description": "Exact common 1.25 MeV coarsening stress test.",
    },
}

OPTIMIZER_SEEDS = (20260806, 20260807)
PROFILE_DELTA_NLL = 0.5


@dataclass
class ModelBundle:
    variant: str
    ds: DatasetConfig
    pred: BlindPrediction
    gpr: object
    source_histogram: object
    source_bin_width_GeV: float
    rebin_factor: int
    target_bin_width_GeV: float
    template_window: np.ndarray
    template_full: np.ndarray
    fit_signed: Mapping[str, Any]
    fit_bounded: Mapping[str, Any]
    null_profile: Mapping[str, Any]
    p0: float
    Z: float
    q0: float
    k_events_per_eps2: float
    interval68: Mapping[str, Any]
    optimizer_seed: int | None
    actual_support: Tuple[float, float]


@dataclass
class VariantResult:
    name: str
    models: List[ModelBundle]
    joint_obs: np.ndarray
    joint_bkg: np.ndarray
    joint_cov: np.ndarray
    joint_template: np.ndarray
    fit_signed: Mapping[str, Any]
    fit_bounded: Mapping[str, Any]
    null_profile: Mapping[str, Any]
    p0: float
    Z: float
    q0: float
    eps2_hat_signed: float
    sigma_eps2_wald: float
    eps2_hat_bounded: float
    interval68: Mapping[str, Any]
    shared_bkg_by_dataset: Dict[str, np.ndarray]
    shared_lambda_by_dataset: Dict[str, np.ndarray]
    q0_contribution_by_dataset: Dict[str, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True, stderr=subprocess.STDOUT
    ).strip()


def json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_scalar) + "\n"
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "legend.fontsize": 8.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> List[Dict[str, Any]]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for suffix, dpi in ((".pdf", None), (".png", 300)):
        path = FIGURE_DIR / f"{stem}{suffix}"
        kwargs: MutableMapping[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        records.append(
            {
                "path": repo_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    plt.close(fig)
    return records


def state_row(table: pd.DataFrame, dataset: str) -> pd.Series:
    rows = table[
        (table["dataset"].astype(str) == dataset)
        & np.isclose(table["mass_GeV"].to_numpy(float), MASS_GEV, atol=5e-10)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one {dataset} state at {MASS_MEV:.0f} MeV; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def source_histogram(ds: DatasetConfig):
    return gp._hist.io._deduce_histogram((ds.root_path, ds.hist_name))


def source_metadata(ds: DatasetConfig, hist) -> Dict[str, Any]:
    edges = np.asarray(hist.axes[0].edges, float)
    values = np.asarray(hist.values(), float)
    widths = np.diff(edges)
    classnames: Mapping[str, str]
    with uproot.open(ds.root_path) as root_file:
        classnames = root_file.classnames(recursive=True)
    trees = [
        {"key": key, "class": cls}
        for key, cls in classnames.items()
        if "TTree" in cls or "RNTuple" in cls
    ]
    return {
        "dataset": ds.key,
        "root_path": str(Path(ds.root_path).resolve()),
        "root_sha256": sha256(Path(ds.root_path)),
        "histogram": ds.hist_name,
        "source_n_bins": int(values.size),
        "source_lo_GeV": float(edges[0]),
        "source_hi_GeV": float(edges[-1]),
        "source_bin_width_GeV": float(np.median(widths)),
        "source_uniform_width": bool(
            np.allclose(widths, np.median(widths), rtol=0.0, atol=2e-14)
        ),
        "source_values_integer": bool(
            np.allclose(values, np.rint(values), rtol=0.0, atol=1e-9)
        ),
        "source_sum_counts": float(values.sum()),
        "event_level_objects": trees,
    }


def build_histogram_model(
    ds: DatasetConfig,
    *,
    rebin_factor: int,
    source_hist=None,
) -> Tuple[SimpleNamespace, Dict[str, Any]]:
    source = source_hist if source_hist is not None else source_histogram(ds)
    edges_source = np.asarray(source.axes[0].edges, float)
    values_source = np.asarray(source.values(), float)
    source_width = float(np.median(np.diff(edges_source)))
    lower = max(
        float(edges_source[0]),
        float(ds.data_low) if ds.data_low is not None else float(edges_source[0]),
    )
    upper = min(
        float(edges_source[-1]),
        float(ds.data_high) if ds.data_high is not None else float(edges_source[-1]),
    )
    histogram = gp._hist.manipulation.rebin_and_limit(
        int(rebin_factor), lower, upper
    )(source)
    edges = np.asarray(histogram.axes[0].edges, float)
    values = np.asarray(histogram.values(), float)
    widths = np.diff(edges)
    expected_width = source_width * int(rebin_factor)
    if not np.allclose(widths, expected_width, rtol=0.0, atol=2e-13):
        raise RuntimeError(
            f"{ds.key}: nonuniform or unexpected model width for factor "
            f"{rebin_factor}: {np.unique(widths)}"
        )
    if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-9):
        raise RuntimeError(f"{ds.key}: rebinned histogram contains fractional counts")
    metadata = {
        "dataset": ds.key,
        "rebin_factor": int(rebin_factor),
        "source_bin_width_GeV": source_width,
        "expected_bin_width_GeV": expected_width,
        "actual_bin_width_GeV": float(np.median(widths)),
        "actual_support_lo_GeV": float(edges[0]),
        "actual_support_hi_GeV": float(edges[-1]),
        "actual_n_bins": int(values.size),
        "actual_sum_counts": float(values.sum()),
        "values_integer": True,
        "source_sum_counts": float(values_source.sum()),
    }
    return SimpleNamespace(histogram=histogram, density_histogram=source), metadata


def kernel_values(gpr) -> Tuple[float, float]:
    kernel = getattr(gpr, "kernel_", getattr(gpr, "kernel", None))
    const = float(getattr(getattr(kernel, "k1", None), "constant_value", np.nan))
    _, _, ls = _extract_rbf_bounds_and_scale(kernel)
    return const, float(ls)


def make_prediction(
    ds: DatasetConfig,
    config,
    *,
    rebin_factor: int,
    optimize: bool,
    optimizer_seed: int | None,
    fixed_state: pd.Series | None,
    source_hist=None,
) -> Tuple[BlindPrediction, object, object, Dict[str, Any]]:
    sigma_val = float(ds.sigma(MASS_GEV))
    blind = (
        MASS_GEV - float(config.blind_nsigma) * sigma_val,
        MASS_GEV + float(config.blind_nsigma) * sigma_val,
    )
    train_nsigma = float(
        getattr(config, "gp_train_exclude_nsigma", None) or config.blind_nsigma
    )
    blind_train = (
        MASS_GEV - train_nsigma * sigma_val,
        MASS_GEV + train_nsigma * sigma_val,
    )
    model, hist_meta = build_histogram_model(
        ds, rebin_factor=rebin_factor, source_hist=source_hist
    )
    x = np.asarray(model.histogram.axes[0].centers, float)
    y = np.asarray(model.histogram.values(), float)
    train_mask = (x < blind_train[0]) | (x > blind_train[1])
    if optimize:
        if optimizer_seed is None:
            raise ValueError("Optimized fits require an explicit seed")
        np.random.seed(int(optimizer_seed))
        kernel = make_kernel_for_dataset(ds, config, mass=MASS_GEV)
        gpr = fit_gpr(
            x[train_mask],
            y[train_mask],
            config,
            restarts=int(config.n_restarts),
            kernel=kernel,
            optimize=True,
        )
    else:
        if fixed_state is None:
            raise ValueError("Fixed native reconstruction requires a reviewed state")
        kernel = make_fixed_kernel(
            float(fixed_state["const_opt"]), float(fixed_state["ls_opt"])
        )
        gpr = fit_gpr(
            x[train_mask],
            y[train_mask],
            config,
            restarts=0,
            kernel=kernel,
            optimize=False,
        )

    mu, cov, obs, edges = _blind_pred_detail(model, gpr, blind, config)
    mu_full = predict_counts_mean_from_log_gpr(gpr, x, config)
    density_nsigma = float(
        getattr(config, "eps2_density_nsigma", None) or config.blind_nsigma
    )
    density, density_meta = _compute_integral_density(
        model,
        MASS_GEV,
        sigma_val,
        density_nsigma=density_nsigma,
        return_metadata=True,
    )
    full_edges = np.asarray(model.histogram.axes[0].edges, float)
    full_widths = np.diff(full_edges)
    blind_mask = np.asarray((x >= blind[0]) & (x <= blind[1]), bool)
    const_opt, ls_opt = kernel_values(gpr)
    ls_info = compute_kernel_ls_bounds(ds, config, mass=MASS_GEV)
    initial_kernel = make_kernel_for_dataset(ds, config, mass=MASS_GEV)
    const_lo, const_hi, const_init = _extract_constant_bounds_and_value(
        initial_kernel
    )
    pred = BlindPrediction(
        mu=np.asarray(mu, float),
        cov=np.asarray(cov, float),
        obs=np.asarray(obs, int),
        edges=np.asarray(edges, float),
        sigma_val=sigma_val,
        blind=blind,
        x_full=x,
        y_full=y,
        mu_full=np.asarray(mu_full, float),
        edges_full=full_edges,
        blind_mask=blind_mask,
        integral_density=float(density),
        **density_meta,
        blind_train=blind_train,
        kernel_str=str(getattr(gpr, "kernel_", kernel)),
        ls_lo=float(ls_info["ls_lo"]),
        ls_hi=float(ls_info["ls_hi"]),
        ls_init=float(ls_info["ls_init"]),
        ls_opt=ls_opt,
        sigma_x=float(ls_info["sigma_x"]),
        const_opt=const_opt,
        lml=float(gpr.log_marginal_likelihood_value_),
        n_train=int(np.count_nonzero(train_mask)),
        n_train_low=int(np.count_nonzero(x < blind_train[0])),
        n_train_high=int(np.count_nonzero(x > blind_train[1])),
        n_full=int(x.size),
        n_blind=int(np.count_nonzero(blind_mask)),
        train_domain_lo=float(full_edges[0]),
        train_domain_hi=float(full_edges[-1]),
        bin_width_median=float(np.median(full_widths)),
        const_init=const_init,
        const_lo=const_lo,
        const_hi=const_hi,
        const_at_lower=_at_kernel_bound(const_opt, const_lo),
        const_at_upper=_at_kernel_bound(const_opt, const_hi),
        ls_at_lower=_at_kernel_bound(ls_opt, float(ls_info["ls_lo"])),
        ls_at_upper=_at_kernel_bound(ls_opt, float(ls_info["ls_hi"])),
        optimizer_restarts=int(config.n_restarts) if optimize else 0,
    )
    return pred, gpr, model.density_histogram, hist_meta


def physical_profile_interval(
    obs: np.ndarray,
    bkg: np.ndarray,
    cov: np.ndarray,
    template: np.ndarray,
    *,
    fit_bounded: Mapping[str, Any],
    sigma_hint: float,
    poi_to_eps2: float,
) -> Dict[str, Any]:
    """Return the physical A>=0 profile-likelihood interval with DeltaNLL=0.5."""

    poi_hat = max(0.0, float(fit_bounded["A_hat"]))
    evaluations: List[Dict[str, Any]] = []

    def evaluate(poi: float) -> float:
        out = profiled_gaussian_fixed_poi_nll(
            obs,
            bkg,
            cov,
            template,
            A_fixed=float(poi),
        )
        nll = float(out["nll"])
        evaluations.append(
            {
                "poi": float(poi),
                "nll": nll,
                "success": bool(out.get("success", False)),
            }
        )
        if not np.isfinite(nll):
            raise RuntimeError(f"Non-finite profiled NLL at POI={poi}")
        return nll

    nll_min = evaluate(poi_hat)

    def crossing(poi: float) -> float:
        return evaluate(poi) - nll_min - PROFILE_DELTA_NLL

    f_zero = crossing(0.0)
    if poi_hat <= 0.0 or f_zero <= 0.0:
        lower = 0.0
        lower_at_boundary = True
    else:
        lower = float(brentq(crossing, 0.0, poi_hat, xtol=1e-8, rtol=1e-10))
        lower_at_boundary = False

    step = max(abs(float(sigma_hint)), 0.25 * max(1.0, poi_hat), 1.0)
    upper_bracket = max(poi_hat + 1.5 * step, step)
    f_upper = crossing(upper_bracket)
    n_expand = 0
    while f_upper <= 0.0 and n_expand < 30:
        upper_bracket = max(2.0 * upper_bracket, upper_bracket + step)
        f_upper = crossing(upper_bracket)
        n_expand += 1
    if f_upper <= 0.0:
        raise RuntimeError("Could not bracket physical profile-likelihood upper root")
    upper = float(
        brentq(crossing, poi_hat, upper_bracket, xtol=1e-8, rtol=1e-10)
    )
    return {
        "poi_hat_bounded": poi_hat,
        "poi_low68": lower,
        "poi_high68": upper,
        "eps2_hat_bounded": poi_hat * poi_to_eps2,
        "eps2_low68": lower * poi_to_eps2,
        "eps2_high68": upper * poi_to_eps2,
        "lower_at_physical_boundary": lower_at_boundary,
        "delta_nll": PROFILE_DELTA_NLL,
        "profile_evaluations": int(len(evaluations)),
        "all_profile_calls_finite": bool(
            all(np.isfinite(row["nll"]) for row in evaluations)
        ),
        "all_profile_calls_success": bool(
            all(bool(row["success"]) for row in evaluations)
        ),
    }


def complete_bundle(
    variant: str,
    ds: DatasetConfig,
    pred: BlindPrediction,
    gpr,
    source_hist,
    hist_meta: Mapping[str, Any],
    *,
    optimizer_seed: int | None,
) -> ModelBundle:
    template_window, template_full = build_window_template_from_full(
        pred.edges_full,
        pred.blind_mask,
        MASS_GEV,
        pred.sigma_val,
        config=load_config(str(CONFIG)),
    )
    fit_signed = fit_A_profiled_gaussian_details(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
        allow_negative=True,
    )
    fit_bounded = fit_A_profiled_gaussian_details(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
        allow_negative=False,
    )
    null_profile = profile_theta_given_A(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
        A_fixed=0.0,
    )
    p0, z, q0, pinfo = p0_profiled_gaussian_LRT(
        pred.obs, pred.mu, pred.cov, template_window
    )
    if not bool(fit_signed.get("success", False)):
        raise RuntimeError(f"{variant}/{ds.key}: signed extraction did not converge")
    if not bool(fit_bounded.get("success", False)):
        raise RuntimeError(f"{variant}/{ds.key}: bounded extraction did not converge")
    if not bool(null_profile.get("success", False)) or not bool(pinfo.get("ok", False)):
        raise RuntimeError(f"{variant}/{ds.key}: p0 profiles did not converge")
    k = float(A_from_epsilon2(ds, MASS_GEV, 1.0, pred.integral_density))
    interval = physical_profile_interval(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
        fit_bounded=fit_bounded,
        sigma_hint=float(fit_signed["sigma_A"]),
        poi_to_eps2=1.0 / k,
    )
    return ModelBundle(
        variant=variant,
        ds=ds,
        pred=pred,
        gpr=gpr,
        source_histogram=source_hist,
        source_bin_width_GeV=float(hist_meta["source_bin_width_GeV"]),
        rebin_factor=int(hist_meta["rebin_factor"]),
        target_bin_width_GeV=float(hist_meta["actual_bin_width_GeV"]),
        template_window=np.asarray(template_window, float),
        template_full=np.asarray(template_full, float),
        fit_signed=fit_signed,
        fit_bounded=fit_bounded,
        null_profile=null_profile,
        p0=float(p0),
        Z=float(z),
        q0=float(q0),
        k_events_per_eps2=k,
        interval68=interval,
        optimizer_seed=optimizer_seed,
        actual_support=(
            float(hist_meta["actual_support_lo_GeV"]),
            float(hist_meta["actual_support_hi_GeV"]),
        ),
    )


def split_vector(
    vector: np.ndarray, models: Sequence[ModelBundle]
) -> Dict[str, np.ndarray]:
    arr = np.asarray(vector, float)
    result: Dict[str, np.ndarray] = {}
    offset = 0
    for model in models:
        size = int(model.pred.obs.size)
        result[model.ds.key] = np.asarray(arr[offset : offset + size], float)
        offset += size
    if offset != arr.size:
        raise RuntimeError(f"Split consumed {offset} of {arr.size} elements")
    return result


def joint_result(name: str, models: List[ModelBundle], config) -> VariantResult:
    obs, bkg, cov, s_unit = build_combined_components(
        MASS_GEV,
        [model.ds for model in models],
        [model.pred for model in models],
        config=config,
    )
    scale = float(config.eps2_lrt_scale)
    template = np.asarray(s_unit, float) / scale
    fit_signed = fit_A_profiled_gaussian_details(
        obs, bkg, cov, template, allow_negative=True
    )
    fit_bounded = fit_A_profiled_gaussian_details(
        obs, bkg, cov, template, allow_negative=False
    )
    null = profile_theta_given_A(obs, bkg, cov, template, A_fixed=0.0)
    p0, z, q0, pinfo = p0_profiled_gaussian_LRT(obs, bkg, cov, template)
    if not bool(fit_signed.get("success", False)):
        raise RuntimeError(f"{name}/combined: signed extraction did not converge")
    if not bool(fit_bounded.get("success", False)):
        raise RuntimeError(f"{name}/combined: bounded extraction did not converge")
    if not bool(null.get("success", False)) or not bool(pinfo.get("ok", False)):
        raise RuntimeError(f"{name}/combined: p0 profiles did not converge")
    interval = physical_profile_interval(
        obs,
        bkg,
        cov,
        template,
        fit_bounded=fit_bounded,
        sigma_hint=float(fit_signed["sigma_A"]),
        poi_to_eps2=1.0 / scale,
    )

    alt_lambda = np.asarray(fit_signed["lambda_hat"], float)
    alt_theta = np.asarray(fit_signed["theta_hat"], float)
    null_lambda = np.asarray(null["lambda_hat"], float)
    null_theta = np.asarray(null["theta_hat"], float)
    obs_float = np.asarray(obs, float)
    contributions: Dict[str, float] = {}
    offset = 0
    for model in models:
        size = int(model.pred.obs.size)
        slc = slice(offset, offset + size)

        def block_nll(expectation: np.ndarray, theta: np.ndarray) -> float:
            lam = np.clip(expectation[slc], 1e-12, None)
            nuisance = theta[slc]
            return float(
                np.sum(lam - obs_float[slc] * np.log(lam))
                + 0.5 * np.dot(nuisance, nuisance)
            )

        contributions[model.ds.key] = 2.0 * (
            block_nll(null_lambda, null_theta)
            - block_nll(alt_lambda, alt_theta)
        )
        offset += size
    if not np.isclose(sum(contributions.values()), q0, rtol=2e-6, atol=1e-4):
        raise RuntimeError(
            f"{name}: q0 block sum mismatch {sum(contributions.values())} vs {q0}"
        )
    return VariantResult(
        name=name,
        models=models,
        joint_obs=np.asarray(obs, int),
        joint_bkg=np.asarray(bkg, float),
        joint_cov=np.asarray(cov, float),
        joint_template=template,
        fit_signed=fit_signed,
        fit_bounded=fit_bounded,
        null_profile=null,
        p0=float(p0),
        Z=float(z),
        q0=float(q0),
        eps2_hat_signed=float(fit_signed["A_hat"]) / scale,
        sigma_eps2_wald=float(fit_signed["sigma_A"]) / scale,
        eps2_hat_bounded=float(fit_bounded["A_hat"]) / scale,
        interval68=interval,
        shared_bkg_by_dataset=split_vector(
            np.asarray(fit_signed["b_fit"], float), models
        ),
        shared_lambda_by_dataset=split_vector(
            np.asarray(fit_signed["lambda_hat"], float), models
        ),
        q0_contribution_by_dataset=contributions,
    )


def build_variant(
    name: str,
    config,
    datasets: Mapping[str, DatasetConfig],
    states: pd.DataFrame,
    source_hists: Mapping[str, Any],
    optimizer_rows: List[Dict[str, Any]],
) -> VariantResult:
    spec = VARIANTS[name]
    models: List[ModelBundle] = []
    for dataset_index, key in enumerate(DATASET_ORDER):
        ds = datasets[key]
        factor = int(spec["factors"][key])
        if not bool(spec["optimize"]):
            pred, gpr, density_hist, hist_meta = make_prediction(
                ds,
                config,
                rebin_factor=factor,
                optimize=False,
                optimizer_seed=None,
                fixed_state=state_row(states, key),
                source_hist=source_hists[key],
            )
            models.append(
                complete_bundle(
                    name,
                    ds,
                    pred,
                    gpr,
                    density_hist,
                    hist_meta,
                    optimizer_seed=None,
                )
            )
            continue

        candidates: List[Tuple[BlindPrediction, object, object, Dict[str, Any], int]] = []
        for seed_base in OPTIMIZER_SEEDS:
            seed = int(seed_base + 101 * dataset_index)
            pred, gpr, density_hist, hist_meta = make_prediction(
                ds,
                config,
                rebin_factor=factor,
                optimize=True,
                optimizer_seed=seed,
                fixed_state=None,
                source_hist=source_hists[key],
            )
            optimizer_rows.append(
                {
                    "variant": name,
                    "dataset": key,
                    "seed": seed,
                    "n_restarts": int(config.n_restarts),
                    "lml": pred.lml,
                    "const_opt": pred.const_opt,
                    "ls_opt": pred.ls_opt,
                    "ls_lo": pred.ls_lo,
                    "ls_hi": pred.ls_hi,
                    "const_at_lower": pred.const_at_lower,
                    "const_at_upper": pred.const_at_upper,
                    "ls_at_lower": pred.ls_at_lower,
                    "ls_at_upper": pred.ls_at_upper,
                    "bin_width_GeV": pred.bin_width_median,
                    "n_train": pred.n_train,
                    "train_domain_lo_GeV": pred.train_domain_lo,
                    "train_domain_hi_GeV": pred.train_domain_hi,
                    "selected": False,
                }
            )
            candidates.append((pred, gpr, density_hist, hist_meta, seed))
        selected_index = int(
            np.argmax([candidate[0].lml for candidate in candidates])
        )
        pred, gpr, density_hist, hist_meta, selected_seed = candidates[selected_index]
        for row in optimizer_rows:
            if (
                row["variant"] == name
                and row["dataset"] == key
                and int(row["seed"]) == int(selected_seed)
            ):
                row["selected"] = True
        models.append(
            complete_bundle(
                name,
                ds,
                pred,
                gpr,
                density_hist,
                hist_meta,
                optimizer_seed=selected_seed,
            )
        )
    return joint_result(name, models, config)


def authoritative_rows(native_fit: pd.DataFrame) -> Dict[str, pd.Series]:
    rows: Dict[str, pd.Series] = {}
    for key in (*DATASET_ORDER, "combined"):
        match = native_fit[native_fit["dataset"].astype(str) == key]
        if len(match) != 1:
            raise RuntimeError(f"Native fit summary has {len(match)} rows for {key}")
        rows[key] = match.iloc[0]
    return rows


def extraction_rows(
    results: Mapping[str, VariantResult],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    native = results["native_v4p2"]
    records: List[Dict[str, Any]] = []
    interval_records: List[Dict[str, Any]] = []
    for variant, result in results.items():
        native_by_key = {
            model.ds.key: model for model in native.models
        }
        for model in result.models:
            key = model.ds.key
            eps_hat = float(model.fit_signed["A_hat"]) / model.k_events_per_eps2
            eps_sigma = (
                float(model.fit_signed["sigma_A"]) / model.k_events_per_eps2
            )
            native_model = native_by_key[key]
            native_eps = (
                float(native_model.fit_signed["A_hat"])
                / native_model.k_events_per_eps2
            )
            records.append(
                {
                    "variant": variant,
                    "scope": key,
                    "sample_label": LABELS[key],
                    "mass_MeV": MASS_MEV,
                    "bin_width_MeV": 1e3 * model.pred.bin_width_median,
                    "source_bin_width_MeV": 1e3 * model.source_bin_width_GeV,
                    "rebin_factor": model.rebin_factor,
                    "support_lo_MeV": 1e3 * model.actual_support[0],
                    "support_hi_MeV": 1e3 * model.actual_support[1],
                    "n_full": model.pred.n_full,
                    "n_blind": model.pred.n_blind,
                    "n_train": model.pred.n_train,
                    "blind_lo_MeV": 1e3 * model.pred.blind[0],
                    "blind_hi_MeV": 1e3 * model.pred.blind[1],
                    "template_fraction_in_window": float(
                        np.sum(model.template_window)
                    ),
                    "integral_density_counts_per_GeV": model.pred.integral_density,
                    "K_events_per_eps2": model.k_events_per_eps2,
                    "Ahat_signed_events": float(model.fit_signed["A_hat"]),
                    "sigmaA_wald_events": float(model.fit_signed["sigma_A"]),
                    "eps2_hat_signed": eps_hat,
                    "eps2_sigma_wald": eps_sigma,
                    "eps2_wald_low": eps_hat - eps_sigma,
                    "eps2_wald_high": eps_hat + eps_sigma,
                    "eps2_hat_bounded": model.interval68["eps2_hat_bounded"],
                    "eps2_profile68_low": model.interval68["eps2_low68"],
                    "eps2_profile68_high": model.interval68["eps2_high68"],
                    "profile68_lower_at_boundary": model.interval68[
                        "lower_at_physical_boundary"
                    ],
                    "p0_asymptotic_local": model.p0,
                    "Z_asymptotic_local": model.Z,
                    "q0_asymptotic_local": model.q0,
                    "delta_eps2_hat_vs_native": eps_hat - native_eps,
                    "delta_Z_vs_native": model.Z - native_model.Z,
                    "delta_p0_vs_native": model.p0 - native_model.p0,
                    "const_opt": model.pred.const_opt,
                    "ls_opt": model.pred.ls_opt,
                    "ls_lo": model.pred.ls_lo,
                    "ls_hi": model.pred.ls_hi,
                    "ls_at_lower": model.pred.ls_at_lower,
                    "ls_at_upper": model.pred.ls_at_upper,
                    "lml": model.pred.lml,
                    "optimizer_seed": model.optimizer_seed,
                    "fit_signed_success": bool(model.fit_signed["success"]),
                    "fit_bounded_success": bool(model.fit_bounded["success"]),
                    "profile_calls_success": bool(
                        model.interval68["all_profile_calls_success"]
                    ),
                }
            )
            interval_records.append(
                {
                    "variant": variant,
                    "scope": key,
                    "eps2_hat_bounded": model.interval68["eps2_hat_bounded"],
                    "eps2_low68": model.interval68["eps2_low68"],
                    "eps2_high68": model.interval68["eps2_high68"],
                    "lower_at_physical_boundary": model.interval68[
                        "lower_at_physical_boundary"
                    ],
                    "definition": "physical eps2>=0 profile set with DeltaNLL<=0.5",
                }
            )

        native_eps = native.eps2_hat_signed
        records.append(
            {
                "variant": variant,
                "scope": "combined",
                "sample_label": "2015 + 2016 + 2021",
                "mass_MeV": MASS_MEV,
                "bin_width_MeV": (
                    float(result.models[0].pred.bin_width_median) * 1e3
                    if all(
                        np.isclose(
                            model.pred.bin_width_median,
                            result.models[0].pred.bin_width_median,
                            rtol=0.0,
                            atol=2e-13,
                        )
                        for model in result.models
                    )
                    else np.nan
                ),
                "eps2_hat_signed": result.eps2_hat_signed,
                "eps2_sigma_wald": result.sigma_eps2_wald,
                "eps2_wald_low": (
                    result.eps2_hat_signed - result.sigma_eps2_wald
                ),
                "eps2_wald_high": (
                    result.eps2_hat_signed + result.sigma_eps2_wald
                ),
                "eps2_hat_bounded": result.interval68["eps2_hat_bounded"],
                "eps2_profile68_low": result.interval68["eps2_low68"],
                "eps2_profile68_high": result.interval68["eps2_high68"],
                "profile68_lower_at_boundary": result.interval68[
                    "lower_at_physical_boundary"
                ],
                "p0_asymptotic_local": result.p0,
                "Z_asymptotic_local": result.Z,
                "q0_asymptotic_local": result.q0,
                "delta_eps2_hat_vs_native": result.eps2_hat_signed - native_eps,
                "delta_Z_vs_native": result.Z - native.Z,
                "delta_p0_vs_native": result.p0 - native.p0,
                "fit_signed_success": bool(result.fit_signed["success"]),
                "fit_bounded_success": bool(result.fit_bounded["success"]),
                "profile_calls_success": bool(
                    result.interval68["all_profile_calls_success"]
                ),
                "q0_block_2015": result.q0_contribution_by_dataset["2015"],
                "q0_block_2016": result.q0_contribution_by_dataset["2016"],
                "q0_block_2021": result.q0_contribution_by_dataset["2021"],
            }
        )
        interval_records.append(
            {
                "variant": variant,
                "scope": "combined",
                "eps2_hat_bounded": result.interval68["eps2_hat_bounded"],
                "eps2_low68": result.interval68["eps2_low68"],
                "eps2_high68": result.interval68["eps2_high68"],
                "lower_at_physical_boundary": result.interval68[
                    "lower_at_physical_boundary"
                ],
                "definition": "physical eps2>=0 profile set with DeltaNLL<=0.5",
            }
        )
    return pd.DataFrame(records), pd.DataFrame(interval_records)


def bin_level_rows(result: VariantResult) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model in result.models:
        key = model.ds.key
        shared_bkg = result.shared_bkg_by_dataset[key]
        shared_lambda = result.shared_lambda_by_dataset[key]
        blind_indices = np.flatnonzero(model.pred.blind_mask)
        lookup = {int(full): int(i) for i, full in enumerate(blind_indices)}
        for full_index, (x, y, mu) in enumerate(
            zip(model.pred.x_full, model.pred.y_full, model.pred.mu_full)
        ):
            blind_index = lookup.get(int(full_index))
            record: Dict[str, Any] = {
                "variant": result.name,
                "dataset": key,
                "bin_center_MeV": 1e3 * float(x),
                "observed": float(y),
                "gp_mean_preprofile": float(mu),
                "in_extraction_window": blind_index is not None,
                "null_profiled_background": np.nan,
                "standalone_profiled_background": np.nan,
                "standalone_profiled_total": np.nan,
                "shared_profiled_background": np.nan,
                "shared_profiled_total": np.nan,
            }
            if blind_index is not None:
                record.update(
                    {
                        "null_profiled_background": float(
                            model.null_profile["lambda_hat"][blind_index]
                        ),
                        "standalone_profiled_background": float(
                            model.fit_signed["b_fit"][blind_index]
                        ),
                        "standalone_profiled_total": float(
                            model.fit_signed["lambda_hat"][blind_index]
                        ),
                        "shared_profiled_background": float(
                            shared_bkg[blind_index]
                        ),
                        "shared_profiled_total": float(
                            shared_lambda[blind_index]
                        ),
                    }
                )
            records.append(record)
    return pd.DataFrame(records)


def display_mask(model: ModelBundle, half_width_sigma: float = 5.0) -> np.ndarray:
    x = np.asarray(model.pred.x_full, float)
    half_width = half_width_sigma * float(model.pred.sigma_val)
    return (x >= MASS_GEV - half_width) & (x <= MASS_GEV + half_width)


def draw_blind_boundaries(ax: plt.Axes, model: ModelBundle) -> None:
    for bound in model.pred.blind:
        ax.axvline(
            1e3 * float(bound),
            color=COLORS["blind"],
            linewidth=1.15,
            linestyle=(0, (4, 3)),
            zorder=2,
        )


def plot_common_figure61(result: VariantResult, config) -> List[Dict[str, Any]]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.6, 7.8),
        sharex="col",
        gridspec_kw={"height_ratios": (1.55, 1.0), "hspace": 0.08, "wspace": 0.18},
    )
    for col, model in enumerate(result.models):
        mask = display_mask(model)
        x = 1e3 * np.asarray(model.pred.x_full[mask], float)
        y = np.asarray(model.pred.y_full[mask], float)
        mu, cov = predict_counts_from_log_gpr(
            model.gpr, model.pred.x_full[mask], config
        )
        gp_pred_sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        total_sigma = np.sqrt(
            np.clip(mu, 1.0, None) + np.clip(np.diag(cov), 0.0, None)
        )
        standalone_signal = (
            float(model.fit_signed["A_hat"]) * model.template_full[mask]
        )
        shared_signal = (
            result.eps2_hat_signed
            * model.k_events_per_eps2
            * model.template_full[mask]
        )
        ax = axes[0, col]
        ax_res = axes[1, col]
        ax.fill_between(
            x,
            np.clip(mu - gp_pred_sigma, 0.0, None),
            mu + gp_pred_sigma,
            color=COLORS["gp"],
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(x, mu, color=COLORS["gp"], linewidth=1.8, zorder=3)
        ax.plot(
            x,
            mu + standalone_signal,
            color=COLORS["standalone"],
            linewidth=1.8,
            linestyle=(0, (5, 2.5)),
            zorder=4,
        )
        ax.plot(
            x,
            mu + shared_signal,
            color=COLORS["shared"],
            linewidth=2.1,
            zorder=5,
        )
        ax.errorbar(
            x,
            y,
            yerr=np.sqrt(np.clip(y, 1.0, None)),
            fmt="o",
            color=COLORS["ink"],
            markersize=2.8,
            elinewidth=0.65,
            capsize=0.0,
            zorder=6,
        )
        draw_blind_boundaries(ax, model)
        ax.set_title(
            f"{LABELS[model.ds.key]}  (0.5 MeV/bin)",
            color=COLORS[model.ds.key],
            fontweight="semibold",
            pad=8,
        )
        ax.set_ylabel("Events / bin" if col == 0 else "")
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(labelbottom=False)

        residual = y - mu
        ax_res.axhline(0.0, color="#555B65", linewidth=0.9, zorder=1)
        ax_res.errorbar(
            x,
            residual,
            yerr=total_sigma,
            fmt="o",
            color=COLORS["ink"],
            markersize=2.8,
            elinewidth=0.7,
            capsize=0.0,
            zorder=4,
        )
        ax_res.plot(
            x,
            standalone_signal,
            color=COLORS["standalone"],
            linewidth=1.8,
            linestyle=(0, (5, 2.5)),
            zorder=5,
        )
        ax_res.plot(
            x,
            shared_signal,
            color=COLORS["shared"],
            linewidth=2.1,
            zorder=6,
        )
        draw_blind_boundaries(ax_res, model)
        ax_res.set_xlabel(r"$m_{e^+e^-}$ (MeV)")
        ax_res.set_ylabel("Data - GP bkg." if col == 0 else "")
        ax_res.yaxis.set_major_locator(MaxNLocator(6))
        ax_res.xaxis.set_major_locator(MaxNLocator(7))
        ax_res.set_xlim(float(x[0]), float(x[-1]))

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=COLORS["ink"],
            markersize=4.0, label="Observed data"
        ),
        Line2D(
            [0], [0], color=COLORS["gp"], linewidth=1.8,
            label="Refit GP background mean"
        ),
        Patch(
            facecolor=COLORS["gp"], alpha=0.16, edgecolor="none",
            label="GP predictive uncertainty"
        ),
        Line2D(
            [0], [0], color=COLORS["standalone"], linewidth=1.8,
            linestyle=(0, (5, 2.5)), label="Standalone best-fit signal"
        ),
        Line2D(
            [0], [0], color=COLORS["shared"], linewidth=2.1,
            label=r"Shared-$\epsilon^2$ best-fit signal"
        ),
        Line2D(
            [0], [0], color=COLORS["blind"], linewidth=1.15,
            linestyle=(0, (4, 3)),
            label=r"Extraction boundaries ($\pm2.25\sigma_m$)"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Observed 65 MeV extraction — exact common 0.5 MeV bins",
        y=0.985,
        fontsize=15.0,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.745)
    return save_figure(fig, "figure61_common_0p5MeV")


def plot_profiled_figure61(result: VariantResult) -> List[Dict[str, Any]]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.6, 7.8),
        sharex="col",
        gridspec_kw={"height_ratios": (1.55, 1.0), "hspace": 0.08, "wspace": 0.18},
    )
    for col, model in enumerate(result.models):
        key = model.ds.key
        x = 1e3 * np.asarray(model.pred.x_full[model.pred.blind_mask], float)
        obs = np.asarray(model.pred.obs, float)
        null_total = np.asarray(model.null_profile["lambda_hat"], float)
        standalone_total = np.asarray(model.fit_signed["lambda_hat"], float)
        shared_total = np.asarray(result.shared_lambda_by_dataset[key], float)
        ax = axes[0, col]
        ax_res = axes[1, col]
        ax.errorbar(
            x,
            obs,
            yerr=np.sqrt(np.clip(obs, 1.0, None)),
            fmt="o",
            color=COLORS["ink"],
            markersize=3.2,
            elinewidth=0.65,
            capsize=0.0,
            zorder=6,
        )
        ax.plot(
            x,
            null_total,
            color=COLORS["gp"],
            linewidth=1.9,
            label="Background-only profile",
            zorder=3,
        )
        ax.plot(
            x,
            standalone_total,
            color=COLORS["standalone"],
            linewidth=1.9,
            linestyle=(0, (5, 2.5)),
            label="Standalone S+B profile",
            zorder=4,
        )
        ax.plot(
            x,
            shared_total,
            color=COLORS["shared"],
            linewidth=2.1,
            label=r"Shared-$\epsilon^2$ S+B profile",
            zorder=5,
        )
        ax.set_title(
            LABELS[key],
            color=COLORS[key],
            fontweight="semibold",
            pad=8,
        )
        ax.set_ylabel("Events / 0.5 MeV" if col == 0 else "")
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(labelbottom=False)

        offset = 0.07
        ax_res.axhline(0.0, color="#555B65", linewidth=0.9, zorder=1)
        ax_res.plot(
            x - offset,
            obs - null_total,
            marker="o",
            markersize=3.3,
            markerfacecolor="white",
            markeredgecolor=COLORS["gp"],
            linestyle="none",
            zorder=4,
        )
        ax_res.plot(
            x,
            obs - standalone_total,
            marker="^",
            markersize=3.5,
            color=COLORS["standalone"],
            linestyle="none",
            zorder=5,
        )
        ax_res.plot(
            x + offset,
            obs - shared_total,
            marker="o",
            markersize=3.2,
            color=COLORS["shared"],
            linestyle="none",
            zorder=6,
        )
        ax_res.set_xlabel(r"$m_{e^+e^-}$ (MeV)")
        ax_res.set_ylabel("Data - profile" if col == 0 else "")
        ax_res.yaxis.set_major_locator(MaxNLocator(6))
        ax_res.xaxis.set_major_locator(MaxNLocator(7))
        ax_res.set_xlim(1e3 * model.pred.blind[0], 1e3 * model.pred.blind[1])

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=COLORS["ink"],
            markersize=4.0, label="Observed data"
        ),
        Line2D(
            [0], [0], color=COLORS["gp"], linewidth=1.9,
            label="Background-only profiled expectation"
        ),
        Line2D(
            [0], [0], color=COLORS["standalone"], linewidth=1.9,
            linestyle=(0, (5, 2.5)), label="Standalone profiled S+B"
        ),
        Line2D(
            [0], [0], color=COLORS["shared"], linewidth=2.1,
            label=r"Shared-$\epsilon^2$ profiled S+B"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Profiled 65 MeV extraction — exact likelihood window",
        y=0.985,
        fontsize=15.0,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.75)
    return save_figure(fig, "figure61_common_0p5MeV_profiled")


def pearson_residual(
    observed: np.ndarray, expectation: np.ndarray
) -> np.ndarray:
    expectation = np.clip(np.asarray(expectation, float), 1.0, None)
    return (
        np.asarray(observed, float) - expectation
    ) / np.sqrt(expectation)


def plot_native_figure62_physical(result: VariantResult) -> List[Dict[str, Any]]:
    """Preserve the native Figure 62 residual panels and fix its coefficient panel."""
    fig = plt.figure(figsize=(14.8, 7.8))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.76,
        hspace=0.46,
        wspace=0.40,
    )
    axes = [fig.add_subplot(grid[index // 2, index % 2]) for index in range(4)]
    for index, model in enumerate(result.models):
        ax = axes[index]
        key = model.ds.key
        x = 1e3 * np.asarray(model.pred.x_full[model.pred.blind_mask], float)
        obs = np.asarray(model.pred.obs, float)
        null_expectation = np.asarray(model.null_profile["lambda_hat"], float)
        standalone_expectation = np.asarray(model.fit_signed["lambda_hat"], float)
        shared_expectation = np.asarray(
            result.shared_lambda_by_dataset[key], float
        )
        bin_width = 1e3 * float(np.median(np.diff(model.pred.edges_full)))
        offset = 0.16 * bin_width
        residual_arrays = [
            pearson_residual(obs, null_expectation),
            pearson_residual(obs, standalone_expectation),
            pearson_residual(obs, shared_expectation),
        ]

        ax.axhspan(-2.0, 2.0, color="#DDE2E8", alpha=0.48, zorder=0)
        ax.axhline(0.0, color="#555B65", linewidth=0.9, zorder=1)
        ax.plot(
            x - offset,
            residual_arrays[0],
            marker="o",
            markersize=3.6,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["gp"],
            markeredgewidth=0.9,
            zorder=4,
        )
        ax.plot(
            x,
            residual_arrays[1],
            marker="^",
            markersize=3.7,
            linestyle="none",
            color=COLORS["standalone"],
            zorder=5,
        )
        ax.plot(
            x + offset,
            residual_arrays[2],
            marker="o",
            markersize=3.4,
            linestyle="none",
            color=COLORS["shared"],
            zorder=6,
        )
        ax.set_title(
            LABELS[key],
            color=COLORS[key],
            fontweight="semibold",
        )
        ax.set_ylabel("")
        ax.set_xlabel(r"$m_{e^+e^-}$ (MeV)" if index == 2 else "")
        ax.set_xlim(1e3 * model.pred.blind[0], 1e3 * model.pred.blind[1])
        ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.yaxis.set_major_locator(MaxNLocator(7))
        max_abs = max(
            3.0,
            float(np.nanmax(np.abs(np.concatenate(residual_arrays)))) * 1.12,
        )
        ax.set_ylim(-max_abs, max_abs)

    ax = axes[3]
    y_positions = np.arange(4, dtype=float)
    row_labels = ["2015", "2016", "2021 10%", "Shared fit"]
    interval_rows = [
        (
            float(model.interval68["eps2_hat_bounded"]),
            float(model.interval68["eps2_low68"]),
            float(model.interval68["eps2_high68"]),
        )
        for model in result.models
    ] + [
        (
            float(result.interval68["eps2_hat_bounded"]),
            float(result.interval68["eps2_low68"]),
            float(result.interval68["eps2_high68"]),
        )
    ]
    point_colors = [COLORS[key] for key in DATASET_ORDER] + [COLORS["shared"]]
    ax.axvline(0.0, color="#555B65", linewidth=0.9, zorder=1)
    for y_pos, (estimate, low, high), color in zip(
        y_positions, interval_rows, point_colors
    ):
        x = estimate * 1e6
        xerr = np.array(
            [
                [max(0.0, estimate - low) * 1e6],
                [max(0.0, high - estimate) * 1e6],
            ]
        )
        ax.errorbar(
            x,
            y_pos,
            xerr=xerr,
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.2,
            elinewidth=1.55,
            capsize=3.0,
            zorder=4,
        )
    ax.set_yticks(y_positions, row_labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0.0)
    ax.set_xlabel(
        r"Physical $\widehat{\epsilon^2}$ and nominal 68% profile set"
        "\n"
        r"($\times10^{-6}$)"
    )
    ax.set_title(
        "Physical-domain signal estimates",
        fontweight="semibold",
        pad=27,
    )
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.text(
        0.5,
        1.01,
        r"$\Delta(-2\ln L)=1$; asymptotic, not coverage calibrated",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#4A4F57",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["gp"],
            label="Profiled background only",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            color=COLORS["standalone"],
            label="Standalone signal + background",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=COLORS["shared"],
            label=r"Shared-$\epsilon^2$ signal + background",
        ),
        Patch(
            facecolor="#DDE2E8",
            alpha=0.48,
            edgecolor="none",
            label=r"$|r|<2$ guide",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=4,
        frameon=False,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Profiled residual diagnostics at 65 MeV — physical coefficient panel",
        y=0.985,
        fontsize=15.0,
        fontweight="semibold",
    )
    fig.text(
        0.022,
        0.43,
        "Conditional Pearson residual",
        rotation=90,
        ha="center",
        va="center",
        fontsize=11.0,
    )
    return save_figure(fig, "figure62_profiled_residuals_physical68")


def plot_physical_coefficients(
    native: VariantResult, common: VariantResult
) -> List[Dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    labels = ["2015", "2016", "2021 10%", "Shared fit"]
    y = np.arange(4, dtype=float)

    def rows(result: VariantResult) -> List[Tuple[float, float, float]]:
        values = [
            (
                float(model.interval68["eps2_hat_bounded"]),
                float(model.interval68["eps2_low68"]),
                float(model.interval68["eps2_high68"]),
            )
            for model in result.models
        ]
        values.append(
            (
                float(result.interval68["eps2_hat_bounded"]),
                float(result.interval68["eps2_low68"]),
                float(result.interval68["eps2_high68"]),
            )
        )
        return values

    for result, offset, color, marker, label, fill in (
        (native, -0.12, COLORS["native"], "o", "Native v4.2", "white"),
        (
            common,
            +0.12,
            COLORS["common"],
            "D",
            "Common 0.5 MeV refit",
            COLORS["common"],
        ),
    ):
        for index, (estimate, low, high) in enumerate(rows(result)):
            x = estimate * 1e6
            xerr = np.array([[max(0.0, estimate - low) * 1e6], [max(0.0, high - estimate) * 1e6]])
            ax.errorbar(
                x,
                y[index] + offset,
                xerr=xerr,
                fmt=marker,
                color=color,
                markerfacecolor=fill,
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=6.4,
                elinewidth=1.7,
                capsize=3.2,
                zorder=4,
            )
    ax.axvline(0.0, color="#555B65", linewidth=1.0, zorder=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Physical $\widehat{\epsilon^2}$ and nominal 68% profile set"
        r"  ($\times10^{-6}$)"
    )
    ax.set_title(
        "65 MeV signal-strength estimates",
        fontweight="semibold",
        pad=32,
    )
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MaxNLocator(8))
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=COLORS["native"],
            markerfacecolor="white", markersize=6.4, label="Native v4.2"
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", color=COLORS["common"],
            markerfacecolor=COLORS["common"], markersize=6.4,
            label="Common 0.5 MeV refit"
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.text(
        0.5,
        1.015,
        r"Physical domain $\epsilon^2\geq0$; "
        r"$\Delta(-2\ln L)=1$ (asymptotic, not coverage calibrated)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#4A4F57",
    )
    fig.tight_layout()
    return save_figure(fig, "figure62_coefficients_physical68")


def copy_native_reference() -> List[Dict[str, Any]]:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        NATIVE_FIT_SUMMARY,
        NATIVE_PLOT_DATA,
        NATIVE_PROVENANCE,
        NATIVE_EXTRACTION_DIR / "observed_extraction_m065_wide.pdf",
        NATIVE_EXTRACTION_DIR / "observed_extraction_m065_wide.png",
        NATIVE_EXTRACTION_DIR / "observed_extraction_m065_profiled_residuals.pdf",
        NATIVE_EXTRACTION_DIR / "observed_extraction_m065_profiled_residuals.png",
    ]
    records: List[Dict[str, Any]] = []
    for source in sources:
        destination = REFERENCE_DIR / source.name
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Reference copy hash mismatch: {source}")
        records.append(
            {
                "source": repo_path(source),
                "copy": repo_path(destination),
                "sha256": sha256(destination),
            }
        )
    return records


def native_validation(
    native: VariantResult, authoritative: Mapping[str, pd.Series]
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    for model in native.models:
        row = authoritative[model.ds.key]
        eps_hat = float(model.fit_signed["A_hat"]) / model.k_events_per_eps2
        eps_sigma = float(model.fit_signed["sigma_A"]) / model.k_events_per_eps2
        for quantity, actual, expected, atol, rtol in (
            (
                "eps2_hat",
                eps_hat,
                float(row["standalone_eps2_hat"]),
                2e-11,
                3e-6,
            ),
            (
                "eps2_sigma",
                eps_sigma,
                float(row["standalone_sigma_eps2"]),
                2e-11,
                3e-6,
            ),
            (
                "p0",
                model.p0,
                float(row["standalone_p0_asymptotic"]),
                3e-8,
                3e-5,
            ),
            (
                "Z",
                model.Z,
                float(row["standalone_Z_asymptotic"]),
                3e-5,
                3e-6,
            ),
        ):
            passed = bool(np.isclose(actual, expected, atol=atol, rtol=rtol))
            checks.append(
                {
                    "scope": model.ds.key,
                    "quantity": quantity,
                    "actual": actual,
                    "authoritative": expected,
                    "abs_delta": abs(actual - expected),
                    "passed": passed,
                }
            )
    row = authoritative["combined"]
    for quantity, actual, expected, atol, rtol in (
        (
            "eps2_hat",
            native.eps2_hat_signed,
            float(row["shared_eps2_hat"]),
            2e-11,
            3e-6,
        ),
        (
            "eps2_sigma",
            native.sigma_eps2_wald,
            float(row["shared_sigma_eps2"]),
            2e-11,
            3e-6,
        ),
        (
            "p0",
            native.p0,
            float(row["shared_p0_asymptotic"]),
            3e-9,
            3e-5,
        ),
        (
            "Z",
            native.Z,
            float(row["shared_Z_asymptotic"]),
            3e-5,
            3e-6,
        ),
    ):
        passed = bool(np.isclose(actual, expected, atol=atol, rtol=rtol))
        checks.append(
            {
                "scope": "combined",
                "quantity": quantity,
                "actual": actual,
                "authoritative": expected,
                "abs_delta": abs(actual - expected),
                "passed": passed,
            }
        )
    return {
        "checks": checks,
        "all_passed": bool(all(row["passed"] for row in checks)),
    }


def optimizer_validation(rows: pd.DataFrame) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    all_stable = True
    for (variant, dataset), group in rows.groupby(["variant", "dataset"]):
        lml_span = float(group["lml"].max() - group["lml"].min())
        ls_values = group["ls_opt"].to_numpy(float)
        const_values = group["const_opt"].to_numpy(float)
        ls_rel_span = float(
            (np.max(ls_values) - np.min(ls_values))
            / max(abs(float(np.max(ls_values))), 1e-12)
        )
        const_rel_span = float(
            (np.max(const_values) - np.min(const_values))
            / max(abs(float(np.max(const_values))), 1e-12)
        )
        stable = bool(
            np.isfinite(lml_span)
            and lml_span <= 1e-3
            and ls_rel_span <= 2e-3
            and const_rel_span <= 2e-3
            and not bool(group["ls_at_lower"].any())
            and not bool(group["ls_at_upper"].any())
        )
        all_stable = all_stable and stable
        groups.append(
            {
                "variant": variant,
                "dataset": dataset,
                "n_repeats": int(len(group)),
                "lml_span": lml_span,
                "ls_relative_span": ls_rel_span,
                "const_relative_span": const_rel_span,
                "any_ls_at_lower": bool(group["ls_at_lower"].any()),
                "any_ls_at_upper": bool(group["ls_at_upper"].any()),
                "stable": stable,
            }
        )
    return {"groups": groups, "all_stable": all_stable}


def impossible_0p625_record(
    source_meta: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    target = 0.000625
    rows = []
    for key in DATASET_ORDER:
        width = float(source_meta[key]["source_bin_width_GeV"])
        ratio = target / width
        exact = bool(np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10))
        rows.append(
            {
                "dataset": key,
                "source_bin_width_MeV": 1e3 * width,
                "target_bin_width_MeV": 1e3 * target,
                "target_over_source": ratio,
                "integer_aggregation": exact,
                "event_level_objects_in_input_root": source_meta[key][
                    "event_level_objects"
                ],
            }
        )
    return {
        "target_bin_width_MeV": 0.625,
        "rows": rows,
        "exact_for_all_inputs": bool(
            all(row["integer_aggregation"] for row in rows)
        ),
        "decision": (
            "REJECT: exact 0.625 MeV inference bins cannot be constructed from "
            "the histogram-only 2015/2016 inputs. Fractional bin splitting, "
            "weighting, rounding, or truncation is forbidden for the Poisson likelihood."
        ),
    }


def write_captions(
    results: Mapping[str, VariantResult],
    table: pd.DataFrame,
) -> None:
    common = results["common_0p5MeV"]
    native = results["native_v4p2"]
    row2016 = table[
        (table["variant"] == "native_v4p2") & (table["scope"] == "2016")
    ].iloc[0]
    text = f"""# Figure captions

## `figure61_common_0p5MeV`

Observed extraction at 65 MeV after an exact, count-preserving common-bin refit.
The 2015, 2016, and 2021 histograms all use 0.5 MeV bins, obtained by integer
aggregation factors 10, 10, and 4 from their source histograms. The top panels
show observed data, the newly optimized sideband-trained GP mean and predictive
uncertainty, and display extensions of the standalone and simultaneous
shared-$\\epsilon^2$ signals. The lower panels show data minus the pre-profile
GP mean with the original Figure 61 display uncertainty
$\\sqrt{{\\mu_{{\\rm GP}}+C_{{ii}}}}$. This is a prior-predictive,
correlated-background diagnostic; the bars are not independent-bin post-fit
errors. This is a local binning-robustness result, not a replacement for the
accepted native v4.2 scan.

## `figure61_common_0p5MeV_profiled`

Count-space profiled extraction for the same exact 0.5 MeV refit, restricted to
the actual $\\pm2.25\\sigma_m$ likelihood window. Blue shows the background-only
profile, orange the standalone signal-plus-background profile, and red the
simultaneous shared-$\\epsilon^2$ signal-plus-background profile. The lower
panels show count residuals relative to each profiled expectation. These are
correlated fit diagnostics, not standardized per-bin significances. The curves
are not extended into sidebands because the v4.2 nuisance likelihood profiles
the GP background only in the extraction window.

## `figure62_profiled_residuals_physical68`

Corrected native-v4.2 Figure 62 composite. The three dataset panels retain the
conditional Pearson residuals for the background-only, standalone
signal-plus-background, and simultaneous shared-$\\epsilon^2$
signal-plus-background profiles in the exact $\\pm2.25\\sigma_m$ likelihood
windows. These correlated fit diagnostics are not independent local
significances. The lower-right panel replaces the signed symmetric-Wald
display with the physical $\\epsilon^2\\geq0$ profile-likelihood sets defined
by $\\Delta(-2\\ln L)=1$. The 68% interpretation is nominal and asymptotic,
not coverage calibrated; the native 2016 lower endpoint is zero.

## `figure62_coefficients_physical68`

Physical-domain 65 MeV signal-strength estimates for the authoritative native
v4.2 extraction and the exact common-0.5-MeV refit. Horizontal intervals are
the $\\epsilon^2\\geq0$ profile-likelihood sets defined by
$\\Delta(-2\\ln L)=1$; their 68% interpretation is nominal and asymptotic, not
coverage calibrated. The native 2016 lower endpoint is zero. In the original
Figure 62, the 2016 signed estimator was
$({row2016.eps2_hat_signed * 1e6:.5f}\\pm{row2016.eps2_sigma_wald * 1e6:.5f})
\\times10^{{-6}}$; its symmetric Wald extension reached
${row2016.eps2_wald_low * 1e6:.5f}\\times10^{{-6}}$. That extension was an
unconstrained estimator uncertainty, not a negative physical coupling and not
a conversion error.

## Numerical scope

At 65 MeV the native combined reconstruction gives
$Z_{{\\rm local}}={native.Z:.5f}$ and
$\\widehat{{\\epsilon^2}}={native.eps2_hat_signed * 1e6:.5f}\\times10^{{-6}}$.
The exact common-0.5-MeV refit gives
$Z_{{\\rm local}}={common.Z:.5f}$ and
$\\widehat{{\\epsilon^2}}={common.eps2_hat_signed * 1e6:.5f}\\times10^{{-6}}$.
These are fixed-mass asymptotic profile-likelihood quantities. No scan-wide
minimum or global significance was recomputed in this local study.
"""
    CAPTIONS_PATH.write_text(text)


def write_readme(
    results: Mapping[str, VariantResult],
    extraction: pd.DataFrame,
    impossible: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    common = results["common_0p5MeV"]
    stress = results["common_1p25MeV"]
    native = results["native_v4p2"]
    rows = extraction[extraction["scope"] == "combined"].set_index("variant")
    native2016 = extraction[
        (extraction["variant"] == "native_v4p2")
        & (extraction["scope"] == "2016")
    ].iloc[0]
    text = f"""# v4.2 65 MeV extraction binning follow-up

Status: **{validation["status"]}**

This directory is a self-contained follow-up to Figures 61 and 62 of the
authoritative HPS-GPR v4.2 note (study commit
`fb1295680bacdd5edbabff9546ee200e3c68b78a`). It does not edit or replace the
analysis note.

## Main finding

An exact 0.625 MeV rebin is not available from the supplied histogram-only
2015/2016 inputs. Their source bins are 0.05 MeV wide, so the required factor is
12.5. Splitting source-bin counts would create fractional pseudo-counts and
invalidate the integer-Poisson likelihood. The ROOT inputs contain no event
TTrees or RNTuples from which the mass histograms could be rebuilt. The study
therefore uses 0.5 MeV, the finer of the equidistant 0.5/0.75 MeV
source-compatible choices and the nearest one that retains all three full
supports, with integer factors 10/10/4. An exact 1.25 MeV coarsening stress
test uses factors 25/25/10.

## Fixed-mass results at 65 MeV

| Binning | Combined $\\widehat{{\\epsilon^2}}$ | Wald $\\sigma$ | local $p_0$ | local $Z$ | $\\Delta Z$ vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native v4.2 | {native.eps2_hat_signed:.7g} | {native.sigma_eps2_wald:.7g} | {native.p0:.7g} | {native.Z:.6f} | 0 |
| Common 0.5 MeV | {common.eps2_hat_signed:.7g} | {common.sigma_eps2_wald:.7g} | {common.p0:.7g} | {common.Z:.6f} | {common.Z - native.Z:+.6f} |
| Common 1.25 MeV | {stress.eps2_hat_signed:.7g} | {stress.sigma_eps2_wald:.7g} | {stress.p0:.7g} | {stress.Z:.6f} | {stress.Z - native.Z:+.6f} |

The 0.5 MeV result is a newly optimized local GP refit using the same physical
mass resolution, training exclusion, fit support, radiative conversion, and
profile likelihood as v4.2. The density normalization remains sourced from the
uncropped fine histogram and the exact $m\\pm1.64\\sigma_m$ window. The 1.25 MeV
stress histogram for 2015 ends at 134 MeV rather than 135 MeV because 121 MeV
of support is not divisible into uniform 1.25 MeV bins; this one-MeV far-side
trim is recorded in the tables and is why 0.5 MeV is the primary comparison.

This is a fixed-mass study. It tests the 65 MeV extraction and local asymptotic
significance only; it does not establish that 65 MeV remains the minimum of a
rebinned full scan and does not recompute the analytic Sidak reference or a
scan-toy global significance.

## Why the original 2016 error bar crossed zero

The native standalone 2016 result is
$\\widehat{{\\epsilon^2}}={native2016.eps2_hat_signed:.7g}$ with symmetric local
Wald uncertainty $\\sigma={native2016.eps2_sigma_wald:.7g}$. Its lower plotted
endpoint was {native2016.eps2_wald_low:.7g}. The fit deliberately allowed a
signed signal-strength estimator and the code divided both the fitted event
amplitude and its positive uncertainty by the same positive conversion factor.
Thus the extension below zero was not a sign or normalization bug. It was a
symmetric curvature uncertainty on an unconstrained estimator. The new
Figure 62 composite preserves the three residual panels and replaces only its
coefficient panel with the physical $\\epsilon^2\\geq0$ profile set; the 2016
lower endpoint is zero. The displayed 68% sets are nominal/asymptotic and have
not been coverage calibrated.

## Outputs

- `reference_v4p2/`: bitwise copies of the authoritative native extraction
  tables, provenance, and Figures 61/62.
- `tables/extraction_comparison.csv`: standalone and combined fit results for
  all three binnings, including differences from native.
- `tables/profile_intervals68.csv`: physical-domain profile intervals.
- `tables/optimizer_repeats.csv`: two independent 12-restart fits per dataset
  and alternative binning, with the maximum-LML branch selected.
- `tables/bin_level_common_0p5MeV.csv`: plotted counts and profiled
  expectations.
- `figures/figure61_common_0p5MeV.*`: clean common-bin extraction.
- `figures/figure61_common_0p5MeV_profiled.*`: exact-window profiled version.
- `figures/figure62_profiled_residuals_physical68.*`: corrected native
  Figure 62 composite preserving all three residual panels.
- `figures/figure62_coefficients_physical68.*`: corrected physical interval
  comparison between native v4.2 and the common-bin refit.
- `validation.json`: machine-readable pass/fail checks.
- `provenance.json`: input, code, commit, histogram, and output hashes.
- `CAPTIONS.md`: publication-ready caption text and interpretation boundaries.
- `VISUAL_QA.md`: manual original-resolution PNG and one-page PDF inspection
  record.

## Reproduce

From the repository root:

```bash
MPLCONFIGDIR=/tmp/codex-mpl-v4p2-m065-followup \\
python3 study_results/v4p2_followups_20260806/m065_extraction/run_m065_common_binning_study.py
```

The script refuses noninteger rebins, validates native reconstruction against
the accepted v4.2 table, runs optimizer-repeat checks, verifies nonnegative
physical interval endpoints, and writes only inside this directory.
"""
    README_PATH.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip the 1.25 MeV stress fit (validation will be partial).",
    )
    args = parser.parse_args()
    set_style()
    for directory in (REFERENCE_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        CONFIG,
        REVIEWED_STATES,
        ENRICHED_STATES,
        NATIVE_FIT_SUMMARY,
        NATIVE_PLOT_DATA,
        NATIVE_PROVENANCE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Authoritative v4.2 config hash mismatch")
    if sha256(REVIEWED_STATES) != EXPECTED_REVIEWED_STATES_SHA256:
        raise RuntimeError("Authoritative reviewed-state hash mismatch")

    config = load_config(str(CONFIG))
    states = pd.read_csv(REVIEWED_STATES)
    native_fit = pd.read_csv(NATIVE_FIT_SUMMARY)
    authoritative = authoritative_rows(native_fit)
    datasets = make_datasets(config)
    if tuple(key for key in DATASET_ORDER if key in datasets) != DATASET_ORDER:
        raise RuntimeError(f"Unexpected enabled datasets: {sorted(datasets)}")
    source_hists = {key: source_histogram(datasets[key]) for key in DATASET_ORDER}
    source_meta = {
        key: source_metadata(datasets[key], source_hists[key])
        for key in DATASET_ORDER
    }
    impossible = impossible_0p625_record(source_meta)
    if impossible["exact_for_all_inputs"]:
        raise RuntimeError("0.625 MeV unexpectedly became an exact aggregation")

    reference_records = copy_native_reference()
    optimizer_rows: List[Dict[str, Any]] = []
    variants_to_run = ["native_v4p2", "common_0p5MeV"]
    if not args.skip_stress:
        variants_to_run.append("common_1p25MeV")
    results: Dict[str, VariantResult] = {}
    for name in variants_to_run:
        print(f"[m065] building {name}", flush=True)
        results[name] = build_variant(
            name,
            config,
            datasets,
            states,
            source_hists,
            optimizer_rows,
        )
        print(
            f"[m065] {name}: eps2hat={results[name].eps2_hat_signed:.9g}, "
            f"p0={results[name].p0:.9g}, Z={results[name].Z:.6f}",
            flush=True,
        )
    if args.skip_stress:
        results["common_1p25MeV"] = results["common_0p5MeV"]

    extraction, intervals = extraction_rows(results)
    optimizer_df = pd.DataFrame(optimizer_rows)
    bin_rows = bin_level_rows(results["common_0p5MeV"])
    extraction_path = TABLE_DIR / "extraction_comparison.csv"
    intervals_path = TABLE_DIR / "profile_intervals68.csv"
    optimizer_path = TABLE_DIR / "optimizer_repeats.csv"
    bins_path = TABLE_DIR / "bin_level_common_0p5MeV.csv"
    extraction.to_csv(extraction_path, index=False, float_format="%.17g")
    intervals.to_csv(intervals_path, index=False, float_format="%.17g")
    optimizer_df.to_csv(optimizer_path, index=False, float_format="%.17g")
    bin_rows.to_csv(bins_path, index=False, float_format="%.17g")

    figures: List[Dict[str, Any]] = []
    figures.extend(plot_common_figure61(results["common_0p5MeV"], config))
    figures.extend(plot_profiled_figure61(results["common_0p5MeV"]))
    figures.extend(plot_native_figure62_physical(results["native_v4p2"]))
    figures.extend(
        plot_physical_coefficients(
            results["native_v4p2"], results["common_0p5MeV"]
        )
    )

    native_check = native_validation(results["native_v4p2"], authoritative)
    optimizer_check = optimizer_validation(optimizer_df)
    exact_width_checks = []
    for variant in ("common_0p5MeV", "common_1p25MeV"):
        expected = float(VARIANTS[variant]["target_width_GeV"])
        for model in results[variant].models:
            passed = bool(
                np.isclose(
                    model.pred.bin_width_median,
                    expected,
                    rtol=0.0,
                    atol=2e-13,
                )
                and np.allclose(
                    model.pred.y_full,
                    np.rint(model.pred.y_full),
                    rtol=0.0,
                    atol=1e-9,
                )
            )
            exact_width_checks.append(
                {
                    "variant": variant,
                    "dataset": model.ds.key,
                    "expected_width_GeV": expected,
                    "actual_width_GeV": model.pred.bin_width_median,
                    "integer_counts": bool(
                        np.allclose(
                            model.pred.y_full,
                            np.rint(model.pred.y_full),
                            rtol=0.0,
                            atol=1e-9,
                        )
                    ),
                    "passed": passed,
                }
            )
    profile_checks = [
        {
            "variant": str(row.variant),
            "scope": str(row.scope),
            "low_nonnegative": bool(float(row.eps2_low68) >= 0.0),
            "ordered": bool(
                float(row.eps2_low68)
                <= float(row.eps2_hat_bounded)
                <= float(row.eps2_high68)
            ),
        }
        for row in intervals.itertuples(index=False)
    ]
    native_2016_row = extraction[
        (extraction["variant"] == "native_v4p2")
        & (extraction["scope"] == "2016")
    ].iloc[0]
    native_2016_signed_wald = {
        "estimator_semantics": (
            "signed unconstrained signal-strength estimator mapped linearly to "
            "epsilon squared; not a physical confidence interval"
        ),
        "eps2_hat_signed": float(native_2016_row["eps2_hat_signed"]),
        "eps2_sigma_wald": float(native_2016_row["eps2_sigma_wald"]),
        "eps2_wald_low": float(native_2016_row["eps2_wald_low"]),
        "eps2_wald_high": float(native_2016_row["eps2_wald_high"]),
        "extends_below_zero": bool(native_2016_row["eps2_wald_low"] < 0.0),
        "K_events_per_eps2": float(native_2016_row["K_events_per_eps2"]),
        "conversion_factor_positive": bool(
            native_2016_row["K_events_per_eps2"] > 0.0
        ),
        "physical_profile68_low": float(
            native_2016_row["eps2_profile68_low"]
        ),
        "physical_profile68_high": float(
            native_2016_row["eps2_profile68_high"]
        ),
        "physical_interval_semantics": (
            "epsilon squared >= 0 profile-likelihood set with "
            "Delta(-2 ln L)=1; nominal asymptotic 68 percent set, "
            "not coverage calibrated"
        ),
        "diagnosis": (
            "The negative extension is the symmetric Wald uncertainty of an "
            "unconstrained estimator. It is not a negative physical coupling "
            "and not a sign or epsilon-squared conversion bug."
        ),
    }
    output_hashes = [
        {
            "path": repo_path(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            extraction_path,
            intervals_path,
            optimizer_path,
            bins_path,
        )
    ] + figures
    status = "PASS"
    reasons: List[str] = []
    if not native_check["all_passed"]:
        status = "FAIL"
        reasons.append("native reconstruction mismatch")
    if not optimizer_check["all_stable"]:
        status = "FAIL"
        reasons.append("optimizer repeats or length-scale boundary check failed")
    if not all(row["passed"] for row in exact_width_checks):
        status = "FAIL"
        reasons.append("exact-width/integer-count check failed")
    if not all(row["low_nonnegative"] and row["ordered"] for row in profile_checks):
        status = "FAIL"
        reasons.append("physical profile interval check failed")
    if args.skip_stress:
        status = "PARTIAL"
        reasons.append("1.25 MeV stress fit skipped")
    validation = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasons": reasons,
        "native_reconstruction": native_check,
        "optimizer_repeats": optimizer_check,
        "exact_width_and_integer_count_checks": exact_width_checks,
        "physical_profile_interval_checks": profile_checks,
        "native_2016_signed_wald_diagnostic": native_2016_signed_wald,
        "impossible_0p625MeV": impossible,
        "interpretation_boundary": (
            "Fixed-mass 65 MeV local asymptotic extraction robustness only; "
            "not a full rebinned scan, analytic Sidak refresh, scan-toy global "
            "significance, expected band, or coverage study."
        ),
        "output_hashes": output_hashes,
    }
    write_json(VALIDATION_PATH, validation)

    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": git_text("rev-parse", "HEAD"),
            "branch": git_text("branch", "--show-current"),
            "status_before_delivery": git_text("status", "--short"),
        },
        "authoritative_v4p2": {
            "study_commit": "fb1295680bacdd5edbabff9546ee200e3c68b78a",
            "config": repo_path(CONFIG),
            "config_sha256": sha256(CONFIG),
            "reviewed_states": repo_path(REVIEWED_STATES),
            "reviewed_states_sha256": sha256(REVIEWED_STATES),
            "native_fit_summary": repo_path(NATIVE_FIT_SUMMARY),
            "native_fit_summary_sha256": sha256(NATIVE_FIT_SUMMARY),
        },
        "source_histograms": source_meta,
        "variant_definitions": VARIANTS,
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "profile_interval_definition": (
            "physical POI >= 0, profiled Gaussian-background Poisson likelihood, "
            "Delta(-2 ln L)=1; nominal asymptotic 68 percent set, not "
            "coverage calibrated"
        ),
        "reference_copies": reference_records,
        "script": {
            "path": repo_path(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "validation": {
            "path": repo_path(VALIDATION_PATH),
            "sha256": sha256(VALIDATION_PATH),
            "status": status,
        },
        "outputs": output_hashes,
    }
    write_json(PROVENANCE_PATH, provenance)
    write_captions(results, extraction)
    write_readme(results, extraction, impossible, validation)

    # Add final text artifacts to the output ledger after they exist.
    for path in (README_PATH, CAPTIONS_PATH, VALIDATION_PATH, PROVENANCE_PATH):
        print(f"[m065] wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)")
    print(json.dumps({"status": status, "reasons": reasons}, indent=2))
    return 0 if status in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
