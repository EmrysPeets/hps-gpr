#!/usr/bin/env python3
"""Build publication-ready v4.2 observed extraction figures at 65 MeV.

The script reconstructs the exact reviewed fixed-GP states used by the v4.2
combined result.  It produces:

* a wide-window observed-count and background-subtracted comparison for the
  full 2015, full 2016, and 2021 10% samples; and
* exact-window conditional Pearson residuals plus a comparison of the three
  standalone signal estimates with the simultaneous shared-epsilon-squared
  estimate.

The wide-window signal curves are display extensions formed by adding the
profiled signal amplitudes to the pre-profile GP mean.  The likelihood itself
profiles the GP-background nuisance parameters only in the configured
plus-or-minus 2.25-sigma extraction window.  The residual figure uses those
exact profiled window expectations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from hps_gpr.config import load_config
from hps_gpr.conversion import A_from_epsilon2
from hps_gpr.dataset import DatasetConfig, make_datasets
from hps_gpr.evaluation import build_combined_components
from hps_gpr.gpr import (
    compute_kernel_ls_bounds,
    fit_gpr,
    make_fixed_kernel,
    predict_counts_from_log_gpr,
    predict_counts_mean_from_log_gpr,
)
from hps_gpr.io import (
    BlindPrediction,
    _blind_pred_detail,
    _build_model,
    _compute_integral_density,
)
from hps_gpr.statistics import (
    fit_A_profiled_gaussian_details,
    p0_profiled_gaussian_LRT,
    profile_theta_given_A,
)
from hps_gpr.template import build_window_template_from_full


MASS_GEV = 0.065
MASS_MEV = 65.0
DATASET_ORDER = ("2015", "2016", "2021")

CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
STATES = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
ENRICHED_STATES = HERE / "derived" / "observed_gp_states_v4p2_enriched.csv"
COMBINED = HERE / "derived" / "combined_bands300_reviewed_v4p2.csv"
OUT = HERE / "note_figures" / "extractions_m065"
NOTE_OUT = (
    REPO
    / "hps_gpr_analysis_note"
    / "final_limit_projection_figs"
    / "v4p2_20260805_combined300"
)
PUBLISH_OUT = REPO / "output" / "pdf"

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
}

LABELS = {
    "2015": "2015 100%",
    "2016": "2016 100%",
    "2021": "2021 10%",
}

EXPECTED_STATE_HASHES = {
    "2015": "2c1561bcc11251efc8c218267db6364814d4aa7117a6d87bb2334cf5017907b0",
    "2016": "9739e8f136c598fb4680fb3bb9852b57d99fca14530b7a8f6807365413ade411",
    "2021": "c02fb8a3fc4bbe27ec9021f61d0eb0bd2f405538aa2a6379fa7dde71dc0102b4",
}


@dataclass
class ModelBundle:
    ds: DatasetConfig
    pred: BlindPrediction
    gpr: object
    template_window: np.ndarray
    template_full: np.ndarray
    standalone_fit: Mapping[str, object]
    standalone_null: Mapping[str, object]
    standalone_p0: float
    standalone_Z: float
    k_events_per_eps2: float
    state_sha256: str


@dataclass
class JointResult:
    config: object
    models: List[ModelBundle]
    eps2_hat: float
    sigma_eps2: float
    p0: float
    Z: float
    q0: float
    fit: Mapping[str, object]
    null: Mapping[str, object]
    shared_bkg_by_dataset: Dict[str, np.ndarray]
    shared_lambda_by_dataset: Dict[str, np.ndarray]
    null_lambda_by_dataset: Dict[str, np.ndarray]
    q0_contribution_by_dataset: Dict[str, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def state_row(states: pd.DataFrame, dataset: str) -> pd.Series:
    rows = states[
        (states["dataset"].astype(str) == dataset)
        & np.isclose(states["mass_GeV"].to_numpy(float), MASS_GEV, atol=5e-10)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one reviewed {dataset} state at {MASS_MEV:.0f} MeV; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def fixed_prediction(
    ds: DatasetConfig,
    config,
    row: pd.Series,
) -> Tuple[BlindPrediction, object]:
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
    model = _build_model(
        ds,
        blind,
        rebin=int(config.neighborhood_rebin),
        config=config,
        mass=MASS_GEV,
    )
    x = np.asarray(model.histogram.axes[0].centers, float)
    y = np.asarray(model.histogram.values(), float)
    train_mask = (x < blind_train[0]) | (x > blind_train[1])
    fixed_kernel = make_fixed_kernel(float(row["const_opt"]), float(row["ls_opt"]))
    gpr = fit_gpr(
        x[train_mask],
        y[train_mask],
        config,
        restarts=0,
        kernel=fixed_kernel,
        optimize=False,
    )
    mu, cov, obs, edges = _blind_pred_detail(model, gpr, blind, config)
    mu_full = predict_counts_mean_from_log_gpr(gpr, x, config)
    density_nsigma = float(
        getattr(config, "eps2_density_nsigma", None) or config.blind_nsigma
    )
    density = _compute_integral_density(
        model,
        MASS_GEV,
        sigma_val,
        density_nsigma=density_nsigma,
    )
    ls_info = compute_kernel_ls_bounds(ds, config, mass=MASS_GEV)
    full_edges = np.asarray(model.histogram.axes[0].edges, float)
    blind_mask = np.asarray((x >= blind[0]) & (x <= blind[1]), bool)
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
        blind_train=blind_train,
        kernel_str=str(getattr(gpr, "kernel_", fixed_kernel)),
        ls_lo=float(ls_info["ls_lo"]),
        ls_hi=float(ls_info["ls_hi"]),
        ls_init=float(ls_info["ls_init"]),
        ls_opt=float(row["ls_opt"]),
        sigma_x=float(ls_info["sigma_x"]),
        const_opt=float(row["const_opt"]),
        lml=float(gpr.log_marginal_likelihood_value_),
        n_train=int(np.count_nonzero(train_mask)),
    )
    return pred, gpr


def assert_close(
    name: str,
    actual: float,
    expected: float,
    *,
    atol: float,
    rtol: float,
) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{name} mismatch: reconstructed={actual:.16g}, "
            f"reviewed={expected:.16g}"
        )


def build_model(
    ds: DatasetConfig,
    config,
    fixed_row: pd.Series,
    metrics_row: pd.Series,
) -> ModelBundle:
    pred, gpr = fixed_prediction(ds, config, fixed_row)
    template_window, template_full = build_window_template_from_full(
        pred.edges_full,
        pred.blind_mask,
        MASS_GEV,
        pred.sigma_val,
        config=config,
    )
    standalone_fit = fit_A_profiled_gaussian_details(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
        allow_negative=True,
    )
    standalone_null = profile_theta_given_A(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
        A_fixed=0.0,
    )
    p0, z, _, info = p0_profiled_gaussian_LRT(
        pred.obs,
        pred.mu,
        pred.cov,
        template_window,
    )
    k = float(A_from_epsilon2(ds, MASS_GEV, 1.0, pred.integral_density))

    assert_close(
        f"{ds.key} sigma",
        pred.sigma_val,
        float(metrics_row["sigma_val"]),
        atol=2e-12,
        rtol=2e-10,
    )
    assert_close(
        f"{ds.key} density",
        pred.integral_density,
        float(metrics_row["integral_density"]),
        atol=2e-4,
        rtol=2e-9,
    )
    assert_close(
        f"{ds.key} LML",
        pred.lml,
        float(fixed_row["lml"]),
        atol=3e-5,
        rtol=0.0,
    )
    assert_close(
        f"{ds.key} standalone Ahat",
        float(standalone_fit["A_hat"]),
        float(metrics_row["A_hat"]),
        atol=0.08,
        rtol=2e-6,
    )
    assert_close(
        f"{ds.key} standalone p0",
        float(p0),
        float(metrics_row["p0_analytic"]),
        atol=2e-8,
        rtol=2e-5,
    )
    if not bool(standalone_fit["success"]) or not bool(info["ok"]):
        raise RuntimeError(f"{ds.key} standalone profile fit did not converge")

    return ModelBundle(
        ds=ds,
        pred=pred,
        gpr=gpr,
        template_window=np.asarray(template_window, float),
        template_full=np.asarray(template_full, float),
        standalone_fit=standalone_fit,
        standalone_null=standalone_null,
        standalone_p0=float(p0),
        standalone_Z=float(z),
        k_events_per_eps2=k,
        state_sha256=EXPECTED_STATE_HASHES[ds.key],
    )


def split_vector(
    vector: np.ndarray,
    models: Sequence[ModelBundle],
) -> Dict[str, np.ndarray]:
    arr = np.asarray(vector, float)
    result: Dict[str, np.ndarray] = {}
    offset = 0
    for model in models:
        size = int(model.pred.obs.size)
        result[model.ds.key] = np.asarray(arr[offset : offset + size], float)
        offset += size
    if offset != arr.size:
        raise RuntimeError(f"Split consumed {offset} of {arr.size} entries")
    return result


def build_joint(
    config,
    models: List[ModelBundle],
    combined_row: pd.Series,
) -> JointResult:
    obs, bkg, cov, s_unit = build_combined_components(
        MASS_GEV,
        [model.ds for model in models],
        [model.pred for model in models],
        config=config,
    )
    scale = float(config.eps2_lrt_scale)
    template = s_unit / scale
    fit = fit_A_profiled_gaussian_details(
        obs,
        bkg,
        cov,
        template,
        allow_negative=True,
    )
    null = profile_theta_given_A(
        obs,
        bkg,
        cov,
        template,
        A_fixed=0.0,
    )
    p0, z, q0, info = p0_profiled_gaussian_LRT(obs, bkg, cov, template)
    eps2_hat = float(fit["A_hat"]) / scale
    sigma_eps2 = float(fit["sigma_A"]) / scale

    assert_close(
        "combined p0",
        float(p0),
        float(combined_row["p0_analytic"]),
        atol=2e-9,
        rtol=2e-5,
    )
    assert_close(
        "combined Z",
        float(z),
        float(combined_row["Z_analytic"]),
        atol=2e-5,
        rtol=2e-6,
    )
    if not bool(fit["success"]) or not bool(info["ok"]):
        raise RuntimeError("Combined profile fit did not converge")

    q0_contributions: Dict[str, float] = {}
    offset = 0
    alt_lambda = np.asarray(fit["lambda_hat"], float)
    alt_theta = np.asarray(fit["theta_hat"], float)
    null_lambda = np.asarray(null["lambda_hat"], float)
    null_theta = np.asarray(null["theta_hat"], float)
    obs_float = np.asarray(obs, float)
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

        q0_contributions[model.ds.key] = 2.0 * (
            block_nll(null_lambda, null_theta)
            - block_nll(alt_lambda, alt_theta)
        )
        offset += size
    assert_close(
        "sum of joint q0 block contributions",
        float(sum(q0_contributions.values())),
        float(q0),
        atol=5e-5,
        rtol=1e-6,
    )

    return JointResult(
        config=config,
        models=models,
        eps2_hat=eps2_hat,
        sigma_eps2=sigma_eps2,
        p0=float(p0),
        Z=float(z),
        q0=float(q0),
        fit=fit,
        null=null,
        shared_bkg_by_dataset=split_vector(
            np.asarray(fit["b_fit"], float), models
        ),
        shared_lambda_by_dataset=split_vector(
            np.asarray(fit["lambda_hat"], float), models
        ),
        null_lambda_by_dataset=split_vector(
            np.asarray(null["lambda_hat"], float), models
        ),
        q0_contribution_by_dataset=q0_contributions,
    )


def display_mask(model: ModelBundle, half_width_sigma: float = 5.0) -> np.ndarray:
    x = np.asarray(model.pred.x_full, float)
    half_width = half_width_sigma * float(model.pred.sigma_val)
    return (x >= MASS_GEV - half_width) & (x <= MASS_GEV + half_width)


def save_pair(fig: plt.Figure, stem: str) -> Dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.mkdir(parents=True, exist_ok=True)
    PUBLISH_OUT.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {"stem": stem}
    for suffix, dpi in ((".pdf", None), (".png", 300)):
        path = OUT / f"{stem}{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        note_path = NOTE_OUT / path.name
        shutil.copy2(path, note_path)
        paths[suffix[1:]] = str(path.relative_to(REPO))
        paths[f"{suffix[1:]}_sha256"] = sha256(path)
        paths[f"note_{suffix[1:]}"] = str(note_path.relative_to(REPO))
        if suffix == ".pdf":
            publish_path = PUBLISH_OUT / path.name
            shutil.copy2(path, publish_path)
            paths["publish_pdf"] = str(publish_path.relative_to(REPO))
    plt.close(fig)
    return paths


def draw_blind_boundaries(ax: plt.Axes, model: ModelBundle) -> None:
    for bound in model.pred.blind:
        ax.axvline(
            1e3 * float(bound),
            color=COLORS["blind"],
            linewidth=1.15,
            linestyle=(0, (4, 3)),
            zorder=2,
        )


def wide_display_arrays(
    result: JointResult,
    model: ModelBundle,
) -> Dict[str, np.ndarray]:
    mask = display_mask(model)
    x = np.asarray(model.pred.x_full, float)
    y = np.asarray(model.pred.y_full, float)
    mu, cov = predict_counts_from_log_gpr(
        model.gpr,
        x[mask],
        result.config,
    )
    variance = np.clip(mu, 1.0, None) + np.clip(np.diag(cov), 0.0, None)
    standalone_signal = (
        float(model.standalone_fit["A_hat"]) * model.template_full[mask]
    )
    shared_signal = (
        result.eps2_hat * model.k_events_per_eps2 * model.template_full[mask]
    )
    return {
        "mask": mask,
        "x_GeV": x[mask],
        "x_MeV": 1e3 * x[mask],
        "observed": y[mask],
        "gp_mean": np.asarray(mu, float),
        "gp_sigma": np.sqrt(variance),
        "gp_predictive_sigma": np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        "standalone_signal": standalone_signal,
        "shared_signal": shared_signal,
    }


def plot_wide_extraction(result: JointResult) -> Dict[str, object]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.6, 7.8),
        sharex="col",
        gridspec_kw={"height_ratios": (1.55, 1.0), "hspace": 0.08, "wspace": 0.18},
    )
    all_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for col, model in enumerate(result.models):
        arrays = wide_display_arrays(result, model)
        all_arrays[model.ds.key] = arrays
        ax = axes[0, col]
        ax_res = axes[1, col]
        x = arrays["x_MeV"]
        y = arrays["observed"]
        mu = arrays["gp_mean"]
        gp_pred_sigma = arrays["gp_predictive_sigma"]
        standalone_signal = arrays["standalone_signal"]
        shared_signal = arrays["shared_signal"]
        bin_width = 1e3 * float(np.median(np.diff(model.pred.edges_full)))

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
            markersize=2.6,
            elinewidth=0.65,
            capsize=0.0,
            zorder=6,
        )
        draw_blind_boundaries(ax, model)
        ax.set_title(
            f"{LABELS[model.ds.key]}  ({bin_width:g} MeV/bin)",
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
            yerr=arrays["gp_sigma"],
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
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=COLORS["ink"],
            markersize=4.0,
            label="Observed data",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["gp"],
            linewidth=1.8,
            label="Fixed-GP background mean",
        ),
        Patch(
            facecolor=COLORS["gp"],
            alpha=0.16,
            edgecolor="none",
            label="GP predictive uncertainty",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["standalone"],
            linewidth=1.8,
            linestyle=(0, (5, 2.5)),
            label="Standalone best-fit signal",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["shared"],
            linewidth=2.1,
            label=r"Shared-$\epsilon^2$ best-fit signal",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["blind"],
            linewidth=1.15,
            linestyle=(0, (4, 3)),
            label=r"Blind-window boundaries ($\pm2.25\sigma_m$)",
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
        "Observed 65 MeV signal extraction",
        y=0.985,
        fontsize=15.0,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.745)
    return save_pair(fig, "observed_extraction_m065_wide")


def pearson_residual(observed: np.ndarray, expectation: np.ndarray) -> np.ndarray:
    expectation = np.clip(np.asarray(expectation, float), 1.0, None)
    return (np.asarray(observed, float) - expectation) / np.sqrt(expectation)


def plot_profiled_residuals(result: JointResult) -> Dict[str, object]:
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
        null_expectation = np.asarray(model.standalone_null["lambda_hat"], float)
        standalone_expectation = np.asarray(
            model.standalone_fit["lambda_hat"], float
        )
        shared_expectation = np.asarray(
            result.shared_lambda_by_dataset[key], float
        )
        bin_width = 1e3 * float(np.median(np.diff(model.pred.edges_full)))
        offset = 0.16 * bin_width

        ax.axhspan(-2.0, 2.0, color="#DDE2E8", alpha=0.48, zorder=0)
        ax.axhline(0.0, color="#555B65", linewidth=0.9, zorder=1)
        ax.plot(
            x - offset,
            pearson_residual(obs, null_expectation),
            marker="o",
            markersize=3.6,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["gp"],
            markeredgewidth=0.9,
            label="Profiled background only",
            zorder=4,
        )
        ax.plot(
            x,
            pearson_residual(obs, standalone_expectation),
            marker="^",
            markersize=3.7,
            linestyle="none",
            color=COLORS["standalone"],
            label="Standalone signal + background",
            zorder=5,
        )
        ax.plot(
            x + offset,
            pearson_residual(obs, shared_expectation),
            marker="o",
            markersize=3.4,
            linestyle="none",
            color=COLORS["shared"],
            label=r"Shared-$\epsilon^2$ signal + background",
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
            float(
                np.nanmax(
                    np.abs(
                        np.concatenate(
                            [
                                pearson_residual(obs, null_expectation),
                                pearson_residual(obs, standalone_expectation),
                                pearson_residual(obs, shared_expectation),
                            ]
                        )
                    )
                )
            )
            * 1.12,
        )
        ax.set_ylim(-max_abs, max_abs)

    ax = axes[3]
    y_positions = np.arange(4, dtype=float)
    row_labels = ["2015", "2016", "2021 10%", "Shared fit"]
    estimates = [
        float(model.standalone_fit["A_hat"]) / model.k_events_per_eps2
        for model in result.models
    ] + [result.eps2_hat]
    uncertainties = [
        float(model.standalone_fit["sigma_A"]) / model.k_events_per_eps2
        for model in result.models
    ] + [result.sigma_eps2]
    point_colors = [COLORS[key] for key in DATASET_ORDER] + [COLORS["shared"]]
    ax.axvline(0.0, color="#555B65", linewidth=0.9)
    ax.axvspan(
        (result.eps2_hat - result.sigma_eps2) * 1e6,
        (result.eps2_hat + result.sigma_eps2) * 1e6,
        color=COLORS["shared"],
        alpha=0.12,
        zorder=0,
    )
    for y_pos, estimate, uncertainty, color in zip(
        y_positions, estimates, uncertainties, point_colors
    ):
        ax.errorbar(
            estimate * 1e6,
            y_pos,
            xerr=uncertainty * 1e6,
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
    ax.set_xlabel(r"Best-fit $\epsilon^2$  ($\times10^{-6}$)")
    ax.set_title(
        "Standalone and simultaneous signal estimates",
        fontweight="semibold",
    )
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MaxNLocator(7))

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
        "Profiled residual diagnostics at 65 MeV",
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
    return save_pair(fig, "observed_extraction_m065_profiled_residuals")


def write_tables(result: JointResult) -> Tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    fit_rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []
    for model in result.models:
        key = model.ds.key
        standalone_eps2 = (
            float(model.standalone_fit["A_hat"]) / model.k_events_per_eps2
        )
        standalone_sigma_eps2 = (
            float(model.standalone_fit["sigma_A"]) / model.k_events_per_eps2
        )
        fit_rows.append(
            {
                "dataset": key,
                "sample_label": LABELS[key],
                "mass_MeV": MASS_MEV,
                "sigma_mass_MeV": 1e3 * model.pred.sigma_val,
                "blind_lo_MeV": 1e3 * model.pred.blind[0],
                "blind_hi_MeV": 1e3 * model.pred.blind[1],
                "integral_density_counts_per_GeV": model.pred.integral_density,
                "K_events_per_eps2": model.k_events_per_eps2,
                "standalone_Ahat_events": float(model.standalone_fit["A_hat"]),
                "standalone_sigmaA_events": float(
                    model.standalone_fit["sigma_A"]
                ),
                "standalone_eps2_hat": standalone_eps2,
                "standalone_sigma_eps2": standalone_sigma_eps2,
                "standalone_p0_asymptotic": model.standalone_p0,
                "standalone_Z_asymptotic": model.standalone_Z,
                "shared_eps2_hat": result.eps2_hat,
                "shared_sigma_eps2": result.sigma_eps2,
                "shared_signal_yield_full": (
                    result.eps2_hat * model.k_events_per_eps2
                ),
                "shared_signal_yield_window": (
                    result.eps2_hat
                    * model.k_events_per_eps2
                    * float(np.sum(model.template_window))
                ),
                "joint_q0_block_contribution": (
                    result.q0_contribution_by_dataset[key]
                ),
                "const_fixed": model.pred.const_opt,
                "ls_fixed": model.pred.ls_opt,
                "lml_fixed_refit": model.pred.lml,
                "n_train": model.pred.n_train,
                "standalone_fit_success": bool(model.standalone_fit["success"]),
                "shared_fit_success": bool(result.fit["success"]),
            }
        )

        arrays = wide_display_arrays(result, model)
        full_indices = np.flatnonzero(arrays["mask"])
        blind_lookup = {
            int(full_index): blind_index
            for blind_index, full_index in enumerate(
                np.flatnonzero(model.pred.blind_mask)
            )
        }
        for local_index, full_index in enumerate(full_indices):
            blind_index = blind_lookup.get(int(full_index))
            row = {
                "dataset": key,
                "mass_hypothesis_MeV": MASS_MEV,
                "bin_center_MeV": float(arrays["x_MeV"][local_index]),
                "observed": float(arrays["observed"][local_index]),
                "gp_mean": float(arrays["gp_mean"][local_index]),
                "gp_predictive_sigma": float(
                    arrays["gp_predictive_sigma"][local_index]
                ),
                "display_total_sigma": float(arrays["gp_sigma"][local_index]),
                "data_minus_gp": float(
                    arrays["observed"][local_index]
                    - arrays["gp_mean"][local_index]
                ),
                "standalone_signal_display": float(
                    arrays["standalone_signal"][local_index]
                ),
                "shared_signal_display": float(
                    arrays["shared_signal"][local_index]
                ),
                "in_blind_window": blind_index is not None,
                "null_profiled_expectation": np.nan,
                "standalone_profiled_expectation": np.nan,
                "shared_profiled_expectation": np.nan,
                "null_profiled_pearson": np.nan,
                "standalone_profiled_pearson": np.nan,
                "shared_profiled_pearson": np.nan,
            }
            if blind_index is not None:
                obs_value = float(model.pred.obs[blind_index])
                null_value = float(
                    model.standalone_null["lambda_hat"][blind_index]
                )
                standalone_value = float(
                    model.standalone_fit["lambda_hat"][blind_index]
                )
                shared_value = float(
                    result.shared_lambda_by_dataset[key][blind_index]
                )
                row.update(
                    {
                        "null_profiled_expectation": null_value,
                        "standalone_profiled_expectation": standalone_value,
                        "shared_profiled_expectation": shared_value,
                        "null_profiled_pearson": float(
                            pearson_residual(
                                np.asarray([obs_value]),
                                np.asarray([null_value]),
                            )[0]
                        ),
                        "standalone_profiled_pearson": float(
                            pearson_residual(
                                np.asarray([obs_value]),
                                np.asarray([standalone_value]),
                            )[0]
                        ),
                        "shared_profiled_pearson": float(
                            pearson_residual(
                                np.asarray([obs_value]),
                                np.asarray([shared_value]),
                            )[0]
                        ),
                    }
                )
            bin_rows.append(row)

    fit_rows.append(
        {
            "dataset": "combined",
            "sample_label": "2015 100% + 2016 100% + 2021 10%",
            "mass_MeV": MASS_MEV,
            "shared_eps2_hat": result.eps2_hat,
            "shared_sigma_eps2": result.sigma_eps2,
            "shared_p0_asymptotic": result.p0,
            "shared_Z_asymptotic": result.Z,
            "shared_q0_asymptotic": result.q0,
            "shared_signal_yield_full": sum(
                result.eps2_hat * model.k_events_per_eps2
                for model in result.models
            ),
            "shared_signal_yield_window": sum(
                result.eps2_hat
                * model.k_events_per_eps2
                * float(np.sum(model.template_window))
                for model in result.models
            ),
            "shared_fit_success": bool(result.fit["success"]),
        }
    )
    fit_path = OUT / "observed_extraction_m065_fit_summary.csv"
    bin_path = OUT / "observed_extraction_m065_plot_data.csv"
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False, float_format="%.17g")
    pd.DataFrame(bin_rows).to_csv(bin_path, index=False, float_format="%.17g")
    return fit_path, bin_path


def write_provenance(
    result: JointResult,
    figures: Sequence[Mapping[str, object]],
    fit_path: Path,
    bin_path: Path,
) -> Path:
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mass_GeV": MASS_GEV,
        "datasets": list(DATASET_ORDER),
        "configuration": {
            "path": str(CONFIG.relative_to(REPO)),
            "sha256": sha256(CONFIG),
            "blind_nsigma": float(result.config.blind_nsigma),
            "gp_train_exclude_nsigma": float(
                result.config.gp_train_exclude_nsigma
            ),
            "eps2_density_nsigma": float(result.config.eps2_density_nsigma),
            "combined_mode": str(result.config.combined_mode),
            "eps2_lrt_scale": float(result.config.eps2_lrt_scale),
        },
        "reviewed_states": {
            "path": str(STATES.relative_to(REPO)),
            "sha256": sha256(STATES),
            "state_sha256_by_dataset": {
                model.ds.key: model.state_sha256 for model in result.models
            },
        },
        "enriched_validation_metrics": {
            "path": str(ENRICHED_STATES.relative_to(REPO)),
            "sha256": sha256(ENRICHED_STATES),
        },
        "combined_reviewed_result": {
            "path": str(COMBINED.relative_to(REPO)),
            "sha256": sha256(COMBINED),
            "p0_asymptotic": result.p0,
            "Z_asymptotic": result.Z,
            "q0_asymptotic": result.q0,
            "q0_block_contribution_by_dataset": (
                result.q0_contribution_by_dataset
            ),
            "eps2_hat": result.eps2_hat,
            "sigma_eps2": result.sigma_eps2,
        },
        "tables": [
            {
                "path": str(fit_path.relative_to(REPO)),
                "sha256": sha256(fit_path),
            },
            {
                "path": str(bin_path.relative_to(REPO)),
                "sha256": sha256(bin_path),
            },
        ],
        "figures": list(figures),
        "plot_semantics": {
            "wide_counts": (
                "Observed counts and pre-profile fixed-GP mean over plus-or-minus "
                "5 sigma; signal display curves add the standalone or shared "
                "profiled amplitude to that mean."
            ),
            "wide_background_subtracted": (
                "Observed counts minus the pre-profile fixed-GP mean; error bars "
                "use sqrt(GP mean + diagonal GP predictive covariance)."
            ),
            "profiled_residuals": (
                "Conditional Pearson residuals (observed - profiled expectation) "
                "/ sqrt(profiled expectation) in the exact plus-or-minus "
                "2.25-sigma likelihood window."
            ),
            "coefficient_comparison": (
                "Standalone Ahat/sigma_A converted to epsilon squared with each "
                "dataset K factor, compared with the one simultaneous shared-"
                "epsilon-squared fit."
            ),
        },
        "interpretation_boundary": (
            "Fixed-card local asymptotic extraction. The residuals are diagnostic "
            "and the result is not a scan-toy-calibrated global discovery claim."
        ),
    }
    path = OUT / "observed_extraction_m065_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    set_style()
    for path in (CONFIG, STATES, ENRICHED_STATES, COMBINED):
        if not path.exists():
            raise FileNotFoundError(path)
    config = load_config(str(CONFIG))
    states = pd.read_csv(STATES)
    enriched_states = pd.read_csv(ENRICHED_STATES)
    combined = pd.read_csv(COMBINED)
    combined_rows = combined[
        np.isclose(combined["mass_GeV"].to_numpy(float), MASS_GEV, atol=5e-10)
    ]
    if len(combined_rows) != 1:
        raise RuntimeError(
            f"Expected one combined result at {MASS_MEV:.0f} MeV; "
            f"found {len(combined_rows)}"
        )
    combined_row = combined_rows.iloc[0]
    reviewed_hashes = json.loads(
        str(combined_row["gp_state_sha256_by_dataset"])
    )
    if reviewed_hashes != EXPECTED_STATE_HASHES:
        raise RuntimeError(
            "The 65 MeV production state hashes do not match the expected "
            f"v4.2 triplet: {reviewed_hashes!r}"
        )
    datasets = make_datasets(config)
    models = [
        build_model(
            datasets[key],
            config,
            state_row(states, key),
            state_row(enriched_states, key),
        )
        for key in DATASET_ORDER
    ]
    result = build_joint(config, models, combined_row)
    figures = [
        plot_wide_extraction(result),
        plot_profiled_residuals(result),
    ]
    fit_path, bin_path = write_tables(result)
    provenance = write_provenance(
        result,
        figures,
        fit_path,
        bin_path,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "mass_MeV": MASS_MEV,
                "combined_eps2_hat": result.eps2_hat,
                "combined_sigma_eps2": result.sigma_eps2,
                "combined_p0": result.p0,
                "combined_Z": result.Z,
                "figures": figures,
                "fit_summary": str(fit_path.relative_to(REPO)),
                "plot_data": str(bin_path.relative_to(REPO)),
                "provenance": str(provenance.relative_to(REPO)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
