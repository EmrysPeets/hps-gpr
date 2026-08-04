#!/usr/bin/env python3
"""Compare the v4.1 factor-12 combined observed result with the v4 reference.

This postprocessor is deliberately observed-only.  It reads only the observed
limit and local asymptotic p0/Z columns from the historical v4 table, never
uses its expected-band or toy-tail columns, and produces no limit bands or
pseudoexperiments.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL = HERE / "final_k12_combined_observed"
DERIVED = HERE / "derived"
PLOTS = HERE / "plots"

NEW_CSV = FINAL / "combined_observed_fixed_reviewed.csv"
NEW_PROVENANCE = FINAL / "combined_observed_fixed_reviewed_provenance.json"
OLD_CSV = (
    REPO
    / "study_results"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "combined_bands_300toy_cached"
    / "ul_bands_combined_all.csv"
)
OLD_SUMMARY = (
    REPO
    / "study_results"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "derived"
    / "combined_bands300_summary.json"
)

COMPARISON_CSV = DERIVED / "combined_observed_k12_vs_v4.csv"
SUMMARY_JSON = DERIVED / "combined_observed_k12_summary.json"
MACROS_TEX = DERIVED / "combined_observed_k12_macros.tex"
MANIFEST_JSON = DERIVED / "combined_observed_k12_product_manifest.json"
CHECKSUMS = DERIVED / "combined_observed_k12_sha256sums.txt"

EXPECTED_MASS_MEV = np.arange(19, 251, dtype=int)
INDEPENDENCE_WIDTH_SIGMA = 2.25
M_MU_GEV = 0.1056583745
DIMUON_THRESHOLD_GEV = 2.0 * M_MU_GEV

COLORS = {
    "new": "#0B5FA5",
    "old": "#7A818B",
    "sidak": "#B2472D",
    "ratio": "#244C66",
    "zero": "#555B65",
}
DATASET_COLORS = {
    "2015": "#5B8FF9",
    "2015+2016": "#61DDAA",
    "2015+2016+2021": "#65789B",
    "2016+2021": "#F6BD16",
    "2021": "#E8684A",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dimuon_factor(mass_gev: np.ndarray) -> np.ndarray:
    masses = np.asarray(mass_gev, dtype=float)
    factor = np.ones_like(masses)
    above = masses > DIMUON_THRESHOLD_GEV
    if np.any(above):
        beta = np.sqrt(
            1.0 - 4.0 * M_MU_GEV**2 / masses[above] ** 2
        )
        factor[above] = 1.0 + beta * (
            1.0 + 2.0 * M_MU_GEV**2 / masses[above] ** 2
        )
    return factor


def effective_trials(mass_gev: np.ndarray, sigma_gev: np.ndarray) -> float:
    mass = np.asarray(mass_gev, dtype=float)
    sigma = np.asarray(sigma_gev, dtype=float)
    delta = np.diff(mass)
    sigma_mid = 0.5 * (sigma[:-1] + sigma[1:])
    if not bool(
        (
            np.isfinite(delta)
            & (delta > 0.0)
            & np.isfinite(sigma_mid)
            & (sigma_mid > 0.0)
        ).all()
    ):
        raise RuntimeError("Invalid resolution-spacing inputs.")
    return float(
        np.clip(
            np.sum(delta / (INDEPENDENCE_WIDTH_SIGMA * sigma_mid)),
            1.0,
            float(mass.size),
        )
    )


def load_and_validate() -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in (NEW_CSV, NEW_PROVENANCE, OLD_CSV, OLD_SUMMARY):
        if not path.is_file():
            raise RuntimeError(f"Required source is missing: {path}")

    new = pd.read_csv(NEW_CSV)
    old_source = pd.read_csv(OLD_CSV)
    old_columns = [
        "dataset_set",
        "mass_GeV",
        "sigma_mass_res_GeV",
        "sigma_mass_res_min_GeV",
        "eps2_obs",
        "p0_analytic",
        "Z_analytic",
    ]
    missing_old = sorted(set(old_columns).difference(old_source.columns))
    if missing_old:
        raise RuntimeError(f"v4 table is missing observed columns: {missing_old}")
    old = old_source[old_columns].copy()

    required_new = set(old_columns).union(
        {
            "mass_MeV",
            "q0_analytic",
            "p0_fit_ok",
            "toy_draws",
            "expected_bands_produced",
            "cls_alpha",
            "cls_calibration",
            "combined_mode",
        }
    )
    missing_new = sorted(required_new.difference(new.columns))
    if missing_new:
        raise RuntimeError(f"v4.1 table is missing columns: {missing_new}")
    if len(new) != 232 or len(old) != 232:
        raise RuntimeError(f"Expected 232 rows, found new={len(new)}, old={len(old)}.")

    expected_gev = EXPECTED_MASS_MEV.astype(float) / 1000.0
    if not np.array_equal(new["mass_MeV"].to_numpy(int), EXPECTED_MASS_MEV):
        raise RuntimeError("v4.1 mass grid is not the exact 19--250 MeV grid.")
    if not np.array_equal(new["mass_GeV"].to_numpy(float), expected_gev):
        raise RuntimeError("v4.1 GeV mass grid is not exact.")
    if not np.array_equal(old["mass_GeV"].to_numpy(float), expected_gev):
        raise RuntimeError("v4 reference GeV mass grid is not exact.")
    if bool((new["toy_draws"].to_numpy(int) != 0).any()):
        raise RuntimeError("v4.1 table records nonzero toy draws.")
    if bool(new["expected_bands_produced"].astype(bool).any()):
        raise RuntimeError("v4.1 table records expected bands.")
    if set(new["cls_calibration"].astype(str)) != {"asymptotic"}:
        raise RuntimeError("v4.1 output is not uniformly asymptotic.")
    if set(new["combined_mode"].astype(str)) != {"count_scale"}:
        raise RuntimeError("v4.1 output is not uniformly count_scale.")
    if not bool(np.isclose(new["cls_alpha"], 0.1, rtol=0.0, atol=0.0).all()):
        raise RuntimeError("v4.1 output is not uniformly 90% CL.")
    if not bool(new["p0_fit_ok"].astype(bool).all()):
        raise RuntimeError("A v4.1 local asymptotic p0 fit is not marked successful.")

    numeric_new = [
        "eps2_obs",
        "p0_analytic",
        "Z_analytic",
        "q0_analytic",
        "sigma_mass_res_min_GeV",
    ]
    numeric_old = [
        "eps2_obs",
        "p0_analytic",
        "Z_analytic",
        "sigma_mass_res_min_GeV",
    ]
    for key in numeric_new:
        if not bool(np.isfinite(new[key].to_numpy(float)).all()):
            raise RuntimeError(f"Non-finite v4.1 values in {key}.")
    for key in numeric_old:
        if not bool(np.isfinite(old[key].to_numpy(float)).all()):
            raise RuntimeError(f"Non-finite v4 values in {key}.")

    comparison = pd.DataFrame(
        {
            "mass_MeV": EXPECTED_MASS_MEV,
            "mass_GeV": expected_gev,
            "dataset_set": new["dataset_set"].astype(str),
            "includes_2016": new["dataset_set"]
            .astype(str)
            .str.split("+")
            .map(lambda values: "2016" in values),
            "sigma_mass_res_min_GeV": new[
                "sigma_mass_res_min_GeV"
            ].to_numpy(float),
            "v4_k8_eps2_obs_ee": old["eps2_obs"].to_numpy(float),
            "v4p1_k12_eps2_obs_ee": new["eps2_obs"].to_numpy(float),
            "v4_k8_p0_local_asymptotic": old["p0_analytic"].to_numpy(float),
            "v4p1_k12_p0_local_asymptotic": new[
                "p0_analytic"
            ].to_numpy(float),
            "v4_k8_Z_local_asymptotic": old["Z_analytic"].to_numpy(float),
            "v4p1_k12_Z_local_asymptotic": new[
                "Z_analytic"
            ].to_numpy(float),
        }
    )
    comparison["minimal_visible_factor"] = dimuon_factor(expected_gev)
    for prefix in ("v4_k8", "v4p1_k12"):
        comparison[f"{prefix}_eps2_obs_minimal_visible"] = (
            comparison[f"{prefix}_eps2_obs_ee"].to_numpy(float)
            * comparison["minimal_visible_factor"].to_numpy(float)
        )
    comparison["observed_limit_ratio_k12_over_k8"] = (
        comparison["v4p1_k12_eps2_obs_ee"].to_numpy(float)
        / comparison["v4_k8_eps2_obs_ee"].to_numpy(float)
    )
    comparison["observed_limit_fractional_change"] = (
        comparison["observed_limit_ratio_k12_over_k8"].to_numpy(float) - 1.0
    )
    comparison["delta_Z_k12_minus_k8"] = (
        comparison["v4p1_k12_Z_local_asymptotic"].to_numpy(float)
        - comparison["v4_k8_Z_local_asymptotic"].to_numpy(float)
    )
    comparison["p0_ratio_k12_over_k8"] = (
        comparison["v4p1_k12_p0_local_asymptotic"].to_numpy(float)
        / comparison["v4_k8_p0_local_asymptotic"].to_numpy(float)
    )

    unchanged = ~comparison["includes_2016"].to_numpy(bool)
    for key in (
        "observed_limit_ratio_k12_over_k8",
        "delta_Z_k12_minus_k8",
    ):
        values = comparison.loc[unchanged, key].to_numpy(float)
        target = 1.0 if "ratio" in key else 0.0
        if not bool(np.array_equal(values, np.full_like(values, target))):
            raise RuntimeError(
                f"Rows without 2016 are not bitwise unchanged for {key}."
            )

    provenance = json.loads(NEW_PROVENANCE.read_text(encoding="utf-8"))
    validation = provenance.get("output_validation", {})
    if validation.get("n_rows") != 232:
        raise RuntimeError("v4.1 provenance does not validate 232 rows.")
    if validation.get("toy_draws") != 0:
        raise RuntimeError("v4.1 provenance records toy draws.")
    if validation.get("expected_bands_produced") is not False:
        raise RuntimeError("v4.1 provenance records expected bands.")
    closures = validation.get("reference_closure", [])
    if len(closures) != 3 or not all(
        bool(entry.get("bitwise_equal")) for entry in closures
    ):
        raise RuntimeError("Reference-solver bitwise closure is incomplete.")

    metadata = {
        "new_provenance": provenance,
        "old_source_columns_used": old_columns,
        "old_source_columns_explicitly_ignored": sorted(
            set(old_source.columns).difference(old_columns)
        ),
    }
    return comparison, metadata


def build_summary(
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    affected = comparison.loc[comparison["includes_2016"].astype(bool)].copy()
    ratio = affected["observed_limit_ratio_k12_over_k8"].to_numpy(float)
    delta_z = affected["delta_Z_k12_minus_k8"].to_numpy(float)
    new_p = comparison["v4p1_k12_p0_local_asymptotic"].to_numpy(float)
    old_p = comparison["v4_k8_p0_local_asymptotic"].to_numpy(float)

    new_min_index = int(np.argmin(new_p))
    old_min_index = int(np.argmin(old_p))
    tight_index = int(np.argmin(ratio))
    loose_index = int(np.argmax(ratio))
    affected_indices = affected.index.to_numpy(int)
    tight_global = int(affected_indices[tight_index])
    loose_global = int(affected_indices[loose_index])

    neff = effective_trials(
        comparison["mass_GeV"].to_numpy(float),
        comparison["sigma_mass_res_min_GeV"].to_numpy(float),
    )
    new_p_min = float(new_p[new_min_index])
    p_sidak = float(
        np.clip(
            -np.expm1(neff * np.log1p(-new_p_min)),
            1.0e-300,
            1.0,
        )
    )
    z_sidak = float(norm.isf(p_sidak))

    by_active_set = []
    for label, group in comparison.groupby("dataset_set", sort=False):
        values = group["observed_limit_ratio_k12_over_k8"].to_numpy(float)
        dz_values = group["delta_Z_k12_minus_k8"].to_numpy(float)
        by_active_set.append(
            {
                "dataset_set": str(label),
                "rows": int(len(group)),
                "ratio_min": float(np.min(values)),
                "ratio_median": float(np.median(values)),
                "ratio_max": float(np.max(values)),
                "delta_Z_min": float(np.min(dz_values)),
                "delta_Z_median": float(np.median(dz_values)),
                "delta_Z_max": float(np.max(dz_values)),
            }
        )

    summary = {
        "study": "v4p1_2016_k12_combined_observed_only",
        "status": "post_v4_observed_asymptotic_diagnostic_candidate",
        "controlled_change": {
            "dataset": "2016 full statistics",
            "kernel_length_scale_upper_factor_before": 8,
            "kernel_length_scale_upper_factor_after": 12,
            "other_parsed_physics_and_statistics_settings_changed": False,
        },
        "grid": {
            "mass_min_MeV": 19,
            "mass_max_MeV": 250,
            "rows": 232,
            "rows_with_2016_active": int(len(affected)),
            "rows_without_2016_active_bitwise_unchanged": int(
                len(comparison) - len(affected)
            ),
        },
        "observed_limit_change_on_2016_active_masses": {
            "ratio_definition": "v4p1_k12 / v4_k8",
            "ratio_min": float(np.min(ratio)),
            "ratio_p05": float(np.quantile(ratio, 0.05)),
            "ratio_median": float(np.median(ratio)),
            "ratio_p95": float(np.quantile(ratio, 0.95)),
            "ratio_max": float(np.max(ratio)),
            "tighter_rows": int(np.sum(ratio < 1.0)),
            "looser_rows": int(np.sum(ratio > 1.0)),
            "unchanged_rows_exact": int(np.sum(ratio == 1.0)),
            "largest_tightening_fraction": float(1.0 - np.min(ratio)),
            "largest_tightening_mass_MeV": int(
                comparison.iloc[tight_global]["mass_MeV"]
            ),
            "largest_loosening_fraction": float(np.max(ratio) - 1.0),
            "largest_loosening_mass_MeV": int(
                comparison.iloc[loose_global]["mass_MeV"]
            ),
        },
        "local_asymptotic_search": {
            "v4_k8_minimum": {
                "mass_MeV": int(comparison.iloc[old_min_index]["mass_MeV"]),
                "p0": float(old_p[old_min_index]),
                "Z": float(
                    comparison.iloc[old_min_index][
                        "v4_k8_Z_local_asymptotic"
                    ]
                ),
            },
            "v4p1_k12_minimum": {
                "mass_MeV": int(comparison.iloc[new_min_index]["mass_MeV"]),
                "p0": new_p_min,
                "Z": float(
                    comparison.iloc[new_min_index][
                        "v4p1_k12_Z_local_asymptotic"
                    ]
                ),
                "v4_k8_p0_at_same_mass": float(old_p[new_min_index]),
                "v4_k8_Z_at_same_mass": float(
                    comparison.iloc[new_min_index][
                        "v4_k8_Z_local_asymptotic"
                    ]
                ),
                "delta_Z_at_same_mass": float(delta_z[new_min_index - 20]),
            },
            "maximum_absolute_delta_Z_on_2016_active_masses": float(
                np.max(np.abs(delta_z))
            ),
            "maximum_absolute_delta_Z_mass_MeV": int(
                affected.iloc[int(np.argmax(np.abs(delta_z)))]["mass_MeV"]
            ),
            "analytic_sidak_reference_at_new_local_minimum": {
                "N_eff_resolution_spacing": neff,
                "independence_width_sigma": INDEPENDENCE_WIDTH_SIGMA,
                "p": p_sidak,
                "Z": z_sidak,
                "scan_toy_calibrated": False,
            },
        },
        "active_set_summary": by_active_set,
        "validation": {
            "new_rows_finite": 232,
            "reference_solver_bitwise_closure_masses_MeV": [
                int(entry["mass_MeV"])
                for entry in metadata["new_provenance"]["output_validation"][
                    "reference_closure"
                ]
            ],
            "reference_solver_bitwise_closure_passed": True,
            "toys_drawn": 0,
            "expected_limit_bands_produced": False,
            "old_v4_expected_band_columns_used": False,
            "old_v4_toy_tail_columns_used": False,
        },
        "interpretation": {
            "factor_selection_used_limits_or_pvalues": False,
            "selection_basis": (
                "first tested factor with zero upper-bound occupancy and "
                "stable nested-LML/limit plateau at factors 15 and 20"
            ),
            "post_observation_hyperparameter_change": True,
            "discovery_or_excess_claim_authorized": False,
            "coverage_calibrated": False,
            "analytic_sidak_is_not_scan_toy_calibration": True,
        },
        "sources": {
            "v4p1_observed_csv": {
                "path": repo_path(NEW_CSV),
                "sha256": sha256(NEW_CSV),
            },
            "v4p1_provenance": {
                "path": repo_path(NEW_PROVENANCE),
                "sha256": sha256(NEW_PROVENANCE),
            },
            "v4_observed_columns_source": {
                "path": repo_path(OLD_CSV),
                "sha256": sha256(OLD_CSV),
                "columns_used": metadata["old_source_columns_used"],
                "band_and_toy_columns_ignored": True,
            },
        },
        "products_excluded": {
            "expected_limit_bands": True,
            "new_pseudoexperiments": True,
            "toy_calibrated_local_pvalues": True,
            "toy_calibrated_global_pvalues": True,
        },
    }
    return summary


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.3,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_mass_ticks(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(19, 250)
    ax.set_xticks([20, 50, 75, 100, 125, 150, 175, 200, 225, 250])
    ax.set_xticks(np.arange(25, 251, 25), minor=True)


def activity_segments(table: pd.DataFrame) -> list[tuple[int, int, str]]:
    labels = table["dataset_set"].astype(str).tolist()
    masses = table["mass_MeV"].to_numpy(int)
    segments: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.append((int(masses[start]), int(masses[index - 1]), labels[start]))
            start = index
    return segments


def plot_activity_strip(ax: mpl.axes.Axes, table: pd.DataFrame) -> None:
    for lo, hi, label in activity_segments(table):
        ax.axvspan(
            lo - 0.5,
            hi + 0.5,
            color=DATASET_COLORS[label],
            alpha=0.9,
            linewidth=0.0,
        )
        if hi - lo >= 8:
            ax.text(
                0.5 * (lo + hi),
                0.5,
                label.replace("+", " + "),
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if label == "2015+2016+2021" else "#20242A",
            )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig: mpl.figure.Figure, stem: str) -> list[Path]:
    outputs = [PLOTS / f"{stem}.pdf", PLOTS / f"{stem}.png"]
    for path in outputs:
        fig.savefig(
            path,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "postprocess_combined_observed.py"},
        )
    plt.close(fig)
    return outputs


def plot_limit_comparison(table: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(11.8, 7.6))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.12, 1.0, 0.42),
        hspace=0.08,
        left=0.10,
        right=0.98,
        top=0.88,
        bottom=0.10,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    ratio_ax = fig.add_subplot(grid[2], sharex=activity)
    plot_activity_strip(activity, table)
    x = table["mass_MeV"].to_numpy(float)
    old = table["v4_k8_eps2_obs_minimal_visible"].to_numpy(float)
    new = table["v4p1_k12_eps2_obs_minimal_visible"].to_numpy(float)
    ratio = table["observed_limit_ratio_k12_over_k8"].to_numpy(float)

    ax.plot(
        x,
        old,
        color=COLORS["old"],
        linestyle="--",
        linewidth=1.55,
        label=r"v4: 2016 $\ell_{\max}=8\sigma_m$",
    )
    ax.plot(
        x,
        new,
        color=COLORS["new"],
        linewidth=1.8,
        label=r"v4.1 candidate: 2016 $\ell_{\max}=12\sigma_m$",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"Observed 90% CL$_s$ upper limit on minimal-visible $\epsilon^2$")
    ax.grid(which="major", axis="y", color="#D9DDE3", linewidth=0.65)
    ax.legend(loc="upper right", frameon=False)
    ax.tick_params(axis="x", labelbottom=False)

    ratio_ax.axhline(1.0, color=COLORS["zero"], linewidth=0.8)
    ratio_ax.plot(x, ratio, color=COLORS["ratio"], linewidth=1.55)
    ratio_ax.fill_between(
        x,
        ratio,
        1.0,
        where=table["includes_2016"].to_numpy(bool),
        color=COLORS["ratio"],
        alpha=0.10,
        linewidth=0.0,
    )
    ratio_ax.set_ylabel("v4.1 / v4")
    ratio_ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ratio_ax.set_ylim(0.62, 1.36)
    ratio_ax.grid(which="major", axis="y", color="#D9DDE3", linewidth=0.65)
    set_mass_ticks(ratio_ax)
    fig.suptitle(
        "Combined observed limit after lifting the 2016 length-scale upper bound",
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.022,
        (
            "Observed/asymptotic comparison only. No expected-limit bands or "
            "pseudoexperiments are shown."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_observed_limit_k12_vs_v4_no_bands",
    )


def plot_pvalue_comparison(
    table: pd.DataFrame,
    summary: dict[str, Any],
) -> list[Path]:
    fig = plt.figure(figsize=(11.8, 7.6))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.12, 1.0, 0.42),
        hspace=0.08,
        left=0.10,
        right=0.98,
        top=0.86,
        bottom=0.10,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    delta_ax = fig.add_subplot(grid[2], sharex=activity)
    plot_activity_strip(activity, table)
    x = table["mass_MeV"].to_numpy(float)
    old = table["v4_k8_p0_local_asymptotic"].to_numpy(float)
    new = table["v4p1_k12_p0_local_asymptotic"].to_numpy(float)
    delta_z = table["delta_Z_k12_minus_k8"].to_numpy(float)

    sidak_info = summary["local_asymptotic_search"][
        "analytic_sidak_reference_at_new_local_minimum"
    ]
    neff = float(sidak_info["N_eff_resolution_spacing"])
    sidak = -np.expm1(neff * np.log1p(-np.clip(new, 1.0e-300, 1.0)))
    sidak = np.clip(sidak, 1.0e-300, 1.0)
    ax.plot(
        x,
        old,
        color=COLORS["old"],
        linestyle="--",
        linewidth=1.45,
        label=r"v4 local asymptotic $p_0$",
    )
    ax.plot(
        x,
        new,
        color=COLORS["new"],
        linewidth=1.8,
        label=r"v4.1 candidate local asymptotic $p_0$",
    )
    ax.plot(
        x,
        sidak,
        color=COLORS["sidak"],
        linestyle=":",
        linewidth=1.5,
        label=rf"v4.1 analytic Šidák reference ($N_{{\rm eff}}={neff:.2f}$)",
    )
    minimum = summary["local_asymptotic_search"]["v4p1_k12_minimum"]
    ax.scatter(
        [minimum["mass_MeV"]],
        [minimum["p0"]],
        s=42,
        color=COLORS["new"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.set_yscale("log")
    positive = np.concatenate([old[old > 0.0], new[new > 0.0], sidak[sidak > 0.0]])
    lower = max(
        1.0e-8,
        10.0 ** math.floor(math.log10(float(np.min(positive)))) / 2.0,
    )
    ax.set_ylim(lower, 1.08)
    ax.set_ylabel("One-sided p-value")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", axis="y", color="#D9DDE3", linewidth=0.65)
    ax.legend(loc="lower right", frameon=False)
    ax.tick_params(axis="x", labelbottom=False)

    delta_ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8)
    delta_ax.plot(x, delta_z, color=COLORS["ratio"], linewidth=1.5)
    delta_ax.fill_between(
        x,
        delta_z,
        0.0,
        where=table["includes_2016"].to_numpy(bool),
        color=COLORS["ratio"],
        alpha=0.10,
        linewidth=0.0,
    )
    delta_ax.set_ylabel(r"$\Delta Z_{\rm local}$")
    delta_ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    delta_ax.set_ylim(-1.0, 0.8)
    delta_ax.grid(which="major", axis="y", color="#D9DDE3", linewidth=0.65)
    set_mass_ticks(delta_ax)
    fig.suptitle(
        "Combined local asymptotic p-values after the 2016 bound change",
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.022,
        (
            "The Šidák curve is an analytic resolution-spacing reference, not "
            "scan-toy calibration. The post-observation change is diagnostic."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_asymptotic_p0_k12_vs_v4",
    )


def write_macros(summary: dict[str, Any]) -> None:
    limit = summary["observed_limit_change_on_2016_active_masses"]
    search = summary["local_asymptotic_search"]
    old = search["v4_k8_minimum"]
    new = search["v4p1_k12_minimum"]
    sidak = search["analytic_sidak_reference_at_new_local_minimum"]
    lines = [
        "% Auto-generated by postprocess_combined_observed.py.",
        r"\newcommand{\VFourPOldMin}{%.6g}" % old["p0"],
        r"\newcommand{\VFourZOldMin}{%.3f}" % old["Z"],
        r"\newcommand{\VFourMassOldMin}{%d}" % old["mass_MeV"],
        r"\newcommand{\VFourPOneMin}{%.6g}" % new["p0"],
        r"\newcommand{\VFourZOneMin}{%.3f}" % new["Z"],
        r"\newcommand{\VFourMassOneMin}{%d}" % new["mass_MeV"],
        r"\newcommand{\VFourOneSidakP}{%.6g}" % sidak["p"],
        r"\newcommand{\VFourOneSidakZ}{%.3f}" % sidak["Z"],
        r"\newcommand{\VFourOneNeff}{%.2f}"
        % sidak["N_eff_resolution_spacing"],
        r"\newcommand{\VFourOneLimitRatioMedian}{%.3f}"
        % limit["ratio_median"],
        r"\newcommand{\VFourOneLimitRatioMin}{%.3f}" % limit["ratio_min"],
        r"\newcommand{\VFourOneLimitRatioMax}{%.3f}" % limit["ratio_max"],
        r"\newcommand{\VFourOneTighterRows}{%d}" % limit["tighter_rows"],
        r"\newcommand{\VFourOneLooserRows}{%d}" % limit["looser_rows"],
        "",
    ]
    MACROS_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(products: list[Path]) -> None:
    entries = []
    for path in products:
        entries.append(
            {
                "path": repo_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(
        MANIFEST_JSON,
        {
            "study": "v4p1_2016_k12_combined_observed_only",
            "products": entries,
            "toys_drawn": 0,
            "expected_limit_bands_produced": False,
        },
    )
    checksum_paths = [*products, MANIFEST_JSON]
    CHECKSUMS.write_text(
        "".join(
            f"{sha256(path)}  {repo_path(path)}\n"
            for path in sorted(checksum_paths)
        ),
        encoding="utf-8",
    )


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    comparison, metadata = load_and_validate()
    summary = build_summary(comparison, metadata)
    comparison.to_csv(COMPARISON_CSV, index=False)
    write_json(SUMMARY_JSON, summary)
    write_macros(summary)

    configure_plotting()
    plot_paths = [
        *plot_limit_comparison(comparison),
        *plot_pvalue_comparison(comparison, summary),
    ]
    products = [COMPARISON_CSV, SUMMARY_JSON, MACROS_TEX, *plot_paths]
    write_manifest(products)
    print(
        "Validated and compared 232 observed-only masses; "
        "wrote two PDF/PNG figure pairs. toys=0, expected bands=false."
    )


if __name__ == "__main__":
    main()
