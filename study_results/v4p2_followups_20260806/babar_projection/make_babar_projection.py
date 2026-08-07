#!/usr/bin/env python3
"""Build the reviewed v4.2 HPS--BaBar observed-equivalent comparison.

The source HPS curve is the exact reviewed v4.2 minimal-visible observed
90% asymptotic CLs result.  The 2021 10% sample is projected to the declared
100%-statistics exposure with the repository's observed-density proxy:

    scale(m) = sqrt(sum_d rho_d / sum_d f_d rho_d),
    (f_2015, f_2016, f_2021) = (1, 1, 10).

This is deliberately an observed-equivalent response projection.  It is not an
expected sensitivity, a future observed limit, or a projection of discovery
p-values.  No expected-limit bands or projected p-values are produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = HERE / "inputs"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

HPS_REL = Path(
    "study_results/"
    "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "derived/combined_bands300_reviewed_v4p2.csv"
)
CONFIG_REL = Path(
    "study_configs/"
    "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
NOTE_REL = Path(
    "hps_gpr_analysis_note/HPS_GPR_Analysis_Note_v4p2_20260805.pdf"
)
HPS_SOURCE = REPO / HPS_REL
CONFIG_SOURCE = REPO / CONFIG_REL
NOTE_SOURCE = REPO / NOTE_REL

FROZEN_BABAR_NAME = "BaBar_Lees2014xha.txt"
FROZEN_BABAR = INPUTS / FROZEN_BABAR_NAME
DEFAULT_EXTERNAL_BABAR = Path(
    "/Users/emryspeets/Desktop/Stanford/2026_winter/"
    "BaBar_Lees2014xha.txt"
)

EXPECTED_V4P2_COMMIT = "fb1295680bacdd5edbabff9546ee200e3c68b78a"
EXPECTED_HPS_SHA256 = (
    "8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd"
)
EXPECTED_CONFIG_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
EXPECTED_NOTE_SHA256 = (
    "8b47486ea2cb71d83e4aea31bbe8671d89aa874670857cb29cfb59f57725ed52"
)
EXPECTED_BABAR_SHA256 = (
    "5b03037c27f248126830114229300f938d89c1509b47eae0088c55bb0b0a2778"
)

DIMUON_THRESHOLD_GEV = 0.211316749
FULL_EXPOSURE_FACTORS = {"2015": 1.0, "2016": 1.0, "2021": 10.0}

OVERLAY_STEM = "v4p2_babar_observed_equivalent_projection_eps2"
RATIO_STEM = "v4p2_babar_observed_equivalent_projection_ratio"
OVERLAY_PROJECTED_OVER_BABAR_STEM = (
    "v4p2_babar_observed_equivalent_projection_eps2_"
    "with_projected_over_babar_ratio"
)

COLOR_BABAR = "#C98200"
COLOR_CURRENT = "#6F7780"
COLOR_PROJECTED = "#164A7B"
COLOR_THRESHOLD = "#8B929A"
COLOR_GRID = "#D9DEE3"
COLOR_MINOR_GRID = "#EDF0F2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prepare_directories() -> None:
    for path in (INPUTS, DERIVED, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def freeze_babar_source() -> Path:
    """Return a verified bundle-local copy of the raw BaBar contour."""
    if not FROZEN_BABAR.is_file():
        source = Path(
            os.environ.get(
                "HPS_BABAR_VISIBLE_LIMIT_TABLE",
                str(DEFAULT_EXTERNAL_BABAR),
            )
        ).expanduser()
        require(source.is_file(), f"Missing BaBar source: {source}")
        require(
            sha256(source) == EXPECTED_BABAR_SHA256,
            "External BaBar source does not match the frozen checksum",
        )
        shutil.copyfile(source, FROZEN_BABAR)
    require(
        sha256(FROZEN_BABAR) == EXPECTED_BABAR_SHA256,
        "Bundle-local BaBar source does not match the frozen checksum",
    )
    return FROZEN_BABAR


def load_babar(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        names=["mass_GeV", "epsilon_90"],
    )
    valid = (
        np.isfinite(raw["mass_GeV"])
        & np.isfinite(raw["epsilon_90"])
        & (raw["epsilon_90"] > 0.0)
        & (raw["epsilon_90"] < 1.0)
    )
    out = raw.loc[valid].copy()
    out["mass_MeV"] = 1000.0 * out["mass_GeV"]
    out["eps2_90"] = np.square(out["epsilon_90"])
    out = (
        out.sort_values("mass_MeV")
        .drop_duplicates("mass_MeV")
        .reset_index(drop=True)
    )
    require(len(out) == 5639, f"Expected 5639 valid BaBar rows, found {len(out)}")
    require(
        out["mass_MeV"].is_monotonic_increasing,
        "BaBar mass grid is not increasing",
    )
    require(
        np.all(np.isfinite(out["eps2_90"]) & (out["eps2_90"] > 0.0)),
        "BaBar epsilon-squared contour contains invalid values",
    )
    return out


def load_v4p2() -> pd.DataFrame:
    require(HPS_SOURCE.is_file(), f"Missing reviewed v4.2 table: {HPS_SOURCE}")
    require(
        sha256(HPS_SOURCE) == EXPECTED_HPS_SHA256,
        "Reviewed v4.2 source table checksum changed",
    )
    require(CONFIG_SOURCE.is_file(), f"Missing v4.2 card: {CONFIG_SOURCE}")
    require(
        sha256(CONFIG_SOURCE) == EXPECTED_CONFIG_SHA256,
        "Reviewed v4.2 card checksum changed",
    )
    require(NOTE_SOURCE.is_file(), f"Missing v4.2 note: {NOTE_SOURCE}")
    require(
        sha256(NOTE_SOURCE) == EXPECTED_NOTE_SHA256,
        "Reviewed v4.2 note checksum changed",
    )

    out = pd.read_csv(HPS_SOURCE)
    required = {
        "mass_GeV",
        "mass_MeV",
        "dataset_set",
        "cls_alpha",
        "cls_statistic",
        "cls_calibration",
        "combined_mode",
        "eps2_obs_ee_channel",
        "eps2_obs_minimal_visible",
        "N_eff_BR",
        "BR_ee_minimal",
        "dimuon_correction_applied",
        "meta",
        "gp_state_sha256_by_dataset",
        "fixed_reviewed_state_metadata_validated",
    }
    missing = sorted(required - set(out.columns))
    require(not missing, f"Reviewed v4.2 table lacks columns: {missing}")
    require(len(out) == 232, f"Expected 232 HPS rows, found {len(out)}")
    require(
        np.array_equal(out["mass_MeV"].to_numpy(int), np.arange(19, 251)),
        "Expected the reviewed 19--250 MeV integer grid",
    )
    require(
        np.allclose(out["cls_alpha"], 0.1, rtol=0.0, atol=1.0e-15),
        "The source is not a 90% CLs result",
    )
    require(
        set(out["cls_statistic"].astype(str)) == {"tilde_q_mu"},
        "Unexpected CLs statistic",
    )
    require(
        set(out["cls_calibration"].astype(str)) == {"asymptotic"},
        "Unexpected CLs calibration",
    )
    require(
        set(out["combined_mode"].astype(str)) == {"count_scale"},
        "Unexpected combined-likelihood coordinate",
    )
    require(
        out["fixed_reviewed_state_metadata_validated"].astype(bool).all(),
        "Not every v4.2 row has validated reviewed-state metadata",
    )
    require(
        np.allclose(
            out["eps2_obs_minimal_visible"],
            out["eps2_obs_ee_channel"] * out["N_eff_BR"],
            rtol=2.0e-13,
            atol=0.0,
        ),
        "Minimal-visible conversion does not close",
    )
    return out


def parse_density_meta(meta: str, dataset_set: str) -> dict[str, float]:
    entries = json.loads(meta)
    require(isinstance(entries, list), "v4.2 metadata is not a list")
    densities: dict[str, float] = {}
    for entry in entries:
        key = str(entry["key"])
        value = float(entry["dens"])
        require(key not in densities, f"Duplicate density key {key}")
        require(np.isfinite(value) and value > 0.0, f"Invalid density for {key}")
        densities[key] = value
    expected = dataset_set.split("+")
    require(
        list(densities) == expected,
        f"Active datasets and density metadata disagree: {dataset_set}",
    )
    return densities


def log_interpolate(
    x: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
) -> np.ndarray:
    """Interpolate a positive contour in log(y), without extrapolation."""
    result = np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)
    inside = (x >= source_x.min()) & (x <= source_x.max())
    result[inside] = np.exp(
        np.interp(x[inside], source_x, np.log(source_y))
    )
    return result


def build_reviewed_projection(
    hps: pd.DataFrame,
    babar: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in hps.iterrows():
        dataset_set = str(source["dataset_set"])
        densities = parse_density_meta(str(source["meta"]), dataset_set)
        current_density = float(sum(densities.values()))
        projected_density = float(
            sum(
                FULL_EXPOSURE_FACTORS[key] * value
                for key, value in densities.items()
            )
        )
        projection_scale = math.sqrt(current_density / projected_density)
        observed = float(source["eps2_obs_minimal_visible"])
        rows.append(
            {
                "mass_GeV": float(source["mass_GeV"]),
                "mass_MeV": int(source["mass_MeV"]),
                "dataset_set": dataset_set,
                "n_active_datasets": len(densities),
                "hps_v4p2_eps2_obs_ee_channel": float(
                    source["eps2_obs_ee_channel"]
                ),
                "N_eff_BR": float(source["N_eff_BR"]),
                "BR_ee_minimal": float(source["BR_ee_minimal"]),
                "dimuon_correction_applied": bool(
                    source["dimuon_correction_applied"]
                ),
                "hps_v4p2_eps2_obs_minimal_visible": observed,
                "density_2015_counts_per_GeV": densities.get("2015", np.nan),
                "density_2016_counts_per_GeV": densities.get("2016", np.nan),
                "density_2021_10pct_counts_per_GeV": densities.get(
                    "2021", np.nan
                ),
                "density_current_counts_per_GeV": current_density,
                "density_2021_100pct_equivalent_counts_per_GeV": (
                    projected_density
                ),
                "full2021_projection_scale_eps2": projection_scale,
                "hps_v4p2_projected_full2021_eps2_minimal_visible": (
                    observed * projection_scale
                ),
                "gp_state_sha256_by_dataset": str(
                    source["gp_state_sha256_by_dataset"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["babar_visible2014_eps2_90_log_interp"] = log_interpolate(
        out["mass_MeV"].to_numpy(float),
        babar["mass_MeV"].to_numpy(float),
        babar["eps2_90"].to_numpy(float),
    )
    out["hps_v4p2_observed_over_babar"] = (
        out["hps_v4p2_eps2_obs_minimal_visible"]
        / out["babar_visible2014_eps2_90_log_interp"]
    )
    out["hps_v4p2_projected_full2021_over_babar"] = (
        out["hps_v4p2_projected_full2021_eps2_minimal_visible"]
        / out["babar_visible2014_eps2_90_log_interp"]
    )
    out["hps_v4p2_observed_below_babar_on_grid"] = (
        out["hps_v4p2_observed_over_babar"] < 1.0
    )
    out["hps_v4p2_projected_full2021_below_babar_on_grid"] = (
        out["hps_v4p2_projected_full2021_over_babar"] < 1.0
    )

    below_2021 = out["mass_MeV"] < 50
    only_2021 = out["mass_MeV"] > 180
    require(
        np.allclose(
            out.loc[below_2021, "full2021_projection_scale_eps2"],
            1.0,
            rtol=0.0,
            atol=1.0e-14,
        ),
        "Projection changed masses below the 2021 search range",
    )
    require(
        np.allclose(
            out.loc[only_2021, "full2021_projection_scale_eps2"],
            1.0 / math.sqrt(10.0),
            rtol=0.0,
            atol=1.0e-14,
        ),
        "2021-only projection is not exactly 1/sqrt(10)",
    )
    return out


def crossing_location(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> float:
    if math.isclose(y0, y1, rel_tol=0.0, abs_tol=1.0e-15):
        return 0.5 * (x0 + x1)
    return x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)


def crossing_intervals(
    comparison: pd.DataFrame,
    ratio_column: str,
    curve: str,
) -> list[dict[str, float | int | str]]:
    x = comparison["mass_MeV"].to_numpy(float)
    ratio = comparison[ratio_column].to_numpy(float)
    indices = np.flatnonzero(np.isfinite(ratio) & (ratio < 1.0))
    if not len(indices):
        return []

    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for item in indices[1:]:
        item = int(item)
        if item != previous + 1:
            groups.append((start, previous))
            start = item
        previous = item
    groups.append((start, previous))

    rows: list[dict[str, float | int | str]] = []
    for interval_index, (start, stop) in enumerate(groups, start=1):
        low = x[start]
        if start > 0 and np.isfinite(ratio[start - 1]):
            low = crossing_location(
                x[start - 1], x[start], ratio[start - 1], ratio[start]
            )
        high = x[stop]
        if stop + 1 < len(x) and np.isfinite(ratio[stop + 1]):
            high = crossing_location(
                x[stop], x[stop + 1], ratio[stop], ratio[stop + 1]
            )
        local = ratio[start : stop + 1]
        minimum_index = start + int(np.argmin(local))
        rows.append(
            {
                "hps_curve": curve,
                "interval_index": interval_index,
                "first_below_grid_mass_MeV": int(x[start]),
                "last_below_grid_mass_MeV": int(x[stop]),
                "linear_crossing_low_MeV": float(low),
                "linear_crossing_high_MeV": float(high),
                "minimum_ratio": float(ratio[minimum_index]),
                "minimum_ratio_mass_MeV": int(x[minimum_index]),
                "grid_points_below": stop - start + 1,
            }
        )
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14.0,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=COLOR_GRID, linewidth=0.75, alpha=0.75)
    ax.grid(
        True,
        which="minor",
        axis="y",
        color=COLOR_MINOR_GRID,
        linewidth=0.45,
        alpha=0.55,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#5D646B")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", width=0.9, length=5.5)
    ax.tick_params(axis="both", which="minor", width=0.65, length=3.0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str, title: str) -> list[Path]:
    paths: list[Path] = []
    metadata = {
        "Title": title,
        "Author": "HPS-GPR v4.2 follow-up",
        "Subject": (
            "Observed and observed-equivalent HPS comparison with the "
            "BaBar 2014 visible-dark-photon limit"
        ),
        "Keywords": "HPS, GPR, BaBar, dark photon, epsilon squared",
    }
    for suffix in ("pdf", "png", "svg"):
        path = FIGURES / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": "white",
        }
        if suffix == "png":
            kwargs["dpi"] = 300
        if suffix == "pdf":
            kwargs["metadata"] = metadata
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_overlay(
    comparison: pd.DataFrame,
    babar: pd.DataFrame,
) -> list[Path]:
    focus = babar[
        (babar["mass_MeV"] >= 19.0) & (babar["mass_MeV"] <= 250.0)
    ]
    x = comparison["mass_MeV"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.4, 6.0), constrained_layout=True)
    ax.plot(
        focus["mass_MeV"],
        focus["eps2_90"],
        color=COLOR_BABAR,
        linewidth=2.6,
        label=r"BaBar visible $A^\prime$ (2014)",
        zorder=2,
    )
    ax.plot(
        x,
        comparison["hps_v4p2_eps2_obs_minimal_visible"],
        color=COLOR_CURRENT,
        linewidth=2.0,
        linestyle=(0, (5.5, 2.7)),
        label="HPS combined observed\n(2021 at 10%)",
        zorder=3,
    )
    ax.plot(
        x,
        comparison[
            "hps_v4p2_projected_full2021_eps2_minimal_visible"
        ],
        color=COLOR_PROJECTED,
        linewidth=2.9,
        label="HPS observed-equivalent proxy\n(2021 at 100%)",
        zorder=4,
    )
    ax.axvline(
        1000.0 * DIMUON_THRESHOLD_GEV,
        color=COLOR_THRESHOLD,
        linewidth=1.0,
        linestyle=(0, (2.0, 2.3)),
        alpha=0.78,
        zorder=1,
    )
    ax.text(
        1000.0 * DIMUON_THRESHOLD_GEV + 2.2,
        0.035,
        r"$2m_\mu$",
        transform=ax.get_xaxis_transform(),
        color="#697078",
        fontsize=10.0,
        ha="left",
        va="bottom",
    )

    plotted = np.concatenate(
        [
            focus["eps2_90"].to_numpy(float),
            comparison["hps_v4p2_eps2_obs_minimal_visible"].to_numpy(float),
            comparison[
                "hps_v4p2_projected_full2021_eps2_minimal_visible"
            ].to_numpy(float),
        ]
    )
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(plotted) * 0.62), float(np.max(plotted) * 1.55))
    ax.set_xlim(19.0, 250.0)
    ax.set_xlabel(r"Dark-photon mass $m_{A^\prime}$ (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.45,
        handlelength=2.5,
    )
    return save_figure(
        fig,
        OVERLAY_STEM,
        "HPS-GPR v4.2 observed-equivalent projection compared with BaBar",
    )


def plot_overlay_with_projected_over_babar_ratio(
    comparison: pd.DataFrame,
    babar: pd.DataFrame,
) -> list[Path]:
    """Draw the primary overlay with projected-HPS/BaBar underneath."""
    focus = babar[
        (babar["mass_MeV"] >= 19.0) & (babar["mass_MeV"] <= 250.0)
    ]
    selected = comparison[
        np.isfinite(
            comparison["hps_v4p2_projected_full2021_over_babar"]
        )
    ].copy()
    x = comparison["mass_MeV"].to_numpy(float)
    ratio_x = selected["mass_MeV"].to_numpy(float)
    projected_over_babar = selected[
        "hps_v4p2_projected_full2021_over_babar"
    ].to_numpy(float)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10.4, 7.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.25, 1.35], "hspace": 0.035},
    )
    ax.plot(
        focus["mass_MeV"],
        focus["eps2_90"],
        color=COLOR_BABAR,
        linewidth=2.6,
        label=r"BaBar visible $A^\prime$ (2014)",
        zorder=2,
    )
    ax.plot(
        x,
        comparison["hps_v4p2_eps2_obs_minimal_visible"],
        color=COLOR_CURRENT,
        linewidth=2.0,
        linestyle=(0, (5.5, 2.7)),
        label="HPS combined observed\n(2021 at 10%)",
        zorder=3,
    )
    ax.plot(
        x,
        comparison[
            "hps_v4p2_projected_full2021_eps2_minimal_visible"
        ],
        color=COLOR_PROJECTED,
        linewidth=2.9,
        label="HPS observed-equivalent proxy\n(2021 at 100%)",
        zorder=4,
    )
    ax.axvline(
        1000.0 * DIMUON_THRESHOLD_GEV,
        color=COLOR_THRESHOLD,
        linewidth=1.0,
        linestyle=(0, (2.0, 2.3)),
        alpha=0.78,
        zorder=1,
    )
    ax.text(
        1000.0 * DIMUON_THRESHOLD_GEV + 2.2,
        0.035,
        r"$2m_\mu$",
        transform=ax.get_xaxis_transform(),
        color="#697078",
        fontsize=10.0,
        ha="left",
        va="bottom",
    )

    plotted = np.concatenate(
        [
            focus["eps2_90"].to_numpy(float),
            comparison["hps_v4p2_eps2_obs_minimal_visible"].to_numpy(float),
            comparison[
                "hps_v4p2_projected_full2021_eps2_minimal_visible"
            ].to_numpy(float),
        ]
    )
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(plotted) * 0.62), float(np.max(plotted) * 1.55))
    ax.set_xlim(19.0, 250.0)
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=9.4,
        columnspacing=1.35,
        handlelength=2.5,
    )

    ratio_lower = min(float(np.min(projected_over_babar) * 0.72), 0.28)
    ratio_upper = max(float(np.max(projected_over_babar) * 1.28), 3.2)
    ratio_ax.axhspan(
        ratio_lower,
        1.0,
        color="#E8F0F7",
        alpha=0.72,
        zorder=0,
    )
    ratio_ax.axhline(
        1.0,
        color="#4E555C",
        linewidth=1.25,
        linestyle=(0, (4.0, 2.4)),
        zorder=2,
    )
    ratio_ax.plot(
        ratio_x,
        projected_over_babar,
        color=COLOR_PROJECTED,
        linewidth=2.35,
        zorder=3,
    )
    ratio_ax.axvline(
        1000.0 * DIMUON_THRESHOLD_GEV,
        color=COLOR_THRESHOLD,
        linewidth=1.0,
        linestyle=(0, (2.0, 2.3)),
        alpha=0.78,
        zorder=1,
    )
    ratio_ax.set_yscale("log")
    ratio_ax.set_ylim(ratio_lower, ratio_upper)
    ratio_ax.set_title(
        "Projected HPS proxy / BaBar 2014",
        loc="left",
        fontsize=9.4,
        color="#52606D",
        pad=5.0,
    )
    ratio_ax.set_xlabel(r"Dark-photon mass $m_{A^\prime}$ (MeV)")
    ratio_ax.set_ylabel("Limit ratio")
    ratio_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ratio_ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ratio_ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60)
    )
    ratio_ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ratio_ax)
    return save_figure(
        fig,
        OVERLAY_PROJECTED_OVER_BABAR_STEM,
        (
            "HPS-GPR v4.2 observed-equivalent projection with "
            "projected-HPS-over-BaBar ratio"
        ),
    )


def plot_ratio(comparison: pd.DataFrame) -> list[Path]:
    selected = comparison[
        np.isfinite(comparison["hps_v4p2_observed_over_babar"])
    ].copy()
    x = selected["mass_MeV"].to_numpy(float)
    current = selected["hps_v4p2_observed_over_babar"].to_numpy(float)
    projected = selected[
        "hps_v4p2_projected_full2021_over_babar"
    ].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.4, 5.35), constrained_layout=True)
    lower = min(float(np.min(projected) * 0.72), 0.32)
    upper = float(max(np.max(current), np.max(projected)) * 1.25)
    ax.axhspan(lower, 1.0, color="#E8F0F7", alpha=0.72, zorder=0)
    ax.axhline(
        1.0,
        color="#4E555C",
        linewidth=1.25,
        linestyle=(0, (4.0, 2.4)),
        label="Equal numerical limit",
        zorder=2,
    )
    ax.plot(
        x,
        current,
        color=COLOR_CURRENT,
        linewidth=2.0,
        linestyle=(0, (5.5, 2.7)),
        label=r"v4.2 observed / BaBar",
        zorder=3,
    )
    ax.plot(
        x,
        projected,
        color=COLOR_PROJECTED,
        linewidth=2.8,
        label=r"combined v4.2 observed-equivalent / BaBar",
        zorder=4,
    )
    ax.axvline(
        1000.0 * DIMUON_THRESHOLD_GEV,
        color=COLOR_THRESHOLD,
        linewidth=1.0,
        linestyle=(0, (2.0, 2.3)),
        alpha=0.78,
        zorder=1,
    )
    ax.text(
        1000.0 * DIMUON_THRESHOLD_GEV + 2.2,
        0.035,
        r"$2m_\mu$",
        transform=ax.get_xaxis_transform(),
        color="#697078",
        fontsize=10.0,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.018,
        0.055,
        "HPS numerically lower than BaBar",
        transform=ax.transAxes,
        color="#526F89",
        fontsize=9.8,
        ha="left",
        va="bottom",
    )
    ax.set_yscale("log")
    ax.set_ylim(lower, upper)
    ax.set_xlim(19.0, 250.0)
    ax.set_xlabel(r"Dark-photon mass $m_{A^\prime}$ (MeV)")
    ax.set_ylabel("HPS / BaBar limit ratio")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.45,
        handlelength=2.5,
    )
    return save_figure(
        fig,
        RATIO_STEM,
        "HPS-GPR v4.2 and BaBar limit-ratio diagnostic",
    )


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
    ).strip()


def minimum_record(
    table: pd.DataFrame,
    column: str,
) -> dict[str, float | int]:
    finite = table[np.isfinite(table[column])]
    row = finite.loc[finite[column].idxmin()]
    return {
        "mass_MeV": int(row["mass_MeV"]),
        "value": float(row[column]),
    }


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_provenance(
    comparison: pd.DataFrame,
    babar: pd.DataFrame,
    intervals: pd.DataFrame,
    generated_paths: list[Path],
) -> Path:
    overlap_50_90 = comparison[
        comparison["mass_MeV"].between(50, 90)
    ]["full2021_projection_scale_eps2"]
    overlap_91_180 = comparison[
        comparison["mass_MeV"].between(91, 180)
    ]["full2021_projection_scale_eps2"]
    payload = {
        "schema_version": 1,
        "status": "GENERATED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "v4p2_followups_20260806_babar_projection",
        "git": {
            "head": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "authoritative_v4p2_publication_commit": EXPECTED_V4P2_COMMIT,
        },
        "sources": {
            "hps_v4p2_reviewed_table": artifact_record(HPS_SOURCE),
            "hps_v4p2_config": artifact_record(CONFIG_SOURCE),
            "hps_v4p2_analysis_note": artifact_record(NOTE_SOURCE),
            "babar_visible2014_raw": {
                **artifact_record(FROZEN_BABAR),
                "publication": (
                    "Lees et al., Phys. Rev. Lett. 113, 201801 (2014), "
                    "arXiv:1406.2980"
                ),
                "quantity": "observed 90% upper limit on epsilon",
                "valid_points": int(len(babar)),
                "mass_range_MeV": [
                    float(babar["mass_MeV"].min()),
                    float(babar["mass_MeV"].max()),
                ],
            },
        },
        "hps_result": {
            "column": "eps2_obs_minimal_visible",
            "confidence_level": 0.90,
            "cls_alpha": 0.10,
            "cls_statistic": "tilde_q_mu",
            "cls_calibration": "asymptotic",
            "combined_mode": "count_scale",
            "samples": {
                "2015": "full",
                "2016": "full",
                "2021": "10%",
            },
            "grid": {
                "rows": int(len(comparison)),
                "mass_low_MeV": int(comparison["mass_MeV"].min()),
                "mass_high_MeV": int(comparison["mass_MeV"].max()),
                "step_MeV": 1,
            },
            "dimuon_threshold_GeV": DIMUON_THRESHOLD_GEV,
            "first_corrected_grid_mass_MeV": 212,
            "observed_minimum": minimum_record(
                comparison,
                "hps_v4p2_eps2_obs_minimal_visible",
            ),
        },
        "projection": {
            "definition": "observed-equivalent density response proxy",
            "formula": (
                "eps2_projected = eps2_observed_minimal_visible * "
                "sqrt(sum_d rho_d / sum_d f_d rho_d)"
            ),
            "exposure_factors": FULL_EXPOSURE_FACTORS,
            "density_definition": (
                "observed counts-per-GeV density in the physical "
                "m +/- 1.64 sigma_m normalization window"
            ),
            "scale_checks": {
                "below_50_MeV": 1.0,
                "50_to_90_MeV_range": [
                    float(overlap_50_90.min()),
                    float(overlap_50_90.max()),
                ],
                "91_to_180_MeV_range": [
                    float(overlap_91_180.min()),
                    float(overlap_91_180.max()),
                ],
                "above_180_MeV": 1.0 / math.sqrt(10.0),
            },
            "projected_minimum": minimum_record(
                comparison,
                "hps_v4p2_projected_full2021_eps2_minimal_visible",
            ),
            "bands_projected": False,
            "pvalues_projected": False,
        },
        "babar_comparison": {
            "raw_contour_plotted_without_interpolation": True,
            "ratio_interpolation": (
                "linear in log(epsilon^2), no extrapolation"
            ),
            "companion_ratio_panel": {
                "quantity": "projected HPS observed-equivalent proxy / BaBar 2014",
                "stronger_projected_limit_region": "ratio below unity",
                "in_axes_explanatory_text": False,
            },
            "current_minimum_ratio": minimum_record(
                comparison,
                "hps_v4p2_observed_over_babar",
            ),
            "projected_minimum_ratio": minimum_record(
                comparison,
                "hps_v4p2_projected_full2021_over_babar",
            ),
            "current_grid_points_below_babar": int(
                comparison[
                    "hps_v4p2_observed_below_babar_on_grid"
                ].sum()
            ),
            "projected_grid_points_below_babar": int(
                comparison[
                    "hps_v4p2_projected_full2021_below_babar_on_grid"
                ].sum()
            ),
            "crossing_intervals": intervals.to_dict(orient="records"),
        },
        "semantic_boundaries": [
            (
                "The projected curve preserves the current observed "
                "fluctuation pattern and is not an expected median sensitivity."
            ),
            (
                "The projected curve is not a refit to full-2021 data or a "
                "future observed limit."
            ),
            "No expected-limit bands or p-values are projected.",
            (
                "Disconnected below-BaBar intervals are numerical crossings "
                "of a fluctuation-derived proxy, not reach probabilities."
            ),
            (
                "BaBar 2014 visible and BaBar 2017 invisible results test "
                "different decay hypotheses and are not interchangeable."
            ),
            (
                "The v4.2 factor-12 2016 setting followed an observed "
                "boundary diagnostic, so post-selection qualifications remain."
            ),
        ],
        "generator": artifact_record(Path(__file__).resolve()),
        "outputs": [artifact_record(path) for path in generated_paths],
    }
    validation_script = HERE / "validate_babar_projection.py"
    if validation_script.is_file():
        payload["validator"] = artifact_record(validation_script)
    path = DERIVED / "provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    prepare_directories()
    setup_style()
    babar_path = freeze_babar_source()
    babar = load_babar(babar_path)
    hps = load_v4p2()
    comparison = build_reviewed_projection(hps, babar)

    current_intervals = crossing_intervals(
        comparison,
        "hps_v4p2_observed_over_babar",
        "v4p2_observed",
    )
    projected_intervals = crossing_intervals(
        comparison,
        "hps_v4p2_projected_full2021_over_babar",
        "v4p2_projected_full2021_observed_equivalent",
    )
    intervals = pd.DataFrame(current_intervals + projected_intervals)

    reviewed_path = DERIVED / "v4p2_babar_projection_reviewed.csv"
    comparison.to_csv(reviewed_path, index=False, float_format="%.17g")
    babar_path_derived = DERIVED / "babar_visible2014_eps2_90.csv"
    babar.to_csv(
        babar_path_derived,
        index=False,
        float_format="%.17g",
    )
    intervals_path = DERIVED / "crossing_intervals.csv"
    intervals.to_csv(intervals_path, index=False, float_format="%.12g")

    figures = (
        plot_overlay(comparison, babar)
        + plot_ratio(comparison)
        + plot_overlay_with_projected_over_babar_ratio(comparison, babar)
    )
    generated = [
        reviewed_path,
        babar_path_derived,
        intervals_path,
        *figures,
    ]
    provenance = write_provenance(
        comparison,
        babar,
        intervals,
        generated,
    )

    summary = {
        "status": "generated",
        "reviewed_csv": relative(reviewed_path),
        "provenance": relative(provenance),
        "figures": [relative(path) for path in figures],
        "observed_minimum": minimum_record(
            comparison,
            "hps_v4p2_eps2_obs_minimal_visible",
        ),
        "projected_minimum": minimum_record(
            comparison,
            "hps_v4p2_projected_full2021_eps2_minimal_visible",
        ),
        "current_minimum_ratio": minimum_record(
            comparison,
            "hps_v4p2_observed_over_babar",
        ),
        "projected_minimum_ratio": minimum_record(
            comparison,
            "hps_v4p2_projected_full2021_over_babar",
        ),
        "projected_grid_points_below_babar": int(
            comparison[
                "hps_v4p2_projected_full2021_below_babar_on_grid"
            ].sum()
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
