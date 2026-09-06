#!/usr/bin/env python3
"""Make the review-ready observed optimized-length-scale comparison.

The figure compares matched-card exposure pairs within each run period:

* 2016 10% versus 100%, both with the v4.1 wide-support k=12 card;
* 2021 1% versus 10%, both with the reviewed wide-support k=15 card.

Only reviewed observed optimizer states are plotted.  This script does not
construct expected bands, toy results, or interpolated optimizer states.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
DERIVED = STUDY / "derived"
PLOTS = STUDY / "plots"

SOURCE_2016_10 = DERIVED / "2016_10pct_observed_k12_reviewed.csv"
SOURCE_2016_100_GRID = (
    REPO
    / "study_results/v4p1_2016_ls_upper_optimization_20260804/derived/"
    "pointwise_factor_grid.csv"
)
SOURCE_2021_1 = (
    REPO
    / "study_results/observed_2021_1pct_vs_10pct_k15_20260803/derived/"
    "observed_2021_1pct_reviewed.csv"
)
SOURCE_2021_10 = (
    REPO
    / "study_results/finalist_k15_2021_10pct_combined100toy_20260803/"
    "derived/observed_2021_reviewed.csv"
)

FIGURE_STEM = PLOTS / "fig_v4p1_ls_observed_dataset_comparison"
SUMMARY_CSV = DERIVED / "fig_v4p1_ls_observed_dataset_comparison_summary.csv"
SUMMARY_JSON = DERIVED / "fig_v4p1_ls_observed_dataset_comparison_summary.json"

BOUNDARY_FRACTION = 0.999
SUPPORT_SENSITIVE_2021_1PCT_MASSES_MEV = (50, 51, 52)

LOW_EXPOSURE_COLOR = "#0072B2"
HIGH_EXPOSURE_COLOR = "#D55E00"
CEILING_COLOR = "#4D4D4D"
EDGE_COLOR = "#7A3E9D"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_series(
    frame: pd.DataFrame,
    *,
    label: str,
    expected_masses_mev: np.ndarray,
    ceiling: float,
) -> pd.DataFrame:
    required = {
        "mass_GeV",
        "ls_opt",
        "sigma_x",
        "ls_opt_over_sigma_x",
        "ls_hi_over_sigma_x",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} is missing columns: {sorted(missing)}")

    frame = frame.sort_values("mass_GeV").reset_index(drop=True).copy()
    if len(frame) != len(expected_masses_mev):
        raise RuntimeError(
            f"{label} has {len(frame)} rows; expected {len(expected_masses_mev)}"
        )
    actual_mass = np.rint(frame["mass_GeV"] * 1000.0).astype(int).to_numpy()
    if not np.array_equal(actual_mass, expected_masses_mev):
        raise RuntimeError(f"{label} does not have the expected exact mass grid")
    if not np.allclose(
        frame["mass_GeV"].to_numpy(float),
        expected_masses_mev / 1000.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(f"{label} mass values are not exact to 1e-12 GeV")

    numeric = frame[
        [
            "mass_GeV",
            "ls_opt",
            "sigma_x",
            "ls_opt_over_sigma_x",
            "ls_hi_over_sigma_x",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{label} contains non-finite optimizer values")
    if np.any(frame["sigma_x"].to_numpy(float) <= 0.0):
        raise RuntimeError(f"{label} contains non-positive sigma_x")
    if not np.allclose(
        frame["ls_opt"].to_numpy(float) / frame["sigma_x"].to_numpy(float),
        frame["ls_opt_over_sigma_x"].to_numpy(float),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(f"{label} has inconsistent ls_opt/sigma_x values")
    if not np.allclose(
        frame["ls_hi_over_sigma_x"].to_numpy(float),
        ceiling,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(f"{label} does not realize ceiling {ceiling:g}")
    if "interpolated" in frame and frame["interpolated"].astype(bool).any():
        raise RuntimeError(f"{label} contains interpolated optimizer states")
    if "review_status" in frame:
        allowed = {
            "resolved_reproduced_max_lml",
            "raw_stable",
            "raw_scan_row",
            "repair_selected_reproduced_max_lml",
        }
        unexpected = set(frame["review_status"].dropna().astype(str)) - allowed
        if unexpected:
            raise RuntimeError(
                f"{label} contains unresolved review states: {sorted(unexpected)}"
            )

    frame["mass_MeV"] = actual_mass
    frame["at_configured_ceiling"] = (
        frame["ls_opt_over_sigma_x"].to_numpy(float)
        >= BOUNDARY_FRACTION
        * frame["ls_hi_over_sigma_x"].to_numpy(float)
    )
    return frame


def series_summary(
    frame: pd.DataFrame,
    *,
    year: int,
    exposure: str,
    source: Path,
    source_filter: str,
    ceiling: float,
    support_sensitive_masses: tuple[int, ...] = (),
) -> dict[str, Any]:
    values = frame["ls_opt_over_sigma_x"].to_numpy(float)
    at_ceiling = frame["at_configured_ceiling"].to_numpy(bool)
    return {
        "year": year,
        "exposure": exposure,
        "source": relative(source),
        "source_sha256": sha256_file(source),
        "source_filter": source_filter,
        "rows": int(len(frame)),
        "mass_min_MeV": int(frame["mass_MeV"].min()),
        "mass_max_MeV": int(frame["mass_MeV"].max()),
        "configured_ceiling": float(ceiling),
        "boundary_fraction_criterion": float(BOUNDARY_FRACTION),
        "ls_opt_over_sigma_x_min": float(np.min(values)),
        "ls_opt_over_sigma_x_median": float(np.median(values)),
        "ls_opt_over_sigma_x_max": float(np.max(values)),
        "at_ceiling_count": int(np.sum(at_ceiling)),
        "at_ceiling_fraction": float(np.mean(at_ceiling)),
        "at_ceiling_masses_MeV": "|".join(
            str(value)
            for value in frame.loc[at_ceiling, "mass_MeV"].astype(int)
        ),
        "support_sensitive_masses_MeV": "|".join(
            str(value) for value in support_sensitive_masses
        ),
        "interpolated_rows": 0,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.5,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "legend.frameon": False,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def legend_label(label: str, summary: dict[str, Any]) -> str:
    return (
        f"{label}  "
        f"(median {summary['ls_opt_over_sigma_x_median']:.2f}; "
        f"at bound {summary['at_ceiling_count']}/{summary['rows']})"
    )


def draw_panel(
    ax: plt.Axes,
    *,
    low: pd.DataFrame,
    high: pd.DataFrame,
    low_label: str,
    high_label: str,
    low_summary: dict[str, Any],
    high_summary: dict[str, Any],
    ceiling: float,
    title: str,
    panel_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    x_major: float,
) -> None:
    ax.plot(
        low["mass_MeV"],
        low["ls_opt_over_sigma_x"],
        color=LOW_EXPOSURE_COLOR,
        linewidth=1.35,
        marker="o",
        markersize=2.5,
        markerfacecolor="white",
        markeredgewidth=0.75,
        label=legend_label(low_label, low_summary),
        zorder=3,
    )
    ax.plot(
        high["mass_MeV"],
        high["ls_opt_over_sigma_x"],
        color=HIGH_EXPOSURE_COLOR,
        linewidth=1.45,
        marker="o",
        markersize=2.1,
        markeredgewidth=0.0,
        label=legend_label(high_label, high_summary),
        zorder=4,
    )
    ax.axhline(
        ceiling,
        color=CEILING_COLOR,
        linewidth=1.05,
        linestyle=(0, (4, 2.5)),
        zorder=1,
    )
    ax.text(
        0.985,
        ceiling,
        rf"$k_{{\max}}={ceiling:g}$",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=CEILING_COLOR,
    )
    ax.set_title(title, pad=8.0)
    ax.text(
        0.018,
        0.965,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ [MeV]")
    ax.legend(loc="lower right", handlelength=2.7, borderaxespad=0.7)


def paired_summary(
    low: pd.DataFrame,
    high: pd.DataFrame,
    *,
    label: str,
) -> dict[str, Any]:
    if not np.array_equal(
        low["mass_MeV"].to_numpy(int), high["mass_MeV"].to_numpy(int)
    ):
        raise RuntimeError(f"{label} mass grids are not matched")
    delta = (
        high["ls_opt_over_sigma_x"].to_numpy(float)
        - low["ls_opt_over_sigma_x"].to_numpy(float)
    )
    return {
        "label": label,
        "matched_mass_grid": True,
        "rows": int(len(delta)),
        "mass_min_MeV": int(low["mass_MeV"].min()),
        "mass_max_MeV": int(low["mass_MeV"].max()),
        "delta_definition": "higher exposure minus lower exposure",
        "delta_ls_opt_over_sigma_x_min": float(np.min(delta)),
        "delta_ls_opt_over_sigma_x_median": float(np.median(delta)),
        "delta_ls_opt_over_sigma_x_max": float(np.max(delta)),
    }


def main() -> None:
    for path in (
        SOURCE_2016_10,
        SOURCE_2016_100_GRID,
        SOURCE_2021_1,
        SOURCE_2021_10,
    ):
        require_file(path)
    DERIVED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    masses_2016 = np.arange(39, 181, dtype=int)
    masses_2021 = np.arange(50, 251, dtype=int)

    frame_2016_10 = require_series(
        pd.read_csv(SOURCE_2016_10),
        label="2016 10%",
        expected_masses_mev=masses_2016,
        ceiling=12.0,
    )
    full_grid = pd.read_csv(SOURCE_2016_100_GRID)
    frame_2016_100 = require_series(
        full_grid.loc[
            np.isclose(full_grid["upper_factor"].to_numpy(float), 12.0)
        ].copy(),
        label="2016 100%",
        expected_masses_mev=masses_2016,
        ceiling=12.0,
    )
    frame_2021_1 = require_series(
        pd.read_csv(SOURCE_2021_1),
        label="2021 1%",
        expected_masses_mev=masses_2021,
        ceiling=15.0,
    )
    frame_2021_10 = require_series(
        pd.read_csv(SOURCE_2021_10),
        label="2021 10%",
        expected_masses_mev=masses_2021,
        ceiling=15.0,
    )

    summaries = [
        series_summary(
            frame_2016_10,
            year=2016,
            exposure="10%",
            source=SOURCE_2016_10,
            source_filter="all reviewed rows",
            ceiling=12.0,
        ),
        series_summary(
            frame_2016_100,
            year=2016,
            exposure="100%",
            source=SOURCE_2016_100_GRID,
            source_filter="upper_factor == 12",
            ceiling=12.0,
        ),
        series_summary(
            frame_2021_1,
            year=2021,
            exposure="1%",
            source=SOURCE_2021_1,
            source_filter="all reviewed rows",
            ceiling=15.0,
            support_sensitive_masses=SUPPORT_SENSITIVE_2021_1PCT_MASSES_MEV,
        ),
        series_summary(
            frame_2021_10,
            year=2021,
            exposure="10%",
            source=SOURCE_2021_10,
            source_filter="all reviewed rows",
            ceiling=15.0,
        ),
    ]
    summary_lookup = {
        (row["year"], row["exposure"]): row for row in summaries
    }

    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.9),
        constrained_layout=True,
    )
    draw_panel(
        axes[0],
        low=frame_2016_10,
        high=frame_2016_100,
        low_label="10%",
        high_label="100%",
        low_summary=summary_lookup[(2016, "10%")],
        high_summary=summary_lookup[(2016, "100%")],
        ceiling=12.0,
        title="2016 observed data",
        panel_label="(a)",
        x_limits=(37.0, 182.0),
        y_limits=(8.55, 12.35),
        x_major=20.0,
    )
    draw_panel(
        axes[1],
        low=frame_2021_1,
        high=frame_2021_10,
        low_label="1%",
        high_label="10%",
        low_summary=summary_lookup[(2021, "1%")],
        high_summary=summary_lookup[(2021, "10%")],
        ceiling=15.0,
        title="2021 observed data",
        panel_label="(b)",
        x_limits=(47.0, 253.0),
        y_limits=(7.55, 15.60),
        x_major=25.0,
    )
    axes[0].set_ylabel(
        r"Optimized length scale $\ell_{\mathrm{opt}}/\sigma_x$"
    )

    edge = frame_2021_1.loc[
        frame_2021_1["mass_MeV"].isin(
            SUPPORT_SENSITIVE_2021_1PCT_MASSES_MEV
        )
    ]
    if tuple(edge["mass_MeV"].astype(int)) != (
        SUPPORT_SENSITIVE_2021_1PCT_MASSES_MEV
    ):
        raise RuntimeError("2021 support-sensitive edge points are missing")
    axes[1].axvspan(
        49.5,
        52.5,
        color=EDGE_COLOR,
        alpha=0.08,
        linewidth=0.0,
        zorder=0,
    )
    axes[1].scatter(
        edge["mass_MeV"],
        edge["ls_opt_over_sigma_x"],
        marker="D",
        s=35,
        facecolor="white",
        edgecolor=EDGE_COLOR,
        linewidth=1.25,
        zorder=7,
    )
    axes[1].annotate(
        "support-sensitive\nlower-edge points",
        xy=(51.0, 15.0),
        xytext=(73.0, 14.25),
        color=EDGE_COLOR,
        fontsize=8.7,
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "-",
            "color": EDGE_COLOR,
            "linewidth": 0.9,
            "shrinkA": 2,
            "shrinkB": 3,
        },
    )

    pdf_path = FIGURE_STEM.with_suffix(".pdf")
    png_path = FIGURE_STEM.with_suffix(".png")
    figure_metadata = {
        "Title": "Observed optimized GP length scale versus dataset exposure",
        "Author": "HPS GPR validation workflow",
        "Subject": (
            "Observed per-mass optimized length scale normalized by local "
            "mass-resolution scale; no limit bands"
        ),
        "Keywords": "HPS, Gaussian process, length scale, observed",
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=figure_metadata)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    for path in (pdf_path, png_path):
        require_file(path)
        if path.stat().st_size < 10_000:
            raise RuntimeError(f"figure output is unexpectedly small: {path}")

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(SUMMARY_CSV, index=False)

    paired = [
        paired_summary(
            frame_2016_10,
            frame_2016_100,
            label="2016 100% minus 10%, same k=12 wide-support card",
        ),
        paired_summary(
            frame_2021_1,
            frame_2021_10,
            label="2021 10% minus 1%, same k=15 wide-support card",
        ),
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "purpose": (
            "Review-ready observed optimized-length-scale exposure comparison; "
            "no expected bands or toy-derived quantities."
        ),
        "normalization": {
            "plotted_quantity": "ls_opt_over_sigma_x",
            "definition": "ls_opt / sigma_x at each mass hypothesis",
            "gp_coordinate": "x = ln(mass)",
            "boundary_criterion": (
                "ls_opt_over_sigma_x >= "
                f"{BOUNDARY_FRACTION:g} * ls_hi_over_sigma_x"
            ),
        },
        "series": summaries,
        "paired_comparisons": paired,
        "support_sensitive_points": {
            "series": "2021 1%",
            "masses_MeV": list(
                SUPPORT_SENSITIVE_2021_1PCT_MASSES_MEV
            ),
            "reason": (
                "lower-edge fits have only 9-12 low-side training bins and "
                "sit at the configured length-scale ceiling"
            ),
        },
        "outputs": {
            "pdf": relative(pdf_path),
            "pdf_sha256": sha256_file(pdf_path),
            "png": relative(png_path),
            "png_sha256": sha256_file(png_path),
            "summary_csv": relative(SUMMARY_CSV),
            "summary_csv_sha256": sha256_file(SUMMARY_CSV),
            "script": relative(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "validation": {
            "exact_mass_grids": True,
            "matched_grids_within_year": True,
            "finite_optimizer_states": True,
            "realized_configured_ceilings": True,
            "interpolated_rows": 0,
            "limit_bands_created": False,
            "toy_results_used": False,
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
