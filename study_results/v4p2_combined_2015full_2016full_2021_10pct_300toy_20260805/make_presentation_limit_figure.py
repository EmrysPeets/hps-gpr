#!/usr/bin/env python3
"""Make the presentation version of the v4.2 combined limit figure.

This is a plotting-only consumer of the reviewed v4.2 table.  It deliberately
omits the observed/median subpanel while retaining the limit curves, the
conditional fixed-GP toy-limit quantiles, and the active-dataset strip.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator


os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p2-presentation-mpl")
# Stabilize PDF metadata so an unchanged source and environment give a stable
# byte-level artifact.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1785898800")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
INPUT = CAMPAIGN_DIR / "derived" / "combined_bands300_reviewed_v4p2.csv"
FIGURE_DIR = CAMPAIGN_DIR / "note_figures"
NOTE_FIGURE_DIR = (
    REPO
    / "hps_gpr_analysis_note"
    / "final_limit_projection_figs"
    / "v4p2_20260805_combined300"
)
DELIVERY_DIR = REPO / "output" / "pdf"
PROVENANCE = (
    CAMPAIGN_DIR
    / "derived"
    / "presentation_limit_figure_provenance_v4p2.json"
)

STEM = "combined_observed_bands300_minimal_visible_presentation"
EXPECTED_INPUT_SHA256 = (
    "8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd"
)
EXPECTED_SOURCE_BANDS_SHA256 = (
    "b90768ab361928c63f57b3981d424fd36506893da2447e40824acdf3d20081c2"
)
MASS_LOW_MEV = 19
MASS_HIGH_MEV = 250
N_TOYS = 300
M_MU_GEV = 0.1056583745
DIMUON_THRESHOLD_MEV = 2000.0 * M_MU_GEV

COLORS = {
    "observed": "#B42318",
    "expected": "#202124",
    "band1": "#4C956C",
    "band2": "#F2C14E",
    "threshold": "#6B7280",
}
ACTIVE_COLORS = {
    "2015": "#DCEAF7",
    "2015+2016": "#F4DDD8",
    "2015+2016+2021": "#DDEFE8",
    "2016+2021": "#F2E4D8",
    "2021": "#D9EEE7",
}
ACTIVE_LABELS = {
    "2015": "2015",
    "2015+2016": "15+16",
    "2015+2016+2021": "15+16+21",
    "2016+2021": "16+21",
    "2021": "2021",
}
EXPECTED_ACTIVE_COUNTS = {
    "2015": 20,
    "2015+2016": 11,
    "2015+2016+2021": 41,
    "2016+2021": 90,
    "2021": 70,
}
QUANTILE_COLUMNS = (
    "eps2_lo2_minimal_visible",
    "eps2_lo1_minimal_visible",
    "eps2_med_minimal_visible",
    "eps2_hi1_minimal_visible",
    "eps2_hi2_minimal_visible",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_reviewed_table() -> pd.DataFrame:
    if not INPUT.is_file():
        raise RuntimeError(f"Reviewed v4.2 table is missing: {INPUT}")
    actual_sha256 = sha256(INPUT)
    if actual_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "Reviewed v4.2 table hash changed: "
            f"expected {EXPECTED_INPUT_SHA256}, got {actual_sha256}"
        )

    frame = pd.read_csv(INPUT).sort_values("mass_MeV").reset_index(drop=True)
    required = {
        "mass_MeV",
        "dataset_set",
        "n_toys_finite",
        "combined_mode",
        "coverage_calibrated",
        "scan_toy_calibrated",
        "source_bands_sha256",
        "eps2_obs_minimal_visible",
        *QUANTILE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Reviewed table lacks columns: {missing}")

    expected_mass = np.arange(MASS_LOW_MEV, MASS_HIGH_MEV + 1, dtype=float)
    if len(frame) != len(expected_mass) or not np.array_equal(
        frame["mass_MeV"].to_numpy(float),
        expected_mass,
    ):
        raise RuntimeError("Reviewed table is not the exact 19--250 MeV grid")
    if not np.all(frame["n_toys_finite"].to_numpy(int) == N_TOYS):
        raise RuntimeError("Not every mass has exactly 300 finite toy limits")
    active_counts = frame["dataset_set"].astype(str).value_counts().to_dict()
    if active_counts != EXPECTED_ACTIVE_COUNTS:
        raise RuntimeError(f"Active-dataset counts changed: {active_counts}")
    if set(frame["combined_mode"].astype(str)) != {"count_scale"}:
        raise RuntimeError("Reviewed table is not the count_scale combination")
    if frame["coverage_calibrated"].astype(bool).any():
        raise RuntimeError("Conditional bands were mislabeled as coverage calibrated")
    if frame["scan_toy_calibrated"].astype(bool).any():
        raise RuntimeError("Mass-local toys were mislabeled as a scan ensemble")
    if set(frame["source_bands_sha256"].astype(str)) != {
        EXPECTED_SOURCE_BANDS_SHA256
    }:
        raise RuntimeError("Reviewed rows do not point to the accepted 300-toy table")

    ordered = frame.loc[:, QUANTILE_COLUMNS].to_numpy(float)
    observed = frame["eps2_obs_minimal_visible"].to_numpy(float)
    if not np.isfinite(ordered).all() or not np.isfinite(observed).all():
        raise RuntimeError("Limit figure inputs contain a non-finite value")
    if np.any(ordered <= 0.0) or np.any(observed <= 0.0):
        raise RuntimeError("Limit figure inputs contain a non-positive value")
    if np.any(np.diff(ordered, axis=1) < 0.0):
        raise RuntimeError("Toy-limit quantiles are not ordered")
    return frame


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 12.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 13.2,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.60,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 11.4,
            "ytick.labelsize": 11.4,
            "legend.fontsize": 10.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def contiguous_segments(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    categories = frame["dataset_set"].astype(str)
    groups = categories.ne(categories.shift()).cumsum()
    for _, segment in frame.groupby(groups, sort=False):
        yield segment


def plot_activity_strip(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for segment in contiguous_segments(frame):
        key = str(segment["dataset_set"].iloc[0])
        x0 = float(segment["mass_MeV"].min()) - 0.5
        x1 = float(segment["mass_MeV"].max()) + 0.5
        ax.axvspan(
            x0,
            x1,
            ymin=0.08,
            ymax=0.92,
            facecolor=ACTIVE_COLORS[key],
            edgecolor="white",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (x0 + x1),
            0.50,
            ACTIVE_LABELS[key],
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
            fontsize=10.4,
            color="#30343B",
        )
    ax.set_xlim(float(MASS_LOW_MEV), float(MASS_HIGH_MEV))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Active", rotation=0, ha="right", va="center", labelpad=10)


def make_figure(frame: pd.DataFrame) -> tuple[Path, Path]:
    setup_style()
    fig = plt.figure(figsize=(13.33, 7.15))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.16, 1.0),
        hspace=0.035,
        left=0.09,
        right=0.985,
        top=0.72,
        bottom=0.125,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    plot_activity_strip(activity, frame)

    x = frame["mass_MeV"].to_numpy(float)
    ax.fill_between(
        x,
        frame["eps2_lo2_minimal_visible"],
        frame["eps2_hi2_minimal_visible"],
        color=COLORS["band2"],
        alpha=0.76,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        x,
        frame["eps2_lo1_minimal_visible"],
        frame["eps2_hi1_minimal_visible"],
        color=COLORS["band1"],
        alpha=0.84,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        x,
        frame["eps2_med_minimal_visible"],
        color=COLORS["expected"],
        linewidth=2.0,
        linestyle="--",
        zorder=3,
    )
    ax.plot(
        x,
        frame["eps2_obs_minimal_visible"],
        color=COLORS["observed"],
        linewidth=2.45,
        zorder=4,
    )
    ax.axvline(
        DIMUON_THRESHOLD_MEV,
        color=COLORS["threshold"],
        linewidth=1.15,
        linestyle=":",
        zorder=5,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(MASS_LOW_MEV), float(MASS_HIGH_MEV))
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax.set_ylabel(
        "90% CL upper limit on\n"
        r"minimal-visible $\epsilon^2$"
    )
    ax.xaxis.set_major_locator(FixedLocator(np.arange(20.0, 251.0, 20.0)))
    ax.xaxis.set_minor_locator(MultipleLocator(10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())

    handles = [
        Patch(
            facecolor=COLORS["band2"],
            alpha=0.76,
            label="Central 95% fixed-GP toy-limit interval",
        ),
        Patch(
            facecolor=COLORS["band1"],
            alpha=0.84,
            label="Central 68% fixed-GP toy-limit interval",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["expected"],
            linewidth=2.0,
            linestyle="--",
            label="Fixed-GP toy-limit median",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["observed"],
            linewidth=2.45,
            label="Observed 90% CL",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["threshold"],
            linewidth=1.15,
            linestyle=":",
            label=rf"$2m_\mu={DIMUON_THRESHOLD_MEV:.3f}$ MeV",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.865),
        frameon=False,
        ncol=3,
        columnspacing=2.0,
        handlelength=2.8,
    )
    fig.suptitle(
        "Combined HPS observed limit and fixed-GP toy quantiles",
        y=0.968,
        fontweight="semibold",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{STEM}.pdf"
    png = FIGURE_DIR / f"{STEM}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": "Combined HPS observed limit and fixed-GP toy quantiles",
            "Author": "HPS-GPR v4.2 analysis",
            "Subject": "Presentation figure without observed-to-median subpanel",
        },
    )
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf, png


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Copied asset differs from source: {destination}")


def main() -> int:
    frame = load_reviewed_table()
    pdf, png = make_figure(frame)

    note_pdf = NOTE_FIGURE_DIR / pdf.name
    note_png = NOTE_FIGURE_DIR / png.name
    delivery_pdf = DELIVERY_DIR / pdf.name
    copy_exact(pdf, note_pdf)
    copy_exact(png, note_png)
    copy_exact(pdf, delivery_pdf)

    outputs = []
    for path in (pdf, png, note_pdf, note_png, delivery_pdf):
        outputs.append(
            {
                "path": repo_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "figure": STEM,
        "purpose": (
            "Presentation version of the v4.2 combined observed limit and "
            "conditional fixed-GP toy-limit quantiles"
        ),
        "input": {
            "path": repo_path(INPUT),
            "sha256": sha256(INPUT),
            "n_masses": int(len(frame)),
            "mass_range_MeV": [MASS_LOW_MEV, MASS_HIGH_MEV],
            "n_toys_finite_per_mass": N_TOYS,
            "source_bands_sha256": EXPECTED_SOURCE_BANDS_SHA256,
        },
        "plot_content": {
            "active_dataset_strip": True,
            "observed_90pct_limit": True,
            "fixed_gp_toy_limit_median": True,
            "central_68pct_toy_limit_interval": True,
            "central_95pct_toy_limit_interval": True,
            "dimuon_threshold": True,
            "observed_over_median_subpanel": False,
            "footer": False,
        },
        "semantic_boundary": (
            "The bands are mass-local descriptive quantiles conditional on "
            "fixed reviewed GP states; they are not coverage calibrated or "
            "a coherent scan-wide toy ensemble."
        ),
        "generator": {
            "path": repo_path(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "outputs": outputs,
    }
    atomic_write_json(payload, PROVENANCE)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
