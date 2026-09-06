#!/usr/bin/env python3
"""Rebuild the support-comparison figure with reader-facing language.

This is a narrowly adapted copy of ``confirmation_figure`` in
``v4p9p5_2021_gp_support_edge_optimization_20260820/make_figures.py``.
It reads the original cell summary directly and preserves the plotted arrays,
styles, confidence intervals, and output stem.  Only the figure title changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parent
SOURCE_STUDY = (
    DERIVATIVE.parent / "v4p9p5_2021_gp_support_edge_optimization_20260820"
)
INPUT_CELLS = SOURCE_STUDY / "derived" / "analysis" / "confirmation_cell_summary.csv"
INPUT_SOURCE = SOURCE_STUDY / "inputs" / "source_2021_10pct.root"
FIGURES = DERIVATIVE / "source" / "v4p9p5_support_figs"
QA = DERIVATIVE / "qa" / "reader_facing_support_figure"

# This checksum identifies the exact source table used by the accepted figure.
EXPECTED_INPUT_SHA256 = "2500ea319390bdf87bb6afefafa7e1707c19cb137c5382f86b7e7f5cdd69fbdd"
EXPECTED_SOURCE_SHA256 = "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4"
SUPPORTS = ("030_300", "032_300", "034_300", "036_300", "038_300", "040_300")
EDGES_MEV = np.array([30, 32, 34, 36, 38, 40], dtype=float)
COLORS = {
    "030_300": "#6b7280",
    "032_300": "#0072B2",
    "034_300": "#009E73",
    "036_300": "#E69F00",
    "038_300": "#D55E00",
    "040_300": "#CC79A7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plotted_array_sha256(cells: pd.DataFrame) -> str:
    """Hash the exact x, y, and asymmetric-error arrays used in every trace."""
    digest = hashlib.sha256()
    full = cells.loc[cells["cohort"] == "full_0_99"].copy()
    for strength in (0.0, 2.0, 5.0):
        for support in ("034_300", "036_300", "038_300"):
            group = full.loc[
                (full["support"] == support) & (full["inj_nsigma"] == strength)
            ]
            x = group["mass_MeV"].to_numpy(float)
            y = group["mean_pull"].to_numpy(float)
            low = y - group["mean_pull_ci90_low"].to_numpy(float)
            high = group["mean_pull_ci90_high"].to_numpy(float) - y
            for array in (x, y, low, high):
                digest.update(np.asarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def save(fig: plt.Figure, stem: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        records.append({"path": str(path.relative_to(DERIVATIVE)), "sha256": sha256_file(path)})
    plt.close(fig)
    return records


def source_figure() -> list[dict[str, str]]:
    """Render the frozen source histogram with its edge key above the axes."""
    with uproot.open(INPUT_SOURCE) as root_file:
        values, edges = root_file["preselection/h_invM_8000"].to_numpy()
    values = np.asarray(values, float)
    edges = np.asarray(edges, float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    selected = values[(centers >= 0.030) & (centers < 0.050)]
    selected_centers = centers[(centers >= 0.030) & (centers < 0.050)]
    if selected.size % 5:
        raise RuntimeError("30--50 MeV source interval is not divisible by rebin factor five")
    rebinned = selected.reshape(-1, 5).sum(axis=1)
    rebinned_centers = selected_centers.reshape(-1, 5).mean(axis=1)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.step(rebinned_centers * 1000.0, rebinned, where="mid", color="#111827", lw=1.8)
    for support, edge_mev in zip(SUPPORTS, EDGES_MEV):
        ax.axvline(
            edge_mev,
            color=COLORS[support],
            lw=1.35,
            ls="--" if support == "030_300" else "-",
            alpha=0.9,
            label=f"{edge_mev:.0f} MeV" + (" control" if edge_mev == 30 else ""),
        )
    ax.set_yscale("log")
    ax.set_xlim(29.7, 50.0)
    ax.set_xlabel(r"Invariant mass $m_{e^+e^-}$ [MeV]")
    ax.set_ylabel("Candidates per 0.625 MeV")
    ax.grid(axis="y", which="both", alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("2021 native-10% threshold spectrum and scanned GP-support edges", y=0.97)
    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=8.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.72)
    return save(fig, "source_threshold_and_support_edges")


def confirmation_figure(cells: pd.DataFrame) -> list[dict[str, str]]:
    """Plot the accepted arrays, changing only the reader-facing main title."""
    full = cells.loc[cells["cohort"] == "full_0_99"].copy()
    colors = {"034_300": "#009E73", "036_300": "#E69F00", "038_300": "#D55E00"}
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.6), sharex=True)
    for ax, strength in zip(axes, (0.0, 2.0, 5.0)):
        for support in ("034_300", "036_300", "038_300"):
            group = full.loc[(full["support"] == support) & (full["inj_nsigma"] == strength)]
            x = group["mass_MeV"].to_numpy(float)
            y = group["mean_pull"].to_numpy(float)
            yerr = np.vstack(
                [
                    y - group["mean_pull_ci90_low"].to_numpy(float),
                    group["mean_pull_ci90_high"].to_numpy(float) - y,
                ]
            )
            label = f"{int(support[:3])} MeV"
            if support == "036_300":
                label += " selected"
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=colors[support],
                marker="o",
                ms=4.4,
                lw=1.45 if support == "036_300" else 1.15,
                capsize=2.2,
                label=label,
            )
        ax.axhspan(-0.75, 0.75, color="#d1d5db", alpha=0.42, zorder=0)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhline(1.25, color="#6b7280", lw=0.8, ls=":")
        ax.axhline(-1.25, color="#6b7280", lw=0.8, ls=":")
        ax.set_ylabel("Mean pull")
        ax.set_title(f"Matched-reference injection: {strength:.0f}$\\sigma$")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Signal hypothesis mass [MeV]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=8.4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
    )
    fig.suptitle("Support comparison using 100 pseudoexperiments", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.84, hspace=0.31)
    return save(fig, "confirmation_full100_pull_means")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)

    input_sha256 = sha256_file(INPUT_CELLS)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "The source cell-summary checksum changed; refusing to rebuild a label-only derivative"
        )
    source_sha256 = sha256_file(INPUT_SOURCE)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "The source histogram checksum changed; refusing to rebuild a display-only derivative"
        )

    cells = pd.read_csv(INPUT_CELLS)
    array_sha256 = plotted_array_sha256(cells)
    products = source_figure() + confirmation_figure(cells)
    manifest = {
        "status": "pass",
        "change_scope": "Reader-facing titles and legends moved outside data regions; numerical arrays are unchanged.",
        "source_script": str(SOURCE_STUDY / "make_figures.py"),
        "source_cells": str(INPUT_CELLS),
        "source_cells_sha256": input_sha256,
        "source_histogram": str(INPUT_SOURCE),
        "source_histogram_sha256": source_sha256,
        "plotted_arrays_sha256": array_sha256,
        "products": products,
    }
    manifest_path = QA / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
