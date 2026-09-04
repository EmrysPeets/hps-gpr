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


HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parent
SOURCE_STUDY = (
    DERIVATIVE.parent / "v4p9p5_2021_gp_support_edge_optimization_20260820"
)
INPUT_CELLS = SOURCE_STUDY / "derived" / "analysis" / "confirmation_cell_summary.csv"
FIGURES = DERIVATIVE / "source" / "v4p9p5_support_figs"
QA = DERIVATIVE / "qa" / "reader_facing_support_figure"

# This checksum identifies the exact source table used by the accepted figure.
EXPECTED_INPUT_SHA256 = "2500ea319390bdf87bb6afefafa7e1707c19cb137c5382f86b7e7f5cdd69fbdd"


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


def confirmation_figure(cells: pd.DataFrame) -> list[dict[str, str]]:
    """Plot the accepted arrays, changing only the reader-facing main title."""
    full = cells.loc[cells["cohort"] == "full_0_99"].copy()
    colors = {"034_300": "#009E73", "036_300": "#E69F00", "038_300": "#D55E00"}
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True, constrained_layout=True)
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
    axes[0].legend(ncol=3, fontsize=8.4, frameon=False, loc="upper right")
    fig.suptitle("Support comparison using 100 pseudoexperiments", fontsize=13)
    return save(fig, "confirmation_full100_pull_means")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)

    input_sha256 = sha256_file(INPUT_CELLS)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "The source cell-summary checksum changed; refusing to rebuild a label-only derivative"
        )

    cells = pd.read_csv(INPUT_CELLS)
    array_sha256 = plotted_array_sha256(cells)
    products = confirmation_figure(cells)
    manifest = {
        "status": "pass",
        "change_scope": "Figure title only; numerical arrays and plotting choices are unchanged.",
        "source_script": str(SOURCE_STUDY / "make_figures.py"),
        "source_cells": str(INPUT_CELLS),
        "source_cells_sha256": input_sha256,
        "plotted_arrays_sha256": array_sha256,
        "products": products,
    }
    manifest_path = QA / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
