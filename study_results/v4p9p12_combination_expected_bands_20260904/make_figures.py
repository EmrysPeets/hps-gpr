#!/usr/bin/env python3
"""Make non-overlapping Brazil-band figures for a completed v4.9.12 stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURES = HERE / "figures"
INDIVIDUAL = (
    "individual_2015_full",
    "individual_2016_full",
    "individual_2021_10pct",
)
COMBINATIONS = (
    "pair_2015_2016",
    "pair_2015_2021",
    "pair_2016_2021",
    "all_2015_2016_2021",
)
LABELS = {
    "individual_2015_full": "2015 full",
    "individual_2016_full": "2016 full",
    "individual_2021_10pct": "2021 10% (optimized support)",
    "pair_2015_2016": "2015 full + 2016 full",
    "pair_2015_2021": "2015 full + 2021 10%",
    "pair_2016_2021": "2016 full + 2021 10%",
    "all_2015_2016_2021": "2015 full + 2016 full + 2021 10%",
}
YELLOW = "#F6D66A"
GREEN = "#69C779"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def handles() -> list[object]:
    return [
        Patch(facecolor=YELLOW, edgecolor="#D6A900", alpha=0.40, label=r"Central 95% expected"),
        Patch(facecolor=GREEN, edgecolor="#2D9B48", alpha=0.58, label=r"Central 68% expected"),
        Line2D([0], [0], color="black", lw=1.8, ls="--", label="Expected median"),
        Line2D([0], [0], color="#4C4C4C", lw=2.1, label="Observed 90% CL$_s$"),
    ]


def panel(ax: plt.Axes, frame: pd.DataFrame, scope: str) -> None:
    frame = frame.sort_values("mass_MeV")
    x = frame.mass_MeV.to_numpy(float)
    q025 = frame.expected_q025.to_numpy(float)
    q16 = frame.expected_q16.to_numpy(float)
    median = frame.expected_median.to_numpy(float)
    q84 = frame.expected_q84.to_numpy(float)
    q975 = frame.expected_q975.to_numpy(float)
    observed = frame.eps2_observed.to_numpy(float)
    ax.fill_between(
        x,
        q025,
        q975,
        color=YELLOW,
        edgecolor="#D6A900",
        linewidth=0.45,
        alpha=0.40,
        zorder=1,
    )
    ax.fill_between(
        x,
        q16,
        q84,
        color=GREEN,
        edgecolor="#2D9B48",
        linewidth=0.55,
        alpha=0.58,
        zorder=2,
    )
    ax.plot(x, median, color="black", lw=1.8, ls="--", zorder=3)
    ax.plot(x, observed, color="black", lw=2.15, zorder=4)
    ax.set_yscale("log")
    values = np.concatenate([q025, q975, median, observed])
    values = values[np.isfinite(values) & (values > 0.0)]
    log_values = np.log10(values)
    padding = max(0.08, 0.08 * float(np.ptp(log_values)))
    low = 10.0 ** (float(np.min(log_values)) - padding)
    high = 10.0 ** (float(np.max(log_values)) + padding)
    ax.set_ylim(low, high)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.margins(x=0.01)
    ax.set_title(LABELS[scope], loc="left", fontweight="semibold", pad=7)
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def single_scope(summary: pd.DataFrame, scope: str, target_toys: int) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    panel(ax, summary[summary.scope_key == scope], scope)
    ax.set_ylabel(r"90% CL$_s$ upper limit on $\epsilon^2$")
    fig.legend(
        handles=handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        fontsize=9.3,
    )
    ax.set_title(LABELS[scope], loc="left", fontweight="semibold", pad=38)
    fig.text(
        0.5,
        0.018,
        (
            f"{target_toys} toys per mass; pointwise, background-only, and conditional "
            "on frozen GP states."
        ),
        ha="center",
        fontsize=8.4,
        color="0.35",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.16)
    save(fig, f"all_three_expected_bands_{target_toys}toys")


def panel_grid(
    summary: pd.DataFrame,
    scopes: tuple[str, ...],
    *,
    target_toys: int,
    stem: str,
    title: str,
    shape: tuple[int, int],
) -> None:
    fig, axes = plt.subplots(*shape, figsize=(12.8, 8.6) if shape[0] == 2 else (15.2, 5.4))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax, scope in zip(axes_array, scopes):
        panel(ax, summary[summary.scope_key == scope], scope)
    for ax in axes_array[len(scopes):]:
        ax.set_visible(False)
    fig.supylabel(r"90% CL$_s$ upper limit on $\epsilon^2$", x=0.018)
    fig.suptitle(title, x=0.5, y=0.99, ha="center", fontweight="semibold", fontsize=14)
    fig.legend(
        handles=handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=9.1,
    )
    fig.text(
        0.5,
        0.012,
        (
            f"{target_toys} toys per mass. Outer quantiles are provisional at this stage; "
            "bands are pointwise and conditional on frozen GP states."
        ),
        ha="center",
        fontsize=8.3,
        color="0.35",
    )
    if shape[0] == 2:
        fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.10, hspace=0.31, wspace=0.20)
    else:
        fig.subplots_adjust(left=0.06, right=0.985, top=0.76, bottom=0.17, wspace=0.22)
    save(fig, f"{stem}_{target_toys}toys")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    source = HERE / "derived" / f"expected_band_summary_{target_toys}toys.csv"
    if not source.is_file():
        raise SystemExit(f"missing completed stage summary: {source}")
    summary = pd.read_csv(source)
    expected_scopes = set(INDIVIDUAL + COMBINATIONS)
    if set(summary.scope_key.astype(str)) != expected_scopes:
        raise RuntimeError("summary does not contain exactly the seven final scopes")
    if set(summary.n_toys.astype(int)) != {target_toys}:
        raise RuntimeError("summary toy count does not match the requested figure stage")
    if not (
        summary.loc[
            summary.scope_key == "individual_2021_10pct", "scope_label"
        ]
        .astype(str)
        .str.contains("2021 10%", regex=False)
        .all()
    ):
        raise RuntimeError("optimized 2021 10% scope is missing")

    style()
    single_scope(summary, "all_2015_2016_2021", target_toys)
    panel_grid(
        summary,
        INDIVIDUAL,
        target_toys=target_toys,
        stem="individual_expected_band_panels",
        title="Standalone final-sample expected bands",
        shape=(1, 3),
    )
    panel_grid(
        summary,
        COMBINATIONS,
        target_toys=target_toys,
        stem="combination_expected_band_panels",
        title=r"Shared-$\epsilon^2$ combination expected bands",
        shape=(2, 2),
    )
    inventory = {
        "stage_toys_per_mass": target_toys,
        "source_summary": str(source.relative_to(REPO)),
        "source_summary_sha256": sha256(source),
        "figures": [
            f"all_three_expected_bands_{target_toys}toys",
            f"individual_expected_band_panels_{target_toys}toys",
            f"combination_expected_band_panels_{target_toys}toys",
        ],
        "style_reference": (
            "v4.2/v4.5 Brazil-band convention: yellow central 95%, green central "
            "68%, dashed black median, solid observed curve"
        ),
        "layout": "one curve family per axis; figure-level legends outside data regions",
        "claim_boundary": (
            "Conditional pointwise expected-limit quantiles; not coverage or scan-global calibration."
        ),
    }
    (FIGURES / f"figure_manifest_{target_toys}toys.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
