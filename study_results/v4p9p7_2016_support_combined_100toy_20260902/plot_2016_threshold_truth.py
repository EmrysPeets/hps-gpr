#!/usr/bin/env python3
"""Render the frozen 2016 threshold-truth construction and source residuals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
FIGURES = HERE / "figures"
FIT_SUMMARY = HERE / "reference" / "2016_threshold_truth_fit_summary.json"
TRUTH_ROOT = INPUTS / "2016_threshold_qualified_background_toys_100.root"
SOURCE_ROOT = INPUTS / "source_2016_10pct.root"
HISTOGRAM = "h_Minv_General_Final_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebin(values: np.ndarray, factor: int) -> np.ndarray:
    if values.size % factor:
        raise RuntimeError("histogram length is not divisible by display rebin")
    return values.reshape(-1, factor).sum(axis=1)


def main() -> None:
    summary = json.loads(FIT_SUMMARY.read_text(encoding="utf-8"))
    with uproot.open(SOURCE_ROOT) as root:
        source, edges = root[HISTOGRAM].to_numpy()
    with uproot.open(TRUTH_ROOT) as root:
        full_mean, truth_edges = root[
            "truth/threshold_qualified/2016_full_mean"
        ].to_numpy()
        local, local_edges = root[
            "truth/local_threshold_fit/2016_10pct_mean"
        ].to_numpy()
        broad, broad_edges = root[
            "truth/broad_tail_baseline/2016_10pct_mean"
        ].to_numpy()
    if not (
        np.array_equal(edges, truth_edges)
        and np.array_equal(edges, local_edges)
        and np.array_equal(edges, broad_edges)
    ):
        raise RuntimeError("histogram edge mismatch")

    scale = float(summary["final_blend_normalization"])
    source_truth = full_mean / scale
    factor = 5
    source5 = rebin(source, factor)
    truth5 = rebin(source_truth, factor)
    local5 = rebin(local, factor)
    broad5 = rebin(broad, factor)
    edges5 = edges[::factor]
    if edges5.size != source5.size + 1:
        edges5 = np.r_[edges5, edges[-1]]
    centers5 = 0.5 * (edges5[:-1] + edges5[1:])
    pulls5 = (source5 - truth5) / np.sqrt(np.maximum(truth5, 1.0))

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.25, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.05},
    )
    ax, residual = axes
    view = (centers5 >= 0.026) & (centers5 <= 0.100)
    local_view = view & (centers5 < 0.080)
    broad_view = view & (centers5 >= 0.070)
    ax.errorbar(
        1.0e3 * centers5[view],
        source5[view],
        yerr=np.sqrt(np.maximum(source5[view], 1.0)),
        fmt=".",
        ms=2.4,
        lw=0.45,
        color="#282828",
        alpha=0.72,
        label="2016 10% development spectrum",
        zorder=2,
    )
    ax.plot(
        1.0e3 * centers5[view],
        truth5[view],
        color="#0072B2",
        lw=1.7,
        label="frozen blended source truth",
        zorder=4,
    )
    ax.plot(
        1.0e3 * centers5[local_view],
        local5[local_view],
        color="#D55E00",
        lw=1.0,
        ls="--",
        label="degree-5 threshold component",
        zorder=3,
    )
    ax.plot(
        1.0e3 * centers5[broad_view],
        broad5[broad_view],
        color="#009E73",
        lw=1.0,
        ls=":",
        label="broad-tail component",
        zorder=3,
    )
    ax.axvspan(75.0, 85.0, color="#0072B2", alpha=0.08, label="C$^2$ blend")
    for edge in range(28, 35):
        ax.axvline(edge, color="#8c8c8c", lw=0.45, alpha=0.55)
    ax.axvline(39.0, color="#CC79A7", lw=1.2, label="39 MeV search threshold")
    ax.set_yscale("log")
    ax.set_ylabel("events / 0.25 MeV")
    ax.set_ylim(bottom=max(5.0, 0.7 * np.min(truth5[view])))
    ax.grid(alpha=0.18, which="both")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.set_title("Frozen 2016 threshold stress truth (source units)", loc="left")

    local_residual = (centers5 >= 0.026) & (centers5 < 0.080)
    residual.axhspan(-2.0, 2.0, color="#0072B2", alpha=0.08)
    residual.axhline(0.0, color="#333333", lw=0.8)
    residual.plot(
        1.0e3 * centers5[local_residual],
        pulls5[local_residual],
        ".",
        ms=2.7,
        color="#282828",
        alpha=0.8,
    )
    residual.axvline(39.0, color="#CC79A7", lw=1.2)
    residual.set_xlabel("invariant mass [MeV]")
    residual.set_ylabel("Pearson pull")
    residual.set_xlim(26.0, 100.0)
    residual.set_ylim(-5.25, 5.25)
    residual.grid(alpha=0.18)
    selected = next(
        row
        for row in summary["candidate_degrees"]
        if row["degree"] == summary["selected_degree"]
    )
    residual.text(
        0.99,
        0.06,
        (
            rf"degree {summary['selected_degree']}; "
            rf"$D/N_{{\rm dof}}={selected['deviance_ndf']:.3f}$; "
            rf"five-bin $D/N_{{\rm dof}}={selected['rebin5_deviance_ndf']:.3f}$"
        ),
        transform=residual.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
    )
    fig.text(
        0.01,
        0.005,
        "Source-conditioned diagnostic; not a physical background generator or coverage model.",
        fontsize=7.6,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.115, right=0.985, top=0.94, bottom=0.13, hspace=0.05
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "2016_threshold_truth_and_residuals.pdf"
    png = FIGURES / "2016_threshold_truth_and_residuals.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "pass",
        "pdf": str(pdf.relative_to(HERE)),
        "pdf_sha256": sha256(pdf),
        "png": str(png.relative_to(HERE)),
        "png_sha256": sha256(png),
        "truth_root_sha256": sha256(TRUTH_ROOT),
        "fit_summary_sha256": sha256(FIT_SUMMARY),
        "display_rebin": factor,
        "claim_boundary": (
            "Source-conditioned diagnostic; not a physical background "
            "generator or coverage model."
        ),
    }
    output = HERE / "qa" / "2016_threshold_truth_figure.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
