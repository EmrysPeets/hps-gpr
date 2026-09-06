#!/usr/bin/env python3
"""Render source-fit qualification and the blocked stress-toy figure scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.lines import Line2D
from numpy.polynomial.chebyshev import chebvander
from scipy.special import expit


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "derived/generator_qualification.json"
FIGURES = HERE / "figures"
SOURCES = {
    "one_pct": (
        Path("/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root"),
        "2021 1% source",
        "#0072B2",
    ),
    "ten_pct": (
        Path("/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"),
        "2021 native 10% source",
        "#D55E00",
    ),
}
REJECTED_SIGPOWEXPQ = {
    "one_pct": (
        HERE / "quarantine/rejected_fsigpowexpq_prototype/inputs/funcform_seed_2021_1pct_support040_300.root",
        1.572,
    ),
    "ten_pct": (
        HERE / "quarantine/rejected_fsigpowexpq_prototype/inputs/funcform_seed_2021_10pct_support040_300.root",
        6.167,
    ),
}
HISTOGRAM = "preselection/h_invM_8000"
DEGREE = 18
LO, HI = 0.030, 0.300


def d18_record(ledger: dict, source: str) -> dict:
    records = ledger["sources"][source]["records"]
    return next(record for record in records if int(record["degree"]) == DEGREE)


def mean_from_record(centers: np.ndarray, record: dict) -> np.ndarray:
    mapped = 2.0 * (centers - LO) / (HI - LO) - 1.0
    matrix = chebvander(mapped, DEGREE)
    turn = expit((centers - float(record["turn_on_gev"])) / float(record["width_gev"]))
    return np.exp(matrix @ np.asarray(record["coefficients"]) + np.log(turn))


def make_source_qualification() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(
        2, 3, figsize=(13.5, 7.2),
        gridspec_kw={"width_ratios": [1.45, 1.05, 1.15]},
        constrained_layout=True,
    )
    for row, (source, (path, label, color)) in enumerate(SOURCES.items()):
        with uproot.open(path) as root_file:
            observed, edges = root_file[HISTOGRAM].to_numpy(flow=False)
        observed = np.asarray(observed, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = (centers >= LO) & (centers < HI)
        centers = centers[mask]
        observed = observed[mask]
        record = d18_record(ledger, source)
        expected = mean_from_record(centers, record)

        ax = axes[row, 0]
        ax.step(centers * 1000, observed, where="mid", color="0.55", lw=0.6, label="source counts")
        ax.plot(centers * 1000, expected, color=color, lw=1.8, label="d18 in-sample fit")
        rejected_path, rejected_pearson = REJECTED_SIGPOWEXPQ[source]
        with uproot.open(rejected_path) as rejected_file:
            rejected_values, rejected_edges = rejected_file[
                "fSigPowExpQ/fSigPowExpQ_analytic_seed_lumi_scaled"
            ].to_numpy(flow=False)
        rejected_centers = 0.5 * (rejected_edges[:-1] + rejected_edges[1:])
        rejected_mask = (rejected_centers >= 0.040) & (rejected_centers < HI)
        ax.plot(
            rejected_centers[rejected_mask] * 1000,
            np.asarray(rejected_values)[rejected_mask],
            color="#CC79A7", lw=1.1, ls="--",
            label=f"rejected fSigPowExpQ (Pearson/ndf {rejected_pearson:.3f})",
        )
        ax.set_yscale("log")
        ax.set_xlim(30, 300)
        ax.set_ylabel(f"{label}\ncounts / 0.125 MeV")
        for edge in (30, 40, 50, 250, 300):
            ax.axvline(edge, color="0.75", lw=0.7, ls=":" if edge not in (40, 300) else "--")
        if row == 0:
            ax.set_title("Source and unqualified analytic candidate")
            ax.legend(frameon=False, fontsize=7.5)

        ax = axes[row, 1]
        residual = (observed - expected) / np.sqrt(np.clip(expected, 1.0, None))
        ax.plot(centers * 1000, residual, color=color, lw=0.55)
        ax.axhline(0, color="0.2", lw=0.8)
        ax.axhline(3, color="0.7", lw=0.7, ls=":")
        ax.axhline(-3, color="0.7", lw=0.7, ls=":")
        ax.set_xlim(30, 300)
        ax.set_ylim(-8, 8)
        ax.set_ylabel(r"$(N-\lambda)/\sqrt{\lambda}$")
        if row == 0:
            ax.set_title("Native-bin source residuals")

        ax = axes[row, 2]
        gaps = record["fake_gaps"]
        masses = np.array([item["mass_gev"] for item in gaps]) * 1000
        values = np.array([
            item["delta_model_diagonal_poisson_projection_sigma"] for item in gaps
        ])
        ax.axhspan(-0.2, 0.2, color="#009E73", alpha=0.16, label="promotion budget")
        ax.axhline(0, color="0.25", lw=0.8)
        ax.plot(masses, values, "o-", color=color, lw=1.3, ms=4)
        ax.set_xticks(masses)
        ax.set_ylabel("full vs gap model shift\n(diagonal-Poisson sigma)")
        cv = float(record["blocked_cv"]["deviance_per_bin"])
        maximum = float(np.max(np.abs(values)))
        ax.text(
            0.03, 0.96,
            f"blocked CV D/bin = {cv:.3f}  (gate <= 1.25)\nmax |gap shift| = {maximum:.3f}  (gate <= 0.20)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        if row == 0:
            ax.set_title("Resolution-window prediction stability")

    for ax in axes[-1, :]:
        ax.set_xlabel("invariant mass [MeV]")
    fig.suptitle(
        "v4.8 reconnaissance: broad in-sample screen passes, predictive qualification fails",
        fontsize=14,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"v4p8_source_generator_qualification_failed.{suffix}", dpi=180)
    plt.close(fig)


def make_blocked_scaffold() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.axis("off")
    lines = [
        "Requested four-lane 25-toy study for the rejected nominal candidate - intentionally not run",
        "",
        "The fSigPowExpQ source generator was rejected (native-10% Pearson chi2/ndf about 6.17; bound contact).",
        "The provisional degree-18 replacement passes a broad in-sample engineering screen but fails predictive gates:",
        "  1% source: blocked-CV deviance/bin 18.107; max fake-gap shift 2.679 sigma",
        "  native 10%: blocked-CV deviance/bin 24.045; max fake-gap shift 9.889 sigma",
        "  required: <= 1.25 and <= 0.20 sigma",
        "",
        "Its current numerical fit also lacks the required multistart/stationarity certificate.",
        "For this rejected nominal branch, toy generation and Figures 48/136 remain blocked.",
        "The later sparse conditional-stress branch is separate and has no card/bound-selection authority.",
        "This is a fail-closed scientific result, not a missing plotting job.",
    ]
    ax.text(
        0.04, 0.94, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=11, linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#fff7e6", "edgecolor": "#D55E00"},
    )
    handles = [
        Line2D([], [], color="#D55E00", lw=6, label="promotion_gate_passed = false"),
        Line2D([], [], color="#0072B2", lw=6, label="frozen v4.2 extraction card unchanged"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.05), frameon=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"v4p8_requested_toy_figures_blocked.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "source", "blocked"), default="all", nargs="?")
    args = parser.parse_args()
    if args.command in ("all", "source"):
        make_source_qualification()
    if args.command in ("all", "blocked"):
        make_blocked_scaffold()


if __name__ == "__main__":
    main()
