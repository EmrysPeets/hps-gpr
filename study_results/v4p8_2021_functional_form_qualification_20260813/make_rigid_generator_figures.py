#!/usr/bin/env python3
"""Make source-fit and Figure-46-style QA for the rigid 20-toy analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
ROOT_PATH = HERE / "inputs/rigid_ggt26_scaled1pct_nested_toys_25.root"
MANIFEST_PATH = HERE / "inputs/rigid_ggt26_scaled1pct_nested_toys_25.manifest.json"
FIGURES = HERE / "figures"
PDF_OUTPUT = Path(
    "/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/"
    "output/pdf/v4p8_2021_rigid_threshold_truth_20260813"
)
SOURCES = {
    "one_pct": Path("/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root"),
    "ten_pct": Path("/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"),
}
HISTOGRAM = "preselection/h_invM_8000"
TAG = "rigid_ggt26_scaled1pct"
N_ANALYSIS_TOYS = 20
SCENARIOS = (
    ("2021_1pct_x10", r"1% $\times 10$"),
    ("2021_10pct", "native 10% (1% shape frozen)"),
    ("2021_1pct_x100", r"1% $\times 100$"),
    ("2021_10pct_x10", r"native 10% $\times 10$ (1% shape frozen)"),
)
SUPPORT = (0.040, 0.300)
SEARCH = (0.050, 0.250)


def rebin(values: np.ndarray, edges: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = values.shape[-1] // factor * factor
    if values.ndim == 1:
        rebinned = values[:usable].reshape(-1, factor).sum(axis=1)
    else:
        rebinned = values[:, :usable].reshape(values.shape[0], -1, factor).sum(axis=2)
    new_edges = edges[:usable + 1:factor]
    centers = 0.5 * (new_edges[:-1] + new_edges[1:])
    return np.asarray(rebinned), centers, new_edges


def source_values(source: str) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(SOURCES[source]) as root_file:
        values, edges = root_file[HISTOGRAM].to_numpy(flow=False)
    return np.asarray(values, dtype=float), np.asarray(edges, dtype=float)


def expected(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(ROOT_PATH) as root_file:
        values, edges = root_file[f"truth/{TAG}/{scenario}_mean"].to_numpy(flow=False)
    return np.asarray(values, dtype=float), np.asarray(edges, dtype=float)


def toys(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    edges = None
    with uproot.open(ROOT_PATH) as root_file:
        for toy in range(N_ANALYSIS_TOYS):
            values, these_edges = root_file[f"toys/{TAG}/{scenario}/toy_{toy:04d}"].to_numpy(flow=False)
            arrays.append(np.asarray(values, dtype=float))
            if edges is None:
                edges = np.asarray(these_edges, dtype=float)
            elif not np.array_equal(edges, these_edges):
                raise RuntimeError("toy edge mismatch")
    return np.asarray(arrays), np.asarray(edges)


def shade_geometry(axis: plt.Axes) -> None:
    axis.axvspan(SUPPORT[0] * 1000.0, SEARCH[0] * 1000.0, color="0.93", zorder=-10)
    axis.axvspan(SEARCH[1] * 1000.0, SUPPORT[1] * 1000.0, color="0.93", zorder=-10)
    axis.axvline(SEARCH[0] * 1000.0, color="0.55", lw=0.8, ls=":")
    axis.axvline(SEARCH[1] * 1000.0, color="0.55", lw=0.8, ls=":")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    for directory in (FIGURES, PDF_OUTPUT):
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")


def source_figure() -> None:
    rows = (
        ("one_pct", "2021_1pct", "native 1%", 6, 1.08807, 1.08818, 1.10585, 1.10325),
        ("ten_pct", "2021_10pct", "native 10% (1% shape frozen)", 1, 2.67613, 2.67565, 2.78945, 2.77290),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex="col")
    colors = ("#2B6CB0", "#B83280")
    for row, (source, scenario, label, npars, pearson, deviance, support_p, support_d) in enumerate(rows):
        observed, edges = source_values(source)
        model, model_edges = expected(scenario)
        if not np.array_equal(edges, model_edges):
            raise RuntimeError("source/model edge mismatch")
        centers = 0.5 * (edges[:-1] + edges[1:])
        keep = (centers >= SUPPORT[0]) & (centers < SUPPORT[1])
        observed5, centers5, _ = rebin(observed[keep], edges[np.r_[np.where(keep)[0], np.where(keep)[0][-1] + 1]], 5)
        model5, _, _ = rebin(model[keep], edges[np.r_[np.where(keep)[0], np.where(keep)[0][-1] + 1]], 5)
        axis = axes[row, 0]
        axis.step(centers5 * 1000.0, observed5, where="mid", color="black", lw=0.75, label="source (5-bin sum)")
        axis.plot(centers5 * 1000.0, model5, color=colors[row], lw=1.8, label="frozen sparse threshold mean")
        axis.set_yscale("log")
        axis.set_ylabel("counts / 0.625 MeV")
        axis.set_title(label)
        shade_geometry(axis)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
        axis.text(
            0.98,
            0.08,
            f"50-250 P/D = {pearson:.3f}/{deviance:.3f}\n40-300 P/D = {support_p:.3f}/{support_d:.3f}\nfree coordinates = {npars}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9),
        )
        residual = (observed[keep] - model[keep]) / np.sqrt(np.clip(model[keep], 1e-12, None))
        axis = axes[row, 1]
        axis.scatter(centers[keep] * 1000.0, residual, s=4, alpha=0.40, color=colors[row], rasterized=True)
        axis.axhline(0.0, color="black", lw=0.8)
        axis.axhline(3.0, color="0.65", lw=0.7, ls="--")
        axis.axhline(-3.0, color="0.65", lw=0.7, ls="--")
        axis.set_ylabel(r"native-bin $(n-\mu)/\sqrt{\mu}$")
        axis.set_ylim(-12.5, 12.5)
        shade_geometry(axis)
        frac = observed5 / np.clip(model5, 1e-12, None) - 1.0
        axis = axes[row, 2]
        axis.step(centers5 * 1000.0, 100.0 * frac, where="mid", color=colors[row], lw=1.0)
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_ylabel("source / mean - 1 [%]")
        axis.margins(y=0.10)
        shade_geometry(axis)
    for axis in axes[-1]:
        axis.set_xlabel("mass [MeV]")
    axes[0, 1].set_title("fine-bin residuals")
    axes[0, 2].set_title("broad residual structure (5-bin sums)")
    fig.suptitle(
        "v4.8 source-only conditional stress decision: sparse T2+T6 threshold mean, 1% shape frozen for 10%",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Primary fidelity region: 50-250 MeV; gray shoulders: GP training support. P/D values are engineering scores, not formal source-model acceptance.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    save(fig, "v4p8_rigid_source_fit_qualification")
    plt.close(fig)


def toy_figure() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.2), sharex="col")
    for row, (scenario, label) in enumerate(SCENARIOS):
        model, edges = expected(scenario)
        samples, toy_edges = toys(scenario)
        if not np.array_equal(edges, toy_edges):
            raise RuntimeError("truth/toy edge mismatch")
        centers = 0.5 * (edges[:-1] + edges[1:])
        keep = (centers >= SUPPORT[0]) & (centers < SUPPORT[1])
        support_edges = edges[np.r_[np.where(keep)[0], np.where(keep)[0][-1] + 1]]
        model20, centers20, _ = rebin(model[keep], support_edges, 20)
        toy20, _, _ = rebin(samples[:, keep], support_edges, 20)
        median = np.median(toy20, axis=0)
        mean = np.mean(toy20, axis=0)
        low, high = np.quantile(toy20, [0.16, 0.84], axis=0)

        source = "one_pct" if "1pct" in scenario else "ten_pct"
        source_values_raw, source_edges = source_values(source)
        source_centers = 0.5 * (source_edges[:-1] + source_edges[1:])
        source_keep = (source_centers >= SUPPORT[0]) & (source_centers < SUPPORT[1])
        source20, source_centers20, _ = rebin(
            source_values_raw[source_keep],
            source_edges[np.r_[np.where(source_keep)[0], np.where(source_keep)[0][-1] + 1]],
            20,
        )
        source20 = source20 * float(np.sum(model20) / np.sum(source20))
        axis = axes[row, 0]
        axis.step(source_centers20 * 1000.0, source20, where="mid", color="0.35", lw=0.8, label="scaled source")
        axis.plot(centers20 * 1000.0, model20, color="#6B46C1", lw=1.7, label="declared mean")
        axis.set_yscale("log")
        axis.set_ylabel(f"{label}\ncounts / 2.5 MeV")
        shade_geometry(axis)
        if row == 0:
            axis.legend(frameon=False, fontsize=8)

        axis = axes[row, 1]
        axis.fill_between(centers20 * 1000.0, low, high, color="#90CDF4", alpha=0.55, label="central 68% of 20")
        axis.plot(centers20 * 1000.0, model20, color="black", lw=1.1, label="analytic mean")
        axis.plot(centers20 * 1000.0, median, color="#2B6CB0", lw=0.9, ls="--", label="toy median")
        axis.plot(centers20 * 1000.0, mean, color="#C53030", lw=0.9, label="toy mean")
        axis.set_yscale("log")
        shade_geometry(axis)
        if row == 0:
            axis.legend(frameon=False, fontsize=7.5, ncol=2)

        axis = axes[row, 2]
        standardized = (mean - model20) / np.sqrt(
            np.clip(model20 / float(N_ANALYSIS_TOYS), 1e-12, None)
        )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.step(centers20 * 1000.0, standardized, where="mid", color="#C53030", lw=1.0)
        axis.set_ylim(-4.5, 4.5)
        shade_geometry(axis)
        axis.text(0.98, 0.89, f"max |r| = {np.max(np.abs(standardized)):.2f}", transform=axis.transAxes, ha="right", va="top", fontsize=8)

    axes[0, 0].set_title("source and scaled declared mean")
    axes[0, 1].set_title("20 nested-Poisson backgrounds")
    axes[0, 2].set_title(r"$(\bar n-\mu)/\sqrt{\mu/20}$")
    for axis in axes[-1]:
        axis.set_xlabel("mass [MeV]")
    fig.suptitle(
        "v4.8 conditional stress-mean sampling QA (20 backgrounds per source family)",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "All lanes use one native-1% shape; native 10% changes normalization only. The ribbon is a pointwise count range, not an expected-limit band.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.975))
    save(fig, "v4p8_rigid_toy_generation_20")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "toys", "all"), nargs="?", default="all")
    args = parser.parse_args()
    if not ROOT_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise RuntimeError("run build_rigid_toys.py build first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("generator_tag") != TAG
        or int(manifest.get("n_toys_per_source_family", -1)) < N_ANALYSIS_TOYS
    ):
        raise RuntimeError("unexpected toy manifest")
    if args.command in ("source", "all"):
        source_figure()
    if args.command in ("toys", "all"):
        toy_figure()


if __name__ == "__main__":
    main()
