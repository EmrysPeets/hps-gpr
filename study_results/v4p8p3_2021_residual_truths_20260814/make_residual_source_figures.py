#!/usr/bin/env python3
"""Make source-qualification and signal-influence figures for v4p8p3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import residual_models as models


OUT = models.HERE / "figures"
COLORS = {"knot_spline": "#3569a8", "regional_blend": "#c55a11"}
LABELS = {
    "knot_spline": "fixed two-knot log spline",
    "regional_blend": "three-region log blend",
}


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def aggregate(values: np.ndarray, centers: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    usable = (values.size // factor) * factor
    return (
        values[:usable].reshape(-1, factor).sum(axis=1),
        centers[:usable].reshape(-1, factor).mean(axis=1),
    )


def source_figure(result: dict) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.0, 9.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.15, 0.8, 0.8]},
    )
    for column, source in enumerate(("one_pct", "ten_pct")):
        histogram = models.load_histogram(source)
        mask = histogram.support_mask
        centers = histogram.centers[mask] * 1000.0
        observed = histogram.values[mask]
        axes[0, column].step(
            centers,
            observed,
            where="mid",
            color="0.25",
            lw=0.75,
            alpha=0.75,
            label="source histogram",
        )
        for model in ("knot_spline", "regional_blend"):
            expected = models.evaluate_frozen_support(model, source, histogram, result)
            axes[0, column].plot(
                centers,
                expected,
                color=COLORS[model],
                lw=1.45,
                label=LABELS[model],
            )
            native = (observed - expected) / np.sqrt(expected)
            axes[1, column].plot(
                centers,
                native,
                color=COLORS[model],
                lw=0.55,
                alpha=0.82,
            )
            obs5, centers5 = aggregate(observed, centers, 5)
            exp5, _ = aggregate(expected, centers, 5)
            axes[2, column].plot(
                centers5,
                (obs5 - exp5) / np.sqrt(exp5),
                color=COLORS[model],
                lw=0.9,
            )
        axes[0, column].set_yscale("log")
        axes[0, column].set_title("Native 1% source" if source == "one_pct" else "Native 10% source")
        axes[0, column].set_ylabel("counts / native bin")
        axes[1, column].set_ylabel("native Pearson residual")
        axes[2, column].set_ylabel("rebin-5 Pearson residual")
        axes[2, column].set_xlabel("mass [MeV]")
        for row in range(3):
            axes[row, column].axvspan(50, 250, color="0.5", alpha=0.055, zorder=-20)
            axes[row, column].grid(alpha=0.16, lw=0.5)
            axes[row, column].set_xlim(40, 300)
        for row in (1, 2):
            axes[row, column].axhline(0.0, color="0.35", lw=0.7)
        # Fixed structures are shown without implying fitted freedom in the audit.
        for knot in (105, 180):
            axes[0, column].axvline(knot, color=COLORS["knot_spline"], ls=":", lw=0.8)
        for low, high in ((85, 125), (165, 215)):
            axes[0, column].axvspan(low, high, color=COLORS["regional_blend"], alpha=0.055)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Residual-structured source means: support-wide qualification fails for both models",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.006,
        "Shaded center: 50–250 MeV primary region. Blue dotted lines: frozen K2 nodes. Orange bands: frozen blend overlaps.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    save(fig, "v4p8p3_source_qualification_and_residuals")


def influence_figure(result: dict) -> None:
    audit = result["signal_influence_audit"]
    rows = audit["rows"]
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), sharex=True)
    for column, source in enumerate(("one_pct", "ten_pct")):
        for model in ("knot_spline", "regional_blend"):
            selected = [row for row in rows if row["source"] == source and row["model"] == model]
            masses = np.asarray(sorted({float(row["mass_MeV"]) for row in selected}))
            gap = []
            tangent = []
            nonlinear = []
            for mass in masses:
                at_mass = [row for row in selected if abs(float(row["mass_MeV"]) - mass) < 1e-7]
                finite_gap = [row["gap_abs_shift_sigmaA"] for row in at_mass if row["gap_abs_shift_sigmaA"] is not None]
                finite_refit = [row["z_times_abs_absorption"] for row in at_mass if row["z_times_abs_absorption"] is not None]
                gap.append(max(finite_gap) if finite_gap else np.nan)
                tangent.append(max(abs(float(row["tangent_absorption_fraction"])) for row in at_mass))
                nonlinear.append(max(finite_refit) if finite_refit else np.nan)
            axes[0, column].plot(masses, gap, marker="o", ms=2.7, lw=1.0, color=COLORS[model], label=LABELS[model])
            axes[1, column].plot(masses, tangent, marker="o", ms=2.7, lw=1.0, color=COLORS[model])
            axes[2, column].plot(masses, nonlinear, marker="o", ms=2.7, lw=1.0, color=COLORS[model])
            failed = [row for row in selected if row["refit_failure"] is not None]
            if failed:
                axes[2, column].scatter(
                    [row["mass_MeV"] for row in failed],
                    [0.015] * len(failed),
                    marker="x",
                    s=34,
                    color=COLORS[model],
                    zorder=10,
                    label="failed constrained refit" if column == 1 else None,
                )
        axes[0, column].axhline(0.20, color="#8b1a1a", ls="--", lw=1.0, label="predeclared gate")
        axes[1, column].axhline(0.04, color="#8b1a1a", ls="--", lw=1.0)
        axes[2, column].axhline(0.20, color="#8b1a1a", ls="--", lw=1.0)
        axes[0, column].set_title("Native 1% source refit" if source == "one_pct" else "Native 10% transfer refit")
        axes[0, column].set_ylabel(r"max gap $|\Delta A|/\sigma_A$")
        axes[1, column].set_ylabel("conservative tangent fraction")
        axes[2, column].set_ylabel(r"max $z|f_{\rm abs}|$")
        axes[2, column].set_xlabel("mass [MeV]")
        for row in range(3):
            axes[row, column].set_yscale("log")
            axes[row, column].set_xlim(50, 250)
            axes[row, column].grid(alpha=0.18, lw=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    handles, labels = axes[2, 1].get_legend_handles_labels()
    if handles:
        axes[2, 1].legend(handles, labels, frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Signal-influence audit: both residual-structured forms fail all rigidity metrics",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "v4p8p3_signal_influence_audit")


def main() -> int:
    result = models.load_fit_result(require_influence=True)
    if any(result["models"][name]["strict_generator_qualification_passed"] for name in ("knot_spline", "regional_blend")):
        raise models.ModelError("figure disposition unexpectedly sees a qualified model")
    source_figure(result)
    influence_figure(result)
    manifest = {
        "schema_version": 1,
        "result_sha256": models.sha256_file(models.FIT_RESULT_PATH),
        "script_sha256": models.sha256_file(Path(__file__)),
        "figures": {
            path.name: models.sha256_file(path)
            for path in sorted(OUT.glob("v4p8p3_*"))
            if path.suffix in {".pdf", ".png"}
        },
    }
    models.atomic_json(OUT / "source_influence_figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
