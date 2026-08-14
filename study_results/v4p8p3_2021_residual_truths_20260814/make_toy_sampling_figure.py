#!/usr/bin/env python3
"""Plot exact-20 nested Poisson construction diagnostics for v4p8p3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

import build_residual_toys as toys
import residual_models as models


OUT = models.HERE / "figures"
SCENARIO_LABELS = {
    "2021_1pct": "native 1%",
    "2021_1pct_x10": "1% × 10",
    "2021_1pct_x100": "1% × 100",
    "2021_10pct": "native 10%",
    "2021_10pct_x10": "10% × 10",
}
MODEL_LABELS = {
    "knot_spline": "fixed two-knot log spline",
    "regional_blend": "three-region log blend",
}


def main() -> int:
    validation = toys.validate()
    if validation["closure_histograms"] != 200 or validation["reserve_histograms"] != 0:
        raise toys.ToyBuildError("unexpected closure inventory")
    fig, axes = plt.subplots(
        len(toys.SCENARIOS),
        len(toys.MODELS),
        figsize=(12.0, 13.0),
        sharex=True,
        sharey=True,
    )
    with uproot.open(toys.ROOT_PATH) as root_file:
        for row, scenario in enumerate(toys.SCENARIOS):
            for column, model in enumerate(toys.MODELS):
                truth, edges = root_file[f"truth/{model}/{scenario}_mean"].to_numpy(
                    flow=False
                )
                samples = np.stack(
                    [
                        root_file[
                            f"toys/{model}/{scenario}/toy_{index:04d}"
                        ].to_numpy(flow=False)[0]
                        for index in range(20)
                    ],
                    axis=0,
                )
                centers = 0.5 * (edges[:-1] + edges[1:]) * 1000.0
                mask = (centers >= 40.0) & (centers <= 300.0) & (truth > 0)
                residuals = (samples[:, mask] - truth[mask]) / np.sqrt(truth[mask])
                q16, median, q84 = np.quantile(residuals, [0.16, 0.50, 0.84], axis=0)
                mean = np.mean(residuals, axis=0)
                sample_mean_z = np.sqrt(samples.shape[0]) * mean
                axis = axes[row, column]
                axis.fill_between(
                    centers[mask],
                    q16,
                    q84,
                    color="#8bb9d6",
                    alpha=0.38,
                    lw=0,
                    label="central 68% of 20 counts" if row == 0 and column == 0 else None,
                )
                axis.plot(
                    centers[mask],
                    median,
                    color="#214f74",
                    lw=0.75,
                    alpha=0.9,
                    label="median" if row == 0 and column == 0 else None,
                )
                axis.plot(
                    centers[mask],
                    mean,
                    color="#c55a11",
                    lw=0.75,
                    alpha=0.9,
                    label="mean" if row == 0 and column == 0 else None,
                )
                axis.axhline(0.0, color="0.25", lw=0.65)
                axis.axvspan(50, 250, color="0.5", alpha=0.045, zorder=-20)
                axis.grid(alpha=0.15, lw=0.45)
                axis.text(
                    0.985,
                    0.92,
                    rf"max $|z_{{\bar n}}|={np.max(np.abs(sample_mean_z)):.2f}$",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.6,
                )
                if column == 0:
                    axis.set_ylabel(
                        SCENARIO_LABELS[scenario] + "\n" + r"$(n-\mu)/\sqrt{\mu}$"
                    )
                if row == 0:
                    axis.set_title(MODEL_LABELS[model])
                if row == len(toys.SCENARIOS) - 1:
                    axis.set_xlabel("mass [MeV]")
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center")
    fig.suptitle(
        "Exact 20-background Poisson-construction QA (five nested exposure lanes)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.006,
        "The ribbon is a pointwise count-sampling diagnostic, not an expected-limit band. Model streams are independent.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.024, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "png"):
        path = OUT / f"v4p8p3_five_lane_toy_sampling_20.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs[path.name] = models.sha256_file(path)
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "toy_root_sha256": models.sha256_file(toys.ROOT_PATH),
        "toy_manifest_sha256": models.sha256_file(toys.MANIFEST_PATH),
        "script_sha256": models.sha256_file(Path(__file__)),
        "outputs": outputs,
        "interpretation": "pointwise Poisson-construction QA; not expected bands or coverage",
    }
    models.atomic_json(OUT / "toy_sampling_figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
