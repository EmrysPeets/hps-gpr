#!/usr/bin/env python3
"""Plot the pull-blind common-ceiling pilot disposition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import residual_models as models


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived/residual_length_pilot"
OUT = HERE / "figures"
MODELS = ("knot_spline", "regional_blend")
SCENARIOS = (
    "2021_1pct",
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
SHORT = {
    "2021_1pct": "native 1%",
    "2021_1pct_x10": "1%×10",
    "2021_1pct_x100": "1%×100",
    "2021_10pct": "native 10%",
    "2021_10pct_x10": "10%×10",
}
COLORS = {"knot_spline": "#3569a8", "regional_blend": "#c55a11"}


def main() -> int:
    disposition_path = DERIVED / "common_ceiling_disposition.json"
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    if disposition["status"] != "pass" or disposition["selected_common_upper_factor"] != 25:
        raise RuntimeError("unexpected pilot disposition")
    selected = pd.read_csv(DERIVED / "selected_trajectory_ledger.csv")
    compare = pd.read_csv(DERIVED / "factor20_to25_comparison.csv")
    if len(selected) != 270 or len(compare) != 90:
        raise RuntimeError("pilot ledger cardinality mismatch")
    selected["contact"] = selected["ell_at_upper_exact"].astype(bool) | selected[
        "ell_near_upper"
    ].astype(bool)
    row_keys = [(model, scenario) for model in MODELS for scenario in SCENARIOS]
    row_labels = [
        ("K2" if model == "knot_spline" else "3-region") + " / " + SHORT[scenario]
        for model, scenario in row_keys
    ]
    factors = (15, 20, 25)
    contacts = np.zeros((len(row_keys), len(factors)), dtype=int)
    minimum_repeats = np.zeros_like(contacts)
    for i, (model, scenario) in enumerate(row_keys):
        for j, factor in enumerate(factors):
            group = selected[
                (selected.model == model)
                & (selected.scenario == scenario)
                & (selected.upper_factor == factor)
            ]
            contacts[i, j] = int(group.contact.sum())
            minimum_repeats[i, j] = int(group.top_branch_replicates.min())

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))
    image = axes[0, 0].imshow(contacts, aspect="auto", cmap="YlOrRd", vmin=0)
    axes[0, 0].set_xticks(range(3), ["factor 15", "factor 20", "factor 25"])
    axes[0, 0].set_yticks(range(len(row_labels)), row_labels, fontsize=7.7)
    axes[0, 0].set_title("Exact or near upper-length contacts (9 states/cell)")
    for i in range(contacts.shape[0]):
        for j in range(contacts.shape[1]):
            axes[0, 0].text(j, i, str(contacts[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0, 0], fraction=0.045, pad=0.025, label="contacts")

    repeat_image = axes[0, 1].imshow(
        minimum_repeats, aspect="auto", cmap="Blues", vmin=0, vmax=5
    )
    axes[0, 1].set_xticks(range(3), ["factor 15", "factor 20", "factor 25"])
    axes[0, 1].set_yticks(range(len(row_labels)), row_labels, fontsize=7.7)
    axes[0, 1].set_title("Minimum reproduced top-branch multiplicity")
    for i in range(minimum_repeats.shape[0]):
        for j in range(minimum_repeats.shape[1]):
            axes[0, 1].text(
                j, i, str(minimum_repeats[i, j]), ha="center", va="center", fontsize=8
            )
    fig.colorbar(repeat_image, ax=axes[0, 1], fraction=0.045, pad=0.025, label="repeats")

    for model in MODELS:
        group = compare[compare.model == model].sort_values(
            "abs_delta_lml_per_training_bin_20_to_25"
        )
        axes[1, 0].plot(
            np.arange(1, len(group) + 1),
            np.maximum(group.abs_delta_lml_per_training_bin_20_to_25, 1e-12),
            marker="o",
            ms=2.5,
            lw=0.8,
            color=COLORS[model],
            label="K2" if model == "knot_spline" else "3-region",
        )
        group = compare[compare.model == model].sort_values(
            "abs_delta_ell_over_sigma_x_20_to_25"
        )
        axes[1, 1].plot(
            np.arange(1, len(group) + 1),
            np.maximum(group.abs_delta_ell_over_sigma_x_20_to_25, 1e-12),
            marker="o",
            ms=2.5,
            lw=0.8,
            color=COLORS[model],
            label="K2" if model == "knot_spline" else "3-region",
        )
    axes[1, 0].axhline(0.001, color="#8b1a1a", ls="--", lw=1.0, label="factor-20 gate")
    axes[1, 1].axhline(0.01, color="#8b1a1a", ls="--", lw=1.0, label="median gate")
    axes[1, 1].axhline(0.03, color="#8b1a1a", ls=":", lw=1.0, label="95th-percentile gate")
    axes[1, 0].set_title("Ranked same-input factor 20→25 LML changes")
    axes[1, 0].set_ylabel(r"$|\Delta\mathrm{LML}|/n_{\rm train}$")
    axes[1, 1].set_title("Ranked same-input factor 20→25 length shifts")
    axes[1, 1].set_ylabel(r"$|\Delta\ell|/\sigma_x$")
    for axis in axes[1, :]:
        axis.set_xlabel("rank within model (45 states)")
        axis.set_yscale("log")
        axis.grid(alpha=0.18, lw=0.5)
        axis.legend(frameon=False, fontsize=7.5)
    observed = disposition["observed_metrics"]
    fig.suptitle(
        "Pull-blind length-ceiling pilot: factor 20 fails; common conditional ceiling = 25",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.006,
        (
            f"Factor-20 contacts={observed['factor20_exact_or_near_upper_contacts']}; "
            f"max |ΔLML|/ntrain={observed['maximum_abs_delta_lml_per_training_bin_20_to_25']:.4g}; "
            f"median/p95 |Δell|/sigma_x={observed['median_abs_delta_ell_over_sigma_x_20_to_25']:.3g}/"
            f"{observed['p95_abs_delta_ell_over_sigma_x_20_to_25']:.3g}. "
            "No pulls, amplitudes, p-values, recovery, or limits were inspected."
        ),
        ha="center",
        fontsize=8.2,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "png"):
        path = OUT / f"v4p8p3_length_ceiling_pilot.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs[path.name] = models.sha256_file(path)
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "disposition_sha256": models.sha256_file(disposition_path),
        "script_sha256": models.sha256_file(Path(__file__)),
        "outputs": outputs,
    }
    models.atomic_json(OUT / "length_pilot_figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
