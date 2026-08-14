#!/usr/bin/env python3
"""Plot the post-closure, pull-blind K2/native-1% ceiling remediation."""

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
DERIVED = HERE / "derived/residual_length_ceiling_remediation"
OUT = HERE / "figures"
MASS_COLORS = {65: "#3569a8", 120: "#6a51a3", 210: "#c55a11"}


def read_stage(stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = DERIVED / stage
    trajectory = pd.read_csv(directory / "selected_trajectory_ledger.csv")
    gate = pd.read_csv(directory / "candidate_mass_gate.csv")
    expected_toys = 3 if stage == "selection" else 5
    expected_trajectory = expected_toys * 3 * 4
    if len(trajectory) != expected_trajectory:
        raise RuntimeError(f"{stage} trajectory cardinality mismatch")
    if set(trajectory["mass_MeV"]) != {65, 120, 210}:
        raise RuntimeError(f"{stage} mass lattice mismatch")
    if not trajectory["background_only"].astype(bool).all():
        raise RuntimeError(f"{stage} includes a non-background state")
    if trajectory["inference_quantities_inspected"].astype(bool).any():
        raise RuntimeError(f"{stage} inspected an inference quantity")
    return trajectory, gate


def plot_trajectories(axis, frame: pd.DataFrame, title: str) -> None:
    for mass in (65, 120, 210):
        group = frame[frame.mass_MeV == mass]
        for toy, toy_rows in group.groupby("background_toy_index"):
            toy_rows = toy_rows.sort_values("upper_factor")
            axis.plot(
                toy_rows.upper_factor,
                toy_rows.ell_over_sigma_x,
                color=MASS_COLORS[mass],
                alpha=0.58,
                lw=1.1,
                marker="o",
                ms=3.0,
                label=f"{mass} MeV" if toy == group.background_toy_index.min() else None,
            )
            contacts = toy_rows[
                toy_rows.ell_at_upper_exact.astype(bool)
                | toy_rows.ell_near_upper.astype(bool)
            ]
            if len(contacts):
                axis.scatter(
                    contacts.upper_factor,
                    contacts.ell_over_sigma_x,
                    marker="x",
                    s=52,
                    lw=1.5,
                    color="#8b1a1a",
                    zorder=5,
                )
    factors = sorted(frame.upper_factor.unique())
    axis.plot(factors, factors, color="#555555", ls="--", lw=0.9, label="upper bound")
    axis.axvline(50, color="#187a3b", ls=":", lw=1.2, label="selected factor 50")
    axis.set_title(title)
    axis.set_xlabel("upper length factor")
    axis.set_ylabel(r"fitted $\ell/\sigma_x$")
    axis.set_xticks(factors)
    axis.grid(alpha=0.18, lw=0.5)


def plot_lml_gate(axis, selection: pd.DataFrame, confirmation: pd.DataFrame) -> None:
    styles = (
        (selection[(selection.candidate == 35) & (selection.sentinel == 50)], "35→50 select", "o", "#8b1a1a"),
        (selection[(selection.candidate == 50) & (selection.sentinel == 75)], "50→75 select", "s", "#3569a8"),
        (confirmation, "50→75 confirm", "D", "#187a3b"),
    )
    for frame, label, marker, color in styles:
        axis.plot(
            frame.mass_MeV,
            frame.maximum_abs_delta_lml_per_training_bin,
            marker=marker,
            color=color,
            lw=1.2,
            ms=5,
            label=label,
        )
    axis.axhline(1e-3, color="#333333", ls="--", lw=1.0, label="gate 0.001")
    axis.set_yscale("log")
    axis.set_xticks([65, 120, 210])
    axis.set_xlabel("mass [MeV]")
    axis.set_ylabel(r"max $|\Delta\mathrm{LML}|/n_{\rm train}$")
    axis.set_title("Same-input likelihood plateau")
    axis.grid(alpha=0.18, lw=0.5)


def plot_length_gate(axis, selection: pd.DataFrame, confirmation: pd.DataFrame) -> None:
    styles = (
        (selection[(selection.candidate == 35) & (selection.sentinel == 50)], "35→50 select", "o", "#8b1a1a"),
        (selection[(selection.candidate == 50) & (selection.sentinel == 75)], "50→75 select", "s", "#3569a8"),
        (confirmation, "50→75 confirm", "D", "#187a3b"),
    )
    for frame, label, marker, color in styles:
        axis.plot(
            frame.mass_MeV,
            frame.p95_abs_delta_ell_over_sigma_x,
            marker=marker,
            color=color,
            lw=1.2,
            ms=5,
            label=f"{label}: p95",
        )
        axis.plot(
            frame.mass_MeV,
            frame.maximum_abs_delta_ell_over_sigma_x,
            marker=marker,
            color=color,
            lw=0.9,
            ls=":",
            ms=4,
            alpha=0.78,
            label=f"{label}: max",
        )
    axis.axhline(0.03, color="#333333", ls="--", lw=1.0, label="p95 gate 0.03")
    axis.axhline(0.05, color="#333333", ls=":", lw=1.0, label="max gate 0.05")
    axis.set_yscale("log")
    axis.set_xticks([65, 120, 210])
    axis.set_xlabel("mass [MeV]")
    axis.set_ylabel(r"$|\Delta\ell|/\sigma_x$")
    axis.set_title("Same-input length-coordinate plateau")
    axis.grid(alpha=0.18, lw=0.5)


def main() -> int:
    selection_trajectory, selection_gate = read_stage("selection")
    confirmation_trajectory, confirmation_gate = read_stage("confirmation")
    selection_disposition_path = DERIVED / "selection/selection_disposition.json"
    confirmation_disposition_path = DERIVED / "confirmation/final_disposition.json"
    selection_disposition = json.loads(selection_disposition_path.read_text(encoding="utf-8"))
    confirmation_disposition = json.loads(confirmation_disposition_path.read_text(encoding="utf-8"))
    if selection_disposition["selected_candidate"] != 50:
        raise RuntimeError("unexpected selected candidate")
    if confirmation_disposition["status"] != "qualified_targeted":
        raise RuntimeError("targeted confirmation has not passed")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))
    plot_trajectories(
        axes[0, 0],
        selection_trajectory,
        "Selection: three fresh backgrounds per mass",
    )
    plot_trajectories(
        axes[0, 1],
        confirmation_trajectory,
        "Confirmation: five fresh backgrounds per mass",
    )
    plot_lml_gate(axes[1, 0], selection_gate, confirmation_gate)
    plot_length_gate(axes[1, 1], selection_gate, confirmation_gate)
    axes[0, 0].legend(frameon=False, fontsize=7.4, ncol=2)
    axes[0, 1].legend(frameon=False, fontsize=7.4, ncol=2)
    axes[1, 0].legend(frameon=False, fontsize=7.2)
    axes[1, 1].legend(frameon=False, fontsize=6.6, ncol=2)
    fig.suptitle(
        "Post-closure pull-blind ceiling remediation: K2 native 1% only",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.006,
        (
            "Factor 35 fails at 120 MeV. Factor 50 versus the factor-75 sentinel passes "
            "all frozen gates in selection and confirmation. This does not re-evaluate "
            "the factor-25 closure or qualify a common all-lane ceiling."
        ),
        ha="center",
        fontsize=8.3,
    )
    fig.tight_layout(rect=(0, 0.038, 1, 0.965))

    OUT.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for suffix in ("pdf", "png"):
        path = OUT / f"v4p8p3_length_ceiling_remediation.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs[path.name] = models.sha256_file(path)
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "script_sha256": models.sha256_file(Path(__file__)),
        "selection_disposition_sha256": models.sha256_file(selection_disposition_path),
        "confirmation_disposition_sha256": models.sha256_file(confirmation_disposition_path),
        "outputs": outputs,
        "claim_boundary": confirmation_disposition["claim_boundary"],
    }
    models.atomic_json(OUT / "ceiling_remediation_figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
