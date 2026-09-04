#!/usr/bin/env python3
"""Make v4.9.7 2016 support-selection figures after both frozen phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "derived" / "analysis"
FIGURES = HERE / "figures"
SUPPORTS = tuple(f"{edge:03d}_210" for edge in range(28, 35))
EDGES = np.arange(28.0, 35.0)
COLORS = {
    support: color
    for support, color in zip(
        SUPPORTS,
        ("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499"),
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: plt.Figure, stem: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        records.append(
            {"path": str(path.relative_to(HERE)), "sha256": sha256(path)}
        )
    plt.close(fig)
    return records


def label(support: str, *, selected: str | None = None) -> str:
    value = f"{int(support[:3])} MeV"
    if support == "034_210":
        value += " control"
    if support == selected:
        value += " selected"
    return value


def phase1_pulls(cells: pd.DataFrame) -> list[dict[str, object]]:
    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 9.2), sharex=True, constrained_layout=True
    )
    for ax, strength in zip(axes, (0.0, 2.0, 5.0)):
        for support in SUPPORTS:
            group = cells.loc[
                (cells["support"] == support)
                & np.isclose(cells["inj_nsigma"], strength)
            ].sort_values("mass_MeV")
            x = group["mass_MeV"].to_numpy(float)
            y = group["mean_pull"].to_numpy(float)
            yerr = np.vstack(
                [
                    y - group["mean_pull_ci90_low"].to_numpy(float),
                    group["mean_pull_ci90_high"].to_numpy(float) - y,
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[support],
                marker="o",
                ms=3.8,
                lw=1.1,
                capsize=2.0,
                ls="--" if support == "034_210" else "-",
                label=label(support),
            )
        ax.axhspan(-0.75, 0.75, color="#d1d5db", alpha=0.38, zorder=0)
        ax.axhline(0.0, color="black", lw=0.75)
        ax.axhline(1.25, color="#6b7280", lw=0.75, ls=":")
        ax.axhline(-1.25, color="#6b7280", lw=0.75, ls=":")
        ax.set_ylabel("mean pull")
        ax.set_title(rf"matched-reference injection: {strength:.0f}$\sigma_A$")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("signal-hypothesis mass [MeV]")
    axes[0].legend(ncol=4, fontsize=7.8, frameon=False, loc="best")
    fig.suptitle(
        "Phase-one 2016-full support scan (25 paired backgrounds)", fontsize=13
    )
    return save(fig, "2016_phase1_pull_means_by_support")


def phase1_score(
    summary: pd.DataFrame, decision: dict[str, object]
) -> list[dict[str, object]]:
    ordered = summary.sort_values("support_low_MeV")
    selected = str(decision["provisional_support"])
    minimum = float(decision["primary_minimum_worst_abs_mean_pull"])
    tie = float(decision["tie_margin"])
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    x = ordered["support_low_MeV"].to_numpy(float)
    ax.plot(
        x,
        ordered["worst_abs_mean_pull"],
        marker="o",
        lw=1.8,
        color="#D55E00",
        label="worst of 12 cells",
    )
    ax.plot(
        x,
        ordered["worst_abs_zero_signal_mean_pull"],
        marker="s",
        lw=1.5,
        color="#0072B2",
        label="worst background-only cell",
    )
    ax.axhspan(minimum, minimum + tie, color="#009E73", alpha=0.11)
    ax.axhline(0.75, color="#374151", ls="--", lw=1.0, label="0.75 practical threshold")
    ax.axhline(1.25, color="#6b7280", ls=":", lw=1.0, label="1.25 gross guard")
    ax.axvspan(33.75, 34.25, color="#9ca3af", alpha=0.18)
    ax.axvline(int(selected[:3]), color="#009E73", lw=1.35, label=f"provisional {int(selected[:3])} MeV")
    failed = ordered.loc[~ordered["technical_gate_pass"].astype(bool)]
    if len(failed):
        ax.scatter(
            failed["support_low_MeV"],
            failed["worst_abs_mean_pull"],
            marker="x",
            s=70,
            lw=1.6,
            color="#000000",
            label="technical gate failed",
            zorder=5,
        )
    ax.set_xticks(EDGES)
    ax.set_xlabel("lower GP-support edge [MeV]")
    ax.set_ylabel("maximum absolute mean pull")
    ax.set_title("Frozen phase-one minimax support ranking")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8.4)
    return save(fig, "2016_phase1_support_score")


def confirmation(
    cells: pd.DataFrame, decision: dict[str, object]
) -> list[dict[str, object]]:
    full = cells.loc[cells["cohort"] == "full_0_99"].copy()
    selected = str(decision["selected_support"])
    supports = [str(value) for value in decision["phase2_supports"]]
    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 9.2), sharex=True, constrained_layout=True
    )
    for ax, strength in zip(axes, (0.0, 2.0, 5.0)):
        for support in supports:
            group = full.loc[
                (full["support"] == support)
                & np.isclose(full["inj_nsigma"], strength)
            ].sort_values("mass_MeV")
            x = group["mass_MeV"].to_numpy(float)
            y = group["mean_pull"].to_numpy(float)
            yerr = np.vstack(
                [
                    y - group["mean_pull_ci90_low"].to_numpy(float),
                    group["mean_pull_ci90_high"].to_numpy(float) - y,
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[support],
                marker="o",
                ms=4.3,
                lw=1.7 if support == selected else 1.05,
                capsize=2.2,
                label=label(support, selected=selected),
            )
        ax.axhspan(-0.75, 0.75, color="#d1d5db", alpha=0.38, zorder=0)
        ax.axhline(0.0, color="black", lw=0.75)
        ax.axhline(1.25, color="#6b7280", lw=0.75, ls=":")
        ax.axhline(-1.25, color="#6b7280", lw=0.75, ls=":")
        ax.set_ylabel("mean pull")
        ax.set_title(rf"matched-reference injection: {strength:.0f}$\sigma_A$")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("signal-hypothesis mass [MeV]")
    axes[0].legend(ncol=len(supports), fontsize=8.2, frameon=False, loc="best")
    fig.suptitle(
        "Independent continuation and full-100 support confirmation", fontsize=13
    )
    return save(fig, "2016_confirmation_full100_pull_means")


def geometry_inventory() -> pd.DataFrame:
    mass = 0.039
    sigma = 0.00038 + 0.041 * mass - 0.27 * mass**2 + 3.49 * mass**3 - 11.11 * mass**4
    training_boundary = mass - 2.25 * sigma
    rows = []
    for edge in EDGES.astype(int):
        centers = np.arange(edge + 0.125, 210.0, 0.25)
        rows.append(
            {
                "support_low_MeV": edge,
                "support_high_MeV": 210,
                "freeze_eligible": edge <= 33,
                "geometry_control": edge == 34,
                "search_threshold_MeV": 39,
                "sigma_at_threshold_MeV": 1000.0 * sigma,
                "low_training_boundary_MeV": 1000.0 * training_boundary,
                "rebinned_low_side_centers": int(np.count_nonzero(centers < 1000.0 * training_boundary)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    phase1_decision = json.loads(
        (ANALYSIS / "phase1_selection_decision.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (ANALYSIS / "support_freeze_decision.json").read_text(encoding="utf-8")
    )
    if phase1_decision.get("status") != "provisional_edge_selected":
        raise RuntimeError("phase-one decision is not a provisional selection")
    if freeze.get("status") != "support_edge_frozen":
        raise RuntimeError("support edge is not frozen")
    phase1_cells = pd.read_csv(ANALYSIS / "phase1_cell_summary.csv")
    phase1_summary = pd.read_csv(ANALYSIS / "phase1_support_summary.csv")
    confirmation_cells = pd.read_csv(ANALYSIS / "confirmation_cell_summary.csv")
    products: list[dict[str, object]] = []
    products.extend(phase1_pulls(phase1_cells))
    products.extend(phase1_score(phase1_summary, phase1_decision))
    products.extend(confirmation(confirmation_cells, freeze))
    inventory = geometry_inventory()
    inventory_path = ANALYSIS / "support_geometry_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    products.append(
        {"path": str(inventory_path.relative_to(HERE)), "sha256": sha256(inventory_path)}
    )
    manifest = {
        "status": "pass",
        "selected_support": freeze["selected_support"],
        "selected_data_range_2016": freeze["data_range_2016"],
        "phase1_decision_sha256": sha256(ANALYSIS / "phase1_selection_decision.json"),
        "support_freeze_decision_sha256": sha256(ANALYSIS / "support_freeze_decision.json"),
        "products": products,
        "claim_boundary": (
            "Conditional source-recovery diagnostics; not direct coverage, "
            "expected sensitivity, exclusion, or observed-data evidence."
        ),
    }
    output = FIGURES / "support_figure_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
