#!/usr/bin/env python3
"""Package the frozen v4.9.7 phase-one no-selection outcome.

This is a post-decision reporting utility. It does not select an edge, open
observed support-specific results, or authorize downstream inference.
"""

from __future__ import annotations

import csv
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
EXPECTED_PRODUCTS = {
    "phase1_accepted_rows.csv",
    "phase1_adjacent_paired_differences.csv",
    "phase1_cell_summary.csv",
    "phase1_support_summary.csv",
}
COLORS = {
    support: color
    for support, color in zip(
        SUPPORTS,
        ("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499"),
    )
}


class ReportError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_figure(fig: plt.Figure, stem: str) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append({"path": str(path.relative_to(HERE)), "sha256": sha256(path)})
    plt.close(fig)
    return outputs


def validate_decision() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    decision_path = ANALYSIS / "phase1_selection_decision.json"
    decision = load_json(decision_path)
    if (
        decision.get("status") != "no_provisional_edge"
        or decision.get("phase2_supports") != []
        or decision.get("observed_scan_authorized") is not False
        or decision.get("holdout_evaluated") is not False
    ):
        raise ReportError("phase-one decision is not the frozen no-selection state")
    products = decision.get("products", {})
    if set(products) != EXPECTED_PRODUCTS:
        raise ReportError("phase-one decision product inventory drift")
    for name, record in products.items():
        path = ANALYSIS / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ReportError(f"phase-one product hash mismatch: {name}")
    support = pd.read_csv(ANALYSIS / "phase1_support_summary.csv")
    cells = pd.read_csv(ANALYSIS / "phase1_cell_summary.csv")
    if tuple(support["support"].astype(str)) != SUPPORTS:
        raise ReportError("support inventory drift")
    if len(cells) != 84 or set(cells["support"].astype(str)) != set(SUPPORTS):
        raise ReportError("phase-one cell inventory drift")
    if support["practical_acceptability_pass"].astype(bool).any():
        raise ReportError("a support unexpectedly passes the practical rule")
    if support["gross_bias_guard_pass"].astype(bool).any():
        raise ReportError("a support unexpectedly passes the gross-bias guard")
    return decision, support, cells


def collect_analytic_means() -> pd.DataFrame:
    frames = []
    for support in SUPPORTS:
        directory = HERE / "derived" / f"2016_threshold_qualified_{support}"
        summary_path = directory / "analytic_mean_closure_summary.json"
        summary = load_json(summary_path)
        selected_path = directory / "analytic_mean_zero_signal_closure.csv"
        attempts_path = directory / "analytic_mean_optimizer_attempts.csv"
        if (
            summary.get("status") != "pass"
            or int(summary.get("rows", -1)) != 4
            or int(summary.get("attempt_rows", -1)) < 12
            or sha256(selected_path) != summary.get("selected_sha256")
            or sha256(attempts_path) != summary.get("attempts_sha256")
        ):
            raise ReportError(f"analytic-mean provenance failed for {support}")
        frame = pd.read_csv(selected_path)
        masses = np.rint(1000.0 * frame["mass_GeV"].to_numpy(float)).astype(int)
        if len(frame) != 4 or not np.array_equal(masses, np.array([44, 49, 54, 59])):
            raise ReportError(f"analytic-mean mass grid drift for {support}")
        if not np.isfinite(frame[["A_hat", "sigma_A", "pull"]].to_numpy(float)).all():
            raise ReportError(f"nonfinite analytic-mean result for {support}")
        frame = frame.copy()
        frame["support"] = support
        frame["support_low_MeV"] = int(support[:3])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["support_low_MeV", "mass_GeV"]
    )


def collect_exclusions() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for support in SUPPORTS:
        run_root = HERE / "runs" / f"2016_threshold_qualified_{support}" / "2016_full"
        canonical = sorted(run_root.glob("toy_[0-9][0-9][0-9][0-9]"))
        if len(canonical) != 25:
            raise ReportError(f"canonical phase-one toy inventory drift for {support}")
        for task in canonical:
            path = task / "exclusions.csv"
            if not path.is_file():
                raise ReportError(f"missing exclusion ledger: {path}")
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    records.append(
                        {
                            "support": support,
                            "toy_index": int(task.name[-4:]),
                            "mass_GeV": float(row["mass_GeV"]),
                            "mass_MeV": int(round(1000.0 * float(row["mass_GeV"]))),
                            "inj_nsigma": float(row["inj_nsigma"]),
                            "reason": str(row["reason"]),
                            "source_ledger": str(path.relative_to(HERE)),
                            "source_ledger_sha256": sha256(path),
                        }
                    )
    return pd.DataFrame(records)


def plot_phase1_cells(cells: pd.DataFrame) -> list[dict[str, object]]:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True, constrained_layout=True)
    for ax, strength in zip(axes, (0.0, 2.0, 5.0)):
        for support in SUPPORTS:
            group = cells.loc[
                (cells["support"] == support)
                & np.isclose(cells["inj_nsigma"], strength)
            ].sort_values("mass_MeV")
            y = group["mean_pull"].to_numpy(float)
            ax.errorbar(
                group["mass_MeV"],
                y,
                yerr=np.vstack(
                    [
                        y - group["mean_pull_ci90_low"].to_numpy(float),
                        group["mean_pull_ci90_high"].to_numpy(float) - y,
                    ]
                ),
                color=COLORS[support],
                marker="o",
                ms=3.7,
                lw=1.05,
                capsize=2.0,
                ls="--" if support == "034_210" else "-",
                label=f"{int(support[:3])} MeV" + (" control" if support == "034_210" else ""),
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
    fig.suptitle("Phase-one full-2016 support scan: no qualifying edge", fontsize=13)
    return save_figure(fig, "2016_phase1_pull_means_no_qualifying_edge")


def plot_failure_score(summary: pd.DataFrame) -> list[dict[str, object]]:
    ordered = summary.sort_values("support_low_MeV")
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    x = ordered["support_low_MeV"].to_numpy(float)
    if not np.allclose(
        ordered["worst_abs_mean_pull"],
        ordered["worst_abs_zero_signal_mean_pull"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ReportError("the recorded worst cell is not background-only at every edge")
    ax.plot(
        x,
        ordered["worst_abs_mean_pull"],
        marker="s",
        lw=1.8,
        color="#0072B2",
        label="worst cell (background-only at every edge)",
    )
    ax.axhline(0.75, color="#374151", ls="--", lw=1.0, label="0.75 practical threshold")
    ax.axhline(1.25, color="#6b7280", ls=":", lw=1.0, label="1.25 gross guard")
    ax.axvspan(33.75, 34.25, color="#9ca3af", alpha=0.18, label="ineligible control")
    failed = ordered.loc[~ordered["technical_gate_pass"].astype(bool)]
    ax.scatter(failed["support_low_MeV"], failed["worst_abs_mean_pull"], marker="x", s=80, lw=1.8, color="black", label="technical gate failed", zorder=5)
    ax.set_xticks(np.arange(28, 35))
    ax.set_xlabel("lower GP-support edge [MeV]")
    ax.set_ylabel("maximum absolute mean pull")
    ax.set_title("Frozen phase-one rule: every eligible edge fails")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8.2)
    return save_figure(fig, "2016_phase1_support_failure_score")


def plot_analytic_means(frame: pd.DataFrame) -> list[dict[str, object]]:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    for support in SUPPORTS:
        group = frame.loc[frame["support"] == support].sort_values("mass_MeV")
        ax.plot(
            group["mass_MeV"],
            group["pull"],
            marker="o",
            ms=4.0,
            lw=1.15,
            ls="--" if support == "034_210" else "-",
            color=COLORS[support],
            label=f"{int(support[:3])} MeV" + (" control" if support == "034_210" else ""),
        )
    ax.axhspan(-0.75, 0.75, color="#d1d5db", alpha=0.38, zorder=0)
    ax.axhline(0.0, color="black", lw=0.75)
    ax.axhline(1.25, color="#6b7280", lw=0.75, ls=":")
    ax.axhline(-1.25, color="#6b7280", lw=0.75, ls=":")
    ax.set_xlabel("signal-hypothesis mass [MeV]")
    ax.set_ylabel(r"analytic-mean fitted amplitude / $\sigma_A$")
    ax.set_title("Post-decision diagnostic: deterministic truth/GP mismatch")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=4, fontsize=8.0)
    return save_figure(fig, "2016_analytic_mean_support_failure_diagnostic")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    decision, support, cells = validate_decision()
    analytic = collect_analytic_means()
    exclusions = collect_exclusions()

    support_path = ANALYSIS / "phase1_support_failure_summary.csv"
    analytic_path = ANALYSIS / "analytic_mean_zero_signal_all_supports.csv"
    exclusion_path = ANALYSIS / "phase1_technical_exclusions.csv"
    support.to_csv(support_path, index=False)
    analytic.to_csv(analytic_path, index=False)
    exclusions.to_csv(exclusion_path, index=False)

    products: list[dict[str, object]] = [
        {"path": str(support_path.relative_to(HERE)), "sha256": sha256(support_path)},
        {"path": str(analytic_path.relative_to(HERE)), "sha256": sha256(analytic_path)},
        {"path": str(exclusion_path.relative_to(HERE)), "sha256": sha256(exclusion_path)},
    ]
    products.extend(plot_phase1_cells(cells))
    products.extend(plot_failure_score(support))
    products.extend(plot_analytic_means(analytic))

    best_numeric = support.loc[support["worst_abs_mean_pull"].idxmin()]
    by_mass = {
        str(int(mass)): {
            "min_pull": float(group["pull"].min()),
            "max_pull": float(group["pull"].max()),
        }
        for mass, group in analytic.groupby("mass_MeV", sort=True)
    }
    payload = {
        "status": "halted_no_provisional_edge",
        "study_id": decision["study_id"],
        "phase1_decision_sha256": sha256(ANALYSIS / "phase1_selection_decision.json"),
        "candidate_edges_MeV": list(range(28, 35)),
        "eligible_edges_MeV": list(range(28, 34)),
        "control_edge_MeV": 34,
        "technical_gate_failed_edges_MeV": support.loc[
            ~support["technical_gate_pass"].astype(bool), "support_low_MeV"
        ].astype(int).tolist(),
        "technical_exclusion_count": int(len(exclusions)),
        "practical_gate_passed_edges_MeV": [],
        "numerically_smallest_worst_pull_edge_MeV": int(best_numeric["support_low_MeV"]),
        "numerically_smallest_worst_abs_mean_pull": float(best_numeric["worst_abs_mean_pull"]),
        "numerically_smallest_edge_not_selected": True,
        "analytic_mean_pull_range_by_mass_MeV": by_mass,
        "phase2_authorized": False,
        "observed_scan_authorized": False,
        "combined_result_authorized": False,
        "products": products,
        "interpretation": (
            "Every eligible edge failed the frozen phase-one practical rule. "
            "The uniform post-decision analytic-mean diagnostic shows the same "
            "alternating mass-dependent mismatch without Poisson fluctuations. "
            "No lowest-score fallback is selected, and no phase-two, observed, "
            "or combined result is authorized."
        ),
        "claim_boundary": (
            "Conditional source-recovery failure diagnostic; not coverage, an "
            "observed-data result, exclusion, calibrated sensitivity, or global significance."
        ),
    }
    output = ANALYSIS / "failed_support_study_summary.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
