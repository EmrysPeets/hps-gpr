#!/usr/bin/env python3
"""Validate and compare the two completed v4p8p3 closure collections.

This is a reporting layer only.  It never reruns an extraction, changes a
selection, or promotes the conditional source-conditioned ensembles to a
coverage or background-truth claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import residual_models as models


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived" / "residual_closure"
FIGURES = HERE / "figures"
MODELS = ("knot_spline", "regional_blend")
SCENARIOS = (
    "2021_1pct",
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
MASSES = (65.0, 90.0, 120.0, 180.0, 210.0)
STRENGTHS = (0.0, 1.0, 3.0, 5.0)
EXPECTED_ROWS_PER_MODEL = 2000
EXPECTED_CELLS_PER_MODEL = 100

MODEL_LABELS = {
    "knot_spline": "fixed two-knot log spline",
    "regional_blend": "three-region log blend",
}
SCENARIO_LABELS = {
    "2021_1pct": "native 1%",
    "2021_1pct_x10": "1% x 10",
    "2021_1pct_x100": "1% x 100",
    "2021_10pct": "native 10%",
    "2021_10pct_x10": "10% x 10",
}
COLORS = {
    "2021_1pct": "#3569a8",
    "2021_1pct_x10": "#5c9bd5",
    "2021_1pct_x100": "#2f855a",
    "2021_10pct": "#c55a11",
    "2021_10pct_x10": "#8c3d70",
}
MARKERS = {
    "2021_1pct": "o",
    "2021_1pct_x10": "s",
    "2021_1pct_x100": "^",
    "2021_10pct": "D",
    "2021_10pct_x10": "v",
}
OFFSETS = dict(zip(SCENARIOS, np.linspace(-2.0, 2.0, len(SCENARIOS))))
CLAIM_BOUNDARY = (
    "Requested conditional source-conditioned stress only; neither model is a "
    "qualified generator or signal-rigid, and these finite ensembles are not "
    "coverage, expected bands, limits, exclusions, or observed-data bias."
)


class ComparisonError(RuntimeError):
    """Raised when the completed closure products violate their contract."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def truthy(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ComparisonError(f"non-canonical boolean value {value!r}")
    return value == "True"


def validate_model(model: str) -> dict[str, Any]:
    directory = DERIVED / model
    collection_path = directory / "collection_summary.json"
    collection = read_json(collection_path)
    required = {
        "accepted_extraction_rows.csv",
        "closure_summary.csv",
        "exclusion_ledger.csv",
        "optimizer_attempt_ledger.csv",
        "raw_primary_extraction_rows.csv",
        "task_product_audit.csv",
        "zero_signal_bias_tests.csv",
    }
    if set(collection.get("derived_sha256", {})) != required:
        raise ComparisonError(f"{model}: unexpected collection file inventory")
    for name, expected_hash in collection["derived_sha256"].items():
        path = directory / name
        if models.sha256_file(path) != expected_hash:
            raise ComparisonError(f"{model}: hash mismatch for {name}")
    expected_collection = {
        "status": "pass",
        "model": model,
        "raw_rows": EXPECTED_ROWS_PER_MODEL,
        "accepted_rows": EXPECTED_ROWS_PER_MODEL,
        "excluded_rows": 0,
        "summary_cells": EXPECTED_CELLS_PER_MODEL,
        "minimum_accepted_per_cell": 20,
        "all_cells_sample_size_eligible": True,
        "selected_extraction_upper_factor": 25.0,
        "production_card_upper_factor": 15.0,
    }
    for key, expected in expected_collection.items():
        if collection.get(key) != expected:
            raise ComparisonError(
                f"{model}: collection {key}={collection.get(key)!r}, expected {expected!r}"
            )
    if collection.get("scientific_diagnostics", {}).get(
        "maximum_abs_pull_identity_residual"
    ) != 0.0:
        raise ComparisonError(f"{model}: pull identity is not exact")

    summary = read_csv(directory / "closure_summary.csv")
    accepted = read_csv(directory / "accepted_extraction_rows.csv")
    raw = read_csv(directory / "raw_primary_extraction_rows.csv")
    zero = read_csv(directory / "zero_signal_bias_tests.csv")
    exclusions = read_csv(directory / "exclusion_ledger.csv")
    if len(summary) != EXPECTED_CELLS_PER_MODEL:
        raise ComparisonError(f"{model}: expected 100 summary cells")
    if len(accepted) != EXPECTED_ROWS_PER_MODEL or len(raw) != EXPECTED_ROWS_PER_MODEL:
        raise ComparisonError(f"{model}: accepted/raw cardinality mismatch")
    if exclusions:
        raise ComparisonError(f"{model}: exclusion ledger is not empty")
    if len(zero) != len(SCENARIOS) * len(MASSES):
        raise ComparisonError(f"{model}: expected 25 zero-signal tests")

    cell_keys = {
        (r["scenario"], as_float(r, "mass_MeV"), as_float(r, "inj_nsigma"))
        for r in summary
    }
    expected_cell_keys = {
        (scenario, mass, strength)
        for scenario in SCENARIOS
        for mass in MASSES
        for strength in STRENGTHS
    }
    if cell_keys != expected_cell_keys:
        raise ComparisonError(f"{model}: incomplete or duplicate cell lattice")
    accepted_keys = {
        (
            r["scenario"],
            as_int(r, "background_toy_index"),
            as_float(r, "mass_MeV"),
            as_float(r, "inj_nsigma"),
        )
        for r in accepted
    }
    expected_accepted_keys = {
        (scenario, toy, mass, strength)
        for scenario in SCENARIOS
        for toy in range(20)
        for mass in MASSES
        for strength in STRENGTHS
    }
    if accepted_keys != expected_accepted_keys:
        raise ComparisonError(f"{model}: incomplete or duplicate accepted lattice")
    if any(not truthy(r["accepted"]) for r in accepted):
        raise ComparisonError(f"{model}: accepted ledger contains rejected rows")
    if max(abs(as_float(r, "pull_identity_residual")) for r in accepted) != 0.0:
        raise ComparisonError(f"{model}: nonzero pull identity residual")
    if any(as_int(r, "accepted_n") != 20 for r in summary):
        raise ComparisonError(f"{model}: non-20 accepted cell")
    return {
        "collection": collection,
        "collection_path": collection_path,
        "summary": summary,
        "accepted": accepted,
        "zero": zero,
    }


def zero_interval_excludes(row: dict[str, str]) -> bool:
    return not (
        as_float(row, "accepted_pull_mean_ci90_low")
        <= 0.0
        <= as_float(row, "accepted_pull_mean_ci90_high")
    )


def boundary_contact(row: dict[str, str]) -> bool:
    return any(
        truthy(row[key])
        for key in (
            "refit_upper_boundary",
            "refit_lower_boundary",
            "refit_constant_lower_boundary",
            "refit_constant_upper_boundary",
        )
    )


def compile_paired_response(
    products: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Form a descriptive same-background response after removing z=0 Ahat.

    This quantity is intentionally downstream of generator freezing.  It is
    useful for separating an extraction baseline offset from signal response,
    but it cannot repair or replace the source-fit influence audit.
    """

    paired_rows: list[dict[str, Any]] = []
    paired_cells: list[dict[str, Any]] = []
    for model in MODELS:
        accepted = products[model]["accepted"]
        indexed = {
            (
                r["scenario"],
                as_int(r, "background_toy_index"),
                as_float(r, "mass_MeV"),
                as_float(r, "inj_nsigma"),
            ): r
            for r in accepted
        }
        for scenario in SCENARIOS:
            for toy in range(20):
                for mass in MASSES:
                    baseline = indexed[(scenario, toy, mass, 0.0)]
                    for strength in (1.0, 3.0, 5.0):
                        injected = indexed[(scenario, toy, mass, strength)]
                        amplitude = as_float(injected, "strength")
                        if not amplitude > 0.0:
                            raise ComparisonError("nonpositive injected amplitude")
                        if abs(
                            as_float(injected, "sigmaA_ref")
                            - as_float(baseline, "sigmaA_ref")
                        ) > 1e-9 * max(1.0, as_float(baseline, "sigmaA_ref")):
                            raise ComparisonError("paired rows do not share sigmaA_ref")
                        delta = as_float(injected, "A_hat") - as_float(
                            baseline, "A_hat"
                        )
                        paired_rows.append(
                            {
                                "model": model,
                                "scenario": scenario,
                                "background_toy_index": toy,
                                "mass_MeV": mass,
                                "inj_nsigma": strength,
                                "injected_amplitude": amplitude,
                                "baseline_A_hat": as_float(baseline, "A_hat"),
                                "injected_A_hat": as_float(injected, "A_hat"),
                                "delta_A_hat": delta,
                                "paired_response": delta / amplitude,
                                "baseline_or_injected_boundary_contact": bool(
                                    boundary_contact(baseline)
                                    or boundary_contact(injected)
                                ),
                                "interpretation": (
                                    "post-hoc descriptive baseline-subtracted response; "
                                    "not a generator signal-rigidity test"
                                ),
                            }
                        )
        model_rows = [r for r in paired_rows if r["model"] == model]
        for scenario in SCENARIOS:
            for mass in MASSES:
                for strength in (1.0, 3.0, 5.0):
                    values = np.asarray(
                        [
                            float(r["paired_response"])
                            for r in model_rows
                            if r["scenario"] == scenario
                            and float(r["mass_MeV"]) == mass
                            and float(r["inj_nsigma"]) == strength
                        ]
                    )
                    if values.size != 20 or not np.all(np.isfinite(values)):
                        raise ComparisonError("paired-response cell is not finite n=20")
                    paired_cells.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "mass_MeV": mass,
                            "inj_nsigma": strength,
                            "n": int(values.size),
                            "mean_paired_response": float(np.mean(values)),
                            "sample_width_paired_response": float(
                                np.std(values, ddof=1)
                            ),
                            "median_paired_response": float(np.median(values)),
                            "paired_response_q16": float(np.quantile(values, 0.16)),
                            "paired_response_q84": float(np.quantile(values, 0.84)),
                            "interpretation": (
                                "post-hoc descriptive baseline-subtracted response; "
                                "not a generator signal-rigidity test"
                            ),
                        }
                    )
    if len(paired_rows) != 3000 or len(paired_cells) != 150:
        raise ComparisonError("unexpected paired-response cardinality")
    return paired_rows, paired_cells


def paired_cell(
    rows: Iterable[dict[str, Any]],
    model: str,
    scenario: str,
    mass: float,
    strength: float,
) -> dict[str, Any]:
    matches = [
        r
        for r in rows
        if r["model"] == model
        and r["scenario"] == scenario
        and float(r["mass_MeV"]) == mass
        and float(r["inj_nsigma"]) == strength
    ]
    if len(matches) != 1:
        raise ComparisonError("expected exactly one paired-response summary cell")
    return matches[0]


def compile_lane_rows(
    products: dict[str, dict[str, Any]], paired_cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        summaries = products[model]["summary"]
        accepted = products[model]["accepted"]
        for scenario in SCENARIOS:
            lane = [r for r in summaries if r["scenario"] == scenario]
            lane_accepted = [r for r in accepted if r["scenario"] == scenario]
            zero = [r for r in lane if as_float(r, "inj_nsigma") == 0.0]
            z5 = [r for r in lane if as_float(r, "inj_nsigma") == 5.0]
            injected = [r for r in lane if as_float(r, "inj_nsigma") > 0.0]
            worst = max(zero, key=lambda r: abs(as_float(r, "accepted_pull_mean")))
            widths = [as_float(r, "accepted_pull_width") for r in lane]
            recoveries = [as_float(r, "accepted_median_recovery") for r in z5]
            paired_z5 = [
                paired_cell(paired_cells, model, scenario, mass, 5.0)
                for mass in MASSES
            ]
            paired_z5_medians = [
                float(r["median_paired_response"]) for r in paired_z5
            ]
            injected_means = [as_float(r, "accepted_pull_mean") for r in injected]
            contacts = sum(boundary_contact(r) for r in lane_accepted)
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "raw_rows": sum(as_int(r, "raw_n") for r in lane),
                    "accepted_rows": sum(as_int(r, "accepted_n") for r in lane),
                    "excluded_rows": sum(as_int(r, "n_excluded") for r in lane),
                    "zero_max_abs_mean_pull": abs(
                        as_float(worst, "accepted_pull_mean")
                    ),
                    "zero_worst_signed_mean_pull": as_float(
                        worst, "accepted_pull_mean"
                    ),
                    "zero_worst_mass_MeV": as_float(worst, "mass_MeV"),
                    "zero_ci90_excluding_zero_cells": sum(
                        zero_interval_excludes(r) for r in zero
                    ),
                    "zero_cells": len(zero),
                    "pull_width_min_all_strengths": min(widths),
                    "pull_width_max_all_strengths": max(widths),
                    "injected_mean_pull_min": min(injected_means),
                    "injected_mean_pull_max": max(injected_means),
                    "z5_median_recovery_min": min(recoveries),
                    "z5_median_recovery_max": max(recoveries),
                    "z5_median_paired_response_min": min(paired_z5_medians),
                    "z5_median_paired_response_max": max(paired_z5_medians),
                    "boundary_contact_rows": contacts,
                    "boundary_contact_fraction": contacts / len(lane_accepted),
                    "selected_extraction_upper_factor": 25.0,
                    "production_card_upper_factor": 15.0,
                    "strict_generator_qualified": False,
                    "signal_rigidity_passed": False,
                    "allowed_interpretation": "requested conditional stress only",
                }
            )
    if len(rows) != 10:
        raise ComparisonError("expected ten model/lane summary rows")
    return rows


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def select_cell(
    rows: Iterable[dict[str, str]], scenario: str, mass: float, strength: float
) -> dict[str, str]:
    matches = [
        r
        for r in rows
        if r["scenario"] == scenario
        and abs(as_float(r, "mass_MeV") - mass) < 1e-9
        and abs(as_float(r, "inj_nsigma") - strength) < 1e-9
    ]
    if len(matches) != 1:
        raise ComparisonError(
            f"expected one cell for {scenario}, {mass} MeV, z={strength}"
        )
    return matches[0]


def plot_zero_signal(products: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2), sharex="col")
    for column, model in enumerate(MODELS):
        summary = products[model]["summary"]
        for scenario in SCENARIOS:
            cells = [select_cell(summary, scenario, mass, 0.0) for mass in MASSES]
            x = np.asarray(MASSES) + OFFSETS[scenario]
            means = np.asarray([as_float(r, "accepted_pull_mean") for r in cells])
            mean_low = np.asarray(
                [as_float(r, "accepted_pull_mean_ci90_low") for r in cells]
            )
            mean_high = np.asarray(
                [as_float(r, "accepted_pull_mean_ci90_high") for r in cells]
            )
            widths = np.asarray([as_float(r, "accepted_pull_width") for r in cells])
            width_low = np.asarray(
                [as_float(r, "accepted_pull_width_ci90_low") for r in cells]
            )
            width_high = np.asarray(
                [as_float(r, "accepted_pull_width_ci90_high") for r in cells]
            )
            style = {
                "color": COLORS[scenario],
                "marker": MARKERS[scenario],
                "ms": 4.0,
                "lw": 1.0,
                "capsize": 1.8,
                "label": SCENARIO_LABELS[scenario],
            }
            axes[0, column].errorbar(
                x,
                means,
                yerr=np.vstack((means - mean_low, mean_high - means)),
                **style,
            )
            axes[1, column].errorbar(
                x,
                widths,
                yerr=np.vstack((widths - width_low, width_high - widths)),
                **style,
            )
            contacted = np.asarray(
                [as_float(r, "accepted_upper_boundary_fraction") > 0.0 for r in cells]
            )
            if np.any(contacted):
                axes[0, column].scatter(
                    x[contacted],
                    means[contacted],
                    marker="x",
                    s=44,
                    linewidths=1.3,
                    color="#111111",
                    zorder=8,
                )
                axes[1, column].scatter(
                    x[contacted],
                    widths[contacted],
                    marker="x",
                    s=44,
                    linewidths=1.3,
                    color="#111111",
                    zorder=8,
                )
        axes[0, column].axhline(0.0, color="0.25", lw=0.85)
        axes[1, column].axhline(1.0, color="0.25", lw=0.85)
        axes[0, column].set_title(MODEL_LABELS[model])
        axes[1, column].set_xlabel("mass [MeV]")
        for row in range(2):
            axes[row, column].set_xticks(MASSES)
            axes[row, column].grid(alpha=0.17, lw=0.55)
    axes[0, 0].set_ylabel(r"mean signed pull $\bar p$")
    axes[1, 0].set_ylabel("sample pull width")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color="#111111",
            marker="x",
            linestyle="None",
            markersize=7,
            label="one or more factor-25 contacts",
        )
    )
    labels.append("one or more factor-25 contacts")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    fig.suptitle(
        "Zero-injection conditional diagnostics (20 backgrounds per cell)",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Bars are cellwise 90% Student-t (mean) and chi-square (width) intervals; model streams are unpaired.",
        ha="center",
        fontsize=8.4,
    )
    fig.tight_layout(rect=(0.03, 0.035, 1, 0.88))
    save_figure(fig, "v4p8p3_zero_signal_conditional_closure_20toy")


def plot_injected(
    products: dict[str, dict[str, Any]], paired_cells: list[dict[str, Any]]
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(15.2, 9.6), sharex="col")
    strengths = (1.0, 3.0, 5.0)
    for model_index, model in enumerate(MODELS):
        summary = products[model]["summary"]
        pull_column = 2 * model_index
        recovery_column = pull_column + 1
        for row_index, strength in enumerate(strengths):
            pull_axis = axes[row_index, pull_column]
            recovery_axis = axes[row_index, recovery_column]
            for scenario in SCENARIOS:
                cells = [
                    select_cell(summary, scenario, mass, strength) for mass in MASSES
                ]
                x = np.asarray(MASSES) + OFFSETS[scenario]
                means = np.asarray(
                    [as_float(r, "accepted_pull_mean") for r in cells]
                )
                mean_low = np.asarray(
                    [as_float(r, "accepted_pull_mean_ci90_low") for r in cells]
                )
                mean_high = np.asarray(
                    [as_float(r, "accepted_pull_mean_ci90_high") for r in cells]
                )
                response_cells = [
                    paired_cell(
                        paired_cells, model, scenario, mass, strength
                    )
                    for mass in MASSES
                ]
                recovery = np.asarray(
                    [float(r["median_paired_response"]) for r in response_cells]
                )
                recovery_low = np.asarray(
                    [float(r["paired_response_q16"]) for r in response_cells]
                )
                recovery_high = np.asarray(
                    [float(r["paired_response_q84"]) for r in response_cells]
                )
                style = {
                    "color": COLORS[scenario],
                    "marker": MARKERS[scenario],
                    "ms": 3.3,
                    "lw": 0.9,
                    "capsize": 1.4,
                    "label": SCENARIO_LABELS[scenario],
                }
                pull_axis.errorbar(
                    x,
                    means,
                    yerr=np.vstack((means - mean_low, mean_high - means)),
                    **style,
                )
                recovery_axis.plot(x, recovery, **{k: v for k, v in style.items() if k != "capsize"})
                recovery_axis.fill_between(
                    x,
                    recovery_low,
                    recovery_high,
                    color=COLORS[scenario],
                    alpha=0.075,
                    linewidth=0,
                )
                contacted = np.asarray(
                    [as_float(r, "accepted_upper_boundary_fraction") > 0.0 for r in cells]
                )
                if np.any(contacted):
                    pull_axis.scatter(
                        x[contacted],
                        means[contacted],
                        marker="x",
                        s=34,
                        linewidths=1.1,
                        color="#111111",
                        zorder=8,
                    )
                    recovery_axis.scatter(
                        x[contacted],
                        recovery[contacted],
                        marker="x",
                        s=34,
                        linewidths=1.1,
                        color="#111111",
                        zorder=8,
                    )
            pull_axis.axhline(0.0, color="0.25", lw=0.75)
            recovery_axis.axhline(1.0, color="0.25", lw=0.75)
            pull_axis.set_ylabel(rf"$z={int(strength)}$: mean pull")
            for axis in (pull_axis, recovery_axis):
                axis.set_xticks(MASSES)
                axis.grid(alpha=0.16, lw=0.5)
        axes[0, pull_column].set_title(f"{MODEL_LABELS[model]}: signed pull")
        axes[0, recovery_column].set_title(
            f"{MODEL_LABELS[model]}: paired response"
        )
        axes[1, recovery_column].set_ylabel(
            r"median $(\hat A_z-\hat A_0)/A_{\rm inj}$"
        )
        axes[2, pull_column].set_xlabel("mass [MeV]")
        axes[2, recovery_column].set_xlabel("mass [MeV]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color="#111111",
            marker="x",
            linestyle="None",
            markersize=6,
            label="one or more factor-25 contacts",
        )
    )
    labels.append("one or more factor-25 contacts")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=8.0,
    )
    fig.suptitle(
        "Injected conditional recovery (20 backgrounds per cell)",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Pull bars are 90% Student-t intervals; paired-response ribbons span the 16th--84th percentiles. The response is post-hoc descriptive and does not replace the source influence audit.",
        ha="center",
        fontsize=8.2,
    )
    fig.tight_layout(rect=(0.025, 0.035, 1, 0.88))
    save_figure(fig, "v4p8p3_injected_recovery_20toy")


def main() -> int:
    products = {model: validate_model(model) for model in MODELS}

    fit_result = models.load_fit_result(require_influence=True)
    for model in MODELS:
        if fit_result["models"][model]["strict_generator_qualification_passed"]:
            raise ComparisonError(f"{model}: unexpectedly qualified source model")
        if fit_result["signal_influence_audit"]["summaries"][model][
            "signal_influence_gate_passed"
        ]:
            raise ComparisonError(f"{model}: unexpectedly signal-rigid")

    pilot_path = HERE / "derived/residual_length_pilot/common_ceiling_disposition.json"
    pilot = read_json(pilot_path)
    if pilot.get("selected_common_upper_factor") != 25.0:
        raise ComparisonError("comparison requires the completed common factor-25 pilot")

    combined_summary = [
        row for model in MODELS for row in products[model]["summary"]
    ]
    combined_zero = [row for model in MODELS for row in products[model]["zero"]]
    paired_rows, paired_cells = compile_paired_response(products)
    lane_rows = compile_lane_rows(products, paired_cells)
    models.atomic_csv(
        DERIVED / "combined_closure_summary.csv",
        combined_summary,
        list(combined_summary[0]),
    )
    models.atomic_csv(
        DERIVED / "combined_zero_signal_bias_tests.csv",
        combined_zero,
        list(combined_zero[0]),
    )
    models.atomic_csv(
        DERIVED / "model_lane_closure_summary.csv",
        lane_rows,
        list(lane_rows[0]),
    )
    models.atomic_csv(
        DERIVED / "paired_baseline_subtracted_response_rows.csv",
        paired_rows,
        list(paired_rows[0]),
    )
    models.atomic_csv(
        DERIVED / "paired_baseline_subtracted_response_summary.csv",
        paired_cells,
        list(paired_cells[0]),
    )

    plot_zero_signal(products)
    plot_injected(products, paired_cells)

    aggregate: dict[str, Any] = {}
    for model in MODELS:
        rows = [r for r in lane_rows if r["model"] == model]
        model_paired_rows = [r for r in paired_rows if r["model"] == model]
        paired_values = np.asarray(
            [float(r["paired_response"]) for r in model_paired_rows]
        )
        model_paired_cells = [r for r in paired_cells if r["model"] == model]
        aggregate[model] = {
            "raw_rows": sum(int(r["raw_rows"]) for r in rows),
            "accepted_rows": sum(int(r["accepted_rows"]) for r in rows),
            "excluded_rows": sum(int(r["excluded_rows"]) for r in rows),
            "maximum_zero_abs_mean_pull": max(
                float(r["zero_max_abs_mean_pull"]) for r in rows
            ),
            "zero_cells_ci90_excluding_zero": sum(
                int(r["zero_ci90_excluding_zero_cells"]) for r in rows
            ),
            "zero_cells": sum(int(r["zero_cells"]) for r in rows),
            "pull_width_range_all_strengths": [
                min(float(r["pull_width_min_all_strengths"]) for r in rows),
                max(float(r["pull_width_max_all_strengths"]) for r in rows),
            ],
            "z5_median_recovery_range": [
                min(float(r["z5_median_recovery_min"]) for r in rows),
                max(float(r["z5_median_recovery_max"]) for r in rows),
            ],
            "z5_median_paired_response_range": [
                min(float(r["z5_median_paired_response_min"]) for r in rows),
                max(float(r["z5_median_paired_response_max"]) for r in rows),
            ],
            "paired_response_all_injected_rows": {
                "n": int(paired_values.size),
                "median": float(np.median(paired_values)),
                "q16": float(np.quantile(paired_values, 0.16)),
                "q84": float(np.quantile(paired_values, 0.84)),
                "minimum": float(np.min(paired_values)),
                "maximum": float(np.max(paired_values)),
                "count_below_0p9": int(np.count_nonzero(paired_values < 0.9)),
                "count_below_0p8": int(np.count_nonzero(paired_values < 0.8)),
                "count_below_0p5": int(np.count_nonzero(paired_values < 0.5)),
                "count_above_1p1": int(np.count_nonzero(paired_values > 1.1)),
                "post_hoc_descriptive": True,
            },
            "paired_response_cell_median_by_injection": {
                str(int(strength)): {
                    "median_across_cells": float(
                        np.median(
                            [
                                float(r["median_paired_response"])
                                for r in model_paired_cells
                                if float(r["inj_nsigma"]) == strength
                            ]
                        )
                    ),
                    "minimum_cell_median": min(
                        float(r["median_paired_response"])
                        for r in model_paired_cells
                        if float(r["inj_nsigma"]) == strength
                    ),
                    "maximum_cell_median": max(
                        float(r["median_paired_response"])
                        for r in model_paired_cells
                        if float(r["inj_nsigma"]) == strength
                    ),
                }
                for strength in (1.0, 3.0, 5.0)
            },
            "boundary_contact_rows": sum(
                int(r["boundary_contact_rows"]) for r in rows
            ),
            "boundary_contact_fraction": sum(
                int(r["boundary_contact_rows"]) for r in rows
            )
            / EXPECTED_ROWS_PER_MODEL,
            "source_qualification_passed": False,
            "signal_rigidity_passed": False,
            "allowed_interpretation": "requested conditional stress only",
        }

    output_names = (
        "combined_closure_summary.csv",
        "combined_zero_signal_bias_tests.csv",
        "model_lane_closure_summary.csv",
        "paired_baseline_subtracted_response_rows.csv",
        "paired_baseline_subtracted_response_summary.csv",
    )
    figure_names = (
        "v4p8p3_zero_signal_conditional_closure_20toy.pdf",
        "v4p8p3_zero_signal_conditional_closure_20toy.png",
        "v4p8p3_injected_recovery_20toy.pdf",
        "v4p8p3_injected_recovery_20toy.png",
    )
    manifest = {
        "schema_version": 1,
        "study_id": "v4p8p3_2021_residual_truths_20260814",
        "status": "pass_reporting_integrity_only",
        "scientific_disposition": "fail_requested_conditional_stress_only",
        "claim_boundary": CLAIM_BOUNDARY,
        "model_stream_comparison": "descriptive_unpaired",
        "counts": {
            "models": 2,
            "lanes_per_model": 5,
            "backgrounds_per_lane_model": 20,
            "masses": 5,
            "strengths": 4,
            "raw_rows": 4000,
            "accepted_rows": 4000,
            "excluded_rows": 0,
            "summary_cells": 200,
        },
        "models": aggregate,
        "input_sha256": {
            "source_fit_and_influence.json": models.sha256_file(
                models.FIT_RESULT_PATH
            ),
            "common_ceiling_disposition.json": models.sha256_file(pilot_path),
            **{
                f"{model}/collection_summary.json": models.sha256_file(
                    products[model]["collection_path"]
                )
                for model in MODELS
            },
        },
        "output_sha256": {
            **{
                name: models.sha256_file(DERIVED / name) for name in output_names
            },
            **{name: models.sha256_file(FIGURES / name) for name in figure_names},
        },
        "script_sha256": models.sha256_file(Path(__file__)),
    }
    models.atomic_json(FIGURES / "closure_comparison_figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
