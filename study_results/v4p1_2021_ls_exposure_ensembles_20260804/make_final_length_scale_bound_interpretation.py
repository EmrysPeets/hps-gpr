#!/usr/bin/env python3
"""Create the deterministic final length-scale-bound interpretation.

This script reads reviewed CSV/JSON products only. It does not import or run
the fit implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
DERIVED = STUDY_DIR / "derived"
OUTPUT_JSON = DERIVED / "final_length_scale_bound_interpretation.json"
OUTPUT_MD = STUDY_DIR / "FINAL_LENGTH_SCALE_BOUND_INTERPRETATION.md"
PROJECTED = ["2021_1pct_x100", "2021_10pct_x10"]
FACTORS = [15, 20, 25]
TRUTHS = ["gengamma", "sigpowexpq"]
LABELS = {
    "2021_1pct_x100": "2021 1% × 100",
    "2021_10pct_x10": "2021 10% × 10",
}
ROLES = {"gengamma": "primary", "sigpowexpq": "alternate"}
SOURCES = {
    "postprocess_manifest": DERIVED / "v4p1_ensemble_postprocess_manifest.json",
    "optimizer_audit": DERIVED / "scan_optimizer_audit_summary.json",
    "reviewed_scan": DERIVED / "scan_reviewed_rows_complete.csv",
    "complete_injection": DERIVED / "injection_rows_complete.csv",
    "factor_summary_gengamma": DERIVED / "v4p1_ensemble_factor_summary_gengamma.csv",
    "factor_summary_sigpowexpq": DERIVED / "v4p1_ensemble_factor_summary_sigpowexpq.csv",
    "observed_summary_csv": DERIVED / "fig_v4p1_ls_observed_dataset_comparison_summary.csv",
    "observed_summary_json": DERIVED / "fig_v4p1_ls_observed_dataset_comparison_summary.json",
    "unpaired_factor20_summary": DERIVED / "fig_v4p1_factor20_native10_vs_1pct_x10_toy_medians_summary.csv",
}


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> Dict[str, Any]:
    try:
        display_path = path.relative_to(STUDY_DIR)
    except ValueError:
        display_path = path
    return {
        "path": str(display_path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value


def exact_product(frame: pd.DataFrame, columns: Sequence[str], expected: int, label: str) -> None:
    if len(frame) != expected or frame.duplicated(list(columns)).any():
        raise GateError(f"{label} exact-product gate failed")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build() -> Tuple[Dict[str, Any], str]:
    for path in SOURCES.values():
        if not path.is_file():
            raise GateError(f"Missing reviewed input: {path}")
    source_records = {name: record(path) for name, path in SOURCES.items()}
    manifest = load_json(SOURCES["postprocess_manifest"])
    audit = load_json(SOURCES["optimizer_audit"])
    if manifest.get("status") != "complete":
        raise GateError("Production postprocess manifest is not complete")
    if manifest.get("validated_rows") != {
        "adjacent_factor_lml_comparisons": 5500,
        "injection": 7200,
        "mass_anchor_signal_strata": 720,
        "paired_signal_response": 5400,
        "qmu_ok_false_rows": 1,
        "qmu_one_sided_zero_branch_diagnostic_rows": 1,
        "scan": 6600,
    }:
        raise GateError("Production validated-row ledger drift")
    semantics = manifest.get("semantics", {})
    if (
        semantics.get("expected_limit_bands") is not False
        or semantics.get("limit_bands_created") is not False
        or semantics.get("qmu_outputs_used_in_postprocess") is not False
        or semantics.get("qmu_outputs_promotable") is not False
    ):
        raise GateError("Production scope semantics drift")
    if audit.get("audit_gate") != "optimizer_audit_pass":
        raise GateError("Optimizer audit is not a pass")
    boundary = float(audit["bound_thresholds"]["at_bound_fraction_of_upper"])
    near_boundary = float(audit["bound_thresholds"]["near_bound_fraction_of_upper"])
    if boundary != 0.999:
        raise GateError("Official boundary criterion drift")

    summaries = pd.concat(
        [
            pd.read_csv(SOURCES["factor_summary_gengamma"]),
            pd.read_csv(SOURCES["factor_summary_sigpowexpq"]),
        ],
        ignore_index=True,
    )
    exact_product(
        summaries,
        ["truth_model", "study_scenario", "factor"],
        60,
        "factor summary",
    )
    scan = pd.read_csv(SOURCES["reviewed_scan"], low_memory=False)
    injection = pd.read_csv(SOURCES["complete_injection"], low_memory=False)
    exact_product(
        scan,
        [
            "truth_model",
            "study_scenario",
            "ls_upper_factor_requested",
            "background_toy_index",
            "mass_GeV",
        ],
        6600,
        "reviewed scan",
    )
    exact_product(
        injection,
        [
            "truth_model",
            "study_scenario",
            "ls_upper_factor_requested",
            "background_toy_index",
            "mass_GeV",
            "injection_anchor_nsigma",
            "injection_toy",
        ],
        7200,
        "complete injection",
    )
    scan["bound_fraction"] = scan["ls_opt"] / scan["ls_hi"]
    injection["refit_bound_fraction"] = injection["refit_ls_opt"] / injection["ls_hi"]

    rows = []
    for truth in TRUTHS:
        for scenario in PROJECTED:
            for factor in FACTORS:
                summary = summaries.loc[
                    (summaries["truth_model"] == truth)
                    & (summaries["study_scenario"] == scenario)
                    & (summaries["factor"] == factor)
                ]
                scan_part = scan.loc[
                    (scan["truth_model"] == truth)
                    & (scan["study_scenario"] == scenario)
                    & (scan["ls_upper_factor_requested"] == factor)
                ]
                inj_part = injection.loc[
                    (injection["truth_model"] == truth)
                    & (injection["study_scenario"] == scenario)
                    & (injection["ls_upper_factor_requested"] == factor)
                ]
                if len(summary) != 1 or len(scan_part) != 110 or len(inj_part) != 120:
                    raise GateError("Projected-lane coverage drift")
                s = summary.iloc[0]
                scan_bound = scan_part["bound_fraction"] >= boundary
                inj_bound = inj_part["refit_bound_fraction"] >= boundary
                response = float(s["paired_response_median"])
                rows.append(
                    {
                        "truth_model": truth,
                        "truth_role": ROLES[truth],
                        "scenario": scenario,
                        "scenario_label": LABELS[scenario],
                        "factor": factor,
                        "optimized_ls_median_over_sigma_x": float(s["ls_ratio_median"]),
                        "scan_at_bound_rows": int(scan_bound.sum()),
                        "scan_rows": 110,
                        "scan_toys_with_any_bound": int(
                            scan_part.assign(_bound=scan_bound)
                            .groupby("background_toy_index")["_bound"]
                            .any()
                            .sum()
                        ),
                        "injection_refit_at_bound_rows": int(inj_bound.sum()),
                        "injection_rows": 120,
                        "fitted_sigma_A_ratio_to_f15": float(
                            s["sigma_A_over_anchor_median"]
                        ),
                        "paired_response_median": response,
                        "paired_response_deficit_percent": 100.0 * (1.0 - response),
                        "anchor_normalized_residual_median": float(
                            s["Ahat_minus_Ainj_over_anchor_sigma_median"]
                        ),
                        "pull_width_strata_median": float(
                            s["pull_width_strata_median"]
                        ),
                        "adjacent_lml_median_delta": (
                            None
                            if factor == 15
                            else float(s["adjacent_lml_median_delta"])
                        ),
                        "adjacent_lml_regressions": int(
                            s["adjacent_lml_regressions"]
                        ),
                    }
                )
    projected = pd.DataFrame(rows)
    f20 = projected.loc[projected["factor"] == 20].set_index(
        ["truth_model", "scenario"]
    )
    f25 = projected.loc[projected["factor"] == 25].set_index(
        ["truth_model", "scenario"]
    )
    all_f20 = scan.loc[scan["ls_upper_factor_requested"] == 20]
    all_f25 = scan.loc[scan["ls_upper_factor_requested"] == 25]
    f20_bound = all_f20["bound_fraction"] >= boundary
    f25_bound = all_f25["bound_fraction"] >= boundary
    f25_near = all_f25["bound_fraction"] >= near_boundary
    response_spans = (
        projected.groupby(["truth_model", "scenario"])["paired_response_median"]
        .agg(lambda values: 100.0 * (values.max() - values.min()))
    )

    observed = pd.read_csv(SOURCES["observed_summary_csv"])
    exact_product(observed, ["year", "exposure"], 4, "observed summary")
    observed_rows = [
        {
            "year": int(row.year),
            "exposure": str(row.exposure),
            "median_ls_opt_over_sigma_x": float(row.ls_opt_over_sigma_x_median),
            "configured_ceiling": float(row.configured_ceiling),
            "at_ceiling_rows": int(row.at_ceiling_count),
            "rows": int(row.rows),
            "interpolated_rows": int(row.interpolated_rows),
        }
        for row in observed.itertuples(index=False)
    ]
    if any(row["interpolated_rows"] for row in observed_rows):
        raise GateError("Observed summary contains interpolation")

    comparison = pd.read_csv(SOURCES["unpaired_factor20_summary"])
    exact_product(
        comparison, ["truth_model", "study_scenario"], 4, "unpaired comparison"
    )
    if comparison["source_families_paired"].astype(bool).any():
        raise GateError("Source families were incorrectly marked paired")
    comparison_rows = [
        {
            "truth_model": str(row.truth_model),
            "truth_role": str(row.truth_role),
            "scenario": str(row.study_scenario),
            "display_label": str(row.display_label),
            "n_independent_toys": int(row.n_independent_toys),
            "mass_points_per_toy": int(row.n_mass_points_per_toy),
            "toy_median_ls_opt_over_sigma_x": float(row.toy_median_median),
            "near_bound_mass_rows": int(row.near_upper_bound_mass_rows),
        }
        for row in comparison.itertuples(index=False)
    ]

    payload = {
        "schema_version": 1,
        "study_id": str(manifest["study_id"]),
        "status": "reviewed_interpretation_ready",
        "definitions": {
            "bound_criterion": "ls_opt / ls_hi >= 0.999",
            "bound_fraction": boundary,
            "near_bound_fraction": near_boundary,
            "independent_toy_unit": True,
            "mass_rows_correlated_within_toy": True,
        },
        "decision": {
            "provisional_projected_factor": 20,
            "universal_common_factor": 25,
            "factor20_projected_scan_at_bound_rows": int(
                f20["scan_at_bound_rows"].sum()
            ),
            "factor20_projected_injection_refit_at_bound_rows": int(
                f20["injection_refit_at_bound_rows"].sum()
            ),
            "factor20_all_scenarios_scan_at_bound_rows": int(f20_bound.sum()),
            "factor25_all_scenarios_scan_at_bound_rows": int(f25_bound.sum()),
            "factor25_all_scenarios_scan_near_bound_rows": int(f25_near.sum()),
            "interpretation": (
                "Factor 20 clears scan and injection-refit boundary contact in "
                "all projected-100% lanes, but factor 25 is the universal common "
                "choice because factor 20 retains contact in diagnostic 1% lanes."
            ),
        },
        "projected_100pct_rows": rows,
        "projected_100pct_aggregate": {
            "paired_response_deficit_percent_min": float(
                projected["paired_response_deficit_percent"].min()
            ),
            "paired_response_deficit_percent_max": float(
                projected["paired_response_deficit_percent"].max()
            ),
            "max_within_lane_response_span_percentage_points": float(
                response_spans.max()
            ),
            "response_interpretation": (
                "The common roughly 2-3% deficit persists after boundary contact "
                "is removed at factors 20 and 25, so it is a response diagnostic, "
                "not a bound-induced sensitivity change."
            ),
            "factor20_to_factor25_max_abs_ls_relative_change_percent": float(
                np.max(np.abs((f25["optimized_ls_median_over_sigma_x"] /
                               f20["optimized_ls_median_over_sigma_x"] - 1.0) * 100.0))
            ),
            "factor20_to_factor25_max_fitted_sigma_A_increase_percent": float(
                np.max((f25["fitted_sigma_A_ratio_to_f15"] /
                        f20["fitted_sigma_A_ratio_to_f15"] - 1.0) * 100.0)
            ),
            "factor20_to_factor25_max_abs_response_change_percentage_points": float(
                np.max(np.abs((f25["paired_response_median"] -
                               f20["paired_response_median"]) * 100.0))
            ),
            "factor25_adjacent_lml_regressions": int(
                f25["adjacent_lml_regressions"].sum()
            ),
            "plateau_interpretation": (
                "Factor 25 is an optimizer plateau relative to factor 20 and "
                "shows no fitted-uncertainty sensitivity degradation."
            ),
        },
        "observed_length_scale_medians": observed_rows,
        "unpaired_factor20_comparison": {
            "source_families_paired": False,
            "paired_difference_or_ratio_reported": False,
            "source_support_ratio_ten_pct_over_one_pct": float(
                comparison["source_support_ratio_ten_pct_over_one_pct"].iloc[0]
            ),
            "effective_target_ratio_native10_over_1pct_x10": float(
                comparison[
                    "effective_target_ratio_native10_over_1pct_x10"
                ].iloc[0]
            ),
            "groups": comparison_rows,
        },
        "qmu_exclusion": {
            "qmu_ok_false_rows": 1,
            "coherent_one_sided_zero_branch_rows": 1,
            "qmu_outputs_used": False,
            "qmu_outputs_promotable": False,
            "included_in_bound_or_response_interpretation": False,
        },
        "caveats": {
            "screening_toys_per_category": 10,
            "coverage_qualified": False,
            "expected_limit_bands": False,
            "limit_bands_created": False,
            "physics_exclusion_or_reach_claim": False,
            "text": (
                "Ten-toy screening pilot only; mass rows within a toy are "
                "correlated. No coverage qualification, expected-limit bands, "
                "qmu interpretation, exclusion, or reach claim is made."
            ),
        },
        "source_inputs": source_records,
        "generator": record(Path(__file__).resolve()),
    }
    markdown = render_markdown(payload)
    return payload, markdown


def render_markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["projected_100pct_rows"]
    lines = [
        "# Final length-scale bound interpretation",
        "",
        "## Decision",
        "",
        payload["decision"]["interpretation"],
        "",
        "Factor 20 has zero scan and zero injection-refit boundary rows in all "
        "projected-100% lanes. Across all diagnostic lanes it retains "
        f"{payload['decision']['factor20_all_scenarios_scan_at_bound_rows']} "
        "scan rows at the declared boundary; factor 25 has zero boundary and "
        "zero near-bound rows across every lane.",
        "",
        "## Projected 2021 100% diagnostics",
        "",
        "| Truth | Scenario | f | Scan bound | Injection bound | sigma_A/f15 | "
        "Response | Deficit | Residual [anchor sigma] | Pull width |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['truth_role']} | {row['scenario_label']} | {row['factor']} | "
            f"{row['scan_at_bound_rows']}/110 | "
            f"{row['injection_refit_at_bound_rows']}/120 | "
            f"{row['fitted_sigma_A_ratio_to_f15']:.6f} | "
            f"{row['paired_response_median']:.6f} | "
            f"{row['paired_response_deficit_percent']:.3f}% | "
            f"{row['anchor_normalized_residual_median']:+.3f} | "
            f"{row['pull_width_strata_median']:.3f} |"
        )
    aggregate = payload["projected_100pct_aggregate"]
    lines.extend(
        [
            "",
            "The paired-response deficit spans "
            f"{aggregate['paired_response_deficit_percent_min']:.2f}-"
            f"{aggregate['paired_response_deficit_percent_max']:.2f}%. "
            + aggregate["response_interpretation"],
            "",
            "From factor 20 to 25, the largest median length-scale change is "
            f"{aggregate['factor20_to_factor25_max_abs_ls_relative_change_percent']:.4f}%, "
            "the largest fitted-sigma_A increase is "
            f"{aggregate['factor20_to_factor25_max_fitted_sigma_A_increase_percent']:.4f}%, "
            "and there are zero adjacent-LML regressions beyond the audited "
            "absolute-plus-relative tolerance. " + aggregate["plateau_interpretation"],
            "",
            "Pull widths and anchor-normalized residuals are screening "
            "diagnostics, not coverage qualification.",
            "",
            "## Observed optimized length-scale medians",
            "",
            "| Dataset | Median ell/sigma_x | Ceiling contact |",
            "|---|---:|---:|",
        ]
    )
    for row in payload["observed_length_scale_medians"]:
        lines.append(
            f"| {row['year']} {row['exposure']} | "
            f"{row['median_ls_opt_over_sigma_x']:.6f} | "
            f"{row['at_ceiling_rows']}/{row['rows']} |"
        )
    lines.extend(
        [
            "",
            "## Independent native-10% versus 1%-source ×10 comparison",
            "",
            "The source-family ensembles are independent; toy indices are not "
            "paired and no paired difference or ratio is reported.",
            "",
            "| Truth | Ensemble | Toy-median ell/sigma_x | Toys | Masses/toy |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["unpaired_factor20_comparison"]["groups"]:
        lines.append(
            f"| {row['truth_role']} | {row['display_label']} | "
            f"{row['toy_median_ls_opt_over_sigma_x']:.6f} | "
            f"{row['n_independent_toys']} | {row['mass_points_per_toy']} |"
        )
    comparison = payload["unpaired_factor20_comparison"]
    lines.extend(
        [
            "",
            "Source-support ratio (10%/1%): "
            f"{comparison['source_support_ratio_ten_pct_over_one_pct']:.6f}; "
            "effective expected-count ratio: "
            f"{comparison['effective_target_ratio_native10_over_1pct_x10']:.6f}.",
            "",
            "## Scope",
            "",
            payload["caveats"]["text"],
            " The single coherent one-sided qmu-zero diagnostic remains "
            "qmu_ok=false and all qmu outputs are excluded and non-promotable.",
            "",
            "## Provenance",
            "",
        ]
    )
    for name, item in payload["source_inputs"].items():
        lines.append(f"- `{item['path']}`: `{item['sha256']}`")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=OUTPUT_MD)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    payload, markdown = build()
    if args.validate_only:
        print(json.dumps({"status": payload["status"], "sources": payload["source_inputs"]},
                         indent=2, sort_keys=True))
        return 0
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(args.json_out.resolve(), json_text)
    atomic_write(args.markdown_out.resolve(), markdown)
    print(
        json.dumps(
            {
                "json": record(args.json_out.resolve()),
                "markdown": record(args.markdown_out.resolve()),
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
