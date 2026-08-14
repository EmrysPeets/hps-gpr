#!/usr/bin/env python3
"""Fit the frozen residual models and run their source-only influence audit."""

from __future__ import annotations

import argparse
import csv
import json

import residual_models as models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "influence", "all", "validate"))
    args = parser.parse_args()
    if args.command in {"fit", "all"}:
        result = models.fit_all_sources()
        print(
            json.dumps(
                {
                    "status": "source_fit_complete",
                    "selected_knot_candidate": result["models"]["knot_spline"]["selected_candidate"],
                    "result": str(models.FIT_RESULT_PATH),
                },
                indent=2,
            )
        )
    if args.command in {"influence", "all"}:
        result = models.load_fit_result(require_influence=False)
        result = models.append_influence(result)
        print(
            json.dumps(
                {
                    "status": "influence_audit_complete",
                    "summaries": result["signal_influence_audit"]["summaries"],
                },
                indent=2,
            )
        )
    if args.command == "validate":
        result = models.load_fit_result(require_influence=True)
        for model in ("knot_spline", "regional_blend"):
            for source in ("one_pct", "ten_pct"):
                mean, edges = models.frozen_mean_full(model, source, result)
                support = (edges[:-1] >= models.SUPPORT[0] - 1e-12) & (
                    edges[1:] <= models.SUPPORT[1] + 1e-12
                )
                if not (mean[support] > 0).all():
                    raise models.ModelError(f"nonpositive frozen mean: {model}/{source}")
        expected_rows = 2 * 2 * 41 * 3
        actual_rows = len(result["signal_influence_audit"]["rows"])
        if actual_rows != expected_rows:
            raise models.ModelError(
                f"influence row cardinality mismatch: {actual_rows} != {expected_rows}"
            )
        for key in ("source_fit_summary_csv", "signal_influence_audit_csv"):
            record = result[key]
            path = models.HERE / record["path"]
            if models.sha256_file(path) != record["sha256"]:
                raise models.ModelError(f"stale or changed CSV artifact: {path}")
            with path.open("r", encoding="utf-8", newline="") as stream:
                if not list(csv.DictReader(stream)):
                    raise models.ModelError(f"empty CSV artifact: {path}")
        print(
            json.dumps(
                {
                    "artifact_integrity_status": "pass",
                    "scientific_qualification": {
                        model: result["models"][model][
                            "strict_generator_qualification_passed"
                        ]
                        for model in ("knot_spline", "regional_blend")
                    },
                    "protocol_sha256": models.sha256_file(models.PROTOCOL_PATH),
                    "result_sha256": models.sha256_file(models.FIT_RESULT_PATH),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
