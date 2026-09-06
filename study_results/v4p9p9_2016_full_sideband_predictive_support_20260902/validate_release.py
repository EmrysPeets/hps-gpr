#!/usr/bin/env python3
"""Independent fail-closed validation of the stopped v4.9.9 selector."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uproot


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "study_configs/v4p9p9_2016_full_sideband_predictive_support_20260902"
FROZEN = {
    HERE / "STUDY_PROTOCOL.md": "38e82537d04330ce66d5e39007df03bb1c7269fb62a16557e3ee96d6d2f380b2",
    CONFIG / "study_spec.json": "f9f410977114cc9a6a9ea3ad381782a017b90cf2a86c7c8e3b2c9db89f3cfecd",
    HERE / "PROTOCOL_AMENDMENT_PRE_EXECUTION.md": "f45dbad8ee99f22d8500e4a8effd74b35854e2f694a8bd44ec4704e6b500d14c",
    CONFIG / "preexecution_amendment.json": "c90946c38d356c5c597627fb242f6437784da9ae2110daa4547b010f602ee0cd",
    HERE / "PROTOCOL_AMENDMENT2_PRE_EXECUTION.md": "d37a934e91d595123fe9ef543a3bb2ae7dce23aa78859d4747aa536360eb4b9e",
    CONFIG / "preexecution_amendment2.json": "0526dafba3094d5463225839f1e8ff8f94e011b83bcc8217a63ff5ee9cc6c768",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def hist_hash(values: np.ndarray, edges: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    digest.update(np.asarray(edges, dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, passed: bool, detail: Any, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def rebin(
    values: np.ndarray,
    edges: np.ndarray,
    lower: float,
    upper: float = 0.210,
    factor: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = 1e-12
    mask = (edges[:-1] >= lower - tolerance) & (edges[1:] <= upper + tolerance)
    indices = np.flatnonzero(mask)
    selected = values[indices]
    if selected.size % factor:
        raise RuntimeError("nondivisible independent rebin")
    counts = selected.reshape(-1, factor).sum(axis=1)
    native_edges = edges[indices[0] : indices[-1] + 2]
    coarse_edges = native_edges[::factor]
    if coarse_edges.size != counts.size + 1:
        coarse_edges = np.append(coarse_edges, native_edges[-1])
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    return np.asarray(centers), np.asarray(counts)


def interval(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return (values >= low - 2e-13) & (values < high - 2e-13)


def main() -> None:
    checks: list[dict[str, Any]] = []
    for path, expected in FROZEN.items():
        actual = sha256_file(path) if path.is_file() else None
        check(
            f"frozen_hash.{path.name}",
            actual == expected,
            {"expected": expected, "actual": actual},
            checks,
        )

    spec = load_json(CONFIG / "study_spec.json")
    declaration = spec["development"]
    root_path = REPO / declaration["path"]
    check(
        "input.root_sha256",
        sha256_file(root_path) == declaration["file_sha256"],
        sha256_file(root_path),
        checks,
    )
    with uproot.open(root_path) as handle:
        values, edges = handle[declaration["histogram"]].to_numpy(flow=False)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    check(
        "input.histogram_sha256",
        hist_hash(values, edges) == declaration["histogram_sha256"],
        hist_hash(values, edges),
        checks,
    )

    out = HERE / "derived/development"
    attempts_path = out / "optimizer_attempts.csv"
    scores_path = out / "selected_predictive_scores.csv"
    manifest = load_json(out / "run_manifest.json")
    attempts = pd.read_csv(attempts_path)
    scores = pd.read_csv(scores_path)
    check(
        "manifest.attempts",
        manifest["outputs"]["optimizer_attempts.csv"]["sha256"]
        == sha256_file(attempts_path)
        and manifest["outputs"]["optimizer_attempts.csv"]["rows"] == len(attempts),
        {"rows": len(attempts), "sha256": sha256_file(attempts_path)},
        checks,
    )
    check(
        "manifest.scores",
        manifest["outputs"]["selected_predictive_scores.csv"]["sha256"]
        == sha256_file(scores_path)
        and manifest["outputs"]["selected_predictive_scores.csv"]["rows"] == len(scores),
        {"rows": len(scores), "sha256": sha256_file(scores_path)},
        checks,
    )
    check("counts.attempts_420", len(attempts) == 420, len(attempts), checks)
    check("counts.scores_140", len(scores) == 140, len(scores), checks)
    check(
        "optimizer.random_states_exact",
        set(attempts["seed"].astype(int)) == {1961, 5813, 9049},
        sorted(set(attempts["seed"].astype(int))),
        checks,
    )

    # Independently rebuild every train and score mask and compare hashes/counts.
    mask_failures: list[dict[str, Any]] = []
    blocks = {str(k): [float(x) for x in v] for k, v in spec["blocks"].items()}
    for _, row in scores.iterrows():
        region = str(row["region"])
        if region == "low":
            lower = float(row["support_lower_MeV"]) / 1000.0
            centers, counts = rebin(values, edges, lower)
            allowed = interval(centers, lower, 0.03875)
        else:
            centers, counts = rebin(values, edges, 0.030)
            allowed = interval(centers, 0.181, 0.210)
        block_low, block_high = blocks[str(row["block"])]
        holdout = interval(centers, block_low, block_high)
        train = allowed & ~holdout
        score = allowed & holdout
        search = interval(centers, 0.039, np.nextafter(0.180, math.inf))
        reconstructed = {
            "n_train": int(np.count_nonzero(train)),
            "n_score": int(np.count_nonzero(score)),
            "train_centers_sha256": array_hash(centers[train]),
            "score_centers_sha256": array_hash(centers[score]),
            "train_counts_sha256": array_hash(counts[train]),
            "score_counts_sha256": array_hash(counts[score]),
            "n_forbidden_search_train_centers": int(np.count_nonzero(train & search)),
            "n_forbidden_search_score_centers": int(np.count_nonzero(score & search)),
        }
        for key, expected in reconstructed.items():
            observed = row[key]
            if str(observed) != str(expected) and not (
                isinstance(expected, int) and int(observed) == expected
            ):
                mask_failures.append(
                    {
                        "region": region,
                        "support": row["support_lower_MeV"],
                        "anchor": row["anchor_GeV"],
                        "block": row["block"],
                        "field": key,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    check(
        "masks.independent_reconstruction",
        not mask_failures,
        {"failure_count": len(mask_failures), "first": mask_failures[:3]},
        checks,
    )
    check(
        "masks.zero_search_centers",
        int(scores["n_forbidden_search_train_centers"].sum()) == 0
        and int(scores["n_forbidden_search_score_centers"].sum()) == 0,
        {
            "train": int(scores["n_forbidden_search_train_centers"].sum()),
            "score": int(scores["n_forbidden_search_score_centers"].sum()),
        },
        checks,
    )

    # Independently evaluate the terminal condition and gross-misfit guards.
    low30 = scores.loc[(scores["region"] == "low") & (scores["support_lower_MeV"] == 30)]
    high = scores.loc[scores["region"] == "high"]
    reference_absolute = bool(
        low30["mahalanobis_per_bin"].mean() < 4.0
        and low30["mahalanobis_per_bin"].max() < 9.0
        and low30["max_abs_marginal_standardized_residual"].max() < 5.0
    )
    high_absolute = bool(
        high["mahalanobis_per_bin"].mean() < 4.0
        and high["mahalanobis_per_bin"].max() < 9.0
        and high["max_abs_marginal_standardized_residual"].max() < 5.0
    )
    reference_technical = bool(low30["technical_pass"].astype(bool).all())
    high_technical = bool(high["technical_pass"].astype(bool).all())
    high_bound_contacts = int(high["length_at_bound"].astype(bool).sum())
    check(
        "gates.reference_low_pass",
        reference_absolute and reference_technical,
        {
            "absolute": reference_absolute,
            "technical": reference_technical,
            "mean_mahalanobis": float(low30["mahalanobis_per_bin"].mean()),
        },
        checks,
    )
    check(
        "gates.high_absolute_pass",
        high_absolute,
        {
            "mean_mahalanobis": float(high["mahalanobis_per_bin"].mean()),
            "max_mahalanobis": float(high["mahalanobis_per_bin"].max()),
            "max_abs_marginal_z": float(
                high["max_abs_marginal_standardized_residual"].max()
            ),
        },
        checks,
    )
    check(
        "gates.high_technical_failure_reproduced",
        not high_technical and high_bound_contacts == len(high) == 20,
        {
            "technical_pass": high_technical,
            "length_bound_contacts": high_bound_contacts,
            "cells": len(high),
        },
        checks,
    )

    decision_path = out / "phase1_decision.json"
    decision = load_json(decision_path)
    check(
        "decision.terminal_status",
        decision["status"] == "stopped_development_absolute_or_technical_failure",
        decision["status"],
        checks,
    )
    check(
        "decision.no_qualifier_or_confirmation",
        decision["phase1_qualifiers"] == []
        and decision["supports_to_confirm"] == []
        and not (HERE / "derived/confirmation").exists()
        and not (HERE / "observed_2016").exists(),
        {
            "qualifiers": decision["phase1_qualifiers"],
            "supports_to_confirm": decision["supports_to_confirm"],
            "confirmation_exists": (HERE / "derived/confirmation").exists(),
            "observed_exists": (HERE / "observed_2016").exists(),
        },
        checks,
    )
    forbidden_columns = {
        "A_hat", "signal_pull", "p0", "Z", "upper_limit", "epsilon2", "eps2_up", "A_up"
    }
    present = sorted(forbidden_columns & set(scores.columns))
    check("scores.no_forbidden_metrics", not present, present, checks)

    payload = {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": spec["study_id"],
        "canonical_state": "stopped_development_absolute_or_technical_failure",
        "selected_support": None,
        "full_confirmation_authorized": False,
        "observed_2016_production_authorized": False,
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": sum(not item["passed"] for item in checks),
        "critical_evidence": {
            "phase1_decision_sha256": sha256_file(decision_path),
            "selected_predictive_scores_sha256": sha256_file(scores_path),
            "optimizer_attempts_sha256": sha256_file(attempts_path),
            "support_summary_sha256": sha256_file(out / "support_summary.csv"),
            "high_control_summary_sha256": sha256_file(out / "high_control_summary.json"),
        },
        "claim_boundary": "failed prospective out-of-search support qualification; not an observed result, coverage statement, sensitivity, exclusion, or significance",
    }
    qa = HERE / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    output = qa / "final_validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
