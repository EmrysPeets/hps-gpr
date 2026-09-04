#!/usr/bin/env python3
"""Independently validate the terminal v4.9.10 Phase-A release."""

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
SPEC_PATH = REPO / "study_configs/v4p9p10_2016_full_low_control_confirmation_20260902/study_spec.json"
PROTOCOL_SHA = "3bc17d683faf50195b632416a7cbb96fb5463a93d714fcb3bff45ef5f2ec8d84"
SPEC_SHA = "680444ef63267cabd88830c0cd5e54ee40b495e8caa3a1b30b0c0ed1a016e33e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def histogram_hash(values: np.ndarray, edges: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    digest.update(np.asarray(edges, dtype=np.float64).tobytes())
    return digest.hexdigest()


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coeffs = [float(item) for item in spec["sigma_coeffs_2016"]]
    m0 = float(spec["sigma_tail_m0_2016"])
    if mass <= m0:
        return float(sum(c * mass**i for i, c in enumerate(coeffs)))
    sigma0 = float(sum(c * m0**i for i, c in enumerate(coeffs)))
    return sigma0 + float(spec["sigma_tail_slope_override_2016"]) * (mass - m0)


def coarse(values: np.ndarray, edges: np.ndarray, support_mev: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (edges[:-1] >= support_mev / 1000 - 1e-12) & (edges[1:] <= 0.210 + 1e-12)
    index = np.flatnonzero(mask)
    selected = values[index]
    assert selected.size % 5 == 0
    counts = selected.reshape(-1, 5).sum(axis=1)
    native_edges = edges[index[0]:index[-1] + 2]
    rebinned_edges = native_edges[::5]
    if rebinned_edges.size != counts.size + 1:
        rebinned_edges = np.append(rebinned_edges, native_edges[-1])
    return 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:]), counts


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    check("protocol_hash", sha256_file(HERE / "STUDY_PROTOCOL.md") == PROTOCOL_SHA)
    check("spec_hash", sha256_file(SPEC_PATH) == SPEC_SHA)

    terminal_hashes: dict[str, str] = {}
    for line in (HERE / "PHASE_A_TERMINAL_SHA256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        terminal_hashes[relative] = digest
    check(
        "terminal_hashes",
        all(sha256_file(HERE / relative) == digest for relative, digest in terminal_hashes.items()),
        terminal_hashes,
    )

    run_dir = HERE / "derived/length_factor_12"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(run_dir / "selected_cells.csv")
    attempts = pd.read_csv(run_dir / "optimizer_attempts.csv")
    decision = json.loads((HERE / "derived/length_factor_decision.json").read_text(encoding="utf-8"))
    check("manifest_complete_length12", manifest["status"] == "complete" and manifest["mode"] == "length" and float(manifest["upper_factor"]) == 12.0)
    check("manifest_protocol_spec", manifest["protocol_sha256"] == PROTOCOL_SHA and manifest["spec_sha256"] == SPEC_SHA)
    check("manifest_output_hashes", manifest["outputs"]["selected_cells.csv"]["sha256"] == sha256_file(run_dir / "selected_cells.csv") and manifest["outputs"]["optimizer_attempts.csv"]["sha256"] == sha256_file(run_dir / "optimizer_attempts.csv"))
    check("ledger_rows", len(cells) == 16 and len(attempts) == 48)
    check("fixed_grid", set(cells["support_lower_MeV"].astype(int)) == {29, 30} and set(np.round(cells["anchor_GeV"], 6)) == set(spec["development_production_geometry_anchors_GeV"]))

    source = REPO / spec["development_input"]["path"]
    check("development_file_hash", sha256_file(source) == spec["development_input"]["file_sha256"])
    with uproot.open(source) as handle:
        values, edges = handle[spec["development_input"]["histogram"]].to_numpy(flow=False)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    check("development_histogram_hash", histogram_hash(values, edges) == spec["development_input"]["histogram_sha256"])

    mask_match = True
    count_access_blind = True
    for row in cells.itertuples(index=False):
        centers, counts = coarse(values, edges, int(row.support_lower_MeV))
        half = float(spec["blind_nsigma"]) * sigma_2016(float(row.anchor_GeV), spec)
        scored = (centers >= float(row.anchor_GeV) - half - 2e-13) & (centers < np.nextafter(float(row.anchor_GeV) + half, math.inf) - 2e-13)
        trained = ~scored
        mask_match &= (
            int(row.n_train) == int(np.count_nonzero(trained))
            and int(row.n_score) == int(np.count_nonzero(scored))
            and row.train_centers_sha256 == array_hash(centers[trained])
            and row.score_centers_sha256 == array_hash(centers[scored])
            and row.train_counts_sha256 == array_hash(counts[trained])
        )
        count_access_blind &= row.score_counts_sha256 == "not_accessed"
    check("independent_phase_a_masks_and_training_hashes", mask_match)
    check("blind_window_counts_not_accessed", count_access_blind)

    forbidden = {str(item).lower() for item in spec["forbidden_selection_metrics"]}
    columns = {str(item).lower() for item in list(cells.columns) + list(attempts.columns)}
    check("no_forbidden_metric_columns", forbidden.isdisjoint(columns), sorted(forbidden & columns))

    reconstructed = []
    for keys, group in attempts.groupby(["support_lower_MeV", "anchor_GeV"], sort=True):
        eligible = group.loc[group["warning_free"].astype(bool) & group["finite_success"].astype(bool)]
        chosen = eligible.loc[eligible["lml"].idxmax()] if len(eligible) else None
        row = cells.loc[(cells["support_lower_MeV"] == keys[0]) & np.isclose(cells["anchor_GeV"], keys[1])].iloc[0]
        reproduced = 0 if chosen is None else int(np.count_nonzero(
            (np.abs(eligible["lml"] - chosen["lml"]) <= float(spec["lml_reproduction_abs_tolerance"]))
            & (np.abs(eligible["length"] - chosen["length"]) <= float(spec["length_reproduction_rel_tolerance"]) * abs(float(chosen["length"])))
        ))
        reconstructed.append(
            int(row.warning_free_repeat_count) == len(eligible)
            and int(row.reproduced_warning_free_count) == reproduced
            and (chosen is None or int(row.selected_seed) == int(chosen["seed"]))
        )
    check("warning_free_branch_selection_reconstructed", all(reconstructed))

    failures = cells.loc[~cells["technical_pass"].astype(bool)]
    exact_failure = (
        len(failures) == 1
        and int(failures.iloc[0]["support_lower_MeV"]) == 29
        and math.isclose(float(failures.iloc[0]["anchor_GeV"]), 0.090)
        and int(failures.iloc[0]["warning_free_repeat_count"]) == 1
        and int(failures.iloc[0]["reproduced_warning_free_count"]) == 1
        and not bool(failures.iloc[0]["length_at_upper"])
        and bool(failures.iloc[0]["covariance_ok"])
    )
    check("exact_non_upper_failure", exact_failure)
    check("no_upper_contacts", int(cells["length_at_upper"].astype(bool).sum()) == 0)
    check("canonical_terminal_decision", decision["status"] == "stopped_non_upper_bound_technical_failure" and decision["selected_upper_factor"] is None and decision["next_factor_authorized"] is None)
    check("decision_input_hash", decision["selected_cells_sha256"] == sha256_file(run_dir / "selected_cells.csv"))

    absent = [
        HERE / "FROZEN_LENGTH_FACTOR_SHA256",
        HERE / "derived/full_low_control_confirmation",
        HERE / "derived/support_freeze_decision.json",
        HERE / "derived/observed_production",
    ]
    check("no_phase_b_or_c_artifacts", all(not path.exists() for path in absent), [str(path.relative_to(HERE)) for path in absent if path.exists()])
    check("manifest_uses_development_not_confirmation", manifest["input"]["file_sha256"] == spec["development_input"]["file_sha256"] and manifest["input"]["file_sha256"] != spec["confirmation_input"]["file_sha256"])

    passed = sum(int(item["pass"]) for item in checks)
    report = {
        "status": "pass" if passed == len(checks) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "canonical_decision_sha256": sha256_file(HERE / "derived/length_factor_decision.json"),
        "validator_sha256": sha256_file(Path(__file__)),
    }
    output = HERE / "qa/final_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
