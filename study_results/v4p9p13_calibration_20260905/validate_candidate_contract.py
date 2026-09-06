#!/usr/bin/env python3
"""Pure synthetic candidate-ledger integration test; no fitting runtime imports."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hashes(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main():
    files = [HERE / "run_sampling_refinement.py", HERE / "REFINEMENT_PROTOCOL.md",
             HERE / "collect_results.py", Path(__file__).resolve()]
    source_hashes = hashes(files)
    driver = load("candidate_contract_driver", files[0])
    collector = load("candidate_contract_collector", files[2])
    assert "calibration_core" not in sys.modules and "run_calibration" not in sys.modules
    plan = {"truths": {
        "gp": {"scan_nodes": [0., 2., 5., 12.], "proposal_nodes": [0., 2., 5., 12.],
               "candidates": []},
        "stress": {"scan_nodes": [0., 2., 5., 9., 10., 18.],
                   "proposal_nodes": [0., 2., 5., 9., 10., 18.],
                   "candidates": [{"centers": [9., 10.]}]}}}
    metrics = dict(mean_error_sd=1e-6, cov_error=2e-6, r_error=3e-6, q_error=4e-6)
    old = [dict(metrics, candidate_id=driver.BASE_CANDIDATE,
                check_stage="original_frozen_audit", truth=truth, proposal=i, passed=True)
           for truth in driver.TRUTHS for i in range(9)]
    old[3].update(passed=False, q_error=.0025)
    strict = [dict(metrics, candidate_id=driver.STRICT_CANDIDATE,
                   check_stage="original_replay", truth=truth, proposal=i, passed=True)
              for truth in driver.TRUTHS for i in range(9)]
    for truth, spec in plan["truths"].items():
        for center in driver.extended_probes(spec)[0]:
            for shift in range(3):
                strict.append(dict(metrics, candidate_id=driver.STRICT_CANDIDATE,
                    check_stage="extended", truth=truth, center=center,
                    proposal_shift=shift, passed=True))
    checks = []
    with tempfile.TemporaryDirectory(prefix="hps-candidate-contract-") as temporary:
        point = Path(temporary) / "point"
        point.mkdir()
        plan_path = point / "point_plan.json"
        plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
        checkpoint = point / "result.json"
        identity = dict(type=driver.TYPE, version=driver.VERSION,
                        numerical_policy=driver.NUMERICAL_POLICY,
                        point_plan_sha256=driver.sha(plan_path))

        def build(rows, backend, cut):
            ctx = SimpleNamespace(numerical_checks=copy.deepcopy(rows),
                                  gp_backend=backend, nuisance_cut=cut)
            audit = driver.candidate_metadata(ctx, "exact_cached_cholesky",
                {"stage": "discrepancy_gate", "failed_checks": 1}, True, plan)
            return dict(sampling_refinement=copy.deepcopy(identity), gp_backend=backend,
                nuisance_eigenvalue_cut=cut, numerical_checks=ctx.numerical_checks,
                approximation_candidate_audit=audit)

        def test(name, data, expected=False, path=checkpoint):
            result = collector.candidate_qa(data, path)
            checks.append(dict(name=name, expected_acceptance=expected,
                actual_acceptance=result["passed"], passed=result["passed"] is expected,
                error_type=result.get("error_type")))

        valid = build(old + strict, "eigenfeature_rtol_1e-15", 1e-7)
        test("strict_candidate_accepted_with_retained_old_failure", valid, True)
        assert valid["numerical_checks"][3]["passed"] is False
        bad = copy.deepcopy(valid)
        bad["numerical_checks"].pop(18)
        test("dropped_original_stage_row_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["numerical_checks"].pop()
        test("dropped_extended_stage_row_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["numerical_checks"][19] = copy.deepcopy(bad["numerical_checks"][18])
        test("duplicated_original_coordinate_at_unchanged_row_count_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["numerical_checks"][-1] = copy.deepcopy(bad["numerical_checks"][-2])
        test("duplicated_extended_coordinate_at_unchanged_row_count_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["nuisance_eigenvalue_cut"] = 1e-5
        bad["approximation_candidate_audit"]["active_nuisance_eigenvalue_cut"] = 1e-5
        test("strict_candidate_wrong_matching_top_level_cut_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["numerical_checks"][18]["q_error"] = .002
        test("active_metric_failed_despite_passed_flag_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["numerical_checks"][18]["passed"] = False
        test("active_failed_flag_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["sampling_refinement"].pop("point_plan_sha256")
        test("missing_plan_hash_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["sampling_refinement"]["point_plan_sha256"] = "0" * 64
        test("wrong_plan_hash_rejected", bad)
        test("missing_sibling_plan_rejected", valid,
             path=Path(temporary) / "missing" / "result.json")
        original_plan = plan_path.read_text()
        plan_path.write_text(original_plan + " ")
        test("modified_sibling_plan_rejected", valid)
        plan_path.write_text(original_plan)
        bad = copy.deepcopy(valid)
        bad.pop("approximation_candidate_audit")
        test("missing_derivative_candidate_audit_rejected", bad)
        bad = copy.deepcopy(valid)
        bad["approximation_candidate_audit"]["policy"] = "undeclared"
        test("wrong_numerical_policy_rejected", bad)
        rejected = copy.deepcopy(old + strict)
        rejected[-1].update(passed=False, q_error=.002)
        exact = build(rejected, "exact_cached_cholesky", 0.)
        test("exact_fallback_with_retained_candidate_failures_accepted", exact, True)
        bad = copy.deepcopy(exact)
        bad["nuisance_eigenvalue_cut"] = 1e-7
        bad["approximation_candidate_audit"]["active_nuisance_eigenvalue_cut"] = 1e-7
        test("exact_fallback_nonzero_matching_cut_rejected", bad)
        fixture = dict(plan=plan, plan_sha256=driver.sha(plan_path), original_rows=18,
            strict_replay_rows=18, strict_extended_rows=12, retained_original_failure_count=1,
            fixture_sha256=hashlib.sha256(json.dumps(valid, sort_keys=True).encode()).hexdigest())

    assert all(row["passed"] for row in checks), [row for row in checks if not row["passed"]]
    assert hashes(files) == source_hashes, "Source changed during integration test; rerun after current edit"
    contract = json.loads((HERE / "derived/contract.json").read_text())
    assert len(contract["hashes"]) == 47
    mismatches = [path for path, expected in contract["hashes"].items()
                  if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected]
    assert not mismatches, mismatches
    summary = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        test="synthetic driver.candidate_metadata to collector.candidate_qa integration",
        claim_boundary="Pure acceptance-ledger test only; no numerical GP accuracy, calibration or speed claim",
        fits_run=0, toy_draws=0, statistical_runtime_imported=False, source_sha256=source_hashes,
        source_unchanged_during_test=True, frozen_source_hashes_checked=47,
        frozen_source_hash_mismatches=0, fixture=fixture, cases=checks, case_count=len(checks),
        passed=all(row["passed"] for row in checks))
    output = HERE / "qa/candidate_contract_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(output), case_count=len(checks), passed=summary["passed"],
                         fits_run=0, source_sha256=source_hashes), indent=2))


if __name__ == "__main__":
    main()
