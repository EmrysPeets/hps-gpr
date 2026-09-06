#!/usr/bin/env python3
"""Collect frozen calibration checkpoints without importing any fitting runtime.

Repeat --input-dir for refinements: a later directory replaces a whole earlier
checkpoint; toys are never pooled. Source contracts must agree, although toy
counts can differ. Default output is summary/; every parent grid row is retained
and unfinished coordinates have explicit missing-checkpoint statuses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np
from scipy.stats import beta, binomtest, norm, t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / "study_results/v4p9p13_background_profiling_20260905/observed/derived/observed_fixed_comparison.csv"
SCOPES = {"individual_2015_full": (19, 90), "individual_2016_full": (39, 180),
          "individual_2021_10pct": (50, 250), "all_2015_2016_2021": (50, 90)}
EXPECTED = {(scope, mass) for scope, bounds in SCOPES.items()
            for mass in range(bounds[0], bounds[1] + 1)}
METHODS = ("profiled", "fixed")
TRUTHS = ("gp", "stress")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value):
    return float(value) if finite(value) else None


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, payload):
    atomic_write(path, lambda stream: json.dump(clean(payload), stream, indent=2, allow_nan=False))


def write_csv(path, rows, minimum_columns=()):
    columns = list(dict.fromkeys([*minimum_columns, *(key for row in rows for key in row)]))
    def writer(stream):
        output = csv.DictWriter(stream, fieldnames=columns)
        output.writeheader()
        for row in rows:
            output.writerow({key: json.dumps(clean(value), sort_keys=True) if isinstance(value, (dict, list))
                             else clean(value) for key, value in row.items()})
    atomic_write(path, writer)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def count(value):
    result = int(float(value))
    if float(value) != result:
        raise ValueError(f"Noninteger count: {value}")
    return result


def exact_interval(k, n):
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k)))


def holm(values):
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values)-rank)*values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def validate_counts(row):
    n = count(row["n"])
    if n <= 0:
        raise ValueError("Nonpositive validation ensemble size")
    for key in ("exclusion_count", "raw_exclusion_count", "local_rejection_count"):
        k = count(row[key])
        if not 0 <= k <= n:
            raise ValueError(f"Invalid {key}: {k}/{n}")
        row[key] = k
    raw_local = float(row["raw_local_rejection_fraction"]) * n
    if abs(raw_local - round(raw_local)) > 1e-7:
        raise ValueError("Raw local rejection fraction does not reconstruct integer counts")
    row["raw_local_rejection_count"] = int(round(raw_local))
    row["n"] = n
    return row


def validation_toy_moments(data, path, validation):
    """Audit saved validation amplitudes and attach separate-cell MC errors."""
    toy_path = path.parent / "validation_toys.csv.gz"
    payload = toy_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sigma = number(data.get("sigma_reference"))
    if sigma is None or sigma <= 0:
        raise ValueError(f"Invalid fixed reference uncertainty: {path}")
    cells = {(row["truth"], count(row["strength"]), row["method"]): row for row in validation}
    groups = {key: {} for key in cells}
    reader = csv.DictReader(io.StringIO(gzip.decompress(payload).decode("utf-8")))
    if not {"truth", "strength", "method", "toy_id", "Ahat"}.issubset(reader.fieldnames or ()):
        raise ValueError(f"Missing validation toy columns: {toy_path}")
    for toy in reader:
        key = toy["truth"], count(toy["strength"]), toy["method"]
        if key not in groups:
            raise ValueError(f"Unexpected validation toy cell {key}: {toy_path}")
        index, amplitude = count(toy["toy_id"]), number(toy["Ahat"])
        if index in groups[key] or amplitude is None:
            raise ValueError(f"Duplicate toy ID or nonfinite amplitude in {key}: {toy_path}")
        groups[key][index] = amplitude
    checks = []
    for key, row in cells.items():
        truth, strength, method = key
        n = count(row["n"])
        if n < 2 or set(groups[key]) != set(range(n)):
            raise ValueError(f"Validation toy count/IDs disagree with summary in {key}: {toy_path}")
        values = np.array([groups[key][i] for i in range(n)])
        mean, sample_sd = float(values.mean()), float(values.std(ddof=1))
        saved_mean, saved_bias = number(row["Ahat_mean"]), number(row["signal_bias_sigma"])
        true_amplitude = number(row["Atrue"])
        if saved_mean is None or saved_bias is None or true_amplitude is None:
            raise ValueError(f"Nonfinite saved validation moment in {key}: {path}")
        tolerance = 1e-10 * max(sigma, abs(saved_mean), 1.)
        bias = (mean-true_amplitude) / sigma
        mean_error, bias_error = abs(mean-saved_mean), abs(bias-saved_bias)
        if mean_error > tolerance or bias_error > 1e-10 * max(1., abs(saved_bias)):
            raise ValueError(f"Validation toy mean disagrees with saved summary in {key}: {toy_path}")
        if not math.isclose(true_amplitude, strength*sigma, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Validation true amplitude/reference mismatch in {key}: {path}")
        mean_se = sample_sd / math.sqrt(n)
        bias_se = mean_se / sigma
        halfwidth = float(student_t.ppf(.975, n-1)) * bias_se
        row.update(sigma_reference=sigma, Ahat_mean_from_toys=mean, Ahat_sample_sd=sample_sd,
            Ahat_mean_mc_se=mean_se, signal_bias_sigma_mc_se=bias_se,
            signal_bias_sigma_mc95_low=saved_bias-halfwidth,
            signal_bias_sigma_mc95_high=saved_bias+halfwidth,
            validation_toy_count=n, validation_toy_ledger_sha256=digest,
            validation_toy_moments_passed=True)
        influence = data.get("provenance", {}).get(truth, {}).get("meta", {}).get("influence", {}).get(method, {})
        linear_bias, linear_sd = number(influence.get("bias")), number(influence.get("sd"))
        available = linear_bias is not None and linear_sd is not None and linear_sd >= 0
        row.update(linearized_zero_noise_bias_sigma=linear_bias/sigma if strength == 0 and available else None,
            linearized_sampling_sd_sigma=linear_sd/sigma if strength == 0 and available else None,
            linearized_comparator_status=("background_only_approximation" if available else "missing_metadata")
                                         if strength == 0 else "not_applicable_signal_injected")
        checks.append(dict(truth=truth, strength=strength, method=method, n=n, passed=True,
                           mean_difference_sigma=mean_error/sigma, bias_difference_sigma=bias_error))
    return dict(path=str(toy_path), sha256=digest, rows=sum(len(g) for g in groups.values()),
                cells=len(cells), all_passed=True, checks=checks)


def validation_tests(rows, complete):
    families = {}
    for row in rows:
        positive = float(row["strength"]) > 0
        for prefix, kfield, target, applicable in (
                ("exclusion", "exclusion_count", .10, positive),
                ("raw_exclusion", "raw_exclusion_count", .10, positive),
                ("local", "local_rejection_count", .05, not positive),
                ("raw_local", "raw_local_rejection_count", .05, not positive)):
            k, n = row[kfield], row["n"]
            low, high = exact_interval(k, n)
            row[prefix + "_fraction"] = k / n
            row[prefix + "_ci95_low"] = low
            row[prefix + "_ci95_high"] = high
            row[prefix + "_test_null"] = target if applicable else None
            row[prefix + "_pvalue"] = float(binomtest(k, n, target, alternative="greater").pvalue) if applicable else None
            row[prefix + "_holm_pvalue"] = None
            row[prefix + "_holm_reject_0p05"] = None
            if applicable:
                families.setdefault(prefix, []).append(row)
        row["validation_family_complete"] = complete
        row["local_rejection_interpretation"] = "false_positive_fraction" if not positive else "power"
    summaries = {}
    for prefix, family in families.items():
        for row, value in zip(family, holm([r[prefix + "_pvalue"] for r in family])):
            row[prefix + "_holm_pvalue"] = value
            row[prefix + "_holm_reject_0p05"] = value < .05
        rejected = [row for row in family if row[prefix + "_holm_reject_0p05"]]
        summaries[prefix] = dict(n_tests=len(family), expected_tests=456*(8 if "exclusion" in prefix else 4),
            rejected_count=len(rejected), minimum_adjusted_pvalue=min(r[prefix + "_holm_pvalue"] for r in family),
            complete=complete, status="pending_grid_completion" if not complete else
            ("excess_rejection_detected" if rejected else "no_excess_rejection_detected"),
            rejected_cells=[{k: row[k] for k in ("scope_key", "mass_MeV", "truth", "strength", "method", "n", prefix + "_holm_pvalue")}
                            for row in rejected])
    return summaries


def pzero_result(record):
    p = record.get("pzero", {})
    estimate, se = number(p.get("p0")), number(p.get("se"))
    status = p.get("status", "missing")
    if estimate is None:
        return None, se, "unresolved_nonfinite", None
    if estimate == 0:
        return None, se, "unresolved_zero_tail", estimate
    if estimate > 1 and se is not None and se > 0 and estimate <= 1+3*se:
        return 1.0, se, "mc_boundary", estimate
    if not 0 < estimate <= 1:
        return None, se, "unresolved_out_of_range", estimate
    if status not in ("resolved", "bounded_atom") or se is None:
        status = "limited_mc"
    return estimate, se, status, estimate


def candidate_qa(data, checkpoint_path=None):
    """Accept a later approximation only through its complete, explicit audit."""
    if not __debug__:
        return dict(present=True, passed=False, error_type="optimized_python_disables_audit_assertions")
    audit = data.get("approximation_candidate_audit")
    if audit is None:
        return dict(present=False, passed=not bool(data.get("sampling_refinement")))
    try:
        assert audit["schema_version"] == 1
        assert data["sampling_refinement"]["type"] == "independent_poisson_mixture_refinement"
        assert audit["policy"] == "original_proposals_then_nuisance1e-7_replay_v1"
        plan_path = Path(checkpoint_path).parent / "point_plan.json"
        assert sha(plan_path) == data["sampling_refinement"]["point_plan_sha256"]
        plan = json.loads(plan_path.read_text())
        expected_extended = {
            (truth, center, shift)
            for truth, spec in plan["truths"].items()
            for center in {max(spec["proposal_nodes"]), *[c for row in spec["candidates"] for c in row["centers"]]}
            for shift in range(3)
        }
        assert set(plan["truths"]) == {"gp", "stress"}
        assert audit["active_candidate_accepted"] is True
        assert audit["active_backend"] == data["gp_backend"]
        assert audit["active_nuisance_eigenvalue_cut"] == data["nuisance_eigenvalue_cut"]
        definitions = {
            "eigenfeature_rtol1e-15_nuisance1e-5": (1e-5, ["original_frozen_audit", "extended"]),
            "eigenfeature_rtol1e-15_nuisance1e-7": (1e-7, ["original_replay", "extended"]),
        }
        numerical = data["numerical_checks"]
        records = {c["candidate_id"]: c for c in audit["candidates"]}
        assert len(audit["candidates"]) == len(records) == 2 and set(records) == set(definitions)
        assert all(row.get("candidate_id") in definitions for row in numerical)
        for cid, (cut, required) in definitions.items():
            record = records[cid]
            indices = [i for i, row in enumerate(numerical) if row["candidate_id"] == cid]
            stages = {stage: [i for i in indices if numerical[i].get("check_stage") == stage] for stage in required}
            assert record["check_indices"] == indices
            assert record["required_stages"] == required and record["stage_check_indices"] == stages
            assert record["target_backend"] == "eigenfeature_rtol_1e-15" and record["nuisance_eigenvalue_cut"] == cut
            assert all(numerical[i].get("check_stage") in required for i in indices)
            counts = {required[0]: 18, "extended": len(expected_extended)}
            assert record["required_check_counts"] == counts
            accepted = (len(indices) == sum(counts.values())
                        and all(len(stages[stage]) == counts[stage] for stage in required)
                        and all(numerical[i].get("passed") is True for i in indices))
            assert record["status"] == ("accepted" if accepted else "rejected" if indices else "not_attempted")
        active = audit["active_candidate_id"]
        if active == "exact_cached_cholesky":
            assert data["gp_backend"] == active and data["nuisance_eigenvalue_cut"] == 0.
        else:
            record = records[active]
            assert record["status"] == "accepted"
            assert data["gp_backend"] == record["target_backend"]
            assert data["nuisance_eigenvalue_cut"] == record["nuisance_eigenvalue_cut"]
            original, extended = record["required_stages"]
            original_rows = [numerical[i] for i in record["stage_check_indices"][original]]
            assert len(original_rows) == 18
            assert {(r["truth"], r["proposal"]) for r in original_rows} == {(t, i) for t in ("gp", "stress") for i in range(9)}
            extended_rows = [numerical[i] for i in record["stage_check_indices"][extended]]
            assert {(r["truth"], r["center"], r["proposal_shift"]) for r in extended_rows} == expected_extended
            for i in record["check_indices"]:
                assert all(finite(numerical[i].get(k)) and 0 <= numerical[i][k] < .001
                           for k in ("mean_error_sd", "cov_error", "r_error", "q_error"))
        return dict(present=True, passed=True, active_candidate_id=active, audit=audit)
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        return dict(present=True, passed=False, error_type=type(error).__name__, audit=audit)


def point_qa(data, path, validation):
    numerical = data.get("numerical_checks", [])
    scalar = data.get("scalar_checks", [])
    backend = data.get("gp_backend", "missing")
    rejected = [row for row in numerical if not row.get("passed", False)]
    scalar_pass = bool(scalar) and all(row.get("passed", False) for row in scalar)
    scores = [number(p.get("max_score")) for p in data.get("provenance", {}).values()]
    scores += [number(row.get("max_score")) for row in validation]
    scores = [score for score in scores if score is not None]
    convergence = bool(scores) and max(scores) < 2e-7
    approximation_ok = bool(numerical) and not rejected
    exact_fallback = backend == "exact_cached_cholesky"
    candidate = candidate_qa(data, path)
    approximation_accepted = candidate["present"] and candidate["passed"]
    numerical_pass = scalar_pass and convergence and candidate["passed"] and (approximation_ok or exact_fallback or approximation_accepted)
    return dict(scope_key=data["scope_key"], mass_MeV=data["mass_MeV"], checkpoint=str(path),
        status=("exact_fallback_success" if exact_fallback and rejected else "numerical_checks_passed")
               if numerical_pass else "missing_or_failed_numerical_audit",
        numerical_pass=numerical_pass, gp_backend=backend, max_score=max(scores, default=None),
        approximation_candidate_qa=candidate,
        rejected_approximation_checks=len(rejected), scalar_checks_passed=scalar_pass,
        gp_fallback_reason=data.get("gp_fallback_reason"), nuisance_eigenvalue_cut=data.get("nuisance_eigenvalue_cut"),
        numerical_checks=numerical, scalar_checks=scalar, prediction_ledger=data.get("prediction_ledger", []),
        calibration_provenance=data.get("provenance", {}))


def scope_metrics(rows):
    def distribution(values):
        values = [float(value) for value in values if finite(value)]
        return dict(n=len(values), minimum=min(values, default=None),
                    median=float(np.median(values)) if values else None, maximum=max(values, default=None))
    summary = {}
    for scope in SCOPES:
        collected = [r for r in rows if r["scope_key"] == scope and r["checkpoint_completed"]]
        resolved = [r for r in collected if all(r["status_"+method] == "resolved" for method in METHODS)]
        item = dict(completed_points=len(collected), both_methods_mc_resolved_points=len(resolved),
            fixed_over_profiled_calibrated_all_finite=distribution(r["ratio_fixed_over_profiled_calibrated"] for r in collected),
            fixed_over_profiled_calibrated_both_resolved=distribution(r["ratio_fixed_over_profiled_calibrated"] for r in resolved))
        for method, raw_column in (("profiled", "eps2_current_display"), ("fixed", "eps2_fixed_display")):
            item[method+"_calibrated_over_parent_asymptotic_all_finite"] = distribution(
                float(r[f"eps2_{method}_calibrated"]) / float(r[raw_column]) for r in collected
                if finite(r[f"eps2_{method}_calibrated"]) and float(r[raw_column]) > 0)
            local = [r for r in collected if r[f"status_p0_{method}"] in ("resolved", "bounded_atom")
                     and finite(r[f"p0_{method}_calibrated"])]
            minimum = min(local, key=lambda r: r[f"p0_{method}_calibrated"], default=None)
            item[method+"_minimum_resolved_conditional_local_p0"] = None if minimum is None else dict(
                mass_MeV=minimum["mass_MeV"], p0=minimum[f"p0_{method}_calibrated"],
                mc_se=minimum[f"p0_{method}_mc_se"], status=minimum[f"status_p0_{method}"])
        summary[scope] = item
    return summary


def populate_point(row, data, path, truth_rows):
    factor = float(row["dimuon_factor"])
    conversion = float(data["signal_yield_per_eps2"])
    expected_conversion = float(row["signal_yield_per_eps2_fitted_window"])
    if not math.isclose(conversion, expected_conversion, rel_tol=2e-10, abs_tol=0):
        raise ValueError(f"Signal conversion mismatch at {path}")
    records = {(r["method"], r["truth"]): r for r in data["results"]}
    if len(records) != 4 or set(records) != {(m, t) for m in METHODS for t in TRUTHS}:
        raise ValueError(f"Missing/duplicate method-truth results at {path}")
    row.update(checkpoint_completed=True, checkpoint_status=data["status"], checkpoint_path=str(path),
               ntoys_per_proposal=data["ntoys_per_proposal"], nvalidation=data["nvalidation"],
               sigma_reference=data["sigma_reference"], gp_backend=data.get("gp_backend"))
    for method in METHODS:
        asym = data["observed"][method]
        row[f"eps2_{method}_asymptotic_ee_raw"] = float(asym["A90"]) / conversion
        row[f"eps2_{method}_asymptotic"] = row[f"eps2_{method}_asymptotic_ee_raw"] * factor
        row[f"p0_{method}_asymptotic"] = float(norm.sf(max(float(asym["signed_r"]), 0)))
        row[f"signed_r_{method}_asymptotic"] = float(asym["signed_r"])
        subset = [records[method, truth] for truth in TRUTHS]
        censored = any(r.get("status") == "right_censored" or not finite(r.get("ul_sigma")) for r in subset)
        status = "right_censored" if censored else "resolved" if all(r.get("status") == "resolved" for r in subset) else "limited_mc"
        row["status_" + method] = status
        for input_key, output_key in (("eps2", f"eps2_{method}_calibrated"),
                                      ("eps2_low", f"eps2_{method}_mc_low"),
                                      ("eps2_high", f"eps2_{method}_mc_high")):
            values = [number(r.get(input_key)) for r in subset]
            value = None if censored or any(v is None for v in values) else max(values)
            row[output_key + "_ee_raw"] = value
            row[output_key] = None if value is None else value * factor
        if status == "resolved" and any(row[f"eps2_{method}_{tail}"] is None for tail in ("calibrated", "mc_low", "mc_high")):
            row["status_" + method] = "limited_mc_nonfinite_endpoint"
        row[f"limiting_truth_{method}"] = None if censored else max(TRUTHS, key=lambda t: records[method, t]["eps2"])
        pvalues = [pzero_result(record) for record in subset]
        if any(value[0] is None for value in pvalues):
            pvalue, se, pstatus, active = None, None, "unresolved_tail", None
        else:
            index = max(range(2), key=lambda i: pvalues[i][3])
            pvalue, se = pvalues[index][:2]
            active = TRUTHS[index]
            pstatus = "bounded_atom" if all(v[2] == "bounded_atom" for v in pvalues) else "mc_boundary" if any(v[2] == "mc_boundary" for v in pvalues) else "resolved" if all(v[2] in ("resolved", "bounded_atom") for v in pvalues) else "limited_mc"
        row.update({f"p0_{method}_calibrated": pvalue, f"p0_{method}_mc_se": se,
                    f"status_p0_{method}": pstatus, f"p0_limiting_truth_{method}": active,
                    f"p0_{method}_weighted_estimate_unclipped": max(v[3] for v in pvalues) if all(v[3] is not None for v in pvalues) else None,
                    f"log_p0_{method}_calibrated": math.log(pvalue) if pvalue else None})
        for truth, record, pv in zip(TRUTHS, subset, pvalues):
            output = dict(scope_key=data["scope_key"], mass_MeV=data["mass_MeV"], method=method, truth=truth,
                          checkpoint_status=data["status"], checkpoint_path=str(path), dimuon_factor=factor,
                          ntoys_per_proposal=data["ntoys_per_proposal"], nvalidation=data["nvalidation"])
            output.update({k: v for k, v in record.items() if k not in ("trace", "pzero", "eps2", "eps2_low", "eps2_high")})
            for source, target in (("eps2", "eps2"), ("eps2_low", "eps2_mc_low"), ("eps2_high", "eps2_mc_high")):
                value = number(record.get(source))
                output[target + "_ee_raw"] = value
                output[target + "_display"] = None if value is None else factor * value
            output.update(p0=pv[0], p0_mc_se=pv[1], status_p0=pv[2], p0_weighted_estimate_unclipped=pv[3],
                          p0_ess=record.get("pzero", {}).get("ess"), source_p0_status=record.get("pzero", {}).get("status"))
            truth_rows.append(output)
    denominator, numerator = row["eps2_profiled_calibrated"], row["eps2_fixed_calibrated"]
    row["ratio_fixed_over_profiled_calibrated"] = numerator / denominator if denominator and numerator is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, help="Repeat for whole-point refinements; later directories take precedence")
    parser.add_argument("--output-dir", type=Path, default=HERE / "summary")
    parser.add_argument("--allow-source-drift", action="store_true", help="For separately labelled historical/pilot collection; never enables a final validated status")
    args = parser.parse_args()
    inputs = [(p if p.is_absolute() else Path.cwd()/p).resolve() for p in (args.input_dir or [HERE/"derived"])]
    output = args.output_dir.resolve()
    if output in inputs:
        raise ValueError("Summary output must not be an input checkpoint directory")
    parent_rows = read_csv(PARENT)
    parent = {(row["scope_key"], int(row["mass_MeV"])): row for row in parent_rows}
    if len(parent_rows) != 456 or set(parent) != EXPECTED:
        raise ValueError("Parent does not match the exact 456-point grid")
    contracts, source_audit, chosen, replacements, pending, failures = [], {}, {}, [], [], []
    reference_hashes = None
    baseline_contract_sha = None
    for directory in inputs:
        contract_path = directory / "contract.json"
        contract = json.loads(contract_path.read_text())
        sampling = contract.get("sampling_refinement")
        sampling_hashes = contract.get("sampling_hashes", {})
        if baseline_contract_sha is None:
            if sampling or sampling_hashes:
                raise ValueError("The original production contract must be the first input")
            baseline_contract_sha = sha(contract_path)
        if sampling or sampling_hashes:
            if (not isinstance(sampling, dict) or sampling.get("type") != "independent_poisson_mixture_refinement"
                    or sampling.get("version") != 1 or sampling.get("baseline_contract_sha256") != baseline_contract_sha
                    or not sampling_hashes or contract.get("ntoy") is not None):
                raise ValueError("Unrecognized sampling derivative; collect scientific procedures separately")
            if set(sampling_hashes) & set(contract["hashes"]):
                raise ValueError("Sampling source hashes must not override frozen inference sources")
        if reference_hashes is not None and contract["hashes"] != reference_hashes:
            raise ValueError("Input source contracts differ; collect them separately rather than pooling scientific procedures")
        reference_hashes = contract["hashes"]
        contracts.append(dict(directory=str(directory), path=str(contract_path), sha256=sha(contract_path), contract=contract))
        for relative, expected in {**contract["hashes"], **sampling_hashes}.items():
            if relative in source_audit and source_audit[relative]["expected_sha256"] != expected:
                raise ValueError(f"Conflicting source identities: {relative}")
            if relative not in source_audit:
                source = ROOT / relative
                actual = sha(source) if source.is_file() else None
                source_audit[relative] = dict(expected_sha256=expected, actual_sha256=actual, passed=actual == expected)
        for path in sorted(directory.glob("*/m*/result.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as error:
                pending.append(dict(path=str(path), reason=str(error)))
                continue
            key = data["scope_key"], count(data["mass_MeV"])
            if key not in EXPECTED or path.parent.name != f"m{key[1]:03}" or path.parent.parent.name != key[0]:
                raise ValueError(f"Unexpected checkpoint coordinate: {path}")
            if data["ntoys_per_proposal"] != contract["ntoy"] or data["nvalidation"] != contract["nvalid"]:
                raise ValueError(f"Checkpoint ensemble counts differ from its contract: {path}")
            if sampling:
                marker = data.get("sampling_refinement", {})
                if marker.get("type") != sampling["type"] or marker.get("baseline_contract_sha256") != baseline_contract_sha:
                    raise ValueError(f"Checkpoint sampling identity differs from its contract: {path}")
                sizes = data.get("ntoys_per_proposal_by_truth", {})
                if set(sizes) != set(TRUTHS):
                    raise ValueError(f"Missing heterogeneous bank counts: {path}")
                for truth in TRUTHS:
                    n = count(sizes[truth]); provenance = data["provenance"][truth]
                    labels = provenance["meta"]["labels"]
                    if n < 2 or count(provenance["n"]) != n * len(labels):
                        raise ValueError(f"Proposal-bank counts do not close: {path}, {truth}")
                    if any(count(r.get("ntoys_per_proposal")) != n for r in data["results"] if r["truth"] == truth):
                        raise ValueError(f"Truth-specific result count mismatch: {path}, {truth}")
            if data.get("status") not in ("completed_point", "pilot") or data["confidence_level"] != .9 or data["cls_target"] != .1:
                raise ValueError(f"Unexpected checkpoint scientific status: {path}")
            valid_path = path.parent / "validation_summary.csv"
            if not valid_path.exists():
                pending.append(dict(path=str(path), reason="validation_summary.csv is missing"))
                continue
            validation = read_csv(valid_path)
            expected_cells = {(t, strength, method) for t in TRUTHS for strength in (0, 2, 5) for method in METHODS}
            cells = {(r["truth"], count(r["strength"]), r["method"]) for r in validation}
            if len(validation) != 12 or cells != expected_cells:
                raise ValueError(f"Validation cell set is incomplete or duplicated: {valid_path}")
            for row in validation:
                if (row["scope_key"], count(row["mass_MeV"])) != key or count(row["n"]) != data["nvalidation"]:
                    raise ValueError(f"Validation coordinate/count mismatch: {valid_path}")
                row.update(mass_MeV=key[1], strength=count(row["strength"]), checkpoint_status=data["status"], checkpoint_path=str(path))
                validate_counts(row)
            if key in chosen:
                replacements.append(dict(scope_key=key[0], mass_MeV=key[1], superseded=str(chosen[key][1]), selected=str(path)))
            chosen[key] = data, path, validation
        for marker in directory.glob("*/m*/FAILURE.txt"):
            failures.append(dict(path=str(marker), has_completed_json=(marker.parent/"result.json").exists(), error=marker.read_text()))
    sources_passed = all(record["passed"] for record in source_audit.values())
    if not sources_passed and not args.allow_source_drift:
        failed = [name for name, record in source_audit.items() if not record["passed"]]
        raise ValueError(f"Frozen source drift; collect separately with --allow-source-drift to report it: {failed}")
    observed, truth_rows, validation_rows, numerical_points, checkpoint_manifest = [], [], [], [], []
    for key, baseline in parent.items():
        row = dict(baseline)
        row.update(mass_MeV=key[1], checkpoint_completed=False, checkpoint_status="missing_checkpoint")
        for method in METHODS:
            row.update({"status_"+method: "missing_checkpoint", "status_p0_"+method: "missing_checkpoint"})
            for column in (f"eps2_{method}_calibrated", f"eps2_{method}_mc_low", f"eps2_{method}_mc_high"):
                row[column] = row[column+"_ee_raw"] = None
            for column in (f"p0_{method}_calibrated", f"p0_{method}_mc_se", f"eps2_{method}_asymptotic", f"eps2_{method}_asymptotic_ee_raw", f"p0_{method}_asymptotic"):
                row[column] = None
        row["ratio_fixed_over_profiled_calibrated"] = None
        if key in chosen:
            data, path, valid = chosen[key]
            populate_point(row, data, path, truth_rows)
            moment_audit = validation_toy_moments(data, path, valid)
            validation_rows.extend(valid)
            qa = point_qa(data, path, valid)
            qa["validation_toy_moment_audit"] = moment_audit
            numerical_points.append(qa)
            row["numerical_audit_passed"] = qa["numerical_pass"]
            checkpoint_manifest.append(dict(path=str(path), sha256=sha(path), validation_sha256=sha(path.parent/"validation_summary.csv"),
                                            validation_toys_sha256=moment_audit["sha256"]))
        observed.append(row)
    complete = len(chosen) == 456
    is_pilot = any(data["status"] == "pilot" for data, _, _ in chosen.values())
    tests = validation_tests(validation_rows, complete and not is_pilot)
    numeric_passed = bool(numerical_points) and all(point["numerical_pass"] for point in numerical_points)
    numerical_qa = dict(source_contracts_passed=sources_passed, sources=source_audit, points=numerical_points,
        all_collected_numerical_checks_passed=numeric_passed, numerical_status_counts=dict(Counter(p["status"] for p in numerical_points)),
        approximation_rejections_are_data_fit_failures=False, checkpoint_failures=failures,
        all_validation_toy_moments_passed=bool(numerical_points) and all(p["validation_toy_moment_audit"]["all_passed"] for p in numerical_points),
        scalar_ledger_missing_points=sum(not p["scalar_checks"] for p in numerical_points))
    per_scope = {scope: dict(expected=bounds[1]-bounds[0]+1, completed=sum(k[0] == scope for k in chosen)) for scope, bounds in SCOPES.items()}
    status_counts = {method: dict(Counter(row["status_"+method] for row in observed)) for method in METHODS}
    p0_counts = {method: dict(Counter(row["status_p0_"+method] for row in observed)) for method in METHODS}
    finite_suite_pass = complete and not is_pilot and sources_passed and numeric_passed and all(
        tests.get(family, {}).get("rejected_count", 1) == 0 for family in ("exclusion", "local"))
    summary = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(), expected_points=456,
        completed_points=len(chosen), missing_points=456-len(chosen), complete_grid=complete, includes_pilot=is_pilot,
        collection_status="source_drift" if not sources_passed else "pilot_snapshot" if is_pilot else "complete_grid" if complete else "partial_production",
        scopes=per_scope, scope_metrics=scope_metrics(observed), endpoint_status_counts=status_counts, p0_status_counts=p0_counts,
        collected_endpoint_all_mc_resolved=bool(chosen) and all(row["status_"+m] == "resolved" for row in observed if row["checkpoint_completed"] for m in METHODS),
        validation_families=tests, finite_validation_suite_screen_passed=finite_suite_pass,
        validation_interpretation="No detected excess rejection in the complete finite validation suite" if finite_suite_pass else "No complete validation-pass claim; inspect completeness, numerical audits and adjusted tests",
        holm_interpretation="Separate calibrated/raw families over collected cells; interim adjusted values change as checkpoints arrive; no complete-family claim while partial",
        envelope_scope="Maximum over exactly gp and stress; all-three uses two joint scenarios, not every mixed constituent truth",
        mc_interval_interpretation="Maximum of the two truth-specific approximate MC endpoints; not a simultaneous coverage band",
        bias_mc_interval_interpretation="Separate-cell mean plus/minus Student t(.975,n-1) times sample SD/sqrt(n)/fixed sigma_reference; approximate pointwise 95% MC intervals, not simultaneous bands or the per-toy spread",
        linearized_bias_interpretation="Background-only residual-projection bias and propagated sampling SD from archived proposal-design influence metadata, normalized by fixed sigma_reference; approximate structural comparators only, never used to replace measured ensemble means or calibrate a statistic",
        p0_mc_se_interpretation="Standard error of the truth attaining the largest estimated p0; not a simultaneous uncertainty for the selected maximum",
        epsilon2_interpretation="Displayed values equal raw electron-only values times the archived dimuon_factor exactly once",
        local_p0_interpretation="Conditional local empirical p0; zero bounded discovery statistic has p0=1; unresolved zero/nonfinite tail estimates are blank. An IS estimate above1 by at most3SE is displayed at1 with mc_boundary status; its unclipped estimate remains archived",
        parent_observed_path=str(PARENT), parent_observed_sha256=sha(PARENT), collector_sha256=sha(Path(__file__)),
        contracts=contracts, checkpoint_manifest=checkpoint_manifest, replacements=replacements, pending_checkpoints=pending,
        missing_coordinates=[dict(scope_key=s, mass_MeV=m) for s, m in sorted(EXPECTED-set(chosen))])
    write_csv(output/"observed_calibrated_limits.csv", observed)
    write_csv(output/"truth_specific_limits.csv", truth_rows, ("scope_key", "mass_MeV", "method", "truth", "status"))
    write_csv(output/"validation_summary.csv", validation_rows, ("scope_key", "mass_MeV", "truth", "strength", "method", "n"))
    write_json(output/"numerical_qa.json", numerical_qa)
    summary["output_sha256"] = {name: sha(output/name) for name in ("observed_calibrated_limits.csv", "truth_specific_limits.csv", "validation_summary.csv", "numerical_qa.json")}
    write_json(output/"calibration_summary.json", summary)
    print(json.dumps(dict(output=str(output), completed_points=len(chosen), expected_points=456,
                         endpoint_status_counts=status_counts, collection_status=summary["collection_status"])))


if __name__ == "__main__":
    main()
