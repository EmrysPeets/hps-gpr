#!/usr/bin/env python3
"""Independent Poisson-mixture refinement with separately gated GP acceleration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "derived"
TYPE = "independent_poisson_mixture_refinement"
VERSION = 1
POLICY = "fisher075_fresh512_retry1024_centers96_v1"
NUMERICAL_POLICY = "original_proposals_then_nuisance1e-7_replay_v1"
BASE_CANDIDATE = "eigenfeature_rtol1e-15_nuisance1e-5"
STRICT_CANDIDATE = "eigenfeature_rtol1e-15_nuisance1e-7"
METHODS, TRUTHS = ("profiled", "fixed"), ("gp", "stress")
GRIDS = {"individual_2015_full": (19, 90), "individual_2016_full": (39, 180),
         "individual_2021_10pct": (50, 250), "all_2015_2016_2021": (50, 90)}
ALIASES = dict(zip(("2015", "2016", "2021", "all"), GRIDS))
EXPECTED = {(scope, mass) for scope, (lo, hi) in GRIDS.items() for mass in range(lo, hi+1)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def encoded(value):
    return json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False)+"\n"


def write_json(path, value, freeze=False):
    text = encoded(value)
    if freeze and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Frozen refinement record changed: {path}; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def scalar(value, default=math.inf):
    return float(value) if finite(value) else default


def endpoint_records(data):
    records = {(r["truth"], r["method"]): r for r in data["results"]}
    if len(data["results"]) != 4 or set(records) != {(t, m) for t in TRUTHS for m in METHODS}:
        raise RuntimeError("Incomplete or duplicate truth/method result set")
    return records


def failed_gates(record):
    if record["status"] == "right_censored" or not finite(record.get("ul_sigma")):
        return ["right_censored"]
    a = record["ul_sigma"]
    failures = []
    if record.get("monotonicity_passed") is not True:
        failures.append("monotonicity")
    if any(scalar(record.get(k), 0.) < 100 for k in ("ess_b", "ess_s")):
        failures.append("tail_ess")
    if a <= 0 or 1.96*scalar(record.get("mc_se_sigma"))/max(a, .01) > .10:
        failures.append("endpoint_mc_precision")
    width = scalar(record.get("bracket_high"))-scalar(record.get("bracket_low"), -math.inf)
    if a <= 0 or width/a > .015:
        failures.append("numerical_bracket")
    for prefix in ("", "background_"):
        mean, se = record.get(prefix+"normalization"), record.get(prefix+"normalization_se")
        if not finite(mean) or not finite(se) or abs(mean-1) > max(.05, 5*se):
            failures.append(prefix+"normalization")
    if record.get("sampling_readiness", {}).get("passed") is False:
        failures.append("sampling_readiness")
    if record["status"] != "resolved" and not failures:
        failures.append("retained_mc_qualification")
    return failures


def eligible_components(data):
    records, selected = endpoint_records(data), []
    for (truth, method), record in records.items():
        failures = failed_gates(record)
        if not failures:
            continue
        other = records["stress" if truth == "gp" else "gp", method]
        if "right_censored" in failures:
            priority, reason = 1, "right_censored"
        elif scalar(record.get("ul_sigma")) >= scalar(other.get("ul_sigma")):
            priority, reason = 2, "envelope_controlling"
        elif not finite(record.get("ul_sigma_high")) or not finite(other.get("ul_sigma_low")):
            priority, reason = 4, "overlap_unknown"
        elif record["ul_sigma_high"] >= other["ul_sigma_low"]:
            priority, reason = 3, "mc_interval_overlap"
        else:
            priority, reason = 5, "noncontrolling_qualified_component"
        selected.append(dict(truth=truth, method=method, priority=priority, reason=reason,
            failed_gates=failures, worst_ess=min(scalar(record.get("ess_b"), 0.), scalar(record.get("ess_s"), 0.))))
    return selected


def fisher_tau(truth, signal_per_sigma, a):
    truth, signal_per_sigma = np.asarray(truth), np.asarray(signal_per_sigma)
    mean = truth+a*signal_per_sigma
    if a < 0 or np.any(truth <= 0) or np.any(signal_per_sigma < 0) or np.any(~np.isfinite(mean)):
        raise ValueError("Invalid full-Poisson Fisher inputs")
    information = float(np.sum(signal_per_sigma**2/mean))
    if not math.isfinite(information) or information <= 0:
        raise ValueError("Nonpositive full-Poisson Fisher information")
    return 1/math.sqrt(information)


def fisher_mesh(truth, g, lower, upper):
    nodes = [float(lower)]
    while nodes[-1] < upper:
        next_node = min(float(upper), nodes[-1]+.75*fisher_tau(truth, g, nodes[-1]))
        if next_node <= nodes[-1]:
            raise ValueError("Fisher mesh failed to advance")
        nodes.append(next_node)
        if len(nodes) > 10000:
            raise ValueError("Fisher window requires more than 10000 centers")
    return nodes


def merge_windows(windows):
    merged = []
    for lo, hi in sorted(windows):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return merged


class PlanDeferred(RuntimeError):
    """An explicit geometry/resource cap; no inferred statistical success."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def memory_estimate(full_bins, window_bins, rank, truth_plans):
    """Conservative array budget including concurrent banks and transient copies."""
    persistent, transient, proposals, detail = 0, 0, 0, {}
    for truth, spec in truth_plans.items():
        n, p = spec["calibration_spectra"], spec["proposal_count"]
        cache = max(64, len(spec["scan_nodes"])+20)
        keep = 8*n*(full_bins+4*window_bins+window_bins*rank+6*(rank+1)+cache)
        work = 8*n*(4*window_bins*rank+8*(rank+1)**2+16*window_bins+4*p)
        persistent += keep
        transient = max(transient, work)
        proposals += 3*8*p*full_bins
        detail[truth] = dict(spectra=n, persistent_bytes=keep, transient_bytes=work)
    validation_work = 8*500*(full_bins+5*window_bins*rank+8*(rank+1)**2+20*window_bins)
    total = int(64*1024**2+persistent+max(transient, validation_work)+proposals)
    return dict(estimated_peak_bytes=total, estimated_peak_gib=total/1024**3,
        full_bins=full_bins, window_bins=window_bins, conservative_rank=rank, per_truth=detail,
        interpretation="Explicit array bound with factor/padding copies, Hessian workspaces, mixture matrices and both retained banks; not measured host memory")


def make_point_plan(ctx, baseline, source, selected, attempt):
    source_records = endpoint_records(source)
    refined = {r["truth"] for r in selected}
    if attempt == 2:
        refined.update(source["sampling_refinement"]["refined_truths"])
    plans = {}
    for truth in TRUTHS:
        original = list(map(float, baseline["nodes"]))
        if truth not in refined:
            plans[truth] = dict(refined=False, proposal_nodes=original, scan_nodes=original,
                candidates=[], windows=[], ntoys_per_proposal=256, proposal_count=3*len(original),
                calibration_spectra=256*3*len(original), ceiling=max(original))
            continue
        t, g = ctx.truths[truth], ctx.sigma*ctx.signal
        windows, candidates, scan = [], [], set(original)
        previous_nodes = source.get("nodes_by_truth", {}).get(truth, source["nodes"])
        previous_ceiling = max(previous_nodes)
        if max(original) > 64:
            raise PlanDeferred("Original scan already exceeds the declared new-strength ceiling")
        active_methods = {r["method"] for r in selected if r["truth"] == truth}
        if attempt == 2:
            active_methods.update(METHODS)
        for method in sorted(active_methods):
            record = source_records[truth, method]
            influence = baseline["provenance"][truth]["meta"]["influence"][method]
            spread = float(influence["sd"])/ctx.sigma
            hint = max(0., (ctx.ofit[method]["Ahat"]-influence["bias"])/ctx.sigma)+2*spread
            if not math.isfinite(hint) or spread <= 0:
                raise ValueError("Invalid archived influence hint")
            if record["status"] == "right_censored" or not finite(record.get("ul_sigma")):
                requested = max(1.5*previous_ceiling, previous_ceiling+4*fisher_tau(t, g, previous_ceiling), hint+3*spread)
                if attempt == 2:
                    requested = max(requested, 2*previous_ceiling)
                hi = min(64., requested)
                lo = max(0., min(previous_ceiling, hint-2*spread))
                if hi <= lo:
                    raise PlanDeferred("No extension room below the strength ceiling")
                centers = [min(hint, hi)]
            else:
                endpoint = float(record["ul_sigma"])
                width = max(.5, 2*spread, 4*fisher_tau(t, g, hint))
                lo, hi = max(0., min(endpoint, hint)-width), min(64., max(endpoint, hint)+width)
                centers = [endpoint, hint]
                if lo >= hi or any(c > 64 for c in centers):
                    raise PlanDeferred("Candidate endpoint or hint exceeds bounded proposal range")
            windows.append([lo, hi])
            candidates.append(dict(method=method, first_or_previous_endpoint=record.get("ul_sigma"),
                influence_hint=hint, propagated_sd_sigma=spread, window=[lo, hi], centers=centers))
            scan.update([lo, hi, *centers])
        windows = merge_windows(windows)
        centers = set(original)
        for lo, hi in windows:
            centers.update(fisher_mesh(t, g, lo, hi))
        centers.update(scan)
        if len(centers) > 96:
            raise PlanDeferred(f"{truth} requires {len(centers)} proposal centers; cap96, no coarsening",
                dict(truth=truth, candidates=candidates, windows=windows,
                     proposal_nodes=sorted(centers), scan_nodes=sorted(scan), partial_truth_plans=plans))
        n = 512 if attempt == 1 else 1024
        plans[truth] = dict(refined=True, proposal_nodes=sorted(centers), scan_nodes=sorted(scan),
            candidates=candidates, windows=windows, ntoys_per_proposal=n,
            proposal_count=3*len(centers), calibration_spectra=n*3*len(centers), ceiling=max(scan))
    window = len(ctx.w)
    approximate_rank = sum(min(12, int(part["p"].blind_mask.sum())) for part in ctx.parts)
    memory = {"exact_cached_cholesky": memory_estimate(len(ctx.signal), window, window, plans),
              "eigenfeature_rtol_1e-15": memory_estimate(len(ctx.signal), window, approximate_rank, plans)}
    return dict(scope_key=ctx.scope[0], mass_MeV=ctx.mass, attempt=attempt, truths=plans,
                calibration_spectra=sum(p["calibration_spectra"] for p in plans.values()),
                validation_spectra=3000, sigma_reference=ctx.sigma,
                signal_yield_per_eps2=ctx.conversion, type=TYPE, policy=POLICY,
                numerical_policy=NUMERICAL_POLICY,
                memory_estimates=memory)


def set_backend(ctx, predictors, nuisance_cut, name):
    for part, predictor in zip(ctx.parts, predictors):
        part["predictor"] = predictor
    ctx.nuisance_cut, ctx.gp_backend = nuisance_cut, name


class ScalarBatchMismatch(RuntimeError):
    pass


class ScalarReferenceFailure(ScalarBatchMismatch):
    pass


def audit_models(core, ctx, counts, b, L, strengths, label):
    results = {}
    for method in METHODS:
        factor = L if method == "profiled" else np.zeros((len(b), 0))
        batch = core.BatchProfile(counts[None, :], b[None, :], factor[None, :, :], ctx.w)
        model = core.c.Profile(b, factor, ctx.w, "linear")
        row = dict(kind="refinement_extended_scalar", label=label, method=method,
            n_spectra=1, counts_sha256=array_sha(counts), q_checks=[], passed=False)
        ctx.scalar_checks.append(row)
        def reference_fit(fixed=None):
            try:
                return model.fit(counts, fixed)
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error), fixed_amplitude=fixed)
                raise ScalarReferenceFailure("Scalar reference fit failed") from error
        free, null = reference_fit(), reference_fit(0.)
        r = np.sign(free["A"])*np.sqrt(max(0., 2*(null["nll"]-free["nll"])))
        row.update(scalar_r=float(r), batch_r=float(batch.r[0]), r_error=float(abs(r-batch.r[0])))
        if not math.isfinite(row["r_error"]) or row["r_error"] > 2e-5:
            raise ScalarBatchMismatch("Extended scalar/batch signed-r disagreement")
        qvalues = {}
        for a in strengths:
            fixed = reference_fit(a*ctx.sigma)
            q = 0. if free["A"] > a*ctx.sigma else max(0., 2*(fixed["nll"]-(free["nll"] if free["A"] >= 0 else null["nll"])))
            qb = float(batch.q(a*ctx.sigma)[0])
            row["q_checks"].append(dict(strength_sigma=a, scalar_q=float(q), batch_q=qb, q_error=abs(q-qb)))
            if not math.isfinite(q-qb) or abs(q-qb) > 1e-4:
                raise ScalarBatchMismatch("Extended scalar/batch q disagreement")
            qvalues[a] = qb
        row["passed"] = True
        results[method] = dict(r=float(batch.r[0]), q=qvalues)
    return results


def comparison_row(label, b0, L0, b1, L1, reference, other, strengths):
    C0, C1 = L0@L0.T, L1@L1.T
    row = dict(**label, mean_error_sd=float(np.max(abs(b1-b0)/np.sqrt(np.diag(C0)))),
        cov_error=float(np.max(abs(C1-C0))/np.diag(C0).max()),
        r_error=max(abs(reference[m]["r"]-other[m]["r"]) for m in METHODS),
        q_error=max(abs(reference[m]["q"][a]-other[m]["q"][a]) for m in METHODS for a in strengths))
    row["passed"] = all(math.isfinite(row[k]) and row[k] < .001 for k in ("mean_error_sd", "cov_error", "r_error", "q_error"))
    return row


def try_stricter_candidate(core, ctx):
    """Replay the original audit draws with a stricter covariance cutoff."""
    exact = [part["exact_predictor"] for part in ctx.parts]
    set_backend(ctx, exact, 0., "exact_cached_cholesky")
    try:
        from gp_lowrank_pilot import LowRankPredictor
        candidate = [LowRankPredictor(p["p"].x_full[p["keep"]], p["p"].x_full[p["p"].blind_mask],
                                     p["kernel"], ctx.cfg, rtol=1e-15) for p in ctx.parts]
    except Exception as error:
        row = dict(kind="refinement_stricter_candidate", candidate_id=STRICT_CANDIDATE,
            check_stage="original_replay", passed=False, error_type=type(error).__name__, error=str(error))
        ctx.numerical_checks.append(row)
        ctx.gp_fallback_reason = dict(stage="stricter_candidate_construction", rejected_check=row)
        return False
    rng = core.seed("numeric-audit", ctx.scope[0], ctx.mass)
    passed = True
    for truth, mean in ctx.truths.items():
        set_backend(ctx, exact, 0., "exact_cached_cholesky")
        audit_proposals, _ = ctx.proposals(mean, [0., 2., 5.])
        for index, proposal in enumerate(audit_proposals):
            whole = rng.poisson(proposal).astype(float)
            label = dict(kind="refinement_stricter_candidate", candidate_id=STRICT_CANDIDATE,
                check_stage="original_replay", truth=truth, proposal=index,
                full_counts_sha256=array_sha(whole), strengths=[2., 5., 12.],
                audit_proposals_sha256=array_sha(audit_proposals),
                seed_namespace=["numeric-audit", ctx.scope[0], ctx.mass])
            set_backend(ctx, exact, 0., "exact_cached_cholesky")
            b0, L0 = ctx.retrain(whole)
            reference = audit_models(core, ctx, whole[ctx.mask], b0, L0, [2., 5., 12.], dict(label, backend="exact"))
            try:
                set_backend(ctx, candidate, 1e-7, "eigenfeature_rtol_1e-15")
                # The unchanged retrain method enforces the twelve-mode cap.
                b1, L1 = ctx.retrain(whole)
                other = audit_models(core, ctx, whole[ctx.mask], b1, L1, [2., 5., 12.], dict(label, backend="approximate"))
                row = comparison_row(label, b0, L0, b1, L1, reference, other, [2., 5., 12.])
            except ScalarBatchMismatch:
                raise
            except Exception as error:
                row = dict(**label, passed=False, error_type=type(error).__name__, error=str(error))
                ctx.numerical_checks.append(row)
                set_backend(ctx, exact, 0., "exact_cached_cholesky")
                ctx.gp_fallback_reason = dict(stage="stricter_candidate_original_replay", rejected_check=row)
                return False
            ctx.numerical_checks.append(row)
            passed = passed and row["passed"]
            set_backend(ctx, exact, 0., "exact_cached_cholesky")
    if passed:
        set_backend(ctx, candidate, 1e-7, "eigenfeature_rtol_1e-15")
        ctx.gp_fallback_reason = None
    else:
        ctx.gp_fallback_reason = dict(stage="stricter_candidate_original_replay", reason="Unchanged discrepancy gate failed")
    return passed


def extended_probes(spec):
    probe = {2., 5., 12., max(spec["scan_nodes"])}
    probe.update(c for row in spec["candidates"] for c in row["centers"])
    centers = sorted(set([max(spec["proposal_nodes"]),
                          *[c for row in spec["candidates"] for c in row["centers"]]]))
    return centers, sorted(a for a in probe if a > 0)


def candidate_metadata(ctx, baseline_backend, baseline_reason, strict_attempted, plan):
    candidates = []
    extended_count = sum(3*len(extended_probes(spec)[0]) for spec in plan["truths"].values())
    for candidate_id, cut, required in (
            (BASE_CANDIDATE, 1e-5, ["original_frozen_audit", "extended"]),
            (STRICT_CANDIDATE, 1e-7, ["original_replay", "extended"])):
        indices = [i for i, row in enumerate(ctx.numerical_checks) if row.get("candidate_id") == candidate_id]
        stages = {stage: [i for i in indices if ctx.numerical_checks[i].get("check_stage") == stage] for stage in required}
        counts = {required[0]: 18, "extended": extended_count}
        accepted = (len(indices) == sum(counts.values())
                    and all(len(stages[stage]) == counts[stage] for stage in required)
                    and all(ctx.numerical_checks[i].get("passed") is True for i in indices))
        candidates.append(dict(candidate_id=candidate_id, target_backend="eigenfeature_rtol_1e-15",
            nuisance_eigenvalue_cut=cut, status="accepted" if accepted else "rejected" if indices else "not_attempted",
            check_indices=indices, required_stages=required, stage_check_indices=stages,
            required_check_counts=counts))
    if ctx.gp_backend == "exact_cached_cholesky" and ctx.nuisance_cut == 0.:
        active = "exact_cached_cholesky"
    elif ctx.gp_backend == "eigenfeature_rtol_1e-15" and ctx.nuisance_cut in (1e-5, 1e-7):
        active = STRICT_CANDIDATE if ctx.nuisance_cut == 1e-7 else BASE_CANDIDATE
    else:
        raise RuntimeError("Undeclared active numerical backend or covariance cutoff")
    accepted = active == "exact_cached_cholesky" or next(c for c in candidates if c["candidate_id"] == active)["status"] == "accepted"
    if not accepted:
        raise RuntimeError("Active approximation lacks its complete passed candidate audit")
    return dict(schema_version=1, policy=NUMERICAL_POLICY, active_candidate_id=active, active_candidate_accepted=accepted,
        active_backend=ctx.gp_backend, active_nuisance_eigenvalue_cut=ctx.nuisance_cut,
        first_pass_backend=baseline_backend, first_pass_fallback_reason=baseline_reason,
        stricter_candidate_attempted=strict_attempted, candidates=candidates,
        interpretation="Rejected inactive candidates remain failures in the audit history; only the explicitly accepted active candidate is used")


def extended_checks(core, ctx, plan, proposal_arrays, plan_hash, candidate_id):
    exact = [part["exact_predictor"] for part in ctx.parts]
    current = [part["predictor"] for part in ctx.parts]
    cut, backend = ctx.nuisance_cut, ctx.gp_backend
    approximate = backend != "exact_cached_cholesky"
    for truth, spec in plan["truths"].items():
        # Cover every candidate region and the highest generated strength.
        centers, strengths = extended_probes(spec)
        rng = core.seed("sampling-refinement-audit-v1", ctx.scope[0], ctx.mass, truth, plan["attempt"], plan_hash)
        for center in centers:
            j = min(range(len(spec["proposal_nodes"])), key=lambda i: abs(spec["proposal_nodes"][i]-center))
            for shift in range(3):
                whole = rng.poisson(proposal_arrays[truth][3*j+shift]).astype(float)
                label = dict(truth=truth, center=center, proposal_shift=shift, stage="extended",
                             check_stage="extended", candidate_id=candidate_id,
                             full_counts_sha256=array_sha(whole), strengths=strengths)
                set_backend(ctx, exact, 0., "exact_cached_cholesky")
                # A genuine exact reference failure is never caught as approximation rejection.
                b0, L0 = ctx.retrain(whole)
                reference = audit_models(core, ctx, whole[ctx.mask], b0, L0, strengths, dict(label, backend="exact"))
                if not approximate:
                    continue
                try:
                    set_backend(ctx, current, cut, backend)
                    b1, L1 = ctx.retrain(whole)
                    other = audit_models(core, ctx, whole[ctx.mask], b1, L1, strengths, dict(label, backend="approximate"))
                    row = comparison_row(dict(kind="refinement_extended_approximation", **label), b0, L0, b1, L1, reference, other, strengths)
                except ScalarBatchMismatch:
                    raise
                except Exception as error:
                    row = dict(kind="refinement_extended_approximation", **label, passed=False,
                               error_type=type(error).__name__, error=str(error))
                ctx.numerical_checks.append(row)
                if not row["passed"]:
                    approximate = False
                    ctx.gp_fallback_reason = dict(stage="refinement_extended_check", rejected_check=row)
                set_backend(ctx, current if approximate else exact, cut if approximate else 0.,
                            backend if approximate else "exact_cached_cholesky")
    set_backend(ctx, current if approximate else exact, cut if approximate else 0.,
                backend if approximate else "exact_cached_cholesky")


def readiness(bank, result, spec):
    if result["status"] == "right_censored":
        return dict(passed=False, reason="right_censored", checks=[])
    center = result["ul_sigma"]
    step = max(.05*center, .025)
    strengths = sorted(set([0., result["bracket_low"], result["bracket_high"], center,
                            max(.001, center-step), center+step]))
    checks = []
    for a in strengths:
        mean, se = bank.moment(bank.weights(a))
        covered = (a in spec["proposal_nodes"] or any(lo <= a <= hi for lo, hi in spec["windows"]))
        checks.append(dict(strength=a, normalization=mean, normalization_se=se, covered=covered,
                           passed=bool(covered and np.isfinite(mean) and np.isfinite(se) and se <= .05
                                       and abs(mean-1) <= max(.05, 5*se))))
    return dict(passed=all(r["passed"] for r in checks), checks=checks,
                interpretation="Additional sampling precision diagnostic; frozen inference target unchanged")


def run_point(core, frozen, ctx, baseline, source, entry, plan, destination, contract):
    start = time.monotonic()
    plan_path = destination/"point_plan.json"
    write_json(plan_path, plan, freeze=True)
    core.enable_lowrank(ctx)
    if ctx.gp_backend != baseline["gp_backend"]:
        raise RuntimeError("First-pass backend did not reproduce before proposal reconstruction")
    baseline_reason = ctx.gp_fallback_reason
    for row in ctx.numerical_checks:
        row.update(candidate_id=BASE_CANDIDATE, check_stage="original_frozen_audit")
    arrays, metadata = {}, {}
    for truth, spec in plan["truths"].items():
        arrays[truth], metadata[truth] = ctx.proposals(ctx.truths[truth], spec["proposal_nodes"])
        if not spec["refined"] and array_sha(arrays[truth]) != baseline["provenance"][truth]["proposals_sha256"]:
            raise RuntimeError("Unrefined proposal hash differs from first pass")
    proposal_plan = dict(point_plan_sha256=sha(plan_path), proposal_backend=ctx.gp_backend,
        proposals={t: dict(meta=metadata[t], sha256=array_sha(arrays[t]), truth_sha256=array_sha(ctx.truths[t])) for t in TRUTHS})
    proposal_path = destination/"proposal_plan.json"
    write_json(proposal_path, proposal_plan, freeze=True)
    strict_attempted = baseline["gp_backend"] == "exact_cached_cholesky"
    active_id = BASE_CANDIDATE
    if strict_attempted:
        if try_stricter_candidate(core, ctx):
            active_id = STRICT_CANDIDATE
        else:
            active_id = "exact_cached_cholesky"
    extended_checks(core, ctx, plan, arrays, sha(proposal_path), active_id)
    ctx.approximation_candidate_audit = candidate_metadata(ctx, baseline["gp_backend"], baseline_reason, strict_attempted, plan)
    memory = plan["memory_estimates"][ctx.gp_backend]
    memory_check = dict(**memory, limit_gib=plan["max_memory_gib"], gp_backend=ctx.gp_backend,
                        passed=memory["estimated_peak_gib"] <= plan["max_memory_gib"])
    write_json(destination/"memory_check.json", memory_check)
    write_json(destination/"pre_generation_numerical_qa.json", dict(
        numerical_checks=ctx.numerical_checks, scalar_checks=ctx.scalar_checks,
        gp_backend=ctx.gp_backend, gp_fallback_reason=ctx.gp_fallback_reason,
        approximation_candidate_audit=ctx.approximation_candidate_audit))
    if not memory_check["passed"]:
        raise PlanDeferred(f"Estimated peak {memory['estimated_peak_gib']:.3f} GiB exceeds explicit {plan['max_memory_gib']:.3f} GiB guard", memory_check)
    banks, results, provenance = {}, [], {}
    for truth, spec in plan["truths"].items():
        n = spec["ntoys_per_proposal"]
        seed_args = (("sampling-refinement-v1", ctx.scope[0], ctx.mass, truth, plan["attempt"], array_sha(arrays[truth]))
                     if spec["refined"] else ("calibration", ctx.scope[0], ctx.mass, truth, 256))
        rng = core.seed(*seed_args)
        whole = np.concatenate([rng.poisson(mean, size=(n, len(mean))) for mean in arrays[truth]])
        whole_hash = array_sha(whole)
        if not spec["refined"] and whole_hash != baseline["provenance"][truth]["whole_sha256"]:
            raise RuntimeError("Regenerated first-pass toy bank hash mismatch")
        bank = core.Bank(ctx, ctx.truths[truth], whole, arrays[truth], np.repeat(np.arange(len(arrays[truth])), n))
        bank.nodes = spec["scan_nodes"]
        banks[truth] = bank
        provenance[truth] = dict(meta=metadata[truth], n=len(whole), ntoys_per_proposal=n,
            truth_sha256=array_sha(ctx.truths[truth]), proposals_sha256=array_sha(arrays[truth]),
            whole_sha256=whole_hash, seed_namespace=list(seed_args), regenerated_first_pass=not spec["refined"])
        for method in METHODS:
            # These functions are imported unchanged from the frozen driver.
            record = frozen.invert(ctx, bank, method)
            record.update(truth=truth, pzero=bank.pzero(method), ntoys_per_proposal=n,
                          frozen_mc_status=record["status"])
            check = readiness(bank, record, spec) if spec["refined"] else dict(passed=None, reason="unrefined_original_sampling")
            record["sampling_readiness"] = check
            if check["passed"] is False and record["status"] != "right_censored":
                record["status"] = "limited_mc"
            results.append(record)
        provenance[truth].update(max_score=max(m.max_score for m in bank.models.values()),
            fallbacks=sum(m.fallbacks for m in bank.models.values()),
            weight_checks={str(a): bank.moment(bank.weights(a)) for a in (0, 2, 5)})
    valid, details = frozen.validation(ctx, banks, 500)
    core.pd.DataFrame(details).to_csv(destination/"validation_toys.csv.gz", index=False, compression="gzip")
    core.pd.DataFrame(valid).to_csv(destination/"validation_summary.csv", index=False)
    identity = dict(type=TYPE, version=VERSION, policy=POLICY, numerical_policy=NUMERICAL_POLICY, attempt=plan["attempt"],
        baseline_contract_sha256=contract["sampling_refinement"]["baseline_contract_sha256"],
        baseline_checkpoint_path=entry["baseline_checkpoint_path"], baseline_checkpoint_sha256=entry["baseline_checkpoint_sha256"],
        source_checkpoint_path=entry["source_checkpoint_path"], source_checkpoint_sha256=entry["source_checkpoint_sha256"],
        selection_record=entry, point_plan_path=str(plan_path), point_plan_sha256=sha(plan_path),
        proposal_plan_path=str(proposal_path), proposal_plan_sha256=sha(proposal_path),
        refined_truths=[t for t in TRUTHS if plan["truths"][t]["refined"]],
        validation="Same 500 independent first-pass holdout spectra rescored; counts are not pooled")
    result = dict(scope_key=ctx.scope[0], mass_MeV=ctx.mass, confidence_level=.9, cls_target=.1,
        ntoys_per_proposal=None, ntoys_per_proposal_by_truth={t: plan["truths"][t]["ntoys_per_proposal"] for t in TRUTHS},
        nvalidation=500, sigma_reference=ctx.sigma, signal_yield_per_eps2=ctx.conversion,
        nodes=sorted(set(a for spec in plan["truths"].values() for a in spec["scan_nodes"])),
        nodes_by_truth={t: plan["truths"][t]["scan_nodes"] for t in TRUTHS},
        proposal_nodes_by_truth={t: plan["truths"][t]["proposal_nodes"] for t in TRUTHS},
        results=results, provenance=provenance, numerical_checks=ctx.numerical_checks,
        scalar_checks=ctx.scalar_checks, gp_backend=ctx.gp_backend, gp_fallback_reason=ctx.gp_fallback_reason,
        nuisance_eigenvalue_cut=ctx.nuisance_cut, prediction_ledger=ctx.ledger,
        observed={m: {k: v[k] for k in ("A90", "Ahat", "signed_r")} for m, v in ctx.ofit.items()},
        sampling_refinement=identity, memory_check=memory_check,
        approximation_candidate_audit=ctx.approximation_candidate_audit,
        status="completed_point", elapsed_seconds=time.monotonic()-start)
    write_json(destination/"result.json", result)
    return result


def load_baseline():
    contract = read_json(BASE/"contract.json")
    if contract["ntoy"] != 256 or contract["nvalid"] != 500 or len(contract["hashes"]) != 47:
        raise RuntimeError("Unexpected frozen first-pass contract")
    for path, expected in contract["hashes"].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError(f"Frozen inference/input hash mismatch: {path}")
    records = {}
    for path in sorted(BASE.glob("*/m*/result.json")):
        data = read_json(path)
        key = data["scope_key"], int(data["mass_MeV"])
        if key in records or key not in EXPECTED or data["status"] != "completed_point":
            raise RuntimeError(f"Invalid first-pass checkpoint: {path}")
        if data.get("ntoys_per_proposal") != 256 or data.get("nvalidation") != 500 or data.get("confidence_level") != .9 or data.get("cls_target") != .1:
            raise RuntimeError(f"First-pass counts or confidence level differ: {path}")
        if any(p["n"] != 256*len(p["meta"]["labels"]) for p in data["provenance"].values()):
            raise RuntimeError(f"First-pass proposal counts disagree: {path}")
        endpoint_records(data)
        records[key] = (path, data)
    if set(records) != EXPECTED:
        raise RuntimeError(f"First-pass production must finish before refinement: {len(records)}/456 complete")
    return contract, records


def check_hashes(contract):
    for group in ("hashes", "sampling_hashes"):
        for relative, expected in contract[group].items():
            if sha(ROOT/relative) != expected:
                raise RuntimeError(f"Source/design drift in {group}: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Reconstruct native contexts and freeze meshes/counts; generate no toys")
    parser.add_argument("--scope", choices=[*GRIDS, *ALIASES])
    parser.add_argument("--masses", help="Comma-separated integers or inclusive ranges, e.g.41-45,74")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--previous-input", type=Path, action="append", help="Required for attempt2; directories containing completed attempt1 coordinates")
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-minutes", type=float, default=60., help="Scheduling slice; values above60 require --batch-size1")
    parser.add_argument("--max-spectra", type=int, default=1500000)
    parser.add_argument("--max-memory-gib", type=float, default=4., help="Explicit conservative peak-array guard; no host-memory probing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not 1 <= args.batch_size <= 24 or args.batch_index < 1
            or not math.isfinite(args.max_minutes) or args.max_minutes <= 0
            or (args.max_minutes > 60 and args.batch_size != 1)
            or not 0 < args.max_spectra <= 1500000):
        parser.error("Use at most24 coordinates/1.5M spectra; a scheduling slice above60 minutes requires batch-size1")
    if not math.isfinite(args.max_memory_gib) or args.max_memory_gib <= 0:
        parser.error("Memory guard must be a positive finite number of GiB")
    if (args.attempt == 2) != bool(args.previous_input):
        parser.error("Attempt2 requires --previous-input; attempt1 must use the original baseline")
    baseline_contract, baseline = load_baseline()
    baseline_hash = sha(BASE/"contract.json")
    sources = dict(baseline)
    if args.attempt == 2:
        sources = {}
        for directory in args.previous_input:
            parent_contract = read_json(directory/"contract.json")
            info = parent_contract.get("sampling_refinement", {})
            if parent_contract["hashes"] != baseline_contract["hashes"] or info.get("type") != TYPE or info.get("version") != VERSION or info.get("baseline_contract_sha256") != baseline_hash:
                raise RuntimeError("Incompatible attempt1 sampling contract")
            check_hashes(parent_contract)
            for path in sorted(directory.glob("*/m*/result.json")):
                data = read_json(path)
                key = data["scope_key"], int(data["mass_MeV"])
                identity = data.get("sampling_refinement", {})
                if (key not in baseline or key in sources or data.get("status") != "completed_point"
                        or data.get("nvalidation") != 500 or identity.get("attempt") != 1
                        or identity.get("type") != TYPE or identity.get("version") != VERSION
                        or identity.get("baseline_contract_sha256") != baseline_hash):
                    raise RuntimeError("Attempt2 inputs must be distinct completed attempt1 coordinates")
                if identity.get("baseline_checkpoint_sha256") != sha(baseline[key][0]):
                    raise RuntimeError("Attempt1 points name a different original checkpoint")
                sources[key] = path, data
    mass_filter = None
    if args.masses:
        mass_filter = set()
        for item in args.masses.split(","):
            ends = item.split("-")
            if len(ends) == 1:
                mass_filter.add(int(item))
            elif len(ends) == 2 and int(ends[1]) >= int(ends[0]):
                mass_filter.update(range(int(ends[0]), int(ends[1])+1))
            else:
                parser.error("Invalid mass list")
    scope_filter = ALIASES.get(args.scope, args.scope)
    eligible = []
    for key, (path, data) in sources.items():
        components = eligible_components(data)
        if not components:
            continue
        base_path, _ = baseline[key]
        eligible.append(dict(scope_key=key[0], mass_MeV=key[1], components=components,
            priority=min(c["priority"] for c in components), worst_ess=min(c["worst_ess"] for c in components),
            baseline_checkpoint_path=str(base_path), baseline_checkpoint_sha256=sha(base_path),
            source_checkpoint_path=str(path), source_checkpoint_sha256=sha(path)))
    eligible.sort(key=lambda e: (e["priority"], e["worst_ess"], list(GRIDS).index(e["scope_key"]), e["mass_MeV"]))
    filtered = [e for e in eligible if (scope_filter is None or e["scope_key"] == scope_filter)
                and (mass_filter is None or e["mass_MeV"] in mass_filter)]
    start_index = (args.batch_index-1)*args.batch_size
    selected = filtered[start_index:start_index+args.batch_size]
    requested_output = args.output or Path("refined_v1")/f"attempt{args.attempt}_batch{args.batch_index:03}"
    out = (requested_output if requested_output.is_absolute() else HERE/requested_output).resolve()
    if not out.is_relative_to(HERE) or out == BASE or BASE in out.parents:
        parser.error("Use a separate refinement output tree inside the calibration study")
    selection = dict(type=TYPE, version=VERSION, policy=POLICY, numerical_policy=NUMERICAL_POLICY, baseline_contract_sha256=baseline_hash,
        attempt=args.attempt, scope=scope_filter, masses=sorted(mass_filter) if mass_filter is not None else None,
        batch_index=args.batch_index, batch_size=args.batch_size, selected=selected,
        eligible=eligible, deferred=[e for e in eligible if e not in selected],
        batch_minutes=args.max_minutes, batch_calibration_spectra=args.max_spectra,
        max_memory_gib=args.max_memory_gib,
        scheduling_interpretation="Batch boundaries defer work; they are not an overall stopping condition",
        selection_inputs="First-pass/refinement MC diagnostics and censoring only; no validation results")
    selection_path = out/"selection.json"
    write_json(selection_path, selection, freeze=True)
    extra = [Path(__file__), HERE/"REFINEMENT_PROTOCOL.md", selection_path,
             HERE/"provenance/additional_runtime_hashes.json"]
    additional = read_json(HERE/"provenance/additional_runtime_hashes.json")
    for row in additional["checks"]:
        if sha(row["path"]) != row["reference_sha256"]:
            raise RuntimeError("Companion runtime hash mismatch")
        extra.append(Path(row["path"]))
    contract = dict(version=1, ntoy=None, nvalid=500, hashes=baseline_contract["hashes"],
        sampling_hashes={str(p.relative_to(ROOT)): sha(p) for p in extra},
        sampling_refinement=dict(type=TYPE, version=VERSION, policy=POLICY, numerical_policy=NUMERICAL_POLICY, attempt=args.attempt,
            baseline_contract_sha256=baseline_hash, baseline_contract_path=str(BASE/"contract.json"),
            selection_path=str(selection_path), selection_sha256=sha(selection_path)))
    write_json(out/"contract.json", contract, freeze=True)
    # The statistical modules are imported only after completion/provenance checks.
    import calibration_core as core
    import run_calibration as frozen
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        cfg = core.c.production.load_config(core.c.production.DEFAULT_CARD)
        core.c.production.validate_card(cfg)
        core.c.production.validate_histogram_inputs(cfg)
        core.c.production.validate_input_provenance(core.c.production.DEFAULT_INPUT_PROVENANCE,
            core.c.production.DEFAULT_CARD, core.c.production.DEFAULT_STATES, cfg)
        datasets = core.c.production.make_datasets(cfg)
        states = core.c.production.state_map(core.pd.read_csv(core.c.production.DEFAULT_STATES))
        scopes = {s[0]: s for s in core.SCOPES}
        clock, generated, completed, scheduled_deferred = time.monotonic(), 0, [], []
        for index, entry in enumerate(selected):
            key = entry["scope_key"], entry["mass_MeV"]
            destination = out/key[0]/f"m{key[1]:03}"
            if (destination/"result.json").exists():
                completed.append(dict(scope_key=key[0], mass_MeV=key[1], resumed_completed=True))
                continue
            if time.monotonic()-clock >= args.max_minutes*60:
                scheduled_deferred.extend(dict(entry=e, reason="batch_time_budget") for e in selected[index:])
                break
            try:
                check_hashes(contract)
                if sha(entry["source_checkpoint_path"]) != entry["source_checkpoint_sha256"] or sha(entry["baseline_checkpoint_path"]) != entry["baseline_checkpoint_sha256"]:
                    raise RuntimeError("Selected checkpoint drift")
                ctx = core.Context(scopes[key[0]], key[1], cfg, datasets, states)
                base_data, source = baseline[key][1], sources[key][1]
                if not math.isclose(ctx.sigma, base_data["sigma_reference"], rel_tol=2e-12) or not math.isclose(ctx.conversion, base_data["signal_yield_per_eps2"], rel_tol=2e-12):
                    raise RuntimeError("Observed reference normalization changed")
                if any(array_sha(ctx.truths[t]) != base_data["provenance"][t]["truth_sha256"] for t in TRUTHS):
                    raise RuntimeError("Full generating truth hash differs from first pass")
                plan = make_point_plan(ctx, base_data, source, entry["components"], args.attempt)
                plan.update(source_checkpoint_sha256=entry["source_checkpoint_sha256"],
                            baseline_contract_sha256=baseline_hash, selection_sha256=sha(selection_path),
                            max_memory_gib=args.max_memory_gib)
                write_json(destination/"point_plan.json", plan, freeze=True)
                nbase = sum(p["n"] for p in base_data["provenance"].values())
                scan_scale = max(len(p["scan_nodes"]) for p in plan["truths"].values())/len(base_data["nodes"])
                estimate = base_data["elapsed_seconds"]*plan["calibration_spectra"]/nbase*max(1., scan_scale)
                print(encoded(dict(event="planned_before_draws", scope_key=key[0], mass_MeV=key[1],
                    calibration_spectra=plan["calibration_spectra"], estimate_seconds=estimate,
                    memory_estimates=plan["memory_estimates"], max_memory_gib=args.max_memory_gib,
                    counts={t: {k: p[k] for k in ("proposal_count", "ntoys_per_proposal")} for t, p in plan["truths"].items()})), flush=True)
                if args.plan_only:
                    completed.append(dict(scope_key=key[0], mass_MeV=key[1], status="planned_no_toys", calibration_spectra=plan["calibration_spectra"]))
                    continue
                remaining = args.max_minutes*60-(time.monotonic()-clock)
                if generated+plan["calibration_spectra"] > args.max_spectra or estimate > remaining:
                    scheduled_deferred.append(dict(entry=entry, reason="batch_count_or_estimated_time_budget", estimate_seconds=estimate))
                    continue
                result = run_point(core, frozen, ctx, base_data, source, entry, plan, destination, contract)
                generated += plan["calibration_spectra"]
                completed.append(dict(scope_key=key[0], mass_MeV=key[1], status=result["status"], elapsed_seconds=result["elapsed_seconds"]))
                check_hashes(contract)
            except PlanDeferred as error:
                scheduled_deferred.append(dict(entry=entry, reason="declared_geometry_or_memory_cap", detail=str(error), diagnostics=error.details))
                write_json(destination/"DEFERRED.json", scheduled_deferred[-1])
                continue
            except Exception:
                destination.mkdir(parents=True, exist_ok=True)
                (destination/"FAILURE.txt").write_text(traceback.format_exc())
                if "ctx" in locals() and (ctx.scope[0], ctx.mass) == key:
                    write_json(destination/"failure_numerical_qa.json", dict(
                        numerical_checks=ctx.numerical_checks, scalar_checks=ctx.scalar_checks,
                        gp_backend=getattr(ctx, "gp_backend", None), gp_fallback_reason=getattr(ctx, "gp_fallback_reason", None),
                        approximation_candidate_audit=getattr(ctx, "approximation_candidate_audit", None)))
                raise
            finally:
                write_json(out/"batch_summary.json", dict(completed=completed, scheduled_deferred=scheduled_deferred,
                    other_batches_deferred=selection["deferred"], generated_calibration_spectra=generated,
                    elapsed_seconds=time.monotonic()-clock, plan_only=args.plan_only,
                    remaining_work="Deferred and unresolved endpoints require continued batches within the user's budget floor"))
        check_hashes(contract)
        write_json(out/"batch_summary.json", dict(completed=completed, scheduled_deferred=scheduled_deferred,
            other_batches_deferred=selection["deferred"], generated_calibration_spectra=generated,
            elapsed_seconds=time.monotonic()-clock, plan_only=args.plan_only,
            scheduling_slice="Current invocation; completed checkpoints were skipped",
            remaining_work="Deferred and unresolved endpoints require continued batches within the user's budget floor"))


if __name__ == "__main__":
    main()
