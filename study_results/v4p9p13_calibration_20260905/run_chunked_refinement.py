#!/usr/bin/env python3
"""Bounded-memory execution of the frozen Poisson-mixture sampling refinement.

Imports fitting modules only inside main, after source and completion checks.
The statistical definitions and sampling policy are imported without mutation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import logsumexp
import run_sampling_refinement as ref

HERE, ROOT, BASE = ref.HERE, ref.ROOT, ref.BASE
METHODS, TRUTHS = ref.METHODS, ref.TRUTHS
EXECUTION_TYPE, EXECUTION_VERSION = "chunked_sampling_execution", 1
CHUNK_SIZE, RUNTIME_RESERVE = 128, 512*1024**2
R_GATE, Q_GATE = 2e-5, 1e-4
LOG_DENSITY_GATE, WEIGHT_RTOL, WEIGHT_ATOL = 1e-7, 2e-7, 1e-12


def array_sha(value):
    """Hash precisely the original C-order bytes, with no whole-array copy."""
    value = np.asarray(value)
    if not value.flags.c_contiguous or value.dtype.hasobject:
        raise ValueError("Hash input must be a contiguous, non-object array")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def generate_whole(rng, proposals, nper):
    """Original per-proposal Poisson calls/shapes; preallocate their concatenation."""
    if nper < 2 or proposals.ndim != 2 or not len(proposals):
        raise ValueError("Invalid proposal-bank shape")
    whole = np.empty((len(proposals)*nper, proposals.shape[1]), dtype=np.int64)
    digest, proposal_hashes = hashlib.sha256(), []
    for j, mean in enumerate(proposals):
        counts = rng.poisson(mean, size=(nper, len(mean)))
        if counts.dtype != whole.dtype or not counts.flags.c_contiguous:
            raise RuntimeError("Original Poisson output dtype/layout changed")
        block = memoryview(counts).cast("B")
        digest.update(block)
        proposal_hashes.append(hashlib.sha256(block).hexdigest())
        whole[j*nper:(j+1)*nper] = counts
        del block, counts
    actual = array_sha(whole)
    if actual != digest.hexdigest():
        raise RuntimeError("Preallocated whole-array concatenation hash failed")
    return whole, dict(passed=True, whole_sha256=actual,
        concatenated_proposal_bytes_sha256=digest.hexdigest(),
        proposal_draw_sha256=proposal_hashes, dtype=str(whole.dtype), shape=list(whole.shape),
        rng_call_shape=[nper, proposals.shape[1]], rng_calls=len(proposals),
        recipe="Original per-proposal RNG calls, same order and shape; C-order concatenation")


def blocked_logmix(whole, truth, proposals, chunk_size=CHUNK_SIZE):
    """The unchanged full-Poisson log mixture, evaluated in row blocks."""
    if chunk_size < 1:
        raise ValueError("Nonpositive density chunk size")
    logratio = np.log(proposals/truth).T
    offsets = np.sum(proposals-truth, axis=1)
    out = np.empty(len(whole), dtype=float)
    for start in range(0, len(whole), chunk_size):
        stop = min(start+chunk_size, len(whole))
        out[start:stop] = logsumexp(whole[start:stop]@logratio-offsets, axis=1)-np.log(len(proposals))
    if not np.isfinite(out).all():
        raise RuntimeError("Nonfinite Poisson mixture log density")
    return out


def blocked_weights(whole, truth, signal, sigma, a, logmix, chunk_size=CHUNK_SIZE):
    """Identical Poisson target/mixture ratio; avoid full count-matrix casts."""
    delta = a*sigma*signal
    logratio, offset = np.log1p(delta/truth), np.sum(delta)
    out = np.empty(len(whole), dtype=float)
    for start in range(0, len(whole), chunk_size):
        stop = min(start+chunk_size, len(whole))
        logtarget = whole[start:stop]@logratio-offset
        out[start:stop] = np.exp(logtarget-logmix[start:stop])
    if not np.isfinite(out).all():
        raise RuntimeError("Nonfinite Poisson importance weight")
    return out


def weight_comparison(reference, blocked):
    """Auditable numeric form of the declared absolute-plus-relative gate."""
    reference, blocked = np.asarray(reference), np.asarray(blocked)
    if reference.shape != blocked.shape or not reference.size:
        raise ValueError("Weight-comparison shapes differ or are empty")
    finite = bool(np.isfinite(reference).all() and np.isfinite(blocked).all())
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        difference = abs(reference-blocked)
        error = float(np.max(difference))
        relative = float(np.max(difference/np.maximum(abs(reference), WEIGHT_ATOL)))
        scaled = float(np.max(difference/(WEIGHT_ATOL+WEIGHT_RTOL*abs(reference))))
    return dict(max_abs_error=error, max_relative_error=relative, max_scaled_error=scaled,
                finite=finite, passed=bool(finite and math.isfinite(scaled) and scaled <= 1.))


class RowLookup:
    """Integer row access without concatenating retained factor arrays."""
    def __init__(self, chunks, offsets, attribute):
        self.chunks, self.offsets, self.attribute = chunks, offsets, attribute

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Chunk row lookup supports integer indexing only")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        j = int(np.searchsorted(self.offsets, index, side="right")-1)
        return getattr(self.chunks[j], self.attribute)[index-self.offsets[j]]


class AggregateModel:
    """Compatible fitted-model view; each q fit uses the original BatchProfile."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.offsets = np.r_[0, np.cumsum([len(m.r) for m in chunks])]
        self.nt = int(self.offsets[-1])
        self.r = np.concatenate([m.r for m in chunks])
        self.free = {"A": np.concatenate([m.free["A"] for m in chunks])}
        self.b = RowLookup(chunks, self.offsets, "b")
        self.L = RowLookup(chunks, self.offsets, "L")

    @property
    def max_score(self):
        return max(m.max_score for m in self.chunks)

    @property
    def fallbacks(self):
        return sum(m.fallbacks for m in self.chunks)

    def q(self, A):
        out = np.empty(self.nt, dtype=float)
        for j, model in enumerate(self.chunks):
            try:
                out[self.offsets[j]:self.offsets[j+1]] = model.q(A)
            except Exception as error:
                raise RuntimeError(f"Chunk q fit failed for rows {self.offsets[j]}:{self.offsets[j+1]}, A={A}") from error
        return out


def runtime_types(core):
    """Ordinary subclasses; no monkey-patching or replacement of imported globals."""
    class ChunkedContext(core.Context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chunked_equivalence_checks = dict(schema_version=1, passed=False)
            self.model_chunk_ledger = []
            self.execution_call_kind = "unspecified"

        def make_models(self, whole, *, audit_chunk_size=None):
            size = CHUNK_SIZE if audit_chunk_size is None else audit_chunk_size
            if not 1 <= size <= CHUNK_SIZE or len(whole) < 1:
                raise ValueError("Invalid model chunk size or empty bank")
            chunks = {method: [] for method in METHODS}
            call = dict(call_id=len(self.model_chunk_ledger), purpose=self.execution_call_kind,
                n_spectra=len(whole), chunk_size=size, chunks=[], passed=False)
            self.model_chunk_ledger.append(call)
            for start in range(0, len(whole), size):
                stop = min(start+size, len(whole))
                before = len(self.scalar_checks)
                chunk_record = dict(start=start, stop=stop, passed=False)
                call["chunks"].append(chunk_record)
                try:
                    models = core.Context.make_models(self, whole[start:stop])
                except Exception as error:
                    chunk_record.update(error_type=type(error).__name__, error=str(error),
                        full_counts_sha256=array_sha(whole[start:stop]),
                        scalar_check_indices=list(range(before, len(self.scalar_checks))))
                    raise
                for row in self.scalar_checks[before:]:
                    row.update(execution_call_id=call["call_id"], chunk_start=start,
                               global_toy_index=start+row["toy_index"])
                chunk_record.update(passed=True,
                    npar_profiled=models["profiled"].npar,
                    profiled_blocks=models["profiled"].blocks,
                    scalar_check_indices=list(range(before, len(self.scalar_checks))))
                for method in METHODS:
                    chunks[method].append(models[method])
            result = {method: AggregateModel(parts) for method, parts in chunks.items()}
            call["passed"] = True
            return result

    class ChunkedBank(core.Bank):
        def __init__(self, ctx, truth, whole, proposals, strata, *, qcache_limit):
            self.ctx, self.truth, self.whole, self.strata = ctx, truth, whole, strata
            self.n, self.K = len(whole), len(proposals)
            if self.n % self.K or not np.array_equal(strata, np.repeat(np.arange(self.K), self.n//self.K)):
                raise RuntimeError("Proposal stratification/order changed")
            self.logmix = blocked_logmix(whole, truth, proposals)
            self.models = ctx.make_models(whole)
            self.qcache, self.qcache_limit = {}, int(qcache_limit)
            self.qcache_peak = 0

        def weights(self, a):
            return blocked_weights(self.whole, self.truth, self.ctx.signal,
                                   self.ctx.sigma, a, self.logmix)

        def q(self, method, a):
            key = method, float(a)
            if key not in self.qcache:
                if len(self.qcache) >= self.qcache_limit:
                    raise RuntimeError(f"Declared q-cache cap {self.qcache_limit} exceeded at {key}; no eviction or result certification")
                self.qcache[key] = self.models[method].q(a*self.ctx.sigma)
                self.qcache_peak = max(self.qcache_peak, len(self.qcache))
            return self.qcache[key]

    return ChunkedContext, ChunkedBank


def memory_estimate(full_bins, window_bins, rank, truth_plans):
    """Explicit conservative array estimate; no measurement of host memory."""
    persistent, proposals, transient, detail = 0, 0, 0, {}
    for truth, spec in truth_plans.items():
        n, p = spec["calibration_spectra"], spec["proposal_count"]
        # Two methods: positive grid, at most14 bisections, center/slope3,
        # and validation strengths2/5. Retain every cached result; never evict.
        cache = max(128, 2*(len(spec["scan_nodes"])+18))
        keep = 8*n*(full_bins+4*window_bins+window_bins*rank+6*(rank+1)+cache)
        work = 8*CHUNK_SIZE*(4*window_bins*rank+8*(rank+1)**2+16*window_bins+4*p+full_bins)
        persistent += keep
        transient = max(transient, work)
        proposals += 3*8*p*full_bins
        detail[truth] = dict(spectra=n, persistent_bytes=keep, chunk_workspace_bytes=work,
            qcache_limit=cache, qcache_bound_bytes=8*n*cache)
    # Python evaluates the next make_models call before replacing the previous
    # validation models, so budget two retained500-row sets at that boundary.
    validation_keep = 2*8*500*(full_bins+4*window_bins+window_bins*rank+6*(rank+1)+4)
    # Original RNG call shape is retained, so one nper-by-fullbins temporary
    # remains. All science factors are not built yet during generation; adding
    # it to the full retained-bank bound is deliberately conservative.
    generation_work = max(8*s["ntoys_per_proposal"]*full_bins for s in truth_plans.values())
    total = int(RUNTIME_RESERVE+persistent+proposals+transient+validation_keep+generation_work)
    return dict(estimated_peak_bytes=total, estimated_peak_gib=total/1024**3,
        full_bins=full_bins, window_bins=window_bins, conservative_rank=rank,
        chunk_size=CHUNK_SIZE, runtime_reserve_bytes=RUNTIME_RESERVE,
        persistent_bytes=persistent, proposal_bytes=proposals, chunk_workspace_bytes=transient,
        validation_retained_bytes=validation_keep, generation_workspace_bytes=generation_work,
        per_truth=detail,
        interpretation="Both banks, bounded q caches, chunk fit/density copies, retained validation models and 512 MiB runtime reserve; source-derived, not measured RSS")


def layout_marker(max_memory_gib):
    paths = [Path(__file__).resolve(), HERE/"CHUNKED_REFINEMENT_PROTOCOL.md"]
    return dict(type=EXECUTION_TYPE, version=EXECUTION_VERSION, chunk_size=CHUNK_SIZE,
        runtime_reserve_bytes=RUNTIME_RESERVE, max_memory_gib=max_memory_gib,
        source_hashes={str(p.relative_to(ROOT)): ref.sha(p) for p in paths},
        sampling_policy_unchanged=True, sampling_policy=ref.POLICY,
        numerical_policy=ref.NUMERICAL_POLICY, blas_threads=1,
        statistical_identity="Same likelihood and Poisson laws; floating-point/fallback paths need not be bit-identical")


def replay_audit_spectra(core, ctx, plan, arrays, proposal_plan_sha):
    """Regenerate exactly the prescribed numerical-audit draws, never calibration draws."""
    active = [p["predictor"] for p in ctx.parts]
    cut, backend = ctx.nuisance_cut, ctx.gp_backend
    exact = [p["exact_predictor"] for p in ctx.parts]
    rows, counts = [], []
    try:
        ref.set_backend(ctx, exact, 0., "exact_cached_cholesky")
        rng = core.seed("numeric-audit", ctx.scope[0], ctx.mass)
        for truth, mean in ctx.truths.items():
            proposals, _ = ctx.proposals(mean, [0., 2., 5.])
            for index, proposal in enumerate(proposals):
                whole = rng.poisson(proposal).astype(float)
                rows.append(dict(stage="original", truth=truth, proposal=index,
                    strengths=[2., 5., 12.], full_counts_sha256=array_sha(whole),
                    audit_proposals_sha256=array_sha(proposals),
                    seed_namespace=["numeric-audit", ctx.scope[0], ctx.mass]))
                counts.append(whole)
    finally:
        ref.set_backend(ctx, active, cut, backend)
    for truth, spec in plan["truths"].items():
        centers, strengths = ref.extended_probes(spec)
        seed_args = ["sampling-refinement-audit-v1", ctx.scope[0], ctx.mass,
                     truth, plan["attempt"], proposal_plan_sha]
        rng = core.seed(*seed_args)
        for center in centers:
            j = min(range(len(spec["proposal_nodes"])), key=lambda i: abs(spec["proposal_nodes"][i]-center))
            for shift in range(3):
                whole = rng.poisson(arrays[truth][3*j+shift]).astype(float)
                rows.append(dict(stage="extended", truth=truth, center=center,
                    proposal_shift=shift, strengths=strengths,
                    full_counts_sha256=array_sha(whole), seed_namespace=seed_args))
                counts.append(whole)
    return np.array(counts), rows


def equivalence_audit(core, ctx, plan, arrays, proposal_plan_sha):
    """Every original18 and extended audit spectrum: split, unsplit and scalar."""
    audit = dict(schema_version=1, passed=False, split_chunk_size=1,
        production_chunk_size=CHUNK_SIZE, r_tolerance=R_GATE, q_tolerance=Q_GATE,
        log_density_atol=LOG_DENSITY_GATE, weight_rtol=WEIGHT_RTOL,
        weight_atol=WEIGHT_ATOL, statistic_checks=[], density_checks=[], generation_checks={})
    ctx.chunked_equivalence_checks = audit
    whole, coordinates = replay_audit_spectra(core, ctx, plan, arrays, proposal_plan_sha)
    # The extended reference audit records every prescribed full-count hash,
    # including when the active backend is exact. Prove that this is a replay.
    archived_extended = {}
    for check in ctx.scalar_checks:
        label = check.get("label", {})
        if label.get("check_stage") == "extended":
            key = label["truth"], label["center"], label["proposal_shift"]
            archived_extended.setdefault(key, set()).add(label["full_counts_sha256"])
    for row in coordinates:
        if row["stage"] == "extended":
            key = row["truth"], row["center"], row["proposal_shift"]
            if archived_extended.get(key) != {row["full_counts_sha256"]}:
                raise RuntimeError("Extended execution audit does not replay its reference counts")
    audit.update(spectra=coordinates, n_original=18, n_extended=len(coordinates)-18,
                 audit_whole_sha256=array_sha(whole), unsplit_n_spectra=len(whole),
                 extended_reference_counts_replayed=True)
    if len(whole) > CHUNK_SIZE:
        raise RuntimeError("Audit reference exceeds the declared one-chunk workspace")
    # These are the same final accepted GP backend and covariance conventions.
    # A one-row split maximizes the change in active-column/global-Newton grouping.
    ctx.execution_call_kind = "execution_equivalence"
    unsplit = core.Context.make_models(ctx, whole)
    split = ctx.make_models(whole, audit_chunk_size=1)
    strengths = sorted({a for row in coordinates for a in row["strengths"]})
    for method in METHODS:
        uq = {a: unsplit[method].q(a*ctx.sigma) for a in strengths}
        sq = {a: split[method].q(a*ctx.sigma) for a in strengths}
        for i, coordinate in enumerate(coordinates):
            row = dict(**coordinate, method=method, audit_index=i, passed=False, q_checks=[])
            audit["statistic_checks"].append(row)
            try:
                model = core.c.Profile(unsplit[method].b[i], unsplit[method].L[i], ctx.w, "linear")
                free, null = model.fit(whole[i, ctx.mask]), model.fit(whole[i, ctx.mask], 0.)
                sr = float(np.sign(free["A"])*np.sqrt(max(0., 2*(null["nll"]-free["nll"]))))
                ur, cr = float(unsplit[method].r[i]), float(split[method].r[i])
                r_error = max(abs(sr-ur), abs(sr-cr), abs(ur-cr))
                row.update(scalar_r=sr, unsplit_r=ur, split_r=cr, r_error=r_error,
                    unsplit_npar=unsplit[method].L[i].shape[1], split_npar=split[method].L[i].shape[1])
                if not math.isfinite(r_error) or r_error > R_GATE:
                    raise RuntimeError("Chunked/unsplit/scalar signed-r disagreement")
                for a in coordinate["strengths"]:
                    f = model.fit(whole[i, ctx.mask], a*ctx.sigma)
                    scalar_q = 0. if free["A"] > a*ctx.sigma else max(0., 2*(f["nll"]-(free["nll"] if free["A"] >= 0 else null["nll"])))
                    qr, qc = float(uq[a][i]), float(sq[a][i])
                    error = max(abs(scalar_q-qr), abs(scalar_q-qc), abs(qr-qc))
                    row["q_checks"].append(dict(strength_sigma=a, scalar_q=scalar_q,
                        unsplit_q=qr, split_q=qc, q_error=error))
                    if not math.isfinite(error) or error > Q_GATE:
                        raise RuntimeError("Chunked/unsplit/scalar bounded-q disagreement")
                row["passed"] = True
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error))
                raise
    # Audit densities in the science bank's integer dtype, against the exact
    # frozen expression. A one-row split exercises a different BLAS grouping.
    integers = whole.astype(np.int64)
    for truth, means in arrays.items():
        t = ctx.truths[truth]
        indices = [i for i, coordinate in enumerate(coordinates) if coordinate["truth"] == truth]
        sample = integers[indices]
        own_strengths = {a for i in indices for a in coordinates[i]["strengths"]}
        full = logsumexp(sample@np.log(means/t).T-np.sum(means-t, axis=1), axis=1)-np.log(len(means))
        blocked = blocked_logmix(sample, t, means, 1)
        density_error = float(np.max(abs(full-blocked)))
        row = dict(truth=truth, passed=False, n_spectra=len(sample), audit_indices=indices,
            full_counts_sha256=array_sha(sample), proposals_sha256=array_sha(means),
            logmix_max_abs_error=density_error, weight_checks=[])
        audit["density_checks"].append(row)
        if not math.isfinite(density_error) or density_error > LOG_DENSITY_GATE:
            raise RuntimeError("Blocked/full Poisson mixture density disagreement")
        for a in sorted(set([0., *own_strengths, *plan["truths"][truth]["scan_nodes"]])):
            delta = a*ctx.sigma*ctx.signal
            old = np.exp(sample@np.log1p(delta/t)-np.sum(delta)-full)
            new = blocked_weights(sample, t, ctx.signal, ctx.sigma, a, blocked, 1)
            comparison = weight_comparison(old, new)
            row["weight_checks"].append(dict(strength_sigma=a, **comparison))
            if not comparison["passed"]:
                raise RuntimeError("Blocked/full Poisson importance-weight disagreement")
        row["passed"] = True
    audit["statistic_and_density_passed"] = True
    # passed is set only after both science-bank generation/hash checks pass.
    del split, unsplit, whole, integers
    gc.collect()
    return audit


def run_point(core, frozen, Bank, ctx, baseline, entry, plan, destination, contract):
    """Plain derivative of ref.run_point; only allocation/layout operations differ."""
    start = time.monotonic()
    plan_path = destination/"point_plan.json"
    ref.write_json(plan_path, plan, freeze=True)
    core.enable_lowrank(ctx)
    if ctx.gp_backend != baseline["gp_backend"]:
        raise RuntimeError("First-pass backend did not reproduce before proposal reconstruction")
    baseline_reason = ctx.gp_fallback_reason
    for row in ctx.numerical_checks:
        row.update(candidate_id=ref.BASE_CANDIDATE, check_stage="original_frozen_audit")
    arrays, metadata = {}, {}
    for truth, spec in plan["truths"].items():
        arrays[truth], metadata[truth] = ctx.proposals(ctx.truths[truth], spec["proposal_nodes"])
        if not spec["refined"] and array_sha(arrays[truth]) != baseline["provenance"][truth]["proposals_sha256"]:
            raise RuntimeError("Unrefined proposal hash differs from first pass")
    proposal_path = destination/"proposal_plan.json"
    ref.write_json(proposal_path, dict(point_plan_sha256=ref.sha(plan_path), proposal_backend=ctx.gp_backend,
        proposals={t: dict(meta=metadata[t], sha256=array_sha(arrays[t]), truth_sha256=array_sha(ctx.truths[t]))
                   for t in TRUTHS}), freeze=True)
    strict_attempted = baseline["gp_backend"] == "exact_cached_cholesky"
    active_id = ref.BASE_CANDIDATE
    if strict_attempted:
        active_id = ref.STRICT_CANDIDATE if ref.try_stricter_candidate(core, ctx) else "exact_cached_cholesky"
    ref.extended_checks(core, ctx, plan, arrays, ref.sha(proposal_path), active_id)
    ctx.approximation_candidate_audit = ref.candidate_metadata(ctx, baseline["gp_backend"],
                                                              baseline_reason, strict_attempted, plan)
    memory = plan["memory_estimates"][ctx.gp_backend]
    memory_check = dict(**memory, limit_gib=plan["max_memory_gib"], gp_backend=ctx.gp_backend,
        passed=memory["estimated_peak_gib"] <= plan["max_memory_gib"])
    ref.write_json(destination/"memory_check.json", memory_check)
    ref.write_json(destination/"pre_generation_numerical_qa.json", dict(
        numerical_checks=ctx.numerical_checks, scalar_checks=ctx.scalar_checks,
        gp_backend=ctx.gp_backend, gp_fallback_reason=ctx.gp_fallback_reason,
        approximation_candidate_audit=ctx.approximation_candidate_audit))
    if not memory_check["passed"]:
        raise ref.PlanDeferred(f"Chunked peak {memory['estimated_peak_gib']:.3f} GiB exceeds explicit "
                               f"{plan['max_memory_gib']:.3f} GiB guard", memory_check)
    equivalence_audit(core, ctx, plan, arrays, ref.sha(proposal_path))
    ref.write_json(destination/"chunked_equivalence_checks.json", ctx.chunked_equivalence_checks)
    banks, results, provenance = {}, [], {}
    for truth, spec in plan["truths"].items():
        n = spec["ntoys_per_proposal"]
        seed_args = (("sampling-refinement-v1", ctx.scope[0], ctx.mass, truth, plan["attempt"], array_sha(arrays[truth]))
                     if spec["refined"] else ("calibration", ctx.scope[0], ctx.mass, truth, 256))
        whole, closure = generate_whole(core.seed(*seed_args), arrays[truth], n)
        closure.update(seed_namespace=list(seed_args), regenerated_first_pass=not spec["refined"],
            baseline_whole_sha256=baseline["provenance"][truth]["whole_sha256"] if not spec["refined"] else None)
        ctx.chunked_equivalence_checks["generation_checks"][truth] = closure
        if not spec["refined"] and closure["whole_sha256"] != closure["baseline_whole_sha256"]:
            closure.update(passed=False, error="Regenerated first-pass whole-array hash differs")
            raise RuntimeError(closure["error"])
        ctx.execution_call_kind = "calibration_"+truth
        bank = Bank(ctx, ctx.truths[truth], whole, arrays[truth], np.repeat(np.arange(len(arrays[truth])), n),
                    qcache_limit=memory["per_truth"][truth]["qcache_limit"])
        bank.nodes = spec["scan_nodes"]
        banks[truth] = bank
        provenance[truth] = dict(meta=metadata[truth], n=len(whole), ntoys_per_proposal=n,
            truth_sha256=array_sha(ctx.truths[truth]), proposals_sha256=array_sha(arrays[truth]),
            whole_sha256=closure["whole_sha256"], seed_namespace=list(seed_args),
            regenerated_first_pass=not spec["refined"])
        for method in METHODS:
            record = frozen.invert(ctx, bank, method)
            record.update(truth=truth, pzero=bank.pzero(method), ntoys_per_proposal=n,
                          frozen_mc_status=record["status"])
            check = ref.readiness(bank, record, spec) if spec["refined"] else dict(passed=None, reason="unrefined_original_sampling")
            record["sampling_readiness"] = check
            if check["passed"] is False and record["status"] != "right_censored":
                record["status"] = "limited_mc"
            results.append(record)
        provenance[truth].update(max_score=max(m.max_score for m in bank.models.values()),
            fallbacks=sum(m.fallbacks for m in bank.models.values()),
            weight_checks={str(a): bank.moment(bank.weights(a)) for a in (0, 2, 5)})
    ctx.execution_call_kind = "validation"
    valid, details = frozen.validation(ctx, banks, 500)
    core.pd.DataFrame(details).to_csv(destination/"validation_toys.csv.gz", index=False, compression="gzip")
    core.pd.DataFrame(valid).to_csv(destination/"validation_summary.csv", index=False)
    audit = ctx.chunked_equivalence_checks
    if (audit.get("statistic_and_density_passed") is not True or set(audit["generation_checks"]) != set(TRUTHS)
            or not all(r["passed"] is True for r in audit["generation_checks"].values())):
        raise RuntimeError("Incomplete chunked execution equivalence/array closure")
    audit["passed"] = True
    ref.write_json(destination/"chunked_equivalence_checks.json", audit)
    ref.write_json(destination/"model_chunk_ledger.json", ctx.model_chunk_ledger)
    identity = dict(type=ref.TYPE, version=ref.VERSION, policy=ref.POLICY,
        numerical_policy=ref.NUMERICAL_POLICY, attempt=plan["attempt"],
        baseline_contract_sha256=contract["sampling_refinement"]["baseline_contract_sha256"],
        baseline_checkpoint_path=entry["baseline_checkpoint_path"], baseline_checkpoint_sha256=entry["baseline_checkpoint_sha256"],
        source_checkpoint_path=entry["source_checkpoint_path"], source_checkpoint_sha256=entry["source_checkpoint_sha256"],
        selection_record=entry, point_plan_path=str(plan_path), point_plan_sha256=ref.sha(plan_path),
        proposal_plan_path=str(proposal_path), proposal_plan_sha256=ref.sha(proposal_path),
        refined_truths=[t for t in TRUTHS if plan["truths"][t]["refined"]],
        validation="Same 500 independent first-pass holdout spectra rescored; counts are not pooled")
    cache_ledger = {t: dict(limit=b.qcache_limit, entries=len(b.qcache), peak_entries=b.qcache_peak,
        retained_bytes=sum(v.nbytes for v in b.qcache.values()),
        keys=[dict(method=m, strength_sigma=a) for m, a in b.qcache], passed=len(b.qcache) <= b.qcache_limit)
        for t, b in banks.items()}
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
        execution_layout=contract["execution_layout"], chunked_equivalence_checks=audit,
        model_chunk_ledger_sha256=ref.sha(destination/"model_chunk_ledger.json"), qcache_ledger=cache_ledger,
        status="completed_point", elapsed_seconds=time.monotonic()-start)
    ref.write_json(destination/"result.json", result)
    print(ref.encoded(dict(event="point_completed", scope_key=ctx.scope[0], mass_MeV=ctx.mass,
                          elapsed_seconds=result["elapsed_seconds"], chunked_equivalence_passed=True)), flush=True)
    return result


def sampling_input(directory, baseline_contract, baseline, required_attempt=None, *, frozen_identity=None):
    """Read only checkpoint/completion metadata; never read validation outcomes."""
    directory = directory.resolve()
    contract_path = directory/"contract.json"
    if frozen_identity is not None:
        if (frozen_identity["directory"] != str(directory)
                or frozen_identity["contract_path"] != str(contract_path)
                or ref.sha(contract_path) != frozen_identity["contract_sha256"]):
            raise RuntimeError("Frozen sampling input contract changed")
        paths = [Path(record["path"]) for record in frozen_identity["checkpoints"]]
        if len(set(paths)) != len(paths):
            raise RuntimeError("Duplicate checkpoint in frozen input snapshot")
        for path, record in zip(paths, frozen_identity["checkpoints"]):
            if not path.is_relative_to(directory) or ref.sha(path) != record["sha256"]:
                raise RuntimeError("Frozen sampling input checkpoint changed")
    else:
        paths = sorted(directory.glob("*/m*/result.json"))
    contract = ref.read_json(contract_path)
    marker = contract.get("sampling_refinement", {})
    if (contract.get("hashes") != baseline_contract["hashes"] or marker.get("type") != ref.TYPE
            or marker.get("version") != ref.VERSION or marker.get("baseline_contract_sha256") != ref.sha(BASE/"contract.json")):
        raise RuntimeError("Incompatible prior sampling input")
    ref.check_hashes(contract)
    records, identities = {}, []
    for path in paths:
        data = ref.read_json(path)
        key = data["scope_key"], int(data["mass_MeV"])
        info = data.get("sampling_refinement", {})
        if (key not in baseline or key in records or data.get("status") != "completed_point"
                or data.get("nvalidation") != 500 or data.get("confidence_level") != .9 or data.get("cls_target") != .1
                or info.get("type") != ref.TYPE or info.get("version") != ref.VERSION
                or info.get("baseline_contract_sha256") != marker["baseline_contract_sha256"]
                or info.get("baseline_checkpoint_sha256") != ref.sha(baseline[key][0])
                or (required_attempt is not None and info.get("attempt") != required_attempt)
                or (path.parent/"FAILURE.txt").exists()):
            raise RuntimeError(f"Invalid completed sampling checkpoint: {path}")
        ref.endpoint_records(data)
        sizes = data.get("ntoys_per_proposal_by_truth", {})
        if (set(sizes) != set(TRUTHS) or data.get("ntoys_per_proposal") is not None
                or any(not isinstance(sizes[t], int) or sizes[t] < 2
                       or data["provenance"][t]["n"] != sizes[t]*len(data["provenance"][t]["meta"]["labels"])
                       for t in TRUTHS)
                or any(r.get("ntoys_per_proposal") != sizes.get(r["truth"]) for r in data["results"])):
            raise RuntimeError("Completed sampling counts do not close")
        records[key] = path, data
        identities.append(dict(scope_key=key[0], mass_MeV=key[1], path=str(path), sha256=ref.sha(path),
                               attempt=info["attempt"]))
    identity = dict(directory=str(directory), contract_path=str(contract_path),
        contract_sha256=ref.sha(contract_path), checkpoints=identities,
        selection_inputs="Completed checkpoint identity/counts only; no validation outcomes")
    if frozen_identity is not None and identity != frozen_identity:
        raise RuntimeError("Frozen sampling input metadata changed")
    return records, identity


def saved_selection(path, configuration, layout, baseline_hash):
    """Resume the saved selection exactly; new input-directory results are ignored."""
    if not path.exists():
        return None
    selection = ref.read_json(path)
    if (selection.get("cli_configuration") != configuration
            or selection.get("execution_layout") != layout
            or selection.get("baseline_contract_sha256") != baseline_hash
            or selection.get("type") != ref.TYPE or selection.get("version") != ref.VERSION
            or selection.get("policy") != ref.POLICY or selection.get("numerical_policy") != ref.NUMERICAL_POLICY):
        raise RuntimeError("Frozen selection configuration/source identity differs; use a new output directory")
    for field, config_field in (("previous_inputs", "previous_input_directories"),
                                ("skip_completed_inputs", "skip_completed_input_directories")):
        if [r["directory"] for r in selection[field]] != configuration[config_field]:
            raise RuntimeError("Frozen input snapshot directory list differs from CLI configuration")
    return selection


def verify_inputs(selection, contract):
    ref.check_hashes(contract)
    for group in ("previous_inputs", "skip_completed_inputs"):
        for record in selection[group]:
            if ref.sha(record["contract_path"]) != record["contract_sha256"]:
                raise RuntimeError("Selected input contract drift")
            for checkpoint in record["checkpoints"]:
                if ref.sha(checkpoint["path"]) != checkpoint["sha256"]:
                    raise RuntimeError("Selected input checkpoint drift")
    for entry in selection["selected"]:
        for kind in ("baseline", "source"):
            if ref.sha(entry[kind+"_checkpoint_path"]) != entry[kind+"_checkpoint_sha256"]:
                raise RuntimeError("Selected science checkpoint drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Native observed reconstruction/planning only; no audit or calibration toys")
    parser.add_argument("--scope", choices=[*ref.GRIDS, *ref.ALIASES])
    parser.add_argument("--masses", help="Comma-separated integers or inclusive ranges")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--previous-input", type=Path, action="append", default=[])
    parser.add_argument("--skip-completed-input", type=Path, action="append", default=[],
                        help="Skip already completed sampling coordinates, irrespective of their outcomes")
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-minutes", type=float, default=60.)
    parser.add_argument("--max-spectra", type=int, default=1500000)
    parser.add_argument("--max-memory-gib", type=float, default=4.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not 1 <= args.batch_size <= 24 or args.batch_index < 1
            or not math.isfinite(args.max_minutes) or args.max_minutes <= 0
            or (args.max_minutes > 60 and args.batch_size != 1)
            or not 0 < args.max_spectra <= 1500000):
        parser.error("At most 24 coordinates/1.5M spectra; scheduling slices above 60 minutes require --batch-size 1")
    if not math.isfinite(args.max_memory_gib) or not 0 < args.max_memory_gib <= 4.:
        parser.error("This execution derivative allows a positive memory guard of at most 4 GiB per worker")
    if (args.attempt == 2) != bool(args.previous_input):
        parser.error("Attempt2 requires --previous-input; attempt1 uses the original baseline")
    baseline_contract, baseline = ref.load_baseline()
    baseline_hash = ref.sha(BASE/"contract.json")
    mass_filter = None
    if args.masses:
        mass_filter = set()
        try:
            for item in args.masses.split(","):
                ends = item.split("-")
                if len(ends) == 1:
                    mass_filter.add(int(item))
                elif len(ends) == 2 and int(ends[1]) >= int(ends[0]):
                    mass_filter.update(range(int(ends[0]), int(ends[1])+1))
                else:
                    raise ValueError(item)
        except ValueError:
            parser.error("Invalid mass list")
    scope_filter = ref.ALIASES.get(args.scope, args.scope)
    requested = args.output or Path("chunked_v1")/f"attempt{args.attempt}_batch{args.batch_index:03}"
    out = (requested if requested.is_absolute() else HERE/requested).resolve()
    if not out.is_relative_to(HERE) or out == HERE or out == BASE or BASE in out.parents:
        parser.error("Use a separate output tree inside the calibration study")
    input_directories = [p.resolve() for p in [*args.previous_input, *args.skip_completed_input]]
    if any(out == p or out in p.parents or p in out.parents for p in input_directories):
        parser.error("Output must be separate from all prior input directories")
    if any(out.glob("*/m*/FAILURE.txt")):
        parser.error("A failure exists in this output tree; use a new output directory")
    layout = layout_marker(args.max_memory_gib)
    selection_path = out/"selection.json"
    configuration = dict(attempt=args.attempt, scope=scope_filter,
        masses=sorted(mass_filter) if mass_filter is not None else None,
        batch_index=args.batch_index, batch_size=args.batch_size, max_minutes=args.max_minutes,
        max_spectra=args.max_spectra, max_memory_gib=args.max_memory_gib, output_directory=str(out),
        previous_input_directories=[str(p.resolve()) for p in args.previous_input],
        skip_completed_input_directories=[str(p.resolve()) for p in args.skip_completed_input])
    # plan_only is intentionally an invocation mode, not a science/selection
    # setting: a saved plan can be executed with otherwise identical arguments.
    selection = saved_selection(selection_path, configuration, layout, baseline_hash)
    previous_snapshot = selection["previous_inputs"] if selection is not None else [None]*len(args.previous_input)
    skip_snapshot = selection["skip_completed_inputs"] if selection is not None else [None]*len(args.skip_completed_input)
    sources, previous_inputs = dict(baseline), []
    if args.attempt == 2:
        sources = {}
        for directory, snapshot in zip(args.previous_input, previous_snapshot):
            records, identity = sampling_input(directory, baseline_contract, baseline,
                required_attempt=1, frozen_identity=snapshot)
            if set(records) & set(sources):
                raise RuntimeError("Attempt2 source coordinates must be distinct")
            sources.update(records)
            previous_inputs.append(identity)
    skipped, skip_inputs = {}, []
    for directory, snapshot in zip(args.skip_completed_input, skip_snapshot):
        records, identity = sampling_input(directory, baseline_contract, baseline,
            required_attempt=args.attempt, frozen_identity=snapshot)
        for key, (path, _) in records.items():
            skipped.setdefault(key, []).append(str(path))
        skip_inputs.append(identity)
    if selection is None:
        eligible = []
        for key, (path, data) in sources.items():
            components = ref.eligible_components(data)
            if not components:
                continue
            base_path, _ = baseline[key]
            eligible.append(dict(scope_key=key[0], mass_MeV=key[1], components=components,
                priority=min(c["priority"] for c in components), worst_ess=min(c["worst_ess"] for c in components),
                baseline_checkpoint_path=str(base_path), baseline_checkpoint_sha256=ref.sha(base_path),
                source_checkpoint_path=str(path), source_checkpoint_sha256=ref.sha(path)))
        eligible.sort(key=lambda e: (e["priority"], e["worst_ess"], list(ref.GRIDS).index(e["scope_key"]), e["mass_MeV"]))
        filtered = [e for e in eligible if (scope_filter is None or e["scope_key"] == scope_filter)
                    and (mass_filter is None or e["mass_MeV"] in mass_filter)
                    and (e["scope_key"], e["mass_MeV"]) not in skipped]
        start_index = (args.batch_index-1)*args.batch_size
        selected = filtered[start_index:start_index+args.batch_size]
        selection = dict(type=ref.TYPE, version=ref.VERSION, policy=ref.POLICY,
            numerical_policy=ref.NUMERICAL_POLICY, baseline_contract_sha256=baseline_hash,
            attempt=args.attempt, scope=scope_filter, masses=configuration["masses"],
            batch_index=args.batch_index, batch_size=args.batch_size, selected=selected, eligible=eligible,
            deferred=[e for e in eligible if e not in selected and (e["scope_key"], e["mass_MeV"]) not in skipped],
            skipped_completed=[dict(scope_key=k[0], mass_MeV=k[1], checkpoint_paths=v) for k, v in sorted(skipped.items())],
            previous_inputs=previous_inputs, skip_completed_inputs=skip_inputs,
            batch_minutes=args.max_minutes, batch_calibration_spectra=args.max_spectra,
            max_memory_gib=args.max_memory_gib, execution_layout=layout, cli_configuration=configuration,
            scheduling_interpretation="Batch boundaries defer work; they are not an overall stopping condition",
            selection_inputs="Original MC eligibility/priority and censoring; completed sampling coordinates skipped without reading validation")
    else:
        # Use the archived selection directly: no live-directory rescan or
        # recomputation of which coordinates should be selected/skipped.
        selected = selection["selected"]
    ref.write_json(selection_path, selection, freeze=True)
    extra = [Path(__file__).resolve(), HERE/"CHUNKED_REFINEMENT_PROTOCOL.md", selection_path,
             Path(ref.__file__).resolve(), HERE/"REFINEMENT_PROTOCOL.md",
             HERE/"provenance/additional_runtime_hashes.json"]
    for row in ref.read_json(HERE/"provenance/additional_runtime_hashes.json")["checks"]:
        if ref.sha(row["path"]) != row["reference_sha256"]:
            raise RuntimeError("Companion runtime hash mismatch")
        extra.append(Path(row["path"]))
    contract = dict(version=1, ntoy=None, nvalid=500, hashes=baseline_contract["hashes"],
        sampling_hashes={str(p.relative_to(ROOT)): ref.sha(p) for p in extra},
        sampling_refinement=dict(type=ref.TYPE, version=ref.VERSION, policy=ref.POLICY,
            numerical_policy=ref.NUMERICAL_POLICY, attempt=args.attempt,
            baseline_contract_sha256=baseline_hash, baseline_contract_path=str(BASE/"contract.json"),
            selection_path=str(selection_path), selection_sha256=ref.sha(selection_path)), execution_layout=layout)
    ref.write_json(out/"contract.json", contract, freeze=True)
    verify_inputs(selection, contract)
    # Fitting imports happen only after complete first-pass/input/source checks.
    import calibration_core as core
    import run_calibration as frozen
    from threadpoolctl import threadpool_limits
    Context, Bank = runtime_types(core)
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
            ctx = None
            if (destination/"result.json").exists():
                saved = ref.read_json(destination/"result.json")
                if (saved.get("status") != "completed_point" or saved.get("execution_layout") != layout
                        or saved.get("chunked_equivalence_checks", {}).get("passed") is not True):
                    raise RuntimeError("Incomplete/incompatible existing chunked result")
                completed.append(dict(scope_key=key[0], mass_MeV=key[1], resumed_completed=True))
                continue
            if time.monotonic()-clock >= args.max_minutes*60:
                scheduled_deferred.extend(dict(entry=e, reason="batch_time_budget") for e in selected[index:])
                break
            try:
                verify_inputs(selection, contract)
                ctx = Context(scopes[key[0]], key[1], cfg, datasets, states)
                base_data, source = baseline[key][1], sources[key][1]
                if (not math.isclose(ctx.sigma, base_data["sigma_reference"], rel_tol=2e-12)
                        or not math.isclose(ctx.conversion, base_data["signal_yield_per_eps2"], rel_tol=2e-12)):
                    raise RuntimeError("Observed reference normalization changed")
                if any(array_sha(ctx.truths[t]) != base_data["provenance"][t]["truth_sha256"] for t in TRUTHS):
                    raise RuntimeError("Full generating truth hash differs from first pass")
                plan = ref.make_point_plan(ctx, base_data, source, entry["components"], args.attempt)
                window = len(ctx.w)
                approx_rank = sum(min(12, int(p["p"].blind_mask.sum())) for p in ctx.parts)
                plan["unchunked_memory_estimates"] = plan["memory_estimates"]
                plan["memory_estimates"] = {
                    "exact_cached_cholesky": memory_estimate(len(ctx.signal), window, window, plan["truths"]),
                    "eigenfeature_rtol_1e-15": memory_estimate(len(ctx.signal), window, approx_rank, plan["truths"])}
                plan.update(source_checkpoint_sha256=entry["source_checkpoint_sha256"],
                    baseline_contract_sha256=baseline_hash, selection_sha256=ref.sha(selection_path),
                    max_memory_gib=args.max_memory_gib, execution_layout=layout)
                ref.write_json(destination/"point_plan.json", plan, freeze=True)
                nbase = sum(p["n"] for p in base_data["provenance"].values())
                scan_scale = max(len(p["scan_nodes"]) for p in plan["truths"].values())/len(base_data["nodes"])
                estimate = base_data["elapsed_seconds"]*plan["calibration_spectra"]/nbase*max(1., scan_scale)
                print(ref.encoded(dict(event="planned_before_draws", scope_key=key[0], mass_MeV=key[1],
                    calibration_spectra=plan["calibration_spectra"], estimate_seconds=estimate,
                    memory_estimates=plan["memory_estimates"], max_memory_gib=args.max_memory_gib,
                    counts={t: {k: p[k] for k in ("proposal_count", "ntoys_per_proposal")} for t, p in plan["truths"].items()})), flush=True)
                if args.plan_only:
                    completed.append(dict(scope_key=key[0], mass_MeV=key[1], status="planned_no_toys",
                                          calibration_spectra=plan["calibration_spectra"]))
                    continue
                remaining = args.max_minutes*60-(time.monotonic()-clock)
                if generated+plan["calibration_spectra"] > args.max_spectra or estimate > remaining:
                    scheduled_deferred.append(dict(entry=entry, reason="batch_count_or_estimated_time_budget",
                                                   estimate_seconds=estimate))
                    continue
                result = run_point(core, frozen, Bank, ctx, base_data, entry, plan, destination, contract)
                generated += plan["calibration_spectra"]
                completed.append(dict(scope_key=key[0], mass_MeV=key[1], status=result["status"],
                                      elapsed_seconds=result["elapsed_seconds"]))
                verify_inputs(selection, contract)
            except ref.PlanDeferred as error:
                scheduled_deferred.append(dict(entry=entry, reason="declared_geometry_or_memory_cap",
                                               detail=str(error), diagnostics=error.details))
                ref.write_json(destination/"DEFERRED.json", scheduled_deferred[-1])
            except Exception:
                destination.mkdir(parents=True, exist_ok=True)
                (destination/"FAILURE.txt").write_text(traceback.format_exc())
                if ctx is not None:
                    ref.write_json(destination/"failure_numerical_qa.json", dict(
                        numerical_checks=ctx.numerical_checks, scalar_checks=ctx.scalar_checks,
                        gp_backend=getattr(ctx, "gp_backend", None), gp_fallback_reason=getattr(ctx, "gp_fallback_reason", None),
                        approximation_candidate_audit=getattr(ctx, "approximation_candidate_audit", None),
                        chunked_equivalence_checks=ctx.chunked_equivalence_checks,
                        model_chunk_ledger=ctx.model_chunk_ledger))
                raise
            finally:
                ctx = None
                gc.collect()
                ref.write_json(out/"batch_summary.json", dict(completed=completed, scheduled_deferred=scheduled_deferred,
                    other_batches_deferred=selection["deferred"], skipped_completed=selection["skipped_completed"],
                    generated_calibration_spectra=generated, elapsed_seconds=time.monotonic()-clock,
                    plan_only=args.plan_only, invocation_finished=False,
                    scheduling_slice="Current invocation in progress; deferred work remains explicit",
                    remaining_work="Deferred and unresolved endpoints require continued batches within the user's budget floor"))
        verify_inputs(selection, contract)
        ref.write_json(out/"batch_summary.json", dict(completed=completed, scheduled_deferred=scheduled_deferred,
            other_batches_deferred=selection["deferred"], skipped_completed=selection["skipped_completed"],
            generated_calibration_spectra=generated, elapsed_seconds=time.monotonic()-clock,
            plan_only=args.plan_only, invocation_finished=True,
            scheduling_slice="Current invocation; completed checkpoints were skipped",
            remaining_work="Deferred and unresolved endpoints require continued batches within the user's budget floor"))


if __name__ == "__main__":
    main()
