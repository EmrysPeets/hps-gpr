#!/usr/bin/env python3
"""Auxiliary 71 MeV reverse-truth diagnostic; never a calibration truth."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
import numpy as np
import run_sampling_refinement as ref
import collect_results as collector

HERE, ROOT = ref.HERE, ref.ROOT
SCOPE, MASS, NVALID = "individual_2021_10pct", 71, 500
STRENGTHS, METHODS = (0, 2, 5), ("profiled", "fixed")
PARENT = ROOT / "study_results/v4p9p13_background_profiling_20260905"
LEGACY = PARENT / "injections/derived"
TRUTH = ROOT / "study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/reverse_injection/derived/common_truth_and_signals.csv"
TOLERANCES = dict(Atrue_abs=1e-7, Ahat_abs_counts=.05, signed_r_abs=2e-5,
                  raw_cls_abs=2e-5, scalar_batch_q_abs=1e-4)


def rows(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight(checkpoint):
    baseline_contract, baseline = ref.load_baseline()  # Requires all 456 completed points.
    data = ref.read_json(checkpoint)
    require((data["scope_key"], data["mass_MeV"]) == (SCOPE, MASS), "Select only 2021 10%, 71 MeV")
    require(data["status"] == "completed_point" and data["nvalidation"] == 500,
            "Select a completed final checkpoint, not a pilot")
    require(data["confidence_level"] == .9 and data["cls_target"] == .1, "Changed inference target")
    contract_path = checkpoint.parents[2] / "contract.json"
    selected_contract = ref.read_json(contract_path)
    require(selected_contract["hashes"] == baseline_contract["hashes"], "Original 47 hashes differ")
    baseline_sha = ref.sha(ref.BASE / "contract.json")
    source_files = [Path(__file__), HERE / "REVERSE_TRUTH_71_PROTOCOL.md", Path(ref.__file__),
                    Path(collector.__file__), checkpoint, contract_path, ref.BASE / "contract.json",
                    checkpoint.parent / "validation_summary.csv", PARENT / "MANIFEST.sha256"]
    derivative = data.get("sampling_refinement")
    if derivative:
        ref.check_hashes(selected_contract)
        for marker in (derivative, selected_contract.get("sampling_refinement", {})):
            require(marker.get("type") == ref.TYPE and marker.get("version") == ref.VERSION
                    and marker.get("baseline_contract_sha256") == baseline_sha,
                    "Unsupported sampling derivative")
        for name in ("point_plan", "proposal_plan"):
            path = checkpoint.parent / (name + ".json")
            require(ref.sha(path) == derivative[name + "_sha256"], "Changed derivative plan")
            source_files.append(path)
        require(derivative["baseline_checkpoint_sha256"] == ref.sha(baseline[SCOPE, MASS][0]),
                "Derivative names a different original point")
    else:
        require(ref.sha(checkpoint) == ref.sha(baseline[SCOPE, MASS][0]), "Not the original completed point")
    qa = collector.point_qa(data, checkpoint, rows(checkpoint.parent / "validation_summary.csv"))
    require(qa["numerical_pass"], "Selected checkpoint lacks passed numerical QA")
    records = ref.endpoint_records(data)
    counts = data.get("ntoys_per_proposal_by_truth") or {t: data["ntoys_per_proposal"] for t in ref.TRUTHS}
    for truth in ref.TRUTHS:
        n = counts[truth]
        require(isinstance(n, int) and n in (256, 512, 1024), "Unsupported proposal count")
        require(data["provenance"][truth]["n"] == n * len(data["provenance"][truth]["meta"]["labels"]),
                "Selected bank count/labels mismatch")
        if derivative:
            require(all(records[truth, m]["ntoys_per_proposal"] == n for m in METHODS), "Truth row count mismatch")
    old_summary = ref.read_json(LEGACY / "summary.json")
    require(old_summary["toys_per_coordinate"] == 500, "Legacy ensemble size changed")
    for path, expected in old_summary["sources"].items():
        require(ref.sha(path) == expected, "Legacy source/input hash mismatch: " + path)
        source_files.append(Path(path))
    manifest = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                for line in (PARENT / "MANIFEST.sha256").read_text().splitlines() if line.strip()}
    protected = [LEGACY / "summary.json", LEGACY / "frozen_injection_strengths.json"]
    protected += [LEGACY / "checkpoints" / f"retrained_sidebands_m071_s{s}.csv.gz" for s in STRENGTHS]
    for path in protected:
        require(ref.sha(path) == manifest[str(path.relative_to(ROOT))], "Legacy artifact differs from release manifest")
        source_files.append(path)
    additional_path = HERE / "provenance/additional_runtime_hashes.json"
    for item in ref.read_json(additional_path)["checks"]:
        require(ref.sha(item["path"]) == item["reference_sha256"], "Companion GP runtime drift")
        source_files.append(Path(item["path"]))
    source_files.append(additional_path)
    frozen = dict(type="reverse_truth_71_auxiliary_validation", version=1,
        checkpoint=str(checkpoint), checkpoint_sha256=ref.sha(checkpoint),
        baseline_contract_sha256=baseline_sha, original_hashes=baseline_contract["hashes"],
        auxiliary_hashes={str(p): ref.sha(p) for p in dict.fromkeys(source_files)},
        selected_sampling_hashes=selected_contract.get("sampling_hashes", {}),
        nvalidation=500, strengths=list(STRENGTHS), original_seed=[491305, 2, 71, "strength"],
        tolerances=TOLERANCES, selected_ntoys_per_proposal_by_truth=counts,
        calibration_truths=["gp", "stress"], validation_truth="legacy_66MeV_kernel_exclude60_86",
        dense_backend="archived_fit_gpr_full_covariance", confidence_level=.9, cls_target=.1)
    return data, baseline[SCOPE, MASS][1], frozen


def verify_sources(contract):
    for path, expected in contract["original_hashes"].items():
        require(ref.sha(ROOT / path) == expected, "Frozen inference drift")
    for path, expected in contract["auxiliary_hashes"].items():
        require(ref.sha(path) == expected, "Auxiliary source drift: " + path)
    for path, expected in contract["selected_sampling_hashes"].items():
        require(ref.sha(ROOT / path) == expected, "Selected sampling source drift")


class ArchivedDensePredictor:
    def __init__(self, core, part, cfg):
        self.core, self.part, self.cfg = core, part, cfg

    def predict(self, y):
        c, p = self.core.c, self.part
        gp = c.fit_gpr(p["p"].x_full[p["keep"]], y, self.cfg, restarts=0,
                      kernel=p["kernel"], optimize=False)
        return c.predict_counts_from_log_gpr(gp, p["p"].x_full[p["p"].blind_mask], self.cfg)


def ess(values):
    square = float(values @ values)
    return float(values.sum()**2 / square) if square > 0 else 0.


def cls_tails(bank, q, wb, ws, thresholds):
    """Frozen Bank.tails formula, with the common density weights evaluated once."""
    result = []
    for threshold in thresholds:
        if threshold <= 0:
            result.append(dict(cls=1., se=0., ess_b=float(bank.n), ess_s=float(bank.n)))
            continue
        indicator = q >= threshold - 1e-10
        vb, vs = wb * indicator, ws * indicator
        pb, _ = bank.moment(vb)
        ps, _ = bank.moment(vs)
        if pb <= 0:
            result.append(dict(cls=math.inf, se=math.inf, ess_b=ess(vb), ess_s=ess(vs)))
            continue
        ratio = ps / pb
        _, se = bank.moment((ws - ratio * wb) * indicator / pb)
        result.append(dict(cls=ratio, se=se, ess_b=ess(vb), ess_s=ess(vs)))
    return result


def local_tails(bank, r, wb, thresholds):
    result = []
    for threshold in thresholds:
        if threshold <= 0:
            result.append(dict(p0=1., se=0., ess=float(bank.n)))
        else:
            values = wb * (r >= threshold - 1e-10)
            value, se = bank.moment(values)
            result.append(dict(p0=value, se=se, ess=ess(values)))
    return result


def run(core, selected, baseline, contract, out, audit_state):
    c, pd = core.c, core.pd
    cfg = c.production.load_config(c.production.DEFAULT_CARD)
    c.production.validate_card(cfg)
    c.production.validate_histogram_inputs(cfg)
    c.production.validate_input_provenance(c.production.DEFAULT_INPUT_PROVENANCE,
        c.production.DEFAULT_CARD, c.production.DEFAULT_STATES, cfg)
    datasets = c.production.make_datasets(cfg)
    states = c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    ctx = core.Context(next(s for s in core.SCOPES if s[0] == SCOPE), MASS, cfg, datasets, states)
    audit_state["ctx"] = ctx
    require(math.isclose(ctx.sigma, selected["sigma_reference"], rel_tol=2e-12), "Reference sigma changed")
    require(math.isclose(ctx.conversion, selected["signal_yield_per_eps2"], rel_tol=2e-12), "Signal conversion changed")
    core.enable_lowrank(ctx)
    require(ctx.gp_backend == baseline["gp_backend"], "First-pass proposal backend did not reproduce")
    arrays, metadata, plans = {}, {}, {}
    for truth in ref.TRUTHS:
        audit_state["stage"] = "proposal_reconstruction_" + truth
        require(ref.array_sha(ctx.truths[truth]) == selected["provenance"][truth]["truth_sha256"], "Generating truth changed")
        nodes = selected.get("proposal_nodes_by_truth", {}).get(truth, selected["nodes"])
        arrays[truth], metadata[truth] = ctx.proposals(ctx.truths[truth], nodes)
        require(ref.array_sha(arrays[truth]) == selected["provenance"][truth]["proposals_sha256"], "Proposal-law hash mismatch")
        require(metadata[truth]["labels"] == selected["provenance"][truth]["meta"]["labels"], "Proposal labels changed")
        n = contract["selected_ntoys_per_proposal_by_truth"][truth]
        plans[truth] = dict(calibration_spectra=n*len(arrays[truth]), proposal_count=len(arrays[truth]), scan_nodes=[0., 2., 5.])
    memory = ref.memory_estimate(len(ctx.signal), len(ctx.w), len(ctx.w), plans)
    memory.update(limit_gib=contract["max_memory_gib"], passed=memory["estimated_peak_gib"] <= contract["max_memory_gib"])
    ref.write_json(out / "memory_check.json", memory)
    require(memory["passed"], "Dense-bank memory guard exceeded; no mesh/count reduction permitted")
    # Proposal generation is now frozen. Both statistic ensembles use the legacy dense refit.
    for part in ctx.parts:
        part["predictor"] = ArchivedDensePredictor(core, part, cfg)
    ctx.nuisance_cut, ctx.gp_backend = 0., "archived_fit_gpr_full_covariance"
    old_reference = next(r for r in ref.read_json(LEGACY / "frozen_injection_strengths.json") if r["mass_MeV"] == MASS)
    Atrue = {s: s * old_reference["sigma_profiled"] for s in STRENGTHS}
    banks, stats, provenance, readiness = {}, {}, {}, []
    for truth in ref.TRUTHS:
        audit_state["stage"] = "dense_calibration_" + truth
        n = contract["selected_ntoys_per_proposal_by_truth"][truth]
        seed_args = selected["provenance"][truth].get("seed_namespace", ["calibration", SCOPE, MASS, truth, n])
        marker = selected.get("sampling_refinement", {})
        expected = (["sampling-refinement-v1", SCOPE, MASS, truth, marker["attempt"], ref.array_sha(arrays[truth])]
                    if truth in marker.get("refined_truths", []) else ["calibration", SCOPE, MASS, truth, 256])
        require(seed_args == expected, "Selected calibration seed namespace differs")
        rng = core.seed(*seed_args)
        whole = np.concatenate([rng.poisson(mean, size=(n, len(mean))) for mean in arrays[truth]])
        require(ref.array_sha(whole) == selected["provenance"][truth]["whole_sha256"], "Whole calibration-bank hash mismatch")
        ref.write_json(out / f"calibration_array_closure_{truth}.json", dict(
            seed_namespace=seed_args, n=len(whole), proposals_sha256=ref.array_sha(arrays[truth]),
            whole_sha256=ref.array_sha(whole), passed=True))
        bank = core.Bank(ctx, ctx.truths[truth], whole, arrays[truth], np.repeat(np.arange(len(arrays[truth])), n))
        banks[truth] = bank
        stored = dict(strata=bank.strata, weights_0=bank.weights(0.))
        for method in METHODS:
            stored["r_" + method] = bank.models[method].r
        for strength in STRENGTHS:
            a = Atrue[strength] / ctx.sigma
            weight = bank.weights(a)
            mean, se = bank.moment(weight)
            check = dict(truth=truth, strength=strength, Atrue=Atrue[strength], strength_new_sigma=a,
                normalization=mean, normalization_se=se,
                passed=bool(np.isfinite(weight).all() and np.isfinite(se) and se <= .05 and abs(mean-1) <= max(.05, 5*se)))
            readiness.append(check)
            stored[f"weights_{strength}"] = weight
            if strength:
                for method in METHODS:
                    stored[f"q_{method}_{strength}"] = bank.models[method].q(Atrue[strength])
        stats[truth] = stored
        np.savez_compressed(out / f"dense_calibration_statistics_{truth}.npz", **stored)
        provenance[truth] = dict(n=len(whole), ntoys_per_proposal=n, seed_namespace=seed_args,
            proposals_sha256=ref.array_sha(arrays[truth]), whole_sha256=ref.array_sha(whole),
            truth_sha256=ref.array_sha(ctx.truths[truth]), arrays_sha256={k: ref.array_sha(v) for k, v in stored.items()},
            max_score=max(m.max_score for m in bank.models.values()),
            scalar_fallbacks=sum(m.fallbacks for m in bank.models.values()))
        ref.write_json(out / "calibration_provenance.json", provenance)
        ref.write_json(out / "normalization_readiness.json", readiness)
        require(all(r["passed"] for r in readiness), "Calibration density normalization is not ready; stop before legacy validation")
        print(f"Reconstructed dense {truth} calibration bank: {len(whole)} spectra", flush=True)

    part = ctx.parts[0]
    p = part["p"]
    truth_table = pd.read_csv(TRUTH)
    require(np.allclose(truth_table.mass_MeV.to_numpy()/1000, p.x_full, rtol=0, atol=1e-15), "Reverse-truth bin mismatch")
    reverse = truth_table.smooth_truth_counts.to_numpy(float)
    window, full = c.build_window_template_from_full(p.edges_full, p.blind_mask, MASS/1000, p.sigma_val, config=cfg)
    fraction = float(window.sum())
    require(abs(fraction-old_reference["window_fraction"]) < 1e-14 and len(ctx.w) == old_reference["n_window"], "Legacy mask/template changed")
    require(np.allclose(window/fraction, ctx.w, rtol=1e-13, atol=1e-15)
            and np.allclose(full/fraction, ctx.signal, rtol=1e-13, atol=1e-15), "Window/full signal normalization mismatch")
    detail, table, closure = [], [], []
    for strength in STRENGTHS:
        audit_state["stage"] = f"legacy_validation_s{strength}"
        rng = np.random.default_rng(np.random.SeedSequence([491305, 2, 71, strength]))
        whole = np.array([rng.poisson(reverse + Atrue[strength]/fraction*full).astype(float) for _ in range(500)])
        np.savez_compressed(out / f"legacy_spectra_s{strength}.npz", whole=whole, x_GeV=p.x_full)
        old = pd.read_csv(LEGACY / "checkpoints" / f"retrained_sidebands_m071_s{strength}.csv.gz")
        require(len(old) == 1000 and set(old.method) == set(METHODS), "Legacy paired table incomplete")
        models = ctx.make_models(whole)
        for method, model in models.items():
            legacy = old[old.method == method].sort_values("toy_id")
            require(np.array_equal(legacy.toy_id.to_numpy(), np.arange(500)), "Legacy toy IDs changed")
            q = model.q(Atrue[strength]) if strength else np.full(500, np.nan)
            raw, q_asimov = np.full(500, np.nan), np.full(500, np.nan)
            scalar_q_error = 0.
            if strength:
                for i in range(500):
                    scalar = c.Profile(model.b[i], model.L[i], ctx.w, "linear")
                    qa = 2*scalar.fit(model.b[i], Atrue[strength])["nll"]
                    q_asimov[i] = qa
                    raw[i] = c.bounded_tildeq_asymptotic_tails(float(q[i]), float(qa)).cls
                    if i < 2:
                        free, null = scalar.fit(model.n[i]), scalar.fit(model.n[i], 0.)
                        fixed = scalar.fit(model.n[i], Atrue[strength])
                        scalar_q = (0. if free["A"] > Atrue[strength] else
                            max(0., 2*(fixed["nll"]-(free["nll"] if free["A"] >= 0 else null["nll"]))))
                        require(np.isfinite(scalar_q-q[i]), "Nonfinite scalar/batch q comparison")
                        scalar_q_error = max(scalar_q_error, abs(scalar_q-q[i]))
            tails = {t: (cls_tails(banks[t], stats[t][f"q_{method}_{strength}"], stats[t]["weights_0"],
                                  stats[t][f"weights_{strength}"], q) if strength else
                          local_tails(banks[t], stats[t]["r_"+method], stats[t]["weights_0"], model.r)) for t in ref.TRUTHS}
            if strength:
                for truth in ref.TRUTHS:
                    reference = banks[truth].tails(method, Atrue[strength]/ctx.sigma, float(q[0]))
                    require(all(np.isclose(tails[truth][0][key], reference[key], rtol=1e-10, atol=1e-12)
                                for key in ("cls", "se", "ess_b", "ess_s")), "Cached tail formula differs from frozen Bank.tails")
            errors = dict(Atrue_abs=float(np.max(abs(legacy.Atrue.to_numpy()-Atrue[strength]))),
                Ahat_abs_counts=float(np.max(abs(legacy.Ahat.to_numpy()-model.free["A"]))),
                signed_r_abs=float(np.max(abs(legacy.signed_r.to_numpy()-model.r))),
                raw_cls_abs=float(np.max(abs(legacy.cls_at_true.to_numpy()-raw))) if strength else 0.,
                scalar_batch_q_abs=float(scalar_q_error))
            passed = all(np.isfinite(v) and v <= TOLERANCES[k] for k, v in errors.items())
            if strength:
                passed = passed and np.array_equal(raw < .1, legacy.true_yield_excluded.to_numpy(bool))
            closure.append(dict(strength=strength, method=method, n=500, passed=bool(passed),
                whole_sha256=ref.array_sha(whole), errors=errors))
            cell = []
            for i in range(500):
                values = [tails[t][i] for t in ref.TRUTHS]
                key, threshold = ("cls", .1) if strength else ("p0", .05)
                estimate = max(v[key] for v in values)
                finite = all(np.isfinite(v[key]) and np.isfinite(v["se"]) for v in values)
                ready = finite and all((v["ess_b"] >= 100 and v["ess_s"] >= 100) if strength else v["ess"] >= 100 for v in values)
                tail_status = "ready" if ready else "limited_mc"
                if not finite:
                    tail_status = "unresolved_nonfinite"
                elif any(v[key] == 0 for v in values):
                    tail_status = "unresolved_zero_tail"
                elif not strength and estimate > 1:
                    active = max(values, key=lambda v: v[key])
                    tail_status = "mc_boundary" if estimate <= 1+3*active["se"] else "unresolved_out_of_range"
                    ready = ready and tail_status == "mc_boundary"
                low = max(max(0., v[key]-1.96*v["se"]) for v in values)
                high = max(v[key]+1.96*v["se"] for v in values)
                record = dict(strength=strength, method=method, toy_id=i, Atrue=Atrue[strength],
                    Ahat=float(model.free["A"][i]), signed_r=float(model.r[i]), q_at_Atrue=float(q[i]),
                    q_asimov_at_Atrue=float(q_asimov[i]), raw_cls_at_Atrue=float(raw[i]), calibrated_cls=estimate if strength else math.nan,
                    calibrated_p0_unclipped=estimate if not strength else math.nan,
                    calibrated_p0=min(1., estimate) if not strength and finite and estimate > 0 and tail_status != "unresolved_out_of_range" else math.nan,
                    raw_rejected=bool(raw[i] < .1) if strength else bool(model.r[i] > core.norm.isf(.05)),
                    calibrated_rejected=bool(estimate < threshold), tail_mc_ready=bool(ready), tail_status=tail_status,
                    mc_low=low, mc_high=high, mc_decision_resolved=bool(ready and (high < threshold or low >= threshold)),
                    max_score=model.max_score)
                for truth in ref.TRUTHS:
                    record.update({truth+"_"+k: v for k, v in tails[truth][i].items()})
                cell.append(record)
            detail.extend(cell)
            for name in ("raw", "calibrated"):
                k = sum(r[name+"_rejected"] for r in cell)
                lo, hi = core.interval(k, 500)
                table.append(dict(scope_key=SCOPE, mass_MeV=71, validation_truth="reverse_exclude60_86",
                    method=method, strength=strength, Atrue=Atrue[strength], test="exclusion" if strength else "local_5pct",
                    procedure=name, rejected=k, n=500, fraction=k/500, binomial_low=lo, binomial_high=hi,
                    tail_mc_ready_count=sum(r["tail_mc_ready"] for r in cell), mc_decision_resolved_count=sum(r["mc_decision_resolved"] for r in cell),
                    legacy_closure_passed=bool(passed)))
            pd.DataFrame(detail).to_csv(out / "validation_toys.csv.gz", index=False, compression="gzip")
            ref.write_json(out / "legacy_closure.json", closure)
            require(passed, f"Legacy Atrue/Ahat/r/asymptotic classification mismatch at strength {strength}, {method}")
        print(f"Legacy 71 MeV strength {strength}: 500 paired spectra checked", flush=True)
    pd.DataFrame(table).to_csv(out / "results_table.csv", index=False)
    ref.write_json(out / "numerical_qa.json", dict(numerical_checks=ctx.numerical_checks,
        scalar_checks=ctx.scalar_checks, calibration=provenance, legacy_closure=closure,
        dense_backend=ctx.gp_backend, nuisance_eigenvalue_cut=0., retained_full_covariance=True))
    verify_sources(contract)
    return dict(status="completed_auxiliary_diagnostic", scope_key=SCOPE, mass_MeV=71,
        validation_spectra=1500, paired_method_rows=3000, calibration_spectra=sum(v["n"] for v in provenance.values()),
        n_tail_mc_limited=sum(not r["tail_mc_ready"] for r in detail),
        n_mc_decision_unresolved=sum(not r["mc_decision_resolved"] for r in detail),
        claim_boundary="Retrospective validation on one known out-of-envelope reverse truth; no unconditional or global coverage, no new calibration truth, and no pooling of reused toys")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-memory-gib", type=float, default=4.)
    args = parser.parse_args()
    require(np.isfinite(args.max_memory_gib) and args.max_memory_gib > 0, "Invalid memory guard")
    checkpoint = args.checkpoint.resolve()
    selected, baseline, contract = preflight(checkpoint)
    contract["max_memory_gib"] = args.max_memory_gib
    out = (args.output or HERE / "reverse_truth_71" / ("checkpoint_"+ref.sha(checkpoint)[:12])).resolve()
    require(out.is_relative_to(HERE / "reverse_truth_71"), "Use a separate reverse_truth_71 output directory")
    require(not (out / "FAILURE.txt").exists() and not (out / "failure_numerical_qa.json").exists(),
            "This auxiliary output contains a failure; select a new --output directory to preserve it")
    ref.write_json(out / "contract.json", contract, freeze=True)
    if args.preflight_only:
        print(json.dumps(dict(status="preflight_passed_no_fits_or_toys", output=str(out)), indent=2))
        return
    require(not (out / "summary.json").exists(), "Completed diagnostic exists; do not pool or silently rerun")
    import calibration_core as core
    from threadpoolctl import threadpool_limits
    start = time.monotonic()
    audit_state = {}
    try:
        with threadpool_limits(limits=1):
            summary = run(core, selected, baseline, contract, out, audit_state)
        summary["elapsed_seconds"] = time.monotonic()-start
        # A redirected live log grows when the final summary is printed.
        summary["output_sha256"] = {p.name: ref.sha(p) for p in sorted(out.iterdir())
                                    if p.is_file() and p.name != "summary.json" and p.suffix != ".log"}
        ref.write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except Exception:
        (out / "FAILURE.txt").write_text(traceback.format_exc())
        ctx = audit_state.get("ctx")
        ref.write_json(out / "failure_numerical_qa.json", dict(stage=audit_state.get("stage"),
            numerical_checks=getattr(ctx, "numerical_checks", []), scalar_checks=getattr(ctx, "scalar_checks", []),
            gp_backend=getattr(ctx, "gp_backend", None)))
        raise


if __name__ == "__main__":
    main()
