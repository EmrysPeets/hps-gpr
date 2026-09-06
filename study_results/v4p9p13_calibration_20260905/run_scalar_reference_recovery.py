#!/usr/bin/env python3
"""Replay only combined m074 attempt2 with an isolated scalar initializer."""
from pathlib import Path
import argparse
import copy
import gc
import math
import time
import traceback
import scalar_reference_recovery as recovery

HERE, ROOT, ref, base = recovery.HERE, recovery.ROOT, recovery.ref, recovery.base


def layout_marker(max_memory_gib=8.):
    recovery.require(type(max_memory_gib) in (int,float) and max_memory_gib == 8.,
                     "This one-coordinate recovery uses the reviewed 8 GiB bound")
    layout = recovery.resource.layout_marker(max_memory_gib)
    layout["scalar_reference_recovery"] = recovery.recovery_marker()
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicit numerical execution after source/diagnostic review")
    parser.add_argument("--diagnostic-dir", type=Path)
    parser.add_argument("--output", type=Path, default=HERE/"scalar_reference_recovery_v1/attempt2_m074")
    args = parser.parse_args()
    layout = layout_marker()
    inputs = set(recovery.verify_recovery_layout(layout))
    if not args.run:
        print(ref.encoded(dict(status="source_checks_only", scope_key=recovery.SCOPE, mass_MeV=74,
              attempt=2, max_memory_gib=8, numerical_execution=False)))
        return
    if args.diagnostic_dir is None:
        parser.error("--run requires an explicit completed --diagnostic-dir")
    inputs.update(recovery.verify_diagnostic(args.diagnostic_dir, layout))
    out = args.output.resolve()
    recovery.require(out.is_relative_to(HERE/"scalar_reference_recovery_v1")
        and out != (HERE/"scalar_reference_recovery_v1") and not out.exists(),
        "Use a new separate output directory; prior failures/results are never overwritten")
    original_contract = ref.read_json(recovery.FAILED_ROOT/"contract.json")
    original_selection = ref.read_json(recovery.FAILED_ROOT/"selection.json")
    entries = [e for e in original_selection["selected"] if (e["scope_key"],e["mass_MeV"]) == (recovery.SCOPE,74)]
    recovery.require(len(entries) == 1, "Frozen failed coordinate is missing")
    entry = entries[0]
    _, baseline = ref.load_baseline()
    baseline_data = baseline[(recovery.SCOPE,74)][1]
    selection = copy.deepcopy(original_selection)
    selection.update(selected=[entry], deferred=[], skipped_completed=[], masses=[74], batch_size=1,
        execution_layout=layout, scope=recovery.SCOPE, attempt=2,
        cli_configuration=dict(output_directory=str(out), diagnostic_directory=str(args.diagnostic_dir.resolve()),
                               scope=recovery.SCOPE, masses=[74], attempt=2, max_memory_gib=8.),
        recovery_source_selection_sha256=ref.sha(recovery.FAILED_ROOT/"selection.json"),
        selection_inputs="Only the pre-existing scalar reference failure; no endpoint/validation selection")
    selection_path = out/"selection.json"
    ref.write_json(selection_path, selection, freeze=True)
    inputs.add(selection_path)
    contract = copy.deepcopy(original_contract)
    contract["execution_layout"] = layout
    contract["sampling_refinement"].update(selection_path=str(selection_path), selection_sha256=ref.sha(selection_path))
    contract["scalar_reference_recovery"] = layout["scalar_reference_recovery"]
    for path in inputs:
        name = str(path.relative_to(ROOT))
        if name in contract["hashes"]:
            recovery.require(ref.sha(path) == contract["hashes"][name], "Original inference source changed")
        else:
            contract["sampling_hashes"][name] = ref.sha(path)
    ref.write_json(out/"contract.json", contract, freeze=True)
    base.verify_inputs(selection, contract)
    plan = ref.read_json(recovery.FAILED_POINT/"point_plan.json")
    original_numeric_plan = {k:v for k,v in plan.items() if k not in ("execution_layout","selection_sha256")}
    plan.update(execution_layout=layout, selection_sha256=ref.sha(selection_path))
    recovery.require(original_numeric_plan == {k:v for k,v in plan.items() if k not in ("execution_layout","selection_sha256")},
                     "Recovery changed the prescribed numerical/sampling plan")
    destination = out/recovery.SCOPE/"m074"
    ctx = None; started = time.monotonic()
    try:
        import calibration_core as core
        import run_calibration as frozen
        from threadpoolctl import threadpool_limits
        facade, Context, Bank = recovery.runtime_types(core)
        with threadpool_limits(limits=1):
            ctx = recovery.load_context(core, Context)
            recovery.require(math.isclose(ctx.sigma,baseline_data["sigma_reference"],rel_tol=2e-12)
                and math.isclose(ctx.conversion,baseline_data["signal_yield_per_eps2"],rel_tol=2e-12),
                "Observed normalization differs")
            recovery.require(all(base.array_sha(ctx.truths[t]) == baseline_data["provenance"][t]["truth_sha256"]
                                 for t in base.TRUTHS), "Full truth identity differs")
            result = base.run_point(facade, frozen, Bank, ctx, baseline_data, entry, plan, destination, contract)
            audit = recovery.recovery_audit(ctx)
        ledger_path = destination/"scalar_reference_recovery.json"
        ref.write_json(ledger_path, audit)
        result["scalar_reference_recovery"] = dict(type=recovery.TYPE, version=recovery.VERSION, passed=True,
            ledger_path=str(ledger_path), ledger_sha256=ref.sha(ledger_path),
            diagnostic_directory=str(args.diagnostic_dir.resolve()),
            diagnostic_summary_sha256=ref.sha(args.diagnostic_dir/"summary.json"),
            original_attempt_contract_sha256=ref.sha(recovery.FAILED_ROOT/"contract.json"),
            identical_whole_bank_sha256={t:result["provenance"][t]["whole_sha256"] for t in base.TRUTHS},
            validation_seed_namespaces=[["validation",recovery.SCOPE,74,t,s] for t in base.TRUTHS for s in (0,2,5)])
        recovery.verify_recovery_layout(layout,result)
        base.verify_inputs(selection,contract)
        ref.write_json(destination/"result.json",result)
        ref.write_json(out/"batch_summary.json",dict(invocation_finished=True,passed=True,
            completed=[dict(scope_key=recovery.SCOPE,mass_MeV=74)],scheduled_deferred=[],
            generated_calibration_spectra=plan["calibration_spectra"],elapsed_seconds=time.monotonic()-started,
            scheduling_slice="One frozen failed coordinate; other work is outside this driver"))
        print(ref.encoded(dict(status="completed_reference_recovery",result=str(destination/"result.json"),
                              fallback_count=audit["fallback_count"])))
    except Exception:
        destination.mkdir(parents=True,exist_ok=True)
        (destination/"FAILURE.txt").write_text(traceback.format_exc())
        if (destination/"result.json").exists():
            (destination/"result.json").rename(destination/"unverified_result.json")
        if ctx is not None:
            ref.write_json(destination/"failure_numerical_qa.json",dict(
                scalar_reference_fallbacks=ctx.scalar_reference_fallbacks,scalar_checks=ctx.scalar_checks,
                numerical_checks=ctx.numerical_checks,model_chunk_ledger=ctx.model_chunk_ledger,
                chunked_equivalence_checks=ctx.chunked_equivalence_checks,
                gp_backend=getattr(ctx,"gp_backend",None),nuisance_eigenvalue_cut=ctx.nuisance_cut))
        raise
    finally:
        ctx=None;gc.collect()


if __name__ == "__main__":main()
