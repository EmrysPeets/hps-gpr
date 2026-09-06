#!/usr/bin/env python3
"""Replay ten existing proposal RNG calls and the known failed 128-row chunk."""
from pathlib import Path
import argparse
import traceback
import scalar_reference_recovery as recovery
from run_scalar_reference_recovery import layout_marker

HERE, ref, base = recovery.HERE, recovery.ref, recovery.base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="Explicit small numerical replay after source review")
    parser.add_argument("--output",type=Path,default=HERE/"scalar_reference_recovery_v1/diagnostic_m074")
    args=parser.parse_args()
    layout=layout_marker();inputs=recovery.verify_recovery_layout(layout)
    if not args.run:
        print(ref.encoded(dict(status="source_checks_only",numerical_execution=False)))
        return
    out=args.output.resolve()
    recovery.require(out.is_relative_to(HERE/"scalar_reference_recovery_v1")
        and out != HERE/"scalar_reference_recovery_v1" and not out.exists(),
        "Use a new diagnostic directory; failed diagnostics must be preserved")
    contract=dict(execution_layout=layout,input_sha256={str(p.relative_to(recovery.ROOT)):ref.sha(p) for p in inputs},
                  rng_calls=10,calibration_or_validation_study=False)
    ref.write_json(out/"contract.json",contract,freeze=True)
    ctx=None
    try:
        import calibration_core as core
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            facade,Context,_=recovery.runtime_types(core)
            ctx=recovery.load_context(core,Context)
            core.enable_lowrank(ctx);recovery.check_backend(ctx)
            plan=ref.read_json(recovery.FAILED_POINT/"point_plan.json")
            saved_proposals=ref.read_json(recovery.FAILED_POINT/"proposal_plan.json")
            arrays={}
            for truth,spec in plan["truths"].items():
                arrays[truth],meta=ctx.proposals(ctx.truths[truth],spec["proposal_nodes"])
                recovery.require(base.array_sha(arrays[truth])==saved_proposals["proposals"][truth]["sha256"]
                    and meta==saved_proposals["proposals"][truth]["meta"],"Proposal reconstruction differs")
            old=recovery.failure_data()["chunked_equivalence_checks"]["generation_checks"]["stress"]
            rng=core.seed(*old["seed_namespace"]);draw_hashes=[]
            for index in range(10):
                counts=rng.poisson(arrays["stress"][index],size=tuple(old["rng_call_shape"]))
                actual=base.array_sha(counts);draw_hashes.append(actual)
                recovery.require(actual==old["proposal_draw_sha256"][index],"Original proposal RNG replay differs")
            chunk=counts[512:640]
            recovery.require(base.array_sha(chunk)==recovery.KNOWN_CHUNK_SHA,"Failed chunk replay differs")
            window_sha=base.array_sha(core.np.ascontiguousarray(chunk[0,ctx.mask]))
            recovery.require(window_sha==recovery.KNOWN_WINDOW_SHA,"Failed spectrum window replay differs")
            ctx.execution_call_kind="diagnostic_failed_chunk"
            models=ctx.make_models(chunk)
            audit=recovery.recovery_audit(ctx)
            recovery.require(audit["known_failure_reproduced"] and all(m.max_score<recovery.SCORE_GATE for m in models.values()),
                             "Diagnostic did not reproduce and recover the scalar failure")
        paths={"contract.json":out/"contract.json"}
        for name,data in (("scalar_reference_recovery.json",audit),("scalar_checks.json",ctx.scalar_checks),
                          ("model_chunk_ledger.json",ctx.model_chunk_ledger)):
            ref.write_json(out/name,data);paths[name]=out/name
        summary=dict(status="completed_replay_diagnostic",passed=True,execution_layout=layout,
            rng_calls=10,rng_call_shape=old["rng_call_shape"],seed_namespace=old["seed_namespace"],
            proposal_draw_sha256=draw_hashes,proposal_index=9,proposal_toy_index=512,global_toy_index=9728,
            chunk_rows=[9728,9856],chunk_sha256=base.array_sha(chunk),window_sha256=window_sha,
            target_full_counts_sha256=base.array_sha(chunk[0]),gp_backend=ctx.gp_backend,
            nuisance_eigenvalue_cut=ctx.nuisance_cut,fallback_count=audit["fallback_count"],
            source_input_sha256=contract["input_sha256"],output_sha256={n:ref.sha(p) for n,p in paths.items()},
            scope="Same ten original proposal draws, one existing chunk; no new calibration or validation observations")
        recovery.verify_recovery_layout(layout)
        ref.write_json(out/"summary.json",summary)
        print(ref.encoded(dict(passed=True,output=str(out),fallback_count=audit["fallback_count"])))
    except Exception:
        (out/"FAILURE.txt").write_text(traceback.format_exc())
        if ctx is not None:
            ref.write_json(out/"failure_numerical_qa.json",dict(scalar_reference_fallbacks=ctx.scalar_reference_fallbacks,
                scalar_checks=ctx.scalar_checks,model_chunk_ledger=ctx.model_chunk_ledger,numerical_checks=ctx.numerical_checks))
        raise


if __name__=="__main__":main()
