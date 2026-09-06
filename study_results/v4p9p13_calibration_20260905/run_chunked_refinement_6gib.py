#!/usr/bin/env python3
"""Combined-only resource wrapper: unchanged chunked inference, at most 6 GiB."""
from pathlib import Path
import argparse
import gc
import math
import sys
import time
import traceback

sys.dont_write_bytecode = True
import run_chunked_refinement as base

# These are the same function objects, not replacements of module globals.
from run_chunked_refinement import (array_sha, memory_estimate, run_point,
    runtime_types, sampling_input, saved_selection, verify_inputs)

np, ref = base.np, base.ref
HERE, ROOT, BASE = base.HERE, base.ROOT, base.BASE
TRUTHS = base.TRUTHS
COMBINED = "all_2015_2016_2021"
RESOURCE_TYPE, RESOURCE_VERSION = "combined_memory_override", 1
BASE_IDENTITIES = {
    "run_chunked_refinement.py": "07ddfe38d74fa8ac5c6d6643606cb64169c645e9cb4aaf1833cac9ba0ebc382a",
    "CHUNKED_REFINEMENT_PROTOCOL.md": "4e4dc3c392a122a343584f2268b1ecc9206574dceac0365a02fecf217d7e9591",
}


def validate_resource_limits(max_memory_gib, scope):
    if (isinstance(max_memory_gib, bool) or not isinstance(max_memory_gib, (int, float))
            or not math.isfinite(max_memory_gib) or not 0 < max_memory_gib <= 6.):
        raise ValueError("Resource wrapper requires a finite positive bound of at most 6 GiB")
    canonical = ref.ALIASES.get(scope, scope)
    if canonical != COMBINED:
        raise ValueError("Resource override is authorized for the all-three combination only")
    return canonical


def verify_base_identity():
    for name, expected in BASE_IDENTITIES.items():
        if ref.sha(HERE/name) != expected:
            raise RuntimeError("Frozen chunked runtime/protocol changed: "+name)


def layout_marker(max_memory_gib):
    """Explicit resource identity consumed by a separate supplemental gate."""
    validate_resource_limits(max_memory_gib, COMBINED)
    verify_base_identity()
    marker = base.layout_marker(max_memory_gib)
    marker["source_hashes"].update({str(p.relative_to(ROOT)): ref.sha(p) for p in
        (Path(__file__).resolve(), HERE/"CHUNKED_RESOURCE_PROTOCOL.md")})
    marker["resource_policy"] = dict(type=RESOURCE_TYPE, version=RESOURCE_VERSION,
        scope_key=COMBINED, max_worker_memory_gib=6., max_companion_workers=1,
        max_companion_memory_gib=4., aggregate_memory_limit_gib=10.,
        fresh_memory_pressure_check_required=True, prelaunch_check_owner="coordinator",
        statistical_policy_unchanged=True)
    return marker


def pure_checks():
    """Only limit, scope, source and function-identity checks; no fits/draws."""
    checks = []
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise AssertionError(name)
    check("accept_six_gib_combined", validate_resource_limits(6., COMBINED) == COMBINED)
    check("accept_four_gib_combined_alias", validate_resource_limits(4., "all") == COMBINED)
    for name, value in (("zero", 0.), ("negative", -1.), ("over_six", 6.000001),
                        ("infinity", math.inf), ("nan", math.nan), ("boolean", True)):
        rejected = False
        try:
            validate_resource_limits(value, COMBINED)
        except ValueError:
            rejected = True
        check("reject_"+name+"_memory", rejected)
    for scope in ("2015", "2016", "2021", "individual_2016_full", None):
        rejected = False
        try:
            validate_resource_limits(6., scope)
        except ValueError:
            rejected = True
        check("reject_scope_"+str(scope), rejected)
    verify_base_identity()
    marker = layout_marker(6.)
    expected_sources = {str((HERE/name).relative_to(ROOT)) for name in
        ("run_chunked_refinement.py", "CHUNKED_REFINEMENT_PROTOCOL.md",
         "run_chunked_refinement_6gib.py", "CHUNKED_RESOURCE_PROTOCOL.md")}
    check("four_source_identities", set(marker["source_hashes"]) == expected_sources
          and all(ref.sha(ROOT/p) == h for p, h in marker["source_hashes"].items()))
    check("original_execution_identity_preserved", all(marker[k] == v for k, v in base.layout_marker(6.).items()
          if k != "source_hashes"))
    check("thread_chunk_and_aggregate_limits", marker["chunk_size"] == 128 and marker["blas_threads"] == 1
          and marker["runtime_reserve_bytes"] == 512*1024**2
          and marker["resource_policy"]["aggregate_memory_limit_gib"] == 10.
          and marker["resource_policy"]["max_companion_workers"] == 1)
    aliases = dict(array_sha=array_sha, memory_estimate=memory_estimate, run_point=run_point,
        runtime_types=runtime_types, sampling_input=sampling_input,
        saved_selection=saved_selection, verify_inputs=verify_inputs)
    for name, function in aliases.items():
        check("identical_function_"+name, function is getattr(base, name))
    inherited = ref.read_json(HERE/"qa/chunked_execution_contract_test.json")
    check("inherited24_case_identity", inherited["passed"] is True and inherited["test_count"] == 24
          and all(row["passed"] is True for row in inherited["checks"])
          and all(ref.sha(ROOT/p) == h for p, h in inherited["source_hashes"].items()))
    original = ref.read_json(BASE/"contract.json")
    check("original47_source_identity", len(original["hashes"]) == 47
          and all(ref.sha(ROOT/p) == h for p, h in original["hashes"].items()))
    check("no_fitting_runtime_imports", all(name not in sys.modules for name in
          ("calibration_core", "run_calibration", "batch_profile", "run_comparison")))
    result = dict(passed=all(r["passed"] for r in checks), test_count=len(checks), checks=checks,
        source_hashes=marker["source_hashes"], resource_policy=marker["resource_policy"],
        inherited_pure_qa_sha256=ref.sha(HERE/"qa/chunked_execution_contract_test.json"),
        baseline_contract_sha256=ref.sha(BASE/"contract.json"),
        scope="Pure resource-limit, combined-scope, source and runtime-function identity checks; no fits or random draws")
    destination = HERE/"qa/chunked_resource_contract_test.json"
    ref.write_json(destination, result)
    print(ref.encoded(dict(passed=result["passed"], test_count=result["test_count"], path=str(destination))))


def verify_resource_qa(marker):
    path = HERE/"qa/chunked_resource_contract_test.json"
    qa = ref.read_json(path)
    if (qa.get("passed") is not True or qa.get("test_count") != 26
            or len(qa.get("checks", [])) != 26 or not all(r.get("passed") is True for r in qa["checks"])
            or qa.get("source_hashes") != marker["source_hashes"]
            or qa.get("resource_policy") != marker["resource_policy"]
            or qa.get("inherited_pure_qa_sha256") != ref.sha(HERE/"qa/chunked_execution_contract_test.json")
            or qa.get("baseline_contract_sha256") != ref.sha(BASE/"contract.json")):
        raise RuntimeError("Current resource-wrapper pure QA is missing or has changed")
    return path


def main():
    """Plain base dispatch; changes are combined scope, resources and source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-check", action="store_true", help="Pure checks only; no native reconstruction, fits or draws")
    parser.add_argument("--plan-only", action="store_true", help="Native observed reconstruction/planning only; no audit or calibration toys")
    parser.add_argument("--scope", choices=("all", COMBINED), default="all")
    parser.add_argument("--masses", help="Comma-separated integers or inclusive ranges")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--previous-input", type=Path, action="append", default=[])
    parser.add_argument("--skip-completed-input", type=Path, action="append", default=[],
                        help="Skip already completed sampling coordinates, irrespective of their outcomes")
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-minutes", type=float, default=60.)
    parser.add_argument("--max-spectra", type=int, default=1500000)
    parser.add_argument("--max-memory-gib", type=float, default=6.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pure_check:
        pure_checks()
        return
    if (not 1 <= args.batch_size <= 24 or args.batch_index < 1
            or not math.isfinite(args.max_minutes) or args.max_minutes <= 0
            or (args.max_minutes > 60 and args.batch_size != 1)
            or not 0 < args.max_spectra <= 1500000):
        parser.error("At most 24 coordinates/1.5M spectra; scheduling slices above 60 minutes require --batch-size 1")
    try:
        scope_filter = validate_resource_limits(args.max_memory_gib, args.scope)
    except ValueError as error:
        parser.error(str(error))
    if (args.attempt == 2) != bool(args.previous_input):
        parser.error("Attempt2 requires --previous-input; attempt1 uses the original baseline")
    layout = layout_marker(args.max_memory_gib)
    resource_qa = verify_resource_qa(layout)
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
    requested = args.output or Path("chunked_resource_v1")/f"attempt{args.attempt}_batch{args.batch_index:03}"
    out = (requested if requested.is_absolute() else HERE/requested).resolve()
    if not out.is_relative_to(HERE) or out == HERE or out == BASE or BASE in out.parents:
        parser.error("Use a separate output tree inside the calibration study")
    input_directories = [p.resolve() for p in [*args.previous_input, *args.skip_completed_input]]
    if any(out == p or out in p.parents or p in out.parents for p in input_directories):
        parser.error("Output must be separate from all prior input directories")
    if any(out.glob("*/m*/FAILURE.txt")):
        parser.error("A failure exists in this output tree; use a new output directory")
    selection_path = out/"selection.json"
    configuration = dict(attempt=args.attempt, scope=scope_filter,
        masses=sorted(mass_filter) if mass_filter is not None else None,
        batch_index=args.batch_index, batch_size=args.batch_size, max_minutes=args.max_minutes,
        max_spectra=args.max_spectra, max_memory_gib=args.max_memory_gib, output_directory=str(out),
        previous_input_directories=[str(p.resolve()) for p in args.previous_input],
        skip_completed_input_directories=[str(p.resolve()) for p in args.skip_completed_input])
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
        filtered = [e for e in eligible if e["scope_key"] == scope_filter
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
        selected = selection["selected"]
    if any(e["scope_key"] != COMBINED for e in selected):
        raise RuntimeError("Frozen resource selection contains a non-combined coordinate")
    ref.write_json(selection_path, selection, freeze=True)
    extra = [Path(__file__).resolve(), HERE/"CHUNKED_RESOURCE_PROTOCOL.md",
             Path(base.__file__).resolve(), HERE/"CHUNKED_REFINEMENT_PROTOCOL.md", selection_path,
             Path(ref.__file__).resolve(), HERE/"REFINEMENT_PROTOCOL.md", resource_qa,
             HERE/"qa/chunked_execution_contract_test.json", HERE/"provenance/additional_runtime_hashes.json"]
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
    # No change to the inherited numerical/runtime calls below this point.
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
