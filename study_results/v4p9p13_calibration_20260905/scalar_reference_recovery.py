#!/usr/bin/env python3
"""Isolated initialization recovery for the combined m074 scalar reference."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import math
import sys
import textwrap

sys.dont_write_bytecode = True
import run_chunked_refinement as base
import run_chunked_refinement_8gib as resource

HERE, ROOT, ref, np = base.HERE, base.ROOT, base.ref, base.np
SCOPE, MASS, ATTEMPT = "all_2015_2016_2021", 74, 2
TYPE, VERSION = "zero_nuisance_scalar_reference_initializer", 1
SCORE_GATE, R_GATE, Q_GATE = 2e-7, 2e-5, 1e-4
BACKEND, NUISANCE_CUT = "eigenfeature_rtol_1e-15", 1e-5
FAILED_ROOT = HERE/"chunked_resource8_v1/attempt2_combined"
FAILED_POINT = FAILED_ROOT/SCOPE/"m074"
KNOWN_WINDOW_SHA = "0839681a696cfdc820017fee4a3d378ecf64ba598dba8b163dc0f38f418a84d3"
KNOWN_CHUNK_SHA = "941b988f24d9d8553c47b039b6c2124ac1c5266062f3caef938f102efaefc397"
SOURCE_NAMES = ("scalar_reference_recovery.py", "run_scalar_reference_recovery.py",
                "diagnose_scalar_reference_recovery.py", "SCALAR_REFERENCE_RECOVERY_PROTOCOL.md")
PINNED = {
 "calibration_core.py":"05c6a6c65bbc0c3e23fa643c74d931f07b92d9ab80993138cee3d330d64db46a",
 "batch_profile.py":"f598f479f7afc07240fa6550b772208f03f8861035c2c02d69745a4bf5974fba",
 "../background_profile_comparison_20260905/run_comparison.py":"ec2b0c0883c8272c65c5716c13806dc29ab2399cea4b1c1a05306562a47ab87a",
 "chunked_resource8_v1/attempt2_combined/contract.json":"841071823aa317420dc82fd9bd9e2a051caf077a9bdbac8eef1085d9e5d6cfaa",
 "chunked_resource8_v1/attempt2_combined/selection.json":"1660aada1adc025b89b561bbe33d46019372e7a89960281c1fd80d576e6be095",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/FAILURE.txt":"1c74b4f4797cb6c21598c15709ff240fcc10106bac5f03191b83b66b4151a1a5",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/failure_numerical_qa.json":"e17a1eb730c52a48961200fbf8e12039b92cb8b949bb25258eea9fb2d47239af",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/point_plan.json":"03e66774b36c4f382c21858316aa7dbf3b3799753826db1b2131fe22b057c66c",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/proposal_plan.json":"7cb607a958ec0225c6cf74158db3485153d2d84fce18edc1661f3e7698f8d22c",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/pre_generation_numerical_qa.json":"56c5018c97deb9011828a075e883af8b79f3106ef41152ccb9a6e2bdb7d49a30",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/chunked_equivalence_checks.json":"da361e1a1bafc30c2003a139b47d0f991665e9589d9f04229123adeae7185530",
 "chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/memory_check.json":"23d7bcd7c297251c48df341ff299eb6e2279d2ead0c6be7bc4ccda0d17fcf69c",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_coordinate(scope, mass, attempt):
    require((scope, mass, attempt) == (SCOPE, MASS, ATTEMPT),
            "Scalar reference recovery is restricted to combined m074 attempt2")


def frozen_inputs():
    paths = {(HERE/name).resolve(): expected for name, expected in PINNED.items()}
    require(all(ref.sha(p) == h for p, h in paths.items()), "Frozen failure/runtime identity changed")
    contract = ref.read_json(FAILED_ROOT/"contract.json")
    ref.check_hashes(contract)
    for name, expected in {**contract["hashes"], **contract["sampling_hashes"]}.items():
        paths[(ROOT/name).resolve()] = expected
    return paths


def failure_data():
    return ref.read_json(FAILED_POINT/"failure_numerical_qa.json")


def source_equivalence():
    """AST comparison only; no generated or rewritten code is ever executed."""
    original = (HERE/"calibration_core.py").read_text()
    context = next(n for n in ast.parse(original).body if isinstance(n, ast.ClassDef) and n.name == "Context")
    node = next(n for n in context.body if isinstance(n, ast.FunctionDef) and n.name == "make_models")
    original_method = ast.get_source_segment(original, node)
    derivative = Path(__file__).read_text()
    copies = [n for n in ast.walk(ast.parse(derivative)) if isinstance(n, ast.FunctionDef) and n.name == "copied_make_models"]
    require(len(copies) == 1, "Reference make_models copy is ambiguous")
    copied = ast.get_source_segment(derivative, copies[0])
    factory = "self.reference_profile(model.b[i],model.L[i],self.w,'linear',whole[i],method,batch_id,i)"
    require(copied.count(factory) == 1, "Reference factory change is missing or duplicated")
    normalized = copied.replace("def copied_make_models(", "def make_models(", 1).replace(
        factory, "c.Profile(model.b[i],model.L[i],self.w,'linear')", 1)
    old_ast = ast.dump(ast.parse(textwrap.dedent(original_method)), include_attributes=False)
    new_ast = ast.dump(ast.parse(textwrap.dedent(normalized)), include_attributes=False)
    require(old_ast == new_ast, "make_models changed beyond the scalar reference factory")
    return dict(passed=True, original_source_sha256=ref.sha(HERE/"calibration_core.py"),
                normalized_ast_sha256=hashlib.sha256(old_ast.encode()).hexdigest(),
                permitted_change="Only c.Profile construction becomes self.reference_profile")


def recovery_marker():
    return dict(type=TYPE, version=VERSION, scope_key=SCOPE, mass_MeV=MASS, attempt=ATTEMPT,
        source_hashes={str((HERE/n).relative_to(ROOT)): ref.sha(HERE/n) for n in SOURCE_NAMES},
        original_failure_sha256=ref.sha(FAILED_POINT/"FAILURE.txt"),
        original_failure_ledger_sha256=ref.sha(FAILED_POINT/"failure_numerical_qa.json"),
        original_attempt_contract_sha256=ref.sha(FAILED_ROOT/"contract.json"),
        policy="Original scalar fit first; free linear zero-nuisance convergence failure only; Brent score root then unchanged fit(initial=root)",
        score_gate=SCORE_GATE, r_gate=R_GATE, q_gate=Q_GATE,
        batch_statistic_unchanged=True, science_banks_identical=True, validation_seeds_unchanged=True,
        original_audit_draws_unchanged=True, extended_audit_draws="Fresh source/plan identity; numerical checks only",
        gp_backend=BACKEND, nuisance_eigenvalue_cut=NUISANCE_CUT)


def eligible_failure(mode, npar, fixed, initial, error):
    if mode != "linear" or npar != 0 or fixed is not None or initial is not None:
        return False
    if not isinstance(error, RuntimeError) or not str(error).startswith("Unconverged fit, score="):
        return False
    try:
        score = float(str(error).split("score=", 1)[1])
    except ValueError:
        return False
    return math.isfinite(score) and score >= SCORE_GATE


def monotone_initializer(profile, n):
    """Root the original scaled score; no substitute objective or acceptance gate."""
    from scipy.optimize import brentq
    n = np.asarray(n, float)
    b, direction = profile.b, profile.scale*profile.w
    require(profile.mode == "linear" and profile.npar == 0 and np.all(b > 0)
            and np.all(direction >= 0) and np.any(direction > 0) and np.all(n >= 0)
            and np.isfinite(n).all(), "Invalid monotone scalar model")
    history = []
    def score(z):
        value, gradient, hessian, _, lam = profile.objective(np.array([z]), n, None)
        require(np.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(hessian).all()
                and np.all(lam > 0) and hessian[0, 0] > 0,
                "Scalar initializer lacks positive expectations/strict score monotonicity")
        history.append(dict(z=float(z), score=float(gradient[0]), curvature=float(hessian[0, 0]),
                            min_lambda=float(lam.min())))
        return float(gradient[0])
    at_zero = score(0.)
    lower = float(np.max(-b[direction > 0]/direction[direction > 0]))
    if at_zero > 0:
        high, high_score = 0., at_zero
        for iteration in range(1, 61):
            low = lower+(0.-lower)*2.**(-iteration)
            require(low > lower, "No representable positive-expectation lower bracket")
            low_score = score(low)
            if low_score <= 0:
                break
        else:
            raise RuntimeError("Scalar initializer could not bracket an interior minimum")
    elif at_zero < 0:
        low, low_score = 0., at_zero
        for iteration in range(61):
            high = 2.**iteration
            high_score = score(high)
            if high_score >= 0:
                break
        else:
            raise RuntimeError("Scalar initializer could not bracket an upper score root")
    else:
        low = high = 0.; low_score = high_score = 0.
    require(low_score <= 0 <= high_score, "Scalar root bracket signs differ")
    xtol, rtol = 5e-15, 4*np.finfo(float).eps
    if low == high:
        root, iterations, calls = low, 0, 1
    else:
        root, solution = brentq(score, low, high, xtol=xtol, rtol=rtol,
                               maxiter=128, full_output=True, disp=True)
        require(solution.converged, "Brent scalar initializer failed")
        iterations, calls = solution.iterations, solution.function_calls
    final_score = score(root)
    require(abs(final_score) < SCORE_GATE, "Root initializer does not meet the original score gate")
    return float(root), dict(bracket_z=[low, high], bracket_score=[low_score, high_score],
        domain_lower_z=lower, root_z=float(root), root_A=float(root*profile.scale),
        root_score=final_score, xtol=xtol, rtol=float(rtol), iterations=int(iterations),
        function_calls=int(calls), evaluations=history, passed=True)


def serial(value):
    if isinstance(value, dict):return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):return [serial(v) for v in value]
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    return value


class ReferenceAdapter:
    def __init__(self, profile, identity, ledger):
        self.profile, self.identity, self.ledger = profile, identity, ledger

    def fit(self, n, fixed=None, initial=None):
        try:
            return self.profile.fit(n, fixed=fixed, initial=initial)
        except Exception as error:
            if self.identity["method"] != "fixed" or not eligible_failure(self.profile.mode, self.profile.npar, fixed, initial, error):
                raise
            event = dict(**self.identity, mode=self.profile.mode, npar=self.profile.npar,
                fixed_value=fixed, initial_supplied=initial is not None,
                original_exception=dict(type=type(error).__name__, message=str(error)),
                original_score=float(str(error).split("score=", 1)[1]), passed=False)
            self.ledger.append(event)
            try:
                root, event["initializer"] = monotone_initializer(self.profile, n)
                fit = self.profile.fit(n, initial=np.array([root]))
                event["final_original_fit"] = serial(fit)
                require(np.isfinite(fit["score"]) and fit["score"] < SCORE_GATE and fit["min_lambda"] > 0,
                        "Original fit restart failed its unchanged score/positivity gate")
                event["restart_passed"] = True
                return fit
            except Exception as recovery_error:
                event["recovery_error"] = dict(type=type(recovery_error).__name__, message=str(recovery_error))
                raise


def check_backend(ctx):
    require(ctx.gp_backend == BACKEND and ctx.nuisance_cut == NUISANCE_CUT,
            "Accepted backend/covariance convention differs from the failed run")


def make_core_facade(core):
    """A new ordinary namespace/subclass; never mutate the original module."""
    c, np, BatchProfile = core.c, core.np, core.BatchProfile
    def copied_make_models(self,whole):
     b=[];L=[]
     for n in whole:
      bb,ll=self.retrain(n);b.append(bb);L.append(ll)
     b=np.array(b);rank=max(ll.shape[1] for ll in L);L=np.array([np.pad(ll,((0,0),(0,rank-ll.shape[1]))) for ll in L]);counts=whole[:,self.mask]
     active=np.any(L!=0.,axis=(0,1));L=L[:,:,active]
     blocks=[];rr=0;cc=0;original_col=0
     for part in self.parts:
      rows=int(part['p'].blind_mask.sum());cols=min(12,rows) if self.nuisance_cut else rows
      kept=int(active[original_col:original_col+cols].sum());original_col+=cols
      blocks.append((rr,rr+rows,cc,cc+kept));rr+=rows;cc+=kept
     models={method:BatchProfile(counts,b,L if method=='profiled' else np.zeros((len(b),len(self.b),0)),self.w,blocks if method=='profiled' else None) for method in ('profiled','fixed')}
     batch_id=self.scalar_check_batches;self.scalar_check_batches+=1
     for method,model in models.items():
      for i in range(min(2,len(b))):
       scalar=self.reference_profile(model.b[i],model.L[i],self.w,'linear',whole[i],method,batch_id,i);f=scalar.fit(counts[i]);z=scalar.fit(counts[i],0.)
       r=np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll'])))
       check=dict(batch_id=batch_id,n_spectra=len(b),method=method,toy_index=i,
        counts_sha256=hashlib.sha256(counts[i].tobytes()).hexdigest(),scalar_r=float(r),batch_r=float(model.r[i]),
        r_error=float(abs(r-model.r[i])),q_checks=[],passed=False)
       self.scalar_checks.append(check)
       if check['r_error']>2e-5:raise RuntimeError('Batch/scalar signed-r disagreement')
       for a in (2,5):
        fixed=scalar.fit(counts[i],a*self.sigma);q=0. if f['A']>a*self.sigma else max(0.,2*(fixed['nll']-(f['nll'] if f['A']>=0 else z['nll'])))
        # Check an isolated two-row batch to avoid extra fits of the complete bank.
        tiny=BatchProfile(counts[i:i+1],b[i:i+1],model.L[i:i+1],self.w,blocks if method=='profiled' else None)
        batch_q=float(tiny.q(a*self.sigma)[0]);error=float(abs(q-batch_q))
        check['q_checks'].append(dict(strength_sigma=a,scalar_q=float(q),batch_q=batch_q,q_error=error))
        if error>1e-4:raise RuntimeError('Batch/scalar q disagreement')
       check['passed']=True
     return models

    class RecoveryContext(core.Context):
        make_models = copied_make_models
        def __init__(self, scope, mass, *args, **kwargs):
            validate_coordinate(scope[0], mass, ATTEMPT)
            self.scalar_reference_fallbacks = []
            super().__init__(scope, mass, *args, **kwargs)

        def reference_profile(self, b, L, w, mode, whole, method, batch_id, toy_index):
            check_backend(self)
            identity = dict(batch_id=batch_id, toy_index=toy_index, method=method,
                purpose=getattr(self, "execution_call_kind", "unspecified"),
                full_counts_sha256=base.array_sha(np.ascontiguousarray(whole)),
                window_counts_sha256=base.array_sha(np.ascontiguousarray(whole[self.mask])),
                count_dtype=str(whole.dtype), background_sha256=base.array_sha(np.ascontiguousarray(b)),
                template_sha256=base.array_sha(np.ascontiguousarray(w)),
                gp_backend=self.gp_backend, nuisance_eigenvalue_cut=self.nuisance_cut)
            ledger = getattr(self, "model_chunk_ledger", [])
            if ledger and ledger[-1]["passed"] is False and ledger[-1]["chunks"]:
                identity.update(execution_call_id=ledger[-1]["call_id"],
                    chunk_start=ledger[-1]["chunks"][-1]["start"],
                    global_toy_index=ledger[-1]["chunks"][-1]["start"]+toy_index)
            return ReferenceAdapter(c.Profile(b, L, w, mode), identity, self.scalar_reference_fallbacks)

    facade = SimpleNamespace(**vars(core))
    facade.Context = RecoveryContext
    require(all(getattr(facade, k) is v for k, v in vars(core).items() if k != "Context"),
            "Core facade changes more than its Context")
    return facade


def runtime_types(core):
    facade = make_core_facade(core)
    Context, OriginalBank = base.runtime_types(facade)
    expected = failure_data()["chunked_equivalence_checks"]["generation_checks"]
    proposal_plan = ref.read_json(FAILED_POINT/"proposal_plan.json")
    class ClosureBank(OriginalBank):
        def __init__(self, ctx, truth, whole, proposals, strata, *, qcache_limit):
            check_backend(ctx)
            names = [t for t in base.TRUTHS if np.array_equal(truth, ctx.truths[t])]
            require(len(names) == 1, "Ambiguous science-bank truth")
            name = names[0]
            require(base.array_sha(proposals) == proposal_plan["proposals"][name]["sha256"]
                    and base.array_sha(whole) == expected[name]["whole_sha256"],
                    "Science proposal/whole-bank identity differs from the failed run")
            super().__init__(ctx, truth, whole, proposals, strata, qcache_limit=qcache_limit)
    return facade, Context, ClosureBank


def recovery_audit(ctx, require_known=True):
    check_backend(ctx)
    for event in ctx.scalar_reference_fallbacks:
        matches = [r for r in ctx.scalar_checks if (r["batch_id"], r["method"], r["toy_index"], r["counts_sha256"]) ==
                   (event["batch_id"], event["method"], event["toy_index"], event["window_counts_sha256"])]
        require(len(matches) == 1 and event.get("restart_passed") is True, "Recovery lacks its scalar agreement row")
        check = matches[0]
        require(check["passed"] is True and check["r_error"] <= R_GATE
                and len(check["q_checks"]) == 2 and all(q["q_error"] <= Q_GATE for q in check["q_checks"]),
                "Recovered reference failed unchanged statistic agreement")
        event.update(scalar_check=check, passed=True)
    known = [r for r in ctx.scalar_reference_fallbacks if r["window_counts_sha256"] == KNOWN_WINDOW_SHA]
    require(not require_known or len(known) == 1, "Known failed spectrum was not recovered exactly once")
    return dict(type=TYPE, version=VERSION, passed=True, fallback_count=len(ctx.scalar_reference_fallbacks),
                known_failure_reproduced=len(known) == 1, fallbacks=ctx.scalar_reference_fallbacks,
                source_equivalence=source_equivalence(), source_hashes=recovery_marker()["source_hashes"],
                original_failure_ledger_sha256=ref.sha(FAILED_POINT/"failure_numerical_qa.json"))


def load_context(core, Context):
    cfg = core.c.production.load_config(core.c.production.DEFAULT_CARD)
    core.c.production.validate_card(cfg); core.c.production.validate_histogram_inputs(cfg)
    core.c.production.validate_input_provenance(core.c.production.DEFAULT_INPUT_PROVENANCE,
        core.c.production.DEFAULT_CARD, core.c.production.DEFAULT_STATES, cfg)
    datasets = core.c.production.make_datasets(cfg)
    states = core.c.production.state_map(core.pd.read_csv(core.c.production.DEFAULT_STATES))
    scope = next(s for s in core.SCOPES if s[0] == SCOPE)
    return Context(scope, MASS, cfg, datasets, states)


def verify_recovery_layout(layout, data=None):
    marker = recovery_marker()
    require(layout.get("scalar_reference_recovery") == marker, "Recovery layout/source policy differs")
    inherited = {k:v for k,v in layout.items() if k != "scalar_reference_recovery"}
    require(inherited == resource.layout_marker(8.), "Inherited 8 GiB execution identity changed")
    paths = set(frozen_inputs()) | {HERE/n for n in SOURCE_NAMES}
    paths.update((resource.verify_resource_qa(inherited), HERE/"qa/chunked_execution_contract_test.json"))
    qa_path = HERE/"qa/scalar_reference_recovery_contract_test.json"
    qa = ref.read_json(qa_path)
    require(qa.get("passed") is True and len(qa["checks"]) == qa["test_count"]
            and len({r["name"] for r in qa["checks"]}) == qa["test_count"]
            and all(r["passed"] is True for r in qa["checks"])
            and qa["source_hashes"] == marker["source_hashes"]
            and qa["source_equivalence"] == source_equivalence(), "Recovery pure QA is missing or stale")
    paths.add(qa_path)
    if data is not None:
        validate_coordinate(data["scope_key"], data["mass_MeV"], data["sampling_refinement"]["attempt"])
        require(data["gp_backend"] == BACKEND and data["nuisance_eigenvalue_cut"] == NUISANCE_CUT
                and data["nvalidation"] == 500, "Recovered result changes inference conventions")
        expected = failure_data()["chunked_equivalence_checks"]["generation_checks"]
        require(all(data["provenance"][t]["whole_sha256"] == expected[t]["whole_sha256"]
                    and data["provenance"][t]["seed_namespace"] == expected[t]["seed_namespace"] for t in base.TRUTHS),
                "Recovered result changes science banks/seeds")
        record = data["scalar_reference_recovery"]
        ledger_path = Path(record["ledger_path"]).resolve()
        require(ledger_path.is_relative_to(HERE) and ref.sha(ledger_path) == record["ledger_sha256"], "Recovery ledger identity differs")
        ledger = ref.read_json(ledger_path)
        require(ledger["passed"] is True and ledger["known_failure_reproduced"] is True
                and ledger["fallback_count"] == len(ledger["fallbacks"]) > 0
                and ledger["source_hashes"] == marker["source_hashes"]
                and ledger["source_equivalence"] == source_equivalence(), "Recovery ledger is incomplete")
        for event in ledger["fallbacks"]:
            original = RuntimeError(event["original_exception"]["message"])
            require(event["method"] == "fixed" and event["original_exception"]["type"] == "RuntimeError"
                    and eligible_failure(event["mode"], event["npar"], event["fixed_value"],
                                         [] if event["initial_supplied"] else None, original)
                    and event["passed"] is True and event["restart_passed"] is True
                    and event["final_original_fit"]["score"] < SCORE_GATE
                    and event["final_original_fit"]["min_lambda"] > 0, "Ineligible or failed reference recovery")
            initializer, check = event["initializer"], event["scalar_check"]
            require(initializer["passed"] is True and abs(initializer["root_score"]) < SCORE_GATE
                    and initializer["bracket_score"][0] <= 0 <= initializer["bracket_score"][1]
                    and all(math.isfinite(r["score"]) and r["curvature"] > 0 and r["min_lambda"] > 0
                            for r in initializer["evaluations"]), "Invalid monotone root record")
            require(check in data["scalar_checks"] and check["passed"] is True
                    and abs(check["scalar_r"]-check["batch_r"]) <= R_GATE
                    and [q["strength_sigma"] for q in check["q_checks"]] == [2,5]
                    and all(abs(q["scalar_q"]-q["batch_q"]) <= Q_GATE for q in check["q_checks"]),
                    "Recovered reference/statistic agreement differs")
        paths.add(ledger_path)
        paths.update(verify_diagnostic(Path(record["diagnostic_directory"]), layout))
    return sorted(paths)


def verify_diagnostic(directory, layout):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(HERE/"scalar_reference_recovery_v1")
            and not (directory/"FAILURE.txt").exists(), "Invalid/failed recovery diagnostic directory")
    summary_path = directory/"summary.json"
    summary = ref.read_json(summary_path)
    require(summary["status"] == "completed_replay_diagnostic" and summary["passed"] is True
            and summary["execution_layout"] == layout and summary["rng_calls"] == 10
            and summary["chunk_sha256"] == KNOWN_CHUNK_SHA and summary["window_sha256"] == KNOWN_WINDOW_SHA
            and summary["gp_backend"] == BACKEND and summary["nuisance_eigenvalue_cut"] == NUISANCE_CUT,
            "Diagnostic does not reproduce the frozen failure")
    paths = {summary_path}
    for name, expected in summary["output_sha256"].items():
        require(Path(name).name == name, "Diagnostic output leaves selected directory")
        path = directory/name
        require(ref.sha(path) == expected, "Diagnostic output changed")
        paths.add(path)
    return sorted(paths)


def pure_checks():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        require(condition, name)
    check("frozen_failure_and_runtime_hashes", bool(frozen_inputs()))
    check("make_models_only_reference_factory_changed", source_equivalence()["passed"])
    inherited = resource.layout_marker(8.)
    check("inherited_resource8_qa_current", resource.verify_resource_qa(inherited).is_file())
    check("original47_inference_sources", len(ref.read_json(FAILED_ROOT/"contract.json")["hashes"]) == 47)
    validate_coordinate(SCOPE, MASS, ATTEMPT); check("accept_combined74_attempt2", True)
    for name, args in (("mass", (SCOPE,73,2)), ("scope", ("individual_2016_full",74,2)), ("attempt", (SCOPE,74,1))):
        try:validate_coordinate(*args); rejected=False
        except RuntimeError:rejected=True
        check("reject_other_"+name, rejected)
    error = RuntimeError("Unconverged fit, score=4.4949521033066375e-07")
    check("accept_original_fixed_free_failure", eligible_failure("linear",0,None,None,error))
    cases = [("nuisance", "linear",1,None,None,error), ("log", "log",0,None,None,error),
             ("fixed_A", "linear",0,0.,None,error), ("initial", "linear",0,None,[0.],error),
             ("line_search", "linear",0,None,None,RuntimeError("Line search failed, score=1")),
             ("exception_type", "linear",0,None,None,ValueError(str(error))),
             ("already_converged", "linear",0,None,None,RuntimeError("Unconverged fit, score=1e-8")),
             ("nan_score", "linear",0,None,None,RuntimeError("Unconverged fit, score=nan"))]
    for name,*args in cases:check("reject_"+name, not eligible_failure(*args))
    failure = failure_data(); call = failure["model_chunk_ledger"][-1]; chunk = call["chunks"][-1]
    check("known_chunk_identity", call["purpose"] == "calibration_stress" and (chunk["start"],chunk["stop"],chunk["full_counts_sha256"]) == (9728,9856,KNOWN_CHUNK_SHA))
    check("known_window_and_order", len(chunk["scalar_check_indices"]) == 2 and failure["scalar_checks"][-2]["counts_sha256"] == KNOWN_WINDOW_SHA
          and all(failure["scalar_checks"][i]["passed"] for i in chunk["scalar_check_indices"]))
    gen = failure["chunked_equivalence_checks"]["generation_checks"]
    check("both_original_whole_bank_hashes", set(gen) == set(base.TRUTHS) and all(len(r["whole_sha256"]) == 64 and r["passed"] for r in gen.values()))
    check("original_stress_rng_shape", gen["stress"]["rng_call_shape"] == [1024,1626] and gen["stress"]["rng_calls"] == 102)
    fake = SimpleNamespace(Context=type("FakeContext",(),{}), c=object(), np=object(), BatchProfile=object(), Bank=object(), seed=object())
    facade = make_core_facade(fake)
    check("ordinary_context_subclass", facade.Context.__mro__[1] is fake.Context)
    check("facade_other_attributes_identical", all(getattr(facade,k) is v for k,v in vars(fake).items() if k != "Context"))
    check("unchanged_inherited_statistic", resource.run_point is base.run_point and resource.runtime_types is base.runtime_types)
    check("four_new_source_identities", len(recovery_marker()["source_hashes"]) == 4)
    check("no_fitting_runtime_imports", not any(n in sys.modules for n in ("calibration_core","run_calibration","batch_profile","run_comparison")))
    qa = dict(passed=True, test_count=len(checks), checks=checks, source_hashes=recovery_marker()["source_hashes"],
              source_equivalence=source_equivalence(), frozen_input_sha256={str(p.relative_to(ROOT)):h for p,h in frozen_inputs().items()},
              scope="Pure policy, source-copy and identity checks only; no fits or random draws")
    path = HERE/"qa/scalar_reference_recovery_contract_test.json"
    ref.write_json(path, qa)
    print(ref.encoded(dict(passed=True, test_count=len(checks), path=str(path))))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-check", action="store_true", required=True)
    parser.parse_args();pure_checks()
