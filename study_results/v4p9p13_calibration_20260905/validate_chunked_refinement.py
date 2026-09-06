#!/usr/bin/env python3
"""Pure array/interface/source checks. No fitting imports or random draws."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
import run_chunked_refinement as driver

np, ref = driver.np, driver.ref
HERE, ROOT = driver.HERE, driver.ROOT


def main():
    checks = []

    def check(name, condition, **details):
        checks.append(dict(name=name, passed=bool(condition), **details))
        if not condition:
            raise AssertionError(name)

    counts = np.arange(257*17, dtype=np.int64).reshape(257, 17)+1000000
    check("zero_copy_sha_matches_original_bytes",
          driver.array_sha(counts) == hashlib.sha256(counts.tobytes()).hexdigest())
    rejected = False
    try:
        driver.array_sha(counts[:, ::2])
    except ValueError:
        rejected = True
    check("noncontiguous_hash_rejected", rejected)

    class DeterministicRecipe:
        """An integer fixture stream; deliberately no Poisson/random sampling."""
        def __init__(self):
            self.calls = []

        def poisson(self, mean, size):
            self.calls.append(dict(mean=mean.copy(), shape=size))
            return np.arange(np.prod(size), dtype=np.int64).reshape(size)+100*len(self.calls)

    proposals = np.array([[10., 11., 12.], [20., 21., 22.], [30., 31., 32.]])
    original_rng, new_rng = DeterministicRecipe(), DeterministicRecipe()
    original = np.concatenate([original_rng.poisson(mean, size=(7, len(mean))) for mean in proposals])
    whole, closure = driver.generate_whole(new_rng, proposals, 7)
    check("preallocated_concatenation_and_whole_sha", np.array_equal(whole, original)
          and closure["whole_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
          and closure["whole_sha256"] == closure["concatenated_proposal_bytes_sha256"])
    check("original_per_proposal_calls_shapes_order", len(new_rng.calls) == len(original_rng.calls)
          and all(a["shape"] == b["shape"] and np.array_equal(a["mean"], b["mean"])
                  for a, b in zip(new_rng.calls, original_rng.calls)))

    truth = np.linspace(1000000., 1010000., 17)
    signal = np.linspace(.02, .09, 17)
    means = np.array([truth+a*signal+shift*np.cos(np.arange(17))
                      for a in (0., 20., 100.) for shift in (-10., 0., 10.)])
    reference = driver.logsumexp(counts@np.log(means/truth).T-np.sum(means-truth, axis=1), axis=1)-np.log(len(means))
    density_errors, weight_errors = {}, []
    for size in (1, 2, 128):
        actual = driver.blocked_logmix(counts, truth, means, size)
        density_errors[str(size)] = float(np.max(abs(reference-actual)))
        for a in (0., 2., 5., 12., 20.):
            delta = a*100.*signal
            old = np.exp(counts@np.log1p(delta/truth)-np.sum(delta)-reference)
            new = driver.blocked_weights(counts, truth, signal, 100., a, actual, size)
            weight_errors.append(dict(chunk_size=size, strength=a,
                                      **driver.weight_comparison(old, new)))
    check("blocked_density_original_full_expression", max(density_errors.values()) <= driver.LOG_DENSITY_GATE,
          max_abs_errors=density_errors)
    check("blocked_weight_original_full_expression", all(r["passed"] for r in weight_errors), rows=weight_errors)
    reference_weight = np.array([0., 1., 1e6])
    allowance = driver.WEIGHT_ATOL+driver.WEIGHT_RTOL*abs(reference_weight)
    allowed = driver.weight_comparison(reference_weight, reference_weight+.5*allowance)
    check("numeric_weight_gate_accepts_finite_within_bound", allowed["passed"]
          and allowed["finite"] and 0. < allowed["max_scaled_error"] <= 1.)
    excessive = driver.weight_comparison(np.array([0.]), np.array([2*driver.WEIGHT_ATOL]))
    check("numeric_weight_gate_rejects_excess", not excessive["passed"]
          and excessive["finite"] and excessive["max_scaled_error"] == 2.)
    bad = driver.weight_comparison(np.array([1.]), np.array([np.nan]))
    check("numeric_weight_gate_rejects_nonfinite", not bad["passed"] and not bad["finite"])

    class FakeModel:
        def __init__(self, n, method):
            self.n, self.b = n, n.astype(float)+1.
            self.npar = 2 if method == "profiled" else 0
            self.L = np.ones((len(n), n.shape[1], self.npar))
            self.blocks = [(0, n.shape[1], 0, self.npar)]
            self.r = n[:, 0].astype(float)
            self.free = {"A": n[:, 1].astype(float)}
            self.max_score, self.fallbacks, self.q_calls = 0., 0, 0

        def q(self, A):
            self.q_calls += 1
            self.max_score += 1e-12
            self.fallbacks += 1
            return np.maximum(0., A-self.free["A"])**2

    class FakeContext:
        def __init__(self):
            self.scalar_checks, self.scalar_check_batches = [], 0
            self.signal, self.sigma = np.ones(3), 1.

        def make_models(self, whole):
            models = {m: FakeModel(whole, m) for m in driver.METHODS}
            for method in driver.METHODS:
                for i in range(min(2, len(whole))):
                    self.scalar_checks.append(dict(toy_index=i, method=method, passed=True))
            return models

    class FakeBank:
        pass

    Context, Bank = driver.runtime_types(SimpleNamespace(Context=FakeContext, Bank=FakeBank))
    ctx = Context()
    fixture = np.arange(257*3, dtype=np.int64).reshape(257, 3)
    models = ctx.make_models(fixture)
    ledger = ctx.model_chunk_ledger[0]
    check("production_chunk_boundaries", [(r["start"], r["stop"]) for r in ledger["chunks"]]
          == [(0, 128), (128, 256), (256, 257)])
    expected_indices = [0, 1, 0, 1, 128, 129, 128, 129, 256, 256]
    check("scalar_global_row_mapping", [r["global_toy_index"] for r in ctx.scalar_checks] == expected_indices)
    for method, aggregate in models.items():
        direct = FakeModel(fixture, method)
        q = aggregate.q(999.)
        check("aggregate_interface_"+method, np.array_equal(q, direct.q(999.))
              and np.array_equal(aggregate.r, direct.r)
              and np.array_equal(aggregate.free["A"], direct.free["A"])
              and all(np.array_equal(aggregate.b[i], direct.b[i])
                      and np.array_equal(aggregate.L[i], direct.L[i]) for i in (0, 127, 128, 255, 256, -1)))
    check("aggregate_dynamic_diagnostics", models["profiled"].fallbacks == 3
          and models["profiled"].max_score == 1e-12)
    bank = Bank(ctx, np.ones(3), np.ones((4, 3), dtype=np.int64), np.ones((2, 3)),
                np.array([0, 0, 1, 1]), qcache_limit=1)
    first = bank.q("fixed", 2.)
    check("cached_q_reused", bank.q("fixed", 2.) is first and bank.qcache_peak == 1)
    rejected = False
    try:
        bank.q("fixed", 5.)
    except RuntimeError:
        rejected = True
    check("cache_cap_fails_without_eviction", rejected and len(bank.qcache) == 1
          and bank.q("fixed", 2.) is first)

    memory = {}
    for scope, mass in (("all_2015_2016_2021", 74), ("individual_2016_full", 75)):
        path = HERE/"refined_v1/attempt1_batch001"/scope/f"m{mass:03}"/"point_plan.json"
        plan = json.loads(path.read_text())
        for backend, previous in plan["memory_estimates"].items():
            estimate = driver.memory_estimate(previous["full_bins"], previous["window_bins"],
                                               previous["conservative_rank"], plan["truths"])
            memory[scope+"/"+backend] = estimate["estimated_peak_gib"]
    check("requested_memory_cases", memory["all_2015_2016_2021/eigenfeature_rtol_1e-15"] < 4.
          and memory["individual_2016_full/exact_cached_cholesky"] < 4.
          and memory["all_2015_2016_2021/exact_cached_cholesky"] > 4., peak_gib=memory)

    # Standalone JSON fixtures exercise resume without importing fitting code.
    with tempfile.TemporaryDirectory(prefix="chunked-resume-pure-") as temporary:
        temp = Path(temporary).resolve()
        directory = temp/"sampling"
        directory.mkdir()
        base_path = temp/"baseline_result.json"
        base_path.write_text('{}\n')
        baseline_hash = ref.sha(HERE/"derived/contract.json")
        base = {("individual_2016_full", mass): (base_path, {}) for mass in (75, 76)}
        marker = dict(type=ref.TYPE, version=ref.VERSION, baseline_contract_sha256=baseline_hash,
                      baseline_checkpoint_sha256=ref.sha(base_path), attempt=1)
        input_contract = dict(hashes={}, sampling_hashes={}, sampling_refinement=marker)
        ref.write_json(directory/"contract.json", input_contract)

        def checkpoint(mass):
            path = directory/"individual_2016_full"/f"m{mass:03}"/"result.json"
            data = dict(scope_key="individual_2016_full", mass_MeV=mass, status="completed_point",
                confidence_level=.9, cls_target=.1, nvalidation=500, sampling_refinement=marker,
                ntoys_per_proposal=None, ntoys_per_proposal_by_truth={t: 2 for t in driver.TRUTHS},
                provenance={t: dict(n=2, meta=dict(labels=["fixture"])) for t in driver.TRUTHS},
                results=[dict(truth=t, method=m, ntoys_per_proposal=2) for t in driver.TRUTHS for m in driver.METHODS])
            ref.write_json(path, data)
            return path

        first_path = checkpoint(75)
        first, snapshot = driver.sampling_input(directory, {"hashes": {}}, base, 1)
        checkpoint(76)
        frozen, replayed = driver.sampling_input(directory, {"hashes": {}}, base, 1,
                                                frozen_identity=snapshot)
        live, _ = driver.sampling_input(directory, {"hashes": {}}, base, 1)
        check("resume_ignores_new_completed_coordinates", set(frozen) == set(first)
              and len(frozen) == 1 and len(live) == 2 and replayed == snapshot)
        original_text = first_path.read_text()
        first_path.write_text(original_text+'\n')
        rejected = False
        try:
            driver.sampling_input(directory, {"hashes": {}}, base, 1, frozen_identity=snapshot)
        except RuntimeError:
            rejected = True
        check("resume_rejects_frozen_checkpoint_drift", rejected)
        first_path.write_text(original_text)
        configuration = dict(previous_input_directories=[],
            skip_completed_input_directories=[str(directory)], max_minutes=60.)
        layout = dict(type=driver.EXECUTION_TYPE, version=1, source_hashes={"fixture": "sha"})
        selection = dict(type=ref.TYPE, version=ref.VERSION, policy=ref.POLICY,
            numerical_policy=ref.NUMERICAL_POLICY, baseline_contract_sha256=baseline_hash,
            cli_configuration=configuration, execution_layout=layout,
            previous_inputs=[], skip_completed_inputs=[snapshot], selected=[dict(mass_MeV=75)])
        selection_path = temp/"selection.json"
        ref.write_json(selection_path, selection)
        check("resume_returns_saved_selection_directly", driver.saved_selection(
            selection_path, configuration, layout, baseline_hash) == selection)
        rejected = False
        try:
            driver.saved_selection(selection_path, dict(configuration, max_minutes=61.), layout, baseline_hash)
        except RuntimeError:
            rejected = True
        check("resume_rejects_changed_cli_configuration", rejected)
        rejected = False
        try:
            driver.saved_selection(selection_path, configuration, dict(layout, source_hashes={}), baseline_hash)
        except RuntimeError:
            rejected = True
        check("resume_rejects_execution_source_change", rejected)

    source = Path(driver.__file__)
    ast.parse(source.read_text())
    check("no_fitting_runtime_imported", not any(m in sys.modules
          for m in ("calibration_core", "run_calibration", "batch_profile", "run_comparison")))
    contract = ref.read_json(HERE/"derived/contract.json")
    mismatches = [p for p, expected in contract["hashes"].items() if ref.sha(ROOT/p) != expected]
    check("original47_dependencies_unchanged", len(contract["hashes"]) == 47 and not mismatches,
          mismatches=mismatches)
    report = dict(passed=all(r["passed"] for r in checks), test_count=len(checks), checks=checks,
        source_hashes={str(p.relative_to(ROOT)): ref.sha(p) for p in
                       [source, Path(__file__).resolve(), HERE/"CHUNKED_REFINEMENT_PROTOCOL.md"]},
        baseline_contract_sha256=ref.sha(HERE/"derived/contract.json"),
        scope="Pure deterministic arrays/interface/source checks only; no random draws, GP fitting or statistical validation")
    destination = HERE/"qa/chunked_execution_contract_test.json"
    ref.write_json(destination, report)
    print(ref.encoded(dict(passed=report["passed"], test_count=len(checks), path=str(destination),
                           estimated_peak_gib=memory)))


if __name__ == "__main__":
    main()
