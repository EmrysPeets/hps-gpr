#!/usr/bin/env python3
"""Supplement the immutable collector with explicit chunked-execution gates."""
from pathlib import Path
import argparse
import json
import math
import sys

sys.dont_write_bytecode = True
import run_chunked_refinement as execution
import run_sampling_refinement as ref
import collect_results as collector

HERE, ROOT = ref.HERE, ref.ROOT


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def bounded(value, maximum):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= maximum


def same_number(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def audit_point(path):
    path = Path(path).resolve()
    data = ref.read_json(path)
    contract = ref.read_json(path.parents[2]/'contract.json')
    ref.check_hashes(contract)
    layout = data['execution_layout']
    inherited_layout = {k:v for k,v in layout.items() if k != 'scalar_reference_recovery'}
    recovery_inputs = []
    recovery_count = 0
    if 'scalar_reference_recovery' in layout:
        import scalar_reference_recovery as recovery
        recovery_inputs = recovery.verify_recovery_layout(layout, data)
        record = data['scalar_reference_recovery']
        ledger = ref.read_json(record['ledger_path'])
        diagnostic = ref.read_json(Path(record['diagnostic_directory'])/'summary.json')
        require(record['passed'] is True and record['type'] == recovery.TYPE and record['version'] == 1,
                'Reference recovery record type/status differs')
        require(record['diagnostic_summary_sha256'] == ref.sha(Path(record['diagnostic_directory'])/'summary.json')
                and record['original_attempt_contract_sha256'] == ref.sha(recovery.FAILED_ROOT/'contract.json'),
                'Reference recovery provenance differs')
        require(record['identical_whole_bank_sha256'] == {t:data['provenance'][t]['whole_sha256'] for t in ref.TRUTHS}
                and record['validation_seed_namespaces'] == [['validation',data['scope_key'],74,t,s] for t in ref.TRUTHS for s in (0,2,5)],
                'Reference recovery bank/validation metadata differs')
        known = [e for e in ledger['fallbacks'] if e['window_counts_sha256'] == recovery.KNOWN_WINDOW_SHA]
        require(len(known) == 1 and known[0]['full_counts_sha256'] == diagnostic['target_full_counts_sha256'],
                'Known failed spectrum is not recovered exactly once')
        for event in ledger['fallbacks']:
            fit, init, check = event['final_original_fit'], event['initializer'], event['scalar_check']
            require(event['method'] == 'fixed' and bounded(fit['score'], recovery.SCORE_GATE)
                    and fit['score'] < recovery.SCORE_GATE and bounded(fit['min_lambda'], math.inf)
                    and fit['min_lambda'] > 0 and all(math.isfinite(fit[k]) for k in ('A','nll','sigma')),
                    'Recovered scalar fit is nonfinite or fails the original gate')
            require(init['domain_lower_z'] < init['bracket_z'][0] <= init['root_z'] <= init['bracket_z'][1]
                    and all(math.isfinite(x) for x in init['bracket_z']+init['bracket_score'])
                    and check['method'] == event['method'] and check['batch_id'] == event['batch_id']
                    and check['toy_index'] == event['toy_index'] and check['counts_sha256'] == event['window_counts_sha256'],
                    'Reference recovery bracket or row identity differs')
        for p in recovery_inputs:
            if p == Path(record['ledger_path']):continue
            name = str(p.relative_to(ROOT))
            require(contract['hashes'].get(name, contract['sampling_hashes'].get(name)) == ref.sha(p),
                    'Reference recovery source/diagnostic input is not frozen')
        recovery_count = ledger['fallback_count']
        if 'postprocessing_finalization' in data:
            import finalize_reference_metadata as finalization
            recovery_inputs.extend(finalization.verify_finalization(data, contract))
    resource_inputs = []
    resource_override = 'resource_policy' in layout
    if resource_override:
        version = layout['resource_policy'].get('version')
        if version == 1:
            import run_chunked_refinement_6gib as resource
            resource_protocol = HERE/'CHUNKED_RESOURCE_PROTOCOL.md'
        elif version == 2:
            import run_chunked_refinement_8gib as resource
            resource_protocol = HERE/'CHUNKED_RESOURCE8_PROTOCOL.md'
        else:
            raise RuntimeError('Unsupported combined resource policy')
        resource.validate_resource_limits(layout['max_memory_gib'], data['scope_key'])
        expected_layout = resource.layout_marker(layout['max_memory_gib'])
        resource_inputs = [Path(resource.__file__), resource_protocol,
                           resource.verify_resource_qa(expected_layout), HERE/'qa/chunked_execution_contract_test.json']
        require(all(contract['sampling_hashes'].get(str(p.relative_to(ROOT))) == ref.sha(p)
                    for p in resource_inputs), 'Resource policy and QA are not frozen')
    else:
        expected_layout = execution.layout_marker(layout['max_memory_gib'])
        require(bounded(layout['max_memory_gib'], 4) and layout['max_memory_gib'] > 0, 'Invalid memory guard')
    require(layout == contract['execution_layout'] and inherited_layout == expected_layout, 'Execution identity differs')
    for name, sha in layout['source_hashes'].items():
        require(contract['sampling_hashes'].get(name) == sha == ref.sha(ROOT/name), 'Execution source is not frozen')
    require(contract['hashes'] == ref.read_json(ref.BASE/'contract.json')['hashes'], 'Original contract changed')
    parent = path.parent
    require(not (parent/'FAILURE.txt').exists(), 'Retained failure in selected point')
    plan = ref.read_json(parent/'point_plan.json')
    require(plan['execution_layout'] == layout, 'Plan execution identity differs')
    require(ref.sha(parent/'point_plan.json') == data['sampling_refinement']['point_plan_sha256'], 'Plan hash differs')
    require(ref.sha(parent/'proposal_plan.json') == data['sampling_refinement']['proposal_plan_sha256'], 'Proposal plan hash differs')
    original_qa = collector.point_qa(data, path, collector.read_csv(parent/'validation_summary.csv'))
    require(original_qa['numerical_pass'], 'Original numerical/candidate gate failed')

    audit = data['chunked_equivalence_checks']
    require(audit == ref.read_json(parent/'chunked_equivalence_checks.json'), 'Execution audit copies differ')
    require(audit['schema_version'] == 1 and audit['passed'] is True and audit['statistic_and_density_passed'] is True, 'Incomplete execution audit')
    require((audit['split_chunk_size'], audit['production_chunk_size'], audit['r_tolerance'], audit['q_tolerance']) == (1, 128, 2e-5, 1e-4), 'Statistic tolerances/layout changed')
    require((audit['log_density_atol'], audit['weight_rtol'], audit['weight_atol']) == (1e-7, 2e-7, 1e-12), 'Density tolerances changed')
    spectra = audit['spectra']
    expected_original = {(t, i) for t in ref.TRUTHS for i in range(9)}
    original = [r for r in spectra if r['stage'] == 'original']
    extended = [r for r in spectra if r['stage'] == 'extended']
    expected_extended = {(t, c, shift) for t in ref.TRUTHS for c in ref.extended_probes(plan['truths'][t])[0] for shift in range(3)}
    require(len(original) == 18 and {(r['truth'], r['proposal']) for r in original} == expected_original, 'Original audit coverage differs')
    require(len(extended) == len(expected_extended) and {(r['truth'], r['center'], r['proposal_shift']) for r in extended} == expected_extended, 'Extended audit coverage differs')
    require(len(spectra) == audit['unsplit_n_spectra'] == 18+len(extended) <= 128 and audit['n_original'] == 18 and audit['n_extended'] == len(extended), 'Audit spectrum count differs')
    require(audit['extended_reference_counts_replayed'] is True and digest(audit['audit_whole_sha256']), 'Missing replay identity')
    reference = {}
    for r in data['scalar_checks']:
        label = r.get('label', {})
        if label.get('check_stage') == 'extended':
            reference.setdefault((label['truth'], label['center'], label['proposal_shift']), set()).add(label['full_counts_sha256'])
    for r in spectra:
        require(digest(r['full_counts_sha256']), 'Missing audit count hash')
        expected = [2., 5., 12.] if r['stage'] == 'original' else ref.extended_probes(plan['truths'][r['truth']])[1]
        require(r['strengths'] == expected, 'Audit strengths differ')
        if r['stage'] == 'extended':
            require(reference.get((r['truth'], r['center'], r['proposal_shift'])) == {r['full_counts_sha256']}, 'Extended replay count identity differs')
    checks = audit['statistic_checks']
    require(len(checks) == 2*len(spectra) and {(r['audit_index'], r['method']) for r in checks} == {(i, m) for i in range(len(spectra)) for m in ref.METHODS}, 'Statistic audit pairs missing/duplicated')
    max_r, max_q = 0., 0.
    for r in checks:
        coordinate = spectra[r['audit_index']]
        require(all(r[k] == v for k, v in coordinate.items()) and r['passed'] is True, 'Statistic row coordinate/flag differs')
        values = [r[k] for k in ('scalar_r', 'unsplit_r', 'split_r')]
        require(all(math.isfinite(v) for v in values), 'Nonfinite signed root')
        error = max(values)-min(values)
        require(bounded(error, 2e-5) and same_number(error, r['r_error']), 'Signed-root gate failed')
        max_r = max(max_r, error)
        require([q['strength_sigma'] for q in r['q_checks']] == coordinate['strengths'], 'q audit strengths differ')
        for q in r['q_checks']:
            values = [q[k] for k in ('scalar_q', 'unsplit_q', 'split_q')]
            require(all(bounded(v, math.inf) for v in values), 'Nonfinite/negative bounded q')
            error = max(values)-min(values)
            require(bounded(error, 1e-4) and same_number(error, q['q_error']), 'Bounded-q gate failed')
            max_q = max(max_q, error)
    density = audit['density_checks']
    require(len(density) == 2 and {r['truth'] for r in density} == set(ref.TRUTHS), 'Density truth coverage differs')
    for r in density:
        truth = r['truth']
        indices = [i for i, x in enumerate(spectra) if x['truth'] == truth]
        require(r['audit_indices'] == indices and r['n_spectra'] == len(indices) and r['passed'] is True, 'Density audit rows differ')
        require(r['proposals_sha256'] == data['provenance'][truth]['proposals_sha256'] and digest(r['full_counts_sha256']), 'Density array identity differs')
        require(bounded(r['logmix_max_abs_error'], 1e-7), 'Density gate failed')
        strengths = sorted({0., *plan['truths'][truth]['scan_nodes'], *(a for i in indices for a in spectra[i]['strengths'])})
        require([q['strength_sigma'] for q in r['weight_checks']] == strengths, 'Weight audit strengths differ')
        require(all(q['passed'] is True and q['finite'] is True and bounded(q['max_scaled_error'], 1.) and bounded(q['max_abs_error'], math.inf) and bounded(q['max_relative_error'], math.inf) for q in r['weight_checks']), 'Weight gate failed')

    memory = data['memory_check']
    require(memory == ref.read_json(parent/'memory_check.json') and memory['passed'] is True, 'Memory audit copies differ')
    expected = execution.memory_estimate(memory['full_bins'], memory['window_bins'], memory['conservative_rank'], plan['truths'])
    require(memory['gp_backend'] == data['gp_backend'] and expected == plan['memory_estimates'][data['gp_backend']], 'Memory backend/plan differs')
    require(all(memory[k] == v for k, v in expected.items()) and memory['estimated_peak_gib'] <= memory['limit_gib'] == layout['max_memory_gib'], 'Memory bound does not close')
    require(set(audit['generation_checks']) == set(ref.TRUTHS) == set(data['qcache_ledger']), 'Bank closure coverage differs')
    baseline = ref.read_json(data['sampling_refinement']['baseline_checkpoint_path'])
    for truth in ref.TRUTHS:
        g, spec, provenance = audit['generation_checks'][truth], plan['truths'][truth], data['provenance'][truth]
        n, k = spec['ntoys_per_proposal'], spec['proposal_count']
        require(g['passed'] is True and g['whole_sha256'] == g['concatenated_proposal_bytes_sha256'] == provenance['whole_sha256'], 'Whole-array closure failed')
        require(g['dtype'] == 'int64' and g['shape'] == [n*k, memory['full_bins']] and g['rng_call_shape'] == [n, memory['full_bins']] and g['rng_calls'] == k, 'RNG shape/count identity differs')
        require(len(g['proposal_draw_sha256']) == k and all(digest(h) for h in g['proposal_draw_sha256']), 'Per-proposal draw hashes missing')
        seed = ['sampling-refinement-v1', data['scope_key'], data['mass_MeV'], truth, plan['attempt'], provenance['proposals_sha256']] if spec['refined'] else ['calibration', data['scope_key'], data['mass_MeV'], truth, 256]
        require(g['seed_namespace'] == provenance['seed_namespace'] == seed and g['regenerated_first_pass'] == (not spec['refined']), 'Seed identity differs')
        if not spec['refined']:
            require(g['baseline_whole_sha256'] == g['whole_sha256'] == baseline['provenance'][truth]['whole_sha256'], 'Unrefined original array differs')
        cache = data['qcache_ledger'][truth]
        keys = [(x['method'], x['strength_sigma']) for x in cache['keys']]
        require(cache['limit'] == max(128, 2*(len(spec['scan_nodes'])+18)) and cache['passed'] is True, 'Cache capacity changed')
        require(cache['entries'] == cache['peak_entries'] == len(keys) == len(set(keys)) <= cache['limit'] and cache['retained_bytes'] == len(keys)*n*k*8, 'Cache bound/count differs')
    ledger_path = parent/'model_chunk_ledger.json'
    require(ref.sha(ledger_path) == data['model_chunk_ledger_sha256'], 'Model chunk ledger hash differs')
    ledger = ref.read_json(ledger_path)
    require(len(ledger) == 9 and [r['purpose'] for r in ledger] == ['execution_equivalence', 'calibration_gp', 'calibration_stress']+['validation']*6, 'Model calls differ')
    used_indices = set()
    for index, call in enumerate(ledger):
        expected_n = len(spectra) if index == 0 else data['provenance'][ref.TRUTHS[index-1]]['n'] if index < 3 else 500
        size = 1 if index == 0 else 128
        require(call['call_id'] == index and call['passed'] is True and call['n_spectra'] == expected_n and call['chunk_size'] == size, 'Chunk call shape differs')
        require([(r['start'], r['stop']) for r in call['chunks']] == [(i, min(i+size, expected_n)) for i in range(0, expected_n, size)], 'Chunk partition differs')
        for chunk in call['chunks']:
            ids = chunk['scalar_check_indices']
            require(chunk['passed'] is True and len(ids) == 2*min(2, chunk['stop']-chunk['start']) and not used_indices.intersection(ids), 'Chunk scalar coverage differs')
            used_indices.update(ids)
            for i in ids:
                row = data['scalar_checks'][i]
                require(row['passed'] is True and row['execution_call_id'] == index and row['chunk_start'] == chunk['start'] and row['global_toy_index'] == chunk['start']+row['toy_index'], 'Chunk scalar identity differs')
    return dict(path=str(path), sha256=ref.sha(path), passed=True, spectra_audited=len(spectra), max_r_error=max_r,
                max_q_error=max_q, memory_peak_gib=memory['estimated_peak_gib'], model_calls=len(ledger),
                memory_limit_gib=layout['max_memory_gib'], resource_override=resource_override,
                reference_recovery_count=recovery_count,
                input_sha256={str(p.relative_to(ROOT)):ref.sha(p) for p in [path, path.parents[2]/'contract.json', parent/'point_plan.json', parent/'proposal_plan.json', parent/'memory_check.json', parent/'chunked_equivalence_checks.json', parent/'validation_summary.csv', parent/'validation_toys.csv.gz', Path(data['sampling_refinement']['baseline_checkpoint_path']), ledger_path, *resource_inputs, *recovery_inputs]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append')
    parser.add_argument('--output', type=Path, default=HERE/'summary/chunked_execution_qa.json')
    args = parser.parse_args()
    source = HERE/'summary/observed_calibrated_limits.csv'
    before = ref.sha(source)
    paths = args.checkpoint or [Path(r['checkpoint_path']) for r in collector.read_csv(source) if r['checkpoint_completed'] == 'True']
    selected = [p for p in paths if ref.read_json(p).get('execution_layout')]
    require(not args.checkpoint or len(selected) == len(paths), 'An explicitly supplied checkpoint is not a chunked result')
    result = dict(passed=False, collection_sha256=None if args.checkpoint else before, selected_chunked_count=len(selected), checks=[])
    try:
        for p in selected:
            result['checks'].append(audit_point(p))
        require(args.checkpoint or before == ref.sha(source), 'Collection changed during execution audit')
        result.update(passed=True, source_sha256={str(p.relative_to(ROOT)):ref.sha(p) for p in [Path(__file__), Path(execution.__file__), HERE/'CHUNKED_REFINEMENT_PROTOCOL.md', Path(collector.__file__)]})
    except Exception as error:
        result['error'] = str(error)
        ref.write_json(args.output, result)
        raise
    ref.write_json(args.output, result)
    print(json.dumps(dict(passed=result['passed'], selected_chunked_count=len(selected), output=str(args.output))))


if __name__ == '__main__':
    main()
