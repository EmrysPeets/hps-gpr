#!/usr/bin/env python3
"""Finalize saved m074 numerics after a mixed-schema bookkeeping exception.

No generation, fitting, endpoint evaluation, or numerical-source modification.
The original failed directory and its unverified result are retained verbatim.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import shutil
import scalar_reference_recovery as recovery

HERE, ROOT, ref = recovery.HERE, recovery.ROOT, recovery.ref
SOURCE = HERE/'scalar_reference_recovery_v1/attempt2_m074'
OUTPUT = HERE/'scalar_reference_recovery_v1/attempt2_m074_finalized'
POINT = Path(recovery.SCOPE)/'m074'
PROTOCOL = HERE/'REFERENCE_METADATA_FINALIZATION.md'
FIELDS = {'scalar_reference_recovery', 'postprocessing_finalization'}
COMPANIONS = ('point_plan.json', 'proposal_plan.json', 'memory_check.json',
              'pre_generation_numerical_qa.json', 'chunked_equivalence_checks.json',
              'model_chunk_ledger.json', 'validation_summary.csv', 'validation_toys.csv.gz')


def input_paths():
    return [Path(__file__).resolve(), PROTOCOL, SOURCE/'contract.json',
            SOURCE/POINT/'unverified_result.json', SOURCE/POINT/'failure_numerical_qa.json',
            SOURCE/POINT/'FAILURE.txt', *[SOURCE/POINT/name for name in COMPANIONS],
            *[SOURCE/name for name in ('selection.json', 'prelaunch_resource_check.json')
              if (SOURCE/name).is_file()]]


def saved_audit(raw, failure):
    recovery.require(raw['scalar_checks'] == failure['scalar_checks']
        and raw['numerical_checks'] == failure['numerical_checks']
        and raw['chunked_equivalence_checks'] == failure['chunked_equivalence_checks'],
        'Saved numerical ledgers disagree')
    legacy = [r for r in raw['scalar_checks'] if 'batch_id' not in r]
    batch = [r for r in raw['scalar_checks'] if 'batch_id' in r]
    recovery.require(len(legacy) == 72 and len(batch) == 3700
        and all(r.get('passed') is True and r.get('kind') == 'refinement_extended_scalar'
                and r.get('label', {}).get('check_stage') == 'extended' for r in legacy)
        and len(failure['scalar_reference_fallbacks']) == 1,
        'Unexpected saved scalar-check schemas or fallback count')
    ctx = SimpleNamespace(gp_backend=raw['gp_backend'], nuisance_cut=raw['nuisance_eigenvalue_cut'],
        scalar_checks=batch, scalar_reference_fallbacks=copy.deepcopy(failure['scalar_reference_fallbacks']))
    # The original audit applies unchanged to its batch-reference row schema.
    # All legacy rows remain unchanged in the result and pass the full collector.
    audit = recovery.recovery_audit(ctx)
    event = audit['fallbacks'][0]
    check = event['scalar_check']
    recovery.require(check['execution_call_id'] == 2 and check['chunk_start'] == 9728
        and check['global_toy_index'] == 9728 and check['n_spectra'] == 128
        and event['window_counts_sha256'] == recovery.KNOWN_WINDOW_SHA,
        'Recovered scalar reference is not the known failed science-bank row')
    return audit


def verify_finalization(data, contract):
    meta = data['postprocessing_finalization']
    paths = input_paths()
    recovery.require(meta['type'] == 'saved_reference_metadata_finalization' and meta['version'] == 1
        and meta['numerical_reexecution'] is False and meta['passed'] is True
        and meta['input_sha256'] == {str(p.relative_to(ROOT)):ref.sha(p) for p in paths},
        'Finalization source identities differ')
    recovery.require(contract['postprocessing_finalization'] == meta
        and all(contract['sampling_hashes'][str(p.relative_to(ROOT))] == ref.sha(p) for p in paths),
        'Finalization inputs are not frozen in the derivative contract')
    raw = ref.read_json(SOURCE/POINT/'unverified_result.json')
    recovery.require({k:v for k,v in data.items() if k not in FIELDS} == raw,
        'Finalization changed a numerical result or source field')
    failure_text = (SOURCE/POINT/'FAILURE.txt').read_text()
    recovery.require('KeyError: \'batch_id\'' in failure_text and 'recovery_audit' in failure_text,
        'The saved failure is not the reviewed metadata exception')
    ledger = ref.read_json(data['scalar_reference_recovery']['ledger_path'])
    destination = Path(data['scalar_reference_recovery']['ledger_path']).parent
    recovery.require(all(ref.sha(destination/name) == ref.sha(SOURCE/POINT/name) for name in COMPANIONS),
        'A copied numerical/validation companion differs from the saved run')
    recovery.require(ledger == saved_audit(raw, ref.read_json(SOURCE/POINT/'failure_numerical_qa.json')),
        'Finalization ledger cannot be reproduced from saved records')
    original = ref.read_json(SOURCE/'contract.json')
    expected = copy.deepcopy(original)
    expected['postprocessing_finalization'] = meta
    expected['sampling_hashes'].update(meta['input_sha256'])
    recovery.require(contract == expected, 'Derivative contract changes original inference inputs')
    return paths


def main():
    recovery.require(not OUTPUT.exists(), 'Use a fresh derivative; never overwrite the failed or finalized run')
    raw = ref.read_json(SOURCE/POINT/'unverified_result.json')
    contract = ref.read_json(SOURCE/'contract.json')
    ref.check_hashes(contract)
    ledger = saved_audit(raw, ref.read_json(SOURCE/POINT/'failure_numerical_qa.json'))
    metadata = dict(type='saved_reference_metadata_finalization', version=1, passed=True,
        numerical_reexecution=False, retained_legacy_scalar_rows=72, retained_batch_scalar_rows=3700,
        input_sha256={str(p.relative_to(ROOT)):ref.sha(p) for p in input_paths()})
    destination = OUTPUT/POINT
    destination.mkdir(parents=True)
    for name in ('selection.json', 'prelaunch_resource_check.json'):
        if (SOURCE/name).exists():shutil.copy2(SOURCE/name, OUTPUT/name)
    for name in COMPANIONS:
        shutil.copy2(SOURCE/POINT/name, destination/name)
    ledger_path = destination/'scalar_reference_recovery.json'
    ref.write_json(ledger_path, ledger)
    diagnostic = HERE/'scalar_reference_recovery_v1/diagnostic_m074'
    result = copy.deepcopy(raw)
    result['scalar_reference_recovery'] = dict(type=recovery.TYPE, version=recovery.VERSION, passed=True,
        ledger_path=str(ledger_path), ledger_sha256=ref.sha(ledger_path), diagnostic_directory=str(diagnostic),
        diagnostic_summary_sha256=ref.sha(diagnostic/'summary.json'),
        original_attempt_contract_sha256=ref.sha(recovery.FAILED_ROOT/'contract.json'),
        identical_whole_bank_sha256={t:raw['provenance'][t]['whole_sha256'] for t in ref.TRUTHS},
        validation_seed_namespaces=[['validation',recovery.SCOPE,74,t,s] for t in ref.TRUTHS for s in (0,2,5)])
    result['postprocessing_finalization'] = metadata
    contract['postprocessing_finalization'] = metadata
    contract['sampling_hashes'].update(metadata['input_sha256'])
    ref.write_json(OUTPUT/'contract.json', contract, freeze=True)
    verify_finalization(result, contract)
    recovery.verify_recovery_layout(result['execution_layout'], result)
    ref.write_json(destination/'result.json', result, freeze=True)
    ref.write_json(OUTPUT/'batch_summary.json', dict(invocation_finished=True, passed=True,
        completed=[dict(scope_key=recovery.SCOPE, mass_MeV=74)], scheduled_deferred=[],
        generated_calibration_spectra=0, reused_original_calibration_spectra=112896,
        scheduling_slice='Saved-result metadata finalization; no numerical reexecution'))
    print(ref.encoded(dict(status='saved_result_finalized', result=str(destination/'result.json'),
                          fallback_count=ledger['fallback_count'])))


if __name__ == '__main__':main()
