#!/usr/bin/env python3
"""Verify fresh execution preserved every numerical result after a resume fix."""
from pathlib import Path
import json,hashlib,difflib
B=Path(__file__).resolve().parents[1]
P=B/'provenance/resume_fix'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    original=json.loads((P/'original_numeric_csv_hashes.json').read_text())
    rows=[dict(file=name,original_sha256=value,reproduced_sha256=sha(B/'results'/name),identical=value==sha(B/'results'/name)) for name,value in original.items()]
    assert all(r['identical'] for r in rows),'Numerical CSV changed'
    old=json.loads((P/'original_toy_protocol.json').read_text());new=json.loads((B/'provenance/toy_protocol.json').read_text())
    old_code=old.pop('script_sha256');new_code=new.pop('script_sha256')
    assert old==new,'Scientific protocol changed beyond runner hash'
    assert old_code==sha(P/'original_run_study.py') and new_code==sha(B/'scripts/run_study.py')
    oldm=json.loads((P/'original_calibration_mismatch_protocol.json').read_text());newm=json.loads((B/'provenance/calibration_mismatch_protocol.json').read_text())
    oldm.pop('protocol_sha256');newm.pop('protocol_sha256');assert oldm==newm
    assert sha(P/'original_frozen_selection.json')==sha(B/'results/frozen_selection.json')
    original_code=(P/'original_run_study.py').read_text()
    expected=original_code.replace("    path=B/'provenance/toy_protocol.json'", "    # JSON stores tuple-valued intervals as lists. Compare the canonical JSON\n    # representation so a valid frozen protocol remains resumable after reload.\n    obj=json.loads(json.dumps(obj))\n    path=B/'provenance/toy_protocol.json'")
    assert expected==(B/'scripts/run_study.py').read_text(),'Unexpected runner edit'
    (P/'serialization_only_fix.diff').write_text(''.join(difflib.unified_diff(original_code.splitlines(keepends=True),expected.splitlines(keepends=True),fromfile='original_run_study.py',tofile='run_study.py')))
    checkpoints=list((B/'results/checkpoints').glob('*.json'))
    for marker in checkpoints:
        data=json.loads(marker.read_text());pp='calibration_mismatch_protocol.json' if marker.name.startswith('mismatch_') else 'toy_protocol.json'
        assert data['protocol_sha256']==sha(B/'provenance'/pp)
        assert data['rows_sha256']==sha(marker.with_suffix('.csv')) and data['draws_sha256']==sha(marker.with_suffix('.npz'))
    report=dict(passed=True,reason='JSON serializes tuple-valued candidate intervals as lists. Canonicalization makes frozen protocol equality valid on resume.',
        scope='Serialization-only change; fresh primary and mismatch studies rerun from empty checkpoint directory using the unchanged seeds and scientific protocol.',
        all_original_numerical_CSVs_byte_identical=True,numerical_csv_count=len(rows),numerical_csvs=rows,
        selected_policy_unchanged=True,scientific_protocol_unchanged=True,new_checkpoint_count=len(checkpoints),
        old_runner_sha256=old_code,new_runner_sha256=new_code,
        preserved_originals='provenance/resume_fix contains original scripts, protocols, selection, checkpoint metadata, QA and numerical hashes; full old checkpoints retained outside the release in tmp/pdfs/v637_original_checkpoints.',
        required_current_checks={name:json.loads((B/'qa'/name).read_text())['passed'] for name in ('toy_validation.json','calibration_mismatch.json','statistics_review.json','fit_replay.json','resume_cache.json')})
    assert all(report['required_current_checks'].values())
    (B/'qa/resume_fix_reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','numerical_csv_count','all_original_numerical_CSVs_byte_identical','new_checkpoint_count')},indent=2))
if __name__=='__main__':main()
