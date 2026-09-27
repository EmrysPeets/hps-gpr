#!/usr/bin/env python3
"""Verify unchanged numerical results and exercise the complete cache path."""
from pathlib import Path
import sys,subprocess,json,hashlib,time,difflib
B=Path(__file__).resolve().parents[1];P=B/'provenance/resume_fix'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    original=json.loads((P/'original_numeric_csv_hashes.json').read_text())
    assert all(sha(B/'results'/name)==digest for name,digest in original.items()),'Fresh numerical reproduction changed'
    checkpoint_files=[p for d in ('scan_checkpoints','calibration_checkpoints') for p in (B/'results'/d).iterdir()]
    before={str(p.relative_to(B)):(p.stat().st_mtime_ns,sha(p)) for p in checkpoint_files}
    start=time.monotonic();run=subprocess.run([sys.executable,str(B/'scripts/run_observed.py')],capture_output=True,text=True)
    (B/'results/cache_replay.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stdout+run.stderr
    assert all((B/p).stat().st_mtime_ns==v[0] and sha(B/p)==v[1] for p,v in before.items()),'Cache replaced checkpoint files'
    assert all(sha(B/'results'/name)==digest for name,digest in original.items()),'Cached numerical results changed'
    old=json.loads((P/'original_protocol.json').read_text());new=json.loads((B/'provenance/protocol.json').read_text());oldsha=old.pop('script_sha256');newsha=new.pop('script_sha256');assert old==new
    oldcode=(P/'original_run_observed.py').read_text();newcode=(B/'scripts/run_observed.py').read_text()
    expected=oldcode.replace("pd.read_csv(path,float_precision='round_trip')","pd.read_csv(path,float_precision='round_trip',keep_default_na=False)")
    assert expected==newcode and oldsha==sha(P/'original_run_observed.py') and newsha==sha(B/'scripts/run_observed.py')
    assert sha(P/'original_selected_regions.json')==sha(B/'results/selected_regions.json')
    (P/'parser_only_fix.diff').write_text(''.join(difflib.unified_diff(oldcode.splitlines(keepends=True),newcode.splitlines(keepends=True),fromfile='original_run_observed.py',tofile='run_observed.py')))
    report=dict(passed=True,reason='Pandas default NA parsing interpreted the valid cohort label null as missing on checkpoint reload. keep_default_na=False preserves the label.',
        fresh_reproduction_all_numerical_CSVs_byte_identical=True,full_cache_replay_all_numerical_CSVs_byte_identical=True,
        original_numeric_csv_sha256=original,unchanged_checkpoint_file_count=len(before),checkpoint_content_and_mtimes_unchanged=True,
        scientific_protocol_unchanged=True,selected_regions_unchanged=True,old_runner_sha256=oldsha,new_runner_sha256=newsha,
        cache_elapsed_seconds=time.monotonic()-start,scope='Fresh scan and every toy extraction were rerun from empty checkpoint directories, then full cache replay verified. Selected fit component NPZs are intentionally regenerated from the observed counts on each build.',
        preserved_originals='provenance/resume_fix stores executed source, protocol, QA, region selection, checkpoint metadata and original numeric hashes. Full original checkpoint files remain outside the package in tmp/pdfs/v638_before_resume_fix.')
    (B/'qa/resume_fix_reproduction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
