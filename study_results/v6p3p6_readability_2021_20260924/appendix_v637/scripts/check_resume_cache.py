#!/usr/bin/env python3
"""Exercise both completed study cache paths without generating new toys."""
from pathlib import Path
import sys,subprocess,time,json,hashlib
B=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    saved={str(p.relative_to(B)):(p.stat().st_mtime_ns,sha(p)) for p in (B/'results/checkpoints').iterdir() if p.suffix in ('.csv','.npz','.json')}
    numeric={p.name:sha(p) for p in (B/'results').glob('*.csv')}
    runs=[]
    for script in ('run_study.py','calibration_mismatch.py'):
        start=time.monotonic()
        result=subprocess.run([sys.executable,str(B/'scripts'/script)],capture_output=True,text=True)
        (B/'results'/('resume_'+script.replace('.py','.log'))).write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        runs.append(dict(script=script,returncode=result.returncode,elapsed_seconds=time.monotonic()-start))
    assert all((B/p).stat().st_mtime_ns==value[0] and sha(B/p)==value[1] for p,value in saved.items())
    assert all(sha(B/'results'/name)==value for name,value in numeric.items())
    audit=dict(passed=True,completed_cache_paths_exercised=2,runs=runs,
        checkpoint_files_untouched=len(saved),numerical_CSVs_unchanged=len(numeric),
        new_toy_draws_or_toy_extraction_fits=False,
        note='The primary cache run still recomputes its 30 deterministic profile-limit diagnostics and saved-row summaries; all toy checkpoint arrays, rows and marker files remain byte-identical with unchanged modification times.',
        script_sha256=sha(__file__))
    (B/'qa/resume_cache.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
