"""Rebuild with four single-thread workers and a30-minute process-group watchdog."""
from pathlib import Path
import os,sys,subprocess,signal,time
B=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+1800
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    env[key]='1'
env['MPLCONFIGDIR']='/tmp/hps-v62-mpl'
for script,args in [('run_study.py',['--workers','4']),('aggregate.py',[]),
                    ('validate_independent.py',[]),('make_report.py',[])]:
    p=subprocess.Popen([sys.executable,str(B/'scripts'/script),*args],env=env,start_new_session=True)
    try:
        code=p.wait(timeout=max(1,deadline-time.monotonic()))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        raise SystemExit('Study watchdog reached1800seconds; saved checkpoints retained.')
    if code:raise SystemExit(code)
print('Scientific products and PDF rebuilt. Render and inspect the PDF before packaging.')
