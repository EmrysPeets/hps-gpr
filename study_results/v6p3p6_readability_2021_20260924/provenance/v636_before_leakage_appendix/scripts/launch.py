"""Real wall-clock watchdog; preserve atomic completed checkpoints on timeout."""
from pathlib import Path
import os,sys,subprocess,signal,json,datetime
B=Path(__file__).resolve().parents[1]
proc=subprocess.Popen([sys.executable,str(B/'scripts/run_study.py'),*sys.argv[1:]],start_new_session=True)
try:
    code=proc.wait(timeout=1800)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    (B/'status.json').write_text(json.dumps(dict(status='watchdog_interrupted',timeout_seconds=1800,
        resume='Re-run scripts/launch.py; completed chunks are retained',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
    code=124
sys.exit(code)
