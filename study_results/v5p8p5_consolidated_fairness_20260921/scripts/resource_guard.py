import os,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEADLINE=1790018755
state={'started_utc':'2026-09-21T18:25:55Z','deadline_utc':'2026-09-21T19:25:55Z','deadline_unix':DEADLINE,'initial_weekly_used_percent':10,'stop_weekly_used_percent':30,'fit_blind_half_width_sigma':2.25,'worker_groups':[]}
(ROOT/'resource_budget.json').write_text(json.dumps(state,indent=2)+'\n')
while time.time()<DEADLINE and not (ROOT/'STOP').exists():
    if (ROOT/'COMPLETE').exists(): break
    time.sleep(2)
if not (ROOT/'COMPLETE').exists():
    (ROOT/'STOP').write_text('Hard time or usage limit reached. Stop calculations and report current artifacts.\n')
    state=json.loads((ROOT/'resource_budget.json').read_text())
    for pid in state.get('worker_groups',[]):
        try: os.killpg(pid,signal.SIGTERM)
        except (ProcessLookupError,PermissionError): pass
