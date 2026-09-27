"""A task-local guard; workers stop at checkpoint boundaries if STOP exists."""
from pathlib import Path
import time, datetime, json
B=Path(__file__).resolve().parents[1]
deadline=datetime.datetime(2026,9,21,23,5,tzinfo=datetime.timezone.utc).timestamp()
(B/'qa/resource_budget.json').write_text(json.dumps({'deadline_utc':'2026-09-21T23:05:00Z','initial_weekly_used_percent':19,'cumulative_original_task_weekly_stop_percent':30,'poll_interval_seconds':2},indent=2)+'\n')
while not (B/'COMPLETE').exists():
    if time.time()>=deadline:
        (B/'STOP').write_text('Task wall-clock limit reached. Stop new numerical work.\n')
        break
    time.sleep(2)
