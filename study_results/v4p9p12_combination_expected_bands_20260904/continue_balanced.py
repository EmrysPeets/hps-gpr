#!/usr/bin/env python3
"""Continue the frozen runner with a small, low-priority pool of mass blocks."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, choices=(100, 300), default=300)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    blocks = [(low, min(low + 7, 250)) for low in range(19, 251, 8)]
    ranges = ((19, 90), (39, 180), (50, 250), (39, 90),
              (50, 90), (50, 180), (50, 90))
    def cost(block):
        return sum(sum(low <= m <= high for low, high in ranges) ** 1.35
                   for m in range(block[0], block[1] + 1))
    blocks.sort(key=lambda block: (-cost(block), block[0]))
    assert sorted(m for low, high in blocks for m in range(low, high + 1)) == list(range(19, 251))
    plan = {"target_toys": args.target_toys, "workers": args.workers,
            "threads_per_worker": 1, "nice_increment": 10,
            "schedule": "heaviest scope-count blocks first; dynamic refill",
            "blocks": blocks}
    print(json.dumps(plan), flush=True)
    if args.plan:
        return
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    logs = HERE / "logs" / f"{args.target_toys}toys"
    logs.mkdir(parents=True, exist_ok=True)
    runner = str(HERE / "run_expected_bands.py")
    base = ["nice", "-n", "10", sys.executable, "-u", runner,
            "--target-toys", str(args.target_toys), "--workers", "1"]
    def run(block):
        low, high = block
        started = time.monotonic()
        path = logs / f"m{low:03d}_{high:03d}.log"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"\nStarted {datetime.now(timezone.utc).isoformat()}\n")
            stream.flush()
            result = subprocess.run(base + ["--mass-min-mev", str(low),
                                            "--mass-max-mev", str(high)],
                                    cwd=REPO, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        return {"mass_min": low, "mass_max": high, "exit_code": result.returncode,
                "seconds": round(time.monotonic() - started, 2),
                "log": str(path.relative_to(REPO))}
    records = []
    queue = iter(blocks)
    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {pool.submit(run, next(queue)) for _ in range(args.workers)}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                records.append(record)
                print(json.dumps({"completed_blocks": len(records), **record}), flush=True)
                failed = failed or record["exit_code"] != 0
            if not failed:
                for _ in done:
                    block = next(queue, None)
                    if block is not None:
                        active.add(pool.submit(run, block))
    report = {**plan, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "block_results": records, "all_blocks_passed": not failed}
    qa = HERE / "qa" / f"execution_{args.target_toys}toys.json"
    qa.parent.mkdir(parents=True, exist_ok=True)
    qa.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("A block failed; completed checkpoints are retained. See logs.")
    subprocess.run(base, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
