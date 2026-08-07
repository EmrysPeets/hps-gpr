#!/usr/bin/env python3
"""Run unchanged-card targeted repeats for unresolved 55--75 MeV states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONFIG_MANIFEST = HERE / "derived" / "config_manifest.json"
CENTRAL_REVIEW = HERE / "derived" / "central_optimizer_review.csv"

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_one(
    record: dict[str, Any],
    mass: float,
    round_number: int,
) -> dict[str, Any]:
    mass_mev = int(round(1000.0 * mass))
    output_dir = (
        HERE
        / "central_repairs"
        / record["window"]
        / f"draw_{int(record['draw_index']):02d}"
        / f"m{mass_mev:03d}"
        / f"attempt_{round_number:02d}"
    )
    result_path = output_dir / "results_single.csv"
    if result_path.exists():
        frame = pd.read_csv(result_path)
        frame = frame[frame["dataset"].astype(str) == "2021"]
        if (
            len(frame) == 1
            and np.isclose(float(frame.iloc[0]["mass_GeV"]), mass)
        ):
            return {
                **record,
                "mass_GeV": mass,
                "round": round_number,
                "status": "skipped_complete",
                "exit_code": 0,
                "duration_seconds": 0.0,
                "output_dir": str(output_dir.relative_to(REPO)),
                "results_single_sha256": sha256_file(result_path),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = REPO / record["config"]
    log_path = output_dir / "scan.log"
    command = [
        sys.executable,
        "-m",
        "hps_gpr.cli",
        "scan",
        "--config",
        str(config_path.relative_to(REPO)),
        "--output-dir",
        str(output_dir.relative_to(REPO)),
        "--mass-min",
        f"{mass:.3f}",
        "--mass-max",
        f"{mass:.3f}",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")
    print(
        f"START repair r{round_number:02d} {record['window']} "
        f"draw {int(record['draw_index']):02d} m={1000.0 * mass:.0f}",
        flush=True,
    )
    started = utc_now()
    start_time = time.monotonic()
    with log_path.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.monotonic() - start_time
    status = "failed_or_incomplete"
    if completed.returncode == 0 and result_path.exists():
        frame = pd.read_csv(result_path)
        frame = frame[frame["dataset"].astype(str) == "2021"]
        if (
            len(frame) == 1
            and np.isclose(float(frame.iloc[0]["mass_GeV"]), mass)
        ):
            status = "complete"
    print(
        f"DONE  repair r{round_number:02d} {record['window']} "
        f"draw {int(record['draw_index']):02d} m={1000.0 * mass:.0f} "
        f"status={status} seconds={duration:.1f}",
        flush=True,
    )
    return {
        **record,
        "mass_GeV": mass,
        "round": round_number,
        "status": status,
        "exit_code": int(completed.returncode),
        "started_utc": started,
        "finished_utc": utc_now(),
        "duration_seconds": duration,
        "command": command,
        "output_dir": str(output_dir.relative_to(REPO)),
        "scan_log": str(log_path.relative_to(REPO)),
        "results_single_sha256": (
            sha256_file(result_path) if result_path.exists() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    if args.round < 3:
        raise SystemExit("Repair round must be at least 3")
    review = pd.read_csv(CENTRAL_REVIEW)
    unresolved = review[
        review["selected_state_reproducing_attempt_count"].to_numpy(int) < 2
    ].copy()
    manifest = json.loads(CONFIG_MANIFEST.read_text())
    config_map = {
        (item["window"], int(item["draw_index"])): item
        for item in manifest["records"]
    }
    jobs = [
        (
            config_map[(row["window"], int(row["draw_index"]))],
            float(row["mass_GeV"]),
        )
        for _, row in unresolved.iterrows()
    ]
    if not jobs:
        print("No unresolved central states; no repair jobs required")
        return
    started = utc_now()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(run_one, record, mass, args.round): (record, mass)
            for record, mass in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(
        key=lambda item: (
            item["window"],
            int(item["draw_index"]),
            float(item["mass_GeV"]),
        )
    )
    output = {
        "schema_version": 1,
        "round": args.round,
        "started_utc": started,
        "finished_utc": utc_now(),
        "job_count": len(results),
        "all_complete": all(
            item["status"] in ("complete", "skipped_complete")
            for item in results
        ),
        "records": results,
    }
    output_path = (
        HERE / "derived" / f"run_batch_central_repairs_round_{args.round:02d}.json"
    )
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")
    if not output["all_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
