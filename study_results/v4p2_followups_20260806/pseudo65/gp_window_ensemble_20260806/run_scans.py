#!/usr/bin/env python3
"""Run pilot or full observed/asymptotic scans for the generated ensemble."""

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

import pandas as pd
import numpy as np
import uproot
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = HERE / "derived" / "config_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete_result(
    output_dir: Path,
    expected_rows: int,
    record: dict[str, Any],
    mode: str,
    *,
    require_binding: bool,
) -> bool:
    path = output_dir / "results_single.csv"
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    frame = frame[frame["dataset"].astype(str) == "2021"]
    basic_complete = bool(
        len(frame) == expected_rows
        and frame["mass_GeV"].nunique() == expected_rows
    )
    if not basic_complete or not require_binding:
        return basic_complete
    binding_path = output_dir / "scan_input_binding.json"
    if not binding_path.exists():
        return False
    try:
        binding = json.loads(binding_path.read_text())
        config_path = REPO / record["config"]
        config = yaml.safe_load(config_path.read_text())
        root_path = REPO / config["path_2021"]
        values, edges = uproot.open(root_path)[record["hist_key"]].to_numpy(
            flow=False
        )
        return bool(
            binding["mode"] == mode
            and binding["config_sha256"] == sha256_file(config_path)
            and binding["root_file_sha256_current"] == sha256_file(root_path)
            and binding["hist_key"] == record["hist_key"]
            and binding["hist_values_sha256"] == sha256_array(values)
            and binding["hist_edges_sha256"] == sha256_array(edges)
            and binding["result_sha256"] == sha256_file(path)
            and int(binding["result_row_count"]) == expected_rows
        )
    except Exception:
        return False


def run_one(record: dict[str, Any], mode: str) -> dict[str, Any]:
    config_path = REPO / record["config"]
    base_output = REPO / record["output_dir"]
    if mode == "pilot":
        output_dir = (
            HERE
            / "pilot"
            / record["window"]
            / f"draw_{int(record['draw_index']):02d}"
        )
        expected_rows = 1
    elif mode == "central_repeat":
        output_dir = (
            HERE
            / "central_repeat"
            / record["window"]
            / f"draw_{int(record['draw_index']):02d}"
            / "attempt_02"
        )
        expected_rows = 21
    else:
        output_dir = base_output
        expected_rows = 201
    if complete_result(
        output_dir,
        expected_rows,
        record,
        mode,
        require_binding=(mode != "pilot"),
    ):
        result_path = output_dir / "results_single.csv"
        return {
            **record,
            "mode": mode,
            "status": "skipped_complete",
            "exit_code": 0,
            "duration_seconds": 0.0,
            "output_dir": str(output_dir.relative_to(REPO)),
            "results_single_sha256": sha256_file(result_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
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
    ]
    if mode == "pilot":
        command.extend(["--mass-min", "0.065", "--mass-max", "0.065"])
    elif mode == "central_repeat":
        command.extend(["--mass-min", "0.055", "--mass-max", "0.075"])
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")
    started = utc_now()
    start_time = time.monotonic()
    print(
        f"START {mode} {record['window']} draw {int(record['draw_index']):02d}",
        flush=True,
    )
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
    result_path = output_dir / "results_single.csv"
    status = (
        "complete"
        if completed.returncode == 0
        and complete_result(
            output_dir,
            expected_rows,
            record,
            mode,
            require_binding=False,
        )
        else "failed_or_incomplete"
    )
    print(
        f"DONE  {mode} {record['window']} draw {int(record['draw_index']):02d} "
        f"status={status} seconds={duration:.1f}",
        flush=True,
    )
    return {
        **record,
        "mode": mode,
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
    parser.add_argument(
        "--mode",
        choices=("pilot", "full", "central_repeat"),
        required=True,
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    if not MANIFEST.exists():
        raise SystemExit("Run build_ensemble.py first")
    payload = json.loads(MANIFEST.read_text())
    records = payload["records"]
    if args.mode == "pilot":
        records = [record for record in records if int(record["draw_index"]) == 0]
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be positive")

    batch_started = utc_now()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(run_one, record, args.mode): record
            for record in records
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["window"], int(item["draw_index"])))
    batch = {
        "schema_version": 1,
        "mode": args.mode,
        "started_utc": batch_started,
        "finished_utc": utc_now(),
        "max_parallel_scans": args.max_parallel,
        "record_count": len(results),
        "all_complete": all(
            item["status"] in ("complete", "skipped_complete") for item in results
        ),
        "optimizer_reproducibility_limitation": (
            "Each full draw has one scan attempt with 12 within-fit restarts. "
            "This batch does not by itself reproduce the selected maximum-LML "
            "state in a second unchanged-card scan."
        ),
        "records": results,
    }
    out = HERE / "derived" / f"run_batch_{args.mode}.json"
    out.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out}")
    if not batch["all_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
