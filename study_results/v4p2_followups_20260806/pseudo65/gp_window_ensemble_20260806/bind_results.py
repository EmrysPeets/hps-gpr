#!/usr/bin/env python3
"""Bind completed scan CSVs to current cards and histogram payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONFIG_MANIFEST = HERE / "derived" / "config_manifest.json"
ROOT_FILE = HERE / "inputs" / "gp_window_ensemble.root"
METADATA_REWRITE_AUDIT = HERE / "derived" / "metadata_rewrite_audit.json"

import numpy as np
import pandas as pd
import uproot
import yaml


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


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def expected_output_dir(item: dict[str, Any], mode: str) -> Path:
    if mode == "full":
        return REPO / item["output_dir"]
    if mode == "central_repeat":
        return (
            HERE
            / "central_repeat"
            / item["window"]
            / f"draw_{int(item['draw_index']):02d}"
            / "attempt_02"
        )
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "central_repeat"), required=True)
    args = parser.parse_args()
    manifest = json.loads(CONFIG_MANIFEST.read_text())
    batch_path = HERE / "derived" / f"run_batch_{args.mode}.json"
    batch = json.loads(batch_path.read_text())
    if not batch["all_complete"] or int(batch["record_count"]) != 20:
        raise RuntimeError(f"Incomplete batch: {batch_path}")
    batch_map = {
        (item["window"], int(item["draw_index"])): item
        for item in batch["records"]
    }
    root = uproot.open(ROOT_FILE)
    root_sha = sha256_file(ROOT_FILE)
    metadata_audit = json.loads(METADATA_REWRITE_AUDIT.read_text())
    bindings = []
    for item in manifest["records"]:
        key = (item["window"], int(item["draw_index"]))
        batch_item = batch_map[key]
        config_path = REPO / item["config"]
        config = yaml.safe_load(config_path.read_text())
        output_dir = expected_output_dir(item, args.mode)
        if repo_relative(output_dir) != batch_item["output_dir"]:
            raise RuntimeError(f"{key}: batch output directory mismatch")
        result_path = output_dir / "results_single.csv"
        log_path = output_dir / "scan.log"
        validation_path = output_dir / "validation_report.json"
        result_sha = sha256_file(result_path)
        if result_sha != batch_item["results_single_sha256"]:
            raise RuntimeError(f"{key}: run-batch/result SHA256 mismatch")
        if sha256_file(config_path) != item["config_sha256"]:
            raise RuntimeError(f"{key}: config SHA256 mismatch")
        log_text = log_path.read_text(errors="replace")
        if "Scan complete!" not in log_text or "Traceback" in log_text:
            raise RuntimeError(f"{key}: scan log is not cleanly complete")
        validation = json.loads(validation_path.read_text())
        report = validation["2021"]
        if (
            report["file"] != config["path_2021"]
            or report["hist"] != config["hist_2021"]
            or not bool(report["ok"])
        ):
            raise RuntimeError(f"{key}: validation report/config binding mismatch")
        values, edges = root[item["hist_key"]].to_numpy(flow=False)
        if not np.isclose(
            float(report["total_counts"]), float(np.sum(values)), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"{key}: validation total count mismatch")
        frame = pd.read_csv(result_path)
        frame = frame[frame["dataset"].astype(str) == "2021"]
        expected_rows = 201 if args.mode == "full" else 21
        if len(frame) != expected_rows or frame["mass_GeV"].nunique() != expected_rows:
            raise RuntimeError(f"{key}: result grid mismatch")
        binding = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "window": item["window"],
            "draw_index": int(item["draw_index"]),
            "config": item["config"],
            "config_sha256": item["config_sha256"],
            "root_path": repo_relative(ROOT_FILE),
            "root_file_sha256_current": root_sha,
            "hist_key": item["hist_key"],
            "hist_values_sha256": sha256_array(values),
            "hist_edges_sha256": sha256_array(edges),
            "hist_total_count": float(np.sum(values)),
            "result_csv": repo_relative(result_path),
            "result_sha256": result_sha,
            "result_row_count": int(len(frame)),
            "scan_log": repo_relative(log_path),
            "scan_log_sha256": sha256_file(log_path),
            "validation_report": repo_relative(validation_path),
            "validation_report_sha256": sha256_file(validation_path),
            "full_scan_metadata_rewrite_note": (
                metadata_audit if args.mode == "full" else None
            ),
        }
        binding_path = output_dir / "scan_input_binding.json"
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        binding["binding_path"] = repo_relative(binding_path)
        binding["binding_sha256"] = sha256_file(binding_path)
        bindings.append(binding)
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "binding_count": len(bindings),
        "pass": len(bindings) == 20,
        "bindings": bindings,
    }
    output_path = HERE / "derived" / f"scan_binding_manifest_{args.mode}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")
    print(f"bindings={len(bindings)} PASS={output['pass']}")


if __name__ == "__main__":
    main()
