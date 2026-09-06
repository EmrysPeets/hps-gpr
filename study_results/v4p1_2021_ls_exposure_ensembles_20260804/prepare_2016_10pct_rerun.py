#!/usr/bin/env python3
"""Recover and audit the archived 2016 10% histogram for a matched v4.1 rerun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import uproot
import yaml


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
INPUT_DIR = STUDY / "inputs"
CONFIG_DIR = STUDY / "configs"

ARCHIVES = [
    Path(
        "/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v3-20260709/"
        "outputs/funcform_toys/funcform_2016_toys.root"
    ),
    Path(
        "/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v4-20260803/"
        "outputs/funcform_toys/funcform_2016_toys.root"
    ),
    Path(
        "/Users/emryspeets/Desktop/gp_mods/funcform_studies/func_form_inputs/"
        "funcform_2016_dataset_mod_toys_2.root"
    ),
    ROOT / "outputs/funcform_toys/funcform_2016_toys.root",
    ROOT / "outputs/funcform_toys/funcform_2016_dataset_mod_toys.root",
]
ARCHIVE_HIST = "input_hist;2"
OUTPUT_HIST = "h_Minv_General_Final_1"
RECOVERED_ROOT = INPUT_DIR / "EventSelection_Data_10Percent_recovered.root"

SOURCE_CONFIG = (
    ROOT
    / "study_configs/v4p1_2016_ls_upper_optimization_20260804/"
    "config_2016full_wide_support_lsupper12.yaml"
)
OUTPUT_CONFIG = CONFIG_DIR / "config_2016_10pct_wide_support_lsupper12.yaml"
OUTPUT_DIR = STUDY / "observed_2016_10pct_k12_wide_support/attempt_01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram_sha256(values: np.ndarray, edges: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    digest.update(np.asarray(edges, dtype=np.float64).tobytes())
    return digest.hexdigest()


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    archive_records: list[dict[str, object]] = []
    reference_values: np.ndarray | None = None
    reference_edges: np.ndarray | None = None
    source_hist = None

    for archive in ARCHIVES:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with uproot.open(archive) as handle:
            hist = handle[ARCHIVE_HIST]
            values, edges = hist.to_numpy(flow=False)
            flow_values, _ = hist.to_numpy(flow=True)
            values = np.asarray(values, dtype=np.float64)
            edges = np.asarray(edges, dtype=np.float64)
            if reference_values is None:
                reference_values = values
                reference_edges = edges
                source_hist = hist
            values_match = np.array_equal(values, reference_values)
            edges_match = np.array_equal(edges, reference_edges)
            archive_records.append(
                {
                    "path": str(archive),
                    "file_sha256": sha256_file(archive),
                    "histogram": ARCHIVE_HIST,
                    "histogram_sha256": histogram_sha256(values, edges),
                    "values_match_reference": bool(values_match),
                    "edges_match_reference": bool(edges_match),
                    "n_bins": int(values.size),
                    "axis_lo_GeV": float(edges[0]),
                    "axis_hi_GeV": float(edges[-1]),
                    "in_range_total": float(np.sum(values)),
                    "underflow": float(flow_values[0]),
                    "overflow": float(flow_values[-1]),
                }
            )
        if not values_match or not edges_match:
            raise RuntimeError(f"archived input histogram mismatch: {archive}")

    assert source_hist is not None
    with uproot.recreate(RECOVERED_ROOT) as output:
        output[OUTPUT_HIST] = source_hist

    with uproot.open(RECOVERED_ROOT) as handle:
        recovered = handle[OUTPUT_HIST]
        recovered_values, recovered_edges = recovered.to_numpy(flow=False)
        recovered_flow, _ = recovered.to_numpy(flow=True)
    if not np.array_equal(recovered_values, reference_values):
        raise RuntimeError("recovered values differ from archived input_hist")
    if not np.array_equal(recovered_edges, reference_edges):
        raise RuntimeError("recovered edges differ from archived input_hist")

    source_card = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    output_card = dict(source_card)
    output_card["path_2016"] = str(RECOVERED_ROOT)
    output_card["output_dir"] = str(OUTPUT_DIR)
    changed = sorted(
        key for key in source_card if source_card[key] != output_card[key]
    )
    if changed != ["output_dir", "path_2016"]:
        raise RuntimeError(f"unexpected configuration changes: {changed}")
    OUTPUT_CONFIG.write_text(
        "# Generated from the reviewed v4.1 2016-full k12 card.\n"
        "# Only path_2016 and output_dir differ; no toys or bands.\n"
        + yaml.safe_dump(output_card, sort_keys=False),
        encoding="utf-8",
    )

    payload = {
        "purpose": (
            "Recover the archived 2016 10% observed histogram and rerun it with "
            "the exact v4.1 2016-full wide-support factor-12 fit card."
        ),
        "archive_histogram": ARCHIVE_HIST,
        "output_histogram": OUTPUT_HIST,
        "independent_archives_checked": len(archive_records),
        "all_archives_bitwise_equal_in_values_and_edges": True,
        "histogram_sha256": archive_records[0]["histogram_sha256"],
        "archived_in_range_total": archive_records[0]["in_range_total"],
        "archived_underflow": archive_records[0]["underflow"],
        "archived_overflow": archive_records[0]["overflow"],
        "recovered_root": str(RECOVERED_ROOT),
        "recovered_root_sha256": sha256_file(RECOVERED_ROOT),
        "recovered_in_range_total": float(np.sum(recovered_values)),
        "recovered_underflow": float(recovered_flow[0]),
        "recovered_overflow": float(recovered_flow[-1]),
        "source_config": str(SOURCE_CONFIG),
        "source_config_sha256": sha256_file(SOURCE_CONFIG),
        "output_config": str(OUTPUT_CONFIG),
        "output_config_sha256": sha256_file(OUTPUT_CONFIG),
        "config_changed_keys": changed,
        "archives": archive_records,
        "limitation": (
            "The original SDF ROOT file itself is not local. The observed "
            "histogram is recovered from five independently retained "
            "functional-form products whose archived input_hist values and "
            "edges are bitwise identical."
        ),
    }
    (STUDY / "derived").mkdir(parents=True, exist_ok=True)
    (STUDY / "derived/2016_10pct_recovery_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
