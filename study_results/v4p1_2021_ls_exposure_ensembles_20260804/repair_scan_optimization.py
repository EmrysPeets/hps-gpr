#!/usr/bin/env python3
"""Targeted, provenance-preserving optimizer repairs for the v4.1 pilot.

The repair plan is generated from ``scan_optimizer_repair_manifest.csv``.
Every target row receives three independently salted fits with the nominal
initialization and, when the audit recorded a feasible source optimum, one
additional independently salted warm-start fit.  Each fit retains the frozen
12 optimizer restarts and runs at one exact mass; no interpolation is
available anywhere in this program.

No fit starts without both a bounded command and the explicit ``--execute``
acknowledgement.  The normal workflow is:

1. finish every nominal scan task and rerun ``audit_scan_optimization.py``;
2. run ``prepare-plan --write``;
3. inspect ``status`` and dry-run one or more attempts;
4. after review authorization, use ``run-attempt`` or bounded ``run-pending``;
5. run ``collect`` and rerun ``audit_scan_optimization.py``.

The executor imports the production runner only to reuse its immutable-code
routing, source/toy preflight, exact one-mass fit, geometry checks, and CSV
enrichment.  It does not modify ``run_ensemble.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import socket
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Set the single-process numerical policy before NumPy/sklearn can be imported.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np
import pandas as pd
import yaml


STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import run_ensemble as runner


REPAIR_MANIFEST_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_manifest.csv"
)
AUDIT_SUMMARY_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_audit_summary.json"
)
TASK_MANIFEST_PATH = STUDY_DIR / "derived" / "task_manifest.jsonl"
PLAN_PATH = STUDY_DIR / "derived" / "scan_optimizer_repair_plan.jsonl"
PLAN_META_PATH = STUDY_DIR / "derived" / "scan_optimizer_repair_plan.json"
DRYRUN_LEDGER_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_dryrun_ledger.csv"
)
DRYRUN_SUMMARY_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_dryrun_summary.json"
)
REPAIR_RUNS_DIR = STUDY_DIR / "runs" / "scan_repairs"
PLAN_HISTORY_DIR = STUDY_DIR / "derived" / "scan_optimizer_repair_plan_history"
ROUND_EVIDENCE_DIR = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_rounds"
)
REPAIR_LEDGER_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_attempt_ledger.csv"
)
REPAIR_ROWS_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_actual_rows.csv"
)
REPAIR_COLLECTION_PATH = (
    STUDY_DIR / "derived" / "scan_optimizer_repair_collection.json"
)

N_SALTED_STARTS = 3
EXPECTED_RESTARTS = 12
WARM_FEASIBILITY_RTOL = 1.0e-8

PAIR_KEY = [
    "truth_model",
    "study_scenario",
    "background_toy_index",
    "mass_GeV",
]


class RepairError(RuntimeError):
    """Fail-closed targeted-repair error."""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RepairError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RepairError(f"Expected object at {path}:{line_number}")
            values.append(dict(value))
    return values


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RepairError(f"Malformed boolean {label}={value!r}")


def _mass_mev(mass_gev: float) -> int:
    mass = float(mass_gev)
    mass_mev = int(round(mass * 1000.0))
    if abs(mass - mass_mev / 1000.0) > 5.0e-10:
        raise RepairError(f"Repair mass is not on the integer-MeV grid: {mass}")
    return mass_mev


def _task_map() -> Dict[str, Dict[str, Any]]:
    tasks = _load_jsonl(TASK_MANIFEST_PATH)
    selected = {
        str(task["task_id"]): task
        for task in tasks
        if task.get("kind") == "scan"
    }
    if len(selected) != len(
        [task for task in tasks if task.get("kind") == "scan"]
    ):
        raise RepairError("Duplicate scan task_id in task manifest")
    return selected


def _nominal_constant_init(config_path: Path) -> float:
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise RepairError(f"Config is not a mapping: {config_path}")
    value = float(config.get("kernel_constant_init", math.nan))
    if not math.isfinite(value) or value <= 0:
        raise RepairError(
            f"Invalid kernel_constant_init in {config_path}: {value}"
        )
    if int(config.get("n_restarts", -1)) != EXPECTED_RESTARTS:
        raise RepairError(
            f"Repair config restart drift in {config_path}: "
            f"{config.get('n_restarts')}"
        )
    if bool(config.get("make_ul_bands", False)):
        raise RepairError(f"Expected-limit bands enabled in {config_path}")
    return value


def _repair_row_id(
    row: Mapping[str, Any],
    repair_round: int = 1,
) -> str:
    if int(repair_round) <= 0:
        raise RepairError(f"Invalid repair round: {repair_round}")
    prefix = (
        "repair"
        if int(repair_round) == 1
        else f"repair_r{int(repair_round):02d}"
    )
    return (
        f"{prefix}__f{int(row['repair_factor']):02d}__"
        f"{row['truth_model']}__{row['study_scenario']}__"
        f"t{int(row['background_toy_index']):04d}__"
        f"m{_mass_mev(float(row['mass_GeV'])):03d}"
    )


def _variant_seed(
    spec: Mapping[str, Any],
    repair_row_id: str,
    variant: str,
    task: Mapping[str, Any],
    mass_gev: float,
) -> Tuple[int, int]:
    repair_base_seed = runner._stable_seed32(
        spec, "optimizer_repair_base_seed", repair_row_id, variant
    )
    seeded_spec = copy.deepcopy(dict(spec))
    seeded_spec["base_seed"] = repair_base_seed
    optimizer_seed = runner._mass_seed(
        seeded_spec,
        str(task["truth_model"]),
        str(task["scenario"]),
        int(task["toy_index"]),
        float(mass_gev),
    )
    return repair_base_seed, optimizer_seed


def build_plan_rows(
    spec: Mapping[str, Any],
    repair_manifest: pd.DataFrame,
    tasks: Mapping[str, Mapping[str, Any]],
    repair_round: int = 1,
) -> List[Dict[str, Any]]:
    required = set(PAIR_KEY) | {
        "repair_factor",
        "target_task_id",
        "target_attempt_path",
        "reason",
        "warm_start_source_factor",
        "warm_start_source_attempt_path",
        "warm_start_ls_opt",
        "warm_start_const_opt",
        "warm_start_is_feasible",
        "current_target_lml",
        "source_lml",
    }
    missing = required - set(repair_manifest.columns)
    if missing:
        raise RepairError(
            f"Repair manifest is missing columns: {sorted(missing)}"
        )

    plan: List[Dict[str, Any]] = []
    seen_rows: set[str] = set()
    for _, raw in repair_manifest.iterrows():
        task_id = str(raw["target_task_id"])
        if task_id not in tasks:
            raise RepairError(f"Repair target task is absent: {task_id}")
        task = dict(tasks[task_id])
        factor = int(raw["repair_factor"])
        mass_gev = float(raw["mass_GeV"])
        repair_row_id = _repair_row_id(raw, repair_round=repair_round)
        if repair_row_id in seen_rows:
            raise RepairError(f"Duplicate repair target: {repair_row_id}")
        seen_rows.add(repair_row_id)

        if int(task["factor"]) != factor:
            raise RepairError(f"Target factor mismatch for {repair_row_id}")
        if str(task["truth_model"]) != str(raw["truth_model"]):
            raise RepairError(f"Target truth mismatch for {repair_row_id}")
        if str(task["scenario"]) != str(raw["study_scenario"]):
            raise RepairError(f"Target scenario mismatch for {repair_row_id}")
        if int(task["toy_index"]) != int(raw["background_toy_index"]):
            raise RepairError(f"Target toy mismatch for {repair_row_id}")
        _mass_mev(mass_gev)

        config_path = Path(str(task["config"])).resolve()
        if not config_path.is_file():
            raise RepairError(f"Target config is absent: {config_path}")
        config_sha = runner._sha256_file(config_path)
        nominal_constant = _nominal_constant_init(config_path)
        variants = [f"salt_{index:02d}" for index in range(1, 4)]
        warm_feasible = _as_bool(
            raw["warm_start_is_feasible"], "warm_start_is_feasible"
        )
        if warm_feasible:
            warm_ls = float(raw["warm_start_ls_opt"])
            warm_const = float(raw["warm_start_const_opt"])
            if (
                not math.isfinite(warm_ls)
                or warm_ls <= 0
                or not math.isfinite(warm_const)
                or warm_const <= 0
            ):
                raise RepairError(
                    f"Invalid feasible warm start in {repair_row_id}"
                )
            variants.append("warm")

        seeds: set[int] = set()
        for variant in variants:
            repair_attempt_id = f"{repair_row_id}__{variant}"
            repair_base_seed, optimizer_seed = _variant_seed(
                spec, repair_row_id, variant, task, mass_gev
            )
            if optimizer_seed in seeds:
                raise RepairError(
                    f"Optimizer-seed collision within {repair_row_id}"
                )
            seeds.add(optimizer_seed)
            is_warm = variant == "warm"
            entry = {
                "schema_version": 1,
                "study_id": spec["study_id"],
                "repair_round": int(repair_round),
                "repair_row_id": repair_row_id,
                "repair_attempt_id": repair_attempt_id,
                "variant": variant,
                "variant_order": (
                    4 if is_warm else int(variant.split("_")[-1])
                ),
                "warm_start": is_warm,
                "repair_reason": str(raw["reason"]),
                "target_task_id": task_id,
                "target_factor": factor,
                "truth_model": str(task["truth_model"]),
                "study_scenario": str(task["scenario"]),
                "background_toy_index": int(task["toy_index"]),
                "mass_GeV": mass_gev,
                "mass_MeV": _mass_mev(mass_gev),
                "target_config": str(config_path),
                "target_config_sha256": config_sha,
                "nominal_kernel_constant_init": nominal_constant,
                "warm_start_source_factor": (
                    int(float(raw["warm_start_source_factor"]))
                    if is_warm
                    else None
                ),
                "warm_start_source_attempt_path": (
                    str(raw["warm_start_source_attempt_path"])
                    if is_warm
                    else ""
                ),
                "warm_start_ls_opt": (
                    float(raw["warm_start_ls_opt"]) if is_warm else None
                ),
                "warm_start_const_opt": (
                    float(raw["warm_start_const_opt"]) if is_warm else None
                ),
                "current_target_lml_at_plan": float(
                    raw["current_target_lml"]
                ),
                "source_lml_at_plan": (
                    float(raw["source_lml"])
                    if is_warm and pd.notna(raw["source_lml"])
                    else None
                ),
                "repair_base_seed": repair_base_seed,
                "planned_optimizer_seed": optimizer_seed,
                "optimizer_restarts": EXPECTED_RESTARTS,
                "expected_limit_bands": False,
                "interpolation_used": False,
                "fit_code_commit": spec["fit_code"]["commit"],
            }
            plan.append(entry)

        if len(seeds) < N_SALTED_STARTS:
            raise RepairError(
                f"Fewer than {N_SALTED_STARTS} independent seeds for "
                f"{repair_row_id}"
            )

    return sorted(
        plan,
        key=lambda row: (
            row["target_factor"],
            row["truth_model"],
            row["study_scenario"],
            row["background_toy_index"],
            row["mass_MeV"],
            row["variant_order"],
        ),
    )


def prepare_plan(
    write: bool,
    write_dryrun_ledger: bool = False,
) -> Dict[str, Any]:
    if write and write_dryrun_ledger:
        raise RepairError(
            "--write and --write-dry-run-ledger are mutually exclusive"
        )
    spec = runner.load_spec()
    audit = _load_json(AUDIT_SUMMARY_PATH)
    repair_manifest = pd.read_csv(REPAIR_MANIFEST_PATH)
    plan = build_plan_rows(spec, repair_manifest, _task_map())
    completion = dict(audit.get("completion", {}))
    full_nominal_scan = (
        int(completion.get("missing_scan_tasks", -1)) == 0
        and int(completion.get("partial_scan_tasks", -1)) == 0
        and int(completion.get("unexpected_scan_tasks", -1)) == 0
    )
    write_allowed = (
        full_nominal_scan
        and int(audit.get("rejected_success_markers", -1)) == 0
        and audit.get("audit_gate")
        in {"repair_required", "optimizer_audit_pass"}
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "created_utc": runner._utc_now(),
        "dry_run": not write,
        "write_allowed": write_allowed,
        "source_audit_gate": audit.get("audit_gate"),
        "source_audit_summary": str(AUDIT_SUMMARY_PATH),
        "source_audit_summary_sha256": runner._sha256_file(
            AUDIT_SUMMARY_PATH
        ),
        "source_repair_manifest": str(REPAIR_MANIFEST_PATH),
        "source_repair_manifest_sha256": runner._sha256_file(
            REPAIR_MANIFEST_PATH
        ),
        "repair_target_rows": len(
            {row["repair_row_id"] for row in plan}
        ),
        "planned_fit_attempts": len(plan),
        "salted_attempts": sum(
            not bool(row["warm_start"]) for row in plan
        ),
        "warm_start_attempts": sum(
            bool(row["warm_start"]) for row in plan
        ),
        "optimizer_restarts_per_fit": EXPECTED_RESTARTS,
        "interpolation_used": False,
        "expected_limit_bands": False,
        "fit_launches": 0,
        "plan_path": str(PLAN_PATH),
    }
    if write_dryrun_ledger:
        dryrun_frame = pd.DataFrame(plan)
        dryrun_frame["dry_run"] = True
        dryrun_frame["execution_authorized"] = False
        dryrun_frame["fit_launched"] = False
        runner._atomic_write_text(
            DRYRUN_LEDGER_PATH, dryrun_frame.to_csv(index=False)
        )
        report["dry_run_ledger"] = str(DRYRUN_LEDGER_PATH)
        report["dry_run_ledger_sha256"] = runner._sha256_file(
            DRYRUN_LEDGER_PATH
        )
        report["dry_run_summary"] = str(DRYRUN_SUMMARY_PATH)
        runner._atomic_write_json(DRYRUN_SUMMARY_PATH, report)
    if not write:
        return report
    if not write_allowed:
        raise RepairError(
            "Repair plan cannot be frozen until the nominal scan and audit "
            "are complete and provenance-valid"
        )
    if PLAN_PATH.exists() or PLAN_META_PATH.exists():
        raise RepairError(
            "A frozen plan already exists. Use refreeze-plan to preserve its "
            "entries, seeds, and completed attempts."
        )

    plan_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in plan
    )
    runner._atomic_write_text(PLAN_PATH, plan_text)
    report["dry_run"] = False
    report["plan_sha256"] = runner._sha256_file(PLAN_PATH)
    runner._atomic_write_json(PLAN_META_PATH, report)
    return report


def _validated_current_audit() -> Dict[str, Any]:
    audit = _load_json(AUDIT_SUMMARY_PATH)
    completion = dict(audit.get("completion", {}))
    complete = (
        int(completion.get("completed_valid_scan_tasks", -1)) == 600
        and int(completion.get("missing_scan_tasks", -1)) == 0
        and int(completion.get("partial_scan_tasks", -1)) == 0
        and int(completion.get("unexpected_scan_tasks", -1)) == 0
    )
    if not complete:
        raise RepairError("Current audit does not have 600/600 nominal closure")
    if int(audit.get("rejected_success_markers", -1)) != 0:
        raise RepairError("Current audit contains rejected success markers")
    if audit.get("audit_gate") not in {
        "repair_required",
        "optimizer_audit_pass",
    }:
        raise RepairError(
            f"Current audit gate is not usable: {audit.get('audit_gate')!r}"
        )
    return audit


def _preserve_artifact(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RepairError(
                f"Preserved evidence collision: {destination}"
            )
        return
    # All preserved artifacts here are UTF-8 JSON/JSONL/CSV products.
    runner._atomic_write_text(destination, payload.decode("utf-8"))


def prepare_next_round(
    repair_round: int,
    write: bool,
    write_dryrun_ledger: bool = False,
) -> Dict[str, Any]:
    """Prepare a new repair generation without mutating prior fit evidence."""

    repair_round = int(repair_round)
    if repair_round < 2:
        raise RepairError("prepare-next-round requires --round >= 2")
    if write and write_dryrun_ledger:
        raise RepairError(
            "--write and --write-dry-run-ledger are mutually exclusive"
        )
    if not PLAN_PATH.is_file() or not PLAN_META_PATH.is_file():
        raise RepairError("Previous frozen plan is absent")
    previous_meta = _load_json(PLAN_META_PATH)
    previous_plan_sha = runner._sha256_file(PLAN_PATH)
    if previous_plan_sha != previous_meta.get("plan_sha256"):
        raise RepairError("Previous frozen plan hash mismatch")
    previous_round = int(previous_meta.get("repair_round", 1))
    if repair_round != previous_round + 1:
        raise RepairError(
            f"New round must be {previous_round + 1}, got {repair_round}"
        )

    audit = _validated_current_audit()
    spec = runner.load_spec()
    repair_manifest = pd.read_csv(REPAIR_MANIFEST_PATH)
    plan = build_plan_rows(
        spec,
        repair_manifest,
        _task_map(),
        repair_round=repair_round,
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "repair_round": repair_round,
        "created_utc": runner._utc_now(),
        "dry_run": not write,
        "write_allowed": True,
        "source_audit_gate": audit.get("audit_gate"),
        "source_audit_summary": str(AUDIT_SUMMARY_PATH),
        "source_audit_summary_sha256": runner._sha256_file(
            AUDIT_SUMMARY_PATH
        ),
        "source_repair_manifest": str(REPAIR_MANIFEST_PATH),
        "source_repair_manifest_sha256": runner._sha256_file(
            REPAIR_MANIFEST_PATH
        ),
        "repair_target_rows": len(
            {row["repair_row_id"] for row in plan}
        ),
        "planned_fit_attempts": len(plan),
        "salted_attempts": sum(
            not bool(row["warm_start"]) for row in plan
        ),
        "warm_start_attempts": sum(
            bool(row["warm_start"]) for row in plan
        ),
        "optimizer_restarts_per_fit": EXPECTED_RESTARTS,
        "interpolation_used": False,
        "expected_limit_bands": False,
        "fit_launches": 0,
        "parent_repair_round": previous_round,
        "parent_plan_sha256": previous_plan_sha,
        "parent_metadata_sha256": runner._sha256_file(PLAN_META_PATH),
        "prior_fit_directories_preserved": True,
        "round_run_directory": str(
            REPAIR_RUNS_DIR / f"round_{repair_round:03d}"
        ),
    }
    round_dir = ROUND_EVIDENCE_DIR / f"round_{repair_round:03d}"
    if write_dryrun_ledger:
        dryrun_path = round_dir / "dryrun_ledger.csv"
        dryrun_summary = round_dir / "dryrun_summary.json"
        frame = pd.DataFrame(plan)
        frame["dry_run"] = True
        frame["execution_authorized"] = False
        frame["fit_launched"] = False
        runner._atomic_write_text(dryrun_path, frame.to_csv(index=False))
        report["dry_run_ledger"] = str(dryrun_path)
        report["dry_run_ledger_sha256"] = runner._sha256_file(
            dryrun_path
        )
        report["dry_run_summary"] = str(dryrun_summary)
        runner._atomic_write_json(dryrun_summary, report)
    if not write:
        return report

    previous_dir = ROUND_EVIDENCE_DIR / f"round_{previous_round:03d}"
    _preserve_artifact(PLAN_PATH, previous_dir / "plan.jsonl")
    _preserve_artifact(
        PLAN_META_PATH, previous_dir / "metadata_final.json"
    )
    if previous_round == 1:
        _preserve_artifact(
            REPAIR_LEDGER_PATH, previous_dir / "attempt_ledger.csv"
        )
        _preserve_artifact(
            REPAIR_ROWS_PATH, previous_dir / "actual_rows.csv"
        )
        _preserve_artifact(
            REPAIR_COLLECTION_PATH, previous_dir / "collection.json"
        )

    plan_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in plan
    )
    runner._atomic_write_text(PLAN_PATH, plan_text)
    report["dry_run"] = False
    report["plan_path"] = str(PLAN_PATH)
    report["plan_sha256"] = runner._sha256_file(PLAN_PATH)
    report["previous_round_evidence"] = str(previous_dir)
    report["round_evidence"] = str(round_dir)
    runner._atomic_write_json(PLAN_META_PATH, report)
    _preserve_artifact(PLAN_PATH, round_dir / "plan.jsonl")
    _preserve_artifact(
        PLAN_META_PATH, round_dir / "metadata_initial.json"
    )
    return report


def refreeze_plan() -> Dict[str, Any]:
    """Re-anchor an existing plan after an intentional audit update.

    The JSONL plan and every recorded optimizer seed remain byte-for-byte
    unchanged.  Only its metadata is re-anchored to the current complete audit
    and repair manifest.  Existing successful repair attempts are verified and
    counted; no fit is launched, deleted, replaced, or rewritten.
    """

    if not PLAN_PATH.is_file() or not PLAN_META_PATH.is_file():
        raise RepairError("No frozen plan exists to re-freeze")
    previous_meta = _load_json(PLAN_META_PATH)
    plan_sha = runner._sha256_file(PLAN_PATH)
    if plan_sha != previous_meta.get("plan_sha256"):
        raise RepairError("Existing repair-plan SHA-256 mismatch")
    plan = _load_jsonl(PLAN_PATH)
    identifiers = [str(row["repair_attempt_id"]) for row in plan]
    if len(identifiers) != len(set(identifiers)):
        raise RepairError("Duplicate repair_attempt_id in existing plan")

    audit = _load_json(AUDIT_SUMMARY_PATH)
    completion = dict(audit.get("completion", {}))
    complete = (
        int(completion.get("completed_valid_scan_tasks", -1)) == 600
        and int(completion.get("missing_scan_tasks", -1)) == 0
        and int(completion.get("partial_scan_tasks", -1)) == 0
        and int(completion.get("unexpected_scan_tasks", -1)) == 0
    )
    if not complete:
        raise RepairError(
            "Cannot re-freeze: nominal 600-task completion gate is not valid"
        )
    if int(audit.get("rejected_success_markers", -1)) != 0:
        raise RepairError(
            "Cannot re-freeze with rejected successful-attempt markers"
        )
    if audit.get("audit_gate") not in {
        "repair_required",
        "optimizer_audit_pass",
    }:
        raise RepairError(
            f"Cannot re-freeze from audit gate {audit.get('audit_gate')!r}"
        )

    successes = 0
    failed_variants = 0
    for entry in plan:
        root = _variant_root(entry)
        if _latest_success(root) is not None:
            successes += 1
        if any(root.glob("attempt_*/_FAILED.json")):
            failed_variants += 1

    previous_meta_sha = runner._sha256_file(PLAN_META_PATH)
    PLAN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    archived_plan = (
        PLAN_HISTORY_DIR / f"plan_{plan_sha[:16]}.jsonl"
    )
    if not archived_plan.exists():
        runner._atomic_write_text(
            archived_plan, PLAN_PATH.read_text()
        )
    archived_meta = (
        PLAN_HISTORY_DIR / f"metadata_{previous_meta_sha[:16]}.json"
    )
    if not archived_meta.exists():
        runner._atomic_write_text(
            archived_meta, PLAN_META_PATH.read_text()
        )

    updated = dict(previous_meta)
    updated.update(
        {
            "refrozen_utc": runner._utc_now(),
            "refreeze_generation": int(
                previous_meta.get("refreeze_generation", 0)
            )
            + 1,
            "refrozen_from_metadata_sha256": previous_meta_sha,
            "source_audit_gate": audit.get("audit_gate"),
            "source_audit_summary_sha256": runner._sha256_file(
                AUDIT_SUMMARY_PATH
            ),
            "source_repair_manifest_sha256": runner._sha256_file(
                REPAIR_MANIFEST_PATH
            ),
            "successful_attempts_preserved": successes,
            "pending_plan_attempts": len(plan) - successes,
            "variants_with_failed_attempt_history": failed_variants,
            "plan_entries_and_seeds_changed": False,
            "fit_launches": 0,
            "interpolation_used": False,
            "expected_limit_bands": False,
            "archived_plan": str(archived_plan),
            "archived_previous_metadata": str(archived_meta),
            "execution_note": (
                "Do not rerun audit_scan_optimization.py while this frozen "
                "plan is executing; an intentional audit rewrite invalidates "
                "the source hashes and stops new children fail-closed."
            ),
        }
    )
    runner._atomic_write_json(PLAN_META_PATH, updated)
    return {
        "status": "refrozen",
        "plan_path": str(PLAN_PATH),
        "plan_sha256": plan_sha,
        "plan_entries": len(plan),
        "successful_attempts_preserved": successes,
        "pending_plan_attempts": len(plan) - successes,
        "variants_with_failed_attempt_history": failed_variants,
        "metadata_path": str(PLAN_META_PATH),
        "metadata_sha256": runner._sha256_file(PLAN_META_PATH),
        "plan_entries_and_seeds_changed": False,
        "fit_launches": 0,
    }


def _load_frozen_plan() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not PLAN_PATH.is_file() or not PLAN_META_PATH.is_file():
        raise RepairError("Frozen repair plan is absent; run prepare-plan --write")
    metadata = _load_json(PLAN_META_PATH)
    if runner._sha256_file(PLAN_PATH) != metadata.get("plan_sha256"):
        raise RepairError("Frozen repair-plan SHA-256 mismatch")
    if (
        runner._sha256_file(REPAIR_MANIFEST_PATH)
        != metadata.get("source_repair_manifest_sha256")
    ):
        raise RepairError(
            "Repair manifest drifted after plan freeze; prepare a new plan"
        )
    if (
        runner._sha256_file(AUDIT_SUMMARY_PATH)
        != metadata.get("source_audit_summary_sha256")
    ):
        raise RepairError(
            "Optimizer audit drifted after plan freeze; prepare a new plan"
        )
    if not bool(metadata.get("write_allowed")):
        raise RepairError("Frozen plan metadata is not execution-authorized")
    rows = _load_jsonl(PLAN_PATH)
    identifiers = [str(row["repair_attempt_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RepairError("Duplicate repair_attempt_id in frozen plan")
    return rows, metadata


def _variant_root(entry: Mapping[str, Any]) -> Path:
    base = REPAIR_RUNS_DIR
    repair_round = int(entry.get("repair_round", 1))
    if repair_round > 1:
        base = base / f"round_{repair_round:03d}"
    return (
        base
        / f"f{int(entry['target_factor']):02d}"
        / str(entry["truth_model"])
        / str(entry["study_scenario"])
        / f"toy_{int(entry['background_toy_index']):04d}"
        / f"m{int(entry['mass_MeV']):03d}MeV"
        / str(entry["variant"])
    )


def _valid_success(attempt: Path) -> Optional[Dict[str, Any]]:
    marker_path = attempt / "_SUCCESS.json"
    if not marker_path.is_file():
        return None
    marker = _load_json(marker_path)
    result = Path(str(marker.get("result_path", ""))).resolve()
    if result.parent != attempt.resolve() or not result.is_file():
        raise RepairError(f"Malformed repair success marker: {marker_path}")
    if runner._sha256_file(result) != marker.get("result_sha256"):
        raise RepairError(f"Repair result hash mismatch: {marker_path}")
    return marker


def _latest_success(root: Path) -> Optional[Path]:
    successes: List[Path] = []
    for attempt in sorted(root.glob("attempt_*")):
        if _valid_success(attempt) is not None:
            successes.append(attempt)
    return successes[-1] if successes else None


def _choose_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    attempts = [
        path
        for path in root.glob("attempt_*")
        if path.is_dir()
    ]
    next_index = (
        1
        if not attempts
        else max(int(path.name.split("_")[-1]) for path in attempts) + 1
    )
    attempt = root / f"attempt_{next_index:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".repair.lock"
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_utc": runner._utc_now(),
    }
    try:
        descriptor = os.open(
            lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
    except FileExistsError as exc:
        raise RepairError(
            f"Repair variant is locked; inspect before clearing: {lock}"
        ) from exc
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return lock


def _write_warm_config(
    entry: Mapping[str, Any],
    attempt: Path,
    ls_bounds: Tuple[float, float],
) -> Tuple[Path, float]:
    target = Path(str(entry["target_config"])).resolve()
    if runner._sha256_file(target) != entry["target_config_sha256"]:
        raise RepairError(f"Target config drift for {entry['repair_attempt_id']}")
    with target.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise RepairError(f"Target config is not a mapping: {target}")
    warm_ls = float(entry["warm_start_ls_opt"])
    warm_const = float(entry["warm_start_const_opt"])
    ls_low, ls_high = (float(ls_bounds[0]), float(ls_bounds[1]))
    if warm_ls < ls_low * (1.0 - WARM_FEASIBILITY_RTOL) or warm_ls > (
        ls_high * (1.0 + WARM_FEASIBILITY_RTOL)
    ):
        raise RepairError(
            f"Recorded warm LS {warm_ls} is outside "
            f"[{ls_low}, {ls_high}]"
        )
    init_by_dataset = dict(config.get("kernel_ls_init_by_dataset", {}) or {})
    init_by_dataset["2021"] = warm_ls
    config["kernel_ls_init_by_dataset"] = init_by_dataset
    # The immutable resolution-scaled-local implementation consults
    # kernel_ls_init_by_dataset only when explicit per-dataset bounds are
    # present.  This repair config is mass-specific, so record the *same*
    # target-card bounds explicitly to activate the warm initial coordinate
    # without changing the feasible domain.
    bounds_by_dataset = dict(
        config.get("kernel_ls_bounds_by_dataset", {}) or {}
    )
    bounds_by_dataset["2021"] = [ls_low, ls_high]
    config["kernel_ls_bounds_by_dataset"] = bounds_by_dataset
    config["kernel_constant_init"] = warm_const
    config["n_restarts"] = EXPECTED_RESTARTS
    config["make_ul_bands"] = False
    config["ul_bands_toys"] = 0
    config["do_combined_bands"] = False
    config["combined_bands_n_toys"] = 0
    config["make_eps2_bands"] = False
    config_path = attempt / "repair_config.yaml"
    runner._atomic_write_text(
        config_path, yaml.safe_dump(config, sort_keys=False)
    )
    runner._atomic_write_json(
        attempt / "repair_config.provenance.json",
        {
            "schema_version": 1,
            "repair_attempt_id": entry["repair_attempt_id"],
            "target_config": str(target),
            "target_config_sha256": entry["target_config_sha256"],
            "repair_config": str(config_path),
            "repair_config_sha256": runner._sha256_file(config_path),
            "kernel_ls_init_by_dataset.2021": warm_ls,
            "kernel_ls_bounds_by_dataset.2021": [ls_low, ls_high],
            "kernel_constant_init": warm_const,
            "optimizer_restarts": EXPECTED_RESTARTS,
            "expected_limit_bands": False,
        },
    )
    return config_path, warm_const


def _target_ls_bounds(
    config_path: Path,
    mass_gev: float,
) -> Tuple[float, float]:
    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.gpr import compute_kernel_ls_bounds

    config = load_config(str(config_path))
    datasets = make_datasets(config)
    if "2021" not in datasets:
        raise RepairError("Target repair config does not enable 2021")
    bounds = compute_kernel_ls_bounds(
        datasets["2021"], config, mass=float(mass_gev)
    )
    return float(bounds["ls_lo"]), float(bounds["ls_hi"])


def _validate_warm_start(
    config_path: Path,
    entry: Mapping[str, Any],
    expected_bounds: Tuple[float, float],
) -> None:
    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.gpr import make_kernel_for_dataset

    config = load_config(str(config_path))
    if int(config.n_restarts) != EXPECTED_RESTARTS:
        raise RepairError("Warm repair restart count drift")
    datasets = make_datasets(config)
    if "2021" not in datasets:
        raise RepairError("Warm repair config does not enable 2021")
    kernel = make_kernel_for_dataset(
        datasets["2021"], config, mass=float(entry["mass_GeV"])
    )
    rbf = kernel.k2
    applied_constant = float(kernel.k1.constant_value)
    applied_init = float(rbf.length_scale)
    applied_bounds = tuple(float(x) for x in rbf.length_scale_bounds)
    warm_ls = float(entry["warm_start_ls_opt"])
    if not np.isclose(
        applied_init, warm_ls, rtol=1.0e-12, atol=1.0e-14
    ):
        raise RepairError(
            f"Warm LS was not applied: requested {warm_ls}, "
            f"kernel has {applied_init}"
        )
    if not np.isclose(
        applied_constant,
        float(entry["warm_start_const_opt"]),
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise RepairError(
            "Warm kernel constant was not applied exactly"
        )
    if not np.allclose(
        applied_bounds,
        np.asarray(expected_bounds, dtype=float),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise RepairError(
            f"Warm config changed target bounds: {applied_bounds} "
            f"!= {expected_bounds}"
        )
    constant_bounds = tuple(float(x) for x in config.kernel_constant_bounds)
    warm_const = float(entry["warm_start_const_opt"])
    if not (constant_bounds[0] <= warm_const <= constant_bounds[1]):
        raise RepairError(
            f"Recorded warm constant {warm_const} is outside "
            f"{constant_bounds}"
        )


def _task_for_entry(
    entry: Mapping[str, Any],
    config_path: Path,
) -> Dict[str, Any]:
    tasks = _task_map()
    task_id = str(entry["target_task_id"])
    if task_id not in tasks:
        raise RepairError(f"Frozen target task is absent: {task_id}")
    task = dict(tasks[task_id])
    task["config"] = str(config_path)
    return task


def _execute_one(
    entry: Mapping[str, Any], attempt: Path
) -> Path:
    spec = runner.load_spec()
    runner._configure_fit_process()
    runner.preflight(spec)
    runner.validate_toys(spec)
    runner._activate_fit_code(spec)

    target_config = Path(str(entry["target_config"])).resolve()
    if runner._sha256_file(target_config) != entry["target_config_sha256"]:
        raise RepairError(
            f"Target config drift for {entry['repair_attempt_id']}"
        )
    if bool(entry["warm_start"]):
        target_bounds = _target_ls_bounds(
            target_config, float(entry["mass_GeV"])
        )
        config_path, constant_init = _write_warm_config(
            entry, attempt, target_bounds
        )
        _validate_warm_start(config_path, entry, target_bounds)
        kernel_ls_init = float(entry["warm_start_ls_opt"])
    else:
        config_path = target_config
        constant_init = _nominal_constant_init(config_path)
        kernel_ls_init = math.nan

    task = _task_for_entry(entry, config_path)
    seeded_spec = copy.deepcopy(spec)
    seeded_spec["base_seed"] = int(entry["repair_base_seed"])
    calculated_seed = runner._mass_seed(
        seeded_spec,
        str(task["truth_model"]),
        str(task["scenario"]),
        int(task["toy_index"]),
        float(entry["mass_GeV"]),
    )
    if calculated_seed != int(entry["planned_optimizer_seed"]):
        raise RepairError("Planned optimizer seed does not reproduce")

    config_sha = runner._sha256_file(config_path)
    part_result = runner._run_one_scan_mass(
        seeded_spec,
        task,
        attempt,
        float(entry["mass_GeV"]),
        config_sha,
    )
    frame = pd.read_csv(part_result)
    if len(frame) != 1:
        raise RepairError("Targeted repair did not return exactly one row")
    if int(float(frame.loc[0, "optimizer_seed"])) != calculated_seed:
        raise RepairError("Repair output optimizer seed mismatch")
    if int(float(frame.loc[0, "optimizer_restarts_requested"])) != (
        EXPECTED_RESTARTS
    ):
        raise RepairError("Repair output restart-count mismatch")
    if not math.isfinite(kernel_ls_init):
        kernel_ls_init = float(frame.loc[0, "ls_init"])

    frame["repair_attempt_id"] = str(entry["repair_attempt_id"])
    frame["repair_row_id"] = str(entry["repair_row_id"])
    frame["repair_round"] = int(entry.get("repair_round", 1))
    frame["repair_variant"] = str(entry["variant"])
    frame["repair_reason"] = str(entry["repair_reason"])
    frame["repair_warm_start"] = bool(entry["warm_start"])
    frame["repair_source_factor"] = entry.get("warm_start_source_factor")
    frame["repair_source_attempt_path"] = str(
        entry.get("warm_start_source_attempt_path", "")
    )
    frame["repair_base_seed"] = int(entry["repair_base_seed"])
    frame["repair_kernel_constant_init"] = float(constant_init)
    frame["repair_kernel_ls_init"] = float(kernel_ls_init)
    frame["repair_config_path"] = str(config_path)
    frame["repair_config_sha256"] = config_sha
    frame["repair_fit_is_actual"] = True
    frame["repair_interpolation_used"] = False
    output = attempt / "repair_row_enriched.csv"
    runner._atomic_write_text(output, frame.to_csv(index=False))
    return output


def run_attempt(
    repair_attempt_id: str,
    execute: bool,
    force: bool = False,
) -> Dict[str, Any]:
    plan, _ = _load_frozen_plan()
    by_id = {str(row["repair_attempt_id"]): row for row in plan}
    if repair_attempt_id not in by_id:
        raise RepairError(f"Unknown repair attempt: {repair_attempt_id}")
    entry = by_id[repair_attempt_id]
    root = _variant_root(entry)
    success = _latest_success(root)
    if success is not None and not force:
        return {
            "repair_attempt_id": repair_attempt_id,
            "status": "already_complete",
            "attempt": str(success),
        }
    if not execute:
        return {
            "repair_attempt_id": repair_attempt_id,
            "status": "dry_run",
            "command": (
                f"{sys.executable} -B {Path(__file__).resolve()} "
                f"run-attempt {repair_attempt_id} --execute"
            ),
            "fit_launches": 0,
            "mass_GeV": entry["mass_GeV"],
            "optimizer_seed": entry["planned_optimizer_seed"],
            "optimizer_restarts": EXPECTED_RESTARTS,
            "warm_start": entry["warm_start"],
        }

    lock = _acquire_lock(root)
    try:
        success_after_lock = _latest_success(root)
        if success_after_lock is not None and not force:
            return {
                "repair_attempt_id": repair_attempt_id,
                "status": "already_complete",
                "attempt": str(success_after_lock),
            }
        attempt = _choose_attempt(root)
        output = _execute_one(entry, attempt)
        output_frame = pd.read_csv(output)
        actual_config = Path(
            str(output_frame.loc[0, "repair_config_path"])
        ).resolve()
        marker = {
            "schema_version": 1,
            "study_id": entry["study_id"],
            "attempt": str(attempt.resolve()),
            "completed_utc": runner._utc_now(),
            "expected_limit_bands": False,
            "fit_code_commit": entry["fit_code_commit"],
            "grid_tag": f"repair_m{int(entry['mass_MeV']):03d}",
            "masses_gev": [float(entry["mass_GeV"])],
            "result_path": str(output.resolve()),
            "result_sha256": runner._sha256_file(output),
            "task": _task_for_entry(
                entry,
                actual_config,
            ),
            "repair": dict(entry),
            "interpolation_used": False,
        }
        runner._atomic_write_json(attempt / "_SUCCESS.json", marker)
        return {
            "repair_attempt_id": repair_attempt_id,
            "status": "completed",
            "attempt": str(attempt),
            "result": str(output),
            "result_sha256": marker["result_sha256"],
        }
    except Exception as exc:
        if "attempt" not in locals():
            raise
        runner._atomic_write_json(
            attempt / "_FAILED.json",
            {
                "schema_version": 1,
                "repair_attempt_id": repair_attempt_id,
                "failed_utc": runner._utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def status() -> Dict[str, Any]:
    plan, _ = _load_frozen_plan()
    rows: List[Dict[str, Any]] = []
    for entry in plan:
        root = _variant_root(entry)
        success = _latest_success(root)
        lock = root / ".repair.lock"
        failures = sorted(root.glob("attempt_*/_FAILED.json"))
        if success is not None:
            state = "complete"
        elif lock.exists():
            state = "locked"
        elif failures:
            state = "failed"
        else:
            state = "pending"
        rows.append(
            {
                "repair_attempt_id": entry["repair_attempt_id"],
                "state": state,
                "successful_attempt": "" if success is None else str(success),
            }
        )
    counts = pd.DataFrame(rows)["state"].value_counts().to_dict()
    return {
        "planned": len(rows),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "rows": rows,
    }


def _subprocess_attempt(entry: Mapping[str, Any]) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "run-attempt",
        str(entry["repair_attempt_id"]),
        "--execute",
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
    )
    return {
        "repair_attempt_id": entry["repair_attempt_id"],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def run_pending(
    max_attempts: int,
    workers: int,
    execute: bool,
) -> Dict[str, Any]:
    if max_attempts <= 0:
        raise RepairError("--max-attempts must be positive")
    if workers <= 0:
        raise RepairError("--workers must be positive")
    plan, _ = _load_frozen_plan()
    pending = [
        row for row in plan if _latest_success(_variant_root(row)) is None
    ][:max_attempts]
    commands = [
        (
            f"{sys.executable} -B {Path(__file__).resolve()} run-attempt "
            f"{row['repair_attempt_id']} --execute"
        )
        for row in pending
    ]
    if not execute:
        return {
            "status": "dry_run",
            "selected_attempts": len(pending),
            "workers": workers,
            "fit_launches": 0,
            "commands": commands,
        }

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_subprocess_attempt, entry): entry
            for entry in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    failures = [row for row in results if int(row["returncode"]) != 0]
    if failures:
        details = "\n".join(
            f"{row['repair_attempt_id']}: {row['stderr'][-2000:]}"
            for row in failures
        )
        raise RepairError(
            f"{len(failures)} targeted repair subprocesses failed:\n{details}"
        )
    return {
        "status": "completed",
        "selected_attempts": len(pending),
        "workers": workers,
        "fit_launches": len(pending),
        "results": results,
    }


def _collection_paths(
    metadata: Mapping[str, Any],
) -> Tuple[Path, Path, Path]:
    repair_round = int(metadata.get("repair_round", 1))
    if repair_round == 1:
        return (
            REPAIR_LEDGER_PATH,
            REPAIR_ROWS_PATH,
            REPAIR_COLLECTION_PATH,
        )
    directory = ROUND_EVIDENCE_DIR / f"round_{repair_round:03d}"
    return (
        directory / "attempt_ledger.csv",
        directory / "actual_rows.csv",
        directory / "collection.json",
    )


def collect() -> Dict[str, Any]:
    plan, metadata = _load_frozen_plan()
    by_id = {str(row["repair_attempt_id"]): row for row in plan}
    ledger: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []
    for entry in plan:
        for marker_path in sorted(
            _variant_root(entry).glob("attempt_*/_SUCCESS.json")
        ):
            attempt = marker_path.parent
            marker = _valid_success(attempt)
            if marker is None:
                continue
            repair = marker.get("repair")
            if not isinstance(repair, dict):
                raise RepairError(f"Repair metadata absent: {marker_path}")
            attempt_id = str(repair["repair_attempt_id"])
            if attempt_id not in by_id:
                raise RepairError(
                    f"Success is absent from frozen plan: {attempt_id}"
                )
            output = Path(str(marker["result_path"]))
            frame = pd.read_csv(output)
            if len(frame) != 1:
                raise RepairError(f"Repair result is not one row: {output}")
            frames.append(frame)
            ledger.append(
                {
                    "repair_round": int(repair.get("repair_round", 1)),
                    "repair_attempt_id": attempt_id,
                    "repair_row_id": repair["repair_row_id"],
                    "variant": repair["variant"],
                    "warm_start": repair["warm_start"],
                    "target_factor": repair["target_factor"],
                    "truth_model": repair["truth_model"],
                    "study_scenario": repair["study_scenario"],
                    "background_toy_index": repair["background_toy_index"],
                    "mass_GeV": repair["mass_GeV"],
                    "optimizer_seed": repair["planned_optimizer_seed"],
                    "optimizer_restarts": repair["optimizer_restarts"],
                    "attempt_path": str(attempt),
                    "result_path": str(output),
                    "result_sha256": marker["result_sha256"],
                    "lml": float(frame.loc[0, "lml"]),
                    "ls_opt": float(frame.loc[0, "ls_opt"]),
                    "ls_init": float(frame.loc[0, "ls_init"]),
                    "const_opt": float(frame.loc[0, "const_opt"]),
                    "kernel_constant_init": float(
                        frame.loc[0, "repair_kernel_constant_init"]
                    ),
                    "kernel_ls_init": float(
                        frame.loc[
                            0,
                            (
                                "repair_kernel_ls_init"
                                if "repair_kernel_ls_init" in frame.columns
                                else "ls_init"
                            ),
                        ]
                    ),
                    "actual_fit": True,
                    "interpolation_used": False,
                }
            )

    ledger_frame = pd.DataFrame(ledger)
    rows_frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    ledger_path, rows_path, collection_path = _collection_paths(metadata)
    runner._atomic_write_text(ledger_path, ledger_frame.to_csv(index=False))
    runner._atomic_write_text(rows_path, rows_frame.to_csv(index=False))
    complete_ids = set(ledger_frame.get("repair_attempt_id", []))
    missing = sorted(set(by_id) - complete_ids)
    report = {
        "schema_version": 1,
        "study_id": runner.load_spec()["study_id"],
        "repair_round": int(metadata.get("repair_round", 1)),
        "collected_utc": runner._utc_now(),
        "planned_attempts": len(by_id),
        "successful_actual_fit_attempts": len(complete_ids),
        "missing_attempts": len(missing),
        "missing_attempt_ids": missing,
        "interpolation_used": False,
        "expected_limit_bands": False,
        "ledger": str(ledger_path),
        "actual_rows": str(rows_path),
        "prior_round_evidence_preserved": True,
    }
    runner._atomic_write_json(collection_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare-plan",
        help="Dry-run a plan, or freeze it after nominal scans finish",
    )
    prepare.add_argument(
        "--write",
        action="store_true",
        help="Freeze plan files; omitted means read-only dry-run",
    )
    prepare.add_argument(
        "--write-dry-run-ledger",
        action="store_true",
        help=(
            "Write a non-executable CSV/JSON ledger with fit_launches=0; "
            "does not freeze the production plan"
        ),
    )

    sub.add_parser("status", help="Summarize frozen repair-plan state")
    next_round = sub.add_parser(
        "prepare-next-round",
        help=(
            "Dry-run or freeze a new repair round while preserving the "
            "previous plan, collection, and fit directories"
        ),
    )
    next_round.add_argument("--round", type=int, required=True)
    next_round.add_argument("--write", action="store_true")
    next_round.add_argument(
        "--write-dry-run-ledger",
        action="store_true",
        help="Write a non-executable round-specific dry-run ledger",
    )
    sub.add_parser(
        "refreeze-plan",
        help=(
            "Re-anchor unchanged plan entries/seeds to a reviewed current "
            "audit while preserving all existing fit attempts"
        ),
    )

    run = sub.add_parser(
        "run-attempt", help="Dry-run or execute one exact repair attempt"
    )
    run.add_argument("repair_attempt_id")
    run.add_argument("--execute", action="store_true")
    run.add_argument(
        "--force",
        action="store_true",
        help="Create a new actual attempt after a reviewed success",
    )

    pending = sub.add_parser(
        "run-pending",
        help="Dry-run or execute a bounded set in fresh subprocesses",
    )
    pending.add_argument("--max-attempts", type=int, required=True)
    pending.add_argument("--workers", type=int, default=1)
    pending.add_argument("--execute", action="store_true")

    sub.add_parser(
        "collect", help="Verify successes and write actual-row attempt ledgers"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare-plan":
        result = prepare_plan(
            write=bool(args.write),
            write_dryrun_ledger=bool(args.write_dry_run_ledger),
        )
    elif args.command == "status":
        result = status()
    elif args.command == "prepare-next-round":
        result = prepare_next_round(
            repair_round=int(args.round),
            write=bool(args.write),
            write_dryrun_ledger=bool(args.write_dry_run_ledger),
        )
    elif args.command == "refreeze-plan":
        result = refreeze_plan()
    elif args.command == "run-attempt":
        result = run_attempt(
            args.repair_attempt_id,
            execute=bool(args.execute),
            force=bool(args.force),
        )
    elif args.command == "run-pending":
        result = run_pending(
            max_attempts=int(args.max_attempts),
            workers=int(args.workers),
            execute=bool(args.execute),
        )
    elif args.command == "collect":
        result = collect()
    else:
        raise RepairError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
