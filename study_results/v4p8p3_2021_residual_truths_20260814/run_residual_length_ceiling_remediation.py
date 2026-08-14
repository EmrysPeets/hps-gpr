#!/usr/bin/env python3
"""Run the targeted pull-blind residual length-ceiling remediation.

The only fit-launching commands are ``run-task`` and ``run-stage``.  Tasks are
single-worker and background-only.  Selection toys 0--2 must be collected into
a passing, hash-locked selection disposition before confirmation toys 3--7 can
run.  No command computes or writes signal-amplitude, pull, recovery, limit, or
coverage quantities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

import build_residual_length_ceiling_remediation_toys as toy_builder
import run_residual_length_pilot as legacy


HERE = Path(__file__).resolve().parent
DRIVER_PATH = Path(__file__).resolve()
ADDENDUM_PATH = HERE / "CEILING_REMEDIATION_ADDENDUM.json"
TOY_ROOT_DEFAULT = HERE / "inputs/residual_length_ceiling_remediation_toys.root"
TOY_MANIFEST_DEFAULT = (
    HERE / "inputs/residual_length_ceiling_remediation_toys.manifest.json"
)
RUNS = HERE / "runs/residual_length_ceiling_remediation"
DERIVED = HERE / "derived/residual_length_ceiling_remediation"
QA = HERE / "qa/residual_length_ceiling_remediation"

EXPECTED_ADDENDUM_SHA256 = (
    "40d81bca0ded24821d2f1213e3df9a6ab1c904242b0e89ea2ad5773533e5fb1d"
)
EXPECTED_LEGACY_RUNNER_SHA256 = (
    "205420bd293404bab08af2cb0230ad66c3dcc87dcfc627fdf3b5a13bd928dbd3"
)
MODEL = "knot_spline"
SOURCE_FAMILY = "one_pct"
SCENARIO = "2021_1pct"
MASS_GRID = (0.065, 0.120, 0.210)
UPPER_FACTORS = (25, 35, 50, 75)
CANDIDATE_SENTINEL = ((35, 50), (50, 75))
SELECTION_INDICES = (0, 1, 2)
CONFIRMATION_INDICES = (3, 4, 5, 6, 7)
ALL_TOY_INDICES = SELECTION_INDICES + CONFIRMATION_INDICES
SUPPORT_GEV = (0.04, 0.30)
OPTIMIZER_RESTARTS = 12
OPTIMIZER_SEED_NAMESPACE = (
    "v4p8p3_residual_length_ceiling_remediation_optimizer_v1"
)

TASK_PRODUCT_NAMES = (
    "optimizer_attempts.csv",
    "selected_trajectories.csv",
    "optimizer_exclusions.csv",
)
EXCLUSION_COLUMNS = (
    "addendum_id",
    "stage",
    "model",
    "scenario",
    "background_toy_index",
    "mass_GeV",
    "mass_MeV",
    "upper_factor",
    "n_attempts",
    "optimizer_gate_status",
    "reason",
    "maximum_lml_candidate_attempt",
    "maximum_lml_candidate",
    "maximum_lml_branch_replicates",
)


class RemediationError(RuntimeError):
    """Raised when the frozen remediation contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RemediationError(f"JSON root must be an object: {path}")
    return payload


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RemediationError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise RemediationError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}: {path}"
        )


def addendum() -> dict[str, Any]:
    require_hash(ADDENDUM_PATH, EXPECTED_ADDENDUM_SHA256, "frozen addendum")
    payload = toy_builder.load_addendum()
    target = payload["target_scope"]
    if (
        target["model"] != MODEL
        or target["source_family"] != SOURCE_FAMILY
        or target["scenario"] != SCENARIO
        or tuple(map(float, target["masses_gev"])) != MASS_GRID
        or tuple(map(int, target["upper_factors"])) != UPPER_FACTORS
        or tuple(
            (int(row["candidate"]), int(row["sentinel"]))
            for row in target["candidate_sentinel_pairs"]
        )
        != CANDIDATE_SENTINEL
        or target.get("fallback", "unexpected") is not None
    ):
        raise RemediationError("addendum target lattice drift")
    if not bool(
        payload["post_closure_initiation"]["original_closure_pulls_were_inspected"]
    ):
        raise RemediationError("post-closure initiation disclosure drift")
    return payload


def stage_indices(stage: str) -> tuple[int, ...]:
    if stage == "select":
        return SELECTION_INDICES
    if stage == "confirm":
        return CONFIRMATION_INDICES
    raise RemediationError(f"unsupported stage: {stage}")


def stage_for(toy_index: int) -> str:
    if toy_index in SELECTION_INDICES:
        return "select"
    if toy_index in CONFIRMATION_INDICES:
        return "confirm"
    raise RemediationError(f"toy index outside remediation lattice: {toy_index}")


def forbidden_tokens() -> set[str]:
    return set(map(str, addendum()["information_firewall"]["forbidden_output_column_substrings"]))


def validate_no_inference_columns(frame: pd.DataFrame, label: str) -> None:
    violations: dict[str, list[str]] = {}
    for column in frame.columns:
        normalized = "".join(
            character for character in str(column).lower() if character.isalnum()
        )
        matched = sorted(
            token
            for token in forbidden_tokens()
            if "".join(
                character for character in token.lower() if character.isalnum()
            )
            in normalized
        )
        if matched:
            violations[str(column)] = matched
    if violations:
        raise RemediationError(f"{label} has prohibited inference columns: {violations}")


def build_config(upper_factor: int) -> Any:
    """Reuse the audited factor-25 config and change only its 2021 ceiling."""

    if int(upper_factor) not in UPPER_FACTORS:
        raise RemediationError(f"unsupported remediation factor: {upper_factor}")
    core = legacy.load_v4p8_core()
    cfg = core.build_config(25)
    cfg.kernel_ls_res_upper_factor_by_dataset = dict(
        cfg.kernel_ls_res_upper_factor_by_dataset
    )
    cfg.kernel_ls_res_upper_factor_by_dataset["2021"] = float(upper_factor)
    core.assert_config(cfg, int(upper_factor))
    return cfg


def controlled_config_audit() -> dict[str, Any]:
    """Prove that remediation cards differ only in the declared 2021 ceiling."""

    core = legacy.load_v4p8_core()
    full_hashes: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    for factor in UPPER_FACTORS:
        payload = core._canonicalize(vars(build_config(factor)))
        full_hashes[str(factor)] = canonical_json_hash(payload)
        normalized = json.loads(json.dumps(payload, default=str))
        normalized["kernel_ls_res_upper_factor_by_dataset"]["2021"] = (
            "__CONTROLLED_UPPER_FACTOR__"
        )
        normalized_hashes[str(factor)] = canonical_json_hash(normalized)
    if len(set(normalized_hashes.values())) != 1:
        raise RemediationError(
            "remediation configurations differ beyond the controlled 2021 ceiling"
        )
    return {
        "controlled_field": "kernel_ls_res_upper_factor_by_dataset.2021",
        "full_config_sha256": full_hashes,
        "normalized_config_sha256": next(iter(normalized_hashes.values())),
        "only_controlled_field_differs": True,
    }


def ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = pd.Series(dtype="object")
    return output


def runtime_preflight() -> dict[str, Any]:
    require_hash(
        Path(legacy.__file__).resolve(),
        EXPECTED_LEGACY_RUNNER_SHA256,
        "audited original length-pilot runner",
    )
    record = legacy._runtime_preflight()
    source_record = legacy._source_fit_preflight(legacy.protocol())
    config_audit = controlled_config_audit()
    gate = legacy.optimizer_gate()
    optimizer_contract = addendum()["optimizer_contract"]
    if (
        int(gate["reference_initial_attempts"])
        != int(optimizer_contract["reference_initial_attempts"])
        or int(gate["maximum_attempts"])
        != int(optimizer_contract["maximum_attempts"])
        or int(gate["top_branch_min_replicates"])
        != int(optimizer_contract["top_branch_minimum_replicates"])
    ):
        raise RemediationError("audited optimizer topology differs from addendum")
    return {
        "legacy_runner_sha256": sha256_file(Path(legacy.__file__).resolve()),
        "runtime": record,
        "source_fit": source_record,
        "upper_factors_config_validated": list(UPPER_FACTORS),
        "controlled_one_factor_config_audit": config_audit,
        "optimizer_gate": gate,
    }


def validate_toy_inputs(root_path: Path, manifest_path: Path) -> dict[str, Any]:
    if root_path.resolve() != TOY_ROOT_DEFAULT.resolve() or manifest_path.resolve() != TOY_MANIFEST_DEFAULT.resolve():
        raise RemediationError("remediation inputs must use the frozen study-local paths")
    validation = toy_builder.validate()
    manifest = load_json(manifest_path)
    if manifest["root"]["sha256"] != sha256_file(root_path):
        raise RemediationError("remediation ROOT hash differs from manifest")
    if tuple(manifest["selection_toy_indices"]) != SELECTION_INDICES:
        raise RemediationError("selection toy inventory drift")
    if tuple(manifest["confirmation_toy_indices"]) != CONFIRMATION_INDICES:
        raise RemediationError("confirmation toy inventory drift")
    rows = manifest.get("toys", [])
    if len(rows) != len(ALL_TOY_INDICES):
        raise RemediationError("remediation toy manifest cardinality drift")
    return {
        "root_path": str(root_path),
        "root_sha256": sha256_file(root_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "builder_validation": validation,
    }


def scan_contract(
    root_path: Path,
    manifest_path: Path,
    toy_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    payload = addendum()
    return {
        "schema_version": 1,
        "study_id": payload["study_id"],
        "addendum_id": payload["addendum_id"],
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "driver_path": DRIVER_PATH.name,
        "driver_sha256": sha256_file(DRIVER_PATH),
        "legacy_runner_sha256": sha256_file(Path(legacy.__file__).resolve()),
        "toy_root": str(root_path),
        "toy_root_sha256": toy_inputs["root_sha256"],
        "toy_manifest": str(manifest_path),
        "toy_manifest_sha256": toy_inputs["manifest_sha256"],
        "model": MODEL,
        "source_family": SOURCE_FAMILY,
        "scenario": SCENARIO,
        "selection_toy_indices": list(SELECTION_INDICES),
        "confirmation_toy_indices": list(CONFIRMATION_INDICES),
        "masses_gev": list(MASS_GRID),
        "upper_factors": list(UPPER_FACTORS),
        "candidate_sentinel_pairs": [
            {"candidate": candidate, "sentinel": sentinel}
            for candidate, sentinel in CANDIDATE_SENTINEL
        ],
        "workers": 1,
        "blas_threads": 1,
        "optimizer_restarts": OPTIMIZER_RESTARTS,
        "optimizer_seed_namespace": OPTIMIZER_SEED_NAMESPACE,
        "optimizer_seed_excludes_upper_factor": True,
        "optimizer_gate": legacy.optimizer_gate(),
        "stage_gate": payload["stage_gate"],
        "background_only": True,
        "pull_blind": True,
        "fallback": None,
        "all_lane_confirmation": False,
        "closure_rerun": False,
        "prohibited_output_column_substrings": sorted(forbidden_tokens()),
        "inference_products_produced": False,
    }


def preflight(
    root_path: Path,
    manifest_path: Path,
    *,
    allow_missing_toys: bool = False,
) -> dict[str, Any]:
    payload = addendum()
    runtime = runtime_preflight()
    builder_record = toy_builder.preflight()
    missing = [str(path) for path in (root_path, manifest_path) if not path.is_file()]
    if missing:
        if not allow_missing_toys:
            raise RemediationError(f"missing remediation toy input(s): {missing}")
        return {
            "status": "waiting_for_fresh_toy_inputs",
            "mode": "read_only_preflight",
            "validated_utc": utc_now(),
            "missing_toy_inputs": missing,
            "addendum_sha256": sha256_file(ADDENDUM_PATH),
            "driver_sha256": sha256_file(DRIVER_PATH),
            "legacy_runner_sha256": runtime["legacy_runner_sha256"],
            "builder_preflight": builder_record,
            "expected_tasks": len(ALL_TOY_INDICES),
            "expected_states": len(ALL_TOY_INDICES)
            * len(MASS_GRID)
            * len(UPPER_FACTORS),
            "selection_fits_authorized": False,
            "confirmation_fits_authorized": False,
            "fits_launched": False,
            "claim_boundary": payload["claim_boundary"],
        }
    toy_inputs = validate_toy_inputs(root_path, manifest_path)
    contract = scan_contract(root_path, manifest_path, toy_inputs)
    return {
        "status": "pass",
        "mode": "read_only_preflight",
        "validated_utc": utc_now(),
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "driver_sha256": sha256_file(DRIVER_PATH),
        "runtime": runtime,
        "builder_preflight": builder_record,
        "toy_inputs": toy_inputs,
        "scan_contract_sha256": canonical_json_hash(contract),
        "expected_tasks": len(ALL_TOY_INDICES),
        "expected_states": len(ALL_TOY_INDICES)
        * len(MASS_GRID)
        * len(UPPER_FACTORS),
        "fits_launched": False,
        "claim_boundary": payload["claim_boundary"],
    }


def manifest_rows(manifest_path: Path) -> dict[int, Mapping[str, Any]]:
    rows = load_json(manifest_path).get("toys", [])
    output = {int(row["toy_index"]): row for row in rows}
    if set(output) != set(ALL_TOY_INDICES):
        raise RemediationError("remediation manifest toy-index drift")
    return output


def make_toy_dataset(
    toy_index: int,
    cfg: Any,
    root_path: Path,
    manifest_path: Path,
) -> Any:
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import (
        FuncFormToySpec,
        build_funcform_toy_dataset,
        load_funcform_toy_hist,
    )

    row = manifest_rows(manifest_path)[int(toy_index)]
    if str(row["stage"]) != stage_for(toy_index):
        raise RemediationError("toy stage differs from frozen index partition")
    key = str(row["key"])
    container, toy_name = key.rsplit("/", 1)
    histogram = load_funcform_toy_hist(
        str(root_path), container=container, toy_name=toy_name
    )
    base = make_datasets(cfg)["2021"]
    if (float(base.data_low), float(base.data_high)) != SUPPORT_GEV:
        raise RemediationError("v4.8 runtime support is not 40--300 MeV")
    toy_spec = FuncFormToySpec(
        source_root=str(root_path),
        container=container,
        function_tag="v4p8p3_knot_spline_native1pct_ceiling_remediation",
        toy_name=toy_name,
        toy_index=int(toy_index),
    )
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def fit_attempt(
    dataset: Any,
    cfg: Any,
    gate: Mapping[str, Any],
    toy_index: int,
    mass: float,
    upper_factor: int,
    attempt: int,
) -> dict[str, Any]:
    original_namespace = legacy.SEED_NAMESPACE
    try:
        legacy.SEED_NAMESPACE = OPTIMIZER_SEED_NAMESPACE
        row = legacy.fit_attempt(
            dataset,
            cfg,
            gate,
            MODEL,
            SCENARIO,
            int(toy_index),
            float(mass),
            int(upper_factor),
            int(attempt),
        )
    finally:
        legacy.SEED_NAMESPACE = original_namespace
    row = dict(row)
    row.update(
        {
            "addendum_id": addendum()["addendum_id"],
            "stage": stage_for(toy_index),
            "source_family": SOURCE_FAMILY,
            "post_closure_initiated": True,
            "optimizer_seed_namespace": OPTIMIZER_SEED_NAMESPACE,
            "seed_includes_upper_factor": False,
        }
    )
    return row


def task_directory(toy_index: int) -> Path:
    return RUNS / stage_for(toy_index) / f"toy_{toy_index:04d}"


def read_task_products(directory: Path) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for name in TASK_PRODUCT_NAMES:
        path = directory / name
        if not path.is_file():
            raise RemediationError(f"missing task product: {path}")
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame(
                columns=EXCLUSION_COLUMNS if "exclusion" in name else []
            )
        validate_no_inference_columns(frame, name)
        output[name] = frame
    return output


def selection_disposition_path() -> Path:
    return DERIVED / "selection" / "selection_disposition.json"


def final_disposition_path() -> Path:
    return DERIVED / "confirmation" / "final_disposition.json"


def load_selection_disposition() -> dict[str, Any]:
    path = selection_disposition_path()
    if not path.is_file():
        raise RemediationError(
            "confirmation is locked until selection_disposition.json exists"
        )
    payload = load_json(path)
    if (
        payload.get("status") != "selection_pass"
        or payload.get("selected_candidate") not in (35, 50)
        or payload.get("selected_sentinel") not in (50, 75)
        or payload.get("fallback_used") is not False
        or payload.get("stage") != "select"
        or tuple(payload.get("toy_indices", ())) != SELECTION_INDICES
        or payload.get("all_masses_passed") is not True
        or payload.get("confirmation_authorized") is not True
        or payload.get("background_only") is not True
        or payload.get("pull_blind") is not True
        or payload.get("inference_quantities_inspected") is not False
        or payload.get("addendum_sha256") != sha256_file(ADDENDUM_PATH)
        or payload.get("driver_sha256") != sha256_file(DRIVER_PATH)
    ):
        raise RemediationError("selection disposition is absent, stale, or failed")
    candidate = int(payload["selected_candidate"])
    sentinel = int(payload["selected_sentinel"])
    if (candidate, sentinel) not in CANDIDATE_SENTINEL:
        raise RemediationError("selection disposition pair is outside contract")
    if not TOY_ROOT_DEFAULT.is_file() or not TOY_MANIFEST_DEFAULT.is_file():
        raise RemediationError("selection disposition inputs are no longer present")
    root_hash = sha256_file(TOY_ROOT_DEFAULT)
    manifest_hash = sha256_file(TOY_MANIFEST_DEFAULT)
    if (
        payload.get("toy_root_sha256") != root_hash
        or payload.get("toy_manifest_sha256") != manifest_hash
    ):
        raise RemediationError("selection disposition toy provenance is stale")
    current_contract_hash = canonical_json_hash(
        scan_contract(
            TOY_ROOT_DEFAULT,
            TOY_MANIFEST_DEFAULT,
            {"root_sha256": root_hash, "manifest_sha256": manifest_hash},
        )
    )
    if payload.get("scan_contract_sha256") != current_contract_hash:
        raise RemediationError("selection disposition scan contract is stale")
    products = payload.get("product_sha256", {})
    expected_products = {
        "optimizer_attempt_ledger.csv",
        "selected_trajectory_ledger.csv",
        "optimizer_exclusion_ledger.csv",
        "task_product_audit.csv",
        "candidate_mass_gate.csv",
    }
    if set(products) != expected_products:
        raise RemediationError("selection disposition product inventory drift")
    for name, expected in products.items():
        product = path.parent / str(name)
        require_hash(product, str(expected), f"selection product {name}")
    return payload


def load_final_disposition() -> dict[str, Any]:
    path = final_disposition_path()
    if not path.is_file():
        raise RemediationError("final_disposition.json does not exist")
    payload = load_json(path)
    selection = load_selection_disposition()
    root_hash = sha256_file(TOY_ROOT_DEFAULT)
    manifest_hash = sha256_file(TOY_MANIFEST_DEFAULT)
    current_contract_hash = canonical_json_hash(
        scan_contract(
            TOY_ROOT_DEFAULT,
            TOY_MANIFEST_DEFAULT,
            {"root_sha256": root_hash, "manifest_sha256": manifest_hash},
        )
    )
    if (
        payload.get("status")
        not in ("qualified_targeted", "failed_targeted_confirmation")
        or payload.get("stage") != "confirm"
        or tuple(payload.get("toy_indices", ())) != CONFIRMATION_INDICES
        or payload.get("selected_candidate") != selection["selected_candidate"]
        or payload.get("selected_sentinel") != selection["selected_sentinel"]
        or payload.get("fallback_used") is not False
        or payload.get("background_only") is not True
        or payload.get("pull_blind") is not True
        or payload.get("inference_quantities_inspected") is not False
        or payload.get("all_lane_qualification") is not False
        or payload.get("closure_rerun_performed") is not False
        or payload.get("independent_confirmation_of_original_closure") is not False
        or payload.get("addendum_sha256") != sha256_file(ADDENDUM_PATH)
        or payload.get("driver_sha256") != sha256_file(DRIVER_PATH)
        or payload.get("toy_root_sha256") != root_hash
        or payload.get("toy_manifest_sha256") != manifest_hash
        or payload.get("scan_contract_sha256") != current_contract_hash
        or payload.get("selection_disposition_sha256")
        != sha256_file(selection_disposition_path())
    ):
        raise RemediationError("final disposition is stale or outside contract")
    products = payload.get("product_sha256", {})
    expected_products = {
        "optimizer_attempt_ledger.csv",
        "selected_trajectory_ledger.csv",
        "optimizer_exclusion_ledger.csv",
        "task_product_audit.csv",
        "candidate_mass_gate.csv",
    }
    if set(products) != expected_products:
        raise RemediationError("final disposition product inventory drift")
    for name, expected in products.items():
        require_hash(path.parent / str(name), str(expected), f"confirmation product {name}")
    return payload


def validate_success(
    toy_index: int,
    contract_hash: str,
) -> tuple[bool, str]:
    directory = task_directory(toy_index)
    marker_path = directory / "_SUCCESS.json"
    if not marker_path.is_file():
        return False, "missing_success_marker"
    try:
        marker = load_json(marker_path)
        if marker.get("status") != "complete":
            return False, "noncomplete_marker"
        if marker.get("scan_contract_sha256") != contract_hash:
            return False, "stale_contract"
        if (
            int(marker.get("background_toy_index", -1)) != int(toy_index)
            or marker.get("stage") != stage_for(toy_index)
            or marker.get("model") != MODEL
            or marker.get("scenario") != SCENARIO
        ):
            return False, "identity_mismatch"
        if marker.get("runner_sha256") != sha256_file(DRIVER_PATH):
            return False, "runner_hash_mismatch"
        if marker.get("addendum_sha256") != sha256_file(ADDENDUM_PATH):
            return False, "addendum_hash_mismatch"
        if stage_for(toy_index) == "confirm":
            disposition = load_selection_disposition()
            if marker.get("selection_disposition_sha256") != sha256_file(
                selection_disposition_path()
            ):
                return False, "selection_disposition_hash_mismatch"
            if int(marker.get("selected_candidate", -1)) != int(
                disposition["selected_candidate"]
            ):
                return False, "selected_candidate_mismatch"
        frames = read_task_products(directory)
        hashes = marker.get("product_sha256", {})
        for name in TASK_PRODUCT_NAMES:
            if hashes.get(name) != sha256_file(directory / name):
                return False, f"product_hash_mismatch:{name}"
        selected = frames["selected_trajectories.csv"]
        exclusions = frames["optimizer_exclusions.csv"]
        if len(selected) + len(exclusions) != len(MASS_GRID) * len(UPPER_FACTORS):
            return False, "state_cardinality_mismatch"
    except Exception as exc:
        return False, f"invalid_success:{type(exc).__name__}:{exc}"
    return True, "current"


def run_task(
    toy_index: int,
    root_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if toy_index not in ALL_TOY_INDICES:
        raise RemediationError("task toy index is outside the frozen lattice")
    stage = stage_for(toy_index)
    validation = preflight(root_path, manifest_path)
    toy_inputs = validation["toy_inputs"]
    contract = scan_contract(root_path, manifest_path, toy_inputs)
    contract_hash = canonical_json_hash(contract)
    selection_record: dict[str, Any] | None = None
    if stage == "confirm":
        selection_record = load_selection_disposition()
    current, reason = validate_success(toy_index, contract_hash)
    if current:
        return {**load_json(task_directory(toy_index) / "_SUCCESS.json"), "cached": True}
    directory = task_directory(toy_index)
    if directory.exists():
        raise RemediationError(
            f"refusing to overwrite stale/incomplete task {directory}: {reason}"
        )

    core = legacy.load_v4p8_core()
    gate = legacy.optimizer_gate()
    configs = {factor: build_config(factor) for factor in UPPER_FACTORS}
    dataset = make_toy_dataset(
        toy_index, configs[UPPER_FACTORS[0]], root_path, manifest_path
    )

    attempt_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for mass in MASS_GRID:
        by_factor: dict[int, list[dict[str, Any]]] = {
            factor: [] for factor in UPPER_FACTORS
        }
        initial_attempts = int(gate["reference_initial_attempts"])
        maximum_attempts = int(gate["maximum_attempts"])
        for attempt in range(initial_attempts):
            for factor in UPPER_FACTORS:
                by_factor[factor].append(
                    fit_attempt(
                        dataset,
                        configs[factor],
                        gate,
                        toy_index,
                        mass,
                        factor,
                        attempt,
                    )
                )
        initial_selected = {
            factor: core.select_branch(by_factor[factor], gate)[0]
            for factor in UPPER_FACTORS
        }
        if any(row is None for row in initial_selected.values()):
            for attempt in range(initial_attempts, maximum_attempts):
                for factor in UPPER_FACTORS:
                    by_factor[factor].append(
                        fit_attempt(
                            dataset,
                            configs[factor],
                            gate,
                            toy_index,
                            mass,
                            factor,
                            attempt,
                        )
                    )
        attempt_sets = {
            factor: tuple(int(row["attempt"]) for row in records)
            for factor, records in by_factor.items()
        }
        if len(set(attempt_sets.values())) != 1:
            raise RemediationError("factor attempt sets differ")
        for attempt in next(iter(attempt_sets.values())):
            seeds = {
                int(
                    next(
                        row
                        for row in by_factor[factor]
                        if int(row["attempt"]) == attempt
                    )["optimizer_seed"]
                )
                for factor in UPPER_FACTORS
            }
            if len(seeds) != 1:
                raise RemediationError("optimizer seeds differ across factors")

        for factor in UPPER_FACTORS:
            records = by_factor[factor]
            selected, replicates, top = core.select_branch(records, gate)
            for row in records:
                row["evaluated_attempt_count"] = len(records)
                row["maximum_lml_candidate_attempt"] = (
                    int(top["attempt"]) if top is not None else -1
                )
                row["matches_maximum_lml_branch"] = bool(
                    top is not None and core.branch_match(top, row, gate)
                )
                row["selected_maximum_lml_reproduced_branch"] = bool(
                    selected is not None
                    and int(row["attempt"]) == int(selected["attempt"])
                )
                row["maximum_lml_branch_replicates"] = int(replicates)
                attempt_rows.append(row)
            if selected is None:
                exclusion_rows.append(
                    {
                        "addendum_id": addendum()["addendum_id"],
                        "stage": stage,
                        "model": MODEL,
                        "scenario": SCENARIO,
                        "background_toy_index": toy_index,
                        "mass_GeV": mass,
                        "mass_MeV": int(round(1000 * mass)),
                        "upper_factor": factor,
                        "n_attempts": len(records),
                        "optimizer_gate_status": "excluded",
                        "reason": "maximum_lml_branch_not_reproduced_or_invalid",
                        "maximum_lml_candidate_attempt": (
                            int(top["attempt"]) if top is not None else -1
                        ),
                        "maximum_lml_candidate": (
                            float(top["gp_lml"]) if top is not None else float("nan")
                        ),
                        "maximum_lml_branch_replicates": int(replicates),
                    }
                )
                continue
            chosen = dict(selected)
            chosen.update(
                {
                    "addendum_id": addendum()["addendum_id"],
                    "stage": stage,
                    "selected_attempt": int(selected["attempt"]),
                    "n_attempts": len(records),
                    "top_branch_replicates": int(replicates),
                    "optimizer_gate_status": "maximum_lml_reproduced",
                    "support_preserved_40_300": True,
                    "common_seeds_across_factors": True,
                    "factor_selection_performed": False,
                    "inference_quantities_inspected": False,
                }
            )
            selected_rows.append(chosen)

    attempts = pd.DataFrame(attempt_rows).sort_values(
        ["mass_MeV", "attempt", "upper_factor"]
    )
    selected = ensure_columns(
        pd.DataFrame(selected_rows),
        (
            "addendum_id",
            "stage",
            "model",
            "scenario",
            "background_toy_index",
            "mass_GeV",
            "mass_MeV",
            "upper_factor",
            "fit_ok",
            "covariance_valid",
            "gp_lml",
            "ell_opt",
            "sigma_x",
            "n_train",
            "ell_at_upper_exact",
            "ell_near_upper",
            "top_branch_replicates",
        ),
    )
    if not selected.empty:
        selected = selected.sort_values(["mass_MeV", "upper_factor"])
    exclusions = ensure_columns(pd.DataFrame(exclusion_rows), EXCLUSION_COLUMNS)
    if not exclusions.empty:
        exclusions = exclusions.sort_values(["mass_MeV", "upper_factor"])
    for label, frame in (
        ("task attempts", attempts),
        ("task selected trajectories", selected),
        ("task exclusions", exclusions),
    ):
        validate_no_inference_columns(frame, label)
    if len(selected) + len(exclusions) != len(MASS_GRID) * len(UPPER_FACTORS):
        raise RemediationError("selected plus excluded task cardinality drift")

    directory.parent.mkdir(parents=True, exist_ok=True)
    work_directory = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent)
    )
    try:
        products = {
            "optimizer_attempts.csv": attempts,
            "selected_trajectories.csv": selected,
            "optimizer_exclusions.csv": exclusions,
        }
        for name, frame in products.items():
            frame.to_csv(work_directory / name, index=False)
        product_hashes = {
            name: sha256_file(work_directory / name) for name in TASK_PRODUCT_NAMES
        }
        marker = {
            "schema_version": 1,
            "generation_uuid": str(uuid.uuid4()),
            "status": "complete",
            "scientific_status": (
                "pull_blind_optimizer_diagnostic_complete"
                if exclusions.empty
                else "pull_blind_optimizer_diagnostic_has_exclusions"
            ),
            "completed_utc": utc_now(),
            "addendum_id": addendum()["addendum_id"],
            "addendum_sha256": sha256_file(ADDENDUM_PATH),
            "runner_sha256": sha256_file(DRIVER_PATH),
            "legacy_runner_sha256": sha256_file(Path(legacy.__file__).resolve()),
            "scan_contract_sha256": contract_hash,
            "model": MODEL,
            "source_family": SOURCE_FAMILY,
            "scenario": SCENARIO,
            "stage": stage,
            "background_toy_index": toy_index,
            "selected_rows": len(selected),
            "excluded_rows": len(exclusions),
            "attempt_rows": len(attempts),
            "product_sha256": product_hashes,
            "background_only": True,
            "pull_blind": True,
            "common_seeds_across_factors": True,
            "factor_selection_performed": False,
            "inference_quantities_inspected": False,
            "selected_candidate": (
                int(selection_record["selected_candidate"])
                if selection_record is not None
                else None
            ),
            "selected_sentinel": (
                int(selection_record["selected_sentinel"])
                if selection_record is not None
                else None
            ),
            "selection_disposition_sha256": (
                sha256_file(selection_disposition_path())
                if selection_record is not None
                else None
            ),
        }
        atomic_json(work_directory / "_SUCCESS.json", marker)
        final_inputs = validate_toy_inputs(root_path, manifest_path)
        final_contract = scan_contract(root_path, manifest_path, final_inputs)
        if canonical_json_hash(final_contract) != contract_hash:
            raise RemediationError("inputs or executable changed during task")
        os.replace(work_directory, directory)
    except Exception:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise
    return marker


def run_stage(
    stage: str,
    root_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    indices = stage_indices(stage)
    preflight(root_path, manifest_path)
    if stage == "confirm":
        load_selection_disposition()
    completed = []
    for toy_index in indices:
        completed.append(run_task(toy_index, root_path, manifest_path))
        print(f"PASS {stage}/toy_{toy_index:04d}", flush=True)
    return {
        "status": "complete",
        "stage": stage,
        "tasks": len(indices),
        "completed": len(completed),
        "workers": 1,
        "blas_threads": 1,
    }


def task_status(root_path: Path, manifest_path: Path) -> dict[str, Any]:
    validation = preflight(root_path, manifest_path)
    contract_hash = canonical_json_hash(
        scan_contract(root_path, manifest_path, validation["toy_inputs"])
    )
    records = []
    for toy_index in ALL_TOY_INDICES:
        current, reason = validate_success(toy_index, contract_hash)
        records.append(
            {
                "toy_index": toy_index,
                "stage": stage_for(toy_index),
                "current": current,
                "reason": reason,
            }
        )
    return {
        "status": "complete" if all(row["current"] for row in records) else "incomplete",
        "selection_current": sum(
            row["current"] and row["stage"] == "select" for row in records
        ),
        "selection_expected": len(SELECTION_INDICES),
        "confirmation_current": sum(
            row["current"] and row["stage"] == "confirm" for row in records
        ),
        "confirmation_expected": len(CONFIRMATION_INDICES),
        "tasks": records,
        "fits_launched": False,
    }


def collect_stage_frames(
    stage: str,
    root_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    validation = preflight(root_path, manifest_path)
    contract_hash = canonical_json_hash(
        scan_contract(root_path, manifest_path, validation["toy_inputs"])
    )
    attempts: list[pd.DataFrame] = []
    selected: list[pd.DataFrame] = []
    exclusions: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for toy_index in stage_indices(stage):
        current, reason = validate_success(toy_index, contract_hash)
        if not current:
            raise RemediationError(
                f"cannot collect {stage}; toy {toy_index} is not current: {reason}"
            )
        directory = task_directory(toy_index)
        frames = read_task_products(directory)
        attempts.append(frames["optimizer_attempts.csv"])
        selected.append(frames["selected_trajectories.csv"])
        if not frames["optimizer_exclusions.csv"].empty:
            exclusions.append(frames["optimizer_exclusions.csv"])
        marker = load_json(directory / "_SUCCESS.json")
        audit.append(
            {
                "stage": stage,
                "toy_index": toy_index,
                "success_marker_sha256": sha256_file(directory / "_SUCCESS.json"),
                **{
                    f"{Path(name).stem}_sha256": marker["product_sha256"][name]
                    for name in TASK_PRODUCT_NAMES
                },
            }
        )
    attempt_frame = pd.concat(attempts, ignore_index=True, sort=False)
    selected_frame = pd.concat(selected, ignore_index=True, sort=False)
    exclusion_frame = (
        pd.concat(exclusions, ignore_index=True, sort=False)
        if exclusions
        else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    )
    audit_frame = pd.DataFrame(audit)
    for label, frame in (
        (f"{stage} attempt collection", attempt_frame),
        (f"{stage} selected collection", selected_frame),
        (f"{stage} exclusion collection", exclusion_frame),
        (f"{stage} task audit", audit_frame),
    ):
        validate_no_inference_columns(frame, label)
    expected_states = len(stage_indices(stage)) * len(MASS_GRID) * len(UPPER_FACTORS)
    if len(selected_frame) + len(exclusion_frame) != expected_states:
        raise RemediationError(f"{stage} collected state cardinality drift")
    key = ["background_toy_index", "mass_GeV", "upper_factor"]
    if not selected_frame.empty and selected_frame.duplicated(key).any():
        raise RemediationError(f"{stage} selected collection has duplicate states")
    return attempt_frame, selected_frame, exclusion_frame, audit_frame, contract_hash


def candidate_mass_gates(
    selected: pd.DataFrame,
    exclusions: pd.DataFrame,
    stage: str,
    candidate_pairs: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    gate = addendum()["stage_gate"]
    expected_toys = len(stage_indices(stage))
    records: list[dict[str, Any]] = []
    for candidate, sentinel in candidate_pairs:
        for mass in MASS_GRID:
            mass_selected = selected[
                np.isclose(pd.to_numeric(selected["mass_GeV"]), mass)
            ].sort_values(["background_toy_index", "upper_factor"])
            mass_exclusions = (
                exclusions[
                    np.isclose(
                        pd.to_numeric(exclusions["mass_GeV"], errors="coerce"),
                        mass,
                    )
                ]
                if not exclusions.empty
                else exclusions
            )
            candidate_rows = selected[
                np.isclose(pd.to_numeric(selected["mass_GeV"]), mass)
                & (pd.to_numeric(selected["upper_factor"]) == candidate)
            ].sort_values("background_toy_index")
            sentinel_rows = selected[
                np.isclose(pd.to_numeric(selected["mass_GeV"]), mass)
                & (pd.to_numeric(selected["upper_factor"]) == sentinel)
            ].sort_values("background_toy_index")
            candidate_toys = tuple(
                map(int, candidate_rows["background_toy_index"].tolist())
            )
            sentinel_toys = tuple(
                map(int, sentinel_rows["background_toy_index"].tolist())
            )
            expected_indices = stage_indices(stage)
            pair_complete = (
                len(candidate_rows) == expected_toys
                and len(sentinel_rows) == expected_toys
                and candidate_toys == expected_indices
                and sentinel_toys == expected_indices
            )
            expected_all_identities = {
                (toy_index, factor)
                for toy_index in expected_indices
                for factor in UPPER_FACTORS
            }
            actual_all_identities = {
                (int(row.background_toy_index), int(row.upper_factor))
                for row in pd.concat(
                    [
                        mass_selected[["background_toy_index", "upper_factor"]],
                        mass_exclusions[["background_toy_index", "upper_factor"]],
                    ],
                    ignore_index=True,
                ).itertuples(index=False)
            }
            all_factor_lattice_complete = (
                len(mass_selected) + len(mass_exclusions)
                == expected_toys * len(UPPER_FACTORS)
                and actual_all_identities == expected_all_identities
            )
            invalid_states = len(mass_selected)
            exclusion_count = len(mass_exclusions)
            minimum_repeats = 0
            if not mass_selected.empty:
                all_valid_mask = (
                    mass_selected["fit_ok"].astype(bool)
                    & mass_selected["covariance_valid"].astype(bool)
                    & np.isfinite(
                        pd.to_numeric(mass_selected["gp_lml"], errors="coerce")
                    )
                    & np.isfinite(
                        pd.to_numeric(mass_selected["ell_opt"], errors="coerce")
                    )
                    & np.isfinite(
                        pd.to_numeric(mass_selected["sigma_x"], errors="coerce")
                    )
                    & (pd.to_numeric(mass_selected["sigma_x"], errors="coerce") > 0)
                    & (pd.to_numeric(mass_selected["n_train"], errors="coerce") > 0)
                )
                invalid_states = int((~all_valid_mask).sum())
                minimum_repeats = int(
                    pd.to_numeric(
                        mass_selected["top_branch_replicates"], errors="coerce"
                    ).min()
                )
            contact_count = 0
            maximum_lml = float("inf")
            median_ell = float("inf")
            p95_ell = float("inf")
            maximum_ell = float("inf")
            sigma_equal = False
            n_train_equal = False
            if pair_complete:
                combined = pd.concat(
                    [candidate_rows, sentinel_rows], ignore_index=True
                )
                contact_count = int(
                    (
                        combined["ell_at_upper_exact"].astype(bool)
                        | combined["ell_near_upper"].astype(bool)
                    ).sum()
                )
                sigma_candidate = pd.to_numeric(candidate_rows["sigma_x"]).to_numpy(float)
                sigma_sentinel = pd.to_numeric(sentinel_rows["sigma_x"]).to_numpy(float)
                sigma_equal = bool(
                    np.allclose(
                        sigma_candidate,
                        sigma_sentinel,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                candidate_n_train = pd.to_numeric(
                    candidate_rows["n_train"]
                ).to_numpy(int)
                sentinel_n_train = pd.to_numeric(
                    sentinel_rows["n_train"]
                ).to_numpy(int)
                n_train_equal = bool(
                    np.array_equal(candidate_n_train, sentinel_n_train)
                )
                n_train = np.maximum(1, candidate_n_train)
                lml_delta = np.abs(
                    pd.to_numeric(sentinel_rows["gp_lml"]).to_numpy(float)
                    - pd.to_numeric(candidate_rows["gp_lml"]).to_numpy(float)
                ) / n_train
                ell_delta = np.abs(
                    pd.to_numeric(sentinel_rows["ell_opt"]).to_numpy(float)
                    - pd.to_numeric(candidate_rows["ell_opt"]).to_numpy(float)
                ) / sigma_candidate
                maximum_lml = float(np.max(lml_delta))
                median_ell = float(np.median(ell_delta))
                p95_ell = float(
                    np.quantile(ell_delta, 0.95, method="linear")
                )
                maximum_ell = float(np.max(ell_delta))
            gates = {
                "pair_complete": pair_complete,
                "all_factor_state_lattice_complete": all_factor_lattice_complete,
                "candidate_and_sentinel_contacts_zero": contact_count
                <= int(gate["candidate_and_sentinel_exact_or_near_upper_contacts"]),
                "optimizer_exclusions_zero": exclusion_count
                <= int(gate["optimizer_exclusions"]),
                "invalid_selected_states_zero": invalid_states
                <= int(gate["invalid_selected_states"]),
                "minimum_repeats_pass": minimum_repeats
                >= int(gate["minimum_top_branch_replicates"]),
                "sigma_x_equal_across_pair": sigma_equal,
                "n_train_equal_across_pair": n_train_equal,
                "maximum_lml_pass": maximum_lml
                <= float(gate["maximum_abs_delta_lml_per_training_bin"]),
                "median_ell_pass": median_ell
                <= float(gate["median_abs_delta_ell_over_sigma_x"]),
                "p95_ell_pass": p95_ell
                <= float(gate["p95_abs_delta_ell_over_sigma_x"]),
                "maximum_ell_pass": maximum_ell
                <= float(gate["maximum_abs_delta_ell_over_sigma_x"]),
            }
            records.append(
                {
                    "stage": stage,
                    "candidate": candidate,
                    "sentinel": sentinel,
                    "mass_GeV": mass,
                    "mass_MeV": int(round(1000 * mass)),
                    "n_toys": expected_toys,
                    "candidate_rows": len(candidate_rows),
                    "sentinel_rows": len(sentinel_rows),
                    "contact_count_candidate_and_sentinel": contact_count,
                    "exclusion_count_all_factors": exclusion_count,
                    "invalid_selected_states_all_factors": invalid_states,
                    "minimum_top_branch_replicates_all_factors": minimum_repeats,
                    "maximum_abs_delta_lml_per_training_bin": maximum_lml,
                    "median_abs_delta_ell_over_sigma_x": median_ell,
                    "p95_abs_delta_ell_over_sigma_x": p95_ell,
                    "maximum_abs_delta_ell_over_sigma_x": maximum_ell,
                    **gates,
                    "mass_gate_passed": all(gates.values()),
                }
            )
    frame = pd.DataFrame(records).sort_values(["candidate", "mass_GeV"])
    validate_no_inference_columns(frame, f"{stage} candidate mass gates")
    return frame


def write_stage_products(
    stage: str,
    attempts: pd.DataFrame,
    selected: pd.DataFrame,
    exclusions: pd.DataFrame,
    audit: pd.DataFrame,
    mass_gates: pd.DataFrame,
) -> tuple[Path, dict[str, str]]:
    directory = DERIVED / stage
    products = {
        "optimizer_attempt_ledger.csv": attempts,
        "selected_trajectory_ledger.csv": selected,
        "optimizer_exclusion_ledger.csv": exclusions,
        "task_product_audit.csv": audit,
        "candidate_mass_gate.csv": mass_gates,
    }
    for name, frame in products.items():
        validate_no_inference_columns(frame, f"{stage}/{name}")
        atomic_csv(directory / name, frame)
    return directory, {
        name: sha256_file(directory / name) for name in products
    }


def collect_selection(
    root_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    attempts, selected, exclusions, audit, contract_hash = collect_stage_frames(
        "select", root_path, manifest_path
    )
    gates = candidate_mass_gates(
        selected, exclusions, "select", CANDIDATE_SENTINEL
    )
    directory, product_hashes = write_stage_products(
        "selection", attempts, selected, exclusions, audit, gates
    )
    selected_pair: tuple[int, int] | None = None
    for pair in CANDIDATE_SENTINEL:
        rows = gates[gates["candidate"] == pair[0]]
        if len(rows) == len(MASS_GRID) and bool(rows["mass_gate_passed"].all()):
            selected_pair = pair
            break
    result = {
        "schema_version": 1,
        "status": "selection_pass" if selected_pair is not None else "selection_fail_no_candidate",
        "scientific_status": "targeted_pull_blind_selection_only",
        "completed_utc": utc_now(),
        "addendum_id": addendum()["addendum_id"],
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "driver_sha256": sha256_file(DRIVER_PATH),
        "legacy_runner_sha256": sha256_file(Path(legacy.__file__).resolve()),
        "scan_contract_sha256": contract_hash,
        "toy_root_sha256": sha256_file(root_path),
        "toy_manifest_sha256": sha256_file(manifest_path),
        "stage": "select",
        "toy_indices": list(SELECTION_INDICES),
        "selected_candidate": selected_pair[0] if selected_pair else None,
        "selected_sentinel": selected_pair[1] if selected_pair else None,
        "candidate_order": [pair[0] for pair in CANDIDATE_SENTINEL],
        "fallback_used": False,
        "all_masses_passed": selected_pair is not None,
        "confirmation_authorized": selected_pair is not None,
        "background_only": True,
        "pull_blind": True,
        "inference_quantities_inspected": False,
        "post_closure_initiated": True,
        "product_sha256": product_hashes,
        "claim_boundary": addendum()["claim_boundary"],
    }
    atomic_json(directory / "selection_disposition.json", result)
    return result


def collect_confirmation(
    root_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    selection = load_selection_disposition()
    candidate = int(selection["selected_candidate"])
    sentinel = int(selection["selected_sentinel"])
    attempts, selected, exclusions, audit, contract_hash = collect_stage_frames(
        "confirm", root_path, manifest_path
    )
    gates = candidate_mass_gates(
        selected, exclusions, "confirm", ((candidate, sentinel),)
    )
    directory, product_hashes = write_stage_products(
        "confirmation", attempts, selected, exclusions, audit, gates
    )
    passed = len(gates) == len(MASS_GRID) and bool(gates["mass_gate_passed"].all())
    result = {
        "schema_version": 1,
        "status": "qualified_targeted" if passed else "failed_targeted_confirmation",
        "scientific_status": (
            "targeted_knot_spline_native_1pct_ceiling_qualified"
            if passed
            else "targeted_knot_spline_native_1pct_ceiling_not_qualified"
        ),
        "completed_utc": utc_now(),
        "addendum_id": addendum()["addendum_id"],
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "driver_sha256": sha256_file(DRIVER_PATH),
        "legacy_runner_sha256": sha256_file(Path(legacy.__file__).resolve()),
        "scan_contract_sha256": contract_hash,
        "toy_root_sha256": sha256_file(root_path),
        "toy_manifest_sha256": sha256_file(manifest_path),
        "selection_disposition_sha256": sha256_file(selection_disposition_path()),
        "stage": "confirm",
        "toy_indices": list(CONFIRMATION_INDICES),
        "selected_candidate": candidate,
        "selected_sentinel": sentinel,
        "fallback_used": False,
        "all_masses_passed": passed,
        "background_only": True,
        "pull_blind": True,
        "inference_quantities_inspected": False,
        "post_closure_initiated": True,
        "all_lane_qualification": False,
        "closure_rerun_performed": False,
        "independent_confirmation_of_original_closure": False,
        "product_sha256": product_hashes,
        "claim_boundary": addendum()["claim_boundary"],
    }
    atomic_json(directory / "final_disposition.json", result)
    return result


def prepare(root_path: Path, manifest_path: Path) -> dict[str, Any]:
    validation = preflight(root_path, manifest_path)
    contract = scan_contract(root_path, manifest_path, validation["toy_inputs"])
    confirmation_authorized = False
    if selection_disposition_path().is_file():
        load_selection_disposition()
        confirmation_authorized = True
    rows = [
        {
            "toy_index": toy_index,
            "stage": stage_for(toy_index),
            "task_directory": str(task_directory(toy_index).relative_to(HERE)),
            "fit_launch_authorized": (
                stage_for(toy_index) == "select"
                or confirmation_authorized
            ),
        }
        for toy_index in ALL_TOY_INDICES
    ]
    QA.mkdir(parents=True, exist_ok=True)
    atomic_json(QA / "scan_contract.json", contract)
    atomic_csv(QA / "task_manifest.csv", pd.DataFrame(rows))
    return {
        "status": "pass",
        "scan_contract_sha256": canonical_json_hash(contract),
        "scan_contract_file_sha256": sha256_file(QA / "scan_contract.json"),
        "task_manifest_sha256": sha256_file(QA / "task_manifest.csv"),
        "selection_tasks": len(SELECTION_INDICES),
        "confirmation_tasks": len(CONFIRMATION_INDICES),
        "fits_launched": False,
    }


def validate_command(
    root_path: Path,
    manifest_path: Path,
    *,
    allow_missing_toys: bool,
) -> dict[str, Any]:
    validation = preflight(
        root_path, manifest_path, allow_missing_toys=allow_missing_toys
    )
    if validation["status"] != "pass":
        return validation
    status = task_status(root_path, manifest_path)
    selection_record = None
    confirmation_record = None
    if selection_disposition_path().is_file():
        selection_record = load_selection_disposition()
    final_path = final_disposition_path()
    if final_path.is_file():
        confirmation_record = load_final_disposition()
    return {
        "status": "pass",
        "preflight": validation,
        "task_status": status,
        "selection_disposition": selection_record,
        "confirmation_disposition": confirmation_record,
        "fits_launched": False,
    }


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--toy-root", type=Path, default=TOY_ROOT_DEFAULT)
    parser.add_argument("--toy-manifest", type=Path, default=TOY_MANIFEST_DEFAULT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    add_input_arguments(preflight_parser)
    preflight_parser.add_argument("--allow-missing-toys", action="store_true")
    prepare_parser = subparsers.add_parser("prepare")
    add_input_arguments(prepare_parser)
    task_parser = subparsers.add_parser("run-task")
    add_input_arguments(task_parser)
    task_parser.add_argument("toy_index", type=int, choices=ALL_TOY_INDICES)
    stage_parser = subparsers.add_parser("run-stage")
    add_input_arguments(stage_parser)
    stage_parser.add_argument("stage", choices=("select", "confirm"))
    status_parser = subparsers.add_parser("status")
    add_input_arguments(status_parser)
    collect_parser = subparsers.add_parser("collect")
    add_input_arguments(collect_parser)
    collect_parser.add_argument("stage", choices=("select", "confirm"))
    validate_parser = subparsers.add_parser("validate")
    add_input_arguments(validate_parser)
    validate_parser.add_argument("--allow-missing-toys", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_path = args.toy_root.resolve()
    manifest_path = args.toy_manifest.resolve()
    if args.command == "preflight":
        result = preflight(
            root_path,
            manifest_path,
            allow_missing_toys=args.allow_missing_toys,
        )
    elif args.command == "prepare":
        result = prepare(root_path, manifest_path)
    elif args.command == "run-task":
        result = run_task(args.toy_index, root_path, manifest_path)
    elif args.command == "run-stage":
        result = run_stage(args.stage, root_path, manifest_path)
    elif args.command == "status":
        result = task_status(root_path, manifest_path)
    elif args.command == "collect":
        result = (
            collect_selection(root_path, manifest_path)
            if args.stage == "select"
            else collect_confirmation(root_path, manifest_path)
        )
    elif args.command == "validate":
        result = validate_command(
            root_path,
            manifest_path,
            allow_missing_toys=args.allow_missing_toys,
        )
    else:
        raise RemediationError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
