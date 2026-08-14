#!/usr/bin/env python3
"""Run the v4.8 rigid-truth 2021 kernel-length diagnostic.

This is a background-only optimizer diagnostic for the already-built
``rigid_ggt26_scaled1pct`` pseudoexperiments.  It evaluates exactly four
scenarios, toys 0--19, masses 50--250 MeV in 20 MeV steps, and 2021
resolution-scaled length upper factors 15, 20, and 25.  The GP support remains
40--300 MeV and every other frozen-card setting remains fixed.  Product toys
20--24 are reserve toys: they are hash/inventory checked but no fit task may
consume them.

The three factors for a (scenario, toy, mass) state are always evaluated with
the same optimizer seeds and the same number of attempts.  The selected state
is the maximum-LML branch only when that branch is independently reproduced
under a reduced, length-only, pull-blind gate using LML, length scale, kernel
constant, and covariance validity.  It reuses the v4.7 numerical thresholds
and 3-to-5 attempt topology but is not the unchanged v4.7 gate.

Hard interpretation boundary
----------------------------
This program MUST NOT compute or select on pulls, signal recovery, observed
amplitudes, p0, CLs, epsilon-squared, limits, or coverage.  It MUST NOT choose
a length-factor/card setting.  Its products are optimizer trajectories,
bound-occupancy counts, and nested-LML checks only.  A factor choice requires
a separate, predeclared review.

The task unit is one (scenario, toy) containing all three factors.  This makes
the common-seed contract structural rather than conventional and gives 80
resumable tasks.  ``preflight``, ``validate``, ``prepare``, ``status``, and
``collect`` never launch GP fits.  Only ``run-task`` and ``run`` do so.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Pin numerical libraries before importing numpy/scipy/sklearn through the
# study-local runtime.  Reproducibility is more important than implicit BLAS
# parallelism; task-level parallelism is explicit in the CLI.
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


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME_ROOT = HERE / "runtime_overlay"
CORE_PATH = Path(__file__).resolve()
LAUNCHER_PATH = HERE / "run_rigid_length_scan.py"
LOCK_PATH = HERE / "rigid_length_scan_lock.json"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

SPEC_PATH = HERE / "rigid_study_spec.json"
RUNS = HERE / "runs/rigid_length_scan"
DERIVED = HERE / "derived/rigid_length_scan"
QA = HERE / "qa/rigid_length_scan"

SCENARIOS = (
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
UPPER_FACTORS = (15, 20, 25)
MASS_MEV = tuple(range(50, 251, 20))
MASS_GRID = tuple(value / 1000.0 for value in MASS_MEV)
ACTIVE_N_TOYS = 20
PRODUCT_N_TOYS = 25
TOY_INDICES = tuple(range(ACTIVE_N_TOYS))
RESERVED_TOY_INDICES = tuple(range(ACTIVE_N_TOYS, PRODUCT_N_TOYS))
PRODUCT_TOY_INDICES = tuple(range(PRODUCT_N_TOYS))
TOY_CONTAINER_PREFIX = "toys/rigid_ggt26_scaled1pct"
SUPPORT_GEV = (0.04, 0.30)
SEARCH_GEV = (0.05, 0.25)
BASE_SEED = 20260813
SEED_NAMESPACE = "v4p8_rigid_length_scan_common_v1"
OPTIMIZER_RESTARTS = 12
EXACT_BOUND_RATIO = 0.999
STRICT_NESTED_LML_TOLERANCE = 1.0e-4

EXPECTED_CARD_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
EXPECTED_ROOT_SHA256 = (
    "216f500792645d6ab3d10699b3f48ca6f3221a3f6814d29b4e01565a53546d32"
)
EXPECTED_MANIFEST_SHA256 = (
    "3d8f981d87a33d01ad1591a6ebeac8c47e8d7351a3ebcd9eab973ee9853eba03"
)
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "667390be8c2c5b79578c4ca933ff94fad289146432859f62ebf851a128a6c2e6"
)
EXPECTED_CLOSURE_DRIVER_SHA256 = (
    "fb63c11517374cf1d6802dc8877412fb402b9ba0797f8d3bed6777ce96fcd887"
)
EXPECTED_RUNTIME_SHA256 = {
    "hps_gpr/gpr.py": "1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5",
    "hps_gpr/io.py": "b36f8da7671a0fc0958b663e11d83a1a4421e90d1aab9b10e40c31ce078035db",
    "hps_gpr/injection.py": "3a38378379650b73159de8b98456a2bd91e5c374794805b0be39e86557e26bf2",
    "hps_gpr/statistics.py": "b8cbd484056925d64bed4d9a4ad3294fbac07d51079e5cb9ed565150b73c1ff2",
    "hps_gpr/template.py": "20c1fbaa632d5e03fa7527d0e4ddf8dc3ba8573927a8f981936721a731440e3e",
    "hps_gpr/config.py": "ec4f50345aebbf5c062e8daaefaaeca9b0e96df12f12b2d726172979df61cf9d",
}

PRODUCT_NAMES = (
    "optimizer_attempts.csv",
    "raw_ell_sigma_x_trajectories.csv",
    "optimizer_exclusions.csv",
    "bound_occupancy.csv",
    "nested_lml.csv",
)
FORBIDDEN_OUTPUT_COLUMN_SUBSTRINGS = {
    "sigmaa",
    "amplitude",
    "ahat",
    "aup",
    "pull",
    "zhat",
    "recovery",
    "p0",
    "pvalue",
    "cls",
    "qmu",
    "eps2",
    "epsilon",
    "limit",
    "coverage",
    "signal_yield",
    "upper_limit",
}
EXCLUSION_COLUMNS = (
    "study_id",
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


class StudyError(RuntimeError):
    """Raised when a frozen diagnostic contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_spec() -> dict[str, Any]:
    if not SPEC_PATH.is_file():
        raise StudyError(f"missing rigid study specification: {SPEC_PATH}")
    payload = load_json(SPEC_PATH)
    if int(payload.get("schema_version", -1)) != 1:
        raise StudyError("unsupported rigid study specification schema")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(namespace: str, *parts: object) -> int:
    material = "|".join(
        [str(BASE_SEED), str(namespace), *[str(part) for part in parts]]
    )
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:4], "little"
    )


def stable_seed_words(namespace: str, *parts: object) -> list[int]:
    material = "|".join(
        [str(BASE_SEED), str(namespace), *[str(part) for part in parts]]
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()[:16]
    return [
        int.from_bytes(digest[index : index + 4], "little")
        for index in range(0, 16, 4)
    ]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
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
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
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


def resolve_study_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else HERE / path


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise StudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise StudyError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}: {path}"
        )


def verify_external_lock() -> dict[str, Any]:
    injected_core_hash = str(
        globals().get("__LENGTH_SCAN_EXECUTED_CORE_SHA256__", "")
    )
    injected_lock_hash = str(
        globals().get("__LENGTH_SCAN_EXTERNAL_LOCK_SHA256__", "")
    )
    injected_launcher_hash = str(
        globals().get("__LENGTH_SCAN_LAUNCHER_SHA256__", "")
    )
    injected_launcher_path = str(
        globals().get("__LENGTH_SCAN_LAUNCHER_PATH__", "")
    )
    if not all(
        (
            injected_core_hash,
            injected_lock_hash,
            injected_launcher_hash,
            injected_launcher_path,
        )
    ):
        raise StudyError(
            "direct core execution is forbidden; invoke the immutable "
            "run_rigid_length_scan.py launcher"
        )
    if Path(injected_launcher_path).resolve() != LAUNCHER_PATH.resolve():
        raise StudyError("injected length-scan launcher path mismatch")
    require_hash(LOCK_PATH, injected_lock_hash, "external length-scan lock")
    require_hash(CORE_PATH, injected_core_hash, "executed length-scan core")
    require_hash(
        LAUNCHER_PATH,
        injected_launcher_hash,
        "executing length-scan launcher",
    )
    lock = load_json(LOCK_PATH)
    if int(lock.get("schema_version", -1)) != 1:
        raise StudyError("unsupported external length-scan lock schema")
    if str(lock.get("lock_type")) != "immutable_launcher_to_core_v1":
        raise StudyError("external length-scan lock type drift")

    expected_records = {
        "driver": CORE_PATH,
        "study_spec": SPEC_PATH,
        "closure_driver": HERE / "run_rigid_study.py",
        "runtime_manifest": HERE / "runtime_overlay_manifest.json",
        "toy_root": HERE / "inputs/rigid_ggt26_scaled1pct_nested_toys_25.root",
        "toy_manifest": HERE
        / "inputs/rigid_ggt26_scaled1pct_nested_toys_25.manifest.json",
        "analysis_card": HERE / "inputs/frozen_v4p2_analysis_card.yaml",
        "rigid_generator": HERE / "rigid_generator_spec.json",
    }
    for name, expected_path in expected_records.items():
        record = lock.get(name)
        if not isinstance(record, Mapping):
            raise StudyError(f"external lock record is missing: {name}")
        locked_path = resolve_study_path(str(record.get("path", ""))).resolve()
        if locked_path != expected_path.resolve():
            raise StudyError(f"external lock path drift: {name}")
        locked_hash = str(record.get("sha256", ""))
        require_hash(expected_path, locked_hash, f"external lock {name}")
    if str(lock["driver"]["sha256"]) != injected_core_hash:
        raise StudyError("executed core hash does not match external lock")

    source_records = lock.get("source_inputs")
    if not isinstance(source_records, Mapping) or set(source_records) != {
        "one_pct",
        "ten_pct",
    }:
        raise StudyError("external lock source-input inventory drift")
    for family, record in source_records.items():
        if not isinstance(record, Mapping):
            raise StudyError(f"external lock source record is invalid: {family}")
        path = Path(str(record.get("path", ""))).resolve()
        require_hash(path, str(record.get("sha256", "")), f"locked {family} source")

    return {
        "lock_sha256": injected_lock_hash,
        "core_sha256": injected_core_hash,
        "launcher_sha256": injected_launcher_hash,
        "launcher_path": str(LAUNCHER_PATH),
        "trust_chain": (
            "launcher hardcodes lock SHA-256; lock pins executed core and all "
            "scientific inputs; launcher executes the already-verified core bytes"
        ),
    }


def _paired_path_key(record: Mapping[str, Any], hash_key: str) -> str | None:
    if hash_key == "sha256":
        for candidate in (
            "archived_path",
            "path",
            "root",
            "manifest",
            "metadata",
            "file",
            "production_driver",
        ):
            if candidate in record:
                return candidate
        return None
    if not hash_key.endswith("_sha256"):
        return None
    stem = hash_key[: -len("_sha256")]
    for candidate in (stem, f"{stem}_path", f"archived_{stem}_path"):
        if candidate in record:
            return candidate
    if stem == "config" and "archived_config_path" in record:
        return "archived_config_path"
    return None


def verify_declared_hashes(
    payload: Any,
    *,
    label: str,
    checks: dict[str, bool],
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            path_key = _paired_path_key(payload, str(key))
            if path_key is None:
                continue
            declared_path = payload.get(path_key)
            if not isinstance(declared_path, str) or not declared_path:
                continue
            path = resolve_study_path(declared_path)
            check_name = f"{label}.{path_key}"
            require_hash(path, value, check_name)
            checks[check_name] = True
        for key, value in payload.items():
            if isinstance(value, (Mapping, list, tuple)):
                verify_declared_hashes(
                    value, label=f"{label}.{key}", checks=checks
                )
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            if isinstance(value, (Mapping, list, tuple)):
                verify_declared_hashes(
                    value, label=f"{label}[{index}]", checks=checks
                )


def background_root(spec: Mapping[str, Any]) -> Path:
    value = spec.get("background_toy_product", {}).get("root")
    if not isinstance(value, str) or not value:
        raise StudyError("background_toy_product.root is missing")
    return resolve_study_path(value)


def toy_key(scenario: str, toy_index: int) -> str:
    return f"{TOY_CONTAINER_PREFIX}/{scenario}/toy_{int(toy_index):04d}"


def assert_reserve_outputs_untouched() -> None:
    touched: list[str] = []
    for scenario in SCENARIOS:
        for toy_index in RESERVED_TOY_INDICES:
            directory = RUNS / scenario / f"toy_{toy_index:04d}"
            if directory.is_dir() and any(path.is_file() for path in directory.rglob("*")):
                touched.append(str(directory))
    if touched:
        raise StudyError(
            "reserved toys 20--24 have length-scan products and are no longer "
            f"untouched: {touched[:8]}"
        )


def assert_spec_contract(spec: Mapping[str, Any]) -> None:
    scenarios = spec.get("scenarios", {})
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIOS):
        raise StudyError(f"scenario set must be exactly {SCENARIOS}")
    expected_scenario_semantics = {
        "2021_1pct_x10": ("one_pct", 10, 125_040_440),
        "2021_1pct_x100": ("one_pct", 100, 1_250_404_400),
        "2021_10pct": ("ten_pct", 1, 141_251_508),
        "2021_10pct_x10": ("ten_pct", 10, 1_412_515_080),
    }
    for scenario, (family, multiplier, target_count) in (
        expected_scenario_semantics.items()
    ):
        record = scenarios.get(scenario, {})
        if (
            str(record.get("source_family")) != family
            or int(record.get("source_multiplier", -1)) != multiplier
            or int(record.get("normalization_target_count", -1)) != target_count
            or str(record.get("function_tag")) != "rigid_ggt26_scaled1pct"
        ):
            raise StudyError(f"scenario semantics drift for {scenario}")
    if tuple(int(value) for value in spec.get("toy_indices", ())) != TOY_INDICES:
        raise StudyError("rigid study analyzed toy indices must be 0--19")

    seed_contract = spec.get("seed_contract", {})
    if seed_contract != {
        "optimizer_namespace": "v4p7_restart_v1",
        "signal_namespace": "v4p7_signal_v1",
        "nested_poisson_namespace": "nested_poisson",
        "base_seed": BASE_SEED,
    }:
        raise StudyError("seed-contract drift")

    expected_card = {
        "search_range_gev": [0.05, 0.25],
        "gp_support_range_gev": [0.04, 0.30],
        "pre_log": True,
        "alpha_model": "1/y",
        "neighborhood_rebin": 5,
        "blind_nsigma": 2.25,
        "gp_train_exclude_nsigma": 2.25,
        "kernel_ls_res_lower_factor_2021": 1.1,
        "kernel_ls_res_upper_factor_2021": 15.0,
        "n_restarts": 12,
        "injection_reference": "matched_refit_bonly",
        "injection_background_mode": "fixed_hist",
        "injection_mode": "poisson",
        "refit_gp_on_toy": True,
        "refit_gp_optimize": True,
        "test_statistic": "tilde_q_mu",
        "cls_alpha": 0.1,
        "confidence_level_percent": 90,
    }
    if spec.get("analysis_card", {}) != expected_card:
        raise StudyError("analysis-card declaration drift")

    rigid = spec.get("rigid_generator", {})
    if (
        str(rigid.get("status")) != "reviewed_conditional_stress_generator"
        or tuple(map(float, rigid.get("selection_region_gev", ())))
        != SEARCH_GEV
        or [list(map(float, pair)) for pair in rigid.get("support_shoulders_gev", ())]
        != [[0.04, 0.05], [0.25, 0.30]]
        or bool(rigid.get("shape_refit_in_native_10pct", True))
    ):
        raise StudyError("rigid-generator edge or transfer policy drift")
    rigid_path = rigid.get("path")
    rigid_sha = rigid.get("sha256")
    if not isinstance(rigid_path, str) or not isinstance(rigid_sha, str):
        raise StudyError("rigid-generator path/hash declaration is missing")
    resolved_rigid = resolve_study_path(rigid_path)
    require_hash(resolved_rigid, rigid_sha, "rigid generator specification")
    rigid_payload = load_json(resolved_rigid)
    if (
        str(rigid_payload.get("generator_tag"))
        != "rigid_ggt26_scaled1pct"
        or str(rigid_payload.get("status"))
        != "reviewed_conditional_stress_generator"
        or tuple(map(float, rigid_payload.get("support_gev", ())))
        != SUPPORT_GEV
        or tuple(
            map(
                float,
                rigid_payload.get(
                    "primary_search_region_metrics_0p125mev_bins", {}
                ).get("region_gev", ()),
            )
        )
        != SEARCH_GEV
        or str(rigid_payload.get("support30_status"))
        != "rejected_for_this_family"
        or bool(rigid_payload.get("kernel_ceiling_selection_allowed", True))
        or bool(
            rigid_payload.get("signal_absorption_policy", {}).get(
                "native_10pct_shape_refit_allowed", True
            )
        )
    ):
        raise StudyError("rigid-generator scientific contract drift")

    product = spec.get("background_toy_product", {})
    if str(product.get("container_prefix")) != TOY_CONTAINER_PREFIX:
        raise StudyError("rigid toy container prefix drift")
    if int(product.get("n_toys_available", -1)) != PRODUCT_N_TOYS:
        raise StudyError("available rigid toy count drift")
    if int(product.get("n_toys_analyzed", -1)) != ACTIVE_N_TOYS:
        raise StudyError("analyzed rigid toy count drift")
    if tuple(int(value) for value in product.get("reserve_toy_indices", ())) != (
        RESERVED_TOY_INDICES
    ):
        raise StudyError("rigid reserve toy indices must be exactly 20--24")
    if int(product.get("base_seed", -1)) != BASE_SEED:
        raise StudyError("rigid toy base seed drift")
    if str(product.get("root_sha256")) != EXPECTED_ROOT_SHA256:
        raise StudyError("rigid toy ROOT declaration drift")
    if str(product.get("manifest_sha256")) != EXPECTED_MANIFEST_SHA256:
        raise StudyError("rigid toy manifest declaration drift")

    state = spec.get("declared_result_state", {})
    if str(state.get("version")) != "v4.2":
        raise StudyError("frozen result-state version must remain v4.2")
    if str(state.get("config_sha256")) != EXPECTED_CARD_SHA256:
        raise StudyError("frozen v4.2 card declaration drift")

    gate = spec.get("optimizer_gate", {})
    expected_gate = {
        "version": "v4p7p1_reference_relative_v1",
        "reference_initial_attempts": 3,
        "maximum_attempts": 5,
        "top_branch_min_replicates": 2,
        "delta_lml_per_train_max": 0.001,
        "abs_log_length_ratio_max": 0.01,
        "abs_log_constant_ratio_max": 0.05,
        "bound_ratio_window": 0.02,
        "covariance_min_eigenvalue_relative": -0.01,
    }
    mismatches: list[str] = []
    for key, expected in expected_gate.items():
        actual = gate.get(key)
        if isinstance(expected, float):
            valid = actual is not None and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15
            )
        else:
            valid = actual == expected
        if not valid:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")
    if mismatches:
        raise StudyError(
            "source thresholds needed by the reduced length-only gate drifted: "
            + "; ".join(mismatches)
        )
    selection_rule = str(gate.get("selection_rule", "")).lower()
    for token in ("maximum", "reproducible", "no pull", "cls"):
        if token not in selection_rule:
            raise StudyError(f"optimizer selection-rule declaration lacks {token!r}")


def build_config(upper_factor: int) -> Any:
    from hps_gpr.config import load_config

    if int(upper_factor) not in UPPER_FACTORS:
        raise StudyError(f"unsupported length upper factor: {upper_factor}")
    spec = load_spec()
    card = resolve_study_path(
        str(spec["declared_result_state"]["archived_config_path"])
    )
    cfg = load_config(str(card))
    cfg.enable_2015 = False
    cfg.enable_2016 = False
    cfg.enable_2021 = True
    cfg.do_combined = False
    cfg.make_ul_bands = False
    cfg.ul_bands_toys = 0
    cfg.do_combined_bands = False
    cfg.combined_bands_n_toys = 0
    cfg.make_eps2_bands = False
    cfg.cls_mode = "asymptotic"
    cfg.cls_num_toys = 0
    cfg.cls_alpha = 0.10
    cfg.kernel_ls_res_lower_factor_by_dataset = dict(
        cfg.kernel_ls_res_lower_factor_by_dataset
    )
    cfg.kernel_ls_res_upper_factor_by_dataset = dict(
        cfg.kernel_ls_res_upper_factor_by_dataset
    )
    cfg.kernel_ls_res_lower_factor_by_dataset["2021"] = 1.1
    cfg.kernel_ls_res_upper_factor_by_dataset["2021"] = float(upper_factor)
    cfg.blind_nsigma = 2.25
    cfg.gp_train_exclude_nsigma = 2.25
    cfg.scan_edge_guard_nsigma = 2.25
    cfg.scan_require_two_sidebands = True
    cfg.neighborhood_rebin = 5
    cfg.n_restarts = OPTIMIZER_RESTARTS
    cfg.extract_allow_negative = True
    cfg.extract_background_mode = "profiled"
    cfg.eps2_density_nsigma = 1.64
    cfg.signal_model = "default"
    cfg.fail_fast = True
    cfg.debug_print = False
    cfg.save_plots = False
    return cfg


def assert_config(cfg: Any, upper_factor: int) -> None:
    checks = {
        "search_50_250": tuple(map(float, cfg.range_2021)) == SEARCH_GEV,
        "support_40_300": tuple(map(float, cfg.data_range_2021)) == SUPPORT_GEV,
        "pre_log": bool(cfg.pre_log),
        "alpha_model": str(cfg.alpha_model) == "1/y",
        "pre_zero_alpha": float(cfg.pre_zero_alpha) == 1.0,
        "local_resolution_policy": str(cfg.kernel_ls_policy)
        == "resolution_scaled_local",
        "lower_factor": float(
            cfg.kernel_ls_res_lower_factor_by_dataset["2021"]
        )
        == 1.1,
        "requested_upper_factor": float(
            cfg.kernel_ls_res_upper_factor_by_dataset["2021"]
        )
        == float(upper_factor),
        "blind_nsigma": float(cfg.blind_nsigma) == 2.25,
        "train_exclude_nsigma": float(cfg.gp_train_exclude_nsigma) == 2.25,
        "edge_guard_nsigma": float(cfg.scan_edge_guard_nsigma) == 2.25,
        "two_sidebands": bool(cfg.scan_require_two_sidebands),
        "rebin_five": int(cfg.neighborhood_rebin) == 5,
        "twelve_restarts": int(cfg.n_restarts) == OPTIMIZER_RESTARTS,
        "no_limit_bands": not bool(cfg.make_ul_bands)
        and not bool(cfg.do_combined_bands)
        and not bool(cfg.make_eps2_bands),
        "zero_cls_toys": int(cfg.cls_num_toys) == 0,
        "ninety_percent_if_inference_were_external": float(cfg.cls_alpha) == 0.10,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise StudyError(
            f"factor {upper_factor} frozen-card assertions failed: "
            + ", ".join(failed)
        )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_canonicalize(item) for item in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def controlled_config_audit() -> dict[str, Any]:
    full_hashes: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    for factor in UPPER_FACTORS:
        cfg = build_config(factor)
        assert_config(cfg, factor)
        payload = _canonicalize(vars(cfg))
        full_hashes[str(factor)] = canonical_json_hash(payload)
        normalized = json.loads(json.dumps(payload, default=str))
        normalized["kernel_ls_res_upper_factor_by_dataset"]["2021"] = (
            "__CONTROLLED_UPPER_FACTOR__"
        )
        normalized_hashes[str(factor)] = canonical_json_hash(normalized)
    if len(set(normalized_hashes.values())) != 1:
        raise StudyError("factor configurations differ beyond the controlled 2021 upper bound")
    return {
        "controlled_field": "kernel_ls_res_upper_factor_by_dataset.2021",
        "full_config_hashes": full_hashes,
        "normalized_config_hash": next(iter(normalized_hashes.values())),
        "only_controlled_field_differs": True,
    }


def validate_toy_product(spec: Mapping[str, Any]) -> dict[str, Any]:
    import uproot

    path = background_root(spec)
    product = spec["background_toy_product"]
    manifest_path = resolve_study_path(str(product["manifest"]))
    manifest = load_json(manifest_path)
    manifest_content = dict(manifest)
    recorded_content_hash = manifest_content.pop("manifest_content_sha256", None)
    if not isinstance(recorded_content_hash, str) or canonical_json_hash(
        manifest_content
    ) != recorded_content_hash:
        raise StudyError("background toy manifest content hash mismatch")

    all_scenarios = (
        "2021_1pct",
        "2021_1pct_x10",
        "2021_1pct_x100",
        "2021_10pct",
        "2021_10pct_x10",
    )
    scenario_policy = {
        "2021_1pct": ("one_pct", 1, None, 1, "base_1x", 12_504_044),
        "2021_1pct_x10": (
            "one_pct",
            10,
            "2021_1pct",
            9,
            "increment_9x",
            125_040_440,
        ),
        "2021_1pct_x100": (
            "one_pct",
            100,
            "2021_1pct_x10",
            90,
            "increment_90x",
            1_250_404_400,
        ),
        "2021_10pct": ("ten_pct", 1, None, 1, "base_1x", 141_251_508),
        "2021_10pct_x10": (
            "ten_pct",
            10,
            "2021_10pct",
            9,
            "increment_9x",
            1_412_515_080,
        ),
    }
    expected_template = (
        "toys/rigid_ggt26_scaled1pct/{scenario}/toy_{toy_index:04d}"
    )
    rigid = spec["rigid_generator"]
    manifest_checks = {
        "schema_version": int(manifest.get("schema_version", -1)) == 1,
        "generator_tag": str(manifest.get("generator_tag", ""))
        == "rigid_ggt26_scaled1pct",
        "generator_spec_sha256": str(
            manifest.get("generator_spec_sha256", "")
        )
        == str(rigid["sha256"]),
        "generator_spec_path": Path(
            str(manifest.get("generator_spec", ""))
        ).resolve()
        == resolve_study_path(str(rigid["path"])).resolve(),
        "promotion_gate": manifest.get("promotion_gate_passed") is False,
        "support": tuple(map(float, manifest.get("support_gev", ())))
        == SUPPORT_GEV,
        "base_seed": int(manifest.get("base_seed", -1)) == BASE_SEED,
        "n_toys": int(manifest.get("n_toys_per_source_family", -1))
        == PRODUCT_N_TOYS,
        "all_scenarios": tuple(manifest.get("all_scenarios", ()))
        == all_scenarios,
        "reported_scenarios": tuple(manifest.get("reported_scenarios", ()))
        == SCENARIOS,
        "toy_key_template": str(manifest.get("toy_key_template", ""))
        == expected_template,
    }
    failed_manifest_checks = [
        name for name, passed in manifest_checks.items() if not passed
    ]
    if failed_manifest_checks:
        raise StudyError(
            "background toy manifest contract drift: "
            + ", ".join(failed_manifest_checks)
        )

    source_policy = manifest.get("source_policy", {})
    if not isinstance(source_policy, Mapping):
        raise StudyError("background toy source policy is missing")
    source_inputs = spec["source_inputs"]
    if (
        bool(source_policy.get("native_10pct_shape_refit", True))
        or str(source_policy.get("one_pct_source_sha256", ""))
        != str(source_inputs["one_pct"]["sha256"])
        or str(source_policy.get("ten_pct_source_sha256", ""))
        != str(source_inputs["ten_pct"]["sha256"])
        or str(source_policy.get("shape_source", "")) != "native 1pct only"
    ):
        raise StudyError("background toy source/transfer policy drift")

    manifest_rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in manifest.get("toys", ()):
        key = (str(row.get("scenario")), int(row.get("toy_index", -1)))
        if key in manifest_rows:
            raise StudyError(f"duplicate background toy manifest key: {key}")
        manifest_rows[key] = row
    expected_keys = {
        (scenario, toy_index)
        for scenario in all_scenarios
        for toy_index in PRODUCT_TOY_INDICES
    }
    if set(manifest_rows) != expected_keys:
        raise StudyError("background toy manifest inventory is not exactly 5 x 25")

    source_edges: np.ndarray | None = None
    for family in ("one_pct", "ten_pct"):
        source_record = source_inputs[family]
        with uproot.open(
            resolve_study_path(str(source_record["path"]))
        ) as source_file:
            _, edges = source_file[str(source_record["histogram"])].to_numpy()
        edges = np.asarray(edges, dtype=float)
        if source_edges is None:
            source_edges = edges
        elif not np.array_equal(edges, source_edges):
            raise StudyError("native source histogram edges differ")
    if source_edges is None:
        raise StudyError("native source edges are unavailable")

    count_hashes: dict[str, str] = {}
    with uproot.open(path) as root_file:
        cached_counts: dict[tuple[str, int], np.ndarray] = {}
        for scenario in all_scenarios:
            family, multiplier, parent, increment_multiplier, stage, _ = (
                scenario_policy[scenario]
            )
            for toy_index in PRODUCT_TOY_INDICES:
                key = toy_key(scenario, toy_index)
                if key not in root_file:
                    raise StudyError(f"missing predeclared toy histogram: {key}")
                values, edges = root_file[key].to_numpy()
                values = np.asarray(values, dtype=float)
                edges = np.asarray(edges, dtype=float)
                if values.ndim != 1 or edges.shape != (values.size + 1,):
                    raise StudyError(f"invalid one-dimensional histogram: {key}")
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise StudyError(f"nonfinite or negative toy counts: {key}")
                rounded = np.rint(values)
                if not np.allclose(values, rounded, rtol=0.0, atol=1e-6):
                    raise StudyError(f"toy counts are not integer-like: {key}")
                if not np.array_equal(edges, source_edges):
                    raise StudyError(f"toy/source edge mismatch: {key}")
                centers = 0.5 * (edges[:-1] + edges[1:])
                outside_support = (centers < SUPPORT_GEV[0]) | (
                    centers >= SUPPORT_GEV[1]
                )
                if np.any(rounded[outside_support] != 0):
                    raise StudyError(f"toy has nonzero counts outside support40: {key}")
                counts_digest = array_hash(rounded, "<i8")
                manifest_row = manifest_rows[(scenario, toy_index)]
                if (
                    str(manifest_row.get("output_histogram", "")) != key
                    or str(manifest_row.get("source_family", "")) != family
                    or manifest_row.get("parent_scenario") != parent
                    or int(manifest_row.get("increment_multiplier", -1))
                    != increment_multiplier
                    or list(manifest_row.get("increment_seed_words", ()))
                    != stable_seed_words(
                        "nested_poisson", family, toy_index, stage
                    )
                ):
                    raise StudyError(f"manifest toy semantics drift: {key}")
                declared_count_hash = manifest_row.get(
                    "counts_sha256", manifest_row.get("counts_sha256_int64")
                )
                if str(declared_count_hash) != counts_digest:
                    raise StudyError(f"manifest count hash mismatch: {key}")
                declared_total = manifest_row.get(
                    "total_count", manifest_row.get("total_040_300", -1)
                )
                if int(declared_total) != int(np.sum(rounded)):
                    raise StudyError(f"manifest count total mismatch: {key}")
                increment = (
                    rounded
                    if parent is None
                    else rounded - cached_counts[(parent, toy_index)]
                )
                if np.any(increment < 0) or array_hash(increment, "<i8") != str(
                    manifest_row.get("increment_sha256_int64", "")
                ):
                    raise StudyError(f"nested increment mismatch: {key}")
                if scenario in SCENARIOS:
                    usable = rounded.size // 5 * 5
                    rebinned = rounded[:usable].reshape(-1, 5).sum(axis=1)
                    rebinned_edges = edges[: usable + 1 : 5]
                    rebinned_centers = 0.5 * (
                        rebinned_edges[:-1] + rebinned_edges[1:]
                    )
                    in_support = (rebinned_centers >= SUPPORT_GEV[0]) & (
                        rebinned_centers < SUPPORT_GEV[1]
                    )
                    if np.any(rebinned[in_support] <= 0):
                        raise StudyError(f"nonpositive pre-log support count: {key}")
                cached_counts[(scenario, toy_index)] = rounded.astype(
                    np.int64, copy=True
                )
                count_hashes[key] = counts_digest

        manifest_truths = {
            str(row.get("scenario")): row for row in manifest.get("truths", ())
        }
        if set(manifest_truths) != set(all_scenarios):
            raise StudyError("background toy manifest truth inventory drift")
        analytic_keys = product.get("analytic_mean_keys", {})
        for scenario in all_scenarios:
            family, multiplier, _, _, _, expected_total = scenario_policy[scenario]
            truth_row = manifest_truths[scenario]
            analytic_key = str(truth_row.get("analytic_mean_key", ""))
            if scenario in SCENARIOS and str(analytic_keys.get(scenario, "")) != (
                analytic_key
            ):
                raise StudyError(f"analytic-mean key mismatch for {scenario}")
            if analytic_key not in root_file:
                raise StudyError(f"missing analytic truth histogram: {analytic_key}")
            values, edges = root_file[analytic_key].to_numpy()
            values = np.asarray(values, dtype=float)
            edges = np.asarray(edges, dtype=float)
            if (
                str(truth_row.get("source_family", "")) != family
                or int(truth_row.get("multiplier", -1)) != multiplier
                or not np.array_equal(edges, source_edges)
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise StudyError(f"analytic-mean semantics mismatch for {scenario}")
            centers = 0.5 * (edges[:-1] + edges[1:])
            outside_support = (centers < SUPPORT_GEV[0]) | (
                centers >= SUPPORT_GEV[1]
            )
            if np.any(values[outside_support] != 0.0):
                raise StudyError(
                    f"analytic truth nonzero outside support40: {scenario}"
                )
            if array_hash(values, "<f8") != str(
                truth_row.get("mean_sha256_float64", "")
            ):
                raise StudyError(f"analytic-mean hash mismatch for {scenario}")
            if not math.isclose(
                float(np.sum(values)),
                float(expected_total),
                rel_tol=0.0,
                abs_tol=max(1e-3, float(expected_total) * 1e-9),
            ):
                raise StudyError(f"analytic-mean total mismatch for {scenario}")

    active_count_hashes = {
        key: value
        for key, value in count_hashes.items()
        if key.split("/")[-2] in SCENARIOS
        and int(key.rsplit("_", 1)[1]) in TOY_INDICES
    }
    reserved_count_hashes = {
        key: value
        for key, value in count_hashes.items()
        if key.split("/")[-2] in SCENARIOS
        and int(key.rsplit("_", 1)[1]) in RESERVED_TOY_INDICES
    }
    return {
        "status": "pass",
        "root": str(path),
        "histograms": len(count_hashes),
        "active_histograms": len(active_count_hashes),
        "reserved_histograms": len(reserved_count_hashes),
        "hidden_parent_histograms": PRODUCT_N_TOYS,
        "reported_histograms": len(SCENARIOS) * PRODUCT_N_TOYS,
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "reserve_policy": "inventory-validation-only; no optimizer task may consume reserve toys",
        "n_bins": int(len(source_edges) - 1),
        "histogram_extent_gev": [
            float(source_edges[0]),
            float(source_edges[-1]),
        ],
        "edges_sha256_float64": array_hash(source_edges, "<f8"),
        "support_gev": list(SUPPORT_GEV),
        "manifest_content_sha256": recorded_content_hash,
        "counts_inventory_sha256": canonical_json_hash(count_hashes),
        "active_counts_inventory_sha256": canonical_json_hash(active_count_hashes),
        "reserved_counts_inventory_sha256": canonical_json_hash(
            reserved_count_hashes
        ),
        "nesting_and_seed_words_checked": True,
        "analytic_truths_checked": len(all_scenarios),
        "source_edges_checked": True,
    }


def preflight(*, validate_inventory: bool = True) -> dict[str, Any]:
    lock_provenance = verify_external_lock()
    spec = load_spec()
    assert_spec_contract(spec)
    assert_reserve_outputs_untouched()
    checks: dict[str, bool] = {}
    checks["reserve_output_directories_untouched"] = True

    state = spec["declared_result_state"]
    card_path = resolve_study_path(str(state["archived_config_path"]))
    require_hash(card_path, EXPECTED_CARD_SHA256, "frozen v4.2 analysis card")
    checks["frozen_v4p2_card"] = True

    product = spec["background_toy_product"]
    root_path = background_root(spec)
    manifest_path = resolve_study_path(str(product["manifest"]))
    require_hash(root_path, EXPECTED_ROOT_SHA256, "rigid nested toy ROOT")
    require_hash(manifest_path, EXPECTED_MANIFEST_SHA256, "rigid toy manifest")
    checks["exact_toy_root"] = True
    checks["exact_toy_manifest"] = True

    for section_name in (
        "source_inputs",
        "rigid_generator",
        "functional_form_models",
        "fit_products",
        "model_products",
    ):
        section = spec.get(section_name)
        if section is not None:
            verify_declared_hashes(
                section, label=section_name, checks=checks
            )

    fit_record = spec.get("fit_product", spec.get("fit_summary"))
    if isinstance(fit_record, Mapping):
        fit_path_value = fit_record.get("path", fit_record.get("file"))
        fit_sha = fit_record.get("sha256")
        if isinstance(fit_path_value, str) and isinstance(fit_sha, str):
            fit_path = resolve_study_path(fit_path_value)
            require_hash(fit_path, fit_sha, "rigid-generator fit summary")
            fit_payload = load_json(fit_path)
            if not bool(
                fit_payload.get("model_selection_frozen_before_injection")
            ):
                raise StudyError(
                    "rigid-generator model selection was not frozen before injection"
                )
            checks["fit_summary_scientific_freeze"] = True

    runtime_record = spec.get("runtime_instrumentation", {})
    if str(runtime_record.get("package_manifest_sha256")) != (
        EXPECTED_RUNTIME_MANIFEST_SHA256
    ):
        raise StudyError("runtime package-manifest declaration drift")
    runtime_manifest_path = resolve_study_path(
        str(runtime_record.get("package_manifest"))
    )
    require_hash(
        runtime_manifest_path,
        EXPECTED_RUNTIME_MANIFEST_SHA256,
        "complete study-local runtime package manifest",
    )
    runtime_manifest = load_json(runtime_manifest_path)
    if int(runtime_manifest.get("schema_version", -1)) != 1:
        raise StudyError("unsupported runtime package-manifest schema")
    runtime_files = runtime_manifest.get("files", {})
    if not isinstance(runtime_files, Mapping) or not runtime_files:
        raise StudyError("runtime package manifest has no file inventory")
    actual_runtime_files = {
        str(path.relative_to(RUNTIME_ROOT))
        for path in (RUNTIME_ROOT / "hps_gpr").rglob("*.py")
        if path.is_file()
    }
    if actual_runtime_files != set(runtime_files):
        missing = sorted(set(runtime_files).difference(actual_runtime_files))
        extra = sorted(actual_runtime_files.difference(runtime_files))
        raise StudyError(
            "runtime package file-set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    for relative, expected_hash in runtime_files.items():
        require_hash(
            RUNTIME_ROOT / str(relative),
            str(expected_hash),
            f"complete runtime file {relative}",
        )
    checks["complete_runtime_package_manifest"] = True

    if str(runtime_record.get("production_driver_sha256")) != (
        EXPECTED_CLOSURE_DRIVER_SHA256
    ):
        raise StudyError("closure production-driver declaration drift")
    require_hash(
        HERE / "run_rigid_study.py",
        EXPECTED_CLOSURE_DRIVER_SHA256,
        "frozen rigid-closure production driver",
    )
    checks["frozen_closure_driver_provenance"] = True

    declared_modules = runtime_record.get("modules", {})
    if set(declared_modules) != set(EXPECTED_RUNTIME_SHA256):
        raise StudyError("study-local runtime module inventory drift")
    for relative, expected_hash in EXPECTED_RUNTIME_SHA256.items():
        record = declared_modules[relative]
        if str(record.get("sha256")) != expected_hash:
            raise StudyError(f"declared runtime hash drift: {relative}")
        archived = resolve_study_path(str(record.get("archived_path")))
        require_hash(archived, expected_hash, f"archived runtime {relative}")
        module_name = relative.replace("/", ".")[:-3]
        imported = importlib.import_module(module_name)
        imported_path = Path(str(imported.__file__)).resolve()
        require_hash(imported_path, expected_hash, f"imported runtime {relative}")
        if RUNTIME_ROOT.resolve() not in imported_path.parents:
            raise StudyError(f"study-local runtime overlay not imported: {relative}")
        checks[f"runtime.{relative}"] = True

    from hps_gpr.gpr import fit_gpr
    from hps_gpr.io import BlindPrediction

    if "random_state" not in inspect.signature(fit_gpr).parameters:
        raise StudyError("audited fit_gpr(random_state=...) runtime is not active")
    fields = set(getattr(BlindPrediction, "__dataclass_fields__", {}))
    required_fields = {
        "ls_opt",
        "ls_lo",
        "ls_hi",
        "sigma_x",
        "const_opt",
        "lml",
        "optimizer_warning_count",
        "optimizer_warnings",
    }
    if not required_fields.issubset(fields):
        raise StudyError("required optimizer instrumentation is not active")
    checks["runtime_import_resolution"] = True

    config_audit = controlled_config_audit()
    checks["controlled_one_factor_configs"] = True
    inventory = validate_toy_product(spec) if validate_inventory else None
    if inventory is not None:
        checks["full_5x25_toy_nesting_seed_truth_inventory"] = True
        checks["reserve_20_24_not_scheduled"] = (
            set(TOY_INDICES).isdisjoint(RESERVED_TOY_INDICES)
            and set(TOY_INDICES).union(RESERVED_TOY_INDICES)
            == set(PRODUCT_TOY_INDICES)
        )

    return {
        "status": "pass",
        "validated_utc": utc_now(),
        "checks": checks,
        "external_lock": lock_provenance,
        "config_audit": config_audit,
        "toy_inventory": inventory,
        "scan_contract": {
            "background_only": True,
            "scenarios": list(SCENARIOS),
            "active_toy_indices": list(TOY_INDICES),
            "reserved_toy_indices": list(RESERVED_TOY_INDICES),
            "reserve_policy": (
                "hash/inventory validation only; no fit task, collection row, "
                "or diagnostic may consume toys 20--24"
            ),
            "masses_gev": list(MASS_GRID),
            "upper_factors": list(UPPER_FACTORS),
            "support_gev": list(SUPPORT_GEV),
            "common_optimizer_seed_namespace": SEED_NAMESPACE,
            "optimizer_gate": "reduced_length_only_pull_blind_v1",
            "gate_coordinates": [
                "gp_lml_per_training_bin",
                "ell_opt_log_ratio",
                "kernel_constant_opt_log_ratio",
                "covariance_validity",
            ],
            "card_selection_performed": False,
        },
    }


def _load_histogram(path: Path, key: str) -> Any:
    from hps_gpr.funcform_toys import load_funcform_toy_hist

    container, name = key.rsplit("/", 1)
    return load_funcform_toy_hist(str(path), container=container, toy_name=name)


def make_toy_dataset(scenario: str, toy_index: int, cfg: Any) -> Any:
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import FuncFormToySpec, build_funcform_toy_dataset

    spec = load_spec()
    root_path = background_root(spec)
    key = toy_key(scenario, toy_index)
    histogram = _load_histogram(root_path, key)
    base = make_datasets(cfg)["2021"]
    if (float(base.data_low), float(base.data_high)) != SUPPORT_GEV:
        raise StudyError("toy dataset base support is not 40--300 MeV")
    function_tag = str(spec["scenarios"][scenario]["function_tag"])
    toy_spec = FuncFormToySpec(
        source_root=str(root_path),
        container=f"{TOY_CONTAINER_PREFIX}/{scenario}",
        function_tag=function_tag,
        toy_name=f"toy_{toy_index:04d}",
        toy_index=toy_index,
    )
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def covariance_diagnostics(covariance: Any, gate: Mapping[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(covariance, dtype=float)
    finite = bool(
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and matrix.size > 0
        and np.isfinite(matrix).all()
    )
    if not finite:
        return {
            "covariance_valid": False,
            "covariance_min_eigenvalue": float("nan"),
            "covariance_min_eigenvalue_relative": float("nan"),
        }
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.diag(symmetric))), 1.0)
    minimum = float(np.min(eigenvalues))
    relative = minimum / scale
    threshold = float(gate["covariance_min_eigenvalue_relative"])
    return {
        "covariance_valid": bool(
            np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8 * scale)
            and relative >= threshold
        ),
        "covariance_min_eigenvalue": minimum,
        "covariance_min_eigenvalue_relative": relative,
    }


def training_geometry(prediction: Any, mass: float, cfg: Any) -> dict[str, Any]:
    x = np.asarray(prediction.x_full, dtype=float).reshape(-1)
    y = np.asarray(prediction.y_full, dtype=float).reshape(-1)
    edges = np.asarray(prediction.edges_full, dtype=float).reshape(-1)
    half_width = float(cfg.gp_train_exclude_nsigma) * float(prediction.sigma_val)
    mask = (x < mass - half_width) | (x > mass + half_width)
    if x.shape != y.shape or edges.shape != (x.size + 1,):
        raise StudyError(f"training geometry mismatch at {mass:.6g} GeV")
    selected = y[mask]
    if selected.size == 0 or not np.all(np.isfinite(selected)):
        raise StudyError(f"invalid empty/nonfinite training set at {mass:.6g} GeV")
    if bool(cfg.pre_log) and np.any(selected <= 0):
        raise StudyError(f"pre_log training counts are nonpositive at {mass:.6g} GeV")
    if int(np.count_nonzero(mask)) != int(prediction.n_train):
        raise StudyError(f"runtime/derived n_train mismatch at {mass:.6g} GeV")
    return {
        "n_train": int(np.count_nonzero(mask)),
        "n_train_low": int(np.count_nonzero(mask & (x < mass))),
        "n_train_high": int(np.count_nonzero(mask & (x > mass))),
        "train_domain_lo": float(edges[0]),
        "train_domain_hi": float(edges[-1]),
        "bin_width_median": float(np.median(np.diff(edges))),
        "training_counts_sha256": array_hash(selected, "<f8"),
    }


def _positive_log_ratio(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or left <= 0 or right <= 0:
        return float("nan")
    return float(math.log(left / right))


def fit_attempt(
    dataset: Any,
    cfg: Any,
    gate: Mapping[str, Any],
    scenario: str,
    toy_index: int,
    mass: float,
    upper_factor: int,
    attempt: int,
) -> dict[str, Any]:
    from hps_gpr.gpr import length_scale_x_to_mass_delta
    from hps_gpr.io import estimate_background_for_dataset

    # Deliberately omit upper_factor: every factor gets the same random restart
    # stream for the paired (scenario, toy, mass, attempt) comparison.
    optimizer_seed = stable_seed(
        SEED_NAMESPACE,
        scenario,
        toy_index,
        f"{mass:.9f}",
        attempt,
    )
    cfg.gp_optimizer_random_state = int(optimizer_seed)
    base = {
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "mass_MeV": int(round(1000.0 * mass)),
        "upper_factor": int(upper_factor),
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "optimizer_seed_namespace": SEED_NAMESPACE,
        "seed_includes_upper_factor": False,
        "optimizer_restarts": OPTIMIZER_RESTARTS,
        "background_only": True,
        "fit_ok": False,
        "error": "",
    }
    try:
        pred = estimate_background_for_dataset(
            dataset,
            float(mass),
            cfg,
            restarts=OPTIMIZER_RESTARTS,
            optimize=True,
        )
        geometry = training_geometry(pred, float(mass), cfg)
        covariance = covariance_diagnostics(pred.cov, gate)
        ell = float(pred.ls_opt)
        ell_lo = float(pred.ls_lo)
        ell_hi = float(pred.ls_hi)
        sigma_x = float(pred.sigma_x)
        constant = float(pred.const_opt)
        constant_lo = float(pred.const_lo)
        constant_hi = float(pred.const_hi)
        ell_over_hi = ell / ell_hi
        ell_over_lo = ell / ell_lo
        near_window = float(gate["bound_ratio_window"])
        record = {
            **base,
            "gp_lml": float(pred.lml),
            "ell_opt": ell,
            "ell_lo": ell_lo,
            "ell_hi": ell_hi,
            "ell_init": float(pred.ls_init),
            "sigma_x": sigma_x,
            "ell_over_sigma_x": ell / sigma_x,
            "ell_lo_over_sigma_x": ell_lo / sigma_x,
            "ell_hi_over_sigma_x": ell_hi / sigma_x,
            "ell_over_ell_hi": ell_over_hi,
            "ell_over_ell_lo": ell_over_lo,
            "ell_mass_delta_GeV": float(
                length_scale_x_to_mass_delta(ell, float(mass), bool(cfg.pre_log))
            ),
            "kernel_constant_opt": constant,
            "kernel_constant_lo": constant_lo,
            "kernel_constant_hi": constant_hi,
            "kernel_constant_init": float(pred.const_init),
            "n_blind": int(pred.n_blind),
            "blind_lo": float(pred.blind[0]),
            "blind_hi": float(pred.blind[1]),
            "sigma_mass_GeV": float(pred.sigma_val),
            "support_lo_GeV": float(pred.edges_full[0]),
            "support_hi_GeV": float(pred.edges_full[-1]),
            "ell_at_lower_exact": bool(ell_over_lo <= 1.0 / EXACT_BOUND_RATIO),
            "ell_at_upper_exact": bool(ell_over_hi >= EXACT_BOUND_RATIO),
            "ell_near_lower": bool(ell_over_lo <= 1.0 + near_window),
            "ell_near_upper": bool(ell_over_hi >= 1.0 - near_window),
            "constant_at_lower_exact": bool(
                np.isfinite(constant_lo)
                and constant_lo > 0
                and constant / constant_lo <= 1.0 / EXACT_BOUND_RATIO
            ),
            "constant_at_upper_exact": bool(
                np.isfinite(constant_hi)
                and constant_hi > 0
                and constant / constant_hi >= EXACT_BOUND_RATIO
            ),
            "optimizer_warning_count": int(pred.optimizer_warning_count),
            "optimizer_warnings": str(pred.optimizer_warnings),
            **geometry,
            **covariance,
        }
        numeric_ok = all(
            np.isfinite(float(record[key]))
            for key in (
                "gp_lml",
                "ell_opt",
                "ell_lo",
                "ell_hi",
                "sigma_x",
                "kernel_constant_opt",
            )
        )
        positive_ok = all(
            float(record[key]) > 0
            for key in (
                "ell_opt",
                "ell_lo",
                "ell_hi",
                "sigma_x",
                "kernel_constant_opt",
            )
        )
        support_ok = math.isclose(
            float(record["support_lo_GeV"]), SUPPORT_GEV[0], abs_tol=1e-12
        ) and math.isclose(
            float(record["support_hi_GeV"]), SUPPORT_GEV[1], abs_tol=1e-12
        )
        factor_ok = math.isclose(
            float(record["ell_hi_over_sigma_x"]),
            float(upper_factor),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        record["fit_ok"] = bool(numeric_ok and positive_ok and support_ok and factor_ok)
        if not record["fit_ok"]:
            record["error"] = "nonfinite/nonpositive/support/factor contract failure"
        return record
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:500]}


def branch_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> bool:
    required = (
        "gp_lml",
        "ell_opt",
        "kernel_constant_opt",
        "n_train",
    )
    if not all(
        np.isfinite(float(first.get(key, np.nan)))
        and np.isfinite(float(second.get(key, np.nan)))
        for key in required
    ):
        return False
    n_train = max(1.0, min(float(first["n_train"]), float(second["n_train"])))
    if (
        abs(float(first["gp_lml"]) - float(second["gp_lml"])) / n_train
        > float(gate["delta_lml_per_train_max"])
    ):
        return False
    for key, threshold in (
        ("ell_opt", gate["abs_log_length_ratio_max"]),
        ("kernel_constant_opt", gate["abs_log_constant_ratio_max"]),
    ):
        log_ratio = _positive_log_ratio(float(first[key]), float(second[key]))
        if not np.isfinite(log_ratio) or abs(log_ratio) > float(threshold):
            return False
    return True


def usable_attempts(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in records
        if bool(row.get("fit_ok"))
        and bool(row.get("covariance_valid"))
        and np.isfinite(float(row.get("gp_lml", np.nan)))
        and np.isfinite(float(row.get("ell_opt", np.nan)))
        and float(row.get("ell_opt", 0.0)) > 0
        and np.isfinite(float(row.get("kernel_constant_opt", np.nan)))
        and float(row.get("kernel_constant_opt", 0.0)) > 0
    ]


def select_branch(
    records: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, int, Mapping[str, Any] | None]:
    usable = usable_attempts(records)
    if not usable:
        return None, 0, None
    top = max(usable, key=lambda row: float(row["gp_lml"]))
    replicates = sum(branch_match(top, row, gate) for row in usable)
    if replicates < int(gate["top_branch_min_replicates"]):
        return None, replicates, top
    return top, replicates, top


def task_directory(scenario: str, toy_index: int) -> Path:
    return RUNS / scenario / f"toy_{toy_index:04d}"


def scan_contract_payload(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if spec is None:
        spec = load_spec()
    gate = spec["optimizer_gate"]
    lock_provenance = verify_external_lock()
    return {
        "schema_version": 1,
        "driver_sha256": lock_provenance["core_sha256"],
        "launcher_sha256": lock_provenance["launcher_sha256"],
        "external_lock_sha256": lock_provenance["lock_sha256"],
        "executable_trust_chain": lock_provenance["trust_chain"],
        "rigid_study_spec_sha256": sha256_file(SPEC_PATH),
        "toy_root_sha256": EXPECTED_ROOT_SHA256,
        "toy_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runtime_package_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
        "closure_driver_sha256": EXPECTED_CLOSURE_DRIVER_SHA256,
        "runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "background_only": True,
        "support_gev": list(SUPPORT_GEV),
        "search_gev": list(SEARCH_GEV),
        "scenarios": list(SCENARIOS),
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "reserve_policy": (
            "hash/inventory validation only; no fit task, collection row, "
            "or diagnostic may consume toys 20--24"
        ),
        "masses_gev": list(MASS_GRID),
        "upper_factors": list(UPPER_FACTORS),
        "optimizer_restarts": OPTIMIZER_RESTARTS,
        "seed_namespace": SEED_NAMESPACE,
        "seed_excludes_upper_factor": True,
        "optimizer_gate": {
            "name": "reduced_length_only_pull_blind_v1",
            "not_unchanged_v4p7_gate": True,
            **{
                key: gate[key]
                for key in (
                    "version",
                    "reference_initial_attempts",
                    "maximum_attempts",
                    "top_branch_min_replicates",
                    "delta_lml_per_train_max",
                    "abs_log_length_ratio_max",
                    "abs_log_constant_ratio_max",
                    "bound_ratio_window",
                    "covariance_min_eigenvalue_relative",
                )
            },
        },
        "gate_coordinates": [
            "gp_lml_per_training_bin",
            "ell_opt_log_ratio",
            "kernel_constant_opt_log_ratio",
            "covariance_validity",
        ],
        "prohibited_output_column_substrings": sorted(
            FORBIDDEN_OUTPUT_COLUMN_SUBSTRINGS
        ),
        "factor_selection_performed": False,
    }


def scan_contract_hash(spec: Mapping[str, Any] | None = None) -> str:
    return canonical_json_hash(scan_contract_payload(spec))


def _frame_schema_hash(frame: pd.DataFrame) -> str:
    return canonical_json_hash([str(column) for column in frame.columns])


def _frame_lattice_hash(
    frame: pd.DataFrame, key_columns: Sequence[str]
) -> str:
    if not all(column in frame.columns for column in key_columns):
        return ""
    records = [
        [str(value) for value in row]
        for row in frame.loc[:, list(key_columns)]
        .sort_values(list(key_columns))
        .itertuples(index=False, name=None)
    ]
    return canonical_json_hash(records)


def _assert_same_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    label: str,
) -> None:
    actual_sorted = actual.sort_values(list(sort_columns)).reset_index(drop=True)
    expected_sorted = expected.sort_values(list(sort_columns)).reset_index(drop=True)
    # ``read_csv`` represents an empty field as NaN.  Canonicalize missing
    # values to the empty string only where the independently recomputed
    # expected column is demonstrably textual.  Numeric columns retain their
    # native values, so this does not weaken the numerical semantic check.
    for column in expected_sorted.columns:
        nonmissing = expected_sorted[column].dropna()
        if len(nonmissing) == 0 or not all(
            isinstance(value, str) for value in nonmissing.tolist()
        ):
            continue
        for frame in (actual_sorted, expected_sorted):
            frame[column] = frame[column].astype(object).where(
                frame[column].notna(), ""
            )
    try:
        pd.testing.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_like=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise StudyError(f"{label} semantic mismatch: {exc}") from exc


def _self_test_csv_nullable_string_roundtrip() -> None:
    expected = pd.DataFrame(
        [
            {
                "mass_MeV": 50,
                "reason": "",
                "delta_lml_upper_minus_lower": 1.25,
            },
            {
                "mass_MeV": 70,
                "reason": "missing_reproducible_selected_branch",
                "delta_lml_upper_minus_lower": np.nan,
            },
        ]
    )
    with tempfile.TemporaryDirectory(prefix="rigid_length_csv_roundtrip_") as raw:
        path = Path(raw) / "nested_lml.csv"
        expected.to_csv(path, index=False)
        actual = pd.read_csv(path)
    if not pd.isna(actual.loc[0, "reason"]):
        raise StudyError(
            "nullable-string CSV self-test did not exercise empty-string to NaN"
        )
    _assert_same_frame(
        actual,
        expected,
        sort_columns=("mass_MeV",),
        label="nullable-string CSV roundtrip self-test",
    )


def _read_and_validate_task_products(
    directory: Path,
    scenario: str,
    toy_index: int,
) -> dict[str, pd.DataFrame]:
    spec = load_spec()
    gate = spec["optimizer_gate"]
    frames = {
        name: pd.read_csv(directory / name) for name in PRODUCT_NAMES
    }
    attempts = frames["optimizer_attempts.csv"]
    selected = frames["raw_ell_sigma_x_trajectories.csv"]
    exclusions = frames["optimizer_exclusions.csv"]
    nested = frames["nested_lml.csv"]
    occupancy = frames["bound_occupancy.csv"]
    for name, frame in frames.items():
        validate_no_inference_columns(frame, name)

    identity_columns = ("scenario", "background_toy_index")
    for name, frame in frames.items():
        if frame.empty and name == "optimizer_exclusions.csv":
            continue
        if not set(identity_columns).issubset(frame.columns):
            raise StudyError(f"{name} lacks exact task identity columns")
        if set(frame["scenario"].astype(str)) != {scenario}:
            raise StudyError(f"{name} scenario identity mismatch")
        toy_values = set(
            pd.to_numeric(
                frame["background_toy_index"], errors="raise"
            ).astype(int)
        )
        if toy_values != {toy_index}:
            raise StudyError(f"{name} toy identity mismatch")

    expected_states = {
        (mass_mev, factor)
        for mass_mev in MASS_MEV
        for factor in UPPER_FACTORS
    }
    required_attempt_columns = {
        "mass_MeV",
        "upper_factor",
        "attempt",
        "optimizer_seed",
        "fit_ok",
        "covariance_valid",
        "gp_lml",
        "ell_opt",
        "kernel_constant_opt",
        "n_train",
    }
    if not required_attempt_columns.issubset(attempts.columns):
        raise StudyError("optimizer attempt ledger schema is incomplete")
    attempt_key_columns = [
        "scenario",
        "background_toy_index",
        "mass_MeV",
        "upper_factor",
        "attempt",
    ]
    if bool(attempts.duplicated(attempt_key_columns).any()):
        raise StudyError("optimizer attempt ledger contains duplicate keys")
    attempt_states = set(
        zip(
            attempts["mass_MeV"].astype(int),
            attempts["upper_factor"].astype(int),
        )
    )
    if attempt_states != expected_states:
        raise StudyError("optimizer attempt ledger state lattice mismatch")

    for mass_mev in MASS_MEV:
        mass_rows = attempts[attempts["mass_MeV"].astype(int) == mass_mev]
        factor_attempts: dict[int, tuple[int, ...]] = {}
        for factor in UPPER_FACTORS:
            state_rows = mass_rows[
                mass_rows["upper_factor"].astype(int) == factor
            ]
            attempt_set = tuple(sorted(state_rows["attempt"].astype(int)))
            if attempt_set not in ((0, 1, 2), (0, 1, 2, 3, 4)):
                raise StudyError(
                    f"invalid attempt set at mass={mass_mev}, factor={factor}: "
                    f"{attempt_set}"
                )
            factor_attempts[factor] = attempt_set
            for row in state_rows.to_dict(orient="records"):
                expected_seed = stable_seed(
                    SEED_NAMESPACE,
                    scenario,
                    toy_index,
                    f"{mass_mev / 1000.0:.9f}",
                    int(row["attempt"]),
                )
                if int(row["optimizer_seed"]) != expected_seed:
                    raise StudyError("optimizer seed replay mismatch")
        if len(set(factor_attempts.values())) != 1:
            raise StudyError(f"factor attempt sets are unpaired at {mass_mev} MeV")

    state_columns = ["mass_MeV", "upper_factor"]
    for name, frame in (("selected", selected), ("excluded", exclusions)):
        if not set(state_columns).issubset(frame.columns):
            raise StudyError(f"{name} ledger lacks state columns")
        if bool(frame.duplicated(state_columns).any()):
            raise StudyError(f"{name} ledger contains duplicate state keys")
    selected_states = set(
        zip(selected["mass_MeV"].astype(int), selected["upper_factor"].astype(int))
    )
    excluded_states = set(
        zip(
            exclusions["mass_MeV"].astype(int),
            exclusions["upper_factor"].astype(int),
        )
    )
    if selected_states.intersection(excluded_states):
        raise StudyError("selected and excluded state ledgers overlap")
    if selected_states.union(excluded_states) != expected_states:
        raise StudyError("selected/excluded union is not the exact 33-state lattice")

    selected_lookup = {
        (int(row["mass_MeV"]), int(row["upper_factor"])): row
        for row in selected.to_dict(orient="records")
    }
    exclusion_lookup = {
        (int(row["mass_MeV"]), int(row["upper_factor"])): row
        for row in exclusions.to_dict(orient="records")
    }
    for state in sorted(expected_states):
        mass_mev, factor = state
        state_attempts = attempts[
            (attempts["mass_MeV"].astype(int) == mass_mev)
            & (attempts["upper_factor"].astype(int) == factor)
        ].to_dict(orient="records")
        recomputed, replicates, top = select_branch(state_attempts, gate)
        if recomputed is None:
            if state not in exclusion_lookup or state in selected_lookup:
                raise StudyError(f"state {state} should be optimizer-excluded")
            recorded = exclusion_lookup[state]
            if int(recorded["maximum_lml_branch_replicates"]) != replicates:
                raise StudyError(f"state {state} exclusion replicate count drift")
        else:
            if state not in selected_lookup or state in exclusion_lookup:
                raise StudyError(f"state {state} should have a selected branch")
            recorded = selected_lookup[state]
            if int(recorded["selected_attempt"]) != int(recomputed["attempt"]):
                raise StudyError(f"state {state} selected attempt drift")
            if int(recorded["top_branch_replicates"]) != replicates:
                raise StudyError(f"state {state} selected replicate count drift")
            for coordinate in (
                "optimizer_seed",
                "gp_lml",
                "ell_opt",
                "kernel_constant_opt",
                "n_train",
            ):
                left = float(recorded[coordinate])
                right = float(recomputed[coordinate])
                if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
                    raise StudyError(
                        f"state {state} selected {coordinate} does not match attempt"
                    )
            if top is None:
                raise StudyError(f"state {state} has no maximum-LML candidate")

    expected_nested = make_nested_lml(selected, [(scenario, toy_index)], gate)
    if len(nested) != len(MASS_MEV) * (len(UPPER_FACTORS) - 1):
        raise StudyError("nested-LML ledger is not the exact 22-row lattice")
    _assert_same_frame(
        nested,
        expected_nested,
        sort_columns=("mass_MeV", "lower_factor", "upper_factor"),
        label="nested-LML ledger",
    )
    expected_occupancy = make_occupancy(
        selected, ("scenario", "background_toy_index", "upper_factor")
    )
    _assert_same_frame(
        occupancy,
        expected_occupancy,
        sort_columns=("upper_factor",),
        label="bound-occupancy ledger",
    )
    return frames


def validate_success(
    directory: Path,
    contract_hash: str,
    scenario: str,
    toy_index: int,
) -> tuple[bool, str]:
    success_path = directory / "_SUCCESS.json"
    if not success_path.is_file():
        return False, "missing_success"
    try:
        payload = load_json(success_path)
        expected_task_id = f"{scenario}__toy_{toy_index:04d}"
        if (
            str(payload.get("task_id")) != expected_task_id
            or str(payload.get("scenario")) != scenario
            or int(payload.get("background_toy_index", -1)) != toy_index
        ):
            return False, "task_identity_mismatch"
        if str(payload.get("scan_contract_sha256")) != contract_hash:
            return False, "stale_contract"
        provenance = verify_external_lock()
        if payload.get("external_lock") != provenance:
            return False, "external_lock_provenance_mismatch"
        hashes = payload.get("product_sha256", {})
        if set(hashes) != set(PRODUCT_NAMES):
            return False, "product_inventory_mismatch"
        for name, expected in hashes.items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != str(expected):
                return False, f"product_hash_mismatch:{name}"
        frames = _read_and_validate_task_products(directory, scenario, toy_index)
        metadata = payload.get("product_metadata", {})
        if set(metadata) != set(PRODUCT_NAMES):
            return False, "product_metadata_inventory_mismatch"
        metadata_key_columns = {
            "optimizer_attempts.csv": (
                "scenario",
                "background_toy_index",
                "mass_MeV",
                "upper_factor",
                "attempt",
            ),
            "raw_ell_sigma_x_trajectories.csv": (
                "scenario",
                "background_toy_index",
                "mass_MeV",
                "upper_factor",
            ),
            "optimizer_exclusions.csv": (
                "scenario",
                "background_toy_index",
                "mass_MeV",
                "upper_factor",
            ),
            "bound_occupancy.csv": (
                "scenario",
                "background_toy_index",
                "upper_factor",
            ),
            "nested_lml.csv": (
                "scenario",
                "background_toy_index",
                "mass_MeV",
                "lower_factor",
                "upper_factor",
            ),
        }
        for name, frame in frames.items():
            path = directory / name
            record = metadata[name]
            if (
                int(record.get("size_bytes", -1)) != path.stat().st_size
                or int(record.get("rows", -1)) != len(frame)
                or str(record.get("schema_sha256")) != _frame_schema_hash(frame)
                or str(record.get("lattice_sha256"))
                != _frame_lattice_hash(frame, metadata_key_columns[name])
            ):
                return False, f"product_metadata_mismatch:{name}"
        if (
            int(payload.get("selected_rows", -1))
            != len(frames["raw_ell_sigma_x_trajectories.csv"])
            or int(payload.get("excluded_rows", -1))
            != len(frames["optimizer_exclusions.csv"])
            or int(payload.get("attempt_rows", -1))
            != len(frames["optimizer_attempts.csv"])
            or payload.get("background_only") is not True
            or payload.get("factor_selection_performed") is not False
            or payload.get("reserve_toys_consumed") is not False
            or str(payload.get("optimizer_gate"))
            != "reduced_length_only_pull_blind_v1"
        ):
            return False, "success_marker_semantics_mismatch"
    except Exception as exc:
        return False, f"invalid_success:{type(exc).__name__}"
    return True, "current"


def _selected_lookup(selected: pd.DataFrame) -> dict[tuple[str, int, int, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for row in selected.to_dict(orient="records"):
        key = (
            str(row["scenario"]),
            int(row["background_toy_index"]),
            int(row["mass_MeV"]),
            int(row["upper_factor"]),
        )
        if key in lookup:
            raise StudyError(f"duplicate selected trajectory state: {key}")
        lookup[key] = row
    return lookup


def make_nested_lml(
    selected: pd.DataFrame,
    scenarios_and_toys: Iterable[tuple[str, int]],
    gate: Mapping[str, Any],
) -> pd.DataFrame:
    lookup = _selected_lookup(selected)
    records: list[dict[str, Any]] = []
    for scenario, toy_index in scenarios_and_toys:
        for mass_mev in MASS_MEV:
            for lower, upper in zip(UPPER_FACTORS[:-1], UPPER_FACTORS[1:]):
                lower_row = lookup.get((scenario, toy_index, mass_mev, lower))
                upper_row = lookup.get((scenario, toy_index, mass_mev, upper))
                base = {
                    "scenario": scenario,
                    "background_toy_index": int(toy_index),
                    "mass_MeV": int(mass_mev),
                    "mass_GeV": mass_mev / 1000.0,
                    "lower_factor": int(lower),
                    "upper_factor": int(upper),
                    "strict_lml_tolerance": STRICT_NESTED_LML_TOLERANCE,
                    "gate_lml_per_train_tolerance": float(
                        gate["delta_lml_per_train_max"]
                    ),
                }
                if lower_row is None or upper_row is None:
                    records.append(
                        {
                            **base,
                            "comparable": False,
                            "reason": "missing_reproducible_selected_branch",
                        }
                    )
                    continue
                same_numeric = all(
                    math.isclose(
                        float(lower_row[key]),
                        float(upper_row[key]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for key in (
                        "sigma_x",
                        "ell_lo",
                        "blind_lo",
                        "blind_hi",
                        "support_lo_GeV",
                        "support_hi_GeV",
                    )
                )
                same_discrete = (
                    int(lower_row["n_train"]) == int(upper_row["n_train"])
                    and str(lower_row["training_counts_sha256"])
                    == str(upper_row["training_counts_sha256"])
                )
                if not same_numeric or not same_discrete:
                    raise StudyError(
                        "same-input geometry differs across upper factors for "
                        f"{scenario} toy {toy_index} mass {mass_mev}"
                    )
                delta = float(upper_row["gp_lml"]) - float(lower_row["gp_lml"])
                n_train = int(lower_row["n_train"])
                records.append(
                    {
                        **base,
                        "comparable": True,
                        "reason": "",
                        "n_train": n_train,
                        "lower_lml": float(lower_row["gp_lml"]),
                        "upper_lml": float(upper_row["gp_lml"]),
                        "delta_lml_upper_minus_lower": delta,
                        "delta_lml_per_train": delta / max(1, n_train),
                        "strict_nested_order_violation": bool(
                            delta < -STRICT_NESTED_LML_TOLERANCE
                        ),
                        "material_nested_order_violation": bool(
                            delta / max(1, n_train)
                            < -float(gate["delta_lml_per_train_max"])
                        ),
                        "same_input_geometry": True,
                    }
                )
    return pd.DataFrame(records)


def make_occupancy(selected: pd.DataFrame, group_keys: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame(columns=[*group_keys, "selected_rows"])
    for values, group in selected.groupby(list(group_keys), sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        record = dict(zip(group_keys, values))
        n_rows = len(group)
        record.update(
            {
                "selected_rows": n_rows,
                "exact_upper_bound_rows": int(group["ell_at_upper_exact"].sum()),
                "exact_upper_bound_fraction": float(
                    group["ell_at_upper_exact"].mean()
                ),
                "near_upper_bound_rows": int(group["ell_near_upper"].sum()),
                "near_upper_bound_fraction": float(group["ell_near_upper"].mean()),
                "exact_lower_bound_rows": int(group["ell_at_lower_exact"].sum()),
                "exact_lower_bound_fraction": float(
                    group["ell_at_lower_exact"].mean()
                ),
                "near_lower_bound_rows": int(group["ell_near_lower"].sum()),
                "near_lower_bound_fraction": float(group["ell_near_lower"].mean()),
                "exact_boundary_ratio": EXACT_BOUND_RATIO,
                "near_boundary_fractional_window": float(
                    load_spec()["optimizer_gate"]["bound_ratio_window"]
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.Series(dtype="object")
    return result


def validate_no_inference_columns(frame: pd.DataFrame, label: str) -> None:
    violations: dict[str, list[str]] = {}
    for column in frame.columns:
        normalized = "".join(
            character
            for character in str(column).lower()
            if character.isalnum()
        )
        matched = sorted(
            token
            for token in FORBIDDEN_OUTPUT_COLUMN_SUBSTRINGS
            if "".join(
                character
                for character in token.lower()
                if character.isalnum()
            )
            in normalized
        )
        if matched:
            violations[str(column)] = matched
    if violations:
        raise StudyError(
            f"{label} contains prohibited inference-column substrings: "
            f"{violations}"
        )


def run_task(
    scenario: str,
    toy_index: int,
    *,
    force: bool = False,
    preflight_done: bool = False,
) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise StudyError(f"unsupported scenario: {scenario}")
    if toy_index not in TOY_INDICES:
        if toy_index in RESERVED_TOY_INDICES:
            raise StudyError(
                f"toy index {toy_index} is reserved/untouched and may not be fit"
            )
        raise StudyError(f"active toy index must be in 0--19: {toy_index}")
    if not preflight_done:
        preflight(validate_inventory=True)

    spec = load_spec()
    gate = spec["optimizer_gate"]
    contract_hash = scan_contract_hash(spec)
    directory = task_directory(scenario, toy_index)
    current, reason = validate_success(
        directory, contract_hash, scenario, toy_index
    )
    if current and not force:
        payload = load_json(directory / "_SUCCESS.json")
        return {**payload, "cached": True}
    if directory.joinpath("_SUCCESS.json").exists() and not current and not force:
        raise StudyError(
            f"stale/corrupt task {scenario} toy {toy_index}: {reason}; use --force"
        )

    configs = {factor: build_config(factor) for factor in UPPER_FACTORS}
    for factor, cfg in configs.items():
        assert_config(cfg, factor)
    dataset = make_toy_dataset(scenario, toy_index, configs[15])

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
                row = fit_attempt(
                    dataset,
                    configs[factor],
                    gate,
                    scenario,
                    toy_index,
                    float(mass),
                    factor,
                    attempt,
                )
                by_factor[factor].append(row)
        initial_selected = {
            factor: select_branch(by_factor[factor], gate)[0]
            for factor in UPPER_FACTORS
        }
        # If any paired factor needs the v4.7 extension, evaluate the extension
        # for every factor so the seed/attempt set stays exactly common.
        if any(value is None for value in initial_selected.values()):
            for attempt in range(initial_attempts, maximum_attempts):
                for factor in UPPER_FACTORS:
                    row = fit_attempt(
                        dataset,
                        configs[factor],
                        gate,
                        scenario,
                        toy_index,
                        float(mass),
                        factor,
                        attempt,
                    )
                    by_factor[factor].append(row)

        attempt_sets = {
            factor: tuple(int(row["attempt"]) for row in by_factor[factor])
            for factor in UPPER_FACTORS
        }
        if len(set(attempt_sets.values())) != 1:
            raise StudyError("paired factors do not have a common attempt set")
        for attempt in attempt_sets[15]:
            seeds = {
                int(by_factor[factor][attempt]["optimizer_seed"])
                for factor in UPPER_FACTORS
            }
            if len(seeds) != 1:
                raise StudyError("paired factors do not have common optimizer seeds")

        for factor in UPPER_FACTORS:
            records = by_factor[factor]
            selected, replicates, top = select_branch(records, gate)
            for row in records:
                row["evaluated_attempt_count"] = len(records)
                row["maximum_lml_candidate_attempt"] = (
                    int(top["attempt"]) if top is not None else -1
                )
                row["matches_maximum_lml_branch"] = bool(
                    top is not None and branch_match(top, row, gate)
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
                        "study_id": spec["study_id"],
                        "scenario": scenario,
                        "background_toy_index": toy_index,
                        "mass_GeV": float(mass),
                        "mass_MeV": int(round(1000.0 * mass)),
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
                    "study_id": spec["study_id"],
                    "selected_attempt": int(selected["attempt"]),
                    "n_attempts": len(records),
                    "top_branch_replicates": int(replicates),
                    "optimizer_gate_status": "maximum_lml_reproduced",
                    "truth_model": str(
                        spec["background_toy_product"]["truth_model"]
                    ),
                    "support_preserved_40_300": True,
                    "common_seeds_across_factors": True,
                    "factor_selection_performed": False,
                }
            )
            selected_rows.append(chosen)

    attempts = pd.DataFrame(attempt_rows).sort_values(
        ["mass_MeV", "attempt", "upper_factor"]
    )
    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected = selected.sort_values(["mass_MeV", "upper_factor"])
    selected = _ensure_columns(
        selected,
        [
            "scenario",
            "background_toy_index",
            "mass_GeV",
            "mass_MeV",
            "upper_factor",
            "ell_opt",
            "sigma_x",
            "ell_over_sigma_x",
            "gp_lml",
        ],
    )
    exclusions = _ensure_columns(pd.DataFrame(exclusion_rows), EXCLUSION_COLUMNS)
    if not exclusions.empty:
        exclusions = exclusions.sort_values(["mass_MeV", "upper_factor"])

    validate_no_inference_columns(attempts, "task optimizer attempts")
    validate_no_inference_columns(selected, "task selected trajectories")
    validate_no_inference_columns(exclusions, "task exclusions")

    expected_states = len(MASS_GRID) * len(UPPER_FACTORS)
    if len(selected) + len(exclusions) != expected_states:
        raise StudyError("selected plus excluded states do not cover the task grid")
    nested = make_nested_lml(selected, [(scenario, toy_index)], gate)
    occupancy = make_occupancy(
        selected,
        ("scenario", "background_toy_index", "upper_factor"),
    )

    directory.mkdir(parents=True, exist_ok=True)
    products = {
        "optimizer_attempts.csv": attempts,
        "raw_ell_sigma_x_trajectories.csv": selected,
        "optimizer_exclusions.csv": exclusions,
        "bound_occupancy.csv": occupancy,
        "nested_lml.csv": nested,
    }
    for name, frame in products.items():
        atomic_csv(directory / name, frame)
    _read_and_validate_task_products(directory, scenario, toy_index)
    product_hashes = {
        name: sha256_file(directory / name) for name in PRODUCT_NAMES
    }
    product_key_columns = {
        "optimizer_attempts.csv": (
            "scenario",
            "background_toy_index",
            "mass_MeV",
            "upper_factor",
            "attempt",
        ),
        "raw_ell_sigma_x_trajectories.csv": (
            "scenario",
            "background_toy_index",
            "mass_MeV",
            "upper_factor",
        ),
        "optimizer_exclusions.csv": (
            "scenario",
            "background_toy_index",
            "mass_MeV",
            "upper_factor",
        ),
        "bound_occupancy.csv": (
            "scenario",
            "background_toy_index",
            "upper_factor",
        ),
        "nested_lml.csv": (
            "scenario",
            "background_toy_index",
            "mass_MeV",
            "lower_factor",
            "upper_factor",
        ),
    }
    product_metadata = {
        name: {
            "size_bytes": (directory / name).stat().st_size,
            "rows": len(frame),
            "schema_sha256": _frame_schema_hash(frame),
            "lattice_sha256": _frame_lattice_hash(
                frame, product_key_columns[name]
            ),
        }
        for name, frame in products.items()
    }
    result = {
        "schema_version": 1,
        "generation_uuid": str(uuid.uuid4()),
        "status": "complete",
        "scientific_status": (
            "optimizer_diagnostic_complete"
            if exclusions.empty
            else "optimizer_diagnostic_has_exclusions"
        ),
        "completed_utc": utc_now(),
        "study_id": spec["study_id"],
        "task_id": f"{scenario}__toy_{toy_index:04d}",
        "scenario": scenario,
        "background_toy_index": toy_index,
        "selected_rows": len(selected),
        "excluded_rows": len(exclusions),
        "attempt_rows": len(attempts),
        "scan_contract_sha256": contract_hash,
        "product_sha256": product_hashes,
        "product_metadata": product_metadata,
        "background_only": True,
        "support_gev": list(SUPPORT_GEV),
        "common_seeds_across_factors": True,
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "reserve_toys_consumed": False,
        "optimizer_gate": "reduced_length_only_pull_blind_v1",
        "pulls_produced": False,
        "cls_produced": False,
        "factor_selection_performed": False,
        "cached": False,
    }
    # Recheck the acyclic launcher -> lock -> core/input chain immediately
    # before publishing the success marker.  This catches on-disk drift during
    # a long-running task and avoids stamping a new-file hash from old code.
    result["external_lock"] = verify_external_lock()
    atomic_json(directory / "_SUCCESS.json", result)
    return result


def expected_tasks(
    scenarios: Sequence[str] = SCENARIOS,
    toy_start: int = 0,
    toy_stop: int = ACTIVE_N_TOYS,
) -> list[tuple[str, int]]:
    if toy_start < 0 or toy_stop > ACTIVE_N_TOYS or toy_stop <= toy_start:
        raise StudyError("toy interval must satisfy 0 <= start < stop <= 20")
    return [
        (scenario, toy_index)
        for scenario in scenarios
        for toy_index in range(toy_start, toy_stop)
    ]


def run_many(
    *,
    scenarios: Sequence[str],
    toy_start: int,
    toy_stop: int,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise StudyError("workers must be positive")
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise StudyError(f"unsupported scenario: {scenario}")
    preflight(validate_inventory=True)
    tasks = expected_tasks(scenarios, toy_start, toy_stop)
    results: list[dict[str, Any]] = []
    if workers == 1:
        for scenario, toy_index in tasks:
            results.append(
                run_task(
                    scenario,
                    toy_index,
                    force=force,
                    preflight_done=True,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_task,
                    scenario,
                    toy_index,
                    force=force,
                    preflight_done=True,
                ): (scenario, toy_index)
                for scenario, toy_index in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())
    return {
        "status": "complete",
        "tasks": len(results),
        "cached_tasks": sum(bool(result.get("cached")) for result in results),
        "tasks_with_exclusions": sum(
            int(result.get("excluded_rows", 0)) > 0 for result in results
        ),
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "reserve_toys_consumed": False,
        "factor_selection_performed": False,
    }


def task_status() -> dict[str, Any]:
    spec = load_spec()
    assert_spec_contract(spec)
    assert_reserve_outputs_untouched()
    contract_hash = scan_contract_hash(spec)
    records: list[dict[str, Any]] = []
    for scenario, toy_index in expected_tasks():
        directory = task_directory(scenario, toy_index)
        current, reason = validate_success(
            directory, contract_hash, scenario, toy_index
        )
        records.append(
            {
                "scenario": scenario,
                "background_toy_index": toy_index,
                "current": current,
                "status": reason,
            }
        )
    frame = pd.DataFrame(records)
    return {
        "status": "complete" if bool(frame.current.all()) else "incomplete",
        "expected_tasks": len(frame),
        "current_tasks": int(frame.current.sum()),
        "remaining_tasks": int((~frame.current).sum()),
        "reserved_untouched_scenario_toys": (
            len(SCENARIOS) * len(RESERVED_TOY_INDICES)
        ),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "status_counts": frame.status.value_counts().sort_index().to_dict(),
    }


def prepare(*, validate_inventory: bool = True) -> dict[str, Any]:
    validation = preflight(validate_inventory=validate_inventory)
    spec = load_spec()
    contract = scan_contract_payload(spec)
    contract_hash = canonical_json_hash(contract)
    rows: list[dict[str, Any]] = []
    for scenario, toy_index in expected_tasks():
        directory = task_directory(scenario, toy_index)
        current, reason = validate_success(
            directory, contract_hash, scenario, toy_index
        )
        rows.append(
            {
                "task_id": f"{scenario}__toy_{toy_index:04d}",
                "scenario": scenario,
                "background_toy_index": toy_index,
                "upper_factors": "15|20|25",
                "mass_grid_MeV": "50:20:250",
                "output_directory": str(directory),
                "command": (
                    f"python3 {LAUNCHER_PATH.name} run-task "
                    f"{scenario} {toy_index}"
                ),
                "current": current,
                "status": reason,
            }
        )
    manifest = pd.DataFrame(rows)
    reserve_manifest = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "background_toy_index": toy_index,
                "status": "reserved_untouched",
                "optimizer_task_permitted": False,
                "collection_permitted": False,
                "policy": "inventory-validation-only",
            }
            for scenario in SCENARIOS
            for toy_index in RESERVED_TOY_INDICES
        ]
    )
    QA.mkdir(parents=True, exist_ok=True)
    atomic_csv(QA / "task_manifest.csv", manifest)
    atomic_csv(QA / "reserved_toy_manifest.csv", reserve_manifest)
    atomic_json(QA / "scan_contract.json", contract)
    atomic_json(QA / "preflight.json", validation)
    return {
        "status": "pass",
        "tasks": len(manifest),
        "current_tasks": int(manifest.current.sum()),
        "reserved_untouched_scenario_toys": len(reserve_manifest),
        "task_manifest": str(QA / "task_manifest.csv"),
        "reserved_toy_manifest": str(QA / "reserved_toy_manifest.csv"),
        "scan_contract": str(QA / "scan_contract.json"),
        "scan_contract_sha256": contract_hash,
        "heavy_scan_launched": False,
    }


def _summarize_nested(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (scenario, lower, upper), group in frame.groupby(
        ["scenario", "lower_factor", "upper_factor"], sort=True
    ):
        comparable = group[group["comparable"].astype(bool)].copy()
        deltas = pd.to_numeric(
            comparable.get("delta_lml_upper_minus_lower"), errors="coerce"
        ).dropna()
        records.append(
            {
                "scenario": scenario,
                "lower_factor": int(lower),
                "upper_factor": int(upper),
                "rows": len(group),
                "comparable_rows": len(comparable),
                "unavailable_rows": len(group) - len(comparable),
                "delta_lml_min": float(deltas.min()) if len(deltas) else float("nan"),
                "delta_lml_median": float(deltas.median())
                if len(deltas)
                else float("nan"),
                "delta_lml_max": float(deltas.max()) if len(deltas) else float("nan"),
                "strict_nested_order_violations": int(
                    comparable.get("strict_nested_order_violation", pd.Series(dtype=bool))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "material_nested_order_violations": int(
                    comparable.get("material_nested_order_violation", pd.Series(dtype=bool))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "strict_lml_tolerance": STRICT_NESTED_LML_TOLERANCE,
            }
        )
    return pd.DataFrame(records)


def collect(*, allow_incomplete: bool = False) -> dict[str, Any]:
    spec = load_spec()
    assert_spec_contract(spec)
    assert_reserve_outputs_untouched()
    contract_hash = scan_contract_hash(spec)
    task_audit: list[dict[str, Any]] = []
    attempts_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    exclusion_parts: list[pd.DataFrame] = []
    current_tasks: list[tuple[str, int]] = []
    for scenario, toy_index in expected_tasks():
        directory = task_directory(scenario, toy_index)
        current, reason = validate_success(
            directory, contract_hash, scenario, toy_index
        )
        task_audit.append(
            {
                "scenario": scenario,
                "background_toy_index": toy_index,
                "current": current,
                "status": reason,
                "directory": str(directory),
            }
        )
        if not current:
            continue
        current_tasks.append((scenario, toy_index))
        attempts_parts.append(pd.read_csv(directory / "optimizer_attempts.csv"))
        selected_parts.append(
            pd.read_csv(directory / "raw_ell_sigma_x_trajectories.csv")
        )
        exclusion_parts.append(pd.read_csv(directory / "optimizer_exclusions.csv"))
    missing = len(SCENARIOS) * ACTIVE_N_TOYS - len(current_tasks)
    if missing and not allow_incomplete:
        raise StudyError(
            f"collection requires all 80 current active tasks; "
            f"{missing} are missing/stale"
        )
    if not current_tasks:
        raise StudyError("no current tasks are available to collect")

    attempts = pd.concat(attempts_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    exclusions = pd.concat(exclusion_parts, ignore_index=True)
    for label, frame in (
        ("optimizer attempts", attempts),
        ("selected trajectories", selected),
        ("optimizer exclusions", exclusions),
    ):
        validate_no_inference_columns(frame, label)

    seed_groups = attempts.groupby(
        ["scenario", "background_toy_index", "mass_MeV", "attempt"],
        sort=False,
    )
    if bool((seed_groups.optimizer_seed.nunique() != 1).any()):
        raise StudyError("collected factors do not share optimizer seeds")
    if bool((seed_groups.upper_factor.nunique() != len(UPPER_FACTORS)).any()):
        raise StudyError("collected paired attempt sets are incomplete")
    attempt_counts = attempts.groupby(
        ["scenario", "background_toy_index", "mass_MeV", "upper_factor"]
    ).size()
    if not set(attempt_counts.unique()).issubset({3, 5}):
        raise StudyError("optimizer state attempt counts are not 3 or 5")
    paired_counts = attempt_counts.unstack("upper_factor")
    if bool((paired_counts.nunique(axis=1) != 1).any()):
        raise StudyError("factor attempt counts are not paired")

    nested = make_nested_lml(selected, current_tasks, spec["optimizer_gate"])
    occupancy = make_occupancy(selected, ("scenario", "upper_factor"))
    occupancy_mass = make_occupancy(
        selected, ("scenario", "upper_factor", "mass_MeV")
    )
    nested_summary = _summarize_nested(nested)
    task_audit_frame = pd.DataFrame(task_audit)

    selected = selected.sort_values(
        ["scenario", "background_toy_index", "mass_MeV", "upper_factor"]
    )
    attempts = attempts.sort_values(
        [
            "scenario",
            "background_toy_index",
            "mass_MeV",
            "attempt",
            "upper_factor",
        ]
    )
    if not exclusions.empty:
        exclusions = exclusions.sort_values(
            ["scenario", "background_toy_index", "mass_MeV", "upper_factor"]
        )

    products = {
        "optimizer_attempt_ledger.csv": attempts,
        "raw_ell_sigma_x_trajectories.csv": selected,
        "optimizer_exclusion_ledger.csv": exclusions,
        "bound_occupancy_by_scenario_factor.csv": occupancy,
        "bound_occupancy_by_scenario_factor_mass.csv": occupancy_mass,
        "nested_lml_pointwise.csv": nested,
        "nested_lml_summary.csv": nested_summary,
        "task_product_audit.csv": task_audit_frame,
    }
    DERIVED.mkdir(parents=True, exist_ok=True)
    for name, frame in products.items():
        atomic_csv(DERIVED / name, frame)
    product_hashes = {name: sha256_file(DERIVED / name) for name in products}

    strict_violations = int(
        nested_summary.get("strict_nested_order_violations", pd.Series(dtype=int)).sum()
    )
    material_violations = int(
        nested_summary.get("material_nested_order_violations", pd.Series(dtype=int)).sum()
    )
    scientific_status = (
        "optimizer_diagnostic_complete"
        if missing == 0 and exclusions.empty and strict_violations == 0
        else "optimizer_diagnostic_attention_required"
    )
    result = {
        "status": "complete" if missing == 0 else "partial",
        "scientific_status": scientific_status,
        "collected_utc": utc_now(),
        "current_tasks": len(current_tasks),
        "missing_or_stale_tasks": missing,
        "attempt_rows": len(attempts),
        "selected_trajectory_rows": len(selected),
        "optimizer_exclusion_rows": len(exclusions),
        "strict_nested_lml_violations": strict_violations,
        "material_nested_lml_violations": material_violations,
        "scan_contract_sha256": contract_hash,
        "derived_sha256": product_hashes,
        "interpretation": (
            "Background-only conditional length-optimizer diagnostic. It is "
            "not a card-selection rule, coverage result, pull study, CLs "
            "result, or limit study."
        ),
        "background_only": True,
        "support_gev": list(SUPPORT_GEV),
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "reserve_toys_consumed": False,
        "optimizer_gate": "reduced_length_only_pull_blind_v1",
        "pulls_produced": False,
        "cls_produced": False,
        "factor_selection_performed": False,
    }
    atomic_json(DERIVED / "collection_summary.json", result)
    return result


def validate_command(*, validate_inventory: bool = True) -> dict[str, Any]:
    result = preflight(validate_inventory=validate_inventory)
    # Validate declared output schemas and task topology without invoking a GP.
    dummy_columns = pd.DataFrame(columns=EXCLUSION_COLUMNS)
    validate_no_inference_columns(dummy_columns, "declared exclusion schema")
    _self_test_csv_nullable_string_roundtrip()
    for prohibited_probe in (
        "sigma_A_reference",
        "gateSigmaA",
        "fitted_amplitude",
        "mean_pull",
        "CLs_alpha",
        "epsilon2_limit",
        "signalYieldEstimate",
    ):
        try:
            validate_no_inference_columns(
                pd.DataFrame(columns=[prohibited_probe]),
                "forbidden-column self-test",
            )
        except StudyError:
            pass
        else:
            raise StudyError(
                f"forbidden-column matcher failed for {prohibited_probe}"
            )
    contract = scan_contract_payload(load_spec())
    if contract["factor_selection_performed"] is not False:
        raise StudyError("length scanner must not perform card selection")
    return {
        "status": "pass",
        "preflight": result,
        "expected_resumable_tasks": len(SCENARIOS) * ACTIVE_N_TOYS,
        "expected_selected_states": (
            len(SCENARIOS)
            * ACTIVE_N_TOYS
            * len(MASS_GRID)
            * len(UPPER_FACTORS)
        ),
        "reserved_untouched_scenario_toys": (
            len(SCENARIOS) * len(RESERVED_TOY_INDICES)
        ),
        "heavy_scan_launched": False,
        "background_only": True,
        "support_gev": list(SUPPORT_GEV),
        "factor_selection_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "validate", "prepare"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--skip-toy-inventory",
            action="store_true",
            help="verify hashes/contracts but skip the full 4x25 ROOT inventory",
        )

    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("scenario", choices=SCENARIOS)
    task_parser.add_argument("toy_index", type=int)
    task_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    run_parser.add_argument("--toy-start", type=int, default=0)
    run_parser.add_argument("--toy-stop", type=int, default=ACTIVE_N_TOYS)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("status")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    verify_external_lock()
    args = parse_args()
    if args.command == "preflight":
        result = preflight(validate_inventory=not args.skip_toy_inventory)
    elif args.command == "validate":
        result = validate_command(validate_inventory=not args.skip_toy_inventory)
    elif args.command == "prepare":
        result = prepare(validate_inventory=not args.skip_toy_inventory)
    elif args.command == "run-task":
        result = run_task(args.scenario, args.toy_index, force=args.force)
    elif args.command == "run":
        scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
        result = run_many(
            scenarios=scenarios,
            toy_start=args.toy_start,
            toy_stop=args.toy_stop,
            workers=args.workers,
            force=args.force,
        )
    elif args.command == "status":
        result = task_status()
    elif args.command == "collect":
        result = collect(allow_incomplete=args.allow_incomplete)
    else:
        raise StudyError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
