#!/usr/bin/env python3
"""Run the v4.8.3 residual-structured conditional closure study.

The two source-fitted generating means are deliberately run in separate output
trees selected by ``--model``.  This driver preserves the frozen v4.2 analysis
card, the effective v4.5 matched-refit/fixed-histogram injection semantics, and
the v4.7.1 pull-blind optimizer-repeat gate.  The only runtime-card overlay is
the common extraction length-scale ceiling selected by the completed pilot;
the production-card value remains unchanged.

There are exactly twenty closure backgrounds, five reported exposure lanes,
five masses, and four injection strengths.  Consequently each model must
produce exactly 2,000 raw primary rows.  There is no reserve partition.  This
is a source-conditioned stress diagnostic, not coverage, expected limits,
exclusion, observed-data bias, or a production-card promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# Cap numerical libraries before NumPy/SciPy are imported.  The subprocess
# launcher repeats this contract so direct ``run-task`` use is equally safe.
for _thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_key] = "1"

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

INPUTS = HERE / "inputs"
PROTOCOL_PATH = HERE / "MODEL_PROTOCOL.json"
SOURCE_PRODUCT_PATH = HERE / "derived/source_fit_and_influence.json"
TOY_ROOT_PATH = INPUTS / "residual_structured_nested_toys.root"
TOY_MANIFEST_PATH = INPUTS / "residual_structured_nested_toys.manifest.json"
PILOT_DISPOSITION_PATH = (
    HERE / "derived/residual_length_pilot/common_ceiling_disposition.json"
)
REFERENCE_STUDY = HERE.parent / "v4p8_2021_functional_form_qualification_20260813"
REFERENCE_DRIVER = REFERENCE_STUDY / "run_rigid_study.py"
FROZEN_CARD_PATH = REFERENCE_STUDY / "inputs/frozen_v4p2_analysis_card.yaml"
RUNTIME_ROOT = REFERENCE_STUDY / "runtime_overlay"
RUNTIME_MANIFEST_PATH = REFERENCE_STUDY / "runtime_overlay_manifest.json"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

MODELS = ("knot_spline", "regional_blend")
SCENARIOS = (
    "2021_1pct",
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
MASS_GRID = (0.065, 0.090, 0.120, 0.180, 0.210)
STRENGTH_GRID = (0.0, 1.0, 3.0, 5.0)
N_TOYS = 20
EXPECTED_RAW_ROWS_PER_MODEL = 2000
BASE_SEED = 20260814
LEDGER_FILES = (
    "optimizer_attempts.csv",
    "accepted_rows.csv",
    "raw_primary_rows.csv",
    "exclusions.csv",
)
V4P6_COMPATIBILITY_CARD_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
REFERENCE_DRIVER_SHA256 = (
    "fb63c11517374cf1d6802dc8877412fb402b9ba0797f8d3bed6777ce96fcd887"
)
RUNTIME_MANIFEST_SHA256 = (
    "667390be8c2c5b79578c4ca933ff94fad289146432859f62ebf851a128a6c2e6"
)
V4P6_COMPATIBILITY_RUNTIME_SHA256 = {
    "hps_gpr/gpr.py": "1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5",
    "hps_gpr/io.py": "b36f8da7671a0fc0958b663e11d83a1a4421e90d1aab9b10e40c31ce078035db",
    "hps_gpr/injection.py": "3a38378379650b73159de8b98456a2bd91e5c374794805b0be39e86557e26bf2",
    "hps_gpr/statistics.py": "b8cbd484056925d64bed4d9a4ad3294fbac07d51079e5cb9ed565150b73c1ff2",
    "hps_gpr/template.py": "20c1fbaa632d5e03fa7527d0e4ddf8dc3ba8573927a8f981936721a731440e3e",
    "hps_gpr/config.py": "ec4f50345aebbf5c062e8daaefaaeca9b0e96df12f12b2d726172979df61cf9d",
}

SCENARIO_POLICY = {
    "2021_1pct": ("one_pct", 1, None, 1, 12_504_044),
    "2021_1pct_x10": ("one_pct", 10, "2021_1pct", 9, 125_040_440),
    "2021_1pct_x100": (
        "one_pct", 100, "2021_1pct_x10", 90, 1_250_404_400
    ),
    "2021_10pct": ("ten_pct", 1, None, 1, 141_251_508),
    "2021_10pct_x10": ("ten_pct", 10, "2021_10pct", 9, 1_412_515_080),
}

OPTIMIZER_GATE = {
    "version": "v4p7p1_reference_relative_v1",
    "reference_initial_attempts": 3,
    "maximum_attempts": 5,
    "top_branch_min_replicates": 2,
    "delta_lml_per_train_max": 0.001,
    "abs_log_length_ratio_max": 0.01,
    "abs_log_constant_ratio_max": 0.05,
    "abs_log_sigma_ratio_max": 0.02,
    "exact_start_abs_log_theta_max": 1e-8,
    "bound_ratio_window": 0.02,
    "sigma_over_reference_trigger": [0.5, 2.0],
    "reference_relative_lml_per_train_trigger": 0.02,
    "reference_relative_abs_log_length_trigger": 0.05,
    "reference_relative_abs_log_constant_trigger": 0.10,
    "covariance_min_eigenvalue_relative": -0.01,
    "minimum_accepted_per_cell_for_closure_claim": 19,
}


class StudyError(RuntimeError):
    """Raised when the frozen study contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise StudyError(f"missing frozen model protocol: {PROTOCOL_PATH}")
    payload = load_json(PROTOCOL_PATH)
    if int(payload.get("schema_version", -1)) != 1:
        raise StudyError("unsupported or missing MODEL_PROTOCOL schema_version")
    return payload


def model_runs(model: str) -> Path:
    return HERE / "runs/residual_closure" / str(model)


def model_derived(model: str) -> Path:
    return HERE / "derived/residual_closure" / str(model)


def model_qa(model: str) -> Path:
    return HERE / "qa/residual_closure" / str(model)


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
        payload, sort_keys=True, separators=(",", ":")
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


def configure_process() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def resolve_study_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return HERE / path


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise StudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise StudyError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}: {path}"
        )


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
    """Recursively verify adjacent path/SHA-256 declarations.

    Declarations such as ``base_sha256`` that have no adjacent path are
    provenance-only and are intentionally skipped.
    """

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


def toy_key(model: str, scenario: str, toy_index: int) -> str:
    return f"toys/{model}/{scenario}/toy_{int(toy_index):04d}"


def assert_protocol_contract(protocol: Mapping[str, Any]) -> None:
    if str(protocol.get("study_id")) != "v4p8p3_2021_residual_truths_20260814":
        raise StudyError("MODEL_PROTOCOL study_id drift")
    if bool(protocol.get("model_selection_uses_gpr_results", True)):
        raise StudyError("model selection must remain independent of GPR results")
    if not bool(protocol.get("model_selection_frozen_before_signal_audit")):
        raise StudyError("model selection was not frozen before signal audit")
    if set(protocol.get("models", {})) != set(MODELS):
        raise StudyError("MODEL_PROTOCOL must contain exactly the two frozen models")

    card = protocol.get("frozen_analysis_contract", {})
    expected_card = {
        "result_version": "v4.2",
        "result_commit": "4c7698e28f0c2c9eedf531a5b614ca727d7c305b",
        "integration_commit": "675b1c65be9238d6d97d7eeb0f09fd860404d13a",
        "card_sha256": V4P6_COMPATIBILITY_CARD_SHA256,
        "search_range_gev": [0.05, 0.25],
        "gp_support_range_gev": [0.04, 0.3],
        "pre_log": True,
        "alpha_model": "1/y",
        "neighborhood_rebin": 5,
        "blind_nsigma": 2.25,
        "gp_train_exclude_nsigma": 2.25,
        "kernel_ls_res_lower_factor_2021": 1.1,
        "production_kernel_ls_res_upper_factor_2021": 15.0,
        "n_restarts": 12,
        "injection_reference": "matched_refit_bonly",
        "signed_extraction": True,
        "test_statistic": "tilde_q_mu",
        "cls_alpha": 0.1,
    }
    if card != expected_card:
        raise StudyError("frozen v4.2/v4.5 analysis contract drift")

    toy = protocol.get("toy_contract", {})
    if (
        int(toy.get("closure_backgrounds_per_model_source", -1)) != N_TOYS
        or int(toy.get("pilot_backgrounds_per_model_source", -1)) != 3
        or tuple(toy.get("reported_scenarios", ())) != SCENARIOS
        or not bool(toy.get("pilot_and_closure_streams_independent"))
        or not bool(toy.get("nested_poisson_within_source_family"))
        or not bool(toy.get("source_families_distinct"))
        or not bool(toy.get("model_streams_distinct"))
        or str(toy.get("reserve_claim")) != "none"
    ):
        raise StudyError("toy-contract drift")

    grid = protocol.get("closure_grid", {})
    masses = tuple(float(value) for value in grid.get("masses_gev", ()))
    strengths = tuple(
        float(value) for value in grid.get("injected_reference_sigmas", ())
    )
    if masses != MASS_GRID or strengths != STRENGTH_GRID:
        raise StudyError("closure mass/injection grid drift")
    if (
        int(grid.get("raw_rows_expected", -1)) != 2 * EXPECTED_RAW_ROWS_PER_MODEL
        or int(grid.get("workers_max", -1)) != 2
        or int(grid.get("blas_threads", -1)) != 1
    ):
        raise StudyError("closure cardinality/resource contract drift")

    pilot = protocol.get("length_ceiling_pilot", {})
    if (
        not bool(pilot.get("background_only"))
        or tuple(map(int, pilot.get("toy_indices", ()))) != (0, 1, 2)
        or tuple(map(float, pilot.get("masses_gev", ()))) != (0.065, 0.12, 0.21)
        or tuple(map(float, pilot.get("upper_factors", ()))) != (15.0, 20.0, 25.0)
        or int(pilot.get("workers", -1)) != 1
        or not bool(pilot.get("common_ceiling_for_models_and_lanes"))
        or "factor 15 remains" not in str(pilot.get("production_card_impact", ""))
    ):
        raise StudyError("length-ceiling pilot contract drift")


def closure_grid(
    protocol: Mapping[str, Any] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    payload = load_protocol() if protocol is None else protocol
    grid = payload["closure_grid"]
    masses = tuple(map(float, grid["masses_gev"]))
    strengths = tuple(map(float, grid["injected_reference_sigmas"]))
    if masses != MASS_GRID or strengths != STRENGTH_GRID:
        raise StudyError("MODEL_PROTOCOL closure grid failed the frozen guard")
    return masses, strengths


def load_source_product(model: str) -> dict[str, Any]:
    if model not in MODELS:
        raise StudyError(f"unsupported model: {model}")
    if not SOURCE_PRODUCT_PATH.is_file():
        raise StudyError(f"missing frozen source-fit product: {SOURCE_PRODUCT_PATH}")
    product = load_json(SOURCE_PRODUCT_PATH)
    if (
        not bool(product.get("model_selection_frozen_before_injection"))
        or bool(product.get("model_selection_uses_gpr_results", True))
        or not bool(product.get("models", {}).get(model, {}).get(
            "conditional_toy_run_authorized", False
        ))
    ):
        raise StudyError(f"source-fit product does not authorize {model} toys")
    return product


def load_pilot_disposition() -> tuple[dict[str, Any], float]:
    if not PILOT_DISPOSITION_PATH.is_file():
        raise StudyError(
            "missing completed common-ceiling pilot disposition: "
            f"{PILOT_DISPOSITION_PATH}"
        )
    payload = load_json(PILOT_DISPOSITION_PATH)
    protocol = load_protocol()
    if (
        int(payload.get("schema_version", -1)) != 1
        or str(payload.get("status")) != "pass"
        or str(payload.get("study_id")) != str(protocol["study_id"])
        or not bool(payload.get("common_ceiling_for_models_and_lanes"))
        or bool(payload.get("inference_quantities_inspected", True))
        or float(
            payload.get("production_v4p2_upper_factor_unchanged", float("nan"))
        )
        != 15.0
    ):
        raise StudyError("length-ceiling pilot disposition is not complete/pass")
    selected = float(payload.get("selected_common_upper_factor", float("nan")))
    allowed = tuple(
        map(float, protocol["length_ceiling_pilot"]["upper_factors"])
    )
    if not np.isfinite(selected) or selected not in allowed:
        raise StudyError("pilot selected_common_upper_factor is missing or invalid")
    factor20_passed = bool(payload.get("factor20_gate_passed"))
    fallback_used = bool(payload.get("fallback_factor_used"))
    fallback = float(protocol["length_ceiling_pilot"]["fallback_factor"])
    expected_selected = 20.0 if factor20_passed else fallback
    if selected != expected_selected or fallback_used == factor20_passed:
        raise StudyError("pilot factor-20/fallback disposition is inconsistent")
    if payload.get("predeclared_thresholds") != protocol[
        "length_ceiling_pilot"
    ]["factor20_gate"]:
        raise StudyError("pilot disposition threshold contract drift")
    gate_results = payload.get("gate_results", {})
    if not isinstance(gate_results, Mapping) or factor20_passed != all(
        bool(value) for value in gate_results.values()
    ):
        raise StudyError("pilot disposition gate-result reduction is inconsistent")
    return payload, selected


def build_config(selected_upper_factor: float | None = None) -> Any:
    from hps_gpr.config import load_config

    if selected_upper_factor is None:
        _, selected_upper_factor = load_pilot_disposition()
    cfg = load_config(str(FROZEN_CARD_PATH))
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
    cfg.kernel_ls_res_upper_factor_by_dataset["2021"] = float(
        selected_upper_factor
    )
    cfg.blind_nsigma = 2.25
    cfg.gp_train_exclude_nsigma = 2.25
    cfg.scan_edge_guard_nsigma = 2.25
    cfg.scan_require_two_sidebands = True
    cfg.neighborhood_rebin = 5
    cfg.n_restarts = 12
    cfg.extract_allow_negative = True
    cfg.extract_background_mode = "profiled"
    cfg.eps2_density_nsigma = 1.64
    cfg.signal_model = "default"
    cfg.fail_fast = True
    cfg.debug_print = False
    cfg.save_plots = False
    return cfg


def assert_config(cfg: Any, selected_upper_factor: float | None = None) -> None:
    if selected_upper_factor is None:
        _, selected_upper_factor = load_pilot_disposition()
    checks = {
        "range_2021": tuple(map(float, cfg.range_2021)) == (0.05, 0.25),
        "data_range_2021": tuple(map(float, cfg.data_range_2021))
        == (0.04, 0.30),
        "pre_log": bool(cfg.pre_log),
        "alpha_model": str(cfg.alpha_model) == "1/y",
        "pre_zero_alpha": float(cfg.pre_zero_alpha) == 1.0,
        "lower_factor": float(
            cfg.kernel_ls_res_lower_factor_by_dataset["2021"]
        )
        == 1.1,
        "upper_factor": float(
            cfg.kernel_ls_res_upper_factor_by_dataset["2021"]
        )
        == float(selected_upper_factor),
        "blind_nsigma": float(cfg.blind_nsigma) == 2.25,
        "gp_train_exclude_nsigma": float(cfg.gp_train_exclude_nsigma) == 2.25,
        "scan_edge_guard_nsigma": float(cfg.scan_edge_guard_nsigma) == 2.25,
        "two_sidebands": bool(cfg.scan_require_two_sidebands),
        "neighborhood_rebin": int(cfg.neighborhood_rebin) == 5,
        "n_restarts": int(cfg.n_restarts) == 12,
        "signed_amplitude": bool(cfg.extract_allow_negative),
        "profiled_background": str(cfg.extract_background_mode) == "profiled",
        "density": float(cfg.eps2_density_nsigma) == 1.64,
        "signal_model": str(cfg.signal_model) == "default",
        "cls_alpha": float(cfg.cls_alpha) == 0.10,
        "cls_mode": str(cfg.cls_mode) == "asymptotic",
        "no_limit_bands": not bool(cfg.make_ul_bands)
        and not bool(cfg.do_combined_bands)
        and not bool(cfg.make_eps2_bands),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise StudyError("frozen-card assertion failed: " + ", ".join(failed))


def toy_seed_words(*parts: object) -> list[int]:
    material = "|".join([str(BASE_SEED), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()


def validate_toy_product(model: str) -> dict[str, Any]:
    """Validate the complete two-model toy product and the selected inventory."""

    import uproot

    if model not in MODELS:
        raise StudyError(f"unsupported model: {model}")
    if not TOY_MANIFEST_PATH.is_file() or not TOY_ROOT_PATH.is_file():
        raise StudyError(
            "missing residual toy ROOT/manifest; run the frozen generator before "
            "closure preflight"
        )
    manifest = load_json(TOY_MANIFEST_PATH)
    protocol = load_protocol()
    expected_top_level = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "claim_boundary": protocol["claim_boundary"],
        "promotion_scope": "requested conditional stress only",
        "base_seed": BASE_SEED,
        "models": list(MODELS),
        "reported_scenarios": list(SCENARIOS),
        "phase_counts_per_model_scenario": {"pilot": 3, "toys": N_TOYS},
        "closure_background_clusters": 80,
        "reserve_backgrounds": 0,
        "nested_poisson_within_source_family": True,
        "source_families_distinct": True,
        "model_streams_distinct": True,
        "pilot_and_closure_streams_independent": True,
    }
    drift = [
        key
        for key, expected in expected_top_level.items()
        if manifest.get(key) != expected
    ]
    if drift:
        raise StudyError("residual toy manifest contract drift: " + ", ".join(drift))

    provenance_records = {
        "protocol": (PROTOCOL_PATH, sha256_file(PROTOCOL_PATH)),
        "source_fit_and_influence": (
            SOURCE_PRODUCT_PATH,
            sha256_file(SOURCE_PRODUCT_PATH),
        ),
        "builder": (HERE / "build_residual_toys.py", None),
        "root": (TOY_ROOT_PATH, None),
    }
    for name, (expected_path, fixed_hash) in provenance_records.items():
        record = manifest.get(name, {})
        if not isinstance(record, Mapping):
            raise StudyError(f"toy manifest {name} provenance is missing")
        declared_path = record.get("path")
        declared_hash = record.get("sha256")
        if not isinstance(declared_path, str) or not isinstance(declared_hash, str):
            raise StudyError(f"toy manifest {name} path/hash is missing")
        if resolve_study_path(declared_path).resolve() != expected_path.resolve():
            raise StudyError(f"toy manifest {name} path drift")
        require_hash(expected_path, declared_hash, f"toy manifest {name}")
        if fixed_hash is not None and declared_hash != fixed_hash:
            raise StudyError(f"toy manifest {name} hash does not bind current input")
    root_record = manifest["root"]
    if int(root_record.get("histograms_expected", -1)) != 240:
        raise StudyError("toy ROOT histogram cardinality declaration drift")

    source_edges: np.ndarray | None = None
    for family in ("one_pct", "ten_pct"):
        source_record = protocol["source_inputs"][family]
        source_path = resolve_study_path(str(source_record["path"]))
        require_hash(source_path, str(source_record["sha256"]), f"{family} source")
        with uproot.open(source_path) as source_file:
            _, edges = source_file[str(source_record["histogram"])].to_numpy()
        edges = np.asarray(edges, dtype=float)
        if source_edges is None:
            source_edges = edges
        elif not np.array_equal(source_edges, edges):
            raise StudyError("native 1% and 10% source histogram edges differ")
    if source_edges is None:
        raise StudyError("source histogram edges are unavailable")
    edge_record = manifest.get("edges", {})
    if (
        int(edge_record.get("n_bins", -1)) != source_edges.size - 1
        or float(edge_record.get("low_GeV", float("nan"))) != float(source_edges[0])
        or float(edge_record.get("high_GeV", float("nan"))) != float(source_edges[-1])
        or str(edge_record.get("sha256_float64", ""))
        != array_hash(source_edges, "<f8")
    ):
        raise StudyError("toy manifest edge contract drift")

    truth_rows = list(manifest.get("truths", ()))
    toy_rows = list(manifest.get("toys", ()))
    truth_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in truth_rows:
        key = (str(row.get("model")), str(row.get("scenario")))
        if key in truth_map:
            raise StudyError(f"duplicate truth-manifest state: {key}")
        truth_map[key] = row
    toy_map: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in toy_rows:
        key = (
            str(row.get("phase")),
            str(row.get("model")),
            str(row.get("scenario")),
            int(row.get("toy_index", -1)),
        )
        if key in toy_map:
            raise StudyError(f"duplicate toy-manifest state: {key}")
        toy_map[key] = row
    expected_truth_states = {
        (selected_model, scenario)
        for selected_model in MODELS
        for scenario in SCENARIOS
    }
    expected_toy_states = {
        (phase, selected_model, scenario, toy_index)
        for phase, count in (("toys", N_TOYS), ("pilot", 3))
        for selected_model in MODELS
        for scenario in SCENARIOS
        for toy_index in range(count)
    }
    if set(truth_map) != expected_truth_states:
        raise StudyError("truth-manifest inventory is incomplete or contains extras")
    if set(toy_map) != expected_toy_states:
        raise StudyError(
            "toy manifest must contain exactly 20 closure and 3 pilot rows per "
            "model/scenario"
        )

    leaf_keys = {
        f"truth/{selected_model}/{scenario}_mean"
        for selected_model in MODELS
        for scenario in SCENARIOS
    } | {
        f"{phase}/{selected_model}/{scenario}/toy_{toy_index:04d}"
        for phase, count in (("toys", N_TOYS), ("pilot", 3))
        for selected_model in MODELS
        for scenario in SCENARIOS
        for toy_index in range(count)
    }
    count_hashes: dict[str, str] = {}
    cached: dict[tuple[str, str, str, int], np.ndarray] = {}
    seed_tokens: set[tuple[int, ...]] = set()
    with uproot.open(TOY_ROOT_PATH) as root_file:
        actual_histograms = {
            key.split(";")[0]
            for key, class_name in root_file.classnames(recursive=True).items()
            if str(class_name).startswith("TH1")
        }
        if actual_histograms != leaf_keys:
            raise StudyError("toy ROOT histogram inventory differs from manifest")

        for selected_model in MODELS:
            for scenario in SCENARIOS:
                family, multiplier, parent, increment_multiplier, expected_total = (
                    SCENARIO_POLICY[scenario]
                )
                row = truth_map[(selected_model, scenario)]
                key = f"truth/{selected_model}/{scenario}_mean"
                if (
                    row.get("key") != key
                    or row.get("source_family") != family
                    or int(row.get("multiplier", -1)) != multiplier
                    or row.get("parent_scenario") != parent
                    or int(row.get("increment_multiplier", -1))
                    != increment_multiplier
                ):
                    raise StudyError(f"truth manifest semantics drift: {key}")
                values, edges = root_file[key].to_numpy(flow=False)
                values = np.asarray(values, dtype=float)
                edges = np.asarray(edges, dtype=float)
                if (
                    not np.array_equal(edges, source_edges)
                    or np.any(~np.isfinite(values))
                    or np.any(values < 0.0)
                    or array_hash(values, "<f8")
                    != str(row.get("mean_sha256_float64", ""))
                    or not math.isclose(
                        float(np.sum(values)),
                        float(row.get("total", float("nan"))),
                        rel_tol=0.0,
                        abs_tol=max(1e-6, float(expected_total) * 1e-12),
                    )
                    or not math.isclose(
                        float(np.sum(values)),
                        float(expected_total),
                        rel_tol=0.0,
                        abs_tol=max(1e-3, float(expected_total) * 1e-9),
                    )
                ):
                    raise StudyError(f"truth histogram content/hash drift: {key}")

        for phase, count in (("toys", N_TOYS), ("pilot", 3)):
            for selected_model in MODELS:
                for scenario in SCENARIOS:
                    family, multiplier, parent, increment_multiplier, _ = (
                        SCENARIO_POLICY[scenario]
                    )
                    stage = (
                        "parent"
                        if parent is None
                        else "increment_x9"
                        if increment_multiplier == 9
                        else "increment_x90"
                    )
                    for toy_index in range(count):
                        row = toy_map[(phase, selected_model, scenario, toy_index)]
                        key = (
                            f"{phase}/{selected_model}/{scenario}/"
                            f"toy_{toy_index:04d}"
                        )
                        expected_namespace = [
                            selected_model,
                            phase,
                            family,
                            toy_index,
                            stage,
                        ]
                        expected_seed = toy_seed_words(*expected_namespace)
                        if (
                            row.get("key") != key
                            or row.get("source_family") != family
                            or int(row.get("multiplier", -1)) != multiplier
                            or row.get("parent_scenario") != parent
                            or int(row.get("increment_multiplier", -1))
                            != increment_multiplier
                            or row.get("namespace") != expected_namespace
                            or row.get("seed_words_uint32") != expected_seed
                        ):
                            raise StudyError(f"toy manifest semantics/seed drift: {key}")
                        seed_token = tuple(map(int, expected_seed))
                        if seed_token in seed_tokens:
                            raise StudyError("pilot/closure/model streams reuse an RNG seed")
                        seed_tokens.add(seed_token)
                        values, edges = root_file[key].to_numpy(flow=False)
                        values = np.asarray(values, dtype=float)
                        rounded = np.rint(values).astype(np.int64)
                        if (
                            not np.array_equal(edges, source_edges)
                            or np.any(~np.isfinite(values))
                            or np.any(rounded < 0)
                            or not np.array_equal(values, rounded.astype(float))
                            or array_hash(rounded, "<i8")
                            != str(row.get("counts_sha256_int64", ""))
                            or int(np.sum(rounded, dtype=np.int64))
                            != int(row.get("total", -1))
                        ):
                            raise StudyError(f"toy histogram content/hash drift: {key}")
                        increment = (
                            rounded
                            if parent is None
                            else rounded
                            - cached[(phase, selected_model, parent, toy_index)]
                        )
                        if np.any(increment < 0):
                            raise StudyError(f"negative nested increment: {key}")
                        declared_increment_hash = row.get("increment_sha256_int64")
                        if parent is None:
                            if declared_increment_hash is not None:
                                raise StudyError(f"parent toy declares an increment hash: {key}")
                        elif array_hash(increment, "<i8") != str(
                            declared_increment_hash or ""
                        ):
                            raise StudyError(f"nested increment hash mismatch: {key}")
                        usable = rounded.size // 5 * 5
                        rebinned = rounded[:usable].reshape(-1, 5).sum(axis=1)
                        rebinned_edges = edges[: usable + 1 : 5]
                        centers = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
                        support = (centers >= 0.04) & (centers < 0.30)
                        if np.any(rebinned[support] <= 0):
                            raise StudyError(f"nonpositive pre-log support count: {key}")
                        cached[(phase, selected_model, scenario, toy_index)] = rounded
                        count_hashes[key] = str(row["counts_sha256_int64"])

    return {
        "status": "pass",
        "model": model,
        "root": str(TOY_ROOT_PATH),
        "root_sha256": str(manifest["root"]["sha256"]),
        "manifest": str(TOY_MANIFEST_PATH),
        "manifest_sha256": sha256_file(TOY_MANIFEST_PATH),
        "count_inventory_sha256": canonical_json_hash(count_hashes),
        "truth_histograms": len(expected_truth_states),
        "closure_histograms": 2 * len(SCENARIOS) * N_TOYS,
        "pilot_histograms": 2 * len(SCENARIOS) * 3,
        "analysis_histograms_for_model": len(SCENARIOS) * N_TOYS,
        "closure_indices": list(range(N_TOYS)),
        "reserve_indices": [],
        "nesting_checked": True,
        "independent_stream_seed_derivation_checked": True,
    }


def preflight(model: str, *, validate_inventory: bool = True) -> dict[str, Any]:
    if model not in MODELS:
        raise StudyError(f"unsupported model: {model}")
    protocol = load_protocol()
    assert_protocol_contract(protocol)
    checks: dict[str, bool] = {}

    require_hash(
        FROZEN_CARD_PATH,
        V4P6_COMPATIBILITY_CARD_SHA256,
        "frozen v4.2 analysis card",
    )
    checks["frozen_v4p2_card"] = True
    require_hash(REFERENCE_DRIVER, REFERENCE_DRIVER_SHA256, "authoritative v4.8 runner")
    checks["authoritative_v4p8_runner"] = True

    source_product = load_source_product(model)
    verify_declared_hashes(
        source_product.get("input_validation", {}),
        label="source_fit.input_validation",
        checks=checks,
    )
    for path_key, hash_key in (
        ("driver_path", "driver_sha256"),
        ("implementation_path", "implementation_sha256"),
    ):
        value, expected = source_product.get(path_key), source_product.get(hash_key)
        if not isinstance(value, str) or not isinstance(expected, str):
            raise StudyError(f"source-fit {path_key}/{hash_key} provenance is missing")
        require_hash(resolve_study_path(value), expected, f"source-fit {path_key}")
        checks[f"source_fit.{path_key}"] = True
    checks["source_fit_authorization"] = True

    require_hash(
        RUNTIME_MANIFEST_PATH,
        RUNTIME_MANIFEST_SHA256,
        "v4.8 runtime overlay manifest",
    )
    runtime_manifest = load_json(RUNTIME_MANIFEST_PATH)
    declared_runtime_files = runtime_manifest.get("files", {})
    actual_runtime_files = {
        str(path.relative_to(RUNTIME_ROOT)): sha256_file(path)
        for path in sorted((RUNTIME_ROOT / "hps_gpr").glob("*.py"))
    }
    if (
        int(runtime_manifest.get("schema_version", -1)) != 1
        or not isinstance(declared_runtime_files, Mapping)
        or dict(declared_runtime_files) != actual_runtime_files
    ):
        raise StudyError("v4.8 runtime overlay file-set or content drift")
    checks["runtime_overlay_full_inventory"] = True
    for module_path, expected in V4P6_COMPATIBILITY_RUNTIME_SHA256.items():
        if str(declared_runtime_files.get(module_path, "")) != expected:
            raise StudyError(f"runtime compatibility hash drift: {module_path}")
        module_name = module_path.replace("/", ".")[:-3]
        imported = importlib.import_module(module_name)
        imported_path = Path(str(getattr(imported, "__file__", ""))).resolve()
        require_hash(imported_path, expected, f"imported runtime {module_path}")
        if RUNTIME_ROOT.resolve() not in imported_path.parents:
            raise StudyError(f"runtime overlay was not imported: {imported_path}")
        checks[f"runtime.{module_path}"] = True
    from hps_gpr.gpr import fit_gpr
    from hps_gpr.io import BlindPrediction

    if "random_state" not in inspect.signature(fit_gpr).parameters:
        raise StudyError("audited fit_gpr(random_state=...) runtime is not active")
    required_prediction_fields = {
        "optimizer_warning_count", "optimizer_warnings", "ls_init", "const_init"
    }
    if not required_prediction_fields.issubset(
        set(getattr(BlindPrediction, "__dataclass_fields__", {}))
    ):
        raise StudyError("audited BlindPrediction instrumentation is not active")
    checks["runtime_import_resolution"] = True

    pilot, selected_upper_factor = load_pilot_disposition()
    pilot_checks: dict[str, bool] = {}
    verify_declared_hashes(pilot, label="pilot_disposition", checks=pilot_checks)
    checks.update(pilot_checks)
    pilot_text = json.dumps(pilot, sort_keys=True, default=str)
    if TOY_MANIFEST_PATH.is_file() and sha256_file(TOY_MANIFEST_PATH) not in pilot_text:
        raise StudyError("pilot disposition is not bound to the current toy manifest")
    checks["completed_common_ceiling_disposition"] = True

    cfg = build_config(selected_upper_factor)
    assert_config(cfg, selected_upper_factor)
    checks["frozen_card_plus_selected_ceiling_assertions"] = True
    inventory = validate_toy_product(model) if validate_inventory else None
    if inventory is not None:
        checks["toy_inventory"] = True
    return {
        "status": "pass",
        "validated_utc": utc_now(),
        "model": model,
        "checks": checks,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_fit_sha256": sha256_file(SOURCE_PRODUCT_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "authoritative_v4p8_runner_sha256": sha256_file(REFERENCE_DRIVER),
        "frozen_v4p2_card_sha256": sha256_file(FROZEN_CARD_PATH),
        "runtime_manifest_sha256": sha256_file(RUNTIME_MANIFEST_PATH),
        "runtime_overlay_file_sha256": dict(declared_runtime_files),
        "toy_manifest_sha256": (
            sha256_file(TOY_MANIFEST_PATH) if TOY_MANIFEST_PATH.is_file() else None
        ),
        "toy_root_sha256": (
            sha256_file(TOY_ROOT_PATH) if TOY_ROOT_PATH.is_file() else None
        ),
        "pilot_disposition_sha256": sha256_file(PILOT_DISPOSITION_PATH),
        "selected_extraction_upper_factor": selected_upper_factor,
        "production_card_upper_factor": 15.0,
        "optimizer_gate": dict(OPTIMIZER_GATE),
        "toy_inventory": inventory,
    }


def _load_histogram(path: Path, key: str) -> Any:
    from hps_gpr.funcform_toys import load_funcform_toy_hist

    container, name = key.rsplit("/", 1)
    return load_funcform_toy_hist(
        str(path), container=container, toy_name=name
    )


def make_toy_dataset(model: str, scenario: str, toy_index: int, cfg: Any) -> Any:
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import FuncFormToySpec, build_funcform_toy_dataset

    key = toy_key(model, scenario, int(toy_index))
    histogram = _load_histogram(TOY_ROOT_PATH, key)
    base = make_datasets(cfg)["2021"]
    toy_spec = FuncFormToySpec(
        source_root=str(TOY_ROOT_PATH),
        container=f"toys/{model}/{scenario}",
        function_tag=model,
        toy_name=f"toy_{int(toy_index):04d}",
        toy_index=int(toy_index),
    )
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def covariance_diagnostics(covariance: Any) -> dict[str, Any]:
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
    return {
        "covariance_valid": bool(
            np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8 * scale)
            and relative >= -1e-2
        ),
        "covariance_min_eigenvalue": minimum,
        "covariance_min_eigenvalue_relative": relative,
    }


def training_geometry(
    x_full: Any,
    y_full: Any,
    train_mask: Any,
    edges_full: Any,
    *,
    mass: float,
) -> dict[str, Any]:
    x = np.asarray(x_full, dtype=float).reshape(-1)
    y = np.asarray(y_full, dtype=float).reshape(-1)
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    edges = np.asarray(edges_full, dtype=float).reshape(-1)
    if x.shape != y.shape or x.shape != mask.shape:
        raise StudyError(f"training geometry shape mismatch at {mass:.6g} GeV")
    if edges.shape != (x.size + 1,):
        raise StudyError(f"training edge shape mismatch at {mass:.6g} GeV")
    selected_x = x[mask]
    selected_y = y[mask]
    if selected_x.size == 0:
        raise StudyError(f"empty GP training set at {mass:.6g} GeV")
    if not np.all(np.isfinite(selected_y)) or np.any(selected_y <= 0):
        raise StudyError(
            f"pre_log requires strictly positive finite GP training counts at "
            f"{mass:.6g} GeV"
        )
    widths = np.diff(edges)
    return {
        "n_train": int(np.count_nonzero(mask)),
        "n_train_low": int(np.count_nonzero(mask & (x < float(mass)))),
        "n_train_high": int(np.count_nonzero(mask & (x > float(mass)))),
        "train_domain_lo": float(edges[0]),
        "train_domain_hi": float(edges[-1]),
        "bin_width_median": float(np.median(widths)),
        "n_zero_train": int(np.count_nonzero(selected_y <= 0)),
        "min_y_train": float(np.min(selected_y)),
        "max_y_train": float(np.max(selected_y)),
        "training_counts_sha256": array_hash(selected_y, "<f8"),
    }


def kernel_bound_diagnostics(
    *,
    ls_value: float,
    ls_lower: float,
    ls_upper: float,
    const_value: float,
    const_lower: float,
    const_upper: float,
) -> dict[str, Any]:
    def near(value: float, bound: float) -> bool:
        return bool(
            np.isfinite(value)
            and np.isfinite(bound)
            and bound > 0
            and np.isclose(value, bound, rtol=1e-3, atol=1e-12)
        )

    return {
        "ls_at_lower": near(ls_value, ls_lower),
        "ls_at_upper": near(ls_value, ls_upper),
        "const_at_lower": near(const_value, const_lower),
        "const_at_upper": near(const_value, const_upper),
    }


def branch_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> bool:
    required = ("gp_lml", "gp_ls", "gp_const", "sigma_A", "n_train")
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
    for key, limit in (
        ("gp_ls", gate["abs_log_length_ratio_max"]),
        ("gp_const", gate["abs_log_constant_ratio_max"]),
        ("sigma_A", gate["abs_log_sigma_ratio_max"]),
    ):
        left, right = float(first[key]), float(second[key])
        if left <= 0 or right <= 0:
            return False
        if abs(math.log(left / right)) > float(limit):
            return False
    return True


def select_branch(
    records: list[dict[str, Any]],
    gate: Mapping[str, Any],
    *,
    require_replication: bool,
) -> tuple[dict[str, Any] | None, int]:
    usable = [
        row
        for row in records
        if bool(row.get("fit_ok"))
        and bool(row.get("covariance_valid"))
        and np.isfinite(float(row.get("gp_lml", np.nan)))
        and np.isfinite(float(row.get("sigma_A", np.nan)))
        and float(row.get("sigma_A", 0.0)) > 0
    ]
    if not usable:
        return None, 0
    selected = max(usable, key=lambda row: float(row["gp_lml"]))
    replicates = sum(branch_match(selected, row, gate) for row in usable)
    if require_replication and replicates < int(gate["top_branch_min_replicates"]):
        return None, replicates
    return selected, replicates


def reference_attempt(
    ds: Any,
    cfg: Any,
    model: str,
    scenario: str,
    toy_index: int,
    mass: float,
    attempt: int,
) -> tuple[dict[str, Any], Any | None]:
    from hps_gpr.conversion import A_from_epsilon2
    from hps_gpr.injection import (
        _fit_A_for_extraction,
        _prediction_blind_mask,
        _prediction_y_full_bonly,
        _sigmaA_reference,
    )
    from hps_gpr.io import estimate_background_for_dataset
    from hps_gpr.template import build_window_template_from_full

    optimizer_seed = stable_seed(
        "v4p7_restart_v1",
        model,
        scenario,
        int(toy_index),
        f"{float(mass):.9f}",
        "reference",
        int(attempt),
    )
    cfg.gp_optimizer_random_state = int(optimizer_seed)
    base = {
        "model": model,
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "inj_nsigma": 0.0,
        "strength": 0.0,
        "role": "reference_bonly",
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "optimizer_restarts": 12,
        "fit_ok": False,
        "refit_fallback_used": False,
        "error": "",
    }
    try:
        pred = estimate_background_for_dataset(
            ds, float(mass), cfg, restarts=12, optimize=True
        )
        blind_mask = _prediction_blind_mask(pred)
        x_full = np.asarray(pred.x_full, dtype=float)
        y_full = np.asarray(_prediction_y_full_bonly(pred), dtype=float)
        train_half_width = (
            float(cfg.gp_train_exclude_nsigma) * float(pred.sigma_val)
        )
        train_mask = (x_full < float(mass) - train_half_width) | (
            x_full > float(mass) + train_half_width
        )
        geometry = training_geometry(
            x_full,
            y_full,
            train_mask,
            pred.edges_full,
            mass=float(mass),
        )
        if int(geometry["n_train"]) != int(pred.n_train):
            raise StudyError(
                f"reference training-mask cardinality mismatch at {mass:.6g} GeV"
            )
        template_window, _ = build_window_template_from_full(
            pred.edges_full,
            blind_mask,
            float(mass),
            pred.sigma_val,
            config=cfg,
        )
        observed = y_full[blind_mask]
        fit = _fit_A_for_extraction(
            cfg,
            observed,
            pred.mu,
            pred.cov,
            template_window,
            allow_negative=True,
        )
        sigma_a = float(fit["sigma_A"])
        sigma_reference = float(
            _sigmaA_reference(pred, float(mass), source="asimov", config=cfg)
        )
        density = float(pred.integral_density)
        record = {
            **base,
            "fit_ok": bool(fit.get("success", False)),
            "gp_lml": float(pred.lml),
            "gp_ls": float(pred.ls_opt),
            "gp_const": float(pred.const_opt),
            "gp_const_lo": float(pred.const_lo),
            "gp_const_hi": float(pred.const_hi),
            "gp_ls_lo": float(pred.ls_lo),
            "gp_ls_hi": float(pred.ls_hi),
            "gp_ls_init": float(pred.ls_init),
            "gp_const_init": float(pred.const_init),
            "sigma_A": sigma_a,
            "sigmaA_reference": sigma_reference,
            "A_hat": float(fit["A_hat"]),
            "Zhat": float(fit["A_hat"]) / sigma_a,
            "pull": float(fit["A_hat"]) / sigma_a,
            "amplitude_nll": float(fit.get("nll", np.nan)),
            "n_blind": int(pred.n_blind),
            "integral_density": density,
            "A_per_eps2_unit": float(
                A_from_epsilon2(ds, float(mass), 1.0, density)
            ),
            "optimizer_warning_count": int(pred.optimizer_warning_count),
            "optimizer_warnings": str(pred.optimizer_warnings),
            **geometry,
            **kernel_bound_diagnostics(
                ls_value=float(pred.ls_opt),
                ls_lower=float(pred.ls_lo),
                ls_upper=float(pred.ls_hi),
                const_value=float(pred.const_opt),
                const_lower=float(pred.const_lo),
                const_upper=float(pred.const_hi),
            ),
            **covariance_diagnostics(pred.cov),
        }
        return record, pred
    except Exception as exc:
        return {
            **base,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }, None


def refit_attempt(
    ds: Any,
    cfg: Any,
    reference_pred: Any,
    reference_row: Mapping[str, Any],
    model: str,
    scenario: str,
    toy_index: int,
    mass: float,
    z_value: float,
    attempt: int,
) -> dict[str, Any]:
    from hps_gpr.gpr import (
        fit_gpr,
        make_kernel_for_dataset,
        predict_counts_from_log_gpr,
    )
    from hps_gpr.injection import (
        _fit_A_for_extraction,
        _fixed_hist_background_counts,
        _gpr_fit_diagnostics,
        _inject_counts_from_template,
        _prediction_blind_mask,
        _prediction_y_full_bonly,
    )
    from hps_gpr.template import build_window_template_from_full

    optimizer_seed = stable_seed(
        "v4p7_restart_v1",
        model,
        scenario,
        int(toy_index),
        f"{float(mass):.9f}",
        f"z{float(z_value):.1f}",
        int(attempt),
    )
    signal_seed = stable_seed(
        "v4p7_signal_v1",
        model,
        scenario,
        int(toy_index),
        f"{float(mass):.9f}",
        f"z{float(z_value):.1f}",
    )
    sigma_reference = float(reference_row["sigma_A"])
    injected = float(z_value) * sigma_reference
    base = {
        "model": model,
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "inj_nsigma": float(z_value),
        "role": "injected_refit",
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "signal_seed": int(signal_seed),
        "optimizer_restarts": 12,
        "fit_ok": False,
        "refit_fallback_used": False,
        "error": "",
        "strength": injected,
        "sigmaA_reference": sigma_reference,
        "reference_attempt_selected": int(reference_row["attempt"]),
        "reference_gp_lml": float(reference_row["gp_lml"]),
        "reference_gp_ls": float(reference_row["gp_ls"]),
        "reference_gp_const": float(reference_row["gp_const"]),
        "A_per_eps2_unit": float(reference_row["A_per_eps2_unit"]),
        "integral_density": float(reference_row["integral_density"]),
    }
    try:
        blind_mask = _prediction_blind_mask(reference_pred)
        x_full = np.asarray(reference_pred.x_full, dtype=float)
        background = _fixed_hist_background_counts(
            _prediction_y_full_bonly(reference_pred),
            dataset_key="2021",
            mass=float(mass),
        )
        template_window, template_full = build_window_template_from_full(
            reference_pred.edges_full,
            blind_mask,
            float(mass),
            reference_pred.sigma_val,
            config=cfg,
        )
        rng = np.random.default_rng(int(signal_seed))
        signal_full, n_signal_full, _ = _inject_counts_from_template(
            template_full, injected, rng, "poisson"
        )
        signal_full = np.asarray(signal_full, dtype=int)
        y_toy = np.asarray(background, dtype=int) + signal_full

        # Keep this tied to the effective card.  A literal 2.25 here would let
        # the refit mask silently drift away from a future audited card value.
        train_half_width = (
            float(cfg.gp_train_exclude_nsigma)
            * float(reference_pred.sigma_val)
        )
        train_mask = (x_full < float(mass) - train_half_width) | (
            x_full > float(mass) + train_half_width
        )
        kernel = make_kernel_for_dataset(ds, cfg, mass=float(mass))
        gpr = fit_gpr(
            x_full[train_mask],
            y_toy[train_mask].astype(float),
            cfg,
            restarts=12,
            kernel=kernel,
            optimize=True,
            random_state=int(optimizer_seed),
        )
        mu, covariance = predict_counts_from_log_gpr(
            gpr, x_full[blind_mask], cfg
        )
        fit = _fit_A_for_extraction(
            cfg,
            y_toy[blind_mask],
            mu,
            covariance,
            template_window,
            allow_negative=True,
        )
        diagnostics = _gpr_fit_diagnostics(gpr)
        sigma_a = float(fit["sigma_A"])
        a_hat = float(fit["A_hat"])
        initial_kernel = getattr(gpr, "kernel", None)
        initial_const = float(
            getattr(
                getattr(initial_kernel, "k1", None),
                "constant_value",
                np.nan,
            )
        )
        initial_ls = float(
            getattr(
                getattr(initial_kernel, "k2", None),
                "length_scale",
                np.nan,
            )
        )
        const_lower = float(reference_row["gp_const_lo"])
        const_upper = float(reference_row["gp_const_hi"])
        geometry = training_geometry(
            x_full,
            y_toy,
            train_mask,
            reference_pred.edges_full,
            mass=float(mass),
        )
        return {
            **base,
            "fit_ok": bool(fit.get("success", False)),
            "gp_lml": float(gpr.log_marginal_likelihood_value_),
            "gp_ls": float(diagnostics["ls_opt"]),
            "gp_const": float(diagnostics["const_opt"]),
            "gp_const_lo": const_lower,
            "gp_const_hi": const_upper,
            "gp_ls_lo": float(reference_row["gp_ls_lo"]),
            "gp_ls_hi": float(reference_row["gp_ls_hi"]),
            "gp_ls_init": initial_ls,
            "gp_const_init": initial_const,
            "sigma_A": sigma_a,
            "A_hat": a_hat,
            "Zhat": a_hat / sigma_a,
            "pull": (a_hat - injected) / sigma_a,
            "amplitude_nll": float(fit.get("nll", np.nan)),
            "n_blind": int(np.count_nonzero(blind_mask)),
            "Nsig_full": int(n_signal_full),
            "Nsig_win": int(np.sum(signal_full[blind_mask])),
            "Nsig_train": int(np.sum(signal_full[train_mask])),
            "signal_counts_sha256": array_hash(signal_full, "<i8"),
            "optimizer_warning_count": len(
                getattr(gpr, "_hps_optimizer_warnings", ())
            ),
            "optimizer_warnings": " | ".join(
                getattr(gpr, "_hps_optimizer_warnings", ())
            ),
            **geometry,
            **kernel_bound_diagnostics(
                ls_value=float(diagnostics["ls_opt"]),
                ls_lower=float(reference_row["gp_ls_lo"]),
                ls_upper=float(reference_row["gp_ls_hi"]),
                const_value=float(diagnostics["const_opt"]),
                const_lower=const_lower,
                const_upper=const_upper,
            ),
            **covariance_diagnostics(covariance),
        }
    except Exception as exc:
        return {
            **base,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def refit_triggers(
    row: Mapping[str, Any], gate: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if not bool(row.get("fit_ok")) or not bool(row.get("covariance_valid")):
        reasons.append("invalid_or_nonfinite")
        return reasons
    ls_value, const_value = float(row["gp_ls"]), float(row["gp_const"])
    ls_initial, const_initial = float(row["gp_ls_init"]), float(
        row["gp_const_init"]
    )
    exact_tolerance = float(gate["exact_start_abs_log_theta_max"])
    if all(
        value > 0
        for value in (ls_value, const_value, ls_initial, const_initial)
    ):
        if max(
            abs(math.log(ls_value / ls_initial)),
            abs(math.log(const_value / const_initial)),
        ) < exact_tolerance:
            reasons.append("exact_start_signature")
    lower, upper = float(row["gp_ls_lo"]), float(row["gp_ls_hi"])
    bound_window = float(gate["bound_ratio_window"])
    if ls_value > 0 and lower > 0 and ls_value / lower <= 1.0 + bound_window:
        reasons.append("near_lower_length_bound")
    if ls_value > 0 and upper > 0 and ls_value / upper >= 1.0 - bound_window:
        reasons.append("near_upper_length_bound")
    ratio = float(row["sigma_A"]) / float(row["sigmaA_reference"])
    ratio_low, ratio_high = map(float, gate["sigma_over_reference_trigger"])
    if not np.isfinite(ratio) or ratio < ratio_low or ratio > ratio_high:
        reasons.append("sigma_reference_ratio")

    # v4.7.1: apply the amended reference-relative thresholds frozen after the
    # first full-ledger numerical audit and before the uniform rerun.  This is
    # a pull-blind repeat trigger only; the final branch is still selected by
    # LML and reproducibility.  No fitted amplitude, pull, recovery, or
    # epsilon-squared coordinate enters this decision.
    reference_values = {
        "lml": float(row.get("reference_gp_lml", np.nan)),
        "ls": float(row.get("reference_gp_ls", np.nan)),
        "const": float(row.get("reference_gp_const", np.nan)),
    }
    comparable = (
        np.isfinite(float(row["gp_lml"]))
        and np.isfinite(reference_values["lml"])
        and int(row["n_train"]) > 0
        and all(
            np.isfinite(value) and value > 0
            for value in (
                float(row["gp_ls"]),
                reference_values["ls"],
                float(row["gp_const"]),
                reference_values["const"],
            )
        )
    )
    if not comparable:
        reasons.append("reference_relative_nonfinite")
        return reasons
    if (
        abs(float(row["gp_lml"]) - reference_values["lml"])
        / float(row["n_train"])
        > float(gate["reference_relative_lml_per_train_trigger"])
    ):
        reasons.append("reference_relative_lml")
    if abs(math.log(float(row["gp_ls"]) / reference_values["ls"])) > float(
        gate["reference_relative_abs_log_length_trigger"]
    ):
        reasons.append("reference_relative_length")
    if abs(
        math.log(float(row["gp_const"]) / reference_values["const"])
    ) > float(gate["reference_relative_abs_log_constant_trigger"]):
        reasons.append("reference_relative_constant")
    return reasons


def accepted_row(
    selected: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    model: str,
    scenario: str,
    toy_index: int,
    mass: float,
    z_value: float,
    attempts: int,
    replicates: int,
    gate_status: str,
    trigger_reasons: Iterable[str],
    protocol: Mapping[str, Any],
    selected_upper_factor: float,
) -> dict[str, Any]:
    row = dict(selected)
    source_family, multiplier, _, _, _ = SCENARIO_POLICY[scenario]
    frozen = protocol["frozen_analysis_contract"]
    row.update(
        {
            "study_id": protocol["study_id"],
            "model": model,
            "scenario": scenario,
            "scenario_label": scenario,
            "source_family": source_family,
            "source_multiplier": multiplier,
            "truth_model": model,
            "truth_function_tag": model,
            "background_toy_index": int(toy_index),
            "mass_GeV": float(mass),
            "mass_MeV": 1000.0 * float(mass),
            "inj_nsigma": float(z_value),
            "n_attempts": int(attempts),
            "top_branch_replicates": int(replicates),
            "optimizer_gate_version": OPTIMIZER_GATE["version"],
            "optimizer_gate_status": gate_status,
            "optimizer_trigger_reasons": ";".join(trigger_reasons),
            "optimizer_selection_pull_blind": True,
            "accepted": True,
            "sigmaA_ref": float(reference_row["sigma_A"]),
            "sigmaA_ref_mode": "matched_refit_bonly_multistart_v1",
            "reference_top_branch_replicates": int(
                reference_row["top_branch_replicates"]
            ),
            "eps2_hat_signed": float(selected["A_hat"])
            / float(reference_row["A_per_eps2_unit"]),
            "eps2_injected": float(selected.get("strength", 0.0))
            / float(reference_row["A_per_eps2_unit"]),
            "eps2_sigma": float(selected["sigma_A"])
            / float(reference_row["A_per_eps2_unit"]),
            "nominal_Z_residual": float(selected["Zhat"]) - float(z_value),
            "pull_identity_residual": float(selected["pull"])
            - (
                (float(selected["A_hat"]) - float(selected.get("strength", 0.0)))
                / float(selected["sigma_A"])
            ),
            "refit_ls_over_hi": float(selected["gp_ls"])
            / float(selected["gp_ls_hi"]),
            "refit_ls_over_lo": float(selected["gp_ls"])
            / float(selected["gp_ls_lo"]),
            "refit_upper_boundary": float(selected["gp_ls"])
            / float(selected["gp_ls_hi"])
            >= 0.999,
            "refit_lower_boundary": float(selected["gp_ls"])
            / float(selected["gp_ls_lo"])
            <= 1.001,
            "refit_constant_lower_boundary": bool(
                selected.get("const_at_lower", False)
            ),
            "refit_constant_upper_boundary": bool(
                selected.get("const_at_upper", False)
            ),
            "selected_extraction_upper_factor": float(selected_upper_factor),
            "production_card_upper_factor": 15.0,
            "analysis_partition": f"residual_{model}_20toy_validation",
            "declared_result_commit": frozen["result_commit"],
            "declared_integration_commit": frozen["integration_commit"],
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    return row


def task_directory(model: str, scenario: str, toy_index: int) -> Path:
    return model_runs(model) / scenario / f"toy_{int(toy_index):04d}"


def successful_task(model: str, scenario: str, toy_index: int) -> Path | None:
    directory = task_directory(model, scenario, toy_index)
    marker_path = directory / "_SUCCESS.json"
    if not marker_path.is_file():
        return None
    try:
        payload = load_json(marker_path)
    except Exception:
        return None
    required_hashes = {
        "protocol_sha256": PROTOCOL_PATH,
        "source_fit_sha256": SOURCE_PRODUCT_PATH,
        "background_toy_root_sha256": TOY_ROOT_PATH,
        "background_toy_manifest_sha256": TOY_MANIFEST_PATH,
        "pilot_disposition_sha256": PILOT_DISPOSITION_PATH,
        "runtime_manifest_sha256": RUNTIME_MANIFEST_PATH,
        "runner_sha256": Path(__file__).resolve(),
    }
    if (
        payload.get("status") != "pass"
        or payload.get("model") != model
        or payload.get("scenario") != scenario
        or int(payload.get("toy_index", -1)) != int(toy_index)
    ):
        return None
    for key, path in required_hashes.items():
        if not path.is_file() or payload.get(key) != sha256_file(path):
            return None
    _, selected_upper_factor = load_pilot_disposition()
    if float(payload.get("selected_extraction_upper_factor", float("nan"))) != float(
        selected_upper_factor
    ):
        return None
    declared = payload.get("ledger_sha256")
    if not isinstance(declared, Mapping) or set(declared) != set(LEDGER_FILES):
        return None
    for name in LEDGER_FILES:
        ledger = directory / name
        if not ledger.is_file() or sha256_file(ledger) != str(declared[name]):
            return None
    return directory / "accepted_rows.csv"


def _empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "scenario",
            "background_toy_index",
            "mass_GeV",
            "inj_nsigma",
            "exclusion_scope",
            "reason",
            "n_attempts",
            "trigger_reasons",
            "selection_pull_blind",
        ]
    )


def run_task(
    model: str, scenario: str, toy_index: int, *, force: bool = False
) -> dict[str, Any]:
    configure_process()
    if model not in MODELS:
        raise StudyError(f"unsupported model: {model}")
    protocol = load_protocol()
    assert_protocol_contract(protocol)
    mass_grid, strength_grid = closure_grid(protocol)
    gate = OPTIMIZER_GATE
    if scenario not in SCENARIOS or not 0 <= int(toy_index) < N_TOYS:
        raise StudyError("invalid reported scenario or closure toy index")
    preflight(model, validate_inventory=False)
    existing = successful_task(model, scenario, int(toy_index))
    if existing is not None and not force:
        return {
            "status": "already_complete",
            "model": model,
            "scenario": scenario,
            "toy_index": int(toy_index),
        }

    final_directory = task_directory(model, scenario, int(toy_index))
    if final_directory.exists():
        if not force:
            raise StudyError(
                f"incomplete task exists; inspect or use --force: {final_directory}"
            )
        archived = final_directory.with_name(
            final_directory.name
            + ".superseded_"
            + datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        if archived.exists():
            raise StudyError(f"superseded task destination already exists: {archived}")
        os.replace(final_directory, archived)

    final_directory.parent.mkdir(parents=True, exist_ok=True)
    work_directory = Path(
        tempfile.mkdtemp(prefix=f".{final_directory.name}.", dir=final_directory.parent)
    )
    _, selected_upper_factor = load_pilot_disposition()
    cfg = build_config(selected_upper_factor)
    assert_config(cfg, selected_upper_factor)
    dataset = make_toy_dataset(model, scenario, int(toy_index), cfg)
    attempt_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    try:
        for mass in mass_grid:
            reference_records: list[dict[str, Any]] = []
            reference_predictions: dict[int, Any] = {}
            for attempt in range(int(gate["reference_initial_attempts"])):
                record, prediction = reference_attempt(
                    dataset,
                    cfg,
                    model,
                    scenario,
                    int(toy_index),
                    float(mass),
                    attempt,
                )
                reference_records.append(record)
                if prediction is not None:
                    reference_predictions[int(attempt)] = prediction
            selected_reference, reference_replicates = select_branch(
                reference_records, gate, require_replication=True
            )
            if selected_reference is None:
                for attempt in range(
                    int(gate["reference_initial_attempts"]),
                    int(gate["maximum_attempts"]),
                ):
                    record, prediction = reference_attempt(
                        dataset,
                        cfg,
                        model,
                        scenario,
                        int(toy_index),
                        float(mass),
                        attempt,
                    )
                    reference_records.append(record)
                    if prediction is not None:
                        reference_predictions[int(attempt)] = prediction
                selected_reference, reference_replicates = select_branch(
                    reference_records, gate, require_replication=True
                )
            attempt_rows.extend(reference_records)

            if selected_reference is None:
                for z_value in strength_grid:
                    placeholder = (
                        dict(reference_records[0]) if reference_records else {}
                    )
                    placeholder.update(
                        {
                            "model": model,
                            "scenario": scenario,
                            "background_toy_index": int(toy_index),
                            "mass_GeV": float(mass),
                            "inj_nsigma": float(z_value),
                            "strength": float("nan"),
                            "A_hat": float("nan"),
                            "sigma_A": float("nan"),
                            "Zhat": float("nan"),
                            "pull": float("nan"),
                            "role": "raw_reference_invalid_placeholder",
                            "accepted": False,
                            "optimizer_gate_status": (
                                "exclude_irreproducible_reference"
                            ),
                            "optimizer_selection_pull_blind": True,
                            "selected_extraction_upper_factor": float(
                                selected_upper_factor
                            ),
                        }
                    )
                    raw_rows.append(placeholder)
                    exclusions.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "background_toy_index": int(toy_index),
                            "mass_GeV": float(mass),
                            "inj_nsigma": float(z_value),
                            "exclusion_scope": "scenario_toy_mass_all_strengths",
                            "reason": (
                                "irreproducible_background_reference_top_branch"
                            ),
                            "n_attempts": len(reference_records),
                            "trigger_reasons": "",
                            "selection_pull_blind": True,
                        }
                    )
                continue

            raw_reference = dict(reference_records[0])
            raw_reference.update(
                {
                    "optimizer_selection_pull_blind": True,
                    "selected_extraction_upper_factor": float(
                        selected_upper_factor
                    ),
                }
            )
            raw_rows.append(raw_reference)
            selected_reference = dict(selected_reference)
            selected_reference["top_branch_replicates"] = int(reference_replicates)
            reference_prediction = reference_predictions[
                int(selected_reference["attempt"])
            ]
            reference_status = (
                "pass_replicated_initial3"
                if len(reference_records) == 3
                else "pass_replicated_after5"
            )
            accepted_rows.append(
                accepted_row(
                    selected_reference,
                    selected_reference,
                    model,
                    scenario,
                    int(toy_index),
                    float(mass),
                    0.0,
                    len(reference_records),
                    reference_replicates,
                    reference_status,
                    (),
                    protocol,
                    selected_upper_factor,
                )
            )

            for z_value in (1.0, 3.0, 5.0):
                records = [
                    refit_attempt(
                        dataset,
                        cfg,
                        reference_prediction,
                        selected_reference,
                        model,
                        scenario,
                        int(toy_index),
                        float(mass),
                        z_value,
                        0,
                    )
                ]
                trigger_reasons = refit_triggers(records[0], gate)
                if trigger_reasons:
                    records.extend(
                        refit_attempt(
                            dataset,
                            cfg,
                            reference_prediction,
                            selected_reference,
                            model,
                            scenario,
                            int(toy_index),
                            float(mass),
                            z_value,
                            attempt,
                        )
                        for attempt in (1, 2)
                    )
                    selected, replicates = select_branch(
                        records, gate, require_replication=True
                    )
                    if selected is None:
                        records.extend(
                            refit_attempt(
                                dataset,
                                cfg,
                                reference_prediction,
                                selected_reference,
                                model,
                                scenario,
                                int(toy_index),
                                float(mass),
                                z_value,
                                attempt,
                            )
                            for attempt in (3, 4)
                        )
                        selected, replicates = select_branch(
                            records, gate, require_replication=True
                        )
                    gate_status = (
                        "pass_trigger_replicated_after3"
                        if len(records) == 3
                        else "pass_trigger_replicated_after5"
                    )
                else:
                    selected, replicates = select_branch(
                        records, gate, require_replication=False
                    )
                    gate_status = "pass_single_untriggered"
                attempt_rows.extend(records)
                raw_primary = dict(records[0])
                raw_primary.update(
                    {
                        "optimizer_trigger_reasons": ";".join(trigger_reasons),
                        "optimizer_selection_pull_blind": True,
                        "selected_extraction_upper_factor": float(
                            selected_upper_factor
                        ),
                    }
                )
                raw_rows.append(raw_primary)
                if selected is None:
                    exclusions.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "background_toy_index": int(toy_index),
                            "mass_GeV": float(mass),
                            "inj_nsigma": float(z_value),
                            "exclusion_scope": "single_injected_fit_row",
                            "reason": "irreproducible_injected_refit_top_branch",
                            "n_attempts": len(records),
                            "trigger_reasons": ";".join(trigger_reasons),
                            "selection_pull_blind": True,
                        }
                    )
                    continue
                accepted_rows.append(
                    accepted_row(
                        selected,
                        selected_reference,
                        model,
                        scenario,
                        int(toy_index),
                        float(mass),
                        z_value,
                        len(records),
                        replicates,
                        gate_status,
                        trigger_reasons,
                        protocol,
                        selected_upper_factor,
                    )
                )

        attempts_frame = pd.DataFrame(attempt_rows)
        accepted_frame = pd.DataFrame(accepted_rows)
        raw_frame = pd.DataFrame(raw_rows)
        exclusions_frame = (
            pd.DataFrame(exclusions) if exclusions else _empty_exclusions()
        )
        expected_rows = len(mass_grid) * len(strength_grid)
        raw_key = ["model", "scenario", "background_toy_index", "mass_GeV", "inj_nsigma"]
        if len(raw_frame) != expected_rows or raw_frame.duplicated(raw_key).any():
            raise StudyError(
                f"raw-primary ledger must have {expected_rows} unique rows"
            )
        frames = {
            "optimizer_attempts.csv": attempts_frame,
            "accepted_rows.csv": accepted_frame,
            "raw_primary_rows.csv": raw_frame,
            "exclusions.csv": exclusions_frame,
        }
        for name, frame in frames.items():
            frame.to_csv(work_directory / name, index=False)
        ledger_hashes = {
            name: sha256_file(work_directory / name) for name in LEDGER_FILES
        }
        marker = {
            "status": "pass",
            "completed_utc": utc_now(),
            "model": model,
            "scenario": scenario,
            "toy_index": int(toy_index),
            "attempt_rows": len(attempts_frame),
            "accepted_rows": len(accepted_frame),
            "raw_primary_rows": len(raw_frame),
            "excluded_rows": len(exclusions_frame),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "source_fit_sha256": sha256_file(SOURCE_PRODUCT_PATH),
            "background_toy_root_sha256": sha256_file(TOY_ROOT_PATH),
            "background_toy_manifest_sha256": sha256_file(TOY_MANIFEST_PATH),
            "pilot_disposition_sha256": sha256_file(PILOT_DISPOSITION_PATH),
            "runtime_manifest_sha256": sha256_file(RUNTIME_MANIFEST_PATH),
            "runtime_overlay_file_sha256": load_json(RUNTIME_MANIFEST_PATH)[
                "files"
            ],
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "selected_extraction_upper_factor": float(selected_upper_factor),
            "optimizer_gate_version": gate["version"],
            "optimizer_selection_pull_blind": True,
            "ledger_sha256": ledger_hashes,
        }
        atomic_json(work_directory / "_SUCCESS.json", marker)
        os.replace(work_directory, final_directory)
        return marker
    except Exception:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise


def run_task_subprocess(
    model: str, scenario: str, toy_index: int, force: bool
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        model,
        "run-task",
        scenario,
        str(int(toy_index)),
    ]
    if force:
        command.append("--force")
    environment = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[key] = "1"
    result = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise StudyError(
            f"task {model}/{scenario} toy {toy_index} failed:\n{result.stdout}"
        )
    return {
        "model": model,
        "scenario": scenario,
        "toy_index": int(toy_index),
        "output": result.stdout,
    }


def run_many(
    model: str,
    toy_start: int,
    toy_stop: int,
    workers: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    preflight(model, validate_inventory=True)
    if not 0 <= int(toy_start) < int(toy_stop) <= N_TOYS:
        raise StudyError("toy interval must satisfy 0 <= start < stop <= 20")
    if not 1 <= int(workers) <= 2:
        raise StudyError("CPU-conscious production permits one or two workers")
    tasks = [
        (scenario, toy_index)
        for scenario in SCENARIOS
        for toy_index in range(int(toy_start), int(toy_stop))
    ]
    completed = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(run_task_subprocess, model, scenario, toy_index, force): (
                scenario,
                toy_index,
            )
            for scenario, toy_index in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"PASS {model}/{result['scenario']} toy {result['toy_index']:04d}",
                flush=True,
            )
    return {
        "status": "pass",
        "model": model,
        "tasks": len(tasks),
        "completed": len(completed),
        "toy_start": int(toy_start),
        "toy_stop": int(toy_stop),
        "workers": int(workers),
    }


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _moments(group: pd.DataFrame, prefix: str) -> dict[str, Any]:
    from scipy.stats import chi2, median_abs_deviation, t, trim_mean

    pull_series = (
        group["pull"] if "pull" in group.columns else pd.Series(dtype=float)
    )
    values = pd.to_numeric(pull_series, errors="coerce").dropna().to_numpy(
        dtype=float
    )
    count = len(values)
    mean = float(np.mean(values)) if count else float("nan")
    width = float(np.std(values, ddof=1)) if count > 1 else float("nan")
    t_critical = float(t.ppf(0.95, count - 1)) if count > 1 else float("nan")
    chi_low = float(chi2.ppf(0.05, count - 1)) if count > 1 else float("nan")
    chi_high = float(chi2.ppf(0.95, count - 1)) if count > 1 else float("nan")
    leave_one_out = (
        [abs(float(np.mean(np.delete(values, index))) - mean) for index in range(count)]
        if count > 1
        else []
    )
    return {
        f"{prefix}_n": count,
        f"{prefix}_pull_mean": mean,
        f"{prefix}_pull_width": width,
        f"{prefix}_pull_median": float(np.median(values))
        if count
        else float("nan"),
        f"{prefix}_pull_mad_scaled": float(
            median_abs_deviation(values, scale="normal")
        )
        if count
        else float("nan"),
        f"{prefix}_pull_trimmed_mean_10pct": float(trim_mean(values, 0.1))
        if count
        else float("nan"),
        f"{prefix}_pull_mean_ci90_low": mean
        - t_critical * width / math.sqrt(count)
        if count > 1
        else float("nan"),
        f"{prefix}_pull_mean_ci90_high": mean
        + t_critical * width / math.sqrt(count)
        if count > 1
        else float("nan"),
        f"{prefix}_pull_width_ci90_low": math.sqrt(
            (count - 1) * width * width / chi_high
        )
        if count > 1
        else float("nan"),
        f"{prefix}_pull_width_ci90_high": math.sqrt(
            (count - 1) * width * width / chi_low
        )
        if count > 1
        else float("nan"),
        f"{prefix}_max_leave_one_out_mean_change": max(leave_one_out)
        if leave_one_out
        else float("nan"),
    }


def collect(model: str) -> dict[str, Any]:
    from scipy.stats import ttest_1samp

    protocol = load_protocol()
    assert_protocol_contract(protocol)
    mass_grid, strength_grid = closure_grid(protocol)
    accepted_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    attempt_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    task_audit: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for toy_index in range(N_TOYS):
            if successful_task(model, scenario, toy_index) is None:
                raise StudyError(
                    f"missing or hash-invalid task {model}/{scenario} "
                    f"toy {toy_index:04d}"
                )
            directory = task_directory(model, scenario, toy_index)
            accepted = _read_optional_csv(directory / "accepted_rows.csv")
            raw = _read_optional_csv(directory / "raw_primary_rows.csv")
            attempts = _read_optional_csv(directory / "optimizer_attempts.csv")
            exclusions = _read_optional_csv(directory / "exclusions.csv")
            if raw is None or attempts is None:
                raise StudyError(
                    f"required task ledger is empty: {scenario} toy {toy_index:04d}"
                )
            if accepted is not None:
                accepted_frames.append(accepted)
            raw_frames.append(raw)
            attempt_frames.append(attempts)
            if exclusions is not None:
                exclusion_frames.append(exclusions)
            marker = load_json(directory / "_SUCCESS.json")
            task_audit.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "toy_index": toy_index,
                    "status": marker["status"],
                    "accepted_rows": marker["accepted_rows"],
                    "raw_primary_rows": marker["raw_primary_rows"],
                    "excluded_rows": marker["excluded_rows"],
                    "success_marker_sha256": sha256_file(
                        directory / "_SUCCESS.json"
                    ),
                    **{
                        f"{Path(name).stem}_sha256": marker["ledger_sha256"][name]
                        for name in LEDGER_FILES
                    },
                }
            )

    accepted = (
        pd.concat(accepted_frames, ignore_index=True, sort=False)
        if accepted_frames
        else pd.DataFrame()
    )
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    attempts = pd.concat(attempt_frames, ignore_index=True, sort=False)
    exclusions = (
        pd.concat(exclusion_frames, ignore_index=True, sort=False)
        if exclusion_frames
        else _empty_exclusions()
    )
    for label, frame in (
        ("raw", raw),
        ("attempt", attempts),
        ("accepted", accepted),
        ("exclusion", exclusions),
    ):
        if frame.empty:
            continue
        if "model" not in frame.columns or set(frame["model"].astype(str)) != {model}:
            raise StudyError(f"{label} ledger crosses model output boundaries")
    key_columns = [
        "model",
        "scenario",
        "background_toy_index",
        "mass_GeV",
        "inj_nsigma",
    ]
    raw = raw.sort_values(key_columns).reset_index(drop=True)
    if not accepted.empty:
        accepted = accepted.sort_values(key_columns).reset_index(drop=True)
    expected_raw = EXPECTED_RAW_ROWS_PER_MODEL
    if len(raw) != expected_raw or raw.duplicated(key_columns).any():
        raise StudyError(
            f"raw ledger must contain {expected_raw} unique states"
        )
    if not accepted.empty and accepted.duplicated(key_columns).any():
        raise StudyError("accepted ledger contains duplicate states")

    summaries: list[dict[str, Any]] = []
    minimum_required = int(
        OPTIMIZER_GATE["minimum_accepted_per_cell_for_closure_claim"]
    )
    for scenario in SCENARIOS:
        for mass in mass_grid:
            for z_value in strength_grid:
                raw_group = raw[
                    (raw.scenario == scenario)
                    & np.isclose(raw.mass_GeV, mass)
                    & np.isclose(raw.inj_nsigma, z_value)
                ]
                accepted_group = (
                    accepted[
                        (accepted.scenario == scenario)
                        & np.isclose(accepted.mass_GeV, mass)
                        & np.isclose(accepted.inj_nsigma, z_value)
                    ]
                    if not accepted.empty
                    else pd.DataFrame()
                )
                record = {
                    "model": model,
                    "scenario": scenario,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "inj_nsigma": float(z_value),
                    "n_generated": N_TOYS,
                    **_moments(raw_group, "raw"),
                    **_moments(accepted_group, "accepted"),
                    "n_excluded": N_TOYS - len(accepted_group),
                    "sample_size_eligible": len(accepted_group)
                    >= minimum_required,
                    "accepted_nominal_Z_residual_mean": float(
                        pd.to_numeric(
                            accepted_group["nominal_Z_residual"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_upper_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_upper_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_lower_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_lower_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_constant_lower_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_constant_lower_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_constant_upper_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_constant_upper_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_pull_identity_max_abs_residual": float(
                        pd.to_numeric(
                            accepted_group["pull_identity_residual"]
                        ).abs().max()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_ls_at_lower_fraction": float(
                        pd.to_numeric(accepted_group["ls_at_lower"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_ls_at_upper_fraction": float(
                        pd.to_numeric(accepted_group["ls_at_upper"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_const_at_lower_fraction": float(
                        pd.to_numeric(accepted_group["const_at_lower"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_const_at_upper_fraction": float(
                        pd.to_numeric(accepted_group["const_at_upper"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_min_y_train": float(
                        pd.to_numeric(accepted_group["min_y_train"]).min()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_max_y_train": float(
                        pd.to_numeric(accepted_group["max_y_train"]).max()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_max_n_zero_train": int(
                        pd.to_numeric(accepted_group["n_zero_train"]).max()
                    )
                    if len(accepted_group)
                    else -1,
                }
                if float(z_value) > 0 and len(accepted_group):
                    recovery = (
                        pd.to_numeric(accepted_group["A_hat"])
                        / pd.to_numeric(accepted_group["strength"])
                    )
                    record.update(
                        {
                            "accepted_median_recovery": float(
                                np.median(recovery)
                            ),
                            "accepted_recovery_q16": float(
                                np.quantile(recovery, 0.16)
                            ),
                            "accepted_recovery_q84": float(
                                np.quantile(recovery, 0.84)
                            ),
                        }
                    )
                else:
                    record.update(
                        {
                            "accepted_median_recovery": float("nan"),
                            "accepted_recovery_q16": float("nan"),
                            "accepted_recovery_q84": float("nan"),
                        }
                    )
                summaries.append(record)

    summary = pd.DataFrame(summaries).sort_values(
        ["scenario", "mass_GeV", "inj_nsigma"]
    )
    if len(summary) != len(SCENARIOS) * len(mass_grid) * len(strength_grid):
        raise StudyError("closure summary cell cardinality drift")
    zero_records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for mass in mass_grid:
            group = (
                accepted[
                    (accepted.scenario == scenario)
                    & np.isclose(accepted.mass_GeV, mass)
                    & np.isclose(accepted.inj_nsigma, 0.0)
                ]
                if not accepted.empty
                else pd.DataFrame()
            )
            values = (
                pd.to_numeric(group["pull"], errors="coerce")
                .dropna()
                .to_numpy(float)
                if len(group)
                else np.array([], dtype=float)
            )
            p_value = (
                float(ttest_1samp(values, 0.0).pvalue)
                if len(values) > 1
                else float("nan")
            )
            zero_records.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "n": len(values),
                    "mean_pull": float(np.mean(values))
                    if len(values)
                    else float("nan"),
                    "width": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else float("nan"),
                    "exploratory_ttest_p": p_value,
                }
            )
    zero = pd.DataFrame(zero_records)
    if len(zero) != len(SCENARIOS) * len(mass_grid):
        raise StudyError("zero-signal diagnostic cell cardinality drift")
    finite_positions = np.where(
        np.isfinite(zero["exploratory_ttest_p"].to_numpy(float))
    )[0]
    adjusted = np.full(len(zero), np.nan, dtype=float)
    if len(finite_positions):
        p_values = zero.loc[
            finite_positions, "exploratory_ttest_p"
        ].to_numpy(float)
        order = np.argsort(p_values)
        running = 0.0
        for rank, ordered_position in enumerate(order):
            candidate = (len(p_values) - rank) * float(
                p_values[ordered_position]
            )
            running = max(running, candidate)
            adjusted[finite_positions[ordered_position]] = min(1.0, running)
    zero["exploratory_holm_p"] = adjusted
    zero["exploratory_material_bias_flag"] = (
        (zero.exploratory_holm_p < 0.05) & (zero.mean_pull.abs() >= 0.2)
    )

    derived = model_derived(model)
    derived.mkdir(parents=True, exist_ok=True)
    products = {
        "accepted_extraction_rows.csv": accepted,
        "raw_primary_extraction_rows.csv": raw,
        "optimizer_attempt_ledger.csv": attempts,
        "exclusion_ledger.csv": exclusions,
        "closure_summary.csv": summary,
        "zero_signal_bias_tests.csv": zero,
        "task_product_audit.csv": pd.DataFrame(task_audit),
    }
    for name, frame in products.items():
        atomic_csv(derived / name, frame)
    product_hashes = {
        name: sha256_file(derived / name) for name in products
    }
    _, selected_upper_factor = load_pilot_disposition()
    result = {
        "status": "pass",
        "collected_utc": utc_now(),
        "model": model,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_fit_sha256": sha256_file(SOURCE_PRODUCT_PATH),
        "background_toy_root_sha256": sha256_file(TOY_ROOT_PATH),
        "background_toy_manifest_sha256": sha256_file(TOY_MANIFEST_PATH),
        "pilot_disposition_sha256": sha256_file(PILOT_DISPOSITION_PATH),
        "runtime_manifest_sha256": sha256_file(RUNTIME_MANIFEST_PATH),
        "runtime_overlay_file_sha256": load_json(RUNTIME_MANIFEST_PATH)["files"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "selected_extraction_upper_factor": selected_upper_factor,
        "production_card_upper_factor": 15.0,
        "raw_rows": len(raw),
        "accepted_rows": len(accepted),
        "excluded_rows": len(exclusions),
        "optimizer_attempt_rows": len(attempts),
        "summary_cells": len(summary),
        "minimum_accepted_per_cell": int(summary.accepted_n.min()),
        "all_cells_sample_size_eligible": bool(
            summary.sample_size_eligible.all()
        ),
        "scientific_diagnostics": {
            "bias_endpoint": "cellwise accepted mean pull with 90% Student-t interval",
            "width_endpoint": "cellwise accepted sample pull width with 90% chi-square interval",
            "sample_size_gate_is_not_closure": True,
            "maximum_abs_pull_identity_residual": float(
                summary.accepted_pull_identity_max_abs_residual.max()
            ),
        },
        "interpretation": (
            "Twenty-background residual-structured conditional "
            "injection-extraction validation; not coverage, expected limits, "
            "exclusion, observed-data bias, or scan-wise calibration."
        ),
        "claim_boundary": protocol["claim_boundary"],
        "derived_sha256": product_hashes,
    }
    atomic_json(derived / "collection_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("scenario", choices=SCENARIOS)
    task_parser.add_argument("toy_index", type=int)
    task_parser.add_argument("--force", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--toy-start", type=int, required=True)
    run_parser.add_argument("--toy-stop", type=int, required=True)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("collect")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        result = preflight(args.model, validate_inventory=True)
    elif args.command == "run-task":
        result = run_task(
            args.model,
            args.scenario,
            int(args.toy_index),
            force=bool(args.force),
        )
    elif args.command == "run":
        result = run_many(
            args.model,
            int(args.toy_start),
            int(args.toy_stop),
            int(args.workers),
            force=bool(args.force),
        )
    elif args.command == "collect":
        result = collect(args.model)
    else:
        raise StudyError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
