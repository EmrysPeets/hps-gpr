#!/usr/bin/env python3
"""Run the pull-blind v4.8p3 residual-truth length-ceiling pilot.

This study-local driver consumes only the independently generated *pilot*
histograms from ``residual_structured_nested_toys.root``.  Its task unit is one
``(model, scenario, pilot toy)`` and contains all three masses and all three
length upper factors.  Keeping factors inside one task makes the common-seed
comparison structural and gives 30 resumable tasks.

The driver is deliberately incapable of computing signal amplitudes, pulls,
recovery, p0, CLs, epsilon-squared, limits, or coverage.  Only ``run-task`` and
``run`` launch GP fits.  ``preflight``, ``validate``, ``prepare``, ``status``,
and ``collect`` do not.  The common-ceiling disposition uses only the
predeclared background-only gates in ``MODEL_PROTOCOL.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# Pin numerical libraries before importing numpy/scipy/sklearn through the
# authoritative v4.8 runtime.  This pilot is explicitly single-worker.
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
import uproot


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DRIVER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = HERE / "MODEL_PROTOCOL.json"
FIT_RESULT_PATH = HERE / "derived/source_fit_and_influence.json"
TOY_ROOT_DEFAULT = HERE / "inputs/residual_structured_nested_toys.root"
TOY_MANIFEST_DEFAULT = HERE / "inputs/residual_structured_nested_toys.manifest.json"
RUNS = HERE / "runs/residual_length_pilot"
DERIVED = HERE / "derived/residual_length_pilot"
QA = HERE / "qa/residual_length_pilot"

V4P8 = REPO / "study_results/v4p8_2021_functional_form_qualification_20260813"
V4P8_CORE_PATH = V4P8 / "rigid_length_scan_core.py"
V4P8_SPEC_PATH = V4P8 / "rigid_study_spec.json"
V4P8_RUNTIME_ROOT = V4P8 / "runtime_overlay"
V4P8_RUNTIME_MANIFEST = V4P8 / "runtime_overlay_manifest.json"
V4P8_CARD_PATH = V4P8 / "inputs/frozen_v4p2_analysis_card.yaml"

EXPECTED_V4P8_CORE_SHA256 = (
    "97c0f41a220da7ec6cdf5666c2ab78db8fa429f1f99ddb0ea20e09f261c072f8"
)
EXPECTED_V4P8_SPEC_SHA256 = (
    "5b65d2b5c98b7afad560934ab07bdd0c0921667d51fcb5ff231916c21c1bfd1a"
)
EXPECTED_V4P8_RUNTIME_MANIFEST_SHA256 = (
    "667390be8c2c5b79578c4ca933ff94fad289146432859f62ebf851a128a6c2e6"
)
EXPECTED_V4P2_CARD_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)

MODELS = ("knot_spline", "regional_blend")
SCENARIOS = (
    "2021_1pct",
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
PILOT_TOY_INDICES = (0, 1, 2)
MASS_MEV = (65, 120, 210)
MASS_GRID = tuple(value / 1000.0 for value in MASS_MEV)
UPPER_FACTORS = (15, 20, 25)
SUPPORT_GEV = (0.04, 0.30)
SEARCH_GEV = (0.05, 0.25)
BASE_SEED = 20260814
SEED_NAMESPACE = "v4p8p3_residual_length_pilot_common_v1"
OPTIMIZER_RESTARTS = 12
EXACT_BOUND_RATIO = 0.999
EXPECTED_TASKS = len(MODELS) * len(SCENARIOS) * len(PILOT_TOY_INDICES)
EXPECTED_STATES = EXPECTED_TASKS * len(MASS_GRID) * len(UPPER_FACTORS)
EXPECTED_COMPARISONS = EXPECTED_TASKS * len(MASS_GRID)

TASK_PRODUCT_NAMES = (
    "optimizer_attempts.csv",
    "selected_trajectories.csv",
    "optimizer_exclusions.csv",
)
EXCLUSION_COLUMNS = (
    "study_id",
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

_CORE: Any | None = None


class PilotError(RuntimeError):
    """Raised when the frozen pilot contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise PilotError(f"JSON root must be an object: {path}")
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


def stable_seed(*parts: object) -> int:
    material = "|".join(
        [str(BASE_SEED), SEED_NAMESPACE, *[str(part) for part in parts]]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")


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
        raise PilotError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise PilotError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}: {path}"
        )


def resolve_study_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else HERE / path


def protocol() -> dict[str, Any]:
    payload = load_json(PROTOCOL_PATH)
    if int(payload.get("schema_version", -1)) != 1:
        raise PilotError("unsupported MODEL_PROTOCOL schema")
    pilot = payload.get("length_ceiling_pilot", {})
    expected = {
        "background_only": True,
        "toy_indices": list(PILOT_TOY_INDICES),
        "masses_gev": list(MASS_GRID),
        "upper_factors": list(UPPER_FACTORS),
        "workers": 1,
        "common_ceiling_for_models_and_lanes": True,
        "fallback_factor": 25,
    }
    for key, value in expected.items():
        if pilot.get(key) != value:
            raise PilotError(
                f"MODEL_PROTOCOL length_ceiling_pilot.{key} drift: "
                f"{pilot.get(key)!r} != {value!r}"
            )
    scenarios = payload.get("toy_contract", {}).get("reported_scenarios", [])
    if tuple(map(str, scenarios)) != SCENARIOS:
        raise PilotError("MODEL_PROTOCOL five-lane order/contents drift")
    return payload


def load_v4p8_core() -> Any:
    global _CORE
    if _CORE is not None:
        return _CORE
    require_hash(V4P8_CORE_PATH, EXPECTED_V4P8_CORE_SHA256, "v4.8 length core")
    # Match the authoritative core's path ordering: repository support first,
    # then the audited overlay at higher precedence.  Pre-seeding both entries
    # also prevents the dynamically loaded core from moving REPO ahead of the
    # runtime overlay when only one entry was already present.
    for entry in (str(V4P8_RUNTIME_ROOT), str(REPO)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(V4P8_RUNTIME_ROOT))
    module_spec = importlib.util.spec_from_file_location(
        "_v4p8_authoritative_rigid_length_core", V4P8_CORE_PATH
    )
    if module_spec is None or module_spec.loader is None:
        raise PilotError("cannot load authoritative v4.8 length core")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    _CORE = module
    return module


def _runtime_preflight() -> dict[str, Any]:
    require_hash(V4P8_SPEC_PATH, EXPECTED_V4P8_SPEC_SHA256, "v4.8 rigid study spec")
    require_hash(
        V4P8_RUNTIME_MANIFEST,
        EXPECTED_V4P8_RUNTIME_MANIFEST_SHA256,
        "v4.8 runtime manifest",
    )
    require_hash(V4P8_CARD_PATH, EXPECTED_V4P2_CARD_SHA256, "frozen v4.2 card")
    manifest = load_json(V4P8_RUNTIME_MANIFEST)
    if int(manifest.get("schema_version", -1)) != 1:
        raise PilotError("unsupported v4.8 runtime-manifest schema")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise PilotError("v4.8 runtime manifest has no file inventory")
    actual_files = {
        str(path.relative_to(V4P8_RUNTIME_ROOT))
        for path in (V4P8_RUNTIME_ROOT / "hps_gpr").rglob("*.py")
        if path.is_file()
    }
    if actual_files != set(map(str, files)):
        raise PilotError("v4.8 runtime file-set differs from its manifest")
    for relative, expected in files.items():
        require_hash(
            V4P8_RUNTIME_ROOT / str(relative),
            str(expected),
            f"v4.8 runtime file {relative}",
        )

    core = load_v4p8_core()
    config_audit = core.controlled_config_audit()
    for factor in UPPER_FACTORS:
        cfg = core.build_config(factor)
        core.assert_config(cfg, factor)
    runtime_modules = dict(getattr(core, "EXPECTED_RUNTIME_SHA256", {}))
    if not runtime_modules:
        raise PilotError("authoritative v4.8 core has no runtime-module inventory")
    if not set(runtime_modules).issubset(set(files)):
        raise PilotError("v4.8 core runtime-module inventory is absent from overlay")
    for relative, expected in runtime_modules.items():
        if str(files[relative]) != str(expected):
            raise PilotError(f"v4.8 core/manifest runtime hash drift: {relative}")
        module_name = str(relative).replace("/", ".")[:-3]
        imported = importlib.import_module(module_name)
        imported_path = Path(str(imported.__file__)).resolve()
        if V4P8_RUNTIME_ROOT.resolve() not in imported_path.parents:
            raise PilotError(f"runtime module did not resolve to v4.8 overlay: {module_name}")
        require_hash(imported_path, str(expected), f"imported runtime {relative}")
    return {
        "v4p8_core_sha256": EXPECTED_V4P8_CORE_SHA256,
        "v4p8_spec_sha256": EXPECTED_V4P8_SPEC_SHA256,
        "runtime_manifest_sha256": EXPECTED_V4P8_RUNTIME_MANIFEST_SHA256,
        "v4p2_card_sha256": EXPECTED_V4P2_CARD_SHA256,
        "runtime_files": len(files),
        "imported_runtime_modules": len(runtime_modules),
        "controlled_one_factor_config_audit": config_audit,
    }


def _source_fit_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import residual_models

    result = residual_models.load_fit_result(require_influence=True)
    if str(result.get("protocol_sha256")) != sha256_file(PROTOCOL_PATH):
        raise PilotError("source-fit result does not match MODEL_PROTOCOL")
    if str(result.get("study_id")) != str(payload.get("study_id")):
        raise PilotError("source-fit study identity mismatch")
    rows = result.get("signal_influence_audit", {}).get("rows", [])
    if len(rows) != 2 * 2 * 41 * 3:
        raise PilotError("signal-influence audit row inventory is incomplete")
    disposition: dict[str, Any] = {}
    for model in MODELS:
        record = result.get("models", {}).get(model, {})
        if not bool(record.get("conditional_toy_run_authorized", False)):
            raise PilotError(f"conditional toy run is not authorized for {model}")
        disposition[model] = {
            "strict_generator_qualification_passed": bool(
                record.get("strict_generator_qualification_passed", False)
            ),
            "promotion_scope": str(record.get("promotion_scope", "")),
            "signal_influence_gate_passed": bool(
                result["signal_influence_audit"]["summaries"][model].get(
                    "signal_influence_gate_passed", False
                )
            ),
        }
    return {
        "fit_result_sha256": sha256_file(FIT_RESULT_PATH),
        "fit_protocol_sha256": str(result["protocol_sha256"]),
        "fit_implementation_sha256": str(result["implementation_sha256"]),
        "fit_driver_sha256": str(result["driver_sha256"]),
        "influence_rows": len(rows),
        "model_disposition": disposition,
        "claim_boundary": str(payload.get("claim_boundary", "")),
    }


def _first_hash(payload: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and len(value) == 64:
            return value
    for container_name in ("root", "fit_provenance", "provenance"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            for name in names:
                value = container.get(name)
                if isinstance(value, str) and len(value) == 64:
                    return value
            value = container.get("sha256")
            if isinstance(value, str) and len(value) == 64 and "root" in names[0]:
                return value
    return None


def _row_seed_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if "seed" in str(key).lower()
    }


def _phase_rows(
    manifest: Mapping[str, Any], phase: str
) -> list[Mapping[str, Any]]:
    """Accept the final combined ledger and the earlier split-ledger schema."""

    split_key = "pilot_toys" if phase == "pilot" else "closure_toys"
    split = manifest.get(split_key)
    if isinstance(split, list):
        return [row for row in split if isinstance(row, Mapping)]
    combined = manifest.get("toys")
    if not isinstance(combined, list):
        return []
    accepted_phases = {phase}
    if phase == "toys":
        accepted_phases.add("closure")
    return [
        row
        for row in combined
        if isinstance(row, Mapping) and str(row.get("phase", "")) in accepted_phases
    ]


def _validate_prelog_counts(values: np.ndarray, edges: np.ndarray, key: str) -> None:
    usable = values.size // 5 * 5
    rebinned = values[:usable].reshape(-1, 5).sum(axis=1)
    rebinned_edges = edges[: usable + 1 : 5]
    if rebinned_edges.size != rebinned.size + 1:
        raise PilotError(f"invalid rebin-5 geometry: {key}")
    centers = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    support = (centers >= SUPPORT_GEV[0]) & (centers < SUPPORT_GEV[1])
    if not np.any(support) or np.any(rebinned[support] <= 0):
        raise PilotError(f"nonpositive pre-log rebin-5 support count: {key}")


def validate_toy_inputs(root_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not root_path.is_file() or not manifest_path.is_file():
        missing = [str(path) for path in (root_path, manifest_path) if not path.is_file()]
        raise PilotError(f"missing residual pilot input(s): {missing}")
    manifest = load_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise PilotError("unsupported residual toy-manifest schema")

    protocol_record = manifest.get("protocol", {})
    fit_record = manifest.get("source_fit_and_influence", {})
    root_hash = _first_hash(manifest, ("root_sha256", "toy_root_sha256"))
    protocol_hash = _first_hash(manifest, ("protocol_sha256",))
    if protocol_hash is None and isinstance(protocol_record, Mapping):
        protocol_hash = str(protocol_record.get("sha256", "")) or None
    result_hash = _first_hash(
        manifest,
        ("result_sha256", "fit_result_sha256", "source_fit_result_sha256"),
    )
    if result_hash is None and isinstance(fit_record, Mapping):
        result_hash = str(fit_record.get("sha256", "")) or None
    if root_hash is None or protocol_hash is None or result_hash is None:
        raise PilotError("toy manifest lacks root/protocol/result SHA-256 provenance")
    if (
        str(manifest.get("study_id", "")) != str(protocol()["study_id"])
        or tuple(map(str, manifest.get("models", ()))) != MODELS
        or tuple(map(str, manifest.get("reported_scenarios", ()))) != SCENARIOS
        or not bool(manifest.get("nested_poisson_within_source_family", True))
        or not bool(manifest.get("pilot_and_closure_streams_independent", True))
        or not bool(manifest.get("model_streams_distinct", True))
    ):
        raise PilotError("toy manifest study/lattice/independence contract drift")
    require_hash(root_path, root_hash, "residual pilot ROOT")
    if protocol_hash != sha256_file(PROTOCOL_PATH):
        raise PilotError("toy manifest protocol hash is stale")
    if result_hash != sha256_file(FIT_RESULT_PATH):
        raise PilotError("toy manifest source-fit result hash is stale")

    builder_record = manifest.get("builder", {})
    builder_hash = _first_hash(manifest, ("builder_sha256",))
    builder_path_value = manifest.get("builder_path")
    if isinstance(builder_record, Mapping):
        builder_hash = builder_hash or str(builder_record.get("sha256", "")) or None
        builder_path_value = builder_path_value or builder_record.get("path")
    if builder_hash is not None and isinstance(builder_path_value, str):
        require_hash(resolve_study_path(builder_path_value), builder_hash, "toy builder")

    rows = _phase_rows(manifest, "pilot")
    if not rows:
        raise PilotError("toy manifest has no pilot toy rows")
    expected = {
        (model, scenario, toy_index)
        for model in MODELS
        for scenario in SCENARIOS
        for toy_index in PILOT_TOY_INDICES
    }
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PilotError("pilot_toys row is not an object")
        identity = (
            str(row.get("model", "")),
            str(row.get("scenario", row.get("lane", ""))),
            int(row.get("toy_index", -1)),
        )
        if identity in indexed:
            raise PilotError(f"duplicate pilot toy identity: {identity}")
        indexed[identity] = row
    if set(indexed) != expected:
        missing = sorted(expected.difference(indexed))
        extra = sorted(set(indexed).difference(expected))
        raise PilotError(f"pilot toy lattice mismatch: missing={missing}, extra={extra}")

    count_hashes: dict[str, str] = {}
    cached_counts: dict[tuple[str, str, int], np.ndarray] = {}
    common_edges: np.ndarray | None = None
    with uproot.open(root_path) as root_file:
        for identity in sorted(expected):
            model, scenario, toy_index = identity
            row = indexed[identity]
            expected_key = f"pilot/{model}/{scenario}/toy_{toy_index:04d}"
            key = str(row.get("key", row.get("root_key", "")))
            if key != expected_key:
                raise PilotError(f"pilot key mismatch: {identity}: {key!r}")
            if key not in root_file:
                raise PilotError(f"missing pilot histogram: {key}")
            values, edges = root_file[key].to_numpy(flow=False)
            values = np.asarray(values, dtype=float)
            edges = np.asarray(edges, dtype=float)
            rounded = np.rint(values)
            if (
                values.ndim != 1
                or edges.shape != (values.size + 1,)
                or np.any(~np.isfinite(values))
                or np.any(values < 0)
                or not np.allclose(values, rounded, rtol=0.0, atol=1e-9)
            ):
                raise PilotError(f"invalid integer pilot histogram: {key}")
            if edges[0] > SUPPORT_GEV[0] or edges[-1] < SUPPORT_GEV[1]:
                raise PilotError(f"pilot histogram does not cover 40--300 MeV: {key}")
            if common_edges is None:
                common_edges = edges.copy()
            elif not np.array_equal(edges, common_edges):
                raise PilotError(f"pilot histogram edge drift: {key}")
            centers = 0.5 * (edges[:-1] + edges[1:])
            outside_support = (centers < SUPPORT_GEV[0]) | (
                centers >= SUPPORT_GEV[1]
            )
            if np.any(rounded[outside_support] != 0):
                raise PilotError(f"pilot has counts outside 40--300 MeV: {key}")
            digest = array_hash(rounded, "<i8")
            declared = str(row.get("counts_sha256_int64", ""))
            if digest != declared:
                raise PilotError(f"pilot count hash mismatch: {key}")
            if int(row.get("total", row.get("total_count", -1))) != int(
                np.sum(rounded)
            ):
                raise PilotError(f"pilot total-count mismatch: {key}")
            _validate_prelog_counts(rounded, edges, key)
            count_hashes[key] = digest
            cached_counts[identity] = rounded.astype(np.int64)

    parent_scenario = {
        "2021_1pct_x10": "2021_1pct",
        "2021_1pct_x100": "2021_1pct_x10",
        "2021_10pct_x10": "2021_10pct",
    }
    scenario_semantics = {
        "2021_1pct": ("one_pct", 1, None, 1),
        "2021_1pct_x10": ("one_pct", 10, "2021_1pct", 9),
        "2021_1pct_x100": ("one_pct", 100, "2021_1pct_x10", 90),
        "2021_10pct": ("ten_pct", 1, None, 1),
        "2021_10pct_x10": ("ten_pct", 10, "2021_10pct", 9),
    }
    for identity, row in indexed.items():
        model, scenario, toy_index = identity
        family, multiplier, parent, increment_multiplier = scenario_semantics[scenario]
        if (
            str(row.get("source_family", "")) != family
            or int(row.get("multiplier", -1)) != multiplier
            or row.get("parent_scenario") != parent
            or int(row.get("increment_multiplier", -1)) != increment_multiplier
        ):
            raise PilotError(f"pilot manifest semantics drift: {identity}")
        namespace = row.get("namespace")
        if isinstance(namespace, list) and namespace[:4] != [
            model,
            "pilot",
            family,
            toy_index,
        ]:
            raise PilotError(f"pilot seed namespace drift: {identity}")
    for model in MODELS:
        for toy_index in PILOT_TOY_INDICES:
            for child, parent in parent_scenario.items():
                increment = (
                    cached_counts[(model, child, toy_index)]
                    - cached_counts[(model, parent, toy_index)]
                )
                if np.any(increment < 0):
                    raise PilotError(
                        f"pilot Poisson nesting failure: {model}/{child}/toy {toy_index}"
                    )
                row = indexed[(model, child, toy_index)]
                declared_increment = row.get("increment_sha256_int64")
                if isinstance(declared_increment, str) and declared_increment:
                    if array_hash(increment, "<i8") != declared_increment:
                        raise PilotError(
                            f"pilot increment hash mismatch: {model}/{child}/toy {toy_index}"
                        )

    closure_rows = _phase_rows(manifest, "toys")
    if closure_rows:
        closure_index = {
            (
                str(row.get("model", "")),
                str(row.get("scenario", row.get("lane", ""))),
                int(row.get("toy_index", -1)),
            ): row
            for row in closure_rows
            if isinstance(row, Mapping)
        }
        for identity, pilot_row in indexed.items():
            closure_row = closure_index.get(identity)
            if closure_row is None:
                raise PilotError(f"closure counterpart missing for pilot independence: {identity}")
            pilot_seed = _row_seed_payload(pilot_row)
            closure_seed = _row_seed_payload(closure_row)
            if not pilot_seed or not closure_seed or pilot_seed == closure_seed:
                raise PilotError(f"pilot/closure seed streams are not distinct: {identity}")

    for scenario in SCENARIOS:
        for toy_index in PILOT_TOY_INDICES:
            if (
                count_hashes[f"pilot/knot_spline/{scenario}/toy_{toy_index:04d}"]
                == count_hashes[
                    f"pilot/regional_blend/{scenario}/toy_{toy_index:04d}"
                ]
            ):
                raise PilotError(
                    f"model pilot streams are not distinct: {scenario}/toy {toy_index}"
                )

    edge_record = manifest.get("edges", {})
    declared_edge_hash = (
        str(edge_record.get("sha256_float64", ""))
        if isinstance(edge_record, Mapping)
        else ""
    )
    measured_edge_hash = array_hash(common_edges, "<f8")
    if declared_edge_hash and declared_edge_hash != measured_edge_hash:
        raise PilotError("pilot ROOT edge hash differs from toy manifest")

    return {
        "root": str(root_path),
        "root_sha256": root_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "pilot_histograms": len(indexed),
        "pilot_count_inventory_sha256": canonical_json_hash(count_hashes),
        "edges_sha256_float64": measured_edge_hash,
        "nested_poisson_checked": True,
        "pilot_closure_stream_independence_checked": bool(closure_rows),
        "model_stream_independence_checked": True,
    }


def optimizer_gate() -> dict[str, Any]:
    spec = load_json(V4P8_SPEC_PATH)
    source = spec.get("optimizer_gate", {})
    keys = (
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
    gate = {key: source[key] for key in keys}
    if (
        int(gate["reference_initial_attempts"]) != 3
        or int(gate["maximum_attempts"]) != 5
        or int(gate["top_branch_min_replicates"]) != 2
    ):
        raise PilotError("v4.8 optimizer-attempt topology drift")
    return gate


def scan_contract(
    root_path: Path, manifest_path: Path, input_record: Mapping[str, Any]
) -> dict[str, Any]:
    payload = protocol()
    pilot_gate = payload["length_ceiling_pilot"]["factor20_gate"]
    return {
        "schema_version": 1,
        "study_id": payload["study_id"],
        "driver_path": DRIVER_PATH.name,
        "driver_sha256": sha256_file(DRIVER_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "fit_result_sha256": sha256_file(FIT_RESULT_PATH),
        "toy_root": str(root_path),
        "toy_root_sha256": input_record["root_sha256"],
        "toy_manifest": str(manifest_path),
        "toy_manifest_sha256": input_record["manifest_sha256"],
        "v4p8_core_sha256": EXPECTED_V4P8_CORE_SHA256,
        "v4p8_runtime_manifest_sha256": EXPECTED_V4P8_RUNTIME_MANIFEST_SHA256,
        "v4p2_card_sha256": EXPECTED_V4P2_CARD_SHA256,
        "background_only": True,
        "models": list(MODELS),
        "scenarios": list(SCENARIOS),
        "pilot_toy_indices": list(PILOT_TOY_INDICES),
        "masses_gev": list(MASS_GRID),
        "upper_factors": list(UPPER_FACTORS),
        "workers": 1,
        "support_gev": list(SUPPORT_GEV),
        "optimizer_restarts": OPTIMIZER_RESTARTS,
        "optimizer_seed_namespace": SEED_NAMESPACE,
        "optimizer_seed_excludes_upper_factor": True,
        "optimizer_gate": optimizer_gate(),
        "factor20_gate": pilot_gate,
        "factor20_comparison_quantile_method": "numpy linear",
        "common_ceiling_for_models_and_lanes": True,
        "fallback_factor": int(payload["length_ceiling_pilot"]["fallback_factor"]),
        "prohibited_output_column_substrings": sorted(
            FORBIDDEN_OUTPUT_COLUMN_SUBSTRINGS
        ),
        "pull_blind": True,
        "inference_products_produced": False,
    }


def preflight(
    root_path: Path,
    manifest_path: Path,
    *,
    allow_missing_toys: bool = False,
) -> dict[str, Any]:
    payload = protocol()
    source_record = _source_fit_preflight(payload)
    runtime_record = _runtime_preflight()
    missing = [str(path) for path in (root_path, manifest_path) if not path.is_file()]
    if missing:
        if not allow_missing_toys:
            raise PilotError(f"missing residual pilot input(s): {missing}")
        return {
            "status": "waiting_for_toy_inputs",
            "validated_utc": utc_now(),
            "missing_toy_inputs": missing,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "fit_result": source_record,
            "runtime": runtime_record,
            "expected_tasks": EXPECTED_TASKS,
            "expected_states": EXPECTED_STATES,
            "heavy_scan_launched": False,
        }
    input_record = validate_toy_inputs(root_path, manifest_path)
    contract = scan_contract(root_path, manifest_path, input_record)
    return {
        "status": "pass",
        "validated_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "fit_result": source_record,
        "runtime": runtime_record,
        "toy_inputs": input_record,
        "scan_contract_sha256": canonical_json_hash(contract),
        "expected_tasks": EXPECTED_TASKS,
        "expected_states": EXPECTED_STATES,
        "heavy_scan_launched": False,
    }


def _load_manifest_rows(manifest_path: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    rows = _phase_rows(load_json(manifest_path), "pilot")
    return {
        (
            str(row["model"]),
            str(row.get("scenario", row.get("lane"))),
            int(row["toy_index"]),
        ): row
        for row in rows
    }


def make_toy_dataset(
    model: str,
    scenario: str,
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

    row = _load_manifest_rows(manifest_path)[(model, scenario, toy_index)]
    key = str(row["key"])
    container, toy_name = key.rsplit("/", 1)
    histogram = load_funcform_toy_hist(
        str(root_path), container=container, toy_name=toy_name
    )
    base = make_datasets(cfg)["2021"]
    if (float(base.data_low), float(base.data_high)) != SUPPORT_GEV:
        raise PilotError("v4.8 runtime 2021 dataset support is not 40--300 MeV")
    toy_spec = FuncFormToySpec(
        source_root=str(root_path),
        container=container,
        function_tag=f"v4p8p3_{model}_{scenario}_pilot",
        toy_name=toy_name,
        toy_index=toy_index,
    )
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def fit_attempt(
    dataset: Any,
    cfg: Any,
    gate: Mapping[str, Any],
    model: str,
    scenario: str,
    toy_index: int,
    mass: float,
    upper_factor: int,
    attempt: int,
) -> dict[str, Any]:
    from hps_gpr.gpr import length_scale_x_to_mass_delta
    from hps_gpr.io import estimate_background_for_dataset

    core = load_v4p8_core()
    seed = stable_seed(model, scenario, toy_index, f"{mass:.9f}", attempt)
    cfg.gp_optimizer_random_state = int(seed)
    base = {
        "model": model,
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "mass_MeV": int(round(1000.0 * mass)),
        "upper_factor": int(upper_factor),
        "attempt": int(attempt),
        "optimizer_seed": int(seed),
        "optimizer_seed_namespace": SEED_NAMESPACE,
        "seed_includes_upper_factor": False,
        "optimizer_restarts": OPTIMIZER_RESTARTS,
        "background_only": True,
        "fit_ok": False,
        "error": "",
    }
    try:
        prediction = estimate_background_for_dataset(
            dataset,
            float(mass),
            cfg,
            restarts=OPTIMIZER_RESTARTS,
            optimize=True,
        )
        geometry = core.training_geometry(prediction, float(mass), cfg)
        covariance = core.covariance_diagnostics(prediction.cov, gate)
        ell = float(prediction.ls_opt)
        ell_lo = float(prediction.ls_lo)
        ell_hi = float(prediction.ls_hi)
        sigma_x = float(prediction.sigma_x)
        constant = float(prediction.const_opt)
        constant_lo = float(prediction.const_lo)
        constant_hi = float(prediction.const_hi)
        ell_over_hi = ell / ell_hi
        ell_over_lo = ell / ell_lo
        near_window = float(gate["bound_ratio_window"])
        record = {
            **base,
            "gp_lml": float(prediction.lml),
            "ell_opt": ell,
            "ell_lo": ell_lo,
            "ell_hi": ell_hi,
            "ell_init": float(prediction.ls_init),
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
            "kernel_constant_init": float(prediction.const_init),
            "n_blind": int(prediction.n_blind),
            "blind_lo": float(prediction.blind[0]),
            "blind_hi": float(prediction.blind[1]),
            "sigma_mass_GeV": float(prediction.sigma_val),
            "support_lo_GeV": float(prediction.edges_full[0]),
            "support_hi_GeV": float(prediction.edges_full[-1]),
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
            "optimizer_warning_count": int(prediction.optimizer_warning_count),
            "optimizer_warnings": str(prediction.optimizer_warnings),
            **geometry,
            **covariance,
        }
        finite = all(
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
        positive = all(
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
        record["fit_ok"] = bool(finite and positive and support_ok and factor_ok)
        if not record["fit_ok"]:
            record["error"] = "nonfinite/nonpositive/support/factor contract failure"
        return record
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:500]}


def validate_no_inference_columns(frame: pd.DataFrame, label: str) -> None:
    violations: dict[str, list[str]] = {}
    for column in frame.columns:
        normalized = "".join(
            character for character in str(column).lower() if character.isalnum()
        )
        matched = sorted(
            token
            for token in FORBIDDEN_OUTPUT_COLUMN_SUBSTRINGS
            if "".join(character for character in token if character.isalnum())
            in normalized
        )
        if matched:
            violations[str(column)] = matched
    if violations:
        raise PilotError(f"{label} has prohibited inference columns: {violations}")


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = pd.Series(dtype="object")
    return output


def task_directory(model: str, scenario: str, toy_index: int) -> Path:
    return RUNS / model / scenario / f"toy_{toy_index:04d}"


def expected_tasks() -> list[tuple[str, str, int]]:
    return [
        (model, scenario, toy_index)
        for model in MODELS
        for scenario in SCENARIOS
        for toy_index in PILOT_TOY_INDICES
    ]


def _read_task_products(directory: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in TASK_PRODUCT_NAMES:
        path = directory / name
        if not path.is_file():
            raise PilotError(f"missing task product: {path}")
        try:
            frames[name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frames[name] = pd.DataFrame(columns=EXCLUSION_COLUMNS if "exclusion" in name else [])
        validate_no_inference_columns(frames[name], name)
    return frames


def validate_success(
    directory: Path,
    contract_hash: str,
    model: str,
    scenario: str,
    toy_index: int,
) -> tuple[bool, str]:
    marker = directory / "_SUCCESS.json"
    if not marker.is_file():
        return False, "missing_success_marker"
    try:
        payload = load_json(marker)
        if payload.get("status") != "complete":
            return False, "noncomplete_marker"
        if str(payload.get("scan_contract_sha256")) != contract_hash:
            return False, "stale_contract"
        identity = (
            str(payload.get("model")),
            str(payload.get("scenario")),
            int(payload.get("background_toy_index", -1)),
        )
        if identity != (model, scenario, toy_index):
            return False, "identity_mismatch"
        frames = _read_task_products(directory)
        hashes = payload.get("product_sha256", {})
        for name in TASK_PRODUCT_NAMES:
            if str(hashes.get(name)) != sha256_file(directory / name):
                return False, f"product_hash_mismatch:{name}"
        selected = frames["selected_trajectories.csv"]
        exclusions = frames["optimizer_exclusions.csv"]
        if len(selected) + len(exclusions) != len(MASS_GRID) * len(UPPER_FACTORS):
            return False, "state_cardinality_mismatch"
    except Exception as exc:
        return False, f"invalid_success:{type(exc).__name__}:{exc}"
    return True, "current"


def run_task(
    model: str,
    scenario: str,
    toy_index: int,
    root_path: Path,
    manifest_path: Path,
    *,
    force: bool = False,
    preflight_done: bool = False,
) -> dict[str, Any]:
    if model not in MODELS or scenario not in SCENARIOS or toy_index not in PILOT_TOY_INDICES:
        raise PilotError("task identity is outside the frozen 2x5x3 pilot lattice")
    if not preflight_done:
        preflight(root_path, manifest_path)
    input_record = validate_toy_inputs(root_path, manifest_path)
    contract = scan_contract(root_path, manifest_path, input_record)
    contract_hash = canonical_json_hash(contract)
    directory = task_directory(model, scenario, toy_index)
    current, reason = validate_success(
        directory, contract_hash, model, scenario, toy_index
    )
    if current and not force:
        return {**load_json(directory / "_SUCCESS.json"), "cached": True}
    if directory.joinpath("_SUCCESS.json").exists() and not current and not force:
        raise PilotError(
            f"stale/corrupt task {model}/{scenario}/toy {toy_index}: {reason}; "
            "use --force to replace only this task's products"
        )

    core = load_v4p8_core()
    gate = optimizer_gate()
    configs = {factor: core.build_config(factor) for factor in UPPER_FACTORS}
    for factor, cfg in configs.items():
        core.assert_config(cfg, factor)
    dataset = make_toy_dataset(
        model, scenario, toy_index, configs[15], root_path, manifest_path
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
                        model,
                        scenario,
                        toy_index,
                        mass,
                        factor,
                        attempt,
                    )
                )
        initial = {
            factor: core.select_branch(by_factor[factor], gate)[0]
            for factor in UPPER_FACTORS
        }
        if any(record is None for record in initial.values()):
            for attempt in range(initial_attempts, maximum_attempts):
                for factor in UPPER_FACTORS:
                    by_factor[factor].append(
                        fit_attempt(
                            dataset,
                            configs[factor],
                            gate,
                            model,
                            scenario,
                            toy_index,
                            mass,
                            factor,
                            attempt,
                        )
                    )
        attempt_sets = {
            factor: tuple(int(row["attempt"]) for row in rows)
            for factor, rows in by_factor.items()
        }
        if len(set(attempt_sets.values())) != 1:
            raise PilotError("paired factors do not have a common attempt set")
        for attempt in attempt_sets[15]:
            seeds = {
                int(next(row for row in by_factor[factor] if int(row["attempt"]) == attempt)["optimizer_seed"])
                for factor in UPPER_FACTORS
            }
            if len(seeds) != 1:
                raise PilotError("paired factors do not share optimizer seeds")

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
                        "study_id": protocol()["study_id"],
                        "model": model,
                        "scenario": scenario,
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
                    "study_id": protocol()["study_id"],
                    "selected_attempt": int(selected["attempt"]),
                    "n_attempts": len(records),
                    "top_branch_replicates": int(replicates),
                    "optimizer_gate_status": "maximum_lml_reproduced",
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
        (
            "model",
            "scenario",
            "background_toy_index",
            "mass_GeV",
            "mass_MeV",
            "upper_factor",
            "ell_opt",
            "sigma_x",
            "ell_over_sigma_x",
            "gp_lml",
        ),
    )
    exclusions = _ensure_columns(pd.DataFrame(exclusion_rows), EXCLUSION_COLUMNS)
    if not exclusions.empty:
        exclusions = exclusions.sort_values(["mass_MeV", "upper_factor"])
    for label, frame in (
        ("task attempts", attempts),
        ("task selected trajectories", selected),
        ("task exclusions", exclusions),
    ):
        validate_no_inference_columns(frame, label)
    if len(selected) + len(exclusions) != len(MASS_GRID) * len(UPPER_FACTORS):
        raise PilotError("selected plus excluded rows do not cover the task state grid")

    directory.mkdir(parents=True, exist_ok=True)
    products = {
        "optimizer_attempts.csv": attempts,
        "selected_trajectories.csv": selected,
        "optimizer_exclusions.csv": exclusions,
    }
    for name, frame in products.items():
        atomic_csv(directory / name, frame)
    product_hashes = {
        name: sha256_file(directory / name) for name in TASK_PRODUCT_NAMES
    }
    result = {
        "schema_version": 1,
        "generation_uuid": str(uuid.uuid4()),
        "status": "complete",
        "scientific_status": (
            "pull_blind_optimizer_diagnostic_complete"
            if exclusions.empty
            else "pull_blind_optimizer_diagnostic_has_exclusions"
        ),
        "completed_utc": utc_now(),
        "study_id": protocol()["study_id"],
        "task_id": f"{model}__{scenario}__pilot_{toy_index:04d}",
        "model": model,
        "scenario": scenario,
        "background_toy_index": toy_index,
        "selected_rows": len(selected),
        "excluded_rows": len(exclusions),
        "attempt_rows": len(attempts),
        "scan_contract_sha256": contract_hash,
        "product_sha256": product_hashes,
        "background_only": True,
        "pull_blind": True,
        "common_seeds_across_factors": True,
        "factor_selection_performed": False,
        "cached": False,
    }
    # Recheck all inputs immediately before publishing the success marker.
    final_input = validate_toy_inputs(root_path, manifest_path)
    if canonical_json_hash(scan_contract(root_path, manifest_path, final_input)) != contract_hash:
        raise PilotError("pilot inputs or executable changed while task was running")
    atomic_json(directory / "_SUCCESS.json", result)
    return result


def prepare(root_path: Path, manifest_path: Path) -> dict[str, Any]:
    validation = preflight(root_path, manifest_path)
    input_record = validation["toy_inputs"]
    contract = scan_contract(root_path, manifest_path, input_record)
    contract_hash = canonical_json_hash(contract)
    rows = []
    for model, scenario, toy_index in expected_tasks():
        directory = task_directory(model, scenario, toy_index)
        current, reason = validate_success(
            directory, contract_hash, model, scenario, toy_index
        )
        rows.append(
            {
                "task_id": f"{model}__{scenario}__pilot_{toy_index:04d}",
                "model": model,
                "scenario": scenario,
                "background_toy_index": toy_index,
                "mass_grid_MeV": "65|120|210",
                "upper_factors": "15|20|25",
                "output_directory": str(directory.relative_to(HERE)),
                "command": (
                    f"python3 {DRIVER_PATH.name} run-task {model} {scenario} {toy_index}"
                ),
                "current": current,
                "status": reason,
            }
        )
    manifest = pd.DataFrame(rows)
    validate_no_inference_columns(manifest, "task manifest")
    atomic_csv(QA / "task_manifest.csv", manifest)
    atomic_json(QA / "scan_contract.json", contract)
    atomic_json(QA / "preflight.json", validation)
    return {
        "status": "pass",
        "tasks": len(manifest),
        "current_tasks": int(manifest["current"].sum()),
        "task_manifest": str((QA / "task_manifest.csv").relative_to(HERE)),
        "scan_contract": str((QA / "scan_contract.json").relative_to(HERE)),
        "scan_contract_sha256": contract_hash,
        "heavy_scan_launched": False,
    }


def task_status(root_path: Path, manifest_path: Path) -> dict[str, Any]:
    validation = preflight(root_path, manifest_path)
    contract_hash = canonical_json_hash(
        scan_contract(root_path, manifest_path, validation["toy_inputs"])
    )
    records = []
    for model, scenario, toy_index in expected_tasks():
        current, reason = validate_success(
            task_directory(model, scenario, toy_index),
            contract_hash,
            model,
            scenario,
            toy_index,
        )
        records.append(current)
    return {
        "status": "complete" if all(records) else "incomplete",
        "expected_tasks": len(records),
        "current_tasks": sum(records),
        "remaining_tasks": len(records) - sum(records),
        "heavy_scan_launched": False,
    }


def run_many(
    root_path: Path,
    manifest_path: Path,
    *,
    models: Sequence[str],
    scenarios: Sequence[str],
    toy_start: int,
    toy_stop: int,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    if workers != 1:
        raise PilotError("MODEL_PROTOCOL requires workers=1 for this pilot")
    if toy_start < 0 or toy_stop > 3 or toy_stop <= toy_start:
        raise PilotError("pilot toy interval must satisfy 0 <= start < stop <= 3")
    if not set(models).issubset(MODELS) or not set(scenarios).issubset(SCENARIOS):
        raise PilotError("run selection is outside the frozen model/lane lattice")
    preflight(root_path, manifest_path)
    results = []
    for model in models:
        for scenario in scenarios:
            for toy_index in range(toy_start, toy_stop):
                results.append(
                    run_task(
                        model,
                        scenario,
                        toy_index,
                        root_path,
                        manifest_path,
                        force=force,
                        preflight_done=True,
                    )
                )
    return {
        "status": "complete",
        "tasks": len(results),
        "cached_tasks": sum(bool(row.get("cached")) for row in results),
        "tasks_with_exclusions": sum(int(row.get("excluded_rows", 0)) > 0 for row in results),
        "workers": 1,
        "background_only": True,
        "pull_blind": True,
    }


def _factor20_comparisons(selected: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["model", "scenario", "background_toy_index", "mass_MeV"]
    for identity, group in selected.groupby(keys, sort=True):
        by_factor = {
            int(row["upper_factor"]): row
            for _, row in group.iterrows()
        }
        row20 = by_factor.get(20)
        row25 = by_factor.get(25)
        comparable = row20 is not None and row25 is not None
        record = dict(zip(keys, identity))
        record["comparable"] = comparable
        if comparable:
            n_train = max(1, min(int(row20["n_train"]), int(row25["n_train"])))
            sigma20 = float(row20["sigma_x"])
            sigma25 = float(row25["sigma_x"])
            if not math.isclose(sigma20, sigma25, rel_tol=0.0, abs_tol=1e-12):
                raise PilotError(f"sigma_x differs between factor 20 and 25: {identity}")
            record.update(
                {
                    "delta_lml_25_minus_20": float(row25["gp_lml"])
                    - float(row20["gp_lml"]),
                    "abs_delta_lml_per_training_bin_20_to_25": abs(
                        float(row25["gp_lml"]) - float(row20["gp_lml"])
                    )
                    / n_train,
                    "abs_delta_ell_over_sigma_x_20_to_25": abs(
                        float(row25["ell_opt"]) - float(row20["ell_opt"])
                    )
                    / sigma20,
                    "factor20_upper_exact": bool(row20["ell_at_upper_exact"]),
                    "factor20_upper_near": bool(row20["ell_near_upper"]),
                    "factor20_top_branch_replicates": int(
                        row20["top_branch_replicates"]
                    ),
                    "factor25_upper_exact": bool(row25["ell_at_upper_exact"]),
                    "factor25_upper_near": bool(row25["ell_near_upper"]),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def collect(
    root_path: Path,
    manifest_path: Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    validation = preflight(root_path, manifest_path)
    contract_hash = canonical_json_hash(
        scan_contract(root_path, manifest_path, validation["toy_inputs"])
    )
    task_rows = []
    attempt_parts = []
    selected_parts = []
    exclusion_parts = []
    for model, scenario, toy_index in expected_tasks():
        directory = task_directory(model, scenario, toy_index)
        current, reason = validate_success(
            directory, contract_hash, model, scenario, toy_index
        )
        task_rows.append(
            {
                "model": model,
                "scenario": scenario,
                "background_toy_index": toy_index,
                "current": current,
                "status": reason,
                "directory": str(directory.relative_to(HERE)),
            }
        )
        if not current:
            continue
        frames = _read_task_products(directory)
        attempt_parts.append(frames["optimizer_attempts.csv"])
        selected_parts.append(frames["selected_trajectories.csv"])
        exclusion_parts.append(frames["optimizer_exclusions.csv"])
    current_tasks = sum(bool(row["current"]) for row in task_rows)
    missing = EXPECTED_TASKS - current_tasks
    if missing and not allow_incomplete:
        raise PilotError(f"collection requires all {EXPECTED_TASKS} tasks; {missing} missing/stale")
    if current_tasks == 0:
        raise PilotError("no current pilot tasks are available to collect")

    attempts = pd.concat(attempt_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    exclusions = pd.concat(exclusion_parts, ignore_index=True)
    task_ledger = pd.DataFrame(task_rows)
    for label, frame in (
        ("task ledger", task_ledger),
        ("attempt ledger", attempts),
        ("selected ledger", selected),
        ("exclusion ledger", exclusions),
    ):
        validate_no_inference_columns(frame, label)

    seed_groups = attempts.groupby(
        ["model", "scenario", "background_toy_index", "mass_MeV", "attempt"],
        sort=False,
    )
    if bool((seed_groups["optimizer_seed"].nunique() != 1).any()):
        raise PilotError("collected factors do not share optimizer seeds")
    if bool((seed_groups["upper_factor"].nunique() != len(UPPER_FACTORS)).any()):
        raise PilotError("collected paired attempt sets are incomplete")

    comparisons = _factor20_comparisons(selected)
    validate_no_inference_columns(comparisons, "factor20-to25 comparison")
    comparable = comparisons[comparisons["comparable"].astype(bool)].copy()
    payload = protocol()
    gates = payload["length_ceiling_pilot"]["factor20_gate"]
    complete = missing == 0
    no_exclusions = exclusions.empty
    comparison_complete = len(comparable) == EXPECTED_COMPARISONS
    factor20_contacts = int(
        (
            comparable.get("factor20_upper_exact", pd.Series(dtype=bool)).fillna(False).astype(bool)
            | comparable.get("factor20_upper_near", pd.Series(dtype=bool)).fillna(False).astype(bool)
        ).sum()
    )
    minimum_repeats = (
        int(comparable["factor20_top_branch_replicates"].min())
        if len(comparable)
        else 0
    )
    lml_values = pd.to_numeric(
        comparable.get("abs_delta_lml_per_training_bin_20_to_25"), errors="coerce"
    ).dropna()
    ell_values = pd.to_numeric(
        comparable.get("abs_delta_ell_over_sigma_x_20_to_25"), errors="coerce"
    ).dropna()
    maximum_lml = float(lml_values.max()) if len(lml_values) else float("inf")
    median_ell = float(ell_values.median()) if len(ell_values) else float("inf")
    p95_ell = (
        float(np.quantile(ell_values.to_numpy(dtype=float), 0.95, method="linear"))
        if len(ell_values)
        else float("inf")
    )
    gate_results = {
        "all_tasks_complete": complete,
        "no_optimizer_exclusions": no_exclusions,
        "all_factor20_to25_states_comparable": comparison_complete,
        "factor20_exact_or_near_upper_contacts_zero": factor20_contacts
        <= int(gates["exact_or_near_contacts"]),
        "factor20_top_branch_minimum_repeats": minimum_repeats
        >= int(gates["top_branch_minimum_repeats"]),
        "maximum_abs_delta_lml_per_training_bin_20_to_25": maximum_lml
        <= float(gates["maximum_abs_delta_lml_per_training_bin_20_to_25"]),
        "median_abs_delta_ell_over_sigma_x_20_to_25": median_ell
        <= float(gates["median_abs_delta_ell_over_sigma_x_20_to_25"]),
        "p95_abs_delta_ell_over_sigma_x_20_to_25": p95_ell
        <= float(gates["p95_abs_delta_ell_over_sigma_x_20_to_25"]),
    }
    factor20_passed = all(gate_results.values())
    selected_factor = 20 if factor20_passed else int(
        payload["length_ceiling_pilot"]["fallback_factor"]
    )
    disposition = {
        "schema_version": 1,
        "status": "pass" if complete else "partial",
        "study_id": payload["study_id"],
        "created_utc": utc_now(),
        "scan_contract_sha256": contract_hash,
        "driver": str(DRIVER_PATH),
        "driver_sha256": sha256_file(DRIVER_PATH),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "fit_result": str(FIT_RESULT_PATH),
        "fit_result_sha256": sha256_file(FIT_RESULT_PATH),
        "toy_root": str(root_path),
        "toy_root_sha256": validation["toy_inputs"]["root_sha256"],
        "toy_manifest": str(manifest_path),
        "toy_manifest_sha256": validation["toy_inputs"]["manifest_sha256"],
        "v4p8_core": str(V4P8_CORE_PATH),
        "v4p8_core_sha256": EXPECTED_V4P8_CORE_SHA256,
        "v4p8_runtime_manifest": str(V4P8_RUNTIME_MANIFEST),
        "v4p8_runtime_manifest_sha256": EXPECTED_V4P8_RUNTIME_MANIFEST_SHA256,
        "v4p2_card": str(V4P8_CARD_PATH),
        "v4p2_card_sha256": EXPECTED_V4P2_CARD_SHA256,
        "selected_common_upper_factor": selected_factor,
        "factor20_gate_passed": factor20_passed,
        "fallback_factor_used": not factor20_passed,
        "common_ceiling_for_models_and_lanes": True,
        "observed_metrics": {
            "current_tasks": current_tasks,
            "optimizer_exclusion_rows": len(exclusions),
            "comparable_factor20_to25_states": len(comparable),
            "factor20_exact_or_near_upper_contacts": factor20_contacts,
            "factor20_minimum_top_branch_repeats": minimum_repeats,
            "maximum_abs_delta_lml_per_training_bin_20_to_25": maximum_lml,
            "median_abs_delta_ell_over_sigma_x_20_to_25": median_ell,
            "p95_abs_delta_ell_over_sigma_x_20_to_25": p95_ell,
        },
        "predeclared_thresholds": gates,
        "gate_results": gate_results,
        "production_v4p2_upper_factor_unchanged": 15,
        "interpretation": (
            "Pull-blind background-only optimizer pilot. The selected common "
            "ceiling applies only to the requested conditional residual-truth "
            "stress toys and does not alter the v4.2 production card."
        ),
        "inference_quantities_inspected": False,
    }

    products = {
        "task_ledger.csv": task_ledger,
        "optimizer_attempt_ledger.csv": attempts,
        "selected_trajectory_ledger.csv": selected,
        "optimizer_exclusion_ledger.csv": exclusions,
        "factor20_to25_comparison.csv": comparisons,
    }
    for name, frame in products.items():
        atomic_csv(DERIVED / name, frame)
    atomic_json(DERIVED / "common_ceiling_disposition.json", disposition)
    summary = {
        "status": "complete" if complete else "partial",
        "collected_utc": utc_now(),
        "current_tasks": current_tasks,
        "missing_or_stale_tasks": missing,
        "attempt_rows": len(attempts),
        "selected_rows": len(selected),
        "optimizer_exclusion_rows": len(exclusions),
        "scan_contract_sha256": contract_hash,
        "selected_common_upper_factor": selected_factor,
        "common_ceiling_disposition": str(
            (DERIVED / "common_ceiling_disposition.json").relative_to(HERE)
        ),
        "derived_sha256": {
            name: sha256_file(DERIVED / name) for name in products
        },
        "background_only": True,
        "pull_blind": True,
        "inference_quantities_inspected": False,
    }
    summary["derived_sha256"]["common_ceiling_disposition.json"] = sha256_file(
        DERIVED / "common_ceiling_disposition.json"
    )
    atomic_json(DERIVED / "collection_summary.json", summary)
    return summary


def validate_command(
    root_path: Path,
    manifest_path: Path,
    *,
    allow_missing_toys: bool = False,
) -> dict[str, Any]:
    result = preflight(
        root_path, manifest_path, allow_missing_toys=allow_missing_toys
    )
    dummy = pd.DataFrame(columns=EXCLUSION_COLUMNS)
    validate_no_inference_columns(dummy, "declared exclusion schema")
    for probe in (
        "mean_pull",
        "fittedAmplitude",
        "local_p0",
        "CLs_alpha",
        "epsilon2_limit",
        "coverage_fraction",
    ):
        try:
            validate_no_inference_columns(pd.DataFrame(columns=[probe]), "self-test")
        except PilotError:
            pass
        else:
            raise PilotError(f"forbidden-column matcher failed for {probe}")
    return {
        "status": result["status"],
        "preflight": result,
        "expected_resumable_tasks": EXPECTED_TASKS,
        "expected_selected_states": EXPECTED_STATES,
        "expected_factor20_to25_comparisons": EXPECTED_COMPARISONS,
        "workers": 1,
        "background_only": True,
        "pull_blind": True,
        "heavy_scan_launched": False,
    }


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--toy-root", type=Path, default=TOY_ROOT_DEFAULT)
    parser.add_argument("--toy-manifest", type=Path, default=TOY_MANIFEST_DEFAULT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "validate"):
        subparser = subparsers.add_parser(name)
        add_input_arguments(subparser)
        subparser.add_argument("--allow-missing-toys", action="store_true")
    prepare_parser = subparsers.add_parser("prepare")
    add_input_arguments(prepare_parser)
    status_parser = subparsers.add_parser("status")
    add_input_arguments(status_parser)
    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("model", choices=MODELS)
    task_parser.add_argument("scenario", choices=SCENARIOS)
    task_parser.add_argument("toy_index", type=int)
    task_parser.add_argument("--force", action="store_true")
    add_input_arguments(task_parser)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", action="append", choices=MODELS)
    run_parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    run_parser.add_argument("--toy-start", type=int, default=0)
    run_parser.add_argument("--toy-stop", type=int, default=3)
    run_parser.add_argument("--workers", type=int, choices=(1,), default=1)
    run_parser.add_argument("--force", action="store_true")
    add_input_arguments(run_parser)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--allow-incomplete", action="store_true")
    add_input_arguments(collect_parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_path = Path(args.toy_root).resolve()
    manifest_path = Path(args.toy_manifest).resolve()
    if args.command == "preflight":
        result = preflight(
            root_path,
            manifest_path,
            allow_missing_toys=args.allow_missing_toys,
        )
    elif args.command == "validate":
        result = validate_command(
            root_path,
            manifest_path,
            allow_missing_toys=args.allow_missing_toys,
        )
    elif args.command == "prepare":
        result = prepare(root_path, manifest_path)
    elif args.command == "status":
        result = task_status(root_path, manifest_path)
    elif args.command == "run-task":
        result = run_task(
            args.model,
            args.scenario,
            args.toy_index,
            root_path,
            manifest_path,
            force=args.force,
        )
    elif args.command == "run":
        result = run_many(
            root_path,
            manifest_path,
            models=tuple(args.model) if args.model else MODELS,
            scenarios=tuple(args.scenario) if args.scenario else SCENARIOS,
            toy_start=args.toy_start,
            toy_stop=args.toy_stop,
            workers=args.workers,
            force=args.force,
        )
    elif args.command == "collect":
        result = collect(
            root_path,
            manifest_path,
            allow_incomplete=args.allow_incomplete,
        )
    else:
        raise PilotError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
