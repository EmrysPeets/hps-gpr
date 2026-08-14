#!/usr/bin/env python3
"""Build eight fresh pull-blind ceiling-remediation backgrounds.

``preflight`` is read-only and does not create the ROOT or manifest products.
``build`` is the only command that generates them, and it refuses to overwrite
an existing product.  ``validate`` performs a complete deterministic inventory
and content check after generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import uproot

import residual_models as models


HERE = Path(__file__).resolve().parent
DRIVER_PATH = Path(__file__).resolve()
ADDENDUM_PATH = HERE / "CEILING_REMEDIATION_ADDENDUM.json"
EXPECTED_ADDENDUM_SHA256 = (
    "40d81bca0ded24821d2f1213e3df9a6ab1c904242b0e89ea2ad5773533e5fb1d"
)
ROOT_PATH = HERE / "inputs/residual_length_ceiling_remediation_toys.root"
MANIFEST_PATH = (
    HERE / "inputs/residual_length_ceiling_remediation_toys.manifest.json"
)
ORIGINAL_MANIFEST_PATH = HERE / "inputs/residual_structured_nested_toys.manifest.json"

MODEL = "knot_spline"
SOURCE_FAMILY = "one_pct"
SCENARIO = "2021_1pct"
TOY_INDICES = tuple(range(8))
SELECTION_INDICES = (0, 1, 2)
CONFIRMATION_INDICES = (3, 4, 5, 6, 7)
BASE_SEED = 20260814
SEED_NAMESPACE = "v4p8p3_residual_length_ceiling_remediation_background_v1"
TRUTH_KEY = f"truth/{MODEL}/{SCENARIO}_mean"


class RemediationToyError(RuntimeError):
    """Raised when the frozen remediation-toy contract is violated."""


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RemediationToyError(f"JSON root must be an object: {path}")
    return payload


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RemediationToyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise RemediationToyError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}: {path}"
        )


def resolve_study_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (HERE / path).resolve()


def load_addendum() -> dict[str, Any]:
    require_hash(ADDENDUM_PATH, EXPECTED_ADDENDUM_SHA256, "frozen addendum")
    payload = load_json(ADDENDUM_PATH)
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("addendum_id")
        != "v4p8p3_residual_length_ceiling_remediation_v1"
        or payload.get("status")
        != "frozen_before_remediation_toy_generation_or_fits"
    ):
        raise RemediationToyError("unsupported or unfrozen remediation addendum")
    if not bool(
        payload.get("post_closure_initiation", {}).get(
            "original_closure_pulls_were_inspected"
        )
    ):
        raise RemediationToyError("post-closure inspection disclosure is missing")
    frozen_inputs = payload.get("frozen_inputs")
    if not isinstance(frozen_inputs, Mapping) or not frozen_inputs:
        raise RemediationToyError("addendum has no frozen-input ledger")
    for label, record in frozen_inputs.items():
        if not isinstance(record, Mapping):
            raise RemediationToyError(f"invalid frozen-input record: {label}")
        require_hash(
            resolve_study_path(str(record["path"])),
            str(record["sha256"]),
            f"addendum input {label}",
        )
    target = payload.get("target_scope", {})
    expected_target = {
        "model": MODEL,
        "source_family": SOURCE_FAMILY,
        "scenario": SCENARIO,
        "background_only": True,
        "masses_gev": [0.065, 0.12, 0.21],
        "upper_factors": [25, 35, 50, 75],
        "candidate_selection_order": [35, 50],
        "fallback": None,
        "all_lane_confirmation": False,
        "closure_rerun": False,
    }
    for key, expected in expected_target.items():
        if target.get(key) != expected:
            raise RemediationToyError(
                f"addendum target_scope.{key} drift: {target.get(key)!r}"
            )
    backgrounds = payload.get("fresh_background_contract", {})
    expected_backgrounds = {
        "count": 8,
        "selection_toy_indices": list(SELECTION_INDICES),
        "confirmation_toy_indices": list(CONFIRMATION_INDICES),
        "base_seed": BASE_SEED,
        "seed_namespace": SEED_NAMESPACE,
        "reserve_toys": 0,
    }
    for key, expected in expected_backgrounds.items():
        if backgrounds.get(key) != expected:
            raise RemediationToyError(
                f"addendum fresh_background_contract.{key} drift"
            )
    return payload


def seed_words(stage: str, toy_index: int) -> list[int]:
    material = "|".join(
        (
            str(BASE_SEED),
            SEED_NAMESPACE,
            MODEL,
            SOURCE_FAMILY,
            SCENARIO,
            stage,
            str(int(toy_index)),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()


def stage_for(toy_index: int) -> str:
    if toy_index in SELECTION_INDICES:
        return "select"
    if toy_index in CONFIRMATION_INDICES:
        return "confirm"
    raise RemediationToyError(f"toy index outside frozen lattice: {toy_index}")


def rng_for(stage: str, toy_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed_words(stage, toy_index)))


def source_mean() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    result = models.load_fit_result(require_influence=True)
    model_record = result.get("models", {}).get(MODEL, {})
    if not bool(model_record.get("conditional_toy_run_authorized")):
        raise RemediationToyError("knot_spline conditional toy authorization is absent")
    if bool(model_record.get("strict_generator_qualification_passed")):
        raise RemediationToyError("remediation must not promote source qualification")
    mean, edges = models.frozen_mean_full(MODEL, SOURCE_FAMILY, result)
    mean = np.asarray(mean, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if (
        len(edges) != len(mean) + 1
        or np.any(~np.isfinite(mean))
        or np.any(mean < 0)
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise RemediationToyError("invalid frozen knot_spline/one_pct mean")
    declared_support = (
        result.get("models", {})
        .get(MODEL, {})
        .get("fits", {})
        .get(SOURCE_FAMILY, {})
        .get("mean_sha256_float64")
    )
    histogram = models.load_histogram(SOURCE_FAMILY)
    actual_support = array_hash(mean[histogram.support_mask], "<f8")
    if declared_support != actual_support:
        raise RemediationToyError(
            "frozen support mean hash differs from source-fit ledger"
        )
    original_manifest = load_json(ORIGINAL_MANIFEST_PATH)
    original_truth = [
        row
        for row in original_manifest.get("truths", [])
        if row.get("model") == MODEL and row.get("scenario") == SCENARIO
    ]
    if (
        len(original_truth) != 1
        or original_truth[0].get("mean_sha256_float64")
        != array_hash(mean, "<f8")
    ):
        raise RemediationToyError(
            "frozen full mean hash differs from the pinned original toy ledger"
        )
    return mean, edges, result


def original_seed_inventory() -> set[tuple[int, ...]]:
    manifest = load_json(ORIGINAL_MANIFEST_PATH)
    rows = manifest.get("toys", [])
    if not isinstance(rows, list) or len(rows) != 230:
        raise RemediationToyError("original toy manifest cardinality drift")
    return {
        tuple(map(int, row["seed_words_uint32"]))
        for row in rows
        if isinstance(row, Mapping) and "seed_words_uint32" in row
    }


def preflight() -> dict[str, Any]:
    addendum = load_addendum()
    mean, edges, _ = source_mean()
    proposed = [
        {
            "toy_index": toy_index,
            "stage": stage_for(toy_index),
            "namespace": [
                SEED_NAMESPACE,
                MODEL,
                SOURCE_FAMILY,
                SCENARIO,
                stage_for(toy_index),
                toy_index,
            ],
            "seed_words_uint32": seed_words(stage_for(toy_index), toy_index),
        }
        for toy_index in TOY_INDICES
    ]
    seeds = [tuple(row["seed_words_uint32"]) for row in proposed]
    if len(set(seeds)) != len(seeds):
        raise RemediationToyError("remediation background seed collision")
    overlap = set(seeds) & original_seed_inventory()
    if overlap:
        raise RemediationToyError("remediation seeds overlap original toy seeds")
    if set(SELECTION_INDICES) & set(CONFIRMATION_INDICES):
        raise RemediationToyError("selection and confirmation toy sets overlap")
    return {
        "status": "pass",
        "mode": "read_only_preflight",
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "builder_sha256": sha256_file(DRIVER_PATH),
        "model": MODEL,
        "source_family": SOURCE_FAMILY,
        "scenario": SCENARIO,
        "mean_sha256_float64": array_hash(mean, "<f8"),
        "edges_sha256_float64": array_hash(edges, "<f8"),
        "n_bins": int(len(mean)),
        "expected_truth_histograms": 1,
        "expected_selection_histograms": len(SELECTION_INDICES),
        "expected_confirmation_histograms": len(CONFIRMATION_INDICES),
        "expected_total_histograms": 1 + len(TOY_INDICES),
        "fresh_seed_words_unique": True,
        "fresh_seed_words_disjoint_from_original_manifest": True,
        "root_exists": ROOT_PATH.is_file(),
        "manifest_exists": MANIFEST_PATH.is_file(),
        "products_generated_by_this_command": False,
        "fits_launched": False,
        "claim_boundary": addendum["claim_boundary"],
    }


def build() -> dict[str, Any]:
    preflight_record = preflight()
    if ROOT_PATH.exists() or MANIFEST_PATH.exists():
        raise RemediationToyError(
            "refusing to overwrite an existing remediation ROOT or manifest"
        )
    mean, edges, _ = source_mean()
    ROOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ROOT_PATH.name}.", suffix=".tmp", dir=ROOT_PATH.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    toy_rows: list[dict[str, Any]] = []
    try:
        with uproot.recreate(temporary) as root_file:
            root_file[TRUTH_KEY] = (mean, edges)
            for toy_index in TOY_INDICES:
                stage = stage_for(toy_index)
                words = seed_words(stage, toy_index)
                counts = rng_for(stage, toy_index).poisson(mean).astype(np.int64)
                key = f"pilot/{MODEL}/{SCENARIO}/toy_{toy_index:04d}"
                root_file[key] = (counts, edges)
                toy_rows.append(
                    {
                        "phase": "pilot",
                        "stage": stage,
                        "model": MODEL,
                        "scenario": SCENARIO,
                        "source_family": SOURCE_FAMILY,
                        "toy_index": toy_index,
                        "key": key,
                        "total": int(np.sum(counts, dtype=np.int64)),
                        "counts_sha256_int64": array_hash(counts, "<i8"),
                        "namespace": [
                            SEED_NAMESPACE,
                            MODEL,
                            SOURCE_FAMILY,
                            SCENARIO,
                            stage,
                            toy_index,
                        ],
                        "seed_words_uint32": words,
                    }
                )
        os.replace(temporary, ROOT_PATH)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    manifest = {
        "schema_version": 1,
        "study_id": "v4p8p3_2021_residual_truths_20260814",
        "addendum_id": "v4p8p3_residual_length_ceiling_remediation_v1",
        "claim_boundary": load_addendum()["claim_boundary"],
        "background_only": True,
        "model": MODEL,
        "source_family": SOURCE_FAMILY,
        "scenario": SCENARIO,
        "base_seed": BASE_SEED,
        "seed_namespace": SEED_NAMESPACE,
        "selection_toy_indices": list(SELECTION_INDICES),
        "confirmation_toy_indices": list(CONFIRMATION_INDICES),
        "selection_and_confirmation_streams_distinct": True,
        "independent_of_original_pilot_and_closure_streams": True,
        "reserve_toys": 0,
        "addendum": {
            "path": ADDENDUM_PATH.name,
            "sha256": sha256_file(ADDENDUM_PATH),
        },
        "protocol": {
            "path": models.PROTOCOL_PATH.name,
            "sha256": sha256_file(models.PROTOCOL_PATH),
        },
        "source_fit_and_influence": {
            "path": str(models.FIT_RESULT_PATH.relative_to(HERE)),
            "sha256": sha256_file(models.FIT_RESULT_PATH),
        },
        "original_toy_manifest": {
            "path": str(ORIGINAL_MANIFEST_PATH.relative_to(HERE)),
            "sha256": sha256_file(ORIGINAL_MANIFEST_PATH),
        },
        "builder": {
            "path": DRIVER_PATH.name,
            "sha256": sha256_file(DRIVER_PATH),
        },
        "root": {
            "path": str(ROOT_PATH.relative_to(HERE)),
            "sha256": sha256_file(ROOT_PATH),
            "histograms_expected": 1 + len(TOY_INDICES),
        },
        "edges": {
            "n_bins": int(len(mean)),
            "low_GeV": float(edges[0]),
            "high_GeV": float(edges[-1]),
            "sha256_float64": array_hash(edges, "<f8"),
        },
        "truth": {
            "model": MODEL,
            "scenario": SCENARIO,
            "source_family": SOURCE_FAMILY,
            "key": TRUTH_KEY,
            "total": float(np.sum(mean)),
            "mean_sha256_float64": array_hash(mean, "<f8"),
        },
        "toys": toy_rows,
        "preflight_builder_sha256": preflight_record["builder_sha256"],
    }
    models.atomic_json(MANIFEST_PATH, manifest)
    return validate()


def validate() -> dict[str, Any]:
    preflight_record = preflight()
    if not ROOT_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise RemediationToyError("missing remediation ROOT or manifest")
    manifest = load_json(MANIFEST_PATH)
    if int(manifest.get("schema_version", -1)) != 1:
        raise RemediationToyError("unsupported remediation manifest schema")
    required_equal = {
        "model": MODEL,
        "source_family": SOURCE_FAMILY,
        "scenario": SCENARIO,
        "base_seed": BASE_SEED,
        "seed_namespace": SEED_NAMESPACE,
        "selection_toy_indices": list(SELECTION_INDICES),
        "confirmation_toy_indices": list(CONFIRMATION_INDICES),
        "selection_and_confirmation_streams_distinct": True,
        "independent_of_original_pilot_and_closure_streams": True,
        "reserve_toys": 0,
        "background_only": True,
    }
    for key, expected in required_equal.items():
        if manifest.get(key) != expected:
            raise RemediationToyError(f"manifest {key} drift")
    provenance = {
        "addendum": ADDENDUM_PATH,
        "protocol": models.PROTOCOL_PATH,
        "source_fit_and_influence": models.FIT_RESULT_PATH,
        "original_toy_manifest": ORIGINAL_MANIFEST_PATH,
        "builder": DRIVER_PATH,
        "root": ROOT_PATH,
    }
    for key, path in provenance.items():
        if manifest.get(key, {}).get("sha256") != sha256_file(path):
            raise RemediationToyError(f"manifest provenance hash drift: {key}")
    rows = manifest.get("toys")
    if not isinstance(rows, list) or len(rows) != len(TOY_INDICES):
        raise RemediationToyError("remediation toy-row cardinality drift")
    identities = {(str(row["stage"]), int(row["toy_index"])) for row in rows}
    expected_identities = {
        (stage_for(toy_index), toy_index) for toy_index in TOY_INDICES
    }
    if identities != expected_identities:
        raise RemediationToyError("remediation stage/toy inventory drift")
    for row in rows:
        toy_index = int(row["toy_index"])
        stage = stage_for(toy_index)
        expected_row = {
            "phase": "pilot",
            "stage": stage,
            "model": MODEL,
            "scenario": SCENARIO,
            "source_family": SOURCE_FAMILY,
            "key": f"pilot/{MODEL}/{SCENARIO}/toy_{toy_index:04d}",
            "namespace": [
                SEED_NAMESPACE,
                MODEL,
                SOURCE_FAMILY,
                SCENARIO,
                stage,
                toy_index,
            ],
        }
        for key, expected in expected_row.items():
            if row.get(key) != expected:
                raise RemediationToyError(
                    f"remediation toy-row {toy_index} {key} drift"
                )
    seeds = [tuple(map(int, row["seed_words_uint32"])) for row in rows]
    if len(set(seeds)) != len(seeds) or set(seeds) & original_seed_inventory():
        raise RemediationToyError("remediation seed independence failure")
    mean, expected_edges, _ = source_mean()
    expected_edges_hash = array_hash(expected_edges, "<f8")
    edges_record = manifest.get("edges", {})
    if (
        int(edges_record.get("n_bins", -1)) != len(mean)
        or float(edges_record.get("low_GeV", math.nan)) != float(expected_edges[0])
        or float(edges_record.get("high_GeV", math.nan)) != float(expected_edges[-1])
        or edges_record.get("sha256_float64") != expected_edges_hash
    ):
        raise RemediationToyError("manifest edge contract drift")
    expected_keys = {TRUTH_KEY} | {str(row["key"]) for row in rows}
    with uproot.open(ROOT_PATH) as root_file:
        histogram_keys = {
            key.split(";")[0]
            for key, class_name in root_file.classnames(recursive=True).items()
            if str(class_name).startswith("TH1")
        }
        if histogram_keys != expected_keys:
            raise RemediationToyError("ROOT histogram inventory drift")
        truth_values, truth_edges = root_file[TRUTH_KEY].to_numpy(flow=False)
        if not np.array_equal(truth_edges, expected_edges) or not np.array_equal(
            np.asarray(truth_values, dtype=np.float64), mean
        ):
            raise RemediationToyError("ROOT truth histogram drift")
        truth_record = manifest.get("truth", {})
        if (
            truth_record.get("model") != MODEL
            or truth_record.get("scenario") != SCENARIO
            or truth_record.get("source_family") != SOURCE_FAMILY
            or truth_record.get("key") != TRUTH_KEY
            or not np.isclose(
                float(truth_record.get("total", math.nan)),
                float(np.sum(mean)),
                rtol=0.0,
                atol=1e-9,
            )
            or truth_record.get("mean_sha256_float64") != array_hash(mean, "<f8")
        ):
            raise RemediationToyError("manifest truth hash drift")
        for row in rows:
            toy_index = int(row["toy_index"])
            stage = stage_for(toy_index)
            expected_words = seed_words(stage, toy_index)
            if list(map(int, row["seed_words_uint32"])) != expected_words:
                raise RemediationToyError("manifest deterministic seed drift")
            values, edges = root_file[str(row["key"])].to_numpy(flow=False)
            rounded = np.rint(values).astype(np.int64)
            if (
                not np.array_equal(edges, expected_edges)
                or not np.array_equal(values, rounded.astype(float))
                or np.any(rounded < 0)
            ):
                raise RemediationToyError(f"invalid ROOT toy {row['key']}")
            regenerated = rng_for(stage, toy_index).poisson(mean).astype(np.int64)
            if not np.array_equal(rounded, regenerated):
                raise RemediationToyError(f"nondeterministic ROOT toy {row['key']}")
            if row.get("counts_sha256_int64") != array_hash(rounded, "<i8"):
                raise RemediationToyError(f"toy hash drift {row['key']}")
            if int(row.get("total", -1)) != int(np.sum(rounded, dtype=np.int64)):
                raise RemediationToyError(f"toy total drift {row['key']}")
    return {
        "artifact_integrity_status": "pass",
        "scientific_status": "inputs_only_not_fitted",
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "builder_sha256": sha256_file(DRIVER_PATH),
        "root_sha256": sha256_file(ROOT_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "truth_histograms": 1,
        "selection_histograms": len(SELECTION_INDICES),
        "confirmation_histograms": len(CONFIRMATION_INDICES),
        "total_histograms": len(expected_keys),
        "reserve_histograms": 0,
        "fresh_seed_words_disjoint_from_original_manifest": True,
        "fits_launched": False,
        "claim_boundary": load_addendum()["claim_boundary"],
        "preflight": preflight_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "build", "validate"))
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight()
    elif args.command == "build":
        result = build()
    else:
        result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
