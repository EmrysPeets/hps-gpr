#!/usr/bin/env python3
"""Build and validate the exact v4p8p3 nested Poisson backgrounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import uproot

import residual_models as models


HERE = Path(__file__).resolve().parent
ROOT_PATH = HERE / "inputs/residual_structured_nested_toys.root"
MANIFEST_PATH = HERE / "inputs/residual_structured_nested_toys.manifest.json"
MODELS = ("knot_spline", "regional_blend")
SCENARIOS = (
    "2021_1pct",
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
POLICY = {
    "2021_1pct": ("one_pct", 1, None, 1),
    "2021_1pct_x10": ("one_pct", 10, "2021_1pct", 9),
    "2021_1pct_x100": ("one_pct", 100, "2021_1pct_x10", 90),
    "2021_10pct": ("ten_pct", 1, None, 1),
    "2021_10pct_x10": ("ten_pct", 10, "2021_10pct", 9),
}
PHASE_COUNTS = {"toys": 20, "pilot": 3}
BASE_SEED = 20260814


class ToyBuildError(RuntimeError):
    """Raised when the frozen nested-toy contract is violated."""


def sha256_file(path: Path) -> str:
    return models.sha256_file(path)


def array_hash(values: Any, dtype: str) -> str:
    return models.array_hash(values, dtype)


def seed_words(*parts: object) -> list[int]:
    material = "|".join([str(BASE_SEED), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()


def rng_for(*parts: object) -> tuple[np.random.Generator, list[int]]:
    words = seed_words(*parts)
    return np.random.default_rng(np.random.SeedSequence(words)), words


def full_means(result: Mapping[str, Any]) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    output: dict[tuple[str, str], np.ndarray] = {}
    common_edges = None
    for model in MODELS:
        for source in ("one_pct", "ten_pct"):
            mean, edges = models.frozen_mean_full(model, source, result)
            if common_edges is None:
                common_edges = np.asarray(edges, dtype=float)
            elif not np.array_equal(common_edges, edges):
                raise ToyBuildError("source edge grids differ")
            if np.any(mean < 0) or np.any(~np.isfinite(mean)):
                raise ToyBuildError(f"invalid frozen mean for {model}/{source}")
            output[(model, source)] = np.asarray(mean, dtype=float)
    if common_edges is None:
        raise ToyBuildError("no frozen means")
    return output, common_edges


def generate_phase(
    model: str,
    phase: str,
    toy_index: int,
    means: Mapping[tuple[str, str], np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if phase not in PHASE_COUNTS:
        raise ToyBuildError(f"unknown phase {phase}")
    arrays: dict[str, np.ndarray] = {}
    seeds: dict[str, Any] = {}
    for source in ("one_pct", "ten_pct"):
        base_mean = means[(model, source)]
        parent_name = "2021_1pct" if source == "one_pct" else "2021_10pct"
        parent_rng, parent_words = rng_for(model, phase, source, toy_index, "parent")
        parent = parent_rng.poisson(base_mean).astype(np.int64)
        arrays[parent_name] = parent
        seeds[parent_name] = {
            "namespace": [model, phase, source, int(toy_index), "parent"],
            "seed_words_uint32": parent_words,
        }
        increment_rng, increment_words = rng_for(
            model, phase, source, toy_index, "increment_x9"
        )
        increment9 = increment_rng.poisson(9.0 * base_mean).astype(np.int64)
        x10_name = "2021_1pct_x10" if source == "one_pct" else "2021_10pct_x10"
        arrays[x10_name] = parent + increment9
        seeds[x10_name] = {
            "namespace": [model, phase, source, int(toy_index), "increment_x9"],
            "seed_words_uint32": increment_words,
            "increment_sha256_int64": array_hash(increment9, "<i8"),
        }
        if source == "one_pct":
            increment_rng, increment_words = rng_for(
                model, phase, source, toy_index, "increment_x90"
            )
            increment90 = increment_rng.poisson(90.0 * base_mean).astype(np.int64)
            arrays["2021_1pct_x100"] = arrays["2021_1pct_x10"] + increment90
            seeds["2021_1pct_x100"] = {
                "namespace": [model, phase, source, int(toy_index), "increment_x90"],
                "seed_words_uint32": increment_words,
                "increment_sha256_int64": array_hash(increment90, "<i8"),
            }
    if tuple(arrays) != SCENARIOS:
        raise ToyBuildError(f"scenario generation order/content drift: {tuple(arrays)}")
    if np.any(arrays["2021_1pct_x10"] < arrays["2021_1pct"]):
        raise ToyBuildError("negative 1pct x9 increment")
    if np.any(arrays["2021_1pct_x100"] < arrays["2021_1pct_x10"]):
        raise ToyBuildError("negative 1pct x90 increment")
    if np.any(arrays["2021_10pct_x10"] < arrays["2021_10pct"]):
        raise ToyBuildError("negative 10pct x9 increment")
    return arrays, seeds


def build() -> dict[str, Any]:
    protocol = models.protocol()
    result = models.load_fit_result(require_influence=True)
    for model in MODELS:
        if result["models"][model]["strict_generator_qualification_passed"]:
            raise ToyBuildError("protocol disposition unexpectedly promotes a model")
        if result["models"][model]["promotion_scope"] != "requested conditional stress only":
            raise ToyBuildError("conditional-only disposition drift")
    if tuple(protocol["toy_contract"]["reported_scenarios"]) != SCENARIOS:
        raise ToyBuildError("protocol scenario order drift")
    if int(protocol["toy_contract"]["closure_backgrounds_per_model_source"]) != 20:
        raise ToyBuildError("closure count drift")
    if int(protocol["toy_contract"]["pilot_backgrounds_per_model_source"]) != 3:
        raise ToyBuildError("pilot count drift")

    means, edges = full_means(result)
    ROOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{ROOT_PATH.name}.", suffix=".tmp", dir=ROOT_PATH.parent
    )
    os.close(handle)
    Path(temporary_name).unlink()
    temporary = Path(temporary_name)
    truth_rows = []
    toy_rows = []
    try:
        with uproot.recreate(temporary) as root_file:
            for model in MODELS:
                for scenario in SCENARIOS:
                    source, multiplier, parent, increment_multiplier = POLICY[scenario]
                    values = float(multiplier) * means[(model, source)]
                    key = f"truth/{model}/{scenario}_mean"
                    root_file[key] = (values, edges)
                    truth_rows.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "source_family": source,
                            "multiplier": int(multiplier),
                            "parent_scenario": parent,
                            "increment_multiplier": int(increment_multiplier),
                            "key": key,
                            "total": float(np.sum(values)),
                            "mean_sha256_float64": array_hash(values, "<f8"),
                        }
                    )
                for phase, count in PHASE_COUNTS.items():
                    for toy_index in range(count):
                        arrays, seeds = generate_phase(
                            model, phase, toy_index, means
                        )
                        for scenario in SCENARIOS:
                            source, multiplier, parent, increment_multiplier = POLICY[
                                scenario
                            ]
                            values = arrays[scenario]
                            key = f"{phase}/{model}/{scenario}/toy_{toy_index:04d}"
                            root_file[key] = (values, edges)
                            toy_rows.append(
                                {
                                    "phase": phase,
                                    "model": model,
                                    "scenario": scenario,
                                    "source_family": source,
                                    "multiplier": int(multiplier),
                                    "parent_scenario": parent,
                                    "increment_multiplier": int(increment_multiplier),
                                    "toy_index": int(toy_index),
                                    "key": key,
                                    "total": int(np.sum(values, dtype=np.int64)),
                                    "counts_sha256_int64": array_hash(values, "<i8"),
                                    **seeds[scenario],
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
        "study_id": protocol["study_id"],
        "claim_boundary": protocol["claim_boundary"],
        "promotion_scope": "requested conditional stress only",
        "base_seed": BASE_SEED,
        "models": list(MODELS),
        "reported_scenarios": list(SCENARIOS),
        "phase_counts_per_model_scenario": PHASE_COUNTS,
        "closure_background_clusters": 80,
        "reserve_backgrounds": 0,
        "nested_poisson_within_source_family": True,
        "source_families_distinct": True,
        "model_streams_distinct": True,
        "pilot_and_closure_streams_independent": True,
        "protocol": {
            "path": models.PROTOCOL_PATH.name,
            "sha256": sha256_file(models.PROTOCOL_PATH),
        },
        "source_fit_and_influence": {
            "path": str(models.FIT_RESULT_PATH.relative_to(HERE)),
            "sha256": sha256_file(models.FIT_RESULT_PATH),
        },
        "builder": {
            "path": Path(__file__).name,
            "sha256": sha256_file(Path(__file__)),
        },
        "root": {
            "path": str(ROOT_PATH.relative_to(HERE)),
            "sha256": sha256_file(ROOT_PATH),
            "histograms_expected": len(truth_rows) + len(toy_rows),
        },
        "edges": {
            "n_bins": int(edges.size - 1),
            "low_GeV": float(edges[0]),
            "high_GeV": float(edges[-1]),
            "sha256_float64": array_hash(edges, "<f8"),
        },
        "truths": truth_rows,
        "toys": toy_rows,
    }
    models.atomic_json(MANIFEST_PATH, manifest)
    return validate()


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ToyBuildError(f"{message}: {actual!r} != {expected!r}")


def validate() -> dict[str, Any]:
    result = models.load_fit_result(require_influence=True)
    if not ROOT_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise ToyBuildError("missing ROOT or manifest product")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _assert_equal(manifest.get("schema_version"), 1, "manifest schema")
    _assert_equal(tuple(manifest.get("models", ())), MODELS, "model inventory")
    _assert_equal(
        tuple(manifest.get("reported_scenarios", ())), SCENARIOS, "scenario inventory"
    )
    _assert_equal(
        manifest["protocol"]["sha256"], sha256_file(models.PROTOCOL_PATH), "protocol hash"
    )
    _assert_equal(
        manifest["source_fit_and_influence"]["sha256"],
        sha256_file(models.FIT_RESULT_PATH),
        "source result hash",
    )
    _assert_equal(
        manifest["builder"]["sha256"], sha256_file(Path(__file__)), "builder hash"
    )
    _assert_equal(manifest["root"]["sha256"], sha256_file(ROOT_PATH), "ROOT hash")
    _assert_equal(manifest.get("reserve_backgrounds"), 0, "reserve count")
    truth_rows = manifest.get("truths", [])
    toy_rows = manifest.get("toys", [])
    _assert_equal(len(truth_rows), 2 * 5, "truth row count")
    _assert_equal(len(toy_rows), 2 * 5 * (20 + 3), "toy row count")
    by_key = {row["key"]: row for row in toy_rows}
    if len(by_key) != len(toy_rows):
        raise ToyBuildError("duplicate toy manifest keys")
    means, expected_edges = full_means(result)
    with uproot.open(ROOT_PATH) as root_file:
        all_keys = {key.split(";")[0] for key in root_file.keys(recursive=True)}
        leaf_keys = {row["key"] for row in truth_rows} | set(by_key)
        missing = sorted(leaf_keys - all_keys)
        if missing:
            raise ToyBuildError(f"missing ROOT histogram keys: {missing[:5]}")
        for row in truth_rows:
            values, edges = root_file[row["key"]].to_numpy(flow=False)
            if not np.array_equal(edges, expected_edges):
                raise ToyBuildError(f"truth edge drift: {row['key']}")
            expected = float(row["multiplier"]) * means[
                (row["model"], row["source_family"])
            ]
            if not np.array_equal(np.asarray(values, dtype=float), expected):
                raise ToyBuildError(f"truth content drift: {row['key']}")
            _assert_equal(
                array_hash(values, "<f8"), row["mean_sha256_float64"], "truth hash"
            )
        cache: dict[tuple[str, str, int, str], np.ndarray] = {}
        for row in toy_rows:
            values, edges = root_file[row["key"]].to_numpy(flow=False)
            if not np.array_equal(edges, expected_edges):
                raise ToyBuildError(f"toy edge drift: {row['key']}")
            rounded = np.rint(values).astype(np.int64)
            if not np.array_equal(values, rounded.astype(float)) or np.any(rounded < 0):
                raise ToyBuildError(f"noninteger/negative toy: {row['key']}")
            _assert_equal(
                array_hash(rounded, "<i8"), row["counts_sha256_int64"], "toy hash"
            )
            _assert_equal(int(np.sum(rounded, dtype=np.int64)), row["total"], "toy total")
            cache[(row["phase"], row["model"], int(row["toy_index"]), row["scenario"])] = rounded
        for phase, count in PHASE_COUNTS.items():
            for model in MODELS:
                for toy_index in range(count):
                    one = cache[(phase, model, toy_index, "2021_1pct")]
                    one10 = cache[(phase, model, toy_index, "2021_1pct_x10")]
                    one100 = cache[(phase, model, toy_index, "2021_1pct_x100")]
                    ten = cache[(phase, model, toy_index, "2021_10pct")]
                    ten10 = cache[(phase, model, toy_index, "2021_10pct_x10")]
                    if np.any(one10 - one < 0) or np.any(one100 - one10 < 0):
                        raise ToyBuildError("1pct nesting failure")
                    if np.any(ten10 - ten < 0):
                        raise ToyBuildError("10pct nesting failure")
    summary = {
        "artifact_integrity_status": "pass",
        "scientific_qualification": {model: False for model in MODELS},
        "promotion_scope": "requested conditional stress only",
        "root_sha256": sha256_file(ROOT_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "truth_histograms": len(truth_rows),
        "closure_histograms": 2 * 5 * 20,
        "pilot_histograms": 2 * 5 * 3,
        "reserve_histograms": 0,
        "closure_background_clusters": 80,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    summary = build() if args.command == "build" else validate()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
