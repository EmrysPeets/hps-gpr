#!/usr/bin/env python3
"""Build the provisional d18 stress-generator means and 25 nested toys.

This command is intentionally unable to promote the generator.  It consumes the
source-GOF candidate recorded by ``fit_qualify.py`` and stamps every product as a
conditional stress screen with ``promotion_gate_passed=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import uproot
from numpy.polynomial.chebyshev import chebvander
from scipy.special import expit


HERE = Path(__file__).resolve().parent
QUALIFICATION = HERE / "derived/generator_qualification.json"
OUTPUT_ROOT = HERE / "inputs/provisional_expcheb18_nested_toys_25.root"
OUTPUT_MANIFEST = HERE / "inputs/provisional_expcheb18_nested_toys_25.manifest.json"
DEGREE = 18
N_TOYS = 25
BASE_SEED = 20260813
SUPPORT_LOW = 0.030
SUPPORT_HIGH = 0.300
REPORTED_SCENARIOS = (
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
ALL_SCENARIOS = ("2021_1pct",) + REPORTED_SCENARIOS
SOURCE_PATHS = {
    "one_pct": Path("/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root"),
    "ten_pct": Path("/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"),
}
HISTOGRAM = "preselection/h_invM_8000"


class BuildError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
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


def stable_seed_words(namespace: str, *parts: object) -> list[int]:
    material = "|".join(
        [str(BASE_SEED), str(namespace), *[str(part) for part in parts]]
    ).encode()
    digest = hashlib.sha256(material).digest()[:16]
    return [
        int.from_bytes(digest[index:index + 4], "little")
        for index in range(0, 16, 4)
    ]


def record_for_source(qualification: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    records = qualification["sources"][source]["records"]
    matches = [record for record in records if int(record["degree"]) == DEGREE]
    if len(matches) != 1:
        raise BuildError(f"expected one d18 record for {source}")
    record = matches[0]
    if float(record.get("gradient_max_abs", math.inf)) > 1e-3:
        raise BuildError(
            f"d18 fit for {source} lacks a stationary/reproducible optimum certificate"
        )
    if not bool(record.get("source_gof_gate_pass")):
        raise BuildError(f"d18 does not pass even the source-GOF gate for {source}")
    if bool(record.get("qualified")):
        raise BuildError("builder is for the provisional stress-only branch")
    return record


def provisional_mean(record: Mapping[str, Any], centers: np.ndarray) -> np.ndarray:
    mapped = 2.0 * (centers - SUPPORT_LOW) / (SUPPORT_HIGH - SUPPORT_LOW) - 1.0
    matrix = chebvander(mapped, DEGREE)
    coefficients = np.asarray(record["coefficients"], dtype=float)
    if coefficients.shape != (DEGREE + 1,):
        raise BuildError("d18 coefficient vector has the wrong shape")
    turn_on = float(record["turn_on_gev"])
    width = float(record["width_gev"])
    log_mean = matrix @ coefficients + np.log(
        np.clip(expit((centers - turn_on) / width), 1e-300, 1.0)
    )
    if np.any(~np.isfinite(log_mean)) or np.any(np.abs(log_mean) >= 700.0):
        raise BuildError("provisional mean requires exponential clipping")
    return np.exp(log_mean)


def source_arrays(source: str, record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = SOURCE_PATHS[source]
    with uproot.open(path) as root_file:
        observed, edges = root_file[HISTOGRAM].to_numpy(flow=False)
    observed = np.asarray(observed, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_support = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
    in_primary = (centers >= 0.040) & (centers < SUPPORT_HIGH)
    mean = np.zeros_like(observed, dtype=float)
    mean[in_support] = provisional_mean(record, centers[in_support])
    primary_observed = float(np.sum(observed[in_primary]))
    primary_unscaled = float(np.sum(mean[in_primary]))
    if not primary_unscaled > 0:
        raise BuildError(f"nonpositive primary mean for {source}")
    scale = primary_observed / primary_unscaled
    mean *= scale
    if np.any(mean[in_support] <= 0) or np.any(mean[~in_support] != 0):
        raise BuildError(f"support/positivity failure for {source}")
    metadata = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "histogram": HISTOGRAM,
        "normalization_rule": "single global scale chosen to match observed 40-300 MeV total",
        "scale": float(scale),
        "observed_total_030_300": float(np.sum(observed[in_support])),
        "observed_total_040_300": primary_observed,
        "mean_total_030_300": float(np.sum(mean[in_support])),
        "mean_total_040_300": float(np.sum(mean[in_primary])),
        "mean_full_sha256_float64": array_hash(mean, "<f8"),
        "mean_ge040_sha256_float64": array_hash(mean[in_primary], "<f8"),
    }
    return mean, edges, metadata


def draw_increment(mean: np.ndarray, source: str, toy: int, stage: str, multiplier: int) -> tuple[np.ndarray, list[int]]:
    words = stable_seed_words("nested_poisson", source, int(toy), stage)
    rng = np.random.default_rng(np.random.SeedSequence(words))
    values = rng.poisson(np.asarray(mean) * int(multiplier)).astype(np.int64)
    return values, words


def scenario_draws(mean: np.ndarray, source: str, toy: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if source == "one_pct":
        one, s1 = draw_increment(mean, source, toy, "base_1x", 1)
        nine, s9 = draw_increment(mean, source, toy, "increment_9x", 9)
        ninety, s90 = draw_increment(mean, source, toy, "increment_90x", 90)
        return (
            {
                "2021_1pct": one,
                "2021_1pct_x10": one + nine,
                "2021_1pct_x100": one + nine + ninety,
            },
            {
                "2021_1pct": (None, 1, one, s1),
                "2021_1pct_x10": ("2021_1pct", 9, nine, s9),
                "2021_1pct_x100": ("2021_1pct_x10", 90, ninety, s90),
            },
        )
    if source == "ten_pct":
        one, s1 = draw_increment(mean, source, toy, "base_1x", 1)
        nine, s9 = draw_increment(mean, source, toy, "increment_9x", 9)
        return (
            {"2021_10pct": one, "2021_10pct_x10": one + nine},
            {
                "2021_10pct": (None, 1, one, s1),
                "2021_10pct_x10": ("2021_10pct", 9, nine, s9),
            },
        )
    raise BuildError(f"unsupported source: {source}")


def validate_product(root_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = {
        (str(row["scenario"]), int(row["toy_index"])): row
        for row in manifest["toys"]
    }
    expected = {(scenario, toy) for scenario in ALL_SCENARIOS for toy in range(N_TOYS)}
    if set(records) != expected:
        raise BuildError("manifest toy inventory mismatch")
    with uproot.open(root_path) as root_file:
        reference_edges = None
        for scenario, toy in sorted(expected):
            key = f"toys/provisional_expcheb18/{scenario}/toy_{toy:04d}"
            values, edges = root_file[key].to_numpy(flow=False)
            values = np.rint(values).astype(np.int64)
            centers = 0.5 * (edges[:-1] + edges[1:])
            in_support = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
            ge040 = (centers >= 0.040) & (centers < SUPPORT_HIGH)
            if np.any(values < 0) or np.any(values[~in_support] != 0):
                raise BuildError(f"invalid counts/support in {key}")
            row = records[(scenario, toy)]
            if array_hash(values, "<i8") != row["counts_sha256_int64"]:
                raise BuildError(f"full count hash mismatch: {key}")
            if array_hash(values[ge040], "<i8") != row["counts_ge040_sha256_int64"]:
                raise BuildError(f">=40 count hash mismatch: {key}")
            if reference_edges is None:
                reference_edges = np.asarray(edges)
            elif not np.array_equal(edges, reference_edges):
                raise BuildError(f"edge mismatch: {key}")
            parent = row["parent_scenario"]
            if parent:
                parent_values = np.rint(
                    root_file[f"toys/provisional_expcheb18/{parent}/toy_{toy:04d}"].values()
                ).astype(np.int64)
                difference = values - parent_values
                if np.any(difference < 0):
                    raise BuildError(f"nested exposure failure: {key}")
                if array_hash(difference, "<i8") != row["increment_sha256_int64"]:
                    raise BuildError(f"increment hash mismatch: {key}")
    return {"status": "pass", "histograms": len(expected), "reported_scenarios": list(REPORTED_SCENARIOS)}


def build(force: bool = False) -> dict[str, Any]:
    if not QUALIFICATION.is_file():
        raise BuildError("run fit_qualify.py before building toys")
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if not bool(qualification.get("optimizer_reproducibility_gate_passed", False)):
        raise BuildError(
            "provisional coefficients are not frozen: optimizer reproducibility gate is false"
        )
    if not bool(qualification.get("conditional_stress_generator_override", False)):
        raise BuildError(
            "no reviewed conditional-stress override is present; qualification is blocked"
        )
    if qualification.get("fully_qualified_common_degrees"):
        raise BuildError("unexpected nominal qualification; use a promotion-reviewed builder")
    if (OUTPUT_ROOT.exists() or OUTPUT_MANIFEST.exists()) and not force:
        raise BuildError("output exists; inspect it or pass --force")

    records = {source: record_for_source(qualification, source) for source in SOURCE_PATHS}
    source_data = {source: source_arrays(source, records[source]) for source in SOURCE_PATHS}
    edges = source_data["one_pct"][1]
    if not np.array_equal(edges, source_data["ten_pct"][1]):
        raise BuildError("source histogram edges do not match")
    multipliers = {
        "2021_1pct": ("one_pct", 1),
        "2021_1pct_x10": ("one_pct", 10),
        "2021_1pct_x100": ("one_pct", 100),
        "2021_10pct": ("ten_pct", 1),
        "2021_10pct_x10": ("ten_pct", 10),
    }

    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.{os.getpid()}.tmp")
    toy_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    try:
        with uproot.recreate(temporary) as root_file:
            for scenario, (source, multiplier) in multipliers.items():
                mean = source_data[source][0] * int(multiplier)
                key = f"truth/provisional_expcheb18/{scenario}_mean"
                root_file[key] = (mean, edges)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ge040 = (centers >= 0.040) & (centers < SUPPORT_HIGH)
                truth_rows.append(
                    {
                        "scenario": scenario,
                        "source_family": source,
                        "multiplier": int(multiplier),
                        "analytic_mean_key": key,
                        "total_030_300": float(np.sum(mean[(centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)])),
                        "total_040_300": float(np.sum(mean[ge040])),
                        "mean_sha256_float64": array_hash(mean, "<f8"),
                        "mean_ge040_sha256_float64": array_hash(mean[ge040], "<f8"),
                    }
                )
            for toy in range(N_TOYS):
                for source in SOURCE_PATHS:
                    base_mean = source_data[source][0]
                    draws, metadata = scenario_draws(base_mean, source, toy)
                    for scenario, values in draws.items():
                        key = f"toys/provisional_expcheb18/{scenario}/toy_{toy:04d}"
                        root_file[key] = (values, edges)
                        parent, increment_multiplier, increment, seed_words = metadata[scenario]
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        ge040 = (centers >= 0.040) & (centers < SUPPORT_HIGH)
                        toy_rows.append(
                            {
                                "scenario": scenario,
                                "source_family": source,
                                "toy_index": toy,
                                "output_histogram": key,
                                "parent_scenario": parent,
                                "increment_multiplier": increment_multiplier,
                                "increment_seed_words": seed_words,
                                "increment_sha256_int64": array_hash(increment, "<i8"),
                                "counts_sha256_int64": array_hash(values, "<i8"),
                                "counts_ge040_sha256_int64": array_hash(values[ge040], "<i8"),
                                "total_030_300": int(np.sum(values[(centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)])),
                                "total_040_300": int(np.sum(values[ge040])),
                            }
                        )
        os.replace(temporary, OUTPUT_ROOT)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "generator": "provisional degree-18 positive log-Chebyshev source-conditioned stress generator",
        "generator_tag": "provisional_expcheb18",
        "qualification_sha256": sha256_file(QUALIFICATION),
        "promotion_gate_passed": False,
        "promotion_blocker": "no common degree passed blocked-CV and fake-gap projection gates",
        "support_gev": [SUPPORT_LOW, SUPPORT_HIGH],
        "primary_support_gev": [0.040, SUPPORT_HIGH],
        "base_seed": BASE_SEED,
        "n_toys_per_source_family": N_TOYS,
        "all_scenarios": list(ALL_SCENARIOS),
        "reported_scenarios": list(REPORTED_SCENARIOS),
        "source_fits": {source: source_data[source][2] for source in SOURCE_PATHS},
        "truths": truth_rows,
        "toys": toy_rows,
        "nesting": "independent Poisson increments within source; source families independent",
        "interpretation": "conditional stress-generator screen, not physical truth or coverage",
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    atomic_json(OUTPUT_MANIFEST, payload)
    validation = validate_product(OUTPUT_ROOT, payload)
    result = {
        "status": "pass_conditional_stress_only",
        "promotion_gate_passed": False,
        "root": str(OUTPUT_ROOT),
        "root_sha256": sha256_file(OUTPUT_ROOT),
        "manifest": str(OUTPUT_MANIFEST),
        "manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "validation": validation,
    }
    atomic_json(HERE / "derived/stress_toy_build_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build(force=bool(args.force))
    else:
        payload = json.loads(OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        result = validate_product(OUTPUT_ROOT, payload)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
