#!/usr/bin/env python3
"""Build and validate 25 nested toys from the frozen sparse rigid generator.

The native-1pct shape is frozen by ``rigid_generator_spec.json``.  Native 10pct
uses the identical shape with the measured 40--300 MeV total-count ratio.  No
shape coordinate is fitted to native 10pct or to any exposure lane here.
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
from scipy.special import expit


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "rigid_generator_spec.json"
OUTPUT_ROOT = HERE / "inputs/rigid_ggt26_scaled1pct_nested_toys_25.root"
OUTPUT_MANIFEST = HERE / "inputs/rigid_ggt26_scaled1pct_nested_toys_25.manifest.json"
OUTPUT_SUMMARY = HERE / "derived/rigid_toy_build_summary.json"
SOURCE_PATHS = {
    "one_pct": Path("/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root"),
    "ten_pct": Path("/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"),
}
HISTOGRAM = "preselection/h_invM_8000"
SUPPORT_LOW = 0.040
SUPPORT_HIGH = 0.300
N_TOYS = 25
BASE_SEED = 20260813
REPORTED_SCENARIOS = (
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
ALL_SCENARIOS = ("2021_1pct",) + REPORTED_SCENARIOS
SIGMA_COEFFS = (0.00184825, -0.001375, 0.085875)
CHECK_MASSES = tuple([0.050 + 0.020 * i for i in range(11)] + [0.065, 0.090, 0.120, 0.180, 0.210])


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
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def sigma_2021(mass: float) -> float:
    return float(sum(value * mass**power for power, value in enumerate(SIGMA_COEFFS)))


def stable_seed_words(namespace: str, *parts: object) -> list[int]:
    material = "|".join([str(BASE_SEED), namespace, *[str(part) for part in parts]]).encode()
    digest = hashlib.sha256(material).digest()[:16]
    return [int.from_bytes(digest[index:index + 4], "little") for index in range(0, 16, 4)]


def load_histogram(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as root_file:
        values, edges = root_file[HISTOGRAM].to_numpy(flow=False)
    return np.asarray(values, dtype=float), np.asarray(edges, dtype=float)


def chebyshev_t2(u: np.ndarray) -> np.ndarray:
    return 2.0 * u * u - 1.0


def chebyshev_t6(u: np.ndarray) -> np.ndarray:
    u2 = u * u
    return 32.0 * u2 * u2 * u2 - 48.0 * u2 * u2 + 18.0 * u2 - 1.0


def frozen_shape(centers: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    dev = spec["development_constants"]
    pars = spec["one_pct_shape_parameters"]
    x0 = float(dev["x0_gev"])
    xt = float(dev["xt_gev"])
    width = float(dev["w_gev"])
    a = float(pars["a"])
    lam = float(pars["lambda_gev"])
    power = float(pars["power"])
    d2 = float(pars["d2"])
    d6 = float(pars["d6"])
    centers = np.asarray(centers, dtype=float)
    z = centers - x0
    if np.any(z <= 0.0) or width <= 0.0 or lam <= 0.0 or power <= 0.0:
        raise BuildError("invalid frozen shape domain or parameter")
    u = 2.0 * (centers - SUPPORT_LOW) / (SUPPORT_HIGH - SUPPORT_LOW) - 1.0
    log_shape = (
        np.log(np.clip(expit((centers - xt) / width), 1e-300, 1.0))
        + a * np.log(z)
        - np.power(z / lam, power)
        + d2 * chebyshev_t2(u)
        + d6 * chebyshev_t6(u)
    )
    if np.any(~np.isfinite(log_shape)) or np.max(np.abs(log_shape)) >= 700.0:
        raise BuildError("frozen shape is nonfinite or requires exponential clipping")
    shape = np.exp(log_shape - np.max(log_shape))
    if np.any(shape <= 0.0) or np.any(~np.isfinite(shape)):
        raise BuildError("frozen shape is not finite and strictly positive")
    return shape


def source_means(spec: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    one_observed, one_edges = load_histogram(SOURCE_PATHS["one_pct"])
    ten_observed, ten_edges = load_histogram(SOURCE_PATHS["ten_pct"])
    if not np.array_equal(one_edges, ten_edges):
        raise BuildError("source histogram edges differ")
    centers = 0.5 * (one_edges[:-1] + one_edges[1:])
    support = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
    shape = frozen_shape(centers[support], spec)
    one_total = float(np.sum(one_observed[support]))
    ten_total = float(np.sum(ten_observed[support]))
    if one_total <= 0.0 or ten_total <= 0.0:
        raise BuildError("source support total is nonpositive")
    one_mean = np.zeros_like(one_observed, dtype=float)
    one_mean[support] = shape * one_total / float(np.sum(shape))
    ten_mean = one_mean * (ten_total / one_total)
    if not math.isclose(float(np.sum(one_mean[support])), one_total, rel_tol=0.0, abs_tol=1e-6):
        raise BuildError("1pct normalization mismatch")
    if not math.isclose(float(np.sum(ten_mean[support])), ten_total, rel_tol=0.0, abs_tol=1e-5):
        raise BuildError("10pct normalization mismatch")
    metadata = {
        "histogram": HISTOGRAM,
        "shape_source": "native 1pct only",
        "native_10pct_shape_refit": False,
        "normalization_rule": "match each source's observed 40-300 MeV support total",
        "one_pct_source_path": str(SOURCE_PATHS["one_pct"]),
        "one_pct_source_sha256": sha256_file(SOURCE_PATHS["one_pct"]),
        "ten_pct_source_path": str(SOURCE_PATHS["ten_pct"]),
        "ten_pct_source_sha256": sha256_file(SOURCE_PATHS["ten_pct"]),
        "one_pct_total_040_300": one_total,
        "ten_pct_total_040_300": ten_total,
        "ten_to_one_normalization_ratio": ten_total / one_total,
        "one_pct_mean_sha256_float64": array_hash(one_mean, "<f8"),
        "ten_pct_mean_sha256_float64": array_hash(ten_mean, "<f8"),
    }
    return {"one_pct": one_mean, "ten_pct": ten_mean}, one_edges, metadata


def draw_increment(mean: np.ndarray, source: str, toy: int, stage: str, multiplier: int) -> tuple[np.ndarray, list[int]]:
    words = stable_seed_words("nested_poisson", source, int(toy), stage)
    rng = np.random.default_rng(np.random.SeedSequence(words))
    return rng.poisson(np.asarray(mean) * int(multiplier)).astype(np.int64), words


def scenario_draws(mean: np.ndarray, source: str, toy: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base, seed_base = draw_increment(mean, source, toy, "base_1x", 1)
    nine, seed_nine = draw_increment(mean, source, toy, "increment_9x", 9)
    if source == "one_pct":
        ninety, seed_ninety = draw_increment(mean, source, toy, "increment_90x", 90)
        return (
            {
                "2021_1pct": base,
                "2021_1pct_x10": base + nine,
                "2021_1pct_x100": base + nine + ninety,
            },
            {
                "2021_1pct": (None, 1, base, seed_base),
                "2021_1pct_x10": ("2021_1pct", 9, nine, seed_nine),
                "2021_1pct_x100": ("2021_1pct_x10", 90, ninety, seed_ninety),
            },
        )
    if source == "ten_pct":
        return (
            {"2021_10pct": base, "2021_10pct_x10": base + nine},
            {
                "2021_10pct": (None, 1, base, seed_base),
                "2021_10pct_x10": ("2021_10pct", 9, nine, seed_nine),
            },
        )
    raise BuildError(f"unsupported source: {source}")


def validate_training_positivity(values: np.ndarray, edges: np.ndarray, scenario: str, toy: int) -> None:
    if scenario not in REPORTED_SCENARIOS:
        return
    usable = values.size // 5 * 5
    rebinned = values[:usable].reshape(-1, 5).sum(axis=1)
    rebinned_edges = edges[:usable + 1:5]
    centers = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    in_support = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
    if np.any(rebinned[in_support] <= 0):
        raise BuildError(f"nonpositive rebin5 support count for {scenario} toy {toy}")
    for mass in CHECK_MASSES:
        mask = np.abs(centers - mass) >= 2.25 * sigma_2021(mass)
        training = in_support & mask
        if not np.any(training) or np.any(rebinned[training] <= 0):
            raise BuildError(f"invalid pre-log training geometry for {scenario} toy {toy} mass {mass}")


def validate_product(root_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = {(str(row["scenario"]), int(row["toy_index"])): row for row in manifest["toys"]}
    expected = {(scenario, toy) for scenario in ALL_SCENARIOS for toy in range(N_TOYS)}
    if set(records) != expected:
        raise BuildError("manifest inventory mismatch")
    with uproot.open(root_path) as root_file:
        reference_edges = None
        for scenario, toy in sorted(expected):
            key = f"toys/rigid_ggt26_scaled1pct/{scenario}/toy_{toy:04d}"
            raw_values, edges = root_file[key].to_numpy(flow=False)
            if np.any(~np.isfinite(raw_values)) or np.any(raw_values != np.rint(raw_values)):
                raise BuildError(f"noninteger ROOT content in {key}")
            values = raw_values.astype(np.int64)
            centers = 0.5 * (edges[:-1] + edges[1:])
            support = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
            if np.any(values < 0) or np.any(values[~support] != 0):
                raise BuildError(f"invalid count/support content in {key}")
            row = records[(scenario, toy)]
            if array_hash(values, "<i8") != row["counts_sha256_int64"]:
                raise BuildError(f"count hash mismatch in {key}")
            if int(np.sum(values[support])) != int(row["total_040_300"]):
                raise BuildError(f"count total mismatch in {key}")
            validate_training_positivity(values, np.asarray(edges), scenario, toy)
            parent = row["parent_scenario"]
            if parent:
                parent_key = f"toys/rigid_ggt26_scaled1pct/{parent}/toy_{toy:04d}"
                parent_values = np.asarray(root_file[parent_key].values(), dtype=np.int64)
                increment = values - parent_values
                if np.any(increment < 0):
                    raise BuildError(f"nesting failure in {key}")
                if array_hash(increment, "<i8") != row["increment_sha256_int64"]:
                    raise BuildError(f"increment hash mismatch in {key}")
            if reference_edges is None:
                reference_edges = np.asarray(edges)
            elif not np.array_equal(edges, reference_edges):
                raise BuildError(f"edge mismatch in {key}")
    return {
        "status": "pass",
        "histograms": len(expected),
        "reported_scenarios": list(REPORTED_SCENARIOS),
        "rebin5_prelog_training_geometry_checked": True,
    }


def build(force: bool = False) -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("status") != "reviewed_conditional_stress_generator":
        raise BuildError("rigid generator lacks reviewed conditional-stress status")
    if spec.get("support30_status") != "rejected_for_this_family":
        raise BuildError("support30 disposition is not fail-closed")
    if bool(spec.get("kernel_ceiling_selection_allowed")):
        raise BuildError("conditional generator cannot select a kernel ceiling")
    if int(spec["confidence_contract"]["confidence_level_percent"]) != 90:
        raise BuildError("inference contract is not 90% CLs")
    if (OUTPUT_ROOT.exists() or OUTPUT_MANIFEST.exists()) and not force:
        raise BuildError("output exists; inspect it or pass --force")
    means, edges, source_metadata = source_means(spec)
    multipliers = {
        "2021_1pct": ("one_pct", 1),
        "2021_1pct_x10": ("one_pct", 10),
        "2021_1pct_x100": ("one_pct", 100),
        "2021_10pct": ("ten_pct", 1),
        "2021_10pct_x10": ("ten_pct", 10),
    }
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.{os.getpid()}.tmp")
    truth_rows: list[dict[str, Any]] = []
    toy_rows: list[dict[str, Any]] = []
    try:
        with uproot.recreate(temporary) as root_file:
            centers = 0.5 * (edges[:-1] + edges[1:])
            support = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
            for scenario, (source, multiplier) in multipliers.items():
                mean = means[source] * int(multiplier)
                key = f"truth/rigid_ggt26_scaled1pct/{scenario}_mean"
                root_file[key] = (mean, edges)
                truth_rows.append({
                    "scenario": scenario,
                    "source_family": source,
                    "multiplier": int(multiplier),
                    "analytic_mean_key": key,
                    "total_040_300": float(np.sum(mean[support])),
                    "mean_sha256_float64": array_hash(mean, "<f8"),
                })
            for toy in range(N_TOYS):
                for source in ("one_pct", "ten_pct"):
                    draws, metadata = scenario_draws(means[source], source, toy)
                    for scenario, values in draws.items():
                        key = f"toys/rigid_ggt26_scaled1pct/{scenario}/toy_{toy:04d}"
                        root_file[key] = (values, edges)
                        parent, increment_multiplier, increment, seed_words = metadata[scenario]
                        toy_rows.append({
                            "scenario": scenario,
                            "source_family": source,
                            "toy_index": toy,
                            "output_histogram": key,
                            "parent_scenario": parent,
                            "increment_multiplier": int(increment_multiplier),
                            "increment_seed_words": seed_words,
                            "increment_sha256_int64": array_hash(increment, "<i8"),
                            "counts_sha256_int64": array_hash(values, "<i8"),
                            "total_040_300": int(np.sum(values[support])),
                        })
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
        "generator": spec["formula"],
        "generator_tag": spec["generator_tag"],
        "generator_spec": str(SPEC_PATH),
        "generator_spec_sha256": sha256_file(SPEC_PATH),
        "promotion_gate_passed": False,
        "promotion_scope": spec["promotion_scope"],
        "support_gev": [SUPPORT_LOW, SUPPORT_HIGH],
        "base_seed": BASE_SEED,
        "n_toys_per_source_family": N_TOYS,
        "toy_key_template": "toys/rigid_ggt26_scaled1pct/{scenario}/toy_{toy_index:04d}",
        "all_scenarios": list(ALL_SCENARIOS),
        "reported_scenarios": list(REPORTED_SCENARIOS),
        "source_policy": source_metadata,
        "truths": truth_rows,
        "toys": toy_rows,
        "nesting": "independent Poisson increments within source family; source families independent",
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
    atomic_json(OUTPUT_SUMMARY, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build(force=bool(args.force))
    else:
        manifest = json.loads(OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        result = validate_product(OUTPUT_ROOT, manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
