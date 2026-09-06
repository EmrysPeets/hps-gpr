#!/usr/bin/env python3
"""Bitwise and timing closure for the v4.9.7 cached profile solver."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


for _thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4-cached-profile-mpl")

import numpy as np


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CAMPAIGN_DIR))

from runtime_guard import (  # noqa: E402
    activate_and_verify,
    assert_import_origins,
)

RUNTIME_PROVENANCE = activate_and_verify()

from cached_profile_solver import (  # noqa: E402
    CACHE_ALGORITHM_VERSION,
    CachedAsymptoticCombinedLimit,
)
from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import (  # noqa: E402
    build_combined_components,
    combined_cls_limit_epsilon2_from_vectors,
)
from hps_gpr.statistics import draw_bkg_mvn_nonneg  # noqa: E402
from run_combined_bands_cached_fixed_reviewed import (  # noqa: E402
    N_FULL_GRID_MASSES,
    N_TOYS_PER_MASS,
    SEED,
    build_fixed_predictions,
    global_seed_index,
    load_reviewed_coordinates,
    load_support_freeze,
    sha256,
    validate_config_provenance,
    validate_input_hashes,
    validate_reviewed_provenance,
    validate_v4_geometry,
)

RUNTIME_IMPORT_ORIGINS = assert_import_origins(
    (
        "hps_gpr",
        "hps_gpr.config",
        "hps_gpr.conversion",
        "hps_gpr.dataset",
        "hps_gpr.evaluation",
        "hps_gpr.gpr",
        "hps_gpr.io",
        "hps_gpr.statistics",
    )
)


DEFAULT_MASSES_MEV = (20, 40, 60, 100, 200)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-provenance-json", type=Path, required=True)
    parser.add_argument("--reviewed-state-csv", type=Path, required=True)
    parser.add_argument(
        "--reviewed-state-provenance-json", type=Path, required=True
    )
    parser.add_argument("--support-freeze-json", type=Path, required=True)
    parser.add_argument("--support-2016-low-mev", type=int, required=True)
    parser.add_argument("--support-2016-high-mev", type=int, required=True)
    parser.add_argument("--toys-per-mass", type=int, default=20)
    parser.add_argument(
        "--mass-mev",
        type=int,
        action="append",
        help=(
            "Benchmark mass in MeV. Defaults to 20, 40, 60, 100, and 200 "
            "to cover both one-dataset regions, both two-dataset overlaps, "
            "and the three-dataset overlap."
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def benchmark_mass(
    mass: float,
    n_toys: int,
    datasets: dict,
    config,
    fixed_here: dict,
) -> dict:
    datasets_here, predictions, _ = build_fixed_predictions(
        mass,
        datasets,
        config,
        fixed_here,
    )
    observed, b_mean, b_cov, s_unit = build_combined_components(
        mass,
        datasets_here,
        predictions,
        config=config,
    )
    solver = CachedAsymptoticCombinedLimit(
        b_mean,
        b_cov,
        s_unit,
        alpha=float(config.cls_alpha),
        combined_mode=str(config.combined_mode),
    )

    index = global_seed_index(mass)
    child = np.random.SeedSequence(SEED).spawn(N_FULL_GRID_MASSES)[index]
    rng = np.random.default_rng(child)
    rng.integers(1, 2**31 - 1)
    lambda_draws = [
        draw_bkg_mvn_nonneg(
            prediction.mu,
            prediction.cov,
            n_toys,
            rng,
            method=str(
                getattr(config, "mvn_trunc_method", "reject_then_clip")
            ),
            max_tries=int(getattr(config, "mvn_trunc_max_tries", 80)),
        )
        for prediction in predictions
    ]
    count_draws = [rng.poisson(values).astype(int) for values in lambda_draws]
    toy_vectors = [
        np.concatenate([draws[index] for draws in count_draws])
        for index in range(n_toys)
    ]
    for _ in toy_vectors:
        rng.integers(1, 2**31 - 1)
    vectors = [observed] + toy_vectors

    started = time.perf_counter()
    reference = np.asarray(
        [
            combined_cls_limit_epsilon2_from_vectors(
                counts,
                b_mean,
                b_cov,
                s_unit,
                config,
                mode="asymptotic",
            )
            for counts in vectors
        ],
        dtype=float,
    )
    reference_seconds = time.perf_counter() - started

    started = time.perf_counter()
    cached = np.asarray(
        [solver.limit(counts) for counts in vectors],
        dtype=float,
    )
    cached_seconds = time.perf_counter() - started

    bitwise_equal = bool(np.array_equal(reference, cached, equal_nan=True))
    finite_pairs = np.isfinite(reference) & np.isfinite(cached)
    max_absolute_difference = (
        float(np.max(np.abs(reference[finite_pairs] - cached[finite_pairs])))
        if bool(np.any(finite_pairs))
        else float("nan")
    )
    return {
        "mass_GeV": float(mass),
        "seed_sequence_index": int(index),
        "active_datasets": [dataset.key for dataset in datasets_here],
        "n_active_datasets": len(datasets_here),
        "n_vectors": len(vectors),
        "n_pseudoexperiments": n_toys,
        "bitwise_equal": bitwise_equal,
        "max_absolute_difference": max_absolute_difference,
        "reference_seconds": float(reference_seconds),
        "cached_seconds": float(cached_seconds),
        "speedup": (
            float(reference_seconds / cached_seconds)
            if cached_seconds > 0.0
            else float("inf")
        ),
        "asimov_fixed_cache_nodes": solver.asimov_fixed_cache_size,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.toys_per_mass < 1 or args.toys_per_mass > N_TOYS_PER_MASS:
        raise SystemExit(
            f"--toys-per-mass must be between 1 and {N_TOYS_PER_MASS}."
        )

    config_path = args.config.expanduser().resolve()
    config_provenance_path = (
        args.config_provenance_json.expanduser().resolve()
    )
    reviewed_csv = args.reviewed_state_csv.expanduser().resolve()
    reviewed_provenance_path = (
        args.reviewed_state_provenance_json.expanduser().resolve()
    )
    support_freeze = args.support_freeze_json.expanduser().resolve()
    output = args.json_out.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite closure report: {output}")
    for path in (
        config_path,
        config_provenance_path,
        reviewed_csv,
        reviewed_provenance_path,
        support_freeze,
    ):
        if not path.is_file():
            raise SystemExit(f"Required file does not exist: {path}")
    masses_mev = (
        sorted(set(args.mass_mev))
        if args.mass_mev
        else list(DEFAULT_MASSES_MEV)
    )
    masses = [value / 1000.0 for value in masses_mev]
    for mass in masses:
        global_seed_index(mass)

    load_support_freeze(
        support_freeze,
        args.support_2016_low_mev,
        args.support_2016_high_mev,
    )
    validate_config_provenance(
        config_provenance_path,
        config_path,
        support_freeze,
        args.support_2016_low_mev,
        args.support_2016_high_mev,
    )
    validate_reviewed_provenance(
        reviewed_provenance_path,
        reviewed_csv,
        support_freeze,
        args.support_2016_low_mev,
        args.support_2016_high_mev,
    )
    config = load_config(str(config_path))
    validate_v4_geometry(
        config,
        args.support_2016_low_mev,
        args.support_2016_high_mev,
    )
    validate_input_hashes(config)
    if str(config.cls_mode).lower().strip() != "asymptotic":
        raise SystemExit("Closure requires cls_mode=asymptotic.")
    if str(config.combined_mode).lower().strip() != "count_scale":
        raise SystemExit("Closure requires combined_mode=count_scale.")
    datasets = make_datasets(config)
    fixed, reviewed_coordinate_provenance = load_reviewed_coordinates(
        reviewed_csv,
        masses,
        datasets,
        config,
    )

    mass_results = [
        benchmark_mass(
            mass,
            args.toys_per_mass,
            datasets,
            config,
            fixed[round(mass, 12)],
        )
        for mass in masses
    ]
    active_counts = {
        int(result["n_active_datasets"])
        for result in mass_results
        if result["bitwise_equal"]
    }
    report = {
        "cache_algorithm_version": CACHE_ALGORITHM_VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "config_provenance": str(config_provenance_path),
        "config_provenance_sha256": sha256(config_provenance_path),
        "reviewed_csv": str(reviewed_csv),
        "reviewed_csv_sha256": reviewed_coordinate_provenance[
            "reviewed_csv_sha256"
        ],
        "reviewed_provenance": str(reviewed_provenance_path),
        "reviewed_provenance_sha256": sha256(reviewed_provenance_path),
        "support_freeze": str(support_freeze),
        "support_freeze_sha256": sha256(support_freeze),
        "support_2016_low_MeV": int(args.support_2016_low_mev),
        "support_2016_high_MeV": int(args.support_2016_high_mev),
        "runtime": RUNTIME_PROVENANCE,
        "runtime_import_origins": RUNTIME_IMPORT_ORIGINS,
        "toys_per_mass": int(args.toys_per_mass),
        "seed": SEED,
        "seed_sequence_index_rule": "mass_MeV - 19",
        "mass_results": mass_results,
        "all_bitwise_equal": bool(
            all(result["bitwise_equal"] for result in mass_results)
            and {1, 2, 3}.issubset(active_counts)
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["all_bitwise_equal"] is not True:
        raise SystemExit("Cached profile solver failed bitwise closure.")


if __name__ == "__main__":
    main()
