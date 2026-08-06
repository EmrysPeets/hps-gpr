#!/usr/bin/env python3
"""Run the v4.2 300-toy combined bands from reviewed fixed GP states.

The production path is intentionally gated by:

* an explicit reviewed-state CSV;
* a matching, passing cached-vs-reference closure report; and
* an explicit production confirmation flag.

The statistical cache is campaign-local.  Core ``hps_gpr`` files are not
patched or monkey-patched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_key in THREAD_ENV_KEYS:
    os.environ.setdefault(_thread_key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4-cached-profile-mpl")

import joblib
import numpy as np
import pandas as pd


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CAMPAIGN_DIR))

from cached_profile_solver import (  # noqa: E402
    CACHE_ALGORITHM_VERSION,
    CachedAsymptoticCombinedLimit,
)
from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import (  # noqa: E402
    _dataset_visibility,
    active_datasets_for_mass,
    build_combined_components,
)
from hps_gpr.gpr import make_fixed_kernel  # noqa: E402
from hps_gpr.io import estimate_background_for_dataset  # noqa: E402
from hps_gpr.statistics import (  # noqa: E402
    bounded_two_sided_tail_pvalue,
    draw_bkg_mvn_nonneg,
    p0_profiled_gaussian_LRT,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - the production environment has it.
    import contextlib

    threadpool_limits = contextlib.nullcontext


DEFAULT_CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
DEFAULT_OUTPUT_DIR = CAMPAIGN_DIR / "combined_bands_300toy_cached"
N_TOYS_PER_MASS = 300
SEED = 24_680
MASS_LOW_MEV = 19
MASS_HIGH_MEV = 250
N_FULL_GRID_MASSES = MASS_HIGH_MEV - MASS_LOW_MEV + 1
EXPECTED_FULL_GRID_GP_STATES = 415
LML_CLOSURE_ATOL = 5.0e-5
EXPECTED_SEARCH_RANGES = {
    "2015": (0.019, 0.090),
    "2016": (0.039, 0.180),
    "2021": (0.050, 0.250),
}
EXPECTED_DATA_RANGES = {
    "2015": (0.014, 0.135),
    "2016": (0.030, 0.210),
    "2021": (0.040, 0.300),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_state_sha256(prediction) -> str:
    digest = hashlib.sha256()
    for array in (prediction.mu, prediction.cov):
        value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(
        np.asarray(
            [prediction.sigma_val, prediction.integral_density],
            dtype="<f8",
        ).tobytes()
    )
    return digest.hexdigest()


def global_seed_index(mass_gev: float) -> int:
    """Map 19--250 MeV to the immutable full-grid SeedSequence child index."""

    mass_mev = int(round(float(mass_gev) * 1000.0))
    if not np.isclose(
        float(mass_gev),
        mass_mev / 1000.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"Mass is not on the 1 MeV grid: {mass_gev!r}")
    index = mass_mev - MASS_LOW_MEV
    if index < 0 or index >= N_FULL_GRID_MASSES:
        raise ValueError(
            f"Mass {mass_mev} MeV is outside {MASS_LOW_MEV}--{MASS_HIGH_MEV} MeV"
        )
    return index


def full_mass_grid() -> List[float]:
    return [mass_mev / 1000.0 for mass_mev in range(MASS_LOW_MEV, MASS_HIGH_MEV + 1)]


def validate_v4_geometry(config) -> None:
    """Fail closed if the production card is not the frozen v4.2 geometry."""

    for dataset_key, expected in EXPECTED_SEARCH_RANGES.items():
        found = tuple(
            float(value)
            for value in getattr(config, f"range_{dataset_key}")
        )
        if found != expected:
            raise RuntimeError(
                f"Unexpected {dataset_key} search range: {found} != {expected}"
            )
    for dataset_key, expected in EXPECTED_DATA_RANGES.items():
        found = tuple(
            float(value)
            for value in getattr(config, f"data_range_{dataset_key}")
        )
        if found != expected:
            raise RuntimeError(
                f"Unexpected {dataset_key} fit support: {found} != {expected}"
            )


def _bool_column_has_true(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return bool(series.fillna(False).any())
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return bool(normalized.isin({"true", "1", "yes"}).any())


def load_reviewed_coordinates(
    reviewed_csv: Path,
    masses: Sequence[float],
    datasets: dict,
    config,
) -> Tuple[Dict[float, Dict[str, Dict[str, float]]], dict]:
    """Load and fail-closed validate reviewed fixed GP coordinates."""

    reviewed = pd.read_csv(reviewed_csv)
    required = {"dataset", "mass_GeV", "const_opt", "ls_opt", "lml"}
    missing = sorted(required.difference(reviewed.columns))
    if missing:
        raise RuntimeError(
            f"Reviewed-state CSV is missing required columns: {missing}"
        )
    if "interpolated" in reviewed.columns and _bool_column_has_true(
        reviewed["interpolated"]
    ):
        raise RuntimeError(
            "Reviewed-state CSV contains interpolated states; stable reruns are required."
        )

    duplicate = reviewed.duplicated(["dataset", "mass_GeV"], keep=False)
    if bool(duplicate.any()):
        sample = reviewed.loc[duplicate, ["dataset", "mass_GeV"]].head(10)
        raise RuntimeError(
            "Duplicate reviewed GP states:\n" + sample.to_string(index=False)
        )

    fixed: Dict[float, Dict[str, Dict[str, float]]] = {}
    selected_states = 0
    for mass in masses:
        here = reviewed[
            np.isclose(
                reviewed["mass_GeV"].to_numpy(float),
                float(mass),
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        expected_keys = {
            dataset.key
            for dataset in active_datasets_for_mass(
                float(mass),
                datasets,
                config,
            )
        }
        found_keys = set(here["dataset"].astype(str))
        if found_keys != expected_keys:
            raise RuntimeError(
                f"Reviewed active-set mismatch at {mass:.3f} GeV: "
                f"found={sorted(found_keys)}, expected={sorted(expected_keys)}"
            )
        key = round(float(mass), 12)
        fixed[key] = {}
        for row in here.itertuples(index=False):
            dataset_key = str(row.dataset)
            values = {
                "const_opt": float(row.const_opt),
                "ls_opt": float(row.ls_opt),
                "reviewed_lml": float(row.lml),
            }
            if not all(np.isfinite(list(values.values()))):
                raise RuntimeError(
                    f"Non-finite reviewed GP coordinate for {dataset_key} "
                    f"at {mass:.3f} GeV"
                )
            fixed[key][dataset_key] = values
            selected_states += 1

    if len(masses) == N_FULL_GRID_MASSES:
        if selected_states != EXPECTED_FULL_GRID_GP_STATES:
            raise RuntimeError(
                f"Expected {EXPECTED_FULL_GRID_GP_STATES} reviewed GP states "
                f"on the full grid, found {selected_states}"
            )
        if len(reviewed) != EXPECTED_FULL_GRID_GP_STATES:
            raise RuntimeError(
                f"Full-grid reviewed CSV should contain exactly "
                f"{EXPECTED_FULL_GRID_GP_STATES} rows, found {len(reviewed)}"
            )

    payload = json.dumps(
        {
            f"{mass:.12f}": values
            for mass, values in sorted(fixed.items())
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    provenance = {
        "reviewed_csv": str(reviewed_csv.resolve()),
        "reviewed_csv_sha256": sha256(reviewed_csv),
        "selected_reviewed_gp_states": int(selected_states),
        "reviewed_gp_coordinates_sha256": hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
    }
    return fixed, provenance


def build_fixed_predictions(
    mass: float,
    datasets: dict,
    config,
    fixed_here: Dict[str, Dict[str, float]],
) -> Tuple[list, list, list]:
    """Reconstruct one mass point at the exact reviewed GP coordinates."""

    datasets_here = active_datasets_for_mass(float(mass), datasets, config)
    expected_keys = [dataset.key for dataset in datasets_here]
    if set(fixed_here) != set(expected_keys):
        raise RuntimeError(
            f"Fixed-coordinate active-set mismatch at {mass:.3f} GeV: "
            f"{sorted(fixed_here)} != {sorted(expected_keys)}"
        )

    predictions = []
    metadata = []
    lml_differences = []
    with threadpool_limits(limits=1):
        for dataset in datasets_here:
            reviewed = fixed_here[dataset.key]
            prediction = estimate_background_for_dataset(
                dataset,
                float(mass),
                config,
                restarts=0,
                train_exclude_nsigma=float(config.gp_train_exclude_nsigma),
                kernel=make_fixed_kernel(
                    float(reviewed["const_opt"]),
                    float(reviewed["ls_opt"]),
                ),
                optimize=False,
            )
            lml_difference = float(
                prediction.lml - float(reviewed["reviewed_lml"])
            )
            if (
                not np.isfinite(lml_difference)
                or abs(lml_difference) > LML_CLOSURE_ATOL
            ):
                raise RuntimeError(
                    f"Reviewed LML closure failed for {dataset.key} at "
                    f"{mass:.3f} GeV: recomputed={prediction.lml:.12g}, "
                    f"reviewed={reviewed['reviewed_lml']:.12g}, "
                    f"delta={lml_difference:.6g}"
                )
            predictions.append(prediction)
            lml_differences.append(lml_difference)
            metadata.append(
                {
                    "key": dataset.key,
                    "sigma": float(prediction.sigma_val),
                    "dens": float(prediction.integral_density),
                    "lml": float(prediction.lml),
                    "reviewed_lml": float(reviewed["reviewed_lml"]),
                    "lml_delta": lml_difference,
                    "ls_opt": float(prediction.ls_opt),
                    "const_opt": float(prediction.const_opt),
                    "state_sha256": prediction_state_sha256(prediction),
                }
            )
    return datasets_here, predictions, metadata


def _quantiles(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"),) * 5
    return tuple(
        float(value)
        for value in np.quantile(finite, [0.025, 0.16, 0.50, 0.84, 0.975])
    )


def run_one_mass(
    mass: float,
    seed_sequence,
    datasets: dict,
    config,
    fixed_here: Dict[str, Dict[str, float]],
) -> dict:
    """Produce the 300-toy combined-band row for one fixed mass."""

    if str(config.cls_mode).lower().strip() != "asymptotic":
        raise RuntimeError("The cached v4 runner requires cls_mode=asymptotic.")
    if str(config.combined_mode).lower().strip() != "count_scale":
        raise RuntimeError("The reviewed v4 runner requires combined_mode=count_scale.")

    rng = np.random.default_rng(seed_sequence)
    datasets_here, predictions, metadata = build_fixed_predictions(
        mass,
        datasets,
        config,
        fixed_here,
    )
    dataset_keys = [dataset.key for dataset in datasets_here]
    if not all(
        _dataset_visibility(datasets[key], config) == "observed"
        for key in dataset_keys
    ):
        raise RuntimeError(
            f"All active datasets must be observed at {mass:.3f} GeV."
        )

    observed, b_mean, b_cov, s_unit = build_combined_components(
        float(mass),
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

    # Preserve the reference runner's RNG stream exactly.  The inner seed is
    # unused by asymptotic CLs, but consuming it before drawing pseudo-data is
    # part of the reviewed SeedSequence semantics.
    rng.integers(1, 2**31 - 1)
    eps2_observed = solver.limit(observed)

    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))
    lambda_draws = [
        draw_bkg_mvn_nonneg(
            prediction.mu,
            prediction.cov,
            N_TOYS_PER_MASS,
            rng,
            method=mvn_method,
            max_tries=mvn_max_tries,
        )
        for prediction in predictions
    ]
    count_draws = [
        rng.poisson(lambdas).astype(int)
        for lambdas in lambda_draws
    ]

    toy_limits = np.empty(N_TOYS_PER_MASS, dtype=float)
    for toy_index in range(N_TOYS_PER_MASS):
        toy_observed = np.concatenate(
            [draws[toy_index] for draws in count_draws]
        )
        rng.integers(1, 2**31 - 1)
        toy_limits[toy_index] = solver.limit(toy_observed)

    finite_limits = toy_limits[np.isfinite(toy_limits)]
    n_finite = int(finite_limits.size)
    q02, q16, q50, q84, q97 = _quantiles(toy_limits)
    mean_limit = (
        float(np.mean(finite_limits))
        if n_finite
        else float("nan")
    )
    if n_finite and np.isfinite(eps2_observed):
        tail_count_strong = int(
            np.count_nonzero(finite_limits <= eps2_observed)
        )
        tail_count_weak = int(
            np.count_nonzero(finite_limits >= eps2_observed)
        )
        tail_count_equal = int(
            np.count_nonzero(finite_limits == eps2_observed)
        )
        tail_count_two_sided_min = min(
            tail_count_strong,
            tail_count_weak,
        )
        p_strong = float(tail_count_strong / n_finite)
        p_weak = float(tail_count_weak / n_finite)
        p_two = bounded_two_sided_tail_pvalue(p_strong, p_weak)
    else:
        tail_count_strong = 0
        tail_count_weak = 0
        tail_count_equal = 0
        tail_count_two_sided_min = 0
        p_strong = p_weak = p_two = float("nan")

    try:
        p0_analytic, z_analytic, _, _ = p0_profiled_gaussian_LRT(
            observed,
            b_mean,
            b_cov,
            s_unit / float(config.eps2_lrt_scale),
        )
    except Exception:
        p0_analytic = z_analytic = float("nan")

    lml = {item["key"]: item["lml"] for item in metadata}
    ls_opt = {item["key"]: item["ls_opt"] for item in metadata}
    const_opt = {item["key"]: item["const_opt"] for item in metadata}
    state_sha = {item["key"]: item["state_sha256"] for item in metadata}
    mass_seed_index = global_seed_index(mass)

    return {
        "dataset_set": "+".join(dataset_keys),
        "mass_GeV": float(mass),
        "sigma_mass_res_GeV": float(
            np.mean([prediction.sigma_val for prediction in predictions])
        ),
        "sigma_mass_res_min_GeV": float(
            np.min([prediction.sigma_val for prediction in predictions])
        ),
        "cls_alpha": float(config.cls_alpha),
        "eps2_obs": float(eps2_observed),
        "p0_analytic": float(p0_analytic),
        "Z_analytic": float(z_analytic),
        "eps2_lo2": q02,
        "eps2_lo1": q16,
        "eps2_med": q50,
        "eps2_hi1": q84,
        "eps2_hi2": q97,
        "eps2_mean": mean_limit,
        "ul_eps2_obs": float(eps2_observed),
        "toy_eps2_uls_q02": q02,
        "toy_eps2_uls_q16": q16,
        "toy_eps2_uls_q50": q50,
        "toy_eps2_uls_q84": q84,
        "toy_eps2_uls_q97": q97,
        "toy_eps2_uls_mean": mean_limit,
        "p_strong": p_strong,
        "p_weak": p_weak,
        "p_two": p_two,
        "tail_count_strong_le_observed": tail_count_strong,
        "tail_count_weak_ge_observed": tail_count_weak,
        "tail_count_equal_observed": tail_count_equal,
        "tail_count_two_sided_min": tail_count_two_sided_min,
        "empirical_tail_resolution": (
            float(1.0 / n_finite) if n_finite else float("nan")
        ),
        "meta": json.dumps(metadata, sort_keys=True),
        "cls_statistic": "tilde_q_mu",
        "cls_calibration": "asymptotic",
        "combined_mode": "count_scale",
        "global_method": "sidak_approx",
        "bands_refit_gp_on_toy": False,
        "bands_train_exclude_nsigma": float(
            config.gp_train_exclude_nsigma
        ),
        "bands_refit_restarts": 0,
        "bands_refit_optimize": False,
        "bands_seed_sequence_index": mass_seed_index,
        "n_toys_requested": N_TOYS_PER_MASS,
        "n_toys_finite": n_finite,
        "gp_lml_by_dataset": json.dumps(lml, sort_keys=True),
        "gp_ls_opt_by_dataset": json.dumps(ls_opt, sort_keys=True),
        "gp_const_opt_by_dataset": json.dumps(const_opt, sort_keys=True),
        "gp_state_sha256_by_dataset": json.dumps(state_sha, sort_keys=True),
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "limit_solver": CACHE_ALGORITHM_VERSION,
        "profile_cache_limit_calls": solver.counters.limit_calls,
        "profile_cache_asimov_fixed_nodes": solver.asimov_fixed_cache_size,
        "profile_cache_asimov_fixed_hits": (
            solver.counters.asimov_fixed_cache_hits
        ),
        "profile_cache_asimov_fixed_misses": (
            solver.counters.asimov_fixed_cache_misses
        ),
    }


def validate_closure_report(
    closure_report: Path,
    config_path: Path,
    reviewed_csv: Path,
) -> dict:
    report = json.loads(closure_report.read_text(encoding="utf-8"))
    if report.get("cache_algorithm_version") != CACHE_ALGORITHM_VERSION:
        raise RuntimeError("Closure report was produced by another cache version.")
    if report.get("config_sha256") != sha256(config_path):
        raise RuntimeError("Closure report does not match the requested config.")
    if report.get("reviewed_csv_sha256") != sha256(reviewed_csv):
        raise RuntimeError(
            "Closure report does not match the requested reviewed-state CSV."
        )
    if report.get("all_bitwise_equal") is not True:
        raise RuntimeError("Closure report did not pass bitwise equality.")
    active_counts = {
        int(entry["n_active_datasets"])
        for entry in report.get("mass_results", [])
        if entry.get("bitwise_equal") is True
    }
    if not {1, 2, 3}.issubset(active_counts):
        raise RuntimeError(
            "Closure report must cover passing one-, two-, and three-dataset masses."
        )
    return report


def selected_masses(args: argparse.Namespace) -> List[float]:
    masses = full_mass_grid()
    if args.mass_mev:
        requested = sorted(set(int(value) for value in args.mass_mev))
        masses = [value / 1000.0 for value in requested]
        for mass in masses:
            global_seed_index(mass)
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count.")
    return [
        mass
        for mass in masses
        if global_seed_index(mass) % args.shard_count == args.shard_index
    ]


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.shard_count == 1:
        stem = "ul_bands_combined_all"
    else:
        stem = (
            "ul_bands_combined_all_"
            f"shard{args.shard_index:03d}of{args.shard_count:03d}"
        )
    return output_dir / f"{stem}.csv", output_dir / f"{stem}_provenance.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reviewed-state-csv", type=Path, required=True)
    parser.add_argument("--closure-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--mass-mev",
        type=int,
        action="append",
        help="Optional explicit mass in MeV; repeat for multiple masses.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required acknowledgement for the 300-toy production run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.confirm_production:
        raise SystemExit(
            "Production was not started. Pass --confirm-production only after "
            "the v4 reviewed-state CSV and matching closure report are final."
        )
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")
    for key in THREAD_ENV_KEYS:
        if os.environ.get(key) != "1":
            raise SystemExit(f"{key}=1 is required, got {os.environ.get(key)!r}.")

    config_path = args.config.expanduser().resolve()
    reviewed_csv = args.reviewed_state_csv.expanduser().resolve()
    closure_report = args.closure_report.expanduser().resolve()
    for path in (config_path, reviewed_csv, closure_report):
        if not path.is_file():
            raise SystemExit(f"Required file does not exist: {path}")

    config = replace(
        load_config(str(config_path)),
        ul_bands_n_workers=int(args.workers),
        ul_bands_parallel_backend="loky",
        ul_bands_threads_per_worker=1,
    )
    if not np.isclose(float(config.cls_alpha), 0.1, rtol=0.0, atol=0.0):
        raise SystemExit(f"Expected 90% CL (alpha=0.1), got {config.cls_alpha}.")
    validate_v4_geometry(config)
    expected_upper_factors = {"2015": 8.0, "2016": 12.0, "2021": 15.0}
    configured_upper_factors = {
        str(key): float(value)
        for key, value in config.kernel_ls_res_upper_factor_by_dataset.items()
    }
    if configured_upper_factors != expected_upper_factors:
        raise SystemExit(
            "Unexpected v4.2 dataset-specific length-scale upper factors: "
            f"{configured_upper_factors} != {expected_upper_factors}"
        )
    if not bool(config.make_ul_bands):
        raise SystemExit("The combined-band outer gate must be enabled.")
    if int(config.ul_bands_toys) != 0 or str(config.run_limit_bands_on) != "":
        raise SystemExit("Individual expected-limit bands must remain disabled.")
    if not bool(config.do_combined_bands):
        raise SystemExit("The v4.2 card must declare combined bands enabled.")
    if int(config.combined_bands_n_toys) != N_TOYS_PER_MASS:
        raise SystemExit(
            "The v4.2 card must declare exactly "
            f"{N_TOYS_PER_MASS} combined toys per mass."
        )
    closure = validate_closure_report(
        closure_report,
        config_path,
        reviewed_csv,
    )

    masses = selected_masses(args)
    if not masses:
        raise SystemExit("This shard contains no requested masses.")
    datasets = make_datasets(config)
    fixed, fixed_provenance = load_reviewed_coordinates(
        reviewed_csv,
        masses,
        datasets,
        config,
    )

    all_children = np.random.SeedSequence(SEED).spawn(N_FULL_GRID_MASSES)
    tasks = [
        (
            mass,
            all_children[global_seed_index(mass)],
            fixed[round(float(mass), 12)],
        )
        for mass in masses
    ]

    started = time.time()
    if args.workers == 1:
        rows = [
            run_one_mass(
                mass,
                seed_sequence,
                datasets,
                config,
                fixed_here,
            )
            for mass, seed_sequence, fixed_here in tasks
        ]
    else:
        rows = joblib.Parallel(
            n_jobs=int(args.workers),
            backend="loky",
        )(
            joblib.delayed(run_one_mass)(
                mass,
                seed_sequence,
                datasets,
                config,
                fixed_here,
            )
            for mass, seed_sequence, fixed_here in tasks
        )
    elapsed = time.time() - started

    bands = pd.DataFrame(rows).sort_values("mass_GeV").reset_index(drop=True)
    expected_indices = [global_seed_index(mass) for mass in masses]
    found_indices = bands["bands_seed_sequence_index"].astype(int).tolist()
    if found_indices != expected_indices:
        raise RuntimeError(
            f"SeedSequence-index closure failed: {found_indices} != {expected_indices}"
        )
    if not bool((bands["n_toys_requested"] == N_TOYS_PER_MASS).all()):
        raise RuntimeError("Requested-toy count changed unexpectedly.")
    if not bool((bands["n_toys_finite"] == N_TOYS_PER_MASS).all()):
        bad = bands.loc[
            bands["n_toys_finite"] != N_TOYS_PER_MASS,
            ["mass_GeV", "n_toys_finite"],
        ]
        raise RuntimeError(
            "Non-finite toy limits were produced:\n" + bad.to_string(index=False)
        )
    for row in bands.itertuples(index=False):
        n_finite = int(row.n_toys_finite)
        n_strong = int(row.tail_count_strong_le_observed)
        n_weak = int(row.tail_count_weak_ge_observed)
        n_equal = int(row.tail_count_equal_observed)
        n_two_min = int(row.tail_count_two_sided_min)
        if n_strong + n_weak - n_equal != n_finite:
            raise RuntimeError(
                f"Tail-count partition failed at {row.mass_GeV:.3f} GeV."
            )
        if n_two_min != min(n_strong, n_weak):
            raise RuntimeError(
                f"Two-sided raw tail count failed at {row.mass_GeV:.3f} GeV."
            )
        if float(row.p_strong) != float(n_strong / n_finite):
            raise RuntimeError(
                f"Strong-tail p-value/count mismatch at {row.mass_GeV:.3f} GeV."
            )
        if float(row.p_weak) != float(n_weak / n_finite):
            raise RuntimeError(
                f"Weak-tail p-value/count mismatch at {row.mass_GeV:.3f} GeV."
            )
        expected_two = min(
            1.0,
            2.0 * min(float(row.p_strong), float(row.p_weak)),
        )
        if float(row.p_two) != float(expected_two):
            raise RuntimeError(
                f"Two-sided p-value/count mismatch at {row.mass_GeV:.3f} GeV."
            )

    csv_path, provenance_path = output_paths(args)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    bands.to_csv(csv_path, index=False)
    provenance = {
        "cache_algorithm_version": CACHE_ALGORITHM_VERSION,
        "physics_config": str(config_path),
        "physics_config_sha256": sha256(config_path),
        **fixed_provenance,
        "closure_report": str(closure_report),
        "closure_report_sha256": sha256(closure_report),
        "closure_mass_results": closure["mass_results"],
        "n_toys_per_mass": N_TOYS_PER_MASS,
        "seed": SEED,
        "seed_sequence_index_rule": "mass_MeV - 19",
        "mass_grid_GeV": [float(value) for value in masses],
        "n_masses": len(masses),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "parallel_workers": int(args.workers),
        "parallel_backend": "loky",
        "threads_per_worker": 1,
        "refit_gp_on_toy": False,
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "toy_construction": "conditional GP posterior MVN then Poisson",
        "inner_cls": "asymptotic tilde_q_mu, alpha=0.1",
        "elapsed_seconds": float(elapsed),
        "output_csv": str(csv_path),
        "output_csv_sha256": sha256(csv_path),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cached_solver": str(
            (CAMPAIGN_DIR / "cached_profile_solver.py").resolve()
        ),
        "cached_solver_sha256": sha256(
            CAMPAIGN_DIR / "cached_profile_solver.py"
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(bands)} masses x {N_TOYS_PER_MASS} toys to {csv_path} "
        f"in {elapsed / 3600.0:.2f} h."
    )


if __name__ == "__main__":
    main()
