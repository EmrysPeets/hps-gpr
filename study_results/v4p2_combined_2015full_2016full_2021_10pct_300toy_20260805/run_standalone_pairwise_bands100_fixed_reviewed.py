#!/usr/bin/env python3
"""Run six v4.2 conditional 100-toy upper-limit band scopes.

The scopes are the three standalone datasets and the three genuine pairwise
shared-epsilon-squared intersections:

* 2015, 2016, and 2021;
* 2015+2016, 2015+2021, and 2016+2021.

Every mass uses the exact accepted v4.2 fixed GP coordinates and reconstructs
the accepted 300-draw parent pseudo-data stream.  Toy indices 0--99 are then
reused across every scope.  Thus one toy index denotes the same dataset
pseudo-spectrum in its standalone and pairwise appearances, while pair members
are combined by the same toy index.  Different masses remain independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_key in THREAD_ENV_KEYS:
    os.environ.setdefault(_thread_key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p2-bands100-mpl")

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from cached_profile_solver import (  # noqa: E402
    CACHE_ALGORITHM_VERSION,
    CachedAsymptoticCombinedLimit,
)
from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.conversion import A_from_epsilon2  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import (  # noqa: E402
    active_datasets_for_mass,
    build_combined_components,
)
from hps_gpr.gpr import make_fixed_kernel  # noqa: E402
from hps_gpr.io import estimate_background_for_dataset  # noqa: E402
from hps_gpr.statistics import (  # noqa: E402
    bounded_two_sided_tail_pvalue,
    draw_bkg_mvn_nonneg,
    p0_profiled_gaussian_LRT,
    profiled_gaussian_likelihood_summary,
)
from run_combined_bands_cached_fixed_reviewed import (  # noqa: E402
    LML_CLOSURE_ATOL,
    N_FULL_GRID_MASSES,
    N_TOYS_PER_MASS as ACCEPTED_PARENT_DRAW_COUNT,
    SEED as ACCEPTED_ROOT_SEED,
    THREAD_ENV_KEYS as PRODUCTION_THREAD_ENV_KEYS,
    global_seed_index,
    prediction_state_sha256,
    sha256,
    validate_closure_report,
    validate_v4_geometry,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    import contextlib

    threadpool_limits = contextlib.nullcontext


DEFAULT_CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
DEFAULT_REVIEWED = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
DEFAULT_CLOSURE = HERE / "derived" / "cached_profile_closure_v4p2.json"
DEFAULT_ACCEPTED_COMBINED = (
    HERE / "derived" / "combined_bands300_reviewed_v4p2.csv"
)
DEFAULT_INDIVIDUAL = (
    HERE / "derived" / "individual_observed_limits_reviewed_v4p2.csv"
)
DEFAULT_OUTPUT_DIR = HERE / "standalone_pairwise_bands100_fixed"

EXPECTED_CONFIG_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
EXPECTED_REVIEWED_SHA256 = (
    "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9"
)
EXPECTED_ACCEPTED_COMBINED_SHA256 = (
    "8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd"
)
EXPECTED_INDIVIDUAL_SHA256 = (
    "1e3e99fb7c0a171d6d496de87ac6664b485928042b2cede242dffab55e0cc410"
)
N_TOYS_PER_MASS = 100
SELECTED_PARENT_TOY_INDICES = tuple(range(N_TOYS_PER_MASS))
REPRESENTATIVE_PARENT_CLOSURE_MEV = (20, 40, 60, 100, 200)
CANONICAL_DATASET_ORDER = ("2015", "2016", "2021")
STANDALONE_OBSERVED_CLOSURE_RTOL = 6.0e-6

# Pairwise scopes deliberately use only the intersection where both named
# datasets contribute.  They do not append standalone-only endpoint segments.
SCOPES: "OrderedDict[str, Mapping[str, object]]" = OrderedDict(
    [
        (
            "individual_2015",
            {
                "scope_type": "standalone",
                "dataset_keys": ("2015",),
                "label": "2015 100%",
                "mass_low_MeV": 19,
                "mass_high_MeV": 90,
            },
        ),
        (
            "individual_2016",
            {
                "scope_type": "standalone",
                "dataset_keys": ("2016",),
                "label": "2016 100%",
                "mass_low_MeV": 39,
                "mass_high_MeV": 180,
            },
        ),
        (
            "individual_2021",
            {
                "scope_type": "standalone",
                "dataset_keys": ("2021",),
                "label": "2021 10%",
                "mass_low_MeV": 50,
                "mass_high_MeV": 250,
            },
        ),
        (
            "pair_2015_2016",
            {
                "scope_type": "pairwise",
                "dataset_keys": ("2015", "2016"),
                "label": "2015 100% + 2016 100%",
                "mass_low_MeV": 39,
                "mass_high_MeV": 90,
            },
        ),
        (
            "pair_2015_2021",
            {
                "scope_type": "pairwise",
                "dataset_keys": ("2015", "2021"),
                "label": "2015 100% + 2021 10%",
                "mass_low_MeV": 50,
                "mass_high_MeV": 90,
            },
        ),
        (
            "pair_2016_2021",
            {
                "scope_type": "pairwise",
                "dataset_keys": ("2016", "2021"),
                "label": "2016 100% + 2021 10%",
                "mass_low_MeV": 50,
                "mass_high_MeV": 180,
            },
        ),
    ]
)


def scope_masses(scope: Mapping[str, object]) -> List[float]:
    low = int(scope["mass_low_MeV"])
    high = int(scope["mass_high_MeV"])
    return [mass_mev / 1000.0 for mass_mev in range(low, high + 1)]


def scopes_at_mass(mass: float) -> List[Tuple[str, Mapping[str, object]]]:
    mass_mev = int(round(1000.0 * float(mass)))
    return [
        (scope_key, scope)
        for scope_key, scope in SCOPES.items()
        if int(scope["mass_low_MeV"])
        <= mass_mev
        <= int(scope["mass_high_MeV"])
    ]


def full_requested_mass_grid() -> List[float]:
    requested = {
        mass
        for scope in SCOPES.values()
        for mass in scope_masses(scope)
    }
    return sorted(requested)


def quantiles(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"),) * 5
    return tuple(
        float(value)
        for value in np.quantile(
            finite,
            [0.025, 0.16, 0.50, 0.84, 0.975],
        )
    )


def load_fixed_coordinates(
    reviewed_path: Path,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    reviewed = pd.read_csv(reviewed_path)
    required = {"dataset", "mass_GeV", "const_opt", "ls_opt", "lml"}
    missing = sorted(required.difference(reviewed.columns))
    if missing:
        raise RuntimeError(f"Reviewed ledger missing columns: {missing}")
    if len(reviewed) != 415:
        raise RuntimeError(
            f"Expected exactly 415 reviewed states, found {len(reviewed)}"
        )
    if reviewed.duplicated(["dataset", "mass_GeV"]).any():
        raise RuntimeError("Reviewed ledger contains duplicate dataset-mass rows")
    if "interpolated" in reviewed.columns:
        interpolated = (
            reviewed["interpolated"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        if bool(interpolated.any()):
            raise RuntimeError("Reviewed ledger contains interpolated states")

    fixed: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in reviewed.itertuples(index=False):
        dataset = str(row.dataset)
        mass_mev = int(round(1000.0 * float(row.mass_GeV)))
        if not np.isclose(
            float(row.mass_GeV),
            mass_mev / 1000.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Reviewed mass is off the 1 MeV grid: {row.mass_GeV!r}"
            )
        values = {
            "const_opt": float(row.const_opt),
            "ls_opt": float(row.ls_opt),
            "reviewed_lml": float(row.lml),
        }
        if not all(np.isfinite(list(values.values()))):
            raise RuntimeError(
                f"Non-finite reviewed coordinate for {dataset} at {mass_mev} MeV"
            )
        fixed[(dataset, mass_mev)] = values
    return fixed


def load_accepted_state_hashes(
    accepted_path: Path,
) -> Dict[Tuple[str, int], str]:
    accepted = pd.read_csv(accepted_path)
    required = {"mass_GeV", "gp_state_sha256_by_dataset"}
    missing = sorted(required.difference(accepted.columns))
    if missing:
        raise RuntimeError(f"Accepted combined table missing columns: {missing}")
    if len(accepted) != 232:
        raise RuntimeError(
            f"Expected 232 accepted combined rows, found {len(accepted)}"
        )
    hashes: Dict[Tuple[str, int], str] = {}
    for row in accepted.itertuples(index=False):
        mass_mev = int(round(1000.0 * float(row.mass_GeV)))
        mapping = json.loads(str(row.gp_state_sha256_by_dataset))
        for dataset, value in mapping.items():
            hashes[(str(dataset), mass_mev)] = str(value)
    if len(hashes) != 415:
        raise RuntimeError(
            f"Expected 415 accepted state hashes, found {len(hashes)}"
        )
    return hashes


def load_authoritative_individual_rows(
    individual_path: Path,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    individual = pd.read_csv(individual_path, dtype={"dataset": str})
    required = {
        "dataset",
        "mass_GeV",
        "A_up",
        "eps2_up",
        "integral_density",
        "p0_analytic",
        "Z_analytic",
    }
    missing = sorted(required.difference(individual.columns))
    if missing:
        raise RuntimeError(f"Individual table missing columns: {missing}")
    if len(individual) != 415:
        raise RuntimeError(
            f"Expected exactly 415 individual rows, found {len(individual)}"
        )
    rows: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in individual.itertuples(index=False):
        dataset = str(row.dataset)
        mass_mev = int(round(1000.0 * float(row.mass_GeV)))
        key = (dataset, mass_mev)
        if key in rows:
            raise RuntimeError(f"Duplicate individual row: {key}")
        rows[key] = {
            "A_obs": float(row.A_up),
            "eps2_obs": float(row.eps2_up),
            "integral_density": float(row.integral_density),
            "p0_analytic": float(row.p0_analytic),
            "Z_analytic": float(row.Z_analytic),
        }
    return rows


def build_predictions_for_mass(
    mass: float,
    required_keys: Sequence[str],
    datasets: Mapping[str, object],
    config,
    fixed: Mapping[Tuple[str, int], Mapping[str, float]],
    accepted_hashes: Mapping[Tuple[str, int], str],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    mass_mev = int(round(1000.0 * float(mass)))
    predictions: Dict[str, object] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    with threadpool_limits(limits=1):
        for dataset_key in required_keys:
            coordinate_key = (dataset_key, mass_mev)
            if coordinate_key not in fixed:
                raise RuntimeError(
                    f"Missing fixed state for {dataset_key} at {mass_mev} MeV"
                )
            coordinate = fixed[coordinate_key]
            prediction = estimate_background_for_dataset(
                datasets[dataset_key],
                float(mass),
                config,
                restarts=0,
                train_exclude_nsigma=float(config.gp_train_exclude_nsigma),
                kernel=make_fixed_kernel(
                    float(coordinate["const_opt"]),
                    float(coordinate["ls_opt"]),
                ),
                optimize=False,
            )
            lml_delta = float(
                prediction.lml - float(coordinate["reviewed_lml"])
            )
            if not np.isfinite(lml_delta) or abs(lml_delta) > LML_CLOSURE_ATOL:
                raise RuntimeError(
                    f"LML closure failed for {dataset_key} at {mass_mev} MeV: "
                    f"delta={lml_delta:.8g}"
                )
            state_hash = prediction_state_sha256(prediction)
            accepted_hash = accepted_hashes.get(coordinate_key)
            if state_hash != accepted_hash:
                raise RuntimeError(
                    f"State-hash closure failed for {dataset_key} at "
                    f"{mass_mev} MeV: {state_hash} != {accepted_hash}"
                )
            predictions[dataset_key] = prediction
            metadata[dataset_key] = {
                "key": dataset_key,
                "sigma": float(prediction.sigma_val),
                "dens": float(prediction.integral_density),
                "lml": float(prediction.lml),
                "reviewed_lml": float(coordinate["reviewed_lml"]),
                "lml_delta": lml_delta,
                "ls_opt": float(prediction.ls_opt),
                "const_opt": float(prediction.const_opt),
                "state_sha256": state_hash,
            }
    return predictions, metadata


def parent_toy_counts(
    mass: float,
    active_keys: Sequence[str],
    predictions: Mapping[str, object],
    config,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
    """Reconstruct the accepted 300-draw parent stream for one mass.

    The accepted runner first consumes one unused inner seed, then generates
    every active dataset's MVN arrays in canonical order, and only afterward
    performs the Poisson draws in that same order.  The reported 100-toy
    ensembles select parent rows 0--99 without changing those semantics.
    """

    mass_index = global_seed_index(mass)
    seed_sequence = np.random.SeedSequence(ACCEPTED_ROOT_SEED).spawn(
        N_FULL_GRID_MASSES
    )[mass_index]
    rng = np.random.default_rng(seed_sequence)
    rng.integers(1, 2**31 - 1)
    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))
    lambda_draws = {
        key: draw_bkg_mvn_nonneg(
            predictions[key].mu,
            predictions[key].cov,
            ACCEPTED_PARENT_DRAW_COUNT,
            rng,
            method=mvn_method,
            max_tries=mvn_max_tries,
        )
        for key in active_keys
    }
    parent_counts = {
        key: rng.poisson(lambda_draws[key]).astype(int)
        for key in active_keys
    }
    stream_metadata = {
        key: {
            "root_seed": int(ACCEPTED_ROOT_SEED),
            "seed_sequence_index": int(mass_index),
            "stream_rule": (
                "SeedSequence(24680).spawn(232)[mass_MeV-19]; "
                "accepted pre-draw inner-seed consumption; all active MVN "
                "arrays then all active Poisson arrays in canonical order"
            ),
            "active_dataset_order": list(active_keys),
            "parent_draw_count": int(ACCEPTED_PARENT_DRAW_COUNT),
            "selected_parent_indices": "0-99",
        }
        for key in active_keys
    }
    return parent_counts, stream_metadata


def standalone_native_limit_cached(
    solver: CachedAsymptoticCombinedLimit,
    counts: np.ndarray,
    likelihood_amplitude_per_eps2: float,
    reported_amplitude_per_eps2: float,
) -> float:
    """Match the reviewed standalone amplitude-coordinate root convention.

    The likelihood and CLs mapping are the same as the shared-epsilon-squared
    solver.  Only the numerical root coordinate and stopping thresholds differ:
    the authoritative standalone scan bisects total signal amplitude for at
    most 40 steps and stops at |CLs-alpha| < 1e-6.  The cached likelihood
    evaluations remain exact under this coordinate transformation.
    """

    counts = np.asarray(counts, dtype=int)
    if counts.shape != solver.b.shape:
        raise ValueError(
            f"Count-vector shape mismatch: {counts.shape} vs {solver.b.shape}"
        )
    likelihood_amplitude_per_eps2 = float(
        likelihood_amplitude_per_eps2
    )
    reported_amplitude_per_eps2 = float(reported_amplitude_per_eps2)
    if (
        not np.isfinite(likelihood_amplitude_per_eps2)
        or likelihood_amplitude_per_eps2 <= 0.0
    ):
        raise ValueError(
            "Invalid likelihood amplitude-per-epsilon-squared: "
            f"{likelihood_amplitude_per_eps2}"
        )
    if (
        not np.isfinite(reported_amplitude_per_eps2)
        or reported_amplitude_per_eps2 <= 0.0
    ):
        raise ValueError(
            "Invalid reported amplitude-per-epsilon-squared: "
            f"{reported_amplitude_per_eps2}"
        )
    solver.counters.limit_calls += 1
    observed_base = profiled_gaussian_likelihood_summary(
        counts,
        solver.b,
        solver.cov,
        solver.signal_template,
        A_fixed=None,
    )

    def cls_at_amplitude(amplitude: float) -> float:
        return float(
            solver._cls_at_eps2(
                float(amplitude) / likelihood_amplitude_per_eps2,
                counts,
                observed_base,
            )
        )

    amplitude_low = 0.0
    amplitude_high = max(
        1.0,
        3.0 * math.sqrt(max(float(np.sum(solver.b)), 1.0)),
    )
    cls_high = cls_at_amplitude(amplitude_high)
    iterations = 0
    while cls_high > solver.alpha and amplitude_high < 1.0e7 and iterations < 40:
        amplitude_high *= 2.0
        cls_high = cls_at_amplitude(amplitude_high)
        iterations += 1

    for _ in range(40):
        midpoint = 0.5 * (amplitude_low + amplitude_high)
        cls_midpoint = cls_at_amplitude(midpoint)
        if abs(cls_midpoint - solver.alpha) < 1.0e-6:
            amplitude_low = amplitude_high = midpoint
            break
        if cls_midpoint > solver.alpha:
            amplitude_low = midpoint
        else:
            amplitude_high = midpoint
        if abs(amplitude_high - amplitude_low) <= max(
            1.0e-12,
            1.0e-6 * max(abs(amplitude_high), abs(amplitude_low)),
        ):
            break
    return float(
        0.5
        * (amplitude_low + amplitude_high)
        / reported_amplitude_per_eps2
    )


def run_scope(
    mass: float,
    scope_key: str,
    scope: Mapping[str, object],
    datasets: Mapping[str, object],
    predictions: Mapping[str, object],
    metadata: Mapping[str, Mapping[str, object]],
    count_draws: Mapping[str, np.ndarray],
    stream_metadata: Mapping[str, Mapping[str, object]],
    authoritative_individual: Mapping[
        Tuple[str, int], Mapping[str, float]
    ],
    config,
) -> Dict[str, object]:
    dataset_keys = tuple(str(key) for key in scope["dataset_keys"])
    datasets_here = [datasets[key] for key in dataset_keys]
    predictions_here = [predictions[key] for key in dataset_keys]
    observed, b_mean, b_cov, s_unit = build_combined_components(
        float(mass),
        datasets_here,
        predictions_here,
        config=config,
    )
    solver = CachedAsymptoticCombinedLimit(
        b_mean,
        b_cov,
        s_unit,
        alpha=float(config.cls_alpha),
        combined_mode=str(config.combined_mode),
    )
    mass_mev = int(round(1000.0 * float(mass)))
    if str(scope["scope_type"]) == "standalone":
        authoritative = authoritative_individual[(dataset_keys[0], mass_mev)]
        likelihood_amplitude_per_eps2 = float(
            A_from_epsilon2(
                datasets_here[0],
                float(mass),
                1.0,
                predictions_here[0].integral_density,
            )
        )
        reported_amplitude_per_eps2 = float(
            authoritative["A_obs"] / authoritative["eps2_obs"]
        )
        observed_limit_solved = standalone_native_limit_cached(
            solver,
            observed,
            likelihood_amplitude_per_eps2,
            reported_amplitude_per_eps2,
        )
        observed_limit = float(authoritative["eps2_obs"])
        observed_amplitude_solved = float(
            observed_limit_solved * reported_amplitude_per_eps2
        )
        limit_root_convention = "standalone_native_amplitude_coordinate"
    else:
        likelihood_amplitude_per_eps2 = None
        reported_amplitude_per_eps2 = None
        observed_limit_solved = float(solver.limit(observed))
        authoritative = None
        observed_limit = observed_limit_solved
        observed_amplitude_solved = float("nan")
        limit_root_convention = "combined_epsilon2_coordinate"
    toy_limits = np.empty(N_TOYS_PER_MASS, dtype=float)
    for toy_index in range(N_TOYS_PER_MASS):
        toy_observed = np.concatenate(
            [count_draws[key][toy_index] for key in dataset_keys]
        )
        if likelihood_amplitude_per_eps2 is None:
            toy_limits[toy_index] = solver.limit(toy_observed)
        else:
            toy_limits[toy_index] = standalone_native_limit_cached(
                solver,
                toy_observed,
                likelihood_amplitude_per_eps2,
                reported_amplitude_per_eps2,
            )

    finite = toy_limits[np.isfinite(toy_limits)]
    n_finite = int(finite.size)
    q02, q16, q50, q84, q97 = quantiles(toy_limits)
    mean_limit = float(np.mean(finite)) if n_finite else float("nan")
    if n_finite and np.isfinite(observed_limit):
        n_strong = int(np.count_nonzero(finite <= observed_limit))
        n_weak = int(np.count_nonzero(finite >= observed_limit))
        n_equal = int(np.count_nonzero(finite == observed_limit))
        p_strong = float(n_strong / n_finite)
        p_weak = float(n_weak / n_finite)
        p_two = float(
            bounded_two_sided_tail_pvalue(p_strong, p_weak)
        )
    else:
        n_strong = n_weak = n_equal = 0
        p_strong = p_weak = p_two = float("nan")

    try:
        p0_solved, z0_solved, _, _ = p0_profiled_gaussian_LRT(
            observed,
            b_mean,
            b_cov,
            s_unit / float(config.eps2_lrt_scale),
        )
    except Exception:
        p0_solved = z0_solved = float("nan")
    if authoritative is not None:
        p0 = float(authoritative["p0_analytic"])
        z0 = float(authoritative["Z_analytic"])
    else:
        p0 = float(p0_solved)
        z0 = float(z0_solved)

    selected_metadata = [metadata[key] for key in dataset_keys]
    return {
        "scope_key": scope_key,
        "scope_type": str(scope["scope_type"]),
        "scope_label": str(scope["label"]),
        "dataset_set": "+".join(dataset_keys),
        "n_datasets": len(dataset_keys),
        "mass_GeV": float(mass),
        "mass_MeV": mass_mev,
        "scope_mass_low_MeV": int(scope["mass_low_MeV"]),
        "scope_mass_high_MeV": int(scope["mass_high_MeV"]),
        "sigma_mass_res_GeV": float(
            np.mean(
                [prediction.sigma_val for prediction in predictions_here]
            )
        ),
        "sigma_mass_res_min_GeV": float(
            np.min(
                [prediction.sigma_val for prediction in predictions_here]
            )
        ),
        "cls_alpha": float(config.cls_alpha),
        "eps2_obs": observed_limit,
        "eps2_obs_solved": observed_limit_solved,
        "eps2_obs_source": (
            "authoritative_individual_reviewed_v4p2"
            if authoritative is not None
            else "pairwise_cached_count_scale"
        ),
        "A_obs_source": (
            float(authoritative["A_obs"])
            if authoritative is not None
            else float("nan")
        ),
        "A_obs_solved": observed_amplitude_solved,
        "likelihood_amplitude_per_eps2": (
            float(likelihood_amplitude_per_eps2)
            if likelihood_amplitude_per_eps2 is not None
            else float("nan")
        ),
        "reported_amplitude_per_eps2": (
            float(reported_amplitude_per_eps2)
            if reported_amplitude_per_eps2 is not None
            else float("nan")
        ),
        "reported_conversion_source": (
            "authoritative_individual_A_up_over_eps2_up"
            if authoritative is not None
            else "accepted_v4p2_combined_config"
        ),
        "p0_analytic": float(p0),
        "Z_analytic": float(z0),
        "p0_analytic_solved": float(p0_solved),
        "Z_analytic_solved": float(z0_solved),
        "eps2_lo2": q02,
        "eps2_lo1": q16,
        "eps2_med": q50,
        "eps2_hi1": q84,
        "eps2_hi2": q97,
        "eps2_mean": mean_limit,
        "p_strong": p_strong,
        "p_weak": p_weak,
        "p_two": p_two,
        "tail_count_strong_le_observed": n_strong,
        "tail_count_weak_ge_observed": n_weak,
        "tail_count_equal_observed": n_equal,
        "tail_count_two_sided_min": min(n_strong, n_weak),
        "empirical_tail_resolution": (
            float(1.0 / n_finite) if n_finite else float("nan")
        ),
        "n_toys_requested": N_TOYS_PER_MASS,
        "n_toys_finite": n_finite,
        "parent_draw_count": ACCEPTED_PARENT_DRAW_COUNT,
        "selected_parent_toy_low": 0,
        "selected_parent_toy_high": N_TOYS_PER_MASS - 1,
        "toy_streams_by_dataset": json.dumps(
            {key: stream_metadata[key] for key in dataset_keys},
            sort_keys=True,
        ),
        "toy_index_shared_within_scope": True,
        "toy_dataset_stream_reused_across_scopes": True,
        "meta": json.dumps(selected_metadata, sort_keys=True),
        "gp_state_sha256_by_dataset": json.dumps(
            {
                item["key"]: item["state_sha256"]
                for item in selected_metadata
            },
            sort_keys=True,
        ),
        "gp_lml_by_dataset": json.dumps(
            {item["key"]: item["lml"] for item in selected_metadata},
            sort_keys=True,
        ),
        "gp_ls_opt_by_dataset": json.dumps(
            {item["key"]: item["ls_opt"] for item in selected_metadata},
            sort_keys=True,
        ),
        "gp_const_opt_by_dataset": json.dumps(
            {item["key"]: item["const_opt"] for item in selected_metadata},
            sort_keys=True,
        ),
        "cls_statistic": "tilde_q_mu",
        "cls_calibration": "asymptotic",
        "combined_mode": "count_scale",
        "bands_refit_gp_on_toy": False,
        "bands_train_exclude_nsigma": float(
            config.gp_train_exclude_nsigma
        ),
        "bands_refit_restarts": 0,
        "bands_refit_optimize": False,
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "limit_solver": CACHE_ALGORITHM_VERSION,
        "limit_root_convention": limit_root_convention,
        "profile_cache_limit_calls": solver.counters.limit_calls,
        "profile_cache_asimov_fixed_nodes": solver.asimov_fixed_cache_size,
        "profile_cache_asimov_fixed_hits": (
            solver.counters.asimov_fixed_cache_hits
        ),
        "profile_cache_asimov_fixed_misses": (
            solver.counters.asimov_fixed_cache_misses
        ),
    }


def reconstruct_parent_closure_row(
    mass: float,
    active_keys: Sequence[str],
    datasets: Mapping[str, object],
    predictions: Mapping[str, object],
    parent_count_draws: Mapping[str, np.ndarray],
    config,
) -> Dict[str, object]:
    datasets_here = [datasets[key] for key in active_keys]
    predictions_here = [predictions[key] for key in active_keys]
    observed, b_mean, b_cov, s_unit = build_combined_components(
        float(mass),
        datasets_here,
        predictions_here,
        config=config,
    )
    solver = CachedAsymptoticCombinedLimit(
        b_mean,
        b_cov,
        s_unit,
        alpha=float(config.cls_alpha),
        combined_mode=str(config.combined_mode),
    )
    eps2_obs = float(solver.limit(observed))
    toy_limits = np.empty(ACCEPTED_PARENT_DRAW_COUNT, dtype=float)
    for toy_index in range(ACCEPTED_PARENT_DRAW_COUNT):
        toy_observed = np.concatenate(
            [parent_count_draws[key][toy_index] for key in active_keys]
        )
        toy_limits[toy_index] = solver.limit(toy_observed)
    finite = toy_limits[np.isfinite(toy_limits)]
    q02, q16, q50, q84, q97 = quantiles(toy_limits)
    return {
        "mass_GeV": float(mass),
        "mass_MeV": int(round(1000.0 * float(mass))),
        "dataset_set": "+".join(active_keys),
        "eps2_obs": eps2_obs,
        "eps2_lo2": q02,
        "eps2_lo1": q16,
        "eps2_med": q50,
        "eps2_hi1": q84,
        "eps2_hi2": q97,
        "eps2_mean": float(np.mean(finite)),
        "n_toys_finite": int(finite.size),
        "tail_count_strong_le_observed": int(
            np.count_nonzero(finite <= eps2_obs)
        ),
        "tail_count_weak_ge_observed": int(
            np.count_nonzero(finite >= eps2_obs)
        ),
        "tail_count_equal_observed": int(
            np.count_nonzero(finite == eps2_obs)
        ),
    }


def run_mass(
    mass: float,
    datasets: Mapping[str, object],
    config,
    fixed: Mapping[Tuple[str, int], Mapping[str, float]],
    accepted_hashes: Mapping[Tuple[str, int], str],
    authoritative_individual: Mapping[
        Tuple[str, int], Mapping[str, float]
    ],
) -> Dict[str, object]:
    scopes_here = scopes_at_mass(mass)
    active_keys = [
        key
        for key in CANONICAL_DATASET_ORDER
        if key
        in {
            dataset.key
            for dataset in active_datasets_for_mass(
                float(mass), dict(datasets), config
            )
        }
    ]
    required_keys = {
        str(dataset_key)
        for _, scope in scopes_here
        for dataset_key in scope["dataset_keys"]
    }
    if not required_keys.issubset(set(active_keys)):
        raise RuntimeError(
            f"Scope requests inactive datasets at {mass:.3f} GeV: "
            f"{sorted(required_keys.difference(active_keys))}"
        )
    if not active_keys:
        raise RuntimeError(f"No active datasets at {mass:.3f} GeV")
    predictions, metadata = build_predictions_for_mass(
        mass,
        active_keys,
        datasets,
        config,
        fixed,
        accepted_hashes,
    )
    count_draws, stream_metadata = parent_toy_counts(
        mass,
        active_keys,
        predictions,
        config,
    )
    rows = [
        run_scope(
            mass,
            scope_key,
            scope,
            datasets,
            predictions,
            metadata,
            count_draws,
            stream_metadata,
            authoritative_individual,
            config,
        )
        for scope_key, scope in scopes_here
    ]
    mass_mev = int(round(1000.0 * float(mass)))
    parent_closure = None
    if mass_mev in REPRESENTATIVE_PARENT_CLOSURE_MEV:
        parent_closure = reconstruct_parent_closure_row(
            mass,
            active_keys,
            datasets,
            predictions,
            count_draws,
            config,
        )
    return {"rows": rows, "parent_closure": parent_closure}


def validate_output(
    bands: pd.DataFrame,
    individual_path: Path,
    parent_closure_rows: Sequence[Mapping[str, object]],
    accepted_combined_path: Path,
) -> Dict[str, object]:
    expected_counts = {
        scope_key: len(scope_masses(scope))
        for scope_key, scope in SCOPES.items()
    }
    found_counts = bands.groupby("scope_key").size().to_dict()
    if found_counts != expected_counts:
        raise RuntimeError(
            f"Scope-row closure failed: {found_counts} != {expected_counts}"
        )
    expected_total = int(sum(expected_counts.values()))
    if len(bands) != expected_total or expected_total != 639:
        raise RuntimeError(
            f"Expected 639 scope-mass rows, found {len(bands)}"
        )
    if not bool((bands["n_toys_requested"] == N_TOYS_PER_MASS).all()):
        raise RuntimeError("Requested toy-count closure failed")
    if not bool((bands["n_toys_finite"] == N_TOYS_PER_MASS).all()):
        bad = bands.loc[
            bands["n_toys_finite"] != N_TOYS_PER_MASS,
            ["scope_key", "mass_MeV", "n_toys_finite"],
        ]
        raise RuntimeError(
            "Non-finite toy limits were produced:\n"
            + bad.to_string(index=False)
        )
    if not bool(
        (
            bands["parent_draw_count"]
            == ACCEPTED_PARENT_DRAW_COUNT
        ).all()
    ):
        raise RuntimeError("Accepted 300-draw parent-count closure failed")
    if not bool((bands["selected_parent_toy_low"] == 0).all()):
        raise RuntimeError("Selected parent-toy lower-index closure failed")
    if not bool(
        (
            bands["selected_parent_toy_high"]
            == N_TOYS_PER_MASS - 1
        ).all()
    ):
        raise RuntimeError("Selected parent-toy upper-index closure failed")
    numeric = [
        "eps2_obs",
        "eps2_lo2",
        "eps2_lo1",
        "eps2_med",
        "eps2_hi1",
        "eps2_hi2",
    ]
    if not np.isfinite(bands[numeric].to_numpy(float)).all():
        raise RuntimeError("Non-finite observed limit or band quantile")
    if not bool((bands[numeric] > 0.0).all().all()):
        raise RuntimeError("Non-positive observed limit or band quantile")
    ordered = (
        (bands["eps2_lo2"] <= bands["eps2_lo1"])
        & (bands["eps2_lo1"] <= bands["eps2_med"])
        & (bands["eps2_med"] <= bands["eps2_hi1"])
        & (bands["eps2_hi1"] <= bands["eps2_hi2"])
    )
    if not bool(ordered.all()):
        raise RuntimeError("Band quantiles are not ordered")

    for row in bands.itertuples(index=False):
        n_finite = int(row.n_toys_finite)
        n_strong = int(row.tail_count_strong_le_observed)
        n_weak = int(row.tail_count_weak_ge_observed)
        n_equal = int(row.tail_count_equal_observed)
        if n_strong + n_weak - n_equal != n_finite:
            raise RuntimeError(
                f"Tail-count partition failed for {row.scope_key} "
                f"at {row.mass_MeV} MeV"
            )
        if int(row.tail_count_two_sided_min) != min(n_strong, n_weak):
            raise RuntimeError("Two-sided tail-count closure failed")

    individual = pd.read_csv(individual_path, dtype={"dataset": str})
    standalone = bands[bands["scope_type"] == "standalone"].copy()
    comparisons: List[Dict[str, object]] = []
    for dataset_key in ("2015", "2016", "2021"):
        left = standalone[
            standalone["dataset_set"].astype(str) == dataset_key
        ][
            [
                "mass_MeV",
                "eps2_obs",
                "eps2_obs_solved",
                "A_obs_source",
                "A_obs_solved",
            ]
        ]
        right = individual[
            individual["dataset"].astype(str) == dataset_key
        ][["mass_MeV", "A_up", "eps2_up"]].copy()
        right["mass_MeV"] = right["mass_MeV"].astype(int)
        merged = left.merge(
            right,
            on="mass_MeV",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not bool((merged["_merge"] == "both").all()):
            raise RuntimeError(
                f"Standalone observed-grid mismatch for {dataset_key}"
            )
        authoritative_delta = np.abs(
            merged["eps2_obs"].to_numpy(float)
            - merged["eps2_up"].to_numpy(float)
        )
        if not np.array_equal(
            merged["eps2_obs"].to_numpy(float),
            merged["eps2_up"].to_numpy(float),
        ):
            raise RuntimeError(
                f"Authoritative standalone observed values drifted for "
                f"{dataset_key}"
            )
        delta = np.abs(
            merged["eps2_obs_solved"].to_numpy(float)
            - merged["eps2_up"].to_numpy(float)
        )
        relative = delta / np.maximum(
            np.abs(merged["eps2_up"].to_numpy(float)),
            1.0e-30,
        )
        max_relative = float(np.max(relative))
        if max_relative > STANDALONE_OBSERVED_CLOSURE_RTOL:
            raise RuntimeError(
                f"Standalone observed-limit closure failed for {dataset_key}: "
                f"max relative delta={max_relative:.6g}"
            )
        amplitude_source_delta = np.abs(
            merged["A_obs_source"].to_numpy(float)
            - merged["A_up"].to_numpy(float)
        )
        if not np.array_equal(
            merged["A_obs_source"].to_numpy(float),
            merged["A_up"].to_numpy(float),
        ):
            raise RuntimeError(
                f"Authoritative standalone amplitudes drifted for "
                f"{dataset_key}"
            )
        amplitude_delta = np.abs(
            merged["A_obs_solved"].to_numpy(float)
            - merged["A_up"].to_numpy(float)
        )
        amplitude_relative = amplitude_delta / np.maximum(
            np.abs(merged["A_up"].to_numpy(float)),
            1.0e-30,
        )
        max_amplitude_relative = float(np.max(amplitude_relative))
        if max_amplitude_relative > STANDALONE_OBSERVED_CLOSURE_RTOL:
            raise RuntimeError(
                f"Standalone observed-amplitude closure failed for "
                f"{dataset_key}: max relative delta="
                f"{max_amplitude_relative:.6g}"
            )
        comparisons.append(
            {
                "dataset": dataset_key,
                "n_masses": len(merged),
                "authoritative_max_abs_delta_eps2": float(
                    np.max(authoritative_delta)
                ),
                "max_abs_delta_eps2": float(np.max(delta)),
                "max_relative_delta": max_relative,
                "authoritative_max_abs_delta_A": float(
                    np.max(amplitude_source_delta)
                ),
                "max_abs_delta_A": float(np.max(amplitude_delta)),
                "max_relative_delta_A": max_amplitude_relative,
            }
        )

    if len(parent_closure_rows) != len(REPRESENTATIVE_PARENT_CLOSURE_MEV):
        raise RuntimeError(
            "Representative accepted-parent closure does not contain five rows"
        )
    accepted = pd.read_csv(accepted_combined_path)
    accepted["mass_MeV"] = np.rint(
        1000.0 * accepted["mass_GeV"].to_numpy(float)
    ).astype(int)
    parent_checks: List[Dict[str, object]] = []
    float_columns = (
        "eps2_obs",
        "eps2_lo2",
        "eps2_lo1",
        "eps2_med",
        "eps2_hi1",
        "eps2_hi2",
        "eps2_mean",
    )
    count_columns = (
        "tail_count_strong_le_observed",
        "tail_count_weak_ge_observed",
        "tail_count_equal_observed",
    )
    for reconstructed in sorted(
        parent_closure_rows, key=lambda row: int(row["mass_MeV"])
    ):
        mass_mev = int(reconstructed["mass_MeV"])
        selected = accepted.loc[accepted["mass_MeV"] == mass_mev]
        if len(selected) != 1:
            raise RuntimeError(
                f"Accepted table has {len(selected)} rows at {mass_mev} MeV"
            )
        reference = selected.iloc[0]
        if str(reconstructed["dataset_set"]) != str(reference["dataset_set"]):
            raise RuntimeError(
                f"Parent active-set closure failed at {mass_mev} MeV"
            )
        if int(reconstructed["n_toys_finite"]) != ACCEPTED_PARENT_DRAW_COUNT:
            raise RuntimeError(
                f"Parent finite-toy closure failed at {mass_mev} MeV"
            )
        deltas: Dict[str, float] = {}
        for column in float_columns:
            reconstructed_value = float(reconstructed[column])
            reference_value = float(reference[column])
            if not np.isclose(
                reconstructed_value,
                reference_value,
                rtol=2.0e-12,
                atol=1.0e-15,
            ):
                raise RuntimeError(
                    f"Accepted parent-stream {column} closure failed at "
                    f"{mass_mev} MeV: {reconstructed_value:.17g} != "
                    f"{reference_value:.17g}"
                )
            deltas[column] = abs(reconstructed_value - reference_value)
        for column in count_columns:
            if int(reconstructed[column]) != int(reference[column]):
                raise RuntimeError(
                    f"Accepted parent-stream {column} closure failed at "
                    f"{mass_mev} MeV"
                )
        parent_checks.append(
            {
                "mass_MeV": mass_mev,
                "dataset_set": str(reconstructed["dataset_set"]),
                "max_abs_float_delta": float(max(deltas.values())),
                "tail_counts_exact": True,
            }
        )

    return {
        "status": "PASS",
        "scope_row_counts": expected_counts,
        "n_scope_mass_rows": expected_total,
        "n_toys_per_scope_mass": N_TOYS_PER_MASS,
        "n_finite_toy_limits": expected_total * N_TOYS_PER_MASS,
        "standalone_observed_limit_closure": comparisons,
        "accepted_parent_stream_reconstruction": parent_checks,
        "quantiles_finite_positive_ordered": True,
        "state_hashes_exact": True,
        "selected_parent_toy_indices": [0, N_TOYS_PER_MASS - 1],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--reviewed-state-csv",
        type=Path,
        default=DEFAULT_REVIEWED,
    )
    parser.add_argument(
        "--closure-report",
        type=Path,
        default=DEFAULT_CLOSURE,
    )
    parser.add_argument(
        "--accepted-combined-table",
        type=Path,
        default=DEFAULT_ACCEPTED_COMBINED,
    )
    parser.add_argument(
        "--individual-observed-table",
        type=Path,
        default=DEFAULT_INDIVIDUAL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required acknowledgement for the six-scope 100-toy run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.confirm_production:
        raise SystemExit(
            "Run not started. Pass --confirm-production after reviewing the "
            "six declared standalone/pairwise scopes."
        )
    if tuple(THREAD_ENV_KEYS) != tuple(PRODUCTION_THREAD_ENV_KEYS):
        raise SystemExit("Thread-control key drift from the production runner")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    for key in THREAD_ENV_KEYS:
        if os.environ.get(key) != "1":
            raise SystemExit(f"{key}=1 is required, got {os.environ.get(key)!r}")

    paths = {
        "config": args.config.expanduser().resolve(),
        "reviewed": args.reviewed_state_csv.expanduser().resolve(),
        "closure": args.closure_report.expanduser().resolve(),
        "accepted_combined": args.accepted_combined_table.expanduser().resolve(),
        "individual": args.individual_observed_table.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing required {label} file: {path}")
    if sha256(paths["config"]) != EXPECTED_CONFIG_SHA256:
        raise SystemExit("The physics config is not the accepted v4.2 card")
    if sha256(paths["reviewed"]) != EXPECTED_REVIEWED_SHA256:
        raise SystemExit("The reviewed-state ledger is not the accepted source")
    if (
        sha256(paths["accepted_combined"])
        != EXPECTED_ACCEPTED_COMBINED_SHA256
    ):
        raise SystemExit(
            "The combined table is not the accepted v4.2 300-toy source"
        )
    if sha256(paths["individual"]) != EXPECTED_INDIVIDUAL_SHA256:
        raise SystemExit(
            "The individual table is not the accepted v4.2 observed source"
        )

    config = replace(
        load_config(str(paths["config"])),
        ul_bands_n_workers=int(args.workers),
        ul_bands_parallel_backend="loky",
        ul_bands_threads_per_worker=1,
    )
    validate_v4_geometry(config)
    if str(config.combined_mode).lower().strip() != "count_scale":
        raise SystemExit("The six-scope study requires combined_mode=count_scale")
    if str(config.cls_mode).lower().strip() != "asymptotic":
        raise SystemExit("The six-scope study requires asymptotic inner CLs")
    if not np.isclose(float(config.cls_alpha), 0.1, rtol=0.0, atol=0.0):
        raise SystemExit("The six-scope study requires 90% CL")
    if int(config.combined_bands_n_toys) != 300:
        raise SystemExit(
            "The accepted base card must remain the original 300-toy "
            "all-dataset card; this script is an explicit derived 100-toy study"
        )
    closure = validate_closure_report(
        paths["closure"],
        paths["config"],
        paths["reviewed"],
    )
    fixed = load_fixed_coordinates(paths["reviewed"])
    accepted_hashes = load_accepted_state_hashes(paths["accepted_combined"])
    authoritative_individual = load_authoritative_individual_rows(
        paths["individual"]
    )
    datasets = make_datasets(config)
    if set(datasets) != {"2015", "2016", "2021"}:
        raise SystemExit(f"Unexpected enabled dataset set: {sorted(datasets)}")

    masses = full_requested_mass_grid()
    started = time.time()
    if args.workers == 1:
        mass_results = [
            run_mass(
                mass,
                datasets,
                config,
                fixed,
                accepted_hashes,
                authoritative_individual,
            )
            for mass in masses
        ]
    else:
        mass_results = joblib.Parallel(
            n_jobs=int(args.workers),
            backend="loky",
        )(
            joblib.delayed(run_mass)(
                mass,
                datasets,
                config,
                fixed,
                accepted_hashes,
                authoritative_individual,
            )
            for mass in masses
        )
    elapsed = time.time() - started
    rows = [
        row
        for mass_result in mass_results
        for row in mass_result["rows"]
    ]
    parent_closure_rows = [
        mass_result["parent_closure"]
        for mass_result in mass_results
        if mass_result["parent_closure"] is not None
    ]
    scope_order = {key: index for index, key in enumerate(SCOPES)}
    bands = pd.DataFrame(rows)
    bands["_scope_order"] = bands["scope_key"].map(scope_order)
    bands = (
        bands.sort_values(["_scope_order", "mass_GeV"])
        .drop(columns="_scope_order")
        .reset_index(drop=True)
    )
    validation = validate_output(
        bands,
        paths["individual"],
        parent_closure_rows,
        paths["accepted_combined"],
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ul_bands_standalone_pairwise_100.csv"
    validation_path = output_dir / "validation_standalone_pairwise_100.json"
    provenance_path = output_dir / "provenance_standalone_pairwise_100.json"
    bands.to_csv(csv_path, index=False, float_format="%.17g")
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "status": "PASS",
        "cache_algorithm_version": CACHE_ALGORITHM_VERSION,
        "physics_config": str(paths["config"]),
        "physics_config_sha256": sha256(paths["config"]),
        "base_card_toy_scope": (
            "accepted all-dataset 300-toy card; unchanged"
        ),
        "derived_study_scope": (
            "three standalone and three genuine pairwise-intersection "
            "conditional 100-toy bands"
        ),
        "reviewed_state_csv": str(paths["reviewed"]),
        "reviewed_state_csv_sha256": sha256(paths["reviewed"]),
        "accepted_combined_table": str(paths["accepted_combined"]),
        "accepted_combined_table_sha256": sha256(paths["accepted_combined"]),
        "individual_observed_table": str(paths["individual"]),
        "individual_observed_table_sha256": sha256(paths["individual"]),
        "closure_report": str(paths["closure"]),
        "closure_report_sha256": sha256(paths["closure"]),
        "closure_mass_results": closure["mass_results"],
        "scopes": {
            key: {
                **dict(scope),
                "dataset_keys": list(scope["dataset_keys"]),
                "n_masses": len(scope_masses(scope)),
            }
            for key, scope in SCOPES.items()
        },
        "pairwise_mass_policy": (
            "intersection only; both named datasets active at every pairwise mass"
        ),
        "n_scope_mass_rows": int(len(bands)),
        "n_toys_per_scope_mass": N_TOYS_PER_MASS,
        "accepted_parent_draw_count": ACCEPTED_PARENT_DRAW_COUNT,
        "selected_parent_toy_indices": [
            SELECTED_PARENT_TOY_INDICES[0],
            SELECTED_PARENT_TOY_INDICES[-1],
        ],
        "n_finite_toy_limits": int(
            len(bands) * N_TOYS_PER_MASS
        ),
        "root_seed": ACCEPTED_ROOT_SEED,
        "mass_seed_rule": (
            "SeedSequence(24680).spawn(232)[mass_MeV-19]"
        ),
        "parent_stream_rule": (
            "consume accepted pre-draw unused inner seed; generate all active "
            "300-row MVN arrays in canonical 2015,2016,2021 order; then "
            "generate all Poisson arrays in that order; select rows 0-99"
        ),
        "toy_index_policy": (
            "exact first 100 rows of the accepted v4.2 300-draw parent stream; "
            "dataset pseudo-spectrum reused across every scope containing that "
            "dataset; pair members joined by common toy index"
        ),
        "representative_parent_stream_closure": (
            validation["accepted_parent_stream_reconstruction"]
        ),
        "mass_independence": True,
        "refit_gp_on_toy": False,
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "inner_cls": "asymptotic tilde_q_mu, alpha=0.1",
        "combined_mode": "count_scale",
        "standalone_limit_root_convention": (
            "reviewed native total-amplitude bisection: 40 iterations, "
            "|CLs-alpha|<1e-6, relative amplitude bracket 1e-6"
        ),
        "standalone_reported_eps2_conversion": (
            "authoritative reviewed A_up/eps2_up conversion snapshot at each "
            "dataset-mass state; toy amplitude limits use the same factor"
        ),
        "standalone_observed_closure_rtol": (
            STANDALONE_OBSERVED_CLOSURE_RTOL
        ),
        "pairwise_limit_root_convention": (
            "accepted v4.2 combined epsilon2-coordinate bisection"
        ),
        "coverage_calibrated": False,
        "scan_toy_calibrated": False,
        "parallel_workers": int(args.workers),
        "parallel_backend": "loky",
        "threads_per_worker": 1,
        "elapsed_seconds": float(elapsed),
        "output_csv": str(csv_path),
        "output_csv_sha256": sha256(csv_path),
        "validation": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cached_solver": str((HERE / "cached_profile_solver.py").resolve()),
        "cached_solver_sha256": sha256(HERE / "cached_profile_solver.py"),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "n_scope_mass_rows": len(bands),
                "n_finite_toy_limits": len(bands) * N_TOYS_PER_MASS,
                "elapsed_seconds": elapsed,
                "csv": str(csv_path),
                "csv_sha256": sha256(csv_path),
                "validation": str(validation_path),
                "provenance": str(provenance_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
