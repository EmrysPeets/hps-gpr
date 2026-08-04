#!/usr/bin/env python3
"""Reconstruct the v4.1 combined observed result from 415 reviewed GP states.

This is an observed-only runner. It reconstructs each GP prediction at the
exact reviewed ``const_opt`` and ``ls_opt`` coordinates, evaluates the shared
``count_scale`` likelihood, and writes the 90% asymptotic CLs upper limit plus
the local asymptotic p0/Z at all 232 masses. It never draws pseudoexperiments
and never creates expected-limit bands.

Use ``--reference-closure`` to compare the cached and repository-reference
observed-limit implementations bit-for-bit at 20, 40, and 60 MeV, which cover
one-, two-, and three-dataset active sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_key in THREAD_ENV_KEYS:
    os.environ.setdefault(_thread_key, "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    "/tmp/hps-gpr-v4p1-k12-combined-observed-mpl",
)

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V4_RUNNER_DIR = (
    REPO
    / "study_results"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V4_RUNNER_DIR))

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
from hps_gpr.statistics import p0_profiled_gaussian_LRT  # noqa: E402
from run_combined_bands_cached_fixed_reviewed import (  # noqa: E402
    EXPECTED_FULL_GRID_GP_STATES,
    build_fixed_predictions,
    full_mass_grid,
    load_reviewed_coordinates,
    sha256,
    validate_v4_geometry,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - production environment provides it.
    import contextlib

    threadpool_limits = contextlib.nullcontext


DEFAULT_CONFIG = (
    REPO
    / "study_configs"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "config_obsUL90_combined_wide_support_v4p1_2016k12_observed_only.yaml"
)
DEFAULT_OUTPUT_CSV = HERE / "combined_observed_fixed_reviewed.csv"
DEFAULT_PROVENANCE_JSON = (
    HERE / "combined_observed_fixed_reviewed_provenance.json"
)
DEFAULT_REFERENCE_CLOSURE_MASSES_MEV = (20, 40, 60)
EXPECTED_DATASET_STATE_COUNTS = {"2015": 72, "2016": 142, "2021": 201}
EXPECTED_UPPER_FACTORS = {"2015": 8.0, "2016": 12.0, "2021": 15.0}
EXPECTED_MASS_COUNT = 232
OBSERVED_RUNNER_VERSION = "fixed_reviewed_combined_observed_v1"


def _bool_column_has_true(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return bool(series.fillna(False).any())
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return bool(normalized.isin({"true", "1", "yes"}).any())


def validate_observed_only_config(config, config_path: Path) -> dict:
    """Fail closed unless the supplied card is the isolated observed-only k12 card."""

    validate_v4_geometry(config)
    if not np.isclose(float(config.cls_alpha), 0.1, rtol=0.0, atol=0.0):
        raise RuntimeError(f"Expected 90% CL (alpha=0.1), got {config.cls_alpha}.")
    if str(config.cls_mode).lower().strip() != "asymptotic":
        raise RuntimeError("Observed runner requires cls_mode=asymptotic.")
    if int(config.cls_num_toys) != 0:
        raise RuntimeError("Observed runner requires cls_num_toys=0.")
    if not bool(config.do_combined):
        raise RuntimeError("Observed runner requires do_combined=true.")
    if str(config.combined_mode).lower().strip() != "count_scale":
        raise RuntimeError("Observed runner requires combined_mode=count_scale.")
    if str(config.run_limit_bands_on).strip():
        raise RuntimeError("Observed runner requires run_limit_bands_on to be empty.")

    false_switches = (
        "make_ul_bands",
        "do_combined_bands",
        "make_eps2_bands",
    )
    for key in false_switches:
        if bool(getattr(config, key)):
            raise RuntimeError(f"Observed runner requires {key}=false.")
    zero_counts = ("ul_bands_toys", "combined_bands_n_toys")
    for key in zero_counts:
        if int(getattr(config, key)) != 0:
            raise RuntimeError(f"Observed runner requires {key}=0.")

    for dataset_key in ("2015", "2016", "2021"):
        if not bool(getattr(config, f"enable_{dataset_key}")):
            raise RuntimeError(f"Dataset {dataset_key} must remain enabled.")
        visibility = str(config.data_visibility.get(dataset_key, "")).lower()
        if visibility != "observed":
            raise RuntimeError(
                f"Dataset {dataset_key} visibility must be observed, got {visibility!r}."
            )

    found_factors = {
        str(key): float(value)
        for key, value in config.kernel_ls_res_upper_factor_by_dataset.items()
    }
    for dataset_key, expected in EXPECTED_UPPER_FACTORS.items():
        found = found_factors.get(dataset_key)
        if found is None or not np.isclose(
            found,
            expected,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"Unexpected {dataset_key} upper factor: {found} != {expected}."
            )

    if Path(str(config.output_dir)).resolve() != HERE.resolve():
        raise RuntimeError(
            f"Config output_dir must be the isolated final directory {HERE}."
        )

    return {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "cls_alpha": float(config.cls_alpha),
        "cls_mode": str(config.cls_mode),
        "combined_mode": str(config.combined_mode),
        "kernel_ls_res_upper_factor_by_dataset": found_factors,
        "toy_draws": 0,
        "expected_bands_produced": False,
    }


def validate_reviewed_card_alignment(reviewed_csv: Path) -> dict:
    """Require an exact 415-state table aligned with the factor-12 card."""

    reviewed = pd.read_csv(reviewed_csv)
    required = {
        "dataset",
        "mass_GeV",
        "const_opt",
        "ls_opt",
        "lml",
        "ls_hi",
        "ls_hi_over_sigma_x",
    }
    missing = sorted(required.difference(reviewed.columns))
    if missing:
        raise RuntimeError(
            f"Reviewed-state CSV is missing card-alignment columns: {missing}"
        )
    if len(reviewed) != EXPECTED_FULL_GRID_GP_STATES:
        raise RuntimeError(
            f"Expected {EXPECTED_FULL_GRID_GP_STATES} reviewed states, "
            f"found {len(reviewed)}."
        )

    reviewed = reviewed.copy()
    reviewed["dataset"] = reviewed["dataset"].astype(str)
    duplicate = reviewed.duplicated(["dataset", "mass_GeV"], keep=False)
    if bool(duplicate.any()):
        sample = reviewed.loc[duplicate, ["dataset", "mass_GeV"]].head(10)
        raise RuntimeError(
            "Duplicate reviewed GP states:\n" + sample.to_string(index=False)
        )
    counts = {
        str(key): int(value)
        for key, value in reviewed.groupby("dataset").size().items()
    }
    if counts != EXPECTED_DATASET_STATE_COUNTS:
        raise RuntimeError(
            f"Reviewed dataset-state counts {counts} != "
            f"{EXPECTED_DATASET_STATE_COUNTS}."
        )

    numeric_columns = (
        "mass_GeV",
        "const_opt",
        "ls_opt",
        "lml",
        "ls_hi",
        "ls_hi_over_sigma_x",
    )
    for key in numeric_columns:
        values = pd.to_numeric(reviewed[key], errors="coerce").to_numpy(float)
        if not bool(np.isfinite(values).all()):
            raise RuntimeError(f"Reviewed column {key} contains non-finite values.")
    if bool((reviewed["const_opt"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError("Reviewed const_opt values must be positive.")
    if bool((reviewed["ls_opt"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError("Reviewed ls_opt values must be positive.")
    if bool((reviewed["ls_hi"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError("Reviewed ls_hi values must be positive.")
    exceeds = (
        reviewed["ls_opt"].to_numpy(float)
        > reviewed["ls_hi"].to_numpy(float) * (1.0 + 1.0e-9) + 1.0e-12
    )
    if bool(exceeds.any()):
        sample = reviewed.loc[
            exceeds,
            ["dataset", "mass_GeV", "ls_opt", "ls_hi"],
        ].head(10)
        raise RuntimeError(
            "Reviewed ls_opt exceeds the declared upper bound:\n"
            + sample.to_string(index=False)
        )

    factor_summary: Dict[str, dict] = {}
    for dataset_key, expected in EXPECTED_UPPER_FACTORS.items():
        here = reviewed.loc[
            reviewed["dataset"] == dataset_key,
            "ls_hi_over_sigma_x",
        ].to_numpy(float)
        if not bool(
            np.isclose(here, expected, rtol=0.0, atol=1.0e-10).all()
        ):
            raise RuntimeError(
                f"Reviewed {dataset_key} states are not aligned with upper "
                f"factor {expected}; range={np.min(here)}..{np.max(here)}."
            )
        factor_summary[dataset_key] = {
            "expected": expected,
            "min": float(np.min(here)),
            "max": float(np.max(here)),
        }

    if "interpolated" in reviewed.columns and _bool_column_has_true(
        reviewed["interpolated"]
    ):
        raise RuntimeError(
            "Reviewed-state CSV contains interpolated states; unchanged-card "
            "stable reruns are required."
        )
    if "review_status" in reviewed.columns:
        status = reviewed["review_status"].fillna("").astype(str).str.lower()
        bad = status.str.contains(
            "unresolved|interpolated|rejected|failed",
            regex=True,
        )
        if bool(bad.any()):
            sample = reviewed.loc[
                bad,
                ["dataset", "mass_GeV", "review_status"],
            ].head(10)
            raise RuntimeError(
                "Reviewed-state CSV contains non-final review statuses:\n"
                + sample.to_string(index=False)
            )

    return {
        "reviewed_csv": str(reviewed_csv),
        "reviewed_csv_sha256": sha256(reviewed_csv),
        "n_states": int(len(reviewed)),
        "dataset_state_counts": counts,
        "upper_factor_alignment": factor_summary,
        "interpolated_states": 0,
    }


def _state_maps(metadata: Sequence[dict]) -> dict:
    return {
        "gp_lml_by_dataset": json.dumps(
            {item["key"]: item["lml"] for item in metadata},
            sort_keys=True,
        ),
        "gp_ls_opt_by_dataset": json.dumps(
            {item["key"]: item["ls_opt"] for item in metadata},
            sort_keys=True,
        ),
        "gp_const_opt_by_dataset": json.dumps(
            {item["key"]: item["const_opt"] for item in metadata},
            sort_keys=True,
        ),
        "gp_state_sha256_by_dataset": json.dumps(
            {item["key"]: item["state_sha256"] for item in metadata},
            sort_keys=True,
        ),
    }


def run_one_mass(
    mass: float,
    datasets: dict,
    config,
    fixed_here: Dict[str, Dict[str, float]],
    reference_closure_masses_mev: Set[int],
) -> dict:
    """Reconstruct one fixed-reviewed observed mass point."""

    with threadpool_limits(limits=1):
        datasets_here, predictions, metadata = build_fixed_predictions(
            float(mass),
            datasets,
            config,
            fixed_here,
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
        eps2_observed = float(solver.limit(observed))
        p0, z_local, q0, p0_info = p0_profiled_gaussian_LRT(
            observed,
            b_mean,
            b_cov,
            s_unit / float(config.eps2_lrt_scale),
        )

        mass_mev = int(round(float(mass) * 1000.0))
        reference_checked = mass_mev in reference_closure_masses_mev
        reference_limit = float("nan")
        reference_bitwise_equal: object = ""
        if reference_checked:
            reference_limit = float(
                combined_cls_limit_epsilon2_from_vectors(
                    observed,
                    b_mean,
                    b_cov,
                    s_unit,
                    config,
                    mode="asymptotic",
                    num_toys=0,
                )
            )
            reference_bitwise_equal = bool(
                np.array_equal(
                    np.asarray([eps2_observed], dtype=np.float64),
                    np.asarray([reference_limit], dtype=np.float64),
                    equal_nan=True,
                )
            )

    dataset_keys = [dataset.key for dataset in datasets_here]
    row = {
        "dataset_set": "+".join(dataset_keys),
        "n_active_datasets": int(len(dataset_keys)),
        "mass_GeV": float(mass),
        "mass_MeV": mass_mev,
        "sigma_mass_res_GeV": float(
            np.mean([prediction.sigma_val for prediction in predictions])
        ),
        "sigma_mass_res_min_GeV": float(
            np.min([prediction.sigma_val for prediction in predictions])
        ),
        "sigma_mass_res_max_GeV": float(
            np.max([prediction.sigma_val for prediction in predictions])
        ),
        "n_likelihood_bins": int(observed.size),
        "observed_count_sum": int(np.sum(observed)),
        "background_mean_sum": float(np.sum(b_mean)),
        "signal_per_eps2_sum": float(np.sum(s_unit)),
        "cls_alpha": float(config.cls_alpha),
        "eps2_obs": eps2_observed,
        "ul_eps2_obs": eps2_observed,
        "p0_analytic": float(p0),
        "Z_analytic": float(z_local),
        "q0_analytic": float(q0),
        "p0_fit_ok": bool(p0_info.get("ok", False)),
        "lrt_A_hat_scaled": float(p0_info.get("A_hat", float("nan"))),
        "lrt_sigma_A_scaled": float(
            p0_info.get("sigma_A", float("nan"))
        ),
        "eps2_lrt_scale": float(config.eps2_lrt_scale),
        "cls_statistic": "tilde_q_mu",
        "cls_calibration": "asymptotic",
        "combined_mode": "count_scale",
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "expected_bands_produced": False,
        "toy_draws": 0,
        "reference_closure_checked": bool(reference_checked),
        "reference_eps2_obs": reference_limit,
        "reference_bitwise_equal": reference_bitwise_equal,
        "limit_solver": CACHE_ALGORITHM_VERSION,
        "profile_cache_limit_calls": int(solver.counters.limit_calls),
        "profile_cache_asimov_fixed_nodes": int(
            solver.asimov_fixed_cache_size
        ),
        "meta": json.dumps(metadata, sort_keys=True),
        **_state_maps(metadata),
    }
    return row


def _selected_masses(args: argparse.Namespace) -> List[float]:
    if not args.mass_mev:
        return full_mass_grid()
    masses_mev = sorted(set(int(value) for value in args.mass_mev))
    if not args.allow_subset:
        raise SystemExit(
            "--mass-mev is a diagnostic subset interface; pass --allow-subset "
            "to acknowledge that it will not produce the 232-point final table."
        )
    if any(value < 19 or value > 250 for value in masses_mev):
        raise SystemExit("Requested masses must lie on the 19--250 MeV grid.")
    return [value / 1000.0 for value in masses_mev]


def _source_hashes() -> dict:
    paths = {
        "observed_runner": Path(__file__).resolve(),
        "cached_profile_solver": V4_RUNNER_DIR / "cached_profile_solver.py",
        "fixed_reviewed_reconstruction_helpers": (
            V4_RUNNER_DIR / "run_combined_bands_cached_fixed_reviewed.py"
        ),
        "config_module": REPO / "hps_gpr" / "config.py",
        "dataset_module": REPO / "hps_gpr" / "dataset.py",
        "evaluation_module": REPO / "hps_gpr" / "evaluation.py",
        "gpr_module": REPO / "hps_gpr" / "gpr.py",
        "io_module": REPO / "hps_gpr" / "io.py",
        "statistics_module": REPO / "hps_gpr" / "statistics.py",
    }
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }


def _validate_output(
    table: pd.DataFrame,
    masses: Sequence[float],
    *,
    require_reference_closure: bool,
    reference_masses_mev: Set[int],
) -> dict:
    if len(table) != len(masses):
        raise RuntimeError(f"Output has {len(table)} rows for {len(masses)} masses.")
    if bool(table["mass_GeV"].duplicated().any()):
        raise RuntimeError("Output contains duplicate masses.")
    expected = np.asarray(sorted(float(value) for value in masses))
    found = table["mass_GeV"].to_numpy(float)
    if not np.array_equal(found, expected):
        raise RuntimeError("Output mass grid is not the requested exact grid.")

    finite_columns = (
        "eps2_obs",
        "p0_analytic",
        "Z_analytic",
        "q0_analytic",
        "lrt_A_hat_scaled",
        "lrt_sigma_A_scaled",
    )
    for key in finite_columns:
        if not bool(np.isfinite(table[key].to_numpy(float)).all()):
            bad = table.loc[
                ~np.isfinite(table[key].to_numpy(float)),
                ["mass_GeV", key],
            ]
            raise RuntimeError(
                f"Non-finite {key} values:\n" + bad.to_string(index=False)
            )
    if bool((table["eps2_obs"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError("Observed limits must be positive.")
    if bool(
        (
            (table["p0_analytic"].to_numpy(float) < 0.0)
            | (table["p0_analytic"].to_numpy(float) > 1.0)
        ).any()
    ):
        raise RuntimeError("Local asymptotic p0 values lie outside [0, 1].")
    if bool((table["Z_analytic"].to_numpy(float) < 0.0).any()):
        raise RuntimeError("Local asymptotic Z values must be nonnegative.")
    if bool((table["toy_draws"].to_numpy(int) != 0).any()):
        raise RuntimeError("Observed-only output unexpectedly records toy draws.")
    if bool(table["expected_bands_produced"].astype(bool).any()):
        raise RuntimeError("Observed-only output unexpectedly records bands.")

    closure_rows = table.loc[table["reference_closure_checked"].astype(bool)]
    closure_summary: List[dict] = []
    for row in closure_rows.itertuples(index=False):
        cached = float(row.eps2_obs)
        reference = float(row.reference_eps2_obs)
        equal = bool(row.reference_bitwise_equal)
        closure_summary.append(
            {
                "mass_MeV": int(row.mass_MeV),
                "active_datasets": str(row.dataset_set).split("+"),
                "n_active_datasets": int(row.n_active_datasets),
                "cached_eps2_obs": cached,
                "reference_eps2_obs": reference,
                "cached_float_hex": cached.hex(),
                "reference_float_hex": reference.hex(),
                "bitwise_equal": equal,
            }
        )
    if require_reference_closure:
        found_masses = {
            int(entry["mass_MeV"]) for entry in closure_summary
        }
        if found_masses != reference_masses_mev:
            raise RuntimeError(
                f"Reference closure masses {found_masses} != "
                f"{reference_masses_mev}."
            )
        if not all(entry["bitwise_equal"] for entry in closure_summary):
            raise RuntimeError("Cached observed limits failed bitwise closure.")
        active_counts = {
            int(entry["n_active_datasets"]) for entry in closure_summary
        }
        if not {1, 2, 3}.issubset(active_counts):
            raise RuntimeError(
                "Reference closure must cover one-, two-, and three-dataset masses."
            )

    return {
        "n_rows": int(len(table)),
        "n_finite_observed_limits": int(np.isfinite(table["eps2_obs"]).sum()),
        "n_finite_local_p0": int(np.isfinite(table["p0_analytic"]).sum()),
        "n_finite_local_Z": int(np.isfinite(table["Z_analytic"]).sum()),
        "reference_closure_requested": bool(require_reference_closure),
        "reference_closure": closure_summary,
        "toy_draws": 0,
        "expected_bands_produced": False,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reviewed-state-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=DEFAULT_PROVENANCE_JSON,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reference-closure",
        action="store_true",
        help=(
            "Require bitwise cached-vs-reference closure at representative "
            "one-, two-, and three-dataset masses."
        ),
    )
    parser.add_argument(
        "--closure-mass-mev",
        type=int,
        action="append",
        help=(
            "Override a reference-closure mass in MeV. Defaults to 20, 40, "
            "and 60 MeV."
        ),
    )
    parser.add_argument(
        "--mass-mev",
        type=int,
        action="append",
        help="Diagnostic subset mass in MeV; repeat for multiple masses.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Acknowledge that --mass-mev does not produce the final 232 rows.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the card and 415-state review without reconstructing fits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CSV/provenance outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")
    if not args.validate_only:
        for key in THREAD_ENV_KEYS:
            if os.environ.get(key) != "1":
                raise SystemExit(
                    f"{key}=1 is required for deterministic production, "
                    f"got {os.environ.get(key)!r}."
                )

    config_path = args.config.expanduser().resolve()
    reviewed_csv = args.reviewed_state_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    provenance_json = args.provenance_json.expanduser().resolve()
    for path in (config_path, reviewed_csv):
        if not path.is_file():
            raise SystemExit(f"Required file does not exist: {path}")

    config = load_config(str(config_path))
    config_contract = validate_observed_only_config(config, config_path)
    reviewed_alignment = validate_reviewed_card_alignment(reviewed_csv)
    datasets = make_datasets(config)
    complete_masses = full_mass_grid()
    fixed, fixed_provenance = load_reviewed_coordinates(
        reviewed_csv,
        complete_masses,
        datasets,
        config,
    )

    validation_payload = {
        "runner_version": OBSERVED_RUNNER_VERSION,
        "config_contract": config_contract,
        "reviewed_alignment": reviewed_alignment,
        "reviewed_coordinate_provenance": fixed_provenance,
    }
    if args.validate_only:
        print(json.dumps(validation_payload, indent=2, sort_keys=True))
        return

    if not args.overwrite:
        existing = [path for path in (output_csv, provenance_json) if path.exists()]
        if existing:
            raise SystemExit(
                "Refusing to replace existing output(s) without --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    masses = _selected_masses(args)
    closure_masses_mev = set(
        int(value)
        for value in (
            args.closure_mass_mev
            if args.closure_mass_mev
            else DEFAULT_REFERENCE_CLOSURE_MASSES_MEV
        )
    )
    if args.reference_closure:
        selected_mev = {int(round(mass * 1000.0)) for mass in masses}
        if not closure_masses_mev.issubset(selected_mev):
            raise SystemExit(
                "Reference-closure masses must be included in the selected "
                f"output masses; missing {sorted(closure_masses_mev - selected_mev)}."
            )
    else:
        closure_masses_mev = set()

    started = time.time()
    tasks: List[Tuple[float, Dict[str, Dict[str, float]]]] = [
        (mass, fixed[round(float(mass), 12)]) for mass in masses
    ]
    if args.workers == 1:
        rows = [
            run_one_mass(
                mass,
                datasets,
                config,
                fixed_here,
                closure_masses_mev,
            )
            for mass, fixed_here in tasks
        ]
    else:
        rows = joblib.Parallel(n_jobs=int(args.workers), backend="loky")(
            joblib.delayed(run_one_mass)(
                mass,
                datasets,
                config,
                fixed_here,
                closure_masses_mev,
            )
            for mass, fixed_here in tasks
        )
    elapsed = time.time() - started

    table = (
        pd.DataFrame(rows)
        .sort_values("mass_GeV")
        .reset_index(drop=True)
    )
    output_validation = _validate_output(
        table,
        masses,
        require_reference_closure=bool(args.reference_closure),
        reference_masses_mev=closure_masses_mev,
    )
    if not args.mass_mev and len(table) != EXPECTED_MASS_COUNT:
        raise RuntimeError(
            f"Full observed reconstruction must have {EXPECTED_MASS_COUNT} rows."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    provenance = {
        **validation_payload,
        "output_validation": output_validation,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_hashes": _source_hashes(),
        "input_data": {
            dataset_key: {
                "path": str(getattr(config, f"path_{dataset_key}")),
                "histogram": str(getattr(config, f"hist_{dataset_key}")),
            }
            for dataset_key in ("2015", "2016", "2021")
        },
        "mass_grid_GeV": table["mass_GeV"].astype(float).tolist(),
        "n_masses": int(len(table)),
        "parallel_workers": int(args.workers),
        "parallel_backend": "loky" if args.workers > 1 else "sequential",
        "threads_per_worker": 1,
        "elapsed_seconds": float(elapsed),
        "toy_draws": 0,
        "expected_bands_produced": False,
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256(output_csv),
        "output_columns": table.columns.tolist(),
        "provenance_json": str(provenance_json),
        "command_argv": sys.argv,
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
    }
    provenance_json.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(table)} observed-only masses to {output_csv} in "
        f"{elapsed:.2f} s; toys=0, expected bands=false."
    )


if __name__ == "__main__":
    main()
