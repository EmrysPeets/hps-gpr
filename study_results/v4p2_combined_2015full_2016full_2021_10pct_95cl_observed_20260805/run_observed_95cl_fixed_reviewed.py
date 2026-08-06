#!/usr/bin/env python3
"""Recompute the v4.2 simultaneous observed limit at 95% CLs.

This is an observed-only pass.  It reuses the exact reviewed v4.2 GP states,
changes only ``cls_alpha`` from 0.10 to 0.05 in memory, and does not generate
expected-limit pseudoexperiments or refit any GP hyperparameter.
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
from typing import Optional, Sequence


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_key in THREAD_ENV_KEYS:
    os.environ[_thread_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p2-observed95-mpl")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_CAMPAIGN = (
    REPO
    / "study_results"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SOURCE_CAMPAIGN))

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
from hps_gpr.statistics import asymptotic_cls_profiled_gaussian  # noqa: E402
from run_combined_bands_cached_fixed_reviewed import (  # noqa: E402
    build_fixed_predictions,
    full_mass_grid,
    load_reviewed_coordinates,
    sha256,
    validate_closure_report,
    validate_v4_geometry,
)


BASE_CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
REVIEWED_STATES = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
CLOSURE_REPORT = (
    SOURCE_CAMPAIGN / "derived" / "cached_profile_closure_v4p2.json"
)
SOURCE_V4P2_TABLE = (
    SOURCE_CAMPAIGN / "derived" / "combined_bands300_reviewed_v4p2.csv"
)
AUTHORITATIVE_OBSERVED_TABLE = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "final_k12_combined_observed"
    / "combined_observed_fixed_reviewed.csv"
)
OUTPUT_DIR = HERE / "derived"
OUTPUT_CSV = OUTPUT_DIR / "combined_observed_95cl_reviewed_v4p2.csv"
VALIDATION_JSON = OUTPUT_DIR / "validation_observed_95cl_v4p2.json"
PROVENANCE_JSON = OUTPUT_DIR / "provenance_observed_95cl_v4p2.json"

REFERENCE_CLOSURE_MASSES_MEV = (19, 39, 50, 65, 91, 181, 250)
EXPECTED_UPPER_FACTORS = {"2015": 8.0, "2016": 12.0, "2021": 15.0}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--reviewed-state-csv", type=Path, default=REVIEWED_STATES)
    parser.add_argument("--closure-report", type=Path, default=CLOSURE_REPORT)
    parser.add_argument("--source-v4p2-table", type=Path, default=SOURCE_V4P2_TABLE)
    parser.add_argument(
        "--authoritative-observed-table",
        type=Path,
        default=AUTHORITATIVE_OBSERVED_TABLE,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def _source_factor(source_row: pd.Series) -> float:
    eps_ee = float(source_row["eps2_obs_ee_channel"])
    eps_visible = float(source_row["eps2_obs_minimal_visible"])
    if not np.isfinite(eps_ee) or eps_ee <= 0.0:
        raise RuntimeError("Invalid source ee-channel observed limit.")
    factor = eps_visible / eps_ee
    if not np.isfinite(factor) or factor <= 0.0:
        raise RuntimeError("Invalid source minimal-visible conversion factor.")
    return float(factor)


def _bit_pattern(value: float) -> str:
    return np.asarray([value], dtype="<f8").tobytes().hex()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    started = time.time()

    config_path = args.base_config.expanduser().resolve()
    reviewed_csv = args.reviewed_state_csv.expanduser().resolve()
    closure_path = args.closure_report.expanduser().resolve()
    source_table_path = args.source_v4p2_table.expanduser().resolve()
    authoritative_table_path = (
        args.authoritative_observed_table.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_csv = output_dir / OUTPUT_CSV.name
    validation_json = output_dir / VALIDATION_JSON.name
    provenance_json = output_dir / PROVENANCE_JSON.name

    for path in (
        config_path,
        reviewed_csv,
        closure_path,
        source_table_path,
        authoritative_table_path,
    ):
        if not path.is_file():
            raise SystemExit(f"Required input does not exist: {path}")

    base_config = load_config(str(config_path))
    if not np.isclose(float(base_config.cls_alpha), 0.10, rtol=0.0, atol=0.0):
        raise SystemExit(
            f"Expected the accepted v4.2 base card at alpha=0.10, "
            f"got {base_config.cls_alpha!r}."
        )
    if str(base_config.cls_mode).lower().strip() != "asymptotic":
        raise SystemExit("The accepted base card is not asymptotic CLs.")
    if str(base_config.combined_mode).lower().strip() != "count_scale":
        raise SystemExit("The accepted base card is not count_scale combined mode.")
    validate_v4_geometry(base_config)
    factors = {
        str(key): float(value)
        for key, value in base_config.kernel_ls_res_upper_factor_by_dataset.items()
    }
    if factors != EXPECTED_UPPER_FACTORS:
        raise SystemExit(
            f"Unexpected dataset-specific upper factors: {factors!r}."
        )

    closure = validate_closure_report(
        closure_path,
        config_path,
        reviewed_csv,
    )
    config = replace(
        base_config,
        cls_alpha=0.05,
        cls_mode="asymptotic",
        cls_num_toys=0,
    )

    masses = full_mass_grid()
    source = pd.read_csv(source_table_path).sort_values("mass_GeV").reset_index(
        drop=True
    )
    authoritative = (
        pd.read_csv(authoritative_table_path)
        .sort_values("mass_GeV")
        .reset_index(drop=True)
    )
    if len(source) != len(masses):
        raise SystemExit(
            f"Source v4.2 table has {len(source)} rows, expected {len(masses)}."
        )
    if not np.allclose(
        source["mass_GeV"].to_numpy(float),
        np.asarray(masses, dtype=float),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise SystemExit("Source v4.2 mass grid differs from the frozen grid.")
    if len(authoritative) != len(masses) or not np.allclose(
        authoritative["mass_GeV"].to_numpy(float),
        np.asarray(masses, dtype=float),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise SystemExit(
            "Authoritative fixed-reviewed observed table differs from the "
            "frozen 232-mass grid."
        )
    required_source_columns = {
        "dataset_set",
        "mass_GeV",
        "mass_MeV",
        "sigma_mass_res_GeV",
        "sigma_mass_res_min_GeV",
        "eps2_obs",
        "eps2_obs_ee_channel",
        "eps2_obs_minimal_visible",
        "p0_analytic",
        "Z_analytic",
        "N_eff_BR",
        "BR_ee_minimal",
        "dimuon_correction_applied",
        "gp_state_sha256_by_dataset",
    }
    missing = required_source_columns.difference(source.columns)
    if missing:
        raise SystemExit(f"Source table is missing columns: {sorted(missing)!r}")

    datasets = make_datasets(config)
    fixed, fixed_provenance = load_reviewed_coordinates(
        reviewed_csv,
        masses,
        datasets,
        config,
    )

    rows = []
    reference_closure = []
    for source_row, authoritative_row, mass in zip(
        source.to_dict(orient="records"),
        authoritative.to_dict(orient="records"),
        masses,
    ):
        mass_key = round(float(mass), 12)
        datasets_here, predictions, metadata = build_fixed_predictions(
            float(mass),
            datasets,
            config,
            fixed[mass_key],
        )
        dataset_keys = [dataset.key for dataset in datasets_here]
        dataset_set = "+".join(dataset_keys)
        if dataset_set != str(source_row["dataset_set"]):
            raise RuntimeError(
                f"Active-set mismatch at {mass:.3f} GeV: "
                f"{dataset_set!r} != {source_row['dataset_set']!r}."
            )
        if dataset_set != str(authoritative_row["dataset_set"]):
            raise RuntimeError(
                f"Authoritative active-set mismatch at {mass:.3f} GeV: "
                f"{dataset_set!r} != "
                f"{authoritative_row['dataset_set']!r}."
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
            alpha=0.05,
            combined_mode="count_scale",
        )
        eps2_95 = float(solver.limit(observed))
        if not np.isfinite(eps2_95) or eps2_95 <= 0.0:
            raise RuntimeError(f"Invalid 95% CLs limit at {mass:.3f} GeV.")

        state_sha = {
            str(item["key"]): str(item["state_sha256"])
            for item in metadata
        }
        authoritative_state_sha = json.loads(
            str(authoritative_row["gp_state_sha256_by_dataset"])
        )
        if state_sha != authoritative_state_sha:
            raise RuntimeError(
                f"GP-state hash mismatch at {mass:.3f} GeV: "
                f"{state_sha!r} != {authoritative_state_sha!r}."
            )

        for column in ("p0_analytic", "Z_analytic"):
            if not np.isclose(
                float(source_row[column]),
                float(authoritative_row[column]),
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise RuntimeError(
                    f"v4.2/{column} closure failed at {mass:.3f} GeV."
                )

        factor = _source_factor(pd.Series(source_row))
        eps2_90 = float(source_row["eps2_obs"])
        if eps2_95 < eps2_90:
            raise RuntimeError(
                f"95% CLs limit is stronger than 90% at {mass:.3f} GeV: "
                f"{eps2_95:.12g} < {eps2_90:.12g}."
            )

        mass_mev = int(round(float(mass) * 1000.0))
        if mass_mev in REFERENCE_CLOSURE_MASSES_MEV:
            reference = float(
                combined_cls_limit_epsilon2_from_vectors(
                    observed,
                    b_mean,
                    b_cov,
                    s_unit,
                    config,
                    seed=1,
                )
            )
            bitwise_equal = _bit_pattern(reference) == _bit_pattern(eps2_95)
            if not bitwise_equal:
                raise RuntimeError(
                    f"Cached/reference 95% CLs closure failed at {mass_mev} MeV: "
                    f"{eps2_95:.17g} != {reference:.17g}."
                )
            relative_probe = 5.0e-6
            test_strength = eps2_95 * solver.signal_scale
            cls_at_limit = float(
                asymptotic_cls_profiled_gaussian(
                    test_strength,
                    observed,
                    b_mean,
                    b_cov,
                    solver.signal_template,
                )[0]
            )
            cls_below = float(
                asymptotic_cls_profiled_gaussian(
                    test_strength * (1.0 - relative_probe),
                    observed,
                    b_mean,
                    b_cov,
                    solver.signal_template,
                )[0]
            )
            cls_above = float(
                asymptotic_cls_profiled_gaussian(
                    test_strength * (1.0 + relative_probe),
                    observed,
                    b_mean,
                    b_cov,
                    solver.signal_template,
                )[0]
            )
            if not np.isfinite([cls_below, cls_at_limit, cls_above]).all():
                raise RuntimeError(
                    f"Non-finite direct CLs root check at {mass_mev} MeV."
                )
            if not cls_below >= 0.05 >= cls_above:
                raise RuntimeError(
                    f"Direct CLs root is not bracketed at {mass_mev} MeV: "
                    f"{cls_below:.12g}, {cls_at_limit:.12g}, "
                    f"{cls_above:.12g}."
                )
            if abs(cls_at_limit - 0.05) > 2.0e-7:
                raise RuntimeError(
                    f"Direct CLs root residual is too large at {mass_mev} MeV: "
                    f"{cls_at_limit:.12g}."
                )
            reference_closure.append(
                {
                    "mass_MeV": mass_mev,
                    "dataset_set": dataset_set,
                    "cached_eps2_95": eps2_95,
                    "reference_eps2_95": reference,
                    "bitwise_equal": True,
                    "relative_root_probe": relative_probe,
                    "cls_below_limit": cls_below,
                    "cls_at_limit": cls_at_limit,
                    "cls_above_limit": cls_above,
                    "absolute_cls_root_residual": abs(cls_at_limit - 0.05),
                    "root_bracketed": True,
                }
            )

        rows.append(
            {
                "dataset_set": dataset_set,
                "mass_GeV": float(mass),
                "mass_MeV": mass_mev,
                "sigma_mass_res_GeV": float(
                    source_row["sigma_mass_res_GeV"]
                ),
                "sigma_mass_res_min_GeV": float(
                    source_row["sigma_mass_res_min_GeV"]
                ),
                "confidence_level": 0.95,
                "cls_alpha": 0.05,
                "cls_statistic": "tilde_q_mu",
                "cls_calibration": "asymptotic",
                "combined_mode": "count_scale",
                "eps2_obs_95_ee_channel": eps2_95,
                "minimal_visible_factor": factor,
                "N_eff_BR": float(source_row["N_eff_BR"]),
                "BR_ee_minimal": float(source_row["BR_ee_minimal"]),
                "dimuon_correction_applied": bool(
                    source_row["dimuon_correction_applied"]
                ),
                "eps2_obs_95_minimal_visible": eps2_95 * factor,
                "eps2_obs_90_ee_channel_reference": eps2_90,
                "eps2_obs_90_minimal_visible_reference": float(
                    source_row["eps2_obs_minimal_visible"]
                ),
                "eps2_95_over_90": eps2_95 / eps2_90,
                "p0_analytic": float(authoritative_row["p0_analytic"]),
                "Z_analytic": float(authoritative_row["Z_analytic"]),
                "p0_source": (
                    "accepted fixed-reviewed v4.1/v4.2 shared-epsilon2 "
                    "observed scan"
                ),
                "gp_state_sha256_by_dataset": json.dumps(
                    state_sha, sort_keys=True
                ),
                "profile_cache_limit_calls": solver.counters.limit_calls,
                "profile_cache_asimov_fixed_nodes": (
                    solver.asimov_fixed_cache_size
                ),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != 232:
        raise RuntimeError(f"Expected 232 rows, found {len(result)}.")
    numeric_columns = (
        "eps2_obs_95_ee_channel",
        "eps2_obs_95_minimal_visible",
        "p0_analytic",
        "Z_analytic",
    )
    for column in numeric_columns:
        if not np.isfinite(result[column].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite values in {column}.")
    if not (result["eps2_95_over_90"].to_numpy(float) >= 1.0).all():
        raise RuntimeError("95%/90% monotonicity check failed.")
    if len(reference_closure) != len(REFERENCE_CLOSURE_MASSES_MEV):
        raise RuntimeError("Representative cached/reference closure is incomplete.")

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, float_format="%.17g")

    p0_index = int(result["p0_analytic"].astype(float).idxmin())
    p0_row = result.loc[p0_index]
    ratio = result["eps2_95_over_90"].to_numpy(float)
    elapsed = float(time.time() - started)
    validation = {
        "schema_version": 1,
        "status": "PASS",
        "n_rows": int(len(result)),
        "mass_min_MeV": int(result["mass_MeV"].min()),
        "mass_max_MeV": int(result["mass_MeV"].max()),
        "n_finite_limits": int(
            np.isfinite(result["eps2_obs_95_ee_channel"].to_numpy(float)).sum()
        ),
        "n_toys": 0,
        "observed_only": True,
        "gp_refit": False,
        "only_card_override": {"cls_alpha": {"from": 0.10, "to": 0.05}},
        "all_95_limits_ge_90_limits": True,
        "ratio_95_over_90_min": float(np.min(ratio)),
        "ratio_95_over_90_median": float(np.median(ratio)),
        "ratio_95_over_90_max": float(np.max(ratio)),
        "p0_reused_from_authoritative_fixed_reviewed_scan": True,
        "p0_v4p2_postprocessing_closure_atol": 1.0e-15,
        "minimum_local_p0": float(p0_row["p0_analytic"]),
        "minimum_local_Z": float(p0_row["Z_analytic"]),
        "minimum_local_p0_mass_MeV": int(p0_row["mass_MeV"]),
        "cached_reference_95cl_closure": reference_closure,
        "maximum_representative_cls_root_residual": float(
            max(
                entry["absolute_cls_root_residual"]
                for entry in reference_closure
            )
        ),
        "all_representative_cls_roots_bracketed": True,
        "elapsed_seconds": elapsed,
    }
    validation_json.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provenance = {
        "schema_version": 1,
        "created_utc_unix": time.time(),
        "base_config": str(config_path),
        "base_config_sha256": sha256(config_path),
        "base_config_cls_alpha": 0.10,
        "runtime_override": {"cls_alpha": 0.05},
        "reviewed_state_csv": str(reviewed_csv),
        "reviewed_state_csv_sha256": sha256(reviewed_csv),
        "closure_report": str(closure_path),
        "closure_report_sha256": sha256(closure_path),
        "closure_report_all_bitwise_equal": bool(
            closure.get("all_bitwise_equal")
        ),
        "source_v4p2_table": str(source_table_path),
        "source_v4p2_table_sha256": sha256(source_table_path),
        "authoritative_observed_table": str(authoritative_table_path),
        "authoritative_observed_table_sha256": sha256(
            authoritative_table_path
        ),
        "source_p0_columns_sha256": _sha256_bytes(
            source[["mass_GeV", "p0_analytic", "Z_analytic"]]
            .to_csv(index=False, float_format="%.17g")
            .encode("utf-8")
        ),
        "fixed_state_provenance": fixed_provenance,
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256(output_csv),
        "validation_json": str(validation_json),
        "validation_json_sha256": sha256(validation_json),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cached_solver": str(
            (SOURCE_CAMPAIGN / "cached_profile_solver.py").resolve()
        ),
        "cached_solver_sha256": sha256(
            SOURCE_CAMPAIGN / "cached_profile_solver.py"
        ),
        "cache_algorithm_version": CACHE_ALGORITHM_VERSION,
        "parallel_workers": 1,
        "threads_per_worker": 1,
        "n_toys": 0,
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "elapsed_seconds": elapsed,
    }
    provenance_json.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_csv": str(output_csv),
                "validation_json": str(validation_json),
                "n_rows": len(result),
                "elapsed_seconds": elapsed,
                "p0_min_mass_MeV": int(p0_row["mass_MeV"]),
                "p0_min": float(p0_row["p0_analytic"]),
                "Z_min": float(p0_row["Z_analytic"]),
                "ratio_95_over_90": {
                    "min": float(np.min(ratio)),
                    "median": float(np.median(ratio)),
                    "max": float(np.max(ratio)),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
