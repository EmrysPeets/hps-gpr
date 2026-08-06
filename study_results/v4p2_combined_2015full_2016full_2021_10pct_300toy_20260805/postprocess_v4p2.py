#!/usr/bin/env python3
"""Fail-closed validation and publication postprocessing for HPS-GPR v4.2.

This script is deliberately downstream-only.  It neither fits a Gaussian
process nor draws a pseudoexperiment.  It validates the frozen v4.2 combined
300-toy campaign, closes every observed result to the accepted v4.1 k=12
result, validates the standalone individual observed results, applies the
minimal-visible branching reinterpretation, and writes publication artifacts.

The statistical families remain separate:

* ``p_strong``, ``p_weak``, and ``p_two`` are fixed-mass empirical
  observed-limit ensemble diagnostics;
* ``p0_analytic`` is a local asymptotic discovery statistic; and
* the resolution-spacing Sidak values are analytic references, not
  scan-pseudoexperiment calibrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p2-postprocess-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter
from scipy.stats import norm


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
DERIVED = CAMPAIGN_DIR / "derived"
FIGURES = CAMPAIGN_DIR / "note_figures"

CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
V41_CONFIG = (
    REPO
    / "study_configs"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "config_obsUL90_combined_wide_support_v4p1_2016k12_observed_only.yaml"
)
DEFAULT_BANDS = CAMPAIGN_DIR / "combined_bands_300toy_cached" / "ul_bands_combined_all.csv"
DEFAULT_BANDS_PROVENANCE = (
    CAMPAIGN_DIR
    / "combined_bands_300toy_cached"
    / "ul_bands_combined_all_provenance.json"
)
DEFAULT_CLOSURE = DERIVED / "cached_profile_closure_v4p2.json"
DEFAULT_STATES = DERIVED / "observed_gp_states_v4p2_enriched.csv"
DEFAULT_INDIVIDUAL = DERIVED / "individual_observed_limits_v4p2.csv"
DEFAULT_INDIVIDUAL_PROVENANCE = DERIVED / "individual_ledger_validation_v4p2.json"

ACCEPTED_COMPACT_STATES = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
ACCEPTED_COMBINED = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "final_k12_combined_observed"
    / "combined_observed_fixed_reviewed.csv"
)
ACCEPTED_COMBINED_PROVENANCE = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "final_k12_combined_observed"
    / "combined_observed_fixed_reviewed_provenance.json"
)
V4_BANDS = (
    REPO
    / "study_results"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "combined_bands_300toy_cached"
    / "ul_bands_combined_all.csv"
)
V4_BANDS_PROVENANCE = (
    REPO
    / "study_results"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "combined_bands_300toy_cached"
    / "ul_bands_combined_all_provenance.json"
)
RUNNER = CAMPAIGN_DIR / "run_combined_bands_cached_fixed_reviewed.py"
CACHED_SOLVER = CAMPAIGN_DIR / "cached_profile_solver.py"

REVIEWED_COMBINED = DERIVED / "combined_bands300_reviewed_v4p2.csv"
GP_STATE_CLOSURE = DERIVED / "combined_gp_state_closure_v4p2.csv"
COMBINED_SIDAK = DERIVED / "combined_sidak_reference_v4p2.csv"
REVIEWED_INDIVIDUAL = DERIVED / "individual_observed_limits_reviewed_v4p2.csv"
INDIVIDUAL_SIDAK = DERIVED / "individual_sidak_reference_v4p2.csv"
SUMMARY_JSON = DERIVED / "postprocessing_summary_v4p2.json"
SUMMARY_CSV = DERIVED / "postprocessing_summary_v4p2.csv"
VALIDATION_JSON = DERIVED / "postprocessing_validation_v4p2.json"
PLOT_MANIFEST = DERIVED / "plot_manifest_v4p2.json"
POSTPROCESS_README = CAMPAIGN_DIR / "POSTPROCESSING_README.md"

MASS_LOW_MEV = 19
MASS_HIGH_MEV = 250
N_MASSES = MASS_HIGH_MEV - MASS_LOW_MEV + 1
N_TOYS = 300
SEED = 24_680
CACHE_VERSION = "campaign_local_deterministic_profile_cache_v1"
INDEPENDENCE_WIDTH_SIGMA = 2.25
M_MU_GEV = 0.1056583745
DIMUON_THRESHOLD_GEV = 2.0 * M_MU_GEV
LML_CLOSURE_ATOL = 5.0e-5

EXPECTED_ACCEPTED_COMPACT_SHA256 = (
    "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9"
)
EXPECTED_ACCEPTED_COMBINED_SHA256 = (
    "fa95a50a8b8ddc1d69a319137038a177c6d6da3afbbf9163d8955cf197182de2"
)
EXPECTED_V4_BANDS_SHA256 = (
    "33f576e09d0e603978b2e0b71eb608663b95806606ab056d1ba8f32c8f5b2cdb"
)

SEARCH_RANGES_GEV = {
    "2015": (0.019, 0.090),
    "2016": (0.039, 0.180),
    "2021": (0.050, 0.250),
}
SUPPORT_RANGES_GEV = {
    "2015": (0.014, 0.135),
    "2016": (0.030, 0.210),
    "2021": (0.040, 0.300),
}
EXPECTED_STATE_ROWS = {"2015": 72, "2016": 142, "2021": 201}
EXPECTED_STATE_COUNT = sum(EXPECTED_STATE_ROWS.values())
EXPECTED_ACTIVE_COUNTS = {
    "2015": 20,
    "2015+2016": 11,
    "2015+2016+2021": 41,
    "2016+2021": 90,
    "2021": 70,
}
EXPECTED_LS_LO_FACTORS = {"2015": 1.0, "2016": 0.9, "2021": 1.1}
EXPECTED_LS_HI_FACTORS = {"2015": 8.0, "2016": 12.0, "2021": 15.0}

RAW_QUANTILE_COLUMNS = (
    "eps2_lo2",
    "eps2_lo1",
    "eps2_med",
    "eps2_hi1",
    "eps2_hi2",
)
COUPLING_COLUMNS = (
    "eps2_obs",
    "ul_eps2_obs",
    "eps2_lo2",
    "eps2_lo1",
    "eps2_med",
    "eps2_hi1",
    "eps2_hi2",
    "eps2_mean",
    "toy_eps2_uls_q02",
    "toy_eps2_uls_q16",
    "toy_eps2_uls_q50",
    "toy_eps2_uls_q84",
    "toy_eps2_uls_q97",
    "toy_eps2_uls_mean",
)
ALIAS_PAIRS = (
    ("eps2_obs", "ul_eps2_obs"),
    ("eps2_lo2", "toy_eps2_uls_q02"),
    ("eps2_lo1", "toy_eps2_uls_q16"),
    ("eps2_med", "toy_eps2_uls_q50"),
    ("eps2_hi1", "toy_eps2_uls_q84"),
    ("eps2_hi2", "toy_eps2_uls_q97"),
    ("eps2_mean", "toy_eps2_uls_mean"),
)

STATE_REQUIRED_COLUMNS = {
    "dataset",
    "mass_GeV",
    "mass_MeV",
    "sigma_val",
    "integral_density",
    "density_nsigma",
    "density_window_lo",
    "density_window_hi",
    "density_source_lo",
    "density_source_hi",
    "density_window_fully_covered",
    "extract_success",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "const_opt",
    "ls_lo",
    "ls_hi",
    "ls_opt",
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "lml",
    "n_train",
    "n_train_low",
    "n_train_high",
    "train_domain_lo",
    "train_domain_hi",
    "optimizer_restarts",
    "selected_source",
    "selected_source_sha256",
    "row_source",
    "review_status",
    "branch_multiplicity",
    "interpolated",
    "cls_statistic",
    "cls_calibration",
    "visibility",
    "signal_model",
    "geometry_density_source",
}
EXPECTED_ENRICHED_COLUMNS = (
    "dataset",
    "mass_GeV",
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "integral_density",
    "density_nsigma",
    "density_window_lo",
    "density_window_hi",
    "density_window_width",
    "density_source_lo",
    "density_source_hi",
    "density_source_n_bins",
    "density_source_bin_width_median",
    "density_window_fully_covered",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "extract_success",
    "cls_statistic",
    "cls_calibration",
    "signal_model",
    "global_method",
    "visibility",
    "kernel_str",
    "ls_lo",
    "ls_hi",
    "ls_init",
    "ls_opt",
    "sigma_x",
    "const_opt",
    "lml",
    "n_train",
    "n_train_low",
    "n_train_high",
    "n_full",
    "n_blind",
    "train_domain_lo",
    "train_domain_hi",
    "bin_width_median",
    "const_init",
    "const_lo",
    "const_hi",
    "const_at_lower",
    "const_at_upper",
    "ls_at_lower",
    "ls_at_upper",
    "optimizer_restarts",
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "ls_opt_over_sigma_x",
    "ls_lo_over_sigma",
    "ls_hi_over_sigma",
    "ls_opt_over_sigma",
    "mass_MeV",
    "selected_attempt",
    "selected_source",
    "selected_source_sha256",
    "row_source",
    "optimizer_repair_applied",
    "review_status",
    "branch_multiplicity",
    "reproducing_attempts",
    "reproducing_other_attempts",
    "reproducing_sources",
    "max_abs_reproducing_delta_lml",
    "max_abs_reproducing_delta_const_opt",
    "max_abs_reproducing_delta_ls_opt",
    "all_attempt_sources",
    "interpolated",
    "selected_repair_reproduced",
    "repair_reproduction_pending",
    "candidate_count",
    "repair_candidate_count",
    "delta_lml_selected_minus_raw",
    "geometry_density_source",
    "geometry_density_source_sha256",
    "accepted_compact_ledger",
    "accepted_compact_ledger_sha256",
    "accepted_config",
    "accepted_config_sha256",
)
EXPECTED_INDIVIDUAL_COLUMNS = (
    "dataset",
    "sample_label",
    "mass_GeV",
    "mass_MeV",
    "A_up",
    "eps2_up",
    "eps2_observed_ee_channel",
    "minimal_visible_factor",
    "BR_ee_minimal",
    "eps2_observed_minimal_visible",
    "dimuon_correction_applied",
    "p0_analytic",
    "Z_analytic",
    "sigma_val",
    "integral_density",
    "const_opt",
    "ls_opt",
    "ls_hi",
    "ls_opt_over_ls_hi",
    "lml",
    "selected_attempt",
    "selected_source",
    "selected_source_sha256",
    "row_source",
    "optimizer_repair_applied",
    "review_status",
    "branch_multiplicity",
    "interpolated",
    "accepted_config",
    "accepted_config_sha256",
    "source_enriched_ledger",
    "source_enriched_ledger_sha256",
    "limit_scope",
    "bands_included",
)
COMPACT_REQUIRED_COLUMNS = {
    "dataset",
    "mass_GeV",
    "const_opt",
    "ls_opt",
    "lml",
    "ls_hi",
    "ls_hi_over_sigma_x",
    "interpolated",
    "selected_source",
    "selected_source_sha256",
    "row_source",
    "review_status",
    "branch_multiplicity",
}
BAND_REQUIRED_COLUMNS = {
    "dataset_set",
    "mass_GeV",
    "sigma_mass_res_GeV",
    "sigma_mass_res_min_GeV",
    "cls_alpha",
    "eps2_obs",
    "p0_analytic",
    "Z_analytic",
    *RAW_QUANTILE_COLUMNS,
    "eps2_mean",
    "ul_eps2_obs",
    "toy_eps2_uls_q02",
    "toy_eps2_uls_q16",
    "toy_eps2_uls_q50",
    "toy_eps2_uls_q84",
    "toy_eps2_uls_q97",
    "toy_eps2_uls_mean",
    "p_strong",
    "p_weak",
    "p_two",
    "tail_count_strong_le_observed",
    "tail_count_weak_ge_observed",
    "tail_count_equal_observed",
    "tail_count_two_sided_min",
    "empirical_tail_resolution",
    "meta",
    "cls_statistic",
    "cls_calibration",
    "combined_mode",
    "bands_refit_gp_on_toy",
    "bands_train_exclude_nsigma",
    "bands_refit_restarts",
    "bands_refit_optimize",
    "bands_seed_sequence_index",
    "n_toys_requested",
    "n_toys_finite",
    "gp_lml_by_dataset",
    "gp_ls_opt_by_dataset",
    "gp_const_opt_by_dataset",
    "gp_state_sha256_by_dataset",
    "observed_gp_fit_mode",
    "observed_gp_optimizer_restarts",
    "limit_solver",
    "profile_cache_limit_calls",
}

COLORS = {
    "local": "#17365D",
    "sidak": "#B2472D",
    "observed": "#B42318",
    "expected": "#202124",
    "band1": "#4C956C",
    "band2": "#F2C14E",
    "strong": "#2166AC",
    "weak": "#D6604D",
    "two": "#252525",
    "2015": "#2166AC",
    "2016": "#D6604D",
    "2021": "#1B9E77",
    "threshold": "#6B7280",
}
ACTIVE_COLORS = {
    "2015": "#DCEAF7",
    "2015+2016": "#F4DDD8",
    "2015+2016+2021": "#DDEFE8",
    "2016+2021": "#F2E4D8",
    "2021": "#D9EEE7",
}
ACTIVE_LABELS = {
    "2015": "2015",
    "2015+2016": "15+16",
    "2015+2016+2021": "15+16+21",
    "2016+2021": "16+21",
    "2021": "2021",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_mass_grid() -> np.ndarray:
    return np.arange(MASS_LOW_MEV, MASS_HIGH_MEV + 1, dtype=int) / 1000.0


def expected_dataset_grid(dataset: str) -> np.ndarray:
    low, high = SEARCH_RANGES_GEV[dataset]
    return np.arange(round(1000.0 * low), round(1000.0 * high) + 1) / 1000.0


def active_datasets(mass_gev: float) -> List[str]:
    return [
        dataset
        for dataset, (low, high) in SEARCH_RANGES_GEV.items()
        if low - 1.0e-12 <= float(mass_gev) <= high + 1.0e-12
    ]


def expected_active_tags() -> List[str]:
    return ["+".join(active_datasets(mass)) for mass in expected_mass_grid()]


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def normalize_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if bool(series.isna().any()):
            raise RuntimeError(f"{label} contains missing booleans")
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    invalid = normalized.isna() | ~normalized.isin(mapping)
    if bool(invalid.any()):
        raise RuntimeError(
            f"{label} contains invalid booleans: "
            f"{series.loc[invalid].head(5).tolist()}"
        )
    return normalized.map(mapping).astype(bool)


def require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label}.{column} contains non-finite values")


def require_finite_positive(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    require_finite(frame, columns, label)
    for column in columns:
        if not bool((frame[column].to_numpy(float) > 0.0).all()):
            raise RuntimeError(f"{label}.{column} is not uniformly positive")


def require_exact_combined_grid(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if len(frame) != N_MASSES:
        raise RuntimeError(f"{label} has {len(frame)} rows; expected {N_MASSES}")
    if bool(frame["mass_GeV"].duplicated().any()):
        raise RuntimeError(f"{label} contains duplicate masses")
    out = frame.sort_values("mass_GeV").reset_index(drop=True).copy()
    if not np.array_equal(out["mass_GeV"].to_numpy(float), expected_mass_grid()):
        if not np.allclose(
            out["mass_GeV"].to_numpy(float),
            expected_mass_grid(),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{label} does not cover the exact 19--250 MeV grid")
        out["mass_GeV"] = expected_mass_grid()
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML is not a mapping: {path}")
    return payload


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON is not a mapping: {path}")
    return payload


def parse_json_mapping(value: Any, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON mapping")
    return {str(key): item for key, item in payload.items()}


def parse_json_metadata(value: Any, label: str) -> Dict[str, Dict[str, Any]]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, list):
        raise RuntimeError(f"{label} is not a JSON list")
    result: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict) or "key" not in item:
            raise RuntimeError(f"{label} has a malformed entry")
        key = str(item["key"])
        if key in result:
            raise RuntimeError(f"{label} has duplicate metadata for {key}")
        result[key] = item
    return result


def compare_exact_series(
    left: pd.Series,
    right: pd.Series,
    label: str,
) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"{label} length mismatch")
    if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
        right.dtype
    ):
        left_values = left.to_numpy()
        right_values = right.to_numpy()
        equal = np.array_equal(left_values, right_values, equal_nan=True)
    else:
        left_values = left.astype("string").fillna("<NA>").to_numpy()
        right_values = right.astype("string").fillna("<NA>").to_numpy()
        equal = np.array_equal(left_values, right_values)
    if not equal:
        mismatch = np.flatnonzero(left_values != right_values)
        sample = mismatch[:5].tolist()
        raise RuntimeError(f"{label} is not bitwise/exactly equal; rows {sample}")


def compare_close_series(
    left: pd.Series,
    right: pd.Series,
    label: str,
    *,
    rtol: float = 2.0e-15,
    atol: float = 1.0e-15,
) -> None:
    left_values = left.to_numpy(float)
    right_values = right.to_numpy(float)
    if not np.allclose(
        left_values,
        right_values,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        delta = np.abs(left_values - right_values)
        index = int(np.nanargmax(delta))
        raise RuntimeError(
            f"{label} is not numerically closed; max delta={delta[index]!r} "
            f"at row {index}"
        )


def validate_config() -> Dict[str, Any]:
    config = load_yaml(CONFIG)
    accepted = load_yaml(V41_CONFIG)
    allowed_differences = {
        "combined_bands_n_toys",
        "do_combined_bands",
        "make_ul_bands",
        "output_dir",
    }
    actual_differences = {
        key
        for key in set(config).union(accepted)
        if config.get(key) != accepted.get(key)
    }
    if actual_differences != allowed_differences:
        details = {
            key: {"v4p1": accepted.get(key), "v4p2": config.get(key)}
            for key in sorted(actual_differences.union(allowed_differences))
            if config.get(key) != accepted.get(key) or key in allowed_differences
        }
        raise RuntimeError(
            "v4.2 differs from accepted v4.1 outside, or fails to differ in, "
            f"the declared four fields: {details}"
        )

    exact_values: Dict[str, Any] = {
        "make_ul_bands": True,
        "do_combined_bands": True,
        "combined_bands_n_toys": N_TOYS,
        "combined_bands_seed": SEED,
        "ul_bands_toys": 0,
        "run_limit_bands_on": "",
        "make_eps2_bands": False,
        "ul_bands_refit_gp_on_toy": False,
        "cls_alpha": 0.1,
        "cls_mode": "asymptotic",
        "combined_mode": "count_scale",
        "blind_nsigma": 2.25,
        "gp_train_exclude_nsigma": 2.25,
        "neighborhood_rebin": 5,
        "n_restarts": 12,
        "eps2_density_nsigma": 1.64,
    }
    for key, expected in exact_values.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Unexpected config {key}: {config.get(key)!r} != {expected!r}")

    for dataset in SEARCH_RANGES_GEV:
        search = tuple(float(value) for value in config[f"range_{dataset}"])
        support = tuple(float(value) for value in config[f"data_range_{dataset}"])
        if search != SEARCH_RANGES_GEV[dataset]:
            raise RuntimeError(f"Unexpected {dataset} search range: {search}")
        if support != SUPPORT_RANGES_GEV[dataset]:
            raise RuntimeError(f"Unexpected {dataset} fit support: {support}")
    lower = {
        str(key): float(value)
        for key, value in config["kernel_ls_res_lower_factor_by_dataset"].items()
    }
    upper = {
        str(key): float(value)
        for key, value in config["kernel_ls_res_upper_factor_by_dataset"].items()
    }
    if lower != EXPECTED_LS_LO_FACTORS:
        raise RuntimeError(f"Unexpected dataset lower length-scale factors: {lower}")
    if upper != EXPECTED_LS_HI_FACTORS:
        raise RuntimeError(f"Unexpected dataset upper length-scale factors: {upper}")
    return {
        "path": repo_path(CONFIG),
        "sha256": sha256(CONFIG),
        "accepted_v4p1_path": repo_path(V41_CONFIG),
        "accepted_v4p1_sha256": sha256(V41_CONFIG),
        "only_declared_differences": sorted(allowed_differences),
        "physics_settings_frozen_to_v4p1": True,
    }


def reviewed_coordinate_digest(compact: pd.DataFrame) -> str:
    frame = compact.copy()
    frame["dataset"] = frame["dataset"].astype(str)
    fixed: Dict[float, Dict[str, Dict[str, float]]] = {}
    for mass in expected_mass_grid():
        here = frame[np.isclose(frame["mass_GeV"], mass, rtol=0.0, atol=1.0e-12)]
        expected = set(active_datasets(mass))
        if set(here["dataset"]) != expected:
            raise RuntimeError(f"Compact-state active set mismatch at {mass:.3f} GeV")
        fixed[round(float(mass), 12)] = {
            str(row.dataset): {
                "const_opt": float(row.const_opt),
                "ls_opt": float(row.ls_opt),
                "reviewed_lml": float(row.lml),
            }
            for row in here.itertuples(index=False)
        }
    payload = json.dumps(
        {f"{mass:.12f}": values for mass, values in sorted(fixed.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_closure(
    payload: Mapping[str, Any],
    compact_sha: str,
    config_sha: str,
) -> Dict[str, Any]:
    expected_masses = [0.020, 0.040, 0.060, 0.100, 0.200]
    expected_indices = [1, 21, 41, 81, 181]
    expected_n_active = [1, 2, 3, 2, 1]
    exact = {
        "cache_algorithm_version": CACHE_VERSION,
        "config_sha256": config_sha,
        "reviewed_csv_sha256": compact_sha,
        "all_bitwise_equal": True,
        "seed": SEED,
        "seed_sequence_index_rule": "mass_MeV - 19",
        "toys_per_mass": 20,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Closure report {key} mismatch: {payload.get(key)!r}")
    results = payload.get("mass_results")
    if not isinstance(results, list) or len(results) != len(expected_masses):
        raise RuntimeError("Closure report must contain exactly five reference masses")
    for entry, mass, index, n_active in zip(
        results,
        expected_masses,
        expected_indices,
        expected_n_active,
    ):
        if not isinstance(entry, dict):
            raise RuntimeError("Closure mass entry is not a mapping")
        if not np.isclose(float(entry.get("mass_GeV")), mass, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("Closure masses differ from the frozen audit set")
        required = {
            "seed_sequence_index": index,
            "n_active_datasets": n_active,
            "n_pseudoexperiments": 20,
            "n_vectors": 21,
            "bitwise_equal": True,
            "max_absolute_difference": 0.0,
        }
        for key, expected in required.items():
            if entry.get(key) != expected:
                raise RuntimeError(
                    f"Closure {mass:.3f} GeV {key} mismatch: "
                    f"{entry.get(key)!r} != {expected!r}"
                )
        if len(entry.get("active_datasets", [])) != n_active:
            raise RuntimeError(f"Closure active-dataset list fails at {mass:.3f} GeV")
    return {
        "path": repo_path(DEFAULT_CLOSURE),
        "sha256": sha256(DEFAULT_CLOSURE),
        "five_mass_bitwise_reference_closure": True,
        "covers_one_two_three_active_datasets": True,
    }


def validate_band_provenance(
    payload: Mapping[str, Any],
    bands_path: Path,
    provenance_path: Path,
    closure_path: Path,
    compact: pd.DataFrame,
) -> Dict[str, Any]:
    config_sha = sha256(CONFIG)
    compact_sha = sha256(ACCEPTED_COMPACT_STATES)
    exact = {
        "cache_algorithm_version": CACHE_VERSION,
        "physics_config_sha256": config_sha,
        "reviewed_csv_sha256": compact_sha,
        "closure_report_sha256": sha256(closure_path),
        "n_toys_per_mass": N_TOYS,
        "seed": SEED,
        "seed_sequence_index_rule": "mass_MeV - 19",
        "n_masses": N_MASSES,
        "shard_index": 0,
        "shard_count": 1,
        "parallel_workers": 8,
        "parallel_backend": "loky",
        "threads_per_worker": 1,
        "refit_gp_on_toy": False,
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "observed_gp_optimizer_restarts": 0,
        "toy_construction": "conditional GP posterior MVN then Poisson",
        "inner_cls": "asymptotic tilde_q_mu, alpha=0.1",
        "output_csv_sha256": sha256(bands_path),
        "runner_sha256": sha256(RUNNER),
        "cached_solver_sha256": sha256(CACHED_SOLVER),
        "selected_reviewed_gp_states": EXPECTED_STATE_COUNT,
        "reviewed_gp_coordinates_sha256": reviewed_coordinate_digest(compact),
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Band provenance {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    masses = np.asarray(payload.get("mass_grid_GeV", []), dtype=float)
    if not np.array_equal(masses, expected_mass_grid()):
        raise RuntimeError("Band provenance does not contain the exact full mass grid")
    elapsed = float(payload.get("elapsed_seconds", float("nan")))
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("Band provenance elapsed time is not finite and positive")
    closure_results = payload.get("closure_mass_results")
    if not isinstance(closure_results, list) or len(closure_results) != 5:
        raise RuntimeError("Band provenance does not embed the five closure results")
    return {
        "path": repo_path(provenance_path),
        "sha256": sha256(provenance_path),
        "output_hash_closure": True,
        "config_hash_closure": True,
        "reviewed_coordinate_digest_closure": True,
        "runner_and_solver_hash_closure": True,
        "elapsed_seconds": elapsed,
    }


def validate_compact_accepted(compact: pd.DataFrame) -> pd.DataFrame:
    if sha256(ACCEPTED_COMPACT_STATES) != EXPECTED_ACCEPTED_COMPACT_SHA256:
        raise RuntimeError("Accepted compact k=12 state ledger SHA-256 changed")
    require_columns(compact, COMPACT_REQUIRED_COLUMNS, "accepted compact states")
    if len(compact) != EXPECTED_STATE_COUNT:
        raise RuntimeError("Accepted compact state ledger is not 415 rows")
    out = compact.copy()
    out["dataset"] = out["dataset"].astype(str)
    if bool(out.duplicated(["dataset", "mass_GeV"]).any()):
        raise RuntimeError("Accepted compact state ledger has duplicate states")
    return out.sort_values(["dataset", "mass_GeV"]).reset_index(drop=True)


def validate_enriched_states(
    frame: pd.DataFrame,
    compact: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if tuple(frame.columns) != EXPECTED_ENRICHED_COLUMNS:
        raise RuntimeError(
            "v4.2 enriched-state schema/order changed: "
            f"{tuple(frame.columns)} != {EXPECTED_ENRICHED_COLUMNS}"
        )
    require_columns(frame, STATE_REQUIRED_COLUMNS, "v4.2 enriched observed states")
    if len(frame) != EXPECTED_STATE_COUNT:
        raise RuntimeError(
            f"Enriched state ledger has {len(frame)} rows; expected {EXPECTED_STATE_COUNT}"
        )
    out = frame.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    if set(out["dataset"]) != set(SEARCH_RANGES_GEV):
        raise RuntimeError(f"Unexpected enriched datasets: {sorted(set(out['dataset']))}")
    if bool(out.duplicated(["dataset", "mass_GeV"]).any()):
        raise RuntimeError("Enriched state ledger has duplicate states")

    for column in ("density_window_fully_covered", "extract_success", "interpolated"):
        out[column] = normalize_boolean(out[column], f"enriched states.{column}")
    if not bool(out["density_window_fully_covered"].all()):
        raise RuntimeError("An enriched density window is not fully covered")
    if not bool(out["extract_success"].all()):
        raise RuntimeError("An enriched standalone extraction failed")
    if bool(out["interpolated"].any()):
        raise RuntimeError("Enriched state ledger contains interpolation")
    status_counts = {
        (str(dataset), str(status)): int(count)
        for (dataset, status), count in out.groupby(
            ["dataset", "review_status"]
        ).size().items()
    }
    expected_status_counts = {
        ("2015", "resolved_reproduced_max_lml"): 72,
        ("2016", "raw_scan_row"): 139,
        ("2016", "repair_selected_reproduced_max_lml"): 3,
        ("2021", "resolved_reproduced_max_lml"): 201,
    }
    if status_counts != expected_status_counts:
        raise RuntimeError(
            f"Enriched review-status composition changed: {status_counts}"
        )
    raw_2016 = out[
        (out["dataset"] == "2016")
        & (out["review_status"] == "raw_scan_row")
    ]
    reproduced = out[out["review_status"] != "raw_scan_row"]
    if set(raw_2016["branch_multiplicity"].astype(int)) != {1}:
        raise RuntimeError("Accepted 2016 raw-scan rows changed branch multiplicity")
    if not bool((reproduced["branch_multiplicity"].to_numpy(int) >= 2).all()):
        raise RuntimeError("A reproduced enriched state lacks an independent branch")

    require_finite(
        out,
        (
            "mass_GeV",
            "mass_MeV",
            "sigma_val",
            "integral_density",
            "density_nsigma",
            "density_window_lo",
            "density_window_hi",
            "density_source_lo",
            "density_source_hi",
            "eps2_up",
            "p0_analytic",
            "Z_analytic",
            "const_opt",
            "ls_lo",
            "ls_hi",
            "ls_opt",
            "lml",
            "n_train",
            "n_train_low",
            "n_train_high",
            "train_domain_lo",
            "train_domain_hi",
            "optimizer_restarts",
        ),
        "v4.2 enriched observed states",
    )
    require_finite_positive(
        out,
        ("sigma_val", "integral_density", "eps2_up", "const_opt", "ls_lo", "ls_hi", "ls_opt"),
        "v4.2 enriched observed states",
    )
    if not np.allclose(out["density_nsigma"], 1.64, rtol=0.0, atol=0.0):
        raise RuntimeError("Enriched density windows are not exactly physical +/-1.64 sigma")
    if not bool(
        (
            out["density_window_lo"].to_numpy(float)
            >= out["density_source_lo"].to_numpy(float) - 1.0e-15
        ).all()
        and (
            out["density_window_hi"].to_numpy(float)
            <= out["density_source_hi"].to_numpy(float) + 1.0e-15
        ).all()
    ):
        raise RuntimeError("An enriched density window extends beyond its source histogram")
    if set(out["cls_statistic"].astype(str)) != {"tilde_q_mu"}:
        raise RuntimeError("Enriched individual states use another CLs statistic")
    if set(out["cls_calibration"].astype(str)) != {"asymptotic"}:
        raise RuntimeError("Enriched individual states are not uniformly asymptotic")
    if set(out["visibility"].astype(str)) != {"observed"}:
        raise RuntimeError("Enriched state ledger contains non-observed data")
    if set(out["signal_model"].astype(str)) != {"default"}:
        raise RuntimeError("Enriched state ledger uses another signal model")
    if set(out["optimizer_restarts"].astype(int)) != {12}:
        raise RuntimeError("Enriched observed-state scans do not uniformly record 12 restarts")
    if not bool(
        (
            out["n_train_low"].to_numpy(int)
            + out["n_train_high"].to_numpy(int)
            == out["n_train"].to_numpy(int)
        ).all()
    ):
        raise RuntimeError("Enriched training side counts do not close")
    if not bool((out["n_train_low"].to_numpy(int) > 0).all()):
        raise RuntimeError("An enriched state lacks low-side training bins")
    if not bool((out["n_train_high"].to_numpy(int) > 0).all()):
        raise RuntimeError("An enriched state lacks high-side training bins")

    for dataset in SEARCH_RANGES_GEV:
        rows = out[out["dataset"] == dataset].sort_values("mass_GeV")
        grid = expected_dataset_grid(dataset)
        if len(rows) != EXPECTED_STATE_ROWS[dataset] or not np.allclose(
            rows["mass_GeV"].to_numpy(float),
            grid,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{dataset} enriched-state grid is incomplete")
        if not np.allclose(
            rows["mass_MeV"].to_numpy(float),
            1000.0 * grid,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError(f"{dataset} mass units disagree")
        support = SUPPORT_RANGES_GEV[dataset]
        if not np.allclose(rows["train_domain_lo"], support[0], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"{dataset} enriched low fit-support edge changed")
        if not np.allclose(rows["train_domain_hi"], support[1], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"{dataset} enriched high fit-support edge changed")
        if not np.allclose(
            rows["ls_lo_over_sigma_x"],
            EXPECTED_LS_LO_FACTORS[dataset],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{dataset} lower length-scale factor changed")
        if not np.allclose(
            rows["ls_hi_over_sigma_x"],
            EXPECTED_LS_HI_FACTORS[dataset],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{dataset} upper length-scale factor changed")

    if not bool(
        (
            out["ls_opt"].to_numpy(float)
            >= out["ls_lo"].to_numpy(float) * (1.0 - 1.0e-10)
        ).all()
        and (
            out["ls_opt"].to_numpy(float)
            <= out["ls_hi"].to_numpy(float) * (1.0 + 1.0e-10)
        ).all()
    ):
        raise RuntimeError("An enriched GP length scale lies outside its frozen bounds")

    for source, rows in out.groupby("selected_source", sort=False):
        source_path = REPO / str(source)
        if not source_path.is_file():
            raise RuntimeError(f"Enriched selected source does not exist: {source}")
        expected_hashes = set(rows["selected_source_sha256"].astype(str).str.lower())
        if expected_hashes != {sha256(source_path)}:
            raise RuntimeError(f"Enriched selected-source SHA-256 fails for {source}")
        if not bool(
            rows["row_source"].astype(str).str.contains(str(source), regex=False).all()
        ):
            raise RuntimeError(f"Enriched row_source does not cite selected source {source}")
    geometry_sources = out["geometry_density_source"].astype("string").str.strip()
    if bool(geometry_sources.isna().any()) or bool((geometry_sources.str.len() == 0).any()):
        raise RuntimeError("Enriched geometry_density_source contains empty provenance")
    for source, rows in out.groupby("geometry_density_source", sort=False):
        source_path = REPO / str(source)
        if not source_path.is_file():
            raise RuntimeError(f"Enriched geometry/density source does not exist: {source}")
        expected_hashes = set(
            rows["geometry_density_source_sha256"].astype(str).str.lower()
        )
        if expected_hashes != {sha256(source_path)}:
            raise RuntimeError(
                f"Enriched geometry/density source SHA-256 fails for {source}"
            )
    if set(out["accepted_compact_ledger"].astype(str)) != {
        repo_path(ACCEPTED_COMPACT_STATES)
    }:
        raise RuntimeError("Enriched ledger cites another compact accepted state ledger")
    if set(out["accepted_compact_ledger_sha256"].astype(str)) != {
        sha256(ACCEPTED_COMPACT_STATES)
    }:
        raise RuntimeError("Enriched compact-ledger SHA-256 provenance fails")
    if set(out["accepted_config"].astype(str)) != {repo_path(V41_CONFIG)}:
        raise RuntimeError("Enriched ledger cites another accepted physics card")
    if set(out["accepted_config_sha256"].astype(str)) != {sha256(V41_CONFIG)}:
        raise RuntimeError("Enriched accepted-config SHA-256 provenance fails")
    pending = normalize_boolean(
        out["repair_reproduction_pending"],
        "enriched states.repair_reproduction_pending",
    )
    if bool(pending.any()):
        raise RuntimeError("Enriched state ledger has a pending repair reproduction")

    compact_sorted = compact.sort_values(["dataset", "mass_GeV"]).reset_index(drop=True)
    enriched_sorted = out.sort_values(["dataset", "mass_GeV"]).reset_index(drop=True)
    compare_exact_series(
        enriched_sorted["dataset"],
        compact_sorted["dataset"],
        "enriched/compact dataset",
    )
    compare_close_series(
        enriched_sorted["mass_GeV"],
        compact_sorted["mass_GeV"],
        "enriched/compact mass",
        rtol=0.0,
        atol=1.0e-15,
    )
    for column in (
        "const_opt",
        "ls_opt",
        "lml",
        "ls_hi",
        "ls_hi_over_sigma_x",
    ):
        compare_close_series(
            enriched_sorted[column],
            compact_sorted[column],
            f"enriched/compact {column}",
        )
    for column in (
        "selected_source",
        "selected_source_sha256",
        "row_source",
        "review_status",
        "branch_multiplicity",
        "interpolated",
    ):
        compare_exact_series(
            enriched_sorted[column],
            compact_sorted[column],
            f"enriched/compact {column}",
        )
    return enriched_sorted, {
        "path": repo_path(DEFAULT_STATES),
        "sha256": sha256(DEFAULT_STATES),
        "rows": int(len(enriched_sorted)),
        "dataset_rows": {
            key: int(value)
            for key, value in enriched_sorted["dataset"].value_counts().sort_index().items()
        },
        "compact_k12_tight_numeric_coordinate_and_exact_provenance_closure": True,
        "physical_density_window_full_coverage": True,
    }


def validate_tail_row(row: Any) -> None:
    mass = float(row.mass_GeV)
    n_finite = int(row.n_toys_finite)
    strong = int(row.tail_count_strong_le_observed)
    weak = int(row.tail_count_weak_ge_observed)
    equal = int(row.tail_count_equal_observed)
    two_min = int(row.tail_count_two_sided_min)
    if any(value < 0 or value > n_finite for value in (strong, weak, equal, two_min)):
        raise RuntimeError(f"Tail count outside 0--{n_finite} at {mass:.3f} GeV")
    if strong + weak - equal != n_finite:
        raise RuntimeError(f"Tail-count partition fails at {mass:.3f} GeV")
    if two_min != min(strong, weak):
        raise RuntimeError(f"Two-sided raw count fails at {mass:.3f} GeV")
    expected = {
        "p_strong": strong / n_finite,
        "p_weak": weak / n_finite,
        "p_two": min(1.0, 2.0 * min(strong / n_finite, weak / n_finite)),
    }
    for column, expected_value in expected.items():
        if not np.isclose(
            float(getattr(row, column)),
            float(expected_value),
            rtol=1.0e-14,
            atol=1.0e-15,
        ):
            raise RuntimeError(f"{column}/raw-count mismatch at {mass:.3f} GeV")
    if not np.isclose(
        float(row.empirical_tail_resolution),
        float(1.0 / n_finite),
        rtol=1.0e-14,
        atol=1.0e-15,
    ):
        raise RuntimeError(f"Tail resolution mismatch at {mass:.3f} GeV")


def validate_bands(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, BAND_REQUIRED_COLUMNS, "v4.2 combined bands")
    out = require_exact_combined_grid(frame, "v4.2 combined bands")
    if out["dataset_set"].astype(str).tolist() != expected_active_tags():
        raise RuntimeError("v4.2 band active sets disagree with the exact mass grid")
    active_counts = out["dataset_set"].astype(str).value_counts().to_dict()
    if active_counts != EXPECTED_ACTIVE_COUNTS:
        raise RuntimeError(f"v4.2 active-set counts differ: {active_counts}")
    require_finite_positive(
        out,
        (
            *RAW_QUANTILE_COLUMNS,
            "eps2_mean",
            "eps2_obs",
            "ul_eps2_obs",
            "sigma_mass_res_GeV",
            "sigma_mass_res_min_GeV",
        ),
        "v4.2 combined bands",
    )
    require_finite(out, ("p0_analytic", "Z_analytic", "p_strong", "p_weak", "p_two"), "v4.2 combined bands")
    for column in ("p0_analytic", "p_strong", "p_weak", "p_two"):
        values = out[column].to_numpy(float)
        if not bool(((values >= 0.0) & (values <= 1.0)).all()):
            raise RuntimeError(f"v4.2 {column} lies outside [0,1]")
    quantiles = out[list(RAW_QUANTILE_COLUMNS)].to_numpy(float)
    if not bool((np.diff(quantiles, axis=1) >= 0.0).all()):
        raise RuntimeError("v4.2 expected-limit quantiles are unordered")
    for primary, alias in ALIAS_PAIRS:
        if not np.array_equal(out[primary].to_numpy(float), out[alias].to_numpy(float)):
            raise RuntimeError(f"v4.2 aliases disagree: {primary} != {alias}")

    exact_sets = {
        "cls_alpha": {0.1},
        "cls_statistic": {"tilde_q_mu"},
        "cls_calibration": {"asymptotic"},
        "combined_mode": {"count_scale"},
        "bands_train_exclude_nsigma": {2.25},
        "bands_refit_restarts": {0},
        "n_toys_requested": {N_TOYS},
        "n_toys_finite": {N_TOYS},
        "observed_gp_fit_mode": {"fixed_reviewed_max_lml"},
        "observed_gp_optimizer_restarts": {0},
        "limit_solver": {CACHE_VERSION},
        "profile_cache_limit_calls": {N_TOYS + 1},
    }
    for column, expected in exact_sets.items():
        values = set(out[column])
        if values != expected:
            raise RuntimeError(f"v4.2 {column} differs: {values} != {expected}")
    for column in ("bands_refit_gp_on_toy", "bands_refit_optimize"):
        out[column] = normalize_boolean(out[column], f"v4.2 combined bands.{column}")
        if bool(out[column].any()):
            raise RuntimeError(f"v4.2 {column} unexpectedly contains true")
    if out["bands_seed_sequence_index"].astype(int).tolist() != list(range(N_MASSES)):
        raise RuntimeError("v4.2 SeedSequence indices differ from mass_MeV - 19")
    for column in (
        "tail_count_strong_le_observed",
        "tail_count_weak_ge_observed",
        "tail_count_equal_observed",
        "tail_count_two_sided_min",
    ):
        values = out[column].to_numpy(float)
        if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
            raise RuntimeError(f"v4.2 {column} is not finite and integer-valued")
    for row in out.itertuples(index=False):
        validate_tail_row(row)
    return out


def validate_gp_state_metadata(
    bands: pd.DataFrame,
    states: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    state_lookup = states.copy()
    state_lookup["_mass_MeV_key"] = np.rint(state_lookup["mass_MeV"]).astype(int)
    if not np.allclose(
        state_lookup["mass_MeV"],
        state_lookup["_mass_MeV_key"],
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError("Enriched state mass_MeV is not integer-valued")
    state_lookup = state_lookup.set_index(["dataset", "_mass_MeV_key"], verify_integrity=True)
    augmented = bands.copy()
    closure_rows: List[Dict[str, Any]] = []
    source_maps: List[str] = []
    status_maps: List[str] = []

    for row in bands.itertuples(index=False):
        mass = float(row.mass_GeV)
        mass_mev = int(round(1000.0 * mass))
        active = str(row.dataset_set).split("+")
        lml = parse_json_mapping(row.gp_lml_by_dataset, f"band lml at {mass:.3f}")
        ls_opt = parse_json_mapping(row.gp_ls_opt_by_dataset, f"band ls at {mass:.3f}")
        const_opt = parse_json_mapping(row.gp_const_opt_by_dataset, f"band const at {mass:.3f}")
        state_hash = parse_json_mapping(
            row.gp_state_sha256_by_dataset,
            f"band state hash at {mass:.3f}",
        )
        metadata = parse_json_metadata(row.meta, f"band metadata at {mass:.3f}")
        for label, mapping in (
            ("lml", lml),
            ("ls", ls_opt),
            ("const", const_opt),
            ("state hash", state_hash),
            ("metadata", metadata),
        ):
            if set(mapping) != set(active):
                raise RuntimeError(f"{label} active-set mismatch at {mass:.3f} GeV")
        source_map: Dict[str, str] = {}
        status_map: Dict[str, str] = {}
        for dataset in active:
            try:
                reviewed = state_lookup.loc[(dataset, mass_mev)]
            except KeyError as error:
                raise RuntimeError(
                    f"Missing enriched {dataset} state at {mass:.3f} GeV"
                ) from error
            band_lml = float(lml[dataset])
            band_ls = float(ls_opt[dataset])
            band_const = float(const_opt[dataset])
            delta_lml = band_lml - float(reviewed["lml"])
            coordinate_pass = bool(
                np.isclose(band_ls, float(reviewed["ls_opt"]), rtol=1.0e-12, atol=1.0e-15)
                and np.isclose(
                    band_const,
                    float(reviewed["const_opt"]),
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            )
            lml_pass = bool(abs(delta_lml) <= LML_CLOSURE_ATOL)
            hash_value = str(state_hash[dataset]).lower()
            if len(hash_value) != 64 or any(c not in "0123456789abcdef" for c in hash_value):
                raise RuntimeError(f"Invalid GP-state SHA-256 for {dataset} at {mass:.3f}")
            meta = metadata[dataset]
            metadata_pass = bool(
                str(meta.get("state_sha256", "")).lower() == hash_value
                and np.isclose(float(meta.get("lml")), band_lml, rtol=0.0, atol=1.0e-12)
                and np.isclose(float(meta.get("ls_opt")), band_ls, rtol=0.0, atol=1.0e-15)
                and np.isclose(float(meta.get("const_opt")), band_const, rtol=0.0, atol=1.0e-15)
                and np.isclose(
                    float(meta.get("reviewed_lml")),
                    float(reviewed["lml"]),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.isclose(float(meta.get("lml_delta")), delta_lml, rtol=0.0, atol=1.0e-12)
                and np.isclose(
                    float(meta.get("sigma")),
                    float(reviewed["sigma_val"]),
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(meta.get("dens")),
                    float(reviewed["integral_density"]),
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            )
            if not (coordinate_pass and lml_pass and metadata_pass):
                raise RuntimeError(
                    f"Fixed reviewed GP-state closure fails for {dataset} at {mass:.3f} GeV"
                )
            source_map[dataset] = str(reviewed["row_source"])
            status_map[dataset] = str(reviewed["review_status"])
            closure_rows.append(
                {
                    "mass_GeV": mass,
                    "mass_MeV": mass_mev,
                    "dataset_set": str(row.dataset_set),
                    "dataset": dataset,
                    "band_lml": band_lml,
                    "reviewed_lml": float(reviewed["lml"]),
                    "delta_lml": delta_lml,
                    "band_ls_opt": band_ls,
                    "reviewed_ls_opt": float(reviewed["ls_opt"]),
                    "delta_ls_opt": band_ls - float(reviewed["ls_opt"]),
                    "band_const_opt": band_const,
                    "reviewed_const_opt": float(reviewed["const_opt"]),
                    "delta_const_opt": band_const - float(reviewed["const_opt"]),
                    "band_sigma_val": float(meta["sigma"]),
                    "reviewed_sigma_val": float(reviewed["sigma_val"]),
                    "delta_sigma_val": float(meta["sigma"]) - float(reviewed["sigma_val"]),
                    "band_integral_density": float(meta["dens"]),
                    "reviewed_integral_density": float(reviewed["integral_density"]),
                    "delta_integral_density": float(meta["dens"])
                    - float(reviewed["integral_density"]),
                    "band_state_sha256": hash_value,
                    "coordinate_closure_pass": coordinate_pass,
                    "lml_closure_pass": lml_pass,
                    "metadata_json_closure_pass": metadata_pass,
                    "fixed_reviewed_state_closure_pass": True,
                    "reviewed_row_source": str(reviewed["row_source"]),
                    "review_status": str(reviewed["review_status"]),
                    "branch_multiplicity": int(reviewed["branch_multiplicity"]),
                    "interpolated": False,
                }
            )
        source_maps.append(json.dumps(source_map, sort_keys=True))
        status_maps.append(json.dumps(status_map, sort_keys=True))
    closure = pd.DataFrame(closure_rows)
    if len(closure) != EXPECTED_STATE_COUNT:
        raise RuntimeError(f"GP-state closure has {len(closure)} rows; expected 415")
    augmented["observed_state_sources_by_dataset"] = source_maps
    augmented["observed_state_status_by_dataset"] = status_maps
    augmented["fixed_reviewed_state_metadata_validated"] = True
    return augmented, closure


def validate_accepted_combined(
    bands: pd.DataFrame,
) -> Dict[str, Any]:
    if sha256(ACCEPTED_COMBINED) != EXPECTED_ACCEPTED_COMBINED_SHA256:
        raise RuntimeError("Accepted v4.1 combined observed table SHA-256 changed")
    provenance = load_json(ACCEPTED_COMBINED_PROVENANCE)
    if provenance.get("output_csv_sha256") != EXPECTED_ACCEPTED_COMBINED_SHA256:
        raise RuntimeError("Accepted v4.1 combined provenance output hash fails")
    accepted = require_exact_combined_grid(
        pd.read_csv(ACCEPTED_COMBINED),
        "accepted v4.1 combined observed table",
    )
    if accepted["dataset_set"].astype(str).tolist() != expected_active_tags():
        raise RuntimeError("Accepted v4.1 combined active sets differ")
    for column in ("eps2_obs", "p0_analytic", "Z_analytic"):
        if not np.array_equal(
            bands[column].to_numpy(float),
            accepted[column].to_numpy(float),
        ):
            delta = np.max(
                np.abs(bands[column].to_numpy(float) - accepted[column].to_numpy(float))
            )
            raise RuntimeError(
                f"v4.2 {column} is not bitwise equal to accepted v4.1; max delta={delta}"
            )
    if not np.array_equal(
        bands["gp_state_sha256_by_dataset"].astype(str).to_numpy(),
        accepted["gp_state_sha256_by_dataset"].astype(str).to_numpy(),
    ):
        raise RuntimeError("v4.2 GP-state hashes differ from accepted v4.1")
    return {
        "path": repo_path(ACCEPTED_COMBINED),
        "sha256": sha256(ACCEPTED_COMBINED),
        "rows": int(len(accepted)),
        "eps2_obs_bitwise_equal": True,
        "p0_analytic_bitwise_equal": True,
        "Z_analytic_bitwise_equal": True,
        "gp_state_hashes_exact": True,
    }


def validate_unaffected_v4(bands: pd.DataFrame) -> Dict[str, Any]:
    if sha256(V4_BANDS) != EXPECTED_V4_BANDS_SHA256:
        raise RuntimeError("Accepted v4 300-toy band table SHA-256 changed")
    v4_provenance = load_json(V4_BANDS_PROVENANCE)
    if v4_provenance.get("output_csv_sha256") != EXPECTED_V4_BANDS_SHA256:
        raise RuntimeError("Accepted v4 band provenance output hash fails")
    previous = require_exact_combined_grid(pd.read_csv(V4_BANDS), "accepted v4 bands")
    unaffected = (bands["mass_GeV"] < 0.039) | (bands["mass_GeV"] > 0.180)
    if int(np.count_nonzero(unaffected)) != 90:
        raise RuntimeError("Unaffected mass mask is not exactly 90 points")
    common = [column for column in previous.columns if column in bands.columns]
    if set(common) != set(previous.columns):
        missing = sorted(set(previous.columns).difference(common))
        raise RuntimeError(f"v4.2 bands cannot close all v4 columns: {missing}")
    for column in common:
        left = bands.loc[unaffected, column].reset_index(drop=True)
        right = previous.loc[unaffected, column].reset_index(drop=True)
        compare_exact_series(left, right, f"unaffected v4/v4.2 {column}")
    return {
        "path": repo_path(V4_BANDS),
        "sha256": sha256(V4_BANDS),
        "unaffected_definition": "mass below 39 MeV or above 180 MeV",
        "n_unaffected_masses": 90,
        "n_columns_bitwise_or_exact": int(len(common)),
        "full_row_bitwise_or_exact_closure": True,
    }


def validate_individual_provenance(
    payload: Mapping[str, Any],
    individual_path: Path,
    states_path: Path,
) -> Dict[str, Any]:
    if payload.get("schema_version") != "hps-gpr-v4.2-individual-ledger-validation-1":
        raise RuntimeError("Individual ledger validation schema changed")
    if payload.get("status") != "pass":
        raise RuntimeError("Individual ledger validation did not pass")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("Individual ledger validation lacks an outputs mapping")
    expected_outputs = {
        "enriched": (
            states_path,
            EXPECTED_ENRICHED_COLUMNS,
            EXPECTED_STATE_COUNT,
        ),
        "individual": (
            individual_path,
            EXPECTED_INDIVIDUAL_COLUMNS,
            EXPECTED_STATE_COUNT,
        ),
    }
    for key, (path, columns, rows) in expected_outputs.items():
        entry = outputs.get(key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"Individual ledger validation lacks outputs.{key}")
        if entry.get("sha256") != sha256(path):
            raise RuntimeError(f"Individual ledger validation {key} hash fails")
        if int(entry.get("rows", -1)) != rows:
            raise RuntimeError(f"Individual ledger validation {key} row count fails")
        if tuple(entry.get("columns", ())) != tuple(columns):
            raise RuntimeError(f"Individual ledger validation {key} schema fails")
        if entry.get("path") != repo_path(path):
            raise RuntimeError(f"Individual ledger validation {key} path fails")
    interpretation = payload.get("interpretation")
    if not isinstance(interpretation, dict):
        raise RuntimeError("Individual ledger validation lacks interpretation metadata")
    exact_interpretation = {
        "combined_bands_only": True,
        "individual_bands_included": False,
        "individual_limits": "observed_only",
        "interpolation_permitted": False,
    }
    for key, expected in exact_interpretation.items():
        if interpretation.get(key) != expected:
            raise RuntimeError(
                f"Individual ledger interpretation {key} fails: "
                f"{interpretation.get(key)!r}"
            )
    dimuon = payload.get("minimal_visible_dimuon")
    if not isinstance(dimuon, dict):
        raise RuntimeError("Individual ledger validation lacks dimuon metadata")
    if (
        int(dimuon.get("corrected_row_count", -1)) != 39
        or int(dimuon.get("first_corrected_mass_MeV", -1)) != 212
        or not np.isclose(
            float(dimuon.get("threshold_GeV")),
            DIMUON_THRESHOLD_GEV,
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            float(dimuon.get("muon_mass_GeV")),
            M_MU_GEV,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise RuntimeError("Individual ledger dimuon validation metadata changed")
    frozen = payload.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise RuntimeError("Individual ledger validation lacks frozen inputs")
    expected_frozen = {
        repo_path(V41_CONFIG): sha256(V41_CONFIG),
        repo_path(ACCEPTED_COMPACT_STATES): sha256(ACCEPTED_COMPACT_STATES),
    }
    for path, expected_hash in expected_frozen.items():
        entry = frozen.get(path)
        if not isinstance(entry, dict) or entry.get("sha256") != expected_hash:
            raise RuntimeError(f"Individual ledger frozen-input hash fails for {path}")
    builder = payload.get("builder")
    if not isinstance(builder, dict):
        raise RuntimeError("Individual ledger validation lacks builder provenance")
    builder_path = REPO / str(builder.get("path"))
    if not builder_path.is_file() or builder.get("sha256") != sha256(builder_path):
        raise RuntimeError("Individual ledger builder provenance fails")
    return {
        "path": repo_path(DEFAULT_INDIVIDUAL_PROVENANCE),
        "sha256": sha256(DEFAULT_INDIVIDUAL_PROVENANCE),
        "output_hash_closure": True,
        "enriched_state_hash_closure": True,
        "accepted_config_hash_closure": True,
        "exact_schema_closure": True,
        "builder_hash_closure": True,
    }


def validate_individual(
    frame: pd.DataFrame,
    states: pd.DataFrame,
    bands: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if tuple(frame.columns) != EXPECTED_INDIVIDUAL_COLUMNS:
        raise RuntimeError(
            "v4.2 individual observed schema/order changed: "
            f"{tuple(frame.columns)} != {EXPECTED_INDIVIDUAL_COLUMNS}"
        )
    out = frame.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    if len(out) != EXPECTED_STATE_COUNT:
        raise RuntimeError(f"Individual table has {len(out)} rows; expected 415")
    if set(out["dataset"]) != set(SEARCH_RANGES_GEV):
        raise RuntimeError("Individual table has unexpected datasets")
    if bool(out.duplicated(["dataset", "mass_GeV"]).any()):
        raise RuntimeError("Individual table has duplicate states")
    require_finite_positive(
        out,
        (
            "A_up",
            "eps2_up",
            "eps2_observed_ee_channel",
            "minimal_visible_factor",
            "BR_ee_minimal",
            "eps2_observed_minimal_visible",
            "sigma_val",
            "integral_density",
            "const_opt",
            "ls_opt",
            "ls_hi",
        ),
        "v4.2 individual observed limits",
    )
    require_finite(
        out,
        ("p0_analytic", "Z_analytic", "lml"),
        "v4.2 individual observed limits",
    )
    if not bool(
        (
            (out["p0_analytic"].to_numpy(float) >= 0.0)
            & (out["p0_analytic"].to_numpy(float) <= 1.0)
        ).all()
    ):
        raise RuntimeError("Individual local p0 lies outside [0,1]")
    if set(out["sample_label"].astype(str)) != {"2015 100%", "2016 100%", "2021 10%"}:
        raise RuntimeError("Individual sample labels differ from the accepted data fractions")
    if set(out["limit_scope"].astype(str)) != {"individual_observed_only"}:
        raise RuntimeError("Individual limit scope is not observed-only")
    if bool(normalize_boolean(out["bands_included"], "individual bands_included").any()):
        raise RuntimeError("Individual expected bands were included")
    if bool(normalize_boolean(out["interpolated"], "individual interpolated").any()):
        raise RuntimeError("Individual table contains interpolation")
    individual_status_counts = {
        (str(dataset), str(status)): int(count)
        for (dataset, status), count in out.groupby(
            ["dataset", "review_status"]
        ).size().items()
    }
    expected_individual_status_counts = {
        ("2015", "resolved_reproduced_max_lml"): 72,
        ("2016", "raw_scan_row"): 139,
        ("2016", "repair_selected_reproduced_max_lml"): 3,
        ("2021", "resolved_reproduced_max_lml"): 201,
    }
    if individual_status_counts != expected_individual_status_counts:
        raise RuntimeError("Individual review-status composition changed")
    individual_raw_2016 = out[
        (out["dataset"] == "2016")
        & (out["review_status"] == "raw_scan_row")
    ]
    individual_reproduced = out[out["review_status"] != "raw_scan_row"]
    if set(individual_raw_2016["branch_multiplicity"].astype(int)) != {1}:
        raise RuntimeError("Individual 2016 raw-scan branch multiplicity changed")
    if not bool(
        (individual_reproduced["branch_multiplicity"].to_numpy(int) >= 2).all()
    ):
        raise RuntimeError("A reproduced individual state lacks an independent branch")
    if set(out["accepted_config"].astype(str)) != {repo_path(V41_CONFIG)}:
        raise RuntimeError("Individual table cites another accepted physics card")
    if set(out["accepted_config_sha256"].astype(str)) != {sha256(V41_CONFIG)}:
        raise RuntimeError("Individual accepted-config SHA-256 provenance fails")
    if set(out["source_enriched_ledger"].astype(str)) != {repo_path(DEFAULT_STATES)}:
        raise RuntimeError("Individual table cites another enriched state ledger")
    if set(out["source_enriched_ledger_sha256"].astype(str)) != {
        sha256(DEFAULT_STATES)
    }:
        raise RuntimeError("Individual enriched-state SHA-256 provenance fails")

    expected_factor = dimuon_factor(out["mass_GeV"].to_numpy(float))
    if not np.array_equal(
        out["eps2_up"].to_numpy(float),
        out["eps2_observed_ee_channel"].to_numpy(float),
    ):
        raise RuntimeError("Individual e+e- observed epsilon2 alias differs from eps2_up")
    if not np.allclose(
        out["minimal_visible_factor"],
        expected_factor,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Individual minimal-visible factor differs from the frozen formula")
    if not np.allclose(
        out["BR_ee_minimal"],
        1.0 / expected_factor,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Individual minimal-visible e+e- branching fraction differs")
    if not np.allclose(
        out["eps2_observed_minimal_visible"],
        out["eps2_up"].to_numpy(float) * expected_factor,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Individual minimal-visible epsilon2 conversion differs")
    corrected = normalize_boolean(
        out["dimuon_correction_applied"],
        "individual dimuon_correction_applied",
    )
    expected_corrected = out["mass_GeV"].to_numpy(float) > DIMUON_THRESHOLD_GEV
    if not np.array_equal(corrected.to_numpy(bool), expected_corrected):
        raise RuntimeError("Individual dimuon-correction mask differs")
    if int(np.count_nonzero(expected_corrected)) != 39:
        raise RuntimeError("Individual dimuon correction does not cover exactly 39 rows")

    state_lookup = states.set_index(["dataset", "mass_GeV"], verify_integrity=True)
    normalized_rows: List[pd.Series] = []
    closure_columns = (
        "mass_MeV",
        "A_up",
        "eps2_up",
        "p0_analytic",
        "Z_analytic",
        "sigma_val",
        "integral_density",
        "const_opt",
        "ls_opt",
        "ls_hi",
        "lml",
        "selected_attempt",
        "selected_source",
        "selected_source_sha256",
        "row_source",
        "optimizer_repair_applied",
        "review_status",
        "branch_multiplicity",
        "interpolated",
    )
    band_hash_lookup: Dict[Tuple[str, int], str] = {}
    for row in bands.itertuples(index=False):
        mass_mev = int(round(1000.0 * float(row.mass_GeV)))
        mapping = parse_json_mapping(
            row.gp_state_sha256_by_dataset,
            f"combined state hashes at {row.mass_GeV:.3f}",
        )
        for dataset, value in mapping.items():
            band_hash_lookup[(dataset, mass_mev)] = str(value)
    for dataset in SEARCH_RANGES_GEV:
        rows = out[out["dataset"] == dataset].sort_values("mass_GeV")
        grid = expected_dataset_grid(dataset)
        if len(rows) != EXPECTED_STATE_ROWS[dataset] or not np.allclose(
            rows["mass_GeV"],
            grid,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{dataset} individual grid is incomplete")
        for _, row in rows.iterrows():
            mass = float(row["mass_GeV"])
            mass_mev = int(round(1000.0 * mass))
            try:
                state = state_lookup.loc[(dataset, mass)]
            except KeyError as error:
                raise RuntimeError(
                    f"Missing enriched state for individual {dataset} at {mass:.3f}"
                ) from error
            for column in closure_columns:
                left = row[column]
                right = state[column]
                if isinstance(left, (float, np.floating)) or isinstance(
                    right, (float, np.floating)
                ):
                    equal = bool(
                        np.array_equal(
                            np.asarray([left], dtype=float),
                            np.asarray([right], dtype=float),
                            equal_nan=True,
                        )
                    )
                else:
                    equal = str(left) == str(right)
                if not equal:
                    raise RuntimeError(
                        f"Individual/enriched exact closure fails for {dataset} "
                        f"at {mass:.3f} in {column}"
                    )
            if (dataset, mass_mev) not in band_hash_lookup:
                raise RuntimeError(
                    f"Combined state hash missing for individual {dataset} at {mass:.3f}"
                )
            row = row.copy()
            row["gp_state_sha256"] = band_hash_lookup[(dataset, mass_mev)]
            normalized_rows.append(row)
    normalized = pd.DataFrame(normalized_rows).reset_index(drop=True)
    normalized["eps2_obs"] = normalized["eps2_up"].to_numpy(float)
    normalized["sigma_mass_res_GeV"] = normalized["sigma_val"].to_numpy(float)
    normalized["cls_alpha"] = 0.1
    normalized["cls_statistic"] = "tilde_q_mu"
    normalized["cls_calibration"] = "asymptotic"
    normalized["observed_gp_fit_mode"] = "fixed_reviewed_max_lml"
    normalized["observed_gp_optimizer_restarts"] = 0
    normalized["toy_draws"] = 0
    normalized["expected_bands_produced"] = False
    return normalized, {
        "path": repo_path(DEFAULT_INDIVIDUAL),
        "sha256": sha256(DEFAULT_INDIVIDUAL),
        "rows": int(len(normalized)),
        "dataset_rows": EXPECTED_STATE_ROWS,
        "no_individual_bands": True,
        "exact_enriched_ledger_closure": True,
        "fixed_state_hashes_attached_from_validated_combined_rows": True,
        "minimal_visible_formula_closure": True,
    }


def dimuon_factor(mass_gev: np.ndarray) -> np.ndarray:
    masses = np.asarray(mass_gev, dtype=float)
    grid_masses = np.rint(1000.0 * masses) / 1000.0
    if not np.allclose(masses, grid_masses, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Minimal-visible conversion received an off-grid mass")
    masses = grid_masses
    factor = np.ones_like(masses)
    above = masses > DIMUON_THRESHOLD_GEV
    if np.any(above):
        phase_space = np.sqrt(1.0 - 4.0 * M_MU_GEV**2 / masses[above] ** 2)
        phase_space *= 1.0 + 2.0 * M_MU_GEV**2 / masses[above] ** 2
        factor[above] = 1.0 + phase_space
    return factor


def empirical_tail_label(count: int, probability: float, two_sided: bool = False) -> str:
    if count == 0:
        return (
            "2*0/300; no pseudoexperiment in the smaller empirical tail; "
            "not an exact zero probability"
            if two_sided
            else "0/300; below one-count resolution 1/300, not exact zero"
        )
    if two_sided:
        return f"2*{count}/300, bounded at 1 = {probability:.12g}"
    return f"{count}/300 = {probability:.12g}"


def build_reviewed_combined(
    bands: pd.DataFrame,
    bands_path: Path,
    states_path: Path,
) -> pd.DataFrame:
    out = bands.copy()
    out["mass_MeV"] = np.rint(1000.0 * out["mass_GeV"]).astype(int)
    factor = dimuon_factor(out["mass_GeV"].to_numpy(float))
    out["dimuon_threshold_GeV"] = DIMUON_THRESHOLD_GEV
    out["dimuon_threshold_MeV"] = 1000.0 * DIMUON_THRESHOLD_GEV
    out["N_eff_BR"] = factor
    out["BR_ee_minimal"] = 1.0 / factor
    out["dimuon_correction_applied"] = out["mass_GeV"] > DIMUON_THRESHOLD_GEV
    for column in COUPLING_COLUMNS:
        out[f"{column}_ee_channel"] = out[column].to_numpy(float)
        out[f"{column}_minimal_visible"] = out[column].to_numpy(float) * factor
    out["observed_over_median_ee_channel"] = (
        out["eps2_obs_ee_channel"] / out["eps2_med_ee_channel"]
    )
    out["observed_over_median_minimal_visible"] = (
        out["eps2_obs_minimal_visible"] / out["eps2_med_minimal_visible"]
    )
    if not np.allclose(
        out["observed_over_median_ee_channel"],
        out["observed_over_median_minimal_visible"],
        rtol=5.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Minimal-visible common factor fails to cancel in obs/median")
    out["p_strong_empirical_label"] = [
        empirical_tail_label(int(count), float(probability))
        for count, probability in zip(
            out["tail_count_strong_le_observed"],
            out["p_strong"],
        )
    ]
    out["p_weak_empirical_label"] = [
        empirical_tail_label(int(count), float(probability))
        for count, probability in zip(
            out["tail_count_weak_ge_observed"],
            out["p_weak"],
        )
    ]
    out["p_two_empirical_label"] = [
        empirical_tail_label(int(count), float(probability), two_sided=True)
        for count, probability in zip(
            out["tail_count_two_sided_min"],
            out["p_two"],
        )
    ]
    out["coverage_calibrated"] = False
    out["scan_toy_calibrated"] = False
    out["tail_pvalue_family"] = "fixed-mass empirical observed-limit diagnostic"
    out["local_p0_family"] = "local asymptotic discovery statistic"
    out["source_bands_table"] = repo_path(bands_path)
    out["source_bands_sha256"] = sha256(bands_path)
    out["source_observed_states"] = repo_path(states_path)
    out["source_observed_states_sha256"] = sha256(states_path)
    return out


def build_reviewed_individual(
    individual: pd.DataFrame,
    individual_path: Path,
) -> pd.DataFrame:
    out = individual.copy()
    factor = dimuon_factor(out["mass_GeV"].to_numpy(float))
    out["N_eff_BR"] = factor
    out["BR_ee_minimal"] = 1.0 / factor
    out["dimuon_correction_applied"] = out["mass_GeV"] > DIMUON_THRESHOLD_GEV
    out["eps2_obs_ee_channel"] = out["eps2_obs"].to_numpy(float)
    out["eps2_obs_minimal_visible"] = out["eps2_obs"].to_numpy(float) * factor
    out["standalone_fit"] = True
    out["combined_fit_component"] = False
    out["expected_bands_produced"] = False
    out["source_individual_table"] = repo_path(individual_path)
    out["source_individual_sha256"] = sha256(individual_path)
    return out


def effective_trials_from_spacing(masses: np.ndarray, sigma_values: np.ndarray) -> float:
    masses = np.asarray(masses, dtype=float)
    sigma_values = np.asarray(sigma_values, dtype=float)
    if masses.size != sigma_values.size or masses.size < 2:
        raise RuntimeError("Mass and resolution arrays do not align")
    delta = np.diff(masses)
    sigma_mid = 0.5 * (sigma_values[:-1] + sigma_values[1:])
    if not bool(
        (
            np.isfinite(delta)
            & (delta > 0.0)
            & np.isfinite(sigma_mid)
            & (sigma_mid > 0.0)
        ).all()
    ):
        raise RuntimeError("Resolution-spacing Sidak inputs are incomplete")
    value = np.sum(delta / (INDEPENDENCE_WIDTH_SIGMA * sigma_mid))
    return float(np.clip(value, 1.0, float(masses.size)))


def sidak_values(local: np.ndarray, neff: float) -> np.ndarray:
    clipped = np.clip(np.asarray(local, dtype=float), 1.0e-300, 1.0)
    values = -np.expm1(neff * np.log1p(-clipped))
    return np.clip(values, 1.0e-300, 1.0)


def build_combined_sidak(reviewed: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    out = reviewed[
        [
            "mass_GeV",
            "mass_MeV",
            "dataset_set",
            "p0_analytic",
            "Z_analytic",
            "sigma_mass_res_min_GeV",
            "sigma_mass_res_GeV",
        ]
    ].copy()
    neff = effective_trials_from_spacing(
        out["mass_GeV"],
        out["sigma_mass_res_min_GeV"],
    )
    out["p0_local_asymptotic"] = np.clip(out["p0_analytic"], 1.0e-300, 1.0)
    out["Z_local_asymptotic"] = out["Z_analytic"]
    out["p_sidak_resolution_spacing_analytic"] = sidak_values(
        out["p0_local_asymptotic"],
        neff,
    )
    out["Z_sidak_resolution_spacing_analytic"] = norm.isf(
        out["p_sidak_resolution_spacing_analytic"]
    )
    out["N_eff_resolution_spacing"] = neff
    out["independence_width_sigma"] = INDEPENDENCE_WIDTH_SIGMA
    out["is_scan_minimum"] = False
    out.loc[int(np.argmin(out["p0_local_asymptotic"].to_numpy(float))), "is_scan_minimum"] = True
    out["scan_toy_calibrated"] = False
    out["uses_limit_tail_pvalues"] = False
    out["interpretation"] = (
        "analytic resolution-spacing Sidak reference; not scan-toy calibrated"
    )
    return out, neff


def build_individual_sidak(reviewed: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frames: List[pd.DataFrame] = []
    neff_by_dataset: Dict[str, float] = {}
    for dataset in SEARCH_RANGES_GEV:
        rows = reviewed[reviewed["dataset"] == dataset].sort_values("mass_GeV").copy()
        neff = effective_trials_from_spacing(
            rows["mass_GeV"],
            rows["sigma_mass_res_GeV"],
        )
        neff_by_dataset[dataset] = neff
        rows["p0_local_asymptotic"] = np.clip(rows["p0_analytic"], 1.0e-300, 1.0)
        rows["Z_local_asymptotic"] = rows["Z_analytic"]
        rows["p_sidak_resolution_spacing_analytic"] = sidak_values(
            rows["p0_local_asymptotic"],
            neff,
        )
        rows["Z_sidak_resolution_spacing_analytic"] = norm.isf(
            rows["p_sidak_resolution_spacing_analytic"]
        )
        rows["N_eff_resolution_spacing"] = neff
        rows["independence_width_sigma"] = INDEPENDENCE_WIDTH_SIGMA
        rows["is_dataset_scan_minimum"] = False
        minimum = rows.index[int(np.argmin(rows["p0_local_asymptotic"].to_numpy(float)))]
        rows.loc[minimum, "is_dataset_scan_minimum"] = True
        rows["scan_toy_calibrated"] = False
        rows["interpretation"] = (
            "standalone local asymptotic p0 with analytic resolution-spacing "
            "Sidak reference; not scan-toy calibrated"
        )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True), neff_by_dataset


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 10.8,
            "axes.titlesize": 14.0,
            "axes.labelsize": 11.8,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.55,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 9.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_mass_ticks(ax: plt.Axes, low: float = MASS_LOW_MEV, high: float = MASS_HIGH_MEV) -> None:
    first = int(math.ceil(low / 10.0) * 10)
    majors = np.arange(first, high + 0.1, 10.0)
    ax.set_xlim(float(low), float(high))
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))


def contiguous_segments(frame: pd.DataFrame, category: str = "dataset_set") -> Iterator[pd.DataFrame]:
    work = frame.sort_values("mass_GeV").reset_index(drop=True)
    groups = work[category].astype(str).ne(work[category].astype(str).shift()).cumsum()
    for _, segment in work.groupby(groups, sort=False):
        yield segment


def plot_activity_strip(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for segment in contiguous_segments(frame):
        key = str(segment["dataset_set"].iloc[0])
        x0 = float(segment["mass_MeV"].min()) - 0.5
        x1 = float(segment["mass_MeV"].max()) + 0.5
        ax.axvspan(
            x0,
            x1,
            ymin=0.08,
            ymax=0.92,
            facecolor=ACTIVE_COLORS[key],
            edgecolor="white",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (x0 + x1),
            0.50,
            ACTIVE_LABELS[key],
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
            fontsize=8.8,
            color="#30343B",
        )
    ax.set_xlim(float(MASS_LOW_MEV), float(MASS_HIGH_MEV))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Active", rotation=0, ha="right", va="center", labelpad=8)


def save_figure(fig: plt.Figure, stem: str, description: str) -> Dict[str, Any]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {
        "stem": stem,
        "description": description,
        "pdf": repo_path(pdf),
        "pdf_sha256": sha256(pdf),
        "pdf_bytes": pdf.stat().st_size,
        "png": repo_path(png),
        "png_sha256": sha256(png),
        "png_bytes": png.stat().st_size,
    }


def plot_combined_bands(reviewed: pd.DataFrame) -> Dict[str, Any]:
    fig = plt.figure(figsize=(12.4, 7.45))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.17, 3.2, 0.95),
        hspace=0.045,
        left=0.09,
        right=0.98,
        top=0.82,
        bottom=0.09,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    ratio_ax = fig.add_subplot(grid[2], sharex=activity)
    plot_activity_strip(activity, reviewed)
    x = reviewed["mass_MeV"].to_numpy(float)
    threshold_mev = 1000.0 * DIMUON_THRESHOLD_GEV
    ax.fill_between(
        x,
        reviewed["eps2_lo2_minimal_visible"],
        reviewed["eps2_hi2_minimal_visible"],
        color=COLORS["band2"],
        alpha=0.76,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        x,
        reviewed["eps2_lo1_minimal_visible"],
        reviewed["eps2_hi1_minimal_visible"],
        color=COLORS["band1"],
        alpha=0.84,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        x,
        reviewed["eps2_med_minimal_visible"],
        color=COLORS["expected"],
        linewidth=1.65,
        linestyle="--",
        zorder=3,
    )
    ax.plot(
        x,
        reviewed["eps2_obs_minimal_visible"],
        color=COLORS["observed"],
        linewidth=2.05,
        zorder=4,
    )
    ax.axvline(threshold_mev, color=COLORS["threshold"], linewidth=1.0, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylabel(r"90% CL upper limit on minimal-visible $\epsilon^2$")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ratio = reviewed["observed_over_median_minimal_visible"].to_numpy(float)
    ratio_ax.plot(x, ratio, color=COLORS["observed"], linewidth=1.85)
    ratio_ax.axhline(1.0, color="#6B7280", linewidth=0.9, linestyle="--")
    ratio_ax.axvline(threshold_mev, color=COLORS["threshold"], linewidth=1.0, linestyle=":")
    ratio_ax.set_ylabel("obs / median")
    ratio_ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ratio_ax.set_ylim(max(0.0, 0.92 * float(np.min(ratio))), 1.08 * float(np.max(ratio)))
    set_mass_ticks(ratio_ax)
    handles = [
        Patch(
            facecolor=COLORS["band2"],
            alpha=0.76,
            label="Central 95% fixed-GP toy-limit interval",
        ),
        Patch(
            facecolor=COLORS["band1"],
            alpha=0.84,
            label="Central 68% fixed-GP toy-limit interval",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["expected"],
            linewidth=1.65,
            linestyle="--",
            label="Fixed-GP toy-limit median",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["observed"],
            linewidth=2.05,
            label="Observed 90% CL",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["threshold"],
            linewidth=1.0,
            linestyle=":",
            label=rf"$2m_\mu={threshold_mev:.3f}$ MeV",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.91),
        frameon=False,
        ncol=3,
        handlelength=2.7,
    )
    fig.suptitle(
        "Combined HPS observed limit and fixed-GP toy quantiles",
        y=0.975,
        fontweight="semibold",
    )
    # Intentionally no footer: the requested Figure-47 reproduction reclaims
    # this space.  Statistical semantics are retained in the README/summary.
    return save_figure(
        fig,
        "combined_observed_bands300_minimal_visible",
        "Figure-47-style combined observed minimal-visible limit, central "
        "68/95% fixed-GP toy-limit quantiles, active-set strip, and "
        "observed/median panel; requested footer removed.",
    )


def plot_empirical_tails(reviewed: pd.DataFrame) -> Dict[str, Any]:
    fig = plt.figure(figsize=(12.4, 6.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.16, 1.0),
        hspace=0.04,
        left=0.09,
        right=0.98,
        top=0.82,
        bottom=0.18,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    plot_activity_strip(activity, reviewed)
    x = reviewed["mass_MeV"].to_numpy(float)
    specs = (
        ("p_strong", "tail_count_strong_le_observed", COLORS["strong"], r"$p_{\rm strong}$", "-"),
        ("p_weak", "tail_count_weak_ge_observed", COLORS["weak"], r"$p_{\rm weak}$", "-"),
        ("p_two", "tail_count_two_sided_min", COLORS["two"], r"$p_{\rm two}$", "--"),
    )
    zero_y = 0.5 / N_TOYS
    for p_column, count_column, color, label, linestyle in specs:
        values = reviewed[p_column].to_numpy(float)
        counts = reviewed[count_column].to_numpy(int)
        nonzero = counts > 0
        plotted = values.copy()
        plotted[~nonzero] = np.nan
        ax.plot(x, plotted, color=color, linewidth=1.65, linestyle=linestyle, label=label)
        if bool((~nonzero).any()):
            ax.scatter(
                x[~nonzero],
                np.full(np.count_nonzero(~nonzero), zero_y),
                marker="v",
                s=22,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=5,
            )
    ax.axhline(
        1.0 / N_TOYS,
        color="#6B7280",
        linewidth=1.0,
        linestyle=":",
        label=r"One-count resolution $1/300$",
    )
    ax.set_yscale("log")
    ax.set_ylim(0.75 / (2.0 * N_TOYS), 1.08)
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax.set_ylabel("Empirical fixed-mass tail fraction")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    set_mass_ticks(ax)
    handles, labels = ax.get_legend_handles_labels()
    zero_handle = Line2D(
        [0],
        [0],
        marker="v",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="#4B5563",
        label="0/300 marker (shown below 1/300)",
    )
    handles.append(zero_handle)
    labels.append(zero_handle.get_label())
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.91),
        frameon=False,
        ncol=3,
    )
    fig.suptitle("Observed-limit empirical tail diagnostics", y=0.975, fontweight="semibold")
    fig.text(
        0.50,
        0.055,
        "A downward marker denotes a 0/300 raw tail count, not exact p=0. "
        "These are limit diagnostics, not discovery p-values.",
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_limit_tail_pvalues300",
        "Combined strong, weak, and bounded two-sided empirical fixed-mass "
        "observed-limit diagnostics with raw-zero markers.",
    )


def plot_combined_local_p0(sidak: pd.DataFrame, neff: float) -> Dict[str, Any]:
    fig = plt.figure(figsize=(12.4, 6.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.16, 1.0),
        hspace=0.04,
        left=0.09,
        right=0.98,
        top=0.82,
        bottom=0.18,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    plot_activity_strip(activity, sidak)
    x = sidak["mass_MeV"].to_numpy(float)
    local = sidak["p0_local_asymptotic"].to_numpy(float)
    reference = sidak["p_sidak_resolution_spacing_analytic"].to_numpy(float)
    ax.plot(x, local, color=COLORS["local"], linewidth=1.9, label=r"Local asymptotic $p_0$")
    ax.plot(
        x,
        reference,
        color=COLORS["sidak"],
        linewidth=1.8,
        linestyle="--",
        label=rf"Analytic Sidak reference ($N_{{\rm eff}}={neff:.2f}$, $W=2.25\sigma_m$)",
    )
    minimum = sidak[sidak["is_scan_minimum"].astype(bool)].iloc[0]
    ax.scatter(
        [float(minimum["mass_MeV"])],
        [float(minimum["p0_local_asymptotic"])],
        s=42,
        color=COLORS["local"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    positive = np.concatenate((local[local > 0.0], reference[reference > 0.0]))
    lower = max(1.0e-8, 10.0 ** math.floor(math.log10(float(np.min(positive)))) / 2.0)
    ax.set_yscale("log")
    ax.set_ylim(lower, 1.08)
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax.set_ylabel("One-sided p-value")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    set_mass_ticks(ax)
    fig.legend(loc="upper center", bbox_to_anchor=(0.50, 0.91), frameon=False, ncol=2)
    fig.suptitle(
        "Combined local asymptotic p-value and analytic look-elsewhere reference",
        y=0.975,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.055,
        "The Sidak curve is an analytic resolution-spacing reference, not a "
        "scan-toy calibration and not a limit-tail p-value.",
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_local_p0_sidak_reference",
        "Combined local asymptotic p0 and separate analytic "
        "resolution-spacing Sidak reference.",
    )


def plot_individual_limits(
    individual: pd.DataFrame,
    combined: pd.DataFrame,
) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14)
    for dataset in SEARCH_RANGES_GEV:
        rows = individual[individual["dataset"] == dataset].sort_values("mass_GeV")
        display_dataset = "2021 10%" if dataset == "2021" else dataset
        ax.plot(
            rows["mass_MeV"],
            rows["eps2_obs_minimal_visible"],
            color=COLORS[dataset],
            linewidth=1.9,
            label=f"{display_dataset} standalone observed 90% CL",
        )
    ax.plot(
        combined["mass_MeV"],
        combined["eps2_obs_minimal_visible"],
        color="#111111",
        linewidth=2.25,
        label="Combined observed 90% CL",
        zorder=5,
    )
    threshold_mev = 1000.0 * DIMUON_THRESHOLD_GEV
    ax.axvline(
        threshold_mev,
        color=COLORS["threshold"],
        linewidth=1.0,
        linestyle=":",
        label=rf"$2m_\mu={threshold_mev:.3f}$ MeV",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax.set_ylabel(r"90% CL upper limit on minimal-visible $\epsilon^2$")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    set_mass_ticks(ax)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.suptitle(
        "Observed limits by HPS dataset and combined search",
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.035,
        "Colored curves are standalone fixed-state fits. The black combined "
        "likelihood result is not an envelope of the individual curves.",
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "individual_observed_limits_minimal_visible",
        "Standalone observed-only 2015, 2016, and 2021 10% minimal-visible "
        "limits with the combined observed result in black for context; no "
        "individual expected bands.",
    )


def plot_individual_local_p0(
    sidak: pd.DataFrame,
    neff_by_dataset: Mapping[str, float],
) -> Dict[str, Any]:
    fig, axes = plt.subplots(3, 1, figsize=(12.4, 9.0), sharey=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.09, hspace=0.19)
    for index, (ax, dataset) in enumerate(zip(axes, SEARCH_RANGES_GEV)):
        rows = sidak[sidak["dataset"] == dataset].sort_values("mass_GeV")
        display_dataset = "2021 10%" if dataset == "2021" else dataset
        x = rows["mass_MeV"].to_numpy(float)
        local = rows["p0_local_asymptotic"].to_numpy(float)
        reference = rows["p_sidak_resolution_spacing_analytic"].to_numpy(float)
        ax.plot(
            x,
            local,
            color=COLORS[dataset],
            linewidth=1.9,
            label=rf"{display_dataset} local asymptotic $p_0$",
        )
        ax.plot(
            x,
            reference,
            color=COLORS["sidak"],
            linewidth=1.55,
            linestyle="--",
            label=rf"Analytic Sidak reference ($N_{{\rm eff}}={neff_by_dataset[dataset]:.2f}$)",
        )
        minimum = rows[rows["is_dataset_scan_minimum"].astype(bool)].iloc[0]
        ax.scatter(
            [float(minimum["mass_MeV"])],
            [float(minimum["p0_local_asymptotic"])],
            s=36,
            color=COLORS[dataset],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        positive = np.concatenate((local[local > 0.0], reference[reference > 0.0]))
        lower = max(1.0e-8, 10.0 ** math.floor(math.log10(float(np.min(positive)))) / 2.0)
        ax.set_yscale("log")
        ax.set_ylim(lower, 1.08)
        ax.set_ylabel("One-sided p-value")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        low, high = SEARCH_RANGES_GEV[dataset]
        set_mass_ticks(ax, 1000.0 * low, 1000.0 * high)
        ax.legend(loc="lower right", frameon=False, ncol=2)
        ax.text(
            0.012,
            0.08,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            fontweight="semibold",
        )
    axes[-1].set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    fig.suptitle(
        "Standalone local asymptotic p-values by HPS dataset",
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.025,
        "Dashed curves are analytic resolution-spacing references, not "
        "scan-toy calibrations; no individual limit-tail ensembles were run.",
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "individual_local_p0",
        "Three standalone local asymptotic p0 panels with separate analytic "
        "resolution-spacing Sidak references.",
    )


def minimum_record(
    frame: pd.DataFrame,
    value_column: str,
    mass_column: str = "mass_MeV",
) -> Dict[str, Any]:
    values = frame[value_column].to_numpy(float)
    minimum = float(np.min(values))
    mask = np.isclose(values, minimum, rtol=1.0e-12, atol=1.0e-300)
    masses = [int(round(value)) for value in frame.loc[mask, mass_column].to_numpy(float)]
    return {"value": minimum, "masses_MeV": masses}


def build_summary(
    combined: pd.DataFrame,
    combined_sidak: pd.DataFrame,
    combined_neff: float,
    individual: pd.DataFrame,
    individual_sidak: pd.DataFrame,
    individual_neff: Mapping[str, float],
    inputs: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    tails: Dict[str, Any] = {}
    for name, p_column, count_column, two_sided in (
        ("strong", "p_strong", "tail_count_strong_le_observed", False),
        ("weak", "p_weak", "tail_count_weak_ge_observed", False),
        ("two_sided", "p_two", "tail_count_two_sided_min", True),
    ):
        record = minimum_record(combined, p_column)
        at_minimum = combined[
            np.isclose(
                combined[p_column],
                record["value"],
                rtol=1.0e-12,
                atol=1.0e-300,
            )
        ]
        counts = sorted({int(value) for value in at_minimum[count_column]})
        zero_masses = [
            int(round(value))
            for value in combined.loc[combined[count_column] == 0, "mass_MeV"]
        ]
        record.update(
            {
                "raw_counts_at_minimum": counts,
                "zero_count_masses_MeV": zero_masses,
                "n_zero_count_masses": len(zero_masses),
                "one_count_resolution": 1.0 / N_TOYS,
                "interpretation": empirical_tail_label(
                    min(counts),
                    record["value"],
                    two_sided=two_sided,
                ),
            }
        )
        tails[name] = record
        rows.append(
            {
                "family": "combined_empirical_limit_tail",
                "dataset": "combined",
                "metric": f"minimum_p_{name}",
                "value": record["value"],
                "masses_MeV": json.dumps(record["masses_MeV"]),
                "raw_counts": json.dumps(counts),
                "interpretation": record["interpretation"],
            }
        )

    combined_local = minimum_record(combined_sidak, "p0_local_asymptotic")
    combined_minimum_row = combined_sidak.loc[
        int(np.argmin(combined_sidak["p0_local_asymptotic"].to_numpy(float)))
    ]
    combined_local["Z_local_asymptotic"] = float(combined_minimum_row["Z_local_asymptotic"])
    combined_local["sidak_p_at_local_minimum"] = float(
        combined_minimum_row["p_sidak_resolution_spacing_analytic"]
    )
    combined_local["sidak_Z_at_local_minimum"] = float(
        combined_minimum_row["Z_sidak_resolution_spacing_analytic"]
    )
    combined_local["N_eff_resolution_spacing"] = combined_neff
    rows.append(
        {
            "family": "combined_local_asymptotic",
            "dataset": "combined",
            "metric": "minimum_local_p0",
            "value": combined_local["value"],
            "masses_MeV": json.dumps(combined_local["masses_MeV"]),
            "raw_counts": "",
            "interpretation": "local asymptotic discovery statistic",
        }
    )

    individual_local: Dict[str, Any] = {}
    for dataset in SEARCH_RANGES_GEV:
        here = individual_sidak[individual_sidak["dataset"] == dataset]
        record = minimum_record(here, "p0_local_asymptotic")
        row = here.loc[int(here["p0_local_asymptotic"].idxmin())]
        record["Z_local_asymptotic"] = float(row["Z_local_asymptotic"])
        record["sidak_p_at_local_minimum"] = float(
            row["p_sidak_resolution_spacing_analytic"]
        )
        record["sidak_Z_at_local_minimum"] = float(
            row["Z_sidak_resolution_spacing_analytic"]
        )
        record["N_eff_resolution_spacing"] = float(individual_neff[dataset])
        individual_local[dataset] = record
        rows.append(
            {
                "family": "individual_local_asymptotic",
                "dataset": dataset,
                "metric": "minimum_local_p0",
                "value": record["value"],
                "masses_MeV": json.dumps(record["masses_MeV"]),
                "raw_counts": "",
                "interpretation": "standalone local asymptotic discovery statistic",
            }
        )

    ratio = combined["observed_over_median_minimal_visible"].to_numpy(float)
    ratio_min = minimum_record(combined, "observed_over_median_minimal_visible")
    maximum = float(np.max(ratio))
    max_mask = np.isclose(ratio, maximum, rtol=1.0e-12, atol=1.0e-300)
    ratio_max = {
        "value": maximum,
        "masses_MeV": [
            int(round(value))
            for value in combined.loc[max_mask, "mass_MeV"].to_numpy(float)
        ],
    }

    summary = {
        "schema_version": 1,
        "created_utc": now_utc(),
        "campaign": CAMPAIGN_DIR.name,
        "status": "validated postprocessing complete",
        "inputs": dict(inputs),
        "grid": {
            "mass_low_MeV": MASS_LOW_MEV,
            "mass_high_MeV": MASS_HIGH_MEV,
            "step_MeV": 1,
            "n_combined_masses": N_MASSES,
            "n_reviewed_states": EXPECTED_STATE_COUNT,
            "active_set_counts": EXPECTED_ACTIVE_COUNTS,
            "individual_state_counts": EXPECTED_STATE_ROWS,
        },
        "ensemble": {
            "n_toys_per_mass": N_TOYS,
            "n_total_finite_limits": N_MASSES * N_TOYS,
            "seed": SEED,
            "seed_sequence_index_rule": "mass_MeV - 19",
            "shared_toy_semantics": (
                "same toy index joins independently drawn active-dataset spectra "
                "within one fixed mass; not a correlated full-mass scan"
            ),
            "fixed_gp_states": True,
            "gp_refit_per_toy": False,
            "inner_cls": "asymptotic tilde_q_mu, alpha=0.1",
            "combined_mode": "count_scale",
            "bands_are_coverage_calibrated": False,
        },
        "combined_empirical_limit_tail_diagnostics": tails,
        "combined_local_asymptotic": combined_local,
        "individual_local_asymptotic": individual_local,
        "observed_over_median": {"minimum": ratio_min, "maximum": ratio_max},
        "minimal_visible_reinterpretation": {
            "muon_mass_GeV": M_MU_GEV,
            "dimuon_threshold_GeV": DIMUON_THRESHOLD_GEV,
            "first_corrected_grid_mass_MeV": 212,
            "pvalues_changed": False,
            "toy_quantiles_recomputed": False,
            "common_factor_applied_to_observed_and_all_quantiles": True,
        },
        "semantic_boundaries": {
            "bands": (
                "conditional fixed-GP background-only limit quantiles; "
                "descriptive rather than coverage-calibrated"
            ),
            "tail_pvalues": (
                "fixed-mass observed-limit ensemble diagnostics; not discovery p-values"
            ),
            "empirical_zero": "0/300 raw count, never an exact p=0 statement",
            "sidak": (
                "analytic resolution-spacing reference; not scan-toy calibrated"
            ),
            "individual_limits": (
                "standalone fixed-state fits; not components of the combined likelihood"
            ),
            "post_selection": (
                "the accepted 2016 k=12 ceiling followed the v4 observed "
                "saturation diagnostic; p-values remain conditional diagnostics"
            ),
        },
        "scope_exclusions": {
            "individual_expected_bands": False,
            "individual_limit_tail_ensembles": False,
            "global_max_q0_toys": False,
            "coverage_calibration": False,
        },
        "artifacts": {
            "reviewed_combined": repo_path(REVIEWED_COMBINED),
            "gp_state_closure": repo_path(GP_STATE_CLOSURE),
            "combined_sidak": repo_path(COMBINED_SIDAK),
            "reviewed_individual": repo_path(REVIEWED_INDIVIDUAL),
            "individual_sidak": repo_path(INDIVIDUAL_SIDAK),
            "summary_csv": repo_path(SUMMARY_CSV),
            "summary_json": repo_path(SUMMARY_JSON),
            "validation_json": repo_path(VALIDATION_JSON),
            "plot_manifest": repo_path(PLOT_MANIFEST),
            "readme": repo_path(POSTPROCESS_README),
        },
        "figures": list(figures),
    }
    return summary, pd.DataFrame(rows)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_readme(summary: Mapping[str, Any]) -> str:
    combined = summary["combined_local_asymptotic"]
    individual = summary["individual_local_asymptotic"]
    lines = [
        "# HPS-GPR v4.2 postprocessing",
        "",
        "This directory contains the publication postprocessing for the accepted "
        "v4.2 combined 2015 100% + 2016 100% + 2021 10% search.",
        "",
        "## Scope",
        "",
        "- The combined search uses exactly 300 finite shared conditional "
        "fixed-GP background-only limit pseudoexperiments at each of 232 masses.",
        "- Expected-limit bands are produced only for the combined search.",
        "- The 2015, 2016, and 2021 10% curves are standalone observed "
        "fixed-state fits. No individual expected bands or individual "
        "limit-tail ensembles were produced.",
        "- Inner 90% CL limits use asymptotic tilde-q_mu CLs with count_scale "
        "combination.",
        "",
        "## Statistical interpretation",
        "",
        "The central 68% and 95% bands are descriptive quantiles of a "
        "conditional fixed-GP limit ensemble. They are not coverage-calibrated "
        "confidence intervals. A shared toy index combines independently drawn "
        "active-dataset spectra at one mass; it is not a coherent correlated "
        "full-mass scan.",
        "",
        "The p_strong, p_weak, and p_two curves are fixed-mass observed-limit "
        "diagnostics. A raw 0/300 count is below one-count resolution and is "
        "not exact p=0. The local p0 curves are asymptotic discovery diagnostics. "
        "The Sidak curves are analytic resolution-spacing references and are "
        "not scan-toy calibrations.",
        "",
        "The 2016 length-scale upper factor of 12 was accepted after the v4 "
        "observed saturation diagnostic. Consequently these p-values remain "
        "conditional, post-selection diagnostics rather than a pre-specified "
        "discovery claim.",
        "",
        "The minimal-visible reinterpretation multiplies the observed limit and "
        "every combined toy-limit quantile by the same visible-width factor "
        "above the dimuon threshold. It does not alter yields, p-values, or the "
        "observed/median ratio.",
        "",
        "## Numerical minima",
        "",
        (
            f"- Combined local minimum: p0={combined['value']:.12g}, "
            f"Z={combined['Z_local_asymptotic']:.6g}, "
            f"mass={combined['masses_MeV']} MeV."
        ),
    ]
    for dataset in SEARCH_RANGES_GEV:
        record = individual[dataset]
        display_dataset = "2021 10%" if dataset == "2021" else dataset
        lines.append(
            f"- {display_dataset} standalone local minimum: "
            f"p0={record['value']:.12g}, "
            f"Z={record['Z_local_asymptotic']:.6g}, "
            f"mass={record['masses_MeV']} MeV."
        )
    lines.extend(
        [
            "",
            "Machine-readable validation, summaries, provenance hashes, "
            "reviewed tables, and the figure manifest are under `derived/`. "
            "Publication PDFs and 300 dpi PNGs are under `note_figures/`.",
            "",
            "## Reproduction",
            "",
            "Run these commands from the repository root. The production "
            "runner requires the explicit confirmation flag and refuses cards, "
            "reviewed ledgers, toy counts, or closure reports outside the "
            "declared v4.2 state.",
            "",
            "```bash",
            "python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_"
            "300toy_20260805/build_v4p2_individual_ledger.py",
            "",
            "python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_"
            "300toy_20260805/benchmark_cached_profile_closure.py \\",
            "  --json-out study_results/v4p2_combined_2015full_2016full_2021_"
            "10pct_300toy_20260805/derived/cached_profile_closure_v4p2.json",
            "",
            "python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_"
            "300toy_20260805/run_combined_bands_cached_fixed_reviewed.py \\",
            "  --closure-report study_results/v4p2_combined_2015full_2016full_"
            "2021_10pct_300toy_20260805/derived/"
            "cached_profile_closure_v4p2.json \\",
            "  --confirm-production",
            "",
            "python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_"
            "300toy_20260805/postprocess_v4p2.py",
            "```",
            "",
            "The reviewed production CSV has SHA-256 "
            "`b90768ab361928c63f57b3981d424fd36506893da2447e40824acdf3d20081c2`; "
            "the v4.2 configuration has SHA-256 "
            "`5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055`.",
            "The authoritative pass/fail gate is "
            "`derived/postprocessing_validation_v4p2.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    figures: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": now_utc(),
        "campaign": CAMPAIGN_DIR.name,
        "inputs": [
            {
                "path": repo_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "derived_artifacts": [
            {
                "path": repo_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
        "figures": list(figures),
        "semantic_boundaries": [
            "Combined fixed-GP toy-limit bands are descriptive, not coverage calibrated.",
            "Empirical 0/300 is below one-count resolution, not exact p=0.",
            "Limit-tail diagnostics are separate from local asymptotic discovery p0.",
            "Sidak curves are analytic and are not scan-toy calibrated.",
            "Individual observed curves are standalone fits and have no bands.",
            "The accepted 2016 k=12 result is conditional on an observed diagnostic.",
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=Path, default=DEFAULT_BANDS)
    parser.add_argument("--bands-provenance", type=Path, default=DEFAULT_BANDS_PROVENANCE)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--individual", type=Path, default=DEFAULT_INDIVIDUAL)
    parser.add_argument(
        "--individual-provenance",
        type=Path,
        default=DEFAULT_INDIVIDUAL_PROVENANCE,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bands_path = args.bands.resolve()
    bands_provenance_path = args.bands_provenance.resolve()
    closure_path = args.closure.resolve()
    states_path = args.states.resolve()
    individual_path = args.individual.resolve()
    individual_provenance_path = args.individual_provenance.resolve()
    required_paths = (
        CONFIG,
        V41_CONFIG,
        bands_path,
        bands_provenance_path,
        closure_path,
        states_path,
        individual_path,
        individual_provenance_path,
        ACCEPTED_COMPACT_STATES,
        ACCEPTED_COMBINED,
        ACCEPTED_COMBINED_PROVENANCE,
        V4_BANDS,
        V4_BANDS_PROVENANCE,
        RUNNER,
        CACHED_SOLVER,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Postprocessing was not run because required frozen inputs are missing:\n"
            + formatted
        )

    config_validation = validate_config()
    compact = validate_compact_accepted(pd.read_csv(ACCEPTED_COMPACT_STATES))
    closure_payload = load_json(closure_path)
    closure_validation = validate_closure(
        closure_payload,
        sha256(ACCEPTED_COMPACT_STATES),
        sha256(CONFIG),
    )
    provenance_validation = validate_band_provenance(
        load_json(bands_provenance_path),
        bands_path,
        bands_provenance_path,
        closure_path,
        compact,
    )
    states, state_validation = validate_enriched_states(
        pd.read_csv(states_path),
        compact,
    )
    bands = validate_bands(pd.read_csv(bands_path))
    bands, gp_closure = validate_gp_state_metadata(bands, states)
    accepted_validation = validate_accepted_combined(bands)
    unaffected_validation = validate_unaffected_v4(bands)
    individual_provenance_validation = validate_individual_provenance(
        load_json(individual_provenance_path),
        individual_path,
        states_path,
    )
    individual, individual_validation = validate_individual(
        pd.read_csv(individual_path),
        states,
        bands,
    )

    reviewed_combined = build_reviewed_combined(bands, bands_path, states_path)
    reviewed_individual = build_reviewed_individual(individual, individual_path)
    combined_sidak, combined_neff = build_combined_sidak(reviewed_combined)
    individual_sidak, individual_neff = build_individual_sidak(reviewed_individual)

    atomic_write_csv(reviewed_combined, REVIEWED_COMBINED)
    atomic_write_csv(gp_closure, GP_STATE_CLOSURE)
    atomic_write_csv(combined_sidak, COMBINED_SIDAK)
    atomic_write_csv(reviewed_individual, REVIEWED_INDIVIDUAL)
    atomic_write_csv(individual_sidak, INDIVIDUAL_SIDAK)

    setup_style()
    figures = [
        plot_combined_bands(reviewed_combined),
        plot_empirical_tails(reviewed_combined),
        plot_combined_local_p0(combined_sidak, combined_neff),
        plot_individual_limits(reviewed_individual, reviewed_combined),
        plot_individual_local_p0(individual_sidak, individual_neff),
    ]
    input_validation = {
        "config": config_validation,
        "closure": closure_validation,
        "band_provenance": provenance_validation,
        "enriched_states": state_validation,
        "accepted_v4p1_combined": accepted_validation,
        "unaffected_v4_bands": unaffected_validation,
        "individual_provenance": individual_provenance_validation,
        "individual_observed": individual_validation,
    }
    summary, summary_frame = build_summary(
        reviewed_combined,
        combined_sidak,
        combined_neff,
        reviewed_individual,
        individual_sidak,
        individual_neff,
        input_validation,
        figures,
    )
    atomic_write_csv(summary_frame, SUMMARY_CSV)
    atomic_write_json(summary, SUMMARY_JSON)
    atomic_write_text(build_readme(summary), POSTPROCESS_README)

    validation = {
        "schema_version": 1,
        "created_utc": now_utc(),
        "campaign": CAMPAIGN_DIR.name,
        "status": "PASS",
        "checks": input_validation,
        "combined_grid_rows": int(len(reviewed_combined)),
        "combined_finite_toy_limits": int(
            reviewed_combined["n_toys_finite"].sum()
        ),
        "gp_state_closure_rows": int(len(gp_closure)),
        "individual_rows": int(len(reviewed_individual)),
        "combined_quantiles_ordered": True,
        "tail_count_identities": True,
        "combined_only_bands": True,
        "figure_47_footer_removed": True,
    }
    atomic_write_json(validation, VALIDATION_JSON)

    input_paths = (
        CONFIG,
        V41_CONFIG,
        bands_path,
        bands_provenance_path,
        closure_path,
        states_path,
        individual_path,
        individual_provenance_path,
        ACCEPTED_COMPACT_STATES,
        ACCEPTED_COMBINED,
        ACCEPTED_COMBINED_PROVENANCE,
        V4_BANDS,
        V4_BANDS_PROVENANCE,
        RUNNER,
        CACHED_SOLVER,
    )
    output_paths = (
        REVIEWED_COMBINED,
        GP_STATE_CLOSURE,
        COMBINED_SIDAK,
        REVIEWED_INDIVIDUAL,
        INDIVIDUAL_SIDAK,
        SUMMARY_CSV,
        SUMMARY_JSON,
        VALIDATION_JSON,
        POSTPROCESS_README,
    )
    atomic_write_json(
        build_manifest(input_paths, output_paths, figures),
        PLOT_MANIFEST,
    )

    print(
        f"PASS: validated {N_MASSES} masses x {N_TOYS} finite combined toys, "
        f"{EXPECTED_STATE_COUNT} fixed GP states, and "
        f"{EXPECTED_STATE_COUNT} standalone observed limits."
    )
    print(f"Validation: {VALIDATION_JSON}")
    print(f"Summary: {SUMMARY_JSON}")
    print(f"Figures: {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
