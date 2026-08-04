#!/usr/bin/env python3
"""Validate, reinterpret, compare, and plot the reviewed v4 combined bands.

This is a postprocessor only.  It does not fit a GP, repair a mass point,
regenerate a pseudoexperiment, or edit the analysis note.  Its two v4 inputs
are the reviewed 415-state observed ledger and the fixed-state 300-toy
combined-band table.

The three p-value families remain separate:

* ``p_strong``, ``p_weak``, and ``p_two`` are fixed-mass empirical
  pseudoexperiment diagnostics for the observed upper limit;
* ``p0_analytic`` is the local asymptotic discovery statistic; and
* the resolution-spacing Sidak curve is an analytic look-elsewhere reference,
  not a toy calibration.
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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4-postprocess-mpl")

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

DEFAULT_OBSERVED_STATES = DERIVED / "observed_gp_states_reviewed.csv"
DEFAULT_BANDS = (
    CAMPAIGN_DIR
    / "combined_bands_300toy_cached"
    / "ul_bands_combined_all.csv"
)
DEFAULT_BASELINE = (
    REPO
    / "study_results"
    / "finalist_k15_2021_10pct_combined100toy_20260803"
    / "derived"
    / "combined_bands100_reviewed.csv"
)
V4_CONFIG = (
    REPO
    / "study_configs"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "config_obsUL90_combined_wide_support_v4_observed_only.yaml"
)
BASELINE_CONFIG = (
    REPO
    / "study_configs"
    / "finalist_k15_2021_10pct_combined100toy_20260803"
    / "config_obsUL90_combined_2015full_2016full_2021_10pct_k15_bands100.yaml"
)

REVIEWED_BANDS = DERIVED / "combined_bands300_reviewed.csv"
GP_STATE_CLOSURE = DERIVED / "combined_bands300_gp_state_closure.csv"
SIDAK_TABLE = DERIVED / "combined_bands300_sidak_reference.csv"
WIDE_NARROW_TABLE = DERIVED / "wide_vs_narrow_observed_limit.csv"
SUMMARY_JSON = DERIVED / "combined_bands300_summary.json"
SUMMARY_CSV = DERIVED / "combined_bands300_summary.csv"
PLOT_MANIFEST = DERIVED / "combined_bands300_plot_manifest.json"

MASS_LOW_MEV = 19
MASS_HIGH_MEV = 250
N_MASSES = MASS_HIGH_MEV - MASS_LOW_MEV + 1
N_TOYS = 300
SEED = 24_680
INDEPENDENCE_WIDTH_SIGMA = 2.25
M_MU_GEV = 0.1056583745
DIMUON_THRESHOLD_GEV = 2.0 * M_MU_GEV
LML_CLOSURE_ATOL = 5.0e-5

SEARCH_RANGES_GEV = {
    "2015": (0.019, 0.090),
    "2016": (0.039, 0.180),
    "2021": (0.050, 0.250),
}
WIDE_SUPPORT_RANGES_GEV = {
    "2015": (0.014, 0.135),
    "2016": (0.030, 0.210),
    "2021": (0.040, 0.300),
}
NARROW_SUPPORT_RANGES_GEV = {
    "2015": (0.019, 0.090),
    "2016": (0.039, 0.180),
    "2021": (0.040, 0.300),
}
EXPECTED_STATE_ROWS = {
    dataset: int(round(1000.0 * (high - low))) + 1
    for dataset, (low, high) in SEARCH_RANGES_GEV.items()
}
EXPECTED_STATE_COUNT = sum(EXPECTED_STATE_ROWS.values())
EXPECTED_ACTIVE_COUNTS = {
    "2015": 20,
    "2015+2016": 11,
    "2015+2016+2021": 41,
    "2016+2021": 90,
    "2021": 70,
}

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
    "density_window_fully_covered",
    "extract_success",
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
    "selected_attempt",
    "selected_source",
    "selected_source_sha256",
    "row_source",
    "review_status",
    "branch_multiplicity",
    "interpolated",
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


def expected_mass_grid() -> np.ndarray:
    return np.arange(MASS_LOW_MEV, MASS_HIGH_MEV + 1, dtype=int) / 1000.0


def active_datasets(mass_gev: float) -> List[str]:
    return [
        dataset
        for dataset, (low, high) in SEARCH_RANGES_GEV.items()
        if low - 1.0e-12 <= float(mass_gev) <= high + 1.0e-12
    ]


def expected_active_tags() -> List[str]:
    return ["+".join(active_datasets(mass)) for mass in expected_mass_grid()]


def normalize_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if bool(series.isna().any()):
            raise RuntimeError(f"{label} contains missing boolean values")
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


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def require_exact_grid(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if len(frame) != N_MASSES:
        raise RuntimeError(f"{label} has {len(frame)} rows; expected {N_MASSES}")
    if bool(frame["mass_GeV"].duplicated().any()):
        raise RuntimeError(f"{label} contains duplicate masses")
    out = frame.sort_values("mass_GeV").reset_index(drop=True).copy()
    actual = out["mass_GeV"].to_numpy(float)
    expected = expected_mass_grid()
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"{label} does not cover the exact 19--250 MeV grid")
    return out


def require_finite_positive(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    for column in columns:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or not bool((values > 0.0).all()):
            raise RuntimeError(f"{label}.{column} is not finite and positive")


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config is not a mapping: {path}")
    return payload


def validate_matched_configs() -> Dict[str, Any]:
    """Prove that the wide/narrow comparison changes support, not inputs."""

    v4 = load_yaml(V4_CONFIG)
    baseline = load_yaml(BASELINE_CONFIG)
    allowed_differences = {
        "combined_bands_n_toys",
        "data_range_2015",
        "data_range_2016",
        "do_combined_bands",
        "make_ul_bands",
        "output_dir",
        "scan_n_workers",
        "scan_require_two_sidebands",
        "ul_bands_n_workers",
        "ul_bands_parallel_backend",
    }
    unexpected = {}
    for key in sorted(set(v4).union(baseline)):
        if key in allowed_differences:
            continue
        if v4.get(key) != baseline.get(key):
            unexpected[key] = {
                "wide": v4.get(key),
                "narrow": baseline.get(key),
            }
    if unexpected:
        raise RuntimeError(
            "Wide/narrow configs differ outside the declared support/runtime "
            f"fields: {unexpected}"
        )
    for dataset in SEARCH_RANGES_GEV:
        wide = tuple(float(value) for value in v4[f"data_range_{dataset}"])
        narrow = tuple(float(value) for value in baseline[f"data_range_{dataset}"])
        if wide != WIDE_SUPPORT_RANGES_GEV[dataset]:
            raise RuntimeError(
                f"Unexpected v4 {dataset} support: {wide}"
            )
        if narrow != NARROW_SUPPORT_RANGES_GEV[dataset]:
            raise RuntimeError(
                f"Unexpected finalist {dataset} support: {narrow}"
            )
    return {
        "wide_config": repo_path(V4_CONFIG),
        "wide_config_sha256": sha256(V4_CONFIG),
        "narrow_config": repo_path(BASELINE_CONFIG),
        "narrow_config_sha256": sha256(BASELINE_CONFIG),
        "matched_except_support_and_runtime_controls": True,
        "declared_difference_keys": sorted(allowed_differences),
    }


def validate_observed_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete, reproduced, non-interpolated 415-state ledger."""

    require_columns(frame, STATE_REQUIRED_COLUMNS, "observed reviewed states")
    if len(frame) != EXPECTED_STATE_COUNT:
        raise RuntimeError(
            f"Observed reviewed states has {len(frame)} rows; "
            f"expected {EXPECTED_STATE_COUNT}"
        )
    out = frame.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    if set(out["dataset"]) != set(SEARCH_RANGES_GEV):
        raise RuntimeError(
            f"Unexpected reviewed datasets: {sorted(set(out['dataset']))}"
        )
    if bool(out.duplicated(["dataset", "mass_GeV"]).any()):
        raise RuntimeError("Observed reviewed states contains duplicate states")

    for column in (
        "density_window_fully_covered",
        "extract_success",
        "interpolated",
    ):
        out[column] = normalize_boolean(
            out[column],
            f"observed reviewed states.{column}",
        )
    if not bool(out["density_window_fully_covered"].all()):
        raise RuntimeError("A reviewed density window is not fully covered")
    if not bool(out["extract_success"].all()):
        raise RuntimeError("A reviewed observed extraction failed")
    if bool(out["interpolated"].any()):
        raise RuntimeError("Observed reviewed states contains interpolation")
    if set(out["review_status"].astype(str)) != {
        "resolved_reproduced_max_lml"
    }:
        raise RuntimeError("Not every observed GP state is a reproduced maximum-LML state")
    if not bool((out["branch_multiplicity"].to_numpy(int) >= 2).all()):
        raise RuntimeError("A reviewed maximum-LML state lacks an independent repeat")

    for column in ("selected_attempt", "selected_source", "row_source"):
        values = out[column].astype("string").str.strip()
        if bool(values.isna().any()) or bool((values.str.len() == 0).any()):
            raise RuntimeError(f"Observed reviewed states has empty {column}")
    source_hashes = out["selected_source_sha256"].astype(str).str.lower()
    valid_source_hash = source_hashes.str.fullmatch(r"[0-9a-f]{64}", na=False)
    if not bool(valid_source_hash.all()):
        raise RuntimeError("Observed reviewed states has an invalid source SHA-256")
    for source, rows in out.groupby("selected_source", sort=False):
        source_path = REPO / str(source)
        if not source_path.is_file():
            raise RuntimeError(f"Reviewed-state source does not exist: {source}")
        expected_hashes = set(
            rows["selected_source_sha256"].astype(str).str.lower()
        )
        if expected_hashes != {sha256(source_path)}:
            raise RuntimeError(
                f"Reviewed-state source hash does not close for {source}"
            )
        attempts = set(rows["selected_attempt"].astype(str))
        if not all(attempt in str(source) for attempt in attempts):
            raise RuntimeError(
                f"Reviewed-state attempt/source provenance disagrees for {source}"
            )
        if not bool(
            rows["row_source"]
            .astype(str)
            .str.contains(str(source), regex=False)
            .all()
        ):
            raise RuntimeError(
                f"Reviewed-state row_source does not cite selected_source {source}"
            )

    numeric_columns = (
        "mass_GeV",
        "mass_MeV",
        "sigma_val",
        "integral_density",
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
    )
    numeric = out[list(numeric_columns)].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Observed reviewed states contains non-finite metadata")
    if not bool((out["sigma_val"].to_numpy(float) > 0.0).all()):
        raise RuntimeError("Observed reviewed states contains non-positive resolution")
    if not bool((out["integral_density"].to_numpy(float) > 0.0).all()):
        raise RuntimeError(
            "Observed reviewed states contains non-positive integral density"
        )
    if not bool((out["ls_lo"].to_numpy(float) > 0.0).all()):
        raise RuntimeError("Observed reviewed states contains non-positive kernel lower bound")
    if not bool(
        (
            out["ls_opt"].to_numpy(float)
            >= out["ls_lo"].to_numpy(float) * (1.0 - 1.0e-10)
        ).all()
    ):
        raise RuntimeError("A reviewed length scale lies below its configured bound")
    if not bool(
        (
            out["ls_opt"].to_numpy(float)
            <= out["ls_hi"].to_numpy(float) * (1.0 + 1.0e-10)
        ).all()
    ):
        raise RuntimeError("A reviewed length scale lies above its configured bound")
    if not bool(
        (
            out["n_train_low"].to_numpy(int)
            + out["n_train_high"].to_numpy(int)
            == out["n_train"].to_numpy(int)
        ).all()
    ):
        raise RuntimeError("Reviewed training-bin side counts do not close")
    if not bool((out["n_train_low"].to_numpy(int) > 0).all()):
        raise RuntimeError("A reviewed state has no low-side training bins")
    if not bool((out["n_train_high"].to_numpy(int) > 0).all()):
        raise RuntimeError("A reviewed state has no high-side training bins")

    for dataset, (low, high) in SEARCH_RANGES_GEV.items():
        rows = out[out["dataset"] == dataset].sort_values("mass_GeV")
        expected = np.round(np.arange(low, high + 0.0005, 0.001), 3)
        actual = rows["mass_GeV"].to_numpy(float)
        if len(rows) != EXPECTED_STATE_ROWS[dataset] or not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{dataset} reviewed-state grid is incomplete")
        expected_support = WIDE_SUPPORT_RANGES_GEV[dataset]
        support_lo = rows["train_domain_lo"].to_numpy(float)
        support_hi = rows["train_domain_hi"].to_numpy(float)
        if not np.allclose(
            support_lo,
            expected_support[0],
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.allclose(
            support_hi,
            expected_support[1],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"{dataset} reviewed fit support differs from "
                f"{expected_support}: "
                f"low=[{support_lo.min()}, {support_lo.max()}], "
                f"high=[{support_hi.min()}, {support_hi.max()}]"
            )
        if not np.allclose(
            rows["mass_MeV"].to_numpy(float),
            1000.0 * actual,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError(f"{dataset} mass_GeV/mass_MeV columns disagree")

    if "cls_statistic" in out.columns and set(
        out["cls_statistic"].astype(str)
    ) != {"tilde_q_mu"}:
        raise RuntimeError("Observed reviewed states uses another CLs statistic")
    if "cls_calibration" in out.columns and set(
        out["cls_calibration"].astype(str)
    ) != {"asymptotic"}:
        raise RuntimeError("Observed reviewed states is not uniformly asymptotic")
    if "visibility" in out.columns and set(out["visibility"].astype(str)) != {
        "observed"
    }:
        raise RuntimeError("Observed reviewed states contains a blinded dataset")
    if "density_nsigma" in out.columns and set(
        out["density_nsigma"].astype(float)
    ) != {1.64}:
        raise RuntimeError("Observed density window is not uniformly 1.64 sigma")

    dataset_order = {dataset: index for index, dataset in enumerate(SEARCH_RANGES_GEV)}
    out["_dataset_order"] = out["dataset"].map(dataset_order)
    return (
        out.sort_values(["_dataset_order", "mass_GeV"])
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )


def parse_json_mapping(value: Any, label: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} is not a JSON mapping")
    return parsed


def parse_json_metadata(value: Any, label: str) -> Dict[str, Dict[str, Any]]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(parsed, list):
        raise RuntimeError(f"{label} is not a JSON list")
    by_key: Dict[str, Dict[str, Any]] = {}
    for entry in parsed:
        if not isinstance(entry, dict) or "key" not in entry:
            raise RuntimeError(f"{label} has a malformed entry")
        key = str(entry["key"])
        if key in by_key:
            raise RuntimeError(f"{label} has duplicate metadata for {key}")
        by_key[key] = entry
    return by_key


def validate_tail_row(row: Any) -> None:
    mass = float(row.mass_GeV)
    n_finite = int(row.n_toys_finite)
    counts = {
        "strong": int(row.tail_count_strong_le_observed),
        "weak": int(row.tail_count_weak_ge_observed),
        "equal": int(row.tail_count_equal_observed),
        "two_min": int(row.tail_count_two_sided_min),
    }
    if any(value < 0 or value > n_finite for value in counts.values()):
        raise RuntimeError(f"Tail count outside 0--{n_finite} at {mass:.3f} GeV")
    if counts["strong"] + counts["weak"] - counts["equal"] != n_finite:
        raise RuntimeError(f"Tail-count partition fails at {mass:.3f} GeV")
    if counts["two_min"] != min(counts["strong"], counts["weak"]):
        raise RuntimeError(f"Two-sided raw count fails at {mass:.3f} GeV")
    expected_strong = counts["strong"] / n_finite
    expected_weak = counts["weak"] / n_finite
    expected_two = min(1.0, 2.0 * min(expected_strong, expected_weak))
    for label, actual, expected in (
        ("p_strong", float(row.p_strong), expected_strong),
        ("p_weak", float(row.p_weak), expected_weak),
        ("p_two", float(row.p_two), expected_two),
    ):
        if not np.isclose(
            actual,
            expected,
            rtol=1.0e-14,
            atol=1.0e-15,
        ):
            raise RuntimeError(
                f"{label}/raw-count mismatch at {mass:.3f} GeV: "
                f"{actual!r} != {expected!r}"
            )
    if not np.isclose(
        float(row.empirical_tail_resolution),
        1.0 / n_finite,
        rtol=1.0e-14,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            f"Empirical tail resolution mismatch at {mass:.3f} GeV"
        )


def validate_bands(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, BAND_REQUIRED_COLUMNS, "combined 300-toy bands")
    out = require_exact_grid(frame, "combined 300-toy bands")
    expected_tags = expected_active_tags()
    actual_tags = out["dataset_set"].astype(str).tolist()
    if actual_tags != expected_tags:
        raise RuntimeError("Band active sets disagree with the exact mass grid")
    active_counts = out["dataset_set"].astype(str).value_counts().to_dict()
    if active_counts != EXPECTED_ACTIVE_COUNTS:
        raise RuntimeError(
            f"Band active-set counts differ: {active_counts} "
            f"!= {EXPECTED_ACTIVE_COUNTS}"
        )

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
        "combined 300-toy bands",
    )
    quantiles = out[list(RAW_QUANTILE_COLUMNS)].to_numpy(float)
    if not bool((np.diff(quantiles, axis=1) >= -1.0e-18).all()):
        raise RuntimeError("Expected-limit quantiles are unordered")
    for primary, alias in ALIAS_PAIRS:
        if not np.array_equal(
            out[primary].to_numpy(float),
            out[alias].to_numpy(float),
        ):
            raise RuntimeError(f"Band aliases disagree: {primary} != {alias}")
    for column in ("p0_analytic", "Z_analytic", "p_strong", "p_weak", "p_two"):
        values = out[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Band column {column} is non-finite")
    for column in ("p0_analytic", "p_strong", "p_weak", "p_two"):
        values = out[column].to_numpy(float)
        if not bool(((values >= 0.0) & (values <= 1.0)).all()):
            raise RuntimeError(f"Band column {column} lies outside [0,1]")

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
        "limit_solver": {"campaign_local_deterministic_profile_cache_v1"},
        "profile_cache_limit_calls": {N_TOYS + 1},
    }
    for column, expected in exact_sets.items():
        values = set(out[column])
        if values != expected:
            raise RuntimeError(f"{column} differs: {values} != {expected}")
    out["bands_refit_gp_on_toy"] = normalize_boolean(
        out["bands_refit_gp_on_toy"],
        "combined 300-toy bands.bands_refit_gp_on_toy",
    )
    out["bands_refit_optimize"] = normalize_boolean(
        out["bands_refit_optimize"],
        "combined 300-toy bands.bands_refit_optimize",
    )
    if bool(out["bands_refit_gp_on_toy"].any()):
        raise RuntimeError("A band row refit its GP on pseudoexperiments")
    if bool(out["bands_refit_optimize"].any()):
        raise RuntimeError("A band row reoptimized its reviewed observed GP")
    if out["bands_seed_sequence_index"].astype(int).tolist() != list(
        range(N_MASSES)
    ):
        raise RuntimeError("Band SeedSequence indices differ from mass_MeV - 19")

    count_columns = (
        "tail_count_strong_le_observed",
        "tail_count_weak_ge_observed",
        "tail_count_equal_observed",
        "tail_count_two_sided_min",
    )
    for column in count_columns:
        values = out[column].to_numpy(float)
        if not np.isfinite(values).all() or not np.array_equal(
            values,
            np.rint(values),
        ):
            raise RuntimeError(f"{column} is not finite and integer-valued")
    for row in out.itertuples(index=False):
        validate_tail_row(row)
    return out


def validate_gp_state_metadata(
    bands: pd.DataFrame,
    states: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Close every band GP coordinate to its reviewed observed-state row."""

    state_lookup_frame = states.copy()
    state_lookup_frame["_mass_MeV_key"] = np.rint(
        state_lookup_frame["mass_MeV"].to_numpy(float)
    ).astype(int)
    if not np.allclose(
        state_lookup_frame["mass_MeV"].to_numpy(float),
        state_lookup_frame["_mass_MeV_key"].to_numpy(float),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError("Reviewed-state mass_MeV contains a non-integer grid point")
    state_lookup = state_lookup_frame.set_index(
        ["dataset", "_mass_MeV_key"],
        verify_integrity=True,
    )
    closure_rows: List[Dict[str, Any]] = []
    augmented = bands.copy()
    source_summaries: List[str] = []
    attempt_summaries: List[str] = []
    status_summaries: List[str] = []
    multiplicity_summaries: List[str] = []

    for row in bands.itertuples(index=False):
        mass = float(row.mass_GeV)
        mass_mev = int(round(1000.0 * mass))
        active = str(row.dataset_set).split("+")
        lml = parse_json_mapping(
            row.gp_lml_by_dataset,
            f"gp_lml_by_dataset at {mass:.3f} GeV",
        )
        ls_opt = parse_json_mapping(
            row.gp_ls_opt_by_dataset,
            f"gp_ls_opt_by_dataset at {mass:.3f} GeV",
        )
        const_opt = parse_json_mapping(
            row.gp_const_opt_by_dataset,
            f"gp_const_opt_by_dataset at {mass:.3f} GeV",
        )
        state_hash = parse_json_mapping(
            row.gp_state_sha256_by_dataset,
            f"gp_state_sha256_by_dataset at {mass:.3f} GeV",
        )
        meta = parse_json_metadata(row.meta, f"meta at {mass:.3f} GeV")
        for label, mapping in (
            ("lml", lml),
            ("ls_opt", ls_opt),
            ("const_opt", const_opt),
            ("state_hash", state_hash),
            ("meta", meta),
        ):
            if set(mapping) != set(active):
                raise RuntimeError(
                    f"{label} active-set mismatch at {mass:.3f} GeV: "
                    f"{sorted(mapping)} != {sorted(active)}"
                )

        sources: List[str] = []
        attempts: List[str] = []
        statuses: List[str] = []
        multiplicities: List[int] = []
        for dataset in active:
            try:
                reviewed = state_lookup.loc[(dataset, mass_mev)]
            except KeyError as error:
                raise RuntimeError(
                    f"Missing reviewed {dataset} state at {mass:.3f} GeV"
                ) from error
            band_lml = float(lml[dataset])
            band_ls = float(ls_opt[dataset])
            band_const = float(const_opt[dataset])
            reviewed_lml = float(reviewed["lml"])
            reviewed_ls = float(reviewed["ls_opt"])
            reviewed_const = float(reviewed["const_opt"])
            reviewed_sigma = float(reviewed["sigma_val"])
            reviewed_density = float(reviewed["integral_density"])
            lml_delta = band_lml - reviewed_lml
            coordinate_closure = bool(
                np.isclose(
                    band_ls,
                    reviewed_ls,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
                and np.isclose(
                    band_const,
                    reviewed_const,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            )
            lml_closure = bool(abs(lml_delta) <= LML_CLOSURE_ATOL)
            hash_value = str(state_hash[dataset])
            if len(hash_value) != 64 or any(
                character not in "0123456789abcdef"
                for character in hash_value.lower()
            ):
                raise RuntimeError(
                    f"Invalid GP-state hash for {dataset} at {mass:.3f} GeV"
                )
            metadata = meta[dataset]
            metadata_closure = bool(
                str(metadata.get("state_sha256")) == hash_value
                and np.isclose(
                    float(metadata.get("lml")),
                    band_lml,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.isclose(
                    float(metadata.get("ls_opt")),
                    band_ls,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(metadata.get("const_opt")),
                    band_const,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(metadata.get("reviewed_lml")),
                    reviewed_lml,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.isclose(
                    float(metadata.get("lml_delta")),
                    lml_delta,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.isclose(
                    float(metadata.get("sigma")),
                    reviewed_sigma,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(metadata.get("dens")),
                    reviewed_density,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            )
            if not coordinate_closure or not lml_closure or not metadata_closure:
                raise RuntimeError(
                    f"Reviewed fixed-state metadata fails for {dataset} at "
                    f"{mass:.3f} GeV"
                )
            sources.append(str(reviewed["row_source"]))
            attempts.append(str(reviewed["selected_attempt"]))
            statuses.append(str(reviewed["review_status"]))
            multiplicities.append(int(reviewed["branch_multiplicity"]))
            closure_rows.append(
                {
                    "mass_GeV": mass,
                    "mass_MeV": int(round(1000.0 * mass)),
                    "dataset_set": str(row.dataset_set),
                    "dataset": dataset,
                    "band_lml": band_lml,
                    "reviewed_lml": reviewed_lml,
                    "delta_lml": lml_delta,
                    "band_ls_opt": band_ls,
                    "reviewed_ls_opt": reviewed_ls,
                    "delta_ls_opt": band_ls - reviewed_ls,
                    "band_const_opt": band_const,
                    "reviewed_const_opt": reviewed_const,
                    "delta_const_opt": band_const - reviewed_const,
                    "band_sigma_val": float(metadata["sigma"]),
                    "reviewed_sigma_val": reviewed_sigma,
                    "delta_sigma_val": (
                        float(metadata["sigma"]) - reviewed_sigma
                    ),
                    "band_integral_density": float(metadata["dens"]),
                    "reviewed_integral_density": reviewed_density,
                    "delta_integral_density": (
                        float(metadata["dens"]) - reviewed_density
                    ),
                    "band_state_sha256": hash_value,
                    "coordinate_closure_pass": coordinate_closure,
                    "lml_closure_pass": lml_closure,
                    "metadata_json_closure_pass": metadata_closure,
                    "fixed_reviewed_state_closure_pass": True,
                    "reviewed_selected_attempt": str(
                        reviewed["selected_attempt"]
                    ),
                    "reviewed_row_source": str(reviewed["row_source"]),
                    "review_status": str(reviewed["review_status"]),
                    "branch_multiplicity": int(
                        reviewed["branch_multiplicity"]
                    ),
                    "interpolated": False,
                }
            )
        source_summaries.append(json.dumps(dict(zip(active, sources)), sort_keys=True))
        attempt_summaries.append(
            json.dumps(dict(zip(active, attempts)), sort_keys=True)
        )
        status_summaries.append(
            json.dumps(dict(zip(active, statuses)), sort_keys=True)
        )
        multiplicity_summaries.append(
            json.dumps(dict(zip(active, multiplicities)), sort_keys=True)
        )

    closure = pd.DataFrame(closure_rows)
    if len(closure) != EXPECTED_STATE_COUNT:
        raise RuntimeError(
            f"GP-state closure has {len(closure)} rows; "
            f"expected {EXPECTED_STATE_COUNT}"
        )
    if not bool(closure["fixed_reviewed_state_closure_pass"].all()):
        raise RuntimeError("A band GP state does not close to the reviewed state")
    augmented["observed_state_sources_by_dataset"] = source_summaries
    augmented["observed_state_attempts_by_dataset"] = attempt_summaries
    augmented["observed_state_status_by_dataset"] = status_summaries
    augmented["observed_state_branch_multiplicity_by_dataset"] = (
        multiplicity_summaries
    )
    augmented["fixed_reviewed_state_metadata_validated"] = True
    return augmented, closure


def dimuon_factor(mass_gev: np.ndarray) -> np.ndarray:
    masses = np.asarray(mass_gev, dtype=float)
    factor = np.ones_like(masses)
    above = masses > DIMUON_THRESHOLD_GEV
    if np.any(above):
        phase_space = np.sqrt(
            1.0 - 4.0 * M_MU_GEV**2 / masses[above] ** 2
        )
        phase_space *= 1.0 + 2.0 * M_MU_GEV**2 / masses[above] ** 2
        factor[above] = 1.0 + phase_space
    return factor


def empirical_tail_label(
    count: int,
    probability: float,
    *,
    two_sided: bool = False,
) -> str:
    if count == 0:
        if two_sided:
            return (
                "2*0/300; no pseudoexperiment in the smaller empirical tail; "
                "not an exact zero probability"
            )
        return (
            "0/300; no pseudoexperiment in this empirical tail; "
            "quote below one-count resolution 1/300, not exact zero"
        )
    if two_sided:
        return f"2*{count}/300, bounded at 1 = {probability:.12g}"
    return f"{count}/300 = {probability:.12g}"


def build_reviewed_table(
    bands: pd.DataFrame,
    bands_path: Path,
    states_path: Path,
) -> pd.DataFrame:
    out = bands.copy()
    out["mass_MeV"] = 1000.0 * out["mass_GeV"].to_numpy(float)
    factor = dimuon_factor(out["mass_GeV"].to_numpy(float))
    out["dimuon_threshold_GeV"] = DIMUON_THRESHOLD_GEV
    out["dimuon_threshold_MeV"] = 1000.0 * DIMUON_THRESHOLD_GEV
    out["N_eff_BR"] = factor
    out["BR_ee_minimal"] = 1.0 / factor
    out["dimuon_correction_applied"] = (
        out["mass_GeV"].to_numpy(float) > DIMUON_THRESHOLD_GEV
    )
    out["first_dimuon_corrected_grid_point"] = False
    first_corrected = float(
        out.loc[out["dimuon_correction_applied"], "mass_GeV"].min()
    )
    out.loc[
        np.isclose(
            out["mass_GeV"].to_numpy(float),
            first_corrected,
            rtol=0.0,
            atol=1.0e-12,
        ),
        "first_dimuon_corrected_grid_point",
    ] = True
    for column in COUPLING_COLUMNS:
        out[f"{column}_ee_channel"] = out[column].to_numpy(float)
        out[f"{column}_minimal_visible"] = (
            out[column].to_numpy(float) * factor
        )

    out["observed_over_median_ee_channel"] = (
        out["eps2_obs_ee_channel"].to_numpy(float)
        / out["eps2_med_ee_channel"].to_numpy(float)
    )
    out["observed_over_median_minimal_visible"] = (
        out["eps2_obs_minimal_visible"].to_numpy(float)
        / out["eps2_med_minimal_visible"].to_numpy(float)
    )
    if not np.allclose(
        out["observed_over_median_ee_channel"].to_numpy(float),
        out["observed_over_median_minimal_visible"].to_numpy(float),
        rtol=5.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Common dimuon factor did not cancel in observed/median")

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
        empirical_tail_label(
            int(count),
            float(probability),
            two_sided=True,
        )
        for count, probability in zip(
            out["tail_count_two_sided_min"],
            out["p_two"],
        )
    ]
    out["pvalues_changed_by_dimuon"] = False
    out["toy_quantiles_recomputed"] = False
    out["toy_ensemble_changed"] = False
    out["fit_result_changed_by_dimuon"] = False
    out["coverage_calibration"] = False
    out["coverage_claimed"] = False
    out["tail_pvalue_family"] = (
        "fixed_mass empirical background-only limit-ensemble diagnostic"
    )
    out["local_p0_family"] = "local asymptotic discovery statistic"
    out["dimuon_reinterpretation"] = (
        "common minimal-visible branching factor applied to observed epsilon2 "
        "and every fixed-GP toy-limit quantile; p-values unchanged"
    )
    out["source_bands_table"] = repo_path(bands_path)
    out["source_bands_sha256"] = sha256(bands_path)
    out["source_observed_states"] = repo_path(states_path)
    out["source_observed_states_sha256"] = sha256(states_path)
    return out


def effective_trials_from_spacing(
    masses: np.ndarray,
    sigma_values: np.ndarray,
) -> float:
    masses = np.asarray(masses, dtype=float)
    sigma_values = np.asarray(sigma_values, dtype=float)
    if masses.size != sigma_values.size or masses.size < 2:
        raise RuntimeError("Mass and resolution arrays do not align")
    delta_mass = np.diff(masses)
    sigma_mid = 0.5 * (sigma_values[:-1] + sigma_values[1:])
    valid = (
        np.isfinite(delta_mass)
        & (delta_mass > 0.0)
        & np.isfinite(sigma_mid)
        & (sigma_mid > 0.0)
    )
    if not bool(valid.all()):
        raise RuntimeError("Resolution-spacing Sidak input is incomplete")
    neff = np.sum(
        delta_mass
        / (INDEPENDENCE_WIDTH_SIGMA * sigma_mid)
    )
    return float(np.clip(neff, 1.0, float(masses.size)))


def build_sidak_table(reviewed: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
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
        out["mass_GeV"].to_numpy(float),
        out["sigma_mass_res_min_GeV"].to_numpy(float),
    )
    p_local = np.clip(out["p0_analytic"].to_numpy(float), 1.0e-300, 1.0)
    p_sidak = -np.expm1(neff * np.log1p(-p_local))
    p_sidak = np.clip(p_sidak, 1.0e-300, 1.0)
    out["p0_local_asymptotic"] = p_local
    out["Z_local_asymptotic"] = out["Z_analytic"].to_numpy(float)
    out["p_sidak_resolution_spacing_analytic"] = p_sidak
    out["Z_sidak_resolution_spacing_analytic"] = norm.isf(p_sidak)
    out["is_scan_minimum"] = False
    out.loc[int(np.nanargmin(p_local)), "is_scan_minimum"] = True
    out["N_eff_resolution_spacing"] = neff
    out["independence_width_sigma"] = INDEPENDENCE_WIDTH_SIGMA
    out["global_reference_method"] = "sidak_resolution_spacing_analytic"
    out["global_reference_scope"] = "combined stitched scan 19-250 MeV"
    out["scan_toy_calibrated"] = False
    out["uses_limit_tail_pvalues"] = False
    out["interpretation"] = (
        "analytic resolution-spacing look-elsewhere reference; "
        "not toy-calibrated and separate from limit-tail pseudoexperiments"
    )
    return out, neff


def build_wide_narrow_comparison(
    reviewed: pd.DataFrame,
    baseline_path: Path,
) -> pd.DataFrame:
    baseline = require_exact_grid(
        pd.read_csv(baseline_path),
        "narrow-support finalist baseline",
    )
    require_columns(
        baseline,
        {"dataset_set", "eps2_obs"},
        "narrow-support finalist baseline",
    )
    if baseline["dataset_set"].astype(str).tolist() != expected_active_tags():
        raise RuntimeError("Narrow-support baseline active sets differ")
    require_finite_positive(
        baseline,
        ("eps2_obs",),
        "narrow-support finalist baseline",
    )
    factor = reviewed["N_eff_BR"].to_numpy(float)
    out = pd.DataFrame(
        {
            "mass_GeV": reviewed["mass_GeV"].to_numpy(float),
            "mass_MeV": reviewed["mass_MeV"].to_numpy(float),
            "dataset_set": reviewed["dataset_set"].astype(str),
            "wide_support_eps2_obs_ee_channel": reviewed[
                "eps2_obs_ee_channel"
            ].to_numpy(float),
            "narrow_support_eps2_obs_ee_channel": baseline[
                "eps2_obs"
            ].to_numpy(float),
            "N_eff_BR": factor,
        }
    )
    out["wide_support_eps2_obs_minimal_visible"] = (
        out["wide_support_eps2_obs_ee_channel"] * factor
    )
    out["narrow_support_eps2_obs_minimal_visible"] = (
        out["narrow_support_eps2_obs_ee_channel"] * factor
    )
    out["wide_over_narrow_observed_ratio_ee_channel"] = (
        out["wide_support_eps2_obs_ee_channel"]
        / out["narrow_support_eps2_obs_ee_channel"]
    )
    out["wide_over_narrow_observed_ratio_minimal_visible"] = (
        out["wide_support_eps2_obs_minimal_visible"]
        / out["narrow_support_eps2_obs_minimal_visible"]
    )
    if not np.allclose(
        out["wide_over_narrow_observed_ratio_ee_channel"].to_numpy(float),
        out[
            "wide_over_narrow_observed_ratio_minimal_visible"
        ].to_numpy(float),
        rtol=5.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Common dimuon factor did not cancel in wide/narrow ratio")
    ratio = out["wide_over_narrow_observed_ratio_ee_channel"].to_numpy(float)
    out["wide_relative_change"] = ratio - 1.0
    out["wide_tightening_percent"] = 100.0 * (1.0 - ratio)
    out["source_wide_table"] = repo_path(REVIEWED_BANDS)
    out["source_narrow_table"] = repo_path(baseline_path)
    out["source_narrow_sha256"] = sha256(baseline_path)
    out["input_fraction_match_validated"] = True
    out["ratio_interpretation"] = (
        "wide-support observed epsilon2 divided by otherwise-matched "
        "narrow-support finalist observed epsilon2"
    )
    return out


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


def set_mass_ticks(ax: plt.Axes) -> None:
    majors = np.arange(20.0, 251.0, 10.0)
    ax.set_xlim(float(MASS_LOW_MEV), float(MASS_HIGH_MEV))
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))


def contiguous_segments(
    frame: pd.DataFrame,
    category: str = "dataset_set",
) -> Iterator[pd.DataFrame]:
    work = frame.sort_values("mass_GeV").reset_index(drop=True)
    group_id = work[category].astype(str).ne(
        work[category].astype(str).shift()
    ).cumsum()
    for _, segment in work.groupby(group_id, sort=False):
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


def save_figure(
    fig: plt.Figure,
    stem: str,
    description: str,
) -> Dict[str, Any]:
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
    fig = plt.figure(figsize=(12.4, 7.8))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.17, 3.2, 0.95),
        hspace=0.045,
        left=0.09,
        right=0.98,
        top=0.82,
        bottom=0.16,
    )
    activity = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=activity)
    ratio_ax = fig.add_subplot(grid[2], sharex=activity)
    plot_activity_strip(activity, reviewed)
    x = reviewed["mass_MeV"].to_numpy(float)
    threshold_mev = 1000.0 * DIMUON_THRESHOLD_GEV

    ax.fill_between(
        x,
        reviewed["eps2_lo2_minimal_visible"].to_numpy(float),
        reviewed["eps2_hi2_minimal_visible"].to_numpy(float),
        color=COLORS["band2"],
        alpha=0.76,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        x,
        reviewed["eps2_lo1_minimal_visible"].to_numpy(float),
        reviewed["eps2_hi1_minimal_visible"].to_numpy(float),
        color=COLORS["band1"],
        alpha=0.84,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        x,
        reviewed["eps2_med_minimal_visible"].to_numpy(float),
        color=COLORS["expected"],
        linewidth=1.65,
        linestyle="--",
        zorder=3,
    )
    ax.plot(
        x,
        reviewed["eps2_obs_minimal_visible"].to_numpy(float),
        color=COLORS["observed"],
        linewidth=2.05,
        zorder=4,
    )
    ax.axvline(
        threshold_mev,
        color=COLORS["threshold"],
        linewidth=1.0,
        linestyle=":",
        zorder=5,
    )
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
    ratio_ax.axvline(
        threshold_mev,
        color=COLORS["threshold"],
        linewidth=1.0,
        linestyle=":",
    )
    ratio_ax.set_ylabel("obs / median")
    ratio_ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ratio_ax.set_ylim(
        max(0.0, 0.92 * float(np.min(ratio))),
        1.08 * float(np.max(ratio)),
    )
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
    fig.text(
        0.50,
        0.045,
        (
            "Exactly 300 conditional fixed-GP limit pseudoexperiments per mass. "
            "Bands are descriptive central quantiles, not coverage-calibrated "
            "intervals."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_observed_bands300_minimal_visible",
        "Observed minimal-visible limit, central 68/95% fixed-GP toy-limit "
        "quantiles, activity strip, and observed/median panel.",
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
    one_count = 1.0 / N_TOYS
    zero_marker_y = 0.5 / N_TOYS

    specifications = (
        (
            "p_strong",
            "tail_count_strong_le_observed",
            COLORS["strong"],
            r"$p_{\rm strong}$",
            "-",
        ),
        (
            "p_weak",
            "tail_count_weak_ge_observed",
            COLORS["weak"],
            r"$p_{\rm weak}$",
            "-",
        ),
        (
            "p_two",
            "tail_count_two_sided_min",
            COLORS["two"],
            r"$p_{\rm two}$",
            "--",
        ),
    )
    for p_column, count_column, color, label, linestyle in specifications:
        values = reviewed[p_column].to_numpy(float)
        counts = reviewed[count_column].to_numpy(int)
        nonzero = counts > 0
        plotted = values.copy()
        plotted[~nonzero] = np.nan
        ax.plot(
            x,
            plotted,
            color=color,
            linewidth=1.65,
            linestyle=linestyle,
            label=label,
        )
        if bool((~nonzero).any()):
            ax.scatter(
                x[~nonzero],
                np.full(np.count_nonzero(~nonzero), zero_marker_y),
                marker="v",
                s=22,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=5,
            )
    ax.axhline(
        one_count,
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
    zero_handle = Line2D(
        [0],
        [0],
        marker="v",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="#4B5563",
        label="0/300 marker (shown below 1/300)",
    )
    handles, labels = ax.get_legend_handles_labels()
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
    fig.suptitle(
        "Observed-limit empirical tail diagnostics",
        y=0.975,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.055,
        (
            "A downward marker means no pseudoexperiment entered that empirical "
            "tail (0/300); it is not an exact zero probability. These are "
            "limit diagnostics, not discovery p-values."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_limit_tail_pvalues300",
        "Strong, weak, and bounded two-sided empirical fixed-mass limit-tail "
        "diagnostics with raw-zero markers and one-count resolution.",
    )


def plot_local_p0_sidak(
    sidak: pd.DataFrame,
    neff: float,
) -> Dict[str, Any]:
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
    global_reference = sidak[
        "p_sidak_resolution_spacing_analytic"
    ].to_numpy(float)
    ax.plot(
        x,
        local,
        color=COLORS["local"],
        linewidth=1.9,
        label=r"Local asymptotic $p_0$",
    )
    ax.plot(
        x,
        global_reference,
        color=COLORS["sidak"],
        linewidth=1.8,
        linestyle="--",
        label=(
            rf"Analytic Šidák reference "
            rf"($N_{{\rm eff}}={neff:.2f}$, $W=2.25\sigma_m$)"
        ),
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
    positive = np.concatenate([local[local > 0.0], global_reference[global_reference > 0.0]])
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
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 0.91),
        frameon=False,
        ncol=2,
    )
    fig.suptitle(
        "Local asymptotic search p-value and analytic look-elsewhere reference",
        y=0.975,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.055,
        (
            "The Šidák curve is a resolution-spacing analytic reference. "
            "It is not calibrated with scan pseudoexperiments and does not use "
            "the fixed-mass limit-tail ensemble."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "combined_local_p0_sidak_reference",
        "Local asymptotic p0 and separate 2.25-sigma resolution-spacing analytic "
        "Sidak look-elsewhere reference.",
    )


def plot_support_kernel_audit(
    states: pd.DataFrame,
) -> Dict[str, Any]:
    """Show the support/search separation and reviewed length-scale occupancy."""

    fig, (support_ax, kernel_ax) = plt.subplots(
        2,
        1,
        figsize=(12.4, 7.8),
        gridspec_kw={"height_ratios": (0.88, 1.55), "hspace": 0.34},
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.86,
        bottom=0.12,
    )
    datasets = list(SEARCH_RANGES_GEV)
    y_positions = np.arange(len(datasets), dtype=float)
    for y, dataset in zip(y_positions, datasets):
        support_low, support_high = WIDE_SUPPORT_RANGES_GEV[dataset]
        search_low, search_high = SEARCH_RANGES_GEV[dataset]
        color = COLORS[dataset]
        support_ax.plot(
            [1000.0 * support_low, 1000.0 * support_high],
            [y + 0.10, y + 0.10],
            color=color,
            linewidth=8.0,
            alpha=0.25,
            solid_capstyle="butt",
        )
        support_ax.scatter(
            [1000.0 * support_low, 1000.0 * support_high],
            [y + 0.10, y + 0.10],
            marker="|",
            s=105,
            color=color,
            linewidth=1.5,
            zorder=3,
        )
        support_ax.plot(
            [1000.0 * search_low, 1000.0 * search_high],
            [y - 0.10, y - 0.10],
            color=color,
            linewidth=4.0,
            solid_capstyle="butt",
        )
        support_ax.scatter(
            [1000.0 * search_low, 1000.0 * search_high],
            [y - 0.10, y - 0.10],
            marker="|",
            s=92,
            color=color,
            linewidth=1.35,
            zorder=3,
        )
    support_ax.set_yticks(y_positions)
    support_ax.set_yticklabels(datasets)
    support_ax.set_xlim(10.0, 305.0)
    support_ax.set_xlabel("Mass (MeV)")
    support_ax.set_title("Fit support extends beyond the reported search region")
    support_ax.grid(axis="x", alpha=0.20)
    support_ax.grid(axis="y", visible=False)
    support_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#52616B",
                alpha=0.28,
                linewidth=8.0,
                label="Reviewed GP fit support",
            ),
            Line2D(
                [0],
                [0],
                color="#52616B",
                linewidth=4.0,
                label="Reported search region / state grid",
            ),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
    )

    upper_bound_counts: Dict[str, int] = {}
    for dataset in datasets:
        rows = states[states["dataset"] == dataset].sort_values("mass_GeV")
        ratio = (
            rows["ls_opt"].to_numpy(float)
            / rows["ls_hi"].to_numpy(float)
        )
        at_upper = np.isclose(
            ratio,
            1.0,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        upper_bound_counts[dataset] = int(np.count_nonzero(at_upper))
        kernel_ax.plot(
            rows["mass_MeV"].to_numpy(float),
            ratio,
            color=COLORS[dataset],
            linewidth=1.75,
            label=(
                f"{dataset}: {upper_bound_counts[dataset]}/"
                f"{len(rows)} at upper bound"
            ),
        )
    kernel_ax.axhline(
        1.0,
        color="#343A40",
        linewidth=1.0,
        linestyle="--",
        label=r"Configured upper bound, $\ell_{\rm opt}/\ell_{\rm hi}=1$",
    )
    kernel_ax.set_xlim(10.0, 305.0)
    kernel_ax.set_ylim(0.48, 1.035)
    kernel_ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    kernel_ax.set_ylabel(r"Reviewed kernel occupancy $\ell_{\rm opt}/\ell_{\rm hi}$")
    kernel_ax.set_title("Reviewed maximum-LML kernel states")
    kernel_ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.suptitle(
        "Wide-support configuration and kernel-bound audit",
        y=0.975,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.035,
        (
            "Bound occupancy is a diagnostic of the frozen reviewed fit, "
            "not a prescription to retune the kernel after looking at the data."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "wide_support_search_kernel_audit",
        "Wide fit supports versus search regions and reviewed GP "
        "length-scale occupancy relative to the configured upper bound.",
    )


def plot_wide_narrow_comparison(
    comparison: pd.DataFrame,
) -> Dict[str, Any]:
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
    plot_activity_strip(activity, comparison)
    x = comparison["mass_MeV"].to_numpy(float)
    ratio = comparison[
        "wide_over_narrow_observed_ratio_minimal_visible"
    ].to_numpy(float)
    if not np.isfinite(ratio).all() or not bool((ratio > 0.0).all()):
        raise RuntimeError("Wide/narrow observed-limit ratio is not finite and positive")
    ratio_min = float(np.min(ratio))
    ratio_max = float(np.max(ratio))
    if ratio_min < 1.0:
        ax.axhspan(
            max(1.0e-12, 0.85 * ratio_min),
            1.0,
            facecolor="#E7F3EC",
            alpha=0.65,
            zorder=0,
        )
    if ratio_max > 1.0:
        ax.axhspan(
            1.0,
            1.15 * ratio_max,
            facecolor="#F8E8E4",
            alpha=0.50,
            zorder=0,
        )
    ax.plot(
        x,
        ratio,
        color="#5B2C83",
        linewidth=1.85,
        label="wide support / narrow finalist",
    )
    ax.axhline(
        1.0,
        color="#343A40",
        linewidth=1.0,
        linestyle="--",
        label="unchanged observed limit",
    )
    min_index = int(np.nanargmin(ratio))
    max_index = int(np.nanargmax(ratio))
    ax.scatter(
        [x[min_index], x[max_index]],
        [ratio[min_index], ratio[max_index]],
        color=["#2E7D4F", "#B2472D"],
        edgecolor="white",
        linewidth=0.7,
        s=42,
        zorder=5,
    )
    ax.set_yscale("log")
    lower = max(1.0e-12, 0.90 * ratio_min)
    upper = 1.10 * ratio_max
    if lower >= upper:
        lower, upper = 0.95, 1.05
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax.set_ylabel("Observed 90% CL ratio")
    set_mass_ticks(ax)
    ax.legend(loc="best", frameon=False)
    fig.suptitle(
        "Observed-limit change from widening the 2015 and 2016 fit supports",
        y=0.975,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.055,
        (
            "Matched 2021 10%, 2016 100%, and 2015 100% inputs. "
            "The common dimuon branching correction cancels in this ratio."
        ),
        ha="center",
        fontsize=9.1,
        color="#555B65",
    )
    return save_figure(
        fig,
        "wide_vs_narrow_observed_limit_ratio",
        "Matched-mass observed-limit ratio for wide versus narrow fit "
        "supports, with the active-dataset strip.",
    )


def mass_list_at_extreme(
    frame: pd.DataFrame,
    column: str,
    *,
    mode: str,
) -> Tuple[float, List[int]]:
    values = frame[column].to_numpy(float)
    if mode == "min":
        extreme = float(np.min(values))
    elif mode == "max":
        extreme = float(np.max(values))
    else:
        raise ValueError(f"Unknown extreme mode: {mode}")
    mask = np.isclose(values, extreme, rtol=1.0e-12, atol=1.0e-300)
    masses = [
        int(round(value))
        for value in frame.loc[mask, "mass_MeV"].to_numpy(float)
    ]
    return extreme, masses


def kernel_bound_summary(states: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for dataset in SEARCH_RANGES_GEV:
        rows = states[states["dataset"] == dataset]
        ratio = rows["ls_opt"].to_numpy(float) / rows["ls_hi"].to_numpy(float)
        at_upper = np.isclose(
            ratio,
            1.0,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        result[dataset] = {
            "n_states": int(len(rows)),
            "n_ls_at_upper_bound": int(np.count_nonzero(at_upper)),
            "fraction_ls_at_upper_bound": float(np.mean(at_upper)),
            "min_ls_opt_over_ls_hi": float(np.min(ratio)),
            "median_ls_opt_over_ls_hi": float(np.median(ratio)),
            "max_ls_opt_over_ls_hi": float(np.max(ratio)),
            "interpretation": (
                "diagnostic occupancy of the frozen reviewed state; "
                "not a post-unblinding retuning rule"
            ),
        }
    return result


def build_summaries(
    reviewed: pd.DataFrame,
    sidak: pd.DataFrame,
    comparison: pd.DataFrame,
    states: pd.DataFrame,
    closure: pd.DataFrame,
    neff: float,
    config_match: Dict[str, Any],
    observed_states_path: Path,
    bands_path: Path,
    baseline_path: Path,
    plot_entries: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    tail_payload: Dict[str, Dict[str, Any]] = {}
    tail_specs = (
        (
            "strong",
            "p_strong",
            "tail_count_strong_le_observed",
            "toy limits less than or equal to the observed limit",
            False,
        ),
        (
            "weak",
            "p_weak",
            "tail_count_weak_ge_observed",
            "toy limits greater than or equal to the observed limit",
            False,
        ),
        (
            "two_sided",
            "p_two",
            "tail_count_two_sided_min",
            "twice the smaller strong/weak raw tail, bounded at one",
            True,
        ),
    )
    for (
        name,
        p_column,
        count_column,
        count_semantics,
        two_sided,
    ) in tail_specs:
        minimum, minimum_masses = mass_list_at_extreme(
            reviewed,
            p_column,
            mode="min",
        )
        minimum_rows = reviewed[
            np.isclose(
                reviewed[p_column].to_numpy(float),
                minimum,
                rtol=1.0e-12,
                atol=1.0e-300,
            )
        ]
        raw_counts = sorted(
            {
                int(value)
                for value in minimum_rows[count_column].to_numpy(int)
            }
        )
        zero_count_masses = [
            int(round(value))
            for value in reviewed.loc[
                reviewed[count_column].to_numpy(int) == 0,
                "mass_MeV",
            ].to_numpy(float)
        ]
        interpretation = empirical_tail_label(
            int(min(raw_counts)),
            minimum,
            two_sided=two_sided,
        )
        tail_payload[name] = {
            "minimum_probability": minimum,
            "minimum_masses_MeV": minimum_masses,
            "raw_counts_at_minimum": raw_counts,
            "raw_count_column": count_column,
            "raw_count_semantics": count_semantics,
            "zero_count_masses_MeV": zero_count_masses,
            "n_zero_count_masses": len(zero_count_masses),
            "n_toys_per_mass": N_TOYS,
            "one_count_resolution": 1.0 / N_TOYS,
            "minimum_interpretation": interpretation,
            "fixed_mass_diagnostic": True,
            "discovery_pvalue": False,
        }
        rows.append(
            {
                "family": "empirical_fixed_mass_limit_tail",
                "metric": f"minimum_p_{name}",
                "value": minimum,
                "raw_count": min(raw_counts),
                "n_toys": N_TOYS,
                "masses_MeV": json.dumps(minimum_masses),
                "n_masses": len(minimum_masses),
                "interpretation": interpretation,
            }
        )

    local_minimum, local_masses = mass_list_at_extreme(
        sidak,
        "p0_local_asymptotic",
        mode="min",
    )
    local_index = int(np.nanargmin(sidak["p0_local_asymptotic"].to_numpy(float)))
    sidak_at_minimum = float(
        sidak.iloc[local_index]["p_sidak_resolution_spacing_analytic"]
    )
    rows.extend(
        [
            {
                "family": "local_asymptotic_discovery",
                "metric": "minimum_local_p0",
                "value": local_minimum,
                "raw_count": "",
                "n_toys": "",
                "masses_MeV": json.dumps(local_masses),
                "n_masses": len(local_masses),
                "interpretation": "local asymptotic discovery statistic",
            },
            {
                "family": "analytic_look_elsewhere_reference",
                "metric": "sidak_p_at_local_minimum",
                "value": sidak_at_minimum,
                "raw_count": "",
                "n_toys": "",
                "masses_MeV": json.dumps(local_masses),
                "n_masses": len(local_masses),
                "interpretation": (
                    "resolution-spacing analytic Sidak reference; "
                    "not scan-toy calibrated"
                ),
            },
        ]
    )

    ratio_metrics: Dict[str, Dict[str, Any]] = {}
    for key, column, label in (
        (
            "observed_over_median",
            "observed_over_median_minimal_visible",
            "observed divided by fixed-GP toy-limit median",
        ),
        (
            "wide_over_narrow_observed",
            "wide_over_narrow_observed_ratio_minimal_visible",
            "wide-support observed limit divided by narrow-support finalist",
        ),
    ):
        source = reviewed if key == "observed_over_median" else comparison
        minimum, minimum_masses = mass_list_at_extreme(
            source,
            column,
            mode="min",
        )
        maximum, maximum_masses = mass_list_at_extreme(
            source,
            column,
            mode="max",
        )
        ratio_metrics[key] = {
            "minimum": minimum,
            "minimum_masses_MeV": minimum_masses,
            "maximum": maximum,
            "maximum_masses_MeV": maximum_masses,
            "interpretation": label,
        }
        rows.extend(
            [
                {
                    "family": "ratio_diagnostic",
                    "metric": f"minimum_{key}",
                    "value": minimum,
                    "raw_count": "",
                    "n_toys": "",
                    "masses_MeV": json.dumps(minimum_masses),
                    "n_masses": len(minimum_masses),
                    "interpretation": label,
                },
                {
                    "family": "ratio_diagnostic",
                    "metric": f"maximum_{key}",
                    "value": maximum,
                    "raw_count": "",
                    "n_toys": "",
                    "masses_MeV": json.dumps(maximum_masses),
                    "n_masses": len(maximum_masses),
                    "interpretation": label,
                },
            ]
        )

    wide_ratio_payload = ratio_metrics["wide_over_narrow_observed"]
    tightening_percent = 100.0 * (
        1.0 - float(wide_ratio_payload["minimum"])
    )
    weakening_percent = 100.0 * (
        float(wide_ratio_payload["maximum"]) - 1.0
    )
    wide_ratio_payload["tightening_percent_at_minimum_ratio"] = (
        tightening_percent
    )
    wide_ratio_payload["weakening_percent_at_maximum_ratio"] = (
        weakening_percent
    )
    rows.extend(
        [
            {
                "family": "ratio_diagnostic",
                "metric": "wide_support_tightening_percent_at_minimum_ratio",
                "value": tightening_percent,
                "raw_count": "",
                "n_toys": "",
                "masses_MeV": json.dumps(
                    wide_ratio_payload["minimum_masses_MeV"]
                ),
                "n_masses": len(wide_ratio_payload["minimum_masses_MeV"]),
                "interpretation": (
                    "positive means the wide-support observed upper limit is "
                    "tighter at the minimum ratio"
                ),
            },
            {
                "family": "ratio_diagnostic",
                "metric": "wide_support_weakening_percent_at_maximum_ratio",
                "value": weakening_percent,
                "raw_count": "",
                "n_toys": "",
                "masses_MeV": json.dumps(
                    wide_ratio_payload["maximum_masses_MeV"]
                ),
                "n_masses": len(wide_ratio_payload["maximum_masses_MeV"]),
                "interpretation": (
                    "positive means the wide-support observed upper limit is "
                    "weaker at the maximum ratio"
                ),
            },
        ]
    )

    kernel_summary = kernel_bound_summary(states)
    for dataset, payload in kernel_summary.items():
        rows.append(
            {
                "family": "reviewed_gp_state_diagnostic",
                "metric": f"{dataset}_ls_at_upper_bound",
                "value": payload["fraction_ls_at_upper_bound"],
                "raw_count": payload["n_ls_at_upper_bound"],
                "n_toys": "",
                "masses_MeV": "",
                "n_masses": payload["n_states"],
                "interpretation": payload["interpretation"],
            }
        )

    factor_250 = float(
        reviewed.loc[
            np.isclose(
                reviewed["mass_MeV"].to_numpy(float),
                250.0,
                rtol=0.0,
                atol=1.0e-10,
            ),
            "N_eff_BR",
        ].iloc[0]
    )
    rows.append(
        {
            "family": "minimal_visible_reinterpretation",
            "metric": "N_eff_BR_at_250_MeV",
            "value": factor_250,
            "raw_count": "",
            "n_toys": "",
            "masses_MeV": "[250]",
            "n_masses": 1,
            "interpretation": (
                "common e+e-/mu+mu- visible-width factor multiplying observed "
                "and every fixed-GP toy-limit quantile"
            ),
        }
    )

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN_DIR.name,
        "status": "reviewed postprocessing complete",
        "inputs": {
            "reviewed_observed_states": {
                "path": repo_path(observed_states_path),
                "sha256": sha256(observed_states_path),
                "rows": int(len(states)),
            },
            "fixed_gp_300toy_bands": {
                "path": repo_path(bands_path),
                "sha256": sha256(bands_path),
                "rows": int(len(reviewed)),
            },
            "narrow_support_finalist": {
                "path": repo_path(baseline_path),
                "sha256": sha256(baseline_path),
                "rows": int(len(comparison)),
            },
            "matched_config_validation": config_match,
        },
        "grid": {
            "mass_low_MeV": MASS_LOW_MEV,
            "mass_high_MeV": MASS_HIGH_MEV,
            "step_MeV": 1,
            "n_mass_points": N_MASSES,
            "active_set_counts": EXPECTED_ACTIVE_COUNTS,
            "reviewed_state_rows": int(len(states)),
            "fixed_state_closure_rows": int(len(closure)),
        },
        "supports_GeV": {
            "fit_support_wide": {
                dataset: list(bounds)
                for dataset, bounds in WIDE_SUPPORT_RANGES_GEV.items()
            },
            "reported_search_region": {
                dataset: list(bounds)
                for dataset, bounds in SEARCH_RANGES_GEV.items()
            },
            "narrow_finalist_support": {
                dataset: list(bounds)
                for dataset, bounds in NARROW_SUPPORT_RANGES_GEV.items()
            },
        },
        "ensemble": {
            "n_toys_requested_per_mass": N_TOYS,
            "n_toys_finite_per_mass": N_TOYS,
            "seed": SEED,
            "child_seed_index": "mass_MeV - 19",
            "gp_refit_per_toy": False,
            "gp_reoptimized_for_bands": False,
            "observed_state_mode": "fixed_reviewed_max_lml",
            "cls_statistic": "tilde_q_mu",
            "cls_calibration": "asymptotic",
            "combined_mode": "count_scale",
            "bands_are_coverage_calibrated": False,
            "bands_interpretation": (
                "descriptive central quantiles of conditional fixed-GP "
                "background-only limit pseudoexperiments"
            ),
        },
        "empirical_limit_tail_diagnostics": tail_payload,
        "local_asymptotic_search": {
            "minimum_p0": local_minimum,
            "minimum_masses_MeV": local_masses,
            "sidak_p_at_local_minimum": sidak_at_minimum,
            "N_eff_resolution_spacing": neff,
            "independence_width_sigma": INDEPENDENCE_WIDTH_SIGMA,
            "sidak_is_scan_toy_calibrated": False,
            "sidak_uses_limit_tail_ensemble": False,
            "interpretation": (
                "local asymptotic p0 plus a separate analytic "
                "resolution-spacing Sidak reference"
            ),
        },
        "minimal_visible_reinterpretation": {
            "muon_mass_GeV": M_MU_GEV,
            "dimuon_threshold_GeV": DIMUON_THRESHOLD_GEV,
            "first_corrected_grid_mass_MeV": 212,
            "N_eff_BR_at_250_MeV": factor_250,
            "pvalues_changed": False,
            "toy_quantiles_recomputed": False,
            "toy_ensemble_changed": False,
            "common_factor_cancels_in_ratios": True,
        },
        "ratio_diagnostics": ratio_metrics,
        "reviewed_kernel_bound_diagnostics": kernel_summary,
        "semantic_boundaries": {
            "empirical_zero": (
                "zero raw count at finite 1/300 resolution; never an exact "
                "probability-zero statement"
            ),
            "tail_pvalues": (
                "fixed-mass observed-limit ensemble diagnostics, "
                "not discovery p-values"
            ),
            "sidak": (
                "analytic resolution-spacing reference, not "
                "pseudoexperiment-calibrated"
            ),
            "bands": (
                "conditional fixed-GP limit quantiles, not a coverage result"
            ),
        },
        "artifacts": {
            "reviewed_bands": repo_path(REVIEWED_BANDS),
            "gp_state_closure": repo_path(GP_STATE_CLOSURE),
            "sidak_reference": repo_path(SIDAK_TABLE),
            "wide_narrow_comparison": repo_path(WIDE_NARROW_TABLE),
            "summary_csv": repo_path(SUMMARY_CSV),
            "summary_json": repo_path(SUMMARY_JSON),
            "plot_manifest": repo_path(PLOT_MANIFEST),
        },
        "figures": list(plot_entries),
    }
    summary_frame = pd.DataFrame(rows)
    return summary, summary_frame


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_plot_manifest(
    plot_entries: Sequence[Dict[str, Any]],
    observed_states_path: Path,
    bands_path: Path,
    baseline_path: Path,
) -> Dict[str, Any]:
    output_tables = (
        REVIEWED_BANDS,
        GP_STATE_CLOSURE,
        SIDAK_TABLE,
        WIDE_NARROW_TABLE,
        SUMMARY_CSV,
        SUMMARY_JSON,
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN_DIR.name,
        "inputs": [
            {
                "path": repo_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                observed_states_path,
                bands_path,
                baseline_path,
                V4_CONFIG,
                BASELINE_CONFIG,
            )
        ],
        "derived_tables": [
            {
                "path": repo_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_tables
        ],
        "figures": list(plot_entries),
        "semantic_boundaries": [
            "Empirical 0/300 is below one-count resolution, not exact p=0.",
            "The Sidak curve is analytic and is not scan-toy calibrated.",
            "The fixed-GP toy-limit bands are descriptive, not coverage calibrated.",
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and postprocess the reviewed v4 fixed-state 300-toy "
            "combined upper-limit ensemble."
        )
    )
    parser.add_argument(
        "--observed-states",
        type=Path,
        default=DEFAULT_OBSERVED_STATES,
        help="Reviewed 415-state observed GP ledger.",
    )
    parser.add_argument(
        "--bands",
        type=Path,
        default=DEFAULT_BANDS,
        help="Combined 300-toy cached-runner table.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Matched narrow-support finalist table.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    observed_states_path = args.observed_states.resolve()
    bands_path = args.bands.resolve()
    baseline_path = args.baseline.resolve()
    required_paths = (
        observed_states_path,
        bands_path,
        baseline_path,
        V4_CONFIG,
        BASELINE_CONFIG,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Postprocessing was not run because required reviewed inputs are "
            f"missing:\n{formatted}"
        )

    config_match = validate_matched_configs()
    states = validate_observed_states(pd.read_csv(observed_states_path))
    bands = validate_bands(pd.read_csv(bands_path))
    bands_with_states, closure = validate_gp_state_metadata(bands, states)
    reviewed = build_reviewed_table(
        bands_with_states,
        bands_path,
        observed_states_path,
    )
    sidak, neff = build_sidak_table(reviewed)
    comparison = build_wide_narrow_comparison(reviewed, baseline_path)

    atomic_write_csv(reviewed, REVIEWED_BANDS)
    atomic_write_csv(closure, GP_STATE_CLOSURE)
    atomic_write_csv(sidak, SIDAK_TABLE)
    atomic_write_csv(comparison, WIDE_NARROW_TABLE)

    setup_style()
    plot_entries = [
        plot_combined_bands(reviewed),
        plot_empirical_tails(reviewed),
        plot_local_p0_sidak(sidak, neff),
        plot_support_kernel_audit(states),
        plot_wide_narrow_comparison(comparison),
    ]
    summary, summary_frame = build_summaries(
        reviewed,
        sidak,
        comparison,
        states,
        closure,
        neff,
        config_match,
        observed_states_path,
        bands_path,
        baseline_path,
        plot_entries,
    )
    atomic_write_csv(summary_frame, SUMMARY_CSV)
    atomic_write_json(summary, SUMMARY_JSON)
    manifest = build_plot_manifest(
        plot_entries,
        observed_states_path,
        bands_path,
        baseline_path,
    )
    atomic_write_json(manifest, PLOT_MANIFEST)

    print(
        "Validated and postprocessed "
        f"{N_MASSES} masses x {N_TOYS} finite fixed-GP toys."
    )
    print(f"Reviewed table: {REVIEWED_BANDS}")
    print(f"Summary: {SUMMARY_JSON}")
    print(f"Figures: {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
