#!/usr/bin/env python3
"""Build the reviewed v4.2 observed-state and individual-limit ledgers.

This script deliberately does not run a fit and does not construct individual
expected bands.  It combines:

* the detailed reviewed v4 observed ledger, which carries the density and
  training-geometry metadata;
* the accepted compact v4.1 k=12 ledger, which selects the reviewed fit state
  at every mass; and
* the selected source row named by that compact ledger.

Every selected source file is content-hash checked before its row is used.
The 2016 entries are also cross-checked against the accepted factor-12
pointwise grid.  Any missing/duplicate key, interpolation, stale source hash,
non-finite result, incomplete density window, or factor-12 upper-bound
occupancy aborts the build.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
RESULT_DIR = SCRIPT.parent
DERIVED_DIR = RESULT_DIR / "derived"

DETAILED_REL = (
    "study_results/v4_wide_support_2015full_2016full_2021_10pct_20260803/"
    "derived/observed_gp_states_reviewed.csv"
)
COMPACT_REL = (
    "study_results/v4p1_2016_ls_upper_optimization_20260804/"
    "derived/observed_gp_states_k12_reviewed.csv"
)
FACTOR_GRID_REL = (
    "study_results/v4p1_2016_ls_upper_optimization_20260804/"
    "derived/pointwise_factor_grid.csv"
)
CONFIG_REL = (
    "study_configs/v4p1_2016_ls_upper_optimization_20260804/"
    "config_obsUL90_combined_wide_support_v4p1_2016k12_observed_only.yaml"
)

EXPECTED_HASHES = {
    DETAILED_REL: "0bf70d1516ffab383dc50278b9adf6b568beafac9c4362ad5b6840b776a65dd0",
    COMPACT_REL: "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9",
    FACTOR_GRID_REL: "6b5be702f1b65688f6ce7f86922716f012f72cb8c9f33320b5783c08a0123657",
    CONFIG_REL: "d8ba2483253b653305d9db3318756fd7c5923d9b414dec1b824e75e226282732",
}

ENRICHED_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "derived/observed_gp_states_v4p2_enriched.csv"
)
INDIVIDUAL_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "derived/individual_observed_limits_v4p2.csv"
)
SUMMARY_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "derived/individual_ledger_validation_v4p2.json"
)

ENRICHED_PATH = REPO / ENRICHED_REL
INDIVIDUAL_PATH = REPO / INDIVIDUAL_REL
SUMMARY_PATH = REPO / SUMMARY_REL

KEY_COLUMNS = ["dataset", "mass_MeV"]
EXPECTED_MASS_GRIDS = {
    2015: set(range(19, 91)),
    2016: set(range(39, 181)),
    2021: set(range(50, 251)),
}

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
    "optimizer_repair_applied",
    "review_status",
    "branch_multiplicity",
    "reproducing_sources",
    "selected_repair_reproduced",
    "repair_reproduction_pending",
    "candidate_count",
    "repair_candidate_count",
    "delta_lml_selected_minus_raw",
}

DETAILED_REQUIRED_COLUMNS = {
    "dataset",
    "mass_GeV",
    "mass_MeV",
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "integral_density",
    "density_window_fully_covered",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "extract_success",
    "kernel_str",
    "ls_lo",
    "ls_hi",
    "ls_opt",
    "sigma_x",
    "const_opt",
    "lml",
    "n_train",
    "selected_attempt",
    "selected_source",
    "selected_source_sha256",
    "row_source",
    "review_status",
    "branch_multiplicity",
    "interpolated",
}

PRESERVED_GEOMETRY_COLUMNS = [
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
    "optimizer_restarts",
    "sigma_x",
    "ls_lo",
]

FINITE_COLUMNS = [
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "const_opt",
    "ls_opt",
    "ls_hi",
    "lml",
    "sigma_val",
    "integral_density",
]

SOURCE_RESULT_COLUMNS = [
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "const_opt",
    "ls_opt",
    "ls_hi",
    "lml",
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "n_train",
]

MINIMAL_VISIBLE_MUON_MASS_GEV = 0.1056583745


class ValidationError(RuntimeError):
    """Raised when an input or derived ledger violates a frozen requirement."""


def fail(message: str) -> None:
    raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        fail(f"invalid repository-relative path: {relative!r}")
    candidate = (REPO / relative).resolve()
    try:
        candidate.relative_to(REPO.resolve())
    except ValueError as exc:
        raise ValidationError(f"path escapes repository: {relative}") from exc
    if not candidate.is_file():
        fail(f"required file is absent: {relative}")
    return candidate


def verify_frozen_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = checked_repo_path(relative)
        actual = sha256_file(path)
        if actual != expected:
            fail(
                f"frozen input hash mismatch for {relative}: "
                f"expected {expected}, observed {actual}"
            )
        observed[relative] = actual
    return observed


def mass_mev(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    rounded = np.rint(values * 1000.0).astype(np.int64)
    if not np.allclose(values, rounded / 1000.0, rtol=0.0, atol=5.0e-10):
        fail("mass_GeV contains a value that is not on an integer-MeV grid")
    return pd.Series(rounded, index=series.index, dtype="int64")


def add_and_validate_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    result["dataset"] = pd.to_numeric(result["dataset"], errors="raise").astype(int)
    derived_mass = mass_mev(result["mass_GeV"])
    if "mass_MeV" in result.columns:
        supplied = pd.to_numeric(result["mass_MeV"], errors="raise").astype(int)
        if not supplied.equals(derived_mass):
            fail(f"{name}: mass_MeV is inconsistent with mass_GeV")
    result["mass_MeV"] = derived_mass
    if result.duplicated(KEY_COLUMNS).any():
        duplicate = result.loc[result.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        fail(f"{name}: duplicate dataset/mass keys: {duplicate.head().to_dict('records')}")
    return result


def validate_expected_grid(frame: pd.DataFrame, name: str) -> None:
    if len(frame) != sum(len(values) for values in EXPECTED_MASS_GRIDS.values()):
        fail(f"{name}: expected 415 rows, observed {len(frame)}")
    observed_datasets = set(frame["dataset"].astype(int))
    if observed_datasets != set(EXPECTED_MASS_GRIDS):
        fail(
            f"{name}: dataset set mismatch: "
            f"expected {sorted(EXPECTED_MASS_GRIDS)}, observed {sorted(observed_datasets)}"
        )
    for dataset, expected in EXPECTED_MASS_GRIDS.items():
        observed = set(
            frame.loc[frame["dataset"].eq(dataset), "mass_MeV"].astype(int)
        )
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            fail(
                f"{name}: {dataset} mass grid mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )


def boolean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.dtype == bool:
        return values
    mapping = {"true": True, "false": False}
    lowered = values.astype(str).str.strip().str.lower()
    if not lowered.isin(mapping).all():
        fail(f"{column}: values are not unambiguous booleans")
    return lowered.map(mapping).astype(bool)


def same_value(left: Any, right: Any, *, label: str) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return bool(left) == bool(right)
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    if not (math.isfinite(left_float) and math.isfinite(right_float)):
        return left_float == right_float
    return math.isclose(left_float, right_float, rel_tol=2.0e-12, abs_tol=2.0e-13)


def assert_columns_match(
    assembled: pd.DataFrame,
    reference: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    left = assembled.set_index(KEY_COLUMNS)
    right = reference.set_index(KEY_COLUMNS)
    if set(left.index) != set(right.index):
        fail(f"{label}: comparison key sets differ")
    right = right.loc[left.index]
    for column in columns:
        if column not in left.columns or column not in right.columns:
            fail(f"{label}: required comparison column absent: {column}")
        mismatches = [
            key
            for key, lvalue, rvalue in zip(
                left.index, left[column].tolist(), right[column].tolist()
            )
            if not same_value(lvalue, rvalue, label=f"{label}:{column}:{key}")
        ]
        if mismatches:
            fail(
                f"{label}: {column} differs at {len(mismatches)} keys; "
                f"first={mismatches[:5]}"
            )


ATTEMPT_PATTERNS = (
    re.compile(r"^observed_(attempt_\d+)$"),
    re.compile(r"^(raw_attempt_\d+)$"),
    re.compile(r"^(m\d+_attempt_\d+)$"),
    re.compile(r"^(attempt_\d+)$"),
)


def attempt_from_source(relative: str) -> str:
    for part in reversed(Path(relative).parts):
        for pattern in ATTEMPT_PATTERNS:
            match = pattern.fullmatch(part)
            if match:
                return match.group(1)
    fail(f"cannot derive attempt identifier from selected source: {relative}")
    raise AssertionError("unreachable")


def split_sources(value: Any) -> list[str]:
    sources = [item.strip() for item in str(value).split("|") if item.strip()]
    if not sources:
        fail(f"empty reproducing_sources field: {value!r}")
    return sources


def source_row(
    relative: str,
    dataset: int,
    mass: int,
    cache: dict[str, pd.DataFrame],
) -> pd.Series:
    if relative not in cache:
        frame = pd.read_csv(checked_repo_path(relative))
        if not {"dataset", "mass_GeV"}.issubset(frame.columns):
            fail(f"source lacks dataset/mass_GeV: {relative}")
        cache[relative] = add_and_validate_keys(frame, relative)
    frame = cache[relative]
    selected = frame.loc[
        frame["dataset"].eq(dataset) & frame["mass_MeV"].eq(mass)
    ]
    if len(selected) != 1:
        fail(
            f"{relative}: expected one row for dataset={dataset}, "
            f"mass_MeV={mass}; observed {len(selected)}"
        )
    return selected.iloc[0]


def selected_sources_checked(
    compact: pd.DataFrame,
    cache: dict[str, pd.DataFrame],
) -> tuple[dict[tuple[int, int], pd.Series], dict[str, str]]:
    selected_rows: dict[tuple[int, int], pd.Series] = {}
    checked_hashes: dict[str, str] = {}
    expected_by_source: dict[str, str] = {}
    for row in compact.itertuples(index=False):
        relative = str(row.selected_source)
        expected = str(row.selected_source_sha256)
        previous = expected_by_source.setdefault(relative, expected)
        if previous != expected:
            fail(f"conflicting expected hashes for selected source {relative}")
        if relative not in checked_hashes:
            actual = sha256_file(checked_repo_path(relative))
            if actual != expected:
                fail(
                    f"selected source hash mismatch for {relative}: "
                    f"expected {expected}, observed {actual}"
                )
            checked_hashes[relative] = actual
        selected_rows[(int(row.dataset), int(row.mass_MeV))] = source_row(
            relative, int(row.dataset), int(row.mass_MeV), cache
        )
    if len(selected_rows) != 415:
        fail(f"expected 415 selected source rows, observed {len(selected_rows)}")
    return selected_rows, checked_hashes


def update_from_selected_rows(
    detailed: pd.DataFrame,
    compact: pd.DataFrame,
    selected_rows: dict[tuple[int, int], pd.Series],
) -> pd.DataFrame:
    assembled = detailed.copy().set_index(KEY_COLUMNS, drop=False)
    compact_indexed = compact.set_index(KEY_COLUMNS, drop=False)

    for key, source in selected_rows.items():
        for column in source.index:
            if column in assembled.columns and column not in KEY_COLUMNS:
                assembled.at[key, column] = source[column]

    # Compact-review provenance is authoritative for all datasets, and its
    # selected fit-state fields explicitly guard the selected-row overlay.
    for column in compact.columns:
        if column not in KEY_COLUMNS and column != "mass_GeV":
            assembled[column] = compact_indexed.loc[assembled.index, column].to_numpy()

    assembled = assembled.reset_index(drop=True)

    for row in compact.itertuples(index=False):
        key = (int(row.dataset), int(row.mass_MeV))
        source = selected_rows[key]
        current = assembled.loc[
            assembled["dataset"].eq(key[0]) & assembled["mass_MeV"].eq(key[1])
        ].iloc[0]
        for column in SOURCE_RESULT_COLUMNS:
            if column in source.index and column in current.index:
                if not same_value(current[column], source[column], label=f"{key}:{column}"):
                    fail(
                        f"assembled value does not match selected source at "
                        f"{key}, column={column}"
                    )
        for column in ["const_opt", "ls_opt", "lml", "ls_hi", "ls_hi_over_sigma_x"]:
            if not same_value(current[column], getattr(row, column), label=f"{key}:{column}"):
                fail(
                    f"compact selected state does not match source at "
                    f"{key}, column={column}"
                )

    return assembled


def rebuild_attempt_provenance(
    assembled: pd.DataFrame,
    source_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = assembled.copy()
    selected_attempts: list[str] = []
    reproducing_attempts: list[str] = []
    reproducing_other_attempts: list[str] = []
    all_attempt_sources: list[str] = []
    delta_lml: list[float] = []
    delta_const: list[float] = []
    delta_ls: list[float] = []

    for row in result.itertuples(index=False):
        selected_source = str(row.selected_source)
        selected_attempt = attempt_from_source(selected_source)
        sources = split_sources(row.reproducing_sources)
        if selected_source not in sources:
            fail(
                f"selected source is absent from reproducing_sources at "
                f"dataset={row.dataset}, mass_MeV={row.mass_MeV}"
            )
        attempts = [attempt_from_source(source) for source in sources]
        selected = source_row(
            selected_source, int(row.dataset), int(row.mass_MeV), source_cache
        )
        reproductions = [
            source_row(source, int(row.dataset), int(row.mass_MeV), source_cache)
            for source in sources
        ]
        for required in ["lml", "const_opt", "ls_opt"]:
            if any(required not in candidate.index for candidate in reproductions):
                fail(
                    f"reproducing source lacks {required} at "
                    f"dataset={row.dataset}, mass_MeV={row.mass_MeV}"
                )

        selected_attempts.append(selected_attempt)
        reproducing_attempts.append("|".join(attempts))
        reproducing_other_attempts.append(
            "|".join(attempt for attempt in attempts if attempt != selected_attempt)
        )
        all_attempt_sources.append("|".join(sources))
        delta_lml.append(
            max(abs(float(candidate["lml"]) - float(selected["lml"])) for candidate in reproductions)
        )
        delta_const.append(
            max(
                abs(float(candidate["const_opt"]) - float(selected["const_opt"]))
                for candidate in reproductions
            )
        )
        delta_ls.append(
            max(
                abs(float(candidate["ls_opt"]) - float(selected["ls_opt"]))
                for candidate in reproductions
            )
        )

    result["selected_attempt"] = selected_attempts
    result["reproducing_attempts"] = reproducing_attempts
    result["reproducing_other_attempts"] = reproducing_other_attempts
    result["all_attempt_sources"] = all_attempt_sources
    result["max_abs_reproducing_delta_lml"] = delta_lml
    result["max_abs_reproducing_delta_const_opt"] = delta_const
    result["max_abs_reproducing_delta_ls_opt"] = delta_ls
    return result


def refresh_2016_bound_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mask = result["dataset"].eq(2016)
    for column in ["const_opt", "const_lo", "const_hi", "ls_opt", "ls_lo", "ls_hi"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result.loc[mask, "const_at_lower"] = (
        result.loc[mask, "const_opt"] / result.loc[mask, "const_lo"]
    ).le(1.001)
    result.loc[mask, "const_at_upper"] = (
        result.loc[mask, "const_opt"] / result.loc[mask, "const_hi"]
    ).ge(0.999)
    result.loc[mask, "ls_at_lower"] = (
        result.loc[mask, "ls_opt"] / result.loc[mask, "ls_lo"]
    ).le(1.001)
    result.loc[mask, "ls_at_upper"] = (
        result.loc[mask, "ls_opt"] / result.loc[mask, "ls_hi"]
    ).ge(0.999)
    return result


def validate_factor12(
    assembled: pd.DataFrame,
    compact: pd.DataFrame,
    factor_grid: pd.DataFrame,
) -> dict[str, Any]:
    factor = factor_grid.loc[
        np.isclose(
            pd.to_numeric(factor_grid["upper_factor"], errors="raise"),
            12.0,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()
    if len(factor) != 142:
        fail(f"factor-12 pointwise grid must have 142 rows; observed {len(factor)}")
    factor["dataset"] = 2016
    factor = add_and_validate_keys(factor, "factor-12 pointwise grid")
    expected_2016 = EXPECTED_MASS_GRIDS[2016]
    if set(factor["mass_MeV"].astype(int)) != expected_2016:
        fail("factor-12 pointwise grid does not cover the full 2016 mass grid")
    if boolean_series(factor, "at_upper_boundary").any():
        occupied = factor.loc[boolean_series(factor, "at_upper_boundary"), "mass_MeV"]
        fail(f"accepted factor-12 grid has upper-bound occupancy at {occupied.tolist()}")

    compact_2016 = compact.loc[compact["dataset"].eq(2016)].copy()
    assembled_2016 = assembled.loc[assembled["dataset"].eq(2016)].copy()
    compare_columns = [
        "A_up",
        "eps2_up",
        "p0_analytic",
        "Z_analytic",
        "lml",
        "ls_lo",
        "ls_hi",
        "ls_opt",
        "sigma_x",
        "n_train",
        "blind_lo",
        "blind_hi",
        "sigma_val",
    ]
    assert_columns_match(
        assembled_2016,
        factor,
        compare_columns,
        label="accepted factor-12 pointwise values",
    )

    factor_idx = factor.set_index(KEY_COLUMNS)
    compact_idx = compact_2016.set_index(KEY_COLUMNS)
    for compact_column, factor_column in [
        ("selected_source", "row_source"),
        ("selected_source_sha256", "row_source_sha256"),
        ("review_status", "review_status"),
    ]:
        left = compact_idx[compact_column].astype(str)
        right = factor_idx.loc[left.index, factor_column].astype(str)
        if not left.equals(right):
            fail(
                f"factor-12 {factor_column} does not match compact "
                f"{compact_column}"
            )

    ratio = (
        pd.to_numeric(assembled_2016["ls_hi"], errors="raise")
        / pd.to_numeric(assembled_2016["sigma_x"], errors="raise")
    )
    if not np.allclose(ratio, 12.0, rtol=2.0e-12, atol=2.0e-12):
        fail("2016 enriched ledger is not uniformly at ls_hi/sigma_x = 12")
    occupancy = (
        pd.to_numeric(assembled_2016["ls_opt"], errors="raise")
        / pd.to_numeric(assembled_2016["ls_hi"], errors="raise")
    ).ge(0.999)
    if occupancy.any():
        fail(
            "2016 enriched ledger has accepted-bound occupancy at "
            f"{assembled_2016.loc[occupancy, 'mass_MeV'].astype(int).tolist()}"
        )

    repair_mask = compact_2016["review_status"].astype(str).str.startswith("repair_")
    repair_masses = sorted(compact_2016.loc[repair_mask, "mass_MeV"].astype(int))
    if repair_masses != [43, 125, 145]:
        fail(f"unexpected factor-12 repair mass set: {repair_masses}")
    if not boolean_series(
        compact_2016.loc[repair_mask], "selected_repair_reproduced"
    ).all():
        fail("one or more accepted factor-12 repairs were not reproduced")

    return {
        "upper_length_scale_factor": 12.0,
        "rows": 142,
        "upper_boundary_occupancy_count": int(occupancy.sum()),
        "repair_masses_MeV": repair_masses,
        "review_status_counts": {
            str(key): int(value)
            for key, value in Counter(compact_2016["review_status"].astype(str)).items()
        },
    }


def validate_assembled(
    assembled: pd.DataFrame,
    detailed: pd.DataFrame,
) -> dict[str, Any]:
    validate_expected_grid(assembled, "enriched ledger")
    if boolean_series(assembled, "interpolated").any():
        fail("enriched ledger contains interpolated rows")
    if boolean_series(assembled, "repair_reproduction_pending").any():
        fail("enriched ledger contains a pending repair reproduction")
    if not boolean_series(assembled, "extract_success").all():
        fail("enriched ledger contains an unsuccessful extraction")
    if not boolean_series(assembled, "density_window_fully_covered").all():
        fail("enriched ledger contains an incompletely covered density window")

    for column in FINITE_COLUMNS:
        values = pd.to_numeric(assembled[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            fail(f"enriched ledger contains non-finite {column}")
    for column in ["A_up", "eps2_up", "sigma_A", "sigma_val", "integral_density"]:
        if not (pd.to_numeric(assembled[column], errors="raise") > 0.0).all():
            fail(f"enriched ledger contains non-positive {column}")
    p0 = pd.to_numeric(assembled["p0_analytic"], errors="raise")
    if not p0.between(0.0, 1.0, inclusive="both").all():
        fail("p0_analytic lies outside [0,1]")
    if not (pd.to_numeric(assembled["Z_analytic"], errors="raise") >= 0.0).all():
        fail("Z_analytic contains a negative value")

    assert_columns_match(
        assembled,
        detailed,
        PRESERVED_GEOMETRY_COLUMNS,
        label="unchanged geometry/density metadata",
    )
    return {
        "rows": int(len(assembled)),
        "dataset_rows": {
            str(dataset): int(count)
            for dataset, count in assembled["dataset"].value_counts().sort_index().items()
        },
        "interpolated_rows": int(boolean_series(assembled, "interpolated").sum()),
        "pending_repair_rows": int(
            boolean_series(assembled, "repair_reproduction_pending").sum()
        ),
        "finite_result_columns": FINITE_COLUMNS,
        "preserved_geometry_columns": PRESERVED_GEOMETRY_COLUMNS,
    }


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def minimal_visible_factor(mass_gev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    factor = np.ones_like(mass_gev, dtype=float)
    threshold = 2.0 * MINIMAL_VISIBLE_MUON_MASS_GEV
    above = mass_gev > threshold
    if np.any(above):
        ratio = (MINIMAL_VISIBLE_MUON_MASS_GEV / mass_gev[above]) ** 2
        beta = np.sqrt(1.0 - 4.0 * ratio)
        factor[above] = 1.0 + beta * (1.0 + 2.0 * ratio)
    return factor, above


def build_individual(
    enriched: pd.DataFrame,
    enriched_sha256: str,
    config_sha256: str,
) -> pd.DataFrame:
    mass = pd.to_numeric(enriched["mass_GeV"], errors="raise").to_numpy(dtype=float)
    factor, corrected = minimal_visible_factor(mass)
    eps2 = pd.to_numeric(enriched["eps2_up"], errors="raise").to_numpy(dtype=float)
    sample_labels = {
        2015: "2015 100%",
        2016: "2016 100%",
        2021: "2021 10%",
    }
    individual = pd.DataFrame(
        {
            "dataset": enriched["dataset"].astype(int),
            "sample_label": enriched["dataset"].map(sample_labels),
            "mass_GeV": mass,
            "mass_MeV": enriched["mass_MeV"].astype(int),
            "A_up": pd.to_numeric(enriched["A_up"], errors="raise"),
            "eps2_up": eps2,
            "eps2_observed_ee_channel": eps2,
            "minimal_visible_factor": factor,
            "BR_ee_minimal": 1.0 / factor,
            "eps2_observed_minimal_visible": eps2 * factor,
            "dimuon_correction_applied": corrected,
            "p0_analytic": pd.to_numeric(enriched["p0_analytic"], errors="raise"),
            "Z_analytic": pd.to_numeric(enriched["Z_analytic"], errors="raise"),
            "sigma_val": pd.to_numeric(enriched["sigma_val"], errors="raise"),
            "integral_density": pd.to_numeric(
                enriched["integral_density"], errors="raise"
            ),
            "const_opt": pd.to_numeric(enriched["const_opt"], errors="raise"),
            "ls_opt": pd.to_numeric(enriched["ls_opt"], errors="raise"),
            "ls_hi": pd.to_numeric(enriched["ls_hi"], errors="raise"),
            "ls_opt_over_ls_hi": (
                pd.to_numeric(enriched["ls_opt"], errors="raise")
                / pd.to_numeric(enriched["ls_hi"], errors="raise")
            ),
            "lml": pd.to_numeric(enriched["lml"], errors="raise"),
            "selected_attempt": enriched["selected_attempt"].astype(str),
            "selected_source": enriched["selected_source"].astype(str),
            "selected_source_sha256": enriched["selected_source_sha256"].astype(str),
            "row_source": enriched["row_source"].astype(str),
            "optimizer_repair_applied": boolean_series(
                enriched, "optimizer_repair_applied"
            ),
            "review_status": enriched["review_status"].astype(str),
            "branch_multiplicity": pd.to_numeric(
                enriched["branch_multiplicity"], errors="raise"
            ).astype(int),
            "interpolated": boolean_series(enriched, "interpolated"),
            "accepted_config": CONFIG_REL,
            "accepted_config_sha256": config_sha256,
            "source_enriched_ledger": ENRICHED_REL,
            "source_enriched_ledger_sha256": enriched_sha256,
            "limit_scope": "individual_observed_only",
            "bands_included": False,
        }
    )
    if individual["sample_label"].isna().any():
        fail("individual table has an unmapped dataset label")
    if boolean_series(individual, "bands_included").any():
        fail("individual table must not contain bands")
    if boolean_series(individual, "interpolated").any():
        fail("individual table contains interpolated rows")
    for column in [
        "eps2_up",
        "eps2_observed_ee_channel",
        "minimal_visible_factor",
        "BR_ee_minimal",
        "eps2_observed_minimal_visible",
        "p0_analytic",
        "Z_analytic",
    ]:
        values = pd.to_numeric(individual[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            fail(f"individual table contains non-finite {column}")
    if not np.allclose(
        individual["eps2_observed_minimal_visible"],
        individual["eps2_observed_ee_channel"] * individual["minimal_visible_factor"],
        rtol=2.0e-15,
        atol=0.0,
    ):
        fail("minimal-visible epsilon-squared transformation is inconsistent")
    if not np.allclose(
        individual["BR_ee_minimal"] * individual["minimal_visible_factor"],
        1.0,
        rtol=2.0e-15,
        atol=2.0e-15,
    ):
        fail("minimal-visible branching fraction is inconsistent")
    return individual


def main() -> None:
    frozen_hashes = verify_frozen_inputs()
    detailed = pd.read_csv(REPO / DETAILED_REL)
    compact = pd.read_csv(REPO / COMPACT_REL)
    factor_grid = pd.read_csv(REPO / FACTOR_GRID_REL)

    missing_detailed = sorted(DETAILED_REQUIRED_COLUMNS - set(detailed.columns))
    missing_compact = sorted(COMPACT_REQUIRED_COLUMNS - set(compact.columns))
    if missing_detailed:
        fail(f"detailed v4 ledger is missing columns: {missing_detailed}")
    if missing_compact:
        fail(f"compact v4.1 ledger is missing columns: {missing_compact}")

    detailed = add_and_validate_keys(detailed, "detailed v4 ledger")
    compact = add_and_validate_keys(compact, "accepted compact v4.1 ledger")
    validate_expected_grid(detailed, "detailed v4 ledger")
    validate_expected_grid(compact, "accepted compact v4.1 ledger")
    detailed_keys = set(map(tuple, detailed[KEY_COLUMNS].to_numpy()))
    compact_keys = set(map(tuple, compact[KEY_COLUMNS].to_numpy()))
    if detailed_keys != compact_keys:
        fail("detailed and compact ledgers do not have identical dataset/mass keys")
    if boolean_series(compact, "interpolated").any():
        fail("accepted compact v4.1 ledger contains interpolation")
    if boolean_series(compact, "repair_reproduction_pending").any():
        fail("accepted compact v4.1 ledger contains pending repair reproduction")

    source_cache: dict[str, pd.DataFrame] = {}
    selected_rows, selected_hashes = selected_sources_checked(compact, source_cache)
    assembled = update_from_selected_rows(detailed, compact, selected_rows)
    assembled = rebuild_attempt_provenance(assembled, source_cache)
    assembled = refresh_2016_bound_flags(assembled)

    # Make the origin of preserved metadata and the accepted card explicit on
    # every row so downstream figures can remain self-describing.
    assembled["geometry_density_source"] = DETAILED_REL
    assembled["geometry_density_source_sha256"] = frozen_hashes[DETAILED_REL]
    assembled["accepted_compact_ledger"] = COMPACT_REL
    assembled["accepted_compact_ledger_sha256"] = frozen_hashes[COMPACT_REL]
    assembled["accepted_config"] = CONFIG_REL
    assembled["accepted_config_sha256"] = frozen_hashes[CONFIG_REL]

    dataset_order = {2015: 0, 2016: 1, 2021: 2}
    assembled["_dataset_order"] = assembled["dataset"].map(dataset_order)
    if assembled["_dataset_order"].isna().any():
        fail("unknown dataset while sorting enriched ledger")
    assembled = (
        assembled.sort_values(["_dataset_order", "mass_MeV"], kind="mergesort")
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )

    assembled_summary = validate_assembled(assembled, detailed)
    factor12_summary = validate_factor12(assembled, compact, factor_grid)

    atomic_csv(assembled, ENRICHED_PATH)
    enriched_hash = sha256_file(ENRICHED_PATH)

    individual = build_individual(
        assembled,
        enriched_sha256=enriched_hash,
        config_sha256=frozen_hashes[CONFIG_REL],
    )
    atomic_csv(individual, INDIVIDUAL_PATH)
    individual_hash = sha256_file(INDIVIDUAL_PATH)

    corrected = boolean_series(individual, "dimuon_correction_applied")
    corrected_masses = individual.loc[corrected, "mass_MeV"].astype(int)
    summary = {
        "schema_version": "hps-gpr-v4.2-individual-ledger-validation-1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "combined_bands_only": True,
            "individual_limits": "observed_only",
            "individual_bands_included": False,
            "interpolation_permitted": False,
            "minimal_visible_conversion": (
                "eps2_observed_minimal_visible = eps2_up * "
                "(1 + Gamma_mumu/Gamma_ee), with only e+e- and mu+mu- "
                "visible channels and unit lepton universality"
            ),
        },
        "frozen_inputs": {
            relative: {"sha256": digest}
            for relative, digest in sorted(frozen_hashes.items())
        },
        "selected_sources": {
            "unique_file_count": len(selected_hashes),
            "all_hashes_verified": True,
            "files": [
                {"path": relative, "sha256": digest}
                for relative, digest in sorted(selected_hashes.items())
            ],
        },
        "enriched_ledger_validation": assembled_summary,
        "accepted_2016_factor12_validation": factor12_summary,
        "minimal_visible_dimuon": {
            "muon_mass_GeV": MINIMAL_VISIBLE_MUON_MASS_GEV,
            "threshold_GeV": 2.0 * MINIMAL_VISIBLE_MUON_MASS_GEV,
            "corrected_row_count": int(corrected.sum()),
            "first_corrected_mass_MeV": (
                int(corrected_masses.min()) if not corrected_masses.empty else None
            ),
        },
        "outputs": {
            "enriched": {
                "path": ENRICHED_REL,
                "sha256": enriched_hash,
                "rows": int(len(assembled)),
                "columns": list(assembled.columns),
            },
            "individual": {
                "path": INDIVIDUAL_REL,
                "sha256": individual_hash,
                "rows": int(len(individual)),
                "columns": list(individual.columns),
            },
        },
        "builder": {
            "path": repo_relative(SCRIPT),
            "sha256": sha256_file(SCRIPT),
        },
    }
    atomic_json(summary, SUMMARY_PATH)

    print(
        json.dumps(
            {
                "status": "pass",
                "enriched": {
                    "path": ENRICHED_REL,
                    "rows": len(assembled),
                    "sha256": enriched_hash,
                },
                "individual": {
                    "path": INDIVIDUAL_REL,
                    "rows": len(individual),
                    "sha256": individual_hash,
                },
                "summary": SUMMARY_REL,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
