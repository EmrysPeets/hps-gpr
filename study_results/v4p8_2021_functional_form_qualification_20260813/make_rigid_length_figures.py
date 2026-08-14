#!/usr/bin/env python3
"""Render the final v4.8 conditional-stress length-optimizer diagnostics.

Run this only after the immutable length scanner has completed all 80 active
scenario-toy tasks and

    python3 run_rigid_length_scan.py collect

has written the complete ``derived/rigid_length_scan`` product set.  Inputs
are checked against the collection hashes and the full expected state lattice
before any output directory or figure is created.

The figures are background-only, pull-blind optimizer diagnostics.  They show
selected reproducible length trajectories, boundary occupancy, nested LML,
and changes in the two gate coordinates (length and kernel constant).  They do
not compute or display a signal amplitude, pull, CLs quantity, limit, coverage
statement, or factor/card selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived/rigid_length_scan"
SCAN_CONTRACT = HERE / "qa/rigid_length_scan/scan_contract.json"
OUTPUT = Path(
    "/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/"
    "output/pdf/v4p8_2021_rigid_threshold_truth_20260813"
)

SCENARIOS = (
    "2021_1pct_x10",
    "2021_10pct",
    "2021_1pct_x100",
    "2021_10pct_x10",
)
LABELS = {
    "2021_1pct_x10": r"1% source $\times 10$",
    "2021_10pct": "native 10% (1% shape frozen)",
    "2021_1pct_x100": r"1% source $\times 100$",
    "2021_10pct_x10": r"native 10% $\times 10$ (1% shape frozen)",
}
UPPER_FACTORS = (15, 20, 25)
TRANSITIONS = ((15, 20), (20, 25))
MASS_GRID_MEV = tuple(range(50, 251, 20))
TOY_INDICES = tuple(range(20))
RESERVED_TOY_INDICES = tuple(range(20, 25))
SUPPORT_GEV = (0.04, 0.30)
EXACT_BOUND_RATIO = 0.999
NEAR_BOUND_WINDOW = 0.02
STRICT_LML_TOLERANCE = 1.0e-4
MATERIAL_LML_PER_TRAIN_TOLERANCE = 1.0e-3

PRODUCT_FILES = (
    "optimizer_attempt_ledger.csv",
    "raw_ell_sigma_x_trajectories.csv",
    "optimizer_exclusion_ledger.csv",
    "bound_occupancy_by_scenario_factor.csv",
    "bound_occupancy_by_scenario_factor_mass.csv",
    "nested_lml_pointwise.csv",
    "nested_lml_summary.csv",
    "task_product_audit.csv",
)
REQUIRED_FILES = ("collection_summary.json", *PRODUCT_FILES)
FORBIDDEN_COLUMN_SUBSTRINGS = {
    "sigmaa",
    "amplitude",
    "ahat",
    "aup",
    "pull",
    "zhat",
    "recovery",
    "p0",
    "pvalue",
    "cls",
    "qmu",
    "eps2",
    "epsilon",
    "limit",
    "coverage",
    "signal_yield",
    "upper_limit",
}

TRAJECTORY_KEY = (
    "scenario",
    "background_toy_index",
    "mass_MeV",
    "upper_factor",
)
TASK_KEY = ("scenario", "background_toy_index")

TRAJECTORY_STEM = "v4p8_rigid_length_trajectories_20toy"
COMPANION_STEM = "v4p8_rigid_length_pullblind_optimizer_companion_20toy"


class FigureInputError(RuntimeError):
    """Raised when final collected inputs fail the frozen figure contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise FigureInputError(f"expected JSON object: {path}")
    return payload


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FigureInputError(f"{label} is missing columns: {missing}")


def forbid_inference_columns(frame: pd.DataFrame, label: str) -> None:
    violations: dict[str, list[str]] = {}
    for column in frame.columns:
        normalized = "".join(
            character for character in str(column).lower() if character.isalnum()
        )
        matches = sorted(
            token
            for token in FORBIDDEN_COLUMN_SUBSTRINGS
            if "".join(
                character for character in token.lower() if character.isalnum()
            )
            in normalized
        )
        if matches:
            violations[str(column)] = matches
    if violations:
        raise FigureInputError(
            f"{label} contains prohibited inference columns: {violations}"
        )


def numeric_values(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    *,
    positive: bool = False,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise FigureInputError(f"{label}.{column} contains nonfinite values")
        if positive and not np.all(values > 0.0):
            raise FigureInputError(f"{label}.{column} is not strictly positive")


def bool_values(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    found = set(normalized)
    if not found.issubset(allowed):
        raise FigureInputError(f"{label} contains non-boolean values: {sorted(found)}")
    return normalized.map({"true": True, "false": False}).astype(bool)


def assert_close(left: float, right: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol):
        raise FigureInputError(f"{label} mismatch: {left!r} versus {right!r}")


def expected_task_keys() -> set[tuple[str, int]]:
    return {(scenario, toy) for scenario in SCENARIOS for toy in TOY_INDICES}


def expected_state_keys() -> set[tuple[str, int, int, int]]:
    return {
        (scenario, toy, mass, factor)
        for scenario in SCENARIOS
        for toy in TOY_INDICES
        for mass in MASS_GRID_MEV
        for factor in UPPER_FACTORS
    }


def frame_keys(
    frame: pd.DataFrame, columns: Sequence[str]
) -> set[tuple[Any, ...]]:
    return set(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def validate_summary_and_hashes() -> dict[str, Any]:
    missing = [str(DERIVED / name) for name in REQUIRED_FILES if not (DERIVED / name).is_file()]
    if not SCAN_CONTRACT.is_file():
        missing.append(str(SCAN_CONTRACT))
    if missing:
        raise FigureInputError(
            "final rigid length collection is absent or incomplete; run the "
            f"80-task scan and collect first. Missing: {missing}"
        )

    summary = load_json(DERIVED / "collection_summary.json")
    exact_contract = {
        "status": "complete",
        "current_tasks": 80,
        "missing_or_stale_tasks": 0,
        "background_only": True,
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "reserve_toys_consumed": False,
        "optimizer_gate": "reduced_length_only_pull_blind_v1",
        "pulls_produced": False,
        "cls_produced": False,
        "factor_selection_performed": False,
    }
    for key, expected in exact_contract.items():
        if summary.get(key) != expected:
            raise FigureInputError(
                f"collection_summary.json {key}={summary.get(key)!r}; "
                f"expected {expected!r}"
            )
    support = [float(value) for value in summary.get("support_gev", [])]
    if len(support) != 2 or not np.allclose(support, SUPPORT_GEV, rtol=0.0, atol=1e-12):
        raise FigureInputError("collection support is not the frozen 40-300 MeV support")

    contract = load_json(SCAN_CONTRACT)
    expected_contract_fields = {
        "background_only": True,
        "scenarios": [
            "2021_1pct_x10",
            "2021_1pct_x100",
            "2021_10pct",
            "2021_10pct_x10",
        ],
        "active_toy_indices": list(TOY_INDICES),
        "reserved_toy_indices": list(RESERVED_TOY_INDICES),
        "upper_factors": list(UPPER_FACTORS),
        "factor_selection_performed": False,
    }
    for key, expected in expected_contract_fields.items():
        if contract.get(key) != expected:
            raise FigureInputError(
                f"scan contract {key}={contract.get(key)!r}; expected {expected!r}"
            )
    gate = contract.get("optimizer_gate", {})
    if not isinstance(gate, Mapping):
        raise FigureInputError("scan contract optimizer_gate is not an object")
    if gate.get("name") != "reduced_length_only_pull_blind_v1":
        raise FigureInputError("scan contract does not use the reduced pull-blind gate")
    assert_close(
        gate.get("bound_ratio_window", float("nan")),
        NEAR_BOUND_WINDOW,
        "scan-contract near-bound window",
    )
    assert_close(
        gate.get("delta_lml_per_train_max", float("nan")),
        MATERIAL_LML_PER_TRAIN_TOLERANCE,
        "scan-contract LML/train tolerance",
    )
    if contract.get("seed_excludes_upper_factor") is not True:
        raise FigureInputError("scan contract does not declare common factor seeds")
    masses = np.rint(np.asarray(contract.get("masses_gev", []), dtype=float) * 1000.0).astype(int)
    if tuple(masses) != MASS_GRID_MEV:
        raise FigureInputError("scan contract mass grid is not 50:20:250 MeV")
    if canonical_json_hash(contract) != str(summary.get("scan_contract_sha256", "")):
        raise FigureInputError("collection scan-contract hash does not match QA contract")

    declared_hashes = summary.get("derived_sha256", {})
    if not isinstance(declared_hashes, Mapping):
        raise FigureInputError("collection has no derived_sha256 mapping")
    for name in PRODUCT_FILES:
        expected = str(declared_hashes.get(name, ""))
        actual = sha256_file(DERIVED / name)
        if not expected or expected != actual:
            raise FigureInputError(
                f"collection hash mismatch or missing declaration for {name}"
            )
    return summary


def load_frames() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    summary = validate_summary_and_hashes()
    frames = {name: pd.read_csv(DERIVED / name) for name in PRODUCT_FILES}
    for name, frame in frames.items():
        forbid_inference_columns(frame, name)
    return summary, frames


def validate_task_audit(frame: pd.DataFrame) -> None:
    label = "task product audit"
    require_columns(frame, (*TASK_KEY, "current", "status"), label)
    if len(frame) != 80 or frame.duplicated(list(TASK_KEY)).any():
        raise FigureInputError("task audit is not 80 unique scenario-toy tasks")
    current = bool_values(frame["current"], f"{label}.current")
    if not bool(current.all()):
        raise FigureInputError("task audit contains missing or stale tasks")
    keys = {
        (str(row.scenario), int(row.background_toy_index))
        for row in frame.itertuples(index=False)
    }
    if keys != expected_task_keys():
        raise FigureInputError("task audit scenario-toy lattice mismatch")
    if set(frame["status"].astype(str)) != {"current"}:
        raise FigureInputError("task audit contains a non-current status")


def validate_attempts(frame: pd.DataFrame) -> None:
    label = "optimizer attempt ledger"
    require_columns(
        frame,
        (*TRAJECTORY_KEY, "attempt", "optimizer_seed", "background_only"),
        label,
    )
    if frame.empty:
        raise FigureInputError("optimizer attempt ledger is empty")
    if frame.duplicated([*TRAJECTORY_KEY, "attempt"]).any():
        raise FigureInputError("optimizer attempt ledger has duplicate state-attempt rows")
    if not bool(bool_values(frame["background_only"], f"{label}.background_only").all()):
        raise FigureInputError("optimizer attempt ledger is not background-only")
    toys = set(pd.to_numeric(frame["background_toy_index"]).astype(int))
    if not toys.issubset(set(TOY_INDICES)):
        raise FigureInputError("optimizer attempt ledger consumed a reserved/unknown toy")
    state_counts = frame.groupby(list(TRAJECTORY_KEY), sort=False).size()
    if set(state_counts.index) != expected_state_keys():
        raise FigureInputError("optimizer attempt ledger state lattice is incomplete")
    if not set(state_counts.unique()).issubset({3, 5}):
        raise FigureInputError("optimizer state attempt counts are not 3 or 5")
    paired_counts = state_counts.unstack("upper_factor")
    if bool((paired_counts.nunique(axis=1) != 1).any()):
        raise FigureInputError("attempt counts are not paired across factors")
    seed_groups = frame.groupby(
        ["scenario", "background_toy_index", "mass_MeV", "attempt"], sort=False
    )
    if bool((seed_groups["optimizer_seed"].nunique() != 1).any()):
        raise FigureInputError("upper factors do not share optimizer seeds")
    if bool((seed_groups["upper_factor"].nunique() != 3).any()):
        raise FigureInputError("paired factor attempt sets are incomplete")


def normalize_trajectory_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    require_columns(result, TRAJECTORY_KEY, label)
    result["scenario"] = result["scenario"].astype(str)
    for column in ("background_toy_index", "mass_MeV", "upper_factor"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.allclose(values, np.rint(values)):
            raise FigureInputError(f"{label}.{column} is not integral")
        result[column] = np.rint(values).astype(int)
    return result


def validate_selected_and_excluded(
    selected: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = normalize_trajectory_keys(selected, "selected trajectories")
    exclusions = normalize_trajectory_keys(exclusions, "optimizer exclusions")
    require_columns(
        selected,
        (
            "mass_GeV",
            "gp_lml",
            "ell_opt",
            "ell_hi",
            "sigma_x",
            "ell_over_sigma_x",
            "ell_hi_over_sigma_x",
            "ell_over_ell_hi",
            "kernel_constant_opt",
            "n_train",
            "ell_at_upper_exact",
            "ell_near_upper",
            "ell_at_lower_exact",
            "ell_near_lower",
            "background_only",
            "selected_maximum_lml_reproduced_branch",
            "support_preserved_40_300",
            "common_seeds_across_factors",
            "factor_selection_performed",
        ),
        "selected trajectories",
    )
    if selected.empty:
        raise FigureInputError("selected trajectory ledger is empty")
    if selected.duplicated(list(TRAJECTORY_KEY)).any():
        raise FigureInputError("selected trajectory ledger has duplicate states")
    if exclusions.duplicated(list(TRAJECTORY_KEY)).any():
        raise FigureInputError("optimizer exclusion ledger has duplicate states")

    selected_keys = frame_keys(selected, TRAJECTORY_KEY)
    excluded_keys = frame_keys(exclusions, TRAJECTORY_KEY)
    expected = expected_state_keys()
    if selected_keys & excluded_keys:
        raise FigureInputError("a state appears in both selected and exclusion ledgers")
    if selected_keys | excluded_keys != expected:
        missing = expected - (selected_keys | excluded_keys)
        extra = (selected_keys | excluded_keys) - expected
        raise FigureInputError(
            f"selected-plus-excluded state lattice mismatch: missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    numeric_values(
        selected,
        (
            "mass_GeV",
            "gp_lml",
            "ell_opt",
            "ell_hi",
            "sigma_x",
            "ell_over_sigma_x",
            "ell_hi_over_sigma_x",
            "ell_over_ell_hi",
            "kernel_constant_opt",
            "n_train",
        ),
        "selected trajectories",
    )
    numeric_values(
        selected,
        (
            "ell_opt",
            "ell_hi",
            "sigma_x",
            "ell_over_sigma_x",
            "kernel_constant_opt",
            "n_train",
        ),
        "selected trajectories",
        positive=True,
    )
    if not np.allclose(
        pd.to_numeric(selected["mass_GeV"]) * 1000.0,
        pd.to_numeric(selected["mass_MeV"]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise FigureInputError("selected mass_GeV and mass_MeV differ")
    if not np.allclose(
        pd.to_numeric(selected["ell_over_sigma_x"]),
        pd.to_numeric(selected["ell_opt"]) / pd.to_numeric(selected["sigma_x"]),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise FigureInputError("selected ell/sigma_x is inconsistent")
    if not np.allclose(
        pd.to_numeric(selected["ell_over_ell_hi"]),
        pd.to_numeric(selected["ell_opt"]) / pd.to_numeric(selected["ell_hi"]),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise FigureInputError("selected ell/ell_hi is inconsistent")
    if not np.allclose(
        pd.to_numeric(selected["ell_hi_over_sigma_x"]),
        pd.to_numeric(selected["upper_factor"]),
        rtol=0.0,
        atol=1e-8,
    ):
        raise FigureInputError("selected upper factor was not applied exactly")

    exact = bool_values(
        selected["ell_at_upper_exact"], "selected.ell_at_upper_exact"
    )
    near = bool_values(selected["ell_near_upper"], "selected.ell_near_upper")
    lower_exact = bool_values(
        selected["ell_at_lower_exact"], "selected.ell_at_lower_exact"
    )
    lower_near = bool_values(selected["ell_near_lower"], "selected.ell_near_lower")
    ratio = pd.to_numeric(selected["ell_over_ell_hi"]).to_numpy(float)
    if not np.array_equal(exact.to_numpy(), ratio >= EXACT_BOUND_RATIO):
        raise FigureInputError("exact upper-bound flags do not match ell/ell_hi")
    if not np.array_equal(near.to_numpy(), ratio >= 1.0 - NEAR_BOUND_WINDOW):
        raise FigureInputError("near upper-bound flags do not match ell/ell_hi")
    if bool((exact & ~near).any()):
        raise FigureInputError("an exact upper contact is not marked near-upper")
    selected["ell_at_upper_exact"] = exact
    selected["ell_near_upper"] = near
    selected["ell_at_lower_exact"] = lower_exact
    selected["ell_near_lower"] = lower_near

    required_true = (
        "background_only",
        "selected_maximum_lml_reproduced_branch",
        "support_preserved_40_300",
        "common_seeds_across_factors",
    )
    for column in required_true:
        if not bool(bool_values(selected[column], f"selected.{column}").all()):
            raise FigureInputError(f"selected.{column} is not true for every row")
    if bool(bool_values(selected["factor_selection_performed"], "selected.factor_selection").any()):
        raise FigureInputError("selected ledger claims factor selection was performed")
    return selected, exclusions


def occupancy_from_selected(
    selected: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for values, group in selected.groupby(list(group_columns), sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        record = dict(zip(group_columns, values))
        n = len(group)
        record.update(
            {
                "selected_rows": n,
                "exact_upper_bound_rows": int(group["ell_at_upper_exact"].sum()),
                "exact_upper_bound_fraction": float(group["ell_at_upper_exact"].mean()),
                "near_upper_bound_rows": int(group["ell_near_upper"].sum()),
                "near_upper_bound_fraction": float(group["ell_near_upper"].mean()),
                "exact_lower_bound_rows": int(group["ell_at_lower_exact"].sum()),
                "exact_lower_bound_fraction": float(group["ell_at_lower_exact"].mean()),
                "near_lower_bound_rows": int(group["ell_near_lower"].sum()),
                "near_lower_bound_fraction": float(group["ell_near_lower"].mean()),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def validate_occupancy(
    selected: pd.DataFrame,
    official: pd.DataFrame,
    group_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    official = normalize_trajectory_keys(
        official.assign(
            background_toy_index=0,
            mass_MeV=(
                official["mass_MeV"] if "mass_MeV" in official else MASS_GRID_MEV[0]
            ),
        ),
        label,
    ).drop(columns=["background_toy_index"])
    if "mass_MeV" not in group_columns:
        official = official.drop(columns=["mass_MeV"])
    count_columns = (
        "selected_rows",
        "exact_upper_bound_rows",
        "near_upper_bound_rows",
        "exact_lower_bound_rows",
        "near_lower_bound_rows",
    )
    fraction_columns = (
        "exact_upper_bound_fraction",
        "near_upper_bound_fraction",
        "exact_lower_bound_fraction",
        "near_lower_bound_fraction",
    )
    require_columns(official, (*group_columns, *count_columns, *fraction_columns), label)
    if official.duplicated(list(group_columns)).any():
        raise FigureInputError(f"{label} has duplicate groups")
    if tuple(group_columns) == ("scenario", "upper_factor"):
        expected_groups = {
            (scenario, factor) for scenario in SCENARIOS for factor in UPPER_FACTORS
        }
    elif tuple(group_columns) == ("scenario", "upper_factor", "mass_MeV"):
        expected_groups = {
            (scenario, factor, mass)
            for scenario in SCENARIOS
            for factor in UPPER_FACTORS
            for mass in MASS_GRID_MEV
        }
    else:
        raise FigureInputError(f"unsupported occupancy grouping: {group_columns}")
    if frame_keys(official, group_columns) != expected_groups:
        raise FigureInputError(f"{label} does not contain the complete group lattice")
    recomputed = occupancy_from_selected(selected, group_columns)
    merged = recomputed.merge(
        official,
        on=list(group_columns),
        how="outer",
        suffixes=("_calc", "_file"),
        indicator=True,
    )
    if set(merged["_merge"]) != {"both"}:
        raise FigureInputError(f"{label} group inventory mismatch")
    for column in count_columns:
        if not np.array_equal(
            pd.to_numeric(merged[f"{column}_calc"]).astype(int),
            pd.to_numeric(merged[f"{column}_file"]).astype(int),
        ):
            raise FigureInputError(f"{label}.{column} differs from selected ledger")
    for column in fraction_columns:
        if not np.allclose(
            pd.to_numeric(merged[f"{column}_calc"]),
            pd.to_numeric(merged[f"{column}_file"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise FigureInputError(f"{label}.{column} differs from selected ledger")
    return official.sort_values(list(group_columns)).reset_index(drop=True)


def validate_nested(
    selected: pd.DataFrame,
    nested: pd.DataFrame,
    nested_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = "nested LML pointwise ledger"
    require_columns(
        nested,
        (
            "scenario",
            "background_toy_index",
            "mass_MeV",
            "lower_factor",
            "upper_factor",
            "comparable",
            "n_train",
            "lower_lml",
            "upper_lml",
            "delta_lml_upper_minus_lower",
            "delta_lml_per_train",
            "strict_nested_order_violation",
            "material_nested_order_violation",
            "same_input_geometry",
        ),
        label,
    )
    key_columns = (
        "scenario",
        "background_toy_index",
        "mass_MeV",
        "lower_factor",
        "upper_factor",
    )
    if len(nested) != len(SCENARIOS) * len(TOY_INDICES) * len(MASS_GRID_MEV) * 2:
        raise FigureInputError("nested LML ledger is not the complete 4x20x11x2 lattice")
    if nested.duplicated(list(key_columns)).any():
        raise FigureInputError("nested LML ledger contains duplicate comparisons")
    found_keys = {
        (
            str(row.scenario),
            int(row.background_toy_index),
            int(row.mass_MeV),
            int(row.lower_factor),
            int(row.upper_factor),
        )
        for row in nested.itertuples(index=False)
    }
    expected_keys = {
        (scenario, toy, mass, lower, upper)
        for scenario in SCENARIOS
        for toy in TOY_INDICES
        for mass in MASS_GRID_MEV
        for lower, upper in TRANSITIONS
    }
    if found_keys != expected_keys:
        raise FigureInputError("nested LML comparison lattice mismatch")

    selected_lookup = {
        tuple(row[column] for column in TRAJECTORY_KEY): row
        for row in selected.to_dict(orient="records")
    }
    comparable_flags = bool_values(nested["comparable"], f"{label}.comparable")
    strict_flags = bool_values(
        nested["strict_nested_order_violation"].fillna(False),
        f"{label}.strict_violation",
    )
    material_flags = bool_values(
        nested["material_nested_order_violation"].fillna(False),
        f"{label}.material_violation",
    )
    same_geometry = bool_values(
        nested["same_input_geometry"].fillna(False), f"{label}.same_input_geometry"
    )
    nested = nested.copy()
    nested["comparable"] = comparable_flags
    nested["strict_nested_order_violation"] = strict_flags
    nested["material_nested_order_violation"] = material_flags
    nested["same_input_geometry"] = same_geometry

    for index, row in nested.iterrows():
        base = (
            str(row["scenario"]),
            int(row["background_toy_index"]),
            int(row["mass_MeV"]),
        )
        lower = int(row["lower_factor"])
        upper = int(row["upper_factor"])
        lower_row = selected_lookup.get((*base, lower))
        upper_row = selected_lookup.get((*base, upper))
        expected_comparable = lower_row is not None and upper_row is not None
        if bool(row["comparable"]) != expected_comparable:
            raise FigureInputError(f"nested comparability mismatch at {(*base, lower, upper)}")
        if not expected_comparable:
            continue
        n_train = int(lower_row["n_train"])
        if n_train != int(upper_row["n_train"]):
            raise FigureInputError(f"nested training-count mismatch at {base}")
        lower_lml = float(lower_row["gp_lml"])
        upper_lml = float(upper_row["gp_lml"])
        delta = upper_lml - lower_lml
        assert_close(row["n_train"], n_train, f"nested n_train {base}")
        assert_close(row["lower_lml"], lower_lml, f"nested lower LML {base}")
        assert_close(row["upper_lml"], upper_lml, f"nested upper LML {base}")
        assert_close(row["delta_lml_upper_minus_lower"], delta, f"nested delta LML {base}")
        assert_close(row["delta_lml_per_train"], delta / n_train, f"nested delta/train {base}")
        if not bool(row["same_input_geometry"]):
            raise FigureInputError(f"nested inputs differ at {base}")
        if bool(row["strict_nested_order_violation"]) != (
            delta < -STRICT_LML_TOLERANCE
        ):
            raise FigureInputError(f"strict nested-LML flag mismatch at {base}")
        if bool(row["material_nested_order_violation"]) != (
            delta / n_train < -MATERIAL_LML_PER_TRAIN_TOLERANCE
        ):
            raise FigureInputError(f"material nested-LML flag mismatch at {base}")

    summary_label = "nested LML summary"
    summary_columns = (
        "scenario",
        "lower_factor",
        "upper_factor",
        "rows",
        "comparable_rows",
        "unavailable_rows",
        "delta_lml_min",
        "delta_lml_median",
        "delta_lml_max",
        "strict_nested_order_violations",
        "material_nested_order_violations",
    )
    require_columns(nested_summary, summary_columns, summary_label)
    if len(nested_summary) != len(SCENARIOS) * len(TRANSITIONS):
        raise FigureInputError("nested LML summary does not contain eight rows")
    if nested_summary.duplicated(["scenario", "lower_factor", "upper_factor"]).any():
        raise FigureInputError("nested LML summary has duplicate transition rows")
    for row in nested_summary.itertuples(index=False):
        group = nested[
            (nested["scenario"].astype(str) == str(row.scenario))
            & (pd.to_numeric(nested["lower_factor"]).astype(int) == int(row.lower_factor))
            & (pd.to_numeric(nested["upper_factor"]).astype(int) == int(row.upper_factor))
        ]
        comparable = group[group["comparable"]]
        deltas = pd.to_numeric(comparable["delta_lml_upper_minus_lower"], errors="coerce")
        if len(group) != int(row.rows) or len(comparable) != int(row.comparable_rows):
            raise FigureInputError("nested LML summary row counts do not reproduce")
        if len(group) - len(comparable) != int(row.unavailable_rows):
            raise FigureInputError("nested LML unavailable count does not reproduce")
        for column, value in (
            ("delta_lml_min", float(deltas.min())),
            ("delta_lml_median", float(deltas.median())),
            ("delta_lml_max", float(deltas.max())),
        ):
            assert_close(getattr(row, column), value, f"nested summary {column}")
        if int(row.strict_nested_order_violations) != int(
            comparable["strict_nested_order_violation"].sum()
        ):
            raise FigureInputError("strict nested-LML summary count does not reproduce")
        if int(row.material_nested_order_violations) != int(
            comparable["material_nested_order_violation"].sum()
        ):
            raise FigureInputError("material nested-LML summary count does not reproduce")
    return nested, nested_summary


def validate_collection() -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    summary, frames = load_frames()
    validate_task_audit(frames["task_product_audit.csv"])
    validate_attempts(frames["optimizer_attempt_ledger.csv"])
    selected, exclusions = validate_selected_and_excluded(
        frames["raw_ell_sigma_x_trajectories.csv"],
        frames["optimizer_exclusion_ledger.csv"],
    )
    occupancy = validate_occupancy(
        selected,
        frames["bound_occupancy_by_scenario_factor.csv"],
        ("scenario", "upper_factor"),
        "scenario-factor occupancy",
    )
    validate_occupancy(
        selected,
        frames["bound_occupancy_by_scenario_factor_mass.csv"],
        ("scenario", "upper_factor", "mass_MeV"),
        "scenario-factor-mass occupancy",
    )
    nested, nested_summary = validate_nested(
        selected,
        frames["nested_lml_pointwise.csv"],
        frames["nested_lml_summary.csv"],
    )
    if int(summary.get("attempt_rows", -1)) != len(frames["optimizer_attempt_ledger.csv"]):
        raise FigureInputError("collection attempt-row count mismatch")
    if int(summary.get("selected_trajectory_rows", -1)) != len(selected):
        raise FigureInputError("collection selected-row count mismatch")
    if int(summary.get("optimizer_exclusion_rows", -1)) != len(exclusions):
        raise FigureInputError("collection exclusion-row count mismatch")
    if int(summary.get("strict_nested_lml_violations", -1)) != int(
        nested_summary["strict_nested_order_violations"].sum()
    ):
        raise FigureInputError("collection strict nested-LML count mismatch")
    if int(summary.get("material_nested_lml_violations", -1)) != int(
        nested_summary["material_nested_order_violations"].sum()
    ):
        raise FigureInputError("collection material nested-LML count mismatch")
    return summary, selected, exclusions, occupancy, nested


def paired_coordinates(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    index = {
        tuple(row[column] for column in TRAJECTORY_KEY): row
        for row in selected.to_dict(orient="records")
    }
    for scenario in SCENARIOS:
        for toy in TOY_INDICES:
            for mass in MASS_GRID_MEV:
                for lower, upper in TRANSITIONS:
                    low = index.get((scenario, toy, mass, lower))
                    high = index.get((scenario, toy, mass, upper))
                    if low is None or high is None:
                        continue
                    records.append(
                        {
                            "scenario": scenario,
                            "background_toy_index": toy,
                            "mass_MeV": mass,
                            "lower_factor": lower,
                            "upper_factor": upper,
                            "abs_log_ell_ratio": abs(
                                math.log(float(high["ell_opt"]) / float(low["ell_opt"]))
                            ),
                            "abs_log_constant_ratio": abs(
                                math.log(
                                    float(high["kernel_constant_opt"])
                                    / float(low["kernel_constant_opt"])
                                )
                            ),
                        }
                    )
    frame = pd.DataFrame(records)
    numeric_values(
        frame,
        ("abs_log_ell_ratio", "abs_log_constant_ratio"),
        "paired coordinate shifts",
    )
    return frame


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "figure.titlesize": 13.0,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def atomic_savefig(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    try:
        figure.savefig(temporary, **kwargs)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def trajectory_figure(
    selected: pd.DataFrame,
    exclusions: pd.DataFrame,
    factor: int,
) -> plt.Figure:
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(TOY_INDICES)))
    figure, axes = plt.subplots(4, 1, figsize=(11.5, 11.2), sharex=True, sharey=True)
    factor_rows = selected[pd.to_numeric(selected["upper_factor"]).astype(int) == factor]
    factor_exclusions = exclusions[
        pd.to_numeric(exclusions["upper_factor"]).astype(int) == factor
    ]
    for axis, scenario in zip(axes, SCENARIOS):
        lane = factor_rows[factor_rows["scenario"].astype(str) == scenario]
        excluded_lane = factor_exclusions[
            factor_exclusions["scenario"].astype(str) == scenario
        ]
        for toy, color in zip(TOY_INDICES, colors):
            toy_rows = lane[
                pd.to_numeric(lane["background_toy_index"]).astype(int) == toy
            ].set_index("mass_MeV")
            trajectory = toy_rows.reindex(MASS_GRID_MEV)
            values = pd.to_numeric(
                trajectory["ell_over_sigma_x"], errors="coerce"
            ).to_numpy(float)
            axis.plot(
                MASS_GRID_MEV,
                values,
                color=color,
                marker="o",
                markersize=2.2,
                linewidth=0.9,
                alpha=0.58,
                zorder=2,
            )
            present = trajectory["ell_over_sigma_x"].notna()
            exact = trajectory["ell_at_upper_exact"].fillna(False).astype(bool)
            near_only = (
                trajectory["ell_near_upper"].fillna(False).astype(bool) & ~exact
            )
            if bool(exact.any()):
                axis.scatter(
                    np.asarray(MASS_GRID_MEV)[exact.to_numpy()],
                    values[exact.to_numpy()],
                    marker="s",
                    s=23,
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=0.85,
                    zorder=5,
                )
            if bool(near_only.any()):
                axis.scatter(
                    np.asarray(MASS_GRID_MEV)[near_only.to_numpy()],
                    values[near_only.to_numpy()],
                    marker="^",
                    s=25,
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=0.85,
                    zorder=5,
                )
            if not bool(present.any()):
                raise FigureInputError(
                    f"factor {factor}, {scenario}, toy {toy} has no selected trajectory"
                )
        axis.axhline(
            factor,
            color="0.18",
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )
        exact_count = int(lane["ell_at_upper_exact"].sum())
        near_only_count = int((lane["ell_near_upper"] & ~lane["ell_at_upper_exact"]).sum())
        axis.text(
            0.995,
            0.95,
            f"selected {len(lane)}/220; exact {exact_count}; "
            f"near-only {near_only_count}; excluded {len(excluded_lane)}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="0.82", alpha=0.88, pad=2.0),
        )
        axis.set_ylabel(r"Selected $\ell/\sigma_x$")
        axis.set_title(LABELS[scenario], loc="left", fontweight="semibold")
        axis.set_xlim(MASS_GRID_MEV[0] - 4, MASS_GRID_MEV[-1] + 4)
        axis.set_ylim(0.0, factor * 1.055)
        axis.grid(axis="y", color="0.90", linewidth=0.55)
    axes[-1].set_xlabel("Test mass [MeV]")
    axes[-1].set_xticks(MASS_GRID_MEV)

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=f"toy {toy:02d}",
        )
        for toy, color in zip(TOY_INDICES, colors)
    ]
    handles.extend(
        [
            Line2D(
                [0], [0], color="0.18", linestyle="--", linewidth=1.1,
                label=f"factor-{factor} ceiling",
            ),
            Line2D(
                [0], [0], color="0.18", marker="s", markerfacecolor="none",
                linestyle="none", label=r"exact: $\ell/\ell_{\max}\geq0.999$",
            ),
            Line2D(
                [0], [0], color="0.18", marker="^", markerfacecolor="none",
                linestyle="none", label=r"near-only: $0.98\leq\ell/\ell_{\max}<0.999$",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.047),
        ncol=8,
        frameon=False,
        title="Raw background-toy index and boundary status",
        title_fontsize=8.0,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    figure.suptitle(
        f"2021 rigid threshold stress mean: raw selected length trajectories, upper factor {factor}\n"
        "20 background toys per lane; 50:20:250 MeV; reproducible selected branches only",
        y=0.992,
    )
    figure.text(
        0.5,
        0.012,
        "Background-only pull-blind optimizer diagnostic. Gaps denote excluded "
        "optimizer states. The tested ceiling is not a factor/card selection.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    figure.tight_layout(rect=(0.035, 0.125, 0.995, 0.945), h_pad=1.0)
    return figure


def descriptive_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise FigureInputError("cannot summarize an empty optimizer coordinate")
    low, median, high = np.quantile(array, [0.05, 0.50, 0.95])
    return float(low), float(median), float(high)


def deterministic_jitter(frame: pd.DataFrame, width: float = 0.13) -> np.ndarray:
    toy = pd.to_numeric(frame["background_toy_index"]).astype(int).to_numpy()
    mass = pd.to_numeric(frame["mass_MeV"]).astype(int).to_numpy()
    code = (toy * 37 + mass * 11) % 101
    return width * (2.0 * code / 100.0 - 1.0)


def companion_figure(
    occupancy: pd.DataFrame,
    nested: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> plt.Figure:
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#4D4D4D"
    figure, axes = plt.subplots(4, 3, figsize=(12.8, 12.8), sharex="col")

    for row_index, scenario in enumerate(SCENARIOS):
        lane_occ = occupancy[occupancy["scenario"].astype(str) == scenario].sort_values(
            "upper_factor"
        )
        axis = axes[row_index, 0]
        axis.plot(
            lane_occ["upper_factor"],
            lane_occ["near_upper_bound_fraction"],
            color=orange,
            marker="^",
            markersize=5,
            linewidth=1.2,
            linestyle="--",
            label="near upper (includes exact)",
        )
        axis.plot(
            lane_occ["upper_factor"],
            lane_occ["exact_upper_bound_fraction"],
            color=blue,
            marker="s",
            markersize=4.5,
            linewidth=1.3,
            label="exact upper",
        )
        for occ_row in lane_occ.itertuples(index=False):
            axis.text(
                float(occ_row.upper_factor),
                min(1.025, float(occ_row.near_upper_bound_fraction) + 0.035),
                f"n={int(occ_row.selected_rows)}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                color="0.35",
            )
        axis.set_ylim(-0.02, 1.08)
        axis.set_ylabel(f"{LABELS[scenario]}\nselected-state fraction")
        axis.set_xticks(UPPER_FACTORS)
        axis.grid(axis="y", color="0.90", linewidth=0.55)

        axis = axes[row_index, 1]
        for transition_index, (lower, upper) in enumerate(TRANSITIONS):
            rows = nested[
                (nested["scenario"].astype(str) == scenario)
                & (pd.to_numeric(nested["lower_factor"]).astype(int) == lower)
                & (pd.to_numeric(nested["upper_factor"]).astype(int) == upper)
                & nested["comparable"].astype(bool)
            ]
            values = pd.to_numeric(rows["delta_lml_per_train"]).to_numpy(float)
            x = transition_index + deterministic_jitter(rows)
            axis.scatter(x, values, s=7, color=gray, alpha=0.20, linewidths=0, rasterized=True)
            low, median, high = descriptive_interval(values)
            axis.errorbar(
                transition_index,
                median,
                yerr=[[median - low], [high - median]],
                color=blue,
                marker="o",
                markersize=4.5,
                capsize=3,
                linewidth=1.3,
                zorder=5,
            )
            axis.annotate(
                f"med {median:.2g}\n{len(values)}/220",
                (transition_index, median),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.6,
                color=blue,
            )
        axis.axhline(0.0, color="0.25", linestyle="--", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=1e-8, linscale=0.8)
        axis.set_ylabel(r"$\Delta\log\mathcal{L}/n_{\rm train}$")
        axis.set_xticks((0, 1), (r"$15\to20$", r"$20\to25$"))
        axis.grid(axis="y", color="0.92", linewidth=0.5)

        axis = axes[row_index, 2]
        for transition_index, (lower, upper) in enumerate(TRANSITIONS):
            rows = coordinates[
                (coordinates["scenario"].astype(str) == scenario)
                & (pd.to_numeric(coordinates["lower_factor"]).astype(int) == lower)
                & (pd.to_numeric(coordinates["upper_factor"]).astype(int) == upper)
            ]
            jitter = deterministic_jitter(rows, width=0.09)
            for metric, offset, color, marker in (
                ("abs_log_ell_ratio", -0.14, blue, "o"),
                ("abs_log_constant_ratio", 0.14, orange, "s"),
            ):
                values = pd.to_numeric(rows[metric]).to_numpy(float)
                axis.scatter(
                    transition_index + offset + jitter,
                    values,
                    s=7,
                    color=color,
                    alpha=0.16,
                    linewidths=0,
                    rasterized=True,
                )
                low, median, high = descriptive_interval(values)
                axis.errorbar(
                    transition_index + offset,
                    median,
                    yerr=[[median - low], [high - median]],
                    color=color,
                    marker=marker,
                    markersize=4.2,
                    capsize=2.5,
                    linewidth=1.15,
                    zorder=5,
                )
        axis.set_yscale("symlog", linthresh=1e-8, linscale=0.8)
        axis.set_ylabel("Absolute log coordinate shift")
        axis.set_xticks((0, 1), (r"$15\to20$", r"$20\to25$"))
        axis.grid(axis="y", color="0.92", linewidth=0.5)

    axes[0, 0].set_title("Upper-bound occupancy")
    axes[0, 1].set_title(r"Nested $\Delta\log\mathcal{L}/n_{\rm train}$ (central 90% span)")
    axes[0, 2].set_title("Coordinate changes (central 90% span)")
    axes[-1, 0].set_xlabel(r"Tested upper factor [$\sigma_x$]")
    axes[-1, 1].set_xlabel("Adjacent tested factors")
    axes[-1, 2].set_xlabel("Adjacent tested factors")

    legend_handles = [
        Line2D([0], [0], color=blue, marker="s", label="exact upper occupancy"),
        Line2D(
            [0], [0], color=orange, marker="^", linestyle="--",
            label="near upper occupancy (includes exact)",
        ),
        Line2D([0], [0], color=gray, marker="o", linestyle="none", alpha=0.35, label="raw comparable state"),
        Line2D([0], [0], color=blue, marker="o", linestyle="none", label=r"$|\log(\ell_U/\ell_L)|$ summary"),
        Line2D([0], [0], color=orange, marker="s", linestyle="none", label=r"$|\log(C_U/C_L)|$ summary"),
        Line2D([0], [0], color=blue, marker="o", label="median and central 90% descriptive span"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.042),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.55,
    )
    figure.suptitle(
        "2021 rigid threshold stress mean: pull-blind length-optimizer diagnostics\n"
        "20 backgrounds per lane; factors 15, 20, 25; 50:20:250 MeV",
        y=0.995,
    )
    figure.text(
        0.5,
        0.010,
        "Occupancy denominators are reproducible selected states. Coordinate spans "
        "are descriptive, not confidence intervals. No factor/card selection is performed.",
        ha="center",
        va="bottom",
        fontsize=8.1,
    )
    figure.tight_layout(rect=(0.035, 0.105, 0.995, 0.95), h_pad=1.15, w_pad=1.45)
    return figure


def render(
    selected: pd.DataFrame,
    exclusions: pd.DataFrame,
    occupancy: pd.DataFrame,
    nested: pd.DataFrame,
) -> list[Path]:
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    combined_path = OUTPUT / f"{TRAJECTORY_STEM}_all_factors.pdf"
    fd, temporary_pdf = tempfile.mkstemp(
        prefix=f".{combined_path.stem}.", suffix=".pdf", dir=OUTPUT
    )
    os.close(fd)
    try:
        with PdfPages(temporary_pdf) as pages:
            for factor in UPPER_FACTORS:
                figure = trajectory_figure(selected, exclusions, factor)
                factor_stem = f"{TRAJECTORY_STEM}_f{factor:02d}"
                pdf_path = OUTPUT / f"{factor_stem}.pdf"
                png_path = OUTPUT / f"{factor_stem}.png"
                pages.savefig(figure, bbox_inches="tight", facecolor="white")
                atomic_savefig(figure, pdf_path, bbox_inches="tight", facecolor="white")
                atomic_savefig(
                    figure,
                    png_path,
                    dpi=220,
                    bbox_inches="tight",
                    facecolor="white",
                )
                outputs.extend((pdf_path, png_path))
                plt.close(figure)
        os.replace(temporary_pdf, combined_path)
    except Exception:
        try:
            os.unlink(temporary_pdf)
        except FileNotFoundError:
            pass
        raise
    outputs.insert(0, combined_path)

    coordinate_rows = paired_coordinates(selected)
    companion = companion_figure(occupancy, nested, coordinate_rows)
    companion_pdf = OUTPUT / f"{COMPANION_STEM}.pdf"
    companion_png = OUTPUT / f"{COMPANION_STEM}.png"
    atomic_savefig(companion, companion_pdf, bbox_inches="tight", facecolor="white")
    atomic_savefig(
        companion,
        companion_png,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(companion)
    outputs.extend((companion_pdf, companion_png))
    return outputs


def main() -> int:
    summary, selected, exclusions, occupancy, nested = validate_collection()
    outputs = render(selected, exclusions, occupancy, nested)
    print(
        json.dumps(
            {
                "status": "pass",
                "input_collection_status": summary["status"],
                "current_tasks": summary["current_tasks"],
                "selected_trajectory_rows": len(selected),
                "optimizer_exclusion_rows": len(exclusions),
                "outputs": [str(path) for path in outputs],
                "background_only": True,
                "pull_blind": True,
                "factor_selection_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
