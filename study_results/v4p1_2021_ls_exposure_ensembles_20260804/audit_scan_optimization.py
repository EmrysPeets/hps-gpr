#!/usr/bin/env python3
"""Fail-closed optimizer audit for the v4.1 length-scale exposure pilot.

This program is deliberately read-only with respect to fit products.  It reads
only scan attempts carrying a valid top-level ``_SUCCESS.json`` marker whose
recorded result hash matches the CSV on disk.  Every comparison is made at an
exact common (truth, scenario, toy, mass) row; no interpolation or mass
matching tolerance is used.

The output tables separate three questions:

* Did increasing the length-scale upper bound expose a higher-likelihood
  solution, as it is mathematically allowed to do?
* Did either fit miss a solution that is demonstrably feasible in its own
  domain?
* Is the fitted length scale at (or close to) the configured upper bound?

The audit can be rerun while production is in progress.  Its JSON gate remains
``incomplete`` until every predeclared scan task has a valid successful
attempt, and it remains ``repair_required`` if any optimizer anomaly survives.
It never launches a fit and never chooses a physics limit or p-value.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
SPEC_PATH = STUDY_DIR / "study_spec.json"
TASK_MANIFEST_PATH = STUDY_DIR / "derived" / "task_manifest.jsonl"
DERIVED_DIR = STUDY_DIR / "derived"

ROW_KEY = [
    "truth_model",
    "study_scenario",
    "background_toy_index",
    "mass_GeV",
    "ls_upper_factor_requested",
]
PAIR_KEY = ROW_KEY[:-1]

# The immutable production optimizer was reproducible at much better than this
# scale in the 2016 repeat audit.  The relative term avoids overreacting to
# sub-millilog-likelihood roundoff when the total LML is O(10^3).
LML_ABS_TOL = 1.0e-4
LML_REL_TOL = 1.0e-6

# "At bound" is an optimizer-state diagnostic; "near bound" is a conservative
# review flag for a potentially truncated likelihood surface.
AT_BOUND_FRACTION = 0.999
NEAR_BOUND_FRACTION = 0.98
FEASIBILITY_REL_TOL = 1.0e-4

REQUIRED_COLUMNS = {
    "task_id",
    "truth_model",
    "study_scenario",
    "background_toy_index",
    "mass_GeV",
    "ls_upper_factor_requested",
    "lml",
    "ls_lo",
    "ls_hi",
    "ls_init",
    "ls_opt",
    "ls_opt_over_sigma_x",
    "ls_hi_over_sigma_x",
    "sigma_x",
    "const_opt",
    "optimizer_seed",
    "extract_success",
    "training_geometry_valid",
    "expected_limit_bands",
    "fit_code_commit",
    "generated_config_sha256",
}


class AuditError(RuntimeError):
    """Raised when fit provenance or table structure fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_reviewed_collection_report(
    *,
    reviewed_rows: pd.DataFrame,
    reviewed_csv: Path,
    source_csv: Path,
    report_path: Path,
    summary_path: Path,
    task_status_path: Path,
    study_id: str,
    completed_tasks: int,
    incomplete_tasks: int,
) -> Dict[str, Any]:
    """Publish the optimizer-reviewed rows as a collector-compatible pair.

    The nominal collector products remain immutable.  This additional complete
    pair lets postprocessors consume the best-LML, exact-row table while
    retaining an explicit link to the optimizer audit and its source table.
    """

    reviewed_rows.to_csv(reviewed_csv, index=False)
    reviewed_sha = _sha256(reviewed_csv)
    source_sha = _sha256(source_csv)
    if reviewed_sha != source_sha:
        raise AuditError(
            "Collector-compatible reviewed CSV differs byte-for-byte from "
            "the optimizer-reviewed source table"
        )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "study_id": study_id,
        "kind": "scan",
        "partial": False,
        "completed_tasks": int(completed_tasks),
        "incomplete_tasks": int(incomplete_tasks),
        "rows": int(len(reviewed_rows)),
        "expected_limit_bands": False,
        "fit_rows_are_actual": True,
        "interpolation_used": False,
        "review_stage": "optimizer_selected_actual_fit_rows",
        "output": str(reviewed_csv.resolve()),
        "output_sha256": reviewed_sha,
        "source_output": str(source_csv.resolve()),
        "source_output_sha256": source_sha,
        "optimizer_audit_summary": str(summary_path.resolve()),
        "task_status": str(task_status_path.resolve()),
        "selectors": {
            "factors": [],
            "scenarios": [],
            "truths": [],
        },
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with report_path.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    values: List[Mapping[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"Expected object at {path}:{line_number}")
            values.append(value)
    return values


def _bool_series(values: pd.Series) -> pd.Series:
    """Normalize bool-like CSV values without treating strings as truthy."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    observed = set(normalized.unique())
    if not observed <= allowed:
        raise AuditError(
            f"Non-boolean values in {values.name}: {sorted(observed - allowed)}"
        )
    return normalized.eq("true")


def _attempt_number(attempt: Path) -> int:
    try:
        return int(attempt.name.split("_")[-1])
    except (ValueError, IndexError) as exc:
        raise AuditError(f"Malformed attempt directory: {attempt}") from exc


def _resolve_recorded_path(recorded: str) -> Path:
    path = Path(recorded)
    if not path.is_absolute():
        path = STUDY_DIR / path
    return path.resolve()


def _read_valid_attempts(
    spec: Mapping[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frames: List[pd.DataFrame] = []
    rejected: List[Dict[str, Any]] = []
    marker_paths = sorted(
        list(STUDY_DIR.glob("runs/scan/**/attempt_*/_SUCCESS.json"))
        + list(
            STUDY_DIR.glob(
                "runs/scan_repairs/**/attempt_*/_SUCCESS.json"
            )
        )
    )
    for marker_path in marker_paths:
        try:
            marker = _load_json(marker_path)
            attempt = marker_path.parent.resolve()
            if _resolve_recorded_path(str(marker["attempt"])) != attempt:
                raise AuditError("marker attempt path does not match its location")
            if marker.get("study_id") != spec["study_id"]:
                raise AuditError("study_id mismatch")
            if marker.get("fit_code_commit") != spec["fit_code"]["commit"]:
                raise AuditError("fit-code commit mismatch")
            if bool(marker.get("expected_limit_bands")):
                raise AuditError("expected-limit bands were enabled")
            task = marker.get("task")
            if not isinstance(task, dict) or task.get("kind") != "scan":
                raise AuditError("marker does not describe a scan task")
            result_path = _resolve_recorded_path(str(marker["result_path"]))
            if result_path.parent.resolve() != attempt:
                raise AuditError("result CSV is not inside the recorded attempt")
            if not result_path.is_file():
                raise AuditError("recorded result CSV is absent")
            if _sha256(result_path) != marker.get("result_sha256"):
                raise AuditError("result SHA-256 mismatch")

            frame = pd.read_csv(result_path)
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                raise AuditError(f"missing CSV columns: {sorted(missing)}")
            if frame.empty:
                raise AuditError("empty scan result")
            if frame["mass_GeV"].duplicated().any():
                raise AuditError("duplicate mass rows in one result")
            marker_masses = marker.get("masses_gev")
            if not isinstance(marker_masses, list):
                raise AuditError("marker masses_gev is absent or malformed")
            if set(frame["mass_GeV"].astype(float)) != {
                float(value) for value in marker_masses
            }:
                raise AuditError("marker/result exact mass-grid mismatch")
            if frame["task_id"].nunique() != 1:
                raise AuditError("more than one task_id in result")
            if str(frame["task_id"].iloc[0]) != str(task["task_id"]):
                raise AuditError("task_id mismatch between marker and result")
            if frame["fit_code_commit"].astype(str).nunique() != 1:
                raise AuditError("more than one fit-code commit in result")
            if str(frame["fit_code_commit"].iloc[0]) != spec["fit_code"]["commit"]:
                raise AuditError("CSV fit-code commit mismatch")
            if _bool_series(frame["expected_limit_bands"]).any():
                raise AuditError("CSV says expected-limit bands were enabled")
            if not _bool_series(frame["extract_success"]).all():
                raise AuditError("one or more extraction rows failed")
            if not _bool_series(frame["training_geometry_valid"]).all():
                raise AuditError("one or more rows failed training geometry")

            factor = int(task["factor"])
            if not (frame["ls_upper_factor_requested"].astype(int) == factor).all():
                raise AuditError("factor mismatch between marker and result")
            if not np.allclose(
                frame["ls_hi_over_sigma_x"].to_numpy(dtype=float),
                factor,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise AuditError("applied upper factor differs from task factor")
            if not np.isfinite(
                frame[
                    [
                        "mass_GeV",
                        "lml",
                        "ls_lo",
                        "ls_hi",
                        "ls_init",
                        "ls_opt",
                        "sigma_x",
                        "const_opt",
                    ]
                ].to_numpy(dtype=float)
            ).all():
                raise AuditError("non-finite optimizer diagnostic")

            frame = frame.copy()
            frame["attempt_number"] = _attempt_number(attempt)
            frame["attempt_path"] = str(attempt)
            frame["result_path"] = str(result_path)
            frame["result_sha256"] = str(marker["result_sha256"])
            frame["completed_utc"] = str(marker.get("completed_utc", ""))
            is_repair = "scan_repairs" in attempt.parts
            frame["fit_origin"] = (
                "targeted_repair" if is_repair else "nominal_scan"
            )
            if is_repair:
                repair_meta = marker.get("repair")
                if not isinstance(repair_meta, dict):
                    raise AuditError("repair marker metadata is absent")
                frame["repair_round"] = int(
                    repair_meta.get("repair_round", 1)
                )
                frame["repair_attempt_id"] = str(
                    repair_meta["repair_attempt_id"]
                )
                frame["repair_variant"] = str(repair_meta["variant"])
            else:
                frame["repair_round"] = 0
                frame["repair_attempt_id"] = ""
                frame["repair_variant"] = ""
            frames.append(frame)
        except Exception as exc:
            rejected.append(
                {
                    "marker_path": str(marker_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not frames:
        return pd.DataFrame(), rejected
    return pd.concat(frames, ignore_index=True), rejected


def _select_reviewed_rows(attempt_rows: pd.DataFrame) -> pd.DataFrame:
    """Choose the highest-LML *actual* fit row when attempts are duplicated."""

    attempt_rows = attempt_rows.copy()
    if "repair_kernel_constant_init" in attempt_rows:
        constant_init = attempt_rows[
            "repair_kernel_constant_init"
        ].astype(float).fillna(1.0)
    else:
        constant_init = pd.Series(
            np.ones(len(attempt_rows), dtype=float),
            index=attempt_rows.index,
        )
    if "repair_kernel_ls_init" in attempt_rows:
        repair_ls_init = pd.to_numeric(
            attempt_rows["repair_kernel_ls_init"], errors="coerce"
        )
        ls_init_for_state = repair_ls_init.where(
            repair_ls_init.notna(),
            attempt_rows["ls_init"].astype(float),
        )
    else:
        ls_init_for_state = attempt_rows["ls_init"].astype(float)
    attempt_rows["attempt_initialization_state"] = np.isclose(
        attempt_rows["const_opt"].to_numpy(dtype=float),
        constant_init.to_numpy(dtype=float),
        rtol=1.0e-5,
        atol=1.0e-8,
    ) & np.isclose(
        attempt_rows["ls_opt"].to_numpy(dtype=float),
        ls_init_for_state.to_numpy(dtype=float),
        rtol=1.0e-5,
        atol=1.0e-10,
    )
    if "repair_warm_start" in attempt_rows:
        warm_values: List[bool] = []
        for value in attempt_rows["repair_warm_start"]:
            if pd.isna(value):
                warm_values.append(False)
                continue
            normalized = str(value).strip().lower()
            if normalized not in {"true", "false"}:
                raise AuditError(
                    f"Malformed repair_warm_start value: {value!r}"
                )
            warm_values.append(normalized == "true")
        warm_start = pd.Series(
            np.asarray(warm_values, dtype=bool),
            index=attempt_rows.index,
        )
    else:
        warm_start = pd.Series(
            np.zeros(len(attempt_rows), dtype=bool),
            index=attempt_rows.index,
        )
    # The failure under review is the frozen card's geometric initialization
    # state.  A warm repair that remains at a *different*, higher-LML source
    # optimum can legitimately supersede it; it is not relabeled as the
    # original lock merely because the warm config starts at that source.
    attempt_rows["attempt_original_initialization_state"] = (
        attempt_rows["attempt_initialization_state"] & ~warm_start
    )

    duplicated_within_attempt = attempt_rows.duplicated(
        ROW_KEY + ["task_id", "attempt_path"], keep=False
    )
    if duplicated_within_attempt.any():
        examples = attempt_rows.loc[
            duplicated_within_attempt, ROW_KEY + ["task_id", "attempt_path"]
        ].head()
        raise AuditError(
            "Duplicate exact rows inside an attempt:\n"
            + examples.to_string(index=False)
        )

    ordered = attempt_rows.sort_values(
        ROW_KEY + ["lml", "attempt_number"],
        ascending=[True] * len(ROW_KEY) + [False, False],
        kind="mergesort",
    )
    selected = ordered.drop_duplicates(ROW_KEY, keep="first").copy()
    selected["actual_fit_selection"] = "highest_finite_lml_actual_row"
    selected["ls_bound_fraction"] = (
        selected["ls_opt"].astype(float) / selected["ls_hi"].astype(float)
    )
    selected["at_upper_bound"] = (
        selected["ls_bound_fraction"] >= AT_BOUND_FRACTION
    )
    selected["near_upper_bound"] = (
        selected["ls_bound_fraction"] >= NEAR_BOUND_FRACTION
    )
    selected["selected_at_config_initialization_state"] = selected[
        "attempt_initialization_state"
    ].astype(bool)
    selected["initialization_lock"] = selected[
        "attempt_original_initialization_state"
    ].astype(bool)

    attempt_metadata: List[Dict[str, Any]] = []
    for key, group in attempt_rows.groupby(ROW_KEY, sort=False):
        config_stasis = group[group["attempt_initialization_state"]]
        stasis = group[group["attempt_original_initialization_state"]]
        nonstasis = group[~group["attempt_original_initialization_state"]]
        best_stasis_lml = (
            math.nan if stasis.empty else float(stasis["lml"].max())
        )
        best_nonstasis_lml = (
            math.nan if nonstasis.empty else float(nonstasis["lml"].max())
        )
        reference = max(
            [
                abs(value)
                for value in (best_stasis_lml, best_nonstasis_lml)
                if math.isfinite(value)
            ]
            + [1.0]
        )
        tolerance = max(LML_ABS_TOL, LML_REL_TOL * reference)
        independently_reproduced = (
            stasis["optimizer_seed"].astype(str).nunique() >= 2
        )
        superseded = (
            math.isfinite(best_stasis_lml)
            and math.isfinite(best_nonstasis_lml)
            and best_nonstasis_lml > best_stasis_lml + tolerance
        )
        unresolved = (
            math.isfinite(best_stasis_lml)
            and not superseded
        )
        if unresolved and independently_reproduced:
            review_status = (
                "reproduced_but_not_validated_stationary_state"
            )
        elif unresolved:
            review_status = "unrepaired_initialization_state"
        elif superseded:
            review_status = "superseded_by_better_actual_branch"
        else:
            review_status = "no_initialization_state_observed"
        attempt_metadata.append(
            {
                **dict(zip(ROW_KEY, key)),
                "n_actual_fit_attempts": int(len(group)),
                "n_independent_optimizer_seeds": int(
                    group["optimizer_seed"].astype(str).nunique()
                ),
                "n_initialization_state_attempts": int(len(stasis)),
                "n_config_initialization_state_attempts": int(
                    len(config_stasis)
                ),
                "n_independent_initialization_state_seeds": int(
                    stasis["optimizer_seed"].astype(str).nunique()
                ),
                "initialization_state_independently_reproduced": bool(
                    independently_reproduced
                ),
                "best_initialization_state_lml": best_stasis_lml,
                "best_noninitialization_lml": best_nonstasis_lml,
                "initialization_state_superseded": bool(superseded),
                "initialization_state_unresolved": bool(unresolved),
                "initialization_state_review_status": review_status,
            }
        )
    selected = selected.merge(
        pd.DataFrame(attempt_metadata),
        on=ROW_KEY,
        how="left",
        validate="one_to_one",
    )
    return selected.sort_values(ROW_KEY, kind="mergesort").reset_index(drop=True)


def _assert_factor_invariants(selected: pd.DataFrame) -> None:
    """Fail if anything other than the intended bound changes across factors."""

    invariant_columns = [
        "sigma_x",
        "ls_lo",
        "mass_GeV",
        "n_train",
        "rebinned_n_full",
        "rebinned_n_train_expected",
    ]
    for _, group in selected.groupby(PAIR_KEY, sort=False):
        if len(group) < 2:
            continue
        for column in invariant_columns:
            if column not in group:
                continue
            values = group[column].to_numpy()
            if np.issubdtype(values.dtype, np.number):
                if not np.allclose(
                    values.astype(float),
                    float(values[0]),
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=False,
                ):
                    raise AuditError(
                        f"Factor-comparison invariant drift in {column} for "
                        f"{tuple(group.iloc[0][PAIR_KEY])}"
                    )
            elif len(set(values.astype(str))) != 1:
                raise AuditError(
                    f"Factor-comparison invariant drift in {column} for "
                    f"{tuple(group.iloc[0][PAIR_KEY])}"
                )


def _attempt_selection_ledger(
    attempt_rows: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Record every actual candidate row and the deterministic review choice."""

    selected_lookup = {
        tuple(row[key] for key in ROW_KEY): (
            str(row["attempt_path"]),
            str(row["result_sha256"]),
            float(row["lml"]),
        )
        for _, row in selected.iterrows()
    }
    ordered = attempt_rows.sort_values(
        ROW_KEY + ["lml", "attempt_number", "attempt_path"],
        ascending=[True] * len(ROW_KEY) + [False, False, True],
        kind="mergesort",
    ).copy()
    ordered["actual_fit_rank_within_exact_row"] = (
        ordered.groupby(ROW_KEY, sort=False).cumcount() + 1
    )
    selected_flags: List[bool] = []
    selected_lml_values: List[float] = []
    for _, row in ordered.iterrows():
        key = tuple(row[column] for column in ROW_KEY)
        if key not in selected_lookup:
            raise AuditError(f"Selection lookup is missing exact row {key}")
        selected_path, selected_hash, selected_lml = selected_lookup[key]
        is_selected = (
            str(row["attempt_path"]) == selected_path
            and str(row["result_sha256"]) == selected_hash
        )
        selected_flags.append(is_selected)
        selected_lml_values.append(selected_lml)
    ordered["selected_for_review"] = selected_flags
    ordered["selected_lml"] = selected_lml_values
    ordered["delta_lml_to_selected"] = (
        ordered["lml"].astype(float)
        - ordered["selected_lml"].astype(float)
    )
    ordered["selection_status"] = np.where(
        ordered["selected_for_review"],
        "selected_highest_finite_lml_actual_row",
        "retained_in_ledger_not_selected",
    )
    if int(ordered["selected_for_review"].sum()) != len(selected):
        raise AuditError(
            "Attempt ledger does not select exactly one candidate per reviewed row"
        )
    if (
        ordered.groupby(ROW_KEY)["selected_for_review"].sum().astype(int) != 1
    ).any():
        raise AuditError("Attempt ledger selection is not one-to-one")
    ordered["interpolation_used"] = False
    ordered["fit_row_is_actual"] = True
    return ordered.reset_index(drop=True)


def _nested_pairs(
    selected: pd.DataFrame, factors: Sequence[int]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    indexed = {
        tuple(row[key] for key in ROW_KEY): row
        for _, row in selected.iterrows()
    }
    base_keys = sorted(
        {tuple(row[key] for key in PAIR_KEY) for _, row in selected.iterrows()}
    )
    factor_pairs = list(itertools.combinations(factors, 2))
    for key in base_keys:
        for low_factor, high_factor in factor_pairs:
            low = indexed.get(key + (low_factor,))
            high = indexed.get(key + (high_factor,))
            if low is None or high is None:
                continue
            low_lml = float(low["lml"])
            high_lml = float(high["lml"])
            delta = high_lml - low_lml
            tolerance = max(
                LML_ABS_TOL,
                LML_REL_TOL * max(abs(low_lml), abs(high_lml), 1.0),
            )
            high_ls_norm = float(high["ls_opt_over_sigma_x"])
            high_feasible_low = high_ls_norm <= low_factor * (
                1.0 + FEASIBILITY_REL_TOL
            )
            if delta < -tolerance:
                status = "higher_factor_optimizer_miss"
                repair_factor = high_factor
                warm_factor = low_factor
            elif delta > tolerance and high_feasible_low:
                status = "lower_factor_optimizer_miss"
                repair_factor = low_factor
                warm_factor = high_factor
            elif abs(delta) <= tolerance:
                status = "consistent_lml_plateau"
                repair_factor = math.nan
                warm_factor = math.nan
            else:
                status = "allowed_domain_gain"
                repair_factor = math.nan
                warm_factor = math.nan
            row = dict(zip(PAIR_KEY, key))
            row.update(
                {
                    "low_factor": low_factor,
                    "high_factor": high_factor,
                    "adjacent_factors": bool(
                        factors.index(high_factor)
                        == factors.index(low_factor) + 1
                    ),
                    "low_lml": low_lml,
                    "high_lml": high_lml,
                    "delta_lml_high_minus_low": delta,
                    "lml_tolerance": tolerance,
                    "low_ls_opt_over_sigma_x": float(
                        low["ls_opt_over_sigma_x"]
                    ),
                    "high_ls_opt_over_sigma_x": high_ls_norm,
                    "low_at_upper_bound": bool(low["at_upper_bound"]),
                    "high_at_upper_bound": bool(high["at_upper_bound"]),
                    "high_solution_feasible_in_low_domain": bool(
                        high_feasible_low
                    ),
                    "status": status,
                    "repair_factor": repair_factor,
                    "warm_start_source_factor": warm_factor,
                    "low_attempt_path": str(low["attempt_path"]),
                    "high_attempt_path": str(high["attempt_path"]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _bound_occupancy(selected: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "truth_model",
        "study_scenario",
        "ls_upper_factor_requested",
    ]
    rows: List[Dict[str, Any]] = []
    for key, group in selected.groupby(groups, sort=True):
        toy_flags = group.groupby("background_toy_index", sort=True).agg(
            any_at_bound=("at_upper_bound", "any"),
            any_near_bound=("near_upper_bound", "any"),
            any_initialization_lock=("initialization_lock", "any"),
            any_unresolved_initialization_state=(
                "initialization_state_unresolved",
                "any",
            ),
        )
        rows.append(
            {
                **dict(zip(groups, key)),
                "n_rows": int(len(group)),
                "n_toys": int(group["background_toy_index"].nunique()),
                "n_mass_points": int(group["mass_GeV"].nunique()),
                "at_bound_rows": int(group["at_upper_bound"].sum()),
                "near_bound_rows": int(group["near_upper_bound"].sum()),
                "at_bound_row_fraction": float(group["at_upper_bound"].mean()),
                "near_bound_row_fraction": float(
                    group["near_upper_bound"].mean()
                ),
                "toys_with_any_bound_row": int(toy_flags["any_at_bound"].sum()),
                "toys_with_any_near_bound_row": int(
                    toy_flags["any_near_bound"].sum()
                ),
                "toys_with_initialization_lock": int(
                    toy_flags["any_initialization_lock"].sum()
                ),
                "toys_with_unresolved_initialization_state": int(
                    toy_flags["any_unresolved_initialization_state"].sum()
                ),
                "median_ls_opt_over_sigma_x": float(
                    group["ls_opt_over_sigma_x"].median()
                ),
                "q05_ls_opt_over_sigma_x": float(
                    group["ls_opt_over_sigma_x"].quantile(0.05)
                ),
                "q95_ls_opt_over_sigma_x": float(
                    group["ls_opt_over_sigma_x"].quantile(0.95)
                ),
                "max_ls_bound_fraction": float(
                    group["ls_bound_fraction"].max()
                ),
                "independence_caveat": (
                    "mass rows share each toy; inferential counting unit is toy"
                ),
            }
        )
    return pd.DataFrame(rows)


def _repair_manifest(
    selected: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Build a row-level repair ledger without fabricating replacement rows."""

    candidates: List[Dict[str, Any]] = []
    by_key = {
        tuple(row[key] for key in ROW_KEY): row
        for _, row in selected.iterrows()
    }

    if not pairs.empty:
        anomalies = pairs[
            pairs["status"].isin(
                ["higher_factor_optimizer_miss", "lower_factor_optimizer_miss"]
            )
        ]
        for _, pair in anomalies.iterrows():
            repair_factor = int(pair["repair_factor"])
            warm_factor = int(pair["warm_start_source_factor"])
            base_key = tuple(pair[key] for key in PAIR_KEY)
            target = by_key[base_key + (repair_factor,)]
            source = by_key[base_key + (warm_factor,)]
            candidates.append(
                {
                    **dict(zip(PAIR_KEY, base_key)),
                    "repair_factor": repair_factor,
                    "target_task_id": str(target["task_id"]),
                    "target_attempt_path": str(target["attempt_path"]),
                    "reason": str(pair["status"]),
                    "warm_start_source_factor": warm_factor,
                    "warm_start_source_attempt_path": str(
                        source["attempt_path"]
                    ),
                    "warm_start_ls_opt": float(source["ls_opt"]),
                    "warm_start_const_opt": float(source["const_opt"]),
                    "warm_start_is_feasible": bool(
                        float(source["ls_opt"])
                        <= float(target["ls_hi"])
                        * (1.0 + FEASIBILITY_REL_TOL)
                    ),
                    "current_target_lml": float(target["lml"]),
                    "source_lml": float(source["lml"]),
                    "required_action": (
                        "new per-mass actual fit at target bound; initialize "
                        "from recorded source optimum and retain full restart set"
                    ),
                }
            )

    for _, target in selected[
        selected["initialization_state_unresolved"]
    ].iterrows():
        base_key = tuple(target[key] for key in PAIR_KEY)
        target_factor = int(target["ls_upper_factor_requested"])
        feasible = selected[
            (selected["truth_model"] == base_key[0])
            & (selected["study_scenario"] == base_key[1])
            & (selected["background_toy_index"] == base_key[2])
            & (selected["mass_GeV"] == base_key[3])
            & (~selected["initialization_state_unresolved"])
            & (
                selected["ls_opt"]
                <= float(target["ls_hi"]) * (1.0 + FEASIBILITY_REL_TOL)
            )
        ].sort_values("lml", ascending=False)
        if feasible.empty:
            warm_factor: Any = math.nan
            warm_path = ""
            warm_ls = math.nan
            warm_const = math.nan
            warm_feasible = False
            required_action = (
                "new per-mass actual fits at target bound with at least three "
                "deterministic optimizer-seed salts; do not reuse the locked row"
            )
        else:
            source = feasible.iloc[0]
            warm_factor = int(source["ls_upper_factor_requested"])
            warm_path = str(source["attempt_path"])
            warm_ls = float(source["ls_opt"])
            warm_const = float(source["const_opt"])
            warm_feasible = True
            required_action = (
                "new per-mass actual fit at target bound; initialize from "
                "recorded feasible optimum and retain full restart set"
            )
        candidates.append(
            {
                **dict(zip(PAIR_KEY, base_key)),
                "repair_factor": target_factor,
                "target_task_id": str(target["task_id"]),
                "target_attempt_path": str(target["attempt_path"]),
                "reason": "unresolved_exact_initialization_state",
                "warm_start_source_factor": warm_factor,
                "warm_start_source_attempt_path": warm_path,
                "warm_start_ls_opt": warm_ls,
                "warm_start_const_opt": warm_const,
                "warm_start_is_feasible": warm_feasible,
                "current_target_lml": float(target["lml"]),
                "source_lml": (
                    math.nan if feasible.empty else float(feasible.iloc[0]["lml"])
                ),
                "required_action": required_action,
            }
        )

    columns = PAIR_KEY + [
        "repair_factor",
        "target_task_id",
        "target_attempt_path",
        "reason",
        "warm_start_source_factor",
        "warm_start_source_attempt_path",
        "warm_start_ls_opt",
        "warm_start_const_opt",
        "warm_start_is_feasible",
        "current_target_lml",
        "source_lml",
        "required_action",
    ]
    if not candidates:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(candidates)
    grouped: List[Dict[str, Any]] = []
    dedup_key = PAIR_KEY + ["repair_factor"]
    for _, group in frame.groupby(dedup_key, sort=True, dropna=False):
        ordered = group.sort_values(
            ["warm_start_is_feasible", "source_lml"],
            ascending=[False, False],
            na_position="last",
        )
        chosen = ordered.iloc[0].to_dict()
        chosen["reason"] = ";".join(sorted(set(group["reason"].astype(str))))
        if "unresolved_exact_initialization_state" in set(
            group["reason"].astype(str)
        ):
            chosen["required_action"] = (
                str(chosen["required_action"])
                + "; initialization lock requires an independently seeded repeat"
            )
        grouped.append(chosen)
    return pd.DataFrame(grouped)[columns].sort_values(
        dedup_key, kind="mergesort"
    )


def _completion(
    selected: pd.DataFrame,
    manifest: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = [row for row in manifest if row.get("kind") == "scan"]
    expected_ids = {str(row["task_id"]) for row in expected}
    grid = spec["default_mass_grid_mev"]
    expected_mass_mev = set(
        range(int(grid["min"]), int(grid["max"]) + 1, int(grid["step"]))
    )
    selected = selected[selected["fit_origin"] == "nominal_scan"].copy()
    selected["mass_mev_exact_grid"] = np.rint(
        selected["mass_GeV"].astype(float) * 1000.0
    ).astype(int)
    residual = (
        selected["mass_GeV"].astype(float)
        - selected["mass_mev_exact_grid"].astype(float) / 1000.0
    ).abs()
    if (residual > 5.0e-10).any():
        raise AuditError("A selected mass row is not on an integer-MeV grid")
    complete_ids = {
        str(task_id)
        for task_id, group in selected.groupby("task_id", sort=False)
        if set(group["mass_mev_exact_grid"].astype(int)) == expected_mass_mev
    }
    partial_ids = sorted(
        set(selected["task_id"].astype(str).unique()) - complete_ids
    )
    missing = sorted(expected_ids - complete_ids)
    unexpected = sorted(complete_ids - expected_ids)
    return {
        "expected_scan_tasks": len(expected_ids),
        "completed_valid_scan_tasks": len(complete_ids & expected_ids),
        "missing_scan_tasks": len(missing),
        "partial_scan_tasks": len(partial_ids),
        "unexpected_scan_tasks": len(unexpected),
        "missing_task_ids": missing,
        "partial_task_ids": partial_ids,
        "unexpected_task_ids": unexpected,
    }


def run_audit(output_dir: Path) -> Dict[str, Any]:
    spec = _load_json(SPEC_PATH)
    manifest = _load_jsonl(TASK_MANIFEST_PATH)
    factors = [int(value) for value in spec["length_scale_upper_factors"]]
    attempt_rows, rejected = _read_valid_attempts(spec)
    if attempt_rows.empty:
        raise AuditError("No valid completed scan attempt is available")
    selected = _select_reviewed_rows(attempt_rows)
    _assert_factor_invariants(selected)
    pairs = _nested_pairs(selected, factors)
    selection_ledger = _attempt_selection_ledger(attempt_rows, selected)
    occupancy = _bound_occupancy(selected)
    repairs = _repair_manifest(selected, pairs)
    # Completion is a property of the frozen nominal production.  Compute it
    # from nominal attempt rows before best-LML selection, because a targeted
    # repair may supersede one mass in ``selected`` without making the original
    # 11-mass task partial.
    completion = _completion(attempt_rows, manifest, spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "reviewed_rows": output_dir / "scan_optimizer_reviewed_actual_rows.csv",
        "selection_ledger": (
            output_dir / "scan_optimizer_actual_fit_selection_ledger.csv"
        ),
        "nested_pairs": output_dir / "scan_optimizer_nested_lml_audit.csv",
        "bound_occupancy": output_dir / "scan_optimizer_bound_occupancy.csv",
        "repair_manifest": output_dir / "scan_optimizer_repair_manifest.csv",
        "rejected_attempts": output_dir / "scan_optimizer_rejected_attempts.csv",
        "summary": output_dir / "scan_optimizer_audit_summary.json",
        "reviewed_complete_rows": (
            output_dir / "scan_reviewed_rows_complete.csv"
        ),
        "reviewed_complete_collection": (
            output_dir / "scan_reviewed_collection_complete.json"
        ),
    }
    selected.to_csv(paths["reviewed_rows"], index=False)
    selection_ledger.to_csv(paths["selection_ledger"], index=False)
    pairs.to_csv(paths["nested_pairs"], index=False)
    occupancy.to_csv(paths["bound_occupancy"], index=False)
    repairs.to_csv(paths["repair_manifest"], index=False)
    pd.DataFrame(
        rejected, columns=["marker_path", "reason"]
    ).to_csv(paths["rejected_attempts"], index=False)

    status_counts = (
        {}
        if pairs.empty
        else {str(k): int(v) for k, v in pairs["status"].value_counts().items()}
    )
    init_locks = int(selected["initialization_lock"].sum())
    unresolved_initialization = int(
        selected["initialization_state_unresolved"].sum()
    )
    reproduced_initialization = int(
        selected["initialization_state_independently_reproduced"].sum()
    )
    optimizer_anomalies = int(
        status_counts.get("higher_factor_optimizer_miss", 0)
        + status_counts.get("lower_factor_optimizer_miss", 0)
        + unresolved_initialization
    )
    complete = (
        completion["missing_scan_tasks"] == 0
        and completion["unexpected_scan_tasks"] == 0
    )
    if rejected:
        gate = "invalid_provenance"
    elif not complete:
        gate = "incomplete"
    elif optimizer_anomalies or not repairs.empty:
        gate = "repair_required"
    else:
        gate = "optimizer_audit_pass"
    summary = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "audit_gate": gate,
        "fit_rows_are_actual": True,
        "interpolation_used": False,
        "expected_limit_bands_constructed": False,
        "lml_tolerance": {
            "absolute": LML_ABS_TOL,
            "relative": LML_REL_TOL,
            "formula": (
                "max(abs_tol, rel_tol*max(abs(low_lml),abs(high_lml),1))"
            ),
        },
        "bound_thresholds": {
            "at_bound_fraction_of_upper": AT_BOUND_FRACTION,
            "near_bound_fraction_of_upper": NEAR_BOUND_FRACTION,
        },
        "completion": completion,
        "valid_attempt_rows_read": int(len(attempt_rows)),
        "reviewed_exact_rows": int(len(selected)),
        "actual_fit_selection_ledger_rows": int(len(selection_ledger)),
        "selected_rows_in_ledger": int(
            selection_ledger["selected_for_review"].sum()
        ),
        "selected_rows_by_fit_origin": {
            str(key): int(value)
            for key, value in selected["fit_origin"].value_counts().items()
        },
        "selected_rows_by_repair_round": {
            str(int(key)): int(value)
            for key, value in selected[
                selected["fit_origin"] == "targeted_repair"
            ]["repair_round"].value_counts().sort_index().items()
        },
        "valid_successful_attempts": int(
            attempt_rows[["task_id", "attempt_path"]]
            .drop_duplicates()
            .shape[0]
        ),
        "rejected_success_markers": len(rejected),
        "nested_pair_status_counts": status_counts,
        "exact_initialization_locks": init_locks,
        "unresolved_initialization_states": unresolved_initialization,
        "independently_reproduced_initialization_states": (
            reproduced_initialization
        ),
        "unique_target_rows_requiring_repair": int(len(repairs)),
        "scientific_scope": (
            "optimizer and hyperparameter-support pilot only; no coverage, "
            "expected-band, exclusion, or discovery calibration claim"
        ),
        "repair_policy": [
            (
                "Freeze the completed nominal production outputs before any "
                "repair fits."
            ),
            (
                "For every repair row, run a new exact-mass fit at the same "
                "target bound. Use the recorded feasible optimum as both the "
                "length-scale and kernel-constant warm start when available."
            ),
            (
                "Initialization locks require independently salted optimizer "
                "seeds even when a warm start is available; retain the nominal "
                "12-restart setting."
            ),
            (
                "Independent reproduction of the exact initialization state is "
                "recorded but remains unresolved unless a better actual branch "
                "supersedes it; it is never silently promoted to a pass."
            ),
            (
                "Select the maximum finite LML only among recorded actual fits "
                "for the same exact target row; keep an attempt ledger."
            ),
            (
                "Repeat this audit. Any surviving nested dominance failure or "
                "initialization lock fails the bound-choice gate; never fill a "
                "row by interpolation."
            ),
        ],
        "occupancy_interpretation": (
            "Row fractions are descriptive because mass points share each toy. "
            "Toy is the independent ensemble unit; ten toys are only a pilot."
        ),
        "candidate_bound_gate": {
            "projected_2021_100pct_scenarios": [
                "2021_1pct_x100",
                "2021_10pct_x10",
            ],
            "required_truth_models": list(spec["truth_models"].keys()),
            "diagnostic_not_projection_veto": [
                "2021_1pct",
                "2021_1pct_x10",
                "2021_10pct",
            ],
            "common_card_caveat": (
                "If one bound is to be frozen for every exposure, require the "
                "same gate across all five scenarios."
            ),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    with paths["summary"].open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    incomplete_tasks = (
        int(completion["missing_scan_tasks"])
        + int(completion["partial_scan_tasks"])
    )
    _write_reviewed_collection_report(
        reviewed_rows=selected,
        reviewed_csv=paths["reviewed_complete_rows"],
        source_csv=paths["reviewed_rows"],
        report_path=paths["reviewed_complete_collection"],
        summary_path=paths["summary"],
        task_status_path=output_dir / "scan_task_status_complete.csv",
        study_id=str(spec["study_id"]),
        completed_tasks=int(completion["completed_valid_scan_tasks"]),
        incomplete_tasks=incomplete_tasks,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DERIVED_DIR,
        help="Directory for audit CSV and JSON products (default: derived/)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_audit(args.output_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
