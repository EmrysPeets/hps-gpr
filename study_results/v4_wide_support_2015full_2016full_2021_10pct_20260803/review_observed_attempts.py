#!/usr/bin/env python3
"""Review three unchanged-card v4 observed scans without interpolation.

Each input attempt must contain the exact 415 active dataset/mass states:

* 2015: 19--90 MeV in 1 MeV steps (72 rows);
* 2016: 39--180 MeV in 1 MeV steps (142 rows);
* 2021: 50--250 MeV in 1 MeV steps (201 rows).

For every dataset/mass state, the exact maximum-LML input row is selected only
when at least one other complete unchanged-card attempt reproduces its LML
within ``abs(delta LML) <= 2e-5``. Coordinate differences are recorded for
review but are not silently averaged or interpolated.

This script does not stitch the 232-row shared-coupling campaign. The scalar
``A_hat``, ``sigma_A``, and density columns do not retain the per-bin observed
counts, GP mean, full GP covariance, and signal template needed to reproduce
the count-scale likelihood exactly. That reconstruction belongs to
``run_combined_bands_cached_fixed_reviewed.py``, which rebuilds the fixed GP
predictions before evaluating the observed likelihood and pseudoexperiments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
DERIVED = CAMPAIGN_DIR / "derived"
CONFIG = (
    REPO
    / "study_configs"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "config_obsUL90_combined_wide_support_v4_observed_only.yaml"
)

ATTEMPTS = (
    ("attempt_01", CAMPAIGN_DIR / "observed_attempt_01" / "results_single.csv"),
    ("attempt_02", CAMPAIGN_DIR / "observed_attempt_02" / "results_single.csv"),
    ("attempt_03", CAMPAIGN_DIR / "observed_attempt_03" / "results_single.csv"),
)
EXPECTED_MASS_MEV = {
    "2015": tuple(range(19, 91)),
    "2016": tuple(range(39, 181)),
    "2021": tuple(range(50, 251)),
}
DATASET_ORDER = {dataset: index for index, dataset in enumerate(EXPECTED_MASS_MEV)}
EXPECTED_STATE_COUNT = sum(len(values) for values in EXPECTED_MASS_MEV.values())
EXPECTED_ATTEMPT_COUNT = len(ATTEMPTS)
LML_REPRODUCTION_ATOL = 2.0e-5
STATIC_NUMERIC_ATOL = 1.0e-12

REQUIRED_COLUMNS = {
    "dataset",
    "mass_GeV",
    "integral_density",
    "density_window_fully_covered",
    "A_hat",
    "sigma_A",
    "extract_success",
    "const_opt",
    "ls_opt",
    "lml",
    "n_train",
    "n_train_low",
    "n_train_high",
}

# These values are fixed by the input/card/bin geometry and must agree between
# complete attempts. Optimizer-dependent coordinates, predictions, limits,
# fitted yields, and boundary flags are intentionally absent.
UNCHANGED_CARD_COLUMNS = (
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
    "cls_statistic",
    "cls_calibration",
    "signal_model",
    "global_method",
    "visibility",
    "ls_lo",
    "ls_hi",
    "ls_init",
    "sigma_x",
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
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "ls_lo_over_sigma",
    "ls_hi_over_sigma",
)

LEDGER_COLUMNS = (
    "dataset",
    "mass_GeV",
    "mass_MeV",
    "attempt_index",
    "attempt_label",
    "attempt_source",
    "attempt_source_sha256",
    "lml",
    "selected_max_lml",
    "delta_lml_from_selected_max",
    "abs_delta_lml_from_selected_max",
    "const_opt",
    "selected_const_opt",
    "delta_const_opt_from_selected_max",
    "abs_delta_const_opt_from_selected_max",
    "relative_abs_delta_const_opt_from_selected_max",
    "ls_opt",
    "selected_ls_opt",
    "delta_ls_opt_from_selected_max",
    "abs_delta_ls_opt_from_selected_max",
    "relative_abs_delta_ls_opt_from_selected_max",
    "within_lml_reproduction_tolerance",
    "is_selected_maximum",
    "selected_attempt",
    "selected_source",
    "branch_multiplicity",
    "reproducing_attempts",
    "reproducing_other_attempts",
    "reproducing_sources",
    "review_status",
    "interpolated",
)

UNRESOLVED_COLUMNS = (
    "dataset",
    "mass_GeV",
    "mass_MeV",
    "reason",
    "selected_attempt",
    "selected_source",
    "selected_lml",
    "selected_const_opt",
    "selected_ls_opt",
    "branch_multiplicity",
    "nearest_other_attempt",
    "nearest_other_source",
    "nearest_abs_delta_lml",
    "nearest_delta_const_opt",
    "nearest_delta_ls_opt",
    "attempt_lml_by_label",
    "attempt_const_opt_by_label",
    "attempt_ls_opt_by_label",
    "attempt_sources",
    "interpolated",
)

REVIEW_PROVENANCE_COLUMNS = (
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
)

STITCHING_REASON = (
    "Not written: A_hat, sigma_A, and integral_density do not encode the "
    "per-bin observed counts, GP mean vector, full posterior covariance "
    "blocks, and signal templates required by the shared count-scale "
    "likelihood. Use run_combined_bands_cached_fixed_reviewed.py to rebuild "
    "fixed GP predictions and compute the exact 232-row observed campaign."
)


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


def expected_keys() -> list[tuple[str, int]]:
    return [
        (dataset, mass_mev)
        for dataset, masses_mev in EXPECTED_MASS_MEV.items()
        for mass_mev in masses_mev
    ]


def normalize_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise RuntimeError(f"{label} contains a missing boolean value")
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    allowed = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    invalid = normalized.isna() | ~normalized.isin(allowed)
    if bool(invalid.any()):
        sample = series.loc[invalid].head(5).tolist()
        raise RuntimeError(f"{label} contains invalid booleans: {sample}")
    return normalized.map(allowed).astype(bool)


def sort_states(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_dataset_order"] = out["dataset"].map(DATASET_ORDER)
    out = out.sort_values(["_dataset_order", "mass_MeV", "attempt_index"])
    return out.drop(columns="_dataset_order").reset_index(drop=True)


def validate_exact_grid(frame: pd.DataFrame, label: str) -> None:
    if len(frame) != EXPECTED_STATE_COUNT:
        raise RuntimeError(
            f"{label} has {len(frame)} rows; expected {EXPECTED_STATE_COUNT}"
        )
    duplicate = frame.duplicated(["dataset", "mass_MeV"], keep=False)
    if bool(duplicate.any()):
        sample = frame.loc[duplicate, ["dataset", "mass_GeV"]].head(10)
        raise RuntimeError(
            f"{label} has duplicate dataset/mass states:\n"
            + sample.to_string(index=False)
        )
    actual = list(
        zip(
            sort_states(frame)["dataset"].astype(str),
            sort_states(frame)["mass_MeV"].astype(int),
        )
    )
    expected = expected_keys()
    if actual != expected:
        actual_set = set(actual)
        expected_set = set(expected)
        missing = sorted(expected_set.difference(actual_set))[:10]
        extra = sorted(actual_set.difference(expected_set))[:10]
        raise RuntimeError(
            f"{label} does not have the exact 415-state grid; "
            f"missing={missing}, extra={extra}"
        )


def load_attempt(
    attempt_index: int,
    attempt_label: str,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise RuntimeError(
            f"{attempt_label} is missing required columns: {missing_columns}"
        )
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise RuntimeError(f"{attempt_label} contains missing values in {missing}")

    frame["dataset"] = frame["dataset"].astype(str).str.strip()
    unexpected_datasets = sorted(set(frame["dataset"]).difference(EXPECTED_MASS_MEV))
    if unexpected_datasets:
        raise RuntimeError(
            f"{attempt_label} has unexpected datasets: {unexpected_datasets}"
        )

    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_columns = [
            column
            for column in numeric.columns
            if not np.isfinite(numeric[column].to_numpy(dtype=float)).all()
        ]
        raise RuntimeError(
            f"{attempt_label} has non-finite numeric values in {bad_columns}"
        )

    frame["extract_success"] = normalize_boolean(
        frame["extract_success"], f"{attempt_label}.extract_success"
    )
    frame["density_window_fully_covered"] = normalize_boolean(
        frame["density_window_fully_covered"],
        f"{attempt_label}.density_window_fully_covered",
    )
    if not bool(frame["extract_success"].all()):
        failed = frame.loc[
            ~frame["extract_success"], ["dataset", "mass_GeV"]
        ].head(10)
        raise RuntimeError(
            f"{attempt_label} has failed extractions:\n"
            + failed.to_string(index=False)
        )
    if not bool(frame["density_window_fully_covered"].all()):
        failed = frame.loc[
            ~frame["density_window_fully_covered"], ["dataset", "mass_GeV"]
        ].head(10)
        raise RuntimeError(
            f"{attempt_label} has uncovered density windows:\n"
            + failed.to_string(index=False)
        )

    if "interpolated" in frame.columns:
        frame["interpolated"] = normalize_boolean(
            frame["interpolated"], f"{attempt_label}.interpolated"
        )
        if bool(frame["interpolated"].any()):
            raise RuntimeError(f"{attempt_label} contains interpolated rows")

    mass_mev = np.rint(1000.0 * frame["mass_GeV"].to_numpy(float)).astype(int)
    if not np.allclose(
        frame["mass_GeV"].to_numpy(float),
        mass_mev / 1000.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(f"{attempt_label} contains an off-grid mass")
    frame["mass_MeV"] = mass_mev

    for column in ("n_train", "n_train_low", "n_train_high"):
        values = frame[column].to_numpy(float)
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=0.0):
            raise RuntimeError(f"{attempt_label}.{column} is not integer-valued")
    if not bool((frame["n_train_low"].to_numpy(float) > 0.0).all()):
        bad = frame.loc[
            frame["n_train_low"].to_numpy(float) <= 0.0,
            ["dataset", "mass_GeV", "n_train_low"],
        ].head(10)
        raise RuntimeError(
            f"{attempt_label} has no low-side training bins:\n"
            + bad.to_string(index=False)
        )
    if not bool((frame["n_train_high"].to_numpy(float) > 0.0).all()):
        bad = frame.loc[
            frame["n_train_high"].to_numpy(float) <= 0.0,
            ["dataset", "mass_GeV", "n_train_high"],
        ].head(10)
        raise RuntimeError(
            f"{attempt_label} has no high-side training bins:\n"
            + bad.to_string(index=False)
        )
    if not bool(
        (
            frame["n_train_low"].to_numpy(int)
            + frame["n_train_high"].to_numpy(int)
            == frame["n_train"].to_numpy(int)
        ).all()
    ):
        raise RuntimeError(f"{attempt_label} fails n_train side-count closure")
    if not bool((frame["sigma_A"].to_numpy(float) > 0.0).all()):
        raise RuntimeError(f"{attempt_label} contains non-positive sigma_A")
    if not bool((frame["integral_density"].to_numpy(float) > 0.0).all()):
        raise RuntimeError(
            f"{attempt_label} contains non-positive integral density"
        )

    frame["attempt_index"] = int(attempt_index)
    frame["attempt_label"] = attempt_label
    frame["attempt_source"] = repo_path(path)
    frame["attempt_source_sha256"] = sha256(path)
    validate_exact_grid(frame, attempt_label)
    frame = sort_states(frame)
    metadata = {
        "label": attempt_label,
        "source": repo_path(path),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "dataset_rows": {
            dataset: int((frame["dataset"] == dataset).sum())
            for dataset in EXPECTED_MASS_MEV
        },
        "minimum_n_train_low": int(frame["n_train_low"].min()),
        "minimum_n_train_high": int(frame["n_train_high"].min()),
    }
    return frame, metadata


def compare_unchanged_card_columns(
    frames: Iterable[pd.DataFrame],
) -> None:
    frames = list(frames)
    reference = frames[0]
    missing = [
        column
        for column in UNCHANGED_CARD_COLUMNS
        if any(column not in frame.columns for frame in frames)
    ]
    if missing:
        raise RuntimeError(
            f"Cannot verify unchanged-card fields; missing columns: {missing}"
        )

    for candidate in frames[1:]:
        for column in UNCHANGED_CARD_COLUMNS:
            left = reference[column]
            right = candidate[column]
            if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
                right.dtype
            ):
                agrees = np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=STATIC_NUMERIC_ATOL,
                )
            else:
                agrees = np.array_equal(
                    left.astype(str).to_numpy(),
                    right.astype(str).to_numpy(),
                )
            if not agrees:
                delta_message = ""
                if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
                    right.dtype
                ):
                    delta = np.abs(
                        left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
                    )
                    delta_message = f", max_abs_delta={float(delta.max()):.12g}"
                raise RuntimeError(
                    f"Unchanged-card check failed for {column} between "
                    f"{reference['attempt_label'].iloc[0]} and "
                    f"{candidate['attempt_label'].iloc[0]}{delta_message}"
                )


def validation_report_audit() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    payloads: list[str] = []
    for attempt_label, csv_path in ATTEMPTS:
        report_path = csv_path.parent / "validation_report.json"
        entry: dict[str, Any] = {
            "attempt": attempt_label,
            "path": repo_path(report_path),
            "present": report_path.is_file(),
        }
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payloads.append(canonical)
            entry["sha256"] = sha256(report_path)
        entries.append(entry)
    identical = len(set(payloads)) <= 1
    if not identical:
        raise RuntimeError(
            "Attempt validation reports differ; inputs are not unchanged"
        )
    return {
        "reports": entries,
        "present_count": int(len(payloads)),
        "present_reports_identical": bool(identical),
    }


def safe_relative_delta(delta: float, reference: float) -> float:
    denominator = max(abs(float(reference)), np.finfo(float).tiny)
    return float(abs(float(delta)) / denominator)


def review_states(
    frames: Iterable[pd.DataFrame],
    raw_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stack = pd.concat(list(frames), ignore_index=True)
    ledger_rows: list[dict[str, Any]] = []
    reviewed_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []

    for dataset, mass_mev in expected_keys():
        group = stack.loc[
            (stack["dataset"] == dataset) & (stack["mass_MeV"] == mass_mev)
        ].sort_values("attempt_index")
        if len(group) != EXPECTED_ATTEMPT_COUNT:
            raise RuntimeError(
                f"{dataset} at {mass_mev} MeV has {len(group)} attempts; "
                f"expected {EXPECTED_ATTEMPT_COUNT}"
            )

        maximum_index = group["lml"].astype(float).idxmax()
        selected = group.loc[maximum_index]
        selected_lml = float(selected["lml"])
        selected_const = float(selected["const_opt"])
        selected_ls = float(selected["ls_opt"])
        delta_lml = group["lml"].astype(float) - selected_lml
        within = delta_lml.abs() <= LML_REPRODUCTION_ATOL
        branch_multiplicity = int(within.sum())
        resolved = branch_multiplicity >= 2
        review_status = (
            "resolved_reproduced_max_lml"
            if resolved
            else "unresolved_max_lml_not_reproduced"
        )

        reproducing = group.loc[within].sort_values("attempt_index")
        reproducing_labels = reproducing["attempt_label"].astype(str).tolist()
        reproducing_other_labels = [
            label
            for label in reproducing_labels
            if label != str(selected["attempt_label"])
        ]
        reproducing_sources = reproducing["attempt_source"].astype(str).tolist()

        for row_index, candidate in group.iterrows():
            candidate_delta_lml = float(candidate["lml"] - selected_lml)
            candidate_delta_const = float(candidate["const_opt"] - selected_const)
            candidate_delta_ls = float(candidate["ls_opt"] - selected_ls)
            ledger_rows.append(
                {
                    "dataset": dataset,
                    "mass_GeV": float(mass_mev / 1000.0),
                    "mass_MeV": int(mass_mev),
                    "attempt_index": int(candidate["attempt_index"]),
                    "attempt_label": str(candidate["attempt_label"]),
                    "attempt_source": str(candidate["attempt_source"]),
                    "attempt_source_sha256": str(
                        candidate["attempt_source_sha256"]
                    ),
                    "lml": float(candidate["lml"]),
                    "selected_max_lml": selected_lml,
                    "delta_lml_from_selected_max": candidate_delta_lml,
                    "abs_delta_lml_from_selected_max": abs(candidate_delta_lml),
                    "const_opt": float(candidate["const_opt"]),
                    "selected_const_opt": selected_const,
                    "delta_const_opt_from_selected_max": candidate_delta_const,
                    "abs_delta_const_opt_from_selected_max": abs(
                        candidate_delta_const
                    ),
                    "relative_abs_delta_const_opt_from_selected_max": (
                        safe_relative_delta(candidate_delta_const, selected_const)
                    ),
                    "ls_opt": float(candidate["ls_opt"]),
                    "selected_ls_opt": selected_ls,
                    "delta_ls_opt_from_selected_max": candidate_delta_ls,
                    "abs_delta_ls_opt_from_selected_max": abs(candidate_delta_ls),
                    "relative_abs_delta_ls_opt_from_selected_max": (
                        safe_relative_delta(candidate_delta_ls, selected_ls)
                    ),
                    "within_lml_reproduction_tolerance": bool(within.loc[row_index]),
                    "is_selected_maximum": bool(row_index == maximum_index),
                    "selected_attempt": str(selected["attempt_label"]),
                    "selected_source": str(selected["attempt_source"]),
                    "branch_multiplicity": branch_multiplicity,
                    "reproducing_attempts": "|".join(reproducing_labels),
                    "reproducing_other_attempts": "|".join(
                        reproducing_other_labels
                    ),
                    "reproducing_sources": "|".join(reproducing_sources),
                    "review_status": review_status,
                    "interpolated": False,
                }
            )

        if resolved:
            reviewed = {column: selected[column] for column in raw_columns}
            reviewed.update(
                {
                    "mass_MeV": int(mass_mev),
                    "selected_attempt": str(selected["attempt_label"]),
                    "selected_source": str(selected["attempt_source"]),
                    "selected_source_sha256": str(
                        selected["attempt_source_sha256"]
                    ),
                    "row_source": (
                        "unchanged_card_max_lml:"
                        + str(selected["attempt_source"])
                    ),
                    "optimizer_repair_applied": bool(
                        str(selected["attempt_label"]) != "attempt_01"
                    ),
                    "review_status": review_status,
                    "branch_multiplicity": branch_multiplicity,
                    "reproducing_attempts": "|".join(reproducing_labels),
                    "reproducing_other_attempts": "|".join(
                        reproducing_other_labels
                    ),
                    "reproducing_sources": "|".join(reproducing_sources),
                    "max_abs_reproducing_delta_lml": float(
                        np.max(np.abs(reproducing["lml"].astype(float) - selected_lml))
                    ),
                    "max_abs_reproducing_delta_const_opt": float(
                        np.max(
                            np.abs(
                                reproducing["const_opt"].astype(float)
                                - selected_const
                            )
                        )
                    ),
                    "max_abs_reproducing_delta_ls_opt": float(
                        np.max(
                            np.abs(
                                reproducing["ls_opt"].astype(float) - selected_ls
                            )
                        )
                    ),
                    "all_attempt_sources": "|".join(
                        group["attempt_source"].astype(str)
                    ),
                    "interpolated": False,
                }
            )
            reviewed_rows.append(reviewed)
        else:
            others = group.loc[group.index != maximum_index].copy()
            others["_abs_delta_lml"] = (
                others["lml"].astype(float) - selected_lml
            ).abs()
            nearest = others.sort_values(
                ["_abs_delta_lml", "attempt_index"]
            ).iloc[0]
            unresolved_rows.append(
                {
                    "dataset": dataset,
                    "mass_GeV": float(mass_mev / 1000.0),
                    "mass_MeV": int(mass_mev),
                    "reason": (
                        "maximum LML branch lacks a second unchanged-card "
                        f"attempt within abs(delta LML) <= {LML_REPRODUCTION_ATOL:g}"
                    ),
                    "selected_attempt": str(selected["attempt_label"]),
                    "selected_source": str(selected["attempt_source"]),
                    "selected_lml": selected_lml,
                    "selected_const_opt": selected_const,
                    "selected_ls_opt": selected_ls,
                    "branch_multiplicity": branch_multiplicity,
                    "nearest_other_attempt": str(nearest["attempt_label"]),
                    "nearest_other_source": str(nearest["attempt_source"]),
                    "nearest_abs_delta_lml": float(nearest["_abs_delta_lml"]),
                    "nearest_delta_const_opt": float(
                        nearest["const_opt"] - selected_const
                    ),
                    "nearest_delta_ls_opt": float(
                        nearest["ls_opt"] - selected_ls
                    ),
                    "attempt_lml_by_label": json.dumps(
                        dict(
                            zip(
                                group["attempt_label"].astype(str),
                                group["lml"].astype(float),
                            )
                        ),
                        sort_keys=True,
                    ),
                    "attempt_const_opt_by_label": json.dumps(
                        dict(
                            zip(
                                group["attempt_label"].astype(str),
                                group["const_opt"].astype(float),
                            )
                        ),
                        sort_keys=True,
                    ),
                    "attempt_ls_opt_by_label": json.dumps(
                        dict(
                            zip(
                                group["attempt_label"].astype(str),
                                group["ls_opt"].astype(float),
                            )
                        ),
                        sort_keys=True,
                    ),
                    "attempt_sources": "|".join(
                        group["attempt_source"].astype(str)
                    ),
                    "interpolated": False,
                }
            )

    ledger = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS)
    reviewed_columns = list(raw_columns) + [
        column
        for column in REVIEW_PROVENANCE_COLUMNS
        if column not in raw_columns
    ]
    reviewed = pd.DataFrame(reviewed_rows, columns=reviewed_columns)
    unresolved = pd.DataFrame(unresolved_rows, columns=UNRESOLVED_COLUMNS)
    ledger = sort_states(ledger)
    if not reviewed.empty:
        reviewed["_dataset_order"] = reviewed["dataset"].map(DATASET_ORDER)
        reviewed = reviewed.sort_values(["_dataset_order", "mass_MeV"])
        reviewed = reviewed.drop(columns="_dataset_order").reset_index(drop=True)
    if not unresolved.empty:
        unresolved["_dataset_order"] = unresolved["dataset"].map(DATASET_ORDER)
        unresolved = unresolved.sort_values(["_dataset_order", "mass_MeV"])
        unresolved = unresolved.drop(columns="_dataset_order").reset_index(
            drop=True
        )
    return ledger, reviewed, unresolved


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    missing_attempts = [path for _, path in ATTEMPTS if not path.is_file()]
    if missing_attempts:
        missing_text = "\n".join(f"  - {path}" for path in missing_attempts)
        raise SystemExit(
            "Review not run: all three complete attempts are required. Missing:\n"
            + missing_text
        )

    frames: list[pd.DataFrame] = []
    attempt_metadata: list[dict[str, Any]] = []
    raw_columns: list[str] | None = None
    for attempt_index, (attempt_label, path) in enumerate(ATTEMPTS, start=1):
        frame, metadata = load_attempt(attempt_index, attempt_label, path)
        current_raw_columns = [
            column
            for column in frame.columns
            if column
            not in {
                "mass_MeV",
                "attempt_index",
                "attempt_label",
                "attempt_source",
                "attempt_source_sha256",
            }
        ]
        if raw_columns is None:
            raw_columns = current_raw_columns
        elif current_raw_columns != raw_columns:
            raise RuntimeError(
                f"{attempt_label} results schema differs from attempt_01"
            )
        frames.append(frame)
        attempt_metadata.append(metadata)

    if raw_columns is None:
        raise RuntimeError("No attempt schemas were loaded")
    compare_unchanged_card_columns(frames)
    validation_reports = validation_report_audit()
    ledger, reviewed, unresolved = review_states(frames, raw_columns)

    if len(ledger) != EXPECTED_STATE_COUNT * EXPECTED_ATTEMPT_COUNT:
        raise RuntimeError(
            f"Reviewer ledger has {len(ledger)} rows; expected "
            f"{EXPECTED_STATE_COUNT * EXPECTED_ATTEMPT_COUNT}"
        )
    if bool(ledger["interpolated"].any()):
        raise RuntimeError("Reviewer ledger unexpectedly contains interpolation")
    if not reviewed.empty and bool(reviewed["interpolated"].any()):
        raise RuntimeError("Reviewed state table unexpectedly contains interpolation")

    passed = unresolved.empty and len(reviewed) == EXPECTED_STATE_COUNT
    DERIVED.mkdir(parents=True, exist_ok=True)
    ledger_path = DERIVED / "observed_attempt_ledger.csv"
    reviewed_path = DERIVED / "observed_gp_states_reviewed.csv"
    unresolved_path = DERIVED / "unresolved_observed_states.csv"
    summary_path = DERIVED / "observed_review_summary.json"
    write_csv_atomic(ledger, ledger_path)
    write_csv_atomic(reviewed, reviewed_path)
    write_csv_atomic(unresolved, unresolved_path)

    state_ledger = ledger.drop_duplicates(["dataset", "mass_MeV"])
    multiplicity_counts = {
        str(int(key)): int(value)
        for key, value in state_ledger["branch_multiplicity"]
        .value_counts()
        .sort_index()
        .items()
    }
    reproducing_rows = ledger.loc[
        ledger["within_lml_reproduction_tolerance"].astype(bool)
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": CAMPAIGN_DIR.name,
        "status": "passed" if passed else "unresolved",
        "all_review_requirements_passed": bool(passed),
        "selection_rule": (
            "retain the exact maximum-LML input row per dataset/mass; require "
            "at least one other complete unchanged-card attempt within "
            "abs(delta LML) <= 2e-5"
        ),
        "lml_reproduction_atol": LML_REPRODUCTION_ATOL,
        "static_numeric_atol": STATIC_NUMERIC_ATOL,
        "interpolation_used": False,
        "expected_states_per_attempt": EXPECTED_STATE_COUNT,
        "expected_grid_MeV": {
            dataset: {
                "low": int(masses[0]),
                "high": int(masses[-1]),
                "step": 1,
                "rows": int(len(masses)),
            }
            for dataset, masses in EXPECTED_MASS_MEV.items()
        },
        "attempts": attempt_metadata,
        "attempt_count": int(len(frames)),
        "unchanged_card_columns_checked": list(UNCHANGED_CARD_COLUMNS),
        "validation_report_audit": validation_reports,
        "physics_config": repo_path(CONFIG),
        "physics_config_sha256": sha256(CONFIG) if CONFIG.is_file() else None,
        "ledger_rows": int(len(ledger)),
        "reviewed_state_rows": int(len(reviewed)),
        "unresolved_state_rows": int(len(unresolved)),
        "branch_multiplicity_counts": multiplicity_counts,
        "max_abs_reproducing_delta_lml": (
            float(
                reproducing_rows[
                    "abs_delta_lml_from_selected_max"
                ].max()
            )
            if not reproducing_rows.empty
            else None
        ),
        "max_abs_reproducing_delta_const_opt": (
            float(
                reproducing_rows[
                    "abs_delta_const_opt_from_selected_max"
                ].max()
            )
            if not reproducing_rows.empty
            else None
        ),
        "max_abs_reproducing_delta_ls_opt": (
            float(
                reproducing_rows[
                    "abs_delta_ls_opt_from_selected_max"
                ].max()
            )
            if not reproducing_rows.empty
            else None
        ),
        "selected_source_counts": (
            {
                str(key): int(value)
                for key, value in reviewed["selected_attempt"]
                .value_counts()
                .sort_index()
                .items()
            }
            if not reviewed.empty
            else {}
        ),
        "outputs": {
            "attempt_ledger": {
                "path": repo_path(ledger_path),
                "sha256": sha256(ledger_path),
                "rows": int(len(ledger)),
            },
            "reviewed_gp_states": {
                "path": repo_path(reviewed_path),
                "sha256": sha256(reviewed_path),
                "rows": int(len(reviewed)),
                "complete_415_state_authority": bool(passed),
            },
            "unresolved_states": {
                "path": repo_path(unresolved_path),
                "sha256": sha256(unresolved_path),
                "rows": int(len(unresolved)),
            },
        },
        "stitched_observed_campaign": {
            "written": False,
            "expected_rows": 232,
            "reason": STITCHING_REASON,
            "exact_authority": repo_path(
                CAMPAIGN_DIR / "run_combined_bands_cached_fixed_reviewed.py"
            ),
        },
    }
    write_json_atomic(summary, summary_path)

    if not passed:
        raise SystemExit(
            f"Observed review unresolved: {len(unresolved)} of "
            f"{EXPECTED_STATE_COUNT} maximum-LML states lack repeat closure. "
            f"See {unresolved_path}."
        )
    print(
        f"Reviewed {len(reviewed)} GP states from {len(frames)} complete "
        f"unchanged-card attempts. Wrote {reviewed_path}."
    )


if __name__ == "__main__":
    main()
