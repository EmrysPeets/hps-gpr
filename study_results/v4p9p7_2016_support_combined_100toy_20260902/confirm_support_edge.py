#!/usr/bin/env python3
"""Confirm the phase-1 2016 support choice on toys 25--99.

The provisional edge is read from the hash-checked phase-1 decision.  Failure
does not choose another edge.  The observed spectrum and 65 MeV holdout are
authorized only when both continuation and full-100 gates pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "study_spec.json"
OUT = HERE / "derived" / "analysis"
PHASE1_DECISION = OUT / "phase1_selection_decision.json"
ALL_SUPPORTS = (
    "028_210",
    "029_210",
    "030_210",
    "031_210",
    "032_210",
    "033_210",
    "034_210",
)
ELIGIBLE = ALL_SUPPORTS[:-1]
SCENARIO = "2016_full"
MASSES = (0.044, 0.049, 0.054, 0.059)
STRENGTHS = (0.0, 2.0, 5.0)
HOLDOUT_MASS_GEV = 0.065
KEYS = ["scenario", "background_toy_index", "mass_GeV", "inj_nsigma"]
FINITE_COLUMNS = [
    "A_hat",
    "sigma_A",
    "pull",
    "A_up_wald90",
    "eps2_up_wald90",
    "A_up_wald90_minus_injected_over_sigmaA_ref",
]
BOUND_COLUMNS = [
    "ls_at_lower",
    "ls_at_upper",
    "const_at_lower",
    "const_at_upper",
    "refit_lower_boundary",
    "refit_upper_boundary",
    "refit_constant_lower_boundary",
    "refit_constant_upper_boundary",
]


class ConfirmationError(RuntimeError):
    """Raised when confirmation inputs violate the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ConfirmationError(f"missing or hash-invalid {label}: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_static_contract(spec: Mapping[str, Any]) -> None:
    script = spec.get("workflow_scripts", {}).get("confirm_support_edge", {})
    require_hash(Path(__file__).resolve(), str(script.get("sha256", "")), "confirmation script")
    protocol = spec.get("frozen_protocol", {})
    require_hash(HERE / str(protocol.get("path", "")), str(protocol.get("sha256", "")), "frozen protocol")
    cohort = spec.get("cohorts", {}).get("phase2_continuation", {})
    if tuple(int(cohort.get(key, -1)) for key in ("start", "stop_exclusive", "n")) != (25, 100, 75):
        raise ConfirmationError("phase-2 continuation cohort drift")
    if tuple(float(value) for value in spec.get("masses_gev", ())) != MASSES:
        raise ConfirmationError("selection mass grid drift")
    if tuple(float(value) for value in spec.get("sigma_strengths", ())) != STRENGTHS:
        raise ConfirmationError("selection strength grid drift")
    if not math.isclose(
        float(spec.get("holdout_mass_gev", math.nan)),
        HOLDOUT_MASS_GEV,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ConfirmationError("holdout declaration drift")


def expected_neighbors(selected: str) -> list[str]:
    position = ALL_SUPPORTS.index(selected)
    values = [selected]
    if position > 0:
        values.append(ALL_SUPPORTS[position - 1])
    if position + 1 < len(ALL_SUPPORTS):
        values.append(ALL_SUPPORTS[position + 1])
    return [support for support in ALL_SUPPORTS if support in values]


def load_phase1_decision(spec: Mapping[str, Any]) -> tuple[dict[str, Any], str, list[str]]:
    decision = load_json(PHASE1_DECISION)
    if (
        decision.get("status") != "provisional_edge_selected"
        or decision.get("study_id") != spec.get("study_id")
        or decision.get("study_spec_sha256") != sha256_file(SPEC_PATH)
        or decision.get("frozen_protocol_sha256")
        != spec.get("frozen_protocol", {}).get("sha256")
        or bool(decision.get("observed_scan_authorized"))
        or bool(decision.get("holdout_evaluated"))
    ):
        raise ConfirmationError("phase-1 decision is missing or contract-invalid")
    for name, record in decision.get("products", {}).items():
        require_hash(OUT / name, str(record.get("sha256", "")), f"phase-1 product {name}")
    selected = str(decision.get("provisional_support", ""))
    if selected not in ELIGIBLE:
        raise ConfirmationError("phase-1 selected an ineligible support")
    supports = [str(value) for value in decision.get("phase2_supports", ())]
    if supports != expected_neighbors(selected):
        raise ConfirmationError("phase-2 neighbor set drift")
    return decision, selected, supports


def load_support(spec: Mapping[str, Any], support: str) -> pd.DataFrame:
    directory = HERE / "derived" / f"2016_threshold_qualified_{support}"
    summary_path = directory / "collection_summary.json"
    if not summary_path.is_file():
        raise ConfirmationError(f"collect the complete {support} lane first")
    summary = load_json(summary_path)
    if (
        summary.get("status") != "pass"
        or summary.get("study_spec_sha256") != sha256_file(SPEC_PATH)
        or int(summary.get("raw_rows", -1)) != 1200
        or int(summary.get("summary_cells", -1)) != 12
        or str(summary.get("gp_support_mode")) != support
    ):
        raise ConfirmationError(f"invalid collection summary for {support}")
    hashes = summary.get("derived_sha256", {})
    for name, expected_hash in hashes.items():
        require_hash(directory / name, str(expected_hash), f"{support} collected {name}")
    raw = pd.read_csv(directory / "raw_primary_extraction_rows.csv")
    if len(raw) != 1200 or raw.duplicated(KEYS).any():
        raise ConfirmationError(f"raw full-100 inventory drift for {support}")
    accepted_path = directory / "accepted_extraction_rows.csv"
    try:
        frame = pd.read_csv(accepted_path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=KEYS)
    if frame.duplicated(KEYS).any():
        raise ConfirmationError(f"duplicate accepted state for {support}")
    if not frame.empty:
        if set(frame["scenario"].astype(str)) != {SCENARIO}:
            raise ConfirmationError(f"scenario drift for {support}")
        if set(frame["gp_support_mode"].astype(str)) != {support}:
            raise ConfirmationError(f"support label drift for {support}")
        if any(np.isclose(pd.to_numeric(frame["mass_GeV"]), HOLDOUT_MASS_GEV)):
            raise ConfirmationError("65 MeV holdout leaked into confirmation")
        valid_cells = {(mass, strength) for mass in MASSES for strength in STRENGTHS}
        actual_cells = set(
            zip(frame["mass_GeV"].astype(float), frame["inj_nsigma"].astype(float))
        )
        if not actual_cells.issubset(valid_cells):
            raise ConfirmationError(f"mass/strength drift for {support}")
    frame = frame.copy()
    frame["support"] = support
    return frame


def interval90(values: pd.Series) -> tuple[float, float, float, float, int]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    mean = float(np.mean(array)) if n else math.nan
    if n < 2:
        return mean, math.nan, math.nan, math.nan, n
    width = float(np.std(array, ddof=1))
    half = float(t.ppf(0.95, n - 1) * width / math.sqrt(n))
    return mean, width, mean - half, mean + half, n


def summarize_cells(rows: pd.DataFrame, supports: list[str]) -> pd.DataFrame:
    cohort_masks = {
        "initial_0_24": lambda frame: frame["background_toy_index"] < 25,
        "continuation_25_99": lambda frame: frame["background_toy_index"] >= 25,
        "full_0_99": lambda frame: np.ones(len(frame), dtype=bool),
    }
    records: list[dict[str, Any]] = []
    for support in supports:
        support_rows = rows.loc[rows["support"] == support]
        for cohort, mask_builder in cohort_masks.items():
            subset = support_rows.loc[mask_builder(support_rows)]
            for mass in MASSES:
                for strength in STRENGTHS:
                    group = subset.loc[
                        np.isclose(pd.to_numeric(subset["mass_GeV"]), mass)
                        & np.isclose(pd.to_numeric(subset["inj_nsigma"]), strength)
                    ]
                    mean, width, low, high, n = interval90(
                        group["pull"] if "pull" in group else pd.Series(dtype=float)
                    )
                    finite = bool(
                        n == len(group)
                        and all(column in group for column in FINITE_COLUMNS)
                        and np.isfinite(group[FINITE_COLUMNS].to_numpy(float)).all()
                    ) if len(group) else False
                    bounds = bool(
                        len(group)
                        and all(column in group for column in BOUND_COLUMNS)
                        and group[BOUND_COLUMNS].astype(bool).to_numpy().any()
                    )
                    covariance = bool(
                        len(group)
                        and "covariance_valid" in group
                        and group["covariance_valid"].astype(bool).all()
                    )
                    records.append(
                        {
                            "support": support,
                            "support_low_MeV": int(support[:3]),
                            "cohort": cohort,
                            "mass_GeV": mass,
                            "mass_MeV": 1000.0 * mass,
                            "inj_nsigma": strength,
                            "n": n,
                            "mean_pull": mean,
                            "pull_width": width,
                            "mean_pull_ci90_low": low,
                            "mean_pull_ci90_high": high,
                            "all_finite": finite,
                            "any_kernel_bound": bounds,
                            "all_covariance_valid": covariance,
                            "median_A_up_wald90": float(group["A_up_wald90"].median()) if len(group) else math.nan,
                            "median_A_up_wald90_over_sigmaA_ref": float(group["A_up_wald90_over_sigmaA_ref"].median()) if len(group) else math.nan,
                            "median_A_up_minus_injected_over_sigmaA_ref": float(group["A_up_wald90_minus_injected_over_sigmaA_ref"].median()) if len(group) else math.nan,
                        }
                    )
    return pd.DataFrame(records)


def summarize_supports(
    spec: Mapping[str, Any], cells: pd.DataFrame
) -> pd.DataFrame:
    rule = spec["support_selection_protocol"]
    records: list[dict[str, Any]] = []
    for (support, cohort), group in cells.groupby(["support", "cohort"], sort=True):
        expected_n = {
            "initial_0_24": 25,
            "continuation_25_99": 75,
            "full_0_99": 100,
        }[str(cohort)]
        required_n = (
            int(rule["minimum_full100_accepted_per_cell"])
            if cohort == "full_0_99"
            else expected_n
        )
        means = group["mean_pull"].abs()
        zero = group.loc[group["inj_nsigma"] == 0.0, "mean_pull"].abs()
        technical = bool(
            len(group) == 12
            and (group["n"] >= required_n).all()
            and group["all_finite"].all()
            and group["all_covariance_valid"].all()
            and not group["any_kernel_bound"].any()
        )
        practical = bool(
            technical
            and int((means < 0.75).sum()) >= int(rule["phase1_min_cells_below_abs_mean_pull_0p75"])
            and int((zero < 0.75).sum()) >= int(rule["phase1_min_zero_cells_below_abs_mean_pull_0p75"])
            and bool((means < float(rule["gross_abs_mean_pull_limit"])).all())
        )
        records.append(
            {
                "support": support,
                "support_low_MeV": int(str(support)[:3]),
                "cohort": cohort,
                "minimum_required_per_cell": required_n,
                "technical_gate_pass": technical,
                "cells_below_abs_mean_pull_0p75": int((means < 0.75).sum()),
                "zero_signal_cells_below_abs_mean_pull_0p75": int((zero < 0.75).sum()),
                "worst_abs_mean_pull": float(means.max()),
                "gross_bias_guard_pass": bool((means < float(rule["gross_abs_mean_pull_limit"])).all()),
                "practical_acceptability_pass": practical,
            }
        )
    return pd.DataFrame(records)


def paired_limits(rows: pd.DataFrame, selected: str, supports: list[str]) -> pd.DataFrame:
    selected_rows = rows.loc[rows["support"] == selected]
    metric = "A_up_wald90_minus_injected_over_sigmaA_ref"
    records: list[dict[str, Any]] = []
    for neighbor in (support for support in supports if support != selected):
        other = rows.loc[rows["support"] == neighbor]
        merged = selected_rows[KEYS + ["pull", metric]].merge(
            other[KEYS + ["pull", metric]],
            on=KEYS,
            suffixes=("_selected", "_neighbor"),
            validate="one_to_one",
        )
        for (mass, strength), group in merged.groupby(
            ["mass_GeV", "inj_nsigma"], sort=True
        ):
            pull_delta = group["pull_selected"] - group["pull_neighbor"]
            limit_delta = group[f"{metric}_selected"] - group[f"{metric}_neighbor"]
            mean, width, low, high, n = interval90(pull_delta)
            records.append(
                {
                    "selected_support": selected,
                    "neighbor_support": neighbor,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "inj_nsigma": float(strength),
                    "n_pairs": n,
                    "mean_pull_difference_selected_minus_neighbor": mean,
                    "pull_difference_sd": width,
                    "pull_difference_ci90_low": low,
                    "pull_difference_ci90_high": high,
                    "median_limit_excess_difference_selected_minus_neighbor": float(limit_delta.median()) if n else math.nan,
                    "median_abs_paired_limit_excess_difference": float(limit_delta.abs().median()) if n else math.nan,
                    "used_for_support_ranking": False,
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    spec = load_json(SPEC_PATH)
    validate_static_contract(spec)
    phase1, selected, supports = load_phase1_decision(spec)
    rows = pd.concat(
        [load_support(spec, support) for support in supports],
        ignore_index=True,
        sort=False,
    )
    if any(np.isclose(pd.to_numeric(rows["mass_GeV"]), HOLDOUT_MASS_GEV)):
        raise ConfirmationError("65 MeV holdout leaked into confirmation aggregate")
    cells = summarize_cells(rows, supports)
    support_summary = summarize_supports(spec, cells)
    paired = paired_limits(rows, selected, supports)
    selected_summary = support_summary.loc[support_summary["support"] == selected]
    initial_pass = bool(
        selected_summary.loc[
            selected_summary["cohort"] == "initial_0_24",
            "practical_acceptability_pass",
        ].iloc[0]
    )
    continuation_pass = bool(
        selected_summary.loc[
            selected_summary["cohort"] == "continuation_25_99",
            "practical_acceptability_pass",
        ].iloc[0]
    )
    full_pass = bool(
        selected_summary.loc[
            selected_summary["cohort"] == "full_0_99",
            "practical_acceptability_pass",
        ].iloc[0]
    )
    frozen = initial_pass and continuation_pass and full_pass
    products = {
        "full100_accepted_rows_selected_neighbors.csv": rows,
        "confirmation_cell_summary.csv": cells,
        "confirmation_support_summary.csv": support_summary,
        "confirmation_paired_limit_differences.csv": paired,
    }
    for name, frame in products.items():
        atomic_csv(OUT / name, frame)
    selected_low = int(selected[:3])
    decision = {
        "status": "support_edge_frozen" if frozen else "support_edge_confirmation_failed",
        "study_id": spec["study_id"],
        "study_spec_sha256": sha256_file(SPEC_PATH),
        "frozen_protocol_sha256": spec["frozen_protocol"]["sha256"],
        "phase1_decision_sha256": sha256_file(PHASE1_DECISION),
        "selected_support": selected,
        "selected_support_low_MeV": selected_low,
        "support_high_MeV": 210,
        "data_range_2016": [selected_low / 1000.0, 0.210],
        "phase2_supports": supports,
        "initial_gate_pass": initial_pass,
        "continuation_gate_pass": continuation_pass,
        "full100_gate_pass": full_pass,
        "observed_scan_authorized": frozen,
        "holdout_65MeV_authorized": frozen,
        "holdout_65MeV_used_for_selection": False,
        "absolute_upper_limit_used_for_selection": False,
        "retuning_after_confirmation": False,
        "selection_basis": "Frozen practical toy-pull rule with the smaller-edge 0.10 minimax tie break; no observed-data or absolute-limit ranking.",
        "claim_boundary": "Source-conditioned recovery criterion, not coverage, expected-band calibration, exclusion, or global significance.",
        "products": {
            name: {"rows": int(len(frame)), "sha256": sha256_file(OUT / name)}
            for name, frame in products.items()
        },
    }
    atomic_json(OUT / "support_freeze_decision.json", decision)
    print(support_summary.to_string(index=False))
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if frozen else 2


if __name__ == "__main__":
    raise SystemExit(main())
