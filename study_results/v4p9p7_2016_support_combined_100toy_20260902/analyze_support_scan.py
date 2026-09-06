#!/usr/bin/env python3
"""Apply the frozen phase-1 v4.9.7 support-selection rule.

This script consumes only the predeclared 2016-full toy extractions at 44,
49, 54, and 59 MeV.  It never opens the full observed spectrum and has no
code path for the 65 MeV holdout.
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
SUPPORTS = (
    "028_210",
    "029_210",
    "030_210",
    "031_210",
    "032_210",
    "033_210",
    "034_210",
)
ELIGIBLE = SUPPORTS[:-1]
SCENARIO = "2016_full"
MASSES = (0.044, 0.049, 0.054, 0.059)
STRENGTHS = (0.0, 2.0, 5.0)
HOLDOUT_MASS_GEV = 0.065
INITIAL = range(0, 25)
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


class AnalysisError(RuntimeError):
    """Raised when a frozen phase-1 input or rule is violated."""


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
        raise AnalysisError(f"missing or hash-invalid {label}: {path}")


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
    if str(spec.get("study_id")) != "v4p9p7_2016_support_combined_100toy_20260902":
        raise AnalysisError("study_id drift")
    script = spec.get("workflow_scripts", {}).get("analyze_support_scan", {})
    require_hash(Path(__file__).resolve(), str(script.get("sha256", "")), "analysis script")
    protocol = spec.get("frozen_protocol", {})
    require_hash(HERE / str(protocol.get("path", "")), str(protocol.get("sha256", "")), "frozen protocol")
    card = spec.get("analysis_card", {})
    supports = tuple(
        f"{int(round(1000.0 * float(edge))):03d}_210"
        for edge in card.get("candidate_gp_support_low_edges_gev", ())
    )
    eligible = tuple(
        f"{int(round(1000.0 * float(edge))):03d}_210"
        for edge in card.get("eligible_freeze_low_edges_gev", ())
    )
    if supports != SUPPORTS or eligible != ELIGIBLE:
        raise AnalysisError("support grid or eligibility drift")
    masses = tuple(float(value) for value in spec.get("masses_gev", ()))
    strengths = tuple(float(value) for value in spec.get("sigma_strengths", ()))
    if masses != MASSES or strengths != STRENGTHS:
        raise AnalysisError("selection mass/strength grid drift")
    if not math.isclose(
        float(spec.get("holdout_mass_gev", math.nan)),
        HOLDOUT_MASS_GEV,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or HOLDOUT_MASS_GEV in masses:
        raise AnalysisError("65 MeV holdout contract drift")
    cohort = spec.get("cohorts", {}).get("phase1", {})
    if tuple(int(cohort.get(key, -1)) for key in ("start", "stop_exclusive", "n")) != (0, 25, 25):
        raise AnalysisError("phase-1 cohort drift")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def expected_cell_keys() -> set[tuple[float, float]]:
    return {(mass, strength) for mass in MASSES for strength in STRENGTHS}


def read_rows(spec: Mapping[str, Any], support: str) -> pd.DataFrame:
    spec_sha = sha256_file(SPEC_PATH)
    product = spec["background_toy_product"]
    root = HERE / "runs" / f"2016_threshold_qualified_{support}" / SCENARIO
    frames: list[pd.DataFrame] = []
    for index in INITIAL:
        task = root / f"toy_{index:04d}"
        marker_path = task / "_SUCCESS.json"
        if not marker_path.is_file():
            raise AnalysisError(f"missing phase-1 task for {support} toy {index:04d}")
        marker = load_json(marker_path)
        if (
            marker.get("status") != "pass"
            or marker.get("scenario") != SCENARIO
            or int(marker.get("toy_index", -1)) != index
            or marker.get("study_spec_sha256") != spec_sha
            or marker.get("background_toy_root_sha256") != product.get("root_sha256")
            or marker.get("background_toy_manifest_sha256") != product.get("manifest_sha256")
        ):
            raise AnalysisError(f"invalid phase-1 marker for {support} toy {index:04d}")
        declared = marker.get("ledger_sha256", {})
        if set(declared) != {
            "optimizer_attempts.csv",
            "accepted_rows.csv",
            "raw_primary_rows.csv",
            "exclusions.csv",
        }:
            raise AnalysisError(f"incomplete ledger declaration for {support} toy {index:04d}")
        for name, expected_hash in declared.items():
            require_hash(task / name, str(expected_hash), f"{support} toy {index:04d} {name}")
        raw = _read_csv(task / "raw_primary_rows.csv")
        if len(raw) != 12 or raw.duplicated(KEYS).any():
            raise AnalysisError(f"raw phase-1 inventory drift for {support} toy {index:04d}")
        raw_cells = set(zip(raw["mass_GeV"].astype(float), raw["inj_nsigma"].astype(float)))
        if raw_cells != expected_cell_keys():
            raise AnalysisError(f"raw mass/strength drift for {support} toy {index:04d}")
        accepted = _read_csv(task / "accepted_rows.csv")
        if not accepted.empty:
            if accepted.duplicated(KEYS).any():
                raise AnalysisError(f"duplicate accepted state for {support} toy {index:04d}")
            if set(accepted["scenario"].astype(str)) != {SCENARIO}:
                raise AnalysisError(f"scenario drift for {support} toy {index:04d}")
            if any(np.isclose(accepted["mass_GeV"].astype(float), HOLDOUT_MASS_GEV)):
                raise AnalysisError("65 MeV holdout leaked into phase-1 selection")
            if not set(zip(accepted["mass_GeV"].astype(float), accepted["inj_nsigma"].astype(float))).issubset(expected_cell_keys()):
                raise AnalysisError(f"accepted mass/strength drift for {support} toy {index:04d}")
            if set(accepted["gp_support_mode"].astype(str)) != {support}:
                raise AnalysisError(f"support label drift for {support} toy {index:04d}")
            accepted = accepted.copy()
            accepted["support"] = support
            accepted["cohort"] = "initial_0_24"
            frames.append(accepted)
    if not frames:
        return pd.DataFrame(columns=KEYS + ["support", "cohort"])
    result = pd.concat(frames, ignore_index=True, sort=False)
    if result.duplicated(KEYS).any():
        raise AnalysisError(f"duplicate accepted extraction key in {support}")
    return result


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


def summarize_cells(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for support in SUPPORTS:
        for mass in MASSES:
            for strength in STRENGTHS:
                group = rows.loc[
                    (rows.get("support") == support)
                    & np.isclose(pd.to_numeric(rows.get("mass_GeV"), errors="coerce"), mass)
                    & np.isclose(pd.to_numeric(rows.get("inj_nsigma"), errors="coerce"), strength)
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
                        "cohort": "initial_0_24",
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


def initial_selection(
    spec: Mapping[str, Any], cells: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rule = spec["support_selection_protocol"]
    records: list[dict[str, Any]] = []
    for support in SUPPORTS:
        group = cells.loc[cells["support"] == support]
        means = group["mean_pull"].abs()
        zero = group.loc[group["inj_nsigma"] == 0.0, "mean_pull"].abs()
        technical = bool(
            len(group) == 12
            and (group["n"] == 25).all()
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
                "support_low_MeV": int(support[:3]),
                "eligible_edge": support in ELIGIBLE,
                "geometry_control": support == "034_210",
                "technical_gate_pass": technical,
                "cells_below_abs_mean_pull_0p75": int((means < 0.75).sum()),
                "zero_signal_cells_below_abs_mean_pull_0p75": int((zero < 0.75).sum()),
                "gross_bias_guard_pass": bool((means < float(rule["gross_abs_mean_pull_limit"])).all()),
                "practical_acceptability_pass": practical,
                "worst_abs_mean_pull": float(means.max()),
                "worst_abs_zero_signal_mean_pull": float(zero.max()),
                "absolute_upper_limit_used_for_ranking": False,
            }
        )
    summary = pd.DataFrame(records)
    qualified = summary.loc[
        summary["eligible_edge"] & summary["practical_acceptability_pass"]
    ].copy()
    if qualified.empty:
        return summary, {
            "status": "no_provisional_edge",
            "reason": "no eligible edge passed the frozen phase-1 practical rule",
            "phase2_supports": [],
            "observed_scan_authorized": False,
            "holdout_evaluated": False,
        }
    minimum = float(qualified["worst_abs_mean_pull"].min())
    tied = qualified.loc[
        qualified["worst_abs_mean_pull"]
        <= minimum + float(rule["minimax_tie_margin"])
    ].sort_values("support_low_MeV")
    selected = str(tied.iloc[0]["support"])
    selected_index = SUPPORTS.index(selected)
    phase2 = [selected]
    if selected_index > 0:
        phase2.append(SUPPORTS[selected_index - 1])
    if selected_index + 1 < len(SUPPORTS):
        phase2.append(SUPPORTS[selected_index + 1])
    phase2 = [support for support in SUPPORTS if support in phase2]
    return summary, {
        "status": "provisional_edge_selected",
        "provisional_support": selected,
        "provisional_support_low_MeV": int(selected[:3]),
        "support_high_MeV": 210,
        "primary_minimum_worst_abs_mean_pull": minimum,
        "tie_margin": float(rule["minimax_tie_margin"]),
        "tied_supports": tied["support"].tolist(),
        "phase2_supports": phase2,
        "observed_scan_authorized": False,
        "holdout_evaluated": False,
        "absolute_upper_limit_used_for_ranking": False,
        "reason": "phase-1 toy-pull selection only; independent continuation required",
    }


def paired_adjacent(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metric = "A_up_wald90_minus_injected_over_sigmaA_ref"
    for lower, higher in zip(SUPPORTS[:-1], SUPPORTS[1:]):
        left = rows.loc[rows["support"] == lower, KEYS + ["pull", metric]]
        right = rows.loc[rows["support"] == higher, KEYS + ["pull", metric]]
        merged = left.merge(
            right,
            on=KEYS,
            suffixes=("_lower", "_higher"),
            validate="one_to_one",
        )
        for (mass, strength), group in merged.groupby(
            ["mass_GeV", "inj_nsigma"], sort=True
        ):
            pull_delta = group["pull_higher"] - group["pull_lower"]
            limit_delta = group[f"{metric}_higher"] - group[f"{metric}_lower"]
            mean, width, low, high, n = interval90(pull_delta)
            records.append(
                {
                    "lower_support": lower,
                    "higher_support": higher,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "inj_nsigma": float(strength),
                    "n_pairs": n,
                    "mean_pull_difference_higher_minus_lower": mean,
                    "pull_difference_sd": width,
                    "pull_difference_ci90_low": low,
                    "pull_difference_ci90_high": high,
                    "median_limit_excess_difference_higher_minus_lower": float(limit_delta.median()) if n else math.nan,
                    "median_abs_paired_limit_excess_difference": float(limit_delta.abs().median()) if n else math.nan,
                    "used_for_support_ranking": False,
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    spec = load_json(SPEC_PATH)
    validate_static_contract(spec)
    rows = pd.concat(
        [read_rows(spec, support) for support in SUPPORTS],
        ignore_index=True,
        sort=False,
    )
    if any(np.isclose(pd.to_numeric(rows["mass_GeV"]), HOLDOUT_MASS_GEV)):
        raise AnalysisError("65 MeV holdout leaked into phase-1 aggregate")
    cells = summarize_cells(rows)
    supports, decision = initial_selection(spec, cells)
    paired = paired_adjacent(rows)
    products = {
        "phase1_accepted_rows.csv": rows,
        "phase1_cell_summary.csv": cells,
        "phase1_support_summary.csv": supports,
        "phase1_adjacent_paired_differences.csv": paired,
    }
    for name, frame in products.items():
        atomic_csv(OUT / name, frame)
    decision.update(
        {
            "study_id": spec["study_id"],
            "study_spec_sha256": sha256_file(SPEC_PATH),
            "frozen_protocol_sha256": spec["frozen_protocol"]["sha256"],
            "selection_grid_masses_MeV": [44, 49, 54, 59],
            "excluded_holdout_mass_MeV": 65,
            "selection_basis": "Frozen practical toy-pull rule; optimizer branches are pull-blind and observed data are not opened.",
            "claim_boundary": "Source-conditioned injection-recovery diagnostic, not coverage or expected-limit calibration.",
            "products": {
                name: {"rows": int(len(frame)), "sha256": sha256_file(OUT / name)}
                for name, frame in products.items()
            },
        }
    )
    atomic_json(OUT / "phase1_selection_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["status"] == "provisional_edge_selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
