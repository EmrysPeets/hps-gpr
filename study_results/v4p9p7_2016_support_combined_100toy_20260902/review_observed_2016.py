#!/usr/bin/env python3
"""Plan and perform pull-blind, no-interpolation review of the 2016 scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from observed_2016_contract import (
    CARD,
    CARD_MANIFEST,
    FREEZE,
    HERE,
    PRIMARY,
    REPAIR_LEDGER,
    REPAIR_PLAN,
    REPEAT_ROOT,
    REVIEWED_CSV,
    REVIEW_SUMMARY,
    STUDY_ID,
    STUDY_SPEC,
    ObservedContractError,
    atomic_csv,
    atomic_json,
    bool_value,
    branch_match,
    candidate_issue_reasons,
    eligible_candidate,
    load_json,
    sha256,
    static_preflight,
    validate_card,
    validate_freeze,
    validate_mass_grid,
)
from run_observed_2016_cli import RUN_MANIFEST


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--support-freeze", type=Path, default=FREEZE)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--support-freeze", type=Path, default=FREEZE)
    plan.add_argument("--card", type=Path, default=CARD)
    plan.add_argument("--card-manifest", type=Path, default=CARD_MANIFEST)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--support-freeze", type=Path, default=FREEZE)
    finalize.add_argument("--card", type=Path, default=CARD)
    finalize.add_argument("--card-manifest", type=Path, default=CARD_MANIFEST)
    finalize.add_argument("--repair-plan", type=Path, default=REPAIR_PLAN)
    return parser.parse_args(argv)


def validate_run_manifest(
    output: Path,
    *,
    freeze_sha: str,
    card_sha: str,
    role: str,
    rows: int,
    plan_sha: Optional[str] = None,
) -> Dict[str, Any]:
    result = output / "results_single.csv"
    manifest_path = output / RUN_MANIFEST
    manifest = load_json(manifest_path)
    expected = {
        "status": "pass",
        "study_id": STUDY_ID,
        "run_role": role,
        "support_freeze_sha256": freeze_sha,
        "card_sha256": card_sha,
        "rows": rows,
        "results_single_sha256": sha256(result),
        "runner_sha256": sha256(HERE / "run_observed_2016_cli.py"),
    }
    if plan_sha is not None:
        expected["repair_plan_sha256"] = plan_sha
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ObservedContractError(
                f"observed run manifest drift at {output}: {key}"
            )
    declared_numbers = manifest.get("numbers_json_sha256", {})
    declared_missing = manifest.get("numbers_json_missing", [])
    if not isinstance(declared_numbers, dict) or not isinstance(
        declared_missing, list
    ):
        raise ObservedContractError(f"invalid numbers inventory at {output}")
    if int(manifest.get("numbers_json_count", -1)) != len(declared_numbers):
        raise ObservedContractError(f"numbers count mismatch at {output}")
    if len(declared_numbers) + len(declared_missing) != rows:
        raise ObservedContractError(f"numbers inventory does not cover rows at {output}")
    if set(declared_numbers) & set(declared_missing):
        raise ObservedContractError(f"overlapping numbers inventory at {output}")
    for relative, expected_sha in declared_numbers.items():
        path = output / str(relative)
        if not path.is_file() or sha256(path) != expected_sha:
            raise ObservedContractError(f"numbers JSON hash mismatch: {path}")
    for relative in declared_missing:
        if (output / str(relative)).exists():
            raise ObservedContractError(
                f"declared-missing numbers JSON now exists: {output / str(relative)}"
            )
    return manifest


def load_primary(freeze_sha: str, card_sha: str) -> pd.DataFrame:
    path = PRIMARY / "results_single.csv"
    if not path.is_file():
        raise ObservedContractError(f"primary observed result is missing: {path}")
    validate_run_manifest(
        PRIMARY,
        freeze_sha=freeze_sha,
        card_sha=card_sha,
        role="primary_142_mass_scan",
        rows=142,
    )
    frame = pd.read_csv(path)
    validate_mass_grid(frame)
    if frame.duplicated(["dataset", "mass_GeV"]).any():
        raise ObservedContractError("primary result has duplicate coordinates")
    return frame


def build_plan(primary: pd.DataFrame, freeze_sha: str, card_sha: str) -> Dict[str, Any]:
    issues: Dict[str, List[str]] = {}
    for _, row in primary.iterrows():
        reasons = list(candidate_issue_reasons(row))
        mass_mev = int(round(1000.0 * float(row["mass_GeV"])))
        numbers_path = PRIMARY / f"m{mass_mev:03d}MeV" / "2016" / "numbers.json"
        if not numbers_path.is_file():
            reasons.append("numbers_json_missing")
        if reasons:
            issues[str(mass_mev)] = reasons
    masses = sorted(map(int, issues))
    return {
        "schema_version": 1,
        "status": "observed_repair_plan_frozen",
        "study_id": STUDY_ID,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "support_freeze_sha256": freeze_sha,
        "card_sha256": card_sha,
        "primary_results_sha256": sha256(PRIMARY / "results_single.csv"),
        "repeat_masses_MeV": masses,
        "repeat_count_per_mass": 3,
        "issue_reasons_by_mass_MeV": issues,
        "unchanged_card_required": True,
        "interpolation_permitted": False,
        "branch_match": {
            "max_delta_lml_per_n_train": 0.001,
            "max_abs_log_ls_ratio": 0.01,
            "max_abs_log_const_ratio": 0.05,
            "max_abs_log_sigma_A_ratio": 0.02,
            "minimum_replicates": 2,
        },
        "selection_rule": (
            "For a flagged mass, select the finite, covariance-valid, "
            "non-bound, non-exact-start branch with maximum LML among "
            "branches reproduced by at least two unchanged-card attempts. "
            "Amplitude, limit, epsilon-squared, p-value, and agreement with "
            "the primary result are not selection inputs."
        ),
    }


def load_repeat(
    mass_mev: int,
    repeat_index: int,
    freeze_sha: str,
    card_sha: str,
    plan_sha: str,
) -> Tuple[pd.Series, Path]:
    output = REPEAT_ROOT / f"m{mass_mev:03d}" / f"repeat{repeat_index}"
    path = output / "results_single.csv"
    validate_run_manifest(
        output,
        freeze_sha=freeze_sha,
        card_sha=card_sha,
        role=f"unchanged_card_repeat_{repeat_index}",
        rows=1,
        plan_sha=plan_sha,
    )
    frame = pd.read_csv(path)
    validate_mass_grid(frame, expected_rows=1)
    if not np.isclose(
        float(frame.iloc[0]["mass_GeV"]), mass_mev / 1000.0, rtol=0.0, atol=1e-12
    ):
        raise ObservedContractError(f"repeat {path} returned another mass")
    return frame.iloc[0].copy(), path


def candidate_record(
    row: pd.Series,
    *,
    mass_mev: int,
    candidate: str,
    source: Path,
    branch_multiplicity: int,
    selected: bool,
) -> Dict[str, Any]:
    reasons = list(candidate_issue_reasons(row))
    numbers = source.parent / f"m{mass_mev:03d}MeV" / "2016" / "numbers.json"
    return {
        "mass_GeV": float(row["mass_GeV"]),
        "mass_MeV": mass_mev,
        "candidate": candidate,
        "source_csv": source.resolve().relative_to(HERE).as_posix(),
        "source_csv_sha256": sha256(source),
        "numbers_json": numbers.resolve().relative_to(HERE).as_posix(),
        "numbers_json_sha256": sha256(numbers) if numbers.is_file() else "",
        "eligible": bool(eligible_candidate(row) and numbers.is_file()),
        "issue_reasons": "|".join(reasons),
        "branch_multiplicity": int(branch_multiplicity),
        "selected": bool(selected),
        "lml": float(row.get("lml", float("nan"))),
        "ls_opt": float(row.get("ls_opt", float("nan"))),
        "const_opt": float(row.get("const_opt", float("nan"))),
        "sigma_A": float(row.get("sigma_A", float("nan"))),
        "n_train": int(row.get("n_train", 0))
        if np.isfinite(float(row.get("n_train", float("nan"))))
        else 0,
        "covariance_valid": bool_value(
            row.get("covariance_valid", False), "covariance_valid"
        ),
        "covariance_min_eigenvalue_relative": float(
            row.get("covariance_min_eigenvalue_relative", float("nan"))
        ),
        "const_at_lower": bool_value(
            row.get("const_at_lower", False), "const_at_lower"
        ),
        "const_at_upper": bool_value(
            row.get("const_at_upper", False), "const_at_upper"
        ),
        "ls_at_lower": bool_value(row.get("ls_at_lower", False), "ls_at_lower"),
        "ls_at_upper": bool_value(row.get("ls_at_upper", False), "ls_at_upper"),
        "optimizer_warning_count": int(row.get("optimizer_warning_count", 0)),
        "optimizer_warnings": str(row.get("optimizer_warnings", "")),
    }


def finalize_review(
    primary: pd.DataFrame,
    plan: Dict[str, Any],
    freeze: Dict[str, Any],
    freeze_sha: str,
    card_sha: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    planned_masses = tuple(int(item) for item in plan["repeat_masses_MeV"])
    if tuple(sorted(set(planned_masses))) != planned_masses:
        raise ObservedContractError("repair plan mass inventory is not sorted/unique")
    plan_sha = sha256(REPAIR_PLAN)
    selected_rows: List[pd.Series] = []
    ledger: List[Dict[str, Any]] = []

    for _, original in primary.iterrows():
        mass_mev = int(round(1000.0 * float(original["mass_GeV"])))
        original_path = PRIMARY / "results_single.csv"
        if mass_mev not in planned_masses:
            if candidate_issue_reasons(original):
                raise ObservedContractError(
                    f"unplanned optimizer issue at {mass_mev} MeV"
                )
            selected = original.copy()
            selected["interpolated"] = False
            selected["branch_multiplicity"] = 1
            selected["selected_source"] = (
                original_path.resolve().relative_to(HERE).as_posix()
            )
            selected["selected_source_sha256"] = sha256(original_path)
            selected_numbers = (
                PRIMARY / f"m{mass_mev:03d}MeV" / "2016" / "numbers.json"
            )
            if not selected_numbers.is_file():
                raise ObservedContractError(
                    f"clean primary row lacks numbers JSON at {mass_mev} MeV"
                )
            selected["selected_numbers_json"] = (
                selected_numbers.resolve().relative_to(HERE).as_posix()
            )
            selected["selected_numbers_json_sha256"] = sha256(selected_numbers)
            selected["row_source"] = "primary"
            selected["review_status"] = "primary_clean"
            selected["repair_reproduction_pending"] = False
            selected_rows.append(selected)
            ledger.append(
                candidate_record(
                    original,
                    mass_mev=mass_mev,
                    candidate="primary",
                    source=original_path,
                    branch_multiplicity=1,
                    selected=True,
                )
            )
            continue

        candidates: List[Tuple[str, pd.Series, Path]] = [
            ("primary", original.copy(), original_path)
        ]
        for repeat_index in (1, 2, 3):
            row, path = load_repeat(
                mass_mev, repeat_index, freeze_sha, card_sha, plan_sha
            )
            candidates.append((f"repeat{repeat_index}", row, path))

        candidate_eligible = [
            bool(
                eligible_candidate(candidate)
                and (
                    source.parent
                    / f"m{mass_mev:03d}MeV"
                    / "2016"
                    / "numbers.json"
                ).is_file()
            )
            for _, candidate, source in candidates
        ]
        multiplicities = [
            (
                sum(
                    other_eligible and branch_match(candidate, other)
                    for (_, other, _), other_eligible in zip(
                        candidates, candidate_eligible
                    )
                )
                if is_eligible
                else 0
            )
            for (_, candidate, _), is_eligible in zip(
                candidates, candidate_eligible
            )
        ]
        selectable = [
            index for index, multiplicity in enumerate(multiplicities) if multiplicity >= 2
        ]
        if not selectable:
            raise ObservedContractError(
                f"no reproducible eligible optimizer branch at {mass_mev} MeV"
            )
        selected_index = max(
            selectable,
            key=lambda index: (float(candidates[index][1]["lml"]), -index),
        )
        selected_name, selected_row, selected_path = candidates[selected_index]
        output = selected_row.copy()
        output["interpolated"] = False
        output["branch_multiplicity"] = int(multiplicities[selected_index])
        output["selected_source"] = (
            selected_path.resolve().relative_to(HERE).as_posix()
        )
        output["selected_source_sha256"] = sha256(selected_path)
        selected_numbers = (
            selected_path.parent
            / f"m{mass_mev:03d}MeV"
            / "2016"
            / "numbers.json"
        )
        output["selected_numbers_json"] = (
            selected_numbers.resolve().relative_to(HERE).as_posix()
        )
        output["selected_numbers_json_sha256"] = sha256(selected_numbers)
        output["row_source"] = selected_name
        output["review_status"] = "repaired_reproducible_max_lml"
        output["repair_reproduction_pending"] = False
        selected_rows.append(output)
        for index, (name, row, path) in enumerate(candidates):
            ledger.append(
                candidate_record(
                    row,
                    mass_mev=mass_mev,
                    candidate=name,
                    source=path,
                    branch_multiplicity=multiplicities[index],
                    selected=index == selected_index,
                )
            )

    reviewed = pd.DataFrame(selected_rows).sort_values("mass_GeV").reset_index(drop=True)
    reviewed["dataset"] = "2016"
    reviewed["selected_support_low_MeV"] = int(
        freeze["selected_support_low_MeV"]
    )
    reviewed["support_high_MeV"] = int(freeze["support_high_MeV"])
    validate_mass_grid(reviewed)
    if len(reviewed) != len(primary):
        raise ObservedContractError("review changed the row count")
    ledger_frame = pd.DataFrame(ledger).sort_values(
        ["mass_GeV", "candidate"]
    ).reset_index(drop=True)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "study_id": STUDY_ID,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "support_freeze_sha256": freeze_sha,
        "card_sha256": card_sha,
        "repair_plan_sha256": plan_sha,
        "rows": int(len(reviewed)),
        "mass_low_MeV": 39,
        "mass_high_MeV": 180,
        "repaired_masses_MeV": list(planned_masses),
        "repaired_mass_count": len(planned_masses),
        "interpolated_rows": 0,
        "selection_rule": plan["selection_rule"],
        "branch_selection_uses_observed_amplitude_limit_pvalue": False,
        "reviewed_csv": str(REVIEWED_CSV.resolve()),
        "repair_ledger": str(REPAIR_LEDGER.resolve()),
    }
    return reviewed, ledger_frame, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    static = static_preflight()
    if args.mode == "preflight":
        status = "production_blocked_no_provisional_edge"
        print(
            json.dumps(
                {
                    "status": status,
                    "observed_data_evaluated": False,
                    "static": static,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    freeze = validate_freeze(args.support_freeze)
    validate_card(args.card, args.card_manifest, freeze)
    freeze_sha = sha256(FREEZE)
    card_sha = sha256(CARD)
    primary = load_primary(freeze_sha, card_sha)
    if args.mode == "plan":
        if REPAIR_PLAN.exists():
            raise ObservedContractError(f"refusing to overwrite {REPAIR_PLAN}")
        plan = build_plan(primary, freeze_sha, card_sha)
        atomic_json(REPAIR_PLAN, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.repair_plan.expanduser().resolve() != REPAIR_PLAN.resolve():
        raise ObservedContractError(f"repair plan must be {REPAIR_PLAN}")
    plan = load_json(REPAIR_PLAN)
    expected_plan = build_plan(primary, freeze_sha, card_sha)
    if plan != expected_plan:
        raise ObservedContractError(
            "frozen repair plan differs from a fresh primary-only classification"
        )
    for output in (REVIEWED_CSV, REPAIR_LEDGER, REVIEW_SUMMARY):
        if output.exists():
            raise ObservedContractError(f"refusing to overwrite {output}")
    reviewed, ledger, summary = finalize_review(
        primary, plan, freeze, freeze_sha, card_sha
    )
    atomic_csv(REVIEWED_CSV, reviewed)
    atomic_csv(REPAIR_LEDGER, ledger)
    summary["reviewed_csv_sha256"] = sha256(REVIEWED_CSV)
    summary["repair_ledger_sha256"] = sha256(REPAIR_LEDGER)
    summary["reviewer"] = str(Path(__file__).resolve())
    summary["reviewer_sha256"] = sha256(Path(__file__).resolve())
    atomic_json(REVIEW_SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
