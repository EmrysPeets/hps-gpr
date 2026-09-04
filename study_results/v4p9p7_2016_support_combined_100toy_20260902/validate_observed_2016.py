#!/usr/bin/env python3
"""Fail-closed validation of the reviewed full-2016 observed state ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from observed_2016_contract import (
    BOUND_COLUMNS,
    CARD,
    CARD_MANIFEST,
    CONFIRMATION_AUDIT,
    CORE_NUMERIC_COLUMNS,
    FREEZE,
    HERE,
    PRIMARY,
    REPAIR_LEDGER,
    REPAIR_PLAN,
    REPEAT_ROOT,
    REVIEWED_CSV,
    REVIEW_ROOT,
    REVIEW_SUMMARY,
    STUDY_ID,
    STUDY_SPEC,
    ObservedContractError,
    atomic_json,
    branch_match,
    bool_series,
    eligible_candidate,
    sha256,
    static_preflight,
    validate_card,
    validate_freeze,
    validate_mass_grid,
)


DEFAULT_QA = Path(__file__).resolve().parent / "qa" / "observed_2016_review_validation.json"
ASSEMBLER_COLUMNS = (
    "dataset",
    "mass_GeV",
    "const_opt",
    "ls_opt",
    "lml",
    "interpolated",
    "branch_multiplicity",
    "selected_source",
    "row_source",
    "review_status",
    "selected_support_low_MeV",
    "support_high_MeV",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--support-freeze", type=Path, default=FREEZE)
    subparsers.add_parser("blocked-state")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--support-freeze", type=Path, default=FREEZE)
    validate.add_argument("--card", type=Path, default=CARD)
    validate.add_argument("--card-manifest", type=Path, default=CARD_MANIFEST)
    validate.add_argument("--reviewed-csv", type=Path, default=REVIEWED_CSV)
    validate.add_argument("--repair-plan", type=Path, default=REPAIR_PLAN)
    validate.add_argument("--repair-ledger", type=Path, default=REPAIR_LEDGER)
    validate.add_argument("--review-summary", type=Path, default=REVIEW_SUMMARY)
    validate.add_argument("--qa-out", type=Path, default=DEFAULT_QA)
    return parser.parse_args(argv)


def require_exact_path(found: Path, expected: Path, label: str) -> Path:
    resolved = found.expanduser().resolve()
    if resolved != expected.resolve():
        raise ObservedContractError(f"{label} must be {expected.resolve()}")
    return resolved


def close(left: Any, right: Any) -> bool:
    return bool(
        np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12)
    )


def validate_products(args: argparse.Namespace) -> Dict[str, Any]:
    freeze = validate_freeze(args.support_freeze)
    validate_card(args.card, args.card_manifest, freeze)
    reviewed_path = require_exact_path(args.reviewed_csv, REVIEWED_CSV, "reviewed CSV")
    plan_path = require_exact_path(args.repair_plan, REPAIR_PLAN, "repair plan")
    ledger_path = require_exact_path(args.repair_ledger, REPAIR_LEDGER, "repair ledger")
    summary_path = require_exact_path(args.review_summary, REVIEW_SUMMARY, "review summary")
    for path in (reviewed_path, plan_path, ledger_path, summary_path):
        if not path.is_file():
            raise ObservedContractError(f"missing review product: {path}")

    reviewed = pd.read_csv(reviewed_path)
    validate_mass_grid(reviewed)
    missing = sorted(set(ASSEMBLER_COLUMNS) - set(reviewed.columns))
    if missing:
        raise ObservedContractError(f"reviewed CSV lacks assembler columns: {missing}")
    missing_numeric = sorted(set(CORE_NUMERIC_COLUMNS) - set(reviewed.columns))
    if missing_numeric:
        raise ObservedContractError(f"reviewed CSV lacks numeric columns: {missing_numeric}")
    numeric = reviewed.loc[:, CORE_NUMERIC_COLUMNS].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ObservedContractError("reviewed CSV contains non-finite core values")
    if not bool((reviewed["sigma_A"].astype(float) > 0.0).all()):
        raise ObservedContractError("reviewed CSV contains non-positive sigma_A")
    if not bool(
        (reviewed["A_up"].astype(float) > 0.0).all()
        and (reviewed["eps2_up"].astype(float) > 0.0).all()
    ):
        raise ObservedContractError("reviewed CSV contains non-positive upper limits")
    if not bool(reviewed["p0_analytic"].astype(float).between(0.0, 1.0).all()):
        raise ObservedContractError("reviewed CSV contains invalid local p0")
    if not bool(bool_series(reviewed["extract_success"]).all()):
        raise ObservedContractError("reviewed CSV contains a failed extraction")
    if not bool(bool_series(reviewed["density_window_fully_covered"]).all()):
        raise ObservedContractError("reviewed CSV lacks full density coverage")
    if any(bool(bool_series(reviewed[column]).any()) for column in BOUND_COLUMNS):
        raise ObservedContractError("reviewed CSV contains a kernel-bound contact")
    if not bool(bool_series(reviewed["covariance_valid"]).all()):
        raise ObservedContractError("reviewed CSV contains invalid covariance")
    eigen_relative = reviewed["covariance_min_eigenvalue_relative"].to_numpy(float)
    if not np.isfinite(eigen_relative).all() or bool((eigen_relative < -0.01).any()):
        raise ObservedContractError("reviewed CSV fails covariance eigenvalue gate")
    exact_start = (
        np.abs(np.log(reviewed["ls_opt"] / reviewed["ls_init"])) < 1.0e-8
    ) & (
        np.abs(np.log(reviewed["const_opt"] / reviewed["const_init"])) < 1.0e-8
    )
    if bool(exact_start.any()):
        raise ObservedContractError("reviewed CSV retains exact-start optimizer rows")
    if bool(bool_series(reviewed["interpolated"]).any()):
        raise ObservedContractError("reviewed CSV contains interpolation")
    if bool(bool_series(reviewed["repair_reproduction_pending"]).any()):
        raise ObservedContractError("reviewed CSV has pending repair reproduction")

    low = int(freeze["selected_support_low_MeV"])
    high = int(freeze["support_high_MeV"])
    supports = set(
        zip(
            reviewed["selected_support_low_MeV"].astype(int),
            reviewed["support_high_MeV"].astype(int),
        )
    )
    if supports != {(low, high)}:
        raise ObservedContractError("reviewed CSV support does not match freeze")
    branch_multiplicity = reviewed["branch_multiplicity"].to_numpy(float)
    if (
        not np.isfinite(branch_multiplicity).all()
        or not np.array_equal(branch_multiplicity, np.rint(branch_multiplicity))
        or bool((branch_multiplicity < 1).any())
    ):
        raise ObservedContractError("reviewed CSV has non-positive branch multiplicity")
    for column in ("selected_source", "row_source", "review_status"):
        if bool(reviewed[column].fillna("").astype(str).str.strip().eq("").any()):
            raise ObservedContractError(f"reviewed CSV has blank {column}")
    for column in (
        "selected_source_sha256",
        "selected_numbers_json",
        "selected_numbers_json_sha256",
    ):
        if column not in reviewed or bool(
            reviewed[column].fillna("").astype(str).str.strip().eq("").any()
        ):
            raise ObservedContractError(f"reviewed CSV has missing/blank {column}")
    allowed_status = {"primary_clean", "repaired_reproducible_max_lml"}
    if set(reviewed["review_status"].astype(str)) - allowed_status:
        raise ObservedContractError("reviewed CSV has an unknown review status")
    repaired_rows = reviewed[
        reviewed["review_status"] == "repaired_reproducible_max_lml"
    ]
    if bool((repaired_rows["branch_multiplicity"].astype(int) < 2).any()):
        raise ObservedContractError("a repaired row lacks two branch replicates")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if plan.get("status") != "observed_repair_plan_frozen":
        raise ObservedContractError("repair plan is not frozen")
    expected_bindings = {
        "study_id": STUDY_ID,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "support_freeze_sha256": sha256(FREEZE),
        "card_sha256": sha256(CARD),
    }
    for key, expected in expected_bindings.items():
        if plan.get(key) != expected or summary.get(key) != expected:
            raise ObservedContractError(f"review product binding drift: {key}")
    if summary.get("status") != "pass" or int(summary.get("rows", -1)) != 142:
        raise ObservedContractError("review summary does not declare 142 passing rows")
    if summary.get("reviewed_csv_sha256") != sha256(reviewed_path):
        raise ObservedContractError("review summary reviewed-CSV hash mismatch")
    if summary.get("repair_ledger_sha256") != sha256(ledger_path):
        raise ObservedContractError("review summary repair-ledger hash mismatch")
    if summary.get("repair_plan_sha256") != sha256(plan_path):
        raise ObservedContractError("review summary repair-plan hash mismatch")
    if summary.get("reviewer_sha256") != sha256(
        Path(__file__).resolve().parent / "review_observed_2016.py"
    ):
        raise ObservedContractError("review summary reviewer hash mismatch")

    planned_masses = tuple(int(item) for item in plan["repeat_masses_MeV"])
    found_repaired = tuple(
        sorted(
            int(round(1000.0 * value))
            for value in repaired_rows["mass_GeV"].to_numpy(float)
        )
    )
    if planned_masses != found_repaired:
        raise ObservedContractError("reviewed repaired masses differ from repair plan")
    ledger = pd.read_csv(ledger_path)
    required_ledger = {
        "mass_GeV",
        "mass_MeV",
        "candidate",
        "source_csv",
        "source_csv_sha256",
        "numbers_json",
        "numbers_json_sha256",
        "eligible",
        "branch_multiplicity",
        "selected",
        "lml",
        "ls_opt",
        "const_opt",
        "sigma_A",
        "n_train",
        "covariance_valid",
    }
    missing_ledger = sorted(required_ledger - set(ledger.columns))
    if missing_ledger:
        raise ObservedContractError(f"repair ledger lacks columns: {missing_ledger}")
    selected_ledger = ledger.loc[bool_series(ledger["selected"])].copy()
    if len(selected_ledger) != 142:
        raise ObservedContractError("repair ledger does not select exactly 142 rows")
    if tuple(selected_ledger["mass_MeV"].astype(int)) != tuple(range(39, 181)):
        raise ObservedContractError("repair-ledger selected grid is not 39--180 MeV")

    source_cache: Dict[Path, pd.DataFrame] = {}
    for _, reviewed_row in reviewed.iterrows():
        mass_mev = int(round(1000.0 * float(reviewed_row["mass_GeV"])))
        candidates = ledger.loc[ledger["mass_MeV"].astype(int) == mass_mev].copy()
        expected_labels = (
            ("primary", "repeat1", "repeat2", "repeat3")
            if mass_mev in planned_masses
            else ("primary",)
        )
        if (
            candidates["candidate"].duplicated().any()
            or set(candidates["candidate"].astype(str)) != set(expected_labels)
        ):
            raise ObservedContractError(
                f"repair-ledger candidate labels drift at {mass_mev} MeV"
            )
        order = {label: index for index, label in enumerate(expected_labels)}
        candidates = candidates.sort_values(
            "candidate", key=lambda values: values.map(order)
        ).reset_index(drop=True)
        selected = candidates.loc[bool_series(candidates["selected"])]
        expected_count = 4 if mass_mev in planned_masses else 1
        if len(candidates) != expected_count or len(selected) != 1:
            raise ObservedContractError(
                f"repair-ledger candidate inventory drift at {mass_mev} MeV"
            )

        raw_rows = []
        independently_eligible = []
        for _, candidate_row in candidates.iterrows():
            label = str(candidate_row["candidate"])
            if label == "primary":
                expected_source = PRIMARY / "results_single.csv"
            else:
                repeat_index = int(label.removeprefix("repeat"))
                expected_source = (
                    REPEAT_ROOT
                    / f"m{mass_mev:03d}"
                    / f"repeat{repeat_index}"
                    / "results_single.csv"
                )
            expected_relative = expected_source.resolve().relative_to(HERE).as_posix()
            if str(candidate_row["source_csv"]) != expected_relative:
                raise ObservedContractError(
                    f"candidate source path drift at {mass_mev} MeV: {label}"
                )
            if (
                not expected_source.is_file()
                or sha256(expected_source) != str(candidate_row["source_csv_sha256"])
            ):
                raise ObservedContractError(
                    f"candidate source hash drift at {mass_mev} MeV: {label}"
                )
            if expected_source not in source_cache:
                source_cache[expected_source] = pd.read_csv(expected_source)
            source_frame = source_cache[expected_source]
            source_dataset = source_frame["dataset"].astype(str).str.replace(
                r"\.0$", "", regex=True
            )
            coordinate = source_frame.loc[
                (source_dataset == "2016")
                & np.isclose(
                    source_frame["mass_GeV"].astype(float),
                    mass_mev / 1000.0,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(coordinate) != 1:
                raise ObservedContractError(
                    f"candidate source coordinate drift at {mass_mev} MeV: {label}"
                )
            raw_row = coordinate.iloc[0]
            raw_rows.append(raw_row)
            expected_numbers = (
                expected_source.parent
                / f"m{mass_mev:03d}MeV"
                / "2016"
                / "numbers.json"
            )
            expected_numbers_relative = (
                expected_numbers.resolve().relative_to(HERE).as_posix()
            )
            if str(candidate_row["numbers_json"]) != expected_numbers_relative:
                raise ObservedContractError(
                    f"candidate numbers path drift at {mass_mev} MeV: {label}"
                )
            declared_numbers_sha = (
                ""
                if pd.isna(candidate_row["numbers_json_sha256"])
                else str(candidate_row["numbers_json_sha256"])
            )
            actual_numbers_sha = (
                sha256(expected_numbers) if expected_numbers.is_file() else ""
            )
            if declared_numbers_sha != actual_numbers_sha:
                raise ObservedContractError(
                    f"candidate numbers hash drift at {mass_mev} MeV: {label}"
                )
            is_eligible = bool(
                eligible_candidate(raw_row) and expected_numbers.is_file()
            )
            independently_eligible.append(is_eligible)
            declared_eligible = bool(
                bool_series(pd.Series([candidate_row["eligible"]])).iloc[0]
            )
            if declared_eligible != is_eligible:
                raise ObservedContractError(
                    f"candidate eligibility drift at {mass_mev} MeV: {label}"
                )

        recomputed_multiplicities = [
            (
                sum(
                    other_eligible and branch_match(raw_row, other_row)
                    for other_row, other_eligible in zip(
                        raw_rows, independently_eligible
                    )
                )
                if is_eligible
                else 0
            )
            for raw_row, is_eligible in zip(raw_rows, independently_eligible)
        ]
        if not np.array_equal(
            candidates["branch_multiplicity"].astype(int).to_numpy(),
            np.asarray(recomputed_multiplicities, dtype=int),
        ):
            raise ObservedContractError(
                f"candidate branch multiplicity drift at {mass_mev} MeV"
            )
        ledger_row = selected.iloc[0]
        for column in ("lml", "ls_opt", "const_opt"):
            if not close(reviewed_row[column], ledger_row[column]):
                raise ObservedContractError(
                    f"reviewed/ledger {column} mismatch at {mass_mev} MeV"
                )
        if str(reviewed_row["selected_source"]) != str(ledger_row["source_csv"]):
            raise ObservedContractError(
                f"reviewed/ledger source mismatch at {mass_mev} MeV"
            )
        selected_source = (HERE / str(reviewed_row["selected_source"])).resolve()
        selected_source_sha = str(reviewed_row["selected_source_sha256"])
        if (
            not selected_source.is_file()
            or sha256(selected_source) != selected_source_sha
            or str(ledger_row["source_csv_sha256"]) != selected_source_sha
        ):
            raise ObservedContractError(
                f"selected source CSV hash mismatch at {mass_mev} MeV"
            )
        numbers_path = (HERE / str(reviewed_row["selected_numbers_json"])).resolve()
        ledger_numbers_path = (HERE / str(ledger_row["numbers_json"])).resolve()
        if numbers_path != ledger_numbers_path:
            raise ObservedContractError(
                f"reviewed/ledger numbers source mismatch at {mass_mev} MeV"
            )
        numbers_sha = str(reviewed_row["selected_numbers_json_sha256"])
        if (
            not numbers_path.is_file()
            or sha256(numbers_path) != numbers_sha
            or str(ledger_row["numbers_json_sha256"]) != numbers_sha
        ):
            raise ObservedContractError(
                f"selected numbers JSON hash mismatch at {mass_mev} MeV"
            )
        numbers_payload = json.loads(numbers_path.read_text(encoding="utf-8"))
        numbers_values = {
            "mass_GeV": numbers_payload.get("mass_GeV"),
            "lml": numbers_payload.get("gp_diagnostics", {})
            .get("optimizer", {})
            .get("log_marginal_likelihood"),
            "ls_opt": numbers_payload.get("gp_diagnostics", {})
            .get("length_scale", {})
            .get("optimized"),
            "const_opt": numbers_payload.get("gp_diagnostics", {})
            .get("constant", {})
            .get("optimized"),
        }
        if str(numbers_payload.get("dataset")) != "2016" or any(
            value is None or not close(value, reviewed_row[column])
            for column, value in numbers_values.items()
        ):
            raise ObservedContractError(
                f"selected numbers JSON content mismatch at {mass_mev} MeV"
            )
        if int(reviewed_row["branch_multiplicity"]) != int(
            ledger_row["branch_multiplicity"]
        ):
            raise ObservedContractError(
                f"reviewed/ledger multiplicity mismatch at {mass_mev} MeV"
            )
        if mass_mev in planned_masses:
            selectable_indices = [
                index
                for index, (is_eligible, multiplicity) in enumerate(
                    zip(independently_eligible, recomputed_multiplicities)
                )
                if is_eligible and multiplicity >= 2
            ]
            if not selectable_indices:
                raise ObservedContractError(
                    f"no selectable repair branch at {mass_mev} MeV"
                )
            selected_index = max(
                selectable_indices,
                key=lambda index: (float(raw_rows[index]["lml"]), -index),
            )
            expected_selected_label = expected_labels[selected_index]
            if str(ledger_row["candidate"]) != expected_selected_label:
                raise ObservedContractError(
                    f"selected repair is not reproducible max LML at {mass_mev} MeV"
                )

    return {
        "schema_version": 1,
        "status": "pass",
        "study_id": STUDY_ID,
        "rows": 142,
        "mass_grid_MeV": [39, 180, 1],
        "repaired_masses_MeV": list(planned_masses),
        "interpolated_rows": 0,
        "selected_support_low_MeV": low,
        "support_high_MeV": high,
        "reviewed_csv": str(reviewed_path),
        "reviewed_csv_sha256": sha256(reviewed_path),
        "repair_plan_sha256": sha256(plan_path),
        "repair_ledger_sha256": sha256(ledger_path),
        "review_summary_sha256": sha256(summary_path),
        "assembler_columns": list(ASSEMBLER_COLUMNS),
        "selection_uses_amplitude_limit_pvalue": False,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    static = static_preflight()
    if args.mode == "blocked-state":
        prohibited = {
            "confirmation_authorization": CONFIRMATION_AUDIT,
            "canonical_support_freeze": FREEZE,
            "generated_observed_card": CARD,
            "generated_observed_card_manifest": CARD_MANIFEST,
            "primary_observed_scan": PRIMARY,
            "unchanged_card_repeats": REPEAT_ROOT,
            "reviewed_observed_products": REVIEW_ROOT,
            "observed_review_validation": DEFAULT_QA,
        }
        present = {
            name: path.resolve().relative_to(HERE).as_posix()
            for name, path in prohibited.items()
            if path.exists()
        }
        if present:
            raise ObservedContractError(
                "blocked observed state contains prohibited products: "
                + json.dumps(present, sort_keys=True)
            )
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "pass",
                    "authorization_state": "blocked_no_support_freeze",
                    "observed_scan_authorized": False,
                    "observed_data_evaluated": False,
                    "required_absent_products": {
                        name: path.resolve().relative_to(HERE).as_posix()
                        for name, path in prohibited.items()
                    },
                    "static": static,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
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
    report = validate_products(args)
    qa_out = args.qa_out.expanduser().resolve()
    if qa_out.exists():
        raise ObservedContractError(f"refusing to overwrite validation QA: {qa_out}")
    atomic_json(qa_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
