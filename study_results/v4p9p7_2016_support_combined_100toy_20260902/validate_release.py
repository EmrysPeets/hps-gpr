#!/usr/bin/env python3
"""Fail-closed validation of the terminal v4.9.7 blocked-state release.

A passing report certifies that the frozen Phase-1 support study is complete,
that it selected no support, and that every downstream production product is
absent.  It never authorizes an observed-2016 scan, combined limit, or band.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FROZEN_REPOSITORY_SNAPSHOT = HERE / "runtime_combined"
REPORT = HERE / "release_validation.json"
MANIFEST = HERE / "release_manifest.json"
NOTE_MIRROR = REPO_ROOT / "output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf"
STUDY_ID = "v4p9p7_2016_support_combined_100toy_20260902"

SUPPORTS = [f"{edge:03d}_210" for edge in range(28, 35)]
MASSES_MEV = [44, 49, 54, 59]
INJECTIONS = [0, 2, 5]
TOYS = list(range(25))
TASK_FILES = {
    "_SUCCESS.json",
    "accepted_rows.csv",
    "exclusions.csv",
    "optimizer_attempts.csv",
    "raw_primary_rows.csv",
}

IMMUTABLE_HASHES = {
    "study_spec.json": "4382bfa6298cafe43d45026708017ca3e43179700f2ab5c76a557411874c8b3f",
    "STUDY_PROTOCOL.md": "81e5954c6bb1073010f32af8ab2fccc94d922f94018abe6416238e9d92cbec02",
    "SCIENTIFIC_SCOPE_CLARIFICATION.md": "7e90ed186396f3e209f6591ccdd28df714b642137797c07e0ed048bd02656b2c",
    "audit/independent_freeze_audit.py": "c53bd7bc066d37bc593b910a109912c26719ecd5d61bd13974a6b2e826a51058",
    "audit/static_truth_audit.json": "f27ff7400a82a8b0667e172766026b9007e2155eb447ccae05bf6adf17094964",
    "audit/phase1_selection_audit.json": "1118f5b293719bffe17217c5d24a6bf32f74a7a453b4ffd038fae7a34fce9416",
    "audit/production_authorization_denied.json": "c71b569da432723715922532e763b79dec6c0f9f04a08f84c0e190345c9d2b60",
    "audit/phase1_selection_audit.failed_path_reporting_20260902.json": "0651b288df55e21793b0cc264b7f8a1f17abcdfc4d98cd4d441f12991c51375f",
    "derived/analysis/phase1_selection_decision.json": "be1ac60e7b0420fc762a030ad579c855f65b20e41e4c32b03d514a804c82e71d",
    "derived/analysis/failed_support_study_summary.json": "4b3c7f8d8ca5cc07fa202a122227bccb9a6b586d767aaa7185703e0624a5e700",
    "observed_2016_workflow_manifest.json": "41bb3749e668816f689a980c2d8040105d1b5ed814ff7c18c9e4143ea69c50d3",
    "combined_scaffold_manifest.json": "1b4a2883a807d9ec4535c1fc5ce9276cbbe908e42460a6efb1476e3e7d98e1dd",
    "qa/combined_scaffold_smoke.json": "a3b100a57df01655dd305411a9569df7a83a9c3c7b14e0966635098db4480d3e",
    "qa/truth_product_validation.json": "18829241566069f75ba7b1069d22b8e134ff06b03bc915c0cde02bfc818ea536",
}

PHASE1_PRODUCTS = {
    "phase1_accepted_rows.csv": (
        2098,
        "228e5bf6b6bc7b30d74afb79f875a39db958bc0c31ac1a72febf25ff438a55bd",
    ),
    "phase1_cell_summary.csv": (
        84,
        "4b370d258f51da017230c7b08a2e15bce81db516a30749047aaadbc856f1078e",
    ),
    "phase1_support_summary.csv": (
        7,
        "29d5c22a50e39d8f538dafd5dd2deb146013711e126e3dd3a01a1499676b2124",
    ),
    "phase1_adjacent_paired_differences.csv": (
        72,
        "6dac7d80e1d63c3c0e28ac80c1c4136ab7e420ef7d486f8fb4d8e2b5498bc47c",
    ),
}

EXPECTED_WORST_PULL = {
    "028_210": 2.6958624791266566,
    "029_210": 2.3608650810237077,
    "030_210": 2.4594623811983656,
    "031_210": 2.5872443205185216,
    "032_210": 2.3279369363501097,
    "033_210": 2.28132791864934,
    "034_210": 2.5325242042854543,
}

BACKGROUND_ROOT_SHA = "689c700dc358db439a5da3eaa4bba4ee37f9d2d157afd10680b80cee1be2e912"
BACKGROUND_MANIFEST_SHA = "2c79965165c7186bb1bab4bb392d58c76974544735512a92f3c33ff8c3496773"


class ReleaseError(RuntimeError):
    """A release invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def reject_json_constant(value: str) -> None:
    raise ReleaseError(f"Non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Missing regular JSON file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"Cannot parse strict JSON {path}: {error}") from error
    require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(relative: str, expected: str) -> None:
    path = HERE / relative
    require(path.is_file() and not path.is_symlink(), f"Missing regular file: {relative}")
    actual = sha256(path)
    require(actual == expected, f"SHA-256 drift for {relative}: {actual} != {expected}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file() and not path.is_symlink(), f"Missing regular CSV file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames is not None, f"CSV has no header: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReleaseError(f"Cannot parse CSV {path}: {error}") from error
    return rows


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    require(text in {"true", "false", "1", "0"}, f"Not a strict boolean: {value!r}")
    return text in {"true", "1"}


def close(actual: Any, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def check_filesystem_objects() -> dict[str, int]:
    files = 0
    directories = 0
    for root, names, filenames in os.walk(HERE, followlinks=False):
        root_path = Path(root)
        for name in names:
            path = root_path / name
            require(not path.is_symlink(), f"Symlink directory forbidden: {path.relative_to(HERE)}")
            require(stat.S_ISDIR(path.lstat().st_mode), f"Non-directory object: {path}")
            directories += 1
        for name in filenames:
            path = root_path / name
            require(not path.is_symlink(), f"Symlink file forbidden: {path.relative_to(HERE)}")
            require(stat.S_ISREG(path.lstat().st_mode), f"Non-regular file: {path}")
            files += 1
    return {"regular_files_including_manifest_and_report": files, "directories": directories}


def check_terminal_decision_and_audits() -> dict[str, Any]:
    for relative, expected in IMMUTABLE_HASHES.items():
        require_sha(relative, expected)

    decision = load_json(HERE / "derived/analysis/phase1_selection_decision.json")
    require(decision.get("study_id") == STUDY_ID, "Phase-1 decision study_id drift")
    require(decision.get("status") == "no_provisional_edge", "Decision is not no_provisional_edge")
    require(decision.get("phase2_supports") == [], "Phase 2 must be empty")
    require(decision.get("holdout_evaluated") is False, "65 MeV holdout was evaluated")
    require(decision.get("observed_scan_authorized") is False, "Observed scan was authorized")
    require(decision.get("selection_grid_masses_MeV") == MASSES_MEV, "Selection mass grid drift")
    require(decision.get("excluded_holdout_mass_MeV") == 65, "Holdout declaration drift")
    require(
        decision.get("reason") == "no eligible edge passed the frozen phase-1 practical rule",
        "Decision reason drift",
    )
    products = decision.get("products")
    require(isinstance(products, dict), "Decision products missing")
    for filename, (rows, expected_hash) in PHASE1_PRODUCTS.items():
        require(products.get(filename) == {"rows": rows, "sha256": expected_hash}, f"Decision binding drift: {filename}")

    phase1_audit = load_json(HERE / "audit/phase1_selection_audit.json")
    require(phase1_audit.get("status") == "pass", "Independent Phase-1 audit did not pass")
    require(phase1_audit.get("exact_candidate_supports") == SUPPORTS, "Audit support grid drift")
    require(phase1_audit.get("independent_selected_support") is None, "Audit selected a support")
    require(phase1_audit.get("independent_phase2_supports") == [], "Audit populated Phase 2")
    require(phase1_audit.get("independent_tied_supports") == [], "Audit populated ties")
    require(phase1_audit.get("observed_scan_authorized") is False, "Audit authorized observed scan")
    audit_summaries = phase1_audit.get("support_summaries")
    require(isinstance(audit_summaries, list) and len(audit_summaries) == 7, "Audit must summarize seven supports")
    for row in audit_summaries:
        support = row.get("support")
        require(support in SUPPORTS, f"Unknown audit support: {support}")
        require(row.get("cells_below_abs_mean_pull_0p75") == 3, f"3/12 practical-cell fact drift: {support}")
        require(row.get("zero_signal_cells_below_abs_mean_pull_0p75") == 1, f"1/4 zero-signal fact drift: {support}")
        require(row.get("gross_bias_guard_pass") is False, f"Gross-bias gate unexpectedly passed: {support}")
        require(row.get("practical_acceptability_pass") is False, f"Practical gate unexpectedly passed: {support}")
        require(close(row.get("worst_abs_mean_pull"), EXPECTED_WORST_PULL[support]), f"Worst pull drift: {support}")
        technical_expected = support not in {"030_210", "032_210"}
        require(row.get("technical_gate_pass") is technical_expected, f"Technical gate drift: {support}")

    denied = load_json(HERE / "audit/production_authorization_denied.json")
    require(denied.get("status") == "production_blocked", "Authorization artifact is not blocked")
    require(denied.get("audit_status") == "pass", "Denied authorization audit did not pass")
    require(denied.get("canonical_phase1_decision_status") == "no_provisional_edge", "Denied artifact decision drift")
    require(denied.get("exact_phase2_supports") == [], "Denied artifact Phase-2 set drift")
    require(denied.get("independent_selected_support") is None, "Denied artifact selected support")
    require(denied.get("canonical_support_freeze_decision_present") is False, "Freeze unexpectedly present")
    authorization = denied.get("authorization", {})
    require(authorization.get("status") == "denied", "Production authorization is not denied")
    for key in (
        "combined_production_authorized",
        "confirmation_authorized",
        "holdout_65MeV_authorized",
        "observed_scan_authorized",
    ):
        require(authorization.get(key) is False, f"Unexpected authorization: {key}")
    require(authorization.get("required_protocol_action") == "stop without retuning", "Stop action drift")

    summary = load_json(HERE / "derived/analysis/failed_support_study_summary.json")
    require(summary.get("status") == "halted_no_provisional_edge", "Failure summary status drift")
    for key in ("phase2_authorized", "observed_scan_authorized", "combined_result_authorized"):
        require(summary.get(key) is False, f"Failure summary authorizes {key}")
    require(summary.get("candidate_edges_MeV") == list(range(28, 35)), "Failure summary edges drift")
    require(summary.get("practical_gate_passed_edges_MeV") == [], "Failure summary has passing edges")
    require(summary.get("technical_gate_failed_edges_MeV") == [30, 32], "Technical failures drift")
    require(summary.get("technical_exclusion_count") == 2, "Exclusion count drift")
    require(summary.get("numerically_smallest_worst_pull_edge_MeV") == 33, "Diagnostic minimum drift")
    require(summary.get("numerically_smallest_edge_not_selected") is True, "Failing edge was selected")

    return {
        "decision": "no_provisional_edge",
        "candidate_supports": SUPPORTS,
        "phase2_supports": [],
        "production_authorization": "denied",
    }


def expected_task_paths() -> set[str]:
    return {
        f"runs/2016_threshold_qualified_{support}/2016_full/toy_{toy:04d}"
        for support in SUPPORTS
        for toy in TOYS
    }


def logical_row_key(row: dict[str, str], support: str, toy_field: str) -> tuple[str, int, int, int]:
    return (
        support,
        int(row[toy_field]),
        int(round(float(row["mass_GeV"]) * 1000.0)),
        int(round(float(row["inj_nsigma"]))),
    )


def check_phase1_task_inventory() -> dict[str, Any]:
    runs = HERE / "runs"
    require(runs.is_dir(), "runs directory missing")
    all_toy_dirs = {
        path.relative_to(HERE).as_posix()
        for path in runs.rglob("toy_*")
        if path.is_dir()
    }
    canonical_re = re.compile(
        r"^runs/2016_threshold_qualified_(028_210|029_210|030_210|031_210|032_210|033_210|034_210)/"
        r"2016_full/toy_(\d{4})$"
    )
    canonical = set()
    for relative in all_toy_dirs:
        match = canonical_re.fullmatch(relative)
        if match:
            toy = int(match.group(2))
            require(toy in TOYS, f"Non-Phase-1 toy directory is forbidden: {relative}")
            canonical.add(relative)
    expected = expected_task_paths()
    require(canonical == expected, f"Canonical task directory mismatch: missing={sorted(expected-canonical)[:5]}, extra={sorted(canonical-expected)[:5]}")
    allowed_lineage = {
        "runs/2016_threshold_qualified_028_210/2016_full/toy_0000.superseded_20260902T151645"
    }
    require(all_toy_dirs - canonical == allowed_lineage, f"Unexpected noncanonical toy directories: {sorted(all_toy_dirs-canonical)}")

    accepted_total = 0
    excluded_total = 0
    raw_total = 0
    attempt_total = 0
    accepted_keys: set[tuple[str, int, int, int]] = set()
    excluded_keys: set[tuple[str, int, int, int]] = set()
    expected_grid = {(mass, injection) for mass in MASSES_MEV for injection in INJECTIONS}

    for relative in sorted(canonical):
        task = HERE / relative
        files = {path.name for path in task.iterdir() if path.is_file()}
        dirs = [path.name for path in task.iterdir() if path.is_dir()]
        require(files == TASK_FILES and not dirs, f"Task file inventory drift: {relative}")
        marker = load_json(task / "_SUCCESS.json")
        support = relative.split("threshold_qualified_", 1)[1].split("/", 1)[0]
        toy = int(relative.rsplit("_", 1)[1])
        require(
            set(marker)
            == {
                "accepted_rows",
                "attempt_rows",
                "background_toy_manifest_sha256",
                "background_toy_root_sha256",
                "completed_utc",
                "excluded_rows",
                "ledger_sha256",
                "raw_primary_rows",
                "scenario",
                "status",
                "study_spec_sha256",
                "toy_index",
            },
            f"Task marker schema drift: {relative}",
        )
        require(marker.get("status") == "pass", f"Task marker failed: {relative}")
        require(marker.get("scenario") == "2016_full", f"Task scenario drift: {relative}")
        require(marker.get("toy_index") == toy, f"Task toy index drift: {relative}")
        require(marker.get("study_spec_sha256") == IMMUTABLE_HASHES["study_spec.json"], f"Task spec binding drift: {relative}")
        require(marker.get("background_toy_root_sha256") == BACKGROUND_ROOT_SHA, f"Task toy ROOT binding drift: {relative}")
        require(marker.get("background_toy_manifest_sha256") == BACKGROUND_MANIFEST_SHA, f"Task toy manifest binding drift: {relative}")

        ledger_hashes = marker.get("ledger_sha256")
        require(isinstance(ledger_hashes, dict) and set(ledger_hashes) == TASK_FILES - {"_SUCCESS.json"}, f"Task ledger inventory drift: {relative}")
        for filename, expected_hash in ledger_hashes.items():
            actual_hash = sha256(task / filename)
            require(actual_hash == expected_hash, f"Task ledger hash drift: {relative}/{filename}")

        raw = csv_rows(task / "raw_primary_rows.csv")
        accepted = csv_rows(task / "accepted_rows.csv")
        excluded = csv_rows(task / "exclusions.csv")
        attempts = csv_rows(task / "optimizer_attempts.csv")
        require(len(raw) == marker.get("raw_primary_rows") == 12, f"Raw row count drift: {relative}")
        require(len(accepted) == marker.get("accepted_rows"), f"Accepted row count drift: {relative}")
        require(len(excluded) == marker.get("excluded_rows"), f"Excluded row count drift: {relative}")
        require(len(attempts) == marker.get("attempt_rows"), f"Attempt row count drift: {relative}")
        raw_grid = {
            (int(round(float(row["mass_GeV"]) * 1000.0)), int(round(float(row["inj_nsigma"]))))
            for row in raw
        }
        require(raw_grid == expected_grid, f"Logical 12-cell task grid drift: {relative}")
        for row in [*raw, *accepted]:
            require(row.get("scenario") == "2016_full", f"Row scenario drift: {relative}")
            require(int(row["background_toy_index"]) == toy, f"Row toy index drift: {relative}")
        here_accepted = {logical_row_key(row, support, "background_toy_index") for row in accepted}
        here_excluded = {
            logical_row_key(row, support, "background_toy_index") for row in excluded
        }
        require(not (here_accepted & here_excluded), f"Accepted/excluded overlap: {relative}")
        require(
            {(key[2], key[3]) for key in here_accepted | here_excluded} == expected_grid,
            f"Accepted/excluded partition drift: {relative}",
        )
        require(len(here_accepted) == len(accepted), f"Duplicate accepted states: {relative}")
        require(len(here_excluded) == len(excluded), f"Duplicate exclusions: {relative}")
        require(not (accepted_keys & here_accepted), f"Duplicate accepted task keys: {relative}")
        accepted_keys.update(here_accepted)
        excluded_keys.update(here_excluded)
        raw_total += len(raw)
        accepted_total += len(accepted)
        excluded_total += len(excluded)
        attempt_total += len(attempts)

    require(len(canonical) == 175, "Expected exactly 175 canonical Phase-1 task directories")
    require(raw_total == 2100, f"Expected 2100 logical task states, found {raw_total}")
    require(accepted_total == 2098, f"Expected 2098 accepted task states, found {accepted_total}")
    require(excluded_total == 2, f"Expected two task exclusions, found {excluded_total}")
    require(attempt_total == 3806, f"Expected 3806 optimizer-attempt rows, found {attempt_total}")
    expected_excluded_keys = {("030_210", 18, 44, 5), ("032_210", 3, 54, 2)}
    require(excluded_keys == expected_excluded_keys, f"Unexpected task exclusions: {sorted(excluded_keys)}")
    return {
        "canonical_task_directories": len(canonical),
        "phase1_toys_per_support": 25,
        "logical_states": raw_total,
        "accepted_states": accepted_total,
        "technical_exclusions": excluded_total,
        "optimizer_attempt_rows": attempt_total,
        "superseded_lineage_task_directories": len(allowed_lineage),
    }


def check_phase1_aggregate_tables() -> dict[str, Any]:
    for filename, (expected_rows, expected_hash) in PHASE1_PRODUCTS.items():
        path = HERE / "derived/analysis" / filename
        require(sha256(path) == expected_hash, f"Aggregate hash drift: {filename}")
        require(len(csv_rows(path)) == expected_rows, f"Aggregate row-count drift: {filename}")

    accepted = csv_rows(HERE / "derived/analysis/phase1_accepted_rows.csv")
    actual_keys = {
        logical_row_key(row, row["support"], "background_toy_index") for row in accepted
    }
    require(len(actual_keys) == 2098, "Aggregate accepted table has duplicate logical states")
    full_grid = {
        (support, toy, mass, injection)
        for support in SUPPORTS
        for toy in TOYS
        for mass in MASSES_MEV
        for injection in INJECTIONS
    }
    expected_missing = {("030_210", 18, 44, 5), ("032_210", 3, 54, 2)}
    require(full_grid - actual_keys == expected_missing, "Aggregate accepted logical-state exclusions drift")
    require(actual_keys <= full_grid, "Aggregate accepted table contains off-grid states")
    require(all(row.get("cohort") == "initial_0_24" for row in accepted), "Aggregate cohort drift")
    require(
        all(row.get("upper_limit_diagnostic") == "90pct_CLs_Wald_tilde_q_mu_from_profiled_Ahat_sigmaA" for row in accepted),
        "Per-row diagnostic upper-limit label drift",
    )

    cells = csv_rows(HERE / "derived/analysis/phase1_cell_summary.csv")
    cell_counts = Counter(row["support"] for row in cells)
    require(cell_counts == Counter({support: 12 for support in SUPPORTS}), "Cell-summary support counts drift")
    expected_24 = {("030_210", 44, 5), ("032_210", 54, 2)}
    actual_24: set[tuple[str, int, int]] = set()
    for row in cells:
        key = (
            row["support"],
            int(round(float(row["mass_MeV"]))),
            int(round(float(row["inj_nsigma"]))),
        )
        count = int(row["n"])
        require(count in {24, 25}, f"Unexpected cell count: {key} -> {count}")
        if count == 24:
            actual_24.add(key)
    require(actual_24 == expected_24, f"24/25 cell locations drift: {sorted(actual_24)}")

    supports = csv_rows(HERE / "derived/analysis/phase1_support_summary.csv")
    require({row["support"] for row in supports} == set(SUPPORTS), "Support-summary labels drift")
    for row in supports:
        support = row["support"]
        require(int(row["cells_below_abs_mean_pull_0p75"]) == 3, f"Support 3/12 fact drift: {support}")
        require(int(row["zero_signal_cells_below_abs_mean_pull_0p75"]) == 1, f"Support 1/4 fact drift: {support}")
        require(as_bool(row["gross_bias_guard_pass"]) is False, f"Gross guard unexpectedly passed: {support}")
        require(as_bool(row["practical_acceptability_pass"]) is False, f"Practical gate unexpectedly passed: {support}")
        require(as_bool(row["absolute_upper_limit_used_for_ranking"]) is False, f"UL ranked support: {support}")
        require(close(row["worst_abs_mean_pull"], EXPECTED_WORST_PULL[support]), f"Worst pull drift: {support}")
        require(as_bool(row["technical_gate_pass"]) is (support not in {"030_210", "032_210"}), f"Technical gate drift: {support}")

    paired = csv_rows(HERE / "derived/analysis/phase1_adjacent_paired_differences.csv")
    require(all(not as_bool(row["used_for_support_ranking"]) for row in paired), "Paired differences ranked support")
    require(len({(row["lower_support"], row["higher_support"]) for row in paired}) == 6, "Adjacent pair count drift")

    exclusions = csv_rows(HERE / "derived/analysis/phase1_technical_exclusions.csv")
    require(
        sha256(HERE / "derived/analysis/phase1_technical_exclusions.csv")
        == "942ec977b37e355412b4ebbcdb5b30a06ec384dc60dcdb9710841cf400163766",
        "Technical-exclusion aggregate hash drift",
    )
    expected_exclusions = {
        ("030_210", 18, 44, 5, "irreproducible_injected_refit_top_branch"),
        ("032_210", 3, 54, 2, "irreproducible_injected_refit_top_branch"),
    }
    actual_exclusions = {
        (
            row["support"],
            int(row["toy_index"]),
            int(round(float(row["mass_MeV"]))),
            int(round(float(row["inj_nsigma"]))),
            row["reason"],
        )
        for row in exclusions
    }
    require(actual_exclusions == expected_exclusions, "Technical-exclusion ledger drift")
    for row in exclusions:
        source = HERE / row["source_ledger"]
        require(source.is_file() and sha256(source) == row["source_ledger_sha256"], "Exclusion source binding drift")

    return {name: rows for name, (rows, _) in PHASE1_PRODUCTS.items()} | {
        "logical_states_before_exclusions": 2100,
        "technical_exclusions": 2,
    }


def check_truth_and_static_caveats() -> dict[str, Any]:
    static = load_json(HERE / "audit/static_truth_audit.json")
    require(static.get("status") == "pass", "Static truth audit did not pass")
    checked = static.get("checked_file_sha256")
    require(isinstance(checked, dict), "Static audit checked-file map missing")
    for relative, expected in checked.items():
        if relative.startswith("repository/"):
            # The static audit binds the source state used by this study.  Resolve
            # those entries against the packaged runtime snapshot so validation is
            # portable and cannot be changed by an unrelated dirty destination
            # checkout.  The live repository mirror is checked separately below.
            path = FROZEN_REPOSITORY_SNAPSHOT / relative.removeprefix("repository/")
        else:
            path = HERE / relative
        require(path.is_file() and not path.is_symlink(), f"Static-audit file missing: {relative}")
        require(sha256(path) == expected, f"Static-audit file hash drift: {relative}")

    degree = static.get("degree_selection", {})
    require(degree.get("candidate_degrees") == list(range(4, 11)), "Degree candidates drift")
    require(degree.get("independently_passing_degrees") == list(range(5, 11)), "Degree pass set drift")
    require(degree.get("selected_degree") == 5, "Truth degree is not five")
    require(degree.get("selection_verified_as_lowest_passing") is True, "Lowest-passing degree gate failed")

    full_shape = static.get("full_observed_shape_use_audit", {})
    require(full_shape.get("ten_pct_statistical_independence_from_full_100pct_unproven") is True, "Unproven 10% independence caveat missing")
    require(full_shape.get("ten_pct_development_shape_entered_source_conditioned_truth") is True, "Partial observed-shape use not acknowledged")
    require(full_shape.get("full_100pct_values_entered_truth_only_as_scalar_26_210MeV_normalization") is True, "Full-data scalar-only statement missing")
    require(full_shape.get("support_specific_full_100pct_fit_p0_or_upper_limit_used_for_ranking") is False, "Support-specific full-data information entered ranking")
    require(full_shape.get("ten_pct_bins_never_exceed_full_100pct_bins") is True, "10% subset bin check failed")

    broad = static.get("broad_tail", {})
    require(broad.get("fit_ok") is False, "Broad-tail fit_ok=false fact drift")
    require(broad.get("waiver_required") is True and broad.get("waiver_acknowledged") is True, "Broad-tail waiver gate failed")
    immutable_shape = broad.get("immutable_shape_checks", {})
    require(isinstance(immutable_shape, dict) and immutable_shape, "Immutable shape checks missing")
    require(all(value is True for value in immutable_shape.values()), "An immutable broad-tail shape check failed")
    waiver = str(broad.get("waiver_scope", "")).lower()
    for phrase in ("conditional source-conditioned stress truth only", "not a physical background generator", "not established as statistically independent"):
        require(phrase in waiver, f"Broad-tail waiver scope missing phrase: {phrase}")

    require(static.get("source_counts_26_210MeV") == {"normalization_2016_full": 73145594, "shape_2016_10pct": 7475607}, "Truth source counts drift")
    seeds = static.get("optimizer_and_signal_seed_audit", {})
    require(seeds.get("base_seed") == 20260902, "Optimizer seed base drift")
    require(seeds.get("logical_seed_identities") == 6800, "Optimizer logical seed count drift")
    require(seeds.get("unique_uint32_seeds") == 6800 and seeds.get("collisions") == 0, "Optimizer seed collision audit failed")
    require(seeds.get("support_label_in_seed_identity") is False, "Support label entered paired seed identity")
    reproduction = static.get("toy_reproduction", {})
    require(reproduction.get("n_toys_reproduced_bitwise") == 100, "100 toy reproduction failed")
    require(reproduction.get("base_seed") == 20260902, "Background seed drift")
    require("independent of support" in str(reproduction.get("paired_background_identity", "")), "Toy pairing semantics drift")

    geometry = static.get("support_geometry")
    require(isinstance(geometry, list) and len(geometry) == 7, "Support geometry must have seven rows")
    for index, row in enumerate(geometry):
        edge = 28 + index
        require(row.get("support") == f"{edge:03d}_210", "Support geometry ordering drift")
        require(row.get("coarse_bins") == 728 - 4 * index, "Support coarse-bin geometry drift")
        require(row.get("low_side_training_bins_at_39MeV") == 28 - 4 * index, "Low-side training geometry drift")
        require(close(row.get("coarse_bin_width_GeV"), 0.00025, tolerance=1e-10), "Coarse width drift")

    truth_qa = load_json(HERE / "qa/truth_product_validation.json")
    require(truth_qa.get("status") == "pass", "Truth-product QA did not pass")
    require(truth_qa.get("n_toys") == 100, "Truth-product toy count drift")
    require(truth_qa.get("selected_degree") == 5, "Truth-product degree drift")
    require(truth_qa.get("full_target_count") == 73145594, "Truth-product normalization drift")
    require(truth_qa.get("root_sha256") == BACKGROUND_ROOT_SHA, "Truth-product ROOT hash drift")
    require(truth_qa.get("manifest_sha256") == BACKGROUND_MANIFEST_SHA, "Truth-product manifest hash drift")

    clarification = (HERE / "SCIENTIFIC_SCOPE_CLARIFICATION.md").read_text(encoding="utf-8").lower()
    for phrase in ("pre-existing 2016 10% development", "establishes statistical disjointness"):
        require(phrase in clarification, f"Scientific clarification missing phrase: {phrase}")
    require(
        "partial observed-" in clarification and "shape information" in clarification,
        "Scientific clarification omits partial observed-shape information",
    )
    require("fit_ok: false" in clarification or "fit_ok=false" in clarification, "Scientific clarification omits fit_ok=false fact")
    return {
        "truth_degree": 5,
        "truth_toys": 100,
        "full_normalization_26_210MeV": 73145594,
        "ten_percent_independence_established": False,
        "broad_tail_fit_ok": False,
        "broad_tail_waiver": "conditional_stress_truth_only",
    }


def check_signal_robustness_audit() -> dict[str, Any]:
    root = HERE / "signal_audit"
    manifest_path = root / "qa/artifact_manifest_sha256.csv"
    entries = csv_rows(manifest_path)
    require(len(entries) == 29, f"Signal audit manifest must have 29 entries, found {len(entries)}")
    require(len({row["path"] for row in entries}) == 29, "Signal audit manifest has duplicate paths")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    listed_files = {row["path"] for row in entries}
    require(listed_files == actual_files, f"Signal audit inventory mismatch: missing={sorted(actual_files-listed_files)}, extra={sorted(listed_files-actual_files)}")
    for row in entries:
        path = root / row["path"]
        require(not path.is_symlink(), f"Signal audit symlink forbidden: {row['path']}")
        require(path.stat().st_size == int(row["bytes"]), f"Signal audit byte count drift: {row['path']}")
        require(sha256(path) == row["sha256"], f"Signal audit hash drift: {row['path']}")

    semantic = load_json(root / "qa/semantic_validation.json")
    require(semantic.get("overall_pass") is True, "Signal semantic validation failed")
    require(semantic.get("n_checks") == 13 and semantic.get("n_pass") == 13, "Signal semantic check count is not 13/13")
    checks = semantic.get("checks")
    require(isinstance(checks, list) and len(checks) == 13, "Signal semantic check records drift")
    require(all(row.get("pass") is True for row in checks), "A signal semantic check failed")

    render = load_json(root / "qa/pdf_render_qa.json")
    require(render.get("all_rendered") is True, "Signal PDFs were not all rendered")
    records = render.get("records")
    require(isinstance(records, list) and len(records) == 3, "Signal PDF QA must have three records")
    expected_pdfs = {
        "figures/m65_scope_hybrid_diagnostic.pdf",
        "figures/m65_support_kernel_mechanism.pdf",
        "figures/old_new_2021_local_z_curves.pdf",
    }
    require({row.get("pdf") for row in records} == expected_pdfs, "Signal PDF inventory drift")
    for row in records:
        require(row.get("rendered") is True, f"Signal PDF did not render: {row.get('pdf')}")
        pdf = root / row["pdf"]
        rendered = root / row["render_path"]
        require(sha256(pdf) == row["pdf_sha256"], f"Signal PDF hash drift: {row['pdf']}")
        require(sha256(rendered) == row["render_sha256"], f"Signal render hash drift: {row['render_path']}")

    summary = load_json(root / "derived/signal_robustness_summary.json")
    old = summary.get("m65_exact", {}).get("old", {})
    new = summary.get("m65_exact", {}).get("new", {})
    require(close(old.get("Z_local_asymptotic"), 4.252493351343015), "Old m65 local Z drift")
    require(close(new.get("Z_local_asymptotic"), 2.4028773513795936), "New m65 local Z drift")
    require(old.get("K_events_per_eps2") == new.get("K_events_per_eps2"), "m65 epsilon2 conversion changed")
    require(old.get("integral_density_counts_per_GeV") == new.get("integral_density_counts_per_GeV"), "m65 normalization changed")
    mechanism = summary.get("mechanism", {})
    require("not robust to the GP-support prescription" in str(mechanism.get("defensible_statement", "")), "Defensible signal-robustness statement missing")
    not_established = str(mechanism.get("not_established", ""))
    require("raw data feature did not disappear" in not_established and "does not prove" in not_established, "Signal/background non-classification statement missing")
    boundary = summary.get("claim_boundary", {})
    require("no look-elsewhere calibration" in str(boundary.get("local_p0_Z", "")), "Local-Z boundary missing")
    require("cannot classify" in str(boundary.get("physics", "")), "Signal/background boundary missing")
    require("not direct coverage" in str(boundary.get("toys", "")), "Toy-coverage boundary missing")
    return {
        "manifest_entries": 29,
        "semantic_checks": "13/13",
        "rendered_pdf_figures": 3,
        "m65_local_asymptotic_Z_old": old["Z_local_asymptotic"],
        "m65_local_asymptotic_Z_new": new["Z_local_asymptotic"],
    }


def check_observed_blocked_state() -> dict[str, Any]:
    payload = load_json(HERE / "observed_2016_workflow_manifest.json")
    require(payload.get("status") == "production_blocked_no_provisional_edge", "Observed workflow is not blocked")
    require(payload.get("study_id") == STUDY_ID, "Observed manifest study_id drift")
    for key in ("observed_data_evaluated", "observed_scan_authorized", "combined_production_authorized", "canonical_support_freeze_present", "confirmation_authorization_present"):
        require(payload.get(key) is False, f"Observed blocked state drift: {key}")
    require(payload.get("required_protocol_action") == "stop without retuning", "Observed stop action drift")
    require(payload.get("expected_mass_grid_MeV") == {"low": 39, "high": 180, "step": 1, "rows": 142}, "Observed mass-grid contract drift")
    require(payload.get("source_2016_full_root_sha256") == "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301", "Full-2016 source binding drift")

    audit_hashes = payload.get("independent_audit_sha256", {})
    for relative, expected in audit_hashes.items():
        require_sha(relative, expected)
    workflow_hashes = payload.get("workflow_files_sha256", {})
    require(isinstance(workflow_hashes, dict) and len(workflow_hashes) == 7, "Observed workflow hash map drift")
    for relative, expected in workflow_hashes.items():
        require_sha(relative, expected)

    products = payload.get("prohibited_absent_observed_products", {})
    require(isinstance(products, dict) and len(products) == 10, "Observed prohibited-product map drift")
    for label, relative in products.items():
        require(not (HERE / relative).exists(), f"Prohibited observed product exists ({label}): {relative}")
    validation = payload.get("blocked_state_validation", {})
    require(validation.get("status") == "pass", "Observed blocked-state validation did not pass")
    require(validation.get("observed_data_evaluated") is False, "Observed data evaluated in blocked-state validation")
    require(validation.get("all_prohibited_products_absent") is True, "Observed prohibited-product absence failed")

    scope = payload.get("scientific_scope", {})
    require(scope.get("ten_percent_statistical_independence_established") is False, "Observed manifest claims 10% independence")
    require("pre-existing 2016 10% development" in str(scope.get("ten_percent_source_description", "")), "Observed 10% source caveat missing")
    require(scope.get("full_100pct_values_entered_truth_only_as_scalar_26_210MeV_normalization") is True, "Observed full-data scalar-only caveat missing")
    require(scope.get("support_specific_full_100pct_fit_p0_or_upper_limit_used_for_ranking") is False, "Observed manifest permits support-specific full-data ranking")
    return {"observed_data_evaluated": False, "observed_scan_authorized": False, "prohibited_products_absent": 10}


def check_combined_scaffold_state() -> dict[str, Any]:
    payload = load_json(HERE / "combined_scaffold_manifest.json")
    require(payload.get("status") == "scaffold_complete_awaiting_frozen_2016_support_and_reviewed_states", "Combined scaffold status drift")
    require(payload.get("production_run") is False, "Combined scaffold claims a production run")
    require(payload.get("study_id") == STUDY_ID, "Combined scaffold study_id drift")
    require(payload.get("missing_production_inputs") == ["support_freeze_decision.json", "142-row reviewed 2016 observed-state CSV"], "Combined missing-input declaration drift")
    contract = payload.get("production_contract", {})
    expected_contract = {
        "datasets": "2015-full + 2016-full + 2021-10%",
        "mass_grid_MeV": [19, 250, 1],
        "n_masses": 232,
        "combined_mode": "count_scale",
        "shared_parameter": "epsilon_squared_nonnegative",
        "n_toys_per_mass": 100,
        "mass_local_toy_limits": 23200,
        "stored_mass_local_toy_limits": 23200,
        "master_seed": 24680,
        "refit_gp_on_toy": False,
        "inner_cls": "90% asymptotic tilde_q_mu",
        "data_range_2021_GeV": [0.036, 0.3],
    }
    require(contract == expected_contract, "Combined 232-mass/23,200-toy contract drift")
    files = payload.get("files", {})
    require(isinstance(files, dict) and len(files) == 16, "Combined scaffold hash inventory drift")
    for relative, expected in files.items():
        require_sha(relative, expected)
    boundary = str(payload.get("claim_boundary", ""))
    for phrase in ("Conditional fixed-GP expected-limit quantiles", "not direct coverage", "toy-calibrated inner CLs", "global significance"):
        require(phrase in boundary, f"Combined scaffold boundary missing phrase: {phrase}")

    smoke = load_json(HERE / "qa/combined_scaffold_smoke.json")
    require(smoke.get("status") == "pass", "Combined scaffold smoke did not pass")
    require(smoke.get("production_run") is False, "Combined smoke claims a production run")
    require(smoke.get("selected_2016_support_available") is False, "Combined smoke claims a selected support")
    synthetic = smoke.get("checks", {}).get("synthetic_release_validator", {})
    require(synthetic.get("structural_mass_rows") == 232, "Synthetic validator mass count drift")
    require(synthetic.get("structural_mass_local_toy_count") == 23200, "Synthetic validator logical toy count drift")
    require(synthetic.get("stored_mass_local_toy_count") == 23200, "Synthetic validator stored toy count drift")
    require("synthetic" in str(synthetic.get("note", "")).lower(), "Synthetic-result caveat missing")
    return {"production_run": False, "n_masses_contract": 232, "n_toys_per_mass_contract": 100, "mass_local_toys_contract": 23200}


def check_prohibited_products_absent() -> dict[str, Any]:
    exact = {
        "audit/confirmation_freeze_audit.json",
        "derived/analysis/support_freeze_decision.json",
        "derived/analysis/confirmation_cell_summary.csv",
        "derived/analysis/confirmation_support_summary.csv",
        "derived/analysis/confirmation_paired_limit_differences.csv",
        "inputs/v4p9p7_observed_2016_full_frozen_support_card.yaml",
        "inputs/v4p9p7_observed_2016_full_frozen_support_card.manifest.json",
        "qa/observed_2016_review_validation.json",
        "qa/cached_profile_closure.json",
        "qa/combined_release_validation.json",
        "combined/config_combined_100toy.yaml",
        "combined/config_combined_100toy_provenance.json",
        "combined/reviewed_gp_states_v4p9p7.csv",
        "combined/reviewed_gp_states_v4p9p7_provenance.json",
        "combined/bands_100toy_cached/ul_bands_combined_all.csv",
        "combined/bands_100toy_cached/ul_bands_combined_all_provenance.json",
        "observed_scan/2016_full_primary/results_single.csv",
        "observed_scan/final_2016/optimizer_repair_plan.json",
        "observed_scan/final_2016/results_single_reviewed.csv",
        "observed_scan/final_2016/optimizer_repair_ledger.csv",
        "observed_scan/final_2016/review_summary.json",
    }
    present = sorted(relative for relative in exact if (HERE / relative).exists())
    require(not present, f"Prohibited production products exist: {present}")
    for directory in (HERE / "combined", HERE / "observed_2016", HERE / "observed_scan"):
        if directory.exists():
            files = [path for path in directory.rglob("*") if path.is_file()]
            require(not files, f"Production output directory is not empty: {directory.relative_to(HERE)}")
    unexpected_confirmation = [
        path.relative_to(HERE).as_posix()
        for path in (HERE / "derived/analysis").glob("confirmation_*")
        if path.is_file()
    ]
    require(not unexpected_confirmation, f"Unexpected Phase-2 confirmation products: {unexpected_confirmation}")
    prohibited_collection_names = {
        "accepted_extraction_rows.csv",
        "raw_primary_extraction_rows.csv",
        "optimizer_attempt_ledger.csv",
        "exclusion_ledger.csv",
        "closure_summary.csv",
        "zero_signal_bias_tests.csv",
        "task_product_audit.csv",
        "full100_accepted_rows_selected_neighbors.csv",
    }
    collection_products = sorted(
        path.relative_to(HERE).as_posix()
        for path in (HERE / "derived").rglob("*")
        if path.is_file() and path.name in prohibited_collection_names
    )
    require(not collection_products, f"Prohibited Phase-2 collection products exist: {collection_products}")
    return {"exact_prohibited_paths_checked": len(exact), "production_output_directories_empty": 3}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    return " ".join(value.casefold().split())


def check_analysis_note() -> dict[str, Any]:
    build = HERE / "note/build/full/main.pdf"
    canonical = HERE / "note/HPS_GPR_Analysis_Note_v4p9p7.pdf"
    internal_mirror = HERE / "output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf"
    pdf_paths = [build, canonical, NOTE_MIRROR]
    if internal_mirror.exists():
        pdf_paths.append(internal_mirror)
    for path in pdf_paths:
        require(path.is_file() and not path.is_symlink(), f"Missing regular final note PDF: {path}")
        require(path.read_bytes()[:5] == b"%PDF-", f"Invalid PDF signature: {path}")
    hashes = {sha256(path) for path in pdf_paths}
    sizes = {path.stat().st_size for path in pdf_paths}
    require(len(hashes) == 1 and len(sizes) == 1, "Build, campaign PDF, and release mirrors are not byte-identical")
    note_hash = next(iter(hashes))

    qa = load_json(HERE / "note/qa/note_render_qa.json")
    require(qa.get("status") == "pass", "Analysis-note QA status is not pass")
    require(qa.get("artifact") == "note/HPS_GPR_Analysis_Note_v4p9p7.pdf", "Analysis-note QA artifact path drift")
    require(qa.get("artifact_sha256") == note_hash, "Analysis-note QA hash does not bind final PDF")
    require(qa.get("artifact_bytes") == canonical.stat().st_size, "Analysis-note QA byte count drift")
    metadata = qa.get("pdf_metadata", {})
    require(metadata.get("title") == "HPS Gaussian-Process Resonance Search Analysis Note, Version 4.9.7", "Analysis-note title drift")
    require(metadata.get("pages") == 237, "Analysis note must have 237 pages")
    require(metadata.get("page_size_points") == [612, 792], "Analysis note is not US Letter")
    semantic = qa.get("semantic_checks", {})
    for key in ("replacement_characters", "unresolved_double_question_markers", "todo_markers_case_insensitive", "tbd_markers_case_insensitive", "double_open_bracket_markers", "undefined_references", "fatal_tex_errors", "overfull_boxes"):
        require(semantic.get(key) == 0, f"Analysis-note semantic/build failure: {key}")
    render = qa.get("render_checks", {})
    require(render.get("all_pages_rendered") is True and render.get("rendered_page_count") == 237, "Analysis-note full render failed")
    require(render.get("contact_sheet_page_coverage") == "1-237" and render.get("contact_sheets_visually_inspected") is True, "Analysis-note contact-sheet QA incomplete")
    require(render.get("selected_pages_visually_inspected") is True, "Analysis-note selected-page QA incomplete")
    selected_pages = set(render.get("selected_pages", []))
    require({94, 95} <= selected_pages, "Repaginated conclusion pages 94-95 were not both inspected")
    require(render.get("unexpected_blank_pages") == 0, "Analysis note has unexpected blank pages")
    require(render.get("visible_clipping_or_overlap") is False, "Analysis note has clipping/overlap")
    claims = qa.get("claim_boundary_checks", {})
    for key in ("no_selected_2016_support_claim", "no_v4p9p7_observed_2016_result_claim", "no_v4p9p7_combined_limit_claim", "no_v4p9p7_combined_band_claim", "historical_v4p2_and_v4p9p5_states_kept_distinct"):
        require(claims.get(key) is True, f"Analysis-note claim-boundary QA failed: {key}")

    extracted_path = HERE / "note/qa/main_extracted.txt"
    require(extracted_path.is_file(), "Extracted note text missing")
    require(sha256(extracted_path) == semantic.get("extracted_text_sha256"), "Extracted note text hash drift")
    extracted = extracted_path.read_text(encoding="utf-8")
    normalized = normalize_text(extracted)
    for phrase in (
        "pre-existing 2016 10% development",
        "statistical disjointness from the full sample is not established",
        "narrow-signal attribution is not robust",
        "the observed counts are unchanged",
        "fit_ok=false",
    ):
        require(phrase in normalized, f"Analysis note missing scientific boundary phrase: {phrase}")
    require(
        "provisional support edge" in normalized
        and (
            "returns no provisional edge" in normalized
            or re.search(r"there is no\d*\s+provisional support edge", normalized)
        ),
        "Analysis note omits the no-provisional-edge conclusion",
    )
    require(
        (
            "audit does not establish that the feature is background, that signal is absent" in normalized
            or "not that the feature has been proved background or that signal has been excluded" in normalized
        ),
        "Analysis note omits the signal/background non-classification boundary",
    )
    require(
        any(
            "the v4.9.7 combined upper limit and 100-toy combined band are also absent by construction"
            in normalize_text(page)
            for page in extracted.split("\f")
        ),
        "Combined-result absence sentence is missing or split across PDF pages",
    )
    for page_number, page in enumerate(extracted.split("\f"), 1):
        normalized_page = normalize_text(page)
        if "updated combined upper limit" in normalized_page:
            require(
                "100-toy combined band is produced" in normalized_page,
                f"Negated combined-result sentence is split across PDF page {page_number}",
            )

    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="strict")
        for path in sorted((HERE / "note/source").rglob("*.tex"))
    )
    normalized_source = normalize_text(source_text.replace("\\%", "%").replace("\\_", "_"))
    for phrase in ("pre-existing 2016 10% development", "no provisional support edge", "observed counts are unchanged", "not robust to the support prescription"):
        require(phrase in normalized_source, f"Analysis-note source missing boundary phrase: {phrase}")
    return {"pages": 237, "sha256": note_hash, "byte_identical_pdf_copies": len(pdf_paths), "claim_boundaries": "pass"}


def release_regular_files() -> list[Path]:
    excluded = {"release_manifest.json", "release_validation.json"}
    files: list[Path] = []
    for root, directories, names in os.walk(HERE, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *names]:
            require(not (root_path / name).is_symlink(), f"Symlink forbidden in manifest tree: {(root_path/name).relative_to(HERE)}")
        for name in names:
            path = root_path / name
            relative = path.relative_to(HERE).as_posix()
            if relative in excluded:
                continue
            require(stat.S_ISREG(path.lstat().st_mode), f"Non-regular manifest object: {relative}")
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(HERE).as_posix())


def tree_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_release_manifest() -> dict[str, Any]:
    payload = load_json(MANIFEST)
    require(payload.get("schema_version") == 1, "Release manifest schema drift")
    require(payload.get("study_id") == STUDY_ID, "Release manifest study_id drift")
    require(payload.get("release_state") == "terminal_phase1_failure_downstream_production_blocked", "Release manifest state drift")
    require(payload.get("production_authorized") is False, "Release manifest authorizes production")
    require(payload.get("hash_algorithm") == "sha256", "Release manifest hash algorithm drift")
    require(payload.get("excluded_mutable_or_self_referential_paths") == ["release_manifest.json", "release_validation.json"], "Release manifest exclusions drift")
    require(payload.get("allowed_roles") == ["canonical", "lineage", "scaffold", "qa"], "Release manifest roles drift")
    role_semantics = payload.get("role_semantics")
    require(
        isinstance(role_semantics, dict)
        and set(role_semantics) == {"canonical", "lineage", "scaffold", "qa"}
        and all(isinstance(value, str) and value for value in role_semantics.values()),
        "Release manifest role semantics are incomplete",
    )

    entries = payload.get("files")
    require(isinstance(entries, list) and entries, "Release manifest file list missing")
    paths = [entry.get("path") for entry in entries]
    require(paths == sorted(paths) and len(paths) == len(set(paths)), "Release manifest paths are not unique and sorted")
    actual_files = release_regular_files()
    actual_paths = [path.relative_to(HERE).as_posix() for path in actual_files]
    require(paths == actual_paths, f"Release manifest inventory drift: missing={sorted(set(actual_paths)-set(paths))[:10]}, extra={sorted(set(paths)-set(actual_paths))[:10]}")
    allowed_roles = set(payload["allowed_roles"])
    role_counts: Counter[str] = Counter()
    role_bytes: defaultdict[str, int] = defaultdict(int)
    for entry, path in zip(entries, actual_files):
        require(set(entry) == {"path", "bytes", "sha256", "role", "role_detail"}, f"Manifest entry schema drift: {entry.get('path')}")
        require(entry["role"] in allowed_roles and entry["role_detail"], f"Manifest role drift: {entry['path']}")
        require(int(entry["bytes"]) == path.stat().st_size, f"Manifest byte count drift: {entry['path']}")
        require(entry["sha256"] == sha256(path), f"Manifest hash drift: {entry['path']}")
        role_counts[entry["role"]] += 1
        role_bytes[entry["role"]] += int(entry["bytes"])
    entries_by_path = {entry["path"]: entry for entry in entries}
    expected_representative_roles = {
        "study_spec.json": "canonical",
        "derived/analysis/phase1_selection_decision.json": "canonical",
        "observed_2016_workflow_manifest.json": "canonical",
        "signal_audit/derived/signal_robustness_summary.json": "canonical",
        "inputs/source_2016_full.root": "lineage",
        "signal_audit/source_snapshots/new_v4p9p5_2021_curve.csv": "lineage",
        "combined_scaffold_manifest.json": "scaffold",
        "run_combined_bands_cached_fixed_reviewed.py": "scaffold",
        "qa/truth_product_validation.json": "qa",
        "signal_audit/qa/semantic_validation.json": "qa",
        "validate_release.py": "qa",
    }
    for relative, expected_role in expected_representative_roles.items():
        require(
            entries_by_path.get(relative, {}).get("role") == expected_role,
            f"Manifest semantic role drift: {relative}",
        )
    require(set(role_counts) == allowed_roles, f"Manifest does not use all four roles: {sorted(role_counts)}")
    summary = payload.get("summary", {})
    require(summary.get("regular_file_count") == len(entries), "Manifest file-count summary drift")
    require(summary.get("regular_file_bytes") == sum(int(entry["bytes"]) for entry in entries), "Manifest byte summary drift")
    require(summary.get("file_count_by_role") == dict(sorted(role_counts.items())), "Manifest role-count summary drift")
    require(summary.get("bytes_by_role") == dict(sorted(role_bytes.items())), "Manifest role-byte summary drift")
    require(payload.get("tree_sha256") == tree_sha256(entries), "Manifest tree SHA-256 drift")

    external = payload.get("external_mirrors", {}).get("analysis_note_repository_mirror", {})
    require(external.get("path_from_repository_root") == "output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf", "External note mirror path drift")
    require(external.get("present") is True, "External note mirror absent from manifest")
    require(NOTE_MIRROR.is_file(), "External note mirror missing")
    require(external.get("bytes") == NOTE_MIRROR.stat().st_size, "External note mirror byte binding drift")
    require(external.get("sha256") == sha256(NOTE_MIRROR), "External note mirror hash binding drift")
    return {"regular_files": len(entries), "tree_sha256": payload["tree_sha256"], "role_counts": dict(sorted(role_counts.items()))}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    checks: list[dict[str, Any]] = []

    def run(name: str, function: Callable[[], dict[str, Any]]) -> None:
        try:
            details = function()
        except Exception as error:  # collect all independent release defects
            checks.append({"name": name, "status": "fail", "error": str(error)})
        else:
            checks.append({"name": name, "status": "pass", "details": details})

    run("filesystem_regular_no_symlinks", check_filesystem_objects)
    run("terminal_no_edge_decision_and_audits", check_terminal_decision_and_audits)
    run("phase1_175_task_inventory_and_success", check_phase1_task_inventory)
    run("phase1_aggregate_tables_2098_84_7_72", check_phase1_aggregate_tables)
    run("truth_product_and_static_scientific_caveats", check_truth_and_static_caveats)
    run("signal_robustness_manifest_29_and_semantics_13_of_13", check_signal_robustness_audit)
    run("observed_2016_blocked_state", check_observed_blocked_state)
    run("combined_232_mass_23200_toy_scaffold_only", check_combined_scaffold_state)
    run("prohibited_production_products_absent", check_prohibited_products_absent)
    run("analysis_note_pdf_mirrors_and_claim_boundaries", check_analysis_note)
    run("recursive_release_manifest", check_release_manifest)

    failures = [check for check in checks if check["status"] != "pass"]
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "pass" if not failures else "fail",
        "release_state": "terminal_phase1_failure_downstream_production_blocked",
        "production_authorized": False,
        "observed_scan_authorized": False,
        "combined_result_authorized": False,
        "passing_check_count": len(checks) - len(failures),
        "failing_check_count": len(failures),
        "checks": checks,
        "claim_boundary": (
            "A pass validates a terminal conditional source-recovery failure and the "
            "absence of downstream production. It is not an observed exclusion, calibrated "
            "sensitivity, coverage statement, global significance, or signal/background classification."
        ),
    }
    atomic_write_json(REPORT, report)
    if failures:
        print(f"FAIL: {len(failures)} of {len(checks)} release checks failed", file=sys.stderr)
        for check in failures:
            print(f"  - {check['name']}: {check['error']}", file=sys.stderr)
        print(f"Wrote {REPORT}", file=sys.stderr)
        return 2
    print(f"PASS: all {len(checks)} release checks passed; production remains blocked")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
