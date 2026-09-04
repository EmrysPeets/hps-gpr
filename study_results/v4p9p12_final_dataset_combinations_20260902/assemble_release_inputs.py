#!/usr/bin/env python3
"""Assemble the exact, certified inputs for the v4.9.12 final combinations.

This script is deliberately fail closed.  It writes no final-combination input
unless the 2015 selected states replay as the maximum-LML branch among the
three archived attempts, the frozen 2021 10% release closes, and the uniform
2016 remediation authorizes all 142 states at support 30--210 MeV with the
resolution-scaled upper length-scale factor fixed to 12.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUTS = HERE / "inputs"
CERTIFICATIONS = INPUTS / "certifications"

V4P9P7 = REPO / "study_results/v4p9p7_2016_support_combined_100toy_20260902"
V4P9P5 = REPO / "study_results/v4p9p5_2021_gp_support_edge_optimization_20260820"
V4P9P11 = REPO / "study_results/v4p9p11_2016_reference30_state_certification_20260902"
V4P9P11P1 = REPO / "study_results/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902"

ARCHIVED_2015 = V4P9P7 / "inputs/archived_2015_source_ledger.csv"
ATTEMPT_2015 = tuple(
    REPO
    / f"study_results/v4_wide_support_2015full_2016full_2021_10pct_20260803/observed_attempt_0{index}/results_single.csv"
    for index in (1, 2, 3)
)
SOURCE_2021 = V4P9P5 / "derived/analysis/observed_2021_10pct_support036_300.csv"
PRIMARY_2021 = V4P9P5 / "observed_scan/support036_300/results_single.csv"
REPAIRED_2021 = V4P9P5 / "observed_scan/final/results_single_repaired.csv"
REPEAT_2021 = {
    (mass, repeat): V4P9P5
    / f"observed_scan/unchanged_card_repeats/m{mass:03d}/repeat{repeat}/results_single.csv"
    for mass in (94, 152, 212)
    for repeat in (1, 2, 3)
}
SOURCE_2016 = V4P9P11P1 / "derived/observed_2016_gp_states_reviewed.csv"
DECISION_2016 = V4P9P11P1 / "derived/state_certification_decision.json"
PATHS_2016 = V4P9P11P1 / "derived/optimizer_paths.csv"
EXCEPTION_2016 = INPUTS / "2016_PROVISIONAL_STATE_NUMERICAL_EXCEPTION.json"

FORBIDDEN_INFERENCE_FIELDS = {
    "A_hat",
    "signal_pull",
    "p0",
    "Z",
    "upper_limit",
    "epsilon2",
    "eps2",
    "expected_band",
    "toy_limit",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_sha256(frame: pd.DataFrame) -> str:
    payload = [
        {key: float(value) for key, value in row.items()}
        for row in frame.sort_values("mass_GeV")[["mass_GeV", "const_opt", "ls_opt", "lml"]]
        .astype(float)
        .to_dict(orient="records")
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def artifact(path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"missing bound artifact: {resolved}")
    return {"path": str(resolved), "sha256": sha256(resolved)}


def json_file(path: Path) -> dict:
    require(path.is_file(), f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_hash_ledger(path: Path) -> None:
    """Replay every SHA-256 entry in a whitespace-delimited freeze ledger."""

    require(path.is_file(), f"missing frozen hash ledger: {path}")
    rows = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, relative = raw.split(maxsplit=1)
        target = (path.parent / relative.strip()).resolve()
        require(target.is_file(), f"frozen ledger target missing: {target}")
        require(sha256(target) == digest, f"frozen ledger target drift: {target}")
        rows += 1
    require(rows > 0, f"empty frozen hash ledger: {path}")


def no_forbidden_columns(frame: pd.DataFrame, label: str) -> None:
    lowered = {str(column).lower() for column in frame.columns}
    multi_character_tokens = {
        token.lower() for token in FORBIDDEN_INFERENCE_FIELDS if len(token) > 1
    }
    bad = sorted(
        column
        for column in lowered
        if any(token in column for token in multi_character_tokens)
        or column == "z"
        or column.startswith("z_")
        or column.endswith("_z")
        or "local_z" in column
    )
    require(not bad, f"{label} contains forbidden inference columns: {bad}")


def exact_grid(frame: pd.DataFrame, dataset: str, low: int, high: int) -> pd.DataFrame:
    selected = frame.copy()
    if "dataset" in selected.columns:
        selected = selected[selected.dataset.astype(str) == dataset].copy()
    masses = np.rint(1000.0 * selected.mass_GeV.astype(float)).astype(int)
    selected["mass_MeV_join"] = masses
    selected = selected.sort_values("mass_MeV_join").reset_index(drop=True)
    require(
        np.array_equal(selected.mass_MeV_join.to_numpy(int), np.arange(low, high + 1)),
        f"{dataset} source does not have the exact {low}--{high} MeV grid",
    )
    require(not selected.mass_MeV_join.duplicated().any(), f"{dataset} source grid is duplicated")
    require(
        np.isfinite(selected[["mass_GeV", "const_opt", "ls_opt", "lml"]].to_numpy(float)).all(),
        f"{dataset} source has non-finite coordinates",
    )
    require((selected.const_opt.astype(float) > 0).all() and (selected.ls_opt.astype(float) > 0).all(), f"{dataset} source has non-positive coordinates")
    return selected


def validate_2015() -> tuple[pd.DataFrame, Dict[str, dict], dict]:
    archived_columns = [
        "dataset",
        "mass_GeV",
        "const_opt",
        "ls_opt",
        "lml",
        "interpolated",
        "selected_source",
        "selected_source_sha256",
        "review_status",
        "branch_multiplicity",
        "candidate_count",
    ]
    archived = exact_grid(
        pd.read_csv(ARCHIVED_2015, usecols=archived_columns), "2015", 19, 90
    )
    require((archived.review_status.astype(str) == "resolved_reproduced_max_lml").all(), "2015 review status drift")
    require((archived.branch_multiplicity.astype(int) == 3).all(), "2015 branch multiplicity drift")
    require((archived.candidate_count.astype(int) == 3).all(), "2015 candidate count drift")
    require(not archived.interpolated.astype(bool).any(), "2015 interpolation is forbidden")

    attempts = []
    attempt_by_path = {}
    for index, path in enumerate(ATTEMPT_2015, start=1):
        fit_columns = [
            "dataset",
            "mass_GeV",
            "const_opt",
            "ls_opt",
            "lml",
            "train_domain_lo",
            "train_domain_hi",
            "ls_hi_over_sigma_x",
            "optimizer_restarts",
            "n_train_low",
            "n_train_high",
            "const_at_lower",
            "const_at_upper",
            "ls_at_lower",
            "ls_at_upper",
        ]
        frame = exact_grid(pd.read_csv(path, usecols=fit_columns), "2015", 19, 90)
        require(
            (frame.train_domain_lo.astype(float) == 0.014).all()
            and (frame.train_domain_hi.astype(float) == 0.135).all()
            and (frame.ls_hi_over_sigma_x.astype(float) == 8.0).all()
            and (frame.optimizer_restarts.astype(int) == 12).all()
            and (frame.n_train_low.astype(int) > 0).all()
            and (frame.n_train_high.astype(int) > 0).all(),
            f"2015 attempt {index} support, k8, restart, or two-sideband evidence drift",
        )
        require(
            not frame[["const_at_lower", "const_at_upper", "ls_at_lower", "ls_at_upper"]]
            .astype(bool)
            .any(axis=None),
            f"2015 attempt {index} contains kernel-bound contact",
        )
        frame = frame[["mass_MeV_join", "mass_GeV", "const_opt", "ls_opt", "lml"]].copy()
        frame["attempt"] = index
        frame["attempt_path"] = str(path.resolve())
        attempts.append(frame)
        attempt_by_path[str(path.resolve())] = frame
    candidates = pd.concat(attempts, ignore_index=True)
    maximum = candidates.groupby("mass_MeV_join", as_index=False).lml.max().rename(columns={"lml": "max_lml"})
    replay = archived.merge(maximum, on="mass_MeV_join", validate="one_to_one")
    require(
        np.allclose(replay.lml, replay.max_lml, rtol=2.0e-13, atol=2.0e-13),
        "2015 archived selected LML does not replay as the maximum over three attempts",
    )
    for row in archived.itertuples(index=False):
        selected_path = (REPO / str(row.selected_source)).resolve()
        require(str(selected_path) in attempt_by_path, f"2015 selected source is not one of the three attempts at {row.mass_MeV_join} MeV")
        require(
            str(row.selected_source_sha256) == sha256(selected_path),
            f"2015 selected-source hash drift at {row.mass_MeV_join} MeV",
        )
        selected_attempt = attempt_by_path[str(selected_path)]
        matched = selected_attempt[selected_attempt.mass_MeV_join == int(row.mass_MeV_join)]
        require(len(matched) == 1, f"2015 selected source row missing at {row.mass_MeV_join} MeV")
        matched_row = matched.iloc[0]
        for coordinate in ("mass_GeV", "const_opt", "ls_opt", "lml"):
            require(
                np.isclose(
                    float(getattr(row, coordinate)),
                    float(matched_row[coordinate]),
                    rtol=2.0e-13,
                    atol=2.0e-13,
                ),
                f"2015 selected-source coordinate drift at {row.mass_MeV_join} MeV: {coordinate}",
            )

    roles = {"archived_state_ledger": artifact(ARCHIVED_2015)}
    roles.update(
        {f"selected_source_attempt_{index}": artifact(path) for index, path in enumerate(ATTEMPT_2015, start=1)}
    )
    evidence = {
        "mass_rows": 72,
        "mass_grid_MeV": [19, 90],
        "attempt_count_per_mass": 3,
        "support_MeV": [14, 135],
        "upper_length_factor": 8.0,
        "optimizer_restarts": 12,
        "two_sidebands_present_all_rows": True,
        "kernel_bound_contacts": 0,
        "max_lml_replay_pass": True,
        "maximum_abs_selected_minus_attempt_max_lml": float(np.max(np.abs(replay.lml - replay.max_lml))),
        "selection_rule": "maximum LML over three unchanged-card attempts, with exact selected-coordinate reproduction",
    }
    return archived, roles, evidence


def validate_2021() -> tuple[pd.DataFrame, Dict[str, dict], dict]:
    fit_coordinates = ["mass_GeV", "const_opt", "ls_opt", "lml"]
    states = exact_grid(pd.read_csv(SOURCE_2021, usecols=fit_coordinates), "2021", 50, 250)
    repaired_states = exact_grid(
        pd.read_csv(REPAIRED_2021, usecols=fit_coordinates), "2021", 50, 250
    )
    primary_columns = ["mass_GeV", "const_opt", "ls_opt", "lml", "n_train"]
    primary = exact_grid(pd.read_csv(PRIMARY_2021, usecols=primary_columns), "2021", 50, 250)
    freeze = json_file(V4P9P5 / "derived/analysis/support_freeze_decision.json")
    repair = json_file(V4P9P5 / "observed_scan/final/optimizer_repair_summary.json")
    validation = json_file(V4P9P5 / "qa/release_validation.json")
    require(
        freeze.get("status") == "support_edge_frozen"
        and freeze.get("selected_support") == "036_300"
        and int(freeze.get("selected_support_low_MeV", -1)) == 36
        and int(freeze.get("support_high_MeV", -1)) == 300
        and freeze.get("observed_scan_authorized") is True,
        "2021 support-freeze decision drift",
    )
    require(
        repair.get("status") == "pass"
        and int(repair.get("rows", -1)) == 201
        and repair.get("results_sha256") == sha256(REPAIRED_2021)
        and repair.get("repaired_masses_MeV") == [94, 152, 212],
        "2021 optimizer-repair summary drift",
    )
    for coordinate in ("mass_GeV", "const_opt", "ls_opt", "lml"):
        require(
            np.allclose(
                states[coordinate].to_numpy(float),
                repaired_states[coordinate].to_numpy(float),
                rtol=2.0e-13,
                atol=2.0e-13,
            ),
            f"2021 compact state source differs from repaired result ledger: {coordinate}",
        )

    # Current fit-only requalification of the three historical repairs.  The
    # historical helper also compared sigma_A; this replay intentionally does
    # not load that inference-facing field.
    selected_names = {}
    selected_replicates = {}
    for mass in (94, 152, 212):
        candidates = []
        original = primary[primary.mass_MeV_join == mass].iloc[0].copy()
        candidates.append(("primary", original))
        for repeat in (1, 2, 3):
            repeat_frame = pd.read_csv(REPEAT_2021[(mass, repeat)], usecols=primary_columns)
            require(len(repeat_frame) == 1, f"2021 repeat row count drift: {mass}/{repeat}")
            repeat_frame["mass_MeV_join"] = np.rint(1000.0 * repeat_frame.mass_GeV.astype(float)).astype(int)
            require(int(repeat_frame.mass_MeV_join.iloc[0]) == mass, f"2021 repeat mass drift: {mass}/{repeat}")
            candidates.append((f"repeat{repeat}", repeat_frame.iloc[0]))

        def branch_match(left: pd.Series, right: pd.Series) -> bool:
            values = ("lml", "ls_opt", "const_opt", "n_train")
            if not all(np.isfinite(float(left[key])) and np.isfinite(float(right[key])) for key in values):
                return False
            n_train = max(1.0, min(float(left.n_train), float(right.n_train)))
            if abs(float(left.lml) - float(right.lml)) / n_train > 0.001:
                return False
            return (
                abs(math.log(float(left.ls_opt) / float(right.ls_opt))) <= 0.01
                and abs(math.log(float(left.const_opt) / float(right.const_opt))) <= 0.05
            )

        ordered = sorted(candidates, key=lambda item: float(item[1].lml), reverse=True)
        chosen = None
        for name, candidate in ordered:
            replicates = sum(branch_match(candidate, other) for _, other in candidates)
            if replicates >= 2:
                chosen = (name, candidate, replicates)
                break
        require(chosen is not None, f"2021 fit-only repair requalification failed at {mass} MeV")
        name, selected, replicates = chosen
        selected_names[mass] = name
        selected_replicates[mass] = int(replicates)
        final = repaired_states[repaired_states.mass_MeV_join == mass].iloc[0]
        for coordinate in fit_coordinates:
            require(
                np.isclose(float(final[coordinate]), float(selected[coordinate]), rtol=2.0e-13, atol=2.0e-13),
                f"2021 repaired coordinate does not match fit-only selection at {mass} MeV: {coordinate}",
            )
    require(
        selected_names == {94: "repeat1", 152: "repeat1", 212: "repeat2"}
        and selected_replicates == {94: 3, 152: 3, 212: 2},
        "2021 fit-only selected branch inventory drift",
    )
    repair_ledger_path = V4P9P5 / "observed_scan/final/optimizer_repair_ledger.csv"
    repair_ledger = pd.read_csv(
        repair_ledger_path,
        usecols=[
            "mass_MeV",
            "candidate",
            "source_csv",
            "selected",
            "selected_branch_replicates",
            "lml",
            "ls_opt",
            "const_opt",
        ],
    )
    selected_ledger = repair_ledger[repair_ledger.selected.astype(bool)].copy()
    require(len(selected_ledger) == 3, "2021 historical repair ledger selected-row count drift")
    selected_source_hashes = {}
    for mass, candidate in selected_names.items():
        ledger_row = selected_ledger[selected_ledger.mass_MeV.astype(int) == mass]
        require(len(ledger_row) == 1, f"2021 historical selected repair row missing at {mass} MeV")
        ledger_row = ledger_row.iloc[0]
        expected_path = REPEAT_2021[(mass, int(candidate.removeprefix("repeat")))]
        expected_relative = str(expected_path.relative_to(V4P9P5))
        require(
            str(ledger_row.candidate) == candidate
            and str(ledger_row.source_csv) == expected_relative
            and int(ledger_row.selected_branch_replicates) == selected_replicates[mass],
            f"2021 historical selected repair provenance drift at {mass} MeV",
        )
        selected = pd.read_csv(expected_path, usecols=primary_columns).iloc[0]
        for coordinate in ("lml", "ls_opt", "const_opt"):
            require(
                np.isclose(float(ledger_row[coordinate]), float(selected[coordinate]), rtol=2.0e-13, atol=2.0e-13),
                f"2021 historical selected repair coordinate drift at {mass} MeV: {coordinate}",
            )
        selected_source_hashes[str(mass)] = sha256(expected_path)
    unchanged = repaired_states[~repaired_states.mass_MeV_join.isin((94, 152, 212))]
    unchanged_primary = primary[~primary.mass_MeV_join.isin((94, 152, 212))]
    for coordinate in fit_coordinates:
        require(
            np.allclose(
                unchanged[coordinate].to_numpy(float),
                unchanged_primary[coordinate].to_numpy(float),
                rtol=2.0e-13,
                atol=2.0e-13,
            ),
            f"2021 non-repaired rows differ from primary: {coordinate}",
        )
    require(
        validation.get("status") == "pass"
        and int(validation.get("checks_passed", -1)) == int(validation.get("checks_total", -2))
        and all(bool(item.get("pass")) for item in validation.get("checks", [])),
        "2021 release validation did not pass",
    )
    roles = {
        "study_protocol": artifact(V4P9P5 / "STUDY_PROTOCOL.md"),
        "observed_card": artifact(V4P9P5 / "inputs/v4p9p5_observed_2021_10pct_support036_300_card.yaml"),
        "support_freeze_decision": artifact(V4P9P5 / "derived/analysis/support_freeze_decision.json"),
        "observed_state_ledger": artifact(SOURCE_2021),
        "primary_result_ledger": artifact(PRIMARY_2021),
        "repaired_full_result_ledger": artifact(REPAIRED_2021),
        "repair_script": artifact(V4P9P5 / "repair_observed_scan.py"),
        "optimizer_repair_ledger": artifact(repair_ledger_path),
        "optimizer_repair_summary": artifact(V4P9P5 / "observed_scan/final/optimizer_repair_summary.json"),
        "release_validation": artifact(V4P9P5 / "qa/release_validation.json"),
    }
    roles.update(
        {
            f"unchanged_repeat_m{mass:03d}_{repeat}": artifact(path)
            for (mass, repeat), path in sorted(REPEAT_2021.items())
        }
    )
    evidence = {
        "mass_rows": 201,
        "mass_grid_MeV": [50, 250],
        "support_MeV": [36, 300],
        "repaired_masses_MeV": [94, 152, 212],
        "fit_only_requalification": {
            "loaded_fields": primary_columns,
            "branch_match_fields": ["lml", "ls_opt", "const_opt", "n_train"],
            "selected_candidate_by_mass_MeV": {str(key): value for key, value in selected_names.items()},
            "selected_branch_replicates_by_mass_MeV": {str(key): value for key, value in selected_replicates.items()},
            "selected_source_sha256_by_mass_MeV": selected_source_hashes,
            "historical_caveat": "the archived repair helper also used sigma_A; the current fit-only replay recovers the identical branches without loading sigma_A",
        },
        "release_checks_passed": int(validation["checks_passed"]),
    }
    return states, roles, evidence


def validate_2016() -> tuple[pd.DataFrame, Dict[str, dict], dict]:
    inherited_freezes = (
        "FROZEN_PROTOCOL_SHA256",
        "FROZEN_EXECUTION_SHA256",
        "FROZEN_CONTROL_PASS_SHA256",
        "FROZEN_DOWNSTREAM_EXECUTION_SHA256",
        "FROZEN_ARCHIVE_CLASSIFICATION_SHA256",
        "TERMINAL_STATE_SHA256",
        "FINAL_VALIDATION_SHA256",
    )
    for name in inherited_freezes:
        validate_hash_ledger(V4P9P11 / name)
    validate_hash_ledger(V4P9P11P1 / "FROZEN_PROTOCOL_SHA256")
    validate_hash_ledger(V4P9P11P1 / "FROZEN_EXECUTION_SHA256")
    validate_hash_ledger(V4P9P11P1 / "TERMINAL_STATE_SHA256")
    terminal = json_file(V4P9P11 / "derived/state_certification_decision.json")
    control = json_file(V4P9P11 / "derived/control_adequacy/control_decision_initial_frozen.json")
    terminal_validation = json_file(V4P9P11 / "qa/final_validation.json")
    preflight = json_file(V4P9P11P1 / "qa/preflight.json")
    decision = json_file(DECISION_2016)
    release_validation = json_file(V4P9P11P1 / "qa/final_validation.json")
    terminal_p1 = json_file(V4P9P11P1 / "TERMINAL_RELEASE_STATUS.json")
    exception = json_file(EXCEPTION_2016)
    paths = pd.read_csv(PATHS_2016)
    states = exact_grid(pd.read_csv(SOURCE_2016), "2016", 39, 180)

    require(
        control.get("status") == "control_adequacy_pass"
        and control.get("technical_pass") is True
        and control.get("absolute_guard_pass") is True
        and control.get("forbidden_centers_zero") is True,
        "inherited 2016 canonical control did not pass",
    )
    require(
        terminal.get("status") == "stopped_unresolved_state"
        and terminal.get("combination_authorized") is False
        and int(terminal.get("state_rows", -1)) == 142
        and int(terminal.get("resolved_rows", -1)) == 49
        and len(terminal.get("unresolved_masses_MeV", [])) == 93
        and terminal.get("inference_fields_accessed") == [],
        "inherited 2016 terminal-stop record drift",
    )
    require(
        terminal_validation.get("status") == "pass"
        and int(terminal_validation.get("checks_passed", -1))
        == int(terminal_validation.get("checks_total", -2))
        and all(bool(item.get("pass")) for item in terminal_validation.get("checks", [])),
        "inherited 2016 terminal validation did not pass",
    )
    require(
        terminal_validation.get("canonical_state_decision_sha256") == sha256(V4P9P11 / "derived/state_certification_decision.json")
        and terminal_validation.get("validator_sha256") == sha256(V4P9P11 / "validate_release.py"),
        "inherited 2016 validation summary hashes do not close",
    )
    require(
        preflight.get("status") == "preflight_pass"
        and int(preflight.get("mass_rows", -1)) == 142
        and int(preflight.get("planned_path_rows", -1)) == 2556,
        "2016 p1 preflight did not pass",
    )
    require(len(paths) == 2556, "2016 p1 optimizer path ledger is not 2,556 rows")
    require(
        np.array_equal(np.sort(paths.mass_MeV.astype(int).unique()), np.arange(39, 181))
        and (paths.groupby(paths.mass_MeV.astype(int)).size() == 18).all(),
        "2016 p1 does not contain exactly 18 paths for every mass",
    )
    no_forbidden_columns(paths, "2016 optimizer-path ledger")
    no_forbidden_columns(states, "2016 state ledger")
    require(
        decision.get("status") == "all_142_states_certified"
        and decision.get("combination_authorized") is True
        and int(decision.get("state_rows", -1)) == 142
        and int(decision.get("resolved_rows", -1)) == 142
        and decision.get("unresolved_masses_MeV") == []
        and int(decision.get("support_lower_MeV", -1)) == 30
        and int(decision.get("support_upper_MeV", -1)) == 210
        and float(decision.get("upper_length_factor_2016", float("nan"))) == 12.0
        and decision.get("states") == {"rows": 142, "sha256": sha256(SOURCE_2016)}
        and decision.get("optimizer_paths") == {"rows": 2556, "sha256": sha256(PATHS_2016)},
        "2016 p1 decision did not authorize the exact state and path ledgers",
    )
    require(
        release_validation.get("status") == "validation_failure"
        and release_validation.get("all_checks_pass") is False
        and release_validation.get("canonical_outcome") == "all_142_states_certified"
        and int(release_validation.get("checks_passed", -1)) == 14
        and int(release_validation.get("checks_total", -1)) == 17
        and [item["name"] for item in release_validation.get("checks", []) if not item.get("pass")]
        == [
            "selected_state_prediction_covariance_replay",
            "decision_semantics_and_hashes",
            "global_stop_or_complete_rule",
        ],
        "2016 p1 failed-validation record drift",
    )
    require(
        release_validation.get("canonical_decision_sha256") == sha256(DECISION_2016)
        and release_validation.get("optimizer_paths_sha256") == sha256(PATHS_2016)
        and release_validation.get("states_sha256") == sha256(SOURCE_2016)
        and release_validation.get("protocol_sha256") == sha256(V4P9P11P1 / "STUDY_PROTOCOL.md")
        and release_validation.get("spec_sha256") == sha256(REPO / "study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json")
        and release_validation.get("runner_sha256") == sha256(V4P9P11P1 / "run_uniform_remediation.py")
        and release_validation.get("preflight_sha256") == sha256(V4P9P11P1 / "qa/preflight.json")
        and release_validation.get("validator_sha256") == sha256(V4P9P11P1 / "validate_release.py")
        and release_validation.get("inference_fields_accessed") == [],
        "2016 p1 validation summary hashes do not close",
    )
    require(
        terminal_p1.get("status") == "terminal_failed_independent_validation"
        and terminal_p1.get("combination_authorized") is False
        and terminal_p1.get("independent_validation", {}).get("independently_resolved_rows") == 87
        and terminal_p1.get("independent_validation", {}).get("independently_unresolved_rows") == 55,
        "2016 p1 terminal status drift",
    )
    evidence = dict(exception.get("p1_evidence", {}))
    require(
        exception.get("status") == "conditional_user_accepted_numerical_exception"
        and exception.get("record_type") == "explicit_numerical_exception_for_provisional_fixed_state_consumption"
        and exception.get("p1_combination_authorized") is False
        and exception.get("independent_state_certification") is False
        and exception.get("exception_accepts_exact_frozen_coordinates_without_reoptimization") is True
        and exception.get("support_2016_MeV") == [30, 210]
        and float(exception.get("upper_length_factor_2016", float("nan"))) == 12.0
        and evidence.get("provisional_states_sha256") == sha256(SOURCE_2016)
        and evidence.get("optimizer_paths_sha256") == sha256(PATHS_2016)
        and evidence.get("provisional_decision_sha256") == sha256(DECISION_2016)
        and evidence.get("failed_validator_sha256") == sha256(V4P9P11P1 / "qa/final_validation.json")
        and evidence.get("terminal_status_sha256") == sha256(V4P9P11P1 / "TERMINAL_RELEASE_STATUS.json")
        and evidence.get("terminal_ledger_sha256") == sha256(V4P9P11P1 / "TERMINAL_STATE_SHA256"),
        "2016 downstream numerical-exception record drift",
    )
    if "state_resolved" in states.columns:
        require(states.state_resolved.astype(bool).all(), "2016 p1 source contains an unresolved state")

    spec = REPO / "study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json"
    inherited_spec = REPO / "study_configs/v4p9p11_2016_reference30_state_certification_20260902/study_spec.json"
    roles = {
        "v4p9p11_study_protocol": artifact(V4P9P11 / "STUDY_PROTOCOL.md"),
        "v4p9p11_study_spec": artifact(inherited_spec),
        "v4p9p11_frozen_protocol": artifact(V4P9P11 / "FROZEN_PROTOCOL_SHA256"),
        "v4p9p11_canonical_control_freeze": artifact(V4P9P11 / "FROZEN_CONTROL_PASS_SHA256"),
        "v4p9p11_canonical_control_script": artifact(V4P9P11 / "run_control_frozen.py"),
        "v4p9p11_canonical_control_decision": artifact(V4P9P11 / "derived/control_adequacy/control_decision_initial_frozen.json"),
        "v4p9p11_control_attempt_ledger": artifact(V4P9P11 / "derived/control_adequacy/optimizer_attempts.csv"),
        "v4p9p11_control_cell_ledger": artifact(V4P9P11 / "derived/control_adequacy/selected_cells.csv"),
        "v4p9p11_code_split_amendment": artifact(V4P9P11 / "PRE_ARCHIVE_CODE_SPLIT_AMENDMENT.md"),
        "v4p9p11_downstream_freeze": artifact(V4P9P11 / "FROZEN_DOWNSTREAM_EXECUTION_SHA256"),
        "v4p9p11_downstream_script": artifact(V4P9P11 / "run_downstream_certification.py"),
        "v4p9p11_archive_freeze": artifact(V4P9P11 / "FROZEN_ARCHIVE_CLASSIFICATION_SHA256"),
        "v4p9p11_archive_decision": artifact(V4P9P11 / "derived/archive_certification/archive_class_decision.json"),
        "v4p9p11_archive_state_certificates": artifact(V4P9P11 / "derived/archive_certification/archived_state_certificates.csv"),
        "v4p9p11_robust_attempt_ledger": artifact(V4P9P11 / "derived/robust_repeats/optimizer_attempts.csv"),
        "v4p9p11_robust_selected_state_ledger": artifact(V4P9P11 / "derived/robust_repeats/selected_states.csv"),
        "v4p9p11_terminal_freeze": artifact(V4P9P11 / "TERMINAL_STATE_SHA256"),
        "v4p9p11_terminal_decision": artifact(V4P9P11 / "derived/state_certification_decision.json"),
        "v4p9p11_terminal_validation": artifact(V4P9P11 / "qa/final_validation.json"),
        "v4p9p11_final_validation_freeze": artifact(V4P9P11 / "FINAL_VALIDATION_SHA256"),
        "v4p9p11_release_validator": artifact(V4P9P11 / "validate_release.py"),
        "p1_study_protocol": artifact(V4P9P11P1 / "STUDY_PROTOCOL.md"),
        "p1_study_spec": artifact(spec),
        "p1_frozen_protocol": artifact(V4P9P11P1 / "FROZEN_PROTOCOL_SHA256"),
        "p1_execution_freeze": artifact(V4P9P11P1 / "FROZEN_EXECUTION_SHA256"),
        "p1_runner": artifact(V4P9P11P1 / "run_uniform_remediation.py"),
        "p1_preflight": artifact(V4P9P11P1 / "qa/preflight.json"),
        "p1_optimizer_path_ledger": artifact(PATHS_2016),
        "p1_state_ledger": artifact(SOURCE_2016),
        "p1_final_support_decision": artifact(DECISION_2016),
        "p1_release_validator": artifact(V4P9P11P1 / "validate_release.py"),
        "p1_release_validation": artifact(V4P9P11P1 / "qa/final_validation.json"),
        "p1_terminal_status": artifact(V4P9P11P1 / "TERMINAL_RELEASE_STATUS.json"),
        "p1_terminal_ledger": artifact(V4P9P11P1 / "TERMINAL_STATE_SHA256"),
        "downstream_numerical_exception": artifact(EXCEPTION_2016),
    }
    evidence = {
        "mass_rows": 142,
        "mass_grid_MeV": [39, 180],
        "support_MeV": [30, 210],
        "upper_length_factor": 12.0,
        "optimizer_path_rows": 2556,
        "optimizer_paths_per_mass": 18,
        "provisional_states_resolved_by_runner": 142,
        "independently_resolved_states": 87,
        "independent_state_certification": False,
        "downstream_status": "conditional_user_accepted_numerical_exception",
        "selection_fields_excluded": sorted(FORBIDDEN_INFERENCE_FIELDS),
        "inherited_terminal_outcome_preserved": "stopped_unresolved_state",
    }
    return states, roles, evidence


def selected_rows(
    source: pd.DataFrame,
    dataset: str,
    support: tuple[int, int],
    source_path: Path,
    combination_authorization_sha: str,
    dataset_support_decision_sha: str,
    source_state: str,
) -> pd.DataFrame:
    result = source[["mass_GeV", "const_opt", "ls_opt", "lml"]].copy()
    result.insert(0, "dataset", dataset)
    result["interpolated"] = False
    result["gp_support_low_MeV"] = support[0]
    result["gp_support_high_MeV"] = support[1]
    result["source_state"] = source_state
    result["source_ledger_path"] = str(source_path.resolve())
    result["source_ledger_sha256"] = sha256(source_path)
    result["combination_authorization_sha256"] = combination_authorization_sha
    result["dataset_support_decision_sha256"] = dataset_support_decision_sha
    return result


def certificate(
    dataset: str,
    states: pd.DataFrame,
    source_path: Path,
    roles: Dict[str, dict],
    evidence: dict,
    status: str = "qualified_for_final_inference",
    independent_state_certification: bool = True,
) -> dict:
    return {
        "dataset": dataset,
        "status": status,
        "passed": True,
        "independent_state_certification": independent_state_certification,
        "claim_boundary": "fixed-model asymptotic inference conditional on a partially unblinded model history; no unconditional coverage or global-significance claim",
        "source_ledger_path": str(source_path.resolve()),
        "source_ledger_sha256": sha256(source_path),
        "certified_coordinate_sha256": coordinate_sha256(states),
        "bound_artifacts": roles,
        "semantic_replay": evidence,
    }


def main() -> None:
    states_2015_source, roles_2015, evidence_2015 = validate_2015()
    states_2021_source, roles_2021, evidence_2021 = validate_2021()
    states_2016_source, roles_2016, evidence_2016 = validate_2016()
    combination_authorization_sha = sha256(DECISION_2016)
    dataset_support_decisions = {
        "2015": V4P9P7 / "inputs/frozen_v4p2_analysis_card.yaml",
        "2016": DECISION_2016,
        "2021": V4P9P5 / "derived/analysis/support_freeze_decision.json",
    }

    reviewed = pd.concat(
        [
            selected_rows(states_2015_source, "2015", (14, 135), ARCHIVED_2015, combination_authorization_sha, sha256(dataset_support_decisions["2015"]), "v4p9p7_archived_reproduced_max_lml"),
            selected_rows(states_2016_source, "2016", (30, 210), SOURCE_2016, combination_authorization_sha, sha256(dataset_support_decisions["2016"]), "v4p9p11p1_uniform_optimizer_remediation"),
            selected_rows(states_2021_source, "2021", (36, 300), SOURCE_2021, combination_authorization_sha, sha256(dataset_support_decisions["2021"]), "v4p9p5_frozen_support_repaired_observed_state"),
        ],
        ignore_index=True,
    )
    reviewed = reviewed.sort_values(["dataset", "mass_GeV"]).reset_index(drop=True)
    require(len(reviewed) == 415, "reviewed-state assembly is not 415 rows")

    cert_payloads = {
        "2015": certificate("2015", reviewed[reviewed.dataset == "2015"], ARCHIVED_2015, roles_2015, evidence_2015),
        "2016": certificate(
            "2016",
            reviewed[reviewed.dataset == "2016"],
            SOURCE_2016,
            roles_2016,
            evidence_2016,
            status="conditional_user_accepted_numerical_exception",
            independent_state_certification=False,
        ),
        "2021": certificate("2021", reviewed[reviewed.dataset == "2021"], SOURCE_2021, roles_2021, evidence_2021),
    }
    CERTIFICATIONS.mkdir(parents=True, exist_ok=True)
    for dataset, payload in cert_payloads.items():
        atomic_json(CERTIFICATIONS / f"{dataset}_gp_state_certification.json", payload)

    base_card = yaml.safe_load((V4P9P7 / "inputs/frozen_v4p2_analysis_card.yaml").read_text(encoding="utf-8"))
    base_card["data_range_2015"] = [0.014, 0.135]
    base_card["data_range_2016"] = [0.030, 0.210]
    base_card["data_range_2021"] = [0.036, 0.300]
    base_card["kernel_ls_res_upper_factor_by_dataset"]["2016"] = 12.0
    base_card["cls_mode"] = "asymptotic"
    base_card["cls_num_toys"] = 0
    base_card["make_ul_bands"] = False
    base_card["ul_bands_toys"] = 0
    base_card["do_combined_bands"] = False
    base_card["combined_bands_n_toys"] = 0
    base_card["make_eps2_bands"] = False
    base_card["do_combined"] = True
    base_card["combined_mode"] = "count_scale"
    base_card["data_visibility"] = {"2015": "observed", "2016": "observed", "2021": "observed"}
    base_card["output_dir"] = str((HERE / "derived").resolve())
    card_text = (
        "# v4.9.12 final-dataset combinations; generated only after all input certifications pass.\n"
        "# No pseudoexperiments or expected-limit bands are enabled.\n"
        + yaml.safe_dump(base_card, sort_keys=False)
    )
    atomic_text(INPUTS / "analysis_card.yaml", card_text)
    atomic_csv(INPUTS / "reviewed_gp_states.csv", reviewed)

    certifications = {}
    for dataset, source_path in {"2015": ARCHIVED_2015, "2016": SOURCE_2016, "2021": SOURCE_2021}.items():
        cert_path = (CERTIFICATIONS / f"{dataset}_gp_state_certification.json").resolve()
        certifications[dataset] = {
            "certificate_path": str(cert_path),
            "certificate_sha256": sha256(cert_path),
            "source_ledger_path": str(source_path.resolve()),
            "source_ledger_sha256": sha256(source_path),
        }
    provenance = {
        "status": "phase_c_conditional_inputs_frozen_with_numerical_exception",
        "claim_boundary": "fixed-model asymptotic inference conditional on a partially unblinded model history; full unblinding is forthcoming",
        "analysis_card_path": str((INPUTS / "analysis_card.yaml").resolve()),
        "analysis_card_sha256": sha256(INPUTS / "analysis_card.yaml"),
        "reviewed_gp_states_path": str((INPUTS / "reviewed_gp_states.csv").resolve()),
        "reviewed_gp_states_sha256": sha256(INPUTS / "reviewed_gp_states.csv"),
        "combination_authorization_path": str(DECISION_2016.resolve()),
        "combination_authorization_sha256": combination_authorization_sha,
        "dataset_support_decisions": {
            dataset: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for dataset, path in dataset_support_decisions.items()
        },
        "selected_support_2016_MeV": [30, 210],
        "selected_ls_upper_factor_2016": 12.0,
        "dataset_certifications": certifications,
        "numerical_exception_path": str(EXCEPTION_2016.resolve()),
        "numerical_exception_sha256": sha256(EXCEPTION_2016),
        "p1_combination_authorized": False,
        "independent_state_certification_2016": False,
        "included_final_datasets": ["2015 full", "2016 full", "2021 10%"],
        "excluded_comparisons": ["2021 1%", "2016 10%"],
        "result_scopes": [
            "2015",
            "2016",
            "2021",
            "2015+2016",
            "2015+2021",
            "2016+2021",
            "2015+2016+2021",
        ],
    }
    atomic_json(INPUTS / "analysis_input_provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": "phase_c_conditional_inputs_frozen_with_numerical_exception",
                "reviewed_rows": len(reviewed),
                "coordinate_sha256_by_dataset": {
                    dataset: coordinate_sha256(reviewed[reviewed.dataset == dataset])
                    for dataset in ("2015", "2016", "2021")
                },
                "analysis_card_sha256": sha256(INPUTS / "analysis_card.yaml"),
                "reviewed_gp_states_sha256": sha256(INPUTS / "reviewed_gp_states.csv"),
                "provenance_sha256": sha256(INPUTS / "analysis_input_provenance.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
