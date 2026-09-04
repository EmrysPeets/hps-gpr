#!/usr/bin/env python3
"""Fail-closed validator and sole release-complete attestation for v4.9.12."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME_CAMPAIGN = REPO / "study_results/v4p9p7_2016_support_combined_100toy_20260902"
sys.path.insert(0, str(RUNTIME_CAMPAIGN))
from runtime_guard import activate_and_verify, assert_import_origins  # noqa: E402


RUNTIME_PROVENANCE = activate_and_verify()
sys.path.insert(0, str(HERE))
from assemble_release_inputs import (  # noqa: E402
    validate_2015 as replay_2015_certification,
    validate_2016 as replay_2016_certification,
    validate_2021 as replay_2021_certification,
)
from runtime.bounded_tildeq_cls import bounded_tildeq_asymptotic_tails  # noqa: E402


RUNTIME_IMPORT_ORIGINS = assert_import_origins(("hps_gpr", "hps_gpr.statistics"))
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
QA = HERE / "qa"
ATTESTATION = QA / "release_attestation.json"
MANIFEST = HERE / "MANIFEST.sha256"

CURVES = DERIVED / "final_dataset_result_curves.csv"
PREDICTIONS = DERIVED / "prediction_state_ledger.csv"
MINIMA = DERIVED / "local_p0_minima.csv"
RUN_SUMMARY = DERIVED / "run_summary.json"
EXTRACTION = DERIVED / "all_three_peak_extraction_table.csv"
EXTRACTION_PLOT = DERIVED / "all_three_peak_extraction_plot_data.csv"
EXTRACTION_SUMMARY = DERIVED / "all_three_peak_extraction_summary.json"
AUDIT_CSV = QA / "numerical_conditioning_impact.csv"
AUDIT_JSON = QA / "numerical_conditioning_impact.json"

SCOPE_GRIDS: Dict[str, Tuple[int, int]] = {
    "individual_2015_full": (19, 90),
    "individual_2016_full": (39, 180),
    "individual_2021_10pct": (50, 250),
    "pair_2015_2016": (39, 90),
    "pair_2015_2021": (50, 90),
    "pair_2016_2021": (50, 180),
    "all_2015_2016_2021": (50, 90),
}
SCOPE_META = {
    "individual_2015_full": ("2015", "2015 full"),
    "individual_2016_full": ("2016", "2016 full"),
    "individual_2021_10pct": ("2021", "2021 10%"),
    "pair_2015_2016": ("2015+2016", "2015 full + 2016 full"),
    "pair_2015_2021": ("2015+2021", "2015 full + 2021 10%"),
    "pair_2016_2021": ("2016+2021", "2016 full + 2021 10%"),
    "all_2015_2016_2021": (
        "2015+2016+2021",
        "2015 full + 2016 full + 2021 10%",
    ),
}
DATASET_GRIDS: Dict[str, Tuple[int, int]] = {
    "2015": (19, 90),
    "2016": (39, 180),
    "2021": (50, 250),
}
AUDIT_COORDINATES = {
    ("individual_2015_full", 19),
    ("individual_2015_full", 50),
    ("individual_2015_full", 90),
    ("individual_2016_full", 39),
    ("individual_2016_full", 65),
    ("individual_2016_full", 102),
    ("individual_2016_full", 180),
    ("individual_2021_10pct", 50),
    ("individual_2021_10pct", 78),
    ("individual_2021_10pct", 150),
    ("individual_2021_10pct", 250),
    *((scope, mass) for mass in (50, 65, 90) for scope in (
        "pair_2015_2016",
        "pair_2015_2021",
        "pair_2016_2021",
        "all_2015_2016_2021",
    )),
}
FIGURE_STEMS = (
    "individual_final_results",
    "combined_final_results",
    "final_asymptotic_pvalues",
    "all_three_peak_extraction",
)
FROZEN_HISTOGRAMS = {
    "2015": (
        "58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f",
        "invariant_mass",
    ),
    "2016": (
        "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301",
        "h_Minv_General_Final_1",
    ),
    "2021": (
        "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4",
        "preselection/h_invM_8000",
    ),
}
HEX64 = re.compile(r"[0-9a-f]{64}")
CERTIFICATE_ROLES = {
    "2015": {
        "archived_state_ledger",
        "selected_source_attempt_1",
        "selected_source_attempt_2",
        "selected_source_attempt_3",
    },
    "2016": {
        "v4p9p11_study_protocol",
        "v4p9p11_study_spec",
        "v4p9p11_frozen_protocol",
        "v4p9p11_canonical_control_freeze",
        "v4p9p11_canonical_control_script",
        "v4p9p11_canonical_control_decision",
        "v4p9p11_control_attempt_ledger",
        "v4p9p11_control_cell_ledger",
        "v4p9p11_code_split_amendment",
        "v4p9p11_downstream_freeze",
        "v4p9p11_downstream_script",
        "v4p9p11_archive_freeze",
        "v4p9p11_archive_decision",
        "v4p9p11_archive_state_certificates",
        "v4p9p11_robust_attempt_ledger",
        "v4p9p11_robust_selected_state_ledger",
        "v4p9p11_terminal_freeze",
        "v4p9p11_terminal_decision",
        "v4p9p11_terminal_validation",
        "v4p9p11_final_validation_freeze",
        "v4p9p11_release_validator",
        "p1_study_protocol",
        "p1_study_spec",
        "p1_frozen_protocol",
        "p1_execution_freeze",
        "p1_runner",
        "p1_preflight",
        "p1_optimizer_path_ledger",
        "p1_state_ledger",
        "p1_final_support_decision",
        "p1_release_validator",
        "p1_release_validation",
        "p1_terminal_status",
        "p1_terminal_ledger",
        "downstream_numerical_exception",
    },
    "2021": {
        "study_protocol",
        "observed_card",
        "support_freeze_decision",
        "observed_state_ledger",
        "primary_result_ledger",
        "repaired_full_result_ledger",
        "repair_script",
        "optimizer_repair_ledger",
        "optimizer_repair_summary",
        "release_validation",
        "unchanged_repeat_m094_1",
        "unchanged_repeat_m094_2",
        "unchanged_repeat_m094_3",
        "unchanged_repeat_m152_1",
        "unchanged_repeat_m152_2",
        "unchanged_repeat_m152_3",
        "unchanged_repeat_m212_1",
        "unchanged_repeat_m212_2",
        "unchanged_repeat_m212_3",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, *, rtol=2.0e-10, atol=1.0e-15) -> bool:
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol))


def require_hash(value: object, label: str) -> None:
    require(bool(HEX64.fullmatch(str(value))), f"malformed SHA-256 for {label}")


def coordinate_sha256(frame: pd.DataFrame) -> str:
    columns = ["mass_GeV", "const_opt", "ls_opt", "lml"]
    payload = [
        {key: float(value) for key, value in row.items()}
        for row in frame.sort_values("mass_GeV")[columns]
        .astype(float)
        .to_dict(orient="records")
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_frozen_protocols() -> Dict[str, str]:
    ledger = HERE / "FROZEN_STATISTICAL_PROTOCOL_SHA256"
    expected: Dict[str, str] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    require(
        set(expected)
        == {"STATISTICAL_PROTOCOL.md", "NUMERICAL_CONDITIONING_AUDIT_PROTOCOL.md"},
        "frozen protocol ledger is not exact",
    )
    for name, digest in expected.items():
        require_hash(digest, name)
        require(sha256(HERE / name) == digest, f"frozen protocol drift: {name}")
    return expected


def validate_input_certifications() -> Dict[str, str]:
    provenance_path = HERE / "inputs/analysis_input_provenance.json"
    states_path = HERE / "inputs/reviewed_gp_states.csv"
    card_path = HERE / "inputs/analysis_card.yaml"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    states = pd.read_csv(states_path)
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    require(
        provenance.get("status")
        == "phase_c_conditional_inputs_frozen_with_numerical_exception",
        "input freeze status drift",
    )
    require(provenance.get("analysis_card_sha256") == sha256(card_path), "input card hash drift")
    require(provenance.get("reviewed_gp_states_sha256") == sha256(states_path), "input state hash drift")
    require(
        [int(round(1000.0 * float(value))) for value in card["data_range_2016"]]
        == [30, 210]
        and float(card["kernel_ls_res_upper_factor_by_dataset"]["2016"])
        == 12.0,
        "analysis card does not encode the exact 2016 reference support",
    )
    require(
        card.get("cls_mode") == "asymptotic"
        and int(card.get("cls_num_toys", -1)) == 0
        and card.get("make_ul_bands") is False
        and int(card.get("ul_bands_toys", -1)) == 0
        and card.get("do_combined_bands") is False
        and int(card.get("combined_bands_n_toys", -1)) == 0
        and card.get("make_eps2_bands") is False,
        "toys or expected-limit bands are enabled in the final card",
    )
    decision_path = Path(str(provenance["combination_authorization_path"]))
    require(
        decision_path.is_file()
        and provenance["combination_authorization_sha256"] == sha256(decision_path),
        "2016 support decision hash drift",
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    require(
        decision.get("status") == "all_142_states_certified"
        and decision.get("combination_authorized") is True
        and int(decision.get("state_rows", -1)) == 142
        and int(decision.get("resolved_rows", -1)) == 142
        and decision.get("unresolved_masses_MeV") == []
        and int(decision.get("support_lower_MeV", -1)) == 30
        and int(decision.get("support_upper_MeV", -1)) == 210
        and float(decision.get("upper_length_factor_2016", float("nan"))) == 12.0
        and provenance.get("selected_support_2016_MeV") == [30, 210]
        and float(provenance.get("selected_ls_upper_factor_2016", float("nan"))) == 12.0,
        "2016 reference support decision semantics drift",
    )
    exception_path = Path(str(provenance["numerical_exception_path"]))
    require(
        exception_path.is_file()
        and provenance.get("numerical_exception_sha256") == sha256(exception_path),
        "2016 numerical-exception hash drift",
    )
    exception = json.loads(exception_path.read_text(encoding="utf-8"))
    require(
        exception.get("status") == "conditional_user_accepted_numerical_exception"
        and exception.get("p1_combination_authorized") is False
        and exception.get("independent_state_certification") is False
        and provenance.get("p1_combination_authorized") is False
        and provenance.get("independent_state_certification_2016") is False,
        "2016 numerical-exception semantics drift",
    )
    certifications = dict(provenance.get("dataset_certifications", {}))
    require(set(certifications) == {"2015", "2016", "2021"}, "dataset certification set drift")
    support_decisions = dict(provenance.get("dataset_support_decisions", {}))
    require(set(support_decisions) == {"2015", "2016", "2021"}, "dataset support-decision set drift")
    for dataset, record in support_decisions.items():
        support_path = Path(str(record["path"]))
        require(
            support_path.is_file() and record.get("sha256") == sha256(support_path),
            f"{dataset} support-decision hash drift",
        )
        selected_support_hashes = set(
            states.loc[
                states.dataset.astype(str) == dataset,
                "dataset_support_decision_sha256",
            ].astype(str)
        )
        require(
            selected_support_hashes == {record["sha256"]},
            f"{dataset} reviewed states do not bind their own support decision",
        )
    require(
        set(states.combination_authorization_sha256.astype(str))
        == {sha256(decision_path)},
        "reviewed states do not bind the exact combination authorization",
    )
    semantic_replays = {
        "2015": replay_2015_certification(),
        "2016": replay_2016_certification(),
        "2021": replay_2021_certification(),
    }
    external_hashes: Dict[str, str] = {}
    for dataset, required_roles in CERTIFICATE_ROLES.items():
        entry = dict(certifications[dataset])
        certificate_path = Path(str(entry["certificate_path"]))
        source_path = Path(str(entry["source_ledger_path"]))
        require(
            certificate_path.is_file()
            and entry["certificate_sha256"] == sha256(certificate_path),
            f"{dataset} certificate hash drift",
        )
        require(
            source_path.is_file()
            and entry["source_ledger_sha256"] == sha256(source_path),
            f"{dataset} source-ledger hash drift",
        )
        if dataset == "2016":
            require(
                int(decision["states"]["rows"]) == 142
                and decision["states"]["sha256"] == sha256(source_path),
                "2016 source ledger does not match final decision",
            )
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        expected_certificate_status = (
            "conditional_user_accepted_numerical_exception"
            if dataset == "2016"
            else "qualified_for_final_inference"
        )
        require(
            certificate.get("status") == expected_certificate_status
            and certificate.get("passed") is True
            and certificate.get("dataset") == dataset
            and certificate.get("source_ledger_sha256") == sha256(source_path),
            f"{dataset} certificate did not pass",
        )
        if dataset == "2016":
            require(
                certificate.get("independent_state_certification") is False,
                "2016 certificate obscures the failed independent replay",
            )
        replay_source, replay_bound, replay_evidence = semantic_replays[dataset]
        require(
            certificate.get("bound_artifacts") == replay_bound
            and certificate.get("semantic_replay") == replay_evidence
            and certificate.get("certified_coordinate_sha256")
            == coordinate_sha256(replay_source),
            f"{dataset} semantic certification replay drift",
        )
        selected = states[states.dataset.astype(str) == dataset].copy()
        require(
            certificate.get("certified_coordinate_sha256") == coordinate_sha256(selected),
            f"{dataset} coordinate certificate drift",
        )
        require(
            set(selected.source_ledger_path.astype(str)) == {str(source_path)}
            and set(selected.source_ledger_sha256.astype(str)) == {sha256(source_path)},
            f"{dataset} states do not bind their source ledger",
        )
        source = pd.read_csv(
            source_path,
            usecols=lambda column: column
            in {"dataset", "mass_GeV", "const_opt", "ls_opt", "lml", "interpolated"},
        )
        if "dataset" in source.columns:
            source = source[source.dataset.astype(str) == dataset].copy()
        source["mass_MeV_join"] = np.rint(1000.0 * source.mass_GeV.astype(float)).astype(int)
        selected["mass_MeV_join"] = np.rint(1000.0 * selected.mass_GeV.astype(float)).astype(int)
        joined = selected.merge(
            source[["mass_MeV_join", "mass_GeV", "const_opt", "ls_opt", "lml"]],
            on="mass_MeV_join",
            how="left",
            validate="one_to_one",
            suffixes=("_selected", "_source"),
        )
        require(len(joined) == len(selected) and not joined.mass_GeV_source.isna().any(), f"{dataset} source row join fails")
        for coordinate in ("mass_GeV", "const_opt", "ls_opt", "lml"):
            require(
                close(joined[f"{coordinate}_selected"], joined[f"{coordinate}_source"], rtol=2.0e-13, atol=2.0e-13),
                f"{dataset} source row differs: {coordinate}",
            )
        bound = dict(certificate.get("bound_artifacts", {}))
        require(set(bound) == required_roles, f"{dataset} certification artifact roles drift")
        for role, record in bound.items():
            artifact = Path(str(record["path"]))
            digest = str(record["sha256"])
            require(artifact.is_file() and sha256(artifact) == digest, f"{dataset} bound artifact drift: {role}")
            external_hashes[str(artifact)] = digest
        external_hashes[str(certificate_path)] = sha256(certificate_path)
        external_hashes[str(source_path)] = sha256(source_path)
    return external_hashes


def validate_grids(curves: pd.DataFrame, predictions: pd.DataFrame) -> None:
    require(len(curves) == 680, "result ledger does not have 680 rows")
    require(len(predictions) == 415, "prediction ledger does not have 415 rows")
    require(
        not curves.duplicated(["scope_key", "mass_MeV"]).any(),
        "duplicate result coordinate",
    )
    require(
        not predictions.duplicated(["dataset", "mass_MeV"]).any(),
        "duplicate prediction coordinate",
    )
    require(set(curves.scope_key) == set(SCOPE_GRIDS), "result scopes are not exact")
    for scope, (low, high) in SCOPE_GRIDS.items():
        here = curves.loc[curves.scope_key == scope]
        found = np.sort(here.mass_MeV.to_numpy(int))
        require(np.array_equal(found, np.arange(low, high + 1)), f"bad grid: {scope}")
        dataset_set, label = SCOPE_META[scope]
        require(set(here.dataset_set.astype(str)) == {dataset_set}, f"bad dataset mapping: {scope}")
        require(set(here.scope_label.astype(str)) == {label}, f"bad scope label: {scope}")
        expected_keys = set(dataset_set.split("+"))
        for encoded in here.gp_state_sha256_by_dataset:
            require(set(json.loads(encoded)) == expected_keys, f"bad GP-state mapping: {scope}")
        for encoded in here.covariance_conditioning_by_dataset:
            require(set(json.loads(encoded)) == expected_keys, f"bad conditioning mapping: {scope}")
    require(set(predictions.dataset.astype(str)) == set(DATASET_GRIDS), "dataset set is not exact")
    for dataset, (low, high) in DATASET_GRIDS.items():
        found = np.sort(
            predictions.loc[predictions.dataset.astype(str) == dataset, "mass_MeV"].to_numpy(int)
        )
        require(np.array_equal(found, np.arange(low, high + 1)), f"bad prediction grid: {dataset}")


def validate_profiles(curves: pd.DataFrame) -> None:
    require(
        close(curves.p0_local_asymptotic, norm.sf(curves.Z_local_asymptotic), atol=1.0e-300),
        "p0/Z mapping fails",
    )
    require(
        close(curves.q0_local_asymptotic, curves.Z_local_asymptotic**2, rtol=2.0e-9),
        "q0/Z mapping fails",
    )
    require(close(curves.cls_at_limit, 0.1, rtol=0.0, atol=2.0e-6), "CLs roots fail")
    require((curves.cls_bracket_low_value > 0.1).all(), "CLs low endpoint is not above alpha")
    require((curves.cls_bracket_high_value <= 0.1).all(), "CLs high endpoint is not at/below alpha")
    require((curves.cls_bracket_low_eps2 < curves.cls_bracket_high_eps2).all(), "CLs bracket collapsed")
    require(
        (
            (curves.eps2_90 >= curves.cls_bracket_low_eps2)
            & (curves.eps2_90 <= curves.cls_bracket_high_eps2)
        ).all(),
        "CLs solution lies outside its saved bracket",
    )
    require(set(curves.limit_solver) == {"v4p9p12_cached_piecewise_bounded_tildeq_v3"}, "solver label drift")
    require(set(curves.combined_mode) == {"count_scale"}, "combined-mode drift")
    require(curves.limit_profile_optimizer_ok.astype(bool).all(), "limit optimizer failure")
    require((curves.effective_v_min_eigenvalue_relative > 0.0).all(), "combined V is not SPD")
    for column in (
        "conditioned_combined_covariance_sha256",
        "core_effective_combined_covariance_sha256",
        "effective_combined_v_sha256",
    ):
        require(curves[column].astype(str).str.fullmatch(HEX64).all(), f"bad {column}")
    require(
        close(
            curves.A90_full_template_events,
            curves.eps2_90 * curves.signal_yield_per_eps2_total,
            rtol=2.0e-12,
        ),
        "full-template yield coordinate fails",
    )
    require(
        close(
            curves.A90_fitted_window_events,
            curves.eps2_90 * curves.signal_yield_per_eps2_fitted_window,
            rtol=2.0e-12,
        ),
        "fitted-window yield coordinate fails",
    )
    require(close(curves.A90_events, curves.A90_full_template_events, rtol=0.0), "A90 alias drift")

    for row in curves.itertuples(index=False):
        tails = bounded_tildeq_asymptotic_tails(
            float(row.qmu_obs_at_limit), float(row.qmu_asimov_b_at_limit)
        )
        checks = {
            "cls": (tails.cls, row.cls_at_limit),
            "cl_sb": (tails.cl_sb, row.cl_sb_at_limit),
            "cl_b": (tails.cl_b, row.cl_b_at_limit),
            "log_cls": (tails.log_cls, row.log_cls_at_limit),
            "log_cl_sb": (tails.log_cl_sb, row.log_cl_sb_at_limit),
            "log_cl_b": (tails.log_cl_b, row.log_cl_b_at_limit),
            "z_sb": (tails.z_sb, row.z_sb_at_limit),
            "z_b": (tails.z_b, row.z_b_at_limit),
        }
        require(str(tails.branch) == str(row.tail_branch_at_limit), "tail branch replay fails")
        for label, (actual, expected) in checks.items():
            require(math.isclose(float(actual), float(expected), rel_tol=2.0e-12, abs_tol=1.0e-14), f"tail replay fails: {label}")
        p0_meta = json.loads(row.p0_profile_status)
        require(p0_meta.get("ok") and p0_meta.get("ok_alt") and p0_meta.get("ok_null"), "p0 optimizer evidence fails")
        require(
            float(p0_meta["nll_alt"]) <= float(p0_meta["nll0"]) + float(p0_meta["nll_nesting_tolerance"]),
            "p0 likelihood nesting fails",
        )
        limit_meta = json.loads(row.limit_profile_status)
        successes = []
        for side in ("observed", "asimov"):
            for item in limit_meta[side]["base"].values():
                successes.append(bool(item.get("success")))
            successes.append(bool(limit_meta[side]["fixed"].get("success")))
        require(all(successes), "limit profile-status evidence fails")


def validate_conditioning(predictions: pd.DataFrame, curves: pd.DataFrame) -> None:
    require(
        predictions.lml_delta.astype(float).abs().max() <= 5.0e-5,
        "fixed-coordinate GP LML replay exceeds the downstream frozen tolerance",
    )
    loads = predictions.selected_diagonal_load_relative.to_numpy(float)
    require(np.isfinite(loads).all() and np.all(loads < 1.0e-4), "forbidden covariance load")
    require((predictions.effective_v_min_eigenvalue_relative > 0.0).all(), "prediction V is not SPD")
    require(not predictions.eigen_clipping_used.astype(bool).any(), "eigenvalue clipping leaked in")
    for column in (
        "raw_covariance_sha256",
        "conditioned_covariance_sha256",
        "core_effective_covariance_sha256",
        "effective_v_sha256",
        "prediction_state_sha256",
    ):
        require(predictions[column].astype(str).str.fullmatch(HEX64).all(), f"bad {column}")
    for encoded in curves.covariance_conditioning_by_dataset:
        records = json.loads(encoded)
        for record in records.values():
            require(float(record["selected_diagonal_load_relative"]) < 1.0e-4, "embedded load reaches cap")
            require(not bool(record["eigen_clipping_used"]), "embedded eigen clipping")
            require_hash(record["effective_v_sha256"], "embedded effective V")


def validate_audit(protocol_hashes: Dict[str, str], predictions: pd.DataFrame) -> None:
    audit = pd.read_csv(AUDIT_CSV)
    report = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    coordinates = set(zip(audit.scope_key.astype(str), audit.mass_MeV.astype(int)))
    require(coordinates == AUDIT_COORDINATES and len(audit) == 23, "conditioning audit coordinates drift")
    require(audit.passed.astype(bool).all(), "one or more conditioning audit rows failed")
    require(
        (audit.relative_limit_difference <= 5.0e-4).all()
        and (audit.absolute_Z_difference <= 5.0e-3).all(),
        "conditioning audit exceeds a frozen tolerance",
    )
    require(report.get("status") == "audit_passed" and report.get("passed") is True, "conditioning audit not passed")
    require(report.get("audit_rows") == 23 and report.get("expected_audit_rows") == 23, "conditioning audit row count drift")
    require(report["tolerances"] == {
        "relative_limit_difference": 5.0e-4,
        "absolute_Z_difference": 5.0e-3,
        "full_grid_load_must_be_strictly_below": 1.0e-4,
    }, "conditioning audit tolerances drift")
    require(
        report["inputs"]["audit_protocol_sha256"] == protocol_hashes["NUMERICAL_CONDITIONING_AUDIT_PROTOCOL.md"],
        "conditioning audit protocol hash drift",
    )
    require(
        report["inputs"]["audit_script_sha256"]
        == sha256(HERE / "audit_conditioning_impact.py"),
        "conditioning audit script hash drift",
    )
    require(report["inputs"]["audit_csv_sha256"] == sha256(AUDIT_CSV), "conditioning audit CSV hash drift")
    require(report["inputs"]["prediction_ledger_sha256"] == sha256(PREDICTIONS), "conditioning audit prediction hash drift")
    require(
        math.isclose(float(report["full_grid_maximum_selected_diagonal_load_relative"]), float(predictions.selected_diagonal_load_relative.max()), rel_tol=0.0, abs_tol=0.0),
        "conditioning full-grid load summary drift",
    )


def validate_minima_and_run_summary(
    protocol_hashes: Dict[str, str], curves: pd.DataFrame
) -> None:
    minima = pd.read_csv(MINIMA).sort_values("scope_key").reset_index(drop=True)
    recomputed = (
        curves.loc[curves.groupby("scope_key").p0_local_asymptotic.idxmin()]
        .sort_values("scope_key")
        .reset_index(drop=True)
    )
    require(len(minima) == 7 and set(minima.scope_key) == set(SCOPE_GRIDS), "minima ledger is not exact")
    for column in (
        "scope_key",
        "mass_MeV",
        "p0_local_asymptotic",
        "Z_local_asymptotic",
        "eps2_hat_bounded_for_p0",
        "sigma_eps2_hat_bounded_for_p0",
        "eps2_90",
    ):
        if column == "scope_key":
            require(minima[column].astype(str).equals(recomputed[column].astype(str)), "minimum scope ordering drift")
        else:
            require(close(minima[column], recomputed[column], rtol=2.0e-13, atol=1.0e-300), f"minimum replay fails: {column}")
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    require(summary.get("status") == "computed", "runner status must remain computed")
    require(summary.get("result_rows") == 680 and summary.get("prediction_rows") == 415, "runner row counts drift")
    require(
        math.isclose(
            float(summary["maximum_abs_gp_lml_replay_difference"]),
            float(pd.read_csv(PREDICTIONS).lml_delta.astype(float).abs().max()),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "run-summary GP LML replay maximum drift",
    )
    hashes = summary["input_and_code_sha256"]
    expected = {
        "analysis_card": HERE / "inputs/analysis_card.yaml",
        "reviewed_gp_states": HERE / "inputs/reviewed_gp_states.csv",
        "runner": HERE / "run_final_combinations.py",
        "cached_solver": HERE / "piecewise_cached_solver.py",
        "tail_mapper": HERE / "runtime/bounded_tildeq_cls.py",
        "statistical_protocol": HERE / "STATISTICAL_PROTOCOL.md",
        "conditioning_audit_protocol": HERE / "NUMERICAL_CONDITIONING_AUDIT_PROTOCOL.md",
        "frozen_protocol_hash_ledger": HERE / "FROZEN_STATISTICAL_PROTOCOL_SHA256",
        "analysis_input_provenance": HERE / "inputs/analysis_input_provenance.json",
    }
    for key, path in expected.items():
        require(hashes.get(key) == sha256(path), f"run-contract hash drift: {key}")
    require(hashes["statistical_protocol"] == protocol_hashes["STATISTICAL_PROTOCOL.md"], "statistical protocol not bound")
    runtime_manifest = Path(str(summary["runtime_provenance"]["runtime_manifest"]))
    require(
        runtime_manifest.is_file()
        and summary["runtime_provenance"]["runtime_manifest_sha256"] == sha256(runtime_manifest)
        and hashes["attested_runtime_manifest"] == sha256(runtime_manifest),
        "attested runtime manifest does not close",
    )
    runner_origins = dict(summary["runtime_import_origins"])
    require(
        {
            "hps_gpr",
            "hps_gpr.config",
            "hps_gpr.conversion",
            "hps_gpr.dataset",
            "hps_gpr.evaluation",
            "hps_gpr.gpr",
            "hps_gpr.io",
            "hps_gpr.statistics",
            "hps_gpr.template",
        }
        == set(runner_origins),
        "runner runtime module set drift",
    )
    runtime_root = Path(str(summary["runtime_provenance"]["runtime_root"])).resolve()
    for module, origin in runner_origins.items():
        try:
            Path(origin).resolve().relative_to(runtime_root)
        except ValueError as error:
            raise RuntimeError(f"runner imported {module} outside attested runtime") from error
    for module, origin in RUNTIME_IMPORT_ORIGINS.items():
        require(runner_origins[module] == origin, f"runtime origin drift: {module}")
    histogram_inputs = dict(summary["immutable_histogram_inputs"])
    require(set(histogram_inputs) == set(FROZEN_HISTOGRAMS), "histogram input set drift")
    for dataset, (expected_hash, expected_histogram) in FROZEN_HISTOGRAMS.items():
        record = dict(histogram_inputs[dataset])
        source = Path(str(record["path"]))
        require(
            source.is_file()
            and record["sha256"] == expected_hash
            and sha256(source) == expected_hash
            and record["histogram"] == expected_histogram,
            f"frozen histogram input drift: {dataset}",
        )
    all_three = recomputed[recomputed.scope_key == "all_2015_2016_2021"].iloc[0]
    for key in (
        "mass_MeV",
        "p0_local_asymptotic",
        "Z_local_asymptotic",
        "eps2_hat_bounded_for_p0",
        "sigma_eps2_hat_bounded_for_p0",
        "eps2_90",
    ):
        require(
            math.isclose(float(summary["all_three_minimum"][key]), float(all_three[key]), rel_tol=2.0e-13, abs_tol=1.0e-300),
            f"run-summary all-three minimum drift: {key}",
        )


def validate_extraction(curves: pd.DataFrame) -> None:
    table = pd.read_csv(EXTRACTION)
    plot = pd.read_csv(EXTRACTION_PLOT)
    summary = json.loads(EXTRACTION_SUMMARY.read_text(encoding="utf-8"))
    require(summary.get("status") == "computed", "extraction status must remain computed")
    require(len(table) == 3 and set(table.dataset.astype(str)) == {"2015", "2016", "2021"}, "extraction dataset rows drift")
    triple = curves[curves.scope_key == "all_2015_2016_2021"]
    peak = triple.loc[triple.p0_local_asymptotic.idxmin()]
    selection = summary["selection"]
    require(int(selection["mass_MeV"]) == int(peak.mass_MeV), "extraction mass is not all-three argmin")
    require(selection.get("look_elsewhere_corrected") is False, "extraction falsely claims scan correction")
    for key in ("p0_local_asymptotic", "Z_local_asymptotic", "q0_local_asymptotic"):
        require(
            math.isclose(float(selection[key]), float(peak[key]), rel_tol=2.0e-10, abs_tol=1.0e-300),
            f"extraction selection closure fails: {key}",
        )
    shared = summary["shared_fit"]
    require(
        dict(shared["fit"]).get("success") is True
        and dict(shared["null"]).get("success") is True,
        "shared extraction optimizer evidence fails",
    )
    require(float(shared["eps2_hat"]) > 0.0, "shared extraction is not a positive excess")
    require(math.isclose(float(shared["eps2_hat"]), float(peak.eps2_hat_bounded_for_p0), rel_tol=2.0e-9, abs_tol=1.0e-18), "shared extraction fit does not close")
    require(math.isclose(float(shared["sigma_eps2"]), float(peak.sigma_eps2_hat_bounded_for_p0), rel_tol=2.0e-9, abs_tol=1.0e-18), "shared extraction uncertainty does not close")
    require(
        float(shared["p0_nll_alt"]) <= float(shared["p0_nll_null"]) + float(shared["p0_nll_nesting_tolerance"]),
        "shared extraction p0 nesting fails",
    )
    expected_inputs = {
        "curves_sha256": CURVES,
        "card_sha256": HERE / "inputs/analysis_card.yaml",
        "states_sha256": HERE / "inputs/reviewed_gp_states.csv",
        "provenance_sha256": HERE / "inputs/analysis_input_provenance.json",
    }
    for key, path in expected_inputs.items():
        require(summary["inputs"].get(key) == sha256(path), f"extraction input hash drift: {key}")
    independent = dict(summary["independent_signed_diagnostics"])
    require(set(independent) == {"2015", "2016", "2021"}, "independent extraction set drift")
    for dataset, record in independent.items():
        fit = dict(record["fit"])
        null = dict(record["null"])
        require(fit.get("success") is True and null.get("success") is True, f"independent {dataset} fit failed")
        tolerance = 1.0e-6 + 1.0e-8 * max(1.0, abs(float(null["nll"]) - float(fit["nll"])))
        require(float(fit["nll"]) <= float(null["nll"]) + tolerance, f"independent {dataset} nesting fails")
        table_row = table[table.dataset.astype(str) == dataset].iloc[0]
        require(
            math.isclose(float(record["eps2_hat"]), float(table_row.independent_signed_eps2_hat), rel_tol=2.0e-13, abs_tol=1.0e-18)
            and math.isclose(float(record["sigma_eps2"]), float(table_row.independent_signed_sigma_eps2), rel_tol=2.0e-13, abs_tol=1.0e-18),
            f"independent {dataset} summary/table drift",
        )
    required = {
        "shared_fitted_window_yield",
        "shared_fitted_window_sigma",
        "shared_full_template_yield",
        "shared_full_template_sigma",
        "independent_signed_fitted_window_yield",
        "independent_signed_fitted_window_sigma",
    }
    require(required <= set(table.columns), "extraction uncertainties are incomplete")
    require(
        close(table.shared_fitted_window_yield, table.shared_eps2_hat * table.signal_yield_per_eps2_fitted_window, rtol=2.0e-12)
        and close(table.shared_fitted_window_sigma, table.shared_sigma_eps2 * table.signal_yield_per_eps2_fitted_window, rtol=2.0e-12)
        and close(table.shared_full_template_yield, table.shared_eps2_hat * table.signal_yield_per_eps2_full_template, rtol=2.0e-12)
        and close(table.shared_full_template_sigma, table.shared_sigma_eps2 * table.signal_yield_per_eps2_full_template, rtol=2.0e-12),
        "shared extraction yield decomposition fails",
    )
    require(
        close(
            table.independent_signed_full_template_yield,
            table.independent_signed_eps2_hat * table.signal_yield_per_eps2_full_template,
            rtol=2.0e-12,
        )
        and close(
            table.independent_signed_full_template_sigma,
            table.independent_signed_sigma_eps2 * table.signal_yield_per_eps2_full_template,
            rtol=2.0e-12,
        ),
        "independent extraction yield decomposition fails",
    )
    require(
        close(
            table.independent_signed_fitted_window_yield,
            table.independent_signed_eps2_hat
            * table.signal_yield_per_eps2_fitted_window,
            rtol=2.0e-12,
        )
        and close(
            table.independent_signed_fitted_window_sigma,
            table.independent_signed_sigma_eps2
            * table.signal_yield_per_eps2_fitted_window,
            rtol=2.0e-12,
        ),
        "independent window-yield decomposition fails",
    )
    require(len(plot) > 0 and set(plot.dataset.astype(str)) == {"2015", "2016", "2021"}, "extraction plot ledger is incomplete")
    require(set(plot.mass_MeV.astype(int)) == {int(peak.mass_MeV)}, "extraction plot mass drift")
    require(not plot.duplicated(["dataset", "bin_center_GeV"]).any(), "duplicate extraction plot bin")
    require(
        close(
            plot.shared_total_events,
            plot.joint_profiled_background_events + plot.shared_signal_events,
            rtol=2.0e-12,
        )
        and close(
            plot.independent_signed_total_events,
            plot.independent_profiled_background_events
            + plot.independent_signed_signal_events,
            rtol=2.0e-12,
        ),
        "extraction plot background-plus-signal identity fails",
    )
    for dataset in ("2015", "2016", "2021"):
        plot_here = plot[plot.dataset.astype(str) == dataset]
        table_here = table[table.dataset.astype(str) == dataset].iloc[0]
        require(
            math.isclose(float(plot_here.shared_signal_events.sum()), float(table_here.shared_full_template_yield), rel_tol=2.0e-10, abs_tol=1.0e-8)
            and math.isclose(float(plot_here.independent_signed_signal_events.sum()), float(table_here.independent_signed_full_template_yield), rel_tol=2.0e-10, abs_tol=1.0e-8),
            f"extraction plot/table signal sum drift: {dataset}",
        )


def validate_figures() -> None:
    manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
    require(set(manifest["figures"]) == set(FIGURE_STEMS[:3]), "figure inventory drift")
    require(manifest["source_curve_sha256"] == sha256(CURVES), "figure source hash drift")
    for stem in FIGURE_STEMS:
        pdf = FIGURES / f"{stem}.pdf"
        png = FIGURES / f"{stem}.png"
        require(pdf.stat().st_size > 10_000 and pdf.read_bytes()[:4] == b"%PDF", f"invalid PDF figure: {stem}")
        require(png.stat().st_size > 10_000 and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"invalid PNG figure: {stem}")


def manifest_paths() -> Iterable[Path]:
    fixed = [
        HERE / "STATISTICAL_PROTOCOL.md",
        HERE / "NUMERICAL_CONDITIONING_AUDIT_PROTOCOL.md",
        HERE / "FROZEN_STATISTICAL_PROTOCOL_SHA256",
        HERE / "README.md",
        HERE / "assemble_release_inputs.py",
        HERE / "run_release_pipeline.py",
        HERE / "run_final_combinations.py",
        HERE / "piecewise_cached_solver.py",
        HERE / "audit_conditioning_impact.py",
        HERE / "make_peak_extraction.py",
        HERE / "make_figures.py",
        HERE / "export_harvard_selected_results.py",
        HERE / "validate_release.py",
        HERE / "runtime/__init__.py",
        HERE / "runtime/bounded_tildeq_cls.py",
        HERE / "tests/test_bounded_tildeq_cls.py",
        HERE / "tests/test_piecewise_cached_solver.py",
        HERE / "inputs/analysis_card.yaml",
        HERE / "inputs/reviewed_gp_states.csv",
        HERE / "inputs/analysis_input_provenance.json",
        HERE / "inputs/2016_PROVISIONAL_STATE_NUMERICAL_EXCEPTION.json",
        CURVES,
        PREDICTIONS,
        MINIMA,
        RUN_SUMMARY,
        EXTRACTION,
        EXTRACTION_PLOT,
        EXTRACTION_SUMMARY,
        AUDIT_CSV,
        AUDIT_JSON,
        FIGURES / "figure_manifest.json",
    ]
    fixed.extend(FIGURES / f"{stem}{suffix}" for stem in FIGURE_STEMS for suffix in (".pdf", ".png"))
    return fixed


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    ATTESTATION.unlink(missing_ok=True)
    curves = pd.read_csv(CURVES)
    predictions = pd.read_csv(PREDICTIONS)
    protocol_hashes = validate_frozen_protocols()
    external_certification_hashes = validate_input_certifications()
    validate_grids(curves, predictions)
    validate_profiles(curves)
    validate_conditioning(predictions, curves)
    validate_audit(protocol_hashes, predictions)
    validate_minima_and_run_summary(protocol_hashes, curves)
    validate_extraction(curves)
    validate_figures()

    joined = " ".join(curves.astype(str).to_numpy().ravel()).lower()
    require("2021 1%" not in joined and "2016 10%" not in joined, "non-final comparison leaked into results")
    require(not any(token in column.lower() for column in curves.columns for token in ("expected", "toy", "band", "global")), "forbidden result field leaked in")

    paths = sorted(set(manifest_paths()), key=lambda path: str(path.relative_to(HERE)))
    for path in paths:
        require(path.is_file(), f"missing release artifact: {path}")
    artifact_hashes = {str(path.relative_to(HERE)): sha256(path) for path in paths}
    manifest_text = "".join(f"{digest}  {name}\n" for name, digest in artifact_hashes.items())
    atomic_write(MANIFEST, manifest_text)
    attestation = {
        "schema_version": 1,
        "status": "conditional_release_complete_with_numerical_exception",
        "passed": True,
        "result_rows": 680,
        "prediction_rows": 415,
        "audit_rows": 23,
        "scope_count": 7,
        "protocol_sha256": protocol_hashes,
        "manifest_sha256": sha256(MANIFEST),
        "artifact_sha256": artifact_hashes,
        "external_certification_sha256": external_certification_hashes,
        "claim_boundary": (
            "Observed fixed-mass asymptotic inference conditional on frozen GP states, "
            "a disclosed 2016 cross-process numerical reproducibility exception, "
            "and a partially unblinded model history; no toys, expected bands, "
            "look-elsewhere correction, or global significance."
        ),
    }
    atomic_write(ATTESTATION, json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(attestation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
