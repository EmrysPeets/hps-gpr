#!/usr/bin/env python3
"""Independent terminal validator for v4.9.11."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uproot


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = REPO / "study_configs/v4p9p11_2016_reference30_state_certification_20260902/study_spec.json"
PROTOCOL_SHA = "bf3253ec0fe34ed72b8569c1f99824387fcd96c46a6ce63206a04a3f470e4481"
SPEC_SHA = "4c1c8355943e29e39c0cae3cce51f6b60e9878424c3b18437f86150bc07c7d4d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def check_freeze(path: Path) -> tuple[bool, dict[str, str]]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        records[relative] = digest
    return all((HERE / relative).is_file() and sha256_file(HERE / relative) == digest for relative, digest in records.items()), records


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coeffs = [float(item) for item in spec["sigma_coeffs_2016"]]
    m0 = float(spec["sigma_tail_m0_2016"])
    if mass <= m0:
        return float(sum(c * mass**i for i, c in enumerate(coeffs)))
    sigma0 = float(sum(c * m0**i for i, c in enumerate(coeffs)))
    return sigma0 + float(spec["sigma_tail_slope_override_2016"]) * (mass - m0)


def rebinned(values: np.ndarray, edges: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    low, high = spec["support_lower_MeV"] / 1000, spec["support_upper_MeV"] / 1000
    mask = (edges[:-1] >= low - 1e-12) & (edges[1:] <= high + 1e-12)
    index = np.flatnonzero(mask)
    selected = values[index]
    factor = int(spec["rebin"])
    counts = selected.reshape(-1, factor).sum(axis=1)
    native_edges = edges[index[0]:index[-1] + 2]
    coarse_edges = native_edges[::factor]
    if coarse_edges.size != counts.size + 1:
        coarse_edges = np.append(coarse_edges, native_edges[-1])
    return 0.5 * (coarse_edges[:-1] + coarse_edges[1:]), counts


def ast_closure_hash(path: Path) -> str:
    names = {
        "sha256_file", "array_hash", "histogram_hash", "json_write", "load_spec",
        "load_histogram", "rebinned", "interval", "sigma_2016", "sigma_x",
        "length_bounds", "FitAttempt", "fit_model", "lognormal_counts",
        "covariance_metrics", "select_repeated", "run_control",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    payload = "\n".join(ast.dump(node, include_attributes=False) for node in selected)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    check("protocol_hash", sha256_file(HERE / "STUDY_PROTOCOL.md") == PROTOCOL_SHA)
    check("spec_hash", sha256_file(SPEC_PATH) == SPEC_SHA)
    for filename in (
        "FROZEN_PROTOCOL_SHA256", "FROZEN_EXECUTION_SHA256",
        "FROZEN_CONTROL_PASS_SHA256", "FROZEN_DOWNSTREAM_EXECUTION_SHA256",
        "FROZEN_ARCHIVE_CLASSIFICATION_SHA256", "TERMINAL_STATE_SHA256",
    ):
        passed, records = check_freeze(HERE / filename)
        check(f"freeze_{filename}", passed, records)

    check("control_script_hash", sha256_file(HERE / "run_control_frozen.py") == "b8a7312a86d5f25eb64ba9da46834a0abd151cad97e968d93c9b3205699219b8")
    check("downstream_script_hash", sha256_file(HERE / "run_downstream_certification.py") == "5b3b05e45962a084c6a4742812f3a6b65aaba1c902bc371dce5021fa3d573710")
    check("code_split_amendment_hash", sha256_file(HERE / "PRE_ARCHIVE_CODE_SPLIT_AMENDMENT.md") == "e25a92795654b0e5dc0cf60e07c05c62b1f2631aa4afcb25d9d5b074ecbd4ef4")
    closure_control = ast_closure_hash(HERE / "run_control_frozen.py")
    closure_downstream = ast_closure_hash(HERE / "run_downstream_certification.py")
    check("control_closure_identical_after_split", closure_control == closure_downstream == "519c27e7fd71c81b41763127a777f5fa2b337f877fd4351f67534858b5011622")
    check("monolithic_scaffold_disabled", "transient monolithic runner is disabled" in (HERE / "run_state_certification.py").read_text(encoding="utf-8"))

    control_dir = HERE / "derived/control_adequacy"
    control = json.loads((control_dir / "control_decision_initial_frozen.json").read_text(encoding="utf-8"))
    control_cells = pd.read_csv(control_dir / "selected_cells.csv")
    control_attempts = pd.read_csv(control_dir / "optimizer_attempts.csv")
    check("canonical_control_pass", control["status"] == "control_adequacy_pass" and control["technical_pass"] and control["absolute_guard_pass"] and control["forbidden_centers_zero"])
    check("canonical_control_script", control["script_sha256"] == sha256_file(HERE / "run_control_frozen.py"))
    check("control_rows_and_hashes", len(control_attempts) == 60 and len(control_cells) == 20 and control["attempts"]["sha256"] == sha256_file(control_dir / "optimizer_attempts.csv") and control["cells"]["sha256"] == sha256_file(control_dir / "selected_cells.csv"))

    root_path = REPO / spec["full_input"]["path"]
    check("full_input_file_hash", sha256_file(root_path) == spec["full_input"]["file_sha256"])
    with uproot.open(root_path) as handle:
        values, edges = handle[spec["full_input"]["histogram"]].to_numpy(flow=False)
    values, edges = np.asarray(values, float), np.asarray(edges, float)
    digest = hashlib.sha256(values.astype(np.float64).tobytes() + edges.astype(np.float64).tobytes()).hexdigest()
    check("full_histogram_hash", digest == spec["full_input"]["histogram_sha256"])
    centers, counts = rebinned(values, edges, spec)
    control_masks_ok = True
    for row in control_cells.itertuples(index=False):
        low, high = spec["low_blocks"][row.block]
        allowed = (centers >= 0.030 - 2e-13) & (centers < 0.03875 - 2e-13)
        score = allowed & (centers >= low - 2e-13) & (centers < high - 2e-13)
        train = allowed & ~score
        control_masks_ok &= (
            int(row.n_train) == int(train.sum()) and int(row.n_score) == int(score.sum())
            and row.train_centers_sha256 == array_hash(centers[train])
            and row.score_centers_sha256 == array_hash(centers[score])
            and row.train_counts_sha256 == array_hash(counts[train])
            and row.score_counts_sha256 == array_hash(counts[score])
            and int(row.n_centers_at_or_above_38p75) == 0 and int(row.n_search_centers) == 0
        )
    check("control_masks_reconstructed_zero_forbidden", control_masks_ok)

    archive_dir = HERE / "derived/archive_certification"
    archive_decision = json.loads((archive_dir / "archive_class_decision.json").read_text(encoding="utf-8"))
    archive = pd.read_csv(archive_dir / "archived_state_certificates.csv")
    check("archive_decision_hash_chain", archive_decision["archive_certificates_sha256"] == sha256_file(archive_dir / "archived_state_certificates.csv") and archive_decision["script_sha256"] == sha256_file(HERE / "run_downstream_certification.py"))
    check("archive_exact_classes", len(archive) == 142 and (archive["provenance_class"] == "raw_single_source").sum() == 139 and (archive["provenance_class"] == "repair_three_source").sum() == 3)
    repair = archive.loc[archive["provenance_class"] == "repair_three_source"]
    check("repair_grid_and_source_reproduction", set(repair["mass_MeV"].astype(int)) == {43, 125, 145} and (repair["reproducing_source_count"] == 3).all() and repair["source_reproduction_pass"].astype(bool).all())
    check("repair_class_correctly_sent_to_uniform_rerun", not archive_decision["repair_class_reuse_pass"] and archive_decision["repair_class_action"] == "robust_repeat_all" and int(repair["archive_reuse_pass"].astype(bool).sum()) == 1)
    check("archive_geometry_prediction_covariance", archive["selected_source_hash_ok"].astype(bool).all() and archive["geometry_closure_pass"].astype(bool).all() and archive["prediction_closure_pass"].astype(bool).all() and archive["coordinates_interior"].astype(bool).all() and archive["covariance_ok"].astype(bool).all())

    attempts_path = HERE / "derived/robust_repeats/optimizer_attempts.csv"
    states_path = HERE / "derived/robust_repeats/selected_states.csv"
    attempts = pd.read_csv(attempts_path)
    states = pd.read_csv(states_path)
    final_path = HERE / "derived/observed_2016_gp_states_reviewed.csv"
    final = pd.read_csv(final_path)
    decision_path = HERE / "derived/state_certification_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    expected_masses = set(range(39, 181))
    check("robust_attempt_exact_grid", len(attempts) == 426 and set(attempts["mass_GeV"].mul(1000).round().astype(int)) == expected_masses and all(set(group["seed"].astype(int)) == set(spec["optimizer_seeds"]) and len(group) == 3 for _, group in attempts.groupby("mass_GeV")))
    check("state_exact_grid", len(states) == 142 and states["mass_MeV"].astype(int).is_unique and set(states["mass_MeV"].astype(int)) == expected_masses)
    check("final_is_robust_state_ledger", sha256_file(final_path) == sha256_file(states_path) and len(final) == 142)

    branch_ok = True
    cert_ok = True
    resolved_ok = True
    for row in states.itertuples(index=False):
        group = attempts.loc[np.isclose(attempts["mass_GeV"], row.mass_GeV, rtol=0, atol=1e-12)]
        eligible = group.loc[group["finite_success"].astype(bool) & group["warning_free"].astype(bool)]
        if len(eligible):
            selected = eligible.loc[eligible["lml"].idxmax()]
            tol_lml = float(spec["lml_reproduction_abs_tolerance"])
            tol_coord = float(spec["coordinate_reproduction_rel_tolerance"])
            reproduced = eligible.loc[
                (np.abs(eligible["lml"] - selected["lml"]) <= tol_lml)
                & (np.abs(eligible["constant"] - selected["constant"]) <= tol_coord * abs(float(selected["constant"])))
                & (np.abs(eligible["length"] - selected["length"]) <= tol_coord * abs(float(selected["length"])))
            ]
            branch_ok &= int(row.selected_seed) == int(selected["seed"]) and math.isclose(float(row.lml), float(selected["lml"]), rel_tol=0, abs_tol=1e-10)
        else:
            reproduced = eligible
            branch_ok &= pd.isna(row.selected_seed) and pd.isna(row.lml)
        branch_ok &= int(row.warning_free_repeat_count) == len(eligible) and int(row.reproduced_warning_free_count) == len(reproduced)
        if len(eligible):
            expected_cert = bool(
                abs(float(row.recorded_lml_difference)) <= float(spec["fixed_lml_abs_tolerance"])
                and bool(row.prediction_closure_pass) and bool(row.coordinates_interior)
                and bool(row.covariance_ok) and float(row.gradient_infinity) < float(spec["analytic_gradient_infinity_max"])
                and bool(row.polish_success) and int(row.polish_warning_count) == 0
                and abs(float(row.polish_lml_improvement)) <= float(spec["local_polish"]["max_lml_improvement"])
                and float(row.polish_constant_relative_movement) <= float(spec["local_polish"]["max_coordinate_relative_movement"])
                and float(row.polish_length_relative_movement) <= float(spec["local_polish"]["max_coordinate_relative_movement"])
            )
        else:
            expected_cert = False
        cert_ok &= bool(row.fixed_certificate_pass) == expected_cert
        expected_resolved = bool(len(eligible) >= 2 and len(reproduced) >= 2 and expected_cert and int(row.n_train_low) > 0 and int(row.n_train_high) > 0)
        resolved_ok &= bool(row.state_resolved) == expected_resolved
    check("warning_free_max_lml_branch_reconstructed", branch_ok)
    check("fixed_certificate_boolean_reconstructed", cert_ok)
    check("state_resolution_boolean_reconstructed", resolved_ok)

    unresolved = states.loc[~states["state_resolved"].astype(bool), "mass_MeV"].astype(int).tolist()
    check("canonical_global_stop", decision["status"] == "stopped_unresolved_state" and not decision["combination_authorized"] and decision["state_rows"] == 142 and decision["resolved_rows"] == 49 and decision["unresolved_masses_MeV"] == unresolved and len(unresolved) == 93)
    check("terminal_output_hashes", decision["attempts"]["sha256"] == sha256_file(attempts_path) and decision["robust_selected"]["sha256"] == sha256_file(states_path) and decision["final_states"]["sha256"] == sha256_file(final_path))
    check("decision_provenance", decision["protocol_sha256"] == PROTOCOL_SHA and decision["spec_sha256"] == SPEC_SHA and decision["script_sha256"] == sha256_file(HERE / "run_downstream_certification.py") and decision["inference_fields_accessed"] == [])

    forbidden = {str(item).lower() for item in spec["forbidden_state_selection_metrics"]}
    computed_columns = {str(col).lower() for frame in (control_cells, control_attempts, archive, attempts, states, final) for col in frame.columns}
    check("no_forbidden_inference_columns", forbidden.isdisjoint(computed_columns), sorted(forbidden & computed_columns))
    check("no_inference_artifacts", not any((HERE / "derived").glob("*pvalue*")) and not any((HERE / "derived").glob("*limit*")) and not any((HERE / "derived").glob("*extraction*")))

    passed = sum(int(item["pass"]) for item in checks)
    report = {
        "status": "pass" if passed == len(checks) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": passed, "checks_total": len(checks), "checks": checks,
        "canonical_outcome": "terminal_global_stop_no_combination",
        "canonical_state_decision_sha256": sha256_file(decision_path),
        "validator_sha256": sha256_file(Path(__file__)),
    }
    output = HERE / "qa/final_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
