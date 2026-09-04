#!/usr/bin/env python3
"""Independent, fail-closed audit of the v4.9.7 2016 support freeze.

This file is deliberately outside the frozen production workflow.  It never
imports ``run_support_scan.py``, ``analyze_support_scan.py``, or
``confirm_support_edge.py``.  The phase-1 and confirmation stages refuse to
open run ledgers until the corresponding atomically-written decision exists.

The broad-tail ROOT fit used only as the >85 MeV part of the conditional
stress truth has ``fit_ok=false`` in its immutable metadata.  Continuing
therefore requires an explicit command-line waiver whose scope is restricted
to this source-conditioned stress truth.  The waiver does not qualify the
model as a physical background generator, coverage ensemble, expected-limit
calibration, exclusion, or significance calibration.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import uproot


AUDIT = Path(__file__).resolve().parent
STUDY = AUDIT.parent
REPO = STUDY.parents[1]
AUDITOR_PATH = Path(__file__).resolve()
SPEC_PATH = STUDY / "study_spec.json"
EXPECTED_SPEC_SHA256 = (
    "4382bfa6298cafe43d45026708017ca3e43179700f2ab5c76a557411874c8b3f"
)
SCOPE_CLARIFICATION_PATH = STUDY / "SCIENTIFIC_SCOPE_CLARIFICATION.md"
SCOPE_CLARIFICATION_SHA256 = (
    "7e90ed186396f3e209f6591ccdd28df714b642137797c07e0ed048bd02656b2c"
)

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
KEYS = ["scenario", "background_toy_index", "mass_GeV", "inj_nsigma"]
TASK_LEDGER_FILES = {
    "optimizer_attempts.csv",
    "accepted_rows.csv",
    "raw_primary_rows.csv",
    "exclusions.csv",
}
PHASE1_PRODUCTS = {
    "phase1_accepted_rows.csv",
    "phase1_cell_summary.csv",
    "phase1_support_summary.csv",
    "phase1_adjacent_paired_differences.csv",
}
COLLECTION_PRODUCTS = {
    "accepted_extraction_rows.csv",
    "raw_primary_extraction_rows.csv",
    "optimizer_attempt_ledger.csv",
    "exclusion_ledger.csv",
    "closure_summary.csv",
    "zero_signal_bias_tests.csv",
    "task_product_audit.csv",
}
CONFIRMATION_PRODUCTS = {
    "full100_accepted_rows_selected_neighbors.csv",
    "confirmation_cell_summary.csv",
    "confirmation_support_summary.csv",
    "confirmation_paired_limit_differences.csv",
}
FINITE_COLUMNS = (
    "A_hat",
    "sigma_A",
    "pull",
    "A_up_wald90",
    "eps2_up_wald90",
    "A_up_wald90_minus_injected_over_sigmaA_ref",
)
BOUND_COLUMNS = (
    "ls_at_lower",
    "ls_at_upper",
    "const_at_lower",
    "const_at_upper",
    "refit_lower_boundary",
    "refit_upper_boundary",
    "refit_constant_lower_boundary",
    "refit_constant_upper_boundary",
)
WAIVER_FLAG = "accept_broad_tail_fit_status_for_conditional_stress_truth_only"
REPOSITORY_FALLBACK_SHA256 = {
    "bands.py": "c339bd6aeb75708bc43ed9311e794553d4e26053008b1a5a953350a5ff2c7965",
    "cli.py": "641f96e1863fd868da30cddc670b3d80b07a26f2527ae4500f5168faf5a10606",
    "config.py": "ec4f50345aebbf5c062e8daaefaaeca9b0e96df12f12b2d726172979df61cf9d",
    "conversion.py": "a6c13f769257c6049b4fde7f65869c8649ce54ffb816111941403cc11be9e628",
    "dataset.py": "ab704592994ee54bf0e3cb16524e5cfb85eb00635ab887dabd79f7a618bf1ff6",
    "evaluation.py": "a1d68d8ba451ed655b9a35c1e465729630c983dae14cfad05e89010f59f2aefa",
    "extraction_display.py": "465524f846e7e757b3ee9d438742b48985cff41100956bf721bd4f3f6bdd6d9d",
    "funcform_toys.py": "319784787eaa91c92ce5d9c6c4c514316d80eb9e801b82a4c87d86110940e51e",
    "gp_toys.py": "abddad5abe2bcb2009e6418cad2e216e8f42271623c4f45d798be74bb8e8088d",
    "injection.py": "3a38378379650b73159de8b98456a2bd91e5c374794805b0be39e86557e26bf2",
    "plotting.py": "cfb5888c19b1491fb7f50558601f5242adbc7ded107cfd4a4cfed9ae0f540ae3",
    "scan.py": "01b30513cb3a5c7c9ca5e5dc16612bb60007fc95fa852069b3b64a3954d67399",
    "slurm.py": "223b6048cf38f37d2b54bec1d4de620e4b528b9762f2777d722f838463075f62",
    "statistics.py": "b8cbd484056925d64bed4d9a4ad3294fbac07d51079e5cb9ed565150b73c1ff2",
    "template.py": "20c1fbaa632d5e03fa7527d0e4ddf8dc3ba8573927a8f981936721a731440e3e",
    "toy_backgrounds.py": "0c976b1f7950e0b16b4f2bb8535c934adcd245ef78d6b83bae5fde53b2dca2d4",
    "validation.py": "d614ffb6a23049f40e266dadf5a4a6efc819d9fed749acf82b9330d9d5d9cd54",
}


class AuditError(RuntimeError):
    """Raised for any failed immutable-input or decision audit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise AuditError(f"missing JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON input is not an object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def resolve_study_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (STUDY / path).resolve()


def require_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise AuditError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def require_record_hash(
    record: Mapping[str, Any], path_key: str, hash_key: str, label: str
) -> tuple[Path, str]:
    path_value = record.get(path_key)
    hash_value = record.get(hash_key)
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise AuditError(f"missing path/hash declaration for {label}")
    path = resolve_study_path(path_value)
    return path, require_hash(path, hash_value, label)


def require_exact_keys(actual: Iterable[Any], expected: set[str], label: str) -> None:
    found = {str(value) for value in actual}
    if found != expected:
        raise AuditError(
            f"{label} inventory mismatch: expected {sorted(expected)}, "
            f"found {sorted(found)}"
        )


def read_csv_allow_empty(path: Path, columns: Iterable[str] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise AuditError(f"missing CSV input: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def boolean_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    output = []
    for column in columns:
        if column not in frame:
            raise AuditError(f"missing required Boolean column: {column}")
        values = frame[column]
        if pd.api.types.is_bool_dtype(values.dtype):
            output.append(values.to_numpy(dtype=bool))
            continue
        normalized = values.astype(str).str.strip().str.lower()
        if not normalized.isin(("true", "false")).all():
            raise AuditError(f"invalid Boolean encoding in column {column}")
        output.append((normalized == "true").to_numpy(dtype=bool))
    return np.column_stack(output)


def row_boolean(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "nan", "none", ""}:
        return False
    raise AuditError(f"invalid Boolean value for {key}: {value!r}")


def stable_seed(namespace: str, *parts: object) -> int:
    material = "|".join(
        ["20260902", str(namespace), *[str(part) for part in parts]]
    )
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:4], "little"
    )


def check_static_contract(spec: Mapping[str, Any]) -> None:
    require_hash(SPEC_PATH, EXPECTED_SPEC_SHA256, "frozen study specification")
    if spec.get("study_id") != "v4p9p7_2016_support_combined_100toy_20260902":
        raise AuditError("study_id drift")
    if spec.get("study_version") != "v4.9.7":
        raise AuditError("study_version drift")
    if tuple(float(x) for x in spec.get("masses_gev", ())) != MASSES:
        raise AuditError("mass grid drift")
    if tuple(float(x) for x in spec.get("sigma_strengths", ())) != STRENGTHS:
        raise AuditError("injection grid drift")
    if tuple(int(x) for x in spec.get("toy_indices", ())) != tuple(range(100)):
        raise AuditError("toy-index grid drift")
    card = spec.get("analysis_card", {})
    supports = tuple(
        f"{round(1000 * float(value)):03d}_210"
        for value in card.get("candidate_gp_support_low_edges_gev", ())
    )
    eligible = tuple(
        f"{round(1000 * float(value)):03d}_210"
        for value in card.get("eligible_freeze_low_edges_gev", ())
    )
    if supports != SUPPORTS or eligible != ELIGIBLE:
        raise AuditError("support grid or eligibility drift")
    expected_card = {
        "search_range_gev": [0.039, 0.180],
        "truth_support_range_gev": [0.026, 0.210],
        "gp_support_high_gev": 0.210,
        "pre_log": True,
        "alpha_model": "1/y",
        "neighborhood_rebin": 5,
        "blind_nsigma": 2.25,
        "gp_train_exclude_nsigma": 2.25,
        "kernel_ls_res_lower_factor_2016": 0.9,
        "kernel_ls_res_upper_factor_2016": 12.0,
        "n_restarts": 12,
        "signed_extraction": True,
        "upper_limit_bands": False,
    }
    for key, expected in expected_card.items():
        actual = card.get(key)
        if isinstance(expected, list):
            passed = [float(value) for value in actual or ()] == expected
        elif isinstance(expected, float):
            passed = actual is not None and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15
            )
        else:
            passed = actual == expected
        if not passed:
            raise AuditError(f"analysis-card drift for {key}")
    expected_gate = {
        "version": "v4p7p1_reference_relative_v1",
        "reference_initial_attempts": 3,
        "maximum_attempts": 5,
        "top_branch_min_replicates": 2,
        "delta_lml_per_train_max": 0.001,
        "abs_log_length_ratio_max": 0.01,
        "abs_log_constant_ratio_max": 0.05,
        "abs_log_sigma_ratio_max": 0.02,
        "exact_start_abs_log_theta_max": 1e-8,
        "bound_ratio_window": 0.02,
        "sigma_over_reference_trigger": [0.5, 2.0],
        "reference_relative_lml_per_train_trigger": 0.02,
        "reference_relative_abs_log_length_trigger": 0.05,
        "reference_relative_abs_log_constant_trigger": 0.10,
        "covariance_min_eigenvalue_relative": -0.01,
        "minimum_accepted_per_cell_for_closure_claim": 95,
        "selection_rule": (
            "maximum GP log marginal likelihood among a reproducible branch; "
            "branch choice is pull-blind and never uses fitted amplitude, recovery, "
            "epsilon-squared, p-value, or upper-limit strength"
        ),
    }
    if spec.get("optimizer_gate") != expected_gate:
        raise AuditError("optimizer-gate contract drift")
    expected_selection = {
        "absolute_upper_limit_may_rank": False,
        "confirmation_failure_action": "stop without retuning",
        "frozen_before_support_extraction": True,
        "gross_abs_mean_pull_limit": 1.25,
        "holdout_may_rank": False,
        "minimax_tie_margin": 0.10,
        "minimum_full100_accepted_per_cell": 95,
        "observed_scan_before_freeze": False,
        "phase1_min_cells_below_abs_mean_pull_0p75": 9,
        "phase1_min_zero_cells_below_abs_mean_pull_0p75": 3,
        "primary_score": (
            "minimum worst absolute mean pull over the twelve cells"
        ),
        "tie_break": "smallest tied eligible edge retains more support",
    }
    if spec.get("support_selection_protocol") != expected_selection:
        raise AuditError("support-selection contract drift")


def static_truth_audit(*, waiver: bool) -> dict[str, Any]:
    spec = load_json(SPEC_PATH)
    check_static_contract(spec)
    checked: dict[str, str] = {
        str(SPEC_PATH.relative_to(STUDY)): sha256_file(SPEC_PATH)
    }

    clarification_sha256 = require_hash(
        SCOPE_CLARIFICATION_PATH,
        SCOPE_CLARIFICATION_SHA256,
        "scientific-scope clarification",
    )
    checked[str(SCOPE_CLARIFICATION_PATH.relative_to(STUDY))] = clarification_sha256

    explicit_records = (
        (spec["frozen_protocol"], "path", "sha256", "frozen protocol"),
        (spec["source_inputs"]["shape_source"], "path", "sha256", "10pct shape source"),
        (spec["source_inputs"]["normalization_source"], "path", "sha256", "full normalization source"),
        (spec["observed_input"], "path", "sha256", "observed 2016 input"),
        (spec["truth_construction"]["builder"], "path", "sha256", "truth builder"),
        (spec["truth_construction"]["broad_tail_generator"], "path", "sha256", "broad-tail generator"),
        (spec["truth_construction"]["thresholdfit_seed"], "root", "root_sha256", "broad-tail ROOT"),
        (spec["truth_construction"]["thresholdfit_seed"], "metadata", "metadata_sha256", "broad-tail metadata"),
        (spec["fit_product"], "path", "sha256", "threshold fit summary"),
        (spec["background_toy_product"], "root", "root_sha256", "qualified toy ROOT"),
        (spec["background_toy_product"], "manifest", "manifest_sha256", "qualified toy manifest"),
        (spec["qa_products"]["truth_product_validation"], "path", "sha256", "truth QA"),
        (spec["declared_result_state"], "archived_config_path", "config_sha256", "frozen analysis card"),
    )
    for record, path_key, hash_key, label in explicit_records:
        path, digest = require_record_hash(record, path_key, hash_key, label)
        checked[str(path.relative_to(STUDY))] = digest
    require_exact_keys(
        spec.get("workflow_scripts", {}).keys(),
        {"run_support_scan", "analyze_support_scan", "confirm_support_edge"},
        "workflow-script",
    )
    for name, record in spec["workflow_scripts"].items():
        path, digest = require_record_hash(record, "path", "sha256", name)
        checked[str(path.relative_to(STUDY))] = digest
    require_exact_keys(
        spec.get("runtime_instrumentation", {}).get("modules", {}).keys(),
        {"hps_gpr/__init__.py", "hps_gpr/gpr.py", "hps_gpr/io.py"},
        "runtime overlay",
    )
    for name, record in spec["runtime_instrumentation"]["modules"].items():
        path, digest = require_record_hash(record, "archived_path", "sha256", name)
        checked[str(path.relative_to(STUDY))] = digest
    for name, expected in REPOSITORY_FALLBACK_SHA256.items():
        path = REPO / "hps_gpr" / name
        digest = require_hash(path, expected, f"repository fallback hps_gpr/{name}")
        checked[f"repository/{path.relative_to(REPO)}"] = digest

    shape_path = resolve_study_path(spec["source_inputs"]["shape_source"]["path"])
    full_path = resolve_study_path(spec["source_inputs"]["normalization_source"]["path"])
    histogram = str(spec["source_inputs"]["shape_source"]["histogram"])
    with uproot.open(shape_path) as root:
        shape, edges = root[histogram].to_numpy()
    with uproot.open(full_path) as root:
        full, full_edges = root[histogram].to_numpy()
    if not np.array_equal(edges, full_edges):
        raise AuditError("10pct/full histogram-edge mismatch")
    centers = 0.5 * (edges[:-1] + edges[1:])
    envelope = (centers >= 0.026) & (centers < 0.210)
    shape_count = int(np.rint(np.sum(shape[envelope])))
    full_count = int(np.rint(np.sum(full[envelope])))
    if (shape_count, full_count) != (7_475_607, 73_145_594):
        raise AuditError("source normalization counts drift")

    fit_path = resolve_study_path(spec["fit_product"]["path"])
    fit = load_json(fit_path)
    candidates = list(fit.get("candidate_degrees", ()))
    if [int(row.get("degree", -1)) for row in candidates] != list(range(4, 11)):
        raise AuditError("candidate-degree inventory drift")
    independently_passing = []
    for row in candidates:
        passed = bool(
            bool(row.get("optimizer_success"))
            and float(row.get("deviance_ndf", math.inf)) <= 1.5
            and float(row.get("rebin5_deviance_ndf", math.inf)) <= 2.0
            and float(row.get("max_abs_rebin5_pull", math.inf)) <= 5.0
            and not bool(row.get("at_bound"))
        )
        if passed != bool(row.get("passes_fixed_gates")):
            raise AuditError(f"stored degree-{row.get('degree')} gate flag drift")
        if passed:
            independently_passing.append(int(row["degree"]))
    selected_degree = int(fit.get("selected_degree", -1))
    if not independently_passing or selected_degree != min(independently_passing):
        raise AuditError("selected degree is not the lowest independently passing degree")
    if bool(fit.get("selection_uses_gp_or_observed_full_shape", True)):
        raise AuditError("fit summary declares GP/full-shape use")

    builder_path = resolve_study_path(spec["truth_construction"]["builder"]["path"])
    tree = ast.parse(builder_path.read_text(encoding="utf-8"), filename=str(builder_path))
    full_value_load_lines = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "full"
        and isinstance(node.ctx, ast.Load)
    )
    if full_value_load_lines != [400]:
        raise AuditError(
            "truth builder no longer limits the full-data value array to one scalar sum"
        )

    baseline_record = spec["truth_construction"]["thresholdfit_seed"]
    baseline_root = resolve_study_path(baseline_record["root"])
    baseline_meta_path = resolve_study_path(baseline_record["metadata"])
    baseline_meta = load_json(baseline_meta_path)
    primary_rows = [row for row in baseline_meta.get("fits", ()) if row.get("is_primary")]
    if len(primary_rows) != 1 or primary_rows[0].get("tag") != "fShiftSigPowTail":
        raise AuditError("broad-tail primary-function inventory drift")
    primary = primary_rows[0]
    finite_parameters = all(
        np.isfinite(float(parameter.get("value", math.nan)))
        for parameter in primary.get("parameters", ())
    )
    interior_parameters = all(
        bool(parameter.get("fixed"))
        or (
            float(parameter.get("value", math.nan))
            > float(parameter.get("min", math.nan))
            and float(parameter.get("value", math.nan))
            < float(parameter.get("max", math.nan))
        )
        for parameter in primary.get("parameters", ())
    )
    validation = primary.get("validation", {})
    with uproot.open(baseline_root) as root:
        baseline, baseline_edges = root[
            "validation/fShiftSigPowTail_expected_counts"
        ].to_numpy()
    baseline_centers = 0.5 * (baseline_edges[:-1] + baseline_edges[1:])
    tail = (baseline_centers >= 0.085) & (baseline_centers < 0.210)
    baseline_shape_checks = {
        "metadata_hash_is_spec_bound": True,
        "root_hash_is_spec_bound": True,
        "primary_validation_pass": bool(primary.get("validation", {}).get("selection_pass")),
        "finite_parameters": finite_parameters,
        "free_parameters_strictly_inside_declared_bounds": interior_parameters,
        "finite_nonnegative_expected_counts": bool(
            np.isfinite(baseline).all() and (baseline >= 0.0).all()
        ),
        "strictly_positive_85_210MeV_tail": bool(np.all(baseline[tail] > 0.0)),
        "histogram_edges_match_sources": bool(np.array_equal(baseline_edges, edges)),
        "expected_total_matches_metadata": math.isclose(
            float(np.sum(baseline)),
            float(validation.get("full_range", {}).get("fit_expected", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-5,
        ),
        "finite_pearson_chi2ndf_below_2": bool(
            np.isfinite(float(primary.get("pearson_chi2ndf", math.nan)))
            and float(primary["pearson_chi2ndf"]) < 2.0
        ),
    }
    if not all(baseline_shape_checks.values()):
        failed = sorted(name for name, passed in baseline_shape_checks.items() if not passed)
        raise AuditError("broad-tail immutable shape checks failed: " + ", ".join(failed))
    broad_tail_fit_ok = bool(primary.get("fit_ok"))
    if not broad_tail_fit_ok and not waiver:
        raise AuditError(
            "broad-tail metadata has fit_ok=false; rerun with the explicit "
            "conditional-stress-truth-only waiver only after review"
        )

    product = spec["background_toy_product"]
    manifest_path = resolve_study_path(product["manifest"])
    manifest = load_json(manifest_path)
    content = dict(manifest)
    recorded_content_hash = content.pop("manifest_content_sha256", None)
    if canonical_sha256(content) != recorded_content_hash:
        raise AuditError("toy manifest content hash mismatch")
    if recorded_content_hash != product.get("manifest_content_sha256"):
        raise AuditError("toy manifest/spec content hash mismatch")
    rows = {
        (str(row.get("scenario")), int(row.get("toy_index", -1))): row
        for row in manifest.get("toys", ())
    }
    if set(rows) != {(SCENARIO, index) for index in range(100)}:
        raise AuditError("toy manifest inventory drift")
    toy_root = resolve_study_path(product["root"])
    with uproot.open(toy_root) as root:
        mean_key = str(product["analytic_mean_keys"][SCENARIO])
        mean, mean_edges = root[mean_key].to_numpy()
        if not np.array_equal(mean_edges, edges):
            raise AuditError("qualified truth/source edge mismatch")
        if not math.isclose(float(np.sum(mean)), 73_145_594.0, rel_tol=0.0, abs_tol=1e-5):
            raise AuditError("qualified truth normalization drift")
        truth_row = manifest.get("truths", [None])[0]
        if not isinstance(truth_row, Mapping) or array_sha256(mean, "<f8") != str(
            truth_row.get("mean_sha256_float64")
        ):
            raise AuditError("qualified truth mean hash mismatch")
        for index in range(100):
            material = "|".join(
                map(str, (20260902, "2016_threshold_poisson", SCENARIO, index))
            ).encode("utf-8")
            raw_seed = hashlib.sha256(material).digest()[:16]
            seed_words = [
                int.from_bytes(raw_seed[offset : offset + 4], "little")
                for offset in range(0, 16, 4)
            ]
            row = rows[(SCENARIO, index)]
            if row.get("seed_words") != seed_words:
                raise AuditError(f"toy {index:04d} seed-word drift")
            generated = np.random.default_rng(
                np.random.SeedSequence(seed_words)
            ).poisson(mean).astype(np.int64)
            values, toy_edges = root[str(row["output_histogram"])].to_numpy()
            stored = np.rint(values).astype(np.int64)
            if (
                not np.array_equal(toy_edges, mean_edges)
                or not np.array_equal(stored, generated)
                or array_sha256(stored, "<i8") != row.get("counts_sha256")
                or int(np.sum(stored)) != int(row.get("total_count", -1))
            ):
                raise AuditError(f"toy {index:04d} seed/count reproduction failure")

    geometry = []
    native_width = float(np.median(np.diff(edges)))
    if not math.isclose(native_width, 0.00005, rel_tol=0.0, abs_tol=1e-12):
        raise AuditError("native bin width is not 0.05 MeV")
    sigma39 = 0.00038 + 0.041 * 0.039 - 0.27 * 0.039**2 + 3.49 * 0.039**3 - 11.11 * 0.039**4
    for support in SUPPORTS:
        low = int(support[:3]) / 1000.0
        support_centers = centers[(centers >= low) & (centers < 0.210)]
        if support_centers.size % 5:
            raise AuditError(f"rebin-five phase mismatch for {support}")
        coarse_centers = support_centers.reshape(-1, 5).mean(axis=1)
        train = (coarse_centers < 0.039 - 2.25 * sigma39) | (
            coarse_centers > 0.039 + 2.25 * sigma39
        )
        low_train = int(np.count_nonzero(train & (coarse_centers < 0.039)))
        geometry.append(
            {
                "support": support,
                "coarse_bins": int(coarse_centers.size),
                "coarse_bin_width_GeV": 5.0 * native_width,
                "low_side_training_bins_at_39MeV": low_train,
            }
        )
    if geometry[-1]["low_side_training_bins_at_39MeV"] != 4:
        raise AuditError("34 MeV geometry-control assertion failed")

    optimizer_and_signal_seeds: dict[int, tuple[Any, ...]] = {}
    for toy_index in range(100):
        for mass in MASSES:
            mass_key = f"{mass:.9f}"
            for attempt in range(5):
                identity = ("reference_optimizer", toy_index, mass, attempt)
                value = stable_seed(
                    "v4p9p7_2016_restart_v1",
                    SCENARIO,
                    toy_index,
                    mass_key,
                    "reference",
                    attempt,
                )
                if value in optimizer_and_signal_seeds:
                    raise AuditError(f"optimizer/signal seed collision: {identity}")
                optimizer_and_signal_seeds[value] = identity
            for strength in (2.0, 5.0):
                strength_key = f"z{strength:.1f}"
                signal_identity = ("signal", toy_index, mass, strength)
                signal_value = stable_seed(
                    "v4p9p7_2016_signal_v1",
                    SCENARIO,
                    toy_index,
                    mass_key,
                    strength_key,
                )
                if signal_value in optimizer_and_signal_seeds:
                    raise AuditError(f"optimizer/signal seed collision: {signal_identity}")
                optimizer_and_signal_seeds[signal_value] = signal_identity
                for attempt in range(5):
                    identity = (
                        "injected_optimizer",
                        toy_index,
                        mass,
                        strength,
                        attempt,
                    )
                    value = stable_seed(
                        "v4p9p7_2016_restart_v1",
                        SCENARIO,
                        toy_index,
                        mass_key,
                        strength_key,
                        attempt,
                    )
                    if value in optimizer_and_signal_seeds:
                        raise AuditError(f"optimizer/signal seed collision: {identity}")
                    optimizer_and_signal_seeds[value] = identity
    if len(optimizer_and_signal_seeds) != 6800:
        raise AuditError("optimizer/signal seed inventory drift")

    result = {
        "status": "pass",
        "stage": "static_truth",
        "independent_auditor": {
            "path": str(AUDITOR_PATH.relative_to(STUDY)),
            "sha256": sha256_file(AUDITOR_PATH),
        },
        "study_spec_sha256": sha256_file(SPEC_PATH),
        "expected_study_spec_sha256": EXPECTED_SPEC_SHA256,
        "checked_file_sha256": checked,
        "scientific_scope_clarification": {
            "path": str(SCOPE_CLARIFICATION_PATH.relative_to(STUDY)),
            "sha256": clarification_sha256,
            "expected_sha256": SCOPE_CLARIFICATION_SHA256,
            "hash_match": True,
        },
        "source_counts_26_210MeV": {
            "shape_2016_10pct": shape_count,
            "normalization_2016_full": full_count,
        },
        "full_observed_shape_use_audit": {
            "builder_full_array_load_lines": full_value_load_lines,
            "permitted_value_use": "one scalar sum over 26--210 MeV",
            "full_100pct_values_entered_truth_only_as_scalar_26_210MeV_normalization": True,
            "support_specific_full_100pct_fit_p0_or_upper_limit_used_for_ranking": False,
            "ten_pct_development_shape_entered_source_conditioned_truth": True,
            "ten_pct_statistical_independence_from_full_100pct_unproven": True,
            "ten_pct_bins_never_exceed_full_100pct_bins": bool(
                np.all(np.asarray(shape, dtype=float) <= np.asarray(full, dtype=float))
            ),
            "required_description": (
                "pre-existing 2016 10pct development sample/subset; partial observed-"
                "shape information entered the stress truth. Do not call it an "
                "independent sample without run/event-level disjointness provenance"
            ),
        },
        "degree_selection": {
            "candidate_degrees": list(range(4, 11)),
            "independently_passing_degrees": independently_passing,
            "selected_degree": selected_degree,
            "selection_verified_as_lowest_passing": True,
        },
        "toy_reproduction": {
            "base_seed": 20260902,
            "seed_namespace": "2016_threshold_poisson",
            "n_toys_reproduced_bitwise": 100,
            "paired_background_identity": "scenario and toy index only; independent of support, mass, and injection strength",
        },
        "optimizer_and_signal_seed_audit": {
            "base_seed": 20260902,
            "logical_seed_identities": 6800,
            "unique_uint32_seeds": len(optimizer_and_signal_seeds),
            "collisions": 0,
            "support_label_in_seed_identity": False,
            "interpretation": (
                "optimizer and signal RNG keys are matched across support edges; "
                "background histograms are exactly paired by toy index"
            ),
        },
        "support_geometry": geometry,
        "broad_tail": {
            "primary_function": "fShiftSigPowTail",
            "fit_ok": broad_tail_fit_ok,
            "pearson_chi2ndf": float(primary["pearson_chi2ndf"]),
            "immutable_shape_checks": baseline_shape_checks,
            "waiver_required": not broad_tail_fit_ok,
            "waiver_acknowledged": bool(waiver),
            "waiver_scope": (
                "conditional source-conditioned stress truth only; not a physical "
                "background generator, coverage ensemble, expected-limit calibration, "
                "exclusion, or significance calibration. The 10pct development shape "
                "is partial observed-shape information and is not established as "
                "statistically independent of the full sample"
            ),
        },
    }
    return result


def expected_cells() -> set[tuple[float, float]]:
    return {(mass, strength) for mass in MASSES for strength in STRENGTHS}


def branch_match(
    first: Mapping[str, Any], second: Mapping[str, Any], gate: Mapping[str, Any]
) -> bool:
    required = ("gp_lml", "gp_ls", "gp_const", "sigma_A", "n_train")
    if not all(
        np.isfinite(float(first.get(key, math.nan)))
        and np.isfinite(float(second.get(key, math.nan)))
        for key in required
    ):
        return False
    n_train = max(1.0, min(float(first["n_train"]), float(second["n_train"])))
    if abs(float(first["gp_lml"]) - float(second["gp_lml"])) / n_train > float(
        gate["delta_lml_per_train_max"]
    ):
        return False
    for key, limit_key in (
        ("gp_ls", "abs_log_length_ratio_max"),
        ("gp_const", "abs_log_constant_ratio_max"),
        ("sigma_A", "abs_log_sigma_ratio_max"),
    ):
        left, right = float(first[key]), float(second[key])
        if left <= 0.0 or right <= 0.0:
            return False
        if abs(math.log(left / right)) > float(gate[limit_key]):
            return False
    return True


def independently_select_branch(
    frame: pd.DataFrame, gate: Mapping[str, Any], *, require_replication: bool
) -> tuple[dict[str, Any] | None, int]:
    usable = [
        row
        for row in frame.to_dict(orient="records")
        if row_boolean(row, "fit_ok")
        and row_boolean(row, "covariance_valid")
        and np.isfinite(float(row.get("gp_lml", math.nan)))
        and np.isfinite(float(row.get("sigma_A", math.nan)))
        and float(row.get("sigma_A", 0.0)) > 0.0
    ]
    if not usable:
        return None, 0
    selected = max(usable, key=lambda row: float(row["gp_lml"]))
    replicates = sum(branch_match(selected, row, gate) for row in usable)
    if require_replication and replicates < int(gate["top_branch_min_replicates"]):
        return None, replicates
    return selected, replicates


def independently_refit_triggers(
    row: Mapping[str, Any], gate: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if not row_boolean(row, "fit_ok") or not row_boolean(row, "covariance_valid"):
        return ["invalid_or_nonfinite"]
    ls_value, const_value = float(row["gp_ls"]), float(row["gp_const"])
    ls_initial, const_initial = float(row["gp_ls_init"]), float(row["gp_const_init"])
    if all(
        value > 0.0
        for value in (ls_value, const_value, ls_initial, const_initial)
    ) and max(
        abs(math.log(ls_value / ls_initial)),
        abs(math.log(const_value / const_initial)),
    ) < float(gate["exact_start_abs_log_theta_max"]):
        reasons.append("exact_start_signature")
    lower, upper = float(row["gp_ls_lo"]), float(row["gp_ls_hi"])
    window = float(gate["bound_ratio_window"])
    if ls_value > 0.0 and lower > 0.0 and ls_value / lower <= 1.0 + window:
        reasons.append("near_lower_length_bound")
    if ls_value > 0.0 and upper > 0.0 and ls_value / upper >= 1.0 - window:
        reasons.append("near_upper_length_bound")
    ratio = float(row["sigma_A"]) / float(row["sigmaA_reference"])
    ratio_low, ratio_high = map(float, gate["sigma_over_reference_trigger"])
    if not np.isfinite(ratio) or ratio < ratio_low or ratio > ratio_high:
        reasons.append("sigma_reference_ratio")
    reference = {
        "lml": float(row.get("reference_gp_lml", math.nan)),
        "ls": float(row.get("reference_gp_ls", math.nan)),
        "const": float(row.get("reference_gp_const", math.nan)),
    }
    comparable = (
        np.isfinite(float(row["gp_lml"]))
        and np.isfinite(reference["lml"])
        and int(row["n_train"]) > 0
        and all(
            np.isfinite(value) and value > 0.0
            for value in (
                float(row["gp_ls"]),
                reference["ls"],
                float(row["gp_const"]),
                reference["const"],
            )
        )
    )
    if not comparable:
        reasons.append("reference_relative_nonfinite")
        return reasons
    if abs(float(row["gp_lml"]) - reference["lml"]) / float(row["n_train"]) > float(
        gate["reference_relative_lml_per_train_trigger"]
    ):
        reasons.append("reference_relative_lml")
    if abs(math.log(float(row["gp_ls"]) / reference["ls"])) > float(
        gate["reference_relative_abs_log_length_trigger"]
    ):
        reasons.append("reference_relative_length")
    if abs(math.log(float(row["gp_const"]) / reference["const"])) > float(
        gate["reference_relative_abs_log_constant_trigger"]
    ):
        reasons.append("reference_relative_constant")
    return reasons


def compare_selected_row(
    accepted: pd.DataFrame,
    selected: Mapping[str, Any] | None,
    *,
    attempts: int,
    replicates: int,
    status: str,
    trigger_reasons: Iterable[str],
    label: str,
) -> None:
    if selected is None:
        if not accepted.empty:
            raise AuditError(f"{label}: accepted row exists without an eligible branch")
        return
    if len(accepted) != 1:
        raise AuditError(f"{label}: expected exactly one accepted branch")
    row = accepted.iloc[0]
    expected_exact = {
        "attempt": int(selected["attempt"]),
        "n_attempts": int(attempts),
        "top_branch_replicates": int(replicates),
    }
    for key, expected in expected_exact.items():
        if int(row[key]) != expected:
            raise AuditError(f"{label}: accepted {key} does not match independent branch")
    for key in ("gp_lml", "gp_ls", "gp_const", "sigma_A"):
        if not math.isclose(
            float(row[key]), float(selected[key]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AuditError(f"{label}: accepted {key} does not match independent branch")
    if str(row["optimizer_gate_status"]) != status:
        raise AuditError(f"{label}: optimizer gate status drift")
    expected_reasons = ";".join(trigger_reasons)
    actual_reasons = "" if pd.isna(row["optimizer_trigger_reasons"]) else str(
        row["optimizer_trigger_reasons"]
    )
    if actual_reasons != expected_reasons:
        raise AuditError(f"{label}: optimizer trigger-reason drift")


def audit_task_branch_gate(
    attempts: pd.DataFrame,
    accepted: pd.DataFrame,
    gate: Mapping[str, Any],
    *,
    support: str,
    toy_index: int,
) -> dict[str, int]:
    required_columns = {
        "scenario",
        "background_toy_index",
        "mass_GeV",
        "inj_nsigma",
        "role",
        "attempt",
        "optimizer_seed",
        "optimizer_restarts",
    }
    missing = sorted(required_columns - set(attempts.columns))
    if missing:
        raise AuditError(
            f"{support} toy {toy_index:04d}: optimizer-attempt columns missing: "
            + ", ".join(missing)
        )
    if (
        set(attempts["scenario"].astype(str)) != {SCENARIO}
        or set(attempts["background_toy_index"].astype(int)) != {toy_index}
        or set(attempts["optimizer_restarts"].astype(int)) != {12}
        or {round(float(value), 12) for value in attempts["mass_GeV"]}
        != {round(value, 12) for value in MASSES}
        or set(attempts["role"].astype(str))
        - {"reference_bonly", "injected_refit"}
    ):
        raise AuditError(
            f"{support} toy {toy_index:04d}: optimizer-attempt identity drift"
        )
    selected_count = 0
    excluded_count = 0
    for mass in MASSES:
        mass_attempts = attempts.loc[
            np.isclose(attempts["mass_GeV"].astype(float), mass)
        ]
        reference = mass_attempts.loc[
            mass_attempts["role"].astype(str) == "reference_bonly"
        ]
        if len(reference) not in {3, 5} or set(reference["attempt"].astype(int)) != set(
            range(len(reference))
        ):
            raise AuditError(
                f"{support} toy {toy_index:04d} mass {mass}: "
                "reference-attempt inventory drift"
            )
        mass_key = f"{mass:.9f}"
        for row in reference.to_dict(orient="records"):
            expected_seed = stable_seed(
                "v4p9p7_2016_restart_v1",
                SCENARIO,
                toy_index,
                mass_key,
                "reference",
                int(row["attempt"]),
            )
            if (
                int(row["optimizer_seed"]) != expected_seed
                or not math.isclose(
                    float(row["inj_nsigma"]), 0.0, rel_tol=0.0, abs_tol=1e-15
                )
            ):
                raise AuditError(
                    f"{support} toy {toy_index:04d} mass {mass}: "
                    "reference seed/strength drift"
                )
        reference_selected, reference_replicates = independently_select_branch(
            reference, gate, require_replication=True
        )
        reference_initial_selected, _ = independently_select_branch(
            reference.loc[reference["attempt"].astype(int) < 3],
            gate,
            require_replication=True,
        )
        if (len(reference) == 3 and reference_selected is None) or (
            len(reference) == 5 and reference_initial_selected is not None
        ):
            raise AuditError(
                f"{support} toy {toy_index:04d} mass {mass}: "
                "reference retry path does not match independent gate"
            )
        reference_accepted = accepted.loc[
            np.isclose(accepted["mass_GeV"].astype(float), mass)
            & np.isclose(accepted["inj_nsigma"].astype(float), 0.0)
        ]
        reference_status = (
            "pass_replicated_initial3"
            if len(reference) == 3
            else "pass_replicated_after5"
        )
        compare_selected_row(
            reference_accepted,
            reference_selected,
            attempts=len(reference),
            replicates=reference_replicates,
            status=reference_status,
            trigger_reasons=(),
            label=f"{support} toy {toy_index:04d} mass {mass} reference",
        )
        if reference_selected is None:
            if not accepted.loc[
                np.isclose(accepted["mass_GeV"].astype(float), mass)
            ].empty:
                raise AuditError(
                    f"{support} toy {toy_index:04d} mass {mass}: "
                    "injected row survived failed reference"
                )
            if not mass_attempts.loc[
                mass_attempts["role"].astype(str) == "injected_refit"
            ].empty:
                raise AuditError(
                    f"{support} toy {toy_index:04d} mass {mass}: "
                    "injection attempts survived failed reference"
                )
            excluded_count += 3
            continue
        selected_count += 1
        for strength in (2.0, 5.0):
            injected = mass_attempts.loc[
                (mass_attempts["role"].astype(str) == "injected_refit")
                & np.isclose(mass_attempts["inj_nsigma"].astype(float), strength)
            ]
            if injected.empty or 0 not in set(injected["attempt"].astype(int)):
                raise AuditError(
                    f"{support} toy {toy_index:04d} mass {mass} z{strength}: "
                    "missing first attempt"
                )
            strength_key = f"z{strength:.1f}"
            expected_signal_seed = stable_seed(
                "v4p9p7_2016_signal_v1",
                SCENARIO,
                toy_index,
                mass_key,
                strength_key,
            )
            for row in injected.to_dict(orient="records"):
                expected_optimizer_seed = stable_seed(
                    "v4p9p7_2016_restart_v1",
                    SCENARIO,
                    toy_index,
                    mass_key,
                    strength_key,
                    int(row["attempt"]),
                )
                if (
                    int(row["optimizer_seed"]) != expected_optimizer_seed
                    or int(row.get("signal_seed", -1)) != expected_signal_seed
                    or int(row.get("reference_attempt_selected", -1))
                    != int(reference_selected["attempt"])
                    or not math.isclose(
                        float(row["inj_nsigma"]),
                        strength,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        float(row.get("strength", math.nan)),
                        strength * float(reference_selected["sigma_A"]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise AuditError(
                        f"{support} toy {toy_index:04d} mass {mass} z{strength}: "
                        "injection seed/reference/strength drift"
                    )
            first = injected.loc[injected["attempt"].astype(int) == 0].iloc[0].to_dict()
            reasons = independently_refit_triggers(first, gate)
            if reasons:
                if len(injected) not in {3, 5} or set(
                    injected["attempt"].astype(int)
                ) != set(range(len(injected))):
                    raise AuditError(
                        f"{support} toy {toy_index:04d} mass {mass} z{strength}: "
                        "repeat inventory drift"
                    )
                chosen, replicates = independently_select_branch(
                    injected, gate, require_replication=True
                )
                injected_initial_selected, _ = independently_select_branch(
                    injected.loc[injected["attempt"].astype(int) < 3],
                    gate,
                    require_replication=True,
                )
                if (len(injected) == 3 and chosen is None) or (
                    len(injected) == 5 and injected_initial_selected is not None
                ):
                    raise AuditError(
                        f"{support} toy {toy_index:04d} mass {mass} z{strength}: "
                        "injected retry path does not match independent gate"
                    )
                status = (
                    "pass_trigger_replicated_after3"
                    if len(injected) == 3
                    else "pass_trigger_replicated_after5"
                )
            else:
                if len(injected) != 1:
                    raise AuditError(
                        f"{support} toy {toy_index:04d} mass {mass} z{strength}: "
                        "untriggered repeats"
                    )
                chosen, replicates = independently_select_branch(
                    injected, gate, require_replication=False
                )
                status = "pass_single_untriggered"
            selected_row = accepted.loc[
                np.isclose(accepted["mass_GeV"].astype(float), mass)
                & np.isclose(accepted["inj_nsigma"].astype(float), strength)
            ]
            compare_selected_row(
                selected_row,
                chosen,
                attempts=len(injected),
                replicates=replicates,
                status=status,
                trigger_reasons=reasons,
                label=f"{support} toy {toy_index:04d} mass {mass} z{strength}",
            )
            if chosen is None:
                excluded_count += 1
            else:
                selected_count += 1
    return {
        "independently_verified_accepted_branches": selected_count,
        "independently_verified_excluded_states": excluded_count,
    }


def validate_task_and_read_accepted(
    spec: Mapping[str, Any], support: str, toy_index: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = (
        STUDY
        / "runs"
        / f"2016_threshold_qualified_{support}"
        / SCENARIO
        / f"toy_{toy_index:04d}"
    )
    marker_path = directory / "_SUCCESS.json"
    marker = load_json(marker_path)
    if (
        marker.get("status") != "pass"
        or marker.get("scenario") != SCENARIO
        or int(marker.get("toy_index", -1)) != toy_index
        or marker.get("study_spec_sha256") != sha256_file(SPEC_PATH)
        or marker.get("background_toy_root_sha256")
        != spec["background_toy_product"]["root_sha256"]
        or marker.get("background_toy_manifest_sha256")
        != spec["background_toy_product"]["manifest_sha256"]
    ):
        raise AuditError(f"invalid task marker for {support} toy {toy_index:04d}")
    declared = marker.get("ledger_sha256", {})
    if not isinstance(declared, Mapping):
        raise AuditError(f"missing task hash inventory for {support} toy {toy_index:04d}")
    require_exact_keys(declared.keys(), TASK_LEDGER_FILES, "task-ledger")
    for name, expected_hash in declared.items():
        require_hash(directory / name, expected_hash, f"{support} toy {toy_index:04d} {name}")
    raw = read_csv_allow_empty(directory / "raw_primary_rows.csv")
    if len(raw) != 12 or raw.duplicated(KEYS).any():
        raise AuditError(f"raw task cardinality drift for {support} toy {toy_index:04d}")
    cells = set(zip(raw["mass_GeV"].astype(float), raw["inj_nsigma"].astype(float)))
    if cells != expected_cells():
        raise AuditError(f"raw task cell drift for {support} toy {toy_index:04d}")
    accepted = read_csv_allow_empty(directory / "accepted_rows.csv", KEYS)
    if not accepted.empty:
        if accepted.duplicated(KEYS).any():
            raise AuditError(f"duplicate accepted task key for {support} toy {toy_index:04d}")
        accepted_cells = set(
            zip(accepted["mass_GeV"].astype(float), accepted["inj_nsigma"].astype(float))
        )
        if (
            not accepted_cells.issubset(expected_cells())
            or set(accepted["scenario"].astype(str)) != {SCENARIO}
            or set(accepted["gp_support_mode"].astype(str)) != {support}
            or set(accepted["background_toy_index"].astype(int)) != {toy_index}
            or np.isclose(accepted["mass_GeV"].astype(float), 0.065).any()
        ):
            raise AuditError(f"accepted task content drift for {support} toy {toy_index:04d}")
        if "accepted" not in accepted or not boolean_matrix(accepted, ("accepted",)).all():
            raise AuditError(f"unaccepted row in accepted ledger for {support} toy {toy_index:04d}")
    attempts = read_csv_allow_empty(directory / "optimizer_attempts.csv")
    if attempts.empty:
        raise AuditError(f"empty optimizer-attempt ledger for {support} toy {toy_index:04d}")
    exclusions = read_csv_allow_empty(directory / "exclusions.csv")
    if (
        int(marker.get("attempt_rows", -1)) != len(attempts)
        or int(marker.get("accepted_rows", -1)) != len(accepted)
        or int(marker.get("raw_primary_rows", -1)) != len(raw)
        or int(marker.get("excluded_rows", -1)) != len(exclusions)
    ):
        raise AuditError(f"marker row-count drift for {support} toy {toy_index:04d}")
    branch_audit = audit_task_branch_gate(
        attempts,
        accepted,
        spec["optimizer_gate"],
        support=support,
        toy_index=toy_index,
    )
    if (
        branch_audit["independently_verified_accepted_branches"] != len(accepted)
        or branch_audit["independently_verified_excluded_states"] != len(exclusions)
        or len(accepted) + len(exclusions) != 12
    ):
        raise AuditError(
            f"accepted/excluded branch partition drift for {support} toy {toy_index:04d}"
        )
    audit_row = {
        "toy_index": toy_index,
        "success_marker_sha256": sha256_file(marker_path),
        "ledger_sha256": {str(name): str(value) for name, value in sorted(declared.items())},
        **branch_audit,
    }
    return accepted, audit_row


def summarize_support(
    rows: pd.DataFrame,
    support: str,
    cohort_name: str,
    cohort_indices: set[int],
    expected_n: int,
    required_n: int,
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    subset = rows.loc[
        (rows["support"] == support)
        & rows["background_toy_index"].astype(int).isin(cohort_indices)
    ]
    cell_rows = []
    for mass in MASSES:
        for strength in STRENGTHS:
            group = subset.loc[
                np.isclose(subset["mass_GeV"].astype(float), mass)
                & np.isclose(subset["inj_nsigma"].astype(float), strength)
            ]
            pulls = pd.to_numeric(group.get("pull"), errors="coerce").to_numpy(float)
            finite_pulls = pulls[np.isfinite(pulls)]
            all_finite = bool(
                len(group)
                and finite_pulls.size == len(group)
                and all(column in group for column in FINITE_COLUMNS)
                and np.isfinite(group[list(FINITE_COLUMNS)].to_numpy(float)).all()
            )
            any_bound = bool(
                len(group) and boolean_matrix(group, BOUND_COLUMNS).any()
            )
            covariance = bool(
                len(group)
                and "covariance_valid" in group
                and boolean_matrix(group, ("covariance_valid",)).all()
            )
            cell_rows.append(
                {
                    "mass_GeV": mass,
                    "inj_nsigma": strength,
                    "n": int(finite_pulls.size),
                    "mean_pull": float(np.mean(finite_pulls)) if finite_pulls.size else math.nan,
                    "all_finite": all_finite,
                    "any_kernel_bound": any_bound,
                    "all_covariance_valid": covariance,
                }
            )
    cells = pd.DataFrame(cell_rows)
    means = cells["mean_pull"].abs()
    zero = cells.loc[cells["inj_nsigma"] == 0.0, "mean_pull"].abs()
    technical_fail_reasons: list[str] = []
    if len(cells) != 12:
        technical_fail_reasons.append("cell_inventory_not_12")
    if bool((cells["n"] < required_n).any()):
        technical_fail_reasons.append("accepted_count_below_required")
    if not bool(cells["all_finite"].all()):
        technical_fail_reasons.append("nonfinite_required_quantity")
    if not bool(cells["all_covariance_valid"].all()):
        technical_fail_reasons.append("invalid_covariance")
    if bool(cells["any_kernel_bound"].any()):
        technical_fail_reasons.append("accepted_kernel_at_bound")
    technical = bool(
        len(cells) == 12
        and (cells["n"] >= required_n).all()
        and cells["all_finite"].all()
        and cells["all_covariance_valid"].all()
        and not cells["any_kernel_bound"].any()
    )
    practical = bool(
        technical
        and int((means < 0.75).sum())
        >= int(rule["phase1_min_cells_below_abs_mean_pull_0p75"])
        and int((zero < 0.75).sum())
        >= int(rule["phase1_min_zero_cells_below_abs_mean_pull_0p75"])
        and bool((means < float(rule["gross_abs_mean_pull_limit"])).all())
    )
    practical_fail_reasons: list[str] = []
    if not technical:
        practical_fail_reasons.append("technical_gate_failed")
    if int((means < 0.75).sum()) < int(
        rule["phase1_min_cells_below_abs_mean_pull_0p75"]
    ):
        practical_fail_reasons.append("fewer_than_9_of_12_cells_below_abs_mean_pull_0p75")
    if int((zero < 0.75).sum()) < int(
        rule["phase1_min_zero_cells_below_abs_mean_pull_0p75"]
    ):
        practical_fail_reasons.append("fewer_than_3_of_4_zero_cells_below_abs_mean_pull_0p75")
    if not bool((means < float(rule["gross_abs_mean_pull_limit"])).all()):
        practical_fail_reasons.append("gross_abs_mean_pull_not_below_1p25")
    failing_cells = []
    for cell in cell_rows:
        reasons = []
        if int(cell["n"]) < required_n:
            reasons.append("accepted_count_below_required")
        if not bool(cell["all_finite"]):
            reasons.append("nonfinite_required_quantity")
        if not bool(cell["all_covariance_valid"]):
            reasons.append("invalid_covariance")
        if bool(cell["any_kernel_bound"]):
            reasons.append("accepted_kernel_at_bound")
        if not abs(float(cell["mean_pull"])) < 0.75:
            reasons.append("abs_mean_pull_not_below_0p75")
        if not abs(float(cell["mean_pull"])) < float(
            rule["gross_abs_mean_pull_limit"]
        ):
            reasons.append("abs_mean_pull_not_below_1p25")
        if reasons:
            failing_cells.append(
                {
                    "mass_GeV": float(cell["mass_GeV"]),
                    "inj_nsigma": float(cell["inj_nsigma"]),
                    "n": int(cell["n"]),
                    "mean_pull": float(cell["mean_pull"]),
                    "reasons": reasons,
                }
            )
    return {
        "support": support,
        "cohort": cohort_name,
        "expected_per_cell": expected_n,
        "minimum_required_per_cell": required_n,
        "technical_gate_pass": technical,
        "practical_acceptability_pass": practical,
        "technical_gate_fail_reasons": technical_fail_reasons,
        "practical_acceptability_fail_reasons": practical_fail_reasons,
        "failing_cells": failing_cells,
        "cells_below_abs_mean_pull_0p75": int((means < 0.75).sum()),
        "zero_signal_cells_below_abs_mean_pull_0p75": int((zero < 0.75).sum()),
        "gross_bias_guard_pass": bool(
            (means < float(rule["gross_abs_mean_pull_limit"])).all()
        ),
        "worst_abs_mean_pull": float(means.max()),
        "cell_summaries": cell_rows,
    }


def phase1_audit(*, waiver: bool) -> dict[str, Any]:
    decision_path = STUDY / "derived" / "analysis" / "phase1_selection_decision.json"
    if not decision_path.is_file():
        raise AuditError(
            "phase-1 analyzer decision is absent; refusing to inspect emerging run ledgers"
        )
    static = static_truth_audit(waiver=waiver)
    static_path = AUDIT / "static_truth_audit.json"
    stored_static = load_json(static_path)
    if (
        stored_static.get("status") != "pass"
        or canonical_sha256(stored_static) != canonical_sha256(static)
    ):
        raise AuditError(
            "stored static-truth audit is missing, stale, or differs from a fresh audit"
        )
    spec = load_json(SPEC_PATH)
    decision = load_json(decision_path)
    if (
        decision.get("study_id") != spec.get("study_id")
        or decision.get("study_spec_sha256") != sha256_file(SPEC_PATH)
        or decision.get("frozen_protocol_sha256") != spec["frozen_protocol"]["sha256"]
        or bool(decision.get("observed_scan_authorized"))
        or bool(decision.get("holdout_evaluated"))
        or bool(decision.get("absolute_upper_limit_used_for_ranking"))
    ):
        raise AuditError("phase-1 decision provenance/authorization drift")
    products = decision.get("products", {})
    if not isinstance(products, Mapping):
        raise AuditError("phase-1 product hash inventory missing")
    require_exact_keys(products.keys(), PHASE1_PRODUCTS, "phase-1 analyzer product")
    for name, record in products.items():
        if not isinstance(record, Mapping):
            raise AuditError(f"invalid phase-1 product record: {name}")
        product_path = STUDY / "derived" / "analysis" / name
        require_hash(
            product_path,
            record.get("sha256"),
            f"phase-1 analyzer product {name}",
        )
        if int(record.get("rows", -1)) != len(read_csv_allow_empty(product_path)):
            raise AuditError(f"phase-1 analyzer product row-count drift: {name}")

    all_frames = []
    lane_hashes: dict[str, str] = {}
    for support in SUPPORTS:
        lane_rows = []
        task_audit = []
        for index in range(25):
            accepted, audit_row = validate_task_and_read_accepted(spec, support, index)
            if not accepted.empty:
                lane_rows.append(accepted)
            task_audit.append(audit_row)
        lane_hashes[support] = canonical_sha256(task_audit)
        if lane_rows:
            frame = pd.concat(lane_rows, ignore_index=True, sort=False)
            frame["support"] = support
            all_frames.append(frame)
    rows = (
        pd.concat(all_frames, ignore_index=True, sort=False)
        if all_frames
        else pd.DataFrame(columns=KEYS + ["support"])
    )
    summaries = [
        summarize_support(
            rows,
            support,
            "initial_0_24",
            set(range(25)),
            25,
            25,
            spec["support_selection_protocol"],
        )
        for support in SUPPORTS
    ]
    official_accepted = read_csv_allow_empty(
        STUDY / "derived" / "analysis" / "phase1_accepted_rows.csv",
        KEYS + ["support"],
    )
    comparison_keys = ["support", *KEYS]
    independent_keys = {
        tuple(row)
        for row in rows[comparison_keys].itertuples(index=False, name=None)
    }
    official_keys = {
        tuple(row)
        for row in official_accepted[comparison_keys].itertuples(index=False, name=None)
    }
    if independent_keys != official_keys or len(rows) != len(official_accepted):
        raise AuditError("phase-1 accepted aggregate differs from hashed task ledgers")

    official_cells = read_csv_allow_empty(
        STUDY / "derived" / "analysis" / "phase1_cell_summary.csv"
    )
    if len(official_cells) != 84:
        raise AuditError("phase-1 cell-summary cardinality drift")
    for summary in summaries:
        for cell in summary["cell_summaries"]:
            match = official_cells.loc[
                (official_cells["support"].astype(str) == summary["support"])
                & np.isclose(
                    official_cells["mass_GeV"].astype(float), cell["mass_GeV"]
                )
                & np.isclose(
                    official_cells["inj_nsigma"].astype(float), cell["inj_nsigma"]
                )
            ]
            if len(match) != 1:
                raise AuditError("phase-1 official cell-summary key drift")
            official = match.iloc[0].to_dict()
            if (
                int(official["n"]) != int(cell["n"])
                or not math.isclose(
                    float(official["mean_pull"]),
                    float(cell["mean_pull"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or row_boolean(official, "all_finite") != bool(cell["all_finite"])
                or row_boolean(official, "any_kernel_bound")
                != bool(cell["any_kernel_bound"])
                or row_boolean(official, "all_covariance_valid")
                != bool(cell["all_covariance_valid"])
            ):
                raise AuditError("phase-1 official cell summary differs from recomputation")

    official_supports = read_csv_allow_empty(
        STUDY / "derived" / "analysis" / "phase1_support_summary.csv"
    )
    if len(official_supports) != 7:
        raise AuditError("phase-1 support-summary cardinality drift")
    for summary in summaries:
        match = official_supports.loc[
            official_supports["support"].astype(str) == summary["support"]
        ]
        if len(match) != 1:
            raise AuditError("phase-1 official support-summary key drift")
        official = match.iloc[0].to_dict()
        if (
            row_boolean(official, "technical_gate_pass")
            != bool(summary["technical_gate_pass"])
            or row_boolean(official, "practical_acceptability_pass")
            != bool(summary["practical_acceptability_pass"])
            or row_boolean(official, "gross_bias_guard_pass")
            != bool(summary["gross_bias_guard_pass"])
            or int(official["cells_below_abs_mean_pull_0p75"])
            != int(summary["cells_below_abs_mean_pull_0p75"])
            or int(official["zero_signal_cells_below_abs_mean_pull_0p75"])
            != int(summary["zero_signal_cells_below_abs_mean_pull_0p75"])
            or not math.isclose(
                float(official["worst_abs_mean_pull"]),
                float(summary["worst_abs_mean_pull"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AuditError("phase-1 official support summary differs from recomputation")
    qualified = [
        row
        for row in summaries
        if row["support"] in ELIGIBLE and row["practical_acceptability_pass"]
    ]
    if not qualified:
        if (
            decision.get("status") != "no_provisional_edge"
            or list(decision.get("phase2_supports", ()))
            or decision.get("provisional_support") not in (None, "")
        ):
            raise AuditError("decision selected an edge but independent audit found none")
        independently_selected = None
        tied_supports: list[str] = []
        phase2_supports: list[str] = []
        minimum = None
    else:
        minimum = min(float(row["worst_abs_mean_pull"]) for row in qualified)
        tie_margin = float(spec["support_selection_protocol"]["minimax_tie_margin"])
        tied_supports = [
            support
            for support in SUPPORTS
            if any(
                row["support"] == support
                and float(row["worst_abs_mean_pull"]) <= minimum + tie_margin
                for row in qualified
            )
        ]
        independently_selected = tied_supports[0]
        position = SUPPORTS.index(independently_selected)
        neighbor_set = {independently_selected}
        if position > 0:
            neighbor_set.add(SUPPORTS[position - 1])
        if position + 1 < len(SUPPORTS):
            neighbor_set.add(SUPPORTS[position + 1])
        phase2_supports = [support for support in SUPPORTS if support in neighbor_set]
        if (
            decision.get("status") != "provisional_edge_selected"
            or decision.get("provisional_support") != independently_selected
            or list(decision.get("tied_supports", ())) != tied_supports
            or list(decision.get("phase2_supports", ())) != phase2_supports
            or not math.isclose(
                float(decision.get("primary_minimum_worst_abs_mean_pull", math.nan)),
                minimum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AuditError("phase-1 decision does not match independent minimax/tie-break recomputation")

    return {
        "status": "pass",
        "stage": "phase1_selection",
        "independent_auditor": {
            "path": str(AUDITOR_PATH.relative_to(STUDY)),
            "sha256": sha256_file(AUDITOR_PATH),
        },
        "study_spec_sha256": sha256_file(SPEC_PATH),
        "static_truth_audit_path": str(static_path.relative_to(STUDY)),
        "static_truth_audit_sha256": sha256_file(static_path),
        "static_truth_audit_content_sha256": canonical_sha256(static),
        "phase1_decision_sha256": sha256_file(decision_path),
        "exact_candidate_supports": list(SUPPORTS),
        "phase1_lane_task_hashes": lane_hashes,
        "support_summaries": summaries,
        "independent_minimum_worst_abs_mean_pull": minimum,
        "independent_tied_supports": tied_supports,
        "independent_selected_support": independently_selected,
        "independent_phase2_supports": phase2_supports,
        "observed_scan_authorized": False,
    }


def confirmation_audit(*, waiver: bool) -> dict[str, Any]:
    freeze_path = STUDY / "derived" / "analysis" / "support_freeze_decision.json"
    if not freeze_path.is_file():
        raise AuditError(
            "support-freeze decision is absent; refusing to inspect emerging continuation ledgers"
        )
    phase1 = phase1_audit(waiver=waiver)
    phase1_path = AUDIT / "phase1_selection_audit.json"
    stored_phase1 = load_json(phase1_path)
    if (
        stored_phase1.get("status") != "pass"
        or canonical_sha256(stored_phase1) != canonical_sha256(phase1)
    ):
        raise AuditError(
            "stored phase-1 audit is missing, stale, or differs from a fresh audit"
        )
    selected = phase1.get("independent_selected_support")
    supports = list(phase1.get("independent_phase2_supports", ()))
    if not isinstance(selected, str) or not supports:
        raise AuditError("phase-1 audit did not select a support")
    spec = load_json(SPEC_PATH)
    rule = spec["support_selection_protocol"]
    all_frames = []
    collection_hashes: dict[str, Any] = {}
    for support in supports:
        directory = STUDY / "derived" / f"2016_threshold_qualified_{support}"
        summary_path = directory / "collection_summary.json"
        summary = load_json(summary_path)
        if (
            summary.get("status") != "pass"
            or summary.get("study_spec_sha256") != sha256_file(SPEC_PATH)
            or summary.get("gp_support_mode") != support
            or int(summary.get("raw_rows", -1)) != 1200
            or int(summary.get("summary_cells", -1)) != 12
        ):
            raise AuditError(f"collection-summary drift for {support}")
        product_hashes = summary.get("derived_sha256", {})
        if not isinstance(product_hashes, Mapping):
            raise AuditError(f"missing collection product hashes for {support}")
        require_exact_keys(product_hashes.keys(), COLLECTION_PRODUCTS, "collection product")
        for name, expected_hash in product_hashes.items():
            require_hash(directory / name, expected_hash, f"{support} collection {name}")

        task_frames = []
        task_audit = []
        for index in range(100):
            accepted, audit_row = validate_task_and_read_accepted(spec, support, index)
            if not accepted.empty:
                task_frames.append(accepted)
            task_audit.append(audit_row)
        frame = (
            pd.concat(task_frames, ignore_index=True, sort=False)
            if task_frames
            else pd.DataFrame(columns=KEYS)
        )
        if not frame.empty:
            frame["support"] = support
            all_frames.append(frame)
        collected = read_csv_allow_empty(directory / "accepted_extraction_rows.csv", KEYS)
        task_keys = set(tuple(row) for row in frame[KEYS].itertuples(index=False, name=None))
        collected_keys = set(
            tuple(row) for row in collected[KEYS].itertuples(index=False, name=None)
        )
        if task_keys != collected_keys or len(frame) != len(collected):
            raise AuditError(f"collected/task accepted-ledger mismatch for {support}")
        collection_hashes[support] = {
            "collection_summary_sha256": sha256_file(summary_path),
            "derived_sha256": {
                str(name): str(value) for name, value in sorted(product_hashes.items())
            },
            "all_100_task_markers_and_ledgers_sha256": canonical_sha256(task_audit),
        }

    rows = pd.concat(all_frames, ignore_index=True, sort=False)
    summaries = []
    for support in supports:
        summaries.extend(
            (
                summarize_support(rows, support, "initial_0_24", set(range(25)), 25, 25, rule),
                summarize_support(rows, support, "continuation_25_99", set(range(25, 100)), 75, 75, rule),
                summarize_support(
                    rows,
                    support,
                    "full_0_99",
                    set(range(100)),
                    100,
                    int(rule["minimum_full100_accepted_per_cell"]),
                    rule,
                ),
            )
        )
    selected_rows = {
        row["cohort"]: row for row in summaries if row["support"] == selected
    }
    independently_frozen = all(
        selected_rows[name]["practical_acceptability_pass"]
        for name in ("initial_0_24", "continuation_25_99", "full_0_99")
    )

    freeze = load_json(freeze_path)
    products = freeze.get("products", {})
    if not isinstance(products, Mapping):
        raise AuditError("confirmation product hashes missing")
    require_exact_keys(products.keys(), CONFIRMATION_PRODUCTS, "confirmation product")
    for name, record in products.items():
        if not isinstance(record, Mapping):
            raise AuditError(f"invalid confirmation product record: {name}")
        require_hash(
            STUDY / "derived" / "analysis" / name,
            record.get("sha256"),
            f"confirmation product {name}",
        )
    expected_status = "support_edge_frozen" if independently_frozen else "support_edge_confirmation_failed"
    expected_low = int(selected[:3])
    if (
        freeze.get("status") != expected_status
        or freeze.get("study_id") != spec.get("study_id")
        or freeze.get("study_spec_sha256") != sha256_file(SPEC_PATH)
        or freeze.get("phase1_decision_sha256") != phase1["phase1_decision_sha256"]
        or freeze.get("selected_support") != selected
        or int(freeze.get("selected_support_low_MeV", -1)) != expected_low
        or int(freeze.get("support_high_MeV", -1)) != 210
        or list(freeze.get("data_range_2016", ())) != [expected_low / 1000.0, 0.210]
        or list(freeze.get("phase2_supports", ())) != supports
        or bool(freeze.get("initial_gate_pass"))
        != bool(selected_rows["initial_0_24"]["practical_acceptability_pass"])
        or bool(freeze.get("continuation_gate_pass"))
        != bool(selected_rows["continuation_25_99"]["practical_acceptability_pass"])
        or bool(freeze.get("full100_gate_pass"))
        != bool(selected_rows["full_0_99"]["practical_acceptability_pass"])
        or bool(freeze.get("observed_scan_authorized")) != independently_frozen
        or bool(freeze.get("holdout_65MeV_authorized")) != independently_frozen
        or bool(freeze.get("holdout_65MeV_used_for_selection"))
        or bool(freeze.get("absolute_upper_limit_used_for_selection"))
        or bool(freeze.get("retuning_after_confirmation"))
    ):
        raise AuditError("freeze decision does not match independent confirmation recomputation")

    static_path = AUDIT / "static_truth_audit.json"
    freeze_sha256 = sha256_file(freeze_path)
    static_sha256 = sha256_file(static_path)
    phase1_sha256 = sha256_file(phase1_path)
    clarification_sha256 = require_hash(
        SCOPE_CLARIFICATION_PATH,
        SCOPE_CLARIFICATION_SHA256,
        "scientific-scope clarification",
    )
    return {
        "status": "pass",
        "stage": "confirmation",
        "independent_auditor": {
            "path": str(AUDITOR_PATH.relative_to(STUDY)),
            "sha256": sha256_file(AUDITOR_PATH),
        },
        "study_spec_sha256": sha256_file(SPEC_PATH),
        "static_truth_audit_path": str(static_path.relative_to(STUDY)),
        "static_truth_audit_sha256": static_sha256,
        "phase1_selection_audit_path": str(phase1_path.relative_to(STUDY)),
        "phase1_selection_audit_sha256": phase1_sha256,
        "phase1_selection_audit_content_sha256": canonical_sha256(phase1),
        "canonical_support_freeze_decision_path": str(freeze_path.relative_to(STUDY)),
        "canonical_support_freeze_decision_sha256": freeze_sha256,
        "scientific_scope_clarification": {
            "path": str(SCOPE_CLARIFICATION_PATH.relative_to(STUDY)),
            "sha256": clarification_sha256,
            "expected_sha256": SCOPE_CLARIFICATION_SHA256,
            "hash_match": True,
        },
        "independent_selected_support": selected,
        "exact_phase2_supports": supports,
        "collection_input_hashes": collection_hashes,
        "support_cohort_summaries": summaries,
        "independently_frozen": independently_frozen,
        "observed_scan_authorized": independently_frozen,
        "authorization": {
            "status": "authorized" if independently_frozen else "denied",
            "canonical_support_freeze_decision_sha256": freeze_sha256,
            "static_truth_audit_sha256": static_sha256,
            "phase1_selection_audit_sha256": phase1_sha256,
            "independent_auditor_sha256": sha256_file(AUDITOR_PATH),
            "scientific_scope_clarification_path": str(
                SCOPE_CLARIFICATION_PATH.relative_to(STUDY)
            ),
            "scientific_scope_clarification_sha256": clarification_sha256,
            "selected_support": selected,
            "selected_support_low_MeV": expected_low,
            "support_high_MeV": 210,
            "data_range_2016": [expected_low / 1000.0, 0.210],
            "broad_tail_waiver_scope": (
                "conditional source-conditioned stress truth only"
            ),
            "blinding_statement": (
                "not full-data blind: the pre-existing 2016 10pct development shape "
                "entered the source-conditioned truth; full-100pct values entered "
                "truth construction only through the scalar 26--210 MeV normalization"
            ),
            "support_ranking_statement": (
                "no support-specific full-100pct fit, local p0, or upper limit was used "
                "to rank support edges"
            ),
        },
        "authorization_scope": (
            "support freeze only; broad-tail waiver remains conditional-stress-truth-only"
        ),
    }


def blocked_state_audit(*, waiver: bool) -> dict[str, Any]:
    """Bind the terminal no-edge decision into a fail-closed authorization."""
    phase1 = phase1_audit(waiver=waiver)
    phase1_path = AUDIT / "phase1_selection_audit.json"
    stored_phase1 = load_json(phase1_path)
    if (
        stored_phase1.get("status") != "pass"
        or canonical_sha256(stored_phase1) != canonical_sha256(phase1)
    ):
        raise AuditError(
            "stored phase-1 audit is missing, stale, or differs from a fresh audit"
        )
    decision_path = STUDY / "derived" / "analysis" / "phase1_selection_decision.json"
    decision = load_json(decision_path)
    if (
        decision.get("status") != "no_provisional_edge"
        or phase1.get("independent_selected_support") is not None
        or list(phase1.get("independent_phase2_supports", ()))
        or bool(phase1.get("observed_scan_authorized"))
    ):
        raise AuditError("phase-1 state is not a terminal no-edge outcome")
    freeze_path = STUDY / "derived" / "analysis" / "support_freeze_decision.json"
    if freeze_path.exists():
        raise AuditError(
            "a support-freeze decision exists despite the terminal no-edge outcome"
        )
    static_path = AUDIT / "static_truth_audit.json"
    clarification_sha256 = require_hash(
        SCOPE_CLARIFICATION_PATH,
        SCOPE_CLARIFICATION_SHA256,
        "scientific-scope clarification",
    )
    compact_gates = [
        {
            "support": row["support"],
            "technical_gate_pass": row["technical_gate_pass"],
            "technical_gate_fail_reasons": row["technical_gate_fail_reasons"],
            "practical_acceptability_pass": row["practical_acceptability_pass"],
            "practical_acceptability_fail_reasons": row[
                "practical_acceptability_fail_reasons"
            ],
            "cells_below_abs_mean_pull_0p75": row[
                "cells_below_abs_mean_pull_0p75"
            ],
            "zero_signal_cells_below_abs_mean_pull_0p75": row[
                "zero_signal_cells_below_abs_mean_pull_0p75"
            ],
            "gross_bias_guard_pass": row["gross_bias_guard_pass"],
            "worst_abs_mean_pull": row["worst_abs_mean_pull"],
        }
        for row in phase1["support_summaries"]
    ]
    auditor_sha256 = sha256_file(AUDITOR_PATH)
    static_sha256 = sha256_file(static_path)
    phase1_sha256 = sha256_file(phase1_path)
    decision_sha256 = sha256_file(decision_path)
    return {
        "audit_status": "pass",
        "status": "production_blocked",
        "stage": "blocked_state",
        "study_spec_sha256": sha256_file(SPEC_PATH),
        "independent_auditor": {
            "path": str(AUDITOR_PATH.relative_to(STUDY)),
            "sha256": auditor_sha256,
        },
        "static_truth_audit_path": str(static_path.relative_to(STUDY)),
        "static_truth_audit_sha256": static_sha256,
        "phase1_selection_audit_path": str(phase1_path.relative_to(STUDY)),
        "phase1_selection_audit_sha256": phase1_sha256,
        "canonical_phase1_decision_path": str(decision_path.relative_to(STUDY)),
        "canonical_phase1_decision_sha256": decision_sha256,
        "canonical_phase1_decision_status": "no_provisional_edge",
        "canonical_support_freeze_decision_present": False,
        "scientific_scope_clarification": {
            "path": str(SCOPE_CLARIFICATION_PATH.relative_to(STUDY)),
            "sha256": clarification_sha256,
        },
        "exact_candidate_supports": list(SUPPORTS),
        "independent_selected_support": None,
        "exact_phase2_supports": [],
        "support_gate_summary": compact_gates,
        "authorization": {
            "status": "denied",
            "reason": "no eligible support passed the frozen phase-1 practical rule",
            "required_protocol_action": "stop without retuning",
            "confirmation_authorized": False,
            "observed_scan_authorized": False,
            "combined_production_authorized": False,
            "holdout_65MeV_authorized": False,
            "independent_auditor_sha256": auditor_sha256,
            "static_truth_audit_sha256": static_sha256,
            "phase1_selection_audit_sha256": phase1_sha256,
            "canonical_phase1_decision_sha256": decision_sha256,
            "scientific_scope_clarification_sha256": clarification_sha256,
        },
        "claim_boundary": (
            "terminal failure of the predeclared conditional source-conditioned "
            "pull-recovery criterion; it is not an observed exclusion, sensitivity, "
            "coverage, or significance statement"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("static", "phase1", "confirmation", "blocked")
    )
    parser.add_argument(
        "--accept-broad-tail-fit-status-for-conditional-stress-truth-only",
        action="store_true",
        dest=WAIVER_FLAG,
        help=(
            "explicitly accept the pinned broad-tail fit_ok=false only for the "
            "source-conditioned stress truth after immutable shape checks pass"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    waiver = bool(getattr(args, WAIVER_FLAG))
    outputs = {
        "static": AUDIT / "static_truth_audit.json",
        "phase1": AUDIT / "phase1_selection_audit.json",
        "confirmation": AUDIT / "confirmation_freeze_audit.json",
        "blocked": AUDIT / "production_authorization_denied.json",
    }
    try:
        if args.stage == "static":
            result = static_truth_audit(waiver=waiver)
        elif args.stage == "phase1":
            result = phase1_audit(waiver=waiver)
        elif args.stage == "confirmation":
            result = confirmation_audit(waiver=waiver)
        else:
            result = blocked_state_audit(waiver=waiver)
    except Exception as exc:
        result = {
            "status": "fail",
            "stage": args.stage,
            "study_spec_sha256": sha256_file(SPEC_PATH) if SPEC_PATH.is_file() else None,
            "error": f"{type(exc).__name__}: {exc}",
            "result_dependent_inputs_opened": False
            if args.stage in {"phase1", "confirmation"}
            and not (
                (STUDY / "derived" / "analysis" / "phase1_selection_decision.json").is_file()
                if args.stage == "phase1"
                else (STUDY / "derived" / "analysis" / "support_freeze_decision.json").is_file()
            )
            else "decision_present_before_stage",
        }
        output = args.output.resolve() if args.output else outputs[args.stage]
        if output.parent != AUDIT:
            raise AuditError("audit output must stay directly inside the audit directory")
        atomic_json(output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    output = args.output.resolve() if args.output else outputs[args.stage]
    if output.parent != AUDIT:
        raise AuditError("audit output must stay directly inside the audit directory")
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
