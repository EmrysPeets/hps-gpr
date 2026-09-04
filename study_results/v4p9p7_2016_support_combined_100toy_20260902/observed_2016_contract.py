#!/usr/bin/env python3
"""Fail-closed shared contract for the post-freeze 2016 observed workflow."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "STUDY_PROTOCOL.md"
STUDY_SPEC = HERE / "study_spec.json"
SCIENTIFIC_SCOPE_CLARIFICATION = HERE / "SCIENTIFIC_SCOPE_CLARIFICATION.md"
BASE_CARD = HERE / "inputs" / "frozen_v4p2_analysis_card.yaml"
OBSERVED_ROOT = HERE / "inputs" / "source_2016_full.root"
FREEZE = HERE / "derived" / "analysis" / "support_freeze_decision.json"
STATIC_TRUTH_AUDIT = HERE / "audit" / "static_truth_audit.json"
PHASE1_AUDIT = HERE / "audit" / "phase1_selection_audit.json"
CONFIRMATION_AUDIT = HERE / "audit" / "confirmation_freeze_audit.json"
PRODUCTION_DENIAL = HERE / "audit" / "production_authorization_denied.json"
INDEPENDENT_AUDITOR = HERE / "audit" / "independent_freeze_audit.py"
ASSEMBLER = HERE / "assemble_reviewed_state_ledger.py"
PHASE1_DECISION = HERE / "derived" / "analysis" / "phase1_selection_decision.json"
CARD = HERE / "inputs" / "v4p9p7_observed_2016_full_frozen_support_card.yaml"
CARD_MANIFEST = CARD.with_suffix(".manifest.json")
PRIMARY = HERE / "observed_scan" / "2016_full_primary"
REPEAT_ROOT = HERE / "observed_scan" / "2016_full_unchanged_card_repeats"
REVIEW_ROOT = HERE / "observed_scan" / "final_2016"
REPAIR_PLAN = REVIEW_ROOT / "optimizer_repair_plan.json"
REVIEWED_CSV = REVIEW_ROOT / "results_single_reviewed.csv"
REPAIR_LEDGER = REVIEW_ROOT / "optimizer_repair_ledger.csv"
REVIEW_SUMMARY = REVIEW_ROOT / "review_summary.json"

STUDY_ID = "v4p9p7_2016_support_combined_100toy_20260902"
EXPECTED_PROTOCOL_SHA256 = (
    "81e5954c6bb1073010f32af8ab2fccc94d922f94018abe6416238e9d92cbec02"
)
EXPECTED_STUDY_SPEC_SHA256 = (
    "4382bfa6298cafe43d45026708017ca3e43179700f2ab5c76a557411874c8b3f"
)
EXPECTED_BASE_CARD_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
EXPECTED_OBSERVED_ROOT_SHA256 = (
    "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301"
)
EXPECTED_INDEPENDENT_AUDITOR_SHA256 = (
    "c53bd7bc066d37bc593b910a109912c26719ecd5d61bd13974a6b2e826a51058"
)
EXPECTED_STATIC_TRUTH_AUDIT_SHA256 = (
    "f27ff7400a82a8b0667e172766026b9007e2155eb447ccae05bf6adf17094964"
)
EXPECTED_PHASE1_AUDIT_SHA256 = (
    "1118f5b293719bffe17217c5d24a6bf32f74a7a453b4ffd038fae7a34fce9416"
)
EXPECTED_PRODUCTION_DENIAL_SHA256 = (
    "c71b569da432723715922532e763b79dec6c0f9f04a08f84c0e190345c9d2b60"
)
EXPECTED_PHASE1_DECISION_SHA256 = (
    "be1ac60e7b0420fc762a030ad579c855f65b20e41e4c32b03d514a804c82e71d"
)
EXPECTED_SCIENTIFIC_SCOPE_CLARIFICATION_SHA256 = (
    "7e90ed186396f3e209f6591ccdd28df714b642137797c07e0ed048bd02656b2c"
)
EXPECTED_ASSEMBLER_SHA256 = (
    "7e749adb00ef8d552580217616e4732e838d08b8295bde7509b36d668ca6854e"
)
EXPECTED_OVERLAY_SHA256 = {
    "hps_gpr/__init__.py": "342aaa16dc390a3b79ef605987de8dc610b87e9bc774fe5edfec5e7a56883687",
    "hps_gpr/gpr.py": "1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5",
    "hps_gpr/io.py": "b36f8da7671a0fc0958b663e11d83a1a4421e90d1aab9b10e40c31ce078035db",
}
EXPECTED_REPOSITORY_RUNTIME_SHA256 = {
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
EXPECTED_GP_PACKAGE_SHA256 = {
    "__init__.py": "94a1bdc2a73f51aaba876412ab6ba49c1a9fc6aa5dd455a731e492d2d962ec65",
    "__main__.py": "abb947711ca644e5cb4b7f2d8eafe695e59e3d6c8e94175051d7061dd998b24a",
    "_fit.py": "aaeb0185770e57c5f441ccc676c82e93f6965f6e9a916c7df72849f382c93767",
    "_limit_setting.py": "8dbba37a929653a60510d8e3332c5592fed10c0646ab2c02297f2d7db8b0a4df",
    "_limit_setting_gaussian.py": "8dbba37a929653a60510d8e3332c5592fed10c0646ab2c02297f2d7db8b0a4df",
    "_mass_resolution.py": "9511413b274a12adff821afbd695b4d91078f45b7c6e4113d387e3b47a49c760",
    "_plot.py": "310cc1e0e96c5f8807b7e825d86b5e48d8749fe1e298cadd405d887a28712180",
    "_hist/__init__.py": "44df621fb101d9a21667209faafba26c5b8407bdf80ad96be5eca9a6c0de2d83",
    "_hist/io.py": "0abcb6fdaebe752c6eae8e4d7a6f70f446a0794e3bd4d56ee0fd9ddd554cd2cb",
    "_hist/manipulation.py": "56911fa0fa9cafb2e5285186a71861067c9ef585c1b9f0620fc5653c92f1b727",
}
EXPECTED_MASS_MEV = tuple(range(39, 181))
EXPECTED_MASS_GEV = np.asarray(EXPECTED_MASS_MEV, dtype=float) / 1000.0
EXPECTED_COLLECTION_PRODUCTS = {
    "accepted_extraction_rows.csv",
    "raw_primary_extraction_rows.csv",
    "optimizer_attempt_ledger.csv",
    "exclusion_ledger.csv",
    "closure_summary.csv",
    "zero_signal_bias_tests.csv",
    "task_product_audit.csv",
}
EXPECTED_BLINDING_STATEMENT = (
    "not full-data blind: the pre-existing 2016 10pct development shape entered "
    "the source-conditioned truth; full-100pct values entered truth construction "
    "only through the scalar 26--210 MeV normalization"
)
EXPECTED_SUPPORT_RANKING_STATEMENT = (
    "no support-specific full-100pct fit, local p0, or upper limit was used to "
    "rank support edges"
)
REQUIRED_OPTIMIZER_FIELDS = {
    "optimizer_restarts",
    "optimizer_random_state",
    "optimizer_warning_count",
    "optimizer_warnings",
}
CORE_NUMERIC_COLUMNS = (
    "mass_GeV",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "lml",
    "ls_init",
    "ls_opt",
    "const_init",
    "const_opt",
    "n_train",
)
BOUND_COLUMNS = (
    "const_at_lower",
    "const_at_upper",
    "ls_at_lower",
    "ls_at_upper",
)


class ObservedContractError(RuntimeError):
    """Raised before observed inference when any frozen declaration drifts."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ObservedContractError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ObservedContractError(
            f"{label} SHA-256 mismatch: {actual} != {expected}: {path}"
        )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ObservedContractError(f"missing JSON input: {path}")

    def reject_nonfinite(token: str) -> None:
        raise ObservedContractError(
            f"non-strict JSON numeric token {token!r} in {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
    )
    if not isinstance(payload, dict):
        raise ObservedContractError(f"JSON input is not an object: {path}")
    return payload


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if bool(invalid.any()):
        raise ObservedContractError(
            f"non-boolean values in {series.name}: "
            f"{sorted(normalized.loc[invalid].unique().tolist())}"
        )
    return normalized.isin({"true", "1", "yes"})


def bool_value(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ObservedContractError(f"invalid boolean {label}: {value!r}")


def validate_production_denial() -> Dict[str, Any]:
    """Validate the terminal, independently audited no-edge authorization state."""
    require_hash(
        INDEPENDENT_AUDITOR,
        EXPECTED_INDEPENDENT_AUDITOR_SHA256,
        "independent freeze auditor",
    )
    require_hash(
        STATIC_TRUTH_AUDIT,
        EXPECTED_STATIC_TRUTH_AUDIT_SHA256,
        "independent static-truth audit",
    )
    require_hash(
        PHASE1_AUDIT,
        EXPECTED_PHASE1_AUDIT_SHA256,
        "independent phase-1 audit",
    )
    require_hash(
        PHASE1_DECISION,
        EXPECTED_PHASE1_DECISION_SHA256,
        "canonical phase-1 decision",
    )
    require_hash(
        PRODUCTION_DENIAL,
        EXPECTED_PRODUCTION_DENIAL_SHA256,
        "production-authorization denial",
    )
    require_hash(
        SCIENTIFIC_SCOPE_CLARIFICATION,
        EXPECTED_SCIENTIFIC_SCOPE_CLARIFICATION_SHA256,
        "scientific-scope clarification",
    )
    denial = load_json(PRODUCTION_DENIAL)
    candidate_supports = [f"{value:03d}_210" for value in range(28, 35)]
    expected_claim_boundary = (
        "terminal failure of the predeclared conditional source-conditioned "
        "pull-recovery criterion; it is not an observed exclusion, sensitivity, "
        "coverage, or significance statement"
    )
    if (
        denial.get("audit_status") != "pass"
        or denial.get("status") != "production_blocked"
        or denial.get("stage") != "blocked_state"
        or denial.get("study_spec_sha256") != sha256(STUDY_SPEC)
        or denial.get("static_truth_audit_path")
        != "audit/static_truth_audit.json"
        or denial.get("static_truth_audit_sha256")
        != sha256(STATIC_TRUTH_AUDIT)
        or denial.get("phase1_selection_audit_path")
        != "audit/phase1_selection_audit.json"
        or denial.get("phase1_selection_audit_sha256") != sha256(PHASE1_AUDIT)
        or denial.get("canonical_phase1_decision_path")
        != "derived/analysis/phase1_selection_decision.json"
        or denial.get("canonical_phase1_decision_sha256")
        != sha256(PHASE1_DECISION)
        or denial.get("canonical_phase1_decision_status")
        != "no_provisional_edge"
        or denial.get("canonical_support_freeze_decision_present") is not False
        or denial.get("independent_selected_support") is not None
        or denial.get("exact_candidate_supports") != candidate_supports
        or denial.get("exact_phase2_supports") != []
        or denial.get("claim_boundary") != expected_claim_boundary
    ):
        raise ObservedContractError(
            "production-denial top-level authorization binding drift"
        )
    auditor = denial.get("independent_auditor", {})
    scope = denial.get("scientific_scope_clarification", {})
    authorization = denial.get("authorization", {})
    if (
        not isinstance(auditor, Mapping)
        or auditor.get("path") != "audit/independent_freeze_audit.py"
        or auditor.get("sha256") != sha256(INDEPENDENT_AUDITOR)
        or not isinstance(scope, Mapping)
        or scope.get("path") != "SCIENTIFIC_SCOPE_CLARIFICATION.md"
        or scope.get("sha256") != sha256(SCIENTIFIC_SCOPE_CLARIFICATION)
        or not isinstance(authorization, Mapping)
        or authorization.get("status") != "denied"
        or authorization.get("confirmation_authorized") is not False
        or authorization.get("observed_scan_authorized") is not False
        or authorization.get("combined_production_authorized") is not False
        or authorization.get("holdout_65MeV_authorized") is not False
        or authorization.get("required_protocol_action") != "stop without retuning"
        or authorization.get("reason")
        != "no eligible support passed the frozen phase-1 practical rule"
        or authorization.get("independent_auditor_sha256")
        != sha256(INDEPENDENT_AUDITOR)
        or authorization.get("static_truth_audit_sha256")
        != sha256(STATIC_TRUTH_AUDIT)
        or authorization.get("phase1_selection_audit_sha256")
        != sha256(PHASE1_AUDIT)
        or authorization.get("canonical_phase1_decision_sha256")
        != sha256(PHASE1_DECISION)
        or authorization.get("scientific_scope_clarification_sha256")
        != sha256(SCIENTIFIC_SCOPE_CLARIFICATION)
    ):
        raise ObservedContractError(
            "production-denial nested authorization binding drift"
        )
    summaries = denial.get("support_gate_summary", ())
    if not isinstance(summaries, list) or len(summaries) != 7:
        raise ObservedContractError("production-denial support inventory drift")
    by_support = {
        str(row.get("support")): row
        for row in summaries
        if isinstance(row, Mapping)
    }
    if set(by_support) != set(candidate_supports) or any(
        row.get("practical_acceptability_pass") is not False
        for row in by_support.values()
    ):
        raise ObservedContractError(
            "production denial does not show all seven practical-gate failures"
        )
    expected_technical_failures = {"030_210", "032_210"}
    found_technical_failures = {
        support
        for support, row in by_support.items()
        if row.get("technical_gate_pass") is False
    }
    if found_technical_failures != expected_technical_failures or any(
        row.get("technical_gate_pass") not in {True, False}
        for row in by_support.values()
    ):
        raise ObservedContractError(
            "production-denial technical-gate inventory drift"
        )
    return {
        "status": "production_blocked",
        "required_protocol_action": "stop without retuning",
        "independent_freeze_auditor_sha256": sha256(INDEPENDENT_AUDITOR),
        "static_truth_audit_sha256": sha256(STATIC_TRUTH_AUDIT),
        "phase1_selection_audit_sha256": sha256(PHASE1_AUDIT),
        "canonical_phase1_decision_sha256": sha256(PHASE1_DECISION),
        "production_authorization_denied_sha256": sha256(PRODUCTION_DENIAL),
        "scientific_scope_clarification_sha256": sha256(
            SCIENTIFIC_SCOPE_CLARIFICATION
        ),
        "exact_candidate_supports": candidate_supports,
        "exact_phase2_supports": [],
        "observed_scan_authorized": False,
        "combined_production_authorized": False,
    }


def static_preflight() -> Dict[str, Any]:
    require_hash(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "frozen protocol")
    require_hash(STUDY_SPEC, EXPECTED_STUDY_SPEC_SHA256, "frozen study spec")
    require_hash(BASE_CARD, EXPECTED_BASE_CARD_SHA256, "frozen v4.2 card")
    require_hash(OBSERVED_ROOT, EXPECTED_OBSERVED_ROOT_SHA256, "2016 full ROOT")
    production_authorization = validate_production_denial()
    require_hash(ASSEMBLER, EXPECTED_ASSEMBLER_SHA256, "combined-ledger assembler")
    spec = load_json(STUDY_SPEC)
    if spec.get("study_id") != STUDY_ID:
        raise ObservedContractError("study_spec.json names another study")
    if spec.get("frozen_protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ObservedContractError("study_spec frozen-protocol declaration drift")
    observed = spec.get("observed_input", {})
    if (
        observed.get("sha256") != EXPECTED_OBSERVED_ROOT_SHA256
        or observed.get("histogram") != "h_Minv_General_Final_1"
    ):
        raise ObservedContractError("study_spec observed-2016 declaration drift")
    runtime = activate_runtime()
    return {
        "status": "pass",
        "study_id": STUDY_ID,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "frozen_protocol_sha256": sha256(PROTOCOL),
        "base_card_sha256": sha256(BASE_CARD),
        "observed_root_sha256": sha256(OBSERVED_ROOT),
        "independent_freeze_auditor_sha256": sha256(INDEPENDENT_AUDITOR),
        "static_truth_audit_sha256": sha256(STATIC_TRUTH_AUDIT),
        "scientific_scope_clarification_sha256": sha256(
            SCIENTIFIC_SCOPE_CLARIFICATION
        ),
        "combined_ledger_assembler_sha256": sha256(ASSEMBLER),
        "production_authorization": production_authorization,
        "expected_mass_rows": len(EXPECTED_MASS_MEV),
        "expected_mass_low_MeV": EXPECTED_MASS_MEV[0],
        "expected_mass_high_MeV": EXPECTED_MASS_MEV[-1],
        "runtime": runtime,
    }


def activate_runtime() -> Dict[str, Any]:
    overlay = HERE / "runtime_overlay"
    package = overlay / "hps_gpr"
    for relative, expected in EXPECTED_OVERLAY_SHA256.items():
        require_hash(overlay / relative, expected, f"observed runtime {relative}")
    for filename, expected in EXPECTED_REPOSITORY_RUNTIME_SHA256.items():
        require_hash(
            REPO / "hps_gpr" / filename,
            expected,
            f"archived-byte-identical fallback hps_gpr/{filename}",
        )
    for filename, expected in EXPECTED_GP_PACKAGE_SHA256.items():
        require_hash(REPO / "gp" / filename, expected, f"frozen gp/{filename}")

    overlay_text = str(overlay.resolve())
    repo_text = str(REPO.resolve())
    sys.path[:] = [
        item
        for item in sys.path
        if str(Path(item or ".").resolve()) not in {overlay_text, repo_text}
    ]
    sys.path.insert(0, repo_text)
    sys.path.insert(0, overlay_text)

    import hps_gpr
    import hps_gpr.cli as runtime_cli
    import hps_gpr.gpr as runtime_gpr
    import hps_gpr.io as runtime_io
    import hps_gpr.scan as runtime_scan
    import gp as runtime_gp

    expected_origins = {
        "hps_gpr": package / "__init__.py",
        "hps_gpr.gpr": package / "gpr.py",
        "hps_gpr.io": package / "io.py",
        "hps_gpr.cli": REPO / "hps_gpr" / "cli.py",
        "hps_gpr.scan": REPO / "hps_gpr" / "scan.py",
        "gp": REPO / "gp" / "__init__.py",
    }
    modules = {
        "hps_gpr": hps_gpr,
        "hps_gpr.gpr": runtime_gpr,
        "hps_gpr.io": runtime_io,
        "hps_gpr.cli": runtime_cli,
        "hps_gpr.scan": runtime_scan,
        "gp": runtime_gp,
    }
    origins: Dict[str, str] = {}
    for name, module in modules.items():
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        expected = expected_origins[name].resolve()
        if origin != expected:
            raise ObservedContractError(
                f"{name} imported from {origin}, not attested path {expected}"
            )
        origins[name] = str(origin)

    fields = set(getattr(runtime_io.BlindPrediction, "__dataclass_fields__", {}))
    missing = sorted(REQUIRED_OPTIMIZER_FIELDS - fields)
    if missing:
        raise ObservedContractError(
            f"BlindPrediction lacks optimizer provenance fields: {missing}"
        )
    if "random_state" not in inspect.signature(runtime_gpr.fit_gpr).parameters:
        raise ObservedContractError("fit_gpr lacks explicit optimizer random_state")
    return {
        "overlay_root": str(overlay.resolve()),
        "origins": origins,
        "optimizer_fields": sorted(REQUIRED_OPTIMIZER_FIELDS),
    }


def validate_freeze(path: Path = FREEZE) -> Dict[str, Any]:
    validate_production_denial()
    raise ObservedContractError(
        "observed/card production is terminally blocked: no support edge passed "
        "the frozen phase-1 practical gate; required action is stop without retuning"
    )

    # Retained as a fail-closed specification of the post-freeze contract.  It
    # is unreachable in this release because the independently audited terminal
    # denial above may not be bypassed or removed.
    path = path.expanduser().resolve()
    if path != FREEZE.resolve():
        raise ObservedContractError(
            f"freeze must be the campaign-local decision {FREEZE.resolve()}"
        )
    decision = load_json(path)
    required_true = (
        "initial_gate_pass",
        "continuation_gate_pass",
        "full100_gate_pass",
        "observed_scan_authorized",
    )
    required_false = (
        "absolute_upper_limit_used_for_selection",
        "retuning_after_confirmation",
        "holdout_65MeV_used_for_selection",
    )
    if decision.get("study_id") != STUDY_ID:
        raise ObservedContractError("support freeze names another study")
    if decision.get("status") != "support_edge_frozen":
        raise ObservedContractError("2016 support edge is not frozen")
    for key in required_true:
        if decision.get(key) is not True:
            raise ObservedContractError(f"freeze does not pass/authorize {key}")
    for key in required_false:
        if decision.get(key) is not False:
            raise ObservedContractError(f"freeze violates control {key}")
    if decision.get("study_spec_sha256") != sha256(STUDY_SPEC):
        raise ObservedContractError("freeze does not bind the live study_spec.json")
    if decision.get("frozen_protocol_sha256") != sha256(PROTOCOL):
        raise ObservedContractError("freeze does not bind the frozen protocol")
    low = int(decision.get("selected_support_low_MeV", -1))
    high = int(decision.get("support_high_MeV", -1))
    if low not in range(28, 34) or high != 210:
        raise ObservedContractError(f"frozen support {(low, high)} is invalid")
    expected_range = [low / 1000.0, high / 1000.0]
    found_range = decision.get("data_range_2016")
    if (
        not isinstance(found_range, list)
        or len(found_range) != 2
        or not np.allclose(found_range, expected_range, rtol=0.0, atol=1e-15)
    ):
        raise ObservedContractError("freeze data_range_2016 is inconsistent")
    for name, record in decision.get("products", {}).items():
        if not isinstance(record, Mapping) or "sha256" not in record:
            raise ObservedContractError(f"invalid frozen product declaration {name}")
        require_hash(
            HERE / "derived" / "analysis" / name,
            str(record["sha256"]),
            f"support-freeze product {name}",
        )
    validate_independent_freeze_audit(decision)
    return decision


def validate_independent_freeze_audit(
    freeze: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require the independent static -> phase-1 -> confirmation audit chain."""
    static = load_json(STATIC_TRUTH_AUDIT)
    phase1 = load_json(PHASE1_AUDIT)
    confirmation = load_json(CONFIRMATION_AUDIT)
    study_spec_sha = sha256(STUDY_SPEC)
    static_file_sha = sha256(STATIC_TRUTH_AUDIT)
    phase1_file_sha = sha256(PHASE1_AUDIT)
    confirmation_file_sha = sha256(CONFIRMATION_AUDIT)
    freeze_sha = sha256(FREEZE)
    scope_sha = sha256(SCIENTIFIC_SCOPE_CLARIFICATION)
    if (
        static.get("status") != "pass"
        or static.get("stage") != "static_truth"
        or static.get("study_spec_sha256") != study_spec_sha
    ):
        raise ObservedContractError("independent static-truth audit is not passing")
    static_scope = static.get("scientific_scope_clarification", {})
    if (
        not isinstance(static_scope, Mapping)
        or static_scope.get("path") != "SCIENTIFIC_SCOPE_CLARIFICATION.md"
        or static_scope.get("sha256") != scope_sha
        or static_scope.get("expected_sha256")
        != EXPECTED_SCIENTIFIC_SCOPE_CLARIFICATION_SHA256
        or static_scope.get("hash_match") is not True
    ):
        raise ObservedContractError(
            "static audit does not bind the scientific-scope clarification"
        )
    broad_tail = static.get("broad_tail", {})
    required_scope = (
        "conditional source-conditioned stress truth only; not a physical "
        "background generator, coverage ensemble, expected-limit calibration, "
        "exclusion, or significance calibration. The 10pct development shape "
        "is partial observed-shape information and is not established as "
        "statistically independent of the full sample"
    )
    if (
        broad_tail.get("fit_ok") is not False
        or broad_tail.get("waiver_required") is not True
        or broad_tail.get("waiver_acknowledged") is not True
        or broad_tail.get("waiver_scope") != required_scope
    ):
        raise ObservedContractError(
            "broad-tail conditional-stress-only waiver is absent or broadened"
        )
    shape_use = static.get("full_observed_shape_use_audit", {})
    required_description = (
        "pre-existing 2016 10pct development sample/subset; partial observed-"
        "shape information entered the stress truth. Do not call it an "
        "independent sample without run/event-level disjointness provenance"
    )
    if (
        not isinstance(shape_use, Mapping)
        or shape_use.get("permitted_value_use")
        != "one scalar sum over 26--210 MeV"
        or shape_use.get(
            "full_100pct_values_entered_truth_only_as_scalar_26_210MeV_normalization"
        )
        is not True
        or shape_use.get(
            "support_specific_full_100pct_fit_p0_or_upper_limit_used_for_ranking"
        )
        is not False
        or shape_use.get(
            "ten_pct_development_shape_entered_source_conditioned_truth"
        )
        is not True
        or shape_use.get(
            "ten_pct_statistical_independence_from_full_100pct_unproven"
        )
        is not True
        or shape_use.get("ten_pct_bins_never_exceed_full_100pct_bins") is not True
        or shape_use.get("required_description") != required_description
        or int(
            static.get("toy_reproduction", {}).get(
                "n_toys_reproduced_bitwise", -1
            )
        )
        != 100
    ):
        raise ObservedContractError("independent static truth controls did not pass")

    supports = tuple(f"{value:03d}_210" for value in range(28, 35))
    phase1_lane_hashes = phase1.get("phase1_lane_task_hashes", {})
    if (
        phase1.get("status") != "pass"
        or phase1.get("stage") != "phase1_selection"
        or phase1.get("study_spec_sha256") != study_spec_sha
        or phase1.get("static_truth_audit_path")
        != "audit/static_truth_audit.json"
        or phase1.get("static_truth_audit_sha256") != static_file_sha
        or phase1.get("static_truth_audit_content_sha256")
        != canonical_json_sha256(static)
        or not isinstance(phase1_lane_hashes, Mapping)
        or set(phase1_lane_hashes) != set(supports)
        or phase1.get("observed_scan_authorized") is not False
    ):
        raise ObservedContractError("independent phase-1 audit chain/inventory drift")
    for support, digest in phase1_lane_hashes.items():
        if (
            support not in supports
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ObservedContractError("invalid phase-1 lane inventory hash")

    low = int(freeze.get("selected_support_low_MeV", -1))
    high = int(freeze.get("support_high_MeV", -1))
    selected = str(freeze.get("selected_support", ""))
    expected_selected = f"{low:03d}_{high}"
    if selected != expected_selected:
        raise ObservedContractError(
            "canonical freeze selected-support label/numeric edges disagree"
        )
    phase2_supports = tuple(map(str, freeze.get("phase2_supports", ())))
    if (
        phase1.get("phase1_decision_sha256")
        != freeze.get("phase1_decision_sha256")
        or phase1.get("independent_selected_support") != selected
        or tuple(map(str, phase1.get("independent_phase2_supports", ())))
        != phase2_supports
    ):
        raise ObservedContractError(
            "independent phase-1 selection differs from canonical freeze"
        )

    expected_authorization_scope = (
        "support freeze only; broad-tail waiver remains conditional-stress-truth-only"
    )
    authorization = confirmation.get("authorization", {})
    confirmation_scope = confirmation.get("scientific_scope_clarification", {})
    if (
        confirmation.get("status") != "pass"
        or confirmation.get("stage") != "confirmation"
        or confirmation.get("study_spec_sha256") != study_spec_sha
        or confirmation.get("static_truth_audit_path")
        != "audit/static_truth_audit.json"
        or confirmation.get("static_truth_audit_sha256") != static_file_sha
        or confirmation.get("phase1_selection_audit_path")
        != "audit/phase1_selection_audit.json"
        or confirmation.get("phase1_selection_audit_sha256") != phase1_file_sha
        or confirmation.get("phase1_selection_audit_content_sha256")
        != canonical_json_sha256(phase1)
        or confirmation.get("canonical_support_freeze_decision_path")
        != "derived/analysis/support_freeze_decision.json"
        or confirmation.get("canonical_support_freeze_decision_sha256")
        != freeze_sha
        or confirmation.get("independent_selected_support") != selected
        or tuple(map(str, confirmation.get("exact_phase2_supports", ())))
        != phase2_supports
        or confirmation.get("independently_frozen") is not True
        or confirmation.get("observed_scan_authorized") is not True
        or confirmation.get("authorization_scope")
        != expected_authorization_scope
        or not isinstance(confirmation_scope, Mapping)
        or confirmation_scope.get("path")
        != "SCIENTIFIC_SCOPE_CLARIFICATION.md"
        or confirmation_scope.get("sha256") != scope_sha
        or confirmation_scope.get("expected_sha256")
        != EXPECTED_SCIENTIFIC_SCOPE_CLARIFICATION_SHA256
        or confirmation_scope.get("hash_match") is not True
    ):
        raise ObservedContractError(
            "independent confirmation audit does not authorize this freeze"
        )
    if not isinstance(authorization, Mapping):
        raise ObservedContractError(
            "independent confirmation authorization is not a mapping"
        )
    expected_range = [low / 1000.0, high / 1000.0]
    authorization_range = authorization.get("data_range_2016")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("canonical_support_freeze_decision_sha256")
        != freeze_sha
        or authorization.get("static_truth_audit_sha256") != static_file_sha
        or authorization.get("phase1_selection_audit_sha256")
        != phase1_file_sha
        or authorization.get("selected_support") != selected
        or int(authorization.get("selected_support_low_MeV", -1)) != low
        or int(authorization.get("support_high_MeV", -1)) != high
        or not isinstance(authorization_range, list)
        or len(authorization_range) != 2
        or not np.allclose(
            authorization_range, expected_range, rtol=0.0, atol=1e-15
        )
        or authorization.get("broad_tail_waiver_scope")
        != "conditional source-conditioned stress truth only"
        or authorization.get("blinding_statement")
        != EXPECTED_BLINDING_STATEMENT
        or authorization.get("support_ranking_statement")
        != EXPECTED_SUPPORT_RANKING_STATEMENT
        or authorization.get("scientific_scope_clarification_path")
        != "SCIENTIFIC_SCOPE_CLARIFICATION.md"
        or authorization.get("scientific_scope_clarification_sha256")
        != scope_sha
    ):
        raise ObservedContractError(
            "independent confirmation authorization payload drift"
        )
    collections = confirmation.get("collection_input_hashes", {})
    if not isinstance(collections, Mapping) or set(collections) != set(
        phase2_supports
    ):
        raise ObservedContractError("independent full-100 support inventory drift")
    for support, record in collections.items():
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "collection_summary_sha256",
                "derived_sha256",
                "all_100_task_markers_and_ledgers_sha256",
            }
            or not isinstance(record.get("collection_summary_sha256"), str)
            or len(record["collection_summary_sha256"]) != 64
            or not isinstance(record.get("derived_sha256"), Mapping)
            or set(record["derived_sha256"]) != EXPECTED_COLLECTION_PRODUCTS
            or not isinstance(
                record.get("all_100_task_markers_and_ledgers_sha256"), str
            )
            or len(record["all_100_task_markers_and_ledgers_sha256"]) != 64
        ):
            raise ObservedContractError(
                f"invalid full-100 task inventory hash for {support}"
            )
        inventory_digests = [
            record["collection_summary_sha256"],
            record["all_100_task_markers_and_ledgers_sha256"],
            *record["derived_sha256"].values(),
        ]
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in inventory_digests
        ):
            raise ObservedContractError(
                f"non-SHA256 full-100 inventory digest for {support}"
            )
    summaries = confirmation.get("support_cohort_summaries", ())
    expected_summary_keys = {
        (support, cohort)
        for support in phase2_supports
        for cohort in ("initial_0_24", "continuation_25_99", "full_0_99")
    }
    found_summary_keys = {
        (str(row.get("support")), str(row.get("cohort")))
        for row in summaries
        if isinstance(row, Mapping)
    }
    if found_summary_keys != expected_summary_keys or len(summaries) != len(
        expected_summary_keys
    ):
        raise ObservedContractError("independent confirmation cohort inventory drift")
    selected_summaries = [
        row for row in summaries if str(row.get("support")) == selected
    ]
    if len(selected_summaries) != 3 or any(
        row.get("technical_gate_pass") is not True
        or row.get("practical_acceptability_pass") is not True
        for row in selected_summaries
    ):
        raise ObservedContractError(
            "independent confirmation does not pass all selected-support cohorts"
        )
    return {
        "independent_freeze_auditor_sha256": sha256(INDEPENDENT_AUDITOR),
        "static_truth_audit_sha256": sha256(STATIC_TRUTH_AUDIT),
        "scientific_scope_clarification_sha256": scope_sha,
        "phase1_selection_audit_sha256": phase1_file_sha,
        "confirmation_freeze_audit_sha256": confirmation_file_sha,
        "canonical_freeze_sha256": freeze_sha,
        "selected_support": selected,
        "phase2_supports": list(phase2_supports),
        "broad_tail_waiver_scope": required_scope,
    }


def validate_mass_grid(frame: pd.DataFrame, expected_rows: int = 142) -> None:
    if len(frame) != expected_rows:
        raise ObservedContractError(
            f"expected {expected_rows} observed rows, found {len(frame)}"
        )
    if "dataset" not in frame or "mass_GeV" not in frame:
        raise ObservedContractError("observed CSV lacks dataset/mass_GeV")
    if set(frame["dataset"].astype(str).str.replace(r"\.0$", "", regex=True)) != {
        "2016"
    }:
        raise ObservedContractError("observed CSV contains a non-2016 dataset")
    masses = frame["mass_GeV"].to_numpy(float)
    expected = EXPECTED_MASS_GEV if expected_rows == 142 else masses
    if expected_rows == 142 and not np.array_equal(masses, expected):
        raise ObservedContractError("observed CSV is not exact ordered 39--180 MeV")


def validate_card(card_path: Path, manifest_path: Path, freeze: Mapping[str, Any]) -> Dict[str, Any]:
    card_path = card_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if card_path != CARD.resolve() or manifest_path != CARD_MANIFEST.resolve():
        raise ObservedContractError("only the campaign-local frozen observed card is allowed")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "observed_2016_card_frozen":
        raise ObservedContractError("observed card manifest is not frozen")
    bindings = {
        "study_id": STUDY_ID,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "frozen_protocol_sha256": sha256(PROTOCOL),
        "support_freeze": "derived/analysis/support_freeze_decision.json",
        "support_freeze_sha256": sha256(FREEZE),
        "independent_freeze_auditor": "audit/independent_freeze_audit.py",
        "independent_freeze_auditor_sha256": sha256(INDEPENDENT_AUDITOR),
        "static_truth_audit": "audit/static_truth_audit.json",
        "static_truth_audit_sha256": sha256(STATIC_TRUTH_AUDIT),
        "scientific_scope_clarification": "SCIENTIFIC_SCOPE_CLARIFICATION.md",
        "scientific_scope_clarification_sha256": sha256(
            SCIENTIFIC_SCOPE_CLARIFICATION
        ),
        "phase1_selection_audit": "audit/phase1_selection_audit.json",
        "phase1_selection_audit_sha256": sha256(PHASE1_AUDIT),
        "confirmation_freeze_audit": "audit/confirmation_freeze_audit.json",
        "confirmation_freeze_audit_sha256": sha256(CONFIRMATION_AUDIT),
        "combined_ledger_assembler": "assemble_reviewed_state_ledger.py",
        "combined_ledger_assembler_sha256": sha256(ASSEMBLER),
        "base_card_sha256": sha256(BASE_CARD),
        "observed_root_sha256": sha256(OBSERVED_ROOT),
        "card_sha256": sha256(CARD),
        "card_builder_sha256": sha256(HERE / "build_observed_2016_card.py"),
    }
    for key, expected in bindings.items():
        if manifest.get(key) != expected:
            raise ObservedContractError(f"observed card manifest binding drift: {key}")
    if (
        int(manifest.get("selected_support_low_MeV", -1))
        != int(freeze["selected_support_low_MeV"])
        or int(manifest.get("support_high_MeV", -1))
        != int(freeze["support_high_MeV"])
        or not np.allclose(
            manifest.get("data_range_2016", []),
            freeze["data_range_2016"],
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ObservedContractError("observed card manifest support binding drift")
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise ObservedContractError("observed card is not a YAML mapping")
    expected_range = list(map(float, freeze["data_range_2016"]))
    exact = {
        "path_2015": "",
        "path_2016": "inputs/source_2016_full.root",
        "path_2021": "",
        "hist_2016": "h_Minv_General_Final_1",
        "range_2016": [0.039, 0.180],
        "data_range_2016": expected_range,
        "enable_2015": False,
        "enable_2016": True,
        "enable_2021": False,
        "mass_step_gev": 0.001,
        "blind_nsigma": 2.25,
        "gp_train_exclude_nsigma": 2.25,
        "neighborhood_rebin": 5,
        "n_restarts": 12,
        "scan_parallel": True,
        "scan_n_workers": 4,
        "scan_parallel_backend": "threading",
        "scan_threads_per_worker": 1,
        "cls_alpha": 0.1,
        "cls_mode": "asymptotic",
        "cls_num_toys": 0,
        "make_ul_bands": False,
        "do_combined_bands": False,
        "inject_signal": False,
        "extract_allow_negative": True,
        "do_combined": False,
        "save_fit_json": True,
        "save_per_mass_folders": True,
        "save_plots": False,
        "scan_require_two_sidebands": True,
        "scan_edge_guard_nsigma": 2.25,
        "eps2_density_nsigma": 1.64,
        "kernel_ls_policy": "resolution_scaled_local",
        "pre_log": True,
        "alpha_model": "1/y",
        "radiative_penalty_on": True,
        "radiative_penalty_frac_2016": 0.07,
        "combined_bands_n_toys": 0,
        "ul_bands_toys": 0,
        "make_eps2_bands": False,
        "run_limit_bands_on": "",
        "output_dir": "observed_scan/2016_full_primary",
    }
    for key, expected in exact.items():
        found = card.get(key)
        if isinstance(expected, list):
            if not np.allclose(found, expected, rtol=0.0, atol=1e-15):
                raise ObservedContractError(f"observed card drift for {key}: {found}")
        elif found != expected:
            raise ObservedContractError(f"observed card drift for {key}: {found}")
    lower = card.get("kernel_ls_res_lower_factor_by_dataset", {}).get("2016")
    upper = card.get("kernel_ls_res_upper_factor_by_dataset", {}).get("2016")
    if not math.isclose(float(lower), 0.9) or not math.isclose(float(upper), 12.0):
        raise ObservedContractError("2016 k0.9/k12 kernel bounds drift")
    if card.get("data_visibility", {}).get("2016") != "observed":
        raise ObservedContractError("2016 data visibility is not observed")

    base = yaml.safe_load(BASE_CARD.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or set(base) != set(card):
        raise ObservedContractError("observed card key inventory differs from base card")
    allowed_changes = {
        "path_2015",
        "path_2016",
        "path_2021",
        "enable_2015",
        "enable_2021",
        "data_range_2016",
        "scan_n_workers",
        "make_ul_bands",
        "do_combined_bands",
        "combined_bands_n_toys",
        "do_combined",
        "debug_print",
        "fail_fast",
        "output_dir",
    }
    unexpected_changes = sorted(
        key for key in card if key not in allowed_changes and card[key] != base[key]
    )
    if unexpected_changes:
        raise ObservedContractError(
            f"observed card has non-authorized base-card changes: {unexpected_changes}"
        )
    return manifest


def candidate_issue_reasons(row: pd.Series) -> Sequence[str]:
    reasons = []
    for column in CORE_NUMERIC_COLUMNS:
        try:
            finite = np.isfinite(float(row[column]))
        except Exception:
            finite = False
        if not finite:
            reasons.append(f"nonfinite_{column}")
    try:
        extraction_success = bool_value(
            row.get("extract_success", False), "extract_success"
        )
    except ObservedContractError:
        extraction_success = False
    if not extraction_success:
        reasons.append("extraction_failed")
    try:
        density_covered = bool_value(
            row.get("density_window_fully_covered", False),
            "density_window_fully_covered",
        )
    except ObservedContractError:
        density_covered = False
    if not density_covered:
        reasons.append("density_not_fully_covered")
    bound_contact = False
    for column in BOUND_COLUMNS:
        try:
            bound_contact = bound_contact or bool_value(
                row.get(column, False), column
            )
        except ObservedContractError:
            bound_contact = True
    if bound_contact:
        reasons.append("kernel_bound_contact")
    try:
        if float(row["sigma_A"]) <= 0.0:
            reasons.append("nonpositive_sigma_A")
    except Exception:
        pass
    try:
        covariance_valid = bool_value(
            row.get("covariance_valid", False), "covariance_valid"
        )
    except ObservedContractError:
        covariance_valid = False
    if not covariance_valid:
        reasons.append("covariance_invalid")
    try:
        covariance_relative = float(
            row.get("covariance_min_eigenvalue_relative", float("nan"))
        )
        if not np.isfinite(covariance_relative):
            reasons.append("covariance_eigenvalue_missing")
        elif covariance_relative < -0.01:
            reasons.append("covariance_eigenvalue_gate")
    except Exception:
        reasons.append("covariance_eigenvalue_missing")
    try:
        exact_start = (
            abs(math.log(float(row["ls_opt"]) / float(row["ls_init"]))) < 1e-8
            and abs(
                math.log(float(row["const_opt"]) / float(row["const_init"]))
            )
            < 1e-8
        )
    except Exception:
        exact_start = False
    if exact_start:
        reasons.append("optimizer_exact_start")
    try:
        if int(row.get("optimizer_warning_count", 0)) > 0:
            reasons.append("optimizer_warning")
    except Exception:
        reasons.append("optimizer_warning_count_invalid")
    return tuple(dict.fromkeys(reasons))


def branch_match(left: pd.Series, right: pd.Series) -> bool:
    required = ("lml", "ls_opt", "const_opt", "sigma_A", "n_train")
    try:
        if not all(
            np.isfinite(float(left[key])) and np.isfinite(float(right[key]))
            for key in required
        ):
            return False
        n_train = max(1.0, min(float(left["n_train"]), float(right["n_train"])))
        if abs(float(left["lml"]) - float(right["lml"])) / n_train > 0.001:
            return False
        for key, limit in (
            ("ls_opt", 0.01),
            ("const_opt", 0.05),
            ("sigma_A", 0.02),
        ):
            a = float(left[key])
            b = float(right[key])
            if a <= 0.0 or b <= 0.0 or abs(math.log(a / b)) > limit:
                return False
        return True
    except Exception:
        return False


def eligible_candidate(row: pd.Series) -> bool:
    disqualifying = {
        reason
        for reason in candidate_issue_reasons(row)
        if reason != "optimizer_warning"
    }
    return not disqualifying
