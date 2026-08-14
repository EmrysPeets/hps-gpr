#!/usr/bin/env python3
"""Fail-closed release audit for the v4.8.3 residual-structured study.

The component validators and preflights are read-only.  This wrapper writes
only the final validation report, release inventory, and note-source manifest.
It never launches a fit or rewrites a task product.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY_ID = "v4p8p3_2021_residual_truths_20260814"
PDF_NAME = f"HPS_GPR_Analysis_Note_{STUDY_ID}.pdf"
NOTE_PDF = HERE / "note" / PDF_NAME
OUTPUT_PDF = REPO / "output" / "pdf" / STUDY_ID / PDF_NAME

MODELS = ("knot_spline", "regional_blend")
SCENARIOS = (
    "2021_1pct",
    "2021_1pct_x10",
    "2021_1pct_x100",
    "2021_10pct",
    "2021_10pct_x10",
)
MASSES = (65, 90, 120, 180, 210)
INJECTIONS = (0.0, 1.0, 3.0, 5.0)

EXPECTED_SHA256 = {
    "MODEL_PROTOCOL.json": "3a68c17913aa16567605399d91c14fb2ffe03aaaa6ccfa98084044514cbbd219",
    "residual_models.py": "6ac6bfe4d872b1cea0c650d2384e78b14d0ad1e758dee9f2e61d50afee736e5d",
    "fit_residual_models.py": "ae2b523dd92ab2332c77114e699d5be86f1040af8e9d3d32750396ee5769e4d8",
    "derived/source_fit_and_influence.json": "f39bb7acdc73709abc2a2483c5d6e99b944c300415653fd692abcae68933ff0e",
    "build_residual_toys.py": "452a402f6c653ae09466e3e4d51e797e9af33b1222427836ee88916ab51755bc",
    "inputs/residual_structured_nested_toys.root": "62938abcc35184ba03ce15d5533b1e6daff1c69c3742eb49143f95f46ebaaca0",
    "inputs/residual_structured_nested_toys.manifest.json": "ed3631d12a348b6da06c1739e05b05cb0fc9d1b7d60ad3adff1b87a18be2fb94",
    "run_residual_length_pilot.py": "205420bd293404bab08af2cb0230ad66c3dcc87dcfc627fdf3b5a13bd928dbd3",
    "derived/residual_length_pilot/common_ceiling_disposition.json": "a29b1a4de2bcfe35983112c0d4fd1435875e208c8bcd0da0fac98d3edd3bb77a",
    "run_residual_closure.py": "385384b311e954ac31cc5cf587b3d963ae361930fa5a53c4ec2127d4d6bfd614",
    "derived/residual_closure/knot_spline/collection_summary.json": "2b95636c0f6a60789a225b9cceebeb4330a0dc7a272b76733c1ef08e815ccfb1",
    "derived/residual_closure/regional_blend/collection_summary.json": "2a9bf11917c843361703f78b3b5976793ff400a13f62de4854ad66ffeb1fe797",
    "compile_residual_closure_comparison.py": "27071b04c0b1be53d4c02f3de2bff61956c116a733a77e6acbf35fbe6d1474be",
    "figures/closure_comparison_figure_manifest.json": "45413aea78c5d3ca40d4d9eb568133c2e7f485390afbdb878d7f5c05f8d83fea",
    "CEILING_REMEDIATION_ADDENDUM.json": "40d81bca0ded24821d2f1213e3df9a6ab1c904242b0e89ea2ad5773533e5fb1d",
    "build_residual_length_ceiling_remediation_toys.py": "3623ef4c9faa85514984fe1255aa698dae18caebd46b7e10ae3eb6684455e772",
    "run_residual_length_ceiling_remediation.py": "3e51b31ec9e358cc6357d7f079d5ad1e348f6c8c0765118e7bbb44e521d44fd3",
    "inputs/residual_length_ceiling_remediation_toys.root": "3a732b40aabc50b2665753b06377bed3daafa069127343df7ecbd6660cdc5857",
    "inputs/residual_length_ceiling_remediation_toys.manifest.json": "2d1203593bd7665f4f763e2513647efd757b7d0e4a505ef637e563b0a0c2d053",
    "derived/residual_length_ceiling_remediation/selection/selection_disposition.json": "80a40a60858674b334630019e7b442e8668b782f52bd2376a8a7dfc6e0e0b7d2",
    "derived/residual_length_ceiling_remediation/confirmation/final_disposition.json": "9497a94a69109ab4519e2d4b02e71fabf8e935013f2a7646cac44822357dcdf9",
    "make_ceiling_remediation_figure.py": "177e25cbd733ba603ae898c7855a9f535079763ecf05f0a9a0efca9f5b6c437f",
    "figures/ceiling_remediation_figure_manifest.json": "4d8867fc87130983f1cab1f162355c7db3282b4ce583798c92cbeba8761d05ca",
    "note/source_overlays/main.tex": "b88b717efcce69c05795f0fa3c25a42b29a1ed830f7da89326430e364fb0157e",
    "note/source_overlays/sections/00_change_log.tex": "a1f44f09f431f5ef7b782bb7a65205fb27fd9bee9ba91b64a330b7072594542a",
    "note/source_overlays/sections/05_toys_validation.tex": "4ee00e6c7f50209c7e18e5a11ae7d29afe46ec2b7079e5cc28cb25ff8ea54792",
    "note/source_overlays/sections/07_conclusions.tex": "b38d13bb1909f57db6f08bcbbf4086b0e3f58645d0ad45514c9c3079ec327817",
    "note/source_overlays/sections/subsection_v4p8p3_residual_truths.tex": "9ef0d8fe93ade15a41eb970db1231200243c39b64424599cee886f0f713e54d8",
    f"note/{PDF_NAME}": "2cc760fea64a5df2f639aa736ee6810baa2b56e9a7d9d4c6bf6ccbce855b50b7",
    "note/tectonic_main.log": "b9e160e634c1b5e21952cd93d126285b90b4cc8441edd1ce68cb9abb5cea3e29",
}

REQUIRED_FIGURES = (
    "v4p8p3_source_qualification_and_residuals",
    "v4p8p3_signal_influence_audit",
    "v4p8p3_five_lane_toy_sampling_20",
    "v4p8p3_length_ceiling_pilot",
    "v4p8p3_length_ceiling_remediation",
    "v4p8p3_zero_signal_conditional_closure_20toy",
    "v4p8p3_injected_recovery_20toy",
)


class AuditError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise AuditError(f"JSON object required: {path}")
    return payload


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("path", "role", "size_bytes", "sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(self, name: str, action: Callable[[], Mapping[str, Any]]) -> None:
        try:
            evidence = dict(action())
            self.checks.append({"id": name, "status": "pass", "evidence": evidence})
        except Exception as exc:
            self.checks.append(
                {"id": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
            )

    @property
    def failed(self) -> int:
        return sum(row["status"] != "pass" for row in self.checks)


def check_hash_locks() -> Mapping[str, Any]:
    for relative, expected in EXPECTED_SHA256.items():
        path = HERE / relative
        require(path.is_file(), f"missing locked artifact: {relative}")
        require(sha256_file(path) == expected, f"hash drift: {relative}")
    require(sha256_file(OUTPUT_PDF) == EXPECTED_SHA256[f"note/{PDF_NAME}"], "output PDF drift")
    return {"locked_artifacts": len(EXPECTED_SHA256) + 1}


def run_component_validators() -> Mapping[str, Any]:
    commands = {
        "source": [sys.executable, "fit_residual_models.py", "validate"],
        "pilot": [sys.executable, "run_residual_length_pilot.py", "validate"],
        "closure_k2": [sys.executable, "run_residual_closure.py", "--model", "knot_spline", "preflight"],
        "closure_regional": [sys.executable, "run_residual_closure.py", "--model", "regional_blend", "preflight"],
        "remediation": [sys.executable, "run_residual_length_ceiling_remediation.py", "validate"],
    }
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    evidence: dict[str, Any] = {}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=HERE, env=env, text=True, capture_output=True)
        require(result.returncode == 0, f"{name} validator failed: {result.stderr[-1000:]}")
        payload = json.loads(result.stdout)
        require(payload.get("status", payload.get("artifact_integrity_status")) == "pass", f"{name} did not pass")
        evidence[name] = "pass"
    return evidence


def check_source_and_influence() -> Mapping[str, Any]:
    result = load_json(HERE / "derived/source_fit_and_influence.json")
    require(result.get("study_id") == STUDY_ID, "source result study id drift")
    require(result.get("model_selection_frozen_before_injection") is True, "model choice was not frozen")
    require(result.get("model_selection_uses_gpr_results") is False, "model choice used GPR results")
    summaries = result["signal_influence_audit"]["summaries"]
    for model in MODELS:
        record = result["models"][model]
        require(record["strict_generator_qualification_passed"] is False, f"{model} unexpectedly qualified")
        require(record["conditional_toy_run_authorized"] is True, f"{model} stress run unauthorized")
        require(record["promotion_scope"] == "requested conditional stress only", f"{model} scope drift")
        require(summaries[model]["signal_influence_gate_passed"] is False, f"{model} rigidity disposition drift")
    require(len(result["signal_influence_audit"]["rows"]) == 492, "influence row inventory drift")
    return {
        "strict_source_qualified": {model: False for model in MODELS},
        "signal_influence_passed": {model: False for model in MODELS},
        "maximum_tangent_fraction": {
            model: summaries[model]["maximum_abs_tangent_absorption_fraction"] for model in MODELS
        },
    }


def check_original_pilot() -> Mapping[str, Any]:
    contract = load_json(HERE / "qa/residual_length_pilot/scan_contract.json")
    disposition = load_json(HERE / "derived/residual_length_pilot/common_ceiling_disposition.json")
    canonical = canonical_json_hash(contract)
    require(canonical == disposition["scan_contract_sha256"], "pilot canonical contract hash drift")
    require(disposition["factor20_gate_passed"] is False, "factor-20 failure drift")
    require(disposition["selected_common_upper_factor"] == 25, "factor-25 fallback drift")
    rows = csv_rows(HERE / "derived/residual_length_pilot/selected_trajectory_ledger.csv")
    require(len(rows) == 270, "pilot selected-state inventory drift")
    k2_native = [
        row for row in rows
        if row["model"] == "knot_spline"
        and row["scenario"] == "2021_1pct"
        and int(row["upper_factor"]) == 25
    ]
    contacts = sum(truth(row["ell_at_upper_exact"]) or truth(row["ell_near_upper"]) for row in k2_native)
    require(len(k2_native) == 9 and contacts == 9, "factor-25 pilot censorship drift")
    return {
        "scan_contract_canonical_sha256": canonical,
        "scan_contract_file_sha256": sha256_file(HERE / "qa/residual_length_pilot/scan_contract.json"),
        "selected_factor": 25,
        "k2_native_1pct_factor25_contacts": contacts,
    }


def accepted_lattice(rows: Sequence[Mapping[str, str]], model: str) -> set[tuple[Any, ...]]:
    return {
        (
            row["model"], row["scenario"], int(row["background_toy_index"]),
            int(round(float(row["mass_MeV"]))), float(row["inj_nsigma"]),
        )
        for row in rows
    }


def check_closure_and_comparison() -> Mapping[str, Any]:
    expected = {
        (model, scenario, toy, mass, injection)
        for model in MODELS for scenario in SCENARIOS for toy in range(20)
        for mass in MASSES for injection in INJECTIONS
    }
    contacts: dict[str, int] = {}
    for model in MODELS:
        directory = HERE / "derived/residual_closure" / model
        summary = load_json(directory / "collection_summary.json")
        require(summary["status"] == "pass", f"{model} collection status drift")
        require((summary["raw_rows"], summary["accepted_rows"], summary["excluded_rows"]) == (2000, 2000, 0), f"{model} counts drift")
        require(summary["summary_cells"] == 100 and summary["minimum_accepted_per_cell"] == 20, f"{model} cell drift")
        require(float(summary["selected_extraction_upper_factor"]) == 25.0, f"{model} extraction factor drift")
        require(float(summary["production_card_upper_factor"]) == 15.0, f"{model} production factor drift")
        rows = csv_rows(directory / "accepted_extraction_rows.csv")
        require(len(rows) == 2000 and accepted_lattice(rows, model) == {x for x in expected if x[0] == model}, f"{model} lattice drift")
        require(max(abs(float(row["pull_identity_residual"])) for row in rows) < 1e-12, f"{model} pull identity drift")
        contacts[model] = sum(truth(row["refit_upper_boundary"]) for row in rows)
        require(len(csv_rows(directory / "closure_summary.csv")) == 100, f"{model} summary cardinality drift")
        require(len(csv_rows(directory / "task_product_audit.csv")) == 100, f"{model} task audit drift")
        require(len(csv_rows(directory / "exclusion_ledger.csv")) == 0, f"{model} exclusions appeared")
    require(contacts == {"knot_spline": 345, "regional_blend": 0}, "closure boundary contacts drift")

    combined = HERE / "derived/residual_closure"
    cardinality = {
        "combined_closure_summary.csv": 200,
        "combined_zero_signal_bias_tests.csv": 50,
        "model_lane_closure_summary.csv": 10,
        "paired_baseline_subtracted_response_rows.csv": 3000,
        "paired_baseline_subtracted_response_summary.csv": 150,
    }
    for name, expected_rows in cardinality.items():
        require(len(csv_rows(combined / name)) == expected_rows, f"{name} cardinality drift")
    bias = csv_rows(combined / "combined_zero_signal_bias_tests.csv")
    flags = {
        model: sum(truth(row["exploratory_material_bias_flag"]) for row in bias if row["model"] == model)
        for model in MODELS
    }
    require(flags == {"knot_spline": 3, "regional_blend": 16}, "zero-bias flag drift")
    lanes = csv_rows(combined / "model_lane_closure_summary.csv")
    require(all(not truth(row["strict_generator_qualified"]) for row in lanes), "qualified lane appeared")
    require(all(not truth(row["signal_rigidity_passed"]) for row in lanes), "rigid lane appeared")

    manifest = load_json(HERE / "figures/closure_comparison_figure_manifest.json")
    require(manifest["status"] == "pass_reporting_integrity_only", "comparison status drift")
    require(manifest["scientific_disposition"] == "fail_requested_conditional_stress_only", "comparison disposition drift")
    for name, expected_hash in manifest["output_sha256"].items():
        root = HERE / "figures" if name.endswith((".pdf", ".png")) else combined
        require(sha256_file(root / name) == expected_hash, f"comparison output hash drift: {name}")
    return {"raw": 4000, "accepted": 4000, "excluded": 0, "boundary_contacts": contacts, "material_flags": flags}


def check_remediation() -> Mapping[str, Any]:
    base = HERE / "derived/residual_length_ceiling_remediation"
    selection = load_json(base / "selection/selection_disposition.json")
    confirmation = load_json(base / "confirmation/final_disposition.json")
    require(selection["status"] == "selection_pass" and selection["selected_candidate"] == 50, "selection disposition drift")
    require(confirmation["status"] == "qualified_targeted", "confirmation disposition drift")
    require(confirmation["selected_candidate"] == 50 and confirmation["selected_sentinel"] == 75, "ceiling pair drift")
    require(confirmation["all_lane_qualification"] is False, "all-lane claim appeared")
    require(confirmation["closure_rerun_performed"] is False, "closure-rerun claim appeared")
    require(confirmation["inference_quantities_inspected"] is False, "inference entered remediation")
    require(sha256_file(base / "selection/selection_disposition.json") == confirmation["selection_disposition_sha256"], "selection chain drift")
    for stage, disposition, expected in (("selection", selection, (6, 36, 116, 0, 3)), ("confirmation", confirmation, (3, 60, 180, 0, 5))):
        directory = base / stage
        files = (
            "candidate_mass_gate.csv", "selected_trajectory_ledger.csv",
            "optimizer_attempt_ledger.csv", "optimizer_exclusion_ledger.csv", "task_product_audit.csv",
        )
        counts = tuple(len(csv_rows(directory / name)) for name in files)
        require(counts == expected, f"{stage} remediation cardinality drift: {counts}")
        for name, recorded in disposition["product_sha256"].items():
            require(sha256_file(directory / name) == recorded, f"{stage} product hash drift: {name}")
    gates = csv_rows(base / "confirmation/candidate_mass_gate.csv")
    require(all(truth(row["mass_gate_passed"]) for row in gates), "confirmation mass gate failed")
    require(all(int(row["contact_count_candidate_and_sentinel"]) == 0 for row in gates), "confirmation contact appeared")
    return {"selected_factor": 50, "sentinel": 75, "selection_toys": 3, "confirmation_toys": 5, "all_lane_qualification": False, "closure_rerun": False}


def check_figures() -> Mapping[str, Any]:
    hashes: dict[str, dict[str, str]] = {}
    for stem in REQUIRED_FIGURES:
        hashes[stem] = {}
        for suffix in ("pdf", "png"):
            path = HERE / "figures" / f"{stem}.{suffix}"
            require(path.is_file() and path.stat().st_size > 1000, f"missing/empty figure: {path.name}")
            hashes[stem][suffix] = sha256_file(path)
    return {"figure_stems": len(hashes), "files": 2 * len(hashes), "sha256": hashes}


def normalized_pdf_text(reader: PdfReader) -> str:
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # TeX fonts commonly expose presentation ligatures (for example, the
    # single Unicode character ``ﬁ``) through PDF text extraction.  Normalize
    # those before testing semantic release phrases.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\S)\d{4}(?=\n)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def write_note_source_manifest() -> Mapping[str, Any]:
    source = HERE / "note/build_source"
    excluded = {"main.pdf", "main.log", "main.blg"}
    rows = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = str(path.relative_to(source))
        if relative in excluded:
            continue
        suffix = path.suffix.lower()
        role = "tex" if suffix == ".tex" else "bibliography" if suffix == ".bib" else "figure_or_source_asset"
        rows.append({"path": relative, "role": role, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_csv(HERE / "note/source_manifest.csv", rows)
    require(len(rows) >= 205, "note source inventory unexpectedly small")
    return {"entries": len(rows), "manifest_sha256": sha256_file(HERE / "note/source_manifest.csv")}


def check_note() -> Mapping[str, Any]:
    require(sha256_file(NOTE_PDF) == sha256_file(OUTPUT_PDF), "stable PDF copies differ")
    reader = PdfReader(str(NOTE_PDF))
    require(len(reader.pages) == 221, "note page count drift")
    text = normalized_pdf_text(reader)
    phrases = (
        "version 4.8: residual-structured 2021 functional-form stress tests",
        "k2 fails source qualification",
        "regional blend also fails source qualification",
        "requested conditional stress only",
        "does not establish that the form cannot learn signal",
        "345 of 400 closure states contact that ceiling",
        "factor 50 against a factor-75 sentinel",
        "2000 raw and accepted states per model",
    )
    for phrase in phrases:
        require(phrase in text, f"note semantic phrase missing: {phrase}")
    for forbidden in ("figure placeholder", "missing figure", "??"):
        require(forbidden not in text, f"note contains forbidden marker: {forbidden}")
    log = (HERE / "note/tectonic_main.log").read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"Overfull \\hbox", r"Overfull \\vbox", r"undefined references",
        r"undefined citations", r"Undefined control sequence", r"Emergency stop", r"Fatal error",
    ):
        require(re.search(pattern, log, flags=re.IGNORECASE) is None, f"build-log failure: {pattern}")
    visual = load_json(HERE / "qa/manual_visual_qa.json")
    require(visual["result"] == "pass", "manual visual QA did not pass")
    require(visual["pdf_sha256"] == sha256_file(NOTE_PDF), "visual-QA PDF hash drift")
    require(visual["page_count"] == len(reader.pages), "visual-QA page count drift")
    section5 = (HERE / "note/source_overlays/sections/05_toys_validation.tex").read_text(encoding="utf-8")
    require(section5.count("\\input{sections/subsection_v4p8p3_residual_truths}") == 1, "Section 5 insertion drift")
    for relative in (
        "main.tex", "sections/00_change_log.tex", "sections/05_toys_validation.tex",
        "sections/07_conclusions.tex", "sections/subsection_v4p8p3_residual_truths.tex",
    ):
        require(sha256_file(HERE / "note/build_source" / relative) == sha256_file(HERE / "note/source_overlays" / relative), f"build/overlay drift: {relative}")
    source_manifest = write_note_source_manifest()
    return {"pages": len(reader.pages), "pdf_sha256": sha256_file(NOTE_PDF), **source_manifest}


def build_release_inventory() -> dict[str, Any]:
    paths: set[Path] = set()
    for relative in EXPECTED_SHA256:
        paths.add(HERE / relative)
    for directory in ("derived", "figures", "inputs", "qa"):
        paths.update(path for path in (HERE / directory).rglob("*") if path.is_file())
    paths.update(
        path for path in HERE.glob("*.py") if path.is_file()
    )
    paths.update((HERE / "README.md", HERE / "CEILING_REMEDIATION_ADDENDUM.json", HERE / "MODEL_PROTOCOL.json", HERE / "note/source_manifest.csv"))
    excluded = {HERE / "qa/release_validation.json", HERE / "RELEASE_MANIFEST.json"}
    rows = []
    for path in sorted(paths - excluded):
        require(path.is_file(), f"release inventory target missing: {path}")
        rows.append({
            "path": str(path.relative_to(HERE)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_utc": utc_now(),
        "base_commit": "e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6",
        "artifact_count": len(rows),
        "artifacts": rows,
        "note": {"path": f"note/{PDF_NAME}", "pages": 221, "sha256": sha256_file(NOTE_PDF)},
        "closure": {"models": 2, "lanes_per_model": 5, "backgrounds_per_lane_model": 20, "raw_rows": 4000, "accepted_rows": 4000, "excluded_rows": 0},
        "scientific_disposition": "fail_requested_conditional_stress_only",
        "ceiling_disposition": "factor25_closure_censored_factor50_targeted_only",
        "claim_boundary": "Neither source model is qualified or signal-rigid. The closure is a finite requested conditional stress, not coverage, expected bands, limits, exclusions, observed-data bias, or a production-card change.",
    }


def main() -> int:
    audit = Audit()
    audit.check("provenance.hash_locks", check_hash_locks)
    audit.check("components.read_only_validators", run_component_validators)
    audit.check("science.source_and_influence", check_source_and_influence)
    audit.check("pilot.original_ceiling", check_original_pilot)
    audit.check("closure.inventory_and_comparison", check_closure_and_comparison)
    audit.check("pilot.targeted_remediation", check_remediation)
    audit.check("artifacts.figures", check_figures)
    audit.check("note.rendered_release", check_note)

    report: dict[str, Any] = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "validated_utc": utc_now(),
        "status": "fail" if audit.failed else "pass",
        "checks_passed": len(audit.checks) - audit.failed,
        "checks_failed": audit.failed,
        "checks": audit.checks,
    }
    if not audit.failed:
        manifest = build_release_inventory()
        atomic_json(HERE / "RELEASE_MANIFEST.json", manifest)
        report["release_manifest_sha256"] = sha256_file(HERE / "RELEASE_MANIFEST.json")
        report["artifact_count"] = manifest["artifact_count"]
    atomic_json(HERE / "qa/release_validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
