#!/usr/bin/env python3
"""Build the deterministic SHA-256 inventory for the terminal v4.9.7 release.

The release is a scientifically complete *blocked-state* result: phase one ran,
no candidate support passed, and every downstream production product is absent.
This inventory therefore describes evidence, lineage, unexecuted scaffolding,
and QA without treating scaffolding as a physics result.

``release_manifest.json`` cannot hash itself.  ``release_validation.json`` is
also excluded because it is the mutable report produced while validating the
otherwise immutable inventory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUTPUT = HERE / "release_manifest.json"
VALIDATION_REPORT = HERE / "release_validation.json"
EXCLUDED = {OUTPUT.name, VALIDATION_REPORT.name}
STUDY_ID = "v4p9p7_2016_support_combined_100toy_20260902"
EXTERNAL_NOTE_MIRROR = REPO_ROOT / "output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf"

CANONICAL_TOY_RE = re.compile(
    r"^runs/2016_threshold_qualified_(?:028|029|030|031|032|033|034)_210/"
    r"2016_full/toy_(?:00(?:0[0-9]|1[0-9]|2[0-4]))/"
)

SCAFFOLD_FILES = {
    "OBSERVED_2016_WORKFLOW.md",
    "README_COMBINED_SCAFFOLD.md",
    "assemble_reviewed_state_ledger.py",
    "benchmark_cached_profile_closure.py",
    "build_observed_2016_card.py",
    "cached_profile_solver.py",
    "combined_scaffold_manifest.json",
    "make_combined_card.py",
    "observed_2016_contract.py",
    "review_observed_2016.py",
    "run_combined_bands_cached_fixed_reviewed.py",
    "run_observed_2016_cli.py",
    "runtime_guard.py",
    "validate_combined_release.py",
    "validate_observed_2016.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_for(relative: str) -> tuple[str, str]:
    """Return one of the four release roles and a more precise description."""

    if ".superseded_" in relative:
        return "lineage", "superseded_phase1_task_preserved_for_audit"
    if CANONICAL_TOY_RE.match(relative):
        return "canonical", "canonical_phase1_task_evidence"
    if relative in {
        "inputs/2016_threshold_qualified_background_toys_100.root",
        "inputs/2016_threshold_qualified_background_toys_100.manifest.json",
    }:
        return "canonical", "frozen_conditional_truth_product"
    if relative.startswith("inputs/"):
        return "lineage", "hash_pinned_analysis_input"
    if relative == "reference/2016_threshold_truth_fit_summary.json":
        return "canonical", "frozen_conditional_truth_fit_record"
    if relative.startswith("reference/"):
        return "lineage", "frozen_truth_fit_reference"
    if relative.startswith("runtime_overlay/"):
        return "canonical", "support_scan_runtime_snapshot"
    if relative.startswith("runtime_combined/"):
        return "scaffold", "unexecuted_combined_runtime_snapshot"
    if relative.startswith("signal_audit/source_snapshots/"):
        return "lineage", "archived_signal_audit_source"
    if relative.startswith("signal_audit/qa/"):
        return "qa", "signal_robustness_qa"
    if relative.startswith("signal_audit/"):
        return "canonical", "canonical_signal_robustness_diagnostic"
    if relative.startswith("note/qa/base_build/"):
        return "lineage", "inherited_v4p9p6_note_build_baseline"
    if relative.startswith("note/qa/"):
        return "qa", "analysis_note_render_or_semantic_qa"
    if relative.startswith("note/build/"):
        return "qa", "analysis_note_build_intermediate"
    if relative.startswith("note/source/archive/"):
        return "lineage", "historical_note_source_archive"
    if relative.startswith("note/"):
        return "canonical", "v4p9p7_analysis_note"
    if relative == "output/pdf/HPS_GPR_Analysis_Note_v4p9p7.pdf":
        return "canonical", "analysis_note_release_mirror"
    if relative.startswith("qa/"):
        return "qa", "campaign_qa"
    if relative in {"build_release_manifest.py", "validate_release.py"}:
        return "qa", "release_packaging_and_validation_code"
    if relative == "audit/phase1_selection_audit.failed_path_reporting_20260902.json":
        return "lineage", "superseded_failed_audit_preserved_for_provenance"
    if relative.startswith("audit/"):
        return "canonical", "independent_frozen_state_audit"
    if relative in SCAFFOLD_FILES or relative.startswith(("combined/", "observed_2016/")):
        return "scaffold", "unexecuted_downstream_scaffolding"
    if relative == "observed_2016_workflow_manifest.json":
        return "canonical", "canonical_observed_blocked_state"
    if relative in {
        "generate_2016_full_background_toys.C",
        "figures/source_2016_10pct_functional_truth_x10.pdf",
        "figures/source_2016_10pct_functional_truth_x10.png",
    }:
        return "lineage", "superseded_truth_development_lineage"
    if relative.startswith("derived/") or relative.startswith("figures/"):
        return "canonical", "canonical_phase1_result_evidence"
    if relative in {"STUDY_PROTOCOL.md", "SCIENTIFIC_SCOPE_CLARIFICATION.md", "study_spec.json"}:
        return "canonical", "frozen_protocol_or_scientific_scope"
    return "canonical", "release_source_or_documentation"


def regular_files() -> list[Path]:
    files: list[Path] = []
    for root, directories, names in os.walk(HERE, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *names]:
            path = root_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Symlink is forbidden in release tree: {path.relative_to(HERE)}")
        for name in names:
            path = root_path / name
            relative = path.relative_to(HERE).as_posix()
            if relative in EXCLUDED:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise RuntimeError(f"Non-regular release object: {relative}")
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(HERE).as_posix())


def stable_tree_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def external_mirror_record() -> dict[str, Any]:
    path = EXTERNAL_NOTE_MIRROR
    record: dict[str, Any] = {
        "path_from_repository_root": path.relative_to(REPO_ROOT).as_posix(),
        "present": path.is_file() and not path.is_symlink(),
    }
    if record["present"]:
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def build_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in regular_files():
        relative = path.relative_to(HERE).as_posix()
        role, detail = role_for(relative)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role,
                "role_detail": detail,
            }
        )

    role_counts: Counter[str] = Counter()
    role_bytes: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        role_counts[entry["role"]] += 1
        role_bytes[entry["role"]] += int(entry["bytes"])

    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "release_state": "terminal_phase1_failure_downstream_production_blocked",
        "production_authorized": False,
        "inventory_root": ".",
        "hash_algorithm": "sha256",
        "excluded_mutable_or_self_referential_paths": sorted(EXCLUDED),
        "allowed_roles": ["canonical", "lineage", "scaffold", "qa"],
        "role_semantics": {
            "canonical": "frozen v4.9.7 scientific evidence and terminal conclusions",
            "lineage": "hash-pinned inputs or preserved historical/superseded provenance",
            "scaffold": "unexecuted downstream code/configuration; never a result",
            "qa": "build, render, packaging, or semantic validation evidence",
        },
        "summary": {
            "regular_file_count": len(entries),
            "regular_file_bytes": sum(int(entry["bytes"]) for entry in entries),
            "file_count_by_role": dict(sorted(role_counts.items())),
            "bytes_by_role": dict(sorted(role_bytes.items())),
        },
        "tree_sha256": stable_tree_sha256(entries),
        "external_mirrors": {
            "analysis_note_repository_mirror": external_mirror_record()
        },
        "files": entries,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    try:
        manifest = build_manifest()
        atomic_write_json(OUTPUT, manifest)
    except Exception as error:  # fail closed with a concise CLI diagnostic
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"Wrote {OUTPUT} with {manifest['summary']['regular_file_count']} files; "
        f"tree_sha256={manifest['tree_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
