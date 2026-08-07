#!/usr/bin/env python3
"""Cross-check the complete v4.2 follow-up bundle and note-local copies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
NOTE_ASSETS = (
    ROOT
    / "hps_gpr_analysis_note"
    / "final_limit_projection_figs"
    / "v4p2_followups_20260806"
)
NOTE_PDF = (
    ROOT
    / "output"
    / "pdf"
    / "v4p2_followups_20260806"
    / "HPS_GPR_Analysis_Note_v4p2_followups_20260806.pdf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    checks: dict[str, bool] = {}
    validators = {
        "babar": BUNDLE / "babar_projection" / "qa" / "validation_report.json",
        "m065": BUNDLE / "m065_extraction" / "validation.json",
        "pseudo65": BUNDLE / "pseudo65" / "derived" / "final_validation.json",
        "functional_mean": (
            BUNDLE
            / "pseudo65"
            / "functional_mean_shape_bias_20260806"
            / "qa"
            / "validation.json"
        ),
        "gp_ensemble": (
            BUNDLE
            / "pseudo65"
            / "gp_window_ensemble_20260806"
            / "derived"
            / "final_validation.json"
        ),
    }
    checks["babar_validator"] = load_json(validators["babar"])["status"] == "PASS"
    checks["m065_validator"] = load_json(validators["m065"])["status"] == "PASS"
    checks["pseudo65_validator"] = bool(load_json(validators["pseudo65"])["pass"])
    checks["functional_mean_validator"] = (
        load_json(validators["functional_mean"])["status"] == "PASS"
    )
    gp_validation = load_json(validators["gp_ensemble"])
    checks["gp_ensemble_validator"] = bool(gp_validation["pass"])
    gp_checks = {entry["name"]: entry for entry in gp_validation["checks"]}
    central_review = gp_checks["central_55_75_max_lml_review"]["detail"]
    checks["gp_ensemble_central_optimizer_closure"] = (
        central_review["row_count"] == 420
        and central_review["unreproduced_selected"] == 0
        and central_review["selected_bounds"] == 0
    )
    checks["gp_ensemble_no_expected_bands"] = bool(
        gp_checks["interpretation_flags"]["detail"]["no_expected_limit_bands"]
    )

    sources = {
        "v4p2_babar_observed_equivalent_projection_eps2":
            BUNDLE / "babar_projection" / "figures",
        "v4p2_babar_observed_equivalent_projection_eps2_with_projected_over_babar_ratio":
            BUNDLE / "babar_projection" / "figures",
        "v4p2_babar_observed_equivalent_projection_ratio":
            BUNDLE / "babar_projection" / "figures",
        "figure61_common_0p5MeV": BUNDLE / "m065_extraction" / "figures",
        "figure61_common_0p5MeV_profiled":
            BUNDLE / "m065_extraction" / "figures",
        "figure62_profiled_residuals_physical68":
            BUNDLE / "m065_extraction" / "figures",
        "figure62_coefficients_physical68":
            BUNDLE / "m065_extraction" / "figures",
        "pseudo65_central_window_zoom": BUNDLE / "pseudo65" / "plots",
        "pseudo65_observed_limit_p0_aligned": BUNDLE / "pseudo65" / "plots",
        "functional_mean_shape_bias_Ahat_p0": (
            BUNDLE
            / "pseudo65"
            / "functional_mean_shape_bias_20260806"
            / "plots"
        ),
        "gp_window_ensemble_central_spectra": (
            BUNDLE / "pseudo65" / "gp_window_ensemble_20260806" / "plots"
        ),
        "gp_window_ensemble_observed_limit_p0": (
            BUNDLE / "pseudo65" / "gp_window_ensemble_20260806" / "plots"
        ),
    }
    copy_ledger = []
    for stem, source_dir in sources.items():
        for suffix in (".pdf", ".png"):
            source = source_dir / f"{stem}{suffix}"
            note_copy = NOTE_ASSETS / source.name
            same = source.is_file() and note_copy.is_file()
            if same:
                same = sha256(source) == sha256(note_copy)
            checks[f"note_copy::{source.name}"] = same
            copy_ledger.append(
                {
                    "source": relative(source),
                    "note_copy": relative(note_copy),
                    "sha256": sha256(source) if source.is_file() else None,
                    "match": same,
                }
            )
            if suffix == ".pdf" and source.is_file():
                checks[f"one_page::{source.name}"] = len(PdfReader(source).pages) == 1

    pseudo_root = BUNDLE / "pseudo65" / "inputs" / "pseudo65_background_replacements.root"
    pseudo_provenance = load_json(
        BUNDLE / "pseudo65" / "derived" / "input_provenance.json"
    )
    checks["pseudo_root_hash"] = (
        pseudo_root.is_file()
        and sha256(pseudo_root) == pseudo_provenance["output"]["root_sha256"]
    )
    checks["pseudo_no_expected_bands"] = not load_json(
        validators["pseudo65"]
    )["expected_limit_bands_present"]

    note_info = None
    if NOTE_PDF.is_file():
        reader = PdfReader(NOTE_PDF)
        full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        checks["note_pdf_pages"] = len(reader.pages) >= 140
        checks["note_pdf_followup_text"] = (
            "Observed-equivalent" in full_text
            and "Version 4.2 Follow-up Studies" in full_text
        )
        checks["note_pdf_new_statistics_text"] = (
            "functional-form shoulder check" in full_text
            and "GP-mean replacement-window ensembles" in full_text
            and "2.25" in full_text
            and "2.5" in full_text
            and "3.0" in full_text
        )
        note_info = {
            "path": relative(NOTE_PDF),
            "pages": len(reader.pages),
            "sha256": sha256(NOTE_PDF),
        }
    else:
        checks["note_pdf_pages"] = False
        checks["note_pdf_followup_text"] = False
        checks["note_pdf_new_statistics_text"] = False

    passed = all(checks.values())
    report = {
        "schema_version": 2,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "note_copies": copy_ledger,
        "note_pdf": note_info,
    }
    output = BUNDLE / "validation_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
