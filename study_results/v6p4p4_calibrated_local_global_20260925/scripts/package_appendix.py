"""Package the validated report, tables and complete reproducible study."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

B = Path(__file__).resolve().parents[1]
ROOT = B.parents[1]
OUT = ROOT / "output/pdf" / B.name
STEM = "HPS_GPR_v6p4p4_Calibrated_Local_Global"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ("calibrated_validation", "common_coupling_audit",
                 "portable_appendix_rebuild", "calibrated_pdf_validation"):
        record = json.loads((B / "qa" / (name + ".json")).read_text())
        assert record["passed"], name
    pdfqa = json.loads((B / "qa/calibrated_pdf_validation.json").read_text())
    assert pdfqa.get("visual_review_passed") is True
    previous = json.loads((B / "provenance/previous_report_hashes.json").read_text())
    parent_pdf = ROOT / "output/pdf/v6p4p3_global_mc_20260925/HPS_GPR_v6p4p3_Global_Significances.pdf"
    previous[str(parent_pdf)] = sha(B / "provenance/parent_v643_report.pdf")
    for p, expected in previous.items():
        assert sha(Path(p)) == expected, p
    originals = json.loads((B / "provenance/parent_v643_hashes.json").read_text())
    parent = B.parent / "v6p4p3_global_mc_20260925"
    for p, expected in originals.items():
        assert sha(parent / p) == expected, p
    (B / "qa/previous_artifacts_unchanged.json").write_text(json.dumps(dict(
        passed=True, previous_report_count=len(previous),
        parent_study_files=len(originals), report_hashes=previous), indent=2) + "\n")
    files = [p for p in sorted(B.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts and p.name != ".DS_Store"
             and p != B / "MANIFEST.sha256"]
    (B / "MANIFEST.sha256").write_text("".join(
        sha(p) + "  " + p.relative_to(B).as_posix() + "\n" for p in files))
    files.append(B / "MANIFEST.sha256")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(B / "pdf/report.pdf", OUT / (STEM + ".pdf"))
    for name in ("calibrated_summary.csv", "calibrated_curves.csv", "threshold_comparison.csv"):
        shutil.copy2(B / "results" / name, OUT / name)
    archive = OUT / (STEM + "_Source.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, (Path(B.name) / p.relative_to(B)).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in (B / "MANIFEST.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256(z.read(B.name + "/" + name)).hexdigest() == expected, name
    deliveries = [OUT / (STEM + ".pdf"), archive] + [OUT / name for name in
                  ("calibrated_summary.csv", "calibrated_curves.csv", "threshold_comparison.csv")]
    (OUT / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in deliveries))
    print(json.dumps(dict(output=str(OUT), archive_files=len(files),
                         pdf_sha256=sha(deliveries[0]), zip_sha256=sha(archive),
                         prior_reports_unchanged=len(previous),
                         parent_study_files_unchanged=len(originals)), indent=2))


if __name__ == "__main__":
    main()
