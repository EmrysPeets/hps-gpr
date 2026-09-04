#!/usr/bin/env python3
"""Remove plot-embedded title/footer notes from the released peak display.

The source PDF is taken byte-for-byte from the frozen numerical release.  This
reader-facing derivative removes only the parenthetical title suffix and the
faint footer text; the plotted panels, axes, legend, and numerical content are
left unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parent
SOURCE = (
    DERIVATIVE.parent
    / "v4p9p12_final_dataset_combinations_20260902"
    / "figures"
    / "all_three_peak_extraction.pdf"
)
OUTPUT = DERIVATIVE / "figures" / "all_three_peak_extraction.pdf"
PNG = OUTPUT.with_suffix(".png")
QA = DERIVATIVE / "qa" / "reader_facing_peak_extraction"
EXPECTED_SOURCE_SHA256 = "367efafd72fadf9933147033d3a00664364c7e3d5fe1b49a3bb74ebd44c9061a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("released peak-extraction figure changed; refusing display-only edit")

    document = fitz.open(SOURCE)
    if len(document) != 1:
        raise RuntimeError("peak-extraction source is not a one-page figure")
    page = document[0]

    title_hits = page.search_for("not scan-corrected")
    footer_blocks = [
        fitz.Rect(block[:4])
        for block in page.get_text("blocks")
        if block[4].strip().startswith("Shared fit uses one common coupling")
    ]
    if len(title_hits) != 1 or len(footer_blocks) != 1:
        raise RuntimeError("could not identify the title suffix and footer exactly once")

    title = title_hits[0]
    title_redaction = fitz.Rect(title.x0 - 5.0, title.y0 - 1.0, title.x1 + 6.0, title.y1 + 1.0)
    footer = footer_blocks[0]
    footer_redaction = fitz.Rect(0.0, footer.y0 - 1.0, page.rect.width, page.rect.height)
    page.add_redact_annot(title_redaction, fill=(1, 1, 1))
    page.add_redact_annot(footer_redaction, fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    page.set_cropbox(fitz.Rect(0.0, 0.0, page.rect.width, footer.y0 - 1.5))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT, garbage=4, deflate=True)
    document.close()

    rendered = fitz.open(OUTPUT)
    pixmap = rendered[0].get_pixmap(matrix=fitz.Matrix(220 / 72, 220 / 72), alpha=False)
    pixmap.save(PNG)
    rendered.close()

    QA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "pass",
        "change_scope": "Removed the not-scan-corrected title suffix and the faint footer; plot content is unchanged.",
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "products": [
            {"path": str(OUTPUT.relative_to(DERIVATIVE)), "sha256": sha256(OUTPUT)},
            {"path": str(PNG.relative_to(DERIVATIVE)), "sha256": sha256(PNG)},
        ],
    }
    (QA / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
