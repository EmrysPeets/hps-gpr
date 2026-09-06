# Final QA: Harvard reader-facing writing sample

Date: 2026-09-02

Status: **PASS**

## Product

- Final PDF: `output/pdf/HPS_GPR_Harvard_Writing_Sample_Reader_Facing.pdf`
- SHA-256: `2e5a034f468384c27d0795e5ce4499c5f9f1f7c1bdc445d5097f8edfb5d80b1d`
- Format: 60 letter-size pages, PDF 1.5, not encrypted
- Metadata title: `HPS Gaussian-Process Resonance Search: Selected Sections 2–6`

The distributed PDF is byte-for-byte identical to both the final Tectonic build and
the copy retained inside this derivative.

## Editorial and semantic checks

- The rendered text contains no `Table-17` or `Table 17` reference.
- The rendered text contains no internal analysis-release labels (`v4.2`, `v4.5`,
  `v4.9`, `v4.9.1`, `v4.9.5`, or `v16`).
- Searches also returned zero occurrences of the principal workflow-like terms flagged
  by the cold-reader audit, including `four-lane`, `exposure lane`, `analysis state`,
  `production baseline`, `matched-refit contract`, `Full-100`,
  `source-conditioned`, and `Independent-confirmed`.
- Cross-references are resolved: the rendered text contains no `??`, and the build log
  contains no undefined-reference, overfull-box, underfull-box, missing-character, or
  missing-file diagnostic. The sole `not found` match is a benign package capability
  message for `pdfdraftmode`.
- Every page has extractable text. No blank pages were found.

## Scientific-preservation checks

- The three relabeled validation figures reproduce every source numeric table entry:
  24 rows x 15 numeric columns, 80 x 15, and 80 x 16. See
  `reader_facing_validation_figures/QA.md`.
- The three selected-results curve figures reproduce all 1,023 source-ledger rows and
  six plotted numerical columns exactly. See `reader_facing_selected_results/QA.md`.
- The support-comparison plot reproduces all numeric Matplotlib artists exactly:
  3 axes, 36 lines, 36 error-bar segments, and 72 path vertices. See
  `reader_facing_support_figure/verification.md`.
- Statistical qualifications remain explicit: the pseudoexperiments establish
  conditional extraction closure, not confidence-limit coverage or scan-global
  significance calibration; fixed-mass p-values remain local diagnostics.

## Visual inspection

All 60 pages were rendered with Poppler and reviewed in five contact sheets. The title
page, event-selection table, validation-model table, threshold study, spurious-signal
table, support-study transition, validation synthesis, and all four selected-results
figures were also inspected at full resolution. No clipped text, overlapping objects,
missing glyphs, unintended blank pages, or unreadable figure labels were found.

## Provenance boundary

This is a derivative editorial product. The accepted Sections 2–5 release at
`study_results/v4p9p6_analysis_note_editorial_20260902/` and the frozen selected-results
release at `study_results/v4p9p8_harvard_selected_results_20260902/` were not edited.
The accepted v4p9p6 PDF checksums still pass its recorded `SHA256SUMS.txt`.
