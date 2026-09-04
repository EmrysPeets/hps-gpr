# v4.9.6 build and rendered-page QA

Date: 2026-09-02 (America/Los_Angeles)

## Frozen base

The editable tree was copied from
`study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/note/build_source/`.
No source or result file in that v4.9.5 bundle was edited.

- v4.9.5 `PROVENANCE.md`: `6bdb27a621a1c05bf1417d287a9348ad71551cf1cf7d9185df731f24d5666636`
- v4.9.5 source driver `main.tex`: `29b5eab9b4a56fe86595c6c8b28e2388e78a95b4c404732807400974f46eea60`
- v4.9.5 canonical PDF: `f32df114de636ef39059ac1eeba2bbc491243cde095db960afcab2f0a07956b7`

Version 4.9.6 is editorial only. It changes no data input, analysis card, fit, figure,
or numerical result.

## Builds

Both documents were built from `source/` with Tectonic 0.15.0 in cached mode:

```bash
tectonic -C --keep-logs --outdir ../qa/build_full main.tex
tectonic -C --keep-logs --outdir ../qa/build_writing_sample writing_sample.tex
```

| Document | Pages | SHA-256 |
|---|---:|---|
| Complete v4.9.6 note | 229 | `c1be68a9a7c251490989dba9001aa0f1dc596202ed0e0b2797ba0717ea2957e9` |
| Harvard writing sample, Sections 2--5 | 60 | `3341f1dd3957546ced51620e3593db65ed455cb74e0d4c51fd9ff1e69a7ebafd` |

The final TeX logs contain no undefined references or citations, duplicate
destinations, missing files, overfull boxes, or fatal errors. The complete note retains
three harmless underfull-box warnings (one short Introduction subfigure caption and two
lines in an inherited appendix) and a `lineno` engine-compatibility warning. Tectonic's
driver also repeats an internal bibliography-change warning, but the final TeX logs and
extracted citations are complete.

## Semantic and typography checks

Both PDFs were opened with `pypdf` 6.12.2. The extracted text contains no unresolved
`??`, placeholder caption, or Unicode replacement character. The complete note contains
the v4.9.6 release marker, the 2021 support-selection subsection, the corresponding
observed-scan subsection, and the Conclusions section.

The writing sample contains Sections 2, 3, 4, and 5 and its own bibliography. A
case-insensitive search of both the excerpt source and extracted PDF text finds zero
instances of the prohibited process term. The excerpt source also contains no
typewriter, path, verbatim, sans-serif, small-caps, or local font-family switches.
PyMuPDF 1.26.5 finds no monospace text span in the final PDF; Latin Modern Roman is the
dominant text font. Different fonts embedded inside scientific figure assets remain
confined to those figures.

## Rendered-page inspection

Poppler rendered all 229 pages of the complete note at 72 dpi and all 60 pages of the
writing sample at 110 dpi. Every rendered page was inspected through contact sheets in
`qa/contact_sheets/full/` and `qa/contact_sheets/sample/`. Full-resolution checks also
covered the two title pages; the two-page event-selection table; the methodology
transition; the Section 5 summary; the 2021 support-selection table and figure; the
complete-note Conclusions transition; and both bibliography endings.

No clipped text, overlapping elements, missing graphics, blank pages, or unintended
font changes were found. The writing-sample title page is unnumbered; its contents page
begins at page 1, and the excerpt preserves section numbers 2 through 5.
