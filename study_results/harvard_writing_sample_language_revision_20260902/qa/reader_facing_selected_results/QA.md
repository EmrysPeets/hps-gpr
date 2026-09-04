# Reader-facing selected-results figure QA

Status: **PASS**

The four Section 6 figures were rebuilt inside the language-revision derivative.
Only visible labels were changed.  The reviewed selected-results study remained
read-only.

## Numerical identity

- The three curve figures read all 1,023 rows from the reviewed
  `selected_result_curves.csv` ledger.
- The six plotted numerical columns compare exactly after canonical sorting:
  `mass_GeV`, `mass_MeV`, `A90_events`, `eps2_90`,
  `p0_local_asymptotic`, and `Z_local_asymptotic`.
- Canonical numerical-array SHA-256:
  `8beba2849b061637eb11860a790c7302632914e954cb264ed37ec1ae6032975c`.
- The 65 MeV extraction is drawn directly from the reviewed plot-data and fit-summary
  CSVs; their SHA-256 values are recorded in `figure_qa_manifest.json`.

## Language check

Text extracted with `pypdf` contains none of `historical`, `v4.`, `(current)`,
`Table-17`, or `Table 17`.
Reader-facing labels identify samples and configurations instead, including
`2016 full`, `2021 10% (36--300 MeV support)`, and
`Three-campaign shared-coupling scans`.

## Visual check

All four one-page PDFs were rendered at 140 dpi with Poppler and inspected.  Titles,
legends, axes, line styles, support labels, and the 65 MeV extraction panels are
legible, with no clipping, overlap, or missing glyphs.  Rendered PNGs are in
`rendered/`.

Machine-readable details and product hashes are in `figure_qa_manifest.json`.
