# HPS GPR PRD Publication Folder

This standalone folder is intended for Overleaf import. The main document is
`pub_main.tex`.

## Build

Use a standard LaTeX workflow with BibTeX:

```bash
pdflatex pub_main
bibtex pub_main
pdflatex pub_main
pdflatex pub_main
```

The document uses REVTeX (`revtex4-2`) for Physical Review D style.

## Contents

- `pub_main.tex`: main PRD-style manuscript entry point.
- `sections/`: compact article sections plus an auxiliary-figure appendix.
- `tables/`: publication-scale dataset and selection/normalization tables.
- `figures/`: copied, note-local figure assets from `origin/main`.
- `pub_refs.bib`: bibliography copied from the analysis note, with PRD entries for the
  HPS 2015 and 2016 papers and a SIMP-paper entry added.
- `OUTLINE_AND_CLAIM_STATUS.md`: organizing outline, figure triage, and claim gates.

## Status

This is a publication skeleton and first prose pass based on the current
`origin/main` HPS GPR analysis note, anchored at commit
`2a0f3f7bef1dcd7d0c19bfc86b0650a8d4c317e4`. It deliberately keeps the central claim
conservative:

- The current combined result is the staged `2015 + 2016 10% + 2021 1%`
  shared-`epsilon^2` workflow.
- The full-statistics curves are projections, not observed exclusions.
- Final discovery/global-significance language still requires a frozen scan-level
  look-elsewhere toy calibration.
- Before journal submission, replace the draft author/acknowledgment block and freeze
  the final data samples, systematic model, radiative-fraction inputs, and final
  figure set.

Primary config references from the note:

- 95% staged baseline:
  `study_configs/config_2015_2016_10pct_2021_1pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k_corrected.yaml`
- 90% comparison suite:
  `study_configs/90pct_configs/`

Known freeze gates:

- Reconcile the 2021 resolution treatment and decide whether the 1.25x smearing study
  is nominal or systematic.
- Finalize the 2021 radiative fraction and systematic penalty model.
- Resolve the 2016 low-mass boundary convention before quoting a final scan range.
- Regenerate the final `CLs` and look-elsewhere calibration for the exact scan.
- Audit expected-band repairs/spikes in the 90% suite before using them as publication
  figures.
