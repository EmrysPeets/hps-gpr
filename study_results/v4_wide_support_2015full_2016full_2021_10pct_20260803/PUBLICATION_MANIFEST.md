# Publication manifest

This directory is the review-facing archive of the version 4 wide-support
campaign. It is intentionally smaller than the original run directory while
preserving the frozen configuration, consolidated observed scans, reviewed
state ledger, 300-toy ensemble, derived tables, figure sources, and scripts
needed to reproduce the postprocessing.

## Included

- `README.md` and `CACHED_PROFILE_RUNNER.md`
- the campaign-local review, cache-closure, production, and postprocessing
  scripts
- the aborted 35 MeV-support explanation and validation report
- `results_single.csv`, `results_combined.csv`, and
  `validation_report.json` from each of the three unchanged-card observed
  attempts
- all reviewed and derived tables under `derived/`
- the final 232-row fixed-GP 300-toy table and provenance JSON under
  `combined_bands_300toy_cached/`
- all five PDF/PNG note-figure pairs under `note_figures/`

The repository also carries the frozen v4 configuration and the two reviewed
narrow-support comparator tables used by the cache benchmark and matched
support plot:

- `study_configs/v4_wide_support_2015full_2016full_2021_10pct_20260803/`
- `study_configs/finalist_k15_2021_10pct_combined100toy_20260803/`
- `study_results/finalist_k15_2021_10pct_combined100toy_20260803/derived/combined_bands100_reviewed.csv`
- `study_results/finalist_k15_2021_10pct_combined100toy_20260803/derived/combined_individual_reviewed.csv`

## Intentionally omitted

The 1,891 mass-by-mass `numbers.json` working files from the three observed
attempts and the stopped 35 MeV-support trial are not versioned. Their
information is flattened into the consolidated attempt CSVs, the complete
review ledger, and the validation summaries listed above. The duplicate
`combined.csv` aliases are also omitted in favor of `results_combined.csv`.

The proprietary ROOT inputs and the locally rendered 120-page review PDF are
not stored in Git. Input paths and hashes are recorded in `README.md`; the
editable note source and all referenced figures are versioned under
`hps_gpr_analysis_note/`.

## Reproduction boundary

The committed bundle is self-contained for review, validation of the recorded
tables, and deterministic regeneration of the derived summaries and figures.
A fresh GP fit or a new pseudoexperiment ensemble additionally requires the
three collaboration-controlled ROOT inputs whose SHA-256 values appear in the
campaign README.
