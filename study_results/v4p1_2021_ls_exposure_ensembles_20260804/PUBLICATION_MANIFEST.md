# GitHub publication manifest

This directory is the review-facing, reproducible export of the v4.1
length-scale study. It includes:

- frozen candidate configurations, commands, and study specification;
- deterministic generators, optimizer-audit and repair code, postprocessing,
  and tests;
- the recovered 2016 10% histogram and paired 2021 toy ROOT input;
- reviewed scan and injection ledgers, manifests, summaries, and provenance;
- the unchanged-card 2016 10% repeat outputs used in the observed comparison;
- all note-facing PDF and PNG plots; and
- the analysis-note patches and build-QA record.

The bulk `runs/scan/` and `runs/injection/` directories are deliberately
excluded from GitHub. They contain 40,808 files (about 166 MB) and 12,600
files (about 96 MB), respectively. Their reviewed numerical content is
consolidated by the committed ledgers, integrity manifests, summaries, and
plots. The `runs/scan_repairs/` subtree is included because it contains the
selected unchanged-card repair configurations and their hash-audited
provenance. These packaging choices do not change any quoted result.

The frozen post-processing manifest retains the production script hash and
absolute paths recorded at execution time. For portable review, the exported
`postprocess_ensemble.py` adds a prefix-relocation shim that re-roots only
paths whose suffix begins at the exact
`study_results/v4p1_2021_ls_exposure_ensembles_20260804` boundary. Numerical
and statistical logic is unchanged. The exported script has SHA-256
`a28116b84240bddec0661d1849735cbc98e995068c2f36ec44fa0b04c6eeb8a0`; the
frozen production hash remains recorded in
`derived/v4p1_ensemble_postprocess_manifest.json`. Portable validation passes
all 39 committed tests.

The duplicate study-local PDF export is also omitted. The canonical compiled
note is
`../../hps_gpr_analysis_note/HPS_GPR_Analysis_Note_v4p1_20260804.pdf`.

This export contains no expected-limit bands. The ten-toy products remain
screening diagnostics and do not establish coverage.
