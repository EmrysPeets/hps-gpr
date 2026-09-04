# v4.9.12 combination expected bands

This continuation adds pointwise background-only expected-limit bands to the
seven final-dataset scopes shown in the latest Harvard results section.  The
observed curves and frozen GP states come from
`study_results/v4p9p12_final_dataset_combinations_20260902/`.

The initial release uses 50 toys per mass.  Toy IDs are cumulative and stable,
so later stages append rather than replace work:

```bash
# Initial stage (toy IDs 0--49)
python3 study_results/v4p9p12_combination_expected_bands_20260904/run_expected_bands.py --target-toys 50 --workers 2

# Later: append IDs 50--99
python3 study_results/v4p9p12_combination_expected_bands_20260904/run_expected_bands.py --target-toys 100 --workers 2

# Final: append IDs 100--299
python3 study_results/v4p9p12_combination_expected_bands_20260904/run_expected_bands.py --target-toys 300 --workers 2
```

The runner writes one atomic checkpoint per mass.  Reissuing the same command
is a no-op for complete checkpoints and resumes incomplete ones.  Use `--plan`
to inspect the amount of missing work without running toys.

After a completed stage, build figures and the compact results-only note with:

```bash
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_figures.py --target-toys 50
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_note_assets.py --target-toys 50
```

The band construction is frozen in `STATISTICAL_PROTOCOL.md`.  The documented
pre-production numerical amendment in
`NUMERICAL_AMENDMENT_PRE_PRODUCTION.md` replaces only the fixed-strength
optimizer objective by its data-constant-centered form; it does not change the
likelihood or statistic.  In particular, the ribbons are pointwise conditional
bands with fixed GP states.  They are not scan-global calibration or
unconditional coverage.
