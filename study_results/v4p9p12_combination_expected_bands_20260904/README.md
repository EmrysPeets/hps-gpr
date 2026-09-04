# v4.9.12 combination expected bands

This continuation adds pointwise background-only expected-limit bands to the
seven final-dataset scopes shown in the latest Harvard results section.  The
observed curves and frozen GP states come from
`study_results/v4p9p12_final_dataset_combinations_20260902/`.

The first release used 50 toys per mass. Toy IDs are cumulative and stable, so
the 100- and 300-toy stages append rather than replace work:

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
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_figures.py --target-toys 100
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_note_assets.py --target-toys 100
python3 study_results/v4p9p12_combination_expected_bands_20260904/validate_release.py --target-toys 100
```

`make_figures.py` also writes the machine-readable fixed-mass diagnostics
`p_strong`, `p_weak`, `p_two`, and the frozen observed analytic local `p0`. The
first three locate the observed upper limit within the background-only
toy-limit ensemble; they are not discovery p-values. The analytic `p0` is the
separate one-sided fixed-mass asymptotic profile-LRT result and is not
look-elsewhere corrected.

The standalone total-window curve follows a fixed maximal-available-dataset
rule: 2015 over 19--38 MeV, 2015+2016 over 39--49 MeV, all three datasets over
50--90 MeV, 2016+2021 over 91--180 MeV, and 2021 over 181--250 MeV. The
composition boundaries are shown explicitly; no observed or expected limit is
used to choose a segment.

The band construction is frozen in `STATISTICAL_PROTOCOL.md`.  The documented
pre-production numerical amendment in
`NUMERICAL_AMENDMENT_PRE_PRODUCTION.md` replaces only the fixed-strength
optimizer objective by its data-constant-centered form; it does not change the
likelihood or statistic. During the 100-toy continuation, the fail-closed raw
free-strength optimizer status at one deterministic coordinate motivated the
separately documented `NUMERICAL_AMENDMENT_100TOY_CONTINUATION.md`. Its centered,
amplitude-scaled free-profile retry is activated only after an inherited fit
fails, retains the same likelihood, and must pass all original nesting and root
gates. The 100-toy grid is recomputed under the amended contract rather than
mixing checkpoints. In particular, the ribbons are pointwise conditional bands
with fixed GP states. They are not scan-global calibration or unconditional
coverage.
