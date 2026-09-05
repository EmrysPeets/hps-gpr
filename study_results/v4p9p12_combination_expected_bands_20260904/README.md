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

# Final: append IDs 100--299 with four low-priority, single-threaded processes
python3 study_results/v4p9p12_combination_expected_bands_20260904/continue_balanced.py --target-toys 300 --workers 4
```

The runner writes one atomic checkpoint per mass.  Reissuing the same command
is a no-op for complete checkpoints and resumes incomplete ones.  Use `--plan`
to inspect the amount of missing work without running toys.

After a completed stage, build figures and the compact results-only note with:

```bash
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_figures.py --target-toys 300
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_global_diagnostics.py --target-toys 300
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_note_assets.py --target-toys 300
python3 study_results/v4p9p12_combination_expected_bands_20260904/pack_stage.py
python3 study_results/v4p9p12_combination_expected_bands_20260904/validate_release.py --target-toys 300
```

The large 300-toy limit CSV is stored in Git as a lossless `.csv.gz` archive.
The working CSV remains available locally. On a fresh checkout, restore its
exact release bytes before validation or plotting with:

```bash
python3 study_results/v4p9p12_combination_expected_bands_20260904/pack_stage.py --restore
```

`continue_balanced.py` dispatches disjoint eight-mass blocks in descending
scope-count cost, dynamically refilling at most four workers. Each worker runs
at nice level +10 with BLAS/OpenMP limited to one thread. `--workers 2` gives a
quieter continuation; `--plan` lists the blocks without running them. Completed
checkpoints are reused, and a failed block stops further dispatch while the
other active blocks finish. Per-block logs and an execution ledger record the
run. This wrapper does not alter the frozen runner or its contract.

`make_figures.py` also writes the machine-readable fixed-mass diagnostics
`p_strong`, `p_weak`, `p_two`, and the frozen observed analytic local `p0`. The
first three locate the observed upper limit within the background-only
toy-limit ensemble; they are not discovery p-values. The analytic `p0` is the
separate one-sided fixed-mass asymptotic profile-LRT result and is not
look-elsewhere corrected.

The 300-toy snapshot includes all individual and combined limits, individual
and combined p-value panels, and the standalone total-window limit. Explanatory
text and toy counts are in report captions; the plots contain axes, titles,
legends, and the total-window composition strip without explanatory footers.

`make_global_diagnostics.py` adds eight scan-level rows: the seven scopes and
the fixed total window. Its resolution-spacing Sidak equivalent and grid
Bonferroni trials correction use the analytic local `p0`, independently of the
number of band toys. The total-window values are `N_eff=35.381377775674345`,
`p_Sidak=0.0958859427902758`, and `p_Bonferroni=0.6600156524263839` for 232 masses.
The separate correction for selecting among all 680 scope-mass tests is
`p_Bonferroni=0.086086884555676`. These are different declared search families.
The resolution ledger and source-hash manifest reproduce the calculation;
`GLOBAL_PVALUE_METHOD.md` states its assumptions and references. A scan-toy
global p-value is unavailable because the band toys are independent between
masses. It is recorded as null, not inferred by joining equal toy IDs.

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
