# v4.9.13: observed fixed-GP comparison

Completed the requested full-2015, full-2016, 2021-10%, and all-three fixed-GP
observed study. All outputs are isolated here. The production GP states,
histograms, results, validations, and existing analysis notes were read only.

The fixed model treats the estimated GP mean as exactly known. Its narrow
limits and very small local asymptotic p-values omit the GP estimation
uncertainty. They are conditional diagnostics, not calibrated significance,
discovery evidence, or demonstrated sensitivity gains. The parent v4.9.13 note
contains the independent injection and extraction investigation.

## Outputs

- `figures/observed_limits_four_scopes.pdf` and `.png`: observed 90% bounded
  piecewise-asymptotic CLs limits, with fixed/released ratio strips.
- `figures/local_asymptotic_pvalues_four_scopes.pdf` and `.png`: one-sided local
  asymptotic discovery p0 for the release and the known-GP-mean assumption.
- `figures/conditional_asimov_limits_four_scopes.pdf` and `.png`: deterministic
  background-only GP-mean Asimov limits from fixed and Gaussian-profile models,
  computed with the same stable solver; no expected-toy bands.
- `derived/observed_fixed_comparison.csv`: 456 scope/mass rows with raw and
  dimuon-corrected limits, signed fixed fits, p0 and log p0, normalization,
  inherited status, conditional Asimov values, and prior-2021 closure.
- `derived/local_p0_minima.csv`: eight local minima (four scopes, two methods).
- `derived/scope_summary.csv`: per-scope ratio summaries.
- `derived/prediction_verification.csv`: 415 exact released-prediction matches.
- `derived/independent_fixed_checks.csv`, `fit_diagnostics.csv`, `summary.json`,
  `validation.json`, and `figure_manifest.json`: numerical checks and provenance.

## Limit comparison

Ratios are fixed divided by the exact released observed limit. The Asimov
ratios use the same new solver for both background treatments, at the identical
frozen GP mean; these are not calibrated sensitivity ratios.

| Scope | Masses (MeV) | Rows | Observed ratio min / median / max | Fixed smaller | Conditional Asimov ratio median |
| --- | --- | ---: | --- | ---: | ---: |
| 2015 full | 19--90 | 72 | 0.102 / 0.422 / 2.234 | 60 / 72 | 0.412 |
| 2016 full | 39--180 | 142 | 0.141 / 0.522 / 1.653 | 114 / 142 | 0.523 |
| 2021 10% | 50--250 | 201 | 0.137 / 0.607 / 1.373 | 180 / 201 | 0.613 |
| All three | 50--90 | 41 | 0.146 / 0.595 / 1.361 | 34 / 41 | 0.524 |

Fixing the mean changes the signal estimate as well as the uncertainty, so the
observed limit can also become weaker. The Asimov calculation removes this
observed-fluctuation effect but still presumes that the selected GP mean and
constraint model are correct.

## Local asymptotic p0 minima

The fixed columns below assume that the GP mean is exactly known. They are
not calibrated significances. No trials factor or additional scale factor has
been applied. Minima from scanning correlated mass hypotheses are not global
p-values, and the two methods need not minimize at the same mass.

| Scope | Released mass / p0 / Z | Fixed mass / p0 / Z |
| --- | --- | --- |
| 2015 full | 51 MeV / 8.46e-4 / 3.139 | 51 MeV / 5.48e-33 / 11.906 |
| 2016 full | 90 MeV / 3.09e-4 / 3.424 | 88 MeV / 1.43e-17 / 8.453 |
| 2021 10% | 78 MeV / 2.47e-3 / 2.810 | 77 MeV / 2.70e-17 / 8.378 |
| All three | 66 MeV / 2.84e-3 / 2.765 | 50 MeV / 1.27e-14 / 7.620 |

## Methods and verification

`PROTOCOL.md` was fixed before execution. The frozen production runtime rebuilt
the native ROOT predictions without reoptimizing any GP state. All 415
prediction hashes match v4.9.12 exactly; all three native histogram hashes,
reviewed coordinates, source ledgers, and the declared exception were checked.
The 2016 state retains `conditional_user_accepted_numerical_exception`; it has
not gained independent certification, and the all-three result inherits that
limitation.

The combined Poisson fit concatenates each dataset's counts, GP means, and
signal counts per unit epsilon squared, and fits one shared coupling. Dataset
resolution, radiative normalization, and signal-window fraction remain encoded
bin by bin. Neither limits nor p-values are combined algebraically.

All 456 observed fixed limits passed independent scalar-score MLE and scalar
likelihood-ratio CLs checks. The largest independent relative limit difference
is 1.53e-9; the largest signed-root difference is 1.92e-11. All 1,368 newly
evaluated limits (456 observed fixed and 912 paired Asimov) passed positive
expectation, likelihood nesting, CLs-root, and monotonicity checks. The exact
released limit and p0 columns were retained as references. Recomputing the 201
previous 2021 fixed limits closes within 8.09e-14 relative, with signed-root
agreement within 9.43e-13.

Every displayed epsilon-squared curve receives the same dimuon branching
correction used by the expanded v4.9.12 note, exactly once. Raw electron-channel
values remain alongside them in the CSV. The factor at 250 MeV is
1.7252323083862526. All p-values are unchanged by this display conversion.

All three figures use serif/math fonts and legends outside their data axes.
Their PNG counterparts are visually checked for labels, clipping, and scale;
PDF and PNG outputs come from the same figures. Numerical validation is saved
in `derived/validation.json`.

## Reproduction

From the repository root, run the following sequentially:

```bash
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/observed/run_observed.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/observed/make_figures.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/observed/validate_observed.py
```

All libraries are restricted to one numerical thread. The observed/Asimov
scan completed in about 16 seconds. This sub-study is complete; no additional
numerical studies are required for these plots. Calibration conclusions belong
to the separately owned injection and extraction study in the parent note.
