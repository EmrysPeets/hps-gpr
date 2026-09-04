# v4.9.7 analysis-note source

This self-contained source tree extends the frozen v4.9.6 editorial source with:

- the halted full-2016 GP-support qualification;
- the resulting fail-closed downstream recombination gate; and
- the post-unblinding audit of the earlier 2021 65 MeV feature.

No v4.9.7 full-2016 observed scan or combined upper limit is present. Every eligible
2016 edge failed the frozen Phase-1 rule, which required the study to stop without
retuning. The v4.2 three-campaign result and v4.9.5 2021-only result are retained as
distinct historical states.

## Build the complete note

Run Tectonic from this directory so every dependency resolves inside the source tree:

```bash
mkdir -p ../build/full
tectonic -C --keep-logs --outdir ../build/full main.tex
cp ../build/full/main.pdf ../HPS_GPR_Analysis_Note_v4p9p7.pdf
```

The canonical output is `../HPS_GPR_Analysis_Note_v4p9p7.pdf`. The `-C` option uses
only the local Tectonic cache.

`writing_sample.tex` is an inherited, separate fellowship artifact from v4.9.6. It is
not a v4.9.7 deliverable and is deliberately unchanged.

## Main-text sequence

- `sections/05_toys_validation.tex`: historical core validation.
- `sections/05a_2021_support_selection.tex`: v4.9.5 2021 support selection.
- `sections/05b_2016_support_selection.tex`: v4.9.7 Phase-1 failure and analytic-mean
  diagnosis.
- `sections/05c_2021_signal_robustness.tex`: 65 MeV support-prescription audit.
- `sections/06_results.tex`: historical v4.2 results.
- `sections/06a_2021_observed_result.tex`: v4.9.5 2021-only result.
- `sections/06b_v4p9p7_combined_result.tex`: enforced absence of v4.9.7 recombination.
- `sections/07_conclusions.tex`: current interpretation and next-decision boundary.

All new figures are copied under `v4p9p7_figs/`; `note/source/` has no parent-directory
graphics dependency.

## Interpretation boundary

The 2016 truth is a conditional stress construction informed by a pre-existing 10%
development sample/subset and one scalar full-2016 normalization. Its broad-tail
component has an explicitly waived nonconverged ROOT fit status. The failed support
test is not coverage or an observed-data result. The 2021 audit establishes
support-prescription dependence of a fitted narrow-signal attribution, not whether the
feature is signal or background.
