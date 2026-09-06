# Continuing the calibration interpretation and global study

The completed 2015 pilot is conditional on one common archived smooth stress
spectrum, fixed observed kernel states, and the 19–90 MeV grid at 1 MeV spacing.
It is a separate statistic from the v4.9.13 two-truth local-p-value envelope.
Keep the current outputs frozen. Consult `global/2015/analysis/summary.json`
and the independent review before quoting probabilities.

## First priority: the 2016 interpolation mismatch

Use predeclared predictive controls to qualify the archived 2016 source and
its 75–85 MeV join. Compare this shape with independently justified smooth
alternatives. On identical complete spectra, compare fixed observed kernels
with a declared reestimation policy. Keep truth construction, kernel policy,
local probability calibration, and global correction as separate comparisons.
Do not discard the stress truth because its observed limit or p-value is
unfavorable. Its existing fit-quality waiver and lack of demonstrated
independence also do not make it a certified physical background.

Resolve the separate 2016 numerical exception and specify common systematics
before promoting the combined result. The existing combination envelopes only
all-GP and all-stress assignments; the six mixed assignments need either a
scientific justification for exclusion or an explicit extension.

## Other individual datasets: runnable ten-toy first steps

From the repository root, use a single numerical worker:

```bash
nice -n 10 python3 -B study_results/v4p9p14_interpretation_global_20260906/run_global.py --dataset 2016 --ensemble pilot10
nice -n 10 python3 -B study_results/v4p9p14_interpretation_global_20260906/run_global.py --dataset 2021 --ensemble pilot10
```

Each command produces ten *complete* scans: ten observations at every mass,
with exactly the same toy spectrum and ID across masses. Inspect its
`summary.json`, per-mass `_qa.json`, and any `_FAILURE.json`. No failure may be
silently removed. An unchanged-source resume uses the same output directory;
a changed source/count contract requires a new one.

If runtime and all numerical checks permit, replace `pilot10` by
`validation1000`, then run `--ensemble asimov`, sequentially for that dataset.
Process it with `analyze_global.py --dataset 2016` or `--dataset 2021`.
Validate the marginal centers/spreads, correlation matrices, and direct
maximum tails before interpreting the GP result. In particular, a large
stress-induced mean offset can make a conditional significance dramatic
without establishing a new-particle signal.

## Increasing precision without conflating uncertainties

The analysis samples 200,000 inexpensive GP fields. To study pure GP Monte
Carlo convergence, copy the `analysis/` products to a named snapshot and run
`analyze_global.py --gp-samples 2000000`. The random stream extends the same
sequence, so these are a larger nested sample, not an independent comparison.
This reduces sampling error for the assumed Gaussian field only; it does not
improve physical background qualification or validate extreme tails.

The frozen runner intentionally has fixed `pilot10`, `validation1000`, and
`asimov` contracts. A larger direct-Poisson validation is a new version: copy
`run_global.py`, add an explicitly named ensemble (for example
`validation10000_v2`) with a 10,000 count and a new seed label, record the
changed script/protocol hashes, and use a new output directory. Keep the
parent's 1,000 scans separate. Update the derivative analyzer's explicit
validation input and sample-size annotations together. Do not simply change
the output path while retaining the same seed/ensemble and call the toys new.
Target enough direct exceedances at the tail of interest; zero exceedances
must be reported as a binomial upper bound.

## A finer mass grid and a combined scan are new contracts

The saved 2 MeV subgrid comparison diagnoses grid dependence but does not
certify the 1 MeV grid against the continuum. A 0.5 or 0.25 MeV scan requires
an explicit kernel-state rule at intermediate masses, fresh observed fits,
identical masks/templates in observed and toy fits, and full-scan covariance
and tail checks on that new grid. Do not interpolate p-values and describe
that as a validated finer search.

For a combined scan, draw one independent full spectrum per dataset within
each joint truth scenario and fit the exact shared-coupling likelihood at
every mass. To globally calibrate the existing envelope, define one common
spectrum per constituent and scenario first; the current mass-dependent GP
truths cannot be spliced into one experiment. Then apply the *same complete
local decision rule* to every toy and observed scan. The maximum of raw roots,
the minimum of truth-specific local probabilities, and the minimum of an
envelope are different orderings. State which one is being calibrated.

## Build and inspect the report

```bash
python3 -B study_results/v4p9p14_interpretation_global_20260906/make_interpretation_figures.py
python3 -B study_results/v4p9p14_interpretation_global_20260906/build_report.py
```

Re-render every PDF page after changing prose, figures or tables. Regenerate
the final source/figure/PDF manifest only after semantic and visual checks.
