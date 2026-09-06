# v4.2 simultaneous observed 95% CLs limit

This observed-only derivative of the accepted v4.2 analysis changes only the
inner asymptotic CLs threshold from `cls_alpha: 0.10` to `0.05`. It uses the
same data, common `count_scale` signal-strength coordinate, signal templates,
mass resolutions, fixed reviewed GP states, and dataset coverage as v4.2.

No expected-limit bands, outer pseudoexperiments, GP refits, or scan-wide
discovery calibration are produced.

## Products

- `derived/combined_observed_95cl_reviewed_v4p2.csv`: 232-mass observed 95% CLs
  scan plus the unchanged shared-epsilon-squared local asymptotic `p0` values.
- `derived/validation_observed_95cl_v4p2.json`: finite-output, state-hash,
  confidence-level monotonicity, and representative cached/reference closure
  gates.
- `derived/provenance_observed_95cl_v4p2.json`: hashes and exact source lineage.
- `figures/combined_observed_95cl_with_local_asymptotic_p0_v4p2.png`: stacked
  observed-limit and local-`p0` figure.
- `figures/combined_observed_95cl_with_local_asymptotic_p0_v4p2.svg`: vector
  version of the same figure.

## Reproduction

From the isolated v4.2 worktree:

```bash
env PYTHONPYCACHEPREFIX=/tmp/hps-gpr-v4p2-pycache \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/hps-gpr-v4p2-observed95-mpl \
  python3 \
  study_results/v4p2_combined_2015full_2016full_2021_10pct_95cl_observed_20260805/run_observed_95cl_fixed_reviewed.py

env PYTHONPYCACHEPREFIX=/tmp/hps-gpr-v4p2-pycache \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/hps-gpr-v4p2-observed95-plot-mpl \
  python3 \
  study_results/v4p2_combined_2015full_2016full_2021_10pct_95cl_observed_20260805/plot_observed_95cl_with_local_p0.py
```

The production pass is intentionally single-worker and caps every numerical
thread pool at one thread.

## Statistical interpretation

The upper curve is the observed one-sided 95% CLs upper limit on
epsilon-squared in the minimal-visible shared-coupling dark-photon model. It
  uses the bounded `tilde_q_mu` statistic with the asymptotic CLs construction. It is not a
95% confidence interval and does not establish direct 95% coverage.

The lower curve is the local one-sided asymptotic background-only `p0` from the
same shared-epsilon-squared likelihood. It is unchanged by the reporting choice
of 90% versus 95% CLs and is not a toy-calibrated global probability. The
65 MeV annotation marks the minimum found in the plotted scan; it was not a
pre-specified mass. The local `p0` is additionally conditional on the v4.2
factor-12 2016 kernel ceiling, which was accepted following an observed-data
boundary diagnostic; the figure does not correct for that analysis-choice
post-selection.

The 2021 input is the distributed 10% development sample. Dataset activity
changes across the union scan: 2015 alone at 19--38 MeV, 2015+2016 at
39--49 MeV, all three at 50--90 MeV, 2016+2021 at 91--180 MeV, and 2021 alone
at 181--250 MeV.

## Validation summary

The completed run contains 232 finite limits and passes representative
bit-for-bit cached/reference closure at 19, 39, 50, 65, 91, 181, and 250 MeV.
Those representative points also directly bracket `CLs = 0.05` around the
reported upper limit and record the maximum absolute root residual.
Every 95% CLs limit is pointwise greater than or equal to its accepted 90% CLs
counterpart. The 95%/90% ratio ranges from 1.06983 to 1.22015, with median
1.19229.

The unchanged shared-coupling local minimum is
`p0 = 3.259182521304132e-05` at 65 MeV, corresponding to
`Z_local = 3.993214141165979`.
