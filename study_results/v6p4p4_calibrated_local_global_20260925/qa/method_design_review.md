# Independent review of the v6.4.4 calibration design

The A/B design is defensible as a conditional Monte Carlo calibration of a specified mass-grid statistic. Keep the B-calibrated minimum-local-rank result primary. Treat both Sidak curves as references: neither replaces that calibration, and a fitted constant effective trial count need not describe the full threshold range.

## Freeze the statistic and handle ties consistently

With n_A = 1024, define the frozen local map explicitly:

`p_A,m(q) = (1 + count_A[q_A,m >= q]) / 1025`.

Use `S_A(q_vector) = min_m p_A,m(q_m)` for the observed spectrum and every fresh B scan. The global count is `k_B = count_B[S_B <= S_observed]`, including all ties, and the reported Monte Carlo rank is `(k_B + 1)/1025`. The exact Clopper–Pearson interval concerns the underlying B exceedance fraction, not the add-one estimator itself. Conditional on the fixed A maps and fixed null generator, B supplies independent draws for this calibration. The fixed-source and earlier-method-selection qualifications still apply.

The proposed A-only ranks `count_A[q_A,m >= q_A,i,m]/1024` include the tested row. They are **exactly add-one leave-one-out ranks**, since removing that row and adding one restores its contribution, with denominator 1024. This is a sound A-only empirical-rank summary. It is not the full 1025-denominator frozen map evaluated on independent data. Each A row uses a different leave-one-out map, and their minima are mutually dependent; do not describe their empirical CDF as an independent calibration of the full frozen map.

For `q0 = max(0,r)^2`, the atom at zero maps to empirical probability one under inclusive tails. Do not replace that probability by the asymptotic convention 0.5. Exact inclusive numerical comparisons, rather than jitter or random tie breaking, preserve the declared procedure. An implementation can use sorted local statistics and `searchsorted(..., side="left")` to count upper-tail ties.

## A simple frozen effective-trials approximation

The proposed zero-intercept least-squares fit is adequate as a descriptive one-parameter approximation. Fix its thresholds, weighting and formula before inspecting B; do not attach ordinary regression errors to five correlated cumulative counts.

A small improvement is to align the fit thresholds with the A rank grid. For the nominated nominal thresholds `.005, .01, .02, .03, .05`, freeze integer ranks `k = (5,10,20,30,51)` and use `alpha_A = k/1024`. Let `G_A(alpha_A)` be the fraction of A leave-one-out minima at or below each threshold. Set

`x = log(1-alpha_A)`; `y = log(1-G_A(alpha_A))`; `N_eff = max(0, sum(x*y)/sum(x*x))`.

Use equal fixed weights, no intercept, and no imposed upper cap of 136 or 181. A cap is an additional assumption, not a general consequence of dependence. The cached A counts are all strictly between zero and 1024 at these thresholds, so no endpoint regularization is needed. If retaining the original nominal-alpha fit instead, document that choice; it changes the fitted coefficients by about 1%, not the core conclusion.

For a finite-grid reference on B, use `alpha_B = floor(1025*alpha)/1025` and display `1-(1-alpha_B)^N`, with either N equal to the number of mass hypotheses or the frozen fitted N_eff. At attainable observed local ranks, alpha_B equals alpha. A smooth `1-(1-alpha)^N` curve is also acceptable if labeled the ideal continuous-uniform reference, with the local-map resolution stated. The grid counts are 136 for 2016 and 181 for 2021 or the combined analysis; the combined count is not multiplied by the number of campaigns.

These references assume uniform local probability behavior and simplify dependence. Conditional on the realized A maps, local rank distributions in B need not be exactly uniform. The B empirical curve tests the combined adequacy of that simplification and the effective-trials approximation.

## Cached A findings, before B

Calculated directly from `v6p4p3_global_mc_20260925/results/global_scans_{scope}.npz`, using its saved signed roots without refitting:

| Scope | A global counts at the five thresholds | N_eff, nominal alpha | N_eff, attainable A alpha |
|---|---|---:|---:|
| 2016 | 221, 366, 555, 688, 851 | 35.8145 | 36.1967 |
| 2021 | 313, 498, 732, 868, 957 | 56.4495 | 57.0327 |
| Combined | 308, 492, 722, 852, 967 | 57.5459 | 58.1679 |

The nominal-threshold effective counts decrease from 48.5 to 34.7 (2016), 72.8 to 53.2 (2021), and 71.4 to 56.3 (combined) as alpha increases from .005 to .05. A constant trial count already has an observable approximation error in A. Report that fact without adjusting the fit based on B. Predictions below .005 are extrapolations of the fitted approximation, even when a direct B calibration is available there.

The observed minimum frozen local ranks are 2/1025 at 91 MeV for 2016, 3/1025 at 67 MeV for 2021, and **1/1025 at 68 MeV for the combined analysis**. The combined local map has zero A exceedances at its selected minimum. All B scans whose minima attain 1/1025 must tie with it in the global calibration. Consequently, this statistic cannot distinguish an observation just beyond an A local maximum from one arbitrarily farther beyond it. It can still be globally calibrated; its finite-map saturation must remain visible and must not be reported as zero probability or resolved extreme local significance.

## Independent checks and pooling

Compare the frozen Sidak approximation with B's empirical `Pr(S_B <= alpha)`, giving counts and pointwise binomial intervals. If a formal simultaneous curve check is wanted, a distribution-free 95% Dvoretzky–Kiefer–Wolfowitz band has half-width `sqrt(log(40)/(2*1024)) = 0.04244`; it remains valid for a discrete distribution conditional on A. Alternatively, predeclare Bonferroni-adjusted exact intervals at the five comparison thresholds. Do not treat overlapping pointwise intervals as an independent multi-point goodness-of-fit test.

Threshold-dependent N_eff computed from B is useful descriptive information, including transformed probability intervals where finite. It is a same-threshold re-expression of the B global probability, not a second validated approximation. At zero or unit empirical CDF values, report bounds or undefined values rather than an invented finite effective count.

Raw-max statistics do not depend on an estimated local map. Therefore A-only, B-only and pooled 2048-scan raw-max results can be compared at the same observed thresholds; pooled ranks use denominator 2049. The 1024 B scans remain the independent calibration cohort for the frozen-A min-rank statistic. Do not pool A leave-one-out minima with B minima, or refit maps using B and then continue calling the resulting calibration independent.

Use identical extraction, mass support and source arrays in A and B; the cached A protocol uses the selected 2016 ±2.5u window. Only the RNG namespace changes from 1 to 2. Reuse each whole campaign draw across masses and scopes, preserve cross-scope correlations, and retain the single common coupling coordinate in every combined fit. The new namespace is a reproducible disjoint stream convention; archive draw hashes and the protocol identity as checks.

This review performed only array/rank calculations. No new spectra were simulated, no likelihood fits were run, and no report or engine file was changed.
