# Exploratory rigid-generator fit results

This directory is isolated scratch work. It does not qualify a generator, make
toys, or change the active v4.8 fail-closed state.

## Candidate selected for further testing

On 40--300 MeV, the best six-free-parameter development candidate is

\[
 \mu(x)=A\,\mathrm{expit}\!\left(\frac{x-x_t}{w}\right)
 (x-x_0)^a
 \exp\!\left[-\left(\frac{x-x_0}{\lambda}\right)^p
 +d_2T_2(u)+d_6T_6(u)\right],
 \qquad u=\frac{2(x-0.040)}{0.300-0.040}-1.
\]

The common threshold constants are frozen from a 1% generalized-gamma
reconnaissance fit:

- `x0 = 0.03363345155 GeV`
- `xt = 0.04980944185 GeV`
- `w = 0.006359634014 GeV`

The six free coordinates in the finalist fit are `A`, `a`, `lambda`, `p`,
`d2`, and `d6`. The 1% best fit is

- `A = 2.588090444e13`
- `a = 4.091830656`
- `lambda = 0.000808994011 GeV`
- `p = 0.5253920171`
- `d2 = -0.1092481565`
- `d6 = +0.01173442681`

This is a generalized-gamma threshold model with only two preselected broad
correction modes. It is not a full sixth-order polynomial: only `T2` and `T6`
are present. A 21-pair development scan over two-mode subsets of `T1`--`T7`
selected `(T2,T6)` both for the 1%-shape-scaled native-10% comparison and for an
independent native-10% refit. The scan is in
`derived/sparse_mode_pair_scan.csv`.

## Fit metrics

All values below use the native 0.125 MeV bins. They are engineering scores,
not evidence that the analytic family is a true Poisson model.

| policy/source | free parameters | Pearson chi2/ndf | deviance/ndf |
|---|---:|---:|---:|
| fit shape to 1% | 6 | 1.10585 | 1.10325 |
| freeze 1% shape, scale normalization to native 10% | 1 | 2.78945 | 2.77290 |
| independent native-10% shape refit, seeded from 1% | 6 | 2.67934 | 2.66892 |

The shape-frozen policy therefore meets the requested `<3` native-bin target
without allowing native 10% to reoptimize its shape. It is the more defensible
anti-absorption policy. If the exposure normalization can be fixed externally,
the native-10% application has zero fitted coordinates rather than one.

The result does **not** give coarse-bin agreement. For the shape-frozen native
10% application, rebin-5 Pearson/deviance are 9.77/9.69, and they grow for wider
bins. The maximum native-bin Pearson residual is 11.0. Thus `<3` here means
average native-bin discrepancy in a very fine histogram; coherent broad
residuals remain and must be shown in the fit/residual figure.

The 30 MeV support is rejected for this rigid family. With the 1% shape frozen
and only normalization scaled, native-10% Pearson/deviance are 13.95/11.09.
Lowering the source-fit edge from 40 to 30 MeV is therefore not supported by
this test.

## Optimizer reproducibility

In a 24-start audit, the best branch was reproduced to `1e-9` in objective by
12/24 starts for 1% and 14/24 for native 10%. The best metrics and parameters
were unchanged. Other starts found inferior branches with objective spreads of
`6.19e-5` and `5.82e-5`, respectively. Production must seed from the 1% branch,
retain multiple restarts, and select by maximum likelihood with parameter and
prediction reproducibility checks. See `derived/finalist_restart_audit.json`.

## Signal-absorption diagnostic

An Asimov Gaussian-injection influence test used the 2021 mass-resolution
parameterization at 65, 90, 120, 180, and 210 MeV and strengths 1, 3, and 5
matched-filter sigma.

- If all six shape coordinates are refit, the Poisson-metric absorbed fraction
  is 17.3%--32.2%; the integrated `+-2.25 sigma` window fraction is
  22.8%--41.0%. The largest absorption occurs at 65 and 210 MeV.
- If the 1% shape is frozen and native 10% refits normalization only, the
  Poisson-metric fraction is 1.3%--10.4% and the window fraction is
  1.7%--13.7%.
- If normalization is also fixed from an external exposure ratio, absorption
  in the native-10% source fit is zero by construction.

The six-coordinate refit is therefore not sufficiently signal-rigid if a
20%-absorption gate is desired. The recommended development policy is to fit
the shape on 1%, freeze it, and scale it into higher-statistics source lanes.
This does not remove the need to test whether the original 1% shape fit can
absorb a signal.

## Smoothing result

Straight adaptive Gaussian smoothing with bandwidth `2.25*sigma(m)` did not
solve the problem. Although some fits look acceptable against the smoothed
target, the comparison to the unsmoothed source worsens sharply. The smoothed
points are also correlated, so a diagonal smoothed-target chi2 is not a Poisson
goodness-of-fit. Linear smoothing preserves signal area and can spread a narrow
bump into a shape that a global function absorbs more easily. Any smoothing
proposal should therefore be paired with an explicit covariance and injected
signal-retention test; it should not be adopted from appearance alone.

## Parameter-count caveat

The finalist has six free coordinates after `x0`, `xt`, and `w` are frozen, but
those constants were estimated in a preceding seven-parameter 1% development
fit. That discrete/frozen preprocessing is extra effective complexity. For a
strict publication claim of "fewer than seven fitted parameters," the threshold
constants and `(T2,T6)` choice should be predeclared from this development stage
and held fixed in a fresh validation stage; they should not be retuned on the
same validation source.

## Reproduction

- `fit_rigid_candidates.py`: families, 30/40 MeV fits, raw/smoothed diagnostics
- `scan_sparse_mode_pairs.py`: 21 two-mode development scan
- `evaluate_scaled_seed_finalist.py`: 1%-shape-frozen scaling comparison
- `test_signal_absorption.py`: full six-coordinate Asimov influence diagnostic
- `verify_finalist_restarts.py`: 24-start optimizer audit
- `derived/rigid_candidate_metrics.csv`: complete metric table
- `derived/scaled_one_pct_seed_metrics.csv`: scaled-shape metrics
- `derived/scaled_one_pct_seed_absorption.csv`: normalization-only influence
- `derived/signal_absorption_diagnostic.csv`: six-coordinate influence
