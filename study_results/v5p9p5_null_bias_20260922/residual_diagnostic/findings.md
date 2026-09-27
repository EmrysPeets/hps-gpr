# Slide 14: conditional residual diagnostic

The expected `Q/Nbin` is close to one under the frozen nominal GP source, with identifiable deterministic prediction bias near the lower edge and selected local regions. The largest observed value, 2.329997 at 78 MeV, is much larger than the local null-mean displacement. It is still only an exploratory conditional residual diagnostic: 20 of 256 complete paired Poisson scans have a maximum at least this large over the specified 201-point scan. The observed curve is one correlated realization, not a plot of the ensemble bias.

## Sources and scope

- Every array in the pinned v5.0.5 2021 spectrum is identical to its v5.8.2 counterpart. The 2021 nominal GP source `B` is bitwise identical to the source underlying the slide-50 archived field; the identity and source hashes are saved in `inputs/source_identity.json` and `inputs/source_hashes.json`.
- The input consists of the released 2021 10% data, 422 bins of width 0.625 MeV, with actual whole-bin edges 36.0–299.75 MeV (nominal support 36–300 MeV). Test masses are 50–250 MeV in 1-MeV steps. Slide 50 has a 0.5-MeV field; this audit uses its matching integer coordinates.
- Every exclusion remains ±2.25σ. Kernel constants and length scales are archived per-mass states. No hyperparameter optimization, signal fit, new toy draw, or full-2021-data access was performed.
- The same 256 saved complete Poisson spectra are reused across all 201 masses and all controls, preserving their correlations. There are 51,456 exact fixed-kernel GP replays, bounded to one process and one BLAS thread. The main run took 66.5 seconds.
- The original slide-14 CSV is reproduced exactly at all 201 points: maximum absolute difference is zero.

## What the denominator means

For each withheld window, let `b = b(B_train)` and `C` be the GP posterior count covariance evaluated at the fixed source. Let `δ = B_window − b`. For the source-frozen denominator,

`V = diag(b) + C`,

`Q = (Y_window − b_hat)^T V^−1 (Y_window − b_hat)`.

The sideband predictor has count-response Jacobian `J = ∂b_hat/∂Y_train`. The window and training masks are disjoint, so the first-order repeated-sampling residual covariance is

`W = diag(B_window) + J diag(B_train) J^T`.

Consequently,

`E[Q]/Nbin ≈ tr(V^−1 W)/Nbin + δ^T V^−1 δ/Nbin`.

`Nbin` is the number of withheld bins, not fitted effective degrees of freedom. The GP posterior covariance `C` and the repeated-sampling covariance of the estimated background are different objects. A unit expectation requires both the relevant mean condition and covariance matching; it is not guaranteed by using a GP posterior covariance.

The Jacobian includes the response of log targets, the count-dependent `alpha=1/Y_train`, and the lognormal mean correction `exp(μ+diag(C_latent)/2)`. Directional finite-difference checks at 60, 78, 120 and 220 MeV have maximum relative L2 discrepancy `9.51e−7`. An adaptive toy denominator is a different statistic; its behavior is measured separately, not inferred by putting an averaged denominator into the trace formula.

## Quantitative decomposition

The repeated-sampling variance contribution ranges from 0.987901 to 0.998573 across the scan. Thus the covariance mismatch here slightly lowers the expectation; it does not produce a broad positive shift. The window-Poisson contribution alone is 0.892936–0.976855. Replaying the sideband estimator adds 0.020739–0.096723.

The complete first-order expectation ranges from 0.997566 to 1.184789. The maximum occurs at 50 MeV, where the deterministic-bias contribution is 0.195686. Representative results are:

| Mass (MeV) | Variance term | Deterministic bias term | Predicted mean Q/Nbin | Exact Poisson mean, adaptive V | 95% MC interval on mean |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.989103 | 0.195686 | 1.184789 | 1.179083 | [1.125135, 1.233031] |
| 71 | 0.993798 | 0.111883 | 1.105681 | 1.095102 | [1.045286, 1.144918] |
| 78 | 0.995797 | 0.030446 | 1.026243 | 1.021726 | [0.976244, 1.067208] |
| 120 | 0.997546 | 0.000811 | 0.998357 | 1.007521 | [0.967678, 1.047364] |

The MC intervals above describe uncertainty on a 256-toy mean, not an event-level 95% acceptance band. Neighboring entries are correlated.

Eight paired controls separate frozen prediction, sideband replay, linearized replay, deterministic-bias removal, and fixed/adaptive denominator choices. Removing `δ` from the paired residual is a mathematical counterfactual; it changes neither the data nor the source. At 78 MeV, exact replay with fixed `V` has mean 1.021684; the bias-removed counterpart has mean 0.990299. At 50 MeV they are 1.178132 and 0.985654. This isolates the mean displacement without confusing it with the extra sampling variation of the refitted background.

Across all masses, the largest absolute paired mean effect from exact versus linearized replay is `2.14e−4` in Q/Nbin. Adaptive versus frozen `V` changes the mean by at most `9.52e−4`. Those effects are small relative to the source mean shifts identified above. The largest exact mean versus first-order expectation discrepancy is 2.36 estimated MC standard errors; those comparisons are correlated and are not a separate multiple-testing claim.

The empirical mean of a squared vector norm has a positive finite-MC floor. `empirical_decomposition_with_MC_floor.csv` subtracts the estimated prediction-mean floor from its squared-bias/Jensen-shift estimators; corrected estimates can be negative. A positive uncorrected squared mean is not evidence of a nonlinear Jensen bias. The much more precise paired scalar exact-minus-linear control supports the small nonlinear correction quoted above. The empirical covariance-plus-mean decomposition reproduces each fixed-V sample mean to `6.67e−16`.

## Why slide 50 can have a ≈ 0.58 while this bias term is only 0.03

At 78 MeV, there are 16 withheld bins. The total source-residual norm is

`δ^T V^−1 δ = 0.487138`; dividing by 16 gives `0.030446`.

The signal fit instead projects the residual along one signal-template direction:

`a_GLS ≈ S^T V^−1 δ / sqrt(S^T V^−1 S)`.

This gives 0.576797, while the archived nonlinear profiled Asimov root is 0.576784. Its square is about 0.33268: roughly 68.30% of the total deterministic residual norm lies in the signal direction. These quantities have different normalizations and answer different questions. There is no contradiction and no claim that Q/Nbin should equal the signed likelihood root.

The archived likelihood uses a very slightly conditioned and rank-truncated `L L^T` in place of raw `C`. Repeating the GLS comparison with that covariance changes the root by at most `4.44e−8` across the 201 coordinates, so this numerical distinction does not explain the displayed bias. The bridge comparison is an approximate linear-Gaussian projection and the saved exact nonlinear root is retained separately.

## Exploratory residual-scan maximum

For `T_Q = max_m Q(m)/Nbin` over the pre-existing 50–250 MeV integer grid, the observed maximum is 2.329997 at 78 MeV. Among the same 256 complete conditional Poisson scans, 20 exceed that threshold. The add-one estimate is 0.081712; the two-sided 95% Clopper–Pearson interval for the binomial exceedance probability is [0.048372, 0.118080]. The median null maximum is 1.797515 and its empirical central 90% interval is [1.432114, 2.478929].

This comparison prevents treating the selected local maximum as an unselected point. It is an exploratory fixed-source residual-scan diagnostic, not resonance significance, a calibrated model-goodness-of-fit probability with source uncertainty, or independent validation. The generating background was estimated from the observed data; source estimation, possible signal absorption, and procedure selection remain outside this conditional calculation. No production recalibration is adopted.

## Reproduction and products

Run `python run.py` followed by `python summarize.py` from this directory using Python with NumPy, SciPy and Matplotlib. Inputs are bundled and checked by SHA-256. Parent repository files are optional additional identity checks; the numeric replay works from this directory alone.

- `results/analytic_scan.csv`: all 201 analytic decompositions and original observed Q.
- `results/poisson_controls.csv`, `paired_contrasts.csv`, and `paired_Q_arrays.npz`: all paired exact controls and finite-MC summaries.
- `results/empirical_decomposition_with_MC_floor.csv`: repeated-sampling and finite-MC mean diagnostics.
- `results/bias_projection_bridge.csv`: source-residual projection and saved slide-50 roots.
- `results/exploratory_residual_scan_tail.json`: counts, interval, scope and interpretation.
- `results/validation.json`: semantic and numerical checks.
- Three vector PDF/raster PNG figures in `figures/` were visually inspected; axes, legends, and labels are legible and unclipped.
