# Candidate-region replacement and conventional fits: final numerical and scientific review

6 September 2026. This review concerns the isolated candidate-removal derivative. The frozen parent studies and released datasets were preserved. The reviewer prepared and ran the conventional implementation after the parent supplied the measured persistence trigger; the parent independently reconstructed those conventional fits. The intervention implementation was reviewed independently by this reviewer.

**Disposition:** accepted as a conditional influence experiment and a comparison of local background models. The local 2021 peak/dip pattern is strongly affected by replacing the selected regions, while appreciable variation remains elsewhere. Neither universal invariance nor a particle interpretation follows. Numerical convergence is established separately from background adequacy.

## What was checked

`independent_intervention_audit.py` passes **3,872 checks** using saved products only. It verifies the complete 72/142/201 mass grids with 42 fixed spectra per dataset, totaling **17,430 profile tests**; exact exterior-count preservation; primary masks; paired single/both-hole draws; integer replacement counts; source-specific lognormal means and latent covariance factors; and the polynomial filler through independent 16-node quadrature. It reconstructs every remote mask by locating the saved original fit counts in the full native histogram and requiring the entire resulting window to be disjoint from both primary holes. Every reported variation metric and the routing flag are independently reproduced.

The parent's separate `qa/numerical_validation.json` records **59,904 passing checks**, including 5,160 saved GP fits and all 30 conventional fits. Its maximum independently reconstructed signed-root errors are 5.30e-13 and 2.90e-11, respectively. The full reference replay differs from the archived accelerated reference by at most 2.09e-5, well within its numerical tolerance. These checks establish implementation closure, not a qualified physical background or tail probability.

The primary latent covariance has only roundoff-scale negative eigenvalues before clipping (absolute magnitude at most 2.81e-12, compared with positive largest eigenvalues of order 1e-5 to 3e-4). The saved factor reproduces the clipped covariance. Inspection of `CachedCholeskyPredictor.latent` confirms it returns the latent conditional covariance, without adding query-bin Poisson noise; the replacement stage supplies that noise once. The lognormal mean is `exp(mu + diag(V)/2)`, rather than the median `exp(mu)`. Reference fills use their own source spectrum. Only observed replicas are generated, so the seed's present omission of a source tag does not create cross-source shared draws.

## Persistence has a restricted, quantitative meaning

For the primary deterministic both-hole replacement, the ratio of remote root standard deviations and the retained sign transitions are:

| Dataset | Observed std ratio | Observed transitions | Reference std ratio | Reference transitions |
|---|---:|---:|---:|---:|
| 2015 full | 0.673 | 7 | 1.102 | 4 |
| 2016 full | 0.778 | 8 | 0.928 | 7 |
| 2021 10% | 0.921 | 17 | 0.865 | 10 |

The remote sets contain 40, 74 and 170 mass coordinates. All six source/dataset cases satisfy the predeclared routing threshold. This threshold is descriptive, not a statistical significance test. The reference row diagnoses how the moving extraction responds to the specific stress-generating spectrum; it is not evidence of a measured data bias.

The ten paired observed both-hole replicas retain standard-deviation ratios 0.573–0.781, 0.762–1.096 and 0.908–0.954 for 2015, 2016 and 2021. These are descriptive ranges from ten conditional replacements. They are not confidence intervals or independent background-null experiments. In 2015, the correlation with the original remote field spans 0.066–0.821: retaining variation does not mean retaining an unchanged pattern. The corresponding primary deterministic correlation is 0.559.

The familiar **2021 dip at 71 MeV is strongly influenced by the selected neighboring regions**. Its observed root is -4.0188. With the primary mean it becomes -1.4784 after removing 78 MeV alone, -3.2562 after removing 65 MeV alone, and -0.7154 after removing both. This agrees qualitatively with the earlier positive-injection echo mechanism. It establishes influence under the declared replacement, not that the removed events were particles. The candidate self-roots also collapse: 78 MeV changes from +2.8086 to +0.00028, and 65 MeV from +2.3962 to -0.2174 under both-hole replacement. That loss is expected when their own fitted counts are replaced.

## Filler and hole-width dependence must remain visible

The polynomial and widened-GP variants are not interchangeable truths. Their remote standard-deviation ratios are:

| Dataset | Polynomial, observed/reference | Widened GP, observed/reference |
|---|---:|---:|
| 2015 full | 1.227 / 3.626 | 0.719 / 0.836 |
| 2016 full | 0.783 / 0.978 | 0.624 / 0.671 |
| 2021 10% | 1.107 / 1.162 | 0.861 / 0.382 |

The 2021 widened-hole reference field is substantially reduced and changes shape: its correlation with the original is -0.326. The comparison deliberately uses the same primary remote set; some windows in that set may overlap the extra bins removed by the wider intervention. The reported ratio therefore does not isolate an exclusively remote effect of the widened holes.

Several polynomial fillers visibly fail to describe their retained sidebands. At 2015/21 MeV the reference polynomial has deviance 267.20 for 33 degrees of freedom, a ratio 8.10; its large increase in the root variation cannot be promoted into evidence that it is a superior physical truth. At 2021/65 MeV the observed and reference polynomial fillers have deviance 96.12/17 and 86.17/17. The reference is deterministic, so a Poisson deviance here is a mismatch scale, not a calibrated goodness-of-fit probability. The observed-sideband fit also warns against treating that alternative filler as qualified. All alternatives should nevertheless remain visible because they expose the intervention's model dependence.

## Conventional fits: all 30 retained, no preferred probability selected

All five predeclared variants at each of six GP-selected masses were fitted to the **original observed native bins**. The baselines are summarized below. The sign of the auxiliary amplitude defines the signed likelihood root; the Gaussian mass and resolution are fixed. The yield is the full, untruncated Gaussian normalization, with the actual fit-window fraction saved separately.

| Dataset | Mass [MeV] | Original GP root | Conventional baseline root | Nominal local p0 | Free-fit D/dof |
|---|---:|---:|---:|---:|---:|
| 2015 full | 51 | +3.139 | +0.131 | 0.448 | 141.7/131 |
| 2015 full | 21 | +2.516 | +5.580 | 1.21e-8 | 92.5/50 |
| 2016 full | 90 | +3.425 | +1.998 | 0.0229 | 125.3/113 |
| 2016 full | 117 | +3.279 | +1.525 | 0.0636 | 165.6/155 |
| 2021 10% | 78 | +2.809 | -1.350 | 0.5 | 29.2/24 |
| 2021 10% | 65 | +2.396 | +1.960 | 0.0250 | 17.4/22 |

At **2015/21 MeV**, the baseline's nominal root 5.580 does not support a 5.6-sigma particle claim. The free fit still has D/dof=1.850 and nominal goodness-of-fit reference 2.42e-4. Increasing the polynomial degree by the predeclared one step gives root 3.461 with D/dof=0.958; the shorter window gives 5.114 with D/dof=1.377; the wider one gives -1.303 with D/dof=3.905. This is substantial model dependence. Better fit quality in one variant does not authorize selecting it as a new significance result after inspecting all outcomes.

At **2021/65 MeV**, the reduced-degree model gives root 8.459 but D/dof=28.105. It is a clear illustration of a badly misspecified background creating a very small formal local reference. The baseline gives 1.960, while degree-plus and both width variants give approximately 0.856–0.972 with much smaller deviance. At **2021/78 MeV**, all five conventional roots are nonpositive, including baseline -1.350. This shows that the original GP excess is not robust across the particular retained conventional descriptions; it does not prove that one method is the true background.

At 2015/51 MeV the baseline is near zero and the retained degree-plus/width variants range from -0.927 to +1.437. The degree-minus root -34.625 occurs with D/dof=22.391 and is not evidence for an enormous physical deficit. At the two 2016 masses, the degree-three/four and width variants give modest roots; the reduced-degree models produce negative roots with poor fit quality. None of these local comparisons has been adjusted for GP mass selection, model inspection, dataset choices, or the conditional decision to run this stage.

The formula `p0=sf(max(r,0))` gives the stated 0.5 convention for nonpositive roots. It is an asymptotic local reference, not a calibrated finite-sample or global probability for this follow-up. An underflowed nominal goodness-of-fit entry at the extremely poor 2015/51 degree-minus fit is numerical underflow, not a probability established to equal zero; the report should use its finite D/dof rather than literal zero.

The HPS literature motivation and the explicit adaptations are documented in `HEP_INTERVENTION_PROTOCOL_REVIEW.md`. The current coarser native bins, fixed 8-sigma 2016/2021 window choice and current datasets prevent calling this an exact historical-analysis replay. The 2015/21 support shift is explicitly recorded. No event outside the already released samples was used.

## Numerical refinement and display checks

The first conventional attempt completed 29 fits and stalled near stationarity in two fixed starts of the remaining 2021/65 degree-minus fit. The predicted likelihood improvement was below floating-point noise. The common implementation now evaluates the deviance in extended precision where available and permits a near-stationary full Newton refinement only when it halves the scaled score and preserves NLL within 1e-9. The original final score, covariance, multistart and quadrature gates were unchanged, and all 30 fixed fits were replayed. The 29 previously completed roots changed by at most 2.08e-12. Initial failure and refinement records remain in `traditional/qa/initial_attempt/`.

The final maximum scaled score is 8.52e-8; the largest fixed-start NLL difference is 5.16e-12; the doubled-quadrature prediction difference is 1.12e-15 relative. The parent further reproduced every fit with independent 48-node integration. The three conventional figures show all variants with individual D/dof labels, plus baseline count and residual panels. Counting bars do not include fitted-background uncertainty, and subtraction induces correlations; the footer says so. The isolated Gaussian component differs from the plotted total-minus-null curve because the jointly fitted polynomial moves. Rendered PDF pages were checked for clipping, label readability and legend/data overlap.

The study supports a model-diagnostic conclusion: selected local structures can induce neighboring responses, broader variation is not wholly explained by those regions under the primary intervention, and both GP and conventional extractions require background qualification before discovery claims. Matched null calibration, spurious-signal controls across independently chosen smooth truths, and an independent validation sample remain requirements for formal inference. This derivative supplies no new calibrated global significance, limit, sensitivity estimate, or particle discovery.
