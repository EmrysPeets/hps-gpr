# HEP review of the 2016 and 2021 global-significance extension

Independent review, 6 September 2026. The reviewer generated no toys, performed no new likelihood fits, and changed no numerical or plotting code. This review covers the declared finite 2016-full and 2021-10% scans separately. Neither methods nor datasets are pooled or selected by their result.

## Statistical definitions

The reviewed construction is internally consistent. Each pseudoexperiment supplies one full spectrum to every mass hypothesis; the archived stress intensity is required to be identical throughout that scan. The exact reference runner is byte-identical to the v4.9.14 runner; the separately contracted, audited numerical derivative described below completed production. The new protocol retains mass-dependent observed kernels, support, templates and numerical tolerances while updating the GP predictions and count-dependent training errors.

The Asimov response is a first-order approximation to the joint signed-root distribution. With independent Poisson bins, the one-standard-deviation response vectors give `C = D.T @ D`; the eigenfactor used by the analyzer reproduces that covariance. The approximation retains its predicted offset `mu = r(B)` and response standard deviation `s` rather than asserting that raw roots already have a standard-normal null distribution. In a nonlinear fitting procedure, `r(B)` need not equal the exact ensemble mean, so the independent mean, width, correlation and maximum comparisons are substantive validation checks.

For a positive observed raw root, the local Gaussian-response probability is `sf((r-mu)/s)`. For `r <= 0`, it is one, including the bounded-statistic atom. Therefore the minimum local probability over masses is equivalently represented by the largest standardized root among masses with positive raw root. The analyzer applies this restriction to observed, direct-Poisson and Gaussian-field scans alike. Its pointwise global curve asks whether any tested mass is at least as extreme as the observed local value at that point.

The separate maximum of the raw nonnegative root preserves the original asymptotic p-value ordering, with its null distribution evaluated under the same common stress background. These are two declared orderings, not interchangeable estimates of one quantity. The report should retain both even when they give very different answers. Neither calibrates the v4.9.13 two-truth envelope.

The direct-Poisson probability calibrates the chosen scan score under the specified generating spectrum even if its Gaussian marginal approximation is imperfect. The Gaussian-field probability additionally depends on the response covariance and joint-Gaussian approximation. Agreement of these two probabilities supports that approximation only at the precision and thresholds actually tested.

## Particular interpretation issue for 2016

Large stress-model offsets can dominate either ordering. A negative mean offset can turn a modest positive observed raw root into a large standardized root; the raw-positive requirement still forces `p = 1` for any nonpositive observed root. Conversely, large positive stress offsets produce large raw null maxima and can make the observed raw maximum entirely unexceptional. These results assess a particular stress truth passed through a particular estimator. They neither identify a particle signal nor certify the background description.

The completed ten-scan pilot already illustrates this: profiled 2016 null raw maxima ranged from approximately 13.02 to 15.38, while the observed maximum is 3.42475. Fixed-background null maxima ranged from approximately 30.48 to 37.87, while the observed maximum is 8.45255. These pilot ranges explain the expected behavior of the raw ordering; ten scans are not a precision estimate of its tail.

At each reported most-extreme point, the report should give the mass, observed raw root, Asimov offset, response standard deviation and standardized root alongside the global probability. This makes any stress-offset contribution explicit. The archived 2016 source-fit waiver, transition region, lack of independent background certification and separate parent numerical exception remain relevant. The 2021 native-10% stress shape also remains conditional rather than a complete physical background model.

## Essential validation and display qualifications

- The analyzer now checks complete ensembles, product hashes, per-point numerical audits and matching generating/source contracts before combining inputs. The earlier sparse-tail inclusion check now uses an exact binomial upper bound.
- The copied legend's hard-coded 72-mass label was identified and corrected to the actual grid size. The extension uses 142 and 201 masses, respectively.
- Zero observed exceedances require a Monte Carlo upper bound, not a probability claim of zero. The one-sided 95% upper bound is about 0.00299 for zero of 1,000 direct scans, and about 1.50e-5 for zero of 200,000 GP fields. The upper endpoint of a two-sided 95% interval is a different, more conservative number; the updated output retains both definitions.
- The marginal normality and maximum-distribution KS checks are diagnostics. Non-rejection cannot certify rare Gaussian tails. Pointwise maximum-tail intervals are not a simultaneous confidence band, and agreement at the observed maximum is not an omnibus validation of the full field distribution.
- A 2 MeV subgrid comparison measures the effect of removing alternate tested masses at a fixed score threshold. It does not certify an untested continuous-mass search.
- Any Gaussian estimate beyond the resolving power of the direct ensembles must remain a model-based extrapolation under the declared stress truth. Increasing the number of cheap GP draws reduces sampling error on that approximation; it does not improve validation of its physical or statistical assumptions.
- Repeating the frozen runner with a different output directory reproduces its existing deterministic toys. Additional independent scans require a newly declared ensemble/seed coordinate and a separately contracted derivative; a new filename alone does not increase statistical information.

The draft `note/reader_report.tex` and `build_report.py` were also reviewed. Their interpretation correctly distinguishes rejection of one stress construction from evidence for a signal, and a raw-maximum probability near one from a goodness-of-fit certificate. The peak table exposes the raw value, offset and response width. A minor conditional issue was reported: the raw-ordering summary table should use the same zero-count bound formatter as the principal table if that ordering has zero exceedances.

## Numerical result audit

The 2016 and 2021 ten-scan pilots passed their numerical checks, taking 50.53 s and 18.45 s respectively. Their full supports contain 720 and 422 bins. The exact larger jobs were subsequently paused at retained checkpoints after their measured timing implied a long continuation. Both separately contracted accelerated datasets are now complete and have passed the numerical acceptance gates. The final independent audit and statistical qualifications are recorded below; numerical acceptance does not itself validate a Gaussian extreme tail or the physical stress background.

## Preproduction low-rank proposal: risk and bounded acceptance gates

This section preserves the preproduction recommendation. The implemented gates and completed results are assessed separately below. The proposed derivative retains the identical spectra, seeds, observed references, truth, support, kernel states, scan grids, likelihood and probability orderings. Re-evaluating the same spectra with a different numerical backend creates a paired numerical comparison; it does not create additional independent experiments. Preserve the original exact pilots, completed exact validation columns, hashes and stopping state.

The reviewed `calibration_core.enable_lowrank(ctx)` implements two approximations together: a relative `1e-15` truncation of joint train/query kernel eigenfeatures, and removal of nuisance-covariance modes below `1e-5` in Poisson-variance units. Count-dependent training errors, posterior means and covariance are still recomputed for every spectrum. The routine tests both generating truths and its predeclared proposals at each mass against exact predictions, and retains the `1e-3` gates for predictive mean/covariance and absolute signed-root/bounded-statistic differences. A failed approximation restores the exact cached-Cholesky backend. This is an appropriate first numerical gate and should be retained unchanged.

Those gates alone do not certify the response covariance. Let the numerical root error on spectrum `n` be `e(n) = r_fast(n) - r_exact(n)`. Then

```
D_fast[i,m] - D_exact[i,m] = e(B + sqrt(B_i)e_i; m) - e(B; m).
```

Even if both root errors are below `1e-3`, their difference can approach `2e-3`. Over 720 bins, the worst allowed Euclidean error of one response column is `2e-3 * sqrt(720) = 0.0537`. That can be several percent of a response width near one. It is a worst-case bound, not a prediction of this implementation's error, but it demonstrates why root agreement is insufficient. Truncation or nuisance-rank changes can also introduce small nonsmooth numerical changes as individual bins are perturbed. Judge the response differences themselves; do not divide by a nearly zero individual response entry.

The following is a bounded proposed audit. Its coordinates and tolerances should be frozen before examining the accelerated results; an already declared equivalent audit should not be replaced retrospectively with easier points.

1. **Identity and inherited tests at every mass.** Verify bitwise equality of the generating means and full count arrays, keep all original convergence/scalar-reference tests, and run the existing `enable_lowrank` checks at every coordinate. Save its checks, selected backend, compression settings and fallback reasons. Keep observed roots at their frozen exact values and verify their existing agreement threshold.
2. **Use every existing exact comparison.** Compare all same-ID roots at every completed exact validation mass, for both methods, and compare the complete ten-scan exact pilots across every mass. Retain the inherited maximum absolute root discrepancy below `1e-3`, with per-mass maxima and RMS errors. Do not pool paired copies into a larger toy count. The overlap is a restricted grid and cannot establish agreement of full-grid maxima where exact columns are absent.
3. **Exact unfluctuated baseline at every mass.** Compare the fast and exact `r(B)` and require both absolute difference below `1e-3` and difference divided by the provisional fast response width below `1e-3`. Record positive finite widths. The normalized check matters because this baseline enters every centered score. It is inexpensive compared with an entire response column.
4. **A small fixed response stencil at every mass.** One concrete choice is 16 uniformly spaced full-support bin indices, augmented by the bins nearest the signal center, each moving-mask boundary and its immediate neighbors, and the known source-transition boundaries. Deduplicate indices and freeze the resulting exact lists. Compare `D_fast` with `D_exact` using their respective baselines, requiring maximum sampled-entry error divided by that mass's response width at most `1e-4`. This is an across-grid diagnostic, not a guarantee about every untested entry.
5. **Complete response columns at fixed diagnostic masses.** A concrete six-mass choice is 39, 66, 75, 85, 120 and 180 MeV for 2016, and 50, 65, 80, 120, 180 and 250 MeV for 2021. These cover edges, the specified threshold/transition mechanisms and separated interior regions. At each, evaluate all one-bin perturbations exactly. With the exact response width as scale, require `||D_fast[:,m]-D_exact[:,m]||_2 / ||D_exact[:,m]||_2 <= 1e-3` and relative width discrepancy at most `1e-3`. Also compare every pair among the complete exact columns and require maximum absolute correlation discrepancy at most `5e-3`. The column-norm gate controls accumulated weak response errors that a few selected bins can miss.
6. **Boundary and scan-score checks.** Explicitly count every exact/fast change in `r > 0` on overlapping spectra. Resolve any such numerical ambiguity with the dense reference and preserve its record. For paired roots, apply the same fixed centering/scaling map to both versions when isolating root-solver error, then compare their scan maxima and threshold decisions on the available common grid. This prevents a change in the covariance approximation from being confused with a change in the root calculation. Record the score differences and any changed decisions; exact comparisons on the ten full pilot scans provide the only full-grid paired check until further exact columns exist.
7. **Fallback and final empirical validation.** A failed mass-level approximation or response gate should select the dense backend for that entire coordinate, consistently in all newly assembled ensembles, with its exact baseline/response column. Retain failures; do not drop offending toys or choose a backend from a more favorable p-value. Repeat the declared independent direct-versus-GP maximum-distribution checks after the numerical backend map is frozen. The final report must distinguish these statistical checks from numerical agreement.

The numerical tolerances above are proposed engineering error budgets for a conditional pilot. They are not theorem-level probability-error bounds or replacements for the direct scan validation. In particular, a small width error can still change a very rare Gaussian tail noticeably, and a subset of exact response columns does not certify the entire untested covariance. Report the actual observed discrepancies and the tested fraction of response entries. The finite-field binomial interval only measures Monte Carlo sampling uncertainty; it does not include numerical approximation, Gaussian extrapolation, background-model or frozen-kernel uncertainty.

Keep the exact baselines and response-column comparisons matched to the same mean spectrum used by production. Do not substitute arbitrary validation spectra for the Asimov perturbations. If low-rank truncation and nuisance-mode compression need to be separated to diagnose a failure, record that as a new numerical policy and contract, rather than changing one setting silently within the frozen derivative.

If the fast derivative is adopted, the reader report must no longer describe its entire production run as byte-identical to the exact runner. It can accurately state that the statistical procedure and spectra were retained, that the original exact runner supplies reference checkpoints, and that a separately audited numerical backend completed the new products.

## Code review of the implemented accelerator and verifier

The separately contracted `run_global_accelerated.py` and `verify_acceleration.py` were reviewed read-only while production was in progress. The implemented full-response mass sets are 39, 56, 66, 75, 120 and 180 MeV for 2016, and 50, 78, 100, 150, 200 and 250 MeV for 2021, as frozen in `ACCELERATION_RESPONSE_GATES.md`. They differ from the earlier illustrative recommendation above and must be identified by their actual declared values. The implemented ordinary stencil contains the uniform 16-bin grid and the blind-window endpoints/center; it does not explicitly include all immediate neighbors suggested in the earlier example.

The response-cancellation arithmetic is correct. Both exact and approximate probes include the unperturbed spectrum first. The code compares the separately centered response vectors, rather than comparing roots alone or inadvertently using a different baseline. Full-column checks use the exact response norm; the verifier constructs covariance and correlation submatrices directly from saved exact and final response columns. Its correlation gate, absolute `1e-3`, is tighter than the earlier proposed `5e-3` budget.

The coordinate fallback is consistent across ensembles. If the inherited low-rank gate, a response test, an exact-overlap root comparison, or approximate execution fails, the runner retains the reason and evaluates that whole mass with the exact backend for the pilot, validation and Asimov ensembles. A failure of an exact reference halts execution rather than accepting an approximate substitute. No offending toy is removed. The global analyzer calls the verifier before computing its probabilities, then checks ensemble hashes and matching contracts.

The verifier additionally requires zero changes in `r > 0` on the available paired exact roots. This is a fail-closed final acceptance gate. Such a sign change is not itself an automatic fallback trigger in the running coordinate code: if one is found, analysis stops and the affected coordinate requires an explicit exact recovery. The report must restrict any “zero atom flips” statement to the paired spectra that were actually compared.

Three inexpensive verifier checks were recommended before accepting complete results: require exact pilot references at every mass, validate each exact reference against its saved successful audit and checkpoint hash, and check that the exact full-response sentinel mass set equals the declared set rather than merely counting six files. All three were implemented in the final verifier and independently checked for both completed datasets. They strengthen provenance and completeness without changing the numerical runner or cohort.

Read-only spot checks of current coordinate records confirm genuine fallback behavior at 2016 masses 41, 62, 75 and 76 MeV. At the complete 39 and 66 MeV response columns, the saved accepted profiled relative L2 response errors were approximately `8.27e-6` and `3.69e-6`, respectively, comfortably below the declared `1e-3` threshold. These examples confirm the intended operation of the checks; they do not replace the final all-coordinate verifier or the independent statistical validation.

## Completed independent numerical audit

The final audit read the saved spectra, coordinate checkpoints, exact references, response sentinels, covariance matrices, maximum vectors, CSV curves and contracts. It made no new random draws or likelihood fits. Both method columns in all 426 assembled 2016 ensemble checkpoints and all 603 assembled 2021 ensemble checkpoints matched their saved checkpoints and successful audits. The reviewer checked 517 recorded input/reference hashes for 2016 and 473 for 2021; all matched. After the explicit missing-comparison metadata refinement, the final verifier passed 161 and 214 checks, respectively. Its SHA-256 is `cc8402ea5083d8c27a92a437152d27c03164026489026155ed7dd8962b4dd272`. The analyzer SHA-256 after the legend-only wording refinement is `f12a4a3d083ea2701f93439e0e831b5c7b61c91ab14293531ee0afe3472b310a`. Both rebuilt analysis summaries were checked against their recorded input hashes and matched.

The exact and accelerated pilot spectra were identical for both datasets, as were the retained 2016 validation spectra. All final ensemble contracts agreed on generating truth, source identities and grids. The exact fallback coordinates are **41, 62, 75, 76 and 112 MeV for 2016**, consistently across all three ensembles. No 2021 coordinate required fallback.

| Dataset and exact comparison | Paired roots per method | Maximum absolute profiled error | Maximum absolute fixed error | Changes in `r > 0` |
|---|---:|---:|---:|---:|
| 2016 pilot, all 142 masses | 1,420 | 2.352e-5 | 3.065e-5 | 0 |
| 2016 validation, 81 retained masses | 81,000 | 2.887e-5 | 3.678e-5 | 0 |
| 2021 pilot, all 201 masses | 2,010 | 5.247e-6 | 1.176e-5 | 0 |

There are **no exact 2021 validation columns**. The final metadata explicitly records `comparison_available = false`, zero paired roots/coordinates, and `null` for maximum root error and atom flips, with no accuracy or atom-stability pass flags for that unavailable comparison. The 2016 validation comparison explicitly remains available with 81,000 paired roots at 81 masses per method. The atom-stability statement is limited to the paired spectra in this table. Paired copies are not added to the independent toy count.

The exact full-response sentinel sets matched their declarations. Recomputing their centered differences gave the following maximum discrepancies, where each relative L2 error uses the corresponding exact response-column norm:

| Dataset, method | Absolute response-entry error | Relative response L2 error | Relative width error | Absolute correlation error |
|---|---:|---:|---:|---:|
| 2016 profiled | 7.983e-7 | 8.274e-6 | 5.806e-7 | 5.464e-7 |
| 2016 fixed | 1.172e-6 | 2.929e-6 | 8.211e-8 | 1.728e-7 |
| 2021 profiled | 4.683e-7 | 1.280e-6 | 3.021e-7 | 4.583e-7 |
| 2021 fixed | 3.258e-7 | 7.572e-7 | 4.559e-8 | 4.417e-7 |

The largest accepted ordinary-stencil response-entry errors were 1.922e-6 for 2016 and 7.108e-7 for 2021. All are comfortably inside the declared gates. Recomputing the full pilot scan scores with the same fixed centering/scaling map gave maximum exact/fast principal-score differences of 4.776e-6 and 3.383e-6 for 2016 profiled/fixed, and 3.166e-6 and 1.336e-6 for 2021. On the restricted 81-mass 2016 validation overlap, the corresponding errors were 2.636e-5 and 1.381e-5. This restricted comparison does not establish full-grid exact/fast agreement for all 1,000 validation scans.

The reviewer independently reconstructed `D`, `C`, `K`, direct scan maxima, the atom-masked local probabilities, all pointwise global exceedance counts, and the reported maximum-distribution KS statistics. They reproduced the saved results. No numerical or ordering bug was found in these checks. The exact sentinel subset remains an empirical numerical audit rather than a proof for every response entry.

## Completed statistical results and interpretation

The principal ordering gives the following decomposition of each most-extreme observed point. The standardized value is a coordinate of the declared stress-centered score; it is **not** a discovery significance.

| Dataset, method | Mass [MeV] | Observed raw `r` | Asimov offset `a` | Response width `s` | `(r-a)/s` |
|---|---:|---:|---:|---:|---:|
| 2016 full, profiled | 42 | 0.041027 | -9.214926 | 0.967375 | 9.568109 |
| 2016 full, fixed | 43 | 0.964720 | -31.905332 | 3.053506 | 10.764693 |
| 2021 10%, profiled | 92 | 1.221703 | -2.176625 | 0.984398 | 3.452188 |
| 2021 10%, fixed | 77 | 8.377566 | -1.809350 | 2.196582 | 4.637621 |

For 2016, the profiled point is only barely positive before subtracting the stress mean. Its conventional raw asymptotic local probability is 0.48364, whereas the stress-centered Gaussian local extrapolation is 5.44e-22. This huge difference diagnoses the interaction of the assumed stress shape and the estimator. The fixed point similarly has raw asymptotic local probability 0.16734 and a Gaussian local extrapolation of 2.53e-27. Neither extrapolation is demonstrated by the toy sample, and neither is evidence for a particle. A small calibrated probability can reject this particular stress construction while leaving other background descriptions viable.

The bounded atom is implemented as declared: all 75 profiled and 77 fixed nonpositive observed 2016 roots have common-truth local probability one. For example, at profiled 41 MeV the standardized root is already about 6.276, but its raw root is negative, so it remains in that atom. The change to a positive raw root at 42 MeV allows the large centered score into the scan maximum. The observed 42 MeV raw root is much larger than the measured numerical discrepancies; this feature is not an observed low-rank sign flip.

| Dataset, method | GP-field principal exceedances | Direct-scan principal exceedances | Interpretation |
|---|---:|---:|---|
| 2016 profiled | 0 / 200,000 | 0 / 1,000 | Both sampled tails unresolved |
| 2016 fixed | 0 / 200,000 | 0 / 1,000 | Both sampled tails unresolved |
| 2021 profiled | 5,038 / 200,000 = 0.02519 | 26 / 1,000 = 0.026 | Agreement at a resolved few-percent tail |
| 2021 fixed | 41 / 200,000 = 0.000205 | 1 / 1,000 = 0.001 | Direct rare-tail precision remains weak |

For each zero-count 2016 result, the one-sided 95% Monte Carlo upper bound is **1.4979e-5 for GP fields** or **0.0029913 for direct scans**. The GP bound applies to the Gaussian-field approximation; the direct bound applies to the specified Poisson stress construction and fixed score. The central two-sided 95% interval upper endpoints are instead 1.8444e-5 and 0.0036821. Do not report a probability of zero. In particular, the saved `gp_global_inside_direct_interval95 = true` flag is automatically unsurprising when both observed counts are zero: it provides no validation of a 9.6- or 10.8-standardized-unit extrapolation.

For 2021 profiled, the direct central 95% interval is [0.01705, 0.03787], containing the GP estimate. This supports the joint-field approximation at this threshold and precision under the chosen stress truth. For 2021 fixed, the direct interval is [2.532e-5, 0.005559]; its inclusion of 0.000205 is a much weaker test because only one direct scan exceeded the threshold. The factor-of-five difference between the two point estimates is not by itself a statistically established disagreement, and the one direct exceedance cannot establish a precise rare-tail probability.

The separate raw ordering remains very different:

| Dataset, method | Raw-order peak [MeV] | Observed maximum raw `r` | GP global probability | Direct global probability |
|---|---:|---:|---:|---:|
| 2016 profiled | 90 | 3.424751 | 200,000 / 200,000 | 1,000 / 1,000 |
| 2016 fixed | 88 | 8.452550 | 200,000 / 200,000 | 1,000 / 1,000 |
| 2021 profiled | 78 | 2.808645 | 0.78486 | 0.778 |
| 2021 fixed | 77 | 8.377566 | 0.884545 | 0.877 |

For 2016, direct raw null maxima ranged from 11.158 to 16.332 for profiling, and 25.419 to 39.894 for fixed background. Thus every generated scan exceeded the corresponding observed raw maximum. A raw global probability estimated as one says that the one-sided maximum test does not reject this stress construction in its specified direction. It does not certify goodness of fit or consistency of the complete observed spectrum.

The maximum-distribution comparisons are:

| Dataset, method | Principal KS distance | Principal nominal p | Raw-maximum nominal p |
|---|---:|---:|---:|
| 2016 profiled | 0.045115 | 0.03377 | 0.61425 |
| 2016 fixed | 0.041445 | 0.06372 | 0.10264 |
| 2021 profiled | 0.024930 | 0.55787 | 0.59207 |
| 2021 fixed | 0.027920 | 0.41244 | 0.94580 |

The 2016 profiled comparison has a modest nominal shape discrepancy. At score 1.8334, the GP CDF is 0.351115 and the direct CDF is 0.306, so the GP construction understates the exceedance probability there by about 0.045. This is a bulk-distribution diagnostic, not evidence about the unresolved observed tail. Four dataset/method comparisons are reported per ordering; the nominal 0.03377 value is not a multiplicity-adjusted family-level rejection. Conversely, the other nominal p-values and zero marginal Holm flags do not certify an exact joint Gaussian field or its far tails. All four sets of marginal Holm flags were independently recomputed and equal zero.

The 2021 alternate 2 MeV profiled subgrids give probabilities 0.02002 and 0.020195 at the same fine-grid observed threshold, compared with 0.02519 on the 1 MeV grid. Fixed-background values are 0.000125 and 0.000155 versus 0.000205. These changes demonstrate a finite-grid effect; they do not establish convergence to a continuous mass search. The 2016 zero-count subgrid comparisons have no resolving power for that question at the observed threshold.

## Final disposition

Both datasets are numerically accepted for this separately contracted **conditional finite-grid study**. The 2021 profiled result has useful direct validation at its few-percent observed threshold. The 2016 centered-order observed tails and the 2021 fixed rare tail remain insufficiently resolved for a precise direct probability. The 2016 maximum-distribution shape tension should remain visible. None of these products establishes unconditional discovery calibration, confidence-interval coverage, expected sensitivity, or a combined search across methods and years.

The controlling numerical records are `global_fast/{2016,2021}/acceleration_validation.json`; the statistical records are `global_fast/{2016,2021}/analysis/{summary.json,pvalue_curves.csv,marginal_diagnostics.csv,covariance.npz,maxima.npz}`. The original exact checkpoints remain under `global/{2016,2021}/`. Future physical conclusions require an independently qualified background description and a declared final ordering; further conditional tail precision requires new independent, contracted full-spectrum scans, with the background and field-approximation uncertainties kept distinct.
