# Independent HEP/statistical review of the v4.9.16 combined runner

Reviewed 6 September 2026. The reviewer generated no toys, performed no likelihood fits, and changed no shared numerical source. Review-owned files are this document and `independent_audit.py`. The runner reviewed has SHA-256 `e68a2d7086263a1abef2355b9ae71401fecce5b2e9d9c52f13f1f83be02d0cfd`.

## Disposition

The declared combination and the reviewed implementation are consistent. No membership, shared-coupling, Asimov-embedding, response-cancellation or cache-key error was found. All 232 coordinates and the final assembled products passed the independent numerical audit. The saved GP/direct probability calculations were also independently reproduced. Both principal observed tails remain unresolved by the available simulations. Numerical acceptance and the conditional distribution checks do not qualify the generating background or establish unconditional discovery calibration or confidence-limit coverage.

## Membership and likelihood

`scope_for` selects the existing scope with exactly the datasets whose declared search ranges include the tested mass. This gives 232 integer coordinates: 2015 alone at 19–38 MeV; 2015+2016 at 39–49; all three at 50–90; 2016+2021 at 91–180; and 2021 alone at 181–250. The 90 single-dataset coordinates are reused and the 142 overlapping coordinates are fitted jointly. Membership is determined by support, not an observed curve.

For joint points, `MemoContext` inherits the actual `Context` count-scale likelihood. Its concatenated signal template and total fitted-window yield conversion define one common coupling, with independent dataset nuisance blocks. It does not combine individual signed roots. The auxiliary unrestricted common-amplitude fit supplies the signed root, while the discovery/upper-limit definitions retain their physical nonnegative-signal boundary.

Joint observed limits are obtained from dense `Context.ofit` before enabling the approximate prediction backend. Single-dataset endpoints come from the matching v4.9.13 dense asymptotic results. Both are pointwise bounded, piecewise-asymptotic CLs limits; the stress-centered GP scan does not recalibrate them. Raw and displayed epsilon-squared values are retained, and the dimuon display factor is applied once. The v4.9.12 comparator is evaluated in matching raw units.

## Coherent spectra and response basis

The runner concatenates equal-index, independently seeded yearly full spectra. The ten pilots and 1,000 validation scans remain separate; they are not pooled into a larger validation count. The same complete joint spectrum is used at every mass. The fixed joint generating mean is the concatenation of the existing per-dataset stress means, which is checked against each newly constructed joint context.

The global bin order is 2015 indices 0–483, 2016 indices 484–1203, and 2021 indices 1204–1625. At a joint mass, selecting Asimov rows `[0, active_indices+1]` and the corresponding columns gives the baseline plus exactly one perturbation for every active bin. Filling the output with the baseline and inserting those active roots gives exactly zero centered response for every inactive bin. The same embedding is used for reused single-dataset columns.

Both sides of each response comparison subtract their own unperturbed baseline. Thus the gate tests the difference between centered exact and approximate response vectors, rather than only comparing raw roots. Full-response references use the declared six masses 39, 49, 50, 90, 91 and 180 MeV. Their exact columns are embedded in the same 1,626-bin basis before comparison of correlations. This preserves correlations across membership changes through any shared dataset; it does not splice independently generated segment maxima.

## MemoContext prediction caching

The cache key contains the dataset-part index, predictor identity, nuisance compression setting and exact training-count bytes. The covariance conditioning and factor construction are otherwise copied from the parent calculation. Within this runner, a context's masks, grid, configuration and predictors' internal kernel quantities are fixed. Exact and approximate predictors have distinct identities; restoring the exact backend restores its own entries. Only one low-rank construction is attempted for each context.

Excluding the moving blind-window counts from the prediction key is correct: these counts are not GP training inputs. A perturbation in that window leaves the prediction unchanged while its changed count still enters the likelihood through `whole[:, ctx.mask]`. Likewise, changing one dataset's spectrum can reuse unchanged predictions for the other datasets. `concatenate` and `block_diag` assemble copies, so downstream likelihood arrays do not mutate the cached entries. These properties make the memoization an exact reuse of the same calculation in the current context lifecycle.

The runner also checks bitwise equality with the uncached parent retraining method at the exact Asimov baseline. Its per-toy scalar/batch checks, exact pilot comparisons and exact response probes provide additional downstream numerical evidence. Cache hits themselves are a performance measure, not a statistical validation.

## Fallback and final acceptance

A failed inherited low-rank gate restores the exact backend. A failed pilot comparison, atom classification, response comparison or approximate evaluation likewise triggers exact evaluation of the entire coordinate across all ensembles. A failed exact reference halts execution; no offending toy or mass is removed. Rejected approximation records are retained, so a final verifier should judge the saved final roots against exact references rather than require rejected pre-fallback records to pass.

The runner records the six-column correlation requirement and any v4.9.12 investigation flags for final validation. `verify_combined.py` now reconstructs the final sentinel correlations and blocks unresolved v4.9.12 excursions. It also attests the complete frozen v4.9.14/v4.9.15 manifests, including the upstream numerical gates used by reused approximate single-dataset results.

The older v4.9.12 comparator CSV was absent from the numerical-run contract. This was resolved without changing the in-flight cohort: `provenance/observed_reference.json` separately binds that CSV to SHA-256 `6f60467b8051ac23d6b7d357d7f325d0fea6be0f6e184497a7c94769ae6e9adc` and its frozen parent manifest. The final verifier checks that binding. This source affects the comparator and investigation trigger, not the newly fitted likelihood or toy ensemble.

## Independent saved-product audit

`independent_audit.py` imports no fitting runner, draws no random values and writes no files. It independently checks source hashes, same-ID spectrum concatenation, the entire Asimov perturbation definition, membership, checkpoint identities, exact single-column reuse, inactive response zeros, paired joint pilot roots and atom classification, separately centered response errors, and available full-sentinel correlations. With `--require-complete`, it additionally requires the complete grid and sentinels and checks all assembled columns and product hashes.

The final `--require-complete` audit passed **7,744 checked conditions across all 232 coordinates**, with no failures. Its stdout is preserved as `independent_final_audit.json`. All six declared full-response sentinels and all assembled columns were checked. The main product verifier separately passed 2,736 checks. The only new joint coordinate using exact fallback is 41 MeV; no v4.9.12 investigation flags remain.

| Final numerical diagnostic | Profiled | Fixed |
|---|---:|---:|
| New joint pilot roots compared with exact references | 1,420 | 1,420 |
| Maximum absolute paired pilot-root error | 2.099e-5 | 1.700e-5 |
| Paired positive-root classification changes | 0 | 0 |
| Maximum centered response-entry error | 8.077e-7 | 1.138e-6 |
| Maximum complete-response relative L2 error | 3.368e-6 | 4.728e-6 |
| Maximum complete-response relative width error | 4.786e-7 | 2.625e-7 |
| Maximum absolute six-sentinel correlation error | 3.833e-7 | 4.653e-7 |

The 1,420 paired roots per method are ten exact joint pilot evaluations at each of 142 newly fitted masses. They do not constitute an exact re-evaluation of all 1,000 joint validation scans. The 90 single-dataset coordinates retain their separately attested parent numerical evidence. All discrepancies above are inside the declared gates; they remain empirical numerical checks, not a theorem bounding probability error in an extreme tail.

Run the final read-only audit with:

```sh
python3 study_results/v4p9p16_combined_global_20260906/review/independent_audit.py --require-complete
```

## Final global-probability audit and the 76 MeV feature

The reviewer read the completed `global/analysis/` products without drawing new Gaussian fields or Poisson toys. Reconstructing both response covariances, direct scan maxima, atom-masked local probabilities, all 232 pointwise GP/direct exceedance counts for both methods, representative selections and normality/maximum-distribution diagnostics reproduced the saved results. All analysis input hashes matched. The inspected analyzer SHA-256 is `778de2599e0d97e5ea263f28b3fa86be8bd790fd4d127822b2374c92e5c98451`; the main verifier SHA-256 is `41807ba8c89cfd33994a4efd4b867053d695bd79fac7224b6ceeb93ca30c4862`.

Both principal extrema occur at 76 MeV:

| Statistic | Raw observed root | Stress Asimov root | Response width | Standardized score | Raw asymptotic local p | Gaussian-response local p |
|---|---:|---:|---:|---:|---:|---:|
| Profiled | 0.165569 | -8.700091 | 0.979354 | 9.052563 | 0.434248 | 6.983e-20 |
| Fixed | 0.175478 | -15.348100 | 2.164129 | 7.173131 | 0.430352 | 3.665e-13 |

The raw fitted excess at that coordinate is small. The large standardized score arises because the specified stress background produces a strongly negative mean signed root under the estimator. It therefore probes the compatibility of that particular background construction with this score. It is not a 9.05-sigma particle discovery, and the extremely small analytic Gaussian local probability is not resolved by the simulations.

The position of the feature also depends on the declared nonnegative-signal boundary. At profiled 74 and 75 MeV, the standardized values are still larger, about 10.199 and 10.383, but the observed raw roots are -1.452 and -0.666. Those points have local probability one because their raw roots are nonpositive. At 76 MeV the raw root becomes positive, so its large stress-centered value enters the principal scan maximum. At 77 MeV it remains extreme, with standardized score 6.616. This sharp local-probability change follows the specified boundary and offset; it is not an observed numerical sign instability.

For each method, no principal exceedance occurred in **200,000 GP fields or 1,000 direct joint scans**. The one-sided 95% Monte Carlo upper bounds are respectively **1.4979e-5** and **0.0029913**. Their central two-sided 95% interval upper endpoints are instead 1.8444e-5 and 0.0036821. The GP bound concerns the Gaussian-field approximation; the direct bound concerns the declared Poisson stress scenario and fixed score. Neither includes physical-background uncertainty. The saved `gp_inside_direct_interval95 = true` flag is not evidence validating the observed extreme tail when both exceedance counts are zero.

The profiled direct principal maxima ranged from 1.055 to 4.213; the sampled GP maxima ranged from 0.901 to 5.600. These ensembles therefore do not resolve the observed score of 9.053. Agreement of their central distributions cannot establish the extrapolation to that score. As a less extreme descriptive comparison, the profiled 78 MeV threshold has GP count 6,140/200,000 and direct count 26/1,000; it is covered by the existing validation at roughly percent-level tail probabilities. This example is not an additional selected test or a change to the representative-mass list.

The raw ordering gives a different result. Its profiled observed maximum is 2.760159 at 66 MeV, while every GP field and every direct scan has a larger raw maximum. Direct raw maxima range from 6.734 to 12.519. The fixed observed raw maximum is 11.359992 at 22 MeV, again exceeded by every simulated scan; its direct raw maxima range from 16.860 to 30.475. Both complete raw-global curves consequently have the estimate one. This says that the positive-maximum statistic does not reject this stress construction in that direction; it is not a goodness-of-fit certificate.

At 66 MeV the profiled stress Asimov root is +8.987, compared with the observed +2.760, explaining why the original positive-root feature is unexceptional under this stress distribution. The contrasts between 66 and 76 MeV are driven by the oscillating stress offsets and the chosen ordering, rather than a choice of a more favorable significance or an established change in sensitivity.

| Maximum-distribution diagnostic | Profiled | Fixed |
|---|---:|---:|
| Principal KS distance | 0.01890 | 0.01244 |
| Principal nominal KS p | 0.86249 | 0.99743 |
| Raw-maximum nominal KS p | 0.63894 | 0.20342 |
| Marginal Holm normality flags | 0 | 0 |

No nominal discrepancy is identified by these particular diagnostics. They have finite power and do not certify Gaussian far tails. The RMS response/direct correlation differences are 0.03114 and 0.03033. Both alternate 2 MeV subgrids also have zero sampled exceedances at the original observed principal peak; this provides no resolved convergence test there. The search remains the declared 232-point union, without additional correction for choosing methods, widths, kernels or other searches.

## Report definitions and final scope

The final note distinguishes the stored unit-electron-branching coordinate from physical coupling: `epsilon_eff^2 = epsilon_phys^2 * BR_ee`, with the display conversion applied once. It defines profiled NLL, componentwise Gaussian roots and both maximum orderings explicitly. The main figure identifies pointwise asymptotic CLs limits, separates zero-tail bounds from estimates, distinguishes local display-floor markers from global Monte Carlo bounds, and labels direct comparison points as belonging to the principal ordering. The representative table includes both raw asymptotic and Gaussian-response local probabilities, which makes the stress-centering effect visible.

The largest observed-limit difference from the matching v4.9.12 raw curve is 2.5017%, inside the declared investigation threshold. The displayed observed limits range from 8.019e-7 to 3.223e-5; their minimum occurs at 72 MeV. These are observed pointwise limits, not expected sensitivity. The GP calculation neither modifies their endpoints nor supplies the signal-plus-background distributions needed for calibrated CLs limits.

The completed products are accepted as a conditional numerical/statistical study under one archived joint stress scenario. Their 2016 source-fit, development-overlap, transition and inherited numerical qualifications remain. Physical background qualification, expected sensitivity, confidence-interval coverage and a continuous-mass or additional-search correction remain separate work. The user's requested full-region figure and combined GP study can be reported with these boundaries intact.
