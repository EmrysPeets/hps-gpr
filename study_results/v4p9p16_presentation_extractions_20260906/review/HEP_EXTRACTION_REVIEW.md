# Independent HEP/statistical review of the presentation extractions

Reviewed 6 September 2026. The reviewer used saved products and read source code; no likelihood fits, toys, additional events or shared-source edits were made. Numerical identities and input hashes are recorded in `independent_final_audit.json`; the read-only calculation is reproducible with `independent_audit.py`.

## Disposition

The extraction, display-binning and exposure calculations are accepted as the declared conditional presentation study. All **2,333 independent checked conditions passed**. No remaining numerical or statistical implementation error was identified. The observed displays, illustrative future means and information-scaling assumptions remain distinct from discovery calibration, expected sensitivity and confidence-limit coverage.

## Likelihood and component reconstruction

The audit checked 15 selected fits and 24 dataset components. It compared every saved full spectrum with the previously released histogram counts, verified prediction-state hashes and selection, and reconstructed the Gaussian-bin signal shape, shared-coupling normalization, profile backgrounds and positive total expectations. The independent per-year fits remain separate from the components of a common-amplitude fit.

Using the exact saved likelihood means, counts and nuisance vectors, the audit evaluated the Poisson deviance contribution plus `theta.T @ theta / 2` for the common free fit, null fit and independent fits. The signed roots follow `sign(Ahat) * sqrt(2*(NLL_null-NLL_free))`. It also reconstructed nuisance and amplitude gradients and the profiled curvature through the nuisance Hessian's Schur complement.

| Independent numerical diagnostic | Maximum discrepancy |
|---|---:|
| Common signed root | 8.58e-14 |
| Independent signed root | 0 |
| Common NLL | 1.42e-14 |
| Common curvature standard error, relative | 2.00e-15 |
| Saved information versus reconstructed information, relative | 0 |

The largest nuisance gradient is 1.964e-7, within the original 2e-7 convergence criterion. Exact likelihood means and nuisance coordinates are now preserved explicitly; reversing tiny background displacements through small covariance modes is unnecessary. The full-grid GP calculations are not rerun or changed by this extraction.

## Selection and display integrity

The independent ranking reproduces individual peak pairs 51/21 MeV (2015), 90/117 MeV (2016) and 78/65 MeV (2021), and combined full-union peaks 66/21 MeV. The second peak with multiple active datasets is 92 MeV. The 21 MeV combined coordinate contains only 2015; it does not acquire sensitivity from more 2021 data. Deficit selections are 19, 102 and 71 MeV individually, and 72 MeV combined. The 2015 deficit is a search endpoint. The separate 76/83 MeV stress diagnostics do not replace the observed-amplitude ranking.

Every individual and common grouping matrix is binary, uses contiguous whole native bins, contains no repeated source bin within a panel, and stays inside the actual likelihood window. All exported sums, edges, yields and retained fractions reproduce. Common maps lie on the specified 1.25 MeV lattice. The signal-plus-background identity holds before and after grouping.

The common display omits different fractions of different years. At 66 MeV it retains 60.1%, 83.6% and 100.0% of fitted-window counts from 2015, 2016 and 2021. At 72 MeV those fractions are 48.0%, 68.0% and 93.1%. The displayed signal fractions differ from the count fractions; both are now explicitly tabulated in `common_display_retention.csv`. The quoted fitted amplitudes and likelihood roots use all native fit bins, so the summed display must not be reinterpreted as their likelihood input.

Counting bars are appropriate as counting-only display errors. The residuals depend on the fitted background and are not independent pulls. The zero-centered gray band is a guide to the original correlated GP constraint's width; it is not the original constraint center, a fitted-background error band or total residual uncertainty. The revised labeling makes this distinction explicit.

## What the datasets say at the selected combined masses

Define the descriptive likelihood loss as `Delta D = 2*(NLL_common-sum(NLL_independent))`. The extra independent-amplitude count is one less than the number of active datasets.

| Mass, MeV | Delta D | Extra amplitudes | Interpretation |
|---|---:|---:|---|
| 66 | 2.845 | 2 | All three individual estimates are positive; no large rate disagreement is apparent in this diagnostic. |
| 92 | 5.362 | 1 | Both estimates are positive, but their preferred rates differ materially. |
| 72 | 0.713 | 2 | All individual estimates are negative; their signed rates align reasonably in this descriptive comparison. |
| 76 | 9.405 | 2 | The 2016 estimate is negative and the 2021 estimate positive; the shared raw amplitude is near zero. |
| 83 | 2.548 | 2 | The raw common deficit is small; the large stress-centered score is driven by the reference offset. |

At 92 MeV, the independent signed rate estimates are `(9.132 +/- 2.932)e-6` for 2016 and `(1.655 +/- 1.354)e-6` for 2021, using local curvature errors. The common value `2.970e-6` is therefore an explicit illustrative hypothesis for future persistence, rather than a rate independently supported to the same degree by both years. At 66 MeV the common value is `4.550e-6`.

These masses were selected from the existing data. Nominal fixed-mass chi-square reference values are retained only as qualified diagnostics in the machine-readable audit; they are not post-selection compatibility probabilities or discovery tests. Agreement of signed deficits does not turn a negative auxiliary amplitude into a physical negative coupling. Likewise, the large 76 MeV stress-centered score cannot be interpreted as a coherent resonance when the observed rate is near zero and the years pull in opposite directions.

## Exposure sequence and inference boundaries

All saved exposure rows reproduce the specified conditional means. The new, independent 20% has means `2B10` or `2(B10+S10)`. The cumulative 30% and 100% views retain the actual first sample and use `N10+2B10` or `N10+2(B10+S10)`, and the corresponding factor-nine expressions. Their background-only future counting variances are `2B10` and `9B10`; the first observed 10% is held fixed. The background reference is the background profiled in the selected common-amplitude fit. No future background refit, uncertainty in this reference or uncertainty in the selected signal rate is included.

The signal-yield table describes a constant assumed template rate in the complete fitted window. Its total-exposure yields are not measured future yields and need not equal the sum of a conditional residual curve that retains fluctuations from the first sample. Only the first exposure column contains observed data. The future columns and their differing vertical scales are labeled accordingly.

The separate precision calculation uses the original GP mean and constraint covariance, `I_d = u_d.T @ (diag(b_d)+C_d)^-1 @ u_d`. Under the declared assumption that the entire 2021 count covariance scales with exposure and acceptance/resolution stay fixed, `I(f)=I_2015+I_2016+f*I_2021`. The original 2021 information shares are 55.27% at 66 MeV and 82.40% at 92 MeV. The resulting combined uncertainty ratios relative to the present sample are:

| Mass, MeV | Cumulative 30% | Cumulative 100% |
|---|---:|---:|
| 66 | 0.6892 | 0.4091 |
| 92 | 0.6145 | 0.3447 |

These are conditional information-scaling illustrations. Fractional systematics, background bias, changing GP constraints and selected-amplitude bias are not covered. They are not multipliers for the observed combined root, calibrated sensitivity forecasts or global probabilities.

The current note and `NEXT_STEPS.md` correctly require testing the disjoint new 20% alone before adding the selected old samples, preserving fixed locations and a declared test family, and treating cumulative 30%/100% looks as correlated. New exposures require exposure-appropriate sampling validation; the existing global-tail bank cannot calibrate them. The inherited 2016 qualifications and unresolved physical-background questions remain. Final rendered-page QA is the parent's responsibility and is separate from this numerical/statistical review.
