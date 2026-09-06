# HEP statistical review of the background-profiling calibration

6 September 2026. Independent review of the frozen v4.9.13 numerical products. No new toys or fits were made. All numerical statements below were checked against the released CSVs and selected saved validation toys; the companion extractor records their SHA-256 identities.

The calibration exposes a serious mismatch between one declared background shape and the fitted procedure, concentrated in 2016 and inherited by the combination. It does not establish that the observed excess is a particle or that it is an artifact. The Gaussian-profiled likelihood remains a useful candidate test statistic, but its current asymptotic probabilities cannot support a blanket final-analysis claim over the tested background family. Much of the most dramatic change is a conditional mean shift rather than a wider uncertainty distribution.

## 1. Which hump is present?

The question plausibly refers to either of two distinct features in the released plots.

**Local p-values near 65–66 MeV in the combination.** The old profiled local-p-value dip becomes a plateau near one in the two-truth calibration envelope. At 66 MeV the observed signed likelihood root remains positive, `r = 2.760159`. The parent asymptotic curve gives `p0 = 0.002844895`; the same solver used by the calibration gives `0.002888665`. This small solver difference cannot explain the change to one.

The truth-specific results identify the reason:

| Combined 66 MeV, profiled method | Local GP generating truth | All-stress generating truth |
|---|---:|---:|
| Calibrated local tail estimate | 0.001823 | 1.008729 before display clipping |
| Monte Carlo standard error | 0.000090 | 0.015986 |
| Tail effective sample size | 354.67 | 1456.56 |
| Mean signed root, 500 independent null toys | −0.10748 | +8.94236 |
| Standard deviation of signed root | 1.00603 | 0.96678 |
| Null toys with root at least the observed root | 1/500 | 500/500 |

The release takes the larger probability across these two backgrounds. Almost every no-signal experiment under the all-stress shape looks more signal-like to this fitting procedure than the actual data at that mass. The result is therefore near one. The importance-weighted estimate exceeds one by only 0.55 Monte Carlo standard errors; the collector displays it at one and retains the original estimate and `mc_boundary` flag. That small boundary fluctuation is not the reason the probability changed by orders of magnitude.

Five hundred direct toys cannot prove that the true probability is exactly one. The two-sided exact 95% interval for 500/500 is [0.99265, 1]. The much smaller GP-truth tail has only one direct exceedance, so its precise value comes from the independently constructed importance-sampling bank, not the direct 500-toy count.

**Upper limits near 72–76 MeV in 2016 and the combination.** Here the originally low upper-limit curve rises into a hump. The observed fitted signal is negative at the most dramatic points. A low upper limit produced by a deficit is not an excess of events.

| Scope and mass | Observed profiled signed root | Parent asymptotic limit on ε² | Calibrated limit on ε² | Ratio |
|---|---:|---:|---:|---:|
| 2016, 75 MeV | −2.59881 | 1.80951 × 10⁻⁶ | 4.19038 × 10⁻⁵ | 23.16 |
| Combined, 73 MeV | −3.00474 | 8.93225 × 10⁻⁷ | 1.13572 × 10⁻⁵ | 12.71 |
| Combined, 74 MeV | −1.45242 | 1.38944 × 10⁻⁶ | 1.64617 × 10⁻⁵ | 11.85 |

Under the stress background the procedure already produces large negative fitted signals with no injected signal: the mean offsets are −17.54 reference standard errors at 2016 75 MeV and −11.52 at combined 74 MeV. A positive injected signal can therefore remain hidden below a prediction that is too high under that generating model. The calibrated limit permits a larger signal accordingly. This is the other sign of the same mass-dependent interpolation problem.

Both features are visible in the original [local-p-value plot](../../v4p9p13_calibration_20260905/figures/local_pvalues.png), [2016 limit plot](../../v4p9p13_calibration_20260905/figures/limits_2016.png), and [combined limit plot](../../v4p9p13_calibration_20260905/figures/limits_combined.png).

## 2. The dominant contribution is the 2016 stress construction

At 66 MeV the profiled background-only fitted-yield offsets, expressed in each scope's fixed reference standard error, are:

| Scope | Measured toy mean offset | Deterministic interpolation projection |
|---|---:|---:|
| 2015 stress | +0.3230 | +0.3229 |
| 2016 stress | +12.4036 | +12.4161 |
| 2021 stress | +1.1013 | +1.1124 |
| Combined all-stress | +8.9380 | +8.9855 |

These individual standardized numbers cannot be added directly because the combination fits one shared coupling with different signal conversions and uncertainties. Their scale nevertheless identifies 2016 as the dominant issue. The agreement with the saved zero-noise residual projection shows that random Poisson fluctuations are not needed to produce the offset under this truth. Its presence follows from passing that smooth spectrum through the frozen-kernel sideband interpolation and signal extraction.

The 2016 alternative is a degree-five logistic-times-exponential-Chebyshev fit to a pre-existing 10% development subset, joined to an archived `fShiftSigPowTail` continuation over 75–85 MeV and normalized to the full-sample count. Its event-level independence from the full sample was never established. The broad component has archived `fit_ok: false`, with an explicitly restricted waiver for use as a conditional stress shape. Earlier source-recovery tests already showed alternating offsets and failed support qualification. These facts are recorded in the [2016 source clarification](../../v4p9p7_2016_support_combined_100toy_20260902/SCIENTIFIC_SCOPE_CLARIFICATION.md) and [2016 study README](../../v4p9p7_2016_support_combined_100toy_20260902/README.md).

Thus the result diagnoses an incompatibility between a stress construction and this conditional fitting procedure. It is not a measurement of bias in the observed data. Nor does it establish that this stress shape is the actual background. The degree of agreement with the data, its source limitations, and the GP's ability to recover plausible alternatives must be assessed independently. Removing the shape because it weakens the observed result would make the model choice depend on the answer.

The saved offset projection does not isolate whether the mismatch originates in fixed hyperparameters, regularization, support boundaries, or the generating construction. A paired truth-by-kernel-policy comparison is needed to separate those contributions. Historical closure reoptimized the kernels; this calibration keeps the observed kernels fixed.

## 3. Profiling and asymptotic calibration are different choices

The profiled fit lets the estimated background vary coherently within its covariance while determining a signal yield. An asymptotic formula then assigns probabilities to that fitted statistic. Calibration changes the probability calculation using repeated experiments. It does not refit the observed spectrum with a different background at every calibrated p-value point.

The distinction is standard: Cowan et al. derive likelihood-ratio approximations from Wilks/Wald assumptions and explicitly require a sufficiently adequate model of the truth. Their discovery statistic sets downward fitted signals to zero, while upper-limit statistics test a different ordering. [Cowan et al., sections 2–3](https://arxiv.org/html/1007.1727v3).

There is no general reason to abandon asymptotic profiling in HEP. There is concrete evidence against applying its present uncorrected mapping throughout this analysis. In the complete independent validation suite, profiled fits alone have:

| Generating truth | Excess local-rejection flags | Excess true-yield-exclusion flags |
|---|---:|---:|
| Local GP | 50/456 cells | 179/912 cells |
| Archived stress | 140/456 cells | 326/912 cells |

These are cell-level exact-binomial tests after the release's Holm adjustment across the complete relevant families; they are not counts of individual rejected toys. The calibrated two-truth procedure has zero such flags. Across both likelihood methods the raw totals are 952/1824 local-rejection flags and 2039/3648 exclusion flags.

The failures even under the local-GP generating control matter. The posterior mean is estimated from fluctuated sidebands by a regularized log-GP; retraining on its own mean need not reproduce that mean. A nearly unit distribution width is insufficient when its center shifts, and a zero-centered distribution with a wrong width is also insufficient. The current toy likelihood uses a data-derived GP constraint and fixed observed kernel states, so the generic success of profiling elsewhere is not a validation of this implementation.

The previous positive latent-log-GP comparison is informative. At identical solver settings it changed the observed 2021 limits by at most 0.232%. The peak–dip structure remained. Simply replacing the Gaussian count constraint with the positive log-GP form does not resolve the present source-recovery problem. [Profiling comparison](../../background_profile_comparison_20260905/README.md).

## 4. Is there a loss of sensitivity?

Most profiled observed limits change modestly, with large localized exceptions:

| Scope | Median calibrated/profiled-asymptotic observed limit | Median calibrated fixed/profiled observed limit |
|---|---:|---:|
| 2015 | 1.087 | 1.446 |
| 2016 | 1.085 | 1.333 |
| 2021 | 1.040 | 1.199 |
| Combined | 1.216 | 1.346 |

The ratios compare observed upper limits, not expected experimental sensitivity. Some calibrated limits become stronger: at 2021 78 MeV the profiled limit changes from 6.32597 × 10⁻⁶ to 5.96709 × 10⁻⁶ even while its local p-value increases from 0.002475 to 0.010296. Discovery tails and CLs upper limits answer different questions.

The saved injected-signal toys do establish conditional local detection power at particular strengths. For combined 66 MeV, the fractions passing the local `p0 < 0.05` threshold are:

| Generating truth and injection | Raw asymptotic | Calibrated two-truth envelope |
|---|---:|---:|
| GP, no signal | 22/500 | 0/500 |
| GP, 2 reference standard errors | 275/500 | 0/500 |
| GP, 5 reference standard errors | 499/500 | 0/500 |
| Stress, no signal | 500/500 | 26/500 |
| Stress, 2 reference standard errors | 500/500 | 333/500 |
| Stress, 5 reference standard errors | 500/500 | 500/500 |

This is a substantial conditional loss of local-test power under the GP truth. It buys protection against a stress background that would otherwise trigger a false signal in essentially every experiment. The raw test's apparent power has no defensible size control over that declared family. At this mass the two backgrounds imply such different fitted offsets that the envelope is conservative under one of them. More calibration toys would improve numerical precision but would not remove that separation.

These counts condition on the selected calibration bank and fixed physical injections; they do not incorporate the bank's Monte Carlo uncertainty. Zero of 500 has an exact 95% binomial interval [0, 0.00735] under that conditioning. The injection size is defined by a fixed reference error, not by a measured discovery significance. A fair method comparison should use predeclared expected-limit quantiles and/or signal-rejection probabilities at matched, validated type-I error, retaining each truth separately. The present review adds no expected-limit bands.

The original 2021 71 MeV reverse-truth diagnostic should also remain separate. Its profiled five-reference-error injections excluded the true signal in 145/500 asymptotic fits. The dense retrospective calibrated follow-up gave 3/500; 131 of those 500 profiled decisions lacked its stated tail/decision precision. The result is strongly conservative under that selected shape, with limited decision precision, rather than proof of uniform coverage. [Reverse-truth follow-up](../../v4p9p13_calibration_20260905/note/reverse_truth_validation.tex).

## 5. Display effects and finite Monte Carlo limits

Three mechanisms have different meanings:

1. For nonpositive observed signed root, the empirical bounded-statistic convention gives `p0 = 1`, whereas the old asymptotic display gives `p0 = 0.5`. This explains many flat upper segments: 233/456 profiled points use the empirical bounded atom. It cannot explain combined 66 MeV, where the observed root is positive.
2. An importance-weighted probability can be slightly above one. The release only displays it at one when it lies within three Monte Carlo standard errors, retaining the original value and an open marker. Combined 64–67 MeV are examples. Their high probabilities are supported by direct toys.
3. Small-tail or endpoint precision can be limited. The original release has 445/456 profiled and 454/456 fixed endpoints passing all gates. The p0 status counts differ: 204 resolved, five limited-MC, fourteen MC-boundary, and 233 bounded-atom points for the profile. A resolved upper limit does not automatically resolve a local p-value tail.

The local-p-value plotting floor of 10⁻⁵ applies to visual display of very small asymptotic values; downward triangles identify them. It is not a toy-imposed lower p-value bound. The calibration uses importance-weighted full-spectrum Poisson draws, so its estimates are not simple exceedance counts divided by the number of validation toys. The sampling identity is described in [Berns](https://arxiv.org/abs/2303.11290), which develops mixture reuse for Feldman–Cousins intervals; this release applies the weighting identity to its CLs tails.

## 6. What a final analysis needs

The current calibrated study is a useful conditional diagnostic and a stronger basis for interpretation than the raw asymptotic curves. A final physics result needs the following work, selected without optimizing the observed answer:

1. **Qualify the background family and recover its relevant structure.** Assess the 2016 source fit, join, normalization and interpolation residuals, and compare predeclared plausible alternatives. Retain the present failures and source qualifications. Show truth-specific results beside any envelope.
2. **Declare the complete fitting policy.** Either justify conditioning on a fixed kernel policy or rerun kernel estimation and every material support/model choice in repeated experiments. Use paired spectra to distinguish these effects from truth construction. Resolve the separate 2016 numerical exception.
3. **Validate local inference at the required accuracy.** Test centering, widths, tails, signal response and true-yield exclusion on independent controls, including boundary and transition regions. Use additional calibration/validation precision where the required inference demands it. Zero Holm flags in 500-toy cells is limited evidence: the first rejection thresholds of the current families are 81/500 against 10% exclusion and 48/500 against 5% local rejection.
4. **Compare expected performance at matched error control.** Use identical spectra and physical injection strengths for candidate methods; compare both useful power and false-rejection behavior. A smaller observed upper limit is not a selection rule.
5. **Include common experimental systematics and the relevant joint alternatives.** The all-three result currently includes exactly the all-GP and all-stress cases; it does not test the six mixed assignments. Their exclusion from the family needs a scientific justification or an explicit extension.
6. **Estimate global significance with coherent whole scans.** One pseudoexperiment must supply every tested mass from the same full spectrum. A mass-local toy ID does not create this correlation. The local-GP truth itself changes with mass in the current calibration, preventing those pointwise controls from defining one coherent scan ensemble.

A Gaussian-process model of the scan statistic can accelerate the last step once the chosen local statistic and common null are specified. It cannot repair the local offsets above. Subtracting a truth-specific mean from signed roots defines a useful conditional diagnostic, but it changes the ordering relative to the released bounded test wherever a nonpositive raw root would have forced `p0 = 1`. Retain that atom when describing the same test, or label the centered-root scan as a separate pilot statistic. Ten complete scans can check execution and coarse correlation behavior; they cannot certify rare global tails or justify Gaussian tail extrapolation by themselves.

## Reproduction and evidence

Run from the repository root:

```bash
python3 -B study_results/v4p9p14_interpretation_global_20260906/review/build_diagnosis.py
```

The read-only extractor verifies the three released summary hashes and the numerical-audit flags, then checks 43 selected coordinates, 172 null cells, and 516 validation cells against saved toy-ledger hashes and mean values. It writes:

- `diagnosis_summary.json`: headline numbers, scope metrics, flag counts, exact source hashes and interpretation boundaries.
- `selected_observed_points.csv`: original and calibrated observed quantities, both parent and same-solver comparisons.
- `selected_null_diagnostics.csv`: direct toy means, widths, quantiles, skewness, kurtosis and observed-threshold exceedance counts. Higher moments are descriptive and do not certify tails.
- `selected_validation_decisions.csv`: separate generating-truth, method and injection counts, with the original checkpoint path.

The principal implementation references are [the truth/retraining and empirical-tail code](../../v4p9p13_calibration_20260905/calibration_core.py), [bounded-statistic inversion and independent validation](../../v4p9p13_calibration_20260905/run_calibration.py), and [envelope construction and display-status handling](../../v4p9p13_calibration_20260905/collect_results.py). The complete conditional scope is frozen in the [calibration protocol](../../v4p9p13_calibration_20260905/PROTOCOL.md). The comparison masses used here were selected retrospectively to explain the user's plots; they are not new independent model-selection controls.
