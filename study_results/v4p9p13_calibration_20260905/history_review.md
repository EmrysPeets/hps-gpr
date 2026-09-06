# Historical 2021 closure and the calibration truth

Read-only review, 2026-09-05. No toys or fits were generated. Numbers below come from archived summaries; the recommended ROOT histogram and its hash/bin geometry were independently read.

## What “bias” means here

The historical endpoint is the ensemble mean of the signed pull, `(Ahat − Ainjected)/sigmahat_A`, with a Student-t interval. A resolved nonzero mean is evidence of conditional estimator miscentering under that generating truth, support, fitting procedure, and signal definition. Calling it a **conditional mean-pull bias** is reasonable. It does not measure bias in the observed spectrum or prove a unique structural cause. With a random fitted denominator it is also distinct from the unstandardized bias `E[Ahat] − Atrue`; report both in new studies.

Centering, width calibration, and exclusion coverage are separate questions. A modest mean inside an accepted practical band can still be statistically inconsistent with zero. A near-unit pull width does not establish CLs coverage. Historical injected yields were `z × sigma_A_reference` from each background toy’s matched reference fit, not one common fixed physical strength across toys. [Extraction definitions](../v4p9p5_2021_gp_support_edge_optimization_20260820/run_support_scan.py), functions `reference_attempt` and `refit_attempt`.

## What the versions actually established

**v4.9:** Twenty-five independent backgrounds per source family tested 55, 60, 65, and 70 MeV at 0, 1, 3, and 5 reference sigmas. Identical backgrounds compared 30–300 against 40–300 MeV support. At 65 MeV, moving the lower edge to 30 shifted mean pulls from +2.783 to +0.973 for 1%×10 and from +1.152 to −0.133 for native 10%. Other threshold cells failed, so the study explicitly declined a production freeze. This is strong evidence of support-dependent response under its truth, not general unbiasedness. [README](../v4p9_2021_threshold_support_qualification_20260817/README.md), [frozen specification](../v4p9_2021_threshold_support_qualification_20260817/study_spec.json).

**v4.9.1:** Two distinct 65 MeV replacements were extended to 100 backgrounds. The 1%×10 Table-17 truth with 40–300 support gave mean +0.139, 90% CI [−0.050,+0.329], width 1.140. Native 10%, using the v4.9 anchored truth with 30–300 support, gave −0.246 [−0.417,−0.076], width 1.025; its independent 75-background continuation remained −0.284 [−0.494,−0.074]. The native offset was reproducible. Both passed a subsequently adopted ±0.5 practical band, but the native lane failed the recorded zero-null screen. That tolerance was post-result, not a predeclared equivalence test. Table-17 width intervals exceeded one. [README, including cohort and interpretation qualifications](../v4p9p1_2021_background_validation_consolidation_20260817/README.md).

**v4.9.5:** Only native 10% was optimized over lower edges 30, 32, 34, 36, 38, and 40 MeV, using the same archived 100 backgrounds and masses 55–70 MeV at 0, 2, and 5 reference sigmas. No edge passed the original uniform `abs(mean pull)<0.5` requirement. A dated post-phase-1 amendment required 9/12 means and 3/4 zero-signal means below 0.75, no magnitude reaching 1.25, and unchanged numerical gates. Edges 36 and 38 qualified; the retained minimax tie rule selected 36. Independent backgrounds 25–99 confirmed without further retuning. At 36, the three 55 MeV means remained −0.773, −0.841, and −0.960; the other nine were −0.110 to +0.325. This is accepted practical recovery with explicit exceptions. [Protocol](../v4p9p5_2021_gp_support_edge_optimization_20260820/STUDY_PROTOCOL.md), [amendment](../v4p9p5_2021_gp_support_edge_optimization_20260820/STEERING_AMENDMENT_20260820.md), [decision](../v4p9p5_2021_gp_support_edge_optimization_20260820/derived/analysis/support_freeze_decision.json).

## The scaled-dataset plots combine different evidence

The v4.9.1 composite uses four exposure scenarios: 1%×10, native 10%, 1%×100, and native 10%×10. The 1% and native 10% source shapes were fitted independently; matching nominal exposure does not make their truths identical or their samples independent. Most composite points retain the v4.6 `fGenGammaThresh` truth and 40–300 support. Only the two lower-exposure 65 MeV points were substituted with the distinct threshold truths above. Consequently, the displayed 65 MeV points cannot be read as a controlled luminosity-scaling experiment.

Outside those substitutions, the archived zero-signal mean pulls are:

| Source/exposure | 90 MeV | 120 MeV | 180 MeV | 210 MeV |
|---|---:|---:|---:|---:|
| 1%×10 | −0.113 | −0.207 | +0.049 | −0.005 |
| Native 10% | −0.022 | −0.195 | +0.256 | −0.021 |
| 1%×100 | −0.217 | −0.280 | +0.116 | −0.057 |
| Native 10%×10 | −0.061 | −0.258 | +0.338 | −0.077 |

Repeated signs, notably negative at 120 MeV, are compatible with conditional response structure; these sparse grids do not isolate its cause. Hyperparameters and fitted uncertainties change with exposure, so normalized bias need not follow a simple square-root luminosity law. [Exact table](../v4p9p1_2021_background_validation_consolidation_20260817/derived/consolidated_pull_moments_90cl.csv), [historical scenario specification](../v4p9p1_2021_background_validation_consolidation_20260817/reference/v4p6_full100/study_spec.json).

## Recommended primary historical stress truth

Use the native-10% **fSigPowExpQ-anchored logistic-Chebyshev6 residual stress truth** inherited by v4.9.5. Do not call it a globally qualified fSigPowExpQ truth: v4.9 rejected every pure restricted fSigPowExpQ/fGenGammaThresh candidate. The adopted intensity uses a logistic turn-on times exp(Chebyshev degree 6) over 30–80 MeV, joined with a C2 blend over 75–85 MeV to the archived fSigPowExpQ tail through 300 MeV. Model choice used source-fit gates before extraction. Native local deviance/ndf was 1.140, but the 85–300 MeV tail remained 6.220. Its qualifications concern the threshold construction, not the entire spectrum. [Builder](../v4p9_2021_threshold_support_qualification_20260817/build_fsig_anchor_truth.py), [fit summary](../v4p9p5_2021_gp_support_edge_optimization_20260820/reference/v4p9_fsig_anchor_fit_summary.json).

Exact reusable input:

```text
study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/inputs/native10_fsig_background_toys_100.root
key: truth/fsig_anchor/2021_10pct_mean
SHA-256: 62832048711912376ab884479d59a9f00bb9f7eae2cf180b46f688300e79e383
```

It has 8000 bins on [0,1] GeV, width 0.000125 GeV, positive means on [0.030,0.300] GeV, total 141321937. Sum five native bins per production bin, then match production edges exactly. The nominal 36–300 card retains 422 bins with edges [0.03625,0.300] GeV, centers [0.0365625,0.2996875], and mean sum 141304124.03878948. Apply no additional exposure factor or post-crop normalization. The `baseline_fSigPowExpQ` key is a different truth. [Continuation builder preserving exact means](../v4p9p1_2021_background_validation_consolidation_20260817/build_continuation_toys.py).

Retain the mass-local GP mean as a model-closure control. Retain the reverse-injection smooth truth as a separate targeted diagnostic: it is data-selected, excludes 60–86 MeV, and uses the 66 MeV frozen kernel. Its 71 MeV failure cannot be directly equated to historical closure results with a different truth and fitting procedure. [Reverse-injection construction](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/reverse_injection/run_reverse_injection.py).

## Frozen-kernel bootstrap versus historical retraining

Historical reference **and injected** fits use `optimize=True`, 12 restarts per attempt, and reproducibility/max-LML/covariance/boundary gates. They update hyperparameters, log-count targets, alpha=1/y, and predictions for each pseudoexperiment. The current accelerated bootstrap updates targets, alpha, posterior mean, and covariance while holding each observed kernel fixed. That is a legitimate conditional bootstrap, but it omits kernel-estimation and selection variability; it is not a replay or calibration of the complete historical fitting procedure. Neither procedure averages over truth-model/support selection. [Historical implementation](../v4p9p5_2021_gp_support_edge_optimization_20260820/run_support_scan.py), [new core](calibration_core.py).

To attribute the 71 MeV response, the discriminating follow-up is a paired truth-by-kernel-policy comparison using identical full-spectrum counts. Until then, report **conditional miscentering under the stated truth and frozen-kernel procedure**, with empirical CLs and independent validation carrying their own scope. A global significance correction addresses multiplicity; it does not correct such estimator miscentering.
