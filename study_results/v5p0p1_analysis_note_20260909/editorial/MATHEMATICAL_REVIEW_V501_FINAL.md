# Final mathematical source review of the v5.0.1 draft

Review scope: independent read-only audit of the revised methodology, background-profile update and significance-GP section against the pinned implementation. A parallel bounded check covered the published-method comparison. No fits, toys, probability estimates, code, figures or manuscript files were changed by this reviewer. This review document is the sole new output of the final check. Rendered-page and compiler QA belong to the root integration pass.

Status at the snapshot below: the revised common-coupling likelihood and significance-field equations agree with the pinned implementation. Six local notation or endpoint corrections were sent to the root editor; the primary-source correction to the 2015 description was sent to the writer. Completion of those changes is not implied by this initial review snapshot.

## Required manuscript corrections

Line references in this section refer to the exact reviewed file hashes listed below.

1. **Hessian typo**, `source/sections/04_methodology.tex:549`: replace `\lambda_{d,i}^{,2}` with `\lambda_{d,i}^{2}`. The nuisance Hessian is \(L_d^T\operatorname{diag}(n_{di}/\lambda_{di}^2)L_d+I\).

2. **Distinguish physical and electron-channel coupling**, `04_methodology.tex:641-667`, with consequences at `1005-1029` and `1130`. The general yield equation contains the physical coupling divided by the inverse electron branching fraction, whereas the following conversion equation omits that factor. Use \(\epsilon_{\rm phys}^2\) and \(N_{\rm eff}^{\rm BR}\) in the general relation; then explicitly define \(\epsilon_{ee}^2=\epsilon_{\rm phys}^2/N_{\rm eff}^{\rm BR}\) and state that subsequent fit equations abbreviate this electron-channel coordinate as \(\epsilon^2\). Retain the later mass-dependent branching map. A BR superscript also distinguishes this quantity from the historical effective-trials symbol.

3. **Density-window wording**, `04_methodology.tex:669-677`: the phrase “same mass interval used for the nominal blind window” conflicts with the immediately specified density half-width 1.64 versus extraction half-width 2.25. Call it the separately declared prompt-density interval. It remains based on the observed histogram, not the fitted GP mean.

4. **Explicit finite-support normalization**, `04_methodology.tex:590-602`: the raw Gaussian bin integral does not mathematically sum to exactly one on a finite histogram. Define \(g_i=\int_{e_i}^{e_{i+1}}\phi(m)\,dm\), then \(w_i^{\rm full}=g_i/\sum_{j\in\mathrm{full}}g_j\), before the already correct unrenormalized window restriction. The pinned `template.py:19-45,180-212` explicitly performs these two steps. The correction expresses the existing implementation; it does not change yield definitions or results.

5. **Zero-exceedance convention beside the earlier global estimate**, `04_methodology.tex:1112-1123`: the displayed empirical fraction followed immediately by inverse-normal conversion would otherwise assign infinite significance at zero exceedances. Mark the fraction as an estimate and reference the one-sided sampling-bound convention of the later GP-global section. No exact probability zero or infinite significance should be inferred.

6. **Zero-statistic atom in the asymptotic tail convention**, `04_methodology.tex:785-800,835-850`: inclusive toy tails \(P(\widetilde q\ge q_{\rm obs})\) are both one at \(q_{\rm obs}=0\), while the pinned continuous asymptotic helper gives \(CL_{s+b}=1/2\) and \(CL_b=\Phi(\sqrt{q_A})\) there. State the displayed continuous-tail equivalence for positive observed statistic and make the zero/tie convention explicit if describing the endpoint. The actual 90% and 95% roots cannot occur at this zero-statistic endpoint because the helper's ratio there is at least one half. This is mathematical bookkeeping, not a solver modification request.

## Verified revised mathematics and claim boundaries

- The centered Poisson negative log likelihood plus \(\theta^T\theta/2\), total-rate positivity, unbounded nuisance coordinates and signed auxiliary signal fit match the reviewed implementations. The count-space GP covariance describes uncertainty on the inferred smooth mean; counting fluctuations remain in the Poisson term. The manuscript correctly keeps the GP mean/covariance fixed within a local profile and distinguishes prior retraining of a complete pseudoexperiment.
- The notation \(C_d=C_{d,\rm eff}=L_dL_d^T\) correctly identifies the numerically conditioned covariance. The text properly qualifies near-zero solver safeguards and avoids calling the constraint an independent auxiliary measurement.
- The Wald explanation correctly describes a local quadratic profile and approximate normal estimator. The boundary branch \((A^2-2A\widehat A^{\rm unc})/\sigma_A^2\) and both piecewise Asimov tail coordinates match the pinned helper for positive statistics. The manuscript states that the actual profiles are numerically evaluated and the probability map remains asymptotic.
- Independent nuisance blocks permit addition of profiled NLLs at a fixed common coupling. The revised combination evaluates every denominator at the same common physical best fit. This repairs the former ambiguity with separately normalized channel likelihood ratios. The block-vector representation and active-dataset membership match the released combination.
- The signed root, deterministic offset, finite one-bin response directions, \(\Gamma=D^TD\) without a sample-count divisor, marginal widths, response-derived correlation matrix and Gaussian vector draws match the archived significance-GP implementation. The offset is not mislabeled as the exact nonlinear Poisson mean.
- The principal ordering preserves the positive raw-fit gate and standardizes relative to the declared reference. Global maxima use the same full search grid; pointwise global curves use the corresponding observed coordinate as threshold against that same maximum distribution. Raw-root maxima remain a separate declared ordering.
- The conditional global section's zero-count one-sided 95% bound, \(1-0.05^{1/N}\), is correct. The 200000-field and 1000-direct-scan bounds are sampling statements under their respective models, not validated particle-tail significances.
- The combined 232-mass grid, independent campaign streams paired into 1000 complete joint experiments, 1626 support directions plus one reference scan and cross-membership correlations are correctly described. The text does not splice pointwise toys or segment maxima.
- The profile-comparison prose retains conditional signal-recovery and coverage failures, including the problematic 71 MeV repeated-exclusion result and separate 2016 numerical exception. The GP-global prose keeps stress-shape qualification, approximation discrepancies and unresolved rare tails explicit.

## Published-comparison check

One additional source correction was sent to the writer: `source/sections/v501_published_comparisons.tex:16-19` must call the original 2015 background an **exponential of a Chebyshev polynomial**. The primary 2015 paper Eq. (2) gives \(P=\mu\phi+B\exp[p(m;\mathbf t)]\), with \(p\) the first-kind Chebyshev polynomial. Its original full windows are 14 resolutions below 39 MeV and 13 above, distinct from the corrected recast parameters. Source: [HPS 2015 paper, Section IV](https://arxiv.org/html/1807.11530#S4).

The reviewed ratios, historical v4/factor-8 and v4.1/factor-12 vintage labels, separation of literal published 2015 from corrected recast, explicit exponentiated 2016 Legendre form and 3.4%/7.4% systematic prescriptions match the earlier source audit. The comparison correctly avoids claiming coverage or a demonstrated sensitivity improvement from observed-limit ratios. The original internal-note error history remains supported by the archived local summary; this reviewer did not independently retrieve the internal-note PDF.

## Source identity and scope

All 13 source hashes in `editorial/math_sources.json` were rechecked during this pass and were unchanged. That manifest pins the older manuscript and exact implementation; it is not a hash manifest of the newly edited draft. The additional finite-template source used in this pass is recorded below. The revised draft hashes identify only this review snapshot; subsequent editor changes require a final recheck.

Review snapshot UTC: 2026-09-09T19:07:37.745445+00:00

- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/04_methodology.tex`
  SHA-256: `755e9a9dd239f54c6023e8afb36f2014a94040b1b5b1bc76374372142eb50a06`; 1249 lines.
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/v5_method_updates.tex`
  SHA-256: `92caf93becd4593b22d09a7057b793a8443fb30652da86d8f145febad2a5a0f0`; 94 lines.
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/v5_global_significance.tex`
  SHA-256: `c600d1e44c28fe286a8c4952801cb13a8932c74cfd36bc8d2324908903dbac3d`; 173 lines.
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/v501_published_comparisons.tex`
  SHA-256: `4343a10e0c62967fa39304f651e72d650e20f7aa435898ad2daf47c7344fd006`; 212 lines.

Additional implementation source: `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v4p9p7_2016_support_combined_100toy_20260902/runtime_combined/hps_gpr/template.py`

SHA-256: `20c1fbaa632d5e03fa7527d0e4ddf8dc3ba8573927a8f981936721a731440e3e`; reviewed lines 19-45 and 180-212.


## Final resolution and packaging status

Verified UTC: 2026-09-09T19:15:47.121726+00:00.

**All six mathematical corrections and the original-2015 wording correction are resolved.** This final resolution supersedes the pending-fix status of the initial review snapshot above. No further required manuscript changes were found in this bounded recheck.

| Correction | Resolution in current source | Status |
|---|---|---|
| Hessian denominator | 04_methodology.tex:549 now uses the squared Poisson mean without a comma. | Resolved |
| Coupling coordinates | 04_methodology.tex:642-660 distinguishes physical and electron-channel coupling, defines the branching ratio factor, and states the abbreviation used in later fit equations. | Resolved |
| Density interval | 04_methodology.tex:674-682 identifies the separate prompt-density interval and retains the 1.64 versus 2.25 distinction. | Resolved |
| Finite-support template | 04_methodology.tex:593-616 separates raw integrals, full-support normalization and unrenormalized window slicing. | Resolved |
| Zero exceedances | 04_methodology.tex:1136-1140 gives the one-sided 95% sampling bound and explicitly forbids infinite significance. | Resolved |
| Zero-statistic ties | 04_methodology.tex:788 and 856-861 restricts the tail identification to positive statistic and documents both the implemented continuous endpoint and inclusive empirical convention. | Resolved |
| Original 2015 background | v501_published_comparisons.tex:16-17 now correctly states exponential of a Chebyshev polynomial. | Resolved |

The shortened final paragraph of `v5_global_significance.tex` preserves the requirements for a justified background/kernel policy, independently validated local probabilities, and sufficient complete mass/model-search simulation. The result remains conditional diagnostics and sampling bounds. Its covariance/order equations and the reviewed numerical table retain their prior definitions and values.

All 13 implementation/source SHA-256 entries in `math_sources.json` were checked again and remain unchanged. The current reviewed manuscript identities for packaging are:

- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/04_methodology.tex`
  SHA-256: `755e9a9dd239f54c6023e8afb36f2014a94040b1b5b1bc76374372142eb50a06`; 1249 lines.
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/v5_method_updates.tex`
  SHA-256: `92caf93becd4593b22d09a7057b793a8443fb30652da86d8f145febad2a5a0f0`; 94 lines.
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/v5_global_significance.tex`
  SHA-256: `fce3fea136d28efe6ed041f6716122e0ab4669d0936c1f266a8d6fa4a724341e`; 174 lines.
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p1-20260909/study_results/v5p0p1_analysis_note_20260909/source/sections/v501_published_comparisons.tex`
  SHA-256: `716f3fb3da76915f6b4643322e9274e31af7652cd7bff8982be99fa218409178`; 203 lines.

This is a mathematical and source-consistency signoff for the stated scope. It does not add statistical calibration, alter frozen results, authorize unblinding, or replace the root build and rendered-page QA. No fits, toys or numerical inference were rerun. Only this editorial review file was appended during the resolution check.
