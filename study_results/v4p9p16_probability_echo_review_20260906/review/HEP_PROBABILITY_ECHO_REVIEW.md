# Independent probability and signal-echo review

Date: 2026-09-06. Scope: saved numerical products and source inspection. The reviewer did not optimize a likelihood, retrain a GP, generate a random field or toy, open additional data, or modify a frozen parent.

**Disposition:** the probabilities reproduce their declared conditional definitions, and the extraction displays did not change the fitted roots. The apparent incompleteness is a presentation problem with identifiable causes; the apparent extreme significance is primarily a change in reference background and ordering. It is not a large fitted signal. The proposed revised Figure 1 and separate conditional diagnostic are appropriate. Final interpretation must retain the qualifications below.

## Numerical findings

`independent_final_audit.json` passes **1,955 checks**. It verifies all 820 entries of the combined manifest and all 96 entries of the presentation manifest. All 464 original method/mass roots match the observed CSV within **4.44e-16**; the 15 extraction roots match their corresponding dense combined/individual references within **8.33e-17**. The audit reconstructs covariance from the saved Asimov responses, both direct maximum statistics from all 1,000 coherent scan vectors, every GP/direct exceedance count from the saved maxima, their binomial intervals, and the saved distribution checks. GP fields were not independently regenerated in this review.

`independent_extension_audit.json` passes **3,364 checks**. It independently checks the new all-mass direct-local/global table and correlations, and reconstructs all **116 deterministic echo likelihoods** from saved counts, constraint means/factors, nuisance coordinates, templates and expectations. Maximum NLL error is **8.88e-16**, root error **2.78e-15**, expectation error **0**, curvature relative error **1.55e-15**, and evaluated likelihood gradient **1.63e-7**. All 29 current observed echo roots match the dense individual 2021 reference exactly. Every free/null likelihood comparison is properly nested. SHA-256 input bindings and executable audit scripts accompany both outputs.

## What the two probability orderings mean

Let `r(m)` be the signed profiled likelihood root, `a(m)` the root obtained by fitting the archived common stress-background expectation, and `s(m)` the response width from the Asimov perturbations. The Gaussian approximation models

\[
r^*(m)=a(m)+s(m)Z(m),\qquad Z\sim N(0,K),\qquad
z(m)=\frac{r(m)-a(m)}{s(m)}.
\]

The frozen principal local rule is `p=sf(z)` only for **raw `r>0`**, and `p=1` otherwise. Its global statistic is the largest `z` over raw-positive coordinates, with an empty maximum equal to negative infinity. The separate raw ordering uses `max_m max(r(m),0)`. The analyzer applies these same rules to observed data and complete null fields and counts exceedances with `>=`.

These definitions are internally coherent under the specified Gaussian approximation to **one fixed common stress background**. In that exact Gaussian model, for `0 <= u < 1`, the local rule obeys

\[
P(p\le u)=\min\{u,\Phi(a/s)\}\le u.
\]

The positive-root gate is therefore conservative under that model, but creates a real probability jump: `p=1` at a nonpositive root, whereas the limit immediately above zero is `Phi(a/s)`. A large negative offset can make this jump enormous. Removing the gate or smoothing across it would define a different test.

The nominal asymptotic reference, `sf(max(r,0))`, instead equals **0.5** at nonpositive roots. Its different zero convention must be stated. It is a reference transformation of the observed fit, not a validation of its calibration. The stress-centered probability is also not the nominal reference with a look-elsewhere penalty attached: changing `a(m)` and `s(m)` first changes which masses are considered extreme.

## Why the curve looks choppy and too significant

All **232 masses from 19 through 250 MeV** were evaluated. For profiling, **117 roots are nonpositive**, with **23 sign-gate crossings**. The original local curve had a display floor of `1e-8`; the 76 and 77 MeV formal Gaussian tails fall below it. The original GP global curve intentionally omitted line segments at those same two zero-count points and showed upper-bound symbols. Direct checks were plotted only at six representative masses, despite all-mass computation. These features explain the apparent missing pieces. Membership changes are additional declared boundaries; they are not missing calculations.

Selected profiled values:

| Mass (MeV) | Raw root `r` | Stress offset `a` | Width `s` | Centered score `z` | Nominal local p | Conditional Gaussian local p | GP / direct global exceedances |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 2.516 | 0.204 | 0.956 | 2.418 | 0.00594 | 0.00779 | 93,598 / 479 |
| 41 | 0.215 | -4.300 | 0.971 | 4.652 | 0.415 | 1.64e-6 | 48 / 0 |
| 66 | 2.760 | 8.987 | 0.980 | -6.352 | 0.00289 | approximately 1 | 200,000 / 1,000 |
| 75 | -0.666 | -10.831 | 0.979 | 10.383 | 0.5 | **1 (gate)** | 200,000 / 1,000 |
| 76 | 0.166 | -8.700 | 0.979 | 9.053 | **0.434** | **6.98e-20 (formal)** | **0 / 0** |
| 77 | 0.883 | -5.599 | 0.980 | 6.616 | 0.189 | 1.84e-11 (formal) | 0 / 0 |
| 78 | 1.529 | -1.839 | 0.980 | 3.437 | 0.0631 | 2.95e-4 | 6,140 / 26 |
| 92 | 2.416 | -2.999 | 0.984 | 5.501 | 0.00785 | 1.89e-8 (formal) | 2 / 0 |

GP counts have denominator 200,000; direct counts have denominator 1,000. The largest raw profiled excess is at **66 MeV**, with nominal local `p=0.0028887`. At **76 MeV**, the fit itself is barely positive: its extreme centered score measures how far it lies above the strong deficit expected from the stress construction. It must not be described as a nine-sigma particle discovery. At 74 and 75 MeV, even larger centered scores are excluded by the raw sign gate, explaining the dramatic 75-to-76 jump.

Zero GP exceedances give a **one-sided 95% sampling upper bound of 1.498e-5** conditional on the Gaussian field model. Zero direct exceedances give **0.002991** for the specified toy truth. Neither establishes the formal `6.98e-20` local tail. At 92 MeV, two GP exceedances imply roughly 71% relative counting uncertainty. These are unresolved or sparse tails, not precision significance measurements.

The direct principal maxima range only up to **4.213**; GP maxima reach **5.600**. Their distribution comparison has nominal KS `p=0.862`, and no marginal test survives Holm correction. This supports the tested bulk approximation but does not validate a score of 9.053 or an unspecified family of backgrounds. The raw-global result is saturated because **all** GP and direct experiments have maxima above the observed raw peak; it is not a goodness-of-fit certificate. Neither ordering establishes frequentist coverage or calibrates the plotted pointwise CLs limits.

The GP method is motivated by [Ananiev and Read, arXiv:2206.12328](https://arxiv.org/abs/2206.12328), which models a significance scan as a Gaussian process using designed background-only responses. That method does not by itself establish the adequacy of this chosen stress truth or validate the extreme tail here. This analysis is a finite, declared 232-point search; additional masses, directions, methods, or selection among orderings would require the corresponding search-family treatment.

## Recommended display and caption

Lead Figure 1 with the unchanged **observed pointwise asymptotic CLs upper limit**, the **signed raw profiled root**, and the **nominal asymptotic local p-value**. Use the exact saved roots, label the latter as local and asymptotic, and retain visible membership boundaries. This directly matches what the extraction panels show.

Place the archived stress-background diagnostic in its own full-range and zoom figure showing `r`, `a`, `s`/`z` together with the conditional probabilities. Display all 232 hypotheses, identify the nonpositive-root atom, and show all direct checks or clearly label any representative subset. Keep zero-count upper-bound triangles separate from probability estimates. Estimates with fewer than 25 exceedances should be visibly sparse and should not form a precise connected tail curve. A `1e-4` analytic display floor is reasonable provided it is called a **display floor, not a validated-tail boundary**; preserve exact formal values in an audit table. Do not smooth across sign gates, join upper bounds as estimated probabilities, or imply that GP correction calibrates the upper limits.

Suggested Figure 1 caption core: “Observed pointwise asymptotic 90% CLs limits, signed profiled likelihood roots and nominal asymptotic local p-values on the complete declared mass grid. These local references share the extraction fits; no look-elsewhere correction or stress-background calibration is applied in this figure. The separate stress-background diagnostic studies different conditional null orderings and finite simulation precision.”

## What the injection replay establishes

The current dense replay uses only the archived released 2021 10% spectrum, the same smooth generating background, and pre-existing positive injection amplitudes. Reconstructed native counts, Gaussian templates, expectations and likelihoods pass the independent audit. At 71 MeV:

| Generating spectrum | Fitted root | Change from the same background-only root |
|---|---:|---:|
| Smooth background only | -0.598 | 0 |
| Plus positive 66 MeV signal | -2.243 | -1.646 |
| Plus positive 78 MeV signal | -2.423 | -1.825 |
| Plus positive 65 and 78 MeV signals | -4.177 | -3.579 |
| Observed 2021 10% data | -4.019 | not an injection change |

The 66 MeV injection also increases the fitted root at 78 MeV by **0.734**, and the 78 MeV injection increases it at 66 MeV by **0.916**. Thus a positive input can cause both positive and negative fitted echoes when it enters another hypothesis's GP training sidebands. A neighboring fitted deficit alone does **not** rule out a positive resonance. Scaling such a deterministic fit response with exposure can also make an echo persist or grow; growth alone does not identify a physical peak.

This mechanism demonstration does not establish one or two particles. The full 65+78 pair produces roots **3.395 and 3.796** at those peaks, compared with observed **2.396 and 2.809**, and its 72 MeV dip is also too deep. The amplitudes came from separate single-signal fits, not a joint two-signal inference. The current standalone low-mass injection is at **66**, whereas the pair contains **65**: do not interpret these displayed three curves as an additivity test. The earlier matched 65/78 study is the appropriate archived additivity example.

The background-regression GP responsible for these echoes and the Gaussian random-field emulator used for global null maxima play different roles. A null covariance describes correlated fluctuations under its generating background; it does not identify a particle or automatically remove signal leakage. The echo truth also differs from the combined global stress truth, so this deterministic response cannot be transported as a quantitative correction to that probability scan. A formal physical interpretation needs a separately specified simultaneous multi-template/background model and independent predictive validation, with the selected masses, directions and subsequent looks accounted for.

Both review scripts write only to stdout when executed; their saved JSON files provide the exact inputs and numerical disposition. Final plot rendering and document layout remain the parent task's responsibility.
