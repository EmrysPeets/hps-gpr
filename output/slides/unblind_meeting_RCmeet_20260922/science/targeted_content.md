# Targeted scientific content and plot provenance

All source paths are repository relative. Assets live in `output/slides/unblind_meeting_RCmeet_20260922/science/assets/`. No fits or new toy ensembles were run. Figures are either regenerated from saved numerical inputs, transferred from the v5.0.5 note, or explicitly labeled illustrations.

## Slide 9 — three GP controls

Asset: `slide09_hyperparameter_roles.png`.

- Length scale ℓ sets how far a fluctuation is correlated in log mass.
- Constant C sets the covariance amplitude; it is not an event-count normalization.
- α is bin-noise variance: larger α reduces that bin's influence. The analysis uses α ≈ 1/y.

Equation if room: `k(x,x′) = C exp[−(x−x′)²/(2ℓ²)]`, `x = log m`.

The third panel is the exact one-training-point posterior response weight C/(C+α), used only to illustrate noise weighting. All panels are analytic illustrations, not HPS fits. Source: v5.0.5 `source/sections/04_methodology.tex`, kernel and preprocessing sections.

## Slide 18 — validation metrics

Equation asset: `equation18_pull.png`.

`p_t(z) = [Â_t(z) − A_inj,t(z)] / σ_A,t(z)`.

- Pull mean tests the yield offset; the reference is zero.
- Pull width tests the returned uncertainty scale; the reference is one.
- These are conditional extraction tests, not direct confidence-limit coverage.

The denominator is the uncertainty from that toy's post-injection extraction. This is distinct from the reference error that sets the injected yield. Keep signed fits for these diagnostics. Source: v5.0.5 `04_methodology.tex`, matched-reference diagnostics.

## Slide 21 — threshold studies are distinct

Asset: `slide21_threshold_comparison.png`; it replots exact table means and 90% intervals. Each panel is one requested explanation, all at 65 MeV and 100 backgrounds.

**1% × 10**

- Fitted native-1% mean, scaled to 10%-equivalent counts.
- Replace the baseline at 65 MeV with the degree-five threshold continuation; GP support stays 40–300 MeV.
- Background-only mean pull: +0.789 → +0.139. Pull width remains mildly broad (1.140).

**Native 10%**

- Independently fitted native-10% source; same nominal exposure does not imply the same source shape.
- Degree-six threshold continuation includes the low-mass shoulder; this diagnostic uses 30–300 MeV support.
- Mean pull: +0.716 → −0.246. Width 1.025; a small residual offset remains.

Short shared footer: “Historical 65 MeV diagnostics; selected production support is separately 36–300 MeV.”

Source: v5.0.5 `source/sections/05_toys_validation.tex`, Table `v491-sixtyfive-results` and threshold-model paragraphs. The shaded ±0.5 band is a post-result practical tolerance, not a calibrated confidence/coverage band.

## Slide 22 — matched-reference signal injection

Assets: `equation22_injection.png`, `slide22_matched_injection.png`.

1. Fit each background-only reference with the same support, binning, masks and GP settings; record σ_A,ref.
2. Set A_inj = z σ_A,ref for z = 0, 1, 3, 5 and add the full Gaussian template.
3. Refit the sideband GP and repeat the signed signal extraction, allowing background adaptation.

Footer: “The reference error fixes the injected yield before injection; z is not a guaranteed measured significance.”

Important exact convention: reference uncertainty is matched to source, toy index and mass, not necessarily one universal error for the entire ensemble. Use toy-level post-injection σ_A,t only in the pull denominator. The four strengths share background realizations and are correlated. Source: v5.0.5 `04_methodology.tex`, equations `v4p9-matched-reference-axis`, `v4p9-pull` and `v4p9-paired-response`.

## Slide 25 — discovery statistic

One added statement: “Profile the background in both hypotheses; a fitted excess gives q₀ > 0, while a deficit is assigned q₀ = 0.”

Keep local/asymptotic qualification. Existing Cowan citation is supported by [Cowan et al., EPJC 71 (2011) 1554](https://arxiv.org/abs/1007.1727), verified 2026-09-22.

## Slide 26 — upper-limit construction

Asset: `slide26_cls_crossing.png`, a clearly labeled illustrative Gaussian-Wald example with Â = 0. It is not an observed HPS profile. Its CLs is 2[1−Φ(A/σ_A)] and it crosses 0.10 at 1.645 σ_A.

- At each tested signal yield A, refit the correlated background nuisance parameters.
- Convert the profile statistic into CLs = CLs+b / CLb.
- Increase A to the crossing CLs(A90) = 0.10; convert the yield limit to ε² using the dataset normalization.

This is a pointwise fixed-mass limit. It does not receive the discovery mass-scan trials penalty. Source: v5.0.5 `04_methodology.tex`, statistical inference/CLs sections.

## Slide 46 — conditional full-sample echoes

Prose to explain the preserved screenshot:

- Scale the fitted 10% continuum to full exposure and inject one candidate at a time.
- Choose the signal yield so its GP Asimov significance matches the √10 scaling target; then draw 20 Poisson spectra and rerun the moving-window analysis.
- Compare A90 with and without injection. R(m) = A90,injected(m)/A90,continuum(m); R < 1 shows a neighboring tighter limit, and D = 1 − R(m_echo) is its fractional depth.

The injected feature can enter nearby training sidebands and lift the local background estimate, producing a deficit and a tighter limit away from the injection. Label this a conditional persistence scenario. It uses no full-2021 observed spectrum, and target matching is imposed by construction. Direct yield scaling and significance matching are distinct.

Source: v5.0.5 `source/sections/v505_projections.tex`; numerical scenario table in `derived/v505_projection_ten_table.tex`.

## Slides 49–50 — correlated significance field

External source: [Ananiev and Read, JINST 18 (2023) P05041](https://arxiv.org/abs/2206.12328), verified 2026-09-22. It supplies the significance-field GP approach; the offset/scale response here is HPS-specific.

**49: build the null response**

Assets: `equation49_response.png`, `slide49_response_correlations.png`.

- A background GP predicts counts; a second GP approximates how the fitted scan fluctuates.
- At fixed generating spectrum B, linear response gives r* ≃ a + Dᵀξ, ξ ∼ N(0,I).
- The covariance Σr = DᵀD includes shared-bin and sideband responses; Rᵢⱼ = Σr,ij/(sᵢsⱼ) is the displayed correlation.

Plot uses the actual saved 2021 correlation matrix and first five saved Poisson-refit raw-root curves. It is not a newly simulated example. The supplied NPZ key `K` is R (unit diagonal), **not** raw covariance Σr.

**50: compare a full-scan maximum**

Assets: `equation50_mapping.png`, `slide50_local_to_global.png`.

- Preserve raw observed ordering: Zlocal(m) = max[r_obs(m),0], with conventional local p = 1−Φ(Zlocal).
- For each null field take T* = max_m max[r*(m),0].
- Count fields whose maximum exceeds Tobs to obtain the conditional global probability.

2021 native-10% example, 50–250 MeV, 0.5 MeV grid, ±2.25σ window:

- Raw local peak: 78 MeV, Z = 2.808645, p = 0.002487524.
- Fixed-source Gaussian raw-maximum global p = 0.25092 from 200,000 saved fields.
- Direct full-refit check: 57/256, 95% interval [0.1732, 0.2786].

Footer: “Fixed observed-data-derived source; conditional calibration. No observed-root centering or rescaling.”

Do not present (r−a)/s as the look-elsewhere correction. Earlier reference-local plots did use this transformation in all datasets and combined; the new raw display does not. These are different orderings of the same observed fits.

Source: `study_results/v5p8p5p3_raw_significance_20260921/inputs/fields/2021.npz`; `source/raw_significance.tex`; `source/raw_peak_table.tex`; `source/raw_global_table.tex`.

## Feedback for untouched slides — do not silently edit

- Slides 15–16: functional-form names differ between historical validation ensembles. v5.0.5 says four-exposure baselines use fGenGammaThresh; year-matched 2015/2016 seeds use fShiftSigPowTail and 2021 1% uses fSigPowExpQ. Avoid one source label applied to all studies.
- Slides 29–30: distinguish one-shared-coupling raw local 2.760 from earlier reference-local combinations. “Individual experiments: 3.9 sigma” refers to a different reference-calibrated combination, not the same likelihood with a trials correction. A single LEE factor of 35 is not the current source-conditional raw-global construction.
- Slides 41,45,47: change future “max significance” and “proven prediction” phrasing when revising; conditional persistence and first-order response studies do not establish a maximum achievable significance or calibrated forecast.
- Slides 62–67: the displayed 7.6,8.88,4.77,3.74,4.71,4.46 are scaling targets (rounded), not all saved toy medians. Actual medians are 7.63,8.64,4.63,3.51,4.39,4.23. “120 MeV” is injected at 123 MeV in the numerical catalogue.
- v5.9 could support the signal-model discussion on a future instructed slide: holding the Gaussian core fixed while dilating tails by 10/20/30% yields median primary-window limit changes about 0.168/0.345/0.529%. These are medians, not maximum changes. Guard-window responses change the fit itself. Current instructions do not authorize adding this to an untouched signal-model slide.
