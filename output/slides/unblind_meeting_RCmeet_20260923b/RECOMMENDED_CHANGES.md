# Recommended follow-up changes

Updated after the latest edits to **slides 13, 14 and 29** on **2026-09-23**. Numbers in **bold** refer to the current **85-slide deck**; “source” identifies the 83-slide snapshot at the start of this revision. These are recommendations for your review; no additional slide changes were applied.

Start with items 1–2 for statistical and source-label consistency, then items 3–4 for the projection discussion. The completed changes are listed in [CHANGELOG.md](CHANGELOG.md), and the [three slide-9 scripts](../unblind_meeting_RCmeet_20260923/SPEAKER_SCRIPTS.md) remain in the slide notes. New explanations are in [Reading slides 13 and 14](READING_SLIDES_13_14.md) and [Speaker notes](SPEAKER_NOTES.md).

## Resolve before presenting

1. **Slides 31–32 (source 29–30): make the significance conventions explicit.** Source 29 has an unfinished “Global” entry under “Individual Experiments.” Give the combination hypothesis, mass, local/global status, and calibration convention for each number. A raw shared-coupling local result and an older reference-standardized result cannot be presented as directly comparable. Source 30 still says “LEE penalty factor of 35” and labels both 2.8 and 1.4 “local”; identify the intended historical calculation or update the panel from the selected current source. Keep conditional raw-global estimates labeled as conditional.

2. **Slides 16–17 (source 15–16): correct the analytic-source labels.** The note assigns `fShiftSigPowTail` to the 2015/2016 year-matched seeds, `fSigPowExpQ` to the 2021 1% year-matched seed, and separately fitted `fGenGammaThresh` sources to the baseline four-exposure study. Current slide text misassigns these functions. Label each displayed panel by its actual ensemble, then use one short source key. “Scaled to 100% exposure” is clearer than “predict 100%.”

3. **Slides 48, 52, 54 and 67–73 (source 46, 50, 52 and 65–71): qualify the remaining projection claims.** The existing procedure on final slide 53 already explains a conditional, target-matched persistence scenario. Carry that scope through the surrounding slides. Replace “max significance” with “conditional full-exposure scenario” and “Have proven ability to predict” with “Reproduces the first-order response in these conditional scenarios.”

4. **Slides 68–73 (source 66–71): distinguish targets from toy medians.** The displayed 7.6, 8.88, 4.77, 3.74, 4.71 and 4.46 are rounded target significances. The saved toy medians are 7.63, 8.64, 4.63, 3.51, 4.39 and 4.23, respectively. Relabel targets or show the measured medians. **Slide 71 (source 69)** says 120 MeV but uses the 123 MeV injection. Use a shared caption defining the continuum Asimov, yield-scaled and target-matched scenarios.

## Useful clarifications if time permits

- **Slide 25:** your concurrent edit is preserved. The delivered snapshot contains a draft “==” marker. Before presenting, clarify the “pull-widths ~±10%” shorthand: give the relevant toy count and interval convention, and retain the measured width departures when summarizing validation. If keeping the ±0.5 mean-pull tolerance, identify it as a practical threshold chosen after seeing the results. The separate fixed-source offset remains documented in v5.9.5 and should not be inferred to vanish from the historical pull summary.
- **Slide 2:** separate the narrow-excess search motivation from the assumptions behind any probability calculation. Different combination hypotheses answer different scientific questions.
- **Slides 44–46 (source 42–44): separate the MC-template study from the analytic tail study.** The new MC and CDF-interpolation material addresses the earlier request for signal-shape context. State which datasets/templates and mass range the comparison covers. Truth matching and any asymmetric-mask study remain distinct follow-ups. The v5.9 core-preserving tail dilation and the MC-template substitution are different tests; changing the mask is a further change to the fit prescription. Avoid using rough curve agreement alone as the justification for unblinding.
- **Slide 56 (source 54): explain the conditional fixed-mass shift before quoting a trials ratio.** The v5.9.5 check at 78 MeV gives raw asymptotic local p = 0.00249, source-conditional Gaussian marginal p = 0.01138, and conditional global p ≈ 0.251. Comparing global with the raw local denominator mixes calibrations. A positive scan maximum is expected from selecting a maximum; the separate source-conditioned offset a ≈ 0.577 is a fixed-mass effect. The existing slide correctly labels its source and the 57/256 Poisson check; retain those qualifications.
- **Slide 52 (source 50):** shorten the long title to “Conditional scaling to the full 2021 sample.” This resolves the earlier title-spacing concern while clarifying the scientific scope.
- **Slide 70 (source 68):** remove the small clipped text remnants above the plot legend. They remain visible in the final rendered deck.
- **Slide 39 (source 37):** this is an existing blank slide. Skip it for the talk or remove it in a later cleanup. The automated structural check flags only its empty title and body placeholders.

## Earlier feedback that is now addressed

- The method slides now show the GP parameter roles, extraction example, explicit pull definition and the conditional interpretation of the residual statistic.
- Slide 13 now says **2021 10%** and defines r_i explicitly. Its plotted bin values remain unchanged.
- Slide 14 now covers **42 excluded-window centers**, with conditional toy medians and pointwise 90% bands. D_side is explicitly defined as total Poisson discrepancy from the GP. The slide explains that a typical ratio is near the toy median (~0.95), with 1 not an exact target. These remain fitted-sideband checks, not a global or withheld-bin validation.
- Training-support and threshold-study labels distinguish historical diagnostic supports from the production choice.
- The injection slide already uses a matched background reference. This revision makes the pre-injection and post-injection errors, shared toy, and input-strength meaning more explicit.
- The upper-limit explanation has its equation walkthrough. Slide 29 now uses one boxed density form of **BEST Eq. (19), rearranged**, matched to the notation/layout of the fourth-year presentation. The current plot and its existing branching correction are preserved.
- Final slide 53 (source 51) already describes the target-matched echo procedure; the remaining projection recommendations concern the surrounding claims.
- Final slides 55–56 (source 53–54) already show the response/correlation machinery and raw-local versus conditional-global calculation with the corrected Ananiev–Read citation.
- The new MC-template slides already provide signal-shape context. The remaining request is clearer scope, not another generic signal-model slide.

The two validation slides and three slide-9 spoken scripts are complete. They are documented in the changelog, rather than listed as remaining work. Numerical statements above are checked against the v5.0.5 note, v5.8.5.3 raw-significance sources, v5.9 tail study and v5.9.5 audit. No unrequested slide was edited by this content review.
