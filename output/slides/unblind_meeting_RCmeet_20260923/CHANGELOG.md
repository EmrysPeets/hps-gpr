# unblind_meeting_RCmeet — revision of 23 September 2026

Updated the [native Google Slides presentation](https://docs.google.com/presentation/d/1aTM2lpk9JPR6zwm20IrLnMxSadJfhNkLiRwTJR_wrng). This revision addresses the latest explicit slide instructions, with the HPS-GPR specialist supplying source checks and plots and the slideshow expert supplying explanations and speaker scripts.

The deck now has **85 slides**. “Original” below means the 83-slide snapshot taken at the start of this revision, not the older 22 September numbering. Slides 1–14 keep their numbers; original 15–23 move forward one; original 24–83 move forward two.

| Original | Final | Changes |
|---:|---:|---|
| 9 | 9 | Clarified ℓ, C and count-noise α. Added three complete spoken options, about 30, 60 and 90 seconds, to speaker notes and the scripts document. Explained that C and ℓ are optimized, while α follows the count-noise rule. |
| 11 | 11 | Replotted the same 78 MeV example. The lower axis explicitly subtracts the unprofiled GP mean, with units of thousands of events/MeV. Added a colored legend defining data, profiled background, total fit, signal and constraint shading; preserved the likelihood equations. |
| 13 | 13 | Replaced selected examples with the full search ranges of all three datasets. Added standardized residuals, a green ±2 reference band and a typeset residual definition. Each bin uses a local prediction excluding its ±2.25σ neighborhood; neighboring residuals remain correlated. |
| 14 | 14 | Replaced the displayed held-out Q curve with sideband-only Poisson deviance outside the exclusion window. Added a selected 78 MeV comparison with 256 refitted conditional toys and its finite-sample tail interval. Retained the previous Q investigation in v5.9.5. |
| New before 15 | 15 | Added **Background validation strategy**: analytic source toys, repeated extraction, and checks of bias, returned errors, recovery and support dependence. |
| 21 | 22 | Removed the words embedded beneath the figure, preserving the historical 65 MeV caption and the plotted numerical results. |
| 22 | 23 | Replaced “matched reference fit” shorthand with the same background toy and mass before injection. Explained that the reference yield error fixes the injection, and that the fitted yield may be positive or negative. Kept the measured significance distinct from the input strength. |
| New after 23 | 25 | Added **Validation: evidence and remaining limits**: recovery, mean pulls, width departures, the distinct v5.9.5 null offset, and the remaining source-uncertainty and coverage questions. |
| 26 | 28 | Added a four-step walkthrough beside the four existing upper-limit equations. Explained the physical zero-yield boundary. Removed the illustrative CLs curve to make room for readable explanations. |
| 27 | 29 | Put the signal-yield and ε² equations to the left of the existing plot, with symbol definitions and the physical branching correction. Preserved the plot and its relative aspect ratio. |

## Scientific interpretation

- **Slide 13:** the line joins overlapping local GP predictions with archived kernel states; it is not a single global fit. The green band is a visual reference, not a calibrated simultaneous or coverage band.
- **Slide 14:** at 78 MeV, 237/256 source-conditional toy refits have at least as large a sideband deviance as the data; the 95% binomial interval is 0.887–0.955. This measures fitted sidebands at a previously selected mass. It does not validate prediction inside the excluded region or establish unconditional background adequacy.
- A secondary binned cumulative shape distance, excluding the blind region, was also computed: 186/256 toy distances exceed the observation. Its calibration repeats the fit procedure. A distribution-free KS probability would be inappropriate for these fitted, binned counts. Full definitions and exact values are in [science/findings.md](science/findings.md).
- The separate [v5.9.5 report](../../pdf/v5p9p5_null_bias_20260922/HPS_GPR_v5.9.5_Null_Bias_Study.pdf) retains the fixed-mass null-offset investigation. A positive scan maximum and a nonzero fixed-mass signed mean are different effects. The 78 MeV mean of 0.548 remains explicit on final slide 25.

## Verification and preserved scope

- Native structure: 85 slides in the requested order; only original slides **9, 11, 13, 14, 21, 22, 26, 27** changed. The other **75 original slides**, masters and layouts are unchanged, apart from automatic page-number updates and expiring image URLs.
- Rendered all 85 slides. Inspected all ten revised/new slides; checked the repaired equation/legend placements again. All 75 unchanged slides have **identical rendered pixels outside the page-number corner** compared with the initial snapshot.
- The structural checker reports only two pre-existing empty placeholders on final slide 39, the existing blank slide. No unrequested cleanup was applied.
- Verified the 78 MeV profile replay against the saved raw signed root. Parent scientific inputs were hashed before and after calculation; no new Poisson spectra or hyperparameter optimizations were needed for this revision.
- Evidence: [native scope comparison](qa/scope-verification.json), [pixel comparison](qa/pixel-verification.json), [scientific protocol](science/provenance/protocol.json), [final rendered PDF](renders/final/presentation.pdf). Final native revision: `wBBoUr3XuU46lg`.

## Review files

- [Updated recommended changes](RECOMMENDED_CHANGES.md) — remaining suggestions, using the final 85-slide numbering.
- [Speaker scripts](SPEAKER_SCRIPTS.md) — three options for slide 9 and walkthroughs for the revised method slides.
- [Previous presentation changelog](../unblind_meeting_RCmeet_20260922/CHANGELOG.md) — the earlier bracket-instruction pass, retained as history.

The before/after native snapshots, request ledgers, reproducible scientific plots, exact plot data and asset sources are retained in this directory. Recommendations are separate from the applied edits.
