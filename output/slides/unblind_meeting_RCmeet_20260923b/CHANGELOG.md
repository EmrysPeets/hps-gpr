# Presentation follow-up — 23 September 2026

Updated [unblind_meeting_RCmeet](https://docs.google.com/presentation/d/1aTM2lpk9JPR6zwm20IrLnMxSadJfhNkLiRwTJR_wrng) in place. Slide numbers remain those of the current 85-slide deck. Our requests target only slides **13, 14 and 29**.

| Slide | Applied changes |
|---:|---|
| 13 | Changed the dataset label to **2021 10%**. Added r_i to the residual axis and explicitly defined it as the standardized data–GP difference in mass bin i. Explained the combined counting/GP uncertainty, zero as agreement, and the green ±2 band. Reused all 1,208 saved bin values without new fitting. |
| 14 | Replaced the three-center/single-anchor display with **42 excluded-window centers**: 50–250 MeV in 5 MeV steps, plus 78 MeV. Displayed the conditional median and pointwise central 90% band from the same 256 saved toy spectra at each center. Defined **D_side as total Poisson discrepancy from the GP**, and N_side as the number of scored sideband bins. Added visible guidance on typical values and speaker notes explaining the formula and a numerical example. |
| 29 | Replaced three separate conversion equations with **one boxed BEST Eq. (19) rearrangement**, using the notation and left-equation/right-plot layout from your earlier fourth-year presentation. Added a compact symbol key and linked paper citation. Preserved the current limit plot and its transform exactly. |

## Interpretation added to slide 14

- A typical ratio is near the conditional toy median, **0.9436–0.9572**, rather than automatically 1.
- Observed D_side/N_side spans **0.8399–0.9588**. All 42 values lie within their own pointwise 90% bands.
- The centers are correlated, and those bands are not a global goodness-of-fit test. Both unusually large and unusually small values can merit follow-up.
- This evaluates fitted sidebands, with the source and archived kernel states held fixed. It does not establish predictive calibration in the excluded window or propagate source/kernel-selection uncertainty.

[Reading slides 13 and 14](READING_SLIDES_13_14.md) explains r_i and D_side with the equations and a 100-predicted/110-observed example. [Scientific findings](science/findings.md) give the exact scan protocol, tail intervals and reproduction instructions. The expanded calculation took 55.8 seconds, used one BLAS thread, and preserved parent-study hashes.

## BEST equation and earlier slide

The boxed density form follows [BEST Eq. (19)](https://arxiv.org/pdf/0906.0580#page=5) after substituting N_rad = f_rad(dN_bkg/dm)δm and canceling δm. It uses N_sig^up for the full-template 90% yield limit and the phase-space-aware N_f = 1/BR(A′→e⁺e⁻). The preserved plot already includes that branching correction.

The inspected style reference was [4th_year_presentation_final, slide 13](https://docs.google.com/presentation/d/1zIMHUyeQhD-DipRmfFCufFdLxBs1-Zf5XVYKYY8Kf6U/edit#slide=id.g272063acd90_1_986). Its old confidence level and historical exclusion claims were not transferred. Detailed source mapping is in [content/slide29_recommendation.md](content/slide29_recommendation.md).

## Verification

All 85 slides were rendered. The three instructed slides were visually checked, including the final D_side definition. Native request targets confirm that our writes affect only slides 13, 14 and 29; masters and layouts remain unchanged. Concurrent user edits on slides 6, 7 and 15 were preserved. The other 79 slides have identical rendered pixels relative to the refreshed starting export. User changes to slides 3 and 25 that predated this revision were also retained.

Evidence: [native verification](verification.json), [pixel verification](pixel-verification.json), [delivered render](renders/delivered/presentation.pdf). The structural checker only flagged the existing blank slide 39. Verified native revision: `j-D0jevy84SVwg`.

[Updated recommendations](RECOMMENDED_CHANGES.md) retain the unresolved review items using current slide numbers. [Previous changelog](../unblind_meeting_RCmeet_20260923/CHANGELOG.md) records the earlier eight-slide update and two insertions.
