# Presentation changelog

Date: 2026-09-22

[Open unblind_meeting_RCmeet](https://docs.google.com/presentation/d/1aTM2lpk9JPR6zwm20IrLnMxSadJfhNkLiRwTJR_wrng/edit?usp=drivesdk)

Edited the existing Google Slides presentation in place. Exactly 15 of 79 slides changed: 14 contained bracket instructions, and slide 7 explicitly included slide 6. The other 64 slides, slide order, masters and layouts are unchanged.

## Slide-by-slide changes

| Slide | Change | Source or interpretation |
|---|---|---|
| 3 | Added an editable selection table, including the eight-hit electron update and removal of the vertex χ² requirement; inserted electron-momentum and vertex-χ² distributions. | v5.0.5 selection table and preselection figures; plots are historical examples, not reprocessed with the new hit cut. |
| 6 | Updated the dataset table and three mass spectra to 2015 search 19–100 MeV and 2021 GP support 36–300 MeV. | Explicitly authorized by the bracket on slide 7. Search/support boundaries are distinguished. |
| 7 | Updated the range table and spectra; replaced the instruction with concise range changes and a support-versus-search explanation. | Same pinned spectra/ranges as slide 6. |
| 9 | Added separate length-scale, covariance-amplitude and bin-noise illustrations; retained/repositioned the RBF equation and defined the parameters. | Analytic illustrations clearly labeled; x=log(m), alpha approximately 1/y. |
| 11 | Inserted the 2021 78 MeV extraction plot with a complete curve/band legend. | Existing likelihood equations and profiling diagram retained. |
| 13 | Added 2021 held-out prediction examples at 60, 120, and 220 MeV, including residuals and Q/Nbin. | Replayed archived kernel states; no optimization or new signal fits. |
| 14 | Added the 201-point covariance-aware residual scan across 50–250 MeV and its equation. | Q/Nbin is explicitly not calibrated χ²/dof or a KS p-value. |
| 18 | Added the exact extraction-pull equation, symbol key, meaning of pull mean/width and reference targets 0/1. | Preserved and repositioned the original source/toy display; conditional validation remains distinct from coverage. |
| 21 | Explained the 1% × 10 and native 10% threshold studies with two saved-result comparisons and brief bullets. | Historical 40–300 / 30–300 MeV diagnostic supports distinguished from 36–300 MeV production. |
| 22 | Defined matched-reference injection strength, template addition and GP refitting; added an illustrative signal-shape display. | Reference error sets yield; it does not guarantee measured significance. Shared background toys are retained. |
| 25 | Replaced the bracket with a short interpretation of nuisance profiling and the one-sided discovery statistic. | Existing formula and Cowan citation retained. |
| 26 | Added the pointwise CLs steps and a labeled illustrative crossing at CLs=0.10. | Existing equations retained; illustration is not an observed HPS limit. |
| 46 | Explained exposure scaling, Asimov target matching, 20 Poisson toys, refitting and the R/D echo definitions. | Conditional one-region persistence scenario; original formula screenshot retained. |
| 49 | Added the correlated scan-response equations, saved 2021 correlation matrix and five saved null scans. | Cited Ananiev–Read correctly; count-background GP distinguished from scan-response GP. |
| 50 | Added the raw local versus full-scan maximum equations and displays; included 57/256 direct-refit check and interval. | Frozen-source conditional global calibration; no centering/rescaling of observed raw roots. |

## Formatting and provenance

Explanatory text and selection/range tables remain editable. Specialized equations are high-resolution rendered math. Scientific plots preserve their aspect ratios; captions identify illustrations and conditional diagnostics. Added source/interpretation notes only on edited slides.

The HPS-GPR specialist checked current v5.0.5, v5.8.5.3 and v5.9 sources; the slideshow specialist checked audience readability. v5.9 recommendations are in the separate summary because no bracket requested a tail-study insertion.

## Verification

All 79 slides were rendered and reviewed. The final source readback confirms exactly the authorized 15 slides changed. No shared master/layout changes or reordering occurred. Original bracket instructions were removed from the edited slides. Table/plot overlaps and clipped footers were corrected; the last slide 6 cleanup was verified with a fresh native Google Slides thumbnail.

No fit optimization, signal extraction or new toy ensemble was run. Slides 13–14 diagnostics replay archived GP predictions with fixed kernel states. The CSV and recipe are in science/assets/.

## Local supporting files

- Original snapshot: raw-template.json; final live snapshot: raw-delivered.json.
- Scope proof: qa/final-verification.json.
- Scientific sources and caveats: science/targeted_content.md and science/v505_sources.md.
- Build requests and final cleanup records: build/ and design/.
- Final delivered artifact is the Google Slides presentation. Rendered PDFs are working QA exports; the final slide 6 thumbnail is qa/final-slide-06.png.
