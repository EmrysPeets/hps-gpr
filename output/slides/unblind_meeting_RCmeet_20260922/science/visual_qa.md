# Science visual QA — slides 1–20

Inspected `renders/after/slide-01.png` through `slide-20.png` in a contact sheet and full-resolution slides 3, 6, 7, 9, 11, 13, 14 and 18. First-pass live-deck renders, 2026-09-22. No deck changes made by reviewer.

## Concrete corrections

| Slide | Finding | Correction |
|---|---|---|
| 3 | The two plots overlap the bottom selection-table rows, obscuring vertex/positron-cluster content. | Separate the actual rendered table bottom and plot tops; shorten/compact rows or move plots below. |
| 6 | Parent-run sentence crosses the bottom border of the table. | Move below the actual table height, with a small gap. |
| 7 | The wrapped “MeV” of the second update overlaps the fit-support explanatory text. | Shorten updates or give heading more height and move explanatory text below. |
| 6, 7 | Caption says “Gold: tested search range”; plots actually mark boundaries with dark dashed vertical lines. | Change to “Dashed lines: tested search range.” |
| 18 | The qualification wraps “result.” onto a final line at the bottom slide edge. | Shorten to “Conditional validation; not coverage.” or move it up. |
| 11 | The cropped source plot drops the shared legend. New native legend identifies colored curves but does not explain its blue band. | Add “Band: GP constraint width.” The band must not be described as a post-fit error. |

## Passed checks

- Slide 3 explicitly labels the electron eight-hit requirement as updated selection, removes the vertex-fit chi-square requirement, and says the archived example plots were not reprocessed with the update.
- Slides 6–7 preserve 2015 19–100, 2016 39–180, and released 2021 50–250 MeV search domains; 2021 support is 36–300 MeV. These remain distinct from the proposed 56-MeV full-sample search start on unchanged slide 4.
- Slide 9 correctly separates length scale, covariance amplitude and bin-noise variance; illustrations are visibly labeled, axes and equations readable.
- Slide 11 uses the appropriate archived 2021 78-MeV extraction. The core GP/profile likelihood equations are preserved.
- Slides 13–14 correctly call the new summary Q/Nbin, retain the correlated covariance in Q, and explicitly deny calibrated chi-square/KS p-values or fitted effective degrees of freedom.
- Slide 18 exact pull denominator uses the post-injection toy error; mean/width targets and source-conditional scope are correct.

Untouched slides in this range retain existing content. Previously identified generator-family naming concerns on slides 15–16 belong in feedback, not an unrequested deck change. Existing label “profiled bkg” under Lθ on slides 10–11 could be refined later to “background deformation”; this is inherited content outside the requested plot-slot insertion.

Contact sheet: `science/qa_slides_01_20.png`.

## Second render check

Inspected `renders/final/slide-03.png`, `slide-06.png`, `slide-07.png`, `slide-11.png`, and `slide-18.png`.

- Slides 3, 7, 11 and 18 now pass: overlap/clipping corrected; slide 11 explicitly labels the GP constraint band.
- Slides 6–7 search-boundary caption is corrected to dashed lines.
- Slide 6 still requires repair: the parent-run sentence now wraps across two lines and the spectra image covers its lower/right portion. Suggested minimal repair: delete this optional sentence, or replace it by a short single line “2021 parent run: ≈160 pb⁻¹” with enough clearance above the spectra. Root notified immediately.

## Delivered-render check, pending final slide-6 clearance

`renders/delivered/slide-11.png` and `slide-18.png` pass. On `slide-06.png`, the parent-run sentence is now full width but its lower half is still obscured by the white top of the spectra image. This is a vertical collision rather than a remaining width problem. Root notified; deleting the optional sentence is the smallest reliable repair.

## Final approval of reviewed scope

Inspected fresh native thumbnail `qa/final-slide-06.png` after removal of the optional parent-run sentence. Slide 6 now passes: table, plots and correct support/search caption are separated and readable. All concrete defects identified in this review of slides 1–20 are resolved. The source/claim-boundary checks on edited slides 3, 6, 7, 9, 11, 13, 14 and 18 pass. No additional deck edits are requested by this reviewer.
