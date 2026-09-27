# Output visual QA — slides 41–79

Reviewed all slides 41–79 using four contact sheets in `design/qa/`, and inspected slides 46, 49 and 50 individually at the 120 dpi output resolution. Source screenshots were already inspected during planning. This report distinguishes new defects from inherited source content.

## Required consolidated repairs

- **Slide 49 — footer clipped:** the first citation line is visible, but the second line (“2021 native 10%; ±2.25σ; 50–250 MeV; frozen-source conditional calibration.”) falls below the canvas. The scientific qualification must remain visible. Suggested repair: move `rcmeet_s49_citation` from y376 to y358, give it height40, and set paragraph spaceBelow to0. Keep its 9.5pt source-caption size.
- **Slide 50 — footer clipped and crowded:** the domain/source qualification is below the canvas, and the citation crowds the direct-Poisson interval. Suggested repair: move `rcmeet_s50_conditional` to y340,height35, with paragraph spaceBelow0; move `rcmeet_s50_citation` to y366,height36, with paragraph spaceBelow0. Preserve equation and figure sizes.

Both slides' equations, plot axes/labels, native text and title fit otherwise. Slide46's new explanation, preserved native bullets and retained equation fit. None of the inspected source-only slides was identified as a new regression; the root agent independently verified unchanged slide structures.

## Inherited content, not a regression

Slide46's retained equation screenshot includes two tiny partial glyphs above the formula. They are present in the source image. A shallow crop of only the blank/scrap top margin would remove them while preserving all math, if included in the consolidated repair. The inherited scientific/label concerns on other unchanged slides are recorded separately in `design/feedback.md`.

## Status

Final confirmation: inspected `renders/delivered/slide-49.png` and `slide-50.png` individually. Both source/domain qualification lines now fit fully within the canvas, and slide50's direct-Poisson interval and citation have clear separation. Equations and plot axes remain readable and undistorted. No outstanding newly introduced defect remains in the reviewed slides41–79. Slide46's inherited screenshot remnants are unchanged and are not a regression.
