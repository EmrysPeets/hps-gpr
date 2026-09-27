# Targeted slide design plan

Source deck: `unblind_meeting_RCmeet`, 79 slides, 720 × 405 pt. Preserve slide IDs, order, layouts, titles and slide numbers. All source media were explicitly mapped below. This is an in-place edit; no exemplar duplication or blank slide creation is needed. Source visual inspection corrected the preliminary “white/blue” description: the deck uses a white field, burgundy serif titles and SLAC branding, with EB Garamond native text. Keep that style.

Use the existing `p7` title/content layout and the same slide as each edit's exemplar. Content normally occupies x36–684, y82–390. Keep existing title geometry. Narrative text stays at 17–21 pt when possible; short qualifications are 13–14 pt and source footers 9.5–10 pt. Render specialized equations as high-resolution intrinsic math images while leaving explanations as native text. Fit plots without cropping or unequal scaling. A figure's entire axes, labels and legend must remain visible.

| Slide | Role and targeted composition | Existing media mapping |
|---:|---|---|
| 3 | Event selection: native compact selection table on left, two representative kinematic plots on right. Highlight the 8-hit electron requirement and removal of vertex χ² requirement with concise native labels. | No source media; replace instruction placeholder. |
| 6 | Dataset overview: update table and three-panel support/range plot; retain upper-right normalized spectrum overlay. | `...1_88` replace; `...1_123` replace; `...1_87` keep. |
| 7 | Range changes: same revised table and lower spectrum panel as slide 6; replace bracket text with 36 MeV support and tentative 100 MeV 2015 search statement. | `...1_515`, `...1_519` replace; all titles preserved. |
| 9 | Three small multiples for ℓ, C and α; short native explanation per control. Keep kernel equation above figures where it remains readable. | `...0_114` keep/reposition. |
| 11 | Preserve the native labels, workflow graphic and profile equations; replace right-hand instruction with one representative 2021 extraction plot in the same space. | `...0_380`, `...0_381`, `...0_386` keep. |
| 13 | Representative 2021 10% fits with a succinct explanation of which prediction/fit state is shown. Use selected panels rather than a tiny full catalogue. | No source media; replace instruction placeholder. |
| 14 | Across-mass goodness-of-fit diagnostic plus a short, accurate definition. If only a conditional fixed-state residual diagnostic can be reproduced, identify it explicitly rather than calling it a rerun χ²/ndf scan or KS test. | No source media; replace instruction placeholder. |
| 18 | Left: exact pull equation, symbol key, mean/width targets, study strengths. Right: retained source/toy screenshot reflowed clear of the title. | `...0_284` keep; uniform fit at x365,y91,w332,h286.8. |
| 21 | Two threshold-study panels and two concise explanation groups. Explicitly separate 1%×10 and native-10% generating sources and historical support choices. | No source media; replace instruction placeholder. |
| 22 | Left: matched-reference equation and three steps; right: labeled injection illustration. Bottom: injection target versus measured significance qualification. | No source media; new equation and plot. |
| 25 | One additional interpretation beneath the existing statistic. Preserve formula and citation. | `...0_21` keep; `...0_24` citation keep. |
| 26 | Three native steps in upper-left and CLs-crossing illustration beneath. Preserve four formula images on the right. | `...0_51`, `...0_52`, `...0_53`, `...0_54` keep. |
| 46 | Four compact paragraphs explaining the conditional test, preserving native custom dash bullets. Explain R and D above the retained equation screenshot. | `...1_396` keep and enlarge uniformly to x75,y318,w596,h81.4. |
| 49 | Left: signed-root field equation and explanation. Right: saved correlation matrix and null scans. Bottom: primary paper citation and exact source/domain qualification. | No source media; new equation and archived-data figure. |
| 50 | Left: raw-local/full-scan equations and explanation. Right: observed local curve and conditional maximum distribution. Include the direct Poisson check and conditional-source footer. | No source media; new equation and archived-data figure. |

`requests-specialist.json` owns only slides 18, 22, 25, 26, 46, 49 and 50. It contains native requests, image placements, and per-slide changelog entries. The root agent owns other listed slides and all live writes. Source screenshot on slide 18 remains dense after reflow, but preserving it respects the requested minimal edit; the native validation explanation carries the presentation's main point.

For output QA: inspect the full output deck once; verify every edited slide's text fit, scientific labels, equation readability, media aspect ratio and absence of brackets. Then compare untouched slide structures against the before snapshot. Collect all concrete visual defects before one consolidated repair pass.
