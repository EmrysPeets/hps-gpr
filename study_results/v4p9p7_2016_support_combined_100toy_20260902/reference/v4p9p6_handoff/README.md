# HPS GPR analysis note v4.9.6

Version 4.9.6 is a standalone editorial release derived from the validated v4.9.5
bundle at
`study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/`. The frozen
v4.9.5 analysis supplies the data products, cards, fits, figures, and numerical
results. This release reorganizes and edits the note without recomputing or replacing
them.

## Deliverables

The source tree builds two documents:

- `HPS_GPR_Analysis_Note_v4p9p6.pdf`: the complete analysis note.
- `HPS_GPR_Harvard_Writing_Sample_Sections_2_to_5.pdf`: a standalone fellowship
  writing sample containing Sections 2--5, with a clean cover and its own references.

The PDFs are build products and are not claimed complete until both have compiled and
passed text and rendered-page checks. See `source/README.md` for the exact commands.

## What changed

- The 2021 support-selection material is integrated as the last subsection of
  Section 5.
- The observed 2021 scan using that support is integrated as the last subsection of
  Section 6.
- Stale source-sample and GP-support descriptions are corrected.
- Prose, transitions, and typography are revised for a more natural and consistent
  presentation.

No data input, analysis card, fit, figure, or numerical result changes in v4.9.6. The
accepted v4.2 three-campaign result remains unchanged. The v4.9.5 2021 support study
remains a conditional diagnostic, while its observed scan remains asymptotic and has
no expected bands, CLs toys, direct-coverage claim, or scan-global significance
calibration.

## Continuing the note

The stable extension points are immediately after
`source/sections/05a_2021_support_selection.tex` for later validation work and after
`source/sections/06a_2021_observed_result.tex` for later observed-result updates.
Keep those additions separate from the frozen subsections and add them explicitly to
the appropriate document driver.
