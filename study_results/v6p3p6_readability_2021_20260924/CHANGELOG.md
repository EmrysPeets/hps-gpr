# Standardized residual panels and combined morphed-template scan — 25 September 2026

- Added a standardized residual context panel to all four observed extractions, with blind-region shading and search/core markers.
- Removed duplicated in-figure reading text and the low-mass qualification from plot headers; explanations remain in captions and prose.
- Appended the four-page v6.3.9 combined scan, comparison curves, campaign contributions and joint background-toy check.
- Preserved prior releases and all original scientific data; validated numerical replays, restored Gaussian reference, portable rebuilds and rendered pages.

## 25 September 2026: observed morphed-template extraction appended

Added Appendix J and Figures33–37: neighboring v16 TC templates from60 to240 MeV;181 observed mass hypotheses under three extraction models; upper-limit and local-p comparisons; the67/79/185 MeV excess regions and226 MeV deficit with full fit-component plots. Selected masses have1,000 background-only toys and200 signal-calibration toys per grid yield. All543 observed fits and40,800 toy extractions passed. The60–80 MeV interpolation and local post-selection scope are explicit. The deficit's empty yield set remains visible. Earlier43pages and scientific results are preserved.

## 25 September 2026: signal-leakage follow-up appended

Added Appendix I (pages 39-42), with four report-sized figures and tables covering nine masses. It distinguishes the wider blind window, actual MC tail leakage, hypothetical Gaussian leakage, and the paired measurement of contamination-induced yield loss. Both GP mean and functional form background sources are included. The original 38-page scientific content is retained; the reading guide identifies the new appendix. The preceding release is preserved under `history/before_leakage_appendix/`.

## 25 September 2026: v6.3.7 template and window study appended

Added Appendices D-H: v16 TC core-and-tail interpolation, omitted-mass validation, eleven window/template candidates, independent calibration and evaluation, and interpolation-calibration mismatch checks. Seven new figures have standalone uncertainty definitions and How to read the figure captions. All 58,800 unique toy-fit rows passed numerical checks. The study shows improved yield response and modest average precision gains, without uniformly better upper limits or negligible sideband leakage.

The first-page reading guide identifies the new appendices. The original main numerical results, observed fits and A-C shape studies are preserved. The preceding release is in `history/before_v637_appendix/` under the output directory. Reproducible v6.3.7 source, inputs, checkpoints and QA are embedded in `appendix_v637/`.

# v6.3.6 editorial revision — 24 September 2026

Parent: HPS_GPR_v6p3p5_Reproducible_Study.zip. Numerical inputs/results/calibration are unchanged.

1. New opening synthesis and reading guide; explicit definitions of toy experiment, selected yield, fitted Gaussian yield, pilot scale and independent evaluation.
2. Section 1 adds the saved logarithmic TC core-shift law beside the unchanged HPS image, the eleven-sample Gaussian/signal-MC catalogue, and the aligned common-core/tail comparison.
3. Distinguishes generated 60 MeV signal MC from the requested low-mass 65 MeV region, and fitted core widths from reference extraction widths.
4. Replaces pole-centered wording with central mass Gaussian and shifted Gaussian; injection and observed axes identify generated or tested mass.
5. Labels background-only mean pull and fitted-yield bias; spells out fixed GP mean and functional form toy sources.
6. Adds calibrated-response panels from frozen corrected-yield summaries. Preserves uncorrected fitted/injected and signal-induced-change diagnostics with equations.
7. Prints uncertainty definitions on figures with bars and gives figure-specific captions, including the separate signal-MC resampling convention.
8. Replaces affine terminology with calibrated yield and pull; explains covariance and approximation limits.
9. Explains exclusion ranks versus background-only excess probabilities, frozen-table variability, empty/zero-only accepted sets, and the inherited coupling-display conversion.
10. Numbers figures, expands captions and preserves standalone provenance, source data, report-only rebuilding and separate current QA.

## v16 signal-MC appendix addition

- Verified v16 TC and UC branch categories in nearby S3DF productions; extracted 22 complete selected-mass histograms and flow normalization from 48 ROOT files.
- Applied the v6.1 local Gaussian-plus-pedestal core algorithm, 32 bin-resampling replicas, fit-span checks and four mass-shift descriptions.
- Added matched TC and UC appendix layouts with shift/model comparison, eleven-sample catalogue and common-core/tail diagnostics.
- Recorded selection and smearing differences; retained the existing inference results and did not calculate UC upper limits.
- Preserved the previous release and extended source-data provenance, rendered QA and the independent-directory rebuild.

## Explicit Gaussian-center leakage comparison (25 September 2026)

- Added page 43 / Figure 32 to Appendix I: central-mass versus shifted Gaussian leakage at nine masses, using identical widths, blind bins and full normalization.
- Made the existing shifted-Gaussian reference explicit in Figures 29-31 and associated text.
- Verified that all original Gaussian leakage values already used the shift; preserved all existing fit, toy and limit results.
- Added reproducible probability integrals, a CSV, standalone figure and validation; preserved the prior 42-page release.
