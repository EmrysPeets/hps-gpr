# Outline And Claim Status

## PRD Organization

This draft follows the compact rhythm of recent HPS PRD papers, especially the SIMP
paper: short abstract, no table of contents, concise numbered sections, a small number
of decisive figures, and auxiliary validation material moved out of the body.

1. Introduction
   - Visible dark-photon motivation.
   - Previous HPS prompt searches and broader HPS dark-sector program.
   - Why Gaussian process regression is the method advance.

2. The HPS Experiment and Data Samples
   - Detector-era summary.
   - 2015, 2016, and 2021 sample scope.
   - Dataset table and selected mass spectra.

3. Event Selection and Normalization
   - Prompt event selection in compact form.
   - Mass-resolution parameterizations.
   - Radiative fraction and signal-yield to `epsilon^2` conversion.

4. Gaussian Process Background Model and Statistical Method
   - Scan grid and `2.25 sigma_m` blind/extraction window.
   - Log-count GPR model and resolution-scaled kernel bounds.
   - Non-renormalized signal template.
   - Profile likelihood, local `p0`, `CLs`, observed vs expected, and shared-coupling
     combination.

5. Validation and Systematic Checks
   - Training exclusion and blind-window choice.
   - Functional-form and full-refit closure tests.
   - Systematic checks.
   - Global-significance calibration status.

6. Results and External Comparisons
   - Staged 95% shared-`epsilon^2` result.
   - 90% suite for external contour comparison.
   - Discovery diagnostics.
   - Projection-only full-data contours.

7. Conclusion
   - Conservative status statement and bounded remaining work.

## Main-Text Figure Set

- Prompt visible-dark-photon context panel.
- HPS detector-era comparison.
- Input invariant-mass spectra.
- Mass-resolution summary.
- GPR method schematic.
- Blind-window validation figure.
- Toy upper-limit proxy.
- Combined staged 95% limit.
- Combined 90% comparison figure.
- Individual/combined diagnostic overlay.
- Combined local `p0`/`Z` diagnostic.
- Projection-only phase-space context.

## Appendix / Auxiliary Figure Set

- GPR validation flowchart.
- Per-dataset closure suites.
- Representative observed extraction display.
- Limit-tail diagnostics.
- 90% vs 95% confidence-level comparison.
- 2021 scaled-resolution comparison.
- Projection-only comparisons to published 2016 HPS and BaBar.

## Claim Gates

Current safe claim:

The folder supports a PRD-style draft for the HPS prompt dark-photon GPR workflow. The
staged result combines 2015, 2016 10%, and 2021 1% samples in a shared-`epsilon^2`
likelihood and documents the analysis method, validation strategy, and comparison
figures.

Do not yet claim:

- A final full-statistics observed exclusion.
- A discovery or final global significance.
- A final systematic uncertainty model.
- A final 2016 or 2021 unblinded result.

Before submission:

- Freeze the full data samples and validated mass ranges.
- Freeze the radiative-fraction and systematic-penalty model.
- Regenerate observed/expected limit figures from final production outputs.
- Replace effective-trials global overlays with scan-level toy calibration.
- Decide whether the paper quotes 90% CL, 95% CL, or both, and make external contours
  consistent with that convention.
- Reconcile the 2021 target-constrained resolution scaling treatment.
- Resolve the 2016 low-mass boundary convention before quoting the final scan range.
- Audit and document any expected-band repairs in the 90% suite.
