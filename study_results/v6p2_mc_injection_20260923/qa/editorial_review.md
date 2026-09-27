# Independent prose and scientific wording review

Review date: 23 September 2026. The reviewer compared the archived 20-toy
v6.2 PDF text and README with the v6.1 LaTeX note, then edited the revised
40-toy `source/report.tex` after an explicit handoff from its author.

## Substantive changes

- Shortened the abstract and opening scope paragraph. The abstract now states
  the final experiment and its result directly; the history of adding twenty
  toys appears once alongside the ensemble-stability comparison.
- Replaced report-like phrases and repeated limitations with connected
  explanations of the physics and fitting procedure. Each section explains
  what a quantity measures before describing its plot. Captions retain the
  normalization, error-bar definition, and marked 260 MeV extension needed to
  read the figure.
- Kept raw recovery, the paired zero-injection-subtracted increment, and the
  pull mean distinct. The text now explains why a background offset affects
  the raw ratio at small injections and why a zero mean pull need not imply
  an unbiased yield when the fitted uncertainty varies with the residual.
- Defined the yield consistently as selected candidates. The note states once
  that signal-daughter association and production-selection equivalence are
  unresolved, and once that the empirical template, center, and background
  truth are fixed in this ensemble.
- Clarified the roles of the controls. Removing the injected tail from
  sideband training isolates the GP response to that tail. Fixing the known
  background also removes the cost of estimating the background and its
  nuisance covariance; its uncertainty cannot replace the primary one.
- At the root reviewer's request, replaced the unrestricted inverse-covariance
  likelihood expression with the implemented nuisance form
  `bhat + L theta + A p`, penalty `theta^T theta / 2`, and `V = L L^T` for
  retained covariance modes. This avoids treating a numerically truncated
  covariance as an invertible matrix over arbitrary background vectors.
- Defined the profiled negative log likelihood explicitly. Profile-set
  inclusion is stated as a finite-ensemble fraction with a binomial interval,
  and the exact-N multinomial versus Poisson-likelihood distinction is kept
  beside the definition where it is relevant.
- Defined `T = 40` through the existing toy-count macro and used `NT` in the
  recovery equations, avoiding the visually ambiguous product `N40`. The
  existing width-uncertainty macro now states the approximate 11% relative
  sampling error under a Gaussian-pull approximation.

All numerical macros, generated tables, figure paths, mass and injection
choices, and fitted results were preserved. The reviewer did not edit fit
code, numerical validation, figure generation, or the copied inputs.

## Verdict

The source now reads as a scientific note with a continuous explanation of
the experiment, measured response, and uncertainty. The recurring caveat
blocks and process-heavy phrasing in the original report have been removed.
The remaining qualifications are attached to the particular claims they
limit. No unsupported efficiency, calibrated coverage, or significance claim
was found in the revised narrative.

Source and final compiled-text review passed. The 17-page PDF was checked
against `qa/report_extracted_text.txt` after the final LaTeX build. The
abstract, equations, main result paragraphs, captions, and all four numerical
tables consistently use the 40-toy ensemble. References to twenty toys occur
only in the documented extension and preservation of the original release.
The text keeps the 60--240 MeV summary ranges separate from the marked
260 MeV extension, and keeps raw recovery separate from the paired increment.
The 240 MeV example, 15/40 and 22/40 inclusion counts, binomial intervals,
and ensemble sizes agree with the generated numerical macros and tables.
The nuisance likelihood and `NT` recovery denominators appear correctly in
the extracted text. No unresolved prose or scientific-wording issue was
found. Rendered-page inspection is recorded separately by the root task.
The final page-2 clarification was checked in the source and rebuilt PDF
text: the generating GP is fitted over the full mass support of the
**2021 10% spectrum**. This replaces the ambiguous phrase "full 2021
spectrum" and preserves the intended data-exposure scope. The rebuilt
document remains 17 pages; no other narrative change was requested.

Reviewed final identities:

- `source/report.tex` SHA-256:
  `5c2203ecda8cd4dc9b0e56a5badf05b322668aad02e544527cb1f1fb17708cb5`.
- `pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf` SHA-256:
  `53ab3061727ddfa7c6a02908cfbaeba7cb3ad1085ba8dda1199d98e5b9724ac3`.
- The PDF has 17 pages; these identities match `qa/report_build.json`.
