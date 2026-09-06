# Final QA: Harvard writing sample with current combinations

Date: 2026-09-02

Status: **CONDITIONAL PASS**

## Product

- Final PDF: `pdf/hps_gpr_writing_sample_junior_fellows.pdf`
- SHA-256: `7ae21cdc3071aa84280de5c50ce6efee249d704c402bd0b140e5a5332ec50633`
- Format: 61 letter-size pages, not encrypted
- Numerical attestation: `conditional_release_complete_with_numerical_exception`

## Numerical and semantic checks

- The release has 680 result rows, 415 frozen-state prediction rows, all seven
  nonempty scopes, and 23 passing covariance-conditioning audit rows.
- The all-three minimum is 66 MeV with local asymptotic
  `p0 = 0.0028448951`, `Z = 2.76514`, and observed 90% CLs
  `epsilon^2 = 6.63565e-6`.
- The opening note says that this selection does not report global
  significances and identifies every quoted `p0` or `Z` as local and fixed-mass.
- The title-page abstract summarizes the validation scales, stable injected-signal
  response, readiness for full-data unblinding, seven observed fit scopes, and the
  66 MeV all-three local fluctuation.
- The title page and PDF metadata list Emrys Peets as the sole document author.
- Selected-results pages 54--59 contain no 2021 1% or 2016 10% comparison.
- The result section explicitly states that p-values are fixed-mass and local,
  with no toy calibration, expected bands, look-elsewhere correction, or global
  significance.
- The disclosed 2016 cross-process numerical-reproducibility exception is
  visible in Section 6; 2016-inclusive curves are labeled preliminary and
  conditional.
- The rejected support-scan table formerly numbered Table 9 is absent. Its
  informative numerical range is retained in prose; the Table 9 now present
  is the signal-yield decomposition associated with Figure 33.
- The Section 5--6 source contains no use of `criterion` or `criteria`.
- The three requested gray-band commentary sentences and the former closing
  paragraph after the signal-yield table are absent.
- The extraction-plot title no longer says `not scan-corrected`. Faint
  plot-embedded footer notes were removed from Figures 24 and 30--33 while the
  standard numbered document captions were retained.
- The build log has no undefined reference, overfull box, underfull box,
  missing-character, missing-file, or TeX error diagnostic.

## Visual inspection

The revised title page, the three shortened-caption pages, the Section 5 validation
conclusion, and the affected figure pages were freshly rendered with Poppler and
inspected at full resolution. Legends or annotations in
Figures 24, 26, and 28--32 sit outside the data regions. Figure 33 and the
renumbered signal-yield Table 9 are readable on the same page. No clipped text,
figure-caption collision, unintended blank page, or missing glyph was found in
the revised page range.
