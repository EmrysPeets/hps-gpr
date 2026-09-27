# Paper and prior-study review for v5.8.0

Reviewed 17 September 2026. Read-only literature/source review; no numerical result below is a new simulation unless explicitly identified by the parent study.

## What the linked paper establishes

[Ananiev and Read, arXiv:2206.12328v3](https://arxiv.org/pdf/2206.12328) is **Gaussian Process-based calculation of look-elsewhere trials factor**, dated 3 May 2023. It contains no HPS 2016 10% dataset or search result. Its Section 2 uses the signed likelihood root, requiring unconstrained signal strength and asymptotically standard-normal null marginals. It estimates scan correlation from one-bin, one-standard-deviation Asimov perturbations, then samples Gaussian fields to calculate a global tail. Unit-diagonal normalization assumes valid local marginals; it is not a background-bias correction. Section 2.2 distinguishes the mass-hypothesis grid from data binning: coarse grids miss maxima and upcrossings. Its squared-exponential example starts losing upcrossing accuracy at grid spacing around one third of the correlation length; this is an example, not a universal HPS prescription. Section 3.1 finds smaller Poisson/Gaussian tail discrepancies at higher background rates. Therefore the paper motivates a 0.5 MeV grid-convergence test but cannot supply a new calibrated 2016 local significance or a dataset-specific 10%-to-100% mapping.

## Existing HPS evidence

The relevant prior note is **v5.1.1**, `study_results/v5p1p1_significance_covariance_injection_20260910/report.tex`, rather than v5.5/v5.6/v5.7. The v5.0.5 method/results are `source/sections/v5_global_significance.tex` and `v5_global_results.tex`; detailed qualification is in `v504_global_appendix.tex`.

* v5.1.1 already adds eleven half-integer nodes, 70.5–80.5 MeV, with positive kernel parameters interpolated in log space. The denser 2016 stress response still spans −17.86 to +13.71. Empirical-versus-response correlation RMS differences are 0.0556, 0.0626, 0.0594 for 2015, 2016, 2021: central covariance accuracy is similar across campaigns. The poor 2016 picture primarily concerns the null offset, not uniquely failed covariance mapping.
* At 2016/76 MeV, existing 64-toy mean roots under local-GP truths are −0.32, +1.62, +4.52 for injected sizes 0, 2, 5. Under the archived stress truth they are −14.64, −12.70, −9.80. Response to signal is present; the large offset remains.
* The stress construction blends a degree-five logistic–Chebyshev fit over 26–80 MeV into a broad continuation across 75–85 MeV. The broad fit has a failed convergence flag, and development-sample overlap remains. Removing the low-mass component changes the 76 MeV response −14.653 → −5.926 but worsens 42 MeV −9.215 → −23.524, increasing full-grid RMS 4.967 → 9.430. Integrating the blend more accurately changes roots by at most 0.001135. Neither control qualifies a replacement null.
* At 76 MeV the saved observed roots are −2.460 (2016) and +0.166 (combined), while stress offsets are −14.653 and −8.700. The combined positive-fit gate opens despite a nearly zero raw signal fit. Its large stress-centered score is thus a reference mismatch; a smaller stress offset does not establish improved discovery calibration.

## Actual 2016 10% input and compatible bounded comparison

The local source exists at `study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/source_2016_10pct.root`, histogram `h_Minv_General_Final_1`. Its SHA-256 was independently rechecked in this review:

`789e619fcbeb5e81f9193d3e224bc17919983477a037bf3d79692327555f9fd4`.

The prior `v5p5p3_92mev_profile_combination_20260912/inputs/subset_input_manifest.json` records 7,483,101 visible rows, 6,000 native 0.05 MeV bins, and a full-histogram ratio 0.1021997813 to the 2016 parent. Selection, exact luminosity fraction, and event overlap are **unverified**. Counts are not exposure or efficiency measurements. The actual subset must remain separate from the deterministic 0.1× full-spectrum control in v5.1.0.

`v5p5p3_92mev_profile_combination_20260912/scripts/subset_checks.py` provides a bounded implementation: exact rebin to parent edges, substitute subset counts in `engine/common.DATA`, and call `moving_context` with the inherited parent kernel/resolution. At 92 MeV that archived matched-policy fit has signed root 1.259169727 and maximum optimizer score 7.88×10⁻⁸. It is neither a historical-card reproduction nor a calibrated particle significance.

## Interpretation and recommended claim boundary

Derived expectation: with a fixed fractional shape mismatch and Poisson-dominated covariance, mismatch yield grows with exposure L while its uncertainty grows as √L, so signed bias can grow approximately as √L. This can explain why a smaller sample looks more centered, but is conditional on matched selection, analysis policy, and uncertainty scaling. Higher counts alone do not imply worse Gaussian asymptotics. Test that mechanism using a deterministic exposure control, and test the real smaller histogram separately.

A meaningful next local result requires an independently justified null (or explicitly conditional fixed-mass control) and validation of signed-root marginal calibration and signal recovery. Scan refinement measures numerical sampling; it cannot repair an arbitrary null. Combined fits should be interpreted through constituent efficient scores and information weights, with any reduction of the 2016 offset distinguished from cancellation and the positive-fit gate.

Memory used only to locate the live prior-study source: `MEMORY.md:61–69`, rollout `01a08814-b35b-7731-bec8-6219690cf1aa`. All substantive local facts above were rechecked against current files.
