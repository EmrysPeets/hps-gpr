# Which 2016 tests passed, and what they establish

Source audit: v5.0.5 files under `study_results/v5p0p5_analysis_note_20260916/source/sections/`. These are existing results, not new v5.8.0 simulations.

| Existing check | Recorded result | Claim supported |
|---|---|---|
| Historical 2016 10% injection study | Aggregate zero-signal mean pull −0.016; nonzero-injection signed average −0.262 | Conditional average recovery for that source, exposure, and fitting policy; not zero local bias everywhere at full exposure. |
| Upper length-scale range | Boundary occupancy 142/142 at upper factor 8, 56/142 at 10, zero at 12/15/20; 12→15 max local-significance change 0.00255, max yield-limit change 0.083% | Factor 12 is the first tested nonbinding numerical range; opening the upper bound further does little. This does not test shortening the fitted kernel or qualify the background. |
| Full-2016 support qualification | All six eligible lower edges 28–33 MeV fail: only 3/12 cell means meet 0.75, only 1/4 zero-signal cells meet 0.75; worst absolute means 2.281–2.696 exceed the 1.25 gross-bias guard | This test did **not** pass. No support was selected by that protocol. It exposes a deterministic stress-truth/GP mismatch. |
| Saved-parameter numerical replay | Fixed saved parameters reproduce predictions; independent optimizer fits meet the 10⁻⁶ parameter tolerance at only 87/142 masses | Reproducibility of the frozen calculation, not repeatability of all optimized parameters or statistical calibration. |
| Two-truth empirical calibration | Zero Holm-adjusted flags after calibration across 3,648 true-yield-exclusion and 1,824 background-only local-rejection cells | Conditional error control for the calibrated envelope and tested truths, not raw asymptotic p-value calibration. |
| Coherent stress-field approximation | 2016 standardized means within 0.099 and widths 0.947–1.047; empirical/response correlation RMS 0.0626 on the v5.1.1 grid | The Gaussian response approximates this biased stress ensemble centrally. Centering subtracts its large offset by definition. It does not show that the raw signed root is centered or the stress mean is the physical null. |

The empirical-calibration result is a substantial passed test and should be credited accurately. The corresponding **raw profiled** test still had excess-rate flags: 50/456 local-GP and 140/456 stress background-only local-rejection cells; exclusion flags were 179/912 and 326/912. With 500 validation spectra per cell, the first Holm local-rejection threshold is 48/500 (9.6%) against the nominal 5% benchmark. Absence of adjusted flags after calibration does not establish percent-level agreement everywhere.

A long kernel can underfit a coherent shape across an excluded window, while a shorter kernel can reduce this bias but absorb injected signals or inflate uncertainty. The upper-bound plateau tests only whether optimization is artificially capped. A meaningful stiffness test varies the kernel under predetermined rules and examines background bias, physical signal recovery, numerical conditioning, and held-out local rejection together. A cosmetically smaller stress offset alone is insufficient.

Sources: `05_toys_validation.tex` (historical aggregate pulls); `v501_historical_methods.tex`, subsection “Controlled 2016 upper length-scale range”; `05b_2016_support_selection.tex` (failed Phase 1); `v5_historical_appendix.tex` (87/142 replay); `v5_calibration_appendix.tex` (validation rates and test power); `v504_global_appendix.tex` (coherent stress field).
