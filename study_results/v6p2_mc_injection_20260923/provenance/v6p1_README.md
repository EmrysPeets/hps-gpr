# HPS-GPR v6.1: reconstructed-MC signal templates

Standalone study dated 22 September 2026, following the model and typography of Analysis Note v5.0.5. It compares reconstructed 2021 A-prime MC templates with a freshly converged Gaussian reference using the original primary window as its baseline. An extension separately tests common fit/training half-widths 2.4, 2.5 and 2.6 nominal sigma, with no separate guard window.

The supplied MC is `/sdf/data/hps/physics2021/preselection/v13/ap_signal_prompt_smeared`, with twelve generated masses from 40 through 260 MeV in 20 MeV steps. The observed 2021 10% input derives from the v16 `data_10pc_prompt_TC_psum2p8` production. Stored target-constrained reconstructed masses receive no additional smearing, recentering, truth matching or event-level cuts. The latest ROOT tree cycle is used once.

The target-constraint and upstream `psum > 2.8 GeV` labels agree. Inspected MC timing windows include 5.7/9.2 ns where the observed sample uses 5.1/7.8 ns; complete production equivalence is not established. The directory name and smear-ratio branches do not alone reconstruct the upstream smearing chain. The selection audits and exact input identities are preserved under `provenance/` and in per-mass histogram metadata.

The supplied candidate distributions contain broad and displaced components and are not established as pure reconstructed resonance responses. Both stored daughter truth-link flags are false in the bounded inspected sample; these flags cannot support a truth-matched selection or prove that no candidates contain signal daughters. Stored `psum` and `psum_scalar` satisfy the cut in all inspected rows; apparent failures from raw particle-vector sums reflect a different convention. The all-selected distribution is used conditionally. The inherited coupling coordinate is not a validated A-prime exclusion: unmatched or accidental continuum need not scale with coupling.

## Results and comparison

The primary comparison uses direct MC templates at ten generated masses, 60–240 MeV. At each mass, Gaussian and MC fits share the observed spectrum, archived GP kernel states, correlated background constraint, original ±2.25 nominal-sigma fit/training mask, density conversion and converged likelihood solver. Upper limits solve bounded piecewise-asymptotic `CLs = 0.10`. Local probabilities use `p0 = Phi(-max(signed_r, 0))`, giving `p0 = 0.5` for deficits.

The dense 50–250 MeV curve is explicitly exploratory. Its morph linearly mixes neighboring CDFs in nominal-resolution standardized residual coordinates, without extrapolating beyond the generated MC grid. Leave-one-mass-out checks compare its conditional-support CDF, core fraction and observed fit with the held-out direct template. Passing those checks does not validate detector response against data or establish coverage.

Limit ratios, paired signed-root differences, extrema, median changes and fractions within 1%, 5% and 10% summarize the correlated curves. They are not independent-point chi-squared results or agreement probabilities. Thirty-two Poisson MC-bin resamples per direct mass assess conditional finite-template precision for the verified unit-weight histograms; they omit detector systematics, production-selection differences and observed-background uncertainty.

Full 2015 and full 2016 Gaussian references are retained. The supplied 2021 MC is not treated as a validated response for either earlier campaign, and no MC-template result for those campaigns is claimed.

## Files

- `histograms/mNNN.npz` and `.json`: reconstructed-mass bin sums of weights and squared weights, chunk information, selected statistics and source metadata. Histogram range is 0–400 MeV with 0.1 MeV bins.
- `derived/scans.csv`: Gaussian, direct-MC and exploratory morph fits.
- `derived/comparison.csv` and `agreement.json`: paired direct-MC comparisons and descriptive summaries.
- `derived/shape_summary.csv`: selected MC statistics, quantiles, bias, tails and support fractions.
- `derived/leave_one_out.csv`, `mc_bootstrap.csv` and `mc_precision.csv`: interpolation and conditional MC-sampling checks.
- `source/report.tex`, `figures/` and `derived/report_*.tex`: standalone report source and generated presentation assets.
- `qa/` and `provenance/`: numerical checks, source audits, input identities and rendered-page review.

MC templates are normalized on the unchanged 36–299.75 MeV analysis support before selecting the primary fit bins. They are not renormalized within the blind window. `full_yield_90` uses this support normalization; `full_selected_yield_90` divides by the recorded support fraction. That additional yield conversion does not establish a calibrated coupling correction outside the support. The inherited minimal-visible branching correction enters displayed coupling limits once above the dimuon threshold and does not change visible yields or local probabilities.

The source histograms retain masses inside 0–400 MeV and record overflow separately. Quantiles are conditional on that stored range; support fractions and full-selected-yield bookkeeping include the selected weight outside it. The solver's `A90` is the dimensionless coordinate `epsilon² / 1e-8`, not an event count.

About 24–61% of the direct-mass selected distributions lie outside the primary blind window. A background-only GP can absorb signal entering its training bins. The fixed-sideband observed comparison does not model that feedback or establish extraction recovery or coverage for these broad shapes.

## Rebuild the report

After the scientific arrays and ledgers exist:

```sh
python3 scripts/make_report.py
python3 scripts/make_window_report.py
cd source
tectonic -C --outdir ../qa report.tex
```

The report generator reads saved numerical products and creates presentation figures/tables; it performs no fits. It requires NumPy, SciPy, pandas and Matplotlib. Tectonic's `-C` assumes cached TeX resources; omit it for an initial resource download. Final delivery requires numerical completeness plus PDF text and rendered-page checks.

The user-selected remote mirror is `/sdf/home/e/epeets/src/hps_gpr_v6p1_mc_signal_templates_20260922`. Source ROOT files and the original analysis note remain unchanged. The study is conditional on the supplied MC selection, finite statistics, fixed background model and asymptotic inference; it adds no calibrated global probability or coverage claim.

## Window and midpoint-gallery extension

`derived/window_scans.csv`, `window_comparison.csv` and `window_agreement.csv` retain the four widths separately. Each width recomputes the observed GP prediction and correlated covariance from its exterior bins with fixed archived kernel parameters; Gaussian and MC share that prediction within the width. Changes across widths combine the different fit bins and sideband prediction. No width is chosen by observed performance. `window_loo.csv` and `window_mc_precision.csv` preserve the corresponding interpolation and 32-resample diagnostics.

Separate figures `figures/window_2p4.pdf`, `window_2p5.pdf` and `window_2p6.pdf` accompany the baseline figures. Final full-range and core-zoom galleries show midpoint morphs at 50, 70, ..., 250 MeV together with their transported neighboring MC distributions. Saved morphs live under `histograms/morphed/`; the gallery displays eleven of the 191 non-native integer-mass exports. This is MC-only CDF interpolation, not a new simulated sample or a truth-associated signal response. The 50/70/90 MeV morphs touch the low-mass region where held-out shape checks failed.

## MC core-centered primary windows: 23 September 2026

The new comparison retains each reconstructed MC distribution and places the fit/training mask at `c(m0) +/- 2.25 sigma_nominal(m0)`. It does not translate the MC, shrink the width to its fitted core width, or use observed data to locate the core. A local bin-integrated Gaussian plus nonnegative affine pedestal estimates `c` from MC only. Mode finding uses smoothing of0.2 nominal sigma and searches within3 nominal sigma of the pole; the central fit spans1.5 nominal sigma around that mode. Alternate spans1.25 and2.0 and32 independent-bin center resamples per native mass are diagnostic checks.

The new MC and a Gaussian centered at the same `c` share the shifted-mask GP prediction. Kernel coordinates, density conversion and branching correction remain anchored at generated mass `m0`. The historical pole-centered Gaussian and MC are retained for comparison. Native centers shift down1.21–4.38MeV; new/old MC limits span0.822–1.201. The native80MeV result has center78.103MeV and local p0=0.007875, versus0.02477 originally. All are conditional comparisons; no global probability or response certification is implied.

New files: `derived/core_centers.csv`, `core_scans.csv`, `core_comparison.csv`, `core_native_diagnostics.csv`, `core_center_sensitivity.csv`, `core_agreement.json`, and `histograms/core_centered_windows/`. Rebuild with `scripts/core_centering.py` then `scripts/make_core_report.py`, or the full reproduction script. The pre-centering PDF and source are preserved in `history/before_core_centering_20260923/`.

## Analytic shift, shared shape and asymmetric windows

`analytic_shapes.py` compares four equal-mass location laws, constructs a shared standardized CDF, tests it with each native histogram held out, and fits five prespecified left/right windows to direct MC and Gaussian templates. `make_analytic_report.py` makes the added figures/tables. The preferred descriptive law is `c(m)-m = -3.22433 - 2.21399 ln(m/150)` MeV for60–240MeV; LOO RMS0.0652MeV. It is an empirical interpolation, not an extrapolation or detector mechanism.

Normalized cores align more closely than full selected-candidate shapes. Common-shape LOO full-CDF discrepancies reach0.461 and corresponding upper-limit ratios reach0.443–1.191, so the saved181 analytic/common templates are exploratory and not adopted as nominal replacements. All plots distinguish full unit normalization from conditional-core normalization. The core-conditioned overlays do not establish a common core fraction. Left external tails dominate120–240MeV; right external tails dominate60–100MeV. Both directions are therefore retained in the asymmetric-window controls, without choosing an observed winner.

New ledgers are `analytic_shift_models.json`, `analytic_shift_predictions.csv`, `shared_shape_metrics.csv`, `shared_shape_heldout.csv`, `asymmetric_scans.csv` and `analytic_shape_summary.json` under `derived/`. Shared templates are under `histograms/shared_shape/`. The pre-extension release is preserved in `history/before_analytic_shape_study_20260923/`.

## Figure 12 MC-point clarification (23 September 2026)

The colored Figure 12 curves are native empirical MC histograms, with CDF rebinning in the standardized panels. The Gaussian-plus-pedestal fit determines only center and core width. The dashed curve is a separate equal-mass common-shape proposal. Direct count overlays for 60 and 160 MeV are in `figures/MC_points_vs_figure12.pdf`. Their integrated-bin agreement is a same-sample bookkeeping identity, not independent goodness of fit.

The 60 MeV sample has 32.4% of its stored selected distribution in |u|<2, and 15.8% retention from the recorded readout cutflow stage. Neither number establishes generated-signal acceptance or candidate purity. The existing ±2.25 nominal-sigma window is ±2.920–3.174 in u; ±2.25u would narrow it by 23.0–29.1%. This extension adds conversions and continuous MC fractions without changing masks or repeating observed fits.

Reproduce with `python3 scripts/make_MC_points_report.py` and rebuild the report. Ledgers: `derived/MC_points_overlay_bins.csv`, `derived/MC_points_window_units.csv`; numerical checks: `qa/MC_points_window_units.json`; previous PDF: `history/before_MC_points_clarification_20260923/`.
