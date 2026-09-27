# Scientific display revisions, 23 September 2026

All files are presentation derivatives. The v5.0.5 and v5.9.5 parents are unchanged; their input/code SHA-256 values were checked before and after computation in `provenance/protocol.json`. No new Poisson spectra or hyperparameter optimizations were performed. Calculations use one BLAS thread and the pinned v5.0.5 predictor.

## Slide 11: explain each curve at the preserved 78 MeV example

`assets/slide11_profile78_clear.png` has absolute event density above and differences from the **same unprofiled GP mean** below. The lower curves are black `n − b_GP`, blue dashed `b_prof − b_GP`, red `b_prof + A w − b_GP`, and purple dotted `A w`. The light blue band is ± the GP constraint standard deviation, not a fitted-background uncertainty. The gold zero line is the GP reference after subtraction. Divide all count quantities by the bin width to obtain events/MeV; plotted units are thousands of events/MeV. Data error bars are Poisson square-root count errors.

The fixed-state signal/background profile at 2021 78 MeV reproduces the saved raw signed root, 2.8086449752. The fitted epsilon-squared is 4.3449128637e-6. Two profile fits were run, with and without signal; no scan or kernel optimization was repeated. Exact plotted count-space arrays are in `data/slide11_profile78_curves.csv`. The separately rendered `slide11_lower_legend.png` is optional if native slide labels supply these definitions.

## Slide 13: GP predictions throughout all three search regions

`assets/slide13_all_datasets_heldout.png` covers 2015 full (19–100 MeV), 2016 full (39–180 MeV), and 2021 native 10% (50–250 MeV). Every displayed bin is predicted using the nearest integer-mass hypothesis, with that hypothesis's ±2.25 sigma window excluded from GP training. An assertion verifies that each displayed bin lies in its own excluded window. The line joins these local predictions; it is not one global GP fit. Archived, data-derived kernel states remain fixed; the 2015 kernel state above 90 MeV freezes at the declared 90 MeV endpoint.

The lower panels show `R_i = (n_i − b_i) / sqrt(b_i + C_GP,ii)`. `b_i` and `C_GP,ii` refer to that bin's own local prediction. The green ±2 band is a visual reference, not a calibrated simultaneous or coverage band. Residuals at different masses are correlated, and the fixed archived hyperparameters mean this is conditional held-out prediction, not a fully repeated cross-validation of hyperparameter choice.

| Dataset | Display bins | Local hypotheses | Fraction outside ±2 | Maximum absolute residual |
|---|---:|---:|---:|---:|
| 2015 full | 324 | 82 | 9.88% | 3.896 |
| 2016 full | 564 | 142 | 8.69% | 3.426 |
| 2021 native 10% | 320 | 200 | 5.00% | 3.630 |

These descriptive fractions do not supply a binomial goodness-of-fit probability because of the cross-mass correlations. Exact bin centers, anchor choices, means, variances and residuals are in `data/slide13_heldout_bins.csv`. Equation asset: `assets/slide13_residual_equation.png`.

Suggested slide wording: **Each bin is predicted with its local ±2.25σ window excluded. The curve joins overlapping local GP predictions; green ±2 is a reference band.**

## Slide 14: an actual sideband-only fit diagnostic

Use `assets/slide14_sideband_deviance.png` and `assets/slide14_deviance_equation.png`. For each illustrated 2021 anchor, GP training uses all available support bins outside the ±2.25 sigma window. The diagnostic then evaluates **only search-region bins outside that same window**:

`D_side = 2 sum_side [n_i log(n_i / b_i) − n_i + b_i]`.

The search interval is 50–250 MeV; available support edges are 36–299.75 MeV. `N_side` is the number of bins, not an effective number of degrees of freedom. This is an in-sample sideband diagnostic: the evaluated bins also train the GP. It does not test prediction inside the withheld signal window.

| Excluded-window center | N_side | D_side | D_side/N_side |
|---|---:|---:|---:|
| 65 MeV | 305 | 271.174 | 0.8891 |
| 78 MeV | 304 | 259.076 | 0.8522 |
| 120 MeV | 299 | 273.566 | 0.9149 |

For the 78 MeV anchor only, the GP prediction was recomputed on the **same 256 saved complete Poisson spectra** used in v5.9.5, from the same frozen, observed-data-derived nominal GP source. The kernel state stays fixed, but targets and count-dependent training noise are recomputed in each toy. This took about 1.5 seconds; it is 256 single-anchor refits, not 256 mass scans.

237/256 toys have deviance at least as large as observed: conditional upper-tail estimate `k/N = 0.92578`, add-one estimate `238/257 = 0.92607`, and exact binomial 95% interval `[0.88652, 0.95473]`. The toy mean of D/N is 0.96265; its central 90% interval is `[0.84468, 1.10412]`. This provides no evidence of excess sideband deviance against this conditional reference. It does **not** establish unconditional model validity, held-out predictive calibration, or resonance significance. The 78 MeV anchor was selected from earlier observed scans, and no selection correction is supplied here. The interval covers finite toy-count uncertainty only, not source or model uncertainty.

A **secondary binned cumulative count-shape distance**, formed after separately normalizing observed and fitted sideband totals, was evaluated with the same refits. At 78 MeV its value is 2.55574e-5; 186/256 toy distances are at least as large, add-one tail 187/257 = 0.72763, with exact 95% interval `[0.66759, 0.78021]`. This is a bespoke, conditional calibration of a KS-like shape distance on the disconnected sideband bins. It is **not** a distribution-free KS p-value, and the separately normalized distance is insensitive to the total sideband normalization. Poisson deviance is the primary display because it retains bin-by-bin count information.

The old Q diagnostic and its interpretation remain documented in `study_results/v5p9p5_null_bias_20260922`; replacing its slide display does not erase or supersede that study.

Suggested slide wording: **Outside the blind window: no excess sideband deviance in the fixed-source reference. This checks fitted sidebands, not held-out prediction.**

## Reproduction and visual checks

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 /Applications/Xcode.app/Contents/Developer/usr/bin/python3 output/slides/unblind_meeting_RCmeet_20260923/science/make_science.py
```

All figures are provided as PNG and vector PDF. The calculation script, exact plot CSVs, summary JSON and source hashes are retained. Figure renders were inspected for clipped labels and overlaps; the gold GP curve is drawn over quiet black data markers, the sideband histogram says “Toy refits,” and each curve's subtraction is explicit. Final placement/readability in the native deck requires the parent agent's slide-render review.
