# APEX initial studies

Standalone LaTeX comparison, 16 September 2026. Open `output/pdf/APEX_Initial_Studies.pdf`; editable source is `source/main.tex`.

The supplied APEX limit and local-p screenshots overlap the released HPS contour at 155–250 MeV. No common integer mass has both local probabilities below 0.1. The APEX minimum is near 229 MeV, p about 0.037. Native full-2021 observed-equivalent limit crossings occur only around 155–175 MeV; only the 158, 166 and 170 MeV nodes survive the adopted pixel envelope. These are numerical contour comparisons with unharmonized confidence levels.

The saved native 2021 signal hypotheses at 160 and 208 MeV imply coupling strengths about 2.4 and 19 times the APEX contour, assuming the same minimal visible-vector definition. The historical 1% scenarios are separate, conditional forecasts with unverified sample equivalence. No new toys or GPR fits are run.

The original APEX source, confidence level and cropped resolution-axis units remain unresolved. “70%” is the user's source description. The report displays both MeV-width and percent-resolution interpretations, retains the supplied probabilities, and does not treat nearby minima as independent evidence or extrapolate HPS beyond 250 MeV.

## Main files

- `derived/apex_limit_digitized.csv`, `apex_pvalue_digitized.csv`: every extracted image column, pixel coordinates, central value and stroke envelope.
- `derived/apex_total_resolution.csv`: all 12 black-square total points, native units and both conditional width mappings.
- `derived/common_mass_scan.csv`: complete 96-point comparison, present limits, full-equivalent displays, probabilities, widths and ratios.
- `derived/dense_display_scan.csv`: APEX-column comparison with log-interpolated HPS p-values; interpolation does not create a new fitted scan.
- `derived/apex_local_minima.csv`, `hps_peak_comparisons.csv`: extrema catalogues and separately labelled nearby-window minima.
- `derived/projected_signal_compatibility.csv`: saved injected-coupling comparisons.
- `figures/`: vector scientific figures and PNG previews, plus direct digitization overlay.
- `inputs/`: exact user images and pinned HPS result snapshots.
- `provenance/`: source/calibration metadata, hashes and checkout identity.
- `qa/`: numerical validation, extracted PDF text, rendered pages and final visual/portable-build records.

## Rebuild

Requirements: Python 3 with NumPy, SciPy, pandas, Matplotlib, Pillow and PyMuPDF; Tectonic with its LaTeX resources. The original HPS checkout and network access are not needed for numerical reproduction.

```bash
bash scripts/build.sh
```

For document-only compilation, run `tectonic main.tex` from `source/`. `snapshot_inputs.py` records the original snapshot operation and is not needed for a portable rebuild. `validate.py` checks the existing parent files when available and skips parent-location checks when rebuilding elsewhere; bundled snapshot hashes are always checked.

The report uses the frozen v5.0.4 released p-values and limits as its primary observed comparator, and the 13 September Figure 2 revision for the optional added-2019 and full-equivalent contours. The word “combined” means 2016+2021 through 180 MeV and 2021 alone above 180 MeV in this overlap. Numerical values retain the inherited preliminary/conditional HPS status; this package is not an observed-result release or unblinding authorization.

Pixel envelopes represent extraction ambiguity, not experimental confidence intervals. APEX p-values and limits are not statistically independent inputs. The same-mass maximum-p diagnostic is conditional on valid local p-values, and its finite-grid Bonferroni result is not a calibrated continuous-scan significance.

The portable `output/pdf/APEX_Initial_Studies_LaTeX_Data.zip` includes the PDF, LaTeX, input snapshots, digitizations, plots, scripts, metadata and checksums. Large reference-search downloads and page previews are omitted; their URLs and hashes remain in `provenance/reference_search.json`. They are not numerical inputs. The document rebuild was checked in an isolated temporary folder, with all eight pages matching in text and pixels.
