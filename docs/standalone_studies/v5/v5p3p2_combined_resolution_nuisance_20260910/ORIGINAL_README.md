# HPS-GPR v5.3.2: combined resolution nuisance and extracted excesses

This standalone LaTeX study preserves the v5.3.1 individual results and adds actual combined likelihood fits with independently varying dataset resolutions. Read `report.pdf`; edit or rebuild `report.tex` using the included vector figures. The prior version remains unchanged.

The primary combination has one common native coupling, independent GP background constraints, and one width fraction per active dataset: sigma_d = sigma_MC,d + t_d (sigma_scaled,d - sigma_MC,d), with 0 <= t_d <= 1. The assumed MC-centered constraint adds sum(t_d^2)/2 to the negative log likelihood. Observed bins, GP mean/covariance, and MC density conversion remain fixed. Width and background are profiled in the observed numerator, common-coupling denominator, and background-Asimov numerator. A bounded-flat independent-width model and a common-fraction model with one constraint provide separate modeling controls. The common and independent Gaussian constructions differ in their constraint structure; their difference is not a pure correlation effect at fixed penalty.

There are 232 combined mass coordinates and 464 results for each of the independent-width and common-fraction studies. The inherited individual ledger contains 850 results. No toys or expected bands are added. The 2015 scan continues through 100 MeV with 90–100 MeV highlighted. Nuisance constraints remain assumptions, and p-values are nominal local asymptotic diagnostics without mass/width-search calibration.

The Gaussian combination increases the median observed limit by 9.7% relative to fixed MC on the same experiment, or 12.3% considering only masses with multiple datasets. Its strongest nominal local excess is at 93 MeV: Z = 3.2223, p0 = 0.00063586, fitted native epsilon squared = 2.8950e-6, and the 90% CLs upper limit = 4.2031e-6. This point includes the exploratory 2015 extension. The fitted width fractions are 0 for 2015, 0 for 2016, and 0.0774 for 2021. At the same mass, extending the allowed width interval from zero to the full correction raises the limit from 3.9942e-6 to 4.2031e-6, about 5.2%, with little change in local p0.

Individual strongest-region plots use each completed Gaussian scan: 2015 at 51 MeV, 2016 at 91 MeV, and 2021 at 78 MeV. They show observed counts, the sideband GP constraint, the profiled background plus signal, and the extracted residual. Residual error bars show Poisson counting errors only; the GP band is the sideband constraint, not a post-fit confidence band on the extracted signal. Peak selection is descriptive and does not provide a global significance.

## Main products

- `derived/combined_independent_resolution_nuisance.csv`: independent dataset widths, both constraints, fixed-width controls, fitted widths and diagnostics.
- `derived/combined_common_nuisance.csv`: common-fraction Gaussian and bounded-flat controls.
- `derived/combined_peak_summary.json` and `combined_peak_fit_arrays.csv`: strongest combined fit and its dataset contributions.
- `derived/individual_peak_summary.json` and `individual_peak_fit_arrays.csv`: strongest individual regions.
- `derived/combined_peak_profile.csv`: observed likelihood and profiled widths versus tested common coupling.
- `derived/*width_allowance.csv`: upper limits and p-values at fixed selected masses as the upper width bound changes through 0, 25%, 50%, 75%, and 100% of the correction. The penalty normalization remains unchanged.
- `statistics_combined_nuisance.md`, `combined_independent_protocol.json`, and `qa/`: statistical assumptions, numerical checks, source identities, and report QA.
- `qa/inherited_v531/`: validation records for the unchanged individual study.

Raw epsilon-squared columns retain the native electron-channel convention. Figures apply the inherited dimuon display correction exactly once. That correction does not affect signal counts or p-values.

## Reproduce

Python requirements include numpy, scipy, pandas, matplotlib, pypdf, scikit-learn and PyYAML. Tectonic builds the LaTeX report. From the study directory:

```sh
python3 scripts/make_latex_report.py
tectonic --keep-logs report.tex
python3 scripts/check_portable_report.py
```

The report builder reads the saved fits; it does not perform scans. Source spectra are bundled arrays, so original ROOT paths in the analysis card are provenance only. To rerun expensive fits, work in a new copy and move the relevant checkpoint folder aside. The independent driver accepts a future UTC deadline:

```sh
python3 scripts/run_combined_independent.py --deadline-utc YYYY-MM-DDTHH:MM:SS+00:00
```

The original continuation deadline is recorded in `protocol.json`. `MANIFEST.sha256` covers the packaged report, code, inputs, data, figures and QA records. The PDF is also rebuilt in an isolated folder from only its LaTeX source and figures, and all rendered pages are inspected.
