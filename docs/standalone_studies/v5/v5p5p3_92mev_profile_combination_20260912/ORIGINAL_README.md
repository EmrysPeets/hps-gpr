# HPS-GPR v5.5.3: 92 MeV combination and exposure checks

The standalone report opens with the unchanged v5.0.4 observed limits, local p-values and Sidak-equivalent references. It then compares shared coupling, an empirical energy response, and independent positive rates while allowing shared or separate centroids and MC-bounded resolutions. Original packages remain unchanged.

## Main findings

- On the original v5.5.0 experiment, raw sqrt(Q0) is 2.520 for common coupling and 4.213 for the fitted energy response. On the wider common experiment needed for shape comparisons, these become 2.447 and 4.180.
- At fixed 92 MeV/scaled resolution, the fitted-energy predictive Gaussian reference gives p = 6.81e-5, Z_G = 3.815. Searching a shared mean over 90–94 MeV gives p = 2.09e-4, Z_G = 3.528.
- Floating separate means and bounded widths gives raw sqrt(Q0) = 4.598. The independent-amplitude reference envelope evaluated at that statistic gives p = 2.879e-4, Z_G = 3.443. This is **not** the energy-model probability or a rigorous continuous-family bound. No unvalidated joint-shape calibration is promoted into significance.
- Preferred centroids are 91.567, 91.735 and 93.327 MeV; widths are 3.928, 2.439 and 1.959 MeV. The 2021 width reaches its pre-additional-TC-scaling reference, not a verified fully unsmeared resolution. Width constraints remain assumed and no sub-reference width is permitted.
- Actual 2016 10% and two available 2021 1% selections are weak positive controls. Neither 2021 1% histogram provides enough metadata to certify the same prompt-TC selection or event overlap. They are not independent combination members.
- If the current fitted 2021 line and same selection persist, nominal 100% exposure gives 77,802 expected signal rows and Asimov roots 4.06 or 4.85 under two assumed GP covariance scalings. These are conditional expectations, not future observed data, calibrated significance forecasts or uncertainty bounds.

## What is fixed and what is varied

`protocol.json` declares means in [90,94] MeV, exponent in [-6,6], independent widths between the inherited MC and scaled values, and main penalty 0.5 sum(t_i^2). Fixed scaled and fixed MC controls have no width penalty. The penalized width family contains its fixed-MC member; treating the unpenalized scaled control as an ordinarily nested member would be incorrect.

All new main models use fixed union windows, frozen 92 MeV GP kernel states (90 MeV for the 2015 extension), the same background constraints and bins, and a density conversion fixed at the MC92 reference. Gaussian templates are normalized over histogram support, then restricted to the fit bins; their outside-window fraction is retained. `consults/audit_history.json` separates historical masks and density conventions from this new common experiment.

The exact observed fits use the Poisson likelihood plus Gaussian background and stated width constraints. The Gaussian score reference uses fitted-null V = diag(lambda0) + L L^T, including a fluctuating GP auxiliary contribution; it is not a Poisson-only ensemble with a fixed auxiliary observation. The assumed width penalty stays fixed. It uses 100,000 importance draws, common across comparisons, with checked normal/chi-bar benchmarks, Monte Carlo errors, compression accuracy and shape-grid sensitivity. Shared means use a finite bank with continuously refined exponent basins. These references do not supply full-search, direct-Poisson or detector/background calibration.

## Files

- `pdf/HPS_GPR_v5p5p3_92MeV_Combined_Profile_Study.pdf`: standalone report with primary citations.
- `derived/fits.csv`, `fit_bins.csv`, `individual_shape_grid.csv`: all 24 exact models, extraction bins and 459 individual grid fits.
- `derived/gaussian_score_calibration.*`, `shared_mass_energy_reference.*`: accepted reference calculations. `report_local_references.csv` selects the proper direct/shared/envelope result used in the report.
- `derived/energy_reference_envelopes.csv`: explicitly labeled broader-family reference values.
- `derived/subsets_*.csv`: three actual smaller-sample fits and 28 conditional exposure fits.
- `inputs/`, `engine/`, `provenance/inputs.json`: pinned spectra, ROOT controls, numerical engines and SHA-256 source identities.
- `consults/`: HEP phenomenology memo, source ledger, dataset provenance, statistical interpretation and independent fit audit.
- `qa/failed_energy_shape_calibration/`: quarantined exploratory coordinate-calibration attempt that failed an exhaustive small-bank gate. Its numerical probabilities are excluded from every report figure and inference table; the retained files document that rejection.
- `qa/`: accepted numerical/document checks, page renders, visual inspection and portable-build validation.

## Reproduce

Python 3 with NumPy, SciPy, pandas, Matplotlib, uproot, PyMuPDF and Pillow; Tectonic for TeX:

```bash
bash scripts/build.sh
```

The build reruns the exact profiles, accepted Gaussian references and subset/projection calculations, then regenerates the figures and note. Numerical processes are sequential with one BLAS/OMP thread. Only the bundled snapshots are required; original source paths are provenance. To rebuild the note alone, run `tectonic main.tex` inside `source/`.

After visually inspecting all final rendered pages, `python3 scripts/package.py` verifies an isolated TeX rebuild, records a manifest and packages the PDF with source and results. The final visual review is bound to the PDF SHA-256. No commit or publication is performed.
