# HPS-GPR v6.2: MC signal injection and recovery

This release uses 40 toys at each of four injection yields (1,000, 5,000, 10,000 and 30,000 selected MC candidates) and eleven native masses (60–260 MeV in 20 MeV steps). The 40 MeV sample is excluded. It contains 1,760 injected spectra and 3,520 primary fits comparing windows centered on the generated mass and on the reconstructed MC core.

The first 20 toys and the original release are preserved under `history/20_toy_release/`. The additional toys use indices 20–39 with the same model and seed prescription. `qa/toy_extension.json` records the preservation check; `results/toy_extension_comparison.csv` compares the two 20-toy groups and their combined results.

## Results

The shifted window improves recovery, but some injected signal is still absorbed by the GP prediction. At 30,000 injected candidates, raw mean recovery spans 79.7–101.1% for pole-centered windows and 82.6–107.3% for shifted windows. After subtracting the fitted yield of each paired zero-signal toy, the incremental response spans 71.5–89.5% and 78.0–93.8%, respectively. These ranges include the separately marked 260 MeV extension. The subtraction is a diagnostic; the primary yields, pulls and intervals retain their original values.

| Mass (MeV) | Pole recovery | Shifted recovery | Pole pull mean / width | Shifted pull mean / width | Pole nominal 95% contains truth | Shifted nominal 95% contains truth |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 85.5% | 94.0% | -0.22 / 1.02 | -0.08 / 0.97 | 38/40 | 38/40 |
| 80 | 101.1% | 107.3% | 0.04 / 0.95 | 0.23 / 0.92 | 40/40 | 39/40 |
| 100 | 86.1% | 92.8% | -0.60 / 0.89 | -0.29 / 0.90 | 39/40 | 39/40 |
| 120 | 90.2% | 96.2% | -0.52 / 1.02 | -0.19 / 0.96 | 37/40 | 38/40 |
| 140 | 90.3% | 95.2% | -0.61 / 1.12 | -0.29 / 1.07 | 36/40 | 36/40 |
| 160 | 84.7% | 90.3% | -1.16 / 0.94 | -0.70 / 0.90 | 33/40 | 38/40 |
| 180 | 87.0% | 92.1% | -1.14 / 0.94 | -0.67 / 0.87 | 32/40 | 38/40 |
| 200 | 85.3% | 89.1% | -1.45 / 0.86 | -1.04 / 0.79 | 29/40 | 37/40 |
| 220 | 85.8% | 88.9% | -1.54 / 1.01 | -1.17 / 1.02 | 28/40 | 32/40 |
| 240 | 82.2% | 84.5% | -2.04 / 0.94 | -1.72 / 0.91 | 15/40 | 22/40 |
| 260* | 79.7% | 82.6% | -2.26 / 0.95 | -1.87 / 0.87 | 15/40 | 21/40 |

*The 260 MeV point uses the archived 250 MeV kernel settings with the resolution evaluated at 260 MeV. It extends beyond the inherited extraction grid and is not a newly validated scan endpoint.

The deterministic controls isolate the effect of signal in the training bins: the clean-sideband incremental response is essentially one, while the injected-sideband response is lower. The report presents this comparison together with raw yields, zero-signal fits, pulls and interval containment. With 40 toys per cell, pull widths and containment fractions still have appreciable sampling uncertainty; these are conditional checks, not a coverage calibration.

## Method

The background is sampled as a full Poisson spectrum from the pinned v5.8.2 nominal 2021 GP mean. One generating mean is used for all masses and both extraction methods. Independent background draws are made per mass and toy; each draw is shared across injection yields and extraction methods.

Each signal draw has exactly N selected candidates. A multinomial draw uses the native smeared-MC histogram rebinned by its CDF onto the analysis bins, together with below- and above-support categories. Histogram overflow remains outside the support. The fitted parameter is the full-selected candidate yield, so the signal probabilities are never renormalized within the fitted window. Counts entering the support, fit window, training bins and outside support are saved for every toy.

Both methods use the unchanged native MC shape and the same half-width, 2.25 times the nominal resolution at the generated mass. Only the fit and training window center changes: the pole mass for one method and the fixed MC-only core estimate for the other. The GP mean and correlated covariance are recomputed from each toy's exterior bins; the archived kernel hyperparameters remain fixed.

The likelihood combines Poisson counts with a correlated Gaussian constraint on the background. Signed fitted yields are retained. Pulls use the observed profile-Hessian error. Nominal 68.27% and 95% likelihood-ratio containment uses thresholds 1 and 3.841459; tables provide counts and Clopper–Pearson 95% intervals. The exact-N signal is multinomial, whereas the inherited extraction likelihood is Poisson, so unit pull width and nominal containment are reference values even in the known-background control.

The selected MC is conditional on the supplied v13 production. Signal-daughter association and complete equivalence to the v16 data selection are unvalidated. Templates and centers are fixed, and this study does not propagate finite-MC, detector-response or generating-background uncertainty.

## Files and reproduction

- `pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf`: the LaTeX note, formatted consistently with v6.1 and v5.0.5.
- `source/report.tex`, `source/generated/`, `figures/`: editable note, numerical tables and vector figures.
- `results/toys.csv`, `summary.csv`, `paired.csv`, `asimov.csv`: complete fitted results, moments, containment and controls.
- `results/checkpoints/`: saved draws, masks, probabilities, per-mass fits and checksums.
- `qa/editorial_review.md`: independent review of the note's wording and scientific claims.
- `qa/independent_validation.json`, `toy_extension.json`, `report_visual_qa.json`: numerical, extension and rendered-page checks.
- `inputs/`, `provenance/`, `MANIFEST.sha256`: pinned dependencies and artifact identities.

Rebuild the scientific products and note with a Python environment satisfying `requirements.txt` and Tectonic installed:

```sh
python3 /path/to/v6p2_mc_injection_20260923/scripts/reproduce.py
```

To rebuild only the tables, figures and PDF:

```sh
python3 scripts/make_report.py
bash scripts/build_note.sh
```

The full launcher limits work to four local workers with one numerical thread each and a 30-minute watchdog. No S3DF jobs are used. Checkpoint dependencies and output hashes are verified before reuse. The final PDF is delivered only after text and rendered-page review.
