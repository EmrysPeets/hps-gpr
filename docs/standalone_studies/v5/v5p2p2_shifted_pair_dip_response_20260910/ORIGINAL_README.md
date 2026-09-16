# HPS GPR v5.2.2: shifted pairs and upper-limit dips

Standalone deterministic continuation of v5.2.1. Open `report.pdf`; the complete design is `PROTOCOL.md`. Inputs are the full2015/full2016/native10%2021 analysis histograms with the retained resolution, yield conversion and archived GP states. This version draws **zero new toys**. It preserves the earlier ten-toys-per-scheme-and-strength pilot in its parent report.

## Reproduce

Use Python3.9+ with NumPy, SciPy, pandas, matplotlib, scikit-learn, PyMuPDF and Pillow; Tectonic builds the PDF. The original checkout uses `venv/bin/python`. Run from the study folder, substituting your Python executable:

```bash
python scripts/fit_families.py
python scripts/response_scan.py
python scripts/response_scan.py --summarize-only
python scripts/plot_response.py
python scripts/make_report.py
bash scripts/build.sh
python scripts/render_verify.py
python scripts/portable_check.py
python scripts/package.py
```

Response scans use two workers and checkpoint each scenario in `derived/response_chunks`. Existing chunks are reused. To intentionally regenerate changed response inputs, move the old chunks to a separate directory first; do not mix results from different scripts or inputs. The final archive manifest pins the actual numerical source and every stored artifact.

## Data products

- `family_summary.csv`, `family_year_estimates.csv`:18 nested mass/rate fits on fixed experiments; all boundary flags retained.
- `family_*_full.npz`:14 full-support fitted continuum/mean spectra, templates, yields, selected masses and GP constraints; no amplitude floors.
- `response_design_*.json`, `response_yields_*.csv`: all declared injections, full-support yields, reconstructed masses and equivalent coupling coordinates.
- `response_limits.csv.gz`: all deterministic and observed0.5MeV CLs solutions, fitted signed roots, fixed-mass asymptotic tail references, profile diagnostics and aligned mass maps.
- `response_metrics.csv`: predeclared central-third and whole-window shape diagnostics. External-pair injection ratios use the external baseline; other main ratios use the shared-fit continuum baseline. These correlated-point shape metrics are not probabilities.
- `qa`: independent family/solver/response reviews, numerical checks, page renders/contact sheets, and portable rebuild record.

`inputs/manifest.json` is an unchanged historical ancestry ledger: its original source paths and references are not all needed or included here. Current portable inputs are `spectrum_*.npz`, `scopes.json` and the local scripts; current file identities are given by `MANIFEST.json`. `inputs/parent/report.pdf` preserves v5.2.1 and its interpretation/qualification caveats. The original ROOT files are not required for deterministic reproduction from the exported bins.

This is an observed-motivated morphology study. Postfit baselines, selected masses, shared-rate assumptions, fixed kernel/resolution/conversion choices, the inherited2016 qualification exception, and the2015extension to100MeV remain explicit limitations. No new coverage, GP-field calibration, global significance or calibration-offset measurement is claimed.
