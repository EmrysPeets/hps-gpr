# Binning and significance fields — v5.1.0

Independent diagnostic report, 9 September 2026. This package is separate from the v5.0.2 analysis note and does not change a production result.

The nine-page report compares three histogram widths in each of 2015, 2016 and 2021, evaluates six new complete significance-response matrices, tests every phase of 1/2/5 MeV mass grids, and propagates illustrative neighboring-bin correlations. It explains the signed discovery statistic, the weak/strong upper-limit toy probabilities, and why perturbing an observed significance curve does not define a background-only experiment.

The 2016 stress bias persists at every tested histogram width. Conditional Gaussian probabilities are diagnostics under specified stress backgrounds, not calibrated discovery significances. The input-bin correlations are illustrative, not measured. No binning is selected by its observed probability; the original binning calculations used no new Poisson toys. The support follow-up adds 64 paired Poisson spectra at six fixed masses under three supports; it does not evaluate a complete mass-scan maximum or run an injection campaign.

## Files

- `report.tex` and `report.pdf`: source and inspected final report.
- `protocol.json` and `covariance_followup_protocol.json`: fixed choices and resource bounds.
- `inputs/manifest.json`: original source paths, bundled snapshots and SHA-256 identities. Original ROOT files are represented by their recorded provenance; full-fit reproduction needs those external files.
- `derived/rebinned_scans.csv`: all 1,245 dataset/mass/binning coordinates (830 newly fitted coarse coordinates, each for observed and stress spectra).
- `derived/response_*/`: six complete matrices with all 830 coordinate checkpoints.
- `derived/binning_global_summary.csv`: conditional Gaussian tail estimates and Monte Carlo standard errors for every histogram choice.
- `derived/grid_tests.csv`: all grid phases, retained observed peaks, fixed-threshold probabilities, intervals and two-sided probabilities.
- `derived/correlated_noise_tests.csv`: illustrative input-covariance sensitivity results.
- `derived/field_summary.csv`: baseline raw, affine-stress and saved direct-stress comparisons.
- `derived/*tail*.npz` and `derived/noise_*.npz`: importance-sampling weights and thresholds.
- `qa/validation.json`: 27 numerical and semantic checks, bound to the PDF hash.
- `qa/visual_review.json` and `qa/portable_build.json`: rendered-page and independent-directory rebuild checks.
- `SHA256SUMS.txt`: package file identities.

## Rebuild

With Python 3, NumPy, SciPy, pandas, Matplotlib, pypdf and Tectonic installed, the bundled results suffice for:

```sh
python3 scripts/summarize.py
python3 scripts/summarize_support_2016.py
bash scripts/build.sh
python3 scripts/validate.py
```

The compiled PDF is `qa/build/report.pdf`. `scripts/common.py` resolves saved source requests through `inputs/manifest.json`; the existing snapshots are hash-checked. Regenerating the original pointwise and perturbation fits additionally requires the HPS checkout, its dependencies, and ROOT inputs at the paths recorded in the scripts/manifests:

```sh
python3 scripts/rebin_fits.py
python3 scripts/rebinned_response.py
python3 scripts/field_tests.py
```

These fit scripts reuse existing checkpoints. To reproduce fits from scratch, use a separate copy of the package and remove only that copy's relevant derived checkpoints first. The recorded run used one process and one linear-algebra thread. Full response fitting took about 4.9 minutes and 243 MiB peak resident memory; the brief baseline Gaussian-field pass peaked at 1.46 GiB.

The prior sigcorr executable cross-check is included as a hashed input. Its conclusion concerns covariance and sampling algebra; the nonzero stress mean and positive-fit gate remain explicit HPS extensions.

## 2016 support follow-up

Pages 7–8 test 30/31/32/33/34 MeV lower edges, two shorter fixed kernel length scales, and two exposure scalings. The fixed protocol is `support_2016_protocol.json`; `derived/support_2016/` contains all 1,278 deterministic fits, the 64 saved paired spectra, 1,152 pointwise toy roots, window residuals, summaries and paired support changes with Monte Carlo errors. Reproduce the fits with `python3 scripts/support_2016.py`, requiring the original HPS inputs. This one-process follow-up took 36.4 seconds and 376 MiB peak memory.

The 32 and 33 MeV edges modestly reduce overall stress RMS to 4.67 and 4.58 from 4.97. The 61–90 MeV region remains strongly biased, while the high-mass RMS increases. No support or kernel change is adopted as a qualified production setting.
