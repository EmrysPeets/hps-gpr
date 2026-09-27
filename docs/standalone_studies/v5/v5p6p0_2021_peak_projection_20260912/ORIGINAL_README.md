# HPS GPR v5.6.0: 2021 conditional peak catalogue

Two source-conditioned 100%-equivalent Monte Carlo lanes: historical 2021 1% x100 and native 2021 10% x10. Each scenario has 20 independent Poisson spectra. Fifteen final scenarios give 300 final toys. Six mass windows requested by the user are examined in both lanes; three additional strong 1% local maxima are included. Regional endpoints are labeled; 82 and 84 MeV are overlapping alternative hypotheses. No 100% observed histogram was used.

The opening report derives the square-root exposure relation. Each main injection is numerically tuned to target Z=sqrt(k)*Z_source. The ordinary yield-scaled A=k*Ahat model is also evaluated as a separate deterministic comparison. Target matching is a construction, not evidence that statistical scaling must hold. Source peak selection is conditional on observations; 20 toys do not calibrate extreme tails, discovery probability, global significance, or a statistical upper bound. 1% selection equivalence and exact exposure relative to 10% are unverified.

## Contents

- `source/main.tex`: report source; figures are under `figures/`.
- `derived/catalogue.csv`: complete scenario summary, yields, target and recovered scores.
- `derived/source_scan.csv`: fresh source fits at 201 mass hypotheses in each lane.
- `derived/selected_peaks.csv`: deterministic scenario selection and targets.
- `derived/*_curves.csv`: Asimov plus 20 toy local scans per scenario.
- `derived/*_toys.csv`: fixed-mass and regional maximum summaries for each toy.
- `toys/*.npz`: all final spectra, source and future expectations, Gaussian signal, fit curves, bins.
- `inputs/`: frozen numerical inputs, source1% ROOT histogram, source-specific reviewed kernels and provenance.
- `consults/`: source/statistical review and implementation review.
- `qa/final_validation.json`: independent final numerical audit.
- `qa/superseded_subtractive_covariance/`: one 20-toy pilot superseded after an inherited high-count covariance cancellation failure. These are not included in the final catalogue.

## Reproduce

Python3 with NumPy, SciPy, pandas, Matplotlib, uproot, and PyMuPDF; Tectonic for LaTeX. All numerical commands force one linear algebra thread and run serially. Existing scenario caches validate numerical dependencies and reject stale inputs. To compute a completely fresh run, copy this package to a new directory and remove the generated files under `derived/` and `toys/` there (preserve `inputs/`). Then run:

```sh
python3 scripts/run_study.py prepare
python3 scripts/run_study.py toys
python3 scripts/validate_study.py
python3 scripts/make_report.py
cd source
tectonic --keep-logs main.tex
```

The report can be rebuilt directly from `source/main.tex` and saved figures without rerunning toys. Consult `qa/package_validation.json` for the archive check. The SHA-256 manifest covers the delivered study files. The notebook-free numerical engine is included locally; external original data paths in provenance are informational because their required inputs are snapshotted.
