# HPS-GPR v6.3.9: combined observed scan with morphed 2021 signal

Read `pdf/HPS_GPR_v6p3p9_Combined_Morph_Scan.pdf`. This four-page study is Appendix K at the end of the updated v6.3.6 note.

The observed scan covers 60–240 MeV in 1 MeV steps. It combines full 2015 and 2016 with the 2021 10% sample, retaining the earlier 2015/2016 models. Campaign support remains 2015 through 100 MeV, 2016 through 180 MeV, and 2021 through 240 MeV. Only 2021 uses the new neighboring v16 TC template and [-4,3] core-width fit/GP-exclusion window. Comparisons retain the shifted Gaussian in its original window and in the new window.

One common parameter psi = epsilon^2 / 1e-8 multiplies campaign-specific signal arrays, with independent GP nuisance blocks. The signal conversion and above-threshold visible multiplier are inherited assumptions, not newly validated physical normalization. All 2021 MC probabilities retain their full selected denominator; the fitted window is never renormalized. There is no new selection-equivalence validation or propagation of signal-template, normalization or GP-source estimation uncertainty. These are conditional observed comparisons, not a physical exclusion release or global significance calibration.

The new combined minimum is at 68 MeV: local asymptotic Z = 2.910 and p = 0.001806. A separate fixed-mass check uses 1,000 independent joint GP-mean Poisson backgrounds, paired across methods. Zero MC-method exceedances give rank p = 1/1001, with an exact two-sided 95% binomial interval [0, 0.003682]. This is a finite-sample bound after selecting the observed mass, not zero probability or a global p-value.

## Rebuild

Requirements: Python with NumPy, SciPy, pandas and matplotlib; Tectonic with its resources cached. The verified runtime is `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`. Tectonic is resolved from `TECTONIC`, PATH or the Homebrew fallback.

```bash
V639_PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3 bash scripts/rebuild.sh
```

Cached checkpoints are checked against a frozen protocol. A fresh run can be made in a copy by preserving and removing that copy's `results/checkpoints/` and `results/toy_checkpoints/` directories. Numerical threads are bounded to one. No S3DF access is required.

## Saved evidence

- `results/combined_scan.csv`: 543 new joint fits, 162 older-campaign individual fits and 543 inherited 2021 fits converted to the shared coordinate.
- `results/normalization.csv`: event-density conversion and visible multiplier by mass and campaign.
- `results/2021_template_geometry.csv`: full normalization, fit containment and GP-training leakage.
- `results/combined_local_toys.csv`: 3,000 joint background-toy fits at 68 MeV.
- `results/combined_local_calibration.csv`: exact tail counts, rank probabilities and binomial intervals.
- `results/replays/`: joint profile components and limit traces at selected masses and campaign boundaries.
- `results/legacy_combination_agreement.csv`: all 181 original-window Gaussian points reproduce the previous combination.
- `qa/`: likelihood-factorization, Hessian, tail-count, toy-replay, input-integrity, portable-rebuild and visual checks.
- `provenance/`: frozen protocol and input hashes. The copied v6.3.8 runtime supplies the 2021 contexts; `run_combined.py` is the entry point for this study.

The two figures show the combined model comparison and campaign contributions, each with upper limits and local p-values. Legends are outside the plotting areas, and the caption supplies the reading instructions.
