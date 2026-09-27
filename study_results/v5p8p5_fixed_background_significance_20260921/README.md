# The outside statistician's fixed-background extension

This report adds actual no-background-nuisance (`C=0`) fits to the earlier outside-statistician review. It covers 2015 full (19–100 MeV), 2016 full (39–180 MeV), 2021 10% (50–250 MeV), and two common-mass combinations: a shared coupling and independent nonnegative amplitudes. Combination results distinguish the full union (19–250 MeV, varying membership) from the all-three overlap (50–100 MeV).

The inherited GP is trained on masked sidebands for every observed or toy spectrum. Its predicted arithmetic mean is held fixed inside the signal likelihood. Masks remain ±2.25 mass-resolution sigma; the grid is 0.5 MeV. Original kernels, supports and signal normalization are retained. No deterministic reference response is subtracted from the data and no response width rescales its statistic.

Nominal likelihood p-values and actual complete-toy tail counts are separate outputs. The 256 coherent Poisson null spectra come from the frozen nominal v5.8.2 source. Their GP sidebands are retrained, exposing fluctuations omitted by the fixed-background likelihood. These checks remain conditional on that source and inherited analysis choices; they do not propagate uncertainty in building the source, or establish physical discovery probabilities. Zero exceedances are unresolved tails with explicit upper bounds.

## Files

- `source/report.pdf` and `source/report.tex`: revised statistician's report; new section in `source/fixed_background.tex`.
- `figures/fixed_background_curves.pdf`: all requested nominal local p-value curves.
- `figures/fixed_background_comparison.pdf`: fixed versus profiled raw nominal mappings, without reference centering.
- `results/significance_curves.csv`: all masses, nominal local p-values, conditional local/global counts, intervals, fitted amplitudes and null diagnostics.
- `results/peaks.json`: strongest nominal excursion in each specified scope and domain, with complete-scan tail counts.
- `results/fields_*.npz`: observed statistics and all 256 complete toy fields.
- `results/profiled_raw_{scan,peaks}.csv`: paired comparison from saved profiled fields, using the same raw-statistic ordering.
- `inputs/parent/`: portable pinned engine, spectra, source means, complete toy counts and prior profiled fields.
- `inputs/earlier_review/`: exact sources for the original statistician's perspective.
- `reviews/`: the statistician's judgement and source audit.
- `qa/`: independent solver/prediction comparisons, likelihood nesting and tail-count checks, rendered pages and package validation.

## Reproduce

The numerical and plot scripts run from their bundled inputs, without the original checkout. They need Python with NumPy, SciPy, pandas and Matplotlib. This run used the Xcode Python runtime; scripts constrain numerical-library threading.

```bash
python3 scripts/run_fixed_scan.py
python3 scripts/profiled_raw_comparison.py
python3 scripts/independent_numerical_qa.py
python3 scripts/validate_scan.py
python3 scripts/make_figures.py
bash scripts/build_report.sh
```

The scan resumes completed per-mass checkpoints and checks a task-local `STOP` marker before batches. `build_report.sh` needs Tectonic and Poppler. The `prepare_report.py` parent-copy step is unnecessary for an archived reproduction. `wallclock_guard.py` records the original task's resource deadline and is not part of numerical reproduction. Package validation additionally checks the earlier release against its original-checkout manifest. Final rendered-page inspection is recorded against the exact PDF hash.

The earlier released reports and all production analysis files remain unchanged. The general likelihood mapping follows Cowan et al., [arXiv:1007.1727](https://arxiv.org/abs/1007.1727). The earlier field-GP approach is discussed by Ananiev and Read, [arXiv:2206.12328](https://arxiv.org/abs/2206.12328); this extension's new empirical tails use complete Poisson toy scans directly.
