# HPS-GPR analysis note v4.9.13

Completed on 5 September 2026. This is the LaTeX successor to the original
four-page background-profile report, with the title, margins, typography,
status box, equations, numbered figures, tables, and references of the
v4.9.12 analysis notes. Earlier studies and their PDFs remain unchanged.

The 15-page deliverable is
`output/pdf/v4p9p13_background_profiling_20260905/HPS_GPR_Analysis_Note_v4p9p13_background_profiling.pdf`
from the repository root. Source: `note/analysis_note.tex`.

## Findings

The literature supports a Poisson likelihood with correlated GP background
constraints, while calibration remains a property of the complete procedure.
Direct positive log-GP profiling changes the 2021 observed limits by at most
1.189% relative to the release, or 0.232% relative to Gaussian profiling with
the same stable solver. The Gaussian numerical control differs from the
release by at most 1.139%. These distinctions remain visible in the note.

Fixing the mean gives substantially narrower conditional limits:

| Scope | Fixed/released observed median | Fixed/profiled conditional Asimov median |
| --- | ---: | ---: |
| 2015, 100% | 0.422 | 0.412 |
| 2016, 100% | 0.522 | 0.523 |
| 2021, 10% | 0.607 | 0.613 |
| All three, shared coupling | 0.595 | 0.524 |

These are not calibrated sensitivity improvements. Signal extraction works
when the fixed background is exactly true. With GP uncertainty propagated in
generation, fixed pull widths are 1.85-2.65, and its nominal 90% CLs limits
exclude the stronger true injection in 22.0-32.2% of toys. The corresponding
profiled range is 8.2-12.2%.

Retraining exposes an additional limitation in both methods. At 71 MeV, the
stronger injected signal is excluded in 349/500 fixed toys (69.8%) and 145/500
profiled toys (29.0%). This selected smooth truth is a conditional stress test;
it is not a measurement of bias in the observed spectrum. The current method's
failure is retained prominently, not removed by its good same-model closure.

The omitted-variance factor kappa ranges from 1.72 to 4.30 across 2021. After
propagating that covariance, the fixed estimator's linearized standard error
is a median 1.151 times the profiled error. A local scale can correct variance
under a specified model, but does not repair bias, calibrate discovery tails,
recalibrate upper limits, or supply a global trials correction. No observed
p-value or limit has been empirically rescaled.

## Contents

- `comparison/figures/`: restyled 2021 three-method limits, numerical/model
  separation, and four selected fits with consistent residual baselines.
- `observed/figures/`: four-scope observed limits, local asymptotic p-values,
  and deterministic Asimov comparisons. `observed/README.md` gives the
  per-scope minima and validation details.
- `observed/derived/observed_fixed_comparison.csv`: all 456 observed scope/mass
  rows, released references, fixed fits, p0 and log p0, raw/displayed limits,
  and inherited status. The combination is one exact shared-coupling
  likelihood, not a combination of separate limits or p-values.
- `injections/derived/toy_results.csv.gz`: all 54,000 fits on 27,000 spectra.
  `extraction_summary.csv` has 108 grouped results; the holdout and Fisher
  tables retain the local-scale checks. `injections/PROTOCOL.md` specifies
  the design frozen before generation.
- `note/`: LaTeX source, data-generated tables, source ledger, and build log.
- `qa/`: numerical validation and rendered-page review record.
- `MANIFEST.sha256`: hashes of this study's deliverables and final PDF.

## Scope and limitations

All 415 native observed predictions exactly reproduce the v4.9.12 ledger.
Full 2016 and the combination retain the previously disclosed numerical
exception; they have not gained independent certification. Local asymptotic
p-values are included for every requested scope. Fixed-model p-values assume
the estimated GP mean is exactly known and are not calibrated significance.

The injection study uses 500 toys at each of six 2021 masses, three strengths
(0, 2, and 5 reference profiled Fisher errors), and three generating models:
known mean, conditional GP uncertainty, and retrained sidebands. Methods are
paired on each spectrum; strengths and masses use separate pointwise ensembles.
The retrained truth is the saved v4.9.12.5 reverse-injection spectrum fitted
outside 60-86 MeV with the 66 MeV kernel. Posterior means and count-dependent
training errors are refitted, but kernel coordinates are frozen. This is not
unconditional coverage, hyperparameter-reoptimized closure, validation of
2015/2016 injections, or a scan-wide significance calibration.

The held-out calibration uses 100 background-only training toys and 400 test
toys per mass/model. Its binomial intervals condition on that fitted correction;
they do not average over repeated training sets. At positive injection,
`false_positive_fraction` in the summary CSV is power; it denotes a false
positive only at zero signal. `mean_signal_response` subtracts independently
generated ensemble means across strengths. See `injections/DATA_DICTIONARY.md`.

## Verification

The saved-artifact validator passes 76 checks, including all toy coordinates,
pairing, strengths, pulls, local p-values, truth exclusion, the holdout split,
and protected source hashes. The observed extension has 30 further persisted
checks, including 456 independent scalar checks and 1,368 successful
observed/Asimov limits. Previous fixed-2021 limits reproduce within 8.1e-14
relative. The 216 shortcut/full-limit classifications all agree. No numerical
failure or toy was dropped. Scientific closure failures are results and remain
in all tables and plots.

Tectonic builds without warnings or unresolved references. All 15 pages are
rendered with Poppler and visually inspected; the page review and semantic
checks are retained under `qa/`.

## Reproduction

Run from the repository root, sequentially. The numerical scripts limit their
libraries to one thread. The observed scan takes about 16 seconds and the
27,000-spectrum injection study about 78 seconds on this machine.

```bash
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/observed/run_observed.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/observed/make_figures.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/observed/validate_observed.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/injections/run_injections.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/injections/make_figures.py
nice -n 10 python3 -B study_results/v4p9p13_background_profiling_20260905/comparison/make_figures.py
python3 -B study_results/v4p9p13_background_profiling_20260905/build_note.py
python3 -B study_results/v4p9p13_background_profiling_20260905/validate_study.py
```

The injection script resumes its frozen checkpoints. Do not change code,
protocol, seed, truth, or strengths and reuse those checkpoints. For a fresh
replication, copy this study into a new derivative and use an empty
`injections/derived/` directory. The original run summary records all 216
full-limit checks; on a checkpoint-only resume its per-run counter is zero,
while the persisted rows and the validator retain the original evidence.
`injections/pilot/` is the separate implementation pilot and is excluded from
the reported toy counts.

## Further scientific work

The requested note and small studies are complete. Adopting a replacement
procedure would require a new predeclared study: independent smooth and
localized-contamination truths, repetition of the complete training and
selection procedure, and local calibration before a coherent global scan.
A simultaneous positive functional-background fit following published HPS,
or a predeclared discrete-profile family, is a meaningful next comparison.
No such model was implicitly substituted for the direct log-GP calculation.

## Coordination

The task `Commit studies and start bands` established separate ownership of
`v4p9p12_targeted_tail_refinement_20260905/`. This task stayed in the v4.9.13
derivative and its matching PDF output directory. A parallel subagent owned
only `observed/` and supplied a bounded statistical review of the injection
code. No production file, previous note, or parent validation was rewritten;
unrelated dirty-checkout work was preserved.
