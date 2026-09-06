# v4.9.15: full 2016 and 2021 10% global-significance study

The [LaTeX reader report](../../output/pdf/v4p9p15_global_2016_2021_20260906/HPS_GPR_v4p9p15_Global_Study_2016_Full_2021_10pct.pdf) extends the frozen
2015 study. Its source is [note/reader_report.tex](note/reader_report.tex).
The implementation follows the covariance construction of
[Ananiev and Read](https://arxiv.org/abs/2206.12328v3), explicitly retaining
nonzero offsets and nonunit widths in the likelihood-root field.

Each dataset has ten full-spectrum pilot scans, 1,000 additional independent
Poisson validation scans, and its complete one-bin Asimov-response ensemble
(721 scans for 2016; 423 for 2021). The analyzer samples 200,000 GP fields per
method per dataset. The 2016 grid has 142 points from 39 to 180 MeV; the 2021
grid has 201 points from 50 to 250 MeV. Both have 1 MeV spacing.

## Principal minimum-local-p ordering

| Dataset | Statistic | Peak mass [MeV] | Common-truth local p | GP global p | Direct exceedances |
|---|---|---:|---:|---:|---:|
| 2016 full | profiled | 42 | 5.44e-22 | <1.5e-05 (95% upper bound) | 0/1000 |
| 2016 full | fixed | 43 | 2.53e-27 | <1.5e-05 (95% upper bound) | 0/1000 |
| 2021 10% | profiled | 92 | 0.000278 | 0.02519 | 26/1000 |
| 2021 10% | fixed | 77 | 1.76e-06 | 0.000205 | 1/1000 |

These are **conditional stress-background diagnostics**, not final discovery
probabilities or a global calibration of the v4.9.13 two-truth envelope.
The separate raw-root ordering is saved and plotted as a different test.
Zero-count tails are limits, not measured zero probabilities. A small 2016
conditional probability can reject the behavior of its particular archived
background construction without identifying a particle. Its source-fit
waiver, source-development overlap, transition region and inherited numerical
exception remain explicit. A raw global probability near one is not a
goodness-of-fit certificate. No combined-dataset or continuous-mass result
and no expected-sensitivity claim is made.

## Numerical implementation and checks

Exact pilot scans were completed first. The exact 2016 1,000-scan calculation
was paused after 81 complete mass columns because measured scaling was poor.
All those references are retained. The accepted derivative uses the existing
calibration accelerator with per-coordinate checks and an exact fallback for
the entire coordinate. The physical/statistical procedure, spectra, seeds and
mass grid stay the same. Replaying the exact and accelerated backend does not
create additional independent toys.

Every mass has an exact Asimov baseline and a declared bin-response stencil;
six masses per dataset have complete exact Asimov columns. The final audit
checks centered responses, widths and correlations, every available exact
pilot/validation root, their bounded-atom classifications, and all source
hashes. See [the numerical amendment](ACCELERATION_PROTOCOL.md),
[full-response gates](ACCELERATION_RESPONSE_GATES.md), and
[the independent HEP review](review/HEP_EXTENSION_REVIEW.md).

## Files and continuation

- `global_fast/<year>/`: accepted pilot, validation and Asimov products,
  numerical gates, full exact response checks, and execution records.
- `global_fast/<year>/analysis/`: p-value CSVs, covariance matrices, direct/GP
  maxima, marginal diagnostics, tail curves and summary JSON.
- `global/<year>/`: preserved exact pilots and the partial exact 2016
  validation reference. Do not pool these paired replays with accepted toys.
- `figures/`: separate p-value, mean/width, correlation and global-tail plots.
- `provenance/`: source references, timings, backend comparisons and PDF inputs.
- `qa/`: 83 product checks and review of every rendered PDF page.

[NEXT_STEPS.md](NEXT_STEPS.md) gives runnable reproduction and continuation
instructions, independent-seed requirements, finite-tail precision guidance,
finer-grid requirements and the conditions for a joint search. The complete
2015 manifest remains unchanged. `MANIFEST.csv` covers this derivative and its
final PDF; the manifest and its self-check companion are excluded from their
own inventory to avoid recursive hashes.
