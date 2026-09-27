# HPS-GPR v5.8.5.3: unshifted significance scans

This 24-page revision explicitly answers the calibration question and adds the requested plots to the consolidated v5.8.5 report. All new work is isolated in this directory. The earlier study is unchanged.

**Yes: the earlier reference-local curves applied `(r-a)/s` in 2015, 2016, 2021, and the shared-coupling combination.** Saying those diagnostic results were not adopted for production did not mean they had not been calculated or plotted.

The new observed local plots use only `Z=max(r,0)` and `p=norm.sf(Z)`, without reference subtraction or rescaling. They reproduce the archived nominal/unshifted columns at all 1,310 mass/scope coordinates. This is the conventional asymptotic display, not a new empirical proof that raw roots are standard normal. Nonpositive fitted amplitudes display p=0.5 and Z=0; the inclusive zero-statistic probability is a different convention.

## Where to look

- Page 2: explicit clarification, formulas and all-dataset peak table.
- Page 3, Figure 2: requested unshifted local p-value and Z plots for 2015, 2016, 2021 native10, and shared coupling.
- Pages 4–5, Figure 3: separate fixed-source global curves for the **raw maximum**, preserving observed peak ordering. These global probabilities still require a specified null source.
- Pages 12–15, Figures 6–9: all four versions of the response diagnostic corresponding to Figure 4 of the previous consolidated v5.8.5 report. Panels A/B are raw; C/D standardize toys only.
- Other pages retain the earlier conditional studies, clearly separated from the newly requested local curves.

The unshifted peaks are:

| Scope | Mass (MeV) | Local Z | Local p |
|---|---:|---:|---:|
| 2015 full | 51 | 3.139237 | 0.000846942 |
| 2016 full | 90.5 | 3.452500 | 0.000277709 |
| 2021 native 10% | 78 | 2.808645 | 0.002487524 |
| Shared coupling | 66 | 2.760159 | 0.002888665 |

The combined raw curve is the one-shared-coupling fit. The earlier Fisher, Stouffer and free-amplitude results have separate definitions; the report now explicitly states where their reference calibration entered.

## Scope

- Fixed blind half-width: 2.25 sigma.
- Mass grid: 0.5 MeV, using each dataset's saved support.
- No new fits, optimized kernels, or random experiments.
- Global raw-ordering curves reuse 200,000 saved Gaussian raw maxima and 256 complete Poisson scans per scope. Their confidence bands describe finite simulation counts, not source uncertainty.
- No replacement production significance or limit is claimed.

## Rebuild

Python requires NumPy, SciPy, Matplotlib and pypdf. Tectonic builds the LaTeX report. The validated scientific interpreter on this machine is `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`; another compatible interpreter can be used elsewhere.

From this directory:

```bash
python3 scripts/make_raw_plots.py
python3 scripts/make_tables.py
bash scripts/build_report.sh
python3 scripts/validate.py
```

The new figures and report require only the included files, not a live parent checkout. Earlier numerical studies are retained as saved results and figures; their original full-refit packages remain separate.

## Contents

- `source/`: complete revised LaTeX report and compiled PDF.
- `figures/raw_local_overview.*`: requested unshifted local p/Z overview.
- `figures/raw_local_{2015,2016,2021,combined}.*`: individual unshifted local plots.
- `figures/raw_global_overview.*`: distinct fixed-source raw-maximum global plots.
- `figures/response_diagnostics_{2015,2016,2021,combined}.*`: expanded Figure 4 diagnostics.
- `inputs/fields/`: the four exact frozen response fields and observations.
- `inputs/v5p8p2_significance_curves.csv`: independent archived nominal-column comparison.
- `inputs/prior_review_results/`: previous review's numerical ledgers.
- `inputs/v5p8p5_prior_report.pdf`: the previous consolidated report.
- `results/`: all new per-mass curve values, toy moments, peak summaries and input hashes.
- `qa/`: numerical/text checks, rendered-page review, standalone rebuild and parent hash comparison.

The source/data archive includes a SHA-256 manifest. Validation checks the raw formulas against the archived nominal columns, verifies raw-max tail counts, and inspects all rendered pages. No previous release is overwritten.
