# HPS-GPR v5.3.1: one-sided resolution nuisance and 2015 extension

Read `report.pdf`, a traditional LaTeX report following the preceding v5.2 study style. Its editable source is `report.tex`; vector figures and plotting sources are included.

This continuation preserves the completed v5.3 study and adds:

- 2015 hypotheses through 100 MeV, with the 90–100 MeV extension highlighted. Actual observed bins and the 14–135 MeV training support are unchanged. Above 90 MeV, the reviewed 90 MeV kernel supplies a starting candidate before the same sideband-likelihood selection and refitting procedure.
- Connected combined-limit curves. All three datasets contribute through 100 MeV; the 2016+2021 combination begins at 101 MeV.
- Probability axes labeled with actual local p-values on a logarithmic scale.
- An individual-dataset resolution nuisance profiled continuously in the observed likelihood and the background-Asimov likelihood, with the same fixed bins, GP mean/covariance and coupling conversion throughout each calculation.

The main nuisance model takes sigma(t) = sigma_MC + t (sigma_scaled - sigma_MC), with 0 <= t <= 1 and an assumed constraint exp(-t^2/2). The full correction is therefore one constraint unit from MC, with a hard upper bound. This is an explicit sensitivity assumption; the Moller width discrepancy has not been interpreted as a measured calibration standard error. A flat bounded-width model is included as a control. Neither construction has new coverage or width-search p-value calibration. Combined results remain resolution scenarios; combined nuisance profiling is not introduced.

There are 425 individual mass coordinates and 850 nuisance results. Relative to an MC-width fit on the same fixed experiment, the Gaussian-constrained nuisance increases the median observed limit by 15.3% (2015), 55.6% (2016), and 7.5% (2021). The report separately shows the change from the previous moving-mask MC experiment, so changes in background geometry are not attributed entirely to the width nuisance.

## Numerical products

- `derived/scans.csv`: the extended four-width scans, with three background-fit control lanes; 2,628 rows per lane.
- `derived/resolution_nuisance.csv`: both continuous nuisance models, fixed-experiment endpoint controls, the previous moving-mask results, fitted widths and numerical diagnostics.
- `derived/resolution_nuisance_controls.csv`: the fixed MC/scaled controls on exactly the same bins and normalization.
- `derived/resolution_nuisance_summary.json`: descriptive ratio and p-value summaries.
- `statistics_nuisance_review.md`: current statistical construction and its assumptions.
- `qa/`: source, extension, solver, dense-width-grid, text and rendered-page checks.

The raw epsilon-squared columns use the inherited electron-channel normalization. Report plots apply the previously released dimuon correction once, identically to every relevant curve; p-values and event yields are unaffected.

## Rebuild the report

Requires Python with numpy, pandas, scipy and matplotlib, and Tectonic with the standard LaTeX packages used by the earlier reports. From this directory:

```sh
python3 scripts/collect_resolution_nuisance.py
python3 scripts/make_latex_report.py
```

The plot builder writes `report.tex` and compiles `report.pdf`. It does not refit data. The supplied `report.tex` and vector `figures/` also permit direct compilation:

```sh
tectonic --keep-logs report.tex
```

The likelihood inputs are bundled histogram arrays; original ROOT paths in the analysis card are provenance only. To rerun the nuisance calculations, use a new copy of this study, move `derived/nuisance_chunks` aside, and run the driver with a future deadline:

```sh
python3 scripts/run_resolution_nuisance.py --deadline-utc YYYY-MM-DDTHH:MM:SS+00:00
python3 scripts/collect_resolution_nuisance.py
```

The extended scenarios are checkpointed under `derived/chunks`. Earlier v5.3 chunks outside 91–100 MeV are reused byte-for-byte; source and script identities distinguish inherited and newly computed fits. The original v5.3 report and manifest remain pinned under `inputs/`. No toys or expected-limit bands were generated.
