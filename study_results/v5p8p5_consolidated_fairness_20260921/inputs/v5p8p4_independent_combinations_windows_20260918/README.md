# HPS-GPR v5.8.4: independent combinations, spacing and blind windows

Read `source/report.pdf` (11 pages). Twelve PDF/PNG figure pairs include the comparisons in the report plus absolute model-Asimov reach and global-Z width curves. The main numerical ledger is `results/significance_and_reach.csv`; `results/peaks.csv` contains every scope/domain/width peak.

## Scope

The study compares integer-MeV scans at blind half-widths 2.25, 2.4, 2.5 and 2.6 times the reviewed resolution. The same mask controls GP exclusion and the likelihood signal window, as in the original fitter. The original signal template is restricted to each window without renormalization. Source spectra, kernel states, supports and physical yield conversion are fixed to the pinned v5.8.2 inputs.

Individual ranges: 2015 19-100 MeV; 2016 39-180 MeV; 2021 10% 50-250 MeV. Combined full domain: 19-250 MeV using active datasets. The separately declared common overlap is 50-100 MeV. All old releases are unchanged.

The independent alternatives are pointwise Fisher with an exact nonpositive-fit atom mixture within the Gaussian response approximation, and equal-weight Stouffer of continuous signed reference scores. Their scan maxima retain correlations across mass. Combining individual experiment-global p estimates is a separately labeled diagnostic that allows different peak masses. Selection among methods, widths or scopes is not calibrated.

Reach is the observed and local-model-background Asimov 90% CLs coupling endpoint, for individual experiments and the physical common-coupling model. There is no invented one-dimensional Fisher reach: a signal-yield model is needed to define one.

## Rebuild from saved results

Python needs NumPy, SciPy, pandas and Matplotlib. Tectonic builds the PDF. Exact versions used are recorded in `qa/environment.json`.

```sh
python3 scripts/make_figures.py
tectonic -X compile source/report.tex
python3 scripts/validate_outputs.py
```

`python3 scripts/check_rebuild.py` performs an isolated saved-results rebuild and compares CSVs, PDF text and rendered pixels. It additionally needs pypdf, Pillow and pdftoppm. `STUDY_SCIENCE_PYTHON` can select a science Python when PDF libraries are provided by another interpreter.

## Replay the numerical study

All fitting code, spectra, fixed source means and 256 full-spectrum Poisson samples per dataset are bundled. The baseline response fields are frozen inputs. Three single-BLAS-thread workers own disjoint width/mass coordinates; existing checkpoints are reused.

```sh
python3 scripts/check_width_response.py
python3 scripts/check_combinations.py
python3 scripts/run_widths.py 0 3
python3 scripts/run_widths.py 1 3
python3 scripts/run_widths.py 2 3
python3 scripts/analyze_widths.py
python3 scripts/check_peak_composition.py
python3 scripts/validate_outputs.py
python3 scripts/make_figures.py
tectonic -X compile source/report.tex
```

The three fitting commands may run concurrently. To force fresh fits, use a separate copy with its checkpoints removed. `analyze_widths.py` uses 100,000 fields per scope/width with fixed documented seeds. Its historical wait deadline only protects the original bounded run; if inputs are incomplete during a future replay, finish the workers before invoking analysis. All required spectra are pinned; no ROOT files or parent checkout are needed.

Inputs are copied with SHA-256 provenance. `engine.py` is the only modified copied engine module: `Context` adds a width argument (default 2.25). Both the original and modified hashes are recorded. Baseline replay differences across numerical runtimes are bounded at 1.32e-6 in the signed root. Initial overly strict 1e-7 replay checks are documented as resolved in `qa/initial_diagnostics/`; the final audit covers all baseline coordinates.

## Validation and limits

There are 928 checkpoints, 2,628 fitted scope/mass configurations, 5,096 displayed rows and 36 peak summaries. Each dataset has 256 independent full-spectrum Poisson draws, reused coherently at every mass and every width; these are not independent ensembles across methods or widths. The Gaussian fields for a shared-coupling scope and the independent alternatives are not a jointly calibrated search over methods.

All 124 numerical/provenance checks and 264 finite-difference checks pass. One statistical comparison is explicitly flagged: 2016 at width 2.6 has Gaussian peak-global p=0.09547, versus 35/256 direct exceedances (95% interval 0.0971-0.1850). The 95%-maximum threshold check gives 14/256 and is compatible with 5%. This modest discrepancy is unresolved at the available direct-scan precision. Independent-combination rare tails are also sparsely sampled directly; zero direct counts retain a probability upper bound.

Every significance remains conditional on fixed observed-data-derived GP sources. Full source-estimation uncertainty, source-signal absorption, correlated physical systematics outside the specified model and selection among alternatives are not calibrated. Model-Asimov limits are deterministic local reference endpoints, not validated ensemble medians or expected bands. The half-MeV versus integer-MeV comparison uses 200,000 paired previously saved fields; the new 100,000-field baseline can differ slightly by Monte Carlo noise.

Primary references are linked in the report: Ananiev and Read's significance-GP method, SciPy's documented Fisher/Stouffer assumptions, and Cowan et al. for the profile-likelihood statistics. The Fisher atom mixture is derived explicitly in the note and checked in `scripts/check_combinations.py`.
