# HPS-GPR v5.9: Gaussian cores with broader signal tails

Standalone study dated 21 September 2026, following the model and typography of Analysis Note v5.0.5. The observed samples are full 2015, full 2016 and the released native 2021 10% sample. Each dataset is fitted separately.

The study evaluates a Gaussian and six declared alternatives: 10%, 20% and 30% tail-scale increases in a smooth dilation family and a separate tail-curvature family. Every unnormalized template equals the nominal Gaussian inside the primary ±2.25σ window. Exact bin integrals are normalized over the inherited support, without renormalizing the fitted-window portion. Nominal detector resolution, prompt-density conversion and radiative normalization are held fixed.

Two window prescriptions are kept separate:

- **Primary:** the inherited ±2.25σ fit and GP training exclusion.
- **Guard:** a ±4.5σ fit and GP training exclusion, with GP prediction and covariance recomputed using only the external bins and the same archived kernel coordinates.

Every shape is compared with the Gaussian under the matching window prescription. Fits use the observed Poisson likelihood and correlated Gaussian background constraints. The reported upper limits solve bounded, piecewise-asymptotic `CLs = 0.10` with explicit observed and background-Asimov profiles. The local probability is `p0 = Phi(-max(r, 0))`, so a deficit has the inherited reference value `p0 = 0.5`.

The primary domains are 19–100 MeV (2015), 39–180 MeV (2016), and 50–250 MeV (2021). The guard domain for 2016 ends at 175 MeV: the five higher hypotheses fail the external-sideband geometric requirement under the wider mask. No missing guard coordinate is interpolated. The seven shapes across 425 primary and 420 guard coordinates give 5,915 observed fits.

All ratios use a freshly computed Gaussian under the same portable machinery. This baseline differs slightly from the older published table, with the largest scan-wide difference about 1.14% in the limit and 0.12366 in local Z. The historical-table comparison is recorded separately from the tail effects; exact reproduction of the old table is not claimed.

The controlled numerical audit identifies legacy optimizer stopping as the source of this difference. At the most affected coordinate, 2016/83 MeV, the old null fit stopped with a substantial remaining gradient; the converged fit changes local Z from about 0.24369 to 0.12003 under the same model. The covariance-factor representation has negligible effect in that comparison. This numerical issue is distinct from the inherited 2016 kernel-state provenance caveat.

## Read the study

The report source is `source/report.tex`. Figures are in `figures/`; numerical tables and scan ledgers are in `derived/`. Input identities and inherited export information are preserved in `inputs/` and `provenance/`, and numerical/artifact checks are in `qa/`. The delivery manifest identifies the exact source, inputs and generated outputs.

The report uses the v5.0.5 article format: 11-point Latin Modern, one-inch margins, numbered equations, compact tables and running headers. It is self-contained and does not modify the original analysis note.

## Interpretation

Changing tails strictly outside a fit region leaves a proportional Gaussian core; with a frozen background, that alone rescales the full-template upper limit without changing the likelihood-ratio probability. In the actual binning, selected bins can straddle the continuous primary boundary, so their integrated contributions can introduce small probability changes. The separate guard calculation explicitly includes additional tail-sensitive bins.

Broader tails can also enter GP training bins. The deterministic leakage diagnostic tests a specified training/extraction response. It is not an ensemble coverage test, a measured detector response, or an estimate of a discovery probability.

Its anchors are 51/92 MeV (2015), 90/92 MeV (2016), and 78/92 MeV (2021). The injected yield is the Gaussian primary observed A90 at each anchor. Each mass/window has a smooth GP source derived from its observed exterior. The clean and signal-contaminated training branches fit the same deterministic fitted-bin expectation, using both matched and Gaussian extraction templates; the reported recovery is fitted yield divided by injected yield. No random draws are made.

Clean matched recovery equals one by construction because the fitted-bin expectation is centered on its own reconstructed background; this checks a numerical identity, not independent background or injection closure. Primary and guard references are constructed separately, so their comparison changes both the window and the reference continuum.

In numerical ledgers, `A90` denotes the solver coordinate `epsilon² / 1e-8`; `full_yield_90` gives the event yield denoted A90 in the report. See `derived/COLUMN_GUIDE.md` for column definitions and conversions.

All observed limits and probabilities remain conditional on the frozen GP states, support, window, signal shape and asymptotic approximation. The 2015 91–100 MeV extension retains the archived 90 MeV kernel endpoint prescription. The documented independent-fit differences for 2016 remain unresolved by this study. No combined result, new expected bands, coherent scan calibration, global p-value or scenario-selection correction is claimed.

## Document rebuild

After the supplied scripts have produced the figure and table dependencies, build from the source directory with Tectonic:

```sh
cd source
tectonic -C report.tex
```

The cached-resource flag `-C` assumes the usual TeX resources are installed; omit it for an initial resource download. The source uses only relative figure and table paths. Reproducing the computations requires the bundled scripts and scientific Python dependencies; consult their command-line help and the saved protocol for the exact numerical configuration. Final QA must include rendered-page inspection and text checks as well as numerical validation.

## Reproduce all calculations

From the study directory, run:

```sh
python3 -m pip install -r requirements.txt
PYTHON_BIN=python3 bash scripts/run_all.sh
```

Tectonic must also be installed with its TeX resource cache available (or remove `-C` from `scripts/build_report.sh` for the first online build). The run uses three dataset workers with one BLAS thread each. Per-mass checkpoints include hashes of the input, protocol and numerical implementation; incompatible checkpoints are recomputed. The final PDF is `pdf/HPS_GPR_v5p9_Signal_Tail_Study.pdf`.

The execution environment used here is recorded in `provenance/runtime.json`. The default fit path reads the bundled NPZ spectra; the original ROOT paths in the provenance are historical identities and are not required to reproduce this study. The inherited scripts in `provenance/` are records, not rebuild entry points.

The full mass-by-mass observed limits and local probabilities are in `derived/scans.csv`; see `derived/COLUMN_GUIDE.md` for amplitude units and column definitions. `qa/validation.json` contains 102 numerical checks. `scripts/audit_legacy.py` reproduces the six-cell optimizer comparison from bundled source/data without the older checkout. `scripts/check_artifacts.py` regenerates PDF renders and checks text/geometry; its automatic results do not replace the recorded manual page review.
