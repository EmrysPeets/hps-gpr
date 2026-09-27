# HPS-GPR v6.3.1: 2021 10% fixed-yield injection study

This package runs the final reviewed handoff in `provenance/HANDOFF.md`. It uses 100 independent pilot background spectra and a separate 100 evaluation background spectra, Gaussian and native-MC generation/extraction templates, and pole masses 60–240 MeV in 20 MeV steps. Each mass and level uses one fixed expected full-selected signal yield for both shapes.

The numerical run is complete: **2,000/2,000 valid pilot free fits and 8,000/8,000 valid evaluation free fits and truth-fixed profiles**. Every planned toy ID is retained in all 20 pilot and 80 evaluation cells. The failure ledger has no failed rows. The saved pilot scale was frozen before evaluation. Numerical, replay and provenance validation is separate from scientific closure: a biased fit or non-nominal containment is an outcome, not a reason to discard a toy.

## Read first

- `pdf/report.pdf`: standalone report, including definitions, finite-sample uncertainty, comparison plots, all six representative spectra and limitations.
- `results/evaluation_summary.csv`: all 80 mass/shape/level summaries, including raw and paired response, bias, pull mean/width, error response and nominal containment.
- `results/paired_shape_comparisons.csv`: paired MC-minus-Gaussian differences with complete-pair accounting and pointwise whole-toy bootstrap intervals.
- `results/pilot_summary.csv` and `pilot_reference.json`: independent pilot diagnostics and the immutable common Gaussian reference scale.
- `results/summary.json`: the same summary data, source hashes, counting/completeness information and uncertainty conventions.
- `qa/independent_validation.json`, `qa/smoke.json`, `qa/resume.json`: validation and deterministic-resume evidence. Rendered-page QA accompanies the final delivery.

At input level `z=5`, Gaussian mean raw recovery spans 0.941–1.097 and mean paired response spans 0.961–0.978 across the ten masses. Native MC spans 0.779–1.042 and 0.716–0.894, respectively. Native-MC paired response is lower at every mass; pointwise 95% paired bootstrap intervals quantify the differences. These are ranges across correlated cells, not pooled estimates. At this same level, nominal 95% signed-set containment ranges from 88–98/100 for Gaussian and 88–97/100 for MC; exact cell counts and 95% Clopper–Pearson intervals are supplied. Equal expected signal counts do not imply equal extraction difficulty: for example, the MC pilot error is 2.79 times the Gaussian error at 60 MeV.

## Scientific definition

The source background is the pinned nominal 2021 GP arithmetic mean in `inputs/null_2021.npz`. The full-support binning and exposure remain unchanged. The observed counts are used only for input-release identity checks. The source file SHA-256 is `306a915b9ba6230aafbe058c94438c6af11f85b60d68f5aae4ebbfec4d4a9424`.

For mass `m`, `s0(m)` is the mean returned yield uncertainty of **all 100 Gaussian pilot fits**. The expected injection is `A_expected = z*s0(m)`, with `z = 0, 1, 3, 5`. The evaluation backgrounds are independent of the pilot; each evaluation background is reused across masses, shapes and levels. Independent Poisson signal-category counts are added in all analysis bins, including training sidebands, and in the two outside-support categories. The realized total is not forced to equal the floating-point expectation. This differs from v6.2's exact-N multinomial experiment.

Both generation and extraction use full-selected candidate probabilities, including support losses. The common pole-centered window is `m ± 2.25*sigma_m`; native-MC offsets/tails are retained. There is no window/support renormalization. The amplitude is a signed full-template candidate yield, not a fit-window count, generated-resonance count or epsilon-squared parameter.

Every spectrum recomputes count-dependent GP preprocessing and the correlated conditional mean/covariance while holding the archived kernel parameters fixed. This is GP conditioning with archived kernel parameters, without hyperparameter optimization. The truth-fixed nuisance profile uses that same toy's GP constraint and fixes the **expected** signal yield. No additional random GP nuisance is drawn in generation.

The primary diagnostics are:

- Bias: `Ahat - A_expected`.
- Pull: `(Ahat - A_expected)/sigma_postfit`, with the returned observed profile-Hessian yield error.
- Raw recovery: `Ahat/A_expected` for positive injections.
- Paired response: `(Ahat_z - Ahat_0)/A_expected`, using the same background, mass and template.
- Error response: `sigma_postfit/s0`, distinct from the empirical fitted-yield sample SD.
- Nominal signed profile-set containment: `q_true <= 1` and `q_true <= 3.841459`, corresponding to the nominal 68.27% and 95% sets. These are not a physical nonnegative-signal upper-limit construction.

The recovery/response summaries are averages of per-toy ratios. Null subtraction is used only in paired response. Pulls and truth profiles use the expected yield, not the realized Poisson signal count. Neither `z` nor a pull is a calibrated discovery significance.

## Reproduce or resume

Known working scientific Python: `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`, with NumPy, SciPy, pandas, Matplotlib and pypdf. PDF builds use `/opt/homebrew/bin/tectonic`; the report builder uses cached TeX resources. Rendering uses `pdftoppm`. On another system, use an equivalent Python environment and adjust the Tectonic executable in `scripts/make_report.py`.

From this package directory:

```bash
STUDY_PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
"$STUDY_PYTHON" scripts/launch.py --workers 4
"$STUDY_PYTHON" scripts/validate.py
"$STUDY_PYTHON" scripts/summarize.py
"$STUDY_PYTHON" scripts/make_report.py --build
```

`launch.py` applies a real 1,800-second wall-clock watchdog. The runner permits at most four local workers with one numerical-library thread per worker. It uses a lock to prevent duplicate launches, saves the cohorts before fitting, checks dependency/configuration signatures, retains atomic completed checkpoints and verifies the frozen reference before evaluation. Re-running the launcher resumes validated existing checkpoints with the same counts and settings. `--stage smoke` and `--stage pilot` select bounded early stages. Do not bypass the watchdog or edit frozen numerical inputs/settings and reuse old checkpoints.

For report-only regeneration, run the last two commands. They do not generate additional scientific toys or refit the GP. The summarizer requires exactly the planned 100 IDs in each cell and checks amplitude consistency, reference-scale identity, pairing, separate cohorts, realized-count identities and containment decisions before writing summaries. The report builder verifies the summary's source hashes before generating vector PDFs, PNGs and LaTeX.

The bootstrap uses 2,000 replicates with master seed `63220260924` and `SeedSequence([master, 4, 0, replicate, 0, 0])`. Each replicate resamples the **same 100 evaluation toy IDs in every cell**, preserving all masses, levels, shapes and null partners. The frozen pilot table stays fixed. Width uncertainties and MC–Gaussian comparisons use this block bootstrap; means also retain standard errors from sample scatter. The output intervals are pointwise, with no multiple-comparison adjustment.

## Files and accounting

| Location | Content |
|---|---|
| `protocol.json` | Frozen numerical specification, normalization, seeds, masks, fit gates/retries and limitations |
| `provenance/HANDOFF.md` | Final instruction document copied before the study |
| `provenance/input_hashes.json` | Pinned source/dependency identities |
| `inputs/cohorts.npz` | The saved 100 pilot and 100 evaluation background spectra |
| `inputs/templates.npz` | Full signal-category probabilities and masks |
| `pilot_reference.json` | Frozen pilot reference and shared expected yields |
| `results/pilot_rows.csv` | All 2,000 pilot free-fit rows and numerical diagnostics |
| `results/evaluation_rows.csv` | All 8,000 evaluation rows, signal counts, hashes and fit/profile diagnostics |
| `results/pilot/`, `results/evaluation/` | Atomic chunk CSV/JSON checkpoints; evaluation NPZ files retain signal draws |
| `results/failure_ledger.csv` | Explicit unresolved free-fit/profile rows (header only for this successful run) |
| `results/representative/` | Fixed toy 0 at `z=5`, masses 60/140/240 MeV, both shapes; source data for spectrum panels |
| `figures/` | Vector PDF and PNG comparison plots and six spectrum displays |
| `source/report.tex` | Reproducible standalone report source |
| `pdf/report.pdf` | Final PDF report |
| `qa/` | Smoke, independent validation, resume and rendered-page QA evidence |
| `MANIFEST.sha256` | Final package identity, generated after validation/QA |

`fit_valid` and `profile_valid` are reported separately. Failed fits would remain in the original row tables, with bounded attempts and reasons; no replacement toys are drawn. Containment reports `k/profile_valid`, `profile_valid/100`, 95% Clopper–Pearson intervals and the all-attempt accounting bounds `[k/100, (k+unresolved)/100]`. Those bounds are not confidence intervals. Complete-pair counts and missing IDs are retained for every shape comparison. The 4 levels and 10 masses are correlated uses of 100 evaluation backgrounds and are never counted as extra independent toys.

## Interpretation limits

The result is conditional on one pinned background mean, fixed empirical MC templates and this specified archived-kernel GP procedure. MC and Gaussian differences can reflect core offset, width, tails and window acceptance together; this run does not attribute them all to GP sideband absorption. The supplied MC selection equivalence and signal-daughter association remain unvalidated. Finite-MC, detector-response and source-estimation uncertainties are not propagated. The study does not establish physical-background adequacy, unconditional coverage, production hyperparameter-optimization performance, calibrated discovery significance, global significance or an exclusion.
