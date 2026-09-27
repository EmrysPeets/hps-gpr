# HPS GPR v6.3.6: understanding the 2021 10% signal extraction

Read `pdf/HPS_GPR_v6p3p6_2021_Signal_Extraction.pdf`. This 19-page standalone note is a language, figure-label and explanatory revision of the supplied v6.3.5 study. No scientific fit, toy, calibration table or observed result has been changed.

## Rebuild the note only

Run from this study directory with Python providing NumPy, SciPy, pandas and matplotlib, and Tectonic installed. The recorded environment uses the Xcode Python and `/opt/homebrew/bin/tectonic`; `make_report.py` uses cached TeX resources. No remote computing, simulation or fitting is required.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
"$PYTHON" scripts/check_editorial_integrity.py
"$PYTHON" scripts/make_intro_figures.py
"$PYTHON" scripts/make_report.py --build
```

`source/report.tex` can also be compiled directly from `source/` with Tectonic. The plot builder reads frozen result CSVs. `scripts/build_narrative.py` supplies the prose and captions. The new calibrated-response panels divide archived corrected-yield means and their standard errors by the fixed expected injected yield; `figure_data/calibrated_response.csv` gives these plotted values explicitly.

All original scientific scripts and checkpoints remain available. The parent scientific reproduction instructions are in `provenance/v635_README.md`. Rebuilding the editorial note does not require rerunning them. `protocol.json` intentionally retains scientific version 6.3.5 and its original seed; `revision.json` identifies this editorial version.

## What changed

- Reorganized the explanation around the signal-core shift, measured response, and toy-calibrated inference.
- Added the logarithmic TC signal-MC relation next to the unchanged user-supplied HPS Internal image; added all eleven 60–260 MeV signal-MC samples and the common-core comparison.
- Distinguished the GP mean background source (Poisson counts about a fixed GP arithmetic mean) from the functional form background source.
- Labeled background-only means and bias explicitly; replaced opaque terminology in prose and axes.
- Added genuinely calibrated response, `(Ahat - delta)/(R A)`, alongside the fitted/injected ratio and the signal-induced change. The simple yield correction remains a diagnostic, not the primary limit construction.
- Explained independent evaluation experiments, uncertainty bars, background-only exceedance probabilities, and event-yield-to-coupling conversion.
- Preserved finite-grid limitations, sample sizes, full selected normalization and scientific qualifications.

There is no independent generated 65 MeV signal-MC sample in the supplied catalogue. The note identifies the actual 60 MeV sample and explains the low-mass acceptance issue without inventing a 65 MeV measurement. The 260 MeV catalogue panel is shape-only and has no logarithmically shifted curve.

## Provenance and QA

`qa/v636_editorial_integrity.json` verifies 2,545 original scientific files against the parent's saved manifest. Original results and validation records are inherited evidence; the parent report/layout QA and original note are preserved under `provenance/v635_*` and `provenance/HPS_GPR_v6p3p5_Unified_2021_Procedure.pdf`. The current note has separate `qa/v636_report_qa.json` and `qa/v636_portable_qa.json` records. Rendered pages are local review products and are excluded from the release archive.

The supplied HPS image is `provenance/intro_signal_mc/hps_internal_user_original.png`. Copied v6.1 source data, source hashes and flow-bin normalization are in the same directory. The left shift uncertainties are 32-replica signal-MC bin-resampling standard deviations. They are distinct from the 2,000-resample whole-toy bootstrap used for the inference study. The right image's error definition was not supplied.

The common-core panel conditions on two fitted core widths only for the explicit core-shape comparison; its full-distribution panel and injection templates retain tails. The fitted signal-MC core width is distinct from the extraction's reference analysis resolution.

`MANIFEST.sha256` covers the packaged files. `scripts/package.py --destination /absolute/path/to/release` packages the note after checking current QA and original numerical integrity.
