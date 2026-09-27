# HPS GPR v6.3.6: understanding the 2021 10% signal extraction

Read `pdf/HPS_GPR_v6p3p6_2021_Signal_Extraction.pdf`. This 43-page standalone note contains the readability revision of v6.3.5, eight pages of v16 TC/UC signal-MC studies, and the eleven-page v6.3.7 template/window study beginning on page 28. Appendix I adds five pages on blind-window size, signal leakage and controlled fit impact, beginning on page 39. The main inference fits, calibration tables and observed results remain unchanged. Appendices D-H add independent tuning, calibration and evaluation of full v16 TC signal-MC injections.

## Rebuild the note only

Run from this study directory with Python providing NumPy, SciPy, pandas and matplotlib, and Tectonic installed. The recorded environment uses the Xcode Python and `/opt/homebrew/bin/tectonic`; `make_report.py` uses cached TeX resources. No remote computing, simulation or fitting is required.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
"$PYTHON" scripts/check_editorial_integrity.py
"$PYTHON" scripts/make_intro_figures.py
"$PYTHON" appendix/scripts/analyze_shifts.py
"$PYTHON" appendix_v637/scripts/make_figures.py
"$PYTHON" appendix_leakage/compare_gaussian_centers.py
"$PYTHON" appendix_leakage/make_note_figures.py
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

`qa/v636_editorial_integrity.json` verifies 2,545 original scientific files against the parent's saved manifest. Original results and validation records are inherited evidence; the parent report/layout QA and original note are preserved under `provenance/v635_*` and `provenance/HPS_GPR_v6p3p5_Unified_2021_Procedure.pdf`. The current release is checked by `qa/leakage_appendix_qa.json`; `qa/v637_appendix_qa.json` and `qa/v637_integration_checks.json` retain the preceding 38-page release checks; `appendix_v637/qa/` contains the new numerical, statistical, rendered-page and relocated-rebuild evidence. Earlier `v636_*` and `v16_appendix_qa.json` records are historical validation of preceding releases and retain their original PDF hashes. Rendered pages are local review products and are excluded from the release archive.

The supplied HPS image is `provenance/intro_signal_mc/hps_internal_user_original.png`. Copied v6.1 source data, source hashes and flow-bin normalization are in the same directory. The left shift uncertainties are 32-replica signal-MC bin-resampling standard deviations. They are distinct from the 2,000-resample whole-toy bootstrap used for the inference study. The right image's error definition was not supplied.

The common-core panel conditions on two fitted core widths only for the explicit core-shape comparison; its full-distribution panel and injection templates retain tails. The fitted signal-MC core width is distinct from the extraction's reference analysis resolution.

`MANIFEST.sha256` covers the packaged files. `scripts/package.py --destination /absolute/path/to/release` packages the note after checking current QA and original numerical integrity.

## v16 TC/UC appendices (24 September 2026)

Appendix A records the verified sources, branch-level audit and shared method. Appendix B studies TC; Appendix C studies UC. All 60–260 MeV samples are shown; the shift-law and common-core domain is 60–240 MeV. Data are under `appendix/inputs/`, results under `appendix/results/`, and six additional figures under `appendix/figures/`. No remote access is needed to rebuild these results from the extracted histograms.

TC again favors a logarithmic shift, with omitted-mass RMS prediction error 0.0652 MeV. UC favors a quadratic, with RMS 0.0422 MeV. These are descriptive model-selection diagnostics, not independent validation. All 22 baseline fits, 66 span checks and 704 Poisson-bin resampling fits are valid. Core uncertainties use 32 replicas per mass. TC and UC selections and stored smearing information differ; their comparison does not isolate the vertex constraint. No new upper limits or observed-data fits are computed.

The read-only remote extractor is `appendix/scripts/extract_v16_remote.py`, driven by `appendix/scripts/fetch_v16.py` through the existing authenticated SSH socket. It is capped at 600 seconds and uses one decompression thread. Do not rerun it merely to build the note. Remote file identity includes paths, size/mtime and SHA-256 hashes of streamed branch payloads; whole ROOT-file hashes were not taken.

The previous approved 19-page release is preserved in the output folder under `history/before_v16_appendix/`; its source/layout QA is also preserved under `provenance/v636_before_v16_appendix/`.

## v6.3.7 signal-template and window study (25 September 2026)

Appendices D-H contain the study requested after the v16 shift comparison. The complete standalone package is embedded in `appendix_v637/`; consult its README to regenerate all numerical results or only its report. Seven new figures use How to read the figure captions and explicit uncertainty conventions. The numerical result is improved signal-yield response with modest average precision gains, not universally smaller upper limits. The tested starter exclusion leaves 16-20% of the selected signal in GP training; that leakage remains present in every full-MC injection.

The preceding 27-page release is preserved in `output/pdf/v6p3p6_readability_2021_20260924/history/before_v637_appendix/` in the project; its editorial sources and QA are in `provenance/v636_before_v637/`. Prior main numerical files are verified unchanged.

## Leakage follow-up appended (25 September 2026)

Appendix I, pages 39-43 and Figures 28-32, embeds the nine-mass window/leakage comparison and the paired contamination controls for both background sources. `appendix_leakage/` includes all original diagnostic rows and standalone plots, the adapted local reproduction script, and figures laid out for this note. The original follow-up source and manifest are preserved under its `provenance/`; only the parent-study lookup path changes in the embedded numerical script. The previous 38-page release is saved under `history/before_leakage_appendix/` in the output directory. All previous scientific data and main results are retained.

To rebuild only the added figures, run `python3 appendix_leakage/make_note_figures.py`, then `python3 scripts/make_report.py --build`. To repeat the saved-data controls, run `python3 appendix_leakage/make_leakage_plots.py`; this reuses the embedded v6.3.7 inputs without drawing new toys.

The Gaussian leakage reference already used the empirical shifted center. Figure 32 and page 43 add an explicit comparison against a Gaussian centered at the generated mass, with width and training bins fixed. The existing numerical results are unchanged. The preceding 42-page release and its QA are preserved under `history/before_gaussian_center_comparison/` in the output directory and `provenance/v636_before_gaussian_center_comparison/` in this package.
