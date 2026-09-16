# Two positive components and the intervening dip — v5.2.0

Standalone observed-data implementation of Section 5 of `03_Signal_Assessment_Plan`, 10 September 2026. All work is isolated in this new study package. No new toys or random Gaussian fields were generated; no production result was modified.

The final catalogue evaluates 4,414 scope/pair fits across all seven original dataset overlap searches, including every eligible scope at each refined pair. The primary union includes all campaigns that support both masses. Each pair fits the same original analysis bins and common external-GP background under H0, HL, HH and HLH. The dip bins enter once.

The final comparison scans a single positive component throughout the entire fitted region and refines its best mass continuously. It retains the signed, nonnested pair-minus-single improvement. The fixed-pair chi-bar-square reference p-values apply to Q2 versus background only; no global or calibrated two-particle significance is claimed. Existing GP local fields guide the additional regions and quantify the expected peak–dip correlation.

## Deliverables

- `report.pdf`, `report.tex`: standalone report and source.
- `derived/region_single_control.csv`: complete final fit catalogue, fixed-pair reference p-values, amplitudes, correlations, numerical diagnostics, single-mass search bounds and signed comparison.
- `derived/catalogue_final.csv`: five separated leading candidates per scope.
- `derived/union_final.csv`: complete maximal-available-dataset union.
- `derived/*components.npz`: original count arrays, common GP covariance, fitted background and signal components for every detailed case.
- `derived/*amplitude_surface.npz`, `amplitude_profiles.csv`: two-dimensional and one-dimensional amplitude profiles; descriptive, not calibrated confidence regions.
- `derived/fixed_region_mass_surfaces.csv`: local mass fits holding data and background constraint fixed.
- `derived/*expected_full.npz`, `forward_scans.csv`: positive fitted full-support expected spectra and deterministic moving-mask responses. No fluctuations.
- `derived/window_stress_controls.csv`, `illustrative_count_correlations.csv`, `campaign_yields.csv`, `information_loss.csv`: background/window/covariance/rate diagnostics and interpolation-information loss.
- `statistics_hep_review.md`, `implementation_review.md`: requested independent statistics and HEP review, with resolved issues and primary-source citations.
- `inputs/manifest.json`, `SHA256SUMS.txt`: source provenance and final package identity.
- `qa/validation.json`, `visual_review.json`, `portable_build.json`: numerical/semantic checks, rendered-page inspection, and source rebuild.

The intermediate endpoint-only and intervening-single catalogues are retained as development provenance. The **final** ranking is `catalogue_final.csv`, derived from `region_single_control.csv`. Earlier leader values must not be substituted for it.

## Rebuild and reproduce

Use Python 3.9 or later with NumPy, SciPy, pandas, Matplotlib and PyMuPDF; Tectonic compiles the PDF. The local fit environment was the repository `venv/bin/python`; versions are in `qa/environment.json`.

To rebuild the document from bundled fit arrays:

```sh
python scripts/make_figures.py
python scripts/make_report.py
bash scripts/build.sh
python scripts/validate.py
```

To rerun the complete analysis from bundled input arrays in a **separate copy** of the package, first remove only that copy's derived files and run:

```sh
python scripts/scan.py
python scripts/single_control.py
python scripts/continuous_control.py
python scripts/region_control.py
python scripts/details.py
python scripts/information_loss.py
python scripts/make_figures.py
python scripts/make_report.py
bash scripts/build.sh
python scripts/validate.py
```

Fit scripts use one process and one numerical thread. Some stages retain checkpoints; their existence is not authorization to reuse them after changing a policy or solver. The complete run can be reconstructed from the exported analysis bins, native density bins, resolution polynomials and archived kernel states without opening ROOT files. `prepare.py` additionally documents the original attested-checkout/ROOT extraction path; running that export requires the original local source paths recorded in the manifest.

The original external GP assumes negligible signal in its training sidebands; full-support tails are included in the forward diagnostic and their omitted fractions are recorded. The campaign normalization and detector calibration uncertainties are fixed, and the 2016 source/state qualifications remain unresolved. The arbitrary-separation continuum, width alternatives, global trials, independent signal recovery, coverage and calibrated discovery probability are outside this no-toy study.
