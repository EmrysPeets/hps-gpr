# HPS-GPR v5.5.2: 2015 high-mass response study

This study-only extension of v5.5.1 includes 2015 mass hypotheses from 100 through 130 MeV at 0.5 MeV spacing. It tests a fixed positive 120 MeV component and compares all three dataset pairs and all three campaigns. Previous study packages and released results remain unchanged.

At 120 MeV, using the full saved 14–150 MeV training support, the 2015 fit has 267.7 ± 163.7 signal events inside the moving window and signed local response r = 1.638. These are a fitted component and curvature uncertainty, not calibrated signal significance. The 2016 and 2021 roots are 2.813 and 0.674. Transferring the frozen beta = −2.90358457 changes the rate penalty from 8.992 to 2.055; refitting beta = −3.543 gives 1.027.

The old 135 MeV endpoint leaves only three upper-sideband bins at 120 MeV and gives r = 3.103; an endpoint at 145 MeV gives 1.816. The primary endpoint was chosen from available spectrum support, not fit height. Its strongest positive interior scan node is 126 MeV (r = 3.803), which is recorded as a retrospectively selected hypothesis. The best three-campaign slope there is −5.638 and has a weakly constrained negative tail. The inherited 117 and 121 MeV hypotheses are also tabulated.

## Definition and limitations

- Native 2015 counts in 0.05 MeV bins are summed by five. The old 14–135 MeV counts are exactly reproduced; archived floating-point coordinates are preserved on the overlap. No interpolation of event counts occurs.
- The main support is 14–150 MeV; 135 and 145 MeV endpoints are comparisons when at least three external bins remain on each side. All fits exclude a moving ±2.25 sigma window from GP training. The 2015 kernel and effective radiative fraction are held at the inherited 90 MeV state; linear resolution is extrapolated. These assumptions have no new high-mass calibration.
- The added energy power acts after the inherited density/radiative conversion. Its amplitude is nonnegative; standalone signed amplitudes are retained as diagnostics. Beam energy is confounded with campaign changes. No theoretical accepted-yield calculation is performed.
- Every penalty compares exactly the same local data and constraints against independent nonnegative amplitudes. Different mass/support windows and overlapping subsets are not pooled. Two positive rates exactly determine a pair amplitude and slope; their zero penalty is automatic.
- Beta profiles span −12 to 12. Connected unit-deviance ranges describe profile shape and are not calibrated confidence intervals. Pair predictions do not propagate parameter uncertainty. No new toys, limits, global significance or approval of an enlarged search is implied.

## Contents and reproduction

- `pdf/HPS_GPR_v5p5p2_2015_Highmass_Response_Study.pdf`: five-page standalone note with primary literature citations.
- `figures/`: three vector PDF figures and PNG previews.
- `derived/standalone_scan.csv`, `support_geometry.csv`, `fixed_response_scan.csv`: full scan and support records.
- `derived/response_fits.csv`, `beta_profiles.csv`, `pair_heldout_120.csv`: all pairs and all-three results at the four comparison masses, with two extra support fits at 120 MeV.
- `derived/extraction_120.csv`, `2015_rebinned_spectrum.csv`: figure inputs and exact count rebinning.
- `protocol.json`, `provenance/inputs.json`: study choices and SHA-256 source identity. `inputs/scopes.json` is an unchanged historical record; the effective extension is declared in `protocol.json` and implemented explicitly in `scripts/run_study.py`.
- `engine/`, `inputs/`: copied solver and data snapshots; no dependency on the original parent path at execution time.
- `references_physics.json`: unchanged v5.5.0 physics reference record. Its derived mass-scaling constants remain historical; use `protocol.json` and the result ledgers for this rate-response extension.
- `qa/`: numerical, semantic, rendered-page and portable-build checks. `MANIFEST.sha256` covers the delivered study files.

Use Python 3 with NumPy, SciPy, pandas, Matplotlib, PyMuPDF and Pillow, plus Tectonic:

```bash
bash scripts/build.sh
```

For just the note, compile `source/main.tex` with Tectonic. After visually reviewing all rendered pages, `python3 scripts/package.py` verifies an isolated TeX rebuild and prepares the PDF and ZIP. The completed run contains 61 primary nodes, 283 standalone fits and 18 slope comparisons; one worker and one BLAS thread are used. It replays the inherited 100 MeV roots within 5.1e-6, verifies exact overlapping counts and positive displayed models, and runs no toys.
