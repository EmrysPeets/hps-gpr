# HPS GPR v6.4.2 — 2016 window-comparison appendix

Appendix A (pages 24–27) compares 2016 MC fit and GP-training exclusion ±2u versus ±3.5u. The main-body analysis and numerical records remain unchanged from v6.4.1. The same MC shape and full selected normalization are used in both windows. The 2021 MC window remains [-4,+3]u; 2015 remains Gaussian.

The narrower interval lowers the median 2016 conditional profile limit by 9.64%, but paired full-MC injections at five masses recover only 87.7–92.2% of added signal, compared with 99.8–100.1% for ±3.5u. After dividing background fitted-yield spread by response, the narrow/wide ratios are 0.998–1.030. This bounded fixed-source diagnostic supports retaining the main window; it does not validate coverage or global significance.

Rebuild just the appendix with `STUDY_PYTHON=/path/to/python3 bash rebuild_appendix.sh`, or add `--fresh` to regenerate its 252 new observed fits and 6000 new toy fits. It reuses 2000 parent null rows with matching seeds. Full `rebuild.sh` includes the appendix too. The new data live under `results/window_comparison`; its protocol and independent validation are explicitly named in `provenance` and `qa`.

Standalone report: `pdf/report.pdf`. Its LaTeX layout and explanatory captions follow v6.3.n. This package extends the original v6.4 shape study; prior reports remain separate.

The 2016 full selected empirical signal template uses neighboring MC histograms, with linearly interpolated core center and width. The same core-centered interval, −3.5≤u≤+3.5, defines fit bins and GP-training exclusion. The 2021 template retains −4≤u≤+3; 2015 retains its Gaussian. Here u=(reconstructed mass−fitted core center)/fitted core width. Only the already FEE-scaled/smeared histogram `h_MinvScSm_GeneralLargeBins_Final_1` determines the 2016 signal model.

The combined scan compares the established all-Gaussian reference (including its already shifted 2021 Gaussian), 2021 MC only, and 2016+2021 MC. All three use identical campaign availability: 2015 through 100 MeV, 2016 through 175 MeV, and 2021 through 240 MeV, on the 60–240 MeV grid. The separate 2016 scan covers 40–175 MeV. Every scan step is 1 MeV.

The 2016 selected fit regions are 91 and 69 MeV, chosen from positive local maxima with disjoint fitted bins. The combined minimum under both MC shapes is at 68 MeV, local asymptotic p=0.002475. At 2016 mass 91 MeV, local asymptotic p=0.000196 differs materially from the fixed-source null check: 3/1000 exceedances, add-one rank 4/1001, 95% Clopper–Pearson interval [0.000619,0.008742]. The null root mean is +0.656. No scan-wide significance or independent coverage validation is claimed.

Continuous 90% profile-CLs curves are conditional model results. A separate selected-point rank inversion compares all extraction methods on the same full MC injection, including training-bin and out-of-support signal. It uses 200 calibration toys per strength on a finite grid. Endpoints are discrete accepted strengths; no interpolated continuous limit or coverage guarantee is inferred. All expected and realized yields, signed fits, rank counts, accepted sets and censoring flags are retained.

## Rebuild

Python 3.9 with `requirements.txt`; Tectonic 0.15.0 with the LaTeX article/LatinModern bundle already cached. The build intentionally uses `tectonic --only-cached` and does not download packages. Set `STUDY_PYTHON` if the default python3 lacks scientific dependencies.

```bash
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild.sh
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild.sh --fresh
```

The first command reuses verified numerical checkpoints. The second deletes only generated numerical checkpoint directories in this package and regenerates 992 observed fits, 5000 background-toy fits and 21600 common-signal calibration fits. Both retain 362 unchanged 2021 individual rows from the pinned v6.3.8 ledger and independently check representative fits. Two process workers and one numerical thread each are enforced. Kernel parameters and conversion inputs are held fixed.

The original 2016 core-fit and resampling ledgers are frozen inputs here. Their original code and raw ROOT files are supplied under `inputs/v64/`; this rebuild verifies their hashes and reproduces the new interpolation comparison and all report figures. It does not silently refit or change the earlier center definitions.

## Records

- `provenance/input_manifest.sha256`: 168 frozen inputs, relative paths and SHA-256.
- `provenance/protocol.json`: observed scan, template/window policies, source-script hashes.
- `provenance/local_check_protocol.json`, `rank_limit_protocol.json`: deterministic toy seeds, paired draws, full-category rules and rank ordering.
- `results/observed_scan.csv`: 1354 observed rows, including source/reuse labels.
- `results/template_geometry.csv`: actual bin edges, core parameters, probability partitions and conversion factors.
- `results/selected_fits/`: observations, GP predictions/covariances, fitted components and traces for all selected methods.
- `results/local_checks.csv`, `local_toy_rows.csv`: background tails and all signed fits.
- `results/rank_limits.csv`, `rank_grid.csv`, `rank_calibration_rows.csv`: finite-grid results and complete toy rows.
- `qa/`: independent likelihood/toy validation, portable numerical rebuild and rendered-PDF review.
- `source/`: standalone LaTeX plus source sections. `figures/`: vector PDF and PNG figures.
- `MANIFEST.sha256`: package files excluding this manifest. Verify from the package root with `shasum -a 256 -c MANIFEST.sha256`.

The coupling conversion is inherited from the earlier analyses; both its unmultiplied ee coordinate and historical visible-decay display are saved. MC statistics, center/width definitions, conversion inputs and GP kernel uncertainty are not profiled as new systematic uncertainties in these curves.

For the appendix toy ledger, read CSV with `keep_default_na=False` to preserve the literal study label `null`; use `float_precision="round_trip"` for exact numerical replays. Parent PDF-review and portable-rebuild records are retained under `qa/parent_v641`; current appendix checks are `qa/window_*validation.json`.
