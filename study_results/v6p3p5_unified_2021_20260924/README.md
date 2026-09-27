# v6.3.5 unified 2021 study

The standalone report is `pdf/HPS_GPR_v6p3p5_Unified_2021_Procedure.pdf`. It connects independent null calibration, native-MC generation with Gaussian extraction, finite-grid full-yield inference, actual observed scans, the unchanged-old-campaign combination, and a separate window sensitivity study.

The core shift improves response but does not give unit full-MC-yield response. Conditional rank calibration and asymptotic Gaussian references are distinct. Empty and zero-only grid inversions are unresolved continuous limits, not zero physical exclusions. Neither global significance nor physical acceptance/exclusion is certified.

## Reproduce from the study directory

The scientific runtime is the verified Xcode Python with NumPy/SciPy/pandas/matplotlib. Commands use existing checkpoints; do not delete them. Run the stages sequentially, with at most two workers per runner and four across all concurrently running studies. Numerical libraries must use one thread per worker.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
"$PYTHON" scripts/launch.py --workers 2
"$PYTHON" scripts/observed.py --stage all --workers 2
"$PYTHON" scripts/window_study.py --toys
"$PYTHON" scripts/window_study.py --summarize
"$PYTHON" scripts/analyze.py
"$PYTHON" scripts/validate.py
"$PYTHON" scripts/validate.py --statistics-only
"$PYTHON" scripts/validate_joint.py
"$PYTHON" scripts/make_report.py --build
```

`launch.py` has a 1,800-second wall-clock watchdog and preserves completed atomic checkpoints. The report builder invokes `tectonic --only-cached`, creates standard matplotlib PDF/PNG figures and `source/report.tex`, and performs no fitting or ensemble generation. Inspect the rendered final PDF before distribution. To regenerate the validated release, run `"$PYTHON" scripts/package.py --destination /absolute/path/to/release`. It verifies the QA records, writes a SHA-256 manifest, tests the archive, and copies the PDF. The included `qa/report_qa.json` records inspection of all fourteen rendered pages; `qa/portable_qa.json` records reproduction in an independent directory.

## Data and uncertainty

- Main: 2,000 pilot, 52,000 calibration, 22,000 evaluation free fits; all 22,000 truth profiles and 16,000 native CLs endpoints valid. Main cohort counts are 100 pilot plus 100 calibration and 100 evaluation per source, with shared backgrounds across mass/strength/policy. Fit rows are not independent experiments.
- `calibration_summary.csv` retains mean null pull, yield offset, centered-yield null width and paired MC response separately, including joint bootstrap covariance of offset and response.
- `heldout_summary.csv` contains moments, uncertainty and likelihood containment; MC-only affine point-estimator diagnostics include approximate propagated calibration uncertainty. They do not modify the primary limits. Gaussian controls are not corrected with an MC response.
- `pointwise_rows.csv`, `limit_rows.csv`, `limit_summary.csv` retain every attempted ID, complete accepted nodes, empty sets, internal holes and right censoring. Truth acceptance is the primary rank-size diagnostic. Envelope coverage at zero includes the stored zero endpoint even for empty sets, so it is not accepted-set coverage.
- `observed_pointwise.csv` uses raw fitted-yield upper-tail local-p0 ordering. `combined_observed_rank.csv` uses signed-profile-root local-p0 ordering. Both use raw-yield lower-tail ordering for exclusion. Their finite-MC pointwise probabilities are not global scan probabilities.
- `observed_display.csv` and `combined_observed_rank_display.csv` retain raw electron-channel proxy values and separate legacy visible-branch display columns. The branch factor is applied once, above the muon-pair threshold.
- The bootstrap uses 2,000 whole-toy resamples. Pull-centering uncertainty bands condition on the frozen calibration mean; its separate calibration uncertainty is retained. Every binomial interval is an exact two-sided 95% Clopper–Pearson interval.

The primary logarithmic-law/calibration domain is 60–240 MeV. The 260 MeV geometry entry uses its directly fitted native core and is shape-only; no 40 MeV point is supplied. Templates retain full-selected normalization including overflow and outside-support probability. The main matched fit/training exclusion is ±2.25 nominal resolution; no wider guard is adopted. The window side study is exploratory and separate.

`protocol.json`, `pilot_reference.json`, `calibration_freeze.json`, the joint freeze files, `provenance/`, and raw checkpoint hashes establish provenance. Pinned earlier matched-MC-template studies do not constitute prior MC-generated/Gaussian-fitted closure. Historical 2016 source/support/optimizer qualifications remain unresolved by these conditional fixed-kernel toys.
