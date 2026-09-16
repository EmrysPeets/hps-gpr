# HPS GPR v5.2.1: fine scans and multiple components

The standalone 43-page report continues the frozen v5.2.0 study. It covers every requested region, extends 2015 hypotheses to 100 MeV, computes actual 0.5 MeV bounded 90% CLs limits, and compares common spacings with campaign mass shifts. The injection pilot uses 10 experiments per scheme and strength: 100 labels, 90 distinct joint 2016+2021 count spectra because the zero-strength schemes share a baseline.

Results are conditional and exploratory. Post-fit morphology, local asymptotic probabilities, selected pair comparisons, and physical common-rate hypotheses remain separate. No global significance or coverage calibration is claimed.

## Read or rebuild

Open `report.pdf`. Its 43 pages include all 25 regional extraction cases, all ten toy curves at every strength in each of the three injection searches, and the shared-spacing comparisons.

Rebuild the supplied LaTeX/figures without numerical reruns:

```bash
bash scripts/build.sh
```

The build uses Tectonic and its cached dependencies. On a new Tectonic installation, populate its package cache with a normal `tectonic report.tex` run first. No analysis inputs need downloading.

## Numerical reproduction

Use Python with NumPy, SciPy, pandas, Matplotlib, scikit-learn, PyMuPDF, and Pillow. Exact runtime versions are recorded in `qa/runtime.json`. To regenerate the same seeded experiment in a new directory:

```bash
python scripts/reproduce.py /absolute/path/to/new_v5p2p1_reproduction --workers 2
```

The destination must not exist. This leaves the frozen package unchanged. It reuses the pinned spectra and repeats the same ten seeded trials per condition; it does not recover a new calibrated toy ensemble. The optional native-parent solver regression runs when the original sibling v5.2.0 directory is available; the supplied portable parent core is also checked independently in `validate.py`.

The injection scanner validates source/input/output checksums before reusing completed curves. A change to the likelihood, truth, conversion, or saved curves requires a new study directory; it is not silently mixed into existing results. Counts are generated as full-spectrum Poisson background plus exact Poisson signal increments, then reused across moving masks and overlapping scopes.

## Data map

- `inputs/spectrum_*.npz`: full retained analysis histograms, resolution, full-yield conversion, archived kernels, native density histograms, and stress means.
- `inputs/manifest.json`: original extraction/reference provenance. `inputs/parent/` pins v5.2.0 catalogue, forward means, and PDF.
- `derived/observed_limits.csv`: 1,433 actual half-MeV limits and fixed-mass p0 values. `kernel_controls.csv` and `kernel_control_ratios.csv`: 754 controls.
- `derived/regional_screen.csv`: 2,968 screening rows. `regional_pairs.csv`: 600 broad-single comparisons at 112 pairs; `regional_catalogue.csv` supplies 26 regional views; `regional_scope_leaders.csv` and `regional_union.csv` preserve scope/union definitions. `regional_union_controls.csv` investigates the 66/93 MeV leader.
- `derived/regional_extractions.csv`:index of 43 campaign NPZs spanning 25 extraction cases. Every case is plotted in `figures/extract_*.pdf` and shown in the report.
- `derived/injection_truth_*.npz`, `injection_amplitudes.csv`: fitted means, benchmark floors, positive reference yields and deterministic control spectra.
- `derived/toys/experiment_ledger.csv`: all 100 scheme/strength/trial labels and 90 unique joint hashes. `counts_trial_*.npz` retains Poisson backgrounds, signal increments and count arrays. `limits_*.csv` contains 48,300 labeled rows (43,470 distinct calculations plus the copied zero-strength curves).
- `derived/deterministic/`:actual limit scans of unfluctuated counts, including the original narrow 90/117 MeV reconstruction. These are not median toy curves.
- `derived/shape_metrics.csv` and `toy_shape_summary.csv`:full-window and fixed-subwindow comparisons, paired ten-trial medians/ranges, and descriptive trough counts. No sampling p-values are assigned.
- `derived/shift_*.csv/json/npz`: 36 mass/rate fits, 78 campaign estimates/extractions, fixed experiment definitions, bounds, optimizer checks and 30 coarse separation-profile points.
- `figures/`:standalone vector PDFs and PNG previews. `qa/`:numerical audits, independent physics/statistics reviews, rendered contact sheets, source/cache signatures, and portable rebuild checks.

The manuscript defines every signal strength, mass domain, kernel policy, selection rule and interpretation boundary. The old GP local probabilities are supplied as prior reference fields; no half-MeV GP calibration or reused trials factor is invented.
