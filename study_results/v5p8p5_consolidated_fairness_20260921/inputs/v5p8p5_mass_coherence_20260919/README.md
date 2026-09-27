# HPS-GPR v5.8.5 — common mass and mass coherence

This is an isolated continuation of the pinned v5.8.4 study. Formatting follows the v5.0.5 analysis-note conventions: 11pt Latin Modern, one-inch margins, title and abstract, numbered sections, conventional figure/table captions and a review-draft header. Read `source/report.pdf` (10 pages). The delivered extended PDF appends this report to the unchanged 11-page v5.8.4 report. All 1,999 files in the parent study and delivered parent package were hash-verified unchanged.

## Main results

The exact likelihood statistic is `q_R = sum_d max(r_d,0)^2`, with independently profiled nonnegative amplitudes at a common mass. The primary scan ordering is `max_m(-log p_R(m))`, where the local tail includes the zero atom and the actual conditional source offsets/scales. Full search: integer masses 19–250 MeV. Active supports: 2015 19–100, 2016 39–180, 2021 (10%) 50–250 MeV.

| Blind half-width | Calibrated peak [MeV] | Local Z (response model) | Global Z (response model) | Direct global k/256 | Direct plus-one p |
|---|---:|---:|---:|---:|---:|
| 2.25 | 92 | 3.583 | 2.060 | 7/256 | 0.03113 |
| 2.40 | 91 | 3.518 | 1.960 | 8/256 | 0.03502 |
| 2.50 | 92 | 3.551 | 2.017 | 6/256 | 0.02724 |
| 2.60 | 92 | 3.610 | 2.104 | 6/256 | 0.02724 |

Raw q_R peaks at 91 MeV for every width; this differs from the calibrated-score ordering. At the nominal selected peak the direct 95% global-p interval is [0.01106, 0.05552]. At 92 MeV, all four direct local tails have zero exceedances: the one-sided 95% upper bound is p < 0.01163. The 256 exact scans cannot measure the much smaller local probabilities of the response approximation.

The requested 80–105 MeV diagnostic interval is post hoc and is clipped to 100 MeV for 2015. Dataset peaks across the four widths are 2015: 92/93/93/93; 2016: 91/91/91/91; 2021: 93/93/80/80 MeV. Delta m / sigma(92) = 0.208 / 0 / 5.309. The 2021 switch is between competing excursions, including the 80 MeV diagnostic boundary; it does not track a resonance moving by 13 MeV.

For a comparable all-dataset diagnostic, the common 80–100 MeV intersection is used. The primary statistic is the mean squared resolution-normalized scatter of the twelve positive-fit peak masses around their inverse-resolution-variance weighted center. T=2.97935; 195/256 exact null scans are at least as coherent (plus-one p=0.76265, 95% interval [0.70471,0.81256]). The response approximation gives p=0.73873. This does not identify unusually tight coherence. Width instability W=9.41019 has lower-tail p=0.98833 (253/256). These p-values are exploratory diagnostics; they are not multiplied and do not correct selection of the region or statistic.

All results condition on the observed-data-derived, pinned GP source and inherited kernel/support policy. Source estimation, source-model choice, method/width selection, and continuous-mass searches are not calibrated. An exact profile statistic does not imply unconditional discovery calibration. The free-amplitude alternative can have zero amplitude in some datasets and does not require a three-dataset signal or a common physical coupling.

## Files

- `protocol.json`: domains, model and interpretation policy.
- `inputs/copy_manifest.json`: source-to-copy SHA-256 ledger. All numerical inputs are local and portable.
- `inputs/parent_fields/`, `inputs/parent_results/`: frozen v5.8.4 response arrays and comparison ledgers.
- `inputs/null_YEAR.npz`: fixed source means, observed spectra, 256 complete Poisson realizations and seeds.
- `scripts/free_amplitude.py`: exact likelihood combination, atom-aware local integration, paired response scans and direct calibration.
- `scripts/stability.py`: signed-score matrix, amplitudes, competing peaks and coherence/stability toys.
- `results/free_curves.csv`: all 928 mass/width rows; local and global p/Z, counts, intervals and upper bounds.
- `results/free_comparison.csv`: all four methods at all four widths, unchanged parent comparisons.
- `results/stability_*.csv`: all signed-score curves, selected peaks, resolution-normalized ranges, and 92 MeV amplitudes.
- `fields/free_w*.npz`, `fields/stability_null.npz`: saved paired direct statistics, correlated-response maxima, and lookup tables.
- `figures/`: eight vector PDF / PNG pairs, including a supplemental four-method width comparison.
- `source/`: LaTeX, generated tables and 10-page extension PDF.
- `qa/`: constrained-fit checks, source replay, numerical validation, portable rebuild evidence, PDF text and final page renders.

The parent report is a pinned PDF input. Rebuilding that older report from its own sources is supported by the unchanged v5.8.4 source bundle in its original release directory; this bundle reproduces the continuation and concatenated report.

## Reproduce

Use a Python with NumPy, SciPy, pandas and Matplotlib. The original numerical runtime is `/opt/homebrew/bin/python3` (Python 3.13.5, NumPy 2.3.2, SciPy 1.16.1). Tectonic compiles the PDF. From this study directory:

```sh
python3 scripts/free_amplitude.py
python3 scripts/stability.py
python3 scripts/make_figures.py
python3 scripts/make_tables.py
python3 scripts/validate_outputs.py
tectonic -X compile source/report.tex --keep-logs
```

On the originating machine substitute `/opt/homebrew/bin/python3` in these numerical commands. There are no ROOT or parent-checkout dependencies. The copied likelihood engine is byte-identical to v5.8.4. The continuation reuses the 256 exact stored profile scans and independently replays 20 joint constrained fits plus 12 observed/paired-toy fits at 92 MeV; it does not claim to have regenerated all historical fits anew.

The free-amplitude response ensemble uses 100,000 draws, seed 58520260919, fixed 1,024-row batches, and joint mass/width covariance factors from D. The separate stability response ensemble uses 100,000 draws and SeedSequence([58520260919, year, 991]), fixed 2,048-row batches. These are distinct supplemental ensembles; within each, masses and widths share a fluctuation and datasets are independent. Direct toy IDs 0–255 match at all masses and widths. The original source seeds are stored in the NPZs and replayed exactly.

For an isolated numerical, figure and PDF rebuild, the orchestrating Python also needs pypdf and Pillow, and `pdftoppm` must be on PATH:

```sh
STUDY_SCIENCE_PYTHON=/opt/homebrew/bin/python3 /usr/bin/python3 scripts/check_rebuild.py
```

This check compares every result CSV byte-for-byte and all 10 PDF pages by text and rendered pixels. `qa/portable_rebuild.json` records the pass. Generic installations may use the same Python for both roles. Cross-runtime floating-point differences are possible; the stated tolerances in numerical checks remain authoritative.

`python3 scripts/package_release.py --output-dir /path/to/output/pdf/v5p8p5_mass_coherence_20260919` assembles the extended PDF, standalone extension, source/data ZIP and plots ZIP. It requires pypdf. SHA256SUMS manifests cover the study bundle and final delivery. Numerical files are deterministic in the recorded environment; PDF/ZIP container metadata need not have identical timestamps.
