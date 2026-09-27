# HPS GPR v6.4.3: global probabilities with selected MC windows

Standalone report: `pdf/report.pdf`. The format follows the v6.3.n LaTeX notes. Numerical conclusions are in `results/global_summary.csv` and `results/global_results.json`.

The 2016 signal fit and GP-training exclusion both use [-2.5,+2.5]u, with u defined by the fitted MC core center and width. The 2021 interval remains [-4,+3]u. Both years retain their full selected neighboring MC templates, without renormalizing to the fit interval. The 2015 Gaussian and its original window remain unchanged. The observations are the archived full 2015 and 2016 spectra and 2021 10% spectrum.

Each of 1,024 simulated experiments contains independent Poisson draws from the fixed archived GP mean of each campaign. A given campaign draw is reused at every mass and in every applicable search. Each mass repeats GP prediction and nuisance profiling with fixed archived kernel parameters. The statistic is the maximum positive signed likelihood root squared, without recentering or rescaling.

The 2016 grid has 136 points from 40 through 175 MeV. The 2021 and combined grids each have 181 points from 60 through 240 MeV. All steps are 1 MeV. The combination uses all three campaigns through 100 MeV, 2016 and 2021 through 175 MeV, and 2021 alone above 175 MeV. A supplementary family statistic also maximizes across the three displayed searches using their correlated toy results.

The null source was fitted to all observed bins with the saved kernel parameters at a 76 MeV reference mass. Its source-estimation uncertainty and earlier window/template selection are outside this conditional calibration. This is a finite-grid global probability, not a continuous-mass calibration or an unconditional physical discovery significance. No upper-limit calculation is updated in this package.

## Rebuild

Use Python 3.9 with `requirements.txt`, and Tectonic 0.15.0 with the article/LatinModern packages already cached. No downloads or remote computation occur during rebuild.

```bash
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild.sh
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild.sh --fresh
```

The default reuses verified mass checkpoints, regenerates observations and all summaries, audits the complete arrays, replays selected toy fits, and rebuilds figures and the PDF. `--fresh` removes only this package's generated per-mass checkpoints and reruns all 1,024 complete experiments. The calculation uses two local workers, one numerical thread each, deterministic seeds and atomic mass checkpoints. Interrupted runs resume with the default command. The checkpoint signature pins the input manifest, seed, sample count, windows and extraction code. A technical timing pilot uses a separate seed namespace and contributes no calibration rows.

The extraction engine used here is frozen from v6.4.1; `scripts/run_global.py` overrides its 2016 fit and exclusion to ±2.5u. Parent protocol and report hashes are retained as provenance. Historical scripts and inputs may describe earlier windows; `provenance/global_protocol.json` is authoritative for this calculation.

## Numerical records

- `provenance/input_manifest.sha256`: hashes of all frozen inputs.
- `provenance/global_protocol.json`: scan definitions and generation/extraction protocol.
- `provenance/global_analysis_protocol.json`: primary and supplementary test definitions.
- `results/observed_scan.csv`: 498 observed scope/mass coordinates.
- `results/template_geometry.csv`: centers, widths, actual bins, probability partitions and anchors.
- `results/global_scans_*.npz`: toy ID × mass × field arrays, with explicit field names and grids.
- `results/checkpoints/m*.npz`: resumable per-mass results, all toy IDs and protocol signature.
- `results/toy_draw_hashes.npz`: SHA-256 of all campaign/experiment count vectors.
- `results/global_summary.csv`: selected peaks, local and global tail counts, probabilities and 95% exact binomial intervals.
- `results/local_global_curves.csv`: corresponding quantities at every mass.
- `results/toy_maxima.csv`: largest statistic and maximizing mass for each complete experiment.
- `results/null_root_moments.csv`: pointwise mean and sample standard deviation of the signed root.
- `results/family_global.json`: supplementary three-search family calibration.
- `qa/global_validation.json`: input, mask, normalization, seed, fit replay and tail checks.
- `source/report.tex`, `figures/`: report source and vector/PNG figures.

Read CSV with `dtype={'scope':str}` and `float_precision='round_trip'` for exact numerical replays. The add-one estimate is (k+1)/(N+1), while the Clopper-Pearson interval describes the underlying exceedance probability from k successes in N independent complete experiments. Gaussian significance is clipped at zero when p ≥ 1/2; probabilities are never clipped. Earlier reports are not modified. Their hashes are checked when those external files are present.

`MANIFEST.sha256` covers all distributed files except itself. Verify from the package root with `shasum -a 256 -c MANIFEST.sha256`. Rebuilding changes execution timing records, so verify before running a rebuild.
