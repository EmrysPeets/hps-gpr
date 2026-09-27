# HPS-GPR v6.3.8: observed morphed signal extraction

This standalone supplement scans the observed 2021 10% histogram at 181 mass hypotheses from 60 to 240 MeV. It uses all verified v16 TC signal-MC anchors, interpolates neighboring core locations, widths and complete aligned distributions, and preserves full selected normalization. The 60–80 MeV acceptance-transition interpolation is explicitly an unvalidated model extension. At an anchor the template equals the direct signal MC.

The three comparisons are the neighboring signal-MC template with a [-4,3] core-width fit/training-exclusion window, the historical shifted Gaussian and window, and that same Gaussian in the new window. Dense observed CLs90 and local p-value curves have asymptotic model-dependent interpretations. At the four selected regions, a common full-morph injection source supplies a separate toy-rank upper-limit comparison.

Selected observed excess regions are 67, 79 and 185 MeV; the displayed deficit is 226 MeV. Their local probabilities are not corrected for selecting masses in this scan. Disjoint fit windows share sidebands and are not statistically independent. The deficit has an empty accepted yield set under the toy-rank ordering, never a physical zero limit.

## Reading and rebuilding

Read `pdf/HPS_GPR_v6p3p8_Observed_Morph_Extraction.pdf`. The same section is included as Appendix J in the large v6.3.6 note. Figures use external legends and annotate the observed fit, local asymptotic significance and conditional background-toy probability. Captions define the model components, count errors and GP constraint band. The third panel shows the blind interval and standardized residuals (data or model minus GP mean, divided by sqrt(GP mean + marginal GP variance)), with dashed black and red markers for the search mass and core center. Reading instructions appear only below the figures.

The package requires Python with NumPy, SciPy, pandas and matplotlib, plus Tectonic. The verified local Python is `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`. Tectonic is resolved from `TECTONIC`, PATH or the Homebrew fallback. The LaTeX build uses cached resources; a new installation must populate its cache first.

Run:

```bash
V638_PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3 bash scripts/rebuild.sh
```

This verifies cached numerical checkpoints and reconstructs summaries, figures and the report. For a report-only build, run `python3 scripts/make_figures.py`, then `python3 scripts/build_report.py --build`. All inputs are included; no S3DF access is needed. For a fresh numerical reproduction, work in a copy and preserve then remove that copy's scan/calibration checkpoints before invoking the unchanged frozen protocol.

## Evidence and checks

- `inputs/`: observed spectra, GP-source spectrum, v16 TC histograms/core fits, pinned numerical implementation and archived Gaussian comparison.
- `results/template_grid.npz`: full grid probabilities and outside-support categories.
- `results/observed_scan.csv`: all 543 observed model fits.
- `results/selected_fits/`: observed counts, GP mean/covariance, profiled backgrounds, signed signal and total model for every selected region and method.
- `results/selected_toy_rows.csv`: 40,800 fit rows, with independent background-only and signal-calibration streams.
- `results/selected_local_calibration.csv`: 1,000-toy local tail counts, rank probabilities and exact 95% count intervals.
- `results/selected_rank_limits.csv`: complete accepted sets using 200 calibration toys per grid strength; empty, hole and censor flags retained.
- `provenance/`: frozen protocols, input identities, side-study evidence and parser-fix provenance.
- `qa/`: independent numerical/statistical checks, archived-baseline agreement, exact fit replays, fresh numerical reproduction, cache and portable/report QA.

The 181-point Gaussian reference reproduces the archived observed results to numerical precision. A checkpoint reader's handling of the literal cohort label `null` was corrected before release; a fresh run reproduced all numerical CSVs byte-for-byte, followed by a successful cache replay. Earlier studies and the prior 43-page combined release are preserved. No UC fit, global search calibration, or physical coupling exclusion is added.

The 25 September display revision preserves all 918 checked input and numerical-result files. It adds a standardized residual context panel, removes duplicate in-figure reading text and low-mass qualification from plot headers, and retains that qualification in the explanatory prose. The preceding release is preserved under the output history and provenance directories.
