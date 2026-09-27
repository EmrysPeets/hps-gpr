# HPS-GPR v5.5.4: combination models and rate uncertainties

This derivative adds four pages to the retained nine-page v5.5.3 study. The new opening compares common coupling with relaxed rate models using upper limits, local probabilities and an explicitly illustrative Sidak mapping. Figure 2 adds common-coupling, power-law and true exponential fits with normalization/slope uncertainty bands. Figure 3 compares simultaneous count-spectrum fits with fits to signed extracted signals; Figure 4 shows joint rate-parameter contours and a fixed 92 MeV comparison table.

## Main numerical result

At fixed 92 MeV and fully scaled widths, local known-covariance Gaussian-reference Z values are 2.447 (common coupling),3.646 (independent positive amplitudes),3.812 (fitted power law),3.782 (fitted exponential),3.962 (equal-weight signed Stouffer), and 3.811 (Fisher of signed one-sided probabilities). These are different pointwise tests. Choosing the strongest after observing the data or scanning mass requires an additional correction. No global discovery probability is supplied.

The power law is a*(E/2.30)^beta with beta=-2.98691. The true exponential is a*exp[-k*(E-2.30)] with k=1.68057 GeV^-1. Neither is a demonstrated physical production model. Their approximate pointwise 68%/95% log-Wald bands include full profiled normalization/slope covariance but condition on fixed 92 MeV and scaled widths. Exact joint parameter contours use nominal two-parameter likelihood thresholds; they are not coverage-calibrated.

Removing a relative-rate law also removes the unique combined epsilon² coordinate. The new upper-limit panels instead use the total expected reconstructed signal rows in all three fixed fit windows. The 90% Gaussian-reference CLs bounds at 92 MeV are 30,944 (shared coupling) and36,523 (independent signal allocation). Original epsilon² upper limits remain in the retained reference section.

The extracted Gaussian estimates and covariance retain all amplitude information in the stated fixed Gaussian experiment, reproducing the joint amplitude log likelihood within 5e-14. Independent experiments are already multiplied in a simultaneous likelihood; compression does not add independent information. Independent positive amplitudes and Fisher do not require a positive contribution from every campaign.

## Reproduce and inspect

Requirements: Python 3, NumPy, SciPy, pandas, Matplotlib, PyMuPDF, and Tectonic. Run:

```bash
bash scripts/build.sh
```

This reruns all new numerical work serially with one BLAS/OMP thread, regenerates new figures, compiles the 13-page report and performs numerical/provenance/semantic checks. It reuses pinned v5.5.3 inputs and retained results; original paths are provenance and are not required to reproduce the new work. `engine/experiment.py` creates the fixed background experiment without writing artifacts. To rebuild only the document, run `tectonic main.tex` in `source/`.

The primary Gaussian covariance is V=diag(fitted null expectation)+L*L^T, with the fitted background state treated as known and its predictive/auxiliary uncertainty fluctuating. This is not a Poisson-only fixed-auxiliary ensemble or a refitted sideband analysis. No new toys are used. Fitted rate-law tails integrate the chi3 radial tail over Gaussian score directions; angular and slope-bank refinement and independent analytic benchmarks are checked. Shared or separate mass/width results are inherited from v5.5.3, retaining their original qualification and reference envelopes.

After inspecting all final page renders, run `python3 scripts/package.py` to verify the isolated TeX rebuild, create the SHA-256 manifest, and package the PDF and complete source/results. Visual-review hashes bind the inspected artifact. No commit or external publication is performed.

## Files

- `source/main.tex`, `source/additions.tex`, `pdf/`: full report.
- `derived/extracted_combination_scan.csv`, `extracted_combination_covariance.npz`: 17 mass hypotheses, signed estimates, covariance, upper limits and combination rules.
- `derived/rate_significance_scan.csv`: 34 exact rate-law fits and their Gaussian-reference probabilities, with separate Gaussian-statistic and exact-statistic columns.
- `derived/rate_band_fits.json`, `rate_bands.csv`, `rate_parameter_contours.csv`: covariance, conditional bands, slope intervals and 9,882 exact contour profiles.
- `derived/fits.csv`, `fit_bins.csv`, other prior ledgers/figures: unchanged v5.5.3 numerical controls.
- `consults/statistics.md`, `rate_uncertainties.md`, `final_statistics_audit.md`: current consultations and independent audit. Other consultation files retain parent interpretation.
- `provenance/parent_v553.json`: retained-file identities and original parent hashes. The new statistics memo is explicitly marked as superseding its derivative copy; the original remains unchanged.
- `qa/`: new numerical checks, report validation, rendered pages, visual review and portable rebuild. `qa/parent_v553/` preserves original validation records.

The parent 2021 small-sample selection uncertainty, conditional exposure forecasts, assumed resolution constraints and uncalibrated flexible-family references remain unchanged. The new fit bands do not include uncertainty from mass or width selection.
