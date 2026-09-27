# HPS-GPR v5.8.3: interpreting the nominal GP significance study

Read `source/report.pdf`: nine standalone pages explaining the resolution/trials correction, the distinction from 90% CLs, the implemented contribution relative to Ananiev and Read, and what the conditional results establish. `figures/` contains nine new summary plots as vector PDFs and PNGs.

This derivative reuses v5.8.2 numerical fields and maxima. It does not modify the fits, observed results, limits, background policy or prior release. No new HPS likelihood fits, Poisson scans or Gaussian fields are generated. The CLs figure is an explicitly labeled dimensionless analytic illustration; the source comparison reuses v5.8.1.

## Rebuild

Use Python with NumPy >=2, SciPy, pandas and Matplotlib, and Tectonic for the PDF:

```sh
python3 scripts/resolution_audit.py
python3 scripts/summary_figures.py
python3 scripts/validate_study.py
tectonic -X compile source/report.tex
```

All plotting/audit scripts use only the bundled `inputs/` subtree and write to this study directory. `inputs/provenance.json` provides hashes and original source paths for provenance; original paths are not needed to rebuild. `inputs/v5p8p2/scripts/` holds audited historical code snapshots, not a complete fitting engine. For full numerical fitting replay, use the separately delivered v5.8.2 source/data package.

## Numerical interpretation

The global calculation already includes covariance between mass hypotheses. The tail-equivalent independent-test count is threshold specific; it is neither the number of grid points nor the spectral rank of the covariance. The FWHM-spacing comparison and the Gaussian-template upcrossing formula are illustrative benchmarks, not proposed replacements for the fitted-field probability. The upcrossing curves are drawn only for Z >=2.5 and omit the changing combined model.

Global probabilities remain conditional on fixed GP sources estimated from the observed spectra. They include neither source-construction uncertainty nor selection among reported scopes. Local fit GPs do respond to each toy's counts. The 256 coherent Poisson spectra per dataset validate accessible tails; 200,000 Gaussian fields do not establish arbitrary rare-tail accuracy or a physical discovery calibration. The finite half-MeV grid is not a demonstrated continuum limit.

The CLs and source-estimation discussions were reviewed independently by a statistics agent and paper expert. Their review memos are included. The final manuscript incorporates their corrections: strict positive upper-limit-statistic domain for the analytic CLs ratio, analysis-bin terminology, and the explicit Poisson weighting of source-injection projections. PDF pages were rendered and inspected, and numerical/provenance checks are in `qa/`.

## Figure guide

- `peak_summary`: unchanged observed results and direct-scan validation.
- `method_map`: the background GP, fitted-response GP, and separate CLs branch.
- `correlation_resolution`: actual four scope correlation matrices.
- `response_resolution`: correlation versus lag and signal-overlap benchmark.
- `trials_comparison`: actual global tails versus simple counting and smooth-template estimates.
- `trials_effective_counts`: threshold dependence and covariance dimension.
- `combined_domain_effect`: identical combined peak under different declared search domains.
- `cls_vs_discovery`: analytic illustration of exclusion versus excess evidence.
- `grid_and_source_choices`: saved scan-spacing and generating-shape comparisons.

Primary references: [Ananiev and Read](https://arxiv.org/pdf/2206.12328), [Cowan et al.](https://arxiv.org/abs/1007.1727), [Read on CLs](https://doi.org/10.1088/0954-3899/28/10/313).
