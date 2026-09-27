# HPS GPR v6.4.5: mass correlations and the look-elsewhere penalty

This revision retains the nine-page v6.4.4 report and appends a three-page Appendix E. No parent files, fits, local maps or global probabilities change. The original 1,024 calibration scans A and 1,024 independent validation scans B are reused; no new toys or likelihood fits are needed.

The appendix directly measures mass-to-mass correlations, compares them in MC-resolution units, and shows their impact on the global probability. It compares the HPS-style resolution count with the full correlated-scan result. An independence control preserves each empirical marginal local distribution while breaking the dependence between mass hypotheses. It is a counterfactual diagnostic, not another physics result.

The primary procedure applies the frozen A local map to each complete B row and counts its minimum, including ties. This preserves mass-resolution overlap, MC tails, moving fit/exclusion masks and GP-fitting dependence. The combined rows retain the archived one-common-coupling fit. The 2016 window remains ±2.5u, the 2021 window remains [-4,+3]u, and 2015 retains its Gaussian.

| Search | Direct global p | Independent-mass control | MC-resolution Sidak p |
|---|---:|---:|---:|
| 2016 | 0.088 | 0.174 | 0.059 |
| 2021 | 0.226 | 0.418 | 0.174 |
| Combined | 0.074 | 0.133 | Not assigned |

Correlations reduce the global penalty compared with this empirical independence control. The simple mass-span/average-core-width prescription reduces it too far at the observed single-year thresholds. Threshold-dependent Sidak-equivalent counts can express the direct toy probability, but are not universal independent-trial counts or a second validation.

The 2016 HPS paper and internal note use the linear approximation with about 30 resolution regions. The 2015 HPS paper and internal note instead use 4,000 whole-spectrum pseudoexperiments. Public references and hashes/page locations of local documents are recorded in provenance/hps_references.json; internal PDFs are not redistributed.

## Rebuild

Use Python with NumPy2.0+, SciPy, pandas, matplotlib and PyMuPDF, plus Tectonic and its cached fonts. No network access, data generation or fitting is required.

```bash
STUDY_PYTHON=/path/to/scientific/python3 bash rebuild.sh
```

The scripts recompute every B local rank directly from the raw signed-root arrays, verify the retained results, create the figures and build pdf/report.pdf. Frozen input hashes are checked before analysis. The source package includes those inputs, all numerical tables, figures, report source and a preserved copy of the parent PDF. The full upstream extraction package remains the v6.4.4 source release.

## Scope

These are conditional finite-grid global probabilities under the fixed observed-derived null sources and fixed A maps. They do not propagate source or map uncertainty, earlier method/window selection, a continuously optimized mass, or mass-resolution-nuisance uncertainty. The combined minimum reaches the A-map floor; inclusive B ties remain counted. Correlation-coefficient bands show variation across mass pairs, not confidence intervals. Independence-control estimates use empirical B marginals and have no claimed binomial interval. No upper limits are recalculated.

Numerical outputs are results/correlation_global_summary.csv, correlation_global_curves.csv, correlation_summary.csv, correlation_by_resolution_distance.csv, resolution_counts.csv and correlations_*.npz. Provenance records describe all definitions, including the retrospective role of the diagnostic comparisons.
