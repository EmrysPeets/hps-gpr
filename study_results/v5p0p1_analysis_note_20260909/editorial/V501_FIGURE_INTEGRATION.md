Figure derivatives are ready under `figures/` and have been inspected as rendered PNGs.

- Replace v5.0.0 Figures 57–60 with `v501_profiled_correlations_2x2.pdf`. The order is exactly 2015/2016 on top and 2021/combined below. Retain old labels as aliases if inherited references need them.
- Replace the Figure 21 image with `v501_kernel_optimization_clean.pdf`. The original deterministic synthetic sidebands and likelihood surface are retained; explanatory text moved out of the panels. No HPS fit was rerun.
- Add `v501_lengthscale_allowed_regions.pdf` to the resolution-scaled length-scale bounds subsection. This evaluates the frozen card with pinned copies of the actual bound function. The dataset-median floor exists but is nonbinding over every plotted point.
- Restore the kinematic series with `v501_kinematic_integration.tex`, replacing the existing excluded block in Section 3. Its five groups use 20 original vector panels whose hashes match v4.9.7. Main captions are retained, and the new summary subsection references all five groups.
- Add `v501_observed_scan_2015.pdf`, `v501_observed_scan_2016.pdf`, and `v501_observed_scan_2021.pdf` to independent results. They reproduce the three-panel arrangement of v4.9.7 Figure 85 using current saved A90, epsilon-squared, and nominal local p0 values.
- Add `v501_current_projected_babar.pdf` with the exact conditional density scaling of v4.9.7 Figure 84, applied to the current maximal combination and current saved normalization densities. Only the 2021 density is multiplied by ten. The 232-point current curve is unchanged. This remains an observed-equivalent proxy, not expected sensitivity or a projected observed limit.

`v501_figure_integration.tex` contains ready figure captions, prose, labels, and equations for all but the kinematic block. Root editor owns source sections and should copy/adapt the relevant fragments, not input that file wholesale.

Rebuild with `/usr/bin/python3 study_results/v5p0p1_analysis_note_20260909/scripts/make_v501_figures.py` from the repository root. Frozen numerical and implementation inputs are byte-identical snapshots in `provenance/figure_inputs/`; `input_map.json` retains original paths and SHA-256 values. The figure provenance ledger and numerical assertions are in `v501_figure_provenance.json` and `v501_figure_checks.json`. Derived CSVs make every new scientific curve inspectable without rerunning a fit.

The inherited v5 band CSV rounded the large 2015 endpoint at 90 MeV very slightly (relative difference 1.12e-13 from its raw-fit conversion). The new epsilon-squared plot preserves the v5 display value exactly, while the yield panel preserves the original fit value. No numerical repair or smoothing was introduced.
