# 2015 length-scale lower-bound optimization, 100 toys

This directory contains a standalone 2015 continuation of the corrected pull-width diagnostics study.

Generated artifacts:
- `optimizing_length_scale_bounds_section.tex`: LaTeX section ready to paste or adapt into the analysis note.
- `comparison_to_note_and_prior_studies.csv`: comparison table spanning the new 100-toy scan, corrected 10-toy screen, 25-toy context, and note values.
- `section_plots/`: Figure-40/43/46-style plots plus score, heatmap, histograms, nuisance, and imported 10-toy context plots.

Ranking summary:
- `profiled_lslb1p1`: score 0.136, RMS mean pull 0.217, median pull width 0.890, median expected eps2 proxy 8.11e-05.
- `profiled_lslb1p0`: score 0.165, RMS mean pull 0.285, median pull width 0.887, median expected eps2 proxy 9.44e-05.
- `profiled_lslb0p9`: score 0.201, RMS mean pull 0.363, median pull width 0.884, median expected eps2 proxy 0.000112.
- `profiled_lslb0p75`: score 0.351, RMS mean pull 0.732, median pull width 0.895, median expected eps2 proxy 0.000154.
- `profiled_lslb0p5`: score 0.915, RMS mean pull 1.897, median pull width 1.110, median expected eps2 proxy 0.000279.

Physics/statistics interpretation:

Plot uncertainty convention: regenerated section plots include toy-derived error bars; mean-like points use standard errors, pull widths use `s/sqrt(2(n-1))`, epsilon-squared proxy points use toy 16-84% intervals, and score/heatmap uncertainties use bootstrap resampling within each mass-strength toy cell.


- Use the profiled-background rows for the candidate selection; fixed-background rows are negative controls.
- Reject `k_min=0.5` because it can let the GP profile follow signal-like local structure and produces large pull bias in the corrected screening study.
- Treat pull-width differences below roughly 0.07 in one mass/strength cell as statistically small for 100 toys; the ranking relies on coherent behavior across cells.
- Treat epsilon quantities as extraction-only reach proxies until the CLs band workflow is rerun.

Plots:

- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/fig40_style_2015_lslb_closure_suite_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/fig46_style_2015_lslb_pull_deltaz_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/fig43_style_2015_lslb_eps2_proxy_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/2015_lslb_score_components_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/2015_lslb_calibration_distance_heatmap_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/2015_lslb_pull_histograms_m075_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/2015_lslb_pull_width_by_strength_100toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/2015_lslb_mean_pull_by_strength_100toy.png`

Imported corrected 10-toy context plots:

- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/primary_2015_top4_pull_width_by_strength.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/primary_2015_top4_mean_pull_by_strength.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/primary_2015_top4_calibration_distance_heatmap.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/primary_2015_top4_score_components.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/nuisance_2015_top4_reach_calibration_tradeoff.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/nuisance_2015_top4_eps2_ratio_by_mass.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2015_100toy_20260630/section_plots/corrected_10toy_context/highstat_2015_pull_and_eps2_proxy_context.png`

Comparison table preview:

| source | n_source_toys | row | score | rms_pull_mean | median_pull_width | eps2_proxy_median |
| --- | --- | --- | --- | --- | --- | --- |
| new 2015 full-toy scan | 100.0 | profiled_lslb1p1 | 0.136 | 0.217 | 0.890 | 8.11e-05 |
| new 2015 full-toy scan | 100.0 | profiled_lslb1p0 | 0.165 | 0.285 | 0.887 | 9.44e-05 |
| new 2015 full-toy scan | 100.0 | profiled_lslb0p9 | 0.201 | 0.363 | 0.884 | 0.000112 |
| new 2015 full-toy scan | 100.0 | profiled_lslb0p75 | 0.351 | 0.732 | 0.895 | 0.000154 |
| new 2015 full-toy scan | 100.0 | profiled_lslb0p5 | 0.915 | 1.897 | 1.110 | 0.000279 |
| corrected 10-toy screen | 10.00 | profiled_lslb1p1 | 0.222 | 0.342 | 0.808 |  |
| corrected 10-toy screen | 10.00 | profiled_lslb1p0 | 0.248 | 0.376 | 0.777 |  |
| corrected 10-toy screen | 10.00 | profiled_lslb0p9 | 0.265 | 0.425 | 0.894 |  |
| corrected 10-toy screen | 10.00 | profiled_lslb0p75 | 0.413 | 0.788 | 1.036 |  |
| corrected 10-toy screen | 10.00 | profiled_lslb0p5 | 1.043 | 2.092 | 1.116 |  |
| existing 25-toy high-stat context | 25.00 | profiled_lslb1p1 | 0.185 | 0.280 | 0.868 | 8.11e-05 |
| existing 25-toy high-stat context | 25.00 | profiled_lslb1p0 | 0.200 | 0.326 | 0.853 | 9.44e-05 |
| note length-scale scan | 100.0 | k_min=0.5 |  |  |  |  |
| note length-scale scan | 100.0 | k_min=1.0 |  |  | 0.968 |  |
| note length-scale scan | 100.0 | k_min=1.5 |  |  | 1.545 |  |
