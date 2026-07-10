# 2016 length-scale lower-bound optimization, 25 toys

Scope: 2016 only; profiled-background rows `0.9`, `1.0`, and `1.1`; masses `60, 90, 105, 120, 150 MeV`; 25 functional-form toys.

Ranking:

| study | score_lower_is_better | rms_pull_mean_nonzero | pull_width_median_nonzero | pull_width_rmse_nonzero | eps2_median |
| --- | --- | --- | --- | --- | --- |
| profiled_lslb0p9 | 0.188 | 0.321 | 0.948 | 0.100 | 0.000141 |
| profiled_lslb1p0 | 0.194 | 0.377 | 0.924 | 0.113 | 0.000114 |
| profiled_lslb1p1 | 0.203 | 0.403 | 0.889 | 0.128 | 9.58e-05 |

Interpretation notes:

Plot uncertainty convention: regenerated section plots include toy-derived error bars; mean-like points use standard errors, pull widths use `s/sqrt(2(n-1))`, epsilon-squared proxy points use toy 16-84% intervals, and score/heatmap uncertainties use bootstrap resampling within each mass-strength toy cell.

- Compare row-level patterns, not one mass/strength cell at a time; a unit-width pull has about 0.144 one-cell width uncertainty for 25 toys.
- The epsilon-squared values are extraction-only proxies, not final CLs expected bands.
- `0.9`, `1.0`, and `1.1` are all local-neighborhood candidates; final promotion should require stable bias, width, coverage, and proxy reach.

Plots:
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2016_100toy_20260630/section_plots/fig40_style_2016_lslb_closure_suite_25toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2016_100toy_20260630/section_plots/fig43_style_2016_lslb_eps2_proxy_25toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2016_100toy_20260630/section_plots/2016_lslb_score_components_25toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2016_100toy_20260630/section_plots/2016_lslb_pull_width_by_strength_25toy.png`
- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/lengthscale_bounds_2016_100toy_20260630/section_plots/2016_lslb_mean_pull_by_strength_25toy.png`
