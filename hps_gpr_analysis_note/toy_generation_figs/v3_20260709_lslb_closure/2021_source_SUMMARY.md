# 2021 projection-support fSigPowExpQ search50 refmatched closure, upper-9

Date: 2026-07-02

## Repair

The blocked July 1 `fSigPowExpQ` seed in `funcform_2021_data_input21_top2_20260701` normalized a raw exponential-quadratic tail over the full `0-1 GeV` histogram while the closure analysis only used the search window. The first repair normalized over `0.05-0.25 GeV`, but that removed the lower sideband needed to project into the 50 MeV blind window.

This bundle uses a projection-support ROOT file instead:

- `/Users/emryspeets/Desktop/gp_mods/funcform_studies/func_form_inputs/funcform_2021_1pct_search50_project30_top2_toys.root`
- container: `fSigPowExpQ`
- toy support: `0.03-0.25 GeV`
- search range: `0.05-0.25 GeV`
- data range: `0.03-0.25 GeV`
- blind/train/injection exclusion: `2.25 sigma_m`
- injection reference: `matched_refit_bonly`
- length-scale upper factor: `9.0`
- requested masses: `50, 60, 90, 120, 180, 210 MeV`
- injected strengths: `0, 1, 3, 5 sigma`

The nominal family is still `fSigPowExpQ`; the repair is the support geometry plus the restored refmatched sigma reference, not a switch to the diagnostic `fShiftSigPow` seed.

## High-stat ranking

| study | n_source_toys_median | score_lower_is_better | rms_pull_mean_nonzero | pull_width_median_nonzero | pull_width_rmse_nonzero | success_rate_min | refit_ok_rate_min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| profiled_lslb0p9 | 25 | 0.1584 | 0.2133 | 0.8383 | 0.164 | 1 | 1 |
| profiled_lslb1p0 | 25 | 0.1588 | 0.2301 | 0.8598 | 0.1437 | 1 | 1 |
| profiled_lslb1p1 | 25 | 0.1683 | 0.2758 | 0.9077 | 0.122 | 1 | 1 |

## Alternate-vs-best impact

| best_study | comparison_study | median_eps2_exp_ratio_alt_over_best | median_eps2_obs_ratio_alt_over_best | median_pull_width_delta_alt_minus_best | median_pull_mean_delta_alt_minus_best |
| --- | --- | --- | --- | --- | --- |
| profiled_lslb0p9 | profiled_lslb1p0 | 0.8191 | 0.8279 | 0.0257 | -0.01643 |
| profiled_lslb0p9 | profiled_lslb1p1 | 0.6952 | 0.7056 | 0.03595 | -0.02653 |

## Matched-reference audit

The previous concerning `DeltaZ` behavior was a denominator mismatch: injected amplitudes were scaled with the prefit Asimov `sigma_A`, while the extracted amplitudes were evaluated after the full refit. In this corrected bundle, `sigma_A(ref)` is recomputed with a matched background-only refit using the same search support, training mask, kernel locking, and tail-alpha settings as the toy extraction.

| study | matched_refit_ok_min | median_sigma_A_over_used_ref | median_matched_ref_over_prefit_ref | median_DeltaZ_minus_pull |
| --- | --- | --- | --- | --- |
| profiled_lslb0p9 | 1 | 1.0237 | 6.182 | -0.1025 |
| profiled_lslb1p0 | 1 | 1.0173 | 5.1066 | -0.0739 |
| profiled_lslb1p1 | 1 | 1.0127 | 4.3797 | -0.0559 |

This is the key closure repair for note use. The matched reference is several times larger than the prefit reference, which explains the earlier inflated `Z`-scale residual. Once the reference is made equivalent to the extraction fit, `DeltaZ` and the pull mean track each other at the few-tenths level or below, while the direct amplitude bias remains small.

## Validation

- High-stat toy rows: `1800`
- Minimum success rate: `1.0`
- Minimum refit-ok rate: `1.0`
- Minimum qmu-ok rate: `1.0`

## Outputs

- `smoke_toy_level.csv`, `smoke_summary.csv`, `smoke_ranking.csv`
- `highstat_toy_level.csv`, `highstat_summary.csv`, `highstat_ranking.csv`
- `highstat_comparison_by_mass_strength.csv`
- `smoke_reference_diagnostics.csv`, `highstat_reference_diagnostics.csv`
- `validation_summary.json`
- `plots/fig40_style_2021_search50_project30_refmatched_lslb_closure_suite_25toy.{png,pdf}`
- `plots/fig43_style_2021_search50_project30_refmatched_lslb_eps2_proxy_25toy.{png,pdf}`
- `plots/2021_search50_project30_refmatched_lslb_score_components_25toy.{png,pdf}`
- `plots/2021_search50_project30_refmatched_lslb_calibration_distance_heatmap_25toy.{png,pdf}`
- `plots/2021_search50_project30_refmatched_pull_width_by_strength_25toy.{png,pdf}`
- `plots/2021_search50_project30_refmatched_mean_pull_by_strength_25toy.{png,pdf}`
- `plots/2021_search50_project30_refmatched_pull_histograms_m120_25toy.{png,pdf}`

## Claim boundary

These are extraction-closure and epsilon-proxy diagnostics. They are suitable for analysis-note closure/model-selection discussion, but final upper-limit language still requires the production CLs/band workflow and direct coverage checks at the frozen configuration.
