# Corrected 10-toy functional-form pull diagnostics

Generated: 2026-06-29

Input CSVs:
- 2015 fixed lslb=1.0: `/Users/emryspeets/Desktop/gp_mods/funcform_studies/2015_toy/inj_extract_toys_2015.csv`
- 2015 profiled lslb=0.5: `/Users/emryspeets/Desktop/gp_mods/funcform_studies/2015_toy/v2/inj_extract_toys_2015.csv`
- 2016 fixed lslb=1.0: `/Users/emryspeets/Desktop/gp_mods/funcform_studies/2016_toy/inj_extract_toys_2016.csv`
- 2016 profiled lslb=0.5: `/Users/emryspeets/Desktop/gp_mods/funcform_studies/2016_toy/v2/inj_extract_toys_2016.csv`

Toy subset: `toy_index = 0..9`, paired exactly across fixed/profiled studies within each dataset.

Correct definitions used here:

```text
pull_i = (A_hat_i - A_inj_i) / sigma_A_i
mean_pull = mean_i(pull_i)
pull_width = std_i(pull_i, ddof=1)
DeltaZ = mean_i(Zhat_i - Zinj)
```

The file `definition_audit_by_group.csv` compares this correct mean pull with two lookalikes:

```text
(mean(A_hat) - mean(A_inj)) / mean(sigma_A)
mean(Zhat - Zinj)
```

Those lookalikes are diagnostics only and are not the pull mean.

Primary outputs:
- `toy_level_10toy.csv`: toy-level paired subset with recomputed pull columns.
- `summary_by_dataset_mass_strength.csv`: group summaries by dataset, study, mass, and injected Z.
- `ranking_table_10toy.csv`: pilot calibration ranking.
- `plots/`: comparison figures.

Important caveat: with only 10 toys per point, pull-width and coverage metrics are intentionally noisy. Treat ranking as a triage screen, not a final configuration freeze.

## Extra lslb subset scan

Additional profiled-background subset runs were generated from the local `func_form_inputs/*_dataset_mod_toys_2.root` files with the same toy indices `0..9`, all five strengths `s0,s1,s2,s3,s5`, and representative masses:

- 2015: `0.045,0.075,0.105` GeV
- 2016: `0.060,0.105,0.150` GeV

These runs test profiled `lslb=0.75` and profiled `lslb=1.0`; they are compared against the matching-mass subsets of the completed fixed `lslb=1.0` and profiled `lslb=0.5` studies.

Extra outputs:

- `subset_lslb_scan_toy_level.csv`
- `subset_lslb_scan_summary.csv`
- `subset_lslb_scan_ranking.csv`
- `plots/*_subset_lslb_scan_mean_pull.png`
- `plots/*_subset_lslb_scan_pull_width.png`
- `plots/subset_lslb_scan_ranking.png`

The generated pilot configs are in `pilot_configs/`. They explicitly set both `kernel_ls_res_lower_factor` and the dataset-specific `kernel_ls_res_lower_factor_by_dataset` override; the first attempted edit missed that override and was discarded.

## Extended 5-mass continuation

The continuation script is `extend_pullwidth_study.py`. It reuses completed CSVs where possible, adds fixed-background `lslb=0.5` and `0.75`, fills missing profiled `lslb=0.75` and `1.0` masses, and adds new profiled `lslb=0.9` and `1.1` scans.

Primary 5-mass grid:

- 2015: `0.045, 0.060, 0.075, 0.090, 0.105` GeV
- 2016: `0.060, 0.090, 0.105, 0.120, 0.150` GeV
- Paired toys: `toy_index = 0..9`
- Strengths: `s0,s1,s2,s3,s5`

Extended outputs:

- `extended_primary_toy_level.csv`
- `extended_primary_summary.csv`
- `extended_primary_ranking.csv`
- `extended_diagnostic_toy_level.csv`
- `extended_diagnostic_summary.csv`
- `extended_diagnostic_ranking.csv`
- `EXTENDED_STUDY_SUMMARY.md`
- `plots/extended_primary_*_top3_*_zoom.png`
- `plots/extended_diagnostic_*_top3_*_zoom.png`

### Ranking summary

The best 5-mass primary candidate in both years is profiled-background `lslb=1.1`.

| dataset | best primary | score | rms mean pull | median pull width | width RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 2015 | `profiled_lslb1p1` | 0.222 | 0.342 | 0.808 | 0.209 |
| 2016 | `profiled_lslb1p1` | 0.213 | 0.421 | 0.913 | 0.112 |

The 2016 result really is closer to unit pull width than implied by the latest note snapshot: on the 5-mass scan, the top profiled rows have median pull widths `0.913` (`lslb=1.1`), `1.028` (`lslb=0.9`), and `0.947` (`lslb=1.0`). The remaining caveat is statistical: this is still only 10 paired toys per point, so it is a strong triage result rather than a final coverage statement.

Fixed-background extraction should not be promoted. The fixed rows have catastrophic pull widths (`~12-38` median width depending year/lslb), even though the fit success rates are nominal. This matches the implementation: fixed extraction is a plug-in diagnostic where the GP covariance is not profiled or marginalized.

### Diagnostic knobs

The no-opt check does not support turning off `refit_optimize` as a meaningful knob here. For both tested candidates (`lslb=1.1` and `lslb=0.9`), setting `refit_optimize=false` changed the CSV flag but left `A_hat`, `sigma_A`, pulls, and refit kernel values identical to the optimized baseline.

The useful third/stress option is a length-scale upper-bound variant, not kernel locking. For `lslb=1.1`, `ls_upper=12` is stable and close to the nominal ranking in both years; `ls_upper=6` is worse. Locking the kernel to the initial fit and tightening the constant-kernel bounds both look acceptable-ish in 2015 but fail as general options because they produce large 2016 mean-pull bias (`rms mean pull ~3`) and worse widths.

### Phi workflow comparison

The Phi `K+K-` workflow uses the same broad statistical lesson: profiled background for interpretation, fixed background only as a diagnostic cross-check. Its profiled mode is not identical to this HPS implementation, though. Phi profiles a smooth multiplicative nuisance, `exp(beta0 + beta1 z + beta2 z^2)`, with Gaussian priors. This HPS study profiles additive Gaussian nuisance directions from the GP background covariance. The expert-statistics recommendation after this scan is therefore:

- keep `extract_background_mode=profiled` as the sensible default;
- keep fixed-background extraction as a negative-control diagnostic only;
- prefer profiled `lslb=1.1` as the current leading candidate, with profiled `lslb=0.9` and `ls_upper=12` as stress/alternate rows;
- do not promote `kernel_lock_mode=initial_fit` or concrete constant-kernel bounds without a new reason.
