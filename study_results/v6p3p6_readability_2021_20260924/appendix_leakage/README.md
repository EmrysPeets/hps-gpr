# Blind-window width, signal leakage, and its effect on the fit

These plots extend the window comparison to nine directly simulated v16 TC masses, 80-240 MeV. They were originally produced as an adjacent follow-up. This copy embeds the results in v6.3.6 Appendix I while preserving all preceding numerical studies.

- `window_width_and_signal_leakage.png` / `.pdf`: window edges in historical sigma_m units, total width increase, and MC versus Gaussian leakage under both window choices.
- `leakage_impact_gp_mean.png` / `.pdf`: controlled contamination comparison using the GP mean background source.
- `leakage_impact_functional.png` / `.pdf`: the same diagnostic using the functional form background source.

## Quantitative result

The requested [-4,3] window in fitted-core-width units is 11.3-19.8% wider continuously than the historical +/-2.25 sigma_m window over these masses. Excluded-bin widths differ because the spectrum is binned. At interior masses, the target core parameters are predicted from adjacent samples with that sample omitted, matching the v6.3.7 validation convention; the 80 and 240 MeV endpoints use their directly fitted cores.

The new window reduces the fraction of full selected MC signal entering GP training by 1.12-3.30 percentage points relative to the historical window. Nevertheless, 16.16-27.90% remains in training. A hypothetical shifted Gaussian with the historical analysis width leaves only 1.29-2.19% in those same starter-window bins: the MC leakage is 8.7-16.8 times greater. The continuous ideal Gaussian outside +/-2.25 sigma_m has 2.445% probability; actual binned masks differ slightly. Gaussian columns refer to a different hypothetical signal distribution, not an ability of a Gaussian extraction fit to remove actual MC tails.

## Direct measurement of contamination impact

Reuse the archived background and full signal-MC draws at A = 3 s0 (s0 is the archived Gaussian background-only yield-error scale). For each method, compare the original fit with a control that removes the injected signal ONLY from GP training bins. The signal-fit counts, extraction template, fit window and kernel settings are identical. Recompute GP prediction and covariance from the clean training counts. No new toy draws, selection optimization, observed-data fit or upper limit is performed.

Define R = mean(Ahat_signal+background - Ahat_background) / A. The contamination-induced response loss is mean(Ahat_clean - Ahat_original) / A. Its uncertainty below is the sample standard error of the 100 paired differences. Since removing injected signal has no effect on background-only spectra, the ratio of response-scaled background-only noise is R_clean/R_original; the final column is 100 times this ratio minus one. Its bracket gives the 95% percentile interval from 2,000 whole-toy bootstrap resamples. Masses and methods stay paired; background sources use separate streams. These are conditional finite-toy uncertainties with fixed signal templates, not total systematic uncertainties.

The following values use the direct MC extraction template, the [-4,3] window, and the GP mean background source:

| Mass [MeV] | MC in GP training | Shifted Gaussian in same training bins | Yield lost to contamination [% of full A; mean +/- SE] | Extra response-scaled noise [95% interval] |
|---:|---:|---:|---:|---:|
| 100 | 19.99% | 1.39% | 4.188 +/- 0.019% | 4.37% [4.32, 4.41] |
| 160 | 16.19% | 1.86% | 4.709 +/- 0.025% | 4.96% [4.90, 5.01] |
| 220 | 19.52% | 1.49% | 9.132 +/- 0.046% | 10.05% [9.92, 10.18] |

At 100/160/220 MeV, removing training contamination changes direct-MC responses from 0.959/0.950/0.909 to 1.001/0.997/1.000. Thus contamination explains almost all the residual mean response deficit for an exact MC extraction template in these tests. The Gaussian extraction retains a substantial response deficit even with clean training because its fitted shape and full-yield convention do not match the actual MC distribution. Leakage probability itself is not the fraction of fitted yield lost.

Lines in the impact figures come from deterministic mean-count spectra at all nine masses. Markers and error bars come from 100 paired saved evaluation toys at 100, 160 and 220 MeV. The intermediate curves are not new toy means. This truth-assisted control diagnoses the effect of sideband contamination; real data do not provide the injected-signal labels needed to perform the subtraction. The precision penalty is not a measured upper-limit improvement or a recommendation to enlarge the blind region indiscriminately.

## Files, validation, and reproduction

`window_leakage_by_mass.csv` contains all window and leakage probabilities, including the separate fraction outside the recorded background spectrum. `paired_training_removal_rows.csv` stores the 3,000 new clean-training fit results alongside their archived comparison fits. `paired_training_removal_summary.csv` gives both sources and all five original extraction policies. `mean_count_training_removal.csv` contains deterministic diagnostics. `validation.json` records numerical checks and input hashes.

All 3,000 clean-training fits and 270 deterministic fits passed the existing optimizer and observed-Hessian checks. Thirty original contaminated fits replayed exactly. The six window-leakage comparisons at the production masses reproduce the saved v6.3.7 table. Every file listed in the parent study manifest was verified unchanged.

Run `python3 make_leakage_plots.py` with NumPy, SciPy, pandas and matplotlib to regenerate the controls and figures. Use `python3 make_leakage_plots.py --plots-only` to rebuild figures from the saved CSVs. The script reads the adjacent `v6p3p7_template_windows_20260925` study and uses one numerical thread. No S3DF access is needed.

## Embedded note version

This copy is included in v6.3.6 Appendix I. The numerical script locates its parent at `../appendix_v637`; its original source is preserved in `provenance/`. `make_note_figures.py` builds the four report-sized figures from saved CSVs without performing fits. The validation records copied from the standalone follow-up describe those original computations.

## Gaussian-center comparison (25 September 2026)

The original Gaussian leakage probabilities already used the empirical logarithmically shifted center, not the generated mass. `gaussian_center_leakage_comparison.png` / `.pdf` now shows both centers explicitly across the same nine masses. Both Gaussians retain the historical width sigma_m and the full selected normalization. Each window uses identical blind/training bins for both centers: the historical window is centered at the shifted Gaussian, and the starter window follows the predicted signal-MC core.

At 100, 160 and 220 MeV, the central-mass Gaussian leaves 8.67%, 10.73% and 6.81% in starter-window GP training, versus 1.39%, 1.86% and 1.49% for the shifted Gaussian. The signal-MC leakage remains 19.99%, 16.19% and 19.52%. The comparison changes only the hypothetical Gaussian center; no fit, toy, calibration or upper limit is rerun. Existing fit-impact controls already used shifted Gaussian extraction.

Run `python3 compare_gaussian_centers.py` to rebuild the additional table and standalone plot. `gaussian_center_validation.json` verifies the new shifted-Gaussian integrals reproduce all 18 original probabilities within 1e-12. The companion CSV records both centers, both windows, low/high training fractions, and probability outside the spectrum. Appendix I of the updated v6.3.6 note includes this comparison on page 43.
