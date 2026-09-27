# Fixed-background combined-result audit

Read-only, bounded source audit for the small external-statistician note. No fits or parent mutations.

## Resolve the ambiguity first

In v5.0.5, **fixed background in the likelihood** means holding the estimated GP mean fixed while fitting signal: L theta=0, equivalently C=0 for the background nuisance treatment. Its interpolation uncertainty is omitted. This does not mean the GP is never retrained: in the calibration pseudoexperiments the sideband mean is retrained for each spectrum, and then held fixed inside that spectrum's signal fit.

A **fixed generating background source** is a different choice. A full spectrum B is frozen for generating pseudoexperiments, while each experiment's masked GP mean and correlated nuisance covariance can still be recomputed and profiled. The v5.8 significance studies do this. Calling them fixed-background likelihood fits would be wrong.

The nominal v5.0.5 shared-coupling result profiles correlated GP uncertainty. It multiplies campaign likelihoods, imposes one common epsilon-squared, retains separate nuisance blocks, and uses each campaign's signal conversion. It does not add independent per-campaign roots or freely fitted yields. No shared luminosity/radiative-fraction/resolution nuisance is profiled in that baseline. Gaussian GP-background covariance is block diagonal between campaigns.

## What the fixed-versus-profiled study actually establishes

Appendix D (v5_calibration_appendix.tex) reports 41/41 all-three common-range mass points for both branches, not the later232-point union. At matched masses the median raw fixed/profiled observed-limit ratio is0.595, but after empirical two-truth calibration it is1.346. These are upper-limit ratios, not significance ratios or measured expected-sensitivity gains. Thus the apparently stronger uncalibrated fixed result does not survive comparable calibration.

The two joint generating scenarios are all campaigns local-GP and all campaigns archived-stress. Their envelope does not cover mixed assignments or arbitrary background uncertainty. Kernels, support, mask and signal conversion remain frozen; preceding model/support selection is not repeated. The archived2016 source has a failed broad-fit status and unestablished independence from the full data. Passing adjusted tests supports that conditional envelope, not every physical null or precise rare-tail coverage.

The 2021 injection evidence favors profiling: with GP uncertainty present in generation, fixed-background null pull widths are1.85–2.65, and stronger-signal true-value exclusions occur22.0–32.2%, versus8.2–12.2% when profiled. At the selected retrained71MeV example, the corresponding failures are349/500 versus145/500. These are source-dependent comparisons, not proof the profiled procedure is universally calibrated.

At combined66MeV, the stress background gives mean signed local mapping+8.94, width.97, while the local-GP truth gives-.11,width1.01. All500 stress null validation spectra exceed the observed statistic, so the conservative envelope local p is near1. This is a *local conditional calibration result*, not a global particle significance, and it should not be conflated with the nominal observed profiled2.765 below. A five-reference-error signal on localGP passes raw local5% criterion499/500 but the conservative envelope0/500: the note explicitly records the cost of protecting against its stress construction.

## Verified nominal combined numbers: these are PROFILED, not C=0

Source ledger v504_union.csv contains the extended all-three-through100MeV membership matching the v5.0.5 text. The table below retains its exact source values rounded for reading. The v5.8 replay has very small differences (e.g.2.760 at66); do not splice those replay numbers into this note's historical table without labelling them.

| Mass MeV | Local p asymptotic | Local Z asymptotic | Observed epsilon²90 |
|---:|---:|---:|---:|
| 65 | 0.005222487 | 2.560739 | 6.471517e-06 |
| 66 | 0.002844895 | 2.765143 | 6.635651e-06 |
| 76 | 0.4341895 | 0.165718 | 2.384311e-06 |
| 90 | 0.1060445 | 1.247842 | 3.19405e-06 |
| 91 | 0.0183742 | 2.088548 | 4.198555e-06 |
| 92 | 0.005863472 | 2.520256 | 4.674823e-06 |
| 93 | 0.006843453 | 2.465377 | 4.661046e-06 |

The all-three local peak in the note is66MeV,Z2.765. Its full19–250MeV heuristic Sidak reference gives p=.09589,Z1.305; the50–100MeV all-three overlap gives p=.02794,Z1.912. These are declared-domain heuristic mappings, not validated full-scan physical p-values. At92MeV the three-campaign common-coupling fit has localZ2.520 and limit4.67482e-6, with fitted epsilon²3.09613e-6. The likelihood loss relative to independent rates is11.552 for two released amplitude relations, reported as a post-selection diagnostic rather than a calibrated incompatibility probability. At76MeV r~.166 is ordinary; the stress-centered~9 coordinate is a reference mismatch.

**Unavailable within this bounded audit:** exact C=0 combined local-p scan values at65/76/90–93. The nominal profiled numbers above must not be relabelled fixed. The original v4p9p13_calibration_20260905 folder currently contains only.DS_Store; the v5.0.5 portable ZIP retains the report material but no matching calibration CSV. No missing scan was recomputed or inferred from a plot.

**Artifact caveat:** v5.0.5 figures/v5_total_limits_plot_data.csv retains the old pair_2016_2021 membership at91–93, contrary to the extended headline text. For those masses use the inherited v504_union.csv and the explicit92MeV extraction prose. This is a stale auxiliary CSV issue; no claim is made here about rendered-figure correctness without a dedicated plot-source audit.

## Direct sources

- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/source/sections/v5_method_updates.tex` SHA256`b4db3f112002a3802fd42e75ad29bb4096631261625fcfce4b41c2fe4321b7d2`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/source/sections/v5_calibration_appendix.tex` SHA256`a2dfc2f15f6ba8952dfc98391b83d16ade42b862f70da4b6833ef9fe7ec769f6`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/source/sections/04_methodology.tex` SHA256`1059bafe6cec00489372bfe08ec24f50d69a55783b442a306d7dfddf4b09f185`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/source/sections/v5_headline_results.tex` SHA256`2b8854ff2fc46ed9eeb22623230608fa4b6e4a0c9b09d97818db058c5bf8444b`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/source/sections/v5_global_results.tex` SHA256`d308ee8bc53d4e4169ef80de2d689601759355f812c2ccaee6617fbbff62c2a1`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/derived/v502_sidak_peak_table.tex` SHA256`fa2ef73125e0398b60aec3695fdaaed1d807b249bcccdda7803a346dbb2eb789`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p4_analysis_note_20260911/derived/v504_union.csv` SHA256`9088955e743999bd95d3ae65e003a2d70ca1448fe9140b05e619709450d654f4`
- `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/figures/v5_total_limits_plot_data.csv` SHA256`a11d6cfa258676ddafb103be43476c76ed3c1ef8ffe3731f3fb4db98bb07f160`
