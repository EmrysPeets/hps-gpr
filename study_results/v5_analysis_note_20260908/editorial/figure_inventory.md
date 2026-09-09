# Version 5 figure inventory

All sources remain frozen. New figures use saved arrays only.

Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.

## Connected observed-line QA
{
  "total_observed_line": {
    "points": 232,
    "continuous_single_polyline": true,
    "exact_x_and_y_match": true,
    "source_sha256": "3fbecde0e595e416e489734adcf3fefba44c8ff47b1e2fc802974ce6837724c4",
    "source": "study_results/v4p9p12_expanded_snapshot_20260905/derived/final_total_search_window_dimuon_300toys.csv",
    "connected_membership_transitions": [
      [
        38,
        39
      ],
      [
        49,
        50
      ],
      [
        90,
        91
      ],
      [
        180,
        181
      ]
    ]
  },
  "overlay_combined_line": {
    "points": 232,
    "continuous_single_polyline": true,
    "exact_x_and_y_match": true,
    "identical_to_total_observed_line": true,
    "source_sha256": "3fbecde0e595e416e489734adcf3fefba44c8ff47b1e2fc802974ce6837724c4"
  },
  "conditional_echo_response": [
    {
      "dataset": "2015",
      "anchor_MeV": 51,
      "n_masses": 72,
      "formula": "delta_r(m)=K(m,m0)*sqrt(C(m,m))",
      "correlation_covariance_consistent": true,
      "anchor_shift_equals_response_sd": true,
      "source_csv_response_sd_max_abs_difference": 1.1102230246251565e-16
    },
    {
      "dataset": "2016",
      "anchor_MeV": 90,
      "n_masses": 142,
      "formula": "delta_r(m)=K(m,m0)*sqrt(C(m,m))",
      "correlation_covariance_consistent": true,
      "anchor_shift_equals_response_sd": true,
      "source_csv_response_sd_max_abs_difference": 1.1102230246251565e-16
    },
    {
      "dataset": "2021",
      "anchor_MeV": 78,
      "n_masses": 201,
      "formula": "delta_r(m)=K(m,m0)*sqrt(C(m,m))",
      "correlation_covariance_consistent": true,
      "anchor_shift_equals_response_sd": true,
      "source_csv_response_sd_max_abs_difference": 1.1102230246251565e-16
    }
  ],
  "dataset_mass_distributions": [
    {
      "dataset": "2015",
      "input_path": "/Users/emryspeets/research_plots/2015_data/invariant_mass_0pt5mm_full.root",
      "histogram_key": "invariant_mass",
      "actual_sha256": "58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f",
      "frozen_sha256": "58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f",
      "frozen_hash_match": true,
      "native_bins": 3000,
      "native_width_MeV": 0.04999999999999716,
      "histogram_total_counts": 21451676.0,
      "support_MeV": [
        14.0,
        135.0
      ],
      "search_MeV": [
        19.0,
        90.0
      ],
      "density_integral_matches_counts": true
    },
    {
      "dataset": "2016",
      "input_path": "/Users/emryspeets/root_files/EventSelection_pass4Full.root",
      "histogram_key": "h_Minv_General_Final_1",
      "actual_sha256": "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301",
      "frozen_sha256": "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301",
      "frozen_hash_match": true,
      "native_bins": 6000,
      "native_width_MeV": 0.05000000000000071,
      "histogram_total_counts": 73220323.0,
      "support_MeV": [
        30.0,
        210.0
      ],
      "search_MeV": [
        39.0,
        180.0
      ],
      "density_integral_matches_counts": true
    },
    {
      "dataset": "2021",
      "input_path": "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root",
      "histogram_key": "preselection/h_invM_8000",
      "actual_sha256": "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4",
      "frozen_sha256": "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4",
      "frozen_hash_match": true,
      "native_bins": 8000,
      "native_width_MeV": 0.125,
      "histogram_total_counts": 141590915.0,
      "support_MeV": [
        36.0,
        300.0
      ],
      "search_MeV": [
        50.0,
        250.0
      ],
      "density_integral_matches_counts": true
    }
  ]
}

## individual_final_results
Requested: 
Change: Unmodified vector source copy.
Caption: Historical Harvard writing sample figure; current result figures supersede its presentation where specified.
Source: `study_results/harvard_writing_sample_final_combinations_20260902/figures/individual_final_results.pdf` SHA256 `b31f1d18631c1e6a02d0510c0dd02c2a4bea4b7743cc7cc53b1bc30ba8830efa`

## combined_final_results
Requested: 
Change: Unmodified vector source copy.
Caption: Historical Harvard writing sample figure; current result figures supersede its presentation where specified.
Source: `study_results/harvard_writing_sample_final_combinations_20260902/figures/combined_final_results.pdf` SHA256 `ccc64a9155da47ea06e148dba469356c038d3bf153884988c59ad9740da925d8`

## final_asymptotic_pvalues
Requested: 
Change: Unmodified vector source copy.
Caption: Historical Harvard writing sample figure; current result figures supersede its presentation where specified.
Source: `study_results/harvard_writing_sample_final_combinations_20260902/figures/final_asymptotic_pvalues.pdf` SHA256 `b5db899ca46150772fd5b37cb0c217650ec1118bee21a29f9599fdc0040a9581`

## all_three_peak_extraction
Requested: 
Change: Unmodified vector source copy.
Caption: Historical Harvard writing sample figure; current result figures supersede its presentation where specified.
Source: `study_results/harvard_writing_sample_final_combinations_20260902/figures/all_three_peak_extraction.pdf` SHA256 `3d67669e4dba0aed7e09f3b69329dd4c79e6d30c87eec218d2cf7f3b5c277fb0`

## v5_nominal_mass_resolutions
Requested: v4p9p12 tail Figure 14
Change: Unmodified vector source copy.
Caption: Frozen nominal resolution input.
Source: `study_results/v4p9p12_targeted_tail_refinement_20260905/figures/nominal_mass_resolutions.pdf` SHA256 `6e7c2c58e03549ce79693c9989e81a042318fe5d85ed7f5e2c47619b69afa62e`

## v5_resolution_width_limits
Requested: v4p9p12 tail Figure 15
Change: Unmodified vector source copy.
Caption: Signal-template width variation with frozen background states; this is a conditional response study, not a resolution-nuisance profile.
Source: `study_results/v4p9p12_targeted_tail_refinement_20260905/figures/resolution_width_limits.pdf` SHA256 `8d8baaba693309321816673a1f13e318d9ff47db2036bab2bc388e7f3c82e0fd`

## v5_profile_background_comparison
Requested: v4p9p13 Figure 1
Change: Regenerated from saved ledger; moved nonlegend figure text into caption.
Caption: Same 2021 10% data compared under the Gaussian profile, direct log-GP profile and fixed GP mean. Ratios use the released limit. Same data, kernels, masks, resolution and yield conversion. Conditional asymptotic observed comparison.
Source: `study_results/v4p9p13_background_profiling_20260905/comparison/make_figures.py` SHA256 `64f7c0ec45789c3270fbe4c98b9e09df5bafa4cdc567322978db4fc46d1a072c`
Source: `study_results/background_profile_comparison_20260905/derived/observed_limits.csv` SHA256 `f23160744ae1f415f521a9db96d2c20f8b78268a65fd1af0913bc82b66db4b13`

## v5_profile_injected_bias
Requested: v4p9p13 Figure 9
Change: Regenerated from saved ledger; moved nonlegend figure text into caption.
Caption: Mean pull for injections at 2 sigma_ref and 5 sigma_ref; both methods use the same physical signal yield, defined using the profiled background-only reference uncertainty. 500 spectra per cell. Bars are Monte Carlo standard errors of the mean pull. The retrained 71 MeV 5 sigma_ref means are -2.33 fixed and -0.81 profiled. Same physical injected yield for both methods; strengths use the reference profiled Fisher error. These are conditional truth tests.
Source: `study_results/v4p9p13_background_profiling_20260905/injections/make_figures.py` SHA256 `883f5efaf9d34a4e20d7f92b1697c3eed7c77efbeb49ea81477811fdc3f5fac4`
Source: `study_results/v4p9p13_background_profiling_20260905/injections/derived/extraction_summary.csv` SHA256 `ed19216fb305b074245028f58d69c9d6f353720820d465b0a77664c4ed8a70ea`

## v5_total_limits_with_local_pvalues
Requested: v4p9p12 tail Figure 1
Change: Rebuilt Figure 1 from exact saved quantiles with asymptotic p0 beneath; adjacent observed points are connected across all active-dataset transitions; no changed fits or new toys.
Caption: Mass-by-mass maximal available combination. Bands are central pointwise quantiles of the original 300 conditional background-only toys per mass, with frozen GP states. The lower panel adds the unchanged nominal local asymptotic p0, with no look-elsewhere correction. Active datasets change at the indicated boundaries; 2021 uses its released 10% sample. The 2016 numerical/source qualification remains applicable.
Source: `study_results/v4p9p12_expanded_snapshot_20260905/derived/final_total_search_window_dimuon_300toys.csv` SHA256 `3fbecde0e595e416e489734adcf3fefba44c8ff47b1e2fc802974ce6837724c4`

## v5_observed_coupling_overlay
Requested: Observed coupling overlay
Change: New requested four-curve observed-only overlay; the union curve connects all 232 adjacent masses including active-dataset transitions.
Caption: Observed 90% CLs coupling limits from the three individual released samples and the maximal-available-dataset common-coupling combination across 19--250 MeV. Its active membership is 2015 at 19--38, 2015+2016 at 39--49, all three at 50--90, 2016+2021 at 91--180 and 2021 at 181--250 MeV. This is the Figure 1 combined curve, not a minimum over individual limits. The inherited dimuon correction is applied once above threshold; 2016 retains its qualification.
Source: `study_results/v4p9p12_expanded_snapshot_20260905/derived/expected_band_summary_dimuon_300toys.csv` SHA256 `5caa6781433bc4a85e9a0fab3bd62b53984422f761f0201c05482d77073a2195`
Source: `study_results/v4p9p12_expanded_snapshot_20260905/derived/final_total_search_window_dimuon_300toys.csv` SHA256 `3fbecde0e595e416e489734adcf3fefba44c8ff47b1e2fc802974ce6837724c4`

## v5_individual_expected_bands
Requested: v4p9p12 tail Figures 2--4
Change: Rebuilt from saved quantiles; removed optimized-support wording from titles.
Caption: Observed limits and unchanged central 68% and 95% pointwise expected bands from 300 conditional toys per mass. The GP state remains frozen in these ensembles; 2016 remains qualified.
Source: `study_results/v4p9p12_expanded_snapshot_20260905/derived/expected_band_summary_dimuon_300toys.csv` SHA256 `5caa6781433bc4a85e9a0fab3bd62b53984422f761f0201c05482d77073a2195`

## v5_combination_expected_bands
Requested: v4p9p12 tail Figures 2--4
Change: Rebuilt from saved quantiles; removed optimized-support wording from titles.
Caption: Observed limits and unchanged central 68% and 95% pointwise expected bands from 300 conditional toys per mass. The GP state remains frozen in these ensembles; 2016 remains qualified.
Source: `study_results/v4p9p12_expanded_snapshot_20260905/derived/expected_band_summary_dimuon_300toys.csv` SHA256 `5caa6781433bc4a85e9a0fab3bd62b53984422f761f0201c05482d77073a2195`

## v5_all_three_expected_bands
Requested: v4p9p12 tail Figures 2--4
Change: Rebuilt from saved quantiles; removed optimized-support wording from titles.
Caption: Observed limits and unchanged central 68% and 95% pointwise expected bands from 300 conditional toys per mass. The GP state remains frozen in these ensembles; 2016 remains qualified.
Source: `study_results/v4p9p12_expanded_snapshot_20260905/derived/expected_band_summary_dimuon_300toys.csv` SHA256 `5caa6781433bc4a85e9a0fab3bd62b53984422f761f0201c05482d77073a2195`

## v5_calibration_limits_2015
Requested: 
Change: Regenerated source plot from saved ledger; removed nonlegend figure text.
Caption: Conditional calibrated and asymptotic observed limits using the released data and saved paired background treatments. Shading is approximate 95% Monte Carlo uncertainty on the calibrated endpoint, not expected bands. Open circles have limited MC precision; triangles indicate unresolved finite endpoints. The calibration is conditional on the two generating truth scenarios. Figure annotations moved to caption: Observed 90% CLs; conditional calibration with reviewed kernels fixed Shading: approximate 95% Monte Carlo uncertainty, not expected-limit bands.
Open circles: limited MC precision. Triangles: no finite endpoint. Both limits target 90% CLs.
Source: `study_results/v4p9p13_calibration_20260905/make_figures.py` SHA256 `1848f4c0c7a179e6e82e072966f09b134e98611ca1cc0d5363769e57bcb710d7`
Source: `study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv` SHA256 `c6916a15307820f648fee76149055bf46405dcfcfe6c5f86f30d50b4a84f3fce`

## v5_calibration_limits_2016
Requested: 
Change: Regenerated source plot from saved ledger; removed nonlegend figure text.
Caption: Conditional calibrated and asymptotic observed limits using the released data and saved paired background treatments. Shading is approximate 95% Monte Carlo uncertainty on the calibrated endpoint, not expected bands. Open circles have limited MC precision; triangles indicate unresolved finite endpoints. The calibration is conditional on the two generating truth scenarios. Figure annotations moved to caption: Observed 90% CLs; conditional calibration with reviewed kernels fixed Shading: approximate 95% Monte Carlo uncertainty, not expected-limit bands.
Open circles: limited MC precision. Triangles: no finite endpoint. Both limits target 90% CLs.
Source: `study_results/v4p9p13_calibration_20260905/make_figures.py` SHA256 `1848f4c0c7a179e6e82e072966f09b134e98611ca1cc0d5363769e57bcb710d7`
Source: `study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv` SHA256 `c6916a15307820f648fee76149055bf46405dcfcfe6c5f86f30d50b4a84f3fce`

## v5_calibration_limits_2021
Requested: 
Change: Regenerated source plot from saved ledger; removed nonlegend figure text.
Caption: Conditional calibrated and asymptotic observed limits using the released data and saved paired background treatments. Shading is approximate 95% Monte Carlo uncertainty on the calibrated endpoint, not expected bands. Open circles have limited MC precision; triangles indicate unresolved finite endpoints. The calibration is conditional on the two generating truth scenarios. Figure annotations moved to caption: Observed 90% CLs; conditional calibration with reviewed kernels fixed Shading: approximate 95% Monte Carlo uncertainty, not expected-limit bands.
Open circles: limited MC precision. Triangles: no finite endpoint. Both limits target 90% CLs.
Source: `study_results/v4p9p13_calibration_20260905/make_figures.py` SHA256 `1848f4c0c7a179e6e82e072966f09b134e98611ca1cc0d5363769e57bcb710d7`
Source: `study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv` SHA256 `c6916a15307820f648fee76149055bf46405dcfcfe6c5f86f30d50b4a84f3fce`

## v5_calibration_limits_combined
Requested: 
Change: Regenerated source plot from saved ledger; removed nonlegend figure text.
Caption: Conditional calibrated and asymptotic observed limits using the released data and saved paired background treatments. Shading is approximate 95% Monte Carlo uncertainty on the calibrated endpoint, not expected bands. Open circles have limited MC precision; triangles indicate unresolved finite endpoints. The calibration is conditional on the two generating truth scenarios. Figure annotations moved to caption: Observed 90% CLs; conditional calibration with reviewed kernels fixed Shading: approximate 95% Monte Carlo uncertainty, not expected-limit bands.
Open circles: limited MC precision. Triangles: no finite endpoint. Both limits target 90% CLs.
Source: `study_results/v4p9p13_calibration_20260905/make_figures.py` SHA256 `1848f4c0c7a179e6e82e072966f09b134e98611ca1cc0d5363769e57bcb710d7`
Source: `study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv` SHA256 `c6916a15307820f648fee76149055bf46405dcfcfe6c5f86f30d50b4a84f3fce`

## v5_calibration_local_pvalues
Requested: 
Change: Regenerated source plot from saved ledger; removed nonlegend figure text.
Caption: Conditional calibrated and asymptotic observed limits using the released data and saved paired background treatments. Shading is approximate 95% Monte Carlo uncertainty on the calibrated endpoint, not expected bands. Open circles have limited MC precision; triangles indicate unresolved finite endpoints. The calibration is conditional on the two generating truth scenarios. Figure annotations moved to caption: No global trials correction. Triangles: asymptotic $p_0<10^{-5}$. Open circles: limited MC precision or MC boundary.
Source: `study_results/v4p9p13_calibration_20260905/make_figures.py` SHA256 `1848f4c0c7a179e6e82e072966f09b134e98611ca1cc0d5363769e57bcb710d7`
Source: `study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv` SHA256 `c6916a15307820f648fee76149055bf46405dcfcfe6c5f86f30d50b4a84f3fce`

## v5_calibration_validation_exclusion
Requested: 
Change: Regenerated exact saved validation cells; moved explanatory text into caption.
Caption: Signal injection $A_{\rm true}=5\sigma_{\rm ref}$; paired background treatments Complete mass grid: 456 of 456 mass hypotheses; 500 independent spectra per cell One point per mass, truth and method; exclusion means CL$_s(A_{\rm true})<0.10$. Dashed guides mark 0.10. Cells retain their own counts; the calibrated test uses the two-truth envelope.
Source: `study_results/v4p9p13_calibration_20260905/make_validation_figures.py` SHA256 `82abf00f513a45b8b9411110270e79acf5746056563fdc26a7a6633ddd98ddcd`
Source: `study_results/v4p9p13_calibration_20260905/summary/validation_summary.csv` SHA256 `fed5a8e548480ef3575942bef0a8dd2d5491f079068d87b8b5eb5cab4f28d7f8`

## v5_calibration_bias_by_truth
Requested: 
Change: Regenerated exact saved validation cells; moved explanatory text into caption.
Caption: Ensemble mean amplitude relative to the fixed reference uncertainty $\sigma_{\rm ref}$ Complete mass grid: 456 of 456 mass hypotheses; 500 independent spectra per cell Shading: approximate pointwise 95% Monte Carlo intervals for the ensemble mean; no simultaneous band. $\sigma_{\rm ref}$ is the frozen Gaussian-profile Fisher uncertainty at each mass, common to both methods. This is a mean amplitude in fixed reference units, not the historical mean of per-toy pulls. No counts are pooled.
Source: `study_results/v4p9p13_calibration_20260905/make_validation_figures.py` SHA256 `82abf00f513a45b8b9411110270e79acf5746056563fdc26a7a6633ddd98ddcd`
Source: `study_results/v4p9p13_calibration_20260905/summary/validation_summary.csv` SHA256 `fed5a8e548480ef3575942bef0a8dd2d5491f079068d87b8b5eb5cab4f28d7f8`

## v5_calibration_truth_dependence
Requested: 
Change: Regenerated exact truth-specific endpoints; removed nonlegend figure text.
Caption: Conditional observed 90% CL$_s$ endpoint: archived stress truth / mass-local GP truth Complete mass grid: 456 of 456 mass hypotheses; each ratio uses two truth-specific endpoints The dashed line at 1 denotes equal observed endpoints; this ratio is not a coverage test. Open circles: at least one endpoint has limited MC precision. Missing or censored endpoints leave gaps. The all-three comparison covers two joint truth scenarios only; it does not cover every mixed constituent truth.
Source: `study_results/v4p9p13_calibration_20260905/make_truth_figure.py` SHA256 `3f2695ea169e0f9ea055df33d10cbbb106159f73b7e1be3baf0f733ade1e8c6f`
Source: `study_results/v4p9p13_calibration_20260905/summary/truth_specific_limits.csv` SHA256 `da5d6fe49348c01615cf87c24251eebb268eb5cef5aa808767899043833e7a84`

## v5_resolution_width_significance
Requested: v4p9p12 tail Figure 16
Change: Changed ordinate labels; regenerated exact saved width scans.
Caption: 2021 signal-template widths from 0.8 to 1.2 times the frozen nominal resolution, leaving the background state fixed. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/derived/width_scan_all_points.csv` SHA256 `8f85d06069ae24597c207e7a343aeb7dcbc1d845915fe4303f0114eaa23c080c`

## v5_correlations_2015
Requested: 
Change: Selected profiled matrix only; updated axis/colorbar language.
Caption: Profiled-response correlations under the archived generating background. Positive correlation denotes common-direction fluctuations; negative correlation denotes opposite-direction fluctuations. This matrix measures statistical dependence and does not assign an echo probability.
Source: `study_results/v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz` SHA256 `50354106be0eb505c8fd4712163197a992c76387303b3e8365851db82a62d1ff`

## v5_correlations_2016
Requested: 
Change: Selected profiled matrix only; updated axis/colorbar language.
Caption: Profiled-response correlations under the archived generating background. Positive correlation denotes common-direction fluctuations; negative correlation denotes opposite-direction fluctuations. This matrix measures statistical dependence and does not assign an echo probability.
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016/analysis/covariance.npz` SHA256 `08d63e7cb3d10bf4b0e260b4af9a4dfa81294ffb99177f5fac7b92927c93a4de`

## v5_correlations_2021
Requested: 
Change: Selected profiled matrix only; updated axis/colorbar language.
Caption: Profiled-response correlations under the archived generating background. Positive correlation denotes common-direction fluctuations; negative correlation denotes opposite-direction fluctuations. This matrix measures statistical dependence and does not assign an echo probability.
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021/analysis/covariance.npz` SHA256 `407e8b1ad3186aff5ae0341801d37c06ad72d65994d5ff8085bbe50d6c893e5d`

## v5_profiled_correlations_all_datasets
Requested: Correlation plots for all datasets; v4p9p15 Figures 5--6
Change: Selected only profiled matrices; unified labels and color scale.
Caption: Profiled-response correlation matrices under the archived generating background for each dataset. Red entries show same-direction fluctuations and blue entries opposite-direction fluctuations. These dimensionless correlations describe statistical dependence, not a significance of an induced echo and not a correlation of physical signal production.
Source: `study_results/v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz` SHA256 `50354106be0eb505c8fd4712163197a992c76387303b3e8365851db82a62d1ff`
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016/analysis/covariance.npz` SHA256 `08d63e7cb3d10bf4b0e260b4af9a4dfa81294ffb99177f5fac7b92927c93a4de`
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021/analysis/covariance.npz` SHA256 `407e8b1ad3186aff5ae0341801d37c06ad72d65994d5ff8085bbe50d6c893e5d`

## v5_correlation_induced_fluctuation_scales
Requested: Induced oscillation scale for all datasets
Change: New analytically derived correlation illustration; no fits or random samples.
Caption: Illustrative Gaussian conditional mean of standardized fluctuations: E[z(m) | z(m0)=3] = 3 K(m,m0), using the archived profiled correlation matrices. Anchor masses are the displayed individual candidates. This expresses the correlated fluctuation scale in nominal standard-deviation units; it is neither a signal-injection response nor a measured echo significance. The explicit 2021 injection response is shown separately.
Source: `study_results/v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz` SHA256 `50354106be0eb505c8fd4712163197a992c76387303b3e8365851db82a62d1ff`
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016/analysis/covariance.npz` SHA256 `08d63e7cb3d10bf4b0e260b4af9a4dfa81294ffb99177f5fac7b92927c93a4de`
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021/analysis/covariance.npz` SHA256 `407e8b1ad3186aff5ae0341801d37c06ad72d65994d5ff8085bbe50d6c893e5d`

## v5_extraction_combined_66
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Common-amplitude fit at 66 MeV; signed local significance r=+2.760. Each dataset uses its own resolution and yield conversion. The final panel sums only common whole bins and has no separate fit. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_extraction_combined_92
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Common-amplitude fit at 92 MeV; signed local significance r=+2.416. Each dataset uses its own resolution and yield conversion. The final panel sums only common whole bins and has no separate fit. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_extraction_combined_72
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Common-amplitude fit at 72 MeV; signed local significance r=-3.490. Each dataset uses its own resolution and yield conversion. The final panel sums only common whole bins and has no separate fit. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_extraction_2015_peaks
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Two leading separated observed individual excesses, selected from the completed scan. 51 MeV: r=+3.139, 21 MeV: r=+2.516. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_extraction_2016_peaks
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Two leading separated observed individual excesses, selected from the completed scan. 90 MeV: r=+3.425, 117 MeV: r=+3.279. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_extraction_2021_peaks
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Two leading separated observed individual excesses, selected from the completed scan. 78 MeV: r=+2.809, 65 MeV: r=+2.396. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_extraction_individual_deficits
Requested: v4p9p16 Candidate Removal Figures 11--17
Change: Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.
Caption: Deepest observed deficit in each individual scan. The 2015 mass is the search endpoint. Negative templates are auxiliary deficit diagnostics, not physical negative event rates. Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_arrays.npz` SHA256 `691f538d31f6ce1e4e1cde4db35c9b9ff0db75fbde142f2c84c9da65546985bf`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_dataset_amplitude_consistency
Requested: v4p9p16 Candidate Removal Figure 18
Change: Replaced delta-D label by shared-rate likelihood loss; retained exact definition in caption.
Caption: Separate year amplitudes and local curvature standard errors, with the common-amplitude estimate and its curvature interval in red. Shared-rate likelihood loss means 2[NLL_common - sum(NLL_individual free)]. The number of additional amplitudes is one fewer than the number of contributing datasets. These selected masses provide descriptive compatibility checks; no calibrated or post-selection compatibility probability is assigned.
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/dataset_consistency.csv` SHA256 `5d6a007b50935021b99f7735ca9e8219d5652c679863a6696c0bd23702dcc03c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_summary.csv` SHA256 `e5a4cdcdacecbb023829b37827a946017aef8dbd227c272aa423654a92f4653c`
Source: `study_results/v4p9p16_presentation_extractions_20260906/derived/fit_closure.json` SHA256 `e5fd9a1b263027300bcca2e2c66dcb31b420bbbb61d232ec1be6c6aa20a11a07`

## v5_signal_echo_dense_replay
Requested: v4p9p16 Candidate Removal Figure 4
Change: Regenerated saved response curves; replaced signed-root labels and removed annotations.
Caption: Archived deterministic positive-signal injections in the 2021 moving-sideband fit. One injected peak induces both positive and negative fitted responses at neighboring hypotheses; two selected positive injections can make a dip while overshooting the observed peaks. This demonstrates a possible response mechanism, not the physical origin of the observed oscillations. The middle panel is a difference of two signed fit responses, not an independently calibrated echo probability. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_probability_echo_review_20260906/derived/echo_dense_scans.csv` SHA256 `d76db1217d1fe6784e5f2fa368c3f95076907f1ea4f288316d19e3a68a2f098b`

## v5_observed_candidate_removal
Requested: v4p9p16 Candidate Removal Figure 5 (related supporting study)
Change: Regenerated exact completed intervention scans; clearer significance label with definition retained.
Caption: Observed scans before and after replacing the preselected candidate regions with conditional GP predictions. Shaded vertical regions identify the replaced bins. The green envelope spans the ten paired conditional replacements and is not a confidence band. Changes at neighboring masses illustrate the coupling introduced by the moving background fit; they do not establish the cause of the original pattern. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/holes.csv` SHA256 `309bff88a5f7b9586a55dddf46a5a7adb01e6183638006cc8077881d0e0d367a`
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/2015/scans.csv` SHA256 `756c95667242ed7a71e0e8c3f81034c72a3acbd58885ef6d129a2ad4eacc561a`
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/2016/scans.csv` SHA256 `ff9ed40972af4df815d2480a9b171b9bcfaf307595fe99efc8f09da357b38111`
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/2021/scans.csv` SHA256 `40b44e63ab44e4ba2a9b057f94737a2a1b4d373460ad09a29dbb79adf1617afc`

## v5_conditional_echo_response
Requested: Requested induced oscillation response for each dataset
Change: New deterministic conditional-response slice from saved covariance matrices; no refits or simulations.
Caption: Gaussian conditional mean response under each archived stress background. Let r(m) have mean a(m), covariance C, standard deviation s(m)=sqrt(C_mm), and correlation K; define z(m)=[r(m)-a(m)]/s(m). The imposed condition is z(m0)=+1 at the fixed displayed anchors 51, 90 and 78 MeV. The curve is E[r(m)-a(m) | z(m0)=1]=K(m,m0)s(m), expressed in units of the nominal signed local significance. It shows the correlated mean shift after removing the reference offset, with no signal injection and no probability assigned to an observed echo. It is distinct from the deterministic 2021 positive-signal injection test. The source response widths and covariance normalization are cross-checked numerically.
Source: `study_results/v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz` SHA256 `50354106be0eb505c8fd4712163197a992c76387303b3e8365851db82a62d1ff`
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016/analysis/covariance.npz` SHA256 `08d63e7cb3d10bf4b0e260b4af9a4dfa81294ffb99177f5fac7b92927c93a4de`
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021/analysis/covariance.npz` SHA256 `407e8b1ad3186aff5ae0341801d37c06ad72d65994d5ff8085bbe50d6c893e5d`

## v5_correlations_combined
Requested: v4p9p16 Signal Extractions Figure 14
Change: Regenerated profiled union correlation matrix with plain-language label.
Caption: Profiled-response correlations across the 232-point union search. Dotted lines mark changes in active datasets. Shared datasets induce correlations across membership boundaries; regions with no common dataset have zero response covariance in the stated independent-dataset model. The color scale is a dimensionless correlation, not a significance or signal-production probability.
Source: `study_results/v4p9p16_combined_global_20260906/global/analysis/covariance.npz` SHA256 `2eb5dcd4723a74be096b18dddfa99e002619ec6209896549d1e6edb9ef9a4262`

## v5_replacement_model_comparison
Requested: Related candidate-removal appendix Figure 6
Change: Regenerated saved intervention scans, clean labels and no numeric boxes.
Caption: Both candidate regions are replaced in observed counts (left) or the archived reference spectrum (right), using each spectrum separately to learn its replacements. Curves compare the original, primary GP, polynomial and wider-hole GP replacements. Shading marks the primary holes, which are narrower than the wider-hole comparison. All outcomes are retained, including unsuccessful polynomial interpolation. The ordinate is the same nominal signed local fit mapping; the reference response is not an event-count spectrum or newly calibrated significance. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/holes.csv` SHA256 `309bff88a5f7b9586a55dddf46a5a7adb01e6183638006cc8077881d0e0d367a`
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/2015/scans.csv` SHA256 `756c95667242ed7a71e0e8c3f81034c72a3acbd58885ef6d129a2ad4eacc561a`
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/2016/scans.csv` SHA256 `ff9ed40972af4df815d2480a9b171b9bcfaf307595fe99efc8f09da357b38111`
Source: `study_results/v4p9p16_candidate_removal_20260906/derived/2021/scans.csv` SHA256 `40b44e63ab44e4ba2a9b057f94737a2a1b4d373460ad09a29dbb79adf1617afc`

## v5_traditional_2015_display
Requested: Related traditional-search appendix
Change: Regenerated stored whole-bin display arrays; moved fit statistics and all explanatory text into caption.
Caption: Conventional polynomial-background fits at GP-selected masses. 51 MeV: polynomial degree 3, total width 13 resolution widths, r=+0.131, deviance/dof=141.69/131; 21 MeV: polynomial degree 5, total width 14 resolution widths, r=+5.580, deviance/dof=92.50/50. Counts and predictions use the saved whole-bin grouping, divided by each actual bin width; the last partial group is retained. Counting errors only. Residuals subtract the null-background fit. Lines connect stored bin-averaged model predictions. The quoted local fit statistics do not incorporate the selection of these masses from the GP scan.
Source: `study_results/v4p9p16_candidate_removal_20260906/traditional/qa/paper_display_groups.npz` SHA256 `57567e007a1f5ecbd1cc3567b573c3e1f7ac0557c4fb034d053391cfa8036e66`
Source: `study_results/v4p9p16_candidate_removal_20260906/traditional/derived/fit_summary.csv` SHA256 `e1bcc03bd8819e3a1db23377f0c61e8cbd12840d040e0dbea7b5e38bc528f9da`

## v5_traditional_2016_display
Requested: Related traditional-search appendix
Change: Regenerated stored whole-bin display arrays; moved fit statistics and all explanatory text into caption.
Caption: Conventional polynomial-background fits at GP-selected masses. 90 MeV: polynomial degree 3, total width 8 resolution widths, r=+1.998, deviance/dof=125.35/113; 117 MeV: polynomial degree 3, total width 8 resolution widths, r=+1.525, deviance/dof=165.59/155. Counts and predictions use the saved whole-bin grouping, divided by each actual bin width; the last partial group is retained. Counting errors only. Residuals subtract the null-background fit. Lines connect stored bin-averaged model predictions. The quoted local fit statistics do not incorporate the selection of these masses from the GP scan.
Source: `study_results/v4p9p16_candidate_removal_20260906/traditional/qa/paper_display_groups.npz` SHA256 `57567e007a1f5ecbd1cc3567b573c3e1f7ac0557c4fb034d053391cfa8036e66`
Source: `study_results/v4p9p16_candidate_removal_20260906/traditional/derived/fit_summary.csv` SHA256 `e1bcc03bd8819e3a1db23377f0c61e8cbd12840d040e0dbea7b5e38bc528f9da`

## v5_traditional_2021_display
Requested: Related traditional-search appendix
Change: Regenerated stored whole-bin display arrays; moved fit statistics and all explanatory text into caption.
Caption: Conventional polynomial-background fits at GP-selected masses. 78 MeV: polynomial degree 3, total width 8 resolution widths, r=-1.350, deviance/dof=29.16/24; 65 MeV: polynomial degree 3, total width 8 resolution widths, r=+1.960, deviance/dof=17.36/22. Counts and predictions use the saved whole-bin grouping, divided by each actual bin width; the last partial group is retained. Counting errors only. Residuals subtract the null-background fit. Lines connect stored bin-averaged model predictions. The quoted local fit statistics do not incorporate the selection of these masses from the GP scan.
Source: `study_results/v4p9p16_candidate_removal_20260906/traditional/qa/paper_display_groups.npz` SHA256 `57567e007a1f5ecbd1cc3567b573c3e1f7ac0557c4fb034d053391cfa8036e66`
Source: `study_results/v4p9p16_candidate_removal_20260906/traditional/derived/fit_summary.csv` SHA256 `e1bcc03bd8819e3a1db23377f0c61e8cbd12840d040e0dbea7b5e38bc528f9da`

## v5_individual_tail_probabilities
Requested: v4p9p12 tail Figures 5--6
Change: Clean titles; retained honest zero-count censoring. Additional tail toys deferred.
Caption: Saved targeted tail refinement: 300 toys at unrefined points and independent 3000 or 10000 fresh toys at selected coordinates. Open downward triangles lie exactly at the one-sided 95% Monte Carlo upper bounds for zero counts (twice that bound for the two-sided diagnostic). Curves join only finite point estimates. Nominal local asymptotic p0 is displayed separately. No new tail samples were generated, so unresolved probabilities remain unresolved.
Source: `study_results/v4p9p12_targeted_tail_refinement_20260905/derived/pvalue_diagnostics_refined.csv` SHA256 `3212327f157f8514ce393173833d76f4c004a08fa372bf8e37dd9faef1643bfa`

## v5_combination_tail_probabilities
Requested: v4p9p12 tail Figures 5--6
Change: Clean titles; retained honest zero-count censoring. Additional tail toys deferred.
Caption: Saved targeted tail refinement: 300 toys at unrefined points and independent 3000 or 10000 fresh toys at selected coordinates. Open downward triangles lie exactly at the one-sided 95% Monte Carlo upper bounds for zero counts (twice that bound for the two-sided diagnostic). Curves join only finite point estimates. Nominal local asymptotic p0 is displayed separately. No new tail samples were generated, so unresolved probabilities remain unresolved.
Source: `study_results/v4p9p12_targeted_tail_refinement_20260905/derived/pvalue_diagnostics_refined.csv` SHA256 `3212327f157f8514ce393173833d76f4c004a08fa372bf8e37dd9faef1643bfa`

## v5_global_probabilities_2015
Requested: v4p9p14/v4p9p15 global-probability series
Change: Regenerated only profiled method; no added scans, no continuation of unresolved tails.
Caption: Profiled-background local and global comparisons under the archived common stress background. The upper panel makes the observed fit and the reference offset visible. Lower curves distinguish nominal local asymptotic p0, the stress-centered local Gaussian approximation, and global tails estimated with 200000 GP fields and 1000 direct full-spectrum scans. Bars are central 95% Monte Carlo intervals; open downward triangles lie exactly at the one-sided 95% upper bounds for zero simulated tails. Filled downward triangles at 1e-8 mark analytic probabilities below the plotting range. Stress-centered tails assess that specified generating spectrum and are not adopted as physical discovery claims. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p14_interpretation_global_20260906/global/2015/analysis/pvalue_curves.csv` SHA256 `93fdccd04e683d2cefd34a244a1a0ae411777db542e61d1df35490c73a233d24`

## v5_global_probabilities_2016
Requested: v4p9p14/v4p9p15 global-probability series
Change: Regenerated only profiled method; no added scans, no continuation of unresolved tails.
Caption: Profiled-background local and global comparisons under the archived common stress background. The upper panel makes the observed fit and the reference offset visible. Lower curves distinguish nominal local asymptotic p0, the stress-centered local Gaussian approximation, and global tails estimated with 200000 GP fields and 1000 direct full-spectrum scans. Bars are central 95% Monte Carlo intervals; open downward triangles lie exactly at the one-sided 95% upper bounds for zero simulated tails. Filled downward triangles at 1e-8 mark analytic probabilities below the plotting range. Stress-centered tails assess that specified generating spectrum and are not adopted as physical discovery claims. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016/analysis/pvalue_curves.csv` SHA256 `72dfa233c7d4f5bf1fa36f8950e1a2f2ef381ce602139cc446f28ada53a5daf6`

## v5_global_probabilities_2021
Requested: v4p9p14/v4p9p15 global-probability series
Change: Regenerated only profiled method; no added scans, no continuation of unresolved tails.
Caption: Profiled-background local and global comparisons under the archived common stress background. The upper panel makes the observed fit and the reference offset visible. Lower curves distinguish nominal local asymptotic p0, the stress-centered local Gaussian approximation, and global tails estimated with 200000 GP fields and 1000 direct full-spectrum scans. Bars are central 95% Monte Carlo intervals; open downward triangles lie exactly at the one-sided 95% upper bounds for zero simulated tails. Filled downward triangles at 1e-8 mark analytic probabilities below the plotting range. Stress-centered tails assess that specified generating spectrum and are not adopted as physical discovery claims. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021/analysis/pvalue_curves.csv` SHA256 `0c2bfa091fa1d50c6858695ef1750999b83e0f319bb88c577619ad2c557d1bca`

## v5_global_probabilities_combined
Requested: v4p9p14/v4p9p15 global-probability series
Change: Regenerated only profiled method; no added scans, no continuation of unresolved tails.
Caption: Profiled-background local and global comparisons under the archived common stress background. The upper panel makes the observed fit and the reference offset visible. Lower curves distinguish nominal local asymptotic p0, the stress-centered local Gaussian approximation, and global tails estimated with 200000 GP fields and 1000 direct full-spectrum scans. Bars are central 95% Monte Carlo intervals; open downward triangles lie exactly at the one-sided 95% upper bounds for zero simulated tails. Filled downward triangles at 1e-8 mark analytic probabilities below the plotting range. Stress-centered tails assess that specified generating spectrum and are not adopted as physical discovery claims. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_combined_global_20260906/global/analysis/pvalue_curves.csv` SHA256 `3ca3a2095a8d32caa3758284c4080247f4889e68a43123dae69e636b3589abdd`

## v5_combined_deficit_scan
Requested: Related joint-deficit appendix
Change: Regenerated only the source plotting block from frozen summaries; no note writes, fits or toys.
Caption: Combined profiled deficit scan. Top: observed signed local fit response and the stress-background offset. Middle: nominal Gaussian and stress-centered local deficit tails; both assign one to nonnegative raw fits. Bottom: two separate global deficit orderings. Filled downward triangles at the local floor indicate analytic values below 1e-8; open downward global triangles lie exactly at the one-sided 95% zero-count Monte Carlo bounds. Direct-scan bars are central 95% intervals. These direction-specific tests were investigated after the excess scan; no extra direction-trials correction is included. 2015 full + 2016 full + 2021 10%  |  19–250 MeV  |  Profiled likelihood Conditional stress background; direction-specific tests after the excess scan. Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.
Source: `study_results/v4p9p16_deficit_extension_20260906/make_report.py` SHA256 `bcbacab72a1fdf7ecf224dfa7dbd63825ec7bd166610f3b0b575fc88e43d790d`
Source: `study_results/v4p9p16_deficit_extension_20260906/analysis/deficit_curves.csv` SHA256 `e26350424db71a5b1bbd4ac7d8a212fd15e378664832d4290f3934fa97e2479e`
Source: `study_results/v4p9p16_deficit_extension_20260906/analysis/summary.json` SHA256 `c790de9497eca2f2f9198753ec591426d5d164e99f19df38446dc7aea824e238`

## v5_dataset_mass_distributions_log
Requested: Current dataset overview replacing historical40--300MeV 2021 support shading
Change: Regenerated only from SHA-verified frozen released ROOT histograms; corrected 2021 support shading to36--300MeV.
Caption: Released invariant-mass histograms for full 2015, full 2016 and the released 2021 10% sample. Each ROOT input SHA256 and histogram key matches the frozen v4.9.12 run ledger before plotting. Curves retain every original histogram bin (0.05, 0.05 and 0.125 MeV, respectively), displayed as counts divided by the native bin width on logarithmic axes. Gray shading shows GP training supports 14--135, 30--210 and 36--300 MeV; dashed boundaries show searches 19--90, 39--180 and 50--250 MeV. Native input binning here precedes the analysis rebinning; no exposure scaling, fitting or new-data access is introduced.
Source: `study_results/v4p9p12_final_dataset_combinations_20260902/inputs/analysis_card.yaml` SHA256 `4a3f3365584743ac1f8b62515ec37c9ca1b908968cfa7497a22609dedbfb79df`
Source: `study_results/v4p9p12_final_dataset_combinations_20260902/derived/run_summary.json` SHA256 `e2b3552bea7bca57fe1806a38aac2c351d54ce5ee16149724cd1e557120d5fc1`
Source: `/Users/emryspeets/research_plots/2015_data/invariant_mass_0pt5mm_full.root` SHA256 `58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f`
Source: `/Users/emryspeets/root_files/EventSelection_pass4Full.root` SHA256 `c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301`
Source: `/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root` SHA256 `3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4`
