# Recent HPS GPR study log

Snapshot: 27 September 2026. Primary scope: 13-27 September; older dependencies and the previously pushed v5.0.5 note are retained. This is a saved-result publication and reading guide.

Start with the [plotbook PDF](../../output/pdf/study_logbook_20260927/HPS_GPR_Recent_Studies_Plot_Logbook.pdf), [illustrated plot guide](PLOTBOOK.md), [searchable local gallery](index.html), or [presentation notes](PRESENTATION_NOTES.md). The catalogue JSON is the machine-readable index.

Study versions can change the template, mask, source and scan ordering. Read each result under its own definition. The v6.3.6 note incorporates several later appendices; these are not additional independent replications.

## Context

### World contours and exposure projections

`v5p0p4_figure2_publication_20260913`

**Question.** How can the current samples be placed in historical reach context?

**Accomplished.** Rebuilt Figure 2 with matched three- and four-campaign observed-equivalent projections, including the 2019 1% high-psum sample.

**Interpretation.** The historical contour survey is incomplete for 2026; exposure scaling and borrowed 2019 response inputs do not establish future sensitivity.

[Full package](../../study_results/v5p0p4_figure2_publication_20260913) | [Definitions / log](../../study_results/v5p0p4_figure2_publication_20260913/README.md) | [Report PDF](../../study_results/v5p0p4_figure2_publication_20260913/pdf/HPS_GPR_Analysis_Note_v5p0p4_Figure2_Revision.pdf) | [P01](PLOTBOOK.md#p01)

### Consolidated analysis note v5.0.5

`v5p0p5_analysis_note_20260916`

**Question.** Where is the long-form baseline procedure and its earlier study history?

**Accomplished.** Integrated geometry, exposure/echo studies, local results, and a bounded binning check into the 252-page note. The September language revision preserved the numerical results.

**Interpretation.** This baseline predates the new v6 template choices; later conditional studies do not silently replace its nominal results.

[Full package](../../study_results/v5p0p5_analysis_note_20260916) | [Definitions / log](../../study_results/v5p0p5_analysis_note_20260916/README.md) | [Report PDF](../../study_results/v5p0p5_analysis_note_20260916/pdf/HPS_GPR_Analysis_Note_v5p0p5.pdf)

### Common coupling and alternative rate models

`v5p5p4_combination_rate_bands_20260913`

**Question.** How much does the assumed relationship between campaign rates matter?

**Accomplished.** Compared shared coupling, independent nonnegative amplitudes, power-law and exponential rate descriptions at 92 MeV, with parameter covariance bands.

**Interpretation.** These are different pointwise tests. Selecting a model or mass requires additional calibration; relaxed rates do not define a unique combined coupling limit.

[Full package](../../study_results/v5p5p4_combination_rate_bands_20260913) | [Definitions / log](../../study_results/v5p5p4_combination_rate_bands_20260913/README.md) | [Report PDF](../../study_results/v5p5p4_combination_rate_bands_20260913/pdf/HPS_GPR_v5p5p4_Combination_Rate_Uncertainties.pdf)

### Projected upper limits and GP echoes

`v5p6p1_2021_upper_limit_echoes_20260913`

**Question.** What structures could an injected excess produce in future limit scans?

**Accomplished.** Reused 15 scenarios and 300 Poisson spectra to map conditional upper-limit depressions and neighboring GP response features.

**Interpretation.** These are source-dependent forecasts, not observed full-2021 data or guaranteed future structures.

[Full package](../../study_results/v5p6p1_2021_upper_limit_echoes_20260913) | [Definitions / log](../../study_results/v5p6p1_2021_upper_limit_echoes_20260913/README.md) | [Report PDF](../../study_results/v5p6p1_2021_upper_limit_echoes_20260913/report.pdf)

### Angular acceptance illustration

`v5p7p0_hps_acceptance_20260913`

**Question.** Why does the characteristic low-mass scale rise with beam energy?

**Accomplished.** Derived the boosted two-body angular model and separated the 15 mrad inner-gap scale from an illustrative outer angular cap.

**Interpretation.** The 70 mrad cap is illustrative, and a selected-spectrum overlay is not a detector-efficiency measurement.

[Full package](../../study_results/v5p7p0_hps_acceptance_20260913) | [Definitions / log](../../study_results/v5p7p0_hps_acceptance_20260913/README.md)

### Geometry and magnetic transport

`v5p7p1_geometry_acceptance_20260913`

**Question.** What changes when tracks are propagated through the actual field and sensor geometry?

**Accomplished.** Replaced the outer-angle surrogate with pinned field maps, sensor rectangles, and campaign-specific hit requirements; supplied reusable geometry and response plates.

**Interpretation.** Conditional decay and hit fractions omit the full production distribution, reconstruction/selection efficiency, and detector systematics.

[Full package](../../study_results/v5p7p1_geometry_acceptance_20260913) | [Definitions / log](../../study_results/v5p7p1_geometry_acceptance_20260913/README.md) | [Report PDF](../../study_results/v5p7p1_geometry_acceptance_20260913/pdf/HPS_v5p7p1_Geometry_Acceptance_Study.pdf) | [P02](PLOTBOOK.md#p02)

### APEX contour comparison

`apex_initial_studies`

**Question.** Do the supplied APEX curves align with the HPS local structures?

**Accomplished.** Digitized the supplied screenshots and compared the 155-250 MeV overlap. No common integer mass has both supplied local probabilities below 0.1.

**Interpretation.** The APEX source, confidence level and resolution-axis units remain unresolved; contour crossings and nearby minima are not combined significance or exclusion.

[Full package](../../apex_initial_studies) | [Definitions / log](../../apex_initial_studies/README.md) | [Report PDF](../../apex_initial_studies/output/pdf/APEX_Initial_Studies.pdf)

## Significance

### Local mapping, stress response and spacing

`v5p8p0_local_significance_mapping_20260917`

**Question.** Which effects change an apparent local excess?

**Accomplished.** Separated grid spacing, binning, window width, source offsets, exposure scaling, and reference-standardized response; retained failed and historical-subset checks.

**Interpretation.** A stress-centered coordinate is not a calibrated physical significance; legacy full-fit dependencies are documented.

[Full package](../../study_results/v5p8p0_local_significance_mapping_20260917) | [Definitions / log](../../study_results/v5p8p0_local_significance_mapping_20260917/README.md) | [Report PDF](../../study_results/v5p8p0_local_significance_mapping_20260917/source/report.pdf)

### Alternative background sources

`v5p8p1_background_truths_20260917`

**Question.** Can a different source remove deterministic bias without hiding signal?

**Accomplished.** Compared frozen 2016 background constructions and showed that lower stress bias can coexist with source-building signal absorption; unsuccessful sources remain recorded.

**Interpretation.** Sources estimated from the observed data are not independent controls; incomplete or failed source ensembles do not receive tail probabilities.

[Full package](../../study_results/v5p8p1_background_truths_20260917) | [Definitions / log](../../study_results/v5p8p1_background_truths_20260917/README.md) | [Report PDF](../../study_results/v5p8p1_background_truths_20260917/source/report.pdf)

### Coherent nominal-GP significance fields

`v5p8p2_nominal_gp_significance_20260917`

**Question.** How can the scan-wide trial effect preserve correlations?

**Accomplished.** Built conditional response fields, compared 200,000 Gaussian-field draws with 256 complete Poisson spectra per campaign, and separated full-union from common-overlap searches.

**Interpretation.** Reference-local centering/scaling is explicit; source uncertainty and selection among searches are outside this calibration.

[Full package](../../study_results/v5p8p2_nominal_gp_significance_20260917) | [Definitions / log](../../study_results/v5p8p2_nominal_gp_significance_20260917/README.md) | [Report PDF](../../study_results/v5p8p2_nominal_gp_significance_20260917/source/report.pdf)

### Resolution and global-probability explanation

`v5p8p3_global_interpretation_20260917`

**Question.** Why is the number of scan points not the effective number of trials?

**Accomplished.** Explained covariance, threshold-dependent effective counts, upcrossing comparisons, and the distinction between a discovery tail and a 90% CLs limit using saved fields.

**Interpretation.** Resolution-count and upcrossing comparisons do not replace the fitted correlated-scan probability.

[Full package](../../study_results/v5p8p3_global_interpretation_20260917) | [Definitions / log](../../study_results/v5p8p3_global_interpretation_20260917/README.md) | [Report PDF](../../study_results/v5p8p3_global_interpretation_20260917/source/report.pdf)

### Combination methods and window widths

`v5p8p4_independent_combinations_windows_20260918`

**Question.** How do shared coupling, Fisher and Stouffer compare under matched scan choices?

**Accomplished.** Kept each combination rule, width (2.25-2.6 reference sigma), search domain, and reach definition separate while preserving coherent mass correlations.

**Interpretation.** Method/width selection is not globally calibrated; a probability combination without a signal-rate model does not yield a physical combined coupling limit.

[Full package](../../study_results/v5p8p4_independent_combinations_windows_20260918) | [Definitions / log](../../study_results/v5p8p4_independent_combinations_windows_20260918/README.md) | [Report PDF](../../study_results/v5p8p4_independent_combinations_windows_20260918/source/report.pdf)

### Common mass with independent amplitudes

`v5p8p5_mass_coherence_20260919`

**Question.** Does a common mass remain interesting when campaign amplitudes are free?

**Accomplished.** Tested a common-mass statistic with independent positive amplitudes and a separate mass-coherence diagnostic. The nominal full-domain direct global count is 7/256 (add-one p=0.03113).

**Interpretation.** This is a different hypothesis from common coupling. The 80-105 MeV diagnostic is post hoc; zero local exceedances are finite-sample bounds.

[Full package](../../study_results/v5p8p5_mass_coherence_20260919) | [Definitions / log](../../study_results/v5p8p5_mass_coherence_20260919/README.md) | [Report PDF](../../study_results/v5p8p5_mass_coherence_20260919/source/report.pdf)

### Consolidated calibration review

`v5p8p5_consolidated_fairness_20260921`

**Question.** Which probability comparisons answer the same statistical question?

**Accomplished.** Documented raw versus reference-standardized ordering, same-field upcrossing comparisons, failures of a simple correlation-cut rule, and limitations of historical-source transfer.

**Interpretation.** Different orderings are not choices to optimize after seeing the data; no new physical-significance calibration or width choice is certified.

[Full package](../../study_results/v5p8p5_consolidated_fairness_20260921) | [Definitions / log](../../study_results/v5p8p5_consolidated_fairness_20260921/README.md) | [Report PDF](../../study_results/v5p8p5_consolidated_fairness_20260921/source/report.pdf)

### Saved-result statistical perspective

`v5p8p5_external_statistician_20260921`

**Question.** What follow-up would most clearly distinguish rate consistency from chance structure?

**Accomplished.** Organized the existing 90-93 MeV and 65-67 MeV evidence around replication, common-rate consistency and additional 2021 data.

**Interpretation.** This is a statistical reading of saved results, not an independent experimental certification or a new likelihood fit.

[Full package](../../study_results/v5p8p5_external_statistician_20260921) | [Definitions / log](../../study_results/v5p8p5_external_statistician_20260921/README.md) | [Report PDF](../../study_results/v5p8p5_external_statistician_20260921/source/report.pdf)

### Fixed background inside the likelihood

`v5p8p5_fixed_background_significance_20260921`

**Question.** What happens when the predicted background is held fixed during signal extraction?

**Accomplished.** Added C=0 likelihood fits and 256 complete conditional toy scans, while retaining count-dependent GP sideband conditioning for every spectrum.

**Interpretation.** A fixed generating source and a fixed likelihood background are different operations. Small asymptotic p-values require comparison with direct complete-toy tails.

[Full package](../../study_results/v5p8p5_fixed_background_significance_20260921) | [Definitions / log](../../study_results/v5p8p5_fixed_background_significance_20260921/README.md) | [Report PDF](../../study_results/v5p8p5_fixed_background_significance_20260921/source/report.pdf)

### Unshifted local and raw-maximum global scans

`v5p8p5p3_raw_significance_20260921`

**Question.** What do the conventional raw local curves say before reference centering?

**Accomplished.** Reproduced 1,310 archived coordinates with Z=max(r,0). Local peaks include 2016: 90.5 MeV, Z=3.4525; shared coupling: 66 MeV, Z=2.7602.

**Interpretation.** These are local asymptotic values. The raw-maximum global calculation has a frozen-source scope and differs from a reference-standardized ordering.

[Full package](../../study_results/v5p8p5p3_raw_significance_20260921) | [Definitions / log](../../study_results/v5p8p5p3_raw_significance_20260921/README.md) | [Report PDF](../../study_results/v5p8p5p3_raw_significance_20260921/source/report.pdf) | [P03](PLOTBOOK.md#p03) | [P04](PLOTBOOK.md#p04)

### Fixed-mass offset versus positive scan maximum

`v5p9p5_null_bias_20260922`

**Question.** Is a positive null maximum evidence of estimator bias?

**Accomplished.** Separated the expected positive scan maximum from a genuine fixed-mass signed-root offset. At 78 MeV the toy mean is 0.548; the source-conditional full-scan tail is 0.25092.

**Interpretation.** This is conditional on the fitted generating source. It does not establish background adequacy, unconditional coverage, or discovery calibration.

[Full package](../../study_results/v5p9p5_null_bias_20260922) | [Definitions / log](../../study_results/v5p9p5_null_bias_20260922/README.md) | [P05](PLOTBOOK.md#p05)

## Templates

### Controlled Gaussian-tail perturbations

`v5p9_tail_structure_20260921`

**Question.** How much does a modest tail change affect matched-window observed limits?

**Accomplished.** Preserved the central Gaussian shape and compared exact-bin-integrated tail alternatives with fresh matching Gaussian references across all three campaigns.

**Interpretation.** The enlarged-window comparison also changes the GP sidebands. Small effects for this tail family do not imply robustness to broad native MC shapes.

[Full package](../../study_results/v5p9_tail_structure_20260921) | [Definitions / log](../../study_results/v5p9_tail_structure_20260921/README.md) | [Report PDF](../../study_results/v5p9_tail_structure_20260921/source/report.pdf) | [P08](PLOTBOOK.md#p08)

### Native 2021 MC shapes and windows

`v6p1_mc_signal_templates_20260922`

**Question.** How do the supplied reconstructed MC distributions differ from Gaussian signals?

**Accomplished.** Mapped broad tails, displaced cores, native templates, neighboring-template interpolation, and MC-centered window comparisons; preserved the production/selection audit.

**Interpretation.** The all-selected MC is not established as a pure resonance response. Selection equivalence and physical coupling normalization remain qualified.

[Full package](../../study_results/v6p1_mc_signal_templates_20260922) | [Definitions / log](../../study_results/v6p1_mc_signal_templates_20260922/README.md) | [Report PDF](../../study_results/v6p1_mc_signal_templates_20260922/pdf/HPS_GPR_v6p1_MC_Signal_Templates.pdf) | [P09](PLOTBOOK.md#p09)

### 2016 already-smeared MC catalogue

`v6p4_2016_mc_shapes_20260925`

**Question.** Does 2016 need the same signal-shift model as 2021?

**Accomplished.** Characterized 29 supplied histograms. Qualified 40-175 MeV cores shift down by 0.08-1.02 MeV; 89.2-93.7% lies within two fitted core widths. Neighbor interpolation has smaller held-out CDF discrepancies than common/Gaussian shapes.

**Interpretation.** 30 and 35 MeV are diagnostic-only; 150 MeV is missing. Use the supplied FEE-smeared/scaled histogram without extra smearing; shape agreement is not efficiency validation.

[Full package](../../study_results/v6p4_2016_mc_shapes_20260925) | [Definitions / log](../../study_results/v6p4_2016_mc_shapes_20260925/README.md) | [Report PDF](../../study_results/v6p4_2016_mc_shapes_20260925/pdf/HPS_GPR_v6p4_2016_MC_Shapes.pdf) | [P20](PLOTBOOK.md#p20) | [P21](PLOTBOOK.md#p21)

## Recovery

### Exact-N MC injection and recovery

`v6p2_mc_injection_20260923`

**Question.** Does moving the extraction window to the MC core recover the full signal?

**Accomplished.** Final 40-toy release: 1,760 injected spectra and 3,520 primary fits. At N=30,000, paired responses span 71.5-89.5% (pole) and 78.0-93.8% (shifted).

**Interpretation.** Raw recovery and paired response differ. The 260 MeV extension uses a 250 MeV kernel anchor; finite conditional containment is not coverage certification.

[Full package](../../study_results/v6p2_mc_injection_20260923) | [Definitions / log](../../study_results/v6p2_mc_injection_20260923/README.md) | [Report PDF](../../study_results/v6p2_mc_injection_20260923/pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf) | [P10](PLOTBOOK.md#p10) | [P11](PLOTBOOK.md#p11)

### Fixed expected yield and reference matching

`v6p3_injection_design_20260923`

**Question.** What should be held fixed in an injection performance test?

**Accomplished.** Distinguished per-toy reference-error matching, fixed expected yield, and exact-N injections; audited the saved baseline and defined expected-yield pulls.

**Interpretation.** This is a design/audit study, not a new HPS toy ensemble or proof that matching corrects bias.

[Full package](../../study_results/v6p3_injection_design_20260923) | [Definitions / log](../../study_results/v6p3_injection_design_20260923/README.md) | [Report PDF](../../study_results/v6p3_injection_design_20260923/pdf/HPS_GPR_v6p3_Injection_Design.pdf)

### 100-pilot / 100-evaluation 2021 study

`v6p3p1_fixed_yield_2021_100toy_20260924`

**Question.** How do Gaussian and native-MC responses compare at the same expected yield?

**Accomplished.** Completed 2,000 pilot and 8,000 evaluation fits with no failed rows. At z=5, paired response is 0.961-0.978 for Gaussian and 0.716-0.894 for native MC.

**Interpretation.** Expected signal is fixed by an independent Gaussian pilot; realized Poisson totals fluctuate. The source, kernels and templates are fixed.

[Full package](../../study_results/v6p3p1_fixed_yield_2021_100toy_20260924) | [Definitions / log](../../study_results/v6p3p1_fixed_yield_2021_100toy_20260924/README.md) | [Report PDF](../../study_results/v6p3p1_fixed_yield_2021_100toy_20260924/pdf/HPS_GPR_v6p3p1_Fixed_Yield_2021_100toy.pdf) | [P12](PLOTBOOK.md#p12) | [P13](PLOTBOOK.md#p13)

### 2016 exposure and calibration transfer

`v6p3p2_2016_offset_transfer_20260924`

**Question.** Can a 10% offset simply be scaled to the full exposure?

**Accomplished.** Completed independent pilot/calibration/evaluation cohorts and found nontrivial failures of offset scaling and nominal-to-stress transfer; same-source affine corrections reduce mean pulls.

**Interpretation.** A fitted yield in one observed sample is not an ensemble bias. Offset subtraction cannot restore response or establish transferable coverage.

[Full package](../../study_results/v6p3p2_2016_offset_transfer_20260924) | [Definitions / log](../../study_results/v6p3p2_2016_offset_transfer_20260924/README.md) | [Report PDF](../../study_results/v6p3p2_2016_offset_transfer_20260924/pdf/HPS_GPR_v6p3p2_2021_Study_with_2016_Offset_Appendix.pdf)

### Unified 2021 procedure

`v6p3p5_unified_2021_20260924`

**Question.** How do core shifting, response, inference and observed scans fit together?

**Accomplished.** Connected null calibration, MC-generated/Gaussian-extracted response, finite-grid inference and observed combinations. Retained the primary matched +/-2.25 reference-sigma mask without an added guard.

**Interpretation.** Core shifts do not produce unit response. Empty, zero-only or censored accepted grids are not precise continuous physical upper limits.

[Full package](../../study_results/v6p3p5_unified_2021_20260924) | [Definitions / log](../../study_results/v6p3p5_unified_2021_20260924/README.md) | [Report PDF](../../study_results/v6p3p5_unified_2021_20260924/pdf/HPS_GPR_v6p3p5_Unified_2021_Procedure.pdf)

### Standalone 2021 reader's note

`v6p3p6_readability_2021_20260924`

**Question.** Where can the complete 2021 narrative and follow-ups be read in one place?

**Accomplished.** Created the 56-page integrated note: readable main procedure, v16 TC/UC shapes, template/window tests, leakage controls, observed morph fits and combined scans.

**Interpretation.** Many appendices reuse earlier results. Do not count them as independent replications; uncertainty methods and normalization qualifications are retained.

[Full package](../../study_results/v6p3p6_readability_2021_20260924) | [Definitions / log](../../study_results/v6p3p6_readability_2021_20260924/README.md) | [Report PDF](../../study_results/v6p3p6_readability_2021_20260924/pdf/HPS_GPR_v6p3p6_2021_Signal_Extraction.pdf) | [P14](PLOTBOOK.md#p14) | [P15](PLOTBOOK.md#p15)

### v16 templates and fit-window study

`v6p3p7_template_windows_20260925`

**Question.** Which full-signal template/window combinations improve extraction?

**Accomplished.** Compared shifted Gaussian, common aligned, neighboring-MC and direct-MC templates using tuning then independent calibration/evaluation. Common starter won the frozen tuning score; the morph is a separate candidate.

**Interpretation.** Improved response is not a universal limit improvement. Starter windows leave training contamination, and finite-grid rejection/empty-set accounting matters.

[Full package](../../study_results/v6p3p7_template_windows_20260925) | [Definitions / log](../../study_results/v6p3p7_template_windows_20260925/README.md) | [Report PDF](../../study_results/v6p3p7_template_windows_20260925/pdf/HPS_GPR_v6p3p7_Template_Window_Study.pdf)

### Controlled training-contamination test

`v6p3p7_window_leakage_followup_20260925`

**Question.** How much of the response loss is caused by signal in the GP sidebands?

**Accomplished.** At 100/160/220 MeV, removing only training-bin signal changes direct-MC response from 0.959/0.950/0.909 to 1.001/0.997/1.000.

**Interpretation.** This diagnostic uses the same saved draws and a control unavailable in real data. Leakage fraction and lost fitted yield are distinct quantities.

[Full package](../../study_results/v6p3p7_window_leakage_followup_20260925) | [Definitions / log](../../study_results/v6p3p7_window_leakage_followup_20260925/README.md) | [P16](PLOTBOOK.md#p16) | [P17](PLOTBOOK.md#p17)

### 2016 narrower-window comparison

`v6p4p2_2016_window_comparison_20260925`

**Question.** Does a lower observed upper limit imply better signal sensitivity?

**Accomplished.** Changing +/-3.5u to +/-2u lowers the median 2016 observed limit by 9.64%, but paired response falls to 0.877-0.922 from 0.998-1.001. Response-adjusted spread ratios are 0.998-1.030.

**Interpretation.** A five-mass, one-strength conditional test does not establish a universal sensitivity gain or coverage; the later +/-2.5u global study is a distinct fixed choice.

[Full package](../../study_results/v6p4p2_2016_window_comparison_20260925) | [Definitions / log](../../study_results/v6p4p2_2016_window_comparison_20260925/README.md) | [Report PDF](../../study_results/v6p4p2_2016_window_comparison_20260925/pdf/report.pdf) | [P22](PLOTBOOK.md#p22)

## Observed

### Observed 2021 morphed-template extraction

`v6p3p8_observed_morph_20260925`

**Question.** How do the new templates describe the observed local regions?

**Accomplished.** Completed 543 observed fits and selected-region toy checks at 67, 79, 185 and 226 MeV; provided detailed count, residual and standardized-residual panels.

**Interpretation.** Masses were selected from the scan. Local checks are not global probabilities; the deficit's empty accepted set is not a zero physical limit.

[Full package](../../study_results/v6p3p8_observed_morph_20260925) | [Definitions / log](../../study_results/v6p3p8_observed_morph_20260925/README.md) | [Report PDF](../../study_results/v6p3p8_observed_morph_20260925/pdf/HPS_GPR_v6p3p8_Observed_Morph_Extraction.pdf) | [P18](PLOTBOOK.md#p18)

### Joint scan with 2021 MC only

`v6p3p9_combined_morph_20260925`

**Question.** What changes in the common-coupling fit when only the 2021 template is updated?

**Accomplished.** The combined local minimum moves to 68 MeV with asymptotic Z=2.910 and p=0.001806; older campaigns remain unchanged.

**Interpretation.** Zero of 1,000 fixed-mass toy exceedances gives a finite rank floor, not zero probability. This release has no global calibration.

[Full package](../../study_results/v6p3p9_combined_morph_20260925) | [Definitions / log](../../study_results/v6p3p9_combined_morph_20260925/README.md) | [Report PDF](../../study_results/v6p3p9_combined_morph_20260925/pdf/HPS_GPR_v6p3p9_Combined_Morph_Scan.pdf) | [P19](PLOTBOOK.md#p19)

### 2016 and joint MC extraction

`v6p4p1_shifted_mc_extraction_20260925`

**Question.** What changes when 2016 also uses its native MC response?

**Accomplished.** Added 2016-only and common-coupling MC scans with full normalization. At 91 MeV, local asymptotic p=0.000196 differs from the 3/1,000 conditional null exceedances.

**Interpretation.** This report uses the wider 2016 +/-3.5 core-width window; it must not be mixed with later +/-2.5 global-calibration definitions.

[Full package](../../study_results/v6p4p1_shifted_mc_extraction_20260925) | [Definitions / log](../../study_results/v6p4p1_shifted_mc_extraction_20260925/README.md) | [Report PDF](../../study_results/v6p4p1_shifted_mc_extraction_20260925/pdf/report.pdf)

## Global

### Complete-scan raw-statistic calibration

`v6p4p3_global_mc_20260925`

**Question.** How often does a whole background scan reach the observed raw maximum?

**Accomplished.** Ran 1,024 coherent scans with 2016 +/-2.5u, 2021 [-4,+3]u and unchanged 2015 Gaussian. Direct raw-maximum p values are 0.0663, 0.2341 and 0.1980 for 2016, 2021 and common coupling.

**Interpretation.** Finite-grid, fixed-source calibration omits source uncertainty and earlier method/window selection. A separate three-search family test is supplementary.

[Full package](../../study_results/v6p4p3_global_mc_20260925) | [Definitions / log](../../study_results/v6p4p3_global_mc_20260925/README.md) | [Report PDF](../../study_results/v6p4p3_global_mc_20260925/pdf/report.pdf)

### Raw versus local-calibrated ordering audit

`v6p4p3_global_method_review_20260925`

**Question.** Why can the selected combined mass depend on scan ordering?

**Accomplished.** Reused existing arrays to show that the combined raw maximum is at 67 MeV while the minimum local rank is at 68 MeV; no new fits were run.

**Interpretation.** An audit of the original ensemble is not independent validation. It motivated the separated A/B procedure in v6.4.4.

[Full package](../../study_results/v6p4p3_global_method_review_20260925) | [Definitions / log](../../study_results/v6p4p3_global_method_review_20260925/method_audit.json)

### Independent local-to-global calibration

`v6p4p4_calibrated_local_global_20260925`

**Question.** What is the scan-wide probability after allowing for mass-dependent local behavior?

**Accomplished.** A fixes the local maps (1,024 scans); independent B validates minimum mapped p (1,024 scans). Combined: 75/1,024, add-one p=0.07415, 95% interval [0.05804,0.09095].

**Interpretation.** The joint fit has one common coupling. This is not the maximum across three searches. The combined local rank hits 1/1025; B ties count inclusively. Source/map uncertainty and prior selection remain outside scope.

[Full package](../../study_results/v6p4p4_calibrated_local_global_20260925) | [Definitions / log](../../study_results/v6p4p4_calibrated_local_global_20260925/README.md) | [Report PDF](../../study_results/v6p4p4_calibrated_local_global_20260925/pdf/report.pdf) | [P23](PLOTBOOK.md#p23) | [P24](PLOTBOOK.md#p24)

### Mass correlation and the look-elsewhere effect

`v6p4p5_mass_correlation_lee_20260926`

**Question.** How much do overlapping mass hypotheses reduce the trial penalty?

**Accomplished.** Reused unchanged A/B scans. Combined direct p=0.074 versus an empirical independent-mass control of 0.133; simple single-year resolution counts understate the observed thresholds' penalty.

**Interpretation.** Correlation bands describe variation across mass pairs, not confidence intervals. The independence control is counterfactual; no new probability or limit replaces v6.4.4.

[Full package](../../study_results/v6p4p5_mass_correlation_lee_20260926) | [Definitions / log](../../study_results/v6p4p5_mass_correlation_lee_20260926/README.md) | [Report PDF](../../study_results/v6p4p5_mass_correlation_lee_20260926/pdf/report.pdf) | [P25](PLOTBOOK.md#p25) | [P26](PLOTBOOK.md#p26)

## Presentation

### Initial presentation revision

`unblind_meeting_RCmeet_20260922`

**Question.** Which presentation material is ready to revisit and explain?

**Accomplished.** Updated the 15 instructed slides in a 79-slide snapshot; added kernel, extraction, validation, CLs, exposure and global-calibration explanations.

**Interpretation.** This is a dated saved snapshot. Consult the next two revisions for later slide numbers and refinements.

[Full package](../../output/slides/unblind_meeting_RCmeet_20260922) | [Definitions / log](../../output/slides/unblind_meeting_RCmeet_20260922/CHANGELOG.md) | [Report PDF](../../output/slides/unblind_meeting_RCmeet_20260922/renders/delivered/presentation.pdf)

### Presentation validation and speaker scripts

`unblind_meeting_RCmeet_20260923`

**Question.** Which presentation material is ready to revisit and explain?

**Accomplished.** Added two validation slides (85 total), revised eight instructed slides, and saved concise/extended spoken explanations and selected-mass sideband checks.

**Interpretation.** The fitted-sideband check does not validate prediction in the excluded signal window; the 78 MeV example was already selected.

[Full package](../../output/slides/unblind_meeting_RCmeet_20260923) | [Definitions / log](../../output/slides/unblind_meeting_RCmeet_20260923/CHANGELOG.md) | [Report PDF](../../output/slides/unblind_meeting_RCmeet_20260923/renders/delivered/presentation.pdf)

### Latest saved presentation follow-up

`unblind_meeting_RCmeet_20260923b`

**Question.** Which presentation material is ready to revisit and explain?

**Accomplished.** Updated slides 13, 14 and 29. All 42 observed sideband-deviance values lie inside their pointwise 90% conditional bands; saved residual guidance, BEST conversion explanation and pixel-preservation evidence.

**Interpretation.** Neighboring centers reuse the same spectra, so pointwise agreement is not a global goodness-of-fit or predictive-coverage result. The saved deck is a snapshot, not a live-deck verification.

[Full package](../../output/slides/unblind_meeting_RCmeet_20260923b) | [Definitions / log](../../output/slides/unblind_meeting_RCmeet_20260923b/CHANGELOG.md) | [Report PDF](../../output/slides/unblind_meeting_RCmeet_20260923b/renders/delivered/presentation.pdf) | [P06](PLOTBOOK.md#p06) | [P07](PLOTBOOK.md#p07)
