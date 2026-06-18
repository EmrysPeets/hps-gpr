# HPS GPR Round 1 Review Responses

Source review document: `/Users/emryspeets/Desktop/summer_26/gpr_note/round1_review_comments.pdf`
Original comparison note: `/Users/emryspeets/Desktop/summer_26/gpr_note/hps_gpr_rev_version1.pdf`
Current analysis note source: `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/hps_gpr_analysis_note`
Additional source for 2021 preselection details: `/Users/emryspeets/Desktop/thesis_things/2021_Full_Analysis_Note.pdf`

Each numbered item gives the original comment, the response, and where the new information appears in the note.

1. [G1] Original comment: Add more introduction to Gaussian process regression in Section 4; explain edge handling, one-sided sidebands, and whether the software is custom or package-based.

   Status: Addressed / patched

   Response: Agree. The current note now gives the GP primer, defines the blinded/training-exclusion geometry, explains that bins outside the mass-dependent exclusion form the sideband training sample, and states that the GP fit uses scikit-learn GaussianProcessRegressor with custom HPS code for masks, templates, likelihoods, and CLs. The scan ranges are chosen to avoid unsupported edge behavior, and validation plots check the residual behavior across the scanned mass range.

   Where new information is in the note: sections/04_methodology.tex, Sec. Observable/scan/blinding and Log-space preprocessing; new ScikitLearn citation in hps_gpr_analysis_note.bib.

   Follow-up or recommended phrasing: No further note edit unless the committee wants an even more tutorial GP paragraph in the introduction.

2. [G2] Original comment: This is the first HPS bump hunt combining multiple datasets/run periods/beam energies. Add introduction comments. How do likelihood uncertainties and normalization uncertainties enter?

   Status: Mostly addressed; final-systematics decision still open

   Response: The note now frames this as a shared-coupling HPS GPR combination. The combined fit uses one shared epsilon^2 parameter, with dataset-specific GP backgrounds, signal templates, mass resolutions, radiative fractions, and yield-to-epsilon^2 factors. Normalization systematics enter the coupling conversion, not the GP background interpolation. The current production model does not yet profile common luminosity/radiative-fraction/resolution nuisance parameters across runs; that remains a final-unblinding audit item.

   Where new information is in the note: sections/01_introduction.tex; sections/04_methodology.tex, shared-epsilon^2 combination; sections/03_event_selection.tex, radiative-fraction normalization.

   Follow-up or recommended phrasing: Before final circulation, decide whether these normalization uncertainties are profiled or documented as external validation-stage systematics.

3. [G3] Original comment: Add mass yield and shape comparison for 2015/2016/2021. EP proposed using dataset_summary_figs/invariant_mass_distributions_2015_2016_2021_log.png, changing 2021 expected search region to 35-250 MeV, and adding a normalized overlay before it.

   Status: Patched this pass

   Response: Done. A new unit-normalized overlay was added before the absolute-yield log plot, and the 2021 expected-region highlight in the absolute-yield plot now spans 35-250 MeV. The normalized plot is for shape comparison; the log-y plot keeps the yield-scale context.

   Where new information is in the note: sections/02_datasets.tex, figs. dataset-mass-shape-overlay and dataset-mass-distributions-log; dataset_summary_figs/invariant_mass_distributions_2015_2016_2021_normalized.* and ..._log.*.

   Follow-up or recommended phrasing: Closed after compile check.

4. [G4] Original comment: Preliminary mass resolution, especially 2021; placeholder text should be addressed after the rest.

   Status: Review-stage documentation addressed; final 2021 validation still open

   Response: The note now adds 2015, 2016, and 2021 mass-resolution provenance plus a three-panel summary of the parameterizations used in the GPR workflow. For 2021, the text labels the curve as the current target-constrained V0 implementation input, not as a completed standalone data/MC validation.

   Where new information is in the note: sections/03_event_selection.tex, mass-resolution subsections; resolution_figs/hps_mass_resolution_three_panel.png; resolution_figs/hps2021_mass_resolution_target_constrained_nohitreq.png.

   Follow-up or recommended phrasing: Add a final 2021 data/MC resolution validation or sigma_m-shift systematic before unblinding.

5. [D01] Original comment: Page 4 line 96 and page 5 line 121: is "10 MeV" correct?

   Status: Already addressed

   Response: The current note no longer carries the old 10 MeV wording at those locations; the relevant scan ranges and mass windows are expressed with the current dataset-specific ranges.

   Where new information is in the note: sections/02_datasets.tex and current main06162026.pdf.

   Follow-up or recommended phrasing: Closed.

6. [D02] Original comment: Page 6 line 141: GP not defined; primer needed.

   Status: Already addressed / patched

   Response: Addressed. GP/GPR is introduced in the introduction and methodology, with the background interpolation role explained before the detailed likelihood machinery.

   Where new information is in the note: sections/01_introduction.tex and sections/04_methodology.tex.

   Follow-up or recommended phrasing: Closed unless a longer tutorial paragraph is desired.

7. [D03] Original comment: Page 6 line 142: define the mean vector.

   Status: Addressed in this pass

   Response: Added the requested sentence to the methodology: "The GP mean vector is the vector of latent log-count means evaluated at the selected training-bin coordinate."

   Where new information is in the note: sections/04_methodology.tex, immediately after the latent smooth mean function definition in the log-space preprocessing discussion.

   Follow-up or recommended phrasing: Closed.

8. [D04] Original comment: Page 6 line 171: hodoscope mention? Single-3 triggers?

   Status: Addressed in this pass

   Response: The 2021 trigger/topology row now states that the baseline preselection uses Single-2 or Single-3 positron single-cluster triggers, with Single-3 carrying the matched hodoscope condition from the 2021 trigger menu.

   Where new information is in the note: tables/event_selection_table.tex, Trigger/topology row; checked against 2021_Full_Analysis_Note.pdf Appendix A, Table 63.

   Follow-up or recommended phrasing: Closed unless the final prompt note uses different trigger bookkeeping.

9. [D05] Original comment: Page 6 line 173: "double-layer stereo" ambiguous.

   Status: Patched this pass

   Response: Done. The detector description now says "six axial/stereo tracking stations," which avoids the ambiguous double-layer wording.

   Where new information is in the note: sections/02_datasets.tex.

   Follow-up or recommended phrasing: Closed.

10. [D06] Original comment: Page 7 line 159: details about blinded event selection? EP: 1% 2021 and 10% 2016 subset of runs spread evenly through the run period.

   Status: Patched this pass

   Response: Added the staged-subset statement: the 2016 10% and 2021 1% development streams are described as spread across their respective run periods rather than contiguous blocks. This answers the blinding-selection question at the level of the current note.

   Where new information is in the note: sections/02_datasets.tex, staged-analysis paragraph.

   Follow-up or recommended phrasing: Confirm the exact bookkeeping sentence against the run-list owner before final sign-off.

11. [D07] Original comment: Page 9 Table 1: table not referenced.

   Status: Patched this pass

   Response: Done. The text now explicitly introduces Table 1 before the table appears.

   Where new information is in the note: sections/02_datasets.tex, paragraph before dataset-summary table.

   Follow-up or recommended phrasing: Closed.

12. [D08] Original comment: Page 9 Table 1: EOT relevance/how calculated; targets differ.

   Status: Patched this pass; calculation details still optional

   Response: The note now says EOT is not an input to the GP background interpolation; it enters the dataset-specific signal normalization used to convert prompt yield to epsilon^2, together with target, acceptance, efficiency, and radiative-fraction inputs. The exact campaign-scale EOT calculation can be cited to run-quality/luminosity bookkeeping if needed.

   Where new information is in the note: sections/02_datasets.tex, paragraph after Table 1.

   Follow-up or recommended phrasing: Optional: add a source citation or footnote for the EOT accounting if the collaboration has a canonical number source.

13. [D09] Original comment: Page 9 Table 1: why scan ranges differ between current and published searches.

   Status: Already addressed

   Response: Addressed in the current table caption. It states that the listed scan ranges are the current GPR analysis ranges rather than the smaller published window-search ranges, and it quotes the published 2015 and 2016 ranges.

   Where new information is in the note: sections/02_datasets.tex, Table 1 caption.

   Follow-up or recommended phrasing: Closed.

14. [D10] Original comment: Page 9 line 223: "line-by-line bookkeeping..." why not?

   Status: Addressed in this pass

   Response: The caveat was replaced by a source-backed statement: the 2021 baseline entries are taken from the full 2021 analysis note, while the GPR-specific prompt-tight addition is the p_sum < 2.8 GeV production-histogram requirement.

   Where new information is in the note: sections/03_event_selection.tex, Prompt event selection; tables/event_selection_table.tex caption.

   Follow-up or recommended phrasing: Closed for the present review response.

15. [D11] Original comment: Page 10 Table 2: cutflow plot would be useful.

   Status: Needs follow-up

   Response: Agree. The response should say a cutflow plot/table is not yet included in this validation note. If the cutflow ROOT/hist output is available, add a compact appendix plot showing cumulative counts for 2015, 2016 10%, and 2021 1%.

   Where new information is in the note: tables/event_selection_table.tex.

   Follow-up or recommended phrasing: Add only if the needed cutflow source is available and stable.

16. [D12] Original comment: Page 10 Table 2: no opposite charge in 2016 trigger.

   Status: Patched this pass

   Response: Done. The 2016 trigger row now says the Pair1 trigger bit requires opposite ECal halves and does not encode track charge.

   Where new information is in the note: tables/event_selection_table.tex, Trigger/topology row.

   Follow-up or recommended phrasing: Closed.

17. [D13] Original comment: Page 10 Table 2: psum < 2.8 typo.

   Status: Already addressed

   Response: The current 2021 row says the production 1% histogram imposes p_sum < 2.8 GeV. The old typo is not present in the table source.

   Where new information is in the note: tables/event_selection_table.tex, Track momenta row.

   Follow-up or recommended phrasing: Closed.

18. [D14] Original comment: Page 10 Table 2: chi2 vs chi2/ndf.

   Status: Addressed for 2021 in this pass

   Response: The table now labels the 2021 track-fit cut as chi^2_trk/ndf < 20 and keeps the vertex requirement as chi^2_vtx < 20, matching the 2021 full-note preselection table. The older 2015/2016 entries remain source-summary entries and can be audited separately if the committee wants raw-variable provenance.

   Where new information is in the note: tables/event_selection_table.tex, Track fit/hits and Vertex/V0 quality rows; 2021_Full_Analysis_Note.pdf Sec. 4.8, Table 38.

   Follow-up or recommended phrasing: Closed for the 2021 concern; optional audit for 2015/2016 variable labels.

19. [D15] Original comment: Page 10 Table 2: 2021 electron track-cluster timing looks bogus.

   Status: Addressed in this pass

   Response: The timing row was corrected from an electron-track/electron-cluster expression to the full-note timing definitions: electron track vs positron cluster, positron track vs positron cluster, and relative electron-track/positron-track time. The data-window values are 6.0 ns, 5.1 ns, and 7.8 ns respectively.

   Where new information is in the note: tables/event_selection_table.tex, Track--cluster timing row; 2021_Full_Analysis_Note.pdf Sec. 4.7 and Table 38.

   Follow-up or recommended phrasing: Closed.

20. [D16] Original comment: Page 10 Table 2: trigger topology undefined, likely singles2 || singles3.

   Status: Addressed in this pass

   Response: The 2021 trigger row now explicitly says Single-2 or Single-3. The source trigger table identifies S2 as positron no-hodoscope and S3 as positron with hodoscope, both with VTP prescale 1 in the reference 3.74 GeV configuration.

   Where new information is in the note: tables/event_selection_table.tex, Trigger/topology row; 2021_Full_Analysis_Note.pdf Appendix A, Table 63.

   Follow-up or recommended phrasing: Closed.

21. [D17] Original comment: Page 10 Table 2: cluster-cluster time coincidence in 2021.

   Status: Addressed in this pass

   Response: The table now says there is no separate offline cluster-cluster coincidence in the full 2021 preselection table. For context, the Pair-0 hardware trigger has a cluster-pair timing window |Delta t| < 12 ns and VTP prescale 100, but the sanity trigger in Table 38 is Single-2 || Single-3.

   Where new information is in the note: tables/event_selection_table.tex, Cluster--cluster coincidence row; 2021_Full_Analysis_Note.pdf Table 38 and Appendix A, Tables 65-66.

   Follow-up or recommended phrasing: Closed unless a separate prompt-tight offline coincidence is later added to the production recipe.

22. [D18] Original comment: Page 10 Table 2: candidate multiplicity in 2021 same as 2016?

   Status: Addressed in this pass

   Response: The 2021 row now records the full-note candidate multiplicity requirement exactly: one selected good vertex, written as N_vtx > 0 and N_vtx < 2.

   Where new information is in the note: tables/event_selection_table.tex, Candidate multiplicity row; 2021_Full_Analysis_Note.pdf Sec. 4.8, Table 38.

   Follow-up or recommended phrasing: Closed.

23. [D19] Original comment: Page 10 Table 2: target-constrained preselection revisited?

   Status: Addressed with source-backed wording

   Response: The 2021 event-selection table now records the full-note vertex requirements p_vtx < 4.0 GeV and chi^2_vtx < 20, while the 2021 resolution section states that the current prompt workflow uses the inclusive target-constrained V0 fit without hit requirements. If the final prompt note adds a more specific target-constrained selection refinement, that should be inserted later.

   Where new information is in the note: tables/event_selection_table.tex, Vertex/V0 quality row; sections/03_event_selection.tex, 2021 mass-resolution subsection; 2021_Full_Analysis_Note.pdf Table 38.

   Follow-up or recommended phrasing: Closed for the preselection-table comment; keep final prompt-note synchronization as a later check.

24. [D20] Original comment: Page 11 line 229: psum < 2.8 typo?

   Status: Already addressed

   Response: The current note consistently uses p_sum < 2.8 GeV for the 2021 production histogram cut where this comment applies.

   Where new information is in the note: sections/03_event_selection.tex and tables/event_selection_table.tex.

   Follow-up or recommended phrasing: Closed.

25. [D21] Original comment: Page 12 line 261: 2015 smearing correction was direct mass-resolution scaling; 2016 real track-level smearing propagated to mass.

   Status: Already addressed

   Response: Addressed in the mass-resolution discussion: the note distinguishes the 2015 internal scaled-resolution treatment from the 2016 track-level smearing correction and its propagated mass resolution.

   Where new information is in the note: sections/03_event_selection.tex, 2015 and 2016 mass-resolution paragraphs.

   Follow-up or recommended phrasing: Closed; keep this distinction in final edits.

26. [D22] Original comment: Page 13 Figure 4: legend "Data" should be "Moller scattering data"; uncertainties are nice; plot stops near 100 MeV and is poorly modeled above 81 MeV, raising concern for 2015 resolution.

   Status: Patched this pass; residual caveat

   Response: The 2015 mass-resolution crop now labels the control point as "Moller scattering data." The note also includes a new three-panel resolution comparison so readers can see the configured ranges. The reviewer concern above the published 2015 range remains a valid caveat; answer by stating the 2015 resolution parameterization is used only as the internal-note input for the GPR validation and that sensitivity to this choice can be assessed by a sigma_m shift.

   Where new information is in the note: resolution_figs/hps2015_mass_resolution_internal_fig24.png; sections/03_event_selection.tex; resolution_figs/hps_mass_resolution_three_panel.*.

   Follow-up or recommended phrasing: Optional final study: shift 2015 sigma_m above the published range to quantify limit impact.

27. [D23] Original comment: Page 13 line 297: define FEE.

   Status: Patched this pass

   Response: Done. The note now expands FEE as full-energy-electron before using the acronym.

   Where new information is in the note: sections/03_event_selection.tex, 2016 mass-resolution paragraph.

   Follow-up or recommended phrasing: Closed.

28. [D24] Original comment: Page 14 line 295: plot modified parameterization full range; EP suggested all three chosen mass resolutions in a 3x1 row.

   Status: Patched this pass

   Response: Done. Added a 3x1 summary of the configured 2015, 2016, and 2021 resolution inputs. The 2016 panel shows the full configured extension to 210 MeV, including the extrapolated high-mass tail.

   Where new information is in the note: resolution_figs/hps_mass_resolution_three_panel.*; sections/03_event_selection.tex.

   Follow-up or recommended phrasing: Closed after compile/render check.

29. [D25] Original comment: Page 14 Figure 5: x-axis missing; caption last sentence does not belong; figure does not show full GPR prompt range; resolution above 180 hard to gauge, maybe use same range as 2016.

   Status: Mostly addressed by new figure

   Response: The new three-panel summary has axis labels and shows the configured 2016 tail through 210 MeV. If Figure 5 remains as a source crop, keep it for provenance and rely on the new summary plot for the full-range comparison.

   Where new information is in the note: sections/03_event_selection.tex; resolution_figs/hps_mass_resolution_three_panel.*.

   Follow-up or recommended phrasing: Check after compile that the original caption no longer carries the stray sentence.

30. [D26] Original comment: Page 14 line 309: no accounting for mass-resolution discrepancy between 2021 data and MC?

   Status: Partly addressed; final diagnostic still open

   Response: The current validation note uses the configured 2021 target-constrained V0 mass-resolution parameterization and now states its provenance and limitations. We agree that the full 2021 unblinding needs either a direct data/MC resolution comparison or a shifted-sigma_m diagnostic propagated to sensitivity.

   Where new information is in the note: sections/03_event_selection.tex, 2021 mass-resolution subsection; resolution_figs/hps2021_mass_resolution_target_constrained_nohitreq.png.

   Follow-up or recommended phrasing: Open final-analysis diagnostic.

31. [D27] Original comment: Page 14 Equation 5: units clear; Figure 6 in MeV.

   Status: Already addressed

   Response: The current resolution equations and plots use explicit MeV/GeV units in the text and captions. If the compiled page still mixes units visually, standardize the axis label in the figure source.

   Where new information is in the note: sections/03_event_selection.tex and resolution figures.

   Follow-up or recommended phrasing: Check compiled figure labels.

32. [D28] Original comment: Page 15 Figure 6: does not show full GPR prompt range.

   Status: Addressed by new figure

   Response: The new three-panel mass-resolution summary shows the configured prompt ranges, including 2021 through 250 MeV. Keep the older figure only as a provenance/source figure if needed.

   Where new information is in the note: resolution_figs/hps_mass_resolution_three_panel.*.

   Follow-up or recommended phrasing: Closed unless reviewer wants the original Figure 6 regenerated too.

33. [D29] Original comment: Page 16 Equation 7: define A_d after "signal yield".

   Status: Already addressed

   Response: The current methodology defines the dataset-specific signal normalization/acceptance factors around the yield-to-epsilon^2 conversion. If the exact symbol A_d appears before definition after compile, move the definition immediately after first use.

   Where new information is in the note: sections/04_methodology.tex, signal-yield and density-normalization subsections.

   Follow-up or recommended phrasing: Compile-text check.

34. [D30] Original comment: Page 16 Figure 7: missing axis titles.

   Status: Needs check

   Response: Recommended response: regenerate or replace the figure with axis titles if the current compiled figure still lacks them. This is a cosmetic figure-source issue and should be fixed directly rather than only explained.

   Where new information is in the note: Figure source used for old Figure 7; current compiled note page TBD.

   Follow-up or recommended phrasing: After compile, inspect the figure page and patch the plotting script if still missing.

35. [D31] Original comment: Page 17 line 346: clarify radiative fraction explodes above 180 MeV.

   Status: Mostly addressed

   Response: The note now avoids relying on an unstable high-mass radiative-fraction extrapolation. The current workflow uses a stable conservative radiative-fraction surrogate, with the penalty treatment documented separately as a coupling-normalization systematic.

   Where new information is in the note: sections/03_event_selection.tex, radiative-fraction and systematic-penalty discussion.

   Follow-up or recommended phrasing: Closed if the final compiled text keeps this caveat explicit.

36. [D32] Original comment: Page 17 Equation 10: what is it based on? current 2021 analysis?

   Status: Partly addressed; final 2021 normalization still provisional

   Response: The equation is the validation-stage yield-to-epsilon^2 normalization used by the current GPR workflow, based on the selected prompt density and dataset-specific radiative-fraction input. For 2021, the note points to the full-analysis-note radiative-fraction study but still treats the current value as provisional pending the final upgraded-detector normalization.

   Where new information is in the note: sections/03_event_selection.tex, 2021 radiative fraction; sections/04_methodology.tex, signal-yield conversion.

   Follow-up or recommended phrasing: Keep as validation-stage wording unless final 2021 normalization is frozen.

37. [D33] Original comment: Page 18 line 363: radiative-fraction penalty and 1.075; EP says corrected.

   Status: Addressed

   Response: Corrected. The note now describes the 7% radiative-fraction penalty as a coherent normalization scenario: reducing f_rad by 7% weakens the inferred epsilon^2 limit by about 1/(1-0.07) ~= 1.075. It is not described as a profiled nuisance in the current likelihood.

   Where new information is in the note: sections/03_event_selection.tex, systematic-penalty discussion.

   Follow-up or recommended phrasing: Closed.

38. [D34] Original comment: Page 19 line 393: "narrow mass-dependent training region" should be testing?

   Status: Patched this pass

   Response: Corrected conceptually. The methodology now uses "training-exclusion region" for the mass-dependent hole around the test mass, avoiding confusion with the sideband training sample.

   Where new information is in the note: sections/04_methodology.tex, opening methodology subsection.

   Follow-up or recommended phrasing: Closed.

39. [D35] Original comment: Page 19 line 398: d labels defined too late.

   Status: Already addressed

   Response: Addressed. The methodology now defines d as the dataset label at the start of the observable/scan/blinding subsection before the equations use it.

   Where new information is in the note: sections/04_methodology.tex, Observable/scan/blinding subsection.

   Follow-up or recommended phrasing: Closed.

40. [D36] Original comment: Page 19 Section 4.2: binned invariant mass and bin edges of blind/signal regions; EP says binning studies show no issue when bins are smaller than resolution.

   Status: Patched this pass

   Response: Done. The note now states that bin centers determine blind/sideband membership, that the bins are narrower than the detector resolution in the scanned range, and that the signal template is still integrated over full bin edges.

   Where new information is in the note: sections/04_methodology.tex, bin-assignment paragraph.

   Follow-up or recommended phrasing: Closed unless reviewer requests a dedicated binning-study plot.

41. [D37] Original comment: Page 20 line 447: RBF not defined.

   Status: Already addressed

   Response: Addressed in the kernel subsection: RBF is expanded as radial basis function and the kernel equation is shown.

   Where new information is in the note: sections/04_methodology.tex, Kernel choice subsection.

   Follow-up or recommended phrasing: Closed.

42. [D38] Original comment: Page 23 Figure 12a: describe curves/LSLB; legend covers title; which plots motivate 1-sigma choice?

   Status: Needs follow-up / partially superseded

   Response: The current methodology no longer presents 1 sigma as the current baseline; it explains that 1.64 was used in earlier validation and 2.25 is the current observed/refmatched baseline. If the old figure remains, update the caption/legend placement and explicitly say which validation plot motivated the displayed choice.

   Where new information is in the note: sections/04_methodology.tex and sections/05_toys_validation.tex.

   Follow-up or recommended phrasing: Inspect compiled Figure 12 and adjust legend/caption if still present.

43. [D39] Original comment: Page 26 line 543: only e+e- channel open; corrected beyond dimuon threshold?

   Status: Already addressed

   Response: Addressed. The current text notes the dimuon-threshold issue and treats the e+e- branching/density conversion accordingly for masses beyond the threshold.

   Where new information is in the note: sections/04_methodology.tex, density-normalization/branching discussion.

   Follow-up or recommended phrasing: Closed after compile-text check.

44. [D40] Original comment: Page 26 Equation 33: numerator should be sum_i b_{d,i}? N_d number of bins below.

   Status: Already addressed / check equation

   Response: Recommended response: verify the compiled equation uses an explicit sum over bins for the background density normalization and that N_d is defined as the number of bins in the normalization window. If the equation still has ambiguous numerator notation, patch it.

   Where new information is in the note: sections/04_methodology.tex, density-normalization equation.

   Follow-up or recommended phrasing: Compile-text check.

45. [D41] Original comment: Page 28 Equation 43: define Phi.

   Status: Already addressed

   Response: Addressed. The current methodology defines Phi as the standard normal cumulative distribution function where it appears in the CLs/normal-approximation expressions.

   Where new information is in the note: sections/04_methodology.tex, CLs/statistical inference subsection.

   Follow-up or recommended phrasing: Closed.

46. [D42] Original comment: Page 28 line 611: define Zlocal(m).

   Status: Already addressed

   Response: Addressed. The local-significance notation is defined in the statistical-inference section before being used in plots.

   Where new information is in the note: sections/04_methodology.tex.

   Follow-up or recommended phrasing: Closed.

47. [D43] Original comment: Page 29 Figure 13: state fixed mass; is it really epsilon-squared? values 1e-4/1e-5; state colors strong/weak.

   Status: Needs figure-caption check

   Response: Recommended response: update the caption to state the fixed mass hypothesis, confirm the injection parameter is epsilon^2 rather than epsilon, and label colors as stronger/weaker signal or larger/smaller epsilon^2. Do not leave this implicit.

   Where new information is in the note: Figure 13 source/caption in sections/04_methodology.tex or sections/05_toys_validation.tex.

   Follow-up or recommended phrasing: Inspect compiled caption and patch if still ambiguous.

48. [D44] Original comment: Page 39 Equations 51-54: common notation for concatenation, n1||n2.

   Status: Needs notation cleanup if still present

   Response: Recommended response: use a single notation for concatenated vectors, e.g. (n_1, n_2) or n_1 || n_2, and define it once before the block of equations.

   Where new information is in the note: sections/04_methodology.tex or combined-likelihood equations.

   Follow-up or recommended phrasing: Search compiled/source equations and standardize if still inconsistent.

49. [D45] Original comment: Page 31 line 665: why only 10% of 2016? EP: staged conservative, avoids biasing method in new parameter space.

   Status: Already addressed / recommended phrasing

   Response: Use the staged-analysis response: the 10% sample is a development and validation stream that avoids tuning the GP/limit machinery on the full 2016 sample before the first 100% unblinding. The note now places 2016 10% beside the 2021 1% validation stream in that staged logic.

   Where new information is in the note: sections/02_datasets.tex, staged-analysis paragraph and Table 1.

   Follow-up or recommended phrasing: Closed after confirming subset bookkeeping.

50. [D46] Original comment: Page 32 Figure 16: stuff off scale; straight 3/5 sigma not helpful; figure not referenced.

   Status: Needs figure cleanup if still present

   Response: Recommended response: either reference the figure explicitly and rescale it so the relevant mass range is visible, or move it to an appendix and remove non-informative 3/5 sigma guide lines.

   Where new information is in the note: sections/05_toys_validation.tex, Figure 16 area.

   Follow-up or recommended phrasing: Inspect compiled validation figures and patch if still crowded.

51. [D47] Original comment: Page 33 Figure 17: What is GPR?

   Status: Already addressed by primer

   Response: The primer now defines GPR before validation figures. If the caption still uses only the acronym, change the caption phrase to "Gaussian-process-regression (GPR) background prediction" on first use.

   Where new information is in the note: sections/04_methodology.tex; sections/05_toys_validation.tex.

   Follow-up or recommended phrasing: Caption check after compile.

52. [D48] Original comment: Page 38 Figure 20: what is prefit; maybe preprocessing? sigmaA/ref diverges high mass; train2.25 optimality; can analytic calculation explain mass variation?

   Status: Partly addressed

   Response: The current note emphasizes the 2.25 sigma_m blind/extraction geometry as closure-motivated, not analytically optimal. The mass dependence in the diagnostic comes from changing resolution, binning, sideband support, background slope, and statistics; a simple Gaussian-containment estimate is useful intuition but not the selection rule.

   Where new information is in the note: sections/04_methodology.tex, blind-window construction; sections/05_toys_validation.tex, guard-band/closure discussion.

   Follow-up or recommended phrasing: If the word prefit remains in a caption, define it as the GP preprocessing/background prediction or rename it.

53. [D49] Original comment: Page 39 Figure 21: pull widths suggest underconfident; discuss why/CLs impact; missing test of sigma_d uncertainty effects. EP asks for options.

   Status: Recommended response; final sigma_m test still open

   Response: Do not describe the widths as near unity if the plot is visibly below one. Say the pull widths are below unity in this toy ensemble, indicating mildly conservative predictive uncertainties for this configuration; coverage and CLs validation are the limit-level checks. A shifted-sigma_m study will be used to decide whether template-width uncertainty needs to become a nuisance.

   Where new information is in the note: sections/05_toys_validation.tex, pull-width and coverage discussion.

   Follow-up or recommended phrasing: Open decision: add shifted-resolution diagnostic now or mark it as final-unblinding follow-up.

54. [D50] Original comment: Page 40 Figure 22: define error boxes/bars/horizontal lines.

   Status: Needs caption check

   Response: Recommended response: update the caption to define the filled boxes, vertical error bars, and horizontal reference lines in the first sentence. This is a direct cosmetic caption fix if still missing.

   Where new information is in the note: sections/05_toys_validation.tex, Figure 22 caption.

   Follow-up or recommended phrasing: Inspect compiled caption and patch if still unclear.

55. [D51] Original comment: Page 46 line 776: broken sentence.

   Status: Already addressed or needs text search

   Response: Recommended response: the current source should be searched after compile; if the broken sentence remains, patch it directly. This is a straightforward prose cleanup.

   Where new information is in the note: sections/05_toys_validation.tex around the old support-check discussion.

   Follow-up or recommended phrasing: Run source/compiled-text check for the broken phrase.

56. [D52] Original comment: Page 46 lines 781-: undefined support check, sideband-fraction check, configured Pearson target.

   Status: Needs definitions if still present

   Response: Recommended response: define the support check as the requirement that enough sideband bins remain after the mass-dependent exclusion, define the sideband-fraction check as the fraction of the nominal sideband retained, and define the configured Pearson target as the reference chi2-like residual threshold used in toy validation.

   Where new information is in the note: sections/05_toys_validation.tex, validation-check definitions.

   Follow-up or recommended phrasing: Patch definitions if terms are still used before definition.

57. [D53] Original comment: Page 49 line 830: "pull widths are near unity within toy precision" is a stretch.

   Status: Recommended response / likely patch

   Response: Agree. Recommended wording: "The pull widths are below unity in this toy ensemble, indicating a mildly conservative predictive uncertainty in the tested configuration; the coverage and CLs checks are used to assess whether this conservatism biases the final limits."

   Where new information is in the note: sections/05_toys_validation.tex, pull-width paragraph.

   Follow-up or recommended phrasing: Patch source if the old phrase remains.

58. [D54] Original comment: Page 53 Figure 32: not referenced.

   Status: Needs figure reference check

   Response: Recommended response: reference Figure 32 in the nearby validation text if it is useful; otherwise remove or move it to an appendix.

   Where new information is in the note: sections/05_toys_validation.tex.

   Follow-up or recommended phrasing: Compile/source check for figure reference.

59. [D55] Original comment: Page 57 Figure 33: make blind window more visible in legend.

   Status: Needs cosmetic plot update if still present

   Response: Recommended response: update the plotting script to use a thicker/darker blind-window legend entry or a filled translucent band with a clear legend label.

   Where new information is in the note: Figure 33 plotting source under hps_gpr_analysis_note/scripts or generated figures.

   Follow-up or recommended phrasing: Patch plot if figure remains in the main note.

60. [D56] Original comment: Page 58 Figure 34: not referenced.

   Status: Needs figure reference check

   Response: Recommended response: reference Figure 34 in text or move/remove it. The response should not claim this is closed unless the compiled source has a text reference.

   Where new information is in the note: sections/05_toys_validation.tex.

   Follow-up or recommended phrasing: Compile/source check for figure reference.

61. [D57] Original comment: Page 59 Figure 35: q## and SEM not relevant?

   Status: Needs caption/plot cleanup

   Response: Recommended response: remove internal quantile/SEM labels from the publication-facing figure unless they are defined and used in the text. Replace with plain-language labels or move the diagnostic to an internal appendix.

   Where new information is in the note: Figure 35 plotting source/caption.

   Follow-up or recommended phrasing: Patch if still visible in compiled note.

62. [D58] Original comment: Page 63 Figure 38: expect more impact from 2016 than 2015 at higher masses, seems opposite.

   Status: Open analysis check

   Response: We agree this behavior is not self-explanatory. Before freezing the result figure we will audit active ranges, exposure/efficiency scaling, prompt-density conversion, radiative inputs, mass-resolution differences, and interpolation masks. If the effect is real, the note will explain the dataset leverage next to the figure.

   Where new information is in the note: sections/06_results.tex and the relevant result-comparison plotting source.

   Follow-up or recommended phrasing: Open high-priority result-figure audit.

63. [D59] Original comment: Page 67 Figure 42: include previous results in (a),(b).

   Status: Needs plot update if final comparison figure remains

   Response: Recommended response: add published 2015 and 2016 curves to panels (a) and (b), or state that those comparisons are moved to the dedicated appendix figures.

   Where new information is in the note: sections/results or comparison plotting script.

   Follow-up or recommended phrasing: Patch figure or cross-reference appendix.

64. [D60] Original comment: Page 69 Figure 45: above 210 MeV only 2021 contributes, so combined should equal 2021-only; why is it not?

   Status: Open high-priority analysis check

   Response: Above the active 2015/2016 ranges, the simultaneous result should reduce to the 2021-only result if the same 2021 inputs and CL machinery are used. If the plotted curves differ, this is a plotting or bookkeeping issue to audit before final circulation.

   Where new information is in the note: sections/06_results.tex and result-comparison plotting source.

   Follow-up or recommended phrasing: Audit active-dataset masks, grids, density/radiative factors, dimuon correction, normalization, and CL extraction.

65. [D61] Original comment: Page 71 Figure 47: vertical red dotted line?

   Status: Needs caption check

   Response: Recommended response: define the vertical red dotted line in the caption and text, or remove it. If it marks the dimuon threshold or active-range boundary, name that explicitly.

   Where new information is in the note: Figure 47 source/caption.

   Follow-up or recommended phrasing: Patch caption after compile inspection.

66. [D62] Original comment: Page 72 Figure 49: panel (a) legend says 3-sigma line absent; remove 3/5 sigma from legend in panel (b).

   Status: Needs cosmetic plot update

   Response: Recommended response: regenerate the figure with legend entries matching only lines actually drawn, and remove the 3/5 sigma entries from panel (b) if those guides are not shown or useful.

   Where new information is in the note: Figure 49 plotting source.

   Follow-up or recommended phrasing: Patch plot if figure remains in main note.

67. [D63] Original comment: Page 73 Figure 50: ideal 2021 resolution? 2015/2016 MC smeared; should expect limit degradation? Need rough 2021 data/MC perspective.

   Status: Partly addressed; final 2021 resolution diagnostic still open

   Response: The current note no longer presents the 2021 resolution as an ideal final detector model; it identifies the implemented target-constrained V0 curve and flags that full 2021 validation is still pending. A direct data/MC resolution comparison or sigma_m-shift sensitivity band should be added before final unblinding.

   Where new information is in the note: sections/03_event_selection.tex; resolution_figs/hps2021_mass_resolution_target_constrained_nohitreq.png; resolution_figs/hps_mass_resolution_three_panel.png.

   Follow-up or recommended phrasing: Same open final-analysis diagnostic as D26.

68. [D64] Original comment: Page 74 Figures 51: state how/where radiative fraction and mass resolution differ from previous; possible comparison for 2015.

   Status: Mostly addressed; caption check still useful

   Response: The note now documents the relevant differences from previous analyses: updated scan ranges, GPR background treatment, non-renormalized Gaussian templates, radiative-fraction conventions, density normalization, and mass-resolution inputs. For 2015 specifically, the appendix provides the direct corrected-polynomial vs GPR comparison.

   Where new information is in the note: sections/03_event_selection.tex; sections/04_methodology.tex; sections/appendix_2015_method_comparison.tex.

   Follow-up or recommended phrasing: Check comparison captions after compile and add one sentence near the figure if still implicit.

69. [D65] Original comment: Page 75 Figure 52: direct comparison of 2015-only published data and this analysis. EP: appendix.

   Status: Addressed by appendix path

   Response: A direct 2015-only comparison is included in the appendix so the main text stays focused on the combined workflow. The appendix comparison is a cross-check of normalization scale, not a standalone validation of coverage or signal absorption.

   Where new information is in the note: sections/appendix_2015_method_comparison.tex; final_limit_projection_figs/individual_datasets/hps2015_95cl_observed_vs_internal_note.png.

   Follow-up or recommended phrasing: Closed if the appendix compiles cleanly.

70. [D66] Original comment: Page 76 Figure 53: if comparing with BaBar at low mass add NA62. EP may compare.

   Status: Decision recommended

   Response: The note already includes broader prompt-visible context with external constraints, and the BaBar comparison is best treated as a projection benchmark. If the low-mass comparison is kept as a physics-reach claim, NA62 or another documented low-mass external contour should be added; otherwise state that BaBar is retained only as a historical benchmark for the projection.

   Where new information is in the note: sections/01_introduction.tex, external-constraint panel; sections/06_results.tex, BaBar/projection comparison.

   Follow-up or recommended phrasing: EP/committee choice: limit the comparison scope or add NA62.
