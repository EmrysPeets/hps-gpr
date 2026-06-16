# HPS GPR Round 1 Review Response Matrix

Source review document: `/Users/emryspeets/Desktop/summer_26/gpr_note/round1_review_comments.pdf`
Old comparison note: `/Users/emryspeets/Desktop/summer_26/gpr_note/hps_gpr_rev_version1.pdf`
Most recent pre-pass analysis note PDF: `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/hps_gpr_analysis_note/main06162026.pdf`
Local compile target after this pass: `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/hps_gpr_analysis_note/build_review_round1/main.pdf`

## Instructions to Address
- **Patched this pass:** Added a unit-normalized 2015/2016/2021 mass-shape overlay and kept the absolute log-y overlay. Updated the 2021 highlighted expected search region to 35-250 MeV.
- **Patched this pass:** Added a three-panel mass-resolution comparison for the configured 2015, 2016, and 2021 resolution inputs.
- **Patched this pass:** Relabeled the 2015 Moller control point in the mass-resolution crop from an internal pass name to "Moller scattering data".
- **Patched this pass:** Defined FEE, fixed the axial/stereo tracker wording, referenced the dataset table, clarified EOT usage, and corrected the 2016 trigger/topology wording.
- **Patched this pass:** Clarified bin-center based blind/sideband assignment, the 2.25 sigma current blind/extraction choice, and the scikit-learn GaussianProcessRegressor implementation.
- **Needs EP/analysis decision:** For the final response, choose how strongly to commit on the 2021 mass-resolution data/MC validation and whether to add a resolution-shift systematic study.
- **Needs EP/analysis decision:** Audit the 2021 event-selection table against the final upgraded-detector prompt note, especially timing, cluster coincidence, candidate multiplicity, and trigger/topology wording.
- **Needs EP/analysis decision:** Investigate the high-mass combined-limit overlay: above the 2015/2016 ranges the combined curve should reduce to the 2021-only result if exactly the same 2021 ingredients are used.
- **Recommended response choice:** Keep the BaBar comparison if it is only a historical benchmark; add NA62 only if the analysis makes a low-mass external-comparison claim and the assumptions can be documented.

## Recommended Responses for Complicated Items
- **Combination and normalization systematics:** Recommended response: the current likelihood combines datasets through one shared epsilon^2 parameter with dataset-specific signal efficiencies, resolutions, radiative fractions, and background predictions. Normalization inputs affect the signal-yield-to-epsilon^2 conversion, not the GP interpolation. If the final note does not profile luminosity/EOT/efficiency uncertainties, state that explicitly and justify the choice as subdominant or deferred.
- **2021 mass resolution:** Recommended response: the current 1% validation note uses the configured target-constrained V0 mass-resolution parameterization. For the final 2021 unblinding note, either add a direct data/MC resolution validation, add a resolution-shift nuisance/diagnostic, or state that the present result is a validation-stage result pending that check.
- **Radiative fraction above 180 MeV:** Recommended response: the 2016-derived extrapolation is not used literally in the high-mass region because it becomes unstable; the 2021 prompt-density normalization instead uses the documented conservative radiative-fraction penalty. If the collaboration prefers, replace the surrogate by a common 7% penalty and show that the limit movement is small.
- **Why 10% of 2016:** Recommended response: the 10% sample is a staged development and validation sample used to avoid tuning the method on the full 2016 dataset before the final unblinding. It also provides a closer statistical bridge to the 2021 1% validation stream.
- **2.25 sigma training/extraction choice:** Recommended response: 2.25 sigma is a conservative closure-motivated baseline, not a claim of analytic sensitivity optimality. The note should say that narrower windows have better naive F/sqrt(k) sensitivity but showed signal absorption in full-refit toys.
- **Pull widths below unity:** Recommended response: do not say "near unity" when widths are visibly below one. Say the tested configuration is mildly conservative in these toys, and point to coverage/CLs validation as the relevant limit-level diagnostic.
- **sigma_m uncertainty test:** Recommended response: add a compact diagnostic in which toys are generated with sigma_m shifted up/down and fit with nominal templates, or add the reverse check, before deciding whether a full template morphing nuisance is necessary.
- **High-mass combined curve:** Recommended response: above the 2015 and 2016 active ranges the combined result should match the 2021-only result. If it does not, audit active-dataset masks, radiative/density normalization, interpolation grids, and CL extraction before freezing the figure.

## Comment-by-Comment Responses
### General
#### G1. Add more introduction to Gaussian process regression in Section 4; explain edge handling, one-sided sidebands, and whether the software is custom or package-based.
- **Status:** Addressed / patched
- **Response:** Agree. The current note now gives the GP primer, defines the blinded/training-exclusion geometry, explains that bins outside the mass-dependent exclusion form the sideband training sample, and states that the GP fit uses scikit-learn GaussianProcessRegressor with custom HPS code for masks, templates, likelihoods, and CLs. The scan ranges are chosen to avoid unsupported edge behavior, and validation plots check the residual behavior across the scanned mass range.
- **Pointer:** sections/04_methodology.tex, Sec. Observable/scan/blinding and Log-space preprocessing; new ScikitLearn citation in hps_gpr_analysis_note.bib.
- **Instruction:** No further note edit unless the committee wants an even more tutorial GP paragraph in the introduction.

#### G2. This is the first HPS bump hunt combining multiple datasets/run periods/beam energies. Add introduction comments. How do likelihood uncertainties and normalization uncertainties enter?
- **Status:** Mostly addressed; decision recommended
- **Response:** Agree. The note now frames the analysis as a combined prompt-resonance search over 2015, 2016, and 2021 datasets. The likelihood combines datasets through a shared epsilon^2 parameter with per-dataset GP background predictions and signal-yield normalizations. Normalization inputs affect the conversion from prompt yield to epsilon^2; they are not GP background-shape nuisance parameters. Recommended final wording: "The current validation workflow does not yet profile correlated luminosity/EOT/efficiency nuisance parameters; these enter the dataset-specific signal normalization and will be audited before the final unblinding."
- **Pointer:** sections/01_introduction.tex; sections/04_methodology.tex, combined-likelihood and density-normalization subsections.
- **Instruction:** EP/committee choice: either state normalization systematics are deferred/subdominant for this validation note, or add/profile the relevant nuisance parameters.

#### G3. Add mass yield and shape comparison for 2015/2016/2021. EP proposed using dataset_summary_figs/invariant_mass_distributions_2015_2016_2021_log.png, changing 2021 expected search region to 35-250 MeV, and adding a normalized overlay before it.
- **Status:** Patched this pass
- **Response:** Done. A new unit-normalized overlay was added before the absolute-yield log plot, and the 2021 expected-region highlight in the absolute-yield plot now spans 35-250 MeV. The normalized plot is for shape comparison; the log-y plot keeps the yield-scale context.
- **Pointer:** sections/02_datasets.tex, figs. dataset-mass-shape-overlay and dataset-mass-distributions-log; dataset_summary_figs/invariant_mass_distributions_2015_2016_2021_normalized.* and ..._log.*.
- **Instruction:** Closed after compile check.

#### G4. Preliminary mass resolution, especially 2021; placeholder text should be addressed after the rest.
- **Status:** Partly addressed; decision recommended
- **Response:** A three-panel resolution comparison has been added, and the 2021 section now points to the configured target-constrained V0 parameterization used in the validation workflow. This should be enough for a review-stage response, but the final note should either add a 2021 data/MC resolution validation or state the resolution systematic treatment explicitly.
- **Pointer:** sections/03_event_selection.tex; resolution_figs/hps_mass_resolution_three_panel.*.
- **Instruction:** Recommended final response: "The current note now shows all three configured mass-resolution inputs. A final 2021 resolution validation or sigma_m systematic is still required before the full 2021 unblinding."

### Details
#### D01. Page 4 line 96 and page 5 line 121: is "10 MeV" correct?
- **Status:** Already addressed
- **Response:** The current note no longer carries the old 10 MeV wording at those locations; the relevant scan ranges and mass windows are expressed with the current dataset-specific ranges.
- **Pointer:** sections/02_datasets.tex and current main06162026.pdf.
- **Instruction:** Closed.

#### D02. Page 6 line 141: GP not defined; primer needed.
- **Status:** Already addressed / patched
- **Response:** Addressed. GP/GPR is introduced in the introduction and methodology, with the background interpolation role explained before the detailed likelihood machinery.
- **Pointer:** sections/01_introduction.tex and sections/04_methodology.tex.
- **Instruction:** Closed unless a longer tutorial paragraph is desired.

#### D03. Page 6 line 142: define the mean vector.
- **Status:** Already addressed
- **Response:** Addressed in the current methodology by describing the latent smooth mean function and the observed covariance with the diagonal alpha term. If the reviewer wants vector notation specifically, add one sentence: "The GP mean vector is the vector of latent log-count means evaluated at the selected training-bin coordinates."
- **Pointer:** sections/04_methodology.tex, log-space preprocessing and covariance equations.
- **Instruction:** Optional wording only.

#### D04. Page 6 line 171: hodoscope mention? Single-3 triggers?
- **Status:** Partly addressed; verify 2021 trigger wording
- **Response:** The dataset section now mentions the positron hodoscope installation and the upgraded detector context. The 2021 trigger/topology wording in the selection table is still deliberately provisional, so do not claim closure until the final upgraded-detector prompt note fixes the Singles-2/Singles-3 trigger statement.
- **Pointer:** sections/02_datasets.tex; tables/event_selection_table.tex.
- **Instruction:** Audit 2021 trigger row against the final 2021 prompt-selection note.

#### D05. Page 6 line 173: "double-layer stereo" ambiguous.
- **Status:** Patched this pass
- **Response:** Done. The detector description now says "six axial/stereo tracking stations," which avoids the ambiguous double-layer wording.
- **Pointer:** sections/02_datasets.tex.
- **Instruction:** Closed.

#### D06. Page 7 line 159: details about blinded event selection? EP: 1% 2021 and 10% 2016 subset of runs spread evenly through the run period.
- **Status:** Patched this pass
- **Response:** Added the staged-subset statement: the 2016 10% and 2021 1% development streams are described as spread across their respective run periods rather than contiguous blocks. This answers the blinding-selection question at the level of the current note.
- **Pointer:** sections/02_datasets.tex, staged-analysis paragraph.
- **Instruction:** Confirm the exact bookkeeping sentence against the run-list owner before final sign-off.

#### D07. Page 9 Table 1: table not referenced.
- **Status:** Patched this pass
- **Response:** Done. The text now explicitly introduces Table 1 before the table appears.
- **Pointer:** sections/02_datasets.tex, paragraph before dataset-summary table.
- **Instruction:** Closed.

#### D08. Page 9 Table 1: EOT relevance/how calculated; targets differ.
- **Status:** Patched this pass; calculation details still optional
- **Response:** The note now says EOT is not an input to the GP background interpolation; it enters the dataset-specific signal normalization used to convert prompt yield to epsilon^2, together with target, acceptance, efficiency, and radiative-fraction inputs. The exact campaign-scale EOT calculation can be cited to run-quality/luminosity bookkeeping if needed.
- **Pointer:** sections/02_datasets.tex, paragraph after Table 1.
- **Instruction:** Optional: add a source citation or footnote for the EOT accounting if the collaboration has a canonical number source.

#### D09. Page 9 Table 1: why scan ranges differ between current and published searches.
- **Status:** Already addressed
- **Response:** Addressed in the current table caption. It states that the listed scan ranges are the current GPR analysis ranges rather than the smaller published window-search ranges, and it quotes the published 2015 and 2016 ranges.
- **Pointer:** sections/02_datasets.tex, Table 1 caption.
- **Instruction:** Closed.

#### D10. Page 9 line 223: "line-by-line bookkeeping..." why not?
- **Status:** Needs review
- **Response:** Recommended response: the current note intentionally summarizes selection inputs at the analysis-note level and does not reproduce full cut-file bookkeeping, because several 2021 prompt-tight definitions are still being synchronized with the upgraded-detector note. If final cut files are frozen, either add an appendix/cutflow table or replace this caveat with exact values.
- **Pointer:** tables/event_selection_table.tex and sections/03_event_selection.tex.
- **Instruction:** EP decision: keep caveat for validation note or add exact cut-file provenance.

#### D11. Page 10 Table 2: cutflow plot would be useful.
- **Status:** Needs follow-up
- **Response:** Agree. The response should say a cutflow plot/table is not yet included in this validation note. If the cutflow ROOT/hist output is available, add a compact appendix plot showing cumulative counts for 2015, 2016 10%, and 2021 1%.
- **Pointer:** tables/event_selection_table.tex.
- **Instruction:** Add only if the needed cutflow source is available and stable.

#### D12. Page 10 Table 2: no opposite charge in 2016 trigger.
- **Status:** Patched this pass
- **Response:** Done. The 2016 trigger row now says the Pair1 trigger bit requires opposite ECal halves and does not encode track charge.
- **Pointer:** tables/event_selection_table.tex, Trigger/topology row.
- **Instruction:** Closed.

#### D13. Page 10 Table 2: psum < 2.8 typo.
- **Status:** Already addressed
- **Response:** The current 2021 row says the production 1% histogram imposes p_sum < 2.8 GeV. The old typo is not present in the table source.
- **Pointer:** tables/event_selection_table.tex, Track momenta row.
- **Instruction:** Closed.

#### D14. Page 10 Table 2: chi2 vs chi2/ndf.
- **Status:** Needs verification
- **Response:** Recommended response: "We will audit whether each quoted chi2 value is a raw chi2, chi2/ndf, or a package-specific goodness variable, and will relabel the table entries accordingly." The current table should not overclaim until that audit is done.
- **Pointer:** tables/event_selection_table.tex, Track fit/hits and Vertex/V0 quality rows.
- **Instruction:** Audit variable definitions against the 2015/2016/2021 cut files.

#### D15. Page 10 Table 2: 2021 electron track-cluster timing looks bogus.
- **Status:** Needs verification
- **Response:** Agree. The 2021 timing row is still marked as working/provisional. Recommended response: "The 2021 timing expressions will be synchronized with the upgraded-detector prompt-selection note before final circulation; the current validation note only records the production histogram inputs."
- **Pointer:** tables/event_selection_table.tex, Track-cluster timing row.
- **Instruction:** Audit and correct the 2021 timing variables before final unblinding.

#### D16. Page 10 Table 2: trigger topology undefined, likely singles2 || singles3.
- **Status:** Needs verification
- **Response:** Recommended response: the note intentionally avoids freezing a 2021 prompt-only trigger/topology value until the final 2021 prompt note is synchronized. If the final definition is Singles-2 OR Singles-3, write that explicitly in the 2021 column.
- **Pointer:** tables/event_selection_table.tex, Trigger/topology row.
- **Instruction:** Replace provisional text with exact trigger bit logic when confirmed.

#### D17. Page 10 Table 2: cluster-cluster time coincidence in 2021.
- **Status:** Needs verification
- **Response:** The current table records that the upgraded-era coincidence requirement is present in the production histogram, but the final note-facing value is still being synchronized. That should remain an open instruction until the value is confirmed.
- **Pointer:** tables/event_selection_table.tex, Cluster-cluster coincidence row.
- **Instruction:** Insert exact 2021 coincidence requirement when confirmed.

#### D18. Page 10 Table 2: candidate multiplicity in 2021 same as 2016?
- **Status:** Needs verification
- **Response:** The current table says the production histogram applies an upgraded-era unique-candidate policy, but the exact note-facing wording is still being synchronized. Do not say it is identical to 2016 unless confirmed from the 2021 prompt selection.
- **Pointer:** tables/event_selection_table.tex, Candidate multiplicity row.
- **Instruction:** Confirm unique-candidate policy and wording.

#### D19. Page 10 Table 2: target-constrained preselection revisited?
- **Status:** Needs verification
- **Response:** The event-selection text identifies the target-constrained V0 mass and the current inclusive target-constrained fit choice for 2021. Recommended response: "The final prompt note will spell out which target-constrained V0 selection enters the production histogram and whether any prompt-tight refinement is added."
- **Pointer:** sections/03_event_selection.tex; tables/event_selection_table.tex, Vertex/V0 quality row.
- **Instruction:** Audit against final 2021 prompt-selection note.

#### D20. Page 11 line 229: psum < 2.8 typo?
- **Status:** Already addressed
- **Response:** The current note consistently uses p_sum < 2.8 GeV for the 2021 production histogram cut where this comment applies.
- **Pointer:** sections/03_event_selection.tex and tables/event_selection_table.tex.
- **Instruction:** Closed.

#### D21. Page 12 line 261: 2015 smearing correction was direct mass-resolution scaling; 2016 real track-level smearing propagated to mass.
- **Status:** Already addressed
- **Response:** Addressed in the mass-resolution discussion: the note distinguishes the 2015 internal scaled-resolution treatment from the 2016 track-level smearing correction and its propagated mass resolution.
- **Pointer:** sections/03_event_selection.tex, 2015 and 2016 mass-resolution paragraphs.
- **Instruction:** Closed; keep this distinction in final edits.

#### D22. Page 13 Figure 4: legend "Data" should be "Moller scattering data"; uncertainties are nice; plot stops near 100 MeV and is poorly modeled above 81 MeV, raising concern for 2015 resolution.
- **Status:** Patched this pass; residual caveat
- **Response:** The 2015 mass-resolution crop now labels the control point as "Moller scattering data." The note also includes a new three-panel resolution comparison so readers can see the configured ranges. The reviewer concern above the published 2015 range remains a valid caveat; answer by stating the 2015 resolution parameterization is used only as the internal-note input for the GPR validation and that sensitivity to this choice can be assessed by a sigma_m shift.
- **Pointer:** resolution_figs/hps2015_mass_resolution_internal_fig24.png; sections/03_event_selection.tex; resolution_figs/hps_mass_resolution_three_panel.*.
- **Instruction:** Optional final study: shift 2015 sigma_m above the published range to quantify limit impact.

#### D23. Page 13 line 297: define FEE.
- **Status:** Patched this pass
- **Response:** Done. The note now expands FEE as full-energy-electron before using the acronym.
- **Pointer:** sections/03_event_selection.tex, 2016 mass-resolution paragraph.
- **Instruction:** Closed.

#### D24. Page 14 line 295: plot modified parameterization full range; EP suggested all three chosen mass resolutions in a 3x1 row.
- **Status:** Patched this pass
- **Response:** Done. Added a 3x1 summary of the configured 2015, 2016, and 2021 resolution inputs. The 2016 panel shows the full configured extension to 210 MeV, including the extrapolated high-mass tail.
- **Pointer:** resolution_figs/hps_mass_resolution_three_panel.*; sections/03_event_selection.tex.
- **Instruction:** Closed after compile/render check.

#### D25. Page 14 Figure 5: x-axis missing; caption last sentence does not belong; figure does not show full GPR prompt range; resolution above 180 hard to gauge, maybe use same range as 2016.
- **Status:** Mostly addressed by new figure
- **Response:** The new three-panel summary has axis labels and shows the configured 2016 tail through 210 MeV. If Figure 5 remains as a source crop, keep it for provenance and rely on the new summary plot for the full-range comparison.
- **Pointer:** sections/03_event_selection.tex; resolution_figs/hps_mass_resolution_three_panel.*.
- **Instruction:** Check after compile that the original caption no longer carries the stray sentence.

#### D26. Page 14 line 309: no accounting for mass-resolution discrepancy between 2021 data and MC?
- **Status:** Needs EP/analysis decision
- **Response:** Recommended response: "The current validation note uses the configured 2021 target-constrained V0 resolution parameterization. A final 2021 data/MC resolution comparison, or an explicit sigma_m systematic diagnostic, will be added before final unblinding."
- **Pointer:** sections/03_event_selection.tex; recommended-options section of this document.
- **Instruction:** Perform shifted-resolution diagnostic or add data/MC validation.

#### D27. Page 14 Equation 5: units clear; Figure 6 in MeV.
- **Status:** Already addressed
- **Response:** The current resolution equations and plots use explicit MeV/GeV units in the text and captions. If the compiled page still mixes units visually, standardize the axis label in the figure source.
- **Pointer:** sections/03_event_selection.tex and resolution figures.
- **Instruction:** Check compiled figure labels.

#### D28. Page 15 Figure 6: does not show full GPR prompt range.
- **Status:** Addressed by new figure
- **Response:** The new three-panel mass-resolution summary shows the configured prompt ranges, including 2021 through 250 MeV. Keep the older figure only as a provenance/source figure if needed.
- **Pointer:** resolution_figs/hps_mass_resolution_three_panel.*.
- **Instruction:** Closed unless reviewer wants the original Figure 6 regenerated too.

#### D29. Page 16 Equation 7: define A_d after "signal yield".
- **Status:** Already addressed
- **Response:** The current methodology defines the dataset-specific signal normalization/acceptance factors around the yield-to-epsilon^2 conversion. If the exact symbol A_d appears before definition after compile, move the definition immediately after first use.
- **Pointer:** sections/04_methodology.tex, signal-yield and density-normalization subsections.
- **Instruction:** Compile-text check.

#### D30. Page 16 Figure 7: missing axis titles.
- **Status:** Needs check
- **Response:** Recommended response: regenerate or replace the figure with axis titles if the current compiled figure still lacks them. This is a cosmetic figure-source issue and should be fixed directly rather than only explained.
- **Pointer:** Figure source used for old Figure 7; current compiled note page TBD.
- **Instruction:** After compile, inspect the figure page and patch the plotting script if still missing.

#### D31. Page 17 line 346: clarify radiative fraction explodes above 180 MeV.
- **Status:** Already addressed / recommended phrasing
- **Response:** The note now explains the high-mass instability and the conservative penalty treatment. Recommended phrasing: "The 2016-derived radiative-fraction extrapolation is not used literally where it becomes unstable; the high-mass 2021 conversion uses the documented conservative penalty instead."
- **Pointer:** sections/03_event_selection.tex, radiative-fraction discussion.
- **Instruction:** Closed if final text keeps this caveat explicit.

#### D32. Page 17 Equation 10: what is it based on? current 2021 analysis?
- **Status:** Partly addressed
- **Response:** Recommended response: the equation is the validation-stage 2021 prompt-density/radiative-fraction treatment used in the current workflow, not yet a final 2021 prompt-analysis measurement. If possible, cite the 2021 full-analysis note or the production configuration used to derive the input.
- **Pointer:** sections/03_event_selection.tex, radiative-fraction/density-normalization paragraphs.
- **Instruction:** Add source/provenance sentence if not already explicit after compile.

#### D33. Page 18 line 363: radiative-fraction penalty and 1.075; EP says corrected.
- **Status:** Already addressed
- **Response:** Addressed. The current text describes the conservative radiative-fraction penalty and avoids the old confusing 1.075 phrasing.
- **Pointer:** sections/03_event_selection.tex.
- **Instruction:** Closed.

#### D34. Page 19 line 393: "narrow mass-dependent training region" should be testing?
- **Status:** Patched this pass
- **Response:** Corrected conceptually. The methodology now uses "training-exclusion region" for the mass-dependent hole around the test mass, avoiding confusion with the sideband training sample.
- **Pointer:** sections/04_methodology.tex, opening methodology subsection.
- **Instruction:** Closed.

#### D35. Page 19 line 398: d labels defined too late.
- **Status:** Already addressed
- **Response:** Addressed. The methodology now defines d as the dataset label at the start of the observable/scan/blinding subsection before the equations use it.
- **Pointer:** sections/04_methodology.tex, Observable/scan/blinding subsection.
- **Instruction:** Closed.

#### D36. Page 19 Section 4.2: binned invariant mass and bin edges of blind/signal regions; EP says binning studies show no issue when bins are smaller than resolution.
- **Status:** Patched this pass
- **Response:** Done. The note now states that bin centers determine blind/sideband membership, that the bins are narrower than the detector resolution in the scanned range, and that the signal template is still integrated over full bin edges.
- **Pointer:** sections/04_methodology.tex, bin-assignment paragraph.
- **Instruction:** Closed unless reviewer requests a dedicated binning-study plot.

#### D37. Page 20 line 447: RBF not defined.
- **Status:** Already addressed
- **Response:** Addressed in the kernel subsection: RBF is expanded as radial basis function and the kernel equation is shown.
- **Pointer:** sections/04_methodology.tex, Kernel choice subsection.
- **Instruction:** Closed.

#### D38. Page 23 Figure 12a: describe curves/LSLB; legend covers title; which plots motivate 1-sigma choice?
- **Status:** Needs follow-up / partially superseded
- **Response:** The current methodology no longer presents 1 sigma as the current baseline; it explains that 1.64 was used in earlier validation and 2.25 is the current observed/refmatched baseline. If the old figure remains, update the caption/legend placement and explicitly say which validation plot motivated the displayed choice.
- **Pointer:** sections/04_methodology.tex and sections/05_toys_validation.tex.
- **Instruction:** Inspect compiled Figure 12 and adjust legend/caption if still present.

#### D39. Page 26 line 543: only e+e- channel open; corrected beyond dimuon threshold?
- **Status:** Already addressed
- **Response:** Addressed. The current text notes the dimuon-threshold issue and treats the e+e- branching/density conversion accordingly for masses beyond the threshold.
- **Pointer:** sections/04_methodology.tex, density-normalization/branching discussion.
- **Instruction:** Closed after compile-text check.

#### D40. Page 26 Equation 33: numerator should be sum_i b_{d,i}? N_d number of bins below.
- **Status:** Already addressed / check equation
- **Response:** Recommended response: verify the compiled equation uses an explicit sum over bins for the background density normalization and that N_d is defined as the number of bins in the normalization window. If the equation still has ambiguous numerator notation, patch it.
- **Pointer:** sections/04_methodology.tex, density-normalization equation.
- **Instruction:** Compile-text check.

#### D41. Page 28 Equation 43: define Phi.
- **Status:** Already addressed
- **Response:** Addressed. The current methodology defines Phi as the standard normal cumulative distribution function where it appears in the CLs/normal-approximation expressions.
- **Pointer:** sections/04_methodology.tex, CLs/statistical inference subsection.
- **Instruction:** Closed.

#### D42. Page 28 line 611: define Zlocal(m).
- **Status:** Already addressed
- **Response:** Addressed. The local-significance notation is defined in the statistical-inference section before being used in plots.
- **Pointer:** sections/04_methodology.tex.
- **Instruction:** Closed.

#### D43. Page 29 Figure 13: state fixed mass; is it really epsilon-squared? values 1e-4/1e-5; state colors strong/weak.
- **Status:** Needs figure-caption check
- **Response:** Recommended response: update the caption to state the fixed mass hypothesis, confirm the injection parameter is epsilon^2 rather than epsilon, and label colors as stronger/weaker signal or larger/smaller epsilon^2. Do not leave this implicit.
- **Pointer:** Figure 13 source/caption in sections/04_methodology.tex or sections/05_toys_validation.tex.
- **Instruction:** Inspect compiled caption and patch if still ambiguous.

#### D44. Page 39 Equations 51-54: common notation for concatenation, n1||n2.
- **Status:** Needs notation cleanup if still present
- **Response:** Recommended response: use a single notation for concatenated vectors, e.g. (n_1, n_2) or n_1 || n_2, and define it once before the block of equations.
- **Pointer:** sections/04_methodology.tex or combined-likelihood equations.
- **Instruction:** Search compiled/source equations and standardize if still inconsistent.

#### D45. Page 31 line 665: why only 10% of 2016? EP: staged conservative, avoids biasing method in new parameter space.
- **Status:** Already addressed / recommended phrasing
- **Response:** Use the staged-analysis response: the 10% sample is a development and validation stream that avoids tuning the GP/limit machinery on the full 2016 sample before the first 100% unblinding. The note now places 2016 10% beside the 2021 1% validation stream in that staged logic.
- **Pointer:** sections/02_datasets.tex, staged-analysis paragraph and Table 1.
- **Instruction:** Closed after confirming subset bookkeeping.

#### D46. Page 32 Figure 16: stuff off scale; straight 3/5 sigma not helpful; figure not referenced.
- **Status:** Needs figure cleanup if still present
- **Response:** Recommended response: either reference the figure explicitly and rescale it so the relevant mass range is visible, or move it to an appendix and remove non-informative 3/5 sigma guide lines.
- **Pointer:** sections/05_toys_validation.tex, Figure 16 area.
- **Instruction:** Inspect compiled validation figures and patch if still crowded.

#### D47. Page 33 Figure 17: What is GPR?
- **Status:** Already addressed by primer
- **Response:** The primer now defines GPR before validation figures. If the caption still uses only the acronym, change the caption phrase to "Gaussian-process-regression (GPR) background prediction" on first use.
- **Pointer:** sections/04_methodology.tex; sections/05_toys_validation.tex.
- **Instruction:** Caption check after compile.

#### D48. Page 38 Figure 20: what is prefit; maybe preprocessing? sigmaA/ref diverges high mass; train2.25 optimality; can analytic calculation explain mass variation?
- **Status:** Partly addressed; recommended response needed
- **Response:** Recommended response: define "prefit" as the GP preprocessing/background prediction before signal injection/refit, or rename it to "preprocessing" if that is the intended meaning. Treat 2.25 sigma as closure-motivated rather than analytically optimal. The mass variation arises from changing resolution, binning, background slope/statistics, and active sideband geometry, not a constant Gaussian-containment calculation.
- **Pointer:** sections/05_toys_validation.tex, Figure 20 discussion.
- **Instruction:** Update caption/text if "prefit" remains unexplained.

#### D49. Page 39 Figure 21: pull widths suggest underconfident; discuss why/CLs impact; missing test of sigma_d uncertainty effects. EP asks for options.
- **Status:** Recommended response needed
- **Response:** Recommended response: do not call pull widths "near unity" if they sit below one. Say the tested configuration is mildly conservative in these toys, then point to coverage/CLs validation for the limit impact. Add a compact sigma_m shifted-template diagnostic before final if possible.
- **Pointer:** sections/05_toys_validation.tex; recommended-options section of this document.
- **Instruction:** EP decision: add sigma_m shift diagnostic now or list it as final-unblinding follow-up.

#### D50. Page 40 Figure 22: define error boxes/bars/horizontal lines.
- **Status:** Needs caption check
- **Response:** Recommended response: update the caption to define the filled boxes, vertical error bars, and horizontal reference lines in the first sentence. This is a direct cosmetic caption fix if still missing.
- **Pointer:** sections/05_toys_validation.tex, Figure 22 caption.
- **Instruction:** Inspect compiled caption and patch if still unclear.

#### D51. Page 46 line 776: broken sentence.
- **Status:** Already addressed or needs text search
- **Response:** Recommended response: the current source should be searched after compile; if the broken sentence remains, patch it directly. This is a straightforward prose cleanup.
- **Pointer:** sections/05_toys_validation.tex around the old support-check discussion.
- **Instruction:** Run source/compiled-text check for the broken phrase.

#### D52. Page 46 lines 781-: undefined support check, sideband-fraction check, configured Pearson target.
- **Status:** Needs definitions if still present
- **Response:** Recommended response: define the support check as the requirement that enough sideband bins remain after the mass-dependent exclusion, define the sideband-fraction check as the fraction of the nominal sideband retained, and define the configured Pearson target as the reference chi2-like residual threshold used in toy validation.
- **Pointer:** sections/05_toys_validation.tex, validation-check definitions.
- **Instruction:** Patch definitions if terms are still used before definition.

#### D53. Page 49 line 830: "pull widths are near unity within toy precision" is a stretch.
- **Status:** Recommended response / likely patch
- **Response:** Agree. Recommended wording: "The pull widths are below unity in this toy ensemble, indicating a mildly conservative predictive uncertainty in the tested configuration; the coverage and CLs checks are used to assess whether this conservatism biases the final limits."
- **Pointer:** sections/05_toys_validation.tex, pull-width paragraph.
- **Instruction:** Patch source if the old phrase remains.

#### D54. Page 53 Figure 32: not referenced.
- **Status:** Needs figure reference check
- **Response:** Recommended response: reference Figure 32 in the nearby validation text if it is useful; otherwise remove or move it to an appendix.
- **Pointer:** sections/05_toys_validation.tex.
- **Instruction:** Compile/source check for figure reference.

#### D55. Page 57 Figure 33: make blind window more visible in legend.
- **Status:** Needs cosmetic plot update if still present
- **Response:** Recommended response: update the plotting script to use a thicker/darker blind-window legend entry or a filled translucent band with a clear legend label.
- **Pointer:** Figure 33 plotting source under hps_gpr_analysis_note/scripts or generated figures.
- **Instruction:** Patch plot if figure remains in the main note.

#### D56. Page 58 Figure 34: not referenced.
- **Status:** Needs figure reference check
- **Response:** Recommended response: reference Figure 34 in text or move/remove it. The response should not claim this is closed unless the compiled source has a text reference.
- **Pointer:** sections/05_toys_validation.tex.
- **Instruction:** Compile/source check for figure reference.

#### D57. Page 59 Figure 35: q## and SEM not relevant?
- **Status:** Needs caption/plot cleanup
- **Response:** Recommended response: remove internal quantile/SEM labels from the publication-facing figure unless they are defined and used in the text. Replace with plain-language labels or move the diagnostic to an internal appendix.
- **Pointer:** Figure 35 plotting source/caption.
- **Instruction:** Patch if still visible in compiled note.

#### D58. Page 63 Figure 38: expect more impact from 2016 than 2015 at higher masses, seems opposite.
- **Status:** Needs analysis check
- **Response:** Recommended response: audit active ranges, efficiencies, luminosity/normalization, background density, and the 2015/2016 resolution/radiative inputs. If the effect is real, add a sentence explaining why; if not, regenerate the overlay.
- **Pointer:** sections/results or Figure 38 source.
- **Instruction:** Investigate before freezing result figures.

#### D59. Page 67 Figure 42: include previous results in (a),(b).
- **Status:** Needs plot update if final comparison figure remains
- **Response:** Recommended response: add published 2015 and 2016 curves to panels (a) and (b), or state that those comparisons are moved to the dedicated appendix figures.
- **Pointer:** sections/results or comparison plotting script.
- **Instruction:** Patch figure or cross-reference appendix.

#### D60. Page 69 Figure 45: above 210 MeV only 2021 contributes, so combined should equal 2021-only; why is it not?
- **Status:** Needs analysis fix
- **Response:** Agree. Recommended response: above the active 2015/2016 scan ranges, the combined curve should reduce to the 2021-only result if the same 2021 normalization and CL machinery are used. Audit active-dataset masks, density/radiative inputs, interpolation grids, and CL extraction. Do not freeze the figure until this is understood.
- **Pointer:** result-comparison plotting source; recommended-options section of this document.
- **Instruction:** High-priority follow-up before final result circulation.

#### D61. Page 71 Figure 47: vertical red dotted line?
- **Status:** Needs caption check
- **Response:** Recommended response: define the vertical red dotted line in the caption and text, or remove it. If it marks the dimuon threshold or active-range boundary, name that explicitly.
- **Pointer:** Figure 47 source/caption.
- **Instruction:** Patch caption after compile inspection.

#### D62. Page 72 Figure 49: panel (a) legend says 3-sigma line absent; remove 3/5 sigma from legend in panel (b).
- **Status:** Needs cosmetic plot update
- **Response:** Recommended response: regenerate the figure with legend entries matching only lines actually drawn, and remove the 3/5 sigma entries from panel (b) if those guides are not shown or useful.
- **Pointer:** Figure 49 plotting source.
- **Instruction:** Patch plot if figure remains in main note.

#### D63. Page 73 Figure 50: ideal 2021 resolution? 2015/2016 MC smeared; should expect limit degradation? Need rough 2021 data/MC perspective.
- **Status:** Needs EP/analysis decision
- **Response:** Recommended response: add a short 2021 data/MC mass-resolution perspective and, if possible, a sensitivity band from shifting sigma_m. The current validation note should not imply the 2021 resolution is final or ideal without that check.
- **Pointer:** sections/03_event_selection.tex and result figure discussion.
- **Instruction:** Same follow-up as D26.

#### D64. Page 74 Figures 51: state how/where radiative fraction and mass resolution differ from previous; possible comparison for 2015.
- **Status:** Needs comparison text / appendix
- **Response:** Recommended response: add a caption or paragraph stating which inputs changed relative to previous analyses: mass-resolution parameterization, radiative fraction treatment, active scan range, and density normalization. For 2015, point to the appendix direct comparison if included.
- **Pointer:** comparison figures and appendix_2015_method_comparison.tex.
- **Instruction:** Patch caption/text if not already explicit after compile.

#### D65. Page 75 Figure 52: direct comparison of 2015-only published data and this analysis. EP: appendix.
- **Status:** Already addressed / appendix path
- **Response:** The current source includes an appendix for the 2015 method comparison. Response: "A direct 2015-only comparison is included in the appendix so the main text stays focused on the combined workflow."
- **Pointer:** sections/appendix_2015_method_comparison.tex.
- **Instruction:** Closed if appendix figure compiles cleanly.

#### D66. Page 76 Figure 53: if comparing with BaBar at low mass add NA62. EP may compare.
- **Status:** Decision recommended
- **Response:** Recommended response: "For this validation note we keep the external comparison limited to the legacy benchmark curves already shown. If the low-mass external-comparison plot is retained as a physics-reach statement, we will add NA62 with a documented source and assumptions."
- **Pointer:** external-comparison/result figure source.
- **Instruction:** EP/committee choice: either add NA62 or explicitly limit the comparison scope.
