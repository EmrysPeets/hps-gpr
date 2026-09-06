# Independent scientific review of the candidate-removal section

6 September 2026. Read-only review of `note/candidate_removal_section.tex`, the enclosing `note/analysis_note.tex`, the saved numerical tables, and the figure-generating code. This review did not repeat fits or draw toys. Numerical and intervention details are in `HEP_INTERVENTION_RESULTS_REVIEW.md` and `independent_intervention_audit.json`.

**Scientific disposition:** the section accurately distinguishes local influence, surviving remote variation, conditional reference response, model dependence, and nominal selected-mass probabilities. No numerical or inferential blocker was found. Both requested minor metadata/reproducibility edits are resolved in the regenerated section. The final two-row captions match the paper-sized figures. No outstanding scientific corrections remain.

## Numerical statements checked

- The primary hole edges, native widths, resolutions, bin counts and all three full mass grids agree with the saved inputs. The count of 17,430 tests means 42 fixed complete spectra evaluated across 72, 142 and 201 mass hypotheses.
- The observed retained-variation fractions 0.673/0.778/0.921 and the reference fractions 1.102/0.928/0.865 match the independently reconstructed metrics. The ten-fill minima/maxima, alternate-filler fractions, reference correlations and counts of remote masses also agree.
- Remote classification is explicitly stated as disjointness of the **entire native fit window**, rather than the hypothesis center. The independent audit reconstructs those windows from the full observed histogram. The wider-hole comparison expressly retains the primary remote set and acknowledges overlap with extra removed bins.
- The illustrative deficit changes agree: 2015/19 from -3.21062 to -0.31409; 2016/102 from -4.49912 to -0.99810; 2021/71 from -4.01880 to -0.71538. All three are outside the remote metric. The 2021/65 change to +0.97988 after replacing only 78 MeV is also correct.
- Both conventional tables reproduce all 30 fixed variants and their rounded deviance ratios. The 2015/21 degree-plus deviance is **46.946220 for 49 degrees of freedom**, so the prose's **46.95/49 is correct**. The baseline is 92.497111/50 with nominal goodness-of-fit reference 2.42221e-4. The 2021/65 degree-minus D/dof=28.10485 and root 8.45923 are correctly paired.
- The numerical reconstruction counts and maximum root errors match the parent's independent validation report. The initial solver stall is disclosed as numerical, with unchanged statistical choices and final gates.

## Interpretation checked

The leading statement is supported when read with the explicit primary-replacement values and the subsequent filler-dependence discussion. The section does not claim an unchanged oscillatory function. It foregrounds the considerable local dip reduction and later distinguishes the strong change in the widened 2021 reference from the persistent observed variation.

The two uses of a GP remain distinct: a GP background interpolation modifies the count spectrum, whereas the earlier global-significance construction approximates a random field of fitted roots. Here the reference quantity is clearly defined as `a_m=r_m(B)`, not a background count rate. Modified spectra never receive probabilities from the original global-tail ensembles.

Ten conditional replacements are correctly identified as a limited interpolation/counting exercise with fixed exterior observations. Their range is explicitly not a confidence interval. Observed and reference spectra have separate fillers. The poor polynomial fills remain visible and are not selected as preferred physical truths. Deviance on the deterministic reference is used as a mismatch diagnostic, without a goodness-of-fit probability being assigned to it.

The conventional fits use the original observed bins. The positive exponential background, joint nuisance profiling, fixed Gaussian, native-bin integration, support shift and total window-width variations are described consistently with the implementation. The plotted isolated signal differs from the total-minus-null curve because the polynomial moves; residual counting bars omit fitted-background uncertainty and correlations, as disclosed.

The nominal formula `p0=sf(max(r,0))` and its 0.5 convention are correctly distinguished from calibrated or global probabilities. The 2015/21 baseline's formal small tail is immediately qualified by poor fit quality and model dependence. The 2021/65 degree-two result is explicitly identified as a background-model failure. No model is chosen by its p-value, and the same-event GP selection prevents treating this as an independent confirmation. A satisfactory deviance is not asserted to establish unbiased extraction or coverage.

The proposed extra 20% of 2021 is treated as a disjoint validation sample as well as part of the cumulative 30%, and the future correlated looks are acknowledged. Growth of either peaks or dips is not taken as unique evidence for a resonance. No claim is made that this derivative unblinded new events.

## Resolved minor corrections

1. The 2016 bibliography title now matches the primary source: *Searching for Prompt and Long-Lived Dark Photons in Electro-Produced e+e- Pairs with the Heavy Photon Search Experiment at JLab*. The arXiv identifier, author and journal locator were already correct. The verified source is [arXiv:2212.10629](https://arxiv.org/abs/2212.10629). The [2015 title and locator](https://arxiv.org/abs/1807.11530) already agree.
2. The section now states the alternate polynomial fill's construction: exterior sidebands within +/-7 resolutions, excluding both primary holes; degree five at 2015/21 MeV and degree three otherwise. This makes the method comparison readable without having to open `PROTOCOL.md`.

The regenerated section uses the paper-sized 2-by-2 count/residual displays and correctly revised two-row captions. The original full variant figures remain unchanged; the all-30 table retains every model. The added deterministic-reference deviance qualification and the 2021 single-region dip changes agree with the saved products. No numerical rerun was needed. Final whole-note pagination/rendering belongs to the parent; all six separate PDF exports passed rendered QA under `traditional/qa/`.

Final accepted candidate-removal section SHA-256: `6803c93d7dbc0606d154938db296897a80721b8226dfeaab14d45565140474eb`. `final_section_bindings.json` binds the source, bibliography, numerical tables and paper-figure QA reviewed here.

Final layout-only prose recheck: the shortened replacement-method paragraph, caption and wider-hole interpretation preserve the filler definitions, 38%/86% comparison, common remote-set caveat, distinction between reference roots and counts, and the prohibition on assigning new probabilities. Accepted without further numerical work.

Final display-only recheck: the three captions correctly describe whole-bin grouping (2015/51: three bins; 2015/21: one; 2016: five; 2021: one), unchanged native-bin inference, retained partial endpoint groups, and division by each actual width. The paper plots show stepped densities and counting errors sqrt(grouped counts)/width. Group sums are saved in `traditional/qa/paper_display_groups.npz` with their complete native-bin map; counts conserve exactly and predictions agree to floating-point roundoff. The parent's independent partition-matrix audit passes 198 checks and confirms all 60 original traditional fit products are unchanged. All original supplementary figures are unchanged. Updated paper PDF renders are accepted; no further scientific corrections or fits are required.
