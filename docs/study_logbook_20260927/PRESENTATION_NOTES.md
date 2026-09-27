# A 12-minute route through the recent studies

Use eight plots at about 90 seconds each. The [plotbook](../../output/pdf/study_logbook_20260927/HPS_GPR_Recent_Studies_Plot_Logbook.pdf) has these IDs and a bookmark for every figure. The full-resolution PNG and vector PDF files are in [assets](assets); the [illustrated guide](PLOTBOOK.md) supplies the exact source and reading notes for each.

| Time | Plot | Point to make |
|---|---|---|
| 0:00-1:30 | [P09](PLOTBOOK.md#p09), native MC distributions | A Gaussian core summary does not describe all selected signal-MC probability. |
| 1:30-3:00 | [P12](PLOTBOOK.md#p12), fixed-yield recovery | A common expected yield makes shape comparisons readable; raw recovery and paired response differ. |
| 3:00-4:30 | [P17](PLOTBOOK.md#p17), clean-training control | The background estimate absorbs signal entering its training bins. |
| 4:30-6:00 | [P20](PLOTBOOK.md#p20), 2016 core positions | Measure the response by campaign; do not transfer the larger 2021 shift law. |
| 6:00-7:30 | [P22](PLOTBOOK.md#p22), 2016 window response | Lower observed limits can accompany lower signal response. |
| 7:30-9:00 | [P19](PLOTBOOK.md#p19), joint observed scan | The updated 2021 template changes the common-coupling local comparison. |
| 9:00-10:30 | [P23](PLOTBOOK.md#p23), independent global calibration | State the declared scan statistic, independent ensembles, count and interval. |
| 10:30-12:00 | [P25](PLOTBOOK.md#p25), mass correlations | Coherent full-scan toys preserve dependence that a simple trials count misses. |

## Suggested opening

“The recent studies ask whether the extracted signal responds correctly to the shapes actually present in the selected MC, and how to interpret the resulting mass scans. The key tests are paired signal injections, controls on signal contamination in GP training, and a global calculation that retains correlations between neighboring masses.”

## Statements to use with the plots

**P09:** “These colored curves are empirical selected MC distributions. Fitting their cores gives a center and width, but does not remove the tails. The full selected probability remains the normalization for the later signal-injection studies.”

**P12:** “The independent Gaussian pilot sets one expected yield for both signal shapes. At input z=5, the Gaussian paired response is about 96-98%, whereas native MC is about 72-89% over the tested masses. Here z specifies the injected expected count relative to a pilot uncertainty; it is not a measured discovery significance.”

**P17:** “We can isolate training contamination in the toys by removing only the injected signal in the training bins. The signal-fit counts do not change. At 100, 160 and 220 MeV the direct-MC response moves from roughly 0.96, 0.95 and 0.91 to approximately one. This is a diagnostic control; real data do not identify which training events are signal.”

**P20:** “The 2016 samples are already FEE-smeared and scaled. Their fitted cores lie about 0.08-1.02 MeV below the generated mass over the qualified range. That is a smaller shift than in the saved 2021 comparison, so a common correction across campaigns is not justified.”

**P22:** “Narrowing the 2016 window reduces the median observed limit by about 9.6%, but it also reduces recovery of the same injected full signal. Once the background spread is divided by the measured response, the two procedures differ by only about zero to three percent in this bounded check. That is why a lower raw limit alone is not evidence of better sensitivity.”

**P19:** “This earlier observed comparison changes only 2021 and uses one coupling across the active campaigns. Its MC-method local minimum is at 68 MeV with asymptotic Z=2.91. This is a local result under that version's model. The following global study uses its own explicitly fixed 2016 and 2021 MC windows.”

**P23:** “For the later local-first global test, 1,024 A scans set the local rank maps. A separate 1,024 B scans test the minimum mapped probability across the declared grid. The common-coupling count is 75 of 1,024, giving an add-one global probability of 0.074 and a conditional 95% interval from 0.058 to 0.091. The combined local map reaches its finite-simulation floor, and ties in B are counted.”

**P25:** “Neighboring fits share data, masks and broad signal components. Whole-scan toys retain these dependencies. Correlations reduce the penalty relative to an empirical independent-mass control, but a simple resolution-count formula reduces it too far at the observed single-year thresholds.”

## Useful backup material

- **Background validation:** P06-P07 and the [latest saved slide-reading guide](../../output/slides/unblind_meeting_RCmeet_20260923b/READING_SLIDES_13_14.md). Pointwise fitted-sideband agreement does not validate predictions in the excluded window.
- **Null bias:** P05. A positive scan maximum and a fixed-mass signed-root offset are different effects.
- **Injection design:** P10-P13. v6.2 uses exact-N injections and its final 40-toy release; v6.3.1 uses fixed expected yield with fluctuating Poisson totals.
- **Calibration:** P15 and the [2016 transfer study](../../study_results/v6p3p2_2016_offset_transfer_20260924/README.md). Offset, response and uncertainty width must remain separate, and same-source calibration need not transfer.
- **Global approximation:** P24-P26. Effective trial counts depend on threshold; fitted Sidak curves are comparisons, not substitutes for the independent B count.
- **Kernel and limit explanations:** [saved 30/60/90-second kernel scripts](../../output/slides/unblind_meeting_RCmeet_20260923/SPEAKER_SCRIPTS.md) and [latest slide 29 conversion explanation](../../output/slides/unblind_meeting_RCmeet_20260923b/SPEAKER_NOTES.md).

## Version and interpretation checks before presenting

The original reference mask is +/-2.25 reference-resolution sigma. The v6.4.1 main 2016 extraction uses +/-3.5 fitted-core widths; v6.4.2 compares +/-2 against that. The v6.4.3-v6.4.5 global protocol instead fixes 2016 at +/-2.5u and 2021 at [-4,+3]u. Do not assign a probability from one of these versions to an observed curve from another.

All stated global probabilities condition on frozen observed-derived null sources, fixed kernel/template/window choices, and a finite 1 MeV mass grid. The A/B result also conditions on its frozen empirical local maps. It omits uncertainty from estimating those sources/maps and earlier analysis choices. This body of work supplies conditional inference and diagnostic evidence, not a discovery, unconditional coverage certificate or newly validated physical exclusion.
