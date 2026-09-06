# v4.9.16: signal extractions and the 30% checkpoint

[Read the complete 23-page analysis note](../../output/pdf/v4p9p16_presentation_extractions_20260906/HPS_GPR_Analysis_Note_v4p9p16_Signal_Extractions.pdf). The new section begins on page 4. The original full-region limit/p-value figure and deficit scan remain on pages 2 and 3.

## Figures for presentation

| Display | PDF | Main point |
|---|---|---|
| Leading combined excess, 66 MeV | [Figure](figures/extraction_combined_66.pdf) | All three separate amplitudes are positive; current combined raw root +2.76. |
| Second full-region excess, 21 MeV | [Figure](figures/extraction_combined_21.pdf) | Only 2015 contributes; no benefit from opening 2021 here. |
| Next peak with multiple datasets, 92 MeV | [Figure](figures/extraction_combined_92.pdf) | 2016 and 2021 prefer different positive rates. |
| Combined deficit, 72 MeV | [Figure](figures/extraction_combined_72.pdf) | Signed root -3.49; an auxiliary missing-event template. |
| 2015 peaks: 51 and 21 MeV | [Figure](figures/extraction_2015_peaks.pdf) | Two leading separated observed profiled maxima. |
| 2016 peaks: 90 and 117 MeV | [Figure](figures/extraction_2016_peaks.pdf) | Existing 2016 qualifications remain. |
| 2021 peaks: 78 and 65 MeV | [Figure](figures/extraction_2021_peaks.pdf) | Released native 10% only. |
| Strongest individual deficits | [Figure](figures/extraction_individual_deficits.pdf) | 2015 19 MeV endpoint, 2016 102 MeV, 2021 71 MeV. |
| Rate consistency | [Figure](figures/dataset_amplitude_consistency.pdf) | Independent amplitudes versus a shared fit. |
| Stress-centered extrema, 76 and 83 MeV | [Figure](figures/extraction_stress_extrema.pdf) | Extreme conditional tails can coexist with small fitted signals. |
| 10% → new 20% → cumulative 30% → 100% | [Figure](figures/exposure_2021_10_30_100.pdf) | Independent increment separated from conditional cumulative means. |

Each figure has a PNG of the same basename. Native whole-bin grouping is set by the mass resolution and original histogram origin. The inference uses the unchanged native-bin likelihood. The combined display sum uses the common window; its retained counts and signal fractions are exported in `derived/common_display_retention.csv`.

## Scientific interpretation

The displays identify follow-up locations, not a particle discovery. At 66 MeV all three individual fitted amplitudes are positive. At 92 MeV the individual positive rates differ appreciably. At 76 MeV 2016 pulls negative and 2021 positive, yielding a near-zero common amplitude despite an extreme stress-centered tail. The deepest raw combined deficit is 72 MeV; the strongest stress-centered deficit is instead 83 MeV.

The exposure illustrations use the selected common rates at 66 and 92 MeV and are explicitly conditional. No future events or new toys were generated. The additional 20% by itself is the independent next check; cumulative 30% retains the original selection sample. The 30% and 100% looks are correlated. The precision table assumes the entire 2021 count covariance scales with exposure; it is not an expected-sensitivity or global-significance forecast. See [continuation instructions](NEXT_STEPS.md).

## Evidence and provenance

- [Independent HEP review](review/HEP_EXTRACTION_REVIEW.md): 2,333 checks passed, including exact objective/root reconstruction, nuisance gradients, local curvatures, bin maps, count conservation, rate consistency, exposure means and variances.
- [Product validation](qa/product_validation.json) and [rendered review](qa/visual_review.json).
- [Exact arrays and grouping maps](derived/fit_arrays.npz), [fit summary](derived/fit_summary.csv), [per-year amplitude estimates](derived/dataset_consistency.csv).
- [LaTeX source](note/analysis_note.tex), [new section](note/extraction_section.tex), [protocol](PROTOCOL.md).
- [Parent backup receipt](provenance/parent_backup.json): prior studies merged in [PR 67](https://github.com/EmrysPeets/hps-gpr/pull/67); all 4,778 scoped file blobs verified on the merged branch. The shared checkout HEAD/index were preserved.

`MANIFEST.csv` seals this derivative and its final PDF, excluding the manifest itself and its verification record. The frozen parents are checked in full. Rebuilding requires fresh numerical/product validation and rendered review; do not silently replace a sealed result. Reproduction commands are in `NEXT_STEPS.md`.
