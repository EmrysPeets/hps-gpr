# HPS-GPR analysis note v4.9.16

This is the completed combined search of full 2015, full 2016 and native
2021 10% over all 232 integer masses from 19 through 250 MeV.

- [Analysis note PDF](../../output/pdf/v4p9p16_combined_global_20260906/HPS_GPR_Analysis_Note_v4p9p16_Combined_Global_Search.pdf)
- [Full observed limit with local and global p-values below](figures/combined_observed_limit_and_pvalues.pdf)
- [LaTeX source](note/analysis_note.tex)
- [Representative p-value table](global/analysis/representative_pvalues.csv)
- [Full observed upper-limit table](global/observed.csv)
- [Independent HEP review](review/HEP_COMBINED_REVIEW.md)
- [Reproduction and continuation](NEXT_STEPS.md)

The upper curve is the pointwise asymptotic 90% CLs result. The new GP method
estimates global discovery-score tails under one joint stress-background scenario;
it does not make the upper-limit curve toy-calibrated or simultaneous.
Legacy expected bands use a different ensemble and are omitted.

## Full search and probabilities

The active sets are 2015 alone at 19–38 MeV, 2015+2016 at 39–49,
all three at 50–90, 2016+2021 at 91–180, and 2021 alone at 181–250.
These choices follow support and are fixed before the combined results.
Every multi-dataset mass uses the actual shared-coupling likelihood.

| Statistic | Principal peak [MeV] | Local common-background p | GP global p | Direct count |
|---|---:|---:|---:|---:|
| profiled | 76 | 6.9826e-20 | <1.4979e-05 (one-sided 95% bound) | 0/1000 |
| fixed | 76 | 3.6651e-13 | <1.4979e-05 (one-sided 95% bound) | 0/1000 |

The separate raw-peak ordering is retained in the note and figure. It is a
different test, not a competing numerical estimate of the same probability.
Zero-count tails are bounds. These conditional results do not establish
discovery evidence, physical background validity, expected sensitivity or
confidence-interval coverage. The inherited 2016 qualifications remain.

## Reuse and numerical work

The study uses ten pilot and 1,000 independent validation joint experiments.
They pair equal row IDs from distinct year-specific full-spectrum RNG streams,
reusing existing spectra without counting copies as new experiments. The
1,626-bin response basis preserves shared-data correlations across membership
boundaries. It contains 1,627 Asimov spectra including the baseline.

There are 142 newly fitted multi-dataset coordinates and 90 reused, validated
single-dataset coordinates. Each new coordinate has an exact pilot and
response stencil. Six complete exact response columns test the boundary
regions. The analyzer samples 200,000 GP fields per method over the whole
union at once; it does not join independent segment maxima.

Final acceptance includes 2736 numerical/product identity
checks, 7744 independently implemented checks,
42 report/probability checks, and visual review of all
8 PDF pages. Earlier manifests, the shared Git HEAD and
the index were preserved. No commit, push or merge was made for this new note.

## Artifact map

- global/points: per-mass joint or reused root vectors and numerical audits.
- global/references: paired exact pilot and response reference vectors.
- global/spectra: coherent joint pilot, validation and Asimov spectra.
- global/analysis: p-values, covariance, GP/direct maxima and diagnostics.
- provenance: input, figure and report bindings.
- qa and review: numerical, semantic, independent and rendered checks.

MANIFEST.csv covers this derivative and the final PDF. The manifest and its
self-verification companion are excluded from their own inventory.
