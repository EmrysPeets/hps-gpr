# Independent statistics memo: Figure 100 functional-lane shoulder

Date: 2026-08-06
Scope: v4.2 2021 10% conditional 65 MeV replacement study only
Repository state inspected: `fb1295680bacdd5edbabff9546ee200e3c68b78a`

## Bottom line

The functional-form lane's peak just below 65 MeV is **not adequately described
as either a pure one-draw fluctuation or a deterministic functional-mean bias
alone**. It contains both:

1. a broad deterministic offset when the `fGenGammaThresh` truth is analyzed by
   the v4.2 GP procedure under the changing mass mask; and
2. an additional favorable Poisson fluctuation in the particular fixed-seed
   draw.

At 62 MeV, the reviewed functional draw has
\(\widehat A=23177.5\), \(\sigma_A=6288.4\),
\(p_0=1.12\times10^{-4}\), and \(Z=3.6896\). Replacing its randomized central
counts by the stored deterministic functional mean, while keeping the observed
outside-window bins and the v4.2 inference card unchanged, still gives
\(\widehat A=12118.7\), \(\sigma_A=6352.0\), and \(Z=1.9092\). The fixed-GP-mean
counterfactual gives only \(\widehat A=2750.1\), \(\sigma_A=6383.4\), and
\(Z=0.4314\). Thus, near 62 MeV, the functional truth adds about
\(9369\) fitted events (about \(1.47\sigma_A\)) relative to the GP-mean
counterfactual, and the realized functional draw adds another \(11059\) events
(about \(1.76\sigma_A\)).

This does not contradict the frozen v4.2 conclusion at the preselected 65 MeV
point: both randomized replacements have a negative signed estimate and
\(p_0=0.5\) there. It does mean that the lower-mass functional-lane shoulder in
Figure 100 must be labeled as an exploratory, conditional
truth-model/analysis-model stress feature amplified by one draw. It is not a
calibrated null probability, a global significance, or evidence that the
functional sideband fit failed its stated goodness-of-fit checks.

## Frozen v4.2 evidence

The appendix correctly records that:

- the two lanes are single conditional Poisson draws, not ensemble means;
- only `[60,70)` MeV is replaced, while all outside bins remain observed;
- the scan retains the v4.2 \(2.25\sigma_m\) extraction/training geometry;
- no expected bands or toys are produced; and
- full-scan minima are conditional fixed-mass diagnostics, not globalized
  significances.

These boundaries appear in
`hps_gpr_analysis_note/sections/appendix_v4p2_followups.tex`, especially lines
121–168 and 218–231.

The reviewed scan values around the feature are:

| Mass (MeV) | GP draw \(Z\) | Functional draw \(\widehat A\) | Functional draw \(\sigma_A\) | Functional draw \(p_0\) | Functional draw \(Z\) |
|---:|---:|---:|---:|---:|---:|
| 59 | 1.7725 | 5,757.1 | 6,005.4 | 0.1686 | 0.9598 |
| 60 | 1.8530 | 14,443.1 | 6,026.0 | 0.00823 | 2.3985 |
| 61 | 1.4391 | 21,474.1 | 6,265.3 | \(3.01\times10^{-4}\) | 3.4309 |
| 62 | 0.6592 | 23,177.5 | 6,288.4 | \(1.12\times10^{-4}\) | 3.6896 |
| 63 | 0 | 20,219.2 | 6,386.1 | \(7.69\times10^{-4}\) | 3.1675 |
| 64 | 0 | 9,351.0 | 6,475.5 | 0.07425 | 1.4449 |
| 65 | 0 | -2,195.6 | 6,603.0 | 0.5 | 0 |

Source ledgers:
`derived/gp_mean_results_reviewed.csv` and
`derived/functional_form_results_reviewed.csv`. Their SHA-256 values are
`7ff22bb70d7ee9c0387d20c66b6c20fd359d80af3ff7e51303db607cd88efb77`
and
`4ef284c894d8ad6be65fefc0b063cf6100add4d6fe735935e6a90363f7ad7ca1`.

## Deterministic-mean counterfactual

I performed a read-only, in-memory counterfactual using the frozen ROOT input
(`c5ea3922ddb70164f6184a8661d803a6d82302747c0d213c3a37bcab31be11ab`):

1. copy the original observed native-bin spectrum;
2. in `[60,70)` MeV, replace counts by the nearest integer to either stored
   expectation `expectations/gp_mean_m065` or
   `expectations/fGenGammaThresh_m065`;
3. retain every outside-window bin unchanged;
4. run the same observed/asymptotic v4.2 card with signed extraction; and
5. compare unchanged-card repetitions by maximum finite GP log marginal
   likelihood, with no interpolation.

The resulting decomposition is:

| Mass (MeV) | Fixed-GP mean \(\widehat A/\sigma_A\) | Fixed-GP mean \(Z\) | Functional mean \(\widehat A/\sigma_A\) | Functional mean \(Z\) | Functional draw \(\widehat A/\sigma_A\) | Functional draw \(Z\) |
|---:|---:|---:|---:|---:|---:|---:|
| 58 | 4,389.4 / 5,974.7 | 0.7348 | 711.6 / 5,947.3 | 0.1201 | -4,760.1 / 5,908.9 | 0.0002 |
| 59 | 7,476.0 / 6,050.0 | 1.2358 | 7,602.0 / 6,029.3 | 1.2613 | 5,757.1 / 6,005.4 | 0.9598 |
| 60 | 7,734.9 / 6,076.1 | 1.2751 | 12,185.9 / 6,053.0 | 2.0139 | 14,443.1 / 6,026.0 | 2.3985 |
| 61 | 6,021.0 / 6,329.8 | 0.9520 | 14,301.4 / 6,301.1 | 2.2685 | 21,474.1 / 6,265.3 | 3.4309 |
| 62 | 2,750.1 / 6,383.4 | 0.4314 | 12,118.7 / 6,352.0 | 1.9092 | 23,177.5 / 6,288.4 | 3.6896 |
| 63 | 1,492.0 / 6,467.0 | 0.2316 | 8,904.4 / 6,440.0 | 1.3833 | 20,219.2 / 6,386.1 | 3.1675 |
| 64 | 25.0 / 6,501.1 | 0.0037 | 2,849.3 / 6,486.0 | 0.4394 | 9,351.0 / 6,475.5 | 1.4449 |
| 65 | 1.6 / 6,603.4 | 0 | -2,839.4 / 6,603.1 | 0 | -2,195.6 / 6,603.0 | 0 |
| 66 | 1,249.0 / 6,507.7 | 0.1924 | -6,365.1 / 6,523.9 | 0 | -12,136.6 / 6,528.7 | 0 |
| 67 | 920.8 / 6,622.3 | 0.1343 | -9,021.8 / 6,646.2 | 0 | -19,852.8 / 6,646.3 | 0 |
| 68 | 1,084.1 / 6,638.2 | 0.2225 | -8,380.0 / 6,660.7 | 0 | -22,538.6 / 6,713.2 | 0 |

The deterministic functional scan therefore has a broad, one-sided shoulder
that rises from \(Z=1.26\) at 59 MeV, peaks at \(Z=2.27\) at 61 MeV, and falls
to the bounded-null branch by 65 MeV. The deterministic fixed-GP scan has only
a smaller maximum, \(Z=1.28\) at 60 MeV, over the same 58–68 MeV interval.
The displayed functional draw shifts the local maximum from 61 to 62 MeV and
raises it from \(Z=2.27\) to \(Z=3.69\).

Expressed as the end-to-end signed-extraction projection of the deterministic
functional mean relative to the deterministic fixed-GP mean, the shifts at
60, 61, 62, 63, and 64 MeV are respectively about
\(+0.74,+1.31,+1.47,+1.15,+0.44\) fitted standard deviations. The shift changes
sign at 65 MeV. This coherent mass dependence is the direct evidence that the
functional interpolation itself induces a shoulder under the GP analysis;
the larger \(Z=3.69\) peak still requires the additional realized Poisson
fluctuation.

Focused repeated fits reproduced the selected deterministic functional result at
62 MeV: five maximum-LML attempts gave
`LML=1657.2054778–1657.2054795`,
\(\widehat A=12118.2–12118.7\), and
\(Z=1.9091–1.9092\). This matters because an additional attempt found a much
worse optimizer branch; the table uses the reproducible maximum-LML state.

The two deterministic expectations are globally very close in the replaced
region: their totals differ by only 2,301 events out of 15.41 million
(0.0149%), and their largest individual 0.625 MeV-bin difference is
0.72 Poisson standard deviations. The difference is nevertheless smooth and
coherent across bins, so its signal-template projection need not be small. The
particular functional draw then has pulls of \(+1.38\) and \(+1.88\) relative
to its own mean in the two analysis bins centered at 62.1875 and 62.8125 MeV;
the GP draw has \(+0.12\) and \(-0.51\) in the same bins.

As an edge diagnostic, I also replaced the entire fitted functional interval
`[50,85)` MeV by its deterministic mean. The shoulder did not disappear:
\(Z=2.5091\) at 61 MeV and \(Z=2.1849\) at 62 MeV. For comparison, replacing
the same wider interval by the accepted fixed-GP mean gave \(Z=0.5759\) and
\(Z=0.6414\). Therefore, the finite 60 and 70 MeV replacement boundaries are
not the sole cause. The result is consistent with a smooth functional truth
that is not perfectly reproduced by the masked GP analysis, plus retained
conditional outside data and the realized Poisson fluctuation.

The functional fit itself still passes its declared sideband QC:
deviance/ndf 1.0690, Pearson \(\chi^2/\mathrm{ndf}=1.0690\),
deviance \(p=0.2430\), five near-optimum starts, and no selected parameter at a
declared bound. That QC tests the functional description of its fitted
sidebands. It is not a guarantee that a GP trained with changing exclusion
windows will give zero signed signal estimate for that truth at every mass.

## Exact replacement-window geometry at 65 MeV

With \(\sigma_m(65)=2.121696875\) MeV and the production 0.625 MeV analysis
grid:

| Half-width | Continuous interval (MeV) | Selected analysis-bin centers (MeV) | Complete-bin edges (MeV) | Analysis/native bins |
|---:|---:|---:|---:|---:|
| \(2.25\sigma_m\) | [60.226182, 69.773818] | 60.3125 through 69.6875 | [60.00, 70.00) | 16 / 80 |
| \(2.5\sigma_m\) | [59.695758, 70.304242] | 60.3125 through 69.6875 | [60.00, 70.00) | 16 / 80 |
| \(3.0\sigma_m\) | [58.634909, 71.365091] | 59.0625 through 70.9375 | [58.75, 71.25) | 20 / 100 |

Consequences:

- \(2.25\sigma_m\) and \(2.5\sigma_m\) are the **same binned replacement**
  at 65 MeV when complete 0.625 MeV analysis bins are honored. Running both
  would duplicate the same pseudo-observation; any differences would be
  optimizer/numerical variation, not a window-scope effect.
- \(3\sigma_m\) adds four analysis bins, or 20 native bins. Relative to the
  accepted fixed-GP mean, the currently observed counts in those added bins
  have pulls \(+0.71,+0.02,+1.53,-0.80\), in ascending mass order. Relative
  to the functional mean they have pulls \(+0.07,-0.69,+2.04,-0.38\).
- If the inference card remains at the frozen v4.2 \(2.25\sigma_m\), those
  four additional replaced bins are training-sideband inputs at 65 MeV and
  can move the GP prediction. If the inference window is also changed to
  \(3\sigma_m\), the observed vector, covariance, signal-template truncation,
  and training mask all change. These are two different questions and must
  not be combined into one unlabeled comparison.

A small deterministic check found no 65 MeV excess from the wider complete-bin
replacement: with a \(3\sigma_m\) replacement and the v4.2 inference card,
the fixed-GP truth gave \(Z=0.362\) and the functional truth \(Z=0\).
With matched \(3\sigma_m\) replacement and inference geometry, they gave
\(Z=0.419\) and \(Z=0\), respectively. These are diagnostics, not a validation
of a changed inference card.

## Recommended ten-draw ensemble design

Ten draws are useful as a **recurrence and mechanism screen**, not as a tail
calibration. The minimum defensible design is ten fresh draws **per truth**;
ten total split between the two truths is too sparse.

### Primary design

1. Truths: (a) exact accepted v4.2 fixed-GP mean at 65 MeV and
   (b) the frozen sideband-only `fGenGammaThresh` mean.
2. Unique complete-bin replacement geometries:
   `[60,70)` MeV (the shared \(2.25/2.5\sigma_m\) geometry) and
   `[58.75,71.25)` MeV (the \(3\sigma_m\) geometry).
3. Generate ten independent background-only draws for each truth:
   \(A_{\rm inj}=0\), independent binwise Poisson counts, outside the selected
   replacement window bitwise unchanged. This is 40 conditional
   pseudo-observations and 40 primary full scans.
4. Use fresh streams that do not include the already inspected Figure 100
   draw. One exact convention is
   `SeedSequence(20260807, spawn_key=(truth_index, draw_index))`, with
   `truth_index=0` for GP and `1` for functional,
   `draw_index=0,...,9`, and `PCG64`.
5. For each truth/draw, generate the maximal 100-native-bin vector once.
   The `[60,70)` version is its central 80-bin subset. This common-random-number
   nesting makes the two window scopes directly comparable without changing
   shared-bin fluctuations. The GP and functional truth streams remain
   independent.
6. Keep the inference card fixed at v4.2 \(2.25\sigma_m\) for the primary
   window-scope comparison: 50–250 MeV in 1 MeV steps, 0.625 MeV analysis
   bins, \(k_{\max}=15\), profiled correlated-background extraction,
   observed/asymptotic 90% CLs, local bounded asymptotic \(p_0\), and no limit
   bands.
7. Before random draws, run one deterministic-mean counterfactual for every
   truth/geometry pair. This identifies structural offsets that should not be
   attributed to random recurrence.
8. Compare two unchanged-card optimizer attempts per scan; repair discrepant
   masses by unchanged-card repeats and select the reproducible maximum finite
   LML. Record every branch and never interpolate.

An optional secondary analysis can apply \(3\sigma_m\) inference to the same
\(3\sigma_m\) pseudo-observations (20 additional scans). It must be labeled an
inference-geometry sensitivity test, separate from the primary replacement
scope study.

### Metrics fixed before opening the new draws

For each truth and geometry, record all ten values—not only averages—for:

- \(\widehat A/\sigma_A\), local \(p_0\), and observed
  \(\epsilon^2_{90}\) at the preselected 65 MeV point;
- the same quantities at 62 MeV, explicitly labeled a post-Figure-100,
  predeclared diagnostic for the fresh ensemble;
- \(Z_{\max}^{60\text{--}65}=\max_{m=60,\ldots,65\ {\rm MeV}}Z(m)\) and its
  mass; and
- paired differences between the two replacement scopes for each
  truth/draw.

For the functional truth, report the count \(K\) of fresh draws satisfying
\(Z(62)\ge3.689615\), and separately the count satisfying
\(Z_{\max}^{60\text{--}65}\ge3.689615\). Give the exact binomial count and a
Clopper–Pearson interval, but do not convert it to a Gaussian significance.
With only ten draws, even \(K=0\) has a 95% upper confidence limit of 0.308;
\(K=1\) has a 95% interval of [0.0025, 0.445]. Report the median and full range
as descriptive summaries, with any empirical 16th/84th percentiles clearly
labeled as coarse order-statistic summaries rather than calibrated bands.

## Interpretation boundaries for ten draws

Ten conditional draws can test:

- whether the lower-mass shoulder repeatedly appears for the functional truth
  but not the fixed-GP truth;
- how much of the feature is already present in deterministic-mean scans;
- whether widening the replacement scope changes the local result; and
- whether the 65 MeV disappearance is robust to several replacement
  fluctuations under the two declared truths.

They cannot establish:

- a global null \(p\)-value or look-elsewhere correction;
- calibration of the analytic \(p_0=1.12\times10^{-4}\) tail;
- expected sensitivity, expected limit bands, coverage, or discovery power;
- the probability that the original 65 MeV feature is background; or
- that either smooth truth is the physical data-generating process.

The smallest nonzero empirical tail fraction from ten draws is 0.1. A
\(10^{-4}\)-scale tail would require of order \(10^5\) properly generated
full-support null pseudoexperiments for even modest direct tail statistics,
with the complete GP procedure refitted in each experiment. Because the
recommended screen retains observed data outside a narrow replacement window,
it remains a conditional counterfactual even if the number of central draws is
increased.

## Recommended Figure 100 wording

> The broad 61–63 MeV feature in the functional-replacement lane is an
> exploratory conditional stress-test feature. A deterministic functional-mean
> counterfactual already produces a smaller shoulder under the masked GP
> analysis, and the displayed single Poisson draw amplifies it. The local
> asymptotic \(p_0\) values are not ensemble- or scan-calibrated and do not
> constitute a global-null probability.
