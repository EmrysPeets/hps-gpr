# Statistics review

Author role: statistics specialist. Reviewed the actual archived implementation,
not only the report prose. Kept the ±2.25 sigma extraction window and did not
write to any parent study.

## Main judgement

The background GP and significance-field GP need not be independent. The latter
should reproduce the null response of the entire likelihood analysis, including
the former's sideband learning. Independence of Monte Carlo validation draws
tests that approximation conditional on a selected generating background; it
does not independently establish that the source describes physical background
or qualify choices informed by the observed results. A plug-in background fit
is not automatically invalid, but unconditional size, source uncertainty,
signal absorption and method selection remain unqualified here.

Reference standardization is a change of test ordering, not an obligatory part
of LEE. The raw-root maximum can be calibrated directly under a chosen null.
Neither raw ordering nor centered ordering should be selected after comparing
their observed probabilities. A raw root kept unchanged still need not have a
standard-normal marginal null distribution.

The arXiv paper's less-conservative claim compares GP sampling to the upcrossing
upper bound for the same field. It is not a guarantee that a data-analysis field
has a smaller trial penalty than a resolution-spacing Sidak heuristic.

## Inspected evidence

- [Ananiev and Read, arXiv:2206.12328v3](https://arxiv.org/pdf/2206.12328),
  especially sections 1, 2, 2.1 and 2.2. Source-derived prose in the report is
  approximately 90 words, below the source limit; new equations and numerical
  comparisons are applied derivations for this analysis.
- v5.8.2 `scripts/prepare.py`: full-support unmasked GP source with reviewed
  source kernel anchored at 76 MeV; fixed conditional Poisson samples.
- v5.8.2 `scripts/engine.py`: deterministic `fit(B)`, derivative including
  GP mean and count-dependent covariance, and blind extraction at ±2.25 sigma.
- v5.8.2 `scripts/run_scan.py`: `a=aa['r']`, `s=norm(D)`, unchanged observed root
  and signed reference score; complete coherent Poisson scans by shared toy ID.
- v5.8.2 `scripts/analyze.py`: Gaussian response generation, positive raw-fit
  gate, saved raw maxima, saved centered maxima and direct Poisson maxima.
- v5.8.3 `scripts/resolution_audit.py`: resolution/FWHM heuristics, template
  overlap, same-threshold curves, effective dimension versus tail count.
- All four v5.8.2 saved field NPZs; hashes are in `results/stats_summary.json`.

The study-history specialist independently checked the analysis-note sections
and old figure numbering. The physics specialist independently audited source
absorption and the candidate 2021 high-psum 1% histogram. We exchanged findings
before the report sections were written.

## New calculations

`scripts/correlation_diagnostic.py` uses copied inputs first, has a STOP-file
check and a four-minute internal cap, uses one BLAS thread, does not fit spectra,
and leaves the Gaussian covariance unchanged. The principal run took 4.4 seconds
before final figure output; machine runtimes vary.

### Raw versus centered 2016 maxima

| Test | Mass | raw r | a | s | Conditional local p | Gaussian global p | Poisson global |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw maximum | 90.5 | 3.452500 | .712849 | .982030 | .00263716 | 13559/200000=.067795 | 18/256=.0703125 |
| Reference maximum | 91.5 | 3.333160 | .496739 | .982836 | .00195114 | 20975/200000=.104875 | 36/256=.140625 |

Direct global Clopper-Pearson 95% intervals are [.042201,.108849] and
[.100475,.189333], respectively. Local direct toys have 0/256 and 2/256
exceedances, so cannot settle the precise local tails. Zero does not mean a zero
probability; the first two-sided 95% upper endpoint is .0143064.

The conventional normal mapping of raw r=3.4525 is .000277709. Comparing that
directly to a global tail from the nonzero-mean fixed-source response combines a
marginal-calibration difference with search multiplicity. All scopes and both
orderings appear in `results/stats_raw_and_reference.csv`, with raw counts,
k/N, add-one estimates and binomial intervals separately labeled.

### Analytic finite-grid upcrossing bound

For a standardized Gaussian pair with correlation rho, the probability of an
upcrossing of u is `2*owens_t(u,sqrt((1-rho)/(1+rho)))`. Summing adjacent
crossing probabilities and adding `norm.sf(u)` gives an upper bound on the
Gaussian-grid maximum tail (clipped at one). This follows from the elementary
event inclusion; no continuum derivative approximation is required.

At each reference peak, saved maximum p and same-grid bound are:

| Scope | Saved p | Bound |
|---|---:|---:|
| 2015 | .023450 | .0252732 |
| 2016 | .104875 | .1203781 |
| 2021 | .333520 | .4268903 |
| combined | .229525 | .2734026 |

All plotted thresholds exceed the maximum positive-fit gate (<1.5 for every
scope), so no extra gate correction is hidden in these comparisons. These
numbers demonstrate the direction expected from the paper using the same
actual HPS response covariance.

### Rho<0.1 representatives and regions

2016 has 283 nodes. Selecting each next mass by its low absolute correlation
with the last selected mass yields 24 representatives, but correlations between
more distant representatives reach |rho|=.834560. Requiring low absolute
correlation with *all* selected points gives 6 (low-to-high) or 7 (high-to-low),
demonstrating order dependence of the greedy approximation. It does not
demonstrate a fundamental count of independent regions. The minimum adjacent
grid rho is .888996; any nontrivial contiguous partition leaves strongly
correlated masses across its boundary.

A midpoint partition around the 24 first-rule representatives was specified
without looking at the observed score heights. Each block uses its actual
maximum. With 500000 new same-covariance draws at u=2.88595489:

- Full grid: 53201/500000=.106402, CP95% [.105549,.107260].
- Independent-product combination of *block maximum* tails: .143410.
- Treat all 24 blocks as one normal point each: .0457915.
- Treat six all-pairs-selected points as normal tests: .0116499.

Thus the point heuristic undercounts the search, while an independent-block
maximum product overcounts it in this particular partition. No universal sign
of approximation error is claimed. The joint field already computes the
desired union probability without a decorrelation threshold.

## Sampling QA and reproducibility

The first 100000 draws of the selected seed had .10910 tail probability,
noticeably above the saved .104875. This was investigated rather than accepted
as routine agreement. The covariance factor reconstruction error is
6.88e-15, minimum eigenvalue .000198643, and the gate threshold is only 1.41290.
The first 100000 sample means have max |mean|=.00733, SD ranges
.99783–1.00683, and correlation RMSE=.003029. Four independent 100000-draw
checks give .10514, .10675, .10553 and .10468. Extending the initial seed to
500000 yields .106402, differing from the frozen 200000 estimate by 1.8803
combined Monte Carlo standard errors; the 95% intervals overlap.

The reported block probabilities use only that final nested 500000 sample.
The independent checks and repeated prefixes are validation computation, not
additional observations or a pooled significance estimate. In total, 900000
unique new Gaussian vectors were examined; repeated verification regenerated
prefixes, for 1600000 Gaussian vectors generated across command attempts.
The standalone script reconstructs the principal 500000-draw result; the
one-time diagnostics are recorded in `results/stats_sampling_qa.json`.

Both new figures were visually inspected at full resolution: labels, legends,
curves, color scale and correlation matrix are readable and unclipped. The root
agent handles integrated rendered-PDF inspection, parent immutability and the
standalone report rebuild.

## Cross-review

The physics and conclusions sections make the fixed-source restriction and
signal-absorption problem correctly. One wording change was requested in the
physics outer-ensemble prescription: jointly generate control and target from
the specified underlying truth with correct overlap/exposure, then refit the
control source, rather than generate each target from its newly estimated
control mean. The latter would risk hiding source-estimation error again.
Root conclusions already largely stated the appropriate repeated experiment.

## Recommended scope of adoption

Retain the reviewed extractor and ±2.25 sigma window, the raw observed scan,
and the full-response GP as a computational tool. Do not adopt reference
subtraction or a rho-threshold count on these diagnostics alone. Specify the
test, domain, source policy and source uncertainty before selecting the result;
test that complete policy under several plausible background truths and under
signal injection before source fitting as well as after it. No new exclusion
curve, discovery reach or calibrated physical discovery significance follows
from this bounded review.
