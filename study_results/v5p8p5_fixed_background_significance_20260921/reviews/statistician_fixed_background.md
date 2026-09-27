# Statistician review of the newly computed C=0 scans

This review extends the earlier outside-reader assessment using an actually
computed alternative likelihood. It does not relabel the historical profiled
values as fixed-background results. The prior distinction and calibration
limitations were checked in
`study_results/v5p8p5_external_statistician_20260921/reviews/fixed_background_audit.md`.

## Scientific specification supplied before the scan

- Recompute the masked GP prediction separately for every dataset, mass and
  observed/toy spectrum, using the inherited positive log-interpolated kernel
  states and ±2.25 sigma mask. Preserve the arithmetic/lognormal mean, including
  the half-posterior-variance correction. Remove the nuisance covariance only
  from the extraction likelihood; do not freeze the GP prediction across toys.
- Use the original unrenormalized signal templates in coupling amplitude units
  `A=epsilon2/1e-8`. Shared coupling concatenates those templates with one common
  amplitude. Independent amplitudes use one nonnegative amplitude per active
  dataset at a common mass.
- Fit an unrestricted signed amplitude where expected bin counts remain
  positive, obtaining the signed likelihood root `r`. Use `q0=max(r,0)^2` for an
  excess. Nominal individual/shared p is `norm.sf(r)` for positive r, otherwise
  the inclusive atom tail is one. The conventional half-chi mapping of 0.5 at
  q0=0 can be retained in a separate audit column; positive peaks are unchanged.
- Independent-amplitude `qR=sum_d max(r_d,0)^2`. For positive qR, use the nominal
  active-K chi-bar-square mixture `sum_{k=1}^K choose(K,k)/2^K * chi2.sf(qR,k)`;
  at qR=0 use inclusive p=1. Never report sqrt(qR) as a significance.
- Use a 0.5-MeV grid: 2015 19–100, 2016 39–180, 2021 50–250 MeV. Primary
  combinations search the 19–250 union with changing dataset membership;
  secondary combinations search 50–100 with all three datasets at every mass.
  No post-comparison domain choice is implied.
- Primary individual/shared ordering is maximum positive raw r. Primary free
  ordering is maximum minus log nominal p, accounting for active-K changes
  without source centering. The free raw-q maximum is an audit-only alternative.
- Apply each statistic, grid and domain identically to the observed scan and
  256 complete, coherent Poisson toy spectra from the inherited fixed source.
  Compare local tails at fixed mass and maxima over the specified search.
  Record counts, raw k/N, add-one estimates, exact binomial intervals and
  one-sided sampling upper bounds separately. No a/s recentering or rescaling
  enters the observed statistics. No new Gaussian-field surrogate is required.

## Observed C=0 results independently checked

Source: `results/observed_peaks.json` and `results/observed_curves.csv` in this
study. These are nominal mappings, not empirically validated rare-tail claims.

| Scope/domain | Peak mass MeV | Nominal local Z | Nominal local p |
|---|---:|---:|---:|
| 2015 full | 51 | 11.906451 | 5.47704e-33 |
| 2016 full | 87.5 | 8.722900 | 1.35583e-18 |
| 2021 10% | 77 | 8.377566 | 2.70168e-17 |
| Shared union | 22 | 11.359992 | 3.30755e-30 |
| Shared all-three overlap | 50.5 | 7.799609 | 3.10495e-15 |
| Free union or overlap | 51 | 11.597214 | 2.12857e-31 |

The union shared maximum is **2015 alone**, because the other datasets do not
cover 22 MeV. The three-amplitude peak at 51 MeV is also overwhelmingly driven
by 2015: its contribution to qR is 141.76358 of 142.18625, or about 99.70%.
Neither entry demonstrates a three-campaign common signal.

At 66 MeV, the C=0 shared root is 7.250965, nominal p=2.06906e-13 and fitted
epsilon2=(5.96166±0.82265)e-6. The free test has qR=58.06045, nominal Z=7.203789.
At 92 MeV, the shared root is 3.852991, nominal p=5.83419e-5, while free qR=79.29708
gives nominal Z=8.521230. Individual raw roots at 92 are 6.37110, 6.15975 and
0.873895. This is the same qualitative amplitude-consistency issue as in the
profiled study, now under a much tighter likelihood.

At 78 MeV, individual C=0 roots are −8.15666, −5.20047 and +7.55240. The free
test retains only the positive 2021 component. Its apparently combined excess
cannot be interpreted as agreement among the three campaigns.

## Why nominal and empirical probabilities must be separated

A C=0 likelihood conditions on an estimated background as though it were
known. Its curvature describes the likelihood with that estimate fixed; the
repeated scan also includes variation of the sideband estimate and its mean
response. Those variations need not produce unit-width, zero-mean signed roots.
Testing the normal/chi-bar mapping is therefore a local issue before any
look-elsewhere correction is applied.

Direct Poisson calibration can nevertheless define a valid conditional test
using the unchanged C=0 statistic. It does not require subtracting the source
offset. Rejecting a reference-centered ordering does not make its nominal
normal mapping automatically correct. Conversely, poor nominal mapping does
not by itself prove that the C=0 statistic has no useful power after calibration.

The comparison with profiling must retain the same raw ordering. The root's
`results/profiled_raw_scan.csv` and `profiled_raw_peaks.csv` do that using the
same coherent Poisson inputs and half-MeV grid. Comparing a C=0 raw maximum to
the earlier reference-standardized maximum would mix two changes.

The inherited source was derived from the observed data and is held fixed for
these experiments. Repeated source construction, its uncertainty, signal
absorption, detector/normalization uncertainty and prior method selection are
not calibrated here. The current exercise computes and checks the requested
statistic; it does not establish a physical background-only truth.

## Direct local checks and completed overlap result

These are complete 256-spectrum checks, not Gaussian approximations to the
tail. At shared 66 MeV the observed r is 7.250965, but the toy mean is −0.059888
and SD is 2.729863. The nominal 5% test rejects 59/256 null experiments (23.05%);
the nominal 1% test rejects 43/256 (16.80%). The Asimov root is only 0.060197.
This establishes scale failure even where deterministic centering is almost
irrelevant. The local count is 0/256, whose one-sided 95% upper probability bound
is 0.011634; it cannot establish the nominal 2.07e-13 tail.

At shared 22 MeV, toy mean 3.9724 and SD 3.7435 show that both mean response and
scale matter. Its apparently eleven-sigma observed root is exceeded locally by
6/256 toys; the nominal 5% threshold rejects 187/256. At 2015 51 MeV the toy SD
is 3.5969 and 71/256 nominal 5% tests reject; the local count remains 0/256.

The completed 50–100 MeV all-three overlap gives global peak exceedances
38/256 for shared coupling and 36/256 for free amplitudes. At 66 MeV the
corresponding overlap-global counts are 55/256 and 173/256; at 92 MeV they are
237/256 and 129/256. Fixed-mass 92 MeV local counts are 7/256 shared and 1/256
free. Its shared null mean and SD are 0.112558 and 1.881874.

These numbers must not be compared to the tiny nominal local p-values as if
their ratio were solely a LEE trials factor. That comparison mixes marginal
miscalibration with the search maximum. Raw ordering is an explicit conditional
test, but its highly variable null scale across mass makes its sensitivity
uneven. Restoring a normal-looking score would define a different ordering; it
was deliberately not done in this requested calculation.

## Final complete-search results

The numerical engine completed the observed, deterministic and complete
256-spectrum half-MeV scans in 559 seconds. Source of the following values:
`results/peaks.json`, identified by the `scope` and `domain` keys; all requested
anchor rows are in `results/significance_curves.csv`.

| Scope | Domain | Observed peak MeV | Global k/256 | Global k/N | CP95% interval |
|---|---|---:|---:|---:|---|
| 2015 | 19–100 | 51 | 10/256 | .0390625 | [.0188883,.0706623] |
| 2016 | 39–180 | 87.5 | 35/256 | .1367188 | [.0971117,.1849815] |
| 2021 | 50–250 | 77 | 96/256 | .3750000 | [.3154999,.4374310] |
| shared | 19–250 union | 22 | 13/256 | .0507813 | [.0273125,.0852721] |
| shared | 50–100 all three | 50.5 | 38/256 | .1484375 | [.1072334,.1980028] |
| free | 19–250 union | 51 | 48/256 | .1875000 | [.1416035,.2407937] |
| free | 50–100 all three | 51 | 36/256 | .1406250 | [.1004746,.1893326] |

Every global peak tail has nonzero exceedances; none supports an extremely
rare global claim. The free raw-q alternative gives 46/256 on the union and
36/256 on the overlap. It remains audit-only, as specified in advance; the
main free ordering uses the active-K nominal p map.

The all-three shared 50.5 MeV maximum must not be conflated with the
2015-dominated free maximum. Its individual C=0 roots are approximately
8.0109, 0.2416 and 7.4587, and fitted epsilon2 values, in units of 1e-6, are
18.619, 0.3935 and 18.113. It contains 2015 and 2021 positive responses and a
weak 2016 response. Its proximity to the 2021 lower search boundary deserves
explicit domain labeling; proximity by itself does not establish an artifact.

The paired profiled shared66 check, extracted directly from the copied parent
`fields/combined.npz` (`masses==66`, `validation`), has raw-root mean .0222171,
SD 1.0540821, 19 nominal 5% rejections and 4 nominal 1% rejections. This supports
the interpretation that the C=0 width 2.7298633 omits a material component of
background-estimation variability in this generating experiment; it does not
prove perfect calibration of all profiled tails.

Matching raw search orderings, the profiled full-domain exceedance counts are
18,18,57,70,11 for 2015,2016,2021,shared,free respectively. Shared/free overlap
counts are 27/256 and 3/256. Their observed peak masses need not match the C=0
peak masses. These are comparisons of two fully specified search procedures,
not differences in one fixed-mass p-value. No statistical test of the paired
change or calibrated power comparison is claimed.

## Updated outside-reader judgement

The requested no-profiling branch has now been computed, and it does create
dramatically smaller **nominal** p-values. The direct repeated-experiment
checks show why those nominal increases cannot be taken as new discovery
evidence. The physical common-rate restriction and dataset membership still
matter: the full shared maximum comes from 2015 alone, while the all-three
shared result is a modest conditional global fluctuation. The free combination
is not independent corroboration of a common-rate signal.

I would keep C=0 as an explicit alternative statistic or diagnostic, rather
than recommend it because its observed curve appears more significant. A
properly calibrated C=0 test could have useful power; determining that requires
size and signal-injection comparisons under qualified background ensembles,
with source fitting repeated where it belongs in the procedure. The current
study does not perform or imply that new power comparison. Nor does it update
an exclusion curve or forecast significance for more 2021 data.

The old profiled priorities remain hypotheses for independent follow-up, not
the peak ranking of this new likelihood. C=0 additionally draws attention to
the 50–51 MeV region, but its local and global calibration behavior must be
stated. More independent 2021 data should test predicted amplitudes and residual
shape with the selected procedure frozen, rather than validate whichever of
the profiled and C=0 versions happens to give the larger observed Z.
