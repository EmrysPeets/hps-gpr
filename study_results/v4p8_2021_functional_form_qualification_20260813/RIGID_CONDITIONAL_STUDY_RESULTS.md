# v4.8 rigid conditional-stress study result

## Disposition

The requested low-complexity pseudoexperiment branch is complete for analysis
indices 0--19.  It is a conditional stress study, not a qualified physical
background truth, a coverage study, or a production-card selection.

The declared mean is a thresholded generalized-gamma core with sparse `T2` and
`T6` broad corrections.  One normalization and five non-normalization
coordinates are fitted to native 1%; the five shape coordinates are frozen for
native 10%, where only normalization is fitted.  On the primary 50--250 MeV
region, native-bin Pearson/deviance ratios are 1.088/1.088 for native 1% and
2.676/2.676 for the normalization-only native-10% application.  These are
engineering fidelity scores under the user-declared `<3` criterion, not formal
Poisson-model acceptance.  The 40--50 and 250--300 MeV intervals remain visible
GP-training shoulders.  This family failed the 30-MeV support control, so the
study remains on 40--300 MeV support.

The cache contains 25 nested backgrounds per source family, but the
authoritative v4p8p2 closure and locked length scan use only indices 0--19.
Indices 20--22 had already been inspected in a superseded one-lane development
run and are therefore not an unopened statistical reserve.  Indices 23--24
were not consumed by the authoritative products, and no reserve-based pooling
or promotion claim is made.

## Conditional extraction result

The four lanes, five masses (65, 90, 120, 180, and 210 MeV), and four injection
strengths produced 1,600 raw extraction states.  The pull-blind optimizer gate
accepted 1,599.  The sole exclusion is native-10% x10, background 12, 65 MeV,
`z=5`, where the top injected-refit optimizer branch did not reproduce after
five attempts; the affected cell retains 19 accepted rows.

Closure is nonuniform.  At 65 MeV the background-only mean pulls are
`-1.305` (90% diagnostic interval `[-1.763,-0.847]`) for 1%x10 and `-1.327`
(`[-1.800,-0.854]`) for native 10%.  At `z=5`, their median recoveries are
0.714 and 0.744.  Additional mass-dependent offsets and pull-width anomalies
are documented in `SECTION5_RECOMMENDED_CHANGES.md`.  These finite conditional
diagnostics are not coverage, observed-data bias, or CLs calibration.

## Pull-blind length-scale result

The independently locked background-only scan used the same 20 backgrounds at
50,70,...,250 MeV and factors 15, 20, and 25.  It produced 8,004 optimizer
attempts and selected 2,637/2,640 factor--mass states.  Three exclusions are all
1%x100 background 8: factor 15 at 90 and 190 MeV and factor 25 at 150 MeV.  In
each case the maximum-LML branch appeared only once after five attempts.

Among the 2,637 selected rows, 2,083 retain at least one optimizer warning,
predominantly L-BFGS abnormal-line-search warnings.  All selected rows still
reproduce the chosen maximum-LML branch and pass the frozen fit/covariance
gate.  Their minimum covariance eigenvalues are tolerance-valid under the
declared `-0.01` relative threshold, not strictly positive-semidefinite.

Exact/near upper-bound occupancy is:

| Factor | Exact | Near, including exact |
| ---: | ---: | ---: |
| 15 | 599/878 (68.2%) | 629/878 (71.6%) |
| 20 | 2/880 (0.2%) | 5/880 (0.6%) |
| 25 | 0/879 | 0/879 |

Factor 15 therefore actively truncates the preferred broad GP scale through
the middle of the search region.  The 20-to-25 comparison is a strong
near-plateau: 869/879 paired states change by less than 0.01 in
`ell/sigma_x`.  Two native-10% background-2 states release factor-20 contacts
at 170 and 210 MeV, so the plateau is not universal.  Ten tiny strict nested-
LML reversals occur, all in the 20-to-25 comparison, but their changes are only
`-4.89e-7` to `-2.78e-7` per training bin and none reaches the material
threshold.  Factor 25 is consequently a clean non-contact conditional stress
control in this ensemble, not an authorized production ceiling.

## Confidence contract and decision

The frozen v4.2 card remains unchanged: support 40--300 MeV, search 50--250
MeV, factor-15 production ceiling, `tilde_q_mu`, and `cls_alpha=0.10` (90% CLs).
This study computes no CLs interval or coverage.  Its Student-t and chi-square
intervals are 90% diagnostic intervals only.

No generator, support-edge, length-scale ceiling, or production-card change is
promoted from this 20-background conditional stress study.

For machine-readable nested-LML rows, the three unavailable comparisons have
nullable boolean flags.  Consumers must first require `comparable == true` and
treat blank flags as false; naive boolean casting of `NaN` is unsafe.
