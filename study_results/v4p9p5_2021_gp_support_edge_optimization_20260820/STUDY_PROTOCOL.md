# v4.9.5 lower GP-support-edge protocol

Frozen before any v4.9.5 support-scan extraction or observed-data scan on
2026-08-20.

## Question

Choose the lower edge of the 2021 native-10% GP support between 32 and 40 MeV
without optimizing on the observed limit or observed local p-value. The 30 MeV
edge is retained as the v4.9/v4.9.1 control but is ineligible for the freeze
because it includes the low-count shoulder identified in the source spectrum.

## One-factor scan

- Fixed upper support edge: 300 MeV.
- Lower-edge controls/candidates: 30, 32, 34, 36, 38, and 40 MeV.
- Native-10% conditional truth and exactly the 100 already archived v4.9.1
  pseudo-datasets; no new background toys are generated.
- Threshold masses: 55, 60, 65, and 70 MeV, matching the v4.9 near-threshold
  grid.
- Matched-reference injections: 0, 2, and 5 sigma.
- Search range remains 50--250 MeV. Only `data_range_2021[0]` changes.
- All other v4.2/v4.5 settings, including log-y preprocessing, `alpha=1/y`,
  five-bin rebinning, 2.25-sigma blind/training exclusions, resolution-scaled
  length bounds, 12 restarts, signed extraction, and the two-sideband gate,
  remain fixed.
- Optimizer branches are selected only through reproducibility, covariance,
  kernel-state, and maximum GP log marginal likelihood. Amplitude, pull,
  signal recovery, epsilon-squared, and upper-limit values are not optimizer
  gates.

## Cohorts and freeze rule

Phase 1 runs indices 0--24 for all six support edges. For each eligible edge,
the primary score is the maximum absolute mean pull across the twelve
mass-strength cells. Edges within 0.10 of the lowest score are tied; the
smallest tied edge is provisionally selected to retain the most support. A
candidate is ineligible if an injected cell has `abs(mean pull) >= 0.5` and its
two-sided 90% Student-t interval excludes zero.

The per-pseudo-dataset observed 90% CLs yield limit is a bounded Wald
`tilde_q_mu` diagnostic calculated from the accepted profiled amplitude and
uncertainty. Paired changes in `(A_up-A_injected)/sigmaA_ref` may reject an
unstable edge or break a primary-score tie. The smallest absolute limit is
never an optimization target.

Phase 2 runs the independently seeded indices 25--99 only at the provisional
edge and its immediate scanned neighbors. The edge is frozen only if every
full-100 cell retains at least 95 accepted rows, no accepted fit contacts a
kernel bound, every full-100 mean satisfies `abs(mean pull) < 0.5`, and no
continuation cell has `abs(mean pull) >= 0.5` with a two-sided 90% interval
excluding zero. Confirmation failure ends the study without retuning to a
different edge.

## Observed products

Only after the support edge is frozen may the native 2021 10% histogram be
scanned. The observed pass uses the full profiled asymptotic `tilde_q_mu` CLs
construction at 90% confidence, the signed-yield extraction, epsilon-squared
conversion with the frozen +/-1.64-sigma density convention, and the local
analytic profiled-LRT p-value. No expected or observed upper-limit bands and no
CLs toys are produced. Local p-values are not scan-global significances.

## Interpretation boundary

The support study is a source-conditioned injection-extraction diagnostic and
not a coverage experiment, physical background generator qualification,
observed-data bias measurement, expected band, exclusion, or trials
calibration. The later observed curves are results conditional on the frozen
card; their aligned appearance is not itself a calibration test.
