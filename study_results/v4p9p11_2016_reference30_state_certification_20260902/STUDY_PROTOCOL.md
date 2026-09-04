# v4.9.11 prospective reference-card adequacy and state-certification protocol

Frozen before any full-2016 low-control fit, any new 39--180 MeV fit, and any
signal extraction or inference in this release.

## Decision inherited without reopening support selection

This study is not another lower-support optimization.  v4.9.10 tested the
fixed shortlist 29--210 and 30--210 MeV on already-open 2016-10% data.  All
eight 30 MeV production-geometry cells passed at the authorized factor 12,
whereas 29 MeV failed its optimizer-reproduction rule at the 90 MeV anchor.
Under the frozen v4.9.10 rule this stopped the study before full controls; it
did not select or confirm 30 MeV.

v4.9.11 abandons the failed alternative and retains the pre-existing reviewed
reference card, 30--210 MeV support and 2016 upper length-scale factor 12.  The
reference is retained because no alternative earned promotion, not because a
full-data comparator or observed inference favored it.  v4.9.9 and v4.9.10
remain terminal under their own rules.

## Phase A: full-2016 low-control absolute adequacy, pass or stop only

The immutable full-2016 histogram is
`h_Minv_General_Final_1` in the v4.9.7 archived input, file SHA-256
`c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301`
and values-plus-edges SHA-256
`c4225e5bacdb2c1791a6ba943b47695cfddee7f9efed1ffff77065350ba48632`.

Only support 30--210 MeV and factor 12 are evaluated.  There is no comparator,
ranking, displacement, or support decision.  The low-control blocks are L1
`[35.25,36.00)`, L2 `[36.00,36.75)`, L3 `[36.75,37.75)`, and L4
`[37.75,38.75)` MeV.  For each held-out block, training uses `[30,35.25)` MeV
plus the other three blocks.  No center at or above 38.75 MeV may be read for
training or scoring.  Kernel anchors are 39, 65, 100, 140, and 180 MeV.

Preprocessing is fixed to five native bins per bin, `x=log(m)`,
`y=log(count)`, and `alpha=1/count`.  The kernel is Constant times RBF with
constant bounds `[1e-8,1e18]`, the reviewed resolution-scaled local length
bounds (2016 lower factor 0.9, upper factor 12, dataset-stat floor 0.8), and 12
optimizer restarts.  Deterministic seeds are 2711, 6043, and 9151.

A repeat is eligible only when its fit and kernel coordinates are finite and
it emits no Python/scikit-learn warning.  Any L-BFGS warning is ineligible.  A
cell selects the largest LML among eligible repeats and requires at least two
eligible repeats to reproduce it within absolute LML `1e-4`, relative
constant `1e-3`, and relative length `1e-3`.  The selected coordinates must be
more than relative `1e-3` from every bound.  The predictive count covariance
`C_GP + diag(mu)` must be finite/SPD with no negative eigenvalue below
`-1e-8 max(diag(V))` and no Cholesky jitter above `1e-8 median(diag(V))`.

All 20 anchor/block cells must pass.  The pragmatic absolute adequacy gates
are mean Mahalanobis per scored bin below 4, every anchor/block Mahalanobis per
bin below 9, and every marginal standardized residual strictly below 5 in
absolute value.  These are gross-misfit guards, not calibrated goodness-of-fit
p-values.  If any technical or absolute gate fails, the release stops.  No
rule change or control-dependent repair is permitted.

## Phase B: provenance classes for the archived 142 states

The reviewed archived k12 ledger is SHA-256
`a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9`.
Only its 142 dataset-2016 rows at integer masses 39--180 MeV are in scope, and
only fit-state/provenance fields may be accessed before state certification.
All rows are noninterpolated actual-fit rows.

Two provenance classes are fixed before inspection of full-data predictions:

1. `raw_single_source`: 139 rows from the k12 raw attempt.  The archive has
   only one fit source and no optimizer-warning log, so this entire class must
   be rerun under the robust-repeat rule below; no raw row may be accepted by
   itself.
2. `repair_three_source`: masses 43, 125, and 145 MeV, each with three
   unchanged-card repair sources.  Reuse is allowed only if at least two
   sources reproduce the selected maximum-LML state within absolute LML
   `1e-4`, relative constant `1e-3`, and relative length `1e-3`, and the
   numerical certification below passes.  Historical warning-free status is
   not claimed because it was not logged.  Failure of any repair row sends all
   three repair rows, as one class, to the robust-repeat rule.

For every archived row, provenance must establish exact input, support,
factor-12 card, mass/mask, rebin, preprocessing, kernel-bound, and source-file
hashes.  The historical state must be reconstructed at fixed coordinates.
Its reconstructed LML must agree with the recorded LML within `1e-6`.
Scikit-learn log-space prediction must agree with an independent direct
Cholesky implementation to relative `1e-9` and absolute `1e-10` for both mean
and covariance.  The state must have two nonempty training sidebands, interior
coordinates, finite/SPD count covariance, and deterministic prediction hashes.

Optimizer optimality is certified numerically at the fixed state by both:

- absolute infinity norm of the analytic log-coordinate LML gradient below
  `0.01`; and
- deterministic L-BFGS-B polish from that exact state, with analytic gradient,
  `maxiter=1000`, `maxls=50`, `ftol=1e-12`, and `gtol=1e-8`, succeeding with
  LML improvement no larger than `1e-4` and relative movement of each kernel
  coordinate no larger than `1e-3`.

These are numerical stationarity/closure tolerances, not statistical tests.

## Phase C: whole-class robust repeats and global state decision

After Phase A passes, every `raw_single_source` mass is refit three times with
the unchanged reference card, seeds 2711, 6043, and 9151, and 12 restarts per
repeat.  If Phase B sends the repair class to rerun, all three repair masses
receive the same treatment.  There are no mass-specific or anomaly-specific
retries.

Each repeat is eligible only if it is finite and warning-free.  The selected
branch is maximum LML among eligible repeats.  At least two eligible repeats
must reproduce maximum LML and both kernel coordinates under the Phase-A
tolerances.  The selected state must also pass the fixed reconstruction,
gradient, polish, bounds, two-sideband, covariance, and prediction-closure
requirements.  Differences from archived coordinates are recorded but never
ranked using signal amplitude, p0, Z, or a limit.

All 142 masses must resolve.  A single unresolved mass stops the entire state
ledger and forbids combination.  A successful ledger contains 142 exact
states plus complete source/attempt/mask/prediction hashes.

## Phase D: inference only after state-ledger validation

Only after an independent validator passes the complete 142-state ledger may
signed signal amplitude and local asymptotic p0 be evaluated.  State selection
cannot use them.  Core uncorrected `A_up`/`eps2_up` values are forbidden: the
final asymptotic limit must use the separately validated piecewise bounded
`tilde_q_mu` mapping and expose optimizer success, `q_obs`, `q_A`, and the
analytic branch at the solved limit.

No toys, expected bands, global significance, or coverage calibration are in
scope.  Because the factor/support history used related open 2016 data, final
full-2016 and combined asymptotic results are fixed-model results conditional
on a partially unblinded model history, not unconditional-coverage or an
independently blinded confirmation claim.
