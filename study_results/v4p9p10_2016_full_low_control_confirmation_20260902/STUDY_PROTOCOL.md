# v4.9.10 prospective 2016-full low-control confirmation protocol

Frozen before any full-2016 control fit or score was evaluated on 2026-09-02.

## Relationship to the terminal v4.9.9 study

The v4.9.9 study remains canonical and terminal under its own frozen rules.  It
completed only the 2016-10% development phase and stopped because all 20
candidate-independent high-only control fits occupied the upper length-scale
bound.  Its independent validator passed 21/21 checks, reconstructed exactly
zero 39--180 MeV training or score centers, and verified that no full-2016
confirmation or observed-production directory exists.

At this freeze:

- v4.9.9 Phase-1 decision SHA-256:
  `2f15b63c1f4d58c13102f4f42123b1e5c9079350b46d1fff32946ad6b87c26c1`;
- v4.9.9 validation SHA-256:
  `0b5102540f1f99ed619be4c6f2f18379ff847dab4a5dc328ac6ced535717fb87`;
- v4.9.9 development score SHA-256:
  `823a9297a9736c976a852908901a73f45a37303711d0fca6ff2b949caedcad63`.

No full-2016 control fit or score has been run or inspected.  Before this
freeze, the full ROOT file's identity, axis, and total count were checked, but
not a candidate-dependent control prediction.

## Structural correction and fixed shortlist

The v4.9.9 high-only check trained on three high blocks at a time, independently
of the lower support edge and without the search interval.  Its upper-bound
contact is evidence that this disconnected high-only GP prefers a smoother
function than the production factor-12 cap; it cannot distinguish 29 from
30 MeV and is not production-like.  Treating it as a terminal lower-edge gate
was intentionally conservative but structurally overrestrictive.

This new study removes the high-only check from lower-edge selection rather
than rewriting v4.9.9.  High controls are not fitted, ranked, or used here.

The shortlist is fixed from the already-open 2016-10% development result:

- 29--210 MeV: the sole nonreference edge that passed every low-only relative,
  absolute, covariance, and deletion-stability gate;
- 30--210 MeV: the later reviewed 2016-10% reference and mandatory null/default.

Edges 31--34 are excluded before full controls are opened.  No post-confirmation
candidate expansion is allowed.

## Phase A: production-like factor-12 check on open 2016-10% data

The upper length-scale factor is 12 by default, as authorized.  Before opening
full controls, production-like GP fits on the already-open 2016-10% development
spectrum test whether factor 12 genuinely constrains either shortlisted
support.  This phase uses no signal extraction, p-value, or limit.

- Supports: 29--210 and 30--210 MeV.
- Mass anchors: 39, 44, 54, 65, 90, 120, 150, and 180 MeV.
- At each anchor, train over the full support except the fixed
  `mass +/- 2.25 sigma_2016(mass)` window, matching production geometry.
- Preprocessing: rebin five, `x=log(m)`, `y=log(count)`, `alpha=1/y_count`.
- Kernel: ConstantKernel times RBF, lower factor 0.9, default upper factor 12,
  dataset-stat floor factor 0.8, production constant bounds.
- Deterministic optimizer seeds: 2711, 6043, 9151; each fit has the nominal
  start plus 12 random restarts.

A repeat is eligible only if it has finite positive kernel parameters, finite
LML, and emits no Python/scikit-learn optimization warning.  In particular,
any L-BFGS convergence warning, including `ABNORMAL_TERMINATION_IN_LNSRCH`,
makes that repeat ineligible.  The selected branch is the maximum LML among
warning-free repeats in the exact support/anchor/factor cell.  At least two
warning-free repeats must reproduce that maximum within absolute LML `1e-4`
and relative length-scale `1e-3`.  The selected constant and length scale must
not be within relative `1e-3` of a bound, and the blind-window count covariance
plus Poisson diagonal must pass the same finite/SPD/jitter rules used in
v4.9.9.

Factor 12 freezes immediately if every support/anchor cell passes and none
occupies the upper bound.  A larger factor may be tested only if every other
technical condition passes and at least one reproduced factor-12 branch
occupies the upper bound.  The sequential candidates are 15, 20, 25, and 30.
The first nonbinding factor `k` can freeze only if the next candidate also is
nonbinding and forms a plateau in every common cell:

- `abs(log(ls_next / ls_k)) < 0.01`; and
- `abs(LML_next - LML_k) / n_train < 1e-5`.

If a factor fails for warning-free reproduction, covariance, a constant-bound
contact, or another reason not uniquely attributable to an upper-length-bound
contact, the study stops instead of expanding the cap.  If factor 30 is the
first nonbinding value, there is no next-factor confirmation and the study
stops.  LML is never compared between support edges.

## Phase B: untouched full-2016 low-control confirmation

Only after Phase A freezes the length factor may the full-2016 low controls be
evaluated.  The immutable input is `h_Minv_General_Final_1` in
`../v4p9p7_2016_support_combined_100toy_20260902/inputs/source_2016_full.root`,
file SHA-256
`c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301`,
values-plus-edges SHA-256
`c4225e5bacdb2c1791a6ba943b47695cfddee7f9efed1ffff77065350ba48632`.

The confirmation repeats the low-only v4.9.9 blocked procedure exactly for
supports 29 and 30, except for the tightened warning-free optimizer rule and
the Phase-A-frozen upper factor:

- common scored controls: L1 `[35.25,36.00)`, L2 `[36.00,36.75)`,
  L3 `[36.75,37.75)`, and L4 `[37.75,38.75)` MeV;
- candidate-specific training extension: `[edge,35.25)` MeV;
- for each held-out block, training uses only that extension and the other
  three low blocks;
- kernel anchors: 39, 65, 100, 140, and 180 MeV;
- optimizer seeds: 2711, 6043, 9151, each with 12 restarts;
- no bin at or above 38.75 MeV enters training or scoring.

Every fit ledger must store exact train/score center and count hashes, extrema,
counts, warnings, selected LML/kernel, covariance diagnostics, and the number
of centers in the 39--180 MeV search.  The search count must be zero.

The primary score remains the joint count-space predictive NLPD with
`V = C_GP + diag(mu)`.  The 29 MeV edge displaces 30 only if:

1. both supports pass every technical and warning-free reproduction gate;
2. both supports pass the pragmatic absolute guard: mean Mahalanobis/bin <4,
   no anchor/block Mahalanobis/bin >=9, and no marginal standardized residual
   with absolute value >=5;
3. paired `Delta_NLPD = NLPD_30 - NLPD_29` across L1--L4 is greater than one
   paired standard error;
4. the mean paired Poisson-deviance improvement is nonnegative; and
5. the mean NLPD improvement remains positive after deleting each one of the
   four blocks.

If 29 fails any rule and 30 passes all absolute/technical rules, freeze the
default 30--210 MeV support.  If 30 fails, stop with no support.  The thresholds
are pragmatic gross-misfit/model-choice gates, not calibrated GOF p-values.

The 10% development source is a subset or related sample, not demonstrated
event-level-independent data.  Agreement between development and full controls
is a same-experiment confirmation and must not be called independent external
validation.

## Phase C: full-2016 production-state qualification

Only after Phases A and B freeze the card may the 39--180 MeV observed scan run.
The production card changes only the selected `data_range_2016` lower edge and,
if Phase A requires it, the 2016 upper length-scale factor relative to the
reviewed v4.1 full-data card.

Every one of the 142 integer-MeV hypotheses is fitted with three complete,
deterministically seeded, unchanged-card attempts.  Branch selection uses only
warning-free maximum LML.  Each accepted state requires at least two
warning-free repeats reproducing maximum LML/length within the Phase-A
tolerances, no constant or length bound, finite/SPD predictive covariance, two
training sidebands, and exact support/card provenance.  There is no adaptive
repair selected from amplitude, p-value, Z, or limit behavior, and no
interpolation.  If any mass fails, the production ledger is not authorized for
combination.

The signed fitted amplitude and local asymptotic p0 may be reported only after
the state is selected on the preceding fit-only criteria.  Core uncorrected
`A_up`/`eps2_up` values are not accepted because the bounded `tilde_q_mu`
analytic CLs implementation lacks the `q_obs > q_A` tail branch.  Any final
upper limit must use the separately validated piecewise Cowan mapping and
record `q_obs`, `q_A`, and the branch at the solved limit.

## Claim boundary

A successful result is a prospectively confirmed, out-of-search lower-support
choice plus a per-mass optimizer-qualified observed/asymptotic state for a
partially unblinded analysis.  It is not coverage calibration, a toy-calibrated
global significance, an expected band, or proof of the physical background
model.  No toys or bands are part of v4.9.10.
