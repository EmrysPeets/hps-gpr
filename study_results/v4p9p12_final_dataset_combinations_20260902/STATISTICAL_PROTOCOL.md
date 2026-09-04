# Frozen statistical protocol for final-dataset combinations

Frozen before evaluating the new 2016 observed scan or any new final-dataset
limit, local p-value, or extraction.

## Inputs and scopes

The only result inputs are 2015 full, 2016 full, and 2021 10%.  The released
ledger must contain exactly the seven nonempty subsets of those inputs:

- three standalone curves on 19--90, 39--180, and 50--250 MeV respectively;
- 2015+2016 on 39--90 MeV;
- 2015+2021 on 50--90 MeV;
- 2016+2021 on 50--180 MeV; and
- all three on 50--90 MeV.

This gives 680 result rows.  The 2016 10% source is development information for
the 2016 support prescription, not a final result curve.  The 2021 1% source is
not read by this workflow.  Immutable ROOT-file and histogram hashes, the 415
reviewed fixed GP states, the final 2016 support decision, and the production
analysis card are required to close before evaluation.

## Frozen GP states and common coupling

At every dataset and mass point, the GP constant and length scale are taken from
the fit-only reviewed maximum-LML state.  Reconstruction uses those exact
coordinates with optimization disabled and requires LML closure within
`5e-5`.  No limit, fitted amplitude, local p-value, or agreement with another
curve can select or repair a GP state.

For a subset of data sets, the concatenated signal vector `s_unit` is the
expected fitted-window count vector per unit electron-channel epsilon squared.
It must be finite and componentwise nonnegative.  The inference coordinate is

`S = sum(s_unit)`, `w = s_unit / S`, and `A_window = epsilon_squared * S`.

This `count_scale` transformation is algebraically identical to a direct
epsilon-squared fit.  Each campaign retains its own resolution, density,
radiative fraction, normalization, background mean, and covariance.  The
combined covariance is block diagonal across data-taking periods.

## Explicit covariance conditioning

Numerical conditioning is frozen before evaluation and is never selected from a
result.  For each symmetric GP covariance block `C`, require finite entries and
nonnegative diagonal variances.  Let

`scale = max(max(abs(diag(C))), 1)`.

Try the deterministic diagonal loads `scale * 10^k I`, in order, for
`k = -10, -9, ..., -4`.  Use the first matrix with a successful Cholesky
factorization.  Stop if none succeeds; eigenvalue clipping is forbidden.  The
selected load must be recorded.  The actual covariance represented by the
attested likelihood's internal factor, and `V = C_effective + diag(mu)`, must
also be hashed; `V` must be strictly positive definite.  The same conditioned
block covariance is used for the limit, local p-value, and extraction.

A predeclared no-material-impact audit compares representative points with the
attested runtime's native regularization.  If any block requires the `1e-4`
cap or the conditioning changes a limit or p-value beyond the frozen numerical
closure tolerances, the release stops.

## Bounded piecewise-asymptotic CLs

All upper limits are observed 90% CLs limits based on the bounded
`tilde(q)_mu` profile-likelihood statistic.  The existing statistic changes its
denominator to the physical null for a negative unconstrained estimator, but
the archived analytic tail conversion omitted the corresponding second branch.
The release-local mapping follows Cowan et al., arXiv:1007.1727, Section 3.7.
For background-only Asimov value `q_A`:

- when `q_obs <= q_A`, use `z_sb = sqrt(q_obs)` and
  `z_b = sqrt(q_A) - sqrt(q_obs)`;
- when `q_obs > q_A`, use
  `z_sb = (q_obs + q_A)/(2 sqrt(q_A))` and
  `z_b = (q_A - q_obs)/(2 sqrt(q_A))`.

Then `CL_sb = sf(z_sb)`, `CL_b = Phi(z_b)`, and
`CLs = CL_sb / CL_b`, with the ratio evaluated in log space.  Positive test
strength with unresolved `q_A=0`, nonfinite tails, unsuccessful optimizer
status, or violated likelihood nesting stops the result.  The root solver must
preserve an oriented bracket, sample a nonincreasing CLs trace within absolute
`5e-5`, converge within 80 bisections, and close at `CLs=0.1` within absolute
`2e-6` and log residual `2e-5`.  Every row stores `q_obs`, `q_A`, the tail and
profile branches, log tails, optimizer evidence, bracket endpoints, and the
convergence reason.

## Local p-values and extraction

The fixed-mass discovery statistic uses the same `w=s_unit/S` count coordinate.
Both alternative and null profiles must succeed and satisfy
`NLL_alt <= NLL_null` within the frozen likelihood-difference tolerance.  The
one-sided value is `p0 = sf(sqrt(q0))` and is constrained to `(0, 0.5]`.

The extraction mass is selected mechanically as the minimum local all-three
`p0` on 50--90 MeV.  At that mass the deliverable reports:

1. the shared-coupling fit and its implied fitted-window and full-template
   signal yields in each campaign; and
2. three independent signed, single-campaign diagnostic fits in the same
   conditioned covariance and count coordinate.

The independent signed fits are concordance diagnostics after selecting the
mass and are not three independent measurements.  The extraction is
selection-biased, and the minimum p-value is not look-elsewhere corrected.

## Claim boundary

No toys or expected bands are produced.  No global p-value, scan-wide
significance, unconditional coverage statement, expected sensitivity, or final
full-2021 exclusion is reported.  The 2016 support/range protocol uses the
already-open related 10% source, so all new asymptotic limits and local p-values
are conditional on a partially unblinded model-selection procedure and the
frozen GP states.  Full unblinding of the remaining data is planned separately.
