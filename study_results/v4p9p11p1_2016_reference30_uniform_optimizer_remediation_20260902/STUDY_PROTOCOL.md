# v4.9.11p1 prospective uniform numerical optimizer remediation

Frozen after v4.9.11 was finalized as a terminal failed certification and
before any v4.9.11p1 optimizer path or selected state was evaluated.

## Scope and inherited model

This study changes no physics or background-model choice.  It retains the
pre-existing reviewed 2016 reference card exactly:

- support 30--210 MeV;
- search hypotheses 39--180 MeV in integer-MeV steps;
- resolution-scaled local Constant-times-RBF kernel with 2016 lower factor
  0.9, upper factor 12, and dataset-stat high-bound floor factor 0.8;
- five-bin rebinning, `x=log(m)`, `y=log(count)`, `alpha=1/count`;
- `mass +/- 2.25 sigma_2016(mass)` excluded from GP training; and
- two nonempty training sidebands required.

The canonical v4.9.11 support-30-only full low-control adequacy result is
inherited by exact hashes and is not rerun.  It used no center at or above
38.75 MeV, passed all 20 technical cells, and passed its predeclared absolute
guards.  v4.9.11 subsequently stopped because only 49/142 production states
passed every deliberately strict numerical certification gate.  Its terminal
decision remains valid and is not retroactively changed.

v4.9.11p1 addresses only optimizer numerics.  It applies one prospectively
fixed algorithm to all 142 masses, including the 49 that previously passed.
There is no failure-only, anomaly-only, or mass-specific repair.

## Immutable inputs and permitted prior coordinates

The immutable full-2016 input, reviewed k12 card, archived v4.1 state ledger,
v4.9.11 426-attempt ledger, and v4.9.11 terminal state ledger/decision are
bound in `study_spec.json`.  Before selection, only mass, kernel coordinates,
LML, optimizer warning/status, training geometry, and prediction/covariance
fields may be read from earlier results.  Signal amplitude, p0, Z, limits, and
their ordering are forbidden.

Prior coordinates are starts only.  They are never automatically accepted:
each final coordinate must be the post-polish result of an eligible v4.9.11p1
path and pass all gates below.

## Uniform deterministic start set

At every mass, construct the same labeled start set in the bounded two-
dimensional log-coordinate space `(log constant, log length)`:

1. the archived reviewed coordinate;
2. each of the three finite v4.9.11 seeded-attempt coordinates;
3. the card initializer: constant 1 and geometric-mean length bound; and
4. a fixed 3-by-3 lattice: constants 10, 100, and 1000 crossed with length
   positions 0.1, 0.5, and 0.9 of the log-bound interval.

Source starts are clipped only when necessary to `1e-8` of the log-bound span
inside a boundary.  Every labeled path is retained even when two numerical
starts coincide, but path-cluster independence below is based on optimizer
method family, not duplicate start labels.

## Uniform optimizer paths

Every mass runs all of the following paths; none is conditionally added after
seeing a failure:

- `direct_lbfgsb`: analytic-gradient L-BFGS-B from every labeled start;
- `powell_lbfgsb`: bounded derivative-free Powell from both the card start and
  the exact-fixed-LML-best prior-source start, followed by analytic-gradient
  L-BFGS-B from the Powell coordinate; and
- `trust_lbfgsb`: bounded `trust-constr` from both the fixed lattice-center
  start and the exact-fixed-LML-best prior-source start, followed by the same
  analytic-gradient L-BFGS-B.

The prior-source start comparison is fit-only and is performed by freshly
reconstructed fixed-coordinate LML on the exact training array.  It cannot use
an optimizer-reported stale objective.

L-BFGS-B uses `maxiter=3000`, `maxls=200`, `ftol=1e-12`, and `gtol=1e-8`.
Powell uses `maxiter=2000`, `maxfev=20000`, `xtol=1e-10`, and `ftol=1e-12`.
`trust-constr` uses `maxiter=1000`, `gtol=1e-8`, `xtol=1e-10`, and
`barrier_tol=1e-10`.  All objectives use float64 and the same exact GP LML.

For a direct path, the L-BFGS-B result is the post-polish coordinate.  For a
Powell or trust path, the following L-BFGS-B result is the post-polish
coordinate.  Pre-polish coordinates and their diagnostics are recorded but
cannot be selected.

## Path eligibility and exact branch selection

A path is eligible only if every stage it invokes:

- terminates with optimizer `success=true`;
- emits no Python/scipy/sklearn warning and no exception;
- returns finite post-polish coordinates strictly interior by the unchanged
  relative `1e-3` bound guard; and
- at its post-polish coordinate has finite exact fixed-coordinate LML,
  agreement between optimizer objective and fixed LML within `1e-6`, and
  analytic log-coordinate KKT/gradient infinity norm strictly below `0.01`.

The candidate branch is the eligible path with the largest freshly evaluated
fixed LML.  It is not chosen by optimizer status ordering, historical source,
path multiplicity, prediction residual, signal amplitude, p0, Z, or limit.

The selected maximum is certified only if its cluster contains at least two
eligible paths whose fixed LML differs from the maximum by at most `1e-4` and
whose constant and length coordinates each differ by at most relative `1e-3`.
The cluster must contain at least two distinct optimizer method families among
`direct_lbfgsb`, `powell_lbfgsb`, and `trust_lbfgsb`.  Thus two duplicate starts
within one optimizer cannot by themselves certify a branch.  A lower-LML
branch is never substituted because it has more paths.

This explicitly covers known fit-only branch splits such as 45 and 143 MeV
without giving either mass special treatment.

## Selected-state reconstruction and production gates

Only the selected post-polish coordinate is reconstructed with optimizer off.
The reconstructed LML must reproduce the selected fixed LML within `1e-6`.
Scikit-learn latent mean/covariance must agree with an independent direct-
Cholesky calculation to relative `1e-9` and absolute `1e-10`.  The lognormal
count mean/covariance plus Poisson diagonal must be finite/SPD with no negative
eigenvalue below `-1e-8 max(diag(V))` and no relative Cholesky jitter above
`1e-8 median(diag(V))`.  Exact train/query center/count hashes, bounds,
coordinates, LML/gradient, all optimizer stages/warnings, and prediction hashes
are recorded.  Both training sidebands must be nonempty.

Every one of the 142 masses must pass all path-cluster and selected-state
gates.  One unresolved mass stops the entire ledger; there is no retry, method
addition, tolerance change, or interpolation after execution.

## Inference boundary

v4.9.11p1 computes no signal amplitude, p0, Z, upper limit, expected band, toy,
or global significance.  Only after an independent validator passes a complete
142-state ledger may another separately versioned release use the states for
inference.  Any limit must use the independently validated piecewise bounded
`tilde_q_mu` CLs mapping and expose `q_obs`, `q_A`, analytic branch, and solver
status.

Any later inference remains fixed-model asymptotic inference conditional on a
partially unblinded model history.  This is not unconditional coverage,
independent blinding, high-side predictive validation, or global significance.
