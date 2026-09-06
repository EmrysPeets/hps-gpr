# v4.9.7 2016 support and combined-limit protocol

Frozen before evaluating any 2016 support-scan extraction on 2026-09-02.

## Scope

The study changes only the lower GP data-support edge for the 2016 full
dataset. It retains the v4.2/v4.1 2016 search range, resolution model,
length-scale factors, blind/training exclusion, rebinning, optimizer settings,
signed extraction, and 90% asymptotic CLs construction. After freezing the
2016 edge, it rebuilds the exact shared-nonnegative-epsilon-squared
2015-full + 2016-full + 2021-10% result using the already frozen 2015 states,
the new 2016 states, and the v4.9.5 2021 states.

## Threshold truth

- Shape source: the independent 2016 10% development histogram
  `h_Minv_General_Final_1`.
- Common truth envelope: 26--210 MeV.
- Normalization: one scalar normalization to the 2016-full observed count in
  that complete envelope (73,145,594); no candidate-support-specific
  renormalization is permitted.
- Local threshold family: logistic turn-on times the exponential of a
  Chebyshev polynomial over 26--80 MeV. Candidate degrees 4--10 are compared
  using only source goodness of fit. The lowest degree passing raw Poisson
  deviance/ndf <= 1.5, five-bin deviance/ndf <= 2.0, maximum absolute
  five-bin pull <= 5, finite optimization, and no parameter-bound contact is
  selected.
- Above 85 MeV the already fitted broad `fShiftSigPowTail` expectation from
  the same 10% source is used. A quintic smootherstep blends the two means
  from 75 to 85 MeV, after which the complete mean is normalized once.
- One hundred independently seeded Poisson spectra are generated once and
  paired by toy index across every support edge, mass, and injection.
- Leave-one-2.25-sigma-window-out refits at 44, 49, 54, 59, and the 65 MeV
  holdout are diagnostics only and cannot select the truth degree.

This is a source-conditioned stress truth, not a physical background
generator or a direct coverage model.

## One-factor support scan

- Search range: 39--180 MeV.
- Fixed upper data-support edge: 210 MeV.
- Lower-edge grid: 28, 29, 30, 31, 32, 33, and 34 MeV.
- Freeze-eligible edges: 28--33 MeV. The 34 MeV point is a declared geometry
  control because it leaves only four rebinned low-side training bins at the
  39 MeV endpoint.
- Threshold masses: 44, 49, 54, and 59 MeV.
- Matched-reference injected strengths: 0, 2, and 5 sigma.
- The 65 MeV point is excluded from selection and evaluated only after the
  support edge is frozen.
- The native 0.05 MeV bins are rebinned by five. Integer-MeV support steps
  therefore retain a common 0.25 MeV coarse-bin phase.

All branches are selected only through covariance validity, kernel state,
reproducibility, and maximum GP log marginal likelihood. Amplitude, pull,
signal recovery, epsilon-squared, p-value, and upper-limit strength are never
branch criteria.

## Cohorts and freeze rule

Phase 1 uses toy indices 0--24 at all seven edges. An eligible edge must have
complete finite rows, valid covariance, reproducible selected branches, no
accepted kernel-bound contact, at least 9/12 cell means with absolute mean
pull below 0.75 including at least 3/4 background-only cells, and no cell with
absolute mean pull at or above 1.25. Qualifying edges are ranked by their
worst absolute cell mean. Edges within 0.10 are tied and the smaller edge is
selected to retain more data support.

Phase 2 uses indices 25--99 only at the provisional edge and its immediate
neighbors. The same practical rule must pass separately in the continuation
cohort and the full-100 ensemble, and every full-100 cell must retain at least
95 accepted fits. Confirmation failure ends the study without retuning.

Per-toy bounded Wald CLs limits may diagnose numerical instability but cannot
rank support edges. Only after a successful freeze may the 2016 full observed
histogram and the 65 MeV holdout be evaluated.

## Combined result and bands

The combined likelihood is evaluated over the 19--250 MeV union grid with
`combined_mode: count_scale`, which is an exact numerical reparameterization
of the shared nonnegative epsilon-squared likelihood. The conditional band
uses exactly 100 background-only pseudoexperiments per mass, seed 24680, the
reviewed fixed GP state for each active dataset, a GP-posterior latent-rate
draw followed by Poisson counts, and an inner 90% asymptotic CLs limit. The GP
is not refit inside the band toys.

The bands are conditional fixed-GP expected-limit quantiles, not direct
coverage, toy-calibrated inner CLs, or a global-significance ensemble. The
mass-local toys are not coherent scans across mass.
