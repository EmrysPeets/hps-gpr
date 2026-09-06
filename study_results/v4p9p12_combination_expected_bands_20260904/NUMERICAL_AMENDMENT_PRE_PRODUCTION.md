# Pre-production numerical amendment: centered fixed-strength profiling

This amendment was written after the 2-toy pilot and before any production
50-toy release was accepted.  It does not change the likelihood, statistic,
analysis inputs, toy draws, scopes, or release gates in
`STATISTICAL_PROTOCOL.md`.

## Trigger

The first full-grid attempt stopped fail-closed at the deterministic coordinate
`mass=25 MeV`, `toy_id=0`, `scope=individual_2015_full`.  The underlying 2015
pseudo-observation has SHA-256
`50cc683b7d1d7887c6a037f9799275c2c58a826e52aa97886e07e2b4a086a8f8`.
No toy was dropped, replaced, or assigned a different seed.

Near the candidate limit, the background-only Asimov fixed-strength profile
jumped between optimizer branches.  The corresponding CLs values straddled
0.10 with a discontinuity too large to satisfy the already-frozen absolute and
log-residual gates.  This was an optimizer failure: the raw Poisson negative
log likelihood carries a very large data-only offset, so an `ftol` stopping
test can terminate before the small profile-likelihood difference is resolved.

## Remediation

For fixed signal strength only, `band_solver.py` minimizes the Poisson deviance
plus the same Gaussian nuisance penalty.  This objective differs from the raw
negative log likelihood only by the data-only constant

`sum_i [n_i log(n_i) - n_i]`,

with the zero-count convention `0 log 0 = 0`.  Its gradient, nuisance model,
Cholesky factor, signal template, and fitted minimum are otherwise unchanged.
The original raw negative log likelihood is recomputed at the fitted nuisance
point before the existing bounded q-tilde and piecewise-asymptotic CLs mapping
is evaluated.

The centered minimizer uses deterministic zero initialization, L-BFGS-B,
`maxiter=2000`, `maxls=100`, `ftol=1e-14`, and `gtol=1e-8`.  A result is
accepted only if the optimizer succeeds, all fitted expectations stay above
the inherited likelihood floor, the analytic gradient is finite, and the
original likelihood-nesting and CLs root gates pass.  The band driver records a
distinct solver version and includes this amendment and the solver source in
its contract hash, so checkpoints from the failed pre-amendment attempt cannot
be mixed into the production ensemble.

The centered fixed-strength minimum is also a known feasible point in both the
bounded and unbounded free-strength parameter spaces whenever its strength is
nonnegative.  If it lies below an older free-fit optimizer result, the solver
uses it as the denominator candidate and records the fallback.  This is the
same likelihood-nesting reconciliation already used by the parent v4.9.12
solver for its null and bounded candidates; it can only replace a candidate by
a fitted point that is explicitly inside that candidate's parameter space.
