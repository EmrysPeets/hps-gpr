# 100-toy continuation numerical amendment: centered free-profile retry

This amendment was written after the accepted 50-toy stage and during the
cumulative extension to 100 toys per mass. It does not change the likelihood,
test statistic, analysis inputs, pseudo-observations, seeds, scope definitions,
or release gates in `STATISTICAL_PROTOCOL.md`. The amended solver has a new
version and the amendment and source are included in a new contract hash; the
100-toy release is recomputed under that contract rather than mixing
checkpoints.

## Trigger

The first continuation attempt stopped fail-closed at the deterministic
coordinate `mass=177 MeV`, `toy_id=86`,
`scope=individual_2021_10pct`. No toy was dropped, replaced, or reseeded. The
unbounded free-strength L-BFGS-B result was finite, had
`A_hat=-200.6225727` fitted-window events and
`NLL=-49600116.43357901`, but carried an unsuccessful optimizer status. The
bounded and null fits both succeeded at `NLL=-49600116.431194216`.

An independent diagnostic minimized the data-constant-centered objective for
the same unbounded free-strength fit, scaling the amplitude coordinate by the
raw profile-information uncertainty. It converged in five iterations with
`A_hat=-200.4356128`, `NLL=-49600116.43357902`, minimum fitted expectation
`112729.5440`, and finite analytic gradient. The raw and centered NLLs agree to
approximately `1e-8` on an objective of magnitude `5e7`; both select the same
negative-amplitude branch and lie below the bounded fit as required.

## Remediation

`band_solver.py` now retries a free-strength profile only when the inherited raw
candidate is unsuccessful or non-finite. The retry minimizes half the Poisson
deviance plus the unchanged Gaussian nuisance penalty. This differs from the
raw Poisson negative log likelihood only by the data-only constant

`sum_i [n_i log(n_i) - n_i]`,

with `0 log 0 = 0`. The signal amplitude is optimized in units of the raw
profile-information uncertainty to avoid a badly scaled joint coordinate.
The original raw NLL is recomputed at the retry point before any likelihood
ratio is evaluated.

The retry uses the raw finite candidate as its deterministic starting point
when available, otherwise the bounded/null feasible point. It uses L-BFGS-B
with `maxiter=2000`, `maxls=100`, `ftol=1e-14`, and `gtol=1e-8`. A retry is
accepted only if the optimizer succeeds, the fitted expectations remain above
the inherited likelihood floor, all values and gradients are finite, and the
existing likelihood-nesting, CLs monotonicity, bracket-orientation, and root
residual gates all pass. Successful inherited free fits are not reoptimized.

The solver records bounded, unbounded, and null centered-retry counters. The
release validator requires exact preservation of the original 50-toy numeric
prefix (allowing only the solver-version and zero-valued new counter metadata
to differ), records every activated retry, and continues to reject dropped or
reseeded toys.
