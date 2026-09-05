# Fixed-background injection and extraction study

Frozen before toy generation, 5 September 2026. This diagnostic assesses whether
the apparent precision of fixing the estimated GP mean survives uncertainty and
retraining. The observed scans are generated separately. No selection depends
on observed limits, pull values, or favorable toy outcomes.

## Scope and pairing

Test the 2021 10% sample at 65, 71, 78, 100, 182 and 231 MeV. The first three
cover the existing peak-dip question; the last three sample the middle and high
mass range, including the earlier high-uncertainty diagnostic. These are
descriptive, partly data-selected points, not independent validation regions.
Use the exact frozen v4.9.12 native histogram, support, per-mass GP coordinates,
nominal resolution, +/-2.25 sigma mask, and bin-integrated signal template.

Generate 500 pseudoexperiments per mass, generating ensemble, and injected
strength. Strengths are 0, 2, and 5 times the reference profiled-background
Fisher standard error. Both methods receive the same spectrum and injected
yield. Seeds are deterministic from master seed 491305 and the coordinates.
The target count-space yields are fixed before toys. Each mass is a separate
pointwise ensemble; no maximum over masses or global significance is estimated.

## Three generating ensembles

1. Known-background control: Poisson fluctuate the exact nominal GP mean plus
   the signal within the fit window. Both fits retain the same nominal mean;
   the profiled fit still uses its covariance. This deliberately ideal control
   tests implementation when the fixed-background assumption is true.
2. Conditional GP uncertainty: draw a background vector from the released
   Gaussian count mean/covariance, then Poisson fluctuate signal plus that
   vector. Both fits keep the same nominal mean/covariance. Reject and redraw
   an entire nonpositive vector, with a recorded count; never clip bins.
   This is an uncertainty-propagation diagnostic conditional on that model.
3. GP retraining: Poisson fluctuate a complete smooth spectrum plus the full
   signal template, refit the log-GP posterior outside the moving signal mask,
   and condition its covariance. The smooth truth is the saved v4.9.12.5
   reverse-injection background fitted outside 60-86 MeV using the 66 MeV
   kernel. Its data-selected origin is retained. Recompute count-dependent
   training errors and GP posterior in each toy, but keep the reviewed kernel
   coordinates frozen. This is not a hyperparameter-reoptimized or independent
   functional-form coverage ensemble. The injected signal's tails remain in
   training exactly as the ordinary mask permits.

## Inference and checks

Compare the exact fixed-mean Poisson fit and the Poisson likelihood with
correlated Gaussian background profiling, using the existing stable centered
deviance solver. Record signed fitted signal, fitted standard error, pull
(Ahat-Atrue)/sigmahat, signed local r, positivity and optimizer evidence.
For each positive injection evaluate bounded piecewise-asymptotic CLs at the
true injected yield: CLs(Atrue)<0.1 means that yield is excluded. Report this
conditional exclusion frequency with a binomial interval. Check the shortcut
against actual solved upper limits for the first three toys of each coordinate.
Do not call a result on these selected truths unconditional coverage.

## Proposed significance rescaling

Before toys, calculate the high-count omitted-covariance inflation

  kappa^2 = 1 + [(w/b)^T C (w/b)] / [sum(w^2/b)].

This is the variance ratio for the fixed-background linearized estimator when
Cov(n-b)=diag(b)+C. Evaluate r_fixed/kappa as a model-based local correction.
It is mass-dependent, addresses variance rather than bias, and is not a global
trials correction. Separately estimate the mean and width of background-only
r from toy IDs 0-99, then evaluate (r-mean_train)/width_train only on IDs 100-499.
This split-sample exercise assesses a local empirical calibration under each
specific generating ensemble. Never use observed r to choose a scale factor.
Neither correction is applied to observed reported p-values or limits.

Report raw, variance-scaled and split-calibrated false-positive fractions at
one-sided nominal p0=0.05. Five hundred toys do not resolve discovery tails.
Also compare the corrected fixed-estimator Fisher uncertainty with the
profiled GLS uncertainty; the comparison is within the assumed Gaussian model.
Preserve all masses, strengths and failures. One nice+10 single-thread process.
No production edits, additional unblinding, or global correction.
