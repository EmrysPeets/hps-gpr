# Conditional rate fits and uncertainty bands

The exact 92 MeV, fully scaled-width likelihood gives:

| Rate model | Reference amplitude at 2.3 GeV | Slope | Raw sqrt(Q0) |
|---|---:|---:|---:|
| Constant common coupling | 3.10168e-6 | — | 2.44664 |
| Power law | 8.51608e-6 | beta = -2.98691 | 4.18024 |
| Decaying exponential | 9.81672e-6 | k = 1.68057 GeV^-1 | 4.14585 |

Here the power law is `a*(E/2.3 GeV)^beta` and the true exponential is `a*exp[-k*(E-2.3 GeV)]`. They are distinct response models. `a` is an equivalent epsilon-squared amplitude under the inherited prompt normalization, not a newly identified physical coupling.

The full profiled observed Hessian is evaluated in `(log a, slope)`. It contains the second derivatives of the nonlinear signal expectation and the Schur complement for all Gaussian-constrained background nuisance parameters. The bands are **pointwise approximate 68% and 95% log-Wald bands, conditional on fixed mass, fixed fully scaled widths, frozen GP inputs, and each specified rate law**. They neither assess the physical validity of that law nor calibrate its discovery statistic. They are not simultaneous confidence bands.

The slope standard errors from this Hessian are 0.50733 (power) and 0.40539 GeV^-1 (exponential). The exact one-parameter profile-likelihood intervals, using nominal asymptotic 68% thresholds, are [-3.51700,-2.46963] and [1.25673,2.08688] GeV^-1, respectively. Both remain interior to the declared parameter bounds. Joint parameter contours in the CSV contain exact twice-profiled-NLL differences, suitable for 2.279/5.991 nominal 68%/95% two-parameter contours; they should not be confused with the pointwise prediction bands.

Validation: 121-point slope scans locate all sampled basins and their bounded scalar refinements; a 241-point independent grid checks the selected minimum. Common and power fits replay the parent Q0 within 1.4e-12 and amplitude within 2.8e-8 relative. Finite differences at two steps check every Hessian; the larger-step maximum relative error is 1.23e-6. All 9,882 exact contour profiles pass the inherited solver's convergence check, with maximum score below 2.0e-7. The exact contour grids are 81 amplitude points by 61 slope points. Their low-amplitude span is extended to at least 2.8 log units below the fit to contain the nonquadratic tails. Every outer-edge point lies beyond the nominal 95% contour: the smallest edge delta(2NLL) is 6.781 for the power law and 7.345 for the exponential, versus the 5.991 threshold. Both nominal 68% and 95% contours therefore close inside the grids. Three band grids contain 726 rows total, including all actual beam energies. Runtime is approximately 10 seconds with one BLAS thread.

Reproduce with `python3 scripts/rate_uncertainties.py` from this package. Outputs are `derived/rate_band_fits.json`, `derived/rate_bands.csv`, `derived/rate_parameter_contours.csv`, and `qa/rate_band_validation.json`. Parent files and the shared experiment engine are not modified.
