# Final bounded statistical audit

Reviewed `scripts/rate_significance_scan.py`, `scripts/new_figures.py`, their numerical ledgers and the accepted v5.5.3 reference. No blocking statistical or implementation issue was found.

## Gaussian rate-law tail

With three independent standardized campaign scores z, write z=R u. Under the null, u is uniform on the sphere and R² follows chi-square with three degrees of freedom, independently of u. For normalized model directions v(s), the profiled nonnegative-amplitude statistic is R² max(0,max_s u·v(s))². At q>0 its tail is therefore the sphere average of chi²_3.sf(q/c²) for c=max_s u·v(s)>0, zero otherwise. This is exactly the integrand used. The Gauss–Legendre weights and uniform azimuth weights integrate normalized sphere measure, not unnormalized solid angle.

The power-law scan includes beta in [-6,6], allowing either slope sign. The exponential includes k in [0,6] GeV^-1, so only decay is allowed. Both include the common-coupling direction. The tail integrates the slope search under the null; it does not mistake the fitted slope for an externally fixed law. Mass and fully scaled widths remain fixed for each plotted point. Neither a mass search nor width optimization is part of this new tail calculation.

Independent bounded checks:

- For every one of the 34 mass/model combinations, a separate 20,001-point slope grid found no Gaussian maximum above the reported continuously refined maximum. The largest grid-minus-optimizer difference was -4.20e-11 in Q.
- At 92 MeV, a fixed rotation of the quadrature axes changed the power-law tail by 2.96e-7 fractionally and the exponential tail by -3.13e-8.
- Saved angular refinement changes are at most 1.57e-5; saved 257-to-513 slope-grid changes are at most 1.051e-4. Known fixed-direction normal and orthant chi-bar benchmarks pass.
- Both fitted rate-law Gaussian Q values remain below the independent-positive-amplitude Q throughout the grid, as nesting requires.

The tail uses a finite 513-node slope bank while the observed Q uses continuously refined slopes. Refinement supports numerical accuracy, but does not mathematically prove equality to the continuous family. Retain the finite-bank quadrature qualifier; do not call it a rigorous continuum probability bound. The Gaussian covariance remains the frozen predictive/joint-auxiliary parent covariance, not direct Poisson/sideband calibration.

## Exact count Q versus Gaussian extracted Q

`p_reference_at_exact_Q` evaluates this Gaussian tail at the observed Poisson profile statistic. `p_gaussian_statistic` evaluates it at the matching Gaussian statistic. The figure columns select the appropriate fields and label the resulting values as Gaussian references. At 92 MeV the power-law exact-Q reference is p=6.9647411e-5, Z=3.809417; the pure Gaussian value is p=6.9000506e-5, Z=3.811724. The former differs from the prior Monte Carlo estimate by about 1.06 prior Monte Carlo standard errors, so there is no resolved conflict.

The left count-spectrum column overlays extracted Stouffer and Fisher results as explicitly labeled comparisons. Those overlays are not additional Poisson-profile fits. Signed probabilities are used correctly.

## Opening comparison

The two upper panels use the same estimand: total expected reconstructed signal rows in the three fixed fitting windows. The right panel's signed-sum Gaussian CLs bound does not assume relative rates; it is not a universal epsilon² bound. Both main upper curves use the same Gaussian CLs convention, with the common-law Poisson-profile curve separately labeled.

The right probability panels correctly use independent-positive-amplitude, Stouffer, power-law and exponential pointwise references. The inherited N=35.381 Sidak operation is visibly labeled illustrative. It has not been calibrated for these alternative tests, the 90–94 MeV grid, or subsequent selection among rules. Choosing the smallest displayed p requires additional correction. The common and unrestricted curves do not prove a signal in all three campaigns separately.

This audit used deterministic calculations only; no new Monte Carlo sampling or broad study was run.
