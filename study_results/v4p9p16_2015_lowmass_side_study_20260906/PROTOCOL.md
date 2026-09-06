# 2015 low-mass rising-edge side study

This exploratory extension was requested in a side conversation. All writes are confined to this derivative and its own PDF output directory. It does not change the frozen scans, production card, shared git state, or the ongoing presentation-extraction work.

## Choices before inspecting new signal fits

- Use the same full 2015 `invariant_mass` histogram as the v4.9.16 study. The original bins are 0.05 MeV. Sum groups of five, starting at the original zero edge, for 0.25 MeV likelihood bins, matching the GP analysis convention. Require integer counts and matching Poisson variances.
- Search 15--20 MeV in 0.25 MeV steps. Continue to 22 MeV only as a labelled bridge to the previously reported 21 MeV feature. Do not include this bridge when selecting the strongest feature in the requested interval.
- The nominal, short GP support is 12--28 MeV. The comparison supports are 12--26, 12--30 and 12.5--28 MeV. The lower edge is below every excluded window; the upper edges bracket the end of the rising spectrum. No support is chosen by its observed p-value.
- Use the archived log-mass/log-count Constant-times-RBF GP and lognormal moment transformation, with a moving exclusion of +/-2.25 sigma. Optimize the two hyperparameters on sidebands only with deterministic restarts. At each hypothesis use length-scale bounds [1,8] times log(1+sigma/m), inherited in spirit from the 2015 card but without full-range coupling. Keep the existing sigma(m) polynomial; below 19 MeV this is an extrapolated signal-shape assumption, not new detector calibration.
- Use the same dense Poisson likelihood and correlated Gaussian background constraint as the current analysis. Profile the background, keep the auxiliary amplitude signed, and require positive total expected counts. Local asymptotic excess p0 is normal-survival(max(r,0)); negative fits therefore have p0=0.5. This is not a global probability or a calibrated final search.
- As a different background-family cross-check, fit a bin-integrated exp(Chebyshev-5) background plus a bin-integrated Gaussian on each moving +/-7 sigma support (clipped at 12 MeV). Profile all polynomial coefficients using the Poisson likelihood. The order and window follow the form of the published HPS low-mass analysis, but the new mass interval is exploratory. Do not select between methods by p-value.
- Extract fixed anchors 15, 17 and 20 MeV and the strongest positive nominal-GP coordinate within 15--20 MeV, if distinct. Plot native inference bins and a second display with whole-bin grouping closest to half sigma, with fixed phase. Show background constraints and fitted components separately. Error bars represent counting uncertainty only.

## Bounded numerical and statistical checks

Verify input identity, fit scores, covariance positivity/loading, component sums, signal normalization and deterministic numerical repeats. Run ten pilot Poisson replicates at each displayed mass under its sideband-conditioned GP background. If this is quick, extend to 100 with disjoint IDs. Each replicate includes the full short-support spectrum, with GP hyperparameters optimized again on its sidebands. Retain the original observed window signal mask and pipeline. Record signed-root mean/width, empirical upper tails and finite-count confidence bounds. These conditional plug-in checks are not independent physical validation or full coverage calibration. The mass-local generating backgrounds must not be combined into a scan-wide p-value.

No epsilon-squared conversion, new exclusion, or claim of a heavy photon is made below 19 MeV: signal efficiency, resolution, acceptance turn-on, radiative fraction, background robustness and look-elsewhere accounting have not been qualified there. A short training support does not remove the trials factor from scanning masses or choices of background method.

### Numerical follow-up triggered by the optimizer boundary

The first nominal and support-comparison fits all reached the length-scale ceiling of eight resolution units. Before any wider-bound fit, declare additional ceilings of 16, 32 and 64 on the nominal 12--28 MeV support. Retain every result and assess likelihood-optimum stability and repeated starts, not favorable signal amplitudes. These checks do not replace the predeclared nominal curve or qualify the physical background. Also compare 8- versus 16-node quadrature for the independent polynomial likelihood at all extraction anchors and 21 MeV. Its goodness-of-fit deviance is a model check, not a calibrated discovery statistic.

Ceiling 16 removes every nominal-support boundary, and 32/64 recover the same likelihood optima. Repeat the same ten-then-100 conditional-toy checks for ceiling 16, using its own continuous sideband-conditioned means and seven optimizer restarts. Keep these toy banks separate from the ceiling-eight banks; each result is conditional on its generating background and fitting rule. This choice follows the optimizer boundary check rather than the p-value ranking.

## Report

Provide a LaTeX section, standalone section PDF, reusable PDF/PNG figures, numeric tables and an augmented copy of the latest available v4.9.16 report. Preserve all upstream sources. Inspect the rendered output using the PDF skill and bind products and sources in a manifest.
