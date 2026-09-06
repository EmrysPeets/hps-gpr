# 90% CLs calibration protocol

Pre-execution protocol, 5 September 2026. The requested confidence level is 90%,
CLs=0.10. Monte Carlo error intervals are separate from this test size.

This derivative retains the fixed observed GP hyperparameters, masks, support,
resolution and count-to-epsilon-squared normalization of v4.9.12. It calibrates
the fixed-GP-mean and nominal Gaussian-profiled bounded q-tilde statistics under
specified full-spectrum Poisson generating truths. Every pseudoexperiment
recomputes the sideband log-GP posterior and its count-dependent alpha, with
full signal tails included in training. It does not optimize hyperparameters
again or rerun the prior support/model selection. Thus calibration is explicitly
conditional on these decisions and the generating family. The 2016 exception
remains. It does not establish global significance.

All 456 individual/all-three mass hypotheses use the native 1 MeV scan grids.
The combination uses one shared epsilon-squared parameter, independent spectra,
and the exact constituent conversion factors. No limits are combined numerically.
The generating families are mass-local GP mean and an archived alternative
shape; the 2021 alternative is the v4.9.5 fSigPowExpQ-anchored logistic-Chebyshev6
residual stress truth. The 2015 alternative is archived fShiftSigPowTail
expected counts; the 2016 alternative is the archived threshold-qualified
blend. Neither is a truth-independent background certificate; source-fit
limitations remain. The all-three envelope covers two declared joint scenarios
(all local GP, or all archived shapes), not every mixed constituent assignment.
Pilot outputs are not final results.

Use the same bounded likelihood ratio for observed data and every toy. CLs is
the ratio of upper q tails under S+B and B, with equality included. Generate
full Poisson spectra from a deterministic equal mixture of strength-specific
and tail-shifted proposals. Exact full-spectrum Poisson-density ratios to the
mixture reweight each toy to every tested signal strength (multiple importance
sampling). The full-spectrum influence approximation designs proposals only;
it is never used as a likelihood, test statistic or tail extrapolation. Keep
unshifted target proposals, use separate seeds for independent validation, and
report weight normalization, tail effective sample sizes and stratified Monte
Carlo standard errors. MC errors are assessed from the actual weighted draws,
not interpreted as binomial counts.

Literature: Read, CLs technique (CERN 2000; J Phys G 28, 2693, 2002);
Cowan et al., arXiv:1007.1727, bounded likelihood ratio;
Berns, arXiv:2303.11290 / Phys Rev D 109, 092002 (2024), reusing toys across
parameter hypotheses via a mixture density. This study applies the sampling
identity to CLs, rather than using Berns' Feldman-Cousins ordering.

Numerical optimization is checked against the archived scalar centered-deviance
solver. The toy GP can use joint eigenfeatures at relative cutoff 1e-15 only
if checks at every coordinate and both truths show mean discrepancies below
0.001 predictive SD, covariance differences below 0.001 maximum diagonal,
and absolute r/q differences below 0.001 (q tested at 2,5,12 reference sigma).
The same gate also checks a numerical nuisance-factor compression: remove
eigenvalues below 1e-5 from covariance whitened by Poisson count variance.
This bounds the discarded covariance relative to the Poisson scale by 1e-5;
actual r/q agreement must pass the stated 0.001 check. Retain up to 12 modes
per constituent with zero padding. Otherwise retain exact cached Cholesky
and the complete covariance factor at that coordinate. This
approximation is disclosed; observed models always use archived dense GPs.
All failures are retained and halt the affected execution. No toy is discarded.

The initial production proposal count is 256 each; independent validation uses
500 direct Poisson spectra at each of 0,2,5 reference sigma and each truth.
Truth-specific CLs inversions are reported separately. Their larger endpoint
forms an explicitly finite-family envelope; independent validation evaluates
the maximum CLs across the two truth families on each validation spectrum.
An endpoint is MC-resolved when both tail ESS exceed 100, its approximate 95%
MC half-width is <=10%, the numerical bracket is <=1.5%, and weight
normalization agrees within 5 SE or 5%. Failed gates remain visible, require
more toys or a censored/uncertain interval, and cannot be silently promoted.
Local p0 is calibrated separately from limits; p0=1 when the bounded discovery
statistic is zero. This empirical tied-atom convention differs from the
usual asymptotic Z=0 to p0=0.5 display. No global correction is inferred.

A completed study must include comparison with historical scaled-2021 closure,
source hashes, separate training/validation seeds, individual and combined plots,
tables, a typeset extension to the v4.9.13 note, and rendered-page QA.

Production source choice finalized after the historical review and before
production generation. Pilot version changes and all early output are retained
in separate pilot directories. Equal proposal strata include unshifted,
profile-influence shifted, and fixed-influence shifted full-spectrum means.
The broad proposal strengths are 0,0.5,1,1.5,2,3,4,5,6,8,12 reference sigma
plus an additional upper node when required by the observed fitted yield.
Importance proposal design may depend on the observed threshold to improve
efficiency; it never changes the generating truth or likelihood.

Bank spectra need not be retained: exact deterministic seed coordinates,
source hashes, whole-array SHA-256 and per-point traces are archived.
Validation toy statistics are retained. The reference released observed
Gaussian curve is shown alongside the same-solver nominal asymptotic control.

## Inversion and independent validation decision

Evaluate the complete declared strength-node grid, retain the last accepted
node, and refine the last bracket. Check all evaluated CLs values for increases
larger than either 0.01 or three combined Monte Carlo standard errors. A
nonmonotone result cannot be classified MC-resolved. Both B and S+B weight
normalization must pass; the numerical bracket must explicitly be <=1.5%
of the reported positive endpoint. A right-censored point is not a finite limit.
The slope-based MC endpoint interval is an approximate uncertainty diagnostic,
not an exact confidence bound on the limit.

On independent validation spectra, report exact binomial intervals for the
exclusion fraction at each positive injection and for the background-only
local rejection fraction at p0<0.05. Test excess exclusion against 0.10 and
excess local rejection against 0.05 with one-sided exact binomial tests and
Holm adjustment within each complete family of reported validation cells.
The description 'no detected undercoverage in this finite validation suite'
requires no Holm-adjusted p<0.05, retained counts, all numerical checks, and
a disclosed calibration MC resolution status. Passing this screen is not
proof of uniform coverage, nor evidence for an unconditional physics limit.
