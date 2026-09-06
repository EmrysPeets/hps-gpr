# Illustrative deficit extension to v4.9.16

This follow-up was requested after the excess scan. It preserves the sealed
v4.9.16 numerical products and original PDF. It changes no fit, limit,
background, mass membership, covariance, or toy spectrum.

At all 232 integer masses from 19 to 250 MeV, let r be the auxiliary signed
likelihood root, a its frozen stress-background Asimov value, s the frozen
response width, and z=(r-a)/s. A negative fitted amplitude is a diagnostic of
missing events with the signal-template shape, not a physical negative rate
or coupling.

Mirror the declared positive-root rule:

- Conditional local deficit probability: Phi(z) for r<0; otherwise 1.
- Principal deficit union score: max(-z) over coordinates with r<0;
  use -infinity when no coordinate is eligible.
- Separate raw-depth ordering: max_m max(0,-r_m).
- Raw-root Gaussian reference: Phi(r) for r<0; otherwise 1.

The last curve is an uncalibrated N(0,1) reference. Both local deficit curves
use the same raw-negative gate; no observed sign is changed. Raw-depth and
stress-centered deficit probabilities are different tests.

Reuse the 1,000 coherent joint Poisson scans and exactly replay the existing
200,000 GP realizations per method, using seed [49160906,method_index],
the original covariance eigendecomposition and 5,000-row batches. Require
bitwise reproduction of both archived positive-maximum vectors before
accepting the negative maxima. These are no additional independent toys.
Do not pool pilot scans, methods, directions, or repeated realizations.

Evaluate full-grid tails for both negative orderings and retain exact
exceedance counts, central 95% binomial intervals and one-sided 95% upper
bounds. Zero tails are bounds, not measured zero probabilities. Neither
these intervals nor the GP approximation include background-model uncertainty.
Check maximum distributions against the direct joint scans; central agreement
does not establish an extreme tail.

The main illustrative figure uses the profiled statistic and shows signed
roots and Asimov offsets, local deficit probabilities, and union-global
deficit probabilities. Fixed-background results remain a tabulated diagnostic.
Representative markers use the principal and raw-depth profiled extrema plus
30, 66, 76, 120 and 220 MeV. The full grid is retained regardless of those
descriptive selections.

This is an illustrative follow-up to an inspected excess scan. Its
direction-specific probabilities do not adjust for choosing between excesses
and deficits, methods, orderings, widths, kernels, or additional searches.
It establishes neither a discovery nor physical background validity,
expected sensitivity, confidence-limit coverage, or a continuous-mass
correction. All inherited 2016 qualifications remain.
