# Archived ROOT-family implementation audit

This is an implementation/optimizer rejection audit of the existing
`root_macros/funcform/funcform_common.h` machinery, not a mathematical proof that
the named formulas can never fit under a different parameterization or fitter.

Each source was refitted at lower edges 30, 35, 40, and 50 MeV with an upper edge
of 300 MeV.  The audit recomputed native Pearson and Poisson-deviance ratios,
factor-five versions, the largest absolute factor-five Pearson residual, fit
validity, and normalized parameter-bound distance from the analytic seed arrays.
The 64-row machine-readable ledger is
`derived/archived_root_family_edge_audit.csv`; its input metadata are archived
under `reference/root_family_edge_audit_metadata/`, and the ledger records the
SHA-256 of every temporary ROOT audit product used for the derived metrics.
The deliberately lenient audit gate required a valid fit, native Pearson and
deviance <=1.5, factor-five values <=2, maximum factor-five residual <=5, and
normalized bound distance >=1e-4.  No family/edge passed both sources.  The more
restrictive v4.8 qualification protocol reaches the same disposition.

The table reports the smallest native Pearson ratio found over the four edges;
it does **not** endorse that edge.  Dropping threshold bins mechanically improves
many scores, so an edge must instead pass the full predictive/support protocol.

| ROOT family | 1% best edge | 1% Pearson / deviance | Native-10% best edge | Native-10% Pearson / deviance | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| `fSigPow` | 50 MeV | 4.658 / 4.672 | 50 MeV | 35.708 / 35.536 | rejected |
| `fShiftSigPow` | 50 MeV | 14.654 / 13.725 | 50 MeV | 155.987 / 147.168 | rejected |
| `fShiftSigPowTail` | 50 MeV | 1.347 / 1.355 | 50 MeV | 4.198 / 4.216 | rejected; 1% fit invalid and rebin-5 fails |
| `fGenGammaThresh` | 50 MeV | 1.167 / 1.168 | 50 MeV | 3.324 / 3.325 | rejected; invalid/bound-adjacent |
| `fGenGammaShift` | 50 MeV | 1.504 / 1.511 | 50 MeV | 7.516 / 7.521 | rejected |
| `fEndpoint` | 50 MeV | 10.183 / 9.582 | 50 MeV | 92.968 / 88.514 | rejected |
| `fLogPolyThresh` | 50 MeV | 2432.414 / 2712.578 | 35 MeV | 65871.467 / 67121.898 | rejected |
| `fBern5` | 50 MeV | 1050.210 / 1195.900 | 30 MeV | 26836.832 / 11228.910 | rejected |

The requested shorthand also has two distinct repository interpretations:

- literal five-parameter `fSigPow`, shown above; and
- seven-parameter `fSigPowExpQ`, whose archived support40 fits have Pearson
  chi-square/ndf 1.572 (1%) and 6.167 (native 10%), put `c1` at the +50 bound,
  and report `fit_ok=false` in both records.

The seven-parameter form contains an additional structural degeneracy:
`exp(-m/theta) * exp(c1*m)` combines into a single linear exponential slope, so
`theta` and `c1` are not independently identified by that factor.  Enlarging the
`c1` bound is therefore not a principled repair.

This audit also explains why the old `primary_validation_pass` must not be used
as a source-GOF decision.  The current validation checks aggregate normalized
totals and sideband fractions, while the candidate ranker can retain the lowest
Pearson candidate even when the minimizer fit is invalid or a parameter is at a
bound.
