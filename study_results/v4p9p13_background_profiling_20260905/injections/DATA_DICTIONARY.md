# Injection table interpretation

The main results use `derived/`; `pilot/` is separate and excluded from the
reported 27,000 spectra. Every mass, strength, and generating model has 500
toy IDs. Both methods use the same spectrum at each coordinate. Different
strengths and masses use independent seeds.

| Field | Meaning |
| --- | --- |
| `Atrue`, `Ahat`, `sigma_A` | Signal counts in the actual target window; signed estimate and fitted curvature error. |
| `strength_sigma` | Injection in reference profiled Fisher errors, fixed before generation. It is not the fixed fit's error. |
| `pull` | `(Ahat - Atrue) / sigma_A`. |
| `signed_r` | Signed square root of the null-versus-free likelihood ratio. |
| `p0_asymptotic` | `Normal.sf(max(signed_r,0))`; conditional local diagnostic. |
| `cls_at_true` | Bounded piecewise-asymptotic CLs evaluated at the true positive yield; undefined at zero yield in this diagnostic. |
| `true_yield_excluded` | `cls_at_true < 0.1`; zero injection is assigned false by the physical boundary and is not informative about coverage. |
| `shortcut_delta_ul` | Solved `A90 - Atrue` for the declared 216 full-limit checks; otherwise missing. |
| `exclusion_fraction` | Conditional frequency of excluding the injected yield, with exact 95% binomial bounds. |
| `false_positive_fraction` | Fraction with nominal local p0 below 0.05. It is a false-positive rate at strength zero and rejection power at positive strength. |
| `mean_signal_response` | `(mean_Ahat at injected strength - mean_Ahat at zero strength) / Atrue`. This compares separate ensemble means, not paired differences across strengths. |
| `kappa_reference` | Analytic omitted-covariance scale computed at the frozen reference mean/covariance. |

The local-calibration table uses toy IDs 0-99 to estimate an empirical mean and
width, then evaluates all methods on IDs 100-499. Its binomial intervals
condition on the single fitted training correction. The Fisher table is a
linearized variance calculation, not a tail calibration or observed correction.

All conditional results and numerical failures are retained. A successful
numerical validator does not turn the retrained 71 MeV closure failure into a
passing scientific result.
