# Independent HEP/statistical review of the illustrative deficit extension

Reviewed 6 September 2026. This review used saved products only: no likelihood fits, random generation, or changes to the sealed v4.9.16 parent. The inspected analyzer SHA-256 is `236e180f54f9c088489f6975038d249951ad12d412a34e0cb93933f640afcf58`.

## Disposition and definitions

The mirror diagnostic is internally consistent. No formula, sign-gate, observed-order, covariance, or count error was found. It is acceptable as the declared conditional, illustrative follow-up; it does not supply a physical negative-signal interpretation or a correction for choosing a direction after inspecting the excess scan.

For `z=(r-a)/s`, the local rule `Phi(z)` for `r<0`, otherwise one, correctly retains the raw-negative boundary. Under the exact conditional Gaussian field, for `0 <= u < 1`, its null probability of being at most `u` is `min(u, Phi(-a/s))`, so the gate is conservative. The global principal score `max(-z : r<0)` and its empty-set value of minus infinity reproduce the event that at least one mass has a sufficiently small eligible local probability. Nonnegative observed roots have local and global probabilities exactly one. The separate raw-depth statistic `max(0,-r)` is a different ordering. All tail comparisons include equality.

The raw-root Gaussian reference is explicitly uncalibrated and uses the same negative-root gate. It should not be interpreted as a deficit discovery significance, a negative physical coupling, or a modification of the parent's pointwise CLs limits.

## Reuse and independent numerical audit

The implementation retains the full 232-point union and the archived joint spectra, response covariance, seed, factor construction and 5,000-row generation batches. Its required positive principal and positive raw maximum vectors replay bitwise for each method before accepting the negative maxima. These 200,000 GP realizations are reused; the 1,000 coherent joint Poisson scans are also reused. There are no additional independent toys, and pilots are not pooled into the validation sample.

`independent_audit.py` passed **939 checked conditions**, including all **820 parent manifest entries** and the input hashes. It independently reconstructed the direct negative maxima, all local probabilities and sign atoms, all GP/direct exceedance counts and binomial intervals at all 232 masses for both orderings and methods, representative selection, and the maximum-distribution KS diagnostics. The result is recorded in `independent_final_audit.json`. It checked the saved GP maxima and reviewed the deterministic replay source and passed gates; it did not regenerate the fields itself.

## Results and interpretation

| Principal deficit diagnostic | Profiled | Fixed |
|---|---:|---:|
| Peak mass, MeV | 83 | 84 |
| Observed signed root | -0.676096 | -1.423916 |
| Stress Asimov root | +7.706823 | +17.198206 |
| Response width | 0.982502 | 2.105264 |
| Standardized deficit score `-z` | 8.532211 | 8.845504 |
| Raw Gaussian reference probability | 0.249490 | 0.077235 |
| Gaussian-response local probability | 7.179e-18 | 4.556e-19 |
| GP global exceedances | 0 / 200,000 | 0 / 200,000 |
| Direct global exceedances | 0 / 1,000 | 0 / 1,000 |

The large centered deficit scores arise from comparing modest negative observed roots with strongly positive stress-background offsets. They probe this specified background construction. They are not measured 8.5–8.8-sigma physics effects. The profiled principal maxima reach only 5.765 in the sampled GP fields and 5.409 in the direct scans, below the observed 8.532.

For both methods, the one-sided 95% Monte Carlo upper bounds are **1.4979e-5** for the GP approximation and **0.0029913** for the direct stress ensemble. The central two-sided interval upper endpoints are instead 1.8444e-5 and 0.0036821. Zero exceedances do not resolve the analytic local tails or establish agreement there. These bounds omit background-model uncertainty.

The profiled raw-depth maximum is 3.490339 at 72 MeV. At that coordinate the stress Asimov root is -7.160215, so the observed deficit is shallower than the stress construction suggests; its Gaussian-response local deficit probability is 0.999913. Every simulated GP and direct raw-depth maximum exceeds the observed maximum. The fixed raw-depth maximum is 8.277901 at 35 MeV, also exceeded by every GP field and direct scan. These are saturated directional tests, not goodness-of-fit certificates.

The nominal principal maximum-distribution KS p-values are 0.43268 (profiled) and 0.82678 (fixed); the raw-depth values are 0.90689 and 0.99687. They identify no discrepancy in these finite-sample checks, but cannot certify the unobserved extreme tails.

## Required presentation boundaries

The figure and note should retain explicit zero-tail bound markers, distinguish the two orderings, and report the observed root alongside its stress offset and response width. The principal and raw extrema are descriptive markers on the unchanged full grid. This extension was chosen after inspecting the excess scan: neither direction-specific result adjusts for selecting excesses versus deficits, fixed versus profiled treatment, or raw versus centered ordering. No pooling or selection of the most favorable result is justified. All inherited 2016 and physical-background qualifications remain; expected sensitivity, interval coverage, background validity and continuous-mass inference are outside this diagnostic.
