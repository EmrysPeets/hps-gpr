# Release-local bounded `tilde_q_mu` CLs tail mapping

This directory fixes one narrowly scoped statistical issue without changing
the repository core or any accepted result artifact.

`hps_gpr.statistics.qmu_tilde_profiled_gaussian` correctly switches the
profile-likelihood denominator to the physical null when the unconstrained
signal estimator is negative.  However,
`hps_gpr.statistics.asymptotic_cls_profiled_gaussian` always applies the
square-root tail formula.  That formula is only the `q_obs <= q_A` branch.  A
negative unconstrained estimate gives `q_obs > q_A` and requires the bounded
piecewise mapping implemented in `bounded_tildeq_cls.py`.

The corrected mapping is

- for `q_obs <= q_A`: `z_sb = sqrt(q_obs)` and
  `z_b = sqrt(q_A) - sqrt(q_obs)`;
- for `q_obs > q_A`: `z_sb = (q_obs + q_A)/(2 sqrt(q_A))` and
  `z_b = (q_A - q_obs)/(2 sqrt(q_A))`;
- `CL_sb = sf(z_sb)`, `CL_b = Phi(z_b)`, and
  `CL_s = CL_sb / CL_b`.

The ratio is evaluated in log space.  No arbitrary `CL_b` floor is used.
Every result should retain `q_obs`, `q_A`, the selected tail branch, the log
tails, and the likelihood-optimizer status.

## Combined-driver integration

The existing cached combined driver already computes `qmu_obs` and
`qmu_asimov_b`.  Its tail-conversion block should import and call only:

```python
from bounded_tildeq_cls import bounded_tildeq_asymptotic_tails

tails = bounded_tildeq_asymptotic_tails(qmu_obs, qmu_asimov_b)
cls_value = tails.cls
```

Alternatively, an uncached call can replace
`hps_gpr.statistics.asymptotic_cls_profiled_gaussian` with
`asymptotic_cls_profiled_gaussian_piecewise`; the signature and first three
return values are compatible, and the fourth value contains branch and
convergence metadata.

For `combined_mode=count_scale`, keep the existing exact coordinate change:
normalize the nonnegative concatenated counts-per-epsilon-squared vector to
unit sum, test the corresponding total signal count, and divide the solved
count strength by the original sum.  The tail mapping is applied to the two
test-statistic values and does not alter that shared-coupling construction.

## Other CLs branches audited

- The bounded `tilde_q_mu` statistic itself uses the correct null denominator
  for a negative unconstrained estimator.
- The empirical toy-CLs path compares the same statistic to background and
  signal-plus-background toy tails, so it does not use the missing analytic
  branch.  Its finite-toy and conditional-model caveats remain separate.
- The `q_obs <= q_A` analytic branch agrees with the existing implementation.
- The discovery `q_0`/local-`p_0` path is a different one-sided statistic and
  is not changed here.
- Root finding must fail closed on non-finite tails, missing Asimov separation,
  or unsuccessful likelihood profiles, and should record the branch at the
  solved limit.  Existing production wrappers do not all expose those checks,
  so the new combined release must validate them explicitly.

## Regression test

From the repository root:

```bash
python3 -m pytest \
  study_results/bounded_tildeq_cls_tail_mapping_20260902/test_bounded_tildeq_cls.py \
  -q
```

The tests cover both analytic branches, continuity, log-tail stability, an
actual negative-estimator profile, agreement on the unaffected branch, and a
fixed example that demonstrates the old/new difference.
