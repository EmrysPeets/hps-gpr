# Scan-level p-value references for v4.9.12

This postprocessing adds analytic scan-level references to the 100- or 300-toy
report. It reads the exact observed result table and frozen analysis card from
`v4p9p12_final_dataset_combinations_20260902`, and activates the same attested
runtime to evaluate detector mass resolutions. It runs no fits or toys.
The target toy count identifies a report stage only: these analytic quantities
do not depend on toy count, band quantiles, or completion of that stage.

## Windows and reporting selection

Report each of the seven existing scopes over its exact mass grid, plus the
fixed maximal-available-dataset total window:

| Mass (MeV) | Selected scope |
|---|---|
| 19--38 | 2015 full |
| 39--49 | 2015 full + 2016 full |
| 50--90 | All three datasets |
| 91--180 | 2016 full + 2021 10% |
| 181--250 | 2021 10% |

The reported local reference is the smallest frozen analytic local `p0` within
each stated window. A scope-specific correction does not account for choosing
among scopes. Therefore the manifest also gives a separate Bonferroni result
for selection among all 680 scope-mass tests. The total-window curve contributes
no additional distinct tests to that family because its rows are a fixed subset
of those 680 tests.

## Resolution-spacing Sidak equivalent

Use the established v4-era postprocessing convention, now recomputed from
current inputs. At mass `m_i`, let `sigma_i` be the minimum mass resolution
among the datasets selected for that scope at that mass. With `W = 2.25`,

```
sigma_mid_i = (sigma_i + sigma_(i+1)) / 2
N_eff = clip(sum_i((m_(i+1) - m_i) / (W * sigma_mid_i)), 1, M)
p_Sidak = 1 - (1 - p0_min)^N_eff
Z_Sidak = NormalInverseSurvival(p_Sidak)
```

All masses and resolutions in the calculation use GeV. Here `M` is the number
of actual tested masses. `W` is the documented 2.25-resolution spacing, not
the 1.64 density window or twice the exclusion width. Adjacent endpoint
resolutions are averaged, including at the fixed composition transitions.
The resolution ledger records every interval's contribution, so the integral
can be reproduced without reconstructing a hidden effective count. The producer
cross-checks the frozen runtime resolutions against an independent evaluation
of the card's polynomials and 2016 tail prescription.

Numerically use `-expm1(N_eff * log1p(-p0_min))`. The Sidak formula is exact for
independent uniform local p-values. Replacing the count with a resolution-based
effective count in this correlated mass search is an approximation, not a
scan-toy calibration. The effective count does not incorporate selection of
models, supports, or dataset subsets from observed results.

The legacy algorithm is
`v4_wide_support_2015full_2016full_2021_10pct_20260803/postprocess_combined_bands300.py`,
function `effective_trials_from_spacing`; legacy observed values are not inputs.

## Actual-grid Bonferroni trials adjustment

For each stated window also report

```
p_Bonferroni = min(1, M * p0_min).
```

This uses all tested masses rather than an estimated effective count. The union
bound does not require independent mass tests, but its interpretation still
depends on the validity of each conditional asymptotic local p-value. It adjusts
the declared finite grid, not unsampled continuum masses or the broader model
development history. The separate 680-test family result uses the same formula
with the minimum among all seven scopes; no Sidak count is assigned to this
overlapping-scope family.

## Empirical trials calibration

The band protocol generates different masses independently. Equal toy IDs at
different masses therefore cannot be joined into coherent whole-spectrum scans.
No calibrated empirical global p-value is available from this ensemble; the
manifest records it as JSON `null` with an explicit reason. A dedicated global
calibration would need coherent background-only spectra and the maximum local
test statistic across every full scan, or a validated correlation-aware
alternative. It is outside this bounded band continuation.

The weak, strong, and two-sided limit-tail fractions remain fixed-mass
upper-limit diagnostics. They are not inputs to these analytic discovery-p0
scan corrections. All results inherit the frozen model assumptions, partially
unblinded development history, and disclosed 2016 numerical exception.

## Reproduction

```
python3 study_results/v4p9p12_combination_expected_bands_20260904/make_global_diagnostics.py --target-toys 300
```

The outputs are `derived/global_pvalue_summary_300toys.csv` (eight windows),
`derived/global_resolution_ledger_300toys.csv` (912 rows), and
`derived/global_pvalue_manifest_300toys.json`. The manifest includes exact source
and output SHA-256 hashes, the method, references, runtime cross-checks, and
the separate 680-test Bonferroni calculation. `--target-toys 100` creates the
corresponding analytic sidecars for the earlier report.

## References

- [SAS/STAT, p-Value Adjustments](https://support.sas.com/documentation/cdl/en/statug/66859/HTML/default/statug_multtest_details11.htm):
  Sidak formula and independence assumptions; Bonferroni adjusted p-values.
- [NIST/SEMATECH, Bonferroni's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm):
  the general probability union bound.
- [Gross and Vitells, Trial factors for the look elsewhere effect in high energy physics](https://arxiv.org/abs/1005.1891):
  scan-level excess probabilities and correlation-aware trials calibration.
