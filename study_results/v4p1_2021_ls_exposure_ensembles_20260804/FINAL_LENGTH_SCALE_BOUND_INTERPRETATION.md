# Final length-scale bound interpretation

## Decision

Factor 20 clears scan and injection-refit boundary contact in all projected-100% lanes, but factor 25 is the universal common choice because factor 20 retains contact in diagnostic 1% lanes.

Factor 20 has zero scan and zero injection-refit boundary rows in all projected-100% lanes. Across all diagnostic lanes it retains 42 scan rows at the declared boundary; factor 25 has zero boundary and zero near-bound rows across every lane.

## Projected 2021 100% diagnostics

| Truth | Scenario | f | Scan bound | Injection bound | sigma_A/f15 | Response | Deficit | Residual [anchor sigma] | Pull width |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | 2021 1% × 100 | 15 | 14/110 | 11/120 | 1.000000 | 0.973801 | 2.620% | +0.167 | 1.055 |
| primary | 2021 1% × 100 | 20 | 0/110 | 0/120 | 0.999993 | 0.973786 | 2.621% | +0.209 | 1.054 |
| primary | 2021 1% × 100 | 25 | 0/110 | 0/120 | 0.999997 | 0.973932 | 2.607% | +0.244 | 1.091 |
| primary | 2021 10% × 10 | 15 | 4/110 | 0/120 | 1.000000 | 0.974565 | 2.543% | -0.215 | 1.222 |
| primary | 2021 10% × 10 | 20 | 0/110 | 0/120 | 1.000007 | 0.975646 | 2.435% | -0.215 | 1.214 |
| primary | 2021 10% × 10 | 25 | 0/110 | 0/120 | 0.999997 | 0.974415 | 2.559% | -0.214 | 1.222 |
| alternate | 2021 1% × 100 | 15 | 38/110 | 29/120 | 1.000000 | 0.971918 | 2.808% | +0.116 | 1.037 |
| alternate | 2021 1% × 100 | 20 | 0/110 | 0/120 | 1.000000 | 0.971921 | 2.808% | +0.185 | 1.114 |
| alternate | 2021 1% × 100 | 25 | 0/110 | 0/120 | 0.999999 | 0.971933 | 2.807% | +0.105 | 1.114 |
| alternate | 2021 10% × 10 | 15 | 77/110 | 76/120 | 1.000000 | 0.979598 | 2.040% | +0.039 | 0.936 |
| alternate | 2021 10% × 10 | 20 | 0/110 | 0/120 | 0.985677 | 0.979603 | 2.040% | +0.023 | 1.019 |
| alternate | 2021 10% × 10 | 25 | 0/110 | 0/120 | 0.985673 | 0.979606 | 2.039% | +0.054 | 0.992 |

The paired-response deficit spans 2.04-2.81%. The common roughly 2-3% deficit persists after boundary contact is removed at factors 20 and 25, so it is a response diagnostic, not a bound-induced sensitivity change.

From factor 20 to 25, the largest median length-scale change is 0.0065%, the largest fitted-sigma_A increase is 0.0004%, and there are zero adjacent-LML regressions beyond the audited absolute-plus-relative tolerance. Factor 25 is an optimizer plateau relative to factor 20 and shows no fitted-uncertainty sensitivity degradation.

Pull widths and anchor-normalized residuals are screening diagnostics, not coverage qualification.

## Observed optimized length-scale medians

| Dataset | Median ell/sigma_x | Ceiling contact |
|---|---:|---:|
| 2016 10% | 10.782324 | 0/142 |
| 2016 100% | 9.715735 | 0/142 |
| 2021 1% | 13.061448 | 7/201 |
| 2021 10% | 12.060030 | 0/201 |

## Independent native-10% versus 1%-source ×10 comparison

The source-family ensembles are independent; toy indices are not paired and no paired difference or ratio is reported.

| Truth | Ensemble | Toy-median ell/sigma_x | Toys | Masses/toy |
|---|---|---:|---:|---:|
| primary | 1%-source x10 | 11.564359 | 10 | 11 |
| primary | native 10% | 11.807978 | 10 | 11 |
| alternate | 1%-source x10 | 16.501223 | 10 | 11 |
| alternate | native 10% | 18.013141 | 10 | 11 |

Source-support ratio (10%/1%): 11.296466; effective expected-count ratio: 1.129647.

## Scope

Ten-toy screening pilot only; mass rows within a toy are correlated. No coverage qualification, expected-limit bands, qmu interpretation, exclusion, or reach claim is made.
 The single coherent one-sided qmu-zero diagnostic remains qmu_ok=false and all qmu outputs are excluded and non-promotable.

## Provenance

- `derived/v4p1_ensemble_postprocess_manifest.json`: `5928fc164c164245ec8e9f6742e06657a1cd9e93961e2ed77f9b62e7a18f9a6e`
- `derived/scan_optimizer_audit_summary.json`: `5333efd06f3e703a03c3eab9bba6d0ae1ea129046fb63b4ab62e1249a3e75178`
- `derived/scan_reviewed_rows_complete.csv`: `16390d0e932609d9c5b8a7ed59b4848e7f596a9e8888b558323583857f9b5ebc`
- `derived/injection_rows_complete.csv`: `748f51b738e42b86a6a907061a50152c80e97591d5c8c95a0f345f9a253e82a9`
- `derived/v4p1_ensemble_factor_summary_gengamma.csv`: `27c9186e736f4c6c98f94fbbca911e3255452e7d284087ab326597c797e9251a`
- `derived/v4p1_ensemble_factor_summary_sigpowexpq.csv`: `ae90000657f5e84d668b00e36ff6b30e902746d61664262fcd9730b3b9c0d9df`
- `derived/fig_v4p1_ls_observed_dataset_comparison_summary.csv`: `bccfffe00b9ed8f2118f652f3c2fa4af7c5a192c8681c461f394dd7c0bed660e`
- `derived/fig_v4p1_ls_observed_dataset_comparison_summary.json`: `f4c9508de1c137f0c5a07366988c8bbe023f13002bde480a9128f807b11b3cfc`
- `derived/fig_v4p1_factor20_native10_vs_1pct_x10_toy_medians_summary.csv`: `0d014872000e758144eac36314b1db4cfa87aa1b38400dd3ea89994b9a075eaa`
