# V3 review-committee result figures

These figures use the July 8, 2026 final-pre-unblinding outputs from
`/Users/emryspeets/Desktop/gp_mods/combined_15_16_10pct_21_1pct/final_pre_unblind`.
The copied configuration fixes the lower length-scale factors to 1.0 (2015),
0.9 (2016), and 1.1 (2021), uses the count-scale simultaneous likelihood, and
sets both the individual and simultaneous expected-limit ensembles to 10,000
background-only toys per evaluated mass hypothesis.

Run

```bash
python3 provenance/make_v3_review_result_plots.py
```

from this directory to regenerate the committee-review figures and their
derived tables.  The raw source CSVs in `provenance/` are never overwritten.

## Documented display repairs

The individual observed-limit scan contains five isolated GP optimizer
failures.  Their optimized length scales are far from the upper boundary used
by adjacent successful fits, and the corresponding upper limits form isolated
spikes.  For the observed-limit curves only, each failed value is replaced by
linear interpolation in log(limit) between the immediately adjacent 1 MeV mass
hypotheses:

| sample | mass [MeV] | raw limit on epsilon squared | v3 review value |
|---|---:|---:|---:|
| 2015 | 24 | 4.75112651e-5 | 1.05210993e-5 |
| 2016 10% | 44 | 2.28065322e-4 | 3.26625665e-5 |
| 2016 10% | 98 | 1.12159194e-4 | 1.85724585e-5 |
| 2016 10% | 133 | 2.41796175e-4 | 5.15171839e-5 |
| 2016 10% | 171 | 1.97945883e-3 | 1.91896546e-4 |

The simultaneous 136 MeV observed limit is repaired by the same rule.  The raw
simultaneous scan gives 2.54859e-5 and the earlier reconciled display table
gives 1.24508e-5; the geometric mean of the 135 and 137 MeV display values is
6.93624897e-6.  That single documented value is used in the combined expected-
band figure, the individual-versus-simultaneous comparison, and the
full-campaign scaling projection.  The 2021-only 136 MeV panel uses the same
neighbor rule for consistency.

The corrected source tables also retain the earlier log-linear expected-band
spike repairs.  The simultaneous table contains replacements at 37, 115, and
136 MeV, while the 2021-only table contains replacements at 79, 173, 181, and
217 MeV.  These are presentation values and remain production-rerun targets.

`provenance/v3_display_repair_audit.csv` contains the full-precision raw,
left-neighbor, right-neighbor, and replacement values for every modified
column.  Because only 2015 is active at 24 MeV, its repaired value is also
propagated to the simultaneous review table; the individual-versus-simultaneous
ratio is therefore exactly one at that mass.  The clean physics figures deliberately omit failure markers, but the
audit table preserves the complete history.  These display repairs do not
replace the production reruns required for a publication-final result.

## P-value figures

`combined_local_p0_points` joins the evaluated fixed-mass asymptotic local
`p0_analytic` values and shows the corresponding Sidak-equivalent scan curve,
using the resolution-spacing estimate of the effective number of trials.
`individual_asymptotic_local_global_pvalues` gives the analogous result for
2015, 2016 10%, and 2021 1%.

P-values from a failed fit are not interpolated.  The 2015 24 MeV and 2016 44,
98, 133, and 171 MeV fits are omitted from the individual p0 curves, and the
lines connect the adjacent valid mass hypotheses.  In particular, the raw
2016 value p0=0.001076 at 171 MeV is not interpreted; after excluding the
failed rows, the smallest valid 2016 value is 0.0277097 at 41 MeV.  The 2021
p0 curve is read from the 2021 expected-band table rather than from the failed
single-scan rows.

`combined_limit_tail_pvalues_points` shows three consistency tests derived
from 10,000 background-only joint pseudoexperiments at each mass.  One toy
spectrum is generated for every active campaign and spectra with the same toy
index are assembled into each simultaneous pseudoexperiment:

- `p_strong`: fraction of toy upper limits less than or equal to the observed
  upper limit;
- `p_weak`: fraction of toy upper limits greater than or equal to the observed
  upper limit;
- `p_two`: `min(1, 2 min(p_strong, p_weak))`.

The stored tail fractions at 24, 37, 44, 59, 91, 98, 115, 133, 136, 159, and
166 MeV were evaluated against observed upper limits later superseded in the
display table.  They cannot be recomputed from the archived quantiles, so those
rows are omitted and adjacent valid hypotheses are connected.  No p-value is
interpolated.  The exclusion list is written to
`provenance/v3_tail_pvalue_exclusions.csv`.

The smallest values are 0.0031 at 45 MeV for `p_strong`, 0.0040 at 51 MeV for
`p_weak`, and 0.0062 at 45 MeV for `p_two`.  These tests describe the placement
of the observed upper limit in its background-only ensemble; they are not
discovery p-values.  Exact minima, effective-trials estimates, and
Sidak-equivalent values are recorded in `provenance/v3_pvalue_summary.csv`.

## Full-campaign scaling projection

`final_preunblind_current_vs_full_campaign_projection_eps2` applies the note's
density-scaling convention to the repaired combined table.  The 2015 sample is
held fixed, while the 2016 10% and 2021 1% density terms are multiplied by 10
and 100.  The resulting curves are continuous across changes in the set of
active datasets.  This is a planning projection based on the current observed
and expected-median curves; it is not a fit to the full 2016 or 2021 datasets
and is not a publication-level expected sensitivity.
