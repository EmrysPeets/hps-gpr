# v4.1 2016 length-scale upper-bound study

## Outcome

This study changes one physics-card value relative to the August 3 v4
observed-only configuration:

```text
kernel_ls_res_upper_factor_by_dataset.2016: 8 -> 12
```

The output directory is the only other parsed YAML difference. The data inputs,
search intervals, wider fit supports, 2.25-sigma blind/training exclusion,
rebin-five geometry, lower length-scale factors, optimizer restart count,
shared-`epsilon^2` `count_scale` likelihood, and 90% asymptotic CLs construction
are unchanged.

Factor 12 is the first tested 2016 upper factor with no occupied upper-bound
states and is followed by a stable factor-15/factor-20 plateau. It is therefore
the v4.1 observed/asymptotic candidate. Because the range study was prompted by
the v4 observed scan, it is not described as a pre-unblinding freeze or a
coverage-qualified nominal.

No expected-limit bands were produced. No new pseudoexperiments were drawn.

## Controlled 2016 scan

The scan covers the exact 39--180 MeV grid at 1 MeV spacing. A state is counted
as bound-occupied when `ls_opt / ls_hi >= 0.999`.

| Upper factor | Occupied / 142 |
| ---: | ---: |
| 8 | 142 |
| 10 | 56 |
| 12 | 0 |
| 15 | 0 |
| 20 | 0 |

Questionable optimizer branches were rerun with the data and card unchanged.
Fourteen selected repair rows use the highest log marginal likelihood among
actual fits, every selected repair was independently reproduced, and no state
was interpolated. The nested log-marginal-likelihood audit has zero violations
at tolerance `1e-4`.

From factor 12 to 15:

- the largest pointwise 2016 observed yield-limit change is 0.083%;
- the largest absolute local-Z change is 0.00255;
- the log-marginal-likelihood differences span approximately
  `[-4.5e-6, 8.0e-6]`.

The factor-15 to factor-20 comparison is similarly stable. The factor choice
uses boundary release and the next-setting plateau, not the direction of an
observed limit or p-value.

## Exact combined observed result

The final reviewed ledger has 415 exact GP states: 72 for 2015, 142 for 2016,
and 201 for 2021. The fixed-state runner reconstructs the GP prediction and
shared-coupling likelihood at all 232 combined mass hypotheses.

- finite observed 90% asymptotic CLs limits: 232 / 232;
- finite local asymptotic p0 and Z values: 232 / 232;
- cached/reference observed-limit closure: bitwise identical at 20, 40, and
  60 MeV, covering one-, two-, and three-dataset likelihoods;
- toy draws: 0;
- expected-limit bands: none.

Only the 142 masses with active 2016 input change. On those masses, the
v4.1/v4 observed-limit ratio has median 0.9933, minimum 0.6941 at 103 MeV
(30.59% tighter), and maximum 1.2789 at 90 MeV (27.89% weaker). There are 76
tighter and 66 weaker matched masses. This is not a uniform sensitivity gain.

The v4 local minimum was `p0 = 1.76364e-4` at 66 MeV
(`Z_local = 3.573`). The v4.1 candidate minimum is
`p0 = 3.25918e-5` at 65 MeV (`Z_local = 3.993`). With the unchanged
resolution-spacing estimate `N_eff = 35.381`, the fixed-card analytic Sidak
reference at the new minimum is `p = 0.0011525` (`Z = 3.048`).

That Sidak value is not a toy-calibrated global p-value: it corrects only the
mass scan conditional on the factor-12 card and does not include the
post-v4 upper-bound study. The p-value change is therefore a diagnostic until
production-matched closure, direct coverage, and a separate scan-wise
maximum-q0 ensemble are complete.

## Principal products

- `derived/run_summary.json`: controlled factor-grid machine summary.
- `derived/selection_decision.json`: factor-12 decision rule, plateau evidence,
  and interpretation boundary.
- `derived/repair_candidate_ledger.csv`: raw and repeated optimizer branches.
- `derived/observed_gp_states_k12_reviewed.csv`: final 415-state ledger.
- `final_k12_combined_observed/combined_observed_fixed_reviewed.csv`: exact
  232-mass observed limit and local asymptotic p0/Z table.
- `final_k12_combined_observed/combined_observed_fixed_reviewed_provenance.json`:
  final card, source hashes, and reference-solver closure.
- `derived/combined_observed_k12_vs_v4.csv`: matched v4.1/v4 comparison.
- `derived/combined_observed_k12_summary.json`: note-ready numerical summary.
- `plots/lml_and_length_scale_boundary_occupancy.pdf`: factor-selection
  diagnostic.
- `plots/combined_observed_limit_k12_vs_v4_no_bands.pdf`: combined observed
  limit comparison.
- `plots/combined_asymptotic_p0_k12_vs_v4.pdf`: local asymptotic p0 comparison
  and conditional analytic Sidak reference.

The updated analysis-note source is in the sibling worktree
`/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v4-20260803`.
The rendered v4.1 PDF is exported under
`output/pdf/hps_gpr_analysis_note_v4p1_20260804/` in this worktree.

## Reproduction

Generate and audit the final card:

```bash
python3 study_configs/v4p1_2016_ls_upper_optimization_20260804/make_final_combined_config.py
```

Rebuild the repair-aware factor comparison:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg \
  python3 study_results/v4p1_2016_ls_upper_optimization_20260804/postprocess_2016_ls_upper_grid.py
```

Reconstruct the 232-mass observed-only result:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 study_results/v4p1_2016_ls_upper_optimization_20260804/final_k12_combined_observed/run_fixed_reviewed_observed.py \
  --reviewed-state-csv study_results/v4p1_2016_ls_upper_optimization_20260804/derived/observed_gp_states_k12_reviewed.csv \
  --workers 1 \
  --reference-closure
```

Rebuild the combined comparisons:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg \
  python3 study_results/v4p1_2016_ls_upper_optimization_20260804/postprocess_combined_observed.py
```

## Planned 2021 10% to 100% hyperparameter toys

The next study should scale a validated smooth 2021 expectation, not multiply
the observed bin contents and not rescale an existing limit. For exposure
factors `s = 1, 2, 5, 10`, define

```text
lambda_i(s) = s * lambda_i(10%)
Y_i(toy, s) ~ Poisson(lambda_i(s))
```

so `s = 10` represents 100%-equivalent statistics. Use paired toy identifiers
across exposure points and upper-bound candidates. Nested independent Poisson
increments between successive exposure points are an optional variance-reduction
implementation with the correct Poisson marginal at each exposure.

Each pseudo-spectrum must retrain the GP with:

- 40--300 MeV 2021 fit support;
- 50--250 MeV search interval;
- rebin five;
- the 2.25-sigma training exclusion;
- the same resolution model and lower bound as the production candidate;
- a deliberately loose upper-factor grid;
- unchanged-toy repeats for questionable optimizer branches.

Run both GP self-closure and independent smooth functional-form truth families.
For every mass, exposure, toy, and upper-factor candidate, retain:

- `ls_opt / sigma_x` and `ls_opt / ls_hi`;
- upper-bound occupancy;
- log marginal likelihood;
- optimizer-repeat stability;
- background bias and pull;
- injected-signal recovery.

Summarize medians and central 68%/95% intervals versus exposure. Those
distributions can determine whether the admissible length-scale range grows
between 10% and 100% statistics. They are hyperparameter-closure products, not
expected-limit bands.

The current convenience `gp-toy-scan` path is not yet production matched for
this purpose: `hps_gpr/gp_toys.py` resets `neighborhood_rebin` to 1 for each
toy and the CLI has no exposure-scale option. The functional-form ROOT seed
generators already expose `toy_lumi_scale`, but the downstream scan must first
preserve rebin five and record the exposure scale. Direct CLs coverage and a
scan-wise discovery ensemble should remain separate follow-up campaigns after
the hyperparameter card is frozen.
