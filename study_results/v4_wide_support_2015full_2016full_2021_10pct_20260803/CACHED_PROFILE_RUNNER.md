# Cached fixed-profile runner

The v4 expected-limit ensemble uses 300 conditional background-only
pseudoexperiments at every 1 MeV mass point from 19 to 250 MeV.  The combined
limit in each pseudoexperiment is still the 90% CL asymptotic
`\(\widetilde q_\mu\)` CLs limit from the reviewed fixed GP state.

The campaign-local cache removes deterministic work that the ordinary
bisection repeats: unconstrained, bounded, and null profiles that do not depend
on the tested signal strength, plus identical background-only Asimov profiles
at repeated bisection nodes.  It does not replace the likelihood, minimizer,
CLs mapping, or bisection.  The implementation is deliberately confined to
this directory.

Before production, make a closure report from the final v4 configuration and
the final reviewed-state CSV:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 benchmark_cached_profile_closure.py \
  --config ../../study_configs/v4_wide_support_2015full_2016full_2021_10pct_20260803/config_obsUL90_combined_wide_support_v4_observed_only.yaml \
  --reviewed-state-csv /absolute/path/to/v4_reviewed_states.csv \
  --json-out cached_profile_closure_v4.json
```

The benchmark defaults to five masses that exercise the 2015-only,
2015+2016, three-dataset, 2016+2021, and 2021-only active sets.  It exits
nonzero unless every cached limit is bitwise identical to the uncached
reference and the passing rows cover one-, two-, and three-dataset cases.

Production is gated on that matching report:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 run_combined_bands_cached_fixed_reviewed.py \
  --reviewed-state-csv /absolute/path/to/v4_reviewed_states.csv \
  --closure-report cached_profile_closure_v4.json \
  --workers 8 \
  --confirm-production
```

`--shard-count N --shard-index I` gives restartable mass shards.  Each shard
still uses the full-grid child `SeedSequence(24680).spawn(232)[mass_MeV-19]`;
changing the number of shards or workers therefore does not change a mass
point's pseudoexperiment stream.  Production is fixed at 300 toys per mass.
The output table includes the raw lower-tail, upper-tail, equality, and
two-sided-minimum counts next to `p_strong`, `p_weak`, and `p_two`.  A zero
empirical tail therefore remains visibly a count of zero at resolution
`1/300`, rather than an unqualified probability statement.

Once the full reviewed table exists, the campaign-local postprocessor checks
the 232-point stitched grid, all five active-dataset blocks, all 415 reviewed
GP states, 300 finite toys at every mass, raw-count tail identities, ordered
quantiles, and the fixed-state/no-refit contract:

```bash
python3 postprocess_combined_bands300.py
```

It writes the reviewed table, GP-state closure ledger, separate analytic
resolution-spacing Šidák reference, matched wide-versus-narrow comparison,
machine-readable summaries, and five PDF/PNG note figures under `derived/`
and `note_figures/`.  The dimuon correction is applied equally to the observed
limit and all toy-limit quantiles.  It therefore changes neither the empirical
tail fractions nor observed-to-median and wide-to-narrow ratios.  The plotted
bands remain descriptive fixed-GP toy-limit quantiles, not a coverage
calibration, and a zero raw tail count remains `0/300`, not an exact
probability of zero.
