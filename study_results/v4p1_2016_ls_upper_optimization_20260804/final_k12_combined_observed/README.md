# Factor-12 combined observed reconstruction

This directory contains the observed-only reconstruction runner for the
reviewed v4.1 card. The card changes the 2016 resolution-scaled GP
length-scale upper factor from 8 to 12 while retaining the v4 search ranges,
fit supports, likelihood, data inputs, 90% asymptotic CLs construction, and
2015/2021 kernel settings.

The runner requires the stitched 415-state review:

```text
study_results/v4p1_2016_ls_upper_optimization_20260804/derived/observed_gp_states_k12_reviewed.csv
```

Validate the final card and state ledger without reconstructing any fit:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 run_fixed_reviewed_observed.py \
  --config ../../../study_configs/v4p1_2016_ls_upper_optimization_20260804/config_obsUL90_combined_wide_support_v4p1_2016k12_observed_only.yaml \
  --reviewed-state-csv ../derived/observed_gp_states_k12_reviewed.csv \
  --validate-only
```

Run all 232 masses and require bitwise cached-vs-reference observed-limit
closure at representative one-, two-, and three-dataset masses. The reviewed
production in this directory used one worker:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 run_fixed_reviewed_observed.py \
  --config ../../../study_configs/v4p1_2016_ls_upper_optimization_20260804/config_obsUL90_combined_wide_support_v4p1_2016k12_observed_only.yaml \
  --reviewed-state-csv ../derived/observed_gp_states_k12_reviewed.csv \
  --workers 1 \
  --reference-closure
```

The default outputs are `combined_observed_fixed_reviewed.csv` and
`combined_observed_fixed_reviewed_provenance.json`. The CSV contains observed
90% CLs limits and local asymptotic p0/Z only. The runner draws zero toys and
has no expected-band output path.

The completed production contains 232 finite limits and 232 finite local
asymptotic p0/Z values. Cached and reference limits are bitwise identical at
20, 40, and 60 MeV. The output CSV SHA-256 is
`fa95a50a8b8ddc1d69a319137038a177c6d6da3afbbf9163d8955cf197182de2`.
