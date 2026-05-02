# Final 2015 Fixed-Histogram Closure Rescue Configs

These configs test the direct fixed-histogram closure question while isolating
signal leakage into the GP refit training sidebands:

1. load each 2015 functional-form toy histogram,
2. fit the pre-injection GP to that histogram,
3. add the nominal full signal shape to the same fixed histogram,
4. optionally refit the GP with a predeclared training guard band,
5. extract the signal yield in the blind/search window.

The generated matrix includes:

- current baselines with `train_exclude_nsigma = blind_nsigma`,
- primary guard-band candidates with `train_exclude_nsigma = 2.58` and `3.0`,
- robustness guard-band candidates with `kernel_ls_res_lower_factor = 1.5`,
- fixed-hist no-refit and window-signal controls at the `3.0 sigma` guard.
- refit-matched `sigmaA_ref` guard scans at `train_exclude_nsigma = 1.96, 2.25, 2.50, 2.58, 2.75, 3.0`
  with `kernel_ls_res_lower_factor = 1.0`,
- kernel-lock diagnostics at `blind1p64_train3p0_lslb1p0` using
  `initial_fit` and cross-fit ensemble lock tables.

The nominal signal model remains `signal_model: default`; kernel locking and
signal-tail alpha inflation are diagnostic knobs in the code, not enabled by
these primary configs.

The refit-matched studies set `inj_sigma_a_ref_mode: matched_refit_bonly`.
That means the injected amplitude is scaled from a signal-free refit using the
same guard, kernel lock, and refit settings used by the injected toys. This is
the diagnostic intended to resolve the Wave 1 convention mismatch where
`Ainj` was scaled by the pre-refit covariance while `Zhat` used the post-refit
covariance.

Default SDF layout:

- run outputs: `/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/<run_tag>/`
- compiled plots: `/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/compiled_plots/<run_tag>/`
- SLURM logs: `/sdf/data/hps/users/epeets/scratch/2015_gpr_logs/<run_tag>/`

Generate the YAMLs:

```bash
python3 study_configs/final_configs_2015/make_final_2015_fixedhist_configs.py
```

Set `GPR_FUNCFORM_OUTPUT_BASE` before generation only if you intentionally want
a different output base.

Generate SLURM scripts:

```bash
bash study_configs/final_configs_2015/generate_slurm_all.sh
```

Useful SDF overrides:

```bash
CONFIG_GLOB='config_2015_blind1p64_train3p0_lslb1p0_primary_95CL_funcform100_fixedhist.yaml' \
TOY_ROOT=/sdf/home/e/epeets/src/hps-gpr-main/outputs/funcform_toys/funcform_2015_dataset_mod_toys_2.root \
CONTAINER=fShiftSigPowTail \
TOY_PATTERN='fShiftSigPowTail_toy_*' \
LOG_ROOT=/sdf/data/hps/users/epeets/scratch/2015_gpr_logs \
bash study_configs/final_configs_2015/generate_slurm_all.sh
```

`CONFIG_GLOB` limits the study family. `TOY_PATTERN` can be narrowed to one
toy, for example `fShiftSigPowTail_toy_0`, for a smoke test. Each matched toy
histogram becomes one SLURM job.

Submit all generated studies:

```bash
SLURM_GLOB='slurm_blind1p64_train3p0_lslb1p0_primary' \
bash study_configs/final_configs_2015/submit_all.sh
```

The submit helper defaults to the SDF style that has worked for this workflow:
`--account hps:hps-prod --partition roma`. Override with `SBATCH_ACCOUNT`,
`SBATCH_PARTITION`, `SBATCH_QOS`, or by passing explicit `sbatch` arguments.

Compile plots after jobs finish:

```bash
CONFIG_GLOB='config_2015_blind1p64_train3p0_lslb1p0_primary_95CL_funcform100_fixedhist.yaml' \
COMPILED_DIR=/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/compiled_plots \
bash study_configs/final_configs_2015/compile_all.sh
```

Recommended four-wave launch:

```bash
# Wave 1: fast guard reference check
bash study_configs/final_configs_2015/run_wave.sh wave1 generate
bash study_configs/final_configs_2015/run_wave.sh wave1 submit --account hps:hps-prod --partition roma

# Compile Wave 1
bash study_configs/final_configs_2015/run_wave.sh wave1 compile
```

Use the same pattern for:

- Wave 2: `train2p50`, `train2p75`, `prefit_reference_control`, `no_refit_control`.
- Wave 3: `none_refmatched`, `initial_fit_lock_refmatched`, `ensemble_p50_lock_refmatched`, `ensemble_p75ls_lock_refmatched`.
- Wave 4: `ensemble_p25ls_lock_refmatched`, `window_only_control`, and reruns.

Build cross-fit ensemble lock tables before the ensemble-lock wave. Use a
B-only or zero-injection toy CSV that contains `dataset`, `mass_GeV`,
`toy_index` or `toy`, and `initial_const_opt`/`initial_ls_opt`:

```bash
python3 study_configs/final_configs_2015/build_kernel_lock_tables.py \
  /sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/blind1p64_train3p0_lslb1p0_none_refmatched/injection_extraction/inj_extract_toys_2015.csv \
  --outdir /sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/kernel_lock_tables \
  --n-folds 5
```
