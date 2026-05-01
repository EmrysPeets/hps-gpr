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

The nominal signal model remains `signal_model: default`; kernel locking and
signal-tail alpha inflation are diagnostic knobs in the code, not enabled by
these primary configs.

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

Compile plots after jobs finish:

```bash
CONFIG_GLOB='config_2015_blind1p64_train3p0_lslb1p0_primary_95CL_funcform100_fixedhist.yaml' \
COMPILED_DIR=/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/compiled_plots \
bash study_configs/final_configs_2015/compile_all.sh
```
