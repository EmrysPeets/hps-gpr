# Final 2015 Fixed-Histogram Closure Configs

These configs test the direct closure question:

1. load each 2015 functional-form toy histogram,
2. fit the GP to that histogram with the configured blind window,
3. add signal on top of that same histogram,
4. refit the GP to the signal-injected histogram,
5. extract the signal yield in the blind window.

The original matrix varies only:

- blind width: `1.64` or `1.96`,
- regular GPR lower length-scale bound: `0.5` or `1.0` times the local mass resolution.

The training exclusion width is not a separate study axis here. For each config,
`gp_train_exclude_nsigma`, `inj_train_exclude_nsigma`, and
`ul_bands_train_exclude_nsigma` are set equal to `blind_nsigma`.

The additional single-purpose config
`config_2015_blind2p25_95CL_funcform100_fixedhist_refit_lslb1p0.yaml` is for the
wide-blind comparison against the guard-band study. It sets
`blind_nsigma = gp_train_exclude_nsigma = inj_train_exclude_nsigma =
ul_bands_train_exclude_nsigma = 2.25`, so the GP training edge is the same as the
`1.64`-blind, `2.25`-guard study while the extraction/blind window itself is widened
to `2.25 sigma`.

Generate the YAMLs:

```bash
python3 study_configs/final_configs_2015/make_final_2015_fixedhist_configs.py
```

Generate SLURM scripts:

```bash
bash study_configs/final_configs_2015/generate_slurm_all.sh
```

Submit all four studies:

```bash
bash study_configs/final_configs_2015/submit_all.sh
```

Compile plots after jobs finish:

```bash
bash study_configs/final_configs_2015/compile_all.sh
```

## 2.25-sigma blind-window comparison on SDF

Use this when you want the direct wide-blind comparison to the `1.64`-blind,
`2.25`-guard functional-form closure study.

From a clean SDF checkout:

```bash
git pull origin main

# Confirm the functional-form ROOT file exists. If it does not, regenerate it with
# root -l -b -q 'root_macros/funcform/make_func_data_output_2015.C()'
ls outputs/funcform_toys/funcform_2015_dataset_mod_toys_2.root
```

Optional one-toy smoke test:

```bash
hps-gpr funcform-inject \
  --config study_configs/final_configs_2015/config_2015_blind2p25_95CL_funcform100_fixedhist_refit_lslb1p0.yaml \
  --dataset 2015 \
  --toy-name-fmt 'fShiftSigPowTail_toy_{i}' \
  --toy-index 0 \
  --masses 0.075 \
  --strengths s0,s2,s5 \
  --n-injection-toys 1 \
  --output-dir outputs/final_2015_funcform_fixedhist_blind2pt25_lslb_1pt0_smoke \
  --write-qmu
```

Generate the full SDF SLURM submission helpers:

```bash
mkdir -p study_configs/final_configs_2015/slurm_blind2p25_95CL_funcform100_fixedhist_refit_lslb1p0

hps-gpr slurm-gen-funcform-inject \
  --config study_configs/final_configs_2015/config_2015_blind2p25_95CL_funcform100_fixedhist_refit_lslb1p0.yaml \
  --dataset 2015 \
  --masses 0.030,0.045,0.060,0.075,0.090,0.105,0.120 \
  --strengths s0,s1,s2,s3,s5 \
  --n-injection-toys 1 \
  --write-qmu \
  --cpus-per-task 1 \
  --job-name hps2015_blind2p25_fixedhist \
  --partition roma \
  --account hps:hps-prod \
  --time 1:00:00 \
  --memory 8G \
  --output study_configs/final_configs_2015/slurm_blind2p25_95CL_funcform100_fixedhist_refit_lslb1p0/submit_funcform_injection_blind2p25_lslb1p0.slurm
```

Submit the generated jobs:

```bash
bash study_configs/final_configs_2015/slurm_blind2p25_95CL_funcform100_fixedhist_refit_lslb1p0/submit_funcform_injection_all.sh
```

After the jobs finish, merge/plot the outputs:

```bash
hps-gpr inject-plot \
  -i outputs/final_2015_funcform_fixedhist_blind2pt25_lslb_1pt0 \
  -o outputs/final_2015_funcform_fixedhist_blind2pt25_lslb_1pt0/injection_summary \
  --dataset 2015 \
  --write-merged-toys
```

Compare the resulting pull, recovery, and `Delta Z` summaries against the
`1.64`-blind, `2.25`-guard study using the same injected masses and strengths. The
main interpretive difference is that this test widens the fitted signal window to
`2.25 sigma`, whereas the guard-band study keeps the extraction window at
`1.64 sigma` and excludes the `1.64--2.25 sigma` tails only from GP training.
