# Final 2015 Fixed-Histogram Closure Configs

These configs test the direct closure question:

1. load each 2015 functional-form toy histogram,
2. fit the GP to that histogram with the configured blind window,
3. add signal on top of that same histogram,
4. refit the GP to the signal-injected histogram,
5. extract the signal yield in the blind window.

The matrix varies only:

- blind width: `1.64` or `1.96`,
- regular GPR lower length-scale bound: `0.5` or `1.0` times the local mass resolution.

The training exclusion width is not a separate study axis here. For each config,
`gp_train_exclude_nsigma`, `inj_train_exclude_nsigma`, and
`ul_bands_train_exclude_nsigma` are set equal to `blind_nsigma`.

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
