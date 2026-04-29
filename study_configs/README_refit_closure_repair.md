# Refit Closure Repair Runbook

This runbook keeps the exploratory refit checks separate from the final
publication configuration.  Freeze the primary/secondary functional forms before
looking at closure outcomes.

## Refit Matrix

Generate configs for the requested matrix:

```bash
python study_configs/make_refit_closure_matrix_configs.py --n-toys 1000
python study_configs/make_refit_closure_matrix_configs.py --signal-model kernel --n-toys 1000
```

This writes configs under `study_configs/refit_closure_matrix/` for:

- `inj_train_exclude_nsigma`: `1.64`, `1.98`, `2.58`, `3.0`
- `kernel_ls_res_lower_factor`: `0.5`, `1.0`, `1.5`
- fixed `blind_nsigma: 1.64`

The helper script also writes:

```bash
bash study_configs/refit_closure_matrix/run_refit_closure_matrix.sh
```

Each refit summary now records the lower/upper length-scale factors, `ls_lo`,
`ls_hi`, `ls_opt`, training-bin counts, `Nsig_train`, `Nsig_win`,
`f_train_frac`, and refit fallback rates.

## True-Luminosity Functional-Form Toys

The functional-form ROOT macros can now scale the analytic expectation before
Poisson toy generation and write a frozen analytic seed histogram named
`<function>_analytic_seed_lumi_scaled` in each function directory.

```bash
root -l -b -q 'root_macros/funcform/make_func_data_output_2016.C("outputs/funcform_toys/funcform_2016_true_lumi_x10_toys.root",100,10.0,true)'
root -l -b -q 'root_macros/funcform/make_func_data_output_2021.C("outputs/funcform_toys/funcform_2021_true_lumi_x10_toys.root",100,10.0,true)'
root -l -b -q 'root_macros/funcform/make_func_data_output_2021.C("outputs/funcform_toys/funcform_2021_true_lumi_x100_toys.root",100,100.0,true)'
```

## Secondary-Seed Contingency

Use the analytic seed, not the 100-toy ensemble mean, as the GPR pseudo-data
background seed:

```bash
python -m hps_gpr.cli gp-toy-scan \
  --config study_configs/config_2016_10pct_blind1p64_95CL_1k_gp_toyscan_fixedtotal.yaml \
  --dataset 2016 \
  --seed-root outputs/funcform_toys/funcform_2016_true_lumi_x10_toys.root \
  --seed-container fGenGammaThresh \
  --seed-hist fGenGammaThresh_analytic_seed_lumi_scaled \
  --seed-label secondary_fGenGammaThresh_x10 \
  --n-toys 100
```

Use the corresponding 2021 config/root file and secondary function tag for the
2021 `10x` and `100x` checks.  Treat disagreement with the primary
functional-form validation as a model-dependence warning, not as permission to
retune the final GP setup after inspecting closure.

## Optional Signal-Kernel Diagnostic

The nominal HPS resonance model remains the fixed detector-resolution Gaussian
template.  A Frate-style localized signal-kernel template is available as an
opt-in diagnostic:

```yaml
signal_model: kernel
signal_kernel_width_factor: 1.0
signal_kernel_length_scale_factor: 1.0
```

Run the same 2015 closure workflow once with `signal_model: default` and once
with `signal_model: kernel`, keeping every GP/refit setting otherwise fixed.
Compare `Ahat/Ainj`, pull mean/width, coverage, `Delta Z`, and spurious-signal
behavior before using the kernel mode in any publication-facing result.
