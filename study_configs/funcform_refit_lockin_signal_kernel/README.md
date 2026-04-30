# 2015 Signal-Kernel Lock-In Comparator

This matrix is a diagnostic for the opt-in `signal_model: kernel` template.
The nominal HPS signal hypothesis remains the detector-resolution Gaussian.

The matrix crosses:

- Background GP lower length-scale bound: `0.5`, `1.0`, `1.5` times the mass resolution.
- Signal-kernel correlation length: `1.0`, `1.5`, `2.0` times the mass resolution.

For each signal-kernel correlation length, the signal-kernel localization width is
calibrated so the leading eigen-template has the same effective sigma and nearly
the same `±1.64σ` and `±1.96σ` containment as the nominal Gaussian HPS signal.
The calibration is recorded in `signal_kernel_physical_equivalence.csv`.

Run on SDF:

```bash
python3 study_configs/make_2015_funcform_refit_lockin_signal_kernel_configs.py
bash study_configs/funcform_refit_lockin_signal_kernel/run_smoke.sh
bash study_configs/funcform_refit_lockin_signal_kernel/generate_slurm_all.sh
bash study_configs/funcform_refit_lockin_signal_kernel/submit_all.sh
```

After all jobs finish:

```bash
bash study_configs/funcform_refit_lockin_signal_kernel/compile_all.sh
```

Comparison plots are written to:

```text
outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_signal_kernel_comparison
```
