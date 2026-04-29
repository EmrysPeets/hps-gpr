#!/usr/bin/env bash
set -euo pipefail

mkdir -p study_configs/funcform_refit_lslb1p5/slurm_main
hps-gpr slurm-gen-funcform-inject \
  --config study_configs/config_2015_blind1p64_95CL_funcform100_refit_lslb1p5.yaml \
  --dataset 2015 \
  --masses 0.030,0.045,0.060,0.075,0.090,0.105,0.120 \
  --strengths s0,s1,s2,s3,s5 \
  --n-injection-toys 1 \
  --write-qmu \
  --cpus-per-task 1 \
  --job-name hps2015_ffinj_lslb1p5 \
  --partition roma \
  --account hps:hps-prod \
  --time 1:00:00 \
  --memory 8G \
  --output study_configs/funcform_refit_lslb1p5/slurm_main/submit_funcform_injection_main.slurm

mkdir -p study_configs/funcform_refit_lslb1p5/slurm_window
hps-gpr slurm-gen-funcform-inject \
  --config study_configs/config_2015_blind1p64_95CL_funcform100_refit_lslb1p5_window.yaml \
  --dataset 2015 \
  --masses 0.030,0.045,0.060,0.075,0.090,0.105,0.120 \
  --strengths s0,s1,s2,s3,s5 \
  --n-injection-toys 1 \
  --write-qmu \
  --cpus-per-task 1 \
  --job-name hps2015_ffinj_lslb1p5_win \
  --partition roma \
  --account hps:hps-prod \
  --time 1:00:00 \
  --memory 8G \
  --output study_configs/funcform_refit_lslb1p5/slurm_window/submit_funcform_injection_window.slurm

mkdir -p study_configs/funcform_refit_lslb1p5/slurm_train1p96
hps-gpr slurm-gen-funcform-inject \
  --config study_configs/config_2015_blind1p64_95CL_funcform100_refit_lslb1p5_train1p96.yaml \
  --dataset 2015 \
  --masses 0.030,0.045,0.060,0.075,0.090,0.105,0.120 \
  --strengths s0,s1,s2,s3,s5 \
  --n-injection-toys 1 \
  --write-qmu \
  --cpus-per-task 1 \
  --job-name hps2015_ffinj_lslb1p5_tr196 \
  --partition roma \
  --account hps:hps-prod \
  --time 1:00:00 \
  --memory 8G \
  --output study_configs/funcform_refit_lslb1p5/slurm_train1p96/submit_funcform_injection_train1p96.slurm
