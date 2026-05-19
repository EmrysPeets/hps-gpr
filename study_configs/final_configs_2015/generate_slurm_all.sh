#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="study_configs/final_configs_2015"
MASSES="0.030,0.045,0.060,0.075,0.090,0.105,0.120"
STRENGTHS="s0,s1,s2,s3,s5"

CONFIGS=(
  "config_2015_blind1p64_95CL_funcform100_fixedhist_refit_lslb0p5.yaml"
  "config_2015_blind1p64_95CL_funcform100_fixedhist_refit_lslb1p0.yaml"
  "config_2015_blind1p96_95CL_funcform100_fixedhist_refit_lslb0p5.yaml"
  "config_2015_blind1p96_95CL_funcform100_fixedhist_refit_lslb1p0.yaml"
)

for config_name in "${CONFIGS[@]}"; do
  tag="${config_name#config_2015_}"
  tag="${tag%.yaml}"
  slurm_dir="${CONFIG_DIR}/slurm_${tag}"
  mkdir -p "${slurm_dir}"

  hps-gpr slurm-gen-funcform-inject \
    --config "${CONFIG_DIR}/${config_name}" \
    --dataset 2015 \
    --masses "${MASSES}" \
    --strengths "${STRENGTHS}" \
    --n-injection-toys 1 \
    --write-qmu \
    --cpus-per-task 1 \
    --job-name "hps2015_${tag}" \
    --partition roma \
    --account hps:hps-prod \
    --time 1:00:00 \
    --memory 8G \
    --output "${slurm_dir}/submit_funcform_injection_${tag}.slurm"
done
