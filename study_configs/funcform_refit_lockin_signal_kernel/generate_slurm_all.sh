#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MASSES="0.030,0.045,0.060,0.075,0.090,0.105,0.120"
STRENGTHS="s0,s1,s2,s3,s5"

generate_one() {
  local tag="$1"
  local config="$2"
  local job_name="$3"
  local slurm_dir="study_configs/funcform_refit_lockin_signal_kernel/slurm_${tag}"
  mkdir -p "${slurm_dir}"
  hps-gpr slurm-gen-funcform-inject \
    --config "${config}" \
    --dataset 2015 \
    --masses "${MASSES}" \
    --strengths "${STRENGTHS}" \
    --n-injection-toys 1 \
    --write-qmu \
    --cpus-per-task 1 \
    --job-name "${job_name}" \
    --partition roma \
    --account hps:hps-prod \
    --time 1:00:00 \
    --memory 8G \
    --output "${slurm_dir}/submit_funcform_injection_${tag}.slurm"
}

generate_one "lslb0p5_sigkl1p0" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb0p5_sigkl1p0.yaml" "hps2015_ffinj_lslb0p5_sigkl1p0_tr196"

generate_one "lslb0p5_sigkl1p5" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb0p5_sigkl1p5.yaml" "hps2015_ffinj_lslb0p5_sigkl1p5_tr196"

generate_one "lslb0p5_sigkl2p0" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb0p5_sigkl2p0.yaml" "hps2015_ffinj_lslb0p5_sigkl2p0_tr196"

generate_one "lslb1p0_sigkl1p0" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p0_sigkl1p0.yaml" "hps2015_ffinj_lslb1p0_sigkl1p0_tr196"

generate_one "lslb1p0_sigkl1p5" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p0_sigkl1p5.yaml" "hps2015_ffinj_lslb1p0_sigkl1p5_tr196"

generate_one "lslb1p0_sigkl2p0" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p0_sigkl2p0.yaml" "hps2015_ffinj_lslb1p0_sigkl2p0_tr196"

generate_one "lslb1p5_sigkl1p0" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p5_sigkl1p0.yaml" "hps2015_ffinj_lslb1p5_sigkl1p0_tr196"

generate_one "lslb1p5_sigkl1p5" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p5_sigkl1p5.yaml" "hps2015_ffinj_lslb1p5_sigkl1p5_tr196"

generate_one "lslb1p5_sigkl2p0" "study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p5_sigkl2p0.yaml" "hps2015_ffinj_lslb1p5_sigkl2p0_tr196"
