#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="study_configs/final_configs_2015"
SUBMIT_SCRIPTS=(
  "${CONFIG_DIR}/slurm_blind1p64_95CL_funcform100_fixedhist_refit_lslb0p5/submit_funcform_injection_all.sh"
  "${CONFIG_DIR}/slurm_blind1p64_95CL_funcform100_fixedhist_refit_lslb1p0/submit_funcform_injection_all.sh"
  "${CONFIG_DIR}/slurm_blind1p96_95CL_funcform100_fixedhist_refit_lslb0p5/submit_funcform_injection_all.sh"
  "${CONFIG_DIR}/slurm_blind1p96_95CL_funcform100_fixedhist_refit_lslb1p0/submit_funcform_injection_all.sh"
)

for script in "${SUBMIT_SCRIPTS[@]}"; do
  echo "[submit] ${script}"
  bash "${script}" "$@"
done
