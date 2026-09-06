#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

CONFIG_DIR="${CONFIG_DIR:-study_configs/final_configs_2015}"
SLURM_GLOB="${SLURM_GLOB:-slurm_*}"

submit_scripts=(${CONFIG_DIR}/${SLURM_GLOB}/submit_funcform_injection_all.sh)
if (( ${#submit_scripts[@]} == 0 )); then
  echo "No submit helpers matched ${CONFIG_DIR}/${SLURM_GLOB}/submit_funcform_injection_all.sh." >&2
  echo "Run ${CONFIG_DIR}/generate_slurm_all.sh first or set SLURM_GLOB." >&2
  exit 1
fi

for script in "${submit_scripts[@]}"; do
  echo "[submit] ${script}"
  bash "${script}" "$@"
done
