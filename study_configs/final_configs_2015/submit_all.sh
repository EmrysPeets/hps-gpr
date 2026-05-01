#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

CONFIG_DIR="${CONFIG_DIR:-study_configs/final_configs_2015}"
SLURM_GLOB="${SLURM_GLOB:-slurm_*}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-hps:hps-prod}"
SBATCH_PARTITION="${SBATCH_PARTITION:-roma}"
SBATCH_QOS="${SBATCH_QOS:-}"

has_sbatch_opt() {
  local wanted="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${wanted}" || "${arg}" == "${wanted}="* ]]; then
      return 0
    fi
  done
  return 1
}

submit_scripts=(${CONFIG_DIR}/${SLURM_GLOB}/submit_funcform_injection_all.sh)
if (( ${#submit_scripts[@]} == 0 )); then
  echo "No submit helpers matched ${CONFIG_DIR}/${SLURM_GLOB}/submit_funcform_injection_all.sh." >&2
  echo "Run ${CONFIG_DIR}/generate_slurm_all.sh first or set SLURM_GLOB." >&2
  exit 1
fi

sbatch_args=()
if ! has_sbatch_opt "--account" "$@" && ! has_sbatch_opt "-A" "$@"; then
  sbatch_args+=(--account "${SBATCH_ACCOUNT}")
fi
if ! has_sbatch_opt "--partition" "$@" && ! has_sbatch_opt "-p" "$@"; then
  sbatch_args+=(--partition "${SBATCH_PARTITION}")
fi
if [[ -n "${SBATCH_QOS}" ]] && ! has_sbatch_opt "--qos" "$@"; then
  sbatch_args+=(--qos "${SBATCH_QOS}")
fi
sbatch_args+=("$@")

for script in "${submit_scripts[@]}"; do
  echo "[submit] ${script} ${sbatch_args[*]}"
  bash "${script}" "${sbatch_args[@]}"
done
