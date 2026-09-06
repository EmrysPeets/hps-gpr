#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SCRIPTS=(
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb0p5_sigkl1p0/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb0p5_sigkl1p5/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb0p5_sigkl2p0/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb1p0_sigkl1p0/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb1p0_sigkl1p5/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb1p0_sigkl2p0/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb1p5_sigkl1p0/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb1p5_sigkl1p5/submit_funcform_injection_all.sh
  study_configs/funcform_refit_lockin_signal_kernel/slurm_lslb1p5_sigkl2p0/submit_funcform_injection_all.sh
)

for script in "${SCRIPTS[@]}"; do
  if [ ! -x "${script}" ]; then
    echo "Missing ${script}; run study_configs/funcform_refit_lockin_signal_kernel/generate_slurm_all.sh first." >&2
    exit 1
  fi
  echo "[submit] ${script}"
  bash "${script}" --account=hps:hps-prod --partition=roma
done
