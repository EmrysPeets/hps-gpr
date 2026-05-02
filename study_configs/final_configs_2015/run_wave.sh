#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${CONFIG_DIR:-study_configs/final_configs_2015}"
COMPILED_BASE="${COMPILED_BASE:-/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/compiled_plots}"

usage() {
  cat <<'EOF'
Usage:
  bash study_configs/final_configs_2015/run_wave.sh <wave1|wave2|wave3|wave4|all> <generate|submit|compile|generate-submit> [sbatch args...]

Examples:
  TOY_ROOT=/path/to/funcform.root CONTAINER=fShiftSigPowTail TOY_PATTERN='fShiftSigPowTail_toy_*' \
    bash study_configs/final_configs_2015/run_wave.sh wave1 generate

  bash study_configs/final_configs_2015/run_wave.sh wave1 submit --account hps:hps-prod --partition roma

  bash study_configs/final_configs_2015/run_wave.sh wave1 compile
EOF
}

wave="${1:-}"
action="${2:-}"
if [[ -z "${wave}" || -z "${action}" ]]; then
  usage >&2
  exit 2
fi
shift 2

tags_for_wave() {
  case "$1" in
    wave1)
      printf '%s\n' \
        blind1p64_train1p96_lslb1p0_guard_refmatched \
        blind1p64_train2p25_lslb1p0_guard_refmatched \
        blind1p64_train2p58_lslb1p0_guard_refmatched \
        blind1p64_train3p0_lslb1p0_guard_refmatched
      ;;
    wave2)
      printf '%s\n' \
        blind1p64_train2p50_lslb1p0_guard_refmatched \
        blind1p64_train2p75_lslb1p0_guard_refmatched \
        blind1p64_train3p0_lslb1p0_prefit_reference_control \
        blind1p64_train3p0_lslb1p0_no_refit_control
      ;;
    wave3)
      printf '%s\n' \
        blind1p64_train3p0_lslb1p0_none_refmatched \
        blind1p64_train3p0_lslb1p0_initial_fit_lock_refmatched \
        blind1p64_train3p0_lslb1p0_ensemble_p50_lock_refmatched \
        blind1p64_train3p0_lslb1p0_ensemble_p75ls_lock_refmatched
      ;;
    wave4)
      printf '%s\n' \
        blind1p64_train3p0_lslb1p0_ensemble_p25ls_lock_refmatched \
        blind1p64_train3p0_lslb1p0_window_only_control
      ;;
    all)
      tags_for_wave wave1
      tags_for_wave wave2
      tags_for_wave wave3
      tags_for_wave wave4
      ;;
    *)
      echo "Unknown wave: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

run_generate() {
  local tag="$1"
  echo "[wave:${wave}] generate ${tag}"
  CONFIG_GLOB="config_2015_${tag}_95CL_funcform100_fixedhist.yaml" \
    bash "${CONFIG_DIR}/generate_slurm_all.sh"
}

run_submit() {
  local tag="$1"
  echo "[wave:${wave}] submit ${tag}"
  SLURM_GLOB="slurm_${tag}" \
    bash "${CONFIG_DIR}/submit_all.sh" "$@"
}

run_compile() {
  local tag="$1"
  echo "[wave:${wave}] compile ${tag}"
  CONFIG_GLOB="config_2015_${tag}_95CL_funcform100_fixedhist.yaml" \
  COMPILED_DIR="${COMPILED_BASE}/${wave}" \
    bash "${CONFIG_DIR}/compile_all.sh"
}

mapfile -t tags < <(tags_for_wave "${wave}")
for tag in "${tags[@]}"; do
  case "${action}" in
    generate)
      run_generate "${tag}"
      ;;
    submit)
      run_submit "${tag}" "$@"
      ;;
    compile)
      run_compile "${tag}"
      ;;
    generate-submit)
      run_generate "${tag}"
      run_submit "${tag}" "$@"
      ;;
    *)
      echo "Unknown action: ${action}" >&2
      usage >&2
      exit 2
      ;;
  esac
done
