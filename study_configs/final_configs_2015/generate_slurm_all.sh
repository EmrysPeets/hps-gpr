#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

CONFIG_DIR="${CONFIG_DIR:-study_configs/final_configs_2015}"
CONFIG_GLOB="${CONFIG_GLOB:-config_2015_*_95CL_funcform100_fixedhist.yaml}"
DATASET="${DATASET:-2015}"
MASSES="${MASSES:-0.030,0.045,0.060,0.075,0.090,0.105,0.120}"
STRENGTHS="${STRENGTHS:-s0,s1,s2,s3,s5}"
N_INJECTION_TOYS="${N_INJECTION_TOYS:-1}"
TOY_ROOT="${TOY_ROOT:-}"
CONTAINER="${CONTAINER:-}"
TOY_PATTERN="${TOY_PATTERN:-}"
PARTITION="${PARTITION:-roma}"
ACCOUNT="${ACCOUNT:-hps:hps-prod}"
QOS="${QOS:-}"
TIME="${TIME:-1:00:00}"
MEMORY="${MEMORY:-8G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
CONDA_ENV="${CONDA_ENV:-}"
HPS_GPR_BIN="${HPS_GPR_BIN:-hps-gpr}"
read -r -a hps_gpr_cmd <<< "${HPS_GPR_BIN}"

configs=(${CONFIG_DIR}/${CONFIG_GLOB})
if (( ${#configs[@]} == 0 )); then
  echo "No generated fixed-hist configs matched ${CONFIG_DIR}/${CONFIG_GLOB}." >&2
  echo "Run ${CONFIG_DIR}/make_final_2015_fixedhist_configs.py first or set CONFIG_GLOB." >&2
  exit 1
fi

for config_path in "${configs[@]}"; do
  config_name="$(basename "${config_path}")"
  tag="${config_name#config_2015_}"
  tag="${tag%_95CL_funcform100_fixedhist.yaml}"
  slurm_dir="${CONFIG_DIR}/slurm_${tag}"
  mkdir -p "${slurm_dir}"

  args=(
    "${hps_gpr_cmd[@]}" slurm-gen-funcform-inject
    --config "${config_path}" \
    --dataset "${DATASET}" \
    --masses "${MASSES}" \
    --strengths "${STRENGTHS}" \
    --n-injection-toys "${N_INJECTION_TOYS}" \
    --write-qmu \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --job-name "hps${DATASET}_${tag}" \
    --partition "${PARTITION}" \
    --account "${ACCOUNT}" \
    --time "${TIME}" \
    --memory "${MEMORY}" \
    --output "${slurm_dir}/submit_funcform_injection_${tag}.slurm"
  )
  if [[ -n "${TOY_ROOT}" ]]; then
    args+=(--toy-root "${TOY_ROOT}")
  fi
  if [[ -n "${CONTAINER}" ]]; then
    args+=(--container "${CONTAINER}")
  fi
  if [[ -n "${TOY_PATTERN}" ]]; then
    args+=(--toy-pattern "${TOY_PATTERN}")
  fi
  if [[ -n "${QOS}" ]]; then
    args+=(--qos "${QOS}")
  fi
  if [[ -n "${CONDA_ENV}" ]]; then
    args+=(--conda-env "${CONDA_ENV}")
  fi

  echo "[generate] dataset=${DATASET} config=${config_name} masses=${MASSES} strengths=${STRENGTHS}"
  "${args[@]}"
done
