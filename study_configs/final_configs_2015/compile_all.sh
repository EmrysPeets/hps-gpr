#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

CONFIG_DIR="${CONFIG_DIR:-study_configs/final_configs_2015}"
CONFIG_GLOB="${CONFIG_GLOB:-config_2015_*_95CL_funcform100_fixedhist.yaml}"
DATASET="${DATASET:-2015}"
HPS_GPR_BIN="${HPS_GPR_BIN:-hps-gpr}"
read -r -a hps_gpr_cmd <<< "${HPS_GPR_BIN}"

configs=(${CONFIG_DIR}/${CONFIG_GLOB})
if (( ${#configs[@]} == 0 )); then
  echo "No generated fixed-hist configs matched ${CONFIG_DIR}/${CONFIG_GLOB}." >&2
  echo "Run ${CONFIG_DIR}/make_final_2015_fixedhist_configs.py first or set CONFIG_GLOB." >&2
  exit 1
fi

for config_path in "${configs[@]}"; do
  outdir="$(
    python3 - "${config_path}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print((yaml.safe_load(handle) or {}).get("output_dir", ""))
PY
  )"
  if [[ -z "${outdir}" ]]; then
    echo "Skipping ${config_path}: no output_dir" >&2
    continue
  fi
  echo "[inject-plot] ${outdir}"
  "${hps_gpr_cmd[@]}" inject-plot \
    -i "${outdir}" \
    -o "${outdir}/injection_summary" \
    --dataset "${DATASET}" \
    --write-merged-toys
done
