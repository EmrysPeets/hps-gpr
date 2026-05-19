#!/usr/bin/env bash
set -euo pipefail

OUTDIRS=(
  outputs/final_2015_funcform_fixedhist_blind1pt64_lslb_0pt5
  outputs/final_2015_funcform_fixedhist_blind1pt64_lslb_1pt0
  outputs/final_2015_funcform_fixedhist_blind1pt96_lslb_0pt5
  outputs/final_2015_funcform_fixedhist_blind1pt96_lslb_1pt0
)

for outdir in "${OUTDIRS[@]}"; do
  echo "[inject-plot] ${outdir}"
  hps-gpr inject-plot \
    -i "${outdir}" \
    -o "${outdir}/injection_summary" \
    --dataset 2015 \
    --write-merged-toys
done
