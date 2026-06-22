#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTDIRS=(
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5
)

for outdir in "${OUTDIRS[@]}"; do
  echo "[inject-plot] ${outdir}"
  hps-gpr inject-plot \
    -i "${outdir}" \
    -o "${outdir}/injection_summary" \
    --dataset 2015 \
    --write-merged-toys
done

python3 study_configs/funcform_refit_lockin_lslb/compare_lslb_studies.py \
  --output-dir outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_comparison
