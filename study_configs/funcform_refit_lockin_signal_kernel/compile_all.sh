#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTDIRS=(
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5_sigkernel_l1pt0_w1pt55
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5_sigkernel_l1pt5_w1pt24
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5_sigkernel_l2pt0_w1pt13
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0_sigkernel_l1pt0_w1pt55
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0_sigkernel_l1pt5_w1pt24
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0_sigkernel_l2pt0_w1pt13
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5_sigkernel_l1pt0_w1pt55
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5_sigkernel_l1pt5_w1pt24
  outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5_sigkernel_l2pt0_w1pt13
)

for outdir in "${OUTDIRS[@]}"; do
  echo "[inject-plot] ${outdir}"
  hps-gpr inject-plot \
    -i "${outdir}" \
    -o "${outdir}/injection_summary" \
    --dataset 2015 \
    --write-merged-toys
done

python3 study_configs/funcform_refit_lockin_signal_kernel/compare_signal_kernel_studies.py \
  --output-dir outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_signal_kernel_comparison
