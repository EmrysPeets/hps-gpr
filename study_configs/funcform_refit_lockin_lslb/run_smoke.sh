#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Smoke-test the new comparator settings on one functional-form toy.

echo "[smoke] lslb0p5"
hps-gpr funcform-inject \
  --config study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb0p5.yaml \
  --dataset 2015 \
  --max-toys 1 \
  --masses 0.060 \
  --strengths s1 \
  --n-injection-toys 1 \
  --write-toy-csv \
  --write-qmu \
  --output-dir outputs/smoke_2015_funcform_refit_lockin_lslb0p5

echo "[smoke] lslb1p0"
hps-gpr funcform-inject \
  --config study_configs/config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb1p0.yaml \
  --dataset 2015 \
  --max-toys 1 \
  --masses 0.060 \
  --strengths s1 \
  --n-injection-toys 1 \
  --write-toy-csv \
  --write-qmu \
  --output-dir outputs/smoke_2015_funcform_refit_lockin_lslb1p0
