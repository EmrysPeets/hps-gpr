#!/usr/bin/env bash
set -euo pipefail

hps-gpr funcform-inject \
  --config study_configs/config_2015_blind1p64_95CL_funcform100_refit_lslb1p5.yaml \
  --dataset 2015 \
  --max-toys 1 \
  --masses 0.060 \
  --strengths s1 \
  --n-injection-toys 1 \
  --write-toy-csv \
  --write-qmu \
  --output-dir outputs/smoke_2015_funcform_refit_lslb_1pt5
