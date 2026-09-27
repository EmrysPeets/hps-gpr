#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/digitize.py
python3 scripts/analyze.py > qa/analysis.log
python3 scripts/make_tables.py
tectonic -X compile source/main.tex --outdir output/pdf --keep-logs
cp output/pdf/main.pdf output/pdf/APEX_Initial_Studies.pdf
python3 scripts/validate.py
