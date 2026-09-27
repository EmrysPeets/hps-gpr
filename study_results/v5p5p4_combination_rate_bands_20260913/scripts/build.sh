#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$study_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 scripts/extracted_combinations.py > qa/extracted_run.log
python3 scripts/rate_uncertainties.py > qa/rate_band_run.log
python3 scripts/rate_significance_scan.py > qa/rate_scan_run.log
python3 scripts/new_figures.py
cd source
tectonic --keep-logs main.tex
cp main.pdf ../pdf/HPS_GPR_v5p5p4_Combination_Rate_Uncertainties.pdf
cp main.log ../qa/main.log
cd ..
python3 scripts/validate.py
