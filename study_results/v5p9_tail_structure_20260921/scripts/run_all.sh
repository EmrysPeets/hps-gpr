#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pids=()
for year in 2015 2016 2021; do
  "$PYTHON_BIN" "$STUDY_DIR/scripts/run_scan.py" --year "$year" > "$STUDY_DIR/qa/run_${year}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON_BIN" "$STUDY_DIR/scripts/aggregate.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/leakage.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/validate.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/audit_legacy.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_figures.py"
bash "$STUDY_DIR/scripts/build_report.sh"
"$PYTHON_BIN" "$STUDY_DIR/scripts/check_artifacts.py"
