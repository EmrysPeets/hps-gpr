#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$PYTHON_BIN" "$STUDY_DIR/scripts/run_mc_study.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/validate.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/extend_windows.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_report.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_window_report.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/core_centering.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/validate.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_core_report.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/analytic_shapes.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_analytic_report.py"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_MC_points_report.py"
cd "$STUDY_DIR/source"
tectonic -C report.tex --keep-logs
mkdir -p "$STUDY_DIR/pdf"
cp report.pdf "$STUDY_DIR/pdf/HPS_GPR_v6p1_MC_Signal_Templates.pdf"
