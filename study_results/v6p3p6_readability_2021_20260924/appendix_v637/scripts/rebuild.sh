#!/usr/bin/env bash
set -eu
V637_PYTHON=${V637_PYTHON:-python3}
V637_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/hps-v637-rebuild-mpl}
"$V637_PYTHON" "$V637_ROOT/scripts/validate_shapes.py"
"$V637_PYTHON" "$V637_ROOT/scripts/run_study.py"
"$V637_PYTHON" "$V637_ROOT/scripts/calibration_mismatch.py"
"$V637_PYTHON" "$V637_ROOT/scripts/summarize_global.py"
"$V637_PYTHON" "$V637_ROOT/scripts/audit_statistics.py"
"$V637_PYTHON" "$V637_ROOT/scripts/make_figures.py"
"$V637_PYTHON" "$V637_ROOT/scripts/build_report.py" --build
