#!/usr/bin/env bash
set -eu
V639_PYTHON=${V639_PYTHON:-python3}
V639_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$V639_PYTHON" "$V639_ROOT/scripts/run_combined.py"
"$V639_PYTHON" "$V639_ROOT/scripts/audit_results.py"
"$V639_PYTHON" "$V639_ROOT/scripts/make_figures.py"
"$V639_PYTHON" "$V639_ROOT/scripts/build_report.py" --build
