#!/usr/bin/env bash
set -eu
V638_PYTHON=${V638_PYTHON:-python3}
V638_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/hps-v638-rebuild-mpl}
"$V638_PYTHON" "$V638_ROOT/scripts/run_observed.py"
"$V638_PYTHON" "$V638_ROOT/scripts/validate_legacy_baseline.py"
"$V638_PYTHON" "$V638_ROOT/scripts/audit_statistics.py"
"$V638_PYTHON" "$V638_ROOT/scripts/replay_fits.py"
"$V638_PYTHON" "$V638_ROOT/scripts/make_figures.py"
"$V638_PYTHON" "$V638_ROOT/scripts/build_report.py" --build
