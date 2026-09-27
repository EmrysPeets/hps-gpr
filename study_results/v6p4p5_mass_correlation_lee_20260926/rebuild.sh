#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/v645_mpl}"
STUDY_PYTHON="${STUDY_PYTHON:-python3}"
"$STUDY_PYTHON" scripts/analyze_correlations.py
"$STUDY_PYTHON" scripts/figures.py
"$STUDY_PYTHON" scripts/build_report.py
"$STUDY_PYTHON" scripts/qa_report.py
