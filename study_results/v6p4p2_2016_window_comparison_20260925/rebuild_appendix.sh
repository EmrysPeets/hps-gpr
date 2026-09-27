#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/v642_mpl"
STUDY_PYTHON="${STUDY_PYTHON:-python3}"
if [[ "${1:-}" == "--fresh" ]]; then
  "$STUDY_PYTHON" - <<'PY'
from pathlib import Path
for p in Path('results/window_comparison/checkpoints').glob('*.json'):p.unlink()
PY
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: bash rebuild_appendix.sh [--fresh]' >&2
  exit 2
fi
"$STUDY_PYTHON" scripts/run_window_comparison.py
"$STUDY_PYTHON" scripts/validate_window_comparison.py
"$STUDY_PYTHON" scripts/make_window_appendix.py
"$STUDY_PYTHON" scripts/build_appendix_report.py
