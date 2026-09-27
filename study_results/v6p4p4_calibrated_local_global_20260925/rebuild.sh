#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hps-v643-mpl}"
STUDY_PYTHON="${STUDY_PYTHON:-python3}"
if [[ "${1:-}" != "" && "${1:-}" != "--fresh" ]]; then
  printf '%s\n' 'Usage: bash rebuild.sh [--fresh]'
  exit 2
fi
"$STUDY_PYTHON" - <<'PY'
from pathlib import Path
import hashlib
for line in Path('provenance/input_manifest.sha256').read_text().splitlines():
    h,p=line.split('  ',1)
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
print('Frozen input hashes verified.')
PY
if [[ "${1:-}" == "--fresh" ]]; then
  "$STUDY_PYTHON" - <<'PY'
from pathlib import Path
for p in Path('results/checkpoints').glob('m*.npz'):p.unlink()
print('This package\x27s generated mass checkpoints cleared for a fresh run.')
PY
fi
"$STUDY_PYTHON" scripts/run_global.py --mode observed
"$STUDY_PYTHON" scripts/run_global.py --mode calibrate
"$STUDY_PYTHON" scripts/analyze_global.py
"$STUDY_PYTHON" scripts/validate_global.py
"$STUDY_PYTHON" scripts/make_global_figures.py
"$STUDY_PYTHON" scripts/build_global_report.py
