#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hps-v644-mpl}"
STUDY_PYTHON="${STUDY_PYTHON:-python3}"
if [[ "${1:-}" != "" && "${1:-}" != "--fresh-validation" ]]; then
  printf '%s\n' 'Usage: bash rebuild_appendix.sh [--fresh-validation]'
  exit 2
fi
"$STUDY_PYTHON" - <<'PY'
from pathlib import Path
import hashlib
for line in Path('provenance/input_manifest.sha256').read_text().splitlines():
    h,p=line.split('  ',1);assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
print('Frozen inputs verified.')
PY
"$STUDY_PYTHON" scripts/analyze_calibrated.py --freeze-only
if [[ "${1:-}" == "--fresh-validation" ]]; then
  "$STUDY_PYTHON" - <<'PY'
from pathlib import Path
for p in Path('results/validation_checkpoints').glob('m*.npz'):p.unlink()
print('Cleared only this package\x27s B checkpoints; ensemble A remains frozen.')
PY
fi
"$STUDY_PYTHON" scripts/run_validation.py
"$STUDY_PYTHON" scripts/analyze_calibrated.py
"$STUDY_PYTHON" qa/audit_common_coupling.py
"$STUDY_PYTHON" scripts/validate_calibrated.py
"$STUDY_PYTHON" scripts/make_calibrated_figures.py
"$STUDY_PYTHON" scripts/build_calibrated_appendix.py
