#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/v641_mpl"
STUDY_PYTHON="${STUDY_PYTHON:-python3}"
if [[ "${1:-}" == "--fresh" ]]; then
  "$STUDY_PYTHON" - <<'PY'
from pathlib import Path
import shutil
for name in ('checkpoints','toy_checkpoints','rank_checkpoints','window_comparison/checkpoints'):
    p=Path('results')/name
    if p.exists(): shutil.rmtree(p)
PY
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: bash rebuild.sh [--fresh]' >&2
  exit 2
fi
mkdir -p results/checkpoints results/selected_fits results/toy_checkpoints results/rank_checkpoints figures pdf qa
"$STUDY_PYTHON" scripts/run_scan.py --workers 2
"$STUDY_PYTHON" scripts/run_local_checks.py
"$STUDY_PYTHON" scripts/run_rank_limits.py
"$STUDY_PYTHON" scripts/check_2016_interpolation.py
"$STUDY_PYTHON" scripts/validate_extraction.py
"$STUDY_PYTHON" scripts/validate_toys.py
"$STUDY_PYTHON" scripts/make_shape_figures.py
"$STUDY_PYTHON" scripts/make_extraction_figures.py
"$STUDY_PYTHON" scripts/build_report.py

bash rebuild_appendix.sh
