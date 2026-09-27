#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
study_python="${HPS_PYTHON:-/Applications/Xcode.app/Contents/Developer/usr/bin/python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/hps-v631-reproduce-mpl}"
"$study_python" "$study_dir/scripts/launch.py" --workers 4
"$study_python" "$study_dir/scripts/validate.py"
"$study_python" "$study_dir/scripts/summarize.py"
"$study_python" "$study_dir/scripts/make_report.py" --build
printf '%s\n' "Numerical validation and PDF rebuild complete. Render and inspect pdf/report.pdf before publishing a rebuilt package."
