#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" scripts/ideal_null.py
"$PYTHON_BIN" scan_maximum/scripts/audit.py
"$PYTHON_BIN" scan_maximum/scripts/make_figures.py
"$PYTHON_BIN" residual_diagnostic/run.py
"$PYTHON_BIN" residual_diagnostic/summarize.py
"$PYTHON_BIN" review/independent_checks.py
"$PYTHON_BIN" review/review_results.py
"$PYTHON_BIN" scripts/build_tables.py
tectonic --outdir . source/report.tex
mv report.pdf HPS_GPR_v5.9.5_Null_Bias_Study.pdf
