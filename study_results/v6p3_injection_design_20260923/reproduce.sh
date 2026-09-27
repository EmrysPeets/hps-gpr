#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
tectonic_bin="${TECTONIC:-tectonic}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/hps_gpr_v63_matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$study_dir/pdf"
"$python_bin" "$study_dir/audit/audit_saved_rows.py"
"$python_bin" "$study_dir/scripts/illustrate_design.py"
"$python_bin" "$study_dir/scripts/make_table.py"
cd "$study_dir/source"
"$tectonic_bin" -C --keep-logs --outdir ../pdf report.tex
cp ../pdf/report.pdf ../pdf/HPS_GPR_v6p3_Injection_Design.pdf
