#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_DIR/source"
tectonic -C report.tex --keep-logs
mkdir -p "$STUDY_DIR/pdf"
cp report.pdf "$STUDY_DIR/pdf/HPS_GPR_v5p9_Signal_Tail_Study.pdf"
