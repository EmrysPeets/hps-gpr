#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x /Applications/Xcode.app/Contents/Developer/usr/bin/python3 ]]; then
    PYTHON_BIN=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
TECTONIC_BIN="${TECTONIC:-}"
if [[ -z "$TECTONIC_BIN" ]]; then
  TECTONIC_BIN="$(command -v tectonic || true)"
fi
if [[ -z "$TECTONIC_BIN" ]]; then
  TECTONIC_BIN=/opt/homebrew/bin/tectonic
fi
export PYTHONDONTWRITEBYTECODE=1
(
  cd "$STUDY_DIR/source"
  "$TECTONIC_BIN" -C --keep-logs --outdir ../pdf report.tex
)
cp "$STUDY_DIR/pdf/report.pdf" "$STUDY_DIR/pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf"
"$PYTHON_BIN" "$STUDY_DIR/scripts/make_report.py" --record-pdf
