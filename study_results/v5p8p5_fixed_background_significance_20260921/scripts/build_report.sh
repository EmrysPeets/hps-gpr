#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$study_dir"
tectonic -X compile source/report.tex --keep-logs
pdftoppm -r 110 -png source/report.pdf qa/page
