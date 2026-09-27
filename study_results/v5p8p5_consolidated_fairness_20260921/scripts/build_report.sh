#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_DIR"
tectonic -X compile source/report.tex --keep-logs
