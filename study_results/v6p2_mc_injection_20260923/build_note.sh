#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$STUDY_DIR/scripts/build_note.sh" "$@"
