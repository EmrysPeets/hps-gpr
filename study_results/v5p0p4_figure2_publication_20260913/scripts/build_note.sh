#!/usr/bin/env bash
set -euo pipefail
v5_bundle_dir="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$v5_bundle_dir/qa/build"
cd "$v5_bundle_dir/source"
tectonic -C --keep-logs --keep-intermediates --outdir ../qa/build main.tex
