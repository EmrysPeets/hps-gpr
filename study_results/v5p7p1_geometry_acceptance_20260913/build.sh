#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p qa pdf source/figures source/tables
python3 inputs/geometry/extract_sensors.py > qa/geometry_extraction.log
nice -n 10 python3 scripts/prepare_field_inputs.py > qa/field_input_validation.log
clang++ -O3 -std=c++17 -dynamiclib scripts/field_transport.cpp -o scripts/field_transport.dylib
nice -n 10 python3 scripts/build_study.py > qa/build_stdout.log
for ext in pdf png svg; do
  mv "figures/HPS_v5p7p1_geometry_illustration.$ext" "figures/HPS_v5p7p1_zero_field_projection.$ext"
done
for ext in pdf png; do
  mv "figures/HPS_v5p7p1_acceptance_overview.$ext" "figures/HPS_v5p7p1_zero_field_overview.$ext"
  mv "figures/HPS_v5p7p1_response_spectra.$ext" "figures/HPS_v5p7p1_zero_field_response.$ext"
done
nice -n 10 python3 scripts/validate_hit_selection.py > qa/hit_validation.log
nice -n 10 python3 scripts/validate_magnetic.py > qa/magnetic_validation.log
nice -n 10 python3 scripts/scan_magnetic.py > qa/magnetic_scan.log
nice -n 10 python3 scripts/plot_magnetic.py
cp figures/*.pdf source/figures/
cp derived/*table.tex source/tables/
(cd source && tectonic --keep-logs --outdir ../qa main.tex)
cp qa/main.pdf pdf/HPS_v5p7p1_Geometry_Acceptance_Study.pdf
