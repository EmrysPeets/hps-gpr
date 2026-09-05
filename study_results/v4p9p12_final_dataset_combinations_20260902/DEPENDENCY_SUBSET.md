# Frozen dependency subset for v4.9.12.5

This directory contains only the original input tables, analysis card,
reviewed states, and solver modules needed by the
[v4.9.12.5 mass-resolution study](../v4p9p12p5_mass_resolution_uncertainty_20260905/README.md).
Their bytes are unchanged. It is not the complete v4.9.12 combination release;
running `run_final_combinations.py` as a full production job requires the
additional provenance, certification, and ROOT inputs from that release.
The 2021 diagnostic runners import its helpers and read the included binned
CSV input instead.
