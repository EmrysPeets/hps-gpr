# Release provenance errata

This release preserves frozen execution contracts and products, so two
post-run provenance clarifications are recorded without rewriting those
hash-bound artifacts.

## Cached-toy partition

The authoritative v4p8p2 closure and locked length scan consume only toy
indices 0--19.  Their reported figures, collected ledgers, and decisions do
not use indices 20--24.

The frozen `rigid_study_spec.json` describes indices 20--24 as an untouched
reserve.  That statement is not globally true: a superseded one-lane
development run had already produced outputs for indices 20, 21, and 22.
Those outputs are retained under
`quarantine/superseded_rigid_closure_development_20260813/` and are excluded
from the authoritative result.  Indices 20--22 must not be treated as an
unopened statistical reserve.  Indices 23--24 were not consumed by the
authoritative products, and this release makes no reserve-pooling claim.

## Toy-build summary and scan QA

The development `rigid_toy_build_summary.json` refers to an earlier manifest
serialization and is quarantined under
`quarantine/stale_rigid_toy_build_summary_20260813/`.  The authoritative input
hashes are the current ROOT and manifest entries in
`FINAL_ARTIFACT_SHA256.txt` and `rigid_study_spec.json`.

The files under `qa/rigid_length_scan/` are frozen preflight and scheduling
records.  In particular, `task_manifest.csv` is a pre-run snapshot.  Final
status is given by `derived/rigid_length_scan/task_product_audit.csv` and
`derived/rigid_length_scan/collection_summary.json`.

## Frozen-path portability

The manifest, runner locks, and task records preserve absolute paths to the
original source histograms and workspace.  This makes the frozen preflight
reproducible in the production workspace but not path-portable to an arbitrary
clone: a relocated checkout will deliberately report a provenance-path drift
even when its tracked bytes match.  The committed ROOT toy input, collected
ledgers, reports, and reviewed figures are self-contained for inspection and
checksum validation; regenerating the source fit requires the external source
ROOT files at a newly declared location.
