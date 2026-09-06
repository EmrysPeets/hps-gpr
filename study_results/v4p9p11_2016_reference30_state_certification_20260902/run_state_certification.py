#!/usr/bin/env python3
"""Noncanonical compatibility pointer; this monolithic runner is disabled."""

raise SystemExit(
    "This transient monolithic runner is disabled. The canonical frozen "
    "control source is run_control_frozen.py and downstream archive/robust "
    "execution is run_downstream_certification.py; see the freeze files and "
    "PRE_ARCHIVE_CODE_SPLIT_AMENDMENT.md."
)
