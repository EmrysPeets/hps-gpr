#!/usr/bin/env python3
"""Build the full-2016 observed card only after support is frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import yaml

from observed_2016_contract import (
    BASE_CARD,
    CARD,
    CARD_MANIFEST,
    CONFIRMATION_AUDIT,
    FREEZE,
    HERE,
    INDEPENDENT_AUDITOR,
    OBSERVED_ROOT,
    PHASE1_AUDIT,
    PROTOCOL,
    SCIENTIFIC_SCOPE_CLARIFICATION,
    STATIC_TRUTH_AUDIT,
    STUDY_ID,
    STUDY_SPEC,
    atomic_json,
    atomic_text,
    sha256,
    static_preflight,
    validate_card,
    validate_freeze,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "build"))
    parser.add_argument("--support-freeze", type=Path, default=FREEZE)
    parser.add_argument("--output-card", type=Path, default=CARD)
    parser.add_argument("--manifest", type=Path, default=CARD_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    static = static_preflight()
    if args.mode == "preflight":
        status = "production_blocked_no_provisional_edge"
        selected = None
        print(
            json.dumps(
                {
                    "status": status,
                    "observed_data_evaluated": False,
                    "selected_support_low_MeV": selected,
                    "static": static,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.output_card.expanduser().resolve() != CARD.resolve():
        raise SystemExit(f"output card must be {CARD}")
    if args.manifest.expanduser().resolve() != CARD_MANIFEST.resolve():
        raise SystemExit(f"card manifest must be {CARD_MANIFEST}")
    if CARD.exists() or CARD_MANIFEST.exists():
        raise SystemExit("refusing to overwrite a frozen observed card or manifest")
    decision = validate_freeze(args.support_freeze)
    card = yaml.safe_load(BASE_CARD.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise SystemExit("frozen v4.2 card is not a YAML mapping")

    card.update(
        {
            "path_2015": "",
            "path_2016": "inputs/source_2016_full.root",
            "path_2021": "",
            "enable_2015": False,
            "enable_2016": True,
            "enable_2021": False,
            "data_range_2016": list(map(float, decision["data_range_2016"])),
            "scan_parallel": True,
            "scan_n_workers": 4,
            "scan_parallel_backend": "threading",
            "scan_threads_per_worker": 1,
            "make_ul_bands": False,
            "ul_bands_toys": 0,
            "do_combined_bands": False,
            "combined_bands_n_toys": 0,
            "make_eps2_bands": False,
            "run_limit_bands_on": "",
            "inject_signal": False,
            "do_combined": False,
            "debug_print": False,
            "fail_fast": True,
            "save_fit_json": True,
            "save_per_mass_folders": True,
            "save_plots": False,
            "output_dir": "observed_scan/2016_full_primary",
        }
    )
    card.setdefault("data_visibility", {})["2016"] = "observed"
    text = yaml.safe_dump(card, sort_keys=False)
    atomic_text(CARD, text)
    manifest = {
        "schema_version": 1,
        "status": "observed_2016_card_frozen",
        "study_id": STUDY_ID,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "frozen_protocol_sha256": sha256(PROTOCOL),
        "support_freeze": str(FREEZE.relative_to(HERE)),
        "support_freeze_sha256": sha256(FREEZE),
        "independent_freeze_auditor": str(
            INDEPENDENT_AUDITOR.relative_to(HERE)
        ),
        "independent_freeze_auditor_sha256": sha256(INDEPENDENT_AUDITOR),
        "static_truth_audit": str(STATIC_TRUTH_AUDIT.relative_to(HERE)),
        "static_truth_audit_sha256": sha256(STATIC_TRUTH_AUDIT),
        "scientific_scope_clarification": str(
            SCIENTIFIC_SCOPE_CLARIFICATION.relative_to(HERE)
        ),
        "scientific_scope_clarification_sha256": sha256(
            SCIENTIFIC_SCOPE_CLARIFICATION
        ),
        "phase1_selection_audit": str(PHASE1_AUDIT.relative_to(HERE)),
        "phase1_selection_audit_sha256": sha256(PHASE1_AUDIT),
        "confirmation_freeze_audit": str(
            CONFIRMATION_AUDIT.relative_to(HERE)
        ),
        "confirmation_freeze_audit_sha256": sha256(CONFIRMATION_AUDIT),
        "base_card": str(BASE_CARD.relative_to(HERE)),
        "base_card_sha256": sha256(BASE_CARD),
        "observed_root": str(OBSERVED_ROOT.relative_to(HERE)),
        "observed_root_sha256": sha256(OBSERVED_ROOT),
        "selected_support_low_MeV": int(
            decision["selected_support_low_MeV"]
        ),
        "support_high_MeV": int(decision["support_high_MeV"]),
        "data_range_2016": list(map(float, decision["data_range_2016"])),
        "analysis_range_2016": [0.039, 0.180],
        "mass_step_GeV": 0.001,
        "expected_rows": 142,
        "kernel_ls_res_upper_factor_2016": 12.0,
        "card": str(CARD.relative_to(HERE)),
        "card_sha256": sha256(CARD),
        "card_builder": str(Path(__file__).resolve().relative_to(HERE)),
        "card_builder_sha256": sha256(Path(__file__).resolve()),
        "observed_data_evaluated_while_building": False,
    }
    atomic_json(CARD_MANIFEST, manifest)
    validate_card(CARD, CARD_MANIFEST, decision)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
