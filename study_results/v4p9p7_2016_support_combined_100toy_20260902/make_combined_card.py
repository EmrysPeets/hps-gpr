#!/usr/bin/env python3
"""Materialize the v4.9.7 combined card from the frozen v4.2 baseline.

The selected 2016 edge is never inferred: both CLI coordinates and the
machine-readable freeze decision must agree before a card can be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


HERE = Path(__file__).resolve().parent
FROZEN_CARD = HERE / "inputs" / "frozen_v4p2_analysis_card.yaml"
DEFAULT_INPUTS = {
    "2015": HERE / "inputs" / "source_2015_full.root",
    "2016": HERE / "inputs" / "source_2016_full.root",
    "2021": HERE / "inputs" / "source_2021_10pct.root",
}
FROZEN_CARD_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
EXPECTED_INPUT_SHA256 = {
    "2015": "58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f",
    "2016": "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301",
    "2021": "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4",
}


class ContractError(RuntimeError):
    """Raised when an immutable campaign input or decision fails validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_freeze(
    path: Path,
    support_low_mev: int,
    support_high_mev: int,
) -> Dict[str, Any]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    study_spec = HERE / "study_spec.json"
    protocol = HERE / "STUDY_PROTOCOL.md"
    if decision.get("study_id") != HERE.name:
        raise ContractError("Support decision names another study.")
    if decision.get("study_spec_sha256") != sha256(study_spec):
        raise ContractError("Support decision does not match study_spec.json.")
    if decision.get("frozen_protocol_sha256") != sha256(protocol):
        raise ContractError("Support decision does not match STUDY_PROTOCOL.md.")
    if decision.get("status") != "support_edge_frozen":
        raise ContractError("Support decision is not support_edge_frozen.")
    if decision.get("observed_scan_authorized") is not True:
        raise ContractError("Support decision does not authorize observed use.")
    found = (
        int(decision["selected_support_low_MeV"]),
        int(decision["support_high_MeV"]),
    )
    requested = (int(support_low_mev), int(support_high_mev))
    if found != requested:
        raise ContractError(
            f"Explicit support {requested} does not match freeze {found}."
        )
    if found[0] not in range(28, 34) or found[1] != 210:
        raise ContractError(
            f"Support {found} is outside the predeclared eligible geometry."
        )
    if decision.get("data_range_2016") != [found[0] / 1000.0, 0.210]:
        raise ContractError("Support decision has inconsistent data_range_2016.")
    for key in (
        "absolute_upper_limit_used_for_selection",
        "retuning_after_confirmation",
        "holdout_65MeV_used_for_selection",
    ):
        if decision.get(key) is not False:
            raise ContractError(f"Support decision violates frozen control {key}.")
    return decision


def require_input(path: Path, dataset: str) -> Dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ContractError(f"Missing {dataset} input: {resolved}")
    digest = sha256(resolved)
    expected = EXPECTED_INPUT_SHA256[dataset]
    if digest != expected:
        raise ContractError(
            f"Unexpected {dataset} input SHA-256: {digest} != {expected}"
        )
    return {"path": str(resolved), "sha256": digest}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-freeze-json", type=Path, required=True)
    parser.add_argument("--support-2016-low-mev", type=int, required=True)
    parser.add_argument("--support-2016-high-mev", type=int, required=True)
    parser.add_argument("--path-2015", type=Path, default=DEFAULT_INPUTS["2015"])
    parser.add_argument("--path-2016", type=Path, default=DEFAULT_INPUTS["2016"])
    parser.add_argument("--path-2021", type=Path, default=DEFAULT_INPUTS["2021"])
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--analysis-output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not FROZEN_CARD.is_file() or sha256(FROZEN_CARD) != FROZEN_CARD_SHA256:
        raise SystemExit("Frozen v4.2 card is absent or has the wrong SHA-256.")

    freeze_path = args.support_freeze_json.expanduser().resolve()
    if not freeze_path.is_file():
        raise SystemExit(f"Missing support freeze: {freeze_path}")
    decision = load_freeze(
        freeze_path,
        args.support_2016_low_mev,
        args.support_2016_high_mev,
    )
    inputs = {
        "2015": require_input(args.path_2015, "2015"),
        "2016": require_input(args.path_2016, "2016"),
        "2021": require_input(args.path_2021, "2021"),
    }

    card = yaml.safe_load(FROZEN_CARD.read_text(encoding="utf-8"))
    immutable_expectations = {
        "range_2015": [0.019, 0.090],
        "range_2016": [0.039, 0.180],
        "range_2021": [0.050, 0.250],
        "data_range_2015": [0.014, 0.135],
        "data_range_2016": [0.030, 0.210],
        "data_range_2021": [0.040, 0.300],
        "combined_mode": "count_scale",
        "cls_mode": "asymptotic",
        "cls_alpha": 0.1,
        "combined_bands_seed": 24680,
    }
    for key, expected in immutable_expectations.items():
        if card.get(key) != expected:
            raise ContractError(
                f"Frozen baseline key {key} changed: {card.get(key)!r} != {expected!r}"
            )
    expected_upper = {"2015": 8.0, "2016": 12.0, "2021": 15.0}
    configured_upper = {
        str(key): float(value)
        for key, value in card["kernel_ls_res_upper_factor_by_dataset"].items()
    }
    if configured_upper != expected_upper:
        raise ContractError("Frozen dataset-specific upper length-scale factors changed.")

    card["path_2015"] = inputs["2015"]["path"]
    card["path_2016"] = inputs["2016"]["path"]
    card["path_2021"] = inputs["2021"]["path"]
    card["data_range_2016"] = [
        args.support_2016_low_mev / 1000.0,
        args.support_2016_high_mev / 1000.0,
    ]
    card["data_range_2021"] = [0.036, 0.300]
    card["combined_bands_n_toys"] = 100
    card["combined_bands_seed"] = 24680
    card["make_ul_bands"] = True
    card["ul_bands_toys"] = 0
    card["run_limit_bands_on"] = ""
    card["ul_bands_refit_gp_on_toy"] = False
    card["do_combined_bands"] = True
    card["combined_mode"] = "count_scale"
    card["output_dir"] = str(args.analysis_output_dir.expanduser().resolve())

    config_out = args.config_out.expanduser().resolve()
    provenance_out = args.provenance_out.expanduser().resolve()
    if config_out.exists() or provenance_out.exists():
        raise SystemExit("Refusing to overwrite a derived card or provenance.")
    rendered = (
        "# Derived v4.9.7 combined card; see adjacent provenance JSON.\n"
        + yaml.safe_dump(card, sort_keys=False)
    )
    atomic_text(config_out, rendered)
    provenance = {
        "schema_version": 1,
        "source_card": str(FROZEN_CARD.resolve()),
        "source_card_sha256": FROZEN_CARD_SHA256,
        "support_freeze": str(freeze_path),
        "support_freeze_sha256": sha256(freeze_path),
        "support_freeze_status": decision["status"],
        "support_2016_low_MeV": int(args.support_2016_low_mev),
        "support_2016_high_MeV": int(args.support_2016_high_mev),
        "data_range_2021_GeV": [0.036, 0.300],
        "n_toys_per_mass": 100,
        "seed": 24680,
        "observed_inputs": inputs,
        "config_out": str(config_out),
        "config_out_sha256": sha256(config_out),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Conditional fixed-GP expected-limit bands with inner asymptotic "
            "CLs; not direct coverage or a global-significance ensemble."
        ),
    }
    atomic_text(
        provenance_out,
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )
    print(f"Wrote {config_out}")
    print(f"Wrote {provenance_out}")


if __name__ == "__main__":
    main()
