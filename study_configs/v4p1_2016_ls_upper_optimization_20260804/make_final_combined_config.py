#!/usr/bin/env python3
"""Generate or audit the isolated v4.1 factor-12 combined observed card.

The parsed card is required to differ from the reviewed v4 combined card in
exactly two leaves:

* ``kernel_ls_res_upper_factor_by_dataset.2016``: 8 -> 12;
* ``output_dir``: the isolated factor-12 combined observed directory.

No expected-band or toy switch is enabled by this generator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = (
    REPO
    / "study_configs"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "config_obsUL90_combined_wide_support_v4_observed_only.yaml"
)
FINAL_CONFIG = (
    HERE
    / "config_obsUL90_combined_wide_support_v4p1_2016k12_observed_only.yaml"
)
FINAL_OUTPUT_DIR = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "final_k12_combined_observed"
)
EXPECTED_PARSED_DIFFERENCES = (
    "kernel_ls_res_upper_factor_by_dataset.2016",
    "output_dir",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten(child, label))
        return flattened
    return {prefix: value}


def parsed_differences(source: dict, candidate: dict) -> list[str]:
    source_flat = flatten(source)
    candidate_flat = flatten(candidate)
    return sorted(
        key
        for key in set(source_flat) | set(candidate_flat)
        if source_flat.get(key) != candidate_flat.get(key)
    )


def build_card(source: dict) -> dict:
    card = copy.deepcopy(source)
    card["kernel_ls_res_upper_factor_by_dataset"]["2016"] = 12.0
    card["output_dir"] = str(FINAL_OUTPUT_DIR)
    return card


def validate_card(source: dict, card: dict) -> dict:
    differences = parsed_differences(source, card)
    if differences != list(EXPECTED_PARSED_DIFFERENCES):
        raise RuntimeError(
            "Final card must have exactly the two reviewed parsed differences; "
            f"found {differences}"
        )
    if float(
        card["kernel_ls_res_upper_factor_by_dataset"]["2016"]
    ) != 12.0:
        raise RuntimeError("The final 2016 upper length-scale factor is not 12.")
    if Path(str(card["output_dir"])).resolve() != FINAL_OUTPUT_DIR.resolve():
        raise RuntimeError("The final output_dir is not the isolated campaign path.")

    false_switches = (
        "make_ul_bands",
        "do_combined_bands",
        "make_eps2_bands",
    )
    for key in false_switches:
        if bool(card[key]):
            raise RuntimeError(f"Band-production switch {key} must remain false.")
    zero_counts = (
        "cls_num_toys",
        "ul_bands_toys",
        "combined_bands_n_toys",
    )
    for key in zero_counts:
        if int(card[key]) != 0:
            raise RuntimeError(f"Toy count {key} must remain zero.")
    if str(card["cls_mode"]).lower().strip() != "asymptotic":
        raise RuntimeError("The final card must retain asymptotic CLs.")
    if not bool(card["do_combined"]):
        raise RuntimeError("The final card must retain the combined likelihood.")
    if str(card["combined_mode"]).lower().strip() != "count_scale":
        raise RuntimeError("The final card must retain combined_mode=count_scale.")

    return {
        "source_config": str(SOURCE.relative_to(REPO)),
        "source_config_sha256": sha256(SOURCE),
        "final_config": str(FINAL_CONFIG.relative_to(REPO)),
        "parsed_differences_from_v4": differences,
        "expected_bands": False,
        "toy_draws": 0,
        "cls": "90% asymptotic CLs",
        "combined_mode": "count_scale",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the generated YAML before auditing it.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    generated = build_card(source)
    validate_card(source, generated)

    if args.write:
        rendered = (
            "# Generated from the reviewed v4 combined observed card.\n"
            "# Only the 2016 upper factor and isolated output_dir are changed.\n"
            + yaml.safe_dump(generated, sort_keys=False)
        )
        FINAL_CONFIG.write_text(rendered, encoding="utf-8")

    if not FINAL_CONFIG.is_file():
        raise SystemExit(
            f"Final config does not exist: {FINAL_CONFIG}. Pass --write to create it."
        )
    on_disk = yaml.safe_load(FINAL_CONFIG.read_text(encoding="utf-8"))
    if on_disk != generated:
        differences = parsed_differences(generated, on_disk)
        raise RuntimeError(
            "On-disk final card does not match the generated card; "
            f"generated-vs-disk differences: {differences}"
        )

    audit = validate_card(source, on_disk)
    audit["final_config_sha256"] = sha256(FINAL_CONFIG)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
