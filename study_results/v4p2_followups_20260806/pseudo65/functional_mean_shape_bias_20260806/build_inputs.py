#!/usr/bin/env python3
"""Build deterministic central-window means for the pseudo65 shape-bias test.

The inputs in this directory are diagnostics, not pseudo-observed datasets.
Inside [60, 70) MeV they use the fractional fGenGammaThresh or fixed-GP
expectation already stored in the validated pseudo65 ROOT file.  The observed
2021 10% spectrum is retained bitwise outside that interval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import uproot
import yaml


HERE = Path(__file__).resolve().parent
PSEUDO65 = HERE.parent
REPO = HERE.parents[3]

SOURCE_ROOT = PSEUDO65 / "inputs" / "pseudo65_background_replacements.root"
BASE_CONFIG = (
    PSEUDO65
    / "configs"
    / "config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml"
)
OUTPUT_ROOT = HERE / "inputs" / "deterministic_central_means.root"
PROVENANCE = HERE / "derived" / "input_provenance.json"

SOURCE_ROOT_SHA256 = (
    "c5ea3922ddb70164f6184a8661d803a6d82302747c0d213c3a37bcab31be11ab"
)
BASE_CONFIG_SHA256 = (
    "74d938bed8372141bafc32addefc17d858bbc7958df9217ff188b98e47de9c76"
)

SOURCE_KEY = "source/preselection/h_invM_8000"
FUNCTIONAL_EXPECTATION_KEY = "expectations/fGenGammaThresh_m065"
GP_EXPECTATION_KEY = "expectations/gp_mean_m065"
FUNCTIONAL_OUTPUT_KEY = (
    "functional_mean_shape_bias/preselection/h_invM_8000"
)
GP_OUTPUT_KEY = "gp_mean_shape_bias/preselection/h_invM_8000"
REPLACEMENT_INTERVAL_GEV = (0.060, 0.070)
SCAN_RANGE_GEV = (0.055, 0.075)

LANES = {
    "functional_mean": {
        "hist_key": FUNCTIONAL_OUTPUT_KEY,
        "config_name": "config_functional_mean_shape_bias.yaml",
    },
    "gp_mean": {
        "hist_key": GP_OUTPUT_KEY,
        "config_name": "config_gp_mean_shape_bias.yaml",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_histogram(
    root_file: uproot.ReadOnlyDirectory,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    values, edges = root_file[key].to_numpy()
    return np.asarray(values, dtype=np.float64), np.asarray(
        edges, dtype=np.float64
    )


def make_config(lane: str, hist_key: str, output_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["path_2021"] = repo_relative(OUTPUT_ROOT)
    payload["hist_2021"] = hist_key
    payload["output_dir"] = repo_relative(HERE / "runs" / lane / "attempt_01")
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    for directory in (HERE / "inputs", HERE / "configs", HERE / "derived"):
        directory.mkdir(parents=True, exist_ok=True)

    require(SOURCE_ROOT.is_file(), f"Missing source ROOT file: {SOURCE_ROOT}")
    require(
        sha256_file(SOURCE_ROOT) == SOURCE_ROOT_SHA256,
        "Validated pseudo65 ROOT checksum changed",
    )
    require(BASE_CONFIG.is_file(), f"Missing canonical card: {BASE_CONFIG}")
    require(
        sha256_file(BASE_CONFIG) == BASE_CONFIG_SHA256,
        "Canonical v4.2 functional replacement card checksum changed",
    )

    with uproot.open(SOURCE_ROOT) as root_file:
        source, edges = load_histogram(root_file, SOURCE_KEY)
        functional_expectation, functional_edges = load_histogram(
            root_file,
            FUNCTIONAL_EXPECTATION_KEY,
        )
        gp_expectation, gp_edges = load_histogram(
            root_file,
            GP_EXPECTATION_KEY,
        )

    require(
        np.array_equal(edges, functional_edges)
        and np.array_equal(edges, gp_edges),
        "Stored expectations do not share the source histogram geometry",
    )
    lower, upper = REPLACEMENT_INTERVAL_GEV
    central = (edges[:-1] >= lower - 1.0e-12) & (
        edges[1:] <= upper + 1.0e-12
    )
    require(int(np.count_nonzero(central)) == 80, "Expected 80 native bins")
    require(
        np.all(functional_expectation[~central] == 0.0)
        and np.all(gp_expectation[~central] == 0.0),
        "Stored central expectations unexpectedly populate outside bins",
    )
    require(
        np.all(functional_expectation[central] > 0.0)
        and np.all(gp_expectation[central] > 0.0),
        "Stored central expectations are not strictly positive",
    )

    functional_mean = source.copy()
    gp_mean = source.copy()
    functional_mean[central] = functional_expectation[central]
    gp_mean[central] = gp_expectation[central]
    require(
        np.array_equal(functional_mean[~central], source[~central])
        and np.array_equal(gp_mean[~central], source[~central]),
        "Outside-window data were not retained bitwise",
    )
    require(
        np.array_equal(
            functional_mean[central],
            functional_expectation[central],
        )
        and np.array_equal(gp_mean[central], gp_expectation[central]),
        "Central deterministic replacements do not match stored expectations",
    )
    require(
        np.any(functional_mean[central] != np.rint(functional_mean[central]))
        and np.any(gp_mean[central] != np.rint(gp_mean[central])),
        "Deterministic mean inputs must retain fractional expectation values",
    )

    metadata = {
        "schema_version": 1,
        "study": "pseudo65 deterministic central-mean shape-bias diagnostic",
        "input_semantics": (
            "fractional deterministic mean inside [60,70) MeV and observed "
            "2021 10% data outside; not an observed dataset or pseudoexperiment"
        ),
        "source_root_sha256": SOURCE_ROOT_SHA256,
        "source_key": SOURCE_KEY,
        "replacement_interval_GeV": list(REPLACEMENT_INTERVAL_GEV),
        "scan_range_GeV": list(SCAN_RANGE_GEV),
        "keys": {
            "functional_mean": FUNCTIONAL_OUTPUT_KEY,
            "gp_mean": GP_OUTPUT_KEY,
            "functional_expectation": FUNCTIONAL_EXPECTATION_KEY,
            "gp_expectation": GP_EXPECTATION_KEY,
        },
    }
    with uproot.recreate(OUTPUT_ROOT) as output:
        output[SOURCE_KEY] = (source, edges)
        output[FUNCTIONAL_OUTPUT_KEY] = (functional_mean, edges)
        output[GP_OUTPUT_KEY] = (gp_mean, edges)
        output[FUNCTIONAL_EXPECTATION_KEY] = (
            functional_expectation,
            edges,
        )
        output[GP_EXPECTATION_KEY] = (gp_expectation, edges)
        output["metadata/json"] = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        )

    config_records = {}
    for lane, info in LANES.items():
        config_path = HERE / "configs" / info["config_name"]
        make_config(lane, info["hist_key"], config_path)
        config_records[lane] = artifact(config_path)

    provenance = {
        "schema_version": 1,
        "status": "GENERATED",
        "interpretation": (
            "Conditional deterministic shape-bias diagnostic using fractional "
            "central means. It is not an observed result, expected result, "
            "coverage study, or independent global-null pseudoexperiment."
        ),
        "sources": {
            "validated_pseudo65_root": artifact(SOURCE_ROOT),
            "canonical_v4p2_card": artifact(BASE_CONFIG),
        },
        "replacement": {
            "interval_GeV": list(REPLACEMENT_INTERVAL_GEV),
            "native_bin_count": int(np.count_nonzero(central)),
            "native_bin_width_GeV": float(np.median(np.diff(edges))),
            "outside_source_bitwise_identical": True,
            "fractional_values_retained": True,
            "functional_central_sum": float(functional_mean[central].sum()),
            "gp_central_sum": float(gp_mean[central].sum()),
            "functional_array_sha256": sha256_array(functional_mean),
            "gp_array_sha256": sha256_array(gp_mean),
            "source_array_sha256": sha256_array(source),
        },
        "scan": {
            "mass_range_GeV": list(SCAN_RANGE_GEV),
            "mass_step_GeV": 0.001,
            "card_difference_whitelist": [
                "path_2021",
                "hist_2021",
                "output_dir",
            ],
            "expected_limit_bands": False,
            "toys": False,
        },
        "output_root": artifact(OUTPUT_ROOT),
        "configs": config_records,
        "generator": artifact(Path(__file__).resolve()),
    }
    PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
