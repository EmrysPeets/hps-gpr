#!/usr/bin/env python3
"""Fail-closed validation for the deterministic pseudo65 shape-bias study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
import uproot
import yaml


HERE = Path(__file__).resolve().parent
PSEUDO65 = HERE.parent
REPO = HERE.parents[3]
QA = HERE / "qa"

SOURCE_ROOT = PSEUDO65 / "inputs" / "pseudo65_background_replacements.root"
OUTPUT_ROOT = HERE / "inputs" / "deterministic_central_means.root"
BASE_CONFIG = (
    PSEUDO65
    / "configs"
    / "config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml"
)
PROVENANCE = HERE / "derived" / "input_provenance.json"
SUMMARY = HERE / "derived" / "summary.json"
COMPARISON = HERE / "derived" / "comparison_55_75MeV.csv"
MEMO = HERE / "MEMO.md"
PLOT_PDF = HERE / "plots" / "functional_mean_shape_bias_Ahat_p0.pdf"
PLOT_PNG = HERE / "plots" / "functional_mean_shape_bias_Ahat_p0.png"

SOURCE_ROOT_SHA256 = (
    "c5ea3922ddb70164f6184a8661d803a6d82302747c0d213c3a37bcab31be11ab"
)
BASE_CONFIG_SHA256 = (
    "74d938bed8372141bafc32addefc17d858bbc7958df9217ff188b98e47de9c76"
)
DRAW_HASHES = {
    "functional_poisson_draw": (
        PSEUDO65 / "derived" / "functional_form_results_reviewed.csv",
        "4ef284c894d8ad6be65fefc0b063cf6100add4d6fe735935e6a90363f7ad7ca1",
    ),
    "gp_poisson_draw": (
        PSEUDO65 / "derived" / "gp_mean_results_reviewed.csv",
        "7ff22bb70d7ee9c0387d20c66b6c20fd359d80af3ff7e51303db607cd88efb77",
    ),
}
EXPECTED_MASSES = np.round(np.arange(0.055, 0.075 + 0.0005, 0.001), 3)
CONFIG_WHITELIST = {"path_2021", "hist_2021", "output_dir"}
ROOT_KEYS = {
    "source": "source/preselection/h_invM_8000",
    "functional": (
        "functional_mean_shape_bias/preselection/h_invM_8000"
    ),
    "gp": "gp_mean_shape_bias/preselection/h_invM_8000",
    "functional_expectation": "expectations/fGenGammaThresh_m065",
    "gp_expectation": "expectations/gp_mean_m065",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_histogram(
    root_file: uproot.ReadOnlyDirectory,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    values, edges = root_file[key].to_numpy()
    return np.asarray(values, float), np.asarray(edges, float)


def validate_inputs() -> dict[str, Any]:
    require(
        sha256_file(SOURCE_ROOT) == SOURCE_ROOT_SHA256,
        "Validated pseudo65 source ROOT checksum changed",
    )
    require(
        sha256_file(BASE_CONFIG) == BASE_CONFIG_SHA256,
        "Canonical v4.2 card checksum changed",
    )
    for path, expected_hash in DRAW_HASHES.values():
        require(
            sha256_file(path) == expected_hash,
            f"Reviewed Poisson-draw comparator changed: {path}",
        )
    require(OUTPUT_ROOT.is_file(), "Deterministic diagnostic ROOT is missing")

    with uproot.open(SOURCE_ROOT) as source_file, uproot.open(
        OUTPUT_ROOT
    ) as output_file:
        source, edges = load_histogram(source_file, ROOT_KEYS["source"])
        original_functional, original_functional_edges = load_histogram(
            source_file,
            ROOT_KEYS["functional_expectation"],
        )
        original_gp, original_gp_edges = load_histogram(
            source_file,
            ROOT_KEYS["gp_expectation"],
        )
        copied_source, copied_edges = load_histogram(
            output_file,
            ROOT_KEYS["source"],
        )
        functional, functional_edges = load_histogram(
            output_file,
            ROOT_KEYS["functional"],
        )
        gp, gp_edges = load_histogram(output_file, ROOT_KEYS["gp"])
        copied_functional, copied_functional_edges = load_histogram(
            output_file,
            ROOT_KEYS["functional_expectation"],
        )
        copied_gp, copied_gp_edges = load_histogram(
            output_file,
            ROOT_KEYS["gp_expectation"],
        )
        metadata_text = str(output_file["metadata/json"])

    all_edges = (
        original_functional_edges,
        original_gp_edges,
        copied_edges,
        functional_edges,
        gp_edges,
        copied_functional_edges,
        copied_gp_edges,
    )
    require(
        all(np.array_equal(edges, item) for item in all_edges),
        "Histogram geometry changed",
    )
    require(np.array_equal(source, copied_source), "Source copy changed")
    require(
        np.array_equal(original_functional, copied_functional)
        and np.array_equal(original_gp, copied_gp),
        "Stored expectation copy changed",
    )
    central = (edges[:-1] >= 0.060 - 1.0e-12) & (
        edges[1:] <= 0.070 + 1.0e-12
    )
    require(int(np.count_nonzero(central)) == 80, "Wrong central-bin count")
    require(
        np.array_equal(functional[~central], source[~central])
        and np.array_equal(gp[~central], source[~central]),
        "Observed outside-window counts changed",
    )
    require(
        np.array_equal(functional[central], original_functional[central])
        and np.array_equal(gp[central], original_gp[central]),
        "Central inputs are not the stored means",
    )
    require(
        np.any(functional[central] != np.rint(functional[central]))
        and np.any(gp[central] != np.rint(gp[central])),
        "Fractional deterministic values were rounded",
    )
    require(
        "not an observed dataset or pseudoexperiment" in metadata_text,
        "ROOT metadata lacks the deterministic interpretation boundary",
    )
    return {
        "output_root": repo_relative(OUTPUT_ROOT),
        "output_root_sha256": sha256_file(OUTPUT_ROOT),
        "native_central_bins": 80,
        "outside_source_bitwise_identical": True,
        "functional_fractional": True,
        "gp_fractional": True,
    }


def validate_configs() -> dict[str, Any]:
    baseline = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    records = {}
    expected_histograms = {
        "functional_mean": ROOT_KEYS["functional"],
        "gp_mean": ROOT_KEYS["gp"],
    }
    for lane, expected_histogram in expected_histograms.items():
        path = HERE / "configs" / f"config_{lane}_shape_bias.yaml"
        require(path.is_file(), f"Missing generated card: {path}")
        candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
        differences = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key) != candidate.get(key)
        }
        require(
            differences == CONFIG_WHITELIST,
            f"{lane}: card differences changed: {sorted(differences)}",
        )
        require(
            candidate["hist_2021"] == expected_histogram,
            f"{lane}: wrong histogram key",
        )
        require(
            candidate["path_2021"] == repo_relative(OUTPUT_ROOT),
            f"{lane}: wrong ROOT path",
        )
        require(candidate["make_ul_bands"] is False, "Limit bands enabled")
        require(candidate["cls_num_toys"] == 0, "CLs toys enabled")
        require(candidate["cls_mode"] == "asymptotic", "Not asymptotic")
        require(candidate["n_restarts"] == 12, "Optimizer card changed")
        require(
            candidate["kernel_ls_res_upper_factor_by_dataset"]["2021"]
            == 15.0,
            "2021 length-scale ceiling changed",
        )
        records[lane] = {
            "path": repo_relative(path),
            "sha256": sha256_file(path),
            "differences_from_canonical": sorted(differences),
        }
    return records


def validate_results() -> dict[str, Any]:
    reviewed = {}
    for lane in ("functional_mean", "gp_mean"):
        path = HERE / "derived" / f"{lane}_results_reviewed.csv"
        require(path.is_file(), f"Missing reviewed scan: {path}")
        frame = pd.read_csv(path)
        require(len(frame) == 21, f"{lane}: expected 21 reviewed rows")
        require(
            np.array_equal(
                np.round(frame["mass_GeV"].to_numpy(float), 3),
                EXPECTED_MASSES,
            ),
            f"{lane}: mass grid is not 55--75 MeV",
        )
        require(
            frame["extract_success"].astype(bool).all(),
            f"{lane}: failed extraction",
        )
        require(
            np.isfinite(
                frame[
                    [
                        "A_hat",
                        "sigma_A",
                        "p0_analytic",
                        "Z_analytic",
                        "lml",
                        "const_opt",
                        "ls_opt",
                    ]
                ].to_numpy(float)
            ).all(),
            f"{lane}: nonfinite reviewed output",
        )
        require(
            (
                frame["selected_state_reproducing_attempt_count"].to_numpy(
                    int
                )
                >= 2
            ).all(),
            f"{lane}: selected state lacks an unchanged-card repeat",
        )
        require(
            not frame["interpolated"].astype(bool).any(),
            f"{lane}: interpolation was used",
        )
        require(
            not frame["selected_at_kernel_bound"].astype(bool).any(),
            f"{lane}: a selected state is at a kernel bound",
        )
        reviewed[lane] = frame

    require(COMPARISON.is_file(), "Comparison CSV is missing")
    comparison = pd.read_csv(COMPARISON)
    require(len(comparison) == 21, "Comparison CSV row count changed")
    require(
        np.array_equal(
            comparison["mass_MeV"].to_numpy(float),
            1000.0 * EXPECTED_MASSES,
        ),
        "Comparison mass grid changed",
    )
    for key, (path, _) in DRAW_HASHES.items():
        reference = pd.read_csv(path)
        reference = reference[
            (reference["dataset"].astype(str) == "2021")
            & reference["mass_GeV"].between(0.055, 0.075)
        ].sort_values("mass_GeV")
        for column in ("A_hat", "sigma_A", "p0_analytic", "Z_analytic"):
            require(
                np.allclose(
                    comparison[f"{key}__{column}"],
                    reference[column],
                    rtol=3.0e-12,
                    atol=1.0e-13,
                ),
                f"{key}: comparison does not reproduce {column}",
            )

    row62 = comparison[
        np.isclose(comparison["mass_MeV"].to_numpy(float), 62.0)
    ].iloc[0]
    preliminary_anchors = {
        "functional_mean_A_hat": 12119.0,
        "functional_mean_sigma_A": 6352.0,
        "functional_mean_Z": 1.909,
        "gp_mean_A_hat": 2750.0,
        "gp_mean_sigma_A": 6383.0,
        "gp_mean_Z": 0.431,
    }
    observed_anchors = {
        "functional_mean_A_hat": float(
            row62["functional_mean_fractional__A_hat"]
        ),
        "functional_mean_sigma_A": float(
            row62["functional_mean_fractional__sigma_A"]
        ),
        "functional_mean_Z": float(
            row62["functional_mean_fractional__Z_analytic"]
        ),
        "gp_mean_A_hat": float(row62["gp_mean_fractional__A_hat"]),
        "gp_mean_sigma_A": float(row62["gp_mean_fractional__sigma_A"]),
        "gp_mean_Z": float(row62["gp_mean_fractional__Z_analytic"]),
    }
    for key, expected in preliminary_anchors.items():
        require(
            np.isclose(
                observed_anchors[key],
                expected,
                rtol=5.0e-3,
                atol=0.02 if key.endswith("_Z") else 2.0,
            ),
            f"Independent 62 MeV anchor disagrees for {key}: "
            f"{observed_anchors[key]} vs {expected}",
        )
    return {
        "reviewed_rows_per_lane": 21,
        "mass_range_MeV": [55, 75],
        "selected_states_reproduced": True,
        "selected_kernel_bound_count": 0,
        "independent_62MeV_anchor_check": observed_anchors,
        "comparison_csv": repo_relative(COMPARISON),
        "comparison_csv_sha256": sha256_file(COMPARISON),
    }


def validate_outputs() -> dict[str, Any]:
    require(SUMMARY.is_file(), "Summary JSON is missing")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    require(summary["status"] == "GENERATED", "Summary is not final")
    forbidden_claims = ("coverage", "expected result", "observed dataset")
    require(
        all(token in summary["interpretation"] for token in forbidden_claims),
        "Summary does not state all interpretation boundaries",
    )
    require(MEMO.is_file(), "Quantitative memo is missing")
    memo = MEMO.read_text(encoding="utf-8")
    for phrase in (
        "fractional deterministic means",
        "not observed datasets",
        "not a coverage statement",
        "additional fluctuation component",
    ):
        require(phrase in memo, f"Memo boundary missing: {phrase}")
    require(PLOT_PDF.is_file() and PLOT_PNG.is_file(), "Plot is missing")
    reader = PdfReader(PLOT_PDF)
    require(len(reader.pages) == 1, "Diagnostic PDF is not one page")
    text = reader.pages[0].extract_text() or ""
    require(
        "Fractional mean curves are deterministic shape diagnostics" in text,
        "Plot does not label fractional means as diagnostics",
    )
    with Image.open(PLOT_PNG) as image:
        require(
            image.width >= 2400 and image.height >= 1800,
            "Diagnostic PNG resolution is too small",
        )
        pixels = [image.width, image.height]
    for record in summary["plots"]:
        path = REPO / record["path"]
        require(path.is_file(), f"Missing summary plot: {path}")
        require(
            sha256_file(path) == record["sha256"],
            f"Summary plot hash changed: {path}",
        )
    require(
        sha256_file(COMPARISON) == summary["comparison_csv"]["sha256"],
        "Summary comparison hash changed",
    )
    require(
        sha256_file(MEMO) == summary["memo"]["sha256"],
        "Summary memo hash changed",
    )
    return {
        "summary": repo_relative(SUMMARY),
        "summary_sha256": sha256_file(SUMMARY),
        "memo": repo_relative(MEMO),
        "memo_sha256": sha256_file(MEMO),
        "plot_pdf": repo_relative(PLOT_PDF),
        "plot_pdf_sha256": sha256_file(PLOT_PDF),
        "plot_png": repo_relative(PLOT_PNG),
        "plot_png_sha256": sha256_file(PLOT_PNG),
        "plot_png_pixels": pixels,
    }


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    report_path = QA / "validation.json"
    try:
        inputs = validate_inputs()
        configs = validate_configs()
        results = validate_results()
        outputs = validate_outputs()
        report = {
            "schema_version": 1,
            "status": "PASS",
            "checks": {
                "frozen_existing_inputs": True,
                "stored_expectations_used_exactly": True,
                "outside_data_bitwise_identical": True,
                "fractional_diagnostic_not_rounded": True,
                "canonical_card_semantics_preserved": True,
                "no_toys_or_expected_bands": True,
                "optimizer_states_reproduced": True,
                "no_selected_kernel_bound": True,
                "reference_draws_hash_locked": True,
                "independent_anchor_agreement": True,
                "memo_interpretation_boundaries": True,
                "plot_rendered_and_labeled": True,
                "output_hashes": True,
            },
            "inputs": inputs,
            "configs": configs,
            "results": results,
            "outputs": outputs,
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
        }
        report_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
