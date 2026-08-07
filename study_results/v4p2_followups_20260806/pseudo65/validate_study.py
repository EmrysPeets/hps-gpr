#!/usr/bin/env python3
"""Validate pseudo65 inputs, scan review products, and presentation figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

import numpy as np
import pandas as pd
import uproot
import yaml


SOURCE_ROOT = Path(
    "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"
)
SOURCE_KEY = "preselection/h_invM_8000"
ROOT_FILE = HERE / "inputs" / "pseudo65_background_replacements.root"
PROVENANCE = HERE / "derived" / "input_provenance.json"
FIT_QC = HERE / "derived" / "functional_fit_qc.json"
PARENT_CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
CONFIGS = {
    "gp_mean": (
        HERE / "configs" / "config_obsUL90_2021_10pct_gpmean_replacement_v4p2.yaml"
    ),
    "functional_form": (
        HERE / "configs" / "config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml"
    ),
}
ROOT_FILE_RELATIVE = (
    "study_results/v4p2_followups_20260806/pseudo65/inputs/"
    "pseudo65_background_replacements.root"
)
ROOT_KEYS = {
    "source": "source/preselection/h_invM_8000",
    "gp_mean": "gp_mean/preselection/h_invM_8000",
    "functional_form": (
        "functional_form_fGenGammaThresh/preselection/h_invM_8000"
    ),
    "gp_expectation": "expectations/gp_mean_m065",
    "functional_expectation": "expectations/fGenGammaThresh_m065",
}


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config is not a mapping: {path}")
    return payload


def assert_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise AssertionError(f"{name}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def validate_inputs() -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text())
    fit_qc = json.loads(FIT_QC.read_text())
    source_values, source_edges = uproot.open(SOURCE_ROOT)[SOURCE_KEY].to_numpy()
    source_values = np.asarray(source_values, float)
    source_edges = np.asarray(source_edges, float)
    centers = 0.5 * (source_edges[:-1] + source_edges[1:])
    replace_mask = (centers >= 0.060) & (centers < 0.070)

    root_file = uproot.open(ROOT_FILE)
    records: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for label, key in ROOT_KEYS.items():
        obj = root_file[key]
        values, edges = obj.to_numpy()
        values = np.asarray(values, float)
        edges = np.asarray(edges, float)
        if obj.classname != "TH1D":
            raise AssertionError(f"{key} is {obj.classname}, expected TH1D")
        if len(values) != 8000 or not np.array_equal(edges, source_edges):
            raise AssertionError(f"{key} does not preserve native geometry")
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{key} contains non-finite values")
        arrays[label] = values
        records[label] = {
            "key": key,
            "class": obj.classname,
            "n_bins": int(len(values)),
            "sum": float(np.sum(values)),
        }

    if not np.array_equal(arrays["source"], source_values):
        raise AssertionError("Source copy in pseudo ROOT differs from original")
    for lane in ("gp_mean", "functional_form"):
        if not np.array_equal(
            arrays[lane][~replace_mask], source_values[~replace_mask]
        ):
            raise AssertionError(f"{lane} changed values outside [60,70) MeV")
        if not np.all(arrays[lane][replace_mask] == np.rint(arrays[lane][replace_mask])):
            raise AssertionError(f"{lane} replacement is not integer-valued")
        if np.any(arrays[lane][replace_mask] < 0.0):
            raise AssertionError(f"{lane} replacement contains negative counts")
    for label in ("gp_expectation", "functional_expectation"):
        if np.any(arrays[label][replace_mask] <= 0.0):
            raise AssertionError(f"{label} has non-positive central expectation")
        if np.any(arrays[label][~replace_mask] != 0.0):
            raise AssertionError(f"{label} must be zero outside [60,70) MeV")

    if sha256_file(ROOT_FILE) != provenance["output"]["root_sha256"]:
        raise AssertionError("Pseudo ROOT SHA256 does not match provenance")
    if sha256_file(SOURCE_ROOT) != provenance["source"]["sha256"]:
        raise AssertionError("Source ROOT SHA256 does not match provenance")
    if provenance["inference_from_ambiguous_request"]["Ainj"] != 0.0:
        raise AssertionError("Study must remain background-only (Ainj=0)")
    if not fit_qc["fit_qc_pass"]:
        raise AssertionError("Functional sideband fit did not pass QC")
    if fit_qc["excluded_interval_GeV"] != [0.06, 0.07]:
        raise AssertionError("Functional fit exclusion is not [60,70) MeV")
    if (
        fit_qc["n_bins_low_sideband"] <= 0
        or fit_qc["n_bins_high_sideband"] <= 0
    ):
        raise AssertionError("Functional fit does not have both sidebands")

    parent = load_yaml(PARENT_CONFIG)
    physics_keys = [
        "range_2021",
        "data_range_2021",
        "sigma_coeffs_2021",
        "frad_coeffs_2021",
        "radiative_penalty_on",
        "radiative_penalty_frac_2021",
        "kernel_constant_init",
        "kernel_constant_bounds",
        "kernel_ls_init",
        "kernel_ls_bounds",
        "kernel_ls_policy",
        "kernel_ls_res_lower_factor",
        "kernel_ls_res_upper_factor",
        "kernel_ls_res_stat",
        "kernel_ls_res_npts",
        "kernel_ls_res_lower_factor_by_dataset",
        "kernel_ls_res_upper_factor_by_dataset",
        "kernel_ls_bounds_by_dataset",
        "kernel_ls_init_by_dataset",
        "kernel_ls_local_hi_floor_mode",
        "kernel_ls_local_hi_floor_factor",
        "pre_log",
        "pre_zero_alpha",
        "alpha_model",
        "pre_alpha_first_n",
        "pre_alpha_first_factor",
        "mass_step_gev",
        "blind_nsigma",
        "gp_train_exclude_nsigma",
        "neighborhood_rebin",
        "n_restarts",
        "scan_require_two_sidebands",
        "scan_edge_guard_nsigma",
        "cls_alpha",
        "cls_mode",
        "cls_num_toys",
        "extract_allow_negative",
        "eps2_density_nsigma",
        "signal_model",
    ]
    config_records = {}
    for lane, path in CONFIGS.items():
        cfg = load_yaml(path)
        stale_temp_values = {
            key: value
            for key, value in cfg.items()
            if isinstance(value, str) and "/private/tmp/" in value
        }
        if stale_temp_values:
            raise AssertionError(
                f"{lane} contains stale temporary absolute paths: "
                f"{stale_temp_values}"
            )
        for key in physics_keys:
            assert_equal(f"{lane}:{key}", cfg.get(key), parent.get(key))
        if not cfg["enable_2021"] or cfg["enable_2015"] or cfg["enable_2016"]:
            raise AssertionError(f"{lane} is not strictly 2021-only")
        if cfg["make_ul_bands"] or cfg["do_combined_bands"]:
            raise AssertionError(f"{lane} unexpectedly enables limit bands")
        if cfg["cls_alpha"] != 0.1 or cfg["cls_mode"] != "asymptotic":
            raise AssertionError(f"{lane} is not asymptotic 90% CLs")
        if cfg["inject_signal"]:
            raise AssertionError(f"{lane} unexpectedly enables signal injection")
        if cfg["do_combined"]:
            raise AssertionError(f"{lane} unexpectedly enables combination")
        if cfg["path_2021"] != ROOT_FILE_RELATIVE:
            raise AssertionError(
                f"{lane} does not use the portable repo-relative ROOT path"
            )
        if Path(cfg["output_dir"]).is_absolute():
            raise AssertionError(f"{lane} output_dir is not repo-relative")
        expected_hist = ROOT_KEYS[lane]
        if cfg["hist_2021"] != expected_hist:
            raise AssertionError(
                f"{lane} hist_2021={cfg['hist_2021']} != {expected_hist}"
            )
        config_records[lane] = {
            "path": repo_relative(path),
            "sha256": sha256_file(path),
            "histogram_key": cfg["hist_2021"],
        }

    payload = {
        "stage": "inputs",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "root_path": repo_relative(ROOT_FILE),
        "root_sha256": sha256_file(ROOT_FILE),
        "n_native_replacement_bins": int(np.count_nonzero(replace_mask)),
        "outside_replacement_exact": True,
        "histograms": records,
        "functional_fit_qc": {
            "pass": True,
            "deviance_per_ndf": fit_qc["poisson_deviance_per_ndf"],
            "pearson_per_ndf": fit_qc["pearson_chi2_per_ndf"],
            "deviance_pvalue": fit_qc["poisson_deviance_pvalue"],
        },
        "configs": config_records,
        "analysis_contract": {
            "Ainj": 0.0,
            "confidence_level": 0.90,
            "calibration": "asymptotic",
            "expected_limit_bands": False,
            "local_p0": True,
            "scan_geometry_nsigma": 2.25,
            "replacement_geometry_complete_bins_GeV": [0.060, 0.070],
        },
    }
    out = HERE / "derived" / "input_validation.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def validate_final() -> dict[str, Any]:
    input_payload = validate_inputs()
    masses_expected = np.round(np.arange(0.050, 0.250 + 0.0005, 0.001), 3)
    csv_records = {}
    m065_records = {}
    for lane in ("gp_mean", "functional_form"):
        path = HERE / "derived" / f"{lane}_results_reviewed.csv"
        frame = pd.read_csv(path)
        if len(frame) != len(masses_expected):
            raise AssertionError(
                f"{lane} has {len(frame)} rows, expected {len(masses_expected)}"
            )
        if not np.array_equal(frame["mass_GeV"].to_numpy(float), masses_expected):
            raise AssertionError(f"{lane} mass grid differs from 50--250 MeV")
        if set(frame["dataset"].astype(str)) != {"2021"}:
            raise AssertionError(f"{lane} includes a non-2021 dataset")
        required_finite = [
            "A_up",
            "eps2_up",
            "p0_analytic",
            "Z_analytic",
            "A_hat",
            "sigma_A",
            "ls_opt",
            "const_opt",
            "lml",
        ]
        if not np.all(np.isfinite(frame[required_finite].to_numpy(float))):
            raise AssertionError(f"{lane} contains non-finite reviewed outputs")
        if not frame["extract_success"].astype(bool).all():
            raise AssertionError(f"{lane} contains failed extraction points")
        if not ((frame["p0_analytic"] >= 0.0) & (frame["p0_analytic"] <= 1.0)).all():
            raise AssertionError(f"{lane} p0 values are outside [0,1]")
        if set(frame["cls_calibration"].astype(str)) != {"asymptotic"}:
            raise AssertionError(f"{lane} contains non-asymptotic outputs")
        if (frame["selected_state_reproducing_attempt_count"] < 2).any():
            bad = frame.loc[
                frame["selected_state_reproducing_attempt_count"] < 2,
                "mass_GeV",
            ].tolist()
            raise AssertionError(f"{lane} has unreproduced selected states: {bad}")
        if frame["review_status"].astype(str).str.contains("pending").any():
            raise AssertionError(f"{lane} contains pending optimizer reviews")
        row65 = frame[np.isclose(frame["mass_GeV"], 0.065)]
        if len(row65) != 1:
            raise AssertionError(f"{lane} does not have exactly one 65 MeV row")
        csv_records[lane] = {
            "path": repo_relative(path),
            "sha256": sha256_file(path),
            "n_rows": int(len(frame)),
            "n_branch_multiplicity_gt1": int(
                np.count_nonzero(frame["branch_multiplicity"].to_numpy(int) > 1)
            ),
        }
        m065_records[lane] = {
            key: float(row65.iloc[0][key])
            for key in (
                "A_hat",
                "sigma_A",
                "A_up",
                "eps2_up",
                "p0_analytic",
                "Z_analytic",
                "integral_density",
            )
        }

    optimizer_audit_path = HERE / "derived" / "optimizer_audit.json"
    optimizer_audit = json.loads(optimizer_audit_path.read_text())
    if not optimizer_audit["pass"]:
        raise AssertionError("Optimizer audit is not closed")
    if optimizer_audit["pending_mass_count_total"] != 0:
        raise AssertionError("Optimizer audit still has pending masses")

    forbidden = list(HERE.glob("**/ul_bands*")) + list(HERE.glob("**/*expected_band*"))
    if forbidden:
        raise AssertionError(f"Unexpected expected-band artifacts: {forbidden}")

    plot_paths = [
        HERE / "plots" / "pseudo65_observed_limit_p0_aligned.png",
        HERE / "plots" / "pseudo65_observed_limit_p0_aligned.pdf",
        HERE / "plots" / "pseudo65_central_window_zoom.png",
        HERE / "plots" / "pseudo65_central_window_zoom.pdf",
    ]
    plot_records = {}
    for path in plot_paths:
        if not path.exists() or path.stat().st_size <= 1000:
            raise AssertionError(f"Missing or empty plot: {path}")
        plot_records[path.name] = {
            "path": repo_relative(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    payload = {
        "stage": "final",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "input_validation_sha256": sha256_file(
            HERE / "derived" / "input_validation.json"
        ),
        "reviewed_scans": csv_records,
        "m065_results": m065_records,
        "optimizer_audit": {
            "path": repo_relative(optimizer_audit_path),
            "sha256": sha256_file(optimizer_audit_path),
            "pending_mass_count_total": 0,
        },
        "plots": plot_records,
        "expected_limit_bands_present": False,
        "interpretation": (
            "Observed/asymptotic conditional replacements only; no expected "
            "sensitivity, coverage, or global-null calibration."
        ),
        "input_contract": input_payload["analysis_contract"],
    }
    out = HERE / "derived" / "final_validation.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inputs", "final"), required=True)
    args = parser.parse_args()
    payload = validate_inputs() if args.stage == "inputs" else validate_final()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
