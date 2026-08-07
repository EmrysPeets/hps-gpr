#!/usr/bin/env python3
"""Validate inputs or final outputs of the GP replacement-window ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PARENT_STUDY = HERE.parent
REPO = HERE.parents[3]

import numpy as np
import pandas as pd
import uproot
import yaml


EXPECTED_SOURCE_SHA256 = (
    "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4"
)
EXPECTED_STATE_LEDGER_SHA256 = (
    "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9"
)
EXPECTED_SELECTED_SOURCE_SHA256 = (
    "fb065cd988534049027c8e3c255b341b97f9ed630b9a27e698ba7452a7f67dcc"
)
SOURCE_ROOT = Path(
    "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"
)
ROOT_FILE = HERE / "inputs" / "gp_window_ensemble.root"
PROVENANCE = HERE / "derived" / "input_provenance.json"
CONFIG_MANIFEST = HERE / "derived" / "config_manifest.json"
TEMPLATE_CONFIG = (
    PARENT_STUDY
    / "configs"
    / "config_obsUL90_2021_10pct_gpmean_replacement_v4p2.yaml"
)
ALLOWED_CONFIG_DIFF = {
    "path_2021",
    "hist_2021",
    "output_dir",
    "scan_n_workers",
}
WINDOWS = ("window_2p25eq2p5", "window_3p0")
REQUIRED_FINITE = (
    "mass_GeV",
    "A_hat",
    "sigma_A",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "lml",
    "const_opt",
    "ls_opt",
    "integral_density",
)
BOUND_COLUMNS = (
    "ls_at_lower",
    "ls_at_upper",
    "const_at_lower",
    "const_at_upper",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.strip().str.lower().eq("true").to_numpy(bool)


def record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any):
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def validate_inputs() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    provenance = json.loads(PROVENANCE.read_text())
    manifest = json.loads(CONFIG_MANIFEST.read_text())
    record(
        checks,
        "external_source_sha256",
        SOURCE_ROOT.exists() and sha256_file(SOURCE_ROOT) == EXPECTED_SOURCE_SHA256,
        str(SOURCE_ROOT),
    )
    record(
        checks,
        "root_sha256",
        sha256_file(ROOT_FILE) == provenance["output"]["root_sha256"],
        provenance["output"]["root_sha256"],
    )
    record(
        checks,
        "generated_config_count",
        manifest["generated_config_count"] == 20,
        manifest["generated_config_count"],
    )
    template = yaml.safe_load(TEMPLATE_CONFIG.read_text())
    config_failures = []
    for item in manifest["records"]:
        path = REPO / item["config"]
        config = yaml.safe_load(path.read_text())
        differences = {
            key
            for key in set(template) | set(config)
            if template.get(key) != config.get(key)
        }
        if differences != ALLOWED_CONFIG_DIFF:
            config_failures.append(
                {"config": item["config"], "differences": sorted(differences)}
            )
        if sha256_file(path) != item["config_sha256"]:
            config_failures.append(
                {"config": item["config"], "error": "sha256 mismatch"}
            )
        expected_bindings = {
            "path_2021": str(ROOT_FILE.resolve().relative_to(REPO.resolve())),
            "hist_2021": item["hist_key"],
            "output_dir": item["output_dir"],
            "scan_n_workers": 5,
        }
        observed_bindings = {
            key: config.get(key) for key in expected_bindings
        }
        if observed_bindings != expected_bindings:
            config_failures.append(
                {
                    "config": item["config"],
                    "error": "manifest/config binding mismatch",
                    "expected": expected_bindings,
                    "observed": observed_bindings,
                }
            )
        if (
            config["cls_mode"] != "asymptotic"
            or int(config["cls_num_toys"]) != 0
            or bool(config["make_ul_bands"])
            or int(config["ul_bands_toys"]) != 0
            or bool(config["do_combined_bands"])
            or int(config["combined_bands_n_toys"]) != 0
            or bool(config["inject_signal"])
        ):
            config_failures.append(
                {"config": item["config"], "error": "forbidden bands/toys/injection"}
            )
    record(
        checks,
        "scan_cards_match_template_except_declared_runtime_io",
        not config_failures,
        config_failures,
    )

    root = uproot.open(ROOT_FILE)
    root_metadata = json.loads(str(root["metadata/json"]))
    metadata_windows = root_metadata["windows"]
    metadata_geometry_ok = (
        metadata_windows["window_2p25eq2p5"]["requested_nsigma"]
        == [2.25, 2.5]
        and metadata_windows["window_3p0"]["requested_nsigma"] == [3.0]
        and metadata_windows["window_2p25eq2p5"][
            "requested_continuous_intervals_GeV"
        ]
        == {
            "2.25": [0.060226182031250006, 0.06977381796875],
            "2.5": [0.0596957578125, 0.0703042421875],
        }
        and metadata_windows["window_3p0"][
            "requested_continuous_intervals_GeV"
        ]
        == {"3.0": [0.058634909375, 0.071365090625]}
        and np.allclose(
            metadata_windows["window_2p25eq2p5"][
                "complete_bin_interval_GeV"
            ],
            [0.060000000000000005, 0.07],
            atol=1.0e-15,
            rtol=0.0,
        )
        and np.allclose(
            metadata_windows["window_3p0"]["complete_bin_interval_GeV"],
            [0.058750000000000004, 0.07125000000000001],
            atol=1.0e-15,
            rtol=0.0,
        )
    )
    record(
        checks,
        "root_metadata_requested_window_geometry",
        metadata_geometry_ok,
        {
            window: {
                "requested_nsigma": metadata_windows[window]["requested_nsigma"],
                "requested_continuous_intervals_GeV": metadata_windows[window][
                    "requested_continuous_intervals_GeV"
                ],
                "complete_bin_interval_GeV": metadata_windows[window][
                    "complete_bin_interval_GeV"
                ],
            }
            for window in WINDOWS
        },
    )
    source, edges = root["source/preselection/h_invM_8000"].to_numpy(flow=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    masks = {
        "window_2p25eq2p5": (centers >= 0.060) & (centers < 0.070),
        "window_3p0": (centers >= 0.05875) & (centers < 0.07125),
    }
    expected_bins = {"window_2p25eq2p5": 80, "window_3p0": 100}
    draw_failures = []
    full_hashes = {window: [] for window in WINDOWS}
    draw_record_map = {
        (item["window"], int(item["draw_index"])): item
        for item in provenance["draws"]
    }
    seed_children = np.random.SeedSequence(
        int(provenance["randomization"]["master_seed"])
    ).spawn(10)
    wide_expectation, expectation_edges = root[
        "expectations/window_3p0/gp_mean_m065"
    ].to_numpy(flow=False)
    if not np.array_equal(expectation_edges, edges):
        draw_failures.append({"error": "wide expectation edges mismatch"})
    for draw_index in range(10):
        values = {}
        for window in WINDOWS:
            key = (
                f"gp/{window}/draw_{draw_index:02d}/"
                "preselection/h_invM_8000"
            )
            array, draw_edges = root[key].to_numpy(flow=False)
            values[window] = array
            mask = masks[window]
            full_hashes[window].append(hashlib.sha256(array.tobytes()).hexdigest())
            if (
                np.count_nonzero(mask) != expected_bins[window]
                or not np.array_equal(draw_edges, edges)
                or not np.array_equal(array[~mask], source[~mask])
                or not np.all(array == np.rint(array))
            ):
                draw_failures.append({"window": window, "draw": draw_index})
            provenance_record = draw_record_map[(window, draw_index)]
            if (
                provenance_record["replacement_draw_sha256"]
                != sha256_array(array[mask])
                or provenance_record["full_histogram_sha256"]
                != sha256_array(array)
                or provenance_record["paired_overlap_sha256"]
                != sha256_array(array[masks["window_2p25eq2p5"]])
            ):
                draw_failures.append(
                    {
                        "window": window,
                        "draw": draw_index,
                        "error": "recorded array hash mismatch",
                    }
                )
        if not np.array_equal(
            values["window_2p25eq2p5"][masks["window_2p25eq2p5"]],
            values["window_3p0"][masks["window_2p25eq2p5"]],
        ):
            draw_failures.append(
                {"draw": draw_index, "error": "paired overlap mismatch"}
            )
        child = seed_children[draw_index]
        rng = np.random.Generator(np.random.PCG64(child))
        replay = rng.poisson(
            wide_expectation[masks["window_3p0"]]
        ).astype(np.int64)
        if not np.array_equal(
            replay, values["window_3p0"][masks["window_3p0"]]
        ):
            draw_failures.append(
                {"draw": draw_index, "error": "SeedSequence/PCG64 replay mismatch"}
            )
        for window in WINDOWS:
            normalized_child_state = json.loads(
                json.dumps(child.state, sort_keys=True)
            )
            if (
                draw_record_map[(window, draw_index)]["child_seed_state"]
                != normalized_child_state
            ):
                draw_failures.append(
                    {
                        "window": window,
                        "draw": draw_index,
                        "error": "recorded child SeedSequence state mismatch",
                    }
                )
    record(checks, "root_draw_geometry_pairing_integer_counts", not draw_failures, draw_failures)
    record(
        checks,
        "ten_distinct_draws_per_geometry",
        all(len(set(hashes)) == 10 for hashes in full_hashes.values()),
        {window: len(set(hashes)) for window, hashes in full_hashes.items()},
    )
    child_states = {
        json.dumps(item["child_seed_state"], sort_keys=True)
        for item in provenance["draws"]
    }
    record(
        checks,
        "ten_independent_recorded_child_streams",
        len(child_states) == 10,
        len(child_states),
    )
    state_hash_ok = (
        provenance["gp_fixed_state"]["state_ledger_sha256"]
        == EXPECTED_STATE_LEDGER_SHA256
    )
    record(
        checks,
        "accepted_fixed_state_ledger_sha256",
        state_hash_ok,
        provenance["gp_fixed_state"]["state_ledger_sha256"],
    )
    selected_source_hash_ok = (
        provenance["gp_fixed_state"]["selected_source_sha256"]
        == EXPECTED_SELECTED_SOURCE_SHA256
    )
    record(
        checks,
        "accepted_selected_source_sha256",
        selected_source_hash_ok,
        provenance["gp_fixed_state"]["selected_source_sha256"],
    )
    record(
        checks,
        "parent_narrow_expectation_reproduced",
        bool(provenance["gp_fixed_state"]["parent_narrow_expectation_reproduced"]),
        provenance["gp_fixed_state"]["reconstructed_lml"],
    )
    return {
        "schema_version": 1,
        "stage": "inputs",
        "checks": checks,
        "pass": all(item["pass"] for item in checks),
    }


def validate_final() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    inputs = validate_inputs()
    record(checks, "input_validation", inputs["pass"], "embedded rerun")
    batch_path = HERE / "derived" / "run_batch_full.json"
    batch = json.loads(batch_path.read_text())
    record(
        checks,
        "full_batch_complete",
        bool(batch["all_complete"]) and int(batch["record_count"]) == 20,
        {"all_complete": batch["all_complete"], "record_count": batch["record_count"]},
    )
    central_batch_path = HERE / "derived" / "run_batch_central_repeat.json"
    central_batch = json.loads(central_batch_path.read_text())
    record(
        checks,
        "central_repeat_batch_complete",
        bool(central_batch["all_complete"])
        and int(central_batch["record_count"]) == 20,
        {
            "all_complete": central_batch["all_complete"],
            "record_count": central_batch["record_count"],
        },
    )
    manifest = json.loads(CONFIG_MANIFEST.read_text())
    expected_mass = np.round(np.arange(0.050, 0.250 + 0.0005, 0.001), 3)
    failures = []
    bound_rows = []
    for item in manifest["records"]:
        path = REPO / item["output_dir"] / "results_single.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["dataset"].astype(str) == "2021"].copy()
        frame = frame.sort_values("mass_GeV").reset_index(drop=True)
        mass = np.round(frame["mass_GeV"].to_numpy(float), 3)
        finite = np.isfinite(frame[list(REQUIRED_FINITE)].to_numpy(float)).all()
        success = bool_values(frame["extract_success"]).all()
        asymptotic = frame["cls_calibration"].astype(str).eq("asymptotic").all()
        bound = np.zeros(len(frame), dtype=bool)
        for column in BOUND_COLUMNS:
            bound |= bool_values(frame[column])
        if (
            len(frame) != 201
            or not np.array_equal(mass, expected_mass)
            or not finite
            or not success
            or not asymptotic
        ):
            failures.append(item["config"])
        if np.any(bound):
            bound_rows.append(
                {
                    "config": item["config"],
                    "masses_MeV": (
                        1000.0 * frame.loc[bound, "mass_GeV"].to_numpy(float)
                    ).tolist(),
                }
            )
    record(checks, "all_20_scans_finite_complete_asymptotic", not failures, failures)
    record(checks, "no_selected_kernel_bounds", not bound_rows, bound_rows)
    central_audit_path = HERE / "derived" / "central_optimizer_audit.json"
    central_audit = json.loads(central_audit_path.read_text())
    record(
        checks,
        "central_55_75_max_lml_review",
        bool(central_audit["pass_finite_complete_no_bounds"])
        and int(central_audit["central_review_row_count"]) == 420
        and int(central_audit["central_unreproduced_selected_state_count"]) == 0
        and int(central_audit["interpolated_row_count"]) == 0,
        {
            "row_count": central_audit["central_review_row_count"],
            "multi_branch": central_audit[
                "central_branch_multiplicity_gt1_count"
            ],
            "unreproduced_selected": central_audit[
                "central_unreproduced_selected_state_count"
            ],
            "selected_bounds": central_audit[
                "central_selected_kernel_bound_count"
            ],
            "full_grid_repeat_stability_established": central_audit[
                "full_grid_repeat_stability_established"
            ],
        },
    )
    reviewed_curves = pd.read_csv(HERE / "derived" / "reviewed_curves.csv")
    record(
        checks,
        "reviewed_curves_complete",
        len(reviewed_curves) == 4020
        and set(reviewed_curves["draw_index"].astype(int)) == set(range(10))
        and not bool_values(reviewed_curves["review_interpolated"]).any(),
        {"row_count": len(reviewed_curves)},
    )

    binding_failures = []
    for mode in ("full", "central_repeat"):
        binding_manifest_path = (
            HERE / "derived" / f"scan_binding_manifest_{mode}.json"
        )
        binding_manifest = json.loads(binding_manifest_path.read_text())
        if (
            not bool(binding_manifest["pass"])
            or int(binding_manifest["binding_count"]) != 20
        ):
            binding_failures.append({"mode": mode, "error": "manifest incomplete"})
            continue
        for binding in binding_manifest["bindings"]:
            path = REPO / binding["binding_path"]
            if (
                not path.exists()
                or sha256_file(path) != binding["binding_sha256"]
            ):
                binding_failures.append(
                    {"mode": mode, "binding_path": binding["binding_path"]}
                )
    record(
        checks,
        "scan_input_result_bindings",
        not binding_failures,
        binding_failures,
    )

    repair_failures = []
    repair_records = []
    expected_repair_counts = {3: 21, 4: 4, 5: 3}
    config_manifest_map = {
        (item["window"], int(item["draw_index"])): item
        for item in manifest["records"]
    }
    root = uproot.open(ROOT_FILE)
    for round_number, expected_count in expected_repair_counts.items():
        repair_batch_path = (
            HERE
            / "derived"
            / f"run_batch_central_repairs_round_{round_number:02d}.json"
        )
        repair_batch = json.loads(repair_batch_path.read_text())
        if (
            not bool(repair_batch["all_complete"])
            or int(repair_batch["job_count"]) != expected_count
            or int(repair_batch["round"]) != round_number
        ):
            repair_failures.append(
                {
                    "round": round_number,
                    "error": "repair batch incomplete or wrong job count",
                    "job_count": repair_batch["job_count"],
                    "all_complete": repair_batch["all_complete"],
                }
            )
        for repair_record in repair_batch["records"]:
            repair_records.append(repair_record)
            identity = (
                repair_record["window"],
                int(repair_record["draw_index"]),
            )
            config_record = config_manifest_map.get(identity)
            config_path = REPO / repair_record["config"]
            output_dir = REPO / repair_record["output_dir"]
            result_path = output_dir / "results_single.csv"
            log_path = output_dir / "scan.log"
            report_path = output_dir / "validation_report.json"
            problems = []
            if (
                config_record is None
                or not config_path.exists()
                or sha256_file(config_path)
                != repair_record["config_sha256"]
                or repair_record["config_sha256"]
                != config_record["config_sha256"]
            ):
                problems.append("config binding/hash mismatch")
            if (
                repair_record["status"] not in ("complete", "skipped_complete")
                or int(repair_record["exit_code"]) != 0
                or int(repair_record["round"]) != round_number
            ):
                problems.append("record status/round mismatch")
            if (
                not result_path.exists()
                or sha256_file(result_path)
                != repair_record["results_single_sha256"]
            ):
                problems.append("result missing/hash mismatch")
            else:
                result = pd.read_csv(result_path)
                result = result[result["dataset"].astype(str) == "2021"]
                if (
                    len(result) != 1
                    or not np.isclose(
                        float(result.iloc[0]["mass_GeV"]),
                        float(repair_record["mass_GeV"]),
                    )
                    or not np.isfinite(
                        result[list(REQUIRED_FINITE)].to_numpy(float)
                    ).all()
                    or not bool_values(result["extract_success"]).all()
                    or not result["cls_calibration"]
                    .astype(str)
                    .eq("asymptotic")
                    .all()
                ):
                    problems.append("result row incomplete/nonfinite/nonasymptotic")
                selected_bound = np.zeros(len(result), dtype=bool)
                for column in BOUND_COLUMNS:
                    selected_bound |= bool_values(result[column])
                if np.any(selected_bound):
                    problems.append("selected kernel bound")
            if not log_path.exists():
                problems.append("scan log missing")
            else:
                log_text = log_path.read_text(errors="replace")
                if "Scan complete!" not in log_text or "Traceback" in log_text:
                    problems.append("scan log incomplete/traceback")
            if not report_path.exists() or config_record is None:
                problems.append("validation report missing")
            else:
                report = json.loads(report_path.read_text()).get("2021", {})
                config = yaml.safe_load(config_path.read_text())
                hist_key = config["hist_2021"]
                root_counts, _ = root[hist_key].to_numpy(flow=False)
                if (
                    not bool(report.get("ok"))
                    or report.get("file") != config["path_2021"]
                    or report.get("hist") != hist_key
                    or not np.isclose(
                        float(report.get("total_counts", np.nan)),
                        float(np.sum(root_counts)),
                        rtol=0.0,
                        atol=0.0,
                    )
                    or repair_record["hist_key"] != hist_key
                    or config_record["hist_key"] != hist_key
                ):
                    problems.append("validation report/ROOT histogram mismatch")
            if problems:
                repair_failures.append(
                    {
                        "round": round_number,
                        "window": repair_record["window"],
                        "draw_index": repair_record["draw_index"],
                        "mass_GeV": repair_record["mass_GeV"],
                        "problems": problems,
                    }
                )
    repair_result_paths = {
        str(Path(item["output_dir"]) / "results_single.csv")
        for item in repair_records
    }
    review_sources = {
        source
        for sources in pd.read_csv(
            HERE / "derived" / "central_optimizer_review.csv"
        )["all_attempt_sources"].astype(str)
        for source in sources.split("|")
    }
    if len(repair_records) != 28 or len(repair_result_paths) != 28:
        repair_failures.append(
            {
                "error": "repair record/path count mismatch",
                "record_count": len(repair_records),
                "unique_result_path_count": len(repair_result_paths),
            }
        )
    missing_from_review = sorted(repair_result_paths - review_sources)
    if missing_from_review:
        repair_failures.append(
            {
                "error": "repair results absent from central optimizer review",
                "paths": missing_from_review,
            }
        )
    record(
        checks,
        "targeted_repair_provenance_and_review_binding",
        not repair_failures,
        {
            "round_job_counts": expected_repair_counts,
            "repair_record_count": len(repair_records),
            "failures": repair_failures,
        },
    )

    qc_path = HERE / "derived" / "scan_qc.csv"
    qc = pd.read_csv(qc_path)
    record(
        checks,
        "scan_qc_20_draws_pass",
        len(qc) == 20
        and bool_values(qc["pass_finite_grid_bound_gates"]).all()
        and int(qc["traceback_count"].sum()) == 0
        and bool_values(qc["scan_complete_marker"]).all(),
        {
            "row_count": len(qc),
            "traceback_count": int(qc["traceback_count"].sum()),
            "convergence_warning_count": int(qc["convergence_warning_count"].sum()),
        },
    )
    summary_path = HERE / "derived" / "ensemble_pointwise_summary.csv"
    summary = pd.read_csv(summary_path)
    record(
        checks,
        "pointwise_summary_complete",
        len(summary) == 402
        and set(summary["draw_count"].astype(int)) == {10}
        and np.isfinite(
            summary.select_dtypes(include=[np.number]).to_numpy(float)
        ).all(),
        {"row_count": len(summary)},
    )
    paired_path = HERE / "derived" / "paired_window_difference_summary.csv"
    paired = pd.read_csv(paired_path)
    record(
        checks,
        "paired_window_summary_complete",
        len(paired) == 201
        and set(paired["paired_draw_count"].astype(int)) == {10}
        and np.isfinite(
            paired.select_dtypes(include=[np.number]).to_numpy(float)
        ).all(),
        {"row_count": len(paired)},
    )
    pilot_path = HERE / "derived" / "pilot_m065_reproduction.csv"
    pilot = pd.read_csv(pilot_path)
    record(
        checks,
        "draw00_m065_pilot_state_reproduced",
        len(pilot) == 2 and bool_values(pilot["state_reproduced"]).all(),
        {
            "row_count": len(pilot),
            "scope": "draw 00 at 65 MeV only",
        },
    )
    plot_manifest_path = HERE / "derived" / "plot_manifest.json"
    plot_manifest = json.loads(plot_manifest_path.read_text())
    plot_failures = []
    for item in plot_manifest["files"]:
        path = REPO / item["path"]
        if (
            not path.exists()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            plot_failures.append(item["path"])
    record(
        checks,
        "plot_artifacts_hash_and_size",
        not plot_failures and len(plot_manifest["files"]) == 4,
        plot_failures,
    )
    record(
        checks,
        "interpretation_flags",
        bool(plot_manifest["no_expected_limit_bands"])
        and bool(plot_manifest["no_cls_calibration_or_limit_band_toys"]),
        {
            "no_expected_limit_bands": plot_manifest["no_expected_limit_bands"],
            "no_cls_calibration_or_limit_band_toys": plot_manifest[
                "no_cls_calibration_or_limit_band_toys"
            ],
        },
    )
    return {
        "schema_version": 1,
        "stage": "final",
        "optimizer_review_limitation": (
            "All 420 rows from 55-75 MeV have a reproduced selected "
            "maximum-finite-LML state after unchanged-card repeats. Outside "
            "that interval each draw has one scan attempt with 12 within-fit "
            "restarts, so full-grid repeat stability is not established."
        ),
        "checks": checks,
        "pass": all(item["pass"] for item in checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inputs", "final"), required=True)
    args = parser.parse_args()
    payload = validate_inputs() if args.stage == "inputs" else validate_final()
    output = HERE / "derived" / f"{args.stage}_validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    print(f"PASS={payload['pass']}")
    for item in payload["checks"]:
        print(f"{'PASS' if item['pass'] else 'FAIL'} {item['name']}: {item['detail']}")
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
