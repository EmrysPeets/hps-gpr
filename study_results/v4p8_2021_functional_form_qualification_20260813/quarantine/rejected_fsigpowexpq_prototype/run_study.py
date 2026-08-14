#!/usr/bin/env python3
"""Run and assemble the v4.8 25-toy fSigPowExpQ truth study.

The runner deliberately separates the pseudoexperiment seed from every GP
optimizer seed.  Optimizer repeats therefore refit identical counts.  Fits are
selected only by GP log marginal likelihood and a predeclared branch-
reproducibility gate; pull size and sign never enter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SPEC_PATH = HERE / "study_spec.json"
INPUTS = HERE / "inputs"
RUNS = HERE / "runs"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
LOGS = HERE / "logs"
QA = HERE / "qa"
TOY_ROOT = INPUTS / "paired_exposure_toys_100.root"

SCENARIOS = (
    "2021_1pct_x10",
    "2021_10pct",
    "2021_1pct_x100",
    "2021_10pct_x10",
)
TRUTH = "sigpowexpq"
FUNCTION_TAG = "fSigPowExpQ"
N_TOYS = 25
BASE_SEED = 20260813


class StudyError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_spec() -> dict[str, Any]:
    payload = load_json(SPEC_PATH)
    if int(payload.get("schema_version", -1)) != 1:
        raise StudyError("unsupported study specification")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(namespace: str, *parts: object) -> int:
    material = "|".join([str(BASE_SEED), namespace, *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "little")


def array_hash(values: Any, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def configure_process() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def build_config() -> Any:
    from hps_gpr.config import load_config

    spec = load_spec()
    cfg = load_config(
        str(HERE / spec["declared_result_state"]["archived_config_path"])
    )
    cfg.enable_2015 = False
    cfg.enable_2016 = False
    cfg.enable_2021 = True
    cfg.do_combined = False
    cfg.make_ul_bands = False
    cfg.ul_bands_toys = 0
    cfg.do_combined_bands = False
    cfg.combined_bands_n_toys = 0
    cfg.make_eps2_bands = False
    cfg.cls_alpha = 0.1
    cfg.cls_mode = "asymptotic"
    cfg.cls_num_toys = 0
    cfg.kernel_ls_res_lower_factor_by_dataset = dict(cfg.kernel_ls_res_lower_factor_by_dataset)
    cfg.kernel_ls_res_upper_factor_by_dataset = dict(cfg.kernel_ls_res_upper_factor_by_dataset)
    cfg.kernel_ls_res_lower_factor_by_dataset["2021"] = 1.1
    cfg.kernel_ls_res_upper_factor_by_dataset["2021"] = 15.0
    cfg.blind_nsigma = 2.25
    cfg.gp_train_exclude_nsigma = 2.25
    cfg.scan_edge_guard_nsigma = 2.25
    cfg.scan_require_two_sidebands = True
    cfg.neighborhood_rebin = 5
    cfg.n_restarts = 12
    cfg.extract_allow_negative = True
    cfg.extract_background_mode = "profiled"
    cfg.eps2_density_nsigma = 1.64
    cfg.signal_model = "default"
    cfg.fail_fast = True
    cfg.debug_print = False
    cfg.save_plots = False
    return cfg


def assert_config(cfg: Any) -> None:
    checks = {
        "range": tuple(map(float, cfg.range_2021)) == (0.05, 0.25),
        "support": tuple(map(float, cfg.data_range_2021)) == (0.04, 0.30),
        "lower": float(cfg.kernel_ls_res_lower_factor_by_dataset["2021"]) == 1.1,
        "upper": float(cfg.kernel_ls_res_upper_factor_by_dataset["2021"]) == 15.0,
        "blind": float(cfg.blind_nsigma) == 2.25,
        "training": float(cfg.gp_train_exclude_nsigma) == 2.25,
        "edge": float(cfg.scan_edge_guard_nsigma) == 2.25,
        "sidebands": bool(cfg.scan_require_two_sidebands),
        "rebin": int(cfg.neighborhood_rebin) == 5,
        "restarts": int(cfg.n_restarts) == 12,
        "density": float(cfg.eps2_density_nsigma) == 1.64,
        "pre_log": bool(cfg.pre_log),
        "alpha_model": str(cfg.alpha_model) == "1/y",
        "cls_90": float(cfg.cls_alpha) == 0.1,
        "cls_asymptotic": str(cfg.cls_mode) == "asymptotic"
        and int(cfg.cls_num_toys) == 0,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise StudyError("frozen-card assertion failed: " + ", ".join(failed))


def preflight() -> dict[str, Any]:
    spec = load_spec()
    checks: dict[str, bool] = {}
    checks["toy_root"] = TOY_ROOT.is_file() and sha256_file(TOY_ROOT) == spec["background_toy_product"]["root_sha256"]
    manifest = HERE / spec["background_toy_product"]["manifest"]
    checks["toy_manifest"] = manifest.is_file() and sha256_file(manifest) == spec["background_toy_product"]["manifest_sha256"]
    config_path = HERE / spec["declared_result_state"]["archived_config_path"]
    checks["config"] = config_path.is_file() and sha256_file(config_path) == spec["declared_result_state"]["config_sha256"]
    for family, record in spec["source_inputs"].items():
        checks[f"source_{family}"] = sha256_file(HERE / record["root"]) == record["root_sha256"]
        checks[f"metadata_{family}"] = sha256_file(HERE / record["metadata"]) == record["metadata_sha256"]
    cfg = build_config()
    assert_config(cfg)
    checks["card_assertions"] = True
    if not all(checks.values()):
        raise StudyError("preflight failed: " + ", ".join(key for key, value in checks.items() if not value))
    return {"status": "pass", "checks": checks, "validated_utc": utc_now()}


def make_toy_dataset(scenario: str, toy_index: int, cfg: Any) -> Any:
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import FuncFormToySpec, build_funcform_toy_dataset, load_funcform_toy_hist

    base = make_datasets(cfg)["2021"]
    toy_name = f"toy_{int(toy_index):04d}"
    toy_spec = FuncFormToySpec(
        source_root=str(TOY_ROOT),
        container=f"toys/{TRUTH}/{scenario}",
        function_tag=FUNCTION_TAG,
        toy_name=toy_name,
        toy_index=int(toy_index),
    )
    histogram = load_funcform_toy_hist(str(TOY_ROOT), container=toy_spec.container, toy_name=toy_name)
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def covariance_diagnostics(covariance: Any) -> dict[str, Any]:
    matrix = np.asarray(covariance, dtype=float)
    finite = bool(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and np.isfinite(matrix).all())
    if not finite or matrix.size == 0:
        return {"covariance_valid": False, "covariance_min_eigenvalue": float("nan"), "covariance_min_eigenvalue_relative": float("nan")}
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.diag(symmetric))), 1.0)
    minimum = float(np.min(eigenvalues))
    relative = minimum / scale
    return {
        # The lognormal moment transform can leave a small negative numerical
        # mode even when the downstream profiled fit's established PSD repair
        # succeeds.  Reject material indefiniteness, not that known rounding
        # residue.
        "covariance_valid": bool(np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8 * scale) and relative >= -1e-2),
        "covariance_min_eigenvalue": minimum,
        "covariance_min_eigenvalue_relative": relative,
    }


def branch_match(first: dict[str, Any], second: dict[str, Any], gate: dict[str, Any]) -> bool:
    required = ("gp_lml", "gp_ls", "gp_const", "sigma_A", "n_train")
    if not all(np.isfinite(float(first.get(key, np.nan))) and np.isfinite(float(second.get(key, np.nan))) for key in required):
        return False
    n_train = max(1.0, min(float(first["n_train"]), float(second["n_train"])))
    if abs(float(first["gp_lml"]) - float(second["gp_lml"])) / n_train > float(gate["delta_lml_per_train_max"]):
        return False
    for key, limit in (
        ("gp_ls", gate["abs_log_length_ratio_max"]),
        ("gp_const", gate["abs_log_constant_ratio_max"]),
        ("sigma_A", gate["abs_log_sigma_ratio_max"]),
    ):
        left, right = float(first[key]), float(second[key])
        if left <= 0 or right <= 0 or abs(math.log(left / right)) > float(limit):
            return False
    return True


def select_branch(records: list[dict[str, Any]], gate: dict[str, Any], *, require_replication: bool) -> tuple[dict[str, Any] | None, int]:
    usable = [
        row for row in records
        if bool(row.get("fit_ok")) and bool(row.get("covariance_valid"))
        and np.isfinite(float(row.get("gp_lml", np.nan)))
        and np.isfinite(float(row.get("sigma_A", np.nan))) and float(row.get("sigma_A", 0.0)) > 0
    ]
    if not usable:
        return None, 0
    selected = max(usable, key=lambda row: float(row["gp_lml"]))
    replicates = sum(branch_match(selected, candidate, gate) for candidate in usable)
    if require_replication and replicates < int(gate["top_branch_min_replicates"]):
        return None, replicates
    return selected, replicates


def reference_attempt(ds: Any, cfg: Any, scenario: str, toy_index: int, mass: float, attempt: int) -> tuple[dict[str, Any], Any | None]:
    from hps_gpr.conversion import A_from_epsilon2
    from hps_gpr.injection import _fit_A_for_extraction, _prediction_blind_mask, _prediction_y_full_bonly, _sigmaA_reference
    from hps_gpr.io import estimate_background_for_dataset

    optimizer_seed = stable_seed("v4p8_restart_v1", scenario, toy_index, f"{mass:.9f}", "reference", attempt)
    cfg.gp_optimizer_random_state = int(optimizer_seed)
    base = {
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "inj_nsigma": 0.0,
        "strength": 0.0,
        "role": "reference_bonly",
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "optimizer_restarts": 12,
        "fit_ok": False,
        "refit_fallback_used": False,
        "error": "",
    }
    try:
        pred = estimate_background_for_dataset(ds, float(mass), cfg, restarts=12, optimize=True)
        mask = _prediction_blind_mask(pred)
        obs = _prediction_y_full_bonly(pred)[mask]
        fit = _fit_A_for_extraction(cfg, obs, pred.mu, pred.cov, __import__("hps_gpr.template", fromlist=["build_window_template_from_full"]).build_window_template_from_full(pred.edges_full, mask, float(mass), pred.sigma_val, config=cfg)[0], allow_negative=True)
        sigma_A = float(fit["sigma_A"])
        sigma_ref = float(_sigmaA_reference(pred, float(mass), source="asimov", config=cfg))
        density = float(pred.integral_density)
        record = {
            **base,
            "fit_ok": bool(fit.get("success", False)),
            "gp_lml": float(pred.lml),
            "gp_ls": float(pred.ls_opt),
            "gp_const": float(pred.const_opt),
            "gp_ls_lo": float(pred.ls_lo),
            "gp_ls_hi": float(pred.ls_hi),
            "gp_ls_init": float(pred.ls_init),
            "gp_const_init": float(pred.const_init),
            "sigma_A": sigma_A,
            "sigmaA_reference": sigma_ref,
            "A_hat": float(fit["A_hat"]),
            "Zhat": float(fit["A_hat"]) / sigma_A,
            "pull": float(fit["A_hat"]) / sigma_A,
            "amplitude_nll": float(fit.get("nll", np.nan)),
            "n_train": int(pred.n_train),
            "n_blind": int(pred.n_blind),
            "integral_density": density,
            "A_per_eps2_unit": float(A_from_epsilon2(ds, float(mass), 1.0, density)),
            "optimizer_warning_count": int(pred.optimizer_warning_count),
            "optimizer_warnings": str(pred.optimizer_warnings),
            **covariance_diagnostics(pred.cov),
        }
        return record, pred
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:500]}, None


def refit_attempt(
    ds: Any,
    cfg: Any,
    reference_pred: Any,
    reference_row: dict[str, Any],
    scenario: str,
    toy_index: int,
    mass: float,
    z: float,
    attempt: int,
) -> dict[str, Any]:
    from hps_gpr.gpr import fit_gpr, make_kernel_for_dataset, predict_counts_from_log_gpr
    from hps_gpr.injection import _fit_A_for_extraction, _fixed_hist_background_counts, _gpr_fit_diagnostics, _inject_counts_from_template, _prediction_blind_mask, _prediction_y_full_bonly
    from hps_gpr.template import build_window_template_from_full

    optimizer_seed = stable_seed("v4p8_restart_v1", scenario, toy_index, f"{mass:.9f}", f"z{z:.1f}", attempt)
    signal_seed = stable_seed("v4p8_signal_v1", scenario, toy_index, f"{mass:.9f}", f"z{z:.1f}")
    sigma_ref = float(reference_row["sigma_A"])
    A_injected = float(z) * sigma_ref
    base = {
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "inj_nsigma": float(z),
        "role": "injected_refit",
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "signal_seed": int(signal_seed),
        "optimizer_restarts": 12,
        "fit_ok": False,
        "refit_fallback_used": False,
        "error": "",
        "strength": A_injected,
        "sigmaA_reference": sigma_ref,
        "reference_attempt_selected": int(reference_row["attempt"]),
        "reference_gp_lml": float(reference_row["gp_lml"]),
        "reference_gp_ls": float(reference_row["gp_ls"]),
        "reference_gp_const": float(reference_row["gp_const"]),
        "A_per_eps2_unit": float(reference_row["A_per_eps2_unit"]),
        "integral_density": float(reference_row["integral_density"]),
    }
    try:
        blind_mask = _prediction_blind_mask(reference_pred)
        x_full = np.asarray(reference_pred.x_full, dtype=float)
        y_background = _fixed_hist_background_counts(
            _prediction_y_full_bonly(reference_pred), dataset_key="2021", mass=float(mass)
        )
        template_window, template_full = build_window_template_from_full(
            reference_pred.edges_full, blind_mask, float(mass), reference_pred.sigma_val, config=cfg
        )
        rng = np.random.default_rng(int(signal_seed))
        signal_full, n_signal_full, _ = _inject_counts_from_template(template_full, A_injected, rng, "poisson")
        signal_full = np.asarray(signal_full, dtype=int)
        y_toy = np.asarray(y_background, dtype=int) + signal_full
        train_half_width = 2.25 * float(reference_pred.sigma_val)
        train_mask = (x_full < float(mass) - train_half_width) | (x_full > float(mass) + train_half_width)
        kernel = make_kernel_for_dataset(ds, cfg, mass=float(mass))
        gpr = fit_gpr(
            x_full[train_mask], y_toy[train_mask].astype(float), cfg,
            restarts=12, kernel=kernel, optimize=True, random_state=int(optimizer_seed),
        )
        mu, covariance = predict_counts_from_log_gpr(gpr, x_full[blind_mask], cfg)
        fit = _fit_A_for_extraction(
            cfg, y_toy[blind_mask], mu, covariance, template_window, allow_negative=True
        )
        diag = _gpr_fit_diagnostics(gpr)
        sigma_A = float(fit["sigma_A"])
        A_hat = float(fit["A_hat"])
        kernel_initial = getattr(gpr, "kernel", None)
        initial_const = float(getattr(getattr(kernel_initial, "k1", None), "constant_value", np.nan))
        initial_ls = float(getattr(getattr(kernel_initial, "k2", None), "length_scale", np.nan))
        record = {
            **base,
            "fit_ok": bool(fit.get("success", False)),
            "gp_lml": float(gpr.log_marginal_likelihood_value_),
            "gp_ls": float(diag["ls_opt"]),
            "gp_const": float(diag["const_opt"]),
            "gp_ls_lo": float(reference_row["gp_ls_lo"]),
            "gp_ls_hi": float(reference_row["gp_ls_hi"]),
            "gp_ls_init": initial_ls,
            "gp_const_init": initial_const,
            "sigma_A": sigma_A,
            "A_hat": A_hat,
            "Zhat": A_hat / sigma_A,
            "pull": (A_hat - A_injected) / sigma_A,
            "amplitude_nll": float(fit.get("nll", np.nan)),
            "n_train": int(np.count_nonzero(train_mask)),
            "n_blind": int(np.count_nonzero(blind_mask)),
            "Nsig_full": int(n_signal_full),
            "Nsig_win": int(np.sum(signal_full[blind_mask])),
            "Nsig_train": int(np.sum(signal_full[train_mask])),
            "signal_counts_sha256": array_hash(signal_full, "<i8"),
            "optimizer_warning_count": len(getattr(gpr, "_hps_optimizer_warnings", ())),
            "optimizer_warnings": " | ".join(getattr(gpr, "_hps_optimizer_warnings", ())),
            **covariance_diagnostics(covariance),
        }
        return record
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:500]}


def refit_triggers(row: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(row.get("fit_ok")) or not bool(row.get("covariance_valid")):
        reasons.append("invalid_or_nonfinite")
        return reasons
    ls, const = float(row["gp_ls"]), float(row["gp_const"])
    ls_init, const_init = float(row["gp_ls_init"]), float(row["gp_const_init"])
    exact_tol = float(gate["exact_start_abs_log_theta_max"])
    if all(value > 0 for value in (ls, const, ls_init, const_init)):
        if max(abs(math.log(ls / ls_init)), abs(math.log(const / const_init))) < exact_tol:
            reasons.append("exact_start_signature")
    lo, hi = float(row["gp_ls_lo"]), float(row["gp_ls_hi"])
    window = float(gate["bound_ratio_window"])
    if ls > 0 and lo > 0 and ls / lo <= 1.0 + window:
        reasons.append("near_lower_length_bound")
    if ls > 0 and hi > 0 and ls / hi >= 1.0 - window:
        reasons.append("near_upper_length_bound")
    ratio = float(row["sigma_A"]) / float(row["sigmaA_reference"])
    low, high = map(float, gate["sigma_over_reference_trigger"])
    if not np.isfinite(ratio) or ratio < low or ratio > high:
        reasons.append("sigma_reference_ratio")
    return reasons


def accepted_row(
    selected: dict[str, Any], reference_row: dict[str, Any], scenario: str,
    toy_index: int, mass: float, z: float, attempts: int, replicates: int,
    gate_status: str, trigger_reasons: Iterable[str], spec: dict[str, Any],
) -> dict[str, Any]:
    row = dict(selected)
    row.update({
        "study_id": spec["study_id"],
        "scenario": scenario,
        "scenario_label": spec["scenarios"][scenario]["label"],
        "source_family": spec["scenarios"][scenario]["source_family"],
        "truth_model": TRUTH,
        "truth_function_tag": FUNCTION_TAG,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "mass_MeV": 1000.0 * float(mass),
        "inj_nsigma": float(z),
        "n_attempts": int(attempts),
        "top_branch_replicates": int(replicates),
        "optimizer_gate_status": gate_status,
        "optimizer_trigger_reasons": ";".join(trigger_reasons),
        "accepted": True,
        "sigmaA_ref": float(reference_row["sigma_A"]),
        "sigmaA_ref_mode": "matched_refit_bonly_multistart_v1",
        "reference_top_branch_replicates": int(reference_row["top_branch_replicates"]),
        "eps2_hat_signed": float(selected["A_hat"]) / float(reference_row["A_per_eps2_unit"]),
        "eps2_injected": float(selected.get("strength", 0.0)) / float(reference_row["A_per_eps2_unit"]),
        "eps2_sigma": float(selected["sigma_A"]) / float(reference_row["A_per_eps2_unit"]),
        "delta_z": float(selected["Zhat"]) - float(z),
        "refit_ls_over_hi": float(selected["gp_ls"]) / float(selected["gp_ls_hi"]),
        "refit_upper_boundary": float(selected["gp_ls"]) / float(selected["gp_ls_hi"]) >= 0.999,
        "analysis_partition": "development_25_toys_0_24",
        "declared_result_commit": spec["declared_result_state"]["result_commit"],
        "declared_integration_commit": spec["declared_result_state"]["integration_commit"],
    })
    return row


def task_directory(scenario: str, toy_index: int) -> Path:
    return RUNS / scenario / f"toy_{int(toy_index):04d}"


def successful_task(scenario: str, toy_index: int) -> Path | None:
    directory = task_directory(scenario, toy_index)
    marker = directory / "_SUCCESS.json"
    result = directory / "accepted_rows.csv"
    if not marker.is_file() or not result.is_file():
        return None
    payload = load_json(marker)
    return result if payload.get("accepted_sha256") == sha256_file(result) else None


def run_task(scenario: str, toy_index: int, force: bool = False) -> dict[str, Any]:
    configure_process()
    spec = load_spec()
    gate = spec["optimizer_gate"]
    if scenario not in SCENARIOS or not 0 <= int(toy_index) < N_TOYS:
        raise StudyError("invalid scenario or toy index")
    preflight()
    existing = successful_task(scenario, toy_index)
    if existing is not None and not force:
        return {"status": "already_complete", "scenario": scenario, "toy_index": int(toy_index)}
    final_dir = task_directory(scenario, toy_index)
    if final_dir.exists():
        if not force:
            raise StudyError(f"incomplete task exists: {final_dir}")
        archived = final_dir.with_name(final_dir.name + ".superseded_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
        os.replace(final_dir, archived)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=final_dir.parent))
    cfg = build_config()
    assert_config(cfg)
    ds = make_toy_dataset(scenario, int(toy_index), cfg)
    attempt_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    try:
        for mass in map(float, spec["masses_gev"]):
            reference_records: list[dict[str, Any]] = []
            reference_predictions: dict[int, Any] = {}
            for attempt in range(int(gate["reference_initial_attempts"])):
                record, pred = reference_attempt(ds, cfg, scenario, int(toy_index), mass, attempt)
                reference_records.append(record)
                if pred is not None:
                    reference_predictions[int(attempt)] = pred
            selected_reference, reference_replicates = select_branch(reference_records, gate, require_replication=True)
            if selected_reference is None:
                for attempt in range(int(gate["reference_initial_attempts"]), int(gate["maximum_attempts"])):
                    record, pred = reference_attempt(ds, cfg, scenario, int(toy_index), mass, attempt)
                    reference_records.append(record)
                    if pred is not None:
                        reference_predictions[int(attempt)] = pred
                selected_reference, reference_replicates = select_branch(reference_records, gate, require_replication=True)
            attempt_rows.extend(reference_records)
            if selected_reference is None:
                for z in map(float, spec["sigma_strengths"]):
                    placeholder = dict(reference_records[0]) if reference_records else {}
                    placeholder.update({
                        "scenario": scenario,
                        "background_toy_index": int(toy_index),
                        "mass_GeV": mass,
                        "inj_nsigma": z,
                        "strength": float("nan"),
                        "A_hat": float("nan"),
                        "sigma_A": float("nan"),
                        "Zhat": float("nan"),
                        "pull": float("nan"),
                        "role": "raw_reference_invalid_placeholder",
                        "accepted": False,
                        "optimizer_gate_status": "exclude_irreproducible_reference",
                    })
                    raw_rows.append(placeholder)
                    exclusions.append({
                        "scenario": scenario, "background_toy_index": int(toy_index),
                        "mass_GeV": mass, "inj_nsigma": z,
                        "exclusion_scope": "scenario_toy_mass_all_strengths",
                        "reason": "irreproducible_background_reference_top_branch",
                        "n_attempts": len(reference_records),
                    })
                continue
            raw_rows.append(dict(reference_records[0]))
            selected_reference = dict(selected_reference)
            selected_reference["top_branch_replicates"] = int(reference_replicates)
            reference_pred = reference_predictions[int(selected_reference["attempt"])]
            ref_gate = "pass_replicated_initial3" if len(reference_records) == 3 else "pass_replicated_after5"
            accepted_rows.append(accepted_row(
                selected_reference, selected_reference, scenario, int(toy_index), mass, 0.0,
                len(reference_records), reference_replicates, ref_gate, (), spec,
            ))
            for z in (1.0, 3.0, 5.0):
                records = [refit_attempt(ds, cfg, reference_pred, selected_reference, scenario, int(toy_index), mass, z, 0)]
                trigger_reasons = refit_triggers(records[0], gate)
                if trigger_reasons:
                    records.extend(
                        refit_attempt(ds, cfg, reference_pred, selected_reference, scenario, int(toy_index), mass, z, attempt)
                        for attempt in (1, 2)
                    )
                    selected, replicates = select_branch(records, gate, require_replication=True)
                    if selected is None:
                        records.extend(
                            refit_attempt(ds, cfg, reference_pred, selected_reference, scenario, int(toy_index), mass, z, attempt)
                            for attempt in (3, 4)
                        )
                        selected, replicates = select_branch(records, gate, require_replication=True)
                    gate_status = "pass_trigger_replicated_after3" if len(records) == 3 else "pass_trigger_replicated_after5"
                else:
                    selected, replicates = select_branch(records, gate, require_replication=False)
                    gate_status = "pass_single_untriggered"
                attempt_rows.extend(records)
                raw_rows.append(dict(records[0]))
                if selected is None:
                    exclusions.append({
                        "scenario": scenario, "background_toy_index": int(toy_index),
                        "mass_GeV": mass, "inj_nsigma": z,
                        "exclusion_scope": "single_injected_fit_row",
                        "reason": "irreproducible_injected_refit_top_branch",
                        "n_attempts": len(records),
                        "trigger_reasons": ";".join(trigger_reasons),
                    })
                    continue
                accepted_rows.append(accepted_row(
                    selected, selected_reference, scenario, int(toy_index), mass, z,
                    len(records), replicates, gate_status, trigger_reasons, spec,
                ))

        attempts_df = pd.DataFrame(attempt_rows)
        accepted_df = pd.DataFrame(accepted_rows)
        raw_df = pd.DataFrame(raw_rows)
        exclusions_df = pd.DataFrame(exclusions)
        if len(raw_df) != 20:
            raise StudyError(f"raw-primary cardinality is {len(raw_df)}, expected 20")
        for name, frame in (
            ("optimizer_attempts.csv", attempts_df),
            ("accepted_rows.csv", accepted_df),
            ("raw_primary_rows.csv", raw_df),
            ("exclusions.csv", exclusions_df),
        ):
            frame.to_csv(work_dir / name, index=False)
        marker = {
            "status": "pass",
            "completed_utc": utc_now(),
            "scenario": scenario,
            "toy_index": int(toy_index),
            "attempt_rows": len(attempts_df),
            "accepted_rows": len(accepted_df),
            "raw_primary_rows": len(raw_df),
            "excluded_rows": len(exclusions_df),
            "accepted_sha256": sha256_file(work_dir / "accepted_rows.csv"),
            "attempts_sha256": sha256_file(work_dir / "optimizer_attempts.csv"),
            "raw_sha256": sha256_file(work_dir / "raw_primary_rows.csv"),
            "exclusions_sha256": sha256_file(work_dir / "exclusions.csv"),
        }
        atomic_json(work_dir / "_SUCCESS.json", marker)
        os.replace(work_dir, final_dir)
        return marker
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def run_task_subprocess(scenario: str, toy_index: int, force: bool) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "run-task", scenario, str(int(toy_index))]
    if force:
        command.append("--force")
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    result = subprocess.run(command, cwd=REPO, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise StudyError(f"task {scenario} toy {toy_index} failed:\n{result.stdout}")
    return {"scenario": scenario, "toy_index": int(toy_index), "output": result.stdout}


def run_many(toy_start: int, toy_stop: int, workers: int, force: bool) -> dict[str, Any]:
    preflight()
    if not 0 <= toy_start < toy_stop <= N_TOYS:
        raise StudyError(
            f"toy interval must satisfy 0 <= start < stop <= {N_TOYS}"
        )
    if not 1 <= int(workers) <= 2:
        raise StudyError("CPU-conscious production permits one or two outer workers")
    tasks = [(scenario, toy_index) for scenario in SCENARIOS for toy_index in range(toy_start, toy_stop)]
    completed = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(run_task_subprocess, scenario, toy_index, force): (scenario, toy_index) for scenario, toy_index in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"PASS {result['scenario']} toy {result['toy_index']:04d}", flush=True)
    return {"status": "pass", "tasks": len(tasks), "completed": len(completed), "toy_start": toy_start, "toy_stop": toy_stop, "workers": int(workers)}


def collect() -> dict[str, Any]:
    from scipy.stats import chi2, median_abs_deviation, t, ttest_1samp

    accepted_frames, raw_frames, attempt_frames, exclusion_frames = [], [], [], []
    for scenario in SCENARIOS:
        for toy_index in range(N_TOYS):
            if successful_task(scenario, toy_index) is None:
                raise StudyError(f"missing completed task {scenario} toy {toy_index:04d}")
            directory = task_directory(scenario, toy_index)
            accepted_frames.append(pd.read_csv(directory / "accepted_rows.csv"))
            raw_frames.append(pd.read_csv(directory / "raw_primary_rows.csv"))
            attempt_frames.append(pd.read_csv(directory / "optimizer_attempts.csv"))
            exclusion_path = directory / "exclusions.csv"
            if exclusion_path.stat().st_size > 1:
                try:
                    exclusion_frames.append(pd.read_csv(exclusion_path))
                except pd.errors.EmptyDataError:
                    pass
    accepted = pd.concat(accepted_frames, ignore_index=True, sort=False)
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    attempts = pd.concat(attempt_frames, ignore_index=True, sort=False)
    exclusions = pd.concat(exclusion_frames, ignore_index=True, sort=False) if exclusion_frames else pd.DataFrame(columns=["scenario", "background_toy_index", "mass_GeV", "inj_nsigma", "reason"])
    key = ["scenario", "background_toy_index", "mass_GeV", "inj_nsigma"]
    accepted = accepted.sort_values(key).reset_index(drop=True)
    raw = raw.sort_values(key).reset_index(drop=True)
    expected_rows = len(SCENARIOS) * N_TOYS * len(load_spec()["masses_gev"]) * len(load_spec()["sigma_strengths"])
    if len(raw) != expected_rows or raw.duplicated(key).any():
        raise StudyError(
            f"raw-primary ledger must contain {expected_rows} unique states"
        )
    if accepted.duplicated(key).any():
        raise StudyError("accepted ledger has duplicate states")
    DERIVED.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(DERIVED / "accepted_extraction_rows.csv", index=False)
    raw.to_csv(DERIVED / "raw_primary_extraction_rows.csv", index=False)
    attempts.to_csv(DERIVED / "optimizer_attempt_ledger.csv", index=False)
    exclusions.to_csv(DERIVED / "exclusion_ledger.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for (scenario, mass, z), raw_group in raw.groupby(["scenario", "mass_GeV", "inj_nsigma"], sort=True):
        accepted_group = accepted[(accepted.scenario == scenario) & np.isclose(accepted.mass_GeV, mass) & np.isclose(accepted.inj_nsigma, z)]
        def moments(group: pd.DataFrame, prefix: str) -> dict[str, Any]:
            values = pd.to_numeric(group["pull"], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            width = float(np.std(values, ddof=1)) if n > 1 else np.nan
            tcrit = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
            chi_lo = float(chi2.ppf(0.025, n - 1)) if n > 1 else np.nan
            chi_hi = float(chi2.ppf(0.975, n - 1)) if n > 1 else np.nan
            loo = []
            if n > 1:
                loo = [abs(float(np.mean(np.delete(values, index))) - mean) for index in range(n)]
            return {
                f"{prefix}_n": n,
                f"{prefix}_pull_mean": mean,
                f"{prefix}_pull_width": width,
                f"{prefix}_pull_median": float(np.median(values)) if n else np.nan,
                f"{prefix}_pull_mad_scaled": float(median_abs_deviation(values, scale="normal")) if n else np.nan,
                f"{prefix}_pull_trimmed_mean_10pct": float(__import__("scipy.stats", fromlist=["trim_mean"]).trim_mean(values, 0.1)) if n else np.nan,
                f"{prefix}_pull_mean_ci95_low": mean - tcrit * width / math.sqrt(n) if n > 1 else np.nan,
                f"{prefix}_pull_mean_ci95_high": mean + tcrit * width / math.sqrt(n) if n > 1 else np.nan,
                f"{prefix}_pull_width_ci95_low": math.sqrt((n - 1) * width * width / chi_hi) if n > 1 else np.nan,
                f"{prefix}_pull_width_ci95_high": math.sqrt((n - 1) * width * width / chi_lo) if n > 1 else np.nan,
                f"{prefix}_max_leave_one_out_mean_change": max(loo) if loo else np.nan,
            }
        summary = {
            "scenario": scenario,
            "mass_GeV": float(mass),
            "mass_MeV": float(mass) * 1000.0,
            "inj_nsigma": float(z),
            "n_generated": N_TOYS,
            **moments(raw_group, "raw"),
            **moments(accepted_group, "accepted"),
            "n_excluded": N_TOYS - len(accepted_group),
            "development_cell_sample_size_eligible": len(accepted_group) >= N_TOYS - 1,
            "accepted_delta_z_mean": float(pd.to_numeric(accepted_group["delta_z"]).mean()) if len(accepted_group) else np.nan,
            "accepted_median_recovery": float(np.median(pd.to_numeric(accepted_group["A_hat"]) / pd.to_numeric(accepted_group["strength"]))) if float(z) > 0 and len(accepted_group) else np.nan,
            "accepted_recovery_q16": float(np.quantile(pd.to_numeric(accepted_group["A_hat"]) / pd.to_numeric(accepted_group["strength"]), 0.16)) if float(z) > 0 and len(accepted_group) else np.nan,
            "accepted_recovery_q84": float(np.quantile(pd.to_numeric(accepted_group["A_hat"]) / pd.to_numeric(accepted_group["strength"]), 0.84)) if float(z) > 0 and len(accepted_group) else np.nan,
            "accepted_eps2_hat_median": float(pd.to_numeric(accepted_group["eps2_hat_signed"]).median()) if len(accepted_group) else np.nan,
            "accepted_eps2_hat_q16": float(pd.to_numeric(accepted_group["eps2_hat_signed"]).quantile(0.16)) if len(accepted_group) else np.nan,
            "accepted_eps2_hat_q84": float(pd.to_numeric(accepted_group["eps2_hat_signed"]).quantile(0.84)) if len(accepted_group) else np.nan,
            "accepted_eps2_injected_median": float(pd.to_numeric(accepted_group["eps2_injected"]).median()) if len(accepted_group) else np.nan,
            "accepted_upper_boundary_fraction": float(pd.to_numeric(accepted_group["refit_upper_boundary"]).mean()) if len(accepted_group) else np.nan,
        }
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries).sort_values(["scenario", "mass_GeV", "inj_nsigma"])

    zero = summary_df[np.isclose(summary_df.inj_nsigma, 0.0)].copy()
    pvalues = []
    for _, cell in zero.iterrows():
        group = accepted[(accepted.scenario == cell.scenario) & np.isclose(accepted.mass_GeV, cell.mass_GeV) & np.isclose(accepted.inj_nsigma, 0.0)]
        pvalues.append(float(ttest_1samp(pd.to_numeric(group["pull"]), 0.0).pvalue))
    zero["exploratory_ttest_p"] = pvalues
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = (len(pvalues) - rank) * float(pvalues[position])
        running = max(running, candidate)
        adjusted[position] = min(1.0, running)
    zero["exploratory_holm_p"] = adjusted
    zero["exploratory_material_bias_flag"] = (zero.exploratory_holm_p < 0.05) & (zero.accepted_pull_mean.abs() >= 0.2)
    zero[["scenario", "mass_GeV", "exploratory_ttest_p", "exploratory_holm_p", "exploratory_material_bias_flag"]].to_csv(DERIVED / "zero_signal_bias_tests.csv", index=False)

    focus = accepted[(accepted.scenario == "2021_1pct_x10") & np.isclose(accepted.mass_GeV, 0.065) & np.isclose(accepted.inj_nsigma, 0.0)].copy()
    later = focus[focus.background_toy_index >= 10]
    full = focus
    def test_record(group: pd.DataFrame, label: str) -> dict[str, Any]:
        values = pd.to_numeric(group["pull"]).to_numpy(float)
        result = ttest_1samp(values, 0.0)
        n, mean, width = len(values), float(np.mean(values)), float(np.std(values, ddof=1))
        half = float(t.ppf(0.975, n - 1)) * width / math.sqrt(n)
        return {"sample": label, "n": n, "mean_pull": mean, "width": width, "median": float(np.median(values)), "p_two_sided": float(result.pvalue), "ci95_low": mean - half, "ci95_high": mean + half, "materiality_threshold": 0.2, "material_bias_flag": bool(result.pvalue < 0.05 and abs(mean) >= 0.2)}
    bias_records = [
        test_record(later, "descriptive_toys_10_24"),
        test_record(full, "development_toys_0_24"),
    ]
    pd.DataFrame(bias_records).to_csv(DERIVED / "onepct_x10_65mev_bias_test.csv", index=False)

    summary_df.to_csv(DERIVED / "closure_summary.csv", index=False)
    result = {
        "status": "pass",
        "collected_utc": utc_now(),
        "raw_rows": len(raw),
        "accepted_rows": len(accepted),
        "excluded_rows": len(exclusions),
        "optimizer_attempt_rows": len(attempts),
        "summary_cells": len(summary_df),
        "minimum_accepted_per_cell": int(summary_df.accepted_n.min()),
        "all_cells_sample_size_eligible": bool(
            summary_df.development_cell_sample_size_eligible.all()
        ),
        "descriptive_later_15_bias_test": bias_records[0],
        "development_25_bias_estimate": bias_records[1],
        "hashes": {path.name: sha256_file(path) for path in sorted(DERIVED.glob("*.csv"))},
    }
    atomic_json(DERIVED / "collection_summary.json", result)
    return result


def analytic_mean_closure() -> dict[str, Any]:
    """Fit each exact analytic truth mean without a Poisson background draw."""
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import FuncFormToySpec, build_funcform_toy_dataset, load_funcform_toy_hist

    spec = load_spec()
    gate = spec["optimizer_gate"]
    cfg = build_config()
    assert_config(cfg)
    base = make_datasets(cfg)["2021"]
    rows: list[dict[str, Any]] = []
    attempts_all: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        entry = spec["scenarios"][scenario]
        family = str(entry["source_family"])
        multiplier = int(entry["source_multiplier"])
        histogram = load_funcform_toy_hist(
            str(TOY_ROOT), container=f"truth/{TRUTH}", toy_name=f"{family}_mean"
        )
        histogram = histogram * multiplier
        toy_spec = FuncFormToySpec(
            source_root=str(TOY_ROOT), container=f"truth/{TRUTH}",
            function_tag=FUNCTION_TAG, toy_name=f"{family}_mean_x{multiplier}", toy_index=-1,
        )
        ds = build_funcform_toy_dataset(base, histogram, toy_spec)
        for mass in map(float, spec["masses_gev"]):
            attempts = []
            for attempt in range(3):
                record, _ = reference_attempt(ds, cfg, scenario, -1, mass, attempt)
                attempts.append(record)
            selected, replicates = select_branch(attempts, gate, require_replication=True)
            if selected is None:
                for attempt in (3, 4):
                    record, _ = reference_attempt(ds, cfg, scenario, -1, mass, attempt)
                    attempts.append(record)
                selected, replicates = select_branch(attempts, gate, require_replication=True)
            attempts_all.extend(attempts)
            if selected is None:
                raise StudyError(f"analytic-mean reference failed stability gate for {scenario} at {mass}")
            rows.append({
                "scenario": scenario,
                "scenario_label": entry["label"],
                "source_family": family,
                "source_multiplier": multiplier,
                "mass_GeV": mass,
                "mass_MeV": mass * 1000.0,
                "selected_attempt": int(selected["attempt"]),
                "n_attempts": len(attempts),
                "top_branch_replicates": int(replicates),
                "pull": float(selected["pull"]),
                "A_hat": float(selected["A_hat"]),
                "sigma_A": float(selected["sigma_A"]),
                "gp_lml": float(selected["gp_lml"]),
                "gp_ls": float(selected["gp_ls"]),
                "gp_const": float(selected["gp_const"]),
                "gp_ls_over_hi": float(selected["gp_ls"]) / float(selected["gp_ls_hi"]),
                "A_per_eps2_unit": float(selected["A_per_eps2_unit"]),
                "eps2_hat_signed": float(selected["A_hat"]) / float(selected["A_per_eps2_unit"]),
            })
    DERIVED.mkdir(parents=True, exist_ok=True)
    output = DERIVED / "analytic_mean_zero_signal_closure.csv"
    attempt_output = DERIVED / "analytic_mean_optimizer_attempts.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(attempts_all).to_csv(attempt_output, index=False)
    result = {
        "status": "pass",
        "rows": len(rows),
        "attempt_rows": len(attempts_all),
        "output_sha256": sha256_file(output),
        "attempts_sha256": sha256_file(attempt_output),
    }
    atomic_json(DERIVED / "analytic_mean_closure_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    task = sub.add_parser("run-task")
    task.add_argument("scenario", choices=SCENARIOS)
    task.add_argument("toy_index", type=int)
    task.add_argument("--force", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--toy-start", type=int, required=True)
    run.add_argument("--toy-stop", type=int, required=True)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--force", action="store_true")
    sub.add_parser("collect")
    sub.add_parser("analytic-mean")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        result = preflight()
    elif args.command == "run-task":
        result = run_task(args.scenario, args.toy_index, args.force)
    elif args.command == "run":
        result = run_many(args.toy_start, args.toy_stop, args.workers, args.force)
    elif args.command == "collect":
        result = collect()
    elif args.command == "analytic-mean":
        result = analytic_mean_closure()
    else:
        raise StudyError(f"unknown command {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
