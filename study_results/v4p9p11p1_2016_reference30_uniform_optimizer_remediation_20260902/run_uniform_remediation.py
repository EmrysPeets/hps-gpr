#!/usr/bin/env python3
"""Frozen v4.9.11p1 uniform fit-only optimizer remediation.

This program deliberately contains no signal, p-value, or limit calculation.
It reruns the same support-30/k12 GP state optimization at all 142 masses and
either certifies the complete ledger or stops globally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
import uproot
import yaml
from scipy.optimize import Bounds, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = REPO / "study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json"
PROTOCOL_PATH = HERE / "STUDY_PROTOCOL.md"
PROTOCOL_FREEZE = HERE / "FROZEN_PROTOCOL_SHA256"
EXECUTION_FREEZE = HERE / "FROZEN_EXECUTION_SHA256"
PREFLIGHT_PATH = HERE / "qa/preflight.json"
EXPECTED_PROTOCOL_SHA = "ae3f0bde2978f07c3e135d1e978fe9528482556b12cecef3f8dc5173776a9235"
EXPECTED_SPEC_SHA = "4a90696216afbd78d41d4a5f0e249a70488fae823ee32a1b0004d31d327557b6"
EXPECTED_MASSES_MEV = list(range(39, 181))
EXPECTED_SEEDS = [2711, 6043, 9151]
FORBIDDEN_TOKENS = (
    "a_hat",
    "signal_pull",
    "p0",
    "z_local",
    "upper_limit",
    "epsilon2",
    "expected_band",
    "toy_limit",
)


class StudyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def histogram_hash(values: np.ndarray, edges: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    digest.update(np.asarray(edges, dtype=np.float64).tobytes())
    return digest.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def warning_text(caught: list[warnings.WarningMessage]) -> str:
    return " | ".join(f"{type(item.message).__name__}: {item.message}" for item in caught)


def read_freeze(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise StudyError(f"required freeze absent: {path.name}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[1] in result:
            raise StudyError(f"malformed freeze: {path}")
        result[parts[1]] = parts[0]
    return result


def verify_freeze(path: Path) -> dict[str, str]:
    entries = read_freeze(path)
    for relative, expected in entries.items():
        target = (HERE / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise StudyError(f"frozen artifact drift: {relative}")
    return entries


def verify_declared_file(declaration: dict[str, Any], hash_key: str = "sha256") -> Path:
    path = REPO / declaration["path"]
    if not path.is_file() or sha256_file(path) != declaration[hash_key]:
        raise StudyError(f"declared input drift: {declaration['path']}")
    return path


def load_spec(require_execution_freeze: bool) -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA:
        raise StudyError("protocol hash drift")
    if sha256_file(SPEC_PATH) != EXPECTED_SPEC_SHA:
        raise StudyError("spec hash drift")
    protocol_entries = verify_freeze(PROTOCOL_FREEZE)
    expected_protocol_entries = {
        "STUDY_PROTOCOL.md": EXPECTED_PROTOCOL_SHA,
        "../../study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json": EXPECTED_SPEC_SHA,
    }
    if protocol_entries != expected_protocol_entries:
        raise StudyError("protocol freeze contents drift")
    if require_execution_freeze:
        execution_entries = verify_freeze(EXECUTION_FREEZE)
        required = {"run_uniform_remediation.py", "qa/preflight.json"}
        if not required.issubset(execution_entries):
            raise StudyError("execution freeze does not bind runner and preflight")

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["study_id"] != HERE.name:
        raise StudyError("study id/path mismatch")
    if [spec["search_masses_MeV"][key] for key in ("first", "last", "step", "count")] != [39, 180, 1, 142]:
        raise StudyError("mass-grid declaration drift")
    if spec["support_lower_MeV"] != 30 or spec["support_upper_MeV"] != 210:
        raise StudyError("support drift")
    if float(spec["upper_length_factor_2016"]) != 12.0:
        raise StudyError("k12 drift")

    verify_declared_file(spec["reviewed_reference_card"])
    verify_declared_file(spec["archived_states"])
    for declaration in spec["v4p9p11_inputs"].values():
        verify_declared_file(declaration)

    terminal_path = REPO / spec["v4p9p11_inputs"]["terminal_decision"]["path"]
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("status") == "stopped_unresolved_state"
        and terminal.get("combination_authorized") is False
        and terminal.get("state_rows") == 142
        and terminal.get("resolved_rows") == 49
        and len(terminal.get("unresolved_masses_MeV", [])) == 93
        and terminal.get("inference_fields_accessed") == []
    ):
        raise StudyError("v4p9p11 terminal semantics drift")
    control_path = REPO / spec["v4p9p11_inputs"]["canonical_control_decision"]["path"]
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if not (
        control.get("status") == "control_adequacy_pass"
        and control.get("technical_pass") is True
        and control.get("absolute_guard_pass") is True
        and control.get("forbidden_centers_zero") is True
    ):
        raise StudyError("canonical inherited control is not passing")

    card_path = REPO / spec["reviewed_reference_card"]["path"]
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    required_card = {
        "data_range_2016": [0.03, 0.21],
        "range_2016": [0.039, 0.18],
        "neighborhood_rebin": 5,
        "pre_log": True,
        "alpha_model": "1/y",
        "blind_nsigma": 2.25,
        "gp_train_exclude_nsigma": 2.25,
        "kernel_ls_policy": "resolution_scaled_local",
        "scan_require_two_sidebands": True,
    }
    for key, expected in required_card.items():
        if card.get(key) != expected:
            raise StudyError(f"reviewed card semantic drift: {key}")
    if float(card["kernel_ls_res_lower_factor_by_dataset"]["2016"]) != 0.9:
        raise StudyError("reviewed card lower factor drift")
    if float(card["kernel_ls_res_upper_factor_by_dataset"]["2016"]) != 12.0:
        raise StudyError("reviewed card upper factor drift")
    if list(map(float, card["kernel_constant_bounds"])) != list(map(float, spec["kernel_constant_bounds"])):
        raise StudyError("reviewed card constant bounds drift")
    return spec


def load_histogram(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    declaration = spec["full_input"]
    path = REPO / declaration["path"]
    if sha256_file(path) != declaration["file_sha256"]:
        raise StudyError("full ROOT hash drift")
    with uproot.open(path) as handle:
        values, edges = handle[declaration["histogram"]].to_numpy(flow=False)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if histogram_hash(values, edges) != declaration["histogram_sha256"]:
        raise StudyError("full histogram hash drift")
    if not np.allclose(np.diff(edges), float(spec["native_bin_width_GeV"]), rtol=0, atol=5e-14):
        raise StudyError("native binning drift")
    return values, edges


def rebinned(values: np.ndarray, edges: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = float(spec["support_lower_MeV"]) / 1000.0
    upper = float(spec["support_upper_MeV"]) / 1000.0
    mask = (edges[:-1] >= lower - 1e-12) & (edges[1:] <= upper + 1e-12)
    indexes = np.flatnonzero(mask)
    if not indexes.size or not np.all(np.diff(indexes) == 1):
        raise StudyError("support slice invalid")
    selected = values[indexes]
    factor = int(spec["rebin"])
    if selected.size % factor:
        raise StudyError("support violates fixed rebin phase")
    counts = selected.reshape(-1, factor).sum(axis=1)
    native_edges = edges[indexes[0] : indexes[-1] + 2]
    coarse_edges = native_edges[::factor]
    if coarse_edges.size != counts.size + 1:
        coarse_edges = np.append(coarse_edges, native_edges[-1])
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    if np.any(counts <= 0):
        raise StudyError("nonpositive rebinned count")
    return np.asarray(centers), np.asarray(counts), np.asarray(coarse_edges)


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coeffs = [float(value) for value in spec["sigma_coeffs_2016"]]
    transition = float(spec["sigma_tail_m0_2016"])
    if mass <= transition:
        return float(sum(coefficient * mass**power for power, coefficient in enumerate(coeffs)))
    sigma_at_transition = float(sum(coefficient * transition**power for power, coefficient in enumerate(coeffs)))
    return sigma_at_transition + float(spec["sigma_tail_slope_override_2016"]) * (mass - transition)


def sigma_x(mass: float, spec: dict[str, Any]) -> float:
    return float(np.log((mass + sigma_2016(mass, spec)) / mass))


def length_bounds(mass: float, spec: dict[str, Any]) -> tuple[float, float]:
    local = sigma_x(mass, spec)
    grid = np.linspace(0.039, 0.180, int(spec["kernel_ls_res_npts"]))
    global_base = float(np.median([sigma_x(float(point), spec) for point in grid]))
    lower = float(spec["kernel_ls_res_lower_factor_2016"]) * local
    upper_factor = float(spec["kernel_ls_res_upper_factor_2016"])
    upper = max(
        upper_factor * local,
        upper_factor * global_base * float(spec["kernel_ls_local_hi_floor_factor"]),
    )
    return float(lower), float(upper)


def masks_for_mass(centers: np.ndarray, mass: float, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_width = float(spec["blind_nsigma"]) * sigma_2016(mass, spec)
    lower_edge = mass - half_width
    upper_edge = mass + half_width
    low = centers < lower_edge
    high = centers > upper_edge
    train = low | high
    query = (centers >= lower_edge) & (centers <= upper_edge)
    return train, query, low, high


def load_start_sources(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    archived_path = REPO / spec["archived_states"]["path"]
    archived_columns = ["dataset", "mass_GeV", "const_opt", "ls_opt", "lml", "interpolated"]
    archived = pd.read_csv(archived_path, usecols=archived_columns)
    archived = archived.loc[
        (archived["dataset"].astype(str) == "2016")
        & archived["mass_GeV"].mul(1000).round().astype(int).isin(EXPECTED_MASSES_MEV)
    ].copy()
    archived["mass_MeV"] = archived["mass_GeV"].mul(1000).round().astype(int)
    archived = archived.sort_values("mass_MeV").reset_index(drop=True)
    if archived["mass_MeV"].tolist() != EXPECTED_MASSES_MEV or archived["interpolated"].astype(bool).any():
        raise StudyError("archived start grid is not exact noninterpolated 2016 grid")
    if not np.isfinite(archived[["const_opt", "ls_opt", "lml"]].to_numpy(dtype=float)).all():
        raise StudyError("archived starts are nonfinite")

    attempts_path = REPO / spec["v4p9p11_inputs"]["optimizer_attempts"]["path"]
    attempts_columns = ["mass_GeV", "seed", "lml", "constant", "length"]
    attempts = pd.read_csv(attempts_path, usecols=attempts_columns)
    attempts["mass_MeV"] = attempts["mass_GeV"].mul(1000).round().astype(int)
    attempts = attempts.sort_values(["mass_MeV", "seed"]).reset_index(drop=True)
    exact_pairs = [(mass, seed) for mass in EXPECTED_MASSES_MEV for seed in EXPECTED_SEEDS]
    actual_pairs = list(zip(attempts["mass_MeV"].astype(int), attempts["seed"].astype(int)))
    if len(attempts) != 426 or actual_pairs != exact_pairs:
        raise StudyError("v4p9p11 start ledger is not exact 142x3 grid")
    if not np.isfinite(attempts[["constant", "length", "lml"]].to_numpy(dtype=float)).all():
        raise StudyError("v4p9p11 starts are nonfinite")
    return archived, attempts


def clip_source_start(theta: np.ndarray, bounds: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    result = np.asarray(theta, dtype=np.float64).copy()
    fraction = float(spec["source_start_interior_clip_logspan_fraction"])
    for index in range(2):
        span = bounds[index, 1] - bounds[index, 0]
        result[index] = np.clip(result[index], bounds[index, 0] + fraction * span, bounds[index, 1] - fraction * span)
    return result


def source_starts_for_mass(
    mass_mev: int,
    archived: pd.DataFrame,
    attempts: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    mass = mass_mev / 1000.0
    length_lower, length_upper = length_bounds(mass, spec)
    bounds = np.log(
        np.asarray(
            [
                [float(spec["kernel_constant_bounds"][0]), float(spec["kernel_constant_bounds"][1])],
                [length_lower, length_upper],
            ],
            dtype=np.float64,
        )
    )
    archived_row = archived.loc[archived["mass_MeV"] == mass_mev].iloc[0]
    prior: list[dict[str, Any]] = [
        {
            "label": "archived",
            "constant": float(archived_row["const_opt"]),
            "length": float(archived_row["ls_opt"]),
            "source_recorded_lml": float(archived_row["lml"]),
        }
    ]
    mass_attempts = attempts.loc[attempts["mass_MeV"] == mass_mev].sort_values("seed")
    for row in mass_attempts.itertuples(index=False):
        prior.append(
            {
                "label": f"v4p9p11_seed_{int(row.seed)}",
                "constant": float(row.constant),
                "length": float(row.length),
                "source_recorded_lml": float(row.lml),
            }
        )
    for item in prior:
        original = np.log([item["constant"], item["length"]])
        clipped = clip_source_start(original, bounds, spec)
        item["theta"] = clipped
        item["source_clipped"] = bool(not np.array_equal(original, clipped))

    card = {
        "label": "card_initializer",
        "constant": float(spec["kernel_constant_init"]),
        "length": math.sqrt(length_lower * length_upper),
        "source_recorded_lml": math.nan,
        "source_clipped": False,
    }
    card["theta"] = np.log([card["constant"], card["length"]])
    lattice: list[dict[str, Any]] = []
    for constant in map(float, spec["lattice_constant_values"]):
        for fraction in map(float, spec["lattice_length_log_fractions"]):
            log_length = bounds[1, 0] + fraction * (bounds[1, 1] - bounds[1, 0])
            fraction_label = str(fraction).replace(".", "p")
            item = {
                "label": f"lattice_c{int(constant)}_f{fraction_label}",
                "constant": constant,
                "length": float(np.exp(log_length)),
                "source_recorded_lml": math.nan,
                "source_clipped": False,
                "theta": np.asarray([math.log(constant), log_length], dtype=np.float64),
            }
            lattice.append(item)
    direct = prior + [card] + lattice
    if len(direct) != 14 or len({item["label"] for item in direct}) != 14:
        raise StudyError(f"mass {mass_mev}: direct start construction is not exactly 14")
    return direct, prior, bounds


def prepare_objective(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mass: float,
    spec: dict[str, Any],
) -> tuple[GaussianProcessRegressor, Callable[[np.ndarray], tuple[float, np.ndarray]], Callable[[np.ndarray], float], str]:
    lower, upper = length_bounds(mass, spec)
    kernel = ConstantKernel(
        float(spec["kernel_constant_init"]), tuple(map(float, spec["kernel_constant_bounds"]))
    ) * RBF(math.sqrt(lower * upper), (lower, upper))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0 / np.asarray(y_train, dtype=np.float64),
        optimizer=None,
        normalize_y=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(np.log(x_train).reshape(-1, 1), np.log(y_train))
    setup_warnings = warning_text(caught)

    def analytic(theta: np.ndarray) -> tuple[float, np.ndarray]:
        lml, gradient = model.log_marginal_likelihood(
            theta=np.asarray(theta, dtype=np.float64), eval_gradient=True, clone_kernel=True
        )
        return -float(lml), -np.asarray(gradient, dtype=np.float64)

    def scalar(theta: np.ndarray) -> float:
        lml = model.log_marginal_likelihood(
            theta=np.asarray(theta, dtype=np.float64), eval_gradient=False, clone_kernel=True
        )
        return -float(lml)

    return model, analytic, scalar, setup_warnings


def execute_stage(
    objective: Callable[..., Any],
    start: np.ndarray,
    method: str,
    bounds: np.ndarray,
    options: dict[str, Any],
    jac: bool,
) -> dict[str, Any]:
    result: Any = None
    caught: list[warnings.WarningMessage] = []
    error = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if method == "trust-constr":
                result = minimize(
                    objective,
                    np.asarray(start, dtype=np.float64),
                    method=method,
                    jac=jac,
                    bounds=Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=True),
                    options=options,
                )
            else:
                result = minimize(
                    objective,
                    np.asarray(start, dtype=np.float64),
                    method=method,
                    jac=jac,
                    bounds=[tuple(row) for row in bounds],
                    options=options,
                )
    except Exception as exception:
        error = repr(exception)
    if result is None:
        final_theta = np.asarray([math.nan, math.nan])
        return {
            "success": False,
            "status": -999,
            "message": "exception",
            "warning_count": len(caught),
            "warnings": warning_text(caught),
            "error": error,
            "nit": -1,
            "nfev": -1,
            "njev": -1,
            "objective": math.nan,
            "theta_constant": math.nan,
            "theta_length": math.nan,
            "final_theta": final_theta,
        }
    final_theta = np.asarray(result.x, dtype=np.float64)
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "warning_count": len(caught),
        "warnings": warning_text(caught),
        "error": error,
        "nit": int(getattr(result, "nit", -1)),
        "nfev": int(getattr(result, "nfev", -1)),
        "njev": int(getattr(result, "njev", -1)),
        "objective": float(result.fun) if np.isfinite(result.fun) else math.nan,
        "theta_constant": float(final_theta[0]) if final_theta.size == 2 else math.nan,
        "theta_length": float(final_theta[1]) if final_theta.size == 2 else math.nan,
        "final_theta": final_theta,
    }


def lbfgsb_options(spec: dict[str, Any]) -> dict[str, Any]:
    values = spec["lbfgsb"]
    return {
        "maxiter": int(values["maxiter"]),
        "maxls": int(values["maxls"]),
        "ftol": float(values["ftol"]),
        "gtol": float(values["gtol"]),
    }


def fixed_lml_and_gradient(
    model: GaussianProcessRegressor, theta: np.ndarray
) -> tuple[float, np.ndarray, str, str]:
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lml, gradient = model.log_marginal_likelihood(
                theta=np.asarray(theta, dtype=np.float64), eval_gradient=True, clone_kernel=True
            )
        return float(lml), np.asarray(gradient, dtype=np.float64), warning_text(caught), ""
    except Exception as exception:
        return math.nan, np.asarray([math.nan, math.nan]), warning_text(caught), repr(exception)


def coordinates_interior(theta: np.ndarray, bounds: np.ndarray, spec: dict[str, Any]) -> bool:
    coordinates = np.exp(np.asarray(theta, dtype=np.float64))
    physical_bounds = np.exp(bounds)
    tolerance = float(spec["kernel_bound_rel_tolerance"])
    return bool(
        np.isfinite(coordinates).all()
        and not np.isclose(coordinates[0], physical_bounds[0, 0], rtol=tolerance, atol=1e-12)
        and not np.isclose(coordinates[0], physical_bounds[0, 1], rtol=tolerance, atol=1e-12)
        and not np.isclose(coordinates[1], physical_bounds[1, 0], rtol=tolerance, atol=1e-12)
        and not np.isclose(coordinates[1], physical_bounds[1, 1], rtol=tolerance, atol=1e-12)
    )


def finalize_path(
    mass_mev: int,
    family: str,
    path_id: str,
    start_label: str,
    start_theta: np.ndarray,
    bounds: np.ndarray,
    setup_warnings: str,
    stages: list[dict[str, Any]],
    model: GaussianProcessRegressor,
    spec: dict[str, Any],
    best_prior_label: str,
) -> dict[str, Any]:
    final_stage = stages[-1]
    final_theta = np.asarray(final_stage["final_theta"], dtype=np.float64)
    fixed_lml, gradient, exact_warnings, exact_error = fixed_lml_and_gradient(model, final_theta)
    objective_lml = -float(final_stage["objective"]) if np.isfinite(final_stage["objective"]) else math.nan
    objective_lml_difference = objective_lml - fixed_lml
    gradient_infinity = float(np.max(np.abs(gradient))) if np.isfinite(gradient).all() else math.nan
    stage_success = bool(all(stage["success"] for stage in stages))
    stage_warning_count = int(sum(int(stage["warning_count"]) for stage in stages))
    warning_free = bool(not setup_warnings and stage_warning_count == 0 and not exact_warnings)
    no_errors = bool(all(not stage["error"] for stage in stages) and not exact_error)
    finite = bool(final_theta.size == 2 and np.isfinite(final_theta).all() and np.isfinite(fixed_lml))
    interior = coordinates_interior(final_theta, bounds, spec) if finite else False
    fixed_closure = bool(
        np.isfinite(objective_lml_difference)
        and abs(objective_lml_difference) <= float(spec["fixed_lml_abs_tolerance"])
    )
    gradient_pass = bool(np.isfinite(gradient_infinity) and gradient_infinity < float(spec["analytic_gradient_infinity_max"]))
    eligible = bool(stage_success and warning_free and no_errors and finite and interior and fixed_closure and gradient_pass)
    coordinates = np.exp(final_theta) if finite else np.asarray([math.nan, math.nan])
    row: dict[str, Any] = {
        "dataset": "2016",
        "mass_GeV": mass_mev / 1000.0,
        "mass_MeV": mass_mev,
        "method_family": family,
        "path_id": path_id,
        "start_label": start_label,
        "best_prior_label": best_prior_label,
        "start_constant": float(np.exp(start_theta[0])),
        "start_length": float(np.exp(start_theta[1])),
        "postpolish_constant": float(coordinates[0]),
        "postpolish_length": float(coordinates[1]),
        "optimizer_objective": float(final_stage["objective"]),
        "optimizer_lml_explicit_negative_objective": objective_lml,
        "fixed_lml": fixed_lml,
        "optimizer_lml_minus_fixed_lml": objective_lml_difference,
        "gradient_constant_log": float(gradient[0]),
        "gradient_length_log": float(gradient[1]),
        "gradient_infinity": gradient_infinity,
        "setup_warnings": setup_warnings,
        "stage_success_all": stage_success,
        "stage_warning_count": stage_warning_count,
        "exact_warning_count": 0 if not exact_warnings else len(exact_warnings.split(" | ")),
        "warning_free_all": warning_free,
        "no_stage_or_exact_error": no_errors,
        "coordinates_finite": finite,
        "coordinates_interior": interior,
        "objective_fixed_lml_closure_pass": fixed_closure,
        "gradient_pass": gradient_pass,
        "path_eligible": eligible,
        "exact_warnings": exact_warnings,
        "exact_error": exact_error,
        "length_lower": float(np.exp(bounds[1, 0])),
        "length_upper": float(np.exp(bounds[1, 1])),
    }
    for index in range(2):
        stage = stages[index] if index < len(stages) else None
        prefix = f"stage{index + 1}_"
        for key in (
            "success",
            "status",
            "message",
            "warning_count",
            "warnings",
            "error",
            "nit",
            "nfev",
            "njev",
            "objective",
            "theta_constant",
            "theta_length",
        ):
            row[prefix + key] = stage[key] if stage is not None else None
    return row


def lognormal_counts(mean_log: np.ndarray, cov_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.clip(np.diag(cov_log), 0.0, None)
    mean = np.exp(mean_log + 0.5 * diagonal)
    covariance = np.outer(mean, mean) * (np.exp(np.clip(cov_log, -40.0, 40.0)) - 1.0)
    return np.asarray(mean), np.asarray(covariance)


def covariance_metrics(mean: np.ndarray, gp_cov: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    total = 0.5 * (gp_cov + gp_cov.T) + np.diag(np.clip(mean, 1e-12, None))
    total = 0.5 * (total + total.T)
    diagonal = np.diag(total)
    maximum_diagonal = float(np.max(diagonal))
    median_diagonal = float(np.median(diagonal))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(total)))
    negative_ok = minimum_eigenvalue >= -float(spec["covariance_negative_eigen_rel_tolerance"]) * maximum_diagonal
    cholesky = None
    jitter = math.nan
    for relative in (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        try:
            cholesky = np.linalg.cholesky(total + np.eye(len(mean)) * relative * median_diagonal)
            jitter = float(relative)
            break
        except np.linalg.LinAlgError:
            continue
    covariance_ok = bool(
        negative_ok
        and cholesky is not None
        and jitter <= float(spec["covariance_max_jitter_rel_median_diag"])
    )
    return {
        "covariance_ok": covariance_ok,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_diagonal": maximum_diagonal,
        "median_diagonal": median_diagonal,
        "jitter_relative_median_diagonal": jitter,
    }


def reconstruct_selected_state(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    mass: float,
    constant: float,
    length: float,
    selected_fixed_lml: float,
    spec: dict[str, Any],
) -> dict[str, Any]:
    lower, upper = length_bounds(mass, spec)
    kernel = ConstantKernel(constant, tuple(map(float, spec["kernel_constant_bounds"]))) * RBF(length, (lower, upper))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0 / np.asarray(y_train, dtype=np.float64),
        optimizer=None,
        normalize_y=False,
    )
    caught: list[warnings.WarningMessage] = []
    error = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(np.log(x_train).reshape(-1, 1), np.log(y_train))
            reconstructed_lml = float(model.log_marginal_likelihood_value_)
            theta = np.log([constant, length])
            analytic_lml, gradient = model.log_marginal_likelihood(theta=theta, eval_gradient=True, clone_kernel=True)
            mean_sklearn, cov_sklearn = model.predict(np.log(x_query).reshape(-1, 1), return_cov=True)

        train_log = np.log(x_train)
        query_log = np.log(x_query)
        distance_train = (train_log[:, None] - train_log[None, :]) / length
        matrix = constant * np.exp(-0.5 * distance_train**2) + np.diag(1.0 / y_train)
        cholesky = np.linalg.cholesky(matrix)
        cross = constant * np.exp(-0.5 * ((train_log[:, None] - query_log[None, :]) / length) ** 2)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, np.log(y_train)))
        mean_direct = cross.T @ alpha
        projected = np.linalg.solve(cholesky, cross)
        query_cov = constant * np.exp(-0.5 * ((query_log[:, None] - query_log[None, :]) / length) ** 2)
        raw_direct_cov = query_cov - projected.T @ projected
        cov_direct = 0.5 * (raw_direct_cov + raw_direct_cov.T)
        prediction_closure = bool(
            np.allclose(
                mean_sklearn,
                mean_direct,
                rtol=float(spec["direct_prediction_relative_tolerance"]),
                atol=float(spec["direct_prediction_absolute_tolerance"]),
            )
            and np.allclose(
                cov_sklearn,
                cov_direct,
                rtol=float(spec["direct_prediction_relative_tolerance"]),
                atol=float(spec["direct_prediction_absolute_tolerance"]),
            )
        )
        mean_counts, cov_counts = lognormal_counts(np.asarray(mean_sklearn), np.asarray(cov_sklearn))
        covariance = covariance_metrics(mean_counts, cov_counts, spec)
        gradient = np.asarray(gradient, dtype=np.float64)
        gradient_infinity = float(np.max(np.abs(gradient)))
        lml_closure = bool(abs(reconstructed_lml - selected_fixed_lml) <= float(spec["fixed_lml_abs_tolerance"]))
        analytic_closure = bool(abs(float(analytic_lml) - reconstructed_lml) <= float(spec["fixed_lml_abs_tolerance"]))
        interior = coordinates_interior(theta, np.log(np.asarray([[spec["kernel_constant_bounds"][0], spec["kernel_constant_bounds"][1]], [lower, upper]], dtype=float)), spec)
        certificate = bool(
            not caught
            and not error
            and lml_closure
            and analytic_closure
            and prediction_closure
            and covariance["covariance_ok"]
            and interior
            and gradient_infinity < float(spec["analytic_gradient_infinity_max"])
        )
        return {
            "reconstruction_success": True,
            "reconstruction_warning_count": len(caught),
            "reconstruction_warnings": warning_text(caught),
            "reconstruction_error": error,
            "reconstructed_lml": reconstructed_lml,
            "reconstructed_minus_selected_fixed_lml": reconstructed_lml - selected_fixed_lml,
            "analytic_lml_minus_reconstructed_lml": float(analytic_lml) - reconstructed_lml,
            "selected_gradient_constant_log": float(gradient[0]),
            "selected_gradient_length_log": float(gradient[1]),
            "selected_gradient_infinity": gradient_infinity,
            "selected_coordinates_interior": interior,
            "prediction_closure_pass": prediction_closure,
            "prediction_mean_max_abs_difference": float(np.max(np.abs(mean_sklearn - mean_direct))),
            "prediction_covariance_max_abs_difference": float(np.max(np.abs(cov_sklearn - cov_direct))),
            "prediction_mean_sha256": array_hash(mean_counts),
            "prediction_covariance_sha256": array_hash(cov_counts),
            **covariance,
            "selected_state_certificate_pass": certificate,
        }
    except Exception as exception:
        error = repr(exception)
        return {
            "reconstruction_success": False,
            "reconstruction_warning_count": len(caught),
            "reconstruction_warnings": warning_text(caught),
            "reconstruction_error": error,
            "reconstructed_lml": math.nan,
            "reconstructed_minus_selected_fixed_lml": math.nan,
            "analytic_lml_minus_reconstructed_lml": math.nan,
            "selected_gradient_constant_log": math.nan,
            "selected_gradient_length_log": math.nan,
            "selected_gradient_infinity": math.nan,
            "selected_coordinates_interior": False,
            "prediction_closure_pass": False,
            "prediction_mean_max_abs_difference": math.nan,
            "prediction_covariance_max_abs_difference": math.nan,
            "prediction_mean_sha256": "",
            "prediction_covariance_sha256": "",
            "covariance_ok": False,
            "minimum_eigenvalue": math.nan,
            "maximum_diagonal": math.nan,
            "median_diagonal": math.nan,
            "jitter_relative_median_diagonal": math.nan,
            "selected_state_certificate_pass": False,
        }


def optimize_mass(payload: tuple[int, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], np.ndarray, dict[str, Any]]) -> dict[str, Any]:
    mass_mev, centers, counts, direct_starts, prior_starts, bounds, spec = payload
    mass = mass_mev / 1000.0
    train, query, low, high = masks_for_mass(centers, mass, spec)
    x_train = centers[train]
    y_train = counts[train]
    model, analytic, scalar, setup_warnings = prepare_objective(x_train, y_train, mass, spec)

    prior_scores: list[tuple[float, str, np.ndarray]] = []
    for item in prior_starts:
        fixed_lml, _, exact_warnings, exact_error = fixed_lml_and_gradient(model, item["theta"])
        if exact_warnings or exact_error or not np.isfinite(fixed_lml):
            raise StudyError(f"mass {mass_mev}: cannot score frozen prior start {item['label']}")
        prior_scores.append((fixed_lml, item["label"], item["theta"]))
    prior_scores.sort(key=lambda value: (-value[0], value[1]))
    _, best_prior_label, best_prior_theta = prior_scores[0]

    path_rows: list[dict[str, Any]] = []
    for item in direct_starts:
        stage = execute_stage(analytic, item["theta"], "L-BFGS-B", bounds, lbfgsb_options(spec), jac=True)
        path_rows.append(
            finalize_path(
                mass_mev,
                "direct_lbfgsb",
                f"direct_lbfgsb__{item['label']}",
                item["label"],
                item["theta"],
                bounds,
                setup_warnings,
                [stage],
                model,
                spec,
                best_prior_label,
            )
        )

    card_theta = next(item["theta"] for item in direct_starts if item["label"] == "card_initializer")
    lattice_center_theta = next(item["theta"] for item in direct_starts if item["label"] == "lattice_c100_f0p5")
    powell_options = {
        "maxiter": int(spec["powell"]["maxiter"]),
        "maxfev": int(spec["powell"]["maxfev"]),
        "xtol": float(spec["powell"]["xtol"]),
        "ftol": float(spec["powell"]["ftol"]),
    }
    trust_options = {
        "maxiter": int(spec["trust_constr"]["maxiter"]),
        "gtol": float(spec["trust_constr"]["gtol"]),
        "xtol": float(spec["trust_constr"]["xtol"]),
        "barrier_tol": float(spec["trust_constr"]["barrier_tol"]),
    }
    for label, start in (("card_initializer", card_theta), ("best_prior_source", best_prior_theta)):
        first = execute_stage(scalar, start, "Powell", bounds, powell_options, jac=False)
        second_start = first["final_theta"] if np.isfinite(first["final_theta"]).all() else start
        second = execute_stage(analytic, second_start, "L-BFGS-B", bounds, lbfgsb_options(spec), jac=True)
        path_rows.append(
            finalize_path(
                mass_mev,
                "powell_lbfgsb",
                f"powell_lbfgsb__{label}",
                label,
                start,
                bounds,
                setup_warnings,
                [first, second],
                model,
                spec,
                best_prior_label,
            )
        )
    for label, start in (("fixed_lattice_center", lattice_center_theta), ("best_prior_source", best_prior_theta)):
        first = execute_stage(analytic, start, "trust-constr", bounds, trust_options, jac=True)
        second_start = first["final_theta"] if np.isfinite(first["final_theta"]).all() else start
        second = execute_stage(analytic, second_start, "L-BFGS-B", bounds, lbfgsb_options(spec), jac=True)
        path_rows.append(
            finalize_path(
                mass_mev,
                "trust_lbfgsb",
                f"trust_lbfgsb__{label}",
                label,
                start,
                bounds,
                setup_warnings,
                [first, second],
                model,
                spec,
                best_prior_label,
            )
        )

    if len(path_rows) != 18:
        raise StudyError(f"mass {mass_mev}: path row count is not 18")
    family_counts = pd.Series([row["method_family"] for row in path_rows]).value_counts().to_dict()
    if family_counts != {"direct_lbfgsb": 14, "powell_lbfgsb": 2, "trust_lbfgsb": 2}:
        raise StudyError(f"mass {mass_mev}: method multiplicity drift")

    eligible = [row for row in path_rows if row["path_eligible"]]
    if eligible:
        eligible.sort(key=lambda row: (-float(row["fixed_lml"]), str(row["path_id"])))
        selected = eligible[0]
        selected_lml = float(selected["fixed_lml"])
        selected_constant = float(selected["postpolish_constant"])
        selected_length = float(selected["postpolish_length"])
        lml_tolerance = float(spec["cluster_lml_abs_tolerance"])
        coordinate_tolerance = float(spec["cluster_coordinate_rel_tolerance"])
        cluster = [
            row
            for row in eligible
            if abs(float(row["fixed_lml"]) - selected_lml) <= lml_tolerance
            and abs(float(row["postpolish_constant"]) - selected_constant) <= coordinate_tolerance * abs(selected_constant)
            and abs(float(row["postpolish_length"]) - selected_length) <= coordinate_tolerance * abs(selected_length)
        ]
        cluster_families = sorted({str(row["method_family"]) for row in cluster})
        cluster_pass = bool(
            len(cluster) >= int(spec["cluster_minimum_paths"])
            and len(cluster_families) >= int(spec["cluster_minimum_method_families"])
        )
        reconstruction = reconstruct_selected_state(
            x_train,
            y_train,
            centers[query],
            mass,
            selected_constant,
            selected_length,
            selected_lml,
            spec,
        )
        selected_path_id = str(selected["path_id"])
        selected_family = str(selected["method_family"])
    else:
        selected_lml = selected_constant = selected_length = math.nan
        cluster = []
        cluster_families = []
        cluster_pass = False
        reconstruction = {
            "reconstruction_success": False,
            "reconstruction_warning_count": 0,
            "reconstruction_warnings": "",
            "reconstruction_error": "no eligible path",
            "reconstructed_lml": math.nan,
            "reconstructed_minus_selected_fixed_lml": math.nan,
            "analytic_lml_minus_reconstructed_lml": math.nan,
            "selected_gradient_constant_log": math.nan,
            "selected_gradient_length_log": math.nan,
            "selected_gradient_infinity": math.nan,
            "selected_coordinates_interior": False,
            "prediction_closure_pass": False,
            "prediction_mean_max_abs_difference": math.nan,
            "prediction_covariance_max_abs_difference": math.nan,
            "prediction_mean_sha256": "",
            "prediction_covariance_sha256": "",
            "covariance_ok": False,
            "minimum_eigenvalue": math.nan,
            "maximum_diagonal": math.nan,
            "median_diagonal": math.nan,
            "jitter_relative_median_diagonal": math.nan,
            "selected_state_certificate_pass": False,
        }
        selected_path_id = ""
        selected_family = ""

    two_sidebands = bool(np.count_nonzero(low) > 0 and np.count_nonzero(high) > 0)
    state_resolved = bool(cluster_pass and reconstruction["selected_state_certificate_pass"] and two_sidebands)
    state_row = {
        "dataset": "2016",
        "mass_GeV": mass,
        "mass_MeV": mass_mev,
        "state_source": "v4p9p11p1_uniform_postpolish_max_fixed_lml",
        "selected_path_id": selected_path_id,
        "selected_method_family": selected_family,
        "const_opt": selected_constant,
        "ls_opt": selected_length,
        "lml": selected_lml,
        "eligible_path_count": len(eligible),
        "selected_cluster_path_count": len(cluster),
        "selected_cluster_method_family_count": len(cluster_families),
        "selected_cluster_method_families": "|".join(cluster_families),
        "selected_cluster_path_ids": "|".join(sorted(str(row["path_id"]) for row in cluster)),
        "selected_cluster_pass": cluster_pass,
        "best_prior_label": best_prior_label,
        "n_train": int(np.count_nonzero(train)),
        "n_train_low": int(np.count_nonzero(low)),
        "n_train_high": int(np.count_nonzero(high)),
        "n_query": int(np.count_nonzero(query)),
        "train_centers_sha256": array_hash(centers[train]),
        "train_counts_sha256": array_hash(counts[train]),
        "query_centers_sha256": array_hash(centers[query]),
        "two_training_sidebands": two_sidebands,
        **reconstruction,
        "state_resolved": state_resolved,
    }
    return {"mass_MeV": mass_mev, "paths": path_rows, "state": state_row}


def preflight(spec: dict[str, Any]) -> None:
    if PREFLIGHT_PATH.exists():
        raise StudyError("preflight output already exists; no overwrite permitted")
    values, edges = load_histogram(spec)
    centers, counts, coarse_edges = rebinned(values, edges, spec)
    archived, attempts = load_start_sources(spec)
    plan_rows: list[dict[str, Any]] = []
    all_sidebands = True
    for mass_mev in EXPECTED_MASSES_MEV:
        direct, prior, bounds = source_starts_for_mass(mass_mev, archived, attempts, spec)
        mass = mass_mev / 1000.0
        train, query, low, high = masks_for_mass(centers, mass, spec)
        all_sidebands = all_sidebands and bool(np.count_nonzero(low) and np.count_nonzero(high))
        plan_rows.append(
            {
                "mass_MeV": mass_mev,
                "direct_paths": len(direct),
                "powell_paths": 2,
                "trust_paths": 2,
                "total_paths": len(direct) + 4,
                "prior_pool": [item["label"] for item in prior],
                "lattice_center_constant": 100.0,
                "lattice_center_log_length_fraction": 0.5,
                "n_train": int(np.count_nonzero(train)),
                "n_query": int(np.count_nonzero(query)),
                "n_train_low": int(np.count_nonzero(low)),
                "n_train_high": int(np.count_nonzero(high)),
                "length_lower": float(np.exp(bounds[1, 0])),
                "length_upper": float(np.exp(bounds[1, 1])),
            }
        )
    planned_paths = sum(row["total_paths"] for row in plan_rows)
    exact = bool(
        len(plan_rows) == 142
        and planned_paths == 2556
        and all(row["direct_paths"] == 14 and row["powell_paths"] == 2 and row["trust_paths"] == 2 for row in plan_rows)
        and all(row["prior_pool"] == ["archived", "v4p9p11_seed_2711", "v4p9p11_seed_6043", "v4p9p11_seed_9151"] for row in plan_rows)
        and all_sidebands
    )
    payload = {
        "status": "preflight_pass" if exact else "preflight_failure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
        "full_input_sha256": spec["full_input"]["file_sha256"],
        "full_histogram_sha256": histogram_hash(values, edges),
        "rebinned_centers_sha256": array_hash(centers),
        "rebinned_counts_sha256": array_hash(counts),
        "rebinned_edges_sha256": array_hash(coarse_edges),
        "mass_rows": len(plan_rows),
        "mass_grid_MeV": EXPECTED_MASSES_MEV,
        "planned_path_rows": planned_paths,
        "per_mass_method_multiplicity": {"direct_lbfgsb": 14, "powell_lbfgsb": 2, "trust_lbfgsb": 2},
        "best_prior_pool_exact": ["archived", "v4p9p11_seed_2711", "v4p9p11_seed_6043", "v4p9p11_seed_9151"],
        "lattice_center": {"constant": 100.0, "log_length_fraction": 0.5},
        "all_masses_have_two_training_sidebands": all_sidebands,
        "forbidden_inference_fields_accessed": [],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "uproot": uproot.__version__,
        },
        "plan": plan_rows,
    }
    json_write(PREFLIGHT_PATH, payload)
    if not exact:
        raise StudyError("preflight failed")
    print(json.dumps({key: payload[key] for key in ("status", "mass_rows", "planned_path_rows", "script_sha256")}, indent=2))


def run(spec: dict[str, Any], workers: int) -> None:
    output_dir = HERE / "derived"
    paths_path = output_dir / "optimizer_paths.csv"
    states_path = output_dir / "observed_2016_gp_states_reviewed.csv"
    decision_path = output_dir / "state_certification_decision.json"
    if any(path.exists() for path in (paths_path, states_path, decision_path)):
        raise StudyError("execution output already exists; frozen no-retry rule forbids overwrite")
    preflight_payload = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if not (
        preflight_payload.get("status") == "preflight_pass"
        and preflight_payload.get("script_sha256") == sha256_file(Path(__file__))
        and preflight_payload.get("planned_path_rows") == 2556
    ):
        raise StudyError("frozen preflight is absent, stale, or failed")

    values, edges = load_histogram(spec)
    centers, counts, _ = rebinned(values, edges, spec)
    archived, attempts = load_start_sources(spec)
    payloads = []
    for mass_mev in EXPECTED_MASSES_MEV:
        direct, prior, bounds = source_starts_for_mass(mass_mev, archived, attempts, spec)
        payloads.append((mass_mev, centers, counts, direct, prior, bounds, spec))

    results: list[dict[str, Any]] = []
    if workers == 1:
        for index, payload in enumerate(payloads, start=1):
            results.append(optimize_mass(payload))
            if index % 10 == 0 or index == len(payloads):
                print(f"completed {index}/142 masses", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(optimize_mass, payload): payload[0] for payload in payloads}
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 10 == 0 or completed == len(payloads):
                    print(f"completed {completed}/142 masses", flush=True)

    results.sort(key=lambda item: int(item["mass_MeV"]))
    path_rows = [row for result in results for row in result["paths"]]
    state_rows = [result["state"] for result in results]
    path_frame = pd.DataFrame(path_rows).sort_values(["mass_MeV", "method_family", "path_id"]).reset_index(drop=True)
    state_frame = pd.DataFrame(state_rows).sort_values("mass_MeV").reset_index(drop=True)
    family_counts = path_frame.groupby(["mass_MeV", "method_family"]).size().unstack(fill_value=0)
    exact_paths = bool(
        len(path_frame) == 2556
        and path_frame.groupby("mass_MeV").size().eq(18).all()
        and family_counts["direct_lbfgsb"].eq(14).all()
        and family_counts["powell_lbfgsb"].eq(2).all()
        and family_counts["trust_lbfgsb"].eq(2).all()
    )
    exact_states = bool(len(state_frame) == 142 and state_frame["mass_MeV"].astype(int).tolist() == EXPECTED_MASSES_MEV)
    if not exact_paths or not exact_states:
        raise StudyError("execution did not produce exact frozen path/state grid")
    forbidden_columns = [
        column
        for column in list(path_frame.columns) + list(state_frame.columns)
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden_columns:
        raise StudyError(f"forbidden inference columns appeared: {forbidden_columns}")

    output_dir.mkdir(parents=True, exist_ok=True)
    path_frame.to_csv(paths_path, index=False)
    state_frame.to_csv(states_path, index=False)
    all_resolved = bool(state_frame["state_resolved"].astype(bool).all())
    unresolved = state_frame.loc[~state_frame["state_resolved"].astype(bool), "mass_MeV"].astype(int).tolist()
    decision = {
        "status": "all_142_states_certified" if all_resolved else "stopped_unresolved_state",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "combination_authorized": all_resolved,
        "support_lower_MeV": 30,
        "support_upper_MeV": 210,
        "upper_length_factor_2016": 12.0,
        "reference_status": "retained_pre_existing_reviewed_reference_not_newly_selected",
        "state_rows": len(state_frame),
        "resolved_rows": int(state_frame["state_resolved"].astype(bool).sum()),
        "unresolved_masses_MeV": unresolved,
        "optimizer_paths": {"rows": len(path_frame), "sha256": sha256_file(paths_path)},
        "states": {"rows": len(state_frame), "sha256": sha256_file(states_path)},
        "path_method_multiplicity_per_mass": {"direct_lbfgsb": 14, "powell_lbfgsb": 2, "trust_lbfgsb": 2},
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "input_terminal_decision_sha256": spec["v4p9p11_inputs"]["terminal_decision"]["sha256"],
        "input_optimizer_attempts_sha256": spec["v4p9p11_inputs"]["optimizer_attempts"]["sha256"],
        "input_archived_states_sha256": spec["archived_states"]["sha256"],
        "selection_metric": "maximum freshly reconstructed fixed-coordinate LML at eligible post-polish coordinates",
        "lower_lml_branch_substitution_permitted": False,
        "inference_fields_accessed": [],
        "inference_artifacts_produced": [],
        "claim_boundary": "fixed-model state certification conditional on a partially unblinded model history; no unconditional coverage or independent-blinding claim",
    }
    json_write(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise StudyError("workers must be in 1..8")
    spec = load_spec(require_execution_freeze=args.mode == "run")
    if args.mode == "preflight":
        preflight(spec)
    else:
        run(spec, args.workers)


if __name__ == "__main__":
    main()
