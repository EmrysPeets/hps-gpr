#!/usr/bin/env python3
"""Execute only the frozen v4.9.11 archive and robust-state phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import uproot
import yaml
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = REPO / "study_configs/v4p9p11_2016_reference30_state_certification_20260902/study_spec.json"
PROTOCOL_PATH = HERE / "STUDY_PROTOCOL.md"
EXPECTED_PROTOCOL_SHA = "bf3253ec0fe34ed72b8569c1f99824387fcd96c46a6ce63206a04a3f470e4481"
EXPECTED_SPEC_SHA = "4c1c8355943e29e39c0cae3cce51f6b60e9878424c3b18437f86150bc07c7d4d"
CONTROL_FREEZE = HERE / "FROZEN_CONTROL_PASS_SHA256"
ARCHIVE_FREEZE = HERE / "FROZEN_ARCHIVE_CLASSIFICATION_SHA256"
CONTROL_DECISION_RELATIVE = "derived/control_adequacy/control_decision_initial_frozen.json"
EXPECTED_CONTROL_SCRIPT_SHA = "b8a7312a86d5f25eb64ba9da46834a0abd151cad97e968d93c9b3205699219b8"


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


def read_freeze(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise StudyError(f"required freeze is absent: {path.name}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise StudyError(f"malformed freeze: {path}")
        result[parts[1]] = parts[0]
    return result


def require_frozen(path: Path, relative: str) -> None:
    frozen = read_freeze(path)
    target = HERE / relative
    if frozen.get(relative) != sha256_file(target):
        raise StudyError(f"frozen hash mismatch: {relative}")


def load_spec() -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA:
        raise StudyError("protocol hash drift")
    if sha256_file(SPEC_PATH) != EXPECTED_SPEC_SHA:
        raise StudyError("spec hash drift")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    for group, path_key, hash_key in (
        (spec["reviewed_reference_card"], "path", "sha256"),
        (spec["v4p9p10_evidence"], "decision_path", "decision_sha256"),
        (spec["v4p9p10_evidence"], "selected_cells_path", "selected_cells_sha256"),
        (spec["v4p9p10_evidence"], "validation_path", "validation_sha256"),
        (spec["archived_states"], "path", "sha256"),
    ):
        if sha256_file(REPO / group[path_key]) != group[hash_key]:
            raise StudyError(f"frozen input drift: {group[path_key]}")
    for relative, digest in spec["archived_states"]["sources"].items():
        if sha256_file(REPO / relative) != digest:
            raise StudyError(f"archived source drift: {relative}")
    card = yaml.safe_load((REPO / spec["reviewed_reference_card"]["path"]).read_text(encoding="utf-8"))
    required = {
        "data_range_2016": [0.03, 0.21],
        "range_2016": [0.039, 0.18],
        "neighborhood_rebin": 5,
        "pre_log": True,
        "alpha_model": "1/y",
        "gp_train_exclude_nsigma": 2.25,
        "blind_nsigma": 2.25,
        "n_restarts": 12,
    }
    for key, expected in required.items():
        if card.get(key) != expected:
            raise StudyError(f"reference-card semantic drift: {key}")
    if float(card["kernel_ls_res_upper_factor_by_dataset"]["2016"]) != 12.0:
        raise StudyError("reference-card 2016 upper factor drift")
    if float(card["kernel_ls_res_lower_factor_by_dataset"]["2016"]) != 0.9:
        raise StudyError("reference-card 2016 lower factor drift")
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
    if not np.allclose(np.diff(edges), spec["native_bin_width_GeV"], rtol=0, atol=5e-14):
        raise StudyError("native binning drift")
    return values, edges


def rebinned(values: np.ndarray, edges: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = float(spec["support_lower_MeV"]) / 1000.0
    upper = float(spec["support_upper_MeV"]) / 1000.0
    mask = (edges[:-1] >= lower - 1e-12) & (edges[1:] <= upper + 1e-12)
    index = np.flatnonzero(mask)
    if not len(index) or not np.all(np.diff(index) == 1):
        raise StudyError("support slice invalid")
    selected = values[index]
    factor = int(spec["rebin"])
    if selected.size % factor:
        raise StudyError("support violates rebin phase")
    counts = selected.reshape(-1, factor).sum(axis=1)
    native_edges = edges[index[0]:index[-1] + 2]
    coarse_edges = native_edges[::factor]
    if coarse_edges.size != counts.size + 1:
        coarse_edges = np.append(coarse_edges, native_edges[-1])
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    return np.asarray(centers), np.asarray(counts), np.asarray(coarse_edges)


def interval(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return (x >= low - 2e-13) & (x < high - 2e-13)


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coeffs = [float(item) for item in spec["sigma_coeffs_2016"]]
    m0 = float(spec["sigma_tail_m0_2016"])
    if mass <= m0:
        return float(sum(c * mass**i for i, c in enumerate(coeffs)))
    sigma0 = float(sum(c * m0**i for i, c in enumerate(coeffs)))
    return sigma0 + float(spec["sigma_tail_slope_override_2016"]) * (mass - m0)


def sigma_x(mass: float, spec: dict[str, Any]) -> float:
    return float(np.log((mass + sigma_2016(mass, spec)) / mass))


def length_bounds(mass: float, spec: dict[str, Any]) -> tuple[float, float, float]:
    local = sigma_x(mass, spec)
    grid = np.linspace(0.039, 0.180, int(spec["kernel_ls_res_npts"]))
    global_base = float(np.median([sigma_x(float(item), spec) for item in grid]))
    lower = float(spec["kernel_ls_res_lower_factor_2016"]) * local
    upper_factor = float(spec["kernel_ls_res_upper_factor_2016"])
    upper = max(upper_factor * local, upper_factor * global_base * float(spec["kernel_ls_local_hi_floor_factor"]))
    return float(lower), float(upper), float(local)


def masks_for_mass(centers: np.ndarray, mass: float, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    half = float(spec["blind_nsigma"]) * sigma_2016(mass, spec)
    low, high = mass - half, mass + half
    train = (centers < low) | (centers > high)
    score = (centers >= low) & (centers <= high)
    return train, score


@dataclass
class FitAttempt:
    mass_GeV: float
    seed: int
    finite_success: bool
    warning_free: bool
    lml: float
    constant: float
    length: float
    warnings: str
    error: str


def fit_model(x_train: np.ndarray, y_train: np.ndarray, mass: float, seed: int, spec: dict[str, Any]) -> tuple[FitAttempt, GaussianProcessRegressor | None]:
    lower, upper, _ = length_bounds(mass, spec)
    kernel = ConstantKernel(float(spec["kernel_constant_init"]), tuple(spec["kernel_constant_bounds"])) * RBF(math.sqrt(lower * upper), (lower, upper))
    if np.any(y_train <= 0):
        return FitAttempt(mass, seed, False, False, math.nan, math.nan, math.nan, "", "nonpositive training count"), None
    model = GaussianProcessRegressor(
        kernel=kernel, alpha=1.0 / y_train, n_restarts_optimizer=int(spec["n_restarts_optimizer"]),
        normalize_y=False, optimizer="fmin_l_bfgs_b", random_state=int(seed),
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(np.log(x_train).reshape(-1, 1), np.log(y_train))
        warning_text = " | ".join(f"{type(item.message).__name__}: {item.message}" for item in caught)
        constant = float(model.kernel_.k1.constant_value)
        length = float(np.asarray(model.kernel_.k2.length_scale).reshape(-1)[0])
        lml = float(model.log_marginal_likelihood_value_)
        finite = bool(np.isfinite(lml) and np.isfinite(constant) and constant > 0 and np.isfinite(length) and length > 0)
        return FitAttempt(mass, seed, finite, finite and not caught, lml, constant, length, warning_text, ""), model
    except Exception as exc:
        return FitAttempt(mass, seed, False, False, math.nan, math.nan, math.nan, "", repr(exc)), None


def lognormal_counts(mean_log: np.ndarray, cov_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.clip(np.diag(cov_log), 0.0, None)
    mean = np.exp(mean_log + 0.5 * diagonal)
    covariance = np.outer(mean, mean) * (np.exp(np.clip(cov_log, -40.0, 40.0)) - 1.0)
    return np.asarray(mean), np.asarray(covariance)


def covariance_metrics(mean: np.ndarray, gp_cov: np.ndarray, spec: dict[str, Any], observed: np.ndarray | None = None) -> dict[str, Any]:
    total = 0.5 * (gp_cov + gp_cov.T) + np.diag(np.clip(mean, 1e-12, None))
    total = 0.5 * (total + total.T)
    diagonal = np.diag(total)
    max_diag = float(np.max(diagonal))
    median_diag = float(np.median(diagonal))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(total)))
    negative_ok = minimum_eigenvalue >= -float(spec["covariance_negative_eigen_rel_tolerance"]) * max_diag
    chol = None
    jitter = math.nan
    for relative in (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        try:
            chol = np.linalg.cholesky(total + np.eye(len(mean)) * relative * median_diag)
            jitter = float(relative)
            break
        except np.linalg.LinAlgError:
            pass
    covariance_ok = bool(negative_ok and chol is not None and jitter <= float(spec["covariance_max_jitter_rel_median_diag"]))
    result = {
        "covariance_ok": covariance_ok, "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_diagonal": max_diag, "median_diagonal": median_diag,
        "jitter_relative_median_diagonal": jitter, "nlpd_per_bin": math.nan,
        "mahalanobis_per_bin": math.nan, "poisson_deviance_per_bin": math.nan,
        "max_abs_marginal_standardized_residual": math.nan,
    }
    if observed is None or chol is None:
        return result
    y = np.asarray(observed, dtype=float)
    residual = y - mean
    whitened = np.linalg.solve(chol, residual)
    mahal = float(whitened @ whitened)
    logdet = float(2 * np.log(np.diag(chol)).sum())
    terms = mean - y
    positive = y > 0
    terms[positive] += y[positive] * np.log(y[positive] / mean[positive])
    result.update({
        "nlpd_per_bin": float((mahal + logdet + len(y) * math.log(2 * math.pi)) / (2 * len(y))),
        "mahalanobis_per_bin": float(mahal / len(y)),
        "poisson_deviance_per_bin": float(2 * np.sum(terms) / len(y)),
        "max_abs_marginal_standardized_residual": float(np.max(np.abs(residual / np.sqrt(np.clip(diagonal, 1e-30, None))))),
    })
    return result


def fixed_certificate(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, mass: float, constant: float, length: float, recorded_lml: float | None, spec: dict[str, Any]) -> dict[str, Any]:
    lower, upper, _ = length_bounds(mass, spec)
    kernel = ConstantKernel(constant, tuple(spec["kernel_constant_bounds"])) * RBF(length, (lower, upper))
    model = GaussianProcessRegressor(kernel=kernel, alpha=1.0 / y_train, optimizer=None, normalize_y=False)
    model.fit(np.log(x_train).reshape(-1, 1), np.log(y_train))
    lml = float(model.log_marginal_likelihood_value_)
    theta = np.log([constant, length])
    lml_gradient, gradient = model.log_marginal_likelihood(theta=theta, eval_gradient=True, clone_kernel=True)
    mean_sklearn, cov_sklearn = model.predict(np.log(x_query).reshape(-1, 1), return_cov=True)

    train_log = np.log(x_train)
    query_log = np.log(x_query)
    distance_train = (train_log[:, None] - train_log[None, :]) / length
    matrix = constant * np.exp(-0.5 * distance_train**2) + np.diag(1.0 / y_train)
    chol = np.linalg.cholesky(matrix)
    cross = constant * np.exp(-0.5 * ((train_log[:, None] - query_log[None, :]) / length) ** 2)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, np.log(y_train)))
    mean_direct = cross.T @ alpha
    projected = np.linalg.solve(chol, cross)
    query_cov = constant * np.exp(-0.5 * ((query_log[:, None] - query_log[None, :]) / length) ** 2)
    cov_direct = 0.5 * ((query_cov - projected.T @ projected) + (query_cov - projected.T @ projected).T)
    prediction_closure = bool(
        np.allclose(mean_sklearn, mean_direct, rtol=float(spec["direct_prediction_relative_tolerance"]), atol=float(spec["direct_prediction_absolute_tolerance"]))
        and np.allclose(cov_sklearn, cov_direct, rtol=float(spec["direct_prediction_relative_tolerance"]), atol=float(spec["direct_prediction_absolute_tolerance"]))
    )
    mean_counts, cov_counts = lognormal_counts(np.asarray(mean_sklearn), np.asarray(cov_sklearn))
    cov_metrics = covariance_metrics(mean_counts, cov_counts, spec)

    bounds = [(math.log(float(spec["kernel_constant_bounds"][0])), math.log(float(spec["kernel_constant_bounds"][1]))), (math.log(lower), math.log(upper))]
    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        value_lml, value_grad = model.log_marginal_likelihood(theta=value, eval_gradient=True, clone_kernel=True)
        return -float(value_lml), -np.asarray(value_grad, dtype=float)
    options = spec["local_polish"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polished = minimize(objective, theta, method="L-BFGS-B", jac=True, bounds=bounds, options={
            "maxiter": int(options["maxiter"]), "maxls": int(options["maxls"]),
            "ftol": float(options["ftol"]), "gtol": float(options["gtol"]),
        })
    polished_coords = np.exp(polished.x)
    lml_improvement = float(-polished.fun - lml)
    movement_constant = float(abs(polished_coords[0] - constant) / constant)
    movement_length = float(abs(polished_coords[1] - length) / length)
    tolerance = float(spec["kernel_bound_rel_tolerance"])
    interior = bool(
        not np.isclose(constant, float(spec["kernel_constant_bounds"][0]), rtol=tolerance, atol=1e-12)
        and not np.isclose(constant, float(spec["kernel_constant_bounds"][1]), rtol=tolerance, atol=1e-12)
        and not np.isclose(length, lower, rtol=tolerance, atol=1e-12)
        and not np.isclose(length, upper, rtol=tolerance, atol=1e-12)
    )
    gradient_inf = float(np.max(np.abs(gradient)))
    recorded_closure = True if recorded_lml is None else abs(lml - float(recorded_lml)) <= float(spec["fixed_lml_abs_tolerance"])
    polish_pass = bool(
        polished.success and not caught and abs(lml_improvement) <= float(options["max_lml_improvement"])
        and movement_constant <= float(options["max_coordinate_relative_movement"])
        and movement_length <= float(options["max_coordinate_relative_movement"])
    )
    certificate_pass = bool(recorded_closure and prediction_closure and interior and cov_metrics["covariance_ok"] and gradient_inf < float(spec["analytic_gradient_infinity_max"]) and polish_pass)
    return {
        "fixed_lml": lml, "recorded_lml_difference": math.nan if recorded_lml is None else float(lml - float(recorded_lml)),
        "analytic_lml_difference": float(lml_gradient - lml), "gradient_constant_log": float(gradient[0]),
        "gradient_length_log": float(gradient[1]), "gradient_infinity": gradient_inf,
        "prediction_closure_pass": prediction_closure,
        "prediction_mean_max_abs_difference": float(np.max(np.abs(mean_sklearn - mean_direct))),
        "prediction_covariance_max_abs_difference": float(np.max(np.abs(cov_sklearn - cov_direct))),
        "prediction_mean_sha256": array_hash(mean_counts), "prediction_covariance_sha256": array_hash(cov_counts),
        "polish_success": bool(polished.success), "polish_status": int(polished.status),
        "polish_message": str(polished.message), "polish_warning_count": len(caught),
        "polish_lml_improvement": lml_improvement, "polish_constant_relative_movement": movement_constant,
        "polish_length_relative_movement": movement_length, "coordinates_interior": interior,
        "length_lower": lower, "length_upper": upper, **cov_metrics, "fixed_certificate_pass": certificate_pass,
    }


def select_repeated(attempts: list[FitAttempt], spec: dict[str, Any]) -> tuple[FitAttempt | None, int, int]:
    eligible = [item for item in attempts if item.warning_free and item.finite_success]
    if not eligible:
        return None, 0, 0
    selected = max(eligible, key=lambda item: item.lml)
    tol_lml = float(spec["lml_reproduction_abs_tolerance"])
    tol_coord = float(spec["coordinate_reproduction_rel_tolerance"])
    reproduced = [item for item in eligible if abs(item.lml - selected.lml) <= tol_lml and abs(item.constant - selected.constant) <= tol_coord * abs(selected.constant) and abs(item.length - selected.length) <= tol_coord * abs(selected.length)]
    return selected, len(eligible), len(reproduced)


def run_control(spec: dict[str, Any]) -> None:
    values, edges = load_histogram(spec)
    centers, counts, _ = rebinned(values, edges, spec)
    allowed = interval(centers, 0.030, 0.03875)
    blocks = {key: [float(x) for x in value] for key, value in spec["low_blocks"].items()}
    attempt_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for anchor in [float(item) for item in spec["control_kernel_anchors_GeV"]]:
        for block_name, (low, high) in blocks.items():
            score = allowed & interval(centers, low, high)
            train = allowed & ~score
            attempts: list[FitAttempt] = []
            models: dict[int, GaussianProcessRegressor] = {}
            for seed in spec["optimizer_seeds"]:
                attempt, model = fit_model(centers[train], counts[train], anchor, int(seed), spec)
                attempts.append(attempt)
                if model is not None:
                    models[int(seed)] = model
                attempt_rows.append({"phase": "control", "block": block_name, **asdict(attempt)})
            selected, eligible_count, reproduced_count = select_repeated(attempts, spec)
            if selected is None:
                cert = {"covariance_ok": False, "nlpd_per_bin": math.nan, "mahalanobis_per_bin": math.nan, "poisson_deviance_per_bin": math.nan, "max_abs_marginal_standardized_residual": math.nan}
                selected_seed = None
                constant = length = lml = math.nan
                at_bound = True
            else:
                model = models[selected.seed]
                mean_log, cov_log = model.predict(np.log(centers[score]).reshape(-1, 1), return_cov=True)
                mean, cov = lognormal_counts(np.asarray(mean_log), np.asarray(cov_log))
                cert = covariance_metrics(mean, cov, spec, counts[score])
                lower_bound, upper_bound, _ = length_bounds(anchor, spec)
                tol = float(spec["kernel_bound_rel_tolerance"])
                at_bound = bool(
                    np.isclose(selected.constant, spec["kernel_constant_bounds"][0], rtol=tol, atol=1e-12)
                    or np.isclose(selected.constant, spec["kernel_constant_bounds"][1], rtol=tol, atol=1e-12)
                    or np.isclose(selected.length, lower_bound, rtol=tol, atol=1e-12)
                    or np.isclose(selected.length, upper_bound, rtol=tol, atol=1e-12)
                )
                selected_seed, constant, length, lml = selected.seed, selected.constant, selected.length, selected.lml
            technical = bool(selected is not None and reproduced_count >= int(spec["warning_free_repeats_required"]) and not at_bound and cert["covariance_ok"])
            cell_rows.append({
                "phase": "control", "anchor_GeV": anchor, "block": block_name,
                "selected_seed": selected_seed, "selected_lml": lml, "selected_constant": constant,
                "selected_length": length, "warning_free_repeat_count": eligible_count,
                "reproduced_warning_free_count": reproduced_count, "kernel_at_bound": at_bound,
                "technical_pass": technical, **cert, "n_train": int(train.sum()), "n_score": int(score.sum()),
                "train_center_min_GeV": float(centers[train].min()), "train_center_max_GeV": float(centers[train].max()),
                "score_center_min_GeV": float(centers[score].min()), "score_center_max_GeV": float(centers[score].max()),
                "train_centers_sha256": array_hash(centers[train]), "score_centers_sha256": array_hash(centers[score]),
                "train_counts_sha256": array_hash(counts[train]), "score_counts_sha256": array_hash(counts[score]),
                "n_centers_at_or_above_38p75": int(np.count_nonzero(centers[train] >= 0.03875) + np.count_nonzero(centers[score] >= 0.03875)),
                "n_search_centers": int(np.count_nonzero(interval(centers[train], 0.039, 0.180001)) + np.count_nonzero(interval(centers[score], 0.039, 0.180001))),
            })
    out = HERE / "derived/control_adequacy"
    out.mkdir(parents=True, exist_ok=True)
    attempts_path, cells_path = out / "optimizer_attempts.csv", out / "selected_cells.csv"
    pd.DataFrame(attempt_rows).to_csv(attempts_path, index=False)
    cells_frame = pd.DataFrame(cell_rows)
    cells_frame.to_csv(cells_path, index=False)
    technical = bool(cells_frame["technical_pass"].astype(bool).all())
    mean_mahal = float(cells_frame["mahalanobis_per_bin"].mean())
    max_mahal = float(cells_frame["mahalanobis_per_bin"].max())
    max_marginal = float(cells_frame["max_abs_marginal_standardized_residual"].max())
    absolute = bool(mean_mahal < float(spec["mean_mahalanobis_per_bin_strict_max"]) and max_mahal < float(spec["individual_anchor_block_mahalanobis_per_bin_exclusive_max"]) and max_marginal < float(spec["max_abs_marginal_standardized_residual_exclusive_max"]))
    forbidden_zero = bool((cells_frame["n_centers_at_or_above_38p75"] == 0).all() and (cells_frame["n_search_centers"] == 0).all())
    status = "control_adequacy_pass" if technical and absolute and forbidden_zero else "stopped_control_adequacy_failure"
    manifest = {
        "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_card_status": spec["reference_status"], "technical_pass": technical,
        "absolute_guard_pass": absolute, "forbidden_centers_zero": forbidden_zero,
        "mean_mahalanobis_per_bin": mean_mahal, "maximum_cell_mahalanobis_per_bin": max_mahal,
        "maximum_abs_marginal_standardized_residual": max_marginal,
        "attempts": {"rows": len(attempt_rows), "sha256": sha256_file(attempts_path)},
        "cells": {"rows": len(cell_rows), "sha256": sha256_file(cells_path)},
        "protocol_sha256": EXPECTED_PROTOCOL_SHA, "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
        "selection_metrics_used": ["warning-free optimizer status", "LML/coordinate repeat", "kernel bounds", "covariance", "absolute control prediction"],
        "selection_metrics_forbidden": spec["forbidden_state_selection_metrics"],
    }
    json_write(out / "control_decision.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_archived_states(spec: dict[str, Any]) -> pd.DataFrame:
    allowed = ["dataset", "mass_GeV", "const_opt", "ls_opt", "lml", "ls_hi", "interpolated", "selected_source", "selected_source_sha256", "row_source", "optimizer_repair_applied", "review_status", "branch_multiplicity", "reproducing_sources", "selected_repair_reproduced", "repair_reproduction_pending", "candidate_count", "repair_candidate_count", "delta_lml_selected_minus_raw"]
    frame = pd.read_csv(REPO / spec["archived_states"]["path"], usecols=allowed)
    frame = frame.loc[frame["dataset"].astype(str) == "2016"].copy()
    expected = np.arange(39, 181) / 1000.0
    if len(frame) != 142 or not np.allclose(frame.sort_values("mass_GeV")["mass_GeV"], expected, rtol=0, atol=1e-12):
        raise StudyError("archived 2016 mass grid drift")
    if frame["interpolated"].astype(bool).any():
        raise StudyError("interpolated archived state")
    return frame.sort_values("mass_GeV").reset_index(drop=True)


def repair_source_reproduction(row: pd.Series, spec: dict[str, Any]) -> dict[str, Any]:
    sources = str(row["reproducing_sources"]).split("|")
    records = []
    for relative in sources:
        source = pd.read_csv(REPO / relative, usecols=["dataset", "mass_GeV", "const_opt", "ls_opt", "lml"])
        match = source.loc[(source["dataset"].astype(str) == "2016") & np.isclose(source["mass_GeV"], float(row["mass_GeV"]), rtol=0, atol=1e-12)]
        if len(match) != 1:
            raise StudyError(f"repair source row mismatch: {relative}")
        item = match.iloc[0]
        records.append({"source": relative, "source_sha256": sha256_file(REPO / relative), "constant": float(item["const_opt"]), "length": float(item["ls_opt"]), "lml": float(item["lml"])})
    best = max(records, key=lambda item: item["lml"])
    tol_lml = float(spec["lml_reproduction_abs_tolerance"])
    tol_coord = float(spec["coordinate_reproduction_rel_tolerance"])
    reproducing = [item for item in records if abs(item["lml"] - best["lml"]) <= tol_lml and abs(item["constant"] - best["constant"]) <= tol_coord * abs(best["constant"]) and abs(item["length"] - best["length"]) <= tol_coord * abs(best["length"])]
    selected_match = bool(abs(float(row["lml"]) - best["lml"]) <= tol_lml and abs(float(row["const_opt"]) - best["constant"]) <= tol_coord * abs(best["constant"]) and abs(float(row["ls_opt"]) - best["length"]) <= tol_coord * abs(best["length"]))
    return {"source_count": len(records), "reproducing_source_count": len(reproducing), "selected_matches_source_max": selected_match, "sources_json": json.dumps(records, sort_keys=True), "source_reproduction_pass": len(reproducing) >= 2 and selected_match}


def selected_source_geometry(relative: str, mass: float) -> dict[str, float]:
    source = pd.read_csv(
        REPO / relative,
        usecols=["dataset", "mass_GeV", "ls_lo", "ls_hi", "n_train"],
    )
    match = source.loc[
        (source["dataset"].astype(str) == "2016")
        & np.isclose(source["mass_GeV"], mass, rtol=0, atol=1e-12)
    ]
    if len(match) != 1:
        raise StudyError(f"selected-source geometry row mismatch: {relative}")
    item = match.iloc[0]
    return {
        "ls_lo": float(item["ls_lo"]),
        "ls_hi": float(item["ls_hi"]),
        "n_train": int(item["n_train"]),
    }


def run_archive(spec: dict[str, Any]) -> None:
    require_frozen(CONTROL_FREEZE, CONTROL_DECISION_RELATIVE)
    control = json.loads((HERE / CONTROL_DECISION_RELATIVE).read_text(encoding="utf-8"))
    if control["status"] != "control_adequacy_pass" or control["script_sha256"] != EXPECTED_CONTROL_SCRIPT_SHA:
        raise StudyError("initial control phase is not frozen/passing")
    if sha256_file(HERE / "run_control_frozen.py") != EXPECTED_CONTROL_SCRIPT_SHA:
        raise StudyError("frozen control execution code drifted")
    values, edges = load_histogram(spec)
    centers, counts, _ = rebinned(values, edges, spec)
    archived = load_archived_states(spec)
    repair_masses = set(int(item) for item in spec["archived_states"]["repair_class_masses_MeV"])
    rows = []
    for item in archived.itertuples(index=False):
        mass = float(item.mass_GeV)
        mass_mev = int(round(1000 * mass))
        train, query = masks_for_mass(centers, mass, spec)
        cert = fixed_certificate(centers[train], counts[train], centers[query], mass, float(item.const_opt), float(item.ls_opt), float(item.lml), spec)
        expected_lower, expected_upper, _ = length_bounds(mass, spec)
        source_geometry = selected_source_geometry(str(item.selected_source), mass)
        geometry_closure = bool(
            int(source_geometry["n_train"]) == int(train.sum())
            and math.isclose(float(source_geometry["ls_lo"]), expected_lower, rel_tol=2e-10, abs_tol=1e-12)
            and math.isclose(float(source_geometry["ls_hi"]), expected_upper, rel_tol=2e-10, abs_tol=1e-12)
        )
        source = REPO / str(item.selected_source)
        source_hash_ok = sha256_file(source) == str(item.selected_source_sha256) == spec["archived_states"]["sources"][str(item.selected_source)]
        base = {
            "dataset": "2016", "mass_GeV": mass, "mass_MeV": mass_mev,
            "provenance_class": "repair_three_source" if mass_mev in repair_masses else "raw_single_source",
            "archived_constant": float(item.const_opt), "archived_length": float(item.ls_opt), "archived_lml": float(item.lml),
            "archived_length_lower": float(source_geometry["ls_lo"]), "archived_length_upper": float(source_geometry["ls_hi"]),
            "archived_n_train": int(source_geometry["n_train"]), "geometry_closure_pass": geometry_closure,
            "selected_source": str(item.selected_source), "selected_source_sha256": str(item.selected_source_sha256),
            "selected_source_hash_ok": source_hash_ok, "row_source": str(item.row_source),
            "archived_branch_multiplicity": int(item.branch_multiplicity), "archived_interpolated": bool(item.interpolated),
            "n_train": int(train.sum()), "n_train_low": int(np.count_nonzero(centers < mass - float(spec["blind_nsigma"]) * sigma_2016(mass, spec))),
            "n_train_high": int(np.count_nonzero(centers > mass + float(spec["blind_nsigma"]) * sigma_2016(mass, spec))),
            "train_centers_sha256": array_hash(centers[train]), "train_counts_sha256": array_hash(counts[train]),
            "query_centers_sha256": array_hash(centers[query]), **cert,
        }
        if mass_mev in repair_masses:
            base.update(repair_source_reproduction(pd.Series(item._asdict()), spec))
        else:
            base.update({"source_count": 1, "reproducing_source_count": 1, "selected_matches_source_max": True, "sources_json": "", "source_reproduction_pass": False})
        base["archive_reuse_pass"] = bool(base["provenance_class"] == "repair_three_source" and base["selected_source_hash_ok"] and base["geometry_closure_pass"] and base["source_reproduction_pass"] and base["fixed_certificate_pass"] and base["n_train_low"] > 0 and base["n_train_high"] > 0)
        rows.append(base)
    out = HERE / "derived/archive_certification"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "archived_state_certificates.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    raw = frame.loc[frame["provenance_class"] == "raw_single_source"]
    repair = frame.loc[frame["provenance_class"] == "repair_three_source"]
    repair_reuse = bool(len(repair) == 3 and repair["archive_reuse_pass"].astype(bool).all())
    decision = {
        "status": "archive_classes_certified_for_rerun" if len(raw) == 139 else "stopped_archive_classification_failure",
        "created_utc": datetime.now(timezone.utc).isoformat(), "raw_class_rows": len(raw),
        "raw_class_action": "robust_repeat_all", "repair_class_rows": len(repair),
        "repair_class_action": "reuse_certified" if repair_reuse else "robust_repeat_all",
        "repair_class_reuse_pass": repair_reuse,
        "archive_certificate_rows": len(frame), "archive_certificates_sha256": sha256_file(path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA, "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
    }
    json_write(out / "archive_class_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def robust_worker(payload: tuple[float, int, np.ndarray, np.ndarray, dict[str, Any]]) -> dict[str, Any]:
    mass, seed, centers, counts, spec = payload
    train, _ = masks_for_mass(centers, mass, spec)
    attempt, _ = fit_model(centers[train], counts[train], mass, seed, spec)
    row = asdict(attempt)
    row.update({"n_train": int(train.sum()), "train_centers_sha256": array_hash(centers[train]), "train_counts_sha256": array_hash(counts[train])})
    return row


def run_robust(spec: dict[str, Any], workers: int) -> None:
    require_frozen(CONTROL_FREEZE, CONTROL_DECISION_RELATIVE)
    require_frozen(ARCHIVE_FREEZE, "derived/archive_certification/archive_class_decision.json")
    archive_decision = json.loads((HERE / "derived/archive_certification/archive_class_decision.json").read_text(encoding="utf-8"))
    if archive_decision["status"] != "archive_classes_certified_for_rerun" or archive_decision["script_sha256"] != sha256_file(Path(__file__)):
        raise StudyError("archive class decision is not frozen/passing or script drifted")
    values, edges = load_histogram(spec)
    centers, counts, _ = rebinned(values, edges, spec)
    archived = load_archived_states(spec)
    repair_masses = set(int(item) for item in spec["archived_states"]["repair_class_masses_MeV"])
    rerun_masses = [float(item) for item in archived.loc[~archived["mass_GeV"].mul(1000).round().astype(int).isin(repair_masses), "mass_GeV"]]
    if archive_decision["repair_class_action"] == "robust_repeat_all":
        rerun_masses.extend(float(item) / 1000.0 for item in sorted(repair_masses))
    rerun_masses = sorted(set(rerun_masses))
    payloads = [(mass, int(seed), centers, counts, spec) for mass in rerun_masses for seed in spec["optimizer_seeds"]]
    attempt_rows = []
    if workers == 1:
        attempt_rows = [robust_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(robust_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                attempt_rows.append(future.result())
    attempts_frame = pd.DataFrame(attempt_rows).sort_values(["mass_GeV", "seed"]).reset_index(drop=True)
    selected_rows = []
    for mass, group in attempts_frame.groupby("mass_GeV", sort=True):
        attempts = [FitAttempt(float(row.mass_GeV), int(row.seed), bool(row.finite_success), bool(row.warning_free), float(row.lml), float(row.constant), float(row.length), str(row.warnings) if pd.notna(row.warnings) else "", str(row.error) if pd.notna(row.error) else "") for row in group.itertuples(index=False)]
        selected, eligible_count, reproduced_count = select_repeated(attempts, spec)
        train, query = masks_for_mass(centers, float(mass), spec)
        if selected is None:
            cert = {"fixed_certificate_pass": False, "coordinates_interior": False, "covariance_ok": False}
            selected_seed = None
            constant = length = lml = math.nan
        else:
            cert = fixed_certificate(centers[train], counts[train], centers[query], float(mass), selected.constant, selected.length, selected.lml, spec)
            selected_seed, constant, length, lml = selected.seed, selected.constant, selected.length, selected.lml
        archived_row = archived.loc[np.isclose(archived["mass_GeV"], mass, rtol=0, atol=1e-12)].iloc[0]
        resolved = bool(selected is not None and eligible_count >= int(spec["warning_free_repeats_required"]) and reproduced_count >= int(spec["warning_free_repeats_required"]) and cert["fixed_certificate_pass"] and int(train.sum()) > 0 and np.count_nonzero(centers < mass - float(spec["blind_nsigma"]) * sigma_2016(float(mass), spec)) > 0 and np.count_nonzero(centers > mass + float(spec["blind_nsigma"]) * sigma_2016(float(mass), spec)) > 0)
        selected_rows.append({
            "dataset": "2016", "mass_GeV": float(mass), "mass_MeV": int(round(1000 * mass)),
            "state_source": "v4p9p11_robust_repeat", "selected_seed": selected_seed,
            "const_opt": constant, "ls_opt": length, "lml": lml,
            "warning_free_repeat_count": eligible_count, "reproduced_warning_free_count": reproduced_count,
            "archived_const_opt": float(archived_row["const_opt"]), "archived_ls_opt": float(archived_row["ls_opt"]), "archived_lml": float(archived_row["lml"]),
            "delta_lml_vs_archived": math.nan if selected is None else float(lml - float(archived_row["lml"])),
            "n_train": int(train.sum()), "n_train_low": int(np.count_nonzero(centers < mass - float(spec["blind_nsigma"]) * sigma_2016(float(mass), spec))),
            "n_train_high": int(np.count_nonzero(centers > mass + float(spec["blind_nsigma"]) * sigma_2016(float(mass), spec))),
            "train_centers_sha256": array_hash(centers[train]), "train_counts_sha256": array_hash(counts[train]),
            "query_centers_sha256": array_hash(centers[query]), **cert, "state_resolved": resolved,
        })
    out = HERE / "derived/robust_repeats"
    out.mkdir(parents=True, exist_ok=True)
    attempts_path, selected_path = out / "optimizer_attempts.csv", out / "selected_states.csv"
    attempts_frame.to_csv(attempts_path, index=False)
    selected_frame = pd.DataFrame(selected_rows).sort_values("mass_GeV")
    selected_frame.to_csv(selected_path, index=False)

    archive_cert = pd.read_csv(HERE / "derived/archive_certification/archived_state_certificates.csv")
    final_rows = []
    for archived_row in archive_cert.sort_values("mass_GeV").itertuples(index=False):
        mass = float(archived_row.mass_GeV)
        mass_mev = int(round(1000 * mass))
        match = selected_frame.loc[np.isclose(selected_frame["mass_GeV"], mass, rtol=0, atol=1e-12)]
        if len(match) == 1:
            row = match.iloc[0].to_dict()
        elif mass_mev in repair_masses and archive_decision["repair_class_action"] == "reuse_certified":
            row = {
                "dataset": "2016", "mass_GeV": mass, "mass_MeV": mass_mev,
                "state_source": "v4p1_repair_three_source_numerically_certified", "selected_seed": None,
                "const_opt": float(archived_row.archived_constant), "ls_opt": float(archived_row.archived_length), "lml": float(archived_row.archived_lml),
                "warning_free_repeat_count": 0, "reproduced_warning_free_count": 0,
                "historical_reproducing_source_count": int(archived_row.reproducing_source_count),
                "archived_const_opt": float(archived_row.archived_constant), "archived_ls_opt": float(archived_row.archived_length), "archived_lml": float(archived_row.archived_lml),
                "delta_lml_vs_archived": 0.0, "n_train": int(archived_row.n_train), "n_train_low": int(archived_row.n_train_low), "n_train_high": int(archived_row.n_train_high),
                "train_centers_sha256": archived_row.train_centers_sha256, "train_counts_sha256": archived_row.train_counts_sha256,
                "query_centers_sha256": archived_row.query_centers_sha256,
                "fixed_lml": float(archived_row.fixed_lml), "recorded_lml_difference": float(archived_row.recorded_lml_difference),
                "gradient_infinity": float(archived_row.gradient_infinity), "prediction_closure_pass": bool(archived_row.prediction_closure_pass),
                "prediction_mean_sha256": archived_row.prediction_mean_sha256, "prediction_covariance_sha256": archived_row.prediction_covariance_sha256,
                "polish_success": bool(archived_row.polish_success), "polish_lml_improvement": float(archived_row.polish_lml_improvement),
                "polish_constant_relative_movement": float(archived_row.polish_constant_relative_movement), "polish_length_relative_movement": float(archived_row.polish_length_relative_movement),
                "coordinates_interior": bool(archived_row.coordinates_interior), "covariance_ok": bool(archived_row.covariance_ok),
                "fixed_certificate_pass": bool(archived_row.fixed_certificate_pass), "state_resolved": bool(archived_row.archive_reuse_pass),
            }
        else:
            raise StudyError(f"missing final state: {mass_mev}")
        final_rows.append(row)
    final = pd.DataFrame(final_rows).sort_values("mass_GeV")
    final_path = HERE / "derived/observed_2016_gp_states_reviewed.csv"
    final.to_csv(final_path, index=False)
    all_resolved = bool(len(final) == 142 and final["state_resolved"].astype(bool).all())
    decision = {
        "status": "all_142_states_certified" if all_resolved else "stopped_unresolved_state",
        "created_utc": datetime.now(timezone.utc).isoformat(), "state_rows": len(final),
        "resolved_rows": int(final["state_resolved"].astype(bool).sum()),
        "unresolved_masses_MeV": final.loc[~final["state_resolved"].astype(bool), "mass_MeV"].astype(int).tolist(),
        "combination_authorized": all_resolved, "support_lower_MeV": 30, "support_upper_MeV": 210,
        "upper_length_factor_2016": 12, "reference_status": spec["reference_status"],
        "attempts": {"rows": len(attempts_frame), "sha256": sha256_file(attempts_path)},
        "robust_selected": {"rows": len(selected_frame), "sha256": sha256_file(selected_path)},
        "final_states": {"rows": len(final), "sha256": sha256_file(final_path)},
        "protocol_sha256": EXPECTED_PROTOCOL_SHA, "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
        "inference_fields_accessed": [], "selection_metrics_forbidden": spec["forbidden_state_selection_metrics"],
        "claim_boundary": "fixed-model asymptotic inference conditional on a partially unblinded model history; no unconditional coverage claim",
    }
    json_write(HERE / "derived/state_certification_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("archive", "robust"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    spec = load_spec()
    if args.mode == "archive":
        run_archive(spec)
    else:
        if not 1 <= args.workers <= 8:
            raise StudyError("workers must be 1..8")
        run_robust(spec, args.workers)


if __name__ == "__main__":
    main()
