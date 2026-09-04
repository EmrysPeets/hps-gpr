#!/usr/bin/env python3
"""Run the frozen low-only support CV and high-only technical check.

This program deliberately has no signal model, extraction, p-value, or limit
code path.  It verifies the immutable inputs and records exact training masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import uproot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = (
    REPO
    / "study_configs/v4p9p9_2016_full_sideband_predictive_support_20260902/study_spec.json"
)
AMENDMENT_PATH = (
    REPO
    / "study_configs/v4p9p9_2016_full_sideband_predictive_support_20260902/preexecution_amendment.json"
)
AMENDMENT2_PATH = (
    REPO
    / "study_configs/v4p9p9_2016_full_sideband_predictive_support_20260902/preexecution_amendment2.json"
)
PROTOCOL_PATH = HERE / "STUDY_PROTOCOL.md"
PROTOCOL_AMENDMENT_PATH = HERE / "PROTOCOL_AMENDMENT_PRE_EXECUTION.md"
PROTOCOL_AMENDMENT2_PATH = HERE / "PROTOCOL_AMENDMENT2_PRE_EXECUTION.md"

EXPECTED_HASHES = {
    PROTOCOL_PATH: "38e82537d04330ce66d5e39007df03bb1c7269fb62a16557e3ee96d6d2f380b2",
    SPEC_PATH: "f9f410977114cc9a6a9ea3ad381782a017b90cf2a86c7c8e3b2c9db89f3cfecd",
    PROTOCOL_AMENDMENT_PATH: "f45dbad8ee99f22d8500e4a8effd74b35854e2f694a8bd44ec4704e6b500d14c",
    AMENDMENT_PATH: "c90946c38d356c5c597627fb242f6437784da9ae2110daa4547b010f602ee0cd",
    PROTOCOL_AMENDMENT2_PATH: "d37a934e91d595123fe9ef543a3bb2ae7dce23aa78859d4747aa536360eb4b9e",
    AMENDMENT2_PATH: "0526dafba3094d5463225839f1e8ff8f94e011b83bcc8217a63ff5ee9cc6c768",
}


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_files() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise StudyError(f"missing frozen file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise StudyError(f"frozen file hash mismatch: {path}: {actual} != {expected}")


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coeffs = [float(value) for value in spec["sigma_coeffs_2016"]]
    m = float(mass)
    m0 = float(spec["sigma_tail_m0_2016"])
    value0 = sum(value * m**power for power, value in enumerate(coeffs))
    if m <= m0:
        return float(value0)
    sigma0 = sum(value * m0**power for power, value in enumerate(coeffs))
    slope = float(spec["sigma_tail_slope_override_2016"])
    return float(sigma0 + slope * (m - m0))


def sigma_x(mass: float, spec: dict[str, Any]) -> float:
    sigma = sigma_2016(mass, spec)
    return float(np.log((float(mass) + sigma) / float(mass)))


def length_bounds(anchor: float, spec: dict[str, Any]) -> tuple[float, float, float]:
    base = sigma_x(anchor, spec)
    grid = np.linspace(0.039, 0.180, int(spec["kernel_ls_res_npts"]))
    global_base = float(np.median([sigma_x(float(m), spec) for m in grid]))
    lower = float(spec["kernel_ls_res_lower_factor_2016"]) * base
    upper_factor = float(spec["kernel_ls_res_upper_factor_2016"])
    upper = upper_factor * base
    upper = max(
        upper,
        upper_factor
        * global_base
        * float(spec["kernel_ls_local_hi_floor_factor"]),
    )
    return float(lower), float(upper), float(base)


def load_histogram(stage: str, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key = "development" if stage == "development" else "confirmation"
    declaration = dict(spec[key])
    path = REPO / str(declaration["path"])
    if sha256_file(path) != declaration["file_sha256"]:
        raise StudyError(f"input ROOT hash mismatch for {stage}")
    with uproot.open(path) as handle:
        values, edges = handle[str(declaration["histogram"])].to_numpy(flow=False)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if histogram_hash(values, edges) != declaration["histogram_sha256"]:
        raise StudyError(f"input histogram hash mismatch for {stage}")
    widths = np.diff(edges)
    if not np.allclose(
        widths,
        float(spec["native_bin_width_GeV"]),
        rtol=0.0,
        atol=5.0e-14,
    ):
        raise StudyError("unexpected native binning")
    return values, edges, declaration


def rebin_support(
    values: np.ndarray,
    edges: np.ndarray,
    lower: float,
    upper: float,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tolerance = 1.0e-12
    native = (edges[:-1] >= lower - tolerance) & (edges[1:] <= upper + tolerance)
    indices = np.flatnonzero(native)
    if indices.size == 0 or not np.all(np.diff(indices) == 1):
        raise StudyError(f"non-contiguous native support [{lower}, {upper}]")
    selected = values[indices]
    if selected.size % factor:
        raise StudyError(f"native support does not divide by rebin={factor}")
    rebinned = selected.reshape(-1, factor).sum(axis=1)
    edge_start = int(indices[0])
    edge_stop = int(indices[-1]) + 1
    selected_edges = edges[edge_start : edge_stop + 1]
    coarse_edges = selected_edges[::factor]
    if coarse_edges.size != rebinned.size + 1:
        coarse_edges = np.append(coarse_edges, selected_edges[-1])
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    if not np.allclose(np.diff(coarse_edges), factor * np.median(np.diff(edges)), atol=1e-12):
        raise StudyError("unexpected coarse bin widths")
    return np.asarray(centers), np.asarray(rebinned), np.asarray(coarse_edges)


def in_interval(values: np.ndarray, low: float, high: float) -> np.ndarray:
    tolerance = 2.0e-13
    return (values >= low - tolerance) & (values < high - tolerance)


@dataclass
class FitAttempt:
    seed: int
    fit: GaussianProcessRegressor | None
    success: bool
    lml: float
    const_opt: float
    length_opt: float
    warning_text: str
    error_text: str


def fit_once(
    x_train: np.ndarray,
    y_train: np.ndarray,
    anchor: float,
    seed: int,
    spec: dict[str, Any],
) -> FitAttempt:
    lower, upper, _ = length_bounds(anchor, spec)
    kernel = ConstantKernel(
        float(spec["kernel_constant_init"]),
        tuple(float(value) for value in spec["kernel_constant_bounds"]),
    ) * RBF(length_scale=math.sqrt(lower * upper), length_scale_bounds=(lower, upper))
    x_in = np.log(np.clip(np.asarray(x_train, dtype=float), 1e-12, None))
    y = np.asarray(y_train, dtype=float)
    if np.any(y <= 0.0):
        return FitAttempt(seed, None, False, math.nan, math.nan, math.nan, "", "nonpositive training count")
    y_in = np.log(y)
    alpha = 1.0 / y
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=int(spec["n_restarts_optimizer"]),
        optimizer="fmin_l_bfgs_b",
        normalize_y=False,
        random_state=int(seed),
    )
    captured: list[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(x_in.reshape(-1, 1), y_in)
        optimized = model.kernel_
        constant = float(optimized.k1.constant_value)
        length = float(np.asarray(optimized.k2.length_scale).reshape(-1)[0])
        lml = float(model.log_marginal_likelihood_value_)
        success = bool(
            np.isfinite(lml)
            and np.isfinite(constant)
            and constant > 0.0
            and np.isfinite(length)
            and length > 0.0
        )
        return FitAttempt(
            int(seed),
            model,
            success,
            lml,
            constant,
            length,
            " | ".join(str(item.message) for item in captured),
            "",
        )
    except Exception as exc:
        return FitAttempt(int(seed), None, False, math.nan, math.nan, math.nan, "", repr(exc))


def predict_count_space(
    model: GaussianProcessRegressor,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.log(np.clip(np.asarray(x_test, dtype=float), 1e-12, None))
    mean_log, covariance_log = model.predict(query.reshape(-1, 1), return_cov=True)
    mean_log = np.asarray(mean_log, dtype=float).reshape(-1)
    covariance_log = np.asarray(covariance_log, dtype=float)
    diagonal = np.clip(np.diag(covariance_log), 0.0, None)
    mean = np.exp(mean_log + 0.5 * diagonal)
    covariance = np.outer(mean, mean) * (
        np.exp(np.clip(covariance_log, -40.0, 40.0)) - 1.0
    )
    return np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)


def score_prediction(
    observed: np.ndarray,
    mean: np.ndarray,
    gp_covariance: np.ndarray,
    spec: dict[str, Any],
) -> dict[str, Any]:
    y = np.asarray(observed, dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(gp_covariance, dtype=float)
    if y.size == 0 or mu.size != y.size or covariance.shape != (y.size, y.size):
        raise StudyError("invalid prediction dimensions")
    covariance = 0.5 * (covariance + covariance.T)
    total = covariance + np.diag(np.clip(mu, 1.0e-12, None))
    total = 0.5 * (total + total.T)
    diagonal = np.diag(total)
    scale_max = float(np.max(diagonal))
    scale_median = float(np.median(diagonal))
    eigenvalues = np.linalg.eigvalsh(total)
    min_eigenvalue = float(np.min(eigenvalues))
    negative_ok = bool(
        min_eigenvalue
        >= -float(spec["covariance_negative_eigen_rel_tolerance"]) * scale_max
    )

    jitter_relative = 0.0
    chol = None
    for relative in (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        try:
            trial = total + np.eye(y.size) * (relative * scale_median)
            chol = np.linalg.cholesky(trial)
            jitter_relative = float(relative)
            total = trial
            break
        except np.linalg.LinAlgError:
            continue
    covariance_ok = bool(
        chol is not None
        and negative_ok
        and jitter_relative
        <= float(spec["covariance_max_jitter_rel_median_diag"])
    )
    if chol is None:
        return {
            "covariance_ok": False,
            "min_eigenvalue": min_eigenvalue,
            "max_diagonal": scale_max,
            "median_diagonal": scale_median,
            "jitter_relative_median_diag": math.nan,
            "nlpd_per_bin": math.nan,
            "mahalanobis_per_bin": math.nan,
            "poisson_deviance_per_bin": math.nan,
            "max_abs_marginal_standardized_residual": math.nan,
        }

    residual = y - mu
    whitened = np.linalg.solve(chol, residual)
    mahalanobis = float(np.dot(whitened, whitened))
    logdet = float(2.0 * np.log(np.diag(chol)).sum())
    nlpd = float(0.5 * (mahalanobis + logdet + y.size * math.log(2.0 * math.pi)))
    positive = y > 0.0
    terms = np.asarray(mu - y, dtype=float)
    terms[positive] += y[positive] * np.log(y[positive] / mu[positive])
    poisson_deviance = float(2.0 * np.sum(terms))
    marginal = residual / np.sqrt(np.clip(np.diag(total), 1.0e-30, None))
    return {
        "covariance_ok": covariance_ok,
        "min_eigenvalue": min_eigenvalue,
        "max_diagonal": scale_max,
        "median_diagonal": scale_median,
        "jitter_relative_median_diag": jitter_relative,
        "nlpd_per_bin": nlpd / y.size,
        "mahalanobis_per_bin": mahalanobis / y.size,
        "poisson_deviance_per_bin": poisson_deviance / y.size,
        "max_abs_marginal_standardized_residual": float(np.max(np.abs(marginal))),
    }


def interval_blocks(spec: dict[str, Any], prefix: str) -> list[tuple[str, float, float]]:
    blocks = []
    for name, interval in spec["blocks"].items():
        if str(name).startswith(prefix):
            blocks.append((str(name), float(interval[0]), float(interval[1])))
    return sorted(blocks)


def evaluate_cell(
    *,
    stage: str,
    region: str,
    support_lower_mev: int | None,
    anchor: float,
    block: tuple[str, float, float],
    centers: np.ndarray,
    counts: np.ndarray,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block_name, block_low, block_high = block
    if region == "low":
        if support_lower_mev is None:
            raise StudyError("low cell requires support lower edge")
        support_low = float(support_lower_mev) / 1000.0
        allowed = in_interval(centers, support_low, 0.03875)
    elif region == "high":
        support_low = math.nan
        allowed = in_interval(centers, 0.181, 0.210)
    else:
        raise StudyError(f"unknown region: {region}")

    holdout = in_interval(centers, block_low, block_high)
    train = allowed & ~holdout
    score = allowed & holdout
    forbidden = in_interval(centers, 0.039, np.nextafter(0.180, math.inf))
    n_forbidden_train = int(np.count_nonzero(train & forbidden))
    n_forbidden_score = int(np.count_nonzero(score & forbidden))
    if n_forbidden_train != 0 or n_forbidden_score != 0:
        raise StudyError("search-region center entered training or scoring")
    if not np.any(train) or not np.any(score):
        raise StudyError(f"empty train/score mask for {region} {block_name}")

    x_train = centers[train]
    y_train = counts[train]
    x_score = centers[score]
    y_score = counts[score]
    attempts: list[FitAttempt] = []
    for seed in (int(value) for value in spec["optimizer_seeds"]):
        attempts.append(fit_once(x_train, y_train, anchor, seed, spec))

    attempt_rows: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_rows.append(
            {
                "stage": stage,
                "region": region,
                "support_lower_MeV": support_lower_mev,
                "anchor_GeV": anchor,
                "block": block_name,
                "seed": attempt.seed,
                "success": attempt.success,
                "lml": attempt.lml,
                "const_opt": attempt.const_opt,
                "length_opt": attempt.length_opt,
                "warning_text": attempt.warning_text,
                "error_text": attempt.error_text,
            }
        )
    successful = [item for item in attempts if item.success and item.fit is not None]
    if not successful:
        selected = None
    else:
        selected = max(successful, key=lambda item: item.lml)

    lower, upper, base = length_bounds(anchor, spec)
    const_lower, const_upper = [float(value) for value in spec["kernel_constant_bounds"]]
    all_repeats_success = len(successful) == len(attempts)
    reproduced = []
    if selected is not None:
        for attempt in successful:
            lml_close = abs(attempt.lml - selected.lml) <= float(
                spec["lml_reproduction_abs_tolerance"]
            )
            length_close = abs(attempt.length_opt - selected.length_opt) <= float(
                spec["length_reproduction_rel_tolerance"]
            ) * max(abs(selected.length_opt), 1e-30)
            if lml_close and length_close:
                reproduced.append(attempt)
    reproduction_pass = len(reproduced) >= 2

    if selected is None:
        metrics = {
            "covariance_ok": False,
            "min_eigenvalue": math.nan,
            "max_diagonal": math.nan,
            "median_diagonal": math.nan,
            "jitter_relative_median_diag": math.nan,
            "nlpd_per_bin": math.nan,
            "mahalanobis_per_bin": math.nan,
            "poisson_deviance_per_bin": math.nan,
            "max_abs_marginal_standardized_residual": math.nan,
        }
        const_at_bound = True
        length_at_bound = True
        selected_seed = None
        selected_lml = math.nan
        selected_const = math.nan
        selected_length = math.nan
    else:
        mean, covariance = predict_count_space(selected.fit, x_score)
        metrics = score_prediction(y_score, mean, covariance, spec)
        tolerance = float(spec["kernel_bound_rel_tolerance"])
        const_at_bound = bool(
            np.isclose(selected.const_opt, const_lower, rtol=tolerance, atol=1e-12)
            or np.isclose(selected.const_opt, const_upper, rtol=tolerance, atol=1e-12)
        )
        length_at_bound = bool(
            np.isclose(selected.length_opt, lower, rtol=tolerance, atol=1e-12)
            or np.isclose(selected.length_opt, upper, rtol=tolerance, atol=1e-12)
        )
        selected_seed = int(selected.seed)
        selected_lml = float(selected.lml)
        selected_const = float(selected.const_opt)
        selected_length = float(selected.length_opt)

    cell_mahal_pass = bool(
        np.isfinite(metrics["mahalanobis_per_bin"])
        and metrics["mahalanobis_per_bin"]
        < float(load_json(AMENDMENT2_PATH)["absolute_predictive_adequacy"]["individual_anchor_block_mahalanobis_per_bin_exclusive_max"])
    )
    marginal_pass = bool(
        np.isfinite(metrics["max_abs_marginal_standardized_residual"])
        and metrics["max_abs_marginal_standardized_residual"]
        < float(load_json(AMENDMENT2_PATH)["absolute_predictive_adequacy"]["max_abs_marginal_standardized_residual_exclusive_max"])
    )
    technical_pass = bool(
        all_repeats_success
        and reproduction_pass
        and not const_at_bound
        and not length_at_bound
        and bool(metrics["covariance_ok"])
        and n_forbidden_train == 0
        and n_forbidden_score == 0
    )
    selected_row = {
        "stage": stage,
        "region": region,
        "support_lower_MeV": support_lower_mev,
        "support_upper_MeV": 210,
        "anchor_GeV": anchor,
        "block": block_name,
        "block_low_GeV": block_low,
        "block_high_GeV": block_high,
        "selected_seed": selected_seed,
        "selected_lml": selected_lml,
        "selected_const": selected_const,
        "selected_length": selected_length,
        "length_lower": lower,
        "length_upper": upper,
        "sigma_x": base,
        "all_repeats_success": all_repeats_success,
        "n_reproduced_branches": len(reproduced),
        "reproduction_pass": reproduction_pass,
        "const_at_bound": const_at_bound,
        "length_at_bound": length_at_bound,
        "covariance_ok": metrics["covariance_ok"],
        "min_eigenvalue": metrics["min_eigenvalue"],
        "max_diagonal": metrics["max_diagonal"],
        "median_diagonal": metrics["median_diagonal"],
        "jitter_relative_median_diag": metrics["jitter_relative_median_diag"],
        "nlpd_per_bin": metrics["nlpd_per_bin"],
        "mahalanobis_per_bin": metrics["mahalanobis_per_bin"],
        "poisson_deviance_per_bin": metrics["poisson_deviance_per_bin"],
        "max_abs_marginal_standardized_residual": metrics[
            "max_abs_marginal_standardized_residual"
        ],
        "cell_mahalanobis_guard_pass": cell_mahal_pass,
        "cell_marginal_residual_guard_pass": marginal_pass,
        "technical_pass": technical_pass,
        "n_train": int(np.count_nonzero(train)),
        "n_score": int(np.count_nonzero(score)),
        "train_center_min_GeV": float(np.min(x_train)),
        "train_center_max_GeV": float(np.max(x_train)),
        "score_center_min_GeV": float(np.min(x_score)),
        "score_center_max_GeV": float(np.max(x_score)),
        "train_centers_sha256": array_hash(x_train),
        "score_centers_sha256": array_hash(x_score),
        "train_counts_sha256": array_hash(y_train),
        "score_counts_sha256": array_hash(y_score),
        "n_forbidden_search_train_centers": n_forbidden_train,
        "n_forbidden_search_score_centers": n_forbidden_score,
        "training_mask_definition": (
            "[edge,38.75) minus held-out low block; low only"
            if region == "low"
            else "[181,210) minus held-out high block; high only"
        ),
    }
    return attempt_rows, selected_row


def parse_supports(value: str, spec: dict[str, Any], stage: str) -> list[int]:
    if stage == "development":
        expected = [int(item) for item in spec["eligible_lower_edges_MeV"]] + [
            int(spec["geometry_control_lower_edge_MeV"])
        ]
        if value:
            provided = [int(item) for item in value.split(",") if item.strip()]
            if provided != expected:
                raise StudyError(f"development supports must be exactly {expected}")
        return expected
    provided = [int(item) for item in value.split(",") if item.strip()]
    eligible = set(int(item) for item in spec["eligible_lower_edges_MeV"])
    if not provided or 30 not in provided or not set(provided).issubset(eligible):
        raise StudyError("confirmation supports must be an eligible set containing 30")
    return sorted(set(provided))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument(
        "--supports",
        default="",
        help="Comma-separated lower edges; development is forced to 29,30,31,32,33,34",
    )
    args = parser.parse_args()

    verify_frozen_files()
    spec = load_json(SPEC_PATH)
    supports = parse_supports(args.supports, spec, args.stage)
    values, edges, declaration = load_histogram(args.stage, spec)
    factor = int(spec["rebin"])
    anchors = [float(item) for item in spec["kernel_anchors_GeV"]]
    low_blocks = interval_blocks(spec, "L")
    high_blocks = interval_blocks(spec, "H")
    out = HERE / "derived" / args.stage
    out.mkdir(parents=True, exist_ok=True)

    attempt_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for lower_mev in supports:
        centers, counts, _ = rebin_support(
            values,
            edges,
            lower=float(lower_mev) / 1000.0,
            upper=0.210,
            factor=factor,
        )
        for anchor in anchors:
            for block in low_blocks:
                attempts, selected = evaluate_cell(
                    stage=args.stage,
                    region="low",
                    support_lower_mev=lower_mev,
                    anchor=anchor,
                    block=block,
                    centers=centers,
                    counts=counts,
                    spec=spec,
                )
                attempt_rows.extend(attempts)
                selected_rows.append(selected)

    # High-only fits are candidate independent.  Rebin from 30 MeV to preserve
    # the reviewed integer-MeV production phase, but use high centers only.
    centers, counts, _ = rebin_support(
        values, edges, lower=0.030, upper=0.210, factor=factor
    )
    for anchor in anchors:
        for block in high_blocks:
            attempts, selected = evaluate_cell(
                stage=args.stage,
                region="high",
                support_lower_mev=None,
                anchor=anchor,
                block=block,
                centers=centers,
                counts=counts,
                spec=spec,
            )
            attempt_rows.extend(attempts)
            selected_rows.append(selected)

    attempts_frame = pd.DataFrame(attempt_rows)
    selected_frame = pd.DataFrame(selected_rows)
    attempts_path = out / "optimizer_attempts.csv"
    selected_path = out / "selected_predictive_scores.csv"
    attempts_frame.to_csv(attempts_path, index=False)
    selected_frame.to_csv(selected_path, index=False)

    forbidden_train = int(selected_frame["n_forbidden_search_train_centers"].sum())
    forbidden_score = int(selected_frame["n_forbidden_search_score_centers"].sum())
    if forbidden_train != 0 or forbidden_score != 0:
        raise StudyError("nonzero forbidden search-center count in completed ledger")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "supports": supports,
        "candidate_specific_regions_bridged": False,
        "search_bins_used_for_training": forbidden_train,
        "search_bins_used_for_scoring": forbidden_score,
        "input": declaration,
        "input_total_counts": float(np.sum(values)),
        "protocol_chain_sha256": {
            str(path.relative_to(REPO)): expected for path, expected in EXPECTED_HASHES.items()
        },
        "script_sha256": sha256_file(Path(__file__)),
        "outputs": {
            "optimizer_attempts.csv": {
                "rows": int(len(attempts_frame)),
                "sha256": sha256_file(attempts_path),
            },
            "selected_predictive_scores.csv": {
                "rows": int(len(selected_frame)),
                "sha256": sha256_file(selected_path),
            },
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "uproot": uproot.__version__,
        },
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
