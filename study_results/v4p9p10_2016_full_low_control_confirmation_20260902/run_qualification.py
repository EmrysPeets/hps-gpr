#!/usr/bin/env python3
"""Run frozen v4.9.10 length-factor or full low-control qualification."""

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
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import uproot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = REPO / "study_configs/v4p9p10_2016_full_low_control_confirmation_20260902/study_spec.json"
PROTOCOL_PATH = HERE / "STUDY_PROTOCOL.md"
EXPECTED_PROTOCOL_SHA = "3bc17d683faf50195b632416a7cbb96fb5463a93d714fcb3bff45ef5f2ec8d84"
EXPECTED_SPEC_SHA = "680444ef63267cabd88830c0cd5e54ee40b495e8caa3a1b30b0c0ed1a016e33e"
ANALYSIS_SCRIPT_PATH = HERE / "analyze_qualification.py"
FACTOR_FREEZE_PATH = HERE / "FROZEN_LENGTH_FACTOR_SHA256"


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


def load_spec() -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA:
        raise StudyError("frozen protocol hash mismatch")
    if sha256_file(SPEC_PATH) != EXPECTED_SPEC_SHA:
        raise StudyError("frozen spec hash mismatch")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    terminal = spec["v4p9p9_terminal_evidence"]
    for key, hash_key in (
        ("phase1_decision", "phase1_decision_sha256"),
        ("validation", "validation_sha256"),
        ("development_scores", "development_scores_sha256"),
    ):
        path = REPO / terminal[key]
        if sha256_file(path) != terminal[hash_key]:
            raise StudyError(f"v4p9p9 terminal evidence drift: {key}")
    if (REPO / "study_results/v4p9p9_2016_full_sideband_predictive_support_20260902/derived/confirmation").exists():
        raise StudyError("v4p9p9 unexpectedly contains full confirmation")
    return spec


def load_histogram(kind: str, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    declaration = dict(spec[f"{kind}_input"])
    path = REPO / declaration["path"]
    if sha256_file(path) != declaration["file_sha256"]:
        raise StudyError(f"{kind} ROOT hash mismatch")
    with uproot.open(path) as handle:
        values, edges = handle[declaration["histogram"]].to_numpy(flow=False)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if histogram_hash(values, edges) != declaration["histogram_sha256"]:
        raise StudyError(f"{kind} histogram hash mismatch")
    if not np.allclose(
        np.diff(edges), float(spec["native_bin_width_GeV"]), rtol=0.0, atol=5e-14
    ):
        raise StudyError("native binning drift")
    return values, edges, declaration


def rebin_support(
    values: np.ndarray, edges: np.ndarray, lower: float, upper: float, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (edges[:-1] >= lower - 1e-12) & (edges[1:] <= upper + 1e-12)
    indices = np.flatnonzero(mask)
    if not len(indices) or not np.all(np.diff(indices) == 1):
        raise StudyError("invalid support slice")
    selected = values[indices]
    if selected.size % factor:
        raise StudyError("support does not preserve rebin phase")
    counts = selected.reshape(-1, factor).sum(axis=1)
    native_edges = edges[indices[0] : indices[-1] + 2]
    coarse_edges = native_edges[::factor]
    if coarse_edges.size != counts.size + 1:
        coarse_edges = np.append(coarse_edges, native_edges[-1])
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    return np.asarray(centers), np.asarray(counts), np.asarray(coarse_edges)


def interval(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return (x >= low - 2e-13) & (x < high - 2e-13)


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coeffs = [float(item) for item in spec["sigma_coeffs_2016"]]
    m = float(mass)
    m0 = float(spec["sigma_tail_m0_2016"])
    if m <= m0:
        return float(sum(c * m**i for i, c in enumerate(coeffs)))
    sigma0 = float(sum(c * m0**i for i, c in enumerate(coeffs)))
    return sigma0 + float(spec["sigma_tail_slope_override_2016"]) * (m - m0)


def sigma_x(mass: float, spec: dict[str, Any]) -> float:
    return float(np.log((mass + sigma_2016(mass, spec)) / mass))


def length_bounds(anchor: float, factor: float, spec: dict[str, Any]) -> tuple[float, float, float]:
    base = sigma_x(float(anchor), spec)
    grid = np.linspace(0.039, 0.180, int(spec["kernel_ls_res_npts"]))
    global_base = float(np.median([sigma_x(float(item), spec) for item in grid]))
    lower = float(spec["kernel_ls_res_lower_factor_2016"]) * base
    upper = max(
        float(factor) * base,
        float(factor) * global_base * float(spec["kernel_ls_local_hi_floor_factor"]),
    )
    return lower, upper, base


@dataclass
class Attempt:
    seed: int
    model: GaussianProcessRegressor | None
    finite_success: bool
    warning_free: bool
    lml: float
    constant: float
    length: float
    warning_text: str
    error_text: str


def fit_once(
    x_train: np.ndarray,
    y_train: np.ndarray,
    anchor: float,
    factor: float,
    seed: int,
    spec: dict[str, Any],
) -> Attempt:
    lower, upper, _ = length_bounds(anchor, factor, spec)
    kernel = ConstantKernel(
        float(spec["kernel_constant_init"]),
        tuple(float(item) for item in spec["kernel_constant_bounds"]),
    ) * RBF(math.sqrt(lower * upper), (lower, upper))
    y = np.asarray(y_train, dtype=float)
    if np.any(y <= 0):
        return Attempt(seed, None, False, False, math.nan, math.nan, math.nan, "", "nonpositive training count")
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0 / y,
        n_restarts_optimizer=int(spec["n_restarts_optimizer"]),
        normalize_y=False,
        optimizer="fmin_l_bfgs_b",
        random_state=int(seed),
    )
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(np.log(x_train).reshape(-1, 1), np.log(y))
        warning_text = " | ".join(str(item.message) for item in captured)
        constant = float(model.kernel_.k1.constant_value)
        length = float(np.asarray(model.kernel_.k2.length_scale).reshape(-1)[0])
        lml = float(model.log_marginal_likelihood_value_)
        finite = bool(
            np.isfinite(lml) and np.isfinite(constant) and constant > 0
            and np.isfinite(length) and length > 0
        )
        return Attempt(
            int(seed), model, finite, finite and len(captured) == 0,
            lml, constant, length, warning_text, "",
        )
    except Exception as exc:
        return Attempt(int(seed), None, False, False, math.nan, math.nan, math.nan, "", repr(exc))


def count_prediction(
    model: GaussianProcessRegressor, x_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean_log, cov_log = model.predict(np.log(x_score).reshape(-1, 1), return_cov=True)
    mean_log = np.asarray(mean_log, dtype=float).reshape(-1)
    cov_log = np.asarray(cov_log, dtype=float)
    diagonal = np.clip(np.diag(cov_log), 0.0, None)
    mean = np.exp(mean_log + 0.5 * diagonal)
    cov = np.outer(mean, mean) * (np.exp(np.clip(cov_log, -40.0, 40.0)) - 1.0)
    return np.asarray(mean), np.asarray(cov)


def covariance_and_score(
    mean: np.ndarray,
    gp_cov: np.ndarray,
    spec: dict[str, Any],
    observed: np.ndarray | None,
) -> dict[str, Any]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    total = 0.5 * (gp_cov + gp_cov.T) + np.diag(np.clip(mean, 1e-12, None))
    total = 0.5 * (total + total.T)
    diagonal = np.diag(total)
    max_diag = float(np.max(diagonal))
    median_diag = float(np.median(diagonal))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(total)))
    negative_ok = minimum_eigenvalue >= -float(
        spec["covariance_negative_eigen_rel_tolerance"]
    ) * max_diag
    chol = None
    jitter = math.nan
    for relative in (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        try:
            chol = np.linalg.cholesky(total + np.eye(len(mean)) * relative * median_diag)
            jitter = float(relative)
            break
        except np.linalg.LinAlgError:
            pass
    covariance_ok = bool(
        negative_ok and chol is not None
        and jitter <= float(spec["covariance_max_jitter_rel_median_diag"])
    )
    result: dict[str, Any] = {
        "covariance_ok": covariance_ok,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_diagonal": max_diag,
        "median_diagonal": median_diag,
        "jitter_relative_median_diagonal": jitter,
        "nlpd_per_bin": math.nan,
        "mahalanobis_per_bin": math.nan,
        "poisson_deviance_per_bin": math.nan,
        "max_abs_marginal_standardized_residual": math.nan,
    }
    if observed is None or chol is None:
        return result
    y = np.asarray(observed, dtype=float).reshape(-1)
    residual = y - mean
    whitened = np.linalg.solve(chol, residual)
    mahal = float(whitened @ whitened)
    logdet = float(2 * np.log(np.diag(chol)).sum())
    nlpd = 0.5 * (mahal + logdet + len(y) * math.log(2 * math.pi))
    terms = mean - y
    positive = y > 0
    terms[positive] += y[positive] * np.log(y[positive] / mean[positive])
    marginal = residual / np.sqrt(np.clip(np.diag(total), 1e-30, None))
    result.update(
        {
            "nlpd_per_bin": float(nlpd / len(y)),
            "mahalanobis_per_bin": float(mahal / len(y)),
            "poisson_deviance_per_bin": float(2 * np.sum(terms) / len(y)),
            "max_abs_marginal_standardized_residual": float(np.max(np.abs(marginal))),
        }
    )
    return result


def evaluate_cell(
    *,
    mode: str,
    support: int,
    factor: float,
    anchor: float,
    block_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_score: np.ndarray,
    y_score: np.ndarray | None,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts = [
        fit_once(x_train, y_train, anchor, factor, int(seed), spec)
        for seed in spec["optimizer_seeds"]
    ]
    rows = []
    for item in attempts:
        rows.append(
            {
                "mode": mode,
                "support_lower_MeV": support,
                "upper_factor": factor,
                "anchor_GeV": anchor,
                "block": block_name,
                "seed": item.seed,
                "finite_success": item.finite_success,
                "warning_free": item.warning_free,
                "lml": item.lml,
                "constant": item.constant,
                "length": item.length,
                "warnings": item.warning_text,
                "error": item.error_text,
            }
        )
    eligible = [item for item in attempts if item.warning_free and item.model is not None]
    selected = max(eligible, key=lambda item: item.lml) if eligible else None
    reproduced: list[Attempt] = []
    if selected is not None:
        for item in eligible:
            if (
                abs(item.lml - selected.lml) <= float(spec["lml_reproduction_abs_tolerance"])
                and abs(item.length - selected.length)
                <= float(spec["length_reproduction_rel_tolerance"])
                * max(abs(selected.length), 1e-30)
            ):
                reproduced.append(item)
    lower, upper, base = length_bounds(anchor, factor, spec)
    constant_bounds = [float(item) for item in spec["kernel_constant_bounds"]]
    if selected is None:
        metrics = covariance_and_score(
            np.ones(len(x_score)), np.zeros((len(x_score), len(x_score))), spec, None
        )
        selected_seed = None
        selected_lml = selected_constant = selected_length = math.nan
        constant_at_bound = length_at_lower = length_at_upper = True
    else:
        mean, gp_cov = count_prediction(selected.model, x_score)
        metrics = covariance_and_score(mean, gp_cov, spec, y_score)
        selected_seed = selected.seed
        selected_lml = selected.lml
        selected_constant = selected.constant
        selected_length = selected.length
        tolerance = float(spec["kernel_bound_rel_tolerance"])
        constant_at_bound = bool(
            np.isclose(selected.constant, constant_bounds[0], rtol=tolerance, atol=1e-12)
            or np.isclose(selected.constant, constant_bounds[1], rtol=tolerance, atol=1e-12)
        )
        length_at_lower = bool(np.isclose(selected.length, lower, rtol=tolerance, atol=1e-12))
        length_at_upper = bool(np.isclose(selected.length, upper, rtol=tolerance, atol=1e-12))
    reproduction_pass = len(reproduced) >= int(spec["warning_free_repeats_required"])
    technical_without_length_upper = bool(
        selected is not None and reproduction_pass and not constant_at_bound
        and not length_at_lower and metrics["covariance_ok"]
    )
    technical_pass = bool(technical_without_length_upper and not length_at_upper)
    search = interval(x_train, 0.039, np.nextafter(0.180, math.inf))
    selected_row = {
        "mode": mode,
        "support_lower_MeV": support,
        "support_upper_MeV": 210,
        "upper_factor": factor,
        "anchor_GeV": anchor,
        "block": block_name,
        "selected_seed": selected_seed,
        "selected_lml": selected_lml,
        "selected_constant": selected_constant,
        "selected_length": selected_length,
        "length_lower": lower,
        "length_upper": upper,
        "sigma_x": base,
        "warning_free_repeat_count": len(eligible),
        "reproduced_warning_free_count": len(reproduced),
        "reproduction_pass": reproduction_pass,
        "constant_at_bound": constant_at_bound,
        "length_at_lower": length_at_lower,
        "length_at_upper": length_at_upper,
        "technical_without_length_upper_pass": technical_without_length_upper,
        "technical_pass": technical_pass,
        **metrics,
        "n_train": len(x_train),
        "n_score": len(x_score),
        "train_center_min_GeV": float(np.min(x_train)),
        "train_center_max_GeV": float(np.max(x_train)),
        "score_center_min_GeV": float(np.min(x_score)),
        "score_center_max_GeV": float(np.max(x_score)),
        "train_centers_sha256": array_hash(x_train),
        "score_centers_sha256": array_hash(x_score),
        "train_counts_sha256": array_hash(y_train),
        "score_counts_sha256": array_hash(y_score) if y_score is not None else "not_accessed",
        "n_search_train_centers": int(np.count_nonzero(search)),
        "n_search_score_centers": int(np.count_nonzero(interval(x_score, 0.039, np.nextafter(0.180, math.inf)))),
    }
    return rows, selected_row


def run_length(factor: float, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if factor not in [float(item) for item in spec["candidate_upper_factors"]]:
        raise StudyError("undeclared length factor")
    values, edges, declaration = load_histogram("development", spec)
    attempts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for support in [int(item) for item in spec["shortlist_lower_edges_MeV"]]:
        centers, counts, _ = rebin_support(values, edges, support / 1000.0, 0.210, int(spec["rebin"]))
        for anchor in [float(item) for item in spec["development_production_geometry_anchors_GeV"]]:
            half = float(spec["blind_nsigma"]) * sigma_2016(anchor, spec)
            score_mask = interval(centers, anchor - half, np.nextafter(anchor + half, math.inf))
            train_mask = ~score_mask
            cell_attempts, cell_selected = evaluate_cell(
                mode="length", support=support, factor=factor, anchor=anchor,
                block_name="production_blind_window", x_train=centers[train_mask],
                y_train=counts[train_mask], x_score=centers[score_mask], y_score=None,
                spec=spec,
            )
            attempts.extend(cell_attempts)
            selected.append(cell_selected)
    return pd.DataFrame(attempts), pd.DataFrame(selected), declaration


def run_confirmation(factor: float, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    factor_decision = HERE / "derived/length_factor_decision.json"
    if not factor_decision.is_file() or not FACTOR_FREEZE_PATH.is_file():
        raise StudyError("length factor is not frozen")
    frozen: dict[str, str] = {}
    for line in FACTOR_FREEZE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise StudyError("malformed length-factor freeze file")
        digest, relative = parts
        frozen[relative] = digest
    decision_relative = "derived/length_factor_decision.json"
    cells_relative = f"derived/length_factor_{factor:g}/selected_cells.csv"
    if frozen.get(decision_relative) != sha256_file(factor_decision):
        raise StudyError("length-factor decision hash is not frozen")
    factor_cells = HERE / cells_relative
    if not factor_cells.is_file() or frozen.get(cells_relative) != sha256_file(factor_cells):
        raise StudyError("length-factor selected-cell hash is not frozen")
    decision = json.loads(factor_decision.read_text(encoding="utf-8"))
    if decision.get("status") != "factor_frozen" or float(decision.get("selected_upper_factor")) != factor:
        raise StudyError("requested factor does not match frozen factor decision")
    if decision.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA or decision.get("spec_sha256") != EXPECTED_SPEC_SHA:
        raise StudyError("length-factor decision protocol/spec provenance mismatch")
    if decision.get("script_sha256") != sha256_file(ANALYSIS_SCRIPT_PATH):
        raise StudyError("length-factor analysis script drift")
    if decision.get("selected_cells_sha256") != sha256_file(factor_cells):
        raise StudyError("length-factor decision selected-cell provenance mismatch")
    values, edges, declaration = load_histogram("confirmation", spec)
    attempts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    blocks = {str(k): [float(x) for x in v] for k, v in spec["low_blocks"].items()}
    for support in [int(item) for item in spec["shortlist_lower_edges_MeV"]]:
        centers, counts, _ = rebin_support(values, edges, support / 1000.0, 0.210, int(spec["rebin"]))
        allowed = interval(centers, support / 1000.0, 0.03875)
        for anchor in [float(item) for item in spec["confirmation_kernel_anchors_GeV"]]:
            for block_name, (low, high) in blocks.items():
                heldout = interval(centers, low, high)
                train = allowed & ~heldout
                score = allowed & heldout
                if np.count_nonzero(interval(centers[train], 0.03875, 0.210)):
                    raise StudyError("forbidden full-data center entered confirmation training")
                cell_attempts, cell_selected = evaluate_cell(
                    mode="confirmation", support=support, factor=factor, anchor=anchor,
                    block_name=block_name, x_train=centers[train], y_train=counts[train],
                    x_score=centers[score], y_score=counts[score], spec=spec,
                )
                attempts.extend(cell_attempts)
                selected.append(cell_selected)
    frame = pd.DataFrame(selected)
    if (frame["n_search_train_centers"] != 0).any() or (frame["n_search_score_centers"] != 0).any():
        raise StudyError("full-data search center entered confirmation ledger")
    return pd.DataFrame(attempts), frame, declaration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("length", "confirmation"), required=True)
    parser.add_argument("--factor", type=float, required=True)
    args = parser.parse_args()
    spec = load_spec()
    if args.mode == "length":
        attempts, selected, declaration = run_length(args.factor, spec)
        out = HERE / "derived" / f"length_factor_{args.factor:g}"
    else:
        attempts, selected, declaration = run_confirmation(args.factor, spec)
        out = HERE / "derived" / "full_low_control_confirmation"
    out.mkdir(parents=True, exist_ok=True)
    attempts_path = out / "optimizer_attempts.csv"
    selected_path = out / "selected_cells.csv"
    attempts.to_csv(attempts_path, index=False)
    selected.to_csv(selected_path, index=False)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "upper_factor": args.factor,
        "input": declaration,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
        "outputs": {
            "optimizer_attempts.csv": {"rows": len(attempts), "sha256": sha256_file(attempts_path)},
            "selected_cells.csv": {"rows": len(selected), "sha256": sha256_file(selected_path)},
        },
        "search_score_centers": int(selected["n_search_score_centers"].sum()),
        "versions": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "uproot": uproot.__version__,
        },
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
