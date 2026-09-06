#!/usr/bin/env python3
"""Fit and qualify positive analytic source generators for the v4.8 study.

This command deliberately performs no GPR extraction.  It writes source-fit
diagnostics and a candidate analytic mean only; toy production is fail-closed
until the qualification status is reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import uproot
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import minimize
from scipy.special import expit


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
INPUTS = HERE / "inputs"
SOURCES = {
    "one_pct": (
        Path("/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root"),
        "preselection/h_invM_8000",
    ),
    "ten_pct": (
        Path("/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"),
        "preselection/h_invM_8000",
    ),
}
DEGREES = (8, 12, 16, 17, 18, 20, 24)
SUPPORT_LOW = 0.030
SUPPORT_HIGH = 0.300
SEARCH_MASSES = (0.065, 0.090, 0.120, 0.180, 0.210)
SIGMA_COEFFS_2021 = (0.00184825, -0.001375, 0.085875)
GOF_LOW = 0.75
GOF_HIGH = 1.25
MAX_BLOCK_CV_DEVIANCE = 1.25
MAX_FAKE_GAP_PROJECTION = 0.20
BASE_SEED = 20260813


class QualificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
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


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise QualificationError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sigma_2021(mass: float) -> float:
    return float(sum(value * mass**power for power, value in enumerate(SIGMA_COEFFS_2021)))


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.clip(np.asarray(expected, dtype=float), 1e-300, None)
    term = np.where(
        observed > 0,
        observed * np.log(observed / expected) - (observed - expected),
        expected,
    )
    return float(2.0 * np.sum(term))


def gof_metrics(observed: np.ndarray, expected: np.ndarray, n_parameters: int) -> dict[str, float]:
    expected = np.clip(np.asarray(expected, dtype=float), 1e-300, None)
    observed = np.asarray(observed, dtype=float)
    ndf = max(1, observed.size - int(n_parameters))
    return {
        "pearson_chi2ndf": float(np.sum((observed - expected) ** 2 / expected) / ndf),
        "poisson_deviance_ndf": float(poisson_deviance(observed, expected) / ndf),
        "ndf": int(ndf),
    }


def rebin_sum(values: np.ndarray, factor: int) -> np.ndarray:
    usable = (values.size // int(factor)) * int(factor)
    return np.asarray(values[:usable], dtype=float).reshape(-1, int(factor)).sum(axis=1)


@dataclass(frozen=True)
class FitResult:
    degree: int
    coefficients: np.ndarray
    turn_on: float
    width: float
    objective: float
    converged: bool
    status: int
    message: str
    iterations: int
    gradient_max_abs: float

    @property
    def parameters(self) -> np.ndarray:
        return np.r_[self.coefficients, self.turn_on, math.log(self.width)]


def design_matrix(centers: np.ndarray, degree: int) -> np.ndarray:
    mapped = 2.0 * (np.asarray(centers) - SUPPORT_LOW) / (SUPPORT_HIGH - SUPPORT_LOW) - 1.0
    return chebvander(mapped, int(degree))


def evaluate_model(centers: np.ndarray, fit: FitResult) -> np.ndarray:
    matrix = design_matrix(centers, fit.degree)
    turn = np.clip(expit((centers - fit.turn_on) / fit.width), 1e-300, 1.0)
    log_mean = np.clip(matrix @ fit.coefficients + np.log(turn), -700.0, 700.0)
    return np.exp(log_mean)


def fit_model(
    observed: np.ndarray,
    centers: np.ndarray,
    degree: int,
    *,
    training_mask: np.ndarray | None = None,
    start: FitResult | None = None,
) -> FitResult:
    observed = np.asarray(observed, dtype=float)
    centers = np.asarray(centers, dtype=float)
    mask = (
        np.ones(observed.size, dtype=bool)
        if training_mask is None
        else np.asarray(training_mask, dtype=bool)
    )
    if observed.shape != centers.shape or mask.shape != observed.shape:
        raise QualificationError("fit arrays have inconsistent shapes")
    if np.any(observed[mask] <= 0) or not np.all(np.isfinite(observed[mask])):
        raise QualificationError("fit domain contains nonpositive or nonfinite counts")
    matrix = design_matrix(centers, degree)
    if start is None:
        initial_turn = 0.046
        initial_width = 0.003
        offset = np.log(np.clip(expit((centers - initial_turn) / initial_width), 1e-300, 1.0))
        coefficients = np.linalg.lstsq(
            matrix[mask], np.log(observed[mask]) - offset[mask], rcond=None
        )[0]
        initial = np.r_[coefficients, initial_turn, math.log(initial_width)]
    else:
        if int(start.degree) != int(degree):
            raise QualificationError("start fit degree mismatch")
        initial = start.parameters.copy()

    scale = float(mask.sum())

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:-2]
        turn_on = float(parameters[-2])
        width = float(np.exp(parameters[-1]))
        q = (centers - turn_on) / width
        sigmoid = np.clip(expit(q), 1e-300, 1.0)
        log_mean = np.clip(matrix @ coefficients + np.log(sigmoid), -700.0, 700.0)
        mean = np.exp(log_mean)
        residual = (mean - observed)[mask]
        value = float(np.sum(mean[mask] - observed[mask] * log_mean[mask]) / scale)
        gradient_coefficients = matrix[mask].T @ residual / scale
        one_minus = 1.0 - sigmoid[mask]
        gradient_turn = float(np.sum(residual * (-one_minus / width)) / scale)
        gradient_log_width = float(np.sum(residual * (-one_minus * q[mask])) / scale)
        return value, np.r_[gradient_coefficients, gradient_turn, gradient_log_width]

    bounds = (
        [(None, None)] * (int(degree) + 1)
        + [(0.015, 0.080), (math.log(0.001), math.log(0.030))]
    )
    result = minimize(
        objective_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50},
    )
    gradient = objective_and_gradient(result.x)[1]
    return FitResult(
        degree=int(degree),
        coefficients=np.asarray(result.x[:-2], dtype=float),
        turn_on=float(result.x[-2]),
        width=float(np.exp(result.x[-1])),
        objective=float(result.fun),
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        gradient_max_abs=float(np.max(np.abs(gradient))),
    )


def load_source(path: Path, histogram: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with uproot.open(path) as root_file:
        values, edges = root_file[histogram].to_numpy(flow=False)
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= SUPPORT_LOW) & (centers < SUPPORT_HIGH)
    if np.any(values[mask] <= 0):
        raise QualificationError(f"{path}: support contains nonpositive counts")
    return values[mask], centers[mask], edges


def blocked_cv(
    observed: np.ndarray,
    centers: np.ndarray,
    full_fit: FitResult,
    block_bins: int,
) -> dict[str, Any]:
    block_ids = np.arange(observed.size, dtype=int) // int(block_bins)
    folds = block_ids % 5
    total_deviance = 0.0
    total_bins = 0
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        test = folds == fold
        fit = fit_model(
            observed,
            centers,
            full_fit.degree,
            training_mask=~test,
            start=full_fit,
        )
        expected = evaluate_model(centers, fit)
        deviance = poisson_deviance(observed[test], expected[test])
        total_deviance += deviance
        total_bins += int(test.sum())
        fold_rows.append(
            {
                "fold": fold,
                "test_bins": int(test.sum()),
                "deviance_per_bin": float(deviance / max(1, int(test.sum()))),
                "converged": fit.converged,
                "turn_on_gev": fit.turn_on,
                "width_gev": fit.width,
            }
        )
    return {
        "block_bins": int(block_bins),
        "block_width_mev": float(block_bins * np.median(np.diff(centers)) * 1000.0),
        "deviance_per_bin": float(total_deviance / max(1, total_bins)),
        "all_converged": bool(all(row["converged"] for row in fold_rows)),
        "folds": fold_rows,
    }


def fake_gap_diagnostics(
    observed: np.ndarray,
    centers: np.ndarray,
    full_fit: FitResult,
    mass: float,
) -> dict[str, Any]:
    sigma = sigma_2021(mass)
    gap = np.abs(centers - mass) < 2.25 * sigma
    fit = fit_model(
        observed,
        centers,
        full_fit.degree,
        training_mask=~gap,
        start=full_fit,
    )
    full_mean = evaluate_model(centers, full_fit)
    gap_mean = evaluate_model(centers, fit)
    signal = np.exp(-0.5 * ((centers - mass) / sigma) ** 2)
    signal /= np.sum(signal)
    variance = np.clip(full_mean, 1.0, None)
    denominator = math.sqrt(float(np.sum(signal * signal / variance)))
    signed_projection = float(
        np.sum(signal * (full_mean - gap_mean) / variance) / denominator
    )
    gap_gof = gof_metrics(observed[gap], gap_mean[gap], 0)
    return {
        "mass_gev": float(mass),
        "sigma_mass_gev": float(sigma),
        "gap_bins": int(gap.sum()),
        "gap_width_mev": float(4.5 * sigma * 1000.0),
        "converged": fit.converged,
        "turn_on_gev": fit.turn_on,
        "width_gev": fit.width,
        "gap_pearson_per_bin": gap_gof["pearson_chi2ndf"],
        "gap_deviance_per_bin": gap_gof["poisson_deviance_ndf"],
        "delta_model_diagonal_poisson_projection_sigma": signed_projection,
        "projection_gate_pass": abs(signed_projection) <= MAX_FAKE_GAP_PROJECTION,
    }


def evaluate_domain(
    observed: np.ndarray,
    centers: np.ndarray,
    expected: np.ndarray,
    n_parameters: int,
    low: float,
) -> dict[str, Any]:
    mask = (centers >= float(low)) & (centers < SUPPORT_HIGH)
    native = gof_metrics(observed[mask], expected[mask], n_parameters)
    rebinned = gof_metrics(
        rebin_sum(observed[mask], 5), rebin_sum(expected[mask], 5), n_parameters
    )
    return {
        "low_gev": float(low),
        "high_gev": SUPPORT_HIGH,
        "native": native,
        "rebin5": rebinned,
        "gof_gate_pass": bool(
            all(
                GOF_LOW <= value <= GOF_HIGH
                for value in (
                    native["pearson_chi2ndf"],
                    native["poisson_deviance_ndf"],
                    rebinned["pearson_chi2ndf"],
                    rebinned["poisson_deviance_ndf"],
                )
            )
        ),
    }


def run() -> dict[str, Any]:
    DERIVED.mkdir(parents=True, exist_ok=True)
    source_payload: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    fit_products: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, FitResult]] = {}
    for source, (path, histogram) in SOURCES.items():
        observed, centers, source_edges = load_source(path, histogram)
        records: list[dict[str, Any]] = []
        for degree in DEGREES:
            fit = fit_model(observed, centers, degree)
            expected = evaluate_model(centers, fit)
            n_parameters = degree + 3
            domains = {
                "support030": evaluate_domain(
                    observed, centers, expected, n_parameters, 0.030
                ),
                "support040": evaluate_domain(
                    observed, centers, expected, n_parameters, 0.040
                ),
            }
            block_bins = max(
                1,
                int(
                    math.ceil(
                        max(4.5 * sigma_2021(mass) for mass in SEARCH_MASSES)
                        / float(np.median(np.diff(centers)))
                    )
                ),
            )
            cv = blocked_cv(observed, centers, fit, block_bins)
            fake_gaps = [
                fake_gap_diagnostics(observed, centers, fit, mass)
                for mass in SEARCH_MASSES
            ]
            source_gof_pass = bool(
                fit.converged
                and np.all(np.isfinite(expected))
                and np.all(expected > 0)
                and all(record["gof_gate_pass"] for record in domains.values())
            )
            blocked_cv_pass = bool(
                cv["all_converged"]
                and cv["deviance_per_bin"] <= MAX_BLOCK_CV_DEVIANCE
            )
            projection_pass = bool(
                all(row["projection_gate_pass"] for row in fake_gaps)
            )
            record = {
                "degree": degree,
                "family": f"fSigmoidExpCheb{degree}",
                "converged": fit.converged,
                "optimizer_status": fit.status,
                "optimizer_message": fit.message,
                "optimizer_iterations": fit.iterations,
                "gradient_max_abs": fit.gradient_max_abs,
                "turn_on_gev": fit.turn_on,
                "width_gev": fit.width,
                "coefficients": fit.coefficients.tolist(),
                "domains": domains,
                "blocked_cv": cv,
                "fake_gaps": fake_gaps,
                "source_gof_gate_pass": source_gof_pass,
                "blocked_cv_gate_pass": blocked_cv_pass,
                "fake_gap_projection_gate_pass": projection_pass,
                "qualified": bool(
                    source_gof_pass and blocked_cv_pass and projection_pass
                ),
                "mean_sha256_float64": sha256_array(expected, "<f8"),
            }
            records.append(record)
            fit_products[(source, degree)] = (centers, expected, fit)
            summary_rows.append(
                {
                    "source": source,
                    "degree": degree,
                    "family": record["family"],
                    "converged": fit.converged,
                    "support030_native_pearson": domains["support030"]["native"]["pearson_chi2ndf"],
                    "support030_native_deviance": domains["support030"]["native"]["poisson_deviance_ndf"],
                    "support030_rebin5_pearson": domains["support030"]["rebin5"]["pearson_chi2ndf"],
                    "support030_rebin5_deviance": domains["support030"]["rebin5"]["poisson_deviance_ndf"],
                    "support040_native_pearson": domains["support040"]["native"]["pearson_chi2ndf"],
                    "support040_native_deviance": domains["support040"]["native"]["poisson_deviance_ndf"],
                    "support040_rebin5_pearson": domains["support040"]["rebin5"]["pearson_chi2ndf"],
                    "support040_rebin5_deviance": domains["support040"]["rebin5"]["poisson_deviance_ndf"],
                    "blocked_cv_deviance_per_bin": cv["deviance_per_bin"],
                    "max_abs_fake_gap_projection_sigma": max(
                        abs(row["delta_model_diagonal_poisson_projection_sigma"])
                        for row in fake_gaps
                    ),
                    "source_gof_gate_pass": source_gof_pass,
                    "blocked_cv_gate_pass": blocked_cv_pass,
                    "fake_gap_projection_gate_pass": projection_pass,
                    "qualified": bool(
                        source_gof_pass and blocked_cv_pass and projection_pass
                    ),
                }
            )
        source_payload[source] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "histogram": histogram,
            "support_counts": int(np.sum(observed)),
            "native_bin_width_gev": float(np.median(np.diff(centers))),
            "records": records,
        }

    common_degrees = [
        degree
        for degree in DEGREES
        if all(
            next(
                record
                for record in source_payload[source]["records"]
                if int(record["degree"]) == int(degree)
            )["qualified"]
            for source in SOURCES
        )
    ]
    source_gof_common_degrees = [
        degree
        for degree in DEGREES
        if all(
            next(
                record
                for record in source_payload[source]["records"]
                if int(record["degree"]) == int(degree)
            )["source_gof_gate_pass"]
            for source in SOURCES
        )
    ]
    selected_degree = min(common_degrees) if common_degrees else None
    output = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "protocol": "GENERATOR_QUALIFICATION_PROTOCOL.md",
        "protocol_sha256": sha256_file(HERE / "GENERATOR_QUALIFICATION_PROTOCOL.md"),
        "family_definition": "sigmoid((m-mt)/w)*exp(sum c_k*T_k(u(m)))",
        "support_gev": [SUPPORT_LOW, SUPPORT_HIGH],
        "candidate_degrees": list(DEGREES),
        "gates": {
            "pearson_and_deviance_interval": [GOF_LOW, GOF_HIGH],
            "maximum_blocked_cv_deviance_per_bin": MAX_BLOCK_CV_DEVIANCE,
            "maximum_abs_fake_gap_projection_sigma": MAX_FAKE_GAP_PROJECTION,
            "fake_gap_half_width_sigma": 2.25,
        },
        "sources": source_payload,
        "in_sample_screen_common_degrees": source_gof_common_degrees,
        "fully_qualified_common_degrees": common_degrees,
        "optimizer_reproducibility_gate_passed": False,
        "optimizer_reproducibility_blocker": (
            "reconnaissance fitter uses one start and lacks a standardized-gradient, "
            "replicated-basin, Hessian, and quadrature certificate"
        ),
        "selected_degree": selected_degree,
        "status": "pass" if selected_degree is not None else "blocked_no_qualified_generator",
        "interpretation": (
            "source-conditioned analytic stress-generator qualification; not physical truth, "
            "GPR closure, coverage, a limit, or a production-card selection"
        ),
    }
    atomic_json(DERIVED / "generator_qualification.json", output)
    atomic_csv(DERIVED / "generator_qualification_summary.csv", summary_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    args = parser.parse_args()
    if args.command == "run":
        result = run()
        print(json.dumps({
            "status": result["status"],
            "in_sample_screen_common_degrees": result[
                "in_sample_screen_common_degrees"
            ],
            "fully_qualified_common_degrees": result["fully_qualified_common_degrees"],
            "selected_degree": result["selected_degree"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
