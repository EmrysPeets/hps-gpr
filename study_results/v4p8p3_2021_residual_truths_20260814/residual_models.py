#!/usr/bin/env python3
"""Frozen residual-model definitions and source-only fitting utilities.

This module contains no GPR extraction logic.  It implements the two
source-conditioned generating means declared in ``MODEL_PROTOCOL.json`` and
the source/influence diagnostics that must be frozen before extraction toys
are inspected.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import uproot
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize
from scipy.special import expit, gammaln
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = HERE / "MODEL_PROTOCOL.json"
FIT_RESULT_PATH = HERE / "derived/source_fit_and_influence.json"
FIT_SUMMARY_CSV = HERE / "derived/source_fit_summary.csv"
INFLUENCE_CSV = HERE / "derived/signal_influence_audit.csv"
DRIVER_PATH = HERE / "fit_residual_models.py"
V4P8 = REPO / "study_results/v4p8_2021_functional_form_qualification_20260813"
RIGID_SPEC_PATH = V4P8 / "rigid_generator_spec.json"
HISTOGRAM = "preselection/h_invM_8000"
SUPPORT = (0.040, 0.300)
PRIMARY = (0.050, 0.250)
SIGMA_COEFFS = (0.00184825, -0.001375, 0.085875)
N_QUAD = 5
BASE_SEED = 20260814


class ModelError(RuntimeError):
    """Raised when a frozen source-model contract is violated."""


@dataclass(frozen=True)
class Histogram:
    values: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    support_mask: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
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


def atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def protocol() -> dict[str, Any]:
    payload = load_json(PROTOCOL_PATH)
    if int(payload.get("schema_version", -1)) != 1:
        raise ModelError("unsupported MODEL_PROTOCOL schema")
    if bool(payload.get("model_selection_uses_gpr_results", True)):
        raise ModelError("protocol permits GPR-result model selection")
    if not bool(payload.get("model_selection_frozen_before_signal_audit", False)):
        raise ModelError("model structure was not frozen before signal audit")
    return payload


def validate_declared_inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for source, record in payload["source_inputs"].items():
        path = Path(record["path"])
        if not path.is_file():
            raise ModelError(f"missing {source} source: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ModelError(
                f"{source} source hash mismatch: {actual} != {record['sha256']}"
            )
        checks[source] = {"path": str(path), "sha256": actual}
    if sha256_file(RIGID_SPEC_PATH) != (
        "40feebe17d37a5b24820bfaf63dc0fe36869e43bf663114c5d23f1836530b0b7"
    ):
        raise ModelError("authoritative v4.8 rigid generator specification drift")
    checks["rigid_generator_spec"] = {
        "path": str(RIGID_SPEC_PATH),
        "sha256": sha256_file(RIGID_SPEC_PATH),
    }
    return checks


def load_histogram(source: str, payload: Mapping[str, Any] | None = None) -> Histogram:
    payload = protocol() if payload is None else payload
    record = payload["source_inputs"][source]
    with uproot.open(record["path"]) as root_file:
        values, edges = root_file[record["histogram"]].to_numpy(flow=False)
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    support = (edges[:-1] >= SUPPORT[0] - 1e-12) & (
        edges[1:] <= SUPPORT[1] + 1e-12
    )
    if not np.any(support) or np.any(values[support] < 0):
        raise ModelError(f"invalid {source} source histogram")
    return Histogram(values, edges, centers, support)


def sigma_2021(mass: float | np.ndarray) -> np.ndarray:
    value = np.asarray(mass, dtype=float)
    return sum(coefficient * value**power for power, coefficient in enumerate(SIGMA_COEFFS))


def quadrature(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(N_QUAD)
    left = edges[:-1, None]
    right = edges[1:, None]
    x = 0.5 * (right + left) + 0.5 * (right - left) * nodes[None, :]
    integration_weights = 0.5 * (right - left) * weights[None, :]
    return x, np.broadcast_to(integration_weights, x.shape)


def _weighted_bin_integral(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(values, dtype=float) * weights, axis=1)


def _rigid_log_shape(x: np.ndarray) -> np.ndarray:
    spec = load_json(RIGID_SPEC_PATH)
    constants = spec["development_constants"]
    parameters = spec["one_pct_shape_parameters"]
    x0 = float(constants["x0_gev"])
    xt = float(constants["xt_gev"])
    width = float(constants["w_gev"])
    z = np.asarray(x, dtype=float) - x0
    if np.any(z <= 0):
        raise ModelError("rigid shape evaluated below x0")
    u = 2.0 * (np.asarray(x) - SUPPORT[0]) / (SUPPORT[1] - SUPPORT[0]) - 1.0
    t2 = 2.0 * u * u - 1.0
    u2 = u * u
    t6 = 32.0 * u2**3 - 48.0 * u2**2 + 18.0 * u2 - 1.0
    return (
        np.log(np.clip(expit((np.asarray(x) - xt) / width), 1e-300, 1.0))
        + float(parameters["a"]) * np.log(z)
        - np.power(z / float(parameters["lambda_gev"]), float(parameters["power"]))
        + float(parameters["d2"]) * t2
        + float(parameters["d6"]) * t6
    )


def _safe_shape_from_log(log_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_values))
    if not np.isfinite(maximum):
        raise ModelError("nonfinite log shape")
    shifted_log = np.asarray(log_values, dtype=float) - maximum
    if float(np.min(shifted_log)) < -700.0:
        raise ModelError("log shape dynamic range exceeds safe integration range")
    shifted = np.exp(shifted_log)
    shape = _weighted_bin_integral(shifted, weights)
    if np.any(~np.isfinite(shape)) or np.any(shape <= 0):
        raise ModelError("bin-integrated shape is not finite and positive")
    return shape


def rigid_shape(edges: np.ndarray) -> np.ndarray:
    xq, wq = quadrature(np.asarray(edges, dtype=float))
    return _safe_shape_from_log(_rigid_log_shape(xq), wq)


def natural_node_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    points = np.asarray([SUPPORT[0], *map(float, knots), SUPPORT[1]], dtype=float)
    target = np.asarray(x, dtype=float)
    columns = []
    for index in range(len(knots)):
        values = np.zeros(len(points), dtype=float)
        values[index + 1] = 1.0
        spline = CubicSpline(points, values, bc_type="natural", extrapolate=False)
        column = np.asarray(spline(target), dtype=float)
        if np.any(~np.isfinite(column)):
            raise ModelError("natural-spline basis extrapolation")
        columns.append(column)
    return np.stack(columns, axis=-1)


def _smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=float) - low) / (high - low), 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t * t)


def regional_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = 1.0 - _smoothstep(x, 0.085, 0.125)
    high = _smoothstep(x, 0.165, 0.215)
    middle = 1.0 - low - high
    if np.any(low < -1e-12) or np.any(middle < -1e-12) or np.any(high < -1e-12):
        raise ModelError("regional blend weights are negative")
    if not np.allclose(low + middle + high, 1.0, rtol=0.0, atol=1e-12):
        raise ModelError("regional blend weights do not form a partition of unity")
    return low, middle, high


def _regional_decode(q: Sequence[float]) -> dict[str, float]:
    q = np.asarray(q, dtype=float)
    if q.shape != (8,):
        raise ModelError("regional parameter vector must have length eight")
    return {
        "a_low": float(q[0]),
        "lambda_low": float(math.exp(q[1])),
        "power_low": float(math.exp(q[2])),
        "b1": float(q[3]),
        "b2": float(q[4]),
        "a_high": float(q[5]),
        "lambda_high": float(math.exp(q[6])),
        "power_high": float(math.exp(q[7])),
    }


def regional_log_shape(x: np.ndarray, q: Sequence[float]) -> np.ndarray:
    parameters = _regional_decode(q)
    rigid = load_json(RIGID_SPEC_PATH)["development_constants"]
    x0 = float(rigid["x0_gev"])
    xt = float(rigid["xt_gev"])
    width = float(rigid["w_gev"])
    x = np.asarray(x, dtype=float)
    z = x - x0
    if np.any(z <= 0):
        raise ModelError("regional low model evaluated below x0")
    low_raw = (
        np.log(np.clip(expit((x - xt) / width), 1e-300, 1.0))
        + parameters["a_low"] * np.log(z)
        - np.power(z / parameters["lambda_low"], parameters["power_low"])
    )
    u = 2.0 * (x - 0.085) / (0.215 - 0.085) - 1.0
    middle_raw = parameters["b1"] * u + parameters["b2"] * (2.0 * u * u - 1.0)
    high_raw = (
        parameters["a_high"] * np.log(x)
        - np.power(x / parameters["lambda_high"], parameters["power_high"])
    )

    def scalar_parts(point: float) -> tuple[float, float, float]:
        zz = point - x0
        low_value = (
            math.log(max(float(expit((point - xt) / width)), 1e-300))
            + parameters["a_low"] * math.log(zz)
            - (zz / parameters["lambda_low"]) ** parameters["power_low"]
        )
        uu = 2.0 * (point - 0.085) / (0.215 - 0.085) - 1.0
        middle_value = parameters["b1"] * uu + parameters["b2"] * (2.0 * uu * uu - 1.0)
        high_value = (
            parameters["a_high"] * math.log(point)
            - (point / parameters["lambda_high"]) ** parameters["power_high"]
        )
        return low_value, middle_value, high_value

    low_anchor, middle_low_anchor, _ = scalar_parts(0.105)
    _, middle_high_anchor, high_anchor = scalar_parts(0.190)
    low = low_raw + (middle_low_anchor - low_anchor)
    middle = middle_raw
    high = high_raw + (middle_high_anchor - high_anchor)
    w_low, w_middle, w_high = regional_weights(x)
    return w_low * low + w_middle * middle + w_high * high


def regional_high_max_derivative(q: Sequence[float]) -> float:
    p = _regional_decode(q)
    grid = np.linspace(0.165, 0.300, 200)
    derivative = p["a_high"] / grid - (
        p["power_high"]
        * np.power(grid, p["power_high"] - 1.0)
        / np.power(p["lambda_high"], p["power_high"])
    )
    return float(np.max(derivative))


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape or np.any(expected <= 0):
        return float("inf")
    term = expected - observed
    positive = observed > 0
    term[positive] += observed[positive] * np.log(observed[positive] / expected[positive])
    return float(2.0 * np.sum(term))


def pearson(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape or np.any(expected <= 0):
        return float("inf")
    return float(np.sum((observed - expected) ** 2 / expected))


def profile_normalization(
    shape: np.ndarray, observed: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    shape = np.asarray(shape, dtype=float)
    observed = np.asarray(observed, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    denominator = float(np.sum(shape[fit_mask]))
    numerator = float(np.sum(observed[fit_mask]))
    if denominator <= 0 or numerator <= 0:
        raise ModelError("normalization cannot be profiled")
    scale = numerator / denominator
    return scale * shape, scale


def rebin_sum(values: np.ndarray, factor: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    size = (values.size // int(factor)) * int(factor)
    return values[:size].reshape(-1, int(factor)).sum(axis=1)


def fit_metrics(
    observed: np.ndarray, expected: np.ndarray, n_shape: int
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ndf = max(1, observed.size - int(n_shape) - 1)
    output = {
        "n_bins": int(observed.size),
        "n_shape_parameters": int(n_shape),
        "pearson_per_ndf": pearson(observed, expected) / ndf,
        "deviance_per_ndf": poisson_deviance(observed, expected) / ndf,
        "maximum_abs_pearson_residual": float(
            np.max(np.abs(observed - expected) / np.sqrt(expected))
        ),
    }
    for factor in (5, 20, 40):
        obs = rebin_sum(observed, factor)
        exp = rebin_sum(expected, factor)
        coarse_ndf = max(1, obs.size - int(n_shape) - 1)
        output[f"rebin{factor}_pearson_per_ndf"] = pearson(obs, exp) / coarse_ndf
        output[f"rebin{factor}_deviance_per_ndf"] = poisson_deviance(obs, exp) / coarse_ndf
        output[f"rebin{factor}_maximum_abs_pearson_residual"] = float(
            np.max(np.abs(obs - exp) / np.sqrt(exp))
        )
    return output


def _seed(*parts: object) -> int:
    material = "|".join([str(BASE_SEED), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _basin_repeats(
    records: Sequence[Mapping[str, Any]],
    bounds: Sequence[Sequence[float]],
    *,
    constraint_keys: Sequence[str] = (),
) -> int:
    """Count successful repeats agreeing in objective and parameters."""

    eligible = [
        row
        for row in records
        if bool(row.get("success", False))
        and np.isfinite(float(row.get("objective", float("inf"))))
        and all(float(row.get(key, -float("inf"))) >= -1e-8 for key in constraint_keys)
    ]
    if not eligible:
        return 0
    best_row = min(eligible, key=lambda row: float(row["objective"]))
    best_objective = float(best_row["objective"])
    best_parameters = np.asarray(best_row["parameters"], dtype=float)
    widths = np.asarray([high - low for low, high in bounds], dtype=float)
    objective_tolerance = max(1e-8, 1e-6 * max(1.0, abs(best_objective)))
    repeats = 0
    for row in eligible:
        parameters = np.asarray(row["parameters"], dtype=float)
        objective_agrees = abs(float(row["objective"]) - best_objective) <= objective_tolerance
        parameter_agrees = bool(
            np.max(np.abs(parameters - best_parameters) / widths) <= 5e-4
        )
        repeats += int(objective_agrees and parameter_agrees)
    return repeats


def _node_shape(
    edges: np.ndarray, knots: Sequence[float], coefficients: Sequence[float]
) -> np.ndarray:
    xq, wq = quadrature(edges)
    basis = natural_node_basis(xq, knots)
    log_shape = _rigid_log_shape(xq) + np.tensordot(
        basis, np.asarray(coefficients, dtype=float), axes=([-1], [0])
    )
    return _safe_shape_from_log(log_shape, wq)


def fit_node_coefficients(
    histogram: Histogram,
    knots: Sequence[float],
    *,
    fit_mask: np.ndarray | None = None,
    starts: int = 24,
    initial: Sequence[float] | None = None,
    namespace: str = "node",
) -> dict[str, Any]:
    support_indices = np.flatnonzero(histogram.support_mask)
    edges = histogram.edges[np.r_[support_indices, support_indices[-1] + 1]]
    observed = histogram.values[histogram.support_mask]
    fit_mask = np.ones_like(observed, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != observed.shape:
        raise ModelError("node fit mask shape mismatch")
    bounds = [(-0.05, 0.05)] * len(knots)
    nominal = np.zeros(len(knots), dtype=float) if initial is None else np.asarray(initial, dtype=float)
    records = []
    for attempt in range(int(starts)):
        rng = np.random.default_rng(_seed(namespace, len(knots), attempt))
        start = nominal.copy() if attempt == 0 else np.clip(
            nominal + rng.normal(0.0, 0.006, len(knots)), -0.045, 0.045
        )

        def objective(coefficients: np.ndarray) -> float:
            try:
                shape = _node_shape(edges, knots, coefficients)
                expected, _ = profile_normalization(shape, observed, fit_mask)
                return poisson_deviance(observed[fit_mask], expected[fit_mask])
            except Exception:
                return 1e300

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
        )
        records.append(
            {
                "attempt": attempt,
                "success": bool(result.success),
                "objective": float(result.fun),
                "parameters": np.asarray(result.x, dtype=float).tolist(),
                "message": str(result.message),
            }
        )
    usable = [
        row
        for row in records
        if bool(row["success"]) and np.isfinite(row["objective"])
    ]
    if not usable:
        raise ModelError("all node fits failed")
    best = min(usable, key=lambda row: row["objective"])
    coefficients = np.asarray(best["parameters"], dtype=float)
    shape = _node_shape(edges, knots, coefficients)
    expected, normalization = profile_normalization(shape, observed, fit_mask)
    distances = [
        min((value - low) / (high - low), (high - value) / (high - low))
        for value, (low, high) in zip(coefficients, bounds)
    ]
    return {
        "parameters": coefficients.tolist(),
        "normalization": normalization,
        "objective": float(best["objective"]),
        "optimizer_success": bool(best["success"]),
        "best_attempt": int(best["attempt"]),
        "best_basin_repeats": _basin_repeats(records, bounds),
        "minimum_normalized_bound_distance": float(min(distances)),
        "attempts": records,
        "expected_support": expected,
        "support_edges": edges,
    }


def _node_basis_mean(
    knots: Sequence[float], reference_histogram: Histogram
) -> np.ndarray:
    center_basis = natural_node_basis(
        reference_histogram.centers[reference_histogram.support_mask], knots
    )
    weights = reference_histogram.values[reference_histogram.support_mask]
    return np.average(center_basis, axis=0, weights=np.clip(weights, 1.0, None))


def _node_delta_basis(
    x: np.ndarray, knots: Sequence[float], frozen_mean: Sequence[float]
) -> np.ndarray:
    mean = np.asarray(frozen_mean, dtype=float)
    if mean.shape != (len(knots),):
        raise ModelError("frozen node-centering mean shape mismatch")
    return natural_node_basis(np.asarray(x), knots) - mean


def fit_node_transfer(
    histogram: Histogram,
    knots: Sequence[float],
    one_pct_coefficients: Sequence[float],
    frozen_delta_mean: Sequence[float],
    *,
    fit_mask: np.ndarray | None = None,
    starts: int = 24,
    initial: Sequence[float] | None = None,
    namespace: str = "node_transfer",
) -> dict[str, Any]:
    support_indices = np.flatnonzero(histogram.support_mask)
    edges = histogram.edges[np.r_[support_indices, support_indices[-1] + 1]]
    observed = histogram.values[histogram.support_mask]
    fit_mask = np.ones_like(observed, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    xq, wq = quadrature(edges)
    base_basis = natural_node_basis(xq, knots)
    delta_mean = np.asarray(frozen_delta_mean, dtype=float)
    delta_basis = _node_delta_basis(xq, knots, delta_mean)
    rigid = _rigid_log_shape(xq)
    one = np.tensordot(base_basis, np.asarray(one_pct_coefficients), axes=([-1], [0]))
    dense = np.linspace(PRIMARY[0], PRIMARY[1], 801)
    dense_delta = _node_delta_basis(dense, knots, delta_mean)
    bounds = [(-0.004, 0.004)] * len(knots)
    ridge_sigma = 0.002
    nominal = np.zeros(len(knots), dtype=float) if initial is None else np.asarray(initial, dtype=float)

    def constraint_values(delta: np.ndarray) -> np.ndarray:
        pointwise = dense_delta @ delta
        values = [0.005 - np.max(np.abs(pointwise))]
        boundary_augmented = np.r_[0.0, delta, 0.0]
        values.append(0.003 - np.max(np.abs(np.diff(boundary_augmented))))
        return np.asarray(values, dtype=float)

    records = []
    for attempt in range(int(starts)):
        rng = np.random.default_rng(_seed(namespace, len(knots), attempt))
        start = nominal.copy() if attempt == 0 else np.clip(
            nominal + rng.normal(0.0, 0.0008, len(knots)), -0.0035, 0.0035
        )

        def objective(delta: np.ndarray) -> float:
            try:
                log_shape = rigid + one + np.tensordot(delta_basis, delta, axes=([-1], [0]))
                shape = _safe_shape_from_log(log_shape, wq)
                expected, _ = profile_normalization(shape, observed, fit_mask)
                return poisson_deviance(observed[fit_mask], expected[fit_mask]) + float(
                    np.sum((delta / ridge_sigma) ** 2)
                )
            except Exception:
                return 1e300

        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraint_values}],
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        records.append(
            {
                "attempt": attempt,
                "success": bool(result.success),
                "objective": float(result.fun),
                "parameters": np.asarray(result.x, dtype=float).tolist(),
                "minimum_constraint_margin": float(np.min(constraint_values(result.x))),
                "message": str(result.message),
            }
        )
    usable = [
        row
        for row in records
        if bool(row["success"])
        and np.isfinite(row["objective"])
        and row["minimum_constraint_margin"] >= -1e-8
    ]
    if not usable:
        raise ModelError("all node-transfer fits failed")
    best = min(usable, key=lambda row: row["objective"])
    delta = np.asarray(best["parameters"], dtype=float)
    log_shape = rigid + one + np.tensordot(delta_basis, delta, axes=([-1], [0]))
    shape = _safe_shape_from_log(log_shape, wq)
    expected, normalization = profile_normalization(shape, observed, fit_mask)
    dense_values = dense_delta @ delta
    distances = [
        min((value - low) / (high - low), (high - value) / (high - low))
        for value, (low, high) in zip(delta, bounds)
    ]
    return {
        "parameters": delta.tolist(),
        "normalization": normalization,
        "objective": float(best["objective"]),
        "optimizer_success": bool(best["success"]),
        "best_attempt": int(best["attempt"]),
        "best_basin_repeats": _basin_repeats(
            records, bounds, constraint_keys=("minimum_constraint_margin",)
        ),
        "delta_weighted_mean_basis": delta_mean.tolist(),
        "maximum_abs_pointwise_delta": float(np.max(np.abs(dense_values))),
        "maximum_adjacent_coefficient_difference": float(
            np.max(np.abs(np.diff(np.r_[0.0, delta, 0.0])))
        ),
        "minimum_constraint_margin": float(best["minimum_constraint_margin"]),
        "minimum_normalized_bound_distance": float(min(distances)),
        "attempts": records,
        "expected_support": expected,
        "support_edges": edges,
    }


REGIONAL_BOUNDS = (
    (0.5, 10.0),
    (math.log(1e-4), math.log(0.02)),
    (math.log(0.2), math.log(2.0)),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-20.0, 20.0),
    (math.log(0.05), math.log(2.0)),
    (math.log(0.2), math.log(5.0)),
)


def _regional_log_prefit(
    histogram: Histogram,
    fit_mask: np.ndarray,
    *,
    namespace: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a deterministic, count-blind-to-GPR seed for the Poisson fit.

    The regional likelihood has a broad, curved basin in its raw coordinates.
    A coarse log-count fit supplies only an optimizer seed; the returned
    parameters are always refined with the native-bin Poisson objective below.
    This helper is used only when no frozen/source-fit initial value is passed.
    """

    centers = histogram.centers[histogram.support_mask]
    observed = histogram.values[histogram.support_mask]
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != observed.shape:
        raise ModelError("regional prefit mask shape mismatch")

    # Aggregate adjacent native bins to make the seed insensitive to the
    # high-frequency residuals that the three-region truth must not chase.
    factor = 20
    usable = (observed.size // factor) * factor
    coarse_observed = observed[:usable].reshape(-1, factor).sum(axis=1)
    coarse_centers = centers[:usable].reshape(-1, factor).mean(axis=1)
    coarse_mask = fit_mask[:usable].reshape(-1, factor).all(axis=1)
    selected = coarse_mask & (coarse_observed > 0)
    if np.count_nonzero(selected) < 20:
        raise ModelError("too few bins for regional log prefit")
    x = coarse_centers[selected]
    target = np.log(coarse_observed[selected])

    default = np.asarray(
        [
            4.1,
            math.log(8.1e-4),
            math.log(0.53),
            -2.0,
            0.0,
            0.0,
            math.log(0.15),
            math.log(1.0),
        ],
        dtype=float,
    )
    lower = np.asarray([item[0] for item in REGIONAL_BOUNDS], dtype=float)
    upper = np.asarray([item[1] for item in REGIONAL_BOUNDS], dtype=float)
    widths = upper - lower
    records: list[dict[str, Any]] = []
    candidates: list[tuple[float, np.ndarray]] = []
    for attempt in range(6):
        rng = np.random.default_rng(_seed(namespace, "log_prefit", attempt))
        if attempt == 0:
            start = default.copy()
        else:
            start = np.clip(
                default + rng.normal(0.0, 0.10, default.size) * widths,
                lower + 1e-7,
                upper - 1e-7,
            )

        def residual(q: np.ndarray) -> np.ndarray:
            try:
                prediction = regional_log_shape(x, q)
                offset = float(np.mean(target - prediction))
                shape_residual = prediction + offset - target
                # A fixed-size soft residual keeps the seed in the declared
                # monotone-high-region domain. The exact inequality is imposed
                # again in the native-bin likelihood fit.
                monotone_violation = max(0.0, regional_high_max_derivative(q))
                return np.r_[shape_residual, 100.0 * monotone_violation]
            except Exception:
                return np.full(target.size + 1, 1e6, dtype=float)

        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=4000,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        score = float(np.mean(np.square(residual(result.x)[:-1])))
        margin = float(-regional_high_max_derivative(result.x))
        records.append(
            {
                "attempt": attempt,
                "success": bool(result.success),
                "mean_squared_log_residual": score,
                "high_monotone_margin": margin,
                "parameters": np.asarray(result.x, dtype=float).tolist(),
                "message": str(result.message),
            }
        )
        if bool(result.success) and np.isfinite(score) and margin >= -1e-7:
            candidates.append((score, np.asarray(result.x, dtype=float)))
    if not candidates:
        raise ModelError("regional log prefit found no monotone seed")
    return min(candidates, key=lambda item: item[0])[1], records


def _regional_shape(edges: np.ndarray, q: Sequence[float]) -> np.ndarray:
    xq, wq = quadrature(edges)
    return _safe_shape_from_log(regional_log_shape(xq, q), wq)


def fit_regional_parameters(
    histogram: Histogram,
    *,
    fit_mask: np.ndarray | None = None,
    starts: int = 24,
    initial: Sequence[float] | None = None,
    namespace: str = "regional",
    include_independent_data_seed: bool = False,
) -> dict[str, Any]:
    support_indices = np.flatnonzero(histogram.support_mask)
    edges = histogram.edges[np.r_[support_indices, support_indices[-1] + 1]]
    observed = histogram.values[histogram.support_mask]
    fit_mask = np.ones_like(observed, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if initial is None:
        nominal, prefit_records = _regional_log_prefit(
            histogram, fit_mask, namespace=namespace
        )
        nominal_start_type = "data_prefit"
    else:
        nominal = np.asarray(initial, dtype=float)
        prefit_records = []
        nominal_start_type = "frozen"
    independent_seed = None
    if initial is not None and include_independent_data_seed:
        independent_seed, prefit_records = _regional_log_prefit(
            histogram, fit_mask, namespace=f"{namespace}_independent"
        )

    def monotone_constraint(q: np.ndarray) -> float:
        return -regional_high_max_derivative(q)

    records = []
    for attempt in range(int(starts)):
        rng = np.random.default_rng(_seed(namespace, attempt))
        if attempt == 0:
            start = nominal.copy()
            start_type = nominal_start_type
        elif attempt == 1 and independent_seed is not None:
            start = independent_seed.copy()
            start_type = "independent_data_prefit"
        else:
            widths = np.asarray([high - low for low, high in REGIONAL_BOUNDS])
            start = nominal + rng.normal(0.0, 0.06, 8) * widths
            start = np.asarray(
                [np.clip(value, low + 1e-6, high - 1e-6) for value, (low, high) in zip(start, REGIONAL_BOUNDS)]
            )
            if monotone_constraint(start) < 0:
                start[5] = min(start[5], -0.1)
            start_type = "deterministic_perturbation"

        def objective(q: np.ndarray) -> float:
            try:
                if monotone_constraint(q) < -1e-8:
                    return 1e200 + 1e8 * abs(monotone_constraint(q))
                shape = _regional_shape(edges, q)
                expected, _ = profile_normalization(shape, observed, fit_mask)
                # Scaling improves finite-difference conditioning without
                # changing the native-bin Poisson minimum.
                return poisson_deviance(observed[fit_mask], expected[fit_mask]) / max(
                    1, int(np.count_nonzero(fit_mask))
                )
            except Exception:
                return 1e300

        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=REGIONAL_BOUNDS,
            constraints=[{"type": "ineq", "fun": monotone_constraint}],
            options={"maxiter": 4000, "ftol": 1e-9},
        )
        records.append(
            {
                "attempt": attempt,
                "start_type": start_type,
                "success": bool(result.success),
                "objective": float(result.fun),
                "parameters": np.asarray(result.x, dtype=float).tolist(),
                "high_monotone_margin": float(monotone_constraint(result.x)),
                "message": str(result.message),
            }
        )
    usable = [
        row
        for row in records
        if bool(row["success"])
        and np.isfinite(row["objective"])
        and row["high_monotone_margin"] >= -1e-8
    ]
    if not usable:
        raise ModelError("all regional fits failed")
    best = min(usable, key=lambda row: row["objective"])
    q = np.asarray(best["parameters"], dtype=float)
    shape = _regional_shape(edges, q)
    expected, normalization = profile_normalization(shape, observed, fit_mask)
    distances = [
        min((value - low) / (high - low), (high - value) / (high - low))
        for value, (low, high) in zip(q, REGIONAL_BOUNDS)
    ]
    raw_deviance = poisson_deviance(observed[fit_mask], expected[fit_mask])
    return {
        "parameters": q.tolist(),
        "decoded_parameters": _regional_decode(q),
        "normalization": normalization,
        "objective": raw_deviance,
        "objective_per_fitted_bin": float(best["objective"]),
        "optimizer_success": bool(best["success"]),
        "best_attempt": int(best["attempt"]),
        "best_basin_repeats": _basin_repeats(
            records, REGIONAL_BOUNDS, constraint_keys=("high_monotone_margin",)
        ),
        "minimum_normalized_bound_distance": float(min(distances)),
        "high_monotone_margin": float(best["high_monotone_margin"]),
        "attempts": records,
        "log_prefit_attempts": prefit_records,
        "expected_support": expected,
        "support_edges": edges,
    }


def _regional_contrast_means(reference_histogram: Histogram) -> np.ndarray:
    center_low, _, center_high = regional_weights(
        reference_histogram.centers[reference_histogram.support_mask]
    )
    weights = np.clip(
        reference_histogram.values[reference_histogram.support_mask], 1.0, None
    )
    return np.asarray(
        [np.average(center_low, weights=weights), np.average(center_high, weights=weights)]
    )


def _regional_contrast_basis(
    x: np.ndarray, frozen_means: Sequence[float]
) -> np.ndarray:
    means = np.asarray(frozen_means, dtype=float)
    if means.shape != (2,):
        raise ModelError("frozen regional-centering means shape mismatch")
    low, _, high = regional_weights(np.asarray(x))
    return np.stack([low - means[0], high - means[1]], axis=-1)


def fit_regional_transfer(
    histogram: Histogram,
    one_pct_parameters: Sequence[float],
    frozen_contrast_means: Sequence[float],
    *,
    fit_mask: np.ndarray | None = None,
    starts: int = 24,
    initial: Sequence[float] | None = None,
    namespace: str = "regional_transfer",
    include_independent_data_seed: bool = False,
) -> dict[str, Any]:
    support_indices = np.flatnonzero(histogram.support_mask)
    edges = histogram.edges[np.r_[support_indices, support_indices[-1] + 1]]
    observed = histogram.values[histogram.support_mask]
    fit_mask = np.ones_like(observed, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    xq, wq = quadrature(edges)
    means = np.asarray(frozen_contrast_means, dtype=float)
    contrast = _regional_contrast_basis(xq, means)
    base = regional_log_shape(xq, one_pct_parameters)
    dense = np.linspace(PRIMARY[0], PRIMARY[1], 801)
    dense_contrast = _regional_contrast_basis(dense, means)
    bounds = [(-0.004, 0.004), (-0.004, 0.004)]
    ridge_sigma = 0.002
    nominal = np.zeros(2, dtype=float) if initial is None else np.asarray(initial, dtype=float)

    def constraint(delta: np.ndarray) -> float:
        return float(0.005 - np.max(np.abs(dense_contrast @ delta)))

    records = []
    for attempt in range(int(starts)):
        rng = np.random.default_rng(_seed(namespace, attempt))
        if attempt == 0:
            start = nominal.copy()
            start_type = "frozen"
        elif attempt == 1 and include_independent_data_seed:
            start = np.zeros(2, dtype=float)
            start_type = "independent_zero"
        else:
            start = np.clip(
                nominal + rng.normal(0.0, 0.0008, 2), -0.0035, 0.0035
            )
            start_type = "deterministic_perturbation"

        def objective(delta: np.ndarray) -> float:
            try:
                shape = _safe_shape_from_log(
                    base + np.tensordot(contrast, delta, axes=([-1], [0])), wq
                )
                expected, _ = profile_normalization(shape, observed, fit_mask)
                return poisson_deviance(observed[fit_mask], expected[fit_mask]) + float(
                    np.sum((delta / ridge_sigma) ** 2)
                )
            except Exception:
                return 1e300

        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraint}],
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        records.append(
            {
                "attempt": attempt,
                "start_type": start_type,
                "success": bool(result.success),
                "objective": float(result.fun),
                "parameters": np.asarray(result.x, dtype=float).tolist(),
                "minimum_constraint_margin": float(constraint(result.x)),
                "message": str(result.message),
            }
        )
    usable = [
        row
        for row in records
        if bool(row["success"])
        and np.isfinite(row["objective"])
        and row["minimum_constraint_margin"] >= -1e-8
    ]
    if not usable:
        raise ModelError("all regional-transfer fits failed")
    best = min(usable, key=lambda row: row["objective"])
    delta = np.asarray(best["parameters"], dtype=float)
    shape = _safe_shape_from_log(
        base + np.tensordot(contrast, delta, axes=([-1], [0])), wq
    )
    expected, normalization = profile_normalization(shape, observed, fit_mask)
    distances = [
        min((value - low) / (high - low), (high - value) / (high - low))
        for value, (low, high) in zip(delta, bounds)
    ]
    return {
        "parameters": delta.tolist(),
        "normalization": normalization,
        "objective": float(best["objective"]),
        "optimizer_success": bool(best["success"]),
        "best_attempt": int(best["attempt"]),
        "best_basin_repeats": _basin_repeats(
            records, bounds, constraint_keys=("minimum_constraint_margin",)
        ),
        "contrast_weighted_means": means.tolist(),
        "maximum_abs_pointwise_delta": float(
            np.max(np.abs(dense_contrast @ delta))
        ),
        "minimum_constraint_margin": float(best["minimum_constraint_margin"]),
        "minimum_normalized_bound_distance": float(min(distances)),
        "attempts": records,
        "expected_support": expected,
        "support_edges": edges,
    }


def _strip_fit_arrays(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"expected_support", "support_edges"}
    }


def _blocked_masks(histogram: Histogram, folds: Sequence[Sequence[float]]) -> list[np.ndarray]:
    centers = histogram.centers[histogram.support_mask]
    masks = []
    for index, (low, high) in enumerate(folds):
        held = (centers >= float(low) - 1e-12) & (
            centers < float(high) - (0.0 if index == len(folds) - 1 else 1e-12)
        )
        masks.append(held)
    return masks


def node_cross_validation(
    histogram: Histogram, candidates: Mapping[str, Sequence[float]], folds: Sequence[Sequence[float]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    held_masks = _blocked_masks(histogram, folds)
    observed = histogram.values[histogram.support_mask]
    for label, knots in candidates.items():
        rows = []
        for fold_index, held in enumerate(held_masks):
            fit = fit_node_coefficients(
                histogram,
                knots,
                fit_mask=~held,
                starts=24,
                namespace=f"node_cv_{label}_{fold_index}",
            )
            expected = np.asarray(fit["expected_support"])
            rows.append(
                {
                    "fold": fold_index,
                    "range_gev": list(map(float, folds[fold_index])),
                    "heldout_bins": int(np.count_nonzero(held)),
                    "heldout_deviance_per_bin": poisson_deviance(
                        observed[held], expected[held]
                    )
                    / max(1, int(np.count_nonzero(held))),
                    "best_basin_repeats": int(fit["best_basin_repeats"]),
                    "optimizer_success": bool(fit["optimizer_success"]),
                    "parameters": fit["parameters"],
                }
            )
        values = np.asarray([row["heldout_deviance_per_bin"] for row in rows])
        output[label] = {
            "knots_gev": list(map(float, knots)),
            "folds": rows,
            "mean_heldout_deviance_per_bin": float(np.mean(values)),
            "standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values))),
            "pooled_heldout_deviance_per_bin": float(
                np.average(values, weights=[row["heldout_bins"] for row in rows])
            ),
        }
    best_label = min(output, key=lambda name: output[name]["mean_heldout_deviance_per_bin"])
    threshold = (
        output[best_label]["mean_heldout_deviance_per_bin"]
        + output[best_label]["standard_error"]
    )
    selected = "K2" if output["K2"]["mean_heldout_deviance_per_bin"] <= threshold else best_label
    return {
        "candidates": output,
        "best_mean_candidate": best_label,
        "one_standard_error_threshold": float(threshold),
        "selected_candidate": selected,
        "selection_rule": "one-standard-error rule favoring K2",
    }


def model_expected_support(
    model: str,
    source: str,
    histogram: Histogram,
    fit_result: Mapping[str, Any],
    *,
    observed_override: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    starts: int = 1,
    namespace: str = "evaluate",
    use_frozen_initial: bool = True,
    include_independent_data_seed: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed_support = (
        histogram.values[histogram.support_mask]
        if observed_override is None
        else np.asarray(observed_override, dtype=float)
    )
    if observed_support.shape != histogram.values[histogram.support_mask].shape:
        raise ModelError("observed override shape mismatch")
    temporary = Histogram(
        values=histogram.values.copy(),
        edges=histogram.edges,
        centers=histogram.centers,
        support_mask=histogram.support_mask,
    )
    temporary.values[temporary.support_mask] = observed_support
    if model == "knot_spline":
        knots = fit_result["models"][model]["selected_knots_gev"]
        if source == "one_pct":
            initial = (
                fit_result["models"][model]["fits"][source]["parameters"]
                if use_frozen_initial
                else None
            )
            fit = fit_node_coefficients(
                temporary,
                knots,
                fit_mask=fit_mask,
                starts=starts,
                initial=initial,
                namespace=namespace,
            )
        else:
            transfer_record = fit_result["models"][model]["fits"][source]
            initial = transfer_record["parameters"] if use_frozen_initial else None
            one = fit_result["models"][model]["fits"]["one_pct"]["parameters"]
            fit = fit_node_transfer(
                temporary,
                knots,
                one,
                transfer_record["delta_weighted_mean_basis"],
                fit_mask=fit_mask,
                starts=starts,
                initial=initial,
                namespace=namespace,
            )
    elif model == "regional_blend":
        if source == "one_pct":
            initial = (
                fit_result["models"][model]["fits"][source]["parameters"]
                if use_frozen_initial
                else None
            )
            fit = fit_regional_parameters(
                temporary,
                fit_mask=fit_mask,
                starts=starts,
                initial=initial,
                namespace=namespace,
                include_independent_data_seed=include_independent_data_seed,
            )
        else:
            transfer_record = fit_result["models"][model]["fits"][source]
            initial = transfer_record["parameters"] if use_frozen_initial else None
            one = fit_result["models"][model]["fits"]["one_pct"]["parameters"]
            fit = fit_regional_transfer(
                temporary,
                one,
                transfer_record["contrast_weighted_means"],
                fit_mask=fit_mask,
                starts=starts,
                initial=initial,
                namespace=namespace,
                include_independent_data_seed=include_independent_data_seed,
            )
    else:
        raise ModelError(f"unknown model {model}")
    return np.asarray(fit["expected_support"], dtype=float), fit


def evaluate_frozen_support(
    model: str,
    source: str,
    histogram: Histogram,
    fit_result: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> np.ndarray:
    """Evaluate stored source parameters without invoking an optimizer."""

    support_indices = np.flatnonzero(histogram.support_mask)
    edges = histogram.edges[np.r_[support_indices, support_indices[-1] + 1]]
    record = fit_result["models"][model]["fits"][source]
    normalization = float(record["normalization"])
    if model == "knot_spline":
        knots = fit_result["models"][model]["selected_knots_gev"]
        one = fit_result["models"][model]["fits"]["one_pct"]["parameters"]
        if source == "one_pct":
            shape = _node_shape(edges, knots, one)
        else:
            xq, wq = quadrature(edges)
            base_basis = natural_node_basis(xq, knots)
            delta_basis = _node_delta_basis(
                xq, knots, record["delta_weighted_mean_basis"]
            )
            log_shape = (
                _rigid_log_shape(xq)
                + np.tensordot(base_basis, np.asarray(one), axes=([-1], [0]))
                + np.tensordot(
                    delta_basis,
                    np.asarray(record["parameters"]),
                    axes=([-1], [0]),
                )
            )
            shape = _safe_shape_from_log(log_shape, wq)
    elif model == "regional_blend":
        one = fit_result["models"][model]["fits"]["one_pct"]["parameters"]
        if source == "one_pct":
            shape = _regional_shape(edges, one)
        else:
            xq, wq = quadrature(edges)
            contrast = _regional_contrast_basis(
                xq, record["contrast_weighted_means"]
            )
            log_shape = regional_log_shape(xq, one) + np.tensordot(
                contrast,
                np.asarray(record["parameters"]),
                axes=([-1], [0]),
            )
            shape = _safe_shape_from_log(log_shape, wq)
    else:
        raise ModelError(f"unknown model {model}")
    expected = normalization * shape
    if np.any(~np.isfinite(expected)) or np.any(expected <= 0):
        raise ModelError(f"invalid frozen mean: {model}/{source}")
    actual_hash = array_hash(expected, "<f8")
    if verify_hash and actual_hash != record["mean_sha256_float64"]:
        raise ModelError(
            f"frozen mean hash mismatch for {model}/{source}: "
            f"{actual_hash} != {record['mean_sha256_float64']}"
        )
    return expected


def gaussian_template(edges: np.ndarray, mass: float) -> np.ndarray:
    sigma = float(sigma_2021(float(mass)))
    cdf = norm.cdf((np.asarray(edges, dtype=float) - float(mass)) / sigma)
    values = np.diff(cdf)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 0:
        raise ModelError("empty Gaussian signal template")
    return values / total


def projection_amplitude(
    shift: np.ndarray, template: np.ndarray, mean: np.ndarray
) -> tuple[float, float]:
    inverse = 1.0 / np.clip(np.asarray(mean, dtype=float), 1e-12, None)
    template = np.asarray(template, dtype=float)
    denominator = float(np.sum(template * template * inverse))
    if denominator <= 0:
        raise ModelError("invalid template information")
    amplitude = float(np.sum(template * np.asarray(shift) * inverse) / denominator)
    sigma_a = 1.0 / math.sqrt(denominator)
    return amplitude, sigma_a


def _active_design(
    model: str,
    source: str,
    histogram: Histogram,
    fit_result: Mapping[str, Any],
    derivative_step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    centers = histogram.centers[histogram.support_mask]
    if model == "knot_spline":
        knots = fit_result["models"][model]["selected_knots_gev"]
        basis = natural_node_basis(centers, knots)
        if source == "ten_pct":
            mean = fit_result["models"][model]["fits"][source][
                "delta_weighted_mean_basis"
            ]
            basis = _node_delta_basis(centers, knots, mean)
            penalty = np.full(basis.shape[1], 1.0 / 0.002**2)
        else:
            penalty = np.zeros(basis.shape[1])
    else:
        if source == "ten_pct":
            means = fit_result["models"][model]["fits"][source][
                "contrast_weighted_means"
            ]
            basis = _regional_contrast_basis(centers, means)
            penalty = np.full(2, 1.0 / 0.002**2)
        else:
            q = np.asarray(
                fit_result["models"][model]["fits"]["one_pct"]["parameters"],
                dtype=float,
            )
            columns = []
            for index in range(q.size):
                step = (
                    float(derivative_step_scale)
                    * 1e-5
                    * max(1.0, abs(float(q[index])))
                )
                plus = q.copy()
                minus = q.copy()
                plus[index] += step
                minus[index] -= step
                columns.append(
                    (regional_log_shape(centers, plus) - regional_log_shape(centers, minus))
                    / (2.0 * step)
                )
            basis = np.stack(columns, axis=1)
            penalty = np.zeros(basis.shape[1])
    design = np.column_stack([np.ones(centers.size), basis])
    return design, np.r_[0.0, penalty]


def tangent_absorption(
    model: str,
    source: str,
    histogram: Histogram,
    fit_result: Mapping[str, Any],
    mean: np.ndarray,
    template: np.ndarray,
) -> dict[str, Any]:
    mean = np.asarray(mean, dtype=float)
    signal = np.asarray(template, dtype=float)
    denominator = float(np.sum(signal * signal / mean))
    if denominator <= 0:
        raise ModelError("invalid tangent signal information")
    derivative_scales = (0.5, 1.0, 2.0) if (model == "regional_blend" and source == "one_pct") else (1.0,)
    relative_cutoffs = (1e-12, 1e-10, 1e-8)
    solutions = []
    for derivative_scale in derivative_scales:
        design, penalty = _active_design(
            model,
            source,
            histogram,
            fit_result,
            derivative_step_scale=derivative_scale,
        )
        information = design.T @ (mean[:, None] * design) + np.diag(penalty)
        information = 0.5 * (information + information.T)
        score = design.T @ signal
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        maximum = float(np.max(eigenvalues))
        positive = eigenvalues[eigenvalues > 0]
        condition = (
            float(maximum / np.min(positive)) if positive.size and maximum > 0 else None
        )
        for cutoff in relative_cutoffs:
            keep = eigenvalues > maximum * cutoff
            if not np.any(keep):
                absorption = 0.0
            else:
                coordinates = eigenvectors[:, keep].T @ score
                delta = eigenvectors[:, keep] @ (coordinates / eigenvalues[keep])
                learned = mean * (design @ delta)
                absorption = float(np.sum(signal * learned / mean) / denominator)
            solutions.append(
                {
                    "derivative_step_scale": float(derivative_scale),
                    "relative_eigenvalue_cutoff": float(cutoff),
                    "effective_rank": int(np.count_nonzero(keep)),
                    "n_directions": int(eigenvalues.size),
                    "condition_number_positive_spectrum": condition,
                    "minimum_eigenvalue": float(np.min(eigenvalues)),
                    "maximum_eigenvalue": maximum,
                    "absorption_fraction": absorption,
                }
            )
    selected = max(solutions, key=lambda row: abs(row["absorption_fraction"]))
    values = np.asarray([row["absorption_fraction"] for row in solutions], dtype=float)
    return {
        "conservative_absorption_fraction": float(selected["absorption_fraction"]),
        "minimum_absorption_fraction_across_solvers": float(np.min(values)),
        "maximum_absorption_fraction_across_solvers": float(np.max(values)),
        "maximum_abs_solver_spread": float(np.max(values) - np.min(values)),
        "solutions": solutions,
    }


def fitted_feature_scale_audit(
    model: str,
    source: str,
    histogram: Histogram,
    fit_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that the fitted correction has no signal-width peak/trough pair."""

    grid = np.linspace(SUPPORT[0], SUPPORT[1], 5201)
    if model == "knot_spline":
        knots = fit_result["models"][model]["selected_knots_gev"]
        basis = natural_node_basis(grid, knots)
        one = np.asarray(
            fit_result["models"][model]["fits"]["one_pct"]["parameters"],
            dtype=float,
        )
        correction = basis @ one
        if source == "ten_pct":
            mean = fit_result["models"][model]["fits"][source][
                "delta_weighted_mean_basis"
            ]
            delta_basis = _node_delta_basis(grid, knots, mean)
            delta = np.asarray(
                fit_result["models"][model]["fits"][source]["parameters"],
                dtype=float,
            )
            correction = correction + delta_basis @ delta
    elif model == "regional_blend":
        one = np.asarray(
            fit_result["models"][model]["fits"]["one_pct"]["parameters"],
            dtype=float,
        )
        correction = regional_log_shape(grid, one) - _rigid_log_shape(grid)
        if source == "ten_pct":
            means = fit_result["models"][model]["fits"][source][
                "contrast_weighted_means"
            ]
            contrast = _regional_contrast_basis(grid, means)
            delta = np.asarray(
                fit_result["models"][model]["fits"][source]["parameters"],
                dtype=float,
            )
            correction = correction + contrast @ delta
    else:
        raise ModelError(f"unknown model {model}")

    derivative = np.gradient(correction, grid)
    signs = np.sign(derivative)
    nonzero = np.flatnonzero(signs)
    if nonzero.size:
        signs[: nonzero[0]] = signs[nonzero[0]]
        for index in range(nonzero[0] + 1, signs.size):
            if signs[index] == 0:
                signs[index] = signs[index - 1]
    extrema = np.flatnonzero(signs[1:] * signs[:-1] < 0) + 1
    extrema = extrema[(grid[extrema] >= PRIMARY[0]) & (grid[extrema] <= PRIMARY[1])]
    pairs = []
    for left, right in zip(extrema[:-1], extrema[1:]):
        midpoint = 0.5 * (grid[left] + grid[right])
        required = 4.5 * float(sigma_2021(midpoint))
        separation = float(grid[right] - grid[left])
        pairs.append(
            {
                "left_mass_GeV": float(grid[left]),
                "right_mass_GeV": float(grid[right]),
                "separation_GeV": separation,
                "required_local_4p5sigma_GeV": required,
                "separation_over_required": separation / required,
            }
        )
    minimum_ratio = min(
        (row["separation_over_required"] for row in pairs), default=None
    )
    return {
        "definition": "adjacent extrema of fitted log-mean correction relative to the v4.8 rigid mean",
        "n_extrema_primary": int(extrema.size),
        "extrema_masses_GeV": grid[extrema].tolist(),
        "adjacent_peak_trough_pairs": pairs,
        "minimum_separation_over_local_full_4p5sigma": minimum_ratio,
        "descriptive_condition_met": bool(
            minimum_ratio is None or minimum_ratio >= 1.0
        ),
    }


def influence_audit(fit_result: Mapping[str, Any]) -> dict[str, Any]:
    payload = protocol()
    grid_spec = payload["signal_influence_audit"]["mass_grid_gev"]
    masses = np.arange(
        float(grid_spec["start"]),
        float(grid_spec["stop"]) + 0.5 * float(grid_spec["step"]),
        float(grid_spec["step"]),
    )
    strengths = list(map(float, payload["signal_influence_audit"]["injected_diagonal_poisson_sigmas"]))
    rows = []
    tangent_details = []
    for model in ("knot_spline", "regional_blend"):
        for source in ("one_pct", "ten_pct"):
            histogram = load_histogram(source, payload)
            observed = histogram.values[histogram.support_mask]
            nominal = evaluate_frozen_support(model, source, histogram, fit_result)
            edges = histogram.edges[
                np.r_[np.flatnonzero(histogram.support_mask), np.flatnonzero(histogram.support_mask)[-1] + 1]
            ]
            centers = histogram.centers[histogram.support_mask]
            for mass in masses:
                template = gaussian_template(edges, float(mass))
                tangent_record = tangent_absorption(
                    model, source, histogram, fit_result, nominal, template
                )
                tangent = tangent_record["conservative_absorption_fraction"]
                tangent_details.append(
                    {
                        "model": model,
                        "source": source,
                        "mass_GeV": float(mass),
                        "mass_MeV": float(mass * 1000.0),
                        **tangent_record,
                    }
                )
                sigma = float(sigma_2021(float(mass)))
                _, sigma_a = projection_amplitude(
                    np.zeros_like(nominal), template, nominal
                )
                gap = np.abs(centers - float(mass)) >= 2.25 * sigma
                try:
                    gap_mean, gap_fit = model_expected_support(
                        model,
                        source,
                        histogram,
                        fit_result,
                        fit_mask=gap,
                        starts=4,
                        namespace=f"gap_{model}_{source}_{mass:.3f}",
                        use_frozen_initial=False,
                    )
                    gap_amplitude, _ = projection_amplitude(
                        gap_mean - nominal, template, nominal
                    )
                    gap_error = None
                except Exception as error:
                    gap_fit = None
                    gap_amplitude = None
                    gap_error = f"{type(error).__name__}: {error}"
                for z_value in strengths:
                    amplitude = z_value * sigma_a
                    injected = amplitude * template
                    try:
                        refit_mean, refit = model_expected_support(
                            model,
                            source,
                            histogram,
                            fit_result,
                            observed_override=observed + injected,
                            starts=4,
                            namespace=f"inj_{model}_{source}_{mass:.3f}_{z_value:.1f}",
                            include_independent_data_seed=True,
                        )
                        learned_amplitude, _ = projection_amplitude(
                            refit_mean - nominal, template, nominal
                        )
                        fraction = learned_amplitude / amplitude
                        refit_error = None
                    except Exception as error:
                        refit = None
                        fraction = None
                        refit_error = f"{type(error).__name__}: {error}"

                    def constraints_feasible(record: Mapping[str, Any]) -> bool:
                        margins = [
                            float(record[key])
                            for key in ("minimum_constraint_margin", "high_monotone_margin")
                            if key in record
                        ]
                        return all(value >= -1e-8 for value in margins)

                    rows.append(
                        {
                            "model": model,
                            "source": source,
                            "mass_GeV": float(mass),
                            "mass_MeV": float(mass * 1000.0),
                            "z": float(z_value),
                            "sigmaA_diagonal_poisson": sigma_a,
                            "injected_amplitude": amplitude,
                            "refit_absorption_fraction": fraction,
                            "z_times_abs_absorption": (
                                z_value * abs(fraction) if fraction is not None else None
                            ),
                            "gap_projection_amplitude": gap_amplitude,
                            "gap_abs_shift_sigmaA": (
                                abs(gap_amplitude) / sigma_a
                                if gap_amplitude is not None
                                else None
                            ),
                            "tangent_absorption_fraction": tangent,
                            "tangent_solver_absorption_spread": tangent_record[
                                "maximum_abs_solver_spread"
                            ],
                            "refit_best_basin_repeats": (
                                int(refit["best_basin_repeats"]) if refit else 0
                            ),
                            "gap_best_basin_repeats": (
                                int(gap_fit["best_basin_repeats"]) if gap_fit else 0
                            ),
                            "refit_optimizer_success": bool(
                                refit and refit["optimizer_success"]
                            ),
                            "gap_optimizer_success": bool(
                                gap_fit and gap_fit["optimizer_success"]
                            ),
                            "refit_constraints_feasible": bool(
                                refit and constraints_feasible(refit)
                            ),
                            "gap_constraints_feasible": bool(
                                gap_fit and constraints_feasible(gap_fit)
                            ),
                            "refit_failure": refit_error,
                            "gap_failure": gap_error,
                        }
                    )
    gates = payload["signal_influence_audit"]
    feature_scales: dict[str, Any] = {}
    for model in ("knot_spline", "regional_blend"):
        feature_scales[model] = {}
        for source in ("one_pct", "ten_pct"):
            feature_scales[model][source] = fitted_feature_scale_audit(
                model, source, load_histogram(source, payload), fit_result
            )

    summaries = {}
    for model in ("knot_spline", "regional_blend"):
        selected = [row for row in rows if row["model"] == model]
        finite_gap = [
            row["gap_abs_shift_sigmaA"]
            for row in selected
            if row["gap_abs_shift_sigmaA"] is not None
        ]
        finite_refit = [
            row["z_times_abs_absorption"]
            for row in selected
            if row["z_times_abs_absorption"] is not None
        ]
        maximum_gap = max(finite_gap) if finite_gap else None
        maximum_refit = max(finite_refit) if finite_refit else None
        maximum_tangent = max(abs(row["tangent_absorption_fraction"]) for row in selected)
        minimum_audit_repeats = int(
            gates["audit_refit_integrity"][
                "minimum_best_basin_repeats_of_four_starts"
            ]
        )
        fit_integrity = all(
            row["refit_optimizer_success"]
            and row["gap_optimizer_success"]
            and row["refit_constraints_feasible"]
            and row["gap_constraints_feasible"]
            and int(row["refit_best_basin_repeats"]) >= minimum_audit_repeats
            and int(row["gap_best_basin_repeats"]) >= minimum_audit_repeats
            for row in selected
        )
        summaries[model] = {
            "maximum_gap_abs_shift_sigmaA": maximum_gap,
            "maximum_z_times_abs_absorption": maximum_refit,
            "maximum_abs_tangent_absorption_fraction": maximum_tangent,
            "gap_missing_cells": int(len(selected) - len(finite_gap)),
            "refit_missing_cells": int(len(selected) - len(finite_refit)),
            "gap_gate_passed": bool(
                len(finite_gap) == len(selected)
                and maximum_gap is not None
                and maximum_gap <= float(gates["gap_projection_abs_sigma_max"])
            ),
            "refit_gate_passed": bool(
                len(finite_refit) == len(selected)
                and maximum_refit is not None
                and maximum_refit
                <= float(gates["refit_z_times_absorption_fraction_max"])
            ),
            "tangent_gate_passed": maximum_tangent <= float(gates["tangent_absorption_fraction_max"]),
            "descriptive_feature_scale_condition_met": all(
                feature_scales[model][source]["descriptive_condition_met"]
                for source in ("one_pct", "ten_pct")
            ),
            "audit_refit_integrity_gate_passed": fit_integrity,
        }
        summaries[model]["signal_influence_gate_passed"] = bool(
            summaries[model]["gap_gate_passed"]
            and summaries[model]["refit_gate_passed"]
            and summaries[model]["tangent_gate_passed"]
            and summaries[model]["audit_refit_integrity_gate_passed"]
        )
    return {
        "rows": rows,
        "tangent_solver_details": tangent_details,
        "fitted_feature_scale_descriptor": feature_scales,
        "summaries": summaries,
    }


def _source_qualification(
    metrics: Mapping[str, Any], fit: Mapping[str, Any], heldout: Mapping[str, Any] | None
) -> dict[str, Any]:
    native = all(
        0.75 <= float(metrics[key]) <= 1.25
        for key in ("pearson_per_ndf", "deviance_per_ndf", "rebin5_pearson_per_ndf", "rebin5_deviance_per_ndf")
    )
    reproducible = int(fit["best_basin_repeats"]) >= 3
    away_from_bound = float(fit.get("minimum_normalized_bound_distance", 1.0)) >= 0.02
    heldout_deviance_pass = bool(
        heldout is not None
        and float(heldout["pooled_heldout_deviance_per_bin"]) <= 1.25
        and all(float(row["heldout_deviance_per_bin"]) <= 1.25 for row in heldout["folds"])
    )
    heldout_integrity_pass = bool(
        heldout is not None
        and all(
            bool(row["optimizer_success"])
            and bool(row["constraints_feasible"])
            and int(row["best_basin_repeats"]) >= 3
            for row in heldout["folds"]
        )
    )
    heldout_pass = heldout_deviance_pass and heldout_integrity_pass
    return {
        "native_and_rebin5_gof_gate_passed": native,
        "best_basin_reproducibility_gate_passed": reproducible,
        "bound_distance_gate_passed": away_from_bound,
        "heldout_deviance_gate_passed": heldout_deviance_pass,
        "heldout_fit_integrity_gate_passed": heldout_integrity_pass,
        "heldout_gate_passed": heldout_pass,
        "strict_source_qualification_passed": bool(native and reproducible and away_from_bound and heldout_pass),
    }


def _heldout_for_fixed_model(
    model: str,
    source: str,
    histogram: Histogram,
    fit_result: Mapping[str, Any],
    folds: Sequence[Sequence[float]],
) -> dict[str, Any]:
    observed = histogram.values[histogram.support_mask]
    rows = []
    for index, held in enumerate(_blocked_masks(histogram, folds)):
        expected, fit = model_expected_support(
            model,
            source,
            histogram,
            fit_result,
            fit_mask=~held,
            starts=8,
            namespace=f"heldout_{model}_{source}_{index}",
            use_frozen_initial=False,
        )
        rows.append(
            {
                "fold": index,
                "range_gev": list(map(float, folds[index])),
                "heldout_bins": int(np.count_nonzero(held)),
                "heldout_deviance_per_bin": poisson_deviance(observed[held], expected[held]) / max(1, int(np.count_nonzero(held))),
                "best_basin_repeats": int(fit["best_basin_repeats"]),
                "optimizer_success": bool(fit["optimizer_success"]),
                "constraints_feasible": all(
                    float(fit[key]) >= -1e-8
                    for key in ("minimum_constraint_margin", "high_monotone_margin")
                    if key in fit
                ),
            }
        )
    return {
        "folds": rows,
        "pooled_heldout_deviance_per_bin": float(
            np.average(
                [row["heldout_deviance_per_bin"] for row in rows],
                weights=[row["heldout_bins"] for row in rows],
            )
        ),
    }


def fit_all_sources() -> dict[str, Any]:
    payload = protocol()
    inputs = validate_declared_inputs(payload)
    one = load_histogram("one_pct", payload)
    ten = load_histogram("ten_pct", payload)
    if not np.array_equal(one.edges, ten.edges):
        raise ModelError("1pct and 10pct source edges differ")
    folds = payload["source_fit"]["blocked_folds_gev"]
    candidate_knots = payload["models"]["knot_spline"]["candidate_interior_knots_gev"]
    cross_validation = node_cross_validation(one, candidate_knots, folds)
    selected_label = cross_validation["selected_candidate"]
    selected_knots = candidate_knots[selected_label]
    node_one = fit_node_coefficients(one, selected_knots, starts=24, namespace="node_final_one")
    node_delta_mean = _node_basis_mean(selected_knots, one)
    node_ten = fit_node_transfer(
        ten,
        selected_knots,
        node_one["parameters"],
        node_delta_mean,
        starts=24,
        namespace="node_final_ten",
    )
    regional_one = fit_regional_parameters(one, starts=24, namespace="regional_final_one")
    regional_contrast_means = _regional_contrast_means(one)
    regional_ten = fit_regional_transfer(
        ten,
        regional_one["parameters"],
        regional_contrast_means,
        starts=24,
        namespace="regional_final_ten",
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "study_id": payload["study_id"],
        "protocol_path": PROTOCOL_PATH.name,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "implementation_path": Path(__file__).name,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "driver_path": DRIVER_PATH.name,
        "driver_sha256": sha256_file(DRIVER_PATH),
        "model_selection_frozen_before_injection": True,
        "model_selection_uses_gpr_results": False,
        "input_validation": inputs,
        "source_totals_040_300": {
            "one_pct": float(np.sum(one.values[one.support_mask])),
            "ten_pct": float(np.sum(ten.values[ten.support_mask])),
        },
        "models": {
            "knot_spline": {
                "selected_candidate": selected_label,
                "selected_knots_gev": list(map(float, selected_knots)),
                "source_only_cross_validation": cross_validation,
                "fits": {
                    "one_pct": _strip_fit_arrays(node_one),
                    "ten_pct": _strip_fit_arrays(node_ten),
                },
            },
            "regional_blend": {
                "fits": {
                    "one_pct": _strip_fit_arrays(regional_one),
                    "ten_pct": _strip_fit_arrays(regional_ten),
                }
            },
        },
    }
    for model, records in (
        ("knot_spline", (node_one, node_ten)),
        ("regional_blend", (regional_one, regional_ten)),
    ):
        for source, histogram, record in zip(("one_pct", "ten_pct"), (one, ten), records):
            expected = np.asarray(record["expected_support"], dtype=float)
            observed = histogram.values[histogram.support_mask]
            result["models"][model]["fits"][source]["metrics"] = fit_metrics(
                observed, expected, len(record["parameters"])
            )
            primary_mask = (
                (histogram.centers[histogram.support_mask] >= PRIMARY[0] - 1e-12)
                & (histogram.centers[histogram.support_mask] <= PRIMARY[1] + 1e-12)
            )
            result["models"][model]["fits"][source]["metrics_scope_gev"] = list(
                SUPPORT
            )
            result["models"][model]["fits"][source]["primary_metrics"] = fit_metrics(
                observed[primary_mask],
                expected[primary_mask],
                len(record["parameters"]),
            )
            result["models"][model]["fits"][source]["primary_metrics_scope_gev"] = list(
                PRIMARY
            )
            result["models"][model]["fits"][source]["mean_sha256_float64"] = array_hash(expected, "<f8")

    # Held-out summaries are computed after the final parameterization and
    # transfer restrictions are frozen.  They can reject qualification but do
    # not change the selected model.
    for model in ("knot_spline", "regional_blend"):
        for source, histogram in (("one_pct", one), ("ten_pct", ten)):
            heldout = _heldout_for_fixed_model(model, source, histogram, result, folds)
            record = result["models"][model]["fits"][source]
            record["heldout"] = heldout
            record["qualification"] = _source_qualification(
                record["metrics"], record, heldout
            )
    summary_rows = []
    for model in ("knot_spline", "regional_blend"):
        for source in ("one_pct", "ten_pct"):
            record = result["models"][model]["fits"][source]
            summary_rows.append(
                {
                    "model": model,
                    "source": source,
                    "support_deviance_per_ndf": record["metrics"]["deviance_per_ndf"],
                    "support_pearson_per_ndf": record["metrics"]["pearson_per_ndf"],
                    "support_rebin5_deviance_per_ndf": record["metrics"]["rebin5_deviance_per_ndf"],
                    "support_rebin5_pearson_per_ndf": record["metrics"]["rebin5_pearson_per_ndf"],
                    "primary_deviance_per_ndf": record["primary_metrics"]["deviance_per_ndf"],
                    "primary_pearson_per_ndf": record["primary_metrics"]["pearson_per_ndf"],
                    "pooled_heldout_deviance_per_bin": record["heldout"]["pooled_heldout_deviance_per_bin"],
                    "best_basin_repeats": record["best_basin_repeats"],
                    "minimum_normalized_bound_distance": record.get("minimum_normalized_bound_distance", ""),
                    "strict_source_qualification_passed": record["qualification"]["strict_source_qualification_passed"],
                }
            )
    atomic_csv(FIT_SUMMARY_CSV, summary_rows, tuple(summary_rows[0].keys()))
    result["source_fit_summary_csv"] = {
        "path": str(FIT_SUMMARY_CSV.relative_to(HERE)),
        "sha256": sha256_file(FIT_SUMMARY_CSV),
    }
    atomic_json(FIT_RESULT_PATH, result)
    return result


def append_influence(result: Mapping[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(result, default=_json_default))
    updated["signal_influence_audit"] = influence_audit(updated)
    for model in ("knot_spline", "regional_blend"):
        source_pass = all(
            updated["models"][model]["fits"][source]["qualification"]["strict_source_qualification_passed"]
            for source in ("one_pct", "ten_pct")
        )
        influence_pass = updated["signal_influence_audit"]["summaries"][model]["signal_influence_gate_passed"]
        updated["models"][model]["strict_generator_qualification_passed"] = bool(
            source_pass and influence_pass
        )
        updated["models"][model]["conditional_toy_run_authorized"] = True
        updated["models"][model]["promotion_scope"] = (
            "qualified conditional generator" if source_pass and influence_pass else "requested conditional stress only"
        )
    influence_rows = updated["signal_influence_audit"]["rows"]
    atomic_csv(INFLUENCE_CSV, influence_rows, tuple(influence_rows[0].keys()))
    updated["signal_influence_audit_csv"] = {
        "path": str(INFLUENCE_CSV.relative_to(HERE)),
        "sha256": sha256_file(INFLUENCE_CSV),
    }
    atomic_json(FIT_RESULT_PATH, updated)
    return updated


def load_fit_result(require_influence: bool = True) -> dict[str, Any]:
    if not FIT_RESULT_PATH.is_file():
        raise ModelError(f"missing source fit result: {FIT_RESULT_PATH}")
    result = load_json(FIT_RESULT_PATH)
    if result.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ModelError("source fit result is stale relative to MODEL_PROTOCOL")
    if result.get("implementation_sha256") != sha256_file(Path(__file__).resolve()):
        raise ModelError("source fit result is stale relative to residual_models.py")
    if result.get("driver_sha256") != sha256_file(DRIVER_PATH):
        raise ModelError("source fit result is stale relative to fit_residual_models.py")
    current_inputs = validate_declared_inputs(protocol())
    if result.get("input_validation") != current_inputs:
        raise ModelError("source fit result is stale relative to declared inputs")
    if require_influence and "signal_influence_audit" not in result:
        raise ModelError("source influence audit has not been completed")
    return result


def frozen_mean_full(model: str, source: str, fit_result: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    fit_result = load_fit_result() if fit_result is None else fit_result
    histogram = load_histogram(source)
    support_expected = evaluate_frozen_support(model, source, histogram, fit_result)
    full = np.zeros_like(histogram.values, dtype=float)
    full[histogram.support_mask] = support_expected
    return full, histogram.edges
