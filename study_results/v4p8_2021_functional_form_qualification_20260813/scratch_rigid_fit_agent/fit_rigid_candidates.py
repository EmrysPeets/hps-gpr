#!/usr/bin/env python3
"""Exploratory low-parameter source fits for the v4.8 functional-form study.

This is deliberately isolated scratch work.  It does not produce toys or alter
the active v4.8 qualification state.  Two six-free-parameter families are fit:

  ggt6: A sigmoid((x-xt)/w) (x-x0)^a exp(-((x-x0)/lambda)^p),
        with x0 fixed from a seven-parameter 1% reconnaissance fit;

  spq6: A sigmoid((x-xt)/w) x^a exp(c1*u + c2*u^2),
        u=(x-xmid)/xscale.

The normalization A is profiled analytically but counted as a free parameter.
Native/rebinned Poisson diagnostics are kept separate from diagonal diagnostics
against an adaptively smoothed engineering target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import uproot
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from numpy.polynomial.chebyshev import chebvander


HERE = Path(__file__).resolve().parent
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
SUPPORT_HI = 0.300
SUPPORT_LOWS = (0.030, 0.040)
SMOOTH_MULTIPLIERS = (0.0, 2.25)
REBIN_FACTORS = (1, 5, 20, 40, 80)
SIGMA_COEFFS = (0.00184825, -0.001375, 0.085875)
BASE_SEED = 20260813


def sigma_2021(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return SIGMA_COEFFS[0] + SIGMA_COEFFS[1] * x + SIGMA_COEFFS[2] * x * x


def load_source(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with uproot.open(path) as handle:
        values, edges = handle[key].to_numpy(flow=False)
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return values, centers, edges


def adaptive_smooth(
    values: np.ndarray,
    centers: np.ndarray,
    output_centers: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Return locally Gaussian-smoothed native-bin counts.

    The full histogram is used as padding, and each output point has bandwidth
    ``multiplier * sigma_2021(x)``.  Rows are normalized, so a flat spectrum is
    unchanged.  The result is rescaled later to the support count.
    """
    if multiplier <= 0:
        index = np.searchsorted(centers, output_centers)
        index = np.clip(index, 0, centers.size - 1)
        return np.asarray(values[index], dtype=float)
    result = np.empty(output_centers.size, dtype=float)
    for i, x in enumerate(output_centers):
        width = float(multiplier * sigma_2021(np.asarray([x]))[0])
        window = np.abs(centers - x) <= 4.5 * width
        delta = (centers[window] - x) / width
        weight = np.exp(-0.5 * delta * delta)
        result[i] = float(np.dot(weight, values[window]) / np.sum(weight))
    return result


def rebin_sum(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.size // factor * factor
    return np.asarray(values[:usable], dtype=float).reshape(-1, factor).sum(axis=1)


def metrics(observed: np.ndarray, expected: np.ndarray, npars: int) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    expected = np.clip(np.asarray(expected, dtype=float), 1e-300, None)
    ndf = max(1, observed.size - int(npars))
    pearson = np.sum((observed - expected) ** 2 / expected)
    term = np.where(
        observed > 0,
        observed * np.log(observed / expected) - (observed - expected),
        expected,
    )
    return {
        "pearson_chi2ndf": float(pearson / ndf),
        "poisson_deviance_ndf": float(2.0 * np.sum(term) / ndf),
        "max_abs_pearson_residual": float(
            np.max(np.abs(observed - expected) / np.sqrt(expected))
        ),
        "ndf": int(ndf),
    }


@dataclass(frozen=True)
class Family:
    name: str
    n_shape: int
    bounds: tuple[tuple[float, float], ...]
    seed: np.ndarray
    logshape: Callable[[np.ndarray, np.ndarray, dict[str, float]], np.ndarray]


@dataclass
class Fit:
    family: str
    shape: np.ndarray
    log_a: float
    objective: float
    success: bool
    status: int
    message: str
    iterations: int
    restart: int
    parameters: dict[str, float]
    expected: np.ndarray


def logshape_ggt6(x: np.ndarray, q: np.ndarray, fixed: dict[str, float]) -> np.ndarray:
    a, log_lam, log_p, xt, log_w = q
    lam = math.exp(log_lam)
    p = math.exp(log_p)
    w = math.exp(log_w)
    z = x - fixed["x0"]
    if np.any(z <= 0):
        return np.full_like(x, -1e100)
    turn = np.clip(expit((x - xt) / w), 1e-300, 1.0)
    return np.log(turn) + a * np.log(z) - np.power(z / lam, p)


def logshape_ggt7_recon(x: np.ndarray, q: np.ndarray, fixed: dict[str, float]) -> np.ndarray:
    a, log_lam, log_p, x0, xt, log_w = q
    return logshape_ggt6(x, np.asarray([a, log_lam, log_p, xt, log_w]), {"x0": x0})


def logshape_spq6(x: np.ndarray, q: np.ndarray, fixed: dict[str, float]) -> np.ndarray:
    a, c1, c2, xt, log_w = q
    w = math.exp(log_w)
    u = (x - fixed["xmid"]) / fixed["xscale"]
    turn = np.clip(expit((x - xt) / w), 1e-300, 1.0)
    return np.log(turn) + a * np.log(x) + c1 * u + c2 * u * u


def logshape_cheb5_fixed_turn(
    x: np.ndarray, q: np.ndarray, fixed: dict[str, float]
) -> np.ndarray:
    u = 2.0 * (x - fixed["support_lo"]) / (
        fixed["support_hi"] - fixed["support_lo"]
    ) - 1.0
    matrix = chebvander(u, 5)[:, 1:]
    turn = np.clip(
        expit((x - fixed["xt_fixed"]) / fixed["w_fixed"]), 1e-300, 1.0
    )
    return np.log(turn) + matrix @ q


def logshape_ggtq6_fixed_x0w(
    x: np.ndarray, q: np.ndarray, fixed: dict[str, float]
) -> np.ndarray:
    a, log_lam, log_p, xt, c2 = q
    lam = math.exp(log_lam)
    p = math.exp(log_p)
    z = x - fixed["x0"]
    if np.any(z <= 0):
        return np.full_like(x, -1e100)
    u = (x - fixed["xmid"]) / fixed["xscale"]
    turn = np.clip(expit((x - xt) / fixed["w_fixed"]), 1e-300, 1.0)
    return np.log(turn) + a * np.log(z) - np.power(z / lam, p) + c2 * u * u


def logshape_pow2exp6_fixed_w(
    x: np.ndarray, q: np.ndarray, fixed: dict[str, float]
) -> np.ndarray:
    a, log_t1, log_t2, log_r, xt = q
    t1 = math.exp(log_t1)
    t2 = math.exp(log_t2)
    turn = np.clip(expit((x - xt) / fixed["w_fixed"]), 1e-300, 1.0)
    mixture = np.logaddexp(-x / t1, log_r - x / t2)
    return np.log(turn) + a * np.log(x) + mixture


def logshape_ggt34_6_fixed_turn(
    x: np.ndarray, q: np.ndarray, fixed: dict[str, float]
) -> np.ndarray:
    a, log_lam, log_p, d3, d4 = q
    lam = math.exp(log_lam)
    p = math.exp(log_p)
    z = x - fixed["x0"]
    if np.any(z <= 0):
        return np.full_like(x, -1e100)
    u = 2.0 * (x - fixed["support_lo"]) / (
        fixed["support_hi"] - fixed["support_lo"]
    ) - 1.0
    cheb = chebvander(u, 4)
    turn = np.clip(
        expit((x - fixed["xt_fixed"]) / fixed["w_fixed"]), 1e-300, 1.0
    )
    return (
        np.log(turn)
        + a * np.log(z)
        - np.power(z / lam, p)
        + d3 * cheb[:, 3]
        + d4 * cheb[:, 4]
    )


def logshape_ggt26_6_fixed_turn(
    x: np.ndarray, q: np.ndarray, fixed: dict[str, float]
) -> np.ndarray:
    a, log_lam, log_p, d2, d6 = q
    lam = math.exp(log_lam)
    p = math.exp(log_p)
    z = x - fixed["x0"]
    if np.any(z <= 0):
        return np.full_like(x, -1e100)
    u = 2.0 * (x - fixed["support_lo"]) / (
        fixed["support_hi"] - fixed["support_lo"]
    ) - 1.0
    cheb = chebvander(u, 6)
    turn = np.clip(
        expit((x - fixed["xt_fixed"]) / fixed["w_fixed"]), 1e-300, 1.0
    )
    return (
        np.log(turn)
        + a * np.log(z)
        - np.power(z / lam, p)
        + d2 * cheb[:, 2]
        + d6 * cheb[:, 6]
    )


def profiled_model(
    x: np.ndarray,
    target: np.ndarray,
    family: Family,
    q: np.ndarray,
    fixed: dict[str, float],
) -> tuple[np.ndarray, float]:
    log_shape = family.logshape(x, q, fixed)
    if np.any(~np.isfinite(log_shape)) or np.max(log_shape) < -1e90:
        return np.full_like(target, 1e-300), -1e100
    log_a = math.log(float(np.sum(target))) - float(logsumexp(log_shape))
    log_mean = np.clip(log_a + log_shape, -700.0, 700.0)
    return np.exp(log_mean), log_a


def fit_family(
    x: np.ndarray,
    target: np.ndarray,
    family: Family,
    fixed: dict[str, float],
    *,
    seed_override: np.ndarray | None = None,
    n_restarts: int = 24,
    rng_seed: int = BASE_SEED,
) -> tuple[Fit, list[Fit]]:
    target = np.asarray(target, dtype=float)
    total = float(np.sum(target))
    if total <= 0 or np.any(target < 0):
        raise RuntimeError("invalid target")

    def objective(q: np.ndarray) -> float:
        mean, _ = profiled_model(x, target, family, q, fixed)
        if np.any(~np.isfinite(mean)) or np.any(mean <= 0):
            return 1e100
        # The mean normalization is profiled, so the sum(mu) term is constant.
        return float(-np.dot(target, np.log(mean)) / total)

    rng = np.random.default_rng(rng_seed)
    starts: list[np.ndarray] = []
    if seed_override is not None:
        starts.append(np.asarray(seed_override, dtype=float))
    starts.append(np.asarray(family.seed, dtype=float))
    while len(starts) < n_restarts:
        q = np.asarray(
            [rng.uniform(low, high) for low, high in family.bounds], dtype=float
        )
        starts.append(q)

    results: list[Fit] = []
    for restart, start in enumerate(starts):
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            bounds=family.bounds,
            options={
                "maxiter": 12000,
                "xatol": 2e-10,
                "fatol": 1e-13,
                "adaptive": True,
            },
        )
        # Clip defensively, then polish with bounded L-BFGS-B.
        clipped = np.asarray(
            [np.clip(v, lo, hi) for v, (lo, hi) in zip(result.x, family.bounds)],
            dtype=float,
        )
        polished = minimize(
            objective,
            clipped,
            method="L-BFGS-B",
            bounds=family.bounds,
            options={"maxiter": 8000, "ftol": 1e-15, "gtol": 1e-9, "maxls": 80},
        )
        mean, log_a = profiled_model(x, target, family, polished.x, fixed)
        pars = decode_parameters(family.name, polished.x, log_a, fixed)
        results.append(
            Fit(
                family=family.name,
                shape=np.asarray(polished.x, dtype=float),
                log_a=float(log_a),
                objective=float(polished.fun),
                success=bool(polished.success),
                status=int(polished.status),
                message=str(polished.message),
                iterations=int(polished.nit),
                restart=int(restart),
                parameters=pars,
                expected=mean,
            )
        )
    results.sort(key=lambda item: item.objective)
    return results[0], results


def decode_parameters(
    family: str, q: np.ndarray, log_a: float, fixed: dict[str, float]
) -> dict[str, float]:
    if family == "ggt6_fixed_x0":
        a, log_lam, log_p, xt, log_w = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "lambda": math.exp(log_lam),
            "power": math.exp(log_p),
            "x0_fixed": fixed["x0"],
            "xt": float(xt),
            "w": math.exp(log_w),
        }
    if family == "ggt7_recon":
        a, log_lam, log_p, x0, xt, log_w = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "lambda": math.exp(log_lam),
            "power": math.exp(log_p),
            "x0": float(x0),
            "xt": float(xt),
            "w": math.exp(log_w),
        }
    if family == "spq6_identifiable":
        a, c1, c2, xt, log_w = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "c1_scaled": float(c1),
            "c2_scaled": float(c2),
            "xmid_fixed": fixed["xmid"],
            "xscale_fixed": fixed["xscale"],
            "xt": float(xt),
            "w": math.exp(log_w),
        }
    if family == "cheb5_fixed_turn":
        return {
            "A": math.exp(log_a),
            **{f"c{index}": float(value) for index, value in enumerate(q, start=1)},
            "xt_fixed": fixed["xt_fixed"],
            "w_fixed": fixed["w_fixed"],
            "support_lo_fixed": fixed["support_lo"],
            "support_hi_fixed": fixed["support_hi"],
        }
    if family == "ggtq6_fixed_x0w":
        a, log_lam, log_p, xt, c2 = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "lambda": math.exp(log_lam),
            "power": math.exp(log_p),
            "x0_fixed": fixed["x0"],
            "w_fixed": fixed["w_fixed"],
            "xt": float(xt),
            "c2_scaled": float(c2),
            "xmid_fixed": fixed["xmid"],
            "xscale_fixed": fixed["xscale"],
        }
    if family == "pow2exp6_fixed_w":
        a, log_t1, log_t2, log_r, xt = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "theta_short": math.exp(log_t1),
            "theta_long": math.exp(log_t2),
            "relative_long_amplitude": math.exp(log_r),
            "xt": float(xt),
            "w_fixed": fixed["w_fixed"],
        }
    if family == "ggt34_6_fixed_turn":
        a, log_lam, log_p, d3, d4 = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "lambda": math.exp(log_lam),
            "power": math.exp(log_p),
            "d3": float(d3),
            "d4": float(d4),
            "x0_fixed": fixed["x0"],
            "xt_fixed": fixed["xt_fixed"],
            "w_fixed": fixed["w_fixed"],
            "support_lo_fixed": fixed["support_lo"],
            "support_hi_fixed": fixed["support_hi"],
        }
    if family == "ggt26_6_fixed_turn":
        a, log_lam, log_p, d2, d6 = q
        return {
            "A": math.exp(log_a),
            "a": float(a),
            "lambda": math.exp(log_lam),
            "power": math.exp(log_p),
            "d2": float(d2),
            "d6": float(d6),
            "x0_fixed": fixed["x0"],
            "xt_fixed": fixed["xt_fixed"],
            "w_fixed": fixed["w_fixed"],
            "support_lo_fixed": fixed["support_lo"],
            "support_hi_fixed": fixed["support_hi"],
        }
    raise KeyError(family)


def families(support_lo: float, x0: float | None = None) -> dict[str, Family]:
    xmid = 0.5 * (support_lo + SUPPORT_HI)
    xscale = 0.5 * (SUPPORT_HI - support_lo)
    result = {
        "ggt7_recon": Family(
            name="ggt7_recon",
            n_shape=6,
            bounds=(
                (0.2, 12.0),
                (math.log(0.0005), math.log(0.30)),
                (math.log(0.25), math.log(3.0)),
                (max(0.0, support_lo - 0.025), support_lo - 0.00005),
                (max(0.015, support_lo - 0.010), min(0.100, support_lo + 0.070)),
                (math.log(0.0007), math.log(0.030)),
            ),
            seed=np.asarray(
                [
                    3.5,
                    math.log(0.004),
                    math.log(0.65),
                    support_lo - 0.002,
                    0.052,
                    math.log(0.006),
                ]
            ),
            logshape=logshape_ggt7_recon,
        ),
        "spq6_identifiable": Family(
            name="spq6_identifiable",
            n_shape=5,
            bounds=(
                (0.0, 30.0),
                (-30.0, 30.0),
                (-30.0, 10.0),
                (max(0.015, support_lo - 0.010), min(0.100, support_lo + 0.070)),
                (math.log(0.0007), math.log(0.030)),
            ),
            seed=np.asarray([5.0, -5.0, -2.0, 0.050, math.log(0.004)]),
            logshape=logshape_spq6,
        ),
        "cheb5_fixed_turn": Family(
            name="cheb5_fixed_turn",
            n_shape=5,
            bounds=tuple([(-30.0, 30.0)] * 5),
            seed=np.zeros(5, dtype=float),
            logshape=logshape_cheb5_fixed_turn,
        ),
    }
    if x0 is not None:
        result["ggt6_fixed_x0"] = Family(
            name="ggt6_fixed_x0",
            n_shape=5,
            bounds=(
                (0.2, 12.0),
                (math.log(0.0005), math.log(0.30)),
                (math.log(0.25), math.log(3.0)),
                (max(0.015, support_lo - 0.010), min(0.100, support_lo + 0.070)),
                (math.log(0.0007), math.log(0.030)),
            ),
            seed=np.asarray(
                [3.5, math.log(0.004), math.log(0.65), 0.052, math.log(0.006)]
            ),
            logshape=logshape_ggt6,
        )
        result["ggtq6_fixed_x0w"] = Family(
            name="ggtq6_fixed_x0w",
            n_shape=5,
            bounds=(
                (0.2, 12.0),
                (math.log(0.0005), math.log(0.30)),
                (math.log(0.25), math.log(3.0)),
                (max(0.015, support_lo - 0.010), min(0.100, support_lo + 0.070)),
                (-10.0, 10.0),
            ),
            seed=np.asarray(
                [3.5, math.log(0.004), math.log(0.65), 0.052, 0.0]
            ),
            logshape=logshape_ggtq6_fixed_x0w,
        )
        result["pow2exp6_fixed_w"] = Family(
            name="pow2exp6_fixed_w",
            n_shape=5,
            bounds=(
                (0.0, 20.0),
                (math.log(0.002), math.log(0.080)),
                (math.log(0.020), math.log(0.800)),
                (-15.0, 15.0),
                (max(0.015, support_lo - 0.010), min(0.100, support_lo + 0.070)),
            ),
            seed=np.asarray(
                [4.0, math.log(0.015), math.log(0.080), -2.0, 0.052]
            ),
            logshape=logshape_pow2exp6_fixed_w,
        )
        result["ggt34_6_fixed_turn"] = Family(
            name="ggt34_6_fixed_turn",
            n_shape=5,
            bounds=(
                (0.2, 12.0),
                (math.log(0.0005), math.log(0.30)),
                (math.log(0.25), math.log(3.0)),
                (-2.0, 2.0),
                (-2.0, 2.0),
            ),
            seed=np.asarray([3.5, math.log(0.004), math.log(0.65), 0.0, 0.0]),
            logshape=logshape_ggt34_6_fixed_turn,
        )
        result["ggt26_6_fixed_turn"] = Family(
            name="ggt26_6_fixed_turn",
            n_shape=5,
            bounds=(
                (0.2, 12.0),
                (math.log(0.0005), math.log(0.30)),
                (math.log(0.25), math.log(3.0)),
                (-2.0, 2.0),
                (-2.0, 2.0),
            ),
            seed=np.asarray([3.5, math.log(0.004), math.log(0.65), 0.0, 0.0]),
            logshape=logshape_ggt26_6_fixed_turn,
        )
    return result


def restart_summary(fits: list[Fit], tolerance: float = 1e-9) -> dict[str, float | int]:
    best = fits[0].objective
    objectives = np.asarray([fit.objective for fit in fits], dtype=float)
    return {
        "n_restarts": int(len(fits)),
        "n_within_1e-9_objective": int(np.sum(objectives - best <= tolerance)),
        "objective_best": float(best),
        "objective_second": float(objectives[1]) if len(objectives) > 1 else math.nan,
        "objective_worst": float(objectives[-1]),
        "objective_spread": float(objectives[-1] - best),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=HERE / "derived")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {name: load_source(*spec) for name, spec in SOURCES.items()}
    payload: dict[str, object] = {
        "status": "exploratory_scratch_only",
        "families": {
            "ggt6_fixed_x0": "A*sigmoid((x-xt)/w)*(x-x0)^a*exp(-((x-x0)/lambda)^power); x0 fixed from 1% seven-parameter reconnaissance",
            "spq6_identifiable": "A*sigmoid((x-xt)/w)*x^a*exp(c1*u+c2*u^2), u=(x-xmid)/xscale; xmid/xscale fixed",
            "cheb5_fixed_turn": "A*fixed-sigmoid((x-xt)/w)*exp(sum_{k=1}^5 c_k T_k(u)); xt,w fixed from 1% generalized-gamma reconnaissance",
            "ggtq6_fixed_x0w": "A*sigmoid((x-xt)/w_fixed)*(x-x0_fixed)^a*exp(-((x-x0_fixed)/lambda)^power+c2*u^2); x0,w fixed from 1% reconnaissance",
            "pow2exp6_fixed_w": "A*sigmoid((x-xt)/w_fixed)*x^a*(exp(-x/theta_short)+r*exp(-x/theta_long)); w fixed from 1% reconnaissance",
            "ggt34_6_fixed_turn": "A*fixed-sigmoid*(x-x0_fixed)^a*exp(-((x-x0_fixed)/lambda)^power+d3*T3(u)+d4*T4(u)); x0,xt,w fixed from 1% reconnaissance",
            "ggt26_6_fixed_turn": "A*fixed-sigmoid*(x-x0_fixed)^a*exp(-((x-x0_fixed)/lambda)^power+d2*T2(u)+d6*T6(u)); x0,xt,w fixed from 1% reconnaissance",
        },
        "normalization_profiled_but_counted_free": True,
        "supports": {},
    }
    csv_rows: list[dict[str, object]] = []

    for support_lo in SUPPORT_LOWS:
        support_key = f"{int(round(1000*support_lo))}MeV"
        base_values, base_centers, _ = loaded["one_pct"]
        mask = (base_centers >= support_lo) & (base_centers < SUPPORT_HI)
        x = base_centers[mask]
        one_raw = base_values[mask]

        # Seven-free-parameter 1% reconnaissance only; its fitted onset is then
        # frozen for every six-parameter fit on this support.
        recon_family = families(support_lo)["ggt7_recon"]
        recon_fixed: dict[str, float] = {}
        recon, recon_all = fit_family(
            x,
            one_raw,
            recon_family,
            recon_fixed,
            n_restarts=args.restarts,
            rng_seed=BASE_SEED + int(1000 * support_lo),
        )
        x0 = float(recon.parameters["x0"])
        support_record: dict[str, object] = {
            "x0_selection": {
                "source": "one_pct_raw",
                "reconnaissance_family": "ggt7_recon",
                "x0_GeV": x0,
                "parameters": recon.parameters,
                "restart_summary": restart_summary(recon_all),
            },
            "fits": {},
        }

        one_seed_by_target_family: dict[tuple[float, str], np.ndarray] = {}
        for source_index, source in enumerate(("one_pct", "ten_pct")):
            full_values, full_centers, _ = loaded[source]
            source_mask = (full_centers >= support_lo) & (full_centers < SUPPORT_HI)
            if not np.array_equal(full_centers[source_mask], x):
                raise RuntimeError("source binning mismatch")
            raw = full_values[source_mask]
            source_record: dict[str, object] = {}
            for multiplier in SMOOTH_MULTIPLIERS:
                target_name = "raw" if multiplier == 0 else f"smooth_{multiplier:g}sigma"
                target = adaptive_smooth(full_values, full_centers, x, multiplier)
                target *= float(np.sum(raw) / np.sum(target))
                target_record: dict[str, object] = {}
                fams = families(support_lo, x0=x0)
                fixed_by_family = {
                    "ggt6_fixed_x0": {"x0": x0},
                    "spq6_identifiable": {
                        "xmid": 0.5 * (support_lo + SUPPORT_HI),
                        "xscale": 0.5 * (SUPPORT_HI - support_lo),
                    },
                    "cheb5_fixed_turn": {
                        "xt_fixed": float(recon.parameters["xt"]),
                        "w_fixed": float(recon.parameters["w"]),
                        "support_lo": support_lo,
                        "support_hi": SUPPORT_HI,
                    },
                    "ggtq6_fixed_x0w": {
                        "x0": x0,
                        "w_fixed": float(recon.parameters["w"]),
                        "xmid": 0.5 * (support_lo + SUPPORT_HI),
                        "xscale": 0.5 * (SUPPORT_HI - support_lo),
                    },
                    "pow2exp6_fixed_w": {
                        "w_fixed": float(recon.parameters["w"]),
                    },
                    "ggt34_6_fixed_turn": {
                        "x0": x0,
                        "xt_fixed": float(recon.parameters["xt"]),
                        "w_fixed": float(recon.parameters["w"]),
                        "support_lo": support_lo,
                        "support_hi": SUPPORT_HI,
                    },
                    "ggt26_6_fixed_turn": {
                        "x0": x0,
                        "xt_fixed": float(recon.parameters["xt"]),
                        "w_fixed": float(recon.parameters["w"]),
                        "support_lo": support_lo,
                        "support_hi": SUPPORT_HI,
                    },
                }
                for family_name in (
                    "ggt6_fixed_x0",
                    "spq6_identifiable",
                    "cheb5_fixed_turn",
                    "ggtq6_fixed_x0w",
                    "pow2exp6_fixed_w",
                    "ggt34_6_fixed_turn",
                    "ggt26_6_fixed_turn",
                ):
                    family = fams[family_name]
                    seed_override = one_seed_by_target_family.get((multiplier, family_name))
                    fit, all_fits = fit_family(
                        x,
                        target,
                        family,
                        fixed_by_family[family_name],
                        seed_override=seed_override,
                        n_restarts=args.restarts,
                        rng_seed=(
                            BASE_SEED
                            + 100000 * source_index
                            + int(1000 * support_lo)
                            + int(100 * multiplier)
                            + {
                                "ggt6_fixed_x0": 0,
                                "spq6_identifiable": 50000,
                                "cheb5_fixed_turn": 70000,
                                "ggtq6_fixed_x0w": 90000,
                                "pow2exp6_fixed_w": 110000,
                                "ggt34_6_fixed_turn": 130000,
                                "ggt26_6_fixed_turn": 150000,
                            }[family_name]
                        ),
                    )
                    if source == "one_pct":
                        one_seed_by_target_family[(multiplier, family_name)] = fit.shape.copy()

                    diagnostics: dict[str, object] = {}
                    for observed_name, observed in (("raw", raw), ("target", target)):
                        for factor in REBIN_FACTORS:
                            label = f"{observed_name}_rebin{factor}"
                            diagnostics[label] = metrics(
                                rebin_sum(observed, factor),
                                rebin_sum(fit.expected, factor),
                                6,
                            )
                            metric = diagnostics[label]
                            csv_rows.append(
                                {
                                    "support_low_MeV": int(round(1000 * support_lo)),
                                    "source": source,
                                    "fit_target": target_name,
                                    "family": family_name,
                                    "comparison": observed_name,
                                    "rebin_factor": factor,
                                    "bin_width_MeV": float(
                                        1000 * factor * np.median(np.diff(x))
                                    ),
                                    "pearson_chi2ndf": metric["pearson_chi2ndf"],
                                    "poisson_deviance_ndf": metric[
                                        "poisson_deviance_ndf"
                                    ],
                                    "max_abs_pearson_residual": metric[
                                        "max_abs_pearson_residual"
                                    ],
                                    "ndf": metric["ndf"],
                                    "fit_success": fit.success,
                                    "best_restart": fit.restart,
                                    "seeded_from_one_pct": bool(
                                        source == "ten_pct" and seed_override is not None
                                    ),
                                    "x0_fixed_GeV": x0 if family_name.startswith("ggt") else "",
                                }
                            )
                    target_record[family_name] = {
                        "n_free_parameters": 6,
                        "parameters": fit.parameters,
                        "success": fit.success,
                        "status": fit.status,
                        "message": fit.message,
                        "iterations": fit.iterations,
                        "best_restart": fit.restart,
                        "seeded_from_one_pct": bool(
                            source == "ten_pct" and seed_override is not None
                        ),
                        "restart_summary": restart_summary(all_fits),
                        "diagnostics": diagnostics,
                    }
                source_record[target_name] = target_record
            support_record["fits"][source] = source_record
        payload["supports"][support_key] = support_record

    json_path = args.output_dir / "rigid_candidate_fits.json"
    csv_path = args.output_dir / "rigid_candidate_metrics.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(csv_rows)}, indent=2))


if __name__ == "__main__":
    main()
