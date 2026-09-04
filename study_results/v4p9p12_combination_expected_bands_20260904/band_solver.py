#!/usr/bin/env python3
"""Numerically centered fixed-strength profiles for the v4.9.12 band run."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from hps_gpr.statistics import _chol_with_jitter, qmu_tilde_profiled_gaussian
from piecewise_cached_solver import (
    SOLVER_VERSION as PARENT_SOLVER_VERSION,
    CachedPiecewiseBoundedLimit as ParentCachedPiecewiseBoundedLimit,
    LimitResult,
    SolverCounters,
    _fit_status,
    _require_likelihood_nesting,
)


SOLVER_VERSION = f"{PARENT_SOLVER_VERSION}_centered_fixed_profile_v2"
OPTIMIZER_VERSION = "poisson_deviance_fixed_profile_lbfgsb_v2"


@dataclass
class BandSolverCounters(SolverCounters):
    bounded_fixed_feasible_fallbacks: int = 0
    unbounded_fixed_feasible_fallbacks: int = 0


def centered_fixed_poi_nll(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    A_fixed: float,
) -> Dict[str, object]:
    """Profile theta at fixed A using a data-constant-centered objective.

    The minimized objective is half the Poisson deviance plus the Gaussian
    nuisance penalty.  It is exactly the parent fixed-A negative log
    likelihood plus a constant that depends only on the observed counts.
    """

    n = np.clip(np.asarray(n_obs, dtype=float), 0.0, None)
    b = np.clip(np.asarray(b_mean, dtype=float), 1.0e-12, None)
    covariance = np.asarray(b_cov, dtype=float)
    signal = np.asarray(template, dtype=float)
    if covariance.shape != (b.size, b.size):
        raise ValueError(
            f"covariance shape mismatch: {covariance.shape} vs {(b.size, b.size)}"
        )
    if signal.shape != b.shape or n.shape != b.shape:
        raise ValueError(
            f"vector shape mismatch: n={n.shape}, b={b.shape}, signal={signal.shape}"
        )
    if not (
        np.isfinite(n).all()
        and np.isfinite(b).all()
        and np.isfinite(covariance).all()
        and np.isfinite(signal).all()
    ):
        raise ValueError("non-finite fixed-profile input")

    factor = _chol_with_jitter(covariance)
    floor = 1.0e-9 * max(1.0, float(np.median(b)))
    strength = float(A_fixed)
    positive = n > 0.0

    def objective_and_gradient(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        th = np.asarray(theta, dtype=float)
        lam = b + factor @ th + strength * signal
        lam_eff = np.maximum(lam, floor)
        terms = lam_eff.copy()
        terms[positive] += n[positive] * (
            np.log(n[positive]) - np.log(lam_eff[positive])
        ) - n[positive]
        objective = float(np.sum(terms) + 0.5 * np.dot(th, th))
        ratio_residual = n / lam_eff - 1.0
        gradient = -(factor.T @ ratio_residual) + th
        return objective, np.asarray(gradient, dtype=float)

    result = minimize(
        fun=lambda theta: objective_and_gradient(theta)[0],
        x0=np.zeros(b.size, dtype=float),
        jac=lambda theta: objective_and_gradient(theta)[1],
        method="L-BFGS-B",
        options={
            "maxiter": 2000,
            "maxls": 100,
            "ftol": 1.0e-14,
            "gtol": 1.0e-8,
        },
    )
    theta_hat = np.asarray(result.x, dtype=float)
    lam_hat = b + factor @ theta_hat + strength * signal
    centered_nll, gradient = objective_and_gradient(theta_hat)
    lam_eff = np.maximum(lam_hat, floor)
    raw_nll = float(
        np.sum(lam_eff - n * np.log(lam_eff))
        + 0.5 * np.dot(theta_hat, theta_hat)
    )
    finite = bool(
        np.isfinite(centered_nll)
        and np.isfinite(raw_nll)
        and np.isfinite(theta_hat).all()
        and np.isfinite(lam_hat).all()
        and np.isfinite(gradient).all()
    )
    floor_clear = bool(np.all(lam_hat > floor))
    success = bool(getattr(result, "success", False) and finite and floor_clear)
    return {
        "theta_hat": theta_hat,
        "nll": raw_nll,
        "success": success,
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "A_fixed": strength,
        "centered_nll": centered_nll,
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "minimum_lambda": float(np.min(lam_hat)),
        "likelihood_floor": float(floor),
        "optimizer_version": OPTIMIZER_VERSION,
    }


class CachedPiecewiseBoundedLimit(ParentCachedPiecewiseBoundedLimit):
    """Parent solver with only its fixed-strength nuisance profile centered."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.counters = BandSolverCounters(**self.counters.__dict__)

    def _qmu_from_base(
        self,
        counts: np.ndarray,
        base: Dict[str, object],
        strength: float,
    ) -> Tuple[float, Dict[str, object]]:
        summary = dict(base)
        fixed = centered_fixed_poi_nll(
            counts,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=float(strength),
        )
        summary["fixed"] = fixed
        fixed_nll = float(fixed["nll"])
        for name, counter_name in (
            ("fit_bounded", "bounded_fixed_feasible_fallbacks"),
            ("fit_unbounded", "unbounded_fixed_feasible_fallbacks"),
        ):
            candidate = dict(summary[name])
            candidate_nll = float(candidate.get("nll", float("inf")))
            if bool(fixed["success"]) and fixed_nll < candidate_nll:
                replacement = dict(fixed)
                replacement.update(
                    A_hat=float(strength),
                    A_hat_bounded=float(strength),
                    sigma_A=float(candidate.get("sigma_A", float("nan"))),
                    branch="fixed_feasible_fallback",
                    raw_optimizer_nll=candidate_nll,
                    fallback_source="fixed_test_strength",
                    fallback_nll_improvement=candidate_nll - fixed_nll,
                )
                summary[name] = replacement
                setattr(
                    self.counters,
                    counter_name,
                    int(getattr(self.counters, counter_name)) + 1,
                )
        qmu, info = qmu_tilde_profiled_gaussian(
            counts,
            self.b,
            self.cov,
            self.signal_template,
            float(strength),
            summary=summary,
        )
        info = dict(info)
        fixed_status = _fit_status(dict(fixed))
        fixed_status.update(
            centered_nll=float(fixed["centered_nll"]),
            gradient_inf_norm=float(fixed["gradient_inf_norm"]),
            minimum_lambda=float(fixed["minimum_lambda"]),
            likelihood_floor=float(fixed["likelihood_floor"]),
            optimizer_version=str(fixed["optimizer_version"]),
        )
        info["fixed_status"] = fixed_status
        if not bool(info.get("ok", False)):
            raise RuntimeError(
                "centered fixed-strength profile did not converge at "
                f"strength={strength:.12g}: status={fixed['status']}, "
                f"message={fixed['message']}"
            )
        _require_likelihood_nesting(info)
        if not np.isfinite(qmu) or qmu < 0.0:
            raise RuntimeError(f"invalid bounded qmu={qmu!r}")
        return float(qmu), info

    def limit(self, counts: np.ndarray) -> LimitResult:
        result = super().limit(counts)
        return replace(result, solver_version=SOLVER_VERSION)
