#!/usr/bin/env python3
"""Numerically centered fixed-strength profiles for the v4.9.12 band run."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from hps_gpr.statistics import (
    _chol_with_jitter,
    profiled_gaussian_likelihood_summary,
    qmu_tilde_profiled_gaussian,
)
from piecewise_cached_solver import (
    SOLVER_VERSION as PARENT_SOLVER_VERSION,
    CachedPiecewiseBoundedLimit as ParentCachedPiecewiseBoundedLimit,
    LimitResult,
    SolverCounters,
    _fit_status,
    _reconcile_feasible_profile_candidates,
    _require_likelihood_nesting,
    _summary_ok,
)


SOLVER_VERSION = (
    f"{PARENT_SOLVER_VERSION}_centered_fixed_profile_v2_centered_free_retry_v1"
)
OPTIMIZER_VERSION = "poisson_deviance_fixed_profile_lbfgsb_v2"
FREE_OPTIMIZER_VERSION = "poisson_deviance_free_profile_scaled_lbfgsb_v1"


@dataclass
class BandSolverCounters(SolverCounters):
    bounded_fixed_feasible_fallbacks: int = 0
    unbounded_fixed_feasible_fallbacks: int = 0
    bounded_free_centered_retries: int = 0
    unbounded_free_centered_retries: int = 0
    null_centered_retries: int = 0


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


def centered_free_poi_nll(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    *,
    allow_negative: bool,
    initial: Dict[str, object],
) -> Dict[str, object]:
    """Retry a failed free-strength profile with a centered, scaled objective."""

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
        raise ValueError("non-finite free-profile input")

    factor = _chol_with_jitter(covariance)
    floor = 1.0e-9 * max(1.0, float(np.median(b)))
    positive = n > 0.0
    raw_a = float(initial.get("A_hat", float("nan")))
    raw_sigma = float(initial.get("sigma_A", float("nan")))
    amplitude_scale = (
        abs(raw_sigma)
        if np.isfinite(raw_sigma) and abs(raw_sigma) > 1.0e-12
        else max(1.0, math.sqrt(float(np.sum(np.clip(b, 1.0, None)))))
    )
    start_a = raw_a if np.isfinite(raw_a) else 0.0
    if not allow_negative:
        start_a = max(0.0, start_a)
    start_theta = np.asarray(
        initial.get("theta_hat", np.zeros(b.size)), dtype=float
    ).reshape(-1)
    if start_theta.shape != b.shape or not np.isfinite(start_theta).all():
        start_theta = np.zeros(b.size, dtype=float)
    start = np.concatenate(([start_a / amplitude_scale], start_theta))
    bounds = None
    if not allow_negative:
        bounds = [(0.0, None)] + [(None, None)] * b.size

    def objective_and_gradient(vector: np.ndarray) -> Tuple[float, np.ndarray]:
        scaled_a = float(vector[0])
        strength = scaled_a * amplitude_scale
        theta = np.asarray(vector[1:], dtype=float)
        lam = b + factor @ theta + strength * signal
        lam_eff = np.maximum(lam, floor)
        terms = lam_eff.copy()
        terms[positive] += n[positive] * (
            np.log(n[positive]) - np.log(lam_eff[positive])
        ) - n[positive]
        objective = float(np.sum(terms) + 0.5 * np.dot(theta, theta))
        ratio_residual = n / lam_eff - 1.0
        gradient_a = -amplitude_scale * float(np.dot(signal, ratio_residual))
        gradient_theta = -(factor.T @ ratio_residual) + theta
        below_floor = lam < floor
        if np.any(below_floor):
            delta = floor - lam[below_floor]
            penalty_scale = 1.0e6
            objective += float(penalty_scale * np.dot(delta, delta))
            penalty_gradient = -2.0 * penalty_scale * delta
            gradient_a += amplitude_scale * float(
                np.dot(signal[below_floor], penalty_gradient)
            )
            gradient_theta += factor[below_floor].T @ penalty_gradient
        gradient = np.concatenate(([gradient_a], gradient_theta))
        return objective, np.asarray(gradient, dtype=float)

    result = minimize(
        fun=lambda vector: objective_and_gradient(vector)[0],
        x0=start,
        jac=lambda vector: objective_and_gradient(vector)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2000,
            "maxls": 100,
            "ftol": 1.0e-14,
            "gtol": 1.0e-8,
        },
    )
    strength_hat = float(result.x[0]) * amplitude_scale
    if not allow_negative:
        strength_hat = max(0.0, strength_hat)
    theta_hat = np.asarray(result.x[1:], dtype=float)
    lam_hat = b + factor @ theta_hat + strength_hat * signal
    lam_eff = np.maximum(lam_hat, floor)
    centered_nll, gradient = objective_and_gradient(result.x)
    raw_nll = float(
        np.sum(lam_eff - n * np.log(lam_eff))
        + 0.5 * np.dot(theta_hat, theta_hat)
    )

    weights = n / (lam_eff**2)
    information_aa = float(np.sum(weights * signal**2))
    information_a_theta = (signal * weights) @ factor
    information_theta_theta = (factor.T * weights) @ factor + np.eye(b.size)
    sigma_a = float("nan")
    try:
        solved = np.linalg.solve(
            information_theta_theta,
            information_a_theta.reshape(-1, 1),
        ).reshape(-1)
        profile_information = information_aa - float(information_a_theta @ solved)
        sigma_a = float(np.sqrt(1.0 / max(profile_information, 1.0e-18)))
    except Exception:
        sigma_a = amplitude_scale

    finite = bool(
        np.isfinite(centered_nll)
        and np.isfinite(raw_nll)
        and np.isfinite(strength_hat)
        and np.isfinite(sigma_a)
        and np.isfinite(theta_hat).all()
        and np.isfinite(lam_hat).all()
        and np.isfinite(gradient).all()
    )
    floor_clear = bool(np.all(lam_hat > floor))
    success = bool(getattr(result, "success", False) and finite and floor_clear)
    return {
        "A_hat": strength_hat,
        "A_hat_bounded": max(0.0, strength_hat),
        "sigma_A": sigma_a,
        "theta_hat": theta_hat,
        "delta_b_hat": factor @ theta_hat,
        "b_fit": b + factor @ theta_hat,
        "lambda_hat": lam_hat,
        "nll": raw_nll,
        "success": success,
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "centered_nll": centered_nll,
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "minimum_lambda": float(np.min(lam_hat)),
        "likelihood_floor": float(floor),
        "amplitude_scale": float(amplitude_scale),
        "optimizer_version": FREE_OPTIMIZER_VERSION,
    }


class CachedPiecewiseBoundedLimit(ParentCachedPiecewiseBoundedLimit):
    """Parent solver with only its fixed-strength nuisance profile centered."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.counters = BandSolverCounters(**self.counters.__dict__)

    def _repair_free_profile_candidates(
        self,
        counts: np.ndarray,
        summary: Dict[str, object],
    ) -> Dict[str, object]:
        repaired = dict(summary)
        for name, allow_negative, counter_name in (
            (
                "fit_unbounded",
                True,
                "unbounded_free_centered_retries",
            ),
            (
                "fit_bounded",
                False,
                "bounded_free_centered_retries",
            ),
        ):
            raw = dict(repaired[name])
            if bool(raw.get("success", False)) and np.isfinite(
                float(raw.get("nll", float("nan")))
            ):
                continue
            retry = centered_free_poi_nll(
                counts,
                self.b,
                self.cov,
                self.signal_template,
                allow_negative=allow_negative,
                initial=raw,
            )
            if not bool(retry["success"]):
                raise RuntimeError(
                    f"centered free-profile retry failed for {name}: "
                    f"status={retry['status']}, message={retry['message']}"
                )
            raw_nll = float(raw.get("nll", float("nan")))
            retry.update(
                branch="centered_free_profile_retry",
                raw_optimizer_success=bool(raw.get("success", False)),
                raw_optimizer_nll=raw_nll,
                fallback_source="centered_free_profile",
                fallback_nll_improvement=(
                    raw_nll - float(retry["nll"])
                    if np.isfinite(raw_nll)
                    else float("nan")
                ),
            )
            repaired[name] = retry
            setattr(
                self.counters,
                counter_name,
                int(getattr(self.counters, counter_name)) + 1,
            )

        null = dict(repaired["null"])
        if not bool(null.get("success", False)) or not np.isfinite(
            float(null.get("nll", float("nan")))
        ):
            retry_null = centered_fixed_poi_nll(
                counts,
                self.b,
                self.cov,
                self.signal_template,
                A_fixed=0.0,
            )
            if not bool(retry_null["success"]):
                raise RuntimeError(
                    "centered null-profile retry failed: "
                    f"status={retry_null['status']}, "
                    f"message={retry_null['message']}"
                )
            raw_nll = float(null.get("nll", float("nan")))
            retry_null.update(
                A_hat=0.0,
                A_hat_bounded=0.0,
                sigma_A=float("nan"),
                branch="centered_null_profile_retry",
                raw_optimizer_success=bool(null.get("success", False)),
                raw_optimizer_nll=raw_nll,
                fallback_source="centered_null_profile",
                fallback_nll_improvement=(
                    raw_nll - float(retry_null["nll"])
                    if np.isfinite(raw_nll)
                    else float("nan")
                ),
            )
            repaired["null"] = retry_null
            self.counters.null_centered_retries += 1
        return repaired

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
        counts_array = np.asarray(counts, dtype=float)
        if counts_array.shape != self.b.shape:
            raise ValueError(
                f"count-vector shape mismatch: {counts_array.shape} vs {self.b.shape}"
            )
        if not np.isfinite(counts_array).all() or np.any(counts_array < 0.0):
            raise ValueError("observed counts must be finite and nonnegative")
        self.counters.limit_calls += 1
        observed_base = profiled_gaussian_likelihood_summary(
            counts_array,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=None,
        )
        observed_base = self._repair_free_profile_candidates(
            counts_array, observed_base
        )
        (
            observed_base,
            bounded_fallbacks,
            unbounded_fallbacks,
        ) = _reconcile_feasible_profile_candidates(observed_base)
        self.counters.bounded_null_feasible_fallbacks += bounded_fallbacks
        self.counters.unbounded_feasible_fallbacks += unbounded_fallbacks
        if not _summary_ok(observed_base):
            raise RuntimeError("observed base profile did not converge")

        eps_lo = 0.0
        cls_lo = 1.0
        if self.combined_mode == "count_scale":
            eps_hi = max(
                1.0e-10,
                max(1.0, 3.0 * math.sqrt(max(self.b_sum, 1.0)))
                / self.signal_scale,
            )
        else:
            eps_hi = max(
                1.0e-10,
                3.0 * math.sqrt(max(self.b_sum, 1.0))
                / max(self.s_sum, 1.0e-12),
            )

        sampled: Dict[float, float] = {eps_lo: cls_lo}
        max_sampled_cls_increase = 0.0
        monotonic_atol = 5.0e-4

        def evaluate(value: float) -> Tuple[float, Dict[str, object]]:
            nonlocal max_sampled_cls_increase
            cls_value, details = self._cls_at_eps2(
                value, counts_array, observed_base
            )
            sampled[float(value)] = float(cls_value)
            ordered = sorted(sampled.items())
            for (_, left), (_, right) in zip(ordered[:-1], ordered[1:]):
                max_sampled_cls_increase = max(
                    max_sampled_cls_increase, float(right - left)
                )
                if right > left + monotonic_atol:
                    raise RuntimeError(
                        "sampled CLs curve is nonmonotonic beyond tolerance: "
                        f"{right:.12g} > {left:.12g} + {monotonic_atol}"
                    )
            return cls_value, details

        expansions = 0
        cls_hi, _ = evaluate(eps_hi)
        while cls_hi > self.alpha and expansions < 80 and eps_hi < 1.0e12:
            eps_lo, cls_lo = eps_hi, cls_hi
            eps_hi *= 2.0
            expansions += 1
            cls_hi, _ = evaluate(eps_hi)
        if not np.isfinite(cls_hi) or cls_hi > self.alpha:
            raise RuntimeError(
                f"failed to bracket CLs={self.alpha}: high={eps_hi}, CLs={cls_hi}"
            )
        if not (cls_lo > self.alpha and cls_hi <= self.alpha):
            raise RuntimeError(
                "invalid initial CLs bracket: "
                f"low=({eps_lo:.12g},{cls_lo:.12g}), "
                f"high=({eps_hi:.12g},{cls_hi:.12g})"
            )

        iterations = 0
        convergence_reason = ""
        root_eps2 = float("nan")
        for iterations in range(1, 81):
            mid = 0.5 * (eps_lo + eps_hi)
            cls_mid, _ = evaluate(mid)
            if cls_mid > self.alpha:
                eps_lo, cls_lo = mid, cls_mid
            else:
                eps_hi, cls_hi = mid, cls_mid
            if not (cls_lo > self.alpha and cls_hi <= self.alpha):
                raise RuntimeError("CLs bisection lost its endpoint-sign invariant")
            if abs(cls_mid - self.alpha) < 1.0e-8:
                root_eps2 = mid
                convergence_reason = "cls_residual"
                break
            if abs(eps_hi - eps_lo) <= max(1.0e-16, 1.0e-6 * eps_hi):
                convergence_reason = "bracket_width"
                break
        else:
            raise RuntimeError(
                "CLs bisection reached 80 iterations without convergence"
            )

        if not (cls_lo > self.alpha and cls_hi <= self.alpha):
            raise RuntimeError("final CLs bracket is not oriented")
        solution = (
            root_eps2 if np.isfinite(root_eps2) else 0.5 * (eps_lo + eps_hi)
        )
        cls_solution, detail = evaluate(solution)
        residual = abs(cls_solution - self.alpha)
        log_residual = abs(math.log(cls_solution) - math.log(self.alpha))
        if residual > 2.0e-6 or log_residual > 2.0e-5:
            raise RuntimeError(
                "CLs root did not meet absolute and log residual gates: "
                f"absolute={residual:.12g}, log={log_residual:.12g}"
            )
        if not detail["optimizer_ok"]:
            raise RuntimeError("solved-limit profile metadata is not successful")
        return LimitResult(
            eps2_90=float(solution),
            alpha=self.alpha,
            confidence_level=float(1.0 - self.alpha),
            cls_at_limit=float(cls_solution),
            cl_sb_at_limit=float(detail["cl_sb"]),
            cl_b_at_limit=float(detail["cl_b"]),
            log_cls_at_limit=float(detail["log_cls"]),
            log_cl_sb_at_limit=float(detail["log_cl_sb"]),
            log_cl_b_at_limit=float(detail["log_cl_b"]),
            qmu_obs_at_limit=float(detail["qmu_obs"]),
            qmu_asimov_b_at_limit=float(detail["qmu_asimov_b"]),
            tail_branch_at_limit=str(detail["tail_branch"]),
            z_sb_at_limit=float(detail["z_sb"]),
            z_b_at_limit=float(detail["z_b"]),
            observed_qmu_branch_at_limit=str(detail["observed_qmu_branch"]),
            observed_unconstrained_strength=float(
                detail["observed_unconstrained_strength"]
            ),
            observed_unconstrained_strength_unit=(
                "total signal counts in fitted windows"
                if self.combined_mode == "count_scale"
                else "epsilon squared"
            ),
            signal_scale_counts_per_eps2=float(self.signal_scale),
            bracket_low_eps2=float(eps_lo),
            bracket_high_eps2=float(eps_hi),
            bracket_low_cls=float(cls_lo),
            bracket_high_cls=float(cls_hi),
            bracket_expansions=int(expansions),
            bisection_iterations=int(iterations),
            convergence_reason=convergence_reason,
            combined_mode=self.combined_mode,
            solver_version=SOLVER_VERSION,
            optimizer_ok=True,
            profile_status={
                "observed": detail["observed_profile_status"],
                "asimov": detail["asimov_profile_status"],
                "numerical_monotonicity": {
                    "maximum_sampled_cls_increase": float(
                        max(0.0, max_sampled_cls_increase)
                    ),
                    "accepted_absolute_tolerance": float(monotonic_atol),
                },
            },
            counters={
                key: int(value) for key, value in asdict(self.counters).items()
            },
        )
