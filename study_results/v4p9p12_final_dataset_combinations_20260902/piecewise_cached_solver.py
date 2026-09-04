#!/usr/bin/env python3
"""Cached, fail-closed asymptotic CLs solver for the v4.9.12 release.

The likelihood and bounded ``tilde(q)_mu`` statistic come from the attested
campaign runtime.  This wrapper adds the missing piecewise asymptotic tail
mapping for ``q_obs > q_A`` and records the solved-limit diagnostics.  It is
used for every standalone and shared-coupling curve in this release.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

try:  # Package import from the repository root.
    from .runtime.bounded_tildeq_cls import bounded_tildeq_asymptotic_tails
except ImportError:  # Script-local import used by the production driver.
    from runtime.bounded_tildeq_cls import bounded_tildeq_asymptotic_tails
from hps_gpr.statistics import (
    profiled_gaussian_fixed_poi_nll,
    profiled_gaussian_likelihood_summary,
    qmu_tilde_profiled_gaussian,
)


SOLVER_VERSION = "v4p9p12_cached_piecewise_bounded_tildeq_v3"


@dataclass
class SolverCounters:
    limit_calls: int = 0
    cls_evaluations: int = 0
    asimov_fixed_cache_hits: int = 0
    asimov_fixed_cache_misses: int = 0
    bounded_null_feasible_fallbacks: int = 0
    unbounded_feasible_fallbacks: int = 0


@dataclass(frozen=True)
class LimitResult:
    eps2_90: float
    alpha: float
    confidence_level: float
    cls_at_limit: float
    cl_sb_at_limit: float
    cl_b_at_limit: float
    log_cls_at_limit: float
    log_cl_sb_at_limit: float
    log_cl_b_at_limit: float
    qmu_obs_at_limit: float
    qmu_asimov_b_at_limit: float
    tail_branch_at_limit: str
    z_sb_at_limit: float
    z_b_at_limit: float
    observed_qmu_branch_at_limit: str
    observed_unconstrained_strength: float
    observed_unconstrained_strength_unit: str
    signal_scale_counts_per_eps2: float
    bracket_low_eps2: float
    bracket_high_eps2: float
    bracket_low_cls: float
    bracket_high_cls: float
    bracket_expansions: int
    bisection_iterations: int
    convergence_reason: str
    combined_mode: str
    solver_version: str
    optimizer_ok: bool
    profile_status: Dict[str, object]
    counters: Dict[str, int]


def _summary_ok(summary: Dict[str, object]) -> bool:
    return all(
        bool(dict(summary[name]).get("success", False))
        for name in ("fit_unbounded", "fit_bounded", "null")
    )


def _fit_status(fit: Dict[str, object]) -> Dict[str, object]:
    """Keep compact scalar optimizer evidence without serializing fit arrays."""

    fields = (
        "success",
        "nll",
        "A_hat",
        "A_hat_bounded",
        "sigma_A",
        "branch",
        "raw_optimizer_nll",
        "fallback_source",
        "fallback_nll_improvement",
    )
    result: Dict[str, object] = {}
    for key in fields:
        if key not in fit:
            continue
        value = fit[key]
        if isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            result[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[key] = float(value)
        else:
            result[key] = str(value)
    return result


def _summary_status(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        name: _fit_status(dict(summary[name]))
        for name in ("fit_unbounded", "fit_bounded", "null")
    }


def _reconcile_feasible_profile_candidates(
    summary: Dict[str, object],
) -> Tuple[Dict[str, object], int, int]:
    """Enforce exact nesting by retaining the best already-fitted feasible point.

    The A=0 null fit is a member of the bounded and unbounded parameter spaces,
    and the bounded fit is a member of the unbounded space.  L-BFGS-B can stop
    a few objective ulps above one of those known feasible candidates on the
    large-count spectra.  Selecting the lower-NLL candidate is an optimizer
    fallback, not a change to the likelihood or statistic.
    """

    out = dict(summary)
    fit_unbounded = dict(out["fit_unbounded"])
    fit_bounded = dict(out["fit_bounded"])
    null = dict(out["null"])
    for name, fit in (
        ("fit_unbounded", fit_unbounded),
        ("fit_bounded", fit_bounded),
        ("null", null),
    ):
        if not bool(fit.get("success", False)) or not np.isfinite(
            float(fit.get("nll", float("nan")))
        ):
            raise RuntimeError(
                f"raw profile candidate is not finite and successful: {name}"
            )
    raw_unbounded_nll = float(fit_unbounded["nll"])
    raw_bounded_nll = float(fit_bounded["nll"])
    fit_unbounded.update(
        raw_optimizer_nll=raw_unbounded_nll,
        fallback_source="none",
        fallback_nll_improvement=0.0,
    )
    fit_bounded.update(
        raw_optimizer_nll=raw_bounded_nll,
        fallback_source="none",
        fallback_nll_improvement=0.0,
    )
    bounded_fallbacks = 0
    unbounded_fallbacks = 0

    if (
        bool(null.get("success", False))
        and np.isfinite(float(null.get("nll", float("nan"))))
        and float(null["nll"]) < float(fit_bounded.get("nll", float("inf")))
    ):
        sigma_a = float(fit_bounded.get("sigma_A", float("nan")))
        fit_bounded = dict(null)
        fit_bounded.update(
            A_hat=0.0,
            A_hat_bounded=0.0,
            sigma_A=sigma_a,
            branch="null_feasible_fallback",
            raw_optimizer_nll=raw_bounded_nll,
            fallback_source="null",
            fallback_nll_improvement=raw_bounded_nll - float(null["nll"]),
        )
        bounded_fallbacks = 1

    if (
        bool(fit_bounded.get("success", False))
        and np.isfinite(float(fit_bounded.get("nll", float("nan"))))
        and float(fit_bounded["nll"])
        < float(fit_unbounded.get("nll", float("inf")))
    ):
        fit_unbounded = dict(fit_bounded)
        fit_unbounded.update(
            branch="bounded_feasible_fallback",
            raw_optimizer_nll=raw_unbounded_nll,
            fallback_source="bounded",
            fallback_nll_improvement=(
                raw_unbounded_nll - float(fit_bounded["nll"])
            ),
        )
        unbounded_fallbacks = 1

    out["fit_bounded"] = fit_bounded
    out["fit_unbounded"] = fit_unbounded
    return out, bounded_fallbacks, unbounded_fallbacks


def _require_likelihood_nesting(info: Dict[str, object]) -> None:
    """Reject successful minimizer flags that violate likelihood nesting."""

    names = ("nll_unbounded", "nll_bounded", "nll_null", "nll_fixed", "denom_nll")
    values = {name: float(info.get(name, float("nan"))) for name in names}
    if not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError(f"non-finite profiled NLL evidence: {values}")
    # Scale on likelihood *differences*, not the potentially huge arbitrary
    # Poisson objective offset shared by every hypothesis.
    difference_scale = max(
        1.0,
        abs(values["nll_bounded"] - values["nll_unbounded"]),
        abs(values["nll_null"] - values["nll_unbounded"]),
        abs(values["nll_fixed"] - values["denom_nll"]),
    )
    tolerance = 1.0e-6 + 1.0e-8 * difference_scale
    inequalities = {
        "unbounded_le_bounded": values["nll_unbounded"]
        <= values["nll_bounded"] + tolerance,
        "unbounded_le_null": values["nll_unbounded"]
        <= values["nll_null"] + tolerance,
        "bounded_le_null": values["nll_bounded"]
        <= values["nll_null"] + tolerance,
        "denominator_le_fixed": values["denom_nll"]
        <= values["nll_fixed"] + tolerance,
    }
    if not all(inequalities.values()):
        raise RuntimeError(
            "profile likelihood nesting failed: "
            f"checks={inequalities}, tolerance={tolerance:.6g}, nll={values}"
        )


class CachedPiecewiseBoundedLimit:
    """Observed asymptotic 90% CLs limit with exact profile-state caching."""

    def __init__(
        self,
        b_mean: np.ndarray,
        b_cov: np.ndarray,
        s_unit: np.ndarray,
        *,
        alpha: float,
        combined_mode: str = "count_scale",
    ) -> None:
        self.b = np.asarray(b_mean, dtype=float)
        self.cov = np.asarray(b_cov, dtype=float)
        self.s_unit = np.asarray(s_unit, dtype=float)
        self.alpha = float(alpha)
        self.combined_mode = str(combined_mode).lower().strip()
        if self.combined_mode not in {"epsilon2", "count_scale"}:
            raise ValueError(f"unsupported combined_mode={combined_mode!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        if self.b.ndim != 1 or self.s_unit.shape != self.b.shape:
            raise ValueError(
                f"vector shape mismatch: b={self.b.shape}, s={self.s_unit.shape}"
            )
        if self.cov.shape != (self.b.size, self.b.size):
            raise ValueError(
                f"covariance shape mismatch: {self.cov.shape} vs "
                f"{(self.b.size, self.b.size)}"
            )
        if not (
            np.isfinite(self.b).all()
            and np.isfinite(self.cov).all()
            and np.isfinite(self.s_unit).all()
        ):
            raise ValueError("non-finite background, covariance, or signal input")
        if np.any(self.b <= 0.0):
            raise ValueError("background means must be strictly positive")
        if np.any(self.s_unit < 0.0):
            raise ValueError(
                "s_unit must be nonnegative; clipping would invalidate the "
                "claimed exact count-scale coordinate change"
            )

        self.signal_scale = 1.0
        self.signal_template = self.s_unit.copy()
        if self.combined_mode == "count_scale":
            self.signal_scale = float(np.sum(self.signal_template))
            if not np.isfinite(self.signal_scale) or self.signal_scale <= 0.0:
                raise ValueError("combined signal scale is not finite and positive")
            self.signal_template /= self.signal_scale

        self.b_sum = float(np.sum(self.b))
        self.s_sum = float(np.sum(np.clip(self.s_unit, 0.0, None)))
        self.counters = SolverCounters()
        self.asimov_base = profiled_gaussian_likelihood_summary(
            self.b,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=None,
        )
        (
            self.asimov_base,
            bounded_fallbacks,
            unbounded_fallbacks,
        ) = _reconcile_feasible_profile_candidates(self.asimov_base)
        self.counters.bounded_null_feasible_fallbacks += bounded_fallbacks
        self.counters.unbounded_feasible_fallbacks += unbounded_fallbacks
        if not _summary_ok(self.asimov_base):
            raise RuntimeError("background-only Asimov base profile did not converge")
        self._asimov_cache: Dict[float, Tuple[float, Dict[str, object]]] = {}

    def _qmu_from_base(
        self,
        counts: np.ndarray,
        base: Dict[str, object],
        strength: float,
    ) -> Tuple[float, Dict[str, object]]:
        summary = dict(base)
        summary["fixed"] = profiled_gaussian_fixed_poi_nll(
            counts,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=float(strength),
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
        info["fixed_status"] = _fit_status(dict(summary["fixed"]))
        if not bool(info.get("ok", False)):
            raise RuntimeError(
                f"profile likelihood did not converge at strength={strength:.12g}"
            )
        _require_likelihood_nesting(info)
        if not np.isfinite(qmu) or qmu < 0.0:
            raise RuntimeError(f"invalid bounded qmu={qmu!r}")
        return float(qmu), info

    def _asimov_qmu(
        self, strength: float
    ) -> Tuple[float, Dict[str, object]]:
        key = float(strength)
        if key in self._asimov_cache:
            self.counters.asimov_fixed_cache_hits += 1
            return self._asimov_cache[key]
        self.counters.asimov_fixed_cache_misses += 1
        value = self._qmu_from_base(self.b, self.asimov_base, key)
        self._asimov_cache[key] = value
        return value

    def _cls_at_eps2(
        self,
        eps2: float,
        counts: np.ndarray,
        observed_base: Dict[str, object],
    ) -> Tuple[float, Dict[str, object]]:
        value = float(eps2)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"epsilon-squared test point is invalid: {eps2!r}")
        if value == 0.0:
            return 1.0, {
                "cls": 1.0,
                "cl_sb": 1.0,
                "cl_b": 1.0,
                "log_cls": 0.0,
                "log_cl_sb": 0.0,
                "log_cl_b": 0.0,
                "qmu_obs": 0.0,
                "qmu_asimov_b": 0.0,
                "tail_branch": "zero_strength",
                "z_sb": 0.0,
                "z_b": 0.0,
                "observed_qmu_branch": "zero_strength",
                "observed_unconstrained_strength": float(
                    dict(observed_base["fit_unbounded"])["A_hat"]
                ),
                "optimizer_ok": True,
                "observed_profile_status": _summary_status(observed_base),
                "asimov_profile_status": _summary_status(self.asimov_base),
            }
        strength = value * self.signal_scale
        q_obs, info_obs = self._qmu_from_base(counts, observed_base, strength)
        q_a, info_a = self._asimov_qmu(strength)
        tails = bounded_tildeq_asymptotic_tails(q_obs, q_a)
        self.counters.cls_evaluations += 1
        if not np.isfinite(tails.cls):
            raise RuntimeError("non-finite asymptotic CLs tail ratio")
        return float(tails.cls), {
            "cls": float(tails.cls),
            "cl_sb": float(tails.cl_sb),
            "cl_b": float(tails.cl_b),
            "log_cls": float(tails.log_cls),
            "log_cl_sb": float(tails.log_cl_sb),
            "log_cl_b": float(tails.log_cl_b),
            "qmu_obs": float(q_obs),
            "qmu_asimov_b": float(q_a),
            "tail_branch": str(tails.branch),
            "z_sb": float(tails.z_sb),
            "z_b": float(tails.z_b),
            "observed_qmu_branch": str(info_obs.get("branch", "undefined")),
            "observed_unconstrained_strength": float(
                info_obs.get("A_hat", float("nan"))
            ),
            "optimizer_ok": bool(info_obs.get("ok") and info_a.get("ok")),
            "observed_profile_status": {
                "base": _summary_status(observed_base),
                "fixed": dict(info_obs.get("fixed_status", {})),
                "qmu": _fit_status(info_obs),
            },
            "asimov_profile_status": {
                "base": _summary_status(self.asimov_base),
                "fixed": dict(info_a.get("fixed_status", {})),
                "qmu": _fit_status(info_a),
            },
        }

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
            # The theoretical CLs curve is nonincreasing.  A small tolerance
            # admits profile-minimizer roundoff but rejects a broken root map.
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
                # Keep the two oriented endpoints as provenance.  The
                # evaluated root is a separate quantity and must not collapse
                # the enclosing (CLs_low > alpha, CLs_high <= alpha) bracket.
                root_eps2 = mid
                convergence_reason = "cls_residual"
                break
            if abs(eps_hi - eps_lo) <= max(1.0e-16, 1.0e-6 * eps_hi):
                convergence_reason = "bracket_width"
                break
        else:
            raise RuntimeError("CLs bisection reached 80 iterations without convergence")

        if not (cls_lo > self.alpha and cls_hi <= self.alpha):
            raise RuntimeError("final CLs bracket is not oriented")
        solution = (
            root_eps2
            if np.isfinite(root_eps2)
            else 0.5 * (eps_lo + eps_hi)
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
                key: int(value)
                for key, value in asdict(self.counters).items()
            },
        )
