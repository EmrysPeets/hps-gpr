#!/usr/bin/env python3
"""Campaign-local exact cache for repeated asymptotic combined-limit solves.

This module deliberately lives with the v4 campaign rather than in ``hps_gpr``.
It evaluates the same profiled-likelihood functions and the same bisection as
``combined_cls_limit_epsilon2_from_vectors``.  The only optimization is to
reuse profile fits whose inputs do not change during a limit solve:

* the unconstrained, bounded, and null fits for one pseudoexperiment;
* the corresponding three background-only Asimov fits for the mass point; and
* fixed-strength Asimov profiles at exactly repeated bisection nodes.

No likelihood, minimizer, convergence criterion, CLs mapping, or limit stopping
criterion is replaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import norm

from hps_gpr.statistics import (
    profiled_gaussian_fixed_poi_nll,
    profiled_gaussian_likelihood_summary,
    qmu_tilde_profiled_gaussian,
)


CACHE_ALGORITHM_VERSION = "campaign_local_deterministic_profile_cache_v1"


@dataclass
class CacheCounters:
    """Small provenance record for one mass-point cache."""

    limit_calls: int = 0
    asimov_fixed_cache_hits: int = 0
    asimov_fixed_cache_misses: int = 0


class CachedAsymptoticCombinedLimit:
    """Exact cached form of the v3/v4 asymptotic combined epsilon-squared limit."""

    def __init__(
        self,
        b_mean: np.ndarray,
        b_cov: np.ndarray,
        s_unit: np.ndarray,
        *,
        alpha: float,
        combined_mode: str,
    ) -> None:
        self.b = np.asarray(b_mean, dtype=float)
        self.cov = np.asarray(b_cov, dtype=float)
        self.s_unit = np.asarray(s_unit, dtype=float)
        self.alpha = float(alpha)
        self.combined_mode = str(combined_mode).lower().strip()

        if self.combined_mode not in {"epsilon2", "count_scale"}:
            raise ValueError(
                f"Unsupported combined_mode={combined_mode!r}; expected "
                "'epsilon2' or 'count_scale'."
            )
        if self.b.ndim != 1 or self.s_unit.shape != self.b.shape:
            raise ValueError(
                f"Vector shape mismatch: b={self.b.shape}, s_unit={self.s_unit.shape}"
            )
        if self.cov.shape != (self.b.size, self.b.size):
            raise ValueError(
                f"Covariance shape mismatch: {self.cov.shape} vs "
                f"{(self.b.size, self.b.size)}"
            )

        self.signal_scale = 1.0
        self.signal_template = self.s_unit
        if self.combined_mode == "count_scale":
            self.signal_template = np.clip(self.s_unit, 0.0, None)
            self.signal_scale = float(np.sum(self.signal_template))
            if not np.isfinite(self.signal_scale) or self.signal_scale <= 0.0:
                raise ValueError(
                    f"Non-positive combined signal scale: {self.signal_scale!r}"
                )
            self.signal_template = self.signal_template / self.signal_scale

        self.b_sum = float(np.sum(np.clip(self.b, 0.0, None)))
        self.s_sum = float(np.sum(np.clip(self.s_unit, 0.0, None)))
        self._asimov_base = profiled_gaussian_likelihood_summary(
            self.b,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=None,
        )
        self._asimov_qmu_by_strength: Dict[float, float] = {}
        self.counters = CacheCounters()

    @property
    def asimov_fixed_cache_size(self) -> int:
        return len(self._asimov_qmu_by_strength)

    def _qmu_from_base(
        self,
        counts: np.ndarray,
        base_summary: dict,
        test_strength: float,
    ) -> float:
        summary = dict(base_summary)
        summary["fixed"] = profiled_gaussian_fixed_poi_nll(
            counts,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=float(test_strength),
        )
        return float(
            qmu_tilde_profiled_gaussian(
                counts,
                self.b,
                self.cov,
                self.signal_template,
                float(test_strength),
                summary=summary,
            )[0]
        )

    def _asimov_qmu(self, test_strength: float) -> float:
        # Bisection nodes are deterministic binary floating-point values.  An
        # exact float key therefore reuses only literally identical test points.
        key = float(test_strength)
        if key in self._asimov_qmu_by_strength:
            self.counters.asimov_fixed_cache_hits += 1
            return self._asimov_qmu_by_strength[key]

        self.counters.asimov_fixed_cache_misses += 1
        value = self._qmu_from_base(self.b, self._asimov_base, key)
        self._asimov_qmu_by_strength[key] = value
        return value

    def _cls_at_eps2(
        self,
        eps2: float,
        counts: np.ndarray,
        observed_base: dict,
    ) -> float:
        eps2 = float(max(eps2, 0.0))
        test_strength = float(eps2 * self.signal_scale)
        if test_strength <= 0.0:
            return 1.0

        qmu_obs = self._qmu_from_base(
            counts,
            observed_base,
            test_strength,
        )
        qmu_asimov = self._asimov_qmu(test_strength)

        sqrt_qmu = (
            float(np.sqrt(max(qmu_obs, 0.0)))
            if np.isfinite(qmu_obs)
            else float("nan")
        )
        sqrt_q_asimov = (
            float(np.sqrt(max(qmu_asimov, 0.0)))
            if np.isfinite(qmu_asimov)
            else float("nan")
        )
        cl_sb = (
            float(norm.sf(sqrt_qmu))
            if np.isfinite(sqrt_qmu)
            else float("nan")
        )
        cl_b = (
            float(norm.cdf(sqrt_q_asimov - sqrt_qmu))
            if np.isfinite(sqrt_qmu) and np.isfinite(sqrt_q_asimov)
            else float("nan")
        )
        cl_b = max(cl_b, 1.0e-12) if np.isfinite(cl_b) else float("nan")
        return (
            float(cl_sb / cl_b)
            if np.isfinite(cl_sb) and np.isfinite(cl_b) and cl_b > 0.0
            else float("nan")
        )

    def limit(self, counts: np.ndarray) -> float:
        """Return the cached asymptotic CLs upper limit for one count vector."""

        counts = np.asarray(counts, dtype=int)
        if counts.shape != self.b.shape:
            raise ValueError(
                f"Count-vector shape mismatch: {counts.shape} vs {self.b.shape}"
            )
        self.counters.limit_calls += 1

        observed_base = profiled_gaussian_likelihood_summary(
            counts,
            self.b,
            self.cov,
            self.signal_template,
            A_fixed=None,
        )

        eps_lo = 0.0
        if self.combined_mode == "count_scale":
            eps_hi = max(
                1.0e-10,
                max(1.0, 3.0 * math.sqrt(max(self.b_sum, 1.0)))
                / max(self.signal_scale, 1.0e-12),
            )
        else:
            eps_hi = max(
                1.0e-10,
                3.0 * math.sqrt(max(self.b_sum, 1.0))
                / max(self.s_sum, 1.0e-12),
            )

        iterations = 0
        while (
            self._cls_at_eps2(eps_hi, counts, observed_base) > self.alpha
            and eps_hi < 1.0e12
            and iterations < 80
        ):
            eps_hi *= 2.0
            iterations += 1

        for _ in range(80):
            mid = 0.5 * (eps_lo + eps_hi)
            cls_value = self._cls_at_eps2(mid, counts, observed_base)
            if abs(cls_value - self.alpha) < 1.0e-8:
                eps_lo = eps_hi = mid
                break
            if cls_value > self.alpha:
                eps_lo = mid
            else:
                eps_hi = mid
            if abs(eps_hi - eps_lo) <= max(1.0e-16, 1.0e-6 * eps_hi):
                break

        return float(0.5 * (eps_lo + eps_hi))
