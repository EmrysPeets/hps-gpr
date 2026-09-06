r"""Release-local bounded-``tilde(q)_mu`` asymptotic CLs mapping.

This module intentionally does not modify :mod:`hps_gpr`.  It supplies the
piecewise asymptotic tail mapping required when the unconstrained estimator is
negative, i.e. when ``qmu_obs > qmu_asimov_b``.

The likelihood and test-statistic construction remain those in
``hps_gpr.statistics``.  Only the conversion from the observed and Asimov test
statistics to ``CL_sb``, ``CL_b``, and ``CL_s`` is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from hps_gpr.statistics import (
    profiled_gaussian_likelihood_summary,
    qmu_tilde_profiled_gaussian,
)


@dataclass(frozen=True)
class BoundedTildeQAsymptoticTails:
    """Asymptotic CLs tails and branch metadata for one tested strength."""

    cls: float
    cl_sb: float
    cl_b: float
    log_cls: float
    log_cl_sb: float
    log_cl_b: float
    qmu_obs: float
    qmu_asimov_b: float
    branch: str
    z_sb: float
    z_b: float


def _finite_nonnegative(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative; got {value!r}")
    return out


def bounded_tildeq_asymptotic_tails(
    qmu_obs: float,
    qmu_asimov_b: float,
    *,
    boundary_rtol: float = 1.0e-12,
    boundary_atol: float = 1.0e-14,
) -> BoundedTildeQAsymptoticTails:
    r"""Map bounded ``tilde(q)_mu`` values to asymptotic CLs tails.

    Let ``q_A`` be the background-only Asimov value.  For
    ``q_obs <= q_A`` the usual square-root mapping applies.  For
    ``q_obs > q_A`` (the negative-unconstrained-estimator branch), the bounded
    statistic has a different asymptotic distribution and requires

    ``z_sb = (q_obs + q_A) / (2 sqrt(q_A))`` and
    ``z_b  = (q_A - q_obs) / (2 sqrt(q_A))``.

    Here ``CL_sb`` and ``CL_b`` are the upper-tail probabilities of the same
    exclusion statistic under signal-plus-background and background,
    respectively, and ``CL_s = CL_sb / CL_b``.  Log probabilities are used for
    the ratio so that large downward fluctuations do not suffer a ``0/0``
    underflow or an arbitrary denominator floor.
    """

    q_obs = _finite_nonnegative(qmu_obs, "qmu_obs")
    q_a = _finite_nonnegative(qmu_asimov_b, "qmu_asimov_b")
    if q_a <= 0.0:
        raise ValueError(
            "qmu_asimov_b must be positive for an asymptotic CLs mapping; "
            "q_A=0 means the tested strength has no resolved Asimov separation"
        )

    sqrt_q = float(np.sqrt(q_obs))
    sqrt_q_a = float(np.sqrt(q_a))
    on_or_below = bool(
        q_obs < q_a
        or np.isclose(q_obs, q_a, rtol=boundary_rtol, atol=boundary_atol)
    )
    if on_or_below:
        branch = "qobs_le_qA"
        z_sb = sqrt_q
        z_b = sqrt_q_a - sqrt_q
    else:
        branch = "qobs_gt_qA_negative_muhat"
        denom = 2.0 * sqrt_q_a
        z_sb = (q_obs + q_a) / denom
        z_b = (q_a - q_obs) / denom

    log_cl_sb = float(norm.logsf(z_sb))
    log_cl_b = float(norm.logcdf(z_b))
    log_cls = float(log_cl_sb - log_cl_b)
    # Individual tails may legitimately underflow in float64.  The ratio is
    # evaluated in log space first and therefore remains usable much farther
    # into the tail.
    cl_sb = float(np.exp(log_cl_sb))
    cl_b = float(np.exp(log_cl_b))
    cls = float(np.exp(np.clip(log_cls, -745.0, 709.0)))

    return BoundedTildeQAsymptoticTails(
        cls=cls,
        cl_sb=cl_sb,
        cl_b=cl_b,
        log_cls=log_cls,
        log_cl_sb=log_cl_sb,
        log_cl_b=log_cl_b,
        qmu_obs=q_obs,
        qmu_asimov_b=q_a,
        branch=branch,
        z_sb=float(z_sb),
        z_b=float(z_b),
    )


def asymptotic_cls_profiled_gaussian_piecewise(
    A_test: float,
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    *,
    observed_base: Optional[Dict[str, object]] = None,
    asimov_base: Optional[Dict[str, object]] = None,
) -> Tuple[float, float, float, Dict[str, object]]:
    """Drop-in-style profiled CLs evaluation with the bounded piecewise tails.

    ``observed_base`` and ``asimov_base`` may be supplied by a cached combined
    driver.  They must be summaries for the same arrays and template.  The
    function recomputes the fixed-strength profiles through
    ``qmu_tilde_profiled_gaussian`` and records both optimizer-success flags.
    """

    strength = float(A_test)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError(f"A_test must be finite and nonnegative; got {A_test!r}")
    if strength == 0.0:
        info: Dict[str, object] = {
            "A_test": 0.0,
            "qmu_obs": 0.0,
            "qmu_asimov_b": 0.0,
            "tail_branch": "zero_strength",
            "cls_statistic": "tilde_q_mu",
            "calibration": "asymptotic_piecewise_bounded",
            "ok": True,
        }
        return 1.0, 1.0, 1.0, info

    obs = np.asarray(n_obs, dtype=float)
    b = np.asarray(b_mean, dtype=float)
    cov = np.asarray(b_cov, dtype=float)
    signal = np.asarray(template, dtype=float)
    if obs.shape != b.shape or signal.shape != b.shape:
        raise ValueError(
            f"Vector shape mismatch: obs={obs.shape}, b={b.shape}, "
            f"template={signal.shape}"
        )
    if cov.shape != (b.size, b.size):
        raise ValueError(f"Covariance shape mismatch: {cov.shape} vs {(b.size, b.size)}")
    if not (
        np.all(np.isfinite(obs))
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(cov))
        and np.all(np.isfinite(signal))
    ):
        raise ValueError("Observed, background, covariance, and signal inputs must be finite")

    if observed_base is None:
        observed_base = profiled_gaussian_likelihood_summary(
            obs, b, cov, signal, A_fixed=None
        )
    if asimov_base is None:
        asimov_base = profiled_gaussian_likelihood_summary(
            b, b, cov, signal, A_fixed=None
        )

    q_obs, info_obs = qmu_tilde_profiled_gaussian(
        obs,
        b,
        cov,
        signal,
        strength,
        summary=observed_base,
    )
    q_a, info_asimov = qmu_tilde_profiled_gaussian(
        b,
        b,
        cov,
        signal,
        strength,
        summary=asimov_base,
    )
    tails = bounded_tildeq_asymptotic_tails(q_obs, q_a)
    ok = bool(info_obs.get("ok", False) and info_asimov.get("ok", False))
    info = {
        "A_test": strength,
        "qmu_obs": float(q_obs),
        "qmu_asimov_b": float(q_a),
        "tail_branch": tails.branch,
        "z_sb": tails.z_sb,
        "z_b": tails.z_b,
        "log_CL_sb": tails.log_cl_sb,
        "log_CL_b": tails.log_cl_b,
        "log_CL_s": tails.log_cls,
        "cls_statistic": "tilde_q_mu",
        "calibration": "asymptotic_piecewise_bounded",
        "observed": info_obs,
        "asimov_b": info_asimov,
        "ok": ok,
    }
    return tails.cls, tails.cl_sb, tails.cl_b, info
