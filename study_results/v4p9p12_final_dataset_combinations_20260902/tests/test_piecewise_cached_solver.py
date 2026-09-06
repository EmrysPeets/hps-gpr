from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
CAMPAIGN = REPO / "study_results/v4p9p7_2016_support_combined_100toy_20260902"
sys.path.insert(0, str(CAMPAIGN))
from runtime_guard import activate_and_verify  # noqa: E402


activate_and_verify()
sys.path.insert(0, str(HERE / "runtime"))
sys.path.insert(0, str(HERE))

from bounded_tildeq_cls import (  # noqa: E402
    asymptotic_cls_profiled_gaussian_piecewise,
)
from piecewise_cached_solver import (  # noqa: E402
    CachedPiecewiseBoundedLimit,
    _reconcile_feasible_profile_candidates,
)
from run_final_combinations import condition_covariance_block  # noqa: E402


def make_solver(*, signal: float = 1.0, mode: str = "count_scale"):
    return CachedPiecewiseBoundedLimit(
        np.array([1000.0]),
        np.array([[1.0e-8]]),
        np.array([signal]),
        alpha=0.1,
        combined_mode=mode,
    )


def test_bounded_fit_uses_exact_null_feasible_candidate() -> None:
    summary = {
        "fit_unbounded": {
            "success": True,
            "nll": -79958929.73904894,
            "A_hat": -3285.7948,
        },
        "fit_bounded": {
            "success": True,
            "nll": -79958929.56975028,
            "A_hat": 0.0,
            "sigma_A": 1000.0,
        },
        "null": {
            "success": True,
            "nll": -79958929.57022439,
        },
    }
    fixed, bounded_count, unbounded_count = (
        _reconcile_feasible_profile_candidates(summary)
    )
    assert bounded_count == 1
    assert unbounded_count == 0
    assert fixed["fit_bounded"]["nll"] == summary["null"]["nll"]
    assert fixed["fit_bounded"]["fallback_source"] == "null"
    assert np.isclose(
        fixed["fit_bounded"]["fallback_nll_improvement"],
        summary["fit_bounded"]["nll"] - summary["null"]["nll"],
    )


def test_feasible_candidate_fallback_does_not_mask_raw_failure() -> None:
    summary = {
        "fit_unbounded": {"success": True, "nll": -3.0},
        "fit_bounded": {"success": False, "nll": -2.0},
        "null": {"success": True, "nll": -2.1},
    }
    try:
        _reconcile_feasible_profile_candidates(summary)
    except RuntimeError as exc:
        assert "raw profile candidate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a failed raw fit must remain fatal")


def test_deficit_root_uses_piecewise_negative_muhat_branch() -> None:
    result = make_solver().limit(np.array([900]))
    assert result.tail_branch_at_limit == "qobs_gt_qA_negative_muhat"
    assert result.observed_qmu_branch_at_limit == "boundary"
    assert np.isclose(result.cls_at_limit, 0.1, rtol=0.0, atol=2.0e-6)
    assert result.qmu_obs_at_limit > result.qmu_asimov_b_at_limit


def test_background_and_excess_roots_use_square_root_branch() -> None:
    for observed in (1000, 1100):
        result = make_solver().limit(np.array([observed]))
        assert result.tail_branch_at_limit == "qobs_le_qA"
        assert np.isclose(result.cls_at_limit, 0.1, rtol=0.0, atol=2.0e-6)
        assert result.optimizer_ok is True


def test_cached_root_matches_uncached_piecewise_evaluation() -> None:
    counts = np.array([900])
    solver = make_solver()
    result = solver.limit(counts)
    cls, clsb, clb, info = asymptotic_cls_profiled_gaussian_piecewise(
        result.eps2_90,
        counts,
        solver.b,
        solver.cov,
        solver.signal_template,
    )
    assert info["ok"] is True
    assert np.isclose(cls, result.cls_at_limit, rtol=0.0, atol=2.0e-10)
    assert np.isclose(clsb, result.cl_sb_at_limit, rtol=0.0, atol=2.0e-10)
    assert np.isclose(clb, result.cl_b_at_limit, rtol=0.0, atol=2.0e-10)
    assert result.bracket_low_eps2 < result.bracket_high_eps2
    assert result.bracket_low_cls > result.alpha
    assert result.bracket_high_cls <= result.alpha
    assert result.bracket_low_eps2 <= result.eps2_90 <= result.bracket_high_eps2


def test_count_scale_is_an_exact_coordinate_change() -> None:
    counts = np.array([1010])
    count_scale = make_solver(signal=2.0, mode="count_scale").limit(counts)
    direct = make_solver(signal=2.0, mode="epsilon2").limit(counts)
    assert np.isclose(count_scale.eps2_90, direct.eps2_90, rtol=2.0e-6)


def test_zero_signal_scale_fails_closed() -> None:
    try:
        make_solver(signal=0.0)
    except ValueError as exc:
        assert "signal scale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("zero signal scale must fail")


def test_negative_signal_component_is_not_silently_clipped() -> None:
    try:
        CachedPiecewiseBoundedLimit(
            np.array([1000.0, 1000.0]),
            np.eye(2) * 1.0e-8,
            np.array([1.0, -1.0e-12]),
            alpha=0.1,
            combined_mode="count_scale",
        )
    except ValueError as exc:
        assert "must be nonnegative" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("negative signal components must fail")


def test_constant_nonroot_cls_map_fails_bracketing() -> None:
    solver = make_solver()

    def broken(*_args, **_kwargs):
        return 1.0, {}

    solver._cls_at_eps2 = broken  # type: ignore[method-assign]
    try:
        solver.limit(np.array([1000]))
    except RuntimeError as exc:
        assert "failed to bracket" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a constant nonroot map must fail")


def test_nonmonotonic_cls_trace_fails_closed() -> None:
    solver = make_solver()
    first_positive = []

    def broken(eps2, *_args, **_kwargs):
        if eps2 == 0.0:
            value = 1.0
        elif not first_positive:
            first_positive.append(float(eps2))
            value = 0.05
        else:
            fraction = float(eps2) / first_positive[0]
            value = 0.20 if fraction <= 0.5 else 0.30
        return value, {"optimizer_ok": True}

    solver._cls_at_eps2 = broken  # type: ignore[method-assign]
    try:
        solver.limit(np.array([1000]))
    except RuntimeError as exc:
        assert "nonmonotonic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a nonmonotonic CLs trace must fail")


def test_covariance_conditioning_uses_smallest_decade_load() -> None:
    raw = np.array([[1.0, 1.000001], [1.000001, 1.0]])
    conditioned, record = condition_covariance_block(raw, np.array([10.0, 12.0]))
    np.linalg.cholesky(conditioned)
    assert record["selected_diagonal_load_relative"] == 1.0e-6
    assert record["raw_min_eigenvalue_relative"] < 0.0
    assert record["eigen_clipping_used"] is False
    assert record["effective_v_min_eigenvalue_relative"] > 0.0


def test_covariance_conditioning_refuses_load_beyond_cap() -> None:
    raw = np.array([[1.0, 1.001], [1.001, 1.0]])
    try:
        condition_covariance_block(raw, np.array([10.0, 12.0]))
    except RuntimeError as exc:
        assert "beyond 1e-4" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("large covariance repair must fail")


def test_covariance_conditioning_refuses_exact_cap() -> None:
    # Eigenvalues are 2.00005 and -5e-5, so the first successful decade load
    # would be the explicitly forbidden 1e-4 cap.
    raw = np.array([[1.0, 1.00005], [1.00005, 1.0]])
    try:
        condition_covariance_block(raw, np.array([10.0, 12.0]))
    except RuntimeError as exc:
        assert "forbidden 1e-4" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("the 1e-4 covariance-loading cap must fail")
