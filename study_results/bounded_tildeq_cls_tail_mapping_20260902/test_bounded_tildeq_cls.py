from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from bounded_tildeq_cls import (  # noqa: E402
    asymptotic_cls_profiled_gaussian_piecewise,
    bounded_tildeq_asymptotic_tails,
)
from hps_gpr.statistics import asymptotic_cls_profiled_gaussian  # noqa: E402


def test_qobs_below_qA_uses_square_root_branch() -> None:
    q_obs = 1.0
    q_a = 4.0
    out = bounded_tildeq_asymptotic_tails(q_obs, q_a)
    expected_clsb = norm.sf(1.0)
    expected_clb = norm.cdf(1.0)
    assert out.branch == "qobs_le_qA"
    assert np.isclose(out.cl_sb, expected_clsb, rtol=0.0, atol=1.0e-15)
    assert np.isclose(out.cl_b, expected_clb, rtol=0.0, atol=1.0e-15)
    assert np.isclose(out.cls, expected_clsb / expected_clb, rtol=0.0, atol=1.0e-15)


def test_qobs_above_qA_uses_negative_muhat_piecewise_branch() -> None:
    q_obs = 8.0
    q_a = 4.0
    out = bounded_tildeq_asymptotic_tails(q_obs, q_a)
    expected_z_sb = (q_obs + q_a) / (2.0 * np.sqrt(q_a))
    expected_z_b = (q_a - q_obs) / (2.0 * np.sqrt(q_a))
    expected_clsb = norm.sf(expected_z_sb)
    expected_clb = norm.cdf(expected_z_b)
    assert out.branch == "qobs_gt_qA_negative_muhat"
    assert np.isclose(out.z_sb, expected_z_sb, rtol=0.0, atol=1.0e-15)
    assert np.isclose(out.z_b, expected_z_b, rtol=0.0, atol=1.0e-15)
    assert np.isclose(out.cl_sb, expected_clsb, rtol=0.0, atol=1.0e-15)
    assert np.isclose(out.cl_b, expected_clb, rtol=0.0, atol=1.0e-15)
    assert np.isclose(out.cls, expected_clsb / expected_clb, rtol=0.0, atol=1.0e-15)


def test_mapping_is_continuous_at_qA() -> None:
    q_a = 3.25
    at = bounded_tildeq_asymptotic_tails(q_a, q_a)
    above = bounded_tildeq_asymptotic_tails(q_a * (1.0 + 1.0e-9), q_a)
    assert at.branch == "qobs_le_qA"
    assert above.branch == "qobs_gt_qA_negative_muhat"
    assert np.isclose(at.cl_sb, above.cl_sb, rtol=2.0e-8, atol=0.0)
    assert np.isclose(at.cl_b, above.cl_b, rtol=2.0e-8, atol=0.0)
    assert np.isclose(at.cls, above.cls, rtol=2.0e-8, atol=0.0)


def test_logspace_ratio_stays_finite_when_individual_tails_underflow() -> None:
    out = bounded_tildeq_asymptotic_tails(1.0e6, 1.0e4)
    assert np.isfinite(out.log_cls)
    assert np.isfinite(out.cls)
    assert out.cls >= 0.0


def test_profiled_wrapper_selects_qobs_above_qA_for_deficit() -> None:
    obs = np.array([900.0])
    b = np.array([1000.0])
    cov = np.array([[1.0e-8]])
    template = np.array([1.0])
    cls, clsb, clb, info = asymptotic_cls_profiled_gaussian_piecewise(
        50.0, obs, b, cov, template
    )
    expected = bounded_tildeq_asymptotic_tails(
        info["qmu_obs"], info["qmu_asimov_b"]
    )
    assert info["tail_branch"] == "qobs_gt_qA_negative_muhat"
    assert info["qmu_obs"] > info["qmu_asimov_b"]
    assert info["ok"] is True
    assert np.isclose(cls, expected.cls, rtol=0.0, atol=1.0e-15)
    assert np.isclose(clsb, expected.cl_sb, rtol=0.0, atol=1.0e-15)
    assert np.isclose(clb, expected.cl_b, rtol=0.0, atol=1.0e-15)


def test_regression_exposes_legacy_missing_tail_branch() -> None:
    obs = np.array([900.0])
    b = np.array([1000.0])
    cov = np.array([[1.0e-8]])
    template = np.array([1.0])
    corrected, _, _, info = asymptotic_cls_profiled_gaussian_piecewise(
        50.0, obs, b, cov, template
    )
    legacy, _, _, legacy_info = asymptotic_cls_profiled_gaussian(
        50.0, obs, b, cov, template
    )
    assert info["qmu_obs"] > info["qmu_asimov_b"]
    assert np.isclose(info["qmu_obs"], legacy_info["qmu_obs"], atol=1.0e-12)
    assert np.isclose(info["qmu_asimov_b"], legacy_info["qmu_asimov_b"], atol=1.0e-12)
    assert not np.isclose(corrected, legacy, rtol=1.0e-2, atol=0.0)
    # Fixed regression anchors for this high-count one-bin example.
    assert np.isclose(legacy, 0.009108556386479448, rtol=1.0e-10, atol=0.0)
    assert np.isclose(corrected, 0.0015819802695072318, rtol=1.0e-10, atol=0.0)


def test_profiled_wrapper_matches_existing_branch_when_qobs_not_above_qA() -> None:
    obs = np.array([1010.0])
    b = np.array([1000.0])
    cov = np.array([[1.0e-8]])
    template = np.array([1.0])
    corrected, _, _, info = asymptotic_cls_profiled_gaussian_piecewise(
        50.0, obs, b, cov, template
    )
    legacy, _, _, legacy_info = asymptotic_cls_profiled_gaussian(
        50.0, obs, b, cov, template
    )
    assert info["qmu_obs"] <= info["qmu_asimov_b"]
    assert info["tail_branch"] == "qobs_le_qA"
    assert np.isclose(info["qmu_obs"], legacy_info["qmu_obs"], atol=1.0e-12)
    assert np.isclose(corrected, legacy, rtol=0.0, atol=1.0e-15)


def test_zero_asimov_separation_fails_closed() -> None:
    try:
        bounded_tildeq_asymptotic_tails(0.0, 0.0)
    except ValueError as exc:
        assert "q_A=0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("q_A=0 must fail closed")

