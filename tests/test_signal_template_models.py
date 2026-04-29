import numpy as np

from hps_gpr.config import Config
from hps_gpr.template import (
    build_full_template,
    build_window_template_from_full,
    signal_kernel_covariance,
)


def test_default_signal_model_matches_existing_gaussian_template():
    edges = np.linspace(0.030, 0.070, 41)
    cfg = Config(signal_model="default")

    implicit = build_full_template(edges, 0.050, 0.002, config=cfg)
    explicit = build_full_template(edges, 0.050, 0.002, signal_model="default")

    assert np.isclose(np.sum(implicit), 1.0)
    np.testing.assert_allclose(implicit, explicit)


def test_kernel_signal_model_is_normalized_and_resolution_localized():
    edges = np.linspace(0.030, 0.070, 41)
    cfg = Config(
        signal_model="kernel",
        signal_kernel_width_factor=1.0,
        signal_kernel_length_scale_factor=1.0,
    )

    w = build_full_template(edges, 0.050, 0.002, config=cfg)
    centers = 0.5 * (edges[:-1] + edges[1:])

    assert np.isclose(np.sum(w), 1.0)
    assert np.argmax(w) in np.where(np.abs(centers - 0.050) <= 0.0015)[0]
    assert np.sum(w[np.abs(centers - 0.050) <= 3.0 * 0.002]) > 0.95


def test_kernel_window_template_preserves_full_range_fraction():
    edges = np.linspace(0.030, 0.070, 41)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = np.abs(centers - 0.050) <= 1.64 * 0.002
    cfg = Config(signal_model="kernel")

    w_win, w_full = build_window_template_from_full(edges, mask, 0.050, 0.002, config=cfg)

    assert np.isclose(np.sum(w_full), 1.0)
    assert np.isclose(np.sum(w_win), np.sum(w_full[mask]))
    assert 0.0 < np.sum(w_win) < 1.0


def test_signal_kernel_covariance_is_symmetric_positive_semidefinite():
    x = np.linspace(0.045, 0.055, 11)

    K = signal_kernel_covariance(x, 0.050, 0.002)

    np.testing.assert_allclose(K, K.T)
    evals = np.linalg.eigvalsh(K)
    assert np.min(evals) > -1e-10
