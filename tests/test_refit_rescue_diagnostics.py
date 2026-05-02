from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.gpr import _extract_rbf_bounds_and_scale, fit_gpr, make_fixed_kernel
from hps_gpr.injection import (
    _InjectionMassContext,
    _resolve_refit_kernel_lock_values,
    _resolve_sigma_a_references,
    _signal_tail_alpha_multiplier,
    _simulate_toy_rows_chunk,
)


def _make_dataset() -> DatasetConfig:
    return DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="dummy.root",
        hist_name="h",
        m_low=0.020,
        m_high=0.130,
        sigma_coeffs=[0.001],
        frad_coeffs=[0.1],
    )


def _make_refit_context(**overrides) -> _InjectionMassContext:
    values = dict(
        ds=_make_dataset(),
        mass=0.05,
        mu=np.array([20.0]),
        cov=np.eye(1),
        mu_full=np.array([1000.0, 1000.0, 1000.0]),
        y_full=np.array([10.0, 20.0, 30.0]),
        x_full=np.array([0.045, 0.050, 0.055]),
        msk_blind=np.array([False, True, False]),
        msk_train=np.array([True, False, True]),
        tmpl_win=np.array([1.0]),
        tmpl_full=np.array([0.02, 0.96, 0.02]),
        sigmaA_ref=1.0,
        sigmaA_ref_prefit=1.0,
        sigmaA_ref_matched=np.nan,
        sigmaA_ref_mode="prefit_asimov",
        sigmaA_ref_matched_ok=np.nan,
        sigmaA_ref_error="",
        sigma_val=0.001,
        sigma_x=0.02,
        kernel_ls_policy="resolution_scaled_local",
        kernel_ls_res_lower_factor=1.0,
        kernel_ls_res_upper_factor=8.0,
        ls_lo=0.01,
        ls_hi=0.08,
        ls_init=0.028,
        initial_ls_opt=0.25,
        initial_const_opt=4.0,
        integral_density=1.0,
        A_per_eps2_unit=1.0,
        f_win=0.96,
        f_full=1.0,
        f_train=0.04,
        f_train_frac=0.04,
        n_train=2,
        n_train_low=1,
        n_train_high=1,
        n_blind=1,
        blind_nsigma=1.64,
        train_exclude_nsigma=3.0,
        signal_model="default",
        inj_mode="poisson",
        inj_shape_mode="full",
        inj_background_mode="fixed_hist",
        refit_gp_on_toy=True,
        refit_restarts=0,
        refit_optimize=True,
        allow_negative=True,
        mvn_method="reject_then_clip",
        mvn_max_tries=80,
    )
    values.update(overrides)
    return _InjectionMassContext(**values)


def test_make_fixed_kernel_freezes_constant_and_length_scale():
    kernel = make_fixed_kernel(3.5, 0.25)

    assert float(kernel.k1.constant_value) == pytest.approx(3.5)
    assert kernel.k1.constant_value_bounds == "fixed"
    assert float(kernel.k2.length_scale) == pytest.approx(0.25)
    assert kernel.k2.length_scale_bounds == "fixed"
    lo, hi, ls = _extract_rbf_bounds_and_scale(kernel)
    assert (lo, hi, ls) == pytest.approx((0.25, 0.25, 0.25))


def test_initial_fit_kernel_lock_uses_preinjection_hyperparameters(monkeypatch):
    import hps_gpr.injection as inj

    captured = {}

    def fake_fit_gpr(X, y, config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kernel_=kwargs["kernel"])

    monkeypatch.setattr(inj, "make_kernel_for_dataset", lambda *args, **kwargs: "base-kernel")
    monkeypatch.setattr(inj, "fit_gpr", fake_fit_gpr)
    monkeypatch.setattr(inj, "predict_counts_from_log_gpr", lambda gpr, x_win, config: (np.array([20.0]), np.eye(1)))
    monkeypatch.setattr(
        inj,
        "fit_A_profiled_gaussian",
        lambda obs, mu, cov, tmpl_win, allow_negative: {
            "A_hat": 0.0,
            "sigma_A": 1.0,
            "success": True,
            "nll": 0.0,
        },
    )

    ctx = _make_refit_context(
        refit_kernel_lock_mode="initial_fit",
        refit_lock_const_opt=4.0,
        refit_lock_ls_opt=0.25,
    )
    rows = _simulate_toy_rows_chunk(
        ctx,
        Config(inj_background_mode="fixed_hist", inj_refit_kernel_lock_mode="initial_fit"),
        toy_indices=[0],
        A_inj=0.0,
        inj_nsigma=0.0,
        point_seed=123,
        threads_per_worker=1,
    )

    kernel = captured["kernel"]
    assert float(kernel.k1.constant_value) == pytest.approx(4.0)
    assert kernel.k1.constant_value_bounds == "fixed"
    assert float(kernel.k2.length_scale) == pytest.approx(0.25)
    assert kernel.k2.length_scale_bounds == "fixed"
    assert captured["optimize"] is False
    assert rows[0]["refit_kernel_lock_mode"] == "initial_fit"
    assert rows[0]["refit_lock_ls_opt"] == pytest.approx(0.25)


def test_ensemble_file_kernel_lock_loads_per_mass_medians(tmp_path):
    lock_file = tmp_path / "kernel_lock.csv"
    pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.05, "const_opt": 2.0, "ls_opt": 0.20},
            {"dataset": "2015", "mass_GeV": 0.05, "const_opt": 4.0, "ls_opt": 0.40},
            {"dataset": "2015", "mass_GeV": 0.06, "const_opt": 9.0, "ls_opt": 0.90},
        ]
    ).to_csv(lock_file, index=False)
    cfg = Config(
        inj_refit_kernel_lock_mode="ensemble_file",
        inj_refit_kernel_lock_file=str(lock_file),
    )

    mode, const, ls = _resolve_refit_kernel_lock_values(
        cfg,
        dataset_key="2015",
        mass=0.05,
        initial_const_opt=1.0,
        initial_ls_opt=0.1,
    )

    assert mode == "ensemble_file"
    assert const == pytest.approx(3.0)
    assert ls == pytest.approx(0.30)


def test_ensemble_file_kernel_lock_prefers_per_toy_row(tmp_path):
    lock_file = tmp_path / "kernel_lock_by_toy.csv"
    pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.05, "toy_index": 6, "const_opt": 2.0, "ls_opt": 0.20},
            {"dataset": "2015", "mass_GeV": 0.05, "toy_index": 7, "const_opt": 9.0, "ls_opt": 0.90},
            {"dataset": "2015", "mass_GeV": 0.05, "toy_index": 8, "const_opt": 4.0, "ls_opt": 0.40},
        ]
    ).to_csv(lock_file, index=False)
    cfg = Config(
        inj_refit_kernel_lock_mode="ensemble_file",
        inj_refit_kernel_lock_file=str(lock_file),
    )

    mode, const, ls = _resolve_refit_kernel_lock_values(
        cfg,
        dataset_key="2015",
        mass=0.05,
        toy_index=7,
        initial_const_opt=1.0,
        initial_ls_opt=0.1,
    )
    assert mode == "ensemble_file"
    assert const == pytest.approx(9.0)
    assert ls == pytest.approx(0.90)

    _, const_fallback, ls_fallback = _resolve_refit_kernel_lock_values(
        cfg,
        dataset_key="2015",
        mass=0.05,
        toy_index=99,
        initial_const_opt=1.0,
        initial_ls_opt=0.1,
    )
    assert const_fallback == pytest.approx(4.0)
    assert ls_fallback == pytest.approx(0.40)


def test_prefit_sigma_reference_mode_reproduces_current_behavior(monkeypatch):
    import hps_gpr.injection as inj

    pred = SimpleNamespace(mu=np.array([10.0]), cov=np.eye(1), edges_full=np.array([0.0, 1.0]), blind=(0.0, 1.0), sigma_val=0.1)
    monkeypatch.setattr(inj, "_sigmaA_reference", lambda *args, **kwargs: 2.5)
    monkeypatch.setattr(inj, "_matched_refit_bonly_sigmaA_reference", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("matched refit should not run")))

    refs = _resolve_sigma_a_references(
        pred,
        _make_dataset(),
        Config(inj_sigma_a_ref_mode="prefit_asimov"),
        mass=0.05,
        source="asimov",
        rng=np.random.default_rng(1),
        refit_gp_on_toy=True,
        refit_restarts=0,
        refit_optimize=True,
        x_full=np.array([0.045, 0.050, 0.055]),
        msk_train=np.array([True, False, True]),
        msk_blind=np.array([False, True, False]),
        y_full_bonly=np.array([10.0, 20.0, 30.0]),
        tmpl_win=np.array([1.0]),
        lock_mode="none",
        lock_const_opt=np.nan,
        lock_ls_opt=np.nan,
        tail_alpha_multiplier=None,
    )

    assert refs["sigmaA_ref"] == pytest.approx(2.5)
    assert refs["sigmaA_ref_prefit"] == pytest.approx(2.5)
    assert np.isnan(refs["sigmaA_ref_matched"])
    assert refs["sigmaA_ref_mode"] == "prefit_asimov"


def test_matched_refit_sigma_reference_uses_bonly_refit_covariance(monkeypatch):
    import hps_gpr.injection as inj

    captured = {}
    pred = SimpleNamespace(mu=np.array([10.0]), cov=np.eye(1), edges_full=np.array([0.0, 1.0]), blind=(0.0, 1.0), sigma_val=0.1)

    def fake_fit_gpr(X, y, config, **kwargs):
        captured["X"] = np.asarray(X)
        captured["y"] = np.asarray(y)
        captured["kwargs"] = kwargs
        return SimpleNamespace(kernel_=kwargs["kernel"])

    monkeypatch.setattr(inj, "_sigmaA_reference", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(inj, "make_kernel_for_dataset", lambda *args, **kwargs: make_fixed_kernel(1.0, 0.5))
    monkeypatch.setattr(inj, "fit_gpr", fake_fit_gpr)
    monkeypatch.setattr(inj, "predict_counts_from_log_gpr", lambda gpr, x_win, config: (np.array([20.0]), np.array([[7.0]])))
    monkeypatch.setattr(inj, "fit_A_profiled_gaussian_details", lambda n_ref, b, cov, tmpl, allow_negative: {"sigma_A": float(cov[0, 0])})

    refs = _resolve_sigma_a_references(
        pred,
        _make_dataset(),
        Config(inj_sigma_a_ref_mode="matched_refit_bonly"),
        mass=0.05,
        source="asimov",
        rng=np.random.default_rng(1),
        refit_gp_on_toy=True,
        refit_restarts=0,
        refit_optimize=True,
        x_full=np.array([0.045, 0.050, 0.055]),
        msk_train=np.array([True, False, True]),
        msk_blind=np.array([False, True, False]),
        y_full_bonly=np.array([10.0, 20.0, 30.0]),
        tmpl_win=np.array([1.0]),
        lock_mode="none",
        lock_const_opt=np.nan,
        lock_ls_opt=np.nan,
        tail_alpha_multiplier=None,
    )

    assert refs["sigmaA_ref_prefit"] == pytest.approx(2.0)
    assert refs["sigmaA_ref_matched"] == pytest.approx(7.0)
    assert refs["sigmaA_ref"] == pytest.approx(7.0)
    assert refs["sigmaA_ref_mode"] == "matched_refit_bonly"
    assert refs["sigmaA_ref_matched_ok"] == pytest.approx(1.0)
    np.testing.assert_allclose(captured["y"], np.array([10.0, 30.0]))


def test_signal_tail_alpha_multiplier_only_changes_training_bins():
    tmpl_full = np.array([0.010, 0.500, 0.000, 0.040])
    msk_train = np.array([True, False, True, True])

    mult, stats = _signal_tail_alpha_multiplier(
        tmpl_full,
        msk_train,
        scale=2.0,
        threshold=0.020,
    )

    np.testing.assert_allclose(mult, np.array([1.0, 1.0, 3.0]))
    assert stats["n_bins"] == 1.0
    assert stats["max"] == pytest.approx(3.0)


def test_fit_gpr_applies_alpha_multiplier_to_training_alpha():
    cfg = Config(pre_log=True, n_restarts=0)
    kernel = make_fixed_kernel(1.0, 1.0)
    gpr = fit_gpr(
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 20.0, 40.0]),
        cfg,
        restarts=0,
        kernel=kernel,
        optimize=False,
        alpha_multiplier=np.array([1.0, 2.0, 4.0]),
    )

    np.testing.assert_allclose(gpr.alpha, np.array([0.1, 0.1, 0.1]))
