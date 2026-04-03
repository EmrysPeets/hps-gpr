import numpy as np

from hps_gpr.bands import expected_ul_bands_for_dataset
from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.evaluation import combined_cls_limit_epsilon2_from_vectors
from hps_gpr.io import BlindPrediction
from hps_gpr.statistics import (
    asymptotic_cls_profiled_gaussian,
    p0_profiled_gaussian_LRT,
    qmu_tilde_profiled_gaussian,
    toy_cls_profiled_gaussian,
)
from hps_gpr.template import cls_limit_for_template


def test_p0_profiled_is_zero_when_bounded_best_fit_sits_at_null():
    n = np.array([20.0, 20.0, 20.0])
    b = np.array([20.0, 20.0, 20.0])
    cov = np.diag([1.0, 1.0, 1.0])
    tmpl = np.array([0.2, 0.5, 0.3])

    p0, Z, q0, _ = p0_profiled_gaussian_LRT(n, b, cov, tmpl)

    assert np.isclose(q0, 0.0, atol=1e-9)
    assert np.isclose(Z, 0.0, atol=1e-9)
    assert np.isclose(p0, 0.5, atol=1e-9)


def test_qmu_tilde_is_zero_when_unbounded_best_fit_exceeds_test_strength():
    n = np.array([30.0, 30.0, 30.0])
    b = np.array([20.0, 20.0, 20.0])
    cov = np.diag([1.0, 1.0, 1.0])
    tmpl = np.array([0.2, 0.5, 0.3])

    qmu, info = qmu_tilde_profiled_gaussian(n, b, cov, tmpl, 3.0)

    assert np.isclose(qmu, 0.0, atol=1e-9)
    assert info["branch"] == "muhat_gt_test"
    assert info["A_hat"] > 3.0


def test_qmu_tilde_uses_boundary_denominator_for_negative_unbounded_best_fit():
    n = np.array([0.0, 0.0, 0.0])
    b = np.array([20.0, 20.0, 20.0])
    cov = np.diag([1.0, 1.0, 1.0])
    tmpl = np.array([0.2, 0.5, 0.3])

    qmu, info = qmu_tilde_profiled_gaussian(n, b, cov, tmpl, 8.0)

    assert qmu > 0.0
    assert info["branch"] == "boundary"
    assert info["A_hat"] < 0.0
    assert np.isclose(qmu, 16.0, atol=1e-6)


def test_asymptotic_and_toy_cls_are_consistent_in_high_stat_regime():
    rng = np.random.default_rng(123)
    n = np.array([81.0, 79.0, 80.0, 82.0])
    b = np.array([80.0, 80.0, 80.0, 80.0])
    cov = np.diag([4.0, 4.0, 4.0, 4.0])
    tmpl = np.array([0.1, 0.4, 0.35, 0.15])

    cls_asym, clsb_asym, clb_asym, _ = asymptotic_cls_profiled_gaussian(20.0, n, b, cov, tmpl)
    cls_toy, clsb_toy, clb_toy, _ = toy_cls_profiled_gaussian(20.0, n, b, cov, tmpl, rng, 200)

    assert abs(cls_asym - cls_toy) < 0.08
    assert abs(clsb_asym - clsb_toy) < 0.05
    assert abs(clb_asym - clb_toy) < 0.05


def test_combined_cls_matches_single_channel_limit_for_equivalent_vectors():
    cfg = Config(cls_mode="asymptotic", cls_alpha=0.05)
    obs = np.array([21.0, 19.0, 20.0, 18.0])
    b = np.array([20.0, 20.0, 20.0, 20.0])
    cov = np.diag([2.0, 2.0, 2.0, 2.0])
    s_unit = np.array([2.5e10, 7.5e10, 1.0e11, 5.0e10])

    eps2_up = combined_cls_limit_epsilon2_from_vectors(obs, b, cov, s_unit, cfg)
    amp_limit, _ = cls_limit_for_template(
        obs,
        b,
        cov,
        s_unit,
        alpha=cfg.cls_alpha,
        mode=cfg.cls_mode,
        use_eps2=False,
    )

    assert np.isclose(eps2_up, amp_limit, rtol=1e-3, atol=1e-12)


def _fake_prediction(ds, mass, config, train_exclude_nsigma=None):
    sigma = 0.0018
    edges_full = np.linspace(float(mass) - 0.010, float(mass) + 0.010, 21)
    x_full = 0.5 * (edges_full[:-1] + edges_full[1:])
    blind = (
        float(mass) - float(config.blind_nsigma) * sigma,
        float(mass) + float(config.blind_nsigma) * sigma,
    )
    mu_full = 40.0 + 7.0 * np.exp(-0.5 * ((x_full - float(mass)) / 0.006) ** 2)
    blind_mask = (x_full >= blind[0]) & (x_full <= blind[1])
    idx = np.where(blind_mask)[0]
    edges = edges_full[idx[0] : idx[-1] + 2]
    mu = mu_full[blind_mask]
    cov = np.diag(np.clip(mu, 1.0, None))
    obs = np.round(mu).astype(int)
    tn = float(train_exclude_nsigma if train_exclude_nsigma is not None else config.blind_nsigma)
    return BlindPrediction(
        mu=np.asarray(mu, float),
        cov=np.asarray(cov, float),
        obs=np.asarray(obs, int),
        edges=np.asarray(edges, float),
        sigma_val=float(sigma),
        blind=blind,
        x_full=np.asarray(x_full, float),
        y_full=np.round(mu_full).astype(int),
        mu_full=np.asarray(mu_full, float),
        edges_full=np.asarray(edges_full, float),
        blind_mask=np.asarray(blind_mask, bool),
        integral_density=1.25e8,
        blind_train=(float(mass) - tn * sigma, float(mass) + tn * sigma),
    )


def test_expected_ul_bands_reports_profiled_cls_metadata(monkeypatch):
    import hps_gpr.bands as bands_mod

    monkeypatch.setattr(bands_mod, "estimate_background_for_dataset", _fake_prediction)

    ds = DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="unused.root",
        hist_name="hist",
        m_low=0.02,
        m_high=0.13,
        sigma_coeffs=[0.0018],
        frad_coeffs=[0.085],
    )
    cfg = Config(
        cls_mode="asymptotic",
        cls_alpha=0.05,
        ul_bands_toys=4,
        blind_nsigma=1.64,
        enable_2015=True,
    )

    df = expected_ul_bands_for_dataset(
        ds,
        [0.040],
        cfg,
        n_toys=4,
        seed=7,
        use_eps2=True,
    )

    assert list(df["cls_statistic"]) == ["tilde_q_mu"]
    assert list(df["cls_calibration"]) == ["asymptotic"]
    assert list(df["global_method"]) == ["sidak_approx"]

