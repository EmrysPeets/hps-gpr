from types import SimpleNamespace

import hist
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.io import (
    _build_model,
    _compute_integral_density,
    estimate_background_for_dataset,
)


def _make_hist(values, lo=0.0, hi=6.0):
    histogram = hist.Hist(
        hist.axis.Regular(len(values), lo, hi, label="Mass / GeV"),
        storage=hist.storage.Weight(),
    )
    view = histogram.view()
    view.value[...] = np.asarray(values, dtype=float)
    view.variance[...] = np.asarray(values, dtype=float)
    return histogram


def test_integral_density_uses_fractional_overlap_for_exact_physical_window():
    histogram = hist.Hist(
        hist.axis.Variable([0.0, 1.0, 3.0, 6.0], label="Mass / GeV"),
        storage=hist.storage.Weight(),
    )
    view = histogram.view()
    view.value[...] = np.array([10.0, 40.0, 90.0])
    view.variance[...] = view.value

    density, metadata = _compute_integral_density(
        SimpleNamespace(density_histogram=histogram),
        mass=2.5,
        sigma_val=2.0,
        density_nsigma=1.0,
        return_metadata=True,
    )

    # [0.5, 4.5] contains half of bin 1, all of bin 2, and half of bin 3.
    assert density == 22.5
    assert metadata["density_window_lo"] == 0.5
    assert metadata["density_window_hi"] == 4.5
    assert metadata["density_window_width"] == 4.0
    assert metadata["density_source_n_bins"] == 3
    assert metadata["density_window_fully_covered"] is True


def test_integral_density_is_independent_of_gp_crop_and_rebin():
    source_histogram = _make_hist(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
        lo=0.0,
        hi=6.0,
    )
    common = dict(
        key="2021",
        label="HPS 2021",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.5,
        m_high=5.5,
        sigma_coeffs=[0.5],
        frad_coeffs=[0.05],
        hist_override=source_histogram,
    )
    narrow = DatasetConfig(**common, data_low=1.0, data_high=5.0)
    wide = DatasetConfig(**common, data_low=0.0, data_high=6.0)

    model_narrow = _build_model(
        narrow,
        blind=(0.5, 1.5),
        rebin=4,
        config=Config(),
        mass=1.0,
    )
    model_wide = _build_model(
        wide,
        blind=(0.5, 1.5),
        rebin=1,
        config=Config(),
        mass=1.0,
    )

    density_narrow = _compute_integral_density(
        model_narrow,
        mass=0.75,
        sigma_val=0.5,
        density_nsigma=1.0,
    )
    density_wide = _compute_integral_density(
        model_wide,
        mass=0.75,
        sigma_val=0.5,
        density_nsigma=1.0,
    )

    assert model_narrow.histogram.axes[0].edges[0] == 1.0
    assert model_wide.histogram.axes[0].edges[0] == 0.0
    assert model_narrow.density_histogram.axes[0].edges[0] == 0.0
    assert model_wide.density_histogram.axes[0].edges[0] == 0.0
    assert density_narrow == density_wide == 6.5


def test_integral_density_rejects_incomplete_physical_window():
    model = SimpleNamespace(
        density_histogram=_make_hist([2, 3, 5, 7], lo=0.0, hi=4.0)
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "does not fully cover physical window",
    ):
        _compute_integral_density(
            model,
            mass=0.25,
            sigma_val=0.5,
            density_nsigma=1.0,
        )


def test_explicit_kernel_is_forwarded_and_optimization_stays_disabled(monkeypatch):
    histogram = _make_hist([1, 2, 4, 8, 16, 32], lo=0.0, hi=6.0)
    dataset = DatasetConfig(
        key="2021",
        label="HPS 2021",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.0,
        m_high=6.0,
        sigma_coeffs=[0.5],
        frad_coeffs=[0.05],
        hist_override=histogram,
    )
    config = Config(neighborhood_rebin=1, n_restarts=7, blind_nsigma=1.0)
    explicit_kernel = ConstantKernel(7.0, (2.0, 50.0)) * RBF(
        0.2,
        (0.1, 0.4),
    )
    fake_gpr = SimpleNamespace(
        kernel=explicit_kernel,
        kernel_=explicit_kernel,
        log_marginal_likelihood_value_=0.0,
    )
    captured = {}

    import hps_gpr.io as io_mod

    def fake_fit_gpr(*args, **kwargs):
        captured.update(kwargs)
        return fake_gpr

    monkeypatch.setattr(io_mod, "fit_gpr", fake_fit_gpr)
    monkeypatch.setattr(
        io_mod,
        "predict_counts_from_log_gpr",
        lambda gpr, X_query, cfg: (
            np.full(len(np.asarray(X_query).reshape(-1)), 10.0),
            np.eye(len(np.asarray(X_query).reshape(-1)), dtype=float),
        ),
    )
    monkeypatch.setattr(
        io_mod,
        "predict_counts_mean_from_log_gpr",
        lambda gpr, X_query, cfg: np.full(
            len(np.asarray(X_query).reshape(-1)),
            12.0,
        ),
    )

    prediction = estimate_background_for_dataset(
        dataset,
        2.5,
        config,
        kernel=explicit_kernel,
        optimize=False,
    )

    assert captured["kernel"] is explicit_kernel
    assert captured["optimize"] is False
    assert prediction.optimizer_restarts == 0
