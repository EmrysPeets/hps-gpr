import json
from types import SimpleNamespace

from click.testing import CliRunner
import hist
import matplotlib.axes
import numpy as np
import pandas as pd

from hps_gpr.cli import main
from hps_gpr.config import Config, save_config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.funcform_toys import write_toy_scan_validation_plots
from hps_gpr.io import _compute_integral_density, estimate_background_for_dataset
from hps_gpr.toy_backgrounds import draw_full_background_toy


def _make_hist(values, lo=0.0, hi=6.0):
    h = hist.Hist(
        hist.axis.Regular(len(values), lo, hi, label="Mass / GeV"),
        storage=hist.storage.Weight(),
    )
    view = h.view()
    view.value[...] = np.asarray(values, dtype=float)
    view.variance[...] = np.asarray(values, dtype=float)
    return h


def test_draw_full_background_toy_fixed_total_preserves_total_and_handles_zero_mean():
    rng = np.random.default_rng(123)
    toy = draw_full_background_toy(
        np.array([1.0, 3.0, 2.0, 4.0]),
        rng,
        mode="fixed_total_multinomial",
        total_count=17,
    )
    assert toy.dtype.kind in {"i", "u"}
    assert int(np.sum(toy)) == 17

    zero_mean_toy = draw_full_background_toy(
        np.zeros(5, dtype=float),
        rng,
        mode="fixed_total_multinomial",
        total_count=9,
    )
    assert int(np.sum(zero_mean_toy)) == 9
    assert np.all(zero_mean_toy >= 0)


def test_integral_density_defaults_to_blind_width_and_allows_legacy_override(monkeypatch):
    toy_hist = _make_hist([1, 10, 1, 10, 1, 10], lo=0.0, hi=6.0)
    ds = DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.0,
        m_high=6.0,
        sigma_coeffs=[1.0],
        frad_coeffs=[0.1],
        hist_override=toy_hist,
    )

    import hps_gpr.io as io_mod

    fake_model = SimpleNamespace(histogram=toy_hist)
    fake_gpr = SimpleNamespace(kernel_=None, kernel=None, log_marginal_likelihood_value_=0.0)
    monkeypatch.setattr(io_mod, "_build_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(io_mod, "fit_gpr", lambda *args, **kwargs: fake_gpr)
    monkeypatch.setattr(
        io_mod,
        "predict_counts_from_log_gpr",
        lambda gpr, X_query, config: (
            np.full(len(np.asarray(X_query).reshape(-1)), 10.0),
            np.eye(len(np.asarray(X_query).reshape(-1)), dtype=float),
        ),
    )
    monkeypatch.setattr(
        io_mod,
        "predict_counts_mean_from_log_gpr",
        lambda gpr, X_query, config: np.full(len(np.asarray(X_query).reshape(-1)), 12.0),
    )
    monkeypatch.setattr(
        io_mod,
        "compute_kernel_ls_bounds",
        lambda *args, **kwargs: {"ls_lo": 0.1, "ls_hi": 0.2, "ls_init": 0.15, "sigma_x": 0.01},
    )
    monkeypatch.setattr(io_mod, "_extract_rbf_bounds_and_scale", lambda kernel: (0.1, 0.2, 0.15))

    cfg_default = Config(neighborhood_rebin=1, n_restarts=0, blind_nsigma=1.0)
    cfg_legacy = Config(
        neighborhood_rebin=1,
        n_restarts=0,
        blind_nsigma=1.0,
        eps2_density_nsigma=2.0,
    )

    expected_blind = _compute_integral_density(fake_model, 2.5, 1.0, density_nsigma=1.0)
    expected_legacy = _compute_integral_density(fake_model, 2.5, 1.0, density_nsigma=2.0)

    pred_default = estimate_background_for_dataset(ds, 2.5, cfg_default)
    pred_legacy = estimate_background_for_dataset(ds, 2.5, cfg_legacy)

    assert pred_default.integral_density == expected_blind
    assert pred_legacy.integral_density == expected_legacy
    assert pred_default.integral_density != pred_legacy.integral_density


def test_gp_toy_scan_cli_writes_compatible_outputs_and_merge(monkeypatch, tmp_path):
    cfg = Config(
        enable_2015=True,
        enable_2016=False,
        enable_2021=False,
        output_dir=str(tmp_path / "gp_toys"),
        full_toy_bkg_mode="fixed_total_multinomial",
        blind_nsigma=1.64,
    )
    cfg_path = tmp_path / "config.yaml"
    save_config(cfg, str(cfg_path))

    import hps_gpr.gp_toys as gp_toys_mod
    import hps_gpr.scan as scan_mod

    monkeypatch.setattr(
        gp_toys_mod,
        "build_gp_propagated_mean",
        lambda ds, config, rebin=None, restarts=None, optimize=True: (
            np.linspace(0.02, 0.14, 7),
            np.array([10, 11, 12, 9, 8, 7], dtype=float),
            np.array([9.5, 10.5, 11.0, 9.0, 8.5, 7.5], dtype=float),
        ),
    )

    def fake_run_scan(datasets, toy_cfg, mass_min=None, mass_max=None):
        ds_key = list(datasets.keys())[0]
        return (
            pd.DataFrame(
                [
                    {
                        "dataset": ds_key,
                        "mass_GeV": 0.040,
                        "Z_analytic": 1.1,
                        "p0_analytic": 0.14,
                        "eps2_up": 1.0e-5,
                        "extract_success": True,
                    },
                    {
                        "dataset": ds_key,
                        "mass_GeV": 0.050,
                        "Z_analytic": 1.8,
                        "p0_analytic": 0.04,
                        "eps2_up": 1.3e-5,
                        "extract_success": True,
                    },
                ]
            ),
            pd.DataFrame(),
        )

    monkeypatch.setattr(scan_mod, "run_scan", fake_run_scan)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "gp-toy-scan",
            "--config", str(cfg_path),
            "--dataset", "2015",
            "--n-toys", "2",
        ],
    )
    assert result.exit_code == 0, result.output

    outdir = tmp_path / "gp_toys"
    source_dir = outdir / "toy_scans" / "2015" / "gp_propagated_mean_refit_fixedtotal"
    toy0 = source_dir / "toy_0000"
    toy1 = source_dir / "toy_0001"
    assert (toy0 / "results_single.csv").exists()
    assert (toy1 / "results_single.csv").exists()

    with open(toy0 / "toy_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["source_model"] == "gp_propagated_mean_refit_fixedtotal"
    assert meta["full_toy_bkg_mode"] == "fixed_total_multinomial"
    assert meta["full_input_total_count"] == 57

    merge_result = runner.invoke(
        main,
        [
            "toy-scan-merge",
            "--input-dir", str(outdir),
            "--output-dir", str(tmp_path / "merged"),
        ],
    )
    assert merge_result.exit_code == 0, merge_result.output

    merged = pd.read_csv(tmp_path / "merged" / "toy_scan_merged.csv")
    assert set(["source_model", "source_label", "toy_index", "toy_hist"]).issubset(merged.columns)
    assert set(merged["source_model"].astype(str)) == {"gp_propagated_mean_refit_fixedtotal"}


def test_write_toy_scan_validation_plots_large_ensemble_avoids_spaghetti(monkeypatch, tmp_path):
    rows = []
    summary_rows = []
    masses = [0.040, 0.050, 0.060]
    n_toys = 60
    for toy_index in range(n_toys):
        z_vals = [0.2 + 0.01 * toy_index, 0.4 + 0.01 * toy_index, 0.3 + 0.01 * toy_index]
        p0_vals = [0.4, 0.08, 0.2]
        eps_vals = [1.0e-5, 1.2e-5, 1.1e-5]
        for mass, z_val, p0_val, eps_val in zip(masses, z_vals, p0_vals, eps_vals):
            rows.append(
                {
                    "dataset": "2015",
                    "toy_index": toy_index,
                    "toy_hist": f"gp_toy_{toy_index:04d}",
                    "function_tag": "gp_propagated_mean_refit_fixedtotal",
                    "source_model": "gp_propagated_mean_refit_fixedtotal",
                    "source_label": "gp_propagated_mean_refit_fixedtotal",
                    "source_root": "",
                    "container": "generated",
                    "mass_GeV": mass,
                    "Z_analytic": z_val,
                    "p0_analytic": p0_val,
                    "eps2_up": eps_val * (1.0 + 0.01 * toy_index),
                    "extract_success": True,
                }
            )
        summary_rows.append(
            {
                "dataset": "2015",
                "toy_index": toy_index,
                "toy_hist": f"gp_toy_{toy_index:04d}",
                "function_tag": "gp_propagated_mean_refit_fixedtotal",
                "source_model": "gp_propagated_mean_refit_fixedtotal",
                "source_label": "gp_propagated_mean_refit_fixedtotal",
                "source_root": "",
                "container": "generated",
                "n_fail": 0,
                "max_Z_analytic": max(z_vals),
                "mass_at_max_Z": masses[int(np.argmax(z_vals))],
                "min_p0_analytic": min(p0_vals),
            }
        )

    merged = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)

    plot_calls = {"n": 0}
    real_plot = matplotlib.axes.Axes.plot

    def counted_plot(self, *args, **kwargs):
        plot_calls["n"] += 1
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", counted_plot)

    stems = write_toy_scan_validation_plots(merged, summary, str(tmp_path))

    assert stems
    assert plot_calls["n"] < 20
