from types import SimpleNamespace

from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest

from hps_gpr.cli import main
from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.injection import (
    collapse_fragmented_injection_summary,
    run_funcform_injection_extraction_toys,
    run_injection_extraction_toys,
    run_injection_extraction_streaming,
    run_injection_extraction_streaming_combined,
    summarize_injection_grid,
)
from hps_gpr.funcform_toys import FuncFormToySpec


def _make_dataset():
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


def _make_dataset_2016():
    return DatasetConfig(
        key="2016",
        label="HPS 2016",
        root_path="dummy_2016.root",
        hist_name="h",
        m_low=0.020,
        m_high=0.130,
        sigma_coeffs=[0.001],
        frad_coeffs=[0.1],
    )


def _install_fast_injection_mocks(monkeypatch):
    import hps_gpr.injection as inj

    def fake_estimate_background_for_dataset(ds, m, config):
        return SimpleNamespace(
            edges=np.array([0.0, 1.0, 2.0]),
            sigma_val=1.0,
            mu=np.array([2.0, 3.0]),
            cov=np.eye(2),
            edges_full=np.array([0.0, 1.0, 2.0]),
            x_full=np.array([0.5, 1.5]),
            blind=(0.0, 2.0),
            train_exclude_nsigma=1.64,
            mu_full=np.array([2.0, 3.0]),
            sigma_x=1.0,
        )

    monkeypatch.setattr(inj, "estimate_background_for_dataset", fake_estimate_background_for_dataset)
    monkeypatch.setattr(inj, "build_template", lambda edges, mass, sigma: np.array([0.6, 0.4]))
    monkeypatch.setattr(inj, "_sigmaA_reference", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(inj, "draw_bkg_mvn_nonneg", lambda mu, cov, n, rng, method, max_tries: np.tile(mu, (n, 1)))
    monkeypatch.setattr(
        inj,
        "fit_A_profiled_gaussian",
        lambda obs, mu, cov, tmpl_win, allow_negative: {
            "A_hat": float(np.sum(obs)),
            "sigma_A": 2.0,
            "success": True,
            "nll": 0.0,
        },
    )


def test_run_injection_extraction_toys_skips_writing_toy_csv_when_disabled(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(output_dir=str(tmp_path), inj_write_toy_csv=False)

    df = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[0.0, 1.0],
        n_toys=2,
    )

    assert len(df) == 4
    assert not (tmp_path / "injection_extraction" / "inj_extract_toys_2015.csv").exists()


def test_run_injection_extraction_toys_writes_toy_csv_when_enabled(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(output_dir=str(tmp_path), inj_write_toy_csv=True)

    df = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[0.0],
        n_toys=1,
    )

    assert len(df) == 1
    assert (tmp_path / "injection_extraction" / "inj_extract_toys_2015.csv").exists()


def test_sigma_mode_uses_explicit_strength_overrides(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(
        output_dir=str(tmp_path),
        inj_write_toy_csv=False,
        inj_strength_mode="sigmaA",
        inj_sigma_multipliers=[1.0, 2.0, 3.0, 5.0],
    )

    df = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[1.0],
        n_toys=2,
        strengths_mode="sigmaA",
    )

    assert len(df) == 2
    assert sorted(df["inj_nsigma"].unique().tolist()) == [1.0]


def test_funcform_injection_uses_fixed_toy_background_counts(tmp_path, monkeypatch):
    import hps_gpr.injection as inj

    monkeypatch.setattr(inj, "load_funcform_toy_hist", lambda *args, **kwargs: object())
    monkeypatch.setattr(inj, "resolve_funcform_scan_range_gev", lambda dataset_key, root_path: (0.02, 0.13))
    monkeypatch.setattr(
        inj,
        "estimate_background_for_dataset",
        lambda ds, m, config, train_exclude_nsigma=None: SimpleNamespace(
            sigma_val=1.0,
            mu=np.array([99.0, 99.0]),
            cov=np.eye(2),
            mu_full=np.array([99.0, 99.0]),
            y_full=np.array([5.0, 7.0]),
            edges_full=np.array([0.0, 1.0, 2.0]),
            x_full=np.array([0.5, 1.5]),
            blind=(0.0, 2.0),
            blind_mask=np.array([True, True]),
            integral_density=10.0,
            sigma_x=1.0,
            train_exclude_nsigma=1.64,
        ),
    )
    monkeypatch.setattr(inj, "_sigmaA_reference", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(
        inj,
        "fit_A_profiled_gaussian",
        lambda obs, mu, cov, tmpl_win, allow_negative: {
            "A_hat": float(np.sum(obs)),
            "sigma_A": 1.5,
            "success": True,
            "nll": 0.0,
        },
    )

    cfg = Config(output_dir=str(tmp_path), inj_write_toy_csv=False, inj_strength_mode="sigmaA")
    specs = [
        FuncFormToySpec(
            source_root="funcform.root",
            container="fShiftSigPowTail",
            function_tag="fShiftSigPowTail",
            toy_name="fShiftSigPowTail_toy_0",
            toy_index=0,
        )
    ]

    df = run_funcform_injection_extraction_toys(
        _make_dataset(),
        cfg,
        specs=specs,
        masses=[0.05],
        strengths=[0.0],
    )

    assert len(df) == 1
    assert df.loc[0, "A_hat"] == 12.0
    assert df.loc[0, "source_model"] == "functional_form"
    assert df.loc[0, "toy_hist"] == "fShiftSigPowTail_toy_0"


def test_streaming_skips_writing_toy_csv_when_disabled(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(
        output_dir=str(tmp_path),
        inj_write_toy_csv=False,
        inj_stream_aggregate=True,
        inj_aggregate_every=2,
        inj_n_workers=2,
        inj_parallel_backend="threading",
    )

    df_sum = run_injection_extraction_streaming(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[0.0, 1.0],
        n_toys=4,
    )

    assert len(df_sum) == 2
    assert not (tmp_path / "injection_extraction" / "inj_extract_toys_2015.csv").exists()


def test_streaming_is_deterministic_across_worker_counts(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg1 = Config(
        output_dir=str(tmp_path / "one"),
        inj_write_toy_csv=False,
        inj_stream_aggregate=True,
        inj_aggregate_every=3,
        inj_n_workers=1,
        inj_parallel_backend="threading",
    )
    cfg2 = Config(
        output_dir=str(tmp_path / "two"),
        inj_write_toy_csv=False,
        inj_stream_aggregate=True,
        inj_aggregate_every=3,
        inj_n_workers=3,
        inj_parallel_backend="threading",
    )

    out1 = run_injection_extraction_streaming(
        _make_dataset(),
        cfg1,
        masses=[0.05],
        strengths=[1.0],
        n_toys=7,
        seed=11,
    )
    out2 = run_injection_extraction_streaming(
        _make_dataset(),
        cfg2,
        masses=[0.05],
        strengths=[1.0],
        n_toys=7,
        seed=11,
    )

    pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True), check_dtype=False)


def test_streaming_summary_schema_matches_legacy_summary(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(output_dir=str(tmp_path), inj_write_toy_csv=False, inj_parallel_backend="threading")

    legacy = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[1.0],
        n_toys=5,
        seed=9,
    )
    legacy_sum = summarize_injection_grid(legacy)

    stream_sum = run_injection_extraction_streaming(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[1.0],
        n_toys=5,
        seed=9,
        n_workers=2,
        parallel_backend="threading",
        aggregate_every=2,
    )

    assert set(stream_sum.columns) == set(legacy_sum.columns)


def test_refit_failure_is_flagged_instead_of_hidden_fallback(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    import hps_gpr.injection as inj

    def raise_fit(*args, **kwargs):
        raise RuntimeError("forced refit failure")

    monkeypatch.setattr(inj, "fit_gpr", raise_fit)
    cfg = Config(
        output_dir=str(tmp_path),
        inj_write_toy_csv=False,
        inj_refit_gp_on_toy=True,
        inj_refit_fail_on_error=False,
    )

    df = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[1.0],
        n_toys=1,
        refit_gp_on_toy=True,
    )

    row = df.iloc[0]
    assert bool(row["refit_fallback_used"])
    assert float(row["refit_ok"]) == 0.0
    assert "RuntimeError: forced refit failure" in row["refit_error"]
    assert "kernel_ls_res_lower_factor" in df.columns
    assert "ls_lo" in df.columns
    assert "n_train" in df.columns

    summary = summarize_injection_grid(df).iloc[0]
    assert float(summary["refit_fallback_rate"]) == 1.0
    assert float(summary["refit_ok_rate"]) == 0.0


def test_refit_failure_can_be_promoted_to_exception(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    import hps_gpr.injection as inj

    def raise_fit(*args, **kwargs):
        raise RuntimeError("forced refit failure")

    monkeypatch.setattr(inj, "fit_gpr", raise_fit)
    cfg = Config(
        output_dir=str(tmp_path),
        inj_write_toy_csv=False,
        inj_refit_gp_on_toy=True,
        inj_refit_fail_on_error=True,
    )

    with pytest.raises(RuntimeError, match="failed for 2015"):
        run_injection_extraction_toys(
            _make_dataset(),
            cfg,
            masses=[0.05],
            strengths=[1.0],
            n_toys=1,
            refit_gp_on_toy=True,
        )


def test_summarize_injection_grid_computes_delta_z_minus_pull_from_delta_z():
    toys = pd.DataFrame(
        [
            {
                "dataset": "2015",
                "mass_GeV": 0.05,
                "strength": 10.0,
                "inj_nsigma": 2.0,
                "sigmaA_ref": 5.0,
                "A_hat": 12.0,
                "sigma_A": 2.0,
                "pull_param": 1.0,
                "Zhat": 6.0,
                "success": True,
            },
            {
                "dataset": "2015",
                "mass_GeV": 0.05,
                "strength": 10.0,
                "inj_nsigma": 2.0,
                "sigmaA_ref": 5.0,
                "A_hat": 8.0,
                "sigma_A": 2.0,
                "pull_param": -1.0,
                "Zhat": 4.0,
                "success": True,
            },
        ]
    )

    out = summarize_injection_grid(toys)
    row = out.iloc[0]
    expected = (float(row["Zhat_mean"]) - float(row["inj_nsigma"])) - float(row["pull_mean"])

    assert float(row["delta_z_minus_pull"]) == expected


def test_collapse_fragmented_summary_groups_one_toy_rows_by_injected_sigma():
    fragments = pd.DataFrame(
        [
            {
                "dataset": "2015",
                "mass_GeV": 0.05,
                "strength": 10.0,
                "n_toys": 1,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 10.0,
                "A_hat_mean": 8.0,
                "A_hat_std": np.nan,
                "sigma_A_mean": 5.0,
                "pull_mean": -0.4,
                "pull_std": np.nan,
                "cov_1sigma": 1.0,
                "cov_2sigma": 1.0,
                "Zhat_mean": 1.6,
                "Zhat_q16": 1.6,
                "Zhat_q84": 1.6,
                "success_rate": 1.0,
            },
            {
                "dataset": "2015",
                "mass_GeV": 0.05,
                "strength": 12.0,
                "n_toys": 1,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 12.0,
                "A_hat_mean": 6.0,
                "A_hat_std": np.nan,
                "sigma_A_mean": 6.0,
                "pull_mean": -1.0,
                "pull_std": np.nan,
                "cov_1sigma": 0.0,
                "cov_2sigma": 1.0,
                "Zhat_mean": 1.0,
                "Zhat_q16": 1.0,
                "Zhat_q84": 1.0,
                "success_rate": 1.0,
            },
        ]
    )

    out = collapse_fragmented_injection_summary(fragments)

    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["n_toys"]) == 2
    assert float(row["inj_nsigma"]) == 1.0
    assert float(row["strength"]) == 11.0
    assert float(row["A_hat_mean"]) == 7.0
    assert float(row["pull_mean"]) == -0.7
    assert float(row["Zhat_mean"]) == 1.3
    assert float(row["delta_z_minus_pull"]) == pytest.approx(1.3 - 1.0 + 0.7)
    assert float(row["cov_1sigma"]) == 0.5


def test_collapse_fragmented_summary_canonicalizes_mass_keys():
    df = pd.DataFrame(
        [
            {
                "dataset": "2015",
                "mass_GeV": 0.105,
                "strength": 0.0,
                "n_toys": 100,
                "inj_nsigma": 0.0,
                "pull_mean": 0.0,
                "pull_std": 1.0,
                "Zhat_mean": 0.0,
            },
            {
                "dataset": "2015",
                "mass_GeV": 0.10500000000000002,
                "strength": 1.0,
                "n_toys": 1,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 1.0,
                "A_hat_mean": 0.5,
                "sigma_A_mean": 1.0,
                "pull_mean": -0.5,
                "Zhat_mean": 0.5,
            },
            {
                "dataset": "2015",
                "mass_GeV": 0.10500000000000002,
                "strength": 1.0,
                "n_toys": 1,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 1.0,
                "A_hat_mean": 1.5,
                "sigma_A_mean": 1.0,
                "pull_mean": 0.5,
                "Zhat_mean": 1.5,
            },
        ]
    )

    out = collapse_fragmented_injection_summary(df)

    assert sorted(f"{m:.12f}" for m in out["mass_GeV"].unique()) == ["0.105000000000"]


def test_collapse_fragmented_summary_does_not_leak_internal_keys_for_regular_summary():
    df = pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.04500000000000001, "strength": 0.0, "n_toys": 100, "inj_nsigma": 0.0},
            {"dataset": "2015", "mass_GeV": 0.04500000000000001, "strength": 1.0, "n_toys": 100, "inj_nsigma": 1.0},
        ]
    )

    out = collapse_fragmented_injection_summary(df)

    assert "_mass_key" not in out.columns
    assert "_inj_key" not in out.columns
    assert sorted(f"{m:.12f}" for m in out["mass_GeV"].unique()) == ["0.045000000000"]


def test_streaming_combined_writes_compact_summaries_without_toy_csv(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(
        output_dir=str(tmp_path),
        inj_write_toy_csv=False,
        inj_stream_aggregate=True,
        inj_aggregate_every=2,
        inj_n_workers=2,
        inj_parallel_backend="threading",
        inj_strength_mode="sigmaA",
        inj_sigma_multipliers=[1.0],
    )

    out_by_ds, out_comb = run_injection_extraction_streaming_combined(
        {"2015": _make_dataset(), "2016": _make_dataset_2016()},
        cfg,
        masses=[0.05],
        strengths=[1.0],
        n_toys=4,
        seed=21,
    )

    assert "2015" in out_by_ds and "2016" in out_by_ds
    assert not out_by_ds["2015"].empty
    assert not out_by_ds["2016"].empty
    assert not out_comb.empty
    assert not (tmp_path / "injection_extraction" / "inj_extract_toys_2015.csv").exists()
    assert not (tmp_path / "injection_extraction" / "inj_extract_toys_2016.csv").exists()
    assert not (tmp_path / "injection_extraction" / "inj_extract_toys_combined.csv").exists()


def test_slurm_gen_inject_cli_infers_cpus_per_task(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "submit_inject.slurm"
    config_path.write_text("output_dir: outputs/test\n")

    def fake_load_config(path):
        assert str(path) == str(config_path)
        return Config(
            output_dir=str(tmp_path / "out"),
            inj_n_workers=3,
            inj_threads_per_worker=2,
        )

    import hps_gpr.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "slurm-gen-inject",
            "--config", str(config_path),
            "--datasets", "2015",
            "--masses", "0.04",
            "--strengths", "1,2",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CPUs per task: 6" in result.output
    assert "#SBATCH --cpus-per-task=6" in output_path.read_text()


def test_slurm_gen_inject_cli_explicit_cpus_override_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "submit_inject_override.slurm"
    config_path.write_text("output_dir: outputs/test\n")

    def fake_load_config(path):
        return Config(
            output_dir=str(tmp_path / "out"),
            inj_n_workers=5,
            inj_threads_per_worker=4,
        )

    import hps_gpr.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "slurm-gen-inject",
            "--config", str(config_path),
            "--datasets", "2015",
            "--masses", "0.04",
            "--strengths", "1",
            "--cpus-per-task", "7",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CPUs per task: 7" in result.output
    assert "#SBATCH --cpus-per-task=7" in output_path.read_text()
