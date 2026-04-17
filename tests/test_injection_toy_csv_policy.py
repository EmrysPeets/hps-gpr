from types import SimpleNamespace

from click.testing import CliRunner
import numpy as np
import pandas as pd

from hps_gpr.cli import main
from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.injection import (
    run_injection_extraction_toys,
    run_injection_extraction_streaming,
    run_injection_extraction_streaming_combined,
    summarize_injection_grid,
)


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
