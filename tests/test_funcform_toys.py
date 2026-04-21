import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from click.testing import CliRunner
import hist
import numpy as np
import pandas as pd
import pytest
import uproot

from hps_gpr.cli import main
from hps_gpr.config import Config, load_config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.funcform_toys import (
    FuncFormToySpec,
    discover_funcform_toys,
    load_funcform_toy_hist,
    merge_toy_scan_results,
    run_funcform_toy_scans,
)
from hps_gpr.gpr import compute_kernel_ls_bounds
from hps_gpr.io import _build_model
from hps_gpr.slurm import generate_toy_scan_slurm_scripts


def _make_hist(values, lo=0.0, hi=0.3):
    h = hist.Hist(
        hist.axis.Regular(len(values), lo, hi, label="Mass / GeV"),
        storage=hist.storage.Weight(),
    )
    view = h.view()
    view.value[...] = np.asarray(values, dtype=float)
    view.variance[...] = np.asarray(values, dtype=float)
    return h


def _write_toy_root(path: Path):
    with uproot.recreate(path) as f:
        f["primary/primary_toy_0"] = _make_hist([10, 9, 8, 7, 6, 5])
        f["primary/primary_toy_1"] = _make_hist([12, 11, 9, 8, 6, 4])
        f["alt/alt_toy_0"] = _make_hist([7, 7, 7, 7, 7, 7])


def test_discover_funcform_toys_supports_container_pattern_and_format(tmp_path):
    root_path = tmp_path / "funcform.root"
    _write_toy_root(root_path)

    specs = discover_funcform_toys(
        str(root_path),
        container="primary",
        toy_pattern="primary_toy_*",
    )
    assert [s.toy_name for s in specs] == ["primary_toy_0", "primary_toy_1"]
    assert [s.function_tag for s in specs] == ["primary", "primary"]
    assert [s.toy_index for s in specs] == [0, 1]

    specs_fmt = discover_funcform_toys(
        str(root_path),
        container="primary",
        toy_name_fmt="primary_toy_{i}",
        toy_indices=[1],
    )
    assert len(specs_fmt) == 1
    assert specs_fmt[0].toy_name == "primary_toy_1"

    htoy = load_funcform_toy_hist(str(root_path), container="primary", toy_name="primary_toy_0")
    assert np.allclose(htoy.values(), np.array([10, 9, 8, 7, 6, 5], dtype=float))


def test_discover_funcform_toys_preserves_actual_indices_and_fallbacks(tmp_path):
    root_path = tmp_path / "funcform_indices.root"
    with uproot.recreate(root_path) as f:
        f["primary/primary_toy_7"] = _make_hist([10, 10, 10, 10])
        f["primary/primary_toy_0009"] = _make_hist([9, 9, 9, 9])
        f["misc/custom_shape"] = _make_hist([8, 8, 8, 8])

    specs = discover_funcform_toys(
        str(root_path),
        container="primary",
        toy_pattern="primary_toy_*",
    )
    assert [s.toy_name for s in specs] == ["primary_toy_7", "primary_toy_0009"]
    assert [s.toy_index for s in specs] == [7, 9]

    specs_fmt = discover_funcform_toys(
        str(root_path),
        container="primary",
        toy_name_fmt="primary_toy_{i}",
        toy_indices=[9, 7],
    )
    assert [s.toy_name for s in specs_fmt] == ["primary_toy_9", "primary_toy_7"]
    assert [s.toy_index for s in specs_fmt] == [9, 7]

    fallback_specs = discover_funcform_toys(
        str(root_path),
        container="misc",
        toy_pattern="custom_*",
    )
    assert [s.toy_name for s in fallback_specs] == ["custom_shape"]
    assert [s.toy_index for s in fallback_specs] == [0]


def test_build_model_accepts_hist_override(monkeypatch):
    toy_hist = _make_hist([5, 6, 7, 8, 9, 10], lo=0.02, hi=0.14)
    ds = DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.02,
        m_high=0.13,
        sigma_coeffs=[0.001],
        frad_coeffs=[0.1],
        hist_override=toy_hist,
    )
    cfg = Config(neighborhood_rebin=1)

    import hps_gpr.io as io

    monkeypatch.setattr(
        io,
        "_gp_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_gp_model should not be used")),
    )

    model = _build_model(ds, blind=(0.05, 0.06), rebin=1, config=cfg, mass=0.055)

    assert np.allclose(model.histogram.axes[0].edges, toy_hist.axes[0].edges)
    assert np.allclose(model.histogram.values(), toy_hist.values())


def test_estimate_background_uses_mean_only_full_prediction(monkeypatch):
    toy_hist = _make_hist([5, 6, 7, 8], lo=0.02, hi=0.06)
    ds = DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.02,
        m_high=0.06,
        sigma_coeffs=[0.001],
        frad_coeffs=[0.1],
        hist_override=toy_hist,
    )
    cfg = Config(neighborhood_rebin=1, n_restarts=0)

    import hps_gpr.io as io

    fake_model = SimpleNamespace(histogram=toy_hist)
    fake_gpr = SimpleNamespace(kernel_=None, kernel=None, log_marginal_likelihood_value_=0.5)
    calls = {"blind": [], "full": []}

    monkeypatch.setattr(io, "_build_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(io, "fit_gpr", lambda *args, **kwargs: fake_gpr)
    monkeypatch.setattr(
        io,
        "predict_counts_from_log_gpr",
        lambda gpr, X_query, config: (
            calls["blind"].append(np.asarray(X_query, float).copy()) or np.full(len(np.asarray(X_query)), 11.0),
            np.eye(len(np.asarray(X_query)), dtype=float),
        ),
    )
    monkeypatch.setattr(
        io,
        "predict_counts_mean_from_log_gpr",
        lambda gpr, X_query, config: (
            calls["full"].append(np.asarray(X_query, float).copy()) or np.full(len(np.asarray(X_query)), 22.0)
        ),
    )
    monkeypatch.setattr(
        io,
        "compute_kernel_ls_bounds",
        lambda *args, **kwargs: {"ls_lo": 0.1, "ls_hi": 0.2, "ls_init": 0.15, "sigma_x": 0.01},
    )
    monkeypatch.setattr(io, "_extract_rbf_bounds_and_scale", lambda kernel: (0.1, 0.2, 0.15))

    pred = io.estimate_background_for_dataset(ds, 0.045, cfg)

    assert len(calls["blind"]) == 1
    assert len(calls["full"]) == 1
    assert pred.cov.shape[0] == pred.cov.shape[1]
    assert np.allclose(pred.mu, 11.0)
    assert np.allclose(pred.mu_full, 22.0)


def test_resolution_scaled_local_bounds_vary_with_mass_when_cap_disabled():
    ds = DatasetConfig(
        key="2021",
        label="HPS 2021",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.03,
        m_high=0.25,
        sigma_coeffs=[0.0014786, -0.0011, 0.0687],
        frad_coeffs=[0.1],
    )
    cfg = Config(
        kernel_ls_policy="resolution_scaled_local",
        kernel_ls_res_upper_factor=8.0,
        kernel_ls_res_lower_factor=0.5,
        kernel_ls_local_hi_floor_mode="none",
        kernel_ls_local_hi_cap_xrange_frac=None,
        pre_log=True,
    )

    info_lo = compute_kernel_ls_bounds(ds, cfg, mass=0.040)
    info_hi = compute_kernel_ls_bounds(ds, cfg, mass=0.044)

    assert info_lo["policy_used"] == "resolution_scaled_local"
    assert info_hi["policy_used"] == "resolution_scaled_local"
    assert info_lo["ls_hi"] != pytest.approx(info_hi["ls_hi"])
    assert info_lo["ls_hi"] > info_hi["ls_hi"]


def test_merge_toy_scan_results_preserves_toy_identity(tmp_path):
    base = tmp_path / "toy_scans" / "2015"
    toy0 = base / "toy_0000"
    toy1 = base / "toy_0001"
    toy0.mkdir(parents=True)
    toy1.mkdir(parents=True)

    meta0 = {
        "dataset": "2015",
        "toy_index": 0,
        "toy_name": "primary_toy_0",
        "function_tag": "primary",
        "source_root": "/tmp/funcform.root",
        "container": "primary",
    }
    meta1 = {
        "dataset": "2015",
        "toy_index": 1,
        "toy_name": "primary_toy_1",
        "function_tag": "primary",
        "source_root": "/tmp/funcform.root",
        "container": "primary",
    }
    (toy0 / "toy_metadata.json").write_text(json.dumps(meta0))
    (toy1 / "toy_metadata.json").write_text(json.dumps(meta1))

    pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.040, "Z_analytic": 1.0, "p0_analytic": 0.2, "extract_success": True},
            {"dataset": "2015", "mass_GeV": 0.050, "Z_analytic": 2.5, "p0_analytic": 0.01, "extract_success": True},
        ]
    ).to_csv(toy0 / "results_single.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.040, "Z_analytic": np.nan, "p0_analytic": np.nan, "extract_success": False},
            {"dataset": "2015", "mass_GeV": 0.050, "Z_analytic": 0.5, "p0_analytic": 0.5, "extract_success": True},
        ]
    ).to_csv(toy1 / "results_single.csv", index=False)

    merged, summary = merge_toy_scan_results(str(tmp_path), output_dir=str(tmp_path / "merged"))

    assert len(merged) == 4
    assert set(["toy_index", "toy_hist", "function_tag", "source_root"]).issubset(set(merged.columns))
    assert merged.loc[0, "toy_hist"] == "primary_toy_0"
    assert len(summary) == 2

    s0 = summary[summary["toy_index"] == 0].iloc[0]
    assert s0["toy_hist"] == "primary_toy_0"
    assert np.isclose(s0["max_Z_analytic"], 2.5)
    assert np.isclose(s0["mass_at_max_Z"], 0.05)
    assert np.isclose(s0["min_p0_analytic"], 0.01)

    s1 = summary[summary["toy_index"] == 1].iloc[0]
    assert s1["n_fail"] == 1


def test_merge_toy_scan_results_recurses_over_job_directories(tmp_path):
    toy_dir = tmp_path / "jobs" / "primary_toy_7" / "toy_scans" / "2015" / "toy_0007"
    toy_dir.mkdir(parents=True)
    (toy_dir / "toy_metadata.json").write_text(json.dumps({
        "dataset": "2015",
        "toy_index": 7,
        "toy_name": "primary_toy_7",
        "function_tag": "primary",
        "source_root": "/tmp/funcform.root",
        "container": "primary",
    }))
    pd.DataFrame(
        [{"dataset": "2015", "mass_GeV": 0.040, "Z_analytic": 1.2, "p0_analytic": 0.1, "extract_success": True}]
    ).to_csv(toy_dir / "results_single.csv", index=False)

    merged, summary = merge_toy_scan_results(
        str(tmp_path / "jobs"),
        output_dir=str(tmp_path / "merged"),
    )

    assert len(merged) == 1
    assert int(summary.loc[0, "toy_index"]) == 7
    assert summary.loc[0, "toy_hist"] == "primary_toy_7"


def test_merge_toy_scan_results_reports_inventory_on_empty_outputs(tmp_path):
    toy_dir = tmp_path / "jobs" / "primary_toy_7" / "toy_scans" / "2015" / "toy_0007"
    toy_dir.mkdir(parents=True)
    (toy_dir / "toy_metadata.json").write_text(json.dumps({
        "dataset": "2015",
        "toy_index": 7,
        "toy_name": "primary_toy_7",
        "function_tag": "primary",
        "source_root": "/tmp/funcform.root",
        "container": "primary",
    }))

    with np.testing.assert_raises_regex(
        FileNotFoundError,
        r"toy_dirs=1, toy_metadata\.json=1, results_single\.csv=0, results_combined\.csv=0",
    ):
        merge_toy_scan_results(str(tmp_path / "jobs"), output_dir=str(tmp_path / "merged"))


def test_toy_scan_merge_cli_writes_validation_plots(tmp_path):
    toy_dir = tmp_path / "jobs" / "primary_toy_7" / "toy_scans" / "2015" / "toy_0007"
    toy_dir.mkdir(parents=True)
    (toy_dir / "toy_metadata.json").write_text(json.dumps({
        "dataset": "2015",
        "toy_index": 7,
        "toy_name": "primary_toy_7",
        "function_tag": "primary",
        "source_root": "/tmp/funcform.root",
        "container": "primary",
    }))
    pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.040, "Z_analytic": 1.2, "p0_analytic": 0.1, "eps2_up": 1.0e-5, "extract_success": True},
            {"dataset": "2015", "mass_GeV": 0.050, "Z_analytic": 2.2, "p0_analytic": 0.02, "eps2_up": 2.0e-5, "extract_success": True},
        ]
    ).to_csv(toy_dir / "results_single.csv", index=False)

    outdir = tmp_path / "merged"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "toy-scan-merge",
            "--input-dir", str(tmp_path / "jobs"),
            "--output-dir", str(outdir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "toy_scan_validation_2015_primary_local_significance" in result.output
    assert "toy_scan_validation_2015_primary_upper_limits" in result.output
    assert "toy_scan_validation_2015_primary_summary" in result.output

    for stem in [
        outdir / "toy_scan_validation_2015_primary_local_significance",
        outdir / "toy_scan_validation_2015_primary_upper_limits",
        outdir / "toy_scan_validation_2015_primary_summary",
    ]:
        assert (stem.with_suffix(".png")).exists()
        assert (stem.with_suffix(".pdf")).exists()


def test_run_funcform_toy_scans_applies_toy_runtime_and_output_overrides(monkeypatch, tmp_path):
    root_path = tmp_path / "funcform.root"
    _write_toy_root(root_path)
    base_output_dir = tmp_path / "toy_out"

    ds = DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.02,
        m_high=0.13,
        sigma_coeffs=[0.001],
        frad_coeffs=[0.1],
    )
    cfg = Config(
        output_dir=str(base_output_dir),
        scan_parallel=True,
        scan_n_workers=5,
        scan_parallel_backend="loky",
        scan_threads_per_worker=2,
        save_plots=True,
        save_fit_json=True,
        save_per_mass_folders=True,
        toy_scan_parallel=False,
        toy_scan_n_workers=1,
        toy_scan_parallel_backend="threading",
        toy_scan_threads_per_worker=1,
        toy_scan_save_plots=False,
        toy_scan_save_fit_json=False,
        toy_scan_save_per_mass_folders=False,
    )
    spec = FuncFormToySpec(
        source_root=str(root_path),
        container="primary",
        function_tag="primary",
        toy_name="primary_toy_0",
        toy_index=0,
    )

    captured = {}

    def fake_run_scan(datasets, toy_cfg, mass_min=None, mass_max=None):
        captured["dataset_keys"] = list(datasets.keys())
        captured["scan_parallel"] = toy_cfg.scan_parallel
        captured["scan_n_workers"] = toy_cfg.scan_n_workers
        captured["scan_parallel_backend"] = toy_cfg.scan_parallel_backend
        captured["scan_threads_per_worker"] = toy_cfg.scan_threads_per_worker
        captured["save_plots"] = toy_cfg.save_plots
        captured["save_fit_json"] = toy_cfg.save_fit_json
        captured["save_per_mass_folders"] = toy_cfg.save_per_mass_folders
        captured["output_dir"] = toy_cfg.output_dir
        return (
            pd.DataFrame([{"dataset": "2015", "mass_GeV": 0.04, "Z_analytic": 1.2, "p0_analytic": 0.1, "extract_success": True}]),
            pd.DataFrame(),
        )

    import hps_gpr.scan as scan_mod

    monkeypatch.setattr(scan_mod, "run_scan", fake_run_scan)

    written = run_funcform_toy_scans(
        ds,
        cfg,
        [spec],
        base_output_dir=str(base_output_dir),
    )

    assert written == [str(base_output_dir / "toy_scans" / "2015" / "toy_0000")]
    assert captured["dataset_keys"] == ["2015"]
    assert captured["scan_parallel"] is False
    assert captured["scan_n_workers"] == 1
    assert captured["scan_parallel_backend"] == "threading"
    assert captured["scan_threads_per_worker"] == 1
    assert captured["save_plots"] is False
    assert captured["save_fit_json"] is False
    assert captured["save_per_mass_folders"] is False
    assert captured["output_dir"].endswith("toy_scans/2015/toy_0000")


def test_toy_scan_cli_uses_toy_defaults_and_prints_effective_settings(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    toy_root = tmp_path / "toys.root"
    config_path.write_text("output_dir: outputs/test\n")
    toy_root.write_text("placeholder\n")

    captured = []

    def fake_load_config(path):
        assert str(path) == str(config_path)
        return Config(
            output_dir=str(tmp_path / "out"),
            save_plots=True,
            save_fit_json=True,
            save_per_mass_folders=True,
            toy_scan_parallel=False,
            toy_scan_n_workers=1,
            toy_scan_parallel_backend="threading",
            toy_scan_threads_per_worker=1,
            toy_scan_save_plots=False,
            toy_scan_save_fit_json=False,
            toy_scan_save_per_mass_folders=False,
        )

    def fake_make_datasets(cfg):
        return {
            "2015": DatasetConfig(
                key="2015",
                label="HPS 2015",
                root_path="unused.root",
                hist_name="unused",
                m_low=0.02,
                m_high=0.13,
                sigma_coeffs=[0.001],
                frad_coeffs=[0.1],
            )
        }

    def fake_discover(*args, **kwargs):
        return [
            FuncFormToySpec(
                source_root=str(toy_root),
                container="primary",
                function_tag="primary",
                toy_name="primary_toy_7",
                toy_index=7,
            )
        ]

    def fake_run(*args, **kwargs):
        captured.append(kwargs)
        return [str(Path(kwargs["base_output_dir"]) / "toy_scans" / "2015" / "toy_0007")]

    import hps_gpr.config as cfg_mod
    import hps_gpr.dataset as dataset_mod
    import hps_gpr.funcform_toys as funcform_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)
    monkeypatch.setattr(dataset_mod, "make_datasets", fake_make_datasets)
    monkeypatch.setattr(funcform_mod, "discover_funcform_toys", fake_discover)
    monkeypatch.setattr(funcform_mod, "run_funcform_toy_scans", fake_run)

    runner = CliRunner()
    result_default = runner.invoke(
        main,
        [
            "toy-scan",
            "--config", str(config_path),
            "--dataset", "2015",
            "--toy-root", str(toy_root),
            "--container", "primary",
            "--toy-pattern", "primary_toy_*",
        ],
    )
    assert result_default.exit_code == 0, result_default.output
    assert captured[0]["save_plots"] is False
    assert captured[0]["save_fit_json"] is False
    assert captured[0]["save_per_mass_folders"] is False
    assert captured[0]["scan_parallel"] is False
    assert captured[0]["scan_n_workers"] == 1
    assert captured[0]["scan_parallel_backend"] == "threading"
    assert captured[0]["scan_threads_per_worker"] == 1
    assert "Mass hypotheses per toy: 111" in result_default.output
    assert "scan_parallel=False" in result_default.output
    assert "scan_backend=threading" in result_default.output


def test_toy_scan_cli_accepts_explicit_runtime_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    toy_root = tmp_path / "toys.root"
    config_path.write_text("output_dir: outputs/test\n")
    toy_root.write_text("placeholder\n")

    captured = []

    def fake_load_config(path):
        assert str(path) == str(config_path)
        return Config(
            output_dir=str(tmp_path / "out"),
            toy_scan_parallel=False,
            toy_scan_n_workers=1,
            toy_scan_parallel_backend="threading",
            toy_scan_threads_per_worker=1,
            toy_scan_save_plots=False,
            toy_scan_save_fit_json=False,
            toy_scan_save_per_mass_folders=False,
        )

    def fake_make_datasets(cfg):
        return {
            "2015": DatasetConfig(
                key="2015",
                label="HPS 2015",
                root_path="unused.root",
                hist_name="unused",
                m_low=0.02,
                m_high=0.13,
                sigma_coeffs=[0.001],
                frad_coeffs=[0.1],
            )
        }

    def fake_discover(*args, **kwargs):
        return [
            FuncFormToySpec(
                source_root=str(toy_root),
                container="primary",
                function_tag="primary",
                toy_name="primary_toy_7",
                toy_index=7,
            )
        ]

    def fake_run(*args, **kwargs):
        captured.append(kwargs)
        return [str(Path(kwargs["base_output_dir"]) / "toy_scans" / "2015" / "toy_0007")]

    import hps_gpr.config as cfg_mod
    import hps_gpr.dataset as dataset_mod
    import hps_gpr.funcform_toys as funcform_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)
    monkeypatch.setattr(dataset_mod, "make_datasets", fake_make_datasets)
    monkeypatch.setattr(funcform_mod, "discover_funcform_toys", fake_discover)
    monkeypatch.setattr(funcform_mod, "run_funcform_toy_scans", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "toy-scan",
            "--config", str(config_path),
            "--dataset", "2015",
            "--toy-root", str(toy_root),
            "--container", "primary",
            "--toy-pattern", "primary_toy_*",
            "--save-plots",
            "--save-fit-json",
            "--save-per-mass-folders",
            "--scan-parallel",
            "--scan-n-workers", "4",
            "--scan-backend", "loky",
            "--scan-threads-per-worker", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0]["save_plots"] is True
    assert captured[0]["save_fit_json"] is True
    assert captured[0]["save_per_mass_folders"] is True
    assert captured[0]["scan_parallel"] is True
    assert captured[0]["scan_n_workers"] == 4
    assert captured[0]["scan_parallel_backend"] == "loky"
    assert captured[0]["scan_threads_per_worker"] == 3


def test_generate_toy_scan_slurm_scripts_writes_expected_commands(tmp_path):
    job = tmp_path / "submit_toy_scan.slurm"

    job_script, submit_script, n_jobs = generate_toy_scan_slurm_scripts(
        config_path="config_example.yaml",
        output_path=str(job),
        dataset="2015",
        toy_root="outputs/funcform_toys/funcform_2015_dataset_mod_toys.root",
        toy_names=["primary_toy_0", "primary_toy_1"],
        toy_indices=[0, 1],
        output_root="outputs/funcform_toys",
        container="primary",
        cpus_per_task=10,
        extra_sbatch=["--account=testacct", "--qos=testqos"],
    )

    assert n_jobs == 2
    job_text = Path(job_script).read_text()
    submit_text = Path(submit_script).read_text()
    assert "python -m hps_gpr.cli toy-scan" in job_text
    assert "#SBATCH --account=testacct" in job_text
    assert "#SBATCH --qos=testqos" in job_text
    assert "#SBATCH --cpus-per-task=10" in job_text
    assert 'cd "${REPO_ROOT}"' in job_text
    assert 'source "${REPO_ROOT}/startup.sh"' in job_text
    assert 'JOB_OUTDIR="${BASE_OUTPUT_DIR}/jobs/${TOY_DIR_NAME}"' in job_text
    assert 'export PYTHONUNBUFFERED=1' in job_text
    assert 'echo "[toy-scan] dataset=${TOY_DATASET} toy=${TOY_NAME} output=${JOB_OUTDIR}"' in job_text
    assert 'mass-hypothesis progress will appear below' in job_text
    assert '--output-dir "${JOB_OUTDIR}"' in job_text
    assert '--toy-root "${TOY_ROOT}"' in job_text
    assert 'CMD+=(--toy-name-fmt "${TOY_NAME}")' in job_text
    assert 'CMD+=(--toy-index "${TOY_INDEX}")' in job_text
    assert 'CMD+=(--toy-pattern "${TOY_NAME}")' in job_text
    assert 'CMD+=(--container "${TOY_CONTAINER}")' in job_text
    assert 'SBATCH_ARGS=("$@")' in submit_text
    assert 'sbatch "${SBATCH_ARGS[@]}" --export=ALL,' in submit_text
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in submit_text
    assert 'cd "${SCRIPT_DIR}"' in submit_text
    assert "TOY_NAME=primary_toy_1" in submit_text
    assert "TOY_INDEX=1" in submit_text
    assert "TOY_DIR_NAME=primary_toy_1" in submit_text
    assert 'REPO_ROOT="' in submit_text


def test_slurm_gen_toy_scan_cli_infers_cpus_per_task(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    toy_root = tmp_path / "toys.root"
    output_path = tmp_path / "submit_toy_scan.slurm"
    config_path.write_text("output_dir: outputs/test\n")
    toy_root.write_text("placeholder\n")

    def fake_load_config(path):
        assert str(path) == str(config_path)
        return Config(
            output_dir=str(tmp_path / "out"),
            scan_parallel=True,
            scan_n_workers=3,
            scan_threads_per_worker=2,
            toy_scan_parallel=False,
            toy_scan_n_workers=4,
            toy_scan_threads_per_worker=3,
        )

    def fake_discover(*args, **kwargs):
        return [
            FuncFormToySpec(
                source_root=str(toy_root),
                container="primary",
                function_tag="primary",
                toy_name="primary_toy_7",
                toy_index=7,
            )
        ]

    import hps_gpr.config as cfg_mod
    import hps_gpr.funcform_toys as funcform_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)
    monkeypatch.setattr(funcform_mod, "discover_funcform_toys", fake_discover)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "slurm-gen-toy-scan",
            "--config", str(config_path),
            "--dataset", "2015",
            "--toy-root", str(toy_root),
            "--container", "primary",
            "--toy-pattern", "primary_toy_*",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CPUs per task: 1" in result.output
    job_text = output_path.read_text()
    assert "#SBATCH --cpus-per-task=1" in job_text


def test_slurm_gen_cli_infers_scan_cpus_per_task(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "submit_scan.slurm"
    config_path.write_text("output_dir: outputs/test\n")

    def fake_load_config(path):
        assert str(path) == str(config_path)
        return Config(
            output_dir=str(tmp_path / "out"),
            scan_parallel=True,
            scan_n_workers=3,
            scan_threads_per_worker=2,
        )

    import hps_gpr.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "slurm-gen",
            "--config", str(config_path),
            "--n-jobs", "4",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CPUs per task: 6" in result.output
    assert "#SBATCH --cpus-per-task=6" in output_path.read_text()


def test_slurm_gen_cli_explicit_scan_cpus_override_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "submit_scan_override.slurm"
    config_path.write_text("output_dir: outputs/test\n")

    def fake_load_config(path):
        return Config(
            output_dir=str(tmp_path / "out"),
            scan_parallel=True,
            scan_n_workers=5,
            scan_threads_per_worker=4,
        )

    import hps_gpr.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "slurm-gen",
            "--config", str(config_path),
            "--n-jobs", "4",
            "--cpus-per-task", "7",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CPUs per task: 7" in result.output
    assert "#SBATCH --cpus-per-task=7" in output_path.read_text()


@pytest.mark.parametrize(
    "config_path",
    [
        "config_2016_10pct_10k.yaml",
        "study_configs/config_2016_10pct_blind1p64_90CL_10k_injection.yaml",
        "study_configs/config_2016_10pct_blind1p64_95CL_10k_injection.yaml",
        "study_configs/config_2016_10pct_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml",
        "study_configs/config_2016_10pct_blind1p96_90CL_10k_injection.yaml",
        "study_configs/config_2016_10pct_blind1p96_95CL_10k_injection.yaml",
    ],
)
def test_2016_only_configs_use_2016_dataset_selectors(config_path):
    cfg = load_config(config_path)
    assert cfg.enable_2015 is False
    assert cfg.enable_2016 is True
    assert cfg.inj_dataset_key == "2016"
    assert cfg.run_limit_bands_on == "2016"


def test_local_resolution_configs_do_not_set_constant_hi_cap():
    repo_root = Path(__file__).resolve().parents[1]
    try:
        tracked = subprocess.check_output(
            "git ls-files '*.yaml' 'study_configs/*.yaml'",
            cwd=repo_root,
            shell=True,
            text=True,
        ).splitlines()
    except Exception as exc:
        pytest.skip(f"git ls-files unavailable: {exc}")

    offenders = []
    for rel_path in tracked:
        text = (repo_root / rel_path).read_text()
        if "kernel_ls_policy: resolution_scaled_local" not in text:
            continue
        if "kernel_ls_local_hi_cap_xrange_frac:" in text:
            offenders.append(rel_path)

    assert offenders == []
