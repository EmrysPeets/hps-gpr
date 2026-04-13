import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import hist
import numpy as np
import pandas as pd
import uproot

from hps_gpr.cli import main
from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.funcform_toys import (
    FuncFormToySpec,
    discover_funcform_toys,
    load_funcform_toy_hist,
    merge_toy_scan_results,
)
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

    calls = []

    def fake_gp_model(h, kernel, **kwargs):
        hist_obj = h
        if kwargs.get("modify_histogram") is not None:
            hist_obj = kwargs["modify_histogram"](hist_obj)
        calls.append(h)
        return SimpleNamespace(histogram=hist_obj)

    import hps_gpr.io as io

    monkeypatch.setattr(io, "_gp_model", fake_gp_model)
    monkeypatch.setattr(io, "make_kernel_for_dataset", lambda *args, **kwargs: object())

    model = _build_model(ds, blind=(0.05, 0.06), rebin=1, config=cfg, mass=0.055)

    assert calls[0] is toy_hist
    assert calls[1] is toy_hist
    assert np.allclose(model.histogram.axes[0].edges, toy_hist.axes[0].edges)


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


def test_toy_scan_cli_respects_config_save_defaults_and_overrides(monkeypatch, tmp_path):
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
    assert captured[0]["save_plots"] is True
    assert captured[0]["save_fit_json"] is True
    assert captured[0]["save_per_mass_folders"] is True

    result_override = runner.invoke(
        main,
        [
            "toy-scan",
            "--config", str(config_path),
            "--dataset", "2015",
            "--toy-root", str(toy_root),
            "--container", "primary",
            "--toy-pattern", "primary_toy_*",
            "--no-save-plots",
            "--no-save-fit-json",
            "--no-save-per-mass-folders",
        ],
    )
    assert result_override.exit_code == 0, result_override.output
    assert captured[1]["save_plots"] is False
    assert captured[1]["save_fit_json"] is False
    assert captured[1]["save_per_mass_folders"] is False


def test_generate_toy_scan_slurm_scripts_writes_expected_commands(tmp_path):
    job = tmp_path / "submit_toy_scan.slurm"

    job_script, submit_script, n_jobs = generate_toy_scan_slurm_scripts(
        config_path="config_example.yaml",
        output_path=str(job),
        dataset="2015",
        toy_root="outputs/funcform_toys/funcform_2015_toys.root",
        toy_names=["primary_toy_0", "primary_toy_1"],
        toy_indices=[0, 1],
        output_root="outputs/funcform_toys",
        container="primary",
    )

    assert n_jobs == 2
    job_text = Path(job_script).read_text()
    submit_text = Path(submit_script).read_text()
    assert "python -m hps_gpr.cli toy-scan" in job_text
    assert 'cd "${REPO_ROOT}"' in job_text
    assert 'source "${REPO_ROOT}/startup.sh"' in job_text
    assert 'JOB_OUTDIR="${BASE_OUTPUT_DIR}/jobs/${TOY_DIR_NAME}"' in job_text
    assert '--output-dir "${JOB_OUTDIR}"' in job_text
    assert '--toy-root "${TOY_ROOT}"' in job_text
    assert 'CMD+=(--toy-name-fmt "${TOY_NAME}")' in job_text
    assert 'CMD+=(--toy-index "${TOY_INDEX}")' in job_text
    assert 'CMD+=(--toy-pattern "${TOY_NAME}")' in job_text
    assert 'CMD+=(--container "${TOY_CONTAINER}")' in job_text
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in submit_text
    assert 'cd "${SCRIPT_DIR}"' in submit_text
    assert "TOY_NAME=primary_toy_1" in submit_text
    assert "TOY_INDEX=1" in submit_text
    assert "TOY_DIR_NAME=primary_toy_1" in submit_text
    assert 'REPO_ROOT="' in submit_text
