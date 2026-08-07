from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import validate_ensemble


def test_input_validation_passes():
    payload = validate_ensemble.validate_inputs()
    assert payload["pass"], payload


def test_two_unique_window_geometries_and_exact_pairing():
    provenance = json.loads(
        (HERE / "derived" / "input_provenance.json").read_text()
    )
    windows = provenance["windows"]
    assert windows["window_2p25eq2p5"]["requested_nsigma"] == [2.25, 2.5]
    assert np.allclose(
        windows["window_2p25eq2p5"]["complete_bin_interval_GeV"],
        [0.060000000000000005, 0.07],
        atol=1.0e-15,
        rtol=0.0,
    )
    assert windows["window_2p25eq2p5"]["n_analysis_bins"] == 16
    assert np.allclose(
        windows["window_3p0"]["complete_bin_interval_GeV"],
        [0.058750000000000004, 0.07125000000000001],
        atol=1.0e-15,
        rtol=0.0,
    )
    assert windows["window_3p0"]["n_analysis_bins"] == 20

    root = uproot.open(HERE / "inputs" / "gp_window_ensemble.root")
    source, edges = root[
        "source/preselection/h_invM_8000"
    ].to_numpy(flow=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shared = (centers >= 0.060) & (centers < 0.070)
    narrow = shared
    wide = (centers >= 0.05875) & (centers < 0.07125)
    for draw_index in range(10):
        narrow_values, _ = root[
            f"gp/window_2p25eq2p5/draw_{draw_index:02d}/"
            "preselection/h_invM_8000"
        ].to_numpy(flow=False)
        wide_values, _ = root[
            f"gp/window_3p0/draw_{draw_index:02d}/"
            "preselection/h_invM_8000"
        ].to_numpy(flow=False)
        assert np.array_equal(narrow_values[shared], wide_values[shared])
        assert np.array_equal(narrow_values[~narrow], source[~narrow])
        assert np.array_equal(wide_values[~wide], source[~wide])


def test_ten_distinct_draws_per_geometry_and_ten_child_streams():
    provenance = json.loads(
        (HERE / "derived" / "input_provenance.json").read_text()
    )
    child_states = {
        json.dumps(item["child_seed_state"], sort_keys=True)
        for item in provenance["draws"]
    }
    assert len(child_states) == 10
    for window in ("window_2p25eq2p5", "window_3p0"):
        hashes = [
            item["full_histogram_sha256"]
            for item in provenance["draws"]
            if item["window"] == window
        ]
        assert len(hashes) == 10
        assert len(set(hashes)) == 10


def test_scan_cards_disable_bands_toys_and_injection():
    manifest = json.loads(
        (HERE / "derived" / "config_manifest.json").read_text()
    )
    assert manifest["generated_config_count"] == 20
    for record in manifest["records"]:
        config = yaml.safe_load((REPO / record["config"]).read_text())
        assert config["cls_mode"] == "asymptotic"
        assert config["cls_num_toys"] == 0
        assert config["make_ul_bands"] is False
        assert config["ul_bands_toys"] == 0
        assert config["do_combined_bands"] is False
        assert config["combined_bands_n_toys"] == 0
        assert config["inject_signal"] is False


def test_final_validation_passes():
    payload = validate_ensemble.validate_final()
    assert payload["pass"], payload


def test_summary_is_descriptive_ten_draw_pointwise_product():
    summary = pd.read_csv(
        HERE / "derived" / "ensemble_pointwise_summary.csv"
    )
    assert len(summary) == 402
    assert set(summary["draw_count"].astype(int)) == {10}
    assert set(summary["summary_scope"]) == {
        "pointwise descriptive statistics across ten conditional draws"
    }
    for metric in ("eps2_up", "p0_analytic"):
        assert np.all(summary[f"{metric}_q16"] <= summary[f"{metric}_median"])
        assert np.all(summary[f"{metric}_median"] <= summary[f"{metric}_q84"])
