from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot
import yaml


HERE = Path(__file__).resolve().parents[1]
ROOT_FILE = HERE / "inputs" / "pseudo65_background_replacements.root"
SOURCE_ROOT = Path(
    "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"
)
SOURCE_KEY = "preselection/h_invM_8000"
GP_KEY = "gp_mean/preselection/h_invM_8000"
FUNC_KEY = "functional_form_fGenGammaThresh/preselection/h_invM_8000"
GP_EXPECTATION_KEY = "expectations/gp_mean_m065"
FUNC_EXPECTATION_KEY = "expectations/fGenGammaThresh_m065"


def source_and_mask():
    source, edges = uproot.open(SOURCE_ROOT)[SOURCE_KEY].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= 0.060) & (centers < 0.070)
    return np.asarray(source, float), np.asarray(edges, float), mask


def test_225_and_25_sigma_select_same_production_bins_at_65_mev():
    sigma = 0.00184825 - 0.001375 * 0.065 + 0.085875 * 0.065**2
    edges = np.arange(0.040, 0.300 + 0.000625 / 2.0, 0.000625)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask225 = (
        (centers >= 0.065 - 2.25 * sigma)
        & (centers <= 0.065 + 2.25 * sigma)
    )
    mask250 = (
        (centers >= 0.065 - 2.50 * sigma)
        & (centers <= 0.065 + 2.50 * sigma)
    )
    assert sigma == pytest.approx(0.002121696875, abs=1.0e-15)
    assert np.array_equal(mask225, mask250)
    assert np.count_nonzero(mask225) == 16
    indices = np.where(mask225)[0]
    assert edges[indices[0]] == pytest.approx(0.060)
    assert edges[indices[-1] + 1] == pytest.approx(0.070)


def test_pseudo_histograms_preserve_every_outside_bin_exactly():
    source, edges, mask = source_and_mask()
    root_file = uproot.open(ROOT_FILE)
    for key in (GP_KEY, FUNC_KEY):
        values, pseudo_edges = root_file[key].to_numpy()
        assert root_file[key].classname == "TH1D"
        assert np.array_equal(pseudo_edges, edges)
        assert np.array_equal(values[~mask], source[~mask])
        assert np.all(values[mask] == np.rint(values[mask]))
        assert np.all(values[mask] >= 0.0)


def test_fixed_seed_poisson_draws_are_exactly_reproducible():
    _, _, mask = source_and_mask()
    root_file = uproot.open(ROOT_FILE)
    gp_expectation = root_file[GP_EXPECTATION_KEY].values()[mask]
    func_expectation = root_file[FUNC_EXPECTATION_KEY].values()[mask]
    master = np.random.SeedSequence(20260806)
    gp_seed, func_seed = master.spawn(2)
    gp_draw = np.random.Generator(np.random.PCG64(gp_seed)).poisson(gp_expectation)
    func_draw = np.random.Generator(np.random.PCG64(func_seed)).poisson(
        func_expectation
    )
    assert np.array_equal(gp_draw, root_file[GP_KEY].values()[mask])
    assert np.array_equal(func_draw, root_file[FUNC_KEY].values()[mask])


def test_functional_fit_is_two_sided_and_passes_predeclared_qc():
    payload = json.loads((HERE / "derived" / "functional_fit_qc.json").read_text())
    assert payload["model"] == "fGenGammaThresh"
    assert payload["excluded_interval_GeV"] == [0.06, 0.07]
    assert payload["n_bins_low_sideband"] > 0
    assert payload["n_bins_high_sideband"] > 0
    assert payload["optimizer"]["n_deterministic_starts"] == 5
    assert payload["poisson_deviance_per_ndf"] < 1.5
    assert payload["pearson_chi2_per_ndf"] < 1.5
    assert payload["poisson_deviance_pvalue"] > 0.01
    assert not any(payload["parameter_at_bound"].values())
    assert payload["fit_qc_pass"]


@pytest.mark.parametrize(
    "name",
    [
        "config_obsUL90_2021_10pct_gpmean_replacement_v4p2.yaml",
        "config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml",
    ],
)
def test_configs_are_portable_observed_asymptotic_cards(name):
    config = yaml.safe_load((HERE / "configs" / name).read_text())
    assert not Path(config["path_2021"]).is_absolute()
    assert not Path(config["output_dir"]).is_absolute()
    assert "/private/tmp/" not in config["path_2021"]
    assert "/private/tmp/" not in config["output_dir"]
    assert config["enable_2021"]
    assert not config["enable_2015"]
    assert not config["enable_2016"]
    assert config["blind_nsigma"] == 2.25
    assert config["gp_train_exclude_nsigma"] == 2.25
    assert config["neighborhood_rebin"] == 5
    assert config["kernel_ls_res_upper_factor_by_dataset"]["2021"] == 15.0
    assert config["cls_alpha"] == 0.1
    assert config["cls_mode"] == "asymptotic"
    assert config["cls_num_toys"] == 0
    assert not config["make_ul_bands"]
    assert not config["do_combined_bands"]
    assert not config["inject_signal"]


@pytest.mark.parametrize("lane", ["gp_mean", "functional_form"])
def test_reviewed_scans_are_complete_when_present(lane):
    path = HERE / "derived" / f"{lane}_results_reviewed.csv"
    if not path.exists():
        pytest.skip("reviewed scan not available yet")
    frame = pd.read_csv(path)
    expected = np.round(np.arange(0.050, 0.250 + 0.0005, 0.001), 3)
    assert len(frame) == len(expected)
    assert np.array_equal(frame["mass_GeV"].to_numpy(float), expected)
    assert frame["extract_success"].astype(bool).all()
    assert np.all(np.isfinite(frame[["eps2_up", "p0_analytic", "lml"]]))
    assert (frame["selected_state_reproducing_attempt_count"] >= 2).all()
    assert not frame["review_status"].astype(str).str.contains("pending").any()


def test_generated_json_has_no_checkout_specific_tmp_paths():
    json_paths = [
        HERE / "derived" / "optimizer_audit.json",
        HERE / "derived" / "plot_manifest.json",
        HERE / "derived" / "input_validation.json",
        HERE / "derived" / "final_validation.json",
        *sorted((HERE / "runs").glob("**/validation_report.json")),
    ]
    for path in json_paths:
        if not path.exists():
            continue
        assert "/private/tmp/" not in path.read_text(), path
