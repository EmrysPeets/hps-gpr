import json
from types import SimpleNamespace

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.io import _at_kernel_bound, _extract_constant_bounds_and_value
from hps_gpr.scan import _density_diagnostics_payload, _gp_diagnostics_payload


def test_constant_bounds_are_read_from_actual_kernel():
    kernel = ConstantKernel(7.0, (2.0, 50.0)) * RBF(0.2, (0.1, 0.4))

    lo, hi, initial = _extract_constant_bounds_and_value(kernel)

    assert (lo, hi, initial) == (2.0, 50.0, 7.0)
    assert _at_kernel_bound(49.99, hi)
    assert not _at_kernel_bound(40.0, hi)


def test_gp_diagnostics_payload_records_support_and_both_kernel_bounds():
    pred = SimpleNamespace(
        kernel_str="7**2 * RBF(length_scale=0.4)",
        ls_lo=0.1,
        ls_hi=0.4,
        ls_init=0.2,
        ls_opt=0.4,
        ls_at_lower=False,
        ls_at_upper=True,
        sigma_x=0.03,
        const_lo=2.0,
        const_hi=50.0,
        const_init=7.0,
        const_opt=49.99,
        const_at_lower=False,
        const_at_upper=True,
        blind_train=(0.061, 0.071),
        train_domain_lo=0.055,
        train_domain_hi=0.120,
        n_full=104,
        n_blind=16,
        n_train=88,
        n_train_low=10,
        n_train_high=78,
        bin_width_median=0.000625,
        optimizer_restarts=12,
        lml=1234.5,
    )

    payload = _gp_diagnostics_payload(pred)

    assert payload["constant"] == {
        "lower": 2.0,
        "upper": 50.0,
        "initial": 7.0,
        "optimized": 49.99,
        "at_lower": False,
        "at_upper": True,
    }
    assert payload["length_scale"]["at_upper"] is True
    assert payload["training"]["n_train_low"] == 10
    assert payload["training"]["n_train_high"] == 78
    assert np.isclose(payload["training"]["bin_width_median_GeV"], 0.000625)
    assert payload["optimizer"]["restarts"] == 12


def test_scan_csv_and_json_record_density_window_provenance(
    monkeypatch,
    tmp_path,
):
    import hps_gpr.scan as scan_mod

    ds = DatasetConfig(
        key="2021",
        label="HPS 2021",
        root_path="unused.root",
        hist_name="unused",
        m_low=0.050,
        m_high=0.050,
        sigma_coeffs=[0.002],
        frad_coeffs=[0.05],
    )
    cfg = Config(
        output_dir=str(tmp_path),
        save_plots=False,
        save_fit_json=True,
        debug_print=False,
        fail_fast=True,
    )
    pred = SimpleNamespace(
        sigma_val=0.002,
        blind=(0.0455, 0.0545),
        integral_density=4.25e8,
        density_nsigma=1.64,
        density_window_lo=0.04672,
        density_window_hi=0.05328,
        density_window_width=0.00656,
        density_source_lo=0.0,
        density_source_hi=1.0,
        density_source_n_bins=8000,
        density_source_bin_width_median=0.000125,
        density_window_fully_covered=True,
    )
    result = SimpleNamespace(
        mass=0.050,
        A_up=12000.0,
        eps2_up=1.2e-5,
        p0_analytic=0.1,
        Z_analytic=1.28155,
        A_hat=100.0,
        sigma_A=5000.0,
        extract_success=True,
    )
    monkeypatch.setattr(
        scan_mod,
        "active_datasets_for_mass",
        lambda mass, datasets, config: [ds],
    )
    monkeypatch.setattr(
        scan_mod,
        "evaluate_single_dataset",
        lambda *args, **kwargs: (result, pred, None),
    )

    single, _ = scan_mod.run_scan({"2021": ds}, cfg)

    row = single.iloc[0]
    assert row["integral_density"] == 4.25e8
    assert row["density_nsigma"] == 1.64
    assert row["density_window_lo"] == 0.04672
    assert row["density_window_hi"] == 0.05328
    assert row["density_source_lo"] == 0.0
    assert row["density_source_hi"] == 1.0
    assert row["density_source_n_bins"] == 8000
    assert bool(row["density_window_fully_covered"]) is True

    numbers_path = tmp_path / "m050MeV" / "2021" / "numbers.json"
    with numbers_path.open("r", encoding="utf-8") as stream:
        numbers = json.load(stream)
    assert numbers["density"] == _density_diagnostics_payload(pred)
    assert numbers["density"]["window_lo_GeV"] == 0.04672
    assert numbers["density"]["window_hi_GeV"] == 0.05328
    assert numbers["density"]["fully_covered"] is True
