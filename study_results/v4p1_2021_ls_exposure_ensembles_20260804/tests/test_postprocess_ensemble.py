from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STUDY_DIR = Path(__file__).resolve().parents[1]
SCRIPT = STUDY_DIR / "postprocess_ensemble.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v4p1_postprocess", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixed_amplitude_frame(*, drift: bool = False) -> pd.DataFrame:
    rows = []
    strengths = [0.0, 10.0, 30.0, 50.0]
    for factor in [6, 15]:
        for anchor_nsigma, strength in zip([0.0, 1.0, 3.0, 5.0], strengths):
            value = strength
            if drift and factor == 6 and strength == 30.0:
                value = 31.0
            sigma_a_ref = 20.0 if factor == 6 else 10.0
            rows.append(
                {
                    "truth_model": "gengamma",
                    "study_scenario": "2021_1pct",
                    "background_toy_index": 0,
                    "mass_mev": 60,
                    "injection_toy": 0,
                    "factor": factor,
                    "strength": value,
                    "inj_nsigma": value / sigma_a_ref,
                    "injection_protocol": "factor15_prefit_asimov_absolute_v1",
                    "injection_anchor_factor": 15,
                    "injection_anchor_nsigma": anchor_nsigma,
                    "injection_anchor_strength": strength,
                    "injection_anchor_sigmaA_ref": 10.0,
                    "injection_anchor_ledger_sha256": "a" * 64,
                    "sigmaA_ref": sigma_a_ref,
                    "sigma_A": 2.0 if factor == 6 else 1.0,
                    "sigmaA_ref_matched": np.nan,
                    "sigmaA_ref_matched_ok": np.nan,
                    "A_hat": value + 0.5,
                }
            )
    return pd.DataFrame(rows)


def _minimal_spec() -> dict:
    return {
        "length_scale_upper_factors": [6, 15],
        "injection_closure": {
            "protocol": "factor15_prefit_asimov_absolute_v1",
            "sigma_strengths": [0.0, 1.0, 3.0, 5.0],
            "fixed_amplitude_anchor_factor": 15,
        },
    }


def test_fixed_amplitude_levels_are_anchored_and_normalized():
    module = _load_module()
    result = module._assign_fixed_amplitude_levels(
        _fixed_amplitude_frame(), _minimal_spec()
    )
    factor_six = result.loc[result["factor"] == 6].sort_values("anchor_nsigma")
    assert factor_six["anchor_nsigma"].tolist() == [0.0, 1.0, 3.0, 5.0]
    assert factor_six["strength"].tolist() == [0.0, 10.0, 30.0, 50.0]
    assert factor_six["sigma_A_over_anchor"].tolist() == [2.0] * 4
    assert factor_six["sigmaA_ref_over_anchor"].tolist() == [2.0] * 4
    assert result["sigmaA_ref_matched"].isna().all()


def test_factor_specific_sigma_scaled_strengths_fail_closed():
    module = _load_module()
    with pytest.raises(module.ReviewGateError, match="absolute strength"):
        module._assign_fixed_amplitude_levels(
            _fixed_amplitude_frame(drift=True), _minimal_spec()
        )


def test_postprocessor_has_no_fit_execution_imports():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import hps_gpr" not in source
    assert "subprocess" not in source
    assert "--execute" not in source


def test_note_facing_scenario_labels_are_human_readable():
    module = _load_module()
    assert module._scenario_display_label("2021_1pct") == "2021 1%"
    assert module._scenario_display_label("2021_1pct_x10") == "2021 1% × 10"
    assert module._scenario_display_label("2021_1pct_x100") == "2021 1% × 100"
    assert module._scenario_display_label("2021_10pct") == "2021 10%"
    assert module._scenario_display_label("2021_10pct_x10") == "2021 10% × 10"
    with pytest.raises(module.ReviewGateError, match="display label"):
        module._scenario_display_label("unknown")


def test_pull_width_is_computed_within_mass_level_strata():
    module = _load_module()
    injection_rows = []
    response_rows = []
    for level in [0.0, 1.0]:
        for toy in range(10):
            injection_rows.append(
                {
                    "truth_model": "gengamma",
                    "study_scenario": "2021_1pct",
                    "factor": 15,
                    "mass_mev": 60,
                    "anchor_nsigma": level,
                    "background_toy_index": toy,
                    "pull_param": level + toy / 10.0,
                    "sigma_A_over_anchor": 1.0,
                    "sigmaA_ref_over_anchor": 1.0,
                    "Ahat_minus_Ainj_over_anchor_sigma": 0.0,
                    "refit_ls_opt_over_sigma_x": 8.0,
                    "refit_at_upper_bound": False,
                    "Nsig_win": int(level * 10),
                    "Nsig_train": 0,
                }
            )
            if level > 0:
                response_rows.append(
                    {
                        "truth_model": "gengamma",
                        "study_scenario": "2021_1pct",
                        "factor": 15,
                        "mass_mev": 60,
                        "anchor_nsigma": level,
                        "paired_response": 1.0,
                        "paired_response_candidate_prefit_sigma_units": 1.0,
                        "paired_response_anchor_prefit_sigma_units": 1.0,
                        "paired_response_anchor_fitted_sigma_units": 1.0,
                    }
                )
    strata = module.build_stratified_signal_summary(
        pd.DataFrame(injection_rows),
        pd.DataFrame(response_rows),
        {
            "n_toys": 10,
            "injection_closure": {"replicas_per_background_toy": 1},
        },
    )
    assert len(strata) == 2
    assert strata["injection_rows"].tolist() == [10, 10]
    assert strata.loc[strata["anchor_nsigma"] == 1.0, "response_rows"].item() == 10
    assert strata["pull_width"].notna().all()


def test_factor_summary_consumes_stratified_signal_metrics():
    module = _load_module()
    scan = pd.DataFrame(
        [
            {
                "truth_model": "gengamma",
                "study_scenario": "2021_1pct",
                "factor": factor,
                "ls_opt_over_sigma_x": 5.0,
                "background_toy_index": 0,
                "mass_mev": 50,
                "at_upper_bound": False,
            }
            for factor in [6, 15]
        ]
    )
    strata_rows = []
    for factor in [6, 15]:
        for level in [0.0, 1.0]:
            strata_rows.append(
                {
                    "truth_model": "gengamma",
                    "study_scenario": "2021_1pct",
                    "factor": factor,
                    "anchor_nsigma": level,
                    "injection_rows": 10,
                    "pull_mean": 0.0,
                    "pull_width": 1.0,
                    "sigma_A_over_anchor_median": 1.0,
                    "sigmaA_ref_over_anchor_median": 1.0,
                    "Ahat_minus_Ainj_over_anchor_sigma_median": 0.0,
                    "refit_ls_ratio_median": 5.0,
                    "refit_bound_row_fraction": 0.0,
                    "response_rows": 10 if level else float("nan"),
                    "paired_response_mean": 1.0 if level else float("nan"),
                    "paired_response_median": 1.0 if level else float("nan"),
                    "paired_response_std": 0.1 if level else float("nan"),
                    "paired_response_candidate_prefit_sigma_units_median": (
                        1.0 if level else float("nan")
                    ),
                    "paired_response_anchor_prefit_sigma_units_median": (
                        1.0 if level else float("nan")
                    ),
                    "paired_response_anchor_fitted_sigma_units_median": (
                        1.0 if level else float("nan")
                    ),
                }
            )
    audit = pd.DataFrame(
        [
            {
                "truth_model": "gengamma",
                "study_scenario": "2021_1pct",
                "factor_current": 15,
                "delta_lml": 1.0,
                "regression_beyond_tolerance": False,
            }
        ]
    )
    summary = module.build_summary(scan, pd.DataFrame(strata_rows), audit)
    assert len(summary) == 2
    assert summary["pull_width_strata_median"].tolist() == [1.0, 1.0]
    assert summary["paired_response_median"].tolist() == [1.0, 1.0]


def _factor20_plot_spec() -> dict:
    scenarios = {
        "2021_1pct": {
            "source_family": "one_pct",
            "exposure_multiplier": 1,
        },
        "2021_1pct_x10": {
            "source_family": "one_pct",
            "exposure_multiplier": 10,
        },
        "2021_1pct_x100": {
            "source_family": "one_pct",
            "exposure_multiplier": 100,
        },
        "2021_10pct": {
            "source_family": "ten_pct",
            "exposure_multiplier": 1,
        },
        "2021_10pct_x10": {
            "source_family": "ten_pct",
            "exposure_multiplier": 10,
        },
    }
    return {
        "n_toys": 10,
        "length_scale_upper_factors": [20],
        "default_mass_grid_mev": {"min": 50, "max": 250, "step": 20},
        "truth_models": {
            "gengamma": {"function_tag": "fGenGammaThresh"},
            "sigpowexpq": {"function_tag": "fSigPowExpQ"},
        },
        "scenarios": scenarios,
    }


def _factor20_scan_frame() -> pd.DataFrame:
    rows = []
    spec = _factor20_plot_spec()
    masses = range(50, 251, 20)
    for truth_index, truth in enumerate(spec["truth_models"]):
        for scenario_index, scenario in enumerate(spec["scenarios"]):
            for toy in range(10):
                for mass in masses:
                    value = (
                        7.5
                        + 4.0 * truth_index
                        + 0.35 * scenario_index
                        + 0.08 * toy
                        + 0.002 * (mass - 50)
                    )
                    rows.append(
                        {
                            "factor": 20,
                            "truth_model": truth,
                            "study_scenario": scenario,
                            "background_toy_index": toy,
                            "mass_mev": mass,
                            "ls_opt_over_sigma_x": value,
                            "at_upper_bound": False,
                        }
                    )
    return pd.DataFrame(rows)


def _comparison_normalization() -> dict:
    one_pct = 12_504_044
    ten_pct = 141_251_508
    return {
        "source_normalization_target_counts": {
            "one_pct": one_pct,
            "ten_pct": ten_pct,
        },
        "source_support_ratio_ten_pct_over_one_pct": ten_pct / one_pct,
        "effective_target_counts": {
            "2021_1pct_x10": one_pct * 10,
            "2021_10pct": ten_pct,
        },
        "effective_target_ratio_native10_over_1pct_x10": (
            ten_pct / (one_pct * 10)
        ),
    }


def test_lml_tolerance_uses_the_frozen_absolute_plus_relative_formula():
    module = _load_module()
    low = -2_000_000.0
    within = low - 1.5
    beyond = low - 2.1
    within_tolerance = module._nested_lml_tolerance(low, within)
    beyond_tolerance = module._nested_lml_tolerance(low, beyond)
    assert within_tolerance == pytest.approx(2.0000015)
    assert within - low >= -within_tolerance
    assert beyond - low < -beyond_tolerance


def test_coherent_one_sided_qmu_zero_is_recorded_but_not_promoted():
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "task_id": "injection__f20__alternate__source__t0005",
                "mass_GeV": 0.12,
                "strength": 3928.5,
                "A_hat": 4546.4,
                "qmu_A_test": 3928.5,
                "qmu_ok": False,
                "qmu_branch": "muhat_gt_test",
                "qmu_tilde": 0.0,
                "tmu_tilde": 0.0,
                "sqrt_qmu_tilde": 0.0,
                "sqrt_tmu_tilde": 0.0,
            }
        ]
    )
    result = module._annotate_qmu_diagnostics(frame.copy())
    assert result["qmu_ok"].tolist() == [False]
    assert result["qmu_ok_parsed"].tolist() == [False]
    assert result["qmu_one_sided_zero_branch_diagnostic"].tolist() == [True]
    assert result["qmu_outputs_used_in_postprocess"].tolist() == [False]
    assert result["qmu_outputs_promotable"].tolist() == [False]

    incoherent = frame.copy()
    incoherent.loc[0, "qmu_branch"] = "unexpected_branch"
    with pytest.raises(module.ReviewGateError, match="incoherent"):
        module._annotate_qmu_diagnostics(incoherent)


def test_all_toy_plot_writes_matching_one_page_factor_pdf(tmp_path):
    module = _load_module()
    outputs = module.plot_all_toy_curves(
        _factor20_scan_frame(),
        _factor20_plot_spec(),
        "gengamma",
        plots_dir=tmp_path,
    )
    names = {path.name for path in outputs}
    assert "fig_v4p1_ensemble_ls_all_toys_gengamma.pdf" in names
    assert "fig_v4p1_ensemble_ls_all_toys_gengamma_f20.pdf" in names
    assert "fig_v4p1_ensemble_ls_all_toys_gengamma_f20.png" in names
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_factor20_comparison_is_unpaired_and_uses_ten_toy_medians(tmp_path):
    module = _load_module()
    normalization = _comparison_normalization()
    toy_rows, summary = module.build_factor20_toy_median_comparison(
        _factor20_scan_frame(),
        _factor20_plot_spec(),
        normalization,
    )
    assert len(toy_rows) == 40
    assert len(summary) == 4
    assert set(toy_rows["n_mass_rows"]) == {11}
    assert set(toy_rows["n_unique_masses"]) == {11}
    assert set(summary["n_independent_toys"]) == {10}
    assert not summary["source_families_paired"].any()
    assert set(summary["expected_limit_bands"]) == {False}
    assert summary[
        "source_support_ratio_ten_pct_over_one_pct"
    ].iloc[0] == pytest.approx(11.296466007317314)
    outputs = module.plot_factor20_toy_median_comparison(
        toy_rows,
        summary,
        normalization,
        plots_dir=tmp_path,
    )
    assert {path.suffix for path in outputs} == {".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_reviewed_scan_validate_only_does_not_open_injection_file(
    tmp_path, capsys
):
    module = _load_module()
    missing_injection = tmp_path / "must_not_be_opened.csv"
    status = module.main(
        [
            "--scan-note-artifacts",
            "--validate-only",
            "--injection-csv",
            str(missing_injection),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert '"injection_data_read": false' in captured.out
    assert not missing_injection.exists()
