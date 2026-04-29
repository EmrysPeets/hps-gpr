#!/usr/bin/env python3
"""Write the 2015 functional-form refit-closure 1.5 study configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


MASSES = [0.030, 0.045, 0.060, 0.075, 0.090, 0.105, 0.120]
STRENGTHS = ["s0", "s1", "s2", "s3", "s5"]


BASE = {
    "path_2015": "/sdf/home/e/epeets/move/2015_IMD.root",
    "path_2016": "/sdf/home/e/epeets/move/EventSelection_Data_10Percent.root",
    "path_2021": "/sdf/home/e/epeets/run/2021_bump/preselection_invM_psumlt2p8_hists.root",
    "path_2021_mc": "",
    "hist_2015": "invariant_mass",
    "hist_2016": "h_Minv_General_Final_1",
    "hist_2021": "preselection/h_invM_8000",
    "range_2015": [0.02, 0.13],
    "range_2016": [0.035, 0.21],
    "range_2021": [0.03, 0.25],
    "sigma_coeffs_2015": [-9.22283032152e-05, 0.0532190838657],
    "sigma_coeffs_2016": [0.00038, 0.041, -0.27, 3.49, -11.11],
    "sigma_tail_m0_2016": 0.18,
    "sigma_tail_slope_floor_2016": 0.0,
    "sigma_tail_slope_override_2016": 0.0239,
    "sigma_coeffs_2021": [0.0014786, -0.0011, 0.0687],
    "frad_coeffs_2015": [0.085],
    "frad_coeffs_2016": [0.05],
    "frad_coeffs_2021": [0.05],
    "enable_2015": True,
    "enable_2016": False,
    "enable_2021": False,
    "kernel_constant_init": 1.0,
    "kernel_constant_bounds": [1.0e-08, 1.0e18],
    "kernel_ls_init": 0.03,
    "kernel_ls_bounds": [0.001, 0.5],
    "kernel_ls_policy": "resolution_scaled_local",
    "kernel_ls_res_lower_factor": 1.5,
    "kernel_ls_res_upper_factor": 8.0,
    "kernel_ls_res_stat": "median",
    "kernel_ls_res_npts": 300,
    "kernel_ls_res_lower_factor_by_dataset": {"2015": 1.5, "2016": 0.5, "2021": 0.5},
    "kernel_ls_res_upper_factor_by_dataset": {"2015": 8.0, "2016": 8.0, "2021": 9.0},
    "kernel_ls_local_hi_floor_mode": "dataset_stat",
    "kernel_ls_local_hi_floor_factor": 0.8,
    "blind_nsigma": 1.64,
    "gp_train_exclude_nsigma": 1.64,
    "neighborhood_rebin": 5,
    "n_restarts": 10,
    "cls_alpha": 0.05,
    "cls_mode": "asymptotic",
    "cls_num_toys": 10000,
    "make_ul_bands": True,
    "ul_bands_toys": 10000,
    "ul_bands_cls_mode": "asymptotic",
    "extract_allow_negative": True,
    "funcform_closure_enable": True,
    "funcform_closure_root_by_dataset": {
        "2015": "outputs/funcform_toys/funcform_2015_dataset_mod_toys_2.root",
        "2016": "outputs/funcform_toys/funcform_2016_dataset_mod_toys_2.root",
        "2021": "outputs/funcform_toys/funcform_2021_dataset_mod_toys_2.root",
    },
    "funcform_closure_container_by_dataset": {
        "2015": "fShiftSigPowTail",
        "2016": "fShiftSigPowTail",
        "2021": "fSigPowExpQ",
    },
    "funcform_closure_toy_pattern_by_dataset": {
        "2015": "fShiftSigPowTail_toy_*",
        "2016": "fShiftSigPowTail_toy_*",
        "2021": "fSigPowExpQ_toy_*",
    },
    "extraction_display_funcform_toy_index": 0,
    "inject_signal": True,
    "inj_dataset_key": "2015",
    "inj_masses_gev": MASSES,
    "inj_mode": "poisson",
    "inj_strength_mode": "sigmaA",
    "inj_sigma_a_source": "asimov",
    "inj_shape_mode": "full",
    "inj_refit_gp_on_toy": True,
    "inj_refit_gp_restarts": 0,
    "inj_refit_gp_optimize": True,
    "inj_train_exclude_nsigma": 1.64,
    "inj_sigma_multipliers": [0.0, 1.0, 2.0, 3.0, 5.0],
    "inj_write_toy_csv": True,
    "inj_write_qmu": True,
    "inj_stream_aggregate": True,
    "inj_aggregate_every": 100,
    "inj_n_workers": 1,
    "inj_threads_per_worker": 1,
    "do_combined": False,
    "run_limit_bands_on": "2015",
    "make_eps2_bands": True,
    "debug_print": True,
    "debug_max_errors": 10,
    "fail_fast": False,
    "save_per_mass_folders": True,
    "save_plots": False,
    "save_fit_json": True,
}


VARIANTS = {
    "main": {
        "output_name": "config_2015_blind1p64_95CL_funcform100_refit_lslb1p5.yaml",
        "output_dir": "outputs/study_2015_w1p64_95CL_funcform100_refit_lslb_1pt5",
        "job_name": "hps2015_ffinj_lslb1p5",
    },
    "window": {
        "output_name": "config_2015_blind1p64_95CL_funcform100_refit_lslb1p5_window.yaml",
        "output_dir": "outputs/study_2015_w1p64_95CL_funcform100_refit_lslb_1pt5_window",
        "job_name": "hps2015_ffinj_lslb1p5_win",
        "inj_shape_mode": "window",
    },
    "train1p96": {
        "output_name": "config_2015_blind1p64_95CL_funcform100_refit_lslb1p5_train1p96.yaml",
        "output_dir": "outputs/study_2015_w1p64_95CL_funcform100_refit_lslb_1pt5_train1p96",
        "job_name": "hps2015_ffinj_lslb1p5_tr196",
        "inj_train_exclude_nsigma": 1.96,
    },
}


def write_configs(repo_root: Path) -> list[Path]:
    out_paths: list[Path] = []
    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(BASE)
        cfg["output_dir"] = overrides["output_dir"]
        if "inj_shape_mode" in overrides:
            cfg["inj_shape_mode"] = overrides["inj_shape_mode"]
        if "inj_train_exclude_nsigma" in overrides:
            cfg["inj_train_exclude_nsigma"] = overrides["inj_train_exclude_nsigma"]
        path = repo_root / "study_configs" / overrides["output_name"]
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        out_paths.append(path)
        print(f"Wrote {path.relative_to(repo_root)}")
    return out_paths


def write_helpers(repo_root: Path) -> None:
    helper_dir = repo_root / "study_configs" / "funcform_refit_lslb1p5"
    helper_dir.mkdir(parents=True, exist_ok=True)
    strengths_arg = ",".join(STRENGTHS)
    masses_arg = ",".join(f"{m:.3f}" for m in MASSES)

    smoke = helper_dir / "run_smoke.sh"
    smoke.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "hps-gpr funcform-inject \\",
                "  --config study_configs/config_2015_blind1p64_95CL_funcform100_refit_lslb1p5.yaml \\",
                "  --dataset 2015 \\",
                "  --max-toys 1 \\",
                "  --masses 0.060 \\",
                "  --strengths s1 \\",
                "  --n-injection-toys 1 \\",
                "  --write-toy-csv \\",
                "  --write-qmu \\",
                "  --output-dir outputs/smoke_2015_funcform_refit_lslb_1pt5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    smoke.chmod(0o755)
    print(f"Wrote {smoke.relative_to(repo_root)}")

    gen = helper_dir / "generate_slurm.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for name, overrides in VARIANTS.items():
        slurm_dir = helper_dir / f"slurm_{name}"
        lines.extend(
            [
                f"mkdir -p {slurm_dir.relative_to(repo_root)}",
                "hps-gpr slurm-gen-funcform-inject \\",
                f"  --config study_configs/{overrides['output_name']} \\",
                "  --dataset 2015 \\",
                f"  --masses {masses_arg} \\",
                f"  --strengths {strengths_arg} \\",
                "  --n-injection-toys 1 \\",
                "  --write-qmu \\",
                "  --cpus-per-task 1 \\",
                f"  --job-name {overrides['job_name']} \\",
                "  --partition roma \\",
                "  --account hps:hps-prod \\",
                "  --time 1:00:00 \\",
                "  --memory 8G \\",
                f"  --output {slurm_dir.relative_to(repo_root)}/submit_funcform_injection_{name}.slurm",
                "",
            ]
        )
    gen.write_text("\n".join(lines), encoding="utf-8")
    gen.chmod(0o755)
    print(f"Wrote {gen.relative_to(repo_root)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    write_configs(repo_root)
    write_helpers(repo_root)


if __name__ == "__main__":
    main()
