#!/usr/bin/env python3
"""Generate final 2015 fixed-histogram functional-form closure configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
STUDY_CONFIGS = HERE.parent
BASE_CONFIG = STUDY_CONFIGS / "config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb0p5.yaml"

MASS_GRID = [0.030, 0.045, 0.060, 0.075, 0.090, 0.105, 0.120]
SIGMA_MULTIPLIERS = [0.0, 1.0, 2.0, 3.0, 5.0]


def tag_float(value: float) -> str:
    value = float(value)
    text = f"{value:.1f}" if value.is_integer() else f"{value:g}"
    return text.replace(".", "p")


def output_float(value: float) -> str:
    value = float(value)
    text = f"{value:.1f}" if value.is_integer() else f"{value:g}"
    return text.replace(".", "pt")


def build_config(base: dict, *, blind_nsigma: float, lslb: float) -> dict:
    cfg = deepcopy(base)
    cfg.update(
        {
            "blind_nsigma": float(blind_nsigma),
            "gp_train_exclude_nsigma": float(blind_nsigma),
            "inj_train_exclude_nsigma": float(blind_nsigma),
            "ul_bands_train_exclude_nsigma": float(blind_nsigma),
            "kernel_ls_res_lower_factor": float(lslb),
            "kernel_ls_res_lower_factor_by_dataset": {
                "2015": float(lslb),
                "2016": 0.5,
                "2021": 0.5,
            },
            "inject_signal": True,
            "inj_dataset_key": "2015",
            "inj_masses_gev": list(MASS_GRID),
            "inj_mode": "poisson",
            "inj_strength_mode": "sigmaA",
            "inj_sigma_a_source": "asimov",
            "inj_shape_mode": "full",
            "inj_background_mode": "fixed_hist",
            "inj_refit_gp_on_toy": True,
            "inj_refit_gp_restarts": 0,
            "inj_refit_gp_optimize": True,
            "inj_sigma_multipliers": list(SIGMA_MULTIPLIERS),
            "inj_write_toy_csv": True,
            "inj_write_qmu": True,
            "inj_stream_aggregate": True,
            "inj_aggregate_every": 100,
            "inj_n_workers": 1,
            "inj_threads_per_worker": 1,
            "signal_model": "default",
            "output_dir": (
                "outputs/"
                f"final_2015_funcform_fixedhist_blind{output_float(blind_nsigma)}_"
                f"lslb_{output_float(lslb)}"
            ),
        }
    )
    return cfg


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    base = yaml.safe_load(BASE_CONFIG.read_text()) or {}

    for blind_nsigma in (1.64, 1.96):
        for lslb in (0.5, 1.0):
            cfg = build_config(base, blind_nsigma=blind_nsigma, lslb=lslb)
            out = HERE / (
                f"config_2015_blind{tag_float(blind_nsigma)}_95CL_"
                f"funcform100_fixedhist_refit_lslb{tag_float(lslb)}.yaml"
            )
            out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            print(f"Wrote {out.relative_to(STUDY_CONFIGS.parent)}")


if __name__ == "__main__":
    main()
