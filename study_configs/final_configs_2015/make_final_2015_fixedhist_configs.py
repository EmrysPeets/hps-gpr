#!/usr/bin/env python3
"""Generate 2015 fixed-histogram functional-form closure rescue configs."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
STUDY_CONFIGS = HERE.parent
BASE_CONFIG = STUDY_CONFIGS / "config_2015_blind1p64_train1p96_95CL_funcform100_refit_lslb0p5.yaml"

MASS_GRID = [0.030, 0.045, 0.060, 0.075, 0.090, 0.105, 0.120]
SIGMA_MULTIPLIERS = [0.0, 1.0, 2.0, 3.0, 5.0]
BLIND_WIDTHS = [1.64, 1.96]
GUARD_WIDTHS = [2.58, 3.0]
DEFAULT_OUTPUT_BASE = "/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies"
OUTPUT_BASE = os.environ.get("GPR_FUNCFORM_OUTPUT_BASE", DEFAULT_OUTPUT_BASE).rstrip("/")
KERNEL_LOCK_TABLE_DIR = os.environ.get(
    "GPR_KERNEL_LOCK_TABLE_DIR",
    f"{OUTPUT_BASE}/kernel_lock_tables",
).rstrip("/")


@dataclass(frozen=True)
class ClosureStudy:
    blind_nsigma: float
    train_exclude_nsigma: float
    lslb: float
    role: str
    inj_shape_mode: str = "full"
    refit_gp_on_toy: bool = True
    kernel_lock_mode: str = "none"
    kernel_lock_file: str = ""
    sigma_ref_mode: str = "prefit_asimov"
    tail_alpha_scale: float = 0.0
    tail_alpha_threshold: float = 0.0
    tag_override: str = ""

    @property
    def tag(self) -> str:
        if self.tag_override:
            return str(self.tag_override)
        parts = [
            f"blind{tag_float(self.blind_nsigma)}",
            f"train{tag_float(self.train_exclude_nsigma)}",
            f"lslb{tag_float(self.lslb)}",
            self.role,
        ]
        if self.inj_shape_mode != "full":
            parts.append(self.inj_shape_mode)
        if not self.refit_gp_on_toy:
            parts.append("no_refit")
        if self.kernel_lock_mode != "none":
            parts.append(f"lock_{self.kernel_lock_mode}")
        if self.tail_alpha_scale > 0.0:
            parts.append("tail_alpha")
        return "_".join(parts)

    @property
    def output_dir(self) -> str:
        return f"{OUTPUT_BASE}/{self.tag}"

    @property
    def config_name(self) -> str:
        return f"config_2015_{self.tag}_95CL_funcform100_fixedhist.yaml"


def tag_float(value: float) -> str:
    value = float(value)
    text = f"{value:.1f}" if value.is_integer() else f"{value:g}"
    return text.replace(".", "p")


def guard_tag(train_nsigma: float) -> str:
    labels = {
        1.96: "1p96",
        2.25: "2p25",
        2.50: "2p50",
        2.58: "2p58",
        2.75: "2p75",
        3.00: "3p0",
    }
    return labels.get(float(train_nsigma), tag_float(float(train_nsigma)))


def build_studies() -> list[ClosureStudy]:
    studies: list[ClosureStudy] = []

    # Current baselines: training exclusion equals the extraction blind width.
    for blind_nsigma in BLIND_WIDTHS:
        for lslb in (0.5, 1.0):
            studies.append(
                ClosureStudy(
                    blind_nsigma=blind_nsigma,
                    train_exclude_nsigma=blind_nsigma,
                    lslb=lslb,
                    role="baseline",
                )
            )

    # Primary guard-band candidates plus a stiffer robustness point.
    for blind_nsigma in BLIND_WIDTHS:
        for train_nsigma in GUARD_WIDTHS:
            for lslb in (1.0, 1.5):
                role = "primary" if lslb == 1.0 else "robustness"
                studies.append(
                    ClosureStudy(
                        blind_nsigma=blind_nsigma,
                        train_exclude_nsigma=train_nsigma,
                        lslb=lslb,
                        role=role,
                    )
                )

    # Minimal controls at the preferred 3 sigma guard.
    for blind_nsigma in BLIND_WIDTHS:
        studies.append(
            ClosureStudy(
                blind_nsigma=blind_nsigma,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="control",
                refit_gp_on_toy=False,
            )
        )
        studies.append(
            ClosureStudy(
                blind_nsigma=blind_nsigma,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="control",
                inj_shape_mode="window",
            )
        )

    # Refit-matched sigmaA reference rescue matrix, focused on 2015 lslb=1.0.
    for train_nsigma in (1.96, 2.25, 2.50, 2.58, 2.75, 3.0):
        studies.append(
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=float(train_nsigma),
                lslb=1.0,
                role="guard_refmatched",
                sigma_ref_mode="matched_refit_bonly",
                tag_override=f"blind1p64_train{guard_tag(float(train_nsigma))}_lslb1p0_guard_refmatched",
            )
        )

    studies.extend(
        [
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="prefit_reference_control",
                sigma_ref_mode="prefit_asimov",
                tag_override="blind1p64_train3p0_lslb1p0_prefit_reference_control",
            ),
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="no_refit_control",
                refit_gp_on_toy=False,
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_no_refit_control",
            ),
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="window_only_control",
                inj_shape_mode="window",
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_window_only_control",
            ),
        ]
    )

    lock_files = {
        "ensemble_p50_lock_refmatched": f"{KERNEL_LOCK_TABLE_DIR}/kernel_lock_p50_crossfit.csv",
        "ensemble_p75ls_lock_refmatched": f"{KERNEL_LOCK_TABLE_DIR}/kernel_lock_p75ls_crossfit.csv",
        "ensemble_p25ls_lock_refmatched": f"{KERNEL_LOCK_TABLE_DIR}/kernel_lock_p25ls_crossfit.csv",
    }
    studies.extend(
        [
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="none_refmatched",
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_none_refmatched",
            ),
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="initial_fit_lock_refmatched",
                kernel_lock_mode="initial_fit",
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_initial_fit_lock_refmatched",
            ),
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="ensemble_p50_lock_refmatched",
                kernel_lock_mode="ensemble_file",
                kernel_lock_file=lock_files["ensemble_p50_lock_refmatched"],
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_ensemble_p50_lock_refmatched",
            ),
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="ensemble_p75ls_lock_refmatched",
                kernel_lock_mode="ensemble_file",
                kernel_lock_file=lock_files["ensemble_p75ls_lock_refmatched"],
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_ensemble_p75ls_lock_refmatched",
            ),
            ClosureStudy(
                blind_nsigma=1.64,
                train_exclude_nsigma=3.0,
                lslb=1.0,
                role="ensemble_p25ls_lock_refmatched",
                kernel_lock_mode="ensemble_file",
                kernel_lock_file=lock_files["ensemble_p25ls_lock_refmatched"],
                sigma_ref_mode="matched_refit_bonly",
                tag_override="blind1p64_train3p0_lslb1p0_ensemble_p25ls_lock_refmatched",
            ),
        ]
    )

    return studies


def build_config(base: dict, study: ClosureStudy) -> dict:
    cfg = deepcopy(base)
    cfg.update(
        {
            "blind_nsigma": float(study.blind_nsigma),
            "gp_train_exclude_nsigma": float(study.train_exclude_nsigma),
            "inj_train_exclude_nsigma": float(study.train_exclude_nsigma),
            "ul_bands_train_exclude_nsigma": float(study.train_exclude_nsigma),
            "extraction_display_train_exclude_nsigma": float(study.train_exclude_nsigma),
            "kernel_ls_res_lower_factor": float(study.lslb),
            "kernel_ls_res_lower_factor_by_dataset": {
                "2015": float(study.lslb),
                "2016": 0.5,
                "2021": 0.5,
            },
            "inject_signal": True,
            "inj_dataset_key": "2015",
            "inj_masses_gev": list(MASS_GRID),
            "inj_mode": "poisson",
            "inj_strength_mode": "sigmaA",
            "inj_sigma_a_source": "asimov",
            "inj_sigma_a_ref_mode": str(study.sigma_ref_mode),
            "inj_shape_mode": str(study.inj_shape_mode),
            "inj_background_mode": "fixed_hist",
            "inj_refit_gp_on_toy": bool(study.refit_gp_on_toy),
            "inj_refit_gp_restarts": 0,
            "inj_refit_gp_optimize": True,
            "inj_refit_kernel_lock_mode": str(study.kernel_lock_mode),
            "inj_refit_kernel_lock_file": str(study.kernel_lock_file),
            "inj_refit_signal_tail_alpha_scale": float(study.tail_alpha_scale),
            "inj_refit_signal_tail_alpha_threshold": float(study.tail_alpha_threshold),
            "inj_sigma_multipliers": list(SIGMA_MULTIPLIERS),
            "inj_write_toy_csv": True,
            "inj_write_qmu": True,
            "inj_stream_aggregate": True,
            "inj_aggregate_every": 100,
            "inj_n_workers": 1,
            "inj_threads_per_worker": 1,
            "signal_model": "default",
            "output_dir": study.output_dir,
        }
    )
    return cfg


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    base = yaml.safe_load(BASE_CONFIG.read_text()) or {}

    for study in build_studies():
        cfg = build_config(base, study)
        out = HERE / study.config_name
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"Wrote {out.relative_to(STUDY_CONFIGS.parent)}")


if __name__ == "__main__":
    main()
