#!/usr/bin/env python3
"""Exploratory mod_2 nuisance-profile and kernel-bound study.

This is a deliberately compact follow-up to
``study_results/funcform_mod2_highstat_20260630``.  It runs ten functional-form
source toys for each dataset/root file and compares:

* fixed-background extraction (no Gaussian nuisance profiling),
* the current profiled Gaussian-nuisance extraction used by HPS-GPR/Majd_phi,
* a Majd_phi-style multiplicative beta nuisance profile,
* length-scale upper-bound stress rows, and
* tighter ConstantKernel amplitude bounds.

The epsilon quantities are extraction proxies, not exact CLs limits:

    eps2_95_exp_proxy = 1.6448536269514722 * sigma_A / A_per_eps2_unit
    eps2_95_obs_proxy = (max(A_hat, 0) + 1.6448536269514722 * sigma_A) / A_per_eps2_unit

They are included because epsilon^2 reach loss is a publication-level risk if
it removes novel parameter space after unblinding.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


STUDY_DIR = Path(__file__).resolve().parent
RUNNER_REPO = Path("/Users/emryspeets/Desktop/gp_mods/tmp_hps_gpr_funcform_origin_main")
INPUT_DIR = Path("/Users/emryspeets/Desktop/gp_mods/funcform_studies/func_form_inputs")
CONFIG_DIR = STUDY_DIR / "configs"
RUN_DIR = STUDY_DIR / "runs"
PLOT_DIR = STUDY_DIR / "plots"
LOG_DIR = STUDY_DIR / "logs"

STRENGTHS = "s0,s1,s2,s3,s5"
Z95 = 1.6448536269514722


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    profiled_config: str
    fixed_config: str
    toy_root: Path
    container: str
    toy_name_fmt: str
    masses: list[float]
    best_lslb: float
    default_upper: float


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    mode: str
    ls_upper: float | None = None
    const_bounds: tuple[float, float] | None = None
    role: str = "stress"


DATASETS: dict[str, DatasetSpec] = {
    "2015": DatasetSpec(
        key="2015",
        profiled_config="config_2015_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
        fixed_config="config_2015_blind2p25_fixedextract_95CL_funcform100_fixedhist_refit_lslb1p0.yaml",
        toy_root=INPUT_DIR / "funcform_2015_dataset_mod_toys_2.root",
        container="fShiftSigPowTail",
        toy_name_fmt="fShiftSigPowTail_toy_{i}",
        masses=[0.045, 0.075, 0.105],
        best_lslb=1.1,
        default_upper=8.0,
    ),
    "2016": DatasetSpec(
        key="2016",
        profiled_config="config_2016_10pct_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
        fixed_config="config_2016_10pct_blind2p25_fixedextract_95CL_funcform100_fixedhist_refit_lslb1p0.yaml",
        toy_root=INPUT_DIR / "funcform_2016_dataset_mod_toys_2.root",
        container="fShiftSigPowTail",
        toy_name_fmt="fShiftSigPowTail_toy_{i}",
        masses=[0.060, 0.105, 0.150],
        best_lslb=0.9,
        default_upper=8.0,
    ),
    "2021": DatasetSpec(
        key="2021",
        profiled_config="config_2021_1pct_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
        fixed_config="config_2021_1pct_blind2p25_fixedextract_95CL_funcform100_fixedhist_refit_lslb1p0.yaml",
        toy_root=INPUT_DIR / "funcform_2021_dataset_mod_toys_2.root",
        container="fSigPowExpQ",
        toy_name_fmt="fSigPowExpQ_toy_{i}",
        masses=[0.060, 0.105, 0.220],
        best_lslb=1.1,
        default_upper=9.0,
    ),
}


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        key="fixed_no_nuisance",
        label="fixed bkg, no nuisance profile",
        mode="fixed",
        role="no_profile_baseline",
    ),
    VariantSpec(
        key="profiled_nominal",
        label="profiled nuisance, nominal bounds",
        mode="profiled",
        role="profile_nominal",
    ),
    VariantSpec(
        key="majd_phi_beta_profiled",
        label="Majd_phi beta-basis profile",
        mode="beta_profiled",
        role="majd_phi_profile_stress",
    ),
    VariantSpec(
        key="profiled_lsub6",
        label="profiled nuisance, ls upper=6 sigma",
        mode="profiled",
        ls_upper=6.0,
        role="upper_bound_stress",
    ),
    VariantSpec(
        key="profiled_lsub12",
        label="profiled nuisance, ls upper=12 sigma",
        mode="profiled",
        ls_upper=12.0,
        role="upper_bound_stress",
    ),
    VariantSpec(
        key="profiled_const1em4_1e4",
        label="profiled nuisance, const=[1e-4,1e4]",
        mode="profiled",
        const_bounds=(1.0e-4, 1.0e4),
        role="constant_bound_stress",
    ),
    VariantSpec(
        key="profiled_const1em2_1e2",
        label="profiled nuisance, const=[1e-2,1e2]",
        mode="profiled",
        const_bounds=(1.0e-2, 1.0e2),
        role="constant_bound_stress",
    ),
]


def tag_float(value: float) -> str:
    if float(value).is_integer():
        text = f"{float(value):.1f}"
    else:
        text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return payload


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def set_dataset_enable(cfg: dict, dataset: str) -> None:
    for key in ("2015", "2016", "2021"):
        cfg[f"enable_{key}"] = key == dataset
    cfg["inj_dataset_key"] = dataset
    cfg["run_limit_bands_on"] = dataset


def set_ls_factors(cfg: dict, dataset: str, lower: float, upper: float) -> None:
    cfg["kernel_ls_res_lower_factor"] = float(lower)
    cfg["kernel_ls_res_upper_factor"] = float(upper)
    lo_by_ds = dict(cfg.get("kernel_ls_res_lower_factor_by_dataset") or {})
    hi_by_ds = dict(cfg.get("kernel_ls_res_upper_factor_by_dataset") or {})
    for key in ("2015", "2016", "2021"):
        lo_by_ds.setdefault(key, 0.5)
        hi_by_ds.setdefault(key, 9.0 if key == "2021" else 8.0)
    lo_by_ds[dataset] = float(lower)
    hi_by_ds[dataset] = float(upper)
    cfg["kernel_ls_res_lower_factor_by_dataset"] = lo_by_ds
    cfg["kernel_ls_res_upper_factor_by_dataset"] = hi_by_ds


def config_for(dataset: str, variant: VariantSpec) -> Path:
    spec = DATASETS[dataset]
    base_name = spec.profiled_config if variant.mode in {"profiled", "beta_profiled"} else spec.fixed_config
    base_path = RUNNER_REPO / "study_configs" / "funcform_pullwidth_diagnostics" / base_name
    cfg = copy.deepcopy(load_yaml(base_path))

    set_dataset_enable(cfg, dataset)
    ls_upper = float(variant.ls_upper if variant.ls_upper is not None else spec.default_upper)
    set_ls_factors(cfg, dataset, lower=spec.best_lslb, upper=ls_upper)

    # The beta-profile row uses the nominal HPS profiled mode for sigmaA strength
    # scaling, then swaps only the final amplitude extraction in this script.
    cfg["extract_background_mode"] = "profiled" if variant.mode == "beta_profiled" else str(variant.mode)
    cfg["funcform_closure_root_by_dataset"] = {k: str(v.toy_root) for k, v in DATASETS.items()}
    cfg["funcform_closure_container_by_dataset"] = {k: v.container for k, v in DATASETS.items()}
    cfg["funcform_closure_toy_pattern_by_dataset"] = {
        k: v.toy_name_fmt.replace("{i}", "*") for k, v in DATASETS.items()
    }
    if variant.const_bounds is not None:
        cfg["kernel_constant_bounds"] = [float(variant.const_bounds[0]), float(variant.const_bounds[1])]

    cfg["output_dir"] = str(RUN_DIR / variant.key / dataset)
    cfg["save_plots"] = False
    cfg["inj_n_workers"] = 1
    cfg["inj_threads_per_worker"] = 1
    cfg["inj_write_qmu"] = True

    path = CONFIG_DIR / f"config_{dataset}_{variant.key}.yaml"
    dump_yaml(path, cfg)
    return path


def toy_indices(max_toys: int, toy_start: int = 0) -> list[int]:
    if int(max_toys) <= 0:
        raise ValueError("--max-toys must be positive")
    return list(range(int(toy_start), int(toy_start) + int(max_toys)))


def run_one(dataset: str, variant: VariantSpec, *, indices: list[int], force: bool) -> Path:
    spec = DATASETS[dataset]
    outdir = RUN_DIR / variant.key / dataset
    csv_path = outdir / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {dataset} {variant.key}: {csv_path}", flush=True)
        return csv_path

    cfg_path = config_for(dataset, variant)
    cmd = [
        sys.executable,
        "-m",
        "hps_gpr.cli",
        "funcform-inject",
        "--config",
        str(cfg_path),
        "--dataset",
        dataset,
        "--toy-root",
        str(spec.toy_root),
        "--container",
        spec.container,
        "--toy-name-fmt",
        spec.toy_name_fmt,
        "--masses",
        ",".join(f"{m:.3f}" for m in spec.masses),
        "--strengths",
        STRENGTHS,
        "--n-injection-toys",
        "1",
        "--output-dir",
        str(outdir),
        "--write-toy-csv",
        "--write-qmu",
    ]
    for idx in indices:
        cmd.extend(["--toy-index", str(idx)])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{dataset}_{variant.key}.log"
    print(
        f"[run] {dataset} {variant.key}: masses={len(spec.masses)} toys={len(indices)} log={log_path}",
        flush=True,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(str(x) for x in cmd) + "\n")
        log.flush()
        try:
            subprocess.run(cmd, cwd=RUNNER_REPO, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        except subprocess.CalledProcessError:
            print(f"[fail] {dataset} {variant.key}; see {log_path}", flush=True)
            raise
    print(f"[done] {dataset} {variant.key}", flush=True)
    return csv_path


def ensure_runner_importable() -> None:
    root = str(RUNNER_REPO)
    if root not in sys.path:
        sys.path.insert(0, root)


def fit_A_majd_phi_beta_profiled(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    x_win: np.ndarray,
    mass: float,
    *,
    allow_negative: bool,
) -> dict[str, Any]:
    """Majd_phi-style multiplicative beta nuisance profile in the blind window."""
    from scipy.optimize import least_squares

    obs = np.asarray(n_obs, float).reshape(-1)
    gp_mu = np.clip(np.asarray(b_mean, float).reshape(-1), 0.0, None)
    tmpl = np.asarray(template, float).reshape(-1)
    cov = np.asarray(b_cov, float)
    if cov.ndim == 2:
        gp_std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        gp_std = np.sqrt(np.clip(cov.reshape(-1), 0.0, None))

    x_arr = np.asarray(x_win, float).reshape(-1)
    x_mev = 1000.0 * x_arr if np.nanmax(np.abs(x_arr)) < 10.0 else x_arr
    center_mev = 1000.0 * float(mass) if abs(float(mass)) < 10.0 else float(mass)
    z = (x_mev - center_mev) / 50.0
    z2 = z * z - float(np.mean(z * z)) if z.size else z
    priors = np.asarray([0.05, 0.10, 0.08], dtype=float)

    template_sum = max(float(np.sum(tmpl)), 1.0e-9)
    raw_excess = float(np.sum(obs - gp_mu))
    a0 = raw_excess / template_sum
    if not bool(allow_negative):
        a0 = max(a0, 0.0)
    scale_counts = max(float(np.sum(np.clip(obs, 0.0, None))), float(np.sum(gp_mu)), abs(a0) * template_sum, 1.0)
    a_hi = max(1.5 * scale_counts, abs(a0) * 4.0 + 1000.0, 1.0)
    a_lo = -a_hi if bool(allow_negative) else 0.0
    a0 = float(np.clip(a0, a_lo + 1.0e-9, a_hi - 1.0e-9))

    def model_from_params(pars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        amp = float(pars[0])
        beta = np.asarray(pars[1:4], float)
        log_shift = beta[0] + beta[1] * z + beta[2] * z2
        bkg = gp_mu * np.exp(np.clip(log_shift, -0.7, 0.7))
        total = bkg + amp * tmpl
        return bkg, total

    def residuals(pars: np.ndarray) -> np.ndarray:
        _bkg, total = model_from_params(pars)
        variance = np.maximum(total, 1.0) + np.square(gp_std)
        resid = (obs - total) / np.sqrt(np.clip(variance, 1.0, None))
        return np.concatenate([resid, np.asarray(pars[1:4], float) / priors])

    x0 = np.asarray([a0, 0.0, 0.0, 0.0], dtype=float)
    lower = np.asarray([a_lo, -0.30, -0.45, -0.45], dtype=float)
    upper = np.asarray([a_hi, 0.30, 0.45, 0.45], dtype=float)
    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        loss="linear",
        max_nfev=800,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    bkg, total = model_from_params(result.x)
    n_free = int(result.x.size)
    sigma_A = float("nan")
    if result.jac.size:
        try:
            jtj_inv = np.linalg.pinv(result.jac.T @ result.jac)
            scale = max(2.0 * float(result.cost) / max(int(result.fun.size) - n_free, 1), 1.0)
            sigma_A = math.sqrt(max(float(jtj_inv[0, 0]) * scale, 0.0))
        except Exception:
            sigma_A = float("nan")
    beta = np.asarray(result.x[1:4], float)
    nuisance_pull = float(np.sqrt(np.sum(np.square(beta / priors))))
    profile_delta = bkg - gp_mu
    obs_counts = float(np.sum(obs))
    return {
        "A_hat": float(result.x[0]),
        "sigma_A": float(sigma_A),
        "success": bool(result.success) and np.isfinite(sigma_A) and sigma_A > 0.0,
        "nll": float(0.5 * np.sum(np.square(result.fun))),
        "optimizer_message": str(result.message),
        "nuisance_beta0": float(beta[0]),
        "nuisance_beta1": float(beta[1]),
        "nuisance_beta2": float(beta[2]),
        "nuisance_pull_norm": float(nuisance_pull),
        "profiled_background_shift_blind_counts": float(np.sum(profile_delta)),
        "profiled_background_abs_shift_blind_counts": float(np.sum(np.abs(profile_delta))),
        "profile_shift_abs_fraction_observed": float(np.sum(np.abs(profile_delta)) / max(obs_counts, 1.0)),
        "profiled_total_blind_counts": float(np.sum(total)),
        "gpr_background_blind_counts": float(np.sum(gp_mu)),
    }


def simulate_beta_toy_rows(
    ctx: Any,
    config: Any,
    *,
    toy_indices: list[int],
    A_inj: float,
    inj_nsigma: float,
    point_seed: int,
    threads_per_worker: int,
) -> list[dict[str, Any]]:
    ensure_runner_importable()
    from hps_gpr.gpr import fit_gpr, make_kernel_for_dataset, predict_counts_from_log_gpr
    from hps_gpr.injection import (
        _dataset_source_metadata,
        _fixed_hist_background_counts,
        _gpr_fit_diagnostics,
        _handle_refit_failure,
        _inject_counts_from_template,
        _kernel_for_refit,
        _stable_toy_seed,
        draw_bkg_mvn_nonneg,
    )

    if not toy_indices:
        return []
    out_rows: list[dict[str, Any]] = []
    toy_mode = "full_refit" if bool(ctx.refit_gp_on_toy) else (
        "fixed_hist_no_refit" if str(ctx.inj_background_mode) == "fixed_hist" else "conditional_gp"
    )
    ker = make_kernel_for_dataset(ctx.ds, config, mass=float(ctx.mass)) if bool(ctx.refit_gp_on_toy) else None
    x_win = np.asarray(ctx.x_full, float)[np.asarray(ctx.msk_blind, bool)]

    for toy_idx in toy_indices:
        rng = np.random.default_rng(_stable_toy_seed(int(point_seed), int(toy_idx)))
        refit_ok = float("nan")
        refit_fallback_used = False
        refit_error = ""
        refit_diag = dict(refit_ls_opt=float("nan"), refit_const_opt=float("nan"))

        if bool(ctx.refit_gp_on_toy):
            if str(ctx.inj_background_mode) == "fixed_hist":
                bkg_full = _fixed_hist_background_counts(ctx.y_full, dataset_key=str(ctx.ds.key), mass=float(ctx.mass))
            else:
                bkg_full = rng.poisson(np.clip(ctx.mu_full, 0.0, None)).astype(int)
            if str(ctx.inj_shape_mode) == "window":
                sig_full = np.zeros_like(bkg_full, dtype=int)
                s_win, Nsig_win, _ = _inject_counts_from_template(ctx.tmpl_win, A_inj, rng, ctx.inj_mode)
                idx_blind = np.where(np.asarray(ctx.msk_blind, bool))[0]
                n = min(len(s_win), len(idx_blind))
                sig_full[idx_blind[:n]] = s_win[:n]
            else:
                s_full, _, _ = _inject_counts_from_template(ctx.tmpl_full, A_inj, rng, ctx.inj_mode)
                sig_full = np.asarray(s_full, dtype=int)
                Nsig_win = int(np.sum(sig_full[np.asarray(ctx.msk_blind, bool)]))

            y_toy = (bkg_full + sig_full).astype(int)
            obs = y_toy[np.asarray(ctx.msk_blind, bool)].astype(int)
            Nsig_train = int(np.sum(sig_full[np.asarray(ctx.msk_train, bool)]))
            mu_fit = np.asarray(ctx.mu, float)
            cov_fit = np.asarray(ctx.cov, float)
            try:
                X_tr = np.asarray(ctx.x_full, float)[np.asarray(ctx.msk_train, bool)]
                y_tr = y_toy[np.asarray(ctx.msk_train, bool)].astype(float)
                ker_fit, optimize_allowed = _kernel_for_refit(
                    base_kernel=ker,
                    lock_mode=str(ctx.refit_kernel_lock_mode),
                    lock_const_opt=float(ctx.refit_lock_const_opt),
                    lock_ls_opt=float(ctx.refit_lock_ls_opt),
                )
                fit_kwargs: dict[str, Any] = dict(
                    restarts=int(ctx.refit_restarts),
                    kernel=ker_fit,
                    optimize=bool(ctx.refit_optimize) and bool(optimize_allowed),
                )
                if ctx.refit_tail_alpha_multiplier is not None:
                    fit_kwargs["alpha_multiplier"] = ctx.refit_tail_alpha_multiplier
                gpr = fit_gpr(X_tr, y_tr, config, **fit_kwargs)
                mu_fit, cov_fit = predict_counts_from_log_gpr(gpr, x_win, config)
                diag = _gpr_fit_diagnostics(gpr)
                refit_diag.update(refit_ls_opt=float(diag["ls_opt"]), refit_const_opt=float(diag["const_opt"]))
                refit_ok = 1.0
            except Exception as exc:
                refit_ok = 0.0
                refit_fallback_used = True
                refit_error = _handle_refit_failure(
                    config,
                    f"{ctx.ds.key} m={float(ctx.mass):.6g} toy={int(toy_idx)} A={float(A_inj):.6g}",
                    exc,
                )
        else:
            if str(ctx.inj_background_mode) == "fixed_hist":
                bkg_full = _fixed_hist_background_counts(ctx.y_full, dataset_key=str(ctx.ds.key), mass=float(ctx.mass))
                if str(ctx.inj_shape_mode) == "window":
                    sig_full = np.zeros_like(bkg_full, dtype=int)
                    sig, Nsig_win, _ = _inject_counts_from_template(ctx.tmpl_win, A_inj, rng, ctx.inj_mode)
                    idx_blind = np.where(np.asarray(ctx.msk_blind, bool))[0]
                    n = min(len(sig), len(idx_blind))
                    sig_full[idx_blind[:n]] = sig[:n]
                else:
                    s_full, _, _ = _inject_counts_from_template(ctx.tmpl_full, A_inj, rng, ctx.inj_mode)
                    sig_full = np.asarray(s_full, dtype=int)
                    Nsig_win = int(np.sum(sig_full[np.asarray(ctx.msk_blind, bool)]))
                obs = (bkg_full + sig_full)[np.asarray(ctx.msk_blind, bool)].astype(int)
                Nsig_train = int(np.sum(sig_full[np.asarray(ctx.msk_train, bool)]))
            else:
                b = draw_bkg_mvn_nonneg(
                    ctx.mu,
                    ctx.cov,
                    1,
                    rng,
                    method=str(ctx.mvn_method),
                    max_tries=int(ctx.mvn_max_tries),
                )[0]
                sig, Nsig_win, _ = _inject_counts_from_template(ctx.tmpl_win, A_inj, rng, ctx.inj_mode)
                lam = np.clip(b, 0.0, None) + np.clip(sig.astype(float), 0.0, None)
                obs = rng.poisson(lam).astype(int)
                Nsig_train = 0
            mu_fit, cov_fit = ctx.mu, ctx.cov

        ls_opt_effective = (
            float(refit_diag["refit_ls_opt"])
            if bool(ctx.refit_gp_on_toy) and refit_ok == 1.0
            else float(ctx.initial_ls_opt)
        )
        const_opt_effective = (
            float(refit_diag["refit_const_opt"])
            if bool(ctx.refit_gp_on_toy) and refit_ok == 1.0
            else float(ctx.initial_const_opt)
        )

        fit = fit_A_majd_phi_beta_profiled(
            obs,
            np.asarray(mu_fit, float),
            np.asarray(cov_fit, float),
            np.asarray(ctx.tmpl_win, float),
            x_win,
            float(ctx.mass),
            allow_negative=bool(ctx.allow_negative),
        )
        A_hat = float(fit["A_hat"])
        sigma_A = float(fit["sigma_A"])
        pull = (A_hat - float(A_inj)) / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")
        Zhat = A_hat / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")

        row = dict(
            dataset=str(ctx.ds.key),
            mass_GeV=float(ctx.mass),
            toy=int(toy_idx),
            strength=float(A_inj),
            inj_nsigma=float(inj_nsigma),
            sigmaA_ref=float(ctx.sigmaA_ref),
            sigmaA_ref_prefit=float(ctx.sigmaA_ref_prefit),
            sigmaA_ref_matched=float(ctx.sigmaA_ref_matched),
            sigmaA_ref_mode=str(ctx.sigmaA_ref_mode),
            sigmaA_ref_matched_ok=float(ctx.sigmaA_ref_matched_ok),
            sigmaA_ref_error=str(ctx.sigmaA_ref_error),
            integral_density=float(ctx.integral_density),
            A_per_eps2_unit=float(ctx.A_per_eps2_unit),
            sigma_val=float(ctx.sigma_val),
            sigma_x=float(ctx.sigma_x),
            kernel_ls_policy=str(ctx.kernel_ls_policy),
            kernel_ls_res_lower_factor=float(ctx.kernel_ls_res_lower_factor),
            kernel_ls_res_upper_factor=float(ctx.kernel_ls_res_upper_factor),
            ls_lo=float(ctx.ls_lo),
            ls_hi=float(ctx.ls_hi),
            ls_init=float(ctx.ls_init),
            initial_ls_opt=float(ctx.initial_ls_opt),
            initial_const_opt=float(ctx.initial_const_opt),
            refit_ls_opt=float(refit_diag["refit_ls_opt"]),
            refit_const_opt=float(refit_diag["refit_const_opt"]),
            ls_opt=float(ls_opt_effective),
            const_opt=float(const_opt_effective),
            f_win=float(ctx.f_win),
            f_full=float(ctx.f_full),
            f_train=float(ctx.f_train),
            f_train_frac=float(ctx.f_train_frac),
            n_train=int(ctx.n_train),
            n_train_low=int(ctx.n_train_low),
            n_train_high=int(ctx.n_train_high),
            n_blind=int(ctx.n_blind),
            blind_nsigma=float(ctx.blind_nsigma),
            train_exclude_nsigma=float(ctx.train_exclude_nsigma),
            signal_model=str(ctx.signal_model),
            inj_shape_mode=str(ctx.inj_shape_mode),
            inj_background_mode=str(ctx.inj_background_mode),
            extract_background_mode="majd_phi_beta_profiled",
            A_hat=float(A_hat),
            sigma_A=float(sigma_A),
            Zhat=float(Zhat),
            pull_param=float(pull),
            Nsig_win=int(Nsig_win),
            Nsig_train=int(Nsig_train),
            success=bool(fit["success"]),
            nll=float(fit.get("nll", float("nan"))),
            toy_mode=str(toy_mode),
            refit_gp_on_toy=bool(ctx.refit_gp_on_toy),
            refit_ok=float(refit_ok),
            refit_restarts=int(ctx.refit_restarts),
            refit_optimize=bool(ctx.refit_optimize),
            refit_kernel_lock_mode=str(ctx.refit_kernel_lock_mode),
            refit_lock_const_opt=float(ctx.refit_lock_const_opt),
            refit_lock_ls_opt=float(ctx.refit_lock_ls_opt),
            refit_tail_alpha_scale=float(ctx.refit_tail_alpha_scale),
            refit_tail_alpha_threshold=float(ctx.refit_tail_alpha_threshold),
            refit_tail_alpha_n_bins=int(ctx.refit_tail_alpha_n_bins),
            refit_tail_alpha_max=float(ctx.refit_tail_alpha_max),
            refit_tail_alpha_mean=float(ctx.refit_tail_alpha_mean),
            refit_fallback_used=bool(refit_fallback_used),
            refit_error=str(refit_error),
        )
        for key, value in fit.items():
            if key not in row and key != "optimizer_message":
                row[key] = value
        row.update(_dataset_source_metadata(ctx.ds))
        out_rows.append(row)

    return out_rows


def run_one_beta(dataset: str, variant: VariantSpec, *, indices: list[int], force: bool) -> Path:
    ensure_runner_importable()
    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import build_funcform_toy_dataset, discover_funcform_toys, load_funcform_toy_hist
    from hps_gpr.injection import (
        _build_injection_mass_context,
        _resolve_inj_background_mode,
        _resolve_injection_strength_tags,
        _stable_point_seed,
    )

    spec = DATASETS[dataset]
    outdir = RUN_DIR / variant.key / dataset
    csv_path = outdir / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {dataset} {variant.key}: {csv_path}", flush=True)
        return csv_path

    cfg_path = config_for(dataset, variant)
    cfg = load_config(str(cfg_path))
    cfg.output_dir = str(outdir)
    datasets = make_datasets(cfg)
    if dataset not in datasets:
        raise KeyError(f"Dataset {dataset!r} was not enabled by {cfg_path}")
    ds = datasets[dataset]

    inj_mode = str(getattr(cfg, "inj_mode", "poisson")).lower().strip()
    sigma_source = str(getattr(cfg, "inj_sigma_a_source", "asimov")).lower().strip()
    refit_gp_on_toy = bool(getattr(cfg, "inj_refit_gp_on_toy", False))
    refit_restarts = int(getattr(cfg, "inj_refit_gp_restarts", 0))
    refit_optimize = bool(getattr(cfg, "inj_refit_gp_optimize", True))
    inj_shape_mode = str(getattr(cfg, "inj_shape_mode", "full")).lower().strip()
    if inj_shape_mode not in ("full", "window"):
        inj_shape_mode = "full"
    train_exclude_nsigma = getattr(cfg, "inj_train_exclude_nsigma", None)
    inj_background_mode = _resolve_inj_background_mode(cfg, refit_gp_on_toy=bool(refit_gp_on_toy))
    mvn_method = str(getattr(cfg, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(cfg, "mvn_trunc_max_tries", 80))
    strengths_mode, strength_tags = _resolve_injection_strength_tags(
        config=cfg,
        strengths=[0.0, 1.0, 2.0, 3.0, 5.0],
        strengths_mode="sigmaa",
    )

    specs = discover_funcform_toys(
        str(spec.toy_root),
        container=spec.container,
        toy_name_fmt=spec.toy_name_fmt,
        toy_indices=indices,
    )
    out_rows: list[dict[str, Any]] = []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{dataset}_{variant.key}.log"
    print(
        f"[run] {dataset} {variant.key}: masses={len(spec.masses)} toys={len(indices)} log={log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[beta-profile] config={cfg_path}\n")
        for toy_spec in specs:
            toy_hist = load_funcform_toy_hist(
                toy_spec.source_root,
                container=(toy_spec.container or None),
                toy_name=toy_spec.toy_name,
            )
            toy_ds = build_funcform_toy_dataset(ds, toy_hist, toy_spec)
            seed = 314159 + int(toy_spec.toy_index)
            print(
                "[funcform-beta] "
                f"{dataset} toy={toy_spec.toy_name} index={int(toy_spec.toy_index)} "
                f"source={toy_spec.source_root}:{toy_spec.container}",
                file=log,
                flush=True,
            )
            for mass in [float(x) for x in spec.masses]:
                ctx = _build_injection_mass_context(
                    toy_ds,
                    cfg,
                    mass=float(mass),
                    seed=int(seed),
                    inj_mode=str(inj_mode),
                    sigma_source=str(sigma_source),
                    refit_gp_on_toy=bool(refit_gp_on_toy),
                    refit_restarts=int(refit_restarts),
                    refit_optimize=bool(refit_optimize),
                    inj_shape_mode=str(inj_shape_mode),
                    inj_background_mode=str(inj_background_mode),
                    train_exclude_nsigma=train_exclude_nsigma,
                    mvn_method=str(mvn_method),
                    mvn_max_tries=int(mvn_max_tries),
                )
                if strengths_mode == "sigmaa":
                    A_inj_list = [float(t) * float(ctx.sigmaA_ref) for t in strength_tags]
                    inj_nsigma_list = [float(t) for t in strength_tags]
                else:
                    A_inj_list = [float(t) for t in strength_tags]
                    inj_nsigma_list = [
                        float(A) / float(ctx.sigmaA_ref) if np.isfinite(ctx.sigmaA_ref) and ctx.sigmaA_ref > 0 else float("nan")
                        for A in A_inj_list
                    ]

                for A_inj, inj_nsigma in zip(A_inj_list, inj_nsigma_list):
                    point_seed = _stable_point_seed(int(seed), str(dataset), float(mass), float(A_inj))
                    rows = simulate_beta_toy_rows(
                        ctx,
                        cfg,
                        toy_indices=[0],
                        A_inj=float(A_inj),
                        inj_nsigma=float(inj_nsigma),
                        point_seed=int(point_seed),
                        threads_per_worker=1,
                    )
                    for row in rows:
                        injection_toy = int(row.get("toy", 0))
                        row["injection_toy"] = int(injection_toy)
                        row["toy"] = int(toy_spec.toy_index) * 1_000_000 + int(injection_toy)
                        row["toy_index"] = int(toy_spec.toy_index)
                        row["toy_hist"] = str(toy_spec.toy_name)
                        row["function_tag"] = str(toy_spec.function_tag)
                        row["source_model"] = "functional_form"
                        row["source_label"] = str(toy_spec.function_tag)
                        row["source_root"] = str(toy_spec.source_root)
                        row["container"] = str(toy_spec.container)
                        out_rows.append(row)
                    print(
                        "[beta][stream] "
                        f"{dataset} m={float(mass):.6g} GeV strength={float(A_inj):.6g}: 1/1 toys",
                        file=log,
                        flush=True,
                    )

    if not out_rows:
        raise RuntimeError(f"No rows produced for {dataset} {variant.key}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(csv_path, index=False)
    print(f"[done] {dataset} {variant.key}", flush=True)
    return csv_path


def run_variant(dataset: str, variant: VariantSpec, *, indices: list[int], force: bool) -> Path:
    if variant.mode == "beta_profiled":
        return run_one_beta(dataset, variant, indices=indices, force=force)
    return run_one(dataset, variant, indices=indices, force=force)


def variant_by_key() -> dict[str, VariantSpec]:
    return {v.key: v for v in VARIANTS}


def normalize_rows(df: pd.DataFrame, dataset: str, variant: VariantSpec, source_csv: Path) -> pd.DataFrame:
    spec = DATASETS[dataset]
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str)
    out["mass_GeV"] = pd.to_numeric(out["mass_GeV"], errors="coerce").round(12)
    out["inj_nsigma"] = pd.to_numeric(out["inj_nsigma"], errors="coerce")
    out["variant"] = variant.key
    out["variant_label"] = variant.label
    out["variant_role"] = variant.role
    out["mode"] = variant.mode
    out["nuisance_profile"] = variant.mode in {"profiled", "beta_profiled"}
    out["lslb"] = float(spec.best_lslb)
    out["ls_upper"] = float(variant.ls_upper if variant.ls_upper is not None else spec.default_upper)
    out["const_lower"] = float(variant.const_bounds[0]) if variant.const_bounds else float("nan")
    out["const_upper"] = float(variant.const_bounds[1]) if variant.const_bounds else float("nan")
    out["source_csv"] = str(source_csv)

    A_hat = pd.to_numeric(out["A_hat"], errors="coerce")
    strength = pd.to_numeric(out["strength"], errors="coerce")
    sigma_A = pd.to_numeric(out["sigma_A"], errors="coerce")
    sigma_ref = pd.to_numeric(out.get("sigmaA_ref", np.nan), errors="coerce")
    aper = pd.to_numeric(out["A_per_eps2_unit"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["pull_correct"] = (A_hat - strength) / sigma_A
        out["delta_z"] = pd.to_numeric(out["Zhat"], errors="coerce") - out["inj_nsigma"]
        out["recovery"] = np.where(strength != 0.0, A_hat / strength, np.nan)
        out["recovery_bias"] = out["recovery"] - 1.0
        out["sigma_ratio_refit"] = sigma_A / sigma_ref
        out["eps2_sigma"] = sigma_A / aper
        out["eps_sigma"] = np.sqrt(np.clip(out["eps2_sigma"], 0.0, None))
        out["eps2_95_exp_proxy"] = Z95 * sigma_A / aper
        out["eps_95_exp_proxy"] = np.sqrt(np.clip(out["eps2_95_exp_proxy"], 0.0, None))
        obs_amp = np.clip(A_hat, 0.0, None) + Z95 * sigma_A
        out["eps2_95_obs_proxy"] = obs_amp / aper
        out["eps_95_obs_proxy"] = np.sqrt(np.clip(out["eps2_95_obs_proxy"], 0.0, None))
        out["eps2_bias_proxy"] = (A_hat - strength) / aper
        out["Ahat_over_strength"] = np.where(strength != 0.0, A_hat / strength, np.nan)
    return out


def load_rows(datasets: list[str], variants: list[VariantSpec]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for variant in variants:
            path = RUN_DIR / variant.key / dataset / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
            if path.exists():
                frames.append(normalize_rows(pd.read_csv(path), dataset, variant, path))
    return frames


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) <= 1:
        return float("nan")
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def q(values: pd.Series, quantile: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, quantile))


def summarize(toy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "dataset",
        "variant",
        "variant_label",
        "variant_role",
        "mode",
        "nuisance_profile",
        "lslb",
        "ls_upper",
        "const_lower",
        "const_upper",
        "mass_GeV",
        "inj_nsigma",
    ]
    for keys, sub in toy.groupby(group_cols, dropna=False):
        pull = pd.to_numeric(sub["pull_correct"], errors="coerce").dropna()
        delta_z = pd.to_numeric(sub["delta_z"], errors="coerce").dropna()
        recovery_bias = pd.to_numeric(sub["recovery_bias"], errors="coerce").dropna()
        eps2_exp = pd.to_numeric(sub["eps2_95_exp_proxy"], errors="coerce").dropna()
        eps2_obs = pd.to_numeric(sub["eps2_95_obs_proxy"], errors="coerce").dropna()
        eps_exp = pd.to_numeric(sub["eps_95_exp_proxy"], errors="coerce").dropna()
        eps_obs = pd.to_numeric(sub["eps_95_obs_proxy"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys))
        row.update(
            n_toys=int(len(sub)),
            n_source_toys=int(sub["toy_index"].nunique()) if "toy_index" in sub.columns else int(len(sub)),
            success_rate=float(pd.to_numeric(sub.get("success", pd.Series(dtype=float)), errors="coerce").mean()),
            refit_ok_rate=float(pd.to_numeric(sub.get("refit_ok", pd.Series(dtype=float)), errors="coerce").mean()),
            fallback_rate=float(pd.to_numeric(sub.get("refit_fallback_used", pd.Series(dtype=float)), errors="coerce").mean()),
            pull_mean_correct=float(pull.mean()) if len(pull) else float("nan"),
            pull_mean_err=sem(pull),
            pull_width_correct=float(pull.std(ddof=1)) if len(pull) > 1 else float("nan"),
            pull_width_err=float(pull.std(ddof=1) / math.sqrt(2.0 * (len(pull) - 1))) if len(pull) > 1 else float("nan"),
            cov_1sigma=float((pull.abs() <= 1.0).mean()) if len(pull) else float("nan"),
            cov_2sigma=float((pull.abs() <= 2.0).mean()) if len(pull) else float("nan"),
            delta_z_mean=float(delta_z.mean()) if len(delta_z) else float("nan"),
            delta_z_width=float(delta_z.std(ddof=1)) if len(delta_z) > 1 else float("nan"),
            recovery_bias_mean=float(recovery_bias.mean()) if len(recovery_bias) else float("nan"),
            recovery_bias_median=float(recovery_bias.median()) if len(recovery_bias) else float("nan"),
            sigma_ratio_median=float(pd.to_numeric(sub["sigma_ratio_refit"], errors="coerce").median()),
            eps2_95_exp_proxy_median=float(eps2_exp.median()) if len(eps2_exp) else float("nan"),
            eps2_95_exp_proxy_q16=q(eps2_exp, 0.16),
            eps2_95_exp_proxy_q84=q(eps2_exp, 0.84),
            eps2_95_obs_proxy_median=float(eps2_obs.median()) if len(eps2_obs) else float("nan"),
            eps2_95_obs_proxy_q16=q(eps2_obs, 0.16),
            eps2_95_obs_proxy_q84=q(eps2_obs, 0.84),
            eps_95_exp_proxy_median=float(eps_exp.median()) if len(eps_exp) else float("nan"),
            eps_95_obs_proxy_median=float(eps_obs.median()) if len(eps_obs) else float("nan"),
            A_per_eps2_unit_mean=float(pd.to_numeric(sub["A_per_eps2_unit"], errors="coerce").mean()),
            sigma_A_mean=float(pd.to_numeric(sub["sigma_A"], errors="coerce").mean()),
            A_hat_mean=float(pd.to_numeric(sub["A_hat"], errors="coerce").mean()),
            initial_ls_opt_mean=float(pd.to_numeric(sub.get("initial_ls_opt", pd.Series(dtype=float)), errors="coerce").mean()),
            initial_const_opt_mean=float(pd.to_numeric(sub.get("initial_const_opt", pd.Series(dtype=float)), errors="coerce").mean()),
            refit_ls_opt_mean=float(pd.to_numeric(sub.get("refit_ls_opt", pd.Series(dtype=float)), errors="coerce").mean()),
            refit_const_opt_mean=float(pd.to_numeric(sub.get("refit_const_opt", pd.Series(dtype=float)), errors="coerce").mean()),
            ls_lo_median=float(pd.to_numeric(sub.get("ls_lo", pd.Series(dtype=float)), errors="coerce").median()),
            ls_hi_median=float(pd.to_numeric(sub.get("ls_hi", pd.Series(dtype=float)), errors="coerce").median()),
            nuisance_pull_norm_median=float(pd.to_numeric(sub.get("nuisance_pull_norm", pd.Series(dtype=float)), errors="coerce").median()),
            profile_shift_abs_fraction_observed_median=float(pd.to_numeric(sub.get("profile_shift_abs_fraction_observed", pd.Series(dtype=float)), errors="coerce").median()),
            source_root=str(sub["source_root"].iloc[0]) if "source_root" in sub.columns else "",
            container=str(sub["container"].iloc[0]) if "container" in sub.columns else "",
            function_tag=str(sub["function_tag"].iloc[0]) if "function_tag" in sub.columns else "",
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "variant", "mass_GeV", "inj_nsigma"]).reset_index(drop=True)


def rank(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, variant), sub in summary.groupby(["dataset", "variant"], dropna=False):
        nonzero = sub[sub["inj_nsigma"].astype(float) > 0]
        source = nonzero if not nonzero.empty else sub
        pull_mean = pd.to_numeric(source["pull_mean_correct"], errors="coerce")
        width = pd.to_numeric(source["pull_width_correct"], errors="coerce")
        cov1 = pd.to_numeric(sub["cov_1sigma"], errors="coerce")
        cov2 = pd.to_numeric(sub["cov_2sigma"], errors="coerce")
        row = {
            "dataset": dataset,
            "variant": variant,
            "variant_label": sub["variant_label"].iloc[0],
            "variant_role": sub["variant_role"].iloc[0],
            "mode": sub["mode"].iloc[0],
            "nuisance_profile": bool(sub["nuisance_profile"].iloc[0]),
            "lslb": float(sub["lslb"].iloc[0]),
            "ls_upper": float(sub["ls_upper"].iloc[0]),
            "const_lower": float(sub["const_lower"].iloc[0]),
            "const_upper": float(sub["const_upper"].iloc[0]),
            "n_groups": int(len(sub)),
            "n_nonzero_groups": int(len(nonzero)),
            "n_toy_rows": int(sub["n_toys"].sum()),
            "n_source_toys_median": float(pd.to_numeric(sub["n_source_toys"], errors="coerce").median()),
            "rms_pull_mean_nonzero": float(np.sqrt(np.nanmean(np.square(pull_mean)))) if len(pull_mean) else float("nan"),
            "median_abs_pull_mean_nonzero": float(np.nanmedian(np.abs(pull_mean))) if len(pull_mean) else float("nan"),
            "pull_width_median_nonzero": float(np.nanmedian(width)) if len(width) else float("nan"),
            "pull_width_rmse_nonzero": float(np.sqrt(np.nanmean(np.square(width - 1.0)))) if len(width) else float("nan"),
            "cov1_abs_resid_mean_all": float(np.nanmean(np.abs(cov1 - 0.6827))) if len(cov1) else float("nan"),
            "cov2_abs_resid_mean_all": float(np.nanmean(np.abs(cov2 - 0.9545))) if len(cov2) else float("nan"),
            "sigma_ratio_median_all": float(pd.to_numeric(sub["sigma_ratio_median"], errors="coerce").median()),
            "eps2_95_exp_proxy_median_nonzero": float(pd.to_numeric(source["eps2_95_exp_proxy_median"], errors="coerce").median()),
            "eps2_95_obs_proxy_median_nonzero": float(pd.to_numeric(source["eps2_95_obs_proxy_median"], errors="coerce").median()),
            "eps_95_exp_proxy_median_nonzero": float(pd.to_numeric(source["eps_95_exp_proxy_median"], errors="coerce").median()),
            "eps_95_obs_proxy_median_nonzero": float(pd.to_numeric(source["eps_95_obs_proxy_median"], errors="coerce").median()),
            "success_rate_min": float(pd.to_numeric(sub["success_rate"], errors="coerce").min()),
            "refit_ok_rate_min": float(pd.to_numeric(sub["refit_ok_rate"], errors="coerce").min()),
            "fallback_rate_max": float(pd.to_numeric(sub["fallback_rate"], errors="coerce").max()),
            "ls_hi_median": float(pd.to_numeric(sub["ls_hi_median"], errors="coerce").median()),
            "refit_ls_opt_mean": float(pd.to_numeric(sub["refit_ls_opt_mean"], errors="coerce").mean()),
            "refit_const_opt_mean": float(pd.to_numeric(sub["refit_const_opt_mean"], errors="coerce").mean()),
            "nuisance_pull_norm_median_all": float(pd.to_numeric(sub["nuisance_pull_norm_median"], errors="coerce").median()),
            "profile_shift_abs_fraction_observed_median_all": float(pd.to_numeric(sub["profile_shift_abs_fraction_observed_median"], errors="coerce").median()),
        }
        row["score_lower_is_better"] = (
            0.30 * row["rms_pull_mean_nonzero"]
            + 0.25 * row["pull_width_rmse_nonzero"]
            + 0.20 * row["median_abs_pull_mean_nonzero"]
            + 0.15 * row["cov1_abs_resid_mean_all"]
            + 0.05 * row["cov2_abs_resid_mean_all"]
            + 0.05 * abs(row["sigma_ratio_median_all"] - 1.0)
            + 2.0 * max(0.0, 1.0 - row["success_rate_min"])
            + 2.0 * max(0.0, 1.0 - row["refit_ok_rate_min"])
            + 2.0 * row["fallback_rate_max"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_nominal_ratios(out, key_cols=["dataset"], value_cols=[
        "eps2_95_exp_proxy_median_nonzero",
        "eps2_95_obs_proxy_median_nonzero",
        "eps_95_exp_proxy_median_nonzero",
        "eps_95_obs_proxy_median_nonzero",
    ])
    out = out.sort_values(["dataset", "score_lower_is_better", "variant"]).reset_index(drop=True)
    out["rank_within_dataset"] = out.groupby("dataset")["score_lower_is_better"].rank(method="first")
    return out


def add_nominal_ratios(frame: pd.DataFrame, *, key_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    nominal = out[out["variant"].astype(str) == "profiled_nominal"].copy()
    if nominal.empty:
        return out
    for col in value_cols:
        ref = nominal[key_cols + [col]].rename(columns={col: f"{col}_nominal"})
        out = out.merge(ref, on=key_cols, how="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{col}_ratio_vs_profiled_nominal"] = pd.to_numeric(out[col], errors="coerce") / pd.to_numeric(
                out[f"{col}_nominal"], errors="coerce"
            )
        out.drop(columns=[f"{col}_nominal"], inplace=True)
    out["eps2_exp_reach_loss_pct_vs_nominal"] = 100.0 * (
        pd.to_numeric(out["eps2_95_exp_proxy_median_nonzero_ratio_vs_profiled_nominal"], errors="coerce") - 1.0
    )
    return out


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    out_cols = [
        "dataset",
        "mass_GeV",
        "inj_nsigma",
        "comparison_variant",
        "comparison_label",
        "comparison_role",
        "pull_mean_delta_alt_minus_nominal",
        "pull_width_delta_alt_minus_nominal",
        "eps2_exp_ratio_alt_over_nominal",
        "eps2_obs_ratio_alt_over_nominal",
        "eps_exp_ratio_alt_over_nominal",
        "eps_obs_ratio_alt_over_nominal",
        "sigma_A_ratio_alt_over_nominal",
        "Ahat_delta_alt_minus_nominal",
        "sigma_ratio_delta_alt_minus_nominal",
        "eps2_exp_reach_loss_pct_alt_vs_nominal",
    ]
    key_cols = ["dataset", "mass_GeV", "inj_nsigma"]
    nominal = summary[summary["variant"].astype(str) == "profiled_nominal"].copy()
    if nominal.empty:
        return pd.DataFrame(columns=out_cols)
    for _, alt in summary[summary["variant"].astype(str) != "profiled_nominal"].iterrows():
        match = nominal[
            (nominal["dataset"].astype(str) == str(alt["dataset"]))
            & (nominal["mass_GeV"].astype(float).round(12) == round(float(alt["mass_GeV"]), 12))
            & (nominal["inj_nsigma"].astype(float) == float(alt["inj_nsigma"]))
        ]
        if match.empty:
            continue
        b = match.iloc[0]
        row = {c: alt[c] for c in key_cols}
        row.update(
            comparison_variant=alt["variant"],
            comparison_label=alt["variant_label"],
            comparison_role=alt["variant_role"],
            pull_mean_delta_alt_minus_nominal=float(alt["pull_mean_correct"]) - float(b["pull_mean_correct"]),
            pull_width_delta_alt_minus_nominal=float(alt["pull_width_correct"]) - float(b["pull_width_correct"]),
            eps2_exp_ratio_alt_over_nominal=float(alt["eps2_95_exp_proxy_median"]) / float(b["eps2_95_exp_proxy_median"]),
            eps2_obs_ratio_alt_over_nominal=float(alt["eps2_95_obs_proxy_median"]) / float(b["eps2_95_obs_proxy_median"]),
            eps_exp_ratio_alt_over_nominal=float(alt["eps_95_exp_proxy_median"]) / float(b["eps_95_exp_proxy_median"]),
            eps_obs_ratio_alt_over_nominal=float(alt["eps_95_obs_proxy_median"]) / float(b["eps_95_obs_proxy_median"]),
            sigma_A_ratio_alt_over_nominal=float(alt["sigma_A_mean"]) / float(b["sigma_A_mean"]),
            Ahat_delta_alt_minus_nominal=float(alt["A_hat_mean"]) - float(b["A_hat_mean"]),
            sigma_ratio_delta_alt_minus_nominal=float(alt["sigma_ratio_median"]) - float(b["sigma_ratio_median"]),
        )
        row["eps2_exp_reach_loss_pct_alt_vs_nominal"] = 100.0 * (row["eps2_exp_ratio_alt_over_nominal"] - 1.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)


def plot_outputs(summary: pd.DataFrame, ranking: pd.DataFrame, comparison: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ds_values = sorted(summary["dataset"].astype(str).unique())
    palette = {
        "fixed_no_nuisance": "#7A7A7A",
        "profiled_nominal": "#0072B2",
        "majd_phi_beta_profiled": "#56B4E9",
        "profiled_lsub6": "#D55E00",
        "profiled_lsub12": "#009E73",
        "profiled_const1em4_1e4": "#CC79A7",
        "profiled_const1em2_1e2": "#E69F00",
    }
    for dataset in ds_values:
        ds = summary[(summary["dataset"].astype(str) == dataset) & (summary["inj_nsigma"].astype(float) > 0)].copy()
        if ds.empty:
            continue
        for metric, ylabel, ref, stem in [
            ("pull_width_correct", "pull width", 1.0, "pull_width"),
            ("pull_mean_correct", "mean pull", 0.0, "mean_pull"),
            ("eps2_95_exp_proxy_median", "median expected epsilon^2 95 proxy", None, "eps2_expected_proxy"),
            ("eps2_95_obs_proxy_median", "median observed-like epsilon^2 95 proxy", None, "eps2_observed_proxy"),
        ]:
            fig, ax = plt.subplots(figsize=(7.6, 4.6))
            for variant, sub in ds.groupby("variant", sort=False):
                by_mass = sub.groupby("mass_GeV")[metric].median().sort_index()
                label = sub["variant_label"].iloc[0]
                ax.plot(
                    1000.0 * by_mass.index.to_numpy(float),
                    by_mass.to_numpy(float),
                    marker="o",
                    lw=1.6,
                    label=label,
                    color=palette.get(str(variant)),
                )
            if ref is not None:
                ax.axhline(ref, color="0.35", lw=1.0, ls="--")
            ax.set_xlabel("mass [MeV]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset}: {ylabel}, 10 source toys")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{dataset}_{stem}.png", dpi=180)
            plt.close(fig)

        comp = comparison[(comparison["dataset"].astype(str) == dataset) & (comparison["inj_nsigma"].astype(float) > 0)]
        if not comp.empty:
            fig, ax = plt.subplots(figsize=(7.6, 4.3))
            for variant, sub in comp.groupby("comparison_variant", sort=False):
                by_mass = sub.groupby("mass_GeV")["eps2_exp_ratio_alt_over_nominal"].median().sort_index()
                ax.plot(
                    1000.0 * by_mass.index.to_numpy(float),
                    by_mass.to_numpy(float),
                    marker="o",
                    lw=1.6,
                    label=sub["comparison_label"].iloc[0],
                    color=palette.get(str(variant)),
                )
            ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
            ax.axhline(1.25, color="#B00020", lw=0.9, ls=":")
            ax.set_xlabel("mass [MeV]")
            ax.set_ylabel("expected epsilon^2 proxy ratio to profiled nominal")
            ax.set_title(f"{dataset}: epsilon^2 reach loss vs nominal profile")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{dataset}_eps2_ratio_vs_profiled_nominal.png", dpi=180)
            plt.close(fig)

    rank = ranking.copy()
    if not rank.empty and "eps2_95_exp_proxy_median_nonzero_ratio_vs_profiled_nominal" in rank.columns:
        pivot = rank.pivot(index="variant", columns="dataset", values="eps2_95_exp_proxy_median_nonzero_ratio_vs_profiled_nominal")
        order = [v.key for v in VARIANTS if v.key in pivot.index]
        pivot = pivot.reindex(order)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        x = np.arange(len(pivot.index), dtype=float)
        datasets = list(pivot.columns)
        width = 0.75 / max(len(datasets), 1)
        for i, dataset in enumerate(datasets):
            ax.bar(x + (i - (len(datasets) - 1) / 2.0) * width, pivot[dataset].to_numpy(float), width, label=dataset)
        ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
        ax.axhline(1.25, color="#B00020", lw=0.9, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("profiled_", "prof_").replace("_", "\n") for v in pivot.index], fontsize=8)
        ax.set_ylabel("median expected epsilon^2 ratio to profiled nominal")
        ax.set_title("Reach impact by nuisance/kernel variant")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "eps2_reach_ratio_by_variant.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        score = rank.pivot(index="variant", columns="dataset", values="score_lower_is_better").reindex(order)
        for i, dataset in enumerate(list(score.columns)):
            ax.bar(x + (i - (len(datasets) - 1) / 2.0) * width, score[dataset].to_numpy(float), width, label=dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("profiled_", "prof_").replace("_", "\n") for v in score.index], fontsize=8)
        ax.set_ylabel("extraction score, lower is better")
        ax.set_title("Extraction calibration score by variant")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "extraction_score_by_variant.png", dpi=180)
        plt.close(fig)


def md_table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    frame = frame.copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = ""

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return ""
            return f"{float(value):.4g}"
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(fmt(row[col]) for col in cols) + " |" for _, row in frame[cols].iterrows()]
    return "\n".join([header, sep] + rows)


def write_markdown(summary: pd.DataFrame, ranking: pd.DataFrame, comparison: pd.DataFrame, *, max_toys: int) -> None:
    top_cols = [
        "dataset",
        "variant",
        "variant_label",
        "n_source_toys_median",
        "score_lower_is_better",
        "rms_pull_mean_nonzero",
        "pull_width_median_nonzero",
        "eps2_95_exp_proxy_median_nonzero_ratio_vs_profiled_nominal",
        "eps2_exp_reach_loss_pct_vs_nominal",
    ]
    top = ranking.sort_values(["dataset", "score_lower_is_better"]).copy()
    comp_reduced = (
        comparison[comparison["inj_nsigma"].astype(float) > 0]
        .groupby(["dataset", "comparison_variant", "comparison_label"], as_index=False)
        .agg(
            median_eps2_exp_ratio_alt_over_nominal=("eps2_exp_ratio_alt_over_nominal", "median"),
            median_eps2_obs_ratio_alt_over_nominal=("eps2_obs_ratio_alt_over_nominal", "median"),
            median_sigma_A_ratio_alt_over_nominal=("sigma_A_ratio_alt_over_nominal", "median"),
            median_pull_width_delta_alt_minus_nominal=("pull_width_delta_alt_minus_nominal", "median"),
            median_pull_mean_delta_alt_minus_nominal=("pull_mean_delta_alt_minus_nominal", "median"),
        )
    )
    comp_reduced["median_eps2_exp_reach_loss_pct"] = 100.0 * (
        comp_reduced["median_eps2_exp_ratio_alt_over_nominal"] - 1.0
    )

    lines = [
        "# Functional-form mod_2 nuisance-profile and kernel-bound exploration",
        "",
        "Generated by `explore_nuisance_bounds.py`.",
        "",
        "Scope:",
        "- ROOT inputs: `funcform_2015_dataset_mod_toys_2.root`, `funcform_2016_dataset_mod_toys_2.root`, `funcform_2021_dataset_mod_toys_2.root`.",
        f"- Source toys per row: `{max_toys}`.",
        "- Masses per dataset: three representative points from the 25-toy high-stat grid.",
        "- Nominal lslb choices: 2015 `1.1`, 2016 `0.9`, 2021 `1.1` from the high-stat ranking.",
        "",
        "Majd_phi nuisance-profile interpretation:",
        "",
        "The direct no-profile comparator is `fixed_no_nuisance`. `profiled_nominal` is the native HPS-GPR Gaussian covariance profile. `majd_phi_beta_profiled` is a separate exploratory port of the Majd_phi three-parameter multiplicative beta profile, using the same generated toys and native HPS sigmaA strength scaling so the reach comparison is paired.",
        "",
        "Epsilon^2 reach convention:",
        "",
        "The epsilon^2 columns are extraction proxies, not exact CLs limits. Ratios above `1` mean weaker expected epsilon^2 reach relative to `profiled_nominal`; this is the publication-risk axis because enough reach loss can remove novel unblinded parameter space.",
        "",
        "## Ranking",
        "",
        md_table(top, top_cols),
        "",
        "## Variant-vs-profiled-nominal comparison",
        "",
        md_table(
            comp_reduced,
            [
                "dataset",
                "comparison_variant",
                "median_eps2_exp_ratio_alt_over_nominal",
                "median_eps2_exp_reach_loss_pct",
                "median_eps2_obs_ratio_alt_over_nominal",
                "median_sigma_A_ratio_alt_over_nominal",
                "median_pull_width_delta_alt_minus_nominal",
                "median_pull_mean_delta_alt_minus_nominal",
            ],
        ),
        "",
        "## Primary outputs",
        "",
        "- `explore_toy_level.csv`: toy-level extraction rows with epsilon^2 proxy columns.",
        "- `explore_summary.csv`: grouped rows by dataset, variant, mass, and injected significance.",
        "- `explore_ranking.csv`: variant ranking and median reach ratios.",
        "- `explore_comparison_by_mass_strength.csv`: per-mass/strength ratios to `profiled_nominal`.",
        "- `plots/`: pull, epsilon^2, reach-ratio, and score plots.",
        "",
        "## Caveats",
        "",
        "- This is a 10-toy exploratory screen, not a coverage result.",
        "- The fixed-background row is a no-nuisance negative-control comparator, not a candidate publication model.",
        "- The Majd_phi beta row is an exploratory extraction-basis stress test; it is not a drop-in publication model without coverage validation.",
        "- Exact publication impact on exclusion contours still requires the final CLs scan/band workflow, not only extraction proxies.",
    ]
    (STUDY_DIR / "NUISANCE_BOUNDS_10TOY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(datasets: list[str], variants: list[VariantSpec], max_toys: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = load_rows(datasets, variants)
    if not frames:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    toy = pd.concat(frames, ignore_index=True, sort=False)
    toy = toy.drop_duplicates(
        subset=["dataset", "variant", "mass_GeV", "inj_nsigma", "toy_index", "injection_toy"],
        keep="last",
    )
    summary = summarize(toy)
    ranking = rank(summary)
    comparison = build_comparison(summary)
    toy.to_csv(STUDY_DIR / "explore_toy_level.csv", index=False)
    summary.to_csv(STUDY_DIR / "explore_summary.csv", index=False)
    ranking.to_csv(STUDY_DIR / "explore_ranking.csv", index=False)
    comparison.to_csv(STUDY_DIR / "explore_comparison_by_mass_strength.csv", index=False)
    plot_outputs(summary, ranking, comparison)
    write_markdown(summary, ranking, comparison, max_toys=max_toys)
    return toy, summary, ranking, comparison


def parse_dataset_list(raw: str) -> list[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in vals if x not in DATASETS]
    if bad:
        raise ValueError(f"Unknown dataset(s): {bad}")
    return vals or list(DATASETS)


def parse_variant_list(raw: str) -> list[VariantSpec]:
    by_key = variant_by_key()
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        return list(VARIANTS)
    bad = [x for x in vals if x not in by_key]
    if bad:
        raise ValueError(f"Unknown variant(s): {bad}")
    return [by_key[x] for x in vals]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="2015,2016,2021", help="Comma-separated dataset keys.")
    parser.add_argument(
        "--variants",
        default=",".join(v.key for v in VARIANTS),
        help="Comma-separated variant keys.",
    )
    parser.add_argument("--max-toys", type=int, default=10, help="Number of functional-form source toys per row.")
    parser.add_argument("--toy-start", type=int, default=0, help="First toy index.")
    parser.add_argument("--jobs", type=int, default=3, help="Parallel dataset/variant commands.")
    parser.add_argument("--force", action="store_true", help="Re-run existing rows.")
    parser.add_argument("--no-run", action="store_true", help="Only generate configs and summarize existing CSVs.")
    parser.add_argument("--only-summarize", action="store_true", help="Skip execution and summarize existing CSVs.")
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    variants = parse_variant_list(args.variants)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(DATASETS[d].toy_root) for d in datasets if not DATASETS[d].toy_root.exists()]
    if missing:
        raise FileNotFoundError("Missing mod_2 ROOT input(s): " + ", ".join(missing))

    for dataset in datasets:
        for variant in variants:
            config_for(dataset, variant)

    if not args.only_summarize and not args.no_run:
        indices = toy_indices(max_toys=int(args.max_toys), toy_start=int(args.toy_start))
        tasks = [(dataset, variant) for dataset in datasets for variant in variants]
        max_workers = max(1, int(args.jobs))
        if max_workers == 1 or len(tasks) == 1:
            for dataset, variant in tasks:
                run_variant(dataset, variant, indices=indices, force=bool(args.force))
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
                futures = [
                    pool.submit(run_variant, dataset, variant, indices=indices, force=bool(args.force))
                    for dataset, variant in tasks
                ]
                for future in as_completed(futures):
                    future.result()

    write_outputs(datasets, variants, max_toys=int(args.max_toys))
    print(f"[done] wrote outputs under {STUDY_DIR}")


if __name__ == "__main__":
    main()
