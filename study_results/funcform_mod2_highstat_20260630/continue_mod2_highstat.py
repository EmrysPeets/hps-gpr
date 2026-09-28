#!/usr/bin/env python3
"""Continue the functional-form mod_2 pull-width study at higher statistics.

This is a focused continuation of
``study_results/pullwidth_diagnostics_10toy_corrected_20260629``.  It carries
forward only the profiled-background rows, adds 2021 support, and translates
extraction width/bias into epsilon-scale proxy columns.

The epsilon quantities here are closure proxies:

    eps2_95_exp_proxy = 1.6448536269514722 * sigma_A / A_per_eps2_unit
    eps2_95_obs_proxy = (max(A_hat, 0) + 1.6448536269514722 * sigma_A) / A_per_eps2_unit

They are useful for comparing lslb impact on the epsilon upper-limit scale, but
they are not a replacement for the exact CLs toy-scan/band workflow.
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
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


STUDY_DIR = Path(__file__).resolve().parent
PREVIOUS_STUDY_DIR = STUDY_DIR.parent / "pullwidth_diagnostics_10toy_corrected_20260629"
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
    base_config: str
    toy_root: Path
    container: str
    toy_name_fmt: str
    masses: list[float]
    pilot_masses: list[float]
    lslb_candidates: list[float]


DATASETS: dict[str, DatasetSpec] = {
    "2015": DatasetSpec(
        key="2015",
        base_config="config_2015_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
        toy_root=INPUT_DIR / "funcform_2015_dataset_mod_toys_2.root",
        container="fShiftSigPowTail",
        toy_name_fmt="fShiftSigPowTail_toy_{i}",
        masses=[0.045, 0.060, 0.075, 0.090, 0.105],
        pilot_masses=[0.045, 0.075, 0.105],
        lslb_candidates=[1.1, 1.0],
    ),
    "2016": DatasetSpec(
        key="2016",
        base_config="config_2016_10pct_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
        toy_root=INPUT_DIR / "funcform_2016_dataset_mod_toys_2.root",
        container="fShiftSigPowTail",
        toy_name_fmt="fShiftSigPowTail_toy_{i}",
        masses=[0.060, 0.090, 0.105, 0.120, 0.150],
        pilot_masses=[0.060, 0.105, 0.150],
        lslb_candidates=[1.1, 0.9],
    ),
    "2021": DatasetSpec(
        key="2021",
        base_config="config_2021_1pct_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
        toy_root=INPUT_DIR / "funcform_2021_dataset_mod_toys_2.root",
        container="fSigPowExpQ",
        toy_name_fmt="fSigPowExpQ_toy_{i}",
        masses=[0.060, 0.090, 0.105, 0.150, 0.220],
        pilot_masses=[0.060, 0.105, 0.180],
        lslb_candidates=[1.1, 1.0, 0.9],
    ),
}


def tag_float(value: float) -> str:
    if float(value).is_integer():
        text = f"{float(value):.1f}"
    else:
        text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def study_name(lslb: float) -> str:
    return f"profiled_lslb{tag_float(lslb)}"


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


def set_ls_factors(cfg: dict, dataset: str, lower: float, upper: float | None = None) -> None:
    upper = 9.0 if upper is None and dataset == "2021" else (8.0 if upper is None else upper)
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


def config_for(dataset: str, lslb: float, phase: str) -> Path:
    spec = DATASETS[dataset]
    base_path = RUNNER_REPO / "study_configs" / "funcform_pullwidth_diagnostics" / spec.base_config
    cfg = copy.deepcopy(load_yaml(base_path))
    set_dataset_enable(cfg, dataset)
    set_ls_factors(cfg, dataset, lower=lslb)
    cfg["extract_background_mode"] = "profiled"
    cfg["funcform_closure_root_by_dataset"] = {k: str(v.toy_root) for k, v in DATASETS.items()}
    cfg["funcform_closure_container_by_dataset"] = {k: str(v.container) for k, v in DATASETS.items()}
    cfg["funcform_closure_toy_pattern_by_dataset"] = {
        k: v.toy_name_fmt.replace("{i}", "*") for k, v in DATASETS.items()
    }
    cfg["output_dir"] = str(RUN_DIR / phase / study_name(lslb) / dataset)
    cfg["save_plots"] = False
    cfg["inj_n_workers"] = 1
    cfg["inj_threads_per_worker"] = 1
    cfg["inj_write_qmu"] = True
    path = CONFIG_DIR / phase / f"config_{dataset}_{study_name(lslb)}.yaml"
    dump_yaml(path, cfg)
    return path


def toy_indices(max_toys: int, toy_start: int = 0) -> list[int]:
    if max_toys <= 0:
        raise ValueError("--max-toys must be positive")
    return list(range(int(toy_start), int(toy_start) + int(max_toys)))


def run_one(
    dataset: str,
    lslb: float,
    *,
    phase: str,
    masses: list[float],
    indices: list[int],
    force: bool,
) -> Path:
    spec = DATASETS[dataset]
    outdir = RUN_DIR / phase / study_name(lslb) / dataset
    csv_path = outdir / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {phase} {dataset} {study_name(lslb)}: {csv_path}")
        return csv_path

    cfg_path = config_for(dataset, lslb, phase)
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
        ",".join(f"{m:.3f}" for m in masses),
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
    log_path = LOG_DIR / f"{phase}_{dataset}_{study_name(lslb)}.log"
    print(
        f"[run] {phase} {dataset} {study_name(lslb)}: "
        f"masses={len(masses)} toys={len(indices)} log={log_path}",
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
            print(f"[fail] {phase} {dataset} {study_name(lslb)}; see {log_path}", flush=True)
            raise
    print(f"[done] {phase} {dataset} {study_name(lslb)}", flush=True)
    return csv_path


def normalize_rows(df: pd.DataFrame, dataset: str, lslb: float, phase: str, source_csv: Path) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str)
    out["mass_GeV"] = pd.to_numeric(out["mass_GeV"], errors="coerce").round(12)
    out["inj_nsigma"] = pd.to_numeric(out["inj_nsigma"], errors="coerce")
    out["study"] = study_name(lslb)
    out["study_label"] = f"profiled bkg, lslb={lslb:g}"
    out["mode"] = "profiled"
    out["lslb"] = float(lslb)
    out["phase"] = phase
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
        out["eps_width_ratio_vs_ref"] = np.sqrt(np.clip(out["sigma_ratio_refit"], 0.0, None))
        out["eps2_sigma"] = sigma_A / aper
        out["eps_sigma"] = np.sqrt(np.clip(out["eps2_sigma"], 0.0, None))
        out["eps2_95_exp_proxy"] = Z95 * sigma_A / aper
        out["eps_95_exp_proxy"] = np.sqrt(np.clip(out["eps2_95_exp_proxy"], 0.0, None))
        obs_amp = np.clip(A_hat, 0.0, None) + Z95 * sigma_A
        out["eps2_95_obs_proxy"] = obs_amp / aper
        out["eps_95_obs_proxy"] = np.sqrt(np.clip(out["eps2_95_obs_proxy"], 0.0, None))
        out["eps2_bias_proxy"] = (A_hat - strength) / aper
    return out


def load_phase_rows(phase: str, candidates: dict[str, list[float]]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for dataset, lslbs in candidates.items():
        for lslb in lslbs:
            path = RUN_DIR / phase / study_name(lslb) / dataset / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
            if path.exists():
                frames.append(normalize_rows(pd.read_csv(path), dataset, lslb, phase, path))
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
    group_cols = ["phase", "dataset", "study", "study_label", "mode", "lslb", "mass_GeV", "inj_nsigma"]
    for keys, sub in toy.groupby(group_cols, dropna=False):
        pull = pd.to_numeric(sub["pull_correct"], errors="coerce").dropna()
        delta_z = pd.to_numeric(sub["delta_z"], errors="coerce").dropna()
        recovery_bias = pd.to_numeric(sub["recovery_bias"], errors="coerce").dropna()
        eps_exp = pd.to_numeric(sub["eps_95_exp_proxy"], errors="coerce").dropna()
        eps_obs = pd.to_numeric(sub["eps_95_obs_proxy"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys))
        row.update(
            n_toys=int(len(sub)),
            n_source_toys=int(sub["toy_index"].nunique()) if "toy_index" in sub.columns else int(len(sub)),
            n_success=int(pd.to_numeric(sub.get("success", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
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
            eps_width_ratio_median=float(pd.to_numeric(sub["eps_width_ratio_vs_ref"], errors="coerce").median()),
            eps_95_exp_proxy_median=float(eps_exp.median()) if len(eps_exp) else float("nan"),
            eps_95_exp_proxy_q16=q(eps_exp, 0.16),
            eps_95_exp_proxy_q84=q(eps_exp, 0.84),
            eps_95_obs_proxy_median=float(eps_obs.median()) if len(eps_obs) else float("nan"),
            eps_95_obs_proxy_q16=q(eps_obs, 0.16),
            eps_95_obs_proxy_q84=q(eps_obs, 0.84),
            eps2_95_exp_proxy_median=float(pd.to_numeric(sub["eps2_95_exp_proxy"], errors="coerce").median()),
            eps2_95_obs_proxy_median=float(pd.to_numeric(sub["eps2_95_obs_proxy"], errors="coerce").median()),
            A_per_eps2_unit_mean=float(pd.to_numeric(sub["A_per_eps2_unit"], errors="coerce").mean()),
            sigma_A_mean=float(pd.to_numeric(sub["sigma_A"], errors="coerce").mean()),
            A_hat_mean=float(pd.to_numeric(sub["A_hat"], errors="coerce").mean()),
            qmu_tilde_median=float(pd.to_numeric(sub.get("qmu_tilde", pd.Series(dtype=float)), errors="coerce").median()),
            source_root=str(sub["source_root"].iloc[0]) if "source_root" in sub.columns else "",
            container=str(sub["container"].iloc[0]) if "container" in sub.columns else "",
            function_tag=str(sub["function_tag"].iloc[0]) if "function_tag" in sub.columns else "",
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase", "dataset", "study", "mass_GeV", "inj_nsigma"]).reset_index(drop=True)


def rank(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (phase, dataset, study), sub in summary.groupby(["phase", "dataset", "study"], dropna=False):
        nonzero = sub[sub["inj_nsigma"].astype(float) > 0]
        source = nonzero if not nonzero.empty else sub
        pull_mean = pd.to_numeric(source["pull_mean_correct"], errors="coerce")
        width = pd.to_numeric(source["pull_width_correct"], errors="coerce")
        cov1 = pd.to_numeric(sub["cov_1sigma"], errors="coerce")
        cov2 = pd.to_numeric(sub["cov_2sigma"], errors="coerce")
        sigma_ratio = pd.to_numeric(sub["sigma_ratio_median"], errors="coerce")
        row = {
            "phase": phase,
            "dataset": dataset,
            "study": study,
            "study_label": sub["study_label"].iloc[0],
            "mode": sub["mode"].iloc[0],
            "lslb": float(sub["lslb"].iloc[0]),
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
            "sigma_ratio_median_all": float(np.nanmedian(sigma_ratio)) if len(sigma_ratio) else float("nan"),
            "eps_width_ratio_median_all": float(pd.to_numeric(sub["eps_width_ratio_median"], errors="coerce").median()),
            "eps_95_exp_proxy_median_nonzero": float(pd.to_numeric(source["eps_95_exp_proxy_median"], errors="coerce").median()),
            "eps_95_obs_proxy_median_nonzero": float(pd.to_numeric(source["eps_95_obs_proxy_median"], errors="coerce").median()),
            "success_rate_min": float(pd.to_numeric(sub["success_rate"], errors="coerce").min()),
            "refit_ok_rate_min": float(pd.to_numeric(sub["refit_ok_rate"], errors="coerce").min()),
            "fallback_rate_max": float(pd.to_numeric(sub["fallback_rate"], errors="coerce").max()),
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
    out = out.sort_values(["phase", "dataset", "score_lower_is_better", "study"]).reset_index(drop=True)
    out["rank_within_dataset"] = out.groupby(["phase", "dataset"])["score_lower_is_better"].rank(method="first")
    return out


def select_top_two(ranking: pd.DataFrame, dataset: str, phase: str) -> list[float]:
    sub = ranking[(ranking["dataset"].astype(str) == dataset) & (ranking["phase"].astype(str) == phase)]
    if sub.empty:
        return DATASETS[dataset].lslb_candidates[:2]
    return [float(x) for x in sub.sort_values("score_lower_is_better")["lslb"].head(2).tolist()]


def build_comparison(summary: pd.DataFrame, ranking: pd.DataFrame, phase: str) -> pd.DataFrame:
    rows = []
    for dataset in sorted(summary["dataset"].astype(str).unique()):
        ds_rank = ranking[(ranking["phase"].astype(str) == phase) & (ranking["dataset"].astype(str) == dataset)]
        if ds_rank.empty:
            continue
        best = str(ds_rank.sort_values("score_lower_is_better")["study"].iloc[0])
        ds = summary[(summary["phase"].astype(str) == phase) & (summary["dataset"].astype(str) == dataset)]
        best_rows = ds[ds["study"].astype(str) == best]
        for _, alt in ds[ds["study"].astype(str) != best].iterrows():
            match = best_rows[
                (best_rows["mass_GeV"].astype(float).round(12) == round(float(alt["mass_GeV"]), 12))
                & (best_rows["inj_nsigma"].astype(float) == float(alt["inj_nsigma"]))
            ]
            if match.empty:
                continue
            b = match.iloc[0]
            rows.append(
                {
                    "phase": phase,
                    "dataset": dataset,
                    "best_study": best,
                    "comparison_study": alt["study"],
                    "mass_GeV": float(alt["mass_GeV"]),
                    "inj_nsigma": float(alt["inj_nsigma"]),
                    "pull_mean_delta_alt_minus_best": float(alt["pull_mean_correct"]) - float(b["pull_mean_correct"]),
                    "pull_width_delta_alt_minus_best": float(alt["pull_width_correct"]) - float(b["pull_width_correct"]),
                    "eps95_exp_ratio_alt_over_best": float(alt["eps_95_exp_proxy_median"]) / float(b["eps_95_exp_proxy_median"]),
                    "eps95_obs_ratio_alt_over_best": float(alt["eps_95_obs_proxy_median"]) / float(b["eps_95_obs_proxy_median"]),
                    "sigma_ratio_delta_alt_minus_best": float(alt["sigma_ratio_median"]) - float(b["sigma_ratio_median"]),
                }
            )
    return pd.DataFrame(rows)


def plot_outputs(summary: pd.DataFrame, ranking: pd.DataFrame, comparison: pd.DataFrame, phase: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ds_values = sorted(summary[summary["phase"].astype(str) == phase]["dataset"].astype(str).unique())
    for dataset in ds_values:
        ds = summary[(summary["phase"].astype(str) == phase) & (summary["dataset"].astype(str) == dataset)]
        ds = ds[ds["inj_nsigma"].astype(float) > 0].copy()
        if ds.empty:
            continue

        for metric, ylabel, ref, stem in [
            ("pull_width_correct", "pull width", 1.0, "pull_width"),
            ("pull_mean_correct", "mean pull", 0.0, "mean_pull"),
            ("eps_95_exp_proxy_median", "median epsilon 95 proxy", None, "eps95_expected_proxy"),
            ("eps_95_obs_proxy_median", "median observed-like epsilon 95 proxy", None, "eps95_observed_proxy"),
        ]:
            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            for study, sub in ds.groupby("study", sort=False):
                by_mass = sub.groupby("mass_GeV")[metric].median().sort_index()
                label = sub["study_label"].iloc[0]
                ax.plot(1000.0 * by_mass.index.to_numpy(float), by_mass.to_numpy(float), marker="o", lw=1.8, label=label)
            if ref is not None:
                ax.axhline(ref, color="0.35", lw=1.0, ls="--")
            ax.set_xlabel("mass [MeV]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset}: {ylabel} ({phase})")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{phase}_{dataset}_{stem}.png", dpi=180)
            plt.close(fig)

        comp = comparison[(comparison["phase"].astype(str) == phase) & (comparison["dataset"].astype(str) == dataset)]
        comp = comp[comp["inj_nsigma"].astype(float) > 0].copy()
        if not comp.empty:
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            for study, sub in comp.groupby("comparison_study", sort=False):
                by_mass = sub.groupby("mass_GeV")["eps95_exp_ratio_alt_over_best"].median().sort_index()
                ax.plot(1000.0 * by_mass.index.to_numpy(float), by_mass.to_numpy(float), marker="o", lw=1.8, label=study)
            ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
            ax.set_xlabel("mass [MeV]")
            ax.set_ylabel("epsilon proxy ratio to best")
            ax.set_title(f"{dataset}: alternate/best epsilon impact ({phase})")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{phase}_{dataset}_eps95_ratio_alt_over_best.png", dpi=180)
            plt.close(fig)


def md_table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "_No rows._"

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
    high = ranking[ranking["phase"].astype(str) == "highstat"].copy()
    top = high.sort_values(["dataset", "rank_within_dataset"]).groupby("dataset").head(2)
    comp_high = comparison[comparison["phase"].astype(str) == "highstat"].copy()
    comp_reduced = (
        comp_high[comp_high["inj_nsigma"].astype(float) > 0]
        .groupby(["dataset", "best_study", "comparison_study"], as_index=False)
        .agg(
            median_eps95_exp_ratio_alt_over_best=("eps95_exp_ratio_alt_over_best", "median"),
            median_eps95_obs_ratio_alt_over_best=("eps95_obs_ratio_alt_over_best", "median"),
            median_pull_width_delta_alt_minus_best=("pull_width_delta_alt_minus_best", "median"),
            median_pull_mean_delta_alt_minus_best=("pull_mean_delta_alt_minus_best", "median"),
        )
    )

    lines = [
        "# Functional-form mod_2 high-stat continuation",
        "",
        "Generated by `continue_mod2_highstat.py`.",
        "",
        "Scope:",
        "- Functional-form ROOT inputs: `funcform_2015_dataset_mod_toys_2.root`, `funcform_2016_dataset_mod_toys_2.root`, `funcform_2021_dataset_mod_toys_2.root`.",
        "- Primary containers: `fShiftSigPowTail` for 2015/2016 and `fSigPowExpQ` for 2021.",
        f"- Toy statistics requested per high-stat row: `{max_toys}` functional-form pseudoexperiments.",
        "- Extraction mode: profiled background only. Fixed background is retained only in the earlier diagnostic study as a negative-control row.",
        "",
        "Mass grids:",
    ]
    for dataset, spec in DATASETS.items():
        lines.append("- " + dataset + ": `" + ", ".join(f"{m:.3f}" for m in spec.masses) + "` GeV")

    lines.extend(
        [
            "",
            "Epsilon-impact convention:",
            "",
            "The epsilon quantities in this directory are extraction proxies, not exact CLs limits:",
            "",
            "```text",
            "eps2_95_exp_proxy = 1.6448536269514722 * sigma_A / A_per_eps2_unit",
            "eps2_95_obs_proxy = (max(A_hat, 0) + 1.6448536269514722 * sigma_A) / A_per_eps2_unit",
            "epsilon proxy = sqrt(eps2 proxy)",
            "```",
            "",
            "They are meaningful for lslb-to-lslb impact comparisons because `A_per_eps2_unit` is the same normalization used by the limit conversion. A final publication upper-limit statement still needs the exact CLs toy-scan/band workflow.",
            "",
            "## Top high-stat rows",
            "",
            md_table(
                top,
                [
                    "dataset",
                    "study",
                    "n_source_toys_median",
                    "score_lower_is_better",
                    "rms_pull_mean_nonzero",
                    "pull_width_median_nonzero",
                    "eps_width_ratio_median_all",
                    "eps_95_exp_proxy_median_nonzero",
                    "eps_95_obs_proxy_median_nonzero",
                ],
            ),
            "",
            "## Alternate-vs-best epsilon impact",
            "",
            md_table(
                comp_reduced,
                [
                    "dataset",
                    "best_study",
                    "comparison_study",
                    "median_eps95_exp_ratio_alt_over_best",
                    "median_eps95_obs_ratio_alt_over_best",
                    "median_pull_width_delta_alt_minus_best",
                    "median_pull_mean_delta_alt_minus_best",
                ],
            ),
            "",
            "## Primary outputs",
            "",
            "- `highstat_toy_level.csv`: toy-level rows with pull, recovery, qmu, and epsilon proxy columns.",
            "- `highstat_summary.csv`: grouped extraction comparisons by dataset, lslb, mass, and injected significance.",
            "- `highstat_ranking.csv`: score table used to rank the two high-stat rows.",
            "- `highstat_comparison_by_mass_strength.csv`: alternate-vs-best comparison at each mass and injected significance.",
            "- `plots/`: pull-width, mean-pull, epsilon-proxy, and alternate/best ratio plots.",
        ]
    )
    (STUDY_DIR / "HIGHSTAT_STUDY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(phase: str, frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not frames:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    toy = pd.concat(frames, ignore_index=True, sort=False)
    toy = toy.drop_duplicates(
        subset=["phase", "dataset", "study", "mass_GeV", "inj_nsigma", "toy_index", "injection_toy"],
        keep="last",
    )
    summary = summarize(toy)
    ranking = rank(summary)
    comparison = build_comparison(summary, ranking, phase)
    toy.to_csv(STUDY_DIR / f"{phase}_toy_level.csv", index=False)
    summary.to_csv(STUDY_DIR / f"{phase}_summary.csv", index=False)
    ranking.to_csv(STUDY_DIR / f"{phase}_ranking.csv", index=False)
    comparison.to_csv(STUDY_DIR / f"{phase}_comparison_by_mass_strength.csv", index=False)
    plot_outputs(summary, ranking, comparison, phase)
    return toy, summary, ranking, comparison


def default_highstat_candidates(pilot_ranking: pd.DataFrame | None = None) -> dict[str, list[float]]:
    out = {
        "2015": [1.1, 1.0],
        "2016": [1.1, 0.9],
        "2021": [1.1, 0.9],
    }
    if pilot_ranking is not None and not pilot_ranking.empty:
        out["2021"] = select_top_two(pilot_ranking, "2021", "pilot")
    return out


def run_phase(
    phase: str,
    candidates: dict[str, list[float]],
    *,
    max_toys: int,
    toy_start: int,
    force: bool,
    no_run: bool,
    pilot: bool,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = toy_indices(max_toys=max_toys, toy_start=toy_start)
    tasks = []
    for dataset, lslbs in candidates.items():
        masses = DATASETS[dataset].pilot_masses if pilot else DATASETS[dataset].masses
        for lslb in lslbs:
            if no_run:
                config_for(dataset, lslb, phase)
                continue
            tasks.append((dataset, lslb, masses))
    if tasks:
        max_workers = max(1, int(jobs))
        if max_workers == 1 or len(tasks) == 1:
            for dataset, lslb, masses in tasks:
                run_one(dataset, lslb, phase=phase, masses=masses, indices=indices, force=force)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
                futures = [
                    pool.submit(
                        run_one,
                        dataset,
                        lslb,
                        phase=phase,
                        masses=masses,
                        indices=indices,
                        force=force,
                    )
                    for dataset, lslb, masses in tasks
                ]
                for future in as_completed(futures):
                    future.result()
    return write_outputs(phase, load_phase_rows(phase, candidates))


def parse_dataset_list(raw: str) -> list[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in vals if x not in DATASETS]
    if bad:
        raise ValueError(f"Unknown dataset(s): {bad}")
    return vals or list(DATASETS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="2015,2016,2021", help="Comma-separated dataset keys to run.")
    parser.add_argument("--max-toys", type=int, default=25, help="Number of functional-form toys per high-stat row.")
    parser.add_argument("--toy-start", type=int, default=0, help="First toy index.")
    parser.add_argument("--pilot-toys", type=int, default=25, help="Number of toys for the 2021 lslb pilot.")
    parser.add_argument("--skip-pilot", action="store_true", help="Use default 2021 lslb choices instead of pilot-ranking 2021.")
    parser.add_argument("--jobs", type=int, default=3, help="Parallel (dataset, lslb) commands to run.")
    parser.add_argument("--force", action="store_true", help="Re-run existing rows.")
    parser.add_argument("--no-run", action="store_true", help="Only generate configs and summarize existing CSVs.")
    parser.add_argument("--only-summarize", action="store_true", help="Skip all command execution and summarize existing CSVs.")
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(DATASETS[d].toy_root) for d in datasets if not DATASETS[d].toy_root.exists()]
    if missing:
        raise FileNotFoundError("Missing mod_2 ROOT input(s): " + ", ".join(missing))

    pilot_ranking = pd.DataFrame()
    if "2021" in datasets and not args.skip_pilot:
        pilot_candidates = {"2021": DATASETS["2021"].lslb_candidates}
        if args.only_summarize:
            _, _, pilot_ranking, _ = write_outputs("pilot", load_phase_rows("pilot", pilot_candidates))
        else:
            _, _, pilot_ranking, _ = run_phase(
                "pilot",
                pilot_candidates,
                max_toys=int(args.pilot_toys),
                toy_start=int(args.toy_start),
                force=bool(args.force),
                no_run=bool(args.no_run),
                pilot=True,
                jobs=int(args.jobs),
            )

    candidates = {k: v for k, v in default_highstat_candidates(pilot_ranking).items() if k in datasets}
    if args.only_summarize:
        toy, summary, ranking, comparison = write_outputs("highstat", load_phase_rows("highstat", candidates))
    else:
        toy, summary, ranking, comparison = run_phase(
            "highstat",
            candidates,
            max_toys=int(args.max_toys),
            toy_start=int(args.toy_start),
            force=bool(args.force),
            no_run=bool(args.no_run),
            pilot=False,
            jobs=int(args.jobs),
        )

    if not summary.empty and not ranking.empty:
        write_markdown(summary, ranking, comparison, max_toys=int(args.max_toys))
    print(f"[done] wrote outputs under {STUDY_DIR}")


if __name__ == "__main__":
    main()
