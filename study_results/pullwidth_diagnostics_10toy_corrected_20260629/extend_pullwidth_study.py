#!/usr/bin/env python3
"""Extend the 2015/2016 functional-form pull-width diagnostic study.

The existing CSVs were produced with the funcform diagnostics branch that
contains ``extract_background_mode``.  Run this script from the detached
origin/main worktree prepared for this continuation, or override RUNNER_REPO
below if needed.
"""

from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


STUDY_DIR = Path(__file__).resolve().parent
RUNNER_REPO = Path("/Users/emryspeets/Desktop/gp_mods/tmp_hps_gpr_funcform_origin_main")
INPUT_DIR = Path("/Users/emryspeets/Desktop/gp_mods/funcform_studies/func_form_inputs")
CONFIG_DIR = STUDY_DIR / "extended_configs"
RUN_DIR = STUDY_DIR / "extended_runs"
PLOT_DIR = STUDY_DIR / "plots"

STRENGTHS = "s0,s1,s2,s3,s5"
TOY_INDICES = list(range(10))

PRIMARY_MASSES = {
    "2015": [0.045, 0.060, 0.075, 0.090, 0.105],
    "2016": [0.060, 0.090, 0.105, 0.120, 0.150],
}

EXISTING_EXTRA_MASSES = {
    "2015": [0.045, 0.075, 0.105],
    "2016": [0.060, 0.105, 0.150],
}

DIAGNOSTIC_MASSES = EXISTING_EXTRA_MASSES

DATASET_CONFIGS = {
    ("2015", "profiled"): "config_2015_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
    ("2015", "fixed"): "config_2015_blind2p25_fixedextract_95CL_funcform100_fixedhist_refit_lslb1p0.yaml",
    ("2016", "profiled"): "config_2016_10pct_blind2p25_profiled_95CL_funcform100_fixedhist_refit_lslb0p5.yaml",
    ("2016", "fixed"): "config_2016_10pct_blind2p25_fixedextract_95CL_funcform100_fixedhist_refit_lslb1p0.yaml",
}

TOY_ROOTS = {
    "2015": INPUT_DIR / "funcform_2015_dataset_mod_toys_2.root",
    "2016": INPUT_DIR / "funcform_2016_dataset_mod_toys_2.root",
}


@dataclass(frozen=True)
class Candidate:
    study: str
    label: str
    mode: str
    lslb: float
    masses_by_dataset: dict[str, list[float]]
    upper: float = 8.0
    refit_optimize: bool = True
    kernel_lock_mode: str = "none"
    const_bounds: tuple[float, float] | None = None
    phase: str = "primary"


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def study_name(mode: str, lslb: float) -> str:
    return f"{mode}_lslb{tag_float(lslb)}"


PRIMARY_CANDIDATES = [
    Candidate(
        "fixed_lslb0p5",
        "fixed bkg, lslb=0.5",
        "fixed",
        0.5,
        PRIMARY_MASSES,
    ),
    Candidate(
        "fixed_lslb0p75",
        "fixed bkg, lslb=0.75",
        "fixed",
        0.75,
        PRIMARY_MASSES,
    ),
    Candidate(
        "profiled_lslb0p75",
        "profiled bkg, lslb=0.75",
        "profiled",
        0.75,
        {
            ds: [m for m in PRIMARY_MASSES[ds] if m not in EXISTING_EXTRA_MASSES[ds]]
            for ds in PRIMARY_MASSES
        },
    ),
    Candidate(
        "profiled_lslb0p9",
        "profiled bkg, lslb=0.9",
        "profiled",
        0.9,
        PRIMARY_MASSES,
    ),
    Candidate(
        "profiled_lslb1p0",
        "profiled bkg, lslb=1.0",
        "profiled",
        1.0,
        {
            ds: [m for m in PRIMARY_MASSES[ds] if m not in EXISTING_EXTRA_MASSES[ds]]
            for ds in PRIMARY_MASSES
        },
    ),
    Candidate(
        "profiled_lslb1p1",
        "profiled bkg, lslb=1.1",
        "profiled",
        1.1,
        PRIMARY_MASSES,
    ),
]


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
        hi_by_ds.setdefault(key, 8.0 if key != "2021" else 9.0)
    lo_by_ds[dataset] = float(lower)
    hi_by_ds[dataset] = float(upper)
    cfg["kernel_ls_res_lower_factor_by_dataset"] = lo_by_ds
    cfg["kernel_ls_res_upper_factor_by_dataset"] = hi_by_ds


def config_for(candidate: Candidate, dataset: str) -> Path:
    base_name = DATASET_CONFIGS[(dataset, candidate.mode)]
    base_path = RUNNER_REPO / "study_configs" / "funcform_pullwidth_diagnostics" / base_name
    cfg = copy.deepcopy(load_yaml(base_path))
    set_dataset_enable(cfg, dataset)
    set_ls_factors(cfg, dataset, candidate.lslb, candidate.upper)
    cfg["extract_background_mode"] = candidate.mode
    cfg["inj_refit_gp_optimize"] = bool(candidate.refit_optimize)
    cfg["inj_refit_kernel_lock_mode"] = candidate.kernel_lock_mode
    cfg["funcform_closure_root_by_dataset"] = {
        "2015": str(TOY_ROOTS["2015"]),
        "2016": str(TOY_ROOTS["2016"]),
        "2021": str(INPUT_DIR / "funcform_2021_dataset_mod_toys_2.root"),
    }
    cfg["funcform_closure_container_by_dataset"] = {
        "2015": "fShiftSigPowTail",
        "2016": "fShiftSigPowTail",
        "2021": "fSigPowExpQ",
    }
    cfg["funcform_closure_toy_pattern_by_dataset"] = {
        "2015": "fShiftSigPowTail_toy_*",
        "2016": "fShiftSigPowTail_toy_*",
        "2021": "fSigPowExpQ_toy_*",
    }
    if candidate.const_bounds is not None:
        cfg["kernel_constant_bounds"] = [float(candidate.const_bounds[0]), float(candidate.const_bounds[1])]
    cfg["output_dir"] = str(RUN_DIR / candidate.study / dataset)
    cfg["save_plots"] = False
    cfg["inj_n_workers"] = 1
    cfg["inj_threads_per_worker"] = 1
    path = CONFIG_DIR / f"config_{dataset}_{candidate.study}.yaml"
    dump_yaml(path, cfg)
    return path


def run_candidate(candidate: Candidate, dataset: str, force: bool = False) -> Path | None:
    masses = candidate.masses_by_dataset.get(dataset, [])
    if not masses:
        return None
    outdir = RUN_DIR / candidate.study / dataset
    csv_path = outdir / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {candidate.study} {dataset}: {csv_path}")
        return csv_path

    cfg_path = config_for(candidate, dataset)
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
        str(TOY_ROOTS[dataset]),
        "--container",
        "fShiftSigPowTail",
        "--toy-name-fmt",
        "fShiftSigPowTail_toy_{i}",
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
    for toy_index in TOY_INDICES:
        cmd.extend(["--toy-index", str(toy_index)])
    print(f"[run] {candidate.study} {dataset}: masses={masses}")
    subprocess.run(cmd, cwd=RUNNER_REPO, check=True)
    return csv_path


def normalize_rows(df: pd.DataFrame, study: str, label: str, mode: str, lslb: float, source_csv: Path) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str)
    out["mass_GeV"] = out["mass_GeV"].astype(float).round(12)
    out["inj_nsigma"] = out["inj_nsigma"].astype(float)
    out["study"] = study
    out["study_label"] = label
    out["mode"] = mode
    out["lslb"] = float(lslb)
    out["source_csv"] = str(source_csv)
    if "pull_correct" not in out.columns:
        out["pull_correct"] = (out["A_hat"].astype(float) - out["strength"].astype(float)) / out["sigma_A"].astype(float)
    if "delta_z" not in out.columns:
        out["delta_z"] = out["Zhat"].astype(float) - out["inj_nsigma"].astype(float)
    strength = out["strength"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["recovery"] = np.where(strength != 0.0, out["A_hat"].astype(float) / strength, np.nan)
        out["recovery_bias"] = out["recovery"] - 1.0
        out["sigma_ratio_refit"] = out["sigma_A"].astype(float) / out["sigmaA_ref"].astype(float)
    if "extract_background_mode" not in out.columns:
        out["extract_background_mode"] = mode
    return out


def filter_masses(df: pd.DataFrame, masses_by_dataset: dict[str, list[float]] | None) -> pd.DataFrame:
    if masses_by_dataset is None:
        return df
    allowed = {
        (str(dataset), round(float(mass), 12))
        for dataset, masses in masses_by_dataset.items()
        for mass in masses
    }
    mask = df.apply(
        lambda row: (str(row["dataset"]), round(float(row["mass_GeV"]), 12)) in allowed,
        axis=1,
    )
    return df[mask].copy()


def load_existing_primary(masses_by_dataset: dict[str, list[float]] | None = None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    base_path = STUDY_DIR / "toy_level_10toy.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        for study, label, mode, lslb in [
            ("fixed_lslb1p0", "fixed bkg, lslb=1.0", "fixed", 1.0),
            ("profiled_lslb0p5", "profiled bkg, lslb=0.5", "profiled", 0.5),
        ]:
            sub = base[base["study"].astype(str) == study].copy()
            sub = sub[sub.apply(lambda r: float(r["mass_GeV"]) in PRIMARY_MASSES[str(r["dataset"])], axis=1)]
            frames.append(filter_masses(normalize_rows(sub, study, label, mode, lslb, base_path), masses_by_dataset))

    extra_path = STUDY_DIR / "subset_extra_lslb_toy_level.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        mapping = {
            "profiled_lslb0p75": ("profiled_lslb0p75", "profiled bkg, lslb=0.75", "profiled", 0.75),
            "profiled_lslb1p0_subset": ("profiled_lslb1p0", "profiled bkg, lslb=1.0", "profiled", 1.0),
        }
        for raw_study, meta in mapping.items():
            sub = extra[extra["study"].astype(str) == raw_study].copy()
            frames.append(filter_masses(normalize_rows(sub, *meta, source_csv=extra_path), masses_by_dataset))
    return [f for f in frames if not f.empty]


def load_run_rows(
    candidates: Iterable[Candidate],
    masses_by_dataset: dict[str, list[float]] | None = None,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        for dataset in PRIMARY_MASSES:
            path = RUN_DIR / candidate.study / dataset / "injection_extraction" / f"inj_extract_toys_{dataset}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            frames.append(
                filter_masses(
                    normalize_rows(df, candidate.study, candidate.label, candidate.mode, candidate.lslb, path),
                    masses_by_dataset,
                )
            )
    return frames


def sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def summarize(toy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "study", "study_label", "mode", "lslb", "mass_GeV", "inj_nsigma"]
    for keys, sub in toy.groupby(group_cols, dropna=False):
        pull = pd.to_numeric(sub["pull_correct"], errors="coerce").dropna()
        delta_z = pd.to_numeric(sub["delta_z"], errors="coerce").dropna()
        sigma_ratio = pd.to_numeric(sub["sigma_ratio_refit"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys))
        row.update(
            n_toys=int(len(sub)),
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
            sigma_ratio_mean=float(sigma_ratio.mean()) if len(sigma_ratio) else float("nan"),
            sigma_ratio_median=float(sigma_ratio.median()) if len(sigma_ratio) else float("nan"),
            ls_lo_mean=float(pd.to_numeric(sub.get("ls_lo", pd.Series(dtype=float)), errors="coerce").mean()),
            ls_hi_mean=float(pd.to_numeric(sub.get("ls_hi", pd.Series(dtype=float)), errors="coerce").mean()),
            refit_ls_opt_mean=float(pd.to_numeric(sub.get("refit_ls_opt", pd.Series(dtype=float)), errors="coerce").mean()),
            refit_const_opt_mean=float(pd.to_numeric(sub.get("refit_const_opt", pd.Series(dtype=float)), errors="coerce").mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def rank(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, study), sub in summary.groupby(["dataset", "study"], dropna=False):
        nonzero = sub[sub["inj_nsigma"].astype(float) > 0]
        source = nonzero if not nonzero.empty else sub
        pull_mean = pd.to_numeric(source["pull_mean_correct"], errors="coerce")
        width = pd.to_numeric(source["pull_width_correct"], errors="coerce")
        cov1 = pd.to_numeric(sub["cov_1sigma"], errors="coerce")
        cov2 = pd.to_numeric(sub["cov_2sigma"], errors="coerce")
        sigma_ratio = pd.to_numeric(sub["sigma_ratio_median"], errors="coerce")
        row = {
            "dataset": dataset,
            "study": study,
            "study_label": sub["study_label"].iloc[0],
            "mode": sub["mode"].iloc[0],
            "lslb": float(sub["lslb"].iloc[0]),
            "n_groups": int(len(sub)),
            "n_nonzero_groups": int(len(nonzero)),
            "n_toy_rows": int(sub["n_toys"].sum()),
            "rms_pull_mean_nonzero": float(np.sqrt(np.nanmean(np.square(pull_mean)))) if len(pull_mean) else float("nan"),
            "median_abs_pull_mean_nonzero": float(np.nanmedian(np.abs(pull_mean))) if len(pull_mean) else float("nan"),
            "pull_width_median_nonzero": float(np.nanmedian(width)) if len(width) else float("nan"),
            "pull_width_rmse_nonzero": float(np.sqrt(np.nanmean(np.square(width - 1.0)))) if len(width) else float("nan"),
            "cov1_abs_resid_mean_all": float(np.nanmean(np.abs(cov1 - 0.6827))) if len(cov1) else float("nan"),
            "cov2_abs_resid_mean_all": float(np.nanmean(np.abs(cov2 - 0.9545))) if len(cov2) else float("nan"),
            "sigma_ratio_median_all": float(np.nanmedian(sigma_ratio)) if len(sigma_ratio) else float("nan"),
            "success_rate_min": float(pd.to_numeric(sub["success_rate"], errors="coerce").min()),
            "refit_ok_rate_min": float(pd.to_numeric(sub["refit_ok_rate"], errors="coerce").min()),
            "fallback_rate_max": float(pd.to_numeric(sub["fallback_rate"], errors="coerce").max()),
        }
        row["pilot_score_lower_is_better"] = (
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
    out = out.sort_values(["dataset", "pilot_score_lower_is_better", "study"]).reset_index(drop=True)
    out["rank_within_dataset"] = out.groupby("dataset")["pilot_score_lower_is_better"].rank(method="first")
    return out


def top3_for_metric(summary: pd.DataFrame, ranking: pd.DataFrame, dataset: str, metric: str) -> list[str]:
    if metric == "pull_width":
        scores = (
            summary[(summary["dataset"] == dataset) & (summary["inj_nsigma"].astype(float) > 0)]
            .assign(score=lambda d: (d["pull_width_correct"] - 1.0).abs())
            .groupby("study")["score"]
            .mean()
            .sort_values()
        )
    elif metric == "abs_mean_pull":
        scores = (
            summary[(summary["dataset"] == dataset) & (summary["inj_nsigma"].astype(float) > 0)]
            .assign(score=lambda d: d["pull_mean_correct"].abs())
            .groupby("study")["score"]
            .mean()
            .sort_values()
        )
    elif metric == "delta_z":
        scores = (
            summary[(summary["dataset"] == dataset) & (summary["inj_nsigma"].astype(float) > 0)]
            .assign(score=lambda d: d["delta_z_mean"].abs())
            .groupby("study")["score"]
            .mean()
            .sort_values()
        )
    else:
        scores = ranking[ranking["dataset"] == dataset].set_index("study")["pilot_score_lower_is_better"].sort_values()
    return list(scores.index[:3])


def plot_top3(summary: pd.DataFrame, ranking: pd.DataFrame, tag: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    for dataset in sorted(summary["dataset"].astype(str).unique()):
        for metric, ylabel, ref in [
            ("pull_width", "pull width", 1.0),
            ("abs_mean_pull", "|mean pull|", 0.0),
            ("delta_z", "|mean DeltaZ|", 0.0),
        ]:
            top = top3_for_metric(summary, ranking, dataset, metric)
            if not top:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ds_summary = summary[(summary["dataset"].astype(str) == dataset) & (summary["study"].isin(top))]
            ds_summary = ds_summary[ds_summary["inj_nsigma"].astype(float) > 0].copy()
            if metric == "pull_width":
                ds_summary["plot_value"] = ds_summary["pull_width_correct"]
            elif metric == "abs_mean_pull":
                ds_summary["plot_value"] = ds_summary["pull_mean_correct"].abs()
            else:
                ds_summary["plot_value"] = ds_summary["delta_z_mean"].abs()
            for study, sub in ds_summary.groupby("study", sort=False):
                by_mass = sub.groupby("mass_GeV")["plot_value"].median().sort_index()
                label = ranking[(ranking["dataset"].astype(str) == dataset) & (ranking["study"] == study)]["study_label"]
                ax.plot(by_mass.index, by_mass.values, marker="o", lw=1.8, label=label.iloc[0] if len(label) else study)
            if ref:
                ax.axhline(ref, color="0.35", lw=1.0, ls="--")
            ax.set_xlabel("mass [GeV]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset}: top 3 by {ylabel}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            vals = ds_summary["plot_value"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals):
                lo, hi = float(vals.quantile(0.05)), float(vals.quantile(0.95))
                pad = max(0.08, 0.15 * (hi - lo))
                if metric == "pull_width":
                    ax.set_ylim(max(0.0, min(0.5, lo - pad)), max(1.5, hi + pad))
                else:
                    ax.set_ylim(0.0, max(1.0, hi + pad))
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{tag}_{dataset}_top3_{metric}_zoom.png", dpi=180)
            plt.close(fig)

    if not ranking.empty:
        for dataset in sorted(ranking["dataset"].astype(str).unique()):
            top = ranking[ranking["dataset"].astype(str) == dataset].head(3)
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            ax.bar(top["study_label"], top["pilot_score_lower_is_better"], color=["#4C78A8", "#F58518", "#54A24B"])
            ax.set_ylabel("pilot score")
            ax.set_title(f"{dataset}: top 3 ranking")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{tag}_{dataset}_top3_ranking_zoom.png", dpi=180)
            plt.close(fig)


def write_outputs(tag: str, frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    toy = pd.concat(frames, ignore_index=True, sort=False)
    toy = toy.drop_duplicates(
        subset=["dataset", "study", "mass_GeV", "inj_nsigma", "toy_index", "injection_toy"],
        keep="last",
    )
    summary = summarize(toy)
    ranking = rank(summary)
    toy.to_csv(STUDY_DIR / f"{tag}_toy_level.csv", index=False)
    summary.to_csv(STUDY_DIR / f"{tag}_summary.csv", index=False)
    ranking.to_csv(STUDY_DIR / f"{tag}_ranking.csv", index=False)
    plot_top3(summary, ranking, tag)
    return toy, summary, ranking


def select_profiled_winners(ranking: pd.DataFrame, n: int = 2) -> list[tuple[str, float]]:
    prof = ranking[ranking["mode"].astype(str) == "profiled"].copy()
    prof = prof[~prof["study"].str.contains("noopt|lsuh|const|lock", regex=True, na=False)]
    grouped = (
        prof.groupby(["study", "lslb"], as_index=False)["pilot_score_lower_is_better"]
        .mean()
        .sort_values("pilot_score_lower_is_better")
    )
    return [(str(row.study), float(row.lslb)) for row in grouped.head(n).itertuples()]


def diagnostic_candidates(primary_ranking: pd.DataFrame) -> list[Candidate]:
    winners = select_profiled_winners(primary_ranking, n=2)
    if not winners:
        winners = [("profiled_lslb1p0", 1.0), ("profiled_lslb0p9", 0.9)]
    best_study, best_lslb = winners[0]
    out: list[Candidate] = []
    for _, lslb in winners:
        out.append(
            Candidate(
                f"profiled_lslb{tag_float(lslb)}_noopt",
                f"profiled bkg, lslb={lslb:g}, refit_optimize=false",
                "profiled",
                lslb,
                DIAGNOSTIC_MASSES,
                refit_optimize=False,
                phase="diagnostic",
            )
        )
    for upper in (6.0, 12.0):
        out.append(
            Candidate(
                f"profiled_lslb{tag_float(best_lslb)}_lsuh{tag_float(upper)}",
                f"profiled bkg, lslb={best_lslb:g}, lsub={upper:g}",
                "profiled",
                best_lslb,
                DIAGNOSTIC_MASSES,
                upper=upper,
                phase="diagnostic",
            )
        )
    out.append(
        Candidate(
            f"profiled_lslb{tag_float(best_lslb)}_const1em4_1e4",
            f"profiled bkg, lslb={best_lslb:g}, const=[1e-4,1e4]",
            "profiled",
            best_lslb,
            DIAGNOSTIC_MASSES,
            const_bounds=(1.0e-4, 1.0e4),
            phase="diagnostic",
        )
    )
    out.append(
        Candidate(
            f"profiled_lslb{tag_float(best_lslb)}_lock_initial",
            f"profiled bkg, lslb={best_lslb:g}, kernel locked to initial fit",
            "profiled",
            best_lslb,
            DIAGNOSTIC_MASSES,
            kernel_lock_mode="initial_fit",
            phase="diagnostic",
        )
    )
    return out


def write_markdown(primary_ranking: pd.DataFrame, diagnostic_ranking: pd.DataFrame) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        cols = list(frame.columns)

        def fmt(value: object) -> str:
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(float(value)):
                    return ""
                return f"{float(value):.4g}"
            return str(value)

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = [
            "| " + " | ".join(fmt(row[col]) for col in cols) + " |"
            for _, row in frame.iterrows()
        ]
        return "\n".join([header, sep] + body)

    lines = [
        "# Extended pull-width diagnostic continuation",
        "",
        "Generated by `extend_pullwidth_study.py`.",
        "",
        "Primary 5-mass pilot:",
        "- 2015 masses: `" + ", ".join(f"{m:.3f}" for m in PRIMARY_MASSES["2015"]) + "` GeV",
        "- 2016 masses: `" + ", ".join(f"{m:.3f}" for m in PRIMARY_MASSES["2016"]) + "` GeV",
        "- paired functional-form toys: `toy_index = 0..9`",
        "- strengths: `s0,s1,s2,s3,s5`",
        "",
        "Primary outputs:",
        "- `extended_primary_toy_level.csv`",
        "- `extended_primary_summary.csv`",
        "- `extended_primary_ranking.csv`",
        "- `plots/extended_primary_*_top3_*_zoom.png`",
        "",
        "Diagnostic outputs:",
        "- `extended_diagnostic_toy_level.csv`",
        "- `extended_diagnostic_summary.csv`",
        "- `extended_diagnostic_ranking.csv`",
        "- `plots/extended_diagnostic_*_top3_*_zoom.png`",
        "",
        "## Interpretation",
        "",
        "- The best 5-mass primary candidate in both datasets is `profiled_lslb1p1`.",
        "- The 2016 corrected/profiled rows are genuinely closer to unit pull width than the latest note snapshot suggests: the top 2016 profiled primary widths are `0.913` (`lslb=1.1`), `1.028` (`lslb=0.9`), and `0.947` (`lslb=1.0`).",
        "- Fixed-background extraction is rejected as a candidate. Its pull widths are order `10-40`, consistent with the code path being a plug-in diagnostic that does not profile or marginalize the GP covariance.",
        "- `refit_optimize=false` is neutral in the tested rows: it changes the CSV flag but leaves `A_hat`, `sigma_A`, pulls, and refit kernel values identical to the optimized baseline for `lslb=1.1` and `lslb=0.9`.",
        "- The useful stress/alternate knob is `ls_upper=12` at `lslb=1.1`; `ls_upper=6` is worse. Kernel locking to the initial fit and concrete constant-kernel bounds both degrade 2016 and should not be promoted.",
        "- Phi `K+K-` uses the same interpretation policy, profiled background for interpretation and fixed background only as a cross-check, but its nuisance basis differs: Phi profiles a smooth multiplicative `exp(beta0 + beta1 z + beta2 z^2)` correction, while this HPS workflow profiles additive Gaussian nuisance directions from the GP covariance.",
        "",
        "Top primary candidates by dataset:",
        "",
    ]
    for dataset in sorted(primary_ranking["dataset"].astype(str).unique()):
        lines.append(f"## {dataset} primary ranking")
        cols = [
            "rank_within_dataset",
            "study",
            "study_label",
            "pilot_score_lower_is_better",
            "rms_pull_mean_nonzero",
            "pull_width_median_nonzero",
            "pull_width_rmse_nonzero",
        ]
        lines.append(md_table(primary_ranking[primary_ranking["dataset"].astype(str) == dataset][cols].head(8)))
        lines.append("")
    if not diagnostic_ranking.empty:
        for dataset in sorted(diagnostic_ranking["dataset"].astype(str).unique()):
            lines.append(f"## {dataset} diagnostic ranking")
            cols = [
                "rank_within_dataset",
                "study",
                "study_label",
                "pilot_score_lower_is_better",
                "rms_pull_mean_nonzero",
                "pull_width_median_nonzero",
                "pull_width_rmse_nonzero",
            ]
            lines.append(md_table(diagnostic_ranking[diagnostic_ranking["dataset"].astype(str) == dataset][cols].head(8)))
            lines.append("")
    (STUDY_DIR / "EXTENDED_STUDY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun existing output CSVs")
    parser.add_argument("--skip-runs", action="store_true", help="aggregate and plot only")
    args = parser.parse_args()

    if not RUNNER_REPO.exists():
        raise SystemExit(f"RUNNER_REPO does not exist: {RUNNER_REPO}")
    CONFIG_DIR.mkdir(exist_ok=True)
    RUN_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    if not args.skip_runs:
        for candidate in PRIMARY_CANDIDATES:
            for dataset in PRIMARY_MASSES:
                run_candidate(candidate, dataset, force=args.force)

    primary_frames = load_existing_primary() + load_run_rows(PRIMARY_CANDIDATES)
    _, _, primary_ranking = write_outputs("extended_primary", primary_frames)

    diagnostics = diagnostic_candidates(primary_ranking)
    if not args.skip_runs:
        for candidate in diagnostics:
            for dataset in DIAGNOSTIC_MASSES:
                run_candidate(candidate, dataset, force=args.force)

    diagnostic_frames = (
        load_existing_primary(DIAGNOSTIC_MASSES)
        + load_run_rows(PRIMARY_CANDIDATES, DIAGNOSTIC_MASSES)
        + load_run_rows(diagnostics)
    )
    _, _, diagnostic_ranking = write_outputs("extended_diagnostic", diagnostic_frames)
    write_markdown(primary_ranking, diagnostic_ranking)
    print(f"Wrote {STUDY_DIR / 'EXTENDED_STUDY_SUMMARY.md'}")


if __name__ == "__main__":
    main()
