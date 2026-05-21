#!/usr/bin/env python3
"""Build functional-form closure and toy upper-limit proxy figures for the note."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = NOTE_DIR / "toy_generation_figs"
ONE_SIDED_95 = 1.6448536269514722


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    function_tag: str
    source_csv: Path
    study_key: str | None = None


DATASETS = [
    DatasetSpec(
        key="2015",
        label="HPS 2015",
        function_tag="fShiftSigPowTail",
        source_csv=Path(
            "/Users/emryspeets/Desktop/gp_mods/2015_gpr/new_refit/new_wave/"
            "blind2p25_refmatch_comparison_suite/toy_rows_with_derived_metrics.csv"
        ),
        study_key="wide_refmatched",
    ),
    DatasetSpec(
        key="2016",
        label="HPS 2016 10%",
        function_tag="fShiftSigPowTail",
        source_csv=Path("/Users/emryspeets/Desktop/gp_mods/2016_gpr/closure_final/inj_extract_toys_2016.csv"),
    ),
    DatasetSpec(
        key="2021",
        label="HPS 2021 1%",
        function_tag="fSigPowExpQ",
        source_csv=Path(
            "/Users/emryspeets/Desktop/gp_mods/2021_gpr/closure_1pct/validation_suite/"
            "merged_inj_extract_toys_2021_fSigPowExpQ.csv"
        ),
    ),
]


def truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0) != 0.0
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def first_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return np.nan
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(vals.iloc[0]) if len(vals) else np.nan


def read_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.source_csv.exists():
        raise FileNotFoundError(spec.source_csv)
    df = pd.read_csv(spec.source_csv)
    if spec.study_key is not None:
        if "study_key" not in df:
            raise ValueError(f"{spec.source_csv} does not contain study_key")
        df = df[df["study_key"] == spec.study_key].copy()
    required = {
        "mass_GeV",
        "inj_nsigma",
        "strength",
        "A_hat",
        "sigma_A",
        "Zhat",
        "pull_param",
        "A_per_eps2_unit",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{spec.source_csv} is missing required columns: {missing}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=sorted(required))
    if "success" in df:
        df = df[truthy_mask(df["success"])].copy()
    df["mass_MeV"] = 1000.0 * df["mass_GeV"].astype(float)
    df["inj_nsigma"] = df["inj_nsigma"].astype(float)
    df["delta_z"] = df["Zhat"].astype(float) - df["inj_nsigma"]
    df["ahat_over_ainj"] = np.where(
        df["inj_nsigma"] > 0.0,
        df["A_hat"].astype(float) / df["strength"].astype(float),
        np.nan,
    )
    return df


def mean_err(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return np.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def summarize_closure(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mass_gev, mass_mev, zinj), group in df.groupby(["mass_GeV", "mass_MeV", "inj_nsigma"]):
        pull = group["pull_param"].astype(float).dropna()
        dz = group["delta_z"].astype(float).dropna()
        ratio = group["ahat_over_ainj"].astype(float).dropna()
        pull_width = float(pull.std(ddof=1)) if len(pull) > 1 else np.nan
        rows.append(
            {
                "mass_GeV": float(mass_gev),
                "mass_MeV": float(mass_mev),
                "inj_nsigma": float(zinj),
                "n_toys": int(len(group)),
                "pull_mean": float(pull.mean()) if len(pull) else np.nan,
                "pull_mean_err": mean_err(pull),
                "pull_width": pull_width,
                "pull_width_err": float(pull_width / math.sqrt(2.0 * (len(pull) - 1))) if len(pull) > 1 else np.nan,
                "delta_z_mean": float(dz.mean()) if len(dz) else np.nan,
                "delta_z_err": mean_err(dz),
                "ahat_over_ainj_mean": float(ratio.mean()) if len(ratio) else np.nan,
                "ahat_over_ainj_err": mean_err(ratio),
                "blind_nsigma": first_numeric(group, "blind_nsigma"),
                "train_exclude_nsigma": first_numeric(group, "train_exclude_nsigma"),
            }
        )
    return pd.DataFrame(rows).sort_values(["mass_MeV", "inj_nsigma"]).reset_index(drop=True)


def summarize_limit_proxy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    zero = df[np.isclose(df["inj_nsigma"], 0.0)].copy()
    zero = zero[(zero["sigma_A"] > 0.0) & (zero["A_per_eps2_unit"] > 0.0)].copy()
    if zero.empty:
        raise ValueError("No usable background-only toy rows")

    zero["A95_toy_proxy"] = np.maximum(0.0, zero["A_hat"].astype(float)) + ONE_SIDED_95 * zero["sigma_A"].astype(float)
    zero["eps2_95_toy_proxy"] = zero["A95_toy_proxy"] / zero["A_per_eps2_unit"].astype(float)

    rows = []
    for (mass_gev, mass_mev), group in zero.groupby(["mass_GeV", "mass_MeV"]):
        ahat_mean = float(group["A_hat"].mean())
        sigma_a_mean = float(group["sigma_A"].mean())
        a_per_eps2_mean = float(group["A_per_eps2_unit"].mean())
        a95_mean_proxy = max(0.0, ahat_mean) + ONE_SIDED_95 * sigma_a_mean
        row = {
            "mass_GeV": float(mass_gev),
            "mass_MeV": float(mass_mev),
            "n_toys": int(len(group)),
            "A_hat_mean": ahat_mean,
            "sigma_A_mean": sigma_a_mean,
            "A_per_eps2_unit_mean": a_per_eps2_mean,
            "A95_mean_Ahat_proxy": a95_mean_proxy,
            "eps2_95_mean_Ahat_proxy": a95_mean_proxy / a_per_eps2_mean,
            "blind_nsigma": first_numeric(group, "blind_nsigma"),
            "train_exclude_nsigma": first_numeric(group, "train_exclude_nsigma"),
        }
        for column in ["A95_toy_proxy", "eps2_95_toy_proxy"]:
            vals = group[column].astype(float).dropna()
            row[f"{column}_q025"] = float(vals.quantile(0.025))
            row[f"{column}_q16"] = float(vals.quantile(0.16))
            row[f"{column}_median"] = float(vals.quantile(0.50))
            row[f"{column}_q84"] = float(vals.quantile(0.84))
            row[f"{column}_q975"] = float(vals.quantile(0.975))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mass_MeV").reset_index(drop=True)
    return zero, summary


def write_summary_csvs(spec: DatasetSpec, closure: pd.DataFrame, limit_rows: pd.DataFrame, limit_summary: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    closure.to_csv(OUT_DIR / f"closure_summary_{spec.key}_{spec.function_tag}.csv", index=False)
    limit_rows.to_csv(OUT_DIR / f"toy_upper_limit_proxy_rows_{spec.key}_{spec.function_tag}.csv", index=False)
    limit_summary.to_csv(OUT_DIR / f"toy_upper_limit_proxy_summary_{spec.key}_{spec.function_tag}.csv", index=False)


def plot_closure_suite(spec: DatasetSpec, summary: pd.DataFrame) -> Path:
    z_values = [0.0, 1.0, 3.0, 5.0]
    colors = {0.0: "#4C78A8", 1.0: "#F58518", 3.0: "#B279A2", 5.0: "#E45756"}
    markers = {0.0: "o", 1.0: "s", 3.0: "D", 5.0: "P"}
    blind = first_numeric(summary, "blind_nsigma")
    train = first_numeric(summary, "train_exclude_nsigma")
    n_per = int(summary["n_toys"].median()) if len(summary) else 0

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), sharex=True, constrained_layout=True)
    ax_pull_mean, ax_pull_width, ax_recovery, ax_dz = axes.ravel()
    for zinj in z_values:
        rows = summary[np.isclose(summary["inj_nsigma"], zinj)].sort_values("mass_MeV")
        if rows.empty:
            continue
        style = dict(
            color=colors[zinj],
            marker=markers[zinj],
            ms=4.5,
            lw=1.5,
            capsize=2.5,
            label=rf"$Z_{{\rm inj}}={zinj:g}$",
        )
        ax_pull_mean.errorbar(rows["mass_MeV"], rows["pull_mean"], yerr=rows["pull_mean_err"], **style)
        ax_pull_width.errorbar(rows["mass_MeV"], rows["pull_width"], yerr=rows["pull_width_err"], **style)
        ax_dz.errorbar(rows["mass_MeV"], rows["delta_z_mean"], yerr=rows["delta_z_err"], **style)
        if zinj > 0.0:
            ax_recovery.errorbar(
                rows["mass_MeV"],
                rows["ahat_over_ainj_mean"],
                yerr=rows["ahat_over_ainj_err"],
                **style,
            )

    ax_pull_mean.axhline(0.0, color="0.25", lw=1.0, ls="--")
    ax_pull_width.axhline(1.0, color="0.25", lw=1.0, ls="--")
    ax_recovery.axhline(1.0, color="0.25", lw=1.0, ls="--")
    ax_dz.axhline(0.0, color="0.25", lw=1.0, ls="--")

    ax_pull_mean.set_title("Pull mean")
    ax_pull_width.set_title("Pull width")
    ax_recovery.set_title(r"Recovered amplitude, $\hat A/A_{\rm inj}$")
    ax_dz.set_title(r"Significance residual, $\hat Z-Z_{\rm inj}$")
    ax_pull_mean.set_ylabel("mean pull")
    ax_pull_width.set_ylabel("pull width")
    ax_recovery.set_ylabel(r"$\hat A/A_{\rm inj}$")
    ax_dz.set_ylabel(r"$\Delta Z$")
    ax_recovery.set_xlabel(r"mass hypothesis [MeV]")
    ax_dz.set_xlabel(r"mass hypothesis [MeV]")
    for ax in axes.ravel():
        ax.grid(True, color="0.88", lw=0.7)
        ax.set_axisbelow(True)
        ax.margins(x=0.03)
    ax_pull_mean.set_ylim(-0.85, 0.85)
    ax_pull_width.set_ylim(0.55, 1.45)
    recovery_rows = summary[summary["inj_nsigma"] > 0.0].copy()
    recovery_low = (
        recovery_rows["ahat_over_ainj_mean"] - recovery_rows["ahat_over_ainj_err"].fillna(0.0)
    ).min()
    recovery_high = (
        recovery_rows["ahat_over_ainj_mean"] + recovery_rows["ahat_over_ainj_err"].fillna(0.0)
    ).max()
    ax_recovery.set_ylim(
        min(0.45, float(recovery_low) - 0.08),
        max(1.55, float(recovery_high) + 0.08),
    )
    ax_dz.set_ylim(-1.1, 1.1)

    handles, labels = ax_pull_mean.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"{spec.label}: {spec.function_tag} full-refit closure "
        f"(blind/train={blind:g}/{train:g} sigma, N~{n_per} toys per point)",
        y=1.055,
        fontsize=11,
        fontweight="semibold",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"closure_validation_suite_{spec.key}_{spec.function_tag}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_limit_proxy(limit_summaries: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 6.0),
        sharey="row",
        constrained_layout=True,
    )
    band95 = "#cfe8f3"
    band68 = "#86bddb"
    line = "#08519c"

    for col, spec in enumerate(DATASETS):
        summary = limit_summaries[spec.key].sort_values("mass_MeV")
        x = summary["mass_MeV"].to_numpy(float)
        for row, (quantity, line_col, ylabel) in enumerate(
            [
                ("A95_toy_proxy", "A95_mean_Ahat_proxy", r"$A_{95}$ proxy [events]"),
                ("eps2_95_toy_proxy", "eps2_95_mean_Ahat_proxy", r"$\epsilon^2_{95}$ proxy"),
            ]
        ):
            ax = axes[row, col]
            ax.fill_between(
                x,
                summary[f"{quantity}_q025"].to_numpy(float),
                summary[f"{quantity}_q975"].to_numpy(float),
                color=band95,
                alpha=0.55,
                linewidth=0,
                label="central 95% toys",
            )
            ax.fill_between(
                x,
                summary[f"{quantity}_q16"].to_numpy(float),
                summary[f"{quantity}_q84"].to_numpy(float),
                color=band68,
                alpha=0.60,
                linewidth=0,
                label="central 68% toys",
            )
            ax.plot(x, summary[line_col].to_numpy(float), color=line, lw=1.9, label=r"mean-$\hat A$ proxy")
            ax.set_yscale("log")
            ax.grid(True, which="both", color="0.88", lw=0.7)
            ax.set_axisbelow(True)
            ax.margins(x=0.04)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(spec.label)
            if row == 1:
                ax.set_xlabel(r"mass hypothesis [MeV]")

    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "toy_upper_limit_proxy_2015_2016_2021.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    limit_summaries: dict[str, pd.DataFrame] = {}
    for spec in DATASETS:
        toys = read_dataset(spec)
        closure = summarize_closure(toys)
        limit_rows, limit_summary = summarize_limit_proxy(toys)
        write_summary_csvs(spec, closure, limit_rows, limit_summary)
        limit_summaries[spec.key] = limit_summary
        print(f"Wrote {plot_closure_suite(spec, closure)}")
    print(f"Wrote {plot_combined_limit_proxy(limit_summaries)}")


if __name__ == "__main__":
    main()
