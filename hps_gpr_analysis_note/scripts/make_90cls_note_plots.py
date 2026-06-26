from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


BASE_DIR = Path("/Users/emryspeets/Desktop/gp_mods/combined_15_16_10pct_21_1pct")
RUN90_V2 = BASE_DIR / "90cls_plots" / "v2"
RUN90_V1 = BASE_DIR / "90cls_plots" / "v1"
CORRECTED95 = BASE_DIR / "corrected_2"
OUTDIR = RUN90_V2 / "note_comparison_plots"

BABAR_SOURCE = Path("/Users/emryspeets/Desktop/2026_winter/BaBar_Lees2014xha.txt")
CL_RATIO_95_OVER_90_EPS2 = 1.6448536269514722 / 1.2815515655446004

BAND_COLS = ["eps2_lo2", "eps2_lo1", "eps2_med", "eps2_hi1", "eps2_hi2"]
BAND_ALIAS = {
    "eps2_lo2": "toy_eps2_uls_q02",
    "eps2_lo1": "toy_eps2_uls_q16",
    "eps2_med": "toy_eps2_uls_q50",
    "eps2_hi1": "toy_eps2_uls_q84",
    "eps2_hi2": "toy_eps2_uls_q97",
}

DATASET_COLORS = {
    "2015": "#4C72B0",
    "2016": "#DD8452",
    "2021": "#55A868",
}
COMBINED_COLOR = "#111111"
PROJECTION_COLOR = "#7A3DBB"
BABAR_COLOR = "#D89400"
SIGMA_REFERENCE_LEVELS = (1.0, 2.0, 3.0)
SIGMA_REFERENCE_COLORS = {
    1.0: "#0072B2",
    2.0: "#D55E00",
    3.0: "#CC79A7",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 10.5,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#d7d7d7", linewidth=0.8, alpha=0.85)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.55, alpha=0.75)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.0, length=6)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)


def save_figure(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUTDIR)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def log_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    valid = np.isfinite(xp) & np.isfinite(fp) & (fp > 0.0)
    xp = xp[valid]
    fp = fp[valid]
    order = np.argsort(xp)
    return np.exp(np.interp(x, xp[order], np.log(fp[order])))


def one_sided_gaussian_p(z: float) -> float:
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def local_p_from_global_sidak(p_global: float, neff: float) -> float:
    pg = float(np.clip(p_global, 0.0, 1.0))
    n_eff = max(float(neff), 1.0)
    return float(np.clip(-np.expm1(np.log1p(-pg) / n_eff), 0.0, 1.0))


def effective_trials_from_spacing(
    masses: np.ndarray,
    sigma_vals,
    *,
    indep_width_sigma: float,
) -> float:
    masses_arr = np.asarray(masses, dtype=float)
    finite_masses = masses_arr[np.isfinite(masses_arr)]
    if finite_masses.size < 2:
        return 1.0
    if sigma_vals is None:
        return float(max(1.0, float(finite_masses.size)))

    sigma_arr = np.asarray(sigma_vals, dtype=float)
    if sigma_arr.size != masses_arr.size:
        return float(max(1.0, float(finite_masses.size)))

    dm = np.diff(masses_arr)
    sig_mid = 0.5 * (sigma_arr[:-1] + sigma_arr[1:])
    ok = np.isfinite(dm) & (dm > 0.0) & np.isfinite(sig_mid) & (sig_mid > 0.0)
    if not np.any(ok):
        return 1.0
    neff = np.sum(dm[ok] / (max(float(indep_width_sigma), 1e-6) * sig_mid[ok]))
    return float(np.clip(neff, 1.0, float(finite_masses.size)))


def plot_segments(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_gap: float = 1.51,
    **kwargs,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return
    label = kwargs.pop("label", None)
    breaks = np.flatnonzero(np.diff(x) > max_gap) + 1
    for iseg, idx in enumerate(np.split(np.arange(x.size), breaks)):
        if idx.size:
            ax.plot(x[idx], y[idx], label=label if iseg == 0 else "_nolegend_", **kwargs)


def fill_between_segments(
    ax: plt.Axes,
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    max_gap: float = 1.51,
    **kwargs,
) -> None:
    x = np.asarray(x, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2) & (y1 > 0.0) & (y2 > 0.0)
    x = x[valid]
    y1 = y1[valid]
    y2 = y2[valid]
    if x.size == 0:
        return
    label = kwargs.pop("label", None)
    breaks = np.flatnonzero(np.diff(x) > max_gap) + 1
    for iseg, idx in enumerate(np.split(np.arange(x.size), breaks)):
        if idx.size:
            ax.fill_between(x[idx], y1[idx], y2[idx], label=label if iseg == 0 else "_nolegend_", **kwargs)


def add_epsilon_columns(df: pd.DataFrame, eps2_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = eps2_cols or [c for c in out.columns if c.startswith("eps2_") or c.startswith("ul_eps2")]
    for col in cols:
        if col in out.columns:
            out[col.replace("eps2", "epsilon")] = np.sqrt(out[col].clip(lower=0.0))
    return out


def parse_dataset_set(value: str) -> list[str]:
    return [part.strip() for part in str(value).split("+") if part.strip()]


def detect_coherent_band_spikes(df: pd.DataFrame) -> list[tuple[int, dict[str, float]]]:
    spikes: list[tuple[int, dict[str, float]]] = []
    work = df.sort_values(["dataset_set", "mass_GeV"]).copy()
    for _, group in work.groupby("dataset_set", sort=False):
        idxs = group.index.to_list()
        masses = group["mass_GeV"].to_numpy(float)
        for pos in range(1, len(group) - 1):
            if masses[pos] - masses[pos - 1] > 0.0016 or masses[pos + 1] - masses[pos] > 0.0016:
                continue
            ratios: dict[str, float] = {}
            coherent = True
            for col in BAND_COLS:
                prev_val = float(group.iloc[pos - 1][col])
                next_val = float(group.iloc[pos + 1][col])
                this_val = float(group.iloc[pos][col])
                if min(prev_val, next_val, this_val) <= 0.0:
                    coherent = False
                    break
                expected = float(np.exp(0.5 * (np.log(prev_val) + np.log(next_val))))
                ratios[col] = this_val / expected
            if coherent and min(ratios.values()) > 1.45:
                spikes.append((idxs[pos], ratios))
    return spikes


def corrected_90_bands() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RUN90_V2 / "combined_ul_bands_combined_all.csv")
    scan = pd.read_csv(RUN90_V2 / "combined_combined.csv").rename(columns={"eps2_up": "eps2_scan"})
    df = raw.copy()
    df["mass_MeV"] = 1000.0 * df["mass_GeV"]
    df["observed_source"] = "combined_ul_bands_combined_all.csv"
    df["expected_band_source"] = "combined_ul_bands_combined_all.csv"
    hard_observed_mismatch = np.log(1.20)

    scan_lookup = scan.set_index("mass_GeV")["eps2_scan"].to_dict()
    obs_records = []
    for idx, row in df.iterrows():
        if "+" not in str(row["dataset_set"]):
            continue
        mass = float(row["mass_GeV"])
        if mass not in scan_lookup:
            continue
        old = float(row["eps2_obs"])
        scan_val = float(scan_lookup[mass])
        if old > 0.0 and scan_val > 0.0 and abs(np.log(scan_val / old)) > hard_observed_mismatch:
            new = min(old, scan_val)
            source = "lower of combined_combined.csv and combined_ul_bands_combined_all.csv"
        else:
            new = scan_val
            source = "combined_combined.csv"

        df.loc[idx, "eps2_obs"] = new
        if "ul_eps2_obs" in df.columns:
            df.loc[idx, "ul_eps2_obs"] = new
        df.loc[idx, "observed_source"] = source
        if np.isfinite(old) and old > 0:
            rel = scan_val / old
        else:
            rel = np.nan
        if np.isfinite(rel) and abs(np.log(rel)) > np.log(1.01):
            obs_records.append(
                {
                    "mass_GeV": mass,
                    "mass_MeV": mass * 1000.0,
                    "dataset_set": row["dataset_set"],
                    "band_eps2_obs_raw": old,
                    "scan_eps2_obs": scan_val,
                    "eps2_obs_used": new,
                    "scan_over_band": rel,
                    "observed_source_used": source,
                }
            )

    spike_records = []
    for idx, ratios in detect_coherent_band_spikes(df):
        dataset_set = df.loc[idx, "dataset_set"]
        mass = float(df.loc[idx, "mass_GeV"])
        group = df[df["dataset_set"] == dataset_set].sort_values("mass_GeV")
        pos_arr = np.flatnonzero(group.index.to_numpy() == idx)
        if not pos_arr.size:
            continue
        pos = int(pos_arr[0])
        if pos == 0 or pos == len(group) - 1:
            continue
        prev_row = group.iloc[pos - 1]
        next_row = group.iloc[pos + 1]
        for col in BAND_COLS:
            old = float(df.loc[idx, col])
            new = float(np.exp(0.5 * (np.log(prev_row[col]) + np.log(next_row[col]))))
            df.loc[idx, col] = new
            alias = BAND_ALIAS.get(col)
            if alias in df.columns:
                df.loc[idx, alias] = new
            spike_records.append(
                {
                    "mass_GeV": mass,
                    "mass_MeV": mass * 1000.0,
                    "dataset_set": dataset_set,
                    "column": col,
                    "raw_value": old,
                    "corrected_value": new,
                    "raw_over_log_linear_neighbors": old / new,
                    "detected_ratio": ratios[col],
                    "correction": "log_linear_neighbors_same_dataset_set",
                }
            )
        df.loc[idx, "expected_band_source"] = "log_linear_neighbors_same_dataset_set"

    obs_df = pd.DataFrame(obs_records)
    spike_df = pd.DataFrame(spike_records)
    return df.sort_values("mass_GeV"), obs_df, spike_df


def load_individual_observed() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    single_fallback = []
    for source in (RUN90_V2 / "combined_single.csv", RUN90_V1 / "combined_single.csv"):
        if source.exists():
            tmp = pd.read_csv(source)
            tmp = tmp.rename(columns={"eps2_up": "eps2_obs"})
            single_fallback.append(tmp[["dataset", "mass_GeV", "eps2_obs"]])
    fallback = pd.concat(single_fallback, ignore_index=True).drop_duplicates(["dataset", "mass_GeV"])

    for dataset in ("2015", "2016", "2021"):
        path = RUN90_V1 / f"combined_ul_bands_{dataset}.csv"
        if path.exists():
            df = pd.read_csv(path)[["dataset", "mass_GeV", "eps2_obs"]].copy()
        else:
            df = fallback[fallback["dataset"].astype(str) == dataset].copy()
        missing = fallback[
            (fallback["dataset"].astype(str) == dataset)
            & (~fallback["mass_GeV"].isin(df["mass_GeV"]))
        ].copy()
        if not missing.empty:
            df = pd.concat([df, missing], ignore_index=True)
        df["mass_MeV"] = 1000.0 * df["mass_GeV"]
        df["epsilon_obs"] = np.sqrt(df["eps2_obs"].clip(lower=0.0))
        out[dataset] = df.sort_values("mass_GeV").drop_duplicates("mass_GeV")
    return out


def load_individual_bands() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for dataset in ("2015", "2016", "2021"):
        path = RUN90_V1 / f"combined_ul_bands_{dataset}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).copy()
        df["dataset"] = df["dataset"].astype(str)
        df["mass_MeV"] = 1000.0 * df["mass_GeV"]
        out[dataset] = df.sort_values("mass_GeV").drop_duplicates("mass_GeV")
    return out


def best_individual_ratio(combined: pd.DataFrame, individuals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lookups = {
        dataset: df.set_index(df["mass_GeV"].round(6))["eps2_obs"].to_dict()
        for dataset, df in individuals.items()
    }
    rows = []
    for _, row in combined.iterrows():
        active = parse_dataset_set(row["dataset_set"])
        if len(active) < 2:
            continue
        mass_key = round(float(row["mass_GeV"]), 6)
        candidates = []
        for dataset in active:
            val = lookups.get(dataset, {}).get(mass_key)
            if val is not None and np.isfinite(val) and val > 0.0:
                candidates.append((dataset, float(val)))
        if not candidates:
            continue
        best_dataset, best_eps2 = min(candidates, key=lambda item: item[1])
        comb_eps2 = float(row["eps2_obs"])
        rows.append(
            {
                "mass_GeV": float(row["mass_GeV"]),
                "mass_MeV": float(row["mass_MeV"]),
                "dataset_set": row["dataset_set"],
                "combined_eps2_obs": comb_eps2,
                "combined_epsilon_obs": np.sqrt(comb_eps2),
                "best_individual_dataset": best_dataset,
                "best_individual_eps2_obs": best_eps2,
                "best_individual_epsilon_obs": np.sqrt(best_eps2),
                "combined_over_best_individual_eps2": comb_eps2 / best_eps2,
                "combined_over_best_individual_epsilon": np.sqrt(comb_eps2 / best_eps2),
            }
        )
    return pd.DataFrame(rows).sort_values("mass_GeV")


def density_projection(combined: pd.DataFrame) -> pd.DataFrame:
    factors = {"2015": 1.0, "2016": 10.0, "2021": 100.0}
    rows = []
    for _, row in combined.iterrows():
        meta = ast.literal_eval(str(row["meta"]))
        current = 0.0
        projected = 0.0
        for item in meta:
            key = str(item["key"])
            dens = float(item["dens"])
            current += dens
            projected += dens * factors.get(key, 1.0)
        scale = np.sqrt(current / projected) if projected > 0.0 else np.nan
        rec = {
            "mass_GeV": float(row["mass_GeV"]),
            "mass_MeV": float(row["mass_MeV"]),
            "dataset_set": row["dataset_set"],
            "density_current": current,
            "density_projected_full": projected,
            "full_projection_scale": scale,
        }
        for col in ["eps2_obs", *BAND_COLS]:
            rec[col] = float(row[col])
            rec[f"projected_full_{col}"] = float(row[col]) * scale
        rec["epsilon_obs"] = np.sqrt(rec["eps2_obs"])
        rec["projected_full_epsilon_obs"] = np.sqrt(rec["projected_full_eps2_obs"])
        rec["projected_full_epsilon_med"] = np.sqrt(rec["projected_full_eps2_med"])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("mass_GeV")


def load_babar_90() -> pd.DataFrame:
    babar = pd.read_csv(
        BABAR_SOURCE,
        comment="#",
        sep=r"\s+",
        names=["mass_GeV", "epsilon_90"],
    )
    babar = babar[np.isfinite(babar["mass_GeV"]) & np.isfinite(babar["epsilon_90"])].copy()
    babar = babar[(babar["epsilon_90"] > 0.0) & (babar["epsilon_90"] < 1.0)].copy()
    babar["mass_MeV"] = 1000.0 * babar["mass_GeV"]
    babar["eps2_90"] = babar["epsilon_90"] ** 2
    return babar.sort_values("mass_MeV").drop_duplicates("mass_MeV")


def load_95_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    bands = pd.read_csv(CORRECTED95 / "combined_ul_bands_combined_all.csv").copy()
    bands["mass_MeV"] = 1000.0 * bands["mass_GeV"]

    smooth = CORRECTED95 / "full_observed_combined_plots_v3_smooth_observed" / "observed_combined_smooth_source.csv"
    merged = CORRECTED95 / "full_observed_combined_plots" / "full_combined_observed_merged.csv"
    if smooth.exists():
        observed = pd.read_csv(smooth).rename(columns={"eps2_obs": "eps2_obs_95"})
    else:
        observed = pd.read_csv(merged).rename(columns={"eps2_combined_observed": "eps2_obs_95"})
    if "mass_MeV" not in observed.columns:
        observed["mass_MeV"] = 1000.0 * observed["mass_GeV"]
    observed = observed[["mass_GeV", "mass_MeV", "eps2_obs_95"]].copy()
    observed["epsilon_obs_95"] = np.sqrt(observed["eps2_obs_95"].clip(lower=0.0))
    return bands.sort_values("mass_GeV"), observed.sort_values("mass_GeV")


def plot_combined_bands(df: pd.DataFrame, stem: str, *, epsilon: bool = False) -> None:
    x = df["mass_MeV"].to_numpy(float)
    if epsilon:
        lo2, lo1, med, hi1, hi2, obs = [np.sqrt(df[c].to_numpy(float)) for c in [*BAND_COLS, "eps2_obs"]]
        ylabel = r"90% CL upper limit on $\epsilon$"
        title = r"90% CL expected bands and observed simultaneous-combination limit"
    else:
        lo2, lo1, med, hi1, hi2, obs = [df[c].to_numpy(float) for c in [*BAND_COLS, "eps2_obs"]]
        ylabel = r"90% CL upper limit on $\epsilon^2$"
        title = r"90% CL expected bands and observed simultaneous-combination limit"

    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    fill_between_segments(ax, x, lo2, hi2, color="#F5C542", alpha=0.24, label=r"$\pm2\sigma$ expected", zorder=1)
    fill_between_segments(ax, x, lo1, hi1, color="#3CB44B", alpha=0.35, label=r"$\pm1\sigma$ expected", zorder=2)
    plot_segments(ax, x, med, color="#111111", linewidth=2.2, linestyle="--", label="Expected median", zorder=3)
    plot_segments(ax, x, obs, color="#000000", linewidth=3.0, label="Observed combined", zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    style_axis(ax)
    save_figure(fig, stem)


def plot_observed_overlay(combined: pd.DataFrame, individuals: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for dataset, df in individuals.items():
        ax.plot(
            df["mass_MeV"],
            df["eps2_obs"],
            color=DATASET_COLORS[dataset],
            linewidth=1.75,
            alpha=0.82,
            label=f"{dataset} observed",
        )
    ax.plot(
        combined["mass_MeV"],
        combined["eps2_obs"],
        color=COMBINED_COLOR,
        linewidth=3.0,
        label="Combined observed 90% CL",
    )
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("90% CL observed limits: individual datasets and simultaneous combination")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    save_figure(fig, "observed_90cls_eps2_overlay_individual_and_combined")


def plot_individual_bands(individual_bands: dict[str, pd.DataFrame]) -> None:
    labels = {"2015": "2015", "2016": "2016 10%", "2021": "2021 1%"}
    for dataset, df in individual_bands.items():
        x = df["mass_MeV"].to_numpy(float)
        lo2, lo1, med, hi1, hi2, obs = [df[c].to_numpy(float) for c in [*BAND_COLS, "eps2_obs"]]

        fig, ax = plt.subplots(figsize=(10.4, 5.9))
        fill_between_segments(
            ax,
            x,
            lo2,
            hi2,
            color="#F5C542",
            alpha=0.24,
            label=r"$\pm2\sigma$ expected",
            zorder=1,
        )
        fill_between_segments(
            ax,
            x,
            lo1,
            hi1,
            color="#3CB44B",
            alpha=0.35,
            label=r"$\pm1\sigma$ expected",
            zorder=2,
        )
        plot_segments(ax, x, med, color="#111111", linewidth=2.0, linestyle="--", label="Expected median", zorder=3)
        plot_segments(
            ax,
            x,
            obs,
            color=DATASET_COLORS[dataset],
            linewidth=2.6,
            label=f"{labels[dataset]} observed",
            zorder=4,
        )
        ax.set_yscale("log")
        ax.set_xlim(max(18, float(np.nanmin(x)) - 2.0), min(252, float(np.nanmax(x)) + 2.0))
        ax.set_xlabel(r"Mass hypothesis (MeV)")
        ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
        ax.set_title(f"{labels[dataset]} individual 90% CL observed and expected limit")
        ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
        ax.yaxis.set_minor_formatter(NullFormatter())
        style_axis(ax)
        save_figure(fig, f"{dataset}_90cls_eps2_coupling_bands_observed_expected")


def plot_best_ratio(ratio: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.45))
    ax.axhspan(0.0, 1.0, color="#55A868", alpha=0.11, label="Combination stronger than best individual")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.35)
    ax.plot(
        ratio["mass_MeV"],
        ratio["combined_over_best_individual_epsilon"],
        color="#000000",
        linewidth=2.5,
        marker="o",
        markersize=3.6,
        label="Combined / best individual observed",
    )
    for dataset, color in DATASET_COLORS.items():
        subset = ratio[ratio["best_individual_dataset"] == dataset]
        if subset.empty:
            continue
        ax.scatter(
            subset["mass_MeV"],
            subset["combined_over_best_individual_epsilon"],
            color=color,
            s=20,
            zorder=4,
            label=f"Best individual: {dataset}",
        )
    ax.set_xlim(28, 212)
    ymin = max(0.35, float(ratio["combined_over_best_individual_epsilon"].min()) * 0.92)
    ymax = min(1.45, float(ratio["combined_over_best_individual_epsilon"].max()) * 1.08)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(r"Ratio of observed $\epsilon$ limits")
    ax.set_title("90% CL combined observed limit compared with the best individual dataset")
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#c9c9c9", ncol=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)
    save_figure(fig, "combined_vs_best_individual_observed_epsilon_ratio_90cls")


def plot_projection(proj: pd.DataFrame) -> None:
    x = proj["mass_MeV"].to_numpy(float)
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    ax.plot(x, proj["eps2_obs"], color="#111111", linewidth=2.2, label="Current observed combined 90% CL")
    ax.plot(x, proj["eps2_med"], color="#111111", linewidth=1.9, linestyle="--", label="Current expected median")
    ax.plot(
        x,
        proj["projected_full_eps2_obs"],
        color=PROJECTION_COLOR,
        linewidth=2.8,
        label="Projected full-unblinded observed-equivalent reach",
    )
    ax.plot(
        x,
        proj["projected_full_eps2_med"],
        color=PROJECTION_COLOR,
        linewidth=2.2,
        linestyle="--",
        label="Projected full-unblinded expected median",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("Projected full-unblinded HPS reach from the 90% CL combined result")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)

    rax.plot(x, proj["full_projection_scale"], color=PROJECTION_COLOR, linewidth=2.0)
    rax.set_ylabel("Scale")
    rax.set_xlabel(r"Mass hypothesis (MeV)")
    rax.set_xlim(18, 252)
    rax.set_ylim(0.0, 1.05)
    rax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(rax)
    save_figure(fig, "projected_full_unblinded_reach_90cls_eps2")


def babar_comparison_table(
    babar: pd.DataFrame,
    projection: pd.DataFrame,
) -> pd.DataFrame:
    focus = (35.0, 190.0)
    proj = projection[(projection["mass_MeV"] >= focus[0]) & (projection["mass_MeV"] <= focus[1])].copy()
    proj["babar_eps2_90_interp"] = log_interp(
        proj["mass_MeV"].to_numpy(float),
        babar["mass_MeV"].to_numpy(float),
        babar["eps2_90"].to_numpy(float),
    )
    proj["projected_full_hps_over_babar_90"] = proj["projected_full_eps2_obs"] / proj["babar_eps2_90_interp"]
    proj["babar_over_projected_full_hps_90"] = proj["babar_eps2_90_interp"] / proj["projected_full_eps2_obs"]
    return proj


def plot_babar90(babar: pd.DataFrame, proj_comp: pd.DataFrame) -> None:
    focus = (35.0, 190.0)
    babar_focus = babar[(babar["mass_MeV"] >= focus[0]) & (babar["mass_MeV"] <= focus[1])]

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.15, 1.0], "hspace": 0.05},
    )
    ax.plot(
        babar_focus["mass_MeV"],
        babar_focus["eps2_90"],
        color=BABAR_COLOR,
        linewidth=2.4,
        label=r"BaBar observed $\epsilon^2$ limit (90% CL)",
    )
    ax.plot(
        proj_comp["mass_MeV"],
        proj_comp["projected_full_eps2_obs"],
        color=PROJECTION_COLOR,
        linewidth=3.0,
        label=r"Projected HPS 2015+2016+2021 100% reach (90% CL)",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("90% CL projected full-data HPS reach compared with BaBar")
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#c9c9c9", handlelength=2.4)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)

    rax.plot(
        proj_comp["mass_MeV"],
        proj_comp["projected_full_hps_over_babar_90"],
        color=PROJECTION_COLOR,
        linewidth=2.2,
        label="Projected HPS / BaBar",
    )
    rax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.3)
    rax.set_yscale("log")
    rax.set_ylabel("HPS / BaBar")
    rax.set_xlabel(r"$m_{A'}$ (MeV)")
    rax.set_xlim(*focus)
    rax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(rax)
    save_figure(fig, "babar90_vs_projected_full_data_100pct")

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.plot(
        babar_focus["mass_MeV"],
        babar_focus["eps2_90"],
        color=BABAR_COLOR,
        linewidth=2.6,
        label=r"BaBar observed $\epsilon^2$ limit (90% CL)",
    )
    ax.plot(
        proj_comp["mass_MeV"],
        proj_comp["projected_full_eps2_obs"],
        color=PROJECTION_COLOR,
        linewidth=3.0,
        label=r"Projected HPS 2015+2016+2021 100% reach (90% CL)",
    )
    ax.set_yscale("log")
    ax.set_xlim(*focus)
    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("90% CL projected full-data HPS reach vs BaBar")
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#c9c9c9", handlelength=2.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    save_figure(fig, "babar90_vs_projected_full_data_100pct_overlay_only")


def plot_90_vs_95(combined90: pd.DataFrame, bands95: pd.DataFrame, obs95: pd.DataFrame) -> pd.DataFrame:
    obs = combined90[["mass_GeV", "mass_MeV", "eps2_obs", "eps2_med"]].copy()
    obs = obs.rename(columns={"eps2_obs": "eps2_obs_90", "eps2_med": "eps2_med_90"})
    obs["eps2_obs_95"] = log_interp(
        obs["mass_MeV"].to_numpy(float),
        obs95["mass_MeV"].to_numpy(float),
        obs95["eps2_obs_95"].to_numpy(float),
    )
    obs["eps2_med_95_interp"] = log_interp(
        obs["mass_MeV"].to_numpy(float),
        bands95["mass_MeV"].to_numpy(float),
        bands95["eps2_med"].to_numpy(float),
    )
    obs["obs_95_over_90_eps2"] = obs["eps2_obs_95"] / obs["eps2_obs_90"]
    obs["med_95_over_90_eps2"] = obs["eps2_med_95_interp"] / obs["eps2_med_90"]

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    ax.plot(combined90["mass_MeV"], combined90["eps2_obs"], color="#111111", linewidth=2.4, label="90% observed")
    ax.plot(combined90["mass_MeV"], combined90["eps2_med"], color="#111111", linewidth=2.0, linestyle="--", label="90% expected median")
    ax.plot(obs95["mass_MeV"], obs95["eps2_obs_95"], color="#C44E52", linewidth=2.2, label="95% observed")
    plot_segments(
        ax,
        bands95["mass_MeV"].to_numpy(float),
        bands95["eps2_med"].to_numpy(float),
        color="#C44E52",
        linewidth=1.9,
        linestyle="--",
        label="95% expected median",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"Upper limit on $\epsilon^2$")
    ax.set_title("Combined observed and expected limits: 90% CL vs 95% CL")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)

    rax.plot(obs["mass_MeV"], obs["obs_95_over_90_eps2"], color="#C44E52", linewidth=2.0, label="Observed")
    rax.plot(obs["mass_MeV"], obs["med_95_over_90_eps2"], color="#111111", linewidth=1.8, linestyle="--", label="Expected median")
    rax.axhline(CL_RATIO_95_OVER_90_EPS2, color="#555555", linestyle=":", linewidth=1.5, label="Gaussian z-ratio")
    rax.set_ylabel("95 / 90")
    rax.set_xlabel(r"Mass hypothesis (MeV)")
    rax.set_xlim(18, 252)
    rax.xaxis.set_minor_locator(AutoMinorLocator(5))
    rax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    style_axis(rax)
    save_figure(fig, "combined_90_vs_95cls_observed_expected_eps2")
    return obs


def plot_spike_diagnostics(raw: pd.DataFrame, corrected: pd.DataFrame, spike_df: pd.DataFrame) -> None:
    masses = sorted(spike_df["mass_GeV"].unique()) if not spike_df.empty else []
    if not masses:
        return
    fig, axes = plt.subplots(1, len(masses), figsize=(6.1 * len(masses), 4.7), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, mass in zip(axes, masses):
        dataset_set = spike_df.loc[spike_df["mass_GeV"] == mass, "dataset_set"].iloc[0]
        lo = mass - 0.006
        hi = mass + 0.006
        raw_window = raw[
            (raw["dataset_set"] == dataset_set)
            & (raw["mass_GeV"] >= lo)
            & (raw["mass_GeV"] <= hi)
        ].copy()
        corr_window = corrected[
            (corrected["dataset_set"] == dataset_set)
            & (corrected["mass_GeV"] >= lo)
            & (corrected["mass_GeV"] <= hi)
        ].copy()
        ax.fill_between(
            raw_window["mass_MeV"],
            raw_window["eps2_lo2"],
            raw_window["eps2_hi2"],
            color="#F5C542",
            alpha=0.18,
            label="Raw 2 sigma band",
        )
        ax.plot(raw_window["mass_MeV"], raw_window["eps2_med"], color="#D55E00", linewidth=2.2, label="Raw median")
        ax.plot(corr_window["mass_MeV"], corr_window["eps2_med"], color="#111111", linewidth=2.2, linestyle="--", label="Corrected median")
        ax.scatter([mass * 1000.0], [raw_window.loc[np.isclose(raw_window["mass_GeV"], mass), "eps2_med"].iloc[0]], color="#D55E00", s=42, zorder=4)
        ax.scatter([mass * 1000.0], [corr_window.loc[np.isclose(corr_window["mass_GeV"], mass), "eps2_med"].iloc[0]], color="#111111", s=42, zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel(r"Mass hypothesis (MeV)")
        ax.set_title(f"{mass * 1000.0:.0f} MeV expected-band repair")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        style_axis(ax)
    axes[0].set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    save_figure(fig, "combined_expected_band_spike_diagnostics_90cls")


def plot_tail_areas(combined: pd.DataFrame, spike_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    masses_gev = combined["mass_GeV"].to_numpy(dtype=float)
    sigma_col = "sigma_mass_res_min_GeV" if "sigma_mass_res_min_GeV" in combined.columns else "sigma_mass_res_GeV"
    sigma_vals = combined[sigma_col].to_numpy(dtype=float) if sigma_col in combined.columns else None
    indep_width_sigma = (
        float(combined["bands_train_exclude_nsigma"].dropna().iloc[0])
        if "bands_train_exclude_nsigma" in combined.columns and combined["bands_train_exclude_nsigma"].notna().any()
        else 1.96
    )
    neff = effective_trials_from_spacing(
        masses_gev,
        sigma_vals,
        indep_width_sigma=indep_width_sigma,
    )
    ref_levels = []
    for z in SIGMA_REFERENCE_LEVELS:
        p_local = one_sided_gaussian_p(z)
        p_global_threshold = local_p_from_global_sidak(p_local, neff)
        ref_levels.extend([p_local, p_global_threshold])

    p_values = combined[["p_strong", "p_weak", "p_two"]].to_numpy(dtype=float)
    positive_values = p_values[np.isfinite(p_values) & (p_values > 0.0)]
    min_data = float(np.nanmin(positive_values)) if positive_values.size else 1.0
    min_ref = float(np.nanmin(np.asarray(ref_levels, dtype=float)))
    p_floor = max(1.0e-8, min(0.8 * min_data, 0.8 * min_ref, 0.2))
    for column, color, label, linewidth in [
        ("p_strong", "#4C72B0", r"$p_{\rm strong}$", 1.75),
        ("p_weak", "#DD8452", r"$p_{\rm weak}$", 1.75),
        ("p_two", "#111111", r"$p_{\rm two}$", 2.0),
    ]:
        values = np.asarray(combined[column], dtype=float)
        ax.plot(
            combined["mass_MeV"],
            np.clip(values, p_floor, 1.0),
            color=color,
            linewidth=linewidth,
            label=label,
        )
    for mass in sorted(spike_df["mass_MeV"].unique()) if not spike_df.empty else []:
        ax.axvline(mass, color="#C44E52", linestyle=":", linewidth=1.4, alpha=0.8)
    ref_handles = []
    ref_labels = []
    for z in SIGMA_REFERENCE_LEVELS:
        p_local = one_sided_gaussian_p(z)
        p_global_threshold = local_p_from_global_sidak(p_local, neff)
        color = SIGMA_REFERENCE_COLORS.get(float(z), "0.25")
        local_visible = p_floor <= p_local <= 1.1
        global_visible = p_floor <= p_global_threshold <= 1.1
        if local_visible:
            local_line = ax.axhline(p_local, color=color, linestyle=":", linewidth=1.35, alpha=0.95)
            ax.text(
                249.0,
                p_local,
                rf"local {int(z)}$\sigma$",
                va="bottom",
                ha="right",
                fontsize=9.0,
                color=color,
            )
            ref_handles.append(local_line)
            ref_labels.append(rf"local {int(z)}$\sigma$")
        if global_visible:
            global_line = ax.axhline(p_global_threshold, color=color, linestyle="--", linewidth=1.35, alpha=0.95)
            ax.text(
                20.0,
                p_global_threshold,
                rf"global {int(z)}$\sigma$ threshold",
                va="bottom",
                ha="left",
                fontsize=9.0,
                color=color,
            )
            ref_handles.append(global_line)
            ref_labels.append(rf"global {int(z)}$\sigma$")
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_ylim(p_floor, 1.1)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel("p-value")
    ax.set_title("90% CL toy-band tail-area diagnostics with significance references")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + ref_handles,
        labels + ref_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        edgecolor="#c9c9c9",
        ncol=2,
        columnspacing=1.0,
        handlelength=2.2,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)
    save_figure(fig, "combined_toy_limit_tail_areas_90cls")


def write_summary(
    corrected: pd.DataFrame,
    obs_replacements: pd.DataFrame,
    spike_df: pd.DataFrame,
    ratio: pd.DataFrame,
    projection: pd.DataFrame,
    proj_babar: pd.DataFrame,
    comp_95: pd.DataFrame,
) -> None:
    better = ratio["combined_over_best_individual_epsilon"] < 1.0
    proj_ratio = proj_babar["projected_full_hps_over_babar_90"].replace([np.inf, -np.inf], np.nan).dropna()
    lines = [
        "# 90% CL Note-Comparison Plot Summary",
        "",
        f"90% v2 source directory: `{RUN90_V2}`",
        f"Output directory: `{OUTDIR}`",
        "",
        "## Plot-level corrections",
        f"- Observed combined curve: uses `{RUN90_V2 / 'combined_combined.csv'}` for masses with a simultaneous combined scan.",
        "- Hard observed scan-vs-band disagreements use the lower of the scan and band values, matching the convention used in the current 95% note products.",
        f"- Single-dataset endpoint rows keep `{RUN90_V2 / 'combined_ul_bands_combined_all.csv'}` values.",
        f"- Coherent expected-band spike repairs: `{len(spike_df['mass_GeV'].unique()) if not spike_df.empty else 0}` mass points.",
    ]
    if not spike_df.empty:
        masses = ", ".join(f"{m:.0f} MeV" for m in sorted(spike_df["mass_MeV"].unique()))
        lines.append(f"- Repaired expected-band masses: {masses}.")
    lines += [
        "",
        "## Quick numbers",
        f"- Corrected combined grid: {len(corrected)} points from {corrected['mass_MeV'].min():.0f} to {corrected['mass_MeV'].max():.0f} MeV.",
        f"- Combined/best individual epsilon ratio median: {ratio['combined_over_best_individual_epsilon'].median():.3f}.",
        f"- Fraction of combined points stronger than best individual: {better.mean():.3f}.",
        f"- Full-unblinded projection scale range: {projection['full_projection_scale'].min():.3f} to {projection['full_projection_scale'].max():.3f}.",
        f"- Projected HPS/BaBar 90% median ratio over 35-190 MeV: {proj_ratio.median():.3f}.",
        f"- 95/90 observed epsilon^2 median ratio: {comp_95['obs_95_over_90_eps2'].median():.3f}.",
        f"- 95/90 expected-median epsilon^2 median ratio: {comp_95['med_95_over_90_eps2'].median():.3f}.",
        "",
        "## Main files",
        "- `2015_90cls_eps2_coupling_bands_observed_expected.png/pdf`",
        "- `2016_90cls_eps2_coupling_bands_observed_expected.png/pdf`",
        "- `2021_90cls_eps2_coupling_bands_observed_expected.png/pdf`",
        "- `combined_90cls_bands_corrected_eps2.png/pdf`",
        "- `combined_90cls_bands_corrected_epsilon.png/pdf`",
        "- `observed_90cls_eps2_overlay_individual_and_combined.png/pdf`",
        "- `combined_vs_best_individual_observed_epsilon_ratio_90cls.png/pdf`",
        "- `combined_90_vs_95cls_observed_expected_eps2.png/pdf`",
        "- `projected_full_unblinded_reach_90cls_eps2.png/pdf`",
        "- `babar90_vs_projected_full_data_100pct.png/pdf`",
        "- `babar90_vs_projected_full_data_100pct_overlay_only.png/pdf`",
        "- `combined_expected_band_spike_diagnostics_90cls.png/pdf`",
        "- `combined_toy_limit_tail_areas_90cls.png/pdf`",
    ]
    (OUTDIR / "summary_90cls_note_comparison_plots.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dir(OUTDIR)
    setup_style()

    corrected, obs_replacements, spike_df = corrected_90_bands()
    raw = pd.read_csv(RUN90_V2 / "combined_ul_bands_combined_all.csv")
    raw["mass_MeV"] = 1000.0 * raw["mass_GeV"]

    individuals = load_individual_observed()
    individual_bands = load_individual_bands()
    ratio = best_individual_ratio(corrected, individuals)
    projection = density_projection(corrected)
    babar = load_babar_90()
    proj_babar = babar_comparison_table(babar, projection)
    bands95, obs95 = load_95_sources()
    comp_95 = plot_90_vs_95(corrected, bands95, obs95)

    corrected.to_csv(OUTDIR / "combined_ul_bands_combined_all_90cls_corrected_for_plotting.csv", index=False)
    obs_replacements.to_csv(OUTDIR / "combined_observed_scan_replacements_90cls.csv", index=False)
    spike_df.to_csv(OUTDIR / "combined_expected_band_spike_repairs_90cls.csv", index=False)
    ratio.to_csv(OUTDIR / "combined_vs_best_individual_observed_epsilon_ratio_90cls.csv", index=False)
    projection.to_csv(OUTDIR / "projected_full_unblinded_reach_90cls.csv", index=False)
    babar.to_csv(OUTDIR / "babar_Lees2014xha_eps2_90.csv", index=False)
    proj_babar.to_csv(OUTDIR / "babar90_projected_full_data_100pct_comparison.csv", index=False)
    comp_95.to_csv(OUTDIR / "combined_90_vs_95cls_observed_expected_comparison.csv", index=False)
    for dataset, df in individual_bands.items():
        df.to_csv(OUTDIR / f"{dataset}_90cls_individual_bands_for_plotting.csv", index=False)

    plot_combined_bands(corrected, "combined_90cls_bands_corrected_eps2", epsilon=False)
    plot_combined_bands(corrected, "combined_90cls_bands_corrected_epsilon", epsilon=True)
    plot_individual_bands(individual_bands)
    plot_observed_overlay(corrected, individuals)
    plot_best_ratio(ratio)
    plot_projection(projection)
    plot_babar90(babar, proj_babar)
    plot_spike_diagnostics(raw, corrected, spike_df)
    plot_tail_areas(corrected, spike_df)
    write_summary(corrected, obs_replacements, spike_df, ratio, projection, proj_babar, comp_95)

    print(f"Wrote {len(list(OUTDIR.glob('*')))} outputs to {OUTDIR}")
    if not spike_df.empty:
        masses = ", ".join(f"{m:.0f} MeV" for m in sorted(spike_df["mass_MeV"].unique()))
        print(f"Repaired expected-band spikes at: {masses}")
    print(f"Median combined/best individual epsilon ratio: {ratio['combined_over_best_individual_epsilon'].median():.3f}")
    print(f"Median projected HPS/BaBar 90% ratio: {proj_babar['projected_full_hps_over_babar_90'].median():.3f}")


if __name__ == "__main__":
    main()
