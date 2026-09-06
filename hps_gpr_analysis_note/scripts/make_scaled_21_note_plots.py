from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


REPO_NOTE_DIR = Path(__file__).resolve().parents[1]
OUTDIR = REPO_NOTE_DIR / "final_limit_projection_figs" / "90cls" / "scaled_21"

SOURCE_DIR = Path("/Users/emryspeets/Desktop/gp_mods/combined_15_16_10pct_21_1pct")
BAND_SOURCE = SOURCE_DIR / "scaled_21_dimuon_bandfixed" / "combined_ul_bands_combined_all_scaled_21_bandfixed_dimuon_for_plotting.csv"
COMPARISON_SOURCE = SOURCE_DIR / "scaled_21_dimuon_bandfixed" / "scaled_21_bandfixed_dimuon_vs_unscaled_common_mass_comparison.csv"

M_MU_GEV = 0.1056583745
M_DIMUON_GEV = 2.0 * M_MU_GEV


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 9.6,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#d7d7d7", linewidth=0.8, alpha=0.85)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.55, alpha=0.75)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.0, length=6)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_segments(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_gap: float = 1.51,
    positive: bool = True,
    **kwargs,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if positive:
        valid &= y > 0.0
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


def configure_eps2_axis(ax: plt.Axes, title: str) -> None:
    ax.axvline(1000.0 * M_DIMUON_GEV, color="#555555", linestyle=":", linewidth=1.3, label=r"$2m_\mu$")
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title(title)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)


def make_overlay_ratio() -> None:
    cmp = pd.read_csv(COMPARISON_SOURCE).sort_values("mass_GeV")
    x = cmp["mass_MeV"].to_numpy(float)
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.05},
    )
    plot_segments(ax, x, cmp["eps2_obs_v2_dimuon"], color="#4C72B0", linewidth=2.1, label="unscaled observed")
    plot_segments(ax, x, cmp["eps2_obs_scaled_21_dimuon"], color="#C44E52", linewidth=2.5, label="scaled observed")
    plot_segments(ax, x, cmp["eps2_med_v2_dimuon"], color="#4C72B0", linewidth=1.8, linestyle="--", label="unscaled median")
    plot_segments(ax, x, cmp["eps2_med_scaled_21_dimuon"], color="#C44E52", linewidth=2.0, linestyle="--", label="scaled median")
    ax.axvline(1000.0 * M_DIMUON_GEV, color="#555555", linestyle=":", linewidth=1.3, label=r"$2m_\mu$")
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("Combined upper limit comparison: scaled vs unscaled")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")

    plot_segments(rax, x, cmp["eps2_obs_scaled_over_unscaled"], color="#C44E52", linewidth=2.2, label="Observed", positive=False)
    plot_segments(rax, x, cmp["eps2_med_scaled_over_unscaled"], color="#111111", linewidth=1.8, linestyle="--", label="Median", positive=False)
    rax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.2)
    rax.axvline(1000.0 * M_DIMUON_GEV, color="#555555", linestyle=":", linewidth=1.2)
    rax.set_xlabel(r"Mass hypothesis (MeV)")
    rax.set_ylabel("scaled / unscaled")
    rax.xaxis.set_minor_locator(AutoMinorLocator(5))
    rax.yaxis.set_minor_locator(AutoMinorLocator(4))
    style_axis(rax)
    rax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#c9c9c9", ncols=2)
    save_figure(fig, "scaled_21_bandfixed_dimuon_vs_unscaled_overlay_ratio")


def make_scaled_band_plot() -> None:
    bands = pd.read_csv(BAND_SOURCE).sort_values("mass_GeV")
    x = bands["mass_MeV"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    fill_between_segments(ax, x, bands["eps2_lo2"], bands["eps2_hi2"], color="#F5C542", alpha=0.24, label=r"$\pm2\sigma$ expected")
    fill_between_segments(ax, x, bands["eps2_lo1"], bands["eps2_hi1"], color="#3CB44B", alpha=0.35, label=r"$\pm1\sigma$ expected")
    plot_segments(ax, x, bands["eps2_obs"], color="#000000", linewidth=2.8, label="scaled observed")
    configure_eps2_axis(ax, "90% Combined upper limit with dimuon correction and 2021 scaled resolution")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    save_figure(fig, "scaled_21_bandfixed_dimuon_90cls_bands_eps2")


def main() -> None:
    setup_style()
    make_overlay_ratio()
    make_scaled_band_plot()
    print(f"Wrote scaled 2021 note plots to {OUTDIR}")


if __name__ == "__main__":
    main()
