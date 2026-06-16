#!/usr/bin/env python3
"""Build a slide-friendly composite of the three mass-resolution curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec


NOTE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = NOTE_DIR.parent
OUT_DIR = NOTE_DIR / "resolution_figs"
CFG_PATH = REPO_DIR / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"


COLORS = {
    "2015": "#2F6FBB",
    "2016": "#8A4FBF",
    "2021": "#1B8A5A",
    "tail": "#5B4A68",
}


def sigma_poly(m_gev: np.ndarray, coeffs: list[float]) -> np.ndarray:
    sigma = np.zeros_like(m_gev, dtype=float)
    for i, coeff in enumerate(coeffs):
        sigma += float(coeff) * m_gev**i
    return sigma


def sigma_poly_deriv(m_gev: float, coeffs: list[float]) -> float:
    return float(sum(i * float(c) * m_gev ** (i - 1) for i, c in enumerate(coeffs) if i > 0))


def sigma_2016(m_gev: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    coeffs = [float(c) for c in cfg["sigma_coeffs_2016"]]
    m0 = float(cfg["sigma_tail_m0_2016"])
    sigma = sigma_poly(m_gev, coeffs)
    tail_mask = m_gev > m0
    if np.any(tail_mask):
        sigma_m0 = float(sigma_poly(np.array([m0]), coeffs)[0])
        slope = sigma_poly_deriv(m0, coeffs)
        override = cfg.get("sigma_tail_slope_override_2016")
        if override is not None:
            slope = float(override)
        slope = max(float(slope), float(cfg.get("sigma_tail_slope_floor_2016", 0.0)))
        sigma[tail_mask] = sigma_m0 + slope * (m_gev[tail_mask] - m0)
    return sigma, tail_mask


def style_axis(ax: plt.Axes, *, title: str, title_color: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title, loc="left", fontsize=12.4, fontweight="bold", color=title_color, pad=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$m(e^+e^-)$ [MeV]", fontsize=11.4, labelpad=4)
    ax.set_ylabel(r"$\sigma_m$ [MeV]", fontsize=11.4, labelpad=4)
    ax.grid(True, which="major", color="#D9DEE2", linewidth=0.9)
    ax.grid(True, which="minor", color="#ECEFF1", linewidth=0.6, alpha=0.65)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=10.7, width=1.1, length=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#222222")


def annotate_curve(ax: plt.Axes, text: str, *, xy: tuple[float, float], color: str, ha: str = "left") -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        fontsize=9.1,
        color=color,
        ha=ha,
        va="center",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D2D6DB",
            "linewidth": 0.7,
            "alpha": 0.92,
        },
    )


def main() -> None:
    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.15,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(7.45, 5.05), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.03, 1.0], hspace=0.45, wspace=0.28)
    ax2016 = fig.add_subplot(gs[0, :])
    ax2015 = fig.add_subplot(gs[1, 0])
    ax2021 = fig.add_subplot(gs[1, 1])

    m2016 = np.linspace(0.035, 0.210, 450)
    s2016, tail_mask = sigma_2016(m2016, cfg)
    m2016_mev = 1e3 * m2016
    s2016_mev = 1e3 * s2016
    regular = ~tail_mask
    ax2016.plot(m2016_mev[regular], s2016_mev[regular], color=COLORS["2016"], lw=2.8)
    ax2016.plot(m2016_mev[tail_mask], s2016_mev[tail_mask], color=COLORS["tail"], lw=2.8, ls="--")
    style_axis(
        ax2016,
        title="2016 10%: smeared A-prime MC",
        title_color=COLORS["2016"],
        xlim=(35, 210),
        ylim=(1.0, 8.8),
    )
    annotate_curve(ax2016, "polynomial fit", xy=(48, 6.9), color=COLORS["2016"])
    annotate_curve(ax2016, "configured tail", xy=(166, 7.45), color=COLORS["tail"])

    m2015 = np.linspace(0.0, 0.100, 250)
    s2015 = sigma_poly(m2015, [float(c) for c in cfg["sigma_coeffs_2015"]])
    ax2015.plot(1e3 * m2015, 1e3 * s2015, color=COLORS["2015"], lw=2.8)
    style_axis(
        ax2015,
        title="2015: scaled A-prime MC",
        title_color=COLORS["2015"],
        xlim=(0, 100),
        ylim=(-0.1, 5.6),
    )
    annotate_curve(ax2015, "Moller anchored", xy=(9, 4.75), color=COLORS["2015"])

    m2021 = np.linspace(0.050, 0.190, 300)
    s2021 = sigma_poly(m2021, [float(c) for c in cfg["sigma_coeffs_2021"]])
    ax2021.plot(1e3 * m2021, 1e3 * s2021, color=COLORS["2021"], lw=2.8)
    style_axis(
        ax2021,
        title="2021 1%: target-constrained V0",
        title_color=COLORS["2021"],
        xlim=(50, 190),
        ylim=(1.45, 3.9),
    )
    annotate_curve(ax2021, "quadratic fit", xy=(58, 3.45), color=COLORS["2021"])

    fig.suptitle("HPS Mass resolutions", fontsize=13.4, fontweight="bold", y=0.985)
    fig.subplots_adjust(top=0.89, bottom=0.105, left=0.082, right=0.985)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "hps_mass_resolution_slide_summary.png"
    pdf_path = OUT_DIR / "hps_mass_resolution_slide_summary.pdf"
    fig.savefig(png_path, dpi=320)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(png_path)
    print(pdf_path)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.15), constrained_layout=False)
    panels = [
        (
            "2015",
            axes[0],
            np.linspace(0.020, 0.130, 360),
            [float(c) for c in cfg["sigma_coeffs_2015"]],
            None,
            (20, 130),
        ),
        (
            "2016",
            axes[1],
            np.linspace(0.035, 0.210, 420),
            [float(c) for c in cfg["sigma_coeffs_2016"]],
            "tail",
            (35, 210),
        ),
        (
            "2021",
            axes[2],
            np.linspace(0.035, 0.250, 430),
            [float(c) for c in cfg["sigma_coeffs_2021"]],
            None,
            (35, 250),
        ),
    ]
    titles = {
        "2015": "2015 scaled A-prime MC parameterization",
        "2016": "2016 smeared A-prime MC parameterization",
        "2021": "2021 target-constrained V0 parameterization",
    }
    for key, ax, m_vals, coeffs, special, xlim in panels:
        if special == "tail":
            sigma_vals, tail_mask = sigma_2016(m_vals, cfg)
            regular = ~tail_mask
            ax.plot(
                1e3 * m_vals[regular],
                1e3 * sigma_vals[regular],
                color=COLORS[key],
                lw=2.4,
                label="polynomial fit",
            )
            ax.plot(
                1e3 * m_vals[tail_mask],
                1e3 * sigma_vals[tail_mask],
                color=COLORS["tail"],
                lw=2.4,
                ls="--",
                label="configured tail",
            )
            ax.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=8.4)
        else:
            sigma_vals = sigma_poly(m_vals, coeffs)
            ax.plot(1e3 * m_vals, 1e3 * sigma_vals, color=COLORS[key], lw=2.4)
        sigma_mev = 1e3 * sigma_vals
        ymin = 0.0 if key == "2015" else max(0.0, float(np.nanmin(sigma_mev)) * 0.82)
        ymax = max(float(np.nanmax(sigma_mev)) * 1.12, 1.0)
        style_axis(
            ax,
            title=titles[key],
            title_color=COLORS[key],
            xlim=xlim,
            ylim=(ymin, ymax),
        )

    fig.suptitle(
        "Mass-resolution parameterizations over current GPR scan ranges",
        fontsize=13.0,
        fontweight="bold",
        y=0.987,
    )
    fig.subplots_adjust(top=0.93, bottom=0.075, left=0.12, right=0.985, hspace=0.48)
    three_png = OUT_DIR / "hps_mass_resolution_three_panel.png"
    three_pdf = OUT_DIR / "hps_mass_resolution_three_panel.pdf"
    fig.savefig(three_png, dpi=300)
    fig.savefig(three_pdf)
    plt.close(fig)
    print(three_png)
    print(three_pdf)


if __name__ == "__main__":
    main()
