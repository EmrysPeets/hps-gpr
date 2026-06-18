#!/usr/bin/env python3
"""
make_ls_bound_plots.py

Standalone reproduction of the "length-scale bounds vs mass" plot sequence used in the v13 notebook.

Produces:
  (1) Overlay: l_max(m) in x=ln(m) units (dimensionless)
  (2) Overlay: sigma(m) in MeV (absolute resolution)
  (3) Overlay: sigma(m)/m (relative resolution)
  (4) Overlay: sigma_x(m)=ln(1+sigma/m) (signal width in log-mass coordinate)
  (5) Overlay: Delta m_equiv(l_max) in MeV, where Delta m_equiv = m (exp(l) - 1)
  (6) Overlay: Delta m_equiv(l_max) / sigma(m)
  (7) Overlay with band fill: [l_min, l_max] in x=ln(m) units
  (8) For 2021: l_max(m) for several k values (to motivate k ~ 8-9)

Mass is internally in GeV. All x-axes are shown in MeV.

Usage:
  python make_ls_bound_plots.py --outdir ls_bound_plots --show

Dependencies:
  numpy, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User-editable configuration
# =============================================================================

# Full plot x-range (GeV) but axis will be MeV
PLOT_XLIM_GEV: Tuple[float, float] = (0.010, 0.235)

# Per-dataset scan/valid ranges (GeV) for drawing curves
RANGES_GEV: Dict[str, Tuple[float, float]] = {
    "2015": (0.015, 0.130),
    "2016": (0.035, 0.200),
    "2021": (0.035, 0.230),
}

# Resolution polynomial coefficients for sigma(m) in GeV.
# Convention: coeffs[i] multiplies m^i, i.e. sigma(m) = sum_i coeffs[i] * m^i
SIGMA_COEFFS: Dict[str, List[float]] = {
    # v13 values:
    "2015": [-0.0000922283032152, 0.0532190838657],                 # linear
    "2016": [0.00038, 0.041, -0.27, 3.49, -11.11],                  # 4th order
    "2021": [0.0014786, -0.0011, 0.0687],                           # quadratic
}

# Optional piecewise "tail" model for sigma(m) beyond m0 (GeV):
# sigma(m>m0) = sigma(m0) + slope_override*(m-m0)
SIGMA_TAIL: Dict[str, Dict[str, float]] = {
    # v13 value:
    "2016": dict(m0=0.18, slope_override=0.0239),
}

# Length-scale bound factors in x=ln(m):
K_HI: Dict[str, float] = {"2015": 8.0, "2016": 8.0, "2021": 9.0}
K_LO: Dict[str, float] = {"2015": 0.5, "2016": 0.5, "2021": 0.5}

# For the "vary k" plot (2021)
K_SWEEP_2021: List[float] = [4, 6, 8, 9, 12]

# Sampling density for smooth curves
N_POINTS: int = 600

# Plot styling (kept conservative; uses default matplotlib color cycle)
FIGSIZE: Tuple[float, float] = (7.6, 4.9)
DPI: int = 300
LINEWIDTH: float = 2.2
GRID_ALPHA: float = 0.3


# =============================================================================
# Core math helpers
# =============================================================================

def poly_eval(coeffs: List[float], m: np.ndarray) -> np.ndarray:
    """Evaluate sigma(m) = sum_i coeffs[i] * m^i."""
    out = np.zeros_like(m, dtype=float)
    for i, c in enumerate(coeffs):
        out += c * (m ** i)
    return out


def sigma_model(year: str, m: np.ndarray) -> np.ndarray:
    """
    Mass resolution sigma(m) in GeV for a given dataset/year.

    Includes optional tail behavior if configured in SIGMA_TAIL.
    """
    m = np.asarray(m, dtype=float)
    if year not in SIGMA_COEFFS:
        raise KeyError(f"Missing SIGMA_COEFFS for year '{year}'")

    sig = poly_eval(SIGMA_COEFFS[year], m)

    # Optional tail override
    if year in SIGMA_TAIL:
        m0 = float(SIGMA_TAIL[year]["m0"])
        slope = float(SIGMA_TAIL[year]["slope_override"])
        sig_m0 = float(poly_eval(SIGMA_COEFFS[year], np.array([m0]))[0])
        sig = np.where(m <= m0, sig, sig_m0 + slope * (m - m0))

    # Safety: avoid non-positive sigma if a polynomial misbehaves out of range
    sig = np.maximum(sig, 1e-9)
    return sig


def sigma_x_logm(year: str, m: np.ndarray) -> np.ndarray:
    """
    Signal width in GP x-space when x = ln(m):
      sigma_x(m) = ln(m + sigma(m)) - ln(m) = ln(1 + sigma(m)/m)
    Dimensionless.
    """
    m = np.asarray(m, dtype=float)
    sig = sigma_model(year, m)
    r = sig / m
    return np.log1p(r)


def ls_bounds_logm(year: str, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Length-scale bounds in x = ln(m):
      l_min = k_lo * sigma_x
      l_max = k_hi * sigma_x
    """
    sx = sigma_x_logm(year, m)
    k_lo = float(K_LO[year])
    k_hi = float(K_HI[year])
    return k_lo * sx, k_hi * sx


def dm_equiv_from_ell(m: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """
    Convert a length-scale in x=ln(m) to an "equivalent" scale in mass units:
      Delta m_equiv = m (exp(ell) - 1)
    """
    m = np.asarray(m, dtype=float)
    ell = np.asarray(ell, dtype=float)
    return m * np.expm1(ell)


# =============================================================================
# Plot utilities
# =============================================================================

def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def _savefig(fig: plt.Figure, outdir: Path, stem: str) -> None:
    out_png = outdir / f"{stem}.png"
    out_pdf = outdir / f"{stem}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")


def _apply_common_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(PLOT_XLIM_GEV[0] * 1e3, PLOT_XLIM_GEV[1] * 1e3)  # MeV


# =============================================================================
# Plot sequence
# =============================================================================

def make_all_plots(outdir: Path, show: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib()

    years = ["2015", "2016", "2021"]

    # ---------- (1) Overlay: l_max(m) in log-space units ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        _, l_hi = ls_bounds_logm(y, m)
        ax.plot(
            m * 1e3,
            l_hi,
            linewidth=LINEWIDTH,
            label=rf"{y}: $\ell_{{\max}}=k\,\sigma_x$, $k={K_HI[y]:g}$",
        )
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Upper bound $\ell_{\max}$ in $x=\ln(m)$ (dimensionless)")
    ax.set_title("GP RBF length-scale upper bounds tied to local mass resolution")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "ls_hi_bounds_logm_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (2) Overlay: sigma(m) in MeV ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        sig = sigma_model(y, m) * 1e3  # MeV
        ax.plot(m * 1e3, sig, linewidth=LINEWIDTH, label=rf"{y}: $\sigma(m)$")
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Mass resolution $\sigma(m)$ [MeV]")
    ax.set_title("Detector mass resolution models (absolute width)")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "sigma_m_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (3) Overlay: sigma(m)/m ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        r = sigma_model(y, m) / m
        ax.plot(m * 1e3, r, linewidth=LINEWIDTH, label=rf"{y}: $\sigma(m)/m$")
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Relative resolution $\sigma(m)/m$ (dimensionless)")
    ax.set_title(r"Relative mass resolution (controls widths in $x=\ln(m)$)")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "sigma_over_m_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (4) Overlay: sigma_x = ln(1+sigma/m) ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        sx = sigma_x_logm(y, m)
        ax.plot(m * 1e3, sx, linewidth=LINEWIDTH, label=rf"{y}: $\sigma_x=\ln(1+\sigma/m)$")
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Signal width in log-space $\sigma_x$ (dimensionless)")
    ax.set_title(r"Signal-template width in GP coordinate $x=\ln(m)$")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "sigma_x_logm_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (5) Overlay: Delta m_equiv(l_max) in MeV ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        _, l_hi = ls_bounds_logm(y, m)
        dm = dm_equiv_from_ell(m, l_hi) * 1e3  # MeV
        ax.plot(m * 1e3, dm, linewidth=LINEWIDTH, label=rf"{y}: $\Delta m_\mathrm{{equiv}}(\ell_\max)$")
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Equivalent scale $\Delta m_\mathrm{equiv}$ [MeV]")
    ax.set_title(r"Interpreting log-space length-scale bounds in mass units")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "dm_equiv_from_lhi_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (6) Overlay: Delta m_equiv(l_max) / sigma(m) ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        sig = sigma_model(y, m)
        _, l_hi = ls_bounds_logm(y, m)
        dm = dm_equiv_from_ell(m, l_hi)
        ratio = dm / sig
        ax.plot(m * 1e3, ratio, linewidth=LINEWIDTH, label=rf"{y}: $\Delta m_\mathrm{{equiv}}/\sigma$")
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Scale ratio $\Delta m_\mathrm{equiv}(\ell_\max)/\sigma(m)$")
    ax.set_title(r"Upper-bound scale in mass units vs. the detector resolution")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "dm_equiv_over_sigma_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (7) Bounds band overlay: [l_min, l_max] ----------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for y in years:
        lo, hi = RANGES_GEV[y]
        m = np.linspace(lo, hi, N_POINTS)
        l_lo, l_hi = ls_bounds_logm(y, m)

        # Plot l_hi first to get its auto-selected color, then reuse it for l_lo + band
        (line_hi,) = ax.plot(
            m * 1e3,
            l_hi,
            linewidth=LINEWIDTH,
            label=rf"{y}: $\ell_\max$ (k={K_HI[y]:g})",
        )
        color = line_hi.get_color()
        ax.plot(
            m * 1e3,
            l_lo,
            linewidth=LINEWIDTH - 0.2,
            linestyle="--",
            color=color,
            label=rf"{y}: $\ell_\min$ (k={K_LO[y]:g})",
        )
        ax.fill_between(m * 1e3, l_lo, l_hi, color=color, alpha=0.12)

    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"Length-scale bounds in $x=\ln(m)$ (dimensionless)")
    ax.set_title(r"GP RBF length-scale bounds tied to the local mass resolution")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    _savefig(fig, outdir, "ls_bounds_band_logm_overlay")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- (8) Illustration: vary k for 2021 ----------
    year = "2021"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    lo, hi = RANGES_GEV[year]
    m = np.linspace(lo, hi, N_POINTS)
    sx = sigma_x_logm(year, m)
    for k in K_SWEEP_2021:
        ax.plot(m * 1e3, k * sx, linewidth=LINEWIDTH, label=rf"$k={k}$")
    ax.set_xlabel(r"Invariant mass $m$ [MeV]")
    ax.set_ylabel(r"$\ell_{\max} = k\,\sigma_x$ in $x=\ln(m)$ (dimensionless)")
    ax.set_title(r"How the resolution-scaled upper bound depends on the factor $k$ (2021)")
    _apply_common_axes(ax)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, outdir, "ls_hi_vs_k_2021")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make length-scale bound and transformation diagnostic plots.")
    p.add_argument("--outdir", type=str, default="ls_bound_plots", help="Output directory for PNG/PDF plots.")
    p.add_argument("--show", action="store_true", help="Display plots interactively as they are generated.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    make_all_plots(outdir=outdir, show=args.show)
    print(f"\nDone. Plots written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
