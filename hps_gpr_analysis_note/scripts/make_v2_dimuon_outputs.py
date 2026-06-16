from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


BASE_DIR = Path("/Users/emryspeets/Desktop/gp_mods/combined_15_16_10pct_21_1pct/90cls_plots")
SOURCE_DIR = BASE_DIR / "v2"
SOURCE_PLOT_DIR = SOURCE_DIR / "note_comparison_plots"
OUTDIR = BASE_DIR / "v2_dimuon"

M_MU_GEV = 0.1056583745
M_DIMUON_GEV = 2.0 * M_MU_GEV

BAND_COLS = ["eps2_lo2", "eps2_lo1", "eps2_med", "eps2_hi1", "eps2_hi2"]
EPS2_PREFIXES = ("eps2_", "ul_eps2", "toy_eps2_uls", "projected_full_eps2")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 10.2,
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
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def dimuon_factor(mass_gev: np.ndarray | pd.Series) -> np.ndarray:
    m = np.asarray(mass_gev, dtype=float)
    factor = np.ones_like(m, dtype=float)
    above = m > M_DIMUON_GEV
    if np.any(above):
        ratio = np.sqrt(1.0 - 4.0 * M_MU_GEV**2 / m[above] ** 2)
        ratio *= 1.0 + 2.0 * M_MU_GEV**2 / m[above] ** 2
        factor[above] = 1.0 + ratio
    return factor


def add_dimuon_columns(df: pd.DataFrame, eps2_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if "mass_MeV" not in out.columns:
        out["mass_MeV"] = 1000.0 * out["mass_GeV"]
    out["dimuon_threshold_GeV"] = M_DIMUON_GEV
    out["dimuon_threshold_MeV"] = 1000.0 * M_DIMUON_GEV
    out["N_eff_dimuon"] = dimuon_factor(out["mass_GeV"])
    out["BR_ee_minimal"] = 1.0 / out["N_eff_dimuon"]

    cols = eps2_cols or [
        col
        for col in out.columns
        if any(col.startswith(prefix) for prefix in EPS2_PREFIXES)
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    for col in cols:
        raw_name = f"{col}_ee_channel"
        corr_name = f"{col}_dimuon"
        out[raw_name] = out[col]
        out[corr_name] = out[col].astype(float) * out["N_eff_dimuon"]
    return out


def dimuon_for_plotting(df: pd.DataFrame, eps2_cols: list[str] | None = None) -> pd.DataFrame:
    out = add_dimuon_columns(df, eps2_cols=eps2_cols)
    cols = eps2_cols or [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in EPS2_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in cols:
        out[col] = out[f"{col}_dimuon"]
    return out


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


def plot_combined_dimuon_bands(df: pd.DataFrame) -> None:
    x = df["mass_MeV"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    fill_between_segments(ax, x, df["eps2_lo2"], df["eps2_hi2"], color="#F5C542", alpha=0.24, label=r"$\pm2\sigma$ expected")
    fill_between_segments(ax, x, df["eps2_lo1"], df["eps2_hi1"], color="#3CB44B", alpha=0.35, label=r"$\pm1\sigma$ expected")
    plot_segments(ax, x, df["eps2_med"], color="#111111", linewidth=2.1, linestyle="--", label="Expected median")
    plot_segments(ax, x, df["eps2_obs"], color="#000000", linewidth=2.8, label="Observed 90% CL")
    ax.axvline(1000.0 * M_DIMUON_GEV, color="#C44E52", linestyle=":", linewidth=1.6, label=r"$2m_\mu$")
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title(r"90% CL limits with minimal-model dimuon-threshold correction")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)
    save_figure(fig, "combined_90cls_dimuon_corrected_eps2")


def plot_raw_vs_dimuon(df: pd.DataFrame) -> None:
    x = df["mass_MeV"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    plot_segments(ax, x, df["eps2_obs_ee_channel"], color="#111111", linewidth=2.1, label=r"Observed, $e^+e^-$ convention")
    plot_segments(ax, x, df["eps2_obs_dimuon"], color="#C44E52", linewidth=2.5, label=r"Observed, minimal model")
    plot_segments(ax, x, df["eps2_med_ee_channel"], color="#111111", linewidth=1.8, linestyle="--", label=r"Median, $e^+e^-$ convention")
    plot_segments(ax, x, df["eps2_med_dimuon"], color="#C44E52", linewidth=2.1, linestyle="--", label=r"Median, minimal model")
    ax.axvline(1000.0 * M_DIMUON_GEV, color="#555555", linestyle=":", linewidth=1.6, label=r"$2m_\mu$")
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title(r"Effect of the dimuon branching correction on the 90% CL limit")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)
    save_figure(fig, "combined_90cls_raw_vs_dimuon_eps2")


def plot_factor() -> None:
    masses = np.linspace(0.205, 0.25, 300)
    factors = dimuon_factor(masses)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(1000.0 * masses, factors, color="#C44E52", linewidth=2.6, label=r"$N_{\rm eff}=1/{\cal B}_{ee}$")
    ax.plot(1000.0 * masses, 1.0 / factors, color="#4C72B0", linewidth=2.2, linestyle="--", label=r"${\cal B}_{ee}$")
    ax.axvline(1000.0 * M_DIMUON_GEV, color="#555555", linestyle=":", linewidth=1.5, label=r"$2m_\mu$")
    ax.set_xlim(205, 250)
    ax.set_ylim(0.52, 1.82)
    ax.set_xlabel(r"$m_{A'}$ (MeV)")
    ax.set_ylabel("Multiplicative factor")
    ax.set_title(r"Minimal-model branching factor above the dimuon threshold")
    ax.legend(loc="center left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)
    save_figure(fig, "dimuon_factor_vs_mass")


def plot_projection_raw_vs_dimuon(proj: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = proj["mass_MeV"].to_numpy(float)
    plot_segments(ax, x, proj["projected_full_eps2_obs_ee_channel"], color="#7A3DBB", linewidth=2.0, label=r"Projected observed-equivalent, $e^+e^-$ convention")
    plot_segments(ax, x, proj["projected_full_eps2_obs_dimuon"], color="#C44E52", linewidth=2.5, label=r"Projected observed-equivalent, minimal model")
    plot_segments(ax, x, proj["projected_full_eps2_med_ee_channel"], color="#7A3DBB", linewidth=1.8, linestyle="--", label=r"Projected median, $e^+e^-$ convention")
    plot_segments(ax, x, proj["projected_full_eps2_med_dimuon"], color="#C44E52", linewidth=2.0, linestyle="--", label=r"Projected median, minimal model")
    ax.axvline(1000.0 * M_DIMUON_GEV, color="#555555", linestyle=":", linewidth=1.5, label=r"$2m_\mu$")
    ax.set_yscale("log")
    ax.set_xlim(18, 252)
    ax.set_xlabel(r"Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title(r"Projected full-data 90% CL reach with dimuon-threshold correction")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#c9c9c9")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    style_axis(ax)
    save_figure(fig, "projected_full_unblinded_reach_90cls_dimuon_eps2")


def write_latex_section(summary: pd.DataFrame) -> None:
    row_250 = summary.loc[np.isclose(summary["mass_MeV"], 250.0)].iloc[0]
    section = r"""\section{Dimuon-threshold convention for the 90\% CL scan}
\label{sec:dimuon-threshold-90cl}

The nominal yield-to-coupling conversion used in the GPR prompt-search workflow treats
the fitted resonance yield as an $A^\prime\to e^+e^-$ signal with the below-threshold
minimal-dark-photon branching convention. This is exact for the minimal visible
interpretation below the dimuon threshold,
\begin{equation}
2m_\mu = \SI{211.3167}{MeV},
\end{equation}
where the only open charged-lepton decay mode is $e^+e^-$. The 2021 90\% CL scan is
technically carried to \SI{250}{MeV} as an electron-channel resonance search, but
the opening of $A^\prime\to\mu^+\mu^-$ means that the same observed $e^+e^-$ yield
corresponds to a weaker minimal-model limit on $\epsilon^2$ above threshold.

For the mass interval considered here, which remains below the main hadronic threshold,
the required correction is the charged-lepton width ratio
\begin{equation}
R_{\mu/e}(m) =
\sqrt{1-\frac{4m_\mu^2}{m^2}}\,
\left(1+\frac{2m_\mu^2}{m^2}\right),
\qquad m>2m_\mu,
\end{equation}
with $R_{\mu/e}=0$ below threshold. The effective inverse branching factor is therefore
\begin{equation}
N_{\rm eff}(m) =
\frac{1}{\mathcal{B}(A^\prime\to e^+e^-)}
= 1 + R_{\mu/e}(m),
\end{equation}
and each electron-channel 90\% CL coupling limit is mapped to the minimal-model
interpretation by
\begin{equation}
\epsilon^2_{90,\rm min}(m)
= N_{\rm eff}(m)\,\epsilon^2_{90,ee}(m).
\label{eq:dimuon-corrected-eps2}
\end{equation}
The local \CLs\ extraction, signal-yield limit, and $p_0$ scan are unchanged by this
operation; only the coupling interpretation changes.

In the current v2 90\% CL files, the combined observed scan CSV stops at
\SI{210}{MeV}, while the combined-band export contains 2021-only rows from
\SIrange{211}{250}{MeV}. The correction is therefore unity through the quoted
below-threshold range and turns on only in the 2021-only high-mass tail. Numerically,
$N_{\rm eff}=1.120$ at \SI{212}{MeV}, $1.406$ at \SI{220}{MeV}, and
__N250__ at \SI{250}{MeV}, corresponding to
__BR250__ $e^+e^-$ branching at \SI{250}{MeV}.
The final look-elsewhere calibration should use the same mass interval as the quoted
physics claim: a result quoted only below \SI{210}{MeV} should calibrate the
hypertest over that restricted range, while a \SIrange{170}{250}{MeV} physics
claim should include the full extended range in the scan-level calibration.

\begin{figure}[H]
\centering
\includegraphics[width=0.82\linewidth]{dimuon_factor_vs_mass.pdf}
\caption{Minimal-model inverse branching factor $N_{\rm eff}=1/\mathcal{B}_{ee}$
used to reinterpret the electron-channel 90\% CL limits above the dimuon threshold.}
\label{fig:dimuon-factor-90cl}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\linewidth]{combined_90cls_raw_vs_dimuon_eps2.pdf}
\caption{Observed and median expected 90\% CL limits before and after applying
Eq.~\eqref{eq:dimuon-corrected-eps2}. The change is visible only above
$2m_\mu$, where the current combined-band export is 2021-only.}
\label{fig:dimuon-corrected-90cl}
\end{figure}
"""
    section = section.replace("__N250__", f"{row_250['N_eff_dimuon']:.3f}")
    section = section.replace("__BR250__", f"{row_250['BR_ee_minimal']:.3f}")
    (OUTDIR / "dimuon_threshold_90cl_section.tex").write_text(section)

    standalone = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{siunitx}
\newcommand{\CLs}{CL$_\mathrm{s}$}
\begin{document}
\input{dimuon_threshold_90cl_section}
\end{document}
"""
    (OUTDIR / "dimuon_threshold_90cl_section_standalone.tex").write_text(standalone)


def write_readme(source_manifest: dict[str, dict[str, str]], summary: pd.DataFrame) -> None:
    n_above = int((summary["mass_GeV"] > M_DIMUON_GEV).sum())
    lines = [
        "# v2_dimuon",
        "",
        "Dimuon-threshold reinterpretation of the June 10, 2026 v2 90% CL outputs.",
        "",
        "The correction multiplies electron-channel `epsilon^2` limits by",
        "",
        "`N_eff(m) = 1 + sqrt(1 - 4 m_mu^2 / m^2) * (1 + 2 m_mu^2 / m^2)`",
        "",
        f"for `m > 2 m_mu = {1000.0 * M_DIMUON_GEV:.4f} MeV`; below threshold the factor is one.",
        "",
        "Generated files:",
        "",
        "- `dimuon_threshold_90cl_section.tex`: note-ready LaTeX section.",
        "- `dimuon_threshold_90cl_section_standalone.tex`: tiny standalone wrapper for the section.",
        "- `combined_ul_bands_combined_all_90cls_with_dimuon_columns.csv`: source combined-band file with raw electron-channel and dimuon-corrected columns.",
        "- `combined_ul_bands_combined_all_90cls_dimuon_for_plotting.csv`: same combined-band table with primary `eps2_*` columns replaced by the minimal-model values.",
        "- `combined_single_90cls_with_dimuon_columns.csv`: single-dataset observed scan with `eps2_up_dimuon`.",
        "- `combined_combined_90cls_with_dimuon_columns.csv`: combined observed scan with `eps2_up_dimuon`.",
        "- `projected_full_unblinded_reach_90cls_with_dimuon_columns.csv`: projection table with corrected projected columns.",
        "- `dimuon_correction_factor_table.csv`: correction factors at review masses.",
        "- `make_v2_dimuon_outputs.py`: the generator used to build this directory.",
        "- `*.png`/`*.pdf`: audit plots used by the LaTeX section.",
        "",
        f"The factor table contains {n_above} masses above threshold.",
        "",
        "Source files:",
        "",
    ]
    for label, meta in source_manifest.items():
        lines.append(f"- `{label}`: `{meta['path']}` (mtime `{meta['mtime']}`)")
    lines.append("")
    (OUTDIR / "README.md").write_text("\n".join(lines))


def file_meta(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def main() -> None:
    ensure_dir(OUTDIR)
    setup_style()

    combined_source = SOURCE_PLOT_DIR / "combined_ul_bands_combined_all_90cls_corrected_for_plotting.csv"
    if not combined_source.exists():
        combined_source = SOURCE_DIR / "combined_ul_bands_combined_all.csv"
    combined = pd.read_csv(combined_source)
    combined_with = add_dimuon_columns(combined)
    combined_plot = dimuon_for_plotting(combined)
    combined_with.to_csv(OUTDIR / "combined_ul_bands_combined_all_90cls_with_dimuon_columns.csv", index=False)
    combined_plot.to_csv(OUTDIR / "combined_ul_bands_combined_all_90cls_dimuon_for_plotting.csv", index=False)

    single_source = SOURCE_DIR / "combined_single.csv"
    single = pd.read_csv(single_source)
    single_with = add_dimuon_columns(single, eps2_cols=["eps2_up"])
    single_with.to_csv(OUTDIR / "combined_single_90cls_with_dimuon_columns.csv", index=False)

    combined_scan_source = SOURCE_DIR / "combined_combined.csv"
    combined_scan = pd.read_csv(combined_scan_source)
    combined_scan_with = add_dimuon_columns(combined_scan, eps2_cols=["eps2_up"])
    combined_scan_with.to_csv(OUTDIR / "combined_combined_90cls_with_dimuon_columns.csv", index=False)

    projection_source = SOURCE_PLOT_DIR / "projected_full_unblinded_reach_90cls.csv"
    source_manifest = {
        "combined_bands": file_meta(combined_source),
        "combined_single": file_meta(single_source),
        "combined_scan": file_meta(combined_scan_source),
    }
    if projection_source.exists():
        projection = pd.read_csv(projection_source)
        projection_with = add_dimuon_columns(projection)
        projection_plot = dimuon_for_plotting(projection)
        projection_with.to_csv(OUTDIR / "projected_full_unblinded_reach_90cls_with_dimuon_columns.csv", index=False)
        projection_plot.to_csv(OUTDIR / "projected_full_unblinded_reach_90cls_dimuon_for_plotting.csv", index=False)
        plot_projection_raw_vs_dimuon(projection_with)
        source_manifest["projection"] = file_meta(projection_source)

    review_masses = np.array([0.210, 0.211, M_DIMUON_GEV, 0.212, 0.215, 0.220, 0.230, 0.240, 0.250])
    factor = dimuon_factor(review_masses)
    summary = pd.DataFrame(
        {
            "mass_GeV": review_masses,
            "mass_MeV": 1000.0 * review_masses,
            "N_eff_dimuon": factor,
            "BR_ee_minimal": 1.0 / factor,
            "epsilon_limit_factor": np.sqrt(factor),
        }
    )
    summary.to_csv(OUTDIR / "dimuon_correction_factor_table.csv", index=False)

    plot_combined_dimuon_bands(combined_plot)
    plot_raw_vs_dimuon(combined_with)
    plot_factor()
    write_latex_section(summary)
    (OUTDIR / "make_v2_dimuon_outputs.py").write_text(Path(__file__).read_text())
    write_readme(source_manifest, summary)
    (OUTDIR / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True))
    print(f"Wrote dimuon-corrected outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
