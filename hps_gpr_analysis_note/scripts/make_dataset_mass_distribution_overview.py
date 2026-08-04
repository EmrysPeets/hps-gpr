#!/usr/bin/env python3
"""Make three-panel invariant-mass distribution overview figures."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


NOTE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = NOTE_DIR / "dataset_summary_figs"
OVERLAY_XLIM_MEV = (0.0, 350.0)


@dataclass(frozen=True)
class DatasetPanel:
    key: str
    title: str
    root_candidates: tuple[Path, ...]
    hist_name: str
    search_mev: tuple[float, float]
    support_mev: tuple[float, float]
    xlim_mev: tuple[float, float]
    color: str


def candidate_paths(environment_variable: str, *defaults: str) -> tuple[Path, ...]:
    """Return an optional environment override followed by portable fallbacks."""
    override = os.environ.get(environment_variable, "").strip()
    paths = [Path(override)] if override else []
    paths.extend(Path(path) for path in defaults)
    return tuple(paths)


PANELS = (
    DatasetPanel(
        key="2015",
        title="HPS 2015",
        root_candidates=candidate_paths(
            "HPS_GPR_2015_ROOT",
            "/Users/emryspeets/research_plots/2015_data/invariant_mass_0pt5mm_full.root",
            "/sdf/home/e/epeets/move/2015_IMD.root",
        ),
        hist_name="invariant_mass",
        search_mev=(19.0, 90.0),
        support_mev=(14.0, 135.0),
        xlim_mev=(0.0, 150.0),
        color="#2E6F9E",
    ),
    DatasetPanel(
        key="2016",
        title="HPS 2016 full",
        root_candidates=candidate_paths(
            "HPS_GPR_2016_ROOT",
            "/Users/emryspeets/root_files/EventSelection_pass4Full.root",
        ),
        hist_name="h_Minv_General_Final_1",
        search_mev=(39.0, 180.0),
        support_mev=(30.0, 210.0),
        xlim_mev=(0.0, 230.0),
        color="#7A4FA3",
    ),
    DatasetPanel(
        key="2021_10pct",
        title="HPS 2021 10%",
        root_candidates=candidate_paths(
            "HPS_GPR_2021_10PCT_ROOT",
            "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root",
        ),
        hist_name="preselection/h_invM_8000",
        search_mev=(50.0, 250.0),
        support_mev=(40.0, 300.0),
        xlim_mev=(0.0, 330.0),
        color="#2F8F6B",
    ),
)


def resolve_path(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"None of the candidate ROOT files exists:\n  {joined}")


def load_histogram(panel: DatasetPanel) -> tuple[np.ndarray, np.ndarray, Path]:
    root_path = resolve_path(panel.root_candidates)
    with uproot.open(root_path) as handle:
        if panel.hist_name not in handle:
            keys = ", ".join(handle.keys()[:20])
            raise KeyError(
                f"Histogram {panel.hist_name!r} not found in {root_path}. "
                f"First keys: {keys}"
            )
        values, edges_gev = handle[panel.hist_name].to_numpy()
    edges_mev = np.asarray(edges_gev, dtype=float) * 1000.0
    values = np.asarray(values, dtype=float)
    return values, edges_mev, root_path


def step_xy(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def nice_ylim(y: np.ndarray, *, log: bool) -> tuple[float, float]:
    finite = y[np.isfinite(y)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return (0.8, 1.2)
    if log:
        return (max(0.6, float(np.nanmin(finite)) * 0.45), float(np.nanmax(finite)) * 2.0)
    return (0.0, float(np.nanmax(finite)) * 1.16)


def add_panel(ax: plt.Axes, panel: DatasetPanel, *, log: bool) -> Path:
    counts, edges_mev, root_path = load_histogram(panel)
    widths_mev = np.diff(edges_mev)
    density = np.divide(
        counts,
        widths_mev,
        out=np.zeros_like(counts, dtype=float),
        where=widths_mev > 0,
    )
    centers = 0.5 * (edges_mev[:-1] + edges_mev[1:])
    show = (centers >= panel.xlim_mev[0]) & (centers <= panel.xlim_mev[1])

    x_step, y_step = step_xy(edges_mev, density)
    step_mask = (x_step >= panel.xlim_mev[0]) & (x_step <= panel.xlim_mev[1])

    ax.axvspan(
        panel.support_mev[0],
        panel.support_mev[1],
        color="#C9CED6",
        alpha=0.34,
        lw=0,
        zorder=0,
    )
    ax.axvspan(
        panel.search_mev[0],
        panel.search_mev[1],
        color="#F2C14E",
        alpha=0.31,
        lw=0,
        zorder=1,
    )
    ax.plot(
        x_step[step_mask],
        y_step[step_mask],
        color=panel.color,
        lw=1.65,
        solid_capstyle="round",
        zorder=3,
    )
    ax.fill_between(
        x_step[step_mask],
        y_step[step_mask],
        step="pre",
        color=panel.color,
        alpha=0.10,
        lw=0,
        zorder=2,
    )

    ax.set_title(panel.title, fontsize=13, pad=10, weight="semibold")
    ax.set_xlim(*panel.xlim_mev)
    ax.set_ylim(*nice_ylim(density[show], log=log))
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="major", color="#E1E5EA", lw=0.85, alpha=0.85)
    if log:
        ax.grid(True, which="minor", color="#EEF1F5", lw=0.55, alpha=0.55)
    ax.tick_params(axis="both", which="major", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#27313B")
        spine.set_linewidth(1.05)

    lo, hi = panel.search_mev
    mid = 0.5 * (lo + hi)
    ymin, ymax = ax.get_ylim()
    if log:
        y_text = 10.0 ** (np.log10(float(ymin)) + 0.18 * (np.log10(float(ymax)) - np.log10(float(ymin))))
    else:
        y_text = float(ymin) + 0.16 * (float(ymax) - float(ymin))

    ax.text(
        mid,
        y_text,
        (
            f"Search {panel.search_mev[0]:.0f}--{panel.search_mev[1]:.0f} MeV\n"
            f"GP support {panel.support_mev[0]:.0f}--{panel.support_mev[1]:.0f} MeV"
        ),
        ha="center",
        va="center",
        fontsize=9.6,
        linespacing=1.15,
        color="#5F4100",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="none", alpha=0.86),
        zorder=5,
    )
    return root_path


def make_figure(*, log: bool) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), constrained_layout=False)
    used_paths = []
    for ax, panel in zip(axes, PANELS):
        used_paths.append(add_panel(ax, panel, log=log))

    fig.supxlabel(r"$e^+e^-$ invariant mass [MeV]", fontsize=13, y=0.047)
    fig.supylabel("Events / MeV", fontsize=13, x=0.028)
    fig.text(
        0.50,
        0.985,
        "Gray: GP fit support     Gold: tested signal-search interval",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4E5661",
    )
    fig.subplots_adjust(left=0.092, right=0.992, bottom=0.18, top=0.84, wspace=0.24)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "log" if log else "linear"
    out_png = OUT_DIR / f"invariant_mass_distributions_2015_2016_2021_{suffix}.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return used_paths


def make_normalized_overlay() -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.0, 4.75), constrained_layout=False)
    used_paths = []
    for panel in PANELS:
        counts, edges_mev, root_path = load_histogram(panel)
        used_paths.append(root_path)
        widths_mev = np.diff(edges_mev)
        total = float(np.sum(counts))
        if total <= 0.0:
            continue
        density = np.divide(
            counts,
            total * widths_mev,
            out=np.zeros_like(counts, dtype=float),
            where=widths_mev > 0,
        )
        x_step, y_step = step_xy(edges_mev, density)
        step_mask = (x_step >= OVERLAY_XLIM_MEV[0]) & (x_step <= OVERLAY_XLIM_MEV[1])
        ax.plot(
            x_step[step_mask],
            y_step[step_mask],
            color=panel.color,
            lw=1.9,
            label=panel.title,
            solid_capstyle="round",
        )

    ax.set_xlim(*OVERLAY_XLIM_MEV)
    ax.set_yscale("log")
    ax.set_xlabel(r"$e^+e^-$ invariant mass [MeV]", fontsize=12.5)
    ax.set_ylabel("Unit-normalized events / MeV", fontsize=12.5)
    ax.grid(True, which="major", color="#E1E5EA", lw=0.85, alpha=0.85)
    ax.grid(True, which="minor", color="#EEF1F5", lw=0.55, alpha=0.55)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#D0D5DD")
    ax.tick_params(axis="both", which="major", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#27313B")
        spine.set_linewidth(1.05)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.96)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "invariant_mass_distributions_2015_2016_2021_normalized.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return used_paths


def main() -> None:
    paths_log = make_figure(log=True)
    paths_norm = make_normalized_overlay()
    print(f"Wrote figures to {OUT_DIR}")
    print("Inputs:")
    for panel, path in zip(PANELS, paths_log):
        print(f"  {panel.title}: {path} :: {panel.hist_name}")
    assert paths_log == paths_norm


if __name__ == "__main__":
    main()
