#!/usr/bin/env python3
"""Render the final-dataset observed curves; no toys or bands are drawn."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CURVES = HERE / "derived" / "final_dataset_result_curves.csv"
FIGURES = HERE / "figures"

LABELS = {
    "individual_2015_full": "2015 full",
    "individual_2016_full": "2016 full",
    "individual_2021_10pct": "2021 10%",
    "pair_2015_2016": "2015 full + 2016 full",
    "pair_2015_2021": "2015 full + 2021 10%",
    "pair_2016_2021": "2016 full + 2021 10%",
    "all_2015_2016_2021": "All three final samples",
}
COLORS = {
    "individual_2015_full": "#0072B2",
    "individual_2016_full": "#D55E00",
    "individual_2021_10pct": "#009E73",
    "pair_2015_2016": "#0072B2",
    "pair_2015_2021": "#009E73",
    "pair_2016_2021": "#D55E00",
    "all_2015_2016_2021": "#7A3E9D",
}
STYLES = {
    "individual_2015_full": "-",
    "individual_2016_full": "--",
    "individual_2021_10pct": "-.",
    "pair_2015_2016": "-",
    "pair_2015_2021": "--",
    "pair_2016_2021": "-.",
    "all_2015_2016_2021": "-",
}
INDIVIDUAL = (
    "individual_2015_full",
    "individual_2016_full",
    "individual_2021_10pct",
)
COMBINED = (
    "pair_2015_2016",
    "pair_2015_2021",
    "pair_2016_2021",
    "all_2015_2016_2021",
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.55,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def add_local_sigma_guides(ax: plt.Axes) -> None:
    guides = ((1, 0.158655), (2, 0.0227501), (3, 0.0013499), (4, 3.1671e-5))
    for sigma, pvalue in guides:
        ax.axhline(pvalue, color="0.72", lw=0.65, ls=":" if sigma < 3 else "--")
        ax.text(
            1.002,
            pvalue,
            rf"{sigma}$\sigma$ local",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=7.3,
            color="0.42",
        )


def result_triptych(
    curves: pd.DataFrame,
    scopes: tuple[str, ...],
    stem: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.1),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.05], "hspace": 0.08},
    )
    for scope in scopes:
        frame = curves[curves.scope_key == scope].sort_values("mass_MeV")
        kwargs = dict(
            color=COLORS[scope],
            ls=STYLES[scope],
            lw=2.1 if scope == "all_2015_2016_2021" else 1.65,
            label=LABELS[scope],
        )
        axes[0].plot(frame.mass_MeV, frame.A90_full_template_events, **kwargs)
        axes[1].plot(frame.mass_MeV, frame.eps2_90, **kwargs)
        axes[2].plot(frame.mass_MeV, frame.p0_local_asymptotic, **kwargs)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_ylabel("90% CL$_s$ upper limit\nfull-template events")
    axes[1].set_ylabel("90% CL$_s$ upper limit\non $\\epsilon^2$")
    axes[2].set_ylabel(r"Local asymptotic $p_0$")
    axes[2].set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    axes[2].set_ylim(1.0e-8, 0.65)
    add_local_sigma_guides(axes[2])
    axes[0].legend(ncol=2, loc="best", fontsize=8.6)
    axes[0].set_title(title, loc="left", fontweight="semibold", pad=8)
    fig.text(
        0.5,
        0.012,
        (
            "Observed, fixed-mass asymptotic results conditional on frozen GP states. "
            "No expected bands or scan-wide calibration."
        ),
        ha="center",
        fontsize=8.2,
        color="0.35",
    )
    fig.subplots_adjust(left=0.14, right=0.90, top=0.955, bottom=0.075)
    save(fig, stem)


def pvalue_panels(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for ax, scopes, title in (
        (axes[0], INDIVIDUAL, "Final data sets"),
        (axes[1], COMBINED, "Shared-coupling combinations"),
    ):
        for scope in scopes:
            frame = curves[curves.scope_key == scope].sort_values("mass_MeV")
            ax.plot(
                frame.mass_MeV,
                frame.p0_local_asymptotic,
                color=COLORS[scope],
                ls=STYLES[scope],
                lw=2.1 if scope == "all_2015_2016_2021" else 1.55,
                label=LABELS[scope],
            )
        ax.set_yscale("log")
        ax.set_ylim(1.0e-8, 0.65)
        ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.legend(fontsize=8.1, loc="best")
        add_local_sigma_guides(ax)
    axes[0].set_ylabel(r"Local asymptotic background-only $p_0$")
    all_three = curves[curves.scope_key == "all_2015_2016_2021"]
    peak = all_three.loc[all_three.p0_local_asymptotic.idxmin()]
    axes[1].scatter(
        [peak.mass_MeV],
        [peak.p0_local_asymptotic],
        s=48,
        marker="o",
        facecolor="white",
        edgecolor=COLORS["all_2015_2016_2021"],
        linewidth=1.6,
        zorder=8,
    )
    axes[1].annotate(
        f"all-three minimum\n{int(peak.mass_MeV)} MeV",
        (peak.mass_MeV, peak.p0_local_asymptotic),
        xytext=(12, 16),
        textcoords="offset points",
        fontsize=8.2,
        color=COLORS["all_2015_2016_2021"],
        arrowprops={"arrowstyle": "-", "color": COLORS["all_2015_2016_2021"], "lw": 0.8},
    )
    fig.suptitle(
        "Observed local p-value scans for the current final-sample set",
        x=0.08,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        "Fixed-mass diagnostics only; the minimum is not look-elsewhere corrected.",
        ha="center",
        fontsize=8.2,
        color="0.35",
    )
    fig.subplots_adjust(left=0.09, right=0.92, top=0.87, bottom=0.15, wspace=0.26)
    save(fig, "final_asymptotic_pvalues")


def main() -> None:
    style()
    curves = pd.read_csv(CURVES)
    if set(curves.scope_key) != set(INDIVIDUAL + COMBINED):
        raise RuntimeError("curve ledger does not contain exactly the seven final scopes")
    if "2021 1%" in " ".join(curves.scope_label.astype(str)):
        raise RuntimeError("2021 1% leaked into final figures")
    result_triptych(
        curves,
        INDIVIDUAL,
        "individual_final_results",
        "Standalone observed results for the three current final samples",
    )
    result_triptych(
        curves,
        COMBINED,
        "combined_final_results",
        r"Simultaneous results with one shared $\epsilon^2$",
    )
    pvalue_panels(curves)
    inventory = {
        "figures": [
            "individual_final_results",
            "combined_final_results",
            "final_asymptotic_pvalues",
        ],
        "source_curve_sha256": __import__("hashlib").sha256(
            CURVES.read_bytes()
        ).hexdigest(),
        "claim_boundary": (
            "Observed local asymptotic results conditional on frozen model states; "
            "no toys, bands, global p-value, or coverage claim."
        ),
    }
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
