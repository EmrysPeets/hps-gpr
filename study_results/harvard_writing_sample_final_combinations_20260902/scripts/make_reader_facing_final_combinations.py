#!/usr/bin/env python3
"""Render the final-combination ledgers without text over plotted curves.

This display-only derivative reads the released numerical ledger directly.  It
moves legends and the all-three-minimum callout into reserved figure margins;
no numerical input, curve, scale, or statistical interpretation is changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parent
SOURCE = DERIVATIVE.parent / "v4p9p12_final_dataset_combinations_20260902"
CURVES = SOURCE / "derived" / "final_dataset_result_curves.csv"
FIGURES = DERIVATIVE / "figures"
QA = DERIVATIVE / "qa" / "reader_facing_final_combinations"
EXPECTED_CURVES_SHA256 = "6f60467b8051ac23d6b7d357d7f325d0fea6be0f6e184497a7c94769ae6e9adc"

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style() -> None:
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


def save(fig: plt.Figure, stem: str) -> list[dict[str, object]]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        records.append(
            {"path": str(path.relative_to(DERIVATIVE)), "sha256": sha256(path)}
        )
    plt.close(fig)
    return records


def add_local_sigma_guides(ax: plt.Axes) -> None:
    for sigma, pvalue in ((1, 0.158655), (2, 0.0227501), (3, 0.0013499), (4, 3.1671e-5)):
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
            clip_on=False,
        )


def result_triptych(
    curves: pd.DataFrame,
    scopes: tuple[str, ...],
    stem: str,
    title: str,
) -> list[dict[str, object]]:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.05], "hspace": 0.08},
    )
    for scope in scopes:
        frame = curves[curves.scope_key == scope].sort_values("mass_MeV")
        kwargs = {
            "color": COLORS[scope],
            "ls": STYLES[scope],
            "lw": 2.1 if scope == "all_2015_2016_2021" else 1.65,
            "label": LABELS[scope],
        }
        axes[0].plot(frame.mass_MeV, frame.A90_full_template_events, **kwargs)
        axes[1].plot(frame.mass_MeV, frame.eps2_90, **kwargs)
        axes[2].plot(frame.mass_MeV, frame.p0_local_asymptotic, **kwargs)
    for ax in axes:
        ax.set_yscale("log")
    axes[0].set_ylabel("90% CL$_s$ upper limit\nfull-template events")
    axes[1].set_ylabel("90% CL$_s$ upper limit\non $\epsilon^2$")
    axes[2].set_ylabel(r"Local asymptotic $p_0$")
    axes[2].set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    axes[2].set_ylim(1.0e-8, 0.65)
    add_local_sigma_guides(axes[2])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, x=0.14, y=0.985, ha="left", fontweight="semibold")
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.935),
        fontsize=8.6,
    )
    fig.subplots_adjust(left=0.14, right=0.90, top=0.82, bottom=0.075)
    return save(fig, stem)


def pvalue_panels(curves: pd.DataFrame) -> list[dict[str, object]]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), sharey=True)
    all_handles = []
    all_labels = []
    for ax, scopes, title in (
        (axes[0], INDIVIDUAL, "Final data sets"),
        (axes[1], COMBINED, "Shared-coupling combinations"),
    ):
        for scope in scopes:
            frame = curves[curves.scope_key == scope].sort_values("mass_MeV")
            line = ax.plot(
                frame.mass_MeV,
                frame.p0_local_asymptotic,
                color=COLORS[scope],
                ls=STYLES[scope],
                lw=2.1 if scope == "all_2015_2016_2021" else 1.55,
                label=LABELS[scope],
            )[0]
            all_handles.append(line)
            all_labels.append(LABELS[scope])
        ax.set_yscale("log")
        ax.set_ylim(1.0e-8, 0.65)
        ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
        ax.set_title(title, loc="left", fontweight="semibold")
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
    fig.suptitle(
        "Observed local p-value scans for the current final-sample set",
        x=0.08,
        y=0.98,
        ha="left",
        fontweight="semibold",
    )
    fig.legend(
        all_handles,
        all_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.895),
        fontsize=8.1,
    )
    fig.text(
        0.5,
        0.755,
        f"The open circle marks the all-three minimum at {int(peak.mass_MeV)} MeV.",
        ha="center",
        fontsize=8.4,
        color="0.35",
    )
    fig.subplots_adjust(left=0.09, right=0.92, top=0.69, bottom=0.15, wspace=0.26)
    return save(fig, "final_asymptotic_pvalues")


def main() -> int:
    if sha256(CURVES) != EXPECTED_CURVES_SHA256:
        raise RuntimeError("Released final curve ledger changed; refusing display-only rebuild")
    configure_style()
    curves = pd.read_csv(CURVES)
    if set(curves.scope_key) != set(INDIVIDUAL + COMBINED):
        raise RuntimeError("Curve ledger does not contain exactly the seven final scopes")

    products = []
    products += result_triptych(
        curves,
        INDIVIDUAL,
        "individual_final_results",
        "Standalone observed results for the three current final samples",
    )
    products += result_triptych(
        curves,
        COMBINED,
        "combined_final_results",
        r"Simultaneous results with one shared $\epsilon^2$",
    )
    products += pvalue_panels(curves)
    QA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "pass",
        "change_scope": "Legends and the minimum callout remain outside data regions; faint footer notes were removed; numerical curves and axes are unchanged.",
        "source_curves": str(CURVES),
        "source_curves_sha256": sha256(CURVES),
        "products": products,
    }
    (QA / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
