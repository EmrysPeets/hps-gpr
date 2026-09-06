#!/usr/bin/env python3
"""Rebuild the four selected-results figures with reader-facing labels.

The numerical inputs and transformations are inherited from the reviewed
selected-results assembly.  This derivative changes plot labels only: internal
release names and chronology markers are replaced by descriptions of the data
sample or analysis configuration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
DERIVATIVE = SCRIPT_PATH.parents[1]
SOURCE_STUDY = DERIVATIVE.parent / "v4p9p8_harvard_selected_results_20260902"
SOURCE_TABLES = SOURCE_STUDY / "source_tables"
SOURCE_DERIVED = SOURCE_STUDY / "derived"
FIGURES = DERIVATIVE / "figures"
QA = DERIVATIVE / "qa" / "reader_facing_selected_results"

SOURCE_BUILD = SOURCE_STUDY / "build_selected_results.py"
spec = importlib.util.spec_from_file_location("selected_results_source_build", SOURCE_BUILD)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SOURCE_BUILD}")
source_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_build)

# Redirect every path used by the imported numerical builder.  The source study
# remains read-only, while all new products land in this derivative study.
source_build.HERE = DERIVATIVE
source_build.SOURCE_TABLES = SOURCE_TABLES
source_build.DERIVED = QA
source_build.FIGURES = FIGURES

READER_LABELS = {
    "individual_2015_full": "2015 full",
    "individual_2016_10pct": "2016 10%",
    "individual_2016_full": "2016 full",
    "individual_2021_1pct": "2021 1% (40--300 MeV support)",
    "individual_2021_10pct": "2021 10% (36--300 MeV support)",
}
source_build.INDIVIDUAL_SPECS = [
    {**item, "scope_label": READER_LABELS[str(item["scope_key"])]}
    for item in source_build.INDIVIDUAL_SPECS
]

NUMERIC_COLUMNS = [
    "mass_GeV",
    "mass_MeV",
    "A90_events",
    "eps2_90",
    "p0_local_asymptotic",
    "Z_local_asymptotic",
]
IDENTITY_COLUMNS = ["group", "scope_key", "dataset_set", "edge_diagnostic"]
CANONICAL_STEMS = [
    "individual_results_triptych",
    "combined_results_triptych",
    "asymptotic_pvalue_series",
    "historical_all_three_m065_extraction",
]
BANNED_RENDERED_TERMS = ["historical", "v4.", "(current)", "table-17", "table 17"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_plotting() -> None:
    source_build.configure_plotting()
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.0,
            "savefig.dpi": 220,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[dict[str, object]]:
    products: list[dict[str, object]] = []
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        products.append(
            {
                "path": str(path.relative_to(DERIVATIVE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    plt.close(fig)
    return products


def style_axes(axes: np.ndarray, xlim: tuple[float, float]) -> None:
    for axis in axes:
        axis.grid(alpha=0.18, which="both")
        axis.set_xlim(*xlim)
    axes[-1].set_xlabel(r"Invariant mass $m_{e^+e^-}$ [MeV]")


def plot_triptych(
    ledger: pd.DataFrame,
    group: str,
    title: str,
    stem: str,
    colors: dict[str, str],
    styles: dict[str, str],
) -> list[dict[str, object]]:
    selected = ledger[ledger["group"] == group]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.3, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.95]},
        constrained_layout=True,
    )
    for scope_key, frame in selected.groupby("scope_key", sort=False):
        frame = frame.sort_values("mass_MeV")
        kwargs = {
            "color": colors[scope_key],
            "ls": styles[scope_key],
            "lw": 2.0 if scope_key == source_build.ALL_THREE_KEY else 1.45,
            "label": str(frame["scope_label"].iloc[0]),
        }
        axes[0].plot(frame["mass_MeV"], frame["A90_events"], **kwargs)
        axes[1].plot(frame["mass_MeV"], frame["eps2_90"], **kwargs)
        axes[2].plot(frame["mass_MeV"], frame["p0_local_asymptotic"], **kwargs)

    for axis in axes:
        axis.set_yscale("log")
    axes[0].set_ylabel(r"Observed $A_{90}$ [events]")
    axes[1].set_ylabel(r"Observed 90% CL$_s$ $\epsilon^2$")
    axes[2].set_ylabel(r"Local asymptotic $p_0$")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[2].axhline(0.05, color="#4B5563", lw=0.9, ls=(0, (4, 3)), label=r"$p_0=0.05$")

    if group == "individual":
        for axis in axes:
            axis.axvspan(49.5, 52.5, color="#9CA3AF", alpha=0.13, zorder=0)
        edge = selected[selected["edge_diagnostic"]].sort_values("mass_MeV")
        axes[2].plot(
            edge["mass_MeV"],
            edge["p0_local_asymptotic"],
            color="#4B5563",
            lw=2.3,
            ls=":",
            marker="o",
            ms=3.4,
            label="2021 1% support-edge diagnostic",
            zorder=5,
        )
        axes[2].legend(loc="lower right", frameon=False)
        xlim = (19.0, 250.0)
    else:
        all_three = selected[selected["scope_key"] == source_build.ALL_THREE_KEY]
        minimum = all_three.loc[all_three["p0_local_asymptotic"].idxmin()]
        axes[2].scatter(
            [minimum["mass_MeV"]],
            [minimum["p0_local_asymptotic"]],
            color="#111827",
            marker="*",
            s=90,
            zorder=6,
            label="three-campaign local minimum",
        )
        axes[2].legend(loc="lower right", frameon=False)
        xlim = (39.0, 180.0)
    style_axes(axes, xlim)
    return save_figure(fig, stem)


def plot_pvalue_series(ledger: pd.DataFrame) -> list[dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.0), constrained_layout=True)
    rows = [
        (
            axes[0],
            "individual",
            source_build.COLORS_INDIVIDUAL,
            source_build.STYLES_INDIVIDUAL,
            "Standalone scans",
        ),
        (
            axes[1],
            "combination",
            source_build.COLORS_COMBINED,
            source_build.STYLES_COMBINED,
            "Three-campaign shared-coupling scans",
        ),
    ]
    for axis, group, colors, styles, title in rows:
        selected = ledger[ledger["group"] == group]
        for scope_key, frame in selected.groupby("scope_key", sort=False):
            frame = frame.sort_values("mass_MeV")
            axis.plot(
                frame["mass_MeV"],
                frame["p0_local_asymptotic"],
                color=colors[scope_key],
                ls=styles[scope_key],
                lw=2.0 if scope_key == source_build.ALL_THREE_KEY else 1.35,
                label=str(frame["scope_label"].iloc[0]),
            )
        axis.set_yscale("log")
        axis.set_ylabel(r"Local asymptotic $p_0$")
        axis.set_title(title, loc="left")
        axis.axhline(0.05, color="#4B5563", lw=0.85, ls=(0, (4, 3)), label=r"$p_0=0.05$")
        axis.grid(alpha=0.18, which="both")
    axes[0].axvspan(49.5, 52.5, color="#9CA3AF", alpha=0.15, zorder=0)
    edge = ledger[(ledger["scope_key"] == "individual_2021_1pct") & ledger["edge_diagnostic"]]
    axes[0].plot(
        edge["mass_MeV"],
        edge["p0_local_asymptotic"],
        color="#4B5563",
        lw=2.3,
        ls=":",
        marker="o",
        ms=3.2,
        label="support-edge diagnostic",
    )
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    axes[1].legend(loc="lower right", frameon=False, ncol=2)
    axes[0].set_xlim(19, 250)
    axes[1].set_xlim(39, 180)
    axes[1].set_xlabel(r"Invariant mass $m_{e^+e^-}$ [MeV]")
    fig.suptitle("Observed fixed-mass asymptotic p-value series; no scan-wide calibration", fontsize=13)
    return save_figure(fig, "asymptotic_pvalue_series")


def plot_three_campaign_extraction() -> list[dict[str, object]]:
    data = pd.read_csv(SOURCE_TABLES / "v4p2_m065_plot_data.csv")
    summary = pd.read_csv(SOURCE_TABLES / "v4p2_m065_fit_summary.csv")
    summary = summary[summary["dataset"].astype(str).isin(["2015", "2016", "2021"])].copy()
    summary["dataset"] = summary["dataset"].astype(str)
    data["dataset"] = data["dataset"].astype(str)

    colors = {"2015": "#0072B2", "2016": "#D55E00", "2021": "#009E73"}
    labels = {
        "2015": "2015 full",
        "2016": "2016 full",
        "2021": "2021 10%, 40--300 MeV support",
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 8.3),
        gridspec_kw={"height_ratios": [1.18, 0.77]},
    )
    for index, dataset in enumerate(["2015", "2016", "2021"]):
        frame = data[data["dataset"] == dataset].sort_values("bin_center_MeV")
        info = summary[summary["dataset"] == dataset].iloc[0]
        x = frame["bin_center_MeV"].to_numpy(float)
        observed = frame["observed"].to_numpy(float)
        gp_mean = frame["gp_mean"].to_numpy(float)
        gp_sigma = frame["gp_predictive_sigma"].to_numpy(float)
        residual = frame["data_minus_gp"].to_numpy(float)
        total_sigma = frame["display_total_sigma"].to_numpy(float)
        standalone = frame["standalone_signal_display"].to_numpy(float)
        shared = frame["shared_signal_display"].to_numpy(float)
        top, bottom = axes[0, index], axes[1, index]

        top.fill_between(x, gp_mean - gp_sigma, gp_mean + gp_sigma, color="#4C78A8", alpha=0.20, lw=0)
        top.plot(x, gp_mean, color="#2F5D8A", lw=1.25)
        top.scatter(x, observed, s=8, color="#20242A", zorder=3)
        bottom.errorbar(
            x,
            residual,
            yerr=total_sigma,
            fmt="o",
            color="#20242A",
            ecolor="#6B7280",
            elinewidth=0.55,
            ms=2.6,
            capsize=0,
            zorder=2,
        )
        bottom.plot(x, standalone, color="#E69F00", lw=1.45, ls=(0, (5, 3)))
        bottom.plot(x, shared, color="#B31B34", lw=1.65)
        bottom.axhline(0.0, color="#6B7280", lw=0.75)
        for axis in (top, bottom):
            axis.axvline(float(info["blind_lo_MeV"]), color="#8C6BB1", lw=1.0, ls=(0, (4, 3)))
            axis.axvline(float(info["blind_hi_MeV"]), color="#8C6BB1", lw=1.0, ls=(0, (4, 3)))
            axis.grid(alpha=0.18)
            axis.set_xlim(float(x.min()), float(x.max()))
        bin_width = float(np.median(np.diff(x)))
        top.set_title(f"{labels[dataset]} ({bin_width:g} MeV/bin)", color=colors[dataset], fontsize=11)
        bottom.set_xlabel(r"$m_{e^+e^-}$ [MeV]")

    axes[0, 0].set_ylabel("Events / bin")
    axes[1, 0].set_ylabel("Data - GP background")
    handles = [
        Line2D([], [], color="#20242A", marker="o", lw=0, ms=4, label="Observed data"),
        Line2D([], [], color="#2F5D8A", lw=1.25, label="Fixed-GP background mean"),
        Patch(facecolor="#4C78A8", alpha=0.20, label="GP predictive uncertainty"),
        Line2D([], [], color="#E69F00", lw=1.45, ls=(0, (5, 3)), label="Standalone best-fit signal"),
        Line2D([], [], color="#B31B34", lw=1.65, label=r"Shared-$\epsilon^2$ best-fit signal"),
        Line2D([], [], color="#8C6BB1", lw=1.0, ls=(0, (4, 3)), label=r"Blind-window boundaries ($\pm2.25\sigma_m$)"),
    ]
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.79, wspace=0.16, hspace=0.12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=3, frameon=False)
    fig.suptitle("Observed signal extraction at 65 MeV", fontsize=14, y=0.98)
    return save_figure(fig, "historical_all_three_m065_extraction")


def verify_numerical_identity(ledger: pd.DataFrame) -> dict[str, object]:
    reference = pd.read_csv(SOURCE_DERIVED / "selected_result_curves.csv")
    order = ["group", "scope_key", "mass_MeV"]
    candidate = ledger.sort_values(order).reset_index(drop=True)
    reference = reference.sort_values(order).reset_index(drop=True)
    assert_frame_equal(
        candidate[IDENTITY_COLUMNS + NUMERIC_COLUMNS],
        reference[IDENTITY_COLUMNS + NUMERIC_COLUMNS],
        check_dtype=False,
        check_exact=True,
    )
    raw_candidate = np.ascontiguousarray(candidate[NUMERIC_COLUMNS].to_numpy(np.float64)).tobytes()
    raw_reference = np.ascontiguousarray(reference[NUMERIC_COLUMNS].to_numpy(np.float64)).tobytes()
    candidate_hash = hashlib.sha256(raw_candidate).hexdigest()
    reference_hash = hashlib.sha256(raw_reference).hexdigest()
    if candidate_hash != reference_hash:
        raise AssertionError("canonical numeric-array hashes differ")
    return {
        "status": "pass",
        "rows": int(len(candidate)),
        "numeric_columns": NUMERIC_COLUMNS,
        "canonical_numeric_array_sha256": candidate_hash,
        "source_csv_sha256": sha256_file(SOURCE_DERIVED / "selected_result_curves.csv"),
        "m065_plot_data_csv_sha256": sha256_file(SOURCE_TABLES / "v4p2_m065_plot_data.csv"),
        "m065_fit_summary_csv_sha256": sha256_file(SOURCE_TABLES / "v4p2_m065_fit_summary.csv"),
    }


def verify_rendered_text() -> dict[str, object]:
    extracted: dict[str, str] = {}
    for stem in CANONICAL_STEMS:
        path = FIGURES / f"{stem}.pdf"
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        extracted[stem] = text
        lower = text.lower()
        found = [term for term in BANNED_RENDERED_TERMS if term.lower() in lower]
        if found:
            raise AssertionError(f"{path.name} retains internal labels: {found}")
    (QA / "extracted_figure_text.txt").write_text(
        "\n\n".join(f"=== {stem} ===\n{text}" for stem, text in extracted.items()) + "\n"
    )
    return {
        "status": "pass",
        "checked_terms": BANNED_RENDERED_TERMS,
        "pdfs": [f"figures/{stem}.pdf" for stem in CANONICAL_STEMS],
    }


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    # Plot the reviewed ledger directly.  Replacing only scope_label keeps the
    # parsed floating-point arrays byte-for-byte identical to the source CSV.
    ledger = pd.read_csv(SOURCE_DERIVED / "selected_result_curves.csv")
    for scope_key, label in READER_LABELS.items():
        ledger.loc[ledger["scope_key"] == scope_key, "scope_label"] = label
    numerical_check = verify_numerical_identity(ledger)

    # This ledger differs from the source only in human-readable display labels.
    ledger.to_csv(QA / "reader_facing_curve_ledger.csv", index=False)
    products: list[dict[str, object]] = []
    products.extend(
        plot_triptych(
            ledger,
            "individual",
            "Observed standalone limits and fixed-mass p-values",
            "individual_results_triptych",
            source_build.COLORS_INDIVIDUAL,
            source_build.STYLES_INDIVIDUAL,
        )
    )
    products.extend(
        plot_triptych(
            ledger,
            "combination",
            "Three-campaign shared-coupling combinations",
            "combined_results_triptych",
            source_build.COLORS_COMBINED,
            source_build.STYLES_COMBINED,
        )
    )
    products.extend(plot_pvalue_series(ledger))
    products.extend(plot_three_campaign_extraction())
    text_check = verify_rendered_text()

    report = {
        "source_study": str(SOURCE_STUDY),
        "derivative_study": str(DERIVATIVE),
        "change_scope": "rendered labels only",
        "numerical_identity": numerical_check,
        "rendered_text": text_check,
        "products": products,
    }
    (QA / "figure_qa_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {len(products)} reader-facing figure products; numerical and text checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
