#!/usr/bin/env python3
"""Build the v4.9.8 Harvard selected-results ledgers and figures.

This is a presentation-only release.  It copies observed curves from reviewed
source tables, derives only an aligned total-yield coordinate for combinations,
and never reads or draws expected-limit bands.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_TABLES = HERE / "source_tables"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

INDIVIDUAL_SPECS = [
    {
        "scope_key": "individual_2015_full",
        "scope_label": "2015 full",
        "source_file": "v4p2_individual_observed.csv",
        "source_state": "historical_v4p2",
        "dataset": "2015",
        "gp_support": "14--135 MeV",
    },
    {
        "scope_key": "individual_2016_10pct",
        "scope_label": "2016 10%",
        "source_file": "v4p1_2016_10pct_observed.csv",
        "source_state": "reviewed_v4p1",
        "dataset": "2016",
        "gp_support": "30--210 MeV",
    },
    {
        "scope_key": "individual_2016_full",
        "scope_label": "2016 full (historical)",
        "source_file": "v4p2_individual_observed.csv",
        "source_state": "historical_v4p2",
        "dataset": "2016",
        "gp_support": "30--210 MeV",
    },
    {
        "scope_key": "individual_2021_1pct",
        "scope_label": "2021 1%",
        "source_file": "v4_2021_1pct_observed.csv",
        "source_state": "reviewed_v4_support040",
        "dataset": "2021",
        "gp_support": "40--300 MeV",
    },
    {
        "scope_key": "individual_2021_10pct",
        "scope_label": "2021 10% (current)",
        "source_file": "v4p9p5_2021_10pct_observed.csv",
        "source_state": "v4p9p5_support036",
        "dataset": "2021",
        "gp_support": "36--300 MeV",
    },
]

PAIR_SPECS = [
    ("pair_2015_2021", "2015 full + 2021 10%", "2015+2021"),
    ("pair_2016_2021", "2016 full + 2021 10%", "2016+2021"),
    ("pair_2015_2016", "2015 full + 2016 full", "2015+2016"),
]

ALL_THREE_KEY = "all_2015_2016_2021"
ALL_THREE_LABEL = "2015 full + 2016 full + 2021 10%"

COLORS_INDIVIDUAL = {
    "individual_2015_full": "#0072B2",
    "individual_2016_10pct": "#56B4E9",
    "individual_2016_full": "#009E73",
    "individual_2021_1pct": "#CC79A7",
    "individual_2021_10pct": "#D55E00",
}
STYLES_INDIVIDUAL = {
    "individual_2015_full": "-",
    "individual_2016_10pct": "--",
    "individual_2016_full": "-",
    "individual_2021_1pct": ":",
    "individual_2021_10pct": "-",
}
COLORS_COMBINED = {
    "pair_2015_2021": "#0072B2",
    "pair_2016_2021": "#D55E00",
    "pair_2015_2016": "#009E73",
    ALL_THREE_KEY: "#7A3E9D",
}
STYLES_COMBINED = {
    "pair_2015_2021": "-",
    "pair_2016_2021": "--",
    "pair_2015_2016": "-.",
    ALL_THREE_KEY: "-",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mass_mev(frame: pd.DataFrame) -> pd.Series:
    if "mass_MeV" in frame.columns:
        return pd.to_numeric(frame["mass_MeV"], errors="raise").astype(float)
    return 1000.0 * pd.to_numeric(frame["mass_GeV"], errors="raise").astype(float)


def load_individuals() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cache: dict[str, pd.DataFrame] = {}
    for spec in INDIVIDUAL_SPECS:
        source_file = str(spec["source_file"])
        if source_file not in cache:
            cache[source_file] = pd.read_csv(SOURCE_TABLES / source_file)
        source = cache[source_file].copy()
        if source_file == "v4p2_individual_observed.csv":
            source = source[source["dataset"].astype(str) == str(spec["dataset"])].copy()
        selected = pd.DataFrame(
            {
                "group": "individual",
                "scope_key": spec["scope_key"],
                "scope_label": spec["scope_label"],
                "dataset_set": spec["dataset"],
                "source_state": spec["source_state"],
                "source_file": source_file,
                "mass_GeV": pd.to_numeric(source["mass_GeV"], errors="raise"),
                "mass_MeV": mass_mev(source),
                "A90_events": pd.to_numeric(source["A_up"], errors="raise"),
                "eps2_90": pd.to_numeric(source["eps2_up"], errors="raise"),
                "p0_local_asymptotic": pd.to_numeric(source["p0_analytic"], errors="raise"),
                "Z_local_asymptotic": pd.to_numeric(source["Z_analytic"], errors="raise"),
                "gp_support": spec["gp_support"],
            }
        )
        selected["edge_diagnostic"] = (
            (selected["scope_key"] == "individual_2021_1pct")
            & selected["mass_MeV"].between(50.0, 52.0, inclusive="both")
        )
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    result["yield_coordinate"] = "standalone fitted signal yield"
    return result


def historical_k_map() -> dict[tuple[str, int], float]:
    source = pd.read_csv(SOURCE_TABLES / "v4p2_individual_observed.csv")
    mapping: dict[tuple[str, int], float] = {}
    for row in source.itertuples(index=False):
        key = (str(row.dataset), int(round(float(row.mass_MeV))))
        mapping[key] = float(row.A_up) / float(row.eps2_up)
    return mapping


def total_signal_yield(eps2: float, dataset_set: str, mass: float, k_map: dict[tuple[str, int], float]) -> float:
    mass_int = int(round(float(mass)))
    keys = str(dataset_set).split("+")
    missing = [key for key in keys if (key, mass_int) not in k_map]
    if missing:
        raise ValueError(f"missing historical K factor at {mass_int} MeV for {missing}")
    return float(eps2) * sum(k_map[(key, mass_int)] for key in keys)


def combination_frame(
    source: pd.DataFrame,
    scope_key: str,
    scope_label: str,
    dataset_set: str,
    source_file: str,
    k_map: dict[tuple[str, int], float],
) -> pd.DataFrame:
    mass = mass_mev(source)
    eps2 = pd.to_numeric(source["eps2_obs"], errors="raise").astype(float)
    selected = pd.DataFrame(
        {
            "group": "combination",
            "scope_key": scope_key,
            "scope_label": scope_label,
            "dataset_set": dataset_set,
            "source_state": "historical_v4p2",
            "source_file": source_file,
            "mass_GeV": pd.to_numeric(source["mass_GeV"], errors="raise").astype(float),
            "mass_MeV": mass,
            "eps2_90": eps2,
            "p0_local_asymptotic": pd.to_numeric(source["p0_analytic"], errors="raise").astype(float),
            "Z_local_asymptotic": pd.to_numeric(source["Z_analytic"], errors="raise").astype(float),
        }
    )
    selected["A90_events"] = [
        total_signal_yield(value, dataset_set, point, k_map)
        for value, point in zip(eps2, mass)
    ]
    selected["gp_support"] = "v4p2 campaign-specific supports"
    selected["edge_diagnostic"] = False
    selected["yield_coordinate"] = "derived total signal yield at shared eps2 limit"
    return selected


def load_combinations() -> pd.DataFrame:
    k_map = historical_k_map()
    pair_source = pd.read_csv(SOURCE_TABLES / "v4p2_standalone_pairwise_source.csv")
    frames: list[pd.DataFrame] = []
    for scope_key, scope_label, dataset_set in PAIR_SPECS:
        source = pair_source[pair_source["scope_key"] == scope_key].copy()
        frames.append(
            combination_frame(
                source,
                scope_key,
                scope_label,
                dataset_set,
                "v4p2_standalone_pairwise_source.csv",
                k_map,
            )
        )
    all_source = pd.read_csv(SOURCE_TABLES / "v4p2_all_period_source.csv")
    all_source = all_source[all_source["dataset_set"] == "2015+2016+2021"].copy()
    frames.append(
        combination_frame(
            all_source,
            ALL_THREE_KEY,
            ALL_THREE_LABEL,
            "2015+2016+2021",
            "v4p2_all_period_source.csv",
            k_map,
        )
    )
    return pd.concat(frames, ignore_index=True)


def finish_ledger(individual: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([individual, combined], ignore_index=True)
    result["limit_method"] = "observed asymptotic 90% CLs"
    result["pvalue_method"] = "fixed-mass local asymptotic profile LRT"
    result["limit_coordinate"] = "electron-channel epsilon^2"
    columns = [
        "group",
        "scope_key",
        "scope_label",
        "dataset_set",
        "source_state",
        "source_file",
        "mass_GeV",
        "mass_MeV",
        "A90_events",
        "eps2_90",
        "p0_local_asymptotic",
        "Z_local_asymptotic",
        "gp_support",
        "edge_diagnostic",
        "yield_coordinate",
        "limit_method",
        "pvalue_method",
        "limit_coordinate",
    ]
    return result[columns].sort_values(["group", "scope_key", "mass_MeV"]).reset_index(drop=True)


def minima_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope_key, frame in ledger.groupby("scope_key", sort=False):
        formal = frame.loc[frame["p0_local_asymptotic"].idxmin()]
        interpretable = frame[~frame["edge_diagnostic"]]
        interpreted = interpretable.loc[interpretable["p0_local_asymptotic"].idxmin()]
        rows.append(
            {
                "group": formal["group"],
                "scope_key": scope_key,
                "scope_label": formal["scope_label"],
                "source_state": formal["source_state"],
                "formal_min_mass_MeV": float(formal["mass_MeV"]),
                "formal_min_p0_local_asymptotic": float(formal["p0_local_asymptotic"]),
                "formal_min_Z_local_asymptotic": float(formal["Z_local_asymptotic"]),
                "interpretable_min_mass_MeV": float(interpreted["mass_MeV"]),
                "interpretable_min_p0_local_asymptotic": float(interpreted["p0_local_asymptotic"]),
                "interpretable_min_Z_local_asymptotic": float(interpreted["Z_local_asymptotic"]),
                "edge_rows_excluded_from_interpretation": int(frame["edge_diagnostic"].sum()),
                "global_calibration": "not performed",
            }
        )
    return pd.DataFrame(rows)


def source_manifest() -> pd.DataFrame:
    paths = sorted(SOURCE_TABLES.glob("*.csv")) + [
        FIGURES / "historical_all_three_m065_extraction.pdf",
        FIGURES / "historical_all_three_m065_extraction.png",
    ]
    return pd.DataFrame(
        [
            {
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    )


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[dict[str, object]]:
    products: list[dict[str, object]] = []
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        products.append(
            {"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
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
            "lw": 2.0 if scope_key == ALL_THREE_KEY else 1.45,
            "label": str(frame["scope_label"].iloc[0]),
        }
        axes[0].plot(frame["mass_MeV"], frame["A90_events"], **kwargs)
        axes[1].plot(frame["mass_MeV"], frame["eps2_90"], **kwargs)
        axes[2].plot(frame["mass_MeV"], frame["p0_local_asymptotic"], **kwargs)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
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
        all_three = selected[selected["scope_key"] == ALL_THREE_KEY]
        minimum = all_three.loc[all_three["p0_local_asymptotic"].idxmin()]
        axes[2].scatter(
            [minimum["mass_MeV"]],
            [minimum["p0_local_asymptotic"]],
            color="#111827",
            marker="*",
            s=90,
            zorder=6,
            label="all-three local minimum",
        )
        axes[2].legend(loc="lower right", frameon=False)
        xlim = (39.0, 180.0)
    style_axes(axes, xlim)
    return save_figure(fig, stem)


def plot_pvalue_series(ledger: pd.DataFrame) -> list[dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.0), constrained_layout=True)
    for axis, group, colors, styles, title in [
        (axes[0], "individual", COLORS_INDIVIDUAL, STYLES_INDIVIDUAL, "Standalone scans"),
        (axes[1], "combination", COLORS_COMBINED, STYLES_COMBINED, "Historical v4.2 shared-coupling scans"),
    ]:
        selected = ledger[ledger["group"] == group]
        for scope_key, frame in selected.groupby("scope_key", sort=False):
            frame = frame.sort_values("mass_MeV")
            axis.plot(
                frame["mass_MeV"],
                frame["p0_local_asymptotic"],
                color=colors[scope_key],
                ls=styles[scope_key],
                lw=2.0 if scope_key == ALL_THREE_KEY else 1.35,
                label=str(frame["scope_label"].iloc[0]),
            )
        axis.set_yscale("log")
        axis.set_ylabel(r"Local asymptotic $p_0$")
        axis.set_title(title, loc="left")
        axis.axhline(
            0.05,
            color="#4B5563",
            lw=0.85,
            ls=(0, (4, 3)),
            label=r"$p_0=0.05$",
        )
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


def write_summary_json(ledger: pd.DataFrame, minima: pd.DataFrame) -> None:
    all_three = minima[minima["scope_key"] == ALL_THREE_KEY].iloc[0]
    edge = minima[minima["scope_key"] == "individual_2021_1pct"].iloc[0]
    current_2021 = minima[minima["scope_key"] == "individual_2021_10pct"].iloc[0]
    payload = {
        "release": "v4p9p8_harvard_selected_results_20260902",
        "release_kind": "presentation-only selected-results assembly",
        "curve_rows": int(len(ledger)),
        "individual_rows": int((ledger["group"] == "individual").sum()),
        "combination_rows": int((ledger["group"] == "combination").sum()),
        "expected_bands_shown": False,
        "global_scan_calibration_performed": False,
        "all_three_historical_minimum": {
            "mass_MeV": float(all_three["formal_min_mass_MeV"]),
            "p0_local_asymptotic": float(all_three["formal_min_p0_local_asymptotic"]),
            "Z_local_asymptotic": float(all_three["formal_min_Z_local_asymptotic"]),
        },
        "current_2021_10pct_minimum": {
            "mass_MeV": float(current_2021["formal_min_mass_MeV"]),
            "p0_local_asymptotic": float(current_2021["formal_min_p0_local_asymptotic"]),
            "Z_local_asymptotic": float(current_2021["formal_min_Z_local_asymptotic"]),
        },
        "2021_1pct_edge_handling": {
            "formal_min_mass_MeV": float(edge["formal_min_mass_MeV"]),
            "formal_min_p0_local_asymptotic": float(edge["formal_min_p0_local_asymptotic"]),
            "interpretable_min_mass_MeV": float(edge["interpretable_min_mass_MeV"]),
            "interpretable_min_p0_local_asymptotic": float(edge["interpretable_min_p0_local_asymptotic"]),
            "excluded_edge_range_MeV": [50, 52],
        },
        "interpretation": (
            "All combinations and the full-2016 individual curve retain the reviewed historical v4.2 state. "
            "The current standalone 2021 10% curve is v4.9.5 and is not substituted into combinations. "
            "The v4.9.7 full-2016 support study did not select a support and contributes no observed curve."
        ),
    }
    (DERIVED / "selected_results_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_generated_tex(minima: pd.DataFrame) -> None:
    all_three = minima[minima["scope_key"] == ALL_THREE_KEY].iloc[0]
    current_2021 = minima[minima["scope_key"] == "individual_2021_10pct"].iloc[0]
    edge = minima[minima["scope_key"] == "individual_2021_1pct"].iloc[0]
    def latex_scientific(value: float, decimals: int = 6) -> str:
        mantissa, exponent = f"{value:.{decimals}e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"

    content = "\n".join(
        [
            "% Generated by ../build_selected_results.py; do not edit by hand.",
            rf"\newcommand{{\SelectedAllThreeMinMass}}{{{all_three['formal_min_mass_MeV']:.0f}}}",
            rf"\newcommand{{\SelectedAllThreeMinP}}{{{latex_scientific(float(all_three['formal_min_p0_local_asymptotic']))}}}",
            rf"\newcommand{{\SelectedAllThreeMinZ}}{{{all_three['formal_min_Z_local_asymptotic']:.6g}}}",
            rf"\newcommand{{\SelectedCurrentTwentyOneMinMass}}{{{current_2021['formal_min_mass_MeV']:.0f}}}",
            rf"\newcommand{{\SelectedCurrentTwentyOneMinP}}{{{latex_scientific(float(current_2021['formal_min_p0_local_asymptotic']))}}}",
            rf"\newcommand{{\SelectedCurrentTwentyOneMinZ}}{{{current_2021['formal_min_Z_local_asymptotic']:.6g}}}",
            rf"\newcommand{{\SelectedOnePercentInteriorMinMass}}{{{edge['interpretable_min_mass_MeV']:.0f}}}",
            rf"\newcommand{{\SelectedOnePercentInteriorMinP}}{{{latex_scientific(float(edge['interpretable_min_p0_local_asymptotic']))}}}",
            rf"\newcommand{{\SelectedOnePercentInteriorMinZ}}{{{edge['interpretable_min_Z_local_asymptotic']:.6g}}}",
            "",
        ]
    )
    path = HERE / "source" / "sections" / "generated_selected_results.tex"
    path.write_text(content)


def main() -> int:
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    individual = load_individuals()
    combined = load_combinations()
    ledger = finish_ledger(individual, combined)
    minima = minima_summary(ledger)

    ledger_path = DERIVED / "selected_result_curves.csv"
    minima_path = DERIVED / "minima_summary.csv"
    states_path = DERIVED / "result_state_ledger.csv"
    manifest_path = DERIVED / "source_manifest_sha256.csv"
    ledger.to_csv(ledger_path, index=False)
    minima.to_csv(minima_path, index=False)
    (
        ledger[
            ["group", "scope_key", "scope_label", "dataset_set", "source_state", "source_file", "gp_support"]
        ]
        .drop_duplicates()
        .sort_values(["group", "scope_key"])
        .to_csv(states_path, index=False)
    )
    source_manifest().to_csv(manifest_path, index=False)
    write_summary_json(ledger, minima)
    write_generated_tex(minima)

    products: list[dict[str, object]] = []
    products.extend(
        plot_triptych(
            ledger,
            "individual",
            "Observed standalone limits and fixed-mass p-values",
            "individual_results_triptych",
            COLORS_INDIVIDUAL,
            STYLES_INDIVIDUAL,
        )
    )
    products.extend(
        plot_triptych(
            ledger,
            "combination",
            "Historical v4.2 shared-coupling combinations",
            "combined_results_triptych",
            COLORS_COMBINED,
            STYLES_COMBINED,
        )
    )
    products.extend(plot_pvalue_series(ledger))
    products.extend(
        {
            "path": str(path.relative_to(HERE)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in [
            FIGURES / "historical_all_three_m065_extraction.pdf",
            FIGURES / "historical_all_three_m065_extraction.png",
        ]
    )
    pd.DataFrame(products).to_csv(DERIVED / "figure_inventory.csv", index=False)
    print(f"wrote {len(ledger)} curve rows, {len(minima)} minima rows, and {len(products)} figure products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
