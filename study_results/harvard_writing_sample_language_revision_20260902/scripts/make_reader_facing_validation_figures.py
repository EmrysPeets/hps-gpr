#!/usr/bin/env python3
"""Build reader-facing 2021 background-validation figures for the writing sample.

The numerical inputs and statistical calculations are unchanged from the source
study.  Only the labels have been rewritten so that a reader does not need the
internal analysis-release history to understand the comparisons.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, t


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_STUDY = next(
    path
    for path in ROOT.parent.glob("*_2021_background_validation_consolidation_20260817")
    if (path / "analyze_and_plot.py").is_file()
)
INPUT_DERIVED = SOURCE_STUDY / "derived"
DERIVED = ROOT / "qa" / "reader_facing_validation_figures"
FIGURES = next(
    (ROOT / "source" / "toy_generation_figs").glob(
        "*_2021_background_validation_consolidation_20260817"
    )
)
HISTORICAL = (
    next((SOURCE_STUDY / "reference").glob("*_full100"))
    / "derived"
    / "accepted_extraction_rows.csv"
)
FIGURE65_STEM = next(
    FIGURES.glob("figure65_onepctx10_65mev_*_vs_historical_90cl.pdf")
).stem
NEW_LANES = {
    "2021_1pct_x10": {
        "path": next(INPUT_DERIVED.glob("onepctx10_*_040_300"))
        / "accepted_extraction_rows.csv",
        "initial_n": 20,
        "study": "threshold-refined 65 MeV model, 40--300 MeV GP support",
        "marker": "D",
        "substitution_color": "#7B2CBF",
    },
    "2021_10pct": {
        "path": next(INPUT_DERIVED.glob("native10_*_030_300"))
        / "accepted_extraction_rows.csv",
        "initial_n": 25,
        "study": "extended-support 65 MeV model, 30--300 MeV GP support",
        "marker": "*",
        "substitution_color": "#D81B60",
    },
}
SCENARIOS = ["2021_1pct_x10", "2021_1pct_x100", "2021_10pct", "2021_10pct_x10"]
SCENARIO_LABELS = {
    "2021_1pct_x10": r"2021 1% source $\times10$",
    "2021_10pct": "2021 native 10% source",
    "2021_1pct_x100": r"2021 1% source $\times100$",
    "2021_10pct_x10": r"2021 10% source $\times10$",
}
SCENARIO_COLORS = {
    "2021_1pct_x10": "#0072B2",
    "2021_10pct": "#D55E00",
    "2021_1pct_x100": "#009E73",
    "2021_10pct_x10": "#CC79A7",
}
STRENGTH_COLORS = {0.0: "#1B1B1B", 1.0: "#0072B2", 3.0: "#E69F00", 5.0: "#009E73"}
STRENGTH_MARKERS = {0.0: "o", 1.0: "s", 3.0: "^", 5.0: "v"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def moment(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        raise RuntimeError("at least two finite pulls are required")
    mean = float(values.mean())
    width = float(values.std(ddof=1))
    tcrit = float(t.ppf(0.95, n - 1))
    qlo = float(chi2.ppf(0.05, n - 1))
    qhi = float(chi2.ppf(0.95, n - 1))
    tstat = mean / (width / math.sqrt(n)) if width > 0 else math.copysign(math.inf, mean)
    pvalue = float(2.0 * t.sf(abs(tstat), n - 1))
    mean_low = mean - tcrit * width / math.sqrt(n)
    mean_high = mean + tcrit * width / math.sqrt(n)
    width_low = math.sqrt((n - 1) * width * width / qhi)
    width_high = math.sqrt((n - 1) * width * width / qlo)
    material = abs(mean) >= 0.2
    rejects_zero_90 = mean_low > 0.0 or mean_high < 0.0
    return {
        "n": n,
        "mean_pull": mean,
        "mean_ci90_low": mean_low,
        "mean_ci90_high": mean_high,
        "sample_width": width,
        "width_ci90_low": width_low,
        "width_ci90_high": width_high,
        "mean_zero_t_statistic": tstat,
        "mean_zero_two_sided_t_pvalue": pvalue,
        "material_abs_mean_ge_0p2": material,
        "mean_ci90_excludes_zero": rejects_zero_90,
        "bias_screen_flag": bool(material and rejects_zero_90),
        "width_ci90_contains_one": bool(width_low <= 1.0 <= width_high),
    }


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    historical = pd.read_csv(HISTORICAL)
    required = {"scenario", "mass_MeV", "inj_nsigma", "background_toy_index", "pull"}
    if not required.issubset(historical.columns):
        raise RuntimeError("historical accepted ledger schema drift")
    lanes: dict[str, pd.DataFrame] = {}
    for scenario, record in NEW_LANES.items():
        frame = pd.read_csv(record["path"])
        if not required.issubset(frame.columns):
            raise RuntimeError(f"new accepted ledger schema drift: {scenario}")
        if set(frame["scenario"].unique()) != {scenario}:
            raise RuntimeError(f"scenario contamination: {scenario}")
        if set(np.round(frame["mass_MeV"].unique(), 6)) != {65.0}:
            raise RuntimeError(f"mass contamination: {scenario}")
        if set(frame["inj_nsigma"].unique()) != {0.0, 1.0, 3.0, 5.0}:
            raise RuntimeError(f"strength inventory drift: {scenario}")
        lanes[scenario] = frame
    return historical, lanes


def summarize_new(lanes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, frame in lanes.items():
        initial_n = int(NEW_LANES[scenario]["initial_n"])
        cohorts = {
            "development_subset": frame[frame.background_toy_index < initial_n],
            "independent_continuation": frame[frame.background_toy_index >= initial_n],
            "full100": frame,
        }
        for cohort, cohort_frame in cohorts.items():
            for strength, group in cohort_frame.groupby("inj_nsigma", sort=True):
                stats = moment(group["pull"].to_numpy())
                rows.append({
                    "scenario": scenario,
                    "mass_MeV": 65.0,
                    "inj_nsigma": float(strength),
                    "cohort": cohort,
                    "truth_and_support": NEW_LANES[scenario]["study"],
                    **stats,
                })
    return pd.DataFrame(rows)


def summarize_historical(historical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = historical[historical.scenario.isin(SCENARIOS)].copy()
    for (scenario, mass, strength), group in selected.groupby(
        ["scenario", "mass_MeV", "inj_nsigma"], sort=True
    ):
        rows.append({
            "scenario": scenario,
            "mass_MeV": float(mass),
            "inj_nsigma": float(strength),
            "cohort": "baseline_full100",
            "truth_and_support": "baseline smooth-threshold model, 40--300 MeV GP support",
            **moment(group["pull"].to_numpy()),
        })
    return pd.DataFrame(rows)


def consolidated_rows(historical_summary: pd.DataFrame, new_summary: pd.DataFrame) -> pd.DataFrame:
    consolidated = historical_summary.copy()
    replace_mask = consolidated.scenario.isin(NEW_LANES) & np.isclose(
        consolidated.mass_MeV, 65.0
    )
    consolidated = consolidated.loc[~replace_mask].copy()
    new_full = new_summary[new_summary.cohort == "full100"].copy()
    consolidated = pd.concat([consolidated, new_full], ignore_index=True)
    consolidated["is_65mev_substitution"] = (
        consolidated.scenario.isin(NEW_LANES) & np.isclose(consolidated.mass_MeV, 65.0)
    )
    return consolidated.sort_values(["scenario", "inj_nsigma", "mass_MeV"]).reset_index(drop=True)


def jitter(indices: np.ndarray, scale: float = 0.9) -> np.ndarray:
    return scale * (((np.asarray(indices, dtype=int) * 37) % 101) / 100.0 - 0.5)


def setup_style() -> None:
    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.22,
    })


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{stem}.pdf", FIGURES / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def make_figure64(
    historical: pd.DataFrame,
    lanes: dict[str, pd.DataFrame],
    consolidated: pd.DataFrame,
) -> list[Path]:
    fig, axes = plt.subplots(4, 3, figsize=(15.0, 15.0), constrained_layout=True)
    masses = [65.0, 90.0, 120.0, 180.0, 210.0]
    for row, scenario in enumerate(SCENARIOS):
        color = SCENARIO_COLORS[scenario]
        hist_zero = historical[(historical.scenario == scenario) & np.isclose(historical.inj_nsigma, 0.0)]
        ax_pull, ax_mean, ax_width = axes[row]
        for mass in masses:
            if scenario in NEW_LANES and mass == 65.0:
                group = lanes[scenario][np.isclose(lanes[scenario].inj_nsigma, 0.0)]
                point_color = NEW_LANES[scenario]["substitution_color"]
                marker = NEW_LANES[scenario]["marker"]
                size = 24 if marker == "*" else 15
            else:
                group = hist_zero[np.isclose(hist_zero.mass_MeV, mass)]
                point_color = color
                marker = "o"
                size = 9
            ax_pull.scatter(
                mass + jitter(group.background_toy_index.to_numpy()),
                group.pull,
                s=size,
                marker=marker,
                color=point_color,
                alpha=0.48,
                linewidths=0.25,
                edgecolors="white",
                rasterized=True,
            )
            ax_pull.plot([mass - 1.6, mass + 1.6], [np.median(group.pull)] * 2, color="black", lw=1.4)
        cell = consolidated[(consolidated.scenario == scenario) & np.isclose(consolidated.inj_nsigma, 0.0)]
        for kind, ax, value, low, high, ylabel in (
            ("mean", ax_mean, "mean_pull", "mean_ci90_low", "mean_ci90_high", "mean pull"),
            ("width", ax_width, "sample_width", "width_ci90_low", "width_ci90_high", "sample pull width"),
        ):
            base = cell[~cell.is_65mev_substitution]
            ax.errorbar(
                base.mass_MeV,
                base[value],
                yerr=np.vstack([base[value] - base[low], base[high] - base[value]]),
                fmt="o-",
                color=color,
                lw=1.5,
                ms=4.5,
                capsize=2.5,
                label="baseline smooth-threshold model (90% interval)" if row == 0 else None,
            )
            if scenario in NEW_LANES:
                new = cell[cell.is_65mev_substitution].iloc[0]
                sub_color = NEW_LANES[scenario]["substitution_color"]
                sub_marker = NEW_LANES[scenario]["marker"]
                ax.errorbar(
                    [65.0], [new[value]],
                    yerr=[[new[value] - new[low]], [new[high] - new[value]]],
                    fmt=sub_marker,
                    color=sub_color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    ms=8 if sub_marker == "*" else 6,
                    capsize=3,
                    zorder=5,
                )
            ax.set_ylabel(ylabel)
            ax.set_xticks(masses)
            if kind == "mean":
                ax.axhline(0.0, color="black", lw=0.9)
                ax.axhspan(-0.5, 0.5, color="0.7", alpha=0.12)
            else:
                ax.axhline(1.0, color="black", lw=0.9, ls="--")
        ax_pull.axhline(0.0, color="black", lw=0.9, ls="--")
        ax_pull.set_ylabel("zero-signal pull")
        ax_pull.set_xticks(masses)
        ax_pull.set_title(SCENARIO_LABELS[scenario] + ": background-only pulls")
        ax_mean.set_title("Spurious-signal mean")
        ax_width.set_title("Spurious-signal width")
        for ax in axes[row]:
            ax.set_xlabel("mass [MeV]")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles.extend([
        mpl.lines.Line2D(
            [], [], color=NEW_LANES["2021_1pct_x10"]["substitution_color"],
            marker="D", markeredgecolor="black", markeredgewidth=0.5,
            linestyle="None", markersize=6,
            label=r"threshold-refined 65 MeV model ($1\%\times10$, 40--300 MeV)",
        ),
        mpl.lines.Line2D(
            [], [], color=NEW_LANES["2021_10pct"]["substitution_color"],
            marker="*", markeredgecolor="black", markeredgewidth=0.5,
            linestyle="None", markersize=8,
            label="extended-support 65 MeV model (native 10%, 30--300 MeV)",
        ),
    ])
    labels.extend([handle.get_label() for handle in handles[-2:]])
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(
        "Background-only pull diagnostics across 100 pseudoexperiments\n"
        "Highlighted 65 MeV points use threshold-focused background models",
        fontsize=15,
        y=1.045,
    )
    return save_figure(fig, "figure64_spurious_signal_consolidated_full100_90cl")


def make_2x4(consolidated: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.0), sharex="col", constrained_layout=True)
    masses = [65.0, 90.0, 120.0, 180.0, 210.0]
    for col, scenario in enumerate(SCENARIOS):
        for strength in (0.0, 1.0, 3.0, 5.0):
            cell = consolidated[(consolidated.scenario == scenario) & np.isclose(consolidated.inj_nsigma, strength)]
            for row, (value, low, high) in enumerate((
                ("mean_pull", "mean_ci90_low", "mean_ci90_high"),
                ("sample_width", "width_ci90_low", "width_ci90_high"),
            )):
                ordinary = cell[~cell.is_65mev_substitution]
                axes[row, col].errorbar(
                    ordinary.mass_MeV,
                    ordinary[value],
                    yerr=np.vstack([ordinary[value] - ordinary[low], ordinary[high] - ordinary[value]]),
                    color=STRENGTH_COLORS[strength],
                    marker=STRENGTH_MARKERS[strength],
                    ms=3.8,
                    lw=1.15,
                    capsize=2,
                    label=rf"$z={int(strength)}$" if row == 0 and col == 0 else None,
                )
                sub = cell[cell.is_65mev_substitution]
                if not sub.empty:
                    item = sub.iloc[0]
                    substitution_marker = NEW_LANES[scenario]["marker"]
                    axes[row, col].errorbar(
                        [65.0], [item[value]],
                        yerr=[[item[value] - item[low]], [item[high] - item[value]]],
                        fmt=substitution_marker,
                        color=STRENGTH_COLORS[strength],
                        markeredgecolor=NEW_LANES[scenario]["substitution_color"],
                        markeredgewidth=1.15,
                        ms=8 if substitution_marker == "*" else 5.5,
                        capsize=2.5,
                        zorder=5,
                    )
        axes[0, col].axhline(0.0, color="black", lw=0.8)
        axes[0, col].axhspan(-0.5, 0.5, color="0.7", alpha=0.10)
        axes[1, col].axhline(1.0, color="black", lw=0.8, ls="--")
        axes[0, col].set_title(SCENARIO_LABELS[scenario])
        axes[1, col].set_xlabel("mass [MeV]")
        axes[1, col].set_xticks(masses)
    axes[0, 0].set_ylabel("pull mean (90% $t$ interval)")
    axes[1, 0].set_ylabel("sample pull width (90% normal-theory interval)")
    handles = [
        mpl.lines.Line2D([], [], color=STRENGTH_COLORS[z], marker=STRENGTH_MARKERS[z], label=rf"$z={int(z)}$")
        for z in (0.0, 1.0, 3.0, 5.0)
    ]
    handles.extend([
        mpl.lines.Line2D([], [], color="#7B2CBF", marker="D", linestyle="None", markersize=6, label=r"threshold-refined 65 MeV model ($1\%\times10$)"),
        mpl.lines.Line2D([], [], color="#D81B60", marker="*", linestyle="None", markersize=8, label="extended-support 65 MeV model (native 10%)"),
    ])
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Signal-injection pull diagnostics across 100 pseudoexperiments per point",
        fontsize=14,
        y=1.09,
    )
    return save_figure(fig, "pull_means_widths_consolidated_2x4_full100_90cl")


def running_mean_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    means, lows, highs = [], [], []
    for stop in range(2, len(values) + 1):
        stats = moment(values[:stop])
        means.append(stats["mean_pull"])
        lows.append(stats["mean_ci90_low"])
        highs.append(stats["mean_ci90_high"])
    return np.asarray(means), np.asarray(lows), np.asarray(highs)


def make_figure65(historical: pd.DataFrame, new: pd.DataFrame) -> list[Path]:
    old = historical[
        (historical.scenario == "2021_1pct_x10")
        & np.isclose(historical.mass_MeV, 65.0)
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    old_zero = old[np.isclose(old.inj_nsigma, 0.0)].sort_values("background_toy_index")
    new_zero = new[np.isclose(new.inj_nsigma, 0.0)].sort_values("background_toy_index")
    bins = np.linspace(
        min(old_zero.pull.min(), new_zero.pull.min()) - 0.2,
        max(old_zero.pull.max(), new_zero.pull.max()) + 0.2,
        24,
    )
    axes[0, 0].hist(old_zero.pull, bins=bins, density=True, histtype="step", lw=1.8, color="#777777", label="baseline smooth-threshold model")
    axes[0, 0].hist(new_zero.pull, bins=bins, density=True, histtype="stepfilled", alpha=0.28, lw=1.8, color="#7B2CBF", label="threshold-refined 65 MeV model")
    axes[0, 0].axvline(0.0, color="black", lw=0.9)
    axes[0, 0].set_xlabel("zero-signal pull")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_title("Pull distributions across 100 pseudoexperiments")
    axes[0, 0].legend(frameon=False)

    for frame, color, label in (
        (old_zero, "#777777", "baseline smooth-threshold model"),
        (new_zero, "#7B2CBF", "threshold-refined 65 MeV model"),
    ):
        vals = frame.pull.to_numpy()
        means, lows, highs = running_mean_interval(vals)
        # The first few Student-t intervals are mathematically valid but so
        # broad that they obscure the development/continuation comparison.
        means, lows, highs = means[8:], lows[8:], highs[8:]
        x = np.arange(10, len(vals) + 1)
        axes[0, 1].plot(x, means, color=color, lw=1.5, label=label)
        axes[0, 1].fill_between(x, lows, highs, color=color, alpha=0.16)
    axes[0, 1].axvline(
        20,
        color="#7B2CBF",
        ls=":",
        lw=1.1,
        label="development subset | independent continuation",
    )
    axes[0, 1].axhline(0.0, color="black", lw=0.9)
    axes[0, 1].axhspan(-0.2, 0.2, color="0.7", alpha=0.10)
    axes[0, 1].set_xlabel("number of pseudoexperiments")
    axes[0, 1].set_ylabel("running mean pull (90% interval)")
    axes[0, 1].set_title("Background-only stability")
    axes[0, 1].legend(frameon=False, fontsize=7)

    strengths = [0.0, 1.0, 3.0, 5.0]
    for frame, color, marker, label in (
        (old, "#777777", "o", "baseline smooth-threshold model"),
        (new, "#7B2CBF", "D", "threshold-refined 65 MeV model"),
    ):
        stats = [moment(frame[np.isclose(frame.inj_nsigma, z)].pull.to_numpy()) for z in strengths]
        means = np.array([v["mean_pull"] for v in stats])
        mlo = np.array([v["mean_ci90_low"] for v in stats])
        mhi = np.array([v["mean_ci90_high"] for v in stats])
        widths = np.array([v["sample_width"] for v in stats])
        wlo = np.array([v["width_ci90_low"] for v in stats])
        whi = np.array([v["width_ci90_high"] for v in stats])
        axes[1, 0].errorbar(strengths, means, yerr=np.vstack([means - mlo, mhi - means]), color=color, marker=marker, lw=1.4, capsize=3, label=label)
        axes[1, 1].errorbar(strengths, widths, yerr=np.vstack([widths - wlo, whi - widths]), color=color, marker=marker, lw=1.4, capsize=3, label=label)
    axes[1, 0].axhline(0.0, color="black", lw=0.9)
    axes[1, 0].axhspan(-0.2, 0.2, color="0.7", alpha=0.10)
    axes[1, 1].axhline(1.0, color="black", lw=0.9, ls="--")
    axes[1, 0].set_ylabel("mean pull (90% $t$ interval)")
    axes[1, 1].set_ylabel("sample pull width (90% normal-theory interval)")
    for ax in axes[1]:
        ax.set_xlabel("injected significance $z$")
        ax.set_xticks(strengths)
        ax.legend(frameon=False)
    axes[1, 0].set_title("Mean response at 65 MeV")
    axes[1, 1].set_title("Width response at 65 MeV")
    fig.suptitle(
        r"Threshold-model comparison for the 1%$\times10$ sample at 65 MeV",
        fontsize=14,
    )
    return save_figure(fig, FIGURE65_STEM)


def main() -> int:
    setup_style()
    historical, lanes = load_inputs()
    new_summary = summarize_new(lanes)
    historical_summary = summarize_historical(historical)
    consolidated = consolidated_rows(historical_summary, new_summary)
    DERIVED.mkdir(parents=True, exist_ok=True)
    new_summary.to_csv(
        DERIVED / "new_65mev_development_and_continuation_moments_90cl.csv",
        index=False,
    )
    historical_summary.to_csv(DERIVED / "baseline_moments_recomputed_90cl.csv", index=False)
    consolidated.to_csv(DERIVED / "consolidated_pull_moments_90cl.csv", index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(make_figure64(historical, lanes, consolidated))
    figure_paths.extend(make_2x4(consolidated))
    figure_paths.extend(make_figure65(historical, lanes["2021_1pct_x10"]))
    manifest = {
        "schema_version": 1,
        "study_id": "reader_facing_validation_figures",
        "intervals": "two-sided 90% Student-t intervals for means and exact normal-theory chi-square intervals for sample widths",
        "historical_source": {
            "path": os.path.relpath(HISTORICAL, ROOT),
            "sha256": sha256_file(HISTORICAL),
        },
        "substitutions": {
            scenario: {
                "accepted_ledger": os.path.relpath(record["path"], ROOT),
                "accepted_ledger_sha256": sha256_file(record["path"]),
                "study": record["study"],
                "mass_MeV": 65.0,
            }
            for scenario, record in NEW_LANES.items()
        },
        "figures": [
            {"path": os.path.relpath(path, ROOT), "sha256": sha256_file(path)}
            for path in figure_paths
        ],
        "tables": [
            {
                "path": os.path.relpath(path, ROOT),
                "sha256": sha256_file(path),
            }
            for path in (
                DERIVED / "new_65mev_development_and_continuation_moments_90cl.csv",
                DERIVED / "baseline_moments_recomputed_90cl.csv",
                DERIVED / "consolidated_pull_moments_90cl.csv",
            )
        ],
    }
    atomic_json(DERIVED / "analysis_figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
