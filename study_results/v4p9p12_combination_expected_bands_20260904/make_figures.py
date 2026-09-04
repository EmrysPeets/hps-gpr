#!/usr/bin/env python3
"""Make non-overlapping Brazil-band figures for a completed v4.9.12 stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURES = HERE / "figures"
DERIVED = HERE / "derived"
OBSERVED = (
    REPO
    / "study_results"
    / "v4p9p12_final_dataset_combinations_20260902"
    / "derived"
    / "final_dataset_result_curves.csv"
)
INDIVIDUAL = (
    "individual_2015_full",
    "individual_2016_full",
    "individual_2021_10pct",
)
COMBINATIONS = (
    "pair_2015_2016",
    "pair_2015_2021",
    "pair_2016_2021",
    "all_2015_2016_2021",
)
LABELS = {
    "individual_2015_full": "2015 full",
    "individual_2016_full": "2016 full",
    "individual_2021_10pct": "2021 10% (optimized support)",
    "pair_2015_2016": "2015 full + 2016 full",
    "pair_2015_2021": "2015 full + 2021 10%",
    "pair_2016_2021": "2016 full + 2021 10%",
    "all_2015_2016_2021": "2015 full + 2016 full + 2021 10%",
}
YELLOW = "#F6D66A"
GREEN = "#69C779"
PVALUE_STYLES = {
    "p_strong": ("#4C72B0", "-", r"$p_{\rm strong}$"),
    "p_weak": ("#DD8452", "-", r"$p_{\rm weak}$"),
    "p_two": ("#7B4EA3", "-", r"$p_{\rm two}$"),
    "p0_local_asymptotic": ("#111111", "--", r"analytic local $p_0$"),
}
TOTAL_WINDOW_SEGMENTS = (
    (19, 38, "individual_2015_full", "2015"),
    (39, 49, "pair_2015_2016", "15+16"),
    (50, 90, "all_2015_2016_2021", "all three"),
    (91, 180, "pair_2016_2021", "16+21"),
    (181, 250, "individual_2021_10pct", "2021"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
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


def handles() -> list[object]:
    return [
        Patch(facecolor=YELLOW, edgecolor="#D6A900", alpha=0.40, label=r"Central 95% expected"),
        Patch(facecolor=GREEN, edgecolor="#2D9B48", alpha=0.58, label=r"Central 68% expected"),
        Line2D([0], [0], color="black", lw=1.8, ls="--", label="Expected median"),
        Line2D([0], [0], color="#4C4C4C", lw=2.1, label="Observed 90% CL$_s$"),
    ]


def panel(ax: plt.Axes, frame: pd.DataFrame, scope: str) -> None:
    frame = frame.sort_values("mass_MeV")
    x = frame.mass_MeV.to_numpy(float)
    q025 = frame.expected_q025.to_numpy(float)
    q16 = frame.expected_q16.to_numpy(float)
    median = frame.expected_median.to_numpy(float)
    q84 = frame.expected_q84.to_numpy(float)
    q975 = frame.expected_q975.to_numpy(float)
    observed = frame.eps2_observed.to_numpy(float)
    ax.fill_between(
        x,
        q025,
        q975,
        color=YELLOW,
        edgecolor="#D6A900",
        linewidth=0.45,
        alpha=0.40,
        zorder=1,
    )
    ax.fill_between(
        x,
        q16,
        q84,
        color=GREEN,
        edgecolor="#2D9B48",
        linewidth=0.55,
        alpha=0.58,
        zorder=2,
    )
    ax.plot(x, median, color="black", lw=1.8, ls="--", zorder=3)
    ax.plot(x, observed, color="black", lw=2.15, zorder=4)
    ax.set_yscale("log")
    values = np.concatenate([q025, q975, median, observed])
    values = values[np.isfinite(values) & (values > 0.0)]
    log_values = np.log10(values)
    padding = max(0.08, 0.08 * float(np.ptp(log_values)))
    low = 10.0 ** (float(np.min(log_values)) - padding)
    high = 10.0 ** (float(np.max(log_values)) + padding)
    ax.set_ylim(low, high)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.margins(x=0.01)
    ax.set_title(LABELS[scope], loc="left", fontweight="semibold", pad=7)
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def build_pvalue_diagnostics(
    limits: pd.DataFrame,
    observed: pd.DataFrame,
    target_toys: int,
) -> pd.DataFrame:
    """Reproduce the established fixed-mass upper-limit tail diagnostics."""
    obs_columns = [
        "scope_key",
        "mass_MeV",
        "eps2_90",
        "p0_local_asymptotic",
        "Z_local_asymptotic",
        "pvalue_method",
    ]
    obs = observed[obs_columns].copy()
    if obs.duplicated(["scope_key", "mass_MeV"]).any():
        raise RuntimeError("observed source has duplicate scope--mass rows")
    merged = limits.merge(
        obs[["scope_key", "mass_MeV", "eps2_90"]],
        on=["scope_key", "mass_MeV"],
        how="left",
        validate="many_to_one",
    )
    if merged.eps2_90_y.isna().any():
        raise RuntimeError("toy rows do not all have an observed-limit match")
    merged["strong_indicator"] = (
        merged.eps2_90_x.to_numpy(float) <= merged.eps2_90_y.to_numpy(float)
    )
    merged["weak_indicator"] = (
        merged.eps2_90_x.to_numpy(float) >= merged.eps2_90_y.to_numpy(float)
    )
    diagnostics = (
        merged.groupby(["scope_key", "mass_MeV"], sort=True, as_index=False)
        .agg(
            scope_label=("scope_label", "first"),
            dataset_set=("dataset_set", "first"),
            n_toys=("toy_id", "count"),
            toy_id_min=("toy_id", "min"),
            toy_id_max=("toy_id", "max"),
            eps2_observed=("eps2_90_y", "first"),
            n_strong=("strong_indicator", "sum"),
            n_weak=("weak_indicator", "sum"),
        )
        .merge(
            obs.drop(columns="eps2_90"),
            on=["scope_key", "mass_MeV"],
            how="left",
            validate="one_to_one",
        )
    )
    diagnostics["n_toys"] = diagnostics.n_toys.astype(int)
    diagnostics["n_strong"] = diagnostics.n_strong.astype(int)
    diagnostics["n_weak"] = diagnostics.n_weak.astype(int)
    diagnostics["p_strong"] = diagnostics.n_strong / diagnostics.n_toys
    diagnostics["p_weak"] = diagnostics.n_weak / diagnostics.n_toys
    diagnostics["p_two"] = np.clip(
        2.0 * np.minimum(diagnostics.p_strong, diagnostics.p_weak), 0.0, 1.0
    )
    diagnostics["empirical_p_resolution"] = 1.0 / diagnostics.n_toys
    diagnostics["diagnostic_definition"] = (
        "p_strong=Pr(UL_toy<=UL_obs); p_weak=Pr(UL_toy>=UL_obs); "
        "p_two=min(1,2*min(p_strong,p_weak))"
    )
    if set(diagnostics.n_toys) != {target_toys}:
        raise RuntimeError("p-value diagnostics do not contain the requested toy count")
    if not np.isfinite(
        diagnostics[
            ["p_strong", "p_weak", "p_two", "p0_local_asymptotic"]
        ].to_numpy(float)
    ).all():
        raise RuntimeError("p-value diagnostics contain non-finite values")
    return diagnostics.sort_values(["scope_key", "mass_MeV"]).reset_index(drop=True)


def build_total_window(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Stitch the maximal available final-dataset scope over 19--250 MeV."""
    pieces = []
    for low, high, scope, short_label in TOTAL_WINDOW_SEGMENTS:
        piece = summary[
            (summary.scope_key == scope)
            & summary.mass_MeV.between(low, high, inclusive="both")
        ].copy()
        if not np.array_equal(
            piece.mass_MeV.to_numpy(int), np.arange(low, high + 1)
        ):
            raise RuntimeError(f"total-window source segment is incomplete: {scope}")
        piece["selected_scope_key"] = scope
        piece["active_scope_short_label"] = short_label
        pieces.append(piece)
    total = pd.concat(pieces, ignore_index=True)
    if not np.array_equal(total.mass_MeV.to_numpy(int), np.arange(19, 251)):
        raise RuntimeError("stitched total-search-window grid is not exactly 19--250 MeV")
    total = total.merge(
        diagnostics.drop(
            columns=[
                "scope_label",
                "dataset_set",
                "n_toys",
                "toy_id_min",
                "toy_id_max",
                "eps2_observed",
            ]
        ),
        on=["scope_key", "mass_MeV"],
        how="left",
        validate="one_to_one",
    )
    total["construction"] = "maximal_available_final_dataset_scope_at_each_mass"
    return total


def single_scope(summary: pd.DataFrame, scope: str, target_toys: int) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    panel(ax, summary[summary.scope_key == scope], scope)
    ax.set_ylabel(r"90% CL$_s$ upper limit on $\epsilon^2$")
    fig.legend(
        handles=handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        fontsize=9.3,
    )
    ax.set_title(LABELS[scope], loc="left", fontweight="semibold", pad=38)
    fig.text(
        0.5,
        0.018,
        (
            f"{target_toys} toys per mass; pointwise, background-only, and conditional "
            "on frozen GP states."
        ),
        ha="center",
        fontsize=8.4,
        color="0.35",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.16)
    save(fig, f"all_three_expected_bands_{target_toys}toys")


def panel_grid(
    summary: pd.DataFrame,
    scopes: tuple[str, ...],
    *,
    target_toys: int,
    stem: str,
    title: str,
    shape: tuple[int, int],
) -> None:
    fig, axes = plt.subplots(*shape, figsize=(12.8, 8.6) if shape[0] == 2 else (15.2, 5.4))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax, scope in zip(axes_array, scopes):
        panel(ax, summary[summary.scope_key == scope], scope)
    for ax in axes_array[len(scopes):]:
        ax.set_visible(False)
    fig.supylabel(r"90% CL$_s$ upper limit on $\epsilon^2$", x=0.018)
    fig.suptitle(title, x=0.5, y=0.99, ha="center", fontweight="semibold", fontsize=14)
    fig.legend(
        handles=handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=9.1,
    )
    fig.text(
        0.5,
        0.012,
        (
            f"{target_toys} toys per mass. Outer quantiles are provisional at this stage; "
            "bands are pointwise and conditional on frozen GP states."
        ),
        ha="center",
        fontsize=8.3,
        color="0.35",
    )
    if shape[0] == 2:
        fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.10, hspace=0.31, wspace=0.20)
    else:
        fig.subplots_adjust(left=0.06, right=0.985, top=0.76, bottom=0.17, wspace=0.22)
    save(fig, f"{stem}_{target_toys}toys")


def total_window_plot(total: pd.DataFrame, target_toys: int) -> None:
    fig = plt.figure(figsize=(11.4, 6.8))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.12, 1.0),
        hspace=0.055,
        left=0.105,
        right=0.985,
        bottom=0.145,
        top=0.815,
    )
    strip = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=strip)
    strip_colors = ("#DDEAF3", "#D8E7DD", "#E8E0F0", "#F2E5D5", "#E5E5E5")
    for (low, high, scope, short_label), color in zip(
        TOTAL_WINDOW_SEGMENTS, strip_colors
    ):
        strip.add_patch(
            Rectangle(
                (low - 0.5, 0.0),
                high - low + 1.0,
                1.0,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        strip.text(
            0.5 * (low + high),
            0.5,
            short_label,
            ha="center",
            va="center",
            fontsize=7.2 if high - low < 20 else 8.2,
            fontweight="semibold",
            color="0.18",
        )
        frame = total[total.selected_scope_key == scope].sort_values("mass_MeV")
        x = frame.mass_MeV.to_numpy(float)
        ax.fill_between(
            x,
            frame.expected_q025.to_numpy(float),
            frame.expected_q975.to_numpy(float),
            color=YELLOW,
            edgecolor="#D6A900",
            linewidth=0.45,
            alpha=0.40,
            zorder=1,
        )
        ax.fill_between(
            x,
            frame.expected_q16.to_numpy(float),
            frame.expected_q84.to_numpy(float),
            color=GREEN,
            edgecolor="#2D9B48",
            linewidth=0.55,
            alpha=0.58,
            zorder=2,
        )
        ax.plot(
            x,
            frame.expected_median.to_numpy(float),
            color="black",
            lw=1.7,
            ls="--",
            zorder=3,
        )
        ax.plot(
            x,
            frame.eps2_observed.to_numpy(float),
            color="black",
            lw=2.05,
            zorder=4,
        )
    for boundary in (38.5, 49.5, 90.5, 180.5):
        ax.axvline(boundary, color="0.38", lw=0.75, ls=":", zorder=0)
    strip.set_ylim(0.0, 1.0)
    strip.set_xlim(18.5, 250.5)
    strip.set_ylabel("active\nscope", rotation=0, ha="right", va="center", fontsize=7.5)
    strip.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in strip.spines.values():
        spine.set_visible(False)
    ax.set_yscale("log")
    values = total[
        [
            "expected_q025",
            "expected_q975",
            "expected_median",
            "eps2_observed",
        ]
    ].to_numpy(float)
    values = values[np.isfinite(values) & (values > 0.0)]
    log_values = np.log10(values)
    padding = max(0.08, 0.08 * float(np.ptp(log_values)))
    ax.set_ylim(
        10.0 ** (float(np.min(log_values)) - padding),
        10.0 ** (float(np.max(log_values)) + padding),
    )
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax.set_ylabel(r"90% CL$_s$ upper limit on $\epsilon^2$")
    fig.suptitle(
        "Final observed limit with expected bands over the total search window",
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )
    fig.legend(
        handles=handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        fontsize=9.3,
    )
    fig.text(
        0.5,
        0.025,
        (
            f"{target_toys} toys per mass; pointwise and conditional. The active "
            "dataset composition changes only at the marked boundaries."
        ),
        ha="center",
        fontsize=8.4,
        color="0.35",
    )
    save(fig, f"final_total_search_window_expected_bands_{target_toys}toys")


def pvalue_panel(
    ax: plt.Axes,
    diagnostics: pd.DataFrame,
    scope: str,
    target_toys: int,
    y_min: float,
) -> None:
    frame = diagnostics[diagnostics.scope_key == scope].sort_values("mass_MeV")
    x = frame.mass_MeV.to_numpy(float)
    zero_display = 0.5 / float(target_toys)
    for column, (color, linestyle, label) in PVALUE_STYLES.items():
        raw = frame[column].to_numpy(float)
        shown = np.where(raw == 0.0, zero_display, raw)
        ax.plot(x, shown, color=color, ls=linestyle, lw=1.65, label=label)
        if column != "p0_local_asymptotic":
            zeros = raw == 0.0
            if zeros.any():
                ax.scatter(
                    x[zeros],
                    np.full(np.count_nonzero(zeros), zero_display),
                    color=color,
                    marker="v",
                    s=18,
                    zorder=5,
                )
    ax.axhline(0.05, color="0.45", lw=0.85, ls=":", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(y_min, 1.05)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_title(LABELS[scope], loc="left", fontweight="semibold", pad=7)
    ax.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")


def pvalue_grid(diagnostics: pd.DataFrame, target_toys: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.7))
    combination_values = diagnostics[
        diagnostics.scope_key.isin(COMBINATIONS)
    ][list(PVALUE_STYLES)].to_numpy(float)
    positive = combination_values[
        np.isfinite(combination_values) & (combination_values > 0.0)
    ]
    y_min = min(0.5 / float(target_toys), float(np.min(positive))) * 0.72
    for ax, scope in zip(axes.reshape(-1), COMBINATIONS):
        pvalue_panel(ax, diagnostics, scope, target_toys, y_min)
    fig.supylabel("fixed-mass p-value or limit-tail fraction", x=0.018)
    fig.suptitle(
        "Combination limit-tail diagnostics and analytic local discovery p-value",
        x=0.5,
        y=0.992,
        ha="center",
        fontweight="semibold",
        fontsize=14,
    )
    pvalue_handles = [
        Line2D([0], [0], color=color, lw=1.8, ls=linestyle, label=label)
        for color, linestyle, label in PVALUE_STYLES.values()
    ]
    fig.legend(
        handles=pvalue_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=9.2,
    )
    fig.text(
        0.5,
        0.012,
        (
            f"One-sided empirical fractions use {target_toys} toys per mass (resolution "
            f"{1.0 / target_toys:.3f}); a zero count is drawn as a downward triangle "
            f"at {0.5 / target_toys:.3f}. The dotted line marks 0.05."
        ),
        ha="center",
        fontsize=8.2,
        color="0.35",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.84,
        bottom=0.105,
        hspace=0.31,
        wspace=0.19,
    )
    save(fig, f"combination_pvalue_panels_{target_toys}toys")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    source = HERE / "derived" / f"expected_band_summary_{target_toys}toys.csv"
    if not source.is_file():
        raise SystemExit(f"missing completed stage summary: {source}")
    limits_source = HERE / "derived" / f"toy_limits_{target_toys}toys.csv"
    if not limits_source.is_file() or not OBSERVED.is_file():
        raise SystemExit("completed toy ledger and frozen observed ledger are required")
    summary = pd.read_csv(source)
    limits = pd.read_csv(limits_source)
    observed = pd.read_csv(OBSERVED)
    expected_scopes = set(INDIVIDUAL + COMBINATIONS)
    if set(summary.scope_key.astype(str)) != expected_scopes:
        raise RuntimeError("summary does not contain exactly the seven final scopes")
    if set(summary.n_toys.astype(int)) != {target_toys}:
        raise RuntimeError("summary toy count does not match the requested figure stage")
    if not (
        summary.loc[
            summary.scope_key == "individual_2021_10pct", "scope_label"
        ]
        .astype(str)
        .str.contains("2021 10%", regex=False)
        .all()
    ):
        raise RuntimeError("optimized 2021 10% scope is missing")

    diagnostics = build_pvalue_diagnostics(limits, observed, target_toys)
    diagnostics_path = DERIVED / f"pvalue_diagnostics_{target_toys}toys.csv"
    diagnostics.to_csv(diagnostics_path, index=False, float_format="%.17g")
    total = build_total_window(summary, diagnostics)
    total_path = DERIVED / f"final_total_search_window_summary_{target_toys}toys.csv"
    total.to_csv(total_path, index=False, float_format="%.17g")

    style()
    single_scope(summary, "all_2015_2016_2021", target_toys)
    panel_grid(
        summary,
        INDIVIDUAL,
        target_toys=target_toys,
        stem="individual_expected_band_panels",
        title="Standalone final-sample expected bands",
        shape=(1, 3),
    )
    panel_grid(
        summary,
        COMBINATIONS,
        target_toys=target_toys,
        stem="combination_expected_band_panels",
        title=r"Shared-$\epsilon^2$ combination expected bands",
        shape=(2, 2),
    )
    total_window_plot(total, target_toys)
    pvalue_grid(diagnostics, target_toys)
    inventory = {
        "stage_toys_per_mass": target_toys,
        "source_summary": str(source.relative_to(REPO)),
        "source_summary_sha256": sha256(source),
        "source_toy_limits": str(limits_source.relative_to(REPO)),
        "source_toy_limits_sha256": sha256(limits_source),
        "source_observed": str(OBSERVED.relative_to(REPO)),
        "source_observed_sha256": sha256(OBSERVED),
        "pvalue_diagnostics": str(diagnostics_path.relative_to(REPO)),
        "pvalue_diagnostics_sha256": sha256(diagnostics_path),
        "total_search_window_summary": str(total_path.relative_to(REPO)),
        "total_search_window_summary_sha256": sha256(total_path),
        "figures": [
            f"all_three_expected_bands_{target_toys}toys",
            f"individual_expected_band_panels_{target_toys}toys",
            f"combination_expected_band_panels_{target_toys}toys",
            f"final_total_search_window_expected_bands_{target_toys}toys",
            f"combination_pvalue_panels_{target_toys}toys",
        ],
        "style_reference": (
            "v4.2/v4.5 Brazil-band convention: yellow central 95%, green central "
            "68%, dashed black median, solid observed curve"
        ),
        "layout": "one curve family per axis; figure-level legends outside data regions",
        "total_search_window_rule": [
            {
                "mass_min_MeV": low,
                "mass_max_MeV": high,
                "scope_key": scope,
            }
            for low, high, scope, _ in TOTAL_WINDOW_SEGMENTS
        ],
        "pvalue_diagnostics_definition": (
            "p_strong=Pr(UL_toy<=UL_obs); p_weak=Pr(UL_toy>=UL_obs); "
            "p_two=min(1,2*min(p_strong,p_weak)); analytic p0 is the frozen "
            "one-sided fixed-mass asymptotic profile-LRT discovery p-value"
        ),
        "claim_boundary": (
            "Conditional pointwise expected-limit quantiles and upper-limit tail "
            "diagnostics; not coverage or scan-global calibration. Analytic p0 is local."
        ),
    }
    (FIGURES / f"figure_manifest_{target_toys}toys.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
