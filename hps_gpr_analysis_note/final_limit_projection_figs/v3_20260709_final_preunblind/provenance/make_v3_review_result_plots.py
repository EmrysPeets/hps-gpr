from __future__ import annotations

"""Build the v3 review-committee result figures from the archived CSVs.

The script deliberately keeps the source tables immutable.  Isolated limit values
associated with documented optimizer failures are repaired only in derived review
tables, by linear interpolation in log(limit) between the adjacent evaluated mass
hypotheses.  Every replacement is written to ``v3_display_repair_audit.csv``.
"""

import ast
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent

SINGLE_SOURCE = HERE / "raw_combined_single.csv"
COMBINED_BANDS_SOURCE = HERE / "combined_ul_bands_combined_all_dimuon_for_plotting.csv"
SOURCE_2021 = HERE / "combined_ul_bands_2021_dimuon_for_plotting.csv"
CONFIG_SOURCE = HERE / "config_obsUL90_combined_finalpass_search50_countscale_bands2021_combined.yaml"

REVIEW_COMBINED = HERE / "combined_ul_bands_combined_all_v3_review.csv"
REVIEW_2021 = HERE / "combined_ul_bands_2021_v3_review.csv"
REVIEW_INDIVIDUAL = HERE / "individual_observed_limits_v3_review.csv"
REPAIR_AUDIT = HERE / "v3_display_repair_audit.csv"
P_VALUE_SUMMARY = HERE / "v3_pvalue_summary.csv"
TAIL_PVALUE_EXCLUSIONS = HERE / "v3_tail_pvalue_exclusions.csv"

M_DIMUON_GEV = 0.211316749
N_LIMIT_TOYS = 10_000
INDEPENDENCE_WIDTH_SIGMA = 2.25

COLORS = {
    "2015": "#4C72B0",
    "2016": "#DD8452",
    "2021": "#55A868",
    "combined": "#111111",
    "global": "#B0442E",
}

# These are the isolated scan rows documented as failed numerical optimizations.
# The mass grid is 1 MeV, so the adjacent evaluated hypotheses are unambiguous.
INDIVIDUAL_LIMIT_REPAIRS_MEV = {
    "2015": (24,),
    "2016": (44, 98, 133, 171),
}
COMBINED_LIMIT_REPAIRS_MEV = (136,)
SOURCE_2021_LIMIT_REPAIRS_MEV = (136,)
TAIL_PVALUE_SUPERSEDED_MASSES_MEV = (24, 37, 44, 59, 91, 98, 115, 133, 136, 159, 166)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 11.2,
            "axes.titlesize": 14.0,
            "axes.labelsize": 12.0,
            "legend.fontsize": 9.1,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.0,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#d7d7d7", linewidth=0.75, alpha=0.85)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.72)
    ax.tick_params(axis="both", which="major", width=1.0, length=5.0)
    ax.tick_params(axis="both", which="minor", width=0.75, length=2.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig: plt.Figure, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def dimuon_factor(mass_gev: np.ndarray | pd.Series) -> np.ndarray:
    mass = np.asarray(mass_gev, dtype=float)
    factor = np.ones_like(mass)
    above = mass > M_DIMUON_GEV
    if np.any(above):
        x = (M_DIMUON_GEV / mass[above]) ** 2
        factor[above] = 1.0 + np.sqrt(np.clip(1.0 - x, 0.0, None)) * (1.0 + 0.5 * x)
    return factor


def neighbor_repair(
    frame: pd.DataFrame,
    *,
    mass_mev: int,
    column: str,
    scope: str,
    dataset: str,
    reason: str,
    records: list[dict[str, object]],
) -> float:
    """Replace one value with the geometric mean of adjacent mass rows."""

    work = frame.sort_values("mass_GeV")
    masses = np.rint(1000.0 * work["mass_GeV"].to_numpy(float)).astype(int)
    positions = np.flatnonzero(masses == int(mass_mev))
    if len(positions) != 1:
        raise RuntimeError(f"expected one {dataset} row at {mass_mev} MeV, found {len(positions)}")
    pos = int(positions[0])
    if pos == 0 or pos == len(work) - 1:
        raise RuntimeError(f"cannot neighbor-repair endpoint {mass_mev} MeV")

    left = work.iloc[pos - 1]
    current = work.iloc[pos]
    right = work.iloc[pos + 1]
    left_mass = int(round(1000.0 * float(left["mass_GeV"])))
    right_mass = int(round(1000.0 * float(right["mass_GeV"])))
    if left_mass != mass_mev - 1 or right_mass != mass_mev + 1:
        raise RuntimeError(
            f"{dataset} {mass_mev} MeV is not bracketed by adjacent 1 MeV rows: "
            f"{left_mass}, {right_mass}"
        )

    raw = float(current[column])
    left_value = float(left[column])
    right_value = float(right[column])
    if min(raw, left_value, right_value) <= 0.0:
        raise RuntimeError(f"non-positive value in {dataset} {mass_mev} MeV repair for {column}")
    corrected = math.sqrt(left_value * right_value)
    index = current.name
    frame.loc[index, column] = corrected
    records.append(
        {
            "scope": scope,
            "dataset": dataset,
            "mass_MeV": mass_mev,
            "column": column,
            "raw_value": raw,
            "left_mass_MeV": left_mass,
            "left_value": left_value,
            "right_mass_MeV": right_mass,
            "right_value": right_value,
            "corrected_value": corrected,
            "raw_over_corrected": raw / corrected,
            "method": "linear interpolation in log(value) between adjacent evaluated masses",
            "reason": reason,
        }
    )
    return corrected


def make_review_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    single_raw = pd.read_csv(SINGLE_SOURCE)
    single_raw["dataset"] = single_raw["dataset"].astype(str)
    single_raw["mass_MeV"] = 1000.0 * single_raw["mass_GeV"].astype(float)

    individual_parts: list[pd.DataFrame] = []
    for dataset in ("2015", "2016"):
        part = single_raw[single_raw["dataset"] == dataset][
            ["dataset", "mass_GeV", "mass_MeV", "eps2_up", "p0_analytic", "Z_analytic"]
        ].copy()
        part = part.rename(columns={"eps2_up": "eps2_observed_raw"})
        part["eps2_observed_v3_review"] = part["eps2_observed_raw"]
        for mass_mev in INDIVIDUAL_LIMIT_REPAIRS_MEV[dataset]:
            neighbor_repair(
                part,
                mass_mev=mass_mev,
                column="eps2_observed_v3_review",
                scope="individual observed limit",
                dataset=dataset,
                reason="documented isolated optimizer failure in the observed scan",
                records=records,
            )
        part["dimuon_factor"] = dimuon_factor(part["mass_GeV"])
        part["eps2_observed_dimuon_v3_review"] = (
            part["eps2_observed_v3_review"] * part["dimuon_factor"]
        )
        individual_parts.append(part)

    review_2021 = pd.read_csv(SOURCE_2021).sort_values("mass_GeV").copy()
    if "mass_MeV" not in review_2021:
        review_2021["mass_MeV"] = 1000.0 * review_2021["mass_GeV"].astype(float)
    # The previously reconciled 2021 table is already smooth at this point.  We
    # nevertheless use the same explicit neighbor rule so every 136 MeV result
    # panel is based on one documented display value.
    for mass_mev in SOURCE_2021_LIMIT_REPAIRS_MEV:
        for column in (
            "eps2_obs",
            "ul_eps2_obs",
            "eps2_obs_ee_channel",
            "eps2_obs_dimuon",
            "ul_eps2_obs_ee_channel",
            "ul_eps2_obs_dimuon",
            "A_obs",
            "ul_A_obs",
        ):
            if column in review_2021.columns:
                neighbor_repair(
                    review_2021,
                    mass_mev=mass_mev,
                    column=column,
                    scope="2021 observed limit",
                    dataset="2021",
                    reason="consistent 136 MeV display repair across v3 result figures",
                    records=records,
                )
    review_2021.to_csv(REVIEW_2021, index=False)

    part_2021 = review_2021[
        ["mass_GeV", "mass_MeV", "eps2_obs_dimuon", "p0_analytic", "Z_analytic"]
    ].copy()
    part_2021["dataset"] = "2021"
    part_2021["eps2_observed_raw"] = np.nan
    part_2021["eps2_observed_v3_review"] = part_2021["eps2_obs_dimuon"]
    part_2021["dimuon_factor"] = dimuon_factor(part_2021["mass_GeV"])
    part_2021["eps2_observed_dimuon_v3_review"] = part_2021["eps2_obs_dimuon"]
    part_2021 = part_2021.drop(columns="eps2_obs_dimuon")
    individual_parts.append(part_2021)

    individual = pd.concat(individual_parts, ignore_index=True, sort=False)
    individual = individual.sort_values(["dataset", "mass_GeV"])
    individual.to_csv(REVIEW_INDIVIDUAL, index=False)

    combined = pd.read_csv(COMBINED_BANDS_SOURCE).sort_values("mass_GeV").copy()
    if "mass_MeV" not in combined:
        combined["mass_MeV"] = 1000.0 * combined["mass_GeV"].astype(float)
    for mass_mev in COMBINED_LIMIT_REPAIRS_MEV:
        for column in (
            "eps2_obs",
            "ul_eps2_obs",
            "eps2_obs_ee_channel",
            "eps2_obs_dimuon",
            "ul_eps2_obs_ee_channel",
            "ul_eps2_obs_dimuon",
        ):
            if column in combined.columns:
                neighbor_repair(
                    combined,
                    mass_mev=mass_mev,
                    column=column,
                    scope="simultaneous observed limit",
                    dataset="2015+2016+2021 active combination",
                    reason="documented 136 MeV combined-limit numerical failure",
                    records=records,
                )

    # At 24 MeV only the 2015 campaign is active.  The simultaneous coordinate
    # must therefore use exactly the same repaired limit as the 2015-only curve;
    # otherwise the comparison ratio would falsely suggest a combination gain
    # or loss where no combination is being performed.
    repaired_2015_24 = individual[
        (individual["dataset"] == "2015") & np.isclose(individual["mass_MeV"], 24.0)
    ]
    combined_24 = combined[np.isclose(combined["mass_MeV"], 24.0)]
    if len(repaired_2015_24) != 1 or len(combined_24) != 1:
        raise RuntimeError("could not identify unique 2015 and simultaneous 24 MeV rows")
    propagated = float(repaired_2015_24["eps2_observed_dimuon_v3_review"].iloc[0])
    combined_index = combined_24.index[0]
    for column in (
        "eps2_obs",
        "ul_eps2_obs",
        "eps2_obs_ee_channel",
        "eps2_obs_dimuon",
        "ul_eps2_obs_ee_channel",
        "ul_eps2_obs_dimuon",
    ):
        if column not in combined.columns:
            continue
        raw = float(combined.loc[combined_index, column])
        combined.loc[combined_index, column] = propagated
        records.append(
            {
                "scope": "single-active simultaneous consistency",
                "dataset": "2015",
                "mass_MeV": 24,
                "column": column,
                "raw_value": raw,
                "left_mass_MeV": np.nan,
                "left_value": np.nan,
                "right_mass_MeV": np.nan,
                "right_value": np.nan,
                "corrected_value": propagated,
                "raw_over_corrected": raw / propagated,
                "method": "propagate the repaired 2015 value to the single-active simultaneous row",
                "reason": "the simultaneous and individual limits must agree when only 2015 is active",
            }
        )
    combined.to_csv(REVIEW_COMBINED, index=False)

    audit = pd.DataFrame(records).sort_values(["mass_MeV", "scope", "column"])
    audit.to_csv(REPAIR_AUDIT, index=False)
    return single_raw, individual, review_2021, combined


def plot_2021_limits(review_2021: pd.DataFrame) -> None:
    work = review_2021.sort_values("mass_GeV")
    x = 1000.0 * work["mass_GeV"].to_numpy(float)
    plots = [
        (
            "A_lo2",
            "A_lo1",
            "A_med",
            "A_hi1",
            "A_hi2",
            "ul_A_obs",
            "Signal-yield upper limit",
            "2021 signal-yield limit",
            "2021_UL_sig_yield_bands",
        ),
        (
            "eps2_lo2_dimuon",
            "eps2_lo1_dimuon",
            "eps2_med_dimuon",
            "eps2_hi1_dimuon",
            "eps2_hi2_dimuon",
            "eps2_obs_dimuon",
            r"90% CL upper limit on $\epsilon^2$",
            r"2021 limit on $\epsilon^2$",
            "2021_UL_eps2_bands",
        ),
    ]
    for lo2, lo1, med, hi1, hi2, obs, ylabel, title, stem in plots:
        fig, ax = plt.subplots(figsize=(10.9, 5.7))
        ax.fill_between(x, work[lo2], work[hi2], color="#F5C542", alpha=0.28, label=r"95% expected interval")
        ax.fill_between(x, work[lo1], work[hi1], color="#3CB44B", alpha=0.38, label=r"68% expected interval")
        ax.plot(x, work[med], color="#333333", linestyle="--", linewidth=2.0, label="Expected median")
        ax.plot(x, work[obs], color="#000000", linewidth=2.45, label="Observed")
        ax.set_yscale("log")
        ax.set_xlim(49.0, 251.0)
        ax.set_xlabel("Mass hypothesis (MeV)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#c9c9c9")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70))
        ax.yaxis.set_minor_formatter(NullFormatter())
        style_axis(ax)
        save(fig, stem)


def plot_combined_bands(combined: pd.DataFrame) -> None:
    work = combined.sort_values("mass_GeV")
    x = work["mass_MeV"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.fill_between(x, work["eps2_lo2_dimuon"], work["eps2_hi2_dimuon"], color="#F5C542", alpha=0.28, label="95% expected interval")
    ax.fill_between(x, work["eps2_lo1_dimuon"], work["eps2_hi1_dimuon"], color="#3CB44B", alpha=0.38, label="68% expected interval")
    ax.plot(x, work["eps2_med_dimuon"], color="#333333", linestyle="--", linewidth=2.0, label="Expected median")
    ax.plot(x, work["eps2_obs_dimuon"], color="#000000", linewidth=2.55, label="Observed")
    ax.set_yscale("log")
    ax.set_xlim(18.0, 252.0)
    ax.set_xlabel("Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("Simultaneous 2015, 2016, and 2021 limit")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#c9c9c9")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    save(fig, "combined_90cls_dimuon_corrected_eps2")


def comparison_table(individual: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    lookup: dict[tuple[str, int], float] = {}
    for row in individual.itertuples(index=False):
        key = (str(row.dataset), int(round(float(row.mass_MeV))))
        lookup[key] = float(row.eps2_observed_dimuon_v3_review)

    records: list[dict[str, object]] = []
    for row in combined.itertuples(index=False):
        mass_mev = int(round(float(row.mass_MeV)))
        active = [item.strip() for item in str(row.dataset_set).split("+") if item.strip()]
        values = {dataset: lookup[(dataset, mass_mev)] for dataset in active}
        best_dataset = min(values, key=values.get)
        best = values[best_dataset]
        comb = float(row.eps2_obs_dimuon)
        records.append(
            {
                "mass_GeV": float(row.mass_GeV),
                "mass_MeV": float(row.mass_MeV),
                "dataset_set": str(row.dataset_set),
                **{f"eps2_observed_{key}": lookup.get((key, mass_mev), np.nan) for key in ("2015", "2016", "2021")},
                "best_active_individual_dataset": best_dataset,
                "best_active_individual_eps2": best,
                "combined_observed_eps2": comb,
                "combined_over_best_active_individual_epsilon": math.sqrt(comb / best),
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(OUTDIR / "observed_individual_vs_combined_90cl_eps2_epsilon_ratio.csv", index=False)
    return out


def plot_comparison(individual: pd.DataFrame, combined: pd.DataFrame) -> None:
    table = comparison_table(individual, combined)
    fig, (top, ratio) = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.08},
    )
    for dataset, label in (("2015", "2015"), ("2016", "2016 10%"), ("2021", "2021 1%")):
        part = individual[individual["dataset"] == dataset].sort_values("mass_MeV")
        top.plot(
            part["mass_MeV"],
            part["eps2_observed_dimuon_v3_review"],
            color=COLORS[dataset],
            linewidth=1.45,
            marker="o",
            markersize=2.0,
            label=label,
        )
    top.plot(
        table["mass_MeV"],
        table["combined_observed_eps2"],
        color=COLORS["combined"],
        linewidth=2.5,
        label="Simultaneous fit",
    )
    top.set_yscale("log")
    top.set_xlim(18.0, 252.0)
    top.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    top.set_title("Observed limits from the individual campaigns and simultaneous fit")
    top.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#c9c9c9", ncol=2)
    top.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70))
    top.yaxis.set_minor_formatter(NullFormatter())
    style_axis(top)

    ratio.plot(
        table["mass_MeV"],
        table["combined_over_best_active_individual_epsilon"],
        color=COLORS["combined"],
        linewidth=1.75,
    )
    ratio.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ratio.set_xlabel("Mass hypothesis (MeV)")
    ratio.set_ylabel(r"$\epsilon_{\rm UL}^{\rm comb}/\epsilon_{\rm UL}^{\rm best}$")
    ratio.xaxis.set_minor_locator(AutoMinorLocator(5))
    ratio.yaxis.set_minor_locator(AutoMinorLocator(2))
    style_axis(ratio)
    save(fig, "observed_individual_vs_combined_90cl_eps2_epsilon_ratio")


def effective_trials(frame: pd.DataFrame, sigma_column: str) -> float:
    work = frame.sort_values("mass_GeV")
    mass = work["mass_GeV"].to_numpy(float)
    sigma = work[sigma_column].to_numpy(float)
    dm = np.diff(mass)
    sigma_mid = 0.5 * (sigma[:-1] + sigma[1:])
    valid = np.isfinite(dm) & (dm > 0.0) & np.isfinite(sigma_mid) & (sigma_mid > 0.0)
    estimate = np.sum(dm[valid] / (INDEPENDENCE_WIDTH_SIGMA * sigma_mid[valid]))
    return float(np.clip(estimate, 1.0, len(work)))


def sidak_global(local_p: np.ndarray | float, n_eff: float) -> np.ndarray:
    local = np.clip(np.asarray(local_p, dtype=float), 0.0, 1.0)
    return np.clip(-np.expm1(n_eff * np.log1p(-local)), 0.0, 1.0)


def pvalue_summary_row(kind: str, label: str, work: pd.DataFrame, n_eff: float | None = None) -> dict[str, object]:
    index = work["p_value"].astype(float).idxmin()
    row = work.loc[index]
    local = float(row["p_value"])
    result: dict[str, object] = {
        "kind": kind,
        "label": label,
        "mass_MeV": 1000.0 * float(row["mass_GeV"]),
        "minimum_p_value": local,
    }
    if n_eff is not None:
        result["N_eff"] = n_eff
        result["sidak_equivalent_global_p_value"] = float(sidak_global(local, n_eff))
        result["local_Z"] = NormalDist().inv_cdf(1.0 - local)
    return result


def plot_combined_p0(combined: pd.DataFrame, summaries: list[dict[str, object]]) -> None:
    work = combined.sort_values("mass_GeV").copy()
    n_eff = effective_trials(work, "sigma_mass_res_min_GeV")
    x = work["mass_MeV"].to_numpy(float)
    local = np.clip(work["p0_analytic"].to_numpy(float), 1.0e-12, 1.0)
    global_p = sidak_global(local, n_eff)

    fig, ax = plt.subplots(figsize=(10.9, 5.25))
    ax.plot(x, local, color=COLORS["combined"], linewidth=2.1, marker="o", markersize=1.8, label=r"Local asymptotic $p_0$")
    ax.plot(x, global_p, color=COLORS["global"], linewidth=1.9, linestyle="--", label=rf"Sidak-equivalent global $p$ ($N_{{\rm eff}}={n_eff:.1f}$)")
    for z in (1.0, 2.0, 3.0):
        p = 1.0 - NormalDist().cdf(z)
        ax.axhline(p, color="#777777", linestyle=":" if z < 3 else "--", linewidth=0.85, label=rf"local {int(z)}$\sigma$")
    ax.set_yscale("log")
    ax.set_xlim(18.0, 252.0)
    ax.set_ylim(max(1.0e-4, 0.6 * float(np.nanmin(local))), 1.05)
    ax.set_xlabel("Mass hypothesis (MeV)")
    ax.set_ylabel("Corresponding p-value")
    ax.set_title("Combined local and scan-equivalent discovery p-values")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#c9c9c9", ncol=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    save(fig, "combined_local_p0_points")

    summary_work = work[["mass_GeV", "p0_analytic"]].rename(columns={"p0_analytic": "p_value"})
    summaries.append(pvalue_summary_row("asymptotic p0", "combined", summary_work, n_eff))


def plot_individual_p0(
    single: pd.DataFrame,
    review_2021: pd.DataFrame,
    summaries: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.9, 9.1), sharex=False)
    for ax, dataset, label in zip(axes, ("2015", "2016", "2021"), ("2015", "2016 10%", "2021 1%")):
        if dataset == "2021":
            # Use the 2021 band-table export, whose p0 column is already free of
            # the isolated scan-optimizer failures in the raw single-scan CSV.
            work = review_2021[
                ["mass_GeV", "sigma_mass_res_GeV", "p0_analytic"]
            ].sort_values("mass_GeV").copy()
            sigma_column = "sigma_mass_res_GeV"
        else:
            work = single[single["dataset"] == dataset].sort_values("mass_GeV").copy()
            failed_masses = set(INDIVIDUAL_LIMIT_REPAIRS_MEV[dataset])
            mass_mev = np.rint(1000.0 * work["mass_GeV"].to_numpy(float)).astype(int)
            # A p-value cannot be repaired by interpolating neighboring
            # p-values.  Omit the known failed fits and connect the adjacent
            # valid hypotheses instead.
            work = work[~pd.Series(mass_mev, index=work.index).isin(failed_masses)].copy()
            sigma_column = "sigma_val"
        work["mass_MeV"] = 1000.0 * work["mass_GeV"].astype(float)
        n_eff = effective_trials(work, sigma_column)
        local = np.clip(work["p0_analytic"].to_numpy(float), 1.0e-12, 1.0)
        global_p = sidak_global(local, n_eff)
        ax.plot(work["mass_MeV"], local, color=COLORS[dataset], linewidth=1.85, label=r"Local asymptotic $p_0$")
        ax.plot(work["mass_MeV"], global_p, color=COLORS["global"], linewidth=1.7, linestyle="--", label=rf"Sidak-equivalent global $p$ ($N_{{\rm eff}}={n_eff:.1f}$)")
        ax.set_yscale("log")
        ax.set_ylim(max(1.0e-4, 0.55 * float(np.nanmin(local))), 1.05)
        ax.set_xlim(float(work["mass_MeV"].min()) - 1.0, float(work["mass_MeV"].max()) + 1.0)
        ax.set_ylabel("Corresponding p-value")
        ax.set_title(label, loc="left", fontweight="normal")
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#c9c9c9")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
        ax.yaxis.set_minor_formatter(NullFormatter())
        style_axis(ax)
        summary_work = work[["mass_GeV", "p0_analytic"]].rename(columns={"p0_analytic": "p_value"})
        summaries.append(pvalue_summary_row("asymptotic p0", dataset, summary_work, n_eff))
    axes[-1].set_xlabel("Mass hypothesis (MeV)")
    fig.suptitle("Individual-campaign local and scan-equivalent discovery p-values", y=0.995)
    fig.tight_layout()
    save(fig, "individual_asymptotic_local_global_pvalues")


def plot_limit_consistency_pvalues(combined: pd.DataFrame, summaries: list[dict[str, object]]) -> None:
    work = combined.sort_values("mass_GeV").copy()
    work["mass_MeV_integer"] = np.rint(1000.0 * work["mass_GeV"].to_numpy(float)).astype(int)
    excluded = work[work["mass_MeV_integer"].isin(TAIL_PVALUE_SUPERSEDED_MASSES_MEV)].copy()
    if set(excluded["mass_MeV_integer"].tolist()) != set(TAIL_PVALUE_SUPERSEDED_MASSES_MEV):
        raise RuntimeError("tail-p exclusion list does not match the archived combined mass grid")
    pd.DataFrame(
        {
            "mass_MeV": excluded["mass_MeV_integer"],
            "dataset_set": excluded["dataset_set"],
            "reason": "stored tail fractions correspond to an observed upper limit superseded by the display-repair table",
            "presentation": "omit the mass row and connect adjacent valid hypotheses; do not interpolate a p-value",
        }
    ).sort_values("mass_MeV").to_csv(TAIL_PVALUE_EXCLUSIONS, index=False)
    work = work[~work["mass_MeV_integer"].isin(TAIL_PVALUE_SUPERSEDED_MASSES_MEV)].copy()
    x = work["mass_MeV"].to_numpy(float)
    panels = [
        ("p_strong", r"$p_{\rm strong}$", "#4C72B0"),
        ("p_weak", r"$p_{\rm weak}$", "#DD8452"),
        ("p_two", r"$p_{\rm two}$", "#111111"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.9, 8.4), sharex=True, sharey=True)
    toy_floor = 1.0 / (N_LIMIT_TOYS + 1.0)
    for ax, (column, label, color) in zip(axes, panels):
        raw = work[column].to_numpy(float)
        values = np.clip(raw, toy_floor, 1.0)
        ax.plot(x, values, color=color, linewidth=1.8, label=label)
        ax.set_yscale("log")
        ax.set_ylim(8.0e-4, 1.05)
        ax.set_ylabel("Corresponding p-value")
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#c9c9c9")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=60))
        ax.yaxis.set_minor_formatter(NullFormatter())
        style_axis(ax)
        summary_work = work[["mass_GeV", column]].rename(columns={column: "p_value"})
        summaries.append(pvalue_summary_row("background-only upper-limit toys", column, summary_work))
    axes[-1].set_xlim(18.0, 252.0)
    axes[-1].set_xlabel("Mass hypothesis (MeV)")
    axes[-1].xaxis.set_minor_locator(AutoMinorLocator(5))
    fig.suptitle("Upper-limit consistency p-values from background-only toys", y=0.995)
    fig.tight_layout()
    save(fig, "combined_limit_tail_pvalues_points")


def campaign_projection_table(combined: pd.DataFrame) -> pd.DataFrame:
    factors = {"2015": 1.0, "2016": 10.0, "2021": 100.0}
    rows: list[dict[str, object]] = []
    for _, source in combined.sort_values("mass_GeV").iterrows():
        parsed = ast.literal_eval(str(source["meta"]))
        current_density = 0.0
        full_campaign_density = 0.0
        density_terms: list[str] = []
        for item in parsed:
            key = str(item["key"])
            density = float(item["dens"])
            current_density += density
            full_campaign_density += density * factors[key]
            density_terms.append(f"{key}:{density:.17g}")
        scale = math.sqrt(current_density / full_campaign_density)
        observed = float(source["eps2_obs_dimuon"])
        median = float(source["eps2_med_dimuon"])
        rows.append(
            {
                "mass_GeV": float(source["mass_GeV"]),
                "mass_MeV": float(source["mass_MeV"]),
                "dataset_set": str(source["dataset_set"]),
                "density_meta_terms": ";".join(density_terms),
                "density_current_samples": current_density,
                "density_full_campaign_scale": full_campaign_density,
                "campaign_scale_sqrt_current_over_full": scale,
                "current_candidate_eps2_observed": observed,
                "current_candidate_eps2_expected_median": median,
                "projected_full_campaign_eps2_observed_equivalent": observed * scale,
                "projected_full_campaign_eps2_expected_median": median * scale,
                "campaign_scale_2015": factors["2015"],
                "campaign_scale_2016": factors["2016"],
                "campaign_scale_2021": factors["2021"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "final_preunblind_current_vs_full_campaign_projection_eps2.csv", index=False)
    return out


def plot_campaign_projection(combined: pd.DataFrame) -> None:
    table = campaign_projection_table(combined)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    series = [
        ("current_candidate_eps2_observed", "#111111", "-", 2.25, "Current samples: observed"),
        ("current_candidate_eps2_expected_median", "#666666", ":", 1.65, "Current samples: expected median"),
        ("projected_full_campaign_eps2_observed_equivalent", "#6A3D9A", "-", 2.05, "Full-campaign scaling: observed-based"),
        ("projected_full_campaign_eps2_expected_median", "#E66101", "--", 1.95, "Full-campaign scaling: expected median"),
    ]
    for column, color, linestyle, width, label in series:
        ax.plot(table["mass_MeV"], table[column], color=color, linestyle=linestyle, linewidth=width, label=label)
    ax.set_yscale("log")
    ax.set_xlim(18.0, 252.0)
    ax.set_xlabel("Mass hypothesis (MeV)")
    ax.set_ylabel(r"90% CL upper limit on $\epsilon^2$")
    ax.set_title("Scaling projection for the full 2016 and 2021 datasets")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#c9c9c9")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=70))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    save(fig, "final_preunblind_current_vs_full_campaign_projection_eps2")


def main() -> None:
    setup_style()
    single, individual, review_2021, combined = make_review_tables()
    plot_2021_limits(review_2021)
    plot_combined_bands(combined)
    plot_comparison(individual, combined)

    summaries: list[dict[str, object]] = []
    plot_combined_p0(combined, summaries)
    plot_individual_p0(single, review_2021, summaries)
    plot_limit_consistency_pvalues(combined, summaries)
    pd.DataFrame(summaries).to_csv(P_VALUE_SUMMARY, index=False)

    plot_campaign_projection(combined)
    print(f"Wrote review figures to {OUTDIR}")
    print(f"Wrote repair audit to {REPAIR_AUDIT}")
    print(f"Wrote p-value summary to {P_VALUE_SUMMARY}")


if __name__ == "__main__":
    main()
