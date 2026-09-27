#!/usr/bin/env python3
"""Publication figures for the v4.6 full-100 refmatched closure study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy import stats


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
TOY_ROOT = HERE / "inputs" / "paired_exposure_toys_100.root"
SPEC = json.loads((HERE / "study_spec.json").read_text())
SCENARIOS = ("2021_1pct_x10", "2021_10pct", "2021_1pct_x100", "2021_10pct_x10")
LABELS = {key: value["label"] for key, value in SPEC["scenarios"].items()}
COLORS = {
    "2021_1pct_x10": "#0072B2",
    "2021_10pct": "#D55E00",
    "2021_1pct_x100": "#009E73",
    "2021_10pct_x10": "#CC79A7",
}
ZCOLORS = {0.0: "#4D4D4D", 1.0: "#0072B2", 3.0: "#E69F00", 5.0: "#D55E00"}

mpl.rcParams.update({
    "font.size": 9.4,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={
        "Title": stem,
        "Author": "HPS-GPR v4.6 study",
        "Subject": "Full-100 refmatched injection-extraction diagnostics",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    return [png, pdf]


def rebin(values: np.ndarray, edges: np.ndarray, factor: int = 5) -> tuple[np.ndarray, np.ndarray]:
    size = (len(values) // factor) * factor
    vals = np.asarray(values[:size], float).reshape(-1, factor).sum(axis=1)
    rebinned_edges = np.asarray(edges[: size + 1 : factor], float)
    if len(rebinned_edges) != len(vals) + 1:
        rebinned_edges = np.append(rebinned_edges, edges[size])
    return vals, rebinned_edges


def source_distribution_figure() -> list[Path]:
    fig, axes = plt.subplots(4, 3, figsize=(15.2, 14.0))
    with uproot.open(TOY_ROOT) as root:
        for row_index, scenario in enumerate(SCENARIOS):
            entry = SPEC["scenarios"][scenario]
            family = entry["source_family"]
            multiplier = int(entry["source_multiplier"])
            source_record = SPEC["source_inputs"][family]
            with uproot.open(HERE / source_record["root"]) as source:
                observed, edges = source["input_hist"].to_numpy()
            truth, truth_edges = root[f"truth/gengamma/{family}_mean"].to_numpy()
            if not np.array_equal(edges, truth_edges):
                raise RuntimeError("source/truth binning mismatch")
            toys = np.vstack([
                root[f"toys/gengamma/{scenario}/toy_{index:04d}"].to_numpy()[0]
                for index in range(100)
            ]).astype(float)
            expected = np.asarray(truth, float) * multiplier
            obs_rebin, display_edges = rebin(np.asarray(observed, float), edges)
            truth_rebin, _ = rebin(np.asarray(truth, float), edges)
            expected_rebin, _ = rebin(expected, edges)
            toy_rebinned = np.vstack([rebin(values, edges)[0] for values in toys])
            mean_rebin = np.mean(toy_rebinned, axis=0)
            median_rebin = np.median(toy_rebinned, axis=0)
            q16 = np.quantile(toy_rebinned, 0.16, axis=0)
            q84 = np.quantile(toy_rebinned, 0.84, axis=0)
            centers = 500.0 * (display_edges[:-1] + display_edges[1:])
            support = (centers >= 40.0) & (centers <= 300.0)
            metadata = json.loads((HERE / source_record["metadata"]).read_text())
            fit_record = next(row for row in metadata["fits"] if row["tag"] == "fGenGammaThresh")

            ax = axes[row_index, 0]
            ax.step(centers[support], obs_rebin[support], where="mid", color="#555555", lw=0.9, label="native source")
            ax.plot(centers[support], truth_rebin[support], color="#0072B2", lw=1.5, label="fGenGammaThresh fit")
            ax.set_yscale("log")
            ax.set_title(f"Native-source stress fit\nPearson $\\chi^2$/ndf = {float(fit_record['pearson_chi2ndf']):.3f}")
            ax.set_ylabel("Counts / 0.625 MeV")
            if row_index == 0:
                ax.legend(frameon=False, loc="upper right")

            ax = axes[row_index, 1]
            ax.fill_between(centers[support], q16[support], q84[support], color=COLORS[scenario], alpha=0.20, label="toy 16-84%")
            ax.plot(centers[support], expected_rebin[support], color="black", lw=1.45, label="analytic expectation")
            ax.plot(centers[support], mean_rebin[support], color="#D55E00", lw=1.15, label="mean of 100 toys")
            ax.plot(centers[support], median_rebin[support], color="#009E73", lw=1.0, ls="--", label="median of 100 toys")
            ax.set_yscale("log")
            ax.set_title(f"{LABELS[scenario]}: scaled pseudoexperiments")
            if row_index == 0:
                ax.legend(frameon=False, loc="upper right")

            ax = axes[row_index, 2]
            valid = support & (expected_rebin > 0)
            residual = (mean_rebin - expected_rebin) / np.sqrt(np.clip(expected_rebin / 100.0, 1e-12, None))
            ax.axhspan(-2, 2, color="#BBBBBB", alpha=0.18)
            ax.axhline(0, color="black", lw=0.8)
            ax.plot(centers[valid], residual[valid], color=COLORS[scenario], lw=1.05)
            ax.set_ylim(-5.2, 5.2)
            ax.set_title("Toy-mean residual / $\\sqrt{E/100}$")

            for panel in axes[row_index]:
                panel.axvspan(40, 50, color="#999999", alpha=0.10)
                panel.axvspan(250, 300, color="#999999", alpha=0.10)
                panel.set_xlim(40, 300)
                panel.grid(alpha=0.20)
                if row_index == 3:
                    panel.set_xlabel("Invariant mass [MeV]")
    fig.suptitle("Full-100 source/exposure pseudoexperiment construction\nCommon smooth stress family fitted independently to the native 1% and 10% sources", y=0.987)
    fig.subplots_adjust(left=0.060, right=0.985, bottom=0.055, top=0.90, hspace=0.40, wspace=0.18)
    return save(fig, "functional_form_exposure_mass_distributions_full100")


def spurious_figure(accepted: pd.DataFrame, raw: pd.DataFrame, summary: pd.DataFrame, analytic: pd.DataFrame, exclusions: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(4, 3, figsize=(15.2, 13.5))
    masses = np.asarray(SPEC["masses_gev"], float) * 1000.0
    rng = np.random.default_rng(20260812)
    excluded_keys = set(zip(exclusions.scenario.astype(str), exclusions.background_toy_index.astype(int), np.round(exclusions.mass_GeV.astype(float), 9), exclusions.inj_nsigma.astype(float)))
    for row_index, scenario in enumerate(SCENARIOS):
        rows = accepted[(accepted.scenario == scenario) & np.isclose(accepted.inj_nsigma, 0.0)]
        summed = summary[(summary.scenario == scenario) & np.isclose(summary.inj_nsigma, 0.0)].sort_values("mass_MeV")
        amean = analytic[analytic.scenario == scenario].sort_values("mass_MeV")
        ax = axes[row_index, 0]
        for mass in sorted(rows.mass_GeV.unique()):
            group = rows[np.isclose(rows.mass_GeV, mass)]
            jitter = rng.uniform(-1.8, 1.8, len(group))
            ax.scatter(1000.0 * mass + jitter, group.pull, s=10, alpha=0.38, color=COLORS[scenario], edgecolors="none")
            ax.plot([1000.0 * mass - 2.2, 1000.0 * mass + 2.2], [np.median(group.pull)] * 2, color="black", lw=1.4)
        raw_zero = raw[(raw.scenario == scenario) & np.isclose(raw.inj_nsigma, 0.0)]
        for _, item in raw_zero.iterrows():
            key = (scenario, int(item.background_toy_index), round(float(item.mass_GeV), 9), 0.0)
            if key in excluded_keys and np.isfinite(item.get("pull", np.nan)):
                ax.scatter(1000.0 * item.mass_GeV, item.pull, marker="x", s=45, lw=1.6, color="#B2182B", zorder=5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("zero-signal pull")
        ax.set_title(f"{LABELS[scenario]}: accepted pulls\nblack bars: medians; red x: excluded raw fit")

        ax = axes[row_index, 1]
        lower = summed.accepted_pull_mean - summed.accepted_pull_mean_ci95_low
        upper = summed.accepted_pull_mean_ci95_high - summed.accepted_pull_mean
        ax.errorbar(summed.mass_MeV, summed.accepted_pull_mean, yerr=np.vstack([lower, upper]), color=COLORS[scenario], marker="o", capsize=2.5, label="accepted mean, 95% t CI")
        ax.scatter(summed.mass_MeV, summed.raw_pull_mean, marker="x", color="#666666", s=26, label="raw first-attempt mean")
        ax.plot(amean.mass_MeV, amean.pull, marker="D", ms=4, lw=0.9, ls="--", color="#7B3294", label="analytic-mean closure")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("mean pull")
        ax.set_title("Spurious-signal mean")
        if row_index == 0:
            ax.legend(frameon=False, loc="best")

        ax = axes[row_index, 2]
        lower = summed.accepted_pull_width - summed.accepted_pull_width_ci95_low
        upper = summed.accepted_pull_width_ci95_high - summed.accepted_pull_width
        ax.errorbar(summed.mass_MeV, summed.accepted_pull_width, yerr=np.vstack([lower, upper]), color=COLORS[scenario], marker="o", capsize=2.5, label="accepted width, 95% normal-theory CI")
        ax.scatter(summed.mass_MeV, summed.raw_pull_width, marker="x", color="#666666", s=26, label="raw first-attempt width")
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_ylabel("sample pull width")
        ax.set_title("Spurious-signal width")
        if row_index == 0:
            ax.legend(frameon=False, loc="best")

        for panel in axes[row_index]:
            panel.set_xticks(masses)
            panel.set_xlabel("Mass [MeV]")
            panel.grid(alpha=0.21)
    fig.suptitle("Full-100 zero-signal diagnostics\nStatistical extremes passing the pull-blind technical gate remain included", y=0.987)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.055, top=0.89, hspace=0.48, wspace=0.20)
    return save(fig, "spurious_signal_zero_sigma_full100")


def closure_figures(accepted: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    for scenario in SCENARIOS:
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.7))
        for z in (0.0, 1.0, 3.0, 5.0):
            summed = summary[(summary.scenario == scenario) & np.isclose(summary.inj_nsigma, z)].sort_values("mass_MeV")
            color = ZCOLORS[z]
            axes[0, 0].errorbar(
                summed.mass_MeV, summed.accepted_pull_mean,
                yerr=np.vstack([summed.accepted_pull_mean - summed.accepted_pull_mean_ci95_low, summed.accepted_pull_mean_ci95_high - summed.accepted_pull_mean]),
                color=color, marker="o", ms=4, capsize=2, label=f"$Z_{{inj}}={z:.0f}$",
            )
            axes[0, 0].scatter(summed.mass_MeV, summed.raw_pull_mean, marker="x", s=20, color=color, alpha=0.45)
            axes[0, 1].errorbar(
                summed.mass_MeV, summed.accepted_pull_width,
                yerr=np.vstack([summed.accepted_pull_width - summed.accepted_pull_width_ci95_low, summed.accepted_pull_width_ci95_high - summed.accepted_pull_width]),
                color=color, marker="o", ms=4, capsize=2,
            )
            rows = accepted[(accepted.scenario == scenario) & np.isclose(accepted.inj_nsigma, z)]
            dz_mean, dz_low, dz_high = [], [], []
            for mass in SPEC["masses_gev"]:
                values = pd.to_numeric(rows[np.isclose(rows.mass_GeV, mass)].delta_z).to_numpy(float)
                mean, se = float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))
                half = float(stats.t.ppf(0.975, len(values) - 1)) * se
                dz_mean.append(mean); dz_low.append(mean - half); dz_high.append(mean + half)
            axes[1, 0].errorbar(
                np.asarray(SPEC["masses_gev"]) * 1000.0, dz_mean,
                yerr=np.vstack([np.asarray(dz_mean) - np.asarray(dz_low), np.asarray(dz_high) - np.asarray(dz_mean)]),
                color=color, marker="o", ms=4, capsize=2,
            )
            if z > 0:
                axes[1, 1].errorbar(
                    summed.mass_MeV, summed.accepted_median_recovery,
                    yerr=np.vstack([summed.accepted_median_recovery - summed.accepted_recovery_q16, summed.accepted_recovery_q84 - summed.accepted_median_recovery]),
                    color=color, marker="o", ms=4, capsize=2,
                )
        axes[0, 0].axhline(0, color="black", lw=0.8); axes[0, 0].set_ylabel("mean pull"); axes[0, 0].set_title("Pull mean (95% t intervals)")
        axes[0, 1].axhline(1, color="black", lw=0.8, ls="--"); axes[0, 1].set_ylabel("sample pull width"); axes[0, 1].set_title("Pull width (95% normal-theory intervals)")
        axes[1, 0].axhline(0, color="black", lw=0.8); axes[1, 0].set_ylabel(r"mean $(\widehat Z-Z_{inj})$"); axes[1, 0].set_title("Residual fitted significance")
        axes[1, 1].axhline(1, color="black", lw=0.8, ls="--"); axes[1, 1].set_ylabel(r"median $\widehat A/A_{inj}$"); axes[1, 1].set_title("Amplitude recovery (central 68% spread)")
        axes[0, 0].legend(frameon=False, ncol=2, loc="best")
        for ax in axes.flat:
            ax.set_xticks(np.asarray(SPEC["masses_gev"]) * 1000.0)
            ax.set_xlabel("Mass [MeV]")
            ax.grid(alpha=0.22)
        minimum = int(summary[summary.scenario == scenario].accepted_n.min())
        fig.suptitle(f"Full-100 refmatched closure: {LABELS[scenario]}\naccepted cell sizes {minimum}-100; x symbols in the mean panel show raw first-attempt moments", y=0.985)
        fig.subplots_adjust(left=0.080, right=0.985, bottom=0.08, top=0.88, hspace=0.34, wspace=0.22)
        outputs.extend(save(fig, f"refmatched_closure_full100_{scenario}"))
    return outputs


def bias_figure(accepted: pd.DataFrame, analytic: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    rows = accepted[(accepted.scenario == "2021_1pct_x10") & np.isclose(accepted.mass_GeV, 0.065)]
    zero = rows[np.isclose(rows.inj_nsigma, 0.0)].sort_values("background_toy_index")
    pilot = zero[zero.background_toy_index < 10].pull.to_numpy(float)
    reserve = zero[zero.background_toy_index >= 10].pull.to_numpy(float)
    full = zero.pull.to_numpy(float)
    asimov = float(analytic[(analytic.scenario == "2021_1pct_x10") & np.isclose(analytic.mass_GeV, 0.065)].pull.iloc[0])

    bins = np.linspace(min(full.min(), -3.0), max(full.max(), 4.0), 22)
    axes[0, 0].hist(reserve, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color=COLORS["2021_1pct_x10"], label="confirmatory toys 10-99")
    axes[0, 0].hist(pilot, bins=bins, density=True, histtype="step", lw=1.6, color="#D55E00", label="pilot toys 0-9")
    x = np.linspace(bins[0], bins[-1], 300)
    axes[0, 0].plot(x, stats.norm.pdf(x, loc=np.mean(reserve), scale=np.std(reserve, ddof=1)), color="#333333", lw=1.0, ls="--", label="reserve fitted normal")
    axes[0, 0].axvline(asimov, color="#7B3294", lw=1.5, label=f"analytic mean: {asimov:.3f}")
    axes[0, 0].axvline(0, color="black", lw=0.8)
    axes[0, 0].set_xlabel("zero-signal pull"); axes[0, 0].set_ylabel("density"); axes[0, 0].set_title("Pilot and independent confirmation")
    axes[0, 0].legend(frameon=False, loc="best")

    values = zero.sort_values("background_toy_index").pull.to_numpy(float)
    indices = np.arange(1, len(values) + 1)
    running = np.cumsum(values) / indices
    low = np.full_like(running, np.nan); high = np.full_like(running, np.nan)
    for index in range(1, len(values)):
        sample = values[: index + 1]
        half = stats.t.ppf(0.975, index) * np.std(sample, ddof=1) / math.sqrt(index + 1)
        low[index] = running[index] - half; high[index] = running[index] + half
    axes[0, 1].fill_between(indices, low, high, color=COLORS["2021_1pct_x10"], alpha=0.18, label="running 95% t interval")
    axes[0, 1].plot(indices, running, color=COLORS["2021_1pct_x10"], lw=1.2, label="running mean")
    axes[0, 1].axvline(10.5, color="#D55E00", lw=1.0, ls="--", label="pilot / confirmation boundary")
    axes[0, 1].axhline(asimov, color="#7B3294", lw=1.2, ls=":", label="analytic-mean closure")
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set_xlabel("number of background toys"); axes[0, 1].set_ylabel("running mean pull"); axes[0, 1].set_title("Convergence of the 65 MeV offset")
    axes[0, 1].legend(frameon=False, loc="best")

    theoretical, ordered = stats.probplot(reserve, dist="norm", fit=False)
    axes[1, 0].scatter(theoretical, ordered, s=18, color=COLORS["2021_1pct_x10"], alpha=0.70)
    q = np.linspace(min(theoretical), max(theoretical), 100)
    axes[1, 0].plot(q, np.mean(reserve) + np.std(reserve, ddof=1) * q, color="black", lw=1.0, label="normal with reserve mean/width")
    axes[1, 0].set_xlabel("standard-normal quantile"); axes[1, 0].set_ylabel("ordered reserve pull"); axes[1, 0].set_title("Confirmation-sample Q-Q diagnostic")
    axes[1, 0].legend(frameon=False, loc="best")

    positions = np.arange(4)
    strengths = (0.0, 1.0, 3.0, 5.0)
    for offset, (label, predicate, marker, color) in enumerate((
        ("pilot 0-9", lambda g: g.background_toy_index < 10, "s", "#D55E00"),
        ("confirmation 10-99", lambda g: g.background_toy_index >= 10, "o", "#0072B2"),
        ("full 0-99", lambda g: np.ones(len(g), dtype=bool), "D", "#009E73"),
    )):
        means, halves = [], []
        for z in strengths:
            group = rows[np.isclose(rows.inj_nsigma, z)]
            sample = group.loc[predicate(group), "pull"].to_numpy(float)
            means.append(np.mean(sample))
            halves.append(stats.t.ppf(0.975, len(sample) - 1) * np.std(sample, ddof=1) / math.sqrt(len(sample)))
        axes[1, 1].errorbar(positions + (offset - 1) * 0.10, means, yerr=halves, marker=marker, color=color, capsize=2, lw=1.0, label=label)
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_xticks(positions, ["0", "1", "3", "5"])
    axes[1, 1].set_xlabel(r"injected strength [$\sigma_A^{ref}$]"); axes[1, 1].set_ylabel("mean pull (95% t interval)"); axes[1, 1].set_title("Correlated strength-response diagnostic")
    axes[1, 1].legend(frameon=False, loc="best")
    for ax in axes.flat:
        ax.grid(alpha=0.20)
    fig.suptitle(r"Investigation of the $1\%\times10$, 65 MeV closure offset" + "\nconfirmatory reserve-90 mean = 0.768, 95% CI [0.570, 0.965]", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.88, hspace=0.32, wspace=0.22)
    return save(fig, "onepct_x10_65mev_bias_confirmation")


def optimizer_figure(accepted: pd.DataFrame, summary: pd.DataFrame, exclusions: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    zero = summary[np.isclose(summary.inj_nsigma, 0.0)]
    axes[0, 0].scatter(zero.raw_pull_mean, zero.accepted_pull_mean, c=[COLORS[item] for item in zero.scenario], s=38, alpha=0.78)
    limits = [min(zero.raw_pull_mean.min(), zero.accepted_pull_mean.min()) - 0.1, max(zero.raw_pull_mean.max(), zero.accepted_pull_mean.max()) + 0.1]
    axes[0, 0].plot(limits, limits, color="black", lw=0.8, ls="--")
    axes[0, 0].set_xlim(limits); axes[0, 0].set_ylim(limits)
    axes[0, 0].set_xlabel("raw first-attempt mean pull"); axes[0, 0].set_ylabel("accepted mean pull")
    axes[0, 0].set_title("Effect of numerical branch selection")
    largest = zero.iloc[np.argmax(np.abs(zero.accepted_pull_mean - zero.raw_pull_mean))]
    axes[0, 0].annotate(f"{LABELS[largest.scenario]}, {largest.mass_MeV:.0f} MeV", (largest.raw_pull_mean, largest.accepted_pull_mean), xytext=(6, -14), textcoords="offset points", fontsize=8)

    for scenario in SCENARIOS:
        group = accepted[accepted.scenario == scenario].groupby("mass_MeV").refit_upper_boundary.mean().reindex(np.asarray(SPEC["masses_gev"]) * 1000.0)
        axes[0, 1].plot(group.index, 100 * group.values, marker="o", lw=1.1, color=COLORS[scenario], label=LABELS[scenario])
    axes[0, 1].set_ylabel("stable upper-bound occupancy [%]"); axes[0, 1].set_xlabel("Mass [MeV]"); axes[0, 1].set_title("Accepted factor-15 contacts")
    axes[0, 1].legend(frameon=False, loc="best")

    counts = accepted.n_attempts.value_counts().reindex([1, 3, 5], fill_value=0)
    axes[1, 0].bar(counts.index.astype(str), counts.values, color=["#777777", "#0072B2", "#D55E00"])
    for x, value in enumerate(counts.values):
        axes[1, 0].text(x, value + 40, f"{value}", ha="center", va="bottom", fontsize=9)
    axes[1, 0].set_xlabel("optimizer attempts per accepted fit state"); axes[1, 0].set_ylabel("accepted state count"); axes[1, 0].set_title("Adaptive restart workload")

    axes[1, 1].axis("off")
    lines = [
        "Predeclared fit-state exclusions (6 / 8000):",
        "",
        r"1%-source $\times100$, toy 83, 90 MeV, $3\sigma$: one row",
        "  no reproducible valid covariance branch in five attempts",
        "",
        r"10%-source $\times10$, toy 28, 180 MeV, $5\sigma$: one row",
        "  unreplicated near-upper-bound branch in five attempts",
        "",
        r"10%-source $\times10$, toy 66, 90 MeV: all four strengths",
        "  irreproducible matched background/reference branch",
        "",
        "All affected cells retain 99 accepted toys.",
        "Statistical extremes passing the pull-blind technical gate remain included.",
        "Repeat-triggered boundary fits remain only when replicated.",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.2, linespacing=1.25)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.grid(alpha=0.20)
    fig.suptitle("Optimizer-repeat gate and exclusion audit\nSelection uses GP log marginal likelihood and branch reproducibility, never pull size or sign", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.88, hspace=0.32, wspace=0.24)
    return save(fig, "optimizer_gate_diagnostics_full100")


def eps2_figure(accepted: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(4, 3, figsize=(15.2, 13.8))
    rng = np.random.default_rng(45612)
    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, z in enumerate((1.0, 3.0, 5.0)):
            ax = axes[row_index, column_index]
            rows = accepted[(accepted.scenario == scenario) & np.isclose(accepted.inj_nsigma, z)]
            summed = summary[(summary.scenario == scenario) & np.isclose(summary.inj_nsigma, z)].sort_values("mass_MeV")
            for mass in sorted(rows.mass_GeV.unique()):
                group = rows[np.isclose(rows.mass_GeV, mass)]
                jitter = rng.uniform(-1.8, 1.8, len(group))
                ax.scatter(1000.0 * mass + jitter, group.eps2_hat_signed, s=8, color=COLORS[scenario], alpha=0.24, edgecolors="none")
            ax.errorbar(
                summed.mass_MeV, summed.accepted_eps2_hat_median,
                yerr=np.vstack([summed.accepted_eps2_hat_median - summed.accepted_eps2_hat_q16, summed.accepted_eps2_hat_q84 - summed.accepted_eps2_hat_median]),
                marker="D", ms=3.8, lw=1.0, capsize=2, color=COLORS[scenario], label="fitted median and 16-84% spread",
            )
            injected = rows.groupby("mass_MeV").eps2_injected.quantile([0.16, 0.5, 0.84]).unstack()
            ax.errorbar(
                injected.index, injected[0.5],
                yerr=np.vstack([injected[0.5] - injected[0.16], injected[0.84] - injected[0.5]]),
                marker="x", ms=4.0, lw=0.8, ls="--", capsize=2, color="#444444", label="injected median and 16-84% spread",
            )
            linthresh = max(float(np.median(np.abs(rows.eps2_sigma))) * 0.5, 1e-12)
            ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(np.asarray(SPEC["masses_gev"]) * 1000.0)
            ax.set_xlabel("Mass [MeV]")
            ax.set_title(f"{LABELS[scenario]}, $Z_{{inj}}={z:.0f}$")
            if column_index == 0:
                ax.set_ylabel(r"signed $\epsilon^2$ extraction coordinate")
            ax.grid(alpha=0.20)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(r"Full-100 extracted $\epsilon^2$-coordinate distributions" + "\nraw accepted fits, medians, and central 68% spreads; panel-wise symmetric-log axes", y=0.987)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.055, top=0.89, hspace=0.46, wspace=0.18)
    return save(fig, "eps2_coordinate_distributions_full100")


def pairwise_figure(summary: pd.DataFrame) -> list[Path]:
    pairs = [
        ("Native 10% vs 1% source x10", "2021_10pct", "2021_1pct_x10"),
        ("10% source x10 vs 1% source x100", "2021_10pct_x10", "2021_1pct_x100"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    for row_index, (title, first, second) in enumerate(pairs):
        for column_index, z in enumerate((1.0, 3.0, 5.0)):
            ax = axes[row_index, column_index]
            for scenario, marker in ((first, "o"), (second, "s")):
                group = summary[(summary.scenario == scenario) & np.isclose(summary.inj_nsigma, z)].sort_values("mass_MeV")
                ax.fill_between(group.mass_MeV, group.accepted_eps2_hat_q16, group.accepted_eps2_hat_q84, color=COLORS[scenario], alpha=0.14)
                ax.plot(group.mass_MeV, group.accepted_eps2_hat_median, marker=marker, ms=4, lw=1.15, color=COLORS[scenario], label=LABELS[scenario])
                ax.plot(group.mass_MeV, group.accepted_eps2_injected_median, lw=0.8, ls="--", color=COLORS[scenario])
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"{title}\n$Z_{{inj}}={z:.0f}$")
            ax.set_xticks(np.asarray(SPEC["masses_gev"]) * 1000.0)
            ax.set_xlabel("Mass [MeV]")
            if column_index == 0:
                ax.set_ylabel(r"signed fitted $\epsilon^2$ coordinate")
            ax.grid(alpha=0.20)
            if row_index == 0 and column_index == 0:
                ax.legend(frameon=False, loc="best")
    fig.suptitle("Full-100 direct unpaired source/exposure comparisons\nsolid: fitted medians with central 68% spread; dashed: injected medians; normalization ratio = 1.1296466", y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.83, hspace=0.42, wspace=0.18)
    return save(fig, "direct_pairwise_eps2_comparisons_full100")


def main() -> int:
    accepted = pd.read_csv(DERIVED / "accepted_extraction_rows.csv")
    raw = pd.read_csv(DERIVED / "raw_primary_extraction_rows.csv")
    summary = pd.read_csv(DERIVED / "closure_summary.csv")
    analytic = pd.read_csv(DERIVED / "analytic_mean_zero_signal_closure.csv")
    exclusions = pd.read_csv(DERIVED / "exclusion_ledger.csv")
    outputs: list[Path] = []
    outputs.extend(source_distribution_figure())
    outputs.extend(spurious_figure(accepted, raw, summary, analytic, exclusions))
    outputs.extend(closure_figures(accepted, summary))
    outputs.extend(bias_figure(accepted, analytic))
    outputs.extend(optimizer_figure(accepted, summary, exclusions))
    outputs.extend(eps2_figure(accepted, summary))
    outputs.extend(pairwise_figure(summary))
    manifest = {
        "study_id": SPEC["study_id"],
        "figure_pairs": len(outputs) // 2,
        "figures": [
            {"path": str(path.relative_to(HERE)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(outputs)
        ],
        "input_hashes": {
            name: sha256(DERIVED / name)
            for name in (
                "accepted_extraction_rows.csv", "raw_primary_extraction_rows.csv",
                "closure_summary.csv", "analytic_mean_zero_signal_closure.csv", "exclusion_ledger.csv",
            )
        },
    }
    (DERIVED / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "figure_pairs": len(outputs) // 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
