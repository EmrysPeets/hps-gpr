#!/usr/bin/env python3
"""Create reviewed summary tables and clean aligned pseudo65 figures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.ticker import LogLocator, MultipleLocator


ROOT_FILE = HERE / "inputs" / "pseudo65_background_replacements.root"
PROVENANCE = HERE / "derived" / "input_provenance.json"
BASELINE_CSV = (
    REPO
    / "study_results"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "derived"
    / "observed_gp_states_v4p2_enriched.csv"
)
LANES = {
    "gp_mean": {
        "title": "GP-mean replacement",
        "short": "GP mean",
        "color": "#0072B2",
        "hist_key": "gp_mean/preselection/h_invM_8000",
        "expectation_key": "expectations/gp_mean_m065",
    },
    "functional_form": {
        "title": "Functional-form replacement",
        "short": r"$f_{\mathrm{GenGammaThresh}}$",
        "color": "#D55E00",
        "hist_key": (
            "functional_form_fGenGammaThresh/preselection/h_invM_8000"
        ),
        "expectation_key": "expectations/fGenGammaThresh_m065",
    },
}
SOURCE_KEY = "source/preselection/h_invM_8000"
COLORS = {
    "original": "#70757D",
    "ink": "#22252A",
    "grid": "#C5CAD1",
    "shade": "#B8A1CF",
}


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.23,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "legend.fontsize": 8.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def rebin_five(values: np.ndarray, edges: np.ndarray):
    vals = np.asarray(values, float)
    edg = np.asarray(edges, float)
    if len(vals) % 5:
        raise RuntimeError("Native histogram is not divisible by rebin factor five")
    rebinned = vals.reshape(-1, 5).sum(axis=1)
    rebinned_edges = edg[::5]
    if len(rebinned_edges) != len(rebinned) + 1:
        rebinned_edges = np.r_[rebinned_edges, edg[-1]]
    centers = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    return rebinned, rebinned_edges, centers


def load_scans():
    scans = {}
    for lane in LANES:
        path = HERE / "derived" / f"{lane}_results_reviewed.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["dataset"].astype(str) == "2021"].copy()
        scans[lane] = frame.sort_values("mass_GeV").reset_index(drop=True)
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline["dataset"].astype(str) == "2021"].copy()
    baseline = baseline.sort_values("mass_GeV").reset_index(drop=True)
    return scans, baseline


def build_m065_table(scans, baseline):
    records = []
    sources = {"v4.2 original": baseline, **scans}
    labels = {
        "v4.2 original": "v4.2 original 2021 10%",
        "gp_mean": "GP-mean conditional replacement",
        "functional_form": "fGenGammaThresh conditional replacement",
    }
    for key, frame in sources.items():
        row = frame[np.isclose(frame["mass_GeV"].to_numpy(float), 0.065)]
        if len(row) != 1:
            raise RuntimeError(f"{key}: expected one reviewed row at 65 MeV")
        row = row.iloc[0]
        record = {
            "sample": key,
            "label": labels[key],
            "mass_MeV": 65.0,
            "A_hat": float(row["A_hat"]),
            "sigma_A": float(row["sigma_A"]),
            "A_up_90cl": float(row["A_up"]),
            "eps2_up_90cl": float(row["eps2_up"]),
            "p0_local_asymptotic": float(row["p0_analytic"]),
            "Z_local_asymptotic": float(row["Z_analytic"]),
            "integral_density_counts_per_GeV": float(row["integral_density"]),
            "cls_calibration": str(row["cls_calibration"]),
            "expected_bands": False,
        }
        if key in scans:
            record.update(
                {
                    "selected_source": str(row["selected_source"]),
                    "review_status": str(row["review_status"]),
                    "selected_state_reproducing_attempt_count": int(
                        row["selected_state_reproducing_attempt_count"]
                    ),
                    "branch_multiplicity": int(row["branch_multiplicity"]),
                }
            )
        records.append(record)
    table = pd.DataFrame(records)
    out = HERE / "derived" / "m065_results_summary.csv"
    table.to_csv(out, index=False)
    return table


def draw_main_figure(scans, baseline):
    root_file = uproot.open(ROOT_FILE)
    source_values, source_edges = root_file[SOURCE_KEY].to_numpy()
    source_rebin, source_edges_rebin, spectrum_x = rebin_five(
        source_values, source_edges
    )
    spectrum_mass_mask = (spectrum_x >= 0.050) & (spectrum_x <= 0.250)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.2, 10.0),
        sharex="col",
        gridspec_kw={"height_ratios": [1.05, 1.0, 1.0], "hspace": 0.08, "wspace": 0.16},
    )
    positive_limits = []
    positive_p0 = []
    for frame in scans.values():
        positive_limits.extend(
            frame.loc[frame["eps2_up"] > 0.0, "eps2_up"].to_numpy(float)
        )
        positive_p0.extend(
            frame.loc[frame["p0_analytic"] > 0.0, "p0_analytic"].to_numpy(float)
        )
    positive_limits.extend(
        baseline.loc[baseline["eps2_up"] > 0.0, "eps2_up"].to_numpy(float)
    )
    positive_p0.extend(
        baseline.loc[baseline["p0_analytic"] > 0.0, "p0_analytic"].to_numpy(float)
    )
    limit_ymin = 10 ** np.floor(np.log10(max(min(positive_limits) * 0.75, 1.0e-9)))
    limit_ymax = 10 ** np.ceil(np.log10(max(positive_limits) * 1.2))
    p0_floor = min(1.0e-7, max(min(positive_p0) * 0.5, 1.0e-12))

    for column, (lane, info) in enumerate(LANES.items()):
        frame = scans[lane]
        pseudo_values, pseudo_edges = root_file[info["hist_key"]].to_numpy()
        expectation_values, expectation_edges = root_file[
            info["expectation_key"]
        ].to_numpy()
        pseudo_rebin, _, _ = rebin_five(pseudo_values, pseudo_edges)
        expectation_rebin, _, _ = rebin_five(
            expectation_values, expectation_edges
        )
        central = (spectrum_x >= 0.060) & (spectrum_x < 0.070)

        top, middle, bottom = axes[:, column]
        top.set_title(info["title"], color=info["color"], fontweight="semibold")
        top.step(
            1000.0 * spectrum_x[spectrum_mass_mask],
            source_rebin[spectrum_mass_mask],
            where="mid",
            color=COLORS["original"],
            linewidth=1.8,
            alpha=0.60,
            label="Original 2021 10%",
            zorder=1,
        )
        top.step(
            1000.0 * spectrum_x[spectrum_mass_mask],
            pseudo_rebin[spectrum_mass_mask],
            where="mid",
            color=info["color"],
            linewidth=1.05,
            alpha=0.95,
            label="Conditional pseudo-data",
            zorder=2,
        )
        top.plot(
            1000.0 * spectrum_x[central],
            expectation_rebin[central],
            color=COLORS["ink"],
            linestyle="--",
            linewidth=1.5,
            label="Replacement mean",
            zorder=3,
        )
        top.set_yscale("log")
        top.set_ylim(1.4e4, 1.6e6)
        top.text(
            0.975,
            0.92,
            r"$A_{\mathrm{inj}}=0$",
            ha="right",
            va="top",
            transform=top.transAxes,
            color=COLORS["ink"],
            fontsize=9.5,
        )
        top.legend(loc="lower left", frameon=False, ncol=1)

        mass_mev = 1000.0 * frame["mass_GeV"].to_numpy(float)
        base_mass_mev = 1000.0 * baseline["mass_GeV"].to_numpy(float)
        middle.plot(
            base_mass_mev,
            baseline["eps2_up"].to_numpy(float),
            color=COLORS["original"],
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            label="Original v4.2 observed",
        )
        middle.plot(
            mass_mev,
            frame["eps2_up"].to_numpy(float),
            color=info["color"],
            linewidth=1.8,
            label="Replacement observed",
        )
        middle.set_yscale("log")
        middle.set_ylim(limit_ymin, limit_ymax)
        middle.legend(loc="upper right", frameon=False)

        bottom.plot(
            base_mass_mev,
            np.clip(
                baseline["p0_analytic"].to_numpy(float),
                p0_floor,
                1.0,
            ),
            color=COLORS["original"],
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            label="Original v4.2",
        )
        bottom.plot(
            mass_mev,
            np.clip(frame["p0_analytic"].to_numpy(float), p0_floor, 1.0),
            color=info["color"],
            linewidth=1.8,
            label="Replacement",
        )
        bottom.set_yscale("log")
        bottom.set_ylim(p0_floor, 1.0)
        for z_value, label in ((3.0, r"$3\sigma$"), (5.0, r"$5\sigma$")):
            p_value = 0.5 * math.erfc(z_value / np.sqrt(2.0))
            bottom.axhline(
                p_value,
                color=COLORS["grid"],
                linestyle=":",
                linewidth=0.9,
            )
            if p_value >= p0_floor:
                bottom.text(
                    249.0,
                    p_value * 1.12,
                    label,
                    ha="right",
                    va="bottom",
                    fontsize=8.0,
                    color=COLORS["original"],
                )

        for ax in (top, middle, bottom):
            ax.axvspan(
                60.0,
                70.0,
                color=COLORS["shade"],
                alpha=0.14,
                linewidth=0,
            )
            ax.axvline(
                65.0,
                color=COLORS["shade"],
                linestyle=":",
                linewidth=0.9,
                alpha=0.9,
            )
            ax.set_xlim(50.0, 250.0)
            ax.xaxis.set_major_locator(MultipleLocator(25.0))
            ax.xaxis.set_minor_locator(MultipleLocator(5.0))
        middle.yaxis.set_major_locator(LogLocator(base=10))
        bottom.yaxis.set_major_locator(LogLocator(base=10))
        bottom.set_xlabel(r"Mass hypothesis $m_{A'}$ [MeV]")

    axes[0, 0].set_ylabel("Events / 0.625 MeV")
    axes[1, 0].set_ylabel(r"Observed 90% CL $\epsilon^2$")
    axes[2, 0].set_ylabel(r"Local asymptotic $p_0$")
    axes[0, 1].set_ylabel("Events / 0.625 MeV")
    axes[1, 1].set_ylabel(r"Observed 90% CL $\epsilon^2$")
    axes[2, 1].set_ylabel(r"Local asymptotic $p_0$")

    fig.suptitle(
        "2021 10% conditional central-window replacements around 65 MeV",
        y=0.995,
        fontsize=14.0,
        fontweight="semibold",
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.006,
        "v4.2 scan geometry (±2.25σ), kmax = 15; asymptotic 90% CLs; no expected bands",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["original"],
    )
    fig.subplots_adjust(top=0.955, bottom=0.065, left=0.085, right=0.985)
    out_png = HERE / "plots" / "pseudo65_observed_limit_p0_aligned.png"
    out_pdf = HERE / "plots" / "pseudo65_observed_limit_p0_aligned.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def draw_central_zoom():
    root_file = uproot.open(ROOT_FILE)
    source_values, source_edges = root_file[SOURCE_KEY].to_numpy()
    source_rebin, _, spectrum_x = rebin_five(source_values, source_edges)
    zoom = (spectrum_x >= 0.0575) & (spectrum_x <= 0.0725)
    central = (spectrum_x >= 0.060) & (spectrum_x < 0.070)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.35), sharex=True, sharey=True)
    for ax, (lane, info) in zip(axes, LANES.items()):
        pseudo_values, pseudo_edges = root_file[info["hist_key"]].to_numpy()
        expectation_values, expectation_edges = root_file[
            info["expectation_key"]
        ].to_numpy()
        pseudo_rebin, _, _ = rebin_five(pseudo_values, pseudo_edges)
        expectation_rebin, _, _ = rebin_five(
            expectation_values, expectation_edges
        )
        ax.step(
            1000.0 * spectrum_x[zoom],
            source_rebin[zoom],
            where="mid",
            color=COLORS["original"],
            linewidth=1.8,
            alpha=0.65,
            label="Original data",
        )
        ax.step(
            1000.0 * spectrum_x[zoom],
            pseudo_rebin[zoom],
            where="mid",
            color=info["color"],
            linewidth=1.5,
            label="Conditional pseudo-data",
        )
        ax.plot(
            1000.0 * spectrum_x[central],
            expectation_rebin[central],
            color=COLORS["ink"],
            linestyle="--",
            marker="o",
            markersize=2.8,
            linewidth=1.25,
            label="Replacement mean",
        )
        ax.axvspan(60.0, 70.0, color=COLORS["shade"], alpha=0.14, linewidth=0)
        ax.axvline(65.0, color=COLORS["shade"], linestyle=":", linewidth=0.9)
        ax.set_title(info["title"], color=info["color"], fontweight="semibold")
        ax.set_xlim(57.5, 72.5)
        ax.xaxis.set_major_locator(MultipleLocator(2.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.625))
        ax.set_xlabel("Invariant mass [MeV]")
        ax.legend(loc="lower left", frameon=False)
    axes[0].set_ylabel("Events / 0.625 MeV")
    fig.suptitle(
        "65 MeV replacement window: one fixed-seed background-only draw",
        fontsize=13.0,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_png = HERE / "plots" / "pseudo65_central_window_zoom.png"
    out_pdf = HERE / "plots" / "pseudo65_central_window_zoom.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def write_caption(table: pd.DataFrame) -> Path:
    rows = {row["sample"]: row for _, row in table.iterrows()}
    gp = rows["gp_mean"]
    func = rows["functional_form"]
    caption = (
        "Conditional 2021 10% central-window replacement study around 65 MeV. "
        "The requested continuous +/-2.5 sigma interval is "
        "[59.6958,70.3042] MeV; on the 0.625 MeV production grid it selects "
        "the same sixteen bin centers as the frozen v4.2 +/-2.25 sigma "
        "geometry. "
        "Only the 80 native bins with complete-bin edges [60,70) MeV were "
        "replaced; all other observed bins are identical to the source data. "
        "The left column uses an independent Poisson draw from the exact accepted "
        "v4.2 fixed-GP mean and the right column uses an independent Poisson draw "
        "from a sideband-only fGenGammaThresh fit. No signal was injected "
        "(Ainj=0). The standard v4.2 2021 scan was rerun over 50--250 MeV with "
        "0.625 MeV analysis bins, +/-2.25 sigma scan geometry, kmax=15, profiled "
        "background extraction, and asymptotic 90% CLs. At 65 MeV the GP lane "
        f"gives epsilon^2_90={gp['eps2_up_90cl']:.6g}, "
        f"p0={gp['p0_local_asymptotic']:.6g} (Z={gp['Z_local_asymptotic']:.3f}), "
        "while the functional-form lane gives "
        f"epsilon^2_90={func['eps2_up_90cl']:.6g}, "
        f"p0={func['p0_local_asymptotic']:.6g} "
        f"(Z={func['Z_local_asymptotic']:.3f}). "
        "The original v4.2 observed curves are shown only as context. No expected "
        "limit bands are constructed. Because the data outside the replacement "
        "window remain observed, these are conditional counterfactual scans, not "
        "independent global-null pseudoexperiments or coverage studies."
    )
    path = HERE / "plots" / "CAPTION.txt"
    path.write_text(caption + "\n")
    return path


def main() -> None:
    (HERE / "plots").mkdir(parents=True, exist_ok=True)
    set_style()
    scans, baseline = load_scans()
    table = build_m065_table(scans, baseline)
    outputs = draw_main_figure(scans, baseline)
    outputs.extend(draw_central_zoom())
    outputs.append(write_caption(table))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_sha256": sha256_file(ROOT_FILE),
        "baseline_csv": repo_relative(BASELINE_CSV),
        "baseline_csv_sha256": sha256_file(BASELINE_CSV),
        "reviewed_csvs": {
            lane: {
                "path": repo_relative(
                    HERE / "derived" / f"{lane}_results_reviewed.csv"
                ),
                "sha256": sha256_file(
                    HERE / "derived" / f"{lane}_results_reviewed.csv"
                ),
            }
            for lane in LANES
        },
        "outputs": {
            path.name: {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        },
        "no_expected_bands": True,
        "Ainj": 0.0,
    }
    path = HERE / "derived" / "plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
