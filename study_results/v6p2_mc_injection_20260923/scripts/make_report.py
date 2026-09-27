#!/usr/bin/env python3
"""Generate v6.2 LaTeX tables and vector figures from saved results; no fits or toys."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE / "qa" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVELS = [1000, 5000, 10000, 30000]
MASSES = list(range(60, 261, 20))
METHODS = ["pole_centered", "core_shifted"]
NAMES = {"pole_centered": "Pole-centered", "core_shifted": "MC-core-shifted"}
COLORS = {"pole_centered": "#216ca1", "core_shifted": "#a64139"}
CONTROL_NAMES = {"contaminated_gp": "Injected sidebands", "clean_sidebands": "Clean sidebands", "known_background": "Known background"}
CONTROL_COLORS = {"contaminated_gp": "#245b8a", "clean_sidebands": "#33927a", "known_background": "#955b9e"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65, "grid.alpha": .22,
    "savefig.dpi": 180, "pdf.fonttype": 42})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(base, expected_toys=40):
    toys = pd.read_csv(base / "results/toys.csv")
    summary = pd.read_csv(base / "results/summary.csv")
    paired = pd.read_csv(base / "results/paired.csv").rename(columns={"mass": "mass_MeV", "N": "injected_N"})
    for frame in (toys, summary, paired):
        frame["mass_MeV"] = frame["mass_MeV"].astype(int)
        frame["injected_N"] = frame["injected_N"].astype(int)
    required = {(m, n, method) for m in MASSES for n in LEVELS for method in METHODS}
    primary = summary[(summary.control == "contaminated_gp") & summary.injected_N.isin(LEVELS)]
    keys = set(primary[["mass_MeV", "injected_N", "method"]].itertuples(index=False, name=None))
    if keys != required or len(primary) != 88 or not (primary.n == expected_toys).all():
        raise ValueError(f"Primary results must contain exactly {expected_toys} toys in all 88 mass/yield/method cells")
    if toys.duplicated(["mass_MeV", "injected_N", "toy", "method", "control"]).any():
        raise ValueError("Duplicate toy result key")
    if not np.isfinite(toys[["Ahat", "sigma_A", "pull"]].to_numpy()).all():
        raise ValueError("Nonfinite reported primary fit coordinate")
    return toys, summary, paired, primary


def style_mass(ax, ylabel, baseline=None):
    ax.set(xlabel="Generated mass [MeV]", ylabel=ylabel, xlim=(55, 269))
    ax.set_xticks([60, 100, 140, 180, 220, 260])
    ax.axvspan(250, 269, color="0.94", zorder=-5)
    ax.grid(axis="y")
    if baseline is not None:
        ax.axhline(baseline, color="0.35", lw=.8, ls="--", zorder=-2)


def save(fig, base, name):
    fig.savefig(base / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(base / "figures" / (name + ".png"), bbox_inches="tight")
    plt.close(fig)
    return base / "figures" / (name + ".png")


def four_panels(base, primary, field, error, ylabel, baseline, name):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6.2), constrained_layout=True)
    for ax, n in zip(axs.flat, LEVELS):
        for j, method in enumerate(METHODS):
            d = primary[(primary.injected_N == n) & (primary.method == method)].sort_values("mass_MeV")
            ax.errorbar(d.mass_MeV + (j - .5) * 1.4, d[field],
                        yerr=d[error] if error else None, color=COLORS[method],
                        label=NAMES[method], marker="o" if j == 0 else "s", ms=3.5,
                        lw=1.1, capsize=2, elinewidth=.7)
        style_mass(ax, ylabel, baseline)
        ax.set_title(f"{n:,} selected candidates per toy", loc="left")
    axs.flat[0].legend(frameon=False)
    return save(fig, base, name)


def make_figures(base, toys, summary, paired, primary):
    figs = {}
    figs["recovery"] = four_panels(base, primary, "mean_recovery", "recovery_se", "Mean fitted / injected yield", 1, "recovery")
    figs["pull_mean"] = four_panels(base, primary, "pull_mean", "pull_mean_se", "Mean pull", 0, "pull_mean")
    figs["pull_width"] = four_panels(base, primary, "pull_width", None, "Sample pull standard deviation", 1, "pull_width")
    fig, axs = plt.subplots(2, 1, figsize=(8, 6.2), constrained_layout=True)
    for ax, control in zip(axs, ["contaminated_gp", "known_background"]):
        for j, method in enumerate(METHODS):
            d = summary[(summary.injected_N == 0) & (summary.control == control) & (summary.method == method)].sort_values("mass_MeV")
            ax.errorbar(d.mass_MeV + (j - .5) * 1.4, d.mean_A, yerr=d.sd_A / np.sqrt(d.n),
                        color=COLORS[method], label=NAMES[method], marker="os"[j], ms=4, lw=1, capsize=2)
        style_mass(ax, "Mean fitted yield [candidates]", 0)
        ax.set_title(CONTROL_NAMES[control] + ": zero injected signal", loc="left")
    axs[0].legend(frameon=False)
    figs["null"] = save(fig, base, "null_bias")

    fig, axs = plt.subplots(2, 2, figsize=(8, 6.2), constrained_layout=True)
    for j, method in enumerate(METHODS):
        for control in CONTROL_NAMES:
            d = summary[(summary.injected_N == 30000) & (summary.method == method) & (summary.control == control)].sort_values("mass_MeV")
            for i, (field, error) in enumerate([("mean_recovery", "recovery_se"), ("pull_mean", "pull_mean_se")]):
                axs[i, j].errorbar(d.mass_MeV, d[field], yerr=d[error], color=CONTROL_COLORS[control],
                    label=CONTROL_NAMES[control], marker="o", ms=3, lw=1, capsize=2)
        axs[0, j].set_title(NAMES[method], loc="left")
        style_mass(axs[0, j], "Mean fitted / injected yield", 1)
        style_mass(axs[1, j], "Mean pull", 0)
    axs[0, 0].legend(frameon=False)
    figs["controls"] = save(fig, base, "controls_30000")

    fig, axs = plt.subplots(2, 2, figsize=(8, 6.2), constrained_layout=True)
    palette = plt.get_cmap("viridis")(np.linspace(.1, .85, 4))
    for n, col in zip(LEVELS, palette):
        d = paired[paired.injected_N == n].sort_values("mass_MeV")
        axs[0, 0].errorbar(d.mass_MeV, d.mean_core_minus_pole / n,
            yerr=d.se_core_minus_pole / n, label=f"{n:,}", color=col, marker="o", ms=3, capsize=2, lw=1)
        for j, method in enumerate(["pole", "core"]):
            axs[1, j].plot(d.mass_MeV, d["mean_incremental_recovery_" + method],
                          color=col, label=f"{n:,}", marker="o", ms=3, lw=1)
    style_mass(axs[0, 0], "(Shifted - pole) mean yield / N", 0)
    axs[0, 0].set_title("Paired extraction difference", loc="left")
    axs[0, 0].legend(title="Injected N", frameon=False, ncol=2)
    axs[0, 1].axis("off")
    axs[0, 1].text(.05, .92, "Same toy, two masks", ha="left", va="top", fontsize=13, transform=axs[0, 1].transAxes)
    axs[0, 1].text(.05, .74, "Upper left: a positive value means\nthe shifted mask extracts more signal.\n\nBelow: subtract each method's paired\nzero-injection fit before dividing by N.\nThis separates the incremental response\nfrom the zero-injection offset.\n\nThe two estimates share the same\nbackground and MC injection draws.",
                  ha="left", va="top", linespacing=1.45, fontsize=10, transform=axs[0, 1].transAxes)
    for ax, method in zip(axs[1], METHODS):
        style_mass(ax, "Mean [A(N) - A(0)] / N", 1)
        ax.set_title(NAMES[method] + ": incremental response", loc="left")
    figs["paired"] = save(fig, base, "paired_comparison")

    for n in LEVELS:
        fig, axs = plt.subplots(4, 3, figsize=(8, 8.3), constrained_layout=True)
        for ax, m in zip(axs.flat, MASSES):
            d = toys[(toys.control == "contaminated_gp") & (toys.injected_N == n) & (toys.mass_MeV == m)]
            vals = d.pull.to_numpy()
            lo, hi = min(-3.0, float(vals.min()) - .25), max(3.0, float(vals.max()) + .25)
            bins = np.linspace(lo, hi, 13)
            for method in METHODS:
                v = d[d.method == method].pull
                ax.hist(v, bins=bins, histtype="step", color=COLORS[method], lw=1.3, label=NAMES[method])
            ax.axvline(0, color=".4", lw=.6, ls="--")
            ax.set_title(f"{m} MeV" + (" (extension)" if m == 260 else ""), loc="left", fontsize=10)
            largest = max(np.histogram(d[d.method == method].pull, bins=bins)[0].max() for method in METHODS)
            ymax = max(10, 5 * int(np.ceil((largest + 1) / 5)))
            ax.set(xlabel="Pull", ylabel="Toys / bin", ylim=(0, ymax))
            ax.set_yticks(np.arange(0, ymax + 1, 5))
            ax.tick_params(labelsize=8)
            if m == 260:
                ax.set_facecolor(".95")
        axs.flat[-1].axis("off")
        handles, labels = axs.flat[0].get_legend_handles_labels()
        axs.flat[-1].legend(handles, labels, frameon=False, loc="center", fontsize=10)
        figs[f"gallery_{n}"] = save(fig, base, f"pull_gallery_N{n:05d}")
    return figs


def tex_table(base, name, header, rows, align):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (base / "source/generated" / (name + ".tex")).write_text("\n".join(lines) + "\n")


def interval(values, digits=3):
    return f"{values.min():.{digits}f}--{values.max():.{digits}f}"


def make_tables(base, toys, summary, paired, primary):
    native = primary[primary.mass_MeV < 260]
    high = native[native.injected_N == 30000]
    inc = paired[(paired.injected_N == 30000) & (paired.mass_MeV < 260)]
    n = int(primary.n.iloc[0])
    control_high = summary[(summary.injected_N == 30000) & (summary.mass_MeV < 260)]
    macro = {"ToyCount": str(n), "TotalFitRows": f"{len(toys):,}", "PrimaryFitRows": f"{len(primary)*n:,}",
             "UniqueInjectedToys": f"{len(MASSES)*len(LEVELS)*n:,}", "UniqueNullToys": f"{len(MASSES)*n:,}",
             "IncPoleRange": interval(inc.mean_incremental_recovery_pole),
             "IncCoreRange": interval(inc.mean_incremental_recovery_core),
             "RawPoleRange": interval(high[high.method == "pole_centered"].mean_recovery),
             "RawCoreRange": interval(high[high.method == "core_shifted"].mean_recovery),
             "CleanRange": interval(control_high[control_high.control == "clean_sidebands"].mean_recovery),
             "KnownRange": interval(control_high[control_high.control == "known_background"].mean_recovery),
             "PullMeanRange": interval(primary.pull_mean, 2), "PullWidthRange": interval(primary.pull_width, 2),
             "ContainmentMin": str(int(primary.profile95_count.min())),
             "ContainmentMax": str(int(primary.profile95_count.max())),
             "ContainmentStep": f"{100/n:g}",
             "WidthRelativeSE": f"{100/np.sqrt(2*(n-1)):.0f}"}
    for method, tag in [("pole_centered", "Pole"), ("core_shifted", "Core")]:
        point = high[(high.mass_MeV == 240) & (high.method == method)].iloc[0]
        macro["Example" + tag + "Yield"] = f"{point.mean_A:,.0f}"
        macro["Example" + tag + "Recovery"] = f"{point.mean_recovery:.3f}"
        macro["Example" + tag + "Pull"] = f"{point.pull_mean:.2f}"
        macro["Example" + tag + "Inclusion"] = str(int(point.profile95_count))
        macro["Example" + tag + "InclusionInterval"] = f"{point.profile95_cp_low:.2f}--{point.profile95_cp_high:.2f}"
    asimov = pd.read_csv(base / "results/asimov.csv")
    ex = asimov[(asimov.mass_MeV == 240) & (asimov.injected_N == 30000)]
    for control, tag in [("contaminated_gp", "Injected"), ("clean_sidebands", "Clean"), ("known_background", "Known")]:
        macro["Asimov" + tag] = "/".join(f"{ex[(ex.control == control) & (ex.method == method)].Ahat.iloc[0]:,.0f}" for method in METHODS)
    old_path = base / "history/20_toy_release/results/paired.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        merged = inc.merge(old[(old.injected_N == 30000) & (old.mass_MeV < 260)],
                           on=["mass_MeV", "injected_N"], suffixes=("_new", "_old"), validate="one_to_one")
        changes = [np.abs(merged[f"mean_incremental_recovery_{method}_new"] - merged[f"mean_incremental_recovery_{method}_old"]).max()
                   for method in ["pole", "core"]]
        macro["IncrementStabilityPercent"] = f"{100*max(changes):.2f}"
    (base / "source/generated/summary_macros.tex").write_text(
        "% Generated from saved CSV ledgers; do not edit numerical values here.\n" +
        "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macro.items()) + "\n")
    rows = []
    for N in LEVELS:
        d = native[native.injected_N == N]
        rows.append([f"{N:,}", *[interval(d[d.method == method].mean_recovery) for method in METHODS]])
    tex_table(base, "recovery_ranges", [r"Injected $N$", "Pole-centered", "MC-core-centered"], rows, "rrr")
    for N in LEVELS:
        rows = []
        d = primary[primary.injected_N == N]
        for mass in MASSES:
            for method in METHODS:
                row = d[(d.mass_MeV == mass) & (d.method == method)].iloc[0]
                rows.append([str(mass) + (r"$^{*}$" if mass == 260 else ""), "P" if method == "pole_centered" else "C",
                    f"{row.mean_A:,.0f}", f"{row.mean_sigma:,.0f}", f"{row.mean_recovery:.3f}",
                    f"{row.pull_mean:.2f}", f"{row.pull_width:.2f}", str(int(row.profile68_count)), str(int(row.profile95_count))])
        tex_table(base, f"primary_N{N:05d}", [r"$m_0$", "Mask", r"$\overline{\widehat A}$", r"$\overline{\sigma_A}$",
            r"$R$", r"$\overline p$", r"$s_p$", r"$k_{68}$", r"$k_{95}$"], rows, "rlrrrrrrr")
    centers = pd.read_csv(base / "results/centers.csv")
    rows = []
    for mass in MASSES:
        row = centers[centers.mass_MeV == mass].iloc[0]
        d = primary[(primary.mass_MeV == mass) & (primary.injected_N == 1000)]
        fractions = [d[d.method == method].expected_window_fraction.iloc[0] for method in METHODS]
        rows.append([str(mass) + (r"$^{*}$" if mass == 260 else ""), f"{row.core_center_MeV:.3f}",
                     f"{row.nominal_sigma_MeV:.3f}", f"{row.support_fraction:.3f}",
                     *[f"{x:.3f}" for x in fractions]])
    tex_table(base, "centers_fractions", [r"$m_0$", r"$c(m_0)$", r"$\sigma_m$", r"$f_D$", r"$f_{W,\mathrm{P}}$", r"$f_{W,\mathrm{C}}$"], rows, "rrrrrr")


def metadata(base, recorded_pdf=False):
    from pypdf import PdfReader
    paths = [base / "results" / (name + ".csv") for name in ["toys", "summary", "paired", "asimov", "centers"]]
    data = {"format": "LaTeX article; vector PDF figures", "created_by": "scripts/make_report.py; no fits or toy generation",
            "input_sha256": {str(p.relative_to(base)): sha256(p) for p in paths},
            "source_sha256": {str(p.relative_to(base)): sha256(p) for p in sorted((base / "source").rglob("*.tex"))},
            "figure_pairs": 10, "rendered_visual_qa": "pending"}
    if recorded_pdf:
        pdf = base / "pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf"
        reader = PdfReader(pdf)
        data.update(pdf=str(pdf.relative_to(base)), pages=len(reader.pages), pdf_sha256=sha256(pdf))
        text = "\n\n".join(page.extract_text() for page in reader.pages)
        (base / "qa/report_extracted_text.txt").write_text(text)
    (base / "qa/report_build.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--record-pdf", action="store_true", help="Record the just-compiled PDF without regenerating assets")
    args = parser.parse_args()
    base = args.base.resolve()
    for directory in ["figures", "pdf", "qa", "source/generated"]:
        (base / directory).mkdir(parents=True, exist_ok=True)
    if not args.record_pdf:
        toys, summary, paired, primary = load_inputs(base)
        make_figures(base, toys, summary, paired, primary)
        make_tables(base, toys, summary, paired, primary)
    metadata(base, recorded_pdf=args.record_pdf)
    print("Recorded compiled LaTeX PDF." if args.record_pdf else "Generated vector figures and LaTeX tables from 40-toy results.")


if __name__ == "__main__":
    main()
