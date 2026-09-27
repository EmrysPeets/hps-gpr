#!/usr/bin/env python3
"""Render saved v6.2 injection results. This script performs no fits or toys."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE / "qa" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

LEVELS = [1000, 5000, 10000, 30000]
MASSES = list(range(60, 261, 20))
METHODS = ["pole_centered", "core_shifted"]
NAMES = {"pole_centered": "Pole-centered", "core_shifted": "MC-core-shifted"}
COLORS = {"pole_centered": "#245b8a", "core_shifted": "#bd583a"}
CONTROL_NAMES = {"contaminated_gp": "Injected sidebands", "clean_sidebands": "Clean sidebands", "known_background": "Known background"}
CONTROL_COLORS = {"contaminated_gp": "#245b8a", "clean_sidebands": "#33927a", "known_background": "#955b9e"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65, "grid.alpha": .22,
    "savefig.dpi": 180, "pdf.fonttype": 42})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(base):
    toys = pd.read_csv(base / "results/toys.csv")
    summary = pd.read_csv(base / "results/summary.csv")
    paired = pd.read_csv(base / "results/paired.csv").rename(columns={"mass": "mass_MeV", "N": "injected_N"})
    for frame in (toys, summary, paired):
        frame["mass_MeV"] = frame["mass_MeV"].astype(int)
        frame["injected_N"] = frame["injected_N"].astype(int)
    required = {(m, n, method) for m in MASSES for n in LEVELS for method in METHODS}
    primary = summary[(summary.control == "contaminated_gp") & summary.injected_N.isin(LEVELS)]
    keys = set(primary[["mass_MeV", "injected_N", "method"]].itertuples(index=False, name=None))
    if keys != required or len(primary) != 88 or not (primary.n == 20).all():
        raise ValueError("Primary results must contain exactly 20 toys in all 88 mass/yield/method cells")
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
        ax.set_title(f"{n:,} selected MC events per toy", loc="left")
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
        style_mass(ax, "Mean fitted yield [events]", 0)
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
    axs[0, 1].text(.05, .74, "Upper left: a positive value means\nthe shifted mask extracts more signal.\n\nBelow: subtract each method's paired\nzero-injection fit before dividing by N.\nThis removes its toy-specific baseline\noffset, but does not correct bias.\n\nThe two estimates share the same\nbackground and MC injection draws.",
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
            ax.set(xlabel="Pull", ylabel="Toys / bin", ylim=(0, 21))
            ax.set_yticks([0, 5, 10, 15, 20])
            ax.tick_params(labelsize=8)
            if m == 260:
                ax.set_facecolor(".95")
        axs.flat[-1].axis("off")
        handles, labels = axs.flat[0].get_legend_handles_labels()
        axs.flat[-1].legend(handles, labels, frameon=False, loc="center", fontsize=10)
        figs[f"gallery_{n}"] = save(fig, base, f"pull_gallery_N{n:05d}")
    return figs


class Report:
    width, height = 612, 792
    left, right = 48, 564
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(self.width, self.height))
        self.c.setTitle("HPS-GPR v6.2: MC injection and signal recovery")
        self.c.setAuthor("HPS-GPR analysis studies")
        self.page = 0
        self.body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.3, leading=14.1,
                                   textColor=colors.HexColor("#182634"), spaceAfter=7)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=8.6, leading=11.4)
        self.head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=12, leading=15,
                                   textColor=colors.HexColor("#245b8a"))
        self.y = 0

    def new_page(self, title, section="MC injection and recovery"):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(colors.HexColor("#245b8a"))
        self.c.setFont("Helvetica", 8)
        self.c.drawString(self.left, 760, "HPS-GPR  /  VERSION 6.2")
        self.c.drawRightString(self.right, 760, section.upper())
        self.c.setStrokeColor(colors.HexColor("#bac9d6"))
        self.c.line(self.left, 749, self.right, 749)
        self.c.setFillColor(colors.HexColor("#182634"))
        self.c.setFont("Times-Bold", 22)
        self.c.drawString(self.left, 716, title)
        self.c.setFont("Helvetica", 8)
        self.c.setFillColor(colors.HexColor("#5d6973"))
        self.c.drawString(self.left, 30, "23 September 2026  |  Conditional selected-MC toy study")
        self.c.drawRightString(self.right, 30, str(self.page))
        self.y = 692

    def paragraph(self, text, small=False):
        p = Paragraph(text, self.small if small else self.body)
        _, h = p.wrap(self.right - self.left, 700)
        if self.y - h < 52:
            raise RuntimeError(f"Page {self.page} overflow: {text[:80]}")
        p.drawOn(self.c, self.left, self.y - h)
        self.y -= h + (7 if small else 10)

    def subhead(self, text):
        p = Paragraph(text, self.head)
        _, h = p.wrap(self.right - self.left, 50)
        p.drawOn(self.c, self.left, self.y - h)
        self.y -= h + 8

    def picture(self, path, max_height):
        im = ImageReader(str(path)); iw, ih = im.getSize()
        w = self.right - self.left
        h = w * ih / iw
        if h > max_height:
            w *= max_height / h; h = max_height
        self.c.drawImage(im, (self.width - w) / 2, self.y - h, width=w, height=h, mask="auto")
        self.y -= h + 12

    def table(self, rows, widths, font_size=8.2, row_height=19):
        table = Table(rows, colWidths=widths, rowHeights=[24] + [row_height] * (len(rows) - 1))
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245b8a")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f7")]),
            ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#bac9d6")),
        ]))
        w, h = table.wrap(516, 680)
        if self.y - h < 52:
            raise RuntimeError(f"Table overflow on page {self.page}")
        table.drawOn(self.c, self.left, self.y - h)
        self.y -= h + 13

    def save(self):
        self.c.save()


def fmt(x, digits=2):
    return f"{float(x):.{digits}f}"


def make_report(base, toys, summary, paired, primary, figs):
    r = Report(base / "pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf")
    native = primary[primary.mass_MeV < 260]
    r.new_page("MC injection and signal recovery", "Study overview")
    r.paragraph("<b>Scope.</b> Exact draws of 1,000, 5,000, 10,000 and 30,000 selected candidates from the supplied smeared 2021 MC distributions are injected into 20 background toy experiments at each available generated mass from 60 through 260 MeV. The 40 MeV sample is excluded. Two prespecified masks compare extraction at the generated mass with extraction around the MC core.")
    r.subhead("Main finding")
    high = native[native.injected_N == 30000]
    inc = paired[(paired.injected_N == 30000) & (paired.mass_MeV < 260)]
    r.paragraph(f"The shifted mask reduces signal loss at the largest injection, but does not close the recovery test. At N = 30,000, after subtracting the paired zero-injection fit, the mean recovered increment is {inc.mean_incremental_recovery_pole.min():.3f}-{inc.mean_incremental_recovery_pole.max():.3f} of N with the pole-centered mask and {inc.mean_incremental_recovery_core.min():.3f}-{inc.mean_incremental_recovery_core.max():.3f} with the shifted mask over 60-240 MeV. The clean-sideband controls identify signal in GP training bins as a material source of the loss.")
    rows = [["Primary, 60-240 MeV", "Pole-centered", "MC-core-shifted"]]
    for n in LEVELS:
        values = []
        for method in METHODS:
            d = native[(native.injected_N == n) & (native.method == method)]
            values.append(f"{d.mean_recovery.min():.3f} to {d.mean_recovery.max():.3f}")
        rows.append([f"N = {n:,}: recovery range", *values])
    r.table(rows, [220, 148, 148], font_size=9, row_height=24)
    r.paragraph("The table gives the range of mean fitted/injected yield across the ten masses in the original 60-240 MeV direct-template comparison. It is a descriptive range, not a confidence interval. Pointwise Monte Carlo errors and the separate 260 MeV extension appear in the figures.", small=True)
    r.subhead("Interpretation boundary")
    r.paragraph("This is a conditional test of the supplied all-selected candidate shapes and a pinned background truth. The MC contains broad and displaced components, and v6.1 did not establish a pure truth-associated resonance response or complete production-selection equivalence. Recovery of these templates does not establish physical signal efficiency, a coupling limit, global significance or detector-calibrated interval coverage.")
    r.paragraph("Only 20 toys populate each cell. Pull widths and interval-inclusion fractions are imprecise. Exact-N multinomial signal draws differ from the fitted Poisson signal model; even the known-background pull width need not be one. The template and MC center are fixed: finite-MC and detector-response uncertainties are not propagated.")
    r.paragraph(f"Saved fit rows: {len(toys):,}. Primary nonzero-injection cells: {len(primary)}, each with 20 fits. Reproducible ledgers: results/toys.csv, summary.csv and paired.csv. Study construction and input identities are recorded in protocol.json and the provenance files.", small=True)

    r.new_page("Toy construction and fit protocol", "Methods")
    blocks = [
        ("Signal draws and normalization", "Each injection draws exactly N entries from the archived, unit-weight, all-selected MC histogram, retaining categories outside the 36-299.75 MeV analysis support. Within-support histogram draws are mapped to the analysis bins. No additional smearing, translation, truth selection or within-window renormalization is applied. The fitted amplitude is in full-selected events: the template sum in the support and in the extraction mask retains the corresponding fractions. Actual support/window counts and expected fractions are saved per toy."),
        ("Background truth and the paired ensemble", "The background truth is the pinned nominal 2021 all-data GP prediction from v5.8.2. Poisson background fluctuations generate 20 toys at each mass; the same background draw is reused across injection levels and both methods. Within each level, both methods use the same histogram injection. The zero-injection baseline permits a paired incremental-response diagnostic."),
        ("The two extraction masks", "The pole-centered method places the fit/training exclusion mask at m0. The shifted method places it at a fixed MC-only core center c(m0), inherited from v6.1 where available. Both use the unchanged half-width 2.25 times the nominal resolution at m0. Only the mask moves. The MC template remains at its reconstructed masses, and kernel coordinates remain attached to generated mass m0. The extension at 260 MeV uses the archived 250 MeV kernel state and is marked separately."),
        ("Three prespecified extraction conditions", "Injected sidebands are the primary condition: signal entering training bins can affect the GP prediction. Clean sidebands train on the same toy's background-only sidebands while the extraction bins retain the injected signal. The known-background control fixes the background to its generating expectation. In the GP conditions, archived hyperparameters are fixed but the conditional mean and correlated covariance are recomputed for each toy; the likelihood includes the correlated background constraint."),
        ("Pulls and interval inclusion", "The signed fitted yield Ahat is retained for a null diagnostic without a nonnegative-yield boundary. A pull is (Ahat - N)/sigma_A, with sigma_A from the profiled likelihood curvature. The profile likelihood ratio at the injected truth gives interval-inclusion indicators for nominal 68.27% (q at most 1) and 95% one-parameter thresholds. Saved 95% Clopper-Pearson intervals quantify the 20-trial sampling uncertainty in those inclusion fractions. These are conditional diagnostic inclusions, not a certification of physical coverage."),
    ]
    for title, text in blocks:
        r.subhead(title); r.paragraph(text, small=True)

    r.new_page("Recovered yield across generated mass")
    r.picture(figs["recovery"], 475)
    r.paragraph("<b>Primary injected-sideband result.</b> Each point is the average fitted full-selected yield divided by the exact injection N. Error bars are the sample standard error across 20 toys. The dashed line denotes exact mean recovery. The gray band marks 260 MeV, outside the earlier 60-240 MeV direct-template comparison and using the archived 250 MeV kernel state.")
    r.paragraph("The mean contains any baseline extraction offset as well as the response to the injected signal. The paired zero-injection subtraction on page 8 separates these effects descriptively. Full-selected yield includes draws outside the analysis support; the template fractions make this normalization explicit.", small=True)

    r.new_page("Mean pulls")
    r.picture(figs["pull_mean"], 475)
    r.paragraph("Each point is the mean of (Ahat - N)/sigma_A in the primary injected-sideband condition. Error bars are the sample pull standard deviation divided by sqrt(20). A zero mean is the unbiased reference for this conditional ensemble; points sharing toys are correlated.")
    r.paragraph("A near-zero pull mean alone does not show reliable extraction: it must be read with yield recovery, the pull width, the zero-injection control and the fit's quoted uncertainty. With only 20 toys, small deviations are not evidence for a calibrated failure or success probability.", small=True)

    r.new_page("Pull widths")
    r.picture(figs["pull_width"], 475)
    r.paragraph("The plotted width is the sample standard deviation of the 20 pulls, using one degree of freedom for the estimated mean. No Gaussian fit to a pull histogram is used. The dashed width-one line is a reference, not an expectation guaranteed by this ensemble.")
    r.paragraph("The injected signal has an exact total N and multinomial fluctuations among bins; the fitted signal likelihood is Poisson. The generating background is fixed conditionally, while GP prediction uncertainty enters the fit. These choices can change widths even in an otherwise consistent estimator. Twenty-toy width estimates are noisy, and the per-mass count galleries preserve their visible distribution shapes.", small=True)

    r.new_page("Zero-injection background diagnostic")
    r.picture(figs["null"], 475)
    r.paragraph("With N = 0, the expected signal yield is zero. Points show the mean signed extracted yield; bars show its sample standard error across the same 20 background toys. The primary and known-background conditions separate background-prediction effects from the fixed-truth likelihood control.")
    r.paragraph("Clean and injected GP sidebands coincide at zero injection, so a duplicate clean-sideband null is unnecessary. A baseline offset can dominate fractional recovery at small injections. The paired incremental response subtracts each toy's corresponding zero-injection fit.", small=True)

    r.new_page("Controls at 30,000 injected events")
    r.picture(figs["controls"], 475)
    r.paragraph("Both masks are shown with three conditions using the same injected toys. Clean sidebands remove the injected component from GP training while preserving it in the extraction bins. Known background fixes the generating expectation and removes GP prediction uncertainty. These are explanatory controls, not candidate production prescriptions.")
    asimov = pd.read_csv(base / "results/asimov.csv")
    example = asimov[(asimov.mass_MeV == 240) & (asimov.injected_N == 30000)]
    def asimov_values(control):
        return "/".join(f"{example[(example.control == control) & (example.method == method)].Ahat.iloc[0]:,.0f}" for method in METHODS)
    r.paragraph(f"A deterministic mean-count (Asimov) example at 240 MeV and N = 30,000 yields {asimov_values('contaminated_gp')} events for injected sidebands, {asimov_values('clean_sidebands')} for clean sidebands, and {asimov_values('known_background')} for known background (pole/shifted). The loss thus persists without Poisson toy fluctuations. Known background also removes the GP nuisance constraint, so its comparison is not a single-factor replacement for the GP fit.", small=True)
    control30 = summary[(summary.injected_N == 30000) & (summary.mass_MeV < 260)]
    clean = control30[control30.control == "clean_sidebands"]
    oracle = control30[control30.control == "known_background"]
    r.paragraph(f"Across both masks and 60-240 MeV, mean recovery is {clean.mean_recovery.min():.3f}-{clean.mean_recovery.max():.3f} with clean sidebands and {oracle.mean_recovery.min():.3f}-{oracle.mean_recovery.max():.3f} with known background. These descriptive ranges support the sideband-absorption mechanism while retaining baseline and finite-toy limitations.", small=True)

    r.new_page("Paired mask and incremental response")
    r.picture(figs["paired"], 480)
    r.paragraph("The upper panel compares shifted and pole-centered yields within each identical toy; error bars use the standard error of the paired difference. Lower panels show the mean fitted increment after subtracting that same toy's zero-injection fit. These quantities retain the full-selected normalization.")
    r.paragraph("Baseline subtraction is a diagnostic only. It is not applied to the primary estimator and does not remove signal absorbed by sideband training. A larger extracted yield does not by itself establish a better mask; bias, dispersion and interval inclusion must be assessed together.", small=True)

    for n in LEVELS:
        r.new_page(f"Pull distributions: N = {n:,}", "Twenty-toy count galleries")
        r.picture(figs[f"gallery_{n}"], 558)
        r.paragraph("Primary injected-sideband fits. Each outline contains exactly 20 pulls. Both methods share the bin edges within each mass panel; axis ranges may differ between masses. The counts are deliberately not normalized to a smooth density or fitted Gaussian. The shaded 260 MeV panel is the separate extension.", small=True)

    for n in LEVELS:
        r.new_page(f"Numerical ledger: N = {n:,}", "Primary-cell appendix")
        r.paragraph("All primary injected-sideband cells, including the marked 260 MeV extension. P = pole-centered; C = MC-core-shifted. Mean uncertainty is the average profiled curvature uncertainty. Inclusion columns give toy counts out of 20.", small=True)
        rows = [["m [MeV]", "Mask", "Mean A", "Mean s", "A/N", "Pull mean", "Pull SD", "68%", "95%"]]
        d = primary[primary.injected_N == n]
        for m in MASSES:
            for method in METHODS:
                row = d[(d.mass_MeV == m) & (d.method == method)].iloc[0]
                rows.append([f"{m}" + (" *" if m == 260 else ""), "P" if method == "pole_centered" else "C",
                    f"{row.mean_A:,.0f}", f"{row.mean_sigma:,.0f}", fmt(row.mean_recovery, 3), fmt(row.pull_mean),
                    fmt(row.pull_width), str(int(row.profile68_count)), str(int(row.profile95_count))])
        r.table(rows, [57, 37, 71, 65, 52, 68, 62, 52, 52], font_size=8.0, row_height=19.7)
        r.paragraph("* 260 MeV extends the original direct-template study and uses the archived 250 MeV kernel state. It is excluded from the overview's 60-240 MeV range summaries.", small=True)
        r.paragraph("The 68%/95% columns are conditional profile-likelihood inclusion counts at the injected truth. With 20 trials the fraction changes in steps of 0.05. Exact binomial Clopper-Pearson intervals, mean-bias values, sample errors and all controls are retained in results/summary.csv. These counts are not calibrated detector coverage.", small=True)
    r.save()
    return r.page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    base = args.base.resolve()
    for directory in ["figures", "pdf", "qa", "source"]:
        (base / directory).mkdir(parents=True, exist_ok=True)
    toys, summary, paired, primary = load_inputs(base)
    figs = make_figures(base, toys, summary, paired, primary)
    pages = make_report(base, toys, summary, paired, primary, figs)
    from pypdf import PdfReader
    pdf = base / "pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf"
    reader = PdfReader(str(pdf))
    text = "\n\n".join(p.extract_text() for p in reader.pages)
    (base / "qa/report_extracted_text.txt").write_text(text)
    (base / "qa/report_build.json").write_text(json.dumps({
        "pdf": str(pdf.relative_to(base)), "pages": len(reader.pages), "expected_pages": pages,
        "pdf_sha256": sha256(pdf), "input_sha256": {str(p.relative_to(base)): sha256(p)
            for p in [base / "results/toys.csv", base / "results/summary.csv", base / "results/paired.csv", base / "results/asimov.csv"]},
        "fit_rows": len(toys), "primary_cells": len(primary), "rendered_visual_qa": "pending",
        "created_by": "scripts/make_report.py; no fits or toy generation",
    }, indent=2) + "\n")
    print(f"Wrote {pdf} ({pages} pages); rendering and visual QA are required.")


if __name__ == "__main__":
    main()
