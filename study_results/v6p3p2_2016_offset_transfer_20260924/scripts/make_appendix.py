#!/usr/bin/env python3
"""Build the 2016 offset/response appendix from saved statistical summaries.

This read-only reporting stage neither generates toys nor refits a spectrum.
The standalone appendix starts at page 10. Packaging joins it to the unchanged
nine-page 2021 report in a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hps-v6p3p2-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MASSES = [42, 44, 60, 66, 76, 90, 92, 117, 160, 178]
EXPOSURES = [.1, 1.]
SOURCES = ["nominal", "stress"]
COLORS = {"nominal": "#25659b", "stress": "#ba6836"}
LABELS = {"nominal": "Nominal", "stress": "Stress"}
MARKERS = {"nominal": "o", "stress": "s"}
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.3,
    "axes.labelsize": 8.3, "axes.titlesize": 9,
    "legend.fontsize": 7.4, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65,
    "grid.alpha": .2, "savefig.dpi": 190, "pdf.fonttype": 42,
    "lines.markersize": 3., "lines.linewidth": .9,
})

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype,fancyhdr,longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\pagestyle{fancy}\fancyhf{}\setlength{\headheight}{14pt}
\fancyhead[L]{\small HPS Gaussian-Process Resonance Search}
\fancyhead[R]{\small v6.3.2 / 2016 offset transfer}\fancyfoot[C]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}\renewcommand{\arraystretch}{1.10}
\newcommand{\Ah}{\widehat A}\newcommand{\sh}{\widehat\sigma}
\newcommand{\CLs}{\mathrm{CL}_{s}}
\hypersetup{pdftitle={HPS-GPR v6.3.2: 2016 offset and response transfer appendix},pdfauthor={Emrys Peets}}
\begin{document}\setcounter{page}{10}
\begin{center}{\LARGE Appendix: offset and response transfer}\\[5pt]
{\large 2016 Gaussian signals at controlled exposure factors 0.1 and 1}\\[4pt]
Emrys Peets\quad---\quad24 September 2026\end{center}
"""


def save(fig, base, name):
    for suffix in ["pdf", "png"]:
        fig.savefig(base / "figures" / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def style(ax, ylabel, baseline=None):
    ax.set(xlabel="Pole mass [MeV]", ylabel=ylabel, xlim=(37, 183))
    ax.set_xticks([42, 76, 117, 160, 178])
    if baseline is not None:
        ax.axhline(baseline, color=".5", ls="--", lw=.7, zorder=-5)
    ax.grid(axis="y")


def number(value, digits=2):
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def escape(value):
    return (str(value).replace("\\", r"\textbackslash{}")
        .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        .replace("#", r"\#"))


def picture(name, caption, width=r"\textwidth"):
    return (rf"\begin{{center}}\includegraphics[width={width}]{{../figures/{name}.pdf}}\end{{center}}"
        + "\n" + rf"{{\small {caption}\par}}" + "\n")


def subsection(title):
    return r"\clearpage\section*{" + title + "}\n"


def verify_hashes(base, mapping):
    for relative, expected in mapping.items():
        path = base / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reporting source changed after analysis: {relative}")


def select(data, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, float):
            data = data[np.isclose(data[key], value)]
        else:
            data = data[data[key].eq(value)]
    return data.sort_values("mass_MeV")


def interval_range(data, field, digits=2):
    lo, hi=data[field].min(),data[field].max()
    if lo < 0 or hi < 0:
        return "["+number(lo,digits)+", "+number(hi,digits)+"]"
    return number(lo, digits) + "--" + number(hi, digits)


def line(ax, d, field, error=None, label=None, source="nominal", **kwargs):
    if len(d) == 0:
        return
    ax.errorbar(d.mass_MeV, d[field], yerr=d[error] if error else None,
        label=label, color=COLORS[source], marker=MARKERS[source],
        capsize=1.5, elinewidth=.65, **kwargs)


def make_figures(base, cal, scaling, diag, limits):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.2), constrained_layout=True)
    for col, source in enumerate(SOURCES):
        d = select(scaling, source=source)
        ax = axes[0, col]
        line(ax, d, "delta100", label="Full exposure", source=source)
        line(ax, d, "scaled_delta10", label=r"$10\delta_{10}$", source=source, ls="--", alpha=.6)
        style(ax, r"Null offset $\delta$ [candidates]", 0)
        ax.set_title(LABELS[source], loc="left"); ax.legend(frameon=False)
        ax = axes[1, col]
        line(ax, d, "pull100", label="Full exposure", source=source)
        line(ax, d, "scaled_pull10", label=r"$\sqrt{10}\,\bar p_{10}$", source=source, ls="--", alpha=.6)
        style(ax, "Mean null pull", 0); ax.legend(frameon=False)
    save(fig, base, "exposure_scaling")

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.8), constrained_layout=True)
    for row, exposure in enumerate(EXPOSURES):
        for col, source in enumerate(SOURCES):
            ax = axes[row, col]
            d = select(cal, source=source, exposure=exposure)
            ax.errorbar(d.mass_MeV, d.R, yerr=d.R_se, color=".2", ls="--", lw=1.,
                label=r"Calibration $R$ ($z=3$)", capsize=1.5)
            for z, ls, alpha in [(1, ":", .55), (3, "-", 1), (5, "-.", .75)]:
                d = select(diag, source=source, calibration_source=source,
                    exposure=exposure, method="raw", z=z)
                line(ax, d, "paired_response_mean", "paired_response_se",
                    label=f"Held-out z={z}", source=source, ls=ls, alpha=alpha)
            style(ax, "Paired response", 1)
            ax.set_title(f"{LABELS[source]}, {100*exposure:g}%", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=6.8, ncol=2)
    save(fig, base, "response_transfer")

    for field, error, name, label, reference in [
            ("pull_mean", "pull_se", "corrected_pull_means", "Mean pull", 0),
            ("pull_sd", "pull_sd_bootstrap_se", "corrected_pull_widths", "Pull width", 1)]:
        fig, axes = plt.subplots(2, 4, figsize=(7.15, 2.8), constrained_layout=False)
        for row, exposure in enumerate(EXPOSURES):
            for col, z in enumerate([0, 1, 3, 5]):
                ax = axes[row, col]
                for source in SOURCES:
                    for method, ls, alpha in [("raw", ":", .45), ("affine", "-", 1)]:
                        d = select(diag, source=source, calibration_source=source,
                            exposure=exposure, method=method, z=z)
                        line(ax, d, field, error, label=f"{LABELS[source]} {method}",
                            source=source, ls=ls, alpha=alpha)
                style(ax, label if col == 0 else "", reference)
                ax.set_xticks([42, 117, 178])
                ax.set_title(f"{100*exposure:g}%, z={z}", loc="left")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.02),
            ncol=4, frameon=False, fontsize=7., columnspacing=1.5)
        fig.tight_layout(rect=(0, 0, 1, .91), pad=.5, h_pad=.8, w_pad=.6)
        save(fig, base, name)

    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.5), constrained_layout=True)
    for row, exposure in enumerate(EXPOSURES):
        for col, z in enumerate([0, 1, 3, 5]):
            ax = axes[row, col]
            for source in SOURCES:
                d = select(limits, source=source, calibration_source=source,
                    exposure=exposure, method="rank_neyman90", z=z)
                y = d.acceptance_fraction
                err = np.array([y-d.acceptance_cp95_lo, d.acceptance_cp95_hi-y])
                ax.errorbar(d.mass_MeV, y, yerr=err, color=COLORS[source],
                    marker=MARKERS[source], label=LABELS[source], capsize=1.3, elinewidth=.6)
            style(ax, "Truth-grid acceptance" if col == 0 else "", .9)
            ax.set(ylim=(-.025, 1.03), xticks=[42, 117, 178])
            ax.set_title(f"{100*exposure:g}%, z={z}", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=6.3, loc="lower left")
    save(fig, base, "neyman_acceptance")


def table(headers, rows, align=None, size=r"\small"):
    align = align or "l" + "r" * (len(headers)-1)
    return (r"\begin{center}" + size + r"\begin{tabular}{" + align + r"}\toprule" + "\n"
        + " & ".join(headers) + r"\\\midrule" + "\n"
        + "\n".join(" & ".join(str(v) for v in row) + r"\\" for row in rows)
        + "\n" + r"\bottomrule\end{tabular}\end{center}" + "\n")


def create_source(base, cal, scaling, diag, limits, quantiles, meta):
    s = [PREAMBLE]
    s.append(r"\section*{Question, scope and independent cohorts}")
    s.append(r"\textbf{No automatic exposure transfer or production correction is justified.} "
        r"The simple null-offset scaling law is not exact in this conditional experiment. ")
    examples=[]
    for source,m in [("nominal",42),("stress",76)]:
        row=select(scaling,source=source,mass_MeV=m).iloc[0]
        if 'delta_difference_ci95_lo' in row:
            examples.append(f"{LABELS[source]} {m} MeV gives $\\delta_{{100}}-10\\delta_{{10}}={row.delta_difference:+.0f}$ candidates "
                +f"(pointwise 95\\% paired bootstrap interval [{row.delta_difference_ci95_lo:.0f}, {row.delta_difference_ci95_hi:.0f}])")
    if examples:s.append("; ".join(examples)+". ")
    s.append(r"Same-source calibration can center held-out toys; transfer to another generator requires separate validation. "
        r"The offset, signal response and finite-grid limits below are conditional on pinned sources, Gaussian generation/extraction and the archived-kernel GP prescription.")
    s.append(r"The masses are 42, 44, 60, 66, 76, 90, 92, 117, 160 and 178 MeV. "
        r"Exposure factors $L=0.1$ and $1$ scale the same full-exposure generating mean. "
        r"They are paired by Poisson thinning within a cohort and source; nominal and stress cohorts are independent. "
        r"Toy IDs are shared across masses and injection levels. Counts of fits below therefore do not count independent experiments.")
    s.append(table(["Cohort", "Sources", "$z$ levels", "Backgrounds/cell", "Free fits"], [
        ["Pilot", "Nominal", "0", "100", "2,000"],
        ["Calibration", "Nominal, stress", r"$0,1,\ldots,8$", "100", "36,000"],
        ["Evaluation", "Nominal, stress", "0, 1, 3, 5", "100", "16,000"],
    ], align="llcrr"))
    if meta:
        s.append("The saved run contains " + ", ".join(
            f"{meta[c+'_valid']:,}/{meta[c+'_rows']:,} valid {c} free fits"
            for c in ["pilot", "calibration", "evaluation"]) + ". "
            + f"Native full-exposure CLs limits are valid for {meta['native_cls90_valid']:,}/{meta['native_cls90_attempted']:,} attempts. "
            + f"The failure ledger has {meta['failure_rows']:,} rows. Numerical validity is separate from scientific closure.")
    s.append(r"The nominal pilot defines $s_0(m,L)$, the mean returned Gaussian yield error, independently at each exposure. "
        r"Signal-bin counts are Poisson with fixed expectation $A=z s_0$ and full-selected Gaussian probabilities. "
        r"Outside-support loss is retained; training-sideband injection is included; there is no window renormalization. "
        r"The same expected yield is used for nominal and stress generators. $z$ is a reference-strength label, not a significance.")
    s.append(r"\subsection*{Why the offset may become more visible}")
    s.append(r"In a local linear approximation, write the mean background residual as $Lr$ and the effective covariance as $LV$. "
        r"For a fixed signal vector $w$,\["
        r"\delta_L=\frac{w^\top(LV)^{-1}(Lr)}{w^\top(LV)^{-1}w}=L\delta_1,\qquad"
        r"\sigma_L\simeq\sqrt{L}\sigma_1,\qquad"
        r"\frac{\delta_L}{\sigma_L}\simeq\sqrt L\frac{\delta_1}{\sigma_1}.\]"
        r"Thus $\delta_{100}\simeq10\delta_{10}$ and $\bar p_{100}\simeq\sqrt{10}\bar p_{10}$. "
        r"These are testable approximations, not identities of the count-dependent GP and nonlinear likelihood. "
        r"Changing the kernel, mask, generator shape or effective covariance invalidates a simple exposure-only transfer.")
    s.append(r"The existing nine-page 2021 study is preserved unchanged. This appendix is a separate 2016 Gaussian experiment; "
        r"it does not revise the earlier MC/Gaussian measurements or turn their diagnostics into calibrated limits.")
    s.append(r"\textbf{Practical interpretation.} One observed 10\% fitted yield or residual is not an ensemble bias: "
        r"it includes counting noise and possible model mismatch. Transferring it by a factor of ten is not justified by the exposure label alone. "
        r"A production correction requires qualified background sources, selection-equivalent historical input and independent full-exposure validation, "
        r"with offset, response and width uncertainties propagated. This appendix changes no production limits.")

    s.append(subsection("Exposure scaling of the null offset"))
    s.append(r"Calibration null fits define $\widehat\delta_L=\overline{\Ah_L(0)}$. "
        r"The null pull is $p=\Ah/\sh_A$. Because the returned errors fluctuate, the mean pull need not equal "
        r"$\widehat\delta_L/\overline{\sh_A}$. Exposure differences preserve the nested background pairing.")
    s.append(picture("exposure_scaling", r"Calibration measurements. Solid: full exposure. Dashed: the 10\% estimate transferred using the stated factor. Nominal (blue) and stress (orange) are separate fixed generating means."))
    rows = []
    for mass in MASSES:
        row = [mass]
        for source in SOURCES:
            d = select(scaling, source=source, mass_MeV=mass).iloc[0]
            row.extend([f"{d.delta_difference:.1f} $\\pm$ {d.delta_difference_se:.1f}",
                f"{d.pull100-d.scaled_pull10:.2f} $\\pm$ {d.pull_difference_se:.2f}"])
        rows.append(row)
    s.append(table([r"$m$ [MeV]", r"Nominal $\Delta\delta$", r"Nominal $\Delta\bar p$", r"Stress $\Delta\delta$", r"Stress $\Delta\bar p$"], rows))
    s.append(r"{\small $\Delta\delta=\delta_{100}-10\delta_{10}$ (candidates), "
        r"$\Delta\bar p=\bar p_{100}-\sqrt{10}\bar p_{10}$. Errors are pointwise standard errors; "
        r"the paired bootstrap intervals are retained in \texttt{scaling\_comparisons.csv}.\par}")

    s.append(subsection("Freeze the response before testing the correction"))
    s.append(r"At each source, exposure and mass, the independent calibration cohort fixes\["
        r"\widehat\delta=\overline{\Ah(0)},\qquad"
        r"\widehat R=\frac{\overline{\Ah(3s_0)-\Ah(0)}}{3s_0},\qquad"
        r"\widetilde A=\frac{\Ah-\widehat\delta}{\widehat R}.\]"
        r"The $z=3$ definition is fixed in advance. Held-out $z=1,3,5$ paired responses test its transfer across strength. "
        r"A constant offset cancels exactly from paired response; subtracting it cannot restore a response slope below one.")
    s.append(picture("response_transfer", r"Frozen calibration $R$ (black dashed, standard-error bars) and held-out raw paired responses at three positive injection levels. Bars summarize toy variation; the independent calibration table is not refitted using evaluation toys."))
    rows = []
    for source in SOURCES:
        for exposure in EXPOSURES:
            c = select(cal, source=source, exposure=exposure)
            row = [LABELS[source], f"{100*exposure:g}\\%", interval_range(c,"R",3)]
            for z in [1,3,5]:
                d = select(diag, source=source, calibration_source=source, exposure=exposure, z=z, method="raw")
                row.append(interval_range(d,"paired_response_mean",3))
            rows.append(row)
    s.append(table(["Source", "Exposure", "Frozen $R$", "$z=1$", "$z=3$", "$z=5$"],rows))
    s.append(r"{\small Ranges span the ten correlated mass cells; they are not pooled estimates. Full cell values, complete-pair counts and uncertainty are in the CSV tables.\par}")
    correction_rows=[]
    for source in SOURCES:
        for z in [0,3]:
            row=[LABELS[source],z]
            for method in ["raw","offset_transfer","direct_offset","affine"]:
                d=select(diag,source=source,calibration_source=source,exposure=1.,z=z,method=method)
                row.append(interval_range(d,"pull_mean"))
            correction_rows.append(row)
    s.append(table(["Source","$z$","Raw","Transfer offset","Direct offset","Affine"],correction_rows,size=r"\footnotesize"))
    s.append(r"{\small Full-exposure mean-pull ranges across mass cells, using each source's own frozen calibration. Brackets also denote ranges, not confidence intervals.\par}")
    s.append(r"The offset-transfer diagnostic uses $10\widehat\delta_{10}$ at full exposure; direct-offset subtraction uses "
        r"the full-exposure calibration offset. The affine diagnostic also divides by the corresponding calibrated response. "
        r"None of these corrected estimators is inserted into the native profile likelihood or used to rescale a published upper limit.")

    s.append(subsection("Held-out pulls after the frozen correction"))
    s.append(r"For positive $\widehat R$, the statistical-only corrected error and pull are\["
        r"\sh_{\widetilde A}=\sh_A/\widehat R,\qquad "
        r"p_{\rm affine}=\frac{\Ah-\widehat\delta-\widehat R A}{\sh_A}.\]"
        r"These treat the calibration table as frozen. Its offset/response covariance is retained separately for approximate propagated uncertainty. "
        r"The calibration null scale $k_0=\operatorname{SD}[(\Ah(0)-\widehat\delta)/\sh_A]$ is also saved; "
        r"dividing by it is a separate diagnostic, not part of the primary pull.")
    s.append(picture("corrected_pull_means", r"Mean raw pulls (dotted, faint) and frozen affine-corrected pulls (solid). Blue: nominal; orange: stress. Bars are standard errors. Each source uses its own calibration."))
    s.append(picture("corrected_pull_widths", r"Corresponding sample pull widths. Bars are bootstrap standard errors. Zero mean and unit width are reference values; successful fits are not selected for agreement with them."))
    s.append(r"\textbf{Calibration transfer across generators.} Nominal calibration is additionally applied to independent stress toys at full exposure. "
        r"This is a misspecification check, distinct from each generator's own conditional closure. ")
    transfer = select(diag, source="stress", calibration_source="nominal", exposure=1., method="affine")
    if len(transfer):
        d = select(transfer, z=3)
        s.append("At $z=3$, transferred affine pull means span " + interval_range(d,"pull_mean") +
            " and widths span " + interval_range(d,"pull_sd") + ". ")
    s.append(r"Centering one pinned generator is therefore insufficient evidence that the unknown physical background has been corrected.")

    s.append(subsection("A finite-rank Neyman construction"))
    s.append(r"At each fixed grid value $A=z s_0$, $z=0,\ldots,8$, use the raw signed fitted yield $T=\Ah$ as an ordering statistic. "
        r"For 100 valid calibration toys, the lower-tail rank probability of a new spectrum is\["
        r"p_A(T)=\frac{1+\#\{j:T^{\rm cal}_{A,j}\leq T\}}{101}.\]"
        r"Reject that grid value when $p_A\leq0.10$. Exchangeability gives rejection probability at most $10/101$ "
        r"marginally over calibration and evaluation, with conservative ties. This is not a guarantee conditional on the one frozen calibration table.")
    s.append(picture("neyman_acceptance", r"Held-out acceptance of the injected grid value using the frozen same-generator table; exact 95\% Clopper--Pearson intervals use each cell's valid independent evaluation denominator. The dashed line is 90\%. Intervals are pointwise; cells share backgrounds."))
    if len(quantiles) and "acceptance_beta95_lo" in quantiles:
        lo = quantiles.acceptance_beta95_lo.dropna().iloc[0]
        hi = quantiles.acceptance_beta95_hi.dropna().iloc[0]
        s.append(f"For continuous $T$, the population lower-tail mass below the tenth calibration order statistic has the "
            r"$\mathrm{Beta}(10,91)$ distribution. Its central 95\% reference interval implies acceptance between "
            f"{100*lo:.1f}\\% and {100*hi:.1f}\\% before observing the calibration table. ")
    s.append(r"That order-statistic uncertainty is distinct from the held-out binomial interval and from the rank marginal size bound. "
        r"It is not an extra confidence interval to add to a pull error.")
    examples=[]
    for source in SOURCES:
        d=select(limits,source=source,calibration_source=source,exposure=1.,mass_MeV=76,z=5,method="rank_neyman90").iloc[0]
        examples.append(f"{LABELS[source].lower()} {int(d.acceptance_k)}/{int(d.acceptance_n)} "
            +f"(95\\% CP [{100*d.acceptance_cp95_lo:.1f}\\%, {100*d.acceptance_cp95_hi:.1f}\\%])")
    s.append("At full-exposure 76 MeV, $z=5$, same-source frozen acceptance is "+" and ".join(examples)+". These cells need not realize exactly 90\\% acceptance.")
    s.append(r"The accepted set is retained as a discrete set. Its reported upper envelope is the largest accepted grid value; "
        r"no interpolation or monotonic repair is applied. Empty sets, holes and acceptance of $z=8$ (right censoring) are explicit. "
        r"An empty set is assigned the physical reporting convention $U=0$ and remains flagged; hence its upper bound contains $A=0$ even if the rank test rejects that grid point. "
        r"Acceptance of the true grid point differs from containment by this upper envelope. The result supplies no off-grid coverage guarantee.")

    s.append(subsection("Toy-CLs diagnostic and native profile limits"))
    s.append(r"The finite-grid toy diagnostic uses the same raw-yield ordering and distributions from a calibration cohort independent of evaluation:\["
        r"\CLs^{\rm toy}(A)=\min\{1,p_A(T)/p_0(T)\}.\]"
        r"The same add-one rank probabilities are used in numerator and denominator. Reject at $\CLs^{\rm toy}\leq0.10$. "
        r"Its rejection set is a subset of the rank-$p_A$ rejection set and inherits the conservative matched-generator marginal bound. "
        r"With 100 calibration toys, this ratio has finite Monte Carlo resolution and substantial tail uncertainty; it is a diagnostic, "
        r"not a high-precision CLs calibration. Its accepted grid and censoring records are preserved without interpolation.")
    mismatch=select(limits,source="stress",calibration_source="nominal",exposure=1.,mass_MeV=76,z=5)
    if len(mismatch):
        n=select(mismatch,method="rank_neyman90").iloc[0]
        c=select(mismatch,method="rank_cls90").iloc[0]
        s.append(r"\textbf{A finite-MC floor can hide failed transfer.} At 76 MeV, full-exposure stress $z=5$ toys with nominal calibration have "
            +f"{int(n.acceptance_k)}/100 Neyman acceptance and {int(n.empty_sets)} empty sets, yet toy CLs accepts {int(c.acceptance_k)}/100 "
            +f"and has {int(c.right_censored)} right-censored endpoints. "
            +r"Here $p_A=p_0=1/101$ gives a ratio of one. This is finite-MC saturation, not successful calibration or a rescue of the native limit; native containment is also 0/100.")
    rows=[]
    for method, label in [("rank_neyman90","Neyman"),("rank_cls90","Toy CLs"),("native_profile_cls90","Native")]:
        for source in SOURCES:
            for exposure in EXPOSURES:
                d = select(limits, method=method, source=source, exposure=exposure,
                    calibration_source="none" if method=="native_profile_cls90" else source, z=3)
                if not len(d): continue
                rows.append([label, LABELS[source], f"{100*exposure:g}\\%",
                    interval_range(d,"acceptance_k",0), interval_range(d,"upper_coverage_k",0),
                    interval_range(d,"median_U_over_s0",2)])
    s.append(table(["Method", "Source", "Exposure", "Accept $A$", "Envelope contains $A$", r"Median endpoint/$s_0$"],rows,size=r"\footnotesize"))
    s.append(r"{\small At $z=3$: ranges across ten masses, with counts out of each cell's valid denominator (normally 100). "
        r"Native acceptance and containment are the same upper-limit decision. A reported grid endpoint of 8 is a lower endpoint for a right-censored row, not a measured finite limit.\par}")
    rows=[]
    for mass in MASSES:
        row=[mass]
        for source in SOURCES:
            for z in [3,5]:
                d=select(limits,method="native_profile_cls90",source=source,exposure=1.,mass_MeV=mass,z=z)
                if len(d):
                    r=d.iloc[0]; row.append(f"{int(r.upper_coverage_k)}/{int(r.upper_coverage_n)}")
                else: row.append("--")
        rows.append(row)
    s.append(table([r"$m$ [MeV]", "Nominal $z=3$", "Nominal $z=5$", "Stress $z=3$", "Stress $z=5$"],rows))
    s.append(r"{\small Full-exposure native 90\% profile-CLs upper-limit containment; each cell's exact 95\% Clopper--Pearson interval is supplied in \texttt{limit\_summary.csv}.\par}")
    s.append(r"The native calculation uses the original raw likelihood, with each toy's GP mean/covariance fixed during nuisance profiling. "
        r"Its denominator is the free fit for $\Ah\geq0$ and the $A=0$ fit otherwise; the GP mean is the null Asimov spectrum. "
        r"The inherited bounded-$\widetilde q$ asymptotic tails give the root $\CLs=0.10$. "
        r"These full-exposure limits are never obtained by correcting a limit with $\widehat\delta$ or $\widehat R$. "
        r"Toy-rank CLs and native profile CLs differ in ordering and approximation; agreement is an empirical result, not an identity.")

    s.append(subsection("Accounting, source boundaries and reproducibility"))
    s.append(r"\textbf{Set accounting.} The following ranges are across all same-generator mass/strength cells. "
        r"Counts record outcomes of 100 attempted evaluation toys per cell; they are not pooled rates. Complete per-cell valid counts, "
        r"truth acceptance, upper-envelope containment, exact binomial intervals and set flags are retained.")
    rows=[]
    for method,label in [("rank_neyman90","Neyman"),("rank_cls90","Toy CLs")]:
        for source in SOURCES:
            for exposure in EXPOSURES:
                d=select(limits,method=method,source=source,calibration_source=source,exposure=exposure)
                rows.append([label,LABELS[source],f"{100*exposure:g}\\%",interval_range(d,"valid",0),
                    interval_range(d,"empty_sets",0),interval_range(d,"holey_sets",0),interval_range(d,"right_censored",0)])
    s.append(table(["Method","Source","Exposure","Valid","Empty","Holey","Right-censored"],rows,size=r"\footnotesize"))
    historical_path=base/"provenance/historical10_manifest.json"
    historical=json.loads(historical_path.read_text()) if historical_path.exists() else {}
    historical_ratio=historical.get("support_count_ratio",.10220157794390532)
    s.append(r"\textbf{Generator provenance.} The nominal generator is the pinned 2016 GP arithmetic mean. "
        r"The archived conditional stress histogram retains the failed broad-component source-fit flag in its source manifest. "
        r"Neither is an independently validated physical-background model. Background-source estimation uncertainty is not sampled. "
        r"The exact $0.1$ exposure branch here is not the historical 2016 subset: its archived support-count ratio is "
        +f"{historical_ratio:.4f}, " +
        r"and exact luminosity fraction, selection equivalence and event overlap remain unverified.")
    s.append(r"\textbf{Fitting and failures.} The inherited signal resolution, pole-centered $\pm2.25\sigma_m$ masks and reviewed mass-dependent kernel parameters remain fixed. "
        r"Each spectrum recomputes count-dependent log-GP targets/noise, arithmetic mean and correlated covariance; no hyperparameter optimization is performed. "
        r"The signed-yield Poisson likelihood profiles the same correlated GP nuisance constraint. Numerical retries retain the original spectrum and toy ID. "
        r"Validity gates concern finite objectives, gradients and positive Poisson means, not closeness to truth or nominal coverage. Unresolved rows remain in the failure ledger.")
    s.append(r"The historical 2016 support/optimizer-qualification exception remains. Successful fixed-kernel conditional toys do not validate the original kernel optimization or repair that source qualification.")
    s.append(r"\textbf{Frozen calibration uncertainty.} Offset, response and their covariance come from calibration toys only. "
        r"Whole-toy resampling preserves null/injected partners, masses and nested exposure pairs. The primary held-out pull treats the table as fixed; "
        r"the separate approximate delta-method error also uses $k_0$ and is labeled accordingly; it does not recalibrate a likelihood interval. "
        r"The finite-rank guarantee assumes exchangeable calibration/evaluation at a fixed generator; it does not establish physical-source coverage.")
    s.append(r"\textbf{Reproduction.} The package saves the numerical protocol, pilot reference, independent cohorts, calibration table, per-toy rows, "
        r"summary tables, failed-row ledger, source hashes, vector figures and this LaTeX source. "
        r"\texttt{scripts/analyze.py} computes summaries; \texttt{scripts/make\_appendix.py --build} regenerates this appendix without fitting. "
        r"The original 2021 PDF is copied and joined only during final packaging. \texttt{README.md} gives the validated commands. "
        r"See the protocol and QA files for exact seeds, worker limits, completeness and final validation status.")
    if meta:
        s.append(f"The master seed is \\texttt{{{meta['master_seed']}}}; uncertainty summaries use "
            f"{meta['bootstrap_replicates']:,} whole-toy bootstrap replicates. Calibration and evaluation bootstrap streams are separate. "
            r"The deterministic mean-spectrum fits in \texttt{asimov\_rows.csv} are not additional toys.")
    s.append(r"\textbf{Claim boundary.} This study can measure conditional offset scaling, response transfer and finite-toy performance under two pinned generators. "
        r"It cannot establish unconditional coverage over unknown backgrounds, physical-model adequacy, calibrated discovery/global significance or an experimental exclusion.")
    literature=base/"provenance/literature.json"
    if literature.exists():
        pdg=json.loads(literature.read_text()).get("PDG",{})
        if pdg.get("url"):
            s.append(r"\textbf{Reference.} Particle Data Group, \emph{Statistics} (2026), sections 40.2 and 40.4.2, "
                r"\url{"+pdg["url"]+r"}. General bias and Neyman-construction context; the exposure approximation and finite-rank convention are defined explicitly above.")
    s.append(r"\end{document}")
    (base/"source/appendix.tex").write_text("\n\n".join(s))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=BASE)
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--tectonic",default="/opt/homebrew/bin/tectonic")
    args=parser.parse_args(); base=args.base.resolve()
    names=["calibration_parameters","scaling_comparisons","evaluation_diagnostics","limit_summary","calibration_quantiles"]
    frames=[pd.read_csv(base/"results"/(name+".csv")) for name in names]
    meta_path=base/"results/summary.json"
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and not meta.get("all_toy_ids_accounted"):
        raise ValueError("Final reporting requires complete attempted-toy accounting")
    if meta.get("source_hashes"):
        verify_hashes(base,meta["source_hashes"])
    for directory in ["source","figures","pdf"]:
        (base/directory).mkdir(exist_ok=True)
    make_figures(base,*frames[:4])
    create_source(base,*frames,meta)
    if args.build:
        subprocess.run([args.tectonic,"--only-cached","--outdir",str(base/"pdf"),str(base/"source/appendix.tex")],
            cwd=base/"source",check=True)
    print(json.dumps({"source":str(base/"source/appendix.tex"),"built_pdf":args.build,
        "input_sha256":{name+".csv":hashlib.sha256((base/"results"/(name+".csv")).read_bytes()).hexdigest() for name in names}}))


if __name__=="__main__":
    main()
