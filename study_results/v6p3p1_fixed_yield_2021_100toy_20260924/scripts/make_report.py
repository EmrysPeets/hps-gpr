#!/usr/bin/env python3
"""Build figures and a standalone LaTeX report from the completed saved study.

No generation, fitting, or scientific tuning occurs in this script. Run
summarize.py first. Representative NPZ files, when present, are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hps-v6p3p1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHAPES = ["gaussian", "mc"]
LABELS = {"gaussian": "Gaussian", "mc": "Native MC"}
COLORS = {"gaussian": "#25659b", "mc": "#b14a32"}
MARKERS = {"gaussian": "o", "mc": "s"}
LEVELS = [0, 1, 3, 5]
plt.rcParams.update({"font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
    "grid.alpha": .2, "savefig.dpi": 190, "pdf.fonttype": 42,
    "lines.markersize": 3.1, "lines.linewidth": .9})


def save(fig, base, name):
    fig.savefig(base / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(base / "figures" / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


def style(ax, label, baseline=None, xlabel=True):
    ax.set(ylabel=label, xlim=(54, 246))
    ax.set_xticks([60, 100, 140, 180, 220, 240])
    if xlabel:
        ax.set_xlabel("Pole mass [MeV]")
    if baseline is not None:
        ax.axhline(baseline, color=".45", ls="--", lw=.75, zorder=-5)
    ax.grid(axis="y")


def curves(ax, data, field, error=None):
    for idx, shape in enumerate(SHAPES):
        d = data[data["shape"].eq(shape)].sort_values("mass_MeV")
        ax.errorbar(d.mass_MeV + (idx-.5)*1.6, d[field],
            yerr=d[error] if error is not None else None, color=COLORS[shape],
            marker=MARKERS[shape], label=LABELS[shape], capsize=1.6, elinewidth=.7)


def four_panels(base, data, field, error, label, baseline, name):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.5), constrained_layout=True)
    for ax, z in zip(axes.flat, LEVELS):
        curves(ax, data[data.z.eq(z)], field, error)
        ax.set_title(f"z = {z}", loc="left")
        style(ax, label, baseline)
    axes.flat[0].legend(frameon=False)
    save(fig, base, name)


def figures(base, pilot, data):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)
    curves(axes[0], pilot, "sigma_postfit_mean", "sigma_postfit_se")
    style(axes[0], r"Mean pilot yield error [candidates]")
    axes[0].legend(frameon=False)
    mc = pilot[pilot["shape"].eq("mc")].sort_values("mass_MeV")
    axes[1].plot(mc.mass_MeV, mc.mean_error_over_gaussian_reference, color=COLORS["mc"], marker="s")
    style(axes[1], "MC / Gaussian pilot mean error", 1.)
    save(fig, base, "pilot_reference")
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 3.8), constrained_layout=True)
    for col, z in enumerate([1, 3, 5]):
        for row, field, label in [(0, "raw_recovery", "Mean raw recovery"), (1, "paired_response", "Mean paired response")]:
            ax = axes[row, col]
            curves(ax, data[data.z.eq(z)], field + "_mean", field + "_se")
            ax.set_title(f"z = {z}", loc="left")
            style(ax, label if col == 0 else "", 1.)
            ax.set_xticks([60, 120, 180, 240])
    axes[0, 0].legend(frameon=False)
    save(fig, base, "recovery")
    four_panels(base, data, "bias_mean", "bias_se", r"Mean $\widehat A-A$ [candidates]", 0., "absolute_bias")
    four_panels(base, data, "error_ratio_mean", "error_ratio_se", r"Mean $\widehat{\sigma}_A/s_0$", 1., "error_response")
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.5), constrained_layout=True)
    for col, z in enumerate(LEVELS):
        for row, field, err, label, baseline in [(0, "pull_mean", "pull_se", "Mean pull", 0.),
                (1, "pull_sd", "pull_sd_bootstrap_se", "Sample pull width", 1.)]:
            ax = axes[row, col]
            curves(ax, data[data.z.eq(z)], field, err)
            ax.set_title(f"z = {z}", loc="left")
            style(ax, label if col == 0 else "", baseline)
            ax.set_xticks([60, 140, 240])
    axes[0, 0].legend(frameon=False, fontsize=7)
    save(fig, base, "pulls")
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.25), constrained_layout=True)
    for col, z in enumerate(LEVELS):
        for row, label, nominal in [(0, "68", .6827), (1, "95", .95)]:
            ax = axes[row, col]
            for idx, shape in enumerate(SHAPES):
                d = data[data.z.eq(z) & data["shape"].eq(shape)].sort_values("mass_MeV")
                y = d[f"contain{label}_fraction"]
                errors = np.stack([y-d[f"contain{label}_cp95_lo"], d[f"contain{label}_cp95_hi"]-y])
                ax.errorbar(d.mass_MeV + (idx-.5)*2.3, y, yerr=errors,
                    marker=MARKERS[shape], color=COLORS[shape], label=LABELS[shape], capsize=1.5, elinewidth=.65)
            ax.set_title(f"z = {z}", loc="left")
            style(ax, f"Nominal {float(nominal)*100:g}% containment" if col == 0 else "", nominal)
            ax.set_xticks([60, 140, 240]); ax.set_ylim(-.02, 1.025)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, base, "containment")


def spectrum_figures(base):
    """Supported fields: edges_GeV, background, signal, truth, gp_mean,
    clean_gp_mean, mask, plus scalar mass_MeV, shape, z, toy, A_expected.
    GP vectors may cover either the full support or window bins only.
    """
    paths = sorted((base / "results" / "representative").glob("*.npz"))
    if not paths:
        return []
    names = []
    for path in paths:
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        if not {"edges_GeV", "background", "signal", "truth", "gp_mean", "mask"} <= d.keys():
            continue
        edges = d["edges_GeV"] * 1000
        x = (edges[:-1] + edges[1:]) / 2
        b, sig, truth = d["background"], d["signal"], d["truth"]
        mask = d["mask"].astype(bool)
        gp = d["gp_mean"]; gpw = gp[mask] if len(gp) == len(x) else gp
        mass = int(d.get("mass_MeV", np.nan)); shape = str(d.get("shape", "mc"))
        fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.1), constrained_layout=True)
        axes[0].step(x, b+sig, where="mid", color=".45", lw=.65, label="Toy data")
        axes[0].plot(x, truth, color="k", lw=.8, label="Pinned background")
        axes[0].step(x, sig, where="mid", color=COLORS[shape], label="Realized signal")
        axes[0].set(yscale="symlog", xlabel="Mass [MeV]", ylabel="Candidates / bin", title="Full support")
        axes[0].axvspan(edges[np.flatnonzero(mask)[0]], edges[np.flatnonzero(mask)[-1]+1], color=COLORS[shape], alpha=.12)
        axes[0].legend(frameon=False, fontsize=6.5)
        axes[1].errorbar(x[mask], (b+sig)[mask], yerr=np.sqrt((b+sig)[mask]), fmt=".", color=".4", label="Toy data", ms=2)
        axes[1].plot(x[mask], truth[mask], color="k", label="Truth background")
        axes[1].plot(x[mask], gpw, color=COLORS[shape], label="GP after injection")
        if "clean_gp_mean" in d:
            null = d["clean_gp_mean"]; nullw = null[mask] if len(null) == len(x) else null
            axes[1].plot(x[mask], nullw, color="#558b68", ls="--", label="GP before injection")
            axes[2].plot(x[mask], gpw-nullw, color=COLORS[shape], label="GP change")
        axes[1].set(xlabel="Mass [MeV]", title="Extraction window", ylabel="Candidates / bin")
        axes[1].legend(frameon=False, fontsize=6.1)
        axes[2].step(x[mask], sig[mask], where="mid", color=".35", label="Realized signal")
        axes[2].plot(x[mask], ((b+sig)[mask]-gpw), color="#735f9c", lw=.65, label="Data minus GP")
        axes[2].axhline(0, color=".7", lw=.6)
        axes[2].set(xlabel="Mass [MeV]", title="Residual and GP response", ylabel="Candidates / bin")
        axes[2].legend(frameon=False, fontsize=6.1)
        for ax in axes:
            ax.grid(axis="y")
        name = f"spectrum_{path.stem}"
        save(fig, base, name)
        names.append((name, mass, shape, int(d.get("z", 5)), int(d.get("toy", 0))))
    return names


def num(value, digits=2):
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def texescape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def metric_range(data, name, shape=None, z=None, digits=2):
    d = data
    if shape is not None:
        d = d[d["shape"].eq(shape)]
    if z is not None:
        d = d[d.z.eq(z)]
    return f"{num(d[name].min(), digits)}--{num(d[name].max(), digits)}"


def image_tex(name, caption, width="\\textwidth"):
    return rf"\begin{{center}}\includegraphics[width={width}]{{../figures/{name}.pdf}}\end{{center}}" + "\n" + rf"{{\small {caption}\par}}" + "\n"


def create_source(base, meta, pilot, data, paired, reps):
    header = r"""\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype,fancyhdr,longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\pagestyle{fancy}\fancyhf{}\setlength{\headheight}{14pt}
\fancyhead[L]{\small HPS Gaussian-Process Resonance Search}
\fancyhead[R]{\small v6.3.1 / 2021 10\%}\fancyfoot[C]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}\renewcommand{\arraystretch}{1.12}
\newcommand{\Ah}{\widehat A}\newcommand{\sh}{\widehat\sigma}
\hypersetup{pdftitle={HPS-GPR v6.3.1: 2021 10 percent fixed-yield injection study},pdfauthor={Emrys Peets}}
\begin{document}
\begin{center}{\LARGE Fixed-yield injection and recovery}\\[5pt]
{\large 2021 10\%: independent pilot, native MC and Gaussian signals}\\[4pt]
Emrys Peets\quad---\quad24 September 2026\end{center}
"""
    s = [header]
    fail = int(meta["failed_rows"])
    s.append(r"\section*{Result and scope}")
    s.append(f"The study completed all {meta['completed_evaluation_cells']} evaluation cells with 100 attempted toys per cell: "
             f"{meta['evaluation_fit_valid']:,}/8,000 valid signed free fits and {meta['evaluation_profile_valid']:,}/8,000 valid truth-fixed profiles. "
             f"The independent pilot contains {meta['pilot_fit_valid']:,}/2,000 valid fits. "
             f"There are {fail} unresolved fit/profile rows in the failure ledger. ")
    s.append(r"The same expected full-selected signal yield $A=z s_0(m)$ is used for both shapes and every evaluation toy at each mass and level. "
             r"The Gaussian pilot mean returned yield error defines the frozen $s_0$; $z$ is an input reference-strength label, not an achieved significance.")
    s.append("At $z=5$, mean raw recovery ranges over the ten masses are " + metric_range(data, "raw_recovery_mean", "gaussian", 5) +
             " for Gaussian signals and " + metric_range(data, "raw_recovery_mean", "mc", 5) + " for native MC. " +
             "Mean paired response, which subtracts each toy's matched null fit, ranges from " + metric_range(data, "paired_response_mean", "gaussian", 5) +
             " and " + metric_range(data, "paired_response_mean", "mc", 5) + ", respectively. These ranges describe cells, not independent pooled measurements. The native-MC paired response is lower than the Gaussian response at every mass at $z=5$; the pointwise paired-difference intervals below quantify the difference. Equal expected candidate counts do not imply equal extraction difficulty.")
    s.append(r"\begin{center}\small\begin{tabular}{rcccccc}\toprule" "\n"
             r"& \multicolumn{2}{c}{Raw recovery} & \multicolumn{2}{c}{Paired response} & \multicolumn{2}{c}{Pull width}\\" "\n"
             r"$m$ [MeV] & Gaussian & MC & Gaussian & MC & Gaussian & MC\\\midrule" "\n")
    for mass in range(60, 241, 20):
        g = data[data.mass_MeV.eq(mass) & data.z.eq(5) & data["shape"].eq("gaussian")].iloc[0]
        m = data[data.mass_MeV.eq(mass) & data.z.eq(5) & data["shape"].eq("mc")].iloc[0]
        vals = [num(d[field]) for field in ["raw_recovery_mean", "paired_response_mean", "pull_sd"] for d in [g, m]]
        s.append(str(mass) + " & " + " & ".join(vals) + r"\\" + "\n")
    s.append(r"\bottomrule\end{tabular}\end{center}")
    s.append(r"{\small Summary at $z=5$. Recovery and response are averages of per-toy ratios. Full cell uncertainties and paired differences are supplied as CSV; figures below show standard errors. Pull width is the sample standard deviation, with $n-1$ denominator.\par}")
    s.append(r"This is a conditional study of one pinned background mean, fixed empirical MC templates and one archived-kernel GP prescription. "
             r"It measures fixed-yield bias, error response and nominal signed profile-set containment. It does not establish physical-background adequacy, unconditional coverage, calibrated discovery significance or an exclusion.")

    s.append(r"\clearpage\section*{Ensemble and frozen reference}")
    s.append(r"Two disjoint cohorts each contain 100 full-support Poisson background spectra, sampled from the bundled nominal 2021 GP arithmetic mean. "
             r"Each cohort reuses its backgrounds across all ten masses. Evaluation also reuses each background across both shapes and four levels. "
             r"The 8,000 evaluation rows are therefore correlated across cells; they do not represent 8,000 independent background experiments.")
    s.append(r"\begin{align*}B^{\rm pilot}_{ji},B^{\rm eval}_{ti}&\sim\operatorname{Poisson}(b_i),&"
             r"s_0(m)&=\frac1{100}\sum_{j=0}^{99}\sh^{\rm pilot}_{A,G,j}(m),\\"
             r"A(m,z)&=z s_0(m),&S_{tmzi}&\sim\operatorname{Poisson}(A(m,z)w_i).\end{align*}")
    s.append(r"The full probability vector includes bins below and above analysis support. Signal is added in all analysis bins, including GP training sidebands. "
             r"The realized full signal total fluctuates around $A$; expected and realized counts are recorded separately. Within each shape case, generation and extraction use identical full-selected-count probabilities. "
             r"No fit-window or support renormalization is applied.")
    s.append(image_tex("pilot_reference", r"Pilot mean returned yield errors (left; bars are standard errors of the 100 pilot errors) and their MC/Gaussian ratio (right). The Gaussian mean alone defines the common frozen yield scale. The ratio panel is descriptive; its sampling uncertainty is not shown."))
    s.append(r"\begin{center}\small\begin{tabular}{rrrrrr}\toprule"
             r"$m$ [MeV] & $s_0$ & SE$(s_0)$ & Gaussian CV [\%] & MC/Gaussian & $A(z=5)$\\\midrule" + "\n")
    for mass in range(60, 241, 20):
        g = pilot[pilot.mass_MeV.eq(mass) & pilot["shape"].eq("gaussian")].iloc[0]
        m = pilot[pilot.mass_MeV.eq(mass) & pilot["shape"].eq("mc")].iloc[0]
        s.append(f"{mass} & {g.s0:.2f} & {g.sigma_postfit_se:.2f} & {100*g.sigma_postfit_cv:.3f} & {m.mean_error_over_gaussian_reference:.3f} & {5*g.s0:.2f}" + r"\\" + "\n")
    s.append(r"\bottomrule\end{tabular}\end{center}")
    s.append(r"The pilot table is frozen before evaluation. Its standard error describes pilot precision; it is not added to a pull denominator or resampled during evaluation bootstraps.")

    s.append(r"\clearpage\section*{Raw recovery and paired response}")
    s.append(r"For every positive injection, two diagnostics retain different information:\["
             r"Q_t=\frac{\Ah_t(A)}{A},\qquad R_t=\frac{\Ah_t(A)-\Ah_t(0)}{A}.\]"
             r"Raw recovery retains any background-only yield offset. Paired response subtracts the signed null fit of the same background, mass and extraction template. "
             r"The subtraction is used only here; it is absent from raw yields, pulls and likelihood profiles.")
    s.append(image_tex("recovery", r"Cell means with standard errors from the toy scatter. The dashed line denotes unit recovery/response. MC and Gaussian receive equal expected full-selected yields at each mass and level; their independent Poisson signal draws need not have equal realized totals."))
    s.append(r"\subsection*{Paired MC--Gaussian response differences at $z=5$}")
    s.append(r"\begin{center}\small\begin{tabular}{rrrr}\toprule $m$ [MeV] & MC minus Gaussian & 95\% bootstrap interval & Complete pairs\\\midrule" + "\n")
    for mass in range(60, 241, 20):
        p = paired[paired.mass_MeV.eq(mass) & paired.z.eq(5) & paired.metric.eq("paired_response")].iloc[0]
        s.append(f"{mass} & {p.difference_mean:.3f} & [{p.bootstrap_ci95_lo:.3f}, {p.bootstrap_ci95_hi:.3f}] & {int(p.complete_pairs)}" + r"\\" + "\n")
    s.append(r"\bottomrule\end{tabular}\end{center}")
    s.append(r"The whole-toy bootstrap uses the same resampled evaluation IDs in every cell and preserves null partners. Each difference uses complete MC/Gaussian pairs; missing pairs are disclosed in the machine-readable table. These are pointwise intervals without a multiple-comparison adjustment.")

    s.append(r"\clearpage\section*{Absolute bias and returned errors}")
    s.append(r"Absolute bias is $\overline{\Ah-A}$ in full-selected candidates. Returned-error response is the average of $\sh_A/s_0$. "
             r"Neither the returned error nor $s_0$ is the empirical spread of fitted yields; the latter is saved separately as \texttt{Ahat\_sd}.")
    s.append(image_tex("absolute_bias", r"Mean absolute bias; bars are standard errors across valid fits. At $z=0$ these are the signed null offsets."))
    s.append(image_tex("error_response", r"Mean returned yield error divided by the frozen Gaussian reference. Bars are standard errors of the per-toy ratios. The dashed line is unity."))

    s.append(r"\clearpage\section*{Pull location and width}")
    s.append(r"The primary pull uses the fixed expected yield and each fit's returned profile-Hessian uncertainty:\[p_t=\frac{\Ah_t-A}{\sh_{A,t}}.\]"
             r"The realized Poisson signal total is not substituted for $A$. Unit pull width and zero pull mean are diagnostic reference values, not numerical acceptance criteria.")
    s.append(image_tex("pulls", r"Top: mean pull with toy-sample standard error. Bottom: sample pull width with one-standard-error whole-toy bootstrap bars (2,000 fixed-seed replicates). Dashed lines mark zero mean and unit width. The accompanying CSV includes pointwise 95\% percentile intervals for the widths."))
    s.append(r"\subsection*{Null offsets cannot be removed by changing the injection scale}")
    s.append(r"At zero signal, each template may project the background residual differently. The null fits remain signed. "
             r"Using a common expected yield changes the positive-injection ensemble; it does not alter either template's null bias.")
    s.append(r"\begin{center}\small\begin{tabular}{rcccc}\toprule & \multicolumn{2}{c}{Null mean pull} & \multicolumn{2}{c}{Null pull width}\\"
             r"$m$ [MeV] & Gaussian & MC & Gaussian & MC\\\midrule" + "\n")
    for mass in range(60, 241, 20):
        g = data[data.mass_MeV.eq(mass) & data.z.eq(0) & data["shape"].eq("gaussian")].iloc[0]
        m = data[data.mass_MeV.eq(mass) & data.z.eq(0) & data["shape"].eq("mc")].iloc[0]
        s.append(str(mass) + " & " + " & ".join(num(d[field]) for field in ["pull_mean", "pull_sd"] for d in [g, m]) + r"\\" + "\n")
    s.append(r"\bottomrule\end{tabular}\end{center}")

    s.append(r"\clearpage\section*{Nominal signed profile-set containment}")
    s.append(r"For each toy, nuisance coordinates are refitted at the fixed expected $A$, with the same GP mean/covariance used by that toy's free fit:\["
             r"q_{\rm true}=2\{\mathrm{NLL}(A,\widehat\theta_A)-\mathrm{NLL}(\Ah,\widehat\theta)\}.\]"
             r"Truth is counted inside the nominal 68.27\% and 95\% signed profile-likelihood sets when $q_{\rm true}\leq1$ and $q_{\rm true}\leq3.841459$, respectively. "
             r"These are signed sets, not a physical nonnegative-signal upper-limit construction.")
    s.append(image_tex("containment", r"Containment among numerically valid truth profiles. Bars are exact 95\% Clopper--Pearson intervals using that cell's valid-profile denominator. Dashed lines show nominal reference levels. Shared backgrounds correlate the cells; the intervals are pointwise."))
    s.append(r"\begin{center}\small\begin{tabular}{rrcccc}\toprule & & \multicolumn{2}{c}{68.27\% count range} & \multicolumn{2}{c}{95\% count range}\\"
             r"$z$ & attempted/cell & Gaussian & MC & Gaussian & MC\\\midrule" + "\n")
    for z in LEVELS:
        row = [str(z), "100"]
        for threshold in ["68", "95"]:
            for shape in SHAPES:
                d = data[data.z.eq(z) & data["shape"].eq(shape)]
                row.append(f"{int(d[f'contain{threshold}_k'].min())}--{int(d[f'contain{threshold}_k'].max())}")
        s.append(" & ".join(row) + r"\\" + "\n")
    s.append(r"\bottomrule\end{tabular}\end{center}")
    s.append(r"Ranges are over mass cells; they are not pooled proportions. Exact cell counts, denominators and intervals are in \texttt{evaluation\_summary.csv}. "
             r"For $k$ containing profiles and $f$ unresolved decisions, all-attempt accounting bounds are $[k/100,(k+f)/100]$. These bounds are not confidence intervals. "
             r"Successful-fit containment is conditional on numerical success; the failure ledger and valid fractions remain part of the result.")
    s.append(f"For this run, valid-profile denominators range from {int(data.profile_valid.min())} to {int(data.profile_valid.max())} out of 100. "
             r"With only 100 toys per cell, percentage-level claims require the reported binomial uncertainty; the shared-cell design cannot be treated as additional independent precision.")

    s.append(r"\clearpage\section*{Representative spectra and GP response}")
    s.append(r"Representative spectra are selected by fixed toy ID and declared masses, without selection by fit quality or recovery. Full-support plots show the added signal tails; window and residual panels show the response of GP conditioning to the same injected spectrum. "
             r"A plotted GP shift is a diagnostic, not a decomposition proving that every recovery difference arises from sideband absorption.")
    if len(reps) != 6 or {(r[1], r[2]) for r in reps} != {(m, shape) for m in [60, 140, 240] for shape in SHAPES}:
        raise ValueError("All six fixed representatives (60/140/240 MeV, Gaussian/MC) are required")
    for shape in SHAPES:
        if shape == "mc":
            s.append(r"\clearpage\section*{Representative spectra: native MC}")
            s.append(r"The same evaluation background (toy 0), masses, pole-centered masks and common expected yields are used as on the Gaussian page. Signal draws are independent across shapes. The nominal native-MC template is used both to generate and fit these signals.")
        else:
            s.append(r"\subsection*{Gaussian signals}")
        for name, mass, case_shape, z, toy in [r for r in reps if r[2] == shape]:
            s.append(image_tex(name, f"{mass} MeV, {LABELS[shape]}, $z={z}$, evaluation toy {toy}. The green dashed curve is the matched background-only GP prediction."))
        if shape == "mc":
            s.append(r"Native MC and Gaussian differences can include core offset, width, tails and pole-centered window acceptance. This run does not isolate these components. Clean-sideband or model-mismatch controls would be separately specified diagnostics.")

    s.append(r"\clearpage\section*{Procedure, numerical accounting and provenance}")
    s.append(r"All masses use the pole-centered window $m\pm2.25\sigma_m$, with the same extraction and training masks for both shapes. "
             r"The nominal resolution is $\sigma_m=0.00184825-0.001375m+0.085875m^2$, with $m$ and $\sigma_m$ both in GeV. Gaussian bin probabilities are CDF differences; native MC probabilities use the pinned histogram CDF with its inherited uniform-within-bin convention and outside-support categories.")
    s.append(r"The GP uses archived mass-dependent kernel parameters. Each spectrum recomputes its count-dependent log targets/noise, conditional mean and correlated covariance from bins outside the window. "
             r"Positive counts use $\log n$ and noise variance $1/n$; zero counts use target zero and variance one. This is GP conditioning with fixed kernel parameters, without hyperparameter optimization. "
             r"The inherited signed count-yield likelihood is\[\lambda_i=\widehat b_i+(L\theta)_i+A w_i,\qquad"
             r"\mathrm{NLL}=\sum_{i\in W}(\lambda_i-n_i\log\lambda_i)+\tfrac12\theta^\top\theta.\]"
             r"Poisson means remain positive. Returned uncertainties are observed profile-Hessian yield errors. Truth profiling holds the toy's GP constraint fixed, without further retraining or an extra random GP nuisance draw.")
    s.append(r"Every planned toy ID is retained, including failures. Bounded numerical retries reuse identical saved counts and are selected by convergence and objective value. "
             r"No toy is accepted by agreement with truth, pull or recovery. Tiny negative likelihood differences are clipped only after the inherited $q_{\rm true}\geq-2\times10^{-6}$ gate. "
             r"Exact acceptance thresholds, retry starts and watchdog settings are recorded in \texttt{protocol.json}; smoke/control checks and numerical attempt records accompany the run.")
    s.append(r"\par\textbf{Randomness and uncertainty.} The master seed is \texttt{63220260924}; namespaces 1, 2, 3 and 4 identify pilot backgrounds, evaluation backgrounds, independent signal draws and bootstrap replicates. "
             r"Bootstrap uncertainty resamples whole evaluation toy IDs, preserving all masses, shapes, levels and null partners. Means use standard errors from sample scatter. "
             r"MC--Gaussian contrasts use paired differences and disclose incomplete pairs. The 2,000-replicate bootstrap does not create extra independent toys or change the pilot reference.")
    s.append(r"\par\textbf{Boundary relative to v6.2.} This ensemble samples Poisson signal counts at fixed expected yields. The prior v6.2 ensemble sampled exact totals with multinomial bin allocation. "
             r"They are different experiments; numerical differences cannot be attributed to one code change without matching yields, counts, pairing, templates and fit settings.")
    s.append(r"\par\textbf{Limitations.} MC selection equivalence and signal-daughter association are unvalidated. Finite-MC, detector-response and source-estimation uncertainties are not propagated. "
             r"The generating background is one fixed mean. Containment here does not validate physical-background adequacy or a full hyperparameter-optimized production search, and it supplies no global significance, discovery probability or exclusion.")
    s.append(r"\par\textbf{Reproduction.} The package includes the immutable final handoff, input/code hashes, saved pilot/evaluation background cohorts, templates, frozen pilot table, per-toy rows and checkpoints, summary scripts, report source and manifest. "
             r"See \texttt{README.md}, \texttt{protocol.json}, \texttt{provenance/} and \texttt{MANIFEST.sha256}. The final handoff and frozen reference hashes are:")
    for label, p in [("Final handoff", base / "provenance/HANDOFF.md"), ("Frozen pilot reference", base / "pilot_reference.json")]:
        s.append(r"{\small " + label + r": \texttt{" + hashlib.sha256(p.read_bytes()).hexdigest() + r"}\par}")
    s.append(r"\end{document}")
    source = "\n".join(s)
    (base / "source/report.tex").write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    meta = json.loads((base / "results/summary.json").read_text())
    if not meta.get("all_planned_toy_ids_present"):
        raise ValueError("Report requires complete attempted-toy accounting")
    for path, expected in meta["source_hashes"].items():
        if hashlib.sha256((base/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after summarization: {path}")
    pilot = pd.read_csv(base / "results/pilot_summary.csv")
    data = pd.read_csv(base / "results/evaluation_summary.csv")
    paired = pd.read_csv(base / "results/paired_shape_comparisons.csv")
    (base / "figures").mkdir(exist_ok=True); (base / "source").mkdir(exist_ok=True)
    figures(base, pilot, data)
    reps = spectrum_figures(base)
    create_source(base, meta, pilot, data, paired, reps)
    if args.build:
        subprocess.run(["/opt/homebrew/bin/tectonic", "--only-cached", "--outdir", str(base / "pdf"), str(base / "source/report.tex")], cwd=base / "source", check=True)
    print(json.dumps({"source": str(base / "source/report.tex"), "representative_figures": len(reps), "built_pdf": args.build}))


if __name__ == "__main__":
    main()
