"""Figures and LaTeX for the frozen v5.6.1 limit-echo extension.

Only reads completed scans and echo summaries; never fits or samples.
Run after scan_limits.py and the echo summary.  Compile source/main.tex
from its own directory.  --source-only reuses the existing figure PDFs.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter

B = Path(__file__).resolve().parents[1]
D, F = B / "derived", B / "figures"
REGIONS = ["65", "75", "92", "120", "160", "210"]
COLORS = {"one": "#216391", "ten": "#B56829"}
DARK, GRAY, PURPLE = "#153C51", "#737C82", "#946AA3"
PALETTE = ["#236B95", "#B36830", "#38836C", "#A54A62", "#7166A4",
           "#6C763D", "#258F9B", "#A86F85", "#6F665B"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65,
    "lines.linewidth": 1.4, "savefig.dpi": 160,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tex(s):
    return str(s).translate(str.maketrans({"\\": r"\textbackslash{}",
        "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}))


def lane_text(lane, math=False):
    if math:
        return r"$1\%\times100$" if lane == "one" else r"$10\%\times10$"
    return "1% x100" if lane == "one" else "10% x10"


def ordered(cat):
    records = []
    for region in REGIONS:
        for lane in ("one", "ten"):
            records.extend(cat[(cat.region.astype(str) == region) &
                               (cat.lane == lane)].to_dict("records"))
    records.extend(cat[~cat.region.astype(str).isin(REGIONS)]
                   .sort_values("mass_MeV").to_dict("records"))
    assert len(records) == len(cat)
    return pd.DataFrame(records)


def savefig(fig, name, png=False):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
    if png:
        fig.savefig(F / f"{name}.png", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def clean(ax):
    ax.grid(axis="y", color="#E0E5E8", lw=.55, zorder=0)
    ax.tick_params(length=3, width=.6)


def scans(sid):
    folder = D / "scans" / sid
    names = ["background_asimov", "matched_asimov", "yield_asimov"]
    names += [f"toy_{i:02d}" for i in range(20)]
    out = {key: pd.read_csv(folder / f"{key}.csv") for key in names}
    masses = out["background_asimov"].mass_MeV.to_numpy()
    required = ["mass_MeV", "A90", "signed_r", "epsilon2_90", "epsilon2_90_fixed_density"]
    for name, q in out.items():
        assert set(required).issubset(q.columns), (sid, name)
        assert np.array_equal(q.mass_MeV.to_numpy(), masses), (sid, name, "mass grid")
        assert (q.A90 > 0).all() and (q.epsilon2_90 > 0).all()
        assert np.isfinite(q[required]).all().all()
    return out


def echo_rows(echo, sid):
    a = echo[echo.scenario == sid]
    assert set(a.side) == {"left", "right"} and len(a) == 2
    return {side: a[a.side == side].iloc[0] for side in ("left", "right")}


def extent(row):
    if not np.isfinite(row.get("ratio90_lo_MeV", np.nan)):
        return "--"
    return f"{row.ratio90_lo_MeV:g}--{row.ratio90_hi_MeV:g}"


def qrange(row, stem):
    if stem == "mass":
        med, lo, hi = [float(row[f"toy_{q}_mass"]) for q in ("median", "q16", "q84")]
        return f"{med:.1f} [{lo:.1f}, {hi:.1f}]"
    med, lo, hi = [float(row[f"toy_ratio_fixed_{q}"]) for q in ("median", "q16", "q84")]
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"


def overview(cat):
    fig, axs = plt.subplots(2, 1, figsize=(7.1, 4.9), sharex=True, sharey=True,
                            layout="constrained")
    for ax, lane in zip(axs, ("ten", "one")):
        subset = cat[cat.lane == lane]
        for i, (_, r) in enumerate(subset.iterrows()):
            a = pd.read_csv(D / "scans" / r.scenario / "matched_asimov.csv")
            extra = str(r.region).startswith("extra")
            label = f"{r.mass_MeV:g} MeV" + (" (extra)" if extra else "")
            ax.plot(a.mass_MeV, a.epsilon2_90, color=PALETTE[i],
                    lw=1.15, ls="--" if extra else "-", label=label)
        ax.set(yscale="log", xlim=(50, 250), ylabel=r"Conditional $\epsilon^2_{90}$")
        ax.set_title(f"{lane_text(lane)}: matched-Asimov injections at the labelled masses",
                     pad=42 if lane == "ten" else 56)
        ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower center",
                  bbox_to_anchor=(.5, 1.01), columnspacing=1.9, borderaxespad=0.)
        ax.yaxis.set_minor_formatter(NullFormatter())
        clean(ax)
    axs[1].set_xlabel("Tested mass [MeV]")
    savefig(fig, "fullrange_overview", png=True)


def scenario_plot(r, echo):
    allscans = scans(r.scenario)
    er = echo_rows(echo, r.scenario)
    bg = allscans["background_asimov"]
    mass = bg.mass_MeV.to_numpy()
    offset = (mass-float(r.mass_MeV))/float(r.sigma_MeV)
    flank = (np.abs(offset) >= 2) & (np.abs(offset) <= 8)
    toy_array = np.array([allscans[f"toy_{i:02d}"].epsilon2_90 for i in range(20)])
    fig = plt.figure(figsize=(7.1, 5.95), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1])
    ax_top = fig.add_subplot(gs[0, :])
    ax_ratio = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1])
    for y in toy_array:
        ax_top.plot(mass, y, color=COLORS[r.lane], lw=.65, alpha=.26)
    ax_top.plot([], [], color=COLORS[r.lane], alpha=.6, lw=1, label="20 saved toys")
    for name, color, ls, label in (
        ("background_asimov", GRAY, "-", "Background Asimov"),
        ("yield_asimov", PURPLE, "--", "Yield-scaled Asimov"),
        ("matched_asimov", DARK, "-", "Matched Asimov"),
    ):
        a = allscans[name]
        ax_top.plot(mass, a.epsilon2_90, color=color, ls=ls,
                    lw=1.55 if name == "matched_asimov" else 1.2, label=label)
        ratio = a.A90.to_numpy()/bg.A90.to_numpy()
        ax_ratio.plot(offset, np.where(flank, ratio, np.nan), color=color, ls=ls,
                      lw=1.55 if name == "matched_asimov" else 1.2)
        ax_r.plot(offset, np.where(flank, a.signed_r, np.nan), color=color,
                  ls=ls, lw=1.55 if name == "matched_asimov" else 1.2)
    for i in range(20):
        a = allscans[f"toy_{i:02d}"]
        ratio = a.A90.to_numpy()/bg.A90.to_numpy()
        ax_ratio.plot(offset, np.where(flank, ratio, np.nan),
                      color=COLORS[r.lane], lw=.6, alpha=.18, zorder=1)
    ax_top.axvline(float(r.mass_MeV), color="#A5AEB3", lw=.7, ls=":")
    ax_top.set(yscale="log", xlim=(mass[0], mass[-1]),
               xlabel="Tested mass [MeV]", ylabel=r"Conditional $\epsilon^2_{90}$")
    ax_top.set_title("Full scan: standard prompt-density conversion for each spectrum", pad=30)
    ax_top.legend(frameon=False, ncol=4, loc="lower center", fontsize=8.5,
                  bbox_to_anchor=(.5, 1.01), columnspacing=1.0,
                  handlelength=1.8, borderaxespad=0.)
    ax_top.yaxis.set_minor_formatter(NullFormatter())
    ax_ratio.axhline(1., color=GRAY, lw=.75)
    ax_ratio.axhline(.9, color="#9C8685", lw=.75, ls=":")
    ax_ratio.set(xlim=(-8, 8), ylim=(0, None), xlabel=r"$(m-m_0)/\sigma_m(m_0)$",
                 ylabel=r"$A_{90}/A_{90}^{\rm background}$", title="Limit ratio in the two search flanks")
    ax_r.axhline(0, color=GRAY, lw=.75)
    ax_r.set(xlim=(-8, 8), xlabel=r"$(m-m_0)/\sigma_m(m_0)$",
             ylabel=r"Signed profile score $r$", title="Asimov fit response in the flanks")
    for side, e in er.items():
        if e.status == "outside_scan":
            ax_ratio.text(.97, .94, "Right flank\noutside scan", transform=ax_ratio.transAxes,
                          ha="right", va="top", fontsize=9, color=GRAY)
            continue
        u = float(e.offset_sigma)
        ax_ratio.scatter([u], [float(e.echo_ratio)], color="#A53C53", marker="D", s=27, zorder=8)
        ax_ratio.annotate(f"{e.echo_mass_MeV:g} MeV", (u, float(e.echo_ratio)),
                          xytext=(-5 if side == "left" else 5, 8),
                          textcoords="offset points", ha="right" if side == "left" else "left",
                          va="bottom", fontsize=9, color="#922F43",
                          bbox={"facecolor":"white", "edgecolor":"none", "alpha":.86, "pad":1.2})
        ax_r.axvline(u, color="#A53C53", lw=.7, ls=":")
    for ax in (ax_ratio, ax_r):
        ax.axvspan(-2, 2, color="#E3E8EB", alpha=.75, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    for ax in (ax_top, ax_ratio, ax_r):
        clean(ax)
    savefig(fig, r.scenario)
    a = allscans["matched_asimov"]
    e2_ratio = a.epsilon2_90.to_numpy()/bg.epsilon2_90.to_numpy()
    count_ratio = a.A90.to_numpy()/bg.A90.to_numpy()
    return dict(scenario=r.scenario,rows_per_scan=len(mass),toy_curves=20,
                scan_min_MeV=float(mass[0]),scan_max_MeV=float(mass[-1]),
                matched_max_abs_density_ratio_change=float(np.max(abs(e2_ratio-count_ratio))),
                minimum_matched_signed_r=float(a.signed_r.min()),
                echo_markers=[{side:float(e.echo_mass_MeV)} for side,e in er.items()
                              if e.status != "outside_scan"])


PREAMBLE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.66in,top=0.60in,bottom=0.61in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,amssymb,booktabs,array,graphicx,xcolor}
\usepackage{fancyhdr,enumitem,hyperref,url}
\definecolor{ink}{HTML}{153C51}
\definecolor{muted}{HTML}{53636D}
\hypersetup{colorlinks=true,urlcolor=ink,linkcolor=ink,pdftitle={2021 conditional upper-limit echoes: v5.6.1},pdfauthor={Emrys Peets}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{1em}
\setlength{\headheight}{13pt}\setlength{\headsep}{10pt}\setlength{\footskip}{18pt}
\addtolength{\topmargin}{8pt}\addtolength{\textheight}{-8pt}
\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.11}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\color{muted}HPS GPR \quad 2021 upper-limit echoes}
\fancyhead[R]{\small\color{muted}v5.6.1}
\fancyfoot[L]{\footnotesize\color{muted}Conditional simulated full-statistics catalogue}
\fancyfoot[R]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.25pt}
\newcommand{\heading}[1]{\vspace{2pt}{\Large\bfseries\color{ink}#1}\par\vspace{3pt}}
\newcommand{\subheading}[1]{\par\vspace{4pt}{\bfseries\color{ink}#1}\par}
\newcommand{\smallnote}[1]{{\small\color{muted}#1}\par}
\newcommand{\nextpage}{\clearpage}
\begin{document}
"""


def opening():
    return PREAMBLE + r"""
{\small\color{muted}13 September 2026 \hfill Extension of the v5.6 peak catalogue}\par
\vspace{6pt}
{\LARGE\bfseries\color{ink}Conditional upper-limit echoes\\[3pt]from projected 2021 peaks}\par
\vspace{5pt}
This extension follows the v5.6 injected peaks across the mass scan to predict
nearby upper-limit dips, or \emph{echoes}. It reuses all 300 saved toys across
15 scenarios, with separate nominal $1\%\times100$ and $10\%\times10$ projections.

\subheading{The local-significance scaling that motivates the injections}
For unchanged selection and shapes, a persistent excess dominated by counting
statistics scales with the exposure multiplier $k=1/f$ as
\begin{equation}
\begin{gathered}
 S_{100\%}=kS_f,\qquad B_{100\%}=kB_f,\qquad
 Z_{100\%}\simeq\frac{kS_f}{\sqrt{kB_f}}=\sqrt{k}\,Z_f,\\[3pt]
 \boxed{Z_{1\%\to100\%}^{\rm target}=10Z_{1\%}}
 \qquad
 \boxed{Z_{10\%\to100\%}^{\rm target}=\sqrt{10}\,Z_{10\%}}.
\end{gathered}
\end{equation}
The injected amplitude was matched to this fixed-mass Asimov target. This is a
conditional construction, not a bound on future significance; a statistical
fluctuation need not persist as a signal.

\subheading{A conditional upper-limit curve for each saved spectrum}
At each mass $m$, a bin-integrated Gaussian count template $t_i(m)$ is fitted
to Poisson counts with a profiled GP-background constraint:
\begin{equation}
 \ell(A,\theta;m)=\sum_i[n_i\ln\lambda_i-\lambda_i]-\tfrac12\theta^T\theta,
 \qquad \lambda_i=b_i+(L\theta)_i+A\,t_i(m).
\end{equation}
The sideband-conditioned $b$ and $LL^T$ are recomputed at every tested mass.
Writing $\widehat A_+=\max(0,\widehat A)$, with the nuisance profiled there,
\begin{equation}
 \widetilde q_A=\begin{cases}
 -2\ln\dfrac{L(A,\widehat{\widehat\theta}_A)}
 {L(\widehat A_+,\widehat\theta_+)}, &\widehat A\leq A,\\
 0,&\widehat A>A,
 \end{cases}
 \qquad
 \mathrm{CL}_s(A)=\frac{\Pr_A(\widetilde q_A\geq\widetilde q_A^{\rm obs})}
 {\Pr_0(\widetilde q_A\geq\widetilde q_A^{\rm obs})}.
\end{equation}
The bounded asymptotic tails use the profiled background Asimov response;
$A_{90}$ solves $\mathrm{CL}_s(A_{90})=0.10$~\cite{cowan}. These are conditional,
simulated, observed-like 90\% CL$_s$ curves.

\subheading{Predicting the positions and depths of the echoes}
Let $A_{90}^{(b)}(m)$ be the result from the same scenario's generator continuum
alone. The reference and injected spectra are each reanalysed on the full grid.
Define a count-limit ratio and the depth of its selected flank minimum by
\begin{equation}
 R(m)=\frac{A_{90}^{\rm injected}(m)}{A_{90}^{(b)}(m)},\qquad
 D=1-R(m_{\rm echo}),\qquad
 m_{\rm echo}\in m_0\pm[2,8]\,\sigma_m(m_0).
\end{equation}
Each flank is clipped to the saved scan support. The deepest local minimum
on the saved scan lying in that flank is selected; if none exists, the flank
minimum is retained and marked.
$D>0$ means a tighter limit than the background reference. The contiguous
saved-grid interval around the selected minimum with $R\leq0.90$ records a
depression of at least 10\%, when present. All choices are declared before
examining observed full-statistics data.

\subheading{Why a strong peak can leave a negative fitted structure nearby}
As the excluded fit window moves away from an injected peak, signal bins and
tails can enter the GP training sidebands. Their influence on the predicted
continuum can produce negative fitted amplitudes at nearby masses and tighten
the upper limit. The signed profile score
$r=\operatorname{sign}(\widehat A)\sqrt{2(\ell_{\rm free}-\ell_{A=0})}$
makes that response visible. The one-sided discovery score would set negative
values to zero. Echoes are conditional responses of this model to the injected
structure; real detector and background effects could produce different shapes.
Generated counts and fitted total expectations remain positive.

\smallnote{No observed 100\% 2021 histogram is used. Neither these simulated
limits nor the 20-toy spreads establish calibrated coverage, a probability of
real echoes, or an unblinding decision. Nearby and overlapping scenarios are
alternative injections and must not be combined.}
"""


def overview_page():
    return r"""
\nextpage\heading{Full-range upper-limit projections}
The curves below show each target-matched Asimov spectrum. The upper panel
compares the six requested regions using the native 10\% source; the lower
panel shows the historical nominal 1\% alternatives. Labels give the actual
injected masses. Each scenario retains its own source-conditioned generator
continuum, so the reference curve also depends on the injection location.
\par\noindent\includegraphics[width=\linewidth]{../figures/fullrange_overview.pdf}\par
\subheading{Count limits and the prompt-density conversion}
The displayed coupling coordinate uses the standard conversion separately for
each tested spectrum $n$:
\begin{equation}
 K_n(m)=\frac{3\pi m f_{\rm rad}^{\rm eff}}{2\alpha}\,\rho_n(m),
 \qquad \epsilon^2_{90,n}(m)=\frac{A_{90,n}(m)}{K_n(m)}\,B_e^{-1}(m).
\end{equation}
Here $f_{\rm rad}^{\rm eff}=0.0477$, $\alpha=1/137$, and $\rho_n$ is the count
density from fractional bin overlap within $\pm1.64\sigma_m$; masses and
density use GeV consistently. Above $2m_\mu=211.316749$ MeV, the inherited
visible branching correction is
$B_e^{-1}=1+\sqrt{1-4u}(1+2u)$, $u=(m_\mu/m)^2$; below threshold it is one.
This explains a physical change in the conversion near 211 MeV.

The echo ledger instead uses $A_{90}/A_{90}^{(b)}$. This is exactly the ratio
of coupling limits evaluated with the same frozen background density in both
numerator and denominator, isolating the fit response from the small change
in the density normalization. Both conversion variants are saved.
"""


def echo_page(cat, echo):
    lines = [r"\nextpage\heading{Predicted echo locations and depths}",
             "The table records the two selected minima of each matched-Asimov limit ratio. "
             "Depth is relative to the scenario's background-only Asimov count limit at the same mass. "
             "The listed interval contains saved 1 MeV hypotheses with $R\\leq0.90$ connected to that minimum.",
             r"\begin{center}\small\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrrrrrrr@{}}\toprule",
             r"Option & $m_0$ & Left $m_e$ & Depth & $R\leq0.90$ & Right $m_e$ & Depth & $R\leq0.90$\\",
             r" & [MeV] & [MeV] & [\%] & [MeV] & [MeV] & [\%] & [MeV]\\\midrule"]
    for _, r in cat.iterrows():
        e = echo_rows(echo, r.scenario)
        entry = [lane_text(r.lane, True), f"{r.mass_MeV:g}"]
        for side in ("left", "right"):
            a = e[side]
            if a.status == "outside_scan":
                entry += ["--", "--", "--"]
                continue
            dagger = r"$^\dagger$" if bool(a.boundary) else ""
            star = r"$^*$" if not bool(a.local_minimum) else ""
            entry += [f"{a.echo_mass_MeV:g}{dagger}{star}", f"{100*a.echo_depth:.1f}", extent(a)]
        lines.append(" & ".join(entry) + r"\\")
        if r.lane == "ten":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule\end{tabular}\end{center}",
              r"\smallnote{$\dagger$: selected point is a search-flank or scan endpoint. $^*$: no local minimum; the fallback flank minimum is reported. A dash means an unavailable flank or no connected $R\leq0.90$ interval. The 244 MeV injection's right search flank lies outside the scan. A positive depth identifies a depression; a nonpositive depth does not. The additional 1\% injections at 84, 145 and 244 MeV follow the six requested pairs.}",
              r"\subheading{The saved spectra and source interpretation}",
              "Every scenario contains a background-only continuum, a target-matched injection, "
              "a simple yield-scaled injection and the same 20 toys used in v5.6.0. "
              "Only the target-matched injection has toys. The simple yield-scaled amplitude is "
              r"$a_{\rm yield}=k\max(\widehat a_f,0)$; the matched amplitude was chosen so that "
              r"its fixed-mass Asimov score equals $\sqrt{k}Z_f$. Agreement with that target is "
              "an injection construction, not evidence that the GP satisfies ideal exposure scaling.",
              r"The historical 1\% input is the last-used study source, with its own reviewed "
              "kernel coordinates and 40--300 MeV conditioning support. The native 10\\% "
              "option uses its own reviewed states and 36--300 MeV support. The respective "
              "limit grids are 53--250 and 50--250 MeV in 1 MeV steps. The 1\\% lower-edge "
              "control region at 50--52 MeV remains excluded. Both use the inherited fully "
              "scaled resolution, 0.625 MeV bins and a $\\pm2.25\\sigma_m$ fit exclusion.",
              "The exact selection, trigger-category membership, overlap and effective exposure "
              "equivalence of the two source samples remain unverified. The source significances "
              "inherited from v5.6 are fresh evaluations at archived kernels, rather than verbatim "
              "older released values. Numerical-method differences must not be read as exposure effects.",
              r"The 1\% injections at 86 and 148 MeV are inherited source-region boundary controls, "
              r"not interior observed peaks. In particular, the 1\% 86 MeV control is distinct from "
              r"the native 10\% injection at 93 MeV.",
              r"\subheading{What the toy summaries mean}",
              "On each flank, the toy minimum is searched over the same declared interval. "
              "The following pages report empirical medians and 16--84\\% ranges for its mass. "
              "To test persistence at a common location, the ratio summary and threshold count "
              "use each toy's ratio at the fixed matched-Asimov echo mass, not at its moving minimum. "
              "The plots contain every saved toy curve. These are descriptive "
              "summaries of 20 conditional realizations. They are not a background expected band, "
              "a simultaneous uncertainty band, or a calibrated confidence interval for the echo."]
    return "\n".join(lines)


def scenario_page(r, echo):
    er = echo_rows(echo, r.scenario)
    title = (f"Additional 1\\% injection: {r.mass_MeV:g} MeV"
             if str(r.region).startswith("extra") else
             f"{r.region} MeV region: {lane_text(r.lane, True)}")
    boundary = (" Inherited source-region boundary control."
                if np.isclose(r.mass_MeV,r.region_lo) or np.isclose(r.mass_MeV,r.region_hi) else "")
    lines = [r"\nextpage\heading{" + title + "}",
             f"$m_0={r.mass_MeV:g}$ MeV, $\\sigma_m={r.sigma_MeV:.3f}$ MeV; "
             f"$Z_{{\\rm target}}={r.target_Z:.2f}$, "
             f"$a_{{\\rm match}}/a_{{\\rm yield}}={r.matched_to_naive_yield_ratio:.3f}$. "
             "The deterministic curves and all 20 saved toy limits use fresh GP conditioning." + boundary,
             r"\par\noindent\includegraphics[width=\linewidth]{../figures/" + r.scenario + r".pdf}\par",
             r"\smallnote{Top: full-range coupling limits with each spectrum's own prompt density. Lower left: count-limit ratios; diamonds mark the selected matched-Asimov minima and the dotted line is $R=0.90$. Lower right: signed Asimov profile scores with the same line styles. Gray central bands omit $|m-m_0|<2\sigma_m$ from the two flank displays.}",
             r"\begin{center}\small\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrllr@{}}\toprule",
             r"Flank & $m_e$ & Depth & $R\leq0.90$ & Toy mass [MeV] & Toy $R(m_e)$ & $R\leq0.9$\\",
             r" & [MeV] & [\%] & [MeV] & median [16,84\%] & median [16,84\%] & count\\\midrule"]
    for side in ("left", "right"):
        a = er[side]
        if a.status == "outside_scan":
            lines.append(side.title() + r" & \multicolumn{6}{l}{Search flank outside the saved mass scan.}\\")
            continue
        flag = r"$^\dagger$" if bool(a.boundary) else ""
        fallback = r"$^*$" if not bool(a.local_minimum) else ""
        lines.append(f"{side.title()} & {a.echo_mass_MeV:g}{flag}{fallback} & "
                     f"{100*a.echo_depth:.1f} & {extent(a)} & "
                     f"{qrange(a,'mass')} & {qrange(a,'ratio')} & {int(a.toys_fixed_ratio_below90)}/20" + r"\\")
    searches = [f"{side}: {e.search_lo_MeV:.2f}--{e.search_hi_MeV:.2f} MeV"
                for side,e in er.items() if e.status != "outside_scan"]
    lines += [r"\bottomrule\end{tabular}\end{center}",
              "Search bounds after clipping to the scan are " + "; ".join(searches) + ". "
              r"$\dagger$ marks an endpoint and $^*$ a fallback minimum; the 1 MeV grid limits position precision."]
    return "\n".join(lines)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def appendix(cat, display):
    total = sum(int(q["rows_per_scan"])*23 for q in display)
    lines = [r"\nextpage\heading{Frozen forecast and reproducibility}",
             "This extension freezes the scan definitions and echo-selection rules before any "
             "comparison with observed 100\\% data. Its forecast concerns the response of the "
             "declared inference model to the saved v5.6 injection hypotheses. It is not an "
             "instruction to select a favorable observed minimum or to modify a subsequent scan.",
             r"\subheading{Immutable parent inputs}",
             "The copied 300 toy histograms and all deterministic generator arrays retain their "
             "parent byte identities. The numerical engine, source-specific kernels and "
             "input catalogue are preserved alongside a copy manifest. The one-worker scan "
             "uses one linear-algebra thread and creates no new random samples.",
             r"\begin{center}\small\begin{tabular}{@{}p{0.36\linewidth}l@{}}\toprule Input & SHA-256\\\midrule"]
    for label, path in [("v5.6 source and injection catalogue", "inputs/catalogue.csv"),
                        ("Extension protocol", "protocol.json"),
                        ("Profiled limit solver", "engine/limit_solver.py"),
                        ("Stable GP conditioning", "engine/stable_gp.py")]:
        h = digest(B / path)
        lines.append(label + r" & \shortstack[l]{\texttt{" + h[:32] + r"}\\\texttt{" + h[32:] + r"}}\\[3pt]")
    lines += [r"\bottomrule\end{tabular}\end{center}",
              r"\subheading{Saved numerical products}",
              f"The extension contains {len(cat)} scenarios, 23 spectra per scenario and "
              f"{total:,} mass-hypothesis rows. Each scan retains the fitted count limit, signed "
              "profile score, CL$_s$ root, fit status, conditioning diagnostics, prompt density "
              "and both coupling-conversion variants.",
              r"\begin{center}\small\begin{tabular}{@{}p{0.45\linewidth}p{0.49\linewidth}@{}}\toprule Product & Contents\\\midrule",
              r"\path{inputs/toys/<scenario>.npz} & Saved v5.6 toy counts and generator arrays.\\",
              r"\path{derived/scans/<scenario>/*.csv} & Three deterministic and 20 toy full-grid scans.\\",
              r"\path{derived/scans/<scenario>/*.json} & Exact dependencies and corresponding CSV hashes.\\",
              r"\path{derived/echo_catalogue.csv} & Frozen selected flank minima and toy summaries.\\",
              r"\path{derived/toy_echo_locations.csv} & Each toy's selected mass and ratio at the fixed Asimov echo.\\",
              r"\path{derived/report_display_ledger.json} & Display completeness and conversion controls.\\",
              r"\path{inputs/copy_manifest.json} & Provenance and byte identities of copied inputs.\\",
              r"\path{qa/} & Numerical, provenance and rendered-page validation records.\\",
              r"\bottomrule\end{tabular}\end{center}",
              r"\subheading{Numerical safeguards and rebuilding}",
              "Validation passed all 2,847 numerical checks and all 756 echo-table integrity checks.",
              r"The log-GP is conditioned in the positive eigenspace of the RBF covariance "
              r"at a relative cutoff of $10^{-14}$, using an SVD ridge calculation to avoid "
              "subtraction of nearly equal posterior covariances. The profiled solver "
              "requires positive fitted means and a converged bounded CL$_s$ root. "
              "Per-spectrum sideband prediction, covariance and Poisson noise are recomputed "
              "at fixed archived kernel coordinates. Kernel optimization variability and "
              "hyperparameter uncertainty are not sampled.",
              r"From the package root, \path{python scripts/scan_limits.py run} resumes the "
              "saved scans after checking their dependencies. "
              r"\path{python scripts/summarize_echoes.py} builds the selected-minimum ledger. Then "
              r"\path{python scripts/make_report.py} regenerates figures and "
              r"\path{source/main.tex}; compile that file from its own directory. "
              "The report generator performs no fits or random draws.",
              r"\smallnote{Numerical checks and deterministic reproducibility establish that "
              "the stated model was evaluated consistently. They do not establish calibrated "
              "coverage, rare-tail probabilities or fidelity to unobserved full-data structure.}",
              r"\vspace{-5pt}\begin{thebibliography}{1}\small",
              r"\bibitem{cowan} G. Cowan, K. Cranmer, E. Gross and O. Vitells, "
              r"\emph{Asymptotic formulae for likelihood-based tests of new physics}, "
              r"Eur. Phys. J. C \textbf{71}, 1554 (2011); erratum \textbf{73}, 2501 (2013). "
              r"\href{https://arxiv.org/abs/1007.1727}{arXiv:1007.1727}. "
              "Bounded upper-limit statistics and Asimov distributions provide the asymptotic conventions.",
              r"\end{thebibliography}\end{document}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-only", action="store_true")
    args = ap.parse_args()
    cat = ordered(pd.read_csv(B / "inputs" / "catalogue.csv"))
    echo = pd.read_csv(D / "echo_catalogue.csv")
    assert len(echo) == 2*len(cat)
    assert set(echo.scenario) == set(cat.scenario)
    F.mkdir(exist_ok=True)
    (B / "source").mkdir(exist_ok=True)
    ledger = D / "report_display_ledger.json"
    if args.source_only:
        display = json.loads(ledger.read_text())
    else:
        overview(cat)
        display = []
        for _, r in cat.iterrows():
            display.append(scenario_plot(r, echo))
            print("figures", r.scenario, flush=True)
        ledger.write_text(json.dumps(display, indent=2) + "\n")
    parts = [opening(), overview_page(), echo_page(cat, echo)]
    parts.extend(scenario_page(r, echo) for _, r in cat.iterrows())
    parts.append(appendix(cat, display))
    path = B / "source" / "main.tex"
    path.write_text("\n".join(parts))
    print(path, flush=True)


if __name__ == "__main__":
    main()
