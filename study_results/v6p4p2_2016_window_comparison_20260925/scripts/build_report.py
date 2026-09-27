"""Assemble the standalone v6.4.1 note in the v6.3.n LaTeX layout."""
from pathlib import Path
import json, subprocess, shutil
import pandas as pd
B=Path(__file__).resolve().parents[1]

PREAMBLE=r'''\documentclass[11pt]{article}
\usepackage[margin=.73in]{geometry}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype,fancyhdr,xurl,hyperref,array}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black,pdftitle={HPS GPR v6.4.1: 2016 and combined signal-MC extraction}}
\pagestyle{fancy}\fancyhf{}\lhead{HPS GPR: 2016 and combined signal-MC extraction}\rhead{v6.4.1}\cfoot{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{2em}
\newcommand{\Ah}{\widehat A}\newcommand{\sh}{\widehat\sigma}
\begin{document}
{\LARGE\bfseries The 2016 signal shape\par and its effect on the combined analysis}\par
{\large HPS Gaussian-process background analysis}\par
25 September 2026\hfill Version 6.4.1

\textbf{The result.} The supplied smeared 2016 signal MC has a reconstructed core slightly below the generated mass. Its displacement is 0.08--1.02 MeV over the adopted 40--175 MeV domain, and its tails are measurably non-Gaussian. The new extraction uses the full empirical signal shape, interpolated between neighboring MC masses, with the fit and GP-training exclusion centered on that shape's fitted core.

For 2016 the interval is $-3.5\le u\le+3.5$; for 2021 the latest $-4\le u\le+3$ interval is retained. Here $u=(m_{\rm rec}-c)/s$, with each year's MC core center $c$ and core width $s$. The 2015 Gaussian remains unchanged.

\textbf{What changes in the observations?} The strongest 2016 MC-template region is at a generated mass of 91 MeV, followed by a disjoint fitted region at 69 MeV. The 91 MeV fit gives $\Ah=20{,}201\pm5{,}699$ full selected MC events and a conditional 90\% profile-limit of 27,505 events. Its local asymptotic probability is $0.000196$, but the fixed-source background check has 3 exceedances in 1,000 toys. The difference is material: the background source produces a positive mean fitted signal in this region. The report shows both probability descriptions.

The combined MC-template analysis has its smallest local asymptotic probability at 68 MeV, $p_0=0.00248$. Over 60--175 MeV, replacing the 2016 Gaussian in the existing 2021-MC combination changes the conditional profile upper limit by a median $+1.26\%$, with a range from $-6.39\%$ to $+12.49\%$. These changes include both the signal shape and the fit/training window.

\begin{center}\small\begin{tabular}{>{\raggedright\arraybackslash}p{1.30in}>{\raggedright\arraybackslash}p{1.12in}>{\raggedright\arraybackslash}p{1.12in}>{\raggedright\arraybackslash}p{1.75in}}\toprule
Combined comparison & 2015 & 2016 & 2021\\\midrule
All Gaussian & Gaussian & Gaussian & Established shifted Gaussian\\
2021 MC only & Gaussian & Gaussian & Shifted neighbor MC, $[-4,+3]u$\\
2016 + 2021 MC & Gaussian & Shifted neighbor MC, $\pm3.5u$ & Shifted neighbor MC, $[-4,+3]u$\\\bottomrule
\end{tabular}\end{center}

\textbf{Reading guide.} The first part explains the native histograms, their centers, overlays and interpolation. The extraction section defines the windows, likelihood and normalization before showing the 2016 scan, two fitted excess regions, and the three combined comparisons. Fixed-source toy checks distinguish the local probability and discrete fixed-source rank endpoints from the continuous profile curves. The final catalogue shows every supplied 2016 histogram.

The observations are the archived full 2015 and 2016 spectra and the archived 2021 10\% spectrum. All probabilities are local and conditional on the specified models. The numerical comparisons do not establish a scan-wide discovery probability or a detector-qualified physical exclusion.
'''

METHOD=r'''
\clearpage\section*{From an MC shape to an observed-data fit}
\label{sec:method}

\textbf{One core-centered coordinate, two year-specific windows.} The local Gaussian fit supplies only the core center and width used to align the native histograms. It does not replace their tails. If $m$ lies between simulated masses $m_L$ and $m_R$, define $t=(m-m_L)/(m_R-m_L)$ and interpolate $c(m)$ and $s(m)$ linearly. For an analysis-bin edge $x$, evaluate
\[
 F_m(x)=(1-t)F_L\!\left(c_L+s_L\frac{x-c(m)}{s(m)}\right)
       +tF_R\!\left(c_R+s_R\frac{x-c(m)}{s(m)}\right).
\]
Each $F$ is the cumulative probability of a full selected native histogram, with piecewise-uniform probability within its bins. Analysis-bin probabilities are differences of $F_m$ at the edges. At an MC anchor the native histogram is recovered exactly. Its reconstructed displacement is already present; no second translation is added.

\begin{center}\begin{tabular}{lll}\toprule
Extraction & Signal-fit bins & GP-training bins\\\midrule
2016 MC & $|u|\le3.5$ & $|u|>3.5$\\
2021 MC & $-4\le u\le+3$ & $u<-4$ or $u>+3$\\
Gaussian reference & $|m_{\rm rec}-c_G|\le2.25\sigma_{\rm ref}$ & Exterior of the same interval\\\bottomrule
\end{tabular}\end{center}

Bin centers determine membership. The actual outer bin edges, selected bin counts and interpolation anchors are saved for every mass. The 2016 scan uses 1 MeV steps from 40 through 175 MeV; the combined scan uses 60 through 240 MeV. No 2016 template is extrapolated into the missing region above 175 MeV. In the 2016 MC scan, 97.93--99.58\% of the selected signal falls in the actual fitted bins. The remaining 0.42--2.07\% falls in training bins and is retained in injected samples. This small fraction can still affect background prediction.

\textbf{Two controls make the 2016 change interpretable.} The inherited Gaussian uses its original center, reference width and window. A second Gaussian control retains exactly that signal shape but adopts the new MC-centered fit/training window. The final method then replaces the Gaussian by the neighboring MC shape in that same window. The control separates the window change from the subsequent shape change; neither difference is solely a correction to the center.

For 2015 and 2016, the reference Gaussian center is the generated mass. For 2021 it retains the established shift,
\[
 c_G(m)=m-3.22433\,{\rm MeV}-2.21399\,{\rm MeV}\ln\!\left(\frac{m}{150\,{\rm MeV}}\right).
\]
Thus ``all Gaussian'' means the established v6.3.n reference, including its already shifted 2021 Gaussian. The Gaussian widths retain the archived resolution curves. The 2015/2016 reference Gaussians retain their historical normalization over the stored spectrum; MC probabilities always retain the full selected normalization, including any probability outside that spectrum.

The 2021 extraction uses the latest saved v16 target-constrained MC bank over 60--240 MeV and the same linear neighbor convention. The earlier 2021 samples used in the shape-comparison figures provide context for the displacement; their fitted parameters are not substituted for this production bank. Interpolation near the 2021 low-mass acceptance transition remains a model assumption.

\clearpage\section*{What do the fitted yield, upper limit and probability mean?}

The Gaussian process (GP) predicts the background in the excluded interval from the exterior bins. Its kernel amplitude and length scale are fixed to the archived mass-dependent values. Conditioning on the logarithms of positive bin counts, with variance $1/n$, gives a lognormal arithmetic mean $b$ and a correlated count-space covariance $V$. Empty training bins retain the archived log-target and variance conventions. The GP prediction is recomputed after each change of window and for every toy dataset; the kernel parameters are not reoptimized.

Write $V=LL^T$, including the inherited numerical regularization. In one campaign, the bin expectation is $\lambda_i=b_i+(L\theta)_i+A p_i$, where $p_i$ is the full selected signal probability in fit bin $i$. The constrained count likelihood is
\[
 -\ln\mathcal L(A,\theta)
 =\sum_{i\in{\rm fit}}\left[\lambda_i-n_i+n_i\ln\frac{n_i}{\lambda_i}\right]
  +\tfrac12\theta^T\theta .
\]
The $n_i=0$ logarithmic term is zero and every expected bin count must remain positive. The nuisance parameters allow correlated background changes penalized by the GP covariance. Their fitted background can differ from the initial GP mean. The likelihood uses the fit bins; exterior bins influence it through the GP constraint.

\textbf{The yield is signed for diagnosis.} The unconstrained estimate $\Ah$ can be negative. Its displayed error is the conditional curvature (Hessian) error after profiling the background. A positive MC-template yield refers to the full selected MC distribution, rather than only the probability inside the fitted interval. No fit-window renormalization is applied. A Gaussian yield describes that Gaussian model and need not measure an injected MC-shaped yield without calibration.

\textbf{The continuous curves are conditional profile limits.} For a tested nonnegative signal strength, the nuisance parameters are profiled again. The bounded likelihood-ratio denominator uses the free fit when its signal is positive and the background-only fit otherwise. The solver uses the inherited asymptotic sampling tails, with a background-mean Asimov profile, and solves $CL_s=0.10$. Thus the plotted curves are observed 90\% profile-$CL_s$ limits under the stated model and sampling approximation. They are not obtained by simply adding $1.64$ errors to $\Ah$.

\textbf{The displayed probability is local.} Define the signed likelihood root and its excess-only statistic by
\[
 r={\rm sign}(\Ah)\sqrt{2\,[\ln\mathcal L(\Ah,\widehat\theta)-\ln\mathcal L(0,\widehat\theta_0)]},
 \qquad q_0=\max(0,r)^2,\qquad p_0=\Phi[-\max(0,r)].
\]
Here $\Phi$ is the standard normal cumulative distribution. Deficits have displayed $p_0=0.5$. This asymptotic number refers to one mass and one fixed template. It does not include scanning masses, choosing two excess regions, or uncertainty in the estimated background source. The selected-point toy checks below test its behavior under a specified fixed background source.

The two 2016 examples are positive local maxima ranked by $q_0$, with the additional requirement that their fitted bins do not overlap. This selects 91 and 69 MeV. It is a way to show two distinct fitted regions, not an independence claim: their GP sidebands can still overlap. The same rule applied campaign by campaign selects 68 and 94 MeV in the combined scan.

\clearpage\section*{How are the years combined?}

\textbf{The combination shares a signal parameter and keeps separate backgrounds.} Let $\psi=\epsilon^2_{ee}/10^{-8}$ denote the inherited coupling-conversion coordinate. For campaign $y$, define $K_y(m)$ as the expected selected event yield per unit $\psi$. The combined likelihood uses
\[
 \lambda_{yi}=b_{yi}+(L_y\theta_y)_i+\psi K_y(m)p_{yi},
 \qquad -\ln\mathcal L_{\rm joint}=\sum_y[-\ln\mathcal L_y].
\]
The background constraints are independent between campaigns, while $\psi$ is common. Local probabilities follow from this joint likelihood; they are not multiplied between years. The signal MC defines $p_{yi}$, and the inherited conversion defines $K_y$. A common signal fit can move individual campaign backgrounds differently.

\begin{center}\begin{tabular}{ll}\toprule
Generated-mass range & Campaigns in every compared policy\\\midrule
60--100 MeV & Full 2015, full 2016, 2021 10\%\\
101--175 MeV & Full 2016, 2021 10\%\\
176--240 MeV & 2021 10\% only\\\bottomrule
\end{tabular}\end{center}

Keeping the same campaign support is essential for the shape comparison. In particular, the old 2016 Gaussian availability through 180 MeV is truncated to 175 MeV here for all policies, because the new MC bank stops there. Above 175 MeV the two MC-based combinations must coincide exactly. Boundaries can produce visible changes because a dataset ceases to contribute.

\textbf{The conversion is inherited, rather than remeasured.} At generated mass $m$, the archived native spectrum provides a local event density $\rho_y(m)$ averaged across $m\pm1.64\sigma_{{\rm ref},y}(m)$, using fractional overlap with the native bins. With the archived effective radiative fraction $f_{{\rm rad},y}$,
\[
 K_y(m)=10^{-8}\frac{3\pi}{2\alpha}\,m\,f_{{\rm rad},y}\,\rho_y(m),\qquad \alpha=1/137,
\]
where mass and density use reciprocal units. This calculation stays centered at the generated mass for every template policy. Moving the reconstructed signal core therefore does not silently move the conversion window. The per-mass values are supplied in \texttt{results/normalization.csv}.

Above $2m_\mu=211.316749$ MeV, the displayed coupling limit retains the earlier visible-decay multiplier
\[
 \epsilon^2_{\rm displayed}=10^{-8}\psi\left[1+\sqrt{1-4r_\mu}\,(1+2r_\mu)\right],
 \qquad r_\mu=(m_\mu/m)^2.
\]
Below this threshold the multiplier is one. Both the unmultiplied $ee$ coordinate and the historical displayed value are saved. This inherited convention, including its selected-spectrum density and radiative fraction, supports an internally consistent comparison. A new efficiency, acceptance, detector calibration or branching-model validation has not been derived from the supplied histograms.

The plotted curves keep MC shape, core location, width, GP kernel settings and the conversion inputs fixed. Their statistical comparison does not include uncertainties in those ingredients. A joint observed upper limit need not be smaller than each individual observed limit, because the datasets fluctuate and share a constrained signal strength.
'''

LOCAL=r'''
\clearpage\section*{Check the local probabilities with fixed-source backgrounds}

For each campaign, the archived GP arithmetic-mean spectrum is held fixed as the Poisson background source. Each toy fluctuates every stored analysis bin independently. The extraction then rebuilds its GP constraint from the toy's sidebands and profiles the same likelihood as for the observations. These toys do not draw new GP functions or retune the kernel. They test a conditional procedure for this source.

There are 1,000 background toys at each selected 2016 mass and at the combined 68 MeV mass. At 68 MeV the three methods receive the same toy counts; campaign draws are independent. At the two 2016 masses the same background draws are reused. For $k$ toys with $q_0$ at least as large as observed, the displayed rank is $(k+1)/1001$. The interval below is the exact two-sided 95\% Clopper--Pearson interval for the underlying exceedance probability.

\begin{center}\small\begin{tabular}{llrrrr}\toprule
Scope & Method/mass & Asymptotic $p_0$ & $k/1000$ & Add-one rank & 95\% interval\\\midrule
2016 & MC, 91 MeV & 0.000196 & 3 & 0.00400 & [0.00062, 0.00874]\\
2016 & MC, 69 MeV & 0.01136 & 5 & 0.00599 & [0.00163, 0.01163]\\
Combined & Gaussian, 68 MeV & 0.00317 & 0 & 0.00100 & [0, 0.00368]\\
Combined & 2021 MC, 68 MeV & 0.00181 & 0 & 0.00100 & [0, 0.00368]\\
Combined & Both MC, 68 MeV & 0.00248 & 0 & 0.00100 & [0, 0.00368]\\\bottomrule
\end{tabular}\end{center}

\textbf{The 91 MeV discrepancy is visible in the null mean.} At 91 MeV, the mean signed likelihood root is $+0.656$, with standard deviation 0.969. A background-only source therefore produces a positive fitted signal on average at this location. This explains why a standard-normal asymptotic reference can make the observed fit look more unusual than it does under this fixed source. The mean is a property of the specified background source and extraction; it is not subtracted from the observed result.

At 69 MeV the mean root is $-0.188$ with standard deviation 0.981. At combined 68 MeV the means are $-0.252$, $-0.225$ and $-0.211$ for all Gaussian, 2021 MC only and both MC, with standard deviations 0.967, 0.985 and 0.985. These means and spreads answer a different question from the signal response when MC events are added.

\textbf{Zero exceedances do not mean zero probability.} The 1,000 combined toys cannot distinguish the three tail probabilities at this precision. Their common upper interval endpoint is 0.00368. No numerical ordering of the three methods is inferred from the zero counts.

All these masses were chosen after examining the observed scans. Neither the asymptotic column nor the fixed-mass toy rank accounts for that search. A global probability would require repeating the full mass search and the same selection rule in each background toy, which was not done. The toy intervals quantify finite-ensemble uncertainty at the displayed mass, conditional on the fixed source; they do not cover uncertainty in the true background shape.
'''

RANK=r'''
\clearpage\section*{Compare upper endpoints for the same injected signal}

The continuous Gaussian and MC profile curves fit different signal distributions. To compare their response to one specified signal, this additional selected-point study injects the full neighboring MC distribution for 2016 and 2021, with the retained Gaussian for 2015, into every method's background toys. The common expected strength is $\psi$; each campaign receives mean signal $\psi K_y$. All stored-bin and below/above-support categories are drawn, without renormalizing into the fitted interval. Realized counts fluctuate around these expected yields.

For each mass and trial strength, 200 calibration toys are fit by all three methods. The backgrounds are independent of the 1,000-toy null check. Within this calibration, methods share exactly the same background and signal draws, and increasing strengths add independent Poisson increments to the preceding signal draw. The scale $s_0$ is the Gaussian-reference error on the fixed background mean, independent of the observed fitted yield. The tested strengths are
\[
 \psi/s_0\in\{0,0.5,1,2,3,4,5,6,8,10,12,16\}.
\]
For each method, compare its signed observed fitted strength to the signed fitted strengths of the injected toys:
\[
 p_\psi=\frac{1+\#\{\widehat\psi_{\rm toy}\le\widehat\psi_{\rm obs}\}}{201}.
\]
Retain a trial strength if $p_\psi>0.10$. The table gives the largest accepted point and the next tested point, which is rejected. No interpolation between them is claimed. This is a one-sided finite-grid rank inversion, a different construction from the profile-$CL_s$ curves.

@@RANK_TABLE@@

All nine accepted sets are nonempty, contain no holes on the tested grid and end below its largest strength. At 69 MeV the three methods share the largest accepted full 2016 yield of 20,617 events, with 27,489 events rejected at the next grid point. At 91 MeV the corresponding values are 21,405 and 26,756 events. Equality on this grid does not establish equality of continuous endpoints.

At combined 68 MeV the both-MC method rejects $6s_0$ with $p_\psi=18/201=0.0896$, while the other methods still accept that point. With only 200 calibration toys, this crossing is close to the 0.10 decision threshold. The apparent improvement in the discrete MC endpoint is therefore a conditional, finite-grid observation, not a precise sensitivity gain. It can coexist with a larger profile limit because the latter treats each extraction model's own signal as the hypothesis.

The 21,600 calibration fits preserve their complete signed-yield rows, rank counts, accepted sets, input-draw hashes and full-signal partitions. This targeted study does not establish 90\% coverage across the mass scan: it has no independent injected-signal evaluation ensemble and explores one estimated background source. The full continuous scans remain clearly labeled conditional profile diagnostics.
'''

END=r'''
\clearpage\section*{What is established, and how can it be reproduced?}

The smeared 2016 samples support a small downward core displacement and a signal shape with tails that a Gaussian does not reproduce. A neighboring-MC template retains those features and gives a defined interpolation over 40--175 MeV. Using the specified core-centered windows produces the observed scans and the combined comparisons in this note. The fixed-source check also identifies a positive background-only fit bias at 91 MeV, so its asymptotic probability should not be read as an empirically calibrated significance.

The selected-point common-signal inversion provides a direct comparison of methods under one injected MC hypothesis. It includes signal in the training bins and outside the stored spectrum, and distinguishes expected yield from the realized Poisson signal. Its coarse grid and finite calibration sample are shown explicitly. Extending that calibration across masses and evaluating it on independent signal-injected samples would be required before treating it as a validated full-scan limit procedure.

\textbf{Frozen sources.} The package contains the 29 supplied 2016 ROOT histograms, extracted arrays and original shape-fit ledgers; the latest archived v16 target-constrained 2021 MC bank; archived observed spectra, resolution curves, kernel settings and conversion inputs; and the fixed GP-mean toy sources. There are 168 frozen input files in \texttt{provenance/input\_manifest.sha256}. The unsmeared 2016 histograms are not used in the extraction. Source comparisons to earlier versions are preserved as numerical inputs rather than dependencies on another working directory.

\textbf{Numerical checks.} The observed ledger has 1,354 rows: 992 fresh fits and 362 unchanged 2021 individual rows reused from v6.3.8. Independent checks verify input hashes, template normalization and anchor closure, fit/training geometry, selected regions, joint likelihood factorization, and targeted likelihood/Hessian replays. The inherited comparisons agree to numerical precision on the common campaign domain; 176--180 MeV is deliberately outside the old joint comparison because the campaign content has changed. All 5,000 null-toy fits and 21,600 injected calibration fits pass the saved numerical criteria.

\textbf{Portable rebuild.} From the package root, run \texttt{bash rebuild.sh}. The default reuses signature-checked numerical checkpoints; \texttt{bash rebuild.sh --fresh} regenerates the new observed fits and both toy studies. The 2021 individual rows are deliberately reused in either mode, with independent direct checks. Two local processes and one numerical thread per process are used. Python package versions and the LaTeX compiler are recorded in \texttt{provenance/runtime.json}. All figures are vector PDFs with PNG companions, and the LaTeX source uses the same article layout, typography, running header and explanatory captions as the v6.3.n notes.

The main numerical entry points are \path{scripts/run_scan.py}, \path{scripts/run_local_checks.py} and \path{scripts/run_rank_limits.py}. Tables and selected-fit arrays are under \texttt{results}; independent checks and PDF review records are under \texttt{qa}. The package manifest identifies the delivered files. Previous reports are preserved separately.

\textbf{Scope.} MC resampling errors describe finite simulation statistics. Core-definition changes describe sensitivity to the locator. Window containment describes geometry. Fitted response describes how the extraction reacts to injected signal. Local ranks describe one fixed source and mass. None of these quantities alone is a detector efficiency, a scan-wide probability or a complete uncertainty on a physical exclusion.
'''

def rank_table():
    d=pd.read_csv(B/'results/rank_limits.csv')
    labels={'gaussian':'Gaussian','gaussian_mc_window':'Gaussian, new window','mc':'MC',
            'all_gaussian':'All Gaussian','mc2021':'2021 MC only','mc2016_2021':'2016 + 2021 MC'}
    order={'gaussian':0,'gaussian_mc_window':1,'mc':2,'all_gaussian':0,'mc2021':1,'mc2016_2021':2}
    d=d.assign(method_order=d.method.map(order)).sort_values(['scope','mass_MeV','method_order'])
    rows=[]
    for r in d.itertuples():
        scope='2016' if r.scope=='2016' else 'Combined'
        rows.append(f'{scope}, {r.mass_MeV} & {labels[r.method]} & {r.largest_accepted_epsilon2*1e6:.3f} & {r.next_grid_epsilon2*1e6:.3f}'+r'\\')
    return r'''\begin{center}\small\begin{tabular}{llrr}\toprule
Scope, mass [MeV] & Extraction & Largest accepted & Next rejected\\
& & \multicolumn{2}{c}{$\epsilon^2$ in units of $10^{-6}$}\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{center}'

def main():
    shapes=(B/'source/shape_sections.tex').read_text()
    marker=r'\clearpage\section*{Native signal-MC catalogue: 30--75 MeV}'
    shape_body,gallery=shapes.split(marker,1)
    extracted=(B/'source/extraction_sections.tex').read_text()
    text=PREAMBLE+shape_body+METHOD+extracted+LOCAL+RANK.replace('@@RANK_TABLE@@',rank_table())+END+marker+gallery+r'\end{document}'+'\n'
    (B/'source/report.tex').write_text(text)
    executable=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic'
    subprocess.run([executable,'--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(B/'source/report.tex')],cwd=B/'source',check=True)

if __name__=='__main__':main()
