#!/usr/bin/env python3
"""Preserve the v6.4.3 numerical baseline and append a v6.4.4 explanation."""
from pathlib import Path
import argparse,json,re,shutil,subprocess
import pandas as pd

B=Path(__file__).resolve().parents[1]
LABELS={'2016':'2016','2021':'2021','combined':'Combined'}

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Parent report passage does not match the pinned source: '+old[:70])
    return text.replace(old,new,1)

def parent_text():
    s=(B/'source/parent_v643_report.tex').read_text().replace('6.4.3','6.4.4')
    s=replace_once(s,r'\textbf{The result.}',r'\textbf{The retained raw-statistic baseline.}')
    s=replace_once(s,'Complete background-only simulations give the mass-scan global probabilities below.',
        'The original 1,024-scan calibration (cohort A) gives the raw-statistic global probabilities below. Appendices A--D add an independent comparison that calibrates local ranks before searching over mass.')
    old=re.search(r'\\textbf\{If the largest result among these three searches is selected\.\}.*?(?=\n\n)',s,re.S)
    if not old:raise ValueError('Missing original across-search paragraph')
    s=s[:old.start()]+r'''\textbf{The combined physics test.} The combination uses one shared coupling at one generated mass. The individual-year searches are diagnostics; the combination does not choose whichever dataset gives the largest excess. Appendix A verifies the shared-coupling fit. The following appendices compare its raw-statistic and locally calibrated mass searches.'''+s[old.end():]
    s=replace_once(s,r'\section*{Exactly what is repeated in a simulated search?}',r'\section*{Raw-statistic baseline: the repeated simulated search}')
    s=replace_once(s,r'\section*{From a local excess to a mass-scan probability}',r'\section*{Raw-statistic baseline: local and global descriptions}')
    s=replace_once(s,'The change from gray to blue tests the local reference distribution under this background source. The change from blue to red then includes the search over mass with the same raw statistic. Neither step assumes a number of independent resolution elements.',
        'Blue calibrates each mass; red calibrates the largest raw statistic. Their difference is not a unique look-elsewhere penalty when local null distributions vary with mass: largest raw $q_0$ need not mean smallest local $p$. Appendix B compares both searches.')
    s=replace_once(s,r'\section*{The distribution of the largest background excess}',r'\section*{Raw-statistic baseline: the largest background excess}')
    s=replace_once(s,r'\section*{Scope, validation and reproducibility}',r'\section*{Baseline scope, validation and reproducibility}')
    old=re.search(r'\\textbf\{What the probabilities answer\.\}.*?(?=\n\n)',s,re.S)
    if not old:raise ValueError('Missing original scope paragraph')
    new=r'''\textbf{What the baseline probabilities answer.} Under each fixed background source, how often does the extraction produce a largest raw excess statistic at least as large as observed on its 1 MeV grid? The combined row tests one shared coupling. The older across-search calculation remains in \texttt{results/family\_global.json} as an archival diagnostic; it is not the combined physics test. These probabilities do not include the earlier choice among 2016 windows, signal templates or background constructions. The appendix evaluates a second, explicitly defined mass-search statistic.'''
    s=s[:old.start()]+new+s[old.end():]
    s=replace_once(s,'This note selects $\\pm2.5u$ for 2016 and recomputes the observed scans and their null calibration.',
        'The retained baseline selected $\\pm2.5u$ for 2016 and recomputed the observed scans and their null calibration. This revision preserves those results and adds 1,024 independent validation scans.')
    s=replace_once(s,r'\textbf{Numerical validation.}',r'\textbf{Validation of the retained baseline.}')
    s=replace_once(s,'The scan maxima, exceedance counts, confidence intervals and family combination were recomputed from the saved arrays.',
        'The original scan maxima, exceedance counts and confidence intervals were recomputed from the saved arrays; their numerical records remain unchanged.')
    s=replace_once(s,r'\textbf{Precision.} The fixed sample size was chosen before calibration.',
        r'\textbf{Baseline precision.} Its fixed sample size was chosen before calibration.')
    return s

def cell(x,dec=3):return f'{float(x):.{dec}f}'

def appendix(summary,fits,audit,thresholds):
    rows=[];raw=[];fitrows=[];auditrows=[]
    for scope in ['2016','2021','combined']:
        s=summary.loc[scope];name=LABELS[scope]
        rows.append(f"{name} & {int(s.raw_peak_mass_MeV)} / {int(s.localfirst_peak_mass_MeV)} & {s.raw_2048_p:.3f} & {int(s.minp_B_k)}/1024 & {s.minp_B_p:.3f} & [{s.minp_B_p95_low:.3f}, {s.minp_B_p95_high:.3f}] & {s.minp_B_Z:.2f}\\\\")
        raw.append(f"{name} & {int(s.raw_peak_mass_MeV)} & {int(s.raw_1024_k)}/1024 & {s.raw_1024_p:.3f} & {int(s.raw_new1024_k)}/1024 & {s.raw_new1024_p:.3f} & {s.raw_2048_p:.3f}\\\\")
        fitrows.append(f"{name} & {fits[scope]['grid_points']} & {fits[scope]['N_eff']:.2f} & {s.sidak_fitted_p:.3f} & {s.sidak_grid_p:.3f} & {s.minp_B_p:.3f}\\\\")
    for a in audit['mass_summaries']:
        if a['mass_MeV'] in [67,68,91]:
            auditrows.append(f"{a['mass_MeV']} & {a['joint_psi_hat']:.1f} & {a['joint_q0']:.3f} & {a['sum_individual_q0']:.3f}\\\\")
    s=summary.loc['combined']
    text=r'''
\clearpage\section*{Appendix A. One shared coupling, then a search over mass}

\textbf{The combination tests one signal hypothesis.} At each common generated mass $m$, campaign $y$ has its own selected template $f_{yi}(m)$ and fixed conversion $K_y(m)$, but every campaign uses the same signal strength $\psi=\epsilon^2/10^{-8}$:
\[
 \lambda_{yi}=b_{yi}+(L_y\eta_y)_i+\psi K_y(m)f_{yi}(m),\qquad
 \mathcal L(\psi,\{\eta_y\};m)=\prod_y\mathcal L_y(\psi,\eta_y;m).
\]
The nuisance vectors are separate. The joint alternative has one common $\psi$ and the joint null fixes that same parameter to zero. Searching over mass changes $m$ for the complete joint fit. It does not select a campaign or add its independently maximized excess statistics. The inherited $ee$-yield normalization and historical display conversion above the dimuon threshold are unchanged; this audit verifies the common parameter, not a new physical normalization.

An independent factorization audit checks the joint profile against the sum of campaign profiles evaluated at the \emph{same} $\psi$. Across 24 profile points at six masses, the largest negative-log-likelihood difference is $6.26\times10^{-13}$. The table illustrates why independent campaign maxima would define a different test.
\begin{center}\small\begin{tabular}{rrrr}\toprule
Mass [MeV] & Joint $\widehat\psi$ & Joint $q_0$ & Sum of individual $q_0$\\\midrule
@@AUDIT_ROWS@@
\bottomrule\end{tabular}\end{center}
At 91 MeV, the individual campaigns prefer different signal strengths, including a negative 2021 fit. The common-strength result is $q_0=0.360$, while adding the individual excess statistics would give 18.106. The latter is not the statistic used anywhere in the combined scan. Campaign support remains the same as on page 1.

\textbf{Calibrate local behavior before ranking masses.} The retained baseline searches for the largest raw $q_0$. The new comparison first uses cohort A, the original 1,024 null scans, to freeze a local upper-tail map at every mass:
\[
 p_{A,m}(q)=\frac{1+\#\{a\in A:q_a(m)\ge q\}}{1025},
 \qquad S=\min_{m\in\mathcal M}p_{A,m}\bigl(q(m)\bigr).
\]
Smaller $S$ is more extreme. The same frozen map is applied to the observations and 1,024 fresh cohort-B scans. Those scans use identical sources and extraction settings, with a disjoint random-number namespace. The global count includes $S_B\le S_{\rm obs}$, including ties; its add-one probability is $(k_B+1)/1025$.

A fixes the maps and the effective-trials approximation described in Appendix C. B neither updates those maps nor tunes that approximation. Each whole campaign draw is reused across masses and applicable searches, preserving their correlations. Both mass-search statistics still use the common-coupling likelihood above; only their ordering of mass hypotheses differs.

\clearpage\section*{Appendix B. Compare the two global calibrations}
\begin{center}\includegraphics[width=\linewidth,height=4.55in,keepaspectratio]{../figures/calibrated_global_mass_comparison.pdf}\end{center}
{\small\refstepcounter{figure}\textbf{Figure \thefigure. How to read the figure.} Gray compares each observed raw $q_0(m)$ with raw maxima from the pooled 2,048 scans. Red compares its frozen A local rank with the minimum local rank in independent B scans; the red band is a pointwise 95\% interval conditional on A. Gold is the frozen effective-count Sidak approximation. Purple is the independent-mass-grid Sidak reference. The right column expresses the same probabilities as $Z=\max[0,\Phi^{-1}(1-p)]$. Red vertical lines mark the smallest observed local rank; dotted combined boundaries mark campaign changes. Gray and red calibrate different orderings, so neither is a correction factor for the other.}\par\medskip

\begin{center}\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrr}\toprule
Search & Peaks: raw/local & Raw pooled $p$ & B count & Local-first $p$ & B 95\% interval & $Z$\\\midrule
@@RESULT_ROWS@@
\bottomrule\end{tabular}\end{center}

The raw and local-first peaks are each selected using their own declared statistic. For the combined analysis, the raw maximum is at @@RAW_MASS@@ MeV and the smallest calibrated local rank is at @@LOCAL_MASS@@ MeV. The two procedures ask related but different null-tail questions when the local distributions vary with mass. Their probabilities should be compared as complete procedures rather than choosing whichever reports the smaller value.

All probabilities remain conditional on the fixed GP source and selected extraction. The B interval accounts for its finite number of experiments, holding the A calibration map fixed. It does not include uncertainty in the map, source, signal template or earlier analysis choices. The two Sidak curves are approximations whose independent check appears next.

\clearpage\section*{Appendix C. Two Sidak references with different assumptions}
\begin{center}\includegraphics[width=\linewidth,height=4.35in,keepaspectratio]{../figures/calibrated_threshold_comparison.pdf}\end{center}
{\small\refstepcounter{figure}\textbf{Figure \thefigure. How to read the figure.} Left: direct B probabilities $G_B(\alpha)=\Pr(S_B\le\alpha)$ and both Sidak references. Right: $\log[1-G_B(\alpha)]/\log(1-\alpha)$ re-expresses B, with transformed pointwise intervals; it is not another fit. Gold shading marks the A fit range; black lines mark observed local minima. Infinite interval endpoints are omitted. Neither reference resolves probabilities below the A-map floor.}\par\medskip

For independent uniform local probabilities, $G(\alpha)=1-(1-\alpha)^N$. Purple uses the grid size; gold fixes $N_{\rm eff}$ from A alone. A's self-inclusive ranks $\#\{q_A\ge q_i\}/1024$ equal add-one leave-one-out ranks. Their minima are dependent and use different maps from B's full frozen map.

The A-only fit uses equal-weight, zero-intercept least squares of $\log(1-G_A)$ versus $\log(1-\alpha)$ at 0.005, 0.01, 0.02, 0.03 and 0.05. All five thresholds were fixed before B.
\begin{center}\small\begin{tabular}{lrrrrr}\toprule
Search & Grid $N$ & A-fit $N_{\rm eff}$ & Gold $p$ & Purple $p$ & Direct B $p$\\\midrule
@@FIT_ROWS@@
\bottomrule\end{tabular}\end{center}
The last three columns use the observed local minima. All gold predictions fall below their direct B 95\% intervals: the constant count understates the global probability here. These are pointwise comparisons, not a simultaneous curve test. These minima are below the fit range, making gold an extrapolation. B's ranks need not be uniform conditional on A; the fitted count describes this estimated-map procedure, not independent windows. The direct B tail is the calibrated global result.

\clearpage\section*{Appendix D. Finite calibration, retained results and references}

\textbf{What the old ``largest selected result'' meant.}\par
That optional diagnostic used $T_{\rm family}=\max(T_{2016},T_{2021},T_{\rm combined})$, where each $T$ already maximizes over mass. It asked whether \emph{any of three displayed searches} looked unusual. It neither defines the joint likelihood nor answers the shared-coupling question. It is archived only. The individual-statistic sum in Appendix A illustrates another different test; neither construction is used for the combined result.

\textbf{The raw-statistic comparison gains an independent cohort.} Because the raw maximum does not depend on a fitted local map, its A and B maxima can be pooled. The table compares the original and fresh cohorts at each raw observed peak. The pooled probability uses 2,048 experiments and denominator 2,049; no B experiment is reused to train a local map.
\begin{center}\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrr}\toprule
Search & Mass & A count & A $p$ & B count & B $p$ & Pooled $p$\\\midrule
@@RAW_ROWS@@
\bottomrule\end{tabular}\end{center}

\textbf{The local-map floor is part of the result.} A's smallest possible local rank is $1/1025$. The combined observed minimum at @@LOCAL_MASS@@ MeV attains this floor: no A statistic at that mass reaches the observed one. This does not resolve a zero tail probability or quantify how far beyond A's largest statistic the observation lies. Every B scan reaching the same minimum rank is counted as a tie in the global tail. The direct B result is still defined, but the finite map cannot distinguish more extreme local observations within that last rank category.

The local-first calibration uses only B's 1,024 minima. A's leave-one-out minima are not pooled with them because they use different training maps. Reported intervals are exact two-sided 95\% Clopper--Pearson intervals for the respective exceedance probabilities. At an observed-selected local mass, A's pointwise interval is not a simultaneous confidence statement across the scan. Threshold-derived effective counts in Appendix C are descriptive ratios; they are neither additional significance measurements nor independent confirmation of the fitted constant.

\textbf{Scope and reproducibility.} Both procedures calibrate the specified 1 MeV mass grids under the fixed archived sources. They do not include the earlier selection of the 2016 window, uncertainty in the source estimated from data, or a continuous mass optimization. No new upper-limit calibration is performed. The five-page baseline and its numerical ledgers are retained, including the older across-search diagnostic, but the combined physics result always means one common coupling. The new tables are \texttt{calibrated\_summary.csv}, \texttt{calibrated\_curves.csv}, \texttt{threshold\_comparison.csv} and \texttt{sidak\_fit.json} in \texttt{results/}; protocol records fix the independent cohorts and the A-only fit.

\textbf{Statistical context.} Cowan's discussion explains the distinction between pointwise and search-wide probabilities and the independent-test reference. Gross and Vitells give an asymptotic upcrossing approximation for a search statistic; this appendix uses direct simulated scans and does not implement that approximation. Algeri and collaborators compare look-elsewhere procedures and their finite-sample behavior. These sources motivate separating a defined search calibration from a convenient trial-count approximation:

{\small
G. Cowan, \href{https://www.pp.rhul.ac.uk/~cowan/stat/cowan_lee_25jun21.pdf}{\emph{The Look Elsewhere Effect}}, ODSL Journal Club, 25 June 2021, especially slides 4--12.\par
E. Gross and O. Vitells, \href{https://arxiv.org/abs/1005.1891}{\emph{Trial factors for the look elsewhere effect in high energy physics}}, Eur. Phys. J. C \textbf{70} (2010) 525--530.\par
S. Algeri, D. A. van Dyk, J. Conrad and B. Anderson, \href{https://arxiv.org/abs/1602.03765}{\emph{On methods for correcting for the look-elsewhere effect in searches for new physics}}, JINST \textbf{11} (2016) P12010.\par}
'''
    replacements={'AUDIT_ROWS':'\n'.join(auditrows),'RESULT_ROWS':'\n'.join(rows),'FIT_ROWS':'\n'.join(fitrows),
        'RAW_ROWS':'\n'.join(raw),'RAW_MASS':str(int(s.raw_peak_mass_MeV)),'LOCAL_MASS':str(int(s.localfirst_peak_mass_MeV))}
    for key,value in replacements.items():text=text.replace('@@'+key+'@@',value)
    assert '@@' not in text
    return text

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-only',action='store_true');args=ap.parse_args()
    summary=pd.read_csv(B/'results/calibrated_summary.csv',dtype={'scope':str}).set_index('scope')
    assert (summary.sidak_fitted_p < summary.minp_B_p95_low).all(), 'Revisit the written Sidak conclusion if results change.'
    fits={r['scope']:r for r in json.loads((B/'results/sidak_fit.json').read_text())['fits']}
    audit=json.loads((B/'qa/common_coupling_audit.json').read_text());assert audit['passed']
    threshold=pd.read_csv(B/'results/threshold_comparison.csv',dtype={'scope':str})
    text=parent_text();assert text.count(r'\end{document}')==1
    text=text.replace(r'\end{document}',appendix(summary,fits,audit,threshold)+r'\end{document}')
    (B/'source/report.tex').write_text(text)
    if not args.source_only:
        command=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic'
        subprocess.run([command,'--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(B/'source/report.tex')],cwd=B/'source',check=True)
    print('Wrote v6.4.4 report from pinned parent source and current tables; parent numerical records unchanged.')

if __name__=='__main__':main()
