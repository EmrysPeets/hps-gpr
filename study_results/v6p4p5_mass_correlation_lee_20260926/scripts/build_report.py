from pathlib import Path
import shutil
import subprocess
import pandas as pd
B=Path(__file__).resolve().parents[1]


def main():
    s=pd.read_csv(B/'results/correlation_global_summary.csv',dtype={'scope':str}).set_index('scope')
    c=pd.read_csv(B/'results/correlation_summary.csv',dtype={'scope':str}).set_index('scope')
    w=pd.read_csv(B/'results/resolution_counts.csv',dtype={'scope':str}).set_index('scope')
    rows=[];widthrows=[];equiv=[]
    for k in ('2016','2021','combined'):
        r=s.loc[k];label='Combined' if k=='combined' else k
        rows.append(f"{label} & {int(r.mass_MeV)} & {int(r.k)}/1024 & {r.p:.3f} & [{r.low:.3f}, {r.high:.3f}] & {r.Z:.2f} & {r.independent_empirical_p:.3f}\\\\")
        equiv.append(f"{label} & {int(r.local_rank_count)}/1025 & {r.N_toy_equivalent:.1f} & [{r.N_toy_equivalent_low:.1f}, {r.N_toy_equivalent_high:.1f}]\\\\")
        if k!='combined':
            q=w.loc[k]
            widthrows.append(f"{label} & {q.mean_MC_core_width_MeV:.3f} & {q.N_MC_resolution:.2f} & {q.N_legacy_resolution:.2f} & {r.MC_resolution_linear_p:.3f} & {r.MC_resolution_sidak_p:.3f} & {r.p:.3f}\\\\")
            assert r.MC_resolution_linear_p<r.low and r.MC_resolution_sidak_p<r.low
    text=r'''
\clearpage\section*{Appendix E. Mass resolution, correlations and the global penalty}

\textbf{E.1. Nearby mass hypotheses are correlated.} The same fluctuation can produce an excess at several nearby masses because their signal templates overlap. Every saved toy uses one full spectrum per campaign across the complete mass grid. The selected MC shapes, moving fit windows and GP sidebands therefore generate the correlations in the repeated search automatically. This appendix measures those correlations and verifies their role in the existing global result; it adds no new fit or simulation.

\begin{center}\includegraphics[width=\linewidth,height=4.8in,keepaspectratio]{../figures/mass_correlation.pdf}\end{center}
{\small\refstepcounter{figure}\textbf{Figure \thefigure. Correlations measured from complete toy scans.} Top: Pearson correlations of the signed likelihood roots across the original 1,024 A scans. Combined boundaries at 100.5 and 175.5 MeV mark changes in available campaigns. Bottom: mass pairs grouped by $d=|c(m_i)-c(m_j)|/\sqrt{[s(m_i)^2+s(m_j)^2]/2}$. Red and blue are median A and independent-B correlations. Gray shows the central 68\% spread across pairs, not a confidence band. The dotted curve is normalized Gaussian-template overlap with flat independent noise; it omits GP fitting, window changes and MC tails.}\par\medskip

Adjacent 1 MeV hypotheses have median A correlations of @@RHO16@@ for 2016, @@RHO21@@ for 2021 and @@RHOC@@ for the joint search. B reproduces them as @@RHOB16@@, @@RHOB21@@ and @@RHOBC@@. Their strong dependence is measured, not inferred by counting resolution widths. Centering and standardizing the roots here only computes correlation coefficients; it does not redefine the local test statistic.

Correlations become negative a few core widths apart, unlike the positive Gaussian-overlap reference. The complete extraction includes background fitting, moving windows and template changes, which a resolution count cannot summarize. The joint matrix uses the same common-coupling fits as Appendix A.

\clearpage\section*{Appendix E, continued: calibrate the complete correlated search}

\textbf{E.2. Count complete searches, not separate mass points.} Keep the A-derived local map from Appendix A and apply it unchanged to each independent B scan. For an observed local threshold $\alpha$, the global event is
\[
 \min_m p_{A,m}(q_b(m))\le\alpha,\qquad
 \widehat p_{\rm global}(\alpha)=\frac{1+\sum_{b=1}^{1024}\mathbf1[\min_m p_{A,m}(q_b(m))\le\alpha]}{1025}.
\]
One scan counts once even if the fluctuation crosses the threshold at several neighboring masses. This directly incorporates the dependence in Figure 5, including effects extending beyond the nominal mass resolution. At the observed minima it reproduces Appendix B exactly:

\begin{center}\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrr}\toprule
Search & Mass [MeV] & B count & Global $p$ & 95\% interval & $Z$ & Independent control\\\midrule
@@GLOBAL@@
\bottomrule\end{tabular}\end{center}

\begin{center}\includegraphics[width=\linewidth,height=2.85in,keepaspectratio]{../figures/correlation_global_tails.pdf}\end{center}
{\small\refstepcounter{figure}\textbf{Figure \thefigure. Keeping and breaking the mass correlations.} Red is the complete-B tail and its pointwise 95\% interval, conditional on A and the fixed null source. Blue keeps the empirical local distributions but independently resamples toy identities at each mass. Green and orange use the MC-resolution count defined on the next page in the Sidak and linear prescriptions. Black vertical lines mark the observed thresholds. No single-campaign resolution count is assigned to the joint fit.}\par\medskip

\textbf{The control isolates dependence.} With $\widehat F_{B,m}(\alpha)$ the fraction of B ranks at mass $m$ below or equal to $\alpha$, independently resampling each column gives
\[
 p_{\rm independent\ control}(\alpha)=1-\prod_m[1-\widehat F_{B,m}(\alpha)].
\]
This product is evaluated exactly from the empirical marginals. It is a finite-sample counterfactual, not a new global calibration, and it does not assume that those marginals are uniform. At the selected thresholds, breaking the correlations increases the probability from 0.088 to 0.174 for 2016, from 0.226 to 0.418 for 2021 and from 0.074 to 0.133 for the combination. The complete-toy result already includes the reduction caused by correlations; multiplying it by another resolution-based penalty would count the search twice.

\clearpage\section*{Appendix E, continued: test the resolution-count approximation}

\textbf{E.3. What HPS used, and what the current toys support.} The 2016 HPS paper and internal note use a linear approximation, $p_{\rm global}\simeq N_{\rm reg}p_{\rm local}$, with $N_{\rm reg}\simeq W/\sigma_{\rm ave}\simeq30$. This is Bonferroni-like; it is also the first-order expansion of Sidak when the total tail is small. The full Sidak expression is $1-(1-p_{\rm local})^{N_{\rm reg}}$. Using an estimated number of resolution regions does not itself provide the guarantee of Bonferroni over an actual list of tests. The 2015 HPS analysis instead obtained the global mapping from 4,000 complete simulated mass scans.\par
{\small Sources: \href{https://arxiv.org/pdf/2212.10629}{2016 HPS paper}, p.16; local 2016 internal note, Sec.6.1.2, p.44. \href{https://arxiv.org/pdf/1807.11530}{2015 HPS paper}, p.4; local 2015 internal note, Sec.5.3, pp.33--34.}\par

Apply the historical resolution prescription to this study's MC core widths without fitting it to the toys:
\[
 W=m_{\max}-m_{\min},\quad \bar s=\frac{1}{W}\int_{m_{\min}}^{m_{\max}}s(m)\,dm,
 \quad N_{\rm res}=W/\bar s.
\]
The integral uses trapezoids on the stored 1 MeV grid: $W=135$ MeV for 2016 and 180 MeV for 2021. The legacy-count column repeats the same calculation with the inherited Gaussian resolution, solely to connect to the earlier prescription. The two probability columns both use the MC count and the observed calibrated local rank.

\begin{center}\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrr}\toprule
Search & $\bar s$ [MeV] & MC $N_{\rm res}$ & Legacy count & Linear $p$ & Sidak $p$ & Direct B $p$\\\midrule
@@WIDTHS@@
\bottomrule\end{tabular}\end{center}

Both resolution-count predictions fall below the direct B 95\% interval at each observed single-year threshold. These are pointwise comparisons, not a simultaneous goodness-of-fit test. Resolution overlap explains why mass points are dependent, but $W/\bar s$ understates the penalty here. The common-coupling search has campaign-dependent shapes, sensitivities and support; its complete-toy calibration needs no invented average resolution.

\textbf{A Sidak-equivalent description can retain the measured penalty.} If desired, rewrite the direct B result at each threshold as
\[
 N_{\rm eq}(\alpha)=\frac{\log[1-\widehat p_{\rm global}(\alpha)]}{\log(1-\alpha)}.
\]
\begin{center}\small\begin{tabular}{lrrr}\toprule
Search & Observed $\alpha$ & Toy-equivalent $N_{\rm eq}$ & Transformed 95\% interval\\\midrule
@@EQUIV@@
\bottomrule\end{tabular}\end{center}
Inserting this threshold-dependent value into Sidak returns the same toy probability by construction. It is not a universal independent-trial count, a new fitted approximation, or an independent check. The direct complete-scan probability remains the result.

\textbf{Scope and reproducibility.} The unchanged 2016 $\pm2.5u$ and 2021 $[-4,+3]u$ windows, 2015 Gaussian, and single shared coupling are retained. Templates and resolutions are fixed. The B intervals exclude A-map uncertainty, source-estimation uncertainty, earlier analysis choices and a continuous mass optimization. The combined local minimum remains at the $1/1025$ calibration floor; all tied B minima are included. This appendix recomputes all 509,952 B local ranks from the stored A/B arrays and verifies the original global counts. Rebuild commands, frozen input hashes and all plotted numbers accompany the report. No parent study is modified.
'''
    replacements=dict(GLOBAL='\n'.join(rows),WIDTHS='\n'.join(widthrows),EQUIV='\n'.join(equiv),
        RHO16=f"{c.loc['2016','adjacent_A_median']:.3f}",RHO21=f"{c.loc['2021','adjacent_A_median']:.3f}",RHOC=f"{c.loc['combined','adjacent_A_median']:.3f}",
        RHOB16=f"{c.loc['2016','adjacent_B_median']:.3f}",RHOB21=f"{c.loc['2021','adjacent_B_median']:.3f}",RHOBC=f"{c.loc['combined','adjacent_B_median']:.3f}")
    for key,value in replacements.items():text=text.replace('@@'+key+'@@',value)
    assert '@@' not in text
    parent=(B/'source/parent_v644_report.tex').read_text().replace('6.4.4','6.4.5').replace('25 September 2026','26 September 2026')
    report=parent.replace(r'\end{document}',text+r'\end{document}')
    (B/'source/report.tex').write_text(report)
    subprocess.run([shutil.which('tectonic') or '/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(B/'source/report.tex')],cwd=B/'source',check=True)


if __name__=='__main__':main()
