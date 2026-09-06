#!/usr/bin/env python3
"""Append the intervention study to the sealed probability-audited v4.9.16 note."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v4p9p16_probability_echo_review_20260906'
NOTE=HERE/'note';OUT=ROOT/'output/pdf'/HERE.name
NOTE.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ptex(p):
    if p>=.001:return f'{p:.3f}'
    mant,exp=f'{p:.2e}'.split('e');return rf'{mant}\times10^{{{int(exp)}}}'
def table(headers,rows,alignment,caption,label):
    return '\\begin{table}[H]\n\\centering\\small\n'+rf'\begin{{tabular}}{{{alignment}}}'+'\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'+rf'\caption{{{caption}}}\label{{{label}}}'+'\n\\end{table}\n'
for p in (PARENT/'note').glob('*.tex'):
    text=p.read_text().replace('../figures/','../../'+PARENT.name+'/figures/')
    (NOTE/p.name).write_text(text)
main=(NOTE/'analysis_note.tex').read_text()
main=main.replace('combined search, audited probabilities and signal echoes','candidate removal and traditional signal searches')
main=main.replace('Analysis note v4.9.16: probability audit and signal echoes','Analysis note v4.9.16: candidate removal and traditional searches')
main=main.replace('The extraction fits and upper limits are unchanged.','The previous GP extraction fits and upper limits are unchanged. Section~\\ref{sec:candidate-removal} adds candidate-removal experiments and traditional fits to the original data.')
anchor='\\clearpage\n\\input{deficit_section.tex}'
assert main.count(anchor)==1
main=main.replace(anchor,'\\clearpage\n\\input{candidate_removal_section.tex}\n\n'+anchor)
main=main.replace('No new events were unblinded and no new random toys were generated for this revision.','No new events were unblinded and no new random toys were generated for that probability-audit revision. The conditional replacements added in Sec.~\\ref{sec:candidate-removal} are separately documented; they do not update the global-probability ensembles.')
main=main.replace('\\end{thebibliography}',r'\bibitem{hps2016traditional} P.~H.~Adrian et al. (HPS), \emph{Searching for Prompt and Long-Lived Dark Photons in Electro-Produced $e^+e^-$ Pairs with the Heavy Photon Search Experiment at JLab}, Phys. Rev. D \textbf{108} (2023) 012015. \href{https://arxiv.org/abs/2212.10629}{arXiv:2212.10629}.'+'\n\\end{thebibliography}')
(NOTE/'analysis_note.tex').write_text(main)

holes=pd.read_csv(HERE/'derived/holes.csv');remote=pd.read_csv(HERE/'derived/remote_summary.csv');traditional=pd.read_csv(HERE/'traditional/derived/fit_summary.csv')
ht=[]
for h in holes.itertuples():
    ht.append([str(h.dataset),str(h.mass_MeV),f'{h.sigma_MeV:.3f}',f'{h.low_MeV:.3f}--{h.high_MeV:.3f}',str(h.n_bins)])
hole_table=table(['Data','Candidate [MeV]',r'$\sigma_m$ [MeV]','Replaced bin edges [MeV]','Bins'],ht,'lrrrr',r'Primary holes contain whole analysis bins centered within $2.25\sigma_m$ of a selected peak. The tabulated edges include the full boundary bins. Native analysis widths are 0.25 MeV for 2015, 0.25 MeV for 2016 and 0.625 MeV for 2021. No display binning enters a fit.','tab:removal-holes')
rt=[]
for h in remote.itertuples():
    rt.append([str(h.dataset),str(h.remote_masses),f'{h.observed_both_mean:.3f}',f'{h.observed_replicate_min:.3f}--{h.observed_replicate_max:.3f}',f'{h.observed_both_poly_mean:.3f}',f'{h.observed_both_wide_mean:.3f}'])
remote_table=table(['Data','Masses','GP mean','10-fill range','Polynomial',r'Wider GP'],rt,'lrrrrr',r'Fraction of original remote variation retained in the \emph{observed} signed root: $\mathrm{sd}(r_{\rm filled})/\mathrm{sd}(r_{\rm original})$, each standard deviation taken about its own mean. Both candidate regions are replaced. The 10-fill column is the minimum--maximum across ten conditional replicas, not a confidence interval. All columns use the same primary remote mass set, including the wider-hole column.','tab:removal-remote')
rr=[]
for h in remote.itertuples():
    rr.append([str(h.dataset),f'{h.reference_both_mean:.3f}',f'{h.reference_both_poly_mean:.3f}',f'{h.reference_both_wide_mean:.3f}',f'{h.reference_correlation:.3f}'])
reference_table=table(['Data','GP mean','Polynomial','Wider GP',r'GP mean correlation'],rr,'lrrrr',r'The same comparison for the response to the archived \emph{reference} spectrum, $a_m=r_m(B)$. Every fill is learned from that reference\textquotesingle s own retained bins. Correlation is with the original reference response on the primary remote mass set. Poor polynomial interpolation can create additional structure; a smaller or larger ratio does not rank physical background models.','tab:removal-reference')
bt=[]
for h in traditional[traditional.variant.eq('baseline')].itertuples():
    bt.append([str(h.dataset),str(h.mass_MeV),f'{h.gp_root:+.2f}',f'{h.root:+.2f}',rf'${ptex(h.p0_nominal)}$',f'{h.deviance:.1f}/{h.ndof}'])
baseline_table=table(['Data','Mass [MeV]',r'$r_{\rm GP}$',r'$r_{\rm poly}$',r'Nominal $p_0$',r'$D/\mathrm{dof}$'],bt,'lrrrrr',r'Traditional baseline fits to the \emph{original} observed data at the fixed GP-selected masses. The two methods use different background models and windows. Here $p_0=\overline\Phi[\max(0,r)]$ with 0.5 for nonpositive roots. These are uncalibrated local references at masses selected with the same events. In particular, the small 2015/21 MeV value accompanies a poor background description and must not be quoted as a particle significance.','tab:traditional-base')
vr=[]
variants=['baseline','degree_minus','degree_plus','width_minus','width_plus']
for (year,mass),group in traditional.groupby(['dataset','mass_MeV'],sort=False):
    g=group.set_index('variant')
    vr.append([f'{year}, {mass}']+[f'{g.loc[v,"root"]:+.2f} ({g.loc[v,"deviance_per_dof"]:.2f})' for v in variants])
variant_table=table(['Data, MeV','Baseline','Degree $-1$','Degree $+1$','Width $-2\sigma$','Width $+2\sigma$'],vr,'lrrrrr',r'All thirty retained traditional fits: each cell gives signed root $r$ followed by deviance per degree of freedom in parentheses. Width changes refer to the \emph{total} window width. Large roots accompanied by a large deviance expose inadequate local background functions. No row or variant is selected by its resulting probability. Actual edges, yields, covariance matrices and nominal local probabilities are retained in the numerical archive.','tab:traditional-all')

section=r'''\section{Candidate removal and traditional signal searches}
\label{sec:candidate-removal}
\textbf{Removing the candidate regions changes the local peak--dip pattern, but does not remove the oscillatory response.} With both primary regions replaced, the remote observed scan retains 67\%, 78\% and 92\% of its original variation in 2015, 2016 and 2021, respectively. The corresponding reference-spectrum response retains 110\%, 93\% and 86\%. Traditional fits to the original counts give substantially different peak strengths, and even reverse the sign of the 2021 feature at 78 MeV. The evidence supports coupled, background-dependent fitted structure. It does not yet establish a robust particle signal.

The large oscillating ``background'' in the probability study is primarily the \emph{signal-like response to a reference spectrum}, $a_m=r_m(B)$. It is not itself an event-count spectrum. The experiment below therefore addresses both objects separately: the scan of the observed counts, and the scan of the archived reference counts. Modifying one does not redefine probabilities for the other.

\subsection*{What was removed and how it was replaced}
The two leading positive observed GP peaks per year were fixed from the existing extraction catalogue before examining these interventions: 51 and 21 MeV in 2015; 90 and 117 MeV in 2016; 78 and 65 MeV in 2021. Selection did not use the extreme reference-centered probabilities. Full 2015, full 2016 and only the released 2021 10\% sample are used.
@@HOLES@@
The primary replacement GP is trained outside \emph{both} holes, with the frozen kernel from that year's leading selected peak. It predicts the joint latent log-intensity $(\mu,V)$ in the missing bins. The deterministic replacement is the expected count $\exp(\mu_i+V_{ii}/2)$. Ten additional replacements draw a joint latent function, then Poisson counts. The latent covariance does not include a second copy of Poisson observation noise. The archived reference spectrum receives its own deterministic replacements learned only from its own retained bins.

Each region is replaced separately and both are replaced together. The single-region comparisons reuse the same joint draws. Every complete modified spectrum stays fixed throughout its mass scan; every bin outside the chosen holes is exactly unchanged. The original moving-mask GP is then retrained at every hypothesis with its frozen per-mass kernel, followed by the exact profiled fit. This gives 42 spectra per year and 17,430 complete mass tests across the three grids. These conditional replacements ask how retained data and the fitting procedure respond to a local edit. They are not independent background experiments or additional calibration toys.

\clearpage
\fig{1.0}{../figures/observed_candidate_removal.pdf}{Observed scans before and after replacement. Black: original data. Blue and dashed orange: deterministic replacement of the first or second listed region. Green: both. The light green envelope spans the ten paired, conditional GP-plus-Poisson replacements with both holes filled; it is not a confidence band. Gray vertical regions show the primary holes. Curves join the existing 1 MeV hypotheses without extra smoothing. The ordinate is a signed profile root, not a newly calibrated significance. Removing a peak's own bins predictably weakens that feature; the informative changes are in the surrounding pattern and retained regions.}{fig:observed-removal}

\clearpage
\subsection*{What changes locally, and what persists farther away}
The neighboring deficits weaken strongly. At 19 MeV in 2015, $r$ changes from $-3.21$ to $-0.31$ after both replacements; at 102 MeV in 2016 it changes from $-4.50$ to $-1.00$; at 71 MeV in 2021 it changes from $-4.02$ to $-0.72$. The last change is already $-4.02\to-1.48$ when only the 78 MeV region is removed, and $-4.02\to-3.26$ when only 65 MeV is removed. These masses were retained as illustrative deficit checks, not selected to maximize the replacement effect. Their signal-fit windows overlap removed bins, so they are deliberately excluded from the remote metric.

There is also coupling between separated candidate regions. Replacing only the 78 MeV region in 2021 changes the 65 MeV root from $+2.40$ to $+0.98$, although the 65 MeV candidate bins are unchanged. The fitted background uses information shared across hypotheses. Together with the controlled signal injections in Sec.~\ref{sec:echo}, this shows that a peak and its surrounding dips need not be independent features. Excision alone cannot tell whether the original cause was a particle, background mismatch, detector structure, or a fluctuation.

For a stricter comparison, a mass is called remote only when its \emph{entire native signal-fit window} avoids both primary holes. The statistic below measures the remaining scan variation about its own mean. It is a descriptive measure of amplitude, not spectral power at a selected frequency or a probability of oscillation.
@@REMOTE@@
@@REFERENCE@@
All six primary observed/reference comparisons retain at least half the original remote standard deviation and at least two sign changes within contiguous remote intervals. This predeclared routing condition triggered the traditional fits at all six masses. It is not a rejection threshold for a physical null hypothesis. Persistence also does not mean the pattern is unchanged: the 2015 observed deterministic correlation is only 0.56, and across its ten replacements it ranges from 0.07 to 0.82.

\clearpage
\subsection*{Dependence on the replacement model}
The polynomial alternative fits sidebands within $\pm7\sigma_m$, excluding both primary holes: degree five at 2015/21 MeV and degree three elsewhere. Its expected counts fill the holes. The wider GP alternative excludes $\pm3\sigma_m$ around both candidates and conditions the same leading-peak kernel on the retained bins.
\fig{1.0}{../figures/replacement_model_comparison.pdf}{Both-hole replacements in observed counts (left) and the reference spectrum (right). Green: primary GP mean. Purple: exponential-polynomial fill. Dashed red: wider GP holes. The reference ordinate $a_m=r_m(B)$ is the fitted root, not the background event rate. Gray shading marks primary holes; wider holes extend beyond it. All outcomes remain visible, including unsuccessful interpolation. No altered spectrum supplies a new observed-data probability.}{fig:removal-methods}
The wider GP replacement reduces the 2021 reference's remote variation to 38\%, while the observed scan retains 86\%. These use the same primary remote mass set; some windows touch the wider holes. The 2016 reference still oscillates strongly below the removed regions. Persistence therefore has appreciable model dependence.

The polynomial filler can create new distortions. At the 2015 21 MeV reference hole, its sideband deviance is $267.20/33$ degrees of freedom; its filled reference scan retains 3.63 times the original remote variation. At the 2021 65 MeV hole, the polynomial sideband deviances are $96.12/17$ for observed counts and $86.17/17$ for the reference. For a deterministic reference, this deviance is a mismatch scale, not a calibrated goodness-of-fit probability. These are poor smooth descriptions of the retained sidebands. Their altered oscillations do not identify a better physical background. The GP fill is also conditional on its kernel and on excluding both candidate regions; it cannot be treated as known signal-free truth.

\clearpage
\subsection*{Traditional local fits on the original data}
The follow-up uses a positive exponential-polynomial background and a bin-integrated Gaussian signal of fixed mass and archived resolution. All polynomial coefficients and the signed signal yield are fitted together with a Poisson likelihood; the null fit profiles the same coefficients at zero signal. Unlike the GP fit, this diagnostic has no Gaussian background constraint learned from the surrounding spectrum. Its different window and nuisance model change both the fitted signal and its uncertainty.

The choices are inspired by published HPS searches~\cite{lowmasshps,hps2016traditional}, with explicit adaptations to the frozen analysis bins. The 2015 baseline uses a degree-five exponential Chebyshev polynomial and total width $14\sigma_m$ at 21 MeV, and degree three with width $13\sigma_m$ at 51 MeV. For 2016 and 2021 it uses a degree-three exponential Legendre polynomial in a total $8\sigma_m$ window. This is a specified cross-check, not an exact reproduction of the published event selection, binning or mass-dependent window schedule.

The actual baseline edges are 14--28.25 and 34--68 MeV for the two 2015 masses, 75.25--104.75 and 97--137 MeV for 2016, and 56.625--73.5 and 69.125--87.25 MeV for 2021. The 21 MeV window is shifted at the frozen support boundary. Four predetermined alternatives change the polynomial degree by $\pm1$ or the total window width by $\pm2\sigma_m$, one change at a time. All thirty fits are retained.
@@BASELINES@@
There is no uniform confirmation of the GP peak strengths. At 2015/51 MeV the baseline root falls from $+3.14$ to $+0.13$. The two 2016 roots fall from $+3.42,+3.28$ to $+2.00,+1.53$. At 2021/78 MeV the sign changes from $+2.81$ to $-1.35$; all five traditional variants give a negative root. The 2021/65 MeV baseline is $+1.96$, with several alternatives near $+1$. This is substantial background-model dependence, not a comparison of two independent measurements.

\clearpage
\fig{1.0}{../traditional/figures/traditional_2015_display.pdf}{Traditional baseline fits to full 2015 data. Adjacent bins are summed for display: 0.75 MeV at the 51 MeV hypothesis, and 0.25 MeV at 21 MeV. Every fit still uses the original 0.25 MeV bins. Counts and residuals are divided by the actual displayed width, including a partial endpoint group. Bottom: data minus the refitted null background, with counting errors only. The red Gaussian is not the blue total change: the polynomial background also moves. Subtraction induces correlated residuals. The large 21 MeV root accompanies an inadequate overall fit. Table~\ref{tab:traditional-all} retains all variants, including the badly fitting degree-two model at 51 MeV; neither extreme is a defensible physical significance.}{fig:traditional-2015}

\clearpage
\fig{1.0}{../traditional/figures/traditional_2016_display.pdf}{Traditional baseline fits to full 2016 data. The display sums five adjacent native bins to 1.25 MeV; any partial endpoint is retained. Counts, residuals and predictions are divided by each displayed width. Fits and probabilities still use the original 0.25 MeV bins. Both baseline positive roots are weaker than their GP values. Reducing the polynomial degree gives negative yields and larger deviances (Table~\ref{tab:traditional-all}). All alternatives are retained. Numerical convergence does not establish that a background describes the data; nominal local probabilities refer only to that model at the already selected mass.}{fig:traditional-2016}

\clearpage
\fig{1.0}{../traditional/figures/traditional_2021_display.pdf}{Traditional baseline fits to the released 2021 10\% sample. Original 0.625 MeV bins are retained and displayed as counts per MeV. The 78 MeV feature becomes a deficit for every specified conventional fit. The 65 MeV baseline remains a modest positive excess. Its degree-two alternative gives a very large positive root but $D/\mathrm{dof}=28.10$, showing a severe background-model failure (Table~\ref{tab:traditional-all}). Counting bars and the fitted Gaussian component are visual aids, not independent residual significances. No additional 2021 data were opened.}{fig:traditional-2021}

\clearpage
\subsection*{How to interpret the strongest-looking conventional outcomes}
@@VARIANTS@@
The most striking baseline is 2015/21 MeV: $r=5.58$ and the formal local reference is $1.21\times10^{-8}$. But the total fitted spectrum has $D/\mathrm{dof}=92.50/50$; a nominal chi-square goodness-of-fit reference is about $2.4\times10^{-4}$. Adding one polynomial degree changes the root to 3.46 and the deviance to $46.95/49$. Widening the degree-five window reverses the sign and makes the fit much worse. The large baseline number therefore cannot be defended by quoting an asymptotic tail alone. These controls and the existing low-mass side study must be considered together.

The same failure mode is conspicuous in the 2021/65 MeV degree-two fit: $r=8.46$ accompanies $D/\mathrm{dof}=28.10$. A polynomial that cannot follow the continuum can force its residual shape into the Gaussian signal coordinate. The extreme signed deficits in some other low-degree fits have the same interpretive problem. These variants are retained to expose that failure, not to choose a different significance after seeing the outcome.

Goodness of fit is a useful warning here, but not a complete validity test. The masses were selected with the same data, polynomial choices were investigated, and no scan-wide or model-selection correction is included in these nominal local values. Even a satisfactory deviance does not demonstrate unbiased signal extraction or correct coverage. Conversely, weakening under a more flexible background does not by itself prove the absence of a signal. A final analysis needs a background family qualified by independent predictive controls and signal-injection recovery before its tail probabilities can be interpreted physically.

\subsection*{What this implies for presentation and the next data step}
The study shows that removing candidate regions can relax their neighboring deficits, while oscillations elsewhere survive. This is compatible with the fitted echoes demonstrated in Sec.~\ref{sec:echo}; it does not identify the observed parent peak or prove that the remaining pattern comes from a resonance. The especially large reference offsets already occur without an explicitly added signal template. Their sensitivity to the continuum construction is itself a reason to qualify the reference before using its very small centered tails as particle evidence.

A useful next test is to freeze the candidate masses, widths, background checks and complete predicted peak--dip response before examining the additional 20\% that would take 2021 from 10\% to 30\%. Analyze that additional portion on its own as well as in the cumulative sample. Real resonances, detector effects and background-model mismatch can all become more visible with more events, so growth alone is not decisive. The sequential 30\% and 100\% looks and any model choices must be included in the declared final inference procedure. No such unblinding was performed here.

\subsection*{Reproducibility and numerical checks}
All 17,430 profile tests are complete, with no mass or replacement dropped. An independent reconstruction checks the complete fixed spectra, untouched bins, paired Poisson replacements, remote geometry and 5,160 saved GP likelihoods. It also checks all thirty traditional fits using separate 48-point quadrature and likelihood derivatives. The largest root differences are $5.3\times10^{-13}$ for reconstructed GP fits and $2.9\times10^{-11}$ for conventional fits. All conventional final stationarity, covariance, fixed-multistart and doubled-quadrature checks pass. An initial numerical stall in one variant was resolved without changing the statistical model; the attempt and refinement are archived.

The derivative, scripts, exact modified spectra, fit components, all thirty conventional outcomes, independent HEP review and verification reports are retained under
\begin{center}\small\path{study_results/v4p9p16_candidate_removal_20260906/}\end{center}
The new manifest binds these products and this PDF. Numerical agreement establishes reproducibility, not physical background validity or calibrated discovery significance. The original combined limits, Figure~\ref{fig:main}, extraction fits and probability ensembles remain unchanged.
'''
for token,value in [('HOLES',hole_table),('REMOTE',remote_table),('REFERENCE',reference_table),('BASELINES',baseline_table),('VARIANTS',variant_table)]:section=section.replace('@@'+token+'@@',value)
section=section.replace(r'\textquotesingle s',"'s")
(NOTE/'candidate_removal_section.tex').write_text(section)

run=subprocess.run(['/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(OUT),str(NOTE/'analysis_note.tex')],cwd=NOTE,capture_output=True,text=True)
(NOTE/'build.log').write_text(run.stdout+run.stderr);print(run.stdout+run.stderr)
if run.returncode:raise RuntimeError('LaTeX build failed')
pdf=OUT/'HPS_GPR_Analysis_Note_v4p9p16_Candidate_Removal_and_Traditional_Searches.pdf'
(OUT/'analysis_note.pdf').replace(pdf)
sources=[Path(__file__),HERE/'PROTOCOL.md',PARENT/'MANIFEST.csv',*(NOTE.glob('*.tex')),*(HERE/'figures').glob('*.pdf'),*(HERE/'traditional/figures').glob('*.pdf'),*(HERE/'derived').glob('*.csv'),HERE/'traditional/derived/fit_summary.csv',HERE/'qa/numerical_validation.json']
(HERE/'provenance/report_build.json').write_text(json.dumps(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=sha(pdf),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}),indent=2)+'\n')
print(pdf)
