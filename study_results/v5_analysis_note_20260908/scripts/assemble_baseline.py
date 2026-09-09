"""Create the curated v5 baseline without changing any frozen source."""
from pathlib import Path
import re, json
B=Path(__file__).resolve().parents[1]; R=B.parents[1]
S=B/'source'; old=R/'study_results/harvard_writing_sample_final_combinations_20260902/source'
# Always start baseline edits from the frozen Harvard prose.
for name in ['01_introduction','02_datasets','03_event_selection','04_methodology','05_toys_validation','05a_2021_support_selection','06_selected_results']:
 (S/'sections'/f'{name}.tex').write_text((old/'sections'/f'{name}.tex').read_text())
def edit(name,fn):
 p=S/'sections'/f'{name}.tex'; p.write_text(fn(p.read_text()))
def intro(t):
 t=t.replace('In v4 the tested','In v5 the tested').replace(r'\SIrange{40}{300}{MeV}',r'\SIrange{36}{300}{MeV}')
 t=t[:t.index(r'\subsection{Scope of this note}')]+r'''\subsection{Scope of this note}

Version 5 consolidates the detector description, statistical method, validation
history and observed results for the proposed full-2021 unblinding review. The
current data comprise full 2015, full 2016 and the released 2021 10\% sample.
No additional 2021 events enter this document. The 2019 run provides detector
context but does not contribute an observed spectrum.

The current seven-scope release uses the frozen 14--135, 30--210 and 36--300~MeV
GP supports for 2015, 2016 and 2021. Earlier configurations are identified as
historical when they are needed to explain an analysis choice. Observed limits
use the bounded, piecewise-asymptotic 90\% \CLs{} construction with profiled
backgrounds. Pointwise expected-limit bands, empirical limit-tail diagnostics,
and the later conditional global-probability studies answer different questions
and are presented separately.

The full-2016 result retains a disclosed cross-process state-replay exception.
Later calibration and global studies also expose sensitivity to the assumed
background spectrum. This note records those limitations as issues for the
unblinding review; it does not certify their resolution or authorize opening
additional data. Appendix~\ref{sec:v5-history} records how the recent studies
changed the analysis and the presentation.
'''
 return t
edit('01_introduction',intro)
def dataset(t):
 a=t.index('We consider two related analyses.'); z=t.index('Table~',a)
 t=t[:a]+r'''The current analysis uses full 2015 and 2016 engineering-run samples and the
released 2021 10\% physics-run sample. The same frozen campaign configuration
enters both standalone and shared-coupling results. The engineering data share
the baseline tracker used in the published prompt searches; the 2021 sample
uses the upgraded tracker and positron hodoscope. No 2019 data enter the fits.

'''+t[z:]
 t=t.replace(r'\SIrange{40}{300}{MeV}',r'\SIrange{36}{300}{MeV}')
 a=t.index('The 2021-only study changes'); z=t.index('Integrated luminosity',a)
 t=t[:a]+r'''The 36--300~MeV support was first developed in a separate 2021 study and was
subsequently used by the v4.9.12 final-sample combination. The older
40--300~MeV support belongs to the historical combination and is not the
configuration of the current result curves.

'''+t[z:]
 return t
edit('02_datasets',dataset)
def methodology(t):
 a=t.index(r'\subsection{Search interval and GP fit support}'); z=t.index(r'\subsection{Observable, scan grid, and blinding}',a)
 t=t[:a]+r'''\subsection{Search interval and GP fit support}
\label{sec:fit-support-policy}

The search interval specifies where a resonance is tested; the wider GP fit
support specifies which bins can constrain its background. At each mass, bins
inside the moving $\pm2.25\sigma_m$ exclusion are omitted from GP training.
Keeping training data beyond the search endpoints avoids unnecessarily
one-sided interpolation. The current settings are given in
Table~\ref{tab:v4p9-card-lineage} and apply to both standalone and combined fits.

\begin{table}[htbp]
\centering
\begin{tabular}{lrrr}
\toprule
Data & Search [MeV] & GP support [MeV] & Upper length factor \\
\midrule
2015 full & 19--90 & 14--135 & 8 \\
2016 full & 39--180 & 30--210 & 12 \\
2021 10\% & 50--250 & 36--300 & 15 \\
\bottomrule
\end{tabular}
\caption{Frozen settings of the current v4.9.12 input release inherited by v5.
The 2016-inclusive results remain conditional on the recorded state-replay
exception. The 2021 lower edge follows the conditional support study;
its earlier 40~MeV value is retained only in historical comparisons.}
\label{tab:v4p9-card-lineage}
\end{table}

The 2016 lower edge at 30~MeV also reflects the actual bin geometry. At the
39~MeV search endpoint, the blind interval begins at 35.063~MeV, whereas a
35~MeV support starts at a bin center of 35.125~MeV and supplies no lower-side
training center. Training masks are defined using bin centers, not only
continuous interval endpoints. The 2021 support study is described in
Section~\ref{sec:v4p9p5-support}; its post-selection practical rule and remaining
55~MeV under-recovery remain part of the qualification of these settings.

'''+t[z:]
 return t
edit('04_methodology',methodology)
def validation(t):
 t=t.replace('This agreement across statistical\nscales is what allows us to say that the analysis procedure is ready for the planned\nfull-data unblinding.',r'''This agreement characterizes conditional extraction recovery for those source
models. It does not alone establish readiness for full-data unblinding. The later
calibration tests in Appendix~\ref{sec:v5-calibration} directly examine rejection
frequencies and reveal limitations that the earlier pull summaries do not test.''')
 return t
edit('05_toys_validation',validation)
edit('05a_2021_support_selection',lambda t:t.replace('2021-only analysis without further tuning.','2021-only analysis without further tuning. The later v4.9.12 release carries\nthis same support into the standalone and combined current-data results.'))
edit('03_event_selection',lambda t:t.replace(r'\subsection{Radiative fraction',r'''The effect of varying this nominal signal width is examined in
Section~\ref{sec:v5-resolution-results}. That comparison changes the extraction
template while retaining the archived GP prediction; it is a model-dependence
study and does not refit the detector resolution from the observed candidates.

\subsection{Radiative fraction''',1))
# The existing selected-results description remains an archived observed-only subset.
edit('06_selected_results',lambda t:t.replace('The\nremaining 2021 data will be unblinded shortly, so the curves below describe the\nobserved result in hand rather than projecting the reach of the full 2021 sample.',r'''The remaining 2021 data are outside the reviewed sample. The curves below
describe the observed result in hand and do not project the full-2021 reach.''').replace('No expected-limit bands or toy-calibrated limit ensembles are included at this\nstage.',r'''This subsection reproduces the observed-only release. The pointwise expected
bands and empirical limit-tail study are added in the next subsection.''').replace('Without a\nscan-wide background-only calibration, none of these curves supports a statement\nabout global significance.',r'''The conditional GP global studies are discussed separately in
Section~\ref{sec:v5-global}; they do not change these local asymptotic curves.'''))
# Main preamble from polished full-format excerpt.
p=(old/'writing_sample.tex').read_text().split(r'\title{')[0]
p=p.replace('Selected Sections 2--6','Analysis Note v5.0.0').replace('Harvard fellowship writing sample; edited excerpt from a collaborative HPS analysis note','Draft for full-2021 unblinding review').replace(r'\small Writing Sample',r'\small v5.0.0 / Review Draft')
p+=r'''
\usepackage{pdflscape}
\newcommand{\fig}[4]{\begin{figure}[p]\centering
\includegraphics[width=#1\linewidth,height=0.78\textheight,keepaspectratio]{#2}
\caption{#3}\label{#4}\end{figure}}
\makeatletter
\newcommand{\startappendixcontents}{%
\let\vfiveoriginaladdcontentsline\addcontentsline
\renewcommand{\addcontentsline}[3]{%
\vfiveoriginaladdcontentsline{##1}{##2}{##3}%
\def\vfivefirst{##1}\def\vfivetoc{toc}%
\ifx\vfivefirst\vfivetoc\vfiveoriginaladdcontentsline{apc}{##2}{##3}\fi}}
\newcommand{\appendixcontents}{\section*{Appendix contents}\@starttoc{apc}}
\makeatother
\title{HPS Gaussian-Process Resonance Search\\[0.5em]
\large Analysis Note v5.0.0\\Draft for full-2021 unblinding review}
\author{Emrys Peets\\Stanford University and SLAC National Accelerator Laboratory
\and Matthew Gignac\\SLAC National Accelerator Laboratory
\and Eden Hsu\\Stanford University and SLAC National Accelerator Laboratory}
\date{8 September 2026}
\begin{document}
\maketitle
\begin{abstract}
This note consolidates the HPS Gaussian-process prompt-resonance analysis and
its subsequent validation studies for review before full-2021 unblinding.
The observed inputs are full 2015, full 2016 and the released 2021 10\% sample.
A common coupling combines the active campaigns through a profiled Poisson
likelihood with correlated Gaussian constraints on the interpolated background.
The primary displayed limits are pointwise, bounded, asymptotic 90\% \CLs{}
limits. Conditional expected-limit bands, empirical tail refinement,
background-calibration comparisons, mass-resolution variations and candidate
extractions are presented with their respective assumptions. The GP global
study supplies a computationally efficient account of correlations across
mass, but its strongest conditional tails expose sensitivity to the generating
background. The inherited 2016 numerical exception and the remaining
background-validation questions are therefore explicit review items.
This is a working consolidation, not a completed authorization or a final
scan-calibrated discovery result.
\end{abstract}
\tableofcontents
\clearpage
\input{sections/01_introduction}
\FloatBarrier\clearpage
\input{sections/02_datasets}
\FloatBarrier\clearpage
\input{sections/03_event_selection}
\FloatBarrier\clearpage
\input{sections/04_methodology}
\input{sections/v5_method_updates}
\input{sections/v5_global_significance}
\FloatBarrier\clearpage
\input{sections/05_toys_validation}
\FloatBarrier\clearpage
\input{sections/06_selected_results}
\input{sections/v5_results_updates}
\FloatBarrier\clearpage
\input{sections/v5_review_summary}
\FloatBarrier\clearpage
\appendix
\startappendixcontents
\input{sections/v5_history}
\FloatBarrier\clearpage
\appendixcontents
\clearpage
\input{sections/v5_calibration_appendix}
\FloatBarrier\clearpage
\input{sections/v5_traditional_appendix}
\FloatBarrier\clearpage
\input{sections/v5_deficit_appendix}
\FloatBarrier\clearpage
\input{sections/v5_historical_appendix}
\FloatBarrier\clearpage
\input{sections/v5_lowmass_appendix}
\FloatBarrier\clearpage
\let\addcontentsline\vfiveoriginaladdcontentsline
\bibliographystyle{unsrt}
\bibliography{hps_gpr_analysis_note}
\end{document}
'''
(S/'main.tex').write_text(p)
print('Baseline prose updated; main.tex assembled')
