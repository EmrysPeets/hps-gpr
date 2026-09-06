#!/usr/bin/env python3
"""Build an isolated revision while preserving all released study products."""
from pathlib import Path
import hashlib,json,subprocess
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v4p9p16_presentation_extractions_20260906'
LOW=HERE.parent/'v4p9p16_2015_lowmass_side_study_20260906'
NOTE=HERE/'note';OUT=ROOT/'output/pdf'/HERE.name

def replace(text,old,new):
 assert text.count(old)==1,(old[:80],text.count(old));return text.replace(old,new)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows,cols,caption):
 return '\\begin{table}[H]\\centering\\small\n\\begin{tabular}{'+cols+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\bottomrule\end{tabular}'+'\n\\caption{'+caption+'}\\end{table}\n'
def pformat(x):
 if x>=.001:return f'{x:.4f}'
 exp=int(f'{x:.2e}'.split('e')[1]);mant=x/10**exp
 return f'${mant:.2f}\\times10^{{{exp}}}$'
def main():
 NOTE.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 for p in (PARENT/'note').glob('*.tex'):
  t=p.read_text()
  if p.name=='extraction_section.tex':t=t.replace('../figures/','../../v4p9p16_presentation_extractions_20260906/figures/')
  (NOTE/p.name).write_text(t)
 t=(NOTE/'analysis_note.tex').read_text()
 t=t.replace('Analysis note v4.9.16: signal extractions and staged exposure checks','Analysis note v4.9.16: probability audit and signal echoes')
 t=t.replace('combined global search, signal extractions and staged exposure checks','combined search, audited probabilities and signal echoes')
 t=replace(t,'The GP calculation estimates scan-wide probabilities under one archived joint stress background. It does not establish a final discovery significance or calibrate confidence-interval coverage.',
  'Figure 1 now separates the observed fit from background-reference probability diagnostics. Its local probabilities are nominal asymptotic values. The GP scan-wide tails remain conditional on specified background spectra; they do not establish a final particle significance or calibrate interval coverage. The extraction fits and upper limits are unchanged.')
 t=t.replace('\\input{summary_table.tex}\n','')
 t=replace(t,'The upper curve limits the allowed particle contribution at each mass. The probability curves below it ask how often the chosen background would produce something this unusual somewhere in the entire search. The look-elsewhere calculation changes those probabilities and does not automatically rescale the upper limits.',
  'The upper curve limits the allowed particle contribution at each mass. Below it, the signed fit root and nominal local probability describe the same likelihood used in the extraction displays. A separate diagnostic shows what happens when that root is compared with a particular background reference and when the full scan is taken into account. Those probability questions have different assumptions; they cannot be interchanged. The look-elsewhere calculation does not rescale the upper limits.')
 a=t.index(r'\fig{0.97}{../../v4p9p16_combined_global_20260906/figures/combined_observed_limit_and_pvalues.pdf}')
 b=t.index('\n\n',a)
 t=t[:a]+r'''\fig{0.97}{../figures/combined_observed_limit_and_pvalues.pdf}{All 232 combined mass hypotheses are shown. Top: unchanged pointwise, asymptotic 90\% CL$_s$ limit, including the visible branching correction once above the dimuon threshold. Middle: observed signed profiled root $r_m$ from the same native-bin likelihood as the extraction displays. Bottom: nominal local $p_0=\overline\Phi[\max(0,r_m)]$, with the retained convention $p_0=0.5$ for nonpositive roots; this is neither a scan-wide probability nor a new calibration. The former ``common-background Gaussian local'' curve is separated into Figs.~\ref{fig:prob-full}--\ref{fig:prob-zoom}: it approximates the \emph{sampling distribution of the root}, $r_m^*\sim N(a_m,s_m^2)$, under one fixed reference spectrum per year reused throughout the scan. ``Common'' means consistent across mass tests, and ``Gaussian'' describes the statistic's fluctuations, not a Gaussian-shaped mass background. That conditional curve is not another estimate of the local probability drawn here.}{fig:main}

\clearpage
\input{probability_audit_section.tex}
\clearpage
\input{signal_echo_section.tex}'''+t[b:]
 t=replace(t,r'\section{Representative probabilities}',r'\section{Representative conditional probabilities}')
 t=replace(t,'The deficit extension and this revised report have a separate manifest under','The frozen deficit extension has a separate manifest under')
 anchor=r'\begin{thebibliography}{9}'
 addition=r'''The probability audit and current-solver echo replay are retained separately in
\begin{center}\small\path{study_results/v4p9p16_probability_echo_review_20260906/}\end{center}
They preserve every original scan value and source manifest. The new probability ledger contains all 232 masses, exact exceedance counts, and pointwise Monte Carlo intervals. The deterministic echo archive stores fitted expectations and nuisance components for independent reconstruction. No new events were unblinded and no new random toys were generated for this revision.

\clearpage
\begingroup
\def\lowmassfigurepath{../../v4p9p16_2015_lowmass_side_study_20260906/figures}
\input{../../v4p9p16_2015_lowmass_side_study_20260906/note/lowmass_section.tex}
\endgroup

'''
 t=replace(t,anchor,addition+'\\clearpage\n'+anchor)
 t=replace(t,r'\end{thebibliography}',r'''\bibitem{lowmasshps} P.~H.~Adrian et al. (HPS), \emph{Search for a Dark Photon in Electro-Produced $e^+e^-$ Pairs with the Heavy Photon Search Experiment at JLab}, Phys. Rev. D \textbf{98} (2018) 091101. \href{https://arxiv.org/abs/1807.11530}{arXiv:1807.11530}.
\end{thebibliography}''')
 (NOTE/'analysis_note.tex').write_text(t)
 (NOTE/'headline.tex').write_text(r'''The largest positive combined fit is at 66 MeV: $r=2.760$, with nominal local $p_0=0.00289$. This is a selected local reference value, not a global discovery probability. At 76 MeV the fit is much smaller, $r=0.166$. Its former extreme conditional Gaussian tail comes from comparing that value with a reference background whose fitted root is $-8.700$. Rebinning the extraction displays changed neither result. The audit below explains the resulting discontinuities and the limits of the probability interpretation.
''')
 d=pd.read_csv(HERE/'derived/probability_grid.csv').set_index('mass_MeV')
 rows=[[m,f'{d.loc[m,"observed_r"]:+.3f}',pformat(d.loc[m,'nominal_local_p']),why] for m,why in [(66,'Largest positive combined fit'),(21,'Next separated peak; 2015 only'),(92,'Next separated peak with multiple years'),(72,'Deepest combined fitted deficit'),(76,'Most extreme reference-centered score')]]
 (NOTE/'summary_table.tex').write_text(table(['Mass [MeV]','Observed $r$','Nominal local $p_0$','Role in the study'],rows,'rrrl','Observed likelihood summaries. The nominal probability uses the asymptotic reference and the stated nonpositive-root convention. All selected locations incur scan selection; this table makes no global claim.'))
 x=(NOTE/'extraction_section.tex').read_text()
 x=replace(x,'The strongest raw combined deficit is at 72 MeV. Showing it alongside the excesses matters: a background interpolation error can produce both positive and negative residuals, while a positive narrow signal does not by itself explain a nearby missing-event feature. Statistical fluctuations can also produce such patterns. An independent data increment and predictive background checks can distinguish these possibilities more directly than enlarging a selected peak display.',
  r'The strongest raw combined deficit is at 72 MeV. Showing it alongside the excesses matters: the moving background fit can turn a positive injected signal into neighboring negative and positive fitted echoes (Sec.~\ref{sec:echo}). A fitted deficit therefore does not by itself argue against a positive signal elsewhere. Background mismatch and correlated statistical fluctuations can also produce the pattern. The full response and an independent data increment are needed to distinguish these possibilities.')
 x=replace(x,'This provides another reason to avoid presenting the stress-centered tail as a coherent resonance.',
  r'This argues against reading the conditional tail as evidence for a coherent resonance \emph{at 76 MeV}; it does not exclude a background-fit echo of a signal at another mass. The reference offset and the echo mechanism are distinct effects, and neither has been identified as the physical origin of these data.')
 x=replace(x,'Rate agreement alone would still leave detector effects or common background structure to investigate.',
  r'Rate agreement alone would still leave detector effects or common background structure to investigate. A real parent signal and its fitted echoes may both grow with exposure, so growth of several extrema is not independent confirmation of several particles. Compare the complete predicted response, including the signs and relative amplitudes of echoes, in the additional 20\% alone (Sec.~\ref{sec:echo}).')
 (NOTE/'extraction_section.tex').write_text(x)
 # Keep the historical conditional table, making its reference explicit in every heading/caption.
 r=(NOTE/'representative_table.tex').read_text().replace('Asymp. local & GP local & GP global','Nominal local & Ref. Gaussian & Ref. GP global')
 r=r.replace('GP local uses the common-background Gaussian response.', 'Ref. Gaussian is a formal local tail of the reference-root approximation, not a validated particle probability. Ref. GP global adds the scan maximum under the same reference.')
 (NOTE/'representative_table.tex').write_text(r)
 peak=(NOTE/'peak_table.tex').read_text().replace('Principal peak decomposition.','Largest reference-centered score decomposition.')
 (NOTE/'peak_table.tex').write_text(peak)
 pi=(NOTE/'peak_interpretation.tex').read_text().replace('At the profiled principal peak,','At the largest profiled reference-centered score,').replace('The next positive raw root becomes the principal peak.','The next positive raw root becomes the reference-centered extremum.')
 (NOTE/'peak_interpretation.tex').write_text(pi)
 write_audit(d);write_echo()
 run=subprocess.run(['/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(OUT),str(NOTE/'analysis_note.tex')],cwd=NOTE,capture_output=True,text=True)
 (NOTE/'build.log').write_text(run.stdout+run.stderr);print(run.stdout+run.stderr,flush=True)
 if run.returncode:raise RuntimeError('LaTeX build failed')
 pdf=OUT/'HPS_GPR_Analysis_Note_v4p9p16_Probability_Audit_and_Echoes.pdf';(OUT/'analysis_note.pdf').replace(pdf)
 sources=[Path(__file__),HERE/'PROTOCOL.md',PARENT/'MANIFEST.csv',LOW/'qa/numerical_validation.json',LOW/'qa/visual_review.json',*(NOTE.glob('*.tex')),*(HERE/'figures').glob('*.pdf'),*(HERE/'derived').glob('*.csv'),LOW/'note/lowmass_section.tex']
 (HERE/'provenance/report_build.json').write_text(json.dumps({'pdf':str(pdf.relative_to(ROOT)),'pdf_sha256':sha(pdf),'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}},indent=2)+'\n')
 print(pdf)

def write_audit(d):
 t=r'''\section{What the probability audit found}
\label{sec:prob-audit}
\textbf{The fits are consistent; the reference distributions answer different questions.} All 232 masses are present. Recomputing every saved probability and tail count reproduces the original scan. All 15 selected extraction roots agree with their source scans to below $10^{-15}$. Display grouping only adds whole native bins; it cannot change a likelihood root. The apparently more significant extraction panels and the tiny conditional probabilities therefore did not arise from a new fit or a change of binning.

The earlier Figure 1 placed the nominal local curve beside a background-reference Gaussian local curve, then placed scan-wide tails underneath. Their labels did not sufficiently explain that they use different null references. The revised Figure 1 displays the unchanged upper limit, the observed root, and its nominal asymptotic local reference. Figures~\ref{fig:prob-full}--\ref{fig:prob-zoom} retain the original conditional calculation as an explicit diagnostic. This presentation change does not substitute a newly calibrated significance or select a more favorable statistical test.

\subsection*{What ``common-background Gaussian local'' actually means}
For each year, choose one complete generating background spectrum $B_d$. Use that same spectrum at every tested mass; different years retain their different spectra. The entire moving-mask training and signal fit is then applied to it. The fitted root at a mass can have a nonzero reference value $a_m=r_m(B)$, even though the generating spectrum has no added signal. Perturbing its bins gives a response width $s_m$ and correlations across masses.

The Gaussian approximation concerns the resulting \emph{statistic}, $r_m^*\simeq a_m+s_m Z_m$, not a Gaussian-shaped invariant-mass background. At one mass, $Z_m$ has a standard-normal marginal. Across the scan, the $Z_m$ are correlated. The retained local rule is
\[
 p_{\rm ref}(m)=\begin{cases}
 \overline\Phi\!\left[(r_m-a_m)/s_m\right],&r_m>0,\\
 1,&r_m\leq0.
 \end{cases}
\]
The principal global rule compares this observed threshold with the largest standardized score over positive-root masses in a complete background experiment. Under a correct reference distribution, this is a conditional one-sided test. Its probability is not the chance that a particle exists. Its numerical value need not equal the nominal asymptotic $p_0$ in Figure~\ref{fig:main}.

\subsection*{Why the plot jumped and why the 76 MeV tail looked extreme}
There are 23 transitions between positive and nonpositive raw roots; 117 masses are assigned $p_{\rm ref}=1$ by the sign rule. At 75 MeV, $r=-0.666$ and the reference is $a=-10.831$: the standardized score is $10.38$, but the sign rule sets the probability to one. At 76 MeV, $r$ becomes slightly positive, $+0.166$, while $a=-8.700$ and $s=0.979$. The score is then $9.05$, giving the formal Gaussian tail $6.98\times10^{-20}$. The corresponding nominal local value is $0.4342$. The discontinuity is built into the declared sign rule plus the large moving reference offset; it is not evidence that the data suddenly acquire a large peak.

The rule is retained and its gated points are marked. Smoothing this jump or removing the sign requirement would change the test. The former global curve also broke where the GP sample had zero exceedances (76 and 77 MeV), and it showed direct-toy checks at only six representative masses. The revised diagnostics show every mass, with zero counts represented by upper bounds and sparse positive counts by open symbols and intervals.
'''
 rows=[]
 for m in [21,66,72,75,76,78,92]:
  q=d.loc[m];rows.append([m,f'{q.observed_r:+.3f}',f'{q.asimov_r:+.3f}',f'{q.response_sd:.3f}',pformat(q.nominal_local_p),pformat(q.conditional_local_gaussian),f'{int(q.gp_global_k)}/200000',f'{int(q.direct_global_k)}/1000'])
 t+=table(['Mass','$r$','$a$','$s$','Nominal $p_0$','Formal $p_{\rm ref}$','GP global tails','Direct global'],rows,'rrrrrrrr',r'An arithmetic audit, not a table of established particle significances. The formal column extrapolates a Gaussian marginal; global columns give actual exceedance counts for the conditional ordering. Nonpositive roots are gated to one. Counts and pointwise intervals for all 232 masses are in the probability ledger.')
 t+=r'''
\textbf{The defensible tail statement is limited by both the null model and validation.} Zero of 1,000 independent direct spectra gives a one-sided 95\% binomial upper bound of $0.00299$ under that generating model. Zero of 200,000 GP fields gives $1.50\times10^{-5}$ within the Gaussian approximation. Neither is a measured zero, and neither validates a $10^{-20}$ physical tail. At 92 MeV there are only two GP exceedances and no direct exceedance; the point estimate $10^{-5}$ has substantial Monte Carlo uncertainty. Agreement in the bulk or a satisfactory KS diagnostic cannot verify such far tails.

\clearpage
\fig{0.96}{../figures/probability_reference_full.pdf}{Complete conditional probability audit. Top: observed signed root and the background-reference offset; the shade is the response width, not an uncertainty band on the observed fit. Middle: the formal Gaussian local rule and direct local counts; crosses at one mark nonpositive roots. Bottom: global tails for the same reference-centered ordering using all 232 masses, with direct checks at every mass. Bars are pointwise central 95\% binomial intervals. Open symbols have fewer than 25 exceedances. Red or green downward triangles denote one-sided 95\% upper bounds after zero exceedances; blue triangles only indicate Gaussian local values below the $10^{-4}$ display floor. The two meanings are deliberately distinguished: no displayed floor is a measured probability. Correlation between mass points remains; intervals are not simultaneous.}{fig:prob-full}
\clearpage
\fig{0.96}{../figures/probability_reference_zoom.pdf}{Detail of the same frozen scan and probability rules, without refitting or resampling. At 76 MeV the observed fitted signal is near zero, while the reference spectrum produces a large negative root. Centering against it creates the extreme formal local tail. At 75 MeV the observed root is negative, so the sign rule instead assigns one. The global search still covers the full 19--250 MeV grid; restricting the display to 60--100 MeV does not restrict the trials factor. At 92 MeV, two GP exceedances are shown with their interval rather than as a precise resolved tail. Symbols and interval conventions match Fig.~\ref{fig:prob-full}.}{fig:prob-zoom}

\clearpage
\subsection*{Can the asymptotic profiled method support a final analysis?}
Yes, profiling and asymptotic likelihood methods are defensible when their background model, constraints and sampling approximations are qualified~\cite{cowan}. This implementation has not established those conditions for a final discovery probability. The nominal curve assumes an appropriately centered, unit-width root under a suitable background null. The large offsets in the current reference construction show that this assumption does not hold for that construction. Subtracting those offsets answers a different conditional question; it does not, by itself, establish the appropriate physical null. Existing source-fit, 2016 numerical and development-sample qualifications still apply.

The GP trials-factor method~\cite{ananiev} efficiently describes correlations of a suitably modeled significance field. Here it has been extended to retain nonzero offsets and measured response widths. It accelerates a declared probability calculation; it does not decide whether the chosen background is an adequate physical description. The very small conditional tails should be presented as tension with that reference, with explicit approximation and sampling limits, rather than as evidence for a particle. The separate raw-maximum test has tail estimate one for the observed maximum in both stored ensembles because the reference itself produces large positive roots elsewhere. This is another diagnostic of the reference and ordering, not a favorable global correction to choose in place of the first test.

The upper limits and apparent calibration changes should be judged separately. A weaker \emph{observed} calibrated limit is not evidence for a loss of expected sensitivity. Expected sensitivity requires the relevant ensembles under qualified backgrounds, while coverage requires signal-plus-background hypotheses. Neither display rebinning nor the GP look-elsewhere calculation supplies that missing qualification.
'''
 (NOTE/'probability_audit_section.tex').write_text(t)

def write_echo():
 summary=json.loads((HERE/'derived/echo_summary.json').read_text());delta=summary['injection_changes'];roots=summary['absolute_roots']
 t=r'''\section{Can a real peak make signal echoes?}
\label{sec:echo}
\textbf{Yes: the extraction procedure can produce them. Whether it explains the observed pattern remains unresolved.} When the tested mass moves, its excluded signal region moves. A real signal hidden from the background training at its own mass can enter the sidebands for a neighboring test. The background GP then changes its prediction and correlated nuisance constraint. Relative to that changed background, the next fit can prefer a negative signal or a smaller positive signal elsewhere. Every added generating component can remain nonnegative throughout. A negative fitted residual is not a physical removal of events by the positive particle.

This is the \emph{background-regression} GP. The second GP used for global significance emulates the correlated field of scan statistics. Accounting for correlations of background fluctuations in a trials factor does not automatically identify which observed peak, if any, generated other features under a signal alternative.

\subsection*{Replay with the current solver}
The earlier 2021 peak--dip study used ten background and ten signal-plus-background toy spectra. Its paired 66 MeV injection shifted the median roots at 71 and 72 MeV by approximately $-1.65$ and $-1.74$. Its later reverse-injection study used a wider, smooth reference trained outside 60--86 MeV, with a kernel anchored at 66 MeV. The latter is a separate reference from the combined global-probability stress spectrum; the two studies must not be conflated.

We reused that wider smooth reference and the saved positive signal templates, then retrained and profiled with the \emph{current dense solver}. There are 29 test masses (60--88 MeV), four deterministic generating spectra, and 29 reconstructions of the current observed spectrum. This adds 116 deterministic profile tests and no random toys. The full injected yields are 17,142 events at 66 MeV, 19,273 at 78 MeV, and 36,373 for the 65+78 MeV pair. These are selected illustrative strengths from the earlier single-signal fits, not an equal-strength comparison or a joint two-signal fit. The standalone low-mass injection is at 66 MeV, whereas the pair contains 65 MeV; the displayed curves are not an additivity test.

Subtracting the result for the same smooth background isolates the injection-induced change, $\Delta r(m)=r_m(B+S)-r_m(B)$. A positive 66 MeV injection gives negative changes at 71/72 MeV and positive changes near 78/80 MeV. A positive 78 MeV injection gives a smaller positive feature near 65/66 MeV and negative features near 71/72 and 85 MeV. Thus some features currently being inspected are precisely in regions where the analysis response can create echoes.
'''
 rows=[]
 for m in [66,71,72,78,80,85]:
  q=delta[str(m)];rows.append([m,f'{q["inject_66"]:+.3f}',f'{q["inject_78"]:+.3f}',f'{q["double_65_78"]:+.3f}'])
 t+=table(['Test mass [MeV]','66 MeV injection','78 MeV injection','65+78 MeV injection'],rows,'rrrr',r'Current-solver deterministic changes in the 2021 signed root, relative to the same background-only reconstruction. A change is a response diagnostic, not a significance, probability, or observed signal yield.')
 t+=r'''
\clearpage
\fig{0.96}{../figures/signal_echo_dense_replay.pdf}{Signal echoes reproduced with the current dense 2021 solver and archived positive templates. Top: deterministic one-signal responses, the same background-only response, and the observed 2021 10\% root. Middle: subtracting the background-only response isolates positive and negative echoes. Bottom: a 65+78 MeV injection can make a deep fitted deficit without a negative generating component, but the selected pair also produces peaks larger than observed. These are conditional mechanism demonstrations. No amplitudes were optimized to the current scan, no new data were opened, and the curves do not assign probabilities to one or two particles. The small differences from the old reconstruction (maximum $|\Delta r|=0.0273$ for shared deterministic lanes) do not remove the echo pattern.}{fig:echo-current}

\clearpage
\subsection*{What the observed pattern does and does not establish}
The selected full 65+78 MeV injection gives $r(71)=-4.177$, close to the observed $-4.019$. That agreement alone is incomplete: it predicts $r(65)=3.395$ and $r(78)=3.796$, above the observed $2.396$ and $2.809$, and a 72 MeV dip of $-3.915$ versus $-3.186$. The earlier three-quarter-strength pair reduced the peaks but also made the 71 MeV dip shallower. Those handpicked deterministic cases are not a likelihood comparison between one particle, two particles and background structure.

The stored combined background correlation study supports treating nearby extrema as dependent. For example, its GP correlations are $\rho(66,72)=-0.689$ and $\rho(71,78)=-0.661$; the corresponding direct-toy estimates are $-0.693$ and $-0.638$. The 66/78 MeV roots have positive correlation $+0.162$ in the GP approximation ($+0.119$ directly). These correlations concern fluctuations under the declared combined background. The injection replay concerns a change of generating mean in 2021 alone. They are complementary evidence for a correlated scan response, not interchangeable calculations and not proof of the same causal explanation in every year.

It is therefore plausible that some observed positive features or deficits are echoes of another feature. It is also plausible that a mismodeled continuum, detector/selection structure or correlated fluctuations produce them. The present results establish that an independent-particle interpretation of every extremum is unwarranted; they do not decide which physical explanation is correct. The extreme conditional tail at 76 MeV is separately explained by its reference offset and sign gate, not by an observed large 76 MeV signal.

\subsection*{How to test the echo explanation at 30\%}
Freeze a small family of parent-signal hypotheses and a background treatment before inspecting the next data increment. A useful comparison fits a single positive template near 65/66 MeV, a single template near 78 MeV, and a joint positive two-template model. Use a common signal-protected training interval that excludes both candidates for this comparison, qualify its interpolation with predeclared predictive controls, and retain each year's own response and normalization. Compare the complete native spectra and the scan pattern predicted after rerunning the moving-mask procedure. Separate one-template fits cannot supply the joint two-signal amplitude estimates or their covariance.

Generate or reuse properly matched paired background and signal-plus-background experiments for those declared hypotheses. Start with ten complete pilot spectra per hypothesis to check fit closure, response and runtime; those ten do not calibrate a far tail. Scale only after those checks pass, keeping full-spectrum mass correlations and nonoverlapping toy IDs. A formal preference must account for the mass and model choices already examined. Existing toys from a different generating spectrum cannot be relabeled as the new hypothesis.

Most decisively, test the additional disjoint 20\% on its own before examining cumulative 30\%. A parent peak and its echoes may all become visually stronger with more data; persistence of several extrema is not several independent confirmations. Predict their relative signs, shapes and fitted responses together. Background bias can also grow in units of statistical error. The staged-exposure displays below remain useful mean scenarios, but their simple signal overlays do not include a fresh background refit and therefore do not predict the full echo pattern. This revision does not unblind the 30\% or 100\% sample.
'''
 (NOTE/'signal_echo_section.tex').write_text(t)
if __name__=='__main__':main()
