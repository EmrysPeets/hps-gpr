from pathlib import Path
import shutil,re,json,hashlib,csv
import fitz
R=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow')
B=R/'study_results/v5p0p5_analysis_note_20260916';S=B/'source';F=B/'figures';D=B/'derived'
P=R/'study_results/v5p6p0_2021_peak_projection_20260912';E=R/'study_results/v5p6p1_2021_upper_limit_echoes_20260913';G=R/'study_results/v5p7p1_geometry_acceptance_20260913';H=R/'study_results/v5p5p4_combination_rate_bands_20260913'
def put(name,text): (S/'sections'/name).write_text(text.strip()+'\n')
def cp(src,name):shutil.copy2(src,F/name)
for yr,page,box in [('2015',4,(54,53,299,300)),('2016',16,(54,386,299,554))]:
 d=fitz.open(B/'inputs'/f'hps_{yr}_publication.pdf');o=fitz.open();rect=fitz.Rect(box);p=o.new_page(width=rect.width,height=rect.height);p.show_pdf_page(p.rect,d,page,clip=rect);o.save(F/f'published_{yr}_limit_vector.pdf')
intro=(S/'sections/01_introduction.tex').read_text();a=intro.index('The published HPS prompt searches');z=intro.index('\\subsection{Complementary',a)
intro=intro[:a]+r'''The 2015 and 2016 engineering runs established the prompt-resonance analysis
used as the reference for this work. The 2015 search used
\SI{1170}{nb^{-1}} at \SI{1.056}{GeV} and covered \SIrange{19}{81}{MeV}.
Its smallest local $p$-value was $1.7\times10^{-3}$ at \SI{37.7}{MeV};
after accounting for the mass scan, the global $p$-value was 17\%
\cite{HPS2015DarkPhoton}. The 2016 search used \SI{10608}{nb^{-1}}
at \SI{2.3}{GeV} over \SIrange{39}{179}{MeV}. The smallest local
$p$-value, $6.38\times10^{-3}$ at \SI{94}{MeV}, corresponded to about
$1\sigma$ globally \cite{HPS2016PromptLong}. Neither search found evidence
for a prompt dark photon.

Figure~\ref{fig:published-hps-context} reproduces the published coupling limits
at their original confidence levels. The 2015 result is shown in its published
world-data context; the 2016 result includes its background-only expected bands.
These provide the experimental benchmarks for the GPR analysis. Comparisons
must also account for the confidence construction, mass resolution and
radiative-fraction treatment; changing the background fit alone does not
make two limits directly equivalent.

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.48\linewidth}
\includegraphics[width=\linewidth]{../figures/published_2015_limit_vector.pdf}
\caption{2015: published 95\% power-constrained limit.}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.48\linewidth}
\includegraphics[width=\linewidth]{../figures/published_2016_limit_vector.pdf}
\caption{2016: published 95\% prompt limit and expected bands.}
\end{subfigure}
\caption{Published HPS prompt-search limits, reproduced from
Refs.~\cite{HPS2015DarkPhoton,HPS2016PromptLong} as vector figures.
The historical exclusions and lepton-anomaly region in (a) are those of the
2015 publication. In (b), green and yellow show the published 68\% and 95\%
background-only quantile ranges. Both panels retain the original analysis
and systematic treatment; no confidence-level rescaling is applied.}
\label{fig:published-hps-context}
\end{figure}

'''+intro[z:]
a=intro.index('The relevant comparison');z=intro.index('\\subsection{Motivation',a)
intro=intro[:a]+r'''Figure~\ref{fig:hps-engineering-phase-space} places the prompt search in the
visible-dark-photon parameter space. The overview uses the updated poster
artwork, with the published HPS engineering-run limits at 95\% CL and the
conditional combined projection at 90\% CL. The lower panels compare the
three-campaign and four-campaign projections on the same axes. The displaced
reach is shown separately because it uses a different search channel and a
simulation-based exposure assumption.

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{../figures/figure2_updated_overview.pdf}\\[3pt]
\includegraphics[width=\linewidth]{../figures/figure2_updated_panels.pdf}
\caption{Visible dark-photon parameter space. Top: archived exclusions,
published HPS 2015 and 2016 limits (95\% CL), the four-campaign prompt
projection (90\% CL), and the poster's simulated displaced full-luminosity
reach. The gray contours and thermal target are historical context, rather
than a complete 2026 survey; their sources and display conventions are
listed in Appendix~\ref{app:figure2-projections}. Lepton-anomaly lines show
central-value references only: $\Delta a_\mu=38\times10^{-11}$ (compatible
with zero) and $\Delta a_e\simeq0.34\times10^{-12}$
\cite{MuonWP2025,ElectronFan2023,AlphaMorel2020}.
(a) 2015+2016+2021; (b) 2015+2016+2019+2021. The prompt projections preserve
the current-sample structure and apply statistics-only density scaling to
full exposure. The 2019 input is a nominal 1\% sample with a 2021-response
proxy. These curves are conditional observed-equivalent projections;
they are not expected sensitivity curves or observed full-data exclusions.}
\label{fig:hps-engineering-phase-space}
\end{figure}

'''+intro[z:]
intro=intro.replace('The 2019 run provides detector\ncontext but does not contribute an observed spectrum.','The 2019 run enters only the explicitly identified Figure~\\ref{fig:hps-engineering-phase-space}\nprojection; it does not enter the observed results in Section~\\ref{sec:selected-results}.')
# Avoid relying on an inherited label that may differ.
intro=intro.replace('Section~\\ref{sec:selected-results}','the results section')
(S/'sections/01_introduction.tex').write_text(intro)

for n in ['HPS_v5p7p1_geometry_illustration.pdf','HPS_v5p7p1_acceptance_overview.pdf','HPS_v5p7p1_response_spectra.pdf']:cp(G/'source/figures'/n,n)
put('v505_geometry_main.tex',r'''
\subsection{Acceptances by campaign and geometry}
\label{sec:v505-geometry}
The different mass ranges of the three campaigns partly reflect the boost
of the pair and the active tracker geometry. For an on-axis parent with
energy $E_A=xE_{\rm beam}$, equal sharing gives the small-angle reference
$m\simeq xE_{\rm beam}\theta$, where $2\theta$ is the opening angle.
At $x=1$, a 15 mrad single-track angle corresponds to approximately
15.84, 34.50 and 56.10 MeV in 2015, 2016 and 2021. These scales help explain
the movement of the low-mass turn-on with beam energy. They do not set a
hard mass threshold: azimuth, momentum sharing, magnetic bending and missed
planes all affect whether a pair crosses enough active silicon.

The v5.7.1 study follows prompt $A'\to e^+e^-$ decays through representative
LCDD sensor geometries and the campaign field maps. The daughter momenta
are generated from exact two-body kinematics, rotated into the detector
frame and propagated numerically. A sensor is hit only when the trajectory
crosses its finite active face. Axial and stereo views are counted separately
and then paired within a station when a three-dimensional point is required.
The event displays in Figure~\ref{v571:geometry-figure} use passing trajectories
from this same calculation; they are generated examples, not reconstructed
data events. Transverse distances are expanded to make the sensor faces and
charge-dependent bending visible.

The 2015 and 2016 curves require at least five paired stations per track;
the 2015 positron also requires paired L1 and L2 hits. The 2021 study instead
uses at least ten individual strip-plane hits for the positron and eight for
the electron. This is the explicit v5.7.1 geometry-study specification.
The archived prompt-TC histogram does not establish equivalence to this
hit-count convention, so the geometric curves do not alter the event
selection recorded above.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{../figures/HPS_v5p7p1_geometry_illustration.pdf}
\caption{Active sensor geometry and accepted daughter trajectories for the
three campaigns, as shown on slide 5 of the September analysis meeting.
Solid and dashed trajectories illustrate the first and last masses with a
passing orientation on the saved grid. Those examples do not establish
physical acceptance endpoints. Red denotes positrons and blue electrons.}
\label{v571:geometry-figure}\end{figure}

Figure~\ref{v571:overview} averages the hit test over 4096 deterministic decay
orientations per mass, weighted for a transversely polarized vector and
sampled at 5 MeV intervals. At full beam energy, the sampled maxima occur
at 40, 90 and 195 MeV, with accepted fractions of about 26.1\%, 25.9\% and
40.1\%, respectively. Curves at $x=0.8$, with zero field, and with all stations
required separate the effects of boost, bending and the hit requirement.
The larger 2021 fraction also depends on its different sensor layout and
selection; it cannot be attributed to beam energy alone.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{../figures/HPS_v5p7p1_acceptance_overview.pdf}
\caption{Conditional geometry-and-field fractions underlying slide 6.
Solid curves use $x=1$ and the campaign hit requirements, dashed curves use
$x=0.8$, dotted curves set the field to zero, and the faint dash--dotted
curves require every station. The parent energy, direction and polarization
are fixed by the stated model. These fractions omit reconstruction, trigger,
material interactions and the remaining analysis cuts.}
\label{v571:overview}\end{figure}

These plots describe which generated daughter trajectories cross the tracker.
A signal efficiency for the resonance search would additionally require the
production energy and angular distributions, material transport, dead channels,
hit finding, trigger and reconstruction, followed by the full event selection.
The geometry calculation therefore supplies no replacement normalization for
the observed limits. Appendix~\ref{app:v505-geometry} documents the geometry,
field units, integration, hit tests and numerical checks in detail.
''')
p=S/'sections/03_event_selection.tex';t=p.read_text();k=t.index('\\subsection{Kinematic distributions');t=t[:k]+'\\input{sections/v505_geometry_main}\n\\FloatBarrier\n\n'+t[k:];p.write_text(t)

# Preserve the detailed derivation and tested numerical settings from the standalone study.
kin=(G/'source/sections/kinematics.tex').read_text();geo=(G/'source/sections/geometry_results.tex').read_text()
geo=re.sub(r'\\clearpage\s*\\begin\{landscape\}.*?\\end\{landscape\}\s*\\clearpage','',geo,flags=re.S)
# The last landscape need not be followed by clearpage.
geo=re.sub(r'\\clearpage\s*\\begin\{landscape\}.*?\\end\{landscape\}','',geo,flags=re.S)
for old,new in [('\\subsubsection{','\\paragraph{'),('\\subsection{','\\subsubsection{'),('\\section{','\\subsection{')]:
 kin=kin.replace(old,new);geo=geo.replace(old,new)
geo=geo.replace('p{3.9cm}','p{3.3cm}').replace('p{3.3cm}','p{3.0cm}').replace('p{4.0cm}','p{4.0cm}')
geo=geo.replace('the current user correction','the v5.7.1 study specification')
shutil.copytree(G/'source/tables',S/'geometry_tables',dirs_exist_ok=True)
geo=geo.replace('\\vFiveSevenRoot/tables/','geometry_tables/')
# Include the sampled magnetic table removed with its original landscape page.
geo=geo.replace('Table~\\ref{v571:magnetic-summary} records', 'Table~\\ref{v571:magnetic-summary} records')
geo += r'''
\begin{table}[htbp]\centering\small
\input{geometry_tables/magnetic_table.tex}
\caption{Saved magnetic response at $x=1$. The first and last nonzero grid
points are sampled observations, not exact endpoints.}\label{v571:magnetic-summary}
\end{table}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{../figures/HPS_v5p7p1_response_spectra.pdf}
\caption{Conditional geometry fractions and the separately measured selected
pair spectra. The spectra retain their native exposures and selections;
their shapes and integrals are not signal efficiencies or production rates.}
\label{v571:spectra}\end{figure}
'''
put('v505_geometry_appendix.tex',r'\section{Geometry and magnetic-transport calculation}\label{app:v505-geometry}'+'\n'+kin+'\n'+geo)
refs=(G/'source/references.tex').read_text();bib=S/'hps_gpr_analysis_note.bib';bt=bib.read_text()
for key,body in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',refs,flags=re.S):
 if '@misc{'+key not in bt:bt+='\n@misc{'+key+',\n title={Geometry study source},\n note={'+body.strip().replace('the current user correction supersedes their older 2021 hit-count statements for this study','the v5.7.1 specification supersedes their older hit-count statements within that study')+'}\n}\n'
bib.write_text(bt)

# Exact saved tables: no transcription of approximate slide labels.
cat=sorted(csv.DictReader((P/'derived/catalogue.csv').open()),key=lambda x:float(x['mass_MeV']))
echo=list(csv.DictReader((E/'derived/echo_catalogue.csv').open()))
shutil.copy2(P/'derived/catalogue.csv',D/'v560_catalogue.csv');shutil.copy2(E/'derived/echo_catalogue.csv',D/'v561_echo_catalogue.csv')
def table(lane):
 out=r'''\begin{table}[htbp]\centering\small
\begin{tabular}{rrrrr}\toprule
$m_0$ [MeV] & $Z_f$ & $\sqrt{k}Z_f$ & Yield-scaled $Z_A$ & Toy $Z$ median [16,84\%]\\\midrule
'''
 for x in cat:
  if x['lane']!=lane:continue
  out+=f"{float(x['mass_MeV']):.0f} & {float(x['source_Z']):.2f} & {float(x['target_Z']):.2f} & {float(x['naive_yield_asimov_Z']):.2f} & {float(x['Z_median']):.2f} [{float(x['Z_q16']):.2f}, {float(x['Z_q84']):.2f}] "+r'\\'+'\n'
 out+=r'''\bottomrule\end{tabular}
\caption{Source local scores, exposure-scaling targets, direct yield-scaled
Asimov scores and the fixed-mass scores of 20 target-matched Poisson toys per
scenario. Quantiles describe this small conditional ensemble.}
\label{tab:v505-'''+lane+r'''}\end{table}'''
 return out
(D/'v505_projection_ten_table.tex').write_text(table('ten'));(D/'v505_projection_one_table.tex').write_text(table('one'))
for scenario in ['ten_65','ten_75','one_65','one_75','one_extra244']:cp(E/'figures'/f'{scenario}.pdf',f'v561_{scenario}.pdf')
cp(P/'figures/ten_65_toys.pdf','v560_ten65_catalogue.pdf');cp(P/'figures/one_65_toys.pdf','v560_one65_catalogue.pdf')
put('v505_projections.tex',r'''
\section{Conditional expectations for the full 2021 sample}
\label{sec:v505-projections}
The released 10\% sample provides a way to ask what the full-sample fit might
look like if one of its local excesses persists. Studies v5.6.0 and v5.6.1
make this assumption explicit: they inject one resolution-shaped peak at a
time into a continuum fitted to the released sample, increase the exposure
by ten and repeat the moving-window analysis. The exercise predicts both a
central excess and its effect on nearby background fits. It uses no observed
full-2021 spectrum.

\subsection{Exposure scaling and the injection strength}
For unchanged selection, signal shape and background composition, an excess
dominated by counting statistics would approximately obey
\begin{equation}
 S_{100}=kS_f,\qquad B_{100}=kB_f,\qquad
 Z_{100}\simeq\frac{kS_f}{\sqrt{kB_f}}=\sqrt{k}\,Z_f,
 \qquad k=1/f.
 \label{eq:v505-scaling}
\end{equation}
Thus $Z_{100}^{\rm target}=\sqrt{10}\,Z_{10\%}$ for the released sample.
This is a conditional persistence estimate, not a maximum possible or most
probable future significance. The regions were selected after inspecting
partial data, and a fluctuation need not grow as a real signal would.
A global probability would also require the mass search and the selection
of these hypotheses to be accounted for.

The saved study compares two amplitudes. Direct yield scaling gives
$A_{\rm yield}=k\max(\widehat A_f,0)$. The second amplitude, $A_{\rm match}$,
is adjusted until the fixed-mass Asimov score reaches the target in
Eq.~\eqref{eq:v505-scaling}. The matched spectrum is deterministic,
$n_i^A=b_i+A_{\rm match}t_i$, and each toy is an independent Poisson draw
from that expectation. Because the GP is conditioned again for each
spectrum and tested mass, direct yield scaling need not reproduce the
simple significance rule. Agreement of the matched injection with the
rule is imposed during construction.

\input{../derived/v505_projection_ten_table.tex}
The largest target among these six regions is $8.88$ at 78 MeV, followed by
$7.58$ at 65 MeV. Their saved toy medians are $8.64$ and $7.63$, respectively.
The region called ``120 MeV'' in the meeting slides is injected at 123 MeV
in the numerical catalogue. The table uses those actual masses and separates
the target from the toy median throughout. The remaining targets are about
$3.7$--$4.8$; none is a forecast of calibrated discovery significance.

\subsection{Nearby fitted deficits and upper-limit structure}
A peak can affect hypotheses outside its own fit window. As the excluded
window moves, some injected counts enter the training sidebands. The GP
then predicts a higher continuum inside a nearby window, which can produce
a negative fitted signal and a tighter upper limit. The signed profile score
retains this response:
\begin{equation}
 r(m)=\operatorname{sign}(\widehat A)
 \sqrt{2[\ell(\widehat A,\widehat\theta)-\ell(0,\widehat\theta_0)]},
 \qquad Z_0(m)=\max(0,r(m)).
\end{equation}
These negative structures are called echoes in the standalone study.
Their width and position depend on the background, resolution and training
exclusion, as illustrated in Figure~\ref{fig:v505-ten-summary}.

At each mass, v5.6.1 profiles the GP-constrained Poisson likelihood
\begin{equation}
 \ell(A,\theta;m)=\sum_i[n_i\log\lambda_i-\lambda_i]
 -\tfrac12\theta^T\theta,\qquad
 \lambda_i=b_i+(L\theta)_i+A\,t_i(m),\quad LL^T=C_b.
\end{equation}
The bounded asymptotic construction gives $A_{90}$ from
$\mathrm{CL}_s(A_{90})=0.10$. To compare the injected and continuum-only
Asimov spectra, the echo calculation uses
\begin{equation}
 R(m)=\frac{A_{90}^{\rm injected}(m)}{A_{90}^{(b)}(m)},\qquad
 D=1-R(m_e),\qquad m_e\in m_0\pm[2,8]\sigma_m(m_0).
 \label{eq:v505-echo}
\end{equation}
The deepest local minimum on each saved flank is selected, with a flank
minimum used when no interior minimum exists. Flanks are clipped to the
scan support. The connected saved-grid interval with $R\leq0.90$ records
the extent of a depression of at least 10\%. These choices precede any
inspection of the full-data spectrum.

\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{../figures/v505_ten_echo_summary.pdf}
\caption{Six alternative injections based on the 10\% sample. Curves show
the matched-Asimov count-limit ratio in Eq.~\eqref{eq:v505-echo}; shaded
central intervals mark $|m-m_0|<2\sigma_m$, outside the echo search.
Each panel uses its own fitted continuum. The alternatives are not a
simultaneous six-peak hypothesis.}\label{fig:v505-ten-summary}\end{figure}

For a 65 MeV injection, the selected Asimov minima occur at 60 and 70 MeV,
with count-limit reductions of 79.2\% and 74.0\%. Both lie at a declared
flank boundary. For the 78 MeV injection the minima occur at 72 and 84 MeV,
with reductions of 77.7\% and 78.2\%. The saved mass grid is 1 MeV, so these
positions should not be read with finer precision.

The coupling plots convert count limits using the prompt density of each
spectrum, whereas $R$ keeps the normalization common:
\begin{equation}
 K_n(m)=\frac{3\pi m f_{\rm rad}^{\rm eff}}{2\alpha}\rho_n(m),\qquad
 \eps^2_{90,n}(m)=\frac{A_{90,n}(m)}{K_n(m)}B_e^{-1}(m).
\end{equation}
Here $f_{\rm rad}^{\rm eff}=0.0477$ and $\rho_n$ uses fractional-bin overlap
within $\pm1.64\sigma_m$, with mass and density in consistent units.
The visible branching correction is one below the dimuon threshold and
$B_e^{-1}=1+\sqrt{1-4u}(1+2u)$ above it, where $u=(m_\mu/m)^2$.
The limit of the matched-Asimov spectrum is not generally the mean or median
of the toy limits, because conditioning, profiling and finding the limit
are nonlinear operations.

\fig{1.0}{../figures/v561_ten_65.pdf}{The 65 MeV example from v5.6.1:
full-range coupling limits, count-limit ratios and signed profile scores.
All 20 saved toy curves are shown. Their spread describes repeated samples
from the specified injection, not a background-only expected band or a
coverage-calibrated interval.}{fig:v505-ten65}
\fig{1.0}{../figures/v561_ten_75.pdf}{The corresponding 78 MeV example.
A central excess can accompany lower upper limits on either side as its
counts enter neighboring GP training regions.}{fig:v505-ten78}

\subsection{What the catalogue can test}
The catalogue provides concrete patterns to compare with a future spectrum:
the signal-shaped excess, its fitted amplitude, and the associated changes
in nearby limits. Agreement in one panel would not validate the generating
background or identify a particle. Real structures can have different
shapes, and several nearby features can interfere through the background
fit. Twenty toys per scenario suffice to illustrate these responses, but do
not determine rare tails, global significance or the chance that a specified
echo will occur. Appendix~\ref{app:v505-one} gives the historical 1\%
projections and selected fit catalogues under their separate source assumptions.
''')
put('v505_one_appendix.tex',r'''
\section{Historical 1\% projections and fit catalogues}
\label{app:v505-one}
The historical nominal 1\% source gives an alternative set of conditional
full-exposure spectra. It is the last-used study histogram,
\texttt{final\_1pct\_invM.root:preselection/h\_invM\_8000}, with its own
reviewed kernel coordinates and 40--300 MeV conditioning support. The
10\% study uses 36--300 MeV. Their saved limit grids are 53--250 and
50--250 MeV, respectively, with 0.625 MeV histogram bins and a training
exclusion of $\pm2.25\sigma_m$. The 1\% lower-edge controls at 50--52 MeV
are excluded from its limit scan.

For this source, Eq.~\eqref{eq:v505-scaling} gives
$Z_{100}^{\rm target}=10Z_{1\%}$ and
$A_{\rm yield}=100\max(\widehat A_{1\%},0)$. The much larger targets in
Table~\ref{tab:v505-one} follow from that conditional prescription. The
selection, trigger-category membership, overlap and effective exposure
relation to the released 10\% histogram remain unverified. These rows
therefore cannot be pooled with the 10\% rows or used to infer an exposure
trend between the two observed samples.

\input{../derived/v505_projection_one_table.tex}
The 86 and 148 MeV entries are source-region boundary controls, rather than
interior observed maxima. The 84, 145 and 244 MeV examples are additional
1\% local maxima. Each is an alternative single injection. The same profile
likelihood, ratio definition and flank rules used in the main text apply.
At 244 MeV, the right-hand search flank lies beyond the saved scan and no
right-hand echo is reported.

\fig{1.0}{../figures/v561_one_65.pdf}{Historical 1\%-based injection at
66 MeV, scaled to full exposure. The selected count-limit minima at 61 and
72 MeV have reductions of 92.9\% and 92.2\% relative to the scenario's
continuum-only Asimov limits; 61 MeV is a flank endpoint.}{fig:v505-one65}
\fig{1.0}{../figures/v561_one_75.pdf}{Historical 1\%-based injection at
82 MeV. This source maximum differs from the 78 MeV maximum selected from
the 10\% sample; the panels describe separate alternatives.}{fig:v505-one82}
\widefig{1.0}{../figures/v560_ten65_catalogue.pdf}{Saved v5.6.0 fit catalogue:
20 Poisson realizations of the 10\%-based 65 MeV injection. The panel
labels and residual conventions are retained from the standalone study.
All draws are shown; none is chosen for resemblance to the data.}{fig:v505-ten-catalogue}
\widefig{1.0}{../figures/v560_one65_catalogue.pdf}{The analogous historical
1\%-based 66 MeV catalogue. Its larger injected strength is conditional on
100-fold exposure scaling and is not an independent prediction to combine
with the preceding catalogue.}{fig:v505-one-catalogue}
''')

# Binning evidence is explicitly bounded to the experiments actually saved.
p=S/'sections/v5_global_results.tex';t=p.read_text();t=t.replace('Between 75 and 76~MeV the combined fit crosses\nzero,','On the saved 1 MeV test-mass grid, the combined signed fit changes sign\nbetween 75 and 76~MeV,')
pos=t.index('\\paragraph{What the stress response means.}')
t=t[:pos]+r'''\paragraph{Dependence on binning.}
The zero crossing refers to a scan of fitted amplitudes, not to a zero in
the measured histogram. Changing histogram bins changes the GP training
counts, the excluded-bin membership and the integrated signal template;
changing the test-mass grid only changes where that response is sampled.
The v5.1.0 fixed-kernel rebinning study gives 2016 signed roots at 76 MeV of
$-2.460$, $-2.545$ and $-2.432$ for 0.25, 0.5 and 1 MeV bins. Its stress
responses are $-14.653$, $-14.620$ and $-14.488$. Thus coarsening those bins
does not remove the 2016 deficit or the large stress offset. The corresponding
2021 roots are $1.808$, $1.429$ and $1.517$ for 0.625, 1.25 and 2.5 MeV bins.

These individual fits do not establish stability of the shared-coupling
zero crossing: the nominal combined root, $0.166$, is small. A joint refit
at each bin width and bin origin would be needed to determine whether the
gate opens at the same mass. No such combined rebinning scan is available
in the saved study. The current statement is therefore restricted to the
nominal grid; the probability spike's location and height should not be
interpreted as a detector-resolved feature. Figure~\ref{fig:v505-binning}
in the appendix shows the archived component checks. Binning should be
chosen using resolution and closure controls, before inspecting which choice
makes the observed probability more striking.

'''+t[pos:];p.write_text(t)
cp(R/'study_results/v5p1p0_binning_significance_20260909/figures/histogram_binning.pdf','v510_histogram_binning.pdf')
shutil.copy2(R/'study_results/v5p1p0_binning_significance_20260909/derived/rebinned_scans.csv',D/'v510_rebinned_scans.csv')
p=S/'sections/v504_global_appendix.tex';t=p.read_text()+r'''
\fig{1.0}{../figures/v510_histogram_binning.pdf}{Archived v5.1.0 histogram-binning
comparison using exact pointwise fits at the saved kernel settings. Histogram
width changes both the GP prediction and the signal fit. These individual
campaign scans are not a rebinned calibration of the combined positive-fit gate.}
{fig:v505-binning}
''';p.write_text(t)

for n in ['rate_fits_with_uncertainties.pdf','spectra_vs_extracted_significance.pdf','likelihood_and_local_reference_comparison.pdf','extraction_model_comparison.pdf']:cp(H/'figures'/n,'v554_'+n)
put('v505_energy_appendix.tex',r'''
\section{Beam-energy and rate hypotheses near 92 MeV}
\label{app:v505-energy}
Studies v5.5.0--v5.5.4 examine why the excesses extracted from different
campaigns need not agree under a common coupling. Two questions are distinct:
whether a feature moves in mass with beam energy, and whether its accepted
amplitude changes with energy after the usual radiative normalization.
A convenient mass hypothesis is
\begin{equation}
 m_y=m_*\left(\frac{E_y}{E_*}\right)^\alpha,
\end{equation}
with $\alpha=0$, $1/2$ and $1$ representing a fixed mass, square-root scaling
and linear scaling. These are comparisons among proposed shapes, not
established production laws. The later studies concentrate on a fixed
92 MeV region and relax the relative campaign rates.

\subsection{A common experiment for the rate comparison}
For campaign $y$ and bin $j$, the model can be written
\begin{equation}
 \lambda_{yj}=b_{yj}+(L_y\theta_y)_j+
 a\,K_y(m_y,\sigma_y)\,R(E_y)\,t_{yj}(m_y,\sigma_y),
 \qquad a\geq0,
\end{equation}
where $t_{yj}$ is the bin-integrated Gaussian signal shape, $K_y$ is the
inherited yield conversion and each background nuisance vector has penalty
$\theta_y^T\theta_y/2$. The reference energy is $E_*=2.30$ GeV. The three
rate laws compared in v5.5.4 are
\begin{equation}
 R(E)=1,\qquad R(E)=(E/E_*)^\beta,\qquad
 R(E)=\exp[-k(E-E_*)].
 \label{eq:v505-rate-laws}
\end{equation}
The first is the common-coupling model. In the other two, $a$ is an
equivalent accepted-amplitude normalization at $E_*$; the fitted energy
factor has no demonstrated production or detector interpretation.
Only three beam energies constrain each two-parameter rate law.

The comparison holds the selected bins and GP constraints fixed across
models. The density conversion is evaluated at the 92 MeV MC reference;
shape variations change the bin-integrated template rather than silently
renormalizing the experiment. The power exponent is searched on $[-6,6]$.
Centroid studies subsequently allow a common mass over 90--94 MeV or separate
campaign centroids, and resolution studies compare the stated fixed and
constrained widths. Those changes introduce additional search freedom.

\begin{table}[htbp]\centering\small
\begin{tabular}{lrr}\toprule
Rate law & $\widehat a$ & Conditional 68\% profile slope interval\\\midrule
Common & $3.102\times10^{-6}$ & ---\\
Power & $8.516\times10^{-6}$ & $\beta=-2.987\;[-3.517,-2.470]$\\
Exponential & $9.817\times10^{-6}$ & $k=1.681\;[1.257,2.087]~\mathrm{GeV}^{-1}$\\
\bottomrule\end{tabular}
\caption{v5.5.4 fits at fixed 92 MeV and fully scaled widths. The common
normalization has a conditional 68\% log-Wald interval
$[2.066,4.657]\times10^{-6}$. These uncertainties condition on the
background and response model.}\label{tab:v505-rates}\end{table}

\widefig{1.0}{../figures/v554_rate_fits_with_uncertainties.pdf}{The v5.5.4
common, power-law and exponential rate fits. Shading gives approximate
68\% and 95\% pointwise log-Wald bands, including normalization--slope
covariance. It is not an uncertainty band on the validity of the energy law.}
{fig:v505-rates}

\subsection{Fit improvement and the cost of additional freedom}
At fixed 92 MeV and scaled widths, the common experiment gives raw likelihood
roots of 2.447 for common coupling, 4.180 for the power law and 4.146 for the
exponential. A root $\sqrt{Q_0}$ is not automatically a Gaussian significance
when a rate slope has been fitted. The v5.5.4 Gaussian reference includes
that slope search: the power-law reference evaluated at the Poisson
statistic gives $Z_G=3.809$. Allowing a common centroid to vary over
90--94 MeV gives a slightly larger root, 4.206, but the earlier matching
Gaussian reference gives $Z_G=3.528$. Better likelihood alone does not imply
stronger evidence once the additional hypotheses are counted.

The most flexible retained shape fit has raw root 4.598 and a broader
independent-amplitude reference-envelope value $Z_G=3.443$. That envelope
is neither the fitted model's own calibrated probability nor a bound over
all continuous shapes. These calculations use a predictive Gaussian
reference with $V_y=\operatorname{diag}(\widehat\lambda_{0,y})+L_yL_y^T$;
they do not calibrate direct Poisson tails, the original mass selection or
the full search. The note's resolution-based Sidak factor is not transferred
to these expanded model families.

\widefig{1.0}{../figures/v554_likelihood_and_local_reference_comparison.pdf}{Raw
likelihood roots and the corresponding conditional reference probabilities
for the retained rate and shape hypotheses. Triangles denoting broader
reference families must not be read as the tested energy model's own tail.}
{fig:v505-energy-reference}
\widefig{1.0}{../figures/v554_extraction_model_comparison.pdf}{The same observed
residuals under common coupling, an energy-dependent rate, and additional
shape freedom. The profiled background adjustment remains visible alongside
the signal contribution. These are alternative descriptions of the same
selected data.}{fig:v505-energy-extraction}

\subsection{Combining extracted amplitudes and count spectra}
The extracted-signal comparison checks whether a second representation of
the same data changes the combination. In the fixed Gaussian experiment,
the signed estimates and their covariance retain all amplitude information;
the extracted and joint amplitude log likelihoods agree to
$5\times10^{-14}$ in the saved numerical check. Independent campaigns
already enter through a product likelihood, so fitting their extracted
amplitudes supplies no additional independence factor.

With independent positive rates, the fixed Gaussian statistic is
\begin{equation}
 Q_G=\sum_{y=1}^3\max(z_y,0)^2,\qquad
 Q_G\sim\tfrac18\chi_0^2+\tfrac38\chi_1^2+
          \tfrac38\chi_2^2+\tfrac18\chi_3^2.
\end{equation}
Signed Stouffer and Fisher combinations test different alternatives. Choosing
among them after seeing their results would require another selection
correction. Without a common rate law, an upper bound on the total expected
signal rows, $T=\sum_yK_y\eps_y^2$, remains defined. The Gaussian-reference
construction used in v5.5.4 is
\begin{equation}
 U_{90}=\widehat T-s_T\Phi^{-1}
 \left[0.1\,\Phi(\widehat T/s_T)\right].
\end{equation}
At 92 MeV it gives 30,944 rows for common coupling and 36,523 for independent
rates. These are conditional known-covariance bounds, not calibrated
uncertainties on an absolute production rate.

\widefig{1.0}{../figures/v554_spectra_vs_extracted_significance.pdf}{Comparison
of simultaneous Poisson fits to count spectra with Gaussian fits to signed
extracted amplitudes. Their near agreement is a check of the retained
amplitude information. Alternative Stouffer and Fisher rules answer different
questions and do not add information to the joint fit.}{fig:v505-extracted}
The studies motivate checks of background modeling and campaign-dependent
response after unblinding. They do not establish a beam-energy law or
supply a calibrated acceptance correction. Version-by-version changes and
the standalone PDFs are indexed in the accompanying study archive.
''')

p=S/'main.tex';t=p.read_text().replace('v5.0.4','v5.0.5');t=t.replace('v5.0.5 with revised Figure 2 and full-exposure-equivalent projections, 13 September 2026','v5.0.5: projections, geometry, binning and rate-study integration, 16 September 2026');t=re.sub(r'\\date\{11 September 2026.*?\}',lambda m:r'\date{16 September 2026\\\normalsize Projection and detector-study revision}',t)
t=t.replace('\\input{sections/v5_review_summary}','\\input{sections/v505_projections}\n\\FloatBarrier\\clearpage\n\\input{sections/v5_review_summary}')
t=t.replace('\\let\\addcontentsline\\vfiveoriginaladdcontentsline','\\input{sections/v505_one_appendix}\n\\FloatBarrier\\clearpage\n\\input{sections/v505_geometry_appendix}\n\\FloatBarrier\\clearpage\n\\input{sections/v505_energy_appendix}\n\\FloatBarrier\\clearpage\n\\let\\addcontentsline\\vfiveoriginaladdcontentsline')
p.write_text(t)
p=S/'sections/v5_history.tex';t=p.read_text();i=t.index('\n',t.index('\\label'))+1;t=t[:i]+r'''
\subsection{Version 5.0.5: projections and detector-study integration}
The 16 September revision replaces Figure 1 with vector reproductions of
the published HPS limits and updates Figure 2 from the poster sources while
retaining both projection panels. It adds the v5.6.1 10\%-based conditional
expectations after the observed results, with historical 1\% examples in
an appendix; integrates v5.7.1 geometry and transport; records the saved
binning evidence near 76 MeV; and summarizes v5.5.4 rate studies at the end
of the appendix. The observed result ledgers and nominal binning are unchanged.
The distinction between exposure-scaling targets and empirical toy medians
is explicit. The standalone version archive records source hashes and the
changes between successive studies.

'''+t[i:];p.write_text(t)
# Update legacy Figure 2 convention only for HPS curves.
p=S/'sections/v504_figure2_appendix.tex';t=p.read_text();t+=r'''
\subsection{Display update in v5.0.5}
The top panel now uses the September poster's contour sources and its
simulated displaced full-luminosity outline. The HPS 2015 and 2016 contours
retain their published 95\% confidence level; the legacy $1.64/1.96$ display
rescaling is no longer applied to those two curves. The historical BaBar
and NA48/2 display convention remains as documented above. Both lower
panels use the same original published HPS contours and the unchanged
three- and four-campaign projection arrays. The displaced outline is
recovered from the poster's archived vector source; it is contextual and
is not recalculated by the prompt fit. Colors distinguish the three-campaign
(teal) and four-campaign (dark red) projections.
''';p.write_text(t)
print('Revised source and copied study assets')
