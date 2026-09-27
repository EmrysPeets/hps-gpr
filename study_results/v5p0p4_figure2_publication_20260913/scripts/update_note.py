from pathlib import Path
B=Path(__file__).resolve().parents[1];P=B.parent/'v5p0p4_analysis_note_20260911'
s=(P/'source/sections/01_introduction.tex').read_text()
s=s.replace('shows that context using legacy HPS phase-space contour inputs, with the old\nlepton-anomaly overlays deliberately omitted.', 'shows that context using the archived visible-exclusion contours, together with\nlepton-anomaly central-value references and two full-exposure-equivalent GPR projections.')
s=s.replace('using the historical reach-notebook rescaling.', 'using the historical reach-notebook rescaling.')
s=s.replace('\\graphicorplaceholder{0.98\\linewidth}{context_figs/hps_engineering_phase_space_context.png}', '\\captionsetup{font=small}\n\\includegraphics[width=\\linewidth,height=0.66\\textheight,keepaspectratio]{../figures/figure2_overview_and_projections.pdf}')
s=s.replace('Visible dark-photon phase space used to contextualize the HPS prompt\nengineering-run results.', 'Visible dark-photon phase space and full-exposure-equivalent HPS projections.\nTop: the published engineering-run limits in their world-data context; the dashed\nrectangle marks the common zoom window in (a) and (b). Gray shading is the union\nof the archived visible exclusions, with their original boundaries retained.')
s=s.replace('Older $g-2$ contours from the source\nfolder are intentionally omitted.', '''The lepton central-value lines use $\\Delta a_\\mu=38\\times10^{-11}$
(WP25) and $\\Delta a_e\\simeq0.34\\times10^{-12}$ (Rb20 with Fan23)
\\cite{MuonWP2025,ElectronFan2023,AlphaMorel2020}. These are reference loci,
not confidence bands; WP25 is compatible with zero.
(a) Full-equivalent 2015+2016+2021; (b) full-equivalent 2015+2016+2019+2021.
The latter adds the measured 2019 nominal 1\\% spectrum as a separate likelihood
input with a 2021-response proxy. Both current-sample combined limits are then scaled by the square root
of the current-to-full summed count-density ratio, taking 2021 from 10\\% to
100\\% and, in (b), 2019 from 1\\% to 100\\%. These are conditional
observed-equivalent projections, not expected sensitivities or full-data
exclusions; Appendix~\\ref{app:figure2-projections} gives the construction.''')
(B/'source/sections/01_introduction.tex').write_text(s)
s=(P/'source/main.tex').read_text().replace('\\let\\addcontentsline\\vfiveoriginaladdcontentsline','\\input{sections/v504_figure2_appendix}\n\\FloatBarrier\\clearpage\n\\let\\addcontentsline\\vfiveoriginaladdcontentsline')
s=s.replace('pdfsubject={Draft for full-2021 unblinding review with v5.1.1 diagnostic support}', 'pdfsubject={v5.0.4 with revised Figure 2 and full-exposure-equivalent projections, 13 September 2026}')
(B/'source/main.tex').write_text(s)
bib=(P/'source/hps_gpr_analysis_note.bib').read_text()+r'''

@article{MuonWP2025,
 author={Aliberti, R. and others},
 title={The anomalous magnetic moment of the muon in the Standard Model: an update},
 journal={Physics Reports}, volume={1143}, pages={1--158}, year={2025},
 doi={10.1016/j.physrep.2025.08.002}, eprint={2505.21476}, archivePrefix={arXiv},
 url={https://arxiv.org/abs/2505.21476v3}}
@article{ElectronFan2023,
 author={Fan, X. and Myers, T. G. and Sukra, B. A. D. and Gabrielse, G.},
 title={Measurement of the Electron Magnetic Moment},
 journal={Physical Review Letters}, volume={130}, pages={071801}, year={2023},
 doi={10.1103/PhysRevLett.130.071801}, eprint={2209.13084}, archivePrefix={arXiv},
 url={https://arxiv.org/abs/2209.13084}}
@article{AlphaMorel2020,
 author={Morel, L. and Yao, Z. and Clade, P. and Guellati-Khelifa, S.},
 title={Determination of the fine-structure constant with an accuracy of 81 parts per trillion},
 journal={Nature}, volume={588}, pages={61--65}, year={2020},
 doi={10.1038/s41586-020-2964-7}, url={https://doi.org/10.1038/s41586-020-2964-7}}
'''
(B/'source/hps_gpr_analysis_note.bib').write_text(bib)
(B/'source/sections/v504_figure2_appendix.tex').write_text(r'''
\section{Construction of the Figure 2 projections}
\label{app:figure2-projections}

Figure~\ref{fig:hps-engineering-phase-space} uses the v5.0.4 spectra,
training supports, archived kernel parameters and 2015 extension through
100~MeV. The 2019 input is the measured nominal 1\% spectrum selected with
$p_{e^-}+p_{e^+}>3.64$~GeV. Its declared search range is 75--250~MeV and its
training support is 50--300~MeV. The archived factor-15 response scan supplies
its kernel coordinates; the mass resolution, radiative fraction and conversion
are borrowed from 2021. Every retained 2019 kernel remains at the upper
length-scale bound. This limitation is part of the projection, not a native
2019 response qualification.

At each mass the constituent spectra enter as separate Poisson factors with
a common nonnegative $\epsilon^2$ and independent Gaussian background
constraints, including the correlations among bins within each campaign.
The background is profiled and the bounded asymptotic 90\% \CLs{} endpoint
is solved numerically. The three- and four-campaign projection inputs use
the same implementation. No inverse-limit or significance combination is used.
This figure-specific replay changes the three-campaign numerical endpoints
by at most 2.51\% relative to the frozen note ledger; the earlier observed
limits and their figures in Section~6 are retained. The 2019 saved limits
are not substituted for the newly profiled joint likelihood.

For the full-exposure-equivalent display, let $d_y(m)$ be the observed native-bin
count density in the $\pm1.64\sigma_m$ window, with fractional edge-bin overlaps,
and let $f_y$ be the ratio of full exposure to the input exposure. The plotted
coordinate is
\begin{equation}
 u_{\mathrm{full\,eq}}(m)=u_{\mathrm{current}}(m)
 \sqrt{\frac{\sum_{y\in\mathcal A(m)}d_y(m)}
 {\sum_{y\in\mathcal A(m)} f_y d_y(m)}},
 \qquad (f_{2015},f_{2016},f_{2019},f_{2021})=(1,1,100,10).
\end{equation}
Here $\mathcal A(m)$ contains every declared active campaign. The two curves
therefore coincide below 75~MeV. The minimal-visible branching correction is
applied once above the dimuon threshold. This density rescaling is a
statistics-only approximation: it retains fluctuations and background
structure from the observed samples, assumes exposure-independent selections
and response, and does not generate or analyze future full-data spectra.
There are no new toys, expected-limit bands, global probabilities or coverage
claims in this figure calculation. In particular, the curves are not medians
of a future-data ensemble.

The world-data layer uses the union of the archived exclusion polygons in
$(\log m,\log\epsilon^2)$ coordinates. It preserves source nodes, closed-contour
ordering and missing-interval breaks; no smoothing changes the fitted limits.
The source notebook identifies the archived APEX 2019 physics-run curve as
a projection, so that curve is omitted from the published-exclusion shading.
The world-data layer is a historical context snapshot, not a claim of a
complete September 2026 exclusion compilation.

For the lepton references the positive-vector one-loop relation is
\begin{equation}
 \Delta a_\ell=\frac{\alpha\epsilon^2}{2\pi}
 \int_0^1\!dx\,\frac{2x(1-x)^2}{(1-x)^2+(m_{A^\prime}/m_\ell)^2x}.
\end{equation}
The muon input is the WP25 residual $38(63)\times10^{-11}$
\cite{MuonWP2025}. The electron curve uses the rounded positive residual
$\Delta a_e\simeq0.34\times10^{-12}$ obtained with the Fan23 measurement and
Rb20 fine-structure constant \cite{ElectronFan2023,AlphaMorel2020}.
The published rounded inverse fine-structure constants are
137.035999166(15) from the electron moment and 137.035999206(11) from Rb recoil;
their difference implies this positive residual through the QED derivative.
These lines are central-value loci, not posterior medians or confidence
regions. The muon interval includes zero, and the separate Cs determination
of $\alpha$ gives a negative electron residual that cannot be represented by
a positive contribution from this vector model. No common preferred region
is inferred from the two lines.
''')
print('Updated Figure 2, three references and a dedicated construction appendix.')
