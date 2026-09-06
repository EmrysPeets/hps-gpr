# Ongoing-report integration

An augmented copy of the current presentation/extraction report has already been built in this side study's own output directory. The active parent sources were not edited. The exact captured parent sources and figures are hashed in `provenance/report_parent.json`.

To incorporate the same section into a later build whose main LaTeX source is in a sibling study's `note/` directory, add this immediately before the bibliography:

```latex
\clearpage
\begingroup
\def\lowmassfigurepath{../../v4p9p16_2015_lowmass_side_study_20260906/figures}
\input{../../v4p9p16_2015_lowmass_side_study_20260906/note/lowmass_section.tex}
\endgroup
```

The section uses the existing `cowan` citation. Add this bibliography entry once:

```latex
\bibitem{lowmasshps} P.~H.~Adrian et al. (HPS),
\emph{Search for a Dark Photon in Electro-Produced $e^+e^-$ Pairs
with the Heavy Photon Search Experiment at JLab},
Phys. Rev. D \textbf{98} (2018) 091101.
\href{https://arxiv.org/abs/1807.11530}{arXiv:1807.11530}.
```

Retain the separation between local asymptotic values, small conditional toy tails, the parent stress-background global study, and physical signal validation. The wider-ceiling cross-check removes an optimizer boundary; it does not validate detector acceptance below the established search range. Neither the 17.25 MeV extraction nor the model-sensitive 21 MeV bridge supports a particle claim.
