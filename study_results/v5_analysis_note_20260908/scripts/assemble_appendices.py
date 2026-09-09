from pathlib import Path
import shutil,re,json
B=Path(__file__).resolve().parents[1]; R=B.parents[1]; S=B/'source'; P=S/'sections'
# Curated full traditional comparison with self-contained figures.
parent=R/'study_results/v4p9p16_candidate_removal_20260906'
t=(parent/'note/candidate_removal_section.tex').read_text()
t=t.replace(r'\section{Candidate removal and traditional signal searches}',r'\section{Candidate removal and traditional background fits}')
t=t.replace(r'\subsection*',r'\subsection')
t=t.replace(r'\cite{lowmasshps,hps2016traditional}',r'\cite{HPS2015DarkPhoton,HPS2016PromptLong}')
t=t.replace(r'Figure~\ref{fig:main}',r'Figure~\ref{fig:v5-union-bands}')
t=t.replace('signed root','signed local significance').replace('signed profile root','signed local significance')
for rel, dest in [('figures','removal_figs'),('traditional/figures','traditional_figs')]:
 d=S/dest;d.mkdir(exist_ok=True)
 for f in (parent/rel).glob('*.pdf'):shutil.copy2(f,d/f.name)
 t=t.replace('../'+rel+'/',dest+'/')
t=t.replace('The new manifest binds these products and this PDF.','The frozen source manifest binds the original numerical products; the v5 manifest\nseparately records the copied figures and revised text.')
(P/'v5_traditional_appendix.tex').write_text(t)
# Joint deficit scan is an appendix; extraction pictures are in the main results.
parent=R/'study_results/v4p9p16_candidate_removal_20260906/note'
t=(parent/'deficit_section.tex').read_text().replace(r'\section{Illustrative scan of deficits}',r'\section{Joint scan of deficits}\label{sec:v5-deficit}')
t=t.replace('../../v4p9p16_deficit_extension_20260906/figures/','deficit_figs/').replace('signed roots','signed local significance')
(S/'deficit_figs').mkdir(exist_ok=True)
shutil.copy2(R/'study_results/v4p9p16_deficit_extension_20260906/figures/combined_deficit_scan.pdf',S/'deficit_figs/combined_deficit_scan.pdf')
(P/'v5_deficit_appendix.tex').write_text(t)
# Low-mass side study is last, preserving full local-only and detector limits.
parent=R/'study_results/v4p9p16_2015_lowmass_side_study_20260906'
t=(parent/'note/lowmass_section.tex').read_text()
t=t.replace(r'\providecommand{\lowmassfigurepath}{../figures}',r'\providecommand{\lowmassfigurepath}{lowmass_figs}')
t=t.replace(r'\subsection*',r'\subsection').replace(r'\cite{lowmasshps}',r'\cite{HPS2015DarkPhoton}').replace(r'\cite{cowan}',r'\cite{Cowan2011}')
t=t.replace('Signed roots and local asymptotic','Signed local significance and local asymptotic')
(S/'lowmass_figs').mkdir(exist_ok=True)
for f in (parent/'figures').glob('*.pdf'):shutil.copy2(f,S/'lowmass_figs'/f.name)
(P/'v5_lowmass_appendix.tex').write_text(t)
# Bibliography supplements, verified against the primary arXiv metadata.
bib=S/'hps_gpr_analysis_note.bib'; bt=bib.read_text()
if 'AnanievRead2023' not in bt:
 bt+='''\n@article{AnanievRead2023, author={Ananiev, V. and Read, A. L.}, title={Gaussian Process-based calculation of look-elsewhere trials factor}, journal={JINST}, volume={18}, year={2023}, pages={P05041}, doi={10.1088/1748-0221/18/05/P05041}, eprint={2206.12328}, archivePrefix={arXiv}, url={https://arxiv.org/abs/2206.12328v3}}\n'''
if 'Berns2024' not in bt:
 bt+='''\n@article{Berns2024, author={Berns, L.}, title={An importance sampling method for Feldman-Cousins confidence intervals}, journal={Phys. Rev. D}, volume={109}, year={2024}, pages={092002}, eprint={2303.11290}, archivePrefix={arXiv}, url={https://arxiv.org/abs/2303.11290}}\n'''
bib.write_text(bt)
print('Traditional, deficit and low-mass appendices assembled')
