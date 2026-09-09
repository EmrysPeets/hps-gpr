#!/usr/bin/env python3
"""Map every adopted revision request to labels in the compiled note."""
from pathlib import Path
import json,re
B=Path(__file__).resolve().parents[1]
rows=[
('Combine v5 Figures 57–60','Single 2x2 correlation figure: 2015/2016 above 2021/combined.',['fig:v501-correlations']),
('Restore v4.9.7 Figures 8–12','All 20 original vector panels and five main captions restored in Section 3.',['fig:presel_vertex','fig:presel_ele_kin','fig:presel_pos_kin','fig:scatter_mom','fig:scatter_chi2']),
('Add kinematic summary subsection','New summary explicitly references the five groups and states the display normalization.',['sec:preselection-kinematics']),
('Clean v5 Figure 21','Overlapping schematic text removed or moved into legends; no HPS fit rerun.',['fig:gpr-mass-hypothesis-optimization']),
('Show allowed length-scale regions in GP space','Actual frozen bound implementation evaluated in log mass for every dataset.',['fig:v501-lengthscale-regions']),
('Formal profiled-background subsection and diagram','Poisson objective, nuisance profile, gradient/Hessian, positivity and conditioning explained.',['sec:v501-profile-implementation','fig:v501-profile-flow']),
('Yield-to-coupling flow at old Section 4.8 start','Four-block flow placed immediately after the signal-model subsection heading.',['sec:v501-signal-conversion','fig:v501-yield-conversion']),
('Reference global method after old Section 4.9','Local inference closes with a cross-reference to the conditional full-field method.',['sec:v5-global']),
('Explain Wald branch','Quadratic-profile approximation, normal estimator, boundary branch and zero-statistic convention explained.',['eq:v501-wald','eq:cls-asymptotic']),
('Clean common-coupling equations and add physical flow','Aligned equations, campaign-specific response blocks, and one common likelihood denominator.',['sec:combination','fig:v501-common-coupling','eq:v501-common-bounded-statistic']),
('Clean global-method equations and illustrate computational gain','Separate response/covariance/correlation/ordering equations and original schematic.',['sec:v5-global','fig:v501-global-gp-flow']),
('Rename Section 5','Background model validation with pseudo-experiments.',['sec:toys']),
('Move old Figure 29 to Section 5 start','Validation workflow precedes the first validation subsection.',['fig:v491-hps-gpr-flowchart']),
('Four exposure bullets and full-statistics role','Bullets distinguish native/source-scaled ensembles; full-equivalent studies inform model qualification before unblinding.',['sec:toys','sec:v491-four-lane-ensemble']),
('Optimize blind window and restore broader diagnostics','Historical yield, significance, pull-mean, pull-width and geometry displays retain source-study limits.',['subsec:v491-guard-band-construction','fig:v501-blind-optimization-history','fig:v501-blind-width-history','fig:v501-blind-geometry-history']),
('Cross-reference extraction implications of fixed-hyperparameter correlations','Validation text links to the result section on correlated fluctuations and echoes.',['sec:v5-echo']),
('Restore study/evidence/decision table','Edited table starts the validation decision subsection.',['sec:v501-validation-decisions','tab:v501-validation-decisions']),
('Reference and summarize all echo-section figures','Body references injection replay, combined correlations, conditional response and observed replacements, with numerical summaries.',['sec:v5-echo','fig:v5-signal-echo','fig:v501-correlations','fig:v5-conditional-echo','fig:v5-removal']),
('State first and second replaced regions in old Figure 62','Caption lists centers and actual whole-bin edges in each year.',['fig:v5-removal']),
('Restore controlled 2016 upper-range study','Original figure and factor-12 boundary/plateau decision, with post-selection qualification.',['sec:v501-2016-upper-range','fig:v501-2016-upper-range']),
('Current and projected result versus BaBar','Historical density prescription applied to current saved limits, scaling only 2021 by ten; explicitly an observed-equivalent proxy.',['sec:v501-babar-proxy','fig:v501-babar-proxy']),
('Three-panel scan for each dataset','Separate yield-limit, coupling-limit and nominal local-p0 plots for all three datasets.',['fig:v501-scan-2015','fig:v501-scan-2016','fig:v501-scan-2021']),
('Restore three toy regimes, exact forms, full-100 source/exposure and 65 MeV threshold studies','Three regimes distinguished; exact source intensities in main text; historical full-100 evidence and threshold modeling restored in appendix.',['sec:v501-toy-regimes','eq:v501-shifted-seed','eq:v501-unshifted-seed','eq:v501-gengamma-seed','sec:v501-v4p6-exposure-refmatched-full100','sec:v501-threshold-history']),
('Restore historical 2015 comparison and saved 2016 95% result with methods','Published versus corrected 2015 distinction; historical 2016 v4.1 comparison at 95% confidence, full exposure, and explicit methods.',['app:v501-published-comparisons','fig:v501-2015-method-comparison','sec:v501-2016-published-method','fig:v501-2016-95cl-comparison'])
]
aux=(B/'qa/build/main.aux').read_text();known={m[0]:(m[1],m[2])for m in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}',aux)}
items=[];lines=['# Implemented v5.0.1 requests','', 'The user adopted the attached revision brief. Old figure/section numbers below refer to v5.0.0 or v4.9.7 as specified. The locations refer to the final v5.0.1 PDF. No HPS observations, fitted values, or inference policies were changed.','','| # | Request | Implementation | Final PDF locations |','|---|---|---|---|']
for i,(request,implementation,labels) in enumerate(rows,1):
 locations=[{'label':x,'number':known[x][0],'page':int(known[x][1])}for x in labels]
 items.append(dict(id=i,request=request,implementation=implementation,labels=labels,locations=locations,status='implemented'))
 loc='; '.join(f'{x["number"]} (p. {x["page"]})'for x in locations)
 lines.append(f'| {i} | {request} | {implementation} | {loc} |')
(B/'editorial/request_checklist.json').write_text(json.dumps({'version':'5.0.1','brief':'provenance/analysis_not_501_instructions.pdf','items':items},indent=2)+'\n')
(B/'editorial/REQUEST_CHECKLIST.md').write_text('\n'.join(lines)+'\n')
print('Mapped',len(items),'requests to',sum(len(x['labels'])for x in items),'compiled targets.')
