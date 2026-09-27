"""Observed v16 morph extraction: readable standalone and integrated narrative."""
from pathlib import Path
import json
import pandas as pd

def build_observed_appendix(B,pic,table,prefix=''):
 B=Path(B);R=B/'results'
 def page(title,body):return '\\clearpage\\section*{'+title+'}\n'+body
 def fig(name,caption,height='4.5in'):
  return pic(prefix+name,r'\textbf{How to read the figure.} '+caption,height)
 doc=page('Appendix J. Extract a morphed signal from the observed 2021 sample',r'''
This section fits the actual 2021 10\% reconstructed-mass histogram. It compares the neighboring-template signal model with the original shifted Gaussian and with a shifted Gaussian using the same new fit window. The scan covers mass hypotheses from 60 to 240 MeV in 1 MeV steps. The earlier observed results remain available in the main body; this section is a new conditional extraction comparison.

\textbf{The signal model.} At a generated mass with a stored v16 TC sample, use its full selected signal-MC distribution. Between generated masses, linearly interpolate the fitted core center and width, align the two neighboring distributions, and mix their complete cumulative probabilities with mass-distance weights. Bin probabilities $p_i(m)$ come from differences at the analysis-bin edges. The full selected normalization, including outside-spectrum categories, is retained; the fit interval is never renormalized to one.

All available anchors from 60 to 240 MeV are used for this observed scan. This differs from the earlier validation tests, which deliberately removed the target mass. In particular, 60 MeV uses its actual simulated distribution, and 60--80 MeV interpolates between those two anchors. This extends the model through the acceptance transition. There is no independent 65 MeV sample or new closure validation in that interval, so the low-mass result is conditional on this interpolation.

\textbf{The background and likelihood.} The morph fit uses $u=(m_{\rm rec}-c)/\sigma_{\rm core}$ and excludes $[-4,3]$ from GP training. The same bins enter the signal likelihood. The sideband counts predict a background mean $b_i$ and correlated covariance $C=LL^{\mathsf T}$. Kernel parameters remain fixed at their archived values. In the fitted bins,
\[
 \lambda_i(A,\boldsymbol\theta)=b_i+(L\boldsymbol\theta)_i+A p_i,
 \qquad
 -\ln\mathcal L=\sum_i\bigl[\lambda_i-n_i\ln\lambda_i\bigr]
 +\tfrac12\boldsymbol\theta^{\mathsf T}\boldsymbol\theta+\mathrm{constant}.
\]
The nuisance parameters $\boldsymbol\theta$ allow correlated background shifts. The amplitude $A$ is allowed to be signed for diagnosing excesses and deficits, while expected bin counts remain positive. Upper limits impose the nonnegative signal hypothesis through the bounded profile-likelihood construction.

\textbf{What the extraction plots show.} The GP prediction is the background before fitting the excluded bins. The background-only profile fixes $A=0$ and adjusts the background nuisance parameters. The signal-plus-background fit adjusts both $A$ and those nuisance parameters; its profiled background is one component of the total fit. These backgrounds need not coincide. The lower panel subtracts the GP mean so that a small excess or deficit is visible on top of the large count spectrum. It is a residual plot, not a per-bin significance plot.

The side study showed that signal entering GP training can bias the yield response. Real data do not identify those signal events, so no truth-assisted subtraction is applied. The new scan therefore retains the existing sideband procedure and separately checks selected fit statistics with simulated backgrounds.
''')
 scan=pd.read_csv(R/'observed_scan.csv');cal=pd.read_csv(R/'selected_local_calibration.csv');limits=pd.read_csv(R/'selected_rank_limits.csv')
 regions=json.loads((R/'selected_regions.json').read_text())['regions']
 fsum=json.loads((R/'figure_summary.json').read_text())
 doc+=page('Appendix J. Compare the observed upper limits and local p-values',
  fig('v638_observed_scan',r'''The same observed 2021 10\% histogram is fitted under three signal assumptions. Blue uses the full neighboring signal-MC shape and $[-4,3]$ core-width fit/exclusion. Gray uses the historical shifted Gaussian and window; gold keeps that Gaussian but uses the MC window. Upper panel: bounded 90\% profile-CLs amplitude limits with asymptotic sampling tails. Their full-yield signal assumptions differ. Lower panel: fixed-mass asymptotic excess probabilities; the deficit has $q_0=0$ and the conventional reference $p_0=0.5$. Points mark the selected regions. Shading flags the 60--80 MeV interpolation extension. No template, kernel or selection uncertainty band is propagated, and no global probability is shown.''','4.8in')+r'''
A larger MC amplitude limit does not by itself imply poorer sensitivity. The Gaussian assigns almost all its signal probability to a narrow distribution; the full MC model assigns substantial probability to tails outside the fit. Near 67 MeV, the fit contains 52.5\% of the assumed full MC signal, so its amplitude counts many more events than appear in the peak. The next comparison uses the same injected signal distribution for every extraction method.

The selected positive regions are the three highest local maxima of the morph excess statistic whose actual fitted bins do not overlap: 67, 79 and 185 MeV. The deficit at 226 MeV has the most negative signed likelihood root among windows disjoint from those regions. These rules were recorded before the scan. Disjoint fitted bins do not make the regions statistically independent, since they share GP sidebands. Both strongest regions lie in the acceptance-transition interpolation range.
''')
 def one(d,m,p):return d[(d.mass_MeV==m)&(d.policy==p)].iloc[0]
 prow=[]
 for r in regions:
  m=int(r['mass_MeV']);a=one(cal,m,'gaussian_baseline');b=one(cal,m,'morph_starter');down=r['region']=='deficit'
  pa=a.p_deficit_asymptotic if down else a.p0_asymptotic;pb=b.p_deficit_asymptotic if down else b.p0_asymptotic
  prow.append([str(m),'Down' if down else 'Up',f'{pa:.4g}',f'{pb:.4g}',f'{a.p_rank:.4g}',f'{b.p_rank:.4g}'])
 doc+=page('Appendix J. Check the selected regions with background-only toys',r'''
The asymptotic curve converts the fitted likelihood ratio into a Gaussian-tail reference. Define the signed root and excess statistic by
\[
 r=\operatorname{sign}(\widehat A)\sqrt{2\,[\mathrm{NLL}_{A=0}-\mathrm{NLL}_{\widehat A}]},
 \qquad q_0=\max(r,0)^2,\qquad Z_{\rm local,asym}=\sqrt{q_0}.
\]
For a positive fitted signal, the local asymptotic probability is $1-\Phi(Z_{\rm local,asym})$. For a deficit, $q_0=0$; the separate downward-tail diagnostic uses the negative signed root. A deficit is not a negative discovery significance.

For each selected mass, fit 1,000 independent background-only spectra generated by Poisson fluctuations about the archived GP mean. The source is a fixed arithmetic mean spectrum, not a randomly drawn GP function. Recompute GP predictions and nuisance fits for every toy. The same toy counts are used across masses and methods. For an excess, count toy $q_0$ values at least as large as observed; for the deficit, count signed roots at least as negative. The finite-sample rank is $(1+k)/1001$.
'''+table(['MeV','Tail','Gaussian asym.','MC asym.','Gaussian toys','MC toys'],prow)+r'''
The table compares the historical Gaussian and the morphed template. The first three rows use upward excess tests; the last row uses downward deficit tests. The starter-window Gaussian is also saved in the numerical tables. These are fixed-mass checks after choosing interesting masses in the observed scan. They do not include the probability of finding a fluctuation anywhere in that scan and are not global search p-values.

For the morph fit at 67 MeV, only one of 1,000 background toys exceeds the observed statistic. The rank value is 0.0020, while the exact two-sided 95\% Clopper--Pearson interval for the underlying exceedance probability is approximately $[0.000025,0.00556]$. The finite toy sample therefore does not resolve a precise rare-tail probability. At 79 MeV, the toy rank is 0.0220, compared with the asymptotic reference 0.00592. That difference illustrates why the two probability labels are kept separate.

Count intervals on the extraction plots retain denominator 1,000. The rank value uses the add-one rule; the confidence interval describes the binomial exceedance fraction. Both condition on the archived source, fixed templates and kernel choices. No source-estimation or signal-template uncertainty is propagated.
''')
 lrows=[]
 for r in regions:
  m=int(r['mass_MeV']);vals=[]
  for p in ['gaussian_baseline','gaussian_starter','morph_starter']:
   q=one(limits,m,p)
   vals.append('Empty set' if bool(q['empty']) else f'{q.largest_accepted_A:,.0f}')
  q=one(limits,m,'morph_starter')
  nxt='--' if bool(q['empty']) else f'{q.next_grid_A:,.0f}'
  lrows.append([str(m)]+vals+[nxt])
 doc+=page('Appendix J. Compare limits on the same full signal yield',r'''
To compare the extraction procedures with a common signal definition, generate the full morphed signal at each selected mass and use it for every method's calibration. This includes signal in GP training bins and outside the measured spectrum. All amplitudes in this comparison refer to the same expected number of selected signal events, even when the fitted extraction function is Gaussian.

At each mass and expected yield, use 200 calibration experiments on the GP mean background source. The fixed grid is $A/s_0=0,0.5,1,2,3,4,5,6,8,10,12,16$. Here $s_0$ is the Gaussian yield error from fitting the deterministic GP-source background spectrum; it is not chosen from the observed excess. Backgrounds and injected counts are paired across extraction methods. Define
\[
 p_A=\frac{1+\#\{\widehat A_{{\rm cal},A}\le\widehat A_{\rm observed}\}}{201},
 \qquad \hbox{accept the grid value }A\hbox{ if }p_A>0.10.
\]
The complete accepted sets are saved. The table reports their largest accepted values in full selected events; these are discrete endpoints of a yield-test inversion, not the continuous profile-CLs limits on the scan figure.
'''+table(['MeV','Gaussian, old','Gaussian, new','Morphed MC','MC next rejected'],lrows)+r'''
At 67 and 79 MeV all three methods have the same largest accepted grid value. At 185 MeV the MC endpoint is one grid step higher. The next rejected value is shown to expose the grid resolution; it is not a confidence interval on the endpoint. No positive-region set has holes or reaches the largest tested yield. The comparison does not establish uniformly stronger limits from the morphed extraction.

\textbf{The deficit gives an empty set.} At 226 MeV, all tested nonnegative yields, including zero, are rejected by this one-sided fitted-yield ordering. There is no endpoint to report. This is the result of the specified local one-sided ordering, not a zero physical signal limit or a globally calibrated rejection of background. Such downward fluctuations can produce empty sets with this construction. The bounded profile-CLs calculation instead remains positive (about 2,921 full MC-like events), because it is a different procedure with different tail ordering.

These calibrations assume the interpolated signal distribution at the selected mass. They are not independent detector-MC validation at 67, 79, 185 or 226 MeV, and the fixed-mass rank construction does not provide simultaneous coverage after scanning. The v16 production and observed source also lack a complete selection-equivalence validation, so no new coupling exclusion is derived here.
''')
 commoncap=r'''Black points are observed counts per MeV with count-only $\sqrt{N}/\Delta m$ bars. Blue is the sideband GP mean with its marginal one-standard-deviation constraint band, not a post-fit confidence band. Gold profiles the background with signal fixed to zero. Green is the background component of the signed signal-plus-background fit; red is its total. The lower panel subtracts the GP mean and shows the signed signal separately in purple. Correlated background uncertainties enter the likelihood. All model curves are restricted to the fitted bins, which are excluded from GP training. Yield errors are conditional profile-Hessian standard deviations. The displayed probabilities are local; selection in the observed mass scan is not included.'''
 for index,r in enumerate(regions):
  m=int(r['mass_MeV']);q=one(scan,m,'morph_starter');t=one(cal,m,'morph_starter');isdef=r['region']=='deficit';name='v638_deficit' if isdef else f'v638_peak_{index+1}'
  heading='A deficit at 226 MeV' if isdef else f'Excess region {index+1}: a {m} MeV mass hypothesis'
  body=fig(name,commoncap+(' The negative signed fit describes a downward fluctuation; the nonnegative-signal best fit is the background-only curve.' if isdef else ' The annotated positive local significance is the asymptotic likelihood reference; the separate toy rank uses 1,000 GP-source backgrounds.'),'4.85in')
  body+=rf'''
The signal-mass hypothesis is {m} MeV, while its reconstructed core is predicted at {q.core_center_MeV:.3f} MeV with width {q.core_width_MeV:.3f} MeV. The actual fitted interval is {q.fit_low_MeV:.3f}--{q.fit_high_MeV:.3f} MeV. It contains {100*q.MC_fit_fraction:.1f}\% of the assumed full selected signal; {100*q.MC_training_fraction:.1f}\% enters GP training. Probability outside the measured spectrum is retained separately. These fractions describe the model, not an observed signal count.
'''
  if m<80:
   body+=r'''\textbf{Acceptance-transition qualification.} This template interpolates the actual 60 and 80 MeV signal-MC samples. Their core fractions and tails change rapidly. No independent sample at this mass establishes the accuracy of that interpolation; the fitted yield, limit and probability therefore depend on the stated signal model.'''
  elif isdef:
   body+=r'''Here the signed amplitude is negative, so the purple contribution lies below zero. Allowing that diagnostic amplitude improves the description of the downward fluctuation without implying negative physical signal production. The excess statistic is $q_0=0$; the conventional upward asymptotic reference is $p_0=0.5$. The separate downward toy rank, about 0.060, measures how often the fixed GP-source background produces a root at least this negative. The empty calibrated yield set is explained on the preceding limit page.'''
  else:
   body+=r'''This template interpolates the 180 and 200 MeV samples. The red residual combines the signal and the nuisance-driven background displacement; it is not the signal curve alone. A visible fitted component is evidence of the best fit under this model, not a globally calibrated signal claim.'''
  doc+=page('Appendix J. '+heading,body)
 doc+=page('Appendix J. Validation and reproducible inputs',r'''
\textbf{Numerical checks.} All 543 observed scan fits and 40,800 selected-region toy fits passed the numerical checks. Every one of the 181 historical Gaussian scan points reproduces the archived 2021 result to numerical precision. This agreement verifies the reference implementation; it does not independently validate its statistical assumptions. The independent statistics audit checks the selected regions, fitted components, interpolation probabilities, rank tails, accepted sets, binomial intervals and checkpoint identities.

The observed scan uses all available 60--240 MeV anchors and retains full normalization. At an anchor, the generated template reproduces its direct signal-MC histogram. Between anchors, the saved neighbor weights, core parameters and bin probabilities specify the exact model. The low-mass extension is explicitly separate from the earlier 80--240 MeV validation domain.

\textbf{Saved evidence.} The standalone study package contains the observed histogram and frozen numerical implementation, v16 signal-MC histograms and core fits, the full 181-mass template grid, all three observed scans, selected-region fit-component arrays, background and signal toy draws, complete fitted rows, frozen protocols and QA. The statistical review records the conventions and claim boundaries. No extra S3DF data access or UC inference is required.

Within the large v6.3.6 package, these files are in \path{appendix_observed_morph/}. The primary entry points are \path{scripts/run_observed.py}, \path{scripts/make_figures.py} and \path{scripts/build_report.py}; the package README gives the verified rebuild commands. The preceding 43-page release is preserved separately. Earlier main-body and appendix results remain intact.

\textbf{What changed scientifically.} This section supplies observed upper-limit and local-probability comparisons with the full neighboring signal shape, then shows the background and signal components that produce the fitted excesses and deficit. It does not establish a uniform limit improvement, exact interpolation through the acceptance transition, or a global discovery probability. The same-generator calibration makes the signal-yield comparison explicit and keeps the deficit's empty accepted set visible.

\textbf{Method references.} Core alignment and mixing are related to the template-morphing discussion of Baak et al., \href{https://arxiv.org/abs/1410.7388}{arXiv:1410.7388}. GP background modeling and the need for ensemble checks are discussed by Frate et al., \href{https://arxiv.org/abs/1709.05681}{arXiv:1709.05681}. The local asymptotic likelihood and bounded profile-limit references follow Cowan et al., \href{https://arxiv.org/abs/1007.1727}{arXiv:1007.1727}. These sources motivate the methods; they do not establish calibration for this selected data sample.
''')
 return doc
