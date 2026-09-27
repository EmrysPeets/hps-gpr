#!/usr/bin/env python3
"""Assemble v6.3.5 saved statistics into a standalone technical report; no fits."""
from pathlib import Path
import argparse,json,os,subprocess,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v635-report')
import numpy as np,pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures';S=B/'source';R=B/'results'
P=['pole','logshift'];C={'pole':'#a54b35','logshift':'#245c91'};NAME={'pole':'Pole center','logshift':'Log core center'}
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'lines.markersize':3,'lines.linewidth':1,'savefig.dpi':160,'pdf.fonttype':42})
def rd(n):return pd.read_csv(R/(n+'.csv'))
def sub(d,**kw):
 for k,v in kw.items():d=d[d[k].eq(v)]
 return d.sort_values('mass_MeV')
def style(ax,y,ref=None):
 ax.set(xlabel='Pole mass [MeV]',ylabel=y,xlim=(55,245),xticks=[60,100,140,180,220,240]);ax.grid(axis='y',alpha=.2)
 if ref is not None:ax.axhline(ref,color='.5',ls='--',lw=.7,zorder=-3)
def save(fig,n,legend=True):
 if legend:
  hs,ls=fig.axes[0].get_legend_handles_labels()
  fig.legend(hs,ls,loc='upper center',ncol=min(4,len(ls)),frameon=False,bbox_to_anchor=(.52,1))
 fig.tight_layout(rect=(0,0,1,.93 if legend else 1));fig.savefig(F/(n+'.pdf'));fig.savefig(F/(n+'.png'));plt.close(fig)
def curves(ax,d,y,e=None):
 for p in P:
  q=sub(d,policy=p);ax.errorbar(q.mass_MeV,q[y],yerr=q[e] if e else None,marker='o' if p=='logshift' else 's',color=C[p],label=NAME[p],capsize=1.5)
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def f(x,n=3):return f'{x:.{n}f}' if np.isfinite(x) else '--'
def table(headers,rows,fmt=None):
 fmt=fmt or 'l'+'r'*(len(headers)-1)
 return '\n'.join([r'\begin{center}\small\begin{tabular}{'+fmt+r'}\toprule',' & '.join(headers)+r'\\\midrule']+[' & '.join(map(str,row))+r'\\' for row in rows]+[r'\bottomrule\end{tabular}\end{center}'])
def pic(n,caption,height=''):
 opt='width=\\linewidth'+(',height='+height+',keepaspectratio' if height else '')
 return r'\begin{center}\includegraphics['+opt+']{../figures/'+n+r'.pdf}\end{center}'+ '\n'+r'{\small '+caption+'}\par\medskip\n'
def page(title,body):return r'\clearpage\section*{'+title+'}\n'+body

def main(build=False):
 F.mkdir(exist_ok=True);S.mkdir(exist_ok=True);(B/'pdf').mkdir(exist_ok=True)
 h=rd('heldout_summary');c=rd('calibration_summary');l=rd('limit_summary');o=rd('observed_display');op=rd('observed_pointwise');j=rd('combined_observed_rank_display');je=rd('combined_evaluation_summary');w=rd('window_toy_summary');wm=rd('window_metrics');wa=rd('window_candidate_asimov');summary=json.loads((R/'summary.json').read_text())
 own=h[h.source.eq(h.calibration_source)];mc=own[own['shape'].eq('mc')]
 # Native probability masses on the analysis grid, full normalization retained.
 t=np.load(B/'inputs/templates.npz');null=np.load(B/'inputs/null_2021.npz');edges=null['edges_GeV']*1000;x=(edges[1:]+edges[:-1])/2;dx=np.diff(edges)
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.9))
 for a,m in zip(ax.flat,[60,100,160,240]):
  i=list(t['masses_MeV']).index(m)
  a.step(x,t['mc_categories'][i,1:-1]/dx,where='mid',color='.2',label='Native MC generator')
  for p in P:
   sigma=1000*(.00184825-.001375*(m/1000)+.085875*(m/1000)**2);center=m if p=='pole' else m-3.2243308692909953-2.213992811446465*np.log(m/150)
   a.plot(x,np.diff(ndtr((edges-center)/sigma))/dx,color=C[p],label=NAME[p]+' Gaussian')
  a.set(xlim=(m-12,m+34),xlabel='Reconstructed mass [MeV]',ylabel='Full probability / MeV',title=f'Pole mass {m} MeV');a.grid(alpha=.15)
 save(fig,'shapes')
 # Calibration and independent null diagnostics.
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.35))
 for row,s in enumerate(['nominal','functional']):
  curves(ax[row,0],sub(c,source=s),'mu0','mu0_se');style(ax[row,0],r'Calibration $\mu_0$',0);ax[row,0].set_title(s+' source',loc='left')
  curves(ax[row,1],sub(mc,source=s,z=0),'centered_pull_mean','centered_pull_se');style(ax[row,1],'Heldout centered null pull',0)
 save(fig,'null')
 # Raw and paired response at each strength.
 for s in ['nominal','functional']:
  fig,ax=plt.subplots(2,3,figsize=(7.1,4.8))
  for col,z in enumerate([1,3,5]):
   for row,key,label in [(0,'raw_recovery','Raw recovery'),(1,'paired_response','Paired response')]:
    curves(ax[row,col],sub(mc,source=s,z=z),key+'_mean',key+'_se');style(ax[row,col],label if col==0 else '',1);ax[row,col].set_title(f'z = {z}',loc='left');ax[row,col].set_xticks([60,120,180,240])
  save(fig,'recovery_'+s)
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.4))
 for col,s in enumerate(['nominal','functional']):
  q=sub(mc,source=s,z=5)
  curves(ax[0,col],q,'centered_pull_mean','centered_pull_se');style(ax[0,col],'Mean centered pull' if col==0 else '',0);ax[0,col].set_title(s+', z = 5',loc='left')
  curves(ax[1,col],q,'pull_sd','pull_sd_bootstrap_se');style(ax[1,col],'Raw pull SD' if col==0 else '',1)
 save(fig,'pulls')
 fig,ax=plt.subplots(1,2,figsize=(7.1,2.9))
 for a,s in zip(ax,['nominal','functional']):
  q=sub(mc,source=s,z=5);curves(a,q,'affine_pull_approx_mean','affine_pull_approx_se');style(a,'Approx. affine pull mean',0);a.set_title(s+' source, z = 5',loc='left')
 save(fig,'affine')
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.6))
 methods=[('rank_neyman90','Raw-yield rank inversion','#245c91'),('rank_cls90','Rank CLs diagnostic','#797127'),('native_cls90','Native asymptotic CLs','#a54b35')]
 for col,s in enumerate(['nominal','functional']):
  for meth,label,color in methods:
   q=sub(l,source=s,policy='logshift',z=5,method=meth);q=q[q.calibration_source.isin([s,'none'])]
   a=ax[0,col];a.errorbar(q.mass_MeV,q.acceptance_fraction,yerr=np.array([q.acceptance_fraction-q.acceptance_cp95_lo,q.acceptance_cp95_hi-q.acceptance_fraction]),color=color,marker='o',label=label,capsize=1)
   q=sub(l,source=s,policy='logshift',z=0,method=meth);q=q[q.calibration_source.isin([s,'none'])];ax[1,col].plot(q.mass_MeV,q.median_U_over_s0,color=color,marker='o',label=label)
  style(ax[0,col],'z = 5 truth acceptance',.9);ax[0,col].set(ylim=(-.04,1.04),title=s+' source');style(ax[1,col],r'Null median grid $U/s_0$')
 save(fig,'limits')
 # Observed curves: proxy conversion only for display; no fitting or interpolation of MC tables.
 for scope in ['2021','combined']:
  fig,ax=plt.subplots(2,1,figsize=(7.1,5.1))
  for p in P:
   q=sub(o,scope=scope,policy=p);ax[0].semilogy(q.mass_MeV,q.epsilon2_90_visible_legacy,color=C[p],label=NAME[p]);ax[1].semilogy(q.mass_MeV,q.p0_asymptotic,color=C[p],label=NAME[p])
   q=sub(op,policy=p,calibration_source='nominal') if scope=='2021' else sub(j,policy=p)
   empty=q['rank_empty'] if scope=='2021' else q['empty'];u=q['rank_epsilon2_visible_legacy'] if scope=='2021' else q['epsilon2_90_grid_visible_legacy'];p0=q.p0_rank
   ok=(u>0)&~empty.astype(bool)
   ax[0].scatter(q.loc[ok,'mass_MeV'],u[ok],s=20,facecolors='white',edgecolors=C[p],zorder=3)
   ax[1].scatter(q.mass_MeV,p0,s=20,facecolors='white',edgecolors=C[p],zorder=3)
  for a in ax:
   a.set(xlim=(60,240),xlabel='Pole mass [MeV]');a.grid(alpha=.2)
   if scope=='combined':
    a.axvline(100,color='.6',lw=.7,ls=':');a.axvline(180,color='.6',lw=.7,ls=':')
  ax[0].set_ylabel(r'Legacy visible $\epsilon^2_{90}$ display');ax[1].set_ylabel(r'Pointwise $p_0$');ax[1].set_ylim(8e-5,1.2)
  save(fig,'observed_'+scope)
 fig,ax=plt.subplots(1,2,figsize=(7.1,3))
 for a,z in zip(ax,[0,5]):
  for p in P:
   q=sub(je,policy=p,z=z);a.errorbar(q.mass_MeV,q.truth_accepted_fraction,yerr=[q.truth_accepted_fraction-q.truth_accepted_cp95_low,q.truth_accepted_cp95_high-q.truth_accepted_fraction],color=C[p],marker='o',label=NAME[p],capsize=1)
  style(a,'Joint truth acceptance',.9);a.set(ylim=(.65,1.02),title=f'z = {z}')
 save(fig,'joint_validation')
 fig,ax=plt.subplots(1,2,figsize=(7.1,3.25))
 names={'tight':'Matched (2, 2)','wideleft':'Matched (2.5, 2)','wideright':'Matched (2, 2.5)','equal95':'Core fit + 95% guard'}
 for p,color in zip(names,['#245c91','#a54b35','#34835a','#795a98']):
  q=sub(w,policy=p);a=ax[1 if p=='equal95' else 0];a.errorbar(q.mass_MeV,q.empirical_precision_ratio,yerr=[q.empirical_precision_ratio-q.empirical_precision_ratio95_low,q.empirical_precision_ratio95_high-q.empirical_precision_ratio],label=names[p],color=color,marker='o',capsize=1)
 for a in ax:style(a,r'Ratio of null SD / response',1)
 handles=sum([a.get_legend_handles_labels()[0] for a in ax],[]);labels=sum([a.get_legend_handles_labels()[1] for a in ax],[]);fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.83));fig.savefig(F/'windows.pdf');fig.savefig(F/'windows.png');plt.close(fig)
 # Text assembly: fixed page boundaries, compact tables, complete definitions.
 doc=r'''\documentclass[10pt]{article}
\usepackage[margin=.76in]{geometry}\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype,fancyhdr,xurl,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}\pagestyle{fancy}\fancyhf{}\lhead{HPS GPR: unified 2021 study}\rhead{v6.3.5}\cfoot{\thepage}\setlength{\headheight}{13pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{2em}
\newcommand{\Ah}{\widehat A}\newcommand{\sh}{\widehat\sigma}
\begin{document}
{\Large\bfseries 2021 signal extraction with an MC-informed Gaussian center}\par
{\large Null calibration, native-MC response, observed inference, and window sensitivity}\par
24 September 2026\hfill Version 6.3.5

The logarithmic MC core shift improves the Gaussian extraction response substantially, but it does not make the fitted full-selected Gaussian yield an unbiased estimate of the full-selected MC yield. The primary study fits Gaussian templates to native MC injections. This is a different experiment from the earlier native-MC template closure, which generated and fitted the same MC template.

For nominal-source heldout injections at $A=3s_0$, the paired response is 0.208--0.422 with a pole-centered Gaussian and 0.265--0.833 with the logarithmically shifted Gaussian. Matched Gaussian controls give 0.963--0.980 with the shifted extraction. The remaining discrepancy therefore includes signal-shape projection as well as GP absorption. A fixed mean-pull subtraction centers a diagnostic; it does not restore missing MC yield.

Native asymptotic Gaussian-template upper limits do not have nominal containment for the MC full-yield truth. At $A=5s_0$, shifted-template containment is 4--72\% under the nominal source and 2--87\% under the functional source. Independent, source-matched rank inversion gives heldout truth-acceptance fractions of 81--96\% and 84--93\%, respectively. Those estimates each use 100 experiments, have finite calibration-table variation, and must not be summarized as universal 90\% coverage.

The observed dense scans are conditional asymptotic reference results. The smallest 2021 pointwise asymptotic $p_0$ moves from 0.00249 at 78 MeV to 0.00346 at 80 MeV. In the combined scan it moves from 0.00289 at 66 MeV to 0.000933 at 67 MeV. These minima were selected across mass and are not global probabilities. The independent 100-toy empirical calibration only resolves pointwise ranks in steps of $1/101$ at native anchors.

The window study does not establish a sensitivity gain from changing the matched $\pm2.25\sigma_{\rm nom}$ prescription. A guard retaining approximately 95\% of selected MC worsens empirical full-yield noise by factors 1.15--3.54. Smaller matched-window variations are mostly compatible with the baseline within paired uncertainty. No alternative is adopted.

\textbf{Scope.} Primary inference is restricted to 60--240 MeV and the supplied 2021 native 10\% data and templates. A 260 MeV shape control uses its directly fitted core, outside the logarithmic-law and search domain. There is no 40 MeV result. Fixed background sources, empirical MC shape, selection association, and archived kernels remain conditional assumptions; the results are not a physical exclusion or discovery calibration.
'''
 doc+=table(['Main component','Attempted','Valid'],[['Pilot free fits','2,000','2,000'],['Calibration free fits','52,000','52,000'],['Heldout free / truth-profile fits','22,000 / 22,000','22,000 / 22,000'],['Heldout native CLs endpoints','16,000','16,000']])
 doc+=page('1. Model, experimental units, and definitions',r'''
The two policies use either the pole mass $m$ or the pinned core law
\[
c(m)=m-3.2243308693-2.2139928114\log(m/150),\qquad 60\le m\le240\ {\rm MeV}.
\]
All quantities in this equation are in MeV. Only the center changes. The nominal resolution remains $\sigma_{\rm nom}=0.00184825-0.001375m+0.085875m^2$, with both $m$ and $\sigma_{\rm nom}$ in GeV. The likelihood window and the bins excluded from GP training share the same center and $\pm2.25\sigma_{\rm nom}$ boundaries. There is no extra guard in the primary study.

Each spectrum is conditioned anew using its own sideband log counts and count-dependent noise. Archived mass-specific kernel hyperparameters are fixed at the pole. The likelihood profiles correlated GP background nuisance parameters with a signed Gaussian signal yield; physical expected bin counts must remain positive. Returned yield errors come from the observed profile Hessian, not a Fisher replacement.

The signal yield $A$ means the expected number of full-selected events. Native histogram underflow, overflow, and events outside the analysis support remain in its probability normalization. Gaussian templates integrate the full Gaussian distribution. Neither template is renormalized to the fit window or support. Poisson signal counts enter training sidebands as well as the likelihood window. Thus $\Ah/A$ is a yield response, not a window acceptance or detector efficiency.

There are 100 independent nominal pilot backgrounds. Their shifted-Gaussian mean error freezes $s_0(m)$; the same $A=zs_0$ is then used for both policies and both sources. Each source has 100 new calibration backgrounds and 100 further heldout backgrounds. Calibration strengths are $z\in\{0,1,2,3,4,5,6,8,10,12,16,20,24\}$; evaluation strengths are $0,1,3,5$. Nominal-source Gaussian-log controls additionally use $1,3,5$ and reuse the MC-null row. These are 500 unique main background experiments, not 76,000 independent experiments. Backgrounds are paired across masses, strengths and policies; identical signal draws are shared across policies. Sources and cohorts are independent.

The frozen null and $z=3$ calibration rows define four different quantities:
\[
\mu_0=\overline{\Ah_0/\sh_0},\quad \delta=\overline{\Ah_0},\quad
R=\overline{(\Ah_3-\Ah_0)/(3s_0)},\quad
k_0={\rm SD}\{(\Ah_0-\delta)/\sh_0\}.
\]
The reported pull is $(\Ah-A)/\sh$. Subtracting $\mu_0$ is dimensionless mean-pull centering. Subtracting $\delta$ is a fixed yield-offset correction. Dividing by $R$ is a separate response correction. $k_0$ is a centered-yield fluctuation scale; raw null-pull SD is also retained. None of these changes the core location $c(m)$.

Means have sample-standard-error bars. Whole-toy bootstrap resampling preserves mass, strength and policy correlations (2,000 resamples); sources use separate streams. Widths have bootstrap uncertainty. Binomial counts retain denominator 100 and exact two-sided 95\% Clopper--Pearson intervals. Failures would remain in the ledger and all-attempt bounds; none occurred in the main run.
''')
 doc+=page('2. The MC shape and the Gaussian extraction',pic('shapes',r'Native full-selected MC probabilities and the two Gaussian fit templates, divided by the analysis-bin width. The plotted ranges emphasize the core and nearby tail; omitted plotted tails remain in every normalization and draw. No window-area matching is applied.')+r'''
The shift law describes the fitted MC core, not the full MC mean. The selected MC distribution has an asymmetric high-mass tail; moving a Gaussian toward the core cannot reproduce its full probability distribution. The nominal analysis resolution and the fitted MC core width are distinct quantities.

The known-background controls in the separate window study give Gaussian projection response 0.378--0.857 even when the background is known exactly. The baseline GP response divided by this control is 0.703--0.974. Shape projection and GP absorption are therefore separate effects. A wider training exclusion can address absorption at the cost of background information; it cannot by itself convert a Gaussian amplitude into full selected MC yield.

The supplied MC metadata identifies selected target-constrained mass from v13 and does not establish full v13/v16 selection equivalence. Finite template statistics, daughter association, selection equivalence and a globally qualified background generator are not certified here. The empirical histogram is held fixed across these ensembles.

Earlier v6.2 and v6.3.1 native-MC rows fitted the MC template. Their closure evidence remains useful for provenance and algorithm checks, but it does not answer the MC-generated/Gaussian-fitted question tested here. The v6.3.2 offset-transfer study concerned controlled 2016 exposure changes; its scaling results are not a 2021 calibration.
''')
 doc+=page('3. Freeze calibration, then test the independent null',pic('null',r'Left: frozen calibration mean null pull with mean SE. Right: independent null pull after subtracting that frozen mean, with evaluation mean SE. Calibration uncertainty is not added to the displayed conditional evaluation bars; its separate covariance is archived.')+r'''
Mean-pull centering removes the estimated null mean for the chosen source. It leaves the pull width unchanged, and a frozen estimate fluctuates between calibration cohorts. Under the shifted policy, heldout centered-null means range from -0.174 to 0.227 for the nominal source and -0.187 to 0.160 for the functional source. Their raw null-pull widths range from 0.832 to 1.131 and 0.893 to 1.121. These are pointwise finite-sample diagnostics, not evidence of exact unit width at every mass.

The functional source is an anchored archived functional stress that had local source assessment; it is not an independently globally qualified data-generating law. The full CSVs also apply nominal calibration to functional evaluation. This is a robustness test of transfer, and is distinct from source-matched calibration.
'''+table(['Mass','Nominal $\mu_0$','Functional $\mu_0$','Nominal $\delta/s_0$','Nominal $k_0$'],[[str(m),f(sub(c,source='nominal',policy='logshift',mass_MeV=m).mu0.iloc[0]),f(sub(c,source='functional',policy='logshift',mass_MeV=m).mu0.iloc[0]),f(sub(c,source='nominal',policy='logshift',mass_MeV=m).delta.iloc[0]/sub(c,source='nominal',policy='logshift',mass_MeV=m).s0.iloc[0]),f(sub(c,source='nominal',policy='logshift',mass_MeV=m).k0.iloc[0])] for m in [60,100,140,180,220,240]]))
 doc+=page('4. Raw recovery and paired response',pic('recovery_nominal',r'Nominal-source heldout MC injections. Top: $\overline{\Ah/A}$. Bottom: $\overline{(\Ah_z-\Ah_0)/A}$ using the same background toy. Bars are mean SE. Only the bottom row cancels an additive background offset exactly.')+r'''
The paired response separates incremental signal response from the null yield offset. A fixed subtraction $\Ah-\delta$ cancels in the paired difference and cannot repair response loss. Mean-pull centering corresponds instead to a count-dependent estimator $\Ah-\mu_0\sh$, whose paired response changes by
\[
-\mu_0\,\overline{(\sh_z-\sh_0)}/A.
\]
That change is archived separately and must not be mistaken for a signal-shape or efficiency correction. Calibration $R$ is frozen at $z=3$ and evaluated at independent strengths $1,3,5$.
'''+table(['Mass [MeV]','Pole MC response','Shifted MC response','Shifted Gaussian control'],[[str(m)]+[f(sub(own,source='nominal',policy=p,shape=sh,z=3,mass_MeV=m).paired_response_mean.iloc[0]) for p,sh in [('pole','mc'),('logshift','mc'),('logshift','gaussian_log')]] for m in range(60,241,20)]))
 doc+=page('5. Functional-source stress and signal-shape controls',pic('recovery_functional',r'Functional-source heldout MC injections. The same full-selected expected yields and paired background design are used as for the nominal source; the source ensembles themselves are independent.')+r'''
The shifted-policy paired response at $z=3$ ranges from 0.265 to 0.834 for the functional source, close to the nominal range but not proof of background-model robustness in data. Raw recovery differs because the null source bias differs. Numerical similarity of incremental response under these two fixed sources does not certify every source shape.

Gaussian controls are generated at the logarithmic center with the same nominal resolution and fitted by both policies. Their shifted-policy response of 0.963--0.980 at $z=3$ is much closer to unity than native-MC response, while still showing the effect of signal entering GP training bins. Gaussian-control signal draws are independent of MC draws; common backgrounds do not make their stochastic signals identical.

The archived paired policy-comparison table reports logshift minus pole differences and whole-toy bootstrap intervals. Policy comparisons share exactly the same injected counts, so their uncertainties retain this correlation. No MC response correction is applied to Gaussian controls: such a correction would answer a different, inappropriate calibration question.
''')
 doc+=page('6. Pulls and nominal profile containment',pic('pulls',r'Heldout full-selected MC truth at $z=5$. Subtracting the frozen null mean centers only the null contribution: it leaves the large signal-response bias. Width bars use whole-toy bootstrap SE. A pull centered about the wrong full-yield expectation can have width near one and a strongly displaced mean.')+r'''
The signed-yield truth profile evaluates $q_A=2\{\ell(\Ah,\widehat\theta)-\ell(A,\widehat\theta_A)\}$. Nominal 68\% and 95\% likelihood sets use $q_A\le1$ and $q_A\le3.841459$. These are Gaussian-template likelihood sets tested against the MC full-yield injection. They are not physical coverage statements or calibrated full-MC-yield intervals.
'''+table(['Mass','Nominal 68\%','Nominal 95\%','Functional 68\%','Functional 95\%'],[[str(m)]+[str(int(sub(mc,source=s,policy='logshift',z=5,mass_MeV=m)[k].iloc[0]))+'/100' for s in ['nominal','functional'] for k in ['contain68_k','contain95_k']] for m in range(60,241,20)])+r'''All counts and their Clopper--Pearson intervals are retained in \texttt{heldout\_summary.csv}. The null, positive-strength, source-transfer and Gaussian-control rows are distinguished explicitly; no favorable subset is substituted for the full planned ensemble.''')
 doc+=page('7. A diagnostic yield correction and its uncertainty',r'''
A conditional point estimator can be written $A_c=(\Ah-\delta)/R$. It uses the independently calibrated null offset and MC response rather than subtracting a dimensionless pull from a count. The bootstrap reestimates $\delta$ and $R$ together, preserving their covariance. An approximate variance is
\[
\operatorname{Var}(A_c)\simeq
\frac{k_0^2\sh^2+\operatorname{Var}(\delta)+A_c^2\operatorname{Var}(R)
+2A_c\operatorname{Cov}(\delta,R)}{R^2}.
\]
The positive covariance term follows from both derivatives with respect to the calibration parameters being negative. This delta-method estimate uses a frozen local response, a null width scale and independent evaluation/calibration cohorts. It omits pilot-scale, source-model and empirical-template uncertainties; nonlinearity and low-response amplification remain limitations. It is a point-estimator error diagnostic, not a replacement Poisson likelihood, confidence construction or calibrated CLs prescription.
'''+pic('affine',r'Mean approximate affine pull $(A_c-A)/\sigma_c$ on independent MC toys at $z=5$. Bars show SE across heldout toys conditional on the frozen calibration table. Calibration parameter uncertainty enters each diagnostic $\sigma_c$ through the displayed formula.')+table(['Mass','Nominal $R$','Bootstrap SE($R$)','SD($\delta$) [counts]','Corr($\delta,R$)'],[[str(m),f(q.R),f(q.R_bootstrap_variance**.5,4),f(q.delta_bootstrap_variance**.5,1),f(q.delta_R_bootstrap_covariance/(q.delta_bootstrap_variance*q.R_bootstrap_variance)**.5)] for m in [60,100,140,180,220,240] for _,q in sub(c,source='nominal',policy='logshift',mass_MeV=m).iterrows()])+r'''The primary finite-MC limits instead calibrate the raw statistic at every injected full yield. They therefore do not require treating this affine approximation as exact or silently dividing a production limit by $R$. Actual data corrections require qualified source and selection equivalence, independent calibration and propagated model uncertainty.''')
 doc+=page('8. Upper limits from a fixed construction',r'''
An upper limit describes signal strengths compatible with an observed statistic under a specified construction. At every grid strength $A$, define the lower-tail rank $p_A=(1+\#\{\Ah_A^{\rm cal}\le\Ah^{\rm eval}\})/101$. Reject when $p_A\le0.10$. Keep the entire accepted grid, its maximum, empty-set flag, internal holes and right-censor flag (largest node $24s_0$ accepted). There is no interpolation. An empty set receives stored endpoint zero by convention, not a zero continuous physical limit; a singleton $\{0\}$ only resolves behavior below the first positive node.

For continuous exchangeable calibration and evaluation statistics, this rank test has size at most $10/101$ marginal over both cohorts. A frozen table can have different acceptance: the continuous order-statistic reference gives a central 95\% interval of 83.60--95.10\% for its acceptance probability. This Beta-table uncertainty differs from the binomial interval of 100 heldout outcomes. Ties make the rank convention conservative.

The estimator-ordering toy-CLs diagnostic is $\min(1,p_A/p_0^{\rm lower})$. The shared $1/101$ floor can make it insensitive; it is not the inherited profile-asymptotic CLs calculation. Native CLs uses the profiled Gaussian model and signed likelihood-root asymptotics. None of these procedures is modified by pull subtraction.
'''+pic('limits',r'Shifted-policy, source-matched results. Upper panels show heldout $z=5$ truth acceptance and 95\% binomial intervals; lower panels show median null endpoints. Rank endpoints are discrete grid maxima. Native endpoints are continuous Gaussian-template references. Censoring and empty sets remain in machine-readable rows.',height='3.75in')+r'''All attempted IDs are retained. The upper-envelope coverage field at $A=0$ includes a stored empty-set endpoint of zero; actual accepted-set coverage uses the separate truth-acceptance field. A reported zero endpoint therefore must always be read with the empty and accepted-grid flags.''')
 doc+=page('9. Actual observed 2021 scan',pic('observed_2021',r'Lines: dense 1 MeV observed conditional asymptotic results. Open points: independent nominal-source raw-yield rank calibration at native masses only. Nonpositive grid endpoints are omitted from the logarithmic limit panel and flagged in the tables. The local empirical $p_0$ uses the upper null tail, not the lower tail used for exclusion.')+r'''
The observed count statistic is $\Ah=\widehat\psi\,C_{2021}(m)10^{-8}$, where $\psi=\epsilon^2/10^{-8}$ and the inherited pole-density conversion is held fixed across center policies. Raw $\epsilon^2$ columns are electron-channel proxies. For the legacy visible display only, multiply once above $2m_\mu=211.316749$ MeV by $1+\sqrt{1-4r}(1+2r)$, $r=(105.6583745/m)^2$. This convention is not independent validation of branching, efficiency, selection or a physical exclusion.

The empirical upper-tail value is $(1+k)/101$ with $k$ null-calibration exceedances. Its binomial Clopper--Pearson interval describes the underlying tail probability estimated by $k/100$; the add-one rank and that interval are distinct quantities. Neither the dense minimum nor the ten empirical anchors provide a global look-elsewhere correction.
'''+table(['Policy','Dense min mass','Asymptotic $p_0$','Native-anchor min mass','Rank $p_0$'],[[NAME[p],str(int(sub(o,scope='2021',policy=p).loc[sub(o,scope='2021',policy=p).p0_asymptotic.idxmin()].mass_MeV)),f(sub(o,scope='2021',policy=p).p0_asymptotic.min(),6),str(int(q.loc[q.p0_rank.idxmin()].mass_MeV)),f(q.p0_rank.min(),4)] for p in P for q in [sub(op,policy=p,calibration_source='nominal')]]))
 doc+=page('10. Combined observed scan: change only 2021',pic('observed_combined',r'Combined conditional observed curves and native-anchor nominal-source rank points. Combined empirical local $p_0$ ranks order the signed profile-likelihood root; standalone 2021 ranks order raw fitted yield. Both exclusion grids order the raw yield parameter. Campaign support changes at 100 and 180 MeV (dotted lines): all three campaigns contribute only at 60--100 MeV, then 2016+2021 through 180 MeV, and 2021 alone above 180 MeV. Empty and zero-only rank sets are omitted from the log endpoint panel.')+r'''
The likelihood uses a common coupling parameter with the inherited 2015 and 2016 signal models and conversions unchanged. Only the 2021 Gaussian center and its matched fit/exclusion window change. Thus this is a comparison of two specified combined models, not an independent recalibration of the old campaigns.

The separate joint pilot freezes a common coupling scale from 100 pole-policy pilot errors. Joint calibration uses 100 independent nominal GP background experiments and the same thirteen strength nodes; another 100 independent experiments evaluate $z=0,1,3,5$. The 2021 signal is native MC, while the old-year signals retain their inherited definitions. The full common-coupling experiment, including overlapping mass support, is calibrated directly at native anchors rather than combining independently corrected amplitudes.
'''+table(['Policy','Dense minimum mass','Signed local root','Asymptotic $p_0$'],[[NAME[p],str(int(q.mass_MeV)),f(q.signed_root),f(q.p0_asymptotic,6)] for p in P for _,q in sub(o,scope='combined',policy=p).nsmallest(1,'p0_asymptotic').iterrows()])+r'''These mass-selected maxima are descriptive. Their local asymptotic probabilities must not be converted to global significance, especially because the empirical 100-toy calibration has far coarser tail resolution.''')
 doc+=page('11. Joint finite-MC validation and observed anchors',pic('joint_validation',r'Independent joint heldout truth acceptance for the frozen nominal-source rank tables. Bars are 95\% Clopper--Pearson intervals from 100 experiments per cell. The same whole experiment is used for both center policies; the native-anchor cells do not constitute a global scan calibration.')+r'''
Across all joint cells, truth acceptance ranges from 78\% to 100\%; 260 of 8,000 evaluated sets are empty. Frozen-table variation, discrete strengths and finite heldout uncertainty prevent an assertion that every table has exactly 90\% coverage. Actual observed inversions contain five empty sets and five additional zero-only sets among twenty policy/mass pairs; none is right-censored or holey. Their stored zero endpoints are unresolved continuous limits, not exclusion of arbitrarily small positive signal.
'''+table(['Mass','Pole rank $p_0$','Shifted rank $p_0$','Pole grid set','Shifted grid set'],[[str(m)]+[f(sub(j,policy=p,mass_MeV=m).p0_rank.iloc[0],4) for p in P]+[('empty' if bool(q['empty']) else ('zero only' if q.upper_psi==0 else f'{q.epsilon2_90_grid_visible_legacy:.2e}')) for p in P for _,q in sub(j,policy=p,mass_MeV=m).iterrows()] for m in range(60,241,20)])+r'''Positive endpoint entries use the legacy visible $\epsilon^2$ display; per-node rank values, raw coupling values and source labels are in \texttt{combined\_observed\_rank.csv}. The smallest joint empirical rank is $4/101=0.03960$ at 80 MeV with the shifted policy; its three null signed-profile-root exceedances do not support the far smaller dense asymptotic minimum as an empirically calibrated tail probability.''')
 doc+=page('12. Window sensitivity: match the fit and exclusion deliberately',r'''
Define the dimensionless core coordinate $u=(m_{\rm rec}-c)/\sigma_{\rm core}$. The proposed wiggle interval is $u\in[-2L_{\rm wig},2R_{\rm wig}]$, hence $m_{\rm rec}=c+\sigma_{\rm core}u$. The variable $u$ is not a mass width. The separate executable candidates below are specified in nominal-resolution half-widths, so conversion between the two coordinates must retain $\sigma_{\rm core}/\sigma_{\rm nom}$ explicitly.

The matched candidates are $(L,R)=(2.25,2.25)$, $(2,2)$, $(2.5,2)$ and $(2,2.5)$ in units of $\sigma_{\rm nom}$, with likelihood and exclusion masks changed together. A fifth diagnostic keeps the baseline fit window and widens only the exclusion to contain the equal-tail 95\% MC interval. The finite-bin saved masks are authoritative. This design answers the sensitivity question rather than selecting an exclusion by containment alone.
'''+pic('windows',r'Independent side study: 100 pilot and 100 evaluation backgrounds, common full-selected $A=3s_0$, native MC injections, paired across five policies. Ratios compare ${\rm SD}(\Ah_0)/R$ to baseline; 95\% intervals use 2,000 whole-toy bootstrap resamples. Lower is better. These are conditional sensitivity proxies, not validated corrected limits.',height='3in')+r'''
The matched-window ratios are 0.983--1.014, mostly with intervals spanning one. The wide-left 100 MeV result is 0.983 [0.972, 0.997], one exploratory point among thirty comparisons. It does not establish a robust policy improvement. The wide 95\% guard gives empirical ratios 1.15--3.54 and returned-error/response ratios 1.23--4.52, despite reducing training leakage to about 5\%.

At 60 MeV the equal-tail exclusion extends approximately 51.0--212.25 MeV. Retaining at least three external bins on both sides is only a geometry check; it does not establish predictive adequacy across this gap. Known-background and GP Asimov controls separate Gaussian projection from absorption. Their response-scaled CLs endpoints are deterministic diagnostics, not coverage-calibrated limits. No guard or matched-window candidate is adopted.

The 260 MeV entry appears only in the shape-geometry tables and uses its directly fitted MC core. It is excluded from this sensitivity ensemble, the logarithmic-law fit domain and all observed/calibrated search plots.
''')
 doc+=page('13. Qualification, provenance, and reproduction',r'''
\textbf{What the results support.} A core-centered Gaussian materially increases response to the supplied native MC relative to a pole-centered Gaussian. Independent null centering, yield offset, response and width calibration answer separate questions. Calibrating raw statistics at fixed injected full yields gives an explicit conditional inference construction; it does not require ad hoc shifts of the observed likelihood or limits.

\textbf{What remains unresolved.} The background source, empirical MC template and archived kernel settings are fixed. Local historical assessment of a functional stress is not global source qualification. The retained historical 2016 support/optimizer qualification exception is not resolved by successful fixed-kernel toys or by leaving the old-year model unchanged in the combination. Native v13/v16 selection equivalence and daughter association remain unqualified. Finite MC template uncertainty, unknown source mismatch, model-selection uncertainty and global look-elsewhere calibration are outside this ensemble. A residual from one observed spectrum is not by itself an ensemble-bias estimate; it mixes counting fluctuations and possible mismatch.

\textbf{Numerical checks.} Every main planned free fit, truth profile and native endpoint passed the recorded finite/positive, score and Hessian gates. Counts, templates, masks, seeds and atomic checkpoints are retained. Independent validation checks seed replay, normalization, pairing, frozen calibration hashes and representative direct calculations. Statistical validation checks all calibration cells and rank/CP conventions. Numerical success is distinct from model or physical qualification.

\textbf{Files.} \texttt{protocol.json} records the frozen scientific design. \texttt{inputs/} contains pinned old inputs, full-normalization templates and cohorts; \texttt{provenance/} records their hashes and source notes. \texttt{pilot\_reference.json} freezes $s_0$, and \texttt{calibration\_freeze.json} records the calibration cohort before evaluation. \texttt{results/} contains raw rows, complete rank-node lists, summary statistics, observed scans, side-study masks and independent validations. \texttt{source/report.tex} and \texttt{scripts/make\_report.py} reproduce this document.

\textbf{Randomness.} Main master seed 63520260924 uses separate pilot, calibration, evaluation, signal and bootstrap namespaces. Each whole background ID is shared where pairing is specified. The report uses no additional signal or background generation. Source streams are independent. Independent observed/joint and window experiments retain their own archived seed definitions; they must not be silently treated as the same cohort.

\textbf{Reproduction.} Use the exact commands in \texttt{README.md}: resume the bounded local runners, run the analyzer, run the main/joint/window validators, and build this report with cached Tectonic resources. Scientific scripts use the verified Xcode Python environment. At most four workers across concurrent stages and one numerical thread per worker were used; no S3DF computation is part of this study. Archive hashes pin the completed data and sources. The build is read-only with respect to fits and input files.

\textbf{Prior evidence.} Pinned v6.1 core-law/native-MC inputs define the model provenance. v6.2 and v6.3.1 supply matched-template closure context, not the present MC-to-Gaussian closure. v6.3.2 supplies controlled 2016 offset-transfer context, not a universal luminosity or 2021 correction. This standalone report replaces the need to infer the current scope from those separate notes.

\textbf{Statistical references.} The bias, Neyman construction and low-sensitivity CLs terminology follows the PDG review, Sections 40.2 and 40.4: \url{https://pdg.lbl.gov/2026/reviews/rpp2026-rev-statistics.pdf}. The conditional asymptotic profile-likelihood reference is Cowan et al., \url{https://arxiv.org/abs/1007.1727}. The finite-rank guarantee and Beta order-statistic calculation used here are stated explicitly in Section 8; neither is a claim of global calibration or physical coverage under an unknown background law.
''')
 doc+='\n'+r'\end{document}'+'\n';(S/'report.tex').write_text(doc)
 meta={'figures':sorted(p.name for p in F.glob('*.pdf')),'report_source_sha256':hashlib.sha256((S/'report.tex').read_bytes()).hexdigest(),'source_summary_sha256':hashlib.sha256((R/'summary.json').read_bytes()).hexdigest()};(S/'report_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 if build:subprocess.run(['/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(S/'report.tex')],check=True,cwd=S)
 if build:(B/'pdf/report.pdf').replace(B/'pdf/HPS_GPR_v6p3p5_Unified_2021_Procedure.pdf')
 print('Wrote '+str(S/'report.tex'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');a=p.parse_args();main(a.build)
