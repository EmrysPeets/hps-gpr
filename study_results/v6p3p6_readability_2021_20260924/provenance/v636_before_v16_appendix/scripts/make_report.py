#!/usr/bin/env python3
"""Assemble v6.3.6 saved statistics into a standalone technical report; no fits."""
from pathlib import Path
import argparse,json,os,subprocess,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v635-report')
import numpy as np,pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures';S=B/'source';R=B/'results'
SOURCE={'nominal':'GP mean background source','functional':'Functional form background source'}
P=['pole','logshift'];C={'pole':'#a54b35','logshift':'#245c91'};NAME={'pole':'Central mass Gaussian','logshift':'Shifted Gaussian'}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'lines.markersize':3,'lines.linewidth':1,'savefig.dpi':160,'pdf.fonttype':42})
def rd(n):return pd.read_csv(R/(n+'.csv'))
def sub(d,**kw):
 for k,v in kw.items():d=d[d[k].eq(v)]
 return d.sort_values('mass_MeV')
def style(ax,y,ref=None):
 ax.set(xlabel='Generated mass [MeV]',ylabel=y,xlim=(55,245),xticks=[60,100,140,180,220,240]);ax.grid(axis='y',alpha=.2)
 if ref is not None:ax.axhline(ref,color='.5',ls='--',lw=.7,zorder=-3)
ERROR_NOTE=("Means: sample standard error (sample SD / sqrt(100)). Widths: bootstrap standard error.\n"
 "Bootstrap: 2,000 whole-toy resamples keep masses, strengths and Gaussian choices together; sources use separate streams.\n"
 "Binomial fractions: all 100 trials retained; exact two-sided 95% Clopper-Pearson intervals.\n"
 "Mean bars after calibration condition on the frozen calibration table; its uncertainty is assessed separately.")
FIG_INFO={
 'null': '2021 10% | Background-only calibration and independent evaluation | 100 toys per source and cohort',
 'recovery_nominal':'2021 10% | Signal-MC injections on the GP mean background source | 100 independent evaluation toys',
 'recovery_functional':'2021 10% | Signal-MC injections on the functional form background source | 100 independent evaluation toys',
 'pulls':'2021 10% | Signal-MC injections, A = 5 s0 | Pull compares fitted Gaussian yield with full selected signal yield',
 'affine':'2021 10% | Signal-MC injections, A = 5 s0 | Ac = (fitted yield - background-only yield bias) / response',
 'limits':'2021 10% | Shifted Gaussian | Source-matched calibration | Upper bars: 95% binomial intervals',
 'joint_validation':'2015 + 2016 + 2021 | Independent evaluation of toy-calibrated yield tests | 100 experiments per point',
 'shapes':'2021 signal MC and extraction templates | Full selected normalization; tails outside the frame are retained',
 'observed_2021':'2021 10% observed data | Lines: Gaussian asymptotic reference; circles: GP-source toy calibration at sampled masses',
 'observed_combined':'Combined observed data | Lines: Gaussian asymptotic reference; circles: joint toy calibration at sampled masses'
}
def finish(fig,n,legend=True):
 error=n in ('null','recovery_nominal','recovery_functional','pulls','affine','limits','joint_validation')
 fig.set_size_inches(fig.get_figwidth(),fig.get_figheight()+(.95 if error else .25))
 if legend:
  hs,ls=fig.axes[0].get_legend_handles_labels()
  fig.legend(hs,ls,loc='upper center',ncol=min(3,len(ls)),frameon=False,bbox_to_anchor=(.52,.985))
 fig.suptitle(FIG_INFO.get(n,''),fontsize=7.5,y=1.0)
 bottom=.18 if error else .025
 fig.tight_layout(rect=(0,bottom,1,.92 if legend else .96))
 if error:fig.text(.045,.014,ERROR_NOTE,fontsize=7,va='bottom',linespacing=1.4)
 fig.savefig(F/(n+'.pdf'));fig.savefig(F/(n+'.png'));plt.close(fig)
def save(fig,n,legend=True):finish(fig,n,legend)
def curves(ax,d,y,e=None):
 for p in P:
  q=sub(d,policy=p);ax.errorbar(q.mass_MeV,q[y],yerr=q[e] if e else None,marker='o' if p=='logshift' else 's',color=C[p],label=NAME[p],capsize=1.5)
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def f(x,n=3):return f'{x:.{n}f}' if np.isfinite(x) else '--'
def table(headers,rows,fmt=None):
 fmt=fmt or 'l'+'r'*(len(headers)-1)
 return '\n'.join([r'\begin{center}\small\begin{tabular}{'+fmt+r'}\toprule',' & '.join(headers)+r'\\\midrule']+[' & '.join(map(str,row))+r'\\' for row in rows]+[r'\bottomrule\end{tabular}\end{center}'])
def pic(n,caption,height=''):
 pic.number=getattr(pic,'number',0)+1
 caption=r'\textbf{Figure '+str(pic.number)+'.} '+caption
 opt='width=\\linewidth'+(',height='+height+',keepaspectratio' if height else '')
 return r'\begin{center}\includegraphics['+opt+']{../figures/'+n+r'.pdf}\end{center}'+ '\n'+r'{\small '+caption+'}\par\medskip\n'
def page(title,body):return r'\clearpage\section*{'+title+'}\n'+body

def main(build=False):
 pic.number=0
 F.mkdir(exist_ok=True);S.mkdir(exist_ok=True);(B/'pdf').mkdir(exist_ok=True)
 h=rd('heldout_summary');c=rd('calibration_summary');l=rd('limit_summary');o=rd('observed_display');op=rd('observed_pointwise');j=rd('combined_observed_rank_display');je=rd('combined_evaluation_summary');w=rd('window_toy_summary');wm=rd('window_metrics');wa=rd('window_candidate_asimov');summary=json.loads((R/'summary.json').read_text())
 h['calibrated_response_mean']=h.affine_yield_mean/h.A_expected.replace(0,np.nan)
 h['calibrated_response_se']=h.affine_yield_se/h.A_expected.replace(0,np.nan)
 own=h[h.source.eq(h.calibration_source)];mc=own[own['shape'].eq('mc')]
 # Native probability masses on the analysis grid, full normalization retained.
 t=np.load(B/'inputs/templates.npz');null=np.load(B/'inputs/null_2021.npz');edges=null['edges_GeV']*1000;x=(edges[1:]+edges[:-1])/2;dx=np.diff(edges)
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.9))
 for a,m in zip(ax.flat,[60,100,160,240]):
  i=list(t['masses_MeV']).index(m)
  a.step(x,t['mc_categories'][i,1:-1]/dx,where='mid',color='.2',label='Signal MC distribution')
  for p in P:
   sigma=1000*(.00184825-.001375*(m/1000)+.085875*(m/1000)**2);center=m if p=='pole' else m-3.2243308692909953-2.213992811446465*np.log(m/150)
   a.plot(x,np.diff(ndtr((edges-center)/sigma))/dx,color=C[p],label=NAME[p])
  a.set(xlim=(m-12,m+34),xlabel='Reconstructed mass [MeV]',ylabel='Full probability / MeV',title=f'Generated signal mass {m} MeV');a.grid(alpha=.15)
 save(fig,'shapes')
 # Calibration and independent null diagnostics.
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.35))
 for row,s in enumerate(['nominal','functional']):
  curves(ax[row,0],sub(c,source=s),'mu0','mu0_se');style(ax[row,0],'Background-only\nmean pull $\mu_0$',0);ax[row,0].set_title(SOURCE[s],loc='left')
  curves(ax[row,1],sub(mc,source=s,z=0),'centered_pull_mean','centered_pull_se');style(ax[row,1],'Background-only mean\npull after centering',0)
 for a in ax.flat:a.set_xlabel('Tested signal mass [MeV]')
 save(fig,'null')
 # Raw and paired response at each strength.
 for s in ['nominal','functional']:
  fig,ax=plt.subplots(3,3,figsize=(7.1,6.3))
  for col,z in enumerate([1,3,5]):
   for row,key,label in [(0,'raw_recovery',r'Fitted / injected: $\hat A/A$'),(1,'paired_response',r'Signal-induced change / $A$'),(2,'calibrated_response',r'Calibrated: $(\hat A-\delta)/(RA)$')]:
    curves(ax[row,col],sub(mc,source=s,z=z),key+'_mean',key+'_se');style(ax[row,col],label if col==0 else '',1);ax[row,col].set_title(f'z = {z}',loc='left');ax[row,col].set_xticks([60,120,180,240])
  save(fig,'recovery_'+s)
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.4))
 for col,s in enumerate(['nominal','functional']):
  q=sub(mc,source=s,z=5)
  curves(ax[0,col],q,'centered_pull_mean','centered_pull_se');style(ax[0,col],r'Mean $[(\hat A-A)/\hat\sigma]-\mu_0$' if col==0 else '',0);ax[0,col].set_title(SOURCE[s]+'; z = 5',loc='left')
  curves(ax[1,col],q,'pull_sd','pull_sd_bootstrap_se');style(ax[1,col],r'SD of $(\hat A-A)/\hat\sigma$' if col==0 else '',1)
 save(fig,'pulls')
 fig,ax=plt.subplots(1,2,figsize=(7.1,2.9))
 for a,s in zip(ax,['nominal','functional']):
  q=sub(mc,source=s,z=5);curves(a,q,'affine_pull_approx_mean','affine_pull_approx_se');style(a,r'Mean calibrated pull $(A_c-A)/\sigma_c$',0);a.set_title(SOURCE[s]+'; z = 5',loc='left')
 save(fig,'affine')
 fig,ax=plt.subplots(2,2,figsize=(7.1,4.6))
 methods=[('rank_neyman90','Toy-calibrated yield test','#245c91'),('rank_cls90','Toy-rank CLs diagnostic','#797127'),('native_cls90','Gaussian asymptotic CLs','#a54b35')]
 for col,s in enumerate(['nominal','functional']):
  for meth,label,color in methods:
   q=sub(l,source=s,policy='logshift',z=5,method=meth);q=q[q.calibration_source.isin([s,'none'])]
   a=ax[0,col];a.errorbar(q.mass_MeV,q.acceptance_fraction,yerr=np.array([q.acceptance_fraction-q.acceptance_cp95_lo,q.acceptance_cp95_hi-q.acceptance_fraction]),color=color,marker='o',label=label,capsize=1)
   q=sub(l,source=s,policy='logshift',z=0,method=meth);q=q[q.calibration_source.isin([s,'none'])];ax[1,col].plot(q.mass_MeV,q.median_U_over_s0,color=color,marker='o',label=label)
  style(ax[0,col],'Injected yield accepted',.9);ax[0,col].set(ylim=(-.04,1.04),title=SOURCE[s]);style(ax[1,col],'Background-only\nmedian $U/s_0$')
 for a in ax[1]:a.set_xlabel('Tested signal mass [MeV]')
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
   a.set(xlim=(58,242),xlabel='Tested signal mass [MeV]');a.grid(alpha=.2)
   if scope=='combined':
    a.axvline(100,color='.6',lw=.7,ls=':');a.axvline(180,color='.6',lw=.7,ls=':')
  ax[0].set_ylabel(r'Yield-to-coupling display: $\epsilon^2_{90}$');ax[1].set_ylabel(r'Local excess probability $p_0$');ax[1].set_ylim(8e-5,1.2)
  save(fig,'observed_'+scope)
 fig,ax=plt.subplots(1,2,figsize=(7.1,3))
 for a,z in zip(ax,[0,5]):
  for p in P:
   q=sub(je,policy=p,z=z);a.errorbar(q.mass_MeV,q.truth_accepted_fraction,yerr=[q.truth_accepted_fraction-q.truth_accepted_cp95_low,q.truth_accepted_cp95_high-q.truth_accepted_fraction],color=C[p],marker='o',label=NAME[p],capsize=1)
  style(a,'Injected yield accepted',.9);a.set(ylim=(.65,1.02),title='Background only (z = 0)' if z==0 else 'Signal + background (z = 5)')
 ax[0].set_xlabel('Tested signal mass [MeV]')
 save(fig,'joint_validation')
 fig,ax=plt.subplots(1,2,figsize=(7.1,3.25))
 names={'tight':'Matched (2, 2)','wideleft':'Matched (2.5, 2)','wideright':'Matched (2, 2.5)','equal95':'Core fit + 95% guard'}
 for p,color in zip(names,['#245c91','#a54b35','#34835a','#795a98']):
  q=sub(w,policy=p);a=ax[1 if p=='equal95' else 0];a.errorbar(q.mass_MeV,q.empirical_precision_ratio,yerr=[q.empirical_precision_ratio-q.empirical_precision_ratio95_low,q.empirical_precision_ratio95_high-q.empirical_precision_ratio],label=names[p],color=color,marker='o',capsize=1)
 for a in ax:style(a,'Relative background-only yield noise\nafter response scaling',1)
 handles=sum([a.get_legend_handles_labels()[0] for a in ax],[]);labels=sum([a.get_legend_handles_labels()[1] for a in ax],[]);fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False);fig.set_size_inches(7.1,4.15);fig.tight_layout(rect=(0,.23,1,.83));fig.text(.045,.015,'2021 10% | Signal MC at A = 3 s0 on the GP mean background source | 100 evaluation toys\nBars: 95% percentile intervals from 2,000 whole-toy bootstrap resamples.\nResampling keeps masses, signal strengths and window choices together; pilot scale is fixed.\nRatio = [SD(background-only fitted yield) / signal response] / baseline value. Lower is better.',fontsize=7,linespacing=1.4);fig.savefig(F/'windows.pdf');fig.savefig(F/'windows.png');plt.close(fig)
 from build_narrative import build_document
 doc=build_document(B,pic,table,sub,f,NAME,h,c,l,o,op,j,je,w,wm,wa,summary)
 doc+='\n'+r'\end{document}'+'\n';(S/'report.tex').write_text(doc)
 meta={'figures':sorted(p.name for p in F.glob('*.pdf')),'report_source_sha256':hashlib.sha256((S/'report.tex').read_bytes()).hexdigest(),'source_summary_sha256':hashlib.sha256((R/'summary.json').read_bytes()).hexdigest()};(S/'report_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 if build:subprocess.run(['/opt/homebrew/bin/tectonic','--only-cached','--keep-logs','--outdir',str(B/'pdf'),str(S/'report.tex')],check=True,cwd=S)
 if build:(B/'pdf/report.pdf').replace(B/'pdf/HPS_GPR_v6p3p6_2021_Signal_Extraction.pdf')
 print('Wrote '+str(S/'report.tex'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');a=p.parse_args();main(a.build)
