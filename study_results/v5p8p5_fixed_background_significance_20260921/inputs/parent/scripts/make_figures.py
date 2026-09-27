from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v582-mpl')
import numpy as np,pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures';S=B/'source'
labels={'2015':'2015 (full)','2016':'2016 (full)','2021':'2021 (10%)','combined':'Shared coupling (active datasets)'}
colors={'2015':'#376c94','2016':'#a1473e','2021':'#337967','combined':'#5d4582'}
plt.rcParams.update({'font.size':11.5,'axes.labelsize':11.5,'axes.titlesize':12.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17,'pdf.fonttype':42})
d=pd.read_csv(B/'results/significance_curves.csv',dtype={'scope':str});summ=pd.read_csv(B/'results/summary.csv',dtype={'scope':str});val=pd.read_csv(B/'results/field_validation.csv',dtype={'scope':str})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=170)
 plt.close(fig)
def gp_display(q):return np.where(q.global_k==0,q.global_upper95,q.global_p_addone)
def annotate_switches(ax,scope):
 if scope=='combined':
  for m in [39,50,100,180]:ax.axvline(m,color='.75',ls=':',lw=.7)
for what in ['Z','p']:
 fig,axes=plt.subplots(2,2,figsize=(10.8,7.5))
 for ax,(scope,label) in zip(axes.flat,labels.items()):
  q=d[d.scope==scope];x=q.mass_MeV.to_numpy();gp=gp_display(q);gZ=np.maximum(0,norm.isf(gp))
  if what=='Z':
   ax.plot(x,q.nominal_local_Z,color='.65',lw=.8,ls='--',label='Unshifted local root')
   ax.plot(x,q.local_Z,c='black',lw=1.15,label='Local, GP reference');ax.plot(x,gZ,c='#b3272b',lw=1.15,label='Global, GP reference');ax.set(ylabel='One-sided Z (positive part)',ylim=(0,max(4,float(q.local_Z.max())+.3)))
  else:
   ax.plot(x,q.nominal_local_p,color='.65',lw=.8,ls='--',label='Unshifted local asymptotic')
   ax.plot(x,q.local_p,c='black',lw=1.15,label='Local, GP reference');ax.plot(x,gp,c='#b3272b',lw=1.15,label='Global, GP reference');ax.set(yscale='log',ylabel='Tail probability',ylim=(max(1e-7,min(q.local_p.min(),gp.min())*.6),1.05))
   ax.fill_between(x,np.maximum(q.global_lo95.to_numpy(),1e-8),q.global_hi95.to_numpy(),color='#b3272b',alpha=.13,lw=0)
  annotate_switches(ax,scope);ax.set(title=label,xlabel='Tested mass [MeV]',xlim=(x.min(),x.max()));ax.legend(fontsize=10.5,loc='upper right' if what=='Z' else 'lower right')
 fig.suptitle('Conditional significance using fixed nominal GP background means',fontsize=14,y=1.02);fig.tight_layout(h_pad=2.0,w_pad=2.0);save(fig,'local_global_'+what+'_overview')
for scope,label in labels.items():
 q=d[d.scope==scope];x=q.mass_MeV.to_numpy();gp=gp_display(q);fig,ax=plt.subplots(2,1,figsize=(7.1,5.1),sharex=True)
 ax[0].plot(x,q.nominal_local_Z,c='.65',lw=.8,ls='--',label='Unshifted local root');ax[0].plot(x,q.local_Z,c='black',label='Local, GP reference');ax[0].plot(x,np.maximum(0,norm.isf(gp)),c='#b3272b',label='Global, GP reference');ax[0].set(ylabel='One-sided Z',ylim=(0,max(4,q.local_Z.max()+.2)));ax[0].legend(fontsize=9.5,ncol=1,loc='upper right')
 ax[1].plot(x,q.local_p,c='black');ax[1].plot(x,gp,c='#b3272b');ax[1].fill_between(x,np.maximum(q.global_lo95.to_numpy(),1e-8),q.global_hi95.to_numpy(),color='#b3272b',alpha=.16);ax[1].set(yscale='log',ylabel='Tail probability',xlabel='Tested mass [MeV]',xlim=(x.min(),x.max()),ylim=(max(1e-7,q.local_p.min()*.6),1.05))
 for aa in ax:annotate_switches(aa,scope)
 fig.suptitle(label+' — conditional on its fixed GP reference',fontsize=12);fig.tight_layout();save(fig,'local_global_'+scope)
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,(scope,label) in zip(axes.flat,labels.items()):
 z=np.load(B/f'fields/{scope}.npz');a=z['a'];s=z['s'];m=z['masses'];V=(z['validation']-a)/s
 ax.plot(m,a,label='Deterministic offset a',c=colors[scope],lw=1.2);ax.plot(m,V.mean(axis=0),label='Mean of centered toy roots',c='black',lw=.9);ax.plot(m,V.std(axis=0,ddof=1),label='SD of centered toy roots',c='.45',ls='--',lw=.9);ax.axhline(0,c='.7',lw=.5);ax.axhline(1,c='.7',lw=.5);ax.set(title=label,xlabel='Mass [MeV]',ylabel='Offset or standardized moment',xlim=(m.min(),m.max()));ax.legend(fontsize=10)
fig.suptitle('Response offsets and 256 complete Poisson scan checks',fontsize=13);fig.tight_layout();save(fig,'response_validation')
fig,axes=plt.subplots(2,2,figsize=(10,5.6))
for ax,(scope,label) in zip(axes.flat,labels.items()):
 z=np.load(B/f'fields/{scope}.npz');g=z['gaussian_maximum'];v=z['direct_maximum'];xx=np.linspace(0,max(4.5,np.quantile(g,.999)),120)
 ax.plot(xx,[np.mean(g>=t) for t in xx],c=colors[scope],label='Response-GP fields (200,000)');ax.step(np.sort(v),np.arange(len(v),0,-1)/len(v),where='pre',c='black',lw=.9,label='Complete Poisson scans (256)');ax.set(title=label,xlabel='Maximum gated reference score',ylabel='Tail probability',yscale='log',ylim=(.001,1),xlim=(0,xx.max()));ax.legend(fontsize=10)
fig.suptitle('Scan-maximum validation under the same fixed background means',fontsize=13);fig.tight_layout();save(fig,'global_validation')
def tex(name,head,rows,align,foot=''):
 (S/name).write_text('\\begin{center}\\small\\begin{tabular}{'+align+'}\\toprule\n'+head+'\\\\\\midrule\n'+'\n'.join(' & '.join(r)+'\\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'+foot+'\n')
rows=[]
for scope,label in labels.items():
 q=summ[summ.scope==scope].iloc[0];lab=label.replace('%','\\%').replace('Shared coupling (active datasets)','Combined')
 rows.append([lab,f'{q.mass_MeV:g}',f'{q.observed_r:.3f}',f'{q.peak_local_Z:.3f}',f'{q.global_p_addone:.4f}',f'{q.peak_global_Z:.3f}'])
tex('peak_table.tex','Scope & $m$ [MeV] & Raw $r$ & Local $Z$ & Global $p$ & Global $Z$',rows,'lrrrrr','Peaks maximize the positive-fit-gated GP-reference local score. Global estimates use the full stated search range for each scope; they do not include selection among the four reported scopes.')
rows=[]
for scope,label in labels.items():
 q=summ[summ.scope==scope].iloc[0];v=val[val.scope==scope].iloc[0]
 rows.append([label.replace('%','\\%').replace('Shared coupling (active datasets)','Combined'),str(int(q.grid_nodes)),f'{q.source_rms_offset:.3f}',f'{v.rms_centered_bias:.3f}',f'{v.mean_centered_width:.3f}',f'{v.correlation_RMSE:.3f}'])
tex('validation_table.tex','Scope & Nodes & RMS $a$ & RMS mean & Mean width & Corr. RMS',rows,'lrrrrr','Toy means and widths refer to $(r-a)/s$. Correlation RMS compares the complete-scan empirical correlation with the response-GP prediction; finite Monte Carlo fluctuations contribute.')
rows=[]
for scope,label in labels.items():
 q=summ[summ.scope==scope].iloc[0];rows.append([label.replace('%','\\%').replace('Shared coupling (active datasets)','Combined'),str(int(q.global_k)),str(int(q.direct_global_k)),f'{q.direct_global_lo95:.3f}--{q.direct_global_hi95:.3f}'])
tex('global_validation_table.tex','Scope & GP $k/200{,}000$ & Direct $k/256$ & Direct 95\\% interval',rows,'lrrr','Both counts compare each simulated scan maximum with the corresponding observed maximum. The direct ensemble is a finite check, not a rare-tail precision measurement.')
print(summ[['scope','mass_MeV','peak_local_Z','global_p_addone','peak_global_Z','source_rms_offset']].to_string(index=False))
# Predeclared common overlap: same exact combined fits, a distinct stated search domain.
from scipy.stats import beta
q=d[(d.scope=='combined')&(d.mass_MeV>=50)&(d.mass_MeV<=100)].copy();f=np.load(B/'fields/combined.npz');g=f['gaussian_overlap_maximum'];cm=(f['masses']>=50)&(f['masses']<=100);vv=(f['validation']-f['a'])/f['s'];vm=np.where(f['validation'][:,cm]>0,vv[:,cm],-np.inf).max(axis=1)
for i,r in q.iterrows():
 if r.observed_positive_fit:
  k=int(np.sum(g>=r.z));n=len(g);p=(k+1)/(n+1);lo=0. if k==0 else beta.ppf(.025,k,n-k+1);hi=1. if k==n else beta.ppf(.975,k+1,n-k);kd=int(np.sum(vm>=r.z))
 else:k=n=len(g);p=lo=hi=1.;kd=len(vm)
 for key,value in dict(global_k=k,global_N=len(g),global_p_addone=p,global_lo95=lo,global_hi95=hi,global_Z=max(0,float(norm.isf(p))),direct_global_k=kd,direct_global_N=len(vm)).items():q.loc[i,key]=value
q.to_csv(B/'results/combined_overlap_significance.csv',index=False,float_format='%.17g')
fig,ax=plt.subplots(2,1,figsize=(7.1,5.1),sharex=True);x=q.mass_MeV.to_numpy()
ax[0].plot(x,q.nominal_local_Z,c='.65',ls='--',lw=.8,label='Unshifted local root');ax[0].plot(x,q.local_Z,c='black',label='Local, GP reference');ax[0].plot(x,q.global_Z,c='#b3272b',label='Global over 50–100 MeV');ax[0].set(ylabel='One-sided Z',ylim=(0,4));ax[0].legend(fontsize=9.5)
ax[1].plot(x,q.local_p,c='black');ax[1].plot(x,q.global_p_addone,c='#b3272b');ax[1].fill_between(x,q.global_lo95.to_numpy(),q.global_hi95.to_numpy(),color='#b3272b',alpha=.16);ax[1].set(yscale='log',ylabel='Tail probability',xlabel='Tested mass [MeV]',xlim=(50,100),ylim=(.001,1.05))
fig.suptitle('All three datasets: predeclared 50–100 MeV overlap',fontsize=12);fig.tight_layout();save(fig,'local_global_combined_common_overlap')
