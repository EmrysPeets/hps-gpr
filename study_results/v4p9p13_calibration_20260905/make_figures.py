#!/usr/bin/env python3
"""Scientific plots from completed calibration ledgers; no fits performed."""
from pathlib import Path
import os,hashlib,io,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hps-v4p9p13-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':11,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.5,'pdf.fonttype':3,'savefig.dpi':180})
SCOPES=[('individual_2015_full','2015, 100%',19,90,'2015'),('individual_2016_full','2016, 100%',39,180,'2016'),('individual_2021_10pct','2021, 10%',50,250,'2021'),('all_2015_2016_2021','All three datasets; shared $\\epsilon^2$',50,90,'combined')]
COLORS={'profiled':'#2166ac','fixed':'#b94728'}

def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),bbox_inches='tight');plt.close(fig)

def main():
 source=HERE/'summary/observed_calibrated_limits.csv';payload=source.read_bytes()
 d=pd.read_csv(io.BytesIO(payload))
 for key,label,lo,hi,slug in SCOPES:
  b=d[d.scope_key==key].sort_values('mass_MeV')
  if b.empty:continue
  x=b.mass_MeV.to_numpy()
  fig,(ax,ratio)=plt.subplots(2,1,figsize=(8.15,6.7),sharex=True,gridspec_kw={'height_ratios':[3.2,1.1]})
  fig.subplots_adjust(top=.80,bottom=.19,left=.105,right=.965,hspace=.32)
  fig.suptitle(label,fontsize=15,y=.98)
  subtitle='Observed 90% CLs; conditional calibration with reviewed kernels fixed'
  completed=int(b.checkpoint_completed.sum())
  if completed!=hi-lo+1:subtitle+=f'\nIn progress: {completed} of {hi-lo+1} coordinates'
  fig.text(.5,.93,subtitle,ha='center',va='top',fontsize=9)
  for method,raw in [('profiled','eps2_current_display'),('fixed','eps2_fixed_display')]:
   name='Gaussian profile' if method=='profiled' else 'Fixed GP mean';color=COLORS[method]
   ax.plot(x,b[raw],color=color,lw=1.,ls='--',alpha=.7,label=name+': asymptotic')
   y=b[f'eps2_{method}_calibrated'].to_numpy(float)
   finite=np.isfinite(y)&(y>0)
   yplot=np.where(finite,y,np.nan)
   ax.plot(x,yplot,color=color,lw=1.6,label=name+': toy calibrated')
   lower=b[f'eps2_{method}_mc_low'].to_numpy(float);upper=b[f'eps2_{method}_mc_high'].to_numpy(float)
   good=finite&np.isfinite(lower)&np.isfinite(upper)&(lower>0)
   ax.fill_between(x,np.where(good,lower,np.nan),np.where(good,upper,np.nan),color=color,alpha=.13,lw=0)
   censored=b[f'status_{method}'].eq('right_censored').to_numpy()
   ax.scatter(x[censored],np.full(censored.sum(),.98),transform=ax.get_xaxis_transform(),marker='^',s=20,color=color)
   limited=finite&~b[f'status_{method}'].eq('resolved').to_numpy()
   ax.scatter(x[limited],y[limited],facecolors='white',edgecolors=color,s=15,lw=.65,zorder=4)
  ax.set_yscale('log');ax.set_ylabel(r'Observed upper limit on $\epsilon^2$')
  ax.set_xlim(lo,hi);ax.tick_params(labelbottom=False)
  ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=2,frameon=False,borderaxespad=0,columnspacing=1.8)
  raw_ratio=b.eps2_fixed_display/b.eps2_current_display
  cal_ratio=b.eps2_fixed_calibrated/b.eps2_profiled_calibrated
  ratio.plot(x,raw_ratio,color='.55',ls='--',lw=1,label='Asymptotic ratio')
  ratio.plot(x,cal_ratio.replace([np.inf,-np.inf],np.nan),color='#624086',lw=1.35,label='Calibrated ratio')
  ratio.axhline(1,color='.3',lw=.65);ratio.set_ylabel('Fixed / profile');ratio.set_xlabel(r'Resonance mass $m_{A\prime}$ [MeV]')
  ratio.legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,ncol=2,fontsize=8,borderaxespad=0)
  fig.text(.105,.018,'Shading: approximate 95% Monte Carlo uncertainty, not expected-limit bands.\nOpen circles: limited MC precision. Triangles: no finite endpoint. Both limits target 90% CLs.',fontsize=8,va='bottom')
  save(fig,'limits_'+slug)
 pcal=d[['p0_profiled_calibrated','p0_fixed_calibrated']].to_numpy(float)
 positive=pcal[np.isfinite(pcal)&(pcal>0)]
 floor_exp=int(np.floor(np.log10(min(1e-5,positive.min())))) if positive.size else -5
 floor=10.**floor_exp
 fig,axes=plt.subplots(2,2,figsize=(9.5,7.7))
 fig.subplots_adjust(top=.82,bottom=.12,left=.085,right=.98,hspace=.35,wspace=.24)
 for ax,(key,label,lo,hi,slug) in zip(axes.flat,SCOPES):
  b=d[d.scope_key==key].sort_values('mass_MeV');x=b.mass_MeV
  for method,raw in [('profiled','p0_current'),('fixed','p0_fixed')]:
   color=COLORS[method];name='Gaussian profile' if method=='profiled' else 'Fixed GP mean'
   ax.plot(x,b[raw].where(b[raw]>0),ls='--',color=color,alpha=.5,lw=.9,label=name+': asymptotic')
   y=b[f'p0_{method}_calibrated'];ax.plot(x,y.where(y>0),color=color,lw=1.4,label=name+': calibrated')
   bad=~b[f'status_p0_{method}'].isin(['resolved','bounded_atom'])&(y>0)
   ax.scatter(x[bad],y[bad],s=12,facecolors='white',edgecolors=color,lw=.5)
  ax.axhline(.05,color='.35',lw=.6,ls=':');ax.set_yscale('log');ax.set_ylim(bottom=floor,top=1.3)
  # Explicit arrows retain the existence of very small asymptotic values.
  for method,raw in [('profiled','p0_current'),('fixed','p0_fixed')]:
   below=b[raw]<floor;ax.scatter(x[below],np.full(below.sum(),floor*1.07),marker='v',s=11,color=COLORS[method],alpha=.5)
  ax.set_xlim(lo,hi);ax.set_title(label);ax.set_xlabel('Mass [MeV]');ax.set_ylabel(r'Local $p_0$')
 handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False,fontsize=9)
 fig.suptitle('Local p-values: asymptotic and conditional toy calibration',fontsize=14,y=.985)
 fig.text(.5,.025,rf'No global trials correction. Triangles: asymptotic $p_0<10^{{{floor_exp}}}$. Open circles: limited MC precision or MC boundary.',fontsize=8,ha='center')
 save(fig,'local_pvalues')
 paths=[OUT/(name+ext) for name in ['limits_'+s[-1] for s in SCOPES]+['local_pvalues'] for ext in ('.pdf','.png')]
 manifest=dict(source=str(source),source_sha256=hashlib.sha256(payload).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completed_points=int(d.checkpoint_completed.sum()),outputs={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 (OUT/'limit_plot_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
