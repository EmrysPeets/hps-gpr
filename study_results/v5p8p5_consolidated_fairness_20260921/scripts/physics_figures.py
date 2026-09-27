"""Explain saved response moments and new conditional high-psum-source checks."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-physics-fairness-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];R=B/'results';F=B/'figures'
P=B/'inputs/v5p8p2_nominal_gp_significance_20260917'
if not P.exists():P=ROOT/'study_results/v5p8p2_nominal_gp_significance_20260917'
if (B/'inputs/physics_engine').exists():P=B/'inputs/physics_engine'
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
colors=['#235b7e','#b34a36','#38836a']
def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=155)
    plt.close(fig)
f=np.load(P/'fields/2016.npz');m=f['masses'];a=f['a'];s=f['s'];v=f['validation'];z=(v-a)/s
fig,ax=plt.subplots(2,2,figsize=(10.4,6.4),sharex=True)
ax[0,0].plot(m,f['observed_r'],c='.25',lw=.9,label='Observed signed root')
ax[0,0].plot(m,a,c=colors[1],lw=1.3,label='Deterministic root a = r(B)')
ax[0,0].set(ylabel='Raw signed likelihood root',title='A. Observed scan and source response')
ax[0,0].legend(fontsize=9)
ax[0,1].plot(m,a,c=colors[1],lw=1.3,label='a = r(B)')
ax[0,1].plot(m,v.mean(axis=0),c='black',lw=.8,label='Mean raw toy root')
ax[0,1].set(ylabel='Raw signed likelihood root',title='B. The conditional toy mean follows a')
ax[0,1].legend(fontsize=9)
ax[1,0].plot(m,z.mean(axis=0),c='black',lw=.9)
ax[1,0].axhline(0,c=colors[1],lw=1,ls='--');ax[1,0].axhspan(-1/16,1/16,color='.8',alpha=.35)
ax[1,0].set(ylabel='Mean of (r toy − a) / s',title='C. Centered moment: compare with zero',ylim=(-.2,.2),xlabel='Tested mass [MeV]')
ax[1,1].plot(m,z.std(axis=0,ddof=1),c='.25',lw=.9,label='SD of (r toy − a) / s')
ax[1,1].axhline(1,c=colors[1],ls='--',lw=1);ax[1,1].axhspan(1-1/np.sqrt(510),1+1/np.sqrt(510),color='.8',alpha=.35)
ax[1,1].set(ylabel='Standardized toy width',title='D. Scaled moment: compare with one',xlabel='Tested mass [MeV]',ylim=(.8,1.23))
for aa in ax.flat:aa.set_xlim(39,180)
fig.suptitle('2016: unpacking v5.8.2 Figure 3 — one frozen GP source, 256 scans',fontsize=13)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'physics_fig3_explained')
source=np.load(R/'physics_sources.npz');scan=pd.read_csv(R/'physics_source_scan.csv');transfer=pd.read_csv(R/'physics_signal_transfer.csv')
fig,ax=plt.subplots(3,1,figsize=(9.5,8.8),gridspec_kw={'height_ratios':[1,1.2,1.3]})
x=source['x']*1000;ax[0].plot(x,source['high1_times10']/source['native10_gp'],c=colors[0],lw=1.3)
ax[0].axhline(1,c='.5',lw=.9,ls='--');ax[0].set(xlim=(50,250),ylabel='10 × B(high 1%) / B(native 10%)',title='A. Nominal ×10 does not reproduce the native-10% spectrum')
labels={'high1_native':'High-psum nominal 1%','high1_times10':'Same fixed mean ×10','native10':'Native 10% GP source'}
for key,color in zip(labels,colors):
    q=scan[scan.source==key];ax[1].plot(q.mass_MeV,q.a,c=color,label=labels[key],lw=1.2)
ax[1].axhline(0,c='.65',lw=.6);ax[1].set(xlim=(50,250),ylabel='Deterministic root a = r(B)',xlabel='Tested mass [MeV]',title='B. Small self-consistency offsets do not certify a background null')
ax[1].legend(fontsize=9,ncol=3,loc='upper right')
i=np.arange(2);width=.24
for offset,key,label,color in [(-1,'fixed_source_fit_recovery','Extraction: fixed source',colors[0]),(0,'rebuilt_source_window_absorption','Source absorbs window counts',colors[1]),(1,'rebuilt_source_template_absorption','Source absorbs matched shape',colors[2])]:
    vals=transfer[key].to_numpy();bars=ax[2].bar(i+offset*width,vals,width,label=label,color=color)
    for bb,val in zip(bars,vals):ax[2].text(bb.get_x()+bb.get_width()/2,val+.02,f'{val:.3f}',ha='center',fontsize=9)
ax[2].axhline(1,c='.55',ls='--',lw=.8);ax[2].set(xticks=i,xticklabels=['76 MeV','90 MeV'],ylim=(0,1.17),ylabel='Fraction of injected amplitude / yield',title='C. Signal still enters a rebuilt 1% source before the ×10 transfer')
ax[2].legend(fontsize=9,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13))
fig.suptitle('2021 high-psum 1% control: conditional source transfer at fixed ±2.25σ',fontsize=13)
fig.tight_layout(rect=(0,0,1,.96),h_pad=2.1);save(fig,'physics_source_transfer')
print('Created physics figures')
