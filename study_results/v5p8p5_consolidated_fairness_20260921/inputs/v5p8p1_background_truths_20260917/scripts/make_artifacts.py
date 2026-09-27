#!/usr/bin/env python3
"""Rebuild figures and report tables from saved numerical results only."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v581-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures';S=B/'source'
labels={'archived_stress':'Archived stress','gp_full_nominal':'Full GP, nominal','gp_full_half_ls':'Full GP, half length','gp_blocked':'Blocked GP','regional_rise_fall':'Regional rise/fall','gp_blocked_local':'Blocked, local kernels'}
colors={'archived_stress':'#af4039','gp_full_nominal':'#286596','gp_full_half_ls':'#27806f','gp_blocked':'#9752a0','regional_rise_fall':'#be8721','gp_blocked_local':'#474b83'}
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.16})
def save(fig,name):
 fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=160)
 plt.close(fig)
def table(filename,head,rows,align,foot=''):
 (S/filename).write_text('\\begin{center}\\small\\begin{tabular}{'+align+'}\\toprule\n'+head+'\\\\\\midrule\n'+'\n'.join(' & '.join(row)+'\\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'+foot+'\n')
z=dict(np.load(B/'truths/backgrounds.npz'));z.update(dict(np.load(B/'truths/supplementary.npz')));names=list(labels);m=(z['edges_GeV'][:-1]+z['edges_GeV'][1:])*500
fig,ax=plt.subplots(2,1,figsize=(6.7,4.6),sharex=True)
ax[0].plot(m,z['observed']/1000,c='.65',lw=.65,label='Observed counts')
for n in names:
 ax[0].plot(m,z[n]/1000,label=labels[n],c=colors[n],lw=1.2)
 
 if n in names[:3]:ax[1].plot(m,100*(z[n]/z['observed']-1),c=colors[n],lw=.8)
ax[0].set(ylabel='Counts per bin [thousands]');ax[0].legend(fontsize=7.5,ncol=2)
ax[1].axhline(0,c='black',lw=.5);ax[1].set(xlabel='Mass [MeV]',ylabel='(Mean / observed − 1) [%]',title='Archived and full-data GP references',xlim=(30,210))
save(fig,'backgrounds')
a=pd.read_csv(B/'results/asimov_scan.csv');summary=pd.read_csv(B/'results/asimov_summary.csv');inj=pd.read_csv(B/'results/injections.csv');local=pd.read_csv(B/'results/local_tail_tests.csv')
fig,ax=plt.subplots(2,1,figsize=(6.7,3.8))
for n in names:
 q=a[a.truth==n];aa=ax[0] if n in names[:3] else ax[1];aa.plot(q.mass_MeV,q.signed_r,label=labels[n],c=colors[n],lw=1.1)
ax[0].set(xlim=(39,180),ylabel='Deterministic signed root',title='Archived and full-data GP controls');ax[0].legend(ncol=2,fontsize=7.5)
ax[1].set(xlim=(39,180),xlabel='Tested mass [MeV]',ylabel='Deterministic signed root',title='Blocked and regional proposals: poor source agreement');ax[1].legend(ncol=2,fontsize=7.5)
for aa in ax:aa.axhline(0,c='black',lw=.6)
save(fig,'response_scan')
rows=[]
for n in names:
 q=summary[summary.truth==n].iloc[0];j=inj[inj.truth==n]
 rows.append([labels[n],f'{q.rms_root:.3f}',f'{q.minimum_root:.3f}',f'{q.maximum_root:.3f}',f'{j.recovered_fraction.min():.3f}--{j.recovered_fraction.max():.3f}'])
table('summary_table.tex','Reference & RMS $a$ & Min $a$ & Max $a$ & Yield recovery',rows,'lrrrr','Yield recovery is the fitted-amplitude increment divided by the fixed injected amplitude, across both strengths and all six anchors. It is not detection probability.')
rows=[]
for mass in [42,66,76,90,92,117]:
 rows.append([str(mass)]+[f'{a[(a.mass_MeV==mass)&(a.truth==n)].iloc[0].signed_r:.2f}' for n in names])
table('anchor_table.tex','$m$ & Old stress & Full GP & Half-$\\ell$ GP & Blocked GP & Regional & Local block',rows,'rrrrrrr')
rows=[]
for mass in [42,66,76,90,92,117]:
 q=local[local.mass_MeV==mass];rows.append([str(mass),f'{q.iloc[0].observed_r:.3f}']+[('fail' if q[q.truth==n].empty else ('atom' if q[q.truth==n].iloc[0].bounded_atom else str(int(q[q.truth==n].iloc[0].exceedances)))) for n in names])
table('tail_table.tex','$m$ & $r_{\\rm obs}$ & Old & Full GP & Half-$\\ell$ & Blocked & Regional & Local block',rows,'rrrrrrrr','Counts are inclusive excess-tail exceedances out of 512 per reference. A nonpositive observed root gives $q_0=0$ and the exact inclusive tail of one (atom). Full estimates and intervals are in the CSV; counts at zero do not resolve the tail.')
fig,ax=plt.subplots(1,2,figsize=(6.7,2.3))
for aa,mass in zip(ax,[76,90]):
 x=np.load(B/f'results/local_{mass}.npz')
 for n in names[:3]:aa.hist(x[n],bins=25,density=True,histtype='step',color=colors[n],lw=1.1,label=labels[n])
 aa.axvline(float(x['observed_r']),c='black',ls='--',lw=1.2,label='Observed');aa.set(title=f'{mass} MeV',xlabel='Signed likelihood root',ylabel='Density')
ax[0].legend(fontsize=6.5);save(fig,'local_distributions')
write={'reference_names':labels,'rms':{r.truth:r.rms_root for r in summary.itertuples()},'scan_fits':len(a),'injections':len(inj),'local_scenarios':len(local),'poisson_experiments':int(local.N.sum()),'local_width_min':float(local.sd_r.min()),'local_width_max':float(local.sd_r.max())}
(B/'results/report_summary.json').write_text(json.dumps(write,indent=2)+'\n')
print(json.dumps(write,indent=2))

leak=pd.read_csv(B/'truths/source_absorption.csv');rows=[]
for n in names[1:]:
 q=leak[leak.truth==n]
 rows.append([labels[n]]+[f'{q[q.mass_MeV==m].iloc[0].source_absorbed_window_sum_fraction*100:.1f}\\%' for m in [76,90]]+[f'{q[q.mass_MeV==m].iloc[0].template_poisson_projection_fraction*100:.1f}\\%' for m in [76,90]])
table('source_leakage_table.tex','Source builder & Window76 & Window90 & Shape76 & Shape90',rows,'lrrrr','Window is the change of the constructed mean summed in the original signal window, divided by the added signal there. Shape is the Poisson-weighted projection onto the injected template. Neither is an efficiency or a measured signal bias.')
