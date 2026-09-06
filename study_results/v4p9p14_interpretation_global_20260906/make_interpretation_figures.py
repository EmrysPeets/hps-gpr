#!/usr/bin/env python3
"""Reader-facing figures built only from frozen v4.9.13 results."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'v4p9p13_calibration_20260905'
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':170})
BLUE='#18699C';RED='#AE3E30';GREEN='#287A55';PURPLE='#74459B'
d=pd.read_csv(PARENT/'summary/observed_calibrated_limits.csv')
t=pd.read_csv(PARENT/'summary/truth_specific_limits.csv')
used=[PARENT/'summary/observed_calibrated_limits.csv',PARENT/'summary/truth_specific_limits.csv',Path(__file__)]
def save(fig,name):
 for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9.4,4.2),sharey=True)
for ax,scope,title in zip(axes,['individual_2016_full','all_2015_2016_2021'],['2016 full sample','Combined 2015 + 2016 + 2021']):
 x=d[(d.scope_key==scope)&d.mass_MeV.between(58,82)].sort_values('mass_MeV')
 ax.plot(x.mass_MeV,x.eps2_profiled_asymptotic,color=BLUE,label='Asymptotic profile')
 ax.plot(x.mass_MeV,x.eps2_profiled_calibrated,color=RED,label='Calibrated profile')
 ax.fill_between(x.mass_MeV,x.eps2_profiled_mc_low,x.eps2_profiled_mc_high,color=RED,alpha=.15)
 ax.set_yscale('log');ax.set_xlabel('Tested mass [MeV]');ax.set_title(title);ax.grid(alpha=.18)
axes[0].set_ylabel(r'Observed 90% $CL_s$ limit on $\epsilon^2$')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,1.025),frameon=False)
fig.tight_layout(rect=(0,0,1,.93));save(fig,'limit_hump_explained')

fig,axes=plt.subplots(1,2,figsize=(9.4,4.2),gridspec_kw={'width_ratios':[1.1,1]})
x=d[(d.scope_key=='all_2015_2016_2021')&d.mass_MeV.between(58,82)].sort_values('mass_MeV')
ax=axes[0]
ax.plot(x.mass_MeV,x.p0_profiled_asymptotic,color=BLUE,label='Asymptotic')
for truth,color,label in [('gp',GREEN,'GP-generated background'),('stress',PURPLE,'Archived stress background')]:
 y=t[(t.scope_key=='all_2015_2016_2021')&(t.method=='profiled')&(t.truth==truth)&t.mass_MeV.between(58,82)].sort_values('mass_MeV')
 ax.plot(y.mass_MeV,y.p0,color=color,ls='--',label=label)
ax.plot(x.mass_MeV,x.p0_profiled_calibrated,color=RED,lw=2,label='Larger of the two tails')
ax.set_yscale('log');ax.set_ylim(1e-4,1.5);ax.set_xlabel('Tested mass [MeV]');ax.set_ylabel('Local background-only tail probability');ax.set_title('Combined profiled local p-value');ax.grid(alpha=.18)
row=x[x.mass_MeV==66].iloc[0]
ledger=Path(row.checkpoint_path).parent/'validation_toys.csv.gz';used.append(ledger)
toys=pd.read_csv(ledger);toys=toys[(toys.method=='profiled')&(toys.strength==0)]
bins=np.linspace(-4,13,35)
for truth,color,label in [('gp',GREEN,'GP background'),('stress',PURPLE,'Stress background')]:
 values=toys[toys.truth==truth].signed_r
 axes[1].hist(values,bins=bins,histtype='step',lw=1.8,color=color,density=True,label=label)
axes[1].axvline(row.signed_r_profiled_asymptotic,color='black',lw=1.5,label='Observed result')
axes[1].set_title('At 66 MeV: 500 null toys per model');axes[1].set_xlabel('Signed likelihood-ratio root');axes[1].set_ylabel('Toy probability density');axes[1].grid(alpha=.15)
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=8,ncol=2,frameon=False)
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=8,ncol=1,frameon=False)
fig.tight_layout();save(fig,'pvalue_hump_explained')

rows=[]
for scope,label in [('individual_2015_full','2015'),('individual_2016_full','2016'),('individual_2021_10pct','2021 (10%)'),('all_2015_2016_2021','Combined')]:
 x=d[d.scope_key==scope]
 rows.append(dict(scope=scope,label=label,
  profiled_cal_over_asym=float((x.eps2_profiled_calibrated/x.eps2_profiled_asymptotic).median()),
  fixed_raw_over_profiled_raw=float((x.eps2_fixed_asymptotic/x.eps2_profiled_asymptotic).median()),
  fixed_cal_over_profiled_cal=float(x.ratio_fixed_over_profiled_calibrated.median())))
summary=pd.DataFrame(rows);summary.to_csv(HERE/'provenance/observed_limit_ratios.csv',index=False)
fig,ax=plt.subplots(figsize=(8.9,3.7));i=np.arange(len(summary));w=.34
ax.bar(i-w/2,summary.fixed_raw_over_profiled_raw,w,color=BLUE,label='Before calibration')
ax.bar(i+w/2,summary.fixed_cal_over_profiled_cal,w,color=RED,label='After calibration')
ax.axhline(1,color='black',lw=1);ax.set_xticks(i,summary.label);ax.set_ylabel('Median fixed / profiled observed limit');ax.set_ylim(0,1.7);ax.grid(axis='y',alpha=.18)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.21),ncol=2,frameon=False)
fig.tight_layout();save(fig,'fixed_profiled_comparison')

(HERE/'provenance/interpretation_figures.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in used},'outputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if any(p.stem==n for n in ['limit_hump_explained','pvalue_hump_explained','fixed_profiled_comparison'])}},indent=2)+'\n')
print(summary.to_string(index=False))
