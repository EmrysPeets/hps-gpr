#!/usr/bin/env python3
"""Figures and compact tables from the completed, fixed intervention scans."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hps-candidate-removal-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE=Path(__file__).resolve().parent; F=HERE/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'savefig.facecolor':'white'})
YEARS=[2015,2016,2021]; SCOPE={2015:'2015 · full data',2016:'2016 · full data',2021:'2021 · 10% data'}
COL={'original':'#20252b','first_mean':'#2679b2','second_mean':'#ce761d','both_mean':'#21815c','both_poly_mean':'#9764b1','both_wide_mean':'#c55552'}
holes=pd.read_csv(HERE/'derived/holes.csv'); metrics=pd.read_csv(HERE/'derived/oscillation_metrics.csv')
def scan(year):return pd.read_csv(HERE/'derived'/str(year)/'scans.csv').pivot(index='mass_MeV',columns='lane',values='r')
def shade(ax,year):
    for h in holes[holes.dataset.eq(year)].itertuples():ax.axvspan(h.low_MeV,h.high_MeV,color='#d2d5d8',alpha=.26,lw=0,zorder=0)
def format_axis(ax):
    ax.axhline(0,color='.5',lw=.8,zorder=0);ax.grid(axis='y',color='.9',lw=.7)
    ax.spines[['top','right']].set_visible(False)
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(7.2,6.0),layout='constrained')
for year,ax in zip(YEARS,axes):
    d=scan(year);shade(ax,year);x=d.index.to_numpy()
    reps=d[[f'observed_both_rep{i:02d}' for i in range(10)]].to_numpy()
    ax.fill_between(x,reps.min(1),reps.max(1),color=COL['both_mean'],alpha=.17,lw=0)
    for suffix in ['original','first_mean','second_mean','both_mean']:
        ax.plot(x,d['observed_'+suffix],color=COL[suffix],lw=1.65 if suffix in ['original','both_mean'] else 1.1,ls='--' if suffix=='second_mean' else '-',zorder=3 if suffix=='both_mean' else 2)
    hh=holes[holes.dataset.eq(year)].sort_values('rank')
    ax.set_title(SCOPE[year]+f"     |     first hole: {int(hh.iloc[0].mass_MeV)} MeV; second: {int(hh.iloc[1].mass_MeV)} MeV",loc='left')
    ax.set_ylabel('Signed fit root $r$');ax.set_xlim(x[0],x[-1]);format_axis(ax)
    ax.set_xlabel('Test mass [MeV]')
handles=[Line2D([],[],color=COL[s],lw=1.7,ls='--' if s=='second_mean' else '-',label=l) for s,l in [('original','Original data'),('first_mean','Replace first'),('second_mean','Replace second'),('both_mean','Replace both')]]
handles.append(Patch(color=COL['both_mean'],alpha=.17,label='Range of 10 paired replacements (both)'))
fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False)
save(fig,'observed_candidate_removal')

fig,axes=plt.subplots(3,2,figsize=(7.2,5.15),layout='constrained')
for row,year in enumerate(YEARS):
    d=scan(year)
    for col,source in enumerate(['observed','reference']):
        ax=axes[row,col];shade(ax,year)
        for suffix in ['original','both_mean','both_poly_mean','both_wide_mean']:
            ax.plot(d.index,d[source+'_'+suffix],color=COL[suffix],lw=1.35,ls='--' if 'wide' in suffix else '-')
        ax.set_title(SCOPE[year]+(' · observed' if source=='observed' else ' · reference'),loc='left',fontsize=8.5)
        ax.set_ylabel('$r_m$' if source=='observed' else '$a_m = r_m(B)$')
        ax.set_xlim(d.index.min(),d.index.max());ax.set_xlabel('Test mass [MeV]');format_axis(ax)
handles=[Line2D([],[],color=COL[s],lw=1.7,ls='--' if 'wide' in s else '-',label=l) for s,l in [('original','Original spectrum'),('both_mean','GP fill, ±2.25σ'),('both_poly_mean','Polynomial fill, ±2.25σ'),('both_wide_mean','GP fill, ±3σ')]]
fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False)
save(fig,'replacement_model_comparison')

summary=[]; contrasts=[]
for year in YEARS:
    mm=metrics[metrics.dataset.eq(year)&metrics.selection.eq('remote')].set_index('lane')
    row=dict(dataset=year,remote_masses=int(mm.loc['observed_original','n_points']))
    for source in ['observed','reference']:
        for suffix in ['both_mean','both_poly_mean','both_wide_mean']:
            row[source+'_'+suffix]=float(mm.loc[source+'_'+suffix,'retained_std'])
        row[source+'_correlation']=float(mm.loc[source+'_both_mean','correlation'])
    reps=mm.loc[[f'observed_both_rep{i:02d}' for i in range(10)],'retained_std']
    row['observed_replicate_min']=float(reps.min());row['observed_replicate_max']=float(reps.max());summary.append(row)
    d=scan(year)
    for mass in list(holes[holes.dataset.eq(year)].mass_MeV)+[{2015:19,2016:102,2021:71}[year]]:
        contrasts.append(dict(dataset=year,mass_MeV=int(mass),kind='selected peak' if mass in holes[holes.dataset.eq(year)].mass_MeV.to_list() else 'illustrative deficit',**{suffix:float(d.loc[mass,'observed_'+suffix]) for suffix in ['original','first_mean','second_mean','both_mean','both_poly_mean','both_wide_mean']}))
pd.DataFrame(summary).to_csv(HERE/'derived/remote_summary.csv',index=False)
pd.DataFrame(contrasts).to_csv(HERE/'derived/selected_root_changes.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
