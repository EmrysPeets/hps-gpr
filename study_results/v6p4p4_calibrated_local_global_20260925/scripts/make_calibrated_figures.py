#!/usr/bin/env python3
"""Plot frozen-A/local-first comparisons from saved v6.4.4 result tables."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v644-calibrated-figures')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

B=Path(__file__).resolve().parents[1];R=B/'results';F=B/'figures'
SCOPES=('2016','2021','combined');LABELS={'2016':'2016','2021':'2021','combined':'Combined'}
GRAY='#777777';RED='#b13c39';GOLD='#b18221';PURPLE='#79539a'
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.8,
    'legend.fontsize':7.3,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    F.mkdir(exist_ok=True)
    fig.savefig(F/(name+'.pdf'))
    fig.savefig(F/(name+'.png'),dpi=180)
    plt.close(fig)

def z(p):return np.maximum(0,norm.isf(np.asarray(p,float)))

def mass_comparison(curves,summary):
    fig,axs=plt.subplots(3,2,figsize=(7.25,6.05))
    lines=[('raw_2048',GRAY,'--','Raw-statistic maximum: pooled 2,048'),
           ('minp_B',RED,'-','Local-first minimum: independent B, 1,024'),
           ('sidak_fitted',GOLD,'-.','Sidak: constant effective count from A'),
           ('sidak_grid',PURPLE,':','Sidak: independent mass-grid reference')]
    for i,scope in enumerate(SCOPES):
        q=curves[curves.scope==scope].sort_values('mass_MeV');s=summary.loc[scope]
        assert len(q)==(136 if scope=='2016' else 181)
        for key,color,style,label in lines:
            for ax,suffix in zip(axs[i],['p','Z']):
                ax.plot(q.mass_MeV,q[key+'_'+suffix],color=color,ls=style,lw=1.25 if key=='minp_B' else 1.0,label=label)
        axs[i,0].fill_between(q.mass_MeV,q.minp_B_p95_low,q.minp_B_p95_high,color=RED,alpha=.10,lw=0)
        lo=z(q.minp_B_p95_high);hi=z(q.minp_B_p95_low)
        finite=np.isfinite(hi);axs[i,1].fill_between(q.mass_MeV,lo,hi,where=finite,color=RED,alpha=.10,lw=0)
        for ax in axs[i]:
            ax.set_xlim(float(q.mass_MeV.min()),float(q.mass_MeV.max()));ax.grid(axis='y',alpha=.16)
            ax.axvline(s.localfirst_peak_mass_MeV,color=RED,lw=.65,ls=':',alpha=.75)
            if scope=='combined':
                for x in [100.5,175.5]:ax.axvline(x,color='.75',lw=.6,ls=':',zorder=-3)
        vals=q[[key+'_p' for key,*_ in lines]].to_numpy()
        axs[i,0].set(yscale='log',ylim=(max(.001,float(vals.min())*.68),1.12),ylabel=LABELS[scope]+'\nGlobal p')
        axs[i,1].set(ylabel='Global Z',ylim=(0,max(1.0,float(q[[key+'_Z' for key,*_ in lines]].max().max())*1.18)))
        if i==0:
            axs[i,0].set_title('Probability after accounting for the mass search',loc='left',fontsize=8)
            axs[i,1].set_title('The same probabilities in Gaussian units',loc='left',fontsize=8)
        if i==2:
            for ax in axs[i]:ax.set_xlabel('Generated signal mass [MeV]')
    handles=[Line2D([],[],color=c,ls=ls,lw=1.4,label=lab) for _,c,ls,lab in lines]
    fig.text(.5,.993,'Two search statistics, two distinct Sidak references',ha='center',va='top',fontsize=10.5)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.96),ncol=2,frameon=False,fontsize=7.0,columnspacing=1.2)
    fig.subplots_adjust(left=.12,right=.98,top=.835,bottom=.075,hspace=.29,wspace=.32)
    save(fig,'calibrated_global_mass_comparison')

def threshold_comparison(table,summary,fits):
    fig,axs=plt.subplots(3,2,figsize=(7.25,6.05))
    for i,scope in enumerate(SCOPES):
        q=table[table.scope==scope].sort_values('alpha');s=summary.loc[scope];fit=fits[scope]
        a=q.alpha.to_numpy();gp=q.B_p.to_numpy();lo=q.B_p95_low.to_numpy();hi=q.B_p95_high.to_numpy()
        for ax in axs[i]:
            ax.axvspan(.005,.05,color=GOLD,alpha=.09,lw=0,zorder=-5)
            ax.set_xscale('log');ax.set_xlim(1/1025,.101);ax.grid(axis='y',alpha=.16)
            ax.axvline(s.local_A_p,color='.25',ls='--',lw=.7,zorder=5)
        axs[i,0].step(a,gp,where='post',color=RED,lw=1.3)
        axs[i,0].fill_between(a,lo,hi,step='post',color=RED,alpha=.13,lw=0)
        axs[i,0].plot(a,q.sidak_fitted_p,color=GOLD,ls='-.',lw=1.2)
        axs[i,0].plot(a,q.sidak_grid_p,color=PURPLE,ls=':',lw=1.2)
        axs[i,0].set(yscale='log',ylim=(max(.003,float(gp.min())*.5),1.12),ylabel=LABELS[scope]+'\nG(alpha)')
        n=q.effective_N.to_numpy();nl=q.effective_N95_low.to_numpy();nh=q.effective_N95_high.to_numpy()
        axs[i,1].plot(a,n,color=RED,lw=1.2)
        axs[i,1].fill_between(a,nl,nh,where=np.isfinite(nl)&np.isfinite(nh),color=RED,alpha=.13,lw=0)
        axs[i,1].axhline(fit['N_eff'],color=GOLD,ls='-.',lw=1.2)
        axs[i,1].axhline(fit['grid_points'],color=PURPLE,ls=':',lw=1.2)
        finite=nh[np.isfinite(nh)];upper=max(fit['grid_points']*1.08,float(finite.max())*1.05)
        axs[i,1].set(ylim=(0,upper),ylabel='Effective trial count')
        if i==0:
            axs[i,0].set_title('Direct B calibration and frozen references',loc='left',fontsize=8)
            axs[i,1].set_title('Descriptive re-expression of B; not another fit',loc='left',fontsize=7.9)
        if i==2:
            for ax in axs[i]:ax.set_xlabel('Local-rank threshold alpha')
    handles=[Line2D([],[],color=RED,lw=1.4,label='Independent B; pointwise 95% interval'),
             Line2D([],[],color=GOLD,ls='-.',lw=1.4,label='Constant effective count learned from A'),
             Line2D([],[],color=PURPLE,ls=':',lw=1.4,label='Independent mass-grid reference'),
             Line2D([],[],color='.25',ls='--',lw=.8,label='Observed smallest local rank')]
    fig.text(.5,.993,'Test the frozen approximation across local thresholds',ha='center',va='top',fontsize=10.5)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.96),ncol=2,frameon=False,fontsize=7.0,columnspacing=1.4)
    fig.subplots_adjust(left=.115,right=.98,top=.835,bottom=.075,hspace=.29,wspace=.34)
    save(fig,'calibrated_threshold_comparison')

def main():
    curves=pd.read_csv(R/'calibrated_curves.csv',dtype={'scope':str})
    summary=pd.read_csv(R/'calibrated_summary.csv',dtype={'scope':str}).set_index('scope')
    thresholds=pd.read_csv(R/'threshold_comparison.csv',dtype={'scope':str})
    fits={r['scope']:r for r in json.loads((R/'sidak_fit.json').read_text())['fits']}
    mass_comparison(curves,summary);threshold_comparison(thresholds,summary,fits)
    print('Saved two calibrated comparison figure pairs; no fits or numerical tables changed.')

if __name__=='__main__':main()
