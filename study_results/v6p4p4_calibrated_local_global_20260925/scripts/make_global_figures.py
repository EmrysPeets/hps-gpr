"""Publication figures from the complete saved scan ledgers; no refitting."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v643-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
B=Path(__file__).resolve().parents[1];F=B/'figures';R=B/'results'
BLUE='#245c91';RED='#ab3d35';GRAY='#747474'
NAMES={'2016':'2016: full data, MC signal, ±2.5u','2021':'2021: 10% data, MC signal, [−4,+3]u','combined':'Combined: 2015 Gaussian + 2016 and 2021 MC'}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':9,'axes.labelsize':9,
    'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    F.mkdir(exist_ok=True);fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=180);plt.close(fig)

def main():
    curves=pd.read_csv(R/'local_global_curves.csv',dtype={'scope':str});summary=pd.read_csv(R/'global_summary.csv',dtype={'scope':str}).set_index('scope')
    fig,axes=plt.subplots(3,2,figsize=(7.2,6.0),gridspec_kw={'width_ratios':[1,1]})
    for row,scope in enumerate(('2016','2021','combined')):
        d=curves[curves.scope==scope].sort_values('mass_MeV');peak=summary.loc[scope,'peak_mass_MeV']
        for prefix,color,style,label in [('asymptotic',GRAY,'--','Local asymptotic reference'),('local',BLUE,'-','Local, calibrated with toys'),('global',RED,'-','Global over the mass grid')]:
            p=d[prefix+'_p' if prefix=='asymptotic' else prefix+'_p_rank'];z=d[prefix+'_Z' if prefix=='asymptotic' else prefix+'_Z_excess']
            axes[row,0].plot(d.mass_MeV,p,color=color,ls=style,lw=1.15,label=label)
            axes[row,1].plot(d.mass_MeV,z,color=color,ls=style,lw=1.15,label=label)
        for j,ax in enumerate(axes[row]):
            ax.set_xlim(d.mass_MeV.min(),d.mass_MeV.max());ax.axvline(peak,color='.68',ls=':',lw=.75)
            ax.grid(axis='y',alpha=.15);ax.set_title(NAMES[scope] if j==0 else f'Largest observed statistic: {int(peak)} MeV',loc='left',fontsize=8)
            if scope=='combined':
                for edge in (100.5,175.5):ax.axvline(edge,color='.8',ls=':',lw=.65,zorder=-3)
            if row==2:ax.set_xlabel('Generated signal mass hypothesis [MeV]')
        axes[row,0].set(yscale='log',ylim=(1e-4,1.35),ylabel='Excess probability p')
        axes[row,1].set(ylim=(-.08,4),ylabel='Excess significance Z [σ]');axes[row,1].yaxis.set_major_locator(MultipleLocator(1))
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.53,.995),ncol=1,frameon=False)
    fig.subplots_adjust(left=.10,right=.985,top=.86,bottom=.08,hspace=.49,wspace=.30);save(fig,'local_and_global_scans')
    maxima=pd.read_csv(R/'toy_maxima.csv',dtype={'scope':str});fig,axes=plt.subplots(3,1,figsize=(7.2,5.7),sharex=True)
    for ax,scope in zip(axes,('2016','2021','combined')):
        a=np.sort(maxima[maxima.scope==scope].max_root.to_numpy());s=summary.loc[scope];observed=float(s.observed_Z_asymptotic)
        x=np.unique(np.r_[0,a,observed,a[-1]+.05]);k=len(a)-np.searchsorted(a,x,side='left');p=(k+1)/(len(a)+1)
        ax.step(x,p,where='pre',color=BLUE,lw=1.3);ax.axvline(observed,color=RED,lw=1.2,ls='--')
        ax.plot(observed,s.global_p_rank,'o',color=RED,ms=4,zorder=4)
        ax.text(.02,.06,f'{int(s.global_exceedances)} / 1,024 complete scans exceed the observation; p = {s.global_p_rank:.3g}',transform=ax.transAxes,fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
        ax.set(yscale='log',ylim=(.0008,1.3),ylabel='Global tail probability')
        ax.set_title(NAMES[scope],loc='left',fontsize=9);ax.grid(axis='y',alpha=.15)
    axes[-1].set_xlabel('Threshold on the largest positive likelihood root in a scan')
    axes[-1].set_xlim(0,max(5,float(maxima.max_root.max())+.1))
    fig.subplots_adjust(left=.115,right=.98,top=.96,bottom=.085,hspace=.36);save(fig,'scan_maximum_tails')

if __name__=='__main__':main()
