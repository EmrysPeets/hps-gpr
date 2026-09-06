#!/usr/bin/env python3
"""Create standalone figures and a compact report without touching parent notes."""
from pathlib import Path
import hashlib
import json
import os
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PARENT=REPO/'study_results/background_profile_comparison_20260905'
OUT=PARENT/'derived'
FIG=HERE/'figures'
BLUE='#2166ac';ORANGE='#d26924';GREY='#7b7b7b';GREEN='#27856a'
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':12.5,'axes.labelsize':13,'axes.titlesize':14,
    'xtick.labelsize':12,'ytick.labelsize':12,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'grid.linewidth':.6,
    'pdf.fonttype':42,'savefig.facecolor':'white'})


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'),dpi=220)
    plt.close(fig)


def branching(m):
    m=np.asarray(m,float);out=np.ones_like(m);sel=m>2*105.6583745
    r=(105.6583745/m[sel])**2
    out[sel]+=np.sqrt(1-4*r)*(1+2*r)
    return out


def limits(d):
    fig,axs=plt.subplots(2,2,figsize=(11.8,7.0),sharex='col',
        gridspec_kw={'height_ratios':[2.1,1]})
    labels=['Released Gaussian profile','Direct log-GP profile','Fixed GP mean']
    specs=[('current',BLUE,'-',2.2),('log_gp',ORANGE,'--',1.8),('fixed','#b2182b','-.',1.4)]
    for j,(lo,hi,title) in enumerate(((50,250,'Full 2021 search range'),(50,100,'Low-mass detail'))):
        a,r=axs[:,j];sel=d.mass_MeV.between(lo,hi);f=d[sel]
        for (name,col,ls,lw),label in zip(specs,labels):
            a.plot(f.mass_MeV,f['eps2_'+name]*branching(f.mass_MeV),ls,color=col,lw=lw,label=label)
            if name!='current':r.plot(f.mass_MeV,f['eps2_'+name]/f.eps2_current,ls,color=col,lw=lw)
        a.set(yscale='log',title=title,xlim=(lo,hi))
        r.axhline(1,color=BLUE,lw=1)
        r.set(ylim=(0,1.5),xlabel=r'$m_{A\prime}$ (MeV)')
        r.set_yticks([0,.5,1,1.5])
    axs[0,0].set_ylabel(r'Observed 90% CL$_s$ upper limit on $\epsilon^2$')
    axs[1,0].set_ylabel('Ratio to released limit')
    fig.suptitle('2021 10%: background profiling comparison',fontsize=16,fontweight='normal',y=.98)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.text(.5,.025,'Same data, kernels, masks, resolution and yield conversion. Conditional asymptotic observed comparison.',ha='center',fontsize=10,color='.3')
    fig.subplots_adjust(left=.09,right=.98,bottom=.13,top=.79,hspace=.13,wspace=.25)
    save(fig,'observed_limits_2021_comparison')


def decomposition(d):
    fig,(a,b)=plt.subplots(1,2,figsize=(11.8,4.0))
    a.plot(d.mass_MeV,100*(d.eps2_log_gp/d.eps2_current-1),color=ORANGE,lw=1.5,label='Log-GP / released')
    a.plot(d.mass_MeV,100*(d.eps2_gaussian_control/d.eps2_current-1),color=BLUE,ls='--',lw=1.2,label='Stable Gaussian / released')
    b.plot(d.mass_MeV,100*(d.eps2_log_gp/d.eps2_gaussian_control-1),color=GREEN,lw=1.5)
    for ax in (a,b):ax.axhline(0,color='.5',lw=.7);ax.set(xlim=(50,250),xlabel='Mass hypothesis (MeV)',ylabel='Limit change (%)')
    a.set_title('Total change and numerical control');b.set_title('Background-model change with the same solver')
    fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.71,wspace=.27)
    save(fig,'profile_model_and_numerics')


def fits(d,new,old,masses,name):
    fig,axs=plt.subplots(2,2,figsize=(11.8,6.8),sharex='col',gridspec_kw={'height_ratios':[1,1.1]})
    for j,mass in enumerate(masses):
        p=new[new.mass_MeV==mass].sort_values('bin_center_MeV').copy()
        o=old[(old.mass_hypothesis_MeV==mass)&old.group.isin(['2021','2021_deficit'])].copy()
        lookup={round(float(t.bin_center_MeV),7):t for t in o.itertuples()}
        original=[lookup[round(float(v),7)] for v in p.bin_center_MeV]
        for k in ('profiled_background','signal','total'):
            p[k]=[getattr(t,k) for t in original]
        assert np.array_equal(p.observed,[t.observed for t in original])
        f=p[p.in_fit];outside=p[~p.in_fit]
        row=d.set_index('mass_MeV').loc[mass]
        sigma=float(row.sigma_MeV);a,r=axs[:,j]
        # Curves stop at bin edges of the actual target window. No splicing.
        lo=float(f.bin_center_MeV.iloc[0]-f.bin_width_MeV.iloc[0]/2)
        hi=float(f.bin_center_MeV.iloc[-1]+f.bin_width_MeV.iloc[-1]/2)
        for ax in (a,r):
            ax.axvspan(lo,hi,color=BLUE,alpha=.045,zorder=0)
            ax.axvline(lo,color='.65',ls=':',lw=.8);ax.axvline(hi,color='.65',ls=':',lw=.8)
        a.plot(p.bin_center_MeV,p.gp_mean/p.bin_width_MeV,color=GREY,lw=1.1)
        for frame,col in ((outside,'.65'),(f,'black')):
            a.errorbar(frame.bin_center_MeV,frame.observed/frame.bin_width_MeV,
                yerr=np.sqrt(frame.observed)/frame.bin_width_MeV,fmt='o',ms=3,color=col,elinewidth=.7)
            r.errorbar(frame.bin_center_MeV,(frame.observed-frame.gp_mean)/frame.bin_width_MeV,
                yerr=np.sqrt(frame.observed)/frame.bin_width_MeV,fmt='o',ms=3,color=col,elinewidth=.7)
        a.plot(f.bin_center_MeV,f.profiled_background/f.bin_width_MeV,color=BLUE,ls=':',lw=1.6)
        a.plot(f.bin_center_MeV,f.total/f.bin_width_MeV,color=BLUE,lw=2)
        a.plot(f.bin_center_MeV,f.total_log_gp/f.bin_width_MeV,color=ORANGE,ls='--',lw=1.6)
        r.fill_between(f.bin_center_MeV,-f.gp_sd/f.bin_width_MeV,f.gp_sd/f.bin_width_MeV,color='.75',alpha=.35,lw=0)
        r.axhline(0,color=GREY,lw=.9)
        r.plot(f.bin_center_MeV,(f.profiled_background-f.gp_mean)/f.bin_width_MeV,color=BLUE,ls=':',lw=1.6)
        r.plot(f.bin_center_MeV,(f.total-f.gp_mean)/f.bin_width_MeV,color=BLUE,lw=2)
        r.plot(f.bin_center_MeV,(f.total_log_gp-f.gp_mean)/f.bin_width_MeV,color=ORANGE,ls='--',lw=1.6)
        a.set_title(f'{mass} MeV'+('  (signed deficit fit)' if mass==71 else ''))
        a.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
        r.set(xlabel='Invariant mass (MeV)',xlim=(mass-4*sigma,mass+4*sigma))
        if j==0:
            a.set_ylabel('Events / MeV')
            r.set_ylabel('Data or model - GP mean\n(events / MeV)')
    handles=[Line2D([],[],color='black',marker='o',ls='',ms=4,label='Data (counting errors)'),
             Line2D([],[],color=BLUE,lw=2,label='Released signal + background'),
             Line2D([],[],color=ORANGE,ls='--',lw=1.8,label='Log-GP signal + background'),
             Line2D([],[],color=BLUE,ls=':',lw=1.8,label='Released profiled background'),
             Patch(color='.75',alpha=.35,label='Sideband GP marginal 1-sigma band')]
    fig.suptitle('2021 10%: fits with a consistent residual baseline',y=.99,fontsize=15,fontweight='normal')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False,fontsize=11)
    fig.text(.5,.02,'Profile curves are drawn only in fitted bins; grey data are outside. The GP band is not an independent residual uncertainty.',ha='center',fontsize=10,color='.3')
    fig.subplots_adjust(left=.095,right=.98,bottom=.125,top=.76,hspace=.13,wspace=.25)
    save(fig,name)


if __name__=='__main__':
    d=pd.read_csv(OUT/'observed_limits.csv')
    new=pd.read_csv(OUT/'fit_plot_data.csv')
    old=pd.read_csv(REPO/'study_results/v4p9p12_expanded_snapshot_20260905/derived/selected_fit_plot_data.csv')
    limits(d);decomposition(d)
    fits(d,new,old,(65,71),'fits_65_71')
    fits(d,new,old,(78,182),'fits_78_182')
    sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        OUT/'observed_limits.csv',OUT/'fit_plot_data.csv',
        REPO/'study_results/v4p9p12_expanded_snapshot_20260905/derived/selected_fit_plot_data.csv']}
    (HERE/'figure_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Created four restyled comparison figures from frozen data')
