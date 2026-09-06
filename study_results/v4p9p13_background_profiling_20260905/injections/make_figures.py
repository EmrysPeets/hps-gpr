#!/usr/bin/env python3
"""Compact paired extraction, exclusion and local-calibration figures."""
from pathlib import Path
import json
import os
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT=HERE/'derived';FIG=HERE/'figures'
LANES=('known_background','gp_uncertainty','retrained_sidebands')
TITLES=('Known-background control','Conditional GP uncertainty','GP retrained on sidebands')
MASSES=(65,71,78,100,182,231)
COLORS={'fixed':'#b2182b','profiled':'#2166ac'}
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':12,'axes.titlesize':13,'axes.labelsize':12,
    'axes.grid':True,'grid.alpha':.17,'grid.linewidth':.6,'axes.spines.right':False,
    'axes.spines.top':False,'pdf.fonttype':42,'legend.frameon':False})


def save(fig,name):
    # Embedded vector glyphs avoid a Type-42 subset-cache rendering defect in
    # Poppler when several STIX figures are included in the same TeX document.
    with plt.rc_context({'pdf.fonttype':3}):
        fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'),dpi=220);plt.close(fig)


def frame_axes(title,rows=2):
    fig,axes=plt.subplots(rows,3,figsize=(11.8,6.8 if rows==2 else 4.2),sharex=True,squeeze=False)
    fig.suptitle(title,y=.99,fontsize=15,fontweight='normal')
    handles=[Line2D([],[],color=COLORS[k],marker='o',ms=4,lw=1.5,label=v)
             for k,v in [('fixed','Fixed GP mean'),('profiled','Profiled GP background')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2)
    for j in range(3):
        axes[0,j].set_title(TITLES[j])
        for a in axes[:,j]:a.set_xticks(range(6),MASSES);a.set_xlim(-.35,5.35)
        axes[-1,j].set_xlabel('Mass hypothesis (MeV)')
    fig.subplots_adjust(left=.085,right=.985,top=.79,bottom=.13,wspace=.23,hspace=.17)
    return fig,axes


def background(d):
    fig,axes=frame_axes('2021 10%: background-only extraction checks')
    for j,lane in enumerate(LANES):
        for method in COLORS:
            q=d[(d.ensemble==lane)&(d.method==method)&(d.strength_sigma==0)].set_index('mass_MeV').loc[list(MASSES)]
            x=np.arange(6)+(-.06 if method=='fixed' else .06)
            axes[0,j].errorbar(x,q.pull_mean,yerr=q.pull_std/np.sqrt(q.n),color=COLORS[method],marker='o',ms=4,lw=1.3,capsize=2)
            axes[1,j].errorbar(x,q.pull_std,yerr=q.pull_std/np.sqrt(2*(q.n-1)),color=COLORS[method],marker='o',ms=4,lw=1.3,capsize=2)
        axes[0,j].axhline(0,color='.4',lw=.8);axes[1,j].axhline(1,color='.4',ls='--',lw=.8)
        axes[0,j].set_ylim(-2,1);axes[1,j].set_ylim(.5,3.)
    axes[0,0].set_ylabel('Mean pull');axes[1,0].set_ylabel('Pull standard deviation')
    fig.text(.5,.02,'500 paired toys per point. Pull = (fitted - injected yield) / fitted error. Error bars show approximate Monte Carlo standard errors.',ha='center',fontsize=10)
    save(fig,'background_closure')


def injected_bias(d):
    fig,axes=frame_axes('2021 10%: mean extraction bias with injected signal')
    for i,strength in enumerate((2,5)):
        for j,lane in enumerate(LANES):
            for method in COLORS:
                q=d[(d.ensemble==lane)&(d.method==method)&(d.strength_sigma==strength)].set_index('mass_MeV').loc[list(MASSES)]
                axes[i,j].errorbar(np.arange(6)+(-.06 if method=='fixed' else .06),q.pull_mean,
                    yerr=q.pull_std/np.sqrt(q.n),color=COLORS[method],marker='o',ms=4,lw=1.3,capsize=2)
            axes[i,j].axhline(0,color='.4',lw=.8);axes[i,j].set_ylim(-2.7,.55)
        axes[i,0].set_ylabel(f'Mean pull: {strength}' + r'$\,\sigma_{\rm prof}$ injection')
    fig.text(.5,.02,'Same physical injected yield for both methods; strengths use the reference profiled Fisher error. These are conditional truth tests.',ha='center',fontsize=10)
    save(fig,'injected_bias')


def exclusion(d):
    fig,axes=frame_axes('2021 10%: exclusion of the true injected yield')
    for i,strength in enumerate((2,5)):
        for j,lane in enumerate(LANES):
            for method in COLORS:
                q=d[(d.ensemble==lane)&(d.method==method)&(d.strength_sigma==strength)].set_index('mass_MeV').loc[list(MASSES)]
                axes[i,j].errorbar(np.arange(6)+(-.06 if method=='fixed' else .06),100*q.exclusion_fraction,
                    yerr=100*np.array([q.exclusion_fraction-q.exclusion_low,q.exclusion_high-q.exclusion_fraction]),
                    color=COLORS[method],marker='o',ms=4,lw=1.3,capsize=2)
            axes[i,j].axhline(10,color='.3',ls='--',lw=1);axes[i,j].set_ylim(0,80)
        axes[i,0].set_ylabel(f'Excluded (%) at {strength}' + r'$\,\sigma_{\rm prof}$')
    fig.text(.5,.02,r'True yield excluded when $CL_s(A_{\rm true})<0.1$. 500 toys per point; exact 95% binomial intervals. Dashed line: 10%.',ha='center',fontsize=10)
    save(fig,'injected_yield_exclusion')


def calibration(c):
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.8),sharey=True)
    options=[('raw','#777777','Raw fixed r'),('variance_scaled','#27856a',r'$r/\kappa(m)$'),
             ('split_calibrated','#a34c87','Center and scale from 100 training toys')]
    for j,lane in enumerate(LANES):
        for k,(key,col,label) in enumerate(options):
            q=c[(c.ensemble==lane)&(c.correction==key)].set_index('mass_MeV').loc[list(MASSES)]
            axes[j].errorbar(np.arange(6)+(k-1)*.14,100*q.false_positive_fraction,
                yerr=100*np.array([q.false_positive_fraction-q.low,q.high-q.false_positive_fraction]),
                color=col,marker='o',ms=3,lw=1.2,capsize=2,label=label)
        axes[j].axhline(5,color='.3',ls='--',lw=1);axes[j].set(ylim=(0,38),title=TITLES[j],xlabel='Mass hypothesis (MeV)')
        axes[j].set_xticks(range(6),MASSES)
    axes[0].set_ylabel(r'False positives at nominal local $p_0<0.05$ (%)')
    fig.suptitle('A local significance scale is conditional on the generating model',y=.99,fontsize=14,fontweight='normal')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,fontsize=10)
    fig.subplots_adjust(left=.085,right=.985,top=.75,bottom=.22,wspace=.2)
    fig.text(.5,.04,'All points use 400 held-out B-only toys; error bars are exact 95% binomial intervals. No observed p-value has been rescaled.',ha='center',fontsize=10)
    save(fig,'local_significance_calibration')


def fisher(d):
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
    axes[0].plot(d.mass_MeV,d.kappa,color='#27856a',lw=1.7)
    axes[0].set_ylabel(r'Omitted-variance factor $\kappa(m)$')
    axes[1].plot(d.mass_MeV,d.corrected_fixed_over_profiled,color='#185781',lw=1.7)
    axes[1].axhline(1,color='.4',ls='--',lw=.9)
    axes[1].set_ylabel(r'$\sigma_{\rm fixed,\ corrected}/\sigma_{\rm profiled}$')
    for a in axes:a.set(xlim=(50,250),xlabel='Mass hypothesis (MeV)')
    fig.suptitle('Propagating GP uncertainty removes the apparent precision gain',fontsize=14,fontweight='normal',y=.98)
    fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.19,wspace=.3)
    fig.text(.5,.03,'High-count linearized calculation using the frozen 2021 covariance; this is a model-based variance comparison.',ha='center',fontsize=10)
    save(fig,'fisher_variance_correction')


def main():
    FIG.mkdir(exist_ok=True)
    d=pd.read_csv(OUT/'extraction_summary.csv');cal=pd.read_csv(OUT/'local_calibration_holdout.csv')
    background(d);injected_bias(d);exclusion(d);calibration(cal);fisher(pd.read_csv(OUT/'fisher_variance_scan.csv'))
    lines=[r'\begin{table}[htbp]\centering\small',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lrrrrrr}\toprule',
        r'Truth and mass & \multicolumn{2}{c}{B-only pull width} & \multicolumn{2}{c}{Mean pull at $5\sigma_{\rm ref}$} & \multicolumn{2}{c}{True yield excluded} \\',
        r' & Fixed & Profiled & Fixed & Profiled & Fixed & Profiled \\ \midrule']
    for lane,title in zip(LANES,TITLES):
        lines.append(r'\multicolumn{7}{l}{\emph{'+title+r'}} \\')
        for mass in MASSES:
            b=d[(d.ensemble==lane)&(d.mass_MeV==mass)&(d.strength_sigma==0)].set_index('method')
            s=d[(d.ensemble==lane)&(d.mass_MeV==mass)&(d.strength_sigma==5)].set_index('method')
            vals=[b.loc[k,'pull_std'] for k in COLORS]+[s.loc[k,'pull_mean'] for k in COLORS]
            lines.append(f'{mass} MeV & '+' & '.join(f'{v:.2f}' for v in vals)+' & '+
                ' & '.join(f"{100*s.loc[k,'exclusion_fraction']:.1f}\\%" for k in COLORS)+r' \\')
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Conditional 2021 toy diagnostics. Each cell uses 500 spectra; the fixed and profiled methods are paired on the same spectra. The stronger injection is five times the reference profiled Fisher error, with the same yield for both methods. Exclusion means $CL_s(A_{\rm true})<0.1$. Binomial intervals and both injection strengths are shown in the figures and retained in the CSV.}',
        r'\label{tab:toy-summary}\end{table}']
    (OUT/'summary_table.tex').write_text('\n'.join(lines)+'\n')
    print('Created five figures and summary_table.tex')


if __name__=='__main__':main()
