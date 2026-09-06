#!/usr/bin/env python3
"""Render completed conventional fits; never performs a fit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
VARIANTS = ['baseline','degree_minus','degree_plus','width_minus','width_plus']
LABELS = ['Base','Degree -1','Degree +1','Width -2σ','Width +2σ']
COLORS = dict(data='#222222',total='#176A9B',null='#777777',background='#B37816',signal='#AE3E30')


def main():
    summary = json.loads((HERE/'derived/summary.json').read_text())
    if not summary['passed']:
        raise RuntimeError('Cannot render an accepted comparison from an incomplete fit set')
    rows = pd.read_csv(HERE/'derived/fit_summary.csv',dtype={'dataset':str},float_precision='round_trip')
    folder = HERE/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42,'savefig.dpi':210,
                         'axes.spines.top':False,'axes.spines.right':False})
    for year,masses in [('2015',[51,21]),('2016',[90,117]),('2021',[78,65])]:
        fig,axes = plt.subplots(3,2,figsize=(11.2,9.6),gridspec_kw={'height_ratios':[2.2,1.8,1.45]})
        fig.subplots_adjust(left=.10,right=.98,bottom=.145,top=.85,hspace=.55,wspace=.28)
        for col,mass in enumerate(masses):
            data = rows[(rows.dataset==year)&(rows.mass_MeV==mass)].set_index('variant')
            base = data.loc['baseline']; fid = base.fit_id
            a = np.load(HERE/'derived/points'/(fid+'__baseline.npz'))
            centers = .5*(a['edges_MeV'][:-1]+a['edges_MeV'][1:]); counts = a['counts']
            top,res,variants = axes[:,col]
            top.errorbar(centers,counts,np.sqrt(counts),fmt='.',ms=2.5,color=COLORS['data'],elinewidth=.65,label='Observed native bins')
            for key,label,color,style in [('total_free','Signal + background','total','-'),
                    ('background_free','Background with signal','background',':'),
                    ('background_null','Background-only fit','null','--')]:
                top.plot(centers,a[key],color=COLORS[color],ls=style,lw=1.15,label=label)
            top.set_title(f'{mass} MeV | degree {int(base.degree)}, total width {base.total_width_sigma:g}σ',fontsize=11)
            top.set_ylabel('Counts / native bin');top.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            top.legend(fontsize=7.8,frameon=False,loc='best')
            top.text(.03,.04,f'Full Gaussian yield {base.amplitude_full:,.0f} ± {base.sigma_amplitude_full:,.0f}\n'
                f'r = {base.root:+.3f}; nominal local p₀ = {base.p0_nominal:.3g}\n'
                f'D / dof = {base.deviance:.1f} / {int(base.ndof)}',transform=top.transAxes,
                fontsize=8.1,bbox={'facecolor':'white','edgecolor':'none','alpha':.82})
            residual = counts-a['background_null']
            res.errorbar(centers,residual,np.sqrt(counts),fmt='.',ms=2.5,color=COLORS['data'],elinewidth=.65)
            res.plot(centers,a['total_free']-a['background_null'],color=COLORS['total'],lw=1.3,label='Fitted total − null background')
            res.plot(centers,base.amplitude_full*a['signal_bin_probability'],color=COLORS['signal'],lw=1.15,ls='--',label='Signal component alone')
            res.axhline(0,color='.65',lw=.65)
            displayed=np.r_[residual-np.sqrt(counts),residual+np.sqrt(counts),
                            a['total_free']-a['background_null'],
                            base.amplitude_full*a['signal_bin_probability']]
            lower,upper=displayed.min(),displayed.max();span=upper-lower
            res.set_ylim(lower-.05*span,upper+.42*span)
            res.legend(fontsize=7.7,frameon=False,loc='upper right')
            res.set(xlabel='Invariant mass [MeV]',ylabel='Counts − null background')
            res.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            y = data.loc[VARIANTS,'root'].to_numpy();positions=np.arange(5)
            variants.scatter(positions,y,c=[COLORS['total']]+[COLORS['background']]*4,s=29,zorder=3)
            variants.axhline(0,color='.7',lw=.65)
            variants.axhline(base.gp_root,color=COLORS['null'],ls=':',lw=1,label=f'Original GP root {base.gp_root:+.2f}')
            labels=[f'{label}\nD/dof {data.loc[variant,"deviance_per_dof"]:.2f}'
                    for label,variant in zip(LABELS,VARIANTS)]
            variants.set_xticks(positions,labels,fontsize=7.5)
            variants.set(ylabel='Signed root r',xlim=(-.5,4.5))
            variants.legend(fontsize=7.7,frameon=False,loc='best')
            variants.set_title('All five fixed fits; labels include fit deviance',fontsize=9.5)
            for position,value in zip(positions,y):
                variants.annotate(f'{value:+.2f}',(position,value),xytext=(0,7),
                                  textcoords='offset points',ha='center',fontsize=7.3,
                                  bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':.2})
            variants.margins(y=.30)
            for ax in (top,res):
                ax.axvline(mass,color='.7',lw=.65,ls=':');ax.set_xlim(a['edges_MeV'][0],a['edges_MeV'][-1])
                ax.grid(axis='y',alpha=.16)
            variants.grid(axis='y',alpha=.16)
        exposure='10%' if year=='2021' else 'full'
        fig.suptitle(f'Traditional local fits at GP-selected masses | {year} {exposure}',fontsize=15,y=.98)
        fig.text(.5,.945,'Original observed data | fixed Gaussian mass and resolution | positive exponential-polynomial background',ha='center',fontsize=10)
        fig.text(.10,.035,'Counting bars only; fitted-background subtraction creates correlated residuals. Variant widths are total window widths.\n'
                 'Large roots can arise from inadequate backgrounds: read the deviance and all variants together. No preferred variant selected.\n'
                 'Selected local references and model checks, not independent confirmation or a global significance.',fontsize=8.7)
        for ext in ('pdf','png'):
            fig.savefig(folder/f'traditional_{year}.{ext}',bbox_inches='tight')
        plt.close(fig)
    paths=list(folder.glob('*'))
    (HERE/'figure_inventory.json').write_text(json.dumps({'figures':[str(p.relative_to(HERE)) for p in paths],
          'sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')


if __name__ == '__main__':
    main()
