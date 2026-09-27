#!/usr/bin/env python3
"""Publication-style displays from saved v6.3.8 fits; no refitting or new inference."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hps-v638-figures')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

B = Path(__file__).resolve().parents[1]
R = B/'results'; F = B/'figures'
BLUE = '#245c91'; RED = '#ab3d35'; GOLD = '#a98224'; GREEN = '#39846b'; PURPLE = '#865697'
POLICIES = ['gaussian_baseline', 'gaussian_starter', 'morph_starter']
NAMES = {'gaussian_baseline': 'Shifted Gaussian: original window',
         'gaussian_starter': 'Shifted Gaussian: starter window',
         'morph_starter': 'Neighboring signal-MC template'}
COLORS = {'gaussian_baseline': '#777777', 'gaussian_starter': GOLD, 'morph_starter': BLUE}
STYLES = {'gaussian_baseline': '--', 'gaussian_starter': ':', 'morph_starter': '-'}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                     'axes.titlesize':9,'legend.fontsize':7.2,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def save(fig, name):
    F.mkdir(exist_ok=True)
    fig.savefig(F/(name+'.pdf'))
    fig.savefig(F/(name+'.png'), dpi=180)
    plt.close(fig)


def format_p(value):
    return f'{value:.3g}'


def scan_figure(scan, selected):
    """Canonical columns: mass, policy, upper_limit, local_p; selected: mass, region."""
    fig, axes = plt.subplots(2,1,figsize=(7.2,5.6),sharex=True,
                             gridspec_kw={'height_ratios':[1,1]})
    for policy in POLICIES:
        q=scan[scan.policy==policy].sort_values('mass')
        if q.empty: raise ValueError(f'Missing policy {policy}')
        axes[0].plot(q.mass,q.upper_limit,color=COLORS[policy],ls=STYLES[policy],lw=1.25,label=NAMES[policy])
        axes[1].plot(q.mass,q.local_p,color=COLORS[policy],ls=STYLES[policy],lw=1.25)
    for ax in axes:
        ax.grid(axis='y',alpha=.15)
        ax.set_xlim(60,240)
    for row in selected.itertuples():
        for ax in axes:ax.axvline(row.mass,color='.62',ls=':',lw=.6,zorder=-2)
        q=scan[(scan.policy=='morph_starter')&np.isclose(scan.mass,row.mass)].iloc[0]
        axes[0].plot(row.mass,q.upper_limit,'o' if row.region!='deficit' else 'v',color=BLUE,ms=3.5)
        axes[1].plot(row.mass,q.local_p,'o' if row.region!='deficit' else 'v',color=BLUE,ms=3.5)
    axes[0].set(ylabel='90% profile-CLs limit\n[full-template yield]',yscale='log')
    pmin=float(scan.local_p.min())
    if pmin<=0:raise ValueError('Nonpositive local p cannot be plotted on a log axis')
    axes[1].set(ylabel='Local excess p\n[asymptotic reference]',yscale='log',
                ylim=(max(1e-16,pmin*.55),.9),xlabel='Signal mass hypothesis [MeV]')
    fig.text(.5,.98,'2021 10% observed spectrum: compare the assumed signal shapes',ha='center',va='top',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.934),
               ncol=2,frameon=False,fontsize=7.4,columnspacing=1.1)
    fig.subplots_adjust(left=.145,right=.98,top=.82,bottom=.105,hspace=.09)
    save(fig,'v638_observed_scan')


def region_figure(meta, data, toy=None):
    """Canonical data: x, width, counts, gp_mean, gp_sd, b_null, b_signal, signal, total."""
    name=meta['region'];mass=float(meta['mass']);c=float(meta['center']);s=float(meta['width'])
    x=np.asarray(data['x']);w=np.asarray(data['width']);n=np.asarray(data['counts'])
    b=np.asarray(data['gp_mean']);sd=np.asarray(data['gp_sd'])
    b0=np.asarray(data['b_null']);bs=np.asarray(data['b_signal']);signal=np.asarray(data['signal']);total=np.asarray(data['total'])
    assert all(len(z)==len(x) for z in [w,n,b,sd,b0,bs,signal,total])
    assert np.all(w>0) and np.all(n>=0) and np.all(sd>=0)
    assert np.allclose(total,bs+signal,rtol=1e-9,atol=1e-5)
    error=np.sqrt(n)/w
    fig,axes=plt.subplots(3,1,figsize=(7.2,6.4),
                          gridspec_kw={'height_ratios':[1.1,.95,1.1]})
    top,bottom,context=axes
    bottom.sharex(top)
    for ax in axes:ax.grid(axis='y',alpha=.12)
    top.fill_between(x,(b-sd)/w,(b+sd)/w,color=BLUE,alpha=.16,lw=0)
    top.plot(x,b/w,color=BLUE,lw=1.2)
    top.plot(x,b0/w,color=GOLD,ls='--',lw=1.3)
    top.plot(x,bs/w,color=GREEN,ls=':',lw=1.35)
    top.plot(x,total/w,color=RED,lw=1.45)
    top.errorbar(x,n/w,yerr=error,fmt='o',ms=2.5,color='#222222',elinewidth=.6,capsize=0,zorder=5)
    bottom.fill_between(x,-sd/w,sd/w,color=BLUE,alpha=.16,lw=0)
    bottom.axhline(0,color=BLUE,lw=.9)
    bottom.plot(x,(b0-b)/w,color=GOLD,ls='--',lw=1.3)
    bottom.plot(x,(bs-b)/w,color=GREEN,ls=':',lw=1.35)
    bottom.plot(x,signal/w,color=PURPLE,ls='-.',lw=1.3)
    bottom.plot(x,(total-b)/w,color=RED,lw=1.45)
    bottom.errorbar(x,(n-b)/w,yerr=error,fmt='o',ms=2.5,color='#222222',elinewidth=.6,capsize=0,zorder=5)
    top.set_ylabel('Counts / MeV')
    bottom.set_ylabel('Counts − GP mean\n[per MeV]')
    top.tick_params(labelbottom=False)
    bottom.tick_params(labelbottom=True)
    top.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
    bottom.ticklabel_format(axis='y',style='plain',useOffset=False)
    top.set_xlim(x[0]-w[0]*.5,x[-1]+w[-1]*.5)
    handles=[Line2D([],[],color='#222222',marker='o',ms=3,lw=0,label='Observed counts'),
             Patch(facecolor=BLUE,edgecolor=BLUE,alpha=.25,label='GP mean and ±1 SD constraint'),
             Line2D([],[],color=GOLD,ls='--',lw=1.3,label='Background-only fit'),
             Line2D([],[],color=GREEN,ls=':',lw=1.3,label='Background in signed S+B fit'),
             Line2D([],[],color=RED,lw=1.45,label='Signed signal + background'),
             Line2D([],[],color=PURPLE,ls='-.',lw=1.3,label='Signed signal (middle panel)')]
    handles += [Patch(facecolor='#b5b5b5',alpha=.22,label='Blind region: excluded from GP training'),
                Line2D([],[],color='black',ls='--',lw=1,label='Search mass hypothesis'),
                Line2D([],[],color=RED,ls='--',lw=1,label='Signal core center')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.873),ncol=3,
               frameon=False,fontsize=7.1,columnspacing=1.25,handlelength=2.2)
    feature=('Selected deficit' if name=='deficit' else 'Selected excess '+name.rsplit('_',1)[-1])
    fig.text(.5,.993,f'2021 10%: {feature.lower()} at a {mass:g} MeV signal hypothesis',ha='center',va='top',fontsize=11)
    model='v16 neighboring signal-MC template'
    fig.text(.5,.966,f'{model} | c = {c:.3f} MeV, s = {s:.3f} MeV',
             ha='center',va='top',fontsize=7.9)
    root=float(meta['signed_root']); ahat=float(meta['Ahat']);err=float(meta['sigma_A'])
    stats=f'Signed full-selected yield = {ahat:,.0f} ± {err:,.0f}'
    if root>=0:stats+=f'; local asymptotic Z = {root:.2f} σ, p = {format_p(float(meta["local_p"]))}'
    else:stats+=f'; signed likelihood root r = {root:.2f}; excess q₀ = 0, p₀ = 0.5'
    fig.text(.5,.939,stats,ha='center',va='top',fontsize=8.1)
    if toy is not None:
        label='deficit' if name=='deficit' else 'excess'
        toytext=f'GP background-only toys: local {label} rank p = {format_p(float(toy["rank_p"]))} ({int(toy["k"])} / {int(toy["n"])} more extreme toys)'
        if 'lo' in toy:toytext+=f'; count 95% interval [{format_p(float(toy["lo"]))}, {format_p(float(toy["hi"]))}]'
    else:toytext='Fixed-mass illustration selected from the observed scan; no global significance is assigned.'
    fig.text(.5,.911,toytext,ha='center',va='top',fontsize=7.0)
    fx=np.asarray(data['full_x']);fw=np.asarray(data['full_width']);fn=np.asarray(data['full_counts'])
    fb=np.asarray(data['full_gp']);fsd=np.asarray(data['full_sd'])
    context.axvspan(meta['fit_low'],meta['fit_high'],color='#b5b5b5',alpha=.22,zorder=-4)
    denom=np.sqrt(fb+fsd**2)
    fitden=np.sqrt(b+sd**2)
    context.fill_between(fx,-fsd/denom,fsd/denom,color=BLUE,alpha=.18,lw=0)
    context.axhline(0,color=BLUE,lw=1.2)
    context.errorbar(fx,(fn-fb)/denom,yerr=np.sqrt(fn)/denom,fmt='o',ms=2,color='.25',elinewidth=.5,zorder=3)
    context.plot(x,(bs-b)/fitden,color=GREEN,ls=':',lw=1.4,zorder=4)
    context.plot(x,(total-b)/fitden,color=RED,lw=1.5,zorder=4)
    context.axvline(mass,color='black',ls='--',lw=1,zorder=5)
    context.axvline(c,color=RED,ls='--',lw=1,zorder=5)
    context.set(xlim=(max(float(fx.min()),c-6*s),min(float(fx.max()),c+5*s)),
                xlabel='Reconstructed pair mass [MeV]',ylabel='Standardized residual')
    visible=(fx>=c-6*s)&(fx<=c+5*s)
    lows=np.r_[-fsd[visible]/denom[visible],((fn-fb-np.sqrt(fn))/denom)[visible]]
    highs=np.r_[fsd[visible]/denom[visible],((fn-fb+np.sqrt(fn))/denom)[visible]]
    span=float(highs.max()-lows.min());context.set_ylim(lows.min()-.07*span,highs.max()+.07*span)
    context.ticklabel_format(axis='y',style='plain',useOffset=False)
    for ax,label in zip(axes,['(a) Spectrum in the fit interval','(b) Residuals relative to the GP mean','(c) Standardized residuals and blind window']):
        ax.set_title(label,loc='left',fontsize=8,pad=5)
    fig.subplots_adjust(left=.135,right=.98,top=.72,bottom=.085,hspace=.45)
    save(fig,'v638_'+name)


def main():
    original=pd.read_csv(R/'observed_scan.csv')
    scan=original.rename(columns={'mass_MeV':'mass','A90':'upper_limit','p0_fixed_mass':'local_p'})
    if not np.isfinite(scan[['mass','upper_limit','local_p']].to_numpy()).all():
        raise ValueError('Scan contains a nonfinite plotted quantity')
    selected_raw=json.loads((R/'selected_regions.json').read_text())['regions']
    selected=pd.DataFrame([dict(mass=r['mass_MeV'],region=r['region'].replace('excess_','peak_')) for r in selected_raw])
    scan_figure(scan,selected)
    toyfile=R/'selected_local_calibration.csv'
    toys=pd.read_csv(toyfile) if toyfile.exists() else None
    captions={'v638_observed_scan': 'How to read the figure. The upper panel compares observed 90% profile-CLs limits under three assumed signal models. Each yield refers to its full signal distribution, so the Gaussian curves are not limits on the same MC-shaped signal as the neighboring-template curve. The lower panel shows fixed-mass asymptotic excess p-values; these are not globally calibrated probabilities. Points identify the three nonoverlapping selected excess windows and a deficit window, whose locations were chosen using these data. Lines have no uncertainty bands; fixed signal-template and GP inputs do not propagate model or selection uncertainty.'}
    summaries=[]
    for selected_row in selected_raw:
        mass=float(selected_row['mass_MeV']);region=selected_row['region']
        q=original[(original.policy=='morph_starter')&np.isclose(original.mass_MeV,mass)]
        assert len(q)==1
        q=q.iloc[0]
        path=R/'selected_fits'/f'{region}_m{mass:03.0f}_morph_starter.npz'
        with np.load(path,allow_pickle=False) as z:
            mask=np.asarray(z['fit_mask'],bool)
            data=dict(x=z['x_GeV'][mask]*1000,width=np.diff(z['edges_GeV'])[mask]*1000,
                      counts=z['fit_counts'],gp_mean=z['fit_prefit_mean'],
                      gp_sd=np.sqrt(np.maximum(np.diag(z['fit_prefit_covariance']),0)),
                      b_null=z['profiled_background_only'],b_signal=z['profiled_background_signed'],
                      signal=z['profiled_signed_signal'],total=z['profiled_signed_total'],
                      full_x=z['x_GeV']*1000,full_width=np.diff(z['edges_GeV'])*1000,full_counts=z['counts'],
                      full_gp=z['prefit_GP_mean'],full_sd=np.sqrt(np.maximum(np.diag(z['prefit_GP_covariance']),0)))
        meta=dict(region=region.replace('excess_','peak_'),mass=mass,center=float(q.core_center_MeV),
                  width=float(q.core_width_MeV),signed_root=float(q.signed_r),Ahat=float(q.Ahat),
                  sigma_A=float(q.sigma_A),local_p=float(q.p0_fixed_mass),
                  fit_low=float(q.fit_low_MeV),fit_high=float(q.fit_high_MeV))
        toy=None
        if toys is not None:
            t=toys[(toys.policy=='morph_starter')&np.isclose(toys.mass_MeV,mass)]
            if len(t)!=1:raise ValueError(f'Ambiguous local calibration at {mass}')
            t=t.iloc[0]
            toy=dict(rank_p=float(t.p_rank),k=int(t.tail_count),n=int(t.null_toys),
                     lo=float(t.tail_probability95_low),hi=float(t.tail_probability95_high))
        region_figure(meta,data,toy)
        key='v638_'+meta['region']
        scope=('This hypothesis lies in the acceptance-transition interpolation between 60 and 80 MeV, without independent intermediate-mass signal MC. ' if mass<80 else '')
        direction=('The fitted signal is negative: it describes a deficit, and the excess statistic is q0=0. The background-only toy probability uses the lower tail of the signed likelihood root. ' if region=='deficit' else 'The fitted positive signal is an excess; the displayed probabilities are fixed-mass local references. ')
        captions[key]=('How to read the figure. Black points show observed counts divided by the bin width, with sqrt(N) count-only errors. '
                       'The blue curve is the GP sideband prediction, and its band is one marginal standard deviation of the background constraint; it is not a post-fit confidence band. '
                       'Gold is the profiled background-only fit. Green is the background component of the signed signal-plus-background fit, and red is its total. '
                       'The middle panel subtracts the GP mean; purple isolates the signed signal. The bottom panel divides data-minus-GP and model-minus-GP by sqrt(GP mean + GP marginal variance), shows neighboring sidebands and shades the actual blind/fit interval; black and red dashed vertical lines mark the mass hypothesis and signal core center. '
                       'All profiled curves are restricted to the likelihood bins, which are excluded from GP training. The quoted yield is full-selected and signed, with a conditional Hessian error. '
                       +scope+direction+'The region was selected using the same observed scan; neither the asymptotic nor toy probability includes the mass-search selection. '
                       'Toy rank probabilities use the add-one rule, while the stated exact two-sided 95% Clopper–Pearson interval describes the underlying exceedance fraction. '
                       'Disjoint fitted intervals still share GP sidebands and are not independent measurements.')
        summaries.append(dict(**meta,figure=key,fit_low_MeV=float(q.fit_low_MeV),fit_high_MeV=float(q.fit_high_MeV),
                              template_fit_fraction=float(q.template_fit_fraction),MC_training_fraction=float(q.MC_training_fraction),
                              toy=toy))
    write(R/'figure_captions.json',captions)
    write(R/'figure_summary.json',dict(figures=list(captions),regions=summaries,toy_calibration_included=toys is not None))
    print(json.dumps({'figures':list(captions),'toy_calibration_included':toys is not None}))


if __name__=='__main__':main()
