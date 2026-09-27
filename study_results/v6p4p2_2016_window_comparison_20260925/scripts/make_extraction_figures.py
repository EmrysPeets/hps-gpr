#!/usr/bin/env python3
"""Read-only figure generation from saved v6.4.1 profiles; no refitting."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hps-v641-extraction-figures')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

B=Path(__file__).resolve().parents[1]
R=B/'results';F=B/'figures'
BLUE='#245c91';RED='#ab3d35';GOLD='#a98224';GREEN='#39846b';PURPLE='#865697';GRAY='#777777'
METHODS2016=('gaussian','gaussian_mc_window','mc')
NAMES2016={'gaussian':'Gaussian: original window',
           'gaussian_mc_window':'Gaussian: MC-centered window',
           'mc':'Neighboring signal-MC template'}
COLOR2016={'gaussian':GRAY,'gaussian_mc_window':GOLD,'mc':BLUE}
STYLE2016={'gaussian':'--','gaussian_mc_window':':','mc':'-'}
METHODSJOINT=('all_gaussian','mc2021','mc2016_2021')
NAMESJOINT={'all_gaussian':'All Gaussian (2021 already shifted)',
            'mc2021':'MC 2021; Gaussian 2015 and 2016',
            'mc2016_2021':'MC 2016 and 2021; Gaussian 2015'}
COLORJOINT={'all_gaussian':GRAY,'mc2021':GOLD,'mc2016_2021':BLUE}
STYLEJOINT={'all_gaussian':'--','mc2021':'-.','mc2016_2021':'-'}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                    'axes.titlesize':9,'legend.fontsize':7.2,'pdf.fonttype':42,
                    'axes.spines.top':False,'axes.spines.right':False})


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def save(fig,name):
    F.mkdir(exist_ok=True)
    fig.savefig(F/(name+'.pdf'))
    fig.savefig(F/(name+'.png'),dpi=180)
    plt.close(fig)


def format_p(value):return f'{value:.3g}'


def scan2016(scan,selected):
    q=scan[scan.scope=='2016']
    fig,axes=plt.subplots(3,1,figsize=(7.2,6.5),sharex=True)
    columns=['A90','epsilon2_90_visible_legacy','p0_asymptotic']
    for method in METHODS2016:
        rows=q[q.method==method].sort_values('mass_MeV')
        assert len(rows)==136
        for ax,column in zip(axes,columns):
            ax.plot(rows.mass_MeV,rows[column],color=COLOR2016[method],ls=STYLE2016[method],lw=1.25,label=NAMES2016[method])
    for r in selected:
        if r['scope']!='2016':continue
        m=r['mass_MeV'];row=q[(q.method=='mc')&(q.mass_MeV==m)].iloc[0]
        for ax,column in zip(axes,columns):
            ax.axvline(m,color='.65',ls=':',lw=.65,zorder=-2)
            ax.plot(m,row[column],'o',color=BLUE,ms=3.5,zorder=6)
    for ax in axes:
        ax.grid(axis='y',alpha=.15);ax.set_xlim(40,175);ax.set_yscale('log')
    axes[0].set_ylabel('90% CLs limit\n[full selected yield]')
    axes[1].set_ylabel('Conditional ε² display\n[90% CLs limit]')
    axes[2].set(ylabel='Local excess p\n[asymptotic reference]',xlabel='Generated signal mass hypothesis [MeV]',ylim=(float(q.p0_asymptotic.min())*.55,.9))
    for ax,title in zip(axes,['(a) Yield under each assumed signal shape','(b) Inherited yield-to-coupling conversion','(c) Fixed-mass local probabilities']):
        ax.set_title(title,loc='left',fontsize=8,pad=4)
    fig.text(.5,.985,'2016 observed spectrum: signal shape and window comparison',ha='center',va='top',fontsize=11)
    handles,_=axes[0].get_legend_handles_labels()
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.945),ncol=2,frameon=False,fontsize=7.4,columnspacing=1.1)
    fig.subplots_adjust(left=.15,right=.98,top=.825,bottom=.08,hspace=.29)
    save(fig,'extraction_2016_scan')


def joint_scan(scan,selected):
    q=scan[scan.scope=='combined']
    fig,axes=plt.subplots(2,1,figsize=(7.2,5.6),sharex=True)
    for method in METHODSJOINT:
        rows=q[q.method==method].sort_values('mass_MeV');assert len(rows)==181
        axes[0].plot(rows.mass_MeV,rows.epsilon2_90_visible_legacy,color=COLORJOINT[method],ls=STYLEJOINT[method],lw=1.25,label=NAMESJOINT[method])
        axes[1].plot(rows.mass_MeV,rows.p0_asymptotic,color=COLORJOINT[method],ls=STYLEJOINT[method],lw=1.25)
    for r in selected:
        if r['scope']!='combined':continue
        m=r['mass_MeV'];row=q[(q.method=='mc2016_2021')&(q.mass_MeV==m)].iloc[0]
        for ax,column in zip(axes,['epsilon2_90_visible_legacy','p0_asymptotic']):
            ax.plot(m,row[column],'o',color=BLUE,ms=3.5,zorder=6)
    for ax in axes:
        ax.grid(axis='y',alpha=.15);ax.set_xlim(60,240);ax.set_yscale('log')
        for boundary in (100.5,175.5):ax.axvline(boundary,color='.72',ls=':',lw=.7,zorder=-2)
    axes[0].set_ylabel('Conditional ε² display\n[90% CLs limit]')
    axes[1].set(ylabel='Local excess p\n[asymptotic reference]',xlabel='Generated signal mass hypothesis [MeV]',ylim=(float(q.p0_asymptotic.min())*.55,.9))
    fig.text(.5,.985,'Combined observed analysis: replace the signal models in stages',ha='center',va='top',fontsize=11)
    handles,_=axes[0].get_legend_handles_labels()
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.94),ncol=1,frameon=False,fontsize=7.5)
    fig.text(.53,.803,'60–100 MeV: three campaigns  |  101–175: 2016 + 2021  |  176–240: 2021',ha='center',va='top',fontsize=7.2)
    fig.subplots_adjust(left=.15,right=.98,top=.755,bottom=.1,hspace=.12)
    save(fig,'extraction_combined_comparison')


def campaign_contributions(scan):
    fig,axes=plt.subplots(2,1,figsize=(7.2,5.4),sharex=True)
    groups=[('2015','gaussian','2015 Gaussian',GRAY,':'),('2016','mc','2016 MC',GREEN,'--'),
            ('2021','mc','2021 MC (10% data)',GOLD,'-.'),('combined','mc2016_2021','Combined: 2015 Gaussian + 2016/2021 MC',BLUE,'-')]
    for scope,method,label,color,ls in groups:
        rows=scan[(scan.scope==scope)&(scan.method==method)&(scan.mass_MeV>=60)].sort_values('mass_MeV')
        axes[0].plot(rows.mass_MeV,rows.epsilon2_90_visible_legacy,color=color,ls=ls,lw=1.3,label=label)
        axes[1].plot(rows.mass_MeV,rows.p0_asymptotic,color=color,ls=ls,lw=1.3)
    for ax in axes:
        ax.grid(axis='y',alpha=.15);ax.set_xlim(60,240);ax.set_yscale('log')
        for boundary in (100.5,175.5):ax.axvline(boundary,color='.72',ls=':',lw=.7,zorder=-2)
    values=scan[(scan.scope=='2016')&(scan.method=='mc')].p0_asymptotic
    axes[0].set_ylabel('Conditional ε² display\n[90% CLs limit]')
    axes[1].set(ylabel='Local excess p\n[asymptotic reference]',xlabel='Generated signal mass hypothesis [MeV]',ylim=(float(values.min())*.55,.9))
    fig.text(.5,.985,'Which campaigns contribute to the combined result?',ha='center',va='top',fontsize=11)
    handles,_=axes[0].get_legend_handles_labels()
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.94),ncol=2,frameon=False,fontsize=7.2,columnspacing=1.2)
    fig.subplots_adjust(left=.15,right=.98,top=.815,bottom=.1,hspace=.12)
    save(fig,'extraction_campaign_contributions')


def joint_ratios(scan):
    p=scan[scan.scope=='combined'].pivot(index='mass_MeV',columns='method',values='epsilon2_90_visible_legacy')
    fig,axes=plt.subplots(2,1,figsize=(7.2,4.5),sharex=True,gridspec_kw={'height_ratios':[1.15,1]})
    for method in ['mc2021','mc2016_2021']:
        axes[0].plot(p.index,p[method]/p.all_gaussian,color=COLORJOINT[method],ls=STYLEJOINT[method],lw=1.3,label=NAMESJOINT[method])
    axes[1].plot(p.index,p.mc2016_2021/p.mc2021,color=BLUE,lw=1.3)
    for ax in axes:
        ax.axhline(1,color='.4',lw=.8,ls=':');ax.grid(axis='y',alpha=.15);ax.set_xlim(60,240)
        for boundary in (100.5,175.5):ax.axvline(boundary,color='.72',ls=':',lw=.7,zorder=-2)
    axes[0].set_ylabel('Limit / all-Gaussian limit')
    axes[1].set(ylabel='Both MC / 2021-only MC',xlabel='Generated signal mass hypothesis [MeV]')
    fig.text(.5,.985,'Separate the two changes to the combined upper limit',ha='center',va='top',fontsize=11)
    handles,_=axes[0].get_legend_handles_labels()
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.935),ncol=1,frameon=False,fontsize=7.4)
    fig.subplots_adjust(left=.16,right=.98,top=.78,bottom=.12,hspace=.15)
    save(fig,'extraction_combined_ratios')

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
    handles += [Patch(facecolor='#b5b5b5',alpha=.22,label='Fit interval: excluded from GP training'),
                Line2D([],[],color='black',ls='--',lw=1,label='Search mass hypothesis'),
                Line2D([],[],color=RED,ls='--',lw=1,label='Signal core center')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.873),ncol=3,
               frameon=False,fontsize=7.1,columnspacing=1.25,handlelength=2.2)
    feature=('Selected deficit' if name=='deficit' else 'Selected excess '+name.rsplit('_',1)[-1])
    fig.text(.5,.993,f'2016: {feature.lower()} at a {mass:g} MeV signal hypothesis',ha='center',va='top',fontsize=11)
    model='Neighboring signal-MC template; window −3.5u to +3.5u'
    fig.text(.5,.966,f'{model} | c = {c:.3f} MeV, s = {s:.3f} MeV',
             ha='center',va='top',fontsize=7.9)
    root=float(meta['signed_root']); ahat=float(meta['Ahat']);err=float(meta['sigma_A'])
    stats=f'Signed full-selected yield = {ahat:,.0f} ± {err:,.0f}'
    if root>=0:stats+=f'; local asymptotic Z = {root:.2f} σ, p = {format_p(float(meta["local_p"]))}'
    else:stats+=f'; signed likelihood root r = {root:.2f}; excess q₀ = 0, p₀ = 0.5'
    fig.text(.5,.939,stats,ha='center',va='top',fontsize=8.1)
    if toy is not None:
        label='deficit' if name=='deficit' else 'excess'
        toytext=f'Fixed-source null toys: local rank p = {format_p(float(toy["rank_p"]))} ({int(toy["k"])} / {int(toy["n"])} exceedances)'
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
    for ax,label in zip(axes,['(a) Spectrum in the fit interval','(b) Residuals relative to the GP mean','(c) Standardized residuals and excluded fit interval']):
        ax.set_title(label,loc='left',fontsize=8,pad=5)
    fig.subplots_adjust(left=.135,right=.98,top=.72,bottom=.085,hspace=.45)
    save(fig,'extraction_2016_'+name)


def main():
    scan=pd.read_csv(R/'observed_scan.csv')
    assert np.isfinite(scan[['mass_MeV','epsilon2_90_visible_legacy','p0_asymptotic']].to_numpy()).all()
    assert np.all(scan.p0_asymptotic>0)
    regions=json.loads((R/'selected_regions.json').read_text())['regions']
    scan2016(scan,regions);joint_scan(scan,regions);campaign_contributions(scan);joint_ratios(scan)
    captions={
      'extraction_2016_scan': 'How to read the figure. Gray is the inherited Gaussian centered at the generated mass in its original plus/minus 2.25 reference-width window. Gold keeps that Gaussian and changes the fit and GP-training exclusion together to the MC-centered minus 3.5 to plus 3.5 core-width window. Blue uses the neighboring MC template in that same window. The top panel reports limits on each model\'s full selected yield; the Gaussian curves are not limits on an MC-shaped signal. The middle panel applies the inherited conditional yield-to-coupling conversion. The bottom panel gives fixed-mass asymptotic excess probabilities. Blue points mark the two selected MC-fit regions, at 91 and 69 MeV, ranked by the excess statistic while requiring disjoint fitted bins. Their locations were selected using the observed spectrum. No curve includes template, calibration or selection uncertainty, and no displayed probability is globally calibrated.',
      'extraction_combined_comparison': 'How to read the figure. The three curves share the same observations, campaign support and inherited conditional yield-to-coupling conversion: archived full 2015 and 2016 spectra, and the archived 2021 10% spectrum. Gray uses Gaussian templates in all campaigns; its 2021 Gaussian already follows the established logarithmic core shift. Gold replaces only 2021 with its shifted neighboring MC template and minus 4 to plus 3 core-width window. Blue also replaces 2016 with its shifted neighboring MC template and symmetric 3.5-core-width window. The 2015 Gaussian is retained throughout. Dotted boundaries indicate the end of 2015 at 100 MeV and 2016 at 175 MeV; above 175 MeV the two MC curves coincide because only 2021 contributes. The upper panel shows conditional observed 90% profile-CLs limits; the lower panel shows local asymptotic excess probabilities. Blue points mark the two observed-selected combined excess regions. These are model comparisons under fixed inputs, not detector-qualified exclusions or scan-wide significance.',
      'extraction_campaign_contributions': 'How to read the figure. Individual-campaign curves use the retained 2015 Gaussian, the shifted 2016 MC template, and the shifted 2021 MC template. Blue is their common-coupling profile combination with separate campaign GP constraints. Campaigns enter only within their fixed support: 2015 through 100 MeV, 2016 through 175 MeV, and 2021 through 240 MeV, starting at 60 MeV in this combined display. A combined observed limit need not be below every individual observed limit, because the datasets fluctuate and the common signal parameter constrains them together. Likewise, local probabilities are obtained from the combined likelihood, not by multiplying individual probabilities. Above 175 MeV the combined and 2021 curves coincide. The coupling conversion and asymptotic sampling approximation remain conditional.',
      'extraction_combined_ratios': 'How to read the figure. The upper panel divides each MC-based combined upper limit by the all-Gaussian combined limit at the same generated mass. The lower panel isolates the additional 2016 change by dividing the both-MC limit by the 2021-only-MC limit. One means no change, values below one mean a smaller conditional upper limit, and values above one mean a larger one. The lower ratio is exactly one above 175 MeV, where 2016 no longer contributes. These ratios compare assumed signal shapes and their fit/training windows together; they do not isolate a pure shift correction or quantify an efficiency improvement.'}
    toyfile=R/'local_checks.csv';toys=pd.read_csv(toyfile) if toyfile.exists() else None
    for selected in regions:
        if selected['scope']!='2016':continue
        mass=int(selected['mass_MeV']);region=selected['region']
        path=R/'selected_fits'/f'2016_{region}_m{mass:03d}_mc.npz'
        with np.load(path,allow_pickle=False) as z:
            mask=np.asarray(z['fit_mask'],bool);geo=json.loads(str(z['geometry_json']));q=json.loads(str(z['summary_json']))
            data=dict(x=z['x_GeV'][mask]*1000,width=np.diff(z['edges_GeV'])[mask]*1000,
                      counts=z['fit_counts'],gp_mean=z['fit_prefit_mean'],
                      gp_sd=np.sqrt(np.maximum(np.diag(z['fit_prefit_covariance']),0)),
                      b_null=z['profiled_background_only'],b_signal=z['profiled_background_signed'],
                      signal=z['profiled_signed_signal'],total=z['profiled_signed_total'],
                      full_x=z['x_GeV']*1000,full_width=np.diff(z['edges_GeV'])*1000,full_counts=z['counts'],
                      full_gp=z['prefit_GP_mean'],full_sd=np.sqrt(np.maximum(np.diag(z['prefit_GP_covariance']),0)))
        meta=dict(region=region,mass=mass,center=float(geo['center_MeV']),width=float(geo['core_width_MeV']),
                  signed_root=float(q['signed_root']),Ahat=float(q['Ahat']),sigma_A=float(q['sigma_A']),
                  local_p=float(q['p0_asymptotic']),fit_low=float(geo['actual_low_MeV']),fit_high=float(geo['actual_high_MeV']))
        toy=None
        if toys is not None:
            t=toys[(toys.scope.astype(str)=='2016')&(toys.method=='mc')&(toys.mass_MeV==mass)]
            assert len(t)<=1
            if len(t)==1:
                t=t.iloc[0];toy=dict(rank_p=float(t.p_rank),k=int(t.tail_count),n=int(t.null_toys),lo=float(t.cp95_low),hi=float(t.cp95_high))
        region_figure(meta,data,toy)
        captions['extraction_2016_'+region]=('How to read the figure. Black points show observed counts per MeV with square-root-count errors. Blue is the GP sideband prediction and one marginal standard deviation of its background constraint; the band is not a post-fit confidence interval. Gold is the profiled background-only fit. Green is the background component of the signed signal-plus-background fit, and red is its total. The middle panel subtracts the GP mean and shows the signed signal separately in purple. The bottom panel divides residuals by the square root of GP mean plus GP marginal variance; it shows neighboring sidebands and shades the actual fit interval, which is excluded from GP training. Black and red vertical lines mark the generated signal hypothesis and reconstructed MC core. The standardized residuals are a display of correlated bins, not independent significance measurements. Profiled curves appear only in likelihood bins. The reported signed yield refers to the full selected MC distribution and has a conditional Hessian error. This region was selected from the observed scan; its local probabilities do not include mass-search selection. Disjoint fit intervals still share GP sidebands and are not independent tests.' + (' The fixed-source null-toy rank uses the add-one rule; its exact two-sided 95% Clopper-Pearson interval describes the underlying exceedance probability.' if toy is not None else ' No null-toy annotation was available when this figure was generated.'))
    write(R/'extraction_figure_captions.json',captions)
    print(json.dumps({'figures':list(captions),'toy_file_present':toys is not None}))

if __name__=='__main__':main()
