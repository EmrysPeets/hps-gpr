#!/usr/bin/env python3
"""Presentation figures and conditional exposure illustrations from frozen fits."""
from pathlib import Path
import hashlib,json,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BLUE,RED,GOLD,GREEN,PURPLE='#1C6795','#BA3A32','#AE801C','#297446','#735496'
LABEL={'2015':'2015 full','2016':'2016 full','2021':'2021 10%','sum':'Combined display sum'}
plt.rcParams.update({'font.family':'sans-serif','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10})
SUMMARY=pd.read_csv(HERE/'derived/fit_summary.csv',dtype={'dataset':str})
CONS=pd.read_csv(HERE/'derived/dataset_consistency.csv',dtype={'dataset':str})
BINS=pd.read_csv(HERE/'derived/display_bins.csv',dtype={'panel':str})
ARRAYS=np.load(HERE/'derived/fit_arrays.npz')
CLOSURE={x['fit_id']:x for x in json.loads((HERE/'derived/fit_closure.json').read_text())['checks']}
FIGURES=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save(fig,name):
    for ext in ('pdf','png'):
        p=HERE/'figures'/f'{name}.{ext}';fig.savefig(p,bbox_inches='tight',pad_inches=.12);FIGURES.append(p)
    plt.close(fig)
def step(ax,lo,hi,y,**kw):
    x=np.array([lo,hi]).T.ravel();v=np.repeat(y,2)
    return ax.plot(x,v,**kw)[0]
def unpack(fid,key):return {n:ARRAYS[fid+'__'+key+'__'+n] for n in
    ('edges','mask','observed','gp_mean','profiled_background','signal','total','null_background')}
def prior_covariance(fid,key):
    if key=='sum':
        cc=[]
        for row in SUMMARY[SUMMARY.fit_id==fid].itertuples():
            prefix=fid+'__'+row.dataset+'__';mask=ARRAYS[prefix+'mask']
            W=ARRAYS[prefix+'common_map'][:,mask];C=ARRAYS[prefix+'fit_covariance'];cc.append(W@C@W.T)
        return sum(cc)
    prefix=fid+'__'+key+'__';mask=ARRAYS[prefix+'mask'];W=ARRAYS[prefix+'display_map'][:,mask]
    return W@ARRAYS[prefix+'fit_covariance']@W.T

def panel(top,bottom,fid,key,title=None):
    d=BINS[(BINS.fit_id==fid)&(BINS.panel==key)].sort_values('bin')
    assert len(d)>0
    x=(d.low_MeV+d.high_MeV)/2;dx=(d.high_MeV-d.low_MeV)/2
    # All displayed quantities retain event units. Axis labels state the scaling.
    scale=1000.;err=np.sqrt(d.observed)/scale
    top.errorbar(x,d.observed/scale,yerr=err,xerr=dx,fmt='o',ms=3.9,lw=.9,color='black',zorder=5)
    bottom.errorbar(x,(d.observed-d.profiled_background)/scale,yerr=err,xerr=dx,
        fmt='o',ms=3.9,lw=.9,color='black',zorder=5)
    step(top,d.low_MeV,d.high_MeV,d.gp_mean/scale,color=GOLD,ls=':',lw=1.5)
    step(top,d.low_MeV,d.high_MeV,d.profiled_background/scale,color=BLUE,ls='--',lw=1.7)
    step(top,d.low_MeV,d.high_MeV,d.total/scale,color=RED,lw=1.8)
    step(bottom,d.low_MeV,d.high_MeV,d.signal/scale,color=RED,lw=1.8)
    sd=np.sqrt(np.diag(prior_covariance(fid,key)))/scale
    xx=np.array([d.low_MeV,d.high_MeV]).T.ravel()
    bottom.fill_between(xx,-np.repeat(sd,2),np.repeat(sd,2),color='.65',alpha=.22,zorder=0)
    bottom.axhline(0,color=BLUE,ls='--',lw=1.)
    m=int(CLOSURE[fid]['mass_MeV'])
    for ax in (top,bottom):
        ax.axvline(m,color='.6',ls=':',lw=.7,zorder=0)
        ax.set_xlim(d.low_MeV.min()-.08,d.high_MeV.max()+.08)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y',alpha=.13)
    ymax=max(float(((d.observed-d.profiled_background)/scale+err).abs().max()),
             float(((d.observed-d.profiled_background)/scale-err).abs().max()),
             float(np.max(np.abs(d.signal/scale))),float(sd.max()))
    bottom.set_ylim(-1.23*ymax,1.23*ymax)
    top.tick_params(labelbottom=False)
    top.set_title(title or LABEL[key],fontweight='semibold',fontsize=12,pad=10)
    top.set_ylabel(r'Events / bin [$10^3$]')
    bottom.set_ylabel('Data − fitted B\n'+r'[$10^3$ events / bin]')
    bottom.set_xlabel(r'$m_{ee}$ [MeV]')
    bw=float(d.high_MeV.iloc[0]-d.low_MeV.iloc[0])
    bottom.text(.02,.96,f'{bw:g} MeV bins',transform=bottom.transAxes,va='top',fontsize=9,color='.3')
    if key!='sum':
        sr=SUMMARY[(SUMMARY.fit_id==fid)&(SUMMARY.dataset==key)].iloc[0]
        bottom.text(.98,.04,r'$\sigma_m$ = '+f'{sr.sigma_MeV:.2f} MeV',transform=bottom.transAxes,
            ha='right',fontsize=8.5,color='.3')
    return d

def extraction(fids,keys,titles,name,heading,subheading,footnote):
    n=len(fids);width=max(10.,3.45*n)
    fig=plt.figure(figsize=(width,6.65))
    grid=fig.add_gridspec(2,n,height_ratios=(1.03,1),left=.12 if width<12 else .09,
        right=.99,bottom=.16,top=.77,hspace=.08,wspace=.37)
    for i,(fid,key,title) in enumerate(zip(fids,keys,titles)):
        top=fig.add_subplot(grid[0,i]);bot=fig.add_subplot(grid[1,i],sharex=top)
        panel(top,bot,fid,key,title)
    fig.suptitle(heading,fontsize=17,fontweight='semibold',y=.985)
    fig.text(.5,.924,subheading,ha='center',fontsize=11)
    handles=[Line2D([],[],color='black',marker='o',ls='none',label='Observed data'),
        Line2D([],[],color=GOLD,ls=':',lw=1.6,label='GP mean before profiling'),
        Line2D([],[],color=BLUE,ls='--',lw=1.7,label='Background fitted with signal'),
        Line2D([],[],color=RED,lw=1.8,label='S+B (top); fitted S (bottom)'),
        Patch(color='.65',alpha=.22,label='GP constraint SD (width guide)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.89),ncol=3,
        frameon=False,fontsize=9.3,columnspacing=1.5,handlelength=2.2)
    fig.text(.5,.069,footnote,ha='center',fontsize=9.7)
    fig.text(.5,.033,'Bars: counting error only. Gray shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty.',
        ha='center',fontsize=9,color='.3')
    save(fig,name)

def make_extractions():
    extraction(['combined_m021'],['2015'],['2015 full: the only active dataset'],
        'extraction_combined_21','Second full-region observed excess: 21 MeV',
        'The combined likelihood equals the 2015 likelihood here  |  signed root r = +2.52',
        'This location has no 2016 or 2021 coverage. Opening more 2021 data cannot test or strengthen this particular feature.')
    for m,description in [(66,'Leading combined observed excess'),(92,'Second leading peak with multiple datasets'),(72,'Deepest combined observed deficit')]:
        fid=f'combined_m{m:03d}';rows=SUMMARY[SUMMARY.fit_id==fid]
        keys=rows.dataset.tolist()+['sum'];r=CLOSURE[fid]['root'];eps=CLOSURE[fid]['eps2_hat']*1e6
        extraction([fid]*len(keys),keys,[LABEL[k] for k in keys],f'extraction_combined_{m}',
            f'{description}: {m} MeV',
            f'Common-amplitude profile fit  |  signed root r = {r:+.2f}  |  fitted amplitude = {eps:+.2f} × 10⁻⁶',
            'All panels use the same fitted amplitude. The summed panel uses only common whole bins; the likelihood uses the complete native windows.')
    for year,masses in [('2015',[51,21]),('2016',[90,117]),('2021',[78,65])]:
        fids=[f'{year}_m{m:03d}' for m in masses]
        titles=[f'{m} MeV: r = {CLOSURE[fid]["root"]:+.2f}' for m,fid in zip(masses,fids)]
        extraction(fids,[year]*2,titles,f'extraction_{year}_peaks',
            f'{LABEL[year]}: two leading observed excesses','Separate single-dataset fits  |  observed-profiled ranking  |  display bins ≈ half the mass resolution',
            'Selected after scanning. Local signed roots describe the fitted templates; these panels do not establish a global particle significance.')
    fids=['2015_m019','2016_m102','2021_m071']
    titles=[f'{LABEL[k]}: {m} MeV, r = {CLOSURE[fid]["root"]:+.2f}'+(' (endpoint)' if k=='2015' else '')
        for fid,k,m in zip(fids,['2015','2016','2021'],[19,102,71])]
    extraction(fids,['2015','2016','2021'],titles,'extraction_individual_deficits',
        'Deepest observed deficit in each dataset','Independent fitted amplitudes  |  same profile construction and resolution-based binning',
        'The negative signal template is an auxiliary diagnostic of missing events; it is not a physical negative event rate.')
    # The stress extrema are not substitute rankings of observed signal amplitude.
    extraction(['combined_m076','combined_m083'],['sum','sum'],
        ['76 MeV: observed r = +0.17','83 MeV: observed r = −0.68'],'extraction_stress_extrema',
        'Extreme stress-centered tails can have small fitted signals',
        '76 MeV: stress offset a = −8.70  |  83 MeV: stress offset a = +7.71',
        'Centering changes the probability assigned to a fit. It does not turn the small red template into a larger fitted signal.')

def make_consistency():
    masses=[66,92,72,76];fig,axes=plt.subplots(1,4,figsize=(13.4,5.5))
    fig.subplots_adjust(left=.075,right=.99,bottom=.27,top=.72,wspace=.42)
    for ax,m in zip(axes,masses):
        fid=f'combined_m{m:03d}';d=CONS[CONS.fit_id==fid]
        p=SUMMARY[SUMMARY.fit_id==fid].iloc[0];colors={'2015':BLUE,'2016':GOLD,'2021':GREEN}
        for j,row in enumerate(d.itertuples()):
            ax.errorbar(row.individual_eps2_hat*1e6,j,xerr=row.individual_sigma_eps2*1e6,
                fmt='o',ms=6,color=colors[row.dataset],lw=1.6,capsize=4)
        ax.axvspan((p.eps2_hat-p.sigma_eps2)*1e6,(p.eps2_hat+p.sigma_eps2)*1e6,color=RED,alpha=.12)
        ax.axvline(p.eps2_hat*1e6,color=RED,lw=1.5,label='Common fit ± local curvature SD')
        ax.axvline(0,color='.4',ls=':',lw=1.)
        ax.set_yticks(range(len(d)),[LABEL[k] for k in d.dataset]);ax.invert_yaxis()
        ax.set_ylim(len(d)-.35,-.65);ax.grid(axis='x',alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(r'Signed fitted amplitude $[10^{-6}]$')
        q=CLOSURE[fid]['individual_common_deviance'];df=CLOSURE[fid]['compatibility_df']
        ax.set_title(f'{m} MeV\nΔD = {q:.2f} for {df} '+('constraint' if df==1 else 'constraints'),fontsize=10.5,fontweight='semibold',pad=12)
    fig.suptitle('Do the years prefer a compatible signal rate?',fontsize=17,fontweight='semibold',y=.98)
    fig.text(.5,.886,'Points: separate profiled fits  |  Red line and shading: shared-amplitude fit ± local curvature SD',ha='center',fontsize=11)
    fig.text(.5,.095,'At 66 MeV all three fitted amplitudes are positive. At 92 MeV the preferred rates differ; at 76 MeV the years pull in opposite directions.',
        ha='center',fontsize=10)
    fig.text(.5,.052,'ΔD compares independent amplitudes with one shared amplitude at a selected mass. No post-selection or global compatibility probability is claimed.',
        ha='center',fontsize=9.5,color='.3')
    save(fig,'dataset_amplitude_consistency')

def make_exposure():
    projected=[];rates=[];precision=[]
    information=pd.read_csv(HERE/'derived/information.csv',dtype={'dataset':str})
    fig,axes=plt.subplots(2,4,figsize=(13.6,7.0))
    fig.subplots_adjust(left=.07,right=.99,bottom=.17,top=.77,wspace=.31,hspace=.39)
    headings=['Observed 10%','Additional 20% only','Cumulative 30%','Cumulative 100%']
    for row,m in enumerate([66,92]):
        fid=f'combined_m{m:03d}';d=BINS[(BINS.fit_id==fid)&(BINS.panel=='2021')].sort_values('bin')
        b=d.profiled_background.to_numpy();s=d.signal.to_numpy();n=d.observed.to_numpy();r=n-b
        x=(d.low_MeV+d.high_MeV)/2;dx=(d.high_MeV-d.low_MeV)/2
        for col,(kind,f) in enumerate([('observed',1.),('increment',2.),('cumulative',3.),('cumulative',10.)]):
            ax=axes[row,col];baseline=0. if kind=='increment' else r
            g=f if kind=='increment' else f-1
            null=np.zeros(len(r))+baseline;persistent=null+g*s
            sd=np.sqrt(g*b)
            if kind=='observed':
                ax.errorbar(x,r/1000,yerr=np.sqrt(n)/1000,xerr=dx,fmt='o',color='black',ms=4,lw=.9)
                step(ax,d.low_MeV,d.high_MeV,s/1000,color=RED,lw=1.8)
                null=r;persistent=r
            else:
                xx=np.array([d.low_MeV,d.high_MeV]).T.ravel()
                ax.fill_between(xx,np.repeat((null-sd)/1000,2),np.repeat((null+sd)/1000,2),color='.65',alpha=.25)
                step(ax,d.low_MeV,d.high_MeV,null/1000,color='.35',ls='--',lw=1.4)
                step(ax,d.low_MeV,d.high_MeV,persistent/1000,color=RED,lw=1.8)
            ax.axhline(0,color='.5',lw=.65);ax.axvline(m,color='.6',ls=':',lw=.7)
            ax.set_xlim(d.low_MeV.min()-.08,d.high_MeV.max()+.08)
            ax.set_title(headings[col] if row==0 else '',fontweight='semibold',fontsize=12)
            ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='y',alpha=.13)
            ax.set_xlabel(r'$m_{ee}$ [MeV]')
            if col==0:ax.set_ylabel(f'{m} MeV hypothesis\n'+r'Residual / bin [$10^3$ events]')
            else:ax.set_ylabel(r'Mean residual [$10^3$]')
            if col==1:ax.text(.04,.05,'Independent new sample',transform=ax.transAxes,va='bottom',fontsize=8.4,color=GREEN,
                bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            if col in (2,3):ax.text(.04,.05,'Includes observed 10%',transform=ax.transAxes,va='bottom',fontsize=8.4,color='.3',
                bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            for i in range(len(d)):
                projected.append(dict(mass_MeV=m,view=kind,exposure_factor=f,bin=i,low_MeV=float(d.low_MeV.iloc[i]),high_MeV=float(d.high_MeV.iloc[i]),
                    original_observed=float(n[i]),reference_background=float(b[i]),assumed_signal_per_10pct=float(s[i]),
                    added_factor=g,null_residual=float(null[i]),persistent_residual=float(persistent[i]),
                    added_background_counting_sd=float(sd[i]),future_view=kind!='observed'))
        sr=SUMMARY[(SUMMARY.fit_id==fid)&(SUMMARY.dataset=='2021')].iloc[0]
        for percent in [10,20,30,100]:
            fac=percent/10
            rates.append(dict(mass_MeV=m,exposure_percent=percent,interpretation='new independent increment' if percent==20 else 'total exposure',
                assumed_eps2=float(sr.eps2_hat),template_yield_window=float(fac*sr.signal_window),
                yield_convention='mean signal for constant selected rate; total window, not display subset'))
        inf=information[information.fit_id==fid];i21=float(inf.loc[inf.dataset=='2021','information'].iloc[0]);iold=float(inf.loc[inf.dataset!='2021','information'].sum())
        for percent in [10,30,100]:
            fac=percent/10;gain=float(np.sqrt((iold+fac*i21)/(iold+i21)))
            precision.append(dict(mass_MeV=m,exposure_percent=percent,exposure_factor=fac,
                original_2021_information_fraction=i21/(iold+i21),combined_precision_gain=gain,
                combined_uncertainty_ratio=1/gain,individual_2021_precision_gain=float(np.sqrt(fac)),
                assumption='all 2021 count covariance scales with exposure; fixed 2015/2016; local Fisher approximation'))
    fig.suptitle('A 30% checkpoint before opening the full 2021 sample',fontsize=17,fontweight='semibold',y=.985)
    fig.text(.5,.925,'Illustrative persistence of the selected common-fit rate at 66 and 92 MeV  |  2015 and 2016 stay fixed',ha='center',fontsize=11)
    fig.legend(handles=[Line2D([],[],color='black',marker='o',ls='none',label='Observed 10% residual'),
        Line2D([],[],color=RED,lw=1.8,label='Selected rate persists (mean)'),
        Line2D([],[],color='.35',ls='--',label='Only background in added data (mean)'),
        Patch(color='.65',alpha=.25,label='Added-sample background counting SD')],
        loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False,fontsize=9.5)
    fig.text(.5,.086,'Future panels are conditional expectations. Cumulative panels retain the actual first 10%; y-axis ranges differ to show the change in event yield.',ha='center',fontsize=10)
    fig.text(.5,.043,'The 92 MeV rates differ between years. These selected-rate examples omit background-model uncertainty; future data and global significances are not supplied.',ha='center',fontsize=9.2,color='.3')
    save(fig,'exposure_2021_10_30_100')
    pd.DataFrame(projected).to_csv(HERE/'derived/exposure_display_bins.csv',index=False)
    pd.DataFrame(rates).to_csv(HERE/'derived/exposure_signal_yields.csv',index=False)
    pd.DataFrame(precision).to_csv(HERE/'derived/exposure_precision.csv',index=False)
    dump(HERE/'derived/exposure_contract.json',dict(passed=True,observed_2021_percent=10,
        future_2021_percent=[30,100],independent_increment_percent=20,other_years_fixed=True,
        new_toys=0,new_unblinded_events=0,assumed_masses_MeV=[66,92],
        baseline='background profiled in the selected common-amplitude fit to the released 10%',
        cumulative_mean='N10+(f-1)*(B10+S10) or N10+(f-1)*B10',
        future_error='counting variation of the added sample only; no GP/background refit or systematic uncertainty',
        at_21_MeV='2015 only; no 2021 exposure benefit'))

def main():
    make_extractions();make_consistency();make_exposure()
    dump(HERE/'provenance/figures.json',dict(passed=True,figures={str(p.relative_to(ROOT)):sha(p) for p in FIGURES},
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'derived/fit_arrays.npz',
            HERE/'derived/fit_summary.csv',HERE/'derived/display_bins.csv',HERE/'derived/fit_closure.json',
            HERE/'derived/dataset_consistency.csv',HERE/'derived/information.csv']}))
    print('Wrote',len(FIGURES),'figure files')
if __name__=='__main__':main()
