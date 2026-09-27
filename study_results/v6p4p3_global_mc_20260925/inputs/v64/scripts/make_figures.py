"""Readable native-histogram overlays; plot data retain full normalization."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.special import ndtr
from analyze import cdf, design

B=Path(__file__).resolve().parents[1]
D=pd.read_csv(B/'results/centers_and_shapes.csv').set_index('mass_MeV')
M=D.index[D.primary_domain].to_numpy()
R=pd.read_csv(B/'inputs/reference_2021/core_native_diagnostics.csv').set_index('mass_MeV')
F=B/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
                    'axes.titlesize':11,'legend.fontsize':8,'axes.grid':True,
                    'grid.alpha':.16,'pdf.fonttype':42,'savefig.facecolor':'white'})
BLUE='#225b88';RED='#a9472d';GREEN='#34826b';GRAY='#666666'
norm=Normalize(40,175);cmap=plt.cm.viridis

def hist(m):
    a=np.load(B/'histograms'/f'm{m:03d}.npz');return a['edges_MeV'],a['counts']

def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

def colorbar(fig,axs):
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs,label='Generated mass (MeV)',fraction=.035,pad=.035)

def main():
    fig,ax=plt.subplots(1,2,figsize=(10.7,4.1),layout='constrained')
    ax[0].errorbar(M,D.loc[M].shift_MeV,yerr=D.loc[M].center_mc_sd_MeV,fmt='o',ms=3,color=BLUE,label='Fitted core; MC resampling SD')
    ax[0].plot(M,D.loc[M].median_shift_MeV,'-',color=GREEN,lw=1.1,label='Median')
    ax[0].plot(M,D.loc[M].mean_shift_MeV,'-',color=RED,lw=1.1,label='Binned mean')
    ax[0].set(title='2016: the center depends on its definition',ylabel='Center minus generated mass (MeV)',ylim=(-1.4,.25))
    ax[0].legend(loc='lower left',frameon=False)
    ax[1].errorbar(M,D.loc[M].shift_MeV,yerr=D.loc[M].center_mc_sd_MeV,fmt='o-',lw=.7,ms=3,color=BLUE,label='2016 core')
    rm=R.index[(R.index>=60)&(R.index<=160)]
    ax[1].errorbar(rm,R.loc[rm].center_shift_MeV,yerr=R.loc[rm].center_MC_bin_resample_std_MeV,fmt='s-',color=RED,ms=4,label='Saved 2021 core')
    ax[1].set(title='The same sign, a much smaller displacement',ylabel='Fitted core minus generated mass (MeV)',ylim=(-3.65,.25))
    ax[1].legend(frameon=False,loc='lower left')
    for a in ax:a.axhline(0,color=GRAY,ls=':',lw=1);a.set_xlabel('Generated mass (MeV)')
    save(fig,'centers_and_2021')

    fig,axs=plt.subplots(1,3,figsize=(10.7,3.5),layout='constrained')
    for ax,m in zip(axs,[60,100,160]):
        e,y=hist(m);r=D.loc[m];x=(e[1:]+e[:-1])/2;dx=np.diff(e);N=y.sum()
        ax.errorbar(x,y/N/dx,yerr=np.sqrt(y)/N/dx,fmt='.',ms=2,color=GRAY,lw=.6,label='Native MC')
        take=(e[:-1]>=r.fit_low_MeV-1e-8)&(e[1:]<=r.fit_high_MeV+1e-8)
        g=r.gaussian_area*(ndtr((e[1:]-r.center_MeV)/r.sigma_core_MeV)-ndtr((e[:-1]-r.center_MeV)/r.sigma_core_MeV))
        mix=(x[take]-x[take].min())/(x[take].max()-x[take].min())
        base=r.pedestal_left*(1-mix)+r.pedestal_right*mix
        ax.plot(x[take],(g[take]+base)/N/dx[take],color=BLUE,lw=1.6,label='Local Gaussian + pedestal')
        ax.plot(x[take],g[take]/N/dx[take],color=GREEN,lw=1.2,ls='--',label='Gaussian component')
        ax.axvline(m,color=RED,ls=':',lw=1.2,label='Generated mass')
        ax.axvline(r.center_MeV,color=BLUE,ls='--',lw=1)
        ax.set(xlim=(m-3*r.sigma_ref_MeV,m+3*r.sigma_ref_MeV),ylim=(0,None),title=f'{m} MeV: core shift {r.shift_MeV:+.3f} MeV',xlabel='Reconstructed mass (MeV)',ylabel=r'Density (MeV$^{-1}$)')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    save(fig,'core_fit_examples')

    fig,axs=plt.subplots(1,2,figsize=(10.7,4.0),layout='constrained')
    d=D.loc[M]
    axs[0].errorbar(M,d.sigma_core_MeV,yerr=d.sigma_mc_sd_MeV,fmt='o',ms=3,color=BLUE,label='Fitted core width')
    axs[0].plot(M,d.sigma_ref_MeV,color=RED,label='Existing 2016 analysis resolution')
    axs[0].plot(M,d.halfwidth68_MeV,color=GREEN,ls='--',label='Half of central 68% interval')
    axs[0].set(title='Core width and analysis width are distinct',ylabel='Width (MeV)',xlabel='Generated mass (MeV)')
    axs[0].legend(frameon=False)
    axs[1].plot(M,100*d.left_tail_2,'o-',color=BLUE,ms=3,label='Below core - 2 core widths')
    axs[1].plot(M,100*d.right_tail_2,'s-',color=RED,ms=3,label='Above core + 2 core widths')
    axs[1].axhline(100*ndtr(-2),color=GRAY,ls=':',label='Gaussian: 2.28% on each side')
    axs[1].set(title='The tails are heavier than Gaussian',xlabel='Generated mass (MeV)',ylabel='Full selected probability (%)',ylim=(0,7.2))
    axs[1].legend(frameon=False,loc='upper right')
    save(fig,'widths_and_tails')

    fig,axs=plt.subplots(1,2,figsize=(10.7,4.1),layout='constrained')
    for m in M:
        e,y=hist(m);r=D.loc[m];x=(e[:-1]+e[1:])/2;den=y/y.sum()/np.diff(e)
        axs[0].step(x-m,den,where='mid',color=cmap(norm(m)),lw=.8)
        axs[1].step((x-m)/r.sigma_ref_MeV,den*r.sigma_ref_MeV,where='mid',color=cmap(norm(m)),lw=.8)
    z=np.linspace(-6,6,600)
    axs[1].plot(z,np.exp(-z*z/2)/np.sqrt(2*np.pi),'k:',lw=1.6,label='Centered Gaussian')
    axs[0].set(xlim=(-18,18),ylim=(0,.25),xlabel='Reconstructed minus generated mass (MeV)',ylabel=r'Density (MeV$^{-1}$)',title='Widths increase with generated mass')
    axs[1].set(xlim=(-6,6),ylim=(1e-4,.55),yscale='log',xlabel=r'$(m_{ee}-m)/\sigma_{ref}$',ylabel='Density per analysis-width unit',title='Scaling by the existing analysis width')
    axs[1].legend(frameon=False,loc='upper left')
    colorbar(fig,axs);save(fig,'pole_aligned_overlays')

    fig,axs=plt.subplots(1,2,figsize=(10.7,4.2),layout='constrained')
    ue=np.arange(-12,12.0001,.10);uc=(ue[:-1]+ue[1:])/2;pool=[]
    for m in M:
        e,y=hist(m);r=D.loc[m];q=cdf(e,y,r.center_MeV+r.sigma_core_MeV*ue);p=np.diff(q)/np.diff(ue);pool.append(q)
        axs[0].plot(uc,p/r.core_fraction_2,color=cmap(norm(m)),lw=.8)
        axs[1].plot(uc,np.where(p>0,p,np.nan),color=cmap(norm(m)),lw=.8)
    q=np.mean(pool,axis=0);p=np.diff(q)/np.diff(ue);f=np.interp(2,ue,q)-np.interp(-2,ue,q)
    axs[0].plot(uc,p/f,'k--',lw=1.8,label='Equal-mass common shape')
    axs[0].plot(uc,np.exp(-uc*uc/2)/np.sqrt(2*np.pi)/(ndtr(2)-ndtr(-2)),color=GRAY,ls=':',lw=1.8,label='Standard Gaussian')
    axs[1].plot(uc,p,'k--',lw=1.8)
    axs[1].plot(uc,np.exp(-uc*uc/2)/np.sqrt(2*np.pi),color=GRAY,ls=':',lw=1.8)
    axs[0].set(xlim=(-2,2),ylim=(0,.50),ylabel='Density conditioned on |u| < 2',title='Cores aligned by fitted center and width')
    axs[0].legend(frameon=False,loc='lower center')
    axs[1].set(xlim=(-8,8),yscale='log',ylim=(1e-5,.5),ylabel='Density with full selected normalization',title='Full tails remain in the normalization')
    for ax in axs:ax.set_xlabel(r'$u=(m_{ee}-c_{core})/\sigma_{core}$')
    colorbar(fig,axs);save(fig,'core_aligned_overlays')

    fig,axs=plt.subplots(1,3,figsize=(10.7,3.35),layout='constrained')
    for ax,m in zip(axs,[60,100,160]):
        e,y=hist(m);r=D.loc[m];ue=np.linspace(-12,12,481);uc=(ue[:-1]+ue[1:])/2
        p=np.diff(cdf(e,y,r.center_MeV+r.sigma_core_MeV*ue))/np.diff(ue)
        ax.plot(uc,np.where(p>0,p,np.nan),color=BLUE,lw=1.4,label=f'2016: {100*r.core_fraction_2:.1f}% inside |u| < 2')
        a=np.load(B/'inputs/reference_2021'/f'm{m:03d}.npz')
        meta=json.loads((B/'inputs/reference_2021'/f'm{m:03d}.json').read_text())
        ee=a['edges_GeV']*1000;yy=a['sumw'];rr=R.loc[m]
        ff=lambda zz: np.interp(zz,ee,np.r_[0,np.cumsum(yy)]/meta['sumw'],left=0,right=yy.sum()/meta['sumw'])
        pp=np.diff(ff(rr.core_center_MeV+rr.fitted_core_sigma_MeV*ue))/np.diff(ue)
        frac=float(np.diff(ff(rr.core_center_MeV+rr.fitted_core_sigma_MeV*np.array([-2,2])))[0])
        ax.plot(uc,np.where(pp>0,pp,np.nan),color=RED,lw=1.2,label=f'2021: {100*frac:.1f}% inside |u| < 2')
        ax.plot(uc,np.exp(-uc**2/2)/np.sqrt(2*np.pi),':',color=GRAY,label='Standard Gaussian')
        ax.set(xlim=(-8,10),ylim=(1e-4,.5),yscale='log',title=f'{m} MeV samples',xlabel=r'$u=(m_{ee}-c_{core})/\sigma_{core}$',ylabel='Full selected density')
        ax.legend(frameon=False,loc='lower left',fontsize=7)
    save(fig,'cross_year_shapes')

    s=pd.read_csv(B/'results/shape_comparisons.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.7,3.8),layout='constrained')
    for ax,part in zip(axs,['full','core']):
        ax.plot(s.mass_MeV,100*s[f'gaussian_{part}_cdf_distance'],'o-',ms=3,color=GRAY,label='Shifted Gaussian, fitted core width')
        ax.plot(s.mass_MeV,100*s[f'common_loo_{part}_cdf_distance'],'s-',ms=3,color=BLUE,label='Common shape; tested mass omitted')
        if part=='full':ax.plot(s.mass_MeV,100*s.morph_full_cdf_distance,'^-',ms=3,color=GREEN,label='Neighbor morph; tested mass omitted')
        ax.set(xlabel='Omitted generated mass (MeV)',ylabel='Maximum CDF difference (percentage points)',title='Full distribution' if part=='full' else 'Core conditioned on |u| < 2',ylim=(0,None))
        ax.legend(frameon=False,loc='upper right',fontsize=7)
    save(fig,'shape_holdout')

    fig,axs=plt.subplots(1,2,figsize=(10.7,3.8),layout='constrained')
    for col,key,lab in [(RED,'pole_ref_fraction_2p25','Generated center; analysis width'),(BLUE,'core_ref_fraction_2p25','Fitted center; analysis width'),(GREEN,'core_fitted_fraction_2p25','Fitted center; fitted core width')]:
        axs[0].plot(M,100*D.loc[M,key],color=col,lw=1.2,label=lab)
    axs[0].axhline(100*(ndtr(2.25)-ndtr(-2.25)),color=GRAY,ls=':',label='Matched Gaussian: 97.56%')
    axs[0].set(title='Probability inside a +/-2.25-width window',xlabel='Generated mass (MeV)',ylabel='Full selected probability (%)',ylim=(90,99))
    axs[0].legend(frameon=False,loc='lower right',fontsize=7)
    for m,col in [(30,RED),(35,BLUE),(40,GREEN)]:
        e,y=hist(m);x=(e[:-1]+e[1:])/2
        axs[1].step(x-m,y/y.sum()/np.diff(e),where='mid',color=col,lw=1.2,label=f'{m} MeV; N = {int(y.sum()):,}')
    axs[1].set(title='Low-mass exceptions shown explicitly',xlim=(-8,25),ylim=(1e-4,.4),yscale='log',xlabel='Reconstructed minus generated mass (MeV)',ylabel=r'Density (MeV$^{-1}$)')
    axs[1].legend(frameon=False,loc='upper right')
    save(fig,'containment_and_low_mass')

    # Catalogue: every native sample, including the failed 30-MeV core locator.
    allm=D.index.to_list()
    for page,start in enumerate(range(0,len(allm),10),1):
        fig,axs=plt.subplots(5,2,figsize=(9.5,10.8),layout='constrained')
        for ax,m in zip(axs.flat,allm[start:start+10]):
            e,y=hist(m);r=D.loc[m];x=(e[:-1]+e[1:])/2;dx=np.diff(e)
            ax.step(x,y/y.sum()/dx,where='mid',color=GRAY,lw=.75,label='Native MC')
            gp=np.diff(ndtr((e-m)/r.sigma_ref_MeV))/dx
            ax.plot(x,gp,color=RED,lw=1.,ls=':',label='Pole / analysis width')
            if r.valid:
                gc=np.diff(ndtr((e-r.center_MeV)/r.sigma_core_MeV))/dx
                ax.plot(x,gc,color=BLUE,lw=1.,ls='--',label='Core / fitted width')
            ax.axvline(m,color=RED,lw=.65,ls=':')
            ax.set(xlim=(max(0,m-6*r.sigma_ref_MeV),min(250,m+8*r.sigma_ref_MeV)),
                   ylim=(2e-6,.6),yscale='log',title=f'{m} MeV; N = {int(y.sum()):,}'+(' (core fit invalid)' if not r.valid else ''),
                   xlabel='Reconstructed mass (MeV)',ylabel=r'Density (MeV$^{-1}$)')
            ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.title.set_size(9)
        for ax in list(axs.flat)[len(allm[start:start+10]):]:ax.axis('off')
        handles,labels=axs.flat[1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        save(fig,f'catalogue_{page}')
    print('Saved 11 figure pairs (PDF and PNG).')

if __name__=='__main__':main()
