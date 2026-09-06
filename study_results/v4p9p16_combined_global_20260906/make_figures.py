#!/usr/bin/env python3
"""Publication figures: full observed limit with local/global p-values below."""
from pathlib import Path
import hashlib,json,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'figures'
BLUE,RED,GREEN,ORANGE,PURPLE='#18699C','#AE3E30','#1C7044','#C98925','#7B4EA3'
SEGMENTS=[(19,38,'2015'),(39,49,'2015\n+2016'),(50,90,'All three'),
          (91,180,'2016 + 2021'),(181,250,'2021')]
BACKS=['#DDEAF3','#D8E7DD','#E8E0F0','#F2E5D5','#E5E5E5']
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':210})
inventory=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name):
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig);inventory.append(name)
def boundaries(ax):
    for m in (38.5,49.5,90.5,180.5):ax.axvline(m,color='.65',ls=':',lw=.7,zorder=0)
    ax.set_xlim(18.5,250.5)
    ax.grid(axis='y',alpha=.18)
def probability_axis(ax,label,floor):
    boundaries(ax);ax.set(yscale='log',ylim=(floor,1.5),ylabel=label)
def clipped_local(ax,x,y,**kwargs):
    floor=1e-8;ax.plot(x,np.maximum(y,floor),**kwargs)
    low=np.asarray(y)<floor
    ax.scatter(np.asarray(x)[low],np.full(np.count_nonzero(low),floor),marker='v',
               color=kwargs.get('color','black'),s=13,clip_on=False,zorder=5)

def main():
    f=HERE/'global';a=f/'analysis'
    obs=pd.read_csv(f/'observed.csv')
    curves=pd.read_csv(a/'pvalue_curves.csv')
    d=curves[curves.method=='profiled'].sort_values('mass_MeV')
    diag=pd.read_csv(a/'marginal_diagnostics.csv')
    tails=pd.read_csv(a/'maximum_tail_curve.csv')
    summary=json.loads((a/'summary.json').read_text())
    cov=np.load(a/'covariance.npz')
    fig=plt.figure(figsize=(10.2,10.5))
    grid=fig.add_gridspec(4,1,height_ratios=(.22,2.5,1.55,1.55),
        left=.125,right=.985,bottom=.065,top=.88,hspace=.23)
    strip=fig.add_subplot(grid[0])
    ul=fig.add_subplot(grid[1],sharex=strip)
    local=fig.add_subplot(grid[2],sharex=strip)
    glob=fig.add_subplot(grid[3],sharex=strip)
    for (lo,hi,label),color in zip(SEGMENTS,BACKS):
        strip.add_patch(Rectangle((lo-.5,0),hi-lo+1,1,facecolor=color,edgecolor='white'))
        strip.text((lo+hi)/2,.5,label,ha='center',va='center',fontsize=8,fontweight='semibold')
    strip.set(ylim=(0,1),xlim=(18.5,250.5))
    strip.set_ylabel('Active\ndata',rotation=0,ha='right',va='center',fontsize=8)
    strip.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
    for spine in strip.spines.values():spine.set_visible(False)
    ul.plot(obs.mass_MeV,obs.v12_eps2_display,color='.64',ls='--',lw=1.2,label='v4.9.12 reference')
    ul.plot(obs.mass_MeV,obs.profiled_eps2_display,color='black',lw=1.8,label='Observed, dense profiled likelihood')
    ul.axvline(2*105.6583745,color=PURPLE,ls='-.',lw=1.05,label=r'Dimuon threshold $2m_\mu$')
    ul.set(yscale='log',ylabel=r'90% CL$_s$ upper limit on $\epsilon^2$')
    ul.set_title(r'Pointwise observed limit: asymptotic CL$_s$',loc='left',fontsize=11,fontweight='semibold')
    boundaries(ul);ul.legend(loc='upper center',fontsize=8.5,frameon=False)
    clipped_local(local,d.mass_MeV,d.p_asymptotic,color='.5',lw=1.2,ls='--',label='Asymptotic local (dense fit)')
    clipped_local(local,d.mass_MeV,d.p_local_common_truth,color=BLUE,lw=1.5,label='Common-background Gaussian local')
    probability_axis(local,'Local tail probability',1e-8)
    local.set_title('Local probabilities from the combined likelihood',loc='left',fontsize=11,fontweight='semibold')
    local.legend(loc='lower right',fontsize=8.3,frameon=False)
    y=np.where(d.gp_k>0,d.p_global_gp,np.nan)
    glob.plot(d.mass_MeV,y,color=RED,lw=1.5,label='GP global: minimum-local-p ordering')
    zero=d.gp_k==0
    glob.scatter(d.loc[zero,'mass_MeV'],d.loc[zero,'p_global_gp_upper95'],s=15,marker='v',color=RED,zorder=6)
    glob.plot(d.mass_MeV,np.where(d.raw_gp_k>0,d.p_global_raw_ordering,np.nan),
              color=ORANGE,ls=':',lw=1.6,label='GP global: separate raw-peak ordering')
    zero_raw=d.raw_gp_k==0
    glob.scatter(d.loc[zero_raw,'mass_MeV'],d.loc[zero_raw,'raw_gp_upper95'],
                 s=15,marker='v',color=ORANGE,zorder=6)
    selected=d[d.mass_MeV.isin(summary['representative_masses_MeV'])]
    positive=selected[selected.direct_k>0]
    glob.errorbar(positive.mass_MeV,positive.p_global_direct,
        yerr=np.array([positive.p_global_direct-positive.p_global_direct_low,
                      positive.p_global_direct_high-positive.p_global_direct]),
        fmt='o',ms=3,color=GREEN,elinewidth=1,capsize=2,zorder=7,
        label='Direct: minimum-local-p, 95% intervals')
    zero=selected[selected.direct_k==0]
    glob.scatter(zero.mass_MeV,zero.p_global_direct_upper95,color=GREEN,marker='v',s=25,zorder=8)
    glob.plot([],[],ls='none',marker='v',ms=4,color='.3',label='Zero tails: one-sided 95% upper bounds')
    probability_axis(glob,'Union-global tail probability',1e-6)
    glob.set_title('Look-elsewhere correction over all 232 masses',loc='left',fontsize=11,fontweight='semibold')
    glob.legend(loc='lower right',fontsize=8.1,frameon=False)
    glob.set_xlabel(r'Mass hypothesis $m_{A^\prime}$ [MeV]')
    for ax in (ul,local):ax.tick_params(labelbottom=False)
    fig.suptitle('Combined HPS search over the full mass region',fontsize=15,fontweight='semibold',y=.97)
    fig.text(.5,.938,'2015 full + 2016 full + 2021 10%  |  19–250 MeV  |  1 MeV grid',
             ha='center',fontsize=11)
    fig.text(.5,.913,'Global probabilities condition on one archived joint stress background.',
             ha='center',fontsize=10,color='.3')
    save(fig,'combined_observed_limit_and_pvalues')

    fig,axes=plt.subplots(2,2,figsize=(10.5,6.5),sharex=True)
    for j,method in enumerate(('profiled','fixed')):
        x=diag[diag.method==method].sort_values('mass_MeV')
        axes[0,j].plot(x.mass_MeV,x.asimov_r,color=BLUE,lw=1.5,label='Unfluctuated background')
        axes[0,j].plot(x.mass_MeV,x.toy_r_mean,color=GREEN,lw=1.1,ls='--',label='1,000 joint toy means')
        axes[1,j].plot(x.mass_MeV,x.response_sd,color=BLUE,lw=1.5,label='Asimov response width')
        axes[1,j].plot(x.mass_MeV,x.toy_r_sd,color=GREEN,lw=1.1,ls='--',label='Joint toy spread')
        axes[0,j].set_title(method.capitalize())
        axes[1,j].set_xlabel('Mass [MeV]')
        for ax in axes[:,j]:boundaries(ax)
    axes[0,0].set_ylabel('Mean signed root');axes[1,0].set_ylabel('Standard deviation')
    handles,labels=axes[0,0].get_legend_handles_labels()
    h,l=axes[1,0].get_legend_handles_labels()
    fig.legend(handles+h,labels+l,loc='upper center',ncol=2,fontsize=9,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,top=.87,bottom=.09,hspace=.14,wspace=.18)
    save(fig,'combined_null_response')

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    for ax,method in zip(axes,('profiled','fixed')):
        im=ax.imshow(cov[method+'_K'],origin='lower',extent=(18.5,250.5,18.5,250.5),
                     cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
        ax.set(title=method.capitalize(),xlabel='Mass [MeV]',ylabel='Mass [MeV]')
        for m in (38.5,49.5,90.5,180.5):
            ax.axhline(m,lw=.5,color='.55',ls=':');ax.axvline(m,lw=.5,color='.55',ls=':')
    fig.colorbar(im,ax=axes,shrink=.88,pad=.025,label='Signed-root correlation')
    save(fig,'combined_correlations')

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),sharey=True)
    from scipy.stats import norm
    for ax,method in zip(axes,('profiled','fixed')):
        t=tails[tails.method==method]
        ax.fill_between(t.threshold,np.maximum(t.direct_low,1e-6),t.direct_high,
                        color=GREEN,alpha=.18,label='Direct: pointwise 95% interval')
        ax.plot(t.threshold,np.where(t.gp_k>0,t.gp_p,np.nan),color=RED,lw=1.6,label='200,000 GP fields')
        ax.plot(t.threshold,np.where(t.direct_k>0,t.direct_p,np.nan),color=GREEN,ls='--',lw=1.4,label='1,000 joint Poisson scans')
        z=t[t.direct_k==0]
        if len(z):ax.scatter(z.threshold.iloc[0],z.direct_high.iloc[0],marker='v',color=GREEN,s=30,zorder=5)
        ax.plot(t.threshold,norm.sf(t.threshold),color='.6',ls=':',label='One standard-normal point')
        ax.set(yscale='log',ylim=(1e-5,1.5),xlim=(0,5.5),xlabel='Scan threshold',title=method.capitalize())
        ax.grid(alpha=.16)
    axes[0].set_ylabel('Probability the union scan exceeds threshold')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',ncol=2,fontsize=9,frameon=False)
    fig.subplots_adjust(left=.10,right=.985,top=.81,bottom=.15,wspace=.12)
    save(fig,'combined_tail_validation')

    fig,axes=plt.subplots(2,1,figsize=(10.5,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    axes[0].plot(obs.mass_MeV,obs.v12_eps2_display,color='.55',ls='--',label='v4.9.12 cached solver')
    axes[0].plot(obs.mass_MeV,obs.profiled_eps2_display,color='black',lw=1.5,label='v4.9.16 dense solver')
    axes[0].set(yscale='log',ylabel=r'90% CL$_s$ upper limit on $\epsilon^2$')
    axes[0].legend(frameon=False)
    axes[1].plot(obs.mass_MeV,100*obs.v12_limit_relative_difference,color=BLUE,lw=1.1)
    axes[1].axhline(0,color='.5',lw=.8)
    axes[1].set(xlabel='Mass [MeV]',ylabel='Change from v4.9.12 [%]')
    for ax in axes:boundaries(ax)
    fig.subplots_adjust(left=.11,right=.985,top=.97,bottom=.11,hspace=.08)
    save(fig,'observed_limit_v12_comparison')
    paths=[f/'observed.csv',a/'summary.json',a/'pvalue_curves.csv',a/'marginal_diagnostics.csv',
           a/'maximum_tail_curve.csv',a/'covariance.npz',Path(__file__)]
    record=dict(figures=inventory,inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
                files={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf')})
    (HERE/'provenance/figure_build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(figures=inventory),indent=2))

if __name__=='__main__':main()
