"""Regenerate introductory figures from copied v6.1 signal-MC data; no fits."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
P=B/'provenance/intro_signal_mc'; F=B/'figures'
P.mkdir(parents=True,exist_ok=True); F.mkdir(exist_ok=True)
# Bootstrap this self-contained provenance directory on first authoring only.
old=B.parent/'v6p1_mc_signal_templates_20260922'
if not (P/'core_native_diagnostics.csv').exists():
    for name in ['core_native_diagnostics.csv','analytic_shift_models.json','shared_shape_metrics.csv','shared_shape_heldout.csv']:
        shutil.copy2(old/'derived'/name,P/name)
    for m in range(60,261,20):
        for ext in ['npz','json']:
            shutil.copy2(old/'histograms'/f'm{m:03d}.{ext}',P/f'm{m:03d}.{ext}')
    shutil.copy2('/var/folders/4p/8kgp9mqx0fl8ps906wt6v8400000gn/T/codex-clipboard-7de7fc3e-b5c7-45ef-b375-424963c16a31.png',P/'hps_internal_user_original.png')
C=pd.read_csv(P/'core_native_diagnostics.csv').set_index('mass_MeV')
C=C.loc[(C.index>=60)&(C.index<=240)]
models=json.loads((P/'analytic_shift_models.json').read_text())
a,b=next(x['coefficients'] for x in models['models'] if x['model']=='logarithmic')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
blue='#245c91';red='#a54b35';gray='#333333'
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(F/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)
def sig(m):return 1000*np.polynomial.polynomial.polyval(m/1000,[.00184825,-.001375,.085875])
def source(m):
    d=np.load(P/f'm{m:03d}.npz'); e=d['edges_GeV']*1000; h=d['sumw']
    meta=json.loads((P/f'm{m:03d}.json').read_text());total=meta['sumw']
    assert meta['stats']['underflow']==0 and h.sum()+meta['stats']['overflow']==total
    return e,h,total
# The image is embedded without cropping, alteration or digitization.
fig,axs=plt.subplots(1,2,figsize=(11,4.35),gridspec_kw={'width_ratios':[1,1.12]},layout='constrained')
x=np.linspace(60,240,400)
axs[0].errorbar(C.index,C.center_shift_MeV,yerr=C.center_MC_bin_resample_std_MeV,fmt='o',color=gray,ms=4,label='TC signal MC core position')
axs[0].plot(x,a+b*np.log(x/150),color=blue,lw=1.6,label='Saved logarithmic relation')
axs[0].axhline(0,color=gray,lw=.7,ls='--')
axs[0].set(xlabel='Generated signal mass $m$ (MeV)',ylabel='Reconstructed core shift $c-m$ (MeV)',title='(a) 2021 target-constrained (TC) signal MC',xlim=(50,250),ylim=(-5,.25))
axs[0].text(.04,.12,r'$c-m=-3.224-2.214\ln[m/(150\,\mathrm{MeV})]$'+'\nShift in MeV; descriptive fit on 60–240 MeV',transform=axs[0].transAxes,fontsize=8)
axs[0].legend(loc='upper right',frameon=False)
axs[1].imshow(plt.imread(P/'hps_internal_user_original.png'));axs[1].set_axis_off();axs[1].set_title('(b) HPS Internal comparison supplied by the author')
fig.text(.01,-.055,'Left bars: core-position SD from 32 signal-MC bin-resampling replicas (often smaller than markers).\nRight panel: original HPS image; its error definition was not supplied. Qualitative comparison only.',fontsize=8)
save(fig,'intro_core_shift_comparison')
# Compare original signal MC to the actual analysis-width Gaussian models.
fig,axs=plt.subplots(4,3,figsize=(10.6,10.7),layout='constrained')
for ax,m in zip(axs.flat,range(60,261,20)):
    e,h,total=source(m); x=(e[:-1]+e[1:])/2; dx=np.diff(e)
    ax.step(x,h/total/dx,where='mid',color=gray,lw=.8,label='Signal MC')
    g=np.diff(ndtr((e-m)/sig(m)))
    ax.plot(x,g/dx,color=red,lw=1.25,label='Central mass Gaussian')
    if m<=240:
        c=m+a+b*np.log(m/150)
        g=np.diff(ndtr((e-c)/sig(m)))
        ax.plot(x,g/dx,color=blue,lw=1.25,ls='--',label='Shifted Gaussian')
    ax.set(xlim=(max(0,m-9*sig(m)),min(400,m+13*sig(m))),ylim=(1e-5,.3),yscale='log',title=f'Generated mass {m} MeV'+(' — shape only' if m==260 else ''),xlabel=r'Reconstructed $m_{ee}$ (MeV)',ylabel=r'Density (MeV$^{-1}$)')
    ax.tick_params(labelsize=8)
handles,labels=axs[0,0].get_legend_handles_labels();axs[-1,-1].axis('off');axs[-1,-1].legend(handles,labels,loc='upper left',frameon=False,fontsize=10)
axs[-1,-1].text(0,.60,'Signal MC divides by ALL selected\nevents, including overflow.\nGaussians use their full integral.\nLocal views are not renormalized.\n\nBoth Gaussians use the analysis\nresolution width, not the narrower\nfitted signal-MC core width.\n\nShift relation used only on\nits 60–240 MeV fitted domain.',transform=axs[-1,-1].transAxes,fontsize=9,va='top',linespacing=1.35)
fig.suptitle('2021 target-constrained signal MC: Gaussian models and reconstructed distributions',fontsize=13)
save(fig,'intro_signal_mc_catalogue')
# Inspect common core shape using empirical saved centres AND fitted core widths.
fig,axs=plt.subplots(1,2,figsize=(10.6,4.55),layout='constrained')
u=np.linspace(-30,60,1801);uc=(u[:-1]+u[1:])/2;du=np.diff(u);pool=[]
colors=plt.cm.viridis(np.linspace(.12,.9,9))
for m,col in zip(range(60,241,20),[red,*colors]):
    e,h,total=source(m);r=C.loc[m];cdf=np.r_[0,np.cumsum(h)]/total
    q=np.interp(r.core_center_MeV+r.fitted_core_sigma_MeV*u,e,cdf,left=0,right=cdf[-1])
    prob=np.diff(q)/du;pool.append(q)
    bounds=np.interp(r.core_center_MeV+r.fitted_core_sigma_MeV*np.array([-2,2]),e,cdf)
    norm=bounds[1]-bounds[0]
    axs[0].plot(uc,prob/norm,color=col,lw=1.4 if m==60 else .8,label=f'{m} MeV')
    axs[1].semilogy(uc,np.maximum(prob,1e-9),color=col,lw=1.4 if m==60 else .8)
q=np.mean(pool,axis=0);p=np.diff(q)/du;core=np.interp(2,u,q)-np.interp(-2,u,q)
axs[0].plot(uc,p/core,'k--',lw=1.7,label='Equal-mass common shape')
axs[0].plot(uc,np.exp(-uc**2/2)/np.sqrt(2*np.pi)/(ndtr(2)-ndtr(-2)),':',color='#777777',lw=1.7,label='Standard Gaussian (core)')
axs[1].semilogy(uc,np.maximum(p,1e-9),'k--',lw=1.7)
axs[0].set(xlim=(-2,2),ylim=(0,.49),ylabel='Density conditioned on $|u|<2$',title='(a) Can one shape describe the signal core?')
axs[1].set(xlim=(-12,25),ylim=(1e-5,.5),ylabel='Density with original tail probabilities',title='(b) Similar cores do not imply identical tails')
for ax in axs:ax.set_xlabel(r'$u=(m_{ee}-c)/\sigma_{\mathrm{core}}$');ax.axvline(-2,color=gray,ls=':',lw=.7);ax.axvline(2,color=gray,ls=':',lw=.7)
axs[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(1.07,-.20),fontsize=8,frameon=False)
fig.text(.01,-.17,'c and σcore: fitted signal-MC core centre and width. Right: probabilities divide by all selected events, including overflow. No toy bars.',fontsize=8)
save(fig,'intro_common_core_shape')
manifest=[]
for p in sorted(P.iterdir()):
    if p.is_file() and p.name!='source_sha256.json':manifest.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(P/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved intro_core_shift_comparison, intro_signal_mc_catalogue, intro_common_core_shape')
