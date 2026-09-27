from pathlib import Path
import os, json, hashlib
os.environ['MPLCONFIGDIR']='/tmp/hps-rc-slides-mpl'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import FancyArrowPatch
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent/'assets'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'axes.titlesize':14,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#1a62b0'; ORANGE='#d66a24'; TEAL='#087d78'; PURPLE='#78469a'; GREY='#525866'
manifest=[]
def save(fig,name,source,kind):
    for ext in ('png','pdf','svg'):
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=280,bbox_inches='tight',facecolor='white')
    manifest.append(dict(asset=name,source=source,kind=kind,sha256_png=hashlib.sha256((OUT/(name+'.png')).read_bytes()).hexdigest()))
    plt.close(fig)
# Slide9: analytic illustrations only, no data or optimized fit.
x=np.linspace(0,5,350)
fig,axs=plt.subplots(1,3,figsize=(12,3.6))
for ell,c in [(0.6,ORANGE),(1.2,BLUE),(2.0,TEAL)]:axs[0].plot(x,np.exp(-x*x/(2*ell*ell)),lw=2.8,c=c,label=rf'$\ell={ell:g}$')
axs[0].set(title='Length scale: correlation range',xlabel='Separation in log-mass (arbitrary units)',ylabel=r'$k(x,x\prime)/C$',ylim=(0,1.05));axs[0].legend(frameon=False)
for C,c in [(0.5,ORANGE),(1,BLUE),(2,TEAL)]:axs[1].plot(x,C*np.exp(-x*x/2),lw=2.8,c=c,label=rf'$C={C:g}$')
axs[1].set(title='Constant: covariance amplitude',xlabel='Separation in log-mass (arbitrary units)',ylabel=r'$k(x,x\prime)$',ylim=(0,2.12));axs[1].legend(frameon=False)
a=np.logspace(-2,1,250)
axs[2].semilogx(a,1/(1+a),c=BLUE,lw=3)
axs[2].set(title=r'$\alpha$: bin noise and influence',xlabel=r'Bin-noise variance $\alpha$ (fixed $C=1$)',ylabel='Response weight',ylim=(0,1.05))
axs[2].text(.09,.2,r'$C/(C+\alpha)$',transform=axs[2].transAxes,fontsize=15,color=GREY)
fig.text(.5,-.015,'Illustration of separate GP roles; not a fit to HPS data. In the analysis: x = log m and α ≈ 1/y.',ha='center',fontsize=12,color=GREY)
fig.tight_layout(w_pad=2.2)
save(fig,'slide09_hyperparameter_roles','v5.0.5 04_methodology.tex kernel and alpha definitions','analytic illustration')
# Slide21: verified published 65MeV pull means and 90% intervals.
fig,axs=plt.subplots(1,2,figsize=(10.8,3.65),sharex=True)
series=[('1% × 10: threshold refinement',[.7890,.1394],[[.6293,.9488],[-.0499,.3288]],['Baseline model','Threshold-refined'],BLUE),('Native 10%: extended support',[.7160,-.2463],[[.5452,.8867],[-.4165,-.0761]],['Baseline model','Threshold + 30–300 MeV'],TEAL)]
for ax,(title,vals,cis,labels,col) in zip(axs,series):
    ax.axvspan(-.5,.5,color='#eaf1f7',zorder=0);ax.axvline(0,c=GREY,lw=1,ls='--')
    for y,(v,(lo,hi)) in enumerate(zip(vals,cis)):
        ax.errorbar(v,1-y,xerr=[[v-lo],[hi-v]],fmt='o',ms=9,lw=2.5,c=ORANGE if y==0 else col,capsize=5)
        ax.text(v,1-y+.13,f'{v:+.3f}',ha='center',fontsize=12,color=GREY)
    ax.set(yticks=[1,0],yticklabels=labels,ylim=(-.5,1.55),xlim=(-.65,1.07),xlabel='Mean extraction pull (90% interval)',title=title)
    ax.grid(axis='y',visible=False)
fig.text(.5,-.025,'65 MeV, 100 backgrounds per ensemble. Shading: post-result practical ±0.5 band; not a coverage test.',ha='center',fontsize=12,color=GREY)
fig.tight_layout(w_pad=2.5)
save(fig,'slide21_threshold_comparison','v5.0.5 05_toys_validation.tex Table v491-sixtyfive-results','published numerical summary')
# Slide22: unitless matched injection convention. No implied data or measured Z.
u=np.linspace(-5,5,501);template=np.exp(-u*u/2)/np.sqrt(2*np.pi)
fig,ax=plt.subplots(figsize=(6.2,4.5))
for z,c in [(1,TEAL),(3,BLUE),(5,ORANGE)]:ax.plot(u,z*template,c=c,lw=3,label=rf'$A_{{\rm inj}}={z}\,\sigma_{{A,\rm ref}}$')
ax.axvspan(-2.25,2.25,color='#e9eff5',zorder=0,label='Blind / extraction window')
ax.set(xlabel=r'Mass offset $(m-m_0)/\sigma_m$',ylabel='Added signal per unit offset / σA,ref',title='One template, fixed injection strengths',xlim=(-5,5),ylim=(0,2.3))
ax.legend(frameon=False,loc='upper right',fontsize=10)
fig.text(.51,-.025,'Illustration; area under each curve gives its injected yield.',ha='center',fontsize=11,color=GREY)
fig.tight_layout()
save(fig,'slide22_matched_injection','v5.0.5 04_methodology.tex eq v4p9-matched-reference-axis','analytic illustration')
# Slide26: exactly specified Wald example for positive amplitude with background-like best fit.
A=np.linspace(0,3.5,600);cls=2*norm.sf(A);aul=norm.isf(.05)
fig,ax=plt.subplots(figsize=(6,3.3))
ax.plot(A,cls,c=BLUE,lw=3,label=r'Illustration: $CL_s=2[1-\Phi(A/\sigma_A)]$')
ax.axhline(.1,c=ORANGE,ls='--',lw=2);ax.plot([aul,aul],[0,.1],c=ORANGE,ls=':',lw=2);ax.scatter([aul],[.1],c=ORANGE,s=60,zorder=4)
ax.annotate(r'$A_{90}=1.645\,\sigma_A$',xy=(aul,.1),xytext=(1.86,.34),fontsize=14,arrowprops=dict(arrowstyle='->',color=GREY))
ax.set(xlabel=r'Tested signal yield $A/\sigma_A$',ylabel=r'$CL_s(A)$',xlim=(0,3.5),ylim=(0,1.02))
ax.text(.03,.95,'Schematic Wald example, Â = 0',transform=ax.transAxes,va='top',fontsize=12)
ax.text(3.42,.13,'0.10',ha='right',color=ORANGE,fontsize=12)
fig.tight_layout()
save(fig,'slide26_cls_crossing','v5.0.5 04_methodology.tex CLs definition; explicit Gaussian-Wald illustration with Ahat=0','analytic illustration; not observed HPS profile')
# Slides49/50: exact pinned saved 2021 field. No fits, new toys or calibration.
fpath=ROOT/'study_results/v5p8p5p3_raw_significance_20260921/inputs/fields/2021.npz'
f=np.load(fpath);m=f['masses'];r=f['observed_r'];K=f['K'];v=f['validation'];rawmax=f['gaussian_raw_maximum'];j=np.argmax(r);tobs=max(r[j],0)
assert len(rawmax)==200000 and v.shape==(256,401)
fig,axs=plt.subplots(2,1,figsize=(6.2,4.9),gridspec_kw={'height_ratios':[1,1.05]})
im=axs[0].imshow(K,origin='lower',extent=[m[0],m[-1],m[0],m[-1]],vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto',interpolation='nearest')
axs[0].set(title='2021 fitted-response correlations',xlabel='Tested mass [MeV]',ylabel='Tested mass [MeV]');axs[0].grid(False)
cb=fig.colorbar(im,ax=axs[0],pad=.025,fraction=.035);cb.set_label('Correlation',fontsize=11)
for k in range(5):axs[1].plot(m,v[k],lw=1.1,alpha=.8)
axs[1].axhline(0,c=GREY,lw=.8)
axs[1].set(title='Five saved background-only Poisson scans',xlabel='Tested mass [MeV]',ylabel='Raw signed root r',xlim=(50,250))
fig.tight_layout(h_pad=1.15)
save(fig,'slide49_response_correlations','v5.8.5.3 inputs/fields/2021.npz K and validation[0:5]','archived conditional null responses')
fig,axs=plt.subplots(2,1,figsize=(6.2,4.9),gridspec_kw={'height_ratios':[1,1.15]})
axs[0].plot(m,np.maximum(r,0),c=TEAL,lw=1.7);axs[0].scatter(m[j],tobs,c=ORANGE,s=40,zorder=4)
axs[0].annotate(f'78 MeV: Zlocal = {tobs:.3f}',xy=(78,tobs),xytext=(109,2.6),fontsize=12,arrowprops=dict(arrowstyle='->',color=GREY))
axs[0].set(title='Observed local scan: raw root, no shifts',xlabel='Tested mass [MeV]',ylabel='Local Z',xlim=(50,250),ylim=(0,3.25))
bins=np.linspace(.6,5.6,70);counts,edges=np.histogram(rawmax,bins=bins,density=True);centers=(edges[:-1]+edges[1:])/2
axs[1].stairs(counts,edges,color=BLUE,fill=True,alpha=.3)
axs[1].stairs(np.where(centers>=tobs,counts,0),edges,color=ORANGE,fill=True,alpha=.8)
axs[1].axvline(tobs,color=ORANGE,lw=2)
p=(np.count_nonzero(rawmax>=tobs)+1)/(len(rawmax)+1)
axs[1].text(.97,.91,rf'$p_{{global}}={p:.3f}$',transform=axs[1].transAxes,ha='right',fontsize=15,color=ORANGE)
axs[1].set(title='Null distribution of the full-scan maximum',xlabel=r'$T^*=\max_m\max(r^*(m),0)$',ylabel='Density',xlim=(.6,5.6))
fig.tight_layout(h_pad=1.2)
save(fig,'slide50_local_to_global','v5.8.5.3 inputs/fields/2021.npz observed_r and 200000 saved gaussian_raw_maximum','archived raw observation and source-conditional global tail')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Created',len(manifest),'figures in',OUT)
