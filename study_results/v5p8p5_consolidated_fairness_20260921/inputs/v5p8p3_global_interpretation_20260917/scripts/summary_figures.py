"""Reproduce interpretation figures from the pinned v5.8.2 fields; no fits."""
from pathlib import Path
import os,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v583-mpl')
import numpy as np
import pandas as pd
from scipy.stats import norm,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
B=Path(__file__).resolve().parents[1];I=B/'inputs/v5p8p2';F=B/'figures';R=B/'results'
labels={'2015':'2015 full','2016':'2016 full','2021':'2021 10%','combined':'Combined'}
colors=['#326a91','#ad5147','#348276','#795494']
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
s=pd.read_csv(I/'results/summary.csv',dtype={'scope':str})
def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=165)
    plt.close(fig)
def cp(k,n):
    return (0 if k==0 else beta.ppf(.025,k,n-k+1),1 if k==n else beta.ppf(.975,k+1,n-k))

# Compare the SAME mass in each scope, avoiding a misleading peak shift.
fig,ax=plt.subplots(1,2,figsize=(10.6,4.2),gridspec_kw={'width_ratios':[1.15,1]})
y=np.arange(4)
for j,r in s.iterrows():
    ax[0].plot([r.peak_global_Z,r.peak_local_Z],[j,j],c=colors[j],lw=2)
    ax[0].scatter(r.observed_r,j,marker='s',s=47,c='.6',zorder=4,label='Unshifted root at this mass' if j==0 else None)
    ax[0].scatter(r.peak_local_Z,j,marker='o',s=57,c='black',zorder=5,label='Reference local Z' if j==0 else None)
    ax[0].scatter(r.peak_global_Z,j,marker='D',s=45,c='#b32d36',zorder=5,label='Full-range global Z' if j==0 else None)
ax[0].set(yticks=y,yticklabels=[f'{labels[r.scope]}\n{r.mass_MeV:g} MeV' for _,r in s.iterrows()],xlim=(0,4.7),ylim=(-1.1,3.55),xlabel='Positive one-sided Z',title='At each reference-local peak');ax[0].invert_yaxis();ax[0].legend(fontsize=8.5,loc='upper right')
for j,r in s.iterrows():
    p=r.global_p_addone
    ax[1].errorbar(p,j,xerr=[[p-r.global_lo95],[r.global_hi95-p]],fmt='D',c=colors[j],capsize=3)
    ax[1].errorbar(r.direct_global_p,j+.16,xerr=[[r.direct_global_p-r.direct_global_lo95],[r.direct_global_hi95-r.direct_global_p]],fmt='o',c='.25',capsize=3)
ax[1].set(yticks=y,yticklabels=list(labels.values()),xlim=(0,.40),ylim=(-1.1,3.55),xlabel='Global background tail probability',title='Independent check with complete scans');ax[1].invert_yaxis()
ax[1].plot([],[],marker='D',c=colors[0],ls='none',label='200,000 GP fields');ax[1].plot([],[],marker='o',c='.25',ls='none',label='256 Poisson scans');ax[1].legend(loc='upper right',fontsize=9)
fig.tight_layout(w_pad=2.5);save(fig,'peak_summary')

# Same combined statistic and same observed threshold, different declared domains.
q=s[s.scope=='combined'].iloc[0];z=np.load(I/'fields/combined.npz');u=q.peak_local_Z
domain=[]
for name,key in [('Full 19-250 MeV','gaussian_maximum'),('Common 50-100 MeV','gaussian_overlap_maximum')]:
    arr=z[key];k=int(np.sum(arr>=u));n=len(arr);lo,hi=cp(k,n);p=(k+1)/(n+1)
    domain.append(dict(domain=name,threshold=u,k=k,N=n,p=p,lo95=lo,hi95=hi,Z=norm.isf(p)))
pd.DataFrame(domain).to_csv(R/'combined_domain_comparison.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(10.3,4.0))
grid=np.linspace(1.5,4.1,100)
for d,key,c in zip(domain,['gaussian_maximum','gaussian_overlap_maximum'],['#795494','#328479']):
    arr=z[key];p=np.array([(np.sum(arr>=v)+1)/(len(arr)+1) for v in grid]);ax[0].semilogy(grid,p,c=c,label=d['domain'])
    ax[0].scatter(u,d['p'],c=c,zorder=5)
ax[0].semilogy(grid,norm.sf(grid),c='black',ls='--',label='One specified mass');ax[0].axvline(u,c='.5',ls=':',lw=1);ax[0].set(xlabel='Local reference-score threshold',ylabel='Probability of an exceedance',ylim=(1e-5,1.1),title='Same field; larger domain adds opportunities');ax[0].legend(fontsize=9)
vs=[u,domain[1]['Z'],domain[0]['Z']];names=['Specified 66 MeV','Search 50-100 MeV','Search 19-250 MeV']
ax[1].barh(np.arange(3),vs,color=['.25','#328479','#795494'],height=.55)
for j,v in enumerate(vs):ax[1].text(v+.06,j,f'{v:.2f}',va='center')
ax[1].set(yticks=np.arange(3),yticklabels=names,xlabel='Z at the same 66 MeV excess',xlim=(0,3.4),title='The question determines the global domain');ax[1].invert_yaxis()
fig.tight_layout(w_pad=2);save(fig,'combined_domain_effect')

# A dimensionless illustration, never presented as recomputed HPS limits.
x=np.linspace(0,4,201);p0=norm.sf(x);upper=x+norm.isf(.1*norm.cdf(x))
test=[dict(observed_estimate_sigma=v,p0=norm.sf(v),upper90_sigma=v+norm.isf(.1*norm.cdf(v))) for v in [0,1,2,3]]
pd.DataFrame(test).to_csv(R/'gaussian_cls_illustration.csv',index=False)
assert np.max(abs(norm.sf(upper-x)/norm.cdf(x)-.1))<1e-14
fig,ax=plt.subplots(1,2,figsize=(10.3,3.65))
ax[0].semilogy(x,p0,c='#b32d36',lw=2);ax[0].axhline(.1,c='.5',ls=':');ax[0].set(xlabel=r'Observed estimate $\hat\mu/\sigma$',ylabel=r'Background excess tail $p_0$',title='Discovery asks about background',ylim=(2e-5,.7))
ax[1].plot(x,upper,c='#326a91',lw=2,label='90% CLs upper endpoint');ax[1].plot(x,x,c='.6',ls='--',label='Observed estimate');ax[1].scatter(0,upper[0],c='#326a91');ax[1].annotate('1.645 at zero excess',xy=(0,upper[0]),xytext=(.5,2.5),arrowprops={'arrowstyle':'->','color':'.4'},fontsize=10);ax[1].set(xlabel=r'Observed estimate $\hat\mu/\sigma$',ylabel=r'Tested signal upper endpoint $\mu_{90}/\sigma$',title='Exclusion asks about a signal strength');ax[1].legend(fontsize=9,loc='lower right')
fig.suptitle('Illustrative Gaussian measurement, known variance; not new HPS limits',fontsize=12,y=1.01);fig.tight_layout();save(fig,'cls_vs_discovery')

# Coarse/fine uses paired fields, not independent toy samples.
grid=pd.read_csv(I/'results/grid_convergence.csv',dtype={'scope':str});v=grid[grid.threshold==3]
fig,ax=plt.subplots(1,2,figsize=(10.4,3.7))
for j,(_,r) in enumerate(v.iterrows()):
    ax[0].plot([1,.5],[r.coarse_p,r.fine_p],'-o',c=colors[j],label=labels[r.scope])
ax[0].set(xticks=[.5,1],xlim=(1.1,.4),ylim=(.055,.185),xlabel='Mass-grid spacing [MeV]',ylabel='P(scan maximum >= 3)',title='Finer sampling finds missed maxima');ax[0].legend(fontsize=9,ncol=2,loc='upper left')
truth=pd.read_csv(B/'inputs/v5p8p1_asimov_summary.csv');keys=['archived_stress','gp_full_nominal','gp_full_half_ls'];tt=truth.set_index('truth').loc[keys]
vals=tt.rms_root.to_numpy();ax[1].bar(np.arange(3),vals,color=['.6','#348276','#ae7551'])
for j,vv in enumerate(vals):ax[1].text(j,vv+.07,f'{vv:.2f}',ha='center')
ax[1].set(xticks=np.arange(3),xticklabels=['Archived\nstress','Nominal GP\nsource','Half-length\nGP source'],ylabel='RMS deterministic signed root',ylim=(0,5.7),title='2016: the generating reference matters')
fig.tight_layout(w_pad=2.5);save(fig,'grid_and_source_choices')

# Visual map of the two distinct GPs and of the separate CLs branch.
fig,ax=plt.subplots(figsize=(10.5,5.1));ax.set(xlim=(0,10.5),ylim=(0,5.1));ax.axis('off')
def box(x,y,w,h,text,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=c,edgecolor='.6',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'.35','lw':1.4})
box(.12,3.95,2.2,.82,'Observed count spectra\n2015 / 2016 / 2021','#e8edf1')
box(3.02,3.95,3.0,.82,'Background GP mean B\nFixed generating reference','#dceee9')
box(7.02,3.95,3.25,.82,'Poisson-response directions D\nOffsets a; response widths s','#e6e2ef')
arrow(2.4,4.35,2.93,4.35);arrow(6.1,4.35,6.93,4.35)
box(.12,2.1,2.2,1.05,'Reviewed likelihood scan\nObserved signed root r\nJoint fit: one coupling','#e8edf1')
box(3.02,2.1,3.0,1.05,'Reference-local score\nz = (r - a) / s\nPositive-fit excess gate','#f3e8e2')
box(7.02,2.1,3.25,1.05,'Significance GP: correlation R\n200,000 correlated fields\nDistribution of scan maxima','#e6e2ef')
arrow(1.22,3.86,1.22,3.23);arrow(2.4,2.63,2.93,2.63);arrow(8.65,3.86,8.65,3.23);arrow(7,2.63,6.1,2.63)
box(.12,.3,2.2,.87,'90% CLs limits\nSeparate signal-tail ratio','#e0eaf4')
box(3.02,.3,3.0,.87,'Local / global probabilities\nConditional on fixed B','#f3e8e2')
box(7.02,.3,3.25,.87,'256 complete Poisson scans\nValidate response and maxima','#eeefdf')
arrow(1.22,2.01,1.22,1.26);arrow(4.52,2.01,4.52,1.26);arrow(8.65,2.01,8.65,1.26)
fig.tight_layout();save(fig,'method_map')
(B/'qa/summary_figures.json').write_text(json.dumps({'passed':True,'new_figures':5,'new_HPS_fits':0,'CLs_illustration_root_max_error':float(np.max(abs(norm.sf(upper-x)/norm.cdf(x)-.1)))},indent=2)+'\n')
print('Built five summary figures and numerical ledgers')
