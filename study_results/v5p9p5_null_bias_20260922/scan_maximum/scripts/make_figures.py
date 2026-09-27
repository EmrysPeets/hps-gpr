import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']='/tmp/hps-v595-mpl'
from pathlib import Path
import numpy as np
import json,csv
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures'
s=json.loads((B/'results/summary.json').read_text());f=dict(np.load(B/'inputs/2021.npz'));p=np.load(B/'results/paired_maxima_2021.npz');m=f['masses'];a=f['a'];scale=f['s'];v=f['validation'];n=len(v);j=int(np.argmax(f['observed_r']));obs=float(f['observed_r'][j]);peak=s['scopes']['2021']['peak']
rows=[r for r in csv.DictReader((B/'results/local_moments_all_scopes.csv').open()) if r['scope']=='2021']
col=lambda key:np.array([float(r[key]) for r in rows])
BLUE='#2167a0';ORANGE='#c56c21';TEAL='#148070';PURPLE='#7c5295';GRAY='#454b55'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14,'pdf.fonttype':42})
def save(fig,name):
 for ext in ('pdf','png'):fig.savefig(F/(name+'.'+ext),dpi=190,bbox_inches='tight')
 plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(10.5,8.6),sharex=True)
axs[0].plot(m,a,c=ORANGE,lw=1.7,label=r'Deterministic response $a=r(B)$')
axs[0].fill_between(m,col('mean_ci95_lo'),col('mean_ci95_hi'),color=BLUE,alpha=.17,label='Pointwise 95% mean interval')
axs[0].plot(m,col('direct_signed_mean'),c=BLUE,lw=1,label='Mean of 256 signed-root scans')
axs[0].axhline(0,c=GRAY,lw=.8,ls=':');axs[0].set(ylabel='Signed-root mean',title='A. The source response is mass dependent, with both signs')
axs[0].legend(ncol=3,loc='upper right')
crit=s['scopes']['2021']['model_simultaneous_mean_check']['critical95'];band=crit*scale/np.sqrt(n)
axs[1].fill_between(m,-band,band,color=TEAL,alpha=.16,label='95% simultaneous envelope under response model')
axs[1].plot(m,col('direct_mean_minus_a'),c=BLUE,lw=1.2,label=r'Direct mean minus $a$')
axs[1].axhline(0,c=GRAY,lw=.9);axs[1].set(ylabel=r'$\overline{r}_{\rm direct}-a$',title='B. No extra scan-wide mean discrepancy is resolved in this conditional check')
axs[1].legend(loc='upper right',ncol=2)
axs[2].fill_between(m,col('sd_ci95_normal_lo')/scale,col('sd_ci95_normal_hi')/scale,color=BLUE,alpha=.17,label='Pointwise 95% normal-theory interval')
axs[2].plot(m,col('direct_signed_sd')/scale,c=BLUE,lw=1.1,label=r'Direct standard deviation / model $s$')
axs[2].axhline(1,c=GRAY,lw=.9,ls='--');axs[2].set(ylabel='Width ratio',xlabel='Tested mass [MeV]',title='C. Scale agreement has finite-toy uncertainty; pointwise bands are not simultaneous',ylim=(.73,1.32))
axs[2].legend(loc='lower right',ncol=2)
for ax in axs:ax.set_xlim(50,250);ax.axvline(78,c='.65',lw=.7,ls=':')
fig.suptitle('2021 native 10% | 50–250 MeV | fixed source and ±2.25σ window',fontsize=13)
fig.tight_layout(rect=(0,0,1,.968),h_pad=1.55);save(fig,'2021_local_response_moments')
# Three distinct random variables, using same source and raw observation.
fig,axs=plt.subplots(1,3,figsize=(12.8,4.15))
x=np.linspace(-3,4.5,500)
axs[0].hist(v[:,j],bins=np.linspace(-3,4.5,24),density=True,color=BLUE,alpha=.3,label='256 direct refits')
axs[0].plot(x,norm.pdf(x,loc=a[j],scale=scale[j]),c=BLUE,lw=2,label='Response model')
axs[0].plot(x,norm.pdf(x),c=GRAY,ls=':',lw=1.4,label='N(0,1) reference')
axs[0].axvline(a[j],c=ORANGE,lw=1.4,ls='--');axs[0].set(title='A. Signed root at 78 MeV',xlabel='r',ylabel='Density',xlim=(-3,4.5));axs[0].legend(loc='upper left',fontsize=8)
axs[0].text(.97,.28,'Raw asymptotic local p = .00249\nFixed-source marginal p = .01138',transform=axs[0].transAxes,ha='right',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
x=np.linspace(0,4.5,500);cdf=norm.cdf((x-a[j])/scale[j]);z=np.maximum(v[:,j],0);zs=np.sort(z)
axs[1].step(zs,np.arange(1,n+1)/n,where='post',c=TEAL,lw=1.2,label='Direct positive-part ECDF')
axs[1].plot(x,cdf,c=BLUE,lw=2,label='Clipped response model');axs[1].plot([0,0],[0,cdf[0]],c=BLUE,lw=2)
axs[1].plot(x,norm.cdf(x),c=GRAY,ls=':',label='Clipped N(0,1) reference');axs[1].plot([0,0],[0,.5],c=GRAY,ls=':')
axs[1].set(title='B. One-sided Z = max(r, 0)',xlabel='Z at 78 MeV',ylabel='Cumulative probability',xlim=(-.12,4),ylim=(0,1.025));axs[1].legend(loc='lower right')
axs[1].text(.96,.40,f'P(Z = 0) = {norm.cdf(-a[j]/scale[j]):.3f}\nE[Z] = {col("model_clipped_mean")[j]:.3f}',transform=axs[1].transAxes,ha='right',fontsize=10)
maxima=f['gaussian_raw_maximum'];bins=np.sort(np.unique(np.r_[np.linspace(.5,5.2,57),obs]));h,edges=np.histogram(maxima,bins=bins,density=True);centers=(edges[:-1]+edges[1:])/2
axs[2].stairs(h,edges,fill=True,color=BLUE,alpha=.32,label='200,000 saved fields');axs[2].stairs(np.where(centers>=obs,h,0),edges,fill=True,color=ORANGE,alpha=.8)
axs[2].hist(np.maximum(v.max(1),0),bins=np.linspace(.5,5.2,21),density=True,histtype='step',color=TEAL,lw=1.3,label='256 direct maxima')
axs[2].axvline(obs,c=ORANGE,lw=1.5);axs[2].set(title='C. Maximum over 401 masses',xlabel=r'$T=\max_m\max(r(m),0)$',ylabel='Density',xlim=(.5,5.2));axs[2].legend(loc='upper right')
axs[2].text(.96,.55,f'Observed T = {obs:.3f}\nConditional tail = .251',transform=axs[2].transAxes,ha='right',fontsize=10)
fig.suptitle('A positive maximum is not, by itself, a biased signed estimator',fontsize=14)
fig.text(.5,-.015,'78 MeV was selected as the observed scan maximum; these marginal probabilities are diagnostic, not an independent fixed-mass claim.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,0,1,.94),w_pad=1.55);save(fig,'2021_signed_clipped_and_maximum')
# Paired counterfactual survival curves, always evaluated against original raw observation.
fig,axs=plt.subplots(1,2,figsize=(11.6,4.5),sharey=True)
labels={'nominal':'Nominal a, s, R','remove_mean_only':'Remove a only','unit_scale_only':'Set s = 1 only','zero_mean_unit_scale':'a = 0 and s = 1','independent_mass_nodes':'Independent masses, same a and s','perfectly_correlated_nodes':'Perfect correlation, same a and s'}
colors={'nominal':GRAY,'remove_mean_only':BLUE,'unit_scale_only':ORANGE,'zero_mean_unit_scale':PURPLE,'independent_mass_nodes':ORANGE,'perfectly_correlated_nodes':TEAL}
xx=np.linspace(0,5.3,800)
for ax,names,title in zip(axs,[['nominal','remove_mean_only','unit_scale_only','zero_mean_unit_scale'],['nominal','independent_mass_nodes','perfectly_correlated_nodes']],['A. Mean and scale controls (same mass correlations)','B. Correlation controls (same mass marginals)']):
 for name in names:
  z=np.sort(p[name]);tail=(len(z)-np.searchsorted(z,xx,side='left'))/len(z);pv=np.mean(z>=obs)
  ax.plot(xx,tail,c=colors[name],lw=2,label=f'{labels[name]}: {pv:.3f}')
 ax.axvline(obs,c='black',ls=':',lw=1);ax.set(title=title,xlabel='Raw threshold t',xlim=(0,5.3),ylim=(-.01,1.02));ax.legend(loc='upper right')
 axs[0].set_ylabel(r'$P(T^*\geq t)$')
fig.suptitle('Paired null controls: the observed raw maximum stays fixed at 2.809',fontsize=13)
fig.text(.5,-.012,'Legend numbers are tails at the vertical line. Controls are diagnostic alternatives, not proposed production probabilities.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,0,1,.94),w_pad=1.5);save(fig,'2021_paired_maximum_controls')
print('Created three figures')
