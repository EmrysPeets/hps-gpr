"""Publication figures from the archived v5.8.5 numerical ledgers."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'qa/mpl'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
B=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42,'savefig.dpi':180,'axes.titleweight':'bold','axes.titlelocation':'left'})
COLORS=['#163f70','#d58020','#188575','#a33c62']; WIDTHS=[2.25,2.4,2.5,2.6]
def save(fig,name):
 fig.savefig(B/f'figures/{name}.pdf',bbox_inches='tight');fig.savefig(B/f'figures/{name}.png',bbox_inches='tight');plt.close(fig)
def decorate(ax):
 ax.grid(alpha=.17);ax.set_axisbelow(True)
s=pd.read_csv(B/'results/stability_curves.csv',dtype={'dataset':str})
fig,axs=plt.subplots(3,1,figsize=(8.5,6.8),sharex=True,layout='constrained')
for ax,year in zip(axs,['2015','2016','2021']):
 for w,c in zip(WIDTHS,COLORS):
  a=s[(s.dataset==year)&np.isclose(s.width_sigma,w)&s.mass_MeV.between(80,105)]
  ax.plot(a.mass_MeV,a.signed_Z,label=rf'$\pm{w:g}\sigma_m$',color=c,linewidth=1.6,marker='.',markersize=3)
 ax.axvline(92,color='k',ls=':',lw=.8);ax.axhline(0,color='gray',lw=.7);ax.set_ylabel('Signed reference $Z$');ax.set_title(year+(' (10% sample)' if year=='2021' else ''));decorate(ax)
 if year=='2015':ax.axvspan(100.05,105,color='.93');ax.text(102.5,.92,'Outside\nsupport',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
axs[0].legend(ncol=4,loc='lower left');axs[-1].set_xlabel('Tested mass [MeV]');axs[-1].set_xlim(80,105)
save(fig,'individual_stability')
M=np.arange(80,106); matrix=np.full((12,len(M)),np.nan);labels=[]
for i,(year,w) in enumerate((y,w) for y in ['2015','2016','2021'] for w in WIDTHS):
 a=s[(s.dataset==year)&np.isclose(s.width_sigma,w)].set_index('mass_MeV');matrix[i]=a.signed_Z.reindex(M).values;labels.append(f'{year}  |  {w:g}'+r'$\sigma_m$')
fig,ax=plt.subplots(figsize=(8.6,4),layout='constrained');cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dedede');v=max(3.,np.nanmax(abs(matrix)))
im=ax.imshow(matrix,aspect='auto',extent=(79.5,105.5,11.5,-.5),cmap=cmap,vmin=-v,vmax=v,interpolation='none')
ax.set_yticks(np.arange(12),labels);ax.set_xticks(np.arange(80,106,2));ax.set_xlabel('Tested mass [MeV]');ax.axvline(92,color='black',ls=':',lw=.9)
for y in [3.5,7.5]:ax.axhline(y,color='white',lw=2)
fig.colorbar(im,ax=ax,label='Signed reference $Z$');save(fig,'stability_matrix')
# Primary common-mass test and archived diagnostic combinations.
f=pd.read_csv(B/'results/free_curves.csv');p=pd.read_csv(B/'inputs/parent_results/significance_and_reach.csv');p=p[p.domain=='full']
fig,axs=plt.subplots(2,1,figsize=(8.6,6.6),sharex=True,layout='constrained')
for i,(scope,label) in enumerate([('free','Free amplitudes'),('combined','Shared coupling'),('fisher','Fisher'),('stouffer','Signed Stouffer')]):
 a=f[np.isclose(f.width_sigma,2.25)] if scope=='free' else p[(p.scope==scope)&np.isclose(p.width_sigma,2.25)]
 axs[0].plot(a.mass_MeV,a.local_Z,label=label,color=COLORS[i],lw=1.5)
 axs[1].plot(a.mass_MeV,a.global_p,label=label,color=COLORS[i],lw=1.5)
for ax in axs:
 decorate(ax);ax.axvspan(80,105,color='gray',alpha=.08);ax.axvline(92,color='gray',ls=':',lw=.8)
axs[0].set_ylabel('Local $Z$');axs[0].legend(ncol=2);axs[1].set_ylabel('Full-domain $p$');axs[1].set_yscale('log');axs[1].set_ylim(1e-3,1.05);axs[1].set_xlabel('Tested mass [MeV]');axs[1].set_xlim(19,250)
save(fig,'methods_full_domain')
fig,axs=plt.subplots(2,1,figsize=(8.6,5.8),sharex=True,layout='constrained')
for w,c in zip(WIDTHS,COLORS):
 a=f[np.isclose(f.width_sigma,w)];axs[0].plot(a.mass_MeV,a.local_Z,color=c,label=rf'$\pm{w:g}\sigma_m$',lw=1.6);axs[1].plot(a.mass_MeV,a.global_p,color=c,lw=1.6)
for ax in axs:decorate(ax);ax.axvline(92,color='gray',ls=':',lw=.8)
axs[0].legend(ncol=4);axs[0].set_ylabel('Free-amplitude local $Z$');axs[1].set_ylabel('Full-domain $p$');axs[1].set_yscale('log');axs[1].set_xlabel('Tested mass [MeV]');axs[1].set_xlim(80,105)
save(fig,'free_widths_diagnostic')
fig,axs=plt.subplots(1,2,figsize=(8.5,3.2),layout='constrained')
for w,c in zip(WIDTHS,COLORS):
 a=f[np.isclose(f.width_sigma,w)];axs[0].plot(a.mass_MeV,a.q_R,color=c,label=rf'$\pm{w:g}\sigma_m$',lw=1.3);axs[1].plot(a.mass_MeV,a.q_R,color=c,lw=1.5)
axs[0].set_xlim(19,250);axs[1].set_xlim(80,105);axs[0].set_ylabel(r'Exact profiled $q_R$');axs[0].legend(fontsize=8,ncol=2)
for ax in axs:ax.set_xlabel('Tested mass [MeV]');decorate(ax)
save(fig,'profile_statistic')
print('Saved six publication figure pairs')
import json
co=json.loads((B/'results/stability_coherence_summary.json').read_text());null=np.load(B/'fields/stability_null.npz')
fig,axs=plt.subplots(1,2,figsize=(8.5,3.3),layout='constrained')
for ax,name in zip(axs,['T','W']):
 for key,label,c in [('direct','256 exact Poisson scans',COLORS[0]),('gaussian','100,000 Gaussian scans',COLORS[2])]:
  arr=np.sort(null[f'{key}_{name}']);finite=arr[np.isfinite(arr)];keep=np.arange(len(finite)) if key=='direct' else np.unique(np.linspace(0,len(finite)-1,2500).astype(int))
  ax.step(finite[keep],(keep+1)/len(arr),where='post',label=label,color=c,lw=1.5)
 ax.axvline(co[f'observed_{name}'],color=COLORS[3],ls='--',lw=1.3,label='Observed')
 ax.set_xlabel('Peak scatter $T$' if name=='T' else 'Width instability $W$');ax.set_ylim(0,1.03);decorate(ax)
 ax.set_xlim(0,max(co[f'observed_{name}']*1.25,np.quantile(null[f'gaussian_{name}'][np.isfinite(null[f'gaussian_{name}'])],.97)))
axs[0].set_ylabel('Null cumulative probability');axs[0].legend(fontsize=8,loc='lower right');save(fig,'coherence_null')
c=pd.read_csv(B/'results/free_comparison.csv')
fig,axs=plt.subplots(1,2,figsize=(8.5,3.3),layout='constrained')
for i,(method,label) in enumerate([('free_amplitude','Free amplitudes'),('shared_coupling','Shared coupling'),('fisher','Fisher'),('stouffer','Signed Stouffer')]):
 a=c[c.method==method]
 if a.empty:
  matches=[m for m in c.method.unique() if method.split('_')[0] in m.lower()];a=c[c.method==matches[0]] if matches else a
 axs[0].plot(a.width_sigma,a.local_Z,'o-',color=COLORS[i],label=label);axs[1].plot(a.width_sigma,a.gaussian_global_Z,'o-',color=COLORS[i])
for ax in axs:ax.set_xlabel(r'Blind half-width [$\sigma_m$]');ax.set_xticks(WIDTHS);decorate(ax)
axs[0].set_ylabel('Peak local $Z$');axs[1].set_ylabel('Full-domain global $Z$');axs[0].legend(fontsize=8);save(fig,'method_width_summary')
print('Saved coherence and method-width figure pairs')
