"""Plot new v5.5.4 comparisons exclusively from saved numerical products."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
B=Path(__file__).resolve().parents[1]
s=pd.read_csv(B/'derived/extracted_combination_scan.csv');r=pd.read_csv(B/'derived/rate_significance_scan.csv');bands=pd.read_csv(B/'derived/rate_bands.csv');fit=json.loads((B/'derived/rate_band_fits.json').read_text());cont=pd.read_csv(B/'derived/rate_parameter_contours.csv');x=s.mass_MeV
C={'common':'#333333','independent':'#18889c','power':'#825099','exponential':'#43844a','stouffer':'#cc742d','fisher':'#a28c2d'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42,'savefig.dpi':175})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(B/f'figures/{name}.{ext}',bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
def sidak(p):return -np.expm1(35.381377775674636*np.log1p(-np.asarray(p)))

fig,axs=plt.subplots(3,2,figsize=(12,7.4),sharex=True,sharey='row')
axs[0,0].plot(x,s.common_GLS_total_yield_upper90/1e3,c=C['common'],label='Common coupling: Gaussian CLs')
axs[0,0].plot(x,s.common_Poisson_total_yield_upper90/1e3,c='#9d7155',ls='--',label='Poisson profile CLs')
axs[0,1].plot(x,s.independent_GLS_total_yield_upper90/1e3,c=C['independent'],label='Independent rates: total-yield CLs')
axs[0,1].plot(x,s.common_GLS_total_yield_upper90/1e3,c=C['common'],ls=':',label='Common-coupling reference')
for i in [1,2]:
 conv=(lambda a:a) if i==1 else sidak
 axs[i,0].plot(x,conv(s.common_GLS_p_local),c=C['common'],label='Shared coupling')
 axs[i,0].plot(x,conv(s.common_Poisson_p_local_reference),c='#9d7155',ls='--',label='Poisson profile reference')
 for p,key,label in [(s.independent_GLS_p_local,'independent','Independent positive rates'),(s.signed_Stouffer_p_local,'stouffer','Equal-weight signed Stouffer')]:axs[i,1].plot(x,conv(p),c=C[key],label=label)
 for kind,label in [('power','Fitted power law'),('exponential','Fitted exponential')]:
  rr=r[r.model==kind];axs[i,1].plot(rr.mass_MeV,conv(rr.p_gaussian_statistic),c=C[kind],label=label)
 axs[i,0].set_yscale('log')
axs[0,0].set_title('Shared coupling');axs[0,1].set_title('Rates unconstrained or fitted to another law')
axs[0,0].set_ylabel('90% CLs upper bound\non total signal rows [$10^3$]')
axs[1,0].set_ylabel('Local Gaussian-reference p');axs[2,0].set_ylabel('Sidák equivalent p\nillustrative fixed N = 35.381')
for a in axs[0]:a.legend(frameon=False,fontsize=8.5,loc='upper left');a.set_ylim(10,49)
axs[1,1].legend(frameon=False,fontsize=8.4,loc='upper left');axs[1,0].set_ylim(1e-5,1);axs[2,0].set_ylim(2e-4,1)
for a in axs.flat:a.axvline(92,c='.65',ls=':',lw=.7);a.set_xlim(90,94);a.set_xticks(np.arange(90,94.1,1))
for a in axs[2]:a.set_xlabel('Fixed mass hypothesis [MeV]')
fig.tight_layout();save(fig,'coupling_vs_unconstrained_limits_pvalues')

fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,kind,title in zip(axs,['common','power','exponential'],['Common coupling',r'Power law: $a(E/2.3)^{\beta}$',r'Exponential: $a e^{-k(E-2.3)}$']):
 b=bands[bands.model==kind]
 ax.fill_between(b.energy_GeV,b.lower_95*1e6,b.upper_95*1e6,color=C[kind],alpha=.12,label='Approx. 95% pointwise')
 ax.fill_between(b.energy_GeV,b.lower_68*1e6,b.upper_68*1e6,color=C[kind],alpha=.28,label='Approx. 68% pointwise')
 ax.plot(b.energy_GeV,b.central_epsilon2*1e6,c=C[kind],lw=1.6,label='Best fit')
 for j,p in enumerate(fit['independent_dataset_estimates']):ax.errorbar(p['beam_energy_GeV'],p['epsilon2']*1e6,yerr=p['curvature_error_epsilon2']*1e6,fmt='o',c=['#2675a6','#b64237','#4b7d40'][j],capsize=4,label=p['year'])
 ax.set(title=title,xlabel='Beam energy [GeV]',xticks=[1.056,2.3,3.74],yscale='log',ylim=(.12,240));ax.legend(frameon=False,fontsize=8,loc='upper right')
axs[0].set_ylabel(r'Fitted $\epsilon^2_{\rm equiv}$ [$10^{-6}$]');fig.tight_layout();save(fig,'rate_fits_with_uncertainties')

fig,axs=plt.subplots(2,2,figsize=(12,6.3),sharex=True,sharey='row')
for col,mode in enumerate(['Poisson','GLS']):
 axs[0,col].plot(x,np.sqrt(s['common_'+mode+'_Q']),c=C['common'],label='Common coupling')
 axs[0,col].plot(x,s['independent_'+mode+'_raw_root'],c=C['independent'],label='Independent positive rates')
 cp=s.common_Poisson_p_local_reference if mode=='Poisson' else s.common_GLS_p_local;ip=s.independent_Poisson_p_chibar_reference if mode=='Poisson' else s.independent_GLS_p_local
 axs[1,col].plot(x,norm.isf(cp),c=C['common']);axs[1,col].plot(x,norm.isf(ip),c=C['independent'])
 for kind,label in [('power','Fitted power law'),('exponential','Fitted exponential')]:
  rr=r[r.model==kind];axs[0,col].plot(x,np.sqrt(rr.Q_exact if mode=='Poisson' else rr.Q_gaussian),c=C[kind],label=label);axs[1,col].plot(x,rr.Z_reference_at_exact_Q if mode=='Poisson' else rr.Z_gaussian_statistic,c=C[kind])
 axs[0,col].set_title('Simultaneous fit to count spectra' if col==0 else 'Simultaneous fit to extracted signals')
 axs[1,col].plot(x,s.signed_Stouffer_Z,c=C['stouffer'],ls='--',label='Signed Stouffer (extracted)');axs[1,col].plot(x,s.signed_Fisher_Z_equivalent,c=C['fisher'],ls=':',label='Fisher (extracted)')
 axs[0,col].legend(frameon=False,fontsize=8.3,loc='lower right');axs[1,col].legend(frameon=False,fontsize=8.3,loc='lower right');axs[1,col].set_xlabel('Fixed mass hypothesis [MeV]')
axs[0,0].set_ylabel(r'Raw likelihood root $\sqrt{Q_0}$');axs[1,0].set_ylabel('Local Gaussian-reference Z')
for a in axs.flat:a.set_xlim(90,94);a.set_ylim(.7,4.6);a.set_yticks([1,2,3,4]);a.axvline(92,c='.65',ls=':',lw=.7)
fig.tight_layout();save(fig,'spectra_vs_extracted_significance')

fig,axs=plt.subplots(1,2,figsize=(11.8,3.5),layout='constrained')
for ax,kind in zip(axs,['power','exponential']):
 c=cont[cont.model==kind];tab=c.pivot(index='slope',columns='amplitude_epsilon2_at_2p3GeV',values='delta_2nll');fm=fit['models'][kind]
 levels=[fm['joint_likelihood_contour_thresholds']['68_percent_2d'],fm['joint_likelihood_contour_thresholds']['95_percent_2d']]
 ax.contourf(tab.columns*1e6,tab.index,tab.to_numpy(),levels=[0,*levels],colors=[C[kind],C[kind]],alpha=.22)
 cs=ax.contour(tab.columns*1e6,tab.index,tab.to_numpy(),levels=levels,colors=[C[kind]],linewidths=[1.6,1.0]);ax.clabel(cs,fmt={levels[0]:'68%',levels[1]:'95%'},fontsize=9)
 ax.plot(fm['amplitude_epsilon2_at_2p3GeV']*1e6,fm['slope'],'x',c='k',ms=8)
 ax.set(xlabel=r'Amplitude at 2.30 GeV [$10^{-6}$]',ylabel=r'Power exponent $\beta$' if kind=='power' else r'Decay slope $k$ [GeV$^{-1}$]',title='Power-law parameters' if kind=='power' else 'Exponential parameters')
save(fig,'rate_joint_parameter_contours')

q=s[s.mass_MeV==92].iloc[0];rr=r[r.mass_MeV==92].set_index('model');rows=[]
for name,root,p,z in [('Common coupling',np.sqrt(q.common_GLS_Q),q.common_GLS_p_local,q.common_GLS_signed_Z),('Independent positive amplitudes',q.independent_GLS_raw_root,q.independent_GLS_p_local,q.independent_GLS_Z_equivalent),('Fitted power law',np.sqrt(rr.loc['power'].Q_gaussian),rr.loc['power'].p_gaussian_statistic,rr.loc['power'].Z_gaussian_statistic),('Fitted exponential',np.sqrt(rr.loc['exponential'].Q_gaussian),rr.loc['exponential'].p_gaussian_statistic,rr.loc['exponential'].Z_gaussian_statistic),('Equal-weight signed Stouffer',np.nan,q.signed_Stouffer_p_local,q.signed_Stouffer_Z),('Fisher of signed one-sided probabilities',np.nan,q.signed_Fisher_p_local,q.signed_Fisher_Z_equivalent)]:
 mant,exp=f'{p:.2e}'.split('e');rows.append(name+' & '+(f'{root:.3f}' if np.isfinite(root) else '--')+f' & ${mant}\\times10^{{{int(exp)}}}$ & {z:.3f}'+r'\\')
(B/'derived/new_92_combinations.tex').write_text(r'\begin{tabular}{lrrr}\toprule Model or rule & $\sqrt{Q_G}$ & Local $p_G$ & $Z_G$\\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n')
print('Four new figures and the 92 MeV comparison table written.')
