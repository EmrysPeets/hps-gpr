"""Audit saved v5.8.2 fields. No fits, random draws, or changed significance inputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import norm,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
OLD=B/'inputs/v5p8p2'
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':8,'savefig.dpi':170,'pdf.fonttype':42})
SCOPES=['2015','2016','2021','combined'];LABELS={'2015':'2015 full','2016':'2016 full','2021':'2021, 10%','combined':'Combined, 19–250 MeV'}
LIMITS={'2015':(19,100),'2016':(39,180),'2021':(50,250)}
DATA={y:dict(np.load(OLD/f'inputs/spectrum_{y}.npz')) for y in SCOPES[:3]}
def sigma(y,m): return 1000*np.polynomial.polynomial.polyval(np.asarray(m)/1000,DATA[y]['sigma_coeffs'])
def widths(s,m):
 if s!='combined':return sigma(s,m)
 # Deliberately only a heuristic: finest available detector resolution, not a joint-fit resolution.
 return np.min(np.array([np.where((m>=lo)&(m<=hi),sigma(y,m),np.inf) for y,(lo,hi) in LIMITS.items()]),axis=0)
def save(fig,name):
 fig.savefig(B/f'figures/{name}.pdf',bbox_inches='tight');fig.savefig(B/f'figures/{name}.png',bbox_inches='tight');plt.close(fig)
def interval(k,n):
 return np.where(k==0,0,beta.ppf(.025,k,n-k+1)),np.where(k==n,1,beta.ppf(.975,k+1,n-k))
summary=pd.read_csv(OLD/'results/summary.csv').set_index('scope');rows=[];tails=[];lags=[];fields={}
for scope in SCOPES:
 f=dict(np.load(OLD/f'fields/{scope}.npz'));fields[scope]=f;m=f['masses'];K=f['K'];s=widths(scope,m);adj=np.diag(K,1);eig=np.linalg.eigvalsh(K);r=summary.loc[scope];mx=f['gaussian_maximum'];N=len(mx);FWHM=2.3548200450309493
 assert float(np.max(-f['a']/f['s'])) < 1.5, 'Plotted thresholds must exceed the positive-fit gate at every mass'
 assert np.count_nonzero(mx>=4.5)>0, 'High-end tail needs nonzero saved exceedances'
 Cresponse=np.sum(np.arccos(np.clip(adj,-1,1)))/(2*np.pi)
 # Smooth Gaussian-template benchmark: ||du/dm||=sqrt(1+sigma_prime^2)/(sqrt(2)*sigma).
 Ctemplate=float(np.trapezoid(np.sqrt(1+np.gradient(s,m)**2)/(np.sqrt(2)*s),m)/(2*np.pi)) if scope!='combined' else None
 row={'scope':scope,'mass_min_MeV':float(m.min()),'mass_max_MeV':float(m.max()),'grid_nodes':len(m),'peak_mass_MeV':float(r.mass_MeV),'peak_local_Z':float(r.peak_local_Z),'peak_local_p':float(r.peak_local_p),'peak_global_p':float(r.global_p_addone),'peak_tail_equivalent_independent_tests':float(np.log1p(-r.global_p_addone)/np.log1p(-r.peak_local_p)),'spectral_participation_rank_not_extreme_trial_count':float(eig.sum()**2/np.sum(eig**2)),'maximum_positive_fit_gate_Z':float(np.max(-f['a']/f['s'])),'saved_exceedances_at_Z_4p5':int(np.count_nonzero(mx>=4.5)),'median_adjacent_correlation':float(np.median(adj)),'minimum_adjacent_correlation':float(adj.min()),'minimum_adjacent_left_mass_MeV':float(m[np.argmin(adj)]),'median_sigma_MeV':float(np.median(s)),'minimum_sigma_MeV':float(s.min()),'maximum_sigma_MeV':float(s.max()),'FWHM_spacing_count_heuristic':float(np.trapezoid(1/(FWHM*s),m)),'sigma_spacing_count_heuristic':float(np.trapezoid(1/s,m)),'adjacent_arc_path_length_over_2pi':float(Cresponse),'Gaussian_template_upcrossing_coefficient':Ctemplate,'arc_path_to_template_ratio':Cresponse/Ctemplate if Ctemplate else None,'minimum_nonadjacent_correlation':float(K.min()),'native_bin_width_MeV':float(np.median(np.diff(DATA[scope]['x']))*1000) if scope!='combined' else None}
 # Compute adjacent-mask changes and pairwise response correlations, preserving native bin geometry.
 if scope!='combined':
  masks=np.abs(DATA[scope]['x'][None,:]*1000-m[:,None])<=2.25*s[:,None]
  switches=np.sum(masks[:-1]!=masks[1:],axis=1)
  row.update(adjacent_pairs_with_mask_change=int(np.sum(switches>0)),adjacent_pairs_with_identical_mask=int(np.sum(switches==0)),median_bins_switching_per_step=float(np.median(switches)))
  if np.any(switches==0):row['median_rho_identical_mask']=float(np.median(adj[switches==0]))
 row['peak_resolution_only_upcrossing_approx_p']=float(min(1.,r.peak_local_p+Ctemplate*np.exp(-r.peak_local_Z**2/2))) if Ctemplate else None
 row['peak_FWHM_spacing_heuristic_p']=float(-np.expm1(row['FWHM_spacing_count_heuristic']*np.log1p(-r.peak_local_p)))
 row['peak_independent_grid_p']=float(-np.expm1(len(m)*np.log1p(-r.peak_local_p)))
 rows.append(row)
 for step in range(1,min(41,len(m))):
  sep=m[step:]-m[:-step];rho=np.diag(K,step)
  if scope!='combined':
   s0=s[:-step];s1=s[step:];benchmark=np.sqrt(2*s0*s1/(s0*s0+s1*s1))*np.exp(-sep**2/(2*(s0*s0+s1*s1)))
  else:benchmark=np.full_like(rho,np.nan)
  lags.append({'scope':scope,'lag_MeV':float(np.mean(sep)),'response_mean_rho':float(rho.mean()),'response_p10_rho':float(np.quantile(rho,.1)),'response_p90_rho':float(np.quantile(rho,.9)),'Gaussian_template_mean_rho':float(np.mean(benchmark)) if scope!='combined' else None})
 for z in np.arange(1.5,4.5001,.05):
  pl=float(norm.sf(z));k=int(np.sum(mx>=z));pg=(k+1)/(N+1);lo,hi=interval(np.array(k),N)
  tails.append({'scope':scope,'local_Z':float(z),'local_p':pl,'global_k':k,'global_N':N,'global_p_addone':pg,'global_p_lo95':float(lo),'global_p_hi95':float(hi),'global_Z':float(norm.isf(pg)),'tail_equivalent_independent_tests':float(np.log1p(-pg)/np.log1p(-pl)) if pg<1 else None,'independent_grid_p':float(-np.expm1(len(m)*np.log1p(-pl))),'FWHM_spacing_heuristic_p':float(-np.expm1(row['FWHM_spacing_count_heuristic']*np.log1p(-pl))),'Gaussian_template_upcrossing_approx_p':min(1.,pl+Ctemplate*np.exp(-z*z/2)) if Ctemplate else None})
R=pd.DataFrame(rows);T=pd.DataFrame(tails);L=pd.DataFrame(lags)
R.to_csv(B/'results/resolution_audit.csv',index=False,float_format='%.12g');T.to_csv(B/'results/trials_curve_audit.csv',index=False,float_format='%.12g');L.to_csv(B/'results/response_correlation_lag.csv',index=False,float_format='%.12g')
meta={'source':'v5.8.2 frozen conditional nominal-GP fields','new_fits':0,'new_random_fields':0,'definition_tail_equivalent_tests':'log(1-p_global)/log(1-p_local); threshold dependent; not a literal count of independent mass positions','definition_spectral_rank':'(tr K)^2/tr(K^2); covariance dimension, not an extreme-value trials factor','FWHM_heuristic':'integral dm/(2.35482 sigma_m); combined uses finest available detector resolution at each mass and is only an illustration','Gaussian_template_benchmark':'Normalized equal-width Gaussian templates with known flat background give rho(delta)=exp(-delta^2/(4 sigma^2)). Here exact pairwise overlap with varying sigma is used. This omits moving sidebands and GP background estimation.','upcrossing_benchmark':'p approximately sf(z)+C exp(-z^2/2), C=integral sqrt(1+sigma_prime^2)/(2 pi sqrt(2) sigma) dm. Smooth Gaussian-template high-threshold approximation only; no combined curve.','response_path_length':'Sum arccos(adjacent rho)/(2 pi), a finite-grid angular length diagnostic; not a calibrated trials factor or continuum derivative estimate.','scope_selection':'Each global probability covers its own declared grid. There is no extra calibration for choosing a scope after inspection.','rows':rows,'source_hashes':{f'fields/{s}.npz':hashlib.sha256((OLD/f'fields/{s}.npz').read_bytes()).hexdigest() for s in SCOPES}}
(B/'results/resolution_audit.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(2,2,figsize=(10,8.4),layout='constrained')
for ax,scope in zip(axes.flat,SCOPES):
 f=fields[scope];m=f['masses'];im=ax.imshow(f['K'],origin='lower',extent=[m[0],m[-1],m[0],m[-1]],vmin=-1,vmax=1,cmap='RdBu_r',aspect='equal');ax.set(title=LABELS[scope],xlabel='Mass hypothesis (MeV)',ylabel='Mass hypothesis (MeV)')
 if scope=='combined':
  for cut in (39,50,100,180):ax.axhline(cut,c='k',lw=.4,alpha=.4);ax.axvline(cut,c='k',lw=.4,alpha=.4)
fig.colorbar(im,ax=list(axes.flat),label='Fitted-score correlation',shrink=.82);fig.suptitle('The global scan uses correlated responses across mass',fontsize=15);save(fig,'correlation_resolution')
fig,axes=plt.subplots(2,2,figsize=(10,7.8),sharex=True,sharey=True,layout='constrained')
for ax,scope in zip(axes.flat,SCOPES):
 d=L[L.scope==scope];ax.fill_between(d.lag_MeV,d.response_p10_rho,d.response_p90_rho,color='#4477AA',alpha=.17,label='10–90% across mass pairs');ax.plot(d.lag_MeV,d.response_mean_rho,c='#225588',label='Full fitted response: mean')
 if scope!='combined':ax.plot(d.lag_MeV,d.Gaussian_template_mean_rho,c='#CC6677',ls='--',label='Gaussian signal overlap only')
 ax.axhline(0,c='.6',lw=.6);ax.set(title=LABELS[scope],xlabel='Separation of mass hypotheses (MeV)',ylabel='Correlation',xlim=(0,20),ylim=(-1,1.02));ax.legend(loc='upper right')
fig.suptitle('Detector resolution explains only part of the scan correlation',fontsize=14);save(fig,'response_resolution')
fig,axes=plt.subplots(2,2,figsize=(10.5,8.5),sharex=True,sharey=True,layout='constrained')
for ax,scope in zip(axes.flat,SCOPES):
 d=T[T.scope==scope];r=R[R.scope==scope].iloc[0];ax.fill_between(d.local_Z,d.global_p_lo95,d.global_p_hi95,color='#225588',alpha=.18);ax.plot(d.local_Z,d.global_p_addone,c='#225588',lw=2,label='Saved correlated field (95% MC band)');ax.plot(d.local_Z,d.independent_grid_p,c='.5',ls=':',label=f'Independent grid: N = {r.grid_nodes}');ax.plot(d.local_Z,d.FWHM_spacing_heuristic_p,c='#EE9944',ls='--',label=f'FWHM-spacing heuristic: N = {r.FWHM_spacing_count_heuristic:.1f}')
 if scope!='combined':ax.plot(d.local_Z,d.Gaussian_template_upcrossing_approx_p.where(d.local_Z>=2.5),c='#228833',ls='-.',label='Smooth-template high-threshold approximation')
 ax.plot(d.local_Z,d.local_p,c='.2',lw=.8,label='One fixed mass');ax.scatter([r.peak_local_Z],[r.peak_global_p],marker='o',s=35,color='#AA3377',zorder=4);ax.set(title=LABELS[scope],xlabel='Reference-local threshold Z',ylabel='Probability of an exceedance',yscale='log',ylim=(1e-5,1.1),xlim=(1.5,4.5));ax.grid(alpha=.15);ax.legend(loc='lower left')
fig.suptitle('Correlated trials correction versus resolution-based comparisons',fontsize=14);save(fig,'trials_comparison')
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
colors=['#4477AA','#CC6677','#228833','#AA3377']
for scope,c in zip(SCOPES,colors):
 d=T[T.scope==scope];axes[0].fill_between(d.local_Z,np.log1p(-d.global_p_lo95)/np.log1p(-d.local_p),np.log1p(-d.global_p_hi95)/np.log1p(-d.local_p),color=c,alpha=.10);axes[0].plot(d.local_Z,d.tail_equivalent_independent_tests,label=LABELS[scope],c=c);r=R[R.scope==scope].iloc[0];axes[0].scatter([r.peak_local_Z],[r.peak_tail_equivalent_independent_tests],color=c,s=24)
x=np.arange(4);w=.23
for k,col,label,c in [(0,'grid_nodes','Grid nodes (not independent)','#BBBBBB'),(1,'peak_tail_equivalent_independent_tests','Tail-equivalent N at observed peak','#4477AA'),(2,'spectral_participation_rank_not_extreme_trial_count','Spectral rank (not a trials count)','#EE9944')]:axes[1].bar(x+(k-1)*w,R[col],w,label=label,color=c)
axes[0].set(xlabel='Reference-local threshold Z',ylabel='Tail-equivalent independent tests',xlim=(1.5,4.5));axes[0].legend();axes[0].grid(alpha=.15);axes[1].set(xticks=x,xticklabels=['2015','2016','2021\n10%','Combined'],ylabel='Count or dimension (distinct definitions)');axes[1].legend();fig.suptitle('There is no single threshold-independent number of trials',fontsize=14);save(fig,'trials_effective_counts')
print(R.to_string(index=False))
