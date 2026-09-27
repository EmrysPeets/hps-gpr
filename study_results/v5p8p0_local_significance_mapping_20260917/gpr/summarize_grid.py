#!/usr/bin/env python3
from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v580-mpl')
import numpy as np,pandas as pd
from scipy.stats import norm,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
FIG=B/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'legend.fontsize':8,'figure.dpi':150,'savefig.bbox':'tight'})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(FIG/(name+'.'+ext))
 plt.close(fig)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def interval(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
d=pd.read_csv(B/'local_mapping.csv');d.dataset=d.dataset.astype(str)
uniform=[]
for year in ['2015','2016','2021']:
 f=np.load(B/f'field_{year}/field.npz');mm=f['masses']
 for step in ([1.,.5] if year=='2016' else [1.]):
  take=np.isclose(mm/step,np.round(mm/step));aa=f['a'][take];ss=f['s'][take];vv=f['validation'][:,take];zz=(vv-aa)/ss;kk=f['K'][np.ix_(take,take)]
  train=vv[:128];hold=(vv[128:]-train.mean(axis=0))/train.std(axis=0,ddof=1)
  uniform.append(dict(dataset=year,step_MeV=step,mass_min_MeV=float(mm[take][0]),mass_max_MeV=float(mm[take][-1]),hypotheses=int(take.sum()),rms_stress_offset=float(np.sqrt(np.mean(aa*aa))),max_abs_stress_offset=float(np.max(abs(aa))),median_abs_stress_offset=float(np.median(abs(aa))),response_sd_min=float(ss.min()),response_sd_max=float(ss.max()),mean_abs_centered_toy_bias=float(np.mean(abs(zz.mean(axis=0)))),average_centered_toy_sd=float(np.mean(zz.std(axis=0,ddof=1))),covariance_correlation_RMSE=float(np.sqrt(np.mean((np.corrcoef(vv,rowvar=False)-kk)**2))),average_holdout_absZ_lt1p96=float(np.mean(abs(hold)<1.96))))
pd.DataFrame(uniform).to_csv(B/'uniform_grid_summary.csv',index=False,float_format='%.17g')
x=np.load(B/'field_2016/field.npz');m=x['masses'];a=x['a'];s=x['s'];obs=x['observed_r'];V=x['validation'];Z=(V-a)/s;zobs=(obs-a)/s
old=np.load(B/'inputs/v511_2016_field.npz')
idx=np.searchsorted(m,old['masses']);invariance={k:float(np.max(abs(x[k][...,idx]-old[k]))) for k in ['a','D','s','validation']}
assert max(invariance.values())<1e-12
modes=[]
for region,lo,hi,steps in [('low',39,50,[1.,.5]),('stress',70,81,[1.,.5]),('quarter',74,79,[1.,.5,.25]),('high',85,100,[1.,.5]),('full',39,180,[1.,.5])]:
 for step in steps:
  mask=(m>=lo)&(m<=hi)
  if step:mask&=np.isclose(m/step,np.round(m/step))
  ind=np.flatnonzero(mask);modes.append((region,step,ind))
rng=np.random.default_rng(58020260917);e,U=np.linalg.eigh(x['K']);factor=U*np.sqrt(np.maximum(e,0.))
N=100000;maxima={f'{region}_{step}':[] for region,step,ind in modes}
gated_maxima={key:[] for key in maxima};raw_maxima={key:[] for key in maxima}
for start in range(0,N,4096):
 W=rng.standard_normal((min(4096,N-start),len(m)))@factor.T
 raw=a+s*W
 for region,step,ind in modes:
  key=f'{region}_{step}';maxima[key].append(W[:,ind].max(axis=1))
  gated_maxima[key].append(np.where(raw[:,ind]>0,W[:,ind],-np.inf).max(axis=1))
  raw_maxima[key].append(np.maximum(0,raw[:,ind].max(axis=1)))
maxima={k:np.concatenate(v) for k,v in maxima.items()};np.savez_compressed(B/'gaussian_grid_maxima.npz',**maxima)
gated_maxima={k:np.concatenate(v) for k,v in gated_maxima.items()};raw_maxima={k:np.concatenate(v) for k,v in raw_maxima.items()}
np.savez_compressed(B/'gaussian_gated_maxima.npz',**gated_maxima);np.savez_compressed(B/'gaussian_raw_maxima.npz',**raw_maxima)
ordering_rows=[];ordering_thresholds=[]
for region,step,ind in modes:
 key=f'{region}_{step}';eligible=obs[ind]>0
 collections=[('ungated_centered',maxima[key],Z[:,ind].max(axis=1),float(zobs[ind].max()),float(m[ind[np.argmax(zobs[ind])]])),('gated_centered',gated_maxima[key],np.where(V[:,ind]>0,Z[:,ind],-np.inf).max(axis=1),float(zobs[ind][eligible].max()) if eligible.any() else None,float(m[ind[eligible][np.argmax(zobs[ind][eligible])]]) if eligible.any() else None),('raw_positive_root',raw_maxima[key],np.maximum(0,V[:,ind].max(axis=1)),float(max(0,obs[ind].max())),float(m[ind[np.argmax(obs[ind])]]))]
 for ordering,gm,vm,threshold,peak in collections:
  k=N if threshold is None else int(np.count_nonzero(gm>=threshold));kv=256 if threshold is None else int(np.count_nonzero(vm>=threshold));ci=interval(k,N);cv=interval(kv,256)
  ordering_rows.append(dict(region=region,step_MeV=step,ordering=ordering,observed_threshold=threshold,observed_peak_mass_MeV=peak,observed_positive_fit_exists=bool(eligible.any()),gaussian_exceedances=k,gaussian_n=N,gaussian_p=k/N,gaussian_low=ci[0],gaussian_high=ci[1],direct_exceedances=kv,direct_n=256,direct_p=kv/256,direct_low=cv[0],direct_high=cv[1],gaussian_fraction_no_positive_fit=float(np.mean(~np.isfinite(gated_maxima[key])))))
  for threshold in [2.,3.,4.]:
   k=int(np.count_nonzero(gm>=threshold));kv=int(np.count_nonzero(vm>=threshold));ci=interval(k,N);cv=interval(kv,256)
   ordering_thresholds.append(dict(region=region,step_MeV=step,ordering=ordering,threshold=threshold,gaussian_exceedances=k,gaussian_n=N,gaussian_p=k/N,gaussian_low=ci[0],gaussian_high=ci[1],direct_exceedances=kv,direct_n=256,direct_p=kv/256,direct_low=cv[0],direct_high=cv[1]))
pd.DataFrame(ordering_rows).to_csv(B/'ordering_observed_maxima.csv',index=False);pd.DataFrame(ordering_thresholds).to_csv(B/'ordering_fixed_thresholds.csv',index=False)
rows=[];tails=[];diffs=[]
for region,step,ind in modes:
 vm=Z[:,ind].max(axis=1);gm=maxima[f'{region}_{step}'];j=ind[np.argmax(obs[ind])];jc=ind[np.argmax(zobs[ind])]
 rows.append(dict(region=region,step_MeV=step,hypotheses=len(ind),observed_raw_max=float(obs[j]),observed_raw_max_mass=float(m[j]),observed_centered_max=float(zobs[jc]),observed_centered_max_mass=float(m[jc]),stress_offset_at_raw_max=float(a[j]),direct_mean_max=float(vm.mean()),gaussian_mean_max=float(gm.mean()),gaussian_q95=float(np.quantile(gm,.95))))
 coarse=maxima[f'{region}_1.0'];increase=gm-coarse
 diffs.append(dict(region=region,step_MeV=step,mean_max_increase=float(increase.mean()),q95_max_increase=float(np.quantile(increase,.95)),fraction_max_increase_gt0p01=float(np.mean(increase>.01)),largest_max_increase=float(increase.max()),same_spectrum_direct_mean_increase=float(np.mean(vm-Z[:,modes[[r[0:2] for r in modes].index((region,1.))][2]].max(axis=1)))))
 for threshold in [2.,3.,4.]:
  added=int(np.sum((gm>=threshold)&(coarse<threshold)));added_ci=interval(added,N)
  k=int(np.sum(gm>=threshold));kv=int(np.sum(vm>=threshold));ci=interval(k,N);cv=interval(kv,len(vm))
  tails.append(dict(region=region,step_MeV=step,threshold=threshold,paired_added_exceedances=added,paired_delta_p=added/N,paired_delta_low=added_ci[0],paired_delta_high=added_ci[1],gaussian_exceedances=k,gaussian_n=N,gaussian_p=k/N,gaussian_low=ci[0],gaussian_high=ci[1],direct_exceedances=kv,direct_n=len(vm),direct_p=kv/len(vm),direct_low=cv[0],direct_high=cv[1]))
pd.DataFrame(rows).to_csv(B/'grid_maximum_summary.csv',index=False)
pd.DataFrame(tails).to_csv(B/'grid_tail_summary.csv',index=False)
pd.DataFrame(diffs).to_csv(B/'paired_grid_changes.csv',index=False)
write(B/'same_node_invariance.json',{'passed':True,'max_abs_changes':invariance,'gaussian_trials':N,'seed':58020260917,'min_raw_K_eigenvalue':float(e.min()),'negative_eigenvalue_clip_sum':float(-e[e<0].sum())})
for name in ['grid_tail_summary.csv','ordering_observed_maxima.csv','ordering_fixed_thresholds.csv']:
 frame=pd.read_csv(B/name)
 for prefix in ['gaussian','direct']:
  counts=frame[prefix+'_exceedances'].to_numpy();numbers=frame[prefix+'_n'].to_numpy()
  frame[prefix+'_status']=np.where(counts==0,'zero_exceedances_upper_bound_only',np.where(counts==numbers,'all_exceedances','finite_MC_estimate'))
  frame[prefix+'_one_sided_95_upper']=[1. if k==n else float(beta.ppf(.95,k+1,n-k)) for k,n in zip(counts,numbers)]
 frame.to_csv(B/name,index=False)
fig,axs=plt.subplots(3,2,figsize=(10.3,8.5),gridspec_kw={'width_ratios':[1.25,1]})
for iy,year in enumerate(['2015','2016','2021']):
 dd=d[d.dataset==year];axx=axs[iy,0]
 axx.plot(dd.mass_MeV,dd.observed_r,label='Observed signed root',color='black',lw=1)
 axx.plot(dd.mass_MeV,dd.stress_offset,label='Stress mean (deterministic)',color='#c35437',lw=1)
 axx.axhline(0,color='.7',lw=.6);axx.set_ylabel(f'{year}: signed root');axx.grid(alpha=.16)
 axx=axs[iy,1];axx.plot(dd.mass_MeV,dd.z_stress_conditional,label='Stress-centered contrast',color='#41699b',lw=1);axx.axhline(0,color='.7',lw=.6);axx.set_ylabel('Conditional centered Z');axx.grid(alpha=.16)
axs[0,0].legend(loc='lower right');axs[0,1].legend(loc='lower right');axs[2,0].set_xlabel('Tested mass [MeV]');axs[2,1].set_xlabel('Tested mass [MeV]')
fig.suptitle('Stress centering changes the reference question; it does not establish a resonance',fontsize=12)
fig.tight_layout(rect=(0,0,1,.97));save(fig,'stress_offsets_and_mapping')
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.2})
fig,axs=plt.subplots(2,2,figsize=(6.8,4.4))
for ax,(lo,hi,title) in zip(axs.flat,[(39,50,'Low-mass stress'),(70,81,'70–81 MeV'),(74,79,'Quarter-step diagnostic'),(85,100,'High-mass peak')]):
 q=(m>=lo)&(m<=hi);coarse=q&np.isclose(m,np.round(m));half=q&np.isclose(2*m,np.round(2*m));quarter=q&~np.isclose(2*m,np.round(2*m))
 ax.plot(m[coarse],obs[coarse],'o-',ms=3,color='black',label='Observed, 1 MeV')
 ax.plot(m[half],obs[half],'.--',ms=4,color='#41699b',label='Observed, 0.5 MeV')
 if np.any(quarter):ax.plot(m[quarter],obs[quarter],'s',ms=3,color='#8a569a',label='Observed, added 0.25 MeV')
 ax.plot(m[q],a[q],'-',lw=.8,color='#c35437',label='Stress offset')
 ax.set_title(title);ax.set_xlabel('Tested mass [MeV]');ax.set_ylabel('Signed root');ax.grid(alpha=.2)
handles,_=axs[1,0].get_legend_handles_labels();fig.legend(handles,['Observed, 1 MeV','Observed, 0.5 MeV','Added 0.25 MeV','Stress offset'],loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,fontsize=7.2,frameon=False);fig.suptitle('Finer grids preserve the shared coordinates',fontsize=10);fig.tight_layout(rect=(0,0,1,.9))
with plt.rc_context({'savefig.bbox':None}):save(fig,'grid_refinement_roots')
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':8})
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for year,color in [('2015','#2d846f'),('2016','#c35437'),('2021','#41699b')]:
 dd=d[d.dataset==year]
 axs[0].plot(dd.mass_MeV,dd.centered_toy_mean,lw=.8,label=year,color=color)
 axs[1].plot(dd.mass_MeV,dd.centered_toy_sd,lw=.8,label=year,color=color)
axs[0].axhline(0,color='black',ls=':',lw=.8);axs[1].axhline(1,color='black',ls=':',lw=.8)
axs[0].set_ylabel('Mean of 256 centered stress toys');axs[1].set_ylabel('SD of 256 centered stress toys')
for ax in axs:ax.set_xlabel('Mass [MeV]');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Conditional bulk validation; 256 toys cannot resolve discovery tails',fontsize=12);fig.tight_layout(rect=(0,0,1,.96));save(fig,'conditional_bulk_validation')
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for region,color in [('low','#2d846f'),('stress','#c35437'),('high','#41699b'),('full','#8a569a')]:
 ds=pd.DataFrame(diffs);dr=ds[(ds.region==region)&(ds.step_MeV!=1)]
 for row in dr.itertuples():
  c=maxima[f'{region}_1.0'];f=maxima[f'{region}_{row.step_MeV}'];xx=np.linspace(0,4.5,100)
  axs[0].plot(xx,[np.mean(c>=v) for v in xx],ls='--',color=color,lw=1)
  axs[0].plot(xx,[np.mean(f>=v) for v in xx],label={'low':'39–50 MeV','stress':'70–81 MeV','high':'85–100 MeV','full':'39–180 MeV'}[region],color=color,lw=1)
  vals=np.sort(f-c);axs[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),label={'low':'39–50 MeV','stress':'70–81 MeV','high':'85–100 MeV','full':'39–180 MeV'}[region],color=color,lw=1)
axs[0].set_yscale('log');axs[0].set_ylim(1e-4,1);axs[0].set_xlabel('Centered local threshold');axs[0].set_ylabel('Conditional ungated maximum tail');axs[0].set_title('Dashed: 1 MeV; solid: added half steps')
axs[1].set_xlim(0,.5);axs[1].set_xlabel('Paired increase in maximum Z');axs[1].set_ylabel('Cumulative fraction')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();save(fig,'gaussian_grid_convergence')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(diffs).to_string(index=False))
