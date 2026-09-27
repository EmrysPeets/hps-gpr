"""Source-conditional audit of saved raw roots and scan maxima; no likelihood refits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import numpy as np
from scipy.stats import norm,t,chi2,beta
import json,csv,hashlib,shutil,time,platform
B=Path(__file__).resolve().parents[1];ROOT=B.parents[2]
for d in ('inputs','results','figures','provenance'): (B/d).mkdir(exist_ok=True)
SEED=202609220595;NMC=200000;CHUNK=2048;CONF=.95

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(name,rows):
 with (B/'results'/name).open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cp(k,n,alpha=.05):
 return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]
def jwrite(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
fields={};identities=[]
identity_path=B/'provenance/input_hashes.json'
prior_identities={r['scope']:r for r in json.loads(identity_path.read_text())} if identity_path.exists() else {}
for scope in ('2015','2016','2021','combined'):
 src=ROOT/'study_results/v5p8p5p3_raw_significance_20260921/inputs/fields'/f'{scope}.npz';dst=B/'inputs'/f'{scope}.npz'
 if not dst.exists():
  if not src.exists():raise FileNotFoundError(f'Portable rebuild requires bundled {dst.name}')
  shutil.copy2(src,dst)
 if src.exists():assert sha(src)==sha(dst)
 elif scope in prior_identities:assert sha(dst)==prior_identities[scope]['sha256']
 fields[scope]=dict(np.load(dst));identities.append(dict(scope=scope,source=str(src.relative_to(ROOT)),local=str(dst.relative_to(B)),sha256=sha(dst)))
jwrite(B/'provenance/input_hashes.json',identities)
moments=[];scopes={}
for scope,f in fields.items():
 m,a,s,v,r=f['masses'],f['a'],f['s'],f['validation'],f['observed_r'];n,p=v.shape
 mu=v.mean(0);sd=v.std(0,ddof=1);se=sd/np.sqrt(n);muerr=mu-a
 lo=mu-t.ppf(.975,n-1)*se;hi=mu+t.ppf(.975,n-1)*se
 blo=mu-t.ppf(1-.05/(2*p),n-1)*se;bhi=mu+t.ppf(1-.05/(2*p),n-1)*se
 slo=np.sqrt((n-1)*sd**2/chi2.ppf(.975,n-1));shi=np.sqrt((n-1)*sd**2/chi2.ppf(.025,n-1))
 bslo=np.sqrt((n-1)*sd**2/chi2.ppf(1-.05/(2*p),n-1));bshi=np.sqrt((n-1)*sd**2/chi2.ppf(.05/(2*p),n-1))
 standardized_mean_error=muerr/(s/np.sqrt(n));qnorm=a/s
 clipmean=a*norm.cdf(qnorm)+s*norm.pdf(qnorm)
 for j in range(p):
  moments.append(dict(scope=scope,mass_MeV=float(m[j]),model_offset_a=float(a[j]),model_scale_s=float(s[j]),direct_signed_mean=float(mu[j]),direct_signed_sd=float(sd[j]),direct_mean_minus_a=float(muerr[j]),direct_mean_mc_se=float(se[j]),mean_ci95_lo=float(lo[j]),mean_ci95_hi=float(hi[j]),sd_ci95_normal_lo=float(slo[j]),sd_ci95_normal_hi=float(shi[j]),mean_bonf95_lo=float(blo[j]),mean_bonf95_hi=float(bhi[j]),sd_bonf95_normal_lo=float(bslo[j]),sd_bonf95_normal_hi=float(bshi[j]),mean_standardized_mc_error=float(standardized_mean_error[j]),direct_zero_atom_fraction=float(np.mean(v[:,j]<=0)),model_zero_atom_fraction=float(norm.cdf(-qnorm[j])),direct_clipped_mean=float(np.maximum(v[:,j],0).mean()),model_clipped_mean=float(clipmean[j]),observed_raw_r=float(r[j]),n_direct=n))
 j=int(np.argmax(r));threshold=max(float(r[j]),0.);rawmax=f['gaussian_raw_maximum'];directmax=np.maximum(v.max(1),0);k=int(np.sum(rawmax>=threshold));dk=int(np.sum(directmax>=threshold));lk=int(np.sum(v[:,j]>=threshold))
 local_asym=float(norm.sf(threshold));local_cond=float(norm.sf((threshold-a[j])/s[j]));pg=(k+1)/(len(rawmax)+1)
 scopes[scope]=dict(n_masses=p,mass_range_MeV=[float(m[0]),float(m[-1])],n_direct=n,source_a_mean=float(a.mean()),source_a_rms=float(np.sqrt(np.mean(a*a))),source_a_range=[float(a.min()),float(a.max())],source_a_extrema_mass_MeV=[float(m[np.argmin(a)]),float(m[np.argmax(a)])],source_s_range=[float(s.min()),float(s.max())],direct_mean_minus_a_rms=float(np.sqrt(np.mean(muerr**2))),direct_mean_minus_a_range=[float(muerr.min()),float(muerr.max())],mean_pointwise_misses=int(np.sum((a<lo)|(a>hi))),mean_bonferroni_misses=int(np.sum((a<blo)|(a>bhi))),sd_pointwise_misses=int(np.sum((s<slo)|(s>shi))),sd_bonferroni_misses=int(np.sum((s<bslo)|(s>bshi))),max_abs_standardized_mean_error=float(np.max(np.abs(standardized_mean_error))),max_mean_error_mass_MeV=float(m[np.argmax(np.abs(standardized_mean_error))]),peak=dict(mass_MeV=float(m[j]),raw_r=threshold,raw_local_asymptotic_p=local_asym,source_a=float(a[j]),source_s=float(s[j]),direct_mean=float(mu[j]),direct_mean_ci95=[float(lo[j]),float(hi[j])],direct_sd=float(sd[j]),direct_sd_ci95_normal=[float(slo[j]),float(shi[j])],conditional_gaussian_marginal_p=local_cond,direct_local_k=lk,direct_local_n=n,direct_local_cp95=cp(lk,n),saved_gaussian_global_k=k,saved_gaussian_global_n=len(rawmax),saved_gaussian_global_p_addone=pg,saved_gaussian_global_cp95=cp(k,len(rawmax)),direct_global_k=dk,direct_global_n=n,direct_global_p=dk/n,direct_global_p_addone=(dk+1)/(n+1),direct_global_cp95=cp(dk,n),same_source_gaussian_global_to_marginal_ratio=pg/local_cond,mixed_global_to_raw_asymptotic_ratio=pg/local_asym),saved_rawmax_quantiles=dict(zip(['q025','q16','q50','q84','q975'],map(float,np.quantile(rawmax,[.025,.16,.5,.84,.975])))))
 print(scope,scopes[scope]['peak'],flush=True)
writecsv('local_moments_all_scopes.csv',moments)
# Paired counterfactual distributions at the unchanged 2021 raw observation.
f=fields['2021'];m,a,s,K=f['masses'],f['a'],f['s'],f['K'];p=len(m);r=f['observed_r'];threshold=max(float(r.max()),0.)
eval,evec=np.linalg.eigh((K+K.T)/2);A=evec*np.sqrt(np.maximum(eval,0))[None,:]
factor_error=float(np.max(np.abs(A@A.T-K)))
assert factor_error<1e-8
rng=np.random.default_rng(SEED)
scenario_names=['nominal','remove_mean_only','unit_scale_only','zero_mean_unit_scale','independent_mass_nodes','perfectly_correlated_nodes']
maxima={name:np.empty(NMC) for name in scenario_names};max_abs_w=np.empty(NMC);single=np.empty((NMC,3));j=int(np.argmax(r));start=time.monotonic()
for begin in range(0,NMC,CHUNK):
 end=min(begin+CHUNK,NMC);xi=rng.standard_normal((end-begin,p));W=xi@A.T
 vals={'nominal':a+s*W,'remove_mean_only':s*W,'unit_scale_only':a+W,'zero_mean_unit_scale':W,'independent_mass_nodes':a+s*xi,'perfectly_correlated_nodes':a+s*xi[:,0,None]}
 for name,arr in vals.items():maxima[name][begin:end]=np.maximum(arr.max(1),0)
 max_abs_w[begin:end]=np.abs(W).max(1)
 single[begin:end,0]=vals['nominal'][:,j];single[begin:end,1]=np.maximum(vals['nominal'][:,j],0);single[begin:end,2]=xi[:,j]
 if begin//50000 != (end-1)//50000 or end==NMC:print('paired draws',end,'elapsed_sec',round(time.monotonic()-start,2),flush=True)
nom=maxima['nominal'];nomex=nom>=threshold;rows=[]
for name,z in maxima.items():
 exc=z>=threshold;k=int(exc.sum());delta=exc.astype(float)-nomex.astype(float);d=float(delta.mean());dse=float(delta.std(ddof=1)/np.sqrt(NMC))
 rows.append(dict(scenario=name,n=NMC,k_above_unchanged_raw_peak=k,p_addone=(k+1)/(NMC+1),p_cp95_lo=cp(k,NMC)[0],p_cp95_hi=cp(k,NMC)[1],mean=float(z.mean()),sd=float(z.std(ddof=1)),q025=float(np.quantile(z,.025)),q16=float(np.quantile(z,.16)),median=float(np.median(z)),q84=float(np.quantile(z,.84)),q975=float(np.quantile(z,.975)),paired_tail_difference_vs_nominal=d,paired_difference_se=dse,paired_difference_normal95_lo=d-1.96*dse,paired_difference_normal95_hi=d+1.96*dse))
writecsv('paired_gaussian_counterfactuals_2021.csv',rows)
# Paired direct-scan diagnostics use the same saved toy rows; marginal center/scale transformations are not refits under a new null.
v=f['validation'];directs={'nominal':v,'remove_mean_only':v-a,'unit_scale_only':a+(v-a)/s,'zero_mean_unit_scale':(v-a)/s};direct_rows=[]
for name,q in directs.items():
 z=np.maximum(q.max(1),0);k=int(np.sum(z>=threshold));direct_rows.append(dict(scenario=name,n=len(z),k_above_unchanged_raw_peak=k,p=k/len(z),p_addone=(k+1)/(len(z)+1),p_cp95_lo=cp(k,len(z))[0],p_cp95_hi=cp(k,len(z))[1],median=float(np.median(z)),mean=float(z.mean()),sd=float(z.std(ddof=1))))
writecsv('transformed_direct_scan_diagnostics_2021.csv',direct_rows)
np.savez_compressed(B/'results/paired_maxima_2021.npz',**maxima,max_abs_standard_field=max_abs_w,single_mass_signed=single[:,0],single_mass_clipped=single[:,1],single_standard_normal=single[:,2],seed=np.array(SEED))
crit=float(np.quantile(max_abs_w,.95));obs=scopes['2021']['max_abs_standardized_mean_error'];simk=int(np.sum(max_abs_w>=obs))
scopes['2021']['model_simultaneous_mean_check']=dict(assumption='Independent direct toys with exactly Gaussian signed-root vector mean a and covariance diag(s) R diag(s); conditional model diagnostic, not source qualification.',observed_max_abs_z=obs,critical95=crit,exceedances=simk,n=NMC,p_addone=(simk+1)/(NMC+1),any_outside_band=bool(obs>crit))
check=rows[0];saved=scopes['2021']['peak'];oldp=saved['saved_gaussian_global_k']/saved['saved_gaussian_global_n'];newp=check['k_above_unchanged_raw_peak']/NMC;se=np.sqrt(oldp*(1-oldp)/200000+newp*(1-newp)/NMC)
summary=dict(scopes=scopes,paired_simulation=dict(seed=SEED,n=NMC,chunk_size=CHUNK,elapsed_s=time.monotonic()-start,min_eigenvalue=float(eval.min()),n_negative_eigenvalues_clipped=int(np.sum(eval<0)),max_covariance_factor_error=factor_error,nominal_tail_difference_from_saved=float(newp-oldp),independent_MC_z=float((newp-oldp)/se),scenarios=rows),definitions=dict(raw_signed='r is signed profile likelihood root; a=r(B) is deterministic reference response, not assumed equal to exact ensemble mean.',clipped='Z=max(r,0); even r~N(0,1) has E[Z]=1/sqrt(2*pi)=0.398942.',scan='T=max over mass of max(r,0). Positive maximum location is selection, not itself evidence of estimator bias.',counterfactuals='All observed raw roots and peak threshold stay fixed. Mean, scale and correlation controls change assumed null distributions only; not alternative production p-values.',scope='Every probability and interval is conditional on frozen observed-data-derived generating source. MC intervals omit source-estimation and model uncertainty. Chosen observed peak mass is post-selection for diagnostics, not pre-specified local testing.'))
jwrite(B/'results/summary.json',summary)
jwrite(B/'provenance/runtime.json',dict(python=platform.python_version(),numpy=np.__version__,seed=SEED,thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},script_sha256=sha(Path(__file__))))
print('DONE',json.dumps(summary['scopes']['2021']['model_simultaneous_mean_check']),flush=True)
