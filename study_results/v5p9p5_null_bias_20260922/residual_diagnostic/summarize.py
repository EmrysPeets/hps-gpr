"""Portable numeric audit, source-bias projection, and exploratory scan-Q summary."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H/'results';F=H/'figures'
def read(p):
    rows=list(csv.DictReader(p.open()))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except ValueError:pass
    return rows
def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def js(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
a=read(R/'analytic_scan.csv');controls=read(R/'poisson_controls.csv');paired=read(R/'paired_contrasts.csv');emp=read(R/'empirical_decomposition.csv')
orig=read(H/'inputs/slide14_original_scan.csv')
replay_delta=max(abs(r['observed_Q_per_bin']-s['Q_per_bin']) for r,s in zip(a,orig));assert replay_delta==0
d=np.load(H/'inputs/spectrum_2021.npz');B=np.load(H/'inputs/null_2021.npz')['truth'];field=np.load(H/'inputs/slide50_field_2021.npz')
projection=[]
for j,r in enumerate(a):
    m=r['mass_MeV'];mask=np.abs(d['x']-m/1000)<=2.25*d['sigma'][j];xt=d['x'][~mask];xq=d['x'][mask];y=B[~mask]
    def kernel(q,z):return d['const'][j]*np.exp(-.5*((np.log(q)[:,None]-np.log(z)[None,:])/d['ls'][j])**2)
    K=kernel(xt,xt);K.flat[::len(K)+1]+=1/y;L=cholesky(K,lower=True);Kqt=kernel(xq,xt);v=solve_triangular(L,Kqt.T,lower=True)
    cl=kernel(xq,xq)-v.T@v;cl=(cl+cl.T)/2;b=np.exp(Kqt@cho_solve((L,True),np.log(y))+.5*np.maximum(np.diag(cl),0));C=np.outer(b,b)*np.expm1(np.clip(cl,-40,40));C=(C+C.T)/2
    V=np.diag(b)+C;delta=B[mask]-b;fac=(cholesky(V,lower=True),True);S=d['templates'][j,mask]
    gls=float(S@cho_solve(fac,delta)/np.sqrt(S@cho_solve(fac,S)));total_bias=float(delta@cho_solve(fac,delta));i=int(np.where(field['masses']==m)[0][0]);rawa=float(field['a'][i])
    # Same small PSD loading and rank rule as archived likelihood factor_cov.
    scale=max(float(np.diag(C).max()),1.)
    for load in (1e-10,1e-9,1e-8,1e-7,1e-6,1e-5):
        try:cholesky(C+load*scale*np.eye(len(b)),lower=True);break
        except np.linalg.LinAlgError:pass
    sd=np.sqrt(b);val,U=np.linalg.eigh((C+load*scale*np.eye(len(b)))/sd[:,None]/sd[None,:]);keep=val>1e-8;facC=sd[:,None]*U[:,keep]*np.sqrt(val[keep]);Ceff=facC@facC.T
    fc=(cholesky(np.diag(b)+Ceff,lower=True),True);glseff=float(S@cho_solve(fc,delta)/np.sqrt(S@cho_solve(fc,S)))
    projection.append(dict(mass_MeV=m,Nbin=int(mask.sum()),deterministic_total_Q=total_bias,deterministic_Q_per_bin=total_bias/mask.sum(),signal_direction_GLS_root=gls,signal_direction_GLS_squared=gls*gls,saved_profiled_root_a=rawa,saved_profiled_root_a_squared=rawa*rawa,GLS_signal_direction_fraction=gls*gls/total_bias if total_bias else 0.,posterior_covariance_relative_change=float(np.linalg.norm(Ceff-C)/np.linalg.norm(C)),conditioned_GLS_minus_raw=glseff-gls))
write(R/'bias_projection_bridge.csv',projection)

# Correct the positive finite-MC norm floor; these corrected estimates can be negative.
for r in emp:
    aa=next(x for x in a if x['mass_MeV']==r['mass_MeV'])
    floor=(r['toy_prediction_covariance_noise_Q_per_bin']-aa['expected_frozen_noise_per_bin'])/r['N_toys']
    r['finite_MC_prediction_mean_norm_floor']=floor
    r['toy_squared_bias_noise_corrected']=r['toy_expected_residual_mean_bias_Q_per_bin']-floor
    r['jensen_squared_shift_noise_corrected']=r['jensen_shift_Q_per_bin']-floor
write(R/'empirical_decomposition_with_MC_floor.csv',emp)
arr=np.load(R/'paired_Q_arrays.npz');masses=[int(r['mass_MeV']) for r in a]
q=np.array([arr[f'm{m}_exact_refit_adaptiveV'] for m in masses]).T;maxima=q.max(axis=1);imax=max(a,key=lambda r:r['observed_Q_per_bin']);obs=imax['observed_Q_per_bin'];n=len(maxima);k=int((maxima>=obs).sum())
scan={'statistic':'T_Q=max over 201 test masses of adaptive Q/Nbin','mass_grid_MeV':[50,250,1],'observed_maximum':obs,'observed_maximum_mass_MeV':imax['mass_MeV'],'N_paired_complete_spectra':n,'exceedances':k,'tail_add_one':(k+1)/(n+1),'binomial_CP95':[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],'null_maximum_median':float(np.median(maxima)),'null_maximum_central90':np.quantile(maxima,[.05,.95]).tolist(),'qualification':'Exploratory fixed-source residual-scan diagnostic. Not resonance significance, calibrated model GOF, or independent validation. Source estimated from observed data; source uncertainty not propagated.'}
js(R/'exploratory_residual_scan_tail.json',scan);write(R/'residual_scan_maxima.csv',[{'toy_index':i,'maximum_Q_per_bin':float(v)} for i,v in enumerate(maxima)])
checks={'slide14_replay_max_absolute_delta':replay_delta,'all_201_masses_complete':len(a)==201 and len(controls)==201*8,'paired_spectra':n,'all_array_entries_finite':bool(np.isfinite(q).all()),'empirical_fixedV_decomposition_max_error':max(abs(r['empirical_fixedV_Q_identity']-next(c['mean'] for c in controls if c['mass_MeV']==r['mass_MeV'] and c['control']=='exact_refit_fixedV')) for r in emp),'jacobian_relative_error_max':max(r['relative_L2_error'] for r in read(R/'jacobian_checks.csv')),'max_abs_exact_minus_linear_mean':max(abs(r['mean']) for r in paired if r['contrast']=='exact_minus_linear'),'max_abs_adaptiveV_minus_fixedV_mean':max(abs(r['mean']) for r in paired if r['contrast']=='adaptiveV_minus_fixedV'),'max_conditioned_covariance_relative_change':max(r['posterior_covariance_relative_change'] for r in projection),'max_conditioned_GLS_root_change':max(abs(r['conditioned_GLS_minus_raw']) for r in projection)}
checks['passed']=checks['all_201_masses_complete'] and checks['all_array_entries_finite'] and replay_delta==0 and checks['empirical_fixedV_decomposition_max_error']<1e-12 and checks['jacobian_relative_error_max']<.003
js(R/'validation.json',checks)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10,4))
axs[0].hist(maxima,bins=22,color='#6c9eb6',alpha=.85);axs[0].axvline(obs,c='#a54732',lw=2,label=f'Observed max = {obs:.3f}');axs[0].set(xlabel=r'$T_Q=\max_m\; Q(m)/N_{\rm bin}$',ylabel='Number of paired Poisson scans');axs[0].legend(frameon=False,fontsize=9)
axs[0].text(.96,.76,f'{k}/{n} exceedances\n95% interval [{scan["binomial_CP95"][0]:.3f}, {scan["binomial_CP95"][1]:.3f}]',transform=axs[0].transAxes,ha='right',fontsize=9)
axs[1].plot(masses,[r['saved_profiled_root_a'] for r in projection],c='#a54732',label='Saved profiled Asimov root a')
axs[1].plot(masses,[r['signal_direction_GLS_root'] for r in projection],c='#24688d',ls='--',label='GLS projection of source residual')
axs[1].axhline(0,c='.6',lw=.7);axs[1].set(xlabel='Test mass [MeV]',ylabel='Signed local coordinate');axs[1].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(F/'residual_scan_and_bias_projection.pdf');fig.savefig(F/'residual_scan_and_bias_projection.png',dpi=180);plt.close(fig)
print(json.dumps({'exploratory_scan':scan,'validation':checks,'bridge_78':next(r for r in projection if r['mass_MeV']==78)},indent=2))
