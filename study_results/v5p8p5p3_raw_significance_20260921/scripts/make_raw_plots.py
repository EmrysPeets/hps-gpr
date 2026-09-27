"""Unshifted observed p/Z displays and all-scope Figure 4 diagnostics.
No fit, optimization, toy generation, or reference calibration of observed curves.
"""
from pathlib import Path
import os,sys,csv,json,hashlib
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v5853-mpl')
sys.dont_write_bytecode=True
import numpy as np
from scipy.stats import norm,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures';R=B/'results'
SCOPES=['2015','2016','2021','combined']
LABELS={'2015':'2015 full','2016':'2016 full','2021':'2021 native 10%','combined':'Combined: shared coupling'}
COLORS={'2015':'#235b7e','2016':'#a94135','2021':'#29745c','combined':'#70548f'}
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=170)
 plt.close(fig)
def intervals(k,n):
 k=np.asarray(k);lo=np.zeros_like(k,dtype=float);hi=np.ones_like(k,dtype=float)
 ix=k>0;lo[ix]=beta.ppf(.025,k[ix],n-k[ix]+1)
 ix=k<n;hi[ix]=beta.ppf(.975,k[ix]+1,n-k[ix])
 return lo,hi

def tail(maxima,threshold):
 vals=np.sort(maxima);k=len(vals)-np.searchsorted(vals,threshold,side='left');n=len(vals);lo,hi=intervals(k,n)
 return k,n,(k+1)/(n+1),lo,hi

def write(name,rows):
 with (R/name).open('w',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

data={};curves=[];peaks=[];momentrows=[];moments=[];identities=[]
for scope in SCOPES:
 path=B/'inputs/fields'/f'{scope}.npz';f=dict(np.load(path));data[scope]=f
 m=f['masses'];r=f['observed_r'];a=f['a'];s=f['s'];v=f['validation'];positive=r>0
 Z=np.maximum(r,0);p=norm.sf(Z);ref=(r-a)/s
 k,n,pg,lo,hi=tail(f['gaussian_raw_maximum'],Z)
 dk,dn,dp,dlo,dhi=tail(np.maximum(0,v.max(axis=1)),Z)
 for arr in [pg,lo,hi,dp,dlo,dhi]:arr[~positive]=1.
 k[~positive]=n;dk[~positive]=dn
 GZ=np.maximum(0,norm.isf(pg));ztoys=(v-a)/s
 f.update(local_Z=Z,local_p=p,global_p=pg,global_Z=GZ,global_lo=lo,global_hi=hi)
 for j,mass in enumerate(m):
  curves.append(dict(scope=scope,mass_MeV=float(mass),signed_raw_r=float(r[j]),local_Z_unshifted=float(Z[j]),local_p_unshifted=float(p[j]),signed_normal_tail=float(norm.sf(r[j])),observed_positive_fit=bool(positive[j]),reference_offset_a=float(a[j]),reference_scale_s=float(s[j]),reference_coordinate_for_comparison=float(ref[j]),raw_order_global_k=int(k[j]),raw_order_global_N=n,raw_order_global_p_addone=float(pg[j]),raw_order_global_Z=float(GZ[j]),raw_order_global_p_lo95=float(lo[j]),raw_order_global_p_hi95=float(hi[j]),direct_raw_global_k=int(dk[j]),direct_raw_global_N=dn,direct_raw_global_p_addone=float(dp[j]),direct_raw_global_p_lo95=float(dlo[j]),direct_raw_global_p_hi95=float(dhi[j])))
  momentrows.append(dict(scope=scope,mass_MeV=float(mass),observed_raw_r=float(r[j]),deterministic_offset_a=float(a[j]),response_scale_s=float(s[j]),mean_raw_toy=float(v[:,j].mean()),sd_raw_toy=float(v[:,j].std(ddof=1)),mean_standardized_toy=float(ztoys[:,j].mean()),sd_standardized_toy=float(ztoys[:,j].std(ddof=1)),independent_toy_scans=len(v)))
 j=int(np.argmax(r));jr=int(np.argmax(np.where(positive,ref,-np.inf)));row=next(q.copy() for q in curves if q['scope']==scope and q['mass_MeV']==m[j]);row.update(reference_peak_mass_MeV=float(m[jr]),reference_peak_Z=float(ref[jr]),reference_local_recalibration_used_in_prior=True);peaks.append(row)
 moments.append(dict(scope=scope,source_rms_offset=float(np.sqrt(np.mean(a*a))),min_offset=float(a.min()),max_offset=float(a.max()),min_standardized_mean=float(ztoys.mean(0).min()),max_standardized_mean=float(ztoys.mean(0).max()),min_standardized_sd=float(ztoys.std(0,ddof=1).min()),max_standardized_sd=float(ztoys.std(0,ddof=1).max())))
 identities.append(dict(scope=scope,file=str(path.relative_to(B)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mass_min_MeV=float(m.min()),mass_max_MeV=float(m.max()),grid_nodes=len(m),toy_scans=len(v),Gaussian_draws=len(f['gaussian_raw_maximum'])))
 assert np.array_equal(Z,np.maximum(f['observed_r'],0))
 assert np.max(np.abs(norm.isf(p)-Z))<1e-12
 assert np.all(np.diff(m)==.5)
 assert np.all(np.isfinite(r)) and np.all(s>0)

# Four rows retain each dataset's actual search support; all rows use common y limits.
for mode in ['local','global']:
 fig,axes=plt.subplots(4,2,figsize=(10.6,10.4))
 for i,scope in enumerate(SCOPES):
  f=data[scope];m=f['masses'];axz,axp=axes[i];color=COLORS[scope]
  if mode=='local':
   z,p=f['local_Z'],f['local_p'];axz.plot(m,z,c=color,lw=1.2);axp.semilogy(m,p,c=color,lw=1.2)
   axz.axhline(3,c='.6',ls=':',lw=.8);axp.axhline(norm.sf(3),c='.6',ls=':',lw=.8)
   axz.set_ylim(-.04,3.9);axp.set_ylim(1e-4,.75)
  else:
   z,p=f['global_Z'],f['global_p'];lo,hi=f['global_lo'],f['global_hi']
   axz.fill_between(m,np.maximum(0,norm.isf(hi)),np.maximum(0,norm.isf(np.maximum(lo,1e-30))),color=color,alpha=.2,lw=0)
   axz.plot(m,z,c=color,lw=1.2);axp.fill_between(m,lo,hi,color=color,alpha=.2,lw=0);axp.semilogy(m,p,c=color,lw=1.2)
   axz.set_ylim(-.035,1.9);axp.set_ylim(.03,1.08)
  j=int(np.argmax(f['observed_r']))
  axz.scatter(m[j],z[j],s=18,c=color,zorder=4);axp.scatter(m[j],p[j],s=18,c=color,zorder=4)
  axz.set_title(f'{LABELS[scope]}  |  peak {m[j]:g} MeV',fontsize=10.2,loc='left',pad=6)
  axp.set_title(LABELS[scope],fontsize=10.2,loc='left',pad=6)
  axz.set_ylabel('Local Z (unshifted)' if mode=='local' else 'Global Z (fixed source)')
  axp.set_ylabel('Local p (unshifted)' if mode=='local' else 'Global p (fixed source)')
  for ax in [axz,axp]:
   ax.set_xlim(m.min(),m.max());ax.set_xlabel('Tested mass [MeV]')
   if scope=='combined':
    for boundary in [39,50,100,180]:ax.axvline(boundary,c='.55',lw=.6,ls=':',alpha=.5)
 title='Observed local curves: no reference centering or rescaling' if mode=='local' else 'Full-range tails of the raw maximum: conditional on the fixed source'
 fig.suptitle(title,fontsize=13)
 fig.tight_layout(rect=(0,0,1,.974),h_pad=1.5,w_pad=2.0);save(fig,'raw_'+mode+'_overview')

for scope in SCOPES:
 f=data[scope];m=f['masses'];r=f['observed_r'];a=f['a'];s=f['s'];v=f['validation'];z=(v-a)/s
 fig,axes=plt.subplots(2,2,figsize=(10.4,6.6),sharex=True)
 axes[0,0].plot(m,r,c='.22',lw=1,label='Observed signed root r')
 axes[0,0].plot(m,a,c='#b34a36',lw=1.3,label='Deterministic source response a = r(B)')
 axes[0,0].set(ylabel='Raw signed likelihood root',title='A. Unshifted observation and source response')
 axes[0,0].legend(fontsize=8.5,loc='lower right')
 axes[0,1].plot(m,a,c='#b34a36',lw=1.3,label='a = r(B)')
 axes[0,1].plot(m,v.mean(0),c='black',lw=.85,label='Mean raw toy root')
 axes[0,1].set(ylabel='Raw toy root / source response',title='B. Raw null mean; no centering')
 axes[0,1].legend(fontsize=9,loc='lower right')
 axes[1,0].plot(m,z.mean(0),c='black',lw=.9)
 axes[1,0].axhline(0,c='#b34a36',ls='--',lw=.9)
 axes[1,0].axhspan(-1/16,1/16,color='.75',alpha=.3)
 axes[1,0].set(ylabel=r'Mean of $(r_{\rm toy}-a)/s$',title='C. Standardized toy mean (diagnostic only)',ylim=(-.22,.22))
 axes[1,1].plot(m,z.std(0,ddof=1),c='.22',lw=.9)
 axes[1,1].axhline(1,c='#b34a36',ls='--',lw=.9)
 axes[1,1].axhspan(1-1/np.sqrt(510),1+1/np.sqrt(510),color='.75',alpha=.3)
 axes[1,1].set(ylabel=r'SD of $(r_{\rm toy}-a)/s$',title='D. Standardized toy width (diagnostic only)',ylim=(.84,1.22))
 for ax in axes.flat:
  ax.set_xlim(m.min(),m.max());ax.set_xlabel('Tested mass [MeV]')
  if scope=='combined':
   for boundary in [39,50,100,180]:ax.axvline(boundary,c='.6',ls=':',lw=.65,alpha=.55)
 fig.suptitle(f'{LABELS[scope]}: extension of v5.8.5 Figure 4\nOne frozen source, 256 complete Poisson scans',fontsize=13)
 fig.tight_layout(rect=(0,0,1,.94),w_pad=2,h_pad=2.1);save(fig,'response_diagnostics_'+scope)
 # Convenient individual local p/Z figures, also included in plot archive.
 fig,axes=plt.subplots(2,1,figsize=(9.0,5.7),sharex=True)
 axes[0].plot(m,f['local_Z'],c=COLORS[scope],lw=1.2);axes[0].set(ylabel='Unshifted local Z',ylim=(-.05,3.9))
 axes[1].semilogy(m,f['local_p'],c=COLORS[scope],lw=1.2);axes[1].set(ylabel='Unshifted local p',xlabel='Tested mass [MeV]',ylim=(1e-4,.75),xlim=(m.min(),m.max()))
 fig.suptitle(LABELS[scope]+': observed local curves, no reference recalibration',fontsize=12)
 fig.tight_layout(rect=(0,0,1,.955));save(fig,'raw_local_'+scope)

write('raw_significance_curves.csv',curves);write('raw_peak_summary.csv',peaks);write('response_moment_curves.csv',momentrows);write('response_moment_summary.csv',moments)
(R/'input_fields.json').write_text(json.dumps(identities,indent=2)+'\n')
summary={'passed':True,'scope_count':4,'mass_scope_coordinates':len(curves),'new_fits':0,'new_toys':0,'blind_half_width_sigma':2.25,'observed_reference_centering':False,'local_p_convention':'norm.sf(max(raw_r,0)); zero/negative root displayed as p=0.5, Z=0','global_p_convention':'inclusive tail of max positive raw root; add-one MC estimate, p=1 for zero observed statistic','global_source_qualification':'conditional on frozen observed-data-derived nominal GP means','global_counts':'200000 saved Gaussian fields and 256 saved complete Poisson scans per scope','toy_diagnostic_interpretation':'C/D standardize saved toys only; this does not modify new observed local curves','peaks':peaks}
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'coordinates':len(curves),'peaks':[(r['scope'],r['mass_MeV'],r['local_Z_unshifted'],r['local_p_unshifted']) for r in peaks]},indent=2))
