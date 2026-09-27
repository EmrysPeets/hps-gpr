"""Re-express saved profiled fields with raw nominal p ordering, without refits."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import fixed_core as F
import numpy as np,pandas as pd
from scipy.stats import norm
B=Path(__file__).resolve().parents[1]
fields={name:dict(np.load(B/f'inputs/parent/fields/{name}.npz')) for name in (*F.YEARS,'combined')}
curves={};rows=[];peaks=[]
for year in (*F.YEARS,'combined'):
    f=fields[year];scope='shared' if year=='combined' else year
    q=np.maximum(f['observed_r'],0)**2;tq=np.maximum(f['validation'],0)**2
    p=np.where(q>0,norm.sf(np.sqrt(q)),1.)
    tp=np.where(tq>0,norm.sf(np.sqrt(tq)),1.)
    curves[scope]=(f['masses'],q,p,tq,tp,f['observed_r'])
masses=fields['combined']['masses'];q=np.zeros(len(masses));tq=np.zeros((256,len(masses)));ks=np.zeros(len(masses),int)
for y in F.YEARS:
    f=fields[y];inds=np.searchsorted(masses,f['masses'])
    q[inds]+=np.maximum(f['observed_r'],0)**2;tq[:,inds]+=np.maximum(f['validation'],0)**2;ks[inds]+=1
p=np.ones(len(masses));tp=np.ones_like(tq)
for k in [1,2,3]:
    sel=ks==k;p[sel]=F.free_p(q[sel],k);tp[:,sel]=F.free_p(tq[:,sel],k)
curves['free']=(masses,q,p,tq,tp,np.full(len(masses),np.nan))
for scope,(m,q,p,tq,tp,r) in curves.items():
    for i in range(len(m)):
        rows.append({'scope':scope,'mass_MeV':m[i],'q0':q[i],'r':r[i],'nominal_local_p':p[i]})
    domains=['full','overlap'] if scope in ['shared','free'] else ['full']
    for domain in domains:
        selected=np.ones(len(m),bool) if domain=='full' else (m>=50)&(m<=100)
        indices=np.flatnonzero(selected);i=indices[np.argmin(p[selected])]
        local=int(np.sum(tq[:,i]>=q[i]));glob=int(np.sum(np.min(tp[:,selected],axis=1)<=p[i]))
        low,high=F.interval(glob,256)
        peaks.append({'scope':scope,'domain':domain,'mass_MeV':m[i],'nominal_local_p':p[i],'q0':q[i], 'local_exceedances':local,'global_exceedances':glob,'toys':256,'global_fraction':glob/256,'global_plus_one_p':(glob+1)/257,'global_CP95_low':low,'global_CP95_high':high})
pd.DataFrame(rows).to_csv(B/'results/profiled_raw_scan.csv',index=False,float_format='%.17g')
pd.DataFrame(peaks).to_csv(B/'results/profiled_raw_peaks.csv',index=False,float_format='%.17g')
if (B/'results/peaks.json').exists():
    fixed=pd.DataFrame(json.loads((B/'results/peaks.json').read_text()))
    fixed['domain']=fixed.domain.replace({'overlap_50_100':'overlap'})
    cols=['scope','domain','mass_MeV','nominal_local_p','conditional_local_k','conditional_global_k','conditional_global_p_raw','conditional_global_lo95','conditional_global_hi95']
    paired=fixed[cols].merge(pd.DataFrame(peaks)[['scope','domain','mass_MeV','nominal_local_p','local_exceedances','global_exceedances','global_fraction']],on=['scope','domain'],suffixes=('_fixed','_profiled'))
    paired.to_csv(B/'results/paired_peak_comparison.csv',index=False,float_format='%.17g')
print(pd.DataFrame(peaks).to_string(index=False))
