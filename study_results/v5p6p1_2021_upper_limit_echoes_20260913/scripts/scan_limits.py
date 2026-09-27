"""Reanalyse saved v5.6 spectra: full mass-grid 90% CLs limits, no new draws."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,time,argparse
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'engine'))
from common import DATA,kernel_state,sigma,factor_cov
from stable_gp import predict_grid
from limit_solver import OneSignalProfile
from scipy.special import ndtr
import numpy as np,pandas as pd
H=pd.read_csv(B/'inputs/historical_one/observed_2021_1pct_reviewed.csv')
CAT=pd.read_csv(B/'inputs/catalogue.csv');D=B/'derived/scans';D.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def masses(lane):return np.arange(53 if lane=='one' else 50,251,dtype=float)

def density(n,m):
 d=DATA['2021'];sig=sigma('2021',m);lo=m/1000-1.64*sig;hi=m/1000+1.64*sig;e=d['edges']
 assert lo>=e[0] and hi<=e[-1]
 overlap=np.maximum(0,np.minimum(e[1:],hi)-np.maximum(e[:-1],lo))
 return float(np.sum(n*overlap/np.diff(e))/(hi-lo))

def conversion(n,m):
 K=3*np.pi*(m/1000)*float(DATA['2021']['frad_effective'])*density(n,m)/(2/137.)
 if m>211.316749:
  r=(105.6583745/m)**2;brinv=1+np.sqrt(1-4*r)*(1+2*r)
 else:brinv=1.
 return K,float(brinv)

def model(n,lane,m,cut=1e-14):
 d=DATA['2021'];x=d['x'];sig=sigma('2021',m);valid=x>=.04 if lane=='one' else np.ones(len(x),bool);mask=(abs(x-m/1000)<=2.25*sig)&valid
 if lane=='one':c,l=[float(np.exp(np.interp(m,H.mass_GeV*1000,np.log(H[k])))) for k in ['const_opt','ls_opt']]
 else:c,l=kernel_state('2021',float(m))
 bg,C,gd=predict_grid(x,n,valid&~mask,mask,c,l,cut=cut);L,diag=factor_cov(C,bg);g=np.diff(ndtr((d['edges']-m/1000)/sig))
 return OneSignalProfile(bg,L,g[mask]),mask,dict(cov_load=diag['load'],cov_load_over_poisson=float(diag['load']*max(np.diag(C).max(),1)/min(bg)),prior_relative_error=gd['prior_relative_error'],feature_rank=gd['feature_rank'])

def calculate(n,bref,lane,m):
 mod,mask,diag=model(n,lane,m);r=mod.limit(n[mask]);K,brinv=conversion(n,m);Kref,_=conversion(bref,m)
 assert r['ok'] and r['A90']>0 and r['min_background']>0 and r['min_lambda']>0 and r['max_score']<=3e-5
 assert abs(r['cls']-.1)<2e-6
 return dict(mass_MeV=float(m),**r,**diag,density_observed=density(n,m),K_counts_per_epsilon2=K,branching_factor=brinv,epsilon2_90=r['A90']/K*brinv,epsilon2_90_fixed_density=r['A90']/Kref*brinv)

def signature(row):
 paths=[Path(__file__)]+sorted((B/'engine').glob('*.py'))+list((B/'inputs').rglob('*npz'))+[B/'inputs/historical_one/observed_2021_1pct_reviewed.csv',B/'inputs/catalogue.csv',B/'inputs/toys'/f'{row.scenario}.npz',B/'protocol.json']
 return {str(p.relative_to(B)):sha(p) for p in paths}

def spectra(row):
 z=dict(np.load(B/'inputs/toys'/f'{row.scenario}.npz'));g=z['signal']/row.injected_yield
 return z,[('background_asimov',z['background']),('matched_asimov',z['truth']),('yield_asimov',z['background']+row.naive_scaled_yield*g)]+[(f'toy_{i:02}',n) for i,n in enumerate(z['counts'])]

def run():
 start=time.time()
 for _,row in CAT.iterrows():
  spec,ns=spectra(row);sig=signature(row);out=D/row.scenario;out.mkdir(exist_ok=True);ng=masses(row.lane)
  for label,n in ns:
   path=out/f'{label}.csv';meta=out/f'{label}.json'
   if path.exists() and meta.exists():
    old=json.loads(meta.read_text());assert old['dependencies']==sig and old['csv_sha256']==sha(path),'Stale output';continue
   t=time.time();rows=[calculate(n,spec['background'],row.lane,m) for m in ng]
   frame=pd.DataFrame(rows);tmp=path.with_suffix('.tmp');frame.to_csv(tmp,index=False,float_format='%.17g');tmp.replace(path)
   dump(meta,dict(scenario=row.scenario,spectrum=label,rows=len(rows),seconds=time.time()-t,dependencies=sig,csv_sha256=sha(path)))
   print(row.scenario,label,'rows',len(rows),'sec',round(time.time()-t,1),'total_min',round((time.time()-start)/60,1),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','run']);a=ap.parse_args()
 if a.mode=='run':run()
 else:
  row=CAT[CAT.scenario=='ten_92'].iloc[0];z,ns=spectra(row);t=time.time();results=[]
  for label,n in ns[:2]:
   for m in [80.,86.,93.,100.,106.]:results.append(dict(spectrum=label,**calculate(n,z['background'],row.lane,m)))
  pd.DataFrame(results).to_csv(B/'qa/pilot.csv',index=False);print('10limits_seconds',time.time()-t);print(pd.DataFrame(results)[['spectrum','mass_MeV','A90','signed_r','epsilon2_90']].to_string(index=False))
