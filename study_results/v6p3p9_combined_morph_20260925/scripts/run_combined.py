"""Joint observed profiles: unchanged 2015/2016, neighboring-MC 2021 signal."""
from pathlib import Path
import os, sys, json, hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
import run_observed as R
import numpy as np
import pandas as pd
from scipy.linalg import block_diag, cholesky, cho_solve
from scipy.special import ndtr
from scipy.stats import beta
B=Path(__file__).resolve().parents[1]; C=R.C
POLICIES=R.POLICIES; SEED=639250925; NTOYS=1000
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def years(m):return [y for y in ('2015','2016','2021') if y=='2021' or y=='2015' and m<=100 or y=='2016' and m<=180]
def conversion(y,m):
 d=C.DATA[y];x=m/1000.;s=C.sigma(y,m);e=d['native_edges'];w=np.diff(e)
 overlap=np.maximum(0.,np.minimum(e[1:],x+1.64*s)-np.maximum(e[:-1],x-1.64*s))
 density=float(np.sum(d['native_counts']*overlap/w)/(3.28*s))
 return float(3*np.pi*x*float(d['frad_effective'])*density/(2/137.))*1e-8
def branch(m):
 r=(105.6583745/m)**2
 return float(1+np.sqrt(max(0,1-4*r))*(1+2*r)) if m>211.316749 else 1.
def oldpart(y,m,counts=None):
 p=C.moving_context(y,m,counts)
 return dict(year=y,n=p['n'],b=p['b'],L=p['L'],S=p['S'][:,0],load=p['diagnostic']['load'])
def newpart(m,policy,counts=None,ctx=None):
 ctx=R.Context(m,policy) if ctx is None else ctx;n=C.DATA['2021']['n'] if counts is None else counts
 b,cov=ctx.prediction(n);L,d=C.factor_cov(cov,b)
 return dict(year='2021',n=n[ctx.fit],b=b,L=L,S=ctx.probability[ctx.fit]*conversion('2021',m),load=d['load'])
def model(parts):
 n=np.concatenate([p['n'] for p in parts]);b=np.concatenate([p['b'] for p in parts]);L=block_diag(*[p['L'] for p in parts]);S=np.concatenate([p['S'] for p in parts])
 return C.OneSignalProfile(b,L,S),n
def fit(parts,m,policy,scope,save=None):
 mod,n=model(parts);q=mod.limit(n,details=save is not None)
 assert q['ok'] and q['max_score']<3e-5 and q['min_lambda']>0 and abs(q['cls']-.1)<2e-6
 if save is not None:
  free=q.pop('free');null=q.pop('null');trace=q.pop('trace')
  np.savez_compressed(save,n=n,b=mod.b,L=mod.L,S=mod.S,free_theta=free['theta'],null_theta=null['theta'],profiled_total=free['lam'],profiled_background=free['bfit'],trace_json=json.dumps(trace))
 return dict(mass_MeV=m,policy=policy,scope=scope,campaigns='+'.join(p['year'] for p in parts),psi_hat=q['Ahat'],sigma_psi=q['sigma_A'],psi90=q['A90'],epsilon2_90_ee_proxy=q['A90']*1e-8,epsilon2_90_visible_legacy=q['A90']*1e-8*branch(m),signed_root=q['signed_r'],Z_local=max(0,q['signed_r']),p0_asymptotic=q['p0_fixed_mass'],cls=q['cls'],max_score=q['max_score'],min_lambda=q['min_lambda'],max_covariance_load=max(p['load'] for p in parts),valid=True)
def protocol():
 q=dict(version='6.3.9',mass_grid_MeV=list(range(60,241)),policies=list(POLICIES),years='2015 full through100;2016 full through180;2021 10% through240',shared_parameter='psi=epsilon2/1e-8; inherited yield conversion at generated mass; 2021 full-selected neighboring MC probabilities',likelihood='Independent campaign Poisson factors; block-diagonal GP constraint; one common signal amplitude, independent background nuisance parameters',scope='Conditional local asymptotic limits and p-values; no global calibration or independently validated physical exclusion',background_toys=NTOYS,seed=SEED,toy_selection='Global minimum of observed local morph p-value; fixed-mass null diagnostic after selection, not a global p-value',script_sha256=sha(__file__),context_sha256=sha(B/'scripts/run_observed.py'),template_sha256=sha(B/'scripts/observed_templates.py'),input_manifest_sha256=sha(B/'provenance/input_manifest.sha256'))
 p=B/'provenance/combined_protocol.json'
 if p.exists():assert json.loads(p.read_text())==q
 else:write(p,q)
 return sha(p)
def run_scan(signature):
 rows=[]
 for m in range(60,241):
  path=B/f'results/checkpoints/m{m:03d}.json'
  if path.exists():
   item=json.loads(path.read_text());assert item['signature']==signature;rows+=item['rows'];continue
  older=[oldpart(y,m) for y in years(m) if y!='2021'];out=[]
  for p in older:out.append(fit([p],m,'unchanged',p['year']))
  for policy in POLICIES:out.append(fit(older+[newpart(m,policy)],m,policy,'combined'))
  write(path,dict(signature=signature,rows=out));rows+=out
  if m%20==0:print('Combined mass',m,'complete',flush=True)
 old2021=pd.read_csv(B/'inputs/v638_observed_scan.csv',float_precision='round_trip')
 for q in old2021.itertuples():
  factor=conversion('2021',q.mass_MeV);r=dict(mass_MeV=q.mass_MeV,policy=q.policy,scope='2021',campaigns='2021',psi_hat=q.Ahat/factor,sigma_psi=q.sigma_A/factor,psi90=q.A90/factor,epsilon2_90_ee_proxy=q.A90/factor*1e-8,epsilon2_90_visible_legacy=q.A90/factor*1e-8*branch(q.mass_MeV),signed_root=q.signed_r,Z_local=max(0,q.signed_r),p0_asymptotic=q.p0_fixed_mass,cls=q.cls,max_score=q.max_score,min_lambda=q.min_lambda,max_covariance_load=q.covariance_load,valid=True);rows.append(r)
 frame=pd.DataFrame(rows);R.csv(B/'results/combined_scan.csv',frame);return frame
def validate(frame):
 old=pd.read_csv(B/'inputs/legacy_observed_display.csv',float_precision='round_trip');old=old[(old.scope=='combined')&(old.policy=='logshift')]
 new=frame[(frame.scope=='combined')&(frame.policy=='gaussian_baseline')];checks=[]
 for q in new.itertuples():
  o=old[old.mass_MeV==q.mass_MeV].iloc[0]
  assert np.isclose(q.psi90,o.psi90,rtol=2e-7,atol=1e-7) and abs(q.signed_root-o.signed_root)<2e-6
  checks.append(dict(mass_MeV=q.mass_MeV,limit_relative_difference=q.psi90/o.psi90-1,root_difference=q.signed_root-o.signed_root))
 R.csv(B/'results/legacy_combination_agreement.csv',pd.DataFrame(checks))
 for p in POLICIES:
  a=frame[(frame.scope=='combined')&(frame.policy==p)&(frame.mass_MeV>180)].sort_values('mass_MeV');b=frame[(frame.scope=='2021')&(frame.policy==p)&(frame.mass_MeV>180)].sort_values('mass_MeV')
  assert np.allclose(a.psi90,b.psi90,rtol=2e-7) and np.allclose(a.signed_root,b.signed_root,atol=2e-6)
 q=frame[(frame.scope=='combined')&(frame.policy=='morph_starter')].sort_values(['p0_asymptotic','mass_MeV']).iloc[0]
 replays=[]
 for m in sorted(set([60,67,79,100,101,180,181,226,240,int(q.mass_MeV)])):
  for p in POLICIES:
   parts=[oldpart(y,m) for y in years(m) if y!='2021']+[newpart(m,p)]
   row=fit(parts,m,p,'combined',B/f'results/replays/m{m:03d}_{p}.npz');ref=frame[(frame.mass_MeV==m)&(frame.policy==p)&(frame.scope=='combined')].iloc[0]
   assert np.isclose(row['psi90'],ref.psi90,rtol=2e-10) and abs(row['signed_root']-ref.signed_root)<1e-8
   mod,n=model(parts);free=mod.fit(n);_,_,H,_=mod._objective(free['z'],n,mod.Jfree,mod.b,mod.penfree);unit=np.zeros(len(H));unit[0]=1
   sd=mod.scale*np.sqrt(cho_solve((cholesky(H,lower=True),True),unit)[0]);assert abs(sd/ref.sigma_psi-1)<1e-9
   fixed=mod.fit(n,fixed=ref.psi90);theta=fixed['theta'];offset=0;total=0.
   for part in parts:
    rank=part['L'].shape[1];z=theta[offset:offset+rank];offset+=rank
    lam=part['b']+part['L']@z+ref.psi90*part['S'];total+=C.poisson_deviance_half(part['n'],lam)+.5*np.dot(z,z)
   assert abs(total-fixed['nll'])<1e-6
   replays.append(dict(mass_MeV=m,policy=p,passed=True))
 write(B/'qa/numerical_validation.json',dict(passed=True,observed_fit_rows=len(frame),new_joint_fits=543,new_older_individual_fits=162,reused_2021_fits=543,legacy_combined_points=181,max_legacy_limit_relative_difference=max(abs(r['limit_relative_difference']) for r in checks),above180_reduces_to_2021=True,replays=replays,shared_parameter_likelihood_factorization=True,scan_sha256=sha(B/'results/combined_scan.csv')))
 return int(q.mass_MeV)
def local_toys(m,frame,signature):
 truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in years(m)};contexts={p:R.Context(m,p) for p in POLICIES};rows=[]
 for first in range(0,NTOYS,100):
  path=B/f'results/toy_checkpoints/m{m:03d}_t{first:04d}.json'
  if path.exists():
   q=json.loads(path.read_text());assert q['signature']==signature;rows+=q['rows'];continue
  out=[]
  for t in range(first,first+100):
   counts={y:np.random.default_rng(np.random.SeedSequence([SEED,int(y),t])).poisson(truths[y]) for y in years(m)}
   old=[oldpart(y,m,counts[y]) for y in years(m) if y!='2021']
   for p in POLICIES:
    mod,n=model(old+[newpart(m,p,counts['2021'],contexts[p])]);f=mod.fit(n);n0=mod.fit(n,fixed=0,initial=f['theta']);q=2*(n0['nll']-f['nll']);assert q>=-2e-6
    assert max(f['score'],n0['score'])<3e-5
    r=float(np.sign(f['A'])*np.sqrt(max(0,q)));out.append(dict(mass_MeV=m,policy=p,toy=t,signed_root=r,q0=max(r,0)**2,psi_hat=f['A'],score=max(f['score'],n0['score'])))
  write(path,dict(signature=signature,rows=out));rows+=out
  print('Joint local background toys',first+100,'/',NTOYS,flush=True)
 toys=pd.DataFrame(rows);R.csv(B/'results/combined_local_toys.csv',toys);summary=[]
 for p in POLICIES:
  ref=frame[(frame.scope=='combined')&(frame.mass_MeV==m)&(frame.policy==p)].iloc[0];qobs=max(0,ref.signed_root)**2;k=int((toys[toys.policy==p].q0>=qobs).sum())
  summary.append(dict(mass_MeV=m,policy=p,local_asymptotic_p=ref.p0_asymptotic,local_asymptotic_Z=ref.Z_local,toys=NTOYS,tail_count=k,rank_p=(1+k)/(1+NTOYS),cp95_low=0. if k==0 else float(beta.ppf(.025,k,NTOYS-k+1)),cp95_high=1. if k==NTOYS else float(beta.ppf(.975,k+1,NTOYS-k))))
 R.csv(B/'results/combined_local_calibration.csv',pd.DataFrame(summary))
 write(B/'qa/toy_validation.json',dict(passed=True,fit_rows=len(toys),mass_MeV=m,scope='Conditional fixed-mass q0 tail after observed selection; no look-elsewhere correction',paired_across_policies=True,rows_sha256=sha(B/'results/combined_local_toys.csv')))
def main():
 for d in ('results/checkpoints','results/toy_checkpoints','results/replays'):(B/d).mkdir(parents=True,exist_ok=True)
 signature=protocol();frame=run_scan(signature);m=validate(frame);local_toys(m,frame,signature)
 summary={}
 for p in POLICIES:
  q=frame[(frame.scope=='combined')&(frame.policy==p)];r=q.loc[q.p0_asymptotic.idxmin()];summary[p]=dict(mass_MeV=int(r.mass_MeV),Z_local=float(r.Z_local),p0=float(r.p0_asymptotic),epsilon2_90_visible_legacy=float(r.epsilon2_90_visible_legacy))
 write(B/'results/summary.json',dict(minima=summary,selected_toy_mass_MeV=m,observed_rows=len(frame),new_profile_fits=705,toy_fit_rows=3000));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
