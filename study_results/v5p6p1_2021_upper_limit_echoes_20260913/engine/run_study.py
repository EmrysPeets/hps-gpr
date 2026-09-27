"""v5.6: conditional exposure and target-Z morphology catalogue, one worker."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from common import *
from scipy.special import ndtr
from stable_gp import predict_grid
from scipy.optimize import brentq
from scipy.signal import find_peaks
import argparse,shutil,time
D=B/'derived';SEED=56020210912
ACTIVE_LANE='ten'
HISTORICAL=None
REGIONS=[('65',60,70),('75',71,83),('92',86,99),('120',111,132),('160',148,177),('210',198,223)]

def template(m):
 d=DATA['2021'];s=sigma('2021',m);return np.diff(ndtr((d['edges']-m/1000)/s))

def fit_at(n,m,full=False):
 d=DATA['2021'];x=d['x'];s=sigma('2021',m);valid=x>=.04 if ACTIVE_LANE=='one' else np.ones(len(x),bool);mask=(abs(x-m/1000)<=2.25*s)&valid
 if ACTIVE_LANE=='one':
  global HISTORICAL
  if HISTORICAL is None:HISTORICAL=pd.read_csv(B/'inputs/historical_one/observed_2021_1pct_reviewed.csv')
  c,l=[float(np.exp(np.interp(m,HISTORICAL.mass_GeV*1000,np.log(HISTORICAL[k])))) for k in ['const_opt','ls_opt']]
 else:c,l=kernel_state('2021',float(m))
 b,C,gpdiag=predict_grid(x,n,valid&~mask,mask,c,l);L,diag=factor_cov(C,b);g=template(m);mod=OneSignalProfile(b,L,g[mask]);f=mod.fit(n[mask]);z=mod.fit(n[mask],0)
 delta=2*(z['nll']-f['nll']);assert delta>=-1e-7
 r=float(np.sign(f['A'])*np.sqrt(max(0,delta)))
 row=dict(mass_MeV=float(m),sigma_MeV=s*1000,A_hat=f['A'],sigma_A=f['sigma'],r=r,Z=max(0.,r),p0=float(ndtr(-max(0,r))),max_score=max(f['score'],z['score']),min_background=min(f['min_background'],z['min_background']),min_lambda=min(float(f['lam'].min()),float(z['lam'].min())),cov_load=diag['load'],rank=diag['rank'],cov_load_over_poisson=float(diag['load']*max(np.diag(C).max(),1)/min(b)),**gpdiag)
 assert row['max_score']<=3e-5 and row['min_background']>0 and row['min_lambda']>0, row
 if not full:return row
 bfull,_,_=predict_grid(x,n,valid&~mask,np.ones(len(x),bool),c,l)
 return row,dict(mask=mask,b=b,C=C,L=L,g=g,bfull=bfull,fit=f,zero=z)

def dump(p,v):write(p,v)

def prepare():
 global ACTIVE_LANE
 d=DATA['2021'];hist=B/'inputs/source_2021_1pct.root'
 import uproot
 if not hist.exists():shutil.copy2('/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root',hist)
 with uproot.open(hist) as f:y,edges=f['preselection/h_invM_8000'].to_numpy()
 idx=np.array([int(np.argmin(abs(edges-x))) for x in d['edges']]);assert max(abs(edges[idx]-d['edges']))<1e-12
 n1=np.diff(np.r_[0.,np.cumsum(y)][idx]);assert np.all(n1>=0)
 np.savez_compressed(B/'inputs/lanes.npz',one=n1,ten=d['n'],x=d['x'],edges=d['edges'])
 dump(B/'inputs/lane_manifest.json',dict(one=dict(source='/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root',snapshot='source_2021_1pct.root',histogram='preselection/h_invM_8000',sha256=sha(hist),nominal_fraction=.01,exact_exposure_verified=False,selection_equivalent_to_ten_verified=False,TC_membership_verified=False),ten=dict(source='pinned v5.2.1 spectrum_2021.npz, native v4.9.12/v5.0.4 10% release',sha256=sha(B/'inputs/spectrum_2021.npz'),nominal_fraction=.1),bin_width_MeV=.625,support_MeV=[36,300],scan_MeV=[50,250],kernel_policy='Reviewed source-specific integer states: historical1% uses its reviewed k15 scan and 40-300 MeV support; native10 uses pinned current states and 36-300 MeV. Fixed coordinates, fresh GP conditioning; no hyperparameter optimization',resolution='Fully scaled 2021 polynomial inherited from pinned input'))
 rows=[]
 for lane,n in [('one',n1),('ten',d['n'])]:
  ACTIVE_LANE=lane
  for m in range(50,251):rows.append(dict(lane=lane,**fit_at(n,m)))
  print('source scan',lane,'complete',flush=True)
 pd.DataFrame(rows).to_csv(D/'source_scan.csv',index=False,float_format='%.17g')
 s=pd.DataFrame(rows);selected=[]
 for lane,k in [('one',100),('ten',10)]:
  q=s[s.lane==lane].reset_index(drop=True);chosen=set()
  for label,lo,hi in REGIONS:
   a=q[q.mass_MeV.between(lo,hi)];r=a.loc[a.Z.idxmax()]
   if r.Z<=0:continue
   chosen.add(float(r.mass_MeV));selected.append(dict(scenario=f'{lane}_{label}',lane=lane,region=label,region_lo=lo,region_hi=hi,k=k,selection='requested regional maximum',**r.drop('lane').to_dict()))
  if lane=='one':
   peaks,_=find_peaks(q.Z.to_numpy());leaders=q.iloc[peaks].sort_values(['Z','mass_MeV'],ascending=[False,True]).head(6)
   for _,r in leaders.iterrows():
    m=float(r.mass_MeV)
    if m in chosen:continue
    chosen.add(m);selected.append(dict(scenario=f'one_extra{m:g}',lane=lane,region=f'extra {m:g}',region_lo=max(50,m-5),region_hi=min(250,m+5),k=k,selection='additional top-six 1% local maximum; distinct grid location',**r.drop('lane').to_dict()))
 out=pd.DataFrame(selected);out['target_Z']=np.sqrt(out.k)*out.Z;out.to_csv(D/'selected_peaks.csv',index=False,float_format='%.17g')
 dump(B/'protocol.json',dict(version='5.6.0',seed=SEED,toys_per_scenario=20,workers=1,linear_algebra_threads=1,regions=REGIONS,extra_one_percent='top six interior local maxima, adding any not already selected; nearby maxima retained and not independent',injection='Gaussian count template; numerical target Z=source Z sqrt(k)',yield_scaling_comparison='k max(Ahat,0) Asimov comparison, reported separately from target-Z matched toys',generator='Independent Poisson counts on scaled source GP continuum plus matched Gaussian signal over all 36-300 MeV support',inference='Recondition GP on each toy sidebands with fixed source-specific reviewed kernels; profile count signal and Gaussian GP nuisance; 1% support40-300,10% support36-300 MeV',regional_scan='1 MeV grid within +/-2 sigma of injected mass, bounded to 50-250 MeV; injected mass always included',selection='post-observation conditional catalogue',gp_numerics='PSD eigenspace of RBF Gram matrix at relative eigenvalue cutoff1e-14; SVD ridge conditioning avoids subtractive posterior covariance',full_observed_2021_used=False,statistical_claim='nominal fixed-mass asymptotic diagnostics; no global calibration, confidence bounds or guaranteed discovery probability'))
 print(out[['scenario','mass_MeV','Z','target_Z']].to_string(index=False),flush=True)

def fingerprint():
 paths=[B/'scripts'/n for n in ['run_study.py','stable_gp.py','common.py','parent_core.py','limit_solver.py']]+[B/'inputs/lanes.npz',B/'inputs/spectrum_2021.npz',B/'inputs/historical_one/observed_2021_1pct_reviewed.csv',D/'selected_peaks.csv']
 return {str(p.relative_to(B)):sha(p) for p in paths}

def scenario(row):
 global ACTIVE_LANE
 ACTIVE_LANE=row.lane
 sid=row.scenario;out=D/f'{sid}.json';toyfile=B/'toys'/f'{sid}.npz';curves=D/f'{sid}_curves.csv'
 if out.exists():
  cached=json.loads(out.read_text());assert cached['dependency_sha256']==fingerprint(), 'Stale cached scenario'
  assert cached['counts_sha256']==sha(toyfile), 'Changed toy data'
  print('cached',sid,flush=True);return
 lane=dict(np.load(B/'inputs/lanes.npz'));n=lane[row.lane];m=float(row.mass_MeV);k=float(row.k);r,p=fit_at(n,m,True);bg=k*p['bfull'];g=p['g'];target=float(row.target_Z);naive=k*max(0,r['A_hat'])
 def az(a):return fit_at(bg+a*g,m)['Z']
 zzero=az(0);znaive=az(naive);assert zzero<target
 hi=max(naive,1)
 while az(hi)<target:hi*=2
 amp=brentq(lambda a:az(a)-target,0,hi,xtol=1e-6,rtol=1e-11);truth=bg+amp*g
 za,pa=fit_at(truth,m,True);assert abs(za['Z']-target)<1e-5
 s=float(row.sigma_MeV);masses=np.unique(np.r_[m,np.arange(max(50,np.ceil(m-2*s)),min(250,np.floor(m+2*s))+1)])
 reps=[];allcurves=[];toys=[];fitted=[];fitted_b=[];expected=[];expected_b=[]
 for mm in masses:allcurves.append(dict(scenario=sid,toy=-1,**fit_at(truth,float(mm))))
 key=int(round(m*1000))+(0 if row.lane=='one' else 1000000)
 for t in range(20):
  rng=np.random.default_rng(np.random.SeedSequence([SEED,key,t]));counts=rng.poisson(truth);toys.append(counts)
  local=[]
  for mm in masses:
   rr=fit_at(counts,float(mm));local.append(rr);allcurves.append(dict(scenario=sid,toy=t,**rr))
  fixed,pp=fit_at(counts,m,True);best=max(local,key=lambda a:a['Z']);reps.append(dict(toy=t,seed_key=key,Z_fixed=fixed['Z'],r_fixed=fixed['r'],Z_region_max=best['Z'],mass_region_max=best['mass_MeV'],A_hat=fixed['A_hat'],sigma_A=fixed['sigma_A'],max_score=max(a['max_score'] for a in local),min_background=min(a['min_background'] for a in local),min_lambda=min(a['min_lambda'] for a in local)))
  fitted.append(pp['fit']['lam']);fitted_b.append(pp['fit']['bfit'])
 np.savez_compressed(toyfile,counts=np.asarray(toys),truth=truth,background=bg,signal=amp*g,x=lane['x'],edges=lane['edges'],mask=p['mask'],source_counts=n,source_background=p['bfull'],source_signal=max(0,r['A_hat'])*g,fitted=np.asarray(fitted),fitted_background=np.asarray(fitted_b),asimov_fitted=pa['fit']['lam'],asimov_background=pa['fit']['bfit'])
 pd.DataFrame(allcurves).to_csv(curves,index=False,float_format='%.17g');q=pd.DataFrame(reps);q.to_csv(D/f'{sid}_toys.csv',index=False,float_format='%.17g')
 z=q.Z_fixed.to_numpy();zm=q.Z_region_max.to_numpy();res=dict(row.to_dict());res.update(source_Z=r['Z'],source_Ahat=r['A_hat'],source_sigmaA=r['sigma_A'],naive_scaled_yield=naive,naive_yield_asimov_Z=znaive,background_asimov_Z=zzero,injected_yield=amp,matched_to_naive_yield_ratio=amp/naive,target_Z=target,matched_asimov_Z=za['Z'],matched_asimov_Ahat=za['A_hat'],asimov_recovery_fraction=za['A_hat']/amp,expected_signal_fitwindow=float((amp*g)[p['mask']].sum()),expected_background_fitwindow=float(bg[p['mask']].sum()),peak_signal_to_background=float(max((amp*g)/bg)),toy_count=20,Z_median=float(np.median(z)),Z_q16=float(np.quantile(z,.16)),Z_q84=float(np.quantile(z,.84)),Z_min=float(min(z)),Z_max=float(max(z)),Z_region_max_median=float(np.median(zm)),Z_region_max_max=float(max(zm)),n_fixed_above3=int(sum(z>=3)),n_fixed_above5=int(sum(z>=5)),max_score=float(q.max_score.max()),min_background=float(q.min_background.min()),min_lambda=float(q.min_lambda.min()),n_mass_points=len(masses),seed_key=key,counts_sha256=sha(toyfile),dependency_sha256=fingerprint())
 dump(out,res);print(sid,'m',m,'target',round(target,3),'toy median',round(res['Z_median'],3),'naive',round(znaive,3),flush=True)

def summarize():
 rows=[json.loads(p.read_text()) for p in sorted(D.glob('one_*.json'))+sorted(D.glob('ten_*.json'))]
 assert {r['scenario'] for r in rows}==set(pd.read_csv(D/'selected_peaks.csv').scenario), 'Incomplete catalogue'
 pd.DataFrame(rows).drop(columns=['dependency_sha256']).to_csv(D/'catalogue.csv',index=False,float_format='%.17g')
 dump(B/'qa/numerical_validation.json',dict(passed=True,scope='Runner summary only; independent release checks in qa/final_validation.json',scenarios=len(rows),toys=sum(r['toy_count'] for r in rows),max_target_error=max(abs(r['target_Z']-r['matched_asimov_Z']) for r in rows),max_fit_score=max(r['max_score'] for r in rows),min_fitted_background=min(r['min_background'] for r in rows),min_fitted_expectation=min(r['min_lambda'] for r in rows),single_worker=True,full_observed_2021_used=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','toys','summary']);args=ap.parse_args()
 if args.mode=='prepare':prepare()
 elif args.mode=='toys':
  for _,row in pd.read_csv(D/'selected_peaks.csv').iterrows():scenario(row)
  summarize()
 else:summarize()
