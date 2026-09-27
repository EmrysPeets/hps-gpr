"""Independent pilot, frozen calibration and held-out MC-to-Gaussian validation."""
import argparse,fcntl,json,time,sys,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import core as I
import numpy as np
import pandas as pd

def protocol():
 return dict(version='6.3.5',dataset='2021 native10%',master_seed=I.MASTER,masses_MeV=list(I.MASSES),policies=list(I.POLICIES),sources=list(I.SOURCES),
  pilot=100,calibration=100,evaluation=100,calibration_grid=list(I.GRID),evaluation_levels=list(I.LEVELS),
  background_generators={'nominal':'Pinned full-support GP arithmetic mean','functional':'Archived locally source-assessed anchored fSigPowExpQ functional stress; no independent global source qualification'},
  fit='Signed Gaussian full-selected amplitude, nominal resolution at pole; correlated GP constrained Poisson likelihood',
  core_law=dict(form='center=m+a+b*ln(m/150), all quantities MeV',coefficients=I.COEF,domain_MeV=[60,240],width_changed=False),
  primary_window='Fit and GP training-exclusion masks match exactly at center +/-2.25 nominal sigma; no added guard',
  guard_user_constraint='Do not adopt broad guard by containment alone; compare expected sensitivity and variance cost in separate MC-only side study.',
  kernel='Archived mass-specific hyperparameters at pole, fixed; recompute log targets, alpha, mean and covariance for every spectrum',
  normalization='Full-selected count yield; MC histogram CDF retains outside-support and overflow categories; Gaussian full-distribution CDF; no window or support renormalization',
  signal_generation='Independent Poisson categories at fixed A=z*s0. Common MC draws across both fit policies; independent shape streams; inject training sidebands too.',
  pilot_rule='100 nominal null backgrounds; s0(m)=mean returned observed-profile-Hessian Gaussian error for logshift policy; same fixed expected yield for both policies and sources',
  calibration_rule='100 independent null and MC-injected backgrounds per source; freeze all rows and hashes before evaluation; grid specified before fitting',
  evaluation_rule='100 independent backgrounds/source, MC at0/1/3/5; nominal-source Gaussian generation at log-law center at1/3/5 for shape control',
  pairing='Shared background within source/cohort across mass, strength, shape, policy. Independent sources/cohorts. Gaussian and MC signal draws independent; policy comparisons use identical counts.',
  seed_namespaces={'pilot_background':1,'calibration_background':2,'evaluation_background':3,'bootstrap':4,'calibration_signal':20,'evaluation_signal':30,'smoke_signal':90,'smoke_background':91},
  background_seed='[master,cohort_namespace,source_id,toy]',signal_seed='[master,cohort_namespace,source_id,mass,toy,z,shape_id]; mc0 gaussian_log1',
  calibration_statistics='mu0=mean(Ahat0/sigma0); delta=mean(Ahat0); R=mean paired(Ahat3-Ahat0)/(3*s0); k0=SD((Ahat0-delta)/sigma0). Keep four quantities distinct.',
  correction_diagnostics='Subtract frozen mu0 from evaluation pulls; compare separate yield offset and affine(Ahat-delta)/R. No production UL is shifted or divided by R.',
  inference='Raw Gaussian nativeprofileCLs/asymptoticlocalp0 are conditional references. Full-MC-yield limits use finite-grid test inversion of rawAhat distributions at fixedtrueA; plus-one lower-tail rank; retain censoring, holes, emptysets.',
  finite_MC='100calibrationtoys: rankp floor1/101. Marginal rank size<=10/101; frozen-table acceptance varies. Held-out binomialCP95 intervals. No raretail/global/discovery claim.',
  observed='New observed and joint study is separate scripts/observed.py; 2021 core shift only, 2015/2016 fixed, common epsilon2 likelihood; dense60–240conditionalraw and native-anchor MCcalibrated results distinguished.',
  acceptance={'score_lt':3e-5,'min_lambda_gt':0,'q_raw_min':-2e-6,'sigma':'finitepositive independent observed-Hessian check; reject Fisher substitution'},
  retry='Atmost3 identical-count attempts: origin2e-7,origin2e-9, deterministicA/scale=.5 at2e-9; warmfreefromtruthprofileifneeded; choosevalidminimumobjective. Never regenerate or omit failures.',
  resources={'main_workers':2,'other_workers':2,'total_workers_max':4,'numerical_threads_each':1,'watchdog_seconds':1800,'checkpoint_toys':10},
  scope_limits=['No40MeV','260MeV only window-shape diagnostic; outsideloglaw and search domain','NoMCshape interpolation for calibration','MCselection and daughterassociation unqualified','No full production/background-model certification from conditionalclosure','Prior2016qualificationexceptionretained'])

def prepare():
 p=I.B/'protocol.json';spec=protocol()
 if p.exists():assert json.loads(p.read_text())==spec
 else:I.write_json(p,spec)
 cohorts={}
 for c in ('pilot','calibration','evaluation'):
  for s in (('nominal',) if c=='pilot' else I.SOURCES):cohorts[c+'_'+s]=np.array([I.rng(I.background_key(c,s,t)).poisson(I.TRUTHS[s]) for t in range(100)])
 p=I.B/'inputs/cohorts.npz'
 if p.exists():
  old=np.load(p)
  for k,a in cohorts.items():assert np.array_equal(old[k],a)
 else:np.savez_compressed(p,**cohorts,truth_nominal=I.TRUTHS['nominal'],truth_functional=I.TRUTHS['functional'],edges_GeV=I.D['edges'])
 hashes=[I.ahash(x) for a in cohorts.values() for x in a];assert len(set(hashes))==500
 p=I.B/'inputs/templates.npz';arrays=dict(masses_MeV=np.array(I.MASSES),mc_categories=np.array([I.mc_categories(m) for m in I.MASSES]),gaussian_log_categories=np.array([I.gaussian_categories(m) for m in I.MASSES]),
  pole_fit=np.array([I.Context(m,'pole').mask for m in I.MASSES]),logshift_fit=np.array([I.Context(m,'logshift').mask for m in I.MASSES]))
 if p.exists():
  old=np.load(p)
  for k,a in arrays.items():assert np.array_equal(old[k],a)
 else:np.savez_compressed(p,**arrays)
 sig=I.signature();I.write_json(I.B/'provenance/computation_signature.json',dict(signature=sig,protocol_sha256=I.sha(I.B/'protocol.json'),cohorts_sha256=I.sha(I.B/'inputs/cohorts.npz'),templates_sha256=I.sha(p)))
 r=I.B/'provenance/runtime.json'
 if not r.exists():I.write_json(r,dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,created_utc=I.utc()))
 return sig

def smoke(sig):
 p=I.B/'qa/smoke.json'
 if p.exists() and json.loads(p.read_text()).get('signature')==sig:return
 rows=[]
 for m in (60,140,240):
  for source in I.SOURCES:
   bg=I.rng([I.MASTER,91,I.SOURCES.index(source),m]).poisson(I.TRUTHS[source])
   for policy in I.POLICIES:
    ctx=I.Context(m,policy);n=I.row(ctx,bg,np.zeros(len(bg)+2,dtype=np.int64),0,0,source,0,'smoke','mc')
    assert n['fit_valid'] and n['profile_valid'] and n['native_cls90_valid'],n
    A=3*n['sigma_postfit'];draw=I.rng(I.signal_key('smoke',source,m,0,3,'mc')).poisson(A*I.mc_categories(m))
    r=I.row(ctx,bg,draw,A,0,source,3,'smoke','mc');assert r['fit_valid'] and r['profile_valid'] and r['native_cls90_valid'],r;rows.extend([n,r])
    b=I.TRUTHS[source][ctx.mask];pfit=ctx.p[ctx.mask]
    free,fixed,valid,_=I.fit_pair(b,np.zeros((len(b),0)),pfit,b+A*pfit,A,True);assert valid and abs(free['A']/A-1)<1e-7
    pred=ctx.predict(bg);changed=bg.copy();changed[ctx.mask]+=11;again=ctx.predict(changed);assert np.array_equal(pred[0],again[0])
 pd.DataFrame(rows).to_csv(I.B/'qa/smoke_rows.csv',index=False,float_format='%.17g')
 I.write_json(p,dict(passed=True,signature=sig,rows=len(rows),created_utc=I.utc(),checks=['signedfree/truthprofile/nativeCLs numericalgates','fullGaussianyield mean-data closure','fittedwindowexcludedfromtraining','source/policy/mass coverage']))

def run(c,workers,sig,ref=None,cal=None):
 (I.B/'results'/c).mkdir(exist_ok=True)
 sources=('nominal',) if c=='pilot' else I.SOURCES
 jobs=[(c,s,m,t,min(t+10,100),sig,ref,cal) for s in sources for m in I.MASSES for t in range(0,100,10)]
 I.write_json(I.B/'status.json',dict(status='running_'+c,signature=sig,updated_utc=I.utc()))
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for f in as_completed([pool.submit(I.run_chunk,*j) for j in jobs]):print(json.dumps(f.result()),flush=True)
 rows=pd.concat([pd.read_csv(I.chunk_base(c,s,m,t).with_suffix('.csv'),float_precision='round_trip') for _,s,m,t,*_ in jobs]).sort_values(['source','policy','mass_MeV','shape','z','toy'])
 assert len(rows)==dict(pilot=2000,calibration=52000,evaluation=22000)[c]
 assert not rows.duplicated(['source','policy','mass_MeV','shape','z','toy']).any()
 rows.to_csv(I.B/f'results/{c}_rows.csv',index=False,float_format='%.17g');return rows

def freeze_pilot(rows,sig):
 p=I.B/'pilot_reference.json'
 if p.exists():
  r=json.loads(p.read_text());assert r['signature']==sig and r['pilot_rows_sha256']==I.sha(I.B/'results/pilot_rows.csv');return
 assert rows.fit_valid.all();masses={}
 for m in I.MASSES:
  q=rows[(rows.mass_MeV==m)&(rows.policy=='logshift')];assert len(q)==100
  masses[str(m)]=dict(s0=float(q.sigma_postfit.mean()),se_sigma=float(q.sigma_postfit.std(ddof=1)/10),pole_mean_sigma=float(rows[(rows.mass_MeV==m)&(rows.policy=='pole')].sigma_postfit.mean()))
 I.write_json(p,dict(frozen_utc=I.utc(),signature=sig,pilot_rows_sha256=I.sha(I.B/'results/pilot_rows.csv'),masses=masses))

def freeze_cal(rows,sig):
 p=I.B/'calibration_freeze.json';assert rows.fit_valid.all(),'Do not drop failed calibration rows'
 if p.exists():
  r=json.loads(p.read_text());assert r['signature']==sig and r['calibration_rows_sha256']==I.sha(I.B/'results/calibration_rows.csv');return
 I.write_json(p,dict(frozen_utc=I.utc(),signature=sig,rows=len(rows),calibration_rows_sha256=I.sha(I.B/'results/calibration_rows.csv'),pilot_reference_sha256=I.sha(I.B/'pilot_reference.json'),
  checkpoint_hashes={str(p.relative_to(I.B)):I.sha(p) for p in sorted((I.B/'results/calibration').glob('*'))}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--stage',choices=['prepare','smoke','pilot','calibration','all'],default='all');a=ap.parse_args();assert 1<=a.workers<=2
 with (I.B/'run.lock').open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Existing main study owns lock')
  sig=prepare()
  if a.stage=='prepare':return
  smoke(sig)
  if a.stage=='smoke':return
  pilot=run('pilot',a.workers,sig);freeze_pilot(pilot,sig)
  if a.stage=='pilot':return
  cal=run('calibration',a.workers,sig,I.sha(I.B/'pilot_reference.json'));freeze_cal(cal,sig)
  if a.stage=='calibration':return
  ev=run('evaluation',a.workers,sig,I.sha(I.B/'pilot_reference.json'),I.sha(I.B/'calibration_freeze.json'))
  I.write_json(I.B/'status.json',dict(status='numerical_complete',updated_utc=I.utc(),signature=sig,pilot_rows=len(pilot),calibration_rows=len(cal),evaluation_rows=len(ev),
   evaluation_fit_valid=int(ev.fit_valid.sum()),evaluation_profile_valid=int(ev.profile_valid.sum()),native_cls90_attempted=int((ev['shape']=='mc').sum()),native_cls90_valid=int(ev.native_cls90_valid.sum())))
if __name__=='__main__':main()
