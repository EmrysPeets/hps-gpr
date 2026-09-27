#!/usr/bin/env python3
"""Whole-support positive candidate backgrounds; construction never reads scan roots."""
from pathlib import Path
import os,sys,json,hashlib,time
from datetime import datetime,timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from scipy.optimize import minimize
B=Path(__file__).resolve().parents[1];R=B.parents[1];OUT=B/'truths';OUT.mkdir(exist_ok=True)
SOURCE=R/'study_results/v5p8p0_local_significance_mapping_20260917/local/2016_76.npz'
LEGACY=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p2-20260909')
sys.path.insert(0,str(LEGACY))
sys.path.insert(0,str(LEGACY/'study_results/v4p9p13_calibration_20260905'))
import calibration_core as core
c=core.c
from gp_refit_pilot import CachedCholeskyPredictor
cfg=c.production.load_config(c.production.DEFAULT_CARD);states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
z=np.load(SOURCE);observed=z['observed'];edges=z['edges_GeV'];x=(edges[:-1]+edges[1:])/2;xm=x*1000
state=states['2016',76];const=float(state['const_opt']);ls=float(state['ls_opt'])
NAMES=['archived_stress','gp_full_nominal','gp_full_half_ls','gp_blocked','regional_rise_fall']
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sigma(m):return 1000*np.polynomial.polynomial.polyval(m/1000,[.00038,.041,-.27,3.49,-11.11])
PROTOCOL={'declared_utc':datetime.now(timezone.utc).isoformat(),'truth_names':NAMES,'scope':'One fixed positive full-support expectation per candidate shared by all scan coordinates; no root-dependent selection.', 'GP_kernel':{'frozen_anchor_MeV':76,'const':const,'ls_log_mass':ls,'half_ls_factor':.5,'hyperparameter_optimization':False},'blocked_GP':{'centers_MeV':list(map(float,np.arange(30,210.001,5))),'blend_radius_MeV':5.,'weight':'compact C2 Wendland (1-u)^4 (1+4u), u=abs(m-center)/5, then partition normalization','excluded_halfwidth_MeV':'max(15,5+2.25*sigma(min(center+5,210)))','minimum_signal_window_exclusion':'Each positively weighted query lies inside an excluded interval at least2.25sigma(query) from its edges over39–180 MeV.','boundary_behavior':'Same exclusion rule; GP extrapolation near support edges, no data overwrite.'},'regional':{'rising_fit_domain_MeV':[30,75],'falling_fit_domain_MeV':[50,210],'rising_basis':'cubic in (m-60)/30','falling_basis':'cubic in log(m/60)','objective':'Poisson log likelihood, four coefficients each','join_MeV':[50,75],'join':'quintic smoothstep of log expected counts; C2 at endpoints','turnover_context':'10MeV binned observed count maximum is60–70MeV; no local roots used.'},'normalization':'No postfit rescaling; native predicted expected-count total recorded. In particular blocked GP does not leak excluded counts through a total-count normalization.','controls':'Full-data GP candidates are self-fit controls and may absorb actual signal; blocked stitching reduces but does not eliminate source dependence.'}
if __name__=='__main__':write(OUT/'protocol.json',PROTOCOL)
KERNELS={name:c.make_fixed_kernel(const,ls*f) for name,f in [('gp_full_nominal',1.),('gp_full_half_ls',.5)]}
FULL={name:CachedCholeskyPredictor(x,x,k,cfg) for name,k in KERNELS.items()}
BLOCKS=[];weights=[];intervals=[]
for center in np.arange(30,210.001,5):
 u=abs(xm-center)/5;w=np.where(u<1,(1-u)**4*(1+4*u),0.)
 query=w>0;half=max(15.,5+2.25*sigma(min(center+5,210.)))
 train=abs(xm-center)>=half
 assert train.sum()>100
 BLOCKS.append((query,train,CachedCholeskyPredictor(x[train],x[query],KERNELS['gp_full_nominal'],cfg)))
 weights.append(w);intervals.append([center-half,center+half])
weights=np.array(weights);weights/=weights.sum(axis=0)
search=(xm>=39)&(xm<=180)
minimum_clearance=np.full(len(x),np.inf)
for (query,train,pred),(lo,hi) in zip(BLOCKS,intervals):
 minimum_clearance[query]=np.minimum(minimum_clearance[query],np.minimum(xm[query]-lo,hi-xm[query]))
assert np.all(minimum_clearance[search]>=2.25*sigma(xm[search])-1e-10)
if __name__=='__main__':np.savez_compressed(OUT/'block_geometry.npz',edges_GeV=edges,centers_MeV=np.arange(30,210.001,5),weights=weights,excluded_intervals_MeV=intervals,minimum_exclusion_clearance_MeV=minimum_clearance)
left=np.column_stack([((xm-60)/30)**i for i in range(4)])
right=np.column_stack([np.log(xm/60)**i for i in range(4)])
t=np.clip((xm-50)/25,0,1);blend=6*t**5-15*t**4+10*t**3

def poisson_fit(design,y,mask):
 X=design[mask];n=y[mask];norm=n.sum();initial=np.linalg.lstsq(X*np.sqrt(n[:,None]),np.log(n)*np.sqrt(n),rcond=None)[0]
 def fun(beta):
  eta=X@beta;mu=np.exp(eta);return (mu.sum()-n@eta)/norm,X.T@(mu-n)/norm
 fit=minimize(fun,initial,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':400})
 grad=float(np.max(abs(fun(fit.x)[1])))
 assert np.isfinite(fit.x).all() and grad<2e-6,(fit.message,grad)
 return design@fit.x,dict(coefficients=list(map(float,fit.x)),gradient_max=grad,optimizer_success=bool(fit.success))

def construct(y,names=None):
 names=set(NAMES if names is None else names);values={};meta={}
 if 'archived_stress' in names:values['archived_stress']=z['stress'].copy()
 for name,predictor in FULL.items():
  if name in names:values[name]=predictor.predict(y)[0]
 if 'gp_blocked' in names:
  means=np.zeros((len(BLOCKS),len(x)))
  for j,(query,train,predictor) in enumerate(BLOCKS):means[j,query]=predictor.predict(y[train])[0]
  # Convex arithmetic combination of positive count expectations; one whole spectrum.
  values['gp_blocked']=np.sum(weights*means,axis=0)
 if 'regional_rise_fall' in names:
  le,lm=poisson_fit(left,y,xm<=75);re,rm=poisson_fit(right,y,xm>=50)
  values['regional_rise_fall']=np.exp((1-blend)*le+blend*re);meta['regional_fits']={'rising':lm,'falling':rm}
 for name,mu in values.items():assert mu.shape==y.shape and np.isfinite(mu).all() and np.all(mu>0),name
 return values,meta

LOCAL_BLOCKS=[]
for center,(query,train,pred),(lo,hi) in zip(np.arange(30,210.001,5),BLOCKS,intervals):
 supported=bool(lo>30 and hi<210)
 st=states['2016',int(np.clip(center,39,180))]
 kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt'])
 local=CachedCholeskyPredictor(x[train],x[query],kernel,cfg) if supported else None
 LOCAL_BLOCKS.append((query,train,local,dict(center_MeV=float(center),excluded_interval_MeV=[float(lo),float(hi)],both_sidebands_available=supported,kernel_anchor_MeV=int(np.clip(center,39,180)),const=float(st['const_opt']),ls=float(st['ls_opt']))))

def construct_local_blocked(y):
 base=FULL['gp_full_nominal'].predict(y)[0];means=np.zeros((len(BLOCKS),len(x)));self_weight=np.zeros(len(x))
 for j,(query,train,predictor,metadata) in enumerate(LOCAL_BLOCKS):
  if predictor is None:
   means[j,query]=base[query];self_weight+=weights[j]
  else:means[j,query]=predictor.predict(y[train])[0]
 mean=np.sum(weights*means,axis=0)
 assert np.all(mean>0) and np.isfinite(mean).all()
 return mean,dict(blocks=[r[3] for r in LOCAL_BLOCKS],edge_self_fit_weight=self_weight.tolist(),edge_fallback='Unavailable two-sided interpolation blocks use fullGP nominal count mean with the sameC2 partition weights; these edges are explicitlynotheldout.',normalization='none')

def main():
 start=time.monotonic();values,meta=construct(observed)
 np.savez_compressed(OUT/'backgrounds.npz',edges_GeV=edges,observed=observed,**values)
 rows=[]
 for name,mu in values.items():
  logmu=np.log(mu);slope=np.gradient(logmu,xm);curvature=np.gradient(slope,xm);third=np.gradient(curvature,xm)
  rows.append(dict(truth=name,total_expected=float(mu.sum()),observed_total=float(observed.sum()),ratio_to_observed_total=float(mu.sum()/observed.sum()),min_expected=float(mu.min()),max_expected=float(mu.max()),max_abs_log_slope=float(abs(slope).max()),max_abs_log_curvature=float(abs(curvature).max()),max_abs_log_third_derivative=float(abs(third).max()),pearson_per_bin=float(np.mean((observed-mu)**2/mu))))
 pd.DataFrame(rows).to_csv(OUT/'construction_checks.csv',index=False,float_format='%.17g')
 definitions={name:('Archived conditional hybrid stress histogram; broad component retains failed source-fit flag.' if name=='archived_stress' else 'Full-support observed-data GP mean, frozen76MeV kernel; self-fit control.' if name=='gp_full_nominal' else 'Full-support observed-data GP mean, same amplitude and half length scale; self-fit control.' if name=='gp_full_half_ls' else 'Fixed-block excluded GP predictions combined by positive C2 partition weights into one spectrum.' if name=='gp_blocked' else 'Poisson-fitted rising/falling log-cubic count models, joined smoothly across50–75MeV.') for name in NAMES}
 manifest={'truth_names':NAMES,'candidate_names':NAMES,'definitions':definitions,'array_file':'backgrounds.npz','array_sha256':sha(OUT/'backgrounds.npz'),'source':str(SOURCE),'source_sha256':sha(SOURCE),'builder_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'block_geometry_sha256':sha(OUT/'block_geometry.npz'),'runtime_sources':{str(p):sha(p) for p in [Path(core.__file__),Path(c.__file__),Path(sys.modules['gp_refit_pilot'].__file__),c.production.DEFAULT_CARD,c.production.DEFAULT_STATES]},'fitted_metadata':meta,'construction_seconds':time.monotonic()-start,'all_positive_finite':True,'same_truth_for_every_test_mass':True,'selection_on_local_roots':False,'workers':1,'BLAS_threads':1}
 write(OUT/'manifest.json',manifest);print(json.dumps(manifest,indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
