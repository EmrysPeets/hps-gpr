"""Full-spectrum, fixed-kernel bootstrap and exact Poisson importance weights."""
from pathlib import Path
import os,sys,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'study_results/background_profile_comparison_20260905'))
import run_comparison as c
from gp_refit_pilot import CachedCholeskyPredictor
from batch_profile import BatchProfile
from scipy.linalg import cho_solve,cholesky,block_diag
from scipy.special import logsumexp
from scipy.stats import norm,beta
import uproot
np,pd=c.np,c.pd
SCOPES=[s for s in c.production.SCOPES if s[0].startswith(('individual_','all_'))]
STRESS={
 '2015':(ROOT/'outputs/funcform_toys/funcform_2015_dataset_mod_toys.root','validation/fShiftSigPowTail_expected_counts'),
 '2016':(ROOT/'study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/2016_threshold_qualified_background_toys_100.root','truth/threshold_qualified/2016_full_mean'),
 '2021':(ROOT/'study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/inputs/native10_fsig_background_toys_100.root','truth/fsig_anchor/2021_10pct_mean')}

def seed(*args):
 raw=hashlib.sha256(('49130905|'+'|'.join(map(str,args))).encode()).digest()
 return np.random.default_rng(np.random.SeedSequence(np.frombuffer(raw[:16],dtype='<u4')))

def interval(k,n):
 return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]

def stress_truth(key,p):
 path,h=STRESS[key]
 if h is None:
  d=pd.read_csv(path)
  if not np.allclose(d.mass_MeV.to_numpy()/1000,p.x_full,rtol=0,atol=1e-15):raise RuntimeError('Truth bin mismatch')
  return d.smooth_truth_counts.to_numpy(float)
 y,edges=uproot.open(path)[h].to_numpy()
 idx=np.array([np.argmin(abs(edges-v)) for v in p.edges_full])
 if np.max(abs(edges[idx]-p.edges_full))>1e-12:raise RuntimeError('Rebin edge mismatch')
 total=np.r_[0.,np.cumsum(y)]
 return np.diff(total[idx])

class Context:
 def __init__(self,scope,mass,cfg,datasets,states):
  self.scope=scope;self.mass=mass;self.cfg=cfg;self.keys=scope[2];self.parts=[]
  preds,covs,_,ledger=c.production.reconstruct_predictions(mass/1000,datasets,cfg,states)
  self.ledger=[r for r in ledger if r['dataset'] in self.keys]
  saved=pd.read_csv(Path(c.production.__file__).parent/'derived/prediction_state_ledger.csv')
  saved['dataset']=saved.dataset.astype(str);saved=saved.set_index(['dataset','mass_MeV'])
  for r in self.ledger:
   if r['prediction_state_sha256']!=saved.loc[(r['dataset'],mass),'prediction_state_sha256']:raise RuntimeError('Frozen prediction hash mismatch')
  obs,b,_,su=c.production.build_combined_components(mass/1000,[datasets[k] for k in self.keys],[preds[k] for k in self.keys],config=cfg)
  self.conversion=float(su.sum());self.w=su/self.conversion;self.b=b;self.obs=obs
  C=c.production.block_diagonal([covs[k] for k in self.keys]);self.L=c._chol_with_jitter(C)
  self.sigma=1/np.sqrt(self.w@np.linalg.solve(np.diag(b)+C,self.w))
  self.observed={m:c.Profile(b,self.L if m=='profiled' else np.zeros((len(b),0)),self.w,'linear') for m in ('profiled','fixed')}
  self.ofit={m:mod.limit(obs) for m,mod in self.observed.items()}
  nominal=[];stress=[];signal=[];mask=[]
  for k in self.keys:
   p=preds[k];st=states[k,mass];ker=c.make_fixed_kernel(st['const_opt'],st['ls_opt']);keep=~p.blind_mask
   gp=c.fit_gpr(p.x_full[keep],p.y_full[keep],cfg,restarts=0,kernel=ker,optimize=False)
   from hps_gpr.gpr import predict_counts_mean_var_from_log_gpr
   truth,_=predict_counts_mean_var_from_log_gpr(gp,p.x_full,cfg)
   _,full=c.build_window_template_from_full(p.edges_full,p.blind_mask,mass/1000,p.sigma_val,config=cfg)
   scale=c.production.A_from_epsilon2(datasets[k],mass/1000,1.,p.integral_density)/self.conversion
   predictor=CachedCholeskyPredictor(p.x_full[keep],p.x_full[p.blind_mask],ker,cfg)
   self.parts.append(dict(key=k,p=p,predictor=predictor,exact_predictor=predictor,kernel=ker,n=len(p.x_full),keep=keep))
   nominal.append(truth);stress.append(stress_truth(k,p));signal.append(full*scale);mask.append(p.blind_mask)
  self.truths={'gp':np.concatenate(nominal),'stress':np.concatenate(stress)}
  self.signal=np.concatenate(signal);self.mask=np.concatenate(mask)
  if not np.allclose(self.signal[self.mask],self.w,rtol=1e-12,atol=1e-15):raise RuntimeError('Full/window signal normalization')
  if any(np.any(v<=0) for v in self.truths.values()):raise RuntimeError('Nonpositive generating truth')
  self.offsets=np.r_[0,np.cumsum([p['n'] for p in self.parts])]
  self.numerical_checks=[];self.scalar_checks=[];self.scalar_check_batches=0;self.nuisance_cut=0.

 def retrain(self,whole):
  bs=[];Ls=[]
  for j,part in enumerate(self.parts):
   y=whole[self.offsets[j]:self.offsets[j+1]]
   b,C=part['predictor'].predict(y[part['keep']]);C,_=c.production.condition_covariance_block(C,b)
   bs.append(b)
   if self.nuisance_cut:
    sd=np.sqrt(b);v,U=np.linalg.eigh(C/sd[:,None]/sd[None,:]);keep=v>self.nuisance_cut
    factor=sd[:,None]*U[:,keep]*np.sqrt(v[keep]);width=min(12,len(b))
    if factor.shape[1]>width:raise RuntimeError('Nuisance rank exceeds declared padding')
    Ls.append(np.pad(factor,((0,0),(0,width-factor.shape[1]))))
   else:Ls.append(c._chol_with_jitter(C))
  return np.concatenate(bs),block_diag(*Ls)

 def influence(self,truth,method='profiled'):
  """Linearized profile response solely to design proposals; never the statistic."""
  b,L=self.retrain(truth);C=L@L.T
  weight=np.linalg.solve(np.diag(b)+C,self.w) if method=='profiled' else self.w/b;weight/=self.w@weight
  gradient=np.zeros(len(truth));gradient[self.mask]=weight
  offset=0
  from hps_gpr.gpr import preprocess_xy_for_gpr
  for j,part in enumerate(self.parts):
   yy=truth[self.offsets[j]:self.offsets[j+1]][part['keep']]
   gp=part.get('exact_predictor',part['predictor']);_,target,alpha=preprocess_xy_for_gpr(gp.x_train,yy,self.cfg)
   M=gp.K.copy();M[gp.diagonal]+=alpha;factor=cholesky(M,lower=True)
   coeff=cho_solve((factor,True),target);h=cho_solve((factor,True),gp.Kqt.T).T
   nb=len(gp.x_query);bw=b[offset:offset+nb];ww=weight[offset:offset+nb]
   # In active log-counts alpha=1/y lane; reject unhandled first-bin rules.
   if not self.cfg.pre_log or not np.allclose(alpha,1/yy,rtol=0,atol=0):raise RuntimeError('Proposal derivative unsupported alpha')
   J=bw[:,None]*(h*(1/yy+coeff/yy**2)-.5*h*h/yy**2)
   local=gradient[self.offsets[j]:self.offsets[j+1]];local[part['keep']]=-ww@J
   offset+=nb
  bias=float(weight@(truth[self.mask]-b));sd=np.sqrt(np.sum(truth*gradient**2))
  return gradient,bias,sd

 def proposals(self,truth,nodes):
  infos={method:self.influence(truth,method) for method in ('profiled','fixed')}
  means=[];labels=[]
  for a in nodes:
   mean=truth+a*self.sigma*self.signal;means.append(mean);labels.append([float(a),'unshifted',0.])
   for method,(g,bias,sd) in infos.items():
    target_a=self.ofit[method]['Ahat']
    shift=np.clip((target_a-bias-a*self.sigma)/sd,-12.,12.)
    tilted=mean+shift*truth*g/sd
    if np.any(tilted<=0):raise RuntimeError('Nonpositive IS proposal')
    means.append(tilted);labels.append([float(a),method,float(shift)])
  return np.array(means),dict(nodes=list(map(float,nodes)),labels=labels,influence={k:dict(bias=v[1],sd=v[2]) for k,v in infos.items()})

 def make_models(self,whole):
  b=[];L=[]
  for n in whole:
   bb,ll=self.retrain(n);b.append(bb);L.append(ll)
  b=np.array(b);rank=max(ll.shape[1] for ll in L);L=np.array([np.pad(ll,((0,0),(0,rank-ll.shape[1]))) for ll in L]);counts=whole[:,self.mask]
  active=np.any(L!=0.,axis=(0,1));L=L[:,:,active]
  blocks=[];rr=0;cc=0;original_col=0
  for part in self.parts:
   rows=int(part['p'].blind_mask.sum());cols=min(12,rows) if self.nuisance_cut else rows
   kept=int(active[original_col:original_col+cols].sum());original_col+=cols
   blocks.append((rr,rr+rows,cc,cc+kept));rr+=rows;cc+=kept
  models={method:BatchProfile(counts,b,L if method=='profiled' else np.zeros((len(b),len(self.b),0)),self.w,blocks if method=='profiled' else None) for method in ('profiled','fixed')}
  batch_id=self.scalar_check_batches;self.scalar_check_batches+=1
  for method,model in models.items():
   for i in range(min(2,len(b))):
    scalar=c.Profile(model.b[i],model.L[i],self.w,'linear');f=scalar.fit(counts[i]);z=scalar.fit(counts[i],0.)
    r=np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll'])))
    check=dict(batch_id=batch_id,n_spectra=len(b),method=method,toy_index=i,
     counts_sha256=hashlib.sha256(counts[i].tobytes()).hexdigest(),scalar_r=float(r),batch_r=float(model.r[i]),
     r_error=float(abs(r-model.r[i])),q_checks=[],passed=False)
    self.scalar_checks.append(check)
    if check['r_error']>2e-5:raise RuntimeError('Batch/scalar signed-r disagreement')
    for a in (2,5):
     fixed=scalar.fit(counts[i],a*self.sigma);q=0. if f['A']>a*self.sigma else max(0.,2*(fixed['nll']-(f['nll'] if f['A']>=0 else z['nll'])))
     # Check an isolated two-row batch to avoid extra fits of the complete bank.
     tiny=BatchProfile(counts[i:i+1],b[i:i+1],model.L[i:i+1],self.w,blocks if method=='profiled' else None)
     batch_q=float(tiny.q(a*self.sigma)[0]);error=float(abs(q-batch_q))
     check['q_checks'].append(dict(strength_sigma=a,scalar_q=float(q),batch_q=batch_q,q_error=error))
     if error>1e-4:raise RuntimeError('Batch/scalar q disagreement')
    check['passed']=True
  return models

class Bank:
 def __init__(self,ctx,truth,whole,proposals,strata):
  self.ctx=ctx;self.truth=truth;self.whole=whole;self.strata=strata;self.n=len(whole);self.K=len(proposals)
  self.logmix=logsumexp(whole@np.log(proposals/truth).T-np.sum(proposals-truth,axis=1),axis=1)-np.log(self.K)
  self.models=ctx.make_models(whole);self.qcache={}
 def weights(self,a):
  delta=a*self.ctx.sigma*self.ctx.signal
  logtarget=self.whole@np.log1p(delta/self.truth)-np.sum(delta)
  return np.exp(logtarget-self.logmix)
 def moment(self,values):
  values=np.asarray(values).reshape(self.K,-1)
  return float(values.mean()),float(np.sqrt(np.sum(values.var(axis=1,ddof=1)/values.shape[1])/self.K**2))

 def q(self,method,a):
  key=(method,float(a))
  if key not in self.qcache:self.qcache[key]=self.models[method].q(a*self.ctx.sigma)
  return self.qcache[key]
 def tails(self,method,a,threshold):
  if threshold<=0:return dict(cls=1.,se=0.,pb=1.,ps=1.,ess_b=float(self.n),ess_s=float(self.n),se_b=0.,se_s=0.)
  indicator=self.q(method,a)>=threshold-1e-10
  wb=self.weights(0.);ws=self.weights(a)
  pb,seb=self.moment(wb*indicator);ps,ses=self.moment(ws*indicator)
  if pb<=0:return dict(cls=float('inf'),se=float('inf'),pb=pb,ps=ps,ess_b=0.,ess_s=0.)
  ratio=ps/pb;_,se=self.moment((ws-ratio*wb)*indicator/pb)
  ess=lambda v:float(v.sum()**2/np.sum(v*v)) if np.sum(v*v)>0 else 0.
  return dict(cls=ratio,se=se,pb=pb,ps=ps,ess_b=ess(wb*indicator),ess_s=ess(ws*indicator),se_b=seb,se_s=ses)
 def pzero(self,method):
  r=self.ctx.ofit[method]['signed_r']
  if r<=0:return dict(p0=1.,se=0.,ess=float(self.n),status='bounded_atom')
  values=self.weights(0.)*(self.models[method].r>=r)
  value,se=self.moment(values);ess=values.sum()**2/np.sum(values**2) if np.any(values) else 0.
  return dict(p0=value,se=se,ess=float(ess),status='resolved' if ess>=100 else 'limited_mc')

def enable_lowrank(ctx):
 """Predeclared per-coordinate numerical approximation gate, with exact fallback."""
 exact=[p.get('exact_predictor',p['predictor']) for p in ctx.parts]
 records=[];passed=True
 ctx.numerical_checks=records;ctx.gp_fallback_reason=None
 def restore_exact():
  ctx.nuisance_cut=0.
  for p,pred in zip(ctx.parts,exact):p['predictor']=pred
  ctx.gp_backend='exact_cached_cholesky'
 def reject_exception(stage,error,**coordinate):
  restore_exact()
  reason=dict(stage=stage,error_type=type(error).__name__,error=str(error),**coordinate)
  ctx.gp_fallback_reason=reason
  records.append(dict(passed=False,status='approximation_exception',fallback_reason=reason))
  return False
 restore_exact()
 try:
  from gp_lowrank_pilot import LowRankPredictor
  approximate=[LowRankPredictor(p['p'].x_full[p['keep']],p['p'].x_full[p['p'].blind_mask],p['kernel'],ctx.cfg,rtol=1e-15) for p in ctx.parts]
 except Exception as error:
  return reject_exception('eigenfeature_construction',error)
 rng=seed('numeric-audit',ctx.scope[0],ctx.mass)
 for truth_name,truth in ctx.truths.items():
  props,_=ctx.proposals(truth,[0.,2.,5.])
  for i,mean in enumerate(props):
   n=rng.poisson(mean).astype(float)
   # Exact predictions and fits deliberately remain outside the approximation
   # exception handler: a failing reference must halt the affected execution.
   b0,L0=ctx.retrain(n)
   C0=L0@L0.T;baseline={}
   for method in ('profiled','fixed'):
    mod=BatchProfile(n[ctx.mask][None,:],b0[None,:],L0[None,:,:] if method=='profiled' else np.zeros((1,len(b0),0)),ctx.w)
    baseline[method]=dict(r=float(mod.r[0]),q={a:float(mod.q(a*ctx.sigma)[0]) for a in (2,5,12)})
   stage='approximate_prediction_or_covariance_compression'
   try:
    ctx.nuisance_cut=1e-5
    for p,pred in zip(ctx.parts,approximate):p['predictor']=pred
    b1,L1=ctx.retrain(n);C1=L1@L1.T
    row=dict(truth=truth_name,proposal=i,mean_error_sd=float(np.max(abs(b1-b0)/np.sqrt(np.diag(C0)))),cov_error=float(np.max(abs(C1-C0))/np.diag(C0).max()),q_error=0.,r_error=0.)
    stage='approximate_profile_fit'
    for method in ('profiled','fixed'):
     mod=BatchProfile(n[ctx.mask][None,:],b1[None,:],L1[None,:,:] if method=='profiled' else np.zeros((1,len(b1),0)),ctx.w)
     row['r_error']=max(row['r_error'],float(abs(baseline[method]['r']-mod.r[0])))
     for a in (2,5,12):row['q_error']=max(row['q_error'],float(abs(baseline[method]['q'][a]-mod.q(a*ctx.sigma)[0])))
   except Exception as error:
    return reject_exception(stage,error,truth=truth_name,proposal=i)
   finally:
    restore_exact()
   row['passed']=all(row[k]<1e-3 for k in ('mean_error_sd','cov_error','r_error','q_error'))
   passed=passed and row['passed'];records.append(row)
 ctx.nuisance_cut=1e-5 if passed else 0.
 for p,pred in zip(ctx.parts,approximate if passed else exact):p['predictor']=pred
 ctx.gp_backend='eigenfeature_rtol_1e-15' if passed else 'exact_cached_cholesky'
 if not passed:ctx.gp_fallback_reason=dict(stage='discrepancy_gate',failed_checks=sum(not row['passed'] for row in records))
 return passed
