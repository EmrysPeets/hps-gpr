#!/usr/bin/env python3
"""Pointwise 90% toy-CLs calibration; one process, resumable hash-bound products."""
from calibration_core import *
import argparse,time,traceback

METHODS=('profiled','fixed')

def observed_q(ctx,method,a):
 mod=ctx.observed[method];f=ctx.ofit[method];free=f['free'];null=f['null']
 if free['A']>a*ctx.sigma:return 0.
 return max(0.,2*(mod.fit(ctx.obs,a*ctx.sigma)['nll']-(free['nll'] if free['A']>=0 else null['nll'])))

def invert(ctx,bank,method):
 trace=[]
 def evaluate(a):
  r=bank.tails(method,a,observed_q(ctx,method,a));trace.append(dict(strength=a,**r));return r
 grid=[(float(a),evaluate(float(a))) for a in bank.nodes if a>0]
 accepted=[a for a,r in grid if r['cls']>=.1]
 if grid[-1][1]['cls']>=.1:return dict(method=method,ul_sigma=float('inf'),status='right_censored',trace=trace)
 low=max(accepted,default=0.);high=min(a for a,r in grid if a>low)
 for _ in range(14):
  mid=(low+high)/2;r=evaluate(mid)
  if r['cls']>.1:low=mid
  else:high=mid
  if high-low<.015*max(mid,.001):break
 center=(low+high)/2;r=evaluate(center)
 # Local slope is only for reported MC error, never used in the test statistic.
 step=max(.05*center,.025);rl=evaluate(max(.001,center-step));rh=evaluate(center+step)
 slope=(rh['cls']-rl['cls'])/(2*step)
 se_ul=r['se']/abs(slope) if slope<0 else float('inf')
 normw,normse=bank.moment(bank.weights(center));normb,normbse=bank.moment(bank.weights(0.))
 ordered=sorted({t['strength']:t for t in trace}.values(),key=lambda t:t['strength'])
 monotone=all(y['cls']-x['cls']<=max(.01,3*np.hypot(x['se'],y['se'])) for x,y in zip(ordered,ordered[1:]))
 resolved=(monotone and (high-low)/center<=.015 and r['ess_b']>=100 and r['ess_s']>=100 and 1.96*se_ul/max(center,.01)<=.10 and abs(normw-1)<=max(.05,5*normse) and abs(normb-1)<=max(.05,5*normbse))
 return dict(method=method,ul_sigma=center,ul_sigma_low=max(0.,center-1.96*se_ul-(high-low)/2),ul_sigma_high=center+1.96*se_ul+(high-low)/2,
  eps2=center*ctx.sigma/ctx.conversion,eps2_low=max(0.,center-1.96*se_ul-(high-low)/2)*ctx.sigma/ctx.conversion,
  eps2_high=(center+1.96*se_ul+(high-low)/2)*ctx.sigma/ctx.conversion,
  monotonicity_passed=monotone,mc_se_sigma=se_ul,bracket_low=low,bracket_high=high,cls=r['cls'],cls_mc_se=r['se'],ess_b=r['ess_b'],ess_s=r['ess_s'],
  normalization=normw,normalization_se=normse,background_normalization=normb,background_normalization_se=normbse,status='resolved' if resolved else 'limited_mc',trace=trace)

def tail_values(q,w,threshold):
 order=np.argsort(q);x=q[order];s=np.r_[np.cumsum(w[order][::-1])[::-1],0.]
 idx=np.searchsorted(x,threshold-1e-10,side='left');return s[idx]/len(q)

def validation(ctx,banks,nvalid):
 rows=[];details=[]
 for truth_name,truth in ctx.truths.items():
  for strength in (0,2,5):
   n=seed('validation',ctx.scope[0],ctx.mass,truth_name,strength).poisson(truth+strength*ctx.sigma*ctx.signal,size=(nvalid,len(truth)))
   models=ctx.make_models(n)
   for method,model in models.items():
    ps=[]
    if strength>0:
     q=model.q(strength*ctx.sigma)
     for bank in banks.values():
      qq=bank.q(method,strength);pb=tail_values(qq,bank.weights(0),q);p=tail_values(qq,bank.weights(strength),q)
      ps.append(np.divide(p,pb,out=np.ones_like(p),where=pb>0))
     cls=np.max(ps,axis=0);excluded=cls<.1
     # Same exact bounded asymptotic construction at the true yield.
     qa=[]
     for i in range(nvalid):
      mod=c.Profile(model.b[i],model.L[i],ctx.w,'linear');qa.append(2*mod.fit(model.b[i],strength*ctx.sigma)['nll'])
     raw=np.array([c.bounded_tildeq_asymptotic_tails(float(qi),float(ai)).cls for qi,ai in zip(q,qa)])
     raw_excluded=raw<.1
    else:
     cls=np.ones(nvalid);excluded=np.zeros(nvalid,bool);raw_excluded=excluded.copy()
    local=[]
    for bank in banks.values():
     p=tail_values(bank.models[method].r,bank.weights(0),model.r);p[model.r<=0]=1.;local.append(np.minimum(p,1.))
    pcal=np.max(local,axis=0);rawp=norm.sf(np.maximum(model.r,0))
    k=int(excluded.sum());fp=int((pcal<.05).sum());kr=int(raw_excluded.sum())
    rows.append(dict(scope_key=ctx.scope[0],mass_MeV=ctx.mass,truth=truth_name,strength=strength,method=method,n=nvalid,
     exclusion_count=k,exclusion_fraction=k/nvalid,exclusion_interval=interval(k,nvalid),raw_exclusion_count=kr,raw_exclusion_fraction=kr/nvalid,
     local_rejection_count=fp,local_rejection_fraction=fp/nvalid,local_rejection_interval=interval(fp,nvalid),raw_local_rejection_fraction=float(np.mean(rawp<.05)),
     Ahat_mean=float(model.free['A'].mean()),Atrue=strength*ctx.sigma,signal_bias_sigma=float((model.free['A'].mean()-strength*ctx.sigma)/ctx.sigma),
     max_score=model.max_score,scalar_fallbacks=model.fallbacks))
    for i in range(nvalid):details.append(dict(truth=truth_name,strength=strength,method=method,toy_id=i,Ahat=model.free['A'][i],signed_r=model.r[i],cls_calibrated=cls[i],calibrated_excluded=bool(excluded[i]),raw_excluded=bool(raw_excluded[i]),p0_calibrated=pcal[i]))
 return rows,details

def run_point(ctx,out,ntoy,nvalid,pilot=False):
 start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
 enable_lowrank(ctx)
 nodes=sorted(set([0.,.5,1.,1.5,2.,3.,4.,5.,6.,8.,12.,float(max(12,np.ceil(ctx.ofit['profiled']['Ahat']/ctx.sigma+6)))]))
 banks={};results=[];provenance={}
 for name,truth in ctx.truths.items():
  proposals,meta=ctx.proposals(truth,nodes)
  rng=seed('calibration',ctx.scope[0],ctx.mass,name,ntoy)
  whole=np.concatenate([rng.poisson(mean,size=(ntoy,len(truth))) for mean in proposals]);strata=np.repeat(np.arange(len(proposals)),ntoy)
  bank=Bank(ctx,truth,whole,proposals,strata);bank.nodes=nodes;banks[name]=bank
  provenance[name]=dict(meta=meta,n=len(whole),truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),proposals_sha256=hashlib.sha256(proposals.tobytes()).hexdigest())
  for method in METHODS:
   result=invert(ctx,bank,method);result.update(truth=name,pzero=bank.pzero(method));results.append(result)
  provenance[name]['whole_sha256']=hashlib.sha256(whole.tobytes()).hexdigest()
  provenance[name]['max_score']=max(model.max_score for model in bank.models.values())
  provenance[name]['fallbacks']=sum(model.fallbacks for model in bank.models.values())
  provenance[name]['weight_checks']={str(a):bank.moment(bank.weights(a)) for a in (0,2,5)}
  print(ctx.scope[0],ctx.mass,name,len(whole),'calibration spectra',[(r['method'],r['ul_sigma'],r['status']) for r in results if r['truth']==name],flush=True)
 valid,details=validation(ctx,banks,nvalid)
 pd.DataFrame(details).to_csv(out/'validation_toys.csv.gz',index=False,compression='gzip')
 pd.DataFrame(valid).to_csv(out/'validation_summary.csv',index=False)
 result=dict(scope_key=ctx.scope[0],mass_MeV=ctx.mass,confidence_level=.9,cls_target=.1,ntoys_per_proposal=ntoy,nvalidation=nvalid,
  sigma_reference=ctx.sigma,signal_yield_per_eps2=ctx.conversion,nodes=nodes,results=results,provenance=provenance,
  numerical_checks=ctx.numerical_checks,scalar_checks=getattr(ctx,'scalar_checks',[]),gp_backend=ctx.gp_backend,gp_fallback_reason=getattr(ctx,'gp_fallback_reason',None),nuisance_eigenvalue_cut=ctx.nuisance_cut,prediction_ledger=ctx.ledger,
  observed={m:{k:f[k] for k in ['A90','Ahat','signed_r']} for m,f in ctx.ofit.items()},
  elapsed_seconds=time.monotonic()-start,status='pilot' if pilot else 'completed_point')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(f'POINT DONE {ctx.scope[0]} {ctx.mass} in {result["elapsed_seconds"]:.1f}s',flush=True)
 return result

def contract(args):
 files=[HERE/n for n in ['run_calibration.py','calibration_core.py','batch_profile.py','gp_refit_pilot.py','gp_lowrank_pilot.py','PROTOCOL.md']]
 files += [c.production.DEFAULT_CARD,c.production.DEFAULT_STATES,Path(c.__file__)]
 files += [p for p,h in STRESS.values()]
 parent=Path(c.production.__file__).parent
 files += [Path(c.production.__file__),parent/'piecewise_cached_solver.py',parent/'runtime/bounded_tildeq_cls.py',parent/'derived/prediction_state_ledger.csv',c.production.DEFAULT_INPUT_PROVENANCE]
 provenance=json.loads(c.production.DEFAULT_INPUT_PROVENANCE.read_text())
 def collect_paths(obj):
  if isinstance(obj,dict):
   for k,v in obj.items():
    if isinstance(v,str) and (k=='path' or k.endswith('_path')) and Path(v).is_file():files.append(Path(v))
    else:collect_paths(v)
  elif isinstance(obj,list):
   for v in obj:collect_paths(v)
 collect_paths(provenance)
 files=list(dict.fromkeys(files))
 files += list((c.production.RUNTIME_CAMPAIGN/'runtime_combined/hps_gpr').glob('*.py'))
 return dict(ntoy=args.ntoy,nvalid=args.nvalid,version=1,hashes={str(p.relative_to(ROOT)):c.sha(p) for p in files})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--ntoy',type=int,default=256);parser.add_argument('--nvalid',type=int,default=500);parser.add_argument('--output',default=None);parser.add_argument('--scope');parser.add_argument('--masses');args=parser.parse_args()
 out=HERE/(args.output or ('pilot' if args.pilot else 'derived'));out.mkdir(exist_ok=True)
 cfg=c.production.load_config(c.production.DEFAULT_CARD);c.production.validate_card(cfg);c.production.validate_histogram_inputs(cfg);c.production.validate_input_provenance(c.production.DEFAULT_INPUT_PROVENANCE,c.production.DEFAULT_CARD,c.production.DEFAULT_STATES,cfg);datasets=c.production.make_datasets(cfg);states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
 frozen=contract(args);path=out/'contract.json'
 if path.exists() and json.loads(path.read_text())!=frozen:raise RuntimeError('Checkpoint contract changed; preserve and use new output')
 path.write_text(json.dumps(frozen,indent=2)+'\n')
 selected=[int(v) for v in args.masses.split(',')] if args.masses else None
 for scope in SCOPES:
  if args.scope and args.scope not in scope[0]:continue
  for mass in range(scope[3],scope[4]+1):
   if selected and mass not in selected:continue
   dest=out/scope[0]/f'm{mass:03}'
   if (dest/'result.json').exists():continue
   try:run_point(Context(scope,mass,cfg,datasets,states),dest,args.ntoy,args.nvalid,args.pilot)
   except Exception:
    dest.mkdir(parents=True,exist_ok=True);(dest/'FAILURE.txt').write_text(traceback.format_exc());raise
 if contract(args)!=frozen:raise RuntimeError('Source drift while running')

if __name__=='__main__':main()
