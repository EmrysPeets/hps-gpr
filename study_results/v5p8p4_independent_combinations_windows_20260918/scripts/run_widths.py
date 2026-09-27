"""Paired one-MeV scans and exact profiled CLs limits for declared blind widths."""
from engine import *
import time,sys
start=time.monotonic();worker=int(sys.argv[1]);workers=int(sys.argv[2]);nulls={y:dict(np.load(B/f'inputs/null_{y}.npz')) for y in YEARS};truths={y:nulls[y]['truth'] for y in YEARS}
old={s:dict(np.load(B/f'inputs/baseline_fields/{s}.npz')) for s in YEARS+['combined']}
jobs=[(w,float(m)) for w in P['blind_halfwidth_sigma'] for m in np.arange(19,251,1.)];done=0
for k,(width,mass) in enumerate(jobs):
 if k%workers!=worker:continue
 tag=f'w{width:.2f}_m{mass:06.1f}';cp=B/f'checkpoints/{tag}.npz';meta=cp.with_suffix('.json')
 if cp.exists() and meta.exists():done+=1;continue
 active=[y for y in YEARS if P['datasets'][y][0]<=mass<=P['datasets'][y][1]]
 try:
  contexts={y:Context(y,mass,width) for y in active};obs={y:contexts[y].predict(nulls[y]['observed']) for y in active}
  if width!=2.25:
   parts={y:contexts[y].predict(truths[y],True) for y in active}
   banks={y:[contexts[y].predict(n) for n in nulls[y]['counts']] for y in active}
  arr={};rows=[];stats={}
  for scope,ys in [(y,[y]) for y in active]+[('combined',active)]:
   if scope=='combined' and len(active)==1:
    row=dict(stats[active[0]],scope=scope);arr[scope+'_D']=arr[active[0]+'_D'];arr[scope+'_validation']=arr[active[0]+'_validation']
   else:
    oo=fit([obs[y] for y in ys]);lim=oo['model'].limit(oo['n']);asimov=oo['model'].limit(oo['model'].b)
    assert abs(lim['signed_r']-oo['r'])<1e-5
    if width==2.25:
     f=old[scope];i=int(np.flatnonzero(f['masses']==mass)[0]);D=f['D'][:,i];roots=f['validation'][:,i];a=float(f['a'][i]);sd=float(f['s'][i]);check={'scalar_error':0.,'max_score':oo['score'],'fallbacks':0,'baseline_reused':True,'baseline_replay_delta':float(oo['r']-f['observed_r'][i])};assert abs(oo['r']-f['observed_r'][i])<2e-5
    else:
     aa=fit([parts[y] for y in ys]);D=response([parts[y] for y in ys],aa,truths);roots,check=batch_fit([banks[y] for y in ys]);a=aa['r'];sd=float(np.linalg.norm(D));check['baseline_reused']=False
    row=dict(scope=scope,width_sigma=width,mass_MeV=mass,datasets='+'.join(ys),a=a,s=sd,observed_r=oo['r'],observed_amplitude=oo['f']['A'],z=(oo['r']-a)/sd,
      epsilon2_90=lim['A90']*1e-8,epsilon2_asimov90=asimov['A90']*1e-8,cls=lim['cls'],asimov_cls=asimov['cls'],cls_branch=lim['cls_branch'],limit_max_score=max(lim['max_score'],asimov['max_score']),min_lambda=min(lim['min_lambda'],asimov['min_lambda']),bins=lim['n_bins'],profiles=lim['n_profiles']+asimov['n_profiles'],**check)
    arr[scope+'_D']=D;arr[scope+'_validation']=roots
   stats[scope]=row;rows.append(row)
  np.savez_compressed(cp,**arr);meta.write_text(json.dumps(rows,indent=2)+'\n');done+=1
  if done%20==0:print(f'worker={worker} done={done} width={width} mass={mass} seconds={time.monotonic()-start:.1f}',flush=True)
 except Exception as exc:
  (B/f'qa/worker_{worker}_failure.json').write_text(json.dumps(dict(width=width,mass=mass,error=repr(exc)),indent=2));raise
q=dict(complete=True,worker=worker,nodes=done,elapsed_seconds=time.monotonic()-start);(B/f'qa/worker_{worker}.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
