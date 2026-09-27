from engine import *
import sys,time,hashlib
start=time.monotonic();worker=int(sys.argv[1]);workers=int(sys.argv[2]);names=YEARS+['combined']
assert json.loads((B/'qa/response_derivative_validation.json').read_text())['passed']
nulls={y:dict(np.load(B/f'inputs/null_{y}.npz')) for y in YEARS};truths={y:nulls[y]['truth'] for y in YEARS}
masses=np.arange(19,250.001,.5);done=0;errors=[]
for k,mass in enumerate(masses):
 if k%workers!=worker:continue
 cp=B/f'checkpoints/m{mass:07.2f}.npz';meta=cp.with_suffix('.json')
 if cp.exists() and meta.exists():done+=1;continue
 active=[y for y in YEARS if P['datasets'][y][0]<=mass<=P['datasets'][y][1]]
 try:
  contexts={y:Context(y,float(mass)) for y in active}
  parts={y:contexts[y].predict(truths[y],True) for y in active};obs={y:contexts[y].predict(nulls[y]['observed']) for y in active}
  banks={y:[contexts[y].predict(n) for n in nulls[y]['counts']] for y in active}
  scoped=[];arr={'mass':mass};stats={}
  for scope,ys in [(y,[y]) for y in active]+[('combined',active)]:
   if scope=='combined' and len(active)==1:
    prior=stats[active[0]];row=dict(prior,scope=scope);arr[scope+'_D']=arr[active[0]+'_D'];arr[scope+'_validation']=arr[active[0]+'_validation']
   else:
    aa=fit([parts[y] for y in ys]);oo=fit([obs[y] for y in ys]);D=response([parts[y] for y in ys],aa,truths)
    roots,check=batch_fit([banks[y] for y in ys]);sd=float(np.linalg.norm(D));assert sd>0 and np.isfinite(sd)
    row=dict(scope=scope,mass_MeV=float(mass),datasets='+'.join(ys),a=aa['r'],s=sd,observed_r=oo['r'],observed_amplitude=oo['f']['A'],source_amplitude=aa['f']['A'],z=(oo['r']-aa['r'])/sd,observed_score=oo['score'],asimov_score=aa['score'],validation_mean=float(roots.mean()),validation_sd=float(roots.std(ddof=1)),centered_validation_mean=float(((roots-aa['r'])/sd).mean()),centered_validation_sd=float(roots.std(ddof=1)/sd),**check)
    arr[scope+'_D']=D;arr[scope+'_validation']=roots
   stats[scope]=row;scoped.append(row)
  np.savez_compressed(cp,**arr);meta.write_text(json.dumps(scoped,indent=2)+'\n');done+=1
  if done%10==0:print('worker',worker,'done',done,'mass',mass,'seconds',round(time.monotonic()-start,1),flush=True)
 except Exception as exc:
  errors.append(dict(mass_MeV=float(mass),error=repr(exc)));(B/f'qa/worker_{worker}_failures.json').write_text(json.dumps(errors,indent=2)+'\n');print('FAILED',mass,repr(exc),flush=True)
  raise
q={'complete':True,'worker':worker,'workers':workers,'nodes':done,'elapsed_seconds':time.monotonic()-start,'failures':errors};(B/f'qa/worker_{worker}.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
