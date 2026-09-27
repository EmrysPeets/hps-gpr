from engine import *
import time,pandas as pd
start=time.monotonic();truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in YEARS};rows=[]
for width in P['blind_halfwidth_sigma']:
 for mass in [39.,66.,92.,180.,250.]:
  ys=[y for y in YEARS if P['datasets'][y][0]<=mass<=P['datasets'][y][1]];ctx={y:Context(y,mass,width) for y in ys};parts={y:ctx[y].predict(truths[y],True) for y in ys}
  for scope,subset in [(y,[y]) for y in ys]+[('combined',ys)]:
   aa=fit([parts[y] for y in subset]);D=response([parts[y] for y in subset],aa,truths)
   for y in subset:
    ind=np.flatnonzero(ctx[y].mask);js=np.unique([int(ind[0]),int(ind[len(ind)//2]),max(0,int(ind[0])-1)])
    for j in js:
     rr=[]
     for sign in [-1,1]:
      n=truths[y].copy();n[j]+=sign*.02*np.sqrt(n[j]);p=ctx[y].predict(n);rr.append(fit([p if yy==y else parts[yy] for yy in subset])['r'])
     fd=(rr[1]-rr[0])/.04;rows.append(dict(width_sigma=width,mass=mass,scope=scope,year=y,bin=int(j),analytic=D[OFFSET[y]+j],finite_difference=fd,error=fd-D[OFFSET[y]+j]))
d=pd.DataFrame(rows);d.to_csv(B/'qa/width_derivative_checks.csv',index=False);q=dict(passed=bool(abs(d.error).max()<5e-4),checks=len(d),maximum_error=float(abs(d.error).max()),seconds=time.monotonic()-start)
(B/'qa/width_derivative_validation.json').write_text(json.dumps(q,indent=2)+'\n');print(q);assert q['passed']
