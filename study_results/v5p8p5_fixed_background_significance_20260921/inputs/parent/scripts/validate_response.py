from engine import *
import time,pandas as pd
start=time.monotonic();truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in YEARS};rows=[];baseline=[]
for mass,years in [(39,['2015','2016']),(51,YEARS),(76,YEARS),(92,YEARS),(117,['2016','2021']),(200,['2021'])]:
 ctxs={y:Context(y,mass) for y in years};parts={y:ctxs[y].predict(truths[y],True) for y in years}
 for scope,ys in [(y,[y]) for y in years]+([('combined',years)] if len(years)>1 else []):
  base=fit([parts[y] for y in ys]);D=response([parts[y] for y in ys],base,truths);baseline.append(dict(mass_MeV=mass,scope=scope,a=base['r'],s=np.linalg.norm(D)))
  for y in ys:
   ctx=ctxs[y];idx=np.unique(np.r_[np.linspace(0,len(truths[y])-1,5).astype(int),np.flatnonzero(ctx.mask)[[0,len(np.flatnonzero(ctx.mask))//2,-1]]])
   for j in idx:
    vals=[]
    for scale in [-.02,.02,1.]:
     t=truths[y].copy();t[j]+=scale*np.sqrt(t[j]);pp=ctx.predict(t);new=[pp if yy==y else parts[yy] for yy in ys];vals.append(fit(new)['r'])
    fd=(vals[1]-vals[0])/.04;exact=vals[2]-base['r'];dv=D[OFFSET[y]+j]
    rows.append(dict(mass_MeV=mass,scope=scope,perturbed_dataset=y,bin_index=int(j),analytic=dv,finite_derivative=fd,finite_error=fd-dv,one_sigma=exact,one_sigma_error=exact-dv))
 print(mass,round(time.monotonic()-start,2),flush=True)
d=pd.DataFrame(rows);d.to_csv(B/'qa/response_derivative_checks.csv',index=False);pd.DataFrame(baseline).to_csv(B/'qa/pilot_baselines.csv',index=False)
q={'passed':bool(abs(d.finite_error).max()<5e-4),'max_finite_difference_error':float(abs(d.finite_error).max()),'max_one_sigma_difference':float(abs(d.one_sigma_error).max()),'checks':len(d),'seconds':time.monotonic()-start,'approximation':'GP covariance-conditioning derivative neglected below recorded numerical scale; checked against recomputed conditioned fits'}
(B/'qa/response_derivative_validation.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2));assert q['passed']
