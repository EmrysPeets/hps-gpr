from engine import *
import time,pandas as pd
start=time.monotonic();truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in YEARS};rows=[];vectors={}
class Moments:
 def __init__(self,ctx,n):
  self.ctx=ctx;self.n=n;self.y=n[ctx.keep];M=ctx.K.copy();M.flat[::len(M)+1]+=1/self.y;f=cho_factor_local(M);self.co=cho_solve(f,np.log(self.y));self.H=cho_solve(f,ctx.Kqt.T).T;self.idiag=np.diag(cho_solve(f,np.eye(len(M))))
  v=solve_triangular(f[0],ctx.Kqt.T,lower=True);self.cl=ctx.Kqq-v.T@v;self.mu=ctx.Kqt@self.co;self.base=ctx.predict(n);self.index={j:i for i,j in enumerate(np.flatnonzero(ctx.keep))}
 def shifted(self,j):
  n=self.n.copy();n[j]+=np.sqrt(n[j]);idx=self.index.get(j)
  if idx is None:return dict(self.base,n=n[self.ctx.mask])
  y=self.y[idx];delta=np.sqrt(y);da=1/(y+delta)-1/y;dt=np.log1p(delta/y);den=1+da*self.idiag[idx];h=self.H[:,idx]
  mu=self.mu+h*(dt-da*self.co[idx])/den;cl=self.cl+(da/den)*np.outer(h,h);cl=.5*(cl+cl.T);b=np.exp(mu+.5*np.maximum(np.diag(cl),0));raw=np.outer(b,b)*np.expm1(np.clip(cl,-40,40));L,diag=C.factor_cov(raw,b)
  return dict(self.base,n=n[self.ctx.mask],b=b,L=L,Craw=raw,load=diag['load'])
for mass,ys,scope in [(51,['2015'],'2015'),(90,['2016'],'2016'),(92,['2021'],'2021'),(76,YEARS,'combined')]:
 ctxs={y:Context(y,mass) for y in ys};parts={y:ctxs[y].predict(truths[y],True) for y in ys};base=fit(list(parts.values()));ana=response(list(parts.values()),base,truths);exact=np.zeros(TOTAL)
 for y in ys:
  mm=Moments(ctxs[y],truths[y])
  for j in range(len(truths[y])):
   pp=mm.shifted(j);exact[OFFSET[y]+j]=fit([pp if yy==y else parts[yy] for yy in ys])['r']-base['r']
  # Check moment update against direct Cholesky at three independent directions.
  for j in [0,len(truths[y])//2,len(truths[y])-1]:
   nn=truths[y].copy();nn[j]+=np.sqrt(nn[j]);direct=ctxs[y].predict(nn);fast=mm.shifted(j);assert np.max(abs(direct['b']-fast['b'])/direct['b'])<2e-8
 vectors[scope+'_analytic']=ana;vectors[scope+'_one_sigma']=exact
 rows.append(dict(scope=scope,mass_MeV=mass,directions=sum(len(truths[y]) for y in ys),analytic_sd=np.linalg.norm(ana),one_sigma_sd=np.linalg.norm(exact),relative_sd_difference=np.linalg.norm(exact)/np.linalg.norm(ana)-1,max_response_difference=np.max(abs(exact-ana)),cosine=np.dot(exact,ana)/(np.linalg.norm(exact)*np.linalg.norm(ana))))
 print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(B/'qa/full_response_checks.csv',index=False);np.savez_compressed(B/'qa/full_response_vectors.npz',**vectors)
q={'passed':all(abs(r['relative_sd_difference'])<.01 and r['cosine']>.999 for r in rows),'total_one_sigma_directions':sum(r['directions'] for r in rows),'maximum_relative_sd_difference':max(abs(r['relative_sd_difference']) for r in rows),'minimum_response_cosine':min(r['cosine'] for r in rows),'elapsed_seconds':time.monotonic()-start}
(B/'qa/full_response_validation.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2));assert q['passed']
