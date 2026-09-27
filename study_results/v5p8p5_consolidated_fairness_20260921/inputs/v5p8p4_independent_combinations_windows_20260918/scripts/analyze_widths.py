from engine import *
from combinations import fisher_scores
from scipy.stats import norm,beta
import pandas as pd,time
start=time.monotonic();M=np.arange(19,251,1.);allrows=[];peaks=[];valid=[];gridrows=[];localvalid=[]
def tail(mx,t):
 n=len(mx);k=int(np.sum(mx>=t));lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1));hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
 return dict(k=k,N=n,p=(k+1)/(n+1),raw_p=k/n,lo95=lo,hi95=hi,upper95=1. if k==n else float(beta.ppf(.95,k+1,n-k)),status='upper_bound_only' if k==0 else 'finite_MC')
def curve(scope,width,m,obs_score,local_p,gm,vm,direct_scores,extra=None,domain='full'):
 rows=[]
 for j,mass in enumerate(m):
  score=obs_score[j];g=tail(gm,score);d=tail(vm,score)
  if local_p[j]>=1:g=dict(k=len(gm),N=len(gm),p=1.,raw_p=1.,lo95=1.,hi95=1.,upper95=1.);d=dict(k=len(vm),N=len(vm),p=1.,raw_p=1.,lo95=1.,hi95=1.,upper95=1.)
  display_p=g['upper95'] if g['k']==0 else g['p']
  row=dict(width_sigma=width,scope=scope,domain=domain,mass_MeV=mass,local_p=local_p[j],local_Z=max(0.,float(norm.isf(local_p[j]))),score=score,global_Z=max(0.,float(norm.isf(display_p))),global_Z_is_lower_bound=bool(g['k']==0),**{'global_'+k:v for k,v in g.items()},**{'direct_global_'+k:v for k,v in d.items()})
  if extra is not None:row.update(extra[j])
  rows.append(row)
 allrows.extend(rows);j=int(np.argmin(local_p));p=dict(rows[j],grid_nodes=len(m),mass_min_MeV=float(m.min()),mass_max_MeV=float(m.max()));peaks.append(p)
 # Fixed-source validation of each method at a useful global-tail threshold.
 q95=float(np.quantile(gm,.95));k=int(np.sum(vm>=q95));iv=tail(vm,q95)
 valid.append(dict(width_sigma=width,scope=scope,domain=domain,N_direct=len(vm),exceedances_above_Gaussian_q95=k,fraction_above_Gaussian_q95=k/len(vm),lo95=iv['lo95'],hi95=iv['hi95'],Gaussian_peak_p=rows[j]['global_p'],direct_peak_k=rows[j]['direct_global_k'],direct_peak_N=rows[j]['direct_global_N'],direct_peak_lo95=rows[j]['direct_global_lo95'],direct_peak_hi95=rows[j]['direct_global_hi95']))
 return rows
for wi,width in enumerate(P['blind_halfwidth_sigma']):
 missing=lambda:any(not (B/f'checkpoints/w{width:.2f}_m{mass:06.1f}.json').exists() for mass in M)
 if missing():print('Waiting for completed width',width,flush=True)
 while missing():
  import datetime
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-09-18T22:22:15+00:00'):raise RuntimeError('Time reserve reached before scan completion')
  time.sleep(3)
 banks={s:[] for s in YEARS+['combined']}
 for mass in M:
  cp=B/f'checkpoints/w{width:.2f}_m{mass:06.1f}.npz';meta=cp.with_suffix('.json');assert cp.exists() and meta.exists(),cp;z=np.load(cp)
  for r in json.loads(meta.read_text()):banks[r['scope']].append((r,z[r['scope']+'_D'],z[r['scope']+'_validation']))
 flds={};factors={};gi={};gv={};allgm={};allcm={};observed={};direct={}
 for scope,items in banks.items():
  rr=[x[0] for x in items];m=np.array([r['mass_MeV'] for r in rr]);a=np.array([r['a'] for r in rr]);s=np.array([r['s'] for r in rr]);r=np.array([x['observed_r'] for x in rr]);D=np.column_stack([x[1] for x in items]);V=np.column_stack([x[2] for x in items]);K=D.T@D/np.outer(s,s);K=(K+K.T)/2;ev,U=np.linalg.eigh(K);assert ev.min()>-1e-9
  flds[scope]=dict(m=m,a=a,s=s,r=r,D=D,V=V,K=K,rr=rr,indices=(m-19).astype(int));factors[scope]=U*np.sqrt(np.maximum(ev,0));zobs=(r-a)/s;v=(V-a)/s
  localvalid.append(dict(width_sigma=width,scope=scope,grid_nodes=len(m),null_offset_RMS=float(np.sqrt(np.mean(a*a))),null_offset_max_abs=float(np.max(abs(a))),standardized_mean_RMS=float(np.sqrt(np.mean(v.mean(axis=0)**2))),average_standardized_SD=float(np.mean(v.std(axis=0,ddof=1))),min_standardized_SD=float(np.min(v.std(axis=0,ddof=1))),max_standardized_SD=float(np.max(v.std(axis=0,ddof=1))),correlation_RMSE=float(np.sqrt(np.mean((np.corrcoef(v,rowvar=False)-K)**2)))))
  observed[scope]=np.where(r>0,zobs,-np.inf);direct[scope]=np.where(V>0,v,-np.inf);allgm[scope]=[];allcm[scope]=[]
 # Five groups of identical active datasets allow exact local Fisher mixtures.
 groups=[]
 for k in range(len(M)):
  ys=tuple(y for y in YEARS if P['datasets'][y][0]<=M[k]<=P['datasets'][y][1])
  if not groups or groups[-1][0]!=ys:groups.append((ys,[k]))
  else:groups[-1][1].append(k)
 def combine(W,raw):
  n=next(iter(W.values())).shape[0];F=np.zeros((n,len(M)));S=np.zeros_like(F);classic=np.ones_like(F)
  for ys,inds0 in groups:
   inds=np.array(inds0);zz=[];rr=[];qs=[]
   for y in ys:
    f=flds[y];ix=inds-f['indices'][0];zz.append(W[y][:,ix]);rr.append(raw[y][:,ix]);qs.append(norm.cdf(f['a'][ix]/f['s'][ix]))
   F[:,inds],_,classic[:,inds]=fisher_scores(np.stack(zz),np.stack(rr),np.stack(qs)[:,None,:]);S[:,inds]=np.sum(zz,axis=0)/np.sqrt(len(ys))
  return F,S,classic
 ow={y:((flds[y]['r']-flds[y]['a'])/flds[y]['s'])[None,:] for y in YEARS};orr={y:flds[y]['r'][None,:] for y in YEARS};fo,so,classic=combine(ow,orr)
 vw={y:(flds[y]['V']-flds[y]['a'])/flds[y]['s'] for y in YEARS};vr={y:flds[y]['V'] for y in YEARS};fv,sv,_=combine(vw,vr)
 observed.update(fisher=fo[0],stouffer=so[0]);direct.update(fisher=fv,stouffer=sv)
 for name in ['fisher','stouffer']:allgm[name]=[];allcm[name]=[]
 rng=np.random.default_rng(np.random.SeedSequence([58420260918,wi]));N=P['gaussian_fields'];over=(M>=50)&(M<=100)
 for first in range(0,N,2048):
  n=min(2048,N-first);W={};raw={}
  for scope,f in flds.items():
   w=rng.standard_normal((n,len(f['m'])))@factors[scope].T;rr=f['a']+f['s']*w;score=np.where(rr>0,w,-np.inf);allgm[scope].append(score.max(axis=1));cm=(f['m']>=50)&(f['m']<=100);allcm[scope].append(score[:,cm].max(axis=1))
   if scope in YEARS:W[scope]=w;raw[scope]=rr
  F,S,_=combine(W,raw)
  for name,score in [('fisher',F),('stouffer',S)]:allgm[name].append(score.max(axis=1));allcm[name].append(score[:,over].max(axis=1))
 for name in allgm:allgm[name]=np.concatenate(allgm[name]);allcm[name]=np.concatenate(allcm[name])
 for scope,f in flds.items():
  z=(f['r']-f['a'])/f['s'];lp=np.where(f['r']>0,norm.sf(z),1.);dm=direct[scope].max(axis=1);curve(scope,width,f['m'],observed[scope],lp,allgm[scope],dm,direct[scope],f['rr'])
  np.savez_compressed(B/f'fields/w{width:.2f}_{scope}.npz',masses=f['m'],a=f['a'],s=f['s'],D=f['D'],K=f['K'],observed_r=f['r'],validation=f['V'],gaussian_maximum=allgm[scope],gaussian_overlap_maximum=allcm[scope],direct_maximum=dm)
  if scope=='combined':
   curve(scope,width,M[over],observed[scope][over],lp[over],allcm[scope],direct[scope][:,over].max(axis=1),direct[scope][:,over],[f['rr'][i] for i in np.flatnonzero(over)],domain='overlap')
 for name in ['fisher','stouffer']:
  lp=np.exp(-observed[name]) if name=='fisher' else norm.sf(observed[name]);extras=[dict(classic_Fisher_local_p=float(classic[0,i])) for i in range(len(M))] if name=='fisher' else None
  curve(name,width,M,observed[name],lp,allgm[name],direct[name].max(axis=1),direct[name],extras)
  curve(name,width,M[over],observed[name][over],lp[over],allcm[name],direct[name][:,over].max(axis=1),direct[name][:,over],None,domain='overlap')
  np.savez_compressed(B/f'fields/w{width:.2f}_{name}.npz',masses=M,observed_score=observed[name],local_p=lp,validation_score=direct[name],gaussian_maximum=allgm[name],gaussian_overlap_maximum=allcm[name],direct_maximum=direct[name].max(axis=1))
 print('width',width,'done seconds',round(time.monotonic()-start,1),flush=True)
 pd.DataFrame(allrows).to_csv(B/'results/significance_and_reach.csv',index=False,float_format='%.17g');pd.DataFrame(peaks).to_csv(B/'results/peaks.csv',index=False,float_format='%.17g');pd.DataFrame(valid).to_csv(B/'results/global_validation.csv',index=False,float_format='%.17g')
 pd.DataFrame(localvalid).to_csv(B/'results/local_validation.csv',index=False,float_format='%.17g')
# Paired prior 200k fields isolate the baseline grid effect at the SAME threshold.
for scope in YEARS+['combined']:
 f=np.load(B/f'inputs/baseline_fields/{scope}.npz');m=f['masses'];z=(f['observed_r']-f['a'])/f['s'];el=np.where(f['observed_r']>0,z,-np.inf);co=np.isclose(m,np.round(m));jfine=np.argmax(el);jco=np.flatnonzero(co)[np.argmax(el[co])]
 for step,j,key in [(.5,jfine,'gaussian_maximum'),(1.,jco,'gaussian_coarse_maximum')]:
  t=tail(f[key],el[j]);t3=tail(f[key],3.);gridrows.append(dict(scope=scope,step_MeV=step,peak_mass_MeV=m[j],local_Z=el[j],global_Z=max(0.,float(norm.isf(t['p']))),**{'global_'+k:v for k,v in t.items()},p_at_fixed_Z3=t3['p']))
pd.DataFrame(gridrows).to_csv(B/'results/paired_grid_comparison.csv',index=False,float_format='%.17g')
(B/'qa/analysis_execution.json').write_text(json.dumps(dict(complete=True,rows=len(allrows),peak_rows=len(peaks),gaussian_fields_per_scope_width=P['gaussian_fields'],elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
print('analysis complete',len(allrows),len(peaks),flush=True)
