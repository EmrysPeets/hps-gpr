from engine import *
import pandas as pd,time
from scipy.stats import norm,beta
start=time.monotonic();masses=np.arange(19,250.001,.5);records=[];banks={s:[] for s in YEARS+['combined']}
for m in masses:
 cp=B/f'checkpoints/m{m:07.2f}.npz';meta=cp.with_suffix('.json');assert cp.exists() and meta.exists(),cp
 z=np.load(cp)
 for r in json.loads(meta.read_text()):
  records.append(r);banks[r['scope']].append((r,z[r['scope']+'_D'],z[r['scope']+'_validation']))
def interval(k,n):return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k)))
def tail(maxima,t):
 k=int(np.count_nonzero(maxima>=t));n=len(maxima);lo,hi=interval(k,n)
 return dict(k=k,N=n,p=k/n,p_addone=(k+1)/(n+1),lo95=lo,hi95=hi,upper95=1. if k==n else float(beta.ppf(.95,k+1,n-k)),status='upper_bound_only' if k==0 else 'all_exceedances' if k==n else 'finite_MC')
allrows=[];summary=[];validation=[];convergence=[]
for si,scope in enumerate(banks):
 items=banks[scope];rr=[v[0] for v in items];m=np.array([r['mass_MeV'] for r in rr]);a=np.array([r['a'] for r in rr]);s=np.array([r['s'] for r in rr]);obs=np.array([r['observed_r'] for r in rr]);D=np.column_stack([v[1] for v in items]);V=np.column_stack([v[2] for v in items]);K=D.T@D/np.outer(s,s);K=(K+K.T)/2
 eig,U=np.linalg.eigh(K);fac=U*np.sqrt(np.maximum(eig,0));rng=np.random.default_rng(np.random.SeedSequence([P['seed_base'],si,991]));N=P['gaussian_fields'];maxima=[];coarse=[];overlap=[];rawmax=[];coarse_mask=np.isclose(m,np.round(m));common_mask=(m>=50)&(m<=100)
 for begin in range(0,N,4096):
  W=rng.standard_normal((min(4096,N-begin),len(m)))@fac.T;R=a+s*W;G=np.where(R>0,W,-np.inf)
  maxima.append(G.max(axis=1));coarse.append(G[:,coarse_mask].max(axis=1));overlap.append(G[:,common_mask].max(axis=1));rawmax.append(np.maximum(0,R.max(axis=1)))
 maxima=np.concatenate(maxima);coarse=np.concatenate(coarse);overlap=np.concatenate(overlap);rawmax=np.concatenate(rawmax)
 assert np.all(maxima>=coarse)
 Zv=(V-a)/s;zobs=(obs-a)/s;Gv=np.where(V>0,Zv,-np.inf);Mv=Gv.max(axis=1);Sv=np.sort(maxima);Sraw=np.sort(rawmax)
 for j,r in enumerate(rr):
  positive=obs[j]>0;lp=float(norm.sf(zobs[j])) if positive else 1.;g=tail(maxima,zobs[j]) if positive else dict(k=N,N=N,p=1.,p_addone=1.,lo95=1.,hi95=1.,upper95=1.,status='exact_zero_statistic_atom')
  emp=tail(V[:,j],obs[j]) if positive else dict(k=len(V),N=len(V),p=1.,p_addone=1.,lo95=1.,hi95=1.,upper95=1.,status='exact_zero_statistic_atom')
  dg=tail(Mv,zobs[j]) if positive else dict(k=len(V),N=len(V),p=1.,p_addone=1.,lo95=1.,hi95=1.,upper95=1.,status='exact_zero_statistic_atom')
  nominal=float(norm.sf(max(0,obs[j])))
  row={**r,'local_p':lp,'local_Z':max(0.,float(norm.isf(lp))) if lp>0 else None,'nominal_local_p':nominal,'nominal_local_Z':max(0,float(obs[j])),'global_Z':max(0.,float(norm.isf(g['p_addone']))),'observed_positive_fit':bool(positive)}
  row.update({'global_'+k:v for k,v in g.items()});row.update({'direct_local_'+k:v for k,v in emp.items()});row.update({'direct_global_'+k:v for k,v in dg.items()});allrows.append(row)
 eligible=np.where(obs>0,zobs,-np.inf);j=int(np.argmax(eligible));peak=tail(maxima,zobs[j]);direct=tail(Mv,zobs[j]);rawj=int(np.argmax(obs));coarsej=np.flatnonzero(coarse_mask)[np.argmax(np.where(obs[coarse_mask]>0,zobs[coarse_mask],-np.inf))]
 commonj=np.flatnonzero(common_mask)[np.argmax(np.where(obs[common_mask]>0,zobs[common_mask],-np.inf))];common=tail(overlap,zobs[commonj]);row={**rr[j], 'scope':scope,'grid_nodes':len(m),'mass_min_MeV':m.min(),'mass_max_MeV':m.max(),'peak_local_p':float(norm.sf(zobs[j])),'peak_local_Z':zobs[j],'peak_global_Z':max(0,float(norm.isf(peak['p_addone']))),'raw_maximum':obs[rawj],'raw_maximum_mass_MeV':m[rawj], 'source_rms_offset':np.sqrt(np.mean(a*a)),'source_max_abs_offset':max(abs(a)),'minimum_field_eigenvalue':eig.min(),'negative_eigenvalue_sum':float(-eig[eig<0].sum()),'minimum_response_sd':s.min(),'maximum_response_sd':s.max(),**{'global_'+k:v for k,v in peak.items()},**{'direct_global_'+k:v for k,v in direct.items()},'common_overlap_peak_mass_MeV':m[commonj],'common_overlap_local_Z':zobs[commonj],**{'common_overlap_global_'+k:v for k,v in common.items()}}
 summary.append(row)
 for th in [2.,3.,4.]:
  c=tail(coarse,th);f=tail(maxima,th);delta=int(np.sum((maxima>=th)&(coarse<th)));ci=interval(delta,N)
  convergence.append(dict(scope=scope,threshold=th,coarse_p=c['p'],fine_p=f['p'],paired_extra_count=delta,paired_delta_p=delta/N,paired_delta_lo95=ci[0],paired_delta_hi95=ci[1]))
 validation.append(dict(scope=scope,N=len(V),nodes=len(m),mean_centered_bias=np.mean(Zv.mean(axis=0)),rms_centered_bias=np.sqrt(np.mean(Zv.mean(axis=0)**2)),max_abs_centered_mean=np.max(abs(Zv.mean(axis=0))),mean_centered_width=np.mean(Zv.std(axis=0,ddof=1)),minimum_centered_width=np.min(Zv.std(axis=0,ddof=1)),maximum_centered_width=np.max(Zv.std(axis=0,ddof=1)),correlation_RMSE=np.sqrt(np.mean((np.corrcoef(V,rowvar=False)-K)**2)),mean_Gaussian_maximum=np.mean(maxima),mean_direct_maximum=np.mean(Mv),gaussian_q95=np.quantile(maxima,.95),direct_q95=np.quantile(Mv,.95),fraction_direct_exceeds_Gaussian_q95=np.mean(Mv>=np.quantile(maxima,.95))))
 np.savez_compressed(B/f'fields/{scope}.npz',masses=m,a=a,s=s,D=D,K=K,observed_r=obs,validation=V,gaussian_maximum=maxima,gaussian_coarse_maximum=coarse,gaussian_overlap_maximum=overlap,gaussian_raw_maximum=rawmax,direct_maximum=Mv)
 print('scope',scope,'peak',m[j],zobs[j],peak['p_addone'],'globalZ',row['peak_global_Z'],'seconds',round(time.monotonic()-start,1),flush=True)
pd.DataFrame(allrows).to_csv(B/'results/significance_curves.csv',index=False,float_format='%.17g');pd.DataFrame(summary).to_csv(B/'results/summary.csv',index=False,float_format='%.17g');pd.DataFrame(validation).to_csv(B/'results/field_validation.csv',index=False,float_format='%.17g');pd.DataFrame(convergence).to_csv(B/'results/grid_convergence.csv',index=False,float_format='%.17g')
q={'complete':True,'rows':len(allrows),'scopes':len(summary),'gaussian_fields_per_scope':P['gaussian_fields'],'validation_complete_spectra_per_dataset':P['validation_Poisson_scans'],'within_scope_nested_grid_monotonicity':True,'elapsed_seconds':time.monotonic()-start,'physical_discovery_calibration':False};(B/'qa/analysis_execution.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
