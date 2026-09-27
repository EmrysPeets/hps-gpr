#!/usr/bin/env python3
"""Conditional fixed-shape rate fits, observed covariance, and exact likelihood contours."""
import os,sys,json,hashlib,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'engine'))
import experiment as X
import numpy as np,pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize_scalar,brentq
from scipy.stats import norm,chi2
start=time.time();C=X.C
base=np.concatenate([X.shape(i,92,1)[0] for i in range(3)])
bin_year=np.concatenate([np.full(len(p['n']),i,dtype=int) for i,p in enumerate(X.parts)])
models={'common':(np.zeros(3),None),'power':(X.logE,(-6.,6.)),'exponential':(-(X.E-2.3),(0.,6.))}
rows=[];contours=[];results={};checks=[];max_inner_score=0.
def checked(f):
 global max_inner_score
 max_inner_score=max(max_inner_score,float(f['score']))
 return f

def profile_at_slope(x,s):
 model=C.OneSignalProfile(X.b,X.L,base*np.exp(s*x[bin_year]))
 f=checked(model.fit(X.n))
 return checked(model.fit(X.n,0)) if f['A']<0 else f

def fixed_parameters(x,z):
 slope=float(z[1]) if len(z)>1 else 0.
 signal=base*np.exp(slope*x[bin_year])
 return checked(C.OneSignalProfile(X.b,X.L,signal).fit(X.n,float(np.exp(z[0]))))

def nll(x,z):return fixed_parameters(x,z)['nll']

def curvature(x,z,f):
 v=np.ones((len(base),len(z)))
 if len(z)>1:v[:,1]=x[bin_year]
 mu=base*np.exp(z[0]+(z[1]*x[bin_year] if len(z)>1 else 0.))
 D=mu[:,None]*v;w=X.n/f['lam']**2;r=1-X.n/f['lam']
 Hraw=(D.T*w)@D+(v.T*(r*mu))@v
 Hcross=(D.T*w)@X.L
 Hnuis=(X.L.T*w)@X.L+np.eye(X.L.shape[1])
 H=Hraw-Hcross@cho_solve(cho_factor(Hnuis,lower=True),Hcross.T)
 return .5*(H+H.T),D.T@r,Hraw,Hcross,Hnuis

def numerical_hessian(x,z,step=.001):
 k=len(z);h=np.zeros((k,k));f0=nll(x,z)
 for i in range(k):
  ei=np.eye(k)[i]*step
  h[i,i]=(nll(x,z+ei)-2*f0+nll(x,z-ei))/step**2
  for j in range(i):
   ej=np.eye(k)[j]*step
   h[i,j]=h[j,i]=(nll(x,z+ei+ej)-nll(x,z+ei-ej)-nll(x,z-ei+ej)+nll(x,z-ei-ej))/(4*step**2)
 return h

for name,(x,bounds) in models.items():
 cache={}
 def prof(s):
  key=float(s)
  if key not in cache:cache[key]=profile_at_slope(x,key)
  return cache[key]
 if bounds is None:
  slope=0.;best=prof(slope);grid=np.array([0.]);values=np.array([best['nll']]);candidates=[(slope,best['nll'])]
 else:
  grid=np.linspace(*bounds,121);values=np.array([prof(s)['nll'] for s in grid]);candidates=[(grid[0],values[0]),(grid[-1],values[-1])]
  for i in range(1,len(grid)-1):
   if values[i]<=values[i-1] and values[i]<=values[i+1]:
    opt=minimize_scalar(lambda s:prof(s)['nll'],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10})
    candidates.append((float(opt.x),float(opt.fun)))
  slope,_=min(candidates,key=lambda v:v[1]);best=prof(slope)
 assert best['A']>0
 z=np.array([np.log(best['A'])]+([] if bounds is None else [slope]))
 f=fixed_parameters(x,z);H,g,Hraw,Hcross,Hnuis=curvature(x,z,f);cov=np.linalg.inv(H)
 Hfd=numerical_hessian(x,z);Hfd2=numerical_hessian(x,z,.0005)
 err=float(np.max(np.abs(Hfd-H))/max(1.,np.max(np.abs(H))))
 err2=float(np.max(np.abs(Hfd2-H))/max(1.,np.max(np.abs(H))))
 eig=np.linalg.eigvalsh(H);gradient=float(np.max(np.abs(g)))
 assert np.min(eig)>0 and err<.001 and err2<.002 and gradient<1e-4,(name,err,err2,gradient)
 if bounds:
  assert bounds[0]<slope<bounds[1], 'Wald covariance needs an interior slope optimum'
  dense=np.linspace(*bounds,241);densebest=min(prof(s)['nll'] for s in dense)
  assert best['nll']<=densebest+1e-8
 else:densebest=best['nll']
 out={'model':name,'amplitude_unit':'epsilon2 equivalent; A is in units of 1e-8','parameter_order':['log_A_1e8']+([] if bounds is None else ['beta' if name=='power' else 'k_per_GeV']),
 'A_1e8':best['A'],'amplitude_epsilon2_at_2p3GeV':best['A']*1e-8,'slope':slope,'slope_bounds':bounds,'Q0':2*(X.nullnll-best['nll']),'sqrtQ0':np.sqrt(max(0.,2*(X.nullnll-best['nll']))),'nll':best['nll'],
 'parameter_vector':z.tolist(),'observed_profiled_hessian':H.tolist(),'observed_profiled_covariance':cov.tolist(),'hessian_includes_second_derivatives':True,
 'gradient_max_abs':gradient,'hessian_fd_relative_error_h1e3':err,'hessian_fd_relative_error_h5e4':err2,'hessian_eigenvalues':eig.tolist(),
 'slope_candidates':candidates,'coarse_slope_grid_points':len(grid),'validation_slope_grid_points':1 if bounds is None else len(dense),'validation_dense_grid_best_nll':densebest,
 'normalization_log_standard_error':float(np.sqrt(cov[0,0])), 'slope_standard_error':None if bounds is None else float(np.sqrt(cov[1,1])),
 'correlation_logA_slope':None if bounds is None else float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])),
 'per_year_fit_epsilon2':{y:float(best['A']*np.exp(slope*x[i])*1e-8) for i,y in enumerate(X.YS)}}
 for conf in (.68,.95):
  cut=float(chi2.ppf(conf,1));sigma=float(norm.ppf((1+conf)/2))
  out['amplitude_logWald_%02d'%round(conf*100)]=[float(best['A']*1e-8*np.exp(sign*sigma*np.sqrt(cov[0,0]))) for sign in (-1,1)]
  if bounds:
   interval=[];boundary=[]
   for edge in bounds:
    fn=lambda s:2*(prof(s)['nll']-best['nll'])-cut
    truncated=fn(edge)<=0
    interval.append(float(edge if truncated else brentq(fn,*sorted((edge,slope)),xtol=1e-9)))
    boundary.append(bool(truncated))
   out['slope_profile_%02d'%round(conf*100)]={'interval':interval,'bounded_by_scan':boundary,'delta_2nll':cut,'interpretation':'Conditional one-parameter likelihood interval; asymptotic chi-square threshold, not coverage-calibrated'}
 energy=np.unique(np.r_[np.linspace(min(X.E),max(X.E),241),X.E])
 for e in energy:
  xe=0. if name=='common' else (np.log(e/2.3) if name=='power' else -(e-2.3))
  v=np.array([1.]+([] if bounds is None else [xe]));mean=float(v@z);sd=float(np.sqrt(v@cov@v));point={'model':name,'energy_GeV':e,'central_epsilon2':float(np.exp(mean)*1e-8),'log_prediction_standard_error':sd}
  for conf in (.68,.95):
   crit=norm.ppf((1+conf)/2)
   point['lower_%02d'%round(conf*100)]=float(np.exp(mean-crit*sd)*1e-8);point['upper_%02d'%round(conf*100)]=float(np.exp(mean+crit*sd)*1e-8)
  rows.append(point)
 if bounds:
  loggrid=np.linspace(z[0]-max(8*np.sqrt(cov[0,0]),2.8),z[0]+4*np.sqrt(cov[0,0]),81)
  slopegrid=np.linspace(max(bounds[0],slope-4*np.sqrt(cov[1,1])),min(bounds[1],slope+4*np.sqrt(cov[1,1])),61)
  # The central best-fit row is retained separately; the exact rectangular grid is not re-centered on its discrete minimum.
  for la in loggrid:
   for s in slopegrid:
    val=nll(x,np.array([la,s]));contours.append({'model':name,'log_A_1e8':la,'amplitude_epsilon2_at_2p3GeV':np.exp(la)*1e-8,'slope':s,'nll':val,'delta_2nll':max(0.,2*(val-best['nll']))})
  out['joint_likelihood_contour_thresholds']={'68_percent_2d':float(chi2.ppf(.68,2)),'95_percent_2d':float(chi2.ppf(.95,2))}
  out['exact_contour_grid_shape']=[81,61]
  cg=pd.DataFrame([row for row in contours if row['model']==name])
  edge_checks=[]
  for column in ('log_A_1e8','slope'):
   for edge in ('min','max'):
    value=float(cg[column].agg(edge));minimum=float(cg.loc[cg[column]==value,'delta_2nll'].min())
    edge_checks.append({'coordinate':column,'edge':edge,'coordinate_value':value,'minimum_delta_2nll':minimum})
  assert all(row['minimum_delta_2nll']>chi2.ppf(.95,2) for row in edge_checks),(name,edge_checks)
  out['contour_outer_edge_checks']=edge_checks
  out['nominal_95_contour_closed_inside_grid']=True
 results[name]=out;checks.append({'model':name,'positive_profile_curvature':bool(np.min(eig)>0),'finite_difference_hessian_passed':bool(err<.001 and err2<.002),'interior_slope_or_common':True,'nll_no_worse_than_dense_grid':bool(best['nll']<=densebest+1e-8),'max_gradient':gradient})
 print('%s: eps2=%.8g, slope=%.8g, sqrtQ=%.8g, Hessian error=%.3g'%(name,best['A']*1e-8,slope,out['sqrtQ0'],err),flush=True)
parent=pd.read_csv(B/'derived/fits.csv').set_index('name');replay=[]
for name,parentname in [('common','scaled_fixed_common'),('power','scaled_fixed_energy_free')]:
 r=results[name];p=parent.loc[parentname];qerr=abs(r['Q0']-p.Q);amprel=abs(r['amplitude_epsilon2_at_2p3GeV']/p.mu_epsilon2-1)
 assert qerr<1e-7 and amprel<1e-6
 replay.append({'model':name,'parent_name':parentname,'Q0_absolute_difference':qerr,'amplitude_relative_difference':amprel})
independent=[]
for i,y in enumerate(X.YS):
 p=X.parts[i];f=checked(C.OneSignalProfile(p['b'],p['L'],X.shape(i,92,1)[0]).fit(p['n']))
 independent.append({'year':y,'beam_energy_GeV':float(X.E[i]),'epsilon2':f['A']*1e-8,'curvature_error_epsilon2':f['sigma']*1e-8,'nll':f['nll']})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={'description':'Fixed 92 MeV and fully scaled widths, identical v5.5.3 union windows; exact conditional Poisson likelihood and profiled Gaussian background constraint.',
 'formulas':{'common':'epsilon2(E)=a','power':'epsilon2(E)=a*(E/2.3 GeV)^beta','exponential':'epsilon2(E)=a*exp[-k*(E-2.3 GeV)]'},
 'uncertainty_interpretation':'Pointwise approximate 68%/95% log-Wald conditional shape bands from the full observed Hessian profiled over background nuisance parameters; not simultaneous bands, model-validity bands, discovery calibration, or guaranteed coverage. Exact parameter contours are separate and use asymptotic 2-parameter thresholds.',
 'nonlinear_hessian_formula':'Hpp=D.T W D + sum_i (1-n_i/lambda_i) d2lambda_i/dpdp; Hprofile=Hpp-Hp_theta Htheta_theta^-1 Htheta_p',
 'script_sha256':sha(Path(__file__)),'experiment_sha256':sha(B/'engine/experiment.py'),'parent_fits_sha256':sha(B/'derived/fits.csv'),
 'models':results,'independent_dataset_estimates':independent}
(B/'derived/rate_band_fits.json').write_text(json.dumps(meta,indent=2)+'\n')
pd.DataFrame(rows).to_csv(B/'derived/rate_bands.csv',index=False,float_format='%.17g')
pd.DataFrame(contours).to_csv(B/'derived/rate_parameter_contours.csv',index=False,float_format='%.17g')
validation={'passed':True,'model_checks':checks,'parent_replays':replay,'max_inner_score':max_inner_score,'contour_outer_edge_checks':{name:r['contour_outer_edge_checks'] for name,r in results.items() if name!='common'},'all_nominal_95_contours_closed_inside_grid':all(r.get('nominal_95_contour_closed_inside_grid',True) for r in results.values()),'contour_rows':len(contours),'band_rows':len(rows),'elapsed_seconds':time.time()-start,'script_sha256':sha(Path(__file__))}
assert len(contours)==9882 and max_inner_score<3e-5
(B/'qa/rate_band_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation),flush=True)
