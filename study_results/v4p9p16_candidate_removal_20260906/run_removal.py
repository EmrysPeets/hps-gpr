#!/usr/bin/env python3
"""Conditional candidate excision with the exact current dense profile solver."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'v4p9p16_combined_global_20260906'))
import run_combined as parent
core,c,np,pd=parent.core,parent.c,parent.np,parent.pd
from scipy.special import ndtr
from scipy.stats import norm
from scipy.optimize import minimize
CANDIDATES={'2015':[51,21],'2016':[90,117],'2021':[78,65]}
SEED=4916160906

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def seed(*tags):return np.random.default_rng(np.random.SeedSequence(np.frombuffer(hashlib.sha256('|'.join(map(str,(SEED,*tags))).encode()).digest()[:16],dtype='<u4')))
def setup():
 sources=json.loads((parent.FOLDER/'contract.json').read_text())['source_sha256']
 for name,digest in sources.items():assert sha(ROOT/name)==digest,name
 cfg=c.production.load_config(c.production.DEFAULT_CARD);c.production.validate_card(cfg);c.production.validate_histogram_inputs(cfg)
 datasets=c.production.make_datasets(cfg);states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
 return cfg,datasets,states,sources

def exp_poly_predict(x,e,n,train,degree):
 """Poisson polynomial sideband MLE, with Gaussian quadrature over whole bins."""
 lo,hi=e[0],e[-1];q,w=np.polynomial.legendre.leggauss(8);dx=np.diff(e)
 xq=x[:,None]+dx[:,None]*q/2;weights=dx[:,None]*w/2
 X=np.polynomial.chebyshev.chebvander(2*(xq-lo)/(hi-lo)-1,degree)
 X0=np.polynomial.chebyshev.chebvander(2*(x-lo)/(hi-lo)-1,degree)
 assert train.sum()>degree+2
 sc=np.sqrt(n[train].sum());start=np.linalg.lstsq(X0[train]*np.sqrt(n[train])[:,None],np.log(n[train]/dx[train])*np.sqrt(n[train]),rcond=None)[0]
 def calc(z):
  coeff=start+z/sc;eta=X@coeff
  if abs(eta).max()>150:return np.inf,np.full(len(z),1e90),np.eye(len(z)),None
  bq=np.exp(eta)*weights;b=bq.sum(1);J=np.einsum('ij,ijk->ik',bq,X)/sc
  V=np.einsum('ij,ijk,ijl->ikl',bq,X,X)/sc**2
  rr=1-n/b;value=c.deviance_half(n[train],b[train]);g=J[train].T@rr[train]
  H=(J[train].T*(n[train]/b[train]**2))@J[train]+np.einsum('i,ijk->jk',rr[train],V[train])
  return value,g,H,b
 z=np.zeros(degree+1)
 for it in range(100):
  value,g,H,b=calc(z)
  if np.max(abs(g))<2e-7:break
  ev=np.linalg.eigvalsh(H).min();H=H+np.eye(len(z))*max(0,1e-8-ev);step=np.linalg.solve(H,-g);rate=1.
  for ls in range(60):
   v=calc(z+rate*step)[0]
   if np.isfinite(v) and v<=value+1e-4*rate*(g@step)+1e-12:z+=rate*step;break
   rate*=.5
  else:raise RuntimeError('Sideband polynomial line search')
 else:raise RuntimeError('Sideband polynomial convergence')
 value,g,H,b=calc(z);return b,dict(degree=degree,gradient=float(abs(g).max()),half_deviance=float(value),dof=int(train.sum()-degree-1),coeff=(start+z/sc).tolist(),iterations=it)

def fill(ctx,source,year,width=2.25,nrep=0):
 p=ctx.parts[0]['p'];x=p.x_full;edges=p.edges_full;masses=CANDIDATES[year]
 holes=[abs(x-m/1000)<=width*float(ctx.datasets_year.sigma(m/1000)) for m in masses]
 union=np.any(holes,axis=0);keep=~union
 predictor=core.CachedCholeskyPredictor(x[keep],x[union],ctx.parts[0]['kernel'],ctx.cfg)
 mu,V=predictor.latent(source[keep]);V=(V+V.T)/2;ev,U=np.linalg.eigh(V)
 assert ev.min()>-1e-10 and ev.max()>0
 V=U@np.diag(np.maximum(ev,0))@U.T;factor=U*np.sqrt(np.maximum(ev,0))
 mean=np.exp(mu+.5*np.diag(V));values=np.broadcast_to(source,(1+nrep,len(source))).copy();values[0,union]=mean
 rng=seed('joint-latent-poisson',year,width)
 for rep in range(nrep):values[rep+1,union]=rng.poisson(np.exp(mu+factor@rng.normal(size=len(mu))))
 lanes={}
 for j,name in enumerate(['first','second','both']):
  mask=holes[j] if j<2 else union
  for i,v in enumerate(values):
   modified=source.copy();modified[mask]=v[mask]
   label=name+'_mean' if i==0 else f'{name}_rep{i-1:02d}'
   lanes[label]=modified;assert np.array_equal(modified[~mask],source[~mask])
 info=dict(width_sigma=width,hole_bins=[int(h.sum()) for h in holes],latent_min_eigenvalue=float(ev.min()),latent_max_eigenvalue=float(ev.max()),source_min_count=float(source.min()),conditional_replicas=nrep,latent_covariance_observation_noise_added=False)
 arrays=dict(hole_first=holes[0],hole_second=holes[1],hole_union=union,latent_mu=mu,latent_cov=V,latent_factor=factor,fill_mean=mean)
 return lanes,arrays,info

def polynomial_fill(ctx,source,year,holes):
 p=ctx.parts[0]['p'];x=p.x_full;e=p.edges_full;modified=source.copy();infos=[]
 union=holes['hole_union']
 for k,mass in enumerate(CANDIDATES[year]):
  sig=float(ctx.datasets_year.sigma(mass/1000));local=abs(x-mass/1000)<=7*sig;ix=np.flatnonzero(local);assert np.all(np.diff(ix)==1)
  keep=~union[ix];degree=5 if year=='2015' and mass<39 else 3
  mean,info=exp_poly_predict(x[ix]*1000,e[np.r_[ix,ix[-1]+1]]*1000,source[ix],keep,degree)
  mask=holes[['hole_first','hole_second'][k]];target=np.flatnonzero(mask);assert np.all(np.isin(target,ix))
  modified[target]=mean[np.searchsorted(ix,target)]
  info.update(mass_MeV=mass,fit_low_MeV=float(e[ix[0]]*1000),fit_high_MeV=float(e[ix[-1]+1]*1000),n_sideband_bins=int(keep.sum()))
  infos.append(info)
 assert np.array_equal(source[~union],modified[~union])
 return modified,infos

def build_inputs(year):
 cfg,datasets,states,sources=setup();sc=next(s for s in c.production.SCOPES if s[2]==(year,));ctx=core.Context(sc,CANDIDATES[year][0],cfg,datasets,states);ctx.datasets_year=datasets[year]
 p=ctx.parts[0]['p'];sources.update({str(Path(__file__).relative_to(ROOT)):sha(__file__),str((HERE/'PROTOCOL.md').relative_to(ROOT)):sha(HERE/'PROTOCOL.md')})
 out=HERE/'derived'/year;out.mkdir(parents=True,exist_ok=True);spectra={};metadata={};arrays={'x_GeV':p.x_full,'edges_GeV':p.edges_full,'observed':p.y_full,'reference':ctx.truths['stress']}
 for source_name,source,nrep in [('observed',p.y_full,10),('reference',ctx.truths['stress'],0)]:
  base,holes,info=fill(ctx,source,year,nrep=nrep);wide,_,wideinfo=fill(ctx,source,year,width=3.,nrep=0)
  poly,polyinfo=polynomial_fill(ctx,source,year,holes)
  spectra[source_name+'_original']=source
  spectra.update({source_name+'_'+k:v for k,v in base.items()})
  spectra[source_name+'_both_poly_mean']=poly;spectra[source_name+'_both_wide_mean']=wide['both_mean']
  metadata[source_name]=dict(primary=info,wide=wideinfo,polynomial=polyinfo)
  arrays.update({source_name+'__'+k:v for k,v in holes.items()})
 names=list(spectra);stack=np.array([spectra[k] for k in names]);assert np.all(stack>0)
 np.savez_compressed(out/'inputs.npz',**arrays,spectra=stack,lane_names=np.array(names))
 rows=[]
 for mass,which in zip(CANDIDATES[year],['first','second']):
  mask=arrays['observed__hole_'+which];idx=np.flatnonzero(mask);sig=float(datasets[year].sigma(mass/1000))*1000
  rows.append(dict(dataset=year,mass_MeV=mass,rank=CANDIDATES[year].index(mass)+1,sigma_MeV=sig,low_MeV=p.edges_full[idx[0]]*1000,high_MeV=p.edges_full[idx[-1]+1]*1000,n_bins=int(mask.sum()),observed_original=float(p.y_full[mask].sum()),observed_gp_fill=float(spectra['observed_both_mean'][mask].sum()),observed_poly_fill=float(spectra['observed_both_poly_mean'][mask].sum()),reference_original=float(ctx.truths['stress'][mask].sum()),reference_gp_fill=float(spectra['reference_both_mean'][mask].sum())))
 pd.DataFrame(rows).to_csv(out/'holes.csv',index=False)
 dump(out/'input_contract.json',dict(dataset=year,scope=sc[0],masses_MeV=list(map(int,c.production.EXPECTED_DATASET_GRIDS[year])),candidates_MeV=CANDIDATES[year],lane_names=names,lanes=len(names),inference_bins=len(p.x_full),bin_width_MeV=float(np.diff(p.edges_full)[0]*1000),filler_kernel=repr(ctx.parts[0]['kernel']),source_sha256=sources,inputs_sha256=sha(out/'inputs.npz'),replacement_details=metadata))
 print('Built',year,len(names),'lanes',flush=True)

def scan(year,pilot=False):
 out=HERE/'derived'/year;data=np.load(out/'inputs.npz');contract=json.loads((out/'input_contract.json').read_text());assert sha(out/'inputs.npz')==contract['inputs_sha256']
 cfg,datasets,states,sources=setup();sc=next(s for s in c.production.SCOPES if s[2]==(year,));names=data['lane_names'];spectra=data['spectra'];union=data['observed__hole_union']
 rows=[];arrays={};original=pd.read_csv(parent.PARENT/'summary/observed_calibrated_limits.csv');original=original[original.scope_key==sc[0]].set_index('mass_MeV')
 # Archived global-response fields use the same deterministic source but may have a gated numerical accelerator.
 vectors=np.load(parent.source_folder(year)/'asimov/scan_vectors.npz') if (parent.source_folder(year)/'asimov/scan_vectors.npz').exists() else None
 masses=contract['masses_MeV']
 if pilot:masses=sorted(set(CANDIDATES[year]+[int(masses[0]),int(masses[-1]),{'2015':35,'2016':102,'2021':71}[year]]))
 began=time.monotonic()
 checkpoint=out/('pilot_scans.csv' if pilot else 'scans.csv')
 for j,mass in enumerate(masses):
  ctx=core.Context(sc,mass,cfg,datasets,states);p=ctx.parts[0]['p'];assert np.array_equal(p.x_full,data['x_GeV']);assert np.array_equal(p.y_full,data['observed']);assert np.array_equal(ctx.truths['stress'],data['reference'])
  obs_ref=float(original.loc[mass,'signed_r_profiled_asymptotic']);assert abs(ctx.ofit['profiled']['signed_r']-obs_ref)<2e-5
  remote=not np.any(ctx.mask&union)
  for lane,whole in zip(names,spectra):
   # Pilot includes every deterministic variant and the first Poisson replicate only.
   if pilot and '_rep' in lane and not lane.endswith('rep00'):continue
   b,L=ctx.retrain(whole);model=c.Profile(b,L,ctx.w,'linear');ff=model.fit(whole[ctx.mask]);nn=model.fit(whole[ctx.mask],0.)
   q=2*(nn['nll']-ff['nll']);assert q>=-1e-7;root=float(np.sign(ff['A'])*np.sqrt(max(q,0)))
   assert model.max_score<1e-5
   if lane=='observed_original':assert abs(root-obs_ref)<2e-5
   reference_error=0.
   if lane=='reference_original' and vectors is not None:
    ii=int(np.flatnonzero(vectors['masses_MeV']==mass)[0]);reference_error=root-float(vectors['profiled'][0,ii])
    assert abs(reference_error)<2e-3,('Archived reference discrepancy',year,mass,reference_error)
   rows.append(dict(dataset=year,mass_MeV=mass,lane=str(lane),reference_root_delta=reference_error,r=root,Ahat_window=ff['A'],sigma_A_window=ff['sigma'],remote=remote,half_deviance_fit=ff['nll'],half_deviance_null=nn['nll'],max_score=model.max_score,min_lambda=min(ff['min_lambda'],nn['min_lambda'])))
   if '_rep' not in lane or mass in CANDIDATES[year]:
    pref=f'm{mass:03d}__{lane}__'
    arrays.update({pref+k:v for k,v in dict(counts=whole[ctx.mask],gp_mean=b,L=L,w=ctx.w,free_theta=ff['z'][1:],null_theta=nn['z'],free_lambda=ff['lam'],null_lambda=nn['lam']).items()})
  pd.DataFrame(rows).to_csv(checkpoint,index=False)
  if j%10==0 or j==len(masses)-1:print(year,'pilot' if pilot else 'scan',j+1,'/',len(masses),'seconds',round(time.monotonic()-began,1),flush=True)
 np.savez_compressed(out/('pilot_components.npz' if pilot else 'components.npz'),**arrays)
 dump(out/('pilot_qa.json' if pilot else 'scan_qa.json'),dict(passed=True,dataset=year,masses=len(masses),profile_tests=len(rows),max_score=float(pd.DataFrame(rows).max_score.max()),elapsed_seconds=time.monotonic()-began,inputs_sha256=sha(out/'inputs.npz'),csv_sha256=sha(checkpoint)))

def metrics():
 summaries=[];global_checks=[];holes=[]
 for year in CANDIDATES:
  out=HERE/'derived'/year;d=pd.read_csv(out/'scans.csv');holes.append(pd.read_csv(out/'holes.csv'));wide=d.pivot(index='mass_MeV',columns='lane',values='r');remote=d[d.lane=='observed_original'].set_index('mass_MeV').remote.to_numpy();m=wide.index.to_numpy()
  for source in ['observed','reference']:
   base=wide[source+'_original'].to_numpy()
   for selection,mask in [('full',np.ones(len(base),bool)),('remote',remote)]:
    for lane in [n for n in wide.columns if n.startswith(source+'_')]:
     y=wide[lane].to_numpy();v=y[mask];b=base[mask]
     crossing=int(np.sum((y[1:]*y[:-1]<0)&mask[1:]&mask[:-1]))
     std=float(np.std(v));ratio=std/np.std(b)
     summaries.append(dict(dataset=year,source=source,selection=selection,lane=lane,n_points=int(mask.sum()),std=std,rms=float(np.sqrt(np.mean(v*v))),peak_to_peak=float(np.ptp(v)),retained_std=float(ratio),correlation=float(np.corrcoef(v,b)[0,1]),max_abs_change=float(np.max(abs(v-b))),rms_change=float(np.sqrt(np.mean((v-b)**2))),sign_transitions=crossing,substantial_persistence=bool(ratio>=.5 and crossing>=2)))
 summary=pd.DataFrame(summaries);summary.to_csv(HERE/'derived/oscillation_metrics.csv',index=False);pd.concat(holes).to_csv(HERE/'derived/holes.csv',index=False)
 selected=summary[(summary.selection=='remote')&summary.lane.isin(['observed_both_mean','reference_both_mean'])]
 trigger=bool(selected.substantial_persistence.any());assert len(selected)==6
 dump(HERE/'derived/persistence_trigger.json',dict(traditional_fits_triggered=trigger,criterion='any dataset/source retains>=50percent remote std and>=2 remote contiguous sign transitions',rows=selected.to_dict('records'),interpretation='descriptive study routing; not a calibrated statistical test'))
 print(selected[['dataset','source','retained_std','sign_transitions','substantial_persistence']].to_string(index=False));print('TRADITIONAL FOLLOW-UP',trigger)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inputs','pilot','scan','metrics']);ap.add_argument('--dataset',choices=list(CANDIDATES));args=ap.parse_args()
 if args.stage=='metrics':metrics();return
 assert args.dataset
 {'inputs':build_inputs,'pilot':lambda y:scan(y,True),'scan':scan}[args.stage](args.dataset)
if __name__=='__main__':main()
