from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.stats import norm
B=Path(__file__).resolve().parents[1];checks=[]
def ck(name,v):
 assert v,name
 checks.append(name)
d=pd.read_csv(B/'results/significance_and_reach.csv',dtype={'scope':str});p=pd.read_csv(B/'results/peaks.csv',dtype={'scope':str});f=d[(d.domain=='full')&d.scope.isin(['2015','2016','2021','combined'])]
ck('all scope/mass rows',len(d)==5096 and len(f)==2628 and len(p)==36)
ck('complete checkpoint pairs',len(list((B/'checkpoints').glob('*.json')))==928 and len(list((B/'checkpoints').glob('*.npz')))==928)
ck('all CLs roots',max(abs(f.cls-.1).max(),abs(f.asimov_cls-.1).max())<2e-6)
ck('profile gradients',f.limit_max_score.max()<2.001e-7)
ck('positive Poisson means',f.min_lambda.min()>0)
ck('positive upper endpoints',f.epsilon2_90.min()>0 and f.epsilon2_asimov90.min()>0)
ck('batch/scalar agreement',f.scalar_error.max()<2e-5)
ck('all probabilities bounded',all(d[k].between(0,1).all() for k in ['local_p','global_p','global_lo95','global_hi95']))
ck('local/global event containment within MC uncertainty',bool((d.global_hi95>=d.local_p-1e-12).all()))
replay=[]
for w in [2.25,2.4,2.5,2.6]:
 for scope in ['2015','2016','2021','combined']:
  z=np.load(B/f'fields/w{w:.2f}_{scope}.npz');R=z['K'];C=z['D'].T@z['D']/np.outer(z['s'],z['s']);ck(f'{w}/{scope} response covariance',np.max(abs(C-R))<1e-10);ck(f'{w}/{scope} PSD',np.linalg.eigvalsh(R).min()>-1e-9);ck(f'{w}/{scope} complete scans',z['validation'].shape==(256,len(z['masses'])));ck(f'{w}/{scope} overlap subset',np.all(z['gaussian_maximum']>=z['gaussian_overlap_maximum']))
  if w==2.25:
   old=np.load(B/f'inputs/baseline_fields/{scope}.npz');ii=np.flatnonzero(np.isclose(old['masses'],np.round(old['masses'])));delta=float(np.max(abs(old['observed_r'][ii]-z['observed_r'])));replay.append(delta);ck(f'{scope} baseline replay',delta<2e-5);ck(f'{scope} exact reused response',np.array_equal(z['D'],old['D'][:,ii]))
 for scope in ['combined','fisher','stouffer']:
  a=p[(p.width_sigma==w)&(p.scope==scope)&(p.domain=='full')].iloc[0];b=p[(p.width_sigma==w)&(p.scope==scope)&(p.domain=='overlap')].iloc[0];ck(f'{w}/{scope} common overlap peak subset',a.local_Z>=b.local_Z-1e-9)
 # In single-experiment regions Fisher exactly reproduces that local test.
 for y,lo,hi in [('2015',19,38),('2021',181,250)]:
  yy=d[(d.width_sigma==w)&(d.scope==y)&(d.domain=='full')&d.mass_MeV.between(lo,hi)].sort_values('mass_MeV');ff=d[(d.width_sigma==w)&(d.scope=='fisher')&(d.domain=='full')&d.mass_MeV.between(lo,hi)].sort_values('mass_MeV');ck(f'{w}/{y} Fisher single identity',np.max(abs(yy.local_p.to_numpy()-ff.local_p.to_numpy()))<1e-13)
for r in json.loads((B/'inputs/provenance.json').read_text()):ck('source hash '+r['copy'],hashlib.sha256((B/r['copy']).read_bytes()).hexdigest()==r['sha256'])
for name in ['width_derivative_validation','fisher_validation']:ck(name,json.loads((B/f'qa/{name}.json').read_text())['passed'])
ident=pd.read_csv(B/'qa/peak_composition_identity.csv');ck('coupling-equality deviance identity',ident.identity_error.max()<2e-5)
gv=pd.read_csv(B/'results/global_validation.csv');outliers=gv[~gv.Gaussian_peak_p.between(gv.direct_peak_lo95,gv.direct_peak_hi95)]
q={'passed':True,'checks':len(checks),'check_names':checks,'baseline_max_root_replay_difference':max(replay),'max_limit_score':float(f.limit_max_score.max()),'max_batch_scalar_error':float(f.scalar_error.max()),'Gaussian_peak_p_outside_direct_95_intervals':outliers.to_dict('records'),'zero_Gaussian_peak_counts':int((p.global_k==0).sum()),'conditional_on_fixed_GP_sources':True,'source_or_method_selection_calibrated':False}
(B/'qa/numerical_validation.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:v for k,v in q.items() if k!='check_names'},indent=2))
