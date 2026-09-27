from engine import *
import pandas as pd,hashlib,platform,scipy
from scipy.stats import norm,beta
from pypdf import PdfReader
rows=pd.read_csv(B/'results/significance_curves.csv',dtype={'scope':str});summ=pd.read_csv(B/'results/summary.csv',dtype={'scope':str});checks=[]
def check(name,ok,detail=None):
 checks.append(dict(check=name,passed=bool(ok),detail=detail));assert ok,(name,detail)
check('1310 unique complete coordinates',len(rows)==1310 and not rows.duplicated(['scope','mass_MeV']).any())
check('All strict scalar and batch fit checks passed',(rows[['scalar_error','max_score','observed_score','asimov_score']].max().values<np.array([2e-5,2e-7,2e-7,2e-7])).all())
check('Conditional local p and Z consistent',np.allclose(rows[rows.observed_positive_fit].local_p,norm.sf(rows[rows.observed_positive_fit].z),rtol=1e-12))
check('Nonpositive observed fits have exact local/global atom',((rows[~rows.observed_positive_fit][['local_p','global_p_addone','direct_local_p','direct_global_p']])==1).all().all())
check('Global Gaussian curve is not below local curve',(rows.global_p_addone+1e-5>=rows.local_p).all())
check('Exact direct scan tails include local tails',(rows.direct_global_k>=rows.direct_local_k).all())
counts={'2015':163,'2016':283,'2021':401,'combined':463}
for scope in counts:
 x=np.load(B/f'fields/{scope}.npz');q=rows[rows.scope==scope];K=x['D'].T@x['D']/np.outer(x['s'],x['s']);check(scope+' grid and 256 coherent complete scans',len(x['masses'])==counts[scope] and x['validation'].shape==(256,counts[scope]))
 check(scope+' response covariance and unit diagonal',np.allclose(K,x['K']) and np.allclose(np.diag(K),1))
 check(scope+' positive semidefinite correlation',np.linalg.eigvalsh(x['K']).min()>-1e-9)
 check(scope+' exact nested-grid maximum ordering',np.all(x['gaussian_maximum']>=x['gaussian_coarse_maximum']))
 check(scope+' finite saved roots',np.isfinite(x['validation']).all() and np.isfinite(x['D']).all())
 for j,r in enumerate(q.itertuples()):
  if r.observed_positive_fit:
   assert int(np.sum(x['gaussian_maximum']>=r.z))==r.global_k
   assert int(np.sum(x['validation'][:,j]>=r.observed_r))==r.direct_local_k
  if r.global_k>0 and r.global_k<r.global_N:
   assert np.isclose(r.global_lo95,beta.ppf(.025,r.global_k,r.global_N-r.global_k+1))
 check(scope+' maxima counts and intervals replay',True)
# The copied likelihood gives the same observed roots as the previous frozen ledger.
old=Path(B.parents[1]/'study_results/v5p8p0_local_significance_mapping_20260917/inputs/observed_reference.csv')
replay=[]
if old.exists():
 ref=pd.read_csv(old);mapping={'2015':'individual_2015_full','2016':'individual_2016_full','2021':'individual_2021_10pct'}
 for scope,key in mapping.items():
  q=rows[rows.scope==scope].merge(ref[ref.scope==key][['mass_MeV','signed_r']],on='mass_MeV');replay.append(dict(scope=scope,coordinates=len(q),maximum_root_difference=float(abs(q.observed_r-q.signed_r).max())))
 q=rows[rows.scope=='combined'].copy();mapped=[]
 for r in q.itertuples():
  key={ '2015':'individual_2015_full','2016':'individual_2016_full','2021':'individual_2021_10pct','2015+2016':'pair_2015_2016','2016+2021':'pair_2016_2021','2015+2016+2021':'all_2015_2016_2021' }[r.datasets]
  rr=ref[(ref.scope==key)&(ref.mass_MeV==r.mass_MeV)]
  if len(rr):mapped.append(abs(r.observed_r-rr.iloc[0].signed_r))
 replay.append(dict(scope='combined',coordinates=len(mapped),maximum_root_difference=float(max(mapped))))
 check('Frozen observed root replay',max(r['maximum_root_difference'] for r in replay)<2e-5,replay)
for name in ['response_derivative_validation.json','full_response_validation.json']:check(name,json.loads((B/'qa'/name).read_text())['passed'])
check('Peak Gaussian tails inside corresponding direct95 intervals',((summ.global_p_addone>=summ.direct_global_lo95)&(summ.global_p_addone<=summ.direct_global_hi95)).all())
q=rows[(rows.scope=='combined')&((rows.mass_MeV<39)|(rows.mass_MeV>180))]
for r in q.itertuples():
 z=rows[(rows.scope==r.datasets)&(rows.mass_MeV==r.mass_MeV)].iloc[0];assert abs(r.observed_r-z.observed_r)<1e-12 and abs(r.a-z.a)<1e-12
check('Single-dataset combined segments preserve individual fits',True)
result={'artifact_arithmetic_passed':True,'checks':checks,'strict_fit_failures':0,'rows':len(rows),'maximum_scalar_error':float(rows.scalar_error.max()),'maximum_optimizer_score':float(rows[['max_score','observed_score','asimov_score']].max().max()),'physical_discovery_calibrated':False,'runtime':{'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__,'platform':platform.platform()}}
(B/'qa/numerical_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
