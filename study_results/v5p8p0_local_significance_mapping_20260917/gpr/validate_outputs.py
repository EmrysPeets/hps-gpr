from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent
checks={}
def check(name,value):
 checks[name]=bool(value)
 if not value:raise AssertionError(name)
manifest=json.loads((B/'input_manifest.json').read_text())
check('all_input_hashes_match',all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==q['sha256'] for p,q in manifest.items()))
x=np.load(B/'field_2016/field.npz');m=x['masses'];expected=np.unique(np.r_[np.arange(39,180.01,.5),np.arange(74,79.01,.25)])
check('complete_half_grid_plus_declared_quarters',np.array_equal(m,expected))
check('old_coordinate_invariance',json.loads((B/'same_node_invariance.json').read_text())['passed'])
check('rank_one_all_roots_reference_validation',json.loads((B/'rank_one_validation.json').read_text())['passed'])
for year in ['2015','2016','2021']:
 x=np.load(B/f'field_{year}/field.npz')
 check(year+'_finite_arrays',all(np.isfinite(x[k]).all() for k in x.files))
 check(year+'_256_coherent_validation_rows',x['validation'].shape[0]==256)
 check(year+'_positive_widths',(x['s']>0).all())
 check(year+'_PSD_response_kernel',np.linalg.eigvalsh(x['K']).min()>-1e-8)
d=pd.read_csv(B/'grid_tail_summary.csv')
check('nested_gaussian_tails_do_not_decrease',all(q.gaussian_p>=d[(d.region==q.region)&(d.step_MeV==1)&(d.threshold==q.threshold)].iloc[0].gaussian_p for q in d.itertuples()))
check('conditional_tails_in_unit_interval',d[['gaussian_p','direct_p']].ge(0).all().all() and d[['gaussian_p','direct_p']].le(1).all().all())
(B/'validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
print(json.dumps(checks,indent=2))
