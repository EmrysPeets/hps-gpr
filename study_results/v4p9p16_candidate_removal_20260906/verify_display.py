#!/usr/bin/env python3
"""Independently verify display aggregation as a partition matrix on native bins."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];T=HERE/'traditional'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def check(v,name):
    global checks
    checks+=1;assert v,name
summary=json.loads((T/'derived/summary.json').read_text())
for path,digest in summary['output_sha256'].items():check(sha(ROOT/path)==digest,path)
groups=np.load(T/'qa/paper_display_groups.npz');rows=pd.read_csv(T/'derived/fit_summary.csv')
record=[];expected={'2015_m051':3,'2015_m021':1,'2016_m090':5,'2016_m117':5,'2021_m078':1,'2021_m065':1}
for row in rows[rows.variant.eq('baseline')].itertuples():
    a=np.load(T/'derived/points'/f'{row.fit_id}__baseline.npz');prefix=row.fit_id+'__'
    starts=groups[prefix+'native_start'];stops=groups[prefix+'native_stop_exclusive'];native=a['native_indices']
    R=((native[None,:]>=starts[:,None])&(native[None,:]<stops[:,None])).astype(float)
    check(np.all(R.sum(0)==1),'Complete, nonoverlapping partition')
    check(np.all(R.sum(1)>0),'No empty display bin')
    check(np.all(R.sum(1)[:-1]==expected[row.fit_id]),'Declared whole-bin grouping')
    check(0<R.sum(1)[-1]<=expected[row.fit_id],'Retained last partial group')
    edges=groups[prefix+'edges_MeV'];width=groups[prefix+'width_MeV']
    check(np.array_equal(edges,np.r_[a['edges_MeV'][starts-native[0]],a['edges_MeV'][-1]]),'Physical edges')
    check(np.array_equal(width,np.diff(edges)),'Actual group widths')
    worst=0.
    for field in ['counts','background_free','background_null','total_free','total_null','signal_bin_probability','signal_counts']:
        original=row.amplitude_full*a['signal_bin_probability'] if field=='signal_counts' else a[field]
        summed=R@original;stored=groups[prefix+field]
        check(np.allclose(summed,stored,rtol=1e-14,atol=1e-10),'Exact component grouping '+field)
        check(np.allclose(summed/width,groups[prefix+field+'_per_MeV'],rtol=1e-14,atol=1e-10),'Correct density '+field)
        worst=max(worst,float(np.max(abs(summed-stored))))
        if field=='counts':
            check(np.array_equal(summed,stored) and summed.sum()==original.sum(),'Integer count conservation')
            check(np.allclose(np.sqrt(summed)/width,groups[prefix+'count_error_per_MeV'],rtol=1e-14,atol=0),'Counting errors')
    check(float(groups[prefix+'amplitude_full'])==row.amplitude_full,'Unchanged fitted amplitude')
    record.append(dict(fit_id=row.fit_id,native_bins=len(native),display_bins=len(starts),max_component_sum_error=worst))
out=dict(passed=True,checks=checks,original_fit_products_unchanged=True,fits=record,grouped_sha256=sha(T/'qa/paper_display_groups.npz'),scope='Display grouping conserves whole-bin counts and model sums; density normalization uses actual widths. No refit or change of statistical binning.')
(HERE/'qa/display_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
