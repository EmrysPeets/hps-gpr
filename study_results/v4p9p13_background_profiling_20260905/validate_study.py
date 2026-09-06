#!/usr/bin/env python3
"""Check saved numerical artifacts; scientific closure failures remain results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main():
    checks=[]
    def check(name, condition):
        passed=bool(condition)
        checks.append(dict(check=name,passed=passed))
        if not passed:
            raise AssertionError(name)
    p=HERE/'injections/derived'
    s=json.loads((p/'summary.json').read_text())
    t=pd.read_csv(p/'toy_results.csv.gz')
    g=pd.read_csv(p/'extraction_summary.csv')
    c=pd.read_csv(p/'local_calibration_holdout.csv')
    coord=['ensemble','mass_MeV','strength_sigma','toy_id']
    group=['ensemble','mass_MeV','strength_sigma','method']
    check('27000 spectra, 54000 method fits, 108 groups',len(t)==54000 and len(g)==108)
    check('unique method/coordinate rows',not t.duplicated(coord+['method']).any())
    check('two paired methods per spectrum',t.groupby(coord).size().eq(2).all())
    check('same injected yield in paired methods',t.groupby(coord).Atrue.nunique().eq(1).all())
    check('500 toys in every coordinate',t.groupby(group).size().eq(500).all())
    check('all toy IDs 0 to 499 retained',all(set(f.toy_id)==set(range(500)) for _,f in t.groupby(group)))
    check('no negative background redraws',s['negative_vector_redraws']==0)
    check('finite fits, errors, roots, scores',np.isfinite(t[['Ahat','sigma_A','pull','signed_r','min_lambda','max_score']]).all().all())
    check('positive fitted expectations and errors',(t.min_lambda>0).all() and (t.sigma_A>0).all())
    check('scaled gradient convergence',(t.max_score<2e-7).all())
    check('pull formula',np.allclose(t.pull,(t.Ahat-t.Atrue)/t.sigma_A,rtol=1e-12,atol=1e-12))
    check('one-sided local p0 mapping',np.allclose(t.p0_asymptotic,norm.sf(np.maximum(t.signed_r,0)),rtol=2e-12,atol=1e-15))
    positive=t.Atrue>0
    check('CLs truth exclusion classification',np.array_equal(t.loc[positive,'true_yield_excluded'],t.loc[positive,'cls_at_true']<.1))
    check('zero-yield physical-bound convention',(~t.loc[~positive,'true_yield_excluded']).all())
    shortcut=t[t.shortcut_delta_ul.notna()]
    check('216 independent full UL classifications',len(shortcut)==216 and np.array_equal(shortcut.true_yield_excluded,shortcut.shortcut_delta_ul<0))
    ref=pd.DataFrame(json.loads((p/'frozen_injection_strengths.json').read_text())).set_index('mass_MeV')
    check('same reference profiled error defines physical strengths',np.allclose(t.Atrue,t.mass_MeV.map(ref.sigma_profiled)*t.strength_sigma,rtol=1e-13,atol=1e-10))
    fresh=t.groupby(group).agg(n=('Ahat','size'),pull_mean=('pull','mean'),pull_std=('pull','std'),exclusion_count=('true_yield_excluded','sum')).sort_index()
    saved=g.set_index(group).sort_index()
    check('summary row means widths and counts close',np.allclose(fresh,saved[fresh.columns],rtol=2e-12,atol=1e-12))
    for row in c.itertuples():
        b=t[(t.ensemble==row.ensemble)&(t.mass_MeV==row.mass_MeV)&(t.strength_sigma==0)&(t.method=='fixed')].sort_values('toy_id')
        train=b[b.toy_id<100].signed_r.to_numpy()
        held=b[b.toy_id>=100].signed_r.to_numpy()
        if row.correction=='variance_scaled':held=held/ref.loc[row.mass_MeV,'kappa']
        elif row.correction=='split_calibrated':held=(held-train.mean())/train.std(ddof=1)
        check(f'held-out count {row.ensemble}/{row.mass_MeV}/{row.correction}',row.test_n==400 and row.false_positive_count==np.sum(held>norm.isf(.05)))
    check('injection sources unchanged',all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in s['sources'].items()))
    old=json.loads((REPO/'study_results/background_profile_comparison_20260905/derived/summary.json').read_text())
    check('original comparison sources unchanged',all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in old['sources'].items()))
    observed=json.loads((HERE/'observed/derived/validation.json').read_text())
    check('30 observed checks passed',observed['status']=='passed' and observed['passed_checks']==30 and all(x['passed'] for x in observed['checks']))
    for ledger in [HERE/'comparison/figure_sources.json',HERE/'note/table_sources.json']:
        values=json.loads(ledger.read_text())
        check(f'{ledger.parent.name} data sources unchanged',all(hashlib.sha256((REPO/k).read_bytes()).hexdigest()==v for k,v in values.items()))
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    result=dict(status='passed',passed_checks=len(checks),checks=checks,
        scientific_status='Conditional diagnostics. The retained 71 MeV retraining result fails nominal exclusion closure in both methods.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (qa/'numerical_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(checks)} saved-artifact checks passed; scientific failures retained.')


if __name__=='__main__':main()
