#!/usr/bin/env python3
"""Recompute likelihood and display identities from saved arrays and toy rows."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm,beta
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
checks=[]
def check(name,test,**detail):
    checks.append(dict(name=name,passed=bool(test),**detail))
    if not test:raise RuntimeError(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dev(n,mu):
    positive=n>0
    out=mu-n
    out[positive]+=n[positive]*np.log(n[positive]/mu[positive])
    return float(out.sum())
scan=pd.read_csv(HERE/'derived/scan.csv');stable=pd.read_csv(HERE/'derived/kernel_stability.csv')
allrows=pd.concat([scan,stable],ignore_index=True)
check('all 232 declared mass/model fits saved',len(allrows)==232)
check('every expected count positive',(allrows.min_lambda>0).all())
check('fit score threshold',(allrows.max_score<2e-7).all())
for row in allrows.itertuples():
    tag=f'{row.method}_m{row.mass_MeV:05.2f}'
    a=np.load(HERE/'derived/fits'/f'{tag}.npz');mask=a['mask'];n=a['n'][mask]
    bf,tot,s=a['bfit'][mask],a['total'][mask],a['signal'][mask]
    check(tag+' components',np.allclose(bf+s,tot,rtol=1e-11,atol=1e-7))
    check(tag+' Gaussian integrals',np.allclose(a['template'],np.diff(ndtr((a['edges']-row.mass_MeV)/row.sigma_MeV)),rtol=1e-13,atol=1e-15))
    check(tag+' integer data',np.array_equal(a['n'],a['n'].round()))
    nll=dev(n,tot);nll0=dev(n,a['bnull'][mask])
    if row.method.startswith('gp_'):
        nll+=.5*float(a['free_z'][1:]@a['free_z'][1:])
        nll0+=.5*float(a['null_z']@a['null_z'])
        check(tag+' positive covariance',np.linalg.eigvalsh(a['fit_cov']).min()>0)
        check(tag+' window amplitude',abs(float(s.sum())-row.Ahat_window)<1e-6)
        check(tag+' signal fraction',abs(a['template'][mask].sum()-row.signal_fraction_in_fit)<1e-12)
    rr=np.sign(row.Ahat_total)*np.sqrt(max(0,2*(nll0-nll)))
    check(tag+' independently computed root',abs(rr-row.r)<2e-5,error=float(abs(rr-row.r)))
    check(tag+' local asymptotic p',abs(norm.sf(max(rr,0))-row.p0)<3e-6)
for upper,suffix in ((8,''),(16,'_ceiling16')):
    toys=pd.read_csv(HERE/'derived'/f'toy_roots{suffix}.csv')
    summary=json.loads((HERE/'derived'/f'toy_summary{suffix}.json').read_text())
    check(f'ceiling {upper} complete toy bank',len(toys)==400 and not toys.duplicated(['mass_MeV','toy_id']).any())
    pilot=pd.read_csv(HERE/'derived'/f'pilot10_toy_roots{suffix}.csv')
    joined=pilot.merge(toys,on=['mass_MeV','toy_id'],suffixes=('_pilot','_full'),validate='one_to_one')
    check(f'ceiling {upper} pilot IDs retained',len(joined)==40 and np.array_equal(joined.r_pilot,joined.r_full))
    for item in summary['anchors']:
        values=toys[toys.mass_MeV==item['mass_MeV']].r
        k=int((values>=item['observed_r']).sum()) if item['observed_r']>0 else len(values)
        check(f'ceiling {upper} tails {item["mass_MeV"]}',k==item['k'] and item['n']==100 and item['p_hat']==k/100)
        lo=0. if k==0 else beta.ppf(.025,k,101-k)
        hi=1. if k==100 else beta.ppf(.975,k+1,100-k)
        check(f'ceiling {upper} intervals {item["mass_MeV"]}',abs(lo-item['low95'])<1e-12 and abs(hi-item['high95'])<1e-12)
for p in (HERE/'derived').glob('display_mapping_*.npz'):
    a=np.load(p);W=a['W'];inside=a['inside']
    check(p.name+' whole-bin non-overlapping map',np.isin(W,[0,1]).all() and np.max(W.sum(0))<=1)
    check(p.name+' counting covariance',np.array_equal(a['count_covariance'],np.diag(a['counts'])))
    check(p.name+' nonempty window display',inside.any())
for ceiling in ('gp_ceiling16','gp_ceiling32','gp_ceiling64'):
    d=stable[stable.method==ceiling]
    check(ceiling+' no active kernel bounds',not d.kernel_at_boundary.any())
    if ceiling!='gp_ceiling16':
        ref=stable[stable.method=='gp_ceiling16'].set_index('mass_MeV');d=d.set_index('mass_MeV')
        error=float(np.max(abs(ref.r-d.r)))
        check(ceiling+' stable optimum',error<.001,max_root_difference=error)
source=json.loads((HERE/'provenance/input.json').read_text())
check('released 2015 source file identity',sha(source['path'])==source['file_sha256']=='58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f')
summary=json.loads((HERE/'derived/summary.json').read_text())
for path,digest in summary['sources'].items():check('numerical source '+path,sha(ROOT/path)==digest)
build=json.loads((HERE/'provenance/report_build.json').read_text())
for path,digest in build['pdfs'].items():
    check('PDF hash '+path,sha(ROOT/path)==digest)
    reader=PdfReader(ROOT/path);txt='\n'.join(page.extract_text() for page in reader.pages)
    for term in ('15–20 MeV','17.25','0.057','conditional','800'):
        check(path+' contains '+term,term in txt)
    check(path+' resolved references','??' not in txt)
    check(path+' no accidental nan','nan' not in txt.lower().split())
    (HERE/'qa'/(Path(path).stem+'.txt')).write_text(txt)
for log in (HERE/'note').glob('*_build.log'):
    content=log.read_text()
    check(log.name+' no LaTeX overflow',not any(s in content for s in ('Overfull','undefined references','Missing character','error:')))
report=dict(passed=True,conditions=len(checks),checks=checks,
    independence='Saved likelihood arrays were recomputed directly; no sub-agent or separate physics review was used in this side conversation.',
    claim='Numerical consistency does not establish physical background validity, global significance, signal acceptance, or interval coverage.')
(HERE/'qa/numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('passed','conditions','claim')},indent=2))
