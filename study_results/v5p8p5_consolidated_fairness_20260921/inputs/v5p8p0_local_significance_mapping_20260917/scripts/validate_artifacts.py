#!/usr/bin/env python3
"""Independent final-ledger, source identity and report completeness checks."""
from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.stats import norm
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1];checks=[]
def check(name,value):
 checks.append(dict(check=name,passed=bool(value)))
 if not value:raise AssertionError(name)
q=pd.read_csv(B/'local/local_tail_tests.csv');check('22 complete analysis-sample local scenarios',len(q)==22)
check('11 independent anchors, two backgrounds each',q.groupby(['dataset','mass_MeV']).size().eq(2).all())
check('1024 experiments per analysis-sample scenario',q.N.eq(1024).all())
check('valid ordered probability intervals',((q.p_lo95>=0)&(q.p_hi95<=1)&(q.p_hi95>=q.p_lo95)).all())
check('exact zero-statistic atom convention',(q.loc[q.bounded_atom,['p_mle','p_addone','p_lo95','p_hi95']]==1).all().all())
check('nominal probabilities match observed likelihood roots',np.max(abs(q.nominal_p0-norm.sf(np.maximum(q.observed_r,0))))<1e-12)
check('independent local reviewer passed',json.loads((B/'statistics/independent_local_review.json').read_text()).get('passed',False))
h=pd.read_csv(B/'historical10/actual10_local_tests.csv')
check('12 historical-subset local scenarios',len(h)==12)
check('6144 historical-subset finite experiments',int(h.n_finite.sum())==6144 and h.n_generated.eq(512).all())
f=np.load(B/'gpr/field_2016/field.npz');m=f['masses'];expected=np.unique(np.r_[np.arange(39,180.01,.5),np.arange(74.25,79,.25)])
check('complete full 0.5 grid with separate quarter-step coordinates',np.array_equal(m,expected))
check('293 total 2016 coordinates',len(m)==293)
check('response covariance reconstructs',np.max(abs(f['D'].T@f['D']-f['C']))<1e-10)
check('unit correlation diagonal',np.max(abs(np.diag(f['K'])-1))<1e-12)
check('correlation positive semidefinite',np.linalg.eigvalsh(f['K']).min()>-1e-9)
check('256 coherent validation spectra per mass',f['validation'].shape==(256,293))
g=pd.read_csv(B/'gpr/local_mapping.csv');check('complete three-campaign mapping ledger',len(g)==293+82+201)
check('nominal GP ledger tails use raw root',np.max(abs(g.p_raw_gaussian-norm.sf(np.maximum(g.observed_r,0))))<1e-12)
check('stiffness validation passed',json.loads((B/'statistics/stiffness_validation.json').read_text())['passed'])
check('combination validation passed',json.loads((B/'statistics/validation.json').read_text())['passed'])
check('exact/accelerated GP execution complete',json.loads((B/'gpr/execution.json').read_text())['complete'])
sources=json.loads((B/'inputs/report_sources.json').read_text())
check('referenced frozen sources unchanged',all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in sources.items()))
reader=PdfReader(B/'source/report.pdf');texts=[p.extract_text() for p in reader.pages];text='\n'.join(texts)
check('report has nine to eleven substantive pages',9<=len(texts)<=11 and min(map(len,texts))>650)
check('no unresolved placeholder or replacement glyph',not any(x in text.lower() for x in ['placeholder','todo','pending figure','\ufffd']))
check('report identifies required version and methods',all(s in text for s in ['5.8.0','4.20','6.11','90.5','0.5','1,024','512']))
check('all six figures referenced',all(f'Figure {i}.' in text for i in range(1,7)))
log=(B/'source/report.log').read_text()
check('no overfull boxes or missing references',not any(s in log for s in ['Overfull','Undefined control','undefined references','LaTeX Error']))
out=dict(passed=all(c['passed'] for c in checks),checks=checks,pages=len(texts),minimum_page_text_characters=min(map(len,texts)),
 new_local_Poisson_experiments=int(q.N.sum()),new_historical_Poisson_experiments=int(h.n_finite.sum()),
 grid_nodes_2016=len(m),PDF_SHA256=hashlib.sha256((B/'source/report.pdf').read_bytes()).hexdigest(),
 physical_discovery_significance_validated=False)
(B/'qa/final_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
