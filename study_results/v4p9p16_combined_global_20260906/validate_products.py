#!/usr/bin/env python3
"""Validate final probability identities, source bindings and rendered note."""
from pathlib import Path
import hashlib,json,re,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.stats import norm,beta
from pypdf import PdfReader
import fitz
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    f=HERE/'global';a=f/'analysis'
    s=json.loads((a/'summary.json').read_text())
    qa=json.loads((HERE/'qa/numerical_validation.json').read_text())
    checks={}
    checks['numerical_audit_passed']=qa['passed'] and qa['verifier_sha256']==sha(HERE/'verify_combined.py')
    checks['analysis_input_hashes']=all(sha(ROOT/p)==h for p,h in s['input_sha256'].items())
    d=pd.read_csv(a/'pvalue_curves.csv')
    cov=np.load(a/'covariance.npz');maxima=np.load(a/'maxima.npz')
    vectors=np.load(f/'scan_vectors.npz')
    obs=pd.read_csv(f/'observed.csv').set_index('mass_MeV')
    for method in ('profiled','fixed'):
        x=d[d.method==method].sort_values('mass_MeV');r=x.observed_r.to_numpy()
        mean=x.asimov_r.to_numpy();sd=x.response_sd.to_numpy()
        z=(r-mean)/sd;score=np.where(r>0,z,-np.inf)
        checks[method+'_grid']=np.array_equal(x.mass_MeV,np.arange(19,251))
        checks[method+'_observed_roots']=np.allclose(r,obs[method+'_r'],rtol=1e-13,atol=1e-13)
        checks[method+'_local_rule']=np.allclose(np.where(r>0,norm.sf(z),1),x.p_local_common_truth,rtol=1e-12,atol=1e-15)
        checks[method+'_bounded_atom']=bool(np.all(x.loc[r<=0,['p_local_common_truth','p_global_gp','p_global_direct']].to_numpy()==1))
        response=vectors['asimov_'+method][1:]-vectors['asimov_'+method][0]
        checks[method+'_response_covariance']=np.allclose(response.T@response,cov[method+'_C'],rtol=1e-13,atol=1e-13)
        checks[method+'_covariance_positive']=bool(np.linalg.eigvalsh(cov[method+'_K']).min()>-1e-9 and np.allclose(np.diag(cov[method+'_K']),1))
        valid=vectors['validation1000_'+method]
        direct=np.where(valid>0,(valid-mean)/sd,-np.inf).max(axis=1)
        checks[method+'_coherent_direct_maximum']=np.allclose(direct,maxima[method+'_direct'],rtol=1e-12,atol=1e-12)
        checks[method+'_direct_raw_maximum']=np.array_equal(np.maximum(valid,0).max(axis=1),maxima[method+'_direct_raw'])
        for source in ('gp','direct'):
            arr=maxima[method+'_'+source]
            count=np.array([np.count_nonzero(arr>=v) for v in score])
            checks[method+'_'+source+'_counts']=np.array_equal(count,x[source+'_k']) and np.allclose(count/len(arr),x['p_global_'+source],rtol=0,atol=1e-15)
            upper=np.array([1. if k==len(arr) else beta.ppf(.95,k+1,len(arr)-k) for k in count])
            checks[method+'_'+source+'_bounds']=np.allclose(upper,x['p_global_'+source+'_upper95'],rtol=1e-12,atol=1e-15)
        raw=np.array([np.count_nonzero(maxima[method+'_gp_raw']>=max(0,v)) for v in r])
        checks[method+'_raw_counts']=np.array_equal(raw,x.raw_gp_k) and np.allclose(raw/len(maxima[method+'_gp_raw']),x.p_global_raw_ordering,rtol=0,atol=1e-15)
        checks[method+'_selected_peak']=int(x.iloc[np.argmax(score)].mass_MeV)==s['methods'][method]['peak_mass_MeV']
    q=s['methods']['profiled']
    expected=sorted({30,65,120,220,q['peak_mass_MeV'],q['raw_ordering']['peak_mass_MeV']})
    checks['representative_selection']=expected==s['representative_masses_MeV']
    checks['full_observed_limit']=len(obs)==232 and np.isfinite(obs.profiled_eps2_display).all() and (obs.profiled_eps2_display>0).all()
    fig=json.loads((HERE/'provenance/figure_build.json').read_text())
    checks['figure_inputs']=all(sha(ROOT/p)==h for p,h in fig['inputs'].items())
    checks['figure_outputs']=all(sha(ROOT/p)==h for p,h in fig['files'].items())
    b=json.loads((HERE/'provenance/report_build.json').read_text())
    checks['report_inputs']=all(sha(ROOT/p)==h for p,h in b['inputs'].items())
    pdf=Path(b['pdf'])
    checks['report_hash']=sha(pdf)==b['pdf_sha256']
    text=[p.extract_text() for p in PdfReader(pdf).pages]
    joined=re.sub(r'-\s*\n\s*','-',' '.join(text))
    full=re.sub(r'\s+',' ',joined)
    checks['no_orphan_pages']=all(len(t)>450 for t in text)
    checks['references_resolved']=not any(w in full for w in ('??','TODO','PLACEHOLDER'))
    required=('v4.9.16','19–250','pointwise','asymptotic','different tests',
              '1,000','2016 numerical exception','signal-plus-background','one-sided 95%')
    checks['scientific_scope_visible']=all(w in full for w in required)
    log=(HERE/'note/build.log').read_text()+(pdf.parent/'analysis_note.log').read_text()
    checks['no_tex_layout_or_reference_warning']=not re.search(r'Overfull|undefined|LaTeX Warning',log,re.I)
    violations=[]
    for i,p in enumerate(fitz.open(pdf),1):
        for block in p.get_text('blocks'):
            if block[4].strip()==str(i) and block[1]>740:continue
            if block[0]<35 or block[1]<30 or block[2]>577 or block[3]>765:
                violations.append(dict(page=i,bbox=block[:4]))
    checks['content_in_bounds']=not violations
    pages=sorted((HERE/'qa/pages').glob('page-*.png'))
    checks['all_pages_rendered']=len(pages)==len(text)
    checks={k:bool(v) for k,v in checks.items()}
    record=dict(passed=all(checks.values()),checks=checks,check_count=len(checks),
        page_count=len(text),pdf_sha256=sha(pdf),bounds_violations=violations,
        rendered_pages_sha256={str(p.relative_to(ROOT)):sha(p) for p in pages})
    (HERE/'qa/product_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not record['passed']:raise RuntimeError('Final product validation failed')
if __name__=='__main__':main()
