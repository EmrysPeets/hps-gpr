#!/usr/bin/env python3
"""Verify saved statistical products and final report without rerunning fits."""
from pathlib import Path
import hashlib,json,re,subprocess
import numpy as np
import pandas as pd
from scipy.stats import norm
from pypdf import PdfReader
import fitz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks={};g=HERE/'global/2015';a=g/'analysis'
    s=json.loads((a/'summary.json').read_text());curves=pd.read_csv(a/'pvalue_curves.csv')
    cov=np.load(a/'covariance.npz');mx=np.load(a/'maxima.npz');asv=np.load(g/'asimov/scan_vectors.npz');valid=np.load(g/'validation1000/scan_vectors.npz')
    checks['three_complete_ensembles']=all(json.loads((g/e/'summary.json').read_text())['complete'] for e in ('pilot10','validation1000','asimov'))
    checks['72_hypotheses_per_method']=curves.groupby('method').size().to_dict()=={'fixed':72,'profiled':72}
    for method in ('profiled','fixed'):
        d=curves[curves.method==method].sort_values('mass_MeV');r=d.observed_r.to_numpy();score=(r-d.asimov_r)/d.response_sd
        response=asv[method][1:]-asv[method][0]
        checks[method+'_covariance_from_saved_response']=bool(np.allclose(response.T@response,cov[method+'_C'],rtol=1e-13,atol=1e-13))
        checks[method+'_valid_correlation']=bool(np.allclose(np.diag(cov[method+'_K']),1) and np.linalg.eigvalsh(cov[method+'_K']).min()>-1e-10)
        p=np.where(r>0,norm.sf(score),1.)
        checks[method+'_local_curve_definition']=bool(np.allclose(p,d.p_local_common_truth,rtol=1e-12,atol=1e-15))
        observed_score=np.where(r>0,score,-np.inf)
        expected=np.array([np.mean(mx[method+'_gp']>=v) for v in observed_score])
        checks[method+'_global_curve_counts']=bool(np.allclose(expected,d.p_global_gp,rtol=0,atol=1e-15))
        raw=np.array([np.mean(mx[method+'_gp_raw']>=max(0,v)) for v in r])
        checks[method+'_raw_ordering_counts']=bool(np.allclose(raw,d.p_global_raw_ordering,rtol=0,atol=1e-15))
        z=(valid[method]-d.asimov_r.to_numpy())/d.response_sd.to_numpy()
        direct=np.where(valid[method]>0,z,-np.inf).max(axis=1)
        checks[method+'_direct_maximum_from_whole_scans']=bool(np.allclose(direct,mx[method+'_direct'],rtol=1e-12,atol=1e-13))
        checks[method+'_bounded_atom']=bool(np.all(d.loc[r<=0,['p_local_common_truth','p_global_gp','p_global_direct']].to_numpy()==1))
        checks[method+'_finite_curve_intervals']=bool(np.all(d.p_global_gp_low<=d.p_global_gp) and np.all(d.p_global_gp_high>=d.p_global_gp))
    build=json.loads((HERE/'provenance/report_build.json').read_text());pdf=Path(build['pdf'])
    checks['report_input_hashes']=all(sha(ROOT/p)==h for p,h in build['inputs'].items())
    checks['report_pdf_hash']=sha(pdf)==build['pdf_sha256']
    reader=PdfReader(pdf);texts=[p.extract_text() for p in reader.pages];full='\n'.join(texts)
    checks['14_pages']=len(texts)==14
    checks['no_orphan_or_empty_page']=all(len(t)>600 for t in texts)
    checks['no_unresolved_references']=not any(x in full for x in ('??','TODO','PLACEHOLDER'))
    log=(HERE/'note/build.log').read_text()+(pdf.parent/'reader_report.log').read_text()
    checks['no_tex_overfull_or_undefined']=not re.search(r'Overfull|undefined|LaTeX Warning',log,re.I)
    checks['limitations_visible']=all(x in full for x in ('two-truth envelope','unresolved by direct scans','different tests','2016 numerical exception'))
    bounds=[]
    for i,p in enumerate(fitz.open(pdf),1):
        for b in p.get_text('blocks'):
            if b[4].strip()==str(i) and b[1]>740:continue
            if b[0]<35 or b[1]<30 or b[2]>577 or b[3]>765:bounds.append({'page':i,'bbox':b[:4]})
    checks['page_content_inside_bounds']=not bounds
    pages=sorted((HERE/'qa/pages').glob('page-*.png'))
    checks['all_final_pages_rendered']=len(pages)==len(reader.pages)
    checks['shared_checkout_branch_and_index_preserved']=(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='cd8f5bf2bae4eff0ce9442be7774bcf74a559c9c' and not subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT))
    record={'passed':all(checks.values()),'checks':checks,'check_count':len(checks),'pdf_sha256':sha(pdf),'page_count':len(texts),'bounds_violations':bounds,'rendered_pages_sha256':{str(p.relative_to(ROOT)):sha(p) for p in pages},'statistical_scope':'conditional finite-grid approximation; fixed rare tail unresolved by direct scans'}
    (HERE/'qa/product_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not record['passed']:raise RuntimeError('Product validation failed')
if __name__=='__main__':main()
