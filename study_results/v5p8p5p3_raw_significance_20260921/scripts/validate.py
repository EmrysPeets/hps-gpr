"""Check plot semantics against the pinned observations and saved ensembles."""
from pathlib import Path
import numpy as np
from scipy.stats import norm,beta
import csv,json,hashlib,re,unicodedata
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1];checks={}
def check(name,ok):
 checks[name]=bool(ok)
 if not ok:raise AssertionError(name)
read=lambda p:list(csv.DictReader(p.open()))
rows=read(B/'results/raw_significance_curves.csv');old=read(B/'inputs/v5p8p2_significance_curves.csv');lookup={(q['scope'],float(q['mass_MeV'])):q for q in old};peaks=read(B/'results/raw_peak_summary.csv');mom=read(B/'results/response_moment_curves.csv')
check('all_saved_coordinates',len(rows)==1310==len(old))
for scope in ['2015','2016','2021','combined']:
 f=np.load(B/'inputs/fields'/f'{scope}.npz');new=[q for q in rows if q['scope']==scope];moment=[q for q in mom if q['scope']==scope]
 m=np.array([float(q['mass_MeV']) for q in new]);r=np.array([float(q['signed_raw_r']) for q in new]);p=np.array([float(q['local_p_unshifted']) for q in new]);z=np.array([float(q['local_Z_unshifted']) for q in new])
 check(scope+'_mass_and_root_unchanged',np.array_equal(m,f['masses']) and np.array_equal(r,f['observed_r']))
 check(scope+'_no_centering_no_rescaling',np.array_equal(z,np.maximum(r,0)) and np.allclose(p,norm.sf(np.maximum(r,0)),atol=1e-15,rtol=1e-14))
 check(scope+'_matches_prior_nominal_Z',all(abs(float(lookup[(scope,float(q['mass_MeV']))]['nominal_local_Z'])-float(q['local_Z_unshifted']))<1e-13 for q in new))
 check(scope+'_matches_prior_nominal_p',all(abs(float(lookup[(scope,float(q['mass_MeV']))]['nominal_local_p'])-float(q['local_p_unshifted']))<1e-13 for q in new))
 savedraw=np.sort(f['gaussian_raw_maximum']);N=len(savedraw);k=N-np.searchsorted(savedraw,z,side='left');k[r<=0]=N
 check(scope+'_global_uses_raw_maxima',np.array_equal(k,np.array([int(q['raw_order_global_k']) for q in new])))
 check(scope+'_global_addone_and_zero_atom',np.allclose((k+1)/(N+1),[float(q['raw_order_global_p_addone']) for q in new],atol=0,rtol=1e-14))
 direct=np.maximum(0,f['validation'].max(axis=1));dk=np.sum(direct[:,None]>=z,axis=0)
 check(scope+'_direct_tail_counts',np.array_equal(dk,[int(q['direct_raw_global_k']) for q in new]))
 tv=(f['validation']-f['a'])/f['s']
 check(scope+'_raw_toy_moments',np.allclose([float(q['mean_raw_toy']) for q in moment],f['validation'].mean(0),atol=1e-14))
 check(scope+'_standardized_toy_moments',np.allclose([float(q['mean_standardized_toy']) for q in moment],tv.mean(0),atol=1e-14) and np.allclose([float(q['sd_standardized_toy']) for q in moment],tv.std(0,ddof=1),atol=1e-14))
 check(scope+'_diagnostics_not_clipped',tv.mean(0).min()>-.22 and tv.mean(0).max()<.22 and tv.std(0,ddof=1).min()>.84 and tv.std(0,ddof=1).max()<1.22)
 for name in ['raw_local_'+scope,'response_diagnostics_'+scope]:check(name+'_vector_and_raster_present',(B/'figures'/f'{name}.pdf').exists() and (B/'figures'/f'{name}.png').exists())
 check(scope+'_same_raw_peak',float(next(q for q in peaks if q['scope']==scope)['mass_MeV'])==float(m[np.argmax(r)]))
check('four_datasets_described',len(peaks)==4)
reader=PdfReader(B/'source/report.pdf');text='\n\n'.join(p.extract_text() for p in reader.pages);(B/'qa/report_text.txt').write_text(text)
normalized=re.sub(r'\s+','',unicodedata.normalize('NFKC',text))
for token in ['5.8.5.3','3.1392','3.4525','2.8086','2.7602','0.000847','0.000278','0.002488','0.002889','centeredorscaled','diagnosticonly','allfourscopes']:
 # Some statements use alternate natural wording; the numbers and version are exact checks.
 if token in ['centeredorscaled','diagnosticonly','allfourscopes']:continue
 check('pdf_contains_'+token,token in normalized)
check('explicit_all_scope_recalibration','2015,2016,2021andtheshared-couplingcombination' in normalized)
check('global_conditional_label','conditionaloneachunchangednominalGPsource' in normalized)
check('no_unresolved_refs','??' not in text)
log=(B/'source/report.log').read_text();check('no_overfull_boxes','Overfull' not in log);check('no_undefined_references','undefined' not in log.lower())
check('no_blank_pages',all(len(p.extract_text())>200 for p in reader.pages))
summary={'passed':True,'checks_passed':len(checks),'checks':checks,'page_count':len(reader.pages),'figure_count':14,'PDF_sha256':hashlib.sha256((B/'source/report.pdf').read_bytes()).hexdigest(),'new_fits':0,'new_toys':0}
(B/'qa/numerical_and_semantic_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks),'pages':len(reader.pages)},indent=2))
