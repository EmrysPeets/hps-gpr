"""Saved-result and PDF semantic audit; requires numpy, scipy, pypdf."""
from pathlib import Path
import json,csv,hashlib,re
import numpy as np
from scipy.stats import beta,norm
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1]
checks={}
def check(name,condition):
 checks[name]=bool(condition)
 if not condition:raise AssertionError(name)
s=json.loads((B/'results/stats_summary.json').read_text())
rows=s['raw_and_reference'];r=next(x for x in rows if x['scope']=='2016' and x['test']=='raw_maximum');z=next(x for x in rows if x['scope']=='2016' and x['test']=='reference_maximum')
check('2016_raw_peak_unchanged',abs(r['raw_r']-3.4525000631180385)<1e-12 and r['peak_mass_MeV']==90.5)
check('2016_reference_peak_distinct',z['peak_mass_MeV']==91.5 and abs(z['reference_z']-2.885954890466647)<1e-12)
check('raw_and_reference_direct_counts',r['Poisson_k']==18 and z['Poisson_k']==36 and r['Poisson_N']==256)
check('raw_tail_conditional_mapping',abs(norm.sf((r['raw_r']-r['a'])/r['response_s'])-r['conditional_local_p'])<1e-14)
for q in rows:
 check(q['scope']+'_'+q['test']+'_global_count',abs(q['Gaussian_k']/q['Gaussian_N']-q['Gaussian_p'])<1e-15)
 check(q['scope']+'_'+q['test']+'_CP',abs(beta.ppf(.025,q['Poisson_k'],q['Poisson_N']-q['Poisson_k']+1)-q['Poisson_lo95'])<1e-12)
for q in s['regions']:
 check(q['scope']+'_samefield_bound',q['Gaussian_full_p']<=q['exact_grid_upcrossing_bound_p'])
check('region_draws_500000',s['block_peak']['Gaussian_joint_N']==500000)
check('new_width_fixed',s['blind_sigma']==2.25)
p=json.loads((B/'results/physics_source_manifest.json').read_text());summary=json.loads((B/'results/physics_source_summary.json').read_text())
check('source_identity',hashlib.sha256((B/'inputs/physics_high_psum_1pct.root').read_bytes()).hexdigest()==p['source_sha256'])
check('source_transfer_qualified',all(p[x] is False for x in ['source_selection_equivalent_verified','exact_exposure_ratio_verified','event_overlap_verified','source_estimation_uncertainty_propagated']))
check('source_fit_counts',summary['deterministic_states']==603 and summary['Poisson_fits']==768)
check('source_fit_convergence',summary['max_fit_score']<2e-7)
check('source_width_fixed',p['analysis_blind_half_width_sigma']==2.25)
with (B/'results/physics_signal_transfer.csv').open() as f:inj=list(csv.DictReader(f))
check('injection_separates_source_and_extractor',all(float(x['fixed_source_fit_recovery'])>.96 and .77<float(x['rebuilt_source_window_absorption'])<.85 for x in inj))
reader=PdfReader(B/'source/report.pdf');raw='\n\n'.join(x.extract_text() for x in reader.pages)
(B/'qa/report_text.txt').write_text(raw)
normtext=re.sub(r'\s+','',raw)
for token in ['3.45250','2.88595','0.067795','0.104875','0.120378','0.14341','0.10640','603','768','2.25','84.04','77.83','90.5','91.5','195/256']:
 check('pdf_contains_'+token,token in normtext)
check('pdf_no_unresolved_reference','??' not in raw)
check('pdf_nonempty_all_pages',all(len(x.extract_text())>100 for x in reader.pages))
log=(B/'source/report.log').read_text()
check('no_overfull_boxes','Overfull' not in log)
check('no_undefined_references','undefined' not in log.lower())
check('outer_truth_explicit','underlyingbackgroundtruth' in normtext and 'jointlygenerate' in normtext)
summary={'passed':all(checks.values()),'checks':checks,'page_count':len(reader.pages),'figure_count':raw.count('Figure '),'pdf_sha256':hashlib.sha256((B/'source/report.pdf').read_bytes()).hexdigest()}
(B/'qa/semantic_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'passed':summary['passed'],'checks':len(checks),'pages':len(reader.pages)},indent=2))
