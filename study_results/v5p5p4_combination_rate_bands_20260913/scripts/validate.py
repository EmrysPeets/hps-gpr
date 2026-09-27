"""Numerical/provenance/semantic checks; render every final report page."""
from pathlib import Path
import hashlib,json,sys,ast
import numpy as np,pandas as pd,fitz
B=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks={}
manifest=json.loads((B/'provenance/parent_v553.json').read_text())
for v in manifest['files']:
 if v['path'] not in manifest.get('superseded_in_derivative',[]):checks['snapshot:'+v['path']]=sha(B/v['path'])==v['sha256']
 p=Path(v['parent'])
 if p.exists():checks['parent_unchanged:'+v['path']]=sha(p)==v['sha256']
for f in (B/'scripts').glob('*.py'):ast.parse(f.read_text(),filename=str(f))
for stem in ['extracted_combination','rate_band','rate_scan']:
 d=json.loads((B/f'qa/{stem}_validation.json').read_text());checks['numerical:'+stem]=d['passed']
s=pd.read_csv(B/'derived/extracted_combination_scan.csv');r=pd.read_csv(B/'derived/rate_significance_scan.csv');rb=pd.read_csv(B/'derived/rate_bands.csv');fits=json.loads((B/'derived/rate_band_fits.json').read_text())
checks['scan_counts']=len(s)==17 and len(r)==34;checks['same_mass_grid']=np.array_equal(s.mass_MeV.to_numpy(),r[r.model=='power'].mass_MeV.to_numpy())
checks['bands_ordered']=bool(((rb.lower_95<=rb.lower_68)&(rb.lower_68<=rb.central_epsilon2)&(rb.central_epsilon2<=rb.upper_68)&(rb.upper_68<=rb.upper_95)&(rb.lower_95>0)).all())
checks['rate_script_identity']=fits['script_sha256']==sha(B/'scripts/rate_uncertainties.py');checks['experiment_identity']=fits['experiment_sha256']==sha(B/'engine/experiment.py')
for kind in ['power','exponential']:
 q=r[(r.mass_MeV==92)&(r.model==kind)].iloc[0];checks['independent_exact_rate_replay:'+kind]=abs(q.Q_exact-fits['models'][kind]['Q0'])<1e-7
mc=pd.read_csv(B/'derived/gaussian_score_calibration.csv').set_index('name').loc['scaled_fixed_energy_free'];q=r[(r.mass_MeV==92)&(r.model=='power')].iloc[0];checks['old_MC_agrees']=abs(q.p_reference_at_exact_Q-mc.p)<3*mc.mcse
checks['coupling_removed_yield_limit']=bool((s.independent_GLS_total_yield_upper90>s.common_GLS_total_yield_upper90).all())
checks['finite_new_inference']=bool(np.isfinite(r.select_dtypes('number')).all().all() and np.isfinite(s).all().all())
pdf=B/'pdf/HPS_GPR_v5p5p4_Combination_Rate_Uncertainties.pdf';doc=fitz.open(pdf);texts=[p.get_text() for p in doc];full='\n\f\n'.join(texts);flat=' '.join(full.split())
checks['13_pages']=len(doc)==13;checks['no_orphan_pages']=all(len(t)>500 for t in texts)
for term in ['v5.5.4','3.962','3.646','3.812','3.782','30,944','36,523','Sid','log-Wald','not a global calibration','Simultaneous fit to extracted signals','Retained v5.5.3','77,802']:
 checks['text:'+term]=term in flat
checks['no_unresolved_citations']='[?]' not in full;checks['no_replacement_characters']='\ufffd' not in full;checks['no_overfull_boxes']='Overfull' not in (B/'qa/main.log').read_text()
out=B/'qa/rendered';out.mkdir(exist_ok=True)
for p in out.glob('page-*.png'):p.unlink()
for i,p in enumerate(doc):p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'page-{i+1:02}.png')
(B/'qa/extracted_text.txt').write_text(full);checks={k:bool(v) for k,v in checks.items()};result=dict(passed=all(checks.values()),checks=checks,check_count=len(checks),pages=len(doc),pdf_sha256=sha(pdf))
(B/'qa/document_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=[k for k,v in checks.items() if not v],pages=len(doc)),indent=2));sys.exit(0 if result['passed'] else 1)
