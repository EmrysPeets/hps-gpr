#!/usr/bin/env python3
"""Validate the v5.0.1 document, frozen values, and bundled provenance."""
from pathlib import Path
import hashlib,json,re,subprocess
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1];R=B.parents[1];S=B/'source';checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(name,result,detail=None):checks.append(dict(name=name,passed=bool(result),detail=detail))
files=[];missing=[]
def walk(p):
 if p in files:return
 if not p.exists():missing.append(str(p));return
 files.append(p)
 for x in re.findall(r'\\input\{([^}]+)\}',p.read_text()):
  if '#' not in x:walk(S/(x if x.endswith('.tex') else x+'.tex'))
walk(S/'main.tex');text='\n'.join(p.read_text() for p in files)
active=re.sub(r'\\ifwritingsample(.*?)\\fi',lambda m:m.group(1).split(r'\else')[0],text,flags=re.S)
aux=(B/'qa/build/main.aux').read_text();labels=re.findall(r'\\newlabel\{([^}]+)\}',aux)
refs=set(re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',active))
check('all_TeX_inputs_exist',not missing,missing)
check('all_references_resolve',not(refs-set(labels)),sorted(refs-set(labels)))
check('unique_labels',len(labels)==len(set(labels)),len(labels))
log=(B/'qa/build/main.log').read_text(errors='replace')
bad=[x for x in log.splitlines() if any(y in x for y in ['undefined','Overfull','Missing character','multiply defined'])]
check('no_missing_or_overfull_TeX',not bad,bad)
bib=(S/'hps_gpr_analysis_note.bib').read_text();keys=set(re.findall(r'@\w+\s*\{([^,]+),',bib));cites={x.strip() for g in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',active) for x in g.split(',')}
check('all_citation_keys_exist',not(cites-keys),sorted(cites-keys))
seen={}
fig=json.loads((B/'editorial/v501_figure_provenance.json').read_text())
for f in fig['records']:
 for x in f.get('inputs',[])+f.get('outputs',[]):seen[x['path']]=x['sha256']
for f in json.loads((B/'editorial/method_diagram_provenance.json').read_text())['figures']:
 for x in f['outputs']:seen[x['path']]=x['sha256']
for x in json.loads((B/'provenance/figure_inputs/input_map.json').read_text()).values():seen[x['snapshot']]=x['sha256']
for x in json.loads((B/'editorial/historical_provenance.json').read_text())['assets']:
 p=Path(x['destination']);seen[str(p.relative_to('study_results/'+B.name))]=x['sha256']
mismatches=[p for p,h in seen.items() if not(B/p).is_file() or sha(B/p)!=h]
check('bundled_figure_and_input_hashes',not mismatches,{'files':len(seen),'mismatches':mismatches})
parent=R/'study_results/v5_analysis_note_20260908';changed=[];frozen=0
for folder in ['figures','derived']:
 for p in (parent/folder).rglob('*'):
  if p.is_file():
   q=B/p.relative_to(parent)
   if q.exists():
    frozen+=1
    if sha(p)!=sha(q):changed.append(str(q.relative_to(B)))
check('frozen_parent_figures_and_tables_unchanged',not changed,{'compared':frozen,'changed':changed})
check('no_new_HPS_fits_data_or_toys',not any(fig[x] for x in ['new_HPS_fits','new_HPS_data','new_toys']))
check('all_232_observed_points_preserved',fig['checks']['babar_projection']['points']==232 and fig['checks']['babar_projection']['current_curve_exact_v500_match'],fig['checks']['babar_projection'])
pdf=B/'qa/build/main.pdf';reader=PdfReader(pdf);pages=[p.extract_text() or '' for p in reader.pages]
check('version_and_review_identity','v5.0.1' in pages[0] and 'Draft' in pages[0])
check('no_unresolved_PDF_references',not any('??' in p or '\ufffd' in p for p in pages))
check('no_nearly_blank_pages',all(len(p)>180 for p in pages),[(i+1,len(p))for i,p in enumerate(pages) if len(p)<=180])
req=json.loads((B/'editorial/request_checklist.json').read_text());absent=[]
for item in req['items']:
 for label in item['labels']:
  if label not in labels:absent.append(label)
check('all_requested_targets_in_built_PDF',not absent,absent)
check('parent_tracked_state_unchanged',not subprocess.check_output(['git','diff','--name-only','3c557e391'],cwd=R,text=True).strip())
report={'version':'5.0.1','passed':all(c['passed']for c in checks),'pdf_sha256':sha(pdf),'page_count':len(pages),'checks':checks,'build_note':'Cached Tectonic emits a BibTeX rerun notice; labels, citations, page text, and portable rebuild equality are independently verified.','scope':'Editorial and display derivative; no new HPS inference or unblinding authorization.'}
(B/'qa/final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'pages':len(pages),'checks':len(checks),'failures':[c for c in checks if not c['passed']]},indent=2))
raise SystemExit(0 if report['passed'] else 1)
