#!/usr/bin/env python3
"""Read-only verification of curated source, figures, provenance and built PDF."""
from pathlib import Path
import hashlib,json,re,subprocess
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1]; R=B.parents[1]; S=B/'source';checks=[]
def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=detail))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
fp=json.loads((B/'editorial/figure_provenance.json').read_text())
seen={}
for fig in fp['figures']:
 for item in fig.get('sources',[])+fig.get('outputs',[]):seen[item['path']]=item['sha256']
for item in json.loads((B/'editorial/source_claims.json').read_text())['claims']:
 for src in item['evidence']:seen[src['path']]=src['sha256']
errors=[p for p,h in seen.items() if not (R/p).is_file() or sha(R/p)!=h]
check('bound_source_and_figure_hashes',not errors,{'files':len(seen),'mismatches':errors})
files=[];missing=[]
def walk(p):
 if p in files:return
 if not p.exists():missing.append(str(p));return
 files.append(p)
 for item in re.findall(r'\\input\{([^}]+)\}',p.read_text()):
  if '#' in item:continue
  walk(S/(item if item.endswith('.tex') else item+'.tex'))
walk(S/'main.tex');text='\n'.join(p.read_text() for p in files)
check('all_tex_inputs_exist',not missing,missing)
labels=re.findall(r'\\label\{([^}]+)\}',text)
# Figure macro labels occur as the fourth braced argument; the TeX aux is canonical.
aux=(B/'qa/build/main.aux').read_text();auxlabels=re.findall(r'\\newlabel\{([^}]+)\}',aux)
active_text=re.sub(r'\\ifwritingsample(.*?)\\fi', lambda m:m.group(1).split(r'\else')[0], text, flags=re.S)
refs=re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',active_text)
check('no_duplicate_labels',len(auxlabels)==len(set(auxlabels)),len(auxlabels))
check('all_references_resolve',not(set(refs)-set(auxlabels)),sorted(set(refs)-set(auxlabels)))
log=(B/'qa/build/main.log').read_text(errors='replace')
bad=[line for line in log.splitlines() if any(k in line for k in ['undefined','multiply defined','Overfull','Missing character'])]
check('no_undefined_or_overfull_tex',not bad,bad)
bib=(S/'hps_gpr_analysis_note.bib').read_text();keys=set(re.findall(r'@\w+\s*\{([^,]+),',bib))
cites={x.strip() for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',text) for x in group.split(',')}
check('all_citation_keys_exist',not(cites-keys),sorted(cites-keys))
pdf=B/'qa/build/main.pdf';reader=PdfReader(pdf);pages=[p.extract_text() or '' for p in reader.pages]
check('pdf_text_references_resolved',not any('??' in p for p in pages))
check('v5_identity_and_draft_status','v5.0.0' in pages[0] and 'Draft' in pages[0])
actual=[(i+1,p) for i,p in enumerate(pages) if i>5]
history=[i for i,p in actual if 'Appendix contents' not in p and re.search(r'\bA\s+Change log and study history',p)]
contents=[i for i,p in actual if p.lstrip().startswith('HPS Gaussian') and 'Appendix contents' in p]
calibration=[i for i,p in actual if 'Appendix contents' not in p and re.search(r'\bB\s+Alternative background modeling',p)]
check('appendix_history_then_own_contents',bool(history and contents and calibration) and min(history)<min(contents)<min(calibration),{'history':history,'contents':contents,'calibration':calibration})
check('no_new_toys_or_fits',fp['new_toys']==0 and fp['new_fits']==0)
check('connected_full_range_polyline_attested',bool(fp.get('connected_line_qa')),fp.get('connected_line_qa'))
check('frozen_parent_tracked_files_unchanged',not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=R,text=True).strip())
small=[]
for i,p in enumerate(pages):
 clean=re.sub(r'HPS Gaussian.Process Resonance Search|v5.0.0 / Review Draft|\s+','',p)
 if len(clean)<100:small.append(i+1)
check('no_blank_or_nearly_blank_pages',not small,small)
report={'pdf':str(pdf.relative_to(R)),'pdf_sha256':sha(pdf),'page_count':len(pages),'source_files':len(files),'checks':checks,'passed':all(c['passed'] for c in checks),'build_note':'Tectonic emits a cached BibTeX rerun notice; TeX labels/citations and final PDF are checked independently.','scientific_status':'Draft editorial consolidation; no additional data or toys; no unconditional calibration or unblinding certification.'}
(B/'qa/final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
(B/'qa/page_text.json').write_text(json.dumps(pages,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'pages':len(pages),'checks':len(checks),'failures':[c for c in checks if not c['passed']]},indent=2))
raise SystemExit(0 if report['passed'] else 1)
