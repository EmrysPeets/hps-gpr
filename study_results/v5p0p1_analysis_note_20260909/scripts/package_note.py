#!/usr/bin/env python3
"""Package only the hash-verified, visually reviewed version 5.0.1."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pdf=B/'qa/build/main.pdf';validation=json.loads((B/'qa/final_validation.json').read_text());visual=json.loads((B/'qa/visual_review.json').read_text());portable=json.loads((B/'qa/portable_build.json').read_text())
assert validation['passed'] and visual['passed'] and portable['passed']
assert sha(pdf)==validation['pdf_sha256']==visual['pdf_sha256']==portable['source_pdf_sha256']
released=B/'pdf/HPS_GPR_Analysis_Note_v5p0p1_Unblinding_Review_Draft.pdf';released.parent.mkdir(exist_ok=True);shutil.copy2(pdf,released)
files=[]
for folder in ['source','figures','derived','scripts','editorial','provenance']:
 files.extend(p for p in(B/folder).rglob('*')if p.is_file() and '__pycache__'not in p.parts)
for p in(B/'qa').rglob('*.json'):
 if not any(x in p.parts for x in ['build','portable']):files.append(p)
for name in ['README.md','HANDOFF.md','qa/build/main.log','qa/build/main.bbl','qa/build/main.aux','qa/build/main.toc','qa/build/main.apc']:
 p=B/name
 if p.exists():files.append(p)
files.append(released);files=sorted(set(files));manifest={'version':'5.0.1 review draft','created_utc':datetime.now(timezone.utc).isoformat(),'base_commit':'3c557e39101c0f73040dc02c46b6555e85d5b4ca','pages':validation['page_count'],'pdf_sha256':sha(released),'scope':'Editorial and figure derivative; no new HPS observations, inference fits or toys. Frozen observed results preserved.','files':[{'path':str(p.relative_to(B)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files]}
mp=B/'MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');archive=B/'HPS_GPR_v5p0p1_Review_Draft_Source.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for p in files+[mp]:z.write(p,str(Path(B.name)/p.relative_to(B)))
with zipfile.ZipFile(archive)as z:assert z.testzip()is None
(B/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n'for p in[released,archive,mp]))
print(json.dumps({'pages':validation['page_count'],'files':len(files),'pdf_sha256':sha(released),'archive_MiB':archive.stat().st_size/2**20},indent=2))
