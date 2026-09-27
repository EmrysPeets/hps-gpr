from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile
B=Path(__file__).resolve().parents[1];O=B.parents[1]/'output/pdf'/B.name;O.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((B/'qa/validation.json').read_text());p=json.loads((B/'qa/portable_build.json').read_text());visual=json.loads((B/'qa/visual_review.json').read_text())
assert v['passed'] and p['passed'] and visual['passed']
assert sha(B/'qa/build/main.pdf')==v['pdf_sha256']==visual['pdf_sha256']
name='HPS_GPR_Analysis_Note_v5p0p4_Figure2_Revision.pdf'
shutil.copy2(B/'qa/build/main.pdf',B/'pdf'/name);shutil.copy2(B/'pdf'/name,O/name)
for stem in ['figure2_clean_overview','figure2_overview_and_projections','figure2_projection_panels']:
 for ext in ['pdf','svg','png']:shutil.copy2(B/'figures'/f'{stem}.{ext}',O/f'{stem}.{ext}')
shutil.copy2(B/'figures/figure2_captioned_proof.pdf',O/'figure2_captioned_proof.pdf')
s=(B/'source/sections/01_introduction.tex').read_text();target=s.index('\\caption{Visible dark-photon phase space and full-exposure-equivalent')+len('\\caption{');depth=1;i=target
while depth:
 if s[i]=='{':depth+=1
 elif s[i]=='}':depth-=1
 i+=1
(B/'figure2_caption.tex').write_text(s[target:i-1]+'\n')
for n in ['README.md','figure2_caption.tex']:shutil.copy2(B/n,O/n)
files=[]
for folder in ['source','figures','derived','inputs','scripts','provenance','pdf']:
 for f in (B/folder).rglob('*'):
  if not f.is_file() or any(x in f.parts for x in ['__pycache__','vendor']) or f.name=='.DS_Store':continue
  if folder=='inputs' and f.suffix=='.pdf':continue
  files.append(f)
files += [B/'README.md',B/'figure2_caption.tex']
for n in ['validation.json','portable_build.json','visual_review.json','projection_run.log','figures_run.log','build_run.log']:files.append(B/'qa'/n)
manifest=B/'MANIFEST.sha256';manifest.write_text(''.join(sha(f)+'  '+f.relative_to(B).as_posix()+'\n' for f in sorted(files)))
archive=O/'HPS_GPR_v5p0p4_Figure2_Source_and_Results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(files+[manifest]):z.write(f,B.name+'/'+f.relative_to(B).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for line in manifest.read_text().splitlines():
  checksum,rel=line.split('  ',1);assert hashlib.sha256(z.read(B.name+'/'+rel)).hexdigest()==checksum
shutil.copy2(manifest,O/manifest.name)
summary=dict(pdf=name,pages=v['page_count'],pdf_sha256=sha(O/name),figure2_page=v['figure2_page'],numerical_document_checks=len(v['checks']),all_checks_passed=True,portable_build=p,source_archive=archive.name,archive_sha256=sha(archive),packaged_files=len(files),created_utc=datetime.now(timezone.utc).isoformat())
(O/'DELIVERY.json').write_text(json.dumps(summary,indent=2)+'\n')
(O/'SHA256SUMS.txt').write_text(''.join(sha(f)+'  '+f.name+'\n' for f in sorted(O.iterdir()) if f.suffix in ['.pdf','.png','.svg','.zip']))
print(json.dumps(dict(output=str(O),pages=v['page_count'],files=len(files),archive_MiB=archive.stat().st_size/1024**2,pdf_sha256=summary['pdf_sha256']),indent=2))
