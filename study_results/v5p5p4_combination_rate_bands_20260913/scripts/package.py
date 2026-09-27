"""Verify portable TeX assets, then manifest and package the inspected result."""
from pathlib import Path
import json,hashlib,shutil,tempfile,subprocess,zipfile
import fitz
B=Path(__file__).resolve().parents[1]
stem='HPS_GPR_v5p5p4_Combination_Rate_Uncertainties'
with tempfile.TemporaryDirectory(prefix='hps-v554-portable-') as td:
 t=Path(td)
 for folder,pattern in [('source','*.tex'),('derived','*.tex'),('figures','*.pdf')]:
  (t/folder).mkdir()
  for f in (B/folder).glob(pattern):shutil.copy2(f,t/folder/f.name)
 r=subprocess.run(['tectonic','main.tex'],cwd=t/'source',capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 a=fitz.open(B/f'pdf/{stem}.pdf');b=fitz.open(t/'source/main.pdf')
 assert len(a)==len(b)==13
 checks=[dict(page=i+1,identical_text=p.get_text()==q.get_text(),identical_render=p.get_pixmap().samples==q.get_pixmap().samples) for i,(p,q) in enumerate(zip(a,b))]
 assert all(c['identical_text'] and c['identical_render'] for c in checks)
 (B/'qa/portable_build.json').write_text(json.dumps(dict(passed=True,mode='Isolated source, derived TeX tables and vector figures only',checks=checks),indent=2)+'\n')
for name in ['document_validation','extracted_combination_validation','rate_band_validation','rate_scan_validation','visual_review']:
 assert json.loads((B/f'qa/{name}.json').read_text())['passed']
assert json.loads((B/'qa/visual_review.json').read_text())['pdf_sha256']==hashlib.sha256((B/f'pdf/{stem}.pdf').read_bytes()).hexdigest()
files=sorted(p for p in B.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(B/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
out=B.parents[1]/'output/pdf'/B.name;out.mkdir(parents=True,exist_ok=True)
pdf=B/f'pdf/{stem}.pdf';shutil.copy2(pdf,out/pdf.name)
archive=out/f'{stem}_Source_and_Results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files+[B/'MANIFEST.sha256']:z.write(p,arcname=str(Path(B.name)/p.relative_to(B)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for p in files:assert hashlib.sha256(z.read(str(Path(B.name)/p.relative_to(B)))).hexdigest()==sha(p)
assert sha(out/pdf.name)==sha(pdf)
delivery=dict(pdf=str(out/pdf.name),pdf_sha256=sha(pdf),archive=str(archive),archive_sha256=sha(archive),manifest_files=len(files),archive_bytes=archive.stat().st_size)
(out/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
print(json.dumps(delivery,indent=2))
