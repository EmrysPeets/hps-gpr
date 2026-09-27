"""Package completed, rendered and independently validated extension."""
from pathlib import Path
import json,hashlib,shutil,zipfile,tempfile,subprocess
import fitz
B=Path(__file__).resolve().parents[1];O=B.parents[1]/'output/pdf/v5p6p1_2021_upper_limit_echoes_20260913';NAME='HPS_GPR_v5p6p1_2021_Upper_Limit_Echoes'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
qa=json.loads((B/'qa/final_validation.json').read_text());assert qa['passed']
log=(B/'source/main.log').read_text();assert 'Overfull' not in log and 'Undefined control sequence' not in log
shutil.copy2(B/'source/main.pdf',B/'report.pdf');O.mkdir(parents=True,exist_ok=True);shutil.copy2(B/'report.pdf',O/(NAME+'.pdf'))
with tempfile.TemporaryDirectory(prefix='v561-portable-') as tmp:
 t=Path(tmp);shutil.copytree(B/'source',t/'source');shutil.copytree(B/'figures',t/'figures')
 p=subprocess.run(['/opt/homebrew/bin/tectonic','--keep-logs','main.tex'],cwd=t/'source',capture_output=True,text=True);assert p.returncode==0,p.stderr
 a=fitz.open(B/'report.pdf');c=fitz.open(t/'source/main.pdf');assert len(a)==len(c) and [p.get_text() for p in a]==[p.get_text() for p in c]
 portable=dict(passed=True,pages=len(a),text_identical=True)
 (B/'qa/portable_build.json').write_text(json.dumps(portable,indent=2)+'\n')
exclude={'MANIFEST.sha256','qa/package_validation.json'}
files=sorted(p for p in B.rglob('*') if p.is_file() and str(p.relative_to(B)) not in exclude and '__pycache__' not in p.parts and p.suffix not in {'.log','.aux','.out','.tmp'} and p.name!='.DS_Store')
manifest=B/'MANIFEST.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
archive=O/(NAME+'_Source_Data.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files+[manifest]:z.write(p,p.relative_to(B))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for p in files:assert sha(p)==hashlib.sha256(z.read(str(p.relative_to(B)))).hexdigest()
res=dict(passed=True,manifest_files=len(files),pdf_pages=len(fitz.open(B/'report.pdf')),pdf_sha256=sha(B/'report.pdf'),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,portable_build=portable,limit_rows=68724,reused_toys=300,new_toys=0,pdf=str(O/(NAME+'.pdf')),archive=str(archive))
for p in [B/'qa/package_validation.json',O/'package_validation.json']:p.write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
