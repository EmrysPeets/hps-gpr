"""Verify an isolated document rebuild and package the reviewed source/results."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile,zipfile
import fitz

B=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pdf=B/'output/pdf/APEX_Initial_Studies.pdf'
qa=json.loads((B/'qa/validation.json').read_text())
visual=json.loads((B/'qa/visual_review.json').read_text())
assert qa['status']=='passed' and qa['pdf_sha256']==sha(pdf)
assert visual['status']=='passed' and visual['pdf_sha256']==sha(pdf)

with tempfile.TemporaryDirectory(prefix='apex_standalone_') as temp:
    t=Path(temp)
    for folder in ['source','figures']:shutil.copytree(B/folder,t/folder)
    run=subprocess.run(['tectonic','main.tex'],cwd=t/'source',capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    a=fitz.open(pdf);b=fitz.open(t/'source/main.pdf');assert len(a)==len(b)
    pages=[]
    for i,(p,q) in enumerate(zip(a,b)):
        same_text=p.get_text()==q.get_text()
        same_pixels=p.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).samples==q.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).samples
        assert same_text and same_pixels,(i,same_text,same_pixels)
        pages.append(dict(page=i+1,text_identical=same_text,pixels_identical=same_pixels))
    (B/'qa/portable_build.json').write_text(json.dumps(dict(status='passed',pdf_sha256=sha(pdf),pages=pages,
       command='tectonic main.tex',dependencies='Only bundled source and figures plus LaTeX runtime resources'),indent=2))

def included(p):
    r=p.relative_to(B).as_posix()
    return (p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in r
       and not r.startswith(('qa/rendered/','qa/reference_previews/','provenance/references/'))
       and r not in ['MANIFEST.sha256','output/pdf/main.pdf','output/pdf/main.log'])
files=sorted(p for p in B.rglob('*') if included(p))
manifest=''.join(f'{sha(p)}  {p.relative_to(B).as_posix()}\n' for p in files)
(B/'MANIFEST.sha256').write_text(manifest)
archive=B/'output/pdf/APEX_Initial_Studies_LaTeX_Data.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files+[B/'MANIFEST.sha256']:z.write(p,Path('apex_initial_studies')/p.relative_to(B))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for line in manifest.splitlines():
        expected,rel=line.split('  ',1)
        assert hashlib.sha256(z.read('apex_initial_studies/'+rel)).hexdigest()==expected,rel
print(json.dumps(dict(pdf=str(pdf),archive=str(archive),files=len(files)+1,
    archive_bytes=archive.stat().st_size,pdf_sha256=sha(pdf),archive_sha256=sha(archive),
    isolated_build='All eight pages text- and pixel-identical'),indent=2))
