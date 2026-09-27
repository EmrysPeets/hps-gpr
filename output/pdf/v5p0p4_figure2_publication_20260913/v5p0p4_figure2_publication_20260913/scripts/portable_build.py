"""Build using only the portable source, figure assets and LaTeX tables."""
from pathlib import Path
import shutil,subprocess,json,hashlib
import fitz
B=Path(__file__).resolve().parents[1];W=B/'qa/portable_rebuild';W.mkdir(exist_ok=True)
for folder in ['source','figures']:
 shutil.copytree(B/folder,W/folder,dirs_exist_ok=True)
(W/'derived').mkdir(exist_ok=True)
for f in (B/'derived').glob('*.tex'):shutil.copy2(f,W/'derived'/f.name)
with (B/'qa/portable_run.log').open('w') as log:subprocess.run(['tectonic','-C','--keep-logs','main.tex'],cwd=W/'source',stdout=log,stderr=subprocess.STDOUT,check=True)
a=fitz.open(B/'qa/build/main.pdf');b=fitz.open(W/'source/main.pdf');same_text=len(a)==len(b) and all(x.get_text()==y.get_text() for x,y in zip(a,b));same_pixels=len(a)==len(b) and all(x.get_pixmap(matrix=fitz.Matrix(.5,.5),alpha=False).samples==y.get_pixmap(matrix=fitz.Matrix(.5,.5),alpha=False).samples for x,y in zip(a,b));r=dict(passed=same_text and same_pixels,pages=len(a),all_page_text_identical=same_text,all_page_renderings_identical=same_pixels,portable_pdf_sha256=hashlib.sha256((W/'source/main.pdf').read_bytes()).hexdigest());(B/'qa/portable_build.json').write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['passed']
