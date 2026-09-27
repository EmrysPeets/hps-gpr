"""Isolated saved-results rebuild; no likelihood refits or new ensembles."""
from pathlib import Path
import os,sys,tempfile,shutil,subprocess,json,time
from pypdf import PdfReader
from PIL import Image,ImageChops
B=Path(__file__).resolve().parents[1];start=time.monotonic();science=os.environ.get('STUDY_SCIENCE_PYTHON',sys.executable);tmp=Path(tempfile.mkdtemp(prefix='hps-v584-rebuild-'));b=tmp/'study'
shutil.copytree(B,b,ignore=shutil.ignore_patterns('final-*.png','page-*.png','*contact.png','*.log','*.aux','__pycache__','SHA256SUMS.txt'))
for p in (b/'figures').glob('*'):p.unlink()
with (B/'qa/rebuild.log').open('w') as log:
 for cmd in [[science,'scripts/make_figures.py'],[science,'scripts/validate_outputs.py'],['tectonic','-X','compile','source/report.tex']]:subprocess.run(cmd,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
 a=PdfReader(B/'source/report.pdf');c=PdfReader(b/'source/report.pdf');assert len(a.pages)==len(c.pages)==11
 assert [p.extract_text() for p in a.pages]==[p.extract_text() for p in c.pages]
 for p in (B/'results').glob('*.csv'):assert p.read_bytes()==(b/'results'/p.name).read_bytes(),p.name
 subprocess.run(['pdftoppm','-r','110','-png',str(b/'source/report.pdf'),str(tmp/'page')],stdout=log,stderr=subprocess.STDOUT,check=True)
 for i in range(1,12):
  x=Image.open(B/f'qa/final-{i:02d}.png').convert('RGB');y=Image.open(tmp/f'page-{i:02d}.png').convert('RGB');assert x.size==y.size and ImageChops.difference(x,y).getbbox() is None,i
q=dict(passed=True,pages=11,CSV_results_identical=True,PDF_text_identical=True,all_rendered_pixels_identical=True,standalone_saved_results_rebuild=True,elapsed_seconds=time.monotonic()-start)
(B/'qa/portable_rebuild.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2));shutil.rmtree(tmp)
