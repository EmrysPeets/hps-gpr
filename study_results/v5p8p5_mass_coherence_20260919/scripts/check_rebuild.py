"""Isolated numeric, figure, PDF-text and pixel reproducibility check."""
from pathlib import Path
import os,sys,tempfile,shutil,subprocess,json
from pypdf import PdfReader
from PIL import Image,ImageChops
B=Path(__file__).resolve().parents[1]
science=os.environ.get('STUDY_SCIENCE_PYTHON',sys.executable)
runlog=[]
def run(args,cwd,env):
 p=subprocess.run(args,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 runlog.append({'command':args,'returncode':p.returncode,'output':p.stdout[-2500:]})
 if p.returncode:raise RuntimeError(p.stdout)
with tempfile.TemporaryDirectory(prefix='portable-',dir=B/'qa') as td:
 t=Path(td)
 for folder in ['inputs','scripts','source']:
  shutil.copytree(B/folder,t/folder)
 shutil.copy2(B/'protocol.json',t/'protocol.json')
 for folder in ['results','fields','figures','qa']: (t/folder).mkdir()
 env=os.environ.copy();env['MPLCONFIGDIR']=str(B/'qa/mpl');env['PYTHONDONTWRITEBYTECODE']='1'
 for script in ['free_amplitude.py','stability.py','make_figures.py','make_tables.py']:
  run([science,str(t/'scripts'/script)],t,env)
 # Numeric CSVs are reproducible byte-for-byte under the recorded runtime.
 compared=[]
 for f in sorted((B/'results').glob('*.csv')):
  assert f.read_bytes()==(t/'results'/f.name).read_bytes(),f.name
  compared.append(f.name)
 run(['tectonic','-X','compile',str(t/'source/report.tex')],t,env)
 old=PdfReader(B/'source/report.pdf');new=PdfReader(t/'source/report.pdf')
 assert [p.extract_text() for p in old.pages]==[p.extract_text() for p in new.pages]
 for label,pdf in [('original',B/'source/report.pdf'),('rebuilt',t/'source/report.pdf')]:
  run(['pdftoppm','-r','90','-png',str(pdf),str(t/label)],t,env)
 old_images=sorted(t.glob('original-*.png'));new_images=sorted(t.glob('rebuilt-*.png'));assert len(old_images)==len(new_images)==len(old.pages)
 for a,b in zip(old_images,new_images):assert ImageChops.difference(Image.open(a).convert('RGB'),Image.open(b).convert('RGB')).getbbox() is None,a.name
 result=dict(complete=True,numeric_csvs_identical=compared,pdf_pages=len(old.pages),pdf_text_identical=True,all_rendered_pixels_identical=True,render_dpi=90,science_python=science)
(B/'qa/portable_rebuild.json').write_text(json.dumps(result,indent=2)+'\n');(B/'qa/portable_rebuild_log.json').write_text(json.dumps(runlog,indent=2)+'\n');print(json.dumps(result))
