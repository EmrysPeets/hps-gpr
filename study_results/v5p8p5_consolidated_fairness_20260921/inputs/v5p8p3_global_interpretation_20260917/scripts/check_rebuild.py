"""Isolated saved-input rebuild: compare numerical ledgers, PDF text and pixels."""
from pathlib import Path
import os,sys,shutil,tempfile,subprocess,json,time
from pypdf import PdfReader
from PIL import Image,ImageChops
B=Path(__file__).resolve().parents[1];start=time.monotonic()
science=os.environ.get('STUDY_SCIENCE_PYTHON',sys.executable)
tmp=Path(tempfile.mkdtemp(prefix='hps-v583-rebuild-'));b=tmp/'study'
shutil.copytree(B,b,ignore=shutil.ignore_patterns('final-*.png','page-*.png','contact.png','*.log','*.aux','__pycache__','SHA256SUMS.txt'))
shutil.rmtree(b/'qa')
(b/'qa').mkdir()
for p in (b/'figures').glob('*'):p.unlink()
with (B/'qa/rebuild.log').open('w') as log:
    for command in [[science,'scripts/resolution_audit.py'],[science,'scripts/summary_figures.py'],[science,'scripts/validate_study.py'],['tectonic','-X','compile','source/report.tex']]:
        subprocess.run(command,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
    old=PdfReader(B/'source/report.pdf');new=PdfReader(b/'source/report.pdf')
    assert len(old.pages)==len(new.pages)==9
    assert [p.extract_text() for p in old.pages]==[p.extract_text() for p in new.pages]
    for p in (B/'results').glob('*.csv'):assert p.read_bytes()==(b/'results'/p.name).read_bytes(),p.name
    subprocess.run(['pdftoppm','-r','115','-png',str(b/'source/report.pdf'),str(tmp/'page')],stdout=log,stderr=subprocess.STDOUT,check=True)
    for i in range(1,10):
        x=Image.open(B/f'qa/final-{i}.png').convert('RGB');y=Image.open(tmp/f'page-{i}.png').convert('RGB')
        assert x.size==y.size and ImageChops.difference(x,y).getbbox() is None,i
result={'passed':True,'pages':9,'numerical_CSVs_identical':True,'PDF_text_identical':True,'all_rendered_pixels_identical':True,'rebuild_uses_only_bundled_inputs':True,'elapsed_seconds':time.monotonic()-start}
(B/'qa/portable_rebuild.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));shutil.rmtree(tmp)
