#!/usr/bin/env python3
"""Rebuild report/plots from saved results in isolation; compare all pixels/text."""
from pathlib import Path
import sys,tempfile,shutil,subprocess,json,hashlib,time
from pypdf import PdfReader
from PIL import Image,ImageChops
b=Path(__file__).resolve().parents[1];t=Path(tempfile.mkdtemp(prefix='hps_v582_rebuild_'));c=t/'study';start=time.monotonic()
shutil.copytree(b,c,ignore=shutil.ignore_patterns('final*.png','page-*.png','contact.png','*.aux','*.log','__pycache__','SHA256SUMS.txt'))
with (b/'qa/portable_build.log').open('w') as log:
 for cmd in [[sys.executable,'scripts/make_figures.py'],['tectonic','-X','compile','source/report.tex']]:subprocess.run(cmd,cwd=c,stdout=log,stderr=subprocess.STDOUT,check=True)
 a=PdfReader(b/'source/report.pdf');r=PdfReader(c/'source/report.pdf');assert len(a.pages)==len(r.pages)==6
 assert [p.extract_text() for p in a.pages]==[p.extract_text() for p in r.pages]
 subprocess.run(['pdftoppm','-r','125','-png',str(c/'source/report.pdf'),str(t/'new')],stdout=log,stderr=subprocess.STDOUT,check=True)
 for i in range(1,7):
  ia=Image.open(b/f'qa/final-{i}.png').convert('RGB');ib=Image.open(t/f'new-{i}.png').convert('RGB');assert ia.size==ib.size and ImageChops.difference(ia,ib).getbbox() is None,i
result={'passed':True,'pages':6,'isolated_saved_results_rebuild':True,'identical_extracted_text':True,'all_rendered_pixels_identical':True,'PDF_SHA256':hashlib.sha256((b/'source/report.pdf').read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start}
(b/'qa/portable_build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));shutil.rmtree(t)
