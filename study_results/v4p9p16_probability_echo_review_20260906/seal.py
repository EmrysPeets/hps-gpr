#!/usr/bin/env python3
"""Seal the reviewed derivative; never rewrite a parent artifact."""
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 product=json.loads((HERE/'qa/product_validation.json').read_text());visual=json.loads((HERE/'qa/visual_review.json').read_text());build=json.loads((HERE/'provenance/report_build.json').read_text());review=json.loads((HERE/'review/final_interpretation_bindings.json').read_text())
 assert product['passed'] and visual['passed'] and review['accepted']
 pdf=ROOT/build['pdf'];assert sha(pdf)==product['pdf_sha256']==visual['pdf_sha256']==build['pdf_sha256']
 for source in (product['rendered_pages_sha256'],review['sha256'],build['input_sha256']):
  assert all(sha(ROOT/p)==h for p,h in source.items())
 excluded={HERE/'MANIFEST.csv',HERE/'qa/manifest_verification.json'}
 files=sorted([p for p in HERE.rglob('*') if p.is_file() and p not in excluded and '__pycache__' not in p.parts]+[pdf])
 with (HERE/'MANIFEST.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
  for p in files:w.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 rows=list(csv.DictReader((HERE/'MANIFEST.csv').open()));assert all(sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==int(r['bytes']) for r in rows)
 result={'passed':True,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'pages':product['pages'],'manifest_sha256':sha(HERE/'MANIFEST.csv'),'pdf_sha256':sha(pdf)}
 (HERE/'qa/manifest_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
