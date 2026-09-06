#!/usr/bin/env python3
"""Seal the verified extraction derivative without touching its parents."""
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    product=json.loads((HERE/'qa/product_validation.json').read_text())
    visual=json.loads((HERE/'qa/visual_review.json').read_text())
    audit=json.loads((HERE/'review/independent_final_audit.json').read_text())
    build=json.loads((HERE/'provenance/report_build.json').read_text())
    assert product['passed'] and visual['passed'] and audit['passed']
    pdf=Path(build['pdf']);assert sha(pdf)==product['pdf_sha256']==visual['pdf_sha256']==build['pdf_sha256']
    assert all(sha(ROOT/p)==h for p,h in product['rendered_pages_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in audit['input_sha256'].items())
    excluded={HERE/'MANIFEST.csv',HERE/'qa/manifest_verification.json'}
    files=sorted([p for p in HERE.rglob('*') if p.is_file() and p not in excluded and '__pycache__' not in p.parts]+[pdf])
    with (HERE/'MANIFEST.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
        for p in files:w.writerow(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    rows=list(csv.DictReader((HERE/'MANIFEST.csv').open()))
    assert all(sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==int(r['bytes']) for r in rows)
    result=dict(passed=True,files=len(files),bytes=sum(p.stat().st_size for p in files),
        manifest_sha256=sha(HERE/'MANIFEST.csv'),pdf_sha256=sha(pdf),pages=product['pages'])
    (HERE/'qa/manifest_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
