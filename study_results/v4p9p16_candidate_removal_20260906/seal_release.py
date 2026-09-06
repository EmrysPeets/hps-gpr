#!/usr/bin/env python3
"""Seal the release only after numerical, document and visual QA pass."""
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['numerical_validation','display_validation','product_validation','visual_review']:
    assert json.loads((HERE/'qa'/f'{name}.json').read_text())['passed'],name
visual=json.loads((HERE/'qa/visual_review.json').read_text());product=json.loads((HERE/'qa/product_validation.json').read_text())
assert visual['pdf_sha256']==product['pdf_sha256']==sha(ROOT/product['pdf'])
assert (HERE/'review/FINAL_SECTION_REVIEW.md').is_file()
review=json.loads((HERE/'review/final_section_bindings.json').read_text())
assert review['accepted'] and not review['unresolved_findings']
for path,digest in review['source_sha256'].items():assert sha(HERE/path)==digest,path
exclude={HERE/'MANIFEST.csv',HERE/'qa/manifest_verification.json'}
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p not in exclude and '__pycache__' not in p.parts and '.DS_Store' not in p.name)
files.extend(sorted(p for p in (ROOT/'output/pdf'/HERE.name).glob('*') if p.is_file()))
rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
with (HERE/'MANIFEST.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
for row in rows:assert sha(ROOT/row['path'])==row['sha256']
(HERE/'qa/manifest_verification.json').write_text(json.dumps(dict(passed=True,entries=len(rows),bytes=sum(r['bytes'] for r in rows),manifest_sha256=sha(HERE/'MANIFEST.csv')),indent=2)+'\n')
print((HERE/'qa/manifest_verification.json').read_text())
