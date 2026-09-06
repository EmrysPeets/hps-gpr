#!/usr/bin/env python3
"""Bind this isolated side study and its two final PDFs by content hash."""
from pathlib import Path
import argparse,csv,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
manifest=HERE/'MANIFEST.csv'
if args.verify:
    rows=list(csv.DictReader(manifest.open()))
    for row in rows:
        p=ROOT/row['path'];assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],row['path']
    print(json.dumps(dict(passed=True,files=len(rows),manifest_sha256=sha(manifest)),indent=2))
else:
    if manifest.exists():raise RuntimeError('Already sealed. Verify or create a new derivative.')
    assert json.loads((HERE/'qa/numerical_validation.json').read_text())['passed']
    visual=json.loads((HERE/'qa/visual_review.json').read_text());assert visual['passed']
    build=json.loads((HERE/'provenance/report_build.json').read_text())
    for p,h in build['pdfs'].items():assert sha(ROOT/p)==h and visual['pdf_sha256'][p]==h
    files=[p for p in HERE.rglob('*') if p.is_file() and 'mpl_cache' not in p.parts and '__pycache__' not in p.parts
        and p.name!='MANIFEST.csv']
    files+=[ROOT/p for p in build['pdfs']]
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(passed=True,files=len(rows),bytes=sum(r['bytes'] for r in rows),manifest_sha256=sha(manifest)),indent=2))
