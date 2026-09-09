#!/usr/bin/env python3
"""Rebuild a document-only copy, with no authoring checkout dependencies."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pdf=B/'qa/build/main.pdf'
with tempfile.TemporaryDirectory(prefix='hps-v501-portable-') as tmp:
 target=Path(tmp)/B.name;target.mkdir()
 for folder in ['source','figures','derived','scripts','editorial','provenance']:
  shutil.copytree(B/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__'))
 r=subprocess.run(['bash','scripts/build_note.sh'],cwd=target,capture_output=True,text=True)
 (B/'qa/portable_build.log').write_text(r.stdout+'\n'+r.stderr)
 if r.returncode:raise RuntimeError('Portable build failed; see qa/portable_build.log')
 rebuilt=target/'qa/build/main.pdf';a=PdfReader(pdf);b=PdfReader(rebuilt)
 at=[p.extract_text() for p in a.pages];bt=[p.extract_text() for p in b.pages]
 dims=lambda d:[(tuple(p.mediabox),p.rotation)for p in d.pages]
 checks={'same_page_count':len(a.pages)==len(b.pages),'same_page_text':at==bt,'same_dimensions_and_rotation':dims(a)==dims(b)}
 report={'passed':all(checks.values()),'source_pdf_sha256':sha(pdf),'rebuilt_pdf_sha256':sha(rebuilt),'pages':len(a.pages),'checks':checks,'method':'Temporary copy of bundled source/assets only; cached Tectonic; no external HPS study directories used. PDF byte identities may differ because of build metadata.'}
 (B/'qa/portable_build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 if not report['passed']:raise SystemExit(1)
