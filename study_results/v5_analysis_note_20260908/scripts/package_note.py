#!/usr/bin/env python3
"""Package the reviewed draft; requires a current successful visual/semantic review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, zipfile

B = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

pdf = B / 'qa/build/main.pdf'
validation = json.loads((B / 'qa/final_validation.json').read_text())
visual = json.loads((B / 'qa/visual_review.json').read_text())
assert validation['passed'] and visual['passed']
assert validation['pdf_sha256'] == visual['pdf_sha256'] == sha(pdf)
(B / 'pdf').mkdir(exist_ok=True)
released = B / 'pdf/HPS_GPR_Analysis_Note_v5p0p0_Unblinding_Review_Draft.pdf'
shutil.copy2(pdf, released)
files = []
for folder in ['source', 'figures', 'derived', 'scripts', 'provenance']:
    files.extend(p for p in (B / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for name in ['README.md', 'HANDOFF.md', 'editorial/REQUEST_CHECKLIST.md',
             'editorial/EDITORIAL_REVIEW.md', 'editorial/source_claims.json',
             'editorial/figure_provenance.json', 'editorial/figure_inventory.md',
             'qa/final_validation.json', 'qa/visual_review.json', 'qa/portable_build.json', 'qa/page_text.json']:
    path = B / name
    if path.exists():
        files.append(path)
files.append(released)
files = sorted(set(files))
manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'version': '5.0.0 review draft', 'pages': validation['page_count'],
    'pdf_sha256': sha(released),
    'scope': 'Editorial consolidation of frozen released results; no new data, fits or toys.',
    'rebuild': 'bash scripts/build_note.sh; TeX source and included figure assets are bundled.',
    'numerical_regeneration': 'Requires the external frozen repository studies listed in the provenance ledgers.',
    'files': [{'path': str(p.relative_to(B)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
}
mp = B / 'MANIFEST.json'
mp.write_text(json.dumps(manifest, indent=2) + '\n')
archive = B / 'HPS_GPR_v5p0p0_Review_Draft_Source.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in files + [mp]:
        z.write(path, str(Path(B.name) / path.relative_to(B)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
(B / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in [released, archive, mp]))
print(json.dumps({'pdf': str(released), 'pages': validation['page_count'], 'files': len(files),
                  'pdf_sha256': sha(released), 'archive_bytes': archive.stat().st_size}, indent=2))
