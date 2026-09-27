"""Verify that editorial changes leave every original scientific input/result unchanged."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[]
 for line in (B/'provenance/v635_original_manifest.sha256').read_text().splitlines():
  digest,name=line.split('  ',1)
  if name.startswith(('inputs/','results/')) or name in ('protocol.json','pilot_reference.json','calibration_freeze.json') or (name.startswith('scripts/') and name not in ('scripts/make_report.py','scripts/package.py')):
   p=B/name;rows.append({'path':name,'sha256':digest,'unchanged':p.is_file() and sha(p)==digest})
 failures=[r['path'] for r in rows if not r['unchanged']]
 out={'passed':not failures,'files_checked':len(rows),'failures':failures,'scope':'Original inputs, numerical results, frozen calibration/protocol and scientific computation scripts; no refitting','checks':rows}
 (B/'qa/v636_editorial_integrity.json').write_text(json.dumps(out,indent=2)+'\n')
 assert not failures,failures
 print(f'PASS: {len(rows)} original scientific files unchanged')
if __name__=='__main__':main()
