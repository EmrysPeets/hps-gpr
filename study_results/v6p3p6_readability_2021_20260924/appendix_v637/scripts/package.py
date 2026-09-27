#!/usr/bin/env python3
"""Package locally validated v6.3.7 numerical evidence and standalone report."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def release(destination):
 report=B/'pdf/HPS_GPR_v6p3p7_Template_Window_Study.pdf'
 for name in ['toy_validation','statistics_review','calibration_mismatch','fit_replay','report_visual_qa','portable_qa','resume_fix_reproduction']:
  q=json.loads((B/'qa'/f'{name}.json').read_text());assert q['passed'],name
  if name=='report_visual_qa':assert q.get('report_sha256',q.get('pdf_sha256'))==sha(report),'Final visual QA PDF identity'
 for line in (B/'provenance/input_manifest.sha256').read_text().splitlines():
  h,path=line.split('  ',1);assert sha(B/path)==h,path
 (B/'status.json').write_text(json.dumps({'status':'complete_validated','version':'6.3.7','unique_toy_fit_rows':58800,'deterministic_profile_limit_calculations':30,'observed_data_fits':0,'report_sha256':sha(report)},indent=2)+'\n')
 manifest=B/'MANIFEST.sha256'
 files=sorted(p for p in B.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts and 'rendered' not in p.relative_to(B).parts and p.suffix not in ('.pyc','.tmp'))
 manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
 destination.mkdir(parents=True,exist_ok=True)
 archive=destination/'HPS_GPR_v6p3p7_Reproducible_Study.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[manifest]:z.write(p,str(Path(B.name)/p.relative_to(B)))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in files:assert hashlib.sha256(z.read(str(Path(B.name)/p.relative_to(B)))).hexdigest()==sha(p)
 final=destination/report.name;shutil.copy2(report,final)
 info={'pdf':str(final),'pdf_sha256':sha(final),'archive':str(archive),'archive_sha256':sha(archive),'files':len(files)+1,'archive_bytes':archive.stat().st_size}
 (destination/'release.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);release(p.parse_args().destination.resolve())
