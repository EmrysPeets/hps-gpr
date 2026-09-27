"""Release the validated standalone study, data, inputs and reproducible sources."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
NAME='HPS_GPR_v6p3p5_Unified_2021_Procedure'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def release(destination):
    required=('independent_validation.json','observed_independent_validation.json','resume_cache.json','observed_resume_cache.json','window_replay.json','report_qa.json','portable_qa.json')
    for name in required:
        q=json.loads((B/'qa'/name).read_text());assert q.get('passed',q.get('status')=='passed'),name
    destination.mkdir(parents=True,exist_ok=True)
    report=B/'pdf'/f'{NAME}.pdf';assert report.exists()
    for row in json.loads((B/'provenance/prior_report_hashes.json').read_text()):
        p=B/row['snapshot'];assert sha(p)==row['sha256']
    (B/'status.json').write_text(json.dumps(dict(status='complete_validated',scientific_scope='Conditional model studies; physical exclusion and discovery not certified',report=str(report.relative_to(B)),report_sha256=sha(report)),indent=2)+'\n')
    manifest=B/'MANIFEST.sha256'
    files=sorted(p for p in B.rglob('*') if p.is_file() and p!=manifest and p.name not in ('run.lock','STOP') and not p.name.endswith(('.pyc','.tmp')) and '__pycache__' not in p.parts and 'rendered' not in p.relative_to(B).parts)
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
    archive=destination/'HPS_GPR_v6p3p5_Reproducible_Study.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,arcname=str(Path(B.name)/p.relative_to(B)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(str(Path(B.name)/p.relative_to(B)))).hexdigest()==sha(p)
    final=destination/report.name;shutil.copy2(report,final)
    info=dict(pdf=str(final),pdf_sha256=sha(final),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=len(files)+1,manifest_sha256=sha(manifest))
    (destination/'release.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args();release(a.destination.resolve())
