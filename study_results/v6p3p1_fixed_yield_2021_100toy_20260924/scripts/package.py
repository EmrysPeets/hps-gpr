"""Package only completed, independently checked, visually reviewed artifacts."""
from pathlib import Path
import hashlib,json,shutil,zipfile,datetime
B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
OUT=ROOT/'output/pdf'/B.name
PDF_NAME='HPS_GPR_v6p3p1_Fixed_Yield_2021_100toy.pdf'
ZIP_NAME='HPS_GPR_v6p3p1_Reproducible_Study.zip'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def excluded(p):
    return p.name in ('MANIFEST.sha256','run.lock') or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp')

def main():
    assert read(B/'qa/independent_validation.json')['status']=='passed'
    assert read(B/'qa/resume.json')['passed']
    qa=read(B/'qa/final_qa.json')
    assert qa['status']=='passed' and qa['pdf_sha256']==sha(B/'pdf/report.pdf')
    summary=read(B/'results/summary.json')
    assert summary['pilot_rows']==2000 and summary['evaluation_rows']==8000
    assert summary['failed_rows']==0
    for name,expected in summary['source_hashes'].items():assert sha(B/name)==expected
    status=dict(status='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pilot_rows=2000,evaluation_rows=8000,planned_cells=80,completed_cells=80,failed_rows=0,
        scope='Conditional fixed-yield bias and nominal signed profile-set containment',
        pdf=PDF_NAME,archive=ZIP_NAME)
    (B/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    shutil.copy2(B/'pdf/report.pdf',B/'pdf'/PDF_NAME)
    paths=sorted(p for p in B.rglob('*') if p.is_file() and not excluded(p))
    (B/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in paths))
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(B/'pdf'/PDF_NAME,OUT/PDF_NAME)
    for name in ('README.md','MANIFEST.sha256','protocol.json','pilot_reference.json'):
        shutil.copy2(B/name,OUT/name)
    with zipfile.ZipFile(OUT/ZIP_NAME,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[B/'MANIFEST.sha256']:
            z.write(p,arcname=str(Path(B.name)/p.relative_to(B)))
    with zipfile.ZipFile(OUT/ZIP_NAME) as z:
        assert z.testzip() is None
        for line in (B/'MANIFEST.sha256').read_text().splitlines():
            digest,rel=line.split('  ',1)
            assert hashlib.sha256(z.read(str(Path(B.name)/rel))).hexdigest()==digest
    assert sha(OUT/PDF_NAME)==sha(B/'pdf/report.pdf')
    release=dict(status='passed',study=B.name,manifest_files=len(paths),archive_files=len(paths)+1,
        pdf_sha256=sha(OUT/PDF_NAME),archive_sha256=sha(OUT/ZIP_NAME),archive_bytes=(OUT/ZIP_NAME).stat().st_size,
        mirrored_pdf_identical=True,all_archive_members_match_manifest=True)
    (OUT/'release_validation.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps(release,indent=2))

if __name__=='__main__':main()
