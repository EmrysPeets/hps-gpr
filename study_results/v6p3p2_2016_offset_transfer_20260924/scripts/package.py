"""Merge the unchanged parent report and the validated appendix; portable release."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
NAME='HPS_GPR_v6p3p2_2021_Study_with_2016_Offset_Appendix'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def merge():
    from pypdf import PdfReader,PdfWriter
    parent=B/'inputs/parent_2021_report.pdf';appendix=B/'pdf/appendix.pdf'
    expected=json.loads((B/'provenance/parent_release_hashes.json').read_text())
    assert sha(parent)==expected['inputs/parent_2021_report.pdf']
    first=PdfReader(parent);second=PdfReader(appendix);assert len(first.pages)==9
    writer=PdfWriter();writer.append(first);writer.append(second)
    writer.add_metadata({'/Title':'HPS-GPR v6.3.2: 2021 fixed-yield study with 2016 offset-transfer appendix','/Author':'Emrys Peets'})
    target=B/'pdf'/f'{NAME}.pdf'
    with target.open('wb') as stream:writer.write(stream)
    merged=PdfReader(target)
    assert len(merged.pages)==len(first.pages)+len(second.pages)
    for i,page in enumerate(first.pages):
        assert page.get_contents().get_data()==merged.pages[i].get_contents().get_data()
        assert page.extract_text()==merged.pages[i].extract_text()
    print(json.dumps(dict(pdf=str(target),pages=len(merged.pages),parent_pages_preserved=9,sha256=sha(target))))
def release(destination):
    for name in ('independent_validation.json','report_qa.json','resume_qa.json','portable_qa.json'):
        q=json.loads((B/'qa'/name).read_text());assert q.get('passed',q.get('status')=='passed'),name
    destination.mkdir(parents=True,exist_ok=True)
    manifest=B/'MANIFEST.sha256'
    files=sorted(p for p in B.rglob('*') if p.is_file() and p!=manifest and p.name not in ('run.lock','STOP')
        and not p.name.endswith(('.pyc','.tmp')) and '__pycache__' not in p.parts
        and not any(part in ('rendered','parent_rendered') for part in p.relative_to(B).parts))
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
    archive=destination/'HPS_GPR_v6p3p2_Reproducible_Study.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,arcname=str(Path(B.name)/p.relative_to(B)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(str(Path(B.name)/p.relative_to(B)))).hexdigest()==sha(p)
    report=destination/f'{NAME}.pdf';shutil.copy2(B/'pdf'/report.name,report)
    info=dict(pdf=str(report),pdf_sha256=sha(report),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        files=len(files)+1,manifest_sha256=sha(manifest))
    (destination/'release.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['merge','release']);ap.add_argument('--destination',type=Path)
    a=ap.parse_args()
    if a.action=='merge':merge()
    else:
        assert a.destination is not None,'Supply explicit release destination'
        release(a.destination.resolve())
