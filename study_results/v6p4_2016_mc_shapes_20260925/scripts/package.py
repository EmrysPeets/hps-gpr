"""Create a checksummed standalone source archive after study QA."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile

B=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args();dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
    validation=json.loads((B/'qa/numerical_validation.json').read_text())
    pdfqa=json.loads((B/'qa/pdf_review.json').read_text())
    portable=json.loads((B/'qa/portable_rebuild.json').read_text())
    assert validation['passed'] and pdfqa['passed'] and portable['passed']
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.sha256']
    lines=[f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(B).as_posix()}' for p in files]
    (B/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n')
    archive=dest/'HPS_GPR_v6p4_2016_MC_Shapes_Reproducible_Study.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[B/'MANIFEST.sha256']:
            z.write(p,Path(B.name)/p.relative_to(B))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in lines:
            sha,name=line.split('  ',1)
            assert hashlib.sha256(z.read(f'{B.name}/{name}')).hexdigest()==sha
    for p in [B/'pdf/HPS_GPR_v6p4_2016_MC_Shapes.pdf',B/'results/centers_and_shapes.csv',B/'README.md']:
        shutil.copy2(p,dest/p.name)
    report=dest/'HPS_GPR_v6p4_2016_MC_Shapes.pdf'
    info=dict(source_files=len(files),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              pdf_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),archive_crc_and_manifest_passed=True,
              numerical_checks=validation['checks'],pdf_pages=pdfqa['pages'],portable_rebuild_passed=True)
    (dest/'delivery_verification.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
