"""Package the checked study with input/output checksums and portable sources."""
from pathlib import Path
import hashlib,json,shutil,zipfile,sys
B=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    for name in ('validation.json','independent_validation.json','report_visual_qa.json'):
        q=json.loads((B/'qa'/name).read_text())
        assert q['passed'],name
    pdf=B/'pdf/HPS_GPR_v6p2_MC_Injection_Recovery.pdf'
    build=json.loads((B/'qa/report_build.json').read_text())
    assert sha(pdf)==build['pdf_sha256']
    visual=json.loads((B/'qa/report_visual_qa.json').read_text())
    assert sha(pdf)==visual['pdf_sha256'] and visual['pages']==build['pages']
    for name,digest in build['input_sha256'].items():assert sha(B/name)==digest
    protocol=json.loads((B/'protocol.json').read_text())
    protocol['status']='complete_conditional_injection_recovery'
    (B/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    excluded={'__pycache__','matplotlib','fontcache','fontconfig-cache','pilot'}
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and not excluded.intersection(p.relative_to(B).parts)
           and p.name not in ('MANIFEST.sha256','STOP') and p.suffix not in ('.pyc','.tmp','.log')]
    manifest=''.join(sha(p)+'  '+p.relative_to(B).as_posix()+'\n' for p in files)
    (B/'MANIFEST.sha256').write_text(manifest)
    args.output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(pdf,args.output/pdf.name)
    for name in ('summary.csv','paired.csv','toys.csv'):
        shutil.copy2(B/'results'/name,args.output/('v6p2_'+name))
    shutil.copy2(B/'README.md',args.output/'README.md')
    archive=args.output/'HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,B/'MANIFEST.sha256']:z.write(p,B.name+'/'+p.relative_to(B).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:
            assert hashlib.sha256(z.read(B.name+'/'+p.relative_to(B).as_posix())).hexdigest()==sha(p)
    delivery=dict(passed=True,manifest_files=len(files),archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive),pdf_sha256=sha(pdf),pdf_copy_equal=sha(args.output/pdf.name)==sha(pdf),
        report_pages=build['pages'],all_archive_member_hashes_verified=True)
    (args.output/'delivery_verification.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(delivery,indent=2))
if __name__=='__main__':main()
