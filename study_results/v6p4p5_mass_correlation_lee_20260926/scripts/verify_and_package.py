from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
import fitz
import numpy as np
B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
OUT=ROOT/'output/pdf'/B.name
STEM='HPS_GPR_v6p4p5_Mass_Correlation_LEE'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    qa=json.loads((B/'qa/pdf_validation.json').read_text())
    assert qa['passed'] and qa['visual_review_passed']
    assert json.loads((B/'qa/numerical_validation.json').read_text())['passed']
    tmp=Path(tempfile.mkdtemp(prefix='v645_portable_',dir='/private/tmp'))/'study'
    shutil.copytree(B,tmp,ignore=shutil.ignore_patterns('__pycache__','.DS_Store'))
    env=os.environ.copy();env.update(STUDY_PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/private/tmp/v645_mpl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (B/'qa/portable_rebuild.log').open('w') as out:
        subprocess.run(['bash','rebuild.sh'],cwd=tmp,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    nc=na=0
    for p in sorted((B/'results').glob('*')):
        q=tmp/p.relative_to(B)
        if p.suffix=='.npz':
            with np.load(p) as a,np.load(q) as b:
                assert set(a.files)==set(b.files)
                for k in a.files:assert np.array_equal(a[k],b[k]);na+=1
        else:assert sha(p)==sha(q);nc+=1
    for n in ('mass_correlation','correlation_global_tails'):
        assert sha(B/f'figures/{n}.png')==sha(tmp/f'figures/{n}.png')
    with fitz.open(B/'pdf/report.pdf') as a,fitz.open(tmp/'pdf/report.pdf') as b:
        assert [p.get_text() for p in a]==[p.get_text() for p in b]
    (B/'qa/portable_rebuild.json').write_text(json.dumps(dict(passed=True,detached_directory=str(tmp),identical_CSVs=nc,identical_arrays=na,identical_figure_PNGs=2,identical_PDF_text_pages=12),indent=2)+'\n')
    old=ROOT/'study_results/v6p4p4_calibrated_local_global_20260925'
    h=json.loads((B/'provenance/parent_v644_hashes.json').read_text())
    for p,v in h.items():assert sha(old/p)==v,p
    old_pdf=ROOT/'output/pdf/v6p4p4_calibrated_local_global_20260925/HPS_GPR_v6p4p4_Calibrated_Local_Global.pdf'
    assert sha(old_pdf)==sha(B/'provenance/parent_v644_report.pdf')
    (B/'qa/parent_preservation.json').write_text(json.dumps(dict(passed=True,unchanged_parent_files=len(h),delivered_parent_PDF_unchanged=True),indent=2)+'\n')
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store' and p!=B/'MANIFEST.sha256']
    (B/'MANIFEST.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(B))+'\n' for p in files))
    files.append(B/'MANIFEST.sha256');OUT.mkdir(parents=True,exist_ok=True)
    pdf=OUT/(STEM+'.pdf');shutil.copy2(B/'pdf/report.pdf',pdf)
    deliver=[pdf]
    for name in ('correlation_global_summary.csv','correlation_global_curves.csv','resolution_counts.csv'):
        shutil.copy2(B/'results'/name,OUT/name);deliver.append(OUT/name)
    archive=OUT/(STEM+'_Source.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,(Path(B.name)/p.relative_to(B)).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in (B/'MANIFEST.sha256').read_text().splitlines():
            h,p=line.split('  ',1);assert hashlib.sha256(z.read(B.name+'/'+p)).hexdigest()==h,p
    deliver.append(archive)
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in deliver))
    print(json.dumps(dict(output=str(OUT),files_in_source=len(files),parent_files_preserved=len(json.loads((B/'provenance/parent_v644_hashes.json').read_text())),pdf_sha256=sha(pdf),source_sha256=sha(archive),portable_rebuild_passed=True),indent=2))


if __name__=='__main__':main()
