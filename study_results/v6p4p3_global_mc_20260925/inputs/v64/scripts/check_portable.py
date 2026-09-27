"""Rebuild in a temporary independent directory and compare scientific outputs."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,time
import numpy as np
import fitz

B=Path(__file__).resolve().parents[1]

def main():
    target=Path(tempfile.mkdtemp(prefix='hps_v64_portable_'))/B.name
    shutil.copytree(B,target,ignore=shutil.ignore_patterns('__pycache__','MANIFEST.sha256'))
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
               MPLCONFIGDIR=str(target.parent/'mplcache'))
    runs=[]
    for name in ['analyze.py','make_figures.py','build_report.py','validate.py']:
        start=time.monotonic()
        p=subprocess.run([sys.executable,str(target/'scripts'/name)],cwd=target,env=env,capture_output=True,text=True)
        (B/'qa'/f'portable_{name[:-3]}.log').write_text(p.stdout+p.stderr)
        runs.append(dict(script=name,exit_code=p.returncode,seconds=round(time.monotonic()-start,3)))
        print(name,p.returncode,flush=True)
        assert p.returncode==0,p.stdout+p.stderr
    tables=[]
    for path in sorted((B/'results').glob('*')):
        other=target/'results'/path.name
        same=path.read_bytes()==other.read_bytes()
        tables.append(dict(file=str(path.relative_to(B)),byte_identical=same,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert same,path
    count=0
    for path in (B/'histograms').glob('*.npz'):
        a=np.load(path);b=np.load(target/'histograms'/path.name)
        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files),path
        count+=1
    pdf='pdf/HPS_GPR_v6p4_2016_MC_Shapes.pdf'
    a=fitz.open(B/pdf);b=fitz.open(target/pdf)
    assert len(a)==len(b)==12
    assert [x.get_text() for x in a]==[x.get_text() for x in b]
    figure_count=0
    for path in (B/'figures').glob('*.png'):
        assert path.read_bytes()==(target/'figures'/path.name).read_bytes(),path
        figure_count+=1
    result=dict(passed=True,temporary_study=str(target),runs=runs,tables=tables,
                histogram_arrays_identical=count,figure_pngs_byte_identical=figure_count,
                report_pages=12,report_text_identical=True,
                note='PDF byte hashes can differ because creation timestamps and document IDs are regenerated.')
    (B/'qa/portable_rebuild.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','tables']},indent=2))

if __name__=='__main__':main()
