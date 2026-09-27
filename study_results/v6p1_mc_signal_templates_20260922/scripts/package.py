"""Create a portable study archive or verify its per-file SHA-256 manifest."""
from pathlib import Path
import argparse,hashlib,json,tarfile,zipfile
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():
    return [p for p in sorted(B.rglob('*')) if p.is_file() and
      '__pycache__' not in p.parts and 'delivery' not in p.parts and
      p.name not in ('MANIFEST.sha256','.DS_Store') and
      not (p.parent.name=='source' and p.suffix in ('.aux','.log','.out','.pdf'))]
def verify():
    errors=[];n=0
    for line in (B/'MANIFEST.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1);p=B/relative;n+=1
        if not p.is_file() or sha(p)!=expected:errors.append(relative)
    result=dict(verified=not errors,files=n,failures=errors,study=str(B))
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)
def package(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    items=files();manifest=B/'MANIFEST.sha256'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(B).as_posix()}\n' for p in items))
    items.append(manifest)
    target=out/'HPS_GPR_v6p1_MC_Signal_Templates_source.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in items:z.write(p,B.name+'/'+p.relative_to(B).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Archive CRC failure')
    tarpath=out/'remote_study_payload.tar.gz'
    with tarfile.open(tarpath,'w:gz') as t:
        for p in items:t.add(p,arcname=p.relative_to(B).as_posix(),recursive=False)
    receipt=dict(files=len(items),archive=target.name,sha256=sha(target),bytes=target.stat().st_size,
                 manifest_sha256=sha(manifest),archive_CRC_pass=True)
    (out/'package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));verify()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');a.add_argument('--output');p=a.parse_args()
    if p.verify:verify()
    elif p.output:package(p.output)
    else:a.error('Provide --verify or --output')
