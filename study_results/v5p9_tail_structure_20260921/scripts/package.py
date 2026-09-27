"""Package the reviewed report, executable source, inputs, tables, figures and QA."""
from pathlib import Path
import hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
OUT=B.parents[1]/'output/pdf/v5p9_tail_structure_20260921'
OUT.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((B/'qa/validation.json').read_text())
artifact=json.loads((B/'qa/artifact_validation.json').read_text())
visual=json.loads((B/'qa/visual_review.json').read_text())
assert validation['passed'] and artifact['passed'] and visual['passed']
assert visual['pdf_sha256']==sha(B/'pdf/HPS_GPR_v5p9_Signal_Tail_Study.pdf')
files=[]
for p in sorted(B.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(B)
    if '__pycache__' in p.parts or rel.name in ('MANIFEST.json','SHA256SUMS.txt'):continue
    if rel.parts[0]=='source' and p.suffix!='.tex':continue
    if rel.parts[0]=='qa' and (len(rel.parts)>1 and rel.parts[1]=='rendered' or p.suffix in ('.png','.pdf')):continue
    files.append(p)
manifest=dict(version='5.9',study='Gaussian core with broader exterior tails',
    pdf_sha256=sha(B/'pdf/HPS_GPR_v5p9_Signal_Tail_Study.pdf'),
    observed_fits=5915,numerical_checks=validation['total_checks'],pages=artifact['pages'],
    conditional_scope='Fixed-state pointwise asymptotic CLs and local p0; no global calibration or coverage claim.',
    files=[dict(path=str(p.relative_to(B)),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
(B/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(B/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
files += [B/'MANIFEST.json',B/'SHA256SUMS.txt']
archive=OUT/'HPS_GPR_v5p9_Source_and_Data.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:z.write(p,str(Path(B.name)/p.relative_to(B)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for record in manifest['files']:
        assert hashlib.sha256(z.read(str(Path(B.name)/record['path']))).hexdigest()==record['sha256']
plots=OUT/'HPS_GPR_v5p9_Plots.zip'
with zipfile.ZipFile(plots,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted((B/'figures').glob('*')):z.write(p,p.name)
for source,name in [(B/'pdf/HPS_GPR_v5p9_Signal_Tail_Study.pdf','HPS_GPR_v5p9_Signal_Tail_Study.pdf'),
                    (B/'derived/scans.csv','HPS_GPR_v5p9_Observed_Limits_and_Local_pvalues.csv'),
                    (B/'README.md','README.md'),(B/'MANIFEST.json','MANIFEST.json')]:
    shutil.copy2(source,OUT/name)
(OUT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
print(json.dumps(dict(output=str(OUT),source_archive_sha256=sha(archive),archived_files=len(files),pdf_sha256=manifest['pdf_sha256']),indent=2))
