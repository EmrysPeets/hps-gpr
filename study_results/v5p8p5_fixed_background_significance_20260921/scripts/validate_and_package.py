"""Run after rendered pages have been inspected and visual_review.json written."""
from pathlib import Path
import hashlib,json,shutil,tempfile,subprocess,zipfile,datetime,re
from pypdf import PdfReader
import pdfplumber
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
hashf=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pdf=B/'source/report.pdf';reader=PdfReader(pdf)
text='\n'.join(p.extract_text() for p in reader.pages)
(B/'qa/report_text.txt').write_text(text)
checks={'no_tex_overfull':'Overfull' not in (B/'source/report.log').read_text(),'no_unresolved_reference':'??' not in text,
        'numerical_qa_passed':json.loads((B/'qa/independent_numerical.json').read_text())['passed'],
        'scan_qa_passed':json.loads((B/'qa/scan_validation.json').read_text())['passed'],
        'earlier_report_unchanged':all(hashf(repo/p['path'])==p['sha256'] for p in json.loads((B/'qa/parent_manifest.json').read_text()))}
vis=json.loads((B/'qa/visual_review.json').read_text())
checks['all_final_pages_inspected']=vis['pdf_sha256']==hashf(pdf) and vis['pages']==list(range(1,len(reader.pages)+1)) and vis['passed']
with pdfplumber.open(pdf) as doc:
    checks['no_body_text_in_footer']=all(all(w['text']==str(i+1) for w in pg.extract_words() if w['top']>735) for i,pg in enumerate(doc.pages))
tmp=Path(tempfile.mkdtemp(prefix='hps_fixed_background_report_'))
for directory in ['source','figures']:shutil.copytree(B/directory,tmp/directory)
(tmp/'source/report.pdf').unlink()
run=subprocess.run(['tectonic','-X','compile','source/report.tex'],cwd=tmp,text=True,capture_output=True,timeout=90)
checks['portable_compile']=run.returncode==0
checks['portable_text_identical']=run.returncode==0 and '\n'.join(p.extract_text() for p in PdfReader(tmp/'source/report.pdf').pages)==text
qa={'passed':all(checks.values()),'checks':checks,'pages':len(reader.pages),'pdf_sha256':hashf(pdf),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(B/'qa/package_validation.json').write_text(json.dumps(qa,indent=2)+'\n');assert qa['passed'],qa
out=repo/'output/pdf'/B.name;out.mkdir(parents=True,exist_ok=True)
shutil.copy2(pdf,out/'HPS_GPR_External_Statistician_Fixed_Background_Review.pdf')
shutil.copy2(B/'README.md',out/'README.md')
shutil.copy2(B/'results/significance_curves.csv',out/'Fixed_Background_Pvalues_All_Masses.csv')
shutil.copy2(B/'results/peaks.json',out/'Fixed_Background_Pvalue_Peaks.json')
exclude={'SHA256SUMS.txt','.DS_Store'}
files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.name not in exclude and p.suffix not in ['.log','.pyc']]
(B/'SHA256SUMS.txt').write_text(''.join(f'{hashf(p)}  {p.relative_to(B)}\n' for p in files));files.append(B/'SHA256SUMS.txt')
with zipfile.ZipFile(out/'HPS_GPR_Fixed_Background_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,f'{B.name}/{p.relative_to(B)}')
with zipfile.ZipFile(out/'HPS_GPR_Fixed_Background_Source.zip') as z:assert z.testzip() is None
(out/'SHA256SUMS.txt').write_text(''.join(f'{hashf(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.name!='SHA256SUMS.txt'))
print(json.dumps(qa,indent=2))
