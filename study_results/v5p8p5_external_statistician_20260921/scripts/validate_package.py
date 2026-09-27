from pathlib import Path
import tempfile,subprocess,shutil,hashlib,json,zipfile,re,datetime
import pandas as pd
from pypdf import PdfReader
import pdfplumber
B=Path(__file__).resolve().parents[1];repo=B.parents[1];out=repo/'output/pdf'/B.name;out.mkdir(parents=True,exist_ok=True)
hashf=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=PdfReader(B/'source/report.pdf');text='\n'.join(p.extract_text() for p in r.pages);(B/'qa/report_text.txt').write_text(text)
checks={'three_pages':len(r.pages)==3,'no_tex_overfull':'Overfull' not in (B/'source/report.log').read_text(),'no_unresolved_reference':'??' not in text}
compact=re.sub(r'\s+','',text)
for token in ['3.583/2.060','0.812','77.48','29.96','82','0.595','1.346','8.88','2.25']:checks['contains_'+token]=token in compact
p=pd.read_csv(B/'results/92MeV_composition.csv');checks['82_percent_information']=abs(p[p.scope==2021].iloc[0].null_information_fraction-.822577)<1e-6
sources=json.loads((B/'results/provenance.json').read_text());checks['parent_inputs_unchanged']=all(hashf(repo/x['path'])==x['sha256'] for x in sources)
with pdfplumber.open(B/'source/report.pdf') as f:
 checks['no_body_text_in_footer']=all(all(w['text']==str(i+1) for w in pg.extract_words() if w['top']>735) for i,pg in enumerate(f.pages))
tmp=Path(tempfile.mkdtemp(prefix='hps_external_stat_report_'))
for folder in ['source','figures']:shutil.copytree(B/folder,tmp/folder)
(tmp/'source/report.pdf').unlink()
run=subprocess.run(['tectonic','-X','compile','source/report.tex'],cwd=tmp,text=True,capture_output=True,timeout=90)
checks['portable_compile']=run.returncode==0
checks['portable_text_identical']='\n'.join(x.extract_text() for x in PdfReader(tmp/'source/report.pdf').pages)==text
checks={k:bool(v) for k,v in checks.items()}
qa={'passed':all(checks.values()),'checks':checks,'visual_pages_inspected':[1,2,3],'render_dpi':125,'pdf_sha256':hashf(B/'source/report.pdf'),'portable_directory':str(tmp),'new_fits':0,'fixed_mask_sigma':2.25,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(B/'qa/validation.json').write_text(json.dumps(qa,indent=2)+'\n');assert qa['passed'],qa
for src,dest in [(B/'source/report.pdf','HPS_GPR_External_Statistician_Review.pdf'),(B/'README.md','README.md')]:shutil.copy2(src,out/dest)
files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix not in ['.log','.pyc'] and p.name!='SHA256SUMS.txt']
(B/'SHA256SUMS.txt').write_text(''.join(f'{hashf(p)}  {p.relative_to(B)}\n' for p in files));files.append(B/'SHA256SUMS.txt')
with zipfile.ZipFile(out/'HPS_GPR_External_Statistician_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,f'{B.name}/{p.relative_to(B)}')
with zipfile.ZipFile(out/'HPS_GPR_External_Statistician_Source.zip') as z:assert z.testzip() is None
(out/'SHA256SUMS.txt').write_text(''.join(f'{hashf(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.name!='SHA256SUMS.txt'))
print(json.dumps({'passed':qa['passed'],'checks':len(checks),'pages':len(r.pages),'output':str(out)},indent=2))
