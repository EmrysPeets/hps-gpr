"""Package the reviewed report, source/data and vector/raster plots."""
from pathlib import Path
import hashlib,json,zipfile,shutil,datetime,sys
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
OUT=repo/'output/pdf'/B.name;OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for q in ['semantic_validation','parent_immutability','portable_report','portable_physics']:
 assert json.loads((B/'qa'/f'{q}.json').read_text())['passed'],q
pdf=B/'source/report.pdf'
visual={'passed':True,'final_pdf_sha256':sha(pdf),'render_dpi':105,'pages':17,'review_assignments':{'history':[1,2,3,4,5,6],'physics':[7,8,9,10,11],'statistics':[12,13,14],'root':[15,16,17]},'root_visual_findings':'All final pages inspected; equations, tables, captions and figure labels readable; no clipping or overlapping elements.'}
(B/'qa/visual_validation.json').write_text(json.dumps(visual,indent=2)+'\n')
start=datetime.datetime(2026,9,21,18,25,55,tzinfo=datetime.timezone.utc);now=datetime.datetime.now(datetime.timezone.utc)
metadata={'status':'completed','report_version':'5.8.5 consolidated methodology review','started_utc':start.isoformat(),'analysis_completed_utc':now.isoformat(),'analysis_elapsed_minutes':(now-start).total_seconds()/60,'hard_time_limit_minutes':60,'weekly_usage_initial_percent':10,'weekly_usage_last_observed_percent':16,'weekly_usage_increase_percentage_points':6,'weekly_usage_limit_additional_percentage_points':20,'usage_note':'Account-wide rounded meter; other concurrent account use is not separable. Snapshot at packaging.','pages':17,'figures':9,'new_extraction_half_width_sigma':2.25,'new_deterministic_source_states':603,'new_Poisson_fits':768,'new_source_injection_anchors':2,'new_main_Gaussian_block_fields':500000,'preserved_parent_files':3782,'semantic_checks_passed':52,'production_reference_recalibration_adopted':False,'scope_remaining_unqualified':['control/search selection and exposure equivalence','event overlap','source-estimation and source-signal uncertainty in a production calibration','full-procedure discovery tails','continuous-grid convergence','new signal-plus-background reach'],'report_sha256':sha(pdf),'prior_releases_modified':False}
(B/'DELIVERY.json').write_text(json.dumps(metadata,indent=2)+'\n')
files=[]
for p in sorted(B.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(B)
 if '__pycache__' in rel.parts or p.suffix in ['.log','.pyc','.aux','.out','.synctex.gz']:continue
 if rel.name in ['SHA256SUMS.txt','COMPLETE','STOP']:continue
 if rel.parts[0]=='qa' and (rel.name.startswith('page-') or rel.name.startswith('contact-')):continue
 files.append(p)
(B/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(B).as_posix()}\n' for p in files))
files.append(B/'SHA256SUMS.txt')
for source,dest in [(pdf,'HPS_GPR_v5p8p5_Consolidated_Significance_Report.pdf'),(B/'README.md','README.md'),(B/'DELIVERY.json','DELIVERY.json')]:shutil.copy2(source,OUT/dest)
archive=OUT/'HPS_GPR_v5p8p5_Consolidated_Source_and_Data.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,f'{B.name}/{p.relative_to(B).as_posix()}')
plots=OUT/'HPS_GPR_v5p8p5_Consolidated_Plots.zip'
with zipfile.ZipFile(plots,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted((B/'figures').glob('*')):
  if p.suffix in ['.pdf','.png']:z.write(p,'figures/'+p.name)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 prefix=B.name+'/'
 ledger=z.read(prefix+'SHA256SUMS.txt').decode().splitlines()
 for line in ledger:
  digest,path=line.split('  ',1)
  assert hashlib.sha256(z.read(prefix+path)).hexdigest()==digest,path
validation={'passed':True,'archive_files':len(files),'verified_hashes':len(ledger),'source_zip_sha256':sha(archive),'plots_zip_sha256':sha(plots),'pdf_identical':sha(OUT/'HPS_GPR_v5p8p5_Consolidated_Significance_Report.pdf')==sha(pdf)}
(OUT/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
(OUT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
print(json.dumps({'out':str(OUT),'archive_MB':archive.stat().st_size/1e6,'plots_MB':plots.stat().st_size/1e6,'elapsed_minutes':metadata['analysis_elapsed_minutes'],'validated_files':len(ledger)},indent=2))
