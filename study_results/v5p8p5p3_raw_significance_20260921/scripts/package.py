from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
B=Path(__file__).resolve().parents[1];R=B.parents[1];O=R/'output/pdf'/B.name;O.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['numerical_and_semantic_validation','portable_rebuild','parent_immutability','visual_validation']:assert json.loads((B/'qa'/f'{name}.json').read_text())['passed'],name
summary={'version':'5.8.5.3','status':'completed','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pages':24,'integrated_figures':14,'new_figure_assets':10,'new_integrated_figures':6,'new_fits':0,'new_toys':0,'scope_mass_coordinates':1310,'reference_centering_applied_to_new_local_curves':False,'local_definition':'Z=max(r,0); p=norm.sf(Z)','global_definition':'Tail of raw maximum under fixed nominal GP source; displayed separately','blind_half_width_sigma':2.25,'parent_files_unchanged':331,'validation_checks':69,'portable_rebuild_passed':True,'new_local_plot_page':3,'new_global_plot_page':5,'expanded_response_pages':[12,13,14,15],'report_sha256':sha(B/'source/report.pdf')}
(B/'DELIVERY.json').write_text(json.dumps(summary,indent=2)+'\n')
files=[]
for p in sorted(B.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.log','.aux','.out']:continue
 rel=p.relative_to(B)
 if rel.name=='SHA256SUMS.txt':continue
 if rel.parts[0]=='qa' and (rel.name.startswith('page-') or rel.name.startswith('contact-')):continue
 files.append(p)
(B/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(B).as_posix()}\n' for p in files));files.append(B/'SHA256SUMS.txt')
for a,b in [(B/'source/report.pdf','HPS_GPR_v5p8p5p3_Unshifted_Significance_Report.pdf'),(B/'figures/raw_local_overview.pdf','HPS_GPR_v5p8p5p3_Raw_Local_p_Z.pdf'),(B/'figures/raw_local_overview.png','HPS_GPR_v5p8p5p3_Raw_Local_p_Z.png'),(B/'README.md','README.md'),(B/'DELIVERY.json','DELIVERY.json')]:shutil.copy2(a,O/b)
archive=O/'HPS_GPR_v5p8p5p3_Source_and_Data.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,B.name+'/'+p.relative_to(B).as_posix())
with zipfile.ZipFile(O/'HPS_GPR_v5p8p5p3_New_Plots.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted((B/'figures').iterdir()):
  if p.is_file() and (p.name.startswith('raw_') or p.name.startswith('response_diagnostics_')):z.write(p,'figures/'+p.name)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 prefix=B.name+'/';lines=z.read(prefix+'SHA256SUMS.txt').decode().splitlines()
 for line in lines:
  digest,path=line.split('  ',1);assert hashlib.sha256(z.read(prefix+path)).hexdigest()==digest,path
validation={'passed':True,'verified_files':len(lines),'source_archive_bytes':archive.stat().st_size,'source_archive_sha256':sha(archive)}
(O/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
(O/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(O.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
print(json.dumps({'output':str(O),'archive_MB':archive.stat().st_size/1e6,'files_verified':len(lines)},indent=2))
