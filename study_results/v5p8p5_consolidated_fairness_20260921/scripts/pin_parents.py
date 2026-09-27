from pathlib import Path
import hashlib,json,shutil
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
parents=sorted(p for p in (repo/'study_results').glob('v5p8*') if p!=B)
entries=[]
for p in parents:
 for f in sorted(p.rglob('*')):
  if f.is_file():entries.append({'path':str(f.relative_to(repo)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
for f in [repo/'study_results/v5p0p5_analysis_note_20260916/source/main.tex',repo/'study_results/v5p0p5_analysis_note_20260916/pdf/HPS_GPR_Analysis_Note_v5p0p5.pdf']:
 entries.append({'path':str(f.relative_to(repo)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
(B/'inputs/parent_manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
print('Pinned',len(entries),'files',sum(x['bytes'] for x in entries),'bytes')
for p in parents:
 dest=B/'inputs'/p.name;dest.mkdir(parents=True,exist_ok=True)
 for folder in ['source','results','scripts','fields']:
  src=p/folder
  if src.exists():shutil.copytree(src,dest/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.log','*.pdf'))
 for name in ['README.md','protocol.json','SHA256SUMS.txt']:
  if (p/name).exists():shutil.copy2(p/name,dest/name)
for v,name in [('v5p8p2_nominal_gp_significance_20260917','response_validation'),('v5p8p3_global_interpretation_20260917','response_resolution'),('v5p8p3_global_interpretation_20260917','trials_comparison'),('v5p8p3_global_interpretation_20260917','combined_domain_effect'),('v5p8p5_mass_coherence_20260919','methods_full_domain'),('v5p8p5_mass_coherence_20260919','coherence_null')]:
 for ext in ['pdf','png']:
  src=repo/'study_results'/v/'figures'/f'{name}.{ext}'
  if src.exists():shutil.copy2(src,B/'figures'/f'archive_{name}.{ext}')
