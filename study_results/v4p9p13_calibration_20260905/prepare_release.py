#!/usr/bin/env python3
"""Prepare an explicit scientific artifact allowlist; never mutate Git."""
from pathlib import Path
import argparse
import hashlib
import json
import csv
import gzip

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PDF=ROOT/'output/pdf/v4p9p13_calibration_20260905/HPS_GPR_Analysis_Note_v4p9p13_calibrated_backgrounds.pdf'
HISTORY=[
 'v4p9_2021_threshold_support_qualification_20260817/README.md',
 'v4p9_2021_threshold_support_qualification_20260817/study_spec.json',
 'v4p9_2021_threshold_support_qualification_20260817/build_fsig_anchor_truth.py',
 'v4p9p1_2021_background_validation_consolidation_20260817/README.md',
 'v4p9p1_2021_background_validation_consolidation_20260817/derived/consolidated_pull_moments_90cl.csv',
 'v4p9p1_2021_background_validation_consolidation_20260817/reference/v4p6_full100/study_spec.json',
 'v4p9p1_2021_background_validation_consolidation_20260817/build_continuation_toys.py',
 'v4p9p5_2021_gp_support_edge_optimization_20260820/run_support_scan.py',
 'v4p9p5_2021_gp_support_edge_optimization_20260820/STUDY_PROTOCOL.md',
 'v4p9p5_2021_gp_support_edge_optimization_20260820/STEERING_AMENDMENT_20260820.md',
 'v4p9p5_2021_gp_support_edge_optimization_20260820/reference/v4p9_fsig_anchor_fit_summary.json',
 'v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/reverse_injection/run_reverse_injection.py',
]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--allow-partial',action='store_true')
 parser.add_argument('--output-dir',type=Path,default=Path('/private/tmp/hps-v4p9p13-calibration-release'))
 args=parser.parse_args()
 summary=json.loads((HERE/'summary/calibration_summary.json').read_text())
 if not summary['complete_grid'] and not args.allow_partial:
  raise RuntimeError('A complete release requires all456 coordinates; partial publication must be explicit')
 qa_path=HERE/'qa/final_qa.json'
 qa=json.loads(qa_path.read_text())
 if not qa.get('passed') or qa.get('pdf_sha256')!=sha(PDF):
  raise RuntimeError('Final rendered/semantic QA is missing or belongs to a different PDF')
 numerical=json.loads((HERE/'summary/numerical_qa.json').read_text())
 if not numerical['source_contracts_passed'] or not numerical['all_collected_numerical_checks_passed']:
  raise RuntimeError('Numerical/source audit has unresolved failures; qualify them before release')
 checks=[];files=set()
 def verify(path,expected,kind,publish=True):
  p=Path(path);actual=sha(p)
  checks.append(dict(path=str(p),kind=kind,expected_sha256=expected,sha256=actual,passed=actual==expected))
  if actual!=expected:raise RuntimeError(f'Frozen identity changed: {p}')
  if publish:files.add(p)
 candidate_test=json.loads((HERE/'qa/candidate_contract_test.json').read_text())
 if not candidate_test['passed'] or not candidate_test['source_unchanged_during_test']:
  raise RuntimeError('Candidate contract integration checks did not pass')
 for name,expected in candidate_test['source_sha256'].items():verify(ROOT/name,expected,'candidate_contract_test_source')
 build=json.loads((HERE/'provenance/note_build.json').read_text())
 verify(PDF,build['pdf_sha256'],'typeset_pdf')
 if build['completed_points']!=summary['completed_points']:raise RuntimeError('Note and collected grid counts differ')
 for name,expected in build['inputs'].items():verify(ROOT/name,expected,'note_input')
 import build_note as note
 collected=note.pd.read_csv(HERE/'summary/observed_calibrated_limits.csv')
 _,chunked_inputs,chunked=note.chunked_execution_section(collected)
 recorded_chunked=build.get('chunked_execution')
 if chunked['status']=='audited':
  if recorded_chunked!=chunked:raise RuntimeError('Chunked execution QA/selection differs from the typeset note')
  for path in chunked_inputs:
   name=str(path.relative_to(ROOT));expected=chunked['input_sha256'][name]
   if build['inputs'].get(name)!=expected:raise RuntimeError('Chunked execution dependency missing from note provenance: '+name)
   verify(path,expected,'chunked_execution_input')
 elif recorded_chunked is not None and recorded_chunked!=chunked:
  raise RuntimeError('Note chunked-execution status differs from the stable selected collection')
 reverse=build.get('reverse_truth')
 if not isinstance(reverse,dict):raise RuntimeError('Rebuild the note with an explicit auxiliary follow-up status')
 if reverse.get('status')=='not_evaluated':
  if reverse.get('directory') is not None:raise RuntimeError('Unevaluated auxiliary result names an output directory')
 elif reverse.get('status')=='completed_auxiliary_diagnostic':
  # Revalidate only the directory explicitly recorded by the note; never search for a latest run.
  directory=(ROOT/reverse['directory']).resolve()
  _,auxiliary_inputs,confirmed=note.reverse_truth_section(directory,collected)
  if confirmed!=reverse:raise RuntimeError('Auxiliary identities differ from the typeset note')
  for path in auxiliary_inputs:
   name=str(path.relative_to(ROOT))
   verify(path,reverse['input_sha256'][name],'selected_auxiliary_input')
  for path,expected in reverse['external_input_sha256'].items():verify(path,expected,'auxiliary_external_native_input',publish=False)
  for path in directory.rglob('*'):
   if not path.is_file() or path.suffix not in ('.json','.csv','.gz','.log'):continue
   name=str(path.relative_to(ROOT))
   if name not in reverse['input_sha256']:raise RuntimeError(f'Auxiliary output was not captured by note provenance: {path}')
   verify(path,reverse['input_sha256'][name],'selected_auxiliary_output')
 else:raise RuntimeError('Unsupported auxiliary follow-up status')
 for name,script in [('limit_plot_provenance.json','make_figures.py'),('validation_plot_provenance.json','make_validation_figures.py'),('truth_plot_provenance.json','make_truth_figure.py')]:
  metadata=json.loads((HERE/'figures'/name).read_text())
  verify(metadata.get('source',metadata.get('input')),metadata['source_sha256'],'plot_input')
  verify(HERE/script,metadata['script_sha256'],'plot_script')
  outputs=metadata['output_sha256'] if 'output_sha256' in metadata else metadata['outputs']
  for path,expected in outputs.items():
   p=Path(path);verify(p if p.is_absolute() else HERE/p,expected,'rendered_figure')
 contract=json.loads((HERE/'derived/contract.json').read_text())
 for name,expected in contract['hashes'].items():verify(ROOT/name,expected,'production_source')
 for entry in summary['contracts']:
  for name,expected in entry['contract'].get('sampling_hashes',{}).items():verify(ROOT/name,expected,'sampling_derivative_source')
 companion=json.loads((HERE/'provenance/additional_runtime_hashes.json').read_text())
 for row in companion['checks']:verify(row['path'],row['reference_sha256'],'companion_runtime')
 parent=json.loads((ROOT/'study_results/v4p9p13_background_profiling_20260905/observed/derived/summary.json').read_text())
 for row in parent['native_histograms'].values():verify(row['path'],row['sha256'],'external_native_histogram',publish=False)
 for name in HISTORY:files.add(ROOT/'study_results'/name)
 for p in HERE.glob('*.py'):files.add(p)
 for name in ['PROTOCOL.md','REFINEMENT_PROTOCOL.md','CHUNKED_REFINEMENT_PROTOCOL.md','CHUNKED_RESOURCE_PROTOCOL.md','CHUNKED_RESOURCE8_PROTOCOL.md','SCALAR_REFERENCE_RECOVERY_PROTOCOL.md','REFERENCE_METADATA_FINALIZATION.md','REVERSE_TRUTH_71_PROTOCOL.md','SAMPLING_REFINEMENT_DESIGN.md','README.md','history_review.md','release_review.md','calibration_sections.tex',
              'gp_refit_pilot.json','gp_lowrank_pilot.json','sampler_validation.json','production.log','collection_inputs.json','NEXT_STEPS.md']:
  files.add(HERE/name)
 for directory,pattern in [('provenance','**/*'),('summary','*'),('figures','*'),('note','*.tex'),('qa/final_pages','*.png')]:
  files.update(p for p in (HERE/directory).glob(pattern) if p.is_file())
 # Preserve the complete audit losslessly without a blob above GitHub's limit.
 large_qa=HERE/'summary/numerical_qa.json'
 compressed=large_qa.with_suffix('.json.gz')
 with compressed.open('wb') as output:
  with gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0,compresslevel=9) as stream:
   with large_qa.open('rb') as source:
    for block in iter(lambda:source.read(1024*1024),b''):stream.write(block)
 restored=hashlib.sha256()
 with gzip.open(compressed,'rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):restored.update(block)
 if restored.hexdigest()!=sha(large_qa):raise RuntimeError('Compressed numerical audit is not lossless')
 compression_record=HERE/'summary/numerical_qa_compression.json'
 compression_record.write_text(json.dumps(dict(passed=True,format='gzip',mtime=0,
  original_path=str(large_qa.relative_to(ROOT)),original_bytes=large_qa.stat().st_size,original_sha256=sha(large_qa),
  compressed_path=str(compressed.relative_to(ROOT)),compressed_bytes=compressed.stat().st_size,compressed_sha256=sha(compressed),
  instruction='Decompress with gzip -dk numerical_qa.json.gz to restore the exact audit consumed by the note release.'),indent=2)+'\n')
 files.discard(large_qa);files.update((compressed,compression_record))
 roots={Path(entry['directory']) for entry in summary['contracts']}
 for directory in roots:
  if HERE not in directory.parents:raise RuntimeError(f'Checkpoint root leaves this study: {directory}')
  files.add(directory/'contract.json')
  for name in ('selection.json','batch_summary.json','run.log','companion_run.log','prelaunch_resource_check.json'):
   if (directory/name).is_file():files.add(directory/name)
  for pattern in ('batch_summary_before_resume*.json','resume*_command.json'):
   files.update(directory.glob(pattern))
  for name in ('result.json','unverified_result.json','validation_summary.csv','validation_toys.csv.gz','FAILURE.txt',
               'point_plan.json','proposal_plan.json','memory_check.json','pre_generation_numerical_qa.json',
               'chunked_equivalence_checks.json','model_chunk_ledger.json','scalar_reference_recovery.json',
               'failure_numerical_qa.json','DEFERRED.json'):
   files.update(directory.glob('*/m*/'+name))
 audit=HERE/'qa/release_input_audit.json'
 audit.write_text(json.dumps(dict(passed=all(c['passed'] for c in checks),checks=checks,
  external_native_root_files_included=False,archived_2015_container_includes_input_histogram=True,
  completed_points=summary['completed_points'],chunked_selected_points=chunked['selected_points'],
  chunked_resource_override_points=chunked.get('resource_override_points',0),
  chunked_resource_override_points_by_version=chunked.get('resource_override_points_by_version',{}),
  chunked_resource_pure_qa_by_version=chunked.get('resource_pure_qa_by_version',{}),
  chunked_reference_recovery_count=chunked.get('reference_recovery_count',0),
  chunked_reference_recovery_coordinate_count=chunked.get('reference_recovery_coordinate_count',0),
  chunked_reference_recovery_coordinates=chunked.get('reference_recovery_coordinates',[]),
  chunked_postprocessing_finalization_coordinate_count=chunked.get('postprocessing_finalization_coordinate_count',0),
  chunked_postprocessing_finalizations=chunked.get('postprocessing_finalizations',[]),
  chunked_max_memory_limit_gib=chunked.get('max_memory_limit_gib',0),
  chunked_supplemental_qa_sha256=chunked.get('supplemental_qa_sha256')),indent=2)+'\n')
 files.update([PDF,qa_path,audit,HERE/'qa/candidate_contract_test.json'])
 if (HERE/'chunked_v1/individual_queue.json').is_file():files.add(HERE/'chunked_v1/individual_queue.json')
 if (HERE/'qa/chunked_execution_contract_test.json').is_file():files.add(HERE/'qa/chunked_execution_contract_test.json')
 if (HERE/'qa/chunked_resource_contract_test.json').is_file():files.add(HERE/'qa/chunked_resource_contract_test.json')
 if (HERE/'qa/chunked_resource8_contract_test.json').is_file():files.add(HERE/'qa/chunked_resource8_contract_test.json')
 if (HERE/'qa/scalar_reference_recovery_contract_test.json').is_file():files.add(HERE/'qa/scalar_reference_recovery_contract_test.json')
 missing=[str(p) for p in files if not p.is_file()]
 if missing:raise RuntimeError(f'Missing release artifacts: {missing}')
 forbidden=[str(p) for p in files if p.suffix in ('.npz','.pyc') or '__pycache__' in p.parts or p.name=='PROGRESS.md']
 if forbidden:raise RuntimeError(f'Forbidden release artifacts: {forbidden}')
 oversized=[str(p) for p in files if p.stat().st_size>=100*1024*1024]
 if oversized:raise RuntimeError(f'GitHub blob size limit exceeded: {oversized}')
 paths=sorted(files,key=lambda p:str(p.relative_to(ROOT)))
 manifest=HERE/'RELEASE_MANIFEST.csv'
 with manifest.open('w',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['path','bytes','sha256'])
  for p in paths:writer.writerow([str(p.relative_to(ROOT)),p.stat().st_size,sha(p)])
 paths.append(manifest)
 args.output_dir.mkdir(parents=True,exist_ok=True)
 names=sorted(str(p.relative_to(ROOT)) for p in paths)
 (args.output_dir/'files.json').write_text(json.dumps(names,indent=2)+'\n')
 (args.output_dir/'paths.nul').write_bytes(b''.join(name.encode()+b'\0' for name in names))
 print(json.dumps(dict(file_count=len(names),bytes=sum(p.stat().st_size for p in paths),
  manifest=str(manifest),allowlist=str(args.output_dir/'files.json'),git_mutated=False)))

if __name__=='__main__':main()
