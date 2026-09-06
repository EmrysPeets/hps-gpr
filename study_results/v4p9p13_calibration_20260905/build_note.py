#!/usr/bin/env python3
"""Extend the frozen v4.9.13 note with data-driven calibration tables and plots."""
from pathlib import Path
import argparse,json,shutil,subprocess,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'study_results/v4p9p13_background_profiling_20260905'
NOTE=HERE/'note';NOTE.mkdir(exist_ok=True)
OUT=ROOT/'output/pdf/v4p9p13_calibration_20260905';OUT.mkdir(parents=True,exist_ok=True)
SCOPES=[('individual_2015_full',r'2015, 100\%','2015'),('individual_2016_full',r'2016, 100\%','2016'),('individual_2021_10pct',r'2021, 10\%','2021'),('all_2015_2016_2021','All three','combined')]

def number(v,fmt='.3f'):
 return format(v,fmt) if np.isfinite(v) else '--'
def sci(v):
 if not np.isfinite(v) or v<=0:return '--'
 a,e=f'{v:.2e}'.split('e');return rf'${a}\times10^{{{int(e)}}}$'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def chunked_execution_section(collected,collection_path=None):
 """Gate the exact selected chunked points using current source/input identities."""
 def check(ok,message):
  if not ok:raise RuntimeError('Chunked-execution note input: '+message)
 collection_path=Path(collection_path or HERE/'summary/observed_calibrated_limits.csv').resolve()
 collection_sha=sha(collection_path)
 selected={};point_data={}
 for row in collected.loc[collected.checkpoint_completed.eq(True)].itertuples():
  path=Path(row.checkpoint_path).resolve()
  check(path.is_relative_to(HERE),'selected checkpoint leaves this study')
  data=json.loads(path.read_text())
  if 'execution_layout' not in data:continue
  check(isinstance(data['execution_layout'],dict) and bool(data['execution_layout']),
        'selected checkpoint has an empty or malformed execution marker')
  check((data['scope_key'],data['mass_MeV'])==(row.scope_key,int(row.mass_MeV))
        and data['status']=='completed_point','selected coordinate/status differs')
  name=str(path.relative_to(ROOT));check(name not in selected,'duplicate selected chunked checkpoint')
  selected[name]=sha(path);point_data[name]=data
 if not selected:
  return '',set(),dict(status='not_used',selected_points=0,collection_sha256=collection_sha)
 inputs={collection_path}
 def verify_map(mapping):
  for name,expected in mapping.items():
   path=(ROOT/name).resolve()
   check(path.is_relative_to(ROOT),'unpublished input outside repository: '+name)
   check(sha(path)==expected,'input/source hash mismatch: '+name);inputs.add(path)
 qa_path=HERE/'summary/chunked_execution_qa.json';pure_path=HERE/'qa/chunked_execution_contract_test.json'
 qa=json.loads(qa_path.read_text());pure=json.loads(pure_path.read_text())
 check(qa.get('passed') is True and qa.get('collection_sha256')==collection_sha
       and qa.get('selected_chunked_count')==len(selected),'supplemental QA is not for this stable collection')
 source_names={str((HERE/name).relative_to(ROOT)) for name in
               ('audit_chunked_results.py','run_chunked_refinement.py','CHUNKED_REFINEMENT_PROTOCOL.md','collect_results.py')}
 check(set(qa.get('source_sha256',{}))==source_names,'supplemental QA source coverage differs')
 verify_map(qa['source_sha256'])
 pure_sources={str((HERE/name).relative_to(ROOT)) for name in
               ('run_chunked_refinement.py','validate_chunked_refinement.py','CHUNKED_REFINEMENT_PROTOCOL.md')}
 check(pure.get('passed') is True and pure.get('test_count')==24 and len(pure.get('checks',[]))==24
       and len({r['name'] for r in pure['checks']})==24 and all(r.get('passed') is True for r in pure['checks'])
       and set(pure.get('source_hashes',{}))==pure_sources,'current 24-case pure execution checks are incomplete')
 verify_map(pure['source_hashes'])
 baseline_contract=HERE/'derived/contract.json'
 baseline=json.loads(baseline_contract.read_text())
 check(pure.get('baseline_contract_sha256')==sha(baseline_contract) and len(baseline['hashes'])==47,
       'pure execution checks name a different original contract')
 inputs.update((qa_path,pure_path,baseline_contract))
 rows=qa.get('checks',[]);audited={}
 for row in rows:
  path=Path(row['path']).resolve()
  check(path.is_relative_to(HERE),'audited checkpoint leaves this study')
  name=str(path.relative_to(ROOT));check(name not in audited,'duplicate supplemental QA checkpoint')
  audited[name]=row['sha256']
 check(len(rows)==len(selected) and audited==selected,'supplemental QA selected-point SHA set differs')
 # The supplemental gate is read-only. Recompute its numerical/schema decisions
 # as well as verifying its saved source/input hashes; it never fits spectra.
 from audit_chunked_results import audit_point
 calibration_spectra=0;by_scope={};max_memory=0.;max_memory_limit=0.;resource_count=0
 resource_qa_by_version={};resource_counts_by_version={}
 reference_recovery_count=0;reference_recovery_coordinates=[]
 postprocessing_finalizations=[]
 resource_versions={1:(6,10,'CHUNKED_RESOURCE_PROTOCOL.md'),2:(8,12,'CHUNKED_RESOURCE8_PROTOCOL.md')}
 for row in rows:
  path=Path(row['path']).resolve();name=str(path.relative_to(ROOT));data=point_data[name]
  check(row.get('passed') is True,'a selected chunked point failed supplemental QA')
  layout=data['execution_layout'];resource_override='resource_policy' in layout
  inherited_layout={k:v for k,v in layout.items() if k!='scalar_reference_recovery'}
  limit=layout['max_memory_gib']
  check(row.get('resource_override') is resource_override and row.get('memory_limit_gib')==limit
        and not isinstance(limit,bool) and np.isfinite(limit) and 0<limit<= (8 if resource_override else 4)
        and data['memory_check']['limit_gib']==limit,'supplemental memory policy differs')
  parent=path.parent;contract_path=path.parents[2]/'contract.json'
  contract=json.loads(contract_path.read_text())
  required={path,contract_path,parent/'point_plan.json',parent/'proposal_plan.json',parent/'memory_check.json',
            parent/'chunked_equivalence_checks.json',parent/'model_chunk_ledger.json',
            parent/'validation_summary.csv',parent/'validation_toys.csv.gz',
            Path(data['sampling_refinement']['baseline_checkpoint_path']).resolve()}
  if 'postprocessing_finalization' in data:
   check('scalar_reference_recovery' in layout,'metadata finalization lacks its declared reference recovery')
   import finalize_reference_metadata as finalizer
   required.update(finalizer.verify_finalization(data,contract))
   finalization=data['postprocessing_finalization']
   verify_map(finalization['input_sha256'])
   postprocessing_finalizations.append(dict(scope_key=data['scope_key'],mass_MeV=data['mass_MeV'],
    checkpoint=name,checkpoint_sha256=selected[name],metadata=finalization))
  else:
   check('postprocessing_finalization' not in contract,'contract names unmarked metadata finalization')
  if 'scalar_reference_recovery' in layout:
   import scalar_reference_recovery as recovery
   required.update(recovery.verify_recovery_layout(layout,data))
   record=data['scalar_reference_recovery'];marker=layout['scalar_reference_recovery']
   ledger_path=Path(record['ledger_path']).resolve();ledger=json.loads(ledger_path.read_text())
   count=ledger['fallback_count']
   check(type(row.get('reference_recovery_count')) is int and row['reference_recovery_count']==count
         and count==len(ledger['fallbacks']) and count>0,'supplemental reference recovery count differs')
   diagnostic=Path(record['diagnostic_directory']).resolve()
   diagnostic_sha=sha(diagnostic/'summary.json')
   check(record['diagnostic_summary_sha256']==diagnostic_sha,'reference diagnostic identity differs')
   verify_map(marker['source_hashes'])
   reference_recovery_count+=count
   reference_recovery_coordinates.append(dict(scope_key=data['scope_key'],mass_MeV=data['mass_MeV'],
    checkpoint=name,checkpoint_sha256=selected[name],fallback_count=count,
    ledger_path=str(ledger_path.relative_to(ROOT)),ledger_sha256=record['ledger_sha256'],
    diagnostic_directory=str(diagnostic.relative_to(ROOT)),diagnostic_summary_sha256=diagnostic_sha,
    source_hashes=marker['source_hashes'],original_failure_sha256=marker['original_failure_sha256'],
    original_failure_ledger_sha256=marker['original_failure_ledger_sha256'],
    original_attempt_contract_sha256=marker['original_attempt_contract_sha256'],
    pure_qa=str((HERE/'qa/scalar_reference_recovery_contract_test.json').relative_to(ROOT)),
    pure_qa_sha256=sha(HERE/'qa/scalar_reference_recovery_contract_test.json')))
  else:
   check(type(row.get('reference_recovery_count')) is int and row['reference_recovery_count']==0
         and 'scalar_reference_recovery' not in data,'unmarked scalar reference recovery')
  if resource_override:
   version=layout['resource_policy'].get('version')
   check(type(version) is int and version in resource_versions,'unsupported combined resource policy version')
   if version==1:import run_chunked_refinement_6gib as resource
   else:import run_chunked_refinement_8gib as resource
   resource.validate_resource_limits(limit,data['scope_key'])
   check(inherited_layout==resource.layout_marker(limit),'combined resource execution identity differs')
   resource_qa_path=resource.verify_resource_qa(inherited_layout)
   required.update((Path(resource.__file__).resolve(),HERE/resource_versions[version][2],resource_qa_path,pure_path))
   verify_map(layout['source_hashes'])
   resource_qa_by_version[str(version)]=dict(path=str(resource_qa_path.relative_to(ROOT)),sha256=sha(resource_qa_path),test_count=26)
   resource_counts_by_version[str(version)]=resource_counts_by_version.get(str(version),0)+1
   resource_count+=1
  check({str(p.relative_to(ROOT)) for p in required}.issubset(row.get('input_sha256',{})),
        'supplemental QA omitted a consumed point input')
  verify_map(row['input_sha256'])
  check(contract['hashes']==baseline['hashes'],'chunked original inference contract differs')
  verify_map(contract['hashes']);verify_map(contract.get('sampling_hashes',{}))
  check(audit_point(path)==row,'fresh supplemental point audit differs from the saved collection audit')
  calibration_spectra+=sum(v['n'] for v in data['provenance'].values())
  by_scope[data['scope_key']]=by_scope.get(data['scope_key'],0)+1
  max_memory=max(max_memory,data['memory_check']['estimated_peak_gib'])
  max_memory_limit=max(max_memory_limit,limit)
 check(sha(collection_path)==collection_sha,'stable collection changed during execution QA')
 n=len(selected)
 resource_text=''
 for version,count in sorted(resource_counts_by_version.items()):
  worker,aggregate,_=resource_versions[int(version)]
  resource_text+=(f'For {count} combined coordinates, the resource policy permits one worker with up to {worker}'+
    r'\,GiB alongside at most one 4\,GiB worker, with an aggregate budget of at most '+f'{aggregate}'+r'\,GiB. ')
 if resource_count:resource_text+='These policies require a fresh memory-pressure check before each launch. '
 paragraph=(r'\paragraph{Bounded-memory execution.} '+
  f'{n} selected mass coordinates, containing {calibration_spectra:,} calibration spectra, use chunks of 128 spectra with one BLAS thread per worker. '+
  f'A source-derived peak-array budget of at most {max_memory_limit:g}'+r'\,GiB per worker includes a 512\,MiB runtime allowance. '+
  resource_text+
  r'Full-spectrum Poisson weights and the bounded statistic are checked against the original expressions and unsplit/scalar fits, with signed-root agreement within $2\times10^{-5}$ and $q$ agreement within $10^{-4}$. '+
  r'The likelihood, proposal laws and inference target are unchanged. Validation reuses the same 500 holdout spectra per truth and injected strength; these counts are not added to previous validation ensembles.')
 if reference_recovery_coordinates:
  coordinates=len(reference_recovery_coordinates)
  paragraph+=('\n\n'+r'\paragraph{Scalar reference initialization.} '+
   f'{reference_recovery_count} fixed-background scalar reference '+('fit required' if reference_recovery_count==1 else 'fits required')+
   f' a bracketed score-root initializer across {coordinates} selected mass '+('coordinate. ' if coordinates==1 else 'coordinates. ')+
   r'The original solver is tried first and its restart must satisfy the same $2\times10^{-7}$ score threshold and all signed-root/$q$ agreement gates. '+
   r'The production batch statistic, GP convention, science proposal arrays, complete science banks and validation seeds are unchanged. '+
   r'The original failure and its successful replay diagnostic are retained. The original 18 numerical-audit draws are unchanged; fresh extended numerical-audit draws use the new source identity and remain separate from calibration and validation.')
 if postprocessing_finalizations:
  paragraph+=(' '+r'A later reporting-schema error was resolved by reconstructing completion metadata from saved results. All numerical result fields were preserved exactly, with no numerical reexecution.')
 metadata=dict(status='audited',selected_points=n,calibration_spectra=calibration_spectra,
  selected_points_by_scope=by_scope,max_estimated_peak_gib=max_memory,max_memory_limit_gib=max_memory_limit,
  resource_override_points=resource_count,resource_override_points_by_version=resource_counts_by_version,
  resource_pure_qa_by_version=resource_qa_by_version,collection_sha256=collection_sha,
  reference_recovery_count=reference_recovery_count,
  reference_recovery_coordinate_count=len(reference_recovery_coordinates),
  reference_recovery_coordinates=reference_recovery_coordinates,
  postprocessing_finalization_coordinate_count=len(postprocessing_finalizations),
  postprocessing_finalizations=postprocessing_finalizations,
  selected_point_sha256=selected,supplemental_qa=str(qa_path.relative_to(ROOT)),supplemental_qa_sha256=sha(qa_path),
  pure_qa=str(pure_path.relative_to(ROOT)),pure_qa_sha256=sha(pure_path),pure_test_count=24,
  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)})
 return paragraph,inputs,metadata

def reverse_truth_section(directory,collected):
 """Read only the explicitly selected auxiliary result; require publishable inputs."""
 def check(ok,message):
  if not ok:raise RuntimeError('Reverse-truth note input: '+message)
 legacy_path=PARENT/'injections/derived/extraction_summary.csv'
 legacy=pd.read_csv(legacy_path)
 old=legacy[legacy.ensemble.eq('retrained_sidebands')&legacy.mass_MeV.eq(71)&legacy.strength_sigma.eq(5)].set_index('method')
 check(len(old)==2 and int(old.loc['fixed','exclusion_count'])==349 and int(old.loc['profiled','exclusion_count'])==145
       and old['n'].eq(500).all(),'the original 71 MeV failure ledger changed')
 inputs={legacy_path,HERE/'validate_reverse_truth_71.py',HERE/'REVERSE_TRUTH_71_PROTOCOL.md'}
 lines=[r'\subsection{Auxiliary check of the known 71 MeV failure}\label{sec:reverse_truth_validation}',
  r'The original reverse-truth five-reference-sigma injections excluded the true yield in 349/500 fixed-background fits and 145/500 Gaussian-profiled fits. These are the recorded uncalibrated asymptotic results. The reverse truth is absent from the two-truth calibration envelope, so the main validation does not establish that this failure has been corrected.']
 if directory is None:
  lines.append(r'A follow-up against the final two-truth calibration has not yet been evaluated in this note. No auxiliary result is selected implicitly.')
  return '\n\n'.join(lines)+'\n',inputs,dict(status='not_evaluated',directory=None)
 directory=Path(directory).resolve()
 check(directory.is_relative_to(HERE/'reverse_truth_71'),'directory must be inside this study\'s reverse_truth_71 tree')
 check(not (directory/'FAILURE.txt').exists() and not (directory/'failure_numerical_qa.json').exists(),'selected directory contains a failure')
 summary=json.loads((directory/'summary.json').read_text());contract=json.loads((directory/'contract.json').read_text())
 check(summary['status']=='completed_auxiliary_diagnostic' and summary['scope_key']=='individual_2021_10pct'
       and summary['mass_MeV']==71 and summary['validation_spectra']==1500 and summary['paired_method_rows']==3000,'auxiliary run is not complete')
 required={'contract.json','results_table.csv','validation_toys.csv.gz','legacy_closure.json','numerical_qa.json',
           'calibration_provenance.json','normalization_readiness.json','memory_check.json',
           'calibration_array_closure_gp.json','calibration_array_closure_stress.json'}
 check(required.issubset(summary['output_sha256']),'missing output identity')
 consumed={directory/'summary.json'};regenerable={}
 for name,expected in summary['output_sha256'].items():
  check(Path(name).name==name,'output path escapes selected directory')
  if Path(name).suffix=='.npz':
   regenerable[name]=expected;continue
  if Path(name).suffix=='.log':continue
  path=directory/name;check(sha(path)==expected,'output hash mismatch: '+name);consumed.add(path)
 consumed.update(directory.glob('*.log'))
 native=json.loads((PARENT/'observed/derived/summary.json').read_text())['native_histograms']
 allowed_external={str(Path(r['path']).resolve()):r['sha256'] for r in native.values()}
 external={}
 expected_inputs={str(ROOT/p):h for p,h in contract['original_hashes'].items()}
 expected_inputs.update(contract['auxiliary_hashes'])
 expected_inputs.update({str(ROOT/p):h for p,h in contract.get('selected_sampling_hashes',{}).items()})
 baseline_contract=HERE/'derived/contract.json'
 check(len(contract['original_hashes'])==47 and contract['original_hashes']==json.loads(baseline_contract.read_text())['hashes']
       and contract['baseline_contract_sha256']==sha(baseline_contract),'original source contract differs')
 for name,expected in expected_inputs.items():
  path=Path(name).resolve()
  if path.is_relative_to(ROOT):
   check(sha(path)==expected,'source hash mismatch: '+name);inputs.add(path)
  else:
   check(allowed_external.get(str(path))==expected,'unpublished non-native input: '+name);external[str(path)]=expected
 inputs.add(PARENT/'observed/derived/summary.json')
 point=collected[collected.scope_key.eq('individual_2021_10pct')&collected.mass_MeV.eq(71)]
 check(len(point)==1 and bool(point.iloc[0].checkpoint_completed),'stable collection lacks completed 2021 m071')
 checkpoint=Path(point.iloc[0].checkpoint_path).resolve()
 check(checkpoint.is_relative_to(HERE) and checkpoint==Path(contract['checkpoint']).resolve()
       and sha(checkpoint)==contract['checkpoint_sha256'],'auxiliary checkpoint is not the stable collected selection')
 selected=json.loads(checkpoint.read_text());inputs.add(checkpoint)
 check(contract['calibration_truths']==['gp','stress'] and contract['nvalidation']==500
       and contract['dense_backend']=='archived_fit_gpr_full_covariance','unexpected calibration/validation convention')
 tolerances=dict(Atrue_abs=1e-7,Ahat_abs_counts=.05,signed_r_abs=2e-5,raw_cls_abs=2e-5,scalar_batch_q_abs=1e-4)
 check(contract['tolerances']==tolerances,'legacy closure tolerances changed')
 closure=json.loads((directory/'legacy_closure.json').read_text())
 check(len(closure)==6 and {(r['method'],r['strength']) for r in closure}=={(m,s) for m in ('fixed','profiled') for s in (0,2,5)},'legacy closure cells incomplete')
 for row in closure:
  check(row['passed'] is True and row['n']==500 and set(row['errors'])==set(tolerances) and all(np.isfinite(v) and 0<=v<=tolerances[k]
        for k,v in row['errors'].items()),'legacy closure failed')
 numerical=json.loads((directory/'numerical_qa.json').read_text())
 check(numerical['dense_backend']=='archived_fit_gpr_full_covariance' and numerical['nuisance_eigenvalue_cut']==0
       and numerical['retained_full_covariance'] is True and numerical['legacy_closure']==closure,'dense numerical convention mismatch')
 check(bool(numerical['scalar_checks']) and all(r['passed'] is True and r['r_error']<=2e-5
       and all(q['q_error']<=1e-4 for q in r['q_checks']) for r in numerical['scalar_checks']),'scalar/batch checks failed')
 calibration=json.loads((directory/'calibration_provenance.json').read_text())
 check(numerical['calibration']==calibration and set(calibration)=={'gp','stress'},'calibration provenance differs')
 check(summary['calibration_spectra']==sum(r['n'] for r in calibration.values()),'calibration spectrum total differs')
 for truth,row in calibration.items():
  array=json.loads((directory/f'calibration_array_closure_{truth}.json').read_text());original=selected['provenance'][truth]
  check(row['max_score']<2e-7 and array['passed'] is True and row['n']==array['n']==original['n']
        and row['truth_sha256']==original['truth_sha256']
        and row['whole_sha256']==array['whole_sha256']==original['whole_sha256']
        and row['proposals_sha256']==array['proposals_sha256']==original['proposals_sha256'],'dense bank identity or fit gate failed')
 normal=json.loads((directory/'normalization_readiness.json').read_text())
 check(len(normal)==6 and {(r['truth'],r['strength']) for r in normal}=={(t,s) for t in ('gp','stress') for s in (0,2,5)}
       and all(r['passed'] is True and np.isfinite(r['normalization_se']) and r['normalization_se']<=.05
       and abs(r['normalization']-1)<=max(.05,5*r['normalization_se']) for r in normal),'normalization readiness failed')
 memory=json.loads((directory/'memory_check.json').read_text())
 check(memory['passed'] is True and memory['estimated_peak_gib']<=memory['limit_gib'],'memory guard failed')
 toys=pd.read_csv(directory/'validation_toys.csv.gz');table=pd.read_csv(directory/'results_table.csv')
 check(len(toys)==3000 and np.isfinite(toys[['Ahat','signed_r','max_score']].to_numpy()).all()
       and toys.max_score.max()<2e-7,'validation fit ledger incomplete/nonfinite')
 check(len(table)==12 and len(table.drop_duplicates(['method','strength','procedure']))==12,'results table incomplete/duplicated')
 for method in ('profiled','fixed'):
  for strength in (0,2,5):
   cell=toys[toys.method.eq(method)&toys.strength.eq(strength)]
   check(len(cell)==500 and set(cell.toy_id)==set(range(500)),'toy IDs/counts differ')
   if strength:check(np.isfinite(cell.q_at_Atrue).all() and cell.q_at_Atrue.ge(0).all(),'bounded statistic is missing')
   for procedure in ('raw','calibrated'):
    item=table[table.method.eq(method)&table.strength.eq(strength)&table.procedure.eq(procedure)]
    check(len(item)==1,'table cell missing');item=item.iloc[0]
    check(item.n==500 and item.legacy_closure_passed and item.rejected==int(cell[procedure+'_rejected'].sum())
          and item.tail_mc_ready_count==int(cell.tail_mc_ready.sum())
          and item.mc_decision_resolved_count==int(cell.mc_decision_resolved.sum()),'table and toy ledger disagree')
   if strength==5:
    check(int(cell.raw_rejected.sum())==int(old.loc[method,'exclusion_count']),'old five-sigma failure changed')
 lines += [r'This auxiliary result reconstructs the selected 71 MeV calibration proposal arrays and their hashes, then evaluates both calibration and validation statistics with dense GP refits and the full conditioned covariance. It uses the original 1,500 spectra (500 at each of zero, two and five reference sigmas), regenerated from the original seeds and checked against the saved fitted yields and signed likelihood roots. The two methods remain paired; these spectra are reused and are not pooled with the main validation.',
  r'The bounded statistic is evaluated at the original physical injected yield, without a Wald reconstruction. Calibration still takes the larger tail result from the same two declared truths. The reverse truth is a known out-of-envelope diagnostic: this retrospective result does not establish truth-independent or global coverage.',
  r'\begin{table}[H]\centering\small',r'\begin{tabular}{llrrrr}\toprule',
  r'Test & Method & Raw & Calibrated & Tail-ready & MC-resolved\\\midrule']
 for strength,label in [(0,r'B-only local $5\%$'),(2,r'$2\sigma_{\rm ref}$ exclusion'),(5,r'$5\sigma_{\rm ref}$ exclusion')]:
  for method,method_label in [('profiled','Gaussian profile'),('fixed','Fixed mean')]:
   block=table[table.method.eq(method)&table.strength.eq(strength)].set_index('procedure');raw,cal=block.loc['raw'],block.loc['calibrated']
   lines.append(f'{label} & {method_label} & {int(raw.rejected)} & {int(cal.rejected)} & {int(cal.tail_mc_ready_count)} & {int(cal.mc_decision_resolved_count)}'+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\caption{Counts out of 500 in each original reverse-truth ensemble. Positive injections test $CL_s<0.10$; B-only spectra test local $p_0<0.05$. The last two columns qualify the calibrated decision: tail-ready requires both truth-specific tail effective sample sizes of at least 100 and finite errors; MC-resolved additionally places the envelope of pointwise $\pm1.96$-SE tail estimates wholly on one side of the threshold. Point estimates retain every toy, including limited decisions. Exact binomial intervals in the CSV condition on the chosen calibration bank and do not include its finite Monte Carlo uncertainty. No counts are pooled across methods, strengths or studies.}',r'\label{tab:reverse_truth_validation}\end{table}']
 positive=table[table.procedure.eq('calibrated')&table.strength.eq(5)].set_index('method')
 limited=int((~toys.tail_mc_ready).sum())
 lines.append(f'The five-reference-sigma exclusion point estimates fall to {int(positive.loc["profiled","rejected"])}/500 for the Gaussian profile and {int(positive.loc["fixed","rejected"])}/500 for the fixed mean. '+r'These values lie well below the nominal 10\% rate and indicate conservative behavior under this truth. '+f'However, {limited} of the 3,000 method--toy decisions have limited Monte Carlo precision; this diagnostic does not resolve coverage to percent-level accuracy.')
 inputs.update(consumed)
 metadata=dict(status='completed_auxiliary_diagnostic',directory=str(directory.relative_to(ROOT)),
  selected_checkpoint=str(checkpoint.relative_to(ROOT)),selected_checkpoint_sha256=sha(checkpoint),
  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)},external_input_sha256=external,
  regenerable_npz_output_sha256=regenerable,summary_sha256=sha(directory/'summary.json'))
 return '\n\n'.join(lines)+'\n',inputs,metadata

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true')
 parser.add_argument('--reverse-truth-dir',type=Path,help='Explicit completed auxiliary result directory; never infer latest')
 args=parser.parse_args()
 collected_path=HERE/'summary/observed_calibrated_limits.csv';collected_sha=sha(collected_path)
 d=pd.read_csv(collected_path);completed=int(d.checkpoint_completed.sum())
 if sha(collected_path)!=collected_sha:raise RuntimeError('Collected selection changed while reading; rebuild from a stable collection')
 if completed!=456 and not args.allow_partial:raise RuntimeError(f'Only {completed}/456 coordinates complete')
 chunked_text,chunked_inputs,chunked_metadata=chunked_execution_section(d,collected_path)
 reverse_text,reverse_inputs,reverse_metadata=reverse_truth_section(args.reverse_truth_dir,d)
 v=pd.read_csv(HERE/'summary/validation_summary.csv')
 for p in (PARENT/'note').glob('*.tex'):
  if p.name!='analysis_note.tex':shutil.copy2(p,NOTE/p.name)
 text=(PARENT/'note/analysis_note.tex').read_text().replace(r'\usepackage{microtype}',r'\usepackage{microtype}'+'\n'+r'\setlength{\emergencystretch}{1em}')
 for sub in ('comparison','observed','injections'):
  text=text.replace(f'../{sub}/',f'../../{PARENT.name}/{sub}/')
 text=text.replace('Background Profiling and Fixed-Background Tests}',r'Background Profiling, Fixed-Background Tests\\and 90\% $CL_s$ Calibration}')
 text=text.replace('and adds 27,000 pointwise pseudoexperiments.',r'and adds the original 27,000 extraction pseudoexperiments plus the conditional 90\% $CL_s$ calibration in Sec.~\ref{sec:calibration}.')
 text=text.replace('The fixed-background curves assume','The uncalibrated fixed-background curves assume')
 text=text.replace('No new\n2015/2016 injection calibration or scan-wide toy ensemble is inferred from them.',r'Those original tests alone imply no 2015/2016 calibration. The new study in Sec.~\ref{sec:calibration} tests the three individual datasets and their exact all-three combination. No scan-wide significance is calibrated.')
 combined=d[d.scope_key.eq('all_2015_2016_2021')]
 combined_ratio=combined.ratio_fixed_over_profiled_calibrated.median()
 combined_resolved=int((combined.status_profiled.eq('resolved')&combined.status_fixed.eq('resolved')).sum())
 old_findings='''Fixing the GP mean produces narrower conditional limits, but the injection
tests identify the price. It recovers signals with appropriate errors when
that mean is the true, known background. With GP uncertainty included in
generation, the fixed method excludes the stronger true injection in
22--32\\% of toys at nominal 90\\% \\CLs. Refitting sidebands exposes bias in both
methods: at 71 MeV the corresponding fractions are 69.8\\% fixed and 29.0\\%
profiled. This is a limitation of the current procedure as well as a reason
not to adopt the uncorrected fixed-background result.'''
 if old_findings not in text:raise RuntimeError('Parent principal-findings anchor changed')
 text=text.replace(old_findings,
  r'Fixing the GP mean produces narrower uncalibrated limits, but the injection tests show that treating an estimated background as known can severely under-cover. '+
  f'The new conditional calibration evaluates all {completed} scope/mass coordinates. For the combined scan, {combined_resolved}/41 masses meet the declared Monte Carlo precision gates for both methods. '+
  f'The median calibrated fixed/profiled observed-limit ratio is {combined_ratio:.2f}: the apparent fixed-background improvement does not persist across the scan. '+
  r'These are observed bounds under the declared background ensembles; they do not measure expected sensitivity or establish unconditional coverage.')
 lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrr}\toprule',r'Scope & Complete & Profile resolved & Fixed resolved & Raw ratio & Calibrated ratio\\\midrule']
 for key,label,slug in SCOPES:
  b=d[d.scope_key==key];n=int(b.checkpoint_completed.sum());finite=b.ratio_fixed_over_profiled_calibrated.replace([np.inf,-np.inf],np.nan)
  paired=b.checkpoint_completed & finite.notna()
  raw=(b.loc[paired,'eps2_fixed_display']/b.loc[paired,'eps2_current_display'])
  lines.append(f'{label} & {n}/{len(b)} & {int(b.status_profiled.eq("resolved").sum())} & {int(b.status_fixed.eq("resolved").sum())} & {number(raw.median())} & {number(finite[paired].median())}'+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\caption{Pointwise completion and Monte Carlo precision status. The last two columns give median fixed/profiled observed-limit ratios on the same finite completed mass subset. A smaller observed limit is not by itself a sensitivity or coverage statement. Resolved counts refer to the declared MC gates, not unconditional background-model qualification.}',r'\label{tab:calibration_summary}\end{table}']
 (NOTE/'calibration_table.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrr}\toprule',r' & \multicolumn{2}{c}{Gaussian profile} & \multicolumn{2}{c}{Fixed GP mean}\\',r'Scope & $m$ (MeV) & Calibrated $p_0$ & $m$ (MeV) & Calibrated $p_0$\\\midrule']
 for key,label,slug in SCOPES:
  b=d[d.scope_key==key];cells=[]
  for method in ('profiled','fixed'):
   usable=b[b[f'p0_{method}_calibrated'].gt(0)&b[f'p0_{method}_calibrated'].notna()]
   if usable.empty:cells+=['--','--']
   else:
    r=usable.loc[usable[f'p0_{method}_calibrated'].idxmin()];suffix=r'$^{\dagger}$' if r[f'status_p0_{method}'] not in ('resolved','bounded_atom') else ''
    cells += [str(int(r.mass_MeV)),sci(r[f'p0_{method}_calibrated'])+suffix]
  lines.append(label+' & '+' & '.join(cells)+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\caption{Minimum calibrated local $p_0$ among completed points. The calibration takes the larger tail probability from the two declared truths. A dagger marks limited MC precision. Mass selection is descriptive; these are not global p-values. The asymptotic minima are retained in Table~\ref{tab:pzero}.}',r'\end{table}']
 (NOTE/'calibrated_pzero_table.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{llrrrr}\toprule',r' & & \multicolumn{2}{c}{Gaussian profile} & \multicolumn{2}{c}{Fixed GP mean}\\',r'Scope & Truth & $|\langle\hat A\rangle|$ & $|\langle\hat A\rangle-\delta A_{\rm lin}|$ & $|\langle\hat A\rangle|$ & $|\langle\hat A\rangle-\delta A_{\rm lin}|$\\\midrule']
 for key,label,slug in SCOPES:
  for truth,truth_label in [('gp','Local GP'),('stress','Archived stress')]:
   cells=[]
   for method in ('profiled','fixed'):
    b=v[v.scope_key.eq(key)&v.strength.eq(0)&v.truth.eq(truth)&v.method.eq(method)]
    residual=b.signal_bias_sigma-b.linearized_zero_noise_bias_sigma
    cells += [number(b.signal_bias_sigma.abs().median()),number(residual.abs().median())]
   lines.append(label+' & '+truth_label+' & '+' & '.join(cells)+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\caption{Descriptive medians over completed background-only mass ensembles, in units of the fixed $\sigma_{\rm ref}$ at each mass. The first column for each method measures the absolute ensemble offset; the second measures its absolute difference from the linearized deterministic residual projection. These summaries retain the generating truths separately and do not pool toy counts or test coverage.}',r'\label{tab:bias_projection}\end{table}']
 (NOTE/'bias_projection_table.tex').write_text('\n'.join(lines)+'\n')
 nvalidation=int(v['n'].sum()/2) if len(v) else 0
 ncalibration=0;refinements=[]
 for name in d.loc[d.checkpoint_completed,'checkpoint_path']:
  r=json.loads(Path(name).read_text());ncalibration+=sum(t['n'] for t in r['provenance'].values())
  if r.get('sampling_refinement'):refinements.append(r)
 sampling=[]
 if refinements:
  strict=sum(r.get('approximation_candidate_audit',{}).get('active_candidate_id')=='eigenfeature_rtol1e-15_nuisance1e-7' for r in refinements)
  second=sum(r['sampling_refinement']['attempt']==2 for r in refinements)
  sampling=[r'\subsubsection{Independent sampling refinement}',
   f'The selected result set contains {len(refinements)} refined coordinates, including {second} second-attempt coordinates. '+
   r'Refinement eligibility uses only censoring and Monte Carlo diagnostics. It does not use validation outcomes or select a background method by its observed limit.',
   r'New proposal centers are placed near unresolved endpoints and extend scans without a crossing. For full-spectrum truth $t$ and signal $g$ per reference error, the local spacing is at most $0.75[\sum_i g_i^2/(t_i+a g_i)]^{-1/2}$. Dense proposal centers are separate from the inversion grid. Every original guard node is retained.',
   r'Refined truths use fresh independent banks with 512 draws per proposal, or 1,024 on a second attempt. Unrefined truths regenerate their original 256-draw banks and verify their full-array hashes. Old draws are never reweighted under a mixture adapted after inspecting them. The same 500 independent validation spectra per cell are rescored and counted once.',
   r'Original resolution tests still apply. Sampling checks also require the endpoint, bracket and slope evaluations to fall within the refined ranges, with normalization standard error at most 0.05. Caps or failed checks leave an explicit unresolved result.',
   f'At {strict} selected coordinates, a stricter nuisance-eigenvalue cutoff of '+r'$10^{-7}$ passes all 18 original proposal checks and the complete extended-range audit. '+
   r'The discrepancy tolerances and twelve-mode cap are unchanged. Original rejected approximation checks remain in the record; failure of the stricter candidate restores the dense calculation. Actual reference-fit or scalar/batch disagreement remains fatal.']
 if chunked_text:sampling.append(chunked_text)
 (NOTE/'sampling_refinement.tex').write_text('\n\n'.join(sampling)+'\n')
 summary=json.loads((HERE/'summary/calibration_summary.json').read_text())
 lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrr}\toprule',r'Validation family & Cells tested & Raw flags & Calibrated flags\\\midrule']
 for family,label in [('exclusion','True-yield exclusion'),('local','Background-only local rejection')]:
  data=summary['validation_families'].get(family,{});raw=summary['validation_families'].get('raw_'+family,{})
  lines.append(f"{label} & {data.get('n_tests',0)} & {raw.get('rejected_count',0)} & {data.get('rejected_count',0)}"+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\caption{One-sided binomial excess-rate screens with Holm adjustment at 0.05. Flags count rejected cell-level null hypotheses, not rejected toys. The exclusion null is 0.10; the local-rejection null is 0.05. Tests are separate families. Counts during an incomplete run are provisional because the family is still growing. No cell is discarded.}',r'\end{table}']
 if completed<456:lines += [rf'\textbf{{In-progress snapshot:}} {completed}/456 coordinates are complete. No complete-suite validation claim is made.']
 else:
  exc=summary['validation_families'].get('exclusion',{}).get('rejected_count',0)
  local=summary['validation_families'].get('local',{}).get('rejected_count',0)
  lines += [f'The completed suite contains {exc} adjusted excess-exclusion flags and {local} adjusted excess-local-rejection flags. All flagged cells, intervals, and MC precision statuses remain in the accompanying ledgers.']
 (NOTE/'validation_results.tex').write_text('\n'.join(lines)+'\n')
 (NOTE/'reverse_truth_validation.tex').write_text(reverse_text)
 (NOTE/'calibration_values.tex').write_text(r'\newcommand{\CalComplete}{'+str(completed)+'}\n'+r'\newcommand{\CalSpectra}{'+f'{ncalibration:,}'+'}\n'+r'\newcommand{\ValidationSpectra}{'+f'{nvalidation:,}'+'}\n')
 extension=(HERE/'calibration_sections.tex').read_text()
 ratios=[d[d.scope_key.eq(key)].ratio_fixed_over_profiled_calibrated.median() for key,_,_ in SCOPES]
 comparison=(f'The median calibrated fixed/profiled observed-limit ratios are {ratios[0]:.2f}, {ratios[1]:.2f}, and {ratios[2]:.2f} for 2015, 2016, and 2021, respectively, and {ratios[3]:.2f} for their combination (Table~'+r'\ref{tab:calibration_summary}). '+
  f'In the combined scan, the fixed statistic gives a smaller calibrated bound at {int(combined.ratio_fixed_over_profiled_calibrated.lt(1).sum())} of 41 masses. '+
  r'The raw fixed-background gain therefore does not provide a basis for changing the nominal analysis. The calibration is mass- and truth-dependent; a universal rescaling of significance would not reproduce these comparisons.')
 extension=extension.replace('%%CALIBRATED_COMPARISON%%',comparison)
 for key,label,slug in SCOPES:
  placeholder='%%LIMIT_'+slug.upper()+'%%'
  statement=r'The 2016 numerical exception is inherited.' if slug in ('2016','combined') else ''
  extension=extension.replace(placeholder,rf'''\clearpage
\subsection{{Observed limits: {label}}}
\fig{{0.99}}{{../figures/limits_{slug}.pdf}}{{
Observed 90\% $CL_s$ limits before and after the conditional toy calibration.
The calibrated curve is the larger endpoint for the mass-local GP and archived
stress truths. Shading takes the componentwise maxima of the two approximate
95\% Monte Carlo endpoint intervals; it is neither a simultaneous confidence
band nor an expected-limit band. Open circles fail at least
one MC precision gate. Triangles indicate no finite endpoint within the tested range.
The ratio panel compares the two background treatments. {statement}
}}{{fig:cal_{slug}}}
''')
 text=text.replace(r'\appendix',extension+'\n'+r'\clearpage'+'\n'+r'\appendix',1)
 text=text.replace(r'\begin{thebibliography}{9}',r'\clearpage'+'\n'+r'\begin{thebibliography}{99}')
 text=text.replace(r'\end{thebibliography}',r'''\bibitem{readcls} A. L. Read, ``Presentation of search results: the $CL_s$ technique,''
\emph{J. Phys. G} \textbf{28}, 2693--2704 (2002),
\href{https://doi.org/10.1088/0954-3899/28/10/313}{doi:10.1088/0954-3899/28/10/313}.
\bibitem{berns} L. Berns, ``An importance sampling method for Feldman--Cousins confidence intervals,''
\emph{Phys. Rev. D} \textbf{109}, 092002 (2024),
\href{https://arxiv.org/abs/2303.11290}{arXiv:2303.11290}.
\end{thebibliography}''')
 text=text.replace(r'\date{5 September 2026}',r'\date{6 September 2026}' if completed==456 else rf'\date{{6 September 2026 --- calibration in progress ({completed}/456)}}')
 (NOTE/'analysis_note.tex').write_text(text)
 tectonic=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic'
 subprocess.run([tectonic,'--keep-logs','--keep-intermediates','analysis_note.tex'],cwd=NOTE,check=True)
 if sha(collected_path)!=collected_sha:raise RuntimeError('Collected selection changed during typesetting; rebuild from a stable collection')
 for name,expected in chunked_metadata.get('input_sha256',{}).items():
  if sha(ROOT/name)!=expected:raise RuntimeError('Chunked execution QA/input changed during typesetting: '+name)
 target=OUT/'HPS_GPR_Analysis_Note_v4p9p13_calibrated_backgrounds.pdf';shutil.copy2(NOTE/'analysis_note.pdf',target);print(target)
 inputs=[HERE/'summary'/name for name in ['observed_calibrated_limits.csv','validation_summary.csv','calibration_summary.json']]
 inputs += [Path(__file__),HERE/'calibration_sections.tex',PARENT/'note/analysis_note.tex']
 inputs += list((HERE/'figures').glob('*.pdf'))+list(NOTE.glob('*.tex'))
 inputs += list(reverse_inputs)+list(chunked_inputs)
 provenance=dict(completed_points=completed,allow_partial=args.allow_partial,reverse_truth=reverse_metadata,
  chunked_execution=chunked_metadata,pdf_path=str(target),pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
  inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 (HERE/'provenance/note_build.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
