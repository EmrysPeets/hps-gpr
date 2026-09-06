#!/usr/bin/env python3
"""Check delivered numerical identities, uncertainty displays and PDF semantics."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.stats import norm,beta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,condition):
 checks.append({'name':name,'passed':bool(condition)})
 if not condition:raise AssertionError(name)
def hashes(mapping,prefix):
 for p,h in mapping.items():check(prefix+': '+p,sha(Path(p) if Path(p).is_absolute() else ROOT/p)==h)
def main():
 parent_counts={}
 for name in ['v4p9p16_combined_global_20260906','v4p9p16_presentation_extractions_20260906','v4p9p16_2015_lowmass_side_study_20260906']:
  rows=list(csv.DictReader((HERE.parent/name/'MANIFEST.csv').open()));parent_counts[name]=len(rows)
  for row in rows:check('frozen '+row['path'],sha(ROOT/row['path'])==row['sha256'])
 for name in ['independent_final_audit.json','independent_extension_audit.json']:
  x=json.loads((HERE/'review'/name).read_text());check(name,x['passed'])
  hashes(x.get('input_sha256',{}),name)
 final=json.loads((HERE/'review/final_interpretation_bindings.json').read_text());check('final HEP acceptance',final['accepted']);hashes(final['sha256'],'accepted review')
 p=pd.read_csv(HERE/'derived/probability_grid.csv');m=p.mass_MeV.to_numpy()
 check('full 232-point grid',np.array_equal(m,np.arange(19,251)))
 check('nominal p from same observed root',np.allclose(p.nominal_local_p,norm.sf(np.maximum(0,p.observed_r)),atol=1e-15,rtol=1e-14))
 check('conditional local sign gate retained',np.allclose(p.conditional_local_gaussian,np.where(p.observed_r>0,norm.sf(p.z),1),atol=1e-15,rtol=1e-14))
 for prefix in ['direct_local','direct_global','gp_global','gp_raw_global','direct_raw_global']:
  for row in p.to_dict('records'):
   k=row[prefix+'_k'];n=row[prefix+'_n'];v=row[prefix+'_p'];lo=row[prefix+'_low95'];hi=row[prefix+'_high95'];u=row[prefix+'_upper95']
   check(prefix+' tail range '+str(row['mass_MeV']),0<=lo<=v<=hi<=1 and 0<u<=1 and v==k/n)
 check('sparse direct interval exact',np.isclose(p.set_index('mass_MeV').loc[22,'direct_local_low95'],beta.ppf(.025,1,1000),atol=1e-15))
 # Runtime inspection of plotted artist data protects against clamping an error-bar endpoint.
 import importlib.util
 spec=importlib.util.spec_from_file_location('audit_figs',HERE/'make_figures.py');figmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(figmod)
 fig,ax=figmod.plt.subplots();figmod.empirical(ax,p[p.mass_MeV==22],'direct_local',figmod.GREEN,'s')
 endpoints=[segment[:,1] for col in ax.collections if hasattr(col,'get_segments') for segment in col.get_segments()]
 check('drawn central interval has exact lower endpoint',len(endpoints)==1 and np.isclose(endpoints[0].min(),beta.ppf(.025,1,1000),atol=1e-15));figmod.plt.close(fig)
 figsource=(HERE/'make_figures.py').read_text();check('no smoothing',not any(w in figsource for w in ['savgol','UnivariateSpline','rolling(']))
 check('local display includes sparse lower endpoint','ylim=(2e-5,1.6)' in figsource)
 check('separate sparse MC estimate markers','sparse=(k>0)&(k<25)' in figsource and "mfc='white'" in figsource)
 build=json.loads((HERE/'provenance/report_build.json').read_text());hashes(build['input_sha256'],'report input')
 pdf=ROOT/build['pdf'];check('PDF hash matches report build',sha(pdf)==build['pdf_sha256'])
 reader=PdfReader(pdf);texts=[x.extract_text() for x in reader.pages];full='\n'.join(texts)
 check('37 completed report pages',len(texts)==37);check('no empty pages',all(len(s.strip())>100 for s in texts))
 check('Figure 1 on page 2','Figure 1:' in texts[1] and 'Full combined search' in texts[1])
 fig1=PdfReader(HERE/'figures/combined_observed_limit_and_pvalues.pdf').pages[0].extract_text()
 check('old reference legend removed','4.9.12' not in fig1)
 check('archived stress subtitle removed',not any(t in fig1.lower() for t in ['archived','stress background']))
 # Exact source checks avoid PDF ligature/line-break differences.
 tex=(HERE/'note/analysis_note.tex').read_text();prob=(HERE/'note/probability_audit_section.tex').read_text();echo=(HERE/'note/signal_echo_section.tex').read_text();ext=(HERE/'note/extraction_section.tex').read_text()
 check('caption distinguishes sampling distribution',r'\emph{sampling distribution of the root}' in tex and 'not a Gaussian-shaped mass background' in tex)
 check('nominal convention explicit','p_0=0.5' in tex)
 check('MC precision and null validity distinct',all(t in prob for t in ['0.00299','10^{-20}','one-sided 95','Neither is a measured zero','display floor']))
 check('false dip argument removed','while a positive narrow signal does not by itself explain' not in ext)
 check('echo distinction and independent increment retained',all(t in echo for t in ['standalone low-mass injection is at 66','pair contains 65','not an additivity test','disjoint 20','joint positive two-template','not several independent confirmations']))
 check('2015 low-mass side study retained','Exploratory 2015 search' in full and '15–20' in full)
 check('no unresolved references','??' not in full)
 log=(pdf.parent/'analysis_note.log').read_text();problems=[s for s in log.splitlines() if re.search(r'Overfull|Underfull|undefined|Missing character|LaTeX Warning',s)]
 check('clean LaTeX layout/reference log',not problems)
 pages=sorted((HERE/'qa').glob('page-*.png'));check('render coverage',len(pages)==len(texts))
 (HERE/'qa/pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+s for i,s in enumerate(texts)))
 result={'passed':True,'checks':len(checks),'pages':len(texts),'pdf_sha256':sha(pdf),'frozen_parent_entries':parent_counts,'independent_numerical_checks':1955+3364,'rendered_pages_sha256':{str(p.relative_to(ROOT)):sha(p) for p in pages},'conditions':checks}
 (HERE/'qa/product_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['conditions','rendered_pages_sha256']},indent=2))
if __name__=='__main__':main()
