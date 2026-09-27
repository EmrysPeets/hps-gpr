"""Export attested input arrays; no fitting policy or earlier artifact is changed."""
from pathlib import Path
import os,sys,json,hashlib,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];R=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p2-20260909')
sys.path.insert(0,str(R/'study_results/v4p9p13_calibration_20260905'))
import calibration_core as core
import numpy as np,pandas as pd,uproot
from hps_gpr.template import build_full_template
from hps_gpr.io import _build_model,_compute_integral_density
c=core.c; prod=c.production
cfg=prod.load_config(prod.DEFAULT_CARD);datasets=prod.make_datasets(cfg);states=prod.state_map(pd.read_csv(prod.DEFAULT_STATES))
ledger={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def pin(p,name=None):
 p=Path(p);h=sha(p);q=B/'inputs'/(name or h[:12]+'_'+p.name);shutil.copy2(p,q)
 ledger[str(p)]={'sha256':h,'snapshot':str(q.relative_to(B))};return q
pin(prod.DEFAULT_CARD,'analysis_card.yaml');pin(prod.DEFAULT_STATES,'reviewed_gp_states.csv')
for year,ds in datasets.items():
 m=66;st=states[year,m]
 p=prod.estimate_background_for_dataset(ds,m/1000,cfg,kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False,restarts=0)
 h=sha(ds.root_path);assert h==prod.EXPECTED_HISTOGRAM_SHA256[year]
 ledger[ds.root_path]={'sha256':h,'histogram':ds.hist_name,'export':f'inputs/spectrum_{year}.npz'}
 model=_build_model(ds,(.065,.067),rebin=5,config=cfg,mass=.066)
 masses=np.array(prod.EXPECTED_DATASET_GRIDS[year]);sigmas=np.array([ds.sigma(v/1000) for v in masses])
 templates=np.array([build_full_template(p.edges_full,v/1000,s,config=cfg) for v,s in zip(masses,sigmas)])
 density=np.array([_compute_integral_density(model,v/1000,s,density_nsigma=1.64) for v,s in zip(masses,sigmas)])
 conversion=np.array([prod.A_from_epsilon2(ds,v/1000,1.,d) for v,d in zip(masses,density)])
 # Preserve positive fixed stress continuum as a diagnostic; no random generation.
 path,key=core.STRESS[year];path=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow')/path.relative_to(R)
 core.STRESS[year]=(path,key);stress=core.stress_truth(year,p)
 ledger[str(path)]={'sha256':sha(path),'histogram':key,'export':f'inputs/spectrum_{year}.npz:stress'}
 np.savez_compressed(B/f'inputs/spectrum_{year}.npz',x=p.x_full,edges=p.edges_full,n=p.y_full,masses=masses,sigma=sigmas,
  templates=templates,conversion=conversion,density=density,stress=stress,
  const=np.array([states[year,int(v)]['const_opt'] for v in masses]),ls=np.array([states[year,int(v)]['ls_opt'] for v in masses]),
  native_counts=uproot.open(ds.root_path)[ds.hist_name].values(),native_edges=uproot.open(ds.root_path)[ds.hist_name].axis().edges(),
  sigma_coeffs=np.array(ds.sigma_coeffs),frad_effective=ds.frad_effective(.066))
 print(year,len(p.y_full),'bins',p.y_full.min(),'minimum counts',flush=True)
 old=R/('study_results/v4p9p14_interpretation_global_20260906/global/2015' if year=='2015' else f'study_results/v4p9p15_global_2016_2021_20260906/global_fast/{year}')
 pin(old/'analysis/pvalue_curves.csv',f'old_pvalues_{year}.csv');pin(old/'analysis/covariance.npz',f'old_covariance_{year}.npz')
for p in [Path(c.__file__),Path(core.__file__),Path(prod.__file__)]:pin(p)
base=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow')
pin(base/'study_results/v5p1p0_binning_significance_20260909/report.pdf','reference_v5p1p0.pdf')
pin(base/'output/pdf/v5p0p2_sigcorr_audit_20260909/HPS_GPR_Analysis_Note_v5p0p2_Unblinding_Review_Draft.pdf','reference_v5p0p2.pdf')
pin(Path('/Users/emryspeets/Desktop/summer_26/HPS_GPR_Review_and_Signal_Plan/03_Signal_Assessment_Plan.pdf'),'reference_signal_plan.pdf')
pin(Path('/Users/emryspeets/Desktop/summer_26/HPS_GPR_Review_and_Signal_Plan/sources/03_Signal_Assessment_Plan.tex'),'reference_signal_plan.tex')
(B/'inputs/manifest.json').write_text(json.dumps(ledger,indent=2)+'\n')
(B/'inputs/scopes.json').write_text(json.dumps(prod.SCOPES,indent=2)+'\n')
