"""Constructors are rerun after a fixed injected source signal; no likelihood fit."""
from pathlib import Path
import sys,time
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'scripts'))
import build_truths as b
import numpy as np,pandas as pd
from hps_gpr.template import build_window_template_from_full
source=np.load(B/'truths/backgrounds.npz');supp=np.load(B/'truths/supplementary.npz')
rows=[];arrays={};start=time.monotonic();datasets=b.c.production.make_datasets(b.cfg)
for mass in [76,90]:
 state=b.states['2016',mass];kernel=b.c.make_fixed_kernel(state['const_opt'],state['ls_opt'])
 p=b.c.production.estimate_background_for_dataset(datasets['2016'],mass/1000,b.cfg,rebin=5,restarts=0,kernel=kernel,optimize=False)
 assert np.array_equal(p.y_full,b.observed) and np.array_equal(p.edges_full,b.edges)
 w,S=build_window_template_from_full(p.edges_full,p.blind_mask,mass/1000,p.sigma_val,config=b.cfg)
 C,_=b.c.production.condition_covariance_block(p.cov,p.mu)
 sigma_ref=float(1/np.sqrt(w@np.linalg.solve(np.diag(p.mu)+C,w)));A=5*sigma_ref;injection=A*S
 new,meta=b.construct(b.observed+injection,names=['gp_full_nominal','gp_full_half_ls','gp_blocked','regional_rise_fall'])
 new['gp_blocked_local'],_=b.construct_local_blocked(b.observed+injection)
 arrays[f'injection_{mass}']=injection;arrays[f'signal_window_{mass}']=p.blind_mask
 for name,mean in new.items():
  baseline=source[name] if name in source else supp[name];delta=mean-baseline;keep=p.blind_mask
  frac=float(delta[keep].sum()/injection[keep].sum());projection=float(delta[keep]@S[keep]/(A*(S[keep]@S[keep])))
  poisson_projection=float((S[keep]/baseline[keep])@delta[keep]/(A*np.sum(S[keep]**2/baseline[keep])))
  rows.append(dict(truth=name,mass_MeV=mass,source_injection_sigma=5.,original_linear_sigma_yield=sigma_ref,injected_yield_window=float(injection[keep].sum()),source_absorbed_window_sum_fraction=frac,template_L2_projection_fraction=projection,template_poisson_projection_fraction=poisson_projection,all_support_sum_fraction=float(delta.sum()/injection.sum()),change_min=float(delta.min()),change_max=float(delta.max()),interpretation='Deterministic source-construction leakage diagnostic; not detection efficiency or calibrated bias.'))
  arrays[f'{name}_{mass}']=mean
np.savez_compressed(B/'truths/source_absorption.npz',edges_GeV=b.edges,observed=b.observed,**arrays)
pd.DataFrame(rows).to_csv(B/'truths/source_absorption.csv',index=False,float_format='%.17g')
b.write(B/'truths/source_absorption_protocol.json',dict(masses_MeV=[76,90],strengths_original_sigma=[5],sigma_definition='1/sqrt(w^T[diag(mu)+C]^-1w) at frozen observed prediction',stage='Signal added to observed source BEFORE rebuilding background; distinct from injection into already frozen truth followed by extraction.',constructor_keys=[r['truth'] for r in rows[:5]],source_builder_sha256=b.sha(B/'scripts/build_truths.py'),seconds=time.monotonic()-start,source_sha256=b.sha(b.SOURCE)))
print(pd.DataFrame(rows).to_string(index=False),flush=True)
