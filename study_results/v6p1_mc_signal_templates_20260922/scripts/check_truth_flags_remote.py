"""Bounded candidate-association diagnostic; never used as a fit selection."""
from pathlib import Path
import json,re
import numpy as np,uproot
base=Path('/sdf/data/hps/physics2021/preselection/v13/ap_signal_prompt_smeared')
rows=[]
for directory in sorted(base.glob('ap*MeV'),key=lambda p:int(re.search(r'ap(\d+)MeV',p.name).group(1))):
    mass=int(re.search(r'ap(\d+)MeV',directory.name).group(1));arrays=[]
    for path in sorted(directory.glob('*.root')):
        with uproot.open(path) as f:
            arrays.append(f['preselection'].arrays(['vertex./vertex.invM_','ele_has_truth_link','pos_has_truth_link','psum','psum_scalar'],entry_stop=100000,library='np'))
    a={k:np.concatenate([d[k] for d in arrays]) for k in arrays[0]};m=a['vertex./vertex.invM_']*1000
    both=(a['ele_has_truth_link']>0)&(a['pos_has_truth_link']>0)
    sigma=(.00184825-.001375*mass/1000+.085875*(mass/1000)**2)*1000
    groups={}
    for label,mask in [('all',np.ones(len(m),dtype=bool)),('both_link_flags',both),('not_both_flags',~both)]:
        v=m[mask];groups[label]=dict(rows=len(v),median_MeV=float(np.median(v)) if len(v) else None,
          fraction_in_primary=float(np.mean(abs(v-mass)<=2.25*sigma)) if len(v) else None)
    rows.append(dict(mass_MeV=mass,first_rows_per_file=100000,flag_values={k:np.unique(a[k]).tolist() for k in ('ele_has_truth_link','pos_has_truth_link')},
                     groups=groups,stored_psum_below_2p8=int(np.sum(a['psum']<2.8-1e-6)),
                     stored_psum_scalar_below_2p8=int(np.sum(a['psum_scalar']<2.8-1e-6))))
print(json.dumps(dict(scope='First 100000 rows per source file; link-flag diagnostic only, no daughter association certification and not used in fitting.',samples=rows),indent=2))
