#!/usr/bin/env python3
"""Postprocess exact q0=0 atoms; preserve counts and independently expose raw-root ranks."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];path=B/'local/local_tail_tests.csv';before=hashlib.sha256(path.read_bytes()).hexdigest();D=pd.read_csv(path)
atom=D.observed_r<=0
D.loc[atom,['p_lo95','p_hi95']]=1.
D['interval_kind']=np.where(atom,'exact inclusive q0 atom; no Monte Carlo uncertainty','two-sided95 Clopper-Pearson conditional binomial interval')
raw=[]
for _,r in D.iterrows():
 a=np.load(B/f'local/{int(r.dataset)}_{int(r.mass_MeV)}.npz');raw.append(int(np.count_nonzero(a[r.truth+'_roots']>=float(a['baseline_roots'][0]))))
D['raw_signed_root_exceedances']=raw
D.to_csv(path,index=False,float_format='%.17g')
meta=dict(rows=len(D),exact_atom_rows=int(atom.sum()),csv_sha256_before=before,csv_sha256_after=hashlib.sha256(path.read_bytes()).hexdigest(),counts_preserved=True,exceedances_definition='Inclusive q0 exceedance count, set N by statistic definition at q0obs=0',raw_signed_root_exceedances_definition='Number of stored signed roots >= observed signed root, without the q0 positive-signal truncation',original_per_anchor_json_retained=True,script=__file__)
(B/'statistics/exact_atom_postprocess.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
