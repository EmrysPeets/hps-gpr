"""Exploratory 2016 empirical template; full probability, including support losses."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

B=Path(__file__).resolve().parents[1]

def make_template(mass_MeV, edges_MeV, excluded_mass=None):
    """Use a native sample or two-sided CDF morph. Never window-renormalize."""
    d=pd.read_csv(B/'results/centers_and_shapes.csv').set_index('mass_MeV')
    d=d[d.primary_domain & d.valid]
    if excluded_mass is not None:
        d=d.drop(excluded_mass)
    masses=d.index.to_numpy();m=float(mass_MeV);e=np.asarray(edges_MeV,dtype=float)
    if not np.isfinite(m) or m<masses.min() or m>masses.max():
        raise ValueError('Mass must lie inside the available two-sided 40-175 MeV template domain.')
    if e.ndim!=1 or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):
        raise ValueError('Requested edges must be finite and strictly increasing.')
    def source_cdf(j,x):
        a=np.load(B/'histograms'/f'm{j:03d}.npz')
        return np.interp(x,a['edges_MeV'],np.r_[0,np.cumsum(a['probability'])],left=0,right=1)
    if m in masses:
        j=int(m);q=source_cdf(j,e);c=float(d.loc[j,'center_MeV']);s=float(d.loc[j,'sigma_core_MeV'])
        kind='native';neighbors=[j];weights=[1.]
    else:
        a=int(masses[masses<m][-1]);b=int(masses[masses>m][0]);t=(m-a)/(b-a)
        c=float((1-t)*d.loc[a,'center_MeV']+t*d.loc[b,'center_MeV'])
        s=float(np.exp((1-t)*np.log(d.loc[a,'sigma_core_MeV'])+t*np.log(d.loc[b,'sigma_core_MeV'])))
        u=(e-c)/s
        q=(1-t)*source_cdf(a,d.loc[a,'center_MeV']+d.loc[a,'sigma_core_MeV']*u)+t*source_cdf(b,d.loc[b,'center_MeV']+d.loc[b,'sigma_core_MeV']*u)
        kind='interpolated';neighbors=[a,b];weights=[1-t,t]
    p=np.diff(q)
    assert p.min()>-1e-12 and q[0]>=-1e-12 and q[-1]<=1+1e-12
    out=dict(mass_MeV=m,kind=kind,source_masses_MeV=neighbors,weights=weights,
             center_MeV=c,sigma_core_MeV=s,below_support_probability=float(q[0]),
             above_support_probability=float(1-q[-1]),support_probability=float(p.sum()),
             normalization='Full selected distribution; outside-support categories retained; no window renormalization',
             scope='Exploratory MC interpolation, not a detector or inference calibration')
    return p,out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass',required=True,type=float)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    e=np.linspace(0,250,401);p,meta=make_template(args.mass,e)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,edges_MeV=e,probability=p,
                        below_support_probability=meta['below_support_probability'],
                        above_support_probability=meta['above_support_probability'])
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
