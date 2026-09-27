"""Read-only fixed v5.5.3 experiment and analytic Gaussian templates; no output mutations."""
from pathlib import Path
import os,sys,json,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'engine'))
import common as C
import numpy as np,pandas as pd
from scipy.linalg import cholesky,solve_triangular
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm,chi2
P=json.loads((B/'protocol.json').read_text());RA=json.loads((B/'inputs/resolution_inputs_audit.json').read_text());YS=['2015','2016','2021'];E=np.array([P['beam_energy_GeV'][y] for y in YS]);logE=np.log(E/2.3)
sigmc=np.array([np.polynomial.polynomial.polyval(.092,RA[y][{'2015':'mc_derived_coeffs','2016':'mc_tabulated_coeffs','2021':'pre_TC_scale_coeffs'}[y]])*1000 for y in YS]);sigscaled=np.array([C.sigma(y,92)*1000 for y in YS]);parts=[]
for i,y in enumerate(YS):
 d=C.DATA[y];lo=90-2.25*sigscaled[i];hi=94+2.25*sigscaled[i]
 p=C.context(y,[92],region=(lo/1000,hi/1000),anchor=92)
 ne=d['native_edges'];bw=np.diff(ne);dl=.092-1.64*sigmc[i]/1000;dh=.092+1.64*sigmc[i]/1000
 rho=float(np.sum(d['native_counts']*np.maximum(0,np.minimum(ne[1:],dh)-np.maximum(ne[:-1],dl))/bw)/(dh-dl));conv=3*np.pi*.092*float(d['frad_effective'])*rho/(2/137.)
 p.update(sigma_mc_MeV=sigmc[i],sigma_scaled_MeV=sigscaled[i],conversion_MC92=conv,x=d['x'][p['mask']]*1000,bin_width=np.diff(d['edges'])[p['mask']]*1000,edges=d['edges']*1000)
 parts.append(p)

def shape(i,m,t):
 p=parts[i];sig=sigmc[i]+t*(sigscaled[i]-sigmc[i]);z=(p['edges']-m)/sig;pdf=np.exp(-.5*z*z)/np.sqrt(2*np.pi);w=np.diff(ndtr(z));dm=-np.diff(pdf)/sig;ds=-np.diff(z*pdf)/sig;total=w.sum()
 dm=(dm*total-w*dm.sum())/total**2;ds=(ds*total-w*ds.sum())/total**2;w/=total
 c=p['conversion_MC92']*1e-8;mask=p['mask'];return c*w[mask],c*dm[mask],c*ds[mask]*(sigscaled[i]-sigmc[i])

nulls=[C.OneSignalProfile(p['b'],p['L'],shape(i,92,1)[0]).fit(p['n'],0) for i,p in enumerate(parts)];nullnll=sum(f['nll'] for f in nulls)
n=np.concatenate([p['n'] for p in parts]);b=np.concatenate([p['b'] for p in parts]);L=C.block_diag(*[p['L'] for p in parts]);offsets=np.cumsum([0]+[len(p['n']) for p in parts]);nullglobal=C.OneSignalProfile(b,L,np.concatenate([shape(i,92,1)[0] for i in range(3)])).fit(n,0)
assert abs(nullglobal['nll']-nullnll)<1e-7
