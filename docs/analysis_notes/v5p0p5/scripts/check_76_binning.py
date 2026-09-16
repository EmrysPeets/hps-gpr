
"""Bounded signed joint-profile diagnostic; no toys, limits, or global tails.

Bin choices precede fitting: original, doubled and quadrupled widths; for
coarse widths also shift the origin by half a coarse bin. Input spectrum,
resolution, kernel coordinates and native-density conversion stay frozen.
Remainder bins are omitted at the far upper support edge, as in the parent
rebin operation. This varies support-edge sampling together with binning.
"""
from common import *
import copy,time
start=time.monotonic();original=copy.deepcopy(DATA);rows=[];audit=[]
for factor,offset in [(1,0),(2,0),(2,1),(4,0),(4,2)]:
 for y,d0 in original.items():
  d=copy.deepcopy(d0);n=d0['n'];ed=d0['edges'];stop=offset+(len(n)-offset)//factor*factor
  d['n']=n[offset:stop].reshape(-1,factor).sum(1)
  d['edges']=ed[offset:stop+1:factor];d['x']=(d['edges'][:-1]+d['edges'][1:])/2
  assert len(d['edges'])==len(d['n'])+1
  assert abs(d['n'].sum()-n[offset:stop].sum())<1e-6
  DATA[y].update(d)
  audit.append(dict(factor=factor,offset_base_bins=offset,year=y,bin_width_MeV=float(np.median(np.diff(d['edges']))*1000),support_MeV=(d['edges'][[0,-1]]*1000).tolist(),dropped_edge_counts=float(n[:offset].sum()+n[stop:].sum())))
 for m in range(72,81):
  if time.monotonic()-start>180:raise RuntimeError('Three-minute diagnostic budget exceeded')
  ps=[moving_context(y,m) for y in ['2015','2016','2021']]
  for scope,parts in [('combined',ps)]+[(p['year'],[p]) for p in ps]:
   b=np.concatenate([p['b'] for p in parts]);L=block_diag(*[p['L'] for p in parts]);S=np.concatenate([p['S'][:,0] for p in parts]);n=np.concatenate([p['n'] for p in parts])
   model=OneSignalProfile(b,L,S);f=model.fit(n);z=model.fit(n,0.)
   r=float(np.sign(f['A'])*np.sqrt(max(0.,2*(z['nll']-f['nll']))))
   rows.append(dict(scope=scope,mass_MeV=m,factor=factor,offset_base_bins=offset,signed_r=r,epsilon2_hat=f['A']*1e-8,max_score=max(f['score'],z['score']),min_lambda=min(f['min_lambda'],z['min_lambda'])))
 print('completed',factor,offset,round(time.monotonic()-start,2),flush=True)
pd.DataFrame(rows).to_csv(B/'derived/v505_76_binning.csv',index=False,float_format='%.17g')
write(B/'derived/v505_76_binning_protocol.json',dict(audit=audit,seconds=time.monotonic()-start,description=__doc__,scope='Observed signed pointwise profile only. No recalibration of stress-centered tails or global probability.',inputs={str(p.relative_to(B)):sha(p) for p in sorted((B/'inputs').glob('spectrum_*.npz'))}))
# Comparison to released baseline establishes the numerical replay scale.
baseline=pd.DataFrame(rows).query("factor==1 and offset_base_bins==0 and mass_MeV==76")
print(baseline.to_string(index=False))
print(pd.DataFrame(rows).query("scope=='combined'").pivot(index='mass_MeV',columns=['factor','offset_base_bins'],values='signed_r').round(4).to_string())
