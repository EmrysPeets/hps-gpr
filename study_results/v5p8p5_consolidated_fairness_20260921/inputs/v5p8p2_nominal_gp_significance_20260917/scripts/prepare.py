from engine import *
import hashlib,pandas as pd
rows=[]
for yi,y in enumerate(YEARS):
 d=C.DATA[y];const,ls=C.kernel_state(y,76);truth,_=C.predict(d['x'],d['n'],np.zeros(len(d['n']),bool),const,ls,query=d['x'])
 rng=np.random.default_rng(np.random.SeedSequence([P['seed_base'],int(y)]));samples=rng.poisson(truth,size=(P['validation_Poisson_scans'],len(truth))).astype(float)
 np.savez_compressed(B/f'inputs/null_{y}.npz',truth=truth,counts=samples,observed=d['n'],edges_GeV=d['edges'],const=const,ls=ls,seed=[P['seed_base'],int(y)])
 rows.append(dict(dataset=y,source_anchor_MeV=76,const=const,ls=ls,bins=len(truth),total_observed=d['n'].sum(),total_mean=truth.sum(),min_mean=truth.min(),source_Pearson_per_bin=np.mean((d['n']-truth)**2/truth)))
 if y=='2016':
  old=np.load(B.parents[1]/'study_results/v5p8p1_background_truths_20260917/truths/backgrounds.npz')['gp_full_nominal'];print('2016truthdelta',np.max(abs(old-truth)));assert np.allclose(old,truth,rtol=2e-9,atol=2e-6)
pd.DataFrame(rows).to_csv(B/'results/source_summary.csv',index=False)
manifest={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((B/'inputs').glob('*'))+list((B/'scripts').glob('*.py'))+[B/'protocol.json']}
(B/'inputs/preparation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(pd.DataFrame(rows).to_string(index=False))
