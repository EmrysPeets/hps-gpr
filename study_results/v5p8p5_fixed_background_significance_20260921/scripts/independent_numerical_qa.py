"""Compare new C=0 scalar fits with the inherited general likelihood solver."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import fixed_core as F
import numpy as np
from limit_solver import OneSignalProfile
B=Path(__file__).resolve().parents[1]
records=[]
for mass in [19.,22.,50.5,51.,66.,77.,78.,87.5,90.5,91.5,92.,100.,180.,250.]:
    parts=[]
    for year in F.YEARS:
        if not F.LIMITS[year][0]<=mass<=F.LIMITS[year][1]:continue
        ctx=F.Context(year,mass);d=F.C.DATA[year]
        b=ctx.predict(d['n'])
        b_original,_=F.C.predict(d['x'],d['n'],ctx.mask,ctx.const,ctx.ls)
        rel=float(np.max(np.abs(b-b_original)/b_original))
        assert rel<1e-9,(year,mass,rel)
        parts.append((year,d['n'][ctx.mask],b,ctx.S,rel))
    for label, subset in [(p[0],[p]) for p in parts]+[('shared',parts)]:
        n=np.concatenate([p[1] for p in subset]);b=np.concatenate([p[2] for p in subset]);S=np.concatenate([p[3] for p in subset])
        new=F.fit(n,b,S)
        model=OneSignalProfile(b,np.zeros((len(b),0)),S)
        f=model.fit(n);z=model.fit(n,0.)
        r=float(np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll']))))
        delta_r=abs(r-new['r'][0]);delta_A=abs(f['A']-new['A'][0])/max(1,abs(f['A']))
        assert delta_r<3e-6 and delta_A<1e-6,(label,mass,delta_r,delta_A)
        records.append({'scope':label,'mass_MeV':mass,'r':r,'root_difference':float(delta_r),'relative_amplitude_difference':float(delta_A),'background_relative_difference':max(p[4] for p in subset)})
result={'passed':True,'checks':len(records),'method':'Inherited general Poisson likelihood solver with zero nuisance columns; full covariance prediction checked against optimized diagonal-only GP prediction.','maximum_root_difference':max(x['root_difference'] for x in records),'records':records}
(B/'qa/independent_numerical.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
