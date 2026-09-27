from tail_model import *

rows=[];missing=[]
for p in sorted((B/'derived/chunks').glob('*.json')):
    j=json.loads(p.read_text());rows+=j['rows'];missing+=j['excluded']
f=pd.DataFrame(rows).sort_values(['window','scope','family','kappa','mass_MeV'])
f.to_csv(B/'derived/scans.csv',index=False,float_format='%.17g')
pd.DataFrame(missing,columns=['scope','mass_MeV','window','reason']).to_csv(B/'derived/excluded.csv',index=False)
pd.DataFrame(shape_metrics()).to_csv(B/'derived/shape_metrics.csv',index=False,float_format='%.17g')
peaks=[];ratios=[]
for (window,year,family,k),g in f.groupby(['window','scope','family','kappa']):
    peak=g.loc[g.p0_fixed_mass.idxmin()];peaks.append(peak.to_dict())
    base=f[(f.window==window)&(f.scope==year)&(f.family=='gaussian')].set_index('mass_MeV')
    j=g.set_index('mass_MeV').join(base[['epsilon2_90','signed_r','p0_fixed_mass']],rsuffix='_base')
    ratio=j.epsilon2_90/j.epsilon2_90_base
    ratios.append(dict(window=window,scope=year,family=family,kappa=k,n=len(j),median_ratio=float(ratio.median()),
                       min_ratio=float(ratio.min()),max_ratio=float(ratio.max()),max_abs_delta_r=float(abs(j.signed_r-j.signed_r_base).max()),
                       max_abs_delta_p0=float(abs(j.p0_fixed_mass-j.p0_fixed_mass_base).max())))
pd.DataFrame(peaks).to_csv(B/'derived/pvalue_minima.csv',index=False,float_format='%.17g')
pd.DataFrame(ratios).to_csv(B/'derived/limit_ratios.csv',index=False,float_format='%.17g')
print(f.groupby(['window','scope']).size().to_string())
print(pd.DataFrame(peaks).query("family != 'curvature'")[['window','scope','kappa','mass_MeV','Z0','p0_fixed_mass','display_epsilon2_90']].to_string(index=False))
