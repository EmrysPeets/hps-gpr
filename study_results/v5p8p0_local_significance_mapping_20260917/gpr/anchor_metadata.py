from run_grid import *
rows=[]
for year,masses in [('2015',[51,78,95]),('2016',[42,66,76,92]),('2021',[71,92])]:
 for mass in masses:
  ctx=setup(year,mass);p=ctx['p'];o=observed(ctx)
  old=pd.read_csv(G[year]/'analysis/pvalue_curves.csv');q=old[(old.method=='profiled')&(old.mass_MeV==mass)]
  diff=float(o['observed_r']-q.iloc[0].observed_r) if len(q) else None
  if diff is not None:assert abs(diff)<2e-5,(year,mass,diff)
  K=float(c.production.A_from_epsilon2(datasets[year],mass/1000,1.,p.integral_density))
  rows.append(dict(dataset=year,mass_MeV=mass,**o,signal_scale_A_per_epsilon2=K,epsilon2_hat=o['Ahat']/K,epsilon2_sigma=o['sigma_A']/K,archived_r_difference=diff,bin_width_MeV=float(np.median(np.diff(p.edges_full))*1000)))
pd.DataFrame(rows).to_csv(B/'anchor_metadata.csv',index=False,float_format='%.17g')
print(pd.DataFrame(rows).to_string(index=False))
