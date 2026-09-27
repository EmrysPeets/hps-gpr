from engine import *
import pandas as pd
rows=[];checks=[]
for width in P['blind_halfwidth_sigma']:
 for mass in [51.,66.,91.,92.]:
  ys=YEARS;parts={y:Context(y,mass,width).predict(C.DATA[y]['n']) for y in ys};fits={y:fit([parts[y]]) for y in ys};joint=fit([parts[y] for y in ys]);info=np.array([1/fits[y]['model'].fisher_sigma()**2 for y in ys]);scorew=np.sqrt(info/info.sum());nllfree=sum(fits[y]['f']['nll'] for y in ys);nll0=sum(fits[y]['z']['nll'] for y in ys);qc=2*(joint['f']['nll']-nllfree)
  assert qc>=-1e-6;assert abs(nll0-joint['z']['nll'])<1e-5
  checks.append(dict(width_sigma=width,mass_MeV=mass,unconstrained_coupling_equality_deviance=qc,shared_raw_r=joint['r'],sum_individual_raw_r2=sum(fits[y]['r']**2 for y in ys),identity_error=abs(qc-(sum(fits[y]['r']**2 for y in ys)-joint['r']**2))))
  for i,y in enumerate(ys):
   f=fits[y];rows.append(dict(width_sigma=width,mass_MeV=mass,scope=y,raw_r=f['r'],epsilon2_hat=f['f']['A']*1e-8,epsilon2_fit_sigma=f['f']['sigma']*1e-8,null_information_fraction=info[i]/info.sum(),linear_score_weight=scorew[i],shared_epsilon2_hat=joint['f']['A']*1e-8,shared_raw_r=joint['r'],coupling_equality_deviance=qc))
pd.DataFrame(rows).to_csv(B/'results/peak_composition.csv',index=False,float_format='%.17g');pd.DataFrame(checks).to_csv(B/'qa/peak_composition_identity.csv',index=False,float_format='%.17g')
print(pd.DataFrame(rows).query('width_sigma==2.25 and mass_MeV==92').to_string(index=False))
