from pathlib import Path
import json
from combinations import *
from scipy.stats import chi2
B=Path(__file__).resolve().parents[1];rng=np.random.default_rng(58420260918);rows=[]
for k in [1,2,3]:
 t=np.linspace(.01,60,301);err=float(np.max(abs(fisher_tail(t,np.ones(k))-chi2.sf(t,2*k))));assert err<1e-13
 q=np.array([.2,.5,.8])[:k];u=rng.random((k,200000));p=np.where(u<q[:,None],u,1.);T=-2*np.log(p).sum(axis=0);pcal=fisher_tail(T,q[:,None]);expected_atom=float(np.prod(1-q));assert abs(np.mean(T==0)-expected_atom)<.006
 for alpha in [.01,.05,.1]:
  rate=float(np.mean(pcal<=alpha));se=np.sqrt(alpha*(1-alpha)/len(T));assert abs(rate-alpha)<5*se
  rows.append(dict(k=k,alpha=alpha,rate=rate,binomial_standard_error=se,uniform_case_error=err,atom_probability=expected_atom))
z=np.linspace(-5,5,301);a=.4;s=.9;r=a+s*z;p=fisher_scores(z[None,:],r[None,:],np.array([[norm.cdf(a/s)]]))[1];expected=np.where(r>0,norm.sf(z),1.);assert np.max(abs(p-expected))<1e-14
result=dict(passed=True,uniform_chi_square_limit=True,exact_single_dataset_identity=True,atom_retained=True,simulation_rows=rows,simulations='Only analytic-mixture unit checks; no new HPS toys')
(B/'qa/fisher_validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Fisher mixture checks passed')
