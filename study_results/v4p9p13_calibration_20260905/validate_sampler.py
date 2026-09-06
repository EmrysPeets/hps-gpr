#!/usr/bin/env python3
"""Independent analytic Poisson check of deterministic mixture tail weighting."""
from calibration_core import *
from scipy.stats import poisson

def main():
 means=np.array([100.,120.,150.,65.,90.,110.]);n=20000
 draws=np.concatenate([seed('analytic-poisson',i).poisson(mu,size=n) for i,mu in enumerate(means)])
 logmix=logsumexp(draws[:,None]*np.log(means/100.)-(means-100.),axis=1)-np.log(len(means))
 checks=[]
 for mu in (100.,115.,135.,150.):
  weights=np.exp(draws*np.log(mu/100.)-(mu-100.)-logmix)
  for threshold in (60,80,100,120,145):
   values=(weights*(draws<=threshold)).reshape(len(means),n)
   estimate=values.mean();se=np.sqrt(np.sum(values.var(axis=1,ddof=1)/n)/len(means)**2)
   exact=poisson.cdf(threshold,mu)
   z=abs(estimate-exact)/max(se,1e-300)
   # Finite runs can sample no events in extremely small tails; label them.
   resolved=bool(values.any())
   checks.append(dict(mean=mu,threshold=threshold,estimate=float(estimate),exact=float(exact),mc_se=float(se),z=float(z),resolved=resolved,passed=bool(not resolved or z<5)))
 assert all(r['passed'] for r in checks),checks
 out=dict(status='passed',spectra=len(draws),checks=checks,max_resolved_z=max(r['z'] for r in checks if r['resolved']),
  interpretation='Checks the sampling identity and stratified variance independently of GP fitting; no model-coverage claim.')
 (HERE/'sampler_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['max_resolved_z'])
if __name__=='__main__':main()
