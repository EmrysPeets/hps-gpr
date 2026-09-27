"""Validate saved scan identities, local ranks and deterministic toy replays."""
import run_combined as J
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import beta
B=J.B
def main():
 d=pd.read_csv(B/'results/combined_scan.csv',float_precision='round_trip');t=pd.read_csv(B/'results/combined_local_toys.csv',float_precision='round_trip');cal=pd.read_csv(B/'results/combined_local_calibration.csv',float_precision='round_trip')
 assert len(d)==1248 and len(t)==3000 and not d.duplicated(['mass_MeV','policy','scope']).any() and not t.duplicated(['mass_MeV','policy','toy']).any()
 assert np.allclose(d.p0_asymptotic,ndtr(-np.maximum(d.signed_root,0)),rtol=1e-12,atol=1e-15)
 assert np.allclose(d.epsilon2_90_ee_proxy,d.psi90*1e-8,rtol=1e-14)
 assert np.allclose(d.epsilon2_90_visible_legacy,d.epsilon2_90_ee_proxy*np.array([J.branch(m) for m in d.mass_MeV]),rtol=1e-14)
 assert d.valid.all() and (d.max_score<3e-5).all() and (d.min_lambda>0).all() and (abs(d.cls-.1)<2e-6).all()
 m=int(cal.mass_MeV.iloc[0]);replays=[]
 for q in cal.itertuples():
  ref=d[(d.mass_MeV==m)&(d.scope=='combined')&(d.policy==q.policy)].iloc[0];a=t[t.policy==q.policy]
  assert len(a)==1000;k=int(np.sum(a.q0>=max(ref.signed_root,0)**2))
  lo=0. if k==0 else float(beta.ppf(.025,k,1001-k));hi=1. if k==1000 else float(beta.ppf(.975,k+1,1000-k))
  assert k==q.tail_count and abs(q.rank_p-(1+k)/1001)<1e-15 and abs(lo-q.cp95_low)<1e-14 and abs(hi-q.cp95_high)<1e-14
 for toy in [0,999]:
  counts={y:np.random.default_rng(np.random.SeedSequence([J.SEED,int(y),toy])).poisson(np.load(B/f'inputs/null_{y}.npz')['truth']) for y in J.years(m)}
  for p in J.POLICIES:
   parts=[J.oldpart(y,m,counts[y]) for y in J.years(m) if y!='2021']+[J.newpart(m,p,counts['2021'])];mod,n=J.model(parts);f=mod.fit(n);n0=mod.fit(n,fixed=0,initial=f['theta']);root=np.sign(f['A'])*np.sqrt(max(0,2*(n0['nll']-f['nll'])))
   q=t[(t.policy==p)&(t.toy==toy)].iloc[0];assert abs(root-q.signed_root)<1e-10 and abs(f['A']-q.psi_hat)<1e-10
   replays.append(dict(toy=toy,policy=p,passed=True))
 norm=[];geometry=[]
 for mass in range(60,241):
  for y in J.years(mass):
   K=J.conversion(y,mass);frad=float(J.C.DATA[y]['frad_effective']);density=K/1e-8*(2/137.)/(3*np.pi*mass/1000*frad)
   norm.append(dict(mass_MeV=mass,year=y,selected_events_per_psi=K,radiative_fraction_effective=frad,density_events_per_GeV=density,sigma_MeV=J.C.sigma(y,mass)*1000,visible_display_multiplier=J.branch(mass)))
  geometry.append(J.R.Context(mass,'morph_starter').geometry())
 J.R.csv(B/'results/normalization.csv',pd.DataFrame(norm));J.R.csv(B/'results/2021_template_geometry.csv',pd.DataFrame(geometry))
 for line in (B/'provenance/input_manifest.sha256').read_text().splitlines():
  digest,path=line.split('  ',1);assert J.sha(B/path)==digest,path
 J.write(B/'qa/statistical_audit.json',dict(passed=True,review_type='Deterministic saved-result audit and independent replay paths; not a separate reviewer',scan_rows=1248,toy_rows=3000,tail_counts_and_exact_intervals_verified=3,toy_replays=replays,normalization_rows=len(norm),full_input_manifest_verified=True,scan_sha256=J.sha(B/'results/combined_scan.csv'),toy_sha256=J.sha(B/'results/combined_local_toys.csv')))
 print('PASS: all scan rows, local ranks, normalization, inputs and six joint-toy replays')
if __name__=='__main__':main()
