"""Freeze descriptive echo landmarks on predeclared2–8sigma flanks."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.signal import find_peaks
B=Path(__file__).resolve().parents[1];D=B/'derived';CAT=pd.read_csv(B/'inputs/catalogue.csv')

def minimum(x,y,lo,hi):
 candidates=np.flatnonzero((x>=lo)&(x<=hi))
 if not len(candidates):return None,False,False
 local,_=find_peaks(-y);local=np.intersect1d(candidates,local)
 pool=local if len(local) else candidates;i=int(pool[np.argmin(y[pool])])
 return i,bool(len(local)),bool(i in [candidates[0],candidates[-1]])

def main():
 echoes=[];allmin=[];summaries=[];pointrows=[]
 for _,r in CAT.iterrows():
  folder=D/'scans'/r.scenario;bg=pd.read_csv(folder/'background_asimov.csv');a=pd.read_csv(folder/'matched_asimov.csv');y=pd.read_csv(folder/'yield_asimov.csv');toys=[pd.read_csv(folder/f'toy_{i:02}.csv') for i in range(20)]
  x=bg.mass_MeV.to_numpy();assert all(np.array_equal(t.mass_MeV,x) for t in [a,y]+toys)
  R=a.A90.to_numpy()/bg.A90.to_numpy();RY=y.A90.to_numpy()/bg.A90.to_numpy();RT=np.array([t.A90.to_numpy()/bg.A90.to_numpy() for t in toys]);eps=np.array([t.epsilon2_90 for t in toys]);z=np.array([t.signed_r for t in toys]);m0=r.mass_MeV;s=r.sigma_MeV
  q=np.quantile(RT,[.16,.5,.84],axis=0);qe=np.quantile(eps,[.16,.5,.84],axis=0)
  for j,m in enumerate(x):pointrows.append(dict(scenario=r.scenario,lane=r.lane,mass_MeV=m,ratio_Asimov=R[j],ratio_yield=RY[j],ratio_q16=q[0,j],ratio_median=q[1,j],ratio_q84=q[2,j],epsilon2_q16=qe[0,j],epsilon2_median=qe[1,j],epsilon2_q84=qe[2,j]))
  for side,lo,hi in [('left',m0-8*s,m0-2*s),('right',m0+2*s,m0+8*s)]:
   base=dict(scenario=r.scenario,lane=r.lane,region=r.region,injected_mass_MeV=m0,sigma_MeV=s,side=side,search_lo_MeV=max(lo,x[0]),search_hi_MeV=min(hi,x[-1]),search_grid_clipped=bool(lo<x[0] or hi>x[-1]))
   i,local,edge=minimum(x,R,lo,hi)
   if i is None:echoes.append(dict(**base,status='outside_scan',echo_mass_MeV=np.nan,echo_ratio=np.nan));continue
   assert x[i]>=lo and x[i]<=hi
   extent_lo=extent_hi=np.nan
   if R[i]<=.9:
    l=h=i
    while l>0 and x[l-1]>=lo and R[l-1]<=.9:l-=1
    while h<len(x)-1 and x[h+1]<=hi and R[h+1]<=.9:h+=1
    extent_lo=x[l];extent_hi=x[h]
   ts=[]
   for ti,tr in enumerate(RT):
    j,jlocal,jedge=minimum(x,tr,lo,hi);ts.append(dict(scenario=r.scenario,side=side,toy=ti,mass_min_MeV=x[j],ratio_min=tr[j],ratio_at_Asimov_echo=tr[i],minimum_is_local=jlocal,minimum_at_search_boundary=jedge))
   summaries+=ts;qs=np.quantile([t['mass_min_MeV'] for t in ts],[.16,.5,.84]);qd=np.quantile(RT[:,i],[.16,.5,.84])
   echoes.append(dict(**base,status='dip' if R[i]<=.9 else 'weak_or_no_dip',echo_mass_MeV=x[i],echo_ratio=R[i],echo_depth=1-R[i],offset_sigma=(x[i]-m0)/s,local_minimum=local,boundary=edge,negative_fit=bool(a.Ahat.iloc[i]<0),Asimov_signed_r=a.signed_r.iloc[i],yield_ratio_at_echo=RY[i],ratio90_lo_MeV=extent_lo,ratio90_hi_MeV=extent_hi,toy_q16_mass=qs[0],toy_median_mass=qs[1],toy_q84_mass=qs[2],toy_ratio_fixed_q16=qd[0],toy_ratio_fixed_median=qd[1],toy_ratio_fixed_q84=qd[2],toys_fixed_ratio_below90=int(np.sum(RT[:,i]<=.9)),toy_count=20))
  loc,_=find_peaks(-R)
  for j in loc:
   if abs(x[j]-m0)>=2*s and R[j]<=.9:allmin.append(dict(scenario=r.scenario,lane=r.lane,injected_mass_MeV=m0,mass_MeV=x[j],ratio=R[j],offset_sigma=(x[j]-m0)/s,inside_eight_sigma=bool(abs(x[j]-m0)<=8*s),signed_r=a.signed_r.iloc[j]))
 pd.DataFrame(echoes).to_csv(D/'echo_catalogue.csv',index=False,float_format='%.17g')
 pd.DataFrame(summaries).to_csv(D/'toy_echo_locations.csv',index=False,float_format='%.17g')
 pd.DataFrame(allmin).to_csv(D/'all_Asimov_dips.csv',index=False,float_format='%.17g')
 pd.DataFrame(pointrows).to_csv(D/'pointwise_summary.csv',index=False,float_format='%.17g')
 print(pd.DataFrame(echoes)[['scenario','side','status','echo_mass_MeV','echo_ratio','ratio90_lo_MeV','ratio90_hi_MeV']].to_string(index=False))
if __name__=='__main__':main()
