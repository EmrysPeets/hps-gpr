"""Exact current-sample common-rate fits, then declared density projections."""
from common import *
import time,uproot
OUT=B/'derived/projection';OUT.mkdir(exist_ok=True)
card=json.loads((B/'inputs/2019_resolved_card.json').read_text())
scan=pd.read_csv(B/'inputs/2019_kernel_scan.csv',float_precision='round_trip').sort_values('mass_MeV')
with uproot.open(B/'inputs/hps_2019_1pct_invariant_mass.root') as f:
 native,edges=f['preselection/h_invM_8000'].to_numpy()
centers=(edges[:-1]+edges[1:])/2;sel=(centers>=.05)&(centers<=.3)
e=edges[np.flatnonzero(sel)[0]:np.flatnonzero(sel)[-1]+2];n=native[sel];assert len(n)%5==0
n=n.reshape(-1,5).sum(1);e=e[::5]
d=dict(x=(e[:-1]+e[1:])/2,edges=e,n=n,native_counts=native,native_edges=edges,masses=scan.mass_MeV.to_numpy(int),const=scan.const_opt.to_numpy(),ls=scan.ls_opt.to_numpy(),sigma_coeffs=np.array(card['resolved_dataset']['sigma_coeffs']),frad_effective=np.array(.05*(1-.046)))
np.savez_compressed(B/'inputs/spectrum_2019.npz',**d)
d['idx']={int(m):i for i,m in enumerate(d['masses'])};DATA['2019']=d;LIMITS['2019']=(75.,250.)
def density(y,m):
 d=DATA[y];ed=d['native_edges'];sig=sigma(y,m);lo=m/1000-1.64*sig;hi=m/1000+1.64*sig
 overlap=np.maximum(0.,np.minimum(ed[1:],hi)-np.maximum(ed[:-1],lo))
 return float(np.sum(d['native_counts']*overlap/np.diff(ed))/(hi-lo))
def fit(parts):
 return OneSignalProfile(np.concatenate([p['b'] for p in parts]),block_diag(*[p['L'] for p in parts]),np.concatenate([p['S'][:,0] for p in parts])).limit(np.concatenate([p['n'] for p in parts]))
def br(m):
 x=(2*.1056583745/(m/1000))**2
 return 1. if x>=1 else 1.+np.sqrt(1-x)*(1+x/2)
u=pd.read_csv(B/'inputs/v504_union.csv',float_precision='round_trip').set_index('mass_MeV');start=time.monotonic()
rows=[]
for m in range(19,251):
 path=OUT/f'm{m:03d}.json'
 if path.exists() and json.loads(path.read_text()).get('current_profile_backbone'):
  rows.append(json.loads(path.read_text()));continue
 old=u.loc[m];ys=old.dataset_set.split('+');dens={y:density(y,m) for y in ys};s3=np.sqrt(sum(dens.values())/sum(v*(10 if y=='2021' else 1) for y,v in dens.items()))
 ps=[moving_context(y,m) for y in ys];replay=fit(ps)
 record=dict(current_profile_backbone=True,mass_MeV=m,datasets_three='+'.join(ys),observed_three_ee=float(old.eps2_90),observed_three_minimal=float(old.eps2_observed),scale_three=float(s3),projected_three_minimal=float(old.eps2_observed*s3),density_2015=dens.get('2015',0),density_2016=dens.get('2016',0),density_2021=dens.get('2021',0),dimuon_factor=br(m))
 record.update(archived_three_ee=record['observed_three_ee'],observed_three_ee=replay['A90']*1e-8,observed_three_minimal=replay['A90']*1e-8*br(m),projected_three_minimal=replay['A90']*1e-8*br(m)*s3,parent_replay_relative_error=replay['A90']*1e-8/float(old.eps2_90)-1,three_cls=replay['cls'],three_max_score=replay['max_score'])
 if m>=75:
  p19=moving_context('2019',m);r19=fit([p19]);r4=fit(ps+[p19]);d19=density('2019',m);dens['2019']=d19
  replay_error=replay['A90']*1e-8/float(old.eps2_90)-1
  if abs(replay_error)>.1:raise RuntimeError(f'Parent replay mismatch needs review {m}: {replay_error}')
  if abs(d19/float(scan[scan.mass_MeV==m].integral_density.iloc[0])-1)>2e-10:raise RuntimeError('2019 density replay failed')
  s4=np.sqrt(sum(dens.values())/sum(v*{'2019':100,'2021':10}.get(y,1) for y,v in dens.items()))
  record.update(datasets_four='+'.join(ys+['2019']),density_2019=d19,observed_four_ee=r4['A90']*1e-8,observed_four_minimal=r4['A90']*1e-8*br(m),scale_four=float(s4),projected_four_minimal=float(r4['A90']*1e-8*br(m)*s4),max_score=r4['max_score'],cls=r4['cls'],parent_replay_relative_error=float(replay_error),standalone_2019_replay_ratio=float(r19['A90']*1e-8/scan[scan.mass_MeV==m].eps2_up_ee_channel.iloc[0]),standalone_2019_ee=float(r19['A90']*1e-8),covariance_load_2019=p19['diagnostic']['load'])
 else:record.update(datasets_four='+'.join(ys),density_2019=0.,observed_four_ee=record['observed_three_ee'],observed_four_minimal=record['observed_three_minimal'],scale_four=float(s3),projected_four_minimal=record['projected_three_minimal'])
 write(path,record);rows.append(record)
 if m%10==0 or m in [75,76,250]:print(m,'elapsed',round(time.monotonic()-start,1),'s',record.get('parent_replay_relative_error'),record.get('standalone_2019_replay_ratio'),flush=True)
pd.DataFrame(rows).to_csv(B/'derived/projected_contours.csv',index=False,float_format='%.17g')
write(B/'derived/projection_protocol.json',dict(input_samples={'2015':1.,'2016':1.,'2019':.01,'2021':.1},target_samples={y:1. for y in DATA},mass_ranges_MeV={y:list(v) for y,v in LIMITS.items()},projection='u_full_equivalent(m)=u_current_joint(m)*sqrt(sum(d_y)/sum(f_y*d_y)); f_2015=f_2016=1,f_2019=100,f_2021=10',current_joint='Actual profiled Poisson likelihood with shared nonnegative epsilon squared and block diagonal GP constraints; no combination of upper limits.',response_2019='Measured nominal 1% psum>3.64 GeV spectrum, with archived factor-15 2021-response proxy; kernel upper bound remains active at every 2019 point.',smoothing='None. Display joins adjacent tested nodes; no fit or averaging of limits.',calibration='Observed-equivalent statistics-only density scaling. No expected sensitivity, coverage or future observation is inferred.'))
