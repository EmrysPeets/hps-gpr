"""Bounded, serial field scan; scientific data are separate from plotting."""
import json,time,csv,hashlib,sys
import numpy as np
import build_study as bs
import magnetic_model as mm
B=mm.B
geo=json.loads((B/'inputs/geometry/active_sensors.json').read_text())['geometries']
rows=[];summaries=[];checks=[];examples={};start=time.perf_counter()
existing={}
if '--reuse-scan' in sys.argv:
 with (B/'derived/magnetic_acceptance.csv').open() as f:
  for r in csv.DictReader(f):existing[(r['year'],float(r['mass_MeV']))]={k:(v if k=='year' else float(v)) for k,v in r.items()}
u=bs.qmc.Sobol(2,scramble=False).random_base2(12);c=2*u[:,0]-1;phi=2*np.pi*u[:,1];w=.75*(1+c*c)
def eval_field(m,E,g,model,year,x=1.,power=12,step=5.,details=False):
 if power==12:cc,pp,ww=c,phi,w
 else:
  uv=bs.qmc.Sobol(2,scramble=False).random_base2(power);cc=2*uv[:,0]-1;pp=2*np.pi*uv[:,1];ww=.75*(1+cc*cc)
 ds=bs.daughters(m,x*E,cc,pp,g['beam_angle_rad'])
 hits=[model.hits(d,q,step) for d,q in zip(ds,(1,-1))]
 beam=np.array(g['nominal_parent_direction_global']);fwd=(ds[0]@beam>0)&(ds[1]@beam>0)
 selected=fwd&bs.selection_mask(year,*hits);allst=fwd&bs.selection_mask(year,*hits,reference=True)
 vals=np.array([np.sum(ww*selected),np.sum(ww*allst)])/ww.sum()
 if details:return vals,selected,ds,hits,cc,pp
 return vals
for year,E,*_ in bs.META:
 g=geo[year];model=mm.Model(g,year);prepared=bs.prepared(g);yr=[]
 print('Starting',year,flush=True)
 for i,mass in enumerate(np.arange(5.,bs.MASS_LIMITS[year]+.01,5.)):
  if (year,mass) in existing:d=existing[(year,mass)]
  else:
   a=eval_field(mass,E,g,model,year);v=eval_field(mass,E,g,model,year,x=.8)[0]
   b=bs.evaluate(mass,E,g,prepared,year,power=12)[0]
   assert 0<=a[1]<=a[0]<=1 and 0<=v<=1 and 0<=b<=1
   d=dict(year=year,mass_MeV=mass,field_selected_x1=a[0],field_all_stations_x1=a[1],field_selected_x0p8=v,zero_field_selected_x1=b)
  rows.append(d);yr.append(d)
  if i%20==0:print(year,'mass',mass,'elapsed',round(time.perf_counter()-start,1),flush=True)
 bs.write_csv(B/'derived/magnetic_acceptance.csv',rows)
 positive=[r for r in yr if r['field_selected_x1']>0];peak=max(yr,key=lambda r:r['field_selected_x1'])
 summaries.append({'year':year,'beam_GeV':E/1000,'sampled_peak_MeV':peak['mass_MeV'],'sampled_peak_fraction':peak['field_selected_x1'],
  'positive_sampled_masses_MeV':[positive[0]['mass_MeV'],positive[-1]['mass_MeV']],
  'scan_last_point_zero':bool(yr[-1]['field_selected_x1']==0),'B_at_vertex_T':model.field_at(g['nominal_vertex_mm']).tolist(),
  'B_at_table_origin_T':model.field_at([21.17,0,457.2]).tolist()})
 for mass in [.025*E,.055*E,.1*E]:
  standard=eval_field(mass,E,g,model,year);refined=eval_field(mass,E,g,model,year,power=13)
  checks.append({'year':year,'mass_MeV':mass,'standard':standard.tolist(),'refined':refined.tolist(),
   'max_absolute_difference':float(abs(standard-refined).max())})
 examples[year]=[]
 for entry in (positive[0],positive[-1]):
  mass=entry['mass_MeV'];vals,mask,ds,hits,cc,pp=eval_field(mass,E,g,model,year,details=True)
  # Choose a passing event with the largest number of hit views; verify it at half-step.
  ids=np.flatnonzero(mask);score=hits[0][0][ids]+hits[1][0][ids];ids=ids[np.argsort(-score)]
  chosen=None
  for idx in ids:
   hh=[model.hits(d[idx:idx+1],q,step=1.25) for d,q in zip(ds,(1,-1))]
   if bs.selection_mask(year,*hh)[0]:chosen=(idx,hh);break
  assert chosen is not None
  idx,hh=chosen;paths=[model.path(d[idx],q,step=2.5)[0].tolist() for d,q in zip(ds,(1,-1))]
  examples[year].append({'mass_MeV':mass,'cos_theta_star':float(cc[idx]),'phi_star_rad':float(pp[idx]),
    'positron_2D_hits':int(hh[0][0][0]),'electron_2D_hits':int(hh[1][0][0]),
    'positron_paired_stations_1based':(np.flatnonzero(hh[0][1][:,0])+1).astype(int).tolist(),
    'electron_paired_stations_1based':(np.flatnonzero(hh[1][1][:,0])+1).astype(int).tolist(),
    'paths_s_xyz_uxuyuz':paths})
(B/'derived/magnetic_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
(B/'derived/magnetic_example_trajectories.json').write_text(json.dumps(examples)+'\n')
qa={'status':'passed','orientation_points':4096,'refinement_points':8192,'mass_step_MeV':5.,'step_max_mm':5.,
 'max_step_bend_rad':.01,'interpolation':'trilinear','integrator':'RK4 in arclength with unit-direction normalization',
 'sensor_intersection':'Hermite segment, residual <= 1e-7 mm; bracketed fallback',
 'forward_transport':'stop at first nonpositive global uz; no later recrossings modeled',
 'refinement_checks':checks,'max_absolute_quadrature_difference':max(q['max_absolute_difference'] for q in checks),
 'elapsed_seconds_this_invocation':time.perf_counter()-start,'reused_saved_scan':'--reuse-scan' in sys.argv,'single_worker':True,'other_detector_effects':'no material, hit inefficiency, trigger, reconstruction, production distribution, or other event cuts'}
(B/'qa/magnetic_scan_validation.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps({'summaries':summaries,'seconds':qa['elapsed_seconds_this_invocation'],'max_quad_delta':qa['max_absolute_quadrature_difference']},indent=2),flush=True)
