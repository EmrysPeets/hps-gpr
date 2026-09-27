"""Independent magnetic-transport checks: analytic helix, zero-field hits, step refinement."""
import json,time
import numpy as np
from pathlib import Path
import magnetic_model as mm,build_study as bs
B=mm.B
geo=json.loads((B/'inputs/geometry/active_sensors.json').read_text())['geometries']
result={'status':'passed','unit_system':'mm,MeV/c,T,q/e','checks':[]}
# Uniform field gives an analytic circle; compare at every saved path length.
m=mm.Model(geo['2021'],'2021');m.rows=np.empty((0,21));m.values=np.zeros((2,2,2,3));m.values[:,:,:,1]=.5
m.dims=np.array([2,2,2],dtype=np.int32);m.grid=np.array([-10000,-10000,-10000,20000,20000,20000.]);m.maxB=.5
for charge in (1,-1):
 path,out=m.path(np.array([0.,0.,1000.]),charge,step=5.)
 s=path[:,0];k=.299792458*charge*.5/1000
 expected=np.column_stack(((np.cos(k*s)-1)/k,0*s,np.sin(k*s)/k))
 residual=float(np.max(np.abs(path[:,1:4]-expected)))
 norm=float(np.max(np.abs(np.linalg.norm(path[:,4:7],axis=1)-1)))
 assert residual<1e-6 and norm<1e-12
 result['checks'].append({'test':'uniform_field_analytic_circle','charge':charge,'max_position_error_mm':residual,'max_direction_norm_error':norm})
# Zero-field transport must reproduce exact finite ray intersections.
u=bs.qmc.Sobol(2,scramble=False).random_base2(10);c=2*u[:,0]-1;phi=2*np.pi*u[:,1]
for year,E,*_ in bs.META:
 g=geo[year];m=mm.Model(g,year,field=False);rows=bs.prepared(g)
 for charge,p in zip((1,-1),bs.daughters(.055*E,E,c,phi,g['beam_angle_rad'])):
  tr=m.hits(p,charge);ray=bs.hit_features(p,rows)
  forward=p[:,2]>0
  assert np.array_equal(tr[0][forward],ray[0][forward]) and np.array_equal(tr[1][:,forward],ray[1][:,forward])
 result['checks'].append({'test':'zero_field_ray_equivalence','year':year,'directions_per_daughter':len(c),'all_counts_identical':True})
# Repeat actual-map transport with smaller steps at a single representative mass/year.
for year,E,*_ in bs.META:
 g=geo[year];m=mm.Model(g,year);ds=bs.daughters(.055*E,E,c,phi,g['beam_angle_rad']);masks=[]
 for step in (5.,2.5,1.25):
  pp=m.hits(ds[0],1,step);ee=m.hits(ds[1],-1,step)
  masks.append(bs.selection_mask(year,pp,ee))
 diff=int(np.count_nonzero(masks[0]!=masks[-1]));assert diff<=3
 result['checks'].append({'test':'field_step_refinement','year':year,'mass_MeV':.055*E,'directions':len(c),'steps_mm':[5,2.5,1.25],
  'accepted_counts':[int(x.sum()) for x in masks],'changed_classifications_5_vs_1p25':diff,
  'B_at_vertex_T':m.field_at(g['nominal_vertex_mm']).tolist()})
(B/'qa/magnetic_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
