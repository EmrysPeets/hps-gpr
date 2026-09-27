#!/usr/bin/env python3
"""Small actual-geometry and semantic checks of the adopted hit criteria."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json, pathlib, importlib.util
import numpy as np
from scipy.stats import qmc
B=pathlib.Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('study',B/'scripts/build_study.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
G=json.loads((B/'inputs/geometry/active_sensors.json').read_text())['geometries']
def independent(dirs,g):
    groups={}
    for s in g['sensors']:
        R=np.array(s['rotation_local_to_global']);o=(np.array(g['nominal_vertex_mm'])-s['center_mm'])@R;v=dirs@R;k=s['normal_axis_local']
        t=np.divide(-o[k],v[:,k],out=np.full(len(dirs),-1.),where=abs(v[:,k])>1e-14);p=o+t[:,None]*v
        hit=t>0
        for a in s['planar_axes_local']:hit&=abs(p[:,a])<=s['size_local_mm'][a]/2+1e-9
        key=(s['station'],s['half'],s['sensor_type']);groups[key]=groups.get(key,np.zeros(len(dirs),bool))|hit
    return M.features_from_groups(groups)
results=[];example=None
for year,E in [('2015',1056.),('2016',2300.),('2021',3740.)]:
    power=12 if year=='2021' else 10;uv=qmc.Sobol(2,scramble=False).random_base2(power);c=2*uv[:,0]-1;phi=2*np.pi*uv[:,1];g=G[year];mass=.055*E
    ds=M.daughters(mass,E,c,phi,g['beam_angle_rad']);fs=[M.hit_features(d,M.prepared(g)) for d in ds]
    for d,f in zip(ds,fs):
        alt=independent(d,g);assert all(np.array_equal(a,b) for a,b in zip(alt,f))
    selected=M.selection_mask(year,*fs);strict=M.selection_mask(year,*fs,reference=True);assert not np.any(strict&~selected)
    lo,hi,intervals=M.vertical_edges(g,year);rl,rh,ri=M.vertical_edges(g,year,True)
    assert all(any(a<=c0+1e-12 and b>=d0-1e-12 for a,b in intervals) for c0,d0 in ri)
    r={'year':year,'mass_MeV':mass,'orientations':len(c),'selected':int(selected.sum()),'strict_all':int(strict.sum()),'strict_is_subset':True,'independent_local_coordinate_intersections_match':True,'selected_projected_intervals_rad':intervals,'all_station_projected_intervals_rad':ri}
    if year=='2021':
        unpaired=(fs[0][0]!=2*fs[0][1].sum(axis=0))|(fs[1][0]!=2*fs[1][1].sum(axis=0));swap=M.selection_mask(year,fs[1],fs[0]);ix=np.flatnonzero(selected&~strict&unpaired&~swap);assert len(ix)>0;i=int(ix[0])
        example={'sobol_index':i,'mass_MeV':mass,'parent_energy_MeV':E,'cos_theta_star_positron':float(c[i]),'phi_star_positron_rad':float(phi[i]),'positron_2D_views':int(fs[0][0][i]),'positron_complete_stations':(np.flatnonzero(fs[0][1][:,i])+1).tolist(),'electron_2D_views':int(fs[1][0][i]),'electron_complete_stations':(np.flatnonzero(fs[1][1][:,i])+1).tolist(),'adopted_selection':bool(selected[i]),'all_station_selection':bool(strict[i]),'charge_swapped_selection':bool(swap[i])}
        r['selected_with_unpaired_views']=int((selected&unpaired).sum());r['selected_failing_charge_swap']=int((selected&~swap).sum())
    results.append(r)
# Targeted semantic criterion: 5 paired stations with electron L1 absent is valid
# in 2015; the same absence on the positron is invalid. No fabricated efficiency.
p=np.ones((6,1),bool);p[5]=False;e=np.ones((6,1),bool);e[0]=False
fp=(2*p.sum(axis=0),p);fe=(2*e.sum(axis=0),e)
assert M.selection_mask('2015',fp,fe)[0]
assert not M.selection_mask('2015',fe,fp)[0]
assert M.selection_mask('2016',fp,fe)[0] and M.selection_mask('2016',fe,fp)[0]
out={'status':'passed','scope':'Small source-backed geometric hit-count checks; no transport or efficiency simulation','single_thread':True,'actual_geometry_checks':results,'actual_2021_unpaired_charge_sensitive_example':example,'2015_only_positron_mandatory_L1_L2':'passed','2016_no_extra_mandatory_inner_station':'passed'}
(B/'qa/hit_selection_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
