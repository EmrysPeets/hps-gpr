"""Extract active silicon boxes from the pinned LCDD GDML hierarchy (no Geant4 run).
GDML passive Euler rotation becomes local->parent (Rz Ry Rx)^T.
See G4GDMLReadDefine::GetRotationMatrix and G4GDMLReadStructure::PhysvolRead.
"""
from pathlib import Path
import json, math, re, hashlib
import xml.etree.ElementTree as ET
import numpy as np
BASE=Path(__file__).resolve().parent
SHA='473c732dafaebdd3f58389cd1052a35c60098a23'
GEOMS={2015:'HPS-EngRun2015-Nominal-v6-0-fieldmap',2016:'HPS-PhysicsRun2016-Pass2',2021:'HPS_Run2021Pass2FEE'}
def rotation(a):
 x,y,z=a;cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
 rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]);rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
 return (rz@ry@rx).T

def extract(path):
 root=ET.parse(path).getroot();gdml=root.find('gdml');define=gdml.find('define')
 vectors={n.attrib['name']:np.array([float(n.get(a,'0')) for a in 'xyz']) for n in define if n.tag in ['position','rotation']}
 boxes={n.attrib['name']:np.array([float(n.get(a)) for a in 'xyz']) for n in gdml.find('solids') if n.tag=='box' and 'sensor_active' in n.attrib['name']}
 vols={n.attrib['name']:n for n in gdml.find('structure') if n.tag=='volume'}; out=[]
 def walk(name,R,t,ids):
  v=vols[name]
  if name.endswith('_sensor_active_volume'):
   d=boxes[v.find('solidref').get('ref')]; thin=int(np.argmin(d)); planar=[i for i in range(3) if i!=thin]
   corners=[]
   for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
    p=np.zeros(3);p[planar[0]]=a*d[planar[0]]/2;p[planar[1]]=b*d[planar[1]]/2;corners.append((t+R@p).tolist())
   m=re.search(r'module_L(\d+)([tb])_',name)
   out.append(dict(name=name,station=int(m[1]),half=m[2],sensor_type=('axial' if 'axial' in name else 'stereo'),center_mm=t.tolist(),rotation_local_to_global=R.tolist(),size_local_mm=d.tolist(),normal_axis_local=thin,planar_axes_local=planar,corners_mm=corners,physvolids=ids))
  for pv in v.findall('physvol'):
   child=pv.find('volumeref').get('ref');p=pv.find('positionref');rot=pv.find('rotationref');pos=vectors[p.get('ref')] if p is not None else np.zeros(3);rr=rotation(vectors[rot.get('ref')]) if rot is not None else np.eye(3)
   newids=ids.copy();newids.update({q.get('field_name'):int(q.get('value')) for q in pv.findall('physvolid')});walk(child,R@rr,t+R@pos,newids)
 walk(gdml.find('setup/world').get('ref'),np.eye(3),np.zeros(3),{})
 return out
res={'repository':'https://github.com/JeffersonLab/hps-java','commit':SHA,'coordinate_system':'LCDD world x horizontal, y vertical, z downstream; units mm. Beam line is tilted +30.52 mrad about y in nominal HPS world coordinates.','rotation_convention':'Local-to-parent is transpose of Rz(z) Ry(y) Rx(x), composed recursively.','scope':'Active silicon geometry only; excludes field, material, dead channels, trigger, reconstruction and selection. Selected 2015/2016 detector versions are representative, not verified source-dataset alignment conditions.','geometries':{}}
for year,name in GEOMS.items():
 p=BASE/'detector-data/detectors'/name/(name+'.lcdd');sensors=extract(p);res['geometries'][str(year)]={'name':name,'source_file':str(p.relative_to(BASE)),'source_url':f'https://github.com/JeffersonLab/hps-java/blob/{SHA}/detector-data/detectors/{name}/{name}.lcdd','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sensors':sensors}
 print(year,name,len(sensors))
 for st in sorted(set(s['station'] for s in sensors)):
  ss=[s for s in sensors if s['station']==st];cs=np.array([p for s in ss for p in s['corners_mm']]);print(st,'z',np.round(cs[:,2].min(),3),np.round(cs[:,2].max(),3),'abs y',np.round(abs(cs[:,1]).min(),3),np.round(abs(cs[:,1]).max(),3))
(BASE/'active_sensors.json').write_text(json.dumps(res,indent=2)+'\n')
# Attach assumptions and field provenance; these do not alter sensor geometry.
qa={}
for year,g in res['geometries'].items():
 compact=ET.parse(BASE/g['source_file'].replace(g['name']+'.lcdd','compact.xml')).getroot()
 alpha=float(next(n.get('value') for n in compact.findall('./define/constant') if n.get('name')=='beam_angle'))
 g['beam_angle_rad']=alpha
 g['nominal_parent_direction_global']=[math.sin(alpha),0,math.cos(alpha)]
 g['nominal_vertex_mm']=[0,0,0]
 g['vertex_status']='Assumed nominal origin; no target volume in LCDD. 2021 local HPSTR SLIC jobs explicitly set target_z=0; transverse zero and 2015/2016 origins are study assumptions.'
 g['fieldmap_definitions_not_applied']=[n.attrib for n in compact.findall('./fields/field')]
 g['lcd_header_comment']=ET.parse(BASE/g['source_file']).getroot().find('header/comment').text
 g['station_convention']='Geometry module number; '+('L1 represents physical L0, L2 upgraded L1; L3..L7 correspond to old L2..L6.' if year=='2021' else 'L1..L6 are physical stations 1..6.')
 Rs=[np.array(s['rotation_local_to_global']) for s in g['sensors']]
 plane=[float(abs((np.array(c)-s['center_mm'])@np.array(s['rotation_local_to_global'])[:,s['normal_axis_local']])) for s in g['sensors'] for c in s['corners_mm']]
 qa[year]={'sensor_count':len(Rs),'max_orthogonality_error':max(float(abs(R.T@R-np.eye(3)).max()) for R in Rs),'min_determinant':min(float(np.linalg.det(R)) for R in Rs),'max_determinant':max(float(np.linalg.det(R)) for R in Rs),'min_absolute_normal_dot_z':min(abs(R[2,s['normal_axis_local']]) for R,s in zip(Rs,g['sensors'])),'all_sensor_centers_in_named_half':all((s['center_mm'][1]>0)==(s['half']=='t') for s in g['sensors']),'max_corner_plane_residual_mm':max(plane),'z_center_min_mm':min(s['center_mm'][2] for s in g['sensors']),'z_center_max_mm':max(s['center_mm'][2] for s in g['sensors'])}
(BASE/'active_sensors.json').write_text(json.dumps(res,indent=2)+'\n')
(BASE/'geometry_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
manifest={'hps_java_commit':SHA,'hps_mc_commit':'308e27be7821f1aad406cb8060602a10fca2d37a','geant4_rotation_reference_commit':'62f62ecae238a7c304c52af4affbe70795475590','geant4_reference_branch_at_download':'geant4-11.3-release','files':[]}
for f in sorted(BASE.rglob('*')):
 if f.is_file() and f.name!='SOURCE_MANIFEST.json':manifest['files'].append({'path':str(f.relative_to(BASE)),'size_bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(BASE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
