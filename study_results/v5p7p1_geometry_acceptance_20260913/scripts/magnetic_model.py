"""ctypes bridge to the single-threaded RK4 magnetic tracker."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import ctypes as ct,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[1]
D=ct.POINTER(ct.c_double);I=ct.POINTER(ct.c_int);U=ct.POINTER(ct.c_uint16)
lib=ct.CDLL(str(B/'scripts/field_transport.dylib'))
lib.transport.argtypes=[D,ct.c_int,D,ct.c_int,D,I,D,D,ct.c_int,ct.c_double,ct.c_double,ct.c_double,U]
lib.trajectory.argtypes=[D,D,ct.c_int,D,I,D,D,ct.c_int,ct.c_double,ct.c_double,ct.c_double,U,D,ct.c_int]
lib.trajectory.restype=ct.c_int
FILES={'2015':'125acm2_3kg_corrected_unfolded_scaled_0.7992','2016':'209acm2_5kg_corrected_unfolded_scaled_1.04545_v4','2021':'334acm3_8kg_corrected_unfolded_scaled_1.0508'}
def ptr(x,t=D):return x.ctypes.data_as(t)
class Model:
 def __init__(self,g,year,field=True):
  self.g=g;self.year=year;self.origin=np.array(g['nominal_vertex_mm'],dtype=float)
  self.rows=[]
  for s in g['sensors']:
   R=np.array(s['rotation_local_to_global']);a,b=s['planar_axes_local'];z=np.array(s['corners_mm'])[:,2]
   self.rows.append([*s['center_mm'],*R[:,s['normal_axis_local']],*R[:,a],*R[:,b],s['size_local_mm'][a]/2,s['size_local_mm'][b]/2,
        z.min(),z.max(),s['station']-1,0 if s['half']=='t' else 1,0 if s['sensor_type']=='axial' else 1,0,0])
  self.rows=np.array(self.rows,dtype=float);self.stations=max(s['station'] for s in g['sensors'])
  cache=B/'derived/field_cache';cache.mkdir(exist_ok=True)
  datafile=cache/f'{year}_tesla.npy';metafile=cache/f'{year}_grid.json'
  config=json.loads((B/'inputs/geometry/fieldmaps/active_field_config.json').read_text())['years'][year]
  path=B/config['input_relative_path'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
  assert sha==config['sha256'],'Pinned field source changed'
  identity={'source_sha256':sha,'source_path':config['input_relative_path'],
            'offset_mm':config['global_offset_mm'],'B_scale_to_T':config['Tesla_per_raw_field_unit']}
  previous=json.loads(metafile.read_text()) if metafile.exists() else {}
  if not datafile.exists() or any(previous.get(k)!=v for k,v in identity.items()):
   with path.open() as f:
    assert f.readline().strip()=='';dims=[int(v) for v in f.readline().split()]
    while not f.readline().strip().startswith('0 End'):pass
    raw=np.loadtxt(f)
   assert len(raw)==np.prod(dims)
   r=raw.reshape(*dims,6)
   axes=[r[:,0,0,0],r[0,:,0,1],r[0,0,:,2]]
   for a in axes:assert np.allclose(np.diff(a),np.diff(a)[0],rtol=0,atol=1e-9)
   assert np.allclose(r[:,:,:,0],axes[0][:,None,None])
   assert np.allclose(r[:,:,:,1],axes[1][None,:,None])
   assert np.allclose(r[:,:,:,2],axes[2][None,None,:])
   offset=np.array(config['global_offset_mm'])
   grid=[*[float(a[0])+o for a,o in zip(axes,offset)],*[float(a[1]-a[0]) for a in axes]]
   values=np.ascontiguousarray(r[:,:,:,3:]*config['Tesla_per_raw_field_unit'])
   np.save(datafile,values)
   metafile.write_text(json.dumps({**identity,'dims':dims,'global_grid_origin_and_steps_mm':grid,
    'max_B_T':float(np.linalg.norm(values,axis=3).max()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
  md=json.loads(metafile.read_text());self.values=np.load(datafile);self.dims=np.array(md['dims'],dtype=np.int32)
  self.grid=np.array(md['global_grid_origin_and_steps_mm'],dtype=float);self.maxB=md['max_B_T']
  corners=np.concatenate([s['corners_mm'] for s in g['sensors']])
  assert np.all(corners[:,:2]>self.grid[:2]) and np.all(corners[:,:2]<self.grid[:2]+(self.dims[:2]-1)*self.grid[3:5])
  assert corners[:,2].max()<940
  if not field:self.values[:]=0;self.maxB=0
 def hits(self,momenta,charge,step=5.):
  p=np.ascontiguousarray(momenta,dtype=float);out=np.zeros((len(p),4),dtype=np.uint16)
  lib.transport(ptr(p),len(p),ptr(self.origin),charge,ptr(self.values),ptr(self.dims,I),ptr(self.grid),ptr(self.rows),len(self.rows),step,self.maxB,940.,ptr(out,U))
  assert not out[:,3].any(),'Transport path or iteration bound reached'
  bits=np.arange(self.stations,dtype=np.uint16)
  strips=(((out[:,0,None]>>bits)&1).sum(axis=1)+((out[:,1,None]>>bits)&1).sum(axis=1))
  paired=((out[:,2,None]>>bits)&1).astype(bool).T
  return strips,paired
 def path(self,momentum,charge,step=2.5):
  p=np.ascontiguousarray(momentum,dtype=float);out=np.zeros(4,dtype=np.uint16);path=np.zeros((50001,7),dtype=float)
  n=lib.trajectory(ptr(p),ptr(self.origin),charge,ptr(self.values),ptr(self.dims,I),ptr(self.grid),ptr(self.rows),len(self.rows),step,self.maxB,940.,ptr(out,U),ptr(path),len(path))
  assert out[3]==0,'Transport path or iteration bound reached'
  return path[:n],out
 def field_at(self,r):
  # Independent Python trilinear read for descriptive on-axis B values.
  q=(np.array(r)-self.grid[:3])/self.grid[3:]
  if np.any(q<0) or np.any(q>self.dims-1):return np.zeros(3)
  i=np.minimum(q.astype(int),self.dims-2);f=q-i;v=np.zeros(3)
  for a in (0,1):
   for b in (0,1):
    for c in (0,1):v+=self.values[tuple(i+[a,b,c])]*np.prod(np.where([a,b,c],f,1-f))
  return v
