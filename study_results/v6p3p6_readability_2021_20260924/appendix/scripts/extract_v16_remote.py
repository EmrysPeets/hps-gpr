"""Read-only ROOT branch streaming; emit one JSON record per completed mass.

No remote products, whole-file hashing passes, extra smearing or extra cuts.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,uproot,json,hashlib,time,socket
ROOT=Path('/sdf/data/hps/physics2021/preselection/v16');edges=np.arange(0,.4000001,.0001);start=time.monotonic()
branches=['vertex./vertex.invM_','vertex./vertex.type_','weight','true_ap./true_ap.mass_','ele_p_smear_ratio','pos_p_smear_ratio','psum','psum_scalar']
for family,directory,expected in [('tc','ap_signal_prompt_smeared','TargetConstrained'),('uc','ap_signal_prompt_dis_smeared','Unconstrained')]:
 for m in range(60,261,20):
  h=np.zeros(4000);h2=h.copy();files=[];types={};entries=selected=bad=nonunit=smear_nonunit=0;tot=tot2=under=over=0.;truthmin=float('inf');truthmax=-float('inf');pmin=float('inf');psmin=float('inf')
  for path in sorted((ROOT/directory/f'ap{m}MeV').glob('*.root')):
   before=path.stat();dig={k:hashlib.sha256() for k in branches}
   with ThreadPoolExecutor(max_workers=1) as ex:
    with uproot.open(path,decompression_executor=ex,interpretation_executor=ex) as f:
     t=f['preselection'];cutflows={}
     for name in ['vertex_cutflow_h','event_cutflow_h']:
      cutflows[name]={'labels':f[name].axis().labels(),'counts':f[name].values().tolist()}
     for a in t.iterate(branches,step_size=100000,library='np'):
      if time.monotonic()-start>600:raise RuntimeError('600-second extraction bound reached; completed records remain usable')
      for k in branches:
       ar=np.asarray(a[k]);dig[k].update(('\0'.join(ar.tolist())+'\0').encode() if ar.dtype.kind in 'OUS' else ar.tobytes())
      mass=np.asarray(a[branches[0]],float);weight=np.asarray(a['weight'],float);truth=np.asarray(a[branches[3]],float);typ=a[branches[1]]
      vals,counts=np.unique(typ,return_counts=True)
      for v,n in zip(vals,counts):types[str(v)]=types.get(str(v),0)+int(n)
      if np.any(typ!=expected):raise ValueError((family,m,'Unexpected vertex type',types))
      if np.any(weight<0):raise ValueError('Negative weights')
      good=np.isfinite(mass)&np.isfinite(weight)&(weight>0);entries+=len(mass);selected+=int(good.sum());bad+=int((~good).sum());nonunit+=int(np.sum(weight!=1))
      h+=np.histogram(mass[good],edges,weights=weight[good])[0];h2+=np.histogram(mass[good],edges,weights=weight[good]**2)[0]
      tot+=float(weight[good].sum());tot2+=float((weight[good]**2).sum());under+=float(weight[good&(mass<edges[0])].sum());over+=float(weight[good&(mass>=edges[-1])].sum())
      smear_nonunit+=int(np.sum((abs(a['ele_p_smear_ratio']-1)>1e-7)|(abs(a['pos_p_smear_ratio']-1)>1e-7)))
      truthmin=min(truthmin,float(truth.min()));truthmax=max(truthmax,float(truth.max()));pmin=min(pmin,float(a['psum'].min()));psmin=min(psmin,float(a['psum_scalar'].min()))
     tree_key=str(t.object_path);fileentries=t.num_entries
   after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
   files.append({'path':str(path),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'entries':fileentries,'tree_key':tree_key,'branch_payload_sha256':{k:v.hexdigest() for k,v in dig.items()},'cutflows':cutflows})
  assert files and abs(h.sum()+under+over-tot)<1e-5
  meta={'family':family,'campaign':'2021','version':'v16','source_directory':str(ROOT/directory),'mass_MeV':m,'vertex_types':types,'reconstruction_mass_branch':branches[0],'mass_units':'GeV','extra_cuts':False,'extra_smearing':False,'sumw':tot,'sumw2':tot2,'underflow_sumw':under,'overflow_sumw':over,'stats':{'entries':entries,'selected':selected,'nonfinite_or_nonpositive':bad,'nonunit_weights':nonunit,'underflow':under,'overflow':over,'smear_nonunit':smear_nonunit},'unit_weights':nonunit==0,'truth_mass_min_GeV':truthmin,'truth_mass_max_GeV':truthmax,'psum_min_GeV':pmin,'psum_scalar_min_GeV':psmin,'files':files,'hash_scope':'Canonical branch payloads in streamed basket order; not whole ROOT file hashes','host':socket.gethostname(),'elapsed_seconds':time.monotonic()-start}
  print(json.dumps({'family':family,'mass_MeV':m,'edges_GeV':edges.tolist(),'sumw':h.tolist(),'sumw2':h2.tolist(),'metadata':meta}),flush=True)
  print(f'{family} {m}: {entries} candidates; {time.monotonic()-start:.1f}s',file=__import__('sys').stderr,flush=True)
