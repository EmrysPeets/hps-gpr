"""Stream only needed branches; write all derived products to the user src study.

The supplied production selection is retained, including all selected candidates.
No extra smearing, recentering, truth matching, or timing cut is applied.
"""
from pathlib import Path
import os,sys,json,time,hashlib,re
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np,uproot
from concurrent.futures import ThreadPoolExecutor

BASE=Path('/sdf/data/hps/physics2021/preselection/v13/ap_signal_prompt_smeared')
OUT=Path(sys.argv[1]);(OUT/'histograms').mkdir(parents=True,exist_ok=True);(OUT/'qa').mkdir(exist_ok=True)
edges=np.arange(0.,.4000001,.0001) # 0.1 MeV bins; common physical range for all samples
branches={'mass':'vertex./vertex.invM_','type':'vertex./vertex.type_',
          'weight':'weight','truth':'true_ap./true_ap.mass_',
          'se':'ele_p_smear_ratio','sp':'pos_p_smear_ratio'}
for particle in ('ele','pos'):
    for axis in ('px','py','pz'):branches[particle+axis]=f'{particle}./{particle}.{axis}_'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

rows=[];start=time.time()
for directory in sorted(BASE.glob('ap*MeV'),key=lambda p:int(re.search(r'ap(\d+)MeV',p.name).group(1))):
    m=int(re.search(r'ap(\d+)MeV',directory.name).group(1));h=np.zeros(len(edges)-1);h2=h.copy();blocks=[];files=[]
    stats=dict(entries=0,selected=0,nonfinite=0,nonpositive_weight=0,wrong_type=0,psum_below_2p8=0,underflow=0,overflow=0,smear_nonunit=0)
    tsum=tsum2=0.;truth_min=np.inf;truth_max=-np.inf;partindex=0
    for path in sorted(directory.glob('*.root')):
        executor=ThreadPoolExecutor(max_workers=1)
        before=path.stat();f=uproot.open(path,decompression_executor=executor,interpretation_executor=executor);t=f['preselection']
        filehist=np.zeros_like(h);filehist2=np.zeros_like(h);cutflow={}
        for name in ('vertex_cutflow_h','event_cutflow_h'):
            cutflow[name]=dict(labels=f[name].axis().labels(),counts=f[name].values().tolist())
        for a in t.iterate(list(branches.values()),step_size=100000,library='np'):
            mass=np.asarray(a[branches['mass']],float);w=np.asarray(a['weight'],float);truth=np.asarray(a[branches['truth']],float)
            typ=a[branches['type']];n=len(mass);stats['entries']+=n
            if np.any(w<0):raise ValueError('Negative signal weights require a separate template prescription')
            good=np.isfinite(mass)&np.isfinite(w)&(w>0)
            stats['nonfinite']+=int(np.sum(~np.isfinite(mass)|~np.isfinite(w)));stats['nonpositive_weight']+=int(np.sum(w<=0))
            stats['wrong_type']+=int(np.sum(typ!='TargetConstrained'))
            if np.any(typ!='TargetConstrained'):raise ValueError('Unexpected vertex type')
            psum=sum(np.sqrt(sum(np.asarray(a[branches[p+axis]],float)**2 for axis in ('px','py','pz'))) for p in ('ele','pos'))
            stats['psum_below_2p8']+=int(np.sum(psum<2.8-1e-6))
            stats['smear_nonunit']+=int(np.sum((abs(a['ele_p_smear_ratio']-1)>1e-7)|(abs(a['pos_p_smear_ratio']-1)>1e-7)))
            truth_min=min(truth_min,float(np.min(truth)));truth_max=max(truth_max,float(np.max(truth)))
            stats['selected']+=int(good.sum());stats['underflow']+=int(np.sum(mass[good]<edges[0]));stats['overflow']+=int(np.sum(mass[good]>=edges[-1]))
            counts=np.histogram(mass[good],edges,weights=w[good])[0];squares=np.histogram(mass[good],edges,weights=w[good]**2)[0]
            h+=counts;h2+=squares;filehist+=counts;filehist2+=squares;blocks.append(counts)
            tsum+=float(w[good].sum());tsum2+=float(np.sum(w[good]**2));partindex+=1
        after=path.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:raise ValueError('Source changed during extraction')
        digest=sha(path)
        files.append(dict(path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=digest,entries=t.num_entries,
                          tree_key=str(t.object_path),cutflows=cutflow))
        np.savez_compressed(OUT/'histograms'/f'm{m:03d}_{path.stem}.npz',edges_GeV=edges,sumw=filehist,sumw2=filehist2)
        f.close()
    metadata=dict(mass_MeV=m,campaign='2021',source_selection='v13 ap_signal_prompt_smeared; stored selected TargetConstrained mass',
                  reconstruction_mass_branch=branches['mass'],extra_smearing=False,extra_cuts=False,stats=stats,
                  truth_mass_min_GeV=truth_min,truth_mass_max_GeV=truth_max,sumw=tsum,sumw2=tsum2,neffective=tsum**2/tsum2,
                  files=files,source_selection_match='TC and upstream psum labels agree; full v13/v16 equivalence not established')
    np.savez_compressed(OUT/'histograms'/f'm{m:03d}.npz',edges_GeV=edges,sumw=h,sumw2=h2,blocks=np.array(blocks),metadata=json.dumps(metadata))
    (OUT/'histograms'/f'm{m:03d}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with uproot.recreate(OUT/'histograms'/f'ap{m}MeV_reconstructed_mass.root') as root:
        root['reconstructed_mass']=(h,edges);root['sumw2_by_mass']=(h2,edges)
    rows.append({k:v for k,v in metadata.items() if k!='files'})
    print('completed',m,'entries',stats['entries'],'seconds',round(time.time()-start,1),flush=True)
(OUT/'qa/extraction_summary.json').write_text(json.dumps(dict(samples=rows,seconds=time.time()-start,complete=True),indent=2)+'\n')
