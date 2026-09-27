"""Conditional source-transfer diagnostic; fixed ±2.25 sigma extraction.
No calibration, selection equivalence, source independence or exposure claim.
Reads pinned v5.8.2 engine and high-psum ROOT source; writes only this study.
"""
from pathlib import Path
import os, sys, time, json, hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
P=B/'inputs/v5p8p2_nominal_gp_significance_20260917'
if not P.exists():P=ROOT/'study_results/v5p8p2_nominal_gp_significance_20260917'
sys.path.insert(0,str(P/'scripts'))
import engine as E
import numpy as np,pandas as pd,uproot
R=B/'results';F=B/'figures'
DEADLINE=time.time()+480

def check():
    if (B/'STOP').exists() or time.time()>=DEADLINE:
        raise SystemExit('Stopped by resource guard')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
check()
source=B/'inputs/physics_high_psum_1pct.root'
if not source.exists():source=Path('/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM_v7.root')
key='preselection/h_invM_psumgt2p8_8000'
assert sha(source)=='412026fc37b65066d906c1d4acab524d274ee9d215960fb65a0b728d70c157a8'
with uproot.open(source) as f:
    h=f[key]; y,edges=h.to_numpy()
d=E.C.DATA['2021'];x=d['x']
idx=np.array([int(np.argmin(abs(edges-v))) for v in d['edges']])
assert np.max(abs(edges[idx]-d['edges']))<1e-12
n1=np.diff(np.r_[0.,np.cumsum(y)][idx])
c,l=E.C.kernel_state('2021',76.)
b1,cov1=E.C.predict(x,n1,np.zeros(len(x),bool),c,l,query=x)
b10=np.load(P/'inputs/null_2021.npz')['truth']
assert np.all(b1>0)
means={'high1_native':b1,'high1_times10':10*b1,'native10':b10}
np.savez_compressed(R/'physics_sources.npz',x=x,edges_GeV=d['edges'],high1_observed=n1,native10_observed=d['n'],high1_gp=b1,high1_gp_cov=cov1,high1_times10=10*b1,native10_gp=b10)
manifest=dict(source_path=str(source),source_histogram=key,source_sha256=sha(source),source_native_bins=len(y),source_native_total=float(y.sum()),rebinned_support_MeV=[float(d['edges'][0]*1000),float(d['edges'][-1]*1000)],rebinned_bin_count=len(n1),source_kernel_anchor_MeV=76,const=c,ls=l,source_training_mask='All support bins included',analysis_blind_half_width_sigma=2.25,scale=10,source_selection_equivalent_verified=False,exact_exposure_ratio_verified=False,TC_membership_verified=False,event_overlap_verified=False,source_estimation_uncertainty_propagated=False,interpretation='Conditional fixed-source response comparison; not calibrated significance or independent validation',source_support_counts=float(n1.sum()),native10_support_counts=float(d['n'].sum()),support_count_ratio=float(n1.sum()/d['n'].sum()),parent_engine_hashes={str(p.relative_to(P)):sha(p) for p in (P/'scripts').glob('*.py')},prior_selection_manifest_sha256=sha(B/'inputs/physics_prior_selection_manifest.json'))
(R/'physics_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[]
for mass in np.arange(50.,251.):
    check();ctx=E.Context('2021',float(mass))
    for name,truth in means.items():
        part=ctx.predict(truth,derivatives=True);f=E.fit([part]);truths={yy:np.load(P/f'inputs/null_{yy}.npz')['truth'] for yy in E.YEARS};truths['2021']=truth
        response=E.response([part],f,truths)
        rows.append(dict(source=name,mass_MeV=mass,a=f['r'],s=float(np.linalg.norm(response)),fitted_yield_coordinate=f['f']['A'],fit_score=f['score']))
    if int(mass)%25==0:print('deterministic',mass,flush=True)
pd.DataFrame(rows).to_csv(R/'physics_source_scan.csv',index=False,float_format='%.17g')
anchors=[65.,76.,90.,92.,120.,180.];N=64;toyrows=[];saved={}
for name in ['high1_times10','native10']:
    truth=means[name];rng=np.random.default_rng(np.random.SeedSequence([58520260921,0 if name=='high1_times10' else 1]));toys=rng.poisson(truth,(N,len(truth))).astype(float)
    saved[name+'_counts']=toys;saved[name+'_truth']=truth
    for mass in anchors:
        check();ctx=E.Context('2021',mass);roots=[];scores=[]
        for counts in toys:
            check();fit=E.fit([ctx.predict(counts)]);roots.append(fit['r']);scores.append(fit['score'])
        rec=next(v for v in rows if v['source']==name and v['mass_MeV']==mass);roots=np.array(roots);z=(roots-rec['a'])/rec['s'];saved[name+f'_r_{int(mass)}']=roots
        toyrows.append(dict(source=name,mass_MeV=mass,N=N,a=rec['a'],s=rec['s'],toy_mean=float(roots.mean()),toy_sd=float(roots.std(ddof=1)),centered_mean=float(z.mean()),centered_sd=float(z.std(ddof=1)),max_fit_score=max(scores)))
        print('toys',name,mass,'mean',z.mean(),'sd',z.std(ddof=1),flush=True)
pd.DataFrame(toyrows).to_csv(R/'physics_source_toys.csv',index=False,float_format='%.17g')
np.savez_compressed(R/'physics_source_toys.npz',**saved)
summary=dict(completed=True,deterministic_states=len(rows),Poisson_fits=len(toyrows)*N,independent_scans_per_source=N,toy_masses_MeV=anchors,seed_base=58520260921,conditional_only=True,omissions=['Source estimation uncertainty','Verified exposure and selection transfer','Event overlap','Scan-global calibration','Rare tails'],max_fit_score=max([r['fit_score'] for r in rows]+[r['max_fit_score'] for r in toyrows]))
(R/'physics_source_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary,flush=True)
