"""Conditional residual-Q audit; fixed archived kernels, no optimization or new data.

One process, one BLAS thread. Reuses paired 256 complete Poisson spectra already
in v5.8.2; all new products remain below this script's directory.
"""
import os, sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import csv,json,hashlib,time,shutil
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.stats import t,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
START=time.monotonic()
DEADLINE=START+900
for sub in ('inputs','results','figures'):(HERE/sub).mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def guard():
    if time.monotonic()>DEADLINE or (HERE.parent/'STOP').exists():raise RuntimeError('Bounded residual audit stopped')
sources={
    'spectrum_2021.npz':ROOT/'study_results/v5p0p5_analysis_note_20260916/inputs/spectrum_2021.npz',
    'null_2021.npz':ROOT/'study_results/v5p8p2_nominal_gp_significance_20260917/inputs/null_2021.npz',
    'slide50_field_2021.npz':ROOT/'study_results/v5p8p5p3_raw_significance_20260921/inputs/fields/2021.npz',
}
manifest_path=HERE/'inputs/source_hashes.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for name,p in sources.items():
    target=HERE/'inputs'/name
    if not target.exists():shutil.copy2(p,target)
    if name in manifest:assert sha(target)==manifest[name]['sha256']
    if p.exists():assert sha(p)==sha(target)
    manifest.setdefault(name,{'source':str(p.relative_to(ROOT)),'sha256':sha(target)})
original_slide=HERE/'inputs/slide14_original_scan.csv'
original_source=ROOT/'output/slides/unblind_meeting_RCmeet_20260922/science/assets/2021_conditional_residual_scan.csv'
if not original_slide.exists() and original_source.exists():shutil.copy2(original_source,original_slide)
if original_slide.exists():manifest.setdefault(original_slide.name,{'source':str(original_source.relative_to(ROOT)),'sha256':sha(original_slide)})
savejson(manifest_path,manifest)
d=dict(np.load(HERE/'inputs/spectrum_2021.npz'));source=dict(np.load(HERE/'inputs/null_2021.npz'))
B=source['truth'];toys=source['counts'];x=d['x'];NTOYS=len(toys)
assert np.array_equal(source['observed'],d['n']) and np.array_equal(source['edges_GeV'],d['edges'])
identity_path=HERE/'inputs/source_identity.json'
identity=json.loads(identity_path.read_text()) if identity_path.exists() else {}
bridge_path=ROOT/'study_results/v5p8p5p3_raw_significance_20260921/inputs/prior_review_results/physics_sources.npz'
other_path=ROOT/'study_results/v5p8p2_nominal_gp_significance_20260917/inputs/spectrum_2021.npz'
if bridge_path.exists():
    bridge=np.load(bridge_path);assert np.array_equal(B,bridge['native10_gp'])
    identity.update(slide50_source_equal=True,bridge_source_sha256=sha(bridge_path),truth_array_sha256=hashlib.sha256(B.tobytes()).hexdigest())
if other_path.exists():
    other=np.load(other_path);assert all(np.array_equal(d[k],other[k]) for k in d)
    identity.update(v582_spectrum_all_arrays_equal=True,v582_spectrum_sha256=sha(other_path))
assert identity.get('slide50_source_equal') and identity.get('v582_spectrum_all_arrays_equal')
assert identity['truth_array_sha256']==hashlib.sha256(B.tobytes()).hexdigest()
savejson(identity_path,identity)

class State:
    def __init__(self,j):
        self.mass=float(d['masses'][j]);self.mask=np.abs(x-self.mass/1000)<=2.25*d['sigma'][j];self.keep=~self.mask
        self.n=int(self.mask.sum());assert not np.any(self.mask&self.keep)
        const,ls=float(d['const'][j]),float(d['ls'][j])
        def kernel(a,b):return const*np.exp(-.5*((np.log(a)[:,None]-np.log(b)[None,:])/ls)**2)
        self.K=kernel(x[self.keep],x[self.keep]);self.Kqt=kernel(x[self.mask],x[self.keep]);self.Kqq=kernel(x[self.mask],x[self.mask])
    def predict(self,counts,derivative=False):
        y=counts[self.keep];assert np.all(y>0)
        K=self.K.copy();K.flat[::len(K)+1]+=1/y
        L=cholesky(K,lower=True,check_finite=False);co=cho_solve((L,True),np.log(y),check_finite=False)
        q=solve_triangular(L,self.Kqt.T,lower=True,check_finite=False)
        cl=self.Kqq-q.T@q;cl=(cl+cl.T)/2
        b=np.exp(self.Kqt@co+.5*np.maximum(np.diag(cl),0))
        C=np.outer(b,b)*np.expm1(np.clip(cl,-40,40));C=(C+C.T)/2
        if not derivative:return b,C
        H=cho_solve((L,True),self.Kqt.T,check_finite=False).T
        # Both alpha=1/y and the lognormal mean correction depend on sidebands.
        J=b[:,None]*(H*(1/y+co/y**2)[None,:]-.5*(np.diag(cl)>0)[:,None]*H**2/y[None,:]**2)
        return b,C,J

analytic=[];states={};fd=[]
for j,m in enumerate(d['masses']):
    guard();s=State(j);b,C,J=s.predict(B,True);V=np.diag(b)+C;F=(cholesky(V,lower=True),True)
    delta=B[s.mask]-b;P=np.diag(B[s.mask]);T=(J*B[s.keep])@J.T;W=P+T
    invW=cho_solve(F,W);invP=cho_solve(F,P);ivd=cho_solve(F,delta)
    variance=float(np.trace(invW));frozen_variance=float(np.trace(invP));bias=float(delta@ivd)
    bo,Co=s.predict(d['n']);ro=d['n'][s.mask]-bo;Vo=np.diag(bo)+Co
    qo=float(ro@cho_solve((cholesky(Vo,lower=True),True),ro));qof=float(ro@cho_solve(F,ro))
    row=dict(mass_MeV=float(m),Nbin=s.n,expected_frozen_noise_per_bin=frozen_variance/s.n,
             expected_sideband_noise_per_bin=float(np.trace(cho_solve(F,T)))/s.n,
             expected_refit_noise_per_bin=variance/s.n,deterministic_bias_Q_per_bin=bias/s.n,
             expected_refit_Q_per_bin=(variance+bias)/s.n,expected_frozen_Q_per_bin=(frozen_variance+bias)/s.n,
             expected_Q_sd_linear_gaussian=float(np.sqrt(2*np.trace(invW@invW)+4*ivd@W@ivd))/s.n,
             observed_Q_per_bin=qo/s.n,observed_Q_frozen_V_per_bin=qof/s.n,
             covariance_mismatch_from_unity=variance/s.n-1,
             min_V_eigenvalue=float(np.linalg.eigvalsh(V).min()),max_delta_over_sqrt_B=float(np.max(np.abs(delta)/np.sqrt(B[s.mask]))))
    analytic.append(row);states[float(m)]=(s,b,C,J,V,F,delta)
    if m in (60,78,120,220):
        direction=np.zeros(len(B));direction[s.keep]=np.random.default_rng(int(m)).normal(size=s.keep.sum())*np.sqrt(B[s.keep]);h=.1
        bp,_=s.predict(B+h*direction);bm,_=s.predict(B-h*direction);num=(bp-bm)/(2*h);ana=J@direction[s.keep]
        fd.append(dict(mass_MeV=float(m),relative_L2_error=float(np.linalg.norm(num-ana)/np.linalg.norm(ana)),maximum_absolute_error=float(np.max(np.abs(num-ana))),step_sigma=h))
csvwrite(HERE/'results/analytic_scan.csv',analytic);csvwrite(HERE/'results/jacobian_checks.csv',fd)
assert max(r['relative_L2_error'] for r in fd)<.003,fd
print('Analytic scan complete',json.dumps({'min_expectation':min(r['expected_refit_Q_per_bin'] for r in analytic),'max_expectation':max(r['expected_refit_Q_per_bin'] for r in analytic),'max_bias_term':max(r['deterministic_bias_Q_per_bin'] for r in analytic),'elapsed_s':time.monotonic()-START}),flush=True)

# Broad anchors plus extrema and nearest expected-Q=1 crossings; no selection on toy outcomes.
anchors={50,56,60,65,71,75,78,90,100,110,120,160,180,200,220,250}
for key in ('expected_refit_Q_per_bin','deterministic_bias_Q_per_bin','expected_refit_noise_per_bin'):
    anchors.add(int(max(analytic,key=lambda r:r[key])['mass_MeV']));anchors.add(int(min(analytic,key=lambda r:r[key])['mass_MeV']))
cross=[]
for a,b in zip(analytic[:-1],analytic[1:]):
    if (a['expected_refit_Q_per_bin']-1)*(b['expected_refit_Q_per_bin']-1)<0:cross.append(int(a['mass_MeV']))
if cross:
    anchors.update(np.array(cross)[np.unique(np.linspace(0,len(cross)-1,min(4,len(cross))).round().astype(int))].tolist())
display_anchors=sorted(anchors)
anchors=[int(m) for m in d['masses']]
savejson(HERE/'results/anchor_protocol.json',{'anchors_MeV':anchors,'display_anchors_MeV':display_anchors,'selection':'Exact controls cover all 201 integer masses; display anchors use broad anchors and analytic extrema/crossings, with no toy-based selection','N_paired_spectra':NTOYS,'reused':True})
summary=[];paired=[];means=[];distributions={}
critical=float(t.ppf(.975,NTOYS-1))
def summarize(values):
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1));se=sd/np.sqrt(NTOYS)
    return dict(mean=mean,sd=sd,mean_se=se,mean_CI95_low=mean-critical*se,mean_CI95_high=mean+critical*se,q025=float(np.quantile(values,.025)),median=float(np.median(values)),q975=float(np.quantile(values,.975)))
for m in anchors:
    guard();s,b,C,J,V,F,delta=states[float(m)];arr={k:[] for k in ('frozen_prediction','frozen_prediction_debiased','linear_refit','linear_refit_debiased','exact_refit_fixedV','exact_refit_fixedV_debiased','exact_refit_adaptiveV','exact_refit_adaptiveV_debiased')};pred=[];residuals=[]
    def q(r,fac=F):return float(r@cho_solve(fac,r,check_finite=False))/s.n
    for i,n in enumerate(toys):
        guard();r0=n[s.mask]-b;rl=r0-J@(n[s.keep]-B[s.keep]);bt,Ct=s.predict(n);rt=n[s.mask]-bt;Ft=(cholesky(np.diag(bt)+Ct,lower=True,check_finite=False),True)
        for k,r,fac in [('frozen_prediction',r0,F),('frozen_prediction_debiased',r0-delta,F),('linear_refit',rl,F),('linear_refit_debiased',rl-delta,F),('exact_refit_fixedV',rt,F),('exact_refit_fixedV_debiased',rt-delta,F),('exact_refit_adaptiveV',rt,Ft),('exact_refit_adaptiveV_debiased',rt-delta,Ft)]:arr[k].append(q(r,fac))
        pred.append(bt);residuals.append(rt)
    for k in arr:arr[k]=np.array(arr[k]);distributions[f'm{m}_{k}']=arr[k]
    expected=next(r for r in analytic if r['mass_MeV']==m)
    for k,v in arr.items():
        r=dict(mass_MeV=m,Nbin=s.n,N_toys=NTOYS,control=k,**summarize(v));summary.append(r)
    for name,aa,bb in [('refit_minus_frozen','exact_refit_fixedV','frozen_prediction'),('exact_minus_linear','exact_refit_fixedV','linear_refit'),('adaptiveV_minus_fixedV','exact_refit_adaptiveV','exact_refit_fixedV'),('bias_removal_effect','exact_refit_fixedV','exact_refit_fixedV_debiased')]:paired.append(dict(mass_MeV=m,N_toys=NTOYS,contrast=name,**summarize(arr[aa]-arr[bb])))
    pred=np.array(pred);residuals=np.array(residuals);jensen=pred.mean(axis=0)-b;emp_delta=B[s.mask]-pred.mean(axis=0)
    predcov=np.cov(pred,rowvar=False,ddof=1);rescov=np.cov(residuals,rowvar=False,ddof=1);resmean=residuals.mean(axis=0)
    k=int(np.count_nonzero(arr['exact_refit_adaptiveV']>=expected['observed_Q_per_bin']))
    means.append(dict(mass_MeV=m,N_toys=NTOYS,analytic_noise_Q_per_bin=expected['expected_refit_noise_per_bin'],analytic_bias_Q_per_bin=expected['deterministic_bias_Q_per_bin'],toy_prediction_covariance_noise_Q_per_bin=float(np.trace(cho_solve(F,np.diag(B[s.mask])+predcov)))/s.n,toy_expected_residual_mean_bias_Q_per_bin=q(emp_delta),jensen_shift_Q_per_bin=q(jensen),sample_residual_mean_Q_per_bin=q(resmean),sample_residual_covariance_trace_per_bin=float(np.trace(cho_solve(F,rescov)))/s.n,empirical_fixedV_Q_identity=(float(np.trace(cho_solve(F,rescov)))*(NTOYS-1)/NTOYS+float(resmean@cho_solve(F,resmean)))/s.n,observed_Q_per_bin=expected['observed_Q_per_bin'],observed_exceedances=k,conditional_tail_add_one=(k+1)/(NTOYS+1),conditional_tail_CP95_low=0. if k==0 else float(beta.ppf(.025,k,NTOYS-k+1)),conditional_tail_CP95_high=1. if k==NTOYS else float(beta.ppf(.975,k+1,NTOYS-k))))
    if m in display_anchors:print(f'Anchor {m:g}: Elinear={expected["expected_refit_Q_per_bin"]:.4f}, exact={arr["exact_refit_adaptiveV"].mean():.4f}, bias={expected["deterministic_bias_Q_per_bin"]:.4g}, noise={expected["expected_refit_noise_per_bin"]:.4f}, elapsed={time.monotonic()-START:.1f}s',flush=True)
    csvwrite(HERE/'results/poisson_controls.csv',summary);csvwrite(HERE/'results/paired_contrasts.csv',paired);csvwrite(HERE/'results/empirical_decomposition.csv',means)
np.savez_compressed(HERE/'results/paired_Q_arrays.npz',**distributions)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def plot_save(fig,name):
    fig.tight_layout();fig.savefig(HERE/'figures'/f'{name}.pdf');fig.savefig(HERE/'figures'/f'{name}.png',dpi=180);plt.close(fig)
mass=np.array([r['mass_MeV'] for r in analytic])
fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
for col,label,color in [('observed_Q_per_bin','Observed residual diagnostic','#222222'),('expected_refit_Q_per_bin','Fixed-source expectation (linear)','#166c97'),('expected_refit_noise_per_bin','Repeated-sampling variance term','#559565'),('deterministic_bias_Q_per_bin','Deterministic prediction-bias term','#b36b2d')]:axs[0].plot(mass,[r[col] for r in analytic],label=label,c=color,lw=1.6)
sr=[r for r in summary if r['control']=='exact_refit_adaptiveV'];axs[0].plot([r['mass_MeV'] for r in sr],[r['mean'] for r in sr],c='#166c97',lw=.8,ls=':',label='256 paired Poisson spectra: mean ±95% MC CI');axs[0].fill_between([r['mass_MeV'] for r in sr],[r['mean_CI95_low'] for r in sr],[r['mean_CI95_high'] for r in sr],color='#166c97',alpha=.16)
axs[0].axhline(1,c='.55',ls='--',lw=.8);axs[0].set_ylabel('Q / Nbin');axs[0].legend(frameon=False,fontsize=8,ncol=2)
for col,label,color in [('expected_frozen_noise_per_bin','Window Poisson term','#777777'),('expected_sideband_noise_per_bin','Sideband-estimator sampling term','#7a4790'),('expected_refit_noise_per_bin','Sum: repeated-sampling variance','#559565')]:axs[1].plot(mass,[r[col] for r in analytic],label=label,c=color,lw=1.6)
axs[1].axhline(1,c='.55',ls='--',lw=.8);axs[1].set(xlabel='Test mass [MeV]',ylabel='Trace contribution / Nbin');axs[1].legend(frameon=False,fontsize=8,ncol=1)
plot_save(fig,'residual_Q_expectation_decomposition')
fig,axs=plt.subplots(1,2,figsize=(10,4))
for control,label,color in [('frozen_prediction','Frozen prediction','#777777'),('exact_refit_fixedV','Sideband refit, fixed V','#166c97'),('exact_refit_fixedV_debiased','Sideband refit, deterministic bias removed','#559565')]:
    rr=[r for r in summary if r['control']==control and r['mass_MeV'] in display_anchors];axs[0].errorbar([r['mass_MeV'] for r in rr],[r['mean'] for r in rr],yerr=[critical*r['mean_se'] for r in rr],label=label,c=color,fmt='o-',ms=3,capsize=2,lw=1)
for contrast,label,color in [('exact_minus_linear','Exact − linear replay','#166c97'),('adaptiveV_minus_fixedV','Adaptive V − fixed V','#b36b2d')]:
    rr=[r for r in paired if r['contrast']==contrast and r['mass_MeV'] in display_anchors];axs[1].errorbar([r['mass_MeV'] for r in rr],[r['mean'] for r in rr],yerr=[critical*r['mean_se'] for r in rr],label=label,c=color,fmt='o-',ms=3,capsize=2,lw=1)
axs[0].set(xlabel='Test mass [MeV]',ylabel='Paired-Poisson mean Q / Nbin');axs[1].set(xlabel='Test mass [MeV]',ylabel='Paired change in Q / Nbin');axs[1].axhline(0,c='.5',lw=.8)
for a in axs:a.legend(frameon=False,fontsize=8)
plot_save(fig,'residual_Q_paired_controls')
protocol={'version':'5.9.5','complete':True,'analytic_masses':len(analytic),'exact_control_masses':anchors,'paired_spectra':NTOYS,'new_toys':0,'new_optimization':0,'new_signal_fits':0,'GP_replays_exact_control':len(anchors)*NTOYS,'source_equals_slide50_nominal_GP':True,'source_spectrum_equals_v505_all_arrays':True,'nominal_support_MeV':[36,300],'actual_whole_bin_edges_MeV':(d['edges'][[0,-1]]*1000).tolist(),'mass_grid_MeV':[50,250,1],'exclusion_half_width_sigma':2.25,'Nbin_is_effective_dof':False,'J_includes_alpha_and_lognormal_variance_correction':True,'max_jacobian_relative_error':max(r['relative_L2_error'] for r in fd),'analytic_approximation':'First order sideband-estimator response; V frozen at source prediction. Toy adaptive-V replay is separately measured.','counterfactual':'Subtract delta=B_window-b(B_training) from each paired residual; keep physical data/source unchanged. Diagnostic only.','conditional_only':True,'source_estimation_uncertainty_propagated':False,'rare_tail_or_global_calibration':False,'elapsed_seconds':time.monotonic()-START,'BLAS_threads':1,'processes':1}
savejson(HERE/'results/protocol.json',protocol)
print(json.dumps(protocol,indent=2),flush=True)
