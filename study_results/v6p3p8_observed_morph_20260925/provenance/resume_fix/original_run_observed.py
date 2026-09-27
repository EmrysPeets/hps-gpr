#!/usr/bin/env python3
"""Observed, conditional v16 neighboring-template scan and selected-point toys."""
from pathlib import Path
import os,sys,json,hashlib,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'inputs/v6p1/scripts'))
import common as C
from observed_templates import BANK
import numpy as np
import pandas as pd
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.special import ndtr
from scipy.stats import beta

D=C.DATA['2021'];TRUTH=np.load(B/'inputs/null_2021.npz')['truth']
MASSES=tuple(range(60,241));POLICIES=('morph_starter','gaussian_baseline','gaussian_starter')
GRID=(0.,.5,1.,2.,3.,4.,5.,6.,8.,10.,12.,16.)
NNULL=1000;NCAL=200;SEED=638250925

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv(path,frame):
    tmp=Path(path).with_suffix('.tmp');frame.to_csv(tmp,index=False,float_format='%.17g');tmp.replace(path)
def cp(k,n):return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k)))

class Context:
    def __init__(self,mass,policy):
        self.mass,self.policy=mass,policy
        self.center,self.width=BANK.parameters(mass)
        self.gaussian_center=mass-3.2243308692909953-2.213992811446465*np.log(mass/150.)
        self.sigma_ref=C.sigma('2021',mass)*1000
        if policy=='gaussian_baseline':lo=self.gaussian_center-2.25*self.sigma_ref;hi=self.gaussian_center+2.25*self.sigma_ref
        else:lo=self.center-4*self.width;hi=self.center+3*self.width
        self.fit=(D['x']*1000>=lo)&(D['x']*1000<=hi);self.guard=self.fit.copy()
        self.categories=BANK.categories(mass,D['edges']*1000)
        self.gaussian_probability=np.diff(ndtr((D['edges']*1000-self.gaussian_center)/self.sigma_ref))
        self.probability=self.categories[1:-1] if policy=='morph_starter' else self.gaussian_probability
        self.const,self.ls=C.kernel_state('2021',mass)
        self.K=C.kernel(D['x'][~self.guard],D['x'][~self.guard],self.const,self.ls)
        self.Kqt=C.kernel(D['x'][self.fit],D['x'][~self.guard],self.const,self.ls)
        self.Kqq=C.kernel(D['x'][self.fit],D['x'][self.fit],self.const,self.ls)
        assert self.fit.sum()>3 and np.sum(D['x']<lo/1000)>=3 and np.sum(D['x']>hi/1000)>=3
        assert np.all(self.categories>=0) and abs(self.categories.sum()-1)<1e-12

    def prediction(self,counts,full=False):
        n=np.asarray(counts,float)[~self.guard];pos=n>0
        target=np.zeros_like(n);target[pos]=np.log(n[pos]);alpha=np.ones_like(n);alpha[pos]=1/n[pos]
        K=self.K.copy();K.flat[::len(K)+1]+=alpha
        L=cholesky(K,lower=True,check_finite=False)
        Kqt=C.kernel(D['x'],D['x'][~self.guard],self.const,self.ls) if full else self.Kqt
        Kqq=C.kernel(D['x'],D['x'],self.const,self.ls) if full else self.Kqq
        mu=Kqt@cho_solve((L,True),target,check_finite=False);v=solve_triangular(L,Kqt.T,lower=True,check_finite=False)
        cov=Kqq-v.T@v;cov=.5*(cov+cov.T);b=np.exp(mu+.5*np.maximum(np.diag(cov),0))
        covariance=np.outer(b,b)*np.expm1(np.clip(cov,-40,40))
        return b,covariance

    def fit_counts(self,counts,limit=False,save=None):
        b,cov=self.prediction(counts);L,diag=C.factor_cov(cov,b);n=np.asarray(counts)[self.fit]
        errors=[]
        for tol in (2e-7,2e-9):
            try:
                model=C.OneSignalProfile(b,L,self.probability[self.fit],score_tolerance=tol)
                if limit:
                    r=model.limit(n,alpha=.1,details=True);free=r.pop('free');null=r.pop('null');trace=r.pop('trace')
                    assert r['ok'] and abs(r['cls']-.1)<2e-6 and r['max_score']<3e-5
                else:
                    free=model.fit(n);null=model.fit(n,fixed=0,initial=free['theta']);trace=[]
                    q=2*(null['nll']-free['nll']);assert q>=-2e-6
                    signed=float(np.sign(free['A'])*np.sqrt(max(0,q)))
                    r=dict(Ahat=free['A'],sigma_A=free['sigma'],signed_r=signed,Z0=max(0,signed),p0_fixed_mass=float(ndtr(-max(0,signed))),nll_null=null['nll'],nll_free=free['nll'],max_score=max(free['score'],null['score']),min_lambda=min(free['min_lambda'],null['min_lambda']))
                assert r['max_score']<3e-5 and r['min_lambda']>0
                _,_,H,_=model._objective(free['z'],n,model.Jfree,model.b,model.penfree)
                unit=np.zeros(len(H));unit[0]=1
                variance=float(cho_solve((cholesky(H,lower=True),True),unit)[0]);sigma=model.scale*np.sqrt(variance)
                assert variance>0 and abs(sigma/free['sigma']-1)<1e-9
                r.update(q0=max(0,r['signed_r'])**2,p_deficit_asymptotic=float(ndtr(r['signed_r'])) if r['signed_r']<0 else 1.,fit_valid=True,failure_reason='',covariance_load=diag['load'],nuisance_rank=diag['rank'])
                if save is not None:
                    fullb,fullcov=self.prediction(counts,full=True)
                    np.savez_compressed(save,edges_GeV=D['edges'],x_GeV=D['x'],counts=np.asarray(counts),fit_mask=self.fit,guard_mask=self.guard,
                        signal_probability=self.probability,full_MC_categories=self.categories,prefit_GP_mean=fullb,prefit_GP_covariance=fullcov,
                        fit_prefit_mean=b,fit_prefit_covariance=cov,fit_covariance_factor=L,
                        profiled_background_only=null['bfit'],profiled_background_signed=free['bfit'],
                        profiled_signed_signal=free['A']*self.probability[self.fit],profiled_signed_total=free['lam'],
                        profiled_bounded_total=free['lam'] if free['A']>=0 else null['lam'],
                        nuisance_null=null['theta'],nuisance_signed=free['theta'],fit_counts=n,
                        fit_summary_json=json.dumps(r),profile_trace_json=json.dumps(trace))
                return r
            except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        return dict(fit_valid=False,failure_reason='; '.join(errors))

    def geometry(self):
        i=np.flatnonzero(self.fit);p=self.categories[1:-1]
        return dict(mass_MeV=self.mass,policy=self.policy,core_center_MeV=self.center,core_width_MeV=self.width,gaussian_center_MeV=self.gaussian_center,sigma_ref_MeV=self.sigma_ref,
            fit_low_MeV=D['edges'][i[0]]*1000,fit_high_MeV=D['edges'][i[-1]+1]*1000,fit_bins=int(self.fit.sum()),training_bins=int((~self.guard).sum()),
            template_fit_fraction=float(self.probability[self.fit].sum()),MC_fit_fraction=float(p[self.fit].sum()),MC_training_fraction=float(p[~self.guard].sum()),MC_outside_support_fraction=float(self.categories[0]+self.categories[-1]),
            gaussian_training_fraction=float(self.gaussian_probability[~self.guard].sum()),kernel_const=self.const,kernel_length_scale=self.ls,
            interpolation_anchors_json=json.dumps(BANK.neighbors(self.mass)),acceptance_transition=bool(self.mass<80))

def protocol():
    p=dict(version='6.3.8',masses=list(MASSES),policies=list(POLICIES),signal_model='All verified v16 TC anchors60..240; linearly interpolate core location,width and aligned full empirical CDF of nearest anchors; at anchor use exact direct MC; retain full selected denominator and outside-support categories.',
        geometry='morph and Gaussianstarter likelihood and GP exclusion both[-4,3] in interpolated core width; Gaussianbaseline original shifted center +/-2.25 archived analysis sigma',
        background='Archived mass-dependent log-GP kernel states; independent Poisson likelihood fit bins excluded from GP training',
        limits='Profiled bounded CLs90 with asymptotic sampling tails, fixed mass; no empirical calibration claim for dense curve',
        q0='max(signed_r,0)^2; signed_r=sign(Ahat)*sqrt(2*(NLL_background_only-NLL_signed_fit)); deficit uses negative signed_r',
        region_selection='Positive local maxima of morph q0, descending q0 with lower-mass ties; retain top3 with disjoint actual fitted-bin masks. Select deepest negative signed_r whose fitted bins overlap none of those regions. Endpoints eligible as local maxima; flat maxima represented by lower mass.',
        local_toys=NNULL,null_seed='SeedSequence([master,1,toy]); same Poisson GPmean-source background across selected masses and methods; recompute GP prediction and covariance each fit',
        calibration_toys=NCAL,calibration_grid=list(GRID),calibration_background_seed='SeedSequence([master,2,toy])',calibration_signal_seed='SeedSequence([master,3,mass,toy]); nested Poisson increments by grid strength; paired across methods',
        calibration_signal='Full production-morphed categories at selected mass; direct MC only at actual anchors; model assumption at intermediate masses',
        calibration_scale='s0 = Gaussianbaseline returned observed-Hessian error fitted to fixed GPmean background counts; computed independently of observed signal yield',
        selection_caveat='Local observed-region toy probabilities are conditional fixed-mass diagnostics; selected after scanning and not corrected for selection or the look-elsewhere effect.',
        rank_limit='p_A=(1+# calibration fitted yields<=observed fitted yield)/201; retain every grid A with p_A>0.1; store empty,holes,upper-grid censor flags',
        seed=SEED,script_sha256=sha(__file__),template_script_sha256=sha(B/'scripts/observed_templates.py'),archived_template_sha256=sha(B/'scripts/archived_templates.py'),
        observed_spectrum_sha256=sha(B/'inputs/v6p1/inputs/spectrum_2021.npz'),GPmean_source_sha256=sha(B/'inputs/null_2021.npz'))
    path=B/'provenance/protocol.json';p=json.loads(json.dumps(p))
    if path.exists():assert json.loads(path.read_text())==p,'Frozen protocol changed'
    else:write(path,p)
    return path

def select_regions(frame):
    q=frame[frame.policy=='morph_starter'].sort_values('mass_MeV').reset_index(drop=True);values=q.q0.to_numpy();candidates=[]
    for i,r in q.iterrows():
        if r.q0>0 and (i==0 or values[i]>values[i-1]) and (i==len(q)-1 or values[i]>=values[i+1]):candidates.append(r)
    candidates=sorted(candidates,key=lambda r:(-r.q0,r.mass_MeV));chosen=[];masks=[]
    for r in candidates:
        ctx=Context(int(r.mass_MeV),'morph_starter')
        if any(np.any(ctx.fit&mask) for mask in masks):continue
        chosen.append(dict(region=f'excess_{len(chosen)+1}',mass_MeV=int(r.mass_MeV),q0=float(r.q0),signed_r=float(r.signed_r),fit_low_MeV=float(r.fit_low_MeV),fit_high_MeV=float(r.fit_high_MeV)))
        masks.append(ctx.fit)
        if len(chosen)==3:break
    assert len(chosen)==3,'Fewer than three distinct positive regions'
    for _,r in q.sort_values(['signed_r','mass_MeV']).iterrows():
        ctx=Context(int(r.mass_MeV),'morph_starter')
        if r.signed_r<0 and not any(np.any(ctx.fit&mask) for mask in masks):
            chosen.append(dict(region='deficit',mass_MeV=int(r.mass_MeV),q0=float(r.q0),signed_r=float(r.signed_r),fit_low_MeV=float(r.fit_low_MeV),fit_high_MeV=float(r.fit_high_MeV)));break
    assert len(chosen)==4
    result=dict(selection_policy='Predeclared disjoint actual morph fit-bin windows; local extrema ranked on observed scan',regions=chosen,observed_selection_requires_global_correction=True)
    path=B/'results/selected_regions.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write(path,result)
    return chosen

def run_scan(pp):
    rows=[];templates=[];centers=[];widths=[];pg=[]
    for mass in MASSES:
        cats=BANK.categories(mass,D['edges']*1000);templates.append(cats);c,w=BANK.parameters(mass);centers.append(c);widths.append(w)
        for policy in POLICIES:
            ctx=Context(mass,policy)
            if policy=='gaussian_baseline':pg.append(ctx.probability)
            path=B/f'results/scan_checkpoints/m{mass:03d}_{policy}.json'
            if path.exists():
                item=json.loads(path.read_text());assert item['protocol_sha256']==sha(pp);r=item['row']
            else:
                r={**ctx.geometry(),**ctx.fit_counts(D['n'],limit=True)}
                write(path,dict(protocol_sha256=sha(pp),row=r))
            assert r['fit_valid'],str(r);rows.append(r)
        if mass%20==0:print('observed scan',mass,'complete',flush=True)
    frame=pd.DataFrame(rows);csv(B/'results/observed_scan.csv',frame)
    np.savez_compressed(B/'results/template_grid.npz',masses_MeV=np.array(MASSES),edges_GeV=D['edges'],anchors_MeV=BANK.anchors,full_MC_categories=np.array(templates),Gaussian_bin_probabilities=np.array(pg),core_centers_MeV=np.array(centers),core_widths_MeV=np.array(widths),observed_counts=D['n'])
    return frame

def selected_toys(regions,observed,pp):
    selected=[int(r['mass_MeV']) for r in regions];contexts={(m,p):Context(m,p) for m in selected for p in POLICIES}
    s0={m:contexts[m,'gaussian_baseline'].fit_counts(TRUTH)['sigma_A'] for m in selected}
    write(B/'results/selected_reference_scales.json',{str(m):dict(s0=v,definition='Asimov GPmean background Gaussianbaseline postfit Hessian error') for m,v in s0.items()})
    pieces=[]
    for cohort,total,zgrid in (('null',NNULL,(0.,)),('calibration',NCAL,GRID)):
        ns=1 if cohort=='null' else 2
        backgrounds=np.array([np.random.default_rng(np.random.SeedSequence([SEED,ns,i])).poisson(TRUTH) for i in range(total)])
        for mass in selected:
            cats=contexts[mass,'morph_starter'].categories
            for first in range(0,total,50):
                path=B/f'results/calibration_checkpoints/{cohort}_m{mass:03d}_t{first:04d}.csv';marker=path.with_suffix('.json')
                if marker.exists():
                    meta=json.loads(marker.read_text());assert meta['protocol_sha256']==sha(pp) and meta['rows_sha256']==sha(path) and meta['draws_sha256']==sha(path.with_suffix('.npz'))
                    pieces.append(pd.read_csv(path,float_precision='round_trip'));continue
                rows=[];draws=[]
                for toy in range(first,min(first+50,total)):
                    rng=np.random.default_rng(np.random.SeedSequence([SEED,3,mass,toy]));draw=np.zeros(len(cats),dtype=np.int64);prev=0.
                    for z in zgrid:
                        draw+=rng.poisson((z-prev)*s0[mass]*cats);prev=z;counts=backgrounds[toy]+draw[1:-1];draws.append(draw.copy())
                        for policy in POLICIES:
                            ctx=contexts[mass,policy];r=ctx.fit_counts(counts)
                            rows.append(dict(cohort=cohort,mass_MeV=mass,policy=policy,toy=toy,z=z,A_expected=z*s0[mass],background_hash=ahash(backgrounds[toy]),signal_hash=ahash(draw),counts_hash=ahash(counts),actual_full=int(draw.sum()),actual_fit=int(draw[1:-1][ctx.fit].sum()),actual_training=int(draw[1:-1][~ctx.guard].sum()),actual_outside_support=int(draw[0]+draw[-1]),**r))
                frame=pd.DataFrame(rows);csv(path,frame)
                np.savez_compressed(path.with_suffix('.npz'),backgrounds=backgrounds[first:first+50],signal_categories=np.array(draws),strengths=np.array(zgrid),toys=np.arange(first,min(first+50,total)),full_signal_probabilities=cats)
                write(marker,dict(protocol_sha256=sha(pp),rows_sha256=sha(path),draws_sha256=sha(path.with_suffix('.npz')),rows=len(frame),valid=int(frame.fit_valid.sum())))
                assert frame.fit_valid.all();pieces.append(frame)
            print('selected toys',cohort,mass,'complete',flush=True)
    allrows=pd.concat(pieces,ignore_index=True);csv(B/'results/selected_toy_rows.csv',allrows)
    summaries=[];rankrows=[]
    for mass in selected:
        for policy in POLICIES:
            obs=observed[(observed.mass_MeV==mass)&(observed.policy==policy)].iloc[0]
            q=allrows[(allrows.mass_MeV==mass)&(allrows.policy==policy)];null=q[q.cohort=='null'];cal=q[q.cohort=='calibration']
            if obs.signed_r>=0:k=int((null.q0>=obs.q0).sum());tail='upward q0'
            else:k=int((null.signed_r<=obs.signed_r).sum());tail='downward signed likelihood ratio'
            low,high=cp(k,len(null));rankp=(k+1)/(len(null)+1)
            ps=np.array([(1+np.count_nonzero(cal[cal.z==z].Ahat<=obs.Ahat))/(NCAL+1) for z in GRID]);accepted=np.flatnonzero(ps>.1);empty=len(accepted)==0
            rankrows.append(dict(mass_MeV=mass,policy=policy,s0=s0[mass],accepted_z_json=json.dumps([GRID[i] for i in accepted]),p_A_json=json.dumps(ps.tolist()),empty=empty,holes=False if empty else len(accepted)!=(accepted[-1]-accepted[0]+1),upper_grid_censored=bool(not empty and accepted[-1]==len(GRID)-1),largest_accepted_A=None if empty else GRID[accepted[-1]]*s0[mass],next_grid_A=None if empty or accepted[-1]==len(GRID)-1 else GRID[accepted[-1]+1]*s0[mass]))
            summaries.append(dict(mass_MeV=mass,policy=policy,observed_Ahat=obs.Ahat,observed_signed_r=obs.signed_r,observed_q0=obs.q0,p0_asymptotic=obs.p0_fixed_mass,p_deficit_asymptotic=obs.p_deficit_asymptotic,empirical_tail=tail,null_toys=len(null),tail_count=k,p_rank=rankp,tail_probability95_low=low,tail_probability95_high=high,background_only_yield_bias=float(null.Ahat.mean()),background_only_mean_SE=float(null.Ahat.std(ddof=1)/np.sqrt(len(null))),background_only_yield_SD=float(null.Ahat.std(ddof=1)),background_only_mean_signed_r=float(null.signed_r.mean()),background_only_SD_signed_r=float(null.signed_r.std(ddof=1)),observed_CLs90_model=obs.A90))
    csv(B/'results/selected_local_calibration.csv',pd.DataFrame(summaries));csv(B/'results/selected_rank_limits.csv',pd.DataFrame(rankrows));return allrows

def main():
    start=time.monotonic()
    pp=protocol();observed=run_scan(pp);regions=select_regions(observed)
    for r in regions:
        mass=r['mass_MeV']
        for policy in POLICIES:
            ctx=Context(mass,policy);saved=ctx.fit_counts(D['n'],limit=True,save=B/f'results/selected_fits/{r["region"]}_m{mass:03d}_{policy}.npz')
            assert saved['fit_valid']
    print('SELECTED',json.dumps(regions),flush=True)
    toys=selected_toys(regions,observed,pp)
    qa=dict(passed=bool(observed.fit_valid.all() and toys.fit_valid.all()),observed_scan_fits=len(observed),selected_toy_fits=len(toys),null_toys_per_mass_policy=NNULL,signal_calibration_toys_per_strength=NCAL,all_anchor_templates_equal_direct_MC=all(np.allclose(BANK.categories(m,D['edges']*1000),BANK.categories(m,D['edges']*1000,kind='direct'),rtol=0,atol=1e-12) for m in BANK.anchors),selected_regions=regions,protocol_sha256=sha(pp),runtime_seconds=time.monotonic()-start,scope='Observed fixed-mass asymptotic profile scan; conditional fixed-mass selected-region toy diagnostics; no global p-value or physical coupling exclusion.')
    assert qa['passed'] and qa['all_anchor_templates_equal_direct_MC'];write(B/'qa/numerical_validation.json',qa);print(json.dumps(qa,indent=2),flush=True)

if __name__=='__main__':main()
