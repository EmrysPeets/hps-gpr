"""Exact C=0 observed and coherent 256-toy scan, resumable at each mass.

Every toy retrains its mass-local GP mean. Only the signal-likelihood nuisance
covariance is omitted. Parent masks, kernels, signal units and source retained.
"""
from pathlib import Path
import sys,time,json,csv,hashlib,argparse
import fixed_core as F
import numpy as np
from scipy.stats import norm,beta

B=F.BASE;R=B/'results';CP=R/'checkpoints'
R.mkdir(exist_ok=True);CP.mkdir(exist_ok=True)
M=np.arange(19.,250.01,.5);SCOPES=(*F.YEARS,'shared','free')
NULL={y:dict(np.load(F.PARENT/f'inputs/null_{y}.npz')) for y in F.YEARS}
N=256
def writecsv(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jwrite(name,d):
    (R/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def massfile(m,phase):return CP/f'{phase}_m{int(round(m*2)):04d}.npz'
def active(m):return [y for y in F.YEARS if F.LIMITS[y][0]<=m<=F.LIMITS[y][1]]
def makestate(m,phase):
    F.check_stop();p=massfile(m,phase)
    if p.exists():return dict(np.load(p))
    ys=active(m);state={'mass':np.array(m),'years':np.array(ys)};parts=[];fits={}
    for y in ys:
        c=F.Context(y,m)
        counts=np.vstack([NULL[y]['observed'],NULL[y]['truth']]) if phase=='observed' else NULL[y]['counts']
        bb=[]
        for j,n in enumerate(counts):
            if j%16==0:F.check_stop()
            bb.append(c.predict(n))
        bb=np.array(bb);nn=counts[:,c.mask]
        fit=F.fit(nn,bb,c.S);fits[y]=fit
        parts.append((nn,bb,c.S))
        state[y+'_full_template_yield']=np.array(c.Sfull.sum())
        state[y+'_window_template_yield']=np.array(c.S.sum())
        state[y+'_window_bins']=np.array(c.mask.sum())
        for key,value in fit.items():state[y+'_'+key]=value
        if phase=='observed':state[y+'_b_observed']=bb[0]
    if len(parts)==1:shared=fits[ys[0]]
    else:shared=F.fit(np.concatenate([p[0] for p in parts],axis=1),np.concatenate([p[1] for p in parts],axis=1),np.concatenate([p[2] for p in parts]))
    for key,value in shared.items():state['shared_'+key]=value
    state['shared_full_template_yield']=sum(state[y+'_full_template_yield'] for y in ys)
    state['shared_window_template_yield']=sum(state[y+'_window_template_yield'] for y in ys)
    q=sum(fits[y]['q0'] for y in ys);p=F.free_p(q,len(ys))
    state.update(free_q0=q,free_nominal_local_p=p,free_nominal_local_Z=np.maximum(0,norm.isf(p)),
                 free_score=np.max([fits[y]['score'] for y in ys],axis=0),
                 free_minimum_lambda=np.min([fits[y]['minimum_lambda'] for y in ys],axis=0))
    np.savez_compressed(massfile(m,phase),**state)
    return state

def observed_rows():
    rows=[]
    for m in M:
        d=dict(np.load(massfile(m,'observed')));ys=active(m)
        for scope in (*ys,'shared','free'):
            r=None if scope=='free' else float(d[scope+'_r'][0])
            q=float(d[scope+'_q0'][0]);p=float(d[scope+'_nominal_local_p'][0])
            A=None if scope=='free' else float(d[scope+'_A'][0]);err=None if scope=='free' else float(d[scope+'_sigma_A'][0])
            row=dict(scope=scope,mass_MeV=m,active_datasets='+'.join(ys) if scope in ['shared','free'] else scope,
                     n_active=len(ys) if scope in ['shared','free'] else 1,r=r,q0=q,nominal_local_p=p,
                     nominal_local_Z=float(d[scope+'_nominal_local_Z'][0]),nominal_atom_zero=bool(q==0),
                     ordering_score=float(-np.log(p)) if scope=='free' else max(0,r),
                     conventional_local_p=None if scope=='free' else float(d[scope+'_conventional_local_p'][0]),
                     A_hat=A,A_error=err,epsilon2_hat=None if A is None else A*1e-8,
                     epsilon2_error=None if err is None else err*1e-8,
                     full_template_yield=None if A is None else float(A*d[scope+'_full_template_yield']),
                     full_template_yield_error=None if err is None else float(err*d[scope+'_full_template_yield']),
                     window_yield=None if A is None else float(A*d[scope+'_window_template_yield']),
                     window_yield_error=None if err is None else float(err*d[scope+'_window_template_yield']),
                     asimov_r=None if scope=='free' else float(d[scope+'_r'][1]),
                     asimov_q0=float(d[scope+'_q0'][1]),asimov_nominal_p=float(d[scope+'_nominal_local_p'][1]),
                     numerical_score=float(d[scope+'_score'][0]),minimum_lambda=float(d[scope+'_minimum_lambda'][0]))
            rows.append(row)
    return rows

def domains(scope):return ['full','overlap_50_100'] if scope in ['shared','free'] else ['full']
def select(rows,scope,domain):return [r for r in rows if r['scope']==scope and (domain=='full' or 50<=r['mass_MeV']<=100)]

def empirical(k,n):
    lo,hi=F.interval(k,n)
    return dict(k=int(k),N=n,p_raw=k/n,p_addone=(k+1)/(n+1),lo95=lo,hi95=hi,
                upper95=1. if k==n else float(beta.ppf(.95,k+1,n-k)),
                status='zero_exceedance_upper_bound' if k==0 else 'finite_MC')

def summarize(rows):
    banks={};curves=[];peaks=[];maxima={}
    for scope in SCOPES:
        ss=select(rows,scope,'full');ms=np.array([r['mass_MeV'] for r in ss])
        d=[dict(np.load(massfile(m,'toys'))) for m in ms]
        q=np.column_stack([x[scope+'_q0'] for x in d]);p=np.column_stack([x[scope+'_nominal_local_p'] for x in d])
        rr=None if scope=='free' else np.column_stack([x[scope+'_r'] for x in d])
        score=-np.log(p) if scope=='free' else np.maximum(rr,0)
        banks[scope]=(ms,q,p,rr,score)
        fields=dict(masses=ms,observed_q0=np.array([r['q0'] for r in ss]),toy_q0=q,
                    observed_nominal_p=np.array([r['nominal_local_p'] for r in ss]),toy_nominal_p=p,
                    toy_ordering_score=score,observed_ordering_score=np.array([r['ordering_score'] for r in ss]))
        if rr is not None:fields.update(toy_r=rr,observed_r=np.array([r['r'] for r in ss]),
                                       asimov_r=np.array([r['asimov_r'] for r in ss]))
        np.savez_compressed(R/f'fields_{scope}.npz',**fields)
        for domain in domains(scope):
            ids=np.arange(len(ms)) if domain=='full' else np.flatnonzero((ms>=50)&(ms<=100))
            mx=score[:,ids].max(axis=1);mxq=q[:,ids].max(axis=1)
            maxima[scope+'_'+domain]=mx
            if scope=='free':maxima[scope+'_'+domain+'_raw_q']=mxq
            for j in ids:
                row=ss[j].copy();row['domain']=domain;row['global_ordering']='max(-log nominal chibar p)' if scope=='free' else 'max positive raw signed root'
                if row['q0']==0:
                    kl=kg=N
                else:
                    kl=int(np.sum(q[:,j]>=row['q0']));kg=int(np.sum(mx>=row['ordering_score']))
                for prefix,k in [('conditional_local',kl),('conditional_global',kg)]:
                    row.update({prefix+'_'+key:value for key,value in empirical(k,N).items()})
                    if row['q0']==0:row[prefix+'_status']='exact_zero_statistic_atom';row[prefix+'_lo95']=row[prefix+'_hi95']=1.
                row.update(toy_q0_mean=float(q[:,j].mean()),toy_q0_sd=float(q[:,j].std(ddof=1)),
                           toy_r_mean=None if rr is None else float(rr[:,j].mean()),
                           toy_r_sd=None if rr is None else float(rr[:,j].std(ddof=1)),
                           toy_nominal_5pct_rejections=int(np.sum(p[:,j]<.05)),
                           toy_nominal_1pct_rejections=int(np.sum(p[:,j]<.01)))
                curves.append(row)
            candidates=[c for c in curves if c['scope']==scope and c['domain']==domain]
            peak=max(candidates,key=lambda v:v['ordering_score']).copy()
            peak['peak_definition']='minimum nominal local p, lowest mass in tie'
            peak['search_nodes']=len(ids);peak['search_min_MeV']=float(ms[ids].min());peak['search_max_MeV']=float(ms[ids].max());peak['mass_step_MeV']=.5
            if scope=='free':
                jq=int(ids[np.argmax([ss[i]['q0'] for i in ids])]);raw=ss[jq]
                peak['raw_q_peak_mass_MeV']=float(ms[jq]);peak['raw_q_max']=raw['q0']
                peak['raw_q_alternative_global']=empirical(int(np.sum(mxq>=raw['q0'])),N)
            peaks.append(peak)
    writecsv('significance_curves.csv',curves);jwrite('peaks.json',peaks)
    np.savez_compressed(R/'toy_maxima.npz',**maxima)
    return curves,peaks

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observed-only',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    inputs=sorted(p for p in (F.PARENT/'inputs').glob('*') if p.is_file())
    jwrite('protocol.json',dict(description=__doc__,blind_half_width_sigma=2.25,grid_step_MeV=.5,
        toy_count=N,local_probability='Nominal likelihood asymptotic reference only; empirical tails kept separate',
        background_likelihood_covariance='C=0; original GP arithmetic mean preserved',
        toy_source='Fixed nominal all-data GP source from v5.8.2; original 256 independent complete Poisson spectra reused',
        nuisance_source_uncertainty_propagated=False,nominal_free_calibration='central independent K-amplitude chi-bar-square mixture; not validated by assumption',
        global_free_ordering='max -log nominal local p to account for changing active K; separate raw max q diagnostic',
        input_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}))
    for phase in (['observed'] if args.observed_only else ['observed','toys']):
        for i,m in enumerate(M):
            makestate(m,phase)
            if i%20==0:print(phase,m,'elapsed_s',round(time.monotonic()-start,1),flush=True)
        if phase=='observed':
            rows=observed_rows();writecsv('observed_curves.csv',rows)
            preliminary=[max(select(rows,s,d),key=lambda r:r['ordering_score'])|{'domain':d} for s in SCOPES for d in domains(s)]
            jwrite('observed_peaks.json',preliminary)
            print('OBSERVED_READY',flush=True)
    if not args.observed_only:
        curves,peaks=summarize(rows)
        jwrite('completion.json',dict(completed=True,elapsed_seconds=time.monotonic()-start,observed_rows=len(rows),
             reported_rows=len(curves),scope_domain_peaks=len(peaks),toy_count=N,grid_nodes=len(M),
             max_observed_score=max(r['numerical_score'] for r in rows)))
        print('COMPLETE',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
