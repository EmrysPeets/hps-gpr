#!/usr/bin/env python3
"""Bounded 2015 rising-edge diagnostic; never modifies parent releases."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, warnings
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'qa/mpl_cache')
sys.path.insert(0,str(ROOT/'study_results/background_profile_comparison_20260905'))
import run_comparison as c
import numpy as np
import pandas as pd
import uproot
from scipy.special import ndtr
from scipy.stats import norm,beta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF
from hps_gpr.gpr import predict_counts_from_log_gpr

SUPPORTS={'gp_12_28':(12.,28.),'gp_12_26':(12.,26.),
          'gp_12_30':(12.,30.),'gp_12p5_28':(12.5,28.)}
MASSES=np.arange(15.,22.00001,.25)
SEED=49161520

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sigma(m):return -.0922283032152+.0532190838657*m
def seed(*values):return int.from_bytes(hashlib.sha256('|'.join(map(str,(SEED,*values))).encode()).digest()[:4],'little')

def load_data():
    cfg=c.production.load_config(c.production.DEFAULT_CARD)
    p=Path(cfg.path_2015)
    assert sha(p)==c.production.EXPECTED_HISTOGRAM_SHA256['2015']
    with uproot.open(p) as f:
        h=f[cfg.hist_2015];n,e=h.to_numpy();v=h.variances()
    assert np.array_equal(n,n.round()) and np.array_equal(n,v)
    assert len(n)%5==0 and np.allclose(np.diff(e),.00005,rtol=0,atol=1e-16)
    data=dict(n=n.reshape(-1,5).sum(1),edges=e[::5]*1000)
    data['x']=(data['edges'][:-1]+data['edges'][1:])/2
    cfg.enable_2016=False;cfg.enable_2021=False
    cfg.n_restarts=3
    source=dict(path=str(p),file_sha256=sha(p),histogram=cfg.hist_2015,
        raw_counts_sha256=hashlib.sha256(n.astype('<f8').tobytes()).hexdigest(),
        raw_edges_sha256=hashlib.sha256(e.astype('<f8').tobytes()).hexdigest(),
        native_bins=len(n),native_bin_width_MeV=.05,inference_bin_width_MeV=.25,
        nonnegative_integer_counts=True,poisson_variances_match=True,
        card_sha256=sha(c.production.DEFAULT_CARD))
    return cfg,data,source

def subset(data,lo,hi):
    ix=np.flatnonzero((data['edges'][:-1]>=lo-1e-10)&(data['edges'][1:]<=hi+1e-10))
    assert np.all(np.diff(ix)==1)
    return dict(n=data['n'][ix],x=data['x'][ix],edges=data['edges'][np.r_[ix,ix[-1]+1]])

def gp_predict(x,n,mask,m,cfg,stream='observed',restarts=3,upper_factor=8.):
    sx=np.log1p(sigma(m)/m)
    gp=GaussianProcessRegressor(kernel=ConstantKernel(100.,(1e-8,1e18))*RBF(3*sx,(sx,upper_factor*sx)),
        alpha=1/np.maximum(n[~mask],1.),normalize_y=False,
        n_restarts_optimizer=restarts,random_state=seed(stream,m))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gp.fit(np.log(x[~mask]/1000)[:,None],np.log(np.maximum(n[~mask],1.)))
        mean,C=predict_counts_from_log_gpr(gp,x/1000,cfg)
    return gp,mean,C,[str(w.message) for w in caught]

def gp_fit(data,m,cfg,method='gp_12_28',stream='observed',restarts=3,upper_factor=8.):
    d=subset(data,*SUPPORTS[method]);x,n,e=d['x'],d['n'],d['edges']
    mask=abs(x-m)<=2.25*sigma(m)
    assert (x[~mask]<m).any() and (x[~mask]>m).any()
    gp,b,C,warnings_=gp_predict(x,n,mask,m,cfg,stream,restarts,upper_factor)
    Cwin,record=c.production.condition_covariance_block(C[np.ix_(mask,mask)],b[mask])
    L=c._chol_with_jitter(Cwin)
    template=np.diff(ndtr((e-m)/sigma(m)));fraction=template[mask].sum()
    w=template[mask]/fraction
    model=c.Profile(b[mask],L,w,'linear')
    fit=model.fit(n[mask]);null=model.fit(n[mask],0.)
    r=float(np.sign(fit['A'])*np.sqrt(max(0,2*(null['nll']-fit['nll']))))
    bfit=np.full(len(x),np.nan);bnull=bfit.copy();total=bfit.copy()
    bfit[mask]=fit['bfit'];bnull[mask]=null['bfit'];total[mask]=fit['lam']
    signal=fit['A']/fraction*template
    np.testing.assert_allclose((bfit+signal)[mask],total[mask],rtol=1e-12,atol=1e-8)
    assert min(fit['min_lambda'],null['min_lambda'])>0
    sx=np.log1p(sigma(m)/m);ls=float(gp.kernel_.k2.length_scale)
    row=dict(method=method,mass_MeV=m,sigma_MeV=sigma(m),r=r,p0=float(norm.sf(max(r,0.))),
        Ahat_window=fit['A'],sigma_A_window=fit['sigma'],
        Ahat_total=fit['A']/fraction,sigma_A_total=fit['sigma']/fraction,
        signal_fraction_in_fit=float(fraction),fit_low_MeV=float(e[:-1][mask].min()),
        fit_high_MeV=float(e[1:][mask].max()),support_low_MeV=float(e[0]),support_high_MeV=float(e[-1]),
        nfit=int(mask.sum()),nleft=int(((x<m)&~mask).sum()),nright=int(((x>m)&~mask).sum()),
        left_events=float(n[(x<m)&~mask].sum()),right_events=float(n[(x>m)&~mask].sum()),
        count_min_fit=float(n[mask].min()),nll_fit=fit['nll'],nll_null=null['nll'],
        max_score=max(fit['score'],null['score']),min_lambda=min(fit['min_lambda'],null['min_lambda']),
        kernel_constant=float(gp.kernel_.k1.constant_value),kernel_ls_logmass=ls,
        kernel_ls_sigma=ls/sx,kernel_upper_factor=upper_factor,
        kernel_at_boundary=bool(ls/sx<1.001 or ls/sx>upper_factor-.001),
        log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
        covariance_load_relative=record['selected_diagonal_load_relative'],warning_count=len(warnings_))
    arrays=dict(x=x,n=n,edges=e,mask=mask,bprior=b,C=C,bfit=bfit,bnull=bnull,
        total=total,signal=signal,template=template,fit_cov=L@L.T,
        free_z=fit['z'],null_z=null['z'])
    return row,arrays,warnings_

def poly_fit(data,m,degree=5,quad_order=8):
    lo=max(12.,np.floor((m-7*sigma(m))/.25)*.25)
    hi=np.ceil((m+7*sigma(m))/.25)*.25
    d=subset(data,lo,hi);n,x,e=d['n'],d['x'],d['edges']
    q,wt=np.polynomial.legendre.leggauss(quad_order)
    qm=x[:,None]+np.diff(e)[:,None]*q[None,:]/2
    weights=np.diff(e)[:,None]*wt[None,:]/2
    X=np.polynomial.chebyshev.chebvander(2*(qm-lo)/(hi-lo)-1,degree)
    X0=np.polynomial.chebyshev.chebvander(2*(x-lo)/(hi-lo)-1,degree)
    start=np.linalg.lstsq(X0*np.sqrt(np.maximum(n,1))[:,None],
        np.log(np.maximum(n,1)/np.diff(e))*np.sqrt(np.maximum(n,1)),rcond=None)[0]
    template=np.diff(ndtr((e-m)/sigma(m)))
    scale=np.sqrt(n.sum())
    def solve(free,base):
        z=np.zeros(degree+1+int(free));offset=int(free)
        def evaluate(v):
            coeff=base+v[offset:]/scale
            t=X@coeff
            if np.max(t)>100:return np.inf,None,None,None,None
            qb=np.exp(t)*weights;b=qb.sum(1)
            lam=b+(v[0]*scale*template if free else 0.)
            if np.any(lam<=0):return np.inf,None,None,b,lam
            Jb=np.einsum('ij,ijk->ik',qb,X)/scale
            J=np.column_stack((scale*template,Jb)) if free else Jb
            rr=1-n/lam;g=J.T@rr
            H=(J.T*(n/lam**2))@J
            H[offset:,offset:]+=np.einsum('i,ij,ijk,ijl->kl',rr,qb,X,X)/scale**2
            return c.deviance_half(n,lam),g,H,b,lam
        for it in range(160):
            value,g,H,b,lam=evaluate(z)
            if g is None:raise RuntimeError('Bad polynomial expectation')
            if np.max(abs(g))<2e-7:break
            eig=np.linalg.eigvalsh(H).min()
            if eig<=1e-10:H=H+(1e-8-eig)*np.eye(len(H))
            delta=np.linalg.solve(H,-g);rate=1.
            for _ in range(60):
                if evaluate(z+rate*delta)[0]<=value+1e-4*rate*(g@delta)+1e-11:
                    z+=rate*delta;break
                rate*=.5
            else:raise RuntimeError('Polynomial line search failed')
        else:raise RuntimeError('Polynomial fit did not converge')
        val,g,H,b,lam=evaluate(z)
        return dict(nll=float(val),coeff=base+z[offset:]/scale,b=b,lam=lam,
            A=float(z[0]*scale) if free else 0.,sigma=float(scale*np.sqrt(np.linalg.inv(H)[0,0])) if free else 0.,
            score=float(np.max(abs(g))),iterations=it)
    null=solve(False,start);fit=solve(True,null['coeff'])
    assert fit['nll']<=null['nll']+1e-6
    r=float(np.sign(fit['A'])*np.sqrt(max(0,2*(null['nll']-fit['nll']))))
    row=dict(method='expcheb5',mass_MeV=m,sigma_MeV=sigma(m),r=r,p0=float(norm.sf(max(r,0.))),
        Ahat_total=fit['A'],sigma_A_total=fit['sigma'],support_low_MeV=lo,support_high_MeV=hi,
        nfit=len(n),nll_fit=fit['nll'],nll_null=null['nll'],
        max_score=max(fit['score'],null['score']),min_lambda=float(min(fit['lam'].min(),null['lam'].min())),
        poisson_deviance=float(2*fit['nll']),gof_nominal_dof=len(n)-degree-2)
    arrays=dict(x=x,n=n,edges=e,mask=np.ones(len(n),bool),bprior=null['b'],bnull=null['b'],
        bfit=fit['b'],total=fit['lam'],signal=fit['A']*template,template=template,
        coeff_fit=fit['coeff'],coeff_null=null['coeff'])
    return row,arrays,[]

def observed():
    began=time.monotonic()
    for name in ('derived/fits','provenance','qa','figures','note'):(HERE/name).mkdir(parents=True,exist_ok=True)
    cfg,data,source=load_data()
    pd.DataFrame(dict(left_MeV=data['edges'][:-1],right_MeV=data['edges'][1:],counts=data['n'])).to_csv(HERE/'derived/input_histogram.csv',index=False)
    dump(HERE/'provenance/input.json',source)
    rows=[];notes={}
    for method in (*SUPPORTS,'expcheb5'):
        for m in MASSES:
            row,arr,ww=poly_fit(data,m) if method=='expcheb5' else gp_fit(data,m,cfg,method)
            rows.append(row);tag=f'{method}_m{m:05.2f}';notes[tag]=ww
            np.savez_compressed(HERE/'derived/fits'/f'{tag}.npz',**arr)
        print(method,'complete',round(time.monotonic()-began,2),'s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'derived/scan.csv',index=False)
    nominal=frame[(frame.method=='gp_12_28')&(frame.mass_MeV<=20)]
    peak=nominal.loc[nominal.r.idxmax()]
    anchors=sorted(set([15.,17.,20.,float(peak.mass_MeV)]))
    repeats=[]
    for m in anchors:
        row,_,_=gp_fit(data,m,cfg,restarts=7)
        prior=nominal[nominal.mass_MeV==m].iloc[0]
        dr=row['r']-prior.r
        repeats.append(dict(mass_MeV=m,delta_r=float(dr),delta_lml=float(row['log_marginal_likelihood']-prior.log_marginal_likelihood),passed=bool(abs(dr)<.002)))
    dump(HERE/'qa/optimizer_repeats.json',repeats)
    if not all(x['passed'] for x in repeats):raise RuntimeError('Optimizer repeat failed; inspect before release')
    summary=dict(status='exploratory_observed',passed=True,requested_grid_MeV=list(map(float,MASSES[MASSES<=20])),
        bridge_grid_MeV=list(map(float,MASSES[MASSES>20])),extraction_masses_MeV=anchors,
        nominal_peak=peak.dropna().to_dict(),n_scan_rows=len(frame),max_score=float(frame.max_score.max()),
        methods=list(frame.method.unique()),elapsed_seconds=time.monotonic()-began,
        sources={str(Path(p).relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'PROTOCOL.md',Path(c.__file__),c.production.DEFAULT_CARD]},
        claim='Post-selection, local and conditional; unqualified detector response below 19 MeV; no exclusion or global significance.')
    dump(HERE/'derived/summary.json',summary);dump(HERE/'qa/gp_warnings.json',notes)
    print(json.dumps(summary,indent=2),flush=True)

def toys(count,upper_factor=8.):
    cfg,data,_=load_data();summary=json.loads((HERE/'derived/summary.json').read_text())
    suffix='' if upper_factor==8 else f'_ceiling{int(upper_factor)}'
    method='gp_12_28' if upper_factor==8 else f'gp_ceiling{int(upper_factor)}'
    records=[];path=HERE/'derived'/f'toy_roots{suffix}.csv'
    if path.exists():records=pd.read_csv(path).to_dict('records')
    done={(r['mass_MeV'],int(r['toy_id'])) for r in records}
    began=time.monotonic()
    for m in summary['extraction_masses_MeV']:
        nominal=np.load(HERE/'derived/fits'/f'{method}_m{m:05.2f}.npz')
        truth=nominal['bprior'].copy()
        # Use one continuous sideband-conditioned prediction, with no observed-window adjustment.
        pd.DataFrame(dict(mass_MeV=nominal['x'],truth=truth)).to_csv(HERE/'derived'/f'toy_truth{suffix}_m{m:05.2f}.csv',index=False)
        for i in range(count):
            if (m,i) in done:continue
            rng=np.random.default_rng(seed('conditional-null',m,i));n=rng.poisson(truth).astype(float)
            local=dict(n=n,x=nominal['x'],edges=nominal['edges'])
            row,_,_=gp_fit(local,m,cfg,stream=f'toy-{i}',restarts=3 if upper_factor==8 else 7,upper_factor=upper_factor)
            records.append(dict(mass_MeV=m,toy_id=i,r=row['r'],Ahat_total=row['Ahat_total'],
                sigma_A_total=row['sigma_A_total'],kernel_ls_sigma=row['kernel_ls_sigma'],max_score=row['max_score']))
        pd.DataFrame(records).to_csv(path,index=False)
        print('toys',m,count,round(time.monotonic()-began,2),'s',flush=True)
    frame=pd.DataFrame(records)
    scan=pd.read_csv(HERE/'derived'/('scan.csv' if upper_factor==8 else 'kernel_stability.csv'));out=[]
    for m,g in frame.groupby('mass_MeV'):
        obs=scan[(scan.method==method)&(scan.mass_MeV==m)].iloc[0]
        n=len(g);k=int((g.r>=obs.r).sum()) if obs.r>0 else n
        out.append(dict(mass_MeV=float(m),n=n,k=k,observed_r=float(obs.r),p_hat=k/n,
            low95=0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            high95=1. if k==n else float(beta.ppf(.975,k+1,n-k)),
            upper95=1. if k==n else float(beta.ppf(.95,k+1,n-k)),
            mean_r=float(g.r.mean()),sd_r=float(g.r.std(ddof=1)),
            true_window_signal=0.,truth='mass-local continuous sideband-conditioned observed GP mean'))
    dump(HERE/'derived'/f'toy_summary{suffix}.json',dict(passed=True,count=count,upper_factor=upper_factor,anchors=out,
        elapsed_seconds=time.monotonic()-began,hyperparameters_refit=True,
        warning='These mass-local plug-in nulls are not a common global background or independent model validation.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--toys',type=int,default=0)
    parser.add_argument('--upper-factor',type=float,default=8.)
    args=parser.parse_args()
    if args.toys:toys(args.toys,args.upper_factor)
    else:observed()
