"""Independent main-study audit; no import of core.py or its fit routines."""
from pathlib import Path
import os,sys,json,hashlib,datetime,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.optimize import minimize
from scipy.stats import beta
B=Path(__file__).resolve().parents[1]
MASTER=63520260924
MASSES=list(range(60,241,20))
SOURCES=['nominal','functional']
POLICIES=['pole','logshift']
GRID=[0,1,2,3,4,5,6,8,10,12,16,20,24]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def date(s):return datetime.datetime.fromisoformat(s)
def read(name):return json.loads((B/name).read_text())

def direct_fit(d,counts,m,policy,coef,expected):
    """Independent fixed log-GP prediction and generic BFGS minimization."""
    center=m+(coef[0]+coef[1]*np.log(m/150.) if policy=='logshift' else 0.)
    sigma=np.polynomial.polynomial.polyval(m/1000,d['sigma_coeffs'])
    mask=(d['x']>=center/1000-2.25*sigma)&(d['x']<=center/1000+2.25*sigma)
    idx=int(np.flatnonzero(d['masses']==m)[0]);amp,length=d['const'][idx],d['ls'][idx]
    def kernel(x,z):return amp*np.exp(-.5*((np.log(x)[:,None]-np.log(z)[None,:])/length)**2)
    train=counts[~mask].astype(float);positive=train>0
    target=np.zeros_like(train);target[positive]=np.log(train[positive])
    noise=np.ones_like(train);noise[positive]=1/train[positive]
    K=kernel(d['x'][~mask],d['x'][~mask]);K.flat[::len(K)+1]+=noise
    chol=cholesky(K,lower=True);cross=kernel(d['x'][mask],d['x'][~mask])
    mean=cross@cho_solve((chol,True),target);v=solve_triangular(chol,cross.T,lower=True)
    cov=kernel(d['x'][mask],d['x'][mask])-v.T@v;cov=.5*(cov+cov.T)
    b=np.exp(mean+.5*np.maximum(np.diag(cov),0));V=np.outer(b,b)*np.expm1(np.clip(cov,-40,40))
    V=.5*(V+V.T);vscale=max(float(np.diag(V).max()),1.)
    for load in (1e-10,1e-9,1e-8,1e-7,1e-6,1e-5):
        try:cholesky(V+load*vscale*np.eye(len(V)),lower=True);break
        except np.linalg.LinAlgError:pass
    else:raise AssertionError('Covariance loading failed')
    V+=load*vscale*np.eye(len(V));sd=np.sqrt(b)
    eig,U=np.linalg.eigh(V/sd[:,None]/sd[None,:]);keep=eig>1e-8
    L=sd[:,None]*U[:,keep]*np.sqrt(eig[keep])
    signal=np.diff(ndtr((d['edges']-center/1000)/sigma))[mask]
    scale=1/np.sqrt(np.sum(signal**2/b));J=np.column_stack([scale*signal,L])
    penalty=np.r_[0.,np.ones(L.shape[1])];n=counts[mask].astype(float)
    def objective(z,J=J,base=b,penalty=penalty):
        lam=base+J@z
        if np.any(lam<=0):return np.inf,np.zeros_like(z)
        t=(lam-n)/np.maximum(n,1.)
        value=float(np.sum(np.where(n>0,n*(t-np.log1p(t)),lam))+.5*np.sum(penalty*z*z))
        gradient=J.T@(1-n/lam)+penalty*z
        return value,gradient
    fit=minimize(objective,np.zeros(J.shape[1]),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
    value,gradient=objective(fit.x);lam=b+J@fit.x
    H=(J.T*(n/lam**2))@J+np.diag(penalty)
    error=scale*np.sqrt(np.linalg.inv(H)[0,0])
    fixed_fun=lambda z:objective(z,L,b+expected*signal,np.ones(L.shape[1]))
    fixed=minimize(fixed_fun,np.zeros(L.shape[1]),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
    fval,fg=fixed_fun(fixed.x)
    assert np.max(np.abs(gradient))<3e-5 and np.max(np.abs(fg))<3e-5
    return dict(Ahat=float(scale*fit.x[0]),sigma=float(error),nll=value,q=max(0.,2*(fval-value)),
        score=float(np.max(np.abs(gradient))),method='SciPy BFGS, independent objective and GP',
        gp_mean_hash=ahash(b),covariance_load=load)

def main():
    started=time.monotonic()
    status=read('status.json');assert status['status']=='numerical_complete','Wait until main computation is complete'
    protocol=read('protocol.json');assert protocol['master_seed']==MASTER
    signature=read('provenance/computation_signature.json')['signature']
    inputs=read('provenance/input_hashes.json');parts=[]
    for r in inputs:
        assert sha(B/r['path'])==r['sha256'];parts.append((r['path'],r['sha256']))
    for rel in ('scripts/core.py','scripts/run_study.py','protocol.json','inputs/cohorts.npz','inputs/templates.npz'):
        parts.append((rel,sha(B/rel)))
    assert hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()==signature
    d=dict(np.load(B/'inputs/v6p1/inputs/spectrum_2021.npz'))
    null=dict(np.load(B/'inputs/null_2021.npz'));cohorts=np.load(B/'inputs/cohorts.npz');templates=np.load(B/'inputs/templates.npz')
    assert np.array_equal(d['n'],null['observed']) and np.array_equal(d['edges'],null['edges_GeV'])
    coef=next(r['coefficients'] for r in read('inputs/v6p1/derived/analytic_shift_models.json')['models'] if r['model']=='logarithmic')
    ref=read('pilot_reference.json');freeze=read('calibration_freeze.json')
    assert ref['signature']==freeze['signature']==signature
    assert ref['pilot_rows_sha256']==sha(B/'results/pilot_rows.csv')
    assert freeze['calibration_rows_sha256']==sha(B/'results/calibration_rows.csv')
    assert freeze['pilot_reference_sha256']==sha(B/'pilot_reference.json')
    for name,h in freeze['checkpoint_hashes'].items():assert sha(B/name)==h
    truth={'nominal':null['truth'],'functional':d['stress']}
    background_replays=0;hashes=[];cats={};fitmasks={};fitcats={};cdf_errors=[]
    for cohort,ns in (('pilot',1),('calibration',2),('evaluation',3)):
        for source in (['nominal'] if cohort=='pilot' else SOURCES):
            a=cohorts[cohort+'_'+source];assert a.shape==(100,len(d['n']))
            for toy in range(100):
                regen=np.random.default_rng(np.random.SeedSequence([MASTER,ns,SOURCES.index(source),toy])).poisson(truth[source])
                assert np.array_equal(a[toy],regen);hashes.append(ahash(regen));background_replays+=1
    assert len(hashes)==500 and len(set(hashes))==500
    for i,m in enumerate(MASSES):
        raw=np.load(B/f'inputs/v6p1/histograms/m{m:03d}.npz');meta=json.loads(str(raw['metadata']))
        assert np.array_equal(raw['sumw'],raw['sumw2']) and meta['stats']['underflow']==0
        assert raw['sumw'].sum()+meta['stats']['overflow']==meta['sumw']
        cdf=np.interp(d['edges'],raw['edges_GeV'],np.r_[0.,np.cumsum(raw['sumw'])]/meta['sumw'],left=0.,right=raw['sumw'].sum()/meta['sumw'])
        independent=np.r_[cdf[0],np.diff(cdf),1-cdf[-1]]
        e=float(np.max(abs(independent-templates['mc_categories'][i])));cdf_errors.append(e);assert e<5e-15
        cats[m,'mc']=templates['mc_categories'][i]
        center=m+coef[0]+coef[1]*np.log(m/150.);sigma=np.polynomial.polynomial.polyval(m/1000,d['sigma_coeffs'])
        gcdf=ndtr((d['edges']-center/1000)/sigma);gcat=np.r_[gcdf[0],np.diff(gcdf),1-gcdf[-1]]
        assert np.array_equal(gcat,templates['gaussian_log_categories'][i]);cats[m,'gaussian_log']=gcat
        for policy in POLICIES:
            c=center if policy=='logshift' else m
            mask=(d['x']>=c/1000-2.25*sigma)&(d['x']<=c/1000+2.25*sigma)
            assert np.array_equal(mask,templates[policy+'_fit'][i])
            assert (d['x']<c/1000-2.25*sigma).sum()>=3 and (d['x']>c/1000+2.25*sigma).sum()>=3
            cdf=ndtr((d['edges']-c/1000)/sigma)
            fitmasks[m,policy]=mask;fitcats[m,policy]=np.r_[cdf[0],np.diff(cdf),1-cdf[-1]]
        assert abs(cats[m,'mc'].sum()-1)<1e-12 and cats[m,'mc'].min()>=0
    markers=[];rows_checked=0;signal_replays=0;dfs={};replay_samples=[]
    for cohort in ('pilot','calibration','evaluation'):
        tables=[]
        for marker in sorted((B/'results'/cohort).glob('*.json')):
            rec=json.loads(marker.read_text());markers.append(rec);assert rec['complete'] and rec['signature']==signature
            assert rec['reference_sha256']==(None if cohort=='pilot' else sha(B/'pilot_reference.json'))
            assert rec['calibration_sha256']==(sha(B/'calibration_freeze.json') if cohort=='evaluation' else None)
            assert date(rec['started_utc'])<=date(rec['completed_utc'])
            if cohort!='pilot':assert date(rec['started_utc'])>=date(ref['frozen_utc'])
            if cohort=='evaluation':assert date(rec['started_utc'])>=date(freeze['frozen_utc'])
            for rel,h in rec['output_hashes'].items():assert sha(B/rel)==h
            source,m=rec['source'],rec['mass_MeV'];s0=ref['masses'][str(m)]['s0']
            df=pd.read_csv(marker.with_suffix('.csv'),float_precision='round_trip');assert len(df)==rec['rows']
            tables.append(df);drawmap={}
            if cohort!='pilot':
                vectors=np.load(marker.with_suffix('.npz'))
                for key,draw in zip(vectors['keys'],vectors['draws']):
                    toy,z,shapeid=map(int,key);shape='mc' if shapeid==0 else 'gaussian_log'
                    keyseed=[MASTER,20 if cohort=='calibration' else 30,SOURCES.index(source),m,toy,z,shapeid]
                    regen=np.zeros(len(d['n'])+2,dtype=np.int64) if z==0 else np.random.default_rng(np.random.SeedSequence(keyseed)).poisson(z*s0*cats[m,shape])
                    assert np.array_equal(regen,draw);drawmap[toy,z,shape]=draw;signal_replays+=1
            for row in df.itertuples():
                A=0. if row.z==0 else row.z*s0;assert row.A_expected==A
                bg=cohorts[cohort+'_'+source][row.toy]
                draw=np.zeros(len(d['n'])+2,dtype=np.int64) if cohort=='pilot' else drawmap[row.toy,row.z,row.shape]
                mask=fitmasks[m,row.policy];counts=bg+draw[1:-1]
                assert row.background_hash==ahash(bg) and row.signal_hash==ahash(draw) and row.counts_hash==ahash(counts)
                assert row.fit_mask_hash==ahash(mask) and row.fit_template_hash==ahash(fitcats[m,row.policy])
                assert row.actual_full==draw.sum() and row.actual_support==draw[1:-1].sum()
                assert row.actual_window==draw[1:-1][mask].sum() and row.actual_training==draw[1:-1][~mask].sum()
                assert row.actual_outside_support==draw[0]+draw[-1]
                rows_checked+=1
                if cohort=='evaluation' and row.toy==17 and row.z==3 and row.shape=='mc' and m in (60,140,240) and ((row.policy=='logshift')==(source=='functional')):
                    replay_samples.append((row,counts.copy()))
        full=pd.concat(tables).sort_values(['source','policy','mass_MeV','shape','z','toy']).reset_index(drop=True)
        aggregate=pd.read_csv(B/f'results/{cohort}_rows.csv',float_precision='round_trip').reset_index(drop=True)
        pd.testing.assert_frame_equal(full,aggregate,check_dtype=False)
        assert len(full)=={'pilot':2000,'calibration':52000,'evaluation':22000}[cohort]
        expected={(s,p,m,shape,z,t) for s in (['nominal'] if cohort=='pilot' else SOURCES) for p in POLICIES for m in MASSES
                  for shape,levels in ([('mc',[0])] if cohort=='pilot' else [('mc',GRID)] if cohort=='calibration' else [('mc',[0,1,3,5])]+([('gaussian_log',[1,3,5])] if s=='nominal' else []))
                  for z in levels for t in range(100)}
        actual=set(zip(full.source,full.policy,full.mass_MeV,full['shape'],full.z,full.toy))
        assert actual==expected and len(actual)==len(full)
        assert full.fit_valid.all() and np.isfinite(full.Ahat).all() and np.isfinite(full.sigma_postfit).all() and (full.sigma_postfit>0).all()
        assert (full.fit_score<3e-5).all() and (full.min_lambda>0).all() and (full.sigma_method=='observed_profile_hessian').all()
        assert np.allclose(full.pull,(full.Ahat-full.A_expected)/full.sigma_postfit,rtol=2e-14,atol=2e-14)
        assert full.attempt_count.between(1,3).all()
        if cohort=='evaluation':
            assert full.profile_valid.all() and (full.profile_score<3e-5).all()
            q=np.maximum(0,2*(full.true_nll-full.free_nll));assert np.allclose(q,full.q_true,atol=1e-9)
            assert np.array_equal(q<=1,full.profile_contains68) and np.array_equal(q<=3.841459,full.profile_contains95)
            mc=full[full['shape']=='mc'];assert len(mc)==16000 and mc.native_cls90_valid.all() and np.isfinite(mc.native_cls90).all()
            assert np.array_equal(mc.native_cls90>=mc.A_expected,mc.native_cls90_contains)
        dfs[cohort]=full
    assert len(markers)==500 and rows_checked==76000 and signal_replays==37000
    assert max(date(r['completed_utc']) for r in markers if r['cohort']=='pilot')<=date(ref['frozen_utc'])
    assert max(date(r['completed_utc']) for r in markers if r['cohort']=='calibration')<=date(freeze['frozen_utc'])
    for m in MASSES:
        pilot=dfs['pilot'];q=pilot[(pilot.mass_MeV==m)&(pilot.policy=='logshift')]
        assert abs(q.sigma_postfit.mean()-ref['masses'][str(m)]['s0'])<1e-10
    fits=[]
    for row,counts in replay_samples:
        f=direct_fit(d,counts,row.mass_MeV,row.policy,coef,row.A_expected)
        f.update(mass_MeV=row.mass_MeV,source=row.source,policy=row.policy,toy=row.toy,
                 yield_difference_in_sigma=(f['Ahat']-row.Ahat)/row.sigma_postfit,
                 sigma_relative_difference=f['sigma']/row.sigma_postfit-1,q_difference=f['q']-row.q_true)
        assert abs(f['yield_difference_in_sigma'])<1e-5 and abs(f['sigma_relative_difference'])<1e-6 and abs(f['q_difference'])<2e-5
        fits.append(f)
    result=dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validator_sha256=sha(__file__),computation_signature=signature,background_seed_replays=background_replays,
        unique_background_spectra=500,signal_vector_seed_replays=signal_replays,checkpoint_markers_verified=len(markers),
        rows_verified=rows_checked,free_fits_valid=76000,evaluation_profiles_valid=22000,native_CLs_valid=16000,
        no_drop_grid_completeness=True,independent_pilot_calibration_evaluation_freeze_order=True,
        independent_MC_CDF_max_absolute_difference=max(cdf_errors),full_selected_normalization_verified=True,
        MC_overflow_retained=True,fit_and_training_exclusion_identical=True,independent_BFGS_checks=fits,
        statistics_formula_audit='Pending analyzer outputs',seconds=time.monotonic()-started)
    out=B/'qa/independent_validation.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='independent_BFGS_checks'},indent=2))

def statistics_only():
    """Check derived estimators and every held-out rank vector independently."""
    r=B/'results'
    columns=['source','policy','mass_MeV','shape','z','toy','A_expected','Ahat','sigma_postfit']
    cal=pd.read_csv(r/'calibration_rows.csv',usecols=columns,float_precision='round_trip')
    ev=pd.read_csv(r/'evaluation_rows.csv',usecols=columns,float_precision='round_trip')
    pars=pd.read_csv(r/'calibration_summary.csv',float_precision='round_trip')
    points=pd.read_csv(r/'pointwise_rows.csv',float_precision='round_trip')
    limits=pd.read_csv(r/'limit_summary.csv',float_precision='round_trip')
    cells={k:q.sort_values('toy') for k,q in cal.groupby(['source','policy','mass_MeV','z'])}
    for row in pars.itertuples():
        null=cells[row.source,row.policy,row.mass_MeV,0]
        signal=cells[row.source,row.policy,row.mass_MeV,3]
        y=null.Ahat.to_numpy();sigma=null.sigma_postfit.to_numpy()
        delta=float(y.mean());mu=float((y/sigma).mean());k0=float(np.std((y-delta)/sigma,ddof=1))
        response=float(np.mean(signal.Ahat.to_numpy()-y)/signal.A_expected.iloc[0])
        assert np.allclose([row.delta,row.mu0,row.k0,row.R],[delta,mu,k0,response],rtol=1e-10,atol=1e-10)
    rank_vectors=0
    for key,q in points.groupby(['source','calibration_source','policy','mass_MeV','z']):
        source,cs,policy,m,z=key
        sample=ev[(ev.source==source)&(ev.policy==policy)&(ev.mass_MeV==m)&(ev.z==z)&(ev['shape']=='mc')].sort_values('toy')
        grid=np.stack([cells[cs,policy,m,g].Ahat.to_numpy() for g in GRID])
        amplitudes=np.array([cells[cs,policy,m,g].A_expected.iloc[0] for g in GRID])
        lower=(1+np.sum(grid[:,:,None]<=sample.Ahat.to_numpy()[None,None,:],axis=1))/101
        upper=(1+np.sum(grid[0,:,None]>=sample.Ahat.to_numpy()[None,:],axis=0))/101
        for row in q.itertuples():
            p=lower[:,row.toy];cls=np.minimum(1,p/p[0])
            assert np.allclose(json.loads(row.p_grid_json),p,atol=1e-14,rtol=0)
            assert np.allclose(json.loads(row.cls_grid_json),cls,atol=1e-14,rtol=0)
            assert abs(row.p0_rank-upper[row.toy])<1e-14
            assert row.p0_k==int(round(101*upper[row.toy]-1))
            for prefix,prob in [('rank',p),('toy_cls',cls)]:
                accept=prob>.1;indices=np.flatnonzero(accept)
                assert getattr(row,prefix+'_empty')==(len(indices)==0)
                assert getattr(row,prefix+'_right_censored')==accept[-1]
                assert json.loads(getattr(row,prefix+'_accepted_z'))==[GRID[i] for i in indices]
                endpoint=amplitudes[indices[-1]] if len(indices) else 0.
                assert abs(getattr(row,prefix+'_U_grid')-endpoint)<1e-8
            rank_vectors+=1
    cp_checks=0
    for row in limits.itertuples():
        for name in ('acceptance','upper_coverage'):
            n=int(getattr(row,name+'_n'));k=int(getattr(row,name+'_k'))
            lo=0. if k==0 else beta.ppf(.025,k,n-k+1)
            hi=1. if k==n else beta.ppf(.975,k+1,n-k)
            assert np.allclose([getattr(row,name+'_cp95_lo'),getattr(row,name+'_cp95_hi')],[lo,hi],rtol=1e-12,atol=1e-12)
            assert n==100 and abs(getattr(row,name+'_fraction')-k/n)<1e-14
            cp_checks+=1
    q=read('qa/independent_validation.json')
    q['validator_sha256']=sha(__file__)
    q['statistics_formula_audit']=dict(passed=True,calibration_parameter_cells=len(pars),rank_vectors_verified=rank_vectors,
        clopper_pearson_intervals_verified=cp_checks,mean_pull_and_yield_offset_distinguished=True,
        k0_is_yield_centered_pull_SD=True,rank_threshold='p > 0.1 accepts; p <= 0.1 rejects',
        empty_sets_preserved=True,right_censoring_preserved=True,
        source_hashes={p.name:sha(p) for p in [r/'calibration_summary.csv',r/'pointwise_rows.csv',r/'limit_summary.csv']})
    (B/'qa/independent_validation.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
    print(json.dumps(q['statistics_formula_audit'],indent=2))

if __name__=='__main__':
    if '--statistics-only' in sys.argv:statistics_only()
    else:main()
