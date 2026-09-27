"""Factorized free-amplitude likelihood and fixed-source null calibrations.

The observed and direct-Poisson q values use the released exact profile roots.
Only the explicitly named Gaussian-response calibration uses r=a+D.T epsilon.
At q>0, polar integration sums all nonempty positive-root subsets.  At zero
the inclusive survival function is one, preserving the all-inactive atom.
"""
from pathlib import Path
import os, sys, json, time, itertools, csv
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, beta, chi2
from scipy.interpolate import PchipInterpolator
from numpy.polynomial.legendre import leggauss

B=Path(__file__).resolve().parents[1]
YEARS=['2015','2016','2021']; WIDTHS=[2.25,2.4,2.5,2.6]; MASSES=np.arange(19.,251.)

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=keys);w.writeheader();w.writerows(rows)

def mc_tail(values, threshold):
    n=len(values);k=int(np.count_nonzero(values>=threshold))
    lo=0. if not k else float(beta.ppf(.025,k,n-k+1))
    hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
    upper=1. if k==n else float(beta.ppf(.95,k+1,n-k))
    estimate=(k+1)/(n+1)
    display=upper if not k else estimate
    return dict(k=k,N=n,p_plus1=estimate,p_raw=k/n,lo95=lo,hi95=hi,
                upper95=upper,Z_signed=float(norm.isf(display)),Z_display=max(0.,float(norm.isf(display))),
                Z_is_lower_bound=bool(k==0))

_angles={}
def directions(dim,n):
    if (dim,n) in _angles:return _angles[dim,n]
    if dim==1:u=np.ones((1,1));w=np.ones(1)
    else:
        x,w0=leggauss(n);theta=(x+1)*np.pi/4;w0=w0*np.pi/4
        if dim==2:u=np.column_stack([np.cos(theta),np.sin(theta)]);w=w0
        else:
            th,ph=np.meshgrid(theta,theta,indexing='ij')
            u=np.column_stack([(np.sin(th)*np.cos(ph)).ravel(),(np.sin(th)*np.sin(ph)).ravel(),np.cos(th).ravel()])
            w=(w0[:,None]*w0[None,:]*np.sin(th)).ravel()
    _angles[dim,n]=(u,w);return u,w

def positive_square_tail(radius,a,s,nquad=20,include_zero_atom=True):
    """P(sum max(a_i+s_i Z_i,0)^2 >= radius**2), independent Z_i.

    Angular Gauss-Legendre quadrature with analytic radial Gaussian moments.
    Positive radii include all inactive-subset probabilities exactly.
    """
    radius=np.atleast_1d(radius).astype(float);a=np.asarray(a);s=np.asarray(s)
    assert np.all(s>0) and len(a)==len(s) and 1<=len(a)<=3
    out=np.zeros_like(radius);inactive=ndtr(-a/s)
    for count in range(1,len(a)+1):
        for subset in itertools.combinations(range(len(a)),count):
            idx=np.array(subset);other=[i for i in range(len(a)) if i not in subset]
            atom=float(np.prod(inactive[other]));aa=a[idx];ss=s[idx]
            u,w=directions(count,nquad);A=np.sum((u/ss)**2,axis=1)
            b=(u@(aa/ss**2))/np.sqrt(A);C=np.sum((aa/ss)**2)
            t=np.sqrt(A)[:,None]*radius[None,:]-b[:,None]
            tail=ndtr(-t)
            if count>1:
                phi=np.exp(-t*t/2)/np.sqrt(2*np.pi)
                if count==2:tail=phi+b[:,None]*tail
                else:tail=(t+2*b[:,None])*phi+(1+b[:,None]**2)*tail
            pref=atom*(2*np.pi)**(-(count-1)/2)/np.prod(ss)*A**(-count/2)*np.exp((b*b-C)/2)
            out+=(w*pref)@tail
    out=np.clip(out,1e-300,1.)
    if include_zero_atom:out=np.where(radius<=0,1.,out)
    return out

def load_fields():
    return {w:{y:dict(np.load(B/f'inputs/parent_fields/w{w:.2f}_{y}.npz')) for y in YEARS} for w in WIDTHS}

class FreeCalibration:
    """Tabulated smooth local tail at each declared node; q=0 handled apart."""
    def __init__(self,fields,build=True):
        self.fields=fields;self.radius=np.arange(0.,12.0001,.025);self.tables={};self.interp={}
        self.active={};self.observed={};self.direct={};self.offsets={}
        qa=[]
        for w,fs in fields.items():
            oq=np.zeros(len(MASSES));dq=np.zeros((256,len(MASSES)));tab=[];active=[];exact=[]
            for j,m in enumerate(MASSES):
                ys=[y for y in YEARS if fs[y]['masses'][0]<=m<=fs[y]['masses'][-1]]
                ii=[int(m-fs[y]['masses'][0]) for y in ys]
                a=np.array([fs[y]['a'][i] for y,i in zip(ys,ii)])
                s=np.array([fs[y]['s'][i] for y,i in zip(ys,ii)])
                rr=np.array([fs[y]['observed_r'][i] for y,i in zip(ys,ii)])
                oq[j]=np.sum(np.maximum(rr,0)**2)
                dq[:,j]=sum(np.maximum(fs[y]['validation'][:,i],0)**2 for y,i in zip(ys,ii))
                tails=positive_square_tail(self.radius,a,s,20,False)
                lp=float(positive_square_tail(np.sqrt(oq[j]),a,s)[0]);exact.append(lp)
                tab.append(-np.log(tails));active.append('+'.join(ys))
                if m in [19,39,50,66,80,92,100,101,180,181,250]:
                    radii=np.array([.001,.3,1.,2.,3.,4.,5.,6.,8.,10.,12.])
                    p20=positive_square_tail(radii,a,s,20)
                    p40=positive_square_tail(radii,a,s,40)
                    qa.append(dict(width=w,mass=m,k=len(ys),max_relative_quad_error=float(np.max(abs(p20/p40-1))),atom_zero=float(np.prod(ndtr(-a/s)))))
            self.tables[w]=np.asarray(tab);self.interp[w]=[PchipInterpolator(self.radius,t) for t in tab]
            self.observed[w]=dict(q=oq,local_p=np.asarray(exact));self.direct[w]=dq;self.active[w]=active
        self.qa=qa
    def score(self,w,q):
        q=np.asarray(q);radius=np.sqrt(np.maximum(q,0));out=np.empty_like(radius)
        if radius.ndim==1:
            for j,f in enumerate(self.interp[w]):out[j]=float(f(radius[j]))
        else:
            for j,f in enumerate(self.interp[w]):out[:,j]=f(radius[:,j])
        assert np.max(radius)<self.radius[-1], 'Extend radius table before evaluating this tail'
        return np.where(q>0,out,0.)

def validate_math(cal):
    out={'quadrature_checks':cal.qa}
    rad=np.r_[.001,.1,.5,1.,2.,3.,4.,5.,6.,8.,10.]
    errors=[]
    from math import comb
    for k in [1,2,3]:
        expected=sum(comb(k,j)/2**k*chi2.sf(rad**2,j) for j in range(1,k+1))
        got=positive_square_tail(rad,np.zeros(k),np.ones(k))
        errors.append(float(np.max(abs(got/expected-1))))
    out['central_chibar_max_relative_errors']=errors
    assert max(errors)<1e-10
    assert max(q['max_relative_quad_error'] for q in cal.qa)<1e-7
    all_node_quad_error=0.;errs=[]
    for w in WIDTHS:
        exact=-np.log(cal.observed[w]['local_p']);tab=cal.score(w,cal.observed[w]['q'])
        errs.append(float(np.max(abs(exact-tab))))
        for m in MASSES:
            fs=cal.fields[w];ys=[y for y in YEARS if fs[y]['masses'][0]<=m<=fs[y]['masses'][-1]]
            ix=[int(m-fs[y]['masses'][0]) for y in ys]
            a=np.array([fs[y]['a'][i] for y,i in zip(ys,ix)]);s=np.array([fs[y]['s'][i] for y,i in zip(ys,ix)])
            p20=positive_square_tail([3.,5.,8.,12.],a,s,20);p40=positive_square_tail([3.,5.,8.,12.],a,s,40)
            all_node_quad_error=max(all_node_quad_error,float(np.max(abs(p20/p40-1))))
    out['observed_interpolation_max_abs_logp_error']=errs;assert max(errs)<2e-5
    out['all_nodes_quadrature_max_relative_error']=all_node_quad_error;assert all_node_quad_error<1e-7
    return out

def simulate(cal,N=100000):
    """Independent datasets, paired masses/widths via concatenated count response."""
    factors={};joint_a={};dims={};factor_qa=[]
    for y in YEARS:
        DD=np.column_stack([cal.fields[w][y]['D'] for w in WIDTHS]);K=DD.T@DD
        ev,U=np.linalg.eigh((K+K.T)/2);assert ev.min()>-1e-9
        keep=ev>max(ev.max(),1)*1e-12
        factor=U[:,keep]*np.sqrt(np.maximum(ev[keep],0));factors[y]=factor
        joint_a[y]=np.concatenate([cal.fields[w][y]['a'] for w in WIDTHS])
        dims[y]=len(cal.fields[WIDTHS[0]][y]['masses'])
        factor_qa.append(dict(dataset=y,columns=len(K),rank=int(keep.sum()),min_eigenvalue=float(ev.min()),max_covariance_reconstruction_error=float(np.max(abs(factor@factor.T-K)))))
    rng=np.random.default_rng(58520260919);maxscores={w:[] for w in WIDTHS};maxqs={w:[] for w in WIDTHS}
    for first in range(0,N,1024):
        n=min(1024,N-first);raw={y:joint_a[y]+rng.standard_normal((n,factors[y].shape[1]))@factors[y].T for y in YEARS}
        for iw,w in enumerate(WIDTHS):
            q=np.zeros((n,len(MASSES)))
            for y in YEARS:
                size=dims[y];roots=raw[y][:,iw*size:(iw+1)*size]
                ix=(cal.fields[w][y]['masses']-19).astype(int);q[:,ix]+=np.maximum(roots,0)**2
            maxscores[w].append(cal.score(w,q).max(axis=1));maxqs[w].append(q.max(axis=1))
        if first%20480==0:print('Gaussian paired scans',first+n,'/',N,flush=True)
    return {w:np.concatenate(v) for w,v in maxscores.items()},{w:np.concatenate(v) for w,v in maxqs.items()},factor_qa

def validate_direct_fits(cal):
    from engine import Context,fit,C
    from scipy.optimize import minimize
    from scipy.linalg import block_diag
    rows=[]
    for w in WIDTHS:
        for m in [66.,92.,100.,150.,220.]:
            parts=[];fs=[];ys=[]
            for y in YEARS:
                fld=cal.fields[w][y]
                if not fld['masses'][0]<=m<=fld['masses'][-1]:continue
                obs=np.load(B/f'inputs/null_{y}.npz')['observed']
                part=Context(y,m,w).predict(obs);f=fit([part]);parts.append(part);fs.append(f);ys.append(y)
            n=np.concatenate([p['n'] for p in parts]);b=np.concatenate([p['b'] for p in parts]);L=block_diag(*[p['L'] for p in parts])
            templates=block_diag(*[p['S'][:,None] for p in parts]);scale=1/np.sqrt(np.sum(templates**2/b[:,None],axis=0));T=templates*scale
            design=np.column_stack([T,L]);k=len(parts);pen=np.r_[np.zeros(k),np.ones(L.shape[1])]
            initial=np.r_[[max(f['f']['A'],0)/sc for f,sc in zip(fs,scale)],np.concatenate([f['f']['theta'] if f['f']['A']>0 else f['z']['theta'] for f in fs])]
            def objective(z):
                lam=b+design@z
                if np.min(lam)<=0:return np.inf,np.zeros_like(z)
                return C.poisson_deviance_half(n,lam)+.5*np.sum(pen*z*z),design.T@(1-n/lam)+pen*z
            opt=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(0,None)]*k+[(None,None)]*L.shape[1],options={'ftol':1e-14,'gtol':1e-8,'maxiter':1000,'maxls':40})
            qjoint=max(0.,2*(sum(f['z']['nll'] for f in fs)-opt.fun));qsum=sum(max(f['r'],0)**2 for f in fs)
            qpin=cal.observed[w]['q'][int(m-19)]
            gradient=objective(opt.x)[1];proj=gradient.copy();proj[:k]=np.where(opt.x[:k]<1e-7,np.minimum(proj[:k],0),proj[:k])
            score=float(np.max(abs(proj)))
            # The shared nonnegative model is nested in the free-amplitude model.
            shared=fit(parts);qshared=max(shared['r'],0)**2
            rows.append(dict(width_sigma=w,mass_MeV=m,datasets='+'.join(ys),q_factorized=qsum,q_joint_constrained=qjoint,q_pinned=qpin,
                             absolute_joint_delta=abs(qjoint-qsum),absolute_replay_delta=abs(qsum-qpin),q_shared=qshared,KKT_projected_gradient=score,
                             joint_optimizer_success=bool(opt.success),min_lambda=float(np.min(b+design@opt.x))))
    assert max(r['absolute_joint_delta'] for r in rows)<2e-5
    assert max(r['absolute_replay_delta'] for r in rows)<2e-4
    assert max(r['KKT_projected_gradient'] for r in rows)<2e-6
    assert all(r['q_factorized']+1e-6>=r['q_shared'] for r in rows)
    return rows

def produce(cal,gm,gq,qa):
    rows=[];peaks=[];comparison=[];valid=[]
    for w in WIDTHS:
        obs=cal.observed[w];scores=-np.log(obs['local_p']);vs=cal.score(w,cal.direct[w]);vm=vs.max(axis=1)
        widthrows=[]
        for j,m in enumerate(MASSES):
            row=dict(width_sigma=w,mass_MeV=m,datasets=cal.active[w][j],n_active=len(cal.active[w][j].split('+')),
                     q_free=float(obs['q'][j]),q_R=float(obs['q'][j]),score=float(scores[j]),local_p=float(obs['local_p'][j]),
                     local_Z_signed=float(norm.isf(obs['local_p'][j])),local_Z=max(0.,float(norm.isf(obs['local_p'][j]))))
            for prefix,values,threshold in [('gaussian_global',gm[w],scores[j]),('direct_global',vm,scores[j]),('direct_local',cal.direct[w][:,j],obs['q'][j])]:
                row.update({prefix+'_'+k:v for k,v in mc_tail(values,threshold).items()})
            row.update(global_p=row['gaussian_global_p_plus1'],global_Z=row['gaussian_global_Z_display'],
                       direct_global_p=row['direct_global_p_plus1'],direct_local_p=row['direct_local_p_plus1'])
            row['domain']='full_19_250_MeV';widthrows.append(row)
        j=int(np.argmax(scores));jq=int(np.argmax(obs['q']))
        peak=dict(widthrows[j],peak_selection='minimum local p over full declared domain',raw_q_peak_mass_MeV=float(MASSES[jq]),raw_q_max=float(obs['q'][jq]))
        peaks.append(peak);rows+=widthrows
        q95=float(np.quantile(gm[w],.95));v=mc_tail(vm,q95)
        valid.append(dict(width_sigma=w,threshold_gaussian95=q95,**v))
        np.savez_compressed(B/f'fields/free_w{w:.2f}.npz',masses=MASSES,observed_q=obs['q'],local_p=obs['local_p'],observed_score=scores,
                            validation_q=cal.direct[w],validation_score=vs,direct_maximum=vm,gaussian_maximum=gm[w],gaussian_q_maximum=gq[w],
                            calibration_radius=cal.radius,calibration_neglogp_positive=cal.tables[w])
    write_csv(B/'results/free_curves.csv',rows);write_csv(B/'results/free_peaks.csv',peaks);write_csv(B/'results/free_global_validation.csv',valid)
    with open(B/'inputs/parent_results/peaks.csv') as f:parent=list(csv.DictReader(f))
    for p in parent:
        if p['domain']=='full' and p['scope'] in ['combined','fisher','stouffer']:
            comparison.append(dict(width_sigma=float(p['width_sigma']),method={'combined':'shared_coupling','fisher':'Fisher','stouffer':'signed_Stouffer'}[p['scope']],peak_mass_MeV=float(p['mass_MeV']),
                local_p=float(p['local_p']),local_Z=float(p['local_Z']),gaussian_global_p=float(p['global_p']),gaussian_global_Z=float(p['global_Z']),
                direct_global_k=int(p['direct_global_k']),direct_global_N=int(p['direct_global_N']),direct_global_lo95=float(p['direct_global_lo95']),direct_global_hi95=float(p['direct_global_hi95'])))
    for p in peaks:
        comparison.append(dict(width_sigma=p['width_sigma'],method='free_amplitudes',peak_mass_MeV=p['mass_MeV'],local_p=p['local_p'],local_Z=p['local_Z'],
             gaussian_global_p=p['gaussian_global_p_plus1'],gaussian_global_Z=p['gaussian_global_Z_display'],direct_global_k=p['direct_global_k'],direct_global_N=p['direct_global_N'],direct_global_lo95=p['direct_global_lo95'],direct_global_hi95=p['direct_global_hi95']))
    write_csv(B/'results/free_comparison.csv',comparison)
    summary=dict(peaks=peaks,at_92=[r for r in rows if r['mass_MeV']==92],null_model='Pinned source spectra, Poisson count fluctuations; GP conditioning repeated for direct scans',
                 direct_calibration='256 reused paired full Poisson exact-profile scans; Monte Carlo interval retained',
                 gaussian_calibration='100000 full correlated derivative-response scans; exact subset/polar local tail of independent a+sZ signed roots',
                 global_ordering='max over m in integer 19..250 of -log calibrated local free-amplitude tail',
                 exact_likelihood='q_R(m)=sum_d max(r_d(m),0)^2, independent dataset nuisances and amplitudes',
                 width_selection_corrected=False,method_selection_corrected=False,qa=qa)
    (B/'results/free_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

def main():
    started=time.monotonic();fields=load_fields();cal=FreeCalibration(fields);qa=validate_math(cal)
    print('local calibration complete',round(time.monotonic()-started,1),flush=True)
    fitrows=validate_direct_fits(cal);write_csv(B/'qa/free_direct_fit_checks.csv',fitrows)
    print('exact likelihood replay checks complete',round(time.monotonic()-started,1),flush=True)
    gm,gq,fq=simulate(cal);qa['joint_width_factors']=fq;qa['elapsed_seconds']=time.monotonic()-started
    qa.update(gaussian_seed=58520260919,gaussian_scans=100000,gaussian_batch_size=1024,
              MC_intervals='Clopper-Pearson exact 95% two-sided; upper95 is one-sided 95%; plus-one estimate (k+1)/(N+1)')
    produce(cal,gm,gq,qa)
    (B/'qa/free_validation.json').write_text(json.dumps(dict(complete=True,**qa),indent=2)+'\n')
    print('complete',round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':main()
