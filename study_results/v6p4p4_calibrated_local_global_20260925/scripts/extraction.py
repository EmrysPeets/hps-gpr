"""Conditional observed profiles with neighboring 2016/2021 signal MC."""
from pathlib import Path
import os,sys,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'inputs/v6p1/scripts'))
import common as C
from archived_templates import TemplateBank
from observed_templates import BANK as BANK2021
import numpy as np
import pandas as pd
from scipy.linalg import cholesky,cho_solve,solve_triangular,block_diag
from scipy.special import ndtr

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv(p,rows):
    pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')

class Bank2016(TemplateBank):
    def __init__(self):
        self.samples={}
        d=pd.read_csv(B/'inputs/v64/results/centers_and_shapes.csv',float_precision='round_trip').set_index('mass_MeV')
        for m,r in d[d.primary_domain & d.valid].iterrows():
            a=np.load(B/'inputs/v64/histograms'/f'm{m:03d}.npz')
            e,y=a['edges_MeV'],a['counts'];total=float(y.sum())
            assert np.all(a['flow_counts']==0)
            self.samples[int(m)]=dict(edges=e,counts=y,total=total,under=0.,over=0.,
                cdf=np.r_[0,np.cumsum(y)]/total,center=float(r.center_MeV),width=float(r.sigma_core_MeV))
        self.anchors=np.array(sorted(self.samples))

BANKS={'2016':Bank2016(),'2021':BANK2021}
KINDS2016=('gaussian','gaussian_mc_window','mc')
METHODS=('all_gaussian','mc2021','mc2016_2021')

def years(m):
    return [y for y in ('2015','2016','2021') if y=='2021' or y=='2015' and m<=100 or y=='2016' and m<=175]

def conversion(y,m):
    d=C.DATA[y];x=m/1000.;s=C.sigma(y,m);e=d['native_edges'];w=np.diff(e)
    overlap=np.maximum(0.,np.minimum(e[1:],x+1.64*s)-np.maximum(e[:-1],x-1.64*s))
    density=float(np.sum(d['native_counts']*overlap/w)/(3.28*s))
    return float(3*np.pi*x*float(d['frad_effective'])*density/(2/137.))*1e-8

def branch(m):
    r=(105.6583745/m)**2
    return float(1+np.sqrt(max(0,1-4*r))*(1+2*r)) if m>211.316749 else 1.

class Context:
    def __init__(self,year,mass,kind):
        self.year,self.mass,self.kind=year,int(mass),kind
        self.data=C.DATA[year];d=self.data
        self.sigma_ref=C.sigma(year,mass)*1000
        self.gaussian_center=float(mass if year!='2021' else mass-3.2243308692909953-2.213992811446465*np.log(mass/150.))
        self.center,self.width=BANKS[year].parameters(mass) if year in BANKS else (float(mass),self.sigma_ref)
        self.mc_categories=BANKS[year].categories(mass,d['edges']*1000) if year in BANKS else None
        gp=np.diff(ndtr((d['edges']*1000-self.gaussian_center)/self.sigma_ref))
        if year in ('2015','2016'):
            # Preserve the established Gaussian full-spectrum normalization.
            gp=gp/gp.sum()
            self.gaussian_categories=np.r_[0.,gp,0.]
        else:
            F=ndtr((d['edges']*1000-self.gaussian_center)/self.sigma_ref)
            self.gaussian_categories=np.r_[F[0],gp,1-F[-1]]
        if kind=='gaussian':
            lo=self.gaussian_center-2.25*self.sigma_ref;hi=self.gaussian_center+2.25*self.sigma_ref
        else:
            left,right=(-3.5,3.5) if year=='2016' else (-4.,3.)
            lo=self.center+left*self.width;hi=self.center+right*self.width
        self.requested_low,self.requested_high=float(lo),float(hi)
        self.fit=(d['x']*1000>=lo)&(d['x']*1000<=hi);self.guard=self.fit.copy()
        self.categories=self.mc_categories if kind=='mc' else self.gaussian_categories
        self.probability=self.categories[1:-1]
        self.const,self.ls=C.kernel_state(year,mass)
        xt=d['x'][~self.guard];xq=d['x'][self.fit]
        self.K=C.kernel(xt,xt,self.const,self.ls)
        self.Kqt=C.kernel(xq,xt,self.const,self.ls);self.Kqq=C.kernel(xq,xq,self.const,self.ls)
        assert self.fit.sum()>3 and np.sum(d['x']*1000<lo)>=3 and np.sum(d['x']*1000>hi)>=3
        assert self.categories.min()>=0 and abs(self.categories.sum()-1)<1e-12

    def prediction(self,counts,full=False):
        d=self.data;n=np.asarray(counts,float)[~self.guard];pos=n>0
        target=np.zeros_like(n);target[pos]=np.log(n[pos]);alpha=np.ones_like(n);alpha[pos]=1/n[pos]
        K=self.K.copy();K.flat[::len(K)+1]+=alpha
        L=cholesky(K,lower=True,check_finite=False)
        Kqt=C.kernel(d['x'],d['x'][~self.guard],self.const,self.ls) if full else self.Kqt
        Kqq=C.kernel(d['x'],d['x'],self.const,self.ls) if full else self.Kqq
        latent=Kqt@cho_solve((L,True),target,check_finite=False)
        v=solve_triangular(L,Kqt.T,lower=True,check_finite=False)
        cov=Kqq-v.T@v;cov=.5*(cov+cov.T)
        b=np.exp(latent+.5*np.maximum(np.diag(cov),0))
        return b,np.outer(b,b)*np.expm1(np.clip(cov,-40,40))

    def part(self,counts=None):
        n=self.data['n'] if counts is None else np.asarray(counts)
        b,cov=self.prediction(n);L,diag=C.factor_cov(cov,b)
        return dict(year=self.year,context=self,counts=n,n=n[self.fit],b=b,cov=cov,L=L,
                    S=self.probability[self.fit]*conversion(self.year,self.mass),load=diag['load'])

    def geometry(self):
        d=self.data;i=np.flatnonzero(self.fit)
        r=dict(year=self.year,mass_MeV=self.mass,kind=self.kind,center_MeV=self.center,
            core_width_MeV=self.width,gaussian_center_MeV=self.gaussian_center,sigma_ref_MeV=self.sigma_ref,
            requested_low_MeV=self.requested_low,requested_high_MeV=self.requested_high,
            actual_low_MeV=float(d['edges'][i[0]]*1000),actual_high_MeV=float(d['edges'][i[-1]+1]*1000),
            fit_bins=int(self.fit.sum()),training_bins=int((~self.guard).sum()),
            left_training_bins=int(np.sum(d['x']*1000<self.requested_low)),right_training_bins=int(np.sum(d['x']*1000>self.requested_high)),
            fit_fraction=float(self.probability[self.fit].sum()),training_fraction=float(self.probability[~self.guard].sum()),
            below_support=float(self.categories[0]),above_support=float(self.categories[-1]),
            kernel_const=self.const,kernel_length_scale=self.ls,conversion_events_per_psi=conversion(self.year,self.mass))
        if self.mc_categories is not None:
            p=self.mc_categories[1:-1]
            r.update(MC_fit_fraction=float(p[self.fit].sum()),MC_training_fraction=float(p[~self.guard].sum()),
                MC_outside_fraction=float(self.mc_categories[0]+self.mc_categories[-1]),
                interpolation_anchors_json=json.dumps(BANKS[self.year].neighbors(self.mass)))
        return r

def model(parts,tolerance=2e-7):
    b=np.concatenate([p['b'] for p in parts]);n=np.concatenate([p['n'] for p in parts])
    L=block_diag(*[p['L'] for p in parts]);S=np.concatenate([p['S'] for p in parts])
    return C.OneSignalProfile(b,L,S,score_tolerance=tolerance),n

def solve(parts,m,method,scope,save=None,limit=True):
    errors=[]
    for tolerance in (2e-7,2e-9):
        try:
            mod,n=model(parts,tolerance)
            if limit:
                q=mod.limit(n,details=True);free=q.pop('free');null=q.pop('null');trace=q.pop('trace')
                assert q['ok'] and abs(q['cls']-.1)<2e-6
            else:
                free=mod.fit(n);null=mod.fit(n,fixed=0,initial=free['theta']);trace=[]
                stat=2*(null['nll']-free['nll']);assert stat>=-2e-6
                signed=float(np.sign(free['A'])*np.sqrt(max(0.,stat)))
                q=dict(Ahat=free['A'],sigma_A=free['sigma'],signed_r=signed,p0_fixed_mass=float(ndtr(-max(0,signed))),
                    max_score=max(free['score'],null['score']),min_lambda=min(free['min_lambda'],null['min_lambda']))
            assert q['max_score']<3e-5 and q['min_lambda']>0
            out=dict(mass_MeV=int(m),scope=scope,method=method,campaigns='+'.join(p['year'] for p in parts),
                psi_hat=q['Ahat'],sigma_psi=q['sigma_A'],signed_root=q['signed_r'],q0=max(0,q['signed_r'])**2,
                Z_local=max(0,q['signed_r']),p0_asymptotic=q['p0_fixed_mass'],max_score=q['max_score'],
                min_lambda=q['min_lambda'],max_covariance_load=max(p['load'] for p in parts),valid=True,origin='fresh_fit')
            if limit:out.update(psi90=q['A90'],epsilon2_90_ee_proxy=q['A90']*1e-8,
                epsilon2_90_visible_legacy=q['A90']*1e-8*branch(m),cls=q['cls'],n_profiles=q['n_profiles'])
            if len(parts)==1:
                factor=conversion(parts[0]['year'],m)
                out.update(Ahat=q['Ahat']*factor,sigma_A=q['sigma_A']*factor)
                if limit:out['A90']=q['A90']*factor
            if save is not None:
                payload=dict(n=n,b=mod.b,L=mod.L,S=mod.S,free_theta=free['theta'],null_theta=null['theta'],
                    profiled_background_only=null['bfit'],profiled_background_signed=free['bfit'],
                    profiled_signed_signal=free['A']*mod.S,profiled_signed_total=free['lam'],
                    summary_json=json.dumps(out),profile_trace_json=json.dumps(trace))
                if len(parts)==1:
                    part=parts[0];ctx=part['context'];fullb,fullcov=ctx.prediction(part['counts'],full=True)
                    payload.update(edges_GeV=ctx.data['edges'],x_GeV=ctx.data['x'],counts=part['counts'],
                        fit_mask=ctx.fit,guard_mask=ctx.guard,signal_probability=ctx.probability,
                        full_MC_categories=ctx.mc_categories,prefit_GP_mean=fullb,prefit_GP_covariance=fullcov,
                        fit_prefit_mean=part['b'],fit_prefit_covariance=part['cov'],fit_counts=part['n'],
                        geometry_json=json.dumps(ctx.geometry()))
                else:
                    payload.update(campaigns=np.array([p['year'] for p in parts]),
                        campaign_bin_lengths=np.array([len(p['n']) for p in parts]),
                        campaign_rank_lengths=np.array([p['L'].shape[1] for p in parts]))
                np.savez_compressed(save,**payload)
            return out
        except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
    raise RuntimeError(f'{scope}/{method}/{m}: '+ '; '.join(errors))

def parts_for_combined(m,method,counts=None):
    result=[]
    for y in years(m):
        kind='mc' if (y=='2021' and method!='all_gaussian') or (y=='2016' and method=='mc2016_2021') else 'gaussian'
        ctx=Context(y,m,kind);result.append(ctx.part(None if counts is None else counts[y]))
    return result

def protocol():
    p=dict(version='6.4.1',year2016_masses=list(range(40,176)),combined_masses=list(range(60,241)),
        methods=list(METHODS),individual2016_methods=list(KINDS2016),
        interpolation='Linear core center and core width; convex mixture of aligned full empirical CDFs of neighboring masses; direct at anchors; no extrapolation',
        windows={'2016_MC':[-3.5,3.5],'2021_MC':[-4.,3.],'units':'fitted core widths about interpolated fitted core center','fit_equals_GP_exclusion':True},
        gaussian_reference='2015/2016 centered at generated mass; 2021 established logarithmically shifted Gaussian; +/-2.25 archived reference resolution; inherited full-spectrum Gaussian normalization for 2015/2016',
        included_campaigns='Same under all methods: 2015 full through100; 2016 full through175; 2021 10pct through240',
        shared_parameter='psi=epsilon2/1e-8 with inherited generated-mass conversion; independent campaign GP constraints',
        limits='Conditional observed bounded profile CLs90 with asymptotic sampling tails',
        local_p='Phi(-max(signed likelihood root,0)); not a global p-value',
        peak_selection='Top2 positive local maxima of 2016 MC q0 with disjoint actual fit-bin masks; ties go to lower mass. Combined top2 selected analogously requiring no overlapping fitted bins in a shared campaign.',
        scripts={name:sha(B/'scripts'/name) for name in ['extraction.py','run_scan.py','archived_templates.py','observed_templates.py']},
        frozen_input_manifest_sha256=sha(B/'provenance/input_manifest.sha256'))
    path=B/'provenance/protocol.json'
    if path.exists():assert json.loads(path.read_text())==p,'Frozen extraction protocol changed'
    else:write(path,p)
    return sha(path)
