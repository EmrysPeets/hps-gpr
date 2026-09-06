#!/usr/bin/env python3
"""Independent reconstruction from stored spectra and likelihood components."""
from pathlib import Path
import os, sys, json, hashlib, csv
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
checks=0; maxima={}
def check(value, message):
    global checks
    checks+=1
    assert value, message
def error(name, value, bound):
    value=float(value); maxima[name]=max(maxima.get(name,0.),value)
    check(value<bound,(name,value,bound))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def halfdev(n,lam):
    t=(lam-n)/n
    return float(np.sum(n*(t-np.log1p(t))))
def source_manifest(folder):
    entries=list(csv.DictReader((folder/'MANIFEST.csv').open()))
    for row in entries: check(sha(ROOT/row['path'])==row['sha256'],row['path'])
    return len(entries)

parent_counts={}
for folder in ['v4p9p16_probability_echo_review_20260906','v4p9p16_presentation_extractions_20260906']:
    parent_counts[folder]=source_manifest(HERE.parent/folder)
metrics=pd.read_csv(HERE/'derived/oscillation_metrics.csv')
profile_count=0; saved_components=0
for year,nmass in [('2015',72),('2016',142),('2021',201)]:
    d=HERE/'derived'/year; a=np.load(d/'inputs.npz'); contract=json.loads((d/'input_contract.json').read_text())
    qa=json.loads((d/'scan_qa.json').read_text()); scan=pd.read_csv(d/'scans.csv')
    check(qa['passed'] and qa['profile_tests']==42*nmass and qa['masses']==nmass,'Complete scan')
    check(sha(d/'inputs.npz')==contract['inputs_sha256']==qa['inputs_sha256'],'Input hash')
    check(sha(d/'scans.csv')==qa['csv_sha256'],'Scan hash')
    for name,digest in contract['source_sha256'].items(): check(sha(ROOT/name)==digest,name)
    names=a['lane_names'].tolist(); spectra=dict(zip(names,a['spectra']))
    check(len(names)==42 and len(set(names))==42 and len(scan)==42*nmass,'42 complete fixed spectra')
    check(scan.groupby('lane').size().eq(nmass).all(),'All mass coordinates')
    check(scan.groupby('mass_MeV').remote.nunique().eq(1).all(),'Common remote mask')
    check(scan[['r','Ahat_window','sigma_A_window','half_deviance_fit','half_deviance_null','max_score','min_lambda']].apply(np.isfinite).all().all(),'Finite fits')
    check(scan.sigma_A_window.gt(0).all() and scan.min_lambda.gt(0).all(),'Positive variances and expectations')
    check(scan.max_score.lt(1e-5).all(),'Numerical stationarity')
    error('archived_reference_root_delta',scan.reference_root_delta.abs().max(),2e-3)
    holes=pd.read_csv(d/'holes.csv')
    masks={which:a['observed__hole_'+which] for which in ['first','second','union']}
    check(not np.any(masks['first']&masks['second']),'Disjoint holes')
    wide=np.any([abs(a['x_GeV']*1000-h.mass_MeV)<=3*h.sigma_MeV for h in holes.itertuples()],axis=0)
    for source in ['observed','reference']:
        original=a[source]
        check(np.array_equal(spectra[source+'_original'],original),'Original unchanged')
        for k in masks: check(np.array_equal(a[source+'__hole_'+k],masks[k]),'Matched source hole geometry')
        mu=a[source+'__latent_mu']; V=a[source+'__latent_cov']; F=a[source+'__latent_factor']
        error('latent_factor_absolute',np.max(abs(F@F.T-V)),1e-12)
        error('mean_count_relative',np.max(abs(np.exp(mu+.5*np.diag(V))/a[source+'__fill_mean']-1)),1e-12)
        check(np.linalg.eigvalsh(V).min()>-1e-12,'Positive latent covariance')
        for name,spectrum in spectra.items():
            if not name.startswith(source+'_'):continue
            suffix=name[len(source)+1:]
            mask=np.zeros(len(original),bool) if suffix=='original' else wide if 'wide' in suffix else masks['union'] if suffix.startswith('both') else masks[suffix.split('_')[0]]
            check(np.array_equal(spectrum[~mask],original[~mask]),'Unchanged outside hole '+name)
            if '_rep' in name:
                check(np.equal(spectrum[mask],np.floor(spectrum[mask])).all(),'Integer Poisson replacements')
                rep=name.split('_rep')[1]
                check(np.array_equal(spectrum[mask],spectra[source+'_both_rep'+rep][mask]),'Paired replicate reuse')
    components=np.load(d/'components.npz')
    lookup=scan.set_index(['mass_MeV','lane']); remote=[]
    for mass in contract['masses_MeV']:
        counts=components[f'm{mass:03d}__observed_original__counts']
        matches=[i for i in np.flatnonzero(a['observed']==counts[0]) if np.array_equal(a['observed'][i:i+len(counts)],counts)]
        check(len(matches)==1,'Unique contiguous native fit window')
        ix=np.arange(matches[0],matches[0]+len(counts))
        expected_remote=not masks['union'][ix].any(); remote.append(expected_remote)
        check(bool(lookup.loc[(mass,'observed_original'),'remote'])==expected_remote,'Independent remote geometry')
        for lane in names:
            pref=f'm{mass:03d}__{lane}__'
            if pref+'counts' not in components:continue
            row=lookup.loc[(mass,lane)]; n=components[pref+'counts']; b=components[pref+'gp_mean']; L=components[pref+'L']; w=components[pref+'w']
            check(np.array_equal(n,spectra[lane][ix]),'Fixed whole-spectrum input per fit')
            vals=[]
            for side,A in [('free',row.Ahat_window),('null',0.)]:
                theta=components[pref+side+'_theta']; lam=b+L@theta+A*w
                error('profile_lambda_relative',np.max(abs(lam/components[pref+side+'_lambda']-1)),1e-10)
                value=halfdev(n,lam)+.5*theta@theta; vals.append(value)
                reported=row.half_deviance_fit if side=='free' else row.half_deviance_null
                error('profile_nll_absolute',abs(value-reported),2e-7)
                g=L.T@(1-n/lam)+theta
                error('profile_nuisance_score',np.max(abs(g)),1e-5)
                if side=='free':
                    H=(L.T*(n/lam**2))@L+np.eye(L.shape[1]); J=(L.T*(n/lam**2))@w
                    information=float(np.sum(n*w*w/lam**2)-J@np.linalg.solve(H,J))
                    check(information>0,'Profile signal information positive')
                    error('profile_sigma_relative',abs(information**-.5/row.sigma_A_window-1),1e-6)
                    error('profile_amplitude_scaled_score',abs(w@(1-n/lam))*row.sigma_A_window,1e-5)
            root=np.sign(row.Ahat_window)*np.sqrt(max(0,2*(vals[1]-vals[0])))
            error('profile_root_absolute',abs(root-row.r),2e-5)
            saved_components+=1
    wide_table=scan.pivot(index='mass_MeV',columns='lane',values='r')
    for row in metrics[metrics.dataset.eq(int(year))].itertuples():
        mask=np.ones(nmass,bool) if row.selection=='full' else np.array(remote)
        values=wide_table[row.lane].to_numpy(); base=wide_table[row.source+'_original'].to_numpy()
        check(row.n_points==mask.sum(),'Metric mass count')
        error('metric_std_absolute',abs(np.std(values[mask])-row.std),1e-10)
        error('metric_ratio_absolute',abs(np.std(values[mask])/np.std(base[mask])-row.retained_std),1e-10)
        crossings=np.sum((values[1:]*values[:-1]<0)&mask[1:]&mask[:-1])
        check(crossings==row.sign_transitions,'Contiguous sign transitions')
    profile_count+=len(scan)
    print('Verified removal',year,flush=True)

# Independent 48-point Gaussian quadrature and likelihood derivatives for the
# conventional fits. No imports from the fitting or GP implementation.
traditional=HERE/'traditional'; summary=json.loads((traditional/'derived/summary.json').read_text())
check(summary['passed'] and summary['completed']==30 and not summary['failures'],'All 30 traditional fits')
for name,digest in summary['source_sha256'].items():check(sha(ROOT/name)==digest,name)
for name,digest in summary['output_sha256'].items():check(sha(ROOT/name)==digest,name)
execution=json.loads((traditional/'derived/execution_contract.json').read_text())
check(sha(traditional/'run_traditional.py')==execution['script_sha256'],'Traditional executed code')
check(sha(HERE/'derived/persistence_trigger.json')==execution['trigger_sha256'],'Frozen trigger')
t=pd.read_csv(traditional/'derived/fit_summary.csv'); check(len(t)==30,'Conventional table complete')
for row in t.itertuples():
    a=np.load(traditional/'derived/points'/f'{row.fit_id}__{row.variant}.npz')
    edges=a['edges_MeV']; n=a['counts']; xx,ww=np.polynomial.legendre.leggauss(48)
    nodes=(edges[1:]+edges[:-1])[:,None]/2+np.diff(edges)[:,None]*xx/2
    weight=np.diff(edges)[:,None]*ww/2
    coordinate=2*(nodes-edges[0])/(edges[-1]-edges[0])-1
    vander=np.polynomial.chebyshev.chebvander if row.basis=='chebyshev' else np.polynomial.legendre.legvander
    P=vander(coordinate,row.degree); signal=np.diff(ndtr((edges-row.mass_MeV)/row.sigma_MeV))
    error('traditional_template_absolute',np.max(abs(signal-a['signal_bin_probability'])),1e-12)
    input_arrays=np.load(HERE/'derived'/str(row.dataset)/'inputs.npz')
    check(np.array_equal(n,input_arrays['observed'][a['native_indices']]),'Conventional uses original counts')
    vals=[]
    for side in ['free','null']:
        beta=a['free_coefficients'] if side=='free' else a['null_coefficients']
        density=np.exp(P@beta)*weight; b=density.sum(1)
        lam=b+(row.amplitude_full*signal if side=='free' else 0)
        error('traditional_lambda_relative',np.max(abs(lam/a['total_'+side]-1)),1e-10)
        v=halfdev(n,lam); vals.append(v)
        error('traditional_nll_absolute',abs(v-(row.nll if side=='free' else row.null_nll)),1e-7)
        J=np.einsum('ij,ijk->ik',density,P); V=np.einsum('ij,ijk,ijl->ikl',density,P,P)
        rr=1-n/lam
        if side=='free':
            J=np.column_stack((signal*row.amplitude_scale,J))
            add=np.zeros((len(n),row.degree+2,row.degree+2)); add[:,1:,1:]=V;V=add
        g=J.T@rr; H=(J.T*(n/lam**2))@J+np.einsum('i,ijk->jk',rr,V)
        error('traditional_scaled_stationarity',np.max(abs(g)/np.sqrt(np.diag(H))),1e-7)
        check(np.linalg.eigvalsh(H).min()>0,'Conventional Hessian positive definite')
        if side=='free':
            err=np.sqrt(np.linalg.inv(H)[0,0])*row.amplitude_scale
            error('traditional_sigma_relative',abs(err/row.sigma_amplitude_full-1),1e-7)
    r=np.sign(row.amplitude_full)*np.sqrt(max(0,2*(vals[1]-vals[0])))
    error('traditional_root_absolute',abs(r-row.root),1e-6)
    error('traditional_p_absolute',abs(norm.sf(max(r,0))-row.p0_nominal),1e-8)
out=dict(passed=True,checks=checks,complete_profile_tests=profile_count,independently_reconstructed_profile_tests=saved_components,traditional_fits=30,frozen_parent_entries=parent_counts,max_errors=maxima,
         scope='Algebra, stationarity, fixed spectra, source identity and completeness; not background adequacy or probability calibration.')
(HERE/'qa/numerical_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
