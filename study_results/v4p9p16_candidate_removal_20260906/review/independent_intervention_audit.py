#!/usr/bin/env python3
"""Read saved interventions and independently reconstruct descriptive metrics.

No fits, toys, data unblinding, or changes outside review/.
"""
from pathlib import Path
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.special import ndtr
from numpy.polynomial import chebyshev,legendre

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[1]
CANDIDATES={'2015':[51,21],'2016':[90,117],'2021':[78,65]}
checks=[]
errors={}
bindings={}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path):
    bindings[str(path.relative_to(ROOT))]=sha(path)


def check(label,condition):
    checks.append({'check':label,'passed':bool(condition)})


def close(label,actual,expected,tolerance=1e-10):
    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    errors[label]=error
    check(label,np.isfinite(error) and error<=tolerance)


def half_deviance(n,lam):
    n=np.asarray(n,dtype=np.longdouble);lam=np.asarray(lam,dtype=np.longdouble)
    t=(lam-n)/n
    return float(np.sum(n*(t-np.log1p(t))))


def main():
    metrics_path=STUDY/'derived/oscillation_metrics.csv';bind(metrics_path)
    metrics=pd.read_csv(metrics_path,dtype={'dataset':str},float_precision='round_trip')
    input_summary=[];primary=[];alternate=[];poly_fit=[];replica_summary=[]
    total_roots=0
    for year,candidates in CANDIDATES.items():
        folder=STUDY/'derived'/year
        paths={name:folder/name for name in ('inputs.npz','input_contract.json','scans.csv','scan_qa.json','holes.csv','components.npz')}
        for path in paths.values():bind(path)
        contract=json.loads(paths['input_contract.json'].read_text());qa=json.loads(paths['scan_qa.json'].read_text())
        arrays=np.load(paths['inputs.npz']);components=np.load(paths['components.npz'])
        scans=pd.read_csv(paths['scans.csv'],float_precision='round_trip')
        holes=pd.read_csv(paths['holes.csv'],float_precision='round_trip').set_index('mass_MeV')
        names=arrays['lane_names'].tolist();spectra=dict(zip(names,arrays['spectra']))
        grid=np.asarray(contract['masses_MeV']);x=arrays['x_GeV'];edges=arrays['edges_GeV']*1000
        check(year+' input hash',sha(paths['inputs.npz'])==contract['inputs_sha256']==qa['inputs_sha256'])
        check(year+' scan hash',sha(paths['scans.csv'])==qa['csv_sha256'])
        check(year+' complete root field',len(scans)==42*len(grid) and qa['profile_tests']==len(scans) and qa['masses']==len(grid) and qa['passed'])
        check(year+' unique mass/lane',not scans.duplicated(['mass_MeV','lane']).any())
        check(year+' finite roots',np.isfinite(scans.r).all())
        for name in names:
            check(year+' grid '+name,np.array_equal(scans[scans.lane==name].mass_MeV,grid))
        total_roots+=len(scans)
        check(year+' frozen candidates',contract['candidates_MeV']==candidates)
        check(year+' source-specific hole agreement',np.array_equal(arrays['observed__hole_union'],arrays['reference__hole_union']))
        reconstructed_holes=[np.abs(x-mass/1000)<=2.25*holes.loc[mass,'sigma_MeV']/1000 for mass in candidates]
        wide_holes=[np.abs(x-mass/1000)<=3*holes.loc[mass,'sigma_MeV']/1000 for mass in candidates]
        for source in ('observed','reference'):
            original=arrays[source];prefix=source+'__';union=arrays[prefix+'hole_union']
            for j,which in enumerate(('first','second')):
                check(f'{year} {source} {which} mask',np.array_equal(arrays[prefix+'hole_'+which],reconstructed_holes[j]))
            check(f'{year} {source} union',np.array_equal(union,np.logical_or(*reconstructed_holes)))
            check(f'{year} {source} disjoint holes',not np.any(reconstructed_holes[0]&reconstructed_holes[1]))
            mu=arrays[prefix+'latent_mu'];V=arrays[prefix+'latent_cov'];F=arrays[prefix+'latent_factor'];mean=arrays[prefix+'fill_mean']
            close(f'{year} {source} covariance factor',V,F@F.T,1e-13)
            close(f'{year} {source} covariance symmetry',V,V.T,1e-13)
            check(f'{year} {source} covariance PSD',np.linalg.eigvalsh(V).min()>-1e-13)
            close(f'{year} {source} lognormal expectation',mean/np.exp(mu+.5*np.diag(V)),1.,1e-13)
            close(f'{year} {source} mean fill identity',spectra[source+'_both_mean'][union],mean,0.)
            check(f'{year} {source} original identity',np.array_equal(original,spectra[source+'_original']))
            for name in [n for n in names if n.startswith(source+'_')]:
                if name.endswith('original'):mask=np.zeros(len(x),bool)
                elif '_first_' in name:mask=reconstructed_holes[0]
                elif '_second_' in name:mask=reconstructed_holes[1]
                elif name.endswith('_wide_mean'):mask=np.logical_or(*wide_holes)
                else:mask=union
                check(f'{year} exterior identity {name}',np.array_equal(spectra[name][~mask],original[~mask]))
                check(f'{year} positive {name}',np.all(spectra[name]>0))
                if '_rep' in name:
                    check(f'{year} integer replacement {name}',np.array_equal(spectra[name][mask],np.rint(spectra[name][mask])))
                    suffix=name.split('_rep')[1]
                    close(f'{year} paired draw {name}',spectra[name][mask],spectra[source+'_both_rep'+suffix][mask],0.)
            info=contract['replacement_details'][source]['primary']
            check(f'{year} {source} no added latent observation noise',info['latent_covariance_observation_noise_added'] is False)
            input_summary.append(dict(dataset=year,source=source,native_bins=len(x),hole_bins=info['hole_bins'],latent_min_eigenvalue_before_clip=info['latent_min_eigenvalue'],latent_max_eigenvalue=info['latent_max_eigenvalue'],replicas=info['conditional_replicas']))
            # Reconstruct each alternative polynomial from its saved coefficient
            # vector with 16 rather than 8 quadrature nodes, without a fit.
            for j,info in enumerate(contract['replacement_details'][source]['polynomial']):
                mass=info['mass_MeV'];sigma=holes.loc[mass,'sigma_MeV'];local=np.abs(x-mass/1000)<=7*sigma/1000
                idx=np.flatnonzero(local);e=edges[idx[0]:idx[-1]+2];center=.5*(e[:-1]+e[1:]);half=.5*np.diff(e)
                nodes,weights=legendre.leggauss(16);points=center[:,None]+half[:,None]*nodes
                P=chebyshev.chebvander(2*(points-e[0])/(e[-1]-e[0])-1,info['degree'])
                b=np.sum(half[:,None]*weights*np.exp(P@np.asarray(info['coeff'])),axis=1)
                retained=~union[idx];h=reconstructed_holes[j][idx]
                close(f'{year} {source} polynomial fill {mass}',spectra[source+'_both_poly_mean'][idx[h]]/b[h],1.,2e-12)
                close(f'{year} {source} polynomial deviance {mass}',half_deviance(original[idx][retained],b[retained]),info['half_deviance'],1e-7)
                check(f'{year} {source} polynomial sideband dof {mass}',info['dof']==int(retained.sum())-info['degree']-1)
                poly_fit.append(dict(dataset=year,source=source,mass_MeV=mass,deviance=2*info['half_deviance'],dof=info['dof'],deviance_per_dof=2*info['half_deviance']/info['dof']))
        wide=scans.pivot(index='mass_MeV',columns='lane',values='r').reindex(grid)
        original_rows=scans[scans.lane=='observed_original'].set_index('mass_MeV').reindex(grid)
        remote=[]
        for mass in grid:
            observed_window=components[f'm{mass:03d}__observed_original__counts']
            sliding=np.lib.stride_tricks.sliding_window_view(arrays['observed'],len(observed_window))
            matches=np.flatnonzero(np.all(sliding==observed_window,axis=1))
            check(f'{year} unique original native window {mass}',len(matches)==1)
            if len(matches)!=1:raise RuntimeError('Ambiguous original window')
            start=matches[0]
            remote.append(not np.any(arrays['observed__hole_union'][start:start+len(observed_window)]))
        remote=np.asarray(remote)
        check(year+' remote entire-window rule',np.array_equal(remote,original_rows.remote.to_numpy()))
        for source in ('observed','reference'):
            base=wide[source+'_original'].to_numpy()
            for selection,mask in (('full',np.ones(len(grid),bool)),('remote',remote)):
                for name in [n for n in names if n.startswith(source+'_')]:
                    field=wide[name].to_numpy();v=field[mask];b=base[mask]
                    sign_changes=int(np.sum((field[1:]*field[:-1]<0)&mask[1:]&mask[:-1]&(np.diff(grid)==1)))
                    expected=dict(n_points=int(mask.sum()),std=float(np.std(v)),rms=float(np.sqrt(np.mean(v*v))),peak_to_peak=float(np.ptp(v)),retained_std=float(np.std(v)/np.std(b)),correlation=float(np.corrcoef(v,b)[0,1]),max_abs_change=float(np.max(abs(v-b))),rms_change=float(np.sqrt(np.mean((v-b)**2))),sign_transitions=sign_changes)
                    matched=metrics[(metrics.dataset==year)&(metrics.source==source)&(metrics.selection==selection)&(metrics.lane==name)]
                    check(f'{year} {selection} unique metric {name}',len(matched)==1)
                    row=matched.iloc[0]
                    for key,value in expected.items():close(f'{year} {selection} {name} {key}',row[key],value,1e-11)
                    persistence=expected['retained_std']>=.5 and sign_changes>=2
                    check(f'{year} {selection} {name} trigger arithmetic',row.substantial_persistence==persistence)
                    record=dict(dataset=year,source=source,lane=name,selection=selection,mean_root=float(v.mean()),**expected)
                    if selection=='remote' and name==source+'_both_mean':primary.append(record)
                    if selection=='remote' and name in (source+'_both_poly_mean',source+'_both_wide_mean'):alternate.append(record)
                    if selection=='remote' and name.startswith('observed_both_rep'):replica_summary.append(record)
    trigger_path=STUDY/'derived/persistence_trigger.json';bind(trigger_path)
    trigger=json.loads(trigger_path.read_text())
    check('routing flag matches independent primary metrics',trigger['traditional_fits_triggered']==any(r['retained_std']>=.5 and r['sign_transitions']>=2 for r in primary))
    for path in (STUDY/'run_removal.py',STUDY/'PROTOCOL.md',Path(__file__).resolve()):bind(path)
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],profile_tests=total_roots,inputs=input_summary,primary_remote=primary,alternative_remote=alternate,replica_remote=replica_summary,polynomial_fill_fit_diagnostics=poly_fit,max_reconstruction_errors=errors,source_sha256=bindings,new_fits=0,new_random_draws=0,scope='Saved-input identity, latent moments, polynomial fill reproduction, complete mass/lane grids, independent native-window remote masks and all descriptive metrics. Numerical checks do not qualify physical truths or asymptotic probabilities.')
    output=HERE/'independent_intervention_audit.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','checks','failures','profile_tests','new_fits','new_random_draws')},indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
