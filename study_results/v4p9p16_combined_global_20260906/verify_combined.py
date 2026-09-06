#!/usr/bin/env python3
"""Read-only acceptance of coherent combined spectra and likelihood vectors."""
from pathlib import Path
import csv,hashlib,json,os,subprocess
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
YEARS=('2015','2016','2021')
SIZES={'2015':484,'2016':720,'2021':422}
STARTS={'2015':0,'2016':484,'2021':1204}
SENTINELS={39,49,50,90,91,180}
METHODS=('profiled','fixed')
ENSEMBLES=('pilot10','validation1000','asimov')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def members(m):
    return tuple(y for y,lo,hi in [('2015',19,90),('2016',39,180),('2021',50,250)] if lo<=m<=hi)
def indices(keys):
    return np.concatenate([np.arange(STARTS[y],STARTS[y]+SIZES[y]) for y in keys])

def verify():
    f=HERE/'global';summary=json.loads((f/'summary.json').read_text())
    contract=json.loads((f/'contract.json').read_text())
    checks={}
    checks['ensemble_complete']=summary['passed'] and summary['complete']
    checks['contract_hash']=sha(f/'contract.json')==summary['contract_sha256']
    checks['all_source_hashes']=all(sha(ROOT/p)==h for p,h in contract['source_sha256'].items())
    checks['final_vector_hash']=sha(f/'scan_vectors.npz')==summary['vectors_sha256']
    checks['observed_hash']=sha(f/'observed.csv')==summary['observed_sha256']
    oldref=json.loads((HERE/'provenance/observed_reference.json').read_text())
    checks['v12_comparator_frozen_hash']=oldref['verified_against_frozen_manifest'] and sha(ROOT/oldref['path'])==oldref['sha256'] and sha(ROOT/oldref['parent_manifest_path'])==oldref['parent_manifest_sha256']
    masses=np.arange(19,251)
    vectors=np.load(f/'scan_vectors.npz')
    observed=pd.read_csv(f/'observed.csv').set_index('mass_MeV')
    checks['complete_grid']=np.array_equal(vectors['masses_MeV'],masses) and np.array_equal(observed.index,masses)
    checks['declared_sentinels']=set(contract['sentinels_MeV'])==SENTINELS
    truth=None;spectra={}
    for e,n in [('pilot10',10),('validation1000',1000),('asimov',1627)]:
        p=f/'spectra'/f'{e}.npz';x=np.load(p)
        checks[e+'_hash']=sha(p)==summary['spectra_sha256'][e]
        spectra[e]=x['counts']
        checks[e+'_size']=spectra[e].shape==(n,1626)
        checks[e+'_truth']=truth is None or np.array_equal(truth,x['truth'])
        truth=x['truth']
        for y in YEARS:
            src=ROOT/contract['upstream'][y][e]['source']
            base=np.load(src/'spectra.npz')
            ii=indices((y,))
            checks[y+'_'+e+'_source_truth']=np.array_equal(truth[ii],base['truth'])
            if e!='asimov':
                checks[y+'_'+e+'_same_id_counts']=np.array_equal(spectra[e][:,ii],base['counts'])
    want=np.broadcast_to(truth,(1627,1626)).copy()
    j=np.arange(1626);want[j+1,j]+=np.sqrt(truth)
    checks['full_asimov_definition']=np.array_equal(want,spectra['asimov'])
    checks['pilot_validation_disjoint_streams']=not np.array_equal(spectra['pilot10'],spectra['validation1000'][:10])
    checks['positive_truth']=bool(np.all(truth>0))
    checks['integer_poisson_counts']=all(np.array_equal(spectra[e],np.rint(spectra[e])) for e in ('pilot10','validation1000'))
    checks['no_saved_failures']=not list((f/'points').glob('*FAILURE*'))
    numerical=[];sentinel_final={m:[] for m in METHODS};sentinel_exact={m:[] for m in METHODS}
    for mass in masses:
        mass=int(mass);p=f/'points'/f'm{mass:03d}.npz';q=p.with_name(f'm{mass:03d}_qa.json')
        a=json.loads(q.read_text());r=np.load(p);keys=members(mass);ii=indices(keys)
        checks[f'm{mass}_identity']=a['passed'] and a['contract_sha256']==summary['contract_sha256'] and a['checkpoint_sha256']==sha(p)
        checks[f'm{mass}_membership']=tuple(a['active_datasets'])==keys and observed.loc[mass,'dataset_set']=='+'.join(keys)
        checks[f'm{mass}_reference_hashes']=all(sha(ROOT/name)==h for name,h in a['source_reference_sha256'].items())
        inactive=np.setdiff1d(np.arange(1626),ii)
        joined=True;zero=True;finite=True
        for e,n in [('pilot10',10),('validation1000',1000),('asimov',1627)]:
            for method in METHODS:
                col=r[e+'_'+method]
                joined &= col.shape==(n,) and np.array_equal(col,vectors[e+'_'+method][:,mass-19])
                finite &= np.all(np.isfinite(col))
                if e=='asimov':
                    zero &= np.all(col[inactive+1]==col[0])
        checks[f'm{mass}_assembled_columns']=bool(joined and finite)
        checks[f'm{mass}_inactive_response_zero']=bool(zero)
        if len(keys)==1:
            base=ROOT/contract['upstream'][keys[0]]['pilot10']['source']
            matched=True
            for e in ENSEMBLES:
                base=ROOT/contract['upstream'][keys[0]][e]['source']
                source=np.load(base/f'm{mass:03d}.npz')
                for method in METHODS:
                    target=r[e+'_'+method]
                    if e=='asimov':target=target[np.r_[0,ii+1]]
                    matched &= np.array_equal(target,source[method])
            checks[f'm{mass}_reused_values']=bool(matched)
            continue
        refpath=f/'references'/f'm{mass:03d}.npz'
        ref=np.load(refpath);refqa=json.loads(refpath.with_suffix('.json').read_text())
        checks[f'm{mass}_exact_reference']=refqa['passed'] and refqa['reference_sha256']==sha(refpath)
        checks[f'm{mass}_exact_reference_numerics']=all(x['score']<2e-7 for name in ('pilot_checks','response_checks') for x in refqa[name]) and all(x['passed'] for name in ('pilot_scalar_checks','response_scalar_checks') for x in refqa[name])
        checks[f'm{mass}_memoization']=a['memoization_parent_baseline_exact']
        checks[f'm{mass}_scalar_and_scores']=all(x['score']<2e-7 for phase in a['phase_checks'].values() for x in phase['checks']) and all(x['passed'] for phase in a['phase_checks'].values() for x in phase['scalar_checks'])
        checks[f'm{mass}_observed_cls']=all(abs(a['observed_checks'][m]['cls']-.1)<2e-6 and a['observed_checks'][m]['max_score']<2e-7 and a['observed_checks'][m]['min_lambda']>0 for m in METHODS)
        checks[f'm{mass}_active_indices']=np.array_equal(ref['active_full_bin_indices'],ii)
        probe=ref['probe_indices']
        for method in METHODS:
            pilot=r['pilot10_'+method];expected=ref['pilot_'+method]
            err=float(np.max(abs(pilot-expected)));flips=int(np.count_nonzero((pilot>0)!=(expected>0)))
            checks[f'm{mass}_{method}_pilot']=err<1e-3 and flips==0
            allroots=r['asimov_'+method];local=allroots[np.r_[0,ii+1]]
            expected=ref['response_'+method];width=np.linalg.norm(local[1:]-local[0])
            delta=(local[probe]-local[0])-(expected-expected[0])
            rooterror=float(np.max(abs(local[probe]-expected)))
            responseerror=float(np.max(abs(delta)))
            checks[f'm{mass}_{method}_response']=bool(width>0 and rooterror<1e-3 and abs(local[0]-expected[0])/width<1e-3 and responseerror<1e-4 and responseerror/width<1e-4)
            rec=dict(mass_MeV=mass,method=method,backend=a['numerical_backend'],
                     pilot_max_root_error=err,pilot_atom_flips=flips,
                     response_max_error=responseerror,full_column=mass in SENTINELS)
            if mass in SENTINELS:
                checks[f'm{mass}_complete_response_reference']=np.array_equal(probe,np.arange(len(ii)+1))
                d=local[1:]-local[0];de=expected[1:]-expected[0]
                rec.update(relative_l2_error=float(np.linalg.norm(d-de)/np.linalg.norm(de)),
                    relative_width_error=float(abs(np.linalg.norm(d)/np.linalg.norm(de)-1)))
                checks[f'm{mass}_{method}_full_response']=rec['relative_l2_error']<1e-3 and rec['relative_width_error']<1e-3
                emb=np.zeros(1626);emb[ii]=de
                sentinel_exact[method].append(emb);sentinel_final[method].append(allroots[1:]-allroots[0])
            numerical.append(rec)
    correlations={}
    for method in METHODS:
        exact=np.array(sentinel_exact[method]).T;final=np.array(sentinel_final[method]).T
        ce=exact.T@exact;cf=final.T@final
        ke=ce/np.sqrt(np.outer(np.diag(ce),np.diag(ce)))
        kf=cf/np.sqrt(np.outer(np.diag(cf),np.diag(cf)))
        error=float(np.max(abs(ke-kf)))
        checks[method+'_sentinel_correlation']=error<1e-3
        correlations[method]=dict(masses=sorted(SENTINELS),max_absolute_error=error)
    for method in METHODS:
        checks[method+'_positive_limits']=bool(np.all(observed[method+'_eps2_ee_raw']>0) and np.all(np.isfinite(observed[method+'_eps2_ee_raw'])))
        checks[method+'_single_dimuon_correction']=bool(np.allclose(observed[method+'_eps2_display'],observed[method+'_eps2_ee_raw']*observed.dimuon_factor,rtol=1e-14,atol=0))
    for y,prefix in [('2015','v4p9p14_interpretation_global_20260906'),('2016_2021','v4p9p15_global_2016_2021_20260906')]:
        manifest=HERE.parent/prefix/'MANIFEST.csv'
        checks['frozen_'+y+'_manifest']=all((ROOT/r['path']).stat().st_size==int(r['bytes']) and sha(ROOT/r['path'])==r['sha256'] for r in csv.DictReader(manifest.open()))
    investigations=summary['v12_dense_investigation_masses']
    checks['v12_excursions_addressed']=not investigations
    if investigations:
        path=HERE/'review/v12_dense_investigations.json'
        if path.exists():
            review=json.loads(path.read_text())
            checks['v12_excursions_addressed']=review['passed'] and set(review['masses_MeV'])==set(investigations) and review['observed_sha256']==sha(f/'observed.csv')
    workspace=json.loads((HERE/'provenance/workspace_state.json').read_text())
    checks['shared_head_preserved']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==workspace['git_head']
    checks['shared_index_preserved']=hashlib.sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT)).hexdigest()==workspace['index_diff_sha256']
    report=dict(passed=all(checks.values()),check_count=len(checks),checks=checks,
        numerical=numerical,sentinel_correlations=correlations,
        contract_sha256=sha(f/'contract.json'),verifier_sha256=sha(__file__),
        v12_investigation_masses=investigations)
    (HERE/'qa/numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),
        failures=[k for k,v in checks.items() if not v],sentinel_correlations=correlations),indent=2))
    if not report['passed']:raise RuntimeError('Combined numerical verification failed')
    return report

if __name__=='__main__':verify()
