#!/usr/bin/env python3
"""Audit the saved paired roots and complete exact response columns."""
from pathlib import Path
import argparse,hashlib,json,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
METHODS=('profiled','fixed')
SENTINELS={'2016':{39,56,66,75,120,180},'2021':{50,78,100,150,200,250}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify(year):
    folder=HERE/'global_fast'/year
    vectors={e:np.load(folder/e/'scan_vectors.npz') for e in ('pilot10','validation1000','asimov')}
    masses=vectors['asimov']['masses_MeV']
    checks={};overlaps={};gates=[];sources={}
    for method in METHODS:
        rows=[]
        for ensemble in ('pilot10','validation1000'):
            maxerror=0.;flips=0;count=0;coordinates=0
            for j,mass in enumerate(masses):
                reference=HERE/'global'/year/ensemble/f'm{mass:03d}.npz'
                if not reference.exists():continue
                auditpath=reference.with_name(reference.stem+'_qa.json')
                audit=json.loads(auditpath.read_text())
                if not audit['passed'] or audit['checkpoint_sha256']!=sha(reference):
                    raise RuntimeError('Invalid exact reference: '+str(reference))
                sources[str(auditpath.relative_to(HERE))]=sha(auditpath)
                old=np.load(reference)[method];new=vectors[ensemble][method][:,j]
                maxerror=max(maxerror,float(np.max(abs(old-new))))
                flips+=int(np.count_nonzero((old>0)!=(new>0)))
                count+=len(old);coordinates+=1
                sources[str(reference.relative_to(HERE))]=sha(reference)
            if ensemble=='pilot10':
                checks[method+'_all_exact_pilot_references_present']=coordinates==len(masses) and count==10*len(masses)
            overlaps[ensemble+'_'+method]=dict(comparison_available=count>0,paired_roots=count,coordinates=coordinates,max_root_error=maxerror if count else None,bounded_atom_flips=flips if count else None)
            if count:
                checks[ensemble+'_'+method+'_paired_root_accuracy']=maxerror<1e-3
                checks[ensemble+'_'+method+'_bounded_atom_stable']=flips==0
    for ensemble in ('pilot10','validation1000'):
        oldfolder=HERE/'global'/year/ensemble
        if not oldfolder.exists():continue
        oldcontract=json.loads((oldfolder/'contract.json').read_text())
        newcontract=json.loads((folder/ensemble/'contract.json').read_text())
        checks[ensemble+'_exact_source_contract']=all(newcontract['source_sha256'].get(k)==v and sha(HERE.parents[1]/k)==v for k,v in oldcontract['source_sha256'].items())
        checks[ensemble+'_identical_reference_spectra']=bool(np.array_equal(np.load(oldfolder/'spectra.npz')['counts'],np.load(folder/ensemble/'spectra.npz')['counts']))
    for mass in masses:
        path=folder/'numerical_gates'/f'm{mass:03d}.json'
        g=json.loads(path.read_text());gates.append(g)
        gate_ok=g['passed']
        if g['final_backend']!='exact_cached_cholesky':
            gate_ok &= g['parent_gate_passed'] and all(x['passed'] for x in g['response_checks'].values()) and all(x['passed'] for x in g['overlap_checks'].values())
        for ensemble in ('pilot10','validation1000','asimov'):
            audit=json.loads((folder/ensemble/f'm{mass:03d}_qa.json').read_text())
            gate_ok &= audit['numerical_gate_sha256']==sha(path) and audit['numerical_backend']==g['final_backend']
        checks[f'm{mass:03d}_coordinate_gate']=bool(gate_ok)
    response={}
    sentinel_files=sorted((folder/'response_audit').glob('m*_exact_full.npz'))
    checks['declared_complete_exact_response_columns']=set(int(p.name[1:4]) for p in sentinel_files)==SENTINELS[year]
    for method in METHODS:
        exact=[];final=[];sentinels=[];rooterror=0.
        for path in sentinel_files:
            mass=int(path.name[1:4]);j=int(np.flatnonzero(masses==mass)[0])
            ref=np.load(path)[method];new=vectors['asimov'][method][:,j]
            rooterror=max(rooterror,float(np.max(abs(ref-new))))
            exact.append(ref[1:]-ref[0]);final.append(new[1:]-new[0]);sentinels.append(mass)
            sources[str(path.relative_to(HERE))]=sha(path)
        exact=np.column_stack(exact);final=np.column_stack(final)
        C0=exact.T@exact;C1=final.T@final
        s0=np.sqrt(np.diag(C0));s1=np.sqrt(np.diag(C1))
        K0=C0/np.outer(s0,s0);K1=C1/np.outer(s1,s1)
        metrics=dict(sentinel_masses_MeV=sentinels,max_root_error=rooterror,
            max_absolute_response_error=float(np.max(abs(final-exact))),
            max_relative_l2_response_error=float(np.max(np.linalg.norm(final-exact,axis=0)/s0)),
            max_relative_width_error=float(np.max(abs(s1/s0-1))),
            max_absolute_correlation_error=float(np.max(abs(K1-K0))))
        response[method]=metrics
        checks[method+'_full_response_accuracy']=metrics['max_root_error']<1e-3 and metrics['max_absolute_response_error']<1e-4 and metrics['max_relative_l2_response_error']<1e-3
        checks[method+'_full_response_width_and_correlation']=metrics['max_relative_width_error']<1e-3 and metrics['max_absolute_correlation_error']<1e-3
    accepted=[g for g in gates if g['final_backend']!='exact_cached_cholesky']
    record=dict(passed=all(checks.values()),checks=checks,overlaps=overlaps,full_response=response,
        exact_fallback_masses_MeV=[g['mass_MeV'] for g in gates if g['final_backend']=='exact_cached_cholesky'],
        max_accepted_probe_response_error=max((r['max_response_error'] for g in accepted for r in g['response_checks'].values()),default=0.),
        max_accepted_probe_response_error_over_width=max((r['max_response_error_over_width'] for g in accepted for r in g['response_checks'].values()),default=0.),
        source_sha256=sources,verifier_sha256=sha(__file__))
    (folder/'acceleration_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    if not record['passed']:raise RuntimeError('Acceleration product audit failed: '+', '.join(k for k,v in checks.items() if not v))
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['2016','2021'])
    print(json.dumps(verify(ap.parse_args().dataset),indent=2))
