#!/usr/bin/env python3
"""Independent numerical identities, source binding, and rendered-report checks."""
from pathlib import Path
import csv, hashlib, json, os, re, subprocess
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from pypdf import PdfReader
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks, numerical, statistical = {}, {}, {}
    for year, first, last, bins in [('2016',39,180,720),('2021',50,250,422)]:
        folder = HERE/'global_fast'/year
        out = folder/'analysis'
        summary = json.loads((out/'summary.json').read_text())
        checks[year+'_analysis_input_hashes'] = all(sha(HERE/name)==digest for name,digest in summary['input_sha256'].items())
        acceleration = json.loads((folder/'acceleration_validation.json').read_text())
        checks[year+'_acceleration_verified'] = acceleration['passed'] and summary['acceleration_audit_passed'] and acceleration['verifier_sha256']==sha(HERE/'verify_acceleration.py')
        d = pd.read_csv(out/'pvalue_curves.csv')
        cov = np.load(out/'covariance.npz')
        maxima = np.load(out/'maxima.npz')
        expected_masses = np.arange(first,last+1)
        checks[year+'_grid_and_rows'] = bool(np.array_equal(summary['masses_MeV'],expected_masses) and d.groupby('method').size().to_dict()=={'fixed':len(expected_masses),'profiled':len(expected_masses)})
        spectra, vectors, contracts = {}, {}, []
        max_r_error, max_q_error, max_score, max_observed_error = 0., 0., 0., 0.
        all_scalar, all_audits, all_hashes = True, True, True
        for ensemble, count in [('pilot10',10),('validation1000',1000),('asimov',bins+1)]:
            base = folder/ensemble
            s = json.loads((base/'summary.json').read_text())
            c = json.loads((base/'contract.json').read_text())
            contracts.append(c)
            checks[year+'_'+ensemble+'_complete'] = bool(s['complete'] and s['passed'] and s['n_spectra']==count and s['hypotheses']==len(expected_masses) and s['full_bins']==bins)
            for filename, key in [('contract.json','contract_sha256'),('scan_vectors.npz','vectors_sha256'),('spectra.npz','spectra_sha256')]:
                all_hashes &= sha(base/filename)==s[key]
            spectra[ensemble] = np.load(base/'spectra.npz')
            vectors[ensemble] = np.load(base/'scan_vectors.npz')
            for j,mass in enumerate(expected_masses):
                audit = json.loads((base/f'm{mass:03d}_qa.json').read_text())
                checkpoint = base/f'm{mass:03d}.npz'
                all_audits &= audit['passed'] and audit['n_spectra']==count
                all_hashes &= sha(checkpoint)==audit['checkpoint_sha256']
                max_observed_error = max(max_observed_error,*audit['observed_r_error'].values())
                max_score = max(max_score,*[x['score'] for x in audit['checks']])
                for x in audit['scalar_checks']:
                    all_scalar &= x['passed']
                    max_r_error = max(max_r_error,x['r_error'])
                    max_q_error = max(max_q_error,*[q['q_error'] for q in x['q_checks']])
                with np.load(checkpoint) as point:
                    for method in ('profiled','fixed'):
                        all_audits &= np.array_equal(point[method],vectors[ensemble][method][:,j])
        checks[year+'_all_numerical_audits'] = bool(all_audits and all_scalar and max_score<2e-7 and max_r_error<=2e-5 and max_q_error<=1e-4 and max_observed_error<=2e-5)
        checks[year+'_all_ensemble_hashes'] = bool(all_hashes)
        checks[year+'_matching_truth_and_sources'] = all(all(c[k]==contracts[0][k] for c in contracts[1:]) for k in ('dataset','masses_MeV','truth_array_sha256','source_sha256','parent_contract_sha256'))
        checks[year+'_source_hashes_current'] = all(sha(ROOT/name)==h for name,h in contracts[0]['source_sha256'].items())
        b = spectra['asimov']['truth']
        wanted = np.broadcast_to(b,(bins+1,bins)).copy()
        ii = np.arange(bins)
        wanted[ii+1,ii] += np.sqrt(b)
        checks[year+'_asimov_spectra_definition'] = bool(np.array_equal(wanted,spectra['asimov']['counts']))
        checks[year+'_pilot_and_validation_distinct'] = bool(not np.array_equal(spectra['pilot10']['counts'],spectra['validation1000']['counts'][:10]))
        checks[year+'_same_truth_all_ensembles'] = all(np.array_equal(s['truth'],b) for s in spectra.values())
        checks[year+'_no_saved_failure'] = not list(folder.rglob('*FAILURE*'))
        for method in ('profiled','fixed'):
            key = year+'_'+method
            p = d[d.method==method].sort_values('mass_MeV')
            r, a, sd = [p[x].to_numpy() for x in ('observed_r','asimov_r','response_sd')]
            z = (r-a)/sd
            score = np.where(r>0,z,-np.inf)
            response = vectors['asimov'][method][1:]-vectors['asimov'][method][0]
            checks[key+'_covariance_identity'] = bool(np.allclose(response.T@response,cov[method+'_C'],rtol=1e-13,atol=1e-13))
            checks[key+'_valid_correlation'] = bool(np.allclose(np.diag(cov[method+'_K']),1) and np.linalg.eigvalsh(cov[method+'_K']).min()>-1e-9)
            checks[key+'_local_rule'] = bool(np.allclose(np.where(r>0,norm.sf(z),1),p.p_local_common_truth,rtol=1e-12,atol=1e-15))
            checks[key+'_bounded_atom'] = bool(np.all(p.loc[r<=0,['p_local_common_truth','p_global_gp','p_global_direct']].to_numpy()==1))
            for name, arr, column in [('gp',maxima[method+'_gp'],'p_global_gp'),('direct',maxima[method+'_direct'],'p_global_direct')]:
                expected = np.array([np.mean(arr>=x) for x in score])
                checks[key+'_'+name+'_tail_counts'] = bool(np.allclose(expected,p[column],rtol=0,atol=1e-15))
            direct_r = vectors['validation1000'][method]
            direct_max = np.where(direct_r>0,(direct_r-a)/sd,-np.inf).max(axis=1)
            checks[key+'_coherent_direct_maximum'] = bool(np.allclose(direct_max,maxima[method+'_direct'],rtol=1e-12,atol=1e-13))
            checks[key+'_raw_direct_maximum'] = bool(np.array_equal(np.maximum(direct_r,0).max(axis=1),maxima[method+'_direct_raw']))
            raw = np.array([np.mean(maxima[method+'_gp_raw']>=max(0,x)) for x in r])
            checks[key+'_raw_curve_counts'] = bool(np.allclose(raw,p.p_global_raw_ordering,rtol=0,atol=1e-15))
            info = summary['methods'][method]
            peak = int(np.argmax(score))
            checks[key+'_peak_selection'] = int(p.iloc[peak].mass_MeV)==info['peak_mass_MeV']
            bounds_ok = True
            for tail in (info['global_gp'],info['global_direct']):
                k,n = tail['k'],tail['n']
                upper = 1. if k==n else float(beta.ppf(.95,k+1,n-k))
                bounds_ok &= abs(upper-tail['upper95_one_sided'])<1e-14
            checks[key+'_one_sided_upper_bounds'] = bounds_ok
            statistical[key] = dict(marginal_normality_flags=info['marginal_normality_holm_flags'],
                minimum_p_maximum_KS_p=info['minimum_local_p_maximum_distribution_KS']['pvalue'],
                raw_maximum_KS_p=info['maximum_raw_root_maximum_distribution_KS']['pvalue'],
                gp_inside_direct_peak_interval=info['gp_global_inside_direct_interval95'],
                direct_tail_exceedances=info['global_direct']['k'],
                rare_tail_unresolved_by_direct=info['global_direct']['k']==0,
                scope='conditional common stress spectrum, declared finite mass grid; not a physics qualification')
        numerical[year] = dict(max_scalar_r_error=max_r_error,max_scalar_q_error=max_q_error,
                               max_score=max_score,max_observed_r_error=max_observed_error)

    previous = HERE.parent/'v4p9p14_interpretation_global_20260906'
    parent_rows = list(csv.DictReader((previous/'MANIFEST.csv').open()))
    checks['frozen_2015_manifest_preserved'] = all((ROOT/row['path']).stat().st_size==int(row['bytes']) and sha(ROOT/row['path'])==row['sha256'] for row in parent_rows)
    checks['exact_reference_runner_byte_identical'] = sha(HERE/'run_global.py')==sha(previous/'run_global.py')
    build = json.loads((HERE/'provenance/report_build.json').read_text())
    pdf = Path(build['pdf'])
    checks['report_input_hashes'] = all(sha(ROOT/p)==h for p,h in build['inputs'].items())
    checks['report_pdf_hash'] = sha(pdf)==build['pdf_sha256']
    texts = [p.extract_text() for p in PdfReader(pdf).pages]
    full = re.sub(r'\s+', ' ', '\n'.join(texts))
    checks['no_empty_or_orphan_pages'] = all(len(t)>500 for t in texts)
    checks['no_unresolved_references'] = not any(x in full for x in ('??','TODO','PLACEHOLDER'))
    log = (HERE/'note/build.log').read_text()+(pdf.parent/'reader_report.log').read_text()
    checks['no_tex_overfull_or_undefined'] = not re.search(r'Overfull|undefined|LaTeX Warning',log,re.I)
    checks['scientific_qualifications_visible'] = all(x in full for x in ('different tests','2016 numerical exception','not a goodness-of-fit','do not establish','10%'))
    bounds = []
    for i,p in enumerate(fitz.open(pdf),1):
        for block in p.get_text('blocks'):
            if block[4].strip()==str(i) and block[1]>740:
                continue
            if block[0]<35 or block[1]<30 or block[2]>577 or block[3]>765:
                bounds.append(dict(page=i,bbox=block[:4]))
    checks['page_content_in_bounds'] = not bounds
    pages = sorted((HERE/'qa/pages').glob('page-*.png'))
    checks['all_final_pages_rendered'] = len(pages)==len(texts)
    parent = json.loads((HERE/'provenance/parent_reference.json').read_text())
    checks['shared_git_head_and_index_preserved'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==parent['git_head'] and not subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT)
    record = dict(passed=all(checks.values()),check_count=len(checks),checks=checks,numerical=numerical,
                  statistical_diagnostics=statistical,page_count=len(texts),pdf_sha256=sha(pdf),bounds_violations=bounds,
                  rendered_pages_sha256={str(p.relative_to(ROOT)):sha(p) for p in pages})
    (HERE/'qa/product_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not record['passed']:
        raise RuntimeError('Product validation failed')

if __name__ == '__main__':
    main()
