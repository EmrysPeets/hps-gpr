"""Bounded shared-coupling audit of the copied frozen observed-search engine.

All writes stay in this script's QA directory. No mass-scan cohort is launched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=HERE.parent
MASSES=(67,68,91,120,175,176)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    args=parser.parse_args();source=args.source.resolve()
    sys.path.insert(0,str(source/'scripts'))
    import run_global as G
    import extraction as E
    import limit_solver
    assert E.B==source
    stored=pd.read_csv(source/'results/observed_scan.csv',float_precision='round_trip')
    report=dict(passed=False,numerical_threads=1,masses_MeV=list(MASSES),
        shared_parameter='Exactly one inherited psi=epsilon2_ee/1e-8 at each common generated mass; campaign yields are psi*K_y(m).',
        background='Independent campaign nuisance vectors; block-diagonal count-space GP covariance factors.',
        statistic='One joint profile likelihood ratio, not a sum of individual q0 or local significances.',
        source_directory='.' if source==HERE.parent else str(source),source_hashes={name:sha(source/name) for name in [
            'scripts/run_global.py','scripts/extraction.py','inputs/v6p1/scripts/limit_solver.py',
            'provenance/global_protocol.json','provenance/input_manifest.sha256']},checks=0)
    def check(ok,message):
        report['checks']+=1
        if not bool(ok):raise AssertionError(message)

    def fixed_profile(model,n,psi):
        if psi>=0:return model.fit(n,fixed=float(psi))
        # The public fixed-limit API permits nonnegative strengths only. For the
        # signed diagnostic optimum, shift the fixed base by psi*S and profile
        # nuisances with its remaining signal parameter fixed to zero. This has
        # precisely the same Poisson means and unchanged Gaussian penalty.
        base=model.b+psi*model.S
        check(np.all(base>0),'Signed diagnostic base must remain positive')
        shifted=E.C.OneSignalProfile(base,model.L,model.S,score_tolerance=2e-9)
        return shifted.fit(n,fixed=0.)

    rows=[];mass_rows=[]
    for mass in MASSES:
        contexts=G.at_mass(mass)
        check(list(contexts)==E.years(mass),'Campaign support differs from protocol')
        parts=[contexts[y].part() for y in E.years(mass)]
        mod,n=E.model(parts,tolerance=2e-9)
        check(mod.Jfree.shape[1]==1+sum(p['L'].shape[1] for p in parts),'Joint model has more than one signal amplitude')
        check(np.count_nonzero(mod.penfree==0)==1,'More than one unpenalized amplitude')
        check(np.allclose(mod.Jfree[:,0]/mod.scale,mod.S,rtol=2e-15,atol=0),'Internal scaling changes physical parameter')
        offset=0;rank_offset=0;individual=[];details=[]
        for part in parts:
            year=part['year'];ctx=part['context'];nb=len(part['n']);nr=part['L'].shape[1]
            check(ctx.mass==mass,'Campaign evaluates a different generated mass')
            check(np.array_equal(ctx.fit,ctx.guard),'Fit/GP exclusion mismatch')
            if year=='2016':
                check(abs((ctx.requested_low-ctx.center)/ctx.width+2.5)<1e-12 and abs((ctx.requested_high-ctx.center)/ctx.width-2.5)<1e-12,'Wrong selected 2016 window')
            K=E.conversion(year,mass)
            d=E.C.DATA[year];m=mass/1000.;sig=E.C.sigma(year,mass)
            overlap=np.maximum(0.,np.minimum(d['native_edges'][1:],m+1.64*sig)-np.maximum(d['native_edges'][:-1],m-1.64*sig))
            density=float(np.sum(d['native_counts']*overlap/np.diff(d['native_edges']))/(3.28*sig))
            independent_K=1e-8*3*np.pi*m*float(d['frad_effective'])*density/(2/137.)
            check(abs(K/independent_K-1)<2e-15,'Conversion mismatch')
            check(np.allclose(part['S'],ctx.probability[ctx.fit]*K,rtol=2e-15,atol=0),'Campaign signal is not fixed K times full-bin probability')
            block=mod.L[offset:offset+nb,rank_offset:rank_offset+nr]
            check(np.array_equal(block,part['L']),'Diagonal nuisance block changed')
            exterior=np.r_[0:rank_offset,rank_offset+nr:mod.rank]
            check(not np.any(mod.L[offset:offset+nb,exterior]),'Cross-campaign nuisance coupling found')
            check(np.array_equal(mod.S[offset:offset+nb],part['S']),'Joint signal is not concatenated campaign S')
            imodel=E.C.OneSignalProfile(part['b'],part['L'],part['S'],score_tolerance=2e-9)
            individual.append((imodel,part['n']))
            details.append(dict(year=year,generated_mass_MeV=mass,
                K_events_per_psi=K,core_center_MeV=float(ctx.center),signal_fit_fraction=float(ctx.probability[ctx.fit].sum()),
                fit_bins=nb,nuisance_rank=nr))
            offset+=nb;rank_offset+=nr
        free=mod.fit(n);null=mod.fit(n,fixed=0.)
        rawq=2*(null['nll']-free['nll']);check(rawq>=-2e-6,'Joint likelihood nesting failure')
        root=float(np.sign(free['A'])*np.sqrt(max(0.,rawq)));q0=max(0.,root)**2
        ref=stored[(stored.scope=='combined')&(stored.mass_MeV==mass)]
        check(len(ref)==1,'No saved combined reference')
        ref=ref.iloc[0]
        check(abs(free['A']-ref.psi_hat)<2e-7*max(1.,abs(ref.psi_hat)),'Saved common optimum disagrees')
        check(abs(root-ref.signed_root)<2e-6,'Saved joint root disagrees')
        scales=(0.,free['sigma'],2*free['sigma'])
        points=[('null',0.),('joint_signed_optimum',float(free['A'])),('one_joint_error',float(scales[1])),('two_joint_errors',float(scales[2]))]
        for label,psi in points:
            joint=free if label=='joint_signed_optimum' else fixed_profile(mod,n,psi)
            separate=[fixed_profile(imodel,nn,psi) for imodel,nn in individual]
            summed=float(sum(item['nll'] for item in separate));difference=float(joint['nll']-summed)
            expected=np.concatenate([item['lam'] for item in separate])
            max_mean_diff=float(np.max(abs(joint['lam']-expected)))
            max_mean_scaled=float(np.max(abs(joint['lam']-expected)/np.sqrt(joint['lam'])))
            check(abs(difference)<2e-6,'Joint NLL not equal to sum of separately profiled campaign NLLs at common psi')
            check(max_mean_scaled<2e-6,'Independent nuisance-profiled expectations differ')
            check(np.min(joint['lam'])>0 and max(item['score'] for item in [joint]+separate)<3e-5,'Profile positivity or score failure')
            # The summed amplitude derivative at the common optimum vanishes,
            # although the campaigns generally have different free optima.
            derivative=sum(float(imodel.S@(1-nn/f['lam'])) for (imodel,nn),f in zip(individual,separate))
            if label=='joint_signed_optimum':check(abs(derivative*mod.scale)<3e-5,'Common optimum is not stationary for sum of campaign profiles')
            rows.append(dict(mass_MeV=mass,campaigns='+'.join(E.years(mass)),point=label,psi=psi,
                epsilon2_coordinate=psi*1e-8,joint_profile_nll=float(joint['nll']),sum_independent_campaign_profile_nll=summed,
                nll_difference=difference,maximum_bin_expectation_difference=max_mean_diff,
                maximum_expectation_difference_in_sqrt_count_units=max_mean_scaled,
                summed_psi_derivative=derivative,maximum_profile_score=max(item['score'] for item in [joint]+separate)))
        individual_q0=[];individual_hat=[]
        for imodel,nn in individual:
            ff=imodel.fit(nn);nn0=imodel.fit(nn,fixed=0.)
            rr=np.sign(ff['A'])*np.sqrt(max(0.,2*(nn0['nll']-ff['nll'])))
            individual_q0.append(max(0.,float(rr))**2);individual_hat.append(float(ff['A']))
        mass_rows.append(dict(mass_MeV=mass,campaigns='+'.join(E.years(mass)),joint_psi_hat=float(free['A']),
            joint_sigma_psi=float(free['sigma']),joint_signed_root=root,joint_q0=q0,
            sum_individual_q0=float(sum(individual_q0)),individual_psi_hat=individual_hat,campaign_details=details))
        print(f'Common-coupling profiles verified at {mass} MeV',flush=True)
    check(any(abs(r['joint_q0']-r['sum_individual_q0'])>.1 for r in mass_rows if '+' in r['campaigns']),
          'Diagnostic did not distinguish joint inference from independently fitted signals')
    table=HERE/'common_coupling_profiles.csv';pd.DataFrame(rows).to_csv(table,index=False,float_format='%.17g')
    report.update(passed=True,profile_points=len(rows),mass_summaries=mass_rows,
        maximum_absolute_profile_NLL_difference=max(abs(r['nll_difference']) for r in rows),
        maximum_expectation_difference_in_sqrt_count_units=max(r['maximum_expectation_difference_in_sqrt_count_units'] for r in rows),
        exactly_one_shared_signal_parameter=True,common_generated_mass=True,year_specific_fixed_conversions=True,
        independent_nuisance_blocks=True,combined_statistic_not_sum_of_local_statistics=True,
        numerical_results_sha256=sha(table),audit_script_sha256=sha(__file__),
        code_pointers={'joint_mass_call':'scripts/run_global.py:22-33','campaign_signal':'scripts/extraction.py:96-100',
            'joint_model':'scripts/extraction.py:120-123','single_amplitude':'inputs/v6p1/scripts/limit_solver.py:67-80'},
        normalization_note='The fitted parameter is the inherited ee-yield coupling coordinate. Above the dimuon threshold, 211.316749 MeV, the historical coupling display multiplies 1e-8*psi by 1+sqrt(1-4*r_mu)*(1+2*r_mu), with r_mu=(m_mu/m)^2. That is one mass-dependent common conversion, not an additional campaign amplitude. Only 2021 remains above175 MeV. This audit changes no normalization and does not establish that inherited display as a newly validated physical coupling exclusion.',
        claim_boundary='This verifies shared-coupling implementation and numerical factorization only; it does not calibrate the search or validate the inherited physical normalization.')
    (HERE/'common_coupling_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','checks','profile_points','maximum_absolute_profile_NLL_difference']},indent=2))


if __name__=='__main__':main()
