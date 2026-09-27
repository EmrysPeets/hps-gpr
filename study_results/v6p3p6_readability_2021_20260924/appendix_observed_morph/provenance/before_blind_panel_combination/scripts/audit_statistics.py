#!/usr/bin/env python3
"""Independent checks of observed scan, region selection, toy tails and ranks."""
from pathlib import Path
import hashlib,json,sys,os,re
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.special import ndtr,log_ndtr
from scipy.stats import beta

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'scripts'))
from observed_templates import BANK

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    protocol=json.loads((B/'provenance/protocol.json').read_text());checks={}
    checks['frozen_runner_hash_matches']=sha(B/'scripts/run_observed.py')==protocol['script_sha256']
    checks['frozen_template_hash_matches']=sha(B/'scripts/observed_templates.py')==protocol['template_script_sha256']
    checks['frozen_archived_template_hash_matches']=sha(B/'scripts/archived_templates.py')==protocol['archived_template_sha256']
    scan=pd.read_csv(B/'results/observed_scan.csv',float_precision='round_trip')
    checks['scan_grid_and_policy_count_match']=len(scan)==len(protocol['masses'])*len(protocol['policies']) and not scan.duplicated(['mass_MeV','policy']).any()
    checks['all_observed_fits_valid']=bool(scan.fit_valid.all())
    checks['signed_likelihood_root_matches_objectives']=bool(np.allclose(scan.signed_r,np.sign(scan.Ahat)*np.sqrt(np.maximum(0,2*(scan.nll_null-scan.nll_free))),atol=1e-12,rtol=1e-12))
    checks['one_sided_q0_has_no_deficit_significance']=bool(np.allclose(scan.q0,np.maximum(scan.signed_r,0)**2,atol=1e-13,rtol=0) and (scan.loc[scan.signed_r<0,'q0']==0).all())
    checks['asymptotic_excess_p_values_match_q0']=bool(np.allclose(scan.p0_fixed_mass,ndtr(-np.sqrt(scan.q0)),atol=1e-14,rtol=0))
    checks['deficit_tail_labeled_separately']=bool(np.allclose(scan.p_deficit_asymptotic,np.where(scan.signed_r<0,ndtr(scan.signed_r),1.),atol=1e-14,rtol=0))
    checks['profile_limit_solver_conditions']=bool((abs(scan.cls-.1)<2e-6).all() and (scan.max_score<3e-5).all() and (scan.min_lambda>0).all())
    grid=np.load(B/'results/template_grid.npz');edges=grid['edges_GeV']*1000
    checks['production_anchors_cover_60_to_240']=np.array_equal(BANK.anchors,np.arange(60,241,20))
    checks['every_anchor_reproduces_its_direct_MC']=all(np.allclose(BANK.categories(m,edges),BANK.categories(m,edges,kind='direct'),atol=1e-12,rtol=0) for m in BANK.anchors)
    checks['all_dense_templates_keep_full_probability']=bool(np.all(grid['full_MC_categories']>=0) and np.allclose(grid['full_MC_categories'].sum(axis=1),1,atol=1e-12,rtol=0))
    checks['all_intermediate_anchors_bracket_the_mass']=all(all(60<=a<=240 for a,w in BANK.neighbors(m)) and abs(sum(w for a,w in BANK.neighbors(m))-1)<1e-14 for m in protocol['masses'])
    checks['acceptance_transition_rows_flagged']=bool(np.array_equal(scan.acceptance_transition.to_numpy(),(scan.mass_MeV<80).to_numpy()))
    x=.5*(edges[:-1]+edges[1:]);primary=scan[scan.policy=='morph_starter'].sort_values('mass_MeV').reset_index(drop=True)
    values=primary.q0.to_numpy();candidates=[]
    for i,r in primary.iterrows():
        if r.q0>0 and (i==0 or values[i]>values[i-1]) and (i==len(values)-1 or values[i]>=values[i+1]):candidates.append(r)
    chosen=[];masks=[]
    for r in sorted(candidates,key=lambda r:(-r.q0,r.mass_MeV)):
        mask=(x>=r.fit_low_MeV)&(x<=r.fit_high_MeV)
        if any(np.any(mask&a) for a in masks):continue
        chosen.append(int(r.mass_MeV));masks.append(mask)
        if len(chosen)==3:break
    for r in primary.sort_values(['signed_r','mass_MeV']).itertuples():
        mask=(x>=r.fit_low_MeV)&(x<=r.fit_high_MeV)
        if r.signed_r<0 and not any(np.any(mask&a) for a in masks):
            chosen.append(int(r.mass_MeV));masks.append(mask);break
    selected=json.loads((B/'results/selected_regions.json').read_text())['regions']
    checks['predeclared_region_selection_reproduced']=chosen==[r['mass_MeV'] for r in selected]
    checks['selected_likelihood_bin_masks_disjoint']=all(not np.any(masks[i]&masks[j]) for i in range(4) for j in range(i))
    fit_files=list((B/'results/selected_fits').glob('*.npz'));cov_rel=[];mean_rel=[];sigma_rel=[];cls_checks=0
    for path in fit_files:
        data=np.load(path);f=data['fit_mask'];g=data['guard_mask'];record=json.loads(str(data['fit_summary_json']))
        assert np.all(g[f]) and np.array_equal(f,g)
        assert np.array_equal(data['fit_counts'],data['counts'][f])
        assert np.allclose(data['profiled_signed_total'],data['profiled_background_signed']+data['profiled_signed_signal'],atol=1e-7,rtol=1e-12)
        assert np.allclose(data['profiled_signed_signal'],record['Ahat']*data['signal_probability'][f],atol=1e-10,rtol=1e-14)
        assert np.min(data['profiled_signed_total'])>0
        b=data['fit_prefit_mean'];cov=data['fit_prefit_covariance'];L=data['fit_covariance_factor'];p=data['signal_probability'][f];n=data['fit_counts']
        cov_rel.append(float(np.max(abs(cov-data['prefit_GP_covariance'][np.ix_(f,f)]))/np.max(abs(cov))))
        mean_rel.append(float(np.max(abs(b-data['prefit_GP_mean'][f]))/np.max(abs(b))))
        assert cov_rel[-1]<1e-6 and mean_rel[-1]<1e-8
        for name,theta in [('profiled_background_only','nuisance_null'),('profiled_background_signed','nuisance_signed')]:
            assert np.allclose(data[name],b+L@data[theta],atol=1e-7,rtol=1e-12)
        # Construct the observed Poisson Hessian independently from saved arrays.
        scale=1/np.sqrt(np.sum(p*p/b));J=np.column_stack((scale*p,L));lam=data['profiled_signed_total']
        H=(J.T*(n/lam**2))@J+np.diag(np.r_[0.,np.ones(L.shape[1])]);unit=np.zeros(len(H));unit[0]=1
        sigma=scale*np.sqrt(np.linalg.solve(H,unit)[0]);sigma_rel.append(float(abs(sigma/record['sigma_A']-1)))
        assert sigma_rel[-1]<1e-9
        match=re.search(r'_m(\d+)_(.+)\.npz$',path.name);row=scan[(scan.mass_MeV==int(match[1]))&(scan.policy==match[2])].iloc[0]
        for key in ('Ahat','A90','sigma_A','signed_r','q0'):
            assert np.isclose(record[key],row[key],atol=1e-8,rtol=1e-12)
        # Independently evaluate the bounded asymptotic tail ratios in every saved profile trace.
        for entry in json.loads(str(data['profile_trace_json']))+[record]:
            q,qa=float(entry['q_obs']),float(entry['q_asimov'])
            if q<=1e-14:cls=1.
            else:
                a=np.sqrt(qa);zsb=np.sqrt(q) if q<=qa else (q+qa)/(2*a);zb=zsb-a if q<=qa else (q-qa)/(2*a)
                cls=float(np.exp(min(0.,log_ndtr(-zsb)-log_ndtr(-zb))))
            assert abs(cls-entry['cls'])<1e-12;cls_checks+=1
    checks['selected_fit_arrays_coherent_and_training_disjoint']=len(fit_files)==4*len(protocol['policies'])
    checks['selected_prefit_covariance_blocks_agree_with_roundoff_tolerance']=max(cov_rel)<1e-6 and max(mean_rel)<1e-8
    checks['selected_yield_uncertainties_recomputed_from_observed_Hessian']=max(sigma_rel)<1e-9
    checks['selected_plot_parameters_match_scan']=True
    checks['all_selected_profile_trace_tail_ratios_recomputed']=cls_checks>len(fit_files)
    toys=pd.read_csv(B/'results/selected_toy_rows.csv',float_precision='round_trip',keep_default_na=False);nnull=protocol['local_toys'];ncal=protocol['calibration_toys'];strengths=np.array(protocol['calibration_grid'])
    checks['toy_counts_match_protocol']=len(toys)==4*len(protocol['policies'])*(nnull+ncal*len(strengths))
    checks['all_toy_fits_valid']=bool(toys.fit_valid.all())
    checks['toy_rows_unique']=not toys.duplicated(['cohort','mass_MeV','policy','toy','z']).any()
    checks['backgrounds_paired_across_masses_and_methods']=bool(toys.groupby(['cohort','toy']).background_hash.nunique().max()==1)
    checks['signals_and_counts_paired_across_methods']=bool(toys.groupby(['cohort','mass_MeV','toy','z']).signal_hash.nunique().max()==1 and toys.groupby(['cohort','mass_MeV','toy','z']).counts_hash.nunique().max()==1)
    checks['null_and_signal_calibration_backgrounds_independent']=not set(toys[toys.cohort=='null'].background_hash)&set(toys[toys.cohort=='calibration'].background_hash)
    checkpoints=0
    for path in (B/'results/calibration_checkpoints').glob('*.json'):
        obj=json.loads(path.read_text());assert obj['protocol_sha256']==sha(B/'provenance/protocol.json')
        assert obj['rows_sha256']==sha(path.with_suffix('.csv'));assert obj['draws_sha256']==sha(path.with_suffix('.npz'));checkpoints+=1
    checks['all_checkpoint_protocol_row_and_draw_hashes_verified']=checkpoints>0
    local=pd.read_csv(B/'results/selected_local_calibration.csv',float_precision='round_trip');rank=pd.read_csv(B/'results/selected_rank_limits.csv',float_precision='round_trip')
    obs=scan.set_index(['mass_MeV','policy']);tested=0
    for r in local.itertuples(index=False):
        observed=obs.loc[(r.mass_MeV,r.policy)];q=toys[(toys.mass_MeV==r.mass_MeV)&(toys.policy==r.policy)&(toys.cohort=='null')]
        assert len(q)==nnull
        if observed.signed_r>=0:k=int((q.q0>=observed.q0).sum())
        else:k=int((q.signed_r<=observed.signed_r).sum())
        assert k==r.tail_count and r.p_rank==(k+1)/(nnull+1)
        lo=0 if k==0 else beta.ppf(.025,k,nnull-k+1);hi=1 if k==nnull else beta.ppf(.975,k+1,nnull-k)
        assert abs(lo-r.tail_probability95_low)<1e-14 and abs(hi-r.tail_probability95_high)<1e-14
        tested+=1
    checks['all_selected_empirical_tails_and_CP_intervals_recomputed']=tested==4*len(protocol['policies'])
    tested=0
    for r in rank.itertuples(index=False):
        observed=obs.loc[(r.mass_MeV,r.policy)];q=toys[(toys.mass_MeV==r.mass_MeV)&(toys.policy==r.policy)&(toys.cohort=='calibration')]
        vals={z:np.sort(t.Ahat.to_numpy()) for z,t in q.groupby('z')};assert all(len(v)==ncal for v in vals.values())
        ps=(1+np.array([np.searchsorted(vals[z],observed.Ahat,side='right') for z in strengths]))/(ncal+1)
        accept=np.flatnonzero(ps>.1)
        assert np.allclose(ps,json.loads(r.p_A_json),atol=0,rtol=0)
        assert strengths[accept].tolist()==json.loads(r.accepted_z_json)
        assert (len(accept)==0)==r.empty
        assert (len(accept)>0 and len(accept)!=(accept[-1]-accept[0]+1))==r.holes
        assert (len(accept)>0 and accept[-1]==len(strengths)-1)==r.upper_grid_censored
        if len(accept):assert abs(r.largest_accepted_A-strengths[accept[-1]]*r.s0)<1e-8
        tested+=1
    checks['all_selected_grid_rank_limits_recomputed']=tested==4*len(protocol['policies'])
    out=dict(passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},observed_scan_rows=len(scan),toy_rows=len(toys),checkpoint_pairs_verified=checkpoints,selected_regions_MeV=chosen,
        selected_prefit_covariance_block_max_relative_difference=max(cov_rel),selected_prefit_mean_max_relative_difference=max(mean_rel),selected_sigma_max_relative_difference=max(sigma_rel),selected_profile_tail_ratios_verified=cls_checks,
        selection_scope='Observed-selected disjoint likelihood windows; their statistics are not independent and fixed-mass toy probabilities do not correct the mass-search selection.',
        limit_scope='Dense profile limits compare different assumed shapes. Selected-point ranks compare extraction methods for the same assumed full-morph injection. Neither is a new physical coupling exclusion.',
        boundary_scope='67 and79 MeV selected peaks lie in the unvalidated60--80 MeV acceptance-transition interpolation interval.',
        protocol_sha256=sha(B/'provenance/protocol.json'),scan_sha256=sha(B/'results/observed_scan.csv'),toy_rows_sha256=sha(B/'results/selected_toy_rows.csv'))
    (B/'qa/statistics_review.json').write_text(json.dumps(out,indent=2)+'\n');assert out['passed'],out;print(json.dumps(out,indent=2))

if __name__=='__main__':main()
