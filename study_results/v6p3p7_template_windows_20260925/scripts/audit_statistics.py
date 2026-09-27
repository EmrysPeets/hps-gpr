#!/usr/bin/env python3
"""Independent semantic audit of the saved v6.3.7 statistical construction."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import beta

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B/'scripts'))
import templates as T

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    protocol=json.loads((B/'provenance/toy_protocol.json').read_text())
    selection=json.loads((B/'results/frozen_selection.json').read_text())
    checks={}
    checks['frozen_runner_hash_matches']=sha(B/'scripts/run_study.py')==protocol['script_sha256']
    checks['frozen_template_hash_matches']=sha(B/'scripts/templates.py')==protocol['templates_sha256']
    checks['selection_uses_saved_tuning']=sha(B/'results/tuning_rows.csv')==selection['tuning_rows_sha256']
    n=protocol['toys_per_cohort']; masses=protocol['masses_MeV']; sources=protocol['sources']
    policies=selection['evaluation_policies']; grid=np.asarray(protocol['strengths_grid'])
    frames={name:pd.read_csv(B/f'results/{name}_rows.csv',float_precision='round_trip')
            for name in ('tuning','calibration','evaluation')}
    expected={'tuning':n*len(masses)*len(protocol['candidates'])*2,
              'calibration':n*len(masses)*len(policies)*len(grid)*len(sources),
              'evaluation':n*len(masses)*len(policies)*len(protocol['evaluation_strengths'])*len(sources)}
    checks['cohort_sizes_match_protocol']=all(len(frames[k])==v for k,v in expected.items())
    checks['all_fits_numerically_valid']=all(f.fit_valid.all() for f in frames.values())
    for name,frame in frames.items():
        checks[name+'_unique_keys']=not frame.duplicated(['source','mass_MeV','policy','toy','z']).any()
        checks[name+'_backgrounds_paired_across_mass_strength_policy']=bool(frame.groupby(['source','toy']).background_hash.nunique().max()==1)
        checks[name+'_signals_paired_across_policies']=bool(frame.groupby(['source','mass_MeV','toy','z']).signal_hash.nunique().max()==1)
    hashsets={name:set(frame.background_hash) for name,frame in frames.items()}
    checks['no_background_draw_reused_across_cohorts']=all(not hashsets[a]&hashsets[b]
        for a,b in (('tuning','calibration'),('tuning','evaluation'),('calibration','evaluation')))
    checks['sources_use_distinct_background_draws']=all(
        not set(frame[frame.source==sources[0]].background_hash)&set(frame[frame.source==sources[1]].background_hash)
        for frame in frames.values() if len(frame.source.unique())>1)
    tuning=pd.read_csv(B/'results/tuning_response_summary.csv')
    objectives={}
    for policy,q in tuning.groupby('policy'):
        if protocol['candidates'][policy]['shape']!='direct' and (q.response>0).all():
            objectives[policy]=float(np.exp(np.mean(np.log(q.sensitivity_SD_over_response))))
    checks['selection_reproduced_from_tuning_only']=min(objectives,key=objectives.get)==selection['selected_policy']
    masks=np.load(B/'results/window_masks.npz')
    checks['no_signal_fit_bin_used_in_GP_training']=all(
        np.all(masks[f'{p}_m{m}_guard'][masks[f'{p}_m{m}_fit']]) for m in masses for p in protocol['candidates'])
    checks['full_MC_probability_categories_normalized']=all(
        abs(masks[f'{p}_m{m}_categories'].sum()-1)<1e-12 and masks[f'{p}_m{m}_categories'].min()>=0
        for m in masses for p in protocol['candidates'])
    bank=T.bank()
    checks['test_mass_omitted_from_morph_anchors']=all(m not in dict(bank.neighbors(m,omit=m)) for m in masses)
    checks['test_mass_omitted_from_common_anchors']=all(m not in bank._retained(m) for m in masses)
    checks['center_and_width_predicted_without_test_mass']=all(
        np.allclose(bank.parameters(m,omit=m),[sum(w*bank.samples[a][k] for a,w in bank.neighbors(m,omit=m)) for k in ('center','width')],atol=0,rtol=0)
        for m in masses)
    inference=pd.read_csv(B/'results/evaluation_inference.csv',float_precision='round_trip')
    cal=frames['calibration']; verified=0
    for (source,mass,policy),q in inference.groupby(['source','mass_MeV','policy']):
        c=cal[(cal.source==source)&(cal.mass_MeV==mass)&(cal.policy==policy)]
        values={float(z):np.sort(g.Ahat.to_numpy()) for z,g in c.groupby('z')}
        assert all(len(x)==n for x in values.values())
        for r in q.itertuples(index=False):
            ranks=(1+np.array([np.searchsorted(values[z],r.Ahat,side='right') for z in grid]))/(n+1)
            accepted=np.flatnonzero(ranks>.1)
            p0=(1+n-np.searchsorted(values[0.],r.Ahat,side='left'))/(n+1)
            assert np.allclose(ranks,json.loads(r.p_A_json),atol=0,rtol=0)
            assert grid[accepted].tolist()==json.loads(r.accepted_z_json)
            assert p0==r.p_background_only and (p0<=.1)==r.reject_background_only
            assert (len(accepted)==0)==r.empty
            holes=len(accepted)>0 and len(accepted)!=(accepted[-1]-accepted[0]+1)
            assert holes==r.holes
            assert (len(accepted)>0 and accepted[-1]==len(grid)-1)==r.upper_grid_censored
            assert (ranks[np.where(grid==r.z)[0][0]]>.1)==r.true_grid_value_accepted
            verified+=1
    checks['all_saved_rank_tests_recomputed']=verified==len(inference)==len(frames['evaluation'])
    summary=pd.read_csv(B/'results/inference_summary.csv',float_precision='round_trip')
    for r in summary.itertuples(index=False):
        for prefix,count in (('reject',r.reject_background_only_count),('true_grid_rejected',r.true_grid_rejected_count)):
            low=0. if count==0 else beta.ppf(.025,count,n-count+1)
            high=1. if count==n else beta.ppf(.975,count+1,n-count)
            assert abs(low-getattr(r,prefix+'95_low'))<1e-14
            assert abs(high-getattr(r,prefix+'95_high'))<1e-14
    checks['binomial_intervals_use_exact_95pct_and_actual_denominator']=bool((summary.toys==n).all())
    checkpoint_count=0
    for path in (B/'results/checkpoints').glob('*.json'):
        marker=json.loads(path.read_text())
        pp='calibration_mismatch_protocol.json' if path.name.startswith('mismatch_') else 'toy_protocol.json'
        assert marker['protocol_sha256']==sha(B/'provenance'/pp)
        assert marker['rows_sha256']==sha(path.with_suffix('.csv'))
        assert marker['draws_sha256']==sha(path.with_suffix('.npz'))
        checkpoint_count+=1
    checks['all_checkpoint_row_and_draw_hashes_valid']=checkpoint_count>0
    mismatch_verified=0
    if (B/'results/mismatch_inference.csv').exists():
        mp=json.loads((B/'provenance/calibration_mismatch_protocol.json').read_text())
        checks['mismatch_frozen_script_hash_matches']=sha(B/'scripts/calibration_mismatch.py')==mp['script_sha256']
        mc=pd.read_csv(B/'results/mismatch_calibration_rows.csv',float_precision='round_trip')
        mi=pd.read_csv(B/'results/mismatch_inference.csv',float_precision='round_trip')
        me=frames['evaluation'];me=me[(me.source=='gp_mean')&me.policy.isin(mp['policies'])]
        me=me.set_index(['mass_MeV','policy','toy','z'])
        checks['mismatch_calibration_fits_valid']=bool(mc.fit_valid.all())
        checks['mismatch_calibration_count_matches']=len(mc)==n*len(masses)*len(mp['policies'])*len(grid)
        checks['mismatch_draws_independent_of_all_primary_cohorts']=not set(mc.background_hash)&set.union(*hashsets.values())
        for (mass,policy),q in mi.groupby(['mass_MeV','policy']):
            c=mc[(mc.mass_MeV==mass)&(mc.policy==policy)]
            values={float(z):np.sort(g.Ahat.to_numpy()) for z,g in c.groupby('z')}
            for r in q.itertuples(index=False):
                y=me.loc[(mass,policy,r.toy,r.z)].Ahat
                ranks=(1+np.array([np.searchsorted(values[z],y,side='right') for z in grid]))/(n+1)
                accepted=np.flatnonzero(ranks>.1)
                p0=(1+n-np.searchsorted(values[0.],y,side='left'))/(n+1)
                assert np.allclose(ranks,json.loads(r.p_A_json),atol=0,rtol=0)
                assert grid[accepted].tolist()==json.loads(r.accepted_z_json)
                assert p0==r.p_background_only and (p0<=.1)==r.reject_background_only
                assert (len(accepted)==0)==r.empty
                holes=len(accepted)>0 and len(accepted)!=(accepted[-1]-accepted[0]+1)
                assert holes==r.holes
                assert (len(accepted)>0 and accepted[-1]==len(grid)-1)==r.upper_grid_censored
                assert (ranks[np.where(grid==r.z)[0][0]]<=.1)==r.true_grid_rejected
                mismatch_verified+=1
        checks['all_mismatch_rank_tests_recomputed']=mismatch_verified==len(mi)==len(me)
        ms=pd.read_csv(B/'results/mismatch_summary.csv',float_precision='round_trip')
        for r in ms.itertuples(index=False):
            for prefix,count in (('reject',r.reject_background_only_count),('true_grid_rejected',r.true_grid_rejected_count)):
                low=0. if count==0 else beta.ppf(.025,count,n-count+1)
                high=1. if count==n else beta.ppf(.975,count+1,n-count)
                assert abs(low-getattr(r,prefix+'95_low'))<1e-14
                assert abs(high-getattr(r,prefix+'95_high'))<1e-14
        checks['mismatch_binomial_intervals_use_exact_95pct_and_actual_denominator']=bool((ms.toys==n).all())
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        ranks_independently_verified=verified,mismatch_ranks_independently_verified=mismatch_verified,
        verified_checkpoint_pairs=checkpoint_count,cohort_rows={k:len(v) for k,v in frames.items()},
        selected_policy=selection['selected_policy'],rank_rejection_bound=10/101,
        endpoint_reporting='Saved endpoint quantiles omit empty accepted sets; top-grid acceptance is right-censored and must be flagged, not called an exact continuous limit.',
        calibration_scope='Primary calibration uses full direct MC at each tested mass; does not establish off-grid detector-shape correctness.',
        count_intervals='Clopper-Pearson intervals are conditional on the saved calibration sample; threshold-estimation uncertainty is separate.',
        inputs={name:sha(B/name) for name in ['provenance/toy_protocol.json','results/frozen_selection.json','results/evaluation_inference.csv','scripts/run_study.py','scripts/templates.py']})
    (B/'qa/statistics_review.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['passed'],result
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
