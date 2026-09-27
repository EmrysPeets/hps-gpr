#!/usr/bin/env python3
"""Replay a bounded, declared sample directly from archived toy count arrays."""
import run_study as R
import numpy as np
import pandas as pd
import json

CASES=(
 ('tuning_gp_mean_m100_t000','common_starter',4,3.,False),
 ('calibration_functional_m220_t030','direct_starter',37,16.,False),
 ('evaluation_gp_mean_m160_t000','gaussian_baseline',3,0.,False),
 ('evaluation_functional_m220_t050','morph_starter',56,5.,False),
 ('mismatch_common_starter_m100','common_starter',17,5.,True),
 ('mismatch_morph_starter_m160','morph_starter',98,3.,True),
)

def main():
    rows=[]
    for name,policy,toy,z,mismatch in CASES:
        path=R.B/f'results/checkpoints/{name}.csv'
        frame=pd.read_csv(path,float_precision='round_trip')
        record=frame[(frame.policy==policy)&(frame.toy==toy)&(frame.z==z)].iloc[0]
        mass=int(record.mass_MeV);ctx=R.Context(mass,policy)
        meta=json.loads(path.with_suffix('.json').read_text())
        assert R.sha(path)==meta['rows_sha256']
        assert R.sha(path.with_suffix('.npz'))==meta['draws_sha256']
        protocol=R.B/('provenance/calibration_mismatch_protocol.json' if mismatch else 'provenance/toy_protocol.json')
        assert R.sha(protocol)==meta['protocol_sha256']
        with np.load(path.with_suffix('.npz')) as saved:
            if mismatch:
                index=toy;strengths=np.array(R.GRID)
                cats=np.asarray(saved['probabilities'])
                assert np.allclose(cats,R.T.categories(mass,R.D['edges']*1000,method=R.CANDIDATES[policy]['shape'],omit=mass),rtol=0,atol=1e-15)
            else:
                index=int(np.flatnonzero(saved['toys']==toy)[0]);strengths=saved['strengths'];cats=ctx.categories
            zi=int(np.flatnonzero(strengths==z)[0])
            bg=saved['backgrounds'][index].copy()
            draw=saved['signal_categories'][index*len(strengths)+zi].copy()
            assert len(draw)==len(bg)+2 and len(cats)==len(draw)
            assert np.all(draw>=0) and np.all(draw==draw.astype(np.int64))
            assert np.min(cats)>=0 and abs(cats.sum()-1)<1e-12
            assert R.ahash(bg)==record.background_hash and R.ahash(draw)==record.signal_hash
            counts=bg+draw[1:-1]
            if not mismatch:
                assert R.ahash(counts)==record.counts_hash
                assert int(draw[1:-1][ctx.fit].sum())==record.actual_window
            assert int(draw.sum())==record.actual_full
            assert int(draw[1:-1][~ctx.guard].sum())==record.actual_training
            assert int(draw[0]+draw[-1])==record.actual_outside_support
            assert not np.any(ctx.fit&~ctx.guard)
            replay=ctx.fit_counts(counts)
            assert replay['fit_valid']
            ay=float(replay['Ahat']-record.Ahat);se=float(replay['sigma']-record.sigma)
            assert np.isclose(replay['Ahat'],record.Ahat,rtol=5e-12,atol=5e-7)
            assert np.isclose(replay['sigma'],record.sigma,rtol=5e-12,atol=5e-8)
            rows.append(dict(checkpoint=name,source=str(record.source),mass_MeV=mass,policy=policy,toy=toy,z=z,
                calibration_signal='predicted omitted-mass shape' if mismatch else 'direct full selected TC MC',
                csv_sha256=R.sha(path),draws_sha256=R.sha(path.with_suffix('.npz')),
                saved_Ahat=float(record.Ahat),replayed_Ahat=float(replay['Ahat']),Ahat_difference=ay,
                saved_sigma=float(record.sigma),replayed_sigma=float(replay['sigma']),sigma_difference=se,
                replay_score=float(replay['score']),full_selected_signal_draw=int(draw.sum()),
                signal_in_fit=int(draw[1:-1][ctx.fit].sum()),signal_in_GP_training=int(draw[1:-1][~ctx.guard].sum()),
                signal_outside_analysis_support=int(draw[0]+draw[-1]),category_probability_sum=float(cats.sum())))
    result=dict(passed=True,replayed_fits=len(rows),sample_selection='Six declared fits spanning three primary cohorts, both background sources, all major extraction shapes, and both mismatch calibration shapes',
        comparison_tolerance=dict(relative=5e-12,Ahat_absolute=5e-7,sigma_absolute=5e-8),
        accounting='Saved signal draw is [below analysis support, one category per analysis bin, above analysis support]. Only interior categories add to the background. Full draw total remains the selected yield.',
        fit_training_disjoint=True,checkpoint_hashes_verified=True,counts_reconstructed_from_archived_arrays=True,
        maximum_absolute_Ahat_difference=max(abs(r['Ahat_difference']) for r in rows),
        maximum_absolute_sigma_difference=max(abs(r['sigma_difference']) for r in rows),
        replay_script_sha256=R.sha(__file__),cases=rows)
    R.write(R.B/'qa/fit_replay.json',result)
    print(json.dumps({k:result[k] for k in ('passed','replayed_fits','maximum_absolute_Ahat_difference','maximum_absolute_sigma_difference')},indent=2))

if __name__=='__main__':main()
