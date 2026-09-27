"""Independent identities, synthetic-fit closure and template-probability checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import uproot
from scipy.special import ndtr
from analyze import locate
from template import make_template

B=Path(__file__).resolve().parents[1]

def main():
    checks=[]
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
        assert passed,name
    sources=json.loads((B/'provenance/source_files.json').read_text())
    for r in sources:
        check('source_hash:'+r['file'],hashlib.sha256((B/r['file']).read_bytes()).hexdigest()==r['sha256'])
    d=pd.read_csv(B/'results/centers_and_shapes.csv').set_index('mass_MeV')
    boot=pd.read_csv(B/'results/bootstrap_fits.csv')
    check('native_masses',list(d.index)==[m for m in range(30,176,5) if m!=150])
    check('all_primary_fits_valid',d.loc[d.primary_domain,'valid'].all())
    check('30MeV_failure_explicit',not d.loc[30,'valid'] and d.loc[30,'shape_bound_hit'])
    check('resampling_ledger_complete',len(boot)==29*64 and (boot.groupby('mass_MeV').size()==64).all())
    check('primary_replica_validity',boot[boot.mass_MeV>=40].valid.all())
    check('low_mass_not_in_primary',not d.loc[[30,35],'primary_domain'].any())
    for m,r in d.iterrows():
        a=np.load(B/'histograms'/f'm{m:03d}.npz')
        with uproot.open(B/'inputs/root'/f'EventSelection_Ap_{m}_MeV.root') as f:
            h=f['h_MinvScSm_GeneralLargeBins_Final_1']
            check(f'ROOT_TH1D_type:{m}',h.classname=='TH1D')
            check(f'ROOT_exact_counts:{m}',np.array_equal(a['counts'],h.values()))
            check(f'ROOT_exact_edges:{m}',np.array_equal(a['edges_MeV'],h.axis().edges()*1000))
        check(f'native_probability:{m}',abs(a['probability'].sum()-1)<1e-12 and np.all(a['probability']>=0))
        check(f'center_identity:{m}',abs((r.center_MeV-m)-r.shift_MeV)<2e-9)
        check(f'full_core_tail_partition:{m}',abs(r.core_fraction_2+r.left_tail_2+r.right_tail_2-1)<2e-11)
        for label in ['pole_ref','core_ref','core_fitted']:
            check(f'window_partition:{m}:{label}',abs(sum(r[f'{label}_{p}_2p25'] for p in ['fraction','left_tail','right_tail'])-1)<2e-11)
    e=np.linspace(0,250,401)
    # Expected bin counts from a known Gaussian recover the prescribed location/width.
    truth_c,truth_s=99.3,4.5
    y=1e6*np.diff(ndtr((e-truth_c)/truth_s))
    for pedestal in [False,True]:
        r=locate(100,e,y,pedestal=pedestal)
        check(f'bin_integrated_synthetic_fit:{pedestal}',r['valid'] and abs(r['center_MeV']-truth_c)<1e-4 and abs(r['sigma_core_MeV']-truth_s)<1e-4,r)
    # Check every one-MeV interpolated location and all native probabilities.
    for m in range(40,176):
        p,meta=make_template(m,e)
        check(f'template_normalization:{m}',np.all(p>=-1e-12) and abs(p.sum()+meta['below_support_probability']+meta['above_support_probability']-1)<1e-12)
        if m in d.index:
            a=np.load(B/'histograms'/f'm{m:03d}.npz')
            check(f'native_template_identity:{m}',np.allclose(p,a['probability'],atol=1e-14,rtol=0))
    p,meta=make_template(150,np.array([148.,149.,150.,151.,152.]))
    check('narrow_support_keeps_losses',p.sum()<.5 and meta['below_support_probability']>0 and meta['above_support_probability']>0 and abs(p.sum()+meta['below_support_probability']+meta['above_support_probability']-1)<1e-12)
    check('150_is_explicit_interpolation',meta['kind']=='interpolated' and meta['source_masses_MeV']==[145,155])
    for m in [30,35,176]:
        failed=False
        try:make_template(m,e)
        except ValueError:failed=True
        check(f'unsupported_mass_rejected:{m}',failed)
    # Independent replay of every held-out morph with the portable template API.
    s=pd.read_csv(B/'results/shape_comparisons.csv')
    for _,r in s.dropna(subset=['morph_full_cdf_distance']).iterrows():
        m=int(r.mass_MeV);p,meta=make_template(m,e,excluded_mass=m)
        h=np.load(B/'histograms'/f'm{m:03d}.npz')
        F=np.r_[0,np.cumsum(h['probability'])]
        Q=meta['below_support_probability']+np.r_[0,np.cumsum(p)]
        dist=np.max(abs(F-Q))
        check(f'heldout_morph_replay:{m}',abs(dist-r.morph_full_cdf_distance)<1e-10)
    a=np.load(B/'histograms/aligned_shapes.npz')
    check('common_shape_conserves_probability',np.all(a['probability']>=-1e-14) and np.allclose(a['probability'].sum(axis=1),1) and abs(a['common_probability'].sum()-1)<1e-12)
    check('pooled_shape_is_equal_mass_average',np.allclose(a['common_cdf'],a['cdf'].mean(axis=0)))
    old=json.loads((B/'provenance/previous_report_hashes.json').read_text())
    available={p:sha for p,sha in old.items() if Path(p).is_file()}
    # These external files can be edited by another task. Record drift separately;
    # never restore them and never mislabel it as a numerical-study failure.
    audit=[]
    for p,sha in available.items():
        after=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        audit.append(dict(path=p,before_sha256=sha,after_sha256=after,unchanged=after==sha,
                          scope='External report read only by v6.4; concurrent changes are preserved'))
    (B/'qa/previous_report_audit.json').write_text(json.dumps(dict(all_hashes_unchanged=all(x['unchanged'] for x in audit),items=audit),indent=2)+'\n')
    out=dict(passed=all(x['passed'] for x in checks),checks=len(checks),prior_reports_available=len(available),items=checks)
    (B/'qa/numerical_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='items'}))

if __name__=='__main__':main()
