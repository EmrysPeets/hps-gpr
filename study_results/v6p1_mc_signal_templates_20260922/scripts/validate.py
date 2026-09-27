"""Independent arithmetic, source conservation, and likelihood QA."""
from pathlib import Path
import json,hashlib,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.special import ndtr
from common import DATA,sigma,moving_context,continuous_signal,OneSignalProfile
B=Path(__file__).resolve().parents[1]
checks=[]
def check(name,condition,**extra):checks.append(dict(name=name,passed=bool(condition),**extra))

def main():
    selected=0;filecount=0
    for mass in range(40,261,20):
        d=dict(np.load(B/f'histograms/m{mass:03d}.npz'));meta=json.loads(str(d['metadata']));selected+=meta['stats']['selected'];filecount+=len(meta['files'])
        h=d['sumw'];edges=d['edges_GeV'];parts=[dict(np.load(p)) for p in sorted((B/'histograms').glob(f'm{mass:03d}_*.npz'))]
        check(f'source_conservation_{mass}',h.sum()+meta['stats']['underflow']+meta['stats']['overflow']==meta['sumw'])
        check(f'batch_sum_{mass}',np.array_equal(sum(p['sumw'] for p in parts),h))
        check(f'unit_weights_{mass}',np.array_equal(h,d['sumw2']) and np.isclose(meta['sumw'],meta['neffective']))
        check(f'valid_mass_and_type_{mass}',meta['stats']['wrong_type']==0 and abs(meta['truth_mass_min_GeV']*1000-mass)<.02 and abs(meta['truth_mass_max_GeV']*1000-mass)<.02)
        if mass<50 or mass>250:continue
        target=DATA['2021']['edges']
        # Explicit overlap integration, independent of the CDF rebinning.
        overlap=np.maximum(0.,np.minimum(target[1:,None],edges[None,1:])-np.maximum(target[:-1,None],edges[None,:-1]))
        prob=(overlap/np.diff(edges))@h;prob/=prob.sum()
        from run_mc_study import distribution
        candidate,fraction=distribution(mass,source=mass)
        check(f'exact_overlap_rebin_{mass}',np.max(abs(prob-candidate))<2e-13,max_absolute_error=float(np.max(abs(prob-candidate))))
        expected_fraction=float(((overlap/np.diff(edges))@h).sum()/meta['sumw'])
        check(f'all_selected_support_fraction_{mass}',abs(fraction-expected_fraction)<2e-13)
    check('complete_input_files',filecount==26,files=filecount,selected=selected)
    f=pd.read_csv(B/'derived/scans.csv');c=pd.read_csv(B/'derived/comparison.csv');lo=pd.read_csv(B/'derived/leave_one_out.csv');bo=pd.read_csv(B/'derived/mc_bootstrap.csv')
    check('observed_grid_complete',len(f)==402 and (f.groupby('mass_MeV').size()==2).all() and len(c)==10)
    check('native_mc_grid',set(f[f.model=='mc_direct'].mass_MeV)==set(range(60,241,20)))
    check('likelihood_endpoints',np.isfinite(f.display_epsilon2_90).all() and (f.display_epsilon2_90>0).all() and abs(f.cls-.1).max()<2e-6 and f.max_score.max()<3e-5 and f.min_lambda.min()>0)
    check('local_probability_identity',np.max(abs(f.p0_fixed_mass-ndtr(-f.Z0)))<1e-14 and (f.Z0>=0).all())
    check('ratio_arithmetic',np.max(abs(c.mc_limit/c.gaussian_limit-c.limit_ratio))<1e-12)
    check('conditional_mc_precision_complete',len(bo)==320 and (bo.groupby('mass_MeV').size()==32).all())
    check('leave_one_out_complete',len(lo)==10 and set(lo.mass_MeV)==set(range(60,241,20)))
    old=pd.read_csv(B/'derived/gaussian_reference.csv',dtype={'scope':str});old=old[old.scope=='2021']
    j=f[f.model=='gaussian'].merge(old,on='mass_MeV',suffixes=('','_reference'))
    check('gaussian_reference_replay',len(j)==201 and abs(j.epsilon2_90/j.epsilon2_90_reference-1).max()<2e-7 and abs(j.Z0-j.Z0_reference).max()<2e-7,
          maximum_relative_limit_error=float(abs(j.epsilon2_90/j.epsilon2_90_reference-1).max()))
    if (B/'derived/core_scans.csv').exists():
        cs=pd.read_csv(B/'derived/core_scans.csv');new=cs[cs.framework=='core_centered'];cent=pd.read_csv(B/'derived/core_centers.csv')
        check('core_centered_grid',len(new)==402 and len(cent)==201 and cent.core_location_valid.all())
        check('core_centered_local_probabilities',np.max(abs(new.p0_fixed_mass-ndtr(-new.Z0)))<1e-14)
        check('core_centered_endpoints',new.ok.all() and abs(new.cls-.1).max()<2e-6 and new.max_score.max()<3e-5 and new.min_lambda.min()>0)
        hw=np.array([2.25*sigma('2021',m)*1000 for m in new.mass_MeV])
        check('core_window_center_and_width',np.max(abs((new.window_low_MeV+new.window_high_MeV)/2-new.core_center_MeV))<1e-10 and np.max(abs((new.window_high_MeV-new.window_low_MeV)/2-hw))<1e-10)
        from run_mc_study import distribution
        unchanged=True
        for m in range(60,241,20):
            h=np.load(B/f'histograms/core_centered_windows/m{m:03d}.npz');w,_=distribution(m)
            unchanged &= np.array_equal(w,h['probability'])
        check('core_centering_keeps_empirical_template',unchanged)
    result=dict(passed=all(r['passed'] for r in checks),checks=checks,check_count=len(checks),selected_MC_candidates=selected,
                scope='Numerical/template validation. Production selection equivalence, detector response calibration and global significance are not established.')
    (B/'qa/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    assert result['passed'],[r for r in checks if not r['passed']]
if __name__=='__main__':main()
