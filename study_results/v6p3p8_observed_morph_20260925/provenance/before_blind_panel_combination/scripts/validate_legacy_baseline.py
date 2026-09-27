#!/usr/bin/env python3
"""Compare the new Gaussian benchmark with the archived observed 2021 scan."""
import run_observed as R
import numpy as np
import pandas as pd

def conversion(mass):
    d=R.D;x=mass/1000.;s=R.C.sigma('2021',mass);e=d['native_edges'];w=np.diff(e)
    overlap=np.maximum(0.,np.minimum(e[1:],x+1.64*s)-np.maximum(e[:-1],x-1.64*s))
    density=float(np.sum(d['native_counts']*overlap/w)/(3.28*s))
    return float(3*np.pi*x*float(d['frad_effective'])*density/(2/137.))*1e-8

def main():
    old=pd.read_csv(R.B/'inputs/legacy_observed_display.csv',float_precision='round_trip')
    old=old[(old.policy=='logshift')&(old.scope=='2021')]
    new=pd.read_csv(R.B/'results/observed_scan.csv',float_precision='round_trip')
    new=new[new.policy=='gaussian_baseline'];rows=[]
    for r in new.itertuples(index=False):
        q=old[old.mass_MeV==r.mass_MeV].iloc[0];factor=conversion(r.mass_MeV)
        ay=float(r.Ahat-q.psi_hat*factor);se=float(r.sigma_A-q.sigma_psi*factor)
        ul=float(r.A90-q.psi90*factor);sr=float(r.signed_r-q.signed_root)
        assert abs(ay)<2e-5*max(r.sigma_A,1.)
        assert abs(se)<1e-7*max(r.sigma_A,1.)
        assert abs(ul)<2e-6*max(r.A90,1.)
        assert abs(sr)<2e-6 and abs(r.p0_fixed_mass-q.p0_asymptotic)<2e-6
        rows.append(dict(mass_MeV=r.mass_MeV,legacy_psi_to_yield=factor,Ahat_difference=ay,sigma_difference=se,A90_difference=ul,signed_r_difference=sr,p0_difference=float(r.p0_fixed_mass-q.p0_asymptotic)))
    frame=pd.DataFrame(rows);R.csv(R.B/'results/legacy_baseline_comparison.csv',frame)
    R.write(R.B/'qa/legacy_baseline.json',dict(passed=True,masses_compared=len(rows),
        archive='Frozen copy of v6.3.6 results/observed_display.csv, 2021/logshift rows only',
        conversion='Archived psi=epsilon2/1e-8 transformed back to full Gaussian yield by the archived local-density conversion at the generated mass; conversion used ONLY for agreement validation, not for new physical coupling limits.',
        maximum_absolute_differences={key:float(frame[key].abs().max()) for key in ('Ahat_difference','sigma_difference','A90_difference','signed_r_difference','p0_difference')},
        archived_input_sha256=R.sha(R.B/'inputs/legacy_observed_display.csv'),new_scan_sha256=R.sha(R.B/'results/observed_scan.csv'),script_sha256=R.sha(__file__)))
    print(frame.drop(columns=['mass_MeV','legacy_psi_to_yield']).abs().max().to_json())

if __name__=='__main__':main()
