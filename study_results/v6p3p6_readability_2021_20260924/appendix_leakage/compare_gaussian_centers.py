#!/usr/bin/env python3
"""Change only the Gaussian center; retain both archived blind-window masks."""
from pathlib import Path
import hashlib, json, sys
sys.dont_write_bytecode = True
import make_leakage_plots as base
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def plot(frame, destination, compact=False):
    plt.rcParams.update({'font.size':8.5, 'axes.labelsize':8.5, 'axes.titlesize':9,
                         'legend.fontsize':7, 'font.family':'serif'})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1 if compact else 4.0))
    for ax, window, title in zip(axes, ['historical','starter'],
            [r'Historical blind window: $\pm2.25\sigma_m$', r'Starter blind window: $u\in[-4,3]$']):
        for key, label, style, color in [
            ('MC','v16 TC signal MC','o-',base.GRAY),
            ('central','Central-mass Gaussian','^--',base.ORANGE),
            ('shifted','Shifted Gaussian','s-',base.BLUE)]:
            ax.plot(frame.mass_MeV, 100*frame[f'{key}_{window}_training_fraction'],
                    style, color=color, label=label, ms=3.5)
        ax.set(title=title, xlabel='Generated signal mass [MeV]',
               ylabel='Full signal in GP training [%]', ylim=(0,35), xticks=[80,120,160,200,240])
        ax.grid(alpha=.18); ax.legend(loc='upper right')
    fig.tight_layout(rect=(0,0 if compact else .24,1,.98))
    if not compact:
        fig.text(.025,.025,
          'How to read: u = (m_rec - c) / sigma_core. Both Gaussian widths = historical sigma_m; only centers differ.\n'
          'Central-mass center = generated mass M. Shifted center = M - 3.22433 - 2.21399 ln(M/150), in MeV.\n'
          'Keep each panel\'s blind/training bins fixed when changing the Gaussian center. Historical window is\n'
          'centered at the shifted Gaussian; starter window uses the predicted MC core (c, sigma_core). Signals retain full\n'
          'selected normalization. Leakage includes only recorded GP training bins. No uncertainty bars:\n'
          'fixed-template probability integrals, not fits. Existing leakage and fit-impact results already used the shift.',
          fontsize=7.2, linespacing=1.3, va='bottom')
    for ext in ['png','pdf']: fig.savefig(destination.with_suffix('.'+ext),dpi=190)
    plt.close(fig)

def main():
    saved=pd.read_csv(B/'window_leakage_by_mass.csv',float_precision='round_trip')
    edges=base.S.D['edges']*1000; x=base.S.D['x']*1000
    records=[]; max_delta=0.
    for r in saved.itertuples():
        m=r.mass_MeV; gc=r.historical_center_MeV; sm=r.sigma_m_MeV
        old=(x>=gc-2.25*sm)&(x<=gc+2.25*sm)
        new=(x>=r.core_center_MeV-4*r.core_sigma_MeV)&(x<=r.core_center_MeV+3*r.core_sigma_MeV)
        row=dict(mass_MeV=m,central_center_MeV=m,shifted_center_MeV=gc,
                 shift_MeV=gc-m,sigma_m_MeV=sm,starter_center_MeV=r.core_center_MeV)
        for key, center in [('central',m),('shifted',gc)]:
            cdf=ndtr((edges-center)/sm); probs=np.diff(cdf)
            row[key+'_outside_spectrum_fraction']=float(cdf[0]+1-cdf[-1])
            assert abs(probs.sum()+row[key+'_outside_spectrum_fraction']-1)<1e-12
            for window,mask in [('historical',old),('starter',new)]:
                assert int(mask.sum())==getattr(r,window+'_bins')
                row[f'{key}_{window}_training_fraction']=float(probs[~mask].sum())
                low=(x<x[mask].min()); high=(x>x[mask].max())
                row[f'{key}_{window}_low_training_fraction']=float(probs[low].sum())
                row[f'{key}_{window}_high_training_fraction']=float(probs[high].sum())
                if key=='shifted':
                    delta=abs(row[f'{key}_{window}_training_fraction']-getattr(r,f'Gaussian_{window}_training_fraction'))
                    max_delta=max(max_delta,delta); assert delta<1e-12
        for window in ['historical','starter']:
            row[f'MC_{window}_training_fraction']=getattr(r,f'MC_{window}_training_fraction')
            row[f'center_change_{window}_percentage_points']=100*(row[f'central_{window}_training_fraction']-row[f'shifted_{window}_training_fraction'])
        records.append(row)
    frame=pd.DataFrame(records)
    frame.to_csv(B/'gaussian_center_leakage_comparison.csv',index=False,float_format='%.17g')
    plot(frame,B/'gaussian_center_leakage_comparison')
    validation=dict(passed=True,masses_MeV=frame.mass_MeV.tolist(),fixed_sigma_m=True,
      fixed_blind_and_training_bins=True,full_normalization=True,new_fits=0,new_toys=0,
      previous_Gaussian_already_shifted=True,maximum_previous_shifted_probability_difference=max_delta,
      geometry_sha256=sha(B/'window_leakage_by_mass.csv'),script_sha256=sha(Path(__file__)),
      results_sha256=sha(B/'gaussian_center_leakage_comparison.csv'),
      scope='Only Gaussian center changes; no updated inference or limit claim. MC reference is unchanged.')
    (B/'gaussian_center_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(frame[['mass_MeV','shift_MeV','central_starter_training_fraction','shifted_starter_training_fraction']].to_string(index=False))
if __name__=='__main__':main()
