#!/usr/bin/env python3
from pathlib import Path
import os,json,hashlib
os.environ['MPLCONFIGDIR']='/tmp/hps-v580-statistics-mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
S=pd.read_csv(OUT/'individual_information.csv');J=pd.read_csv(OUT/'coupling_decomposition.csv');E=pd.read_csv(OUT/'fixed_shape_exposure_control.csv')
old=ROOT/'study_results/v5p0p4_analysis_note_20260911/derived'
replay=[]
for y in ['2015','2016','2021']:
 f=old/('extension_global/2015_curves.csv' if y=='2015' else f'v502_global_{y}.csv')
 d=pd.read_csv(f)
 for truth,col in [('observed','observed_r'),('stress','asimov_r')]:
  x=S[(S.dataset==int(y))&(S.truth==truth)].merge(d[['mass_MeV',col]],on='mass_MeV')
  replay.append(dict(dataset=y,truth=truth,n=len(x),max_abs_root_error=float(abs(x.signed_r-x[col]).max()),source=str(f.relative_to(ROOT))))
pd.DataFrame(replay).to_csv(OUT/'archived_replay.csv',index=False)
B=ROOT/'study_results/v5p1p1_significance_covariance_injection_20260910/derived';quality=[]
for y in ['2015','2016','2021']:
 f=B/f'field_{y}/field.npz';a=np.load(f);v=a['validation'];offset=a['a'];sd=a['s'];mean=v.mean(axis=0);emp=v.std(axis=0,ddof=1)
 quality.append(dict(dataset=y,grid_points=len(offset),complete_poisson_spectra=len(v),offset_min=float(offset.min()),offset_max=float(offset.max()),offset_rms=float(np.sqrt(np.mean(offset**2))),median_abs_offset=float(np.median(abs(offset))),fraction_abs_offset_gt2=float(np.mean(abs(offset)>2)),median_response_sd=float(np.median(sd)),max_abs_poisson_mean_minus_asimov=float(abs(mean-offset).max()),rms_poisson_mean_minus_asimov=float(np.sqrt(np.mean((mean-offset)**2))),median_poisson_sd_over_response_sd=float(np.median(emp/sd)),source=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
pd.DataFrame(quality).to_csv(OUT/'archived_field_quality.csv',index=False)
colors={'2015':'#4477AA','2016':'#CC6677','2021':'#228833'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8})
fig,ax=plt.subplots(2,2,figsize=(6.8,5.6),constrained_layout=True)
for truth,a,title in [('observed',ax[0,0],'Observed signed local roots'),('stress',ax[0,1],'Deterministic stress response')]:
 for y in ['2015','2016','2021']:
  x=S[(S.truth==truth)&(S.dataset==int(y))&(S.mass_MeV.between(60,100))]
  a.plot(x.mass_MeV,x.signed_r,label=f'{y}'+(' 10%' if y=='2021' else ''),color=colors[y],lw=1.4)
 x=J[(J.truth==truth)&J.mass_MeV.between(60,100)]
 a.plot(x.mass_MeV,x.signed_r,color='black',lw=2,label='Common coupling')
 a.axhline(0,color='.65',lw=.8);a.set(title=title,xlabel='Test mass [MeV]',ylabel='Signed root r' if truth=='observed' else 'Stress offset a')
 a.axvline(76,color='.55',lw=.8,ls=':')
 if truth=='observed':
  a.set_ylim(-5.8,4.0);a.legend(ncol=2,frameon=False,loc='lower left')
a=ax[1,0]
for y in ['2015','2016','2021']:
 x=S[(S.truth=='observed')&(S.dataset==int(y))&S.mass_MeV.between(50,100)]
 a.plot(x.mass_MeV,x.information_fraction,label=f'{y}'+(' 10%' if y=='2021' else ''),color=colors[y],lw=1.8)
a.set(title='Coupling information shares',xlabel='Test mass [MeV]',ylabel='Fraction of coupling information',ylim=(0,1));a.legend(frameon=False)
a=ax[1,1]
for m,color in [(42,'#4477AA'),(76,'#CC6677'),(92,'#228833')]:
 x=E[E.mass_MeV==m];a.plot(x.relative_exposure,x.signed_stress_r,'o-',label=f'{m} MeV',color=color)
 full=float(x.loc[x.relative_exposure==1,'signed_stress_r'].iloc[0]);a.plot(x.relative_exposure,np.sqrt(x.relative_exposure)*full,ls=':',color=color,lw=1)
a.set(title='2016 full-shape exposure control',xlabel='Exposure / full exposure',ylabel='Stress offset a');a.legend(frameon=False,loc='lower left',bbox_to_anchor=(.01,.19))
a.text(.03,.04,'Dotted: sqrt(exposure) scaling\nNot historical 2016 10% data',transform=a.transAxes,fontsize=7.3)
for ext in ('png','pdf'):fig.savefig(OUT/f'coupling_and_exposure.{ext}',dpi=170)
plt.close(fig)
print(pd.DataFrame(quality).round(4).to_string(index=False));print(pd.DataFrame(replay).to_string(index=False))
