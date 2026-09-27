"""Read-only derivatives of pinned v5.0.5 inputs; no fit optimization or new data."""
from pathlib import Path
import os, sys, csv, json, hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
BASE=Path(__file__).resolve().parents[4]
SOURCE=BASE/'study_results/v5p0p5_analysis_note_20260916'
OUT=Path(__file__).resolve().parent/'assets'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(SOURCE/'scripts'))
from common import DATA, predict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200})
fig,axs=plt.subplots(1,3,figsize=(12.6,3.4),sharey=True)
for ax,year,col,support,search in zip(axs,['2015','2016','2021'],['#226f9b','#bb3f36','#55814b'],[(14,135),(30,210),(36,300)],[(19,100),(39,180),(50,250)]):
    dd=DATA[year];edges=dd['native_edges']*1000;xx=(edges[1:]+edges[:-1])/2;yy=dd['native_counts']/np.diff(edges)
    ax.plot(xx,np.where(yy>0,yy,np.nan),color=col,lw=.7)
    ax.axvspan(*support,color='#777777',alpha=.12)
    for edge in search:ax.axvline(edge,color='#444444',ls='--',lw=1)
    ax.set(yscale='log',xlim=(max(0,support[0]-10),support[1]+15),ylim=(10,4e6),xlabel=r'$m_{ee}$ [MeV]',title=year+(' (10%)' if year=='2021' else ' (full)'))
    ax.grid(alpha=.12);ax.tick_params(labelsize=10)
axs[0].set_ylabel('Events / MeV')
fig.tight_layout();fig.savefig(OUT/'datasets_current_horizontal.png');fig.savefig(OUT/'datasets_current_horizontal.pdf');plt.close(fig)
d=DATA['2021']; rows=[]; selected={}
for j,m in enumerate(d['masses']):
    mask=np.abs(d['x']-m/1000)<=2.25*d['sigma'][j]
    b,C=predict(d['x'],d['n'],mask,d['const'][j],d['ls'][j])
    V=C+np.diag(b); delta=d['n'][mask]-b
    Q=float(delta@cho_solve(cho_factor(V,lower=True),delta))
    rows.append({'mass_MeV':float(m),'Q':Q,'N_window_bins':int(mask.sum()),'Q_per_bin':Q/mask.sum(),'min_covariance_eigenvalue':float(np.linalg.eigvalsh(V).min())})
    if m in (60,120,220): selected[m]=(mask,b,C,Q)
with (OUT/'2021_conditional_residual_scan.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(10.4,3.65))
ax.plot([r['mass_MeV'] for r in rows],[r['Q_per_bin'] for r in rows],color='#177ca4',lw=2)
ax.axhline(1,color='#777777',lw=1,ls='--',label='Unit scale (reference only)')
ax.set(xlim=(50,250),xlabel='Test mass [MeV]',ylabel=r'$Q\,/\,N_{\mathrm{bin}}$')
ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=10)
fig.tight_layout();fig.savefig(OUT/'2021_conditional_residual_scan.png');fig.savefig(OUT/'2021_conditional_residual_scan.pdf');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(13,6.1),gridspec_kw={'height_ratios':[1.6,1]},sharex='col')
for k,(m,(mask,b,C,Q)) in enumerate(selected.items()):
    x=d['x'][mask]*1000;n=d['n'][mask];width=np.diff(d['edges'])[mask]*1000;sd=np.sqrt(np.maximum(0,np.diag(C)))
    a=axs[0,k];a.errorbar(x,n/width,yerr=np.sqrt(n)/width,fmt='.',color='black',label='2021 10% data',ms=5)
    a.plot(x,b/width,color='#c68918',label='Sideband GP prediction',lw=2)
    a.fill_between(x,(b-sd)/width,(b+sd)/width,color='#75b6d2',alpha=.4,label='GP constraint width')
    a.set_title(f'{m:g} MeV',loc='left');a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.grid(alpha=.12)
    a=axs[1,k];res=(n-b)/np.sqrt(b+np.diag(C));a.axhline(0,color='#777777',lw=1)
    a.errorbar(x,res,yerr=np.sqrt(n)/np.sqrt(b+np.diag(C)),fmt='.',color='black',ms=5)
    a.set(xlabel=r'$m_{ee}$ [MeV]');a.grid(alpha=.12)
    a.text(.03,.94,f'Q/Nbin = {Q/len(b):.2f}',transform=a.transAxes,va='top',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','alpha':.95},zorder=20)
axs[0,0].set_ylabel('Events / MeV');axs[1,0].set_ylabel('Standardized residual')
h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=11)
fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'2021_fixed_state_examples.png');fig.savefig(OUT/'2021_fixed_state_examples.pdf');plt.close(fig)
protocol={'input':str(SOURCE/'inputs/spectrum_2021.npz'),'sha256':hashlib.sha256((SOURCE/'inputs/spectrum_2021.npz').read_bytes()).hexdigest(),'recipe':'Archived const/ls states; fixed log-GP replay; exclusion +/-2.25 sigma; Q=delta^T (diag(b)+C_GP)^-1 delta','GP_support_MeV':[36,300],'search_MeV':[50,250],'new_data_opened':False,'hyperparameter_optimizations':0,'signal_fits':0,'toy_calibration':False,'interpretation':'Conditional held-out residual diagnostic. Nbin is the number of window bins, not fitted effective dof. No chi-square/KS p-values or GOF calibration claimed. Adjacent windows are correlated.','n_points':len(rows),'min_covariance_eigenvalue':min(r['min_covariance_eigenvalue'] for r in rows),'Q_per_bin_range':[min(r['Q_per_bin'] for r in rows),max(r['Q_per_bin'] for r in rows)]}
(OUT/'2021_conditional_residual_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(protocol,indent=2))
