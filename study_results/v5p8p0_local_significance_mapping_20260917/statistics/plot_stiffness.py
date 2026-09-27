from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hps-v580-statistics-mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;D=pd.read_csv(B/'stiffness_scan.csv')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
fig,ax=plt.subplots(2,2,figsize=(6.8,4.4),constrained_layout=True)
for mass,color,marker in [(76,'#CC6677','o'),(90,'#4477AA','s'),(92,'#228833','^')]:
 x=D[(D.mass_MeV==mass)&(D.truth=='stress')&(D.strength==0)];ax[0,0].plot(x.ls_factor,x.signed_r,marker+'-',color=color,label=f'{mass} MeV',lw=1.3,ms=4)
 x=D[(D.mass_MeV==mass)&(D.truth=='observed')];ax[0,1].plot(x.ls_factor,x.sigma_fisher_over_ref,marker+'-',color=color,lw=1.3,ms=4)
 nominal=float(x.loc[x.ls_factor==1,'sideband_log_marginal_likelihood'].iloc[0]);ax[1,1].plot(x.ls_factor,x.sideband_log_marginal_likelihood-nominal,marker+'-',color=color,lw=1.3,ms=4)
 x=D[(D.mass_MeV==mass)&(D.truth=='nominal_local_GP_control')&(D.strength==5)];ax[1,0].plot(x.ls_factor,x.delta_r,marker+'-',color=color,lw=1.3,ms=4)
for a in ax.ravel():
 a.set_xlabel('Length-scale multiplier');a.set_xticks([.5,.75,1,1.25]);a.axvline(1,color='.65',lw=.8,ls=':')
ax[0,0].axhline(0,color='.65',lw=.8);ax[0,0].set(title='Stress bias remains shape dependent',ylabel='Deterministic stress root a');ax[0,0].legend(frameon=False)
ax[0,1].set(title='More flexibility raises uncertainty',ylabel='Signal uncertainty / nominal');ax[0,1].axhline(1,color='.65',lw=.8)
ax[1,0].set(title='Same injected yield, weaker response',ylabel='Change in root: size-five injection');ax[1,0].axhline(5,color='.65',lw=.8,ls='--')
ax[1,1].set(title='Sideband model score favors nominal',ylabel='Change in log marginal likelihood');ax[1,1].axhline(0,color='.65',lw=.8)
for e in ('pdf','png'):fig.savefig(B/f'stiffness_tradeoff.{e}',dpi=180)
rows=[]
for mass in [76,90,92]:
 for factor in [.5,.75,1.,1.25]:
  x=D[(D.mass_MeV==mass)&(D.ls_factor==factor)]
  obs=x[x.truth=='observed'].iloc[0];stress=x[(x.truth=='stress')&(x.strength==0)].iloc[0];inj=x[(x.truth=='nominal_local_GP_control')&(x.strength==5)].iloc[0]
  nom=D[(D.mass_MeV==mass)&(D.ls_factor==1)&(D.truth=='observed')].iloc[0]
  rows.append(dict(mass_MeV=mass,ls_factor=factor,stress_offset=stress.signed_r,signal_uncertainty_ratio=obs.sigma_fisher_over_ref,injected_size5_delta_r=inj.delta_r,injected_size5_recovered_fraction=inj.recovered_signal_fraction,observed_r=obs.signed_r,sideband_delta_LML=obs.sideband_log_marginal_likelihood-nom.sideband_log_marginal_likelihood))
T=pd.DataFrame(rows);T.to_csv(B/'stiffness_selected_table.csv',index=False,float_format='%.17g')
print(T.round(4).to_string(index=False))
