
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
cat=pd.read_csv(B/'derived/v560_catalogue.csv').set_index('scenario')
fig,axs=plt.subplots(3,2,figsize=(8.7,7.6),sharey=True)
for ax,sc in zip(axs.flat,['ten_65','ten_75','ten_92','ten_120','ten_160','ten_210']):
 d=pd.read_csv(B/'inputs/echo_scans'/sc/'matched_asimov.csv');b=pd.read_csv(B/'inputs/echo_scans'/sc/'background_asimov.csv');c=cat.loc[sc];m=c.mass_MeV;s=c.sigma_MeV
 ax.axvspan(m-2*s,m+2*s,color='#edf0f2',zorder=0)
 ax.axhline(1,color='#555555',lw=.8);ax.axhline(.9,color='#909090',lw=.7,ls=':')
 ax.plot(d.mass_MeV,d.A90/b.A90,color='#156f83',lw=1.8)
 ax.set(xlim=(max(50,m-12*s),min(250,m+12*s)),ylim=(.08,20),yscale='log',title=f'{m:.0f} MeV; target $Z$ = {c.target_Z:.2f}')
 ax.set_xlabel('Test mass [MeV]');ax.set_ylabel(r'$A_{90}^{\rm inj}/A_{90}^{(b)}$')
 ax.grid(alpha=.16)
fig.tight_layout(pad=1.2)
for ext in ['pdf','png']:fig.savefig(B/'figures'/f'v505_ten_echo_summary.{ext}',dpi=180,bbox_inches='tight')
