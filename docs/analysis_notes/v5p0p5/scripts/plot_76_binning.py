from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
d=pd.read_csv(B/'derived/v505_76_binning.csv')
assert len(d)==180 and d.min_lambda.min()>0 and d.max_score.max()<3e-5
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(8,6.1),sharex=True)
styles=[(1,0,'#222222','-',r'Nominal'),(2,0,'#0072b2','-',r'$2\times$, original origin'),(2,1,'#0072b2','--',r'$2\times$, shifted origin'),(4,0,'#d55e00','-',r'$4\times$, original origin'),(4,2,'#d55e00','--',r'$4\times$, shifted origin')]
for ax,scope in zip(axs.flat,['combined','2015','2016','2021']):
 for f,o,c,ls,label in styles:
  s=d.query('scope==@scope and factor==@f and offset_base_bins==@o')
  ax.plot(s.mass_MeV,s.signed_r,color=c,ls=ls,lw=1.4,marker='o',ms=2.5,label=label)
 ax.axhline(0,color='#777777',lw=.7);ax.axvline(76,color='#aaaaaa',lw=.7,ls=':');ax.set_title('All three campaigns' if scope=='combined' else scope)
 ax.set_ylabel('Signed profile root $r$');ax.set_xlabel('Test mass [MeV]');ax.grid(alpha=.12)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9)
fig.tight_layout(rect=[0,0,1,.89])
for e in ['pdf','png']:fig.savefig(B/'figures'/f'v505_76_joint_binning.{e}',dpi=180,bbox_inches='tight')
print('Validated',len(d),'pointwise fits; max score',d.max_score.max())
