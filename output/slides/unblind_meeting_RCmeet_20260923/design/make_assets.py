"""Exact threshold-table replot and typeset yield-conversion equations."""
from pathlib import Path
import os,json,hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1]/'assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(10.8,3.65),sharex=True)
series=[('1% × 10: threshold refinement',[.7890,.1394],[[.6293,.9488],[-.0499,.3288]],['Baseline model','Threshold-refined'],'#1a62b0'),('Native 10%: extended support',[.7160,-.2463],[[.5452,.8867],[-.4165,-.0761]],['Baseline model','Threshold + 30–300 MeV'],'#087d78')]
for ax,(title,vals,cis,labels,col) in zip(axs,series):
    ax.axvspan(-.5,.5,color='#eaf1f7',zorder=0);ax.axvline(0,c='#525866',lw=1,ls='--')
    for y,(v,(lo,hi)) in enumerate(zip(vals,cis)):
        ax.errorbar(v,1-y,xerr=[[v-lo],[hi-v]],fmt='o',ms=9,lw=2.5,c='#d66a24' if y==0 else col,capsize=5)
        ax.text(v,1-y+.13,f'{v:+.3f}',ha='center',fontsize=12,color='#525866')
    ax.set(yticks=[1,0],yticklabels=labels,ylim=(-.5,1.55),xlim=(-.65,1.07),xlabel='Mean extraction pull (90% interval)',title=title)
    ax.grid(axis='x',alpha=.17)
fig.tight_layout(w_pad=2.5)
for ext in ['png','pdf','svg']:fig.savefig(O/f'slide21_threshold_no_footer.{ext}',dpi=280,bbox_inches='tight',facecolor='white')
plt.close(fig)
def equation(name,text,figsize=(6,1.05),fontsize=28):
    fig=plt.figure(figsize=figsize,facecolor='white');fig.text(.01,.5,text,va='center',fontsize=fontsize,color='#172b4d')
    for ext in ['png','pdf','svg']:fig.savefig(O/f'{name}.{ext}',dpi=320,bbox_inches='tight',pad_inches=.025,facecolor='white')
    plt.close(fig)
equation('yield_relation',r'$A_d=K_d(m)\,\varepsilon_{ee}^{\,2}$',figsize=(5.3,.75))
equation('epsilon_from_yield',r'$\varepsilon_{90,ee}^{\,2}=\frac{A_{90,d}}{K_d(m)}=\frac{2\alpha_{\rm EM}}{3\pi}\,\frac{A_{90,d}}{m\,f_{{\rm rad},d}(m)\,\rho_d(m)}$',figsize=(10.3,1),fontsize=27)
equation('epsilon_physical',r'$\varepsilon_{90,\rm phys}^{\,2}=N_{\rm eff}^{\rm BR}\,\varepsilon_{90,ee}^{\,2}$',figsize=(6,.85),fontsize=26)
equation('epsilon_compact',r'$\varepsilon_{90,ee}^{\,2}=\frac{2\alpha_{\rm EM}\,A_{90,d}}{3\pi\,m\,f_{{\rm rad},d}\,\rho_d}$',figsize=(5.3,1.25),fontsize=28)
equation('injection_yield',r'$A_{{\rm inj},t}=z\,\sigma_{A,{\rm before},t},\quad z=0,1,3,5$',figsize=(8,.75),fontsize=26)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*') if p.is_file()}
(O/'asset_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Threshold figure and equations generated.')
