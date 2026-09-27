"""Rebuild the opening vector schematic, without external data or fits."""
from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hps-585-fairness-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
B=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42})
fig=plt.figure(figsize=(7.4,5.7));ax=fig.add_axes([.015,.02,.97,.96]);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
colors={'data':'#e5ecf3','analysis':'#e1eee8','source':'#f4eadb','field':'#eae4f2','result':'#e1ebef'}
def box(x,y,w,h,title,body,c):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.5,rounding_size=1.0',facecolor=colors[c],edgecolor='#617180',lw=.7))
 ax.text(x+w/2,y+h-2.5,title,ha='center',va='top',fontweight='bold',fontsize=9.2,color='#1d364e')
 ax.text(x+w/2,y+2.0,body,ha='center',va='bottom',fontsize=8.2,linespacing=1.45,color='#243241')
def arrow(x,y,xx,yy):
 ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'-|>','lw':1.0,'color':'#586574','shrinkA':2,'shrinkB':2})
def stage(y,txt):ax.text(1,y,txt,fontsize=9.8,fontweight='bold',color='#1d364e')
stage(97,'THE OBSERVED ANALYSIS')
box(1,75,29,17,'Observed spectrum',r'Counts $n_j$'+'\nFrozen input','data')
box(35,75,30,17,'Masked GP + likelihood',r'$\pm2.25\sigma_m$ exclusion'+'\nSideband fit; profile amplitude','analysis')
box(70,75,29,17,'Observed scan',r'Signed root $r_{\rm obs}(m)$'+'\nKeep the raw curve','data')
arrow(30.5,83.5,34.5,83.5);arrow(65.5,83.5,69.5,83.5)
stage(67,'THE NULL EXPERIMENT AND ITS EMULATOR')
box(1,42,29,20,'Choose source $B$',r'Generating mean $B_j$'+'\nControl: validate transfer\nSame-data GP: conditional','source')
box(35,42,30,20,'Repeat the analysis',r'$n_j^*\sim{\rm Poisson}(B_j)$'+'\nRefit the masked GP\nObtain complete scans $r^*(m)$','analysis')
box(70,42,29,20,'Significance-field GP',r'Offsets $a$; scales $s$; matrix $R$'+'\nSample correlated maxima\nCheck complete Poisson fits','field')
arrow(30.5,52,34.5,52);arrow(65.5,52,69.5,52)
stage(34,'SAME DECLARED TEST FOR OBSERVED AND NULL SCANS')
box(1,7,46,22,'Keep the raw ordering',r'$T_{\rm raw}=\max_m\max[r(m),0]$'+'\nCalibrate the same raw maximum.\nNo local subtraction is required.','result')
box(53,7,46,22,'Reference-standardized ordering',r'$T_{\rm ref}=\max_{m:r(m)>0}[r(m)-a_m]/s_m$'+'\nReranking is possible.\nThis defines a different diagnostic.','result')
ax.text(50,1.3,'A shared source is not an independent background measurement.',ha='center',fontsize=8.7,color='#624e36')
for ext in ['pdf','png']:fig.savefig(B/'figures'/f'significance_schematic.{ext}',bbox_inches='tight',dpi=180)
plt.close(fig)
