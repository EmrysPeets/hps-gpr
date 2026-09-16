from pathlib import Path
import shutil
R=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow');B=R/'study_results/v5p0p5_analysis_note_20260916'
poster=Path('/Users/emryspeets/Desktop/summer_26/HPS_General_Poster_fold/fall26_assets')
shutil.copytree(poster/'projection_sources',B/'inputs/poster_projection_sources',dirs_exist_ok=True,ignore=shutil.ignore_patterns('mplconfig','__pycache__'))
s=(poster/'make_hps_combined_projection.py').read_text().split('# Larger, white-backed inset.')[0]
s=s.replace("B = Path(__file__).resolve().parent","B = Path(__file__).resolve().parents[1]")
s=s.replace("S = B / 'projection_sources'","S = B / 'inputs/poster_projection_sources'")
s=s.replace("figsize=(10.4,7.3)","figsize=(10.4,5.8)")
s+=r'''
for ext in ['pdf','png','svg']:
 fig.savefig(B/'figures'/f'figure2_updated_overview.{ext}',bbox_inches='tight',pad_inches=.06)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.4,3.15))
fig.subplots_adjust(left=.055,right=.93,bottom=.21,top=.83,wspace=.32)
for ax,which,col,panel in zip(axs,['three','four'],['#00857d','#8C1515'],['a','b']):
 style(ax,(15,275),(1.2e-7,1.3e-4),True);world(ax,True)
 ax.xaxis.set_major_locator(FixedLocator([50,100,150,200,250]))
 ax.plot(q.mass_MeV,q[f'projected_{which}_minimal'],color='white',lw=3.3,zorder=9)
 ax.plot(q.mass_MeV,q[f'projected_{which}_minimal'],color=col,lw=2.2,zorder=10)
 ax.tick_params(labelsize=9)
 ax.set_xlabel(r'$m_{A^\prime}$ [MeV]',fontsize=12);ax.set_ylabel(r'$\epsilon^2$',fontsize=12)
 ax.set_title(f'({panel})  '+('2015 + 2016 + 2021' if which=='three' else '2015 + 2016 + 2019 + 2021'),loc='left',fontsize=11)
 ax.text(.96,.07,'Conditional full-exposure projection',transform=ax.transAxes,ha='right',fontsize=8.6,color=col,bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=.6))
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color=C['h15'],label='Published 2015 (95% CL)'),Line2D([],[],color=C['h16'],label='Published 2016 (95% CL)')],loc='upper center',ncol=2,frameon=False,fontsize=9)
for ext in ['pdf','png','svg']:
 fig.savefig(B/'figures'/f'figure2_updated_panels.{ext}',bbox_inches='tight',pad_inches=.06)
plt.close(fig)
'''
(B/'scripts/make_updated_figure2.py').write_text(s)
# Copy the compact input scan ledgers needed for the new overview.
E=R/'study_results/v5p6p1_2021_upper_limit_echoes_20260913'
for sc in ['ten_65','ten_75','ten_92','ten_120','ten_160','ten_210']:
 out=B/'inputs/echo_scans'/sc;out.mkdir(parents=True,exist_ok=True)
 for f in ['matched_asimov.csv','background_asimov.csv']:shutil.copy2(E/'derived/scans'/sc/f,out/f)
(B/'scripts/make_echo_overview.py').write_text(r'''
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
''')
