"""Publication figures from saved field-scan data; no trajectory scan on plotting."""
import json,csv,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import build_study as bs
B=bs.B
G=json.loads((B/'inputs/geometry/active_sensors.json').read_text())['geometries']
T=json.loads((B/'derived/magnetic_example_trajectories.json').read_text())
S=json.loads((B/'derived/magnetic_summary.json').read_text())
with (B/'derived/magnetic_acceptance.csv').open() as f:rows=list(csv.DictReader(f))
with (B/'derived/selected_mass_spectra.csv').open() as f:spectra=list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
def beamframe(p,g):
 a=g['beam_angle_rad'];p=np.array(p)
 return np.column_stack((p[:,0]*np.sin(a)+p[:,2]*np.cos(a),p[:,0]*np.cos(a)-p[:,2]*np.sin(a),p[:,1]))
def geometry(ax,meta,compact=False):
 year,E,color,*_=meta;g=G[year]
 for s in g['sensors']:
  corners=beamframe(s['corners_mm'],g)
  ax.add_collection3d(Poly3DCollection([corners],facecolor=color,edgecolor=color,alpha=.15,linewidth=.5))
 ax.plot([0,940],[0,0],[0,0],color='#737d87',lw=1)
 for i,ex in enumerate(T[year]):
  for p,col in zip(ex['paths_s_xyz_uxuyuz'],['#bc3d37','#2166ac']):
   r=beamframe(np.array(p)[:,1:4],g)
   ax.plot(*r.T,color=col,lw=1.7,ls='-' if i==0 else '--')
 ax.set(xlim=(-10,950),ylim=(-65,120),zlim=(-70,70),xticks=[0,450,900],yticks=[-50,50],zticks=[-50,0,50])
 ax.set_xlabel('Along beam [mm]',labelpad=2);ax.set_ylabel('Horizontal [mm]',labelpad=0);ax.set_zlabel('Vertical [mm]',labelpad=0)
 ax.set_box_aspect((3,1.1,1.2));ax.view_init(elev=19,azim=-65);ax.tick_params(pad=0,labelsize=8 if compact else 9)
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.label.set_size(8.5 if compact else 10)
 ax.set_title(f'HPS {year} | {E/1000:.3f} GeV',color=color,weight='bold',pad=0,fontsize=12 if compact else 14)
 ax.text2D(.5,-.02,f"Passing examples: {T[year][0]['mass_MeV']:.0f} and {T[year][1]['mass_MeV']:.0f} MeV",transform=ax.transAxes,ha='center',color=color,weight='semibold',fontsize=10 if compact else 12)
 ax.text2D(.5,-.105,f'Nominal 15 mrad scale ≈ {E*np.sin(.015):.2f} MeV',transform=ax.transAxes,ha='center',fontsize=9 if compact else 11)
def curve(ax,year,color,labels=False):
 r=[d for d in rows if d['year']==year];m=np.array([float(d['mass_MeV']) for d in r])
 keys=['field_selected_x1','field_selected_x0p8','zero_field_selected_x1','field_all_stations_x1']
 for key,ls,col,lw in zip(keys,['-','--',':','-.'],[color,color,'#777f87',color],[2,1.7,1.6,1.]):
  ax.plot(m,[100*float(d[key]) for d in r],color=col,ls=ls,lw=lw,alpha=.65 if 'all_stations' in key else 1)
 ax.set(xlim=(0,bs.MASS_LIMITS[year]),ylim=(0,65),xlabel='True A′ mass [MeV]')
 ax.set_xticks({'2015':[0,50,100,150,200,250],'2016':[0,100,200,300,400,500],'2021':[0,200,400,600,800,950]}[year]);ax.set_yticks([0,20,40,60]);ax.grid(alpha=.17)
 if labels:ax.set_ylabel('Accepted decay fraction [%]')
 ax.spines[['top','right']].set_visible(False)
def spectrum(ax,meta,labels=False):
 year,E,color,key,crop,endpoint,sample=meta;r=[d for d in spectra if d['year']==year]
 v=np.array([float(d['selected_pairs']) if d['status']=='displayed' and float(d['selected_pairs'])>0 else np.nan for d in r])
 ax.stairs(v,np.arange(301),color=color,lw=1.5,baseline=None);ax.axvspan(0,crop,color='#d9dfe6',alpha=.5,lw=0)
 if endpoint<300:
  ax.axvspan(endpoint,300,color='#d9dfe6',alpha=.5,lw=0);ax.text(.74,.48,'No source bins\nabove 150 MeV',transform=ax.transAxes,ha='center',color='#687582',fontsize=9)
 ax.set(xlim=(0,300),ylim=(50,3e6),yscale='log',xticks=[0,100,200,300],xlabel='Reconstructed mass [MeV]')
 if labels:ax.set_ylabel('Selected pairs / MeV')
 ax.text(.95,.9,sample,transform=ax.transAxes,ha='right',color=color,weight='bold');ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
legend=[Line2D([0],[0],color=c,ls=s,lw=w) for c,s,w in [('#485460','-',2),('#485460','--',1.7),('#777f87',':',1.6),('#485460','-.',1)]]
labels=['Field + hit cuts, x = 1','Field + hit cuts, x = 0.8','B = 0 + hit cuts, x = 1','Field + all stations, x = 1']
fig=plt.figure(figsize=(14.7,5.8))
for i,meta in enumerate(bs.META):geometry(fig.add_axes([.012+i*.327,.24,.295,.64],projection='3d'),meta)
fig.suptitle('HPS magnetic transport through the active silicon tracker',y=.97,fontsize=19,weight='semibold')
fig.legend([Line2D([0],[0],color='#bc3d37',lw=2),Line2D([0],[0],color='#2166ac',lw=2),Line2D([0],[0],color='#485460',lw=2),Line2D([0],[0],color='#485460',lw=2,ls='--')],
 ['Positron','Electron','Low-mass example','High-mass example'],loc='lower center',bbox_to_anchor=(.5,.084),ncol=4,frameon=False,fontsize=11)
fig.text(.5,.05,'Examples pass the run-dependent hit cuts at the first and last nonzero sampled masses; they do not define exact endpoints.',ha='center',fontsize=10.5)
fig.text(.5,.013,'Pinned 3D field maps; transverse display scales expanded. 15 mrad labels use equal sharing, x = 1 and negligible electron mass.',ha='center',fontsize=10.5)
for ext in ('pdf','png','svg'):fig.savefig(B/'figures'/f'HPS_v5p7p1_geometry_illustration.{ext}',dpi=190)
plt.close(fig)
fig=plt.figure(figsize=(13.8,7.3));fig.suptitle('Magnetic-field acceptance and selected mass spectra',y=.99,fontsize=18,weight='semibold')
fig.legend(legend,labels,loc='upper center',bbox_to_anchor=(.51,.947),ncol=2,frameon=False,fontsize=11)
for i,meta in enumerate(bs.META):
 year,E,color,*_=meta;left=.065+i*.325
 fig.text(left+.12,.824,f'HPS {year} | {E/1000:.3f} GeV',ha='center',color=color,weight='bold',fontsize=13)
 curve(fig.add_axes([left,.505,.25,.285]),year,color,i==0)
 spectrum(fig.add_axes([left,.082,.25,.28]),meta,i==0)
fig.text(.065,.405,'Recorded pair spectra; no exposure scaling',fontsize=12,weight='semibold')
for ext in ('pdf','png'):fig.savefig(B/'figures'/f'HPS_v5p7p1_response_spectra.{ext}',dpi=190)
plt.close(fig)
fig=plt.figure(figsize=(15.3,13.6));fig.suptitle('HPS A′ acceptance: geometry, magnetic transport and selected spectra | v5.7.1',y=.985,fontsize=17.8,weight='semibold')
for i,meta in enumerate(bs.META):
 year,E,color,*_=meta
 geometry(fig.add_axes([.006+i*.327,.70,.297,.23],projection='3d'),meta,True)
 curve(fig.add_axes([.068+i*.325,.385,.255,.20]),year,color,i==0)
 spectrum(fig.add_axes([.068+i*.325,.105,.255,.18]),meta,i==0)
fig.text(.065,.954,'A | Active faces and passing e⁺ (red) / e⁻ (blue): solid low-mass, dashed high-mass examples',fontsize=12,weight='semibold')
fig.text(.065,.64,'B | Decay fraction through the magnetic field and required sensor views',fontsize=12,weight='semibold')
fig.legend(legend,labels,loc='upper center',bbox_to_anchor=(.52,.623),ncol=4,frameon=False,fontsize=10.3)
fig.text(.065,.313,'C | Existing selected-pair spectra',fontsize=12,weight='semibold')
fig.text(.065,.034,'Hit cuts: 2015 ≥5 paired stations each + e⁺ L1/L2; 2016 ≥5 each; 2021 ≥10 e⁺ / ≥8 e⁻ individual strip hits.\n'
 'Illustrated low/high masses are sampled passing examples. Curves include the field; material, hit finding, trigger and reconstruction remain omitted.',fontsize=10.5,linespacing=1.5)
for ext in ('pdf','png'):fig.savefig(B/'figures'/f'HPS_v5p7p1_acceptance_overview.{ext}',dpi=175)
plt.close(fig)
# Generated field table precedes the report figure; sampled supports are explicitly not exact endpoints.
with (B/'derived/magnetic_table.tex').open('w') as f:
 f.write('\\begin{tabular}{lrrrr}\\toprule\nRun & $B_y(\\boldsymbol r_{\\rm map,0})$ [T] & Sampled masses [MeV] & Peak [MeV] & Peak fraction\\\\\\midrule\n')
 for s in S:f.write(f"{s['year']} & {s['B_at_table_origin_T'][1]:.4f} & {s['positive_sampled_masses_MeV'][0]:.0f}--{s['positive_sampled_masses_MeV'][1]:.0f} & {s['sampled_peak_MeV']:.0f} & {100*s['sampled_peak_fraction']:.1f}\\%\\\\\n")
 f.write('\\bottomrule\\end{tabular}\n')
