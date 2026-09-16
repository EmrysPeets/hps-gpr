"""Poster derivative of the frozen 2026-09-13 Figure 2 inputs.

Run with Python 3 and numpy, pandas, matplotlib, PyMuPDF and shapely.
No fit, interpolation of observed limits, or new sensitivity model is performed.
The displaced full-luminosity reach is recovered from its exact PDF vector path.
"""
from pathlib import Path
import os, sys, json, hashlib
B = Path(__file__).resolve().parents[1]
S = B / 'inputs/poster_projection_sources'
sys.path.insert(0, str(S / 'vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(S / 'mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, MultipleLocator
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely import make_valid
import fitz

# Arial is Helvetica-compatible and provides reliable regular/bold/italic
# embedding here; custom math uses the same family, including Greek symbols.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',
 'mathtext.bf':'Arial:bold','mathtext.bfit':'Arial:italic:bold',
 'mathtext.sf':'Arial','mathtext.fallback':'stixsans',
 'font.size':12,'axes.labelsize':15,
 'xtick.labelsize':11,'ytick.labelsize':11,'axes.linewidth':.9,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,
 'lines.solid_capstyle':'round','lines.solid_joinstyle':'round','path.simplify':False})
C = dict(world='#e3e6e9', edge='#737d85', h15='#24669c', h16='#cc652c',
 combined='#8C1515', full='#688d43', mu='#794594', e='#ac4274', thermal='#bd7e77')
W=S/'world_contours'
SPECS=[('babar.csv','BaBar','gev','eps',True),('kloe.csv','KLOE','mev','eps2',False),('2014_kloe.csv','KLOE-2','mev','eps2',False),('2014_mainz.csv','A1/MAMI','gev','eps2',False),('apex_test_run.csv','APEX','mev','eps2',False),('2015_na482_rescaled_to_95cl.csv','NA48/2','gev','eps',True),('hades.csv','HADES','gev','eps2',False),('2014_phenix.csv','PHENIX','mev','eps2',False),('lhcb_2019_results_prompt_data.csv','LHCb','gev','eps2',False)]
PSPECS=[('e137_andreas_log.csv','E137','loggev','logeps',False),('e141_andreas_log.csv','E141','loggev','logeps',False),('e774_andreas_log.csv','E774','loggev','logeps',False),('orsay_andreas_95pct.dat','Orsay','loggev','logeps',False),('kek.csv','KEK','gev','eps',False),('u70_serpuhov.csv','U70','gev','eps',True),('na64_2018.csv','NA64','gev','eps',False),('na64_2019.csv','NA64','gev','eps',False)]
def xy(name,unit,kind,scale=False,swap=False):
 a=np.genfromtxt(W/name,delimiter=','); x,y=a[:,int(swap)],a[:,int(not swap)]
 if unit=='gev': x=1000*x
 elif unit=='loggev': x=1000*10**x
 if kind=='eps': y=y*y
 elif kind=='logeps': y=10**(2*y)
 if scale: y=y*(1.64/1.96)
 return x,y
def segments(x,y):
 ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)&(y<1e-3)
 idx=np.flatnonzero(ok)
 breaks=np.flatnonzero((np.diff(idx)>1)|(x[idx[1:]]/x[idx[:-1]]>1.45))+1
 return [(x[j],y[j]) for j in np.split(idx,breaks) if len(j)>1]
curves=[]; polys=[]
for name,label,unit,kind,scale in SPECS:
 x,y=xy(name,unit,kind,scale)
 for sx,sy in segments(x,y):
  pts=np.column_stack([np.log10(sx),np.log10(sy)])
  polys.append(make_valid(Polygon(np.vstack([pts,[pts[-1,0],0.],[pts[0,0],0.]]))))
  curves.append((label,sx,sy))
for name,label,unit,kind,swap in PSPECS:
 x,y=xy(name,unit,kind,swap=swap); ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
 polys.append(make_valid(Polygon(np.column_stack([np.log10(x[ok]),np.log10(y[ok])]))))
union=unary_union(polys)
def polygon_paths(g):
 if g.is_empty: return
 if g.geom_type=='Polygon':
  verts=[]; codes=[]
  for ring in [g.exterior,*g.interiors]:
   v=10**np.asarray(ring.coords)
   verts.extend(v);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(v)-2)+[MPath.CLOSEPOLY])
  yield MPath(verts,codes)
 elif hasattr(g,'geoms'):
  for s in g.geoms: yield from polygon_paths(s)
def world(ax,zoom=False):
 xl=ax.get_xlim();yl=ax.get_ylim()
 g=union.intersection(box(np.log10(xl[0]),np.log10(yl[0]),np.log10(xl[1]),np.log10(yl[1])))
 for path in polygon_paths(g): ax.add_patch(PathPatch(path,facecolor=C['world'],edgecolor=C['edge'],lw=.65,zorder=1))
 for lab,x,y in curves: ax.plot(x,y,color=C['edge'],lw=.4,alpha=.27,zorder=2)
 for name,col in [('engineering_run2015_fix.csv','h15'),('engineering_run2016_reach.csv','h16')]:
  x,y=xy(name,'mev','eps2',False)
  for xx,yy in segments(x,y): ax.plot(xx,yy,color=C[col],lw=1.35,zorder=7)
 if not zoom:
  x1,y1=xy('thermal_targets_lower.csv','gev','eps2');x2,y2=xy('thermal_targets_upper.csv','gev','eps2')
  x=np.geomspace(max(min(x1),min(x2)),min(max(x1),max(x2)),400)
  a=np.exp(np.interp(np.log(x),np.log(x1),np.log(y1)));b=np.exp(np.interp(np.log(x),np.log(x2),np.log(y2)))
  ax.fill_between(x,np.minimum(a,b),np.maximum(a,b),color=C['thermal'],alpha=.19,zorder=0,lw=0)
  ax.plot(x,a,color=C['thermal'],lw=.8);ax.plot(x,b,color=C['thermal'],lw=.8)
def style(ax,xlim,ylim,zoom=False):
 ax.set(xscale='linear' if zoom else 'log',yscale='log',xlim=xlim,ylim=ylim,xlabel=r'$m_{A^\prime}\;[\mathrm{MeV}]$',ylabel=r'$\epsilon^2$')
 ticks=([40,75,100,150,200,250] if zoom else [1,2,5,10,20,50,100,200,500,1000])
 ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
 ax.xaxis.set_minor_locator(MultipleLocator(10) if zoom else LogLocator(base=10,subs=np.arange(2,10)));ax.xaxis.set_minor_formatter(NullFormatter())
 ax.yaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]));ax.yaxis.set_minor_formatter(NullFormatter())
 ax.tick_params(which='both',direction='in',top=True,right=True,length=4,labelleft=False,labelright=True)
 ax.yaxis.set_label_position('right')
 ax.tick_params(which='minor',length=2)
 ax.grid(axis='y',which='major',lw=.4,color='#cbd0d5',alpha=.45)
 ax.set_axisbelow(True)
def label(ax,x,y,s,color='#57626a',fs=11,**kw):
 return ax.text(x,y,s,color=color,fontsize=fs,ha='center',va='center',zorder=15,**kw)
halo=[]

# Recover original green polyline using its identifying stroke color and width.
# These calibrations are exact PDF major-tick centers: x=10 and 1000 MeV;
# y=1e-4 and 1e-10.  Logarithmic transformation preserves all 33 source nodes.
original=fitz.open(S/'original_reach_full_lumi.pdf')
p=original[0]
paths=[d for d in p.get_drawings() if d.get('color') and np.allclose(d['color'],(109/255,144/255,79/255),atol=1e-6) and d.get('width')==4]
assert len(paths)==1
items=paths[0]['items']; assert all(it[0]=='l' for it in items)
pts=np.array([tuple(items[0][1])]+[tuple(it[2]) for it in items])
mx=10**(1+(pts[:,0]-286.36328125)/(752.1620483398438-286.36328125)*2)
ey=10**(-4-(pts[:,1]-17.57769775390625)/(613.4424438476562-17.57769775390625)*6)
full=pd.DataFrame(dict(mass_MeV=mx,epsilon_squared=ey,pdf_x_pt=pts[:,0],pdf_y_pt=pts[:,1]))
full.to_csv(S/'full_luminosity_vector_nodes.csv',index=False,float_format='%.17g')
assert len(full)==33 and np.allclose(pts[0],pts[-1])

q=pd.read_csv(S/'projected_contours.csv',float_precision='round_trip')
def combined(ax,lw=2.6):
 ax.plot(q.mass_MeV,q.projected_four_minimal,color='white',lw=lw+1.4,zorder=9)
 ax.plot(q.mass_MeV,q.projected_four_minimal,color=C['combined'],lw=lw,ls='-',zorder=10)

fig=plt.figure(figsize=(10.4,5.8))
ax=fig.add_axes([.035,.10,.88,.865])
style(ax,(1,1000),(8e-11,5e-4));world(ax);combined(ax)
g=pd.read_csv(S/'g2_central_curves.csv')
ax.plot(g.mass_MeV,g.muon_central_eps2,color=C['mu'],lw=1.2,ls=(0,(5,2,1.5,2)),zorder=6)
ax.plot(g.mass_MeV,g.electron_Rb_central_eps2,color=C['e'],lw=1.2,ls=(0,(2.5,1.8)),zorder=6)
ax.plot(mx,ey,color=C['full'],lw=2.1,zorder=8)
label(ax,4.3,7e-5,'E774 / E141')
label(ax,3.5,1.0e-6,'Orsay / KEK')
label(ax,6.8,7.5e-7,'NA64 / E137',fs=10.5)
label(ax,10.5,7e-6,'KLOE');label(ax,29,2.5e-6,'NA48/2')
label(ax,110,3.1e-4,'A1 / MAMI');label(ax,345,3.1e-5,'HADES')
label(ax,730,8e-5,'KLOE-2');label(ax,450,4.3e-6,'LHCb')
label(ax,380,1.6e-6,'BaBar')
label(ax,350,1.9e-9,'Thermal relic targets',color='#975b57',rotation=19,fs=11)
label(ax,145,2.5e-8,'HPS full-luminosity reach',color=C['full'],fs=11.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.90,pad=.5))
label(ax,145,1.65e-8,'displaced search simulation',color=C['full'],fs=8.7,bbox=dict(facecolor='white',edgecolor='none',alpha=.90,pad=.5))
label(ax,470,1.8e-4,r'$(g-2)_e$ central',color=C['e'],fs=10.5,rotation=22,path_effects=halo)
label(ax,550,1.13e-5,r'$(g-2)_\mu$ central',color=C['mu'],fs=10.5,rotation=19,path_effects=halo)
label(ax,27,1.1e-4,'HPS 2015 published\n(95% CL)',color=C['h15'],fs=10.5)
label(ax,125,1.3e-4,'HPS 2016 published\n(95% CL)',color=C['h16'],fs=10.5)
ax.annotate('HPS Combined Projection',xy=(205,float(q.loc[q.mass_MeV==205,'projected_four_minimal'].iloc[0])),xytext=(210,1.7e-7),color=C['combined'],fontsize=12.8,ha='center',va='center',zorder=15,arrowprops=dict(arrowstyle='-',color=C['combined'],lw=.9),bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.8))
label(ax,210,1.1e-7,'2015 + 2016 + 2019 + 2021 (90% CL)',color=C['combined'],fs=9.8,bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.8))


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
