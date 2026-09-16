"""Vector publication contours, with source nodes and polygon topology retained."""
from pathlib import Path
import sys,os,json
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'scripts/vendor'))
os.environ.setdefault('MPLCONFIGDIR',str(B/'qa/mpl'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch,Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import FixedLocator,FuncFormatter,LogLocator,NullFormatter
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from shapely import make_valid
from scipy.integrate import quad
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':320,'lines.solid_capstyle':'round','lines.solid_joinstyle':'round','path.simplify':False})
F=B/'figures';W=B/'inputs/world_contours';F.mkdir(exist_ok=True)
C={'world':'#e3e6e9','edge':'#727b83','h15':'#27669b','h16':'#cb662d','three':'#20754a','four':'#007d8a','mu':'#7b4996','e':'#ab4573','thermal':'#c1807b'}
SPECS=[('babar.csv','BaBar','gev','eps',True),('kloe.csv','KLOE','mev','eps2',False),('2014_kloe.csv','KLOE-2','mev','eps2',False),('2014_mainz.csv','A1/MAMI','gev','eps2',False),('apex_test_run.csv','APEX','mev','eps2',False),('2015_na482_rescaled_to_95cl.csv','NA48/2','gev','eps',True),('hades.csv','HADES','gev','eps2',False),('2014_phenix.csv','PHENIX','mev','eps2',False),('lhcb_2019_results_prompt_data.csv','LHCb','gev','eps2',False)]
PSPECS=[('e137_andreas_log.csv','E137','loggev','logeps',False),('e141_andreas_log.csv','E141','loggev','logeps',False),('e774_andreas_log.csv','E774','loggev','logeps',False),('orsay_andreas_95pct.dat','Orsay','loggev','logeps',False),('kek.csv','KEK','gev','eps',False),('u70_serpuhov.csv','U70','gev','eps',True),('na64_2018.csv','NA64','gev','eps',False),('na64_2019.csv','NA64','gev','eps',False)]
def xy(name,unit,kind,scale=False,swap=False):
 a=np.genfromtxt(W/name,delimiter=',');x,y=a[:,int(swap)],a[:,int(not swap)]
 if unit=='gev':x=1000*x
 elif unit=='loggev':x=1000*10**x
 if kind=='eps':y=y*y
 elif kind=='logeps':y=10**(2*y)
 if scale:y=y*(1.64/1.96)
 return x,y
def segments(x,y):
 # Keep invalid/sentinel rows as breaks, rather than connecting across them.
 ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)&(y<1e-3)
 idx=np.flatnonzero(ok)
 breaks=np.flatnonzero((np.diff(idx)>1)|(x[idx[1:]]/x[idx[:-1]]>1.45))+1
 return [(x[j],y[j]) for j in np.split(idx,breaks) if len(j)>1]
curves=[];polys=[];audit=[]
for name,label,unit,kind,scale in SPECS:
 x,y=xy(name,unit,kind,scale)
 for sx,sy in segments(x,y):
  points=np.column_stack([np.log10(sx),np.log10(sy)])
  poly=make_valid(Polygon(np.vstack([points,[points[-1,0],0.],[points[0,0],0.]])))
  polys.append(poly);curves.append((label,sx,sy))
 audit.append(dict(file=name,role='upper-limit boundary',mass_unit=unit,y_kind=kind,display_95_to_90=scale))
for name,label,unit,kind,swap in PSPECS:
 x,y=xy(name,unit,kind,swap=swap);ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
 p=make_valid(Polygon(np.column_stack([np.log10(x[ok]),np.log10(y[ok])])));polys.append(p)
 audit.append(dict(file=name,role='closed exclusion polygon',mass_unit=unit,y_kind=kind,swap_columns=swap))
union=unary_union(polys)
def polygon_paths(g):
 if g.is_empty:return
 if g.geom_type=='Polygon':
  verts=[];codes=[]
  for ring in [g.exterior,*g.interiors]:
   v=np.asarray(ring.coords);v=10**v
   verts.extend(v);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(v)-2)+[MPath.CLOSEPOLY])
  yield MPath(verts,codes)
 elif hasattr(g,'geoms'):
  for s in g.geoms:yield from polygon_paths(s)
def world(ax,zoom=False):
 xl=ax.get_xlim();yl=ax.get_ylim();g=union.intersection(box(np.log10(xl[0]),np.log10(yl[0]),np.log10(xl[1]),np.log10(yl[1])))
 for path in polygon_paths(g):ax.add_patch(PathPatch(path,facecolor=C['world'],edgecolor=C['edge'],lw=.5,zorder=1))
 # The unobscured boundary of the union is primary; internal boundaries stay quiet.
 for lab,x,y in curves:ax.plot(x,y,color=C['edge'],lw=.35,alpha=.33,zorder=2)
 for name,col in [('engineering_run2015_fix.csv','h15'),('engineering_run2016_reach.csv','h16')]:
  x,y=xy(name,'mev','eps2',True)
  for xx,yy in segments(x,y):ax.plot(xx,yy,color=C[col],lw=1.2 if zoom else 1.25,zorder=7)
 if not zoom:
  x1,y1=xy('thermal_targets_lower.csv','gev','eps2');x2,y2=xy('thermal_targets_upper.csv','gev','eps2')
  x=np.geomspace(max(min(x1),min(x2)),min(max(x1),max(x2)),400)
  a=np.exp(np.interp(np.log(x),np.log(x1),np.log(y1)));b=np.exp(np.interp(np.log(x),np.log(x2),np.log(y2)))
  ax.fill_between(x,np.minimum(a,b),np.maximum(a,b),color=C['thermal'],alpha=.20,zorder=0,lw=0)
  ax.plot(x,a,color=C['thermal'],lw=.65);ax.plot(x,b,color=C['thermal'],lw=.65)
alpha=1/137.035999206
xg=np.geomspace(1,1000,900)
def vector_f(m,ml):
 r=m/ml
 return quad(lambda z:2*z*(1-z)**2/((1-z)**2+r*r*z),0,1,epsabs=1e-18,epsrel=2e-10,limit=200)[0]
gmu=np.array([2*np.pi*3.8e-10/(alpha*vector_f(m,105.6583745)) for m in xg])
ge=np.array([2*np.pi*3.4e-13/(alpha*vector_f(m,.510998950)) for m in xg])
pd.DataFrame({'mass_MeV':xg,'muon_central_eps2':gmu,'electron_Rb_central_eps2':ge}).to_csv(B/'derived/g2_central_curves.csv',index=False,float_format='%.17g')
def g2(ax):
 ax.plot(xg,gmu,color=C['mu'],lw=1.2,ls=(0,(5,2,1.5,2)),zorder=8)
 ax.plot(xg,ge,color=C['e'],lw=1.2,ls=(0,(2.5,1.8)),zorder=8)
def style(ax,xlim,ylim,zoom=False):
 ax.set(xscale='log',yscale='log',xlim=xlim,ylim=ylim,xlabel=r'$m_{A^\prime}\;[\mathrm{MeV}]$',ylabel=r'$\epsilon^2$')
 ticks=([20,30,50,100,200] if zoom else [1,2,5,10,20,50,100,200,500,1000]);ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
 ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)));ax.xaxis.set_minor_formatter(NullFormatter())
 ax.yaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]));ax.yaxis.set_minor_formatter(NullFormatter())
 ax.tick_params(which='both',direction='in',top=True,right=True,length=3);ax.tick_params(which='minor',length=1.8)
 ax.set_axisbelow(True)
 ax.grid(axis='y',which='major',lw=.35,color='#cbd0d5',alpha=.5)
def label(ax,x,y,s,**kw):
 return ax.text(x,y,s,color='#535e66',fontsize=8.3,ha='center',va='center',zorder=10,**kw)
def overview(ax,zoom_box=False):
 style(ax,(1,1000),(8e-11,5e-4));world(ax);g2(ax)
 label(ax,4.3,7e-5,'E774 / E141');label(ax,3.5,1.0e-6,'Orsay / KEK');label(ax,14,3.5e-8,'NA64');label(ax,13,9e-10,'E137')
 label(ax,10.5,7e-6,'KLOE');label(ax,31,2.5e-6,'NA48/2');label(ax,110,1.9e-4,'A1 / MAMI');label(ax,350,2.9e-5,'HADES');label(ax,720,8e-5,'KLOE-2');label(ax,470,5.3e-6,'LHCb');label(ax,390,1.9e-6,'BaBar')
 ax.text(280,1.65e-9,'Thermal relic targets',fontsize=8.5,color='#975b57',rotation=20,zorder=8,ha='center')
 if zoom_box:ax.add_patch(Rectangle((15,1.2e-7),275-15,9e-5-1.2e-7,fill=False,edgecolor='#839097',lw=.65,linestyle=(0,(3,3)),zorder=6))
q=pd.read_csv(B/'derived/projected_contours.csv',float_precision='round_trip')
def zoom(ax,which,panel):
 style(ax,(15,275),(1.2e-7,9e-5),True);world(ax,True);g2(ax)
 col=C[which];values=q[f'projected_{which}_minimal']
 ax.plot(q.mass_MeV,values,color='white',lw=2.5,zorder=10)
 ax.plot(q.mass_MeV,values,color=col,lw=1.6,ls=(0,(5,1.8)),zorder=11)
 title='2015 + 2016 + 2021' if which=='three' else '2015 + 2016 + 2019 + 2021'
 ax.set_title(f'({panel})  {title}',fontsize=10.5,pad=7,loc='left')
 ax.text(.96,.11,'Full-exposure equivalent',transform=ax.transAxes,color=col,ha='right',fontsize=8.8,zorder=12)
 label(ax,23,1.8e-6,'NA48/2');label(ax,180,4.2e-6,'BaBar')
 if which=='four':ax.text(.96,.035,'2019: 2021-response proxy',transform=ax.transAxes,color='#526369',ha='right',fontsize=7.5,zorder=12)
 return ax
handles=[Line2D([],[],color=C['h15'],lw=1.3,label='HPS 2015 published'),Line2D([],[],color=C['h16'],lw=1.3,label='HPS 2016 published'),Line2D([],[],color=C['mu'],lw=1.2,ls=(0,(5,2,1.5,2)),label=r'$(g-2)_\mu$ central (WP25)'),Line2D([],[],color=C['e'],lw=1.2,ls=(0,(2.5,1.8)),label=r'$(g-2)_e$ central (Rb20 + Fan23)')]
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',pad_inches=.06)
 plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.9));fig.subplots_adjust(left=.105,right=.987,bottom=.135,top=.855);overview(ax)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.997),ncol=2,frameon=False,fontsize=8.8,columnspacing=1.4,handlelength=2.6)
save(fig,'figure2_clean_overview')
fig=plt.figure(figsize=(7.2,7.0));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1],left=.09,right=.986,bottom=.07,top=.89,hspace=.36,wspace=.25)
ax=fig.add_subplot(gs[0,:]);overview(ax,True);zoom(fig.add_subplot(gs[1,0]),'three','a');zoom(fig.add_subplot(gs[1,1]),'four','b')
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.999),ncol=2,frameon=False,fontsize=8.8,columnspacing=1.4,handlelength=2.6)
save(fig,'figure2_overview_and_projections')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.9));fig.subplots_adjust(left=.08,right=.987,bottom=.15,top=.77,wspace=.25)
zoom(axs[0],'three','a');zoom(axs[1],'four','b');fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1.005),ncol=2,frameon=False,fontsize=8.4,columnspacing=1.4,handlelength=2.6)
save(fig,'figure2_projection_panels')
(B/'derived/contour_display_protocol.json').write_text(json.dumps(dict(contours=audit,excluded_inputs={'apex_2019_physics_run.csv':'The source notebook explicitly identifies this as a projection, not a published exclusion.','dark_light.csv':'Prospective reach.'},geometry='Exact union of source polygons in log mass and log epsilon squared. Source vertex order and invalid-row breaks retained; no numerical smoothing of limits.',g2={'formula':'Delta a_l = alpha epsilon^2 F_V(m/m_l)/(2 pi), F_V=integral_0^1 2x(1-x)^2/((1-x)^2+(m/m_l)^2*x) dx','muon_delta_a':3.8e-10,'muon_source':'Aliberti et al., arXiv:2505.21476v3, WP25: 38(63)e-11. The central curve does not imply a nonzero favored band.','electron_delta_a':3.4e-13,'electron_source':'Rounded positive central residual for Fan et al. 2023 with Morel et al. 2020 rubidium alpha. Independent Cs alpha gives a negative residual and is not mapped into a positive-vector favored line.','bands':'Only central-value loci, not confidence regions or posterior medians.'}),indent=2)+'\n')
print('Generated three vector figures and PNG previews.')
