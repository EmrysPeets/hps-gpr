#!/usr/bin/env python3
"""Curate v5 display derivatives from saved arrays; never fit, sample or alter sources."""
from pathlib import Path
import os,json,hashlib,shutil,importlib.util,sys
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/v5-figure-mpl')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];S=ROOT/'study_results';F=HERE/'figures';E=HERE/'editorial'
RECORDS=[];LINE_QA={};BOUND_MARKER_QA=[]
BLUE,RED,GOLD,GREEN,PURPLE='#236d9b','#b64038','#b58c28','#3b8052','#8258a0'
TAIL=S/'v4p9p12_targeted_tail_refinement_20260905';BAND=S/'v4p9p12_combination_expected_bands_20260904';EX=S/'v4p9p16_presentation_extractions_20260906'
LABEL={'individual_2015_full':'2015 full','individual_2016_full':'2016 full','individual_2021_10pct':'2021 10%','pair_2015_2016':'2015 + 2016','pair_2015_2021':'2015 + 2021','pair_2016_2021':'2016 + 2021','all_2015_2016_2021':'All three'}
IND=list(LABEL)[:3];COM=list(LABEL)[3:]
SEG=[(19,38,IND[0],'2015'),(39,49,COM[0],'15+16'),(50,90,COM[-1],'All three'),(91,180,COM[2],'2016 + 2021'),(181,250,IND[2],'2021')]
ROOT_DEF='Signed local significance r = sign(Ahat) sqrt(2[NLL(A=0)-NLL(Ahat)]), using the signed auxiliary profile fit. On the positive branch, nominal local discovery Z=max(r,0) and p0=1-Phi(Z); negative r is a deficit diagnostic, not a discovery significance.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def style():plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42,'savefig.dpi':160})
def record(name,sources,caption,edits,request=''):
 RECORDS.append(dict(name=name,request=request,caption=caption,edits=edits,sources=[dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),sha256=sha(p)) for p in sources],outputs=[dict(path=str((F/(name+'.'+ext)).relative_to(ROOT)),sha256=sha(F/(name+'.'+ext))) for ext in ('pdf','png') if (F/(name+'.'+ext)).exists()]))
def save(fig,name,sources,caption,edits,request=''):
 for ext in ('pdf','png'):fig.savefig(F/(name+'.'+ext),bbox_inches='tight',pad_inches=.12)
 plt.close(fig);record(name,sources,caption,edits,request)
def copy_asset(src,name,caption='',request=''):
 for ext in ('pdf','png'):
  p=src.with_suffix('.'+ext)
  if p.exists():shutil.copy2(p,F/(name+'.'+ext))
 record(name,[src],caption,'Unmodified vector source copy.',request)
def bound_marker(ax,x,upper,color,key):
 xx=np.asarray(x,dtype=float);uu=np.asarray(upper,dtype=float)
 artist=ax.scatter(xx,uu,facecolors='none',edgecolors=color,marker='v',s=24,linewidths=.95,zorder=6)
 offsets=np.asarray(artist.get_offsets(),dtype=float)
 assert np.array_equal(offsets[:,0],xx) and np.array_equal(offsets[:,1],uu)
 BOUND_MARKER_QA.append(dict(figure_context=key,n_bounds=len(uu),marker='open downward triangle',exact_x_match=True,exact_y_equals_upper_bound=True,maximum_absolute_y_error=0.0))
 return artist


def band(ax,d):
 x=d.mass_MeV.to_numpy();ax.fill_between(x,d.expected_q025,d.expected_q975,color='#F6D66A',alpha=.65);ax.fill_between(x,d.expected_q16,d.expected_q84,color='#69C779',alpha=.7);ax.plot(x,d.expected_median,'k--',lw=1.3);ax.plot(x,d.eps2_observed,'k-',lw=1.5);ax.set_yscale('log')
def limits():
 source=S/'v4p9p12_expanded_snapshot_20260905/derived/expected_band_summary_dimuon_300toys.csv';d=pd.read_csv(source)
 ts=S/'v4p9p12_expanded_snapshot_20260905/derived/final_total_search_window_dimuon_300toys.csv';t=pd.read_csv(ts)
 fig=plt.figure(figsize=(10.2,7.0));g=fig.add_gridspec(3,1,height_ratios=(.18,2.8,1.35),left=.105,right=.985,bottom=.09,top=.83,hspace=.12)
 st=fig.add_subplot(g[0]);a=fig.add_subplot(g[1],sharex=st);p=fig.add_subplot(g[2],sharex=st)
 for (lo,hi,key,label),col in zip(SEG,['#dce9f2','#dae7de','#e9e0ef','#f2e6d7','#e7e7e7']):
  q=t[t.selected_scope_key.eq(key)];st.add_patch(Rectangle((lo-.5,0),hi-lo+1,1,color=col));st.text((lo+hi)/2,.5,label,ha='center',va='center',fontsize=8)
 band(a,t);p.plot(t.mass_MeV,t.p0_local_asymptotic,color=BLUE,lw=1.4)
 assert np.array_equal(a.lines[-1].get_xdata(),np.arange(19,251))
 assert np.array_equal(a.lines[-1].get_ydata(),t.eps2_observed.to_numpy())
 LINE_QA['total_observed_line']=dict(points=232,continuous_single_polyline=True,exact_x_and_y_match=True,source_sha256=sha(ts),source=str(ts.relative_to(ROOT)),connected_membership_transitions=[[38,39],[49,50],[90,91],[180,181]])
 t[['mass_MeV','eps2_observed','expected_median','p0_local_asymptotic','selected_scope_key']].to_csv(F/'v5_total_limits_plot_data.csv',index=False,float_format='%.17g')
 for ax in (a,p):
  for v in (38.5,49.5,90.5,180.5):ax.axvline(v,color='.55',ls=':',lw=.7)
 st.set(xlim=(18.5,250.5),ylim=(0,1));st.axis('off');a.tick_params(labelbottom=False);a.set_ylabel(r'90% CL$_s$ upper limit on $\epsilon^2$');p.set(yscale='log',ylim=(.001,1),ylabel='Nominal local $p_0$',xlabel='Mass hypothesis [MeV]');p.axhline(.05,color='.5',ls=':',lw=.8)
 fig.suptitle('Observed limit and expected bands over the full search range',y=.98,fontsize=13,fontweight='semibold')
 fig.legend(handles=[Patch(color='#F6D66A',label='Central 95% expected'),Patch(color='#69C779',label='Central 68% expected'),Line2D([],[],color='k',ls='--',label='Expected median'),Line2D([],[],color='k',label='Observed 90% CL$_s$')],loc='upper center',bbox_to_anchor=(.5,.945),ncol=4,frameon=False)
 save(fig,'v5_total_limits_with_local_pvalues',[ts], 'Mass-by-mass maximal available combination. Bands are central pointwise quantiles of the original 300 conditional background-only toys per mass, with frozen GP states. The lower panel adds the unchanged nominal local asymptotic p0, with no look-elsewhere correction. Active datasets change at the indicated boundaries; 2021 uses its released 10% sample. The 2016 numerical/source qualification remains applicable.','Rebuilt Figure 1 from exact saved quantiles with asymptotic p0 beneath; adjacent observed points are connected across all active-dataset transitions; no changed fits or new toys.','v4p9p12 tail Figure 1')
 fig,ax=plt.subplots(figsize=(9.4,4.9))
 for key,col in zip(IND,[BLUE,GOLD,GREEN]):
  q=d[d.scope_key.eq(key)];ax.plot(q.mass_MeV,q.eps2_observed,label=LABEL[key],color=col,lw=1.7)
 ax.plot(t.mass_MeV,t.eps2_observed,label='Combined: all available datasets',color='black',lw=1.8)
 assert np.array_equal(ax.lines[-1].get_xdata(),np.arange(19,251)) and np.array_equal(ax.lines[-1].get_ydata(),t.eps2_observed.to_numpy())
 LINE_QA['overlay_combined_line']=dict(points=232,continuous_single_polyline=True,exact_x_and_y_match=True,identical_to_total_observed_line=True,source_sha256=sha(ts))
 ax.set(xlabel='Mass hypothesis [MeV]',ylabel=r'Observed 90% CL$_s$ upper limit on $\epsilon^2$',yscale='log');ax.legend(frameon=False,ncol=2);ax.set_title('Observed coupling limits',loc='left')
 save(fig,'v5_observed_coupling_overlay',[source,ts],'Observed 90% CLs coupling limits from the three individual released samples and the maximal-available-dataset common-coupling combination across 19--250 MeV. Its active membership is 2015 at 19--38, 2015+2016 at 39--49, all three at 50--90, 2016+2021 at 91--180 and 2021 at 181--250 MeV. This is the Figure 1 combined curve, not a minimum over individual limits. The inherited dimuon correction is applied once above threshold; 2016 retains its qualification.','New requested four-curve observed-only overlay; the union curve connects all 232 adjacent masses including active-dataset transitions.','Observed coupling overlay')
 for scopes,name,shape in [(IND,'v5_individual_expected_bands',(3,1)),(COM,'v5_combination_expected_bands',(2,2)),([COM[-1]],'v5_all_three_expected_bands',(1,1))]:
  fig,axs=plt.subplots(*shape,figsize=(8.5,9) if shape==(3,1) else ((10.5,7.3) if shape==(2,2) else (8.4,4.5)))
  for ax,key in zip(np.atleast_1d(axs).ravel(),scopes):q=d[d.scope_key.eq(key)];band(ax,q);ax.set_title(LABEL[key],loc='left');ax.set(xlabel='Mass hypothesis [MeV]',ylabel=r'90% CL$_s$ limit on $\epsilon^2$')
  fig.tight_layout();save(fig,name,[source],'Observed limits and unchanged central 68% and 95% pointwise expected bands from 300 conditional toys per mass. The GP state remains frozen in these ensembles; 2016 remains qualified.','Rebuilt from saved quantiles; removed optimized-support wording from titles.','v4p9p12 tail Figures 2--4')
 # Keep unresolved empirical tails censored, with no invented continuation.
 ps=TAIL/'derived/pvalue_diagnostics_refined.csv';z=pd.read_csv(ps)
 for scopes,name,shape in [(IND,'v5_individual_tail_probabilities',(3,1)),(COM,'v5_combination_tail_probabilities',(2,2))]:
  fig,axs=plt.subplots(*shape,figsize=(8.8,9.2) if shape==(3,1) else (10.6,7.6))
  for ax,key in zip(np.atleast_1d(axs).ravel(),scopes):
   q=z[z.scope_key.eq(key)].sort_values('mass_MeV')
   for col,color,label,ls in [('p_strong',BLUE,'Stronger-limit tail','-'),('p_weak',GOLD,'Weaker-limit tail','-'),('p_two',PURPLE,'Two-sided limit-tail diagnostic','-'),('p0_local_asymptotic','black','Nominal local asymptotic $p_0$','--')]:
    ax.plot(q.mass_MeV,q[col].where(q[col]>0),color=color,label=label,lw=1,ls=ls)
    zero=q[col].eq(0)
    if zero.any() and col!='p0_local_asymptotic':
     u=q.loc[zero,col+'_zero_upper95'];bound_marker(ax,q.loc[zero,'mass_MeV'],u,color,name+':'+key+':'+col)
   ax.set(title=LABEL[key],xlabel='Mass hypothesis [MeV]',ylabel='Fixed-mass probability',yscale='log',ylim=(1e-5,1.2));ax.axhline(.05,color='.5',ls=':',lw=.6)
  h,l=np.atleast_1d(axs).ravel()[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.91 if shape==(2,2) else .94))
  save(fig,name,[ps],'Saved targeted tail refinement: 300 toys at unrefined points and independent 3000 or 10000 fresh toys at selected coordinates. Open downward triangles lie exactly at the one-sided 95% Monte Carlo upper bounds for zero counts (twice that bound for the two-sided diagnostic). Curves join only finite point estimates. Nominal local asymptotic p0 is displayed separately. No new tail samples were generated, so unresolved probabilities remain unresolved.','Clean titles; retained honest zero-count censoring. Additional tail toys deferred.','v4p9p12 tail Figures 5--6')

def curated_copies():
 h=S/'harvard_writing_sample_final_combinations_20260902/figures'
 for name in ('individual_final_results','combined_final_results','final_asymptotic_pvalues','all_three_peak_extraction'):
  if (h/(name+'.pdf')).exists():copy_asset(h/(name+'.pdf'),name,'Historical Harvard writing sample figure; current result figures supersede its presentation where specified.')
 for name,num in [('nominal_mass_resolutions',14),('resolution_width_limits',15)]:copy_asset(TAIL/'figures'/f'{name}.pdf','v5_'+name,'Frozen nominal resolution input.' if num==14 else 'Signal-template width variation with frozen background states; this is a conditional response study, not a resolution-nuisance profile.',f'v4p9p12 tail Figure {num}')
 bg=S/'v4p9p13_background_profiling_20260905'
 copy_asset(bg/'comparison/figures/observed_limits_2021_comparison.pdf','v5_profile_background_comparison','Same 2021 10% data compared under the Gaussian profile, direct log-GP profile and fixed GP mean. Ratios use the released limit.','v4p9p13 Figure 1')
 copy_asset(bg/'injections/figures/injected_bias.pdf','v5_profile_injected_bias','Mean pull for injections at 2 sigma_ref and 5 sigma_ref; both methods use the same physical signal yield, defined using the profiled background-only reference uncertainty. 500 spectra per cell. Bars are Monte Carlo standard errors of the mean pull. The retrained 71 MeV 5 sigma_ref means are -2.33 fixed and -0.81 profiled.','v4p9p13 Figure 9')


def load_module(path):
 spec=importlib.util.spec_from_file_location('v5_'+path.stem,path);mod=importlib.util.module_from_spec(spec);exec(compile(path.read_text().replace('str(p.relative_to(HERE))','str(p)'),str(path),'exec'),mod.__dict__);return mod

def clean_text(fig):
 removed=[]
 for t in list(fig.texts):
  if t is not fig._suptitle:removed.append(t.get_text());t.remove()
 for ax in fig.axes:
  for t in list(ax.texts):
   if t.get_bbox_patch() is not None:removed.append(t.get_text());t.remove()
 return removed

def clean_profile_figures():
 global RECORDS
 bg=S/'v4p9p13_background_profiling_20260905'
 for kind,script,data,name in [('comparison',bg/'comparison/make_figures.py',S/'background_profile_comparison_20260905/derived/observed_limits.csv','v5_profile_background_comparison'),('injection',bg/'injections/make_figures.py',bg/'injections/derived/extraction_summary.csv','v5_profile_injected_bias')]:
  old=next(r for r in RECORDS if r['name']==name);RECORDS=[r for r in RECORDS if r['name']!=name];m=load_module(script)
  def cleaned(fig,stem):
   moved=clean_text(fig)
   # Retain source's Type-3 glyph workaround for assembled-note PDF rendering.
   with plt.rc_context({'pdf.fonttype':3}):save(fig,name,[script,data],old['caption']+' '+' '.join(moved),'Regenerated from saved ledger; moved nonlegend figure text into caption.',old['request'])
  m.save=cleaned
  if kind=='comparison':m.limits(pd.read_csv(data))
  else:m.injected_bias(pd.read_csv(data))
 style()


def calibrations():
 base=S/'v4p9p13_calibration_20260905'
 script=base/'make_figures.py';mod=load_module(script);mod.OUT=F
 cap='Conditional calibrated and asymptotic observed limits using the released data and saved paired background treatments. Shading is approximate 95% Monte Carlo uncertainty on the calibrated endpoint, not expected bands. Open circles have limited MC precision; triangles indicate unresolved finite endpoints. The calibration is conditional on the two generating truth scenarios.'
 def save_cal(fig,name):
  removed=clean_text(fig)
  # Preserve source main() manifest generation, then rename below.
  for ext in ('pdf','png'):fig.savefig(F/(name+'.'+ext),bbox_inches='tight')
  plt.close(fig)
  for ext in ('pdf','png'):shutil.copy2(F/(name+'.'+ext),F/('v5_calibration_'+name+'.'+ext))
  record('v5_calibration_'+name,[script,base/'summary/observed_calibrated_limits.csv'],cap+' Figure annotations moved to caption: '+' '.join(removed),'Regenerated source plot from saved ledger; removed nonlegend figure text.')
 mod.save=save_cal;mod.main()
 for name in ['limits_2015','limits_2016','limits_2021','limits_combined','local_pvalues']:
  for ext in ('pdf','png'):(F/(name+'.'+ext)).unlink()
 (F/'limit_plot_provenance.json').unlink()
 vscript=base/'make_validation_figures.py';v=load_module(vscript);data=pd.read_csv(base/'summary/validation_summary.csv')
 def save_v(fig,output,name):
  removed=clean_text(fig);save(fig,'v5_calibration_'+name,[vscript,base/'summary/validation_summary.csv'],' '.join(removed),'Regenerated exact saved validation cells; moved explanatory text into caption.')
 v.save=save_v;v.exclusion_figure(data,F);v.bias_figure(data,F)
 tscript=base/'make_truth_figure.py';tm=load_module(tscript)
 def save_t(fig,output):
  removed=clean_text(fig);save(fig,'v5_calibration_truth_dependence',[tscript,base/'summary/truth_specific_limits.csv'],' '.join(removed),'Regenerated exact truth-specific endpoints; removed nonlegend figure text.');return []
 tm.save=save_t;tm.draw(pd.read_csv(base/'summary/truth_specific_limits.csv'),F)
 style()

def resolution_response():
 src=S/'v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/derived/width_scan_all_points.csv';d=pd.read_csv(src)
 fig,axes=plt.subplots(1,3,figsize=(11,3.9))
 for ax,(lo,hi),title in zip(axes,[(60,85),(50,250),(50,250)],['Peak and deficit region','Positive branch','Negative branch']):
  for col,scale in zip(plt.get_cmap('viridis')(np.linspace(.05,.95,5)),[.8,.9,1.,1.1,1.2]):
   q=d[d.width_scale.eq(scale)&d.mass_MeV.between(lo,hi)];r=q.signed_r.to_numpy();r=np.maximum(r,0) if title=='Positive branch' else np.minimum(r,0) if title=='Negative branch' else r
   ax.plot(q.mass_MeV,r,color='black' if scale==1 else col,lw=1.2,label=f'{scale:.1f} nominal width')
  ax.axhline(0,color='.5',lw=.6);ax.set(title=title,xlabel='Mass hypothesis [MeV]',ylabel='Nominal local significance' if title=='Positive branch' else 'Signed local significance')
 fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False);fig.tight_layout(rect=(0,0,1,.9))
 save(fig,'v5_resolution_width_significance',[src],'2021 signal-template widths from 0.8 to 1.2 times the frozen nominal resolution, leaving the background state fixed. '+ROOT_DEF,'Changed ordinate labels; regenerated exact saved width scans.','v4p9p12 tail Figure 16')

def correlations():
 sources=[S/'v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz']+[S/f'v4p9p15_global_2016_2021_20260906/global_fast/{year}/analysis/covariance.npz' for year in ('2016','2021')]
 fig,axes=plt.subplots(1,3,figsize=(11.4,3.8),layout='constrained')
 for ax,year,src in zip(axes,['2015','2016','2021'],sources):
  a=np.load(src);m=a['masses_MeV'];im=ax.imshow(a['profiled_K'],origin='lower',extent=[m[0]-.5,m[-1]+.5]*2,vmin=-1,vmax=1,cmap='RdBu_r');ax.set(title=year+(' 10%' if year=='2021' else ' full'),xlabel='Mass hypothesis [MeV]',ylabel='Mass hypothesis [MeV]')
 for year,src in zip(['2015','2016','2021'],sources):
  data=np.load(src);m=data['masses_MeV'];ff,aa=plt.subplots(figsize=(5.5,4.7));ii=aa.imshow(data['profiled_K'],origin='lower',extent=[m[0]-.5,m[-1]+.5]*2,vmin=-1,vmax=1,cmap='RdBu_r');aa.set(title=year+(' 10%' if year=='2021' else ' full')+': profiled response',xlabel='Mass hypothesis [MeV]',ylabel='Mass hypothesis [MeV]');ff.colorbar(ii,ax=aa,label='Correlation of local fit fluctuations');save(ff,'v5_correlations_'+year,[src],'Profiled-response correlations under the archived generating background. Positive correlation denotes common-direction fluctuations; negative correlation denotes opposite-direction fluctuations. This matrix measures statistical dependence and does not assign an echo probability.','Selected profiled matrix only; updated axis/colorbar language.')
 fig.colorbar(im,ax=axes,shrink=.8,label='Correlation of local fit fluctuations')
 save(fig,'v5_profiled_correlations_all_datasets',sources,'Profiled-response correlation matrices under the archived generating background for each dataset. Red entries show same-direction fluctuations and blue entries opposite-direction fluctuations. These dimensionless correlations describe statistical dependence, not a significance of an induced echo and not a correlation of physical signal production.','Selected only profiled matrices; unified labels and color scale.','Correlation plots for all datasets; v4p9p15 Figures 5--6')
 # Correlation-conditioned response in local-standard-deviation units; not a p-value.
 fig,axes=plt.subplots(3,1,figsize=(8.8,7.8))
 for ax,year,src,anchors in zip(axes,['2015','2016','2021'],sources,[[51,21],[90,117],[78,65]]):
  a=np.load(src);m=a['masses_MeV'];K=a['profiled_K']
  for mass,color in zip(anchors,[BLUE,RED]):
   j=int(np.where(m==mass)[0][0]);ax.plot(m,3*K[:,j],color=color,label=f'Condition on +3 at {mass} MeV')
  ax.axhline(0,color='.5',lw=.7);ax.set(title=year+(' 10%' if year=='2021' else ' full'),xlabel='Mass hypothesis [MeV]',ylabel='Conditional mean shift\n[local standard deviations]');ax.legend(frameon=False,ncol=2)
 fig.tight_layout()
 save(fig,'v5_correlation_induced_fluctuation_scales',sources,'Illustrative Gaussian conditional mean of standardized fluctuations: E[z(m) | z(m0)=3] = 3 K(m,m0), using the archived profiled correlation matrices. Anchor masses are the displayed individual candidates. This expresses the correlated fluctuation scale in nominal standard-deviation units; it is neither a signal-injection response nor a measured echo significance. The explicit 2021 injection response is shown separately.','New analytically derived correlation illustration; no fits or random samples.','Induced oscillation scale for all datasets')


def step(ax,edges,y,**kw):
 return ax.plot((np.asarray(edges[:-1])+np.asarray(edges[1:]))/2,y,**kw)[0]

def extractions():
 arrays=EX/'derived/fit_arrays.npz';summary=EX/'derived/fit_summary.csv';closure=EX/'derived/fit_closure.json'
 A=np.load(arrays);D=pd.read_csv(summary,dtype={'dataset':str});C={v['fit_id']:v for v in json.loads(closure.read_text())['checks']}
 def native(fid,key):
  prefix=fid+'__'+key+'__';return {k:A[prefix+k] for k in ['edges','mask','observed','gp_mean','profiled_background','signal','total','fit_covariance']}
 def sum_parts(fid):
  parts=[native(fid,k) for k in D[D.fit_id.eq(fid)].dataset];sets=[set(np.round(p['edges'],7)) for p in parts];edges=np.array(sorted(set.intersection(*sets)))
  m=C[fid]['mass_MeV'];sig=max(D[D.fit_id.eq(fid)].sigma_MeV);edges=edges[(edges>=m-3.6*sig)&(edges<=m+3.6*sig)]
  out={k:np.zeros(len(edges)-1) for k in ['observed','gp_mean','profiled_background','signal','total']};allmask=np.ones(len(edges)-1,dtype=bool);cov=np.zeros((len(edges)-1,len(edges)-1))
  for part in parts:
   lo=part['edges'][:-1];hi=part['edges'][1:];W=np.array([((lo>=a-1e-6)&(hi<=b+1e-6)).astype(float) for a,b in zip(edges[:-1],edges[1:])]);mask=part['mask'];valid=(W[:,~mask].sum(axis=1)==0)&(W.sum(axis=1)>0);allmask &= valid
   for k in out:out[k]+=W@np.nan_to_num(part[k],nan=0.)
   V=W[:,mask];cov+=V@part['fit_covariance']@V.T
  for k in ['profiled_background','total']:out[k][~allmask]=np.nan
  return dict(**out,edges=edges,mask=allmask,fit_covariance=cov[np.ix_(allmask,allmask)])
 def panel(top,bot,fid,key,title):
  m=C[fid]['mass_MeV'];p=sum_parts(fid) if key=='sum' else native(fid,key);e=p['edges'];x=(e[:-1]+e[1:])/2;mask=p['mask'];w=np.diff(e);sig=max(D[D.fit_id.eq(fid)].sigma_MeV) if key=='sum' else D[D.fit_id.eq(fid)&D.dataset.eq(key)].sigma_MeV.iloc[0]
  select=(x>=m-3.5*sig)&(x<=m+3.5*sig);idx=np.flatnonzero(select);lo,hi=idx[0],idx[-1]+1
  # Exactly the stored native bins: no fitted interpolation or fake data bins.
  n=p['observed'];g=p['gp_mean'];b=p['profiled_background'];tot=p['total'];signal=p['signal'];sc=w*1000
  for ax,values in [(top,n),(bot,n-g)]:ax.errorbar(x[select],(values/sc)[select],yerr=(np.sqrt(n)/sc)[select],fmt='o',ms=2.5,color='black',lw=.7,zorder=4)
  for vals,col,ls in [(g,GOLD,':'),(b,BLUE,'--'),(tot,RED,'-')]:step(top,e[lo:hi+1],(vals/sc)[lo:hi],color=col,lw=1.4,linestyle=ls)
  for vals,col,ls in [(b-g,BLUE,'--'),(tot-g,RED,'-'),(np.where(mask,signal,np.nan),PURPLE,':')]:step(bot,e[lo:hi+1],(vals/sc)[lo:hi],color=col,lw=1.3,linestyle=ls)
  sd=np.full(len(x),np.nan);sd[mask]=np.sqrt(np.diag(p['fit_covariance']))/sc[mask]
  bot.fill_between(x,-sd,sd,color='#a8cce2',alpha=.28,step='mid',lw=0);bot.axhline(0,color=GOLD,lw=.8,ls=':')
  for ax in (top,bot):
   ax.set_xlim(e[lo],e[hi]);ax.axvline(e[np.flatnonzero(mask)[0]],color='.55',ls=':',lw=.8);ax.axvline(e[np.flatnonzero(mask)[-1]+1],color='.55',ls=':',lw=.8)
  top.set_title(title,fontsize=11);top.tick_params(labelbottom=False);top.set_ylabel('Events / MeV [$10^3$]');bot.set_ylabel('Data or model minus GP mean\n[$10^3$ events / MeV]');bot.set_xlabel('$m_{ee}$ [MeV]')
 def draw(fids,keys,titles,name,caption):
  n=len(keys);fig=plt.figure(figsize=(max(8,3.7*n),5.8));grid=fig.add_gridspec(2,n,left=.1 if n<3 else .07,right=.99,top=.80,bottom=.10,hspace=.08,wspace=.35)
  for i,(fid,key,title) in enumerate(zip(fids,keys,titles)):
   top=fig.add_subplot(grid[0,i]);bot=fig.add_subplot(grid[1,i],sharex=top);panel(top,bot,fid,key,title)
  handles=[Line2D([],[],marker='o',color='black',ls='none',label='Observed data'),Line2D([],[],color=GOLD,ls=':',label='GP mean'),Line2D([],[],color=BLUE,ls='--',label='Profiled background'),Line2D([],[],color=RED,label='Background + signal'),Line2D([],[],color=PURPLE,ls=':',label='Fitted signal (residual panel)'),Patch(color='#a8cce2',alpha=.28,label='GP constraint width')]
  fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.96),fontsize=9)
  shared=' Native-bin density displays retain the stored bin edges and extend approximately 1.25 nominal resolution widths beyond the fitted-window boundary into each available sideband. The residual panel subtracts the same GP mean everywhere, and shows the profiled background displacement, total model displacement, and fitted signal separately. Profiled curves stop at the actual fit window (vertical dotted boundaries). Bars: counting error only. Blue shading: zero-centered GP constraint width; not fitted-background error or total residual uncertainty. Lines connect saved bin-averaged model predictions at native bin centers; no additional model evaluations, fitted interpolation or refit is introduced. '+ROOT_DEF
  save(fig,name,[arrays,summary,closure],caption+shared,'Rebuilt from saved native bins, joined model bin centers with straight lines, added sidebands, removed explanatory text boxes.','v4p9p16 Candidate Removal Figures 11--17')
 for m in [66,92,72]:
  fid=f'combined_m{m:03d}';keys=D[D.fit_id.eq(fid)].dataset.tolist()+['sum'];names=[k+(' 10%' if k=='2021' else ' full') if k!='sum' else 'Common-bin sum' for k in keys]
  draw([fid]*len(keys),keys,[f'{m} MeV: '+v for v in names],f'v5_extraction_combined_{m}',f'Common-amplitude fit at {m} MeV; signed local significance r={C[fid]["root"]:+.3f}. Each dataset uses its own resolution and yield conversion. The final panel sums only common whole bins and has no separate fit.')
 for year,masses in [('2015',[51,21]),('2016',[90,117]),('2021',[78,65])]:
  fids=[f'{year}_m{m:03d}' for m in masses];draw(fids,[year]*2,[f'{year}: {m} MeV' for m in masses],f'v5_extraction_{year}_peaks','Two leading separated observed individual excesses, selected from the completed scan. '+', '.join(f'{m} MeV: r={C[fid]["root"]:+.3f}' for m,fid in zip(masses,fids))+'.')
 fids=['2015_m019','2016_m102','2021_m071'];draw(fids,['2015','2016','2021'],['2015: 19 MeV','2016: 102 MeV','2021: 71 MeV'],'v5_extraction_individual_deficits','Deepest observed deficit in each individual scan. The 2015 mass is the search endpoint. Negative templates are auxiliary deficit diagnostics, not physical negative event rates.')
 # Source consistency graph regenerated with plain-language likelihood-loss heading.
 cons=EX/'derived/dataset_consistency.csv';q=pd.read_csv(cons,dtype={'dataset':str});fig,axes=plt.subplots(1,4,figsize=(12,4.7))
 for ax,m in zip(axes,[66,92,72,76]):
  fid=f'combined_m{m:03d}';dd=q[q.fit_id.eq(fid)];row=D[D.fit_id.eq(fid)].iloc[0]
  for j,z in enumerate(dd.itertuples()):ax.errorbar(z.individual_eps2_hat*1e6,j,xerr=z.individual_sigma_eps2*1e6,fmt='o',color={'2015':BLUE,'2016':GOLD,'2021':GREEN}[z.dataset],capsize=3)
  ax.axvspan((row.eps2_hat-row.sigma_eps2)*1e6,(row.eps2_hat+row.sigma_eps2)*1e6,color=RED,alpha=.13);ax.axvline(row.eps2_hat*1e6,color=RED,lw=1.3);ax.axvline(0,color='.5',ls=':',lw=.6);ax.set_yticks(range(len(dd)),dd.dataset.tolist());ax.invert_yaxis();ax.set_xlabel('Signed fitted amplitude [$10^{-6}$]');ax.set_title(f'{m} MeV\nShared-rate likelihood loss: {C[fid]["individual_common_deviance"]:.2f}',fontsize=10)
 fig.tight_layout();save(fig,'v5_dataset_amplitude_consistency',[cons,summary,closure],'Separate year amplitudes and local curvature standard errors, with the common-amplitude estimate and its curvature interval in red. Shared-rate likelihood loss means 2[NLL_common - sum(NLL_individual free)]. The number of additional amplitudes is one fewer than the number of contributing datasets. These selected masses provide descriptive compatibility checks; no calibrated or post-selection compatibility probability is assigned.','Replaced delta-D label by shared-rate likelihood loss; retained exact definition in caption.','v4p9p16 Candidate Removal Figure 18')


def echoes_and_removal():
 src=S/'v4p9p16_probability_echo_review_20260906/derived/echo_dense_scans.csv';e=pd.read_csv(src);d=e.pivot(index='mass_MeV',columns='lane',values='r');fig,axes=plt.subplots(3,1,figsize=(8.8,8.0))
 for lane,label,col,ls in [('observed','Observed 2021 10%','black','-'),('background','Smooth background',GOLD,':'),('inject_66','Positive 66 MeV injection',BLUE,'-'),('inject_78','Positive 78 MeV injection',RED,'-')]:axes[0].plot(d.index,d[lane],color=col,ls=ls,label=label,lw=1.3)
 for lane,label,col in [('inject_66','66 MeV injection',BLUE),('inject_78','78 MeV injection',RED)]:axes[1].plot(d.index,d[lane]-d.background,color=col,label=label,lw=1.3)
 for lane,label,col,ls in [('observed','Observed 2021 10%','black','-'),('background','Smooth background',GOLD,':'),('double_65_78','Positive 65 + 78 MeV injection',PURPLE,'-')]:axes[2].plot(d.index,d[lane],color=col,ls=ls,label=label,lw=1.3)
 for ax,title,yl in zip(axes,['A positive injected signal can create nearby peaks and deficits','Injection-induced change after subtracting the background response','Two positive injections can produce an intervening fitted deficit'],['Signed local significance','Change in signed local significance','Signed local significance']):
  ax.axhline(0,color='.5',lw=.7);ax.set(title=title,xlabel='Mass hypothesis [MeV]',ylabel=yl,xlim=(60,88));ax.legend(frameon=False,ncol=2,fontsize=8)
 fig.tight_layout();save(fig,'v5_signal_echo_dense_replay',[src],'Archived deterministic positive-signal injections in the 2021 moving-sideband fit. One injected peak induces both positive and negative fitted responses at neighboring hypotheses; two selected positive injections can make a dip while overshooting the observed peaks. This demonstrates a possible response mechanism, not the physical origin of the observed oscillations. The middle panel is a difference of two signed fit responses, not an independently calibrated echo probability. '+ROOT_DEF,'Regenerated saved response curves; replaced signed-root labels and removed annotations.','v4p9p16 Signal Extractions requested Figure 14 / echo mechanism')
 base=S/'v4p9p16_candidate_removal_20260906';holes=pd.read_csv(base/'derived/holes.csv');sources=[base/'derived/holes.csv'];fig,axes=plt.subplots(3,1,figsize=(8.9,7.4))
 colors={'original':'black','first_mean':BLUE,'second_mean':GOLD,'both_mean':GREEN}
 for ax,year in zip(axes,[2015,2016,2021]):
  src=base/f'derived/{year}/scans.csv';sources.append(src);w=pd.read_csv(src).pivot(index='mass_MeV',columns='lane',values='r');reps=w[[f'observed_both_rep{i:02d}' for i in range(10)]];ax.fill_between(w.index,reps.min(axis=1),reps.max(axis=1),color=GREEN,alpha=.17)
  for k,label in [('original','Original data'),('first_mean','Replace first peak region'),('second_mean','Replace second peak region'),('both_mean','Replace both regions')]:ax.plot(w.index,w['observed_'+k],color=colors[k],ls='--' if k=='second_mean' else '-',lw=1.2,label=label)
  for h in holes[holes.dataset.eq(year)].itertuples():ax.axvspan(h.low_MeV,h.high_MeV,color='.7',alpha=.16,lw=0)
  ax.axhline(0,color='.5',lw=.7);ax.set(title=str(year)+(' 10%' if year==2021 else ' full'),xlabel='Mass hypothesis [MeV]',ylabel='Signed local significance');ax.set_xlim(w.index.min(),w.index.max())
 fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.925));save(fig,'v5_observed_candidate_removal',sources,'Observed scans before and after replacing the preselected candidate regions with conditional GP predictions. Shaded vertical regions identify the replaced bins. The green envelope spans the ten paired conditional replacements and is not a confidence band. Changes at neighboring masses illustrate the coupling introduced by the moving background fit; they do not establish the cause of the original pattern. '+ROOT_DEF,'Regenerated exact completed intervention scans; clearer significance label with definition retained.','v4p9p16 Candidate Removal Figure 4')


def global_probabilities():
 sources=[S/'v4p9p14_interpretation_global_20260906/global/2015/analysis/pvalue_curves.csv']+[S/f'v4p9p15_global_2016_2021_20260906/global_fast/{y}/analysis/pvalue_curves.csv' for y in ('2016','2021')]+[S/'v4p9p16_combined_global_20260906/global/analysis/pvalue_curves.csv']
 for year,src in zip(['2015','2016','2021','combined'],sources):
  d=pd.read_csv(src);d=d[d.method.eq('profiled')].sort_values('mass_MeV');x=d.mass_MeV.to_numpy();fig,(a,b)=plt.subplots(2,1,figsize=(8.6,6.9),sharex=True,gridspec_kw={'height_ratios':[1,2]})
  a.plot(x,d.observed_r,color='black',lw=1.2,label='Observed signed local significance');a.plot(x,d.asimov_r,color=GOLD,lw=1.2,label='Stress-background mean response');a.axhline(0,color='.5',lw=.7);a.set_ylabel('Signed local significance');a.legend(frameon=False,ncol=2,fontsize=8)
  floor=1e-8
  for col,color,label,ls in [('p_asymptotic','.4','Nominal local asymptotic $p_0$','--'),('p_local_common_truth',BLUE,'Conditional local Gaussian tail','-')]:
   v=d[col].to_numpy();b.plot(x,np.where(v>=floor,v,np.nan),color=color,label=label,ls=ls,lw=1.2);cl=v<floor;b.scatter(x[cl],np.full(cl.sum(),floor),marker='v',color=color,s=18)
  for prefix,col,label,marker,n in [('p_global_gp',RED,'Conditional GP global tail','o',200000),('p_global_direct',GREEN,'Direct complete-scan check','s',1000)]:
   p=d[prefix].to_numpy();valid=p>0;zero=~valid;lo=d[prefix+'_low'].to_numpy();hi=d[prefix+'_high'].to_numpy()
   if prefix=='p_global_gp':b.plot(x,np.where(valid,p,np.nan),color=col,lw=1.2)
   b.errorbar(x[valid],p[valid],yerr=[p[valid]-lo[valid],hi[valid]-p[valid]],color=col,fmt=marker,ms=2,elinewidth=.5,alpha=.65,label=label)
   if zero.any():
    u=d[prefix+'_upper95'].to_numpy()[zero] if prefix+'_upper95' in d else np.full(zero.sum(),1-.05**(1/n));bound_marker(b,x[zero],u,col,'v5_global_probabilities_'+year+':'+prefix)
  b.set(yscale='log',ylim=(floor/1.7,1.6),ylabel='Probability under stated reference',xlabel='Mass hypothesis [MeV]');b.legend(frameon=False,ncol=2,fontsize=8)
  fig.suptitle(('Combined full search' if year=='combined' else year+(' 10%' if year=='2021' else ' full'))+': profiled local and global diagnostics',fontsize=12);fig.tight_layout(rect=(0,0,1,.96))
  save(fig,'v5_global_probabilities_'+year,[src],'Profiled-background local and global comparisons under the archived common stress background. The upper panel makes the observed fit and the reference offset visible. Lower curves distinguish nominal local asymptotic p0, the stress-centered local Gaussian approximation, and global tails estimated with 200000 GP fields and 1000 direct full-spectrum scans. Bars are central 95% Monte Carlo intervals; open downward triangles lie exactly at the one-sided 95% upper bounds for zero simulated tails. Filled downward triangles at 1e-8 mark analytic probabilities below the plotting range. Stress-centered tails assess that specified generating spectrum and are not adopted as physical discovery claims. '+ROOT_DEF,'Regenerated only profiled method; no added scans, no continuation of unresolved tails.','v4p9p14/v4p9p15 global-probability series')


def conditional_echo_response():
 sources=[S/'v4p9p14_interpretation_global_20260906/global/2015/analysis/covariance.npz']+[S/f'v4p9p15_global_2016_2021_20260906/global_fast/{y}/analysis/covariance.npz' for y in ('2016','2021')]
 rows=[];checks=[];fig,axes=plt.subplots(3,1,figsize=(8.8,7.3))
 for ax,year,src,m0 in zip(axes,['2015','2016','2021'],sources,[51,90,78]):
  a=np.load(src);mass=a['masses_MeV'];C=a['profiled_C'];K=a['profiled_K'];sd=np.sqrt(np.diag(C));j=int(np.flatnonzero(mass==m0)[0]);delta=K[:,j]*sd
  assert np.allclose(K,C/np.outer(sd,sd),rtol=1e-12,atol=1e-12)
  assert np.isclose(delta[j],sd[j],rtol=1e-12,atol=1e-12)
  q=pd.read_csv(src.parent/'pvalue_curves.csv');q=q[q.method.eq('profiled')].sort_values('mass_MeV');assert np.array_equal(q.mass_MeV,mass);assert np.allclose(q.response_sd,sd,rtol=1e-14,atol=1e-14)
  ax.plot(mass,delta,color=BLUE,lw=1.5);ax.axhline(0,color='.5',lw=.7);ax.axvline(m0,color=GOLD,ls=':',lw=1);ax.set(title=f'{year}'+(' 10%' if year=='2021' else ' full')+f': +1 standardized fluctuation at {m0} MeV',xlabel='Mass hypothesis [MeV]',ylabel='Conditional change in\nlocal significance');ax.set_xlim(mass[0],mass[-1])
  rows.extend(dict(dataset=year,mass_MeV=int(m),anchor_MeV=m0,conditioned_standardized_fluctuation=1.,correlation=float(k),response_sd=float(ss),conditional_r_shift=float(v)) for m,k,ss,v in zip(mass,K[:,j],sd,delta))
  checks.append(dict(dataset=year,anchor_MeV=m0,n_masses=len(mass),formula='delta_r(m)=K(m,m0)*sqrt(C(m,m))',correlation_covariance_consistent=True,anchor_shift_equals_response_sd=True,source_csv_response_sd_max_abs_difference=float(np.max(np.abs(q.response_sd.to_numpy()-sd)))))
 fig.tight_layout();pd.DataFrame(rows).to_csv(F/'v5_conditional_echo_response_plot_data.csv',index=False,float_format='%.17g');LINE_QA['conditional_echo_response']=checks
 save(fig,'v5_conditional_echo_response',sources,'Gaussian conditional mean response under each archived stress background. Let r(m) have mean a(m), covariance C, standard deviation s(m)=sqrt(C_mm), and correlation K; define z(m)=[r(m)-a(m)]/s(m). The imposed condition is z(m0)=+1 at the fixed displayed anchors 51, 90 and 78 MeV. The curve is E[r(m)-a(m) | z(m0)=1]=K(m,m0)s(m), expressed in units of the nominal signed local significance. It shows the correlated mean shift after removing the reference offset, with no signal injection and no probability assigned to an observed echo. It is distinct from the deterministic 2021 positive-signal injection test. The source response widths and covariance normalization are cross-checked numerically.','New deterministic conditional-response slice from saved covariance matrices; no refits or simulations.','Requested induced oscillation response for each dataset')
 src=S/'v4p9p16_combined_global_20260906/global/analysis/covariance.npz';a=np.load(src);m=a['masses_MeV'];fig,ax=plt.subplots(figsize=(6.3,5.4));im=ax.imshow(a['profiled_K'],origin='lower',extent=[m[0]-.5,m[-1]+.5]*2,vmin=-1,vmax=1,cmap='RdBu_r')
 for x in [38.5,49.5,90.5,180.5]:ax.axvline(x,color='.4',ls=':',lw=.7);ax.axhline(x,color='.4',ls=':',lw=.7)
 ax.set(title='Combined profiled search: shared-data correlations',xlabel='Mass hypothesis [MeV]',ylabel='Mass hypothesis [MeV]');fig.colorbar(im,ax=ax,label='Correlation of local fit fluctuations');fig.tight_layout();save(fig,'v5_correlations_combined',[src],'Profiled-response correlations across the 232-point union search. Dotted lines mark changes in active datasets. Shared datasets induce correlations across membership boundaries; regions with no common dataset have zero response covariance in the stated independent-dataset model. The color scale is a dimensionless correlation, not a significance or signal-production probability.','Regenerated profiled union correlation matrix with plain-language label.','v4p9p16 Signal Extractions Figure 14')


def original_reference_map():
 mapping={
 'v5_total_limits_with_local_pvalues':TAIL/'figures/final_total_search_window_expected_bands_300toys.pdf',
 'v5_observed_coupling_overlay':TAIL/'figures/final_total_search_window_expected_bands_300toys.pdf',
 'v5_all_three_expected_bands':TAIL/'figures/all_three_expected_bands_300toys.pdf',
 'v5_combination_expected_bands':TAIL/'figures/combination_expected_band_panels_300toys.pdf',
 'v5_individual_expected_bands':TAIL/'figures/individual_expected_band_panels_300toys.pdf',
 'v5_individual_tail_probabilities':TAIL/'figures/individual_pvalue_panels_refined.pdf',
 'v5_combination_tail_probabilities':TAIL/'figures/combination_pvalue_panels_refined.pdf',
 'v5_resolution_width_significance':TAIL/'figures/resolution_width_signed.pdf',
 'v5_signal_echo_dense_replay':S/'v4p9p16_probability_echo_review_20260906/figures/signal_echo_dense_replay.pdf',
 'v5_observed_candidate_removal':S/'v4p9p16_candidate_removal_20260906/figures/observed_candidate_removal.pdf',
 'v5_correlations_combined':S/'v4p9p16_combined_global_20260906/figures/combined_correlations.pdf'}
 for r in RECORDS:
  name=r['name']
  if name.startswith('v5_extraction_') or name=='v5_dataset_amplitude_consistency':mapping[name]=EX/'figures'/(name[3:]+'.pdf')
  if name.startswith('v5_calibration_'):mapping[name]=S/'v4p9p13_calibration_20260905/figures'/(name[len('v5_calibration_'):]+'.pdf')
  if name in mapping:
   path=mapping[name];r['original_reference_figure']=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
  if name=='v5_signal_echo_dense_replay':r['request']='v4p9p16 Candidate Removal Figure 4'
  if name=='v5_observed_candidate_removal':r['request']='v4p9p16 Candidate Removal Figure 5 (related supporting study)'
 return {k:dict(path=str(v.relative_to(ROOT)),sha256=sha(v)) for k,v in mapping.items()}


def appendix_display_cleanup():
 base=S/'v4p9p16_candidate_removal_20260906';holes=pd.read_csv(base/'derived/holes.csv');fig,axes=plt.subplots(3,2,figsize=(10,8.4));sources=[base/'derived/holes.csv']
 for row,year in enumerate([2015,2016,2021]):
  src=base/f'derived/{year}/scans.csv';sources.append(src);d=pd.read_csv(src).pivot(index='mass_MeV',columns='lane',values='r')
  for ax,source in zip(axes[row],['observed','reference']):
   for suffix,col,ls,label in [('original','black','-','Original'),('both_mean',GREEN,'-','GP replacement'),('both_poly_mean',PURPLE,'-','Polynomial replacement'),('both_wide_mean',RED,'--','Wider GP replacement')]:ax.plot(d.index,d[source+'_'+suffix],color=col,ls=ls,lw=1.1,label=label)
   for h in holes[holes.dataset.eq(year)].itertuples():ax.axvspan(h.low_MeV,h.high_MeV,color='.7',alpha=.15,lw=0)
   ax.axhline(0,color='.5',lw=.7);ax.set(title=str(year)+' '+('observed data' if source=='observed' else 'reference spectrum'),xlabel='Mass hypothesis [MeV]',ylabel='Signed local significance');ax.set_xlim(d.index.min(),d.index.max())
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False);fig.tight_layout(rect=(0,0,1,.95));save(fig,'v5_replacement_model_comparison',sources,'Both candidate regions are replaced in observed counts (left) or the archived reference spectrum (right), using each spectrum separately to learn its replacements. Curves compare the original, primary GP, polynomial and wider-hole GP replacements. Shading marks the primary holes, which are narrower than the wider-hole comparison. All outcomes are retained, including unsuccessful polynomial interpolation. The ordinate is the same nominal signed local fit mapping; the reference response is not an event-count spectrum or newly calibrated significance. '+ROOT_DEF,'Regenerated saved intervention scans, clean labels and no numeric boxes.','Related candidate-removal appendix Figure 6')
 base=base/'traditional';src=base/'qa/paper_display_groups.npz';A=np.load(src);sums=base/'derived/fit_summary.csv';D=pd.read_csv(sums,dtype={'dataset':str})
 for year,masses in [('2015',[51,21]),('2016',[90,117]),('2021',[78,65])]:
  fig,axes=plt.subplots(2,2,figsize=(9,6.5),sharex='col');numbers=[]
  for j,m in enumerate(masses):
   fid=f'{year}_m{m:03d}';prefix=fid+'__';g={k:A[prefix+k] for k in ['edges_MeV','counts_per_MeV','count_error_per_MeV','background_free_per_MeV','background_null_per_MeV','total_free_per_MeV','signal_counts_per_MeV']};x=(g['edges_MeV'][:-1]+g['edges_MeV'][1:])/2;n=g['counts_per_MeV']/1000;err=g['count_error_per_MeV']/1000;b=g['background_null_per_MeV']/1000
   axes[0,j].errorbar(x,n,yerr=err,fmt='o',ms=2.7,lw=.7,color='black');axes[1,j].errorbar(x,n-b,yerr=err,fmt='o',ms=2.7,lw=.7,color='black')
   for key,col,ls in [('total_free_per_MeV',BLUE,'-'),('background_free_per_MeV',GOLD,':'),('background_null_per_MeV','.5','--')]:axes[0,j].plot(x,g[key]/1000,color=col,ls=ls,lw=1.2)
   axes[1,j].plot(x,g['total_free_per_MeV']/1000-b,color=BLUE,lw=1.3);axes[1,j].plot(x,g['signal_counts_per_MeV']/1000,color=RED,ls='--',lw=1.3);axes[1,j].axhline(0,color='.5',lw=.7)
   axes[0,j].set(title=f'{year}: {m} MeV',ylabel='Events / MeV [$10^3$]');axes[1,j].set(xlabel='$m_{ee}$ [MeV]',ylabel='Data or model minus null\n[$10^3$ events / MeV]')
   r=D[D.dataset.eq(year)&D.mass_MeV.eq(m)&D.variant.eq('baseline')].iloc[0];numbers.append(f'{m} MeV: polynomial degree {int(r.degree)}, total width {r.total_width_sigma:g} resolution widths, r={r.root:+.3f}, deviance/dof={r.deviance:.2f}/{int(r.ndof)}')
  fig.legend(handles=[Line2D([],[],marker='o',color='black',ls='none',label='Observed bins'),Line2D([],[],color=BLUE,label='Fitted total'),Line2D([],[],color=GOLD,ls=':',label='Profiled background'),Line2D([],[],color='.5',ls='--',label='Null background'),Line2D([],[],color=RED,ls='--',label='Signal (residual panel)')],loc='upper center',ncol=3,frameon=False)
  fig.tight_layout(rect=(0,0,1,.9));save(fig,f'v5_traditional_{year}_display',[src,sums],'Conventional polynomial-background fits at GP-selected masses. '+ '; '.join(numbers)+'. Counts and predictions use the saved whole-bin grouping, divided by each actual bin width; the last partial group is retained. Counting errors only. Residuals subtract the null-background fit. Lines connect stored bin-averaged model predictions. The quoted local fit statistics do not incorporate the selection of these masses from the GP scan.','Regenerated stored whole-bin display arrays; moved fit statistics and all explanatory text into caption.','Related traditional-search appendix')
 src=S/'v4p9p16_deficit_extension_20260906/make_report.py';code=src.read_text();i=code.index('    figures=[]');j=code.index('    note=HERE',i);code=code[:i]+'    v5_emit(fig)\n    return\n'+code[j:]
 code=code.replace("glob.scatter(d.mass_MeV[zeros],d.loc[zeros,prefix+'_upper95'],color=color,marker='v',s=17,zorder=5)", "v5_bound(glob,d.mass_MeV[zeros],d.loc[zeros,prefix+'_upper95'],color,'v5_combined_deficit_scan:'+prefix)").replace("glob.scatter(zero.mass_MeV,zero.direct_upper95,color=GREEN,marker='v',s=25,zorder=6)", "v5_bound(glob,zero.mass_MeV,zero.direct_upper95,GREEN,'v5_combined_deficit_scan:direct')").replace("marker='v',color='.3',ls='none',ms=4","marker='v',color='.3',markerfacecolor='none',ls='none',ms=4")
 code=code.replace('Observed signed root','Observed signed local significance').replace("r'Signed root $r$'","'Signed local significance'").replace('Raw-root reference:', 'Nominal Gaussian reference:')
 spec=importlib.util.spec_from_file_location('v5_deficit_source',src);mod=importlib.util.module_from_spec(spec);exec(compile(code,str(src),'exec'),mod.__dict__)
 def emit(fig):
  removed=clean_text(fig);save(fig,'v5_combined_deficit_scan',[src,src.parent/'analysis/deficit_curves.csv',src.parent/'analysis/summary.json'],'Combined profiled deficit scan. Top: observed signed local fit response and the stress-background offset. Middle: nominal Gaussian and stress-centered local deficit tails; both assign one to nonnegative raw fits. Bottom: two separate global deficit orderings. Filled downward triangles at the local floor indicate analytic values below 1e-8; open downward global triangles lie exactly at the one-sided 95% zero-count Monte Carlo bounds. Direct-scan bars are central 95% intervals. These direction-specific tests were investigated after the excess scan; no extra direction-trials correction is included. '+ ' '.join(removed)+' '+ROOT_DEF,'Regenerated only the source plotting block from frozen summaries; no note writes, fits or toys.','Related joint-deficit appendix')
 mod.v5_emit=emit;mod.v5_bound=bound_marker;mod.main();style()


def dataset_mass_distributions():
 import uproot,yaml
 base=S/'v4p9p12_final_dataset_combinations_20260902';card=base/'inputs/analysis_card.yaml';summary=base/'derived/run_summary.json';cfg=yaml.safe_load(card.read_text());frozen=json.loads(summary.read_text())['immutable_histogram_inputs'];records=[];checks=[];sources=[card,summary];fig,axes=plt.subplots(3,1,figsize=(9.1,8.1))
 for ax,year,col in zip(axes,['2015','2016','2021'],[BLUE,GOLD,GREEN]):
  path=Path(cfg['path_'+year]);key=cfg['hist_'+year];actual=sha(path);assert actual==frozen[year]['sha256'] and key==frozen[year]['histogram'];sources.append(path)
  with uproot.open(path) as f:counts,edges=f[key].to_numpy(flow=False)
  assert np.all(np.isfinite(counts)) and np.all(counts>=0) and np.all(np.diff(edges)>0)
  edges=edges*1000;centers=(edges[:-1]+edges[1:])/2;width=np.diff(edges);density=counts/width;support=np.array(cfg['data_range_'+year])*1000;search=np.array(cfg['range_'+year])*1000
  expected_support={'2015':[14,135],'2016':[30,210],'2021':[36,300]}[year];expected_search={'2015':[19,90],'2016':[39,180],'2021':[50,250]}[year];assert np.allclose(support,expected_support) and np.allclose(search,expected_search)
  ax.axvspan(*support,color='.8',alpha=.24,lw=0,label='GP training support')
  ax.stairs(np.where(counts>0,density,np.nan),edges,baseline=None,color=col,lw=1.05,label='Released invariant-mass counts')
  for i,x in enumerate(search):ax.axvline(x,color='.25',lw=1.0,ls='--',label='Search boundaries' if i==0 else None)
  lo=max(0,support[0]-.08*np.ptp(support));hi=min(edges[-1],support[1]+.06*np.ptp(support));sel=(centers>=lo)&(centers<=hi)&(density>0)
  ax.set(xlim=(lo,hi),yscale='log',ylabel='Events / MeV',xlabel='$m_{ee}$ [MeV]',title=year+(' 10%' if year=='2021' else ' full'));ax.set_ylim(max(1,float(density[sel].min())*.65),float(density[sel].max())*1.6)
  checks.append(dict(dataset=year,input_path=str(path),histogram_key=key,actual_sha256=actual,frozen_sha256=frozen[year]['sha256'],frozen_hash_match=True,native_bins=len(counts),native_width_MeV=float(np.median(width)),histogram_total_counts=float(counts.sum()),support_MeV=support.tolist(),search_MeV=search.tolist(),density_integral_matches_counts=bool(np.allclose(density*width,counts,rtol=1e-14,atol=1e-9))))
  records.extend(dict(dataset=year,native_bin=i,low_MeV=float(a),high_MeV=float(b),counts=float(n),events_per_MeV=float(v),inside_support=bool(support[0]<=c<=support[1]),inside_search=bool(search[0]<=c<=search[1])) for i,(a,b,n,v,c) in enumerate(zip(edges[:-1],edges[1:],counts,density,centers)))
 handles=[Patch(facecolor='.8',alpha=.24,label='GP training support'),Line2D([],[],color='black',ls='--',label='Search boundaries'),Line2D([],[],color=BLUE,label='Released invariant-mass counts')];fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.95))
 datafile=F/'v5_dataset_mass_distributions_plot_data.csv';pd.DataFrame(records).to_csv(datafile,index=False,float_format='%.17g');LINE_QA['dataset_mass_distributions']=checks
 save(fig,'v5_dataset_mass_distributions_log',sources,'Released invariant-mass histograms for full 2015, full 2016 and the released 2021 10% sample. Each ROOT input SHA256 and histogram key matches the frozen v4.9.12 run ledger before plotting. Curves retain every original histogram bin (0.05, 0.05 and 0.125 MeV, respectively), displayed as counts divided by the native bin width on logarithmic axes. Gray shading shows GP training supports 14--135, 30--210 and 36--300 MeV; dashed boundaries show searches 19--90, 39--180 and 50--250 MeV. Native input binning here precedes the analysis rebinning; no exposure scaling, fitting or new-data access is introduced.','Regenerated only from SHA-verified frozen released ROOT histograms; corrected 2021 support shading to36--300MeV.','Current dataset overview replacing historical40--300MeV 2021 support shading')
 RECORDS[-1]['plot_data']=dict(path=str(datafile.relative_to(ROOT)),sha256=sha(datafile),rows=len(records));(F/'v5_dataset_mass_distributions_provenance.json').write_text(json.dumps(dict(checked_against=str(summary.relative_to(ROOT)),summary_sha256=sha(summary),inputs=checks,plot_data_sha256=sha(datafile),new_data_opened=False,new_fits=0),indent=2)+'\n')


def write_inventory():
 refs=original_reference_map();reference_digest=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest()
 (E/'figure_provenance.json').write_text(json.dumps(dict(source_commit='0c1d4692b',new_toys=0,new_fits=0,statistics_definition=ROOT_DEF,connected_line_qa=LINE_QA,zero_count_marker_qa=BOUND_MARKER_QA,reference_map=refs,reference_map_sha256=reference_digest,figures=RECORDS),indent=2)+'\n')
 lines=['# Version 5 figure inventory','','All sources remain frozen. New figures use saved arrays only.','',ROOT_DEF,'','## Connected observed-line QA',json.dumps(LINE_QA,indent=2),'']
 for r in RECORDS:lines.extend([f"## {r['name']}",f"Requested: {r['request']}",f"Change: {r['edits']}",f"Caption: {r['caption']}",*['Source: `'+s['path']+'` SHA256 `'+s['sha256']+'`' for s in r['sources']],''])
 (E/'figure_inventory.md').write_text('\n'.join(lines))

if __name__=='__main__':
 F.mkdir(parents=True,exist_ok=True);E.mkdir(exist_ok=True);style();curated_copies();clean_profile_figures();limits();calibrations();resolution_response();correlations();extractions();echoes_and_removal();global_probabilities();conditional_echo_response();appendix_display_cleanup();dataset_mass_distributions();write_inventory()
