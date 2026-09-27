"""Pinned GP display derivatives and one conditional sideband diagnostic. No release writes."""
import os,sys,time,json,csv,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']='/tmp/hps_rcmeet_0923_mpl'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
SRC=ROOT/'study_results/v5p0p5_analysis_note_20260916';V595=ROOT/'study_results/v5p9p5_null_bias_20260922/residual_diagnostic'
sys.path.insert(0,str(SRC/'scripts'))
from common import DATA,predict,kernel_state,sigma,moving_context,OneSignalProfile
import numpy as np
from scipy.special import xlogy
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42})
BLUE='#1b77a4';GOLD='#c99522';RED='#bb3636';PURPLE='#8054a4';GREEN='#2b9a50';BLACK='#1b1b1b'
paths=[SRC/'inputs'/f'spectrum_{y}.npz' for y in ['2015','2016','2021']]+[SRC/'scripts'/f for f in ['common.py','parent_core.py','limit_solver.py']]+[SRC/'inputs/scopes.json',V595/'inputs/null_2021.npz']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
def save(fig,name):
 for ext in ('png','pdf'):fig.savefig(OUT/'assets'/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def csvout(name,rows):
 with (OUT/'data'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cp(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
protocol={'input_hashes':hashes,'hyperparameter_optimizations':0,'new_poisson_spectra':0,'raw_observed_data_unchanged':True,'thread_limit':1}
# Slide11 one fixed-state profile at preserved 78 MeV; all lower curves share GP subtraction.
p=moving_context('2021',78.);mod=OneSignalProfile(p['b'],p['L'],p['S'][:,0]);fit=mod.fit(p['n']);null=mod.fit(p['n'],0.)
x=DATA['2021']['x'][p['mask']]*1000;bw=np.diff(DATA['2021']['edges'])[p['mask']]*1000
n,b,bprof,S=p['n'],p['b'],fit['bfit'],p['S'][:,0]*fit['A'];cdiag=np.diag(p['C']);unit=1000.
rows=[dict(mass_bin_MeV=float(xi),width_MeV=float(w),observed=float(ni),GP_mean=float(bi),profiled_background=float(bp),fitted_signal=float(si),data_minus_GP=float(ni-bi),profiled_background_minus_GP=float(bp-bi),total_fit_minus_GP=float(bp+si-bi),GP_sd=float(np.sqrt(ci))) for xi,w,ni,bi,bp,si,ci in zip(x,bw,n,b,bprof,S,cdiag)]
csvout('slide11_profile78_curves.csv',rows)
fig,axs=plt.subplots(2,1,figsize=(5.25,6.1),sharex=True,gridspec_kw={'height_ratios':[1,1.05]})
err=np.sqrt(n)/bw/unit
axs[0].errorbar(x,n/bw/unit,yerr=err,fmt='o',c=BLACK,ms=3.4,lw=.7)
axs[0].plot(x,b/bw/unit,c=GOLD,lw=1.7,ls=':');axs[0].plot(x,bprof/bw/unit,c=BLUE,lw=1.7,ls='--');axs[0].plot(x,(bprof+S)/bw/unit,c=RED,lw=1.7)
axs[0].set(title='2021 native 10% · 78 MeV',ylabel=r'Events / MeV [$10^3$]')
axs[1].fill_between(x,-np.sqrt(cdiag)/bw/unit,np.sqrt(cdiag)/bw/unit,color=BLUE,alpha=.13)
axs[1].errorbar(x,(n-b)/bw/unit,yerr=err,fmt='o',c=BLACK,ms=3.4,lw=.7)
axs[1].plot(x,(bprof-b)/bw/unit,c=BLUE,lw=1.8,ls='--');axs[1].plot(x,(bprof+S-b)/bw/unit,c=RED,lw=1.8);axs[1].plot(x,S/bw/unit,c=PURPLE,lw=1.8,ls=':');axs[1].axhline(0,c=GOLD,lw=1.4,ls=':')
axs[1].set(xlabel=r'$m_{ee}$ [MeV]',ylabel=r'Difference from GP mean'+'\n'+r'[10$^3$ events / MeV]')
fig.tight_layout(h_pad=.5);save(fig,'slide11_profile78_clear')
# Wide alternative carries its own unambiguous colored legend.
fig,ax=plt.subplots(figsize=(9.8,1.05));ax.axis('off')
legitems=[Line2D([],[],c=BLACK,marker='o',ls='',label=r'Data: $n-b_{GP}$'),Line2D([],[],c=BLUE,ls='--',lw=2,label=r'Background: $b_{prof}-b_{GP}$'),Line2D([],[],c=RED,lw=2,label=r'Total fit: $b_{prof}+Aw-b_{GP}$'),Line2D([],[],c=PURPLE,ls=':',lw=2,label=r'Signal: $Aw$'),Patch(color=BLUE,alpha=.15,label='Band: GP constraint width')]
ax.legend(handles=legitems,ncol=3,loc='center',frameon=False,fontsize=12);save(fig,'slide11_lower_legend')
protocol['slide11']={'mass_MeV':78,'source':'v5.0.5 fixed-state profile replay','signal_profile_fits':2,'kernel_constant':p['const'],'kernel_length_scale':p['ls'],'raw_signed_root':float(np.sign(fit['A'])*np.sqrt(2*(null['nll']-fit['nll']))),'epsilon2_hat':float(fit['A']*1e-8),'lower_panel_definition':'Every curve is relative to the same unprofiled sideband GP mean; standalone signal Aw equals (GP+signal)-GP. Band is GP constraint width, not post-fit covariance.'}
print('slide11 ready',flush=True)
# Slide13: one leave-window-out prediction for every search bin, using nearest integer hypothesis.
fig,axs=plt.subplots(2,3,figsize=(13.2,5.5),sharex='col',gridspec_kw={'height_ratios':[1.65,1]});allrows=[];pdata={};t0=time.monotonic()
for k,(year,lo,hi) in enumerate([('2015',19,100),('2016',39,180),('2021',50,250)]):
 d=DATA[year];xx=d['x']*1000;idx=np.flatnonzero((xx>=lo)&(xx<=hi));anchors=np.clip(np.floor(xx[idx]+.5).astype(int),lo,hi);bpred=np.empty(len(idx));cvar=np.empty(len(idx));maxselfdistance=0
 for anchor in np.unique(anchors):
  use=np.flatnonzero(anchors==anchor);di=idx[use];mask=np.abs(d['x']-anchor/1000)<=2.25*sigma(year,float(anchor));assert np.all(mask[di])
  const,ls=kernel_state(year,float(anchor));bb,CC=predict(d['x'],d['n'],mask,const,ls,query=d['x'][di]);bpred[use]=bb;cvar[use]=np.diag(CC)
 n=d['n'][idx];width=np.diff(d['edges'])[idx]*1000;res=(n-bpred)/np.sqrt(bpred+cvar)
 a=axs[0,k];a.errorbar(xx[idx],n/width,yerr=np.sqrt(n)/width,fmt='.',c=BLACK,ms=2.3,lw=.35,alpha=.6);a.plot(xx[idx],bpred/width,c=GOLD,lw=2.2,zorder=4);a.set(title=year+(' native 10%' if year=='2021' else ' full'),yscale='log',xlim=(lo,hi))
 a=axs[1,k];a.axhspan(-2,2,color=GREEN,alpha=.18);a.axhline(0,c='.5',lw=.7);a.plot(xx[idx],res,c=BLACK,marker='.',ls='-',lw=.55,ms=2.4);a.set(xlabel=r'$m_{ee}$ [MeV]',xlim=(lo,hi));ylim=max(3.5,np.ceil(np.max(np.abs(res))));a.set_ylim(-ylim,ylim)
 for j,i in enumerate(idx):allrows.append(dict(dataset=year,bin_center_MeV=float(xx[i]),bin_width_MeV=float(width[j]),excluded_hypothesis_MeV=int(anchors[j]),observed_counts=float(n[j]),heldout_GP_mean=float(bpred[j]),GP_variance=float(cvar[j]),standardized_residual=float(res[j]),denominator=float(np.sqrt(bpred[j]+cvar[j]))))
 pdata[year]={'bins':len(idx),'hypotheses_used':int(len(np.unique(anchors))),'search_MeV':[lo,hi],'max_abs_standardized_residual':float(np.max(abs(res))),'fraction_outside_two':float(np.mean(abs(res)>2)),'every_query_bin_was_withheld':True,'kernel_policy':'archived nearest integer state;2015 freezes90MeV endpoint above90'}
axs[0,0].set_ylabel('Events / MeV');axs[1,0].set_ylabel('Standardized residual')
fig.legend(handles=[Line2D([],[],c=BLACK,marker='.',ls='',label='Observed counts'),Line2D([],[],c=GOLD,lw=2,label='Held-out GP prediction'),Patch(color=GREEN,alpha=.18,label='±2 reference band')],loc='upper center',ncol=3,frameon=False,fontsize=12)
fig.tight_layout(rect=(0,0,1,.93),h_pad=.35,w_pad=1.8);save(fig,'slide13_all_datasets_heldout')
csvout('slide13_heldout_bins.csv',allrows);protocol['slide13']={'datasets':pdata,'runtime_s':time.monotonic()-t0,'method':'Each display bin uses prediction from nearest integer-mass hypothesis excluding its ±2.25σ window. All displayed bins are held out of their own prediction; line is assembled from overlapping local fits, not one global fit.','residual':'(n_i-b_i)/sqrt(b_i+C_GP,ii)','caution':'Residuals across mass correlated. Green±2 is descriptive, not a simultaneous confidence/coverage band. Other observed bins train each prediction; hyperparameters fixed at archived data-derived states.'}
print('slide13 ready',round(time.monotonic()-t0,2),'s',flush=True)
# Slide14 sideband-only fit metric: evaluate only SEARCH bins outside chosen blind window.
d=DATA['2021'];search=(d['x']*1000>=50)&(d['x']*1000<=250)
def sideband_metric(counts,anchor,ret=False):
 mask=np.abs(d['x']-anchor/1000)<=2.25*sigma('2021',float(anchor));side=search&~mask;const,ls=kernel_state('2021',float(anchor));b,C=predict(d['x'],counts,mask,const,ls,query=d['x'][side]);n=np.asarray(counts)[side];dev=float(2*np.sum(xlogy(n,n/b)-n+b));ks=float(np.max(np.abs(np.cumsum(n)/n.sum()-np.cumsum(b)/b.sum())))
 result={'anchor_MeV':anchor,'N_side_bins':int(side.sum()),'poisson_deviance':dev,'D_per_side_bin':dev/side.sum(),'cumulative_shape_distance':ks}
 return (result,side,b) if ret else result
anchorrows=[sideband_metric(d['n'],m) for m in [65,78,120]];csvout('slide14_observed_sideband_metrics.csv',anchorrows)
nulls=np.load(V595/'inputs/null_2021.npz');assert np.array_equal(nulls['observed'],d['n']);assert np.array_equal(nulls['edges_GeV'],d['edges'])
t0=time.monotonic();toyrows=[]
for it,counts in enumerate(nulls['counts']):
 row=sideband_metric(counts,78);row['toy_id']=it;toyrows.append(row)
 if (it+1)%64==0:print('slide14 toys',it+1,'elapsed',round(time.monotonic()-t0,2),flush=True)
csvout('slide14_conditional_sideband_toys78.csv',toyrows)
obs=anchorrows[1];vals=np.array([r['D_per_side_bin'] for r in toyrows]);ksvals=np.array([r['cumulative_shape_distance'] for r in toyrows]);k=int(np.sum(vals>=obs['D_per_side_bin']));kks=int(np.sum(ksvals>=obs['cumulative_shape_distance']));N=len(vals)
summary={'anchor_metrics':anchorrows,'primary_anchor_MeV':78,'anchor_is_postselected':True,'N_toys':N,'D_conditional_tail':{'exceedances':k,'N':N,'p_k_over_N':k/N,'p_addone':(k+1)/(N+1),'cp95':cp(k,N),'null_mean':float(vals.mean()),'null_median':float(np.median(vals)),'null_central90':list(map(float,np.quantile(vals,[.05,.95])))},'KS_shape_conditional_tail':{'exceedances':kks,'N':N,'p_addone':(kks+1)/(N+1),'cp95':cp(kks,N),'meaning':'Binned cumulative count-shape distance on the disconnected sideband set, with both totals normalized separately; this is not a distribution-free KS p-value.'},'runtime_s':time.monotonic()-t0}
fig,axs=plt.subplots(1,2,figsize=(11.8,4.0),gridspec_kw={'width_ratios':[1,1.7]})
labels=[str(r['anchor_MeV']) for r in anchorrows];values=[r['D_per_side_bin'] for r in anchorrows]
axs[0].bar(labels,values,color=[BLUE,RED,BLUE],alpha=.85,width=.55);axs[0].set(xlabel='Excluded-window center [MeV]',ylabel=r'$D_{side}/N_{side}$',title='Only sidebands enter the metric',ylim=(0,max(1.1,max(values)*1.22)))
for i,val in enumerate(values):axs[0].text(i,val+.02,f'{val:.3f}',ha='center',fontsize=12)
axs[1].hist(vals,bins=20,color=BLUE,alpha=.7,label='256 fixed-source refits');axs[1].axvline(obs['D_per_side_bin'],c=RED,lw=2.4,label='Observed, 78 MeV mask');axs[1].set(xlabel=r'$D_{side}/N_{side}$',ylabel='Toy refits',title='Conditional reference at 78 MeV')
axs[1].legend(loc='upper right',frameon=False,fontsize=10);axs[1].text(.97,.72,f'{k}/{N} at least as large\n95% tail interval: [{cp(k,N)[0]:.3f}, {cp(k,N)[1]:.3f}]',transform=axs[1].transAxes,ha='right',va='top',fontsize=11)
fig.tight_layout(w_pad=2.2);save(fig,'slide14_sideband_deviance')
(OUT/'data/slide14_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
protocol['slide14']={'definition':'D_side=2 sum_i[n_i log(n_i/bhat_i)-n_i+bhat_i], evaluated over search-region bins outside ±2.25sigma; GP trains on all support bins outside same mask. N_side is bin count, not effective dof.','support_bin_edges_MeV':list(map(float,d['edges'][[0,-1]]*1000)),'search_MeV':[50,250],'observed_anchor_masses_MeV':[65,78,120],'conditional_refit_anchor_MeV':78,'toy_GP_replays':256,'source':'Same frozen observed-data-derived nominal GP source and 256 saved full Poisson spectra as v5.9.5.','scope':'This is an in-sample sideband-fit diagnostic, not held-out-window validation, unconditional model GOF, or resonance significance. The 78 MeV anchor was selected in prior observed scans. Tail interval accounts only finite toy counts.','summary':summary}
assert all(sha(ROOT/k)==v for k,v in hashes.items())
protocol['all_parent_inputs_and_code_unchanged']=True
(OUT/'provenance/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for name,tex in [('slide13_residual_equation',r'$R_i=\dfrac{n_i-\widehat b_i}{\sqrt{\widehat b_i+C_{{\rm GP},ii}}}$'),('slide14_deviance_equation',r'$D_{\rm side}=2\sum_{i\in{\rm side}}\!\left[n_i\log\!\left(\dfrac{n_i}{\widehat b_i}\right)-n_i+\widehat b_i\right]$')]:
 fig,ax=plt.subplots(figsize=(10,.7) if name.startswith('slide14') else (7.5,.9));ax.axis('off');ax.text(.5,.5,tex,ha='center',va='center',fontsize=24);save(fig,name)
print(json.dumps({'slide14':summary,'slide13':pdata},indent=2))
