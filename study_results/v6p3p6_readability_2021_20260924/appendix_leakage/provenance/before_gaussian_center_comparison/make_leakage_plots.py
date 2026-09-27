#!/usr/bin/env python3
"""Window/leakage comparison and paired signal-in-training removal controls.
Reads the released v6.3.7 study without changing it. No new toy draws or limits.
"""
from pathlib import Path
import os,sys,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
P=B.parent/'appendix_v637'
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v637-leakage-followup-mpl')
sys.path.insert(0,str(P/'scripts'))
import run_study as S
import numpy as np,pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
MASSES=list(range(80,241,20));POLICIES=['gaussian_baseline','gaussian_starter','direct_starter','common_starter','morph_starter']
BLUE='#245c91';ORANGE='#ad5536';GREEN='#2d865f';GRAY='#777777'
SOURCE={'gp_mean':'GP mean background source','functional':'Functional form background source'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def core(m):
 return S.T.center_width(m,method='morph',omit=m if 80<m<240 else None)

def geometry(m):
 c,s=core(m);sm=S.C.sigma('2021',m)*1000;gc=m-3.2243308692909953-2.213992811446465*np.log(m/150.)
 x=S.D['x']*1000;edges=S.D['edges']*1000
 old=(x>=gc-2.25*sm)&(x<=gc+2.25*sm);new=(x>=c-4*s)&(x<=c+3*s)
 mc=S.T.categories(m,edges,method='direct');gauss=np.r_[ndtr((edges[0]-gc)/sm),np.diff(ndtr((edges-gc)/sm)),1-ndtr((edges[-1]-gc)/sm)]
 def width(mask):
  idx=np.flatnonzero(mask);return edges[idx[-1]+1]-edges[idx[0]]
 row=dict(mass_MeV=m,historical_center_MeV=gc,core_center_MeV=c,core_sigma_MeV=s,sigma_m_MeV=sm,core_parameters='neighbor prediction with sample omitted' if 80<m<240 else 'direct fitted endpoint',
  lower_edge_sigma_m=-4*s/sm,upper_edge_sigma_m=3*s/sm,historical_continuous_width_MeV=4.5*sm,starter_continuous_width_MeV=7*s,
  continuous_width_increase_percent=100*(7*s/(4.5*sm)-1),binned_width_increase_percent=100*(width(new)/width(old)-1),historical_bins=int(old.sum()),starter_bins=int(new.sum()),
  MC_outside_spectrum_fraction=float(mc[0]+mc[-1]),Gaussian_outside_spectrum_fraction=float(gauss[0]+gauss[-1]))
 for label,mask in [('historical',old),('starter',new)]:
  row[f'MC_{label}_training_fraction']=float(mc[1:-1][~mask].sum());row[f'Gaussian_{label}_training_fraction']=float(gauss[1:-1][~mask].sum())
  row[f'MC_{label}_all_outside_window_fraction']=float(1-mc[1:-1][mask].sum())
  row[f'MC_over_Gaussian_{label}_leakage_ratio']=row[f'MC_{label}_training_fraction']/row[f'Gaussian_{label}_training_fraction']
 row['MC_leakage_reduction_percentage_points']=100*(row['MC_historical_training_fraction']-row['MC_starter_training_fraction'])
 return row,old,new,mc,gauss

class Context:
 predict=S.Context.predict
 fit_counts=S.Context.fit_counts
 def __init__(self,m,policy):
  row,old,new,mc,gauss=geometry(m);self.fit=old if policy=='gaussian_baseline' else new;self.guard=self.fit.copy()
  self.categories=mc
  method={'direct_starter':'direct','common_starter':'common','morph_starter':'morph'}.get(policy)
  self.probability=gauss[1:-1] if method is None else S.T.probabilities(m,S.D['edges']*1000,method=method,omit=m if method!='direct' and 80<m<240 else None)
  const,ls=S.C.kernel_state('2021',m);xt=S.D['x'][~self.guard];xq=S.D['x'][self.fit]
  self.K=S.C.kernel(xt,xt,const,ls);self.Kqt=S.C.kernel(xq,xt,const,ls);self.Kqq=S.C.kernel(xq,xq,const,ls)
  assert not np.any(self.fit&~self.guard)

def checked(ctx,counts):
 r=ctx.fit_counts(counts);assert r['fit_valid'],r;return r

def run_controls():
 original=pd.read_csv(P/'results/evaluation_rows.csv',float_precision='round_trip')
 rows=[];as_rows=[];replays=[]
 for source in SOURCE:
  for m in MASSES:
   ctxs={p:Context(m,p) for p in POLICIES};bg=S.TRUTHS[source];A=3*float(S.REF[str(m)]['s0']);signal=A*next(iter(ctxs.values())).categories[1:-1]
   for policy,ctx in ctxs.items():
    full=bg+signal;clean=full.copy();clean[~ctx.guard]=bg[~ctx.guard]
    assert np.array_equal(full[ctx.fit],clean[ctx.fit])
    r0=checked(ctx,bg);rf=checked(ctx,full);rc=checked(ctx,clean)
    as_rows.append(dict(source=source,mass_MeV=m,policy=policy,A_expected=A,background_Ahat=r0['Ahat'],full_Ahat=rf['Ahat'],clean_training_Ahat=rc['Ahat'],
     full_response=(rf['Ahat']-r0['Ahat'])/A,clean_response=(rc['Ahat']-r0['Ahat'])/A,leakage_loss_fraction=(rc['Ahat']-rf['Ahat'])/A,
     response_scaling_penalty_percent=100*((rc['Ahat']-r0['Ahat'])/(rf['Ahat']-r0['Ahat'])-1)))
   if m not in S.MASSES:continue
   for first in range(0,100,10):
    path=P/f'results/checkpoints/evaluation_{source}_m{m:03d}_t{first:03d}.npz'
    meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['draws_sha256']
    data=np.load(path);strengths=list(data['strengths']);iz=strengths.index(3);draws=data['signal_categories'].reshape(10,len(strengths),-1)
    for j,toy in enumerate(data['toys']):
     background=data['backgrounds'][j];signal=draws[j,iz,1:-1];full=background+signal
     q=original[(original.source==source)&(original.mass_MeV==m)&(original.toy==toy)]
     for policy,ctx in ctxs.items():
      ref=q[(q.policy==policy)&(q.z==3)].iloc[0];ref0=q[(q.policy==policy)&(q.z==0)].iloc[0]
      assert S.ahash(full)==ref.counts_hash
      clean=full.copy();clean[~ctx.guard]=background[~ctx.guard]
      assert np.array_equal(clean[ctx.fit],full[ctx.fit]);rc=checked(ctx,clean)
      if toy==0:
       rf=checked(ctx,full);assert abs(rf['Ahat']-ref.Ahat)<1e-8 and abs(rf['sigma']-ref.sigma)<1e-8
       replays.append(dict(source=source,mass=m,policy=policy,yield_difference=rf['Ahat']-ref.Ahat))
      rows.append(dict(source=source,mass_MeV=m,policy=policy,toy=int(toy),A_expected=A,background_Ahat=ref0.Ahat,full_Ahat=ref.Ahat,clean_training_Ahat=rc['Ahat'],
       full_response=(ref.Ahat-ref0.Ahat)/A,clean_response=(rc['Ahat']-ref0.Ahat)/A,leakage_loss_fraction=(rc['Ahat']-ref.Ahat)/A,
       actual_signal_removed_from_training=int(signal[~ctx.guard].sum()),full_sigma=ref.sigma,clean_sigma=rc['sigma']))
   print('Paired controls complete',source,m,flush=True)
 df=pd.DataFrame(rows);df.to_csv(B/'paired_training_removal_rows.csv',index=False,float_format='%.17g')
 pd.DataFrame(as_rows).to_csv(B/'mean_count_training_removal.csv',index=False,float_format='%.17g')
 summaries=[]
 for (source,m,policy),q in df.groupby(['source','mass_MeV','policy']):
  q=q.sort_values('toy');n=len(q);assert n==100
  inds=np.random.default_rng(np.random.SeedSequence([925637,1 if source=='gp_mean' else 2])).integers(0,n,(2000,n))
  full=q.full_response.to_numpy();clean=q.clean_response.to_numpy();loss=clean-full
  penalty=100*(clean.mean()/full.mean()-1);boot=100*(clean[inds].mean(axis=1)/full[inds].mean(axis=1)-1);lo,hi=np.quantile(boot,[.025,.975])
  summaries.append(dict(source=source,mass_MeV=m,policy=policy,toys=n,full_response=full.mean(),full_response_SE=full.std(ddof=1)/np.sqrt(n),
    clean_response=clean.mean(),clean_response_SE=clean.std(ddof=1)/np.sqrt(n),leakage_loss_fraction=loss.mean(),leakage_loss_SE=loss.std(ddof=1)/np.sqrt(n),
    response_scaling_penalty_percent=penalty,penalty95_low=lo,penalty95_high=hi,bootstrap_resamples=2000))
 summary=pd.DataFrame(summaries);summary.to_csv(B/'paired_training_removal_summary.csv',index=False,float_format='%.17g')
 return summary,pd.DataFrame(as_rows),replays

def save(fig,name,footer,rect):
 fig.tight_layout(rect=rect)
 fig.text(.035,.014,footer,fontsize=8.1,va='bottom',linespacing=1.4)
 for ext in ['png','pdf']:fig.savefig(B/f'{name}.{ext}',dpi=190)
 plt.close(fig)

def plots(g,q,a):
 fig,axes=plt.subplots(2,2,figsize=(11,8.2));x=g.mass_MeV
 ax=axes[0,0];ax.plot(x,g.lower_edge_sigma_m,'o-',color=BLUE,label='Starter lower edge');ax.plot(x,g.upper_edge_sigma_m,'s-',color=GREEN,label='Starter upper edge')
 for edge in [-2.25,2.25]:ax.axhline(edge,color=GRAY,ls='--',label='Historical edges: +/-2.25' if edge>0 else None)
 ax.set(ylabel=r'Edge offset from its window center [$\sigma_m$]',title='A. Convert the requested [-4u, +3u] interval');ax.legend(loc='center right')
 ax=axes[0,1];ax.plot(x,g.continuous_width_increase_percent,'o-',color=BLUE,label='Continuous interval');ax.plot(x,g.binned_width_increase_percent,'s--',color=GRAY,label='Actual excluded bins')
 ax.set(ylabel='Width increase over historical window [%]',title='B. How much wider is the blind window?');ax.legend()
 ax=axes[1,0]
 for prefix,color,label in [('historical',ORANGE,'Historical window'),('starter',BLUE,'Starter window')]:
  ax.plot(x,100*g[f'MC_{prefix}_training_fraction'],'o-',color=color,label='Signal MC: '+label)
  ax.plot(x,100*g[f'Gaussian_{prefix}_training_fraction'],'s--',color=color,label='Gaussian: '+label)
 ax.set(ylabel='Full signal probability in GP training [%]',title='C. MC tails versus a Gaussian signal');ax.legend(ncol=1,fontsize=8.2)
 ax=axes[1,1]
 for prefix,color,label in [('historical',ORANGE,'Historical window'),('starter',BLUE,'Starter window')]:ax.plot(x,g[f'MC_over_Gaussian_{prefix}_leakage_ratio'],'o-',color=color,label=label)
 ax.set(ylabel='MC / Gaussian training leakage ratio',title='D. Compare the shapes within the same window');ax.legend()
 for ax in axes.flat:ax.set(xlabel='Generated signal mass [MeV]',xticks=[80,120,160,200,240]);ax.grid(alpha=.18)
 fig.suptitle(r'2021 v16 TC signal MC: historical $\pm2.25\sigma_m$ versus requested $u\in[-4,3]$',fontsize=13,y=.992)
 save(fig,'window_width_and_signal_leakage',
  'How to read: u = (reconstructed mass - core center) / core width; sigma_m is the historical Gaussian analysis width.\n'
  'Interior core parameters are predicted from neighboring samples with the tested mass omitted, matching v6.3.7; endpoints use their measured cores.\n'
  'Gaussian curves use the shifted Gaussian with sigma_m. Leakage counts only GP training bins within the recorded background spectrum.\n'
  'All probabilities retain the full selected signal normalization. Probability outside the spectrum is saved separately. No uncertainty bars: fixed-template geometry.',(0,.145,1,.955))
 for source in SOURCE:
  fig,axes=plt.subplots(1,3,figsize=(12,5.25));aa=a[a.source==source];qq=q[q.source==source]
  for policy,color,label in [('direct_starter',BLUE,'Direct MC, starter window'),('gaussian_baseline',ORANGE,'Gaussian, historical window')]:
   aa1=aa[aa.policy==policy].sort_values('mass_MeV');qq1=qq[qq.policy==policy].sort_values('mass_MeV')
   axes[0].plot(aa1.mass_MeV,aa1.full_response,color=color,label=label+'; contaminated')
   axes[0].plot(aa1.mass_MeV,aa1.clean_response,color=color,ls='--',label=label+'; clean training')
   axes[0].errorbar(qq1.mass_MeV,qq1.full_response,yerr=qq1.full_response_SE,fmt='o',color=color,capsize=2)
   axes[0].errorbar(qq1.mass_MeV,qq1.clean_response,yerr=qq1.clean_response_SE,fmt='s',mfc='white',color=color,capsize=2)
   axes[1].plot(aa1.mass_MeV,100*aa1.leakage_loss_fraction,color=color,label=label)
   axes[1].errorbar(qq1.mass_MeV,100*qq1.leakage_loss_fraction,yerr=100*qq1.leakage_loss_SE,fmt='o',color=color,capsize=2)
   axes[2].plot(aa1.mass_MeV,aa1.response_scaling_penalty_percent,color=color,label=label)
   axes[2].errorbar(qq1.mass_MeV,qq1.response_scaling_penalty_percent,yerr=[qq1.response_scaling_penalty_percent-qq1.penalty95_low,qq1.penalty95_high-qq1.response_scaling_penalty_percent],fmt='o',color=color,capsize=2)
  axes[0].axhline(1,color=GRAY,lw=.7);axes[0].set(ylabel='Mean added fitted yield / injected yield A',title='A. Response with and without contamination')
  axes[1].set(ylabel='Yield suppressed by GP training [% of A]',title='B. Isolate the effect of signal in sidebands')
  axes[2].set(ylabel='Extra response-scaled yield noise [%]',title='C. Precision penalty from contamination')
  handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.955),fontsize=9)
  for ax in axes:ax.set(xlabel='Generated signal mass [MeV]',xticks=[80,120,160,200,240]);ax.grid(alpha=.18)
  fig.suptitle('2021 10% | '+SOURCE[source]+' | Full v16 TC signal-MC injection, A = 3 s0',fontsize=12,y=.998)
  save(fig,'leakage_impact_'+source,
   'How to read: clean training removes only the injected signal from GP training bins; the signal-fit counts, template and window are identical.\n'
   'Lines: deterministic mean-count spectra at nine masses, not toy means. Markers: 100 paired saved evaluation toys at 100, 160 and 220 MeV.\n'
   'A/B bars: sample standard errors of paired means. C bars: 95% intervals from 2,000 whole-toy bootstrap resamples; masses/methods stay paired.\n'
   'Panel B = mean(Ahat_clean - Ahat_contaminated) / A. Panel C = 100 x (R_clean / R_contaminated - 1); background-only noise is unchanged.\n'
   's0 is the archived Gaussian yield-error scale. This truth-assisted control diagnoses absorption; it is not a usable data fit or a measured upper-limit gain.',(0,.27,1,.84))

def main():
 before=sha(P/'MANIFEST.sha256')
 records=[geometry(m)[0] for m in MASSES];g=pd.DataFrame(records);g.to_csv(B/'window_leakage_by_mass.csv',index=False,float_format='%.17g')
 q,a,replays=run_controls();plots(g,q,a)
 # Existing study tables are a cross-check at all three production masses.
 original=pd.read_csv(P/'results/window_geometry.csv');checks=[]
 for m in S.MASSES:
  for policy,label in [('gaussian_baseline','historical'),('morph_starter','starter')]:
   row=original[(original.mass_MeV==m)&(original.policy==policy)].iloc[0]
   delta=float(g[g.mass_MeV==m].iloc[0][f'MC_{label}_training_fraction']-row.MC_training_fraction)
   assert abs(delta)<1e-12;checks.append(dict(mass=m,policy=policy,leakage_probability_difference=delta))
 assert sha(P/'MANIFEST.sha256')==before
 result=dict(passed=True,masses=MASSES,existing_toys_per_setting=100,new_toy_draws=0,new_clean_training_fits=3000,mean_count_fits=270,contaminated_fit_replays=30,
  same_fitted_counts_verified=True,all_new_fits_valid=True,geometry_matches_v637=checks,replay_checks=replays,parent_manifest_sha256=before,parent_unchanged=True,
  source_files={str(p.relative_to(P)):sha(p) for p in [P/'scripts/run_study.py',P/'scripts/templates.py',P/'results/evaluation_rows.csv',P/'results/window_geometry.csv',P/'inputs/window_pilot_reference.json']},
  scope='Additional local diagnostics only; original v6.3.7 and v6.3.6 studies and releases unchanged. Means/ratios conditional on fixed templates, source, A=3s0 and existing ensembles. No new upper limits or coverage claims.')
 (B/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(q[(q.source=='gp_mean')&q.policy.isin(['direct_starter','gaussian_baseline'])].to_string(index=False),flush=True)
 print(g[['mass_MeV','continuous_width_increase_percent','MC_historical_training_fraction','MC_starter_training_fraction','Gaussian_starter_training_fraction','MC_leakage_reduction_percentage_points']].to_string(index=False),flush=True)
if __name__=='__main__':
 if '--plots-only' in sys.argv:
  plots(pd.read_csv(B/'window_leakage_by_mass.csv'),pd.read_csv(B/'paired_training_removal_summary.csv'),pd.read_csv(B/'mean_count_training_removal.csv'))
 else:main()
