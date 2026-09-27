from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v584-mpl')
import numpy as np,pandas as pd
from scipy.stats import norm,chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures';S=B/'source'
d=pd.read_csv(B/'results/significance_and_reach.csv',dtype={'scope':str});p=pd.read_csv(B/'results/peaks.csv',dtype={'scope':str});g=pd.read_csv(B/'results/paired_grid_comparison.csv',dtype={'scope':str});v=pd.read_csv(B/'results/local_validation.csv',dtype={'scope':str});gv=pd.read_csv(B/'results/global_validation.csv',dtype={'scope':str})
widths=[2.25,2.4,2.5,2.6];wc={2.25:'#222222',2.4:'#287ca7',2.5:'#ca8133',2.6:'#8855a2'}
labs={'2015':'2015 full','2016':'2016 full','2021':'2021 10%','combined':'Shared coupling','fisher':'Fisher (atom calibrated)','stouffer':'Equal-weight signed Stouffer'}
mc={'combined':'#292929','fisher':'#267da4','stouffer':'#b44d48'}
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=155)
 plt.close(fig)
def curves(scope,w,domain='full'):return d[(d.scope==scope)&(d.width_sigma==w)&(d.domain==domain)]
def panels(metric,ylabel,name,log=False,ratio=False):
 fig,axes=plt.subplots(2,2,figsize=(10.7,6.8))
 for ax,scope in zip(axes.flat,['2015','2016','2021','combined']):
  base=curves(scope,2.25).set_index('mass_MeV')
  for w in widths:
   q=curves(scope,w);yy=q[metric].to_numpy()
   if ratio:yy=yy/base.loc[q.mass_MeV,metric].to_numpy()
   ax.plot(q.mass_MeV,yy,c=wc[w],lw=1.15,label=rf'$\pm{w:g}\sigma$')
   if ratio and w!=2.25:ax.plot(q.mass_MeV,q.epsilon2_asimov90.to_numpy()/base.loc[q.mass_MeV,'epsilon2_asimov90'].to_numpy(),c=wc[w],ls='--',lw=.9)
  ax.set(title=labs[scope],xlabel='Mass [MeV]',ylabel=ylabel)
  if log:ax.set_yscale('log')
  if metric=='local_Z':ax.set_ylim(0,4.5)
  if metric=='global_Z':ax.set_ylim(0,3.2)
  if metric=='global_p':ax.set_ylim(.01,1.08)
  ax.legend(fontsize=9,ncol=2,loc='best')
  if ratio:ax.axhline(1,c='.6',lw=.6)
 fig.tight_layout(h_pad=2,w_pad=2);save(fig,name)
panels('local_Z','Reference-local Z','width_local_Z')
panels('global_Z','Full-scope global Z','width_global_Z')
panels('global_p','Full-scope global p','width_global_p',True)
panels('epsilon2_90',r'Observed 90% CLs $\epsilon^2$','width_observed_reach',True)
panels('epsilon2_asimov90',r'Model-Asimov 90% CLs $\epsilon^2$','width_asimov_reach',True)
panels('epsilon2_90','Upper endpoint / baseline endpoint','width_reach_ratios',False,True)

fig,ax=plt.subplots(2,2,figsize=(10.7,6.8))
for row,domain in enumerate(['full','overlap']):
 for col,metric in enumerate(['local_Z','global_Z']):
  for method in mc:
   q=curves(method,2.25,domain);ax[row,col].plot(q.mass_MeV,q[metric],c=mc[method],lw=1.25,label=labs[method])
  ax[row,col].set(title=('19-250 MeV active datasets' if domain=='full' else '50-100 MeV: all three datasets'),xlabel='Mass [MeV]',ylabel='Local Z' if col==0 else 'Domain-global Z');ax[row,col].legend(fontsize=8.5)
  ax[row,col].set_ylim(0,4.5 if col==0 else 3.4)
fig.tight_layout(h_pad=2,w_pad=2);save(fig,'combination_methods_baseline')
fig,ax=plt.subplots(2,2,figsize=(10.7,6.8))
for row,method in enumerate(['fisher','stouffer']):
 for col,metric in enumerate(['local_Z','global_p']):
  for w in widths:
   q=curves(method,w);ax[row,col].plot(q.mass_MeV,q[metric],c=wc[w],lw=1.1,label=rf'$\pm{w:g}\sigma$')
  ax[row,col].set(title=labs[method],xlabel='Mass [MeV]',ylabel='Local Z' if col==0 else '19-250 MeV global p');ax[row,col].legend(fontsize=9,ncol=2)
  if col==1:ax[row,col].set(yscale='log',ylim=(.001,1.08))
  else:ax[row,col].set_ylim(0,4.5)
fig.tight_layout(h_pad=2,w_pad=2);save(fig,'independent_methods_widths')

comp=pd.read_csv(B/'results/peak_composition.csv',dtype={'scope':str});c92=comp[(comp.width_sigma==2.25)&(comp.mass_MeV==92)]
fig,ax=plt.subplots(1,2,figsize=(10.5,3.6));ix=np.arange(3)
ax[0].errorbar(c92.epsilon2_hat,ix,xerr=c92.epsilon2_fit_sigma,fmt='o',c='#267da4',capsize=4);ax[0].axvline(c92.shared_epsilon2_hat.iloc[0],c='#b44d48',ls='--',label='Shared-coupling estimate');ax[0].set(xscale='log',yticks=ix,yticklabels=['2015','2016','2021 10%'],xlabel=r'Fitted $\epsilon^2$ (curvature 1-sigma errors)',title='92 MeV: separate fitted couplings');ax[0].invert_yaxis();ax[0].legend(fontsize=9,loc='lower right')
ax[1].bar(ix,100*c92.null_information_fraction,color=['#3279a2','#b55747','#398570'])
for i,x in enumerate(c92.null_information_fraction):ax[1].text(i,100*x+1.5,f'{100*x:.2f}%',ha='center')
ax[1].set(xticks=ix,xticklabels=['2015','2016','2021 10%'],ylabel='Fraction of local null information [%]',ylim=(0,99),title='Why equal-weight and coupling scores differ');fig.tight_layout(w_pad=2);save(fig,'peak_92_coupling_weights')

fig,ax=plt.subplots(1,2,figsize=(10.5,4.0));names=['2015','2016','2021','combined'];xx=np.arange(4)
for j,step in enumerate([.5,1.]):
 q=g[g.step_MeV==step].set_index('scope').loc[names];ax[0].bar(xx+(j-.5)*.32,q.local_Z,.32,label=f'{step:g} MeV grid');ax[1].bar(xx+(j-.5)*.32,q.global_Z,.32,label=f'{step:g} MeV grid')
for a,metric in zip(ax,['Local peak Z','Global Z at each grid peak']):a.set(xticks=xx,xticklabels=['2015','2016','2021\n10%','Shared\ncoupling'],ylabel=metric);a.legend(fontsize=9)
fig.tight_layout();save(fig,'grid_peak_comparison')

fig,axes=plt.subplots(2,2,figsize=(10.5,6.8));reach=[]
for ax,domain,metric,title in [(axes[0,0],'full','local_Z','Local peaks: full scopes'),(axes[0,1],'full','global_Z','Global: full scopes'),(axes[1,0],'overlap','local_Z','Local peaks: common overlap'),(axes[1,1],'overlap','global_Z','Global: common overlap')]:
 scopes=list(labs) if domain=='full' else list(mc);mat=np.array([[p[(p.width_sigma==w)&(p.scope==s)&(p.domain==domain)][metric].iloc[0] for w in widths] for s in scopes]);im=ax.imshow(mat,aspect='auto',vmin=0,vmax=max(4.,float(p.local_Z.max())),cmap='YlGnBu');ax.set(xticks=range(4),xticklabels=[f'{w:g}' for w in widths],yticks=range(len(scopes)),yticklabels=[labs[s] for s in scopes],xlabel='Blind half-width / resolution sigma',title=title)
 for i in range(len(scopes)):
  for j in range(4):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>2.5 else 'black',fontsize=10)
fig.tight_layout(w_pad=2,h_pad=2);save(fig,'peak_comparison_matrix')

fig,ax=plt.subplots(1,2,figsize=(10.5,4.1))
for scope,c in zip(names,['#3279a2','#b55747','#398570','#825595']):
 q=v[v.scope==scope];ax[0].plot(q.width_sigma,q.null_offset_RMS,'-o',c=c,label=labs[scope]);ax[1].plot(q.width_sigma,q.average_standardized_SD,'-o',c=c,label=labs[scope])
ax[0].set(xlabel='Blind half-width / sigma',ylabel='RMS deterministic source root',title='Background-reference offsets');ax[0].legend(fontsize=9);ax[1].axhline(1,c='.5',ls='--');ax[1].set(xlabel='Blind half-width / sigma',ylabel='Mean standardized Poisson SD',title='Paired complete-scan validation');ax[1].legend(fontsize=9);fig.tight_layout();save(fig,'width_response_validation')

for w in widths[1:]:
 for scope in names:
  q=curves(scope,w).set_index('mass_MeV');base=curves(scope,2.25).set_index('mass_MeV')
  for typ,key in [('observed','epsilon2_90'),('model_Asimov','epsilon2_asimov90')]:
   r=q[key]/base[key];reach.append(dict(width_sigma=w,scope=scope,type=typ,median_ratio=float(r.median()),minimum_ratio=float(r.min()),maximum_ratio=float(r.max()),minimum_mass_MeV=float(r.idxmin()),maximum_mass_MeV=float(r.idxmax())))
pd.DataFrame(reach).to_csv(B/'results/reach_ratios_summary.csv',index=False)
# Separate question: any excess anywhere in EACH experiment, masses may differ.
experiment=[]
for w in widths:
 pp=p[(p.width_sigma==w)&(p.domain=='full')&p.scope.isin(['2015','2016','2021'])].set_index('scope').loc[['2015','2016','2021']];pg=pp.global_p.to_numpy();t=-2*np.log(pg).sum();pv=float(chi2.sf(t,6));experiment.append(dict(width_sigma=w,statistic=t,p_fisher_of_individual_global=pv,Z=max(0.,float(norm.isf(pv))),peak_masses_MeV=','.join(f'{x:g}' for x in pp.mass_MeV),interpretation='Approximate combination of independent experiment-global p estimates; permits different peak masses; not common-resonance evidence'))
pd.DataFrame(experiment).to_csv(B/'results/fisher_of_experiment_globals.csv',index=False)
def tex(name,head,rows,align):
 (S/name).write_text('\\begin{center}\\small\\begin{tabular}{'+align+'}\\toprule\n'+head+'\\\\\\midrule\n'+'\n'.join(' & '.join(r)+'\\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n')
rows=[]
for scope in labs:
 r=p[(p.width_sigma==2.25)&(p.domain=='full')&(p.scope==scope)].iloc[0];rows.append([labs[scope].replace('%','\\%'),f'{r.mass_MeV:g}',f'{r.local_Z:.3f}',f'{r.global_p:.4f}',f'{r.global_Z:.3f}'])
tex('baseline_peaks.tex','Search / combination & $m$ [MeV] & Local $Z$ & Global $p$ & Global $Z$',rows,'lrrrr')
rows=[]
for w in widths:
 row=[f'{w:g}']
 for scope in labs:row.append(f"{p[(p.width_sigma==w)&(p.domain=='full')&(p.scope==scope)].global_Z.iloc[0]:.3f}")
 rows.append(row)
tex('width_global_peaks.tex','Half-width & 2015 & 2016 & 2021 & Coupled & Fisher & Stouffer',rows,'lrrrrrr')
rows=[]
for r in experiment:rows.append([f"{r['width_sigma']:g}",r['peak_masses_MeV'],f"{r['p_fisher_of_individual_global']:.4f}",f"{r['Z']:.3f}"])
tex('experiment_global_table.tex','Half-width & Individual peak masses [MeV] & Fisher $p$ & $Z$',rows,'llrr')
q=g.set_index(['scope','step_MeV']);rows=[]
for scope in names:
 fine=q.loc[(scope,.5)];co=q.loc[(scope,1.)];rows.append([labs[scope].replace('%','\\%'),f'{fine.peak_mass_MeV:g} / {co.peak_mass_MeV:g}',f'{fine.local_Z:.3f} / {co.local_Z:.3f}',f'{fine.global_Z:.3f} / {co.global_Z:.3f}',f'{fine.p_at_fixed_Z3:.4f} / {co.p_at_fixed_Z3:.4f}'])
tex('grid_table.tex','Scope & Peak masses & Local $Z$ & Global $Z$ & $p$ at fixed $Z=3$',rows,'lrrrr')
summary={'baseline':p[(p.width_sigma==2.25)&(p.domain=='full')][['scope','mass_MeV','local_Z','global_p','global_Z']].to_dict('records'),'reach_ratios':reach,'experiment_global_combinations':experiment,'figures':len(list(F.glob('*.pdf')))}
(B/'results/report_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(p[['width_sigma','scope','domain','mass_MeV','local_Z','global_p','global_Z']].to_string(index=False));print('Figures',summary['figures'])
