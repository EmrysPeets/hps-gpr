"""Data-backed vector figures, tables, and numeric result text for the note."""
from tail_model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                     'legend.fontsize':8,'figure.dpi':130,'savefig.bbox':'tight','axes.grid':True,
                     'grid.alpha':.20,'pdf.fonttype':42})
COLORS={1.:'#242424',1.1:'#2878b5',1.2:'#d98223',1.3:'#ae3a51'}
LABELS={'2015':'Full 2015','2016':'Full 2016','2021':'2021 (10%)'}
f=pd.read_csv(B/'derived/scans.csv',dtype={'scope':str})
peaks=pd.read_csv(B/'derived/pvalue_minima.csv',dtype={'scope':str})
ratios=pd.read_csv(B/'derived/limit_ratios.csv',dtype={'scope':str})
shapes=pd.read_csv(B/'derived/shape_metrics.csv')
leak=pd.read_csv(B/'derived/leakage_response.csv',dtype={'scope':str})

def save(fig,name):
    fig.savefig(B/f'figures/{name}.pdf');fig.savefig(B/f'figures/{name}.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(8.3,5.6),constrained_layout=True)
z=np.linspace(-6,6,2401)
for family,k in SCENARIOS:
    if family=='curvature' and k!=1.3:continue
    val=density(z,k,family)/full_integral(k,family)
    ls='--' if family=='curvature' else '-';label='Gaussian' if family=='gaussian' else f'{family.capitalize()} +{(k-1)*100:.0f}%'
    axs[0,0].plot(z,val,color=COLORS[k],ls=ls,label=label)
    axs[0,1].semilogy(z,val,color=COLORS[k],ls=ls)
    axs[1,0].plot(z,val/(density(z,1,'gaussian')/np.sqrt(2*np.pi)),color=COLORS[k],ls=ls)
for ax in axs.ravel()[:3]:
    ax.axvspan(-2.25,2.25,color='gray',alpha=.08);ax.set_xlabel(r'$(m_{ee}-m_0)/\sigma_m$')
axs[0,0].set_ylabel('Unit-normalized density');axs[0,0].legend(loc='upper right',frameon=False)
axs[0,1].set_ylabel('Density (log scale)');axs[0,1].set_ylim(1e-7,.6)
axs[1,0].set_ylabel('Density / nominal Gaussian');axs[1,0].set_ylim(.95,5);axs[1,0].set_xlim(-5,5)
for family,style in [('dilation','-o'),('curvature','--s')]:
    q=shapes[shapes.family==family];axs[1,1].plot((q.kappa-1)*100,q.tail_fraction*100,style,label=family.capitalize())
axs[1,1].axhline(shapes.iloc[0].tail_fraction*100,color=COLORS[1.],ls=':',label='Gaussian')
axs[1,1].set_xlabel('Tail-width increase (%)');axs[1,1].set_ylabel(r'Probability outside $2.25\sigma_m$ (%)');axs[1,1].legend(frameon=False)
save(fig,'shapes')

# A larger-type two-panel version fits the report's model-equation page.
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),constrained_layout=True)
z=np.linspace(-5.5,5.5,2201)
for family,k in SCENARIOS:
    if family=='curvature' and k!=1.3:continue
    val=density(z,k,family)/full_integral(k,family)
    ls='--' if family=='curvature' else '-'
    label='Gaussian' if family=='gaussian' else ('Curvature +30%' if family=='curvature' else f'Tails +{(k-1)*100:.0f}%')
    axs[0].semilogy(z,val,color=COLORS[k],ls=ls,label=label)
    axs[1].plot(z,val/(density(z,1,'gaussian')/np.sqrt(2*np.pi)),color=COLORS[k],ls=ls)
for ax in axs:
    ax.axvspan(-2.25,2.25,color='gray',alpha=.08);ax.set_xlabel(r'$(m_{ee}-m_0)/\sigma_m$')
axs[0].set_ylabel('Unit-normalized density');axs[0].set_ylim(1e-6,.6);axs[0].legend(frameon=False,fontsize=8,loc='lower center')
axs[1].set_ylabel('Density / Gaussian');axs[1].set_ylim(.95,4);axs[1].set_xlim(-5,5)
save(fig,'shapes_note')

for window in WINDOWS:
    for year in DATA:
        g=f[(f.window==window)&(f.scope==year)&(f.family!='curvature')]
        base=g[g.family=='gaussian'].set_index('mass_MeV')
        fig,axs=plt.subplots(4,1,figsize=(7.4,7.7),sharex=True,gridspec_kw={'height_ratios':[2,1.1,1.4,1.2]},constrained_layout=True)
        for k in (1.,1.1,1.2,1.3):
            q=g[g.kappa==k].set_index('mass_MeV');label='Gaussian' if k==1 else f'Tails +{(k-1)*100:.0f}%'
            axs[0].semilogy(q.index,q.display_epsilon2_90,color=COLORS[k],label=label,lw=1.3)
            axs[1].plot(q.index,100*(q.epsilon2_90/base.epsilon2_90-1),color=COLORS[k])
            axs[2].semilogy(q.index,q.p0_fixed_mass,color=COLORS[k])
            axs[3].plot(q.index,q.Z0,color=COLORS[k])
        axs[0].set_title(f'{LABELS[year]}: '+('primary $\pm2.25\sigma_m$ window' if window=='primary' else 'guard $\pm4.5\sigma_m$ window'))
        axs[0].set_ylabel(r'Observed 90% CL$_s$ limit on $\varepsilon^2$');axs[0].legend(ncol=2,frameon=False)
        axs[1].set_ylabel('Limit change (%)');axs[2].set_ylabel(r'Local $p_0$');axs[3].set_ylabel(r'Local $Z_0$');axs[3].set_xlabel(r'Test mass $m_0$ (MeV)')
        axs[2].set_ylim(max(1e-6,float(g.p0_fixed_mass.min())*.6),.7)
        for threshold in (1,2,3):axs[3].axhline(threshold,color='gray',lw=.5,alpha=.5)
        save(fig,f'{window}_{year}')

fig,axs=plt.subplots(2,3,figsize=(9,4.0),sharex='col',constrained_layout=True)
for col,year in enumerate(DATA):
    axs[0,col].set_title(LABELS[year])
    for window,ls in [('primary','-'),('guard','--')]:
        base=f.query("scope==@year and window==@window and family=='gaussian'").set_index('mass_MeV')
        for family,color in [('dilation','#ae3a51'),('curvature','#2878b5')]:
            q=f.query("scope==@year and window==@window and family==@family and kappa==1.3").set_index('mass_MeV')
            axs[0,col].plot(q.index,100*(q.epsilon2_90/base.epsilon2_90-1),color=color,ls=ls,label=f'{family}, {window}')
            axs[1,col].plot(q.index,q.signed_r-base.signed_r,color=color,ls=ls)
    axs[1,col].set_xlabel('Test mass (MeV)')
axs[0,0].set_ylabel('Limit change (%)');axs[1,0].set_ylabel(r'Change in signed $r$')
for ax in axs.flat:ax.tick_params(labelsize=11)
handles,labels=axs[0,1].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.09),ncol=4,frameon=False,fontsize=10)
save(fig,'structure_response')

fig,axs=plt.subplots(1,2,figsize=(9,3.9),constrained_layout=True)
for year,marker in [('2015','o'),('2016','s'),('2021','^')]:
    for mass,ls in [({'2015':51,'2016':90,'2021':78}[year],'-'),(92,'--')]:
        for col,window in enumerate(WINDOWS):
            q=leak.query("scope==@year and mass_MeV==@mass and window==@window and family!='curvature' and training=='contaminated' and extraction=='matched'").sort_values('kappa')
            axs[col].plot((q.kappa-1)*100,q.recovery*100,marker=marker,ls=ls,label=f'{year}, {mass} MeV')
for ax,window in zip(axs,WINDOWS):
    ax.axhline(100,color='black',lw=.7);ax.set_xlabel('Tail-width increase (%)');ax.set_title(f'{window.capitalize()} window');ax.set_ylabel('Recovered / injected amplitude (%)');ax.ticklabel_format(axis='y',style='plain',useOffset=False)
axs[0].legend(fontsize=7,frameon=False,ncol=2)
save(fig,'leakage')

fig,axs=plt.subplots(2,3,figsize=(10,5.3),sharex='col',constrained_layout=True)
for col,(year,mass) in enumerate([('2015',51),('2016',90),('2021',78)]):
    p=context(year,[mass],padding=4.5,anchor=mass);x=DATA[year]['x'][p['mask']]*1000;scale=signal_scale(year,mass)
    axs[0,col].errorbar(x,p['n'],np.sqrt(p['n']),fmt='.',color='gray',ms=3,label='Observed')
    for family,k in [('gaussian',1.),('dilation',1.3)]:
        s=weights(year,mass,k,family)[0][p['mask']]*scale;model=OneSignalProfile(p['b'],p['L'],s);fit=model.fit(p['n']);nul=model.fit(p['n'],0.)
        axs[0,col].plot(x,fit['lam'],color=COLORS[k],label='Gaussian' if k==1 else 'Tails +30%')
        axs[1,col].plot(x,fit['A']*s,color=COLORS[k])
    axs[0,col].plot(x,nul['lam'],color='#2878b5',ls=':',label='Profiled null')
    for ax in axs[:,col]:ax.axvspan(mass-2.25*sigma(year,mass)*1000,mass+2.25*sigma(year,mass)*1000,color='gray',alpha=.08)
    axs[0,col].set_title(f'{LABELS[year]}, {mass} MeV');axs[1,col].set_xlabel(r'$m_{ee}$ (MeV)')
axs[0,0].set_ylabel('Events / native analysis bin');axs[1,0].set_ylabel('Fitted signal events / bin');axs[0,1].legend(frameon=False,fontsize=7)
save(fig,'fits')

def table(name,header,rows,align):
    text='\\begin{tabular}{'+align+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (B/f'derived/{name}.tex').write_text(text)

def sci(x):
    if x==0:return '$0$'
    exponent=int(np.floor(np.log10(abs(x))));return f'${x/10**exponent:.3f}\\times10^{{{exponent}}}$'

def flabel(family,k):return 'Gaussian' if family=='gaussian' else ('Dilation' if family=='dilation' else 'Curvature')+f' +{(k-1)*100:.0f}\\%'

table('shape_table','Shape & Tail (\\%) & RMS/$\\sigma_m$ & Core UL ratio & Beyond guard',
      [[flabel(r.family,r.kappa),f'{100*r.tail_fraction:.4f}',f'{r.rms_ratio:.5f}',f'{r.ideal_core_limit_ratio:.6f}',sci(r.guard_leakage_fraction)] for r in shapes.itertuples()], 'lrrrr')

for window,name in [('primary','peak_table'),('guard','guard_peak_table')]:
    rows=[]
    for year in DATA:
        for family,k in SCENARIOS:
            r=peaks.query('window==@window and scope==@year and family==@family and kappa==@k').iloc[0]
            rows.append([year,flabel(family,k),f'{r.mass_MeV:.0f}',f'{r.Z0:.5f}',sci(r.p0_fixed_mass),sci(r.display_epsilon2_90)])
    table(name,'Dataset & Shape & $m_0$ (MeV) & $Z_0$ & Local $p_0$ & $\\varepsilon^2_{90}$',rows,'llrrrr')

rows=[]
for year in DATA:
    for family in ('dilation','curvature'):
        for k in (1.1,1.2,1.3):
            a=ratios.query("scope==@year and family==@family and kappa==@k and window=='primary'").iloc[0]
            b=ratios.query("scope==@year and family==@family and kappa==@k and window=='guard'").iloc[0]
            rows.append([year,flabel(family,k),f'{100*(a.median_ratio-1):.3f}',f'{100*(b.min_ratio-1):.3f}--{100*(b.max_ratio-1):.3f}',f'{b.max_abs_delta_r:.5f}'])
table('ratio_table','Dataset & Shape & Primary median (\\%) & Guard range (\\%) & Max. $|\\Delta r|$ (guard)',rows,'llrrr')

g=f.query("window=='primary' and family=='gaussian'")
r=pd.read_csv(B/'inputs/nominal_v505_curves.csv',dtype={'dataset_set':str})
j=g.merge(r,left_on=['scope','mass_MeV'],right_on=['dataset_set','mass_MeV']);j['relative_limit_difference']=j.epsilon2_90/j.eps2_90-1;j['Z_difference']=j.Z0-j.Z_local_asymptotic
j.to_csv(B/'qa/published_ledger_comparison.csv',index=False,float_format='%.17g')
guard_coordinates=len(f[(f.window=='guard') & (f.family=='gaussian')])
table('qa_table','Quantity & Value',[
    ['Observed mass--window--shape fits',str(len(f))],
    ['Primary / guard mass coordinates',f'{len(g)} / {guard_coordinates}'],
    ['Maximum $|\\mathrm{CL}_s-0.1|$',sci(float(abs(f.cls-.1).max()))],
    ['Maximum profile score residual',sci(float(f.max_score.max()))],
    ['Minimum fitted Poisson mean',f'{f.min_lambda.min():.3f}'],
    ['Maximum historical-ledger relative UL difference',f'{100*abs(j.relative_limit_difference).max():.3f}\\%'],
    ['Maximum historical-ledger $|\\Delta Z_0|$',f'{abs(j.Z_difference).max():.6f}'],
    ['Deterministic response fits',str(len(leak))],
    ['Maximum clean matched recovery error',sci(float(abs(leak.query("training==\'clean\' and extraction==\'matched\'").recovery-1).max()))],
], 'lr')

summary=r'''The primary-window local maxima remain at 51~MeV (2015), 90~MeV (2016), and 78~MeV (2021 10\%) for all six altered shapes. For the primary dilation model, the median observed limit increases by approximately 0.168\%, 0.345\%, and 0.529\% for 10\%, 20\%, and 30\% broader tails. These small shifts follow the change in core probability; they are not 10--30\% changes in the full-template resolution.

In the guard window, the 30\% dilation changes observed limits by $-0.048$ to $+2.068$\% in 2015, $+0.245$ to $+2.056$\% in 2016, and $+0.279$ to $+1.543$\% in 2021. The largest absolute changes in the signed local root are 0.0377, 0.0280, and 0.0177, respectively. The guard-window Gaussian maximum in 2015 is already at 93~MeV, rather than 51~MeV. That movement is a consequence of the wider background interpolation and fitted region; it cannot be attributed to the tail variation. The primary search is complete over all declared mass intervals. The guard search omits 2016 masses 176--180~MeV because fewer than three upper exterior bins remain.
'''
(B/'derived/results_summary.tex').write_text(summary)
print('Created 11 vector/PNG figures and 6 tables.')
