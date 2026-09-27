"""Build report figures/tables from saved v6.1 arrays; never fit observed data."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parents[1]
F = B / 'figures'
D = B / 'derived'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,
    'axes.titlesize':10,'legend.fontsize':8,'axes.grid':True,'grid.alpha':.18,
    'pdf.fonttype':42,'savefig.bbox':'tight'})
BLUE = '#216ca1'
RED = '#a64139'
GRAY = '#484848'

def esc(s):
    for a,b in [('\\',r'\textbackslash{}'),('_',r'\_'),('%',r'\%'),('&',r'\&')]:
        s=str(s).replace(a,b)
    return s

def sci(x, digits=3):
    x=float(x)
    if not np.isfinite(x): raise ValueError('Nonfinite table entry')
    if x==0:return '$0$'
    e=int(np.floor(np.log10(abs(x))))
    return f'${x/10**e:.{digits}f}\\times10^{{{e}}}$'

def table(name,header,rows,align):
    text='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (D/f'report_{name}.tex').write_text(text)

def save(fig,name):
    fig.savefig(F/f'{name}.pdf')
    fig.savefig(F/f'{name}.png',dpi=145)
    plt.close(fig)

def sigma(mass_MeV):
    return np.polynomial.polynomial.polyval(np.asarray(mass_MeV)/1000.,
        [0.00184825,-0.001375,0.085875])

def field(row,names,required=True):
    for name in names:
        if name in row and pd.notna(row[name]):return float(row[name])
    if required:raise KeyError(f'Required field absent: {names}; available {list(row.index)}')
    return None

def main():
    scans=pd.read_csv(D/'scans.csv')
    year_col='year' if 'year' in scans else 'scope'
    scans[year_col]=scans[year_col].astype(str)
    scans=scans[scans[year_col]=='2021'].copy()
    # Require the scientific ledgers; the report never invents their contents.
    comparison=pd.read_csv(D/'comparison.csv')
    shapes=pd.read_csv(D/'shape_summary.csv')
    agreement=json.loads((D/'agreement.json').read_text())
    if not len(comparison) or not len(shapes) or not agreement:
        raise ValueError('Empty result ledger')
    g=scans[scans.model=='gaussian'].sort_values('mass_MeV')
    n=scans[scans.model=='mc_direct'].sort_values('mass_MeV')
    morph=scans[scans.model=='mc_morph'].sort_values('mass_MeV')
    if not len(g) or not len(n):raise ValueError('Need Gaussian and direct MC fits')
    paired=n.merge(g,on='mass_MeV',suffixes=('_mc','_gaussian'),validate='one_to_one')
    paired['ratio']=paired.display_epsilon2_90_mc/paired.display_epsilon2_90_gaussian
    paired['delta_r']=paired.signed_r_mc-paired.signed_r_gaussian
    if not np.all(np.isfinite(paired[['ratio','delta_r']])):raise ValueError('Invalid paired comparison')

    samples=[]
    for path in sorted((B/'histograms').glob('m[0-9][0-9][0-9].npz')):
        mass=int(path.stem[1:]);data=np.load(path,allow_pickle=False)
        meta_path=path.with_suffix('.json')
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else json.loads(str(data['metadata']))
        edges=data['edges_GeV'];h=data['sumw'];v=data['sumw2']
        if len(edges)!=len(h)+1 or h.sum()<=0 or np.any(h<0):raise ValueError(f'Bad template: {path}')
        samples.append((mass,edges,h,v,meta))
    if len(samples)!=12:raise ValueError(f'Expected twelve MC samples, got {len(samples)}')

    fig,axs=plt.subplots(4,3,figsize=(9.0,9.0),constrained_layout=True)
    for ax,(mass,edges,h,v,meta) in zip(axs.flat,samples):
        center=(edges[1:]+edges[:-1])*500.;width=np.diff(edges)*1000.
        sig=sigma(mass);w=np.diff(ndtr((edges-mass/1000.)/sig));w/=w.sum()
        ax.step(center,h/h.sum()/width,where='mid',color=BLUE,lw=.8,label='Stored reconstructed MC')
        ax.plot(center,w/width,color=GRAY,lw=1,label='Nominal Gaussian')
        for sign in (-1,1):ax.axvline(mass+sign*2.25*sig*1000,color=RED,ls=':',lw=.8)
        ax.set_xlim(0,400)
        ax.set_yscale('log');ax.set_ylim(1e-6,.4)
        ax.set_title(f'{mass} MeV; '+r'$N_{\mathrm{eff}}$'+f'={meta["neffective"]:,.0f}')
        ax.tick_params(labelsize=7)
    for ax in axs[-1]:ax.set_xlabel(r'Reconstructed $m_{ee}$ (MeV)')
    for ax in axs[:,0]:ax.set_ylabel(r'Density (MeV$^{-1}$)')
    axs[0,0].legend(frameon=False,fontsize=6)
    save(fig,'mc_shapes')

    rows=[]
    for mass,edges,h,v,meta in samples:
        q=shapes.loc[shapes.mass_MeV==mass]
        if len(q)!=1:raise ValueError(f'Missing or duplicated shape summary {mass}')
        q=q.iloc[0]
        bias=field(q,['median_bias_MeV'])
        width=field(q,['sigma68_MeV','width68_MeV','halfwidth68_MeV','quantile_sigma_MeV'])
        core=1-field(q,['tail_fraction_2p25sigma'])
        rows.append([str(mass),f'{meta["stats"]["selected"]:,}',str(meta['stats']['overflow']),
                     f'{bias:+.3f}',f'{width:.3f}',f'{100*core:.2f}'])
    table('samples',[r'$m_0$ (MeV)','Selected rows','Overflow','Median bias',r'$\sigma_{68}$ (MeV)',r'Core (\%)'],rows,'rrrrrr')
    below=np.array([meta['stats']['psum_below_2p8']/meta['stats']['entries'] for _,_,_,_,meta in samples])
    unit=all(np.isclose(meta['sumw'],meta['sumw2']) for _,_,_,_,meta in samples)
    frac=[h.sum()/meta['sumw'] for _,_,h,_,meta in samples]
    context=(f'The stored 0--400~MeV range contains {100*min(frac):.4f}--{100*max(frac):.4f}\\% '
       f'of the selected weight across samples; overflow counts are shown separately. '
       +('All extracted weights are unit weights, so $N_{\\rm eff}$ equals the selected count. ' if unit else '')
       +f'The raw particle-vector momentum diagnostic falls below 2.8~GeV for '
       f'{100*min(below):.2f}--{100*max(below):.2f}\\% of rows. '
       'A bounded read finds that stored \\texttt{psum} and \\texttt{psum\\_scalar} satisfy the cut in every inspected row; '
       'the raw-vector discrepancy therefore reflects an incompatible convention, not an established selection failure.\n')
    (D/'report_sample_context.tex').write_text(context)

    fig,axs=plt.subplots(3,1,figsize=(8.2,7.6),sharex=True,constrained_layout=True)
    for data,color,ls,label in [(g,GRAY,'-','Gaussian'),(morph,RED,'--','MC morph: exploratory')]:
        if not len(data):continue
        axs[0].semilogy(data.mass_MeV,data.display_epsilon2_90,color=color,ls=ls,lw=1.1,label=label)
        axs[1].semilogy(data.mass_MeV,data.p0_fixed_mass,color=color,ls=ls,lw=1.1)
        axs[2].plot(data.mass_MeV,data.signal_fraction_in_fit,color=color,ls=ls,lw=1.1)
    axs[0].scatter(n.mass_MeV,n.display_epsilon2_90,color=BLUE,s=24,zorder=5,label='Direct MC at generated masses')
    axs[1].scatter(n.mass_MeV,n.p0_fixed_mass,color=BLUE,s=24,zorder=5)
    axs[2].scatter(n.mass_MeV,n.signal_fraction_in_fit,color=BLUE,s=24,zorder=5)
    axs[0].set_ylabel(r'Observed 90% CL$_s$ limit on $\varepsilon^2$')
    axs[1].set_ylabel(r'Local $p_0$');axs[1].set_ylim(max(1e-9,scans.p0_fixed_mass.min()*.5),.75)
    axs[2].set_ylabel('Signal fraction in fitted bins');axs[2].set_xlabel('Test mass (MeV)')
    axs[0].legend(frameon=False,fontsize=8)
    save(fig,'observed_2021')

    rows=[]
    for r in paired.itertuples():
        rows.append([f'{r.mass_MeV:g}',sci(r.display_epsilon2_90_gaussian),sci(r.display_epsilon2_90_mc),
                     f'{r.ratio:.4f}',sci(r.p0_fixed_mass_gaussian),sci(r.p0_fixed_mass_mc),f'{r.delta_r:+.3f}'])
    table('native',['$m_0$','$\eps^2_{90,G}$','$\eps^2_{90,MC}$','Ratio','$p_{0,G}$','$p_{0,MC}$','$\Delta r$'],rows,'rrrrrrr')
    fig,axs=plt.subplots(2,1,figsize=(8,4.1),sharex=True,constrained_layout=True)
    axs[0].plot(paired.mass_MeV,100*(paired.ratio-1),'o-',color=BLUE)
    axs[1].plot(paired.mass_MeV,paired.delta_r,'o-',color=BLUE)
    axs[0].axhline(0,color=GRAY,lw=.7);axs[1].axhline(0,color=GRAY,lw=.7)
    axs[0].set_ylabel('Observed limit change (%)');axs[1].set_ylabel(r'Change in signed $r$')
    axs[1].set_xlabel('Generated/test mass (MeV)');save(fig,'native_agreement')
    r=paired.ratio.to_numpy();z=paired.delta_r.to_numpy()
    table('agreement',['Descriptive quantity','Direct MC versus Gaussian'],[
       ['Common generated masses',str(len(r))],['Median ratio',f'{np.median(r):.5f}'],
       ['16th--84th percentile ratio',f'{np.quantile(r,.16):.5f}--{np.quantile(r,.84):.5f}'],
       ['Minimum / maximum ratio',f'{r.min():.5f} / {r.max():.5f}'],
       ['Median absolute relative change',f'{100*np.median(abs(r-1)):.3f}\\%'],
       ['RMS log ratio',f'{np.sqrt(np.mean(np.log(r)**2)):.5f}'],
       ['Within 1 / 5 / 10\\%', ' / '.join(f'{100*np.mean(abs(r-1)<=t):.1f}\\%' for t in [.01,.05,.1])],
       ['Maximum $|\\Delta r|$',f'{abs(z).max():.5f}']], 'lr')
    largest=paired.iloc[int(np.argmax(abs(r-1)))]
    peak=n.loc[n.p0_fixed_mass.idxmin()]
    summary=(f'At the {len(r)} shared generated masses, the median MC-to-Gaussian limit ratio is '
      f'{np.median(r):.4f}; the range is {r.min():.4f}--{r.max():.4f}. '
      f'The median absolute relative limit change is {100*np.median(abs(r-1)):.2f}\\%. '
      f'The largest absolute relative limit change occurs at {largest.mass_MeV:g}~MeV '
      f'({100*(largest.ratio-1):+.2f}\\%). '
      f'The smallest direct-MC local probability among these tested points occurs at '
      f'{peak.mass_MeV:g}~MeV, with $p_0$ = {sci(peak.p0_fixed_mass)} and '
      f'$Z_0={peak.Z0:.3f}$. This is a sparse-grid local comparison, not a scan-global probability.\n')
    (D/'report_summary.tex').write_text(summary)
    loo=pd.read_csv(D/'leave_one_out.csv')
    precision=pd.read_csv(D/'mc_precision.csv')
    rows=[]
    for row in loo.itertuples():
        passed=str(row.shape_check_pass).lower() in ('true','1')
        rows.append([f'{row.mass_MeV:g}',f'{row.conditional_support_cdf_distance:.4f}',
          f'{row.core_fraction_difference:+.4f}',f'{row.limit_ratio_to_direct:.4f}',
          f'{row.delta_r_to_direct:+.4f}','pass' if passed else 'fail'])
    table('loo',[r'$m_0$',r'$D_{\rm CDF}$',r'$\Delta F_{\rm core}$','UL ratio',r'$\Delta r$','Shape check'],rows,'rrrrrl')
    rows=[]
    for row in precision.itertuples():
        rows.append([f'{row.mass_MeV:g}',f'{int(row.toys)}',f'{row.ratio_q16:.4f}',f'{row.ratio_median:.4f}',
                     f'{row.ratio_q84:.4f}',f'{100*row.relative_limit_std:.3f}'])
    table('precision',[r'$m_0$','Resamples',r'$R_{16}$',r'$R_{50}$',r'$R_{84}$','UL spread (\%)'],rows,'rrrrrr')
    passed=loo.shape_check_pass.astype(str).str.lower().isin(['true','1'])
    summary=(f'{int(passed.sum())} of {len(loo)} held-out shape checks pass the declared '
      f'$D_{{\\rm CDF}}\\leq0.03$ and $|\\Delta F_{{\\rm core}}|\\leq0.02$ criteria. '
      f'The largest conditional-support CDF distance is {loo.conditional_support_cdf_distance.max():.4f}. '
      f'The held-out limit ratios to direct MC range from {loo.limit_ratio_to_direct.min():.4f} '
      f'to {loo.limit_ratio_to_direct.max():.4f}. The dense curve remains exploratory regardless of this outcome.\n')
    (D/'report_morph_validation.tex').write_text(summary)
    print('Report arrays consumed; three figure pairs and five tables generated. No fits performed.')

if __name__=='__main__':main()
