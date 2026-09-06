#!/usr/bin/env python3
"""Figures and a portable LaTeX addition, with an isolated augmented report."""
from pathlib import Path
import hashlib,json,os,re,subprocess
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'qa/mpl_cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
BLUE,GRAY,GREEN,GOLD,RED,PURPLE='#166A9B','#62676B','#25816A','#BF871B','#BA4939','#80529B'
OUT=ROOT/'output/pdf'/HERE.name
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':190,'axes.labelsize':11})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save(fig,name):
    for ext in ('pdf','png'):fig.savefig(HERE/'figures'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)
def arr(method,m):return np.load(HERE/'derived/fits'/f'{method}_m{m:05.2f}.npz')
def row(method,m):return ALL[(ALL.method==method)&(ALL.mass_MeV==m)].iloc[0]
def pstr(v):return f'{v:.3f}' if v>=.001 else f'{v:.2g}'

SCAN=pd.read_csv(HERE/'derived/scan.csv')
STABLE=pd.read_csv(HERE/'derived/kernel_stability.csv')
ALL=pd.concat([SCAN,STABLE],ignore_index=True)
SUMMARY=json.loads((HERE/'derived/summary.json').read_text())
TOYS={8:json.loads((HERE/'derived/toy_summary.json').read_text()),
      16:json.loads((HERE/'derived/toy_summary_ceiling16.json').read_text())}

def overview():
    d=pd.read_csv(HERE/'derived/input_histogram.csv')
    fig,ax=plt.subplots(figsize=(10.6,4.2))
    x=(d.left_MeV+d.right_MeV)/2;use=(x>=10)&(x<=32)
    ax.axvspan(12,28,color=BLUE,alpha=.085,label='Short GP support: 12–28 MeV')
    ax.axvspan(15,20,color=GREEN,alpha=.14,label='Requested search: 15–20 MeV')
    ax.errorbar(x[use],d.counts[use],yerr=np.sqrt(d.counts[use]),fmt='.',color='.2',ms=4,
        elinewidth=.6,label='2015 full data: 0.25 MeV bins')
    for v in (12,28):ax.axvline(v,color=BLUE,lw=1)
    ax.axvline(19,color=RED,lw=1.2,ls='--',label='Established scan starts at 19 MeV')
    ax.set(yscale='log',ylim=(.7,5e5),xlim=(10,32),xlabel=r'$e^+e^-$ invariant mass [MeV]',
        ylabel='Events / 0.25 MeV',title='2015: a short support isolates the rising edge')
    ax.legend(loc='lower right',fontsize=9.2,frameon=False)
    ax.grid(axis='y',alpha=.17);fig.tight_layout()
    save(fig,'rising_edge_support')

def scans():
    fig,axes=plt.subplots(2,1,figsize=(10.6,7),sharex=True,
        gridspec_kw={'height_ratios':[1,1.15],'hspace':.12})
    top,bot=axes
    choices=[('gp_12_26','#C3A077',':','12–26, ceiling 8'),
        ('gp_12_30','#A891BC',':','12–30, ceiling 8'),
        ('gp_12p5_28','#6BA596',':','12.5–28, ceiling 8'),
        ('gp_12_28',GRAY,'--','12–28, ceiling 8'),
        ('gp_ceiling16',BLUE,'-','12–28, ceiling 16 (stable optimum)'),
        ('expcheb5',RED,'-.','Local exp(Chebyshev-5)')]
    for method,col,ls,label in choices:
        d=ALL[ALL.method==method].sort_values('mass_MeV')
        lw=1.9 if method in ('gp_ceiling16','expcheb5') else 1.1
        top.plot(d.mass_MeV,d.r,color=col,ls=ls,lw=lw,label=label)
        bot.plot(d.mass_MeV,np.maximum(d.p0,1e-6),color=col,ls=ls,lw=lw)
        below=d.p0<1e-6
        bot.scatter(d.loc[below,'mass_MeV'],np.full(sum(below),1e-6),color=col,marker='v',s=25,clip_on=False)
    legacy=pd.read_csv(ROOT/'study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv')
    legacy=legacy[(legacy.scope_key=='individual_2015_full')&(legacy.mass_MeV>=19)&(legacy.mass_MeV<=22)]
    top.scatter(legacy.mass_MeV,legacy.signed_r_profiled_asymptotic,color='black',marker='x',s=35,zorder=5,label='Frozen wide-support GP (integer masses)')
    bot.scatter(legacy.mass_MeV,norm.sf(np.maximum(legacy.signed_r_profiled_asymptotic,0)),color='black',marker='x',s=35,zorder=5)
    for ax in axes:
        ax.axvspan(20,22,color='.91',zorder=-5)
        ax.axvline(19,color='.65',ls=':',lw=1)
        ax.axvline(20,color='.6',lw=1)
        ax.set_xlim(15,22);ax.grid(axis='y',alpha=.15)
    top.axhline(0,color='.3',lw=.7)
    top.set_ylabel(r'Signed profile root $r$')
    top.legend(loc='upper left',fontsize=8.2,ncol=2,frameon=False)
    bot.set(yscale='log',ylim=(1e-6,.8),ylabel=r'Local asymptotic $p_0$',xlabel='Mass hypothesis [MeV]')
    for z in (1,2,3):
        bot.axhline(norm.sf(z),lw=.6,color='.75',ls=':')
        bot.text(15.04,norm.sf(z)*.78,f'{z}σ reference',fontsize=8,color='.45')
    top.text(.79,.08,'20–22 MeV:\ncontext only',transform=top.transAxes,fontsize=9,color='.35')
    top.set_ylim(min(ALL.r.min()-.4,-4.5),max(ALL.r.max()+1,7.3))
    fig.suptitle('15–20 MeV: modest excesses; background choice matters',fontsize=14,y=.986)
    fig.text(.5,.94,'GP sidebands exclude ±2.25σ  |  No global or detector-response calibration below 19 MeV',
        ha='center',fontsize=9.4,color='.35')
    fig.subplots_adjust(top=.88,left=.10,right=.98,bottom=.085)
    save(fig,'local_pvalues_and_supports')

def extraction(masses,name,title):
    fig=plt.figure(figsize=(10.6,6.8));grid=fig.add_gridspec(2,len(masses),height_ratios=[1.55,1],
        left=.085,right=.985,bottom=.19,top=.80,hspace=.12,wspace=.27)
    mappings=[]
    for j,m in enumerate(masses):
        d=arr('gp_ceiling16',m);r=row('gp_ceiling16',m);e=d['edges'];n=d['n'];x=d['x'];mask=d['mask']
        group=max(1,int(np.floor(.5*r.sigma_MeV/.25+.5)))
        # All groups are anchored to the original zero edge, never the fitted peak.
        start_index=np.rint(e[:-1]/.25).astype(int)
        gids=start_index//group
        W=[];le=[];ri=[];inside=[]
        for gid in sorted(set(gids)):
            ii=np.flatnonzero(gids==gid)
            if len(ii)!=group:continue
            left,right=e[ii[0]],e[ii[-1]+1]
            if right<m-3.3*r.sigma_MeV or left>m+3.3*r.sigma_MeV:continue
            w=np.zeros(len(n));w[ii]=1;W.append(w);le.append(left);ri.append(right);inside.append(bool(mask[ii].all()))
        W=np.array(W);le=np.array(le);ri=np.array(ri);xx=(le+ri)/2;inside=np.array(inside)
        nn=W@n;bp=W@d['bprior'];C=W@d['C']@W.T
        bs=W@np.nan_to_num(d['bfit']);tot=W@np.nan_to_num(d['total']);ss=W@d['signal']
        cov=W@np.diag(n)@W.T
        np.testing.assert_allclose(cov,np.diag(nn))
        np.testing.assert_allclose((bs+ss)[inside],tot[inside],atol=1e-8)
        top=fig.add_subplot(grid[0,j]);bot=fig.add_subplot(grid[1,j],sharex=top)
        for ax in (top,bot):
            ax.axvspan(r.fit_low_MeV,r.fit_high_MeV,color=BLUE,alpha=.055)
            ax.axvline(m,color='.6',ls=':',lw=.9)
            ax.grid(axis='y',alpha=.14)
            ax.set_xlim(m-3.2*r.sigma_MeV,m+3.2*r.sigma_MeV)
        top.errorbar(xx,nn,yerr=np.sqrt(nn),xerr=(ri-le)/2,fmt='o',color='.15',ms=3.3,elinewidth=.85,label='Data')
        top.step(xx,bp,where='mid',color=GOLD,lw=1.4,ls='--',label='Sideband GP mean')
        top.fill_between(xx,np.maximum(0,bp-np.sqrt(np.maximum(np.diag(C),0))),bp+np.sqrt(np.maximum(np.diag(C),0)),
            step='mid',color=GOLD,alpha=.14,label='GP constraint ±1 SD')
        top.step(xx,np.where(inside,bs,np.nan),where='mid',color=GREEN,lw=1.6,label='Background profiled with signal')
        top.step(xx,np.where(inside,tot,np.nan),where='mid',color=BLUE,lw=1.6,label='Signal + background')
        top.set_ylabel(f'Events / {group*.25:g} MeV')
        top.set_title(f'{m:g} MeV  |  local p₀ = {r.p0:.3f}',fontsize=11.3,pad=9)
        top.ticklabel_format(axis='y',style='sci',scilimits=(3,3));top.tick_params(labelbottom=False)
        top.yaxis.set_major_locator(MaxNLocator(5))
        bot.errorbar(xx[inside],(nn-bs)[inside],yerr=np.sqrt(nn[inside]),xerr=(ri-le)[inside]/2,
            fmt='o',color='.15',ms=3.5,elinewidth=.9)
        bot.step(xx,np.where(inside,ss,np.nan),where='mid',color=BLUE,lw=1.8)
        bot.axhline(0,color='.4',lw=.75)
        bot.set(xlabel=r'$e^+e^-$ invariant mass [MeV]',ylabel='Data − background')
        bot.yaxis.set_major_locator(MaxNLocator(5))
        bot.text(.04,.94,f'Fitted Gaussian yield: {r.Ahat_total:,.0f} ± {r.sigma_A_total:,.0f}',
            transform=bot.transAxes,va='top',fontsize=8.4,color='.25')
        mappings.append(dict(mass_MeV=m,group_native_inference_bins=group,display_width_MeV=group*.25,
            retained_fit_counts=float(nn[inside].sum()),all_fit_counts=float(n[mask].sum()),
            full_groups_only=True,fixed_zero_edge_phase=True))
        np.savez_compressed(HERE/'derived'/f'display_mapping_m{m:05.2f}.npz',W=W,left=le,right=ri,
            inside=inside,count_covariance=cov,background_constraint_covariance=C,counts=nn,signal=ss,
            profiled_background=bs)
    handles,labels=top.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,fontsize=8.8,frameon=False)
    fig.suptitle(title,fontsize=14,y=.995)
    fig.text(.5,.060,'2015 full  |  GP support 12–28 MeV, stable ceiling 16  |  Shading: fitted signal window',
        ha='center',fontsize=9,color='.3')
    fig.text(.5,.022,'Counting error bars only. Residuals share fitted-background uncertainty; they are not independent significances.',
        ha='center',fontsize=8.6,color='.3')
    save(fig,name)
    return mappings

def toy_figure():
    fig,ax=plt.subplots(figsize=(10.6,4.2))
    for upper,col,shift in ((8,GRAY,-.035),(16,BLUE,.035)):
        d=pd.DataFrame(TOYS[upper]['anchors']);xx=d.mass_MeV+shift
        ax.errorbar(xx,d.p_hat,yerr=np.array([d.p_hat-d.low95,d.high95-d.p_hat]),
            fmt='o',color=col,ms=5,capsize=3,label=f'100 conditional toys / mass, ceiling {upper}')
        method='gp_12_28' if upper==8 else 'gp_ceiling16'
        pp=[row(method,m).p0 for m in d.mass_MeV]
        ax.scatter(xx,pp,color=col,marker='x',s=50,label=f'Asymptotic reference, ceiling {upper}')
    ax.set(xlim=(14.7,20.3),ylim=(0,.85),xlabel='Fixed mass hypothesis [MeV]',ylabel='Local excess probability',
        title='Small conditional toy checks expose shifts in the null statistic')
    ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,loc='upper left',fontsize=9)
    fig.tight_layout();save(fig,'conditional_toy_checks')

def section():
    table=[]
    for m in SUMMARY['extraction_masses_MeV']:
        a,b,p=row('gp_12_28',m),row('gp_ceiling16',m),row('expcheb5',m)
        table.append(f'{m:g} & {a.r:.2f} & {b.r:.2f} & {b.p0:.3f} & {p.p0:.3f} & ${b.Ahat_total:,.0f}\\pm{b.sigma_A_total:,.0f}$ \\\\')
    trows=[]
    for upper in (8,16):
        for v in TOYS[upper]['anchors']:
            lower=f"{v['low95']:.5f}" if 0<v['low95']<.001 else f"{v['low95']:.3f}"
            trows.append(f"{upper} & {v['mass_MeV']:g} & {v['k']}/{v['n']} & {v['p_hat']:.2f} & [{lower}, {v['high95']:.3f}] & {v['mean_r']:+.2f} & {v['sd_r']:.2f} \\\\")
    main=r'''\section{Exploratory 2015 search on the 15--20 MeV rising edge}
\label{sec:2015lowmass}
\textbf{Result.} This short-support study does not reveal a persuasive narrow excess in 15--20 MeV. The largest nominal-GP upward fluctuation is at 17.25 MeV. Releasing an active smoothness ceiling gives a stable fit with $r=1.58$ and local asymptotic $p_0=0.057$; the independently fitted polynomial background gives $p_0=0.254$ there. These values are model-conditional, selected from a scan, and do not establish or exclude a heavy photon.

\subsection*{Why a separate support is useful}
The released 2015 scan begins at 19 MeV, with GP training support 14--135 MeV. Its input histogram extends below that support. The spectrum contains 81 events in 12--13 MeV, 7,653 in 14--15 MeV and 330,090 in 19--20 MeV. A local GP can isolate this turn-on without fitting the distant falling spectrum. It must still describe the steep background accurately across a missing signal window.

\begin{figure}[H]\centering
\includegraphics[width=.97\linewidth]{../figures/rising_edge_support.pdf}
\caption{The same released full-2015 histogram, summed from 0.05 to 0.25 MeV bins. The chosen GP support is 12--28 MeV; the requested search is 15--20 MeV. The dashed line marks the lower boundary of the established scan, not the lower boundary of the available counts.}\end{figure}

The new scan has 21 hypotheses, separated by 0.25 MeV. A labelled continuation to 22 MeV connects it to the previously selected 21 MeV point; that continuation is excluded from selection of the leading 15--20 MeV feature. The comparison supports 12--26, 12--30 and 12.5--28 MeV were declared before examining the new signal fits. Their p-values do not select the nominal support.

\textbf{Detector boundary.} The inherited Gaussian width is $\sigma_m=-0.09223+0.053219m$ MeV for $m$ in MeV, giving about 0.71--0.97 MeV here. Below 19 MeV it is an extrapolated shape assumption. The published 2015 prompt search covered 19--81 MeV~\cite{lowmasshps}. Low-mass signal efficiency, acceptance and the selected signal shape need direct simulation and control checks before a physical coupling result can be quoted.

\clearpage
\subsection*{Local probabilities and dependence on the background}
The GP fits log counts versus log mass outside $m\pm2.25\sigma_m$. Its prediction and correlated count covariance constrain the background in an exact Poisson window likelihood. Both the signal amplitude and background nuisance coordinates are fitted; the auxiliary amplitude may be negative while every total expectation remains positive. We report
\[
r=\operatorname{sign}(\widehat A)\sqrt{2[\mathrm{NLL}(0)-\mathrm{NLL}(\widehat A)]},
\qquad p_0=\overline\Phi\!\left(\max(r,0)\right).
\]
Thus the asymptotic convention assigns $p_0=0.5$ to negative fits~\cite{cowan}. No GP estimate of a global trials factor is used in this section.

\begin{figure}[H]\centering
\includegraphics[width=.98\linewidth]{../figures/local_pvalues_and_supports.pdf}
\caption{Signed roots and local asymptotic excess probabilities. All four initial GP supports reach the inherited ceiling of eight resolution units. Raising the nominal-support ceiling to 16 removes every active boundary; ceilings 32 and 64 recover the same optima, with maximum root changes below 0.00054. The local polynomial is a separate background-family check. Gray shading identifies the 20--22 MeV bridge. Triangles mark the $10^{-6}$ display floor, not resolved global tails.}\end{figure}

For the polynomial check, a positive $\exp(\mathrm{Chebyshev}_5)$ background and Gaussian signal are integrated over each 0.25 MeV bin. All background coefficients are profiled over a moving $\pm7\sigma_m$ support, clipped at 12 MeV. Its form and window scale follow the published low-mass procedure~\cite{lowmasshps}; its suitability below 19 MeV is tested here rather than assumed. In 15--20 MeV its largest positive root is only 0.74. At 21 MeV it gives a much larger root, 6.15, but a signal-plus-background Poisson deviance of 98.7 for 51 nominal degrees of freedom. That poor-fit diagnostic and the GP dependence prevent interpreting the large value as a robust signal.

\clearpage
\subsection*{Signal extraction at fixed low-mass anchors}
The displays use the stable ceiling-16 GP on 12--28 MeV. The 15 and 17 MeV hypotheses were fixed before the scan. Curves for profiled components are confined to the fitted window. Outside it, only data and the sideband GP constraint are shown. The gold uncertainty is the GP constraint before fitting the window, not an uncertainty band for the final fitted background.

\begin{figure}[H]\centering
\includegraphics[width=.99\linewidth]{../figures/extractions_15_17.pdf}
\caption{Fixed 15 and 17 MeV hypotheses. Top: data, sideband background constraint, profiled background and signal-plus-background prediction. Bottom: data minus the background profiled with signal, with the fitted Gaussian contribution. The error bars describe counting uncertainty only. The subtraction uses the same data to fit the background and is not an independent pull test.}\end{figure}

The likelihood uses 0.25 MeV bins. For these figures, complete groups of bins are chosen nearest to half a mass resolution and anchored at the original zero edge. The resulting display widths are 0.25 MeV at 15 MeV and 0.50 MeV at the other displayed masses. Boundary groups crossing the fitted window are omitted from the profiled display, while all original window bins remain in the likelihood. No grouping or phase is chosen to make a peak sharper.

\begin{center}\small
\begin{tabular}{rrrrrr}\toprule
$m$ [MeV] & $r$, ceiling 8 & $r$, ceiling 16 & $p_0$, ceiling 16 & $p_0$, polynomial & Fitted yield, ceiling 16\\\midrule
__FIT_TABLE__
\bottomrule\end{tabular}\end{center}
Yield errors are local curvature errors for the assumed Gaussian template; they are not post-selection confidence intervals. No conversion to $\epsilon^2$ is made.

\clearpage
\subsection*{The leading fluctuation and the upper edge of the requested interval}
At 17.25 MeV the stable GP fit gives approximately 2,542$\pm$1,612 Gaussian-template events. Its maximum is selected after scanning 15--20 MeV, so this amplitude can be upward biased. It does not acquire a stronger interpretation because the scan was restricted to a rising edge. The 20 MeV panel supplies the fixed upper anchor.

\begin{figure}[H]\centering
\includegraphics[width=.99\linewidth]{../figures/extractions_peak_20.pdf}
\caption{The selected 17.25 MeV upward fluctuation and the fixed 20 MeV hypothesis, with the same conventions and fixed-phase grouping as the preceding displays. Their local asymptotic roots are 1.58 and 0.16. A convincing resonance would need a stable excess across defensible background choices and a validated accepted signal shape.}\end{figure}

At 21 MeV, the nominal GP changes from $r=1.46$ to $r=3.24$ when its active smoothness restriction is removed; the frozen wide-support fit has $r\simeq2.52$. This is evidence that the inference is sensitive to the background description. It is not evidence that the low-mass study has uncovered a second established particle feature. Resolving this dependence requires predictive background controls and detector-level checks, not choosing the smallest p-value.

\clearpage
\subsection*{Small conditional toy checks and what remains to establish}
Ten pilot spectra were first generated at each of the four displayed masses, then extended to 100 at each mass for each of ceilings 8 and 16: 800 distinct mass/model toy fits in total. The first ten are included in each 100. Each Poisson spectrum covers the complete 12--28 MeV support, and its GP hyperparameters and background are fitted again. The generating mean is the continuous sideband-conditioned GP prediction at that mass, without an observed-window adjustment. These mass-local backgrounds cannot be joined into one global experiment.

\begin{figure}[H]\centering
\includegraphics[width=.88\linewidth]{../figures/conditional_toy_checks.pdf}
\caption{Toy upper-tail fractions with central 95\% binomial intervals, compared with the asymptotic local reference. The two ceiling choices have their own conditional generating backgrounds. These small plug-in ensembles assess this fitting procedure under those means; they are not independent background validation or a final p-value calibration.}\end{figure}

\begin{center}\small
\begin{tabular}{rrrrrrr}\toprule
Ceiling & $m$ [MeV] & Tails/toys & Fraction & 95\% interval & Mean $r$ & SD $r$\\\midrule
__TOY_TABLE__
\bottomrule\end{tabular}\end{center}

The shifts of the toy-root means away from zero demonstrate why an asymptotic reference alone is insufficient to qualify this extension. A small toy-tail fraction is conditional on a data-derived mean and has finite sampling uncertainty. It does not remove the choice of mass, support or background family.

Next, validate accepted signal templates and the turn-on; test background prediction and signal recovery under alternative smooth truths; fix choices independently of signal-window residuals; and calibrate a coherent scan for a global probability. A shorter GP support does not remove look-elsewhere accounting. The accompanying study bundle retains scripts, fit components, display maps, 800 toy roots and source hashes.
'''
    main=main.replace('__FIT_TABLE__','\n'.join(table)).replace('__TOY_TABLE__','\n'.join(trows))
    main=r'\providecommand{\lowmassfigurepath}{../figures}'+'\n'+main.replace('{../figures/','{\\lowmassfigurepath/')
    # The published polynomial number is tied to the actual row, never a prose-only input.
    main=main.replace('$p_0=0.254$',f"$p_0={row('expcheb5',17.25).p0:.3f}$")
    return main

HPS_BIB=r'\bibitem{lowmasshps} P.~H.~Adrian et al. (HPS), \emph{Search for a Dark Photon in Electro-Produced $e^+e^-$ Pairs with the Heavy Photon Search Experiment at JLab}, Phys. Rev. D \textbf{98} (2018) 091101. \href{https://arxiv.org/abs/1807.11530}{arXiv:1807.11530}.'
COWAN_BIB=r'\bibitem{cowan} G.~Cowan, K.~Cranmer, E.~Gross and O.~Vitells, \emph{Asymptotic formulae for likelihood-based tests of new physics}, Eur. Phys. J. C \textbf{71} (2011) 1554. \href{https://arxiv.org/abs/1007.1727}{arXiv:1007.1727}.'

def build(tex,name):
    p=HERE/'note'/tex
    result=subprocess.run(['tectonic','--only-cached','--keep-logs','--outdir',str(OUT),str(p)],
        cwd=HERE/'note',capture_output=True,text=True)
    (HERE/'note'/f'{p.stem}_build.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    generated=OUT/(p.stem+'.pdf');final=OUT/name
    final.write_bytes(generated.read_bytes());generated.unlink()
    return final

def augment():
    candidates=['v4p9p16_presentation_extractions_20260906','v4p9p16_deficit_extension_20260906']
    source=next(ROOT/'study_results'/name/'note/analysis_note.tex' for name in candidates
        if (ROOT/'study_results'/name/'note/analysis_note.tex').exists())
    notes=HERE/'note';copied={};hashes={}
    assets=HERE/'inherited_figures';assets.mkdir(exist_ok=True)
    def convert(path):
        path=path.resolve()
        if path in copied:return copied[path]
        name=f'inherited_{len(copied):02d}_{path.name}';copied[path]=name
        text=path.read_text();hashes[str(path.relative_to(ROOT))]=sha(path)
        def inc(match):
            target=path.parent/match.group(1)
            if not target.suffix:target=target.with_suffix('.tex')
            return r'\input{'+convert(target)+'}'
        text=re.sub(r'\\input\{([^}]+)\}',inc,text)
        def graphic(match):
            target=(path.parent/match.group(1)).resolve()
            data=target.read_bytes();digest=hashlib.sha256(data).hexdigest()
            hashes[str(target.relative_to(ROOT))]=digest
            dest=assets/(digest[:12]+'_'+target.name);dest.write_bytes(data)
            return '{'+os.path.relpath(dest,notes)+'}'
        text=re.sub(r'\{([^{}]+\.(?:pdf|png))\}',graphic,text)
        (notes/name).write_text(text)
        return name
    mainfile=convert(source);text=(notes/mainfile).read_text()
    marker=r'\begin{thebibliography}'
    assert text.count(marker)==1
    text=text.replace(marker,r'\clearpage'+'\n'+r'\input{lowmass_section.tex}'+'\n'+r'\clearpage'+'\n'+marker)
    text=text.replace(r'\end{thebibliography}',HPS_BIB+'\n'+r'\end{thebibliography}')
    text=text.replace(r'\begin{document}',r'\hypersetup{pdftitle={HPS GPR v4.9.16 with exploratory 2015 low-mass study}}'+'\n'+r'\begin{document}')
    (notes/'analysis_note_with_lowmass.tex').write_text(text)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest
    pdf=build('analysis_note_with_lowmass.tex','HPS_GPR_v4p9p16_with_2015_LowMass_Study.pdf')
    dump(HERE/'provenance/report_parent.json',dict(source=str(source.relative_to(ROOT)),source_sha256=hashes,
        policy='Isolated augmented copy; active parent sources and outputs are untouched.',pdf=str(pdf.relative_to(ROOT))))
    return pdf

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    overview();scans();maps=[]
    maps+=extraction([15.,17.],'extractions_15_17','2015 signal extraction: fixed 15 and 17 MeV hypotheses')
    maps+=extraction([17.25,20.],'extractions_peak_20','2015 signal extraction: the leading fluctuation and 20 MeV')
    toy_figure();dump(HERE/'derived/display_summary.json',maps)
    (HERE/'note/lowmass_section.tex').write_text(section())
    preamble=r'''\documentclass[11pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,xcolor,xurl,float}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\usepackage[font=small]{caption}
\hypersetup{pdftitle={HPS GPR v4.9.16: 2015 low-mass rising-edge study},pdfauthor={Emrys Peets}}
\begin{document}
\begin{center}{\Large HPS GPR analysis note v4.9.16}\\[3pt]
{\large 2015 low-mass rising-edge side study}\\[3pt]6 September 2026\end{center}
\input{lowmass_section.tex}
\begin{thebibliography}{9}
'''
    (HERE/'note/standalone.tex').write_text(preamble+HPS_BIB+'\n'+COWAN_BIB+'\n'+r'\end{thebibliography}'+'\n'+r'\end{document}'+'\n')
    stand=build('standalone.tex','HPS_GPR_v4p9p16_2015_LowMass_Section.pdf')
    full=augment()
    dump(HERE/'provenance/report_build.json',dict(pdfs={str(p.relative_to(ROOT)):sha(p) for p in (stand,full)},
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'derived/scan.csv',
            HERE/'derived/kernel_stability.csv',HERE/'derived/toy_summary.json',HERE/'derived/toy_summary_ceiling16.json']},
        figures={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'figures').glob('*.*')}))
    print(stand);print(full)

if __name__=='__main__':main()
