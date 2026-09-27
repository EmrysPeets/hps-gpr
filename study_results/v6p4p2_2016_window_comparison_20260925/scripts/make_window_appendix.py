"""Figures and concise four-page appendix for the 2016 window comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];R=B/'results/window_comparison';F=B/'figures'
BLUE='#245c91';ORANGE='#b37524'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'legend.fontsize':8,
                    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS={3.5:BLUE,2.:ORANGE};LABELS={3.5:r'2016 $\pm3.5u$ (main analysis)',2.:r'2016 $\pm2u$ (comparison)'}
def save(fig,name):
    for ext in ('pdf','png'):fig.savefig(F/f'window_{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def caption(s):return r'{\small\refstepcounter{figure}\textbf{Figure \thefigure. How to read the figure.} '+s+r'}\par\medskip'+'\n'
def plot(name,height):return r'\begin{center}\includegraphics[width=\linewidth,height='+height+r'in,keepaspectratio]{../figures/window_'+name+r'.pdf}\end{center}'+'\n'

def main():
    obs=pd.read_csv(R/'observed_scan.csv');geom=pd.read_csv(R/'geometry.csv');resp=pd.read_csv(R/'signal_response.csv');null=pd.read_csv(R/'null_checks.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.4),layout='constrained')
    for h in (3.5,2.):
        q=geom[geom.half_width_u==h].sort_values('mass_MeV')
        axs[0].plot(q.mass_MeV,100*q.fit_fraction,color=COLORS[h],label=LABELS[h])
        axs[1].plot(q.mass_MeV,100*q.training_fraction,color=COLORS[h],label=LABELS[h])
    axs[0].set(ylabel='Full selected MC in fitted bins (%)',title='Signal retained in the fit')
    axs[1].set(ylabel='Full selected MC in training bins (%)',title='Signal outside the exclusion')
    for ax in axs:ax.set_xlabel('Generated mass (MeV)');ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig,'geometry')
    for scope in ('2016','combined'):
        fig,axs=plt.subplots(3,1,figsize=(7.2,5.8),sharex=True,gridspec_kw={'height_ratios':[1,0.65,1]})
        q=obs[obs.scope==scope];col='A90' if scope=='2016' else 'epsilon2_90_visible_legacy'
        for h in (3.5,2.):
            x=q[q.half_width_u==h].sort_values('mass_MeV')
            axs[0].plot(x.mass_MeV,x[col],color=COLORS[h],label=LABELS[h]);axs[2].plot(x.mass_MeV,x.p0_asymptotic,color=COLORS[h])
        p=q.pivot(index='mass_MeV',columns='half_width_u',values=col)
        axs[1].plot(p.index,p[2.]/p[3.5],color=ORANGE);axs[1].axhline(1,color='.5',ls=':',lw=.8)
        axs[0].set_yscale('log');axs[2].set_yscale('log');axs[0].legend(frameon=False,ncol=1,fontsize=8)
        axs[0].set_ylabel('90% CLs limit\n'+('full selected yield' if scope=='2016' else 'conditional ε² display'))
        axs[1].set_ylabel('Limit ratio\n2u / 3.5u');axs[2].set(ylabel='Local excess p\n(asymptotic)',xlabel='Generated mass hypothesis (MeV)',ylim=(q.p0_asymptotic.min()*.55,.7))
        for ax in axs:
            ax.grid(axis='y',alpha=.15);ax.set_xlim((40,175) if scope=='2016' else (60,240))
            if scope=='combined':
                for x in (100.5,175.5):ax.axvline(x,color='.6',ls=':',lw=.7)
        fig.subplots_adjust(left=.16,right=.98,top=.98,bottom=.09,hspace=.14)
        save(fig,scope)
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.2),layout='constrained')
    for h in (3.5,2.):
        q=resp[resp.half_width_u==h].sort_values('mass_MeV')
        axs[0].errorbar(q.mass_MeV,q.response,yerr=q.response_mc_se,color=COLORS[h],fmt='o-',ms=4,label=LABELS[h])
    axs[0].axhline(1,color='.5',ls=':',lw=.8);axs[0].set(ylabel='Mean added fitted yield / expected signal',title='Response to the same full MC injection',ylim=(.85,1.035));axs[0].legend(frameon=False)
    a=resp.pivot(index='mass_MeV',columns='half_width_u',values='null_sd_Ahat');b=resp.pivot(index='mass_MeV',columns='half_width_u',values='response_adjusted_null_sd')
    axs[1].plot(a.index,a[2]/a[3.5],'o--',color='.45',label='Background spread, uncorrected')
    axs[1].plot(b.index,b[2]/b[3.5],'s-',color=ORANGE,label='Spread divided by signal response')
    axs[1].axhline(1,color='.5',ls=':',lw=.8);axs[1].set(ylabel='2u / 3.5u',title='Smaller fitted spread versus signal response',ylim=(.85,1.07));axs[1].legend(frameon=False)
    for ax in axs:ax.set_xlabel('Generated mass (MeV)');ax.grid(alpha=.15)
    save(fig,'response')
    s=r'''\clearpage\section*{Appendix A. Test a narrower 2016 window: $\pm2u$}
\label{sec:windowappendix}

\textbf{The finding.} Reducing the 2016 MC fit and GP-training exclusion from $\pm3.5u$ to $\pm2u$ lowers most conditional observed profile limits. However, it also reduces the fitted response to an injected full MC signal. In the five tested injections, the narrower window recovers 87.7--92.2\% of the added selected yield, compared with 99.8--100.1\% for the main window. Its smaller raw fit uncertainty therefore does not establish an improvement in signal sensitivity.

\textbf{A controlled window comparison.} Both choices use the same neighboring MC template, $u=(m_{\rm rec}-c)/s$, and the same interpolated core center and width. Only the interval changes. For the comparison, bins with centers satisfying $|u|\le2$ enter the signal likelihood and are excluded from GP training. All other bins enter training. There is no separate extra exclusion. The full selected template normalization, GP kernel settings and inherited coupling conversion are retained; the GP mean and covariance are recomputed for the new sidebands.
'''
    s+=plot('geometry','2.65')+caption(r'Blue is the main $\pm3.5u$ prescription; gold is $\pm2u$. Left: full selected MC probability in the actual analysis bins used for fitting. Right: probability in the GP-training bins. Bin membership follows bin centers, so the plotted fractions include the finite analysis binning. These are geometric fractions, not detector efficiencies or fitted recovery fractions.')
    s+=r'''
The narrower fitted bins contain 88.68--93.66\% of the full selected signal over 40--175 MeV, compared with 97.93--99.58\% in the main window. Consequently, 6.34--11.32\% can enter training rather than 0.42--2.07\%. The tail probability is retained in the injections. It is never dropped or renormalized into the fitted interval.

\begin{center}\small\begin{tabular}{rrrrr}\toprule
Mass [MeV] & Window & Actual fit interval [MeV] & Fit bins & MC fraction [\%]\\\midrule
'''
    for m in (69,91):
        for h in (3.5,2.):
            q=geom[(geom.mass_MeV==m)&(geom.half_width_u==h)].iloc[0]
            s+=f'{m} & $\\pm{h:g}u$ & {q.actual_low_MeV:.2f}--{q.actual_high_MeV:.2f} & {int(q.fit_bins)} & {100*q.fit_fraction:.2f}'+r'\\'+'\n'
    s+=r'''\bottomrule\end{tabular}\end{center}
The main-body results remain the $\pm3.5u$ analysis. This appendix tests an alternative; it does not replace that prescription.

\clearpage\section*{Appendix A.1. The 2016 observed scan}
The two windows are evaluated at the same 136 hypotheses, 40--175 MeV in 1 MeV steps. Limits continue to refer to the full selected MC yield. The signal shape itself is identical in the two fits.
'''
    s+=plot('2016','4.55')+caption(r'Top: conditional observed 90\% profile-$CL_s$ upper limits for the two windows. Middle: the $\pm2u$ limit divided by the $\pm3.5u$ limit at the same mass. Bottom: fixed-mass local asymptotic excess probabilities. These curves use the likelihood and conversion defined in the main text; they are not full-scan toy-calibrated limits or global probabilities.')
    s+=r'''
The median limit change is $-9.64\%$, with a range of $-20.63\%$ at 41 MeV to $+0.05\%$ at 82 MeV. At 91 MeV, the strongest local feature under both choices, the fitted yield changes from $20{,}201\pm5{,}699$ to $17{,}815\pm5{,}221$ events. The upper limit falls from 27,505 to 24,507 events, while the local asymptotic probability increases from 0.000196 to 0.000321 ($Z=3.55$ to 3.41).

At the other previously selected region, 69 MeV, the fitted yield changes from $16{,}791\pm7{,}371$ to $16{,}115\pm6{,}705$ events, the upper limit from 26,285 to 24,739 events, and the local asymptotic probability from 0.01136 to 0.00812. These fixed-mass comparisons keep the two earlier examples; they do not select a new pair of independent excesses.

\clearpage\section*{Appendix A.2. Propagate only the 2016 window change}
This comparison starts from the final main-body combination: 2015 Gaussian, 2016 neighboring MC and 2021 neighboring MC. The 2021 interval remains $[-4,+3]u$. Campaign availability, observations, full signal probabilities and coupling conversion are identical; only the 2016 fit/exclusion changes.
'''
    s+=plot('combined','4.55')+caption(r'Top: the conditional coupling-limit display for the main combination and the combination with 2016 narrowed to $\pm2u$. Middle: their ratio. Bottom: local asymptotic excess probabilities from the joint common-signal likelihood. Vertical dotted lines mark the ends of 2015 and 2016 participation. Above 175 MeV only 2021 contributes, so both curves coincide exactly. No new global probability or efficiency correction is inferred.')
    s+=r'''
Over 60--175 MeV, where 2016 contributes, the combined upper limit changes by a median $-1.36\%$. The range is $-17.83\%$ at 65 MeV to $+4.49\%$ at 112 MeV. The effect is smaller than the typical 2016-only shift because the common signal parameter must also describe the other included campaigns.

The strongest combined local feature stays at 68 MeV: $p_0$ changes only from 0.002475 to 0.002539 ($Z=2.810$ to 2.802). The narrower-window scan remains a conditional profile comparison. The main-body selected-point rank calibration used the wider 2016 prescription and is not transferred to this alternative.

\clearpage\section*{Appendix A.3. Does the narrower window recover the signal?}

At 60, 69, 91, 100 and 160 MeV, 200 paired experiments add the same full MC signal to the same fixed GP-mean background for both windows. The expected selected yield is $A=3s_0$, where $s_0$ is the wider-window MC fit's Hessian error on the fixed background mean. Each stored-bin and outside-support signal category is Poisson sampled. Background-only and signal-injected fits share their background draw; neither the observed spectrum nor a fitted null bias determines the injected yield.

The response below is $R=\langle\Ah_{s+b}-\Ah_b\rangle/A$. Pairing removes the background-only mean from this response measurement; no correction is applied to the observed fits.
'''
    s+=plot('response','2.65')+caption(r'Left: response to the same expected full selected signal; error bars are Monte Carlo standard errors of the paired mean. Right: the ratio of background-toy fitted-yield spreads, before and after dividing each spread by its response. The ratios are descriptive point estimates from 200 paired experiments; their few-percent differences are not precise sensitivity improvements. Neither panel measures coverage.')
    s+=r'''
The narrower response is 0.877--0.922, while the wider response is 0.998--1.001. Dividing the background fitted-yield spread by the response gives narrow/wide ratios of 0.998--1.030. Thus the smaller raw fitted spread is largely offset by the reduced response. Together with the extra signal in training bins, this is consistent with increased signal absorption by the background estimate. It supports retaining $\pm3.5u$ as the main choice in this bounded comparison.

\textbf{Local background check.} Both windows also use the same 1,000 fixed-source null draws at 69 and 91 MeV. At 91 MeV, $\pm2u$ has 1 exceedance, rank $2/1001=0.0020$, and a 95\% exact binomial interval [0.000025, 0.00556]; $\pm3.5u$ has 3, rank 0.0040, with [0.00062, 0.00874]. At 69 MeV the counts are 3 and 5, with ranks 0.0040 and 0.0060 and intervals [0.00062, 0.00874] and [0.00163, 0.01163]. These sparse, paired tails do not establish a significant difference between windows. The positive mean null root at 91 MeV remains: 0.601 versus 0.656.

\textbf{Record and scope.} The appendix adds 252 observed fits and 6,000 toy fits, reusing 2,000 matching main-body null rows. Complete signed yields, Hessian errors, pulls, realized signal partitions, pairing hashes and probability intervals are under \path{results/window_comparison}. This is a five-mass, one-strength, fixed-source response check, not an independent coverage validation or a new calibrated upper-limit construction. Rebuild the appendix with \path{rebuild_appendix.sh}; the parent numerical records remain unchanged.
'''
    (B/'source/window_appendix.tex').write_text(s)
    print('Wrote four appendix figures and four source pages.')

if __name__=='__main__':main()
