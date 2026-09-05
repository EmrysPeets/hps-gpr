#!/usr/bin/env python3
"""Create standalone figures and a compact report without touching parent notes."""
from pathlib import Path
import hashlib
import json
import os
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=HERE/'derived'
FIG=HERE/'figures'
PDF=REPO/'output/pdf/background_profile_comparison_20260905'
BLUE='#185781';ORANGE='#d26924';GREY='#7b7b7b';GREEN='#27856a'
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12,
    'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'grid.linewidth':.6,
    'pdf.fonttype':42,'savefig.facecolor':'white'})


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'),dpi=220)
    plt.close(fig)


def branching(m):
    m=np.asarray(m,float);out=np.ones_like(m);sel=m>2*105.6583745
    r=(105.6583745/m[sel])**2
    out[sel]+=np.sqrt(1-4*r)*(1+2*r)
    return out


def limits(d):
    fig,axs=plt.subplots(2,2,figsize=(11.8,7.0),sharex='col',
        gridspec_kw={'height_ratios':[2.1,1]})
    labels=['Released Gaussian profile','Direct log-GP profile','Fixed GP mean']
    specs=[('current',BLUE,'-',2.2),('log_gp',ORANGE,'--',1.8),('fixed',GREY,'-',1.4)]
    for j,(lo,hi,title) in enumerate(((50,250,'Full 2021 search range'),(50,100,'Low-mass detail'))):
        a,r=axs[:,j];sel=d.mass_MeV.between(lo,hi);f=d[sel]
        for (name,col,ls,lw),label in zip(specs,labels):
            a.plot(f.mass_MeV,f['eps2_'+name]*branching(f.mass_MeV),ls,color=col,lw=lw,label=label)
            if name!='current':r.plot(f.mass_MeV,f['eps2_'+name]/f.eps2_current,ls,color=col,lw=lw)
        a.set(yscale='log',title=title,xlim=(lo,hi))
        r.axhline(1,color=BLUE,lw=1)
        r.set(ylim=(0,1.5),xlabel=r'$m_{A\prime}$ (MeV)')
        r.set_yticks([0,.5,1,1.5])
    axs[0,0].set_ylabel(r'Observed 90% CL$_s$ upper limit on $\epsilon^2$')
    axs[1,0].set_ylabel('Ratio to released limit')
    fig.suptitle('2021 10%: background profiling comparison',fontsize=16,fontweight='semibold',y=.98)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.text(.5,.025,'Same data, kernels, masks, resolution and yield conversion. Conditional asymptotic limits; zero new toys.',ha='center',fontsize=10,color='.3')
    fig.subplots_adjust(left=.09,right=.98,bottom=.13,top=.79,hspace=.13,wspace=.25)
    save(fig,'observed_limits_2021_comparison')


def decomposition(d):
    fig,(a,b)=plt.subplots(1,2,figsize=(11.8,4.0))
    a.plot(d.mass_MeV,100*(d.eps2_log_gp/d.eps2_current-1),color=ORANGE,lw=1.5,label='Log-GP / released')
    a.plot(d.mass_MeV,100*(d.eps2_gaussian_control/d.eps2_current-1),color=BLUE,ls='--',lw=1.2,label='Stable Gaussian / released')
    b.plot(d.mass_MeV,100*(d.eps2_log_gp/d.eps2_gaussian_control-1),color=GREEN,lw=1.5)
    for ax in (a,b):ax.axhline(0,color='.5',lw=.7);ax.set(xlim=(50,250),xlabel='Mass hypothesis (MeV)',ylabel='Limit change (%)')
    a.set_title('Total change and numerical control');b.set_title('Background-model change with the same solver')
    fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.71,wspace=.27)
    save(fig,'profile_model_and_numerics')


def fits(d,new,old,masses,name):
    fig,axs=plt.subplots(2,2,figsize=(11.8,6.8),sharex='col',gridspec_kw={'height_ratios':[1,1.1]})
    for j,mass in enumerate(masses):
        p=new[new.mass_MeV==mass].sort_values('bin_center_MeV').copy()
        o=old[(old.mass_hypothesis_MeV==mass)&old.group.isin(['2021','2021_deficit'])].copy()
        lookup={round(float(t.bin_center_MeV),7):t for t in o.itertuples()}
        original=[lookup[round(float(v),7)] for v in p.bin_center_MeV]
        for k in ('profiled_background','signal','total'):
            p[k]=[getattr(t,k) for t in original]
        assert np.array_equal(p.observed,[t.observed for t in original])
        f=p[p.in_fit];outside=p[~p.in_fit]
        row=d.set_index('mass_MeV').loc[mass]
        sigma=float(row.sigma_MeV);a,r=axs[:,j]
        # Curves stop at bin edges of the actual target window. No splicing.
        lo=float(f.bin_center_MeV.iloc[0]-f.bin_width_MeV.iloc[0]/2)
        hi=float(f.bin_center_MeV.iloc[-1]+f.bin_width_MeV.iloc[-1]/2)
        for ax in (a,r):
            ax.axvspan(lo,hi,color=BLUE,alpha=.045,zorder=0)
            ax.axvline(lo,color='.65',ls=':',lw=.8);ax.axvline(hi,color='.65',ls=':',lw=.8)
        a.plot(p.bin_center_MeV,p.gp_mean/p.bin_width_MeV,color=GREY,lw=1.1)
        for frame,col in ((outside,'.65'),(f,'black')):
            a.errorbar(frame.bin_center_MeV,frame.observed/frame.bin_width_MeV,
                yerr=np.sqrt(frame.observed)/frame.bin_width_MeV,fmt='o',ms=3,color=col,elinewidth=.7)
            r.errorbar(frame.bin_center_MeV,(frame.observed-frame.gp_mean)/frame.bin_width_MeV,
                yerr=np.sqrt(frame.observed)/frame.bin_width_MeV,fmt='o',ms=3,color=col,elinewidth=.7)
        a.plot(f.bin_center_MeV,f.profiled_background/f.bin_width_MeV,color=BLUE,ls=':',lw=1.6)
        a.plot(f.bin_center_MeV,f.total/f.bin_width_MeV,color=BLUE,lw=2)
        a.plot(f.bin_center_MeV,f.total_log_gp/f.bin_width_MeV,color=ORANGE,ls='--',lw=1.6)
        r.fill_between(f.bin_center_MeV,-f.gp_sd/f.bin_width_MeV,f.gp_sd/f.bin_width_MeV,color='.75',alpha=.35,lw=0)
        r.axhline(0,color=GREY,lw=.9)
        r.plot(f.bin_center_MeV,(f.profiled_background-f.gp_mean)/f.bin_width_MeV,color=BLUE,ls=':',lw=1.6)
        r.plot(f.bin_center_MeV,(f.total-f.gp_mean)/f.bin_width_MeV,color=BLUE,lw=2)
        r.plot(f.bin_center_MeV,(f.total_log_gp-f.gp_mean)/f.bin_width_MeV,color=ORANGE,ls='--',lw=1.6)
        a.set_title(f'{mass} MeV'+('  (signed deficit fit)' if mass==71 else ''))
        a.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
        r.set(xlabel='Invariant mass (MeV)',xlim=(mass-4*sigma,mass+4*sigma))
        if j==0:
            a.set_ylabel('Events / MeV')
            r.set_ylabel('Data or model - GP mean\n(events / MeV)')
    handles=[Line2D([],[],color='black',marker='o',ls='',ms=4,label='Data (counting errors)'),
             Line2D([],[],color=BLUE,lw=2,label='Released signal + background'),
             Line2D([],[],color=ORANGE,ls='--',lw=1.8,label='Log-GP signal + background'),
             Line2D([],[],color=BLUE,ls=':',lw=1.8,label='Released profiled background'),
             Patch(color='.75',alpha=.35,label='Sideband GP marginal 1-sigma band')]
    fig.suptitle('2021 10%: fits with a consistent residual baseline',y=.99,fontsize=15,fontweight='semibold')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False,fontsize=9.5)
    fig.text(.5,.02,'Profile curves are drawn only in fitted bins; grey data are outside. The GP band is not an independent residual uncertainty.',ha='center',fontsize=9,color='.3')
    fig.subplots_adjust(left=.095,right=.98,bottom=.125,top=.76,hspace=.13,wspace=.25)
    save(fig,name)


def build_pdf(d,s):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall',parent=styles['BodyText'],fontSize=10.5,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='CaptionSmall',parent=styles['BodyText'],fontSize=9,leading=12,spaceAfter=8))
    styles.add(ParagraphStyle(name='ReportTitle',parent=styles['Title'],fontSize=19,leading=23,spaceAfter=14))
    elements=[]
    def para(text,style='BodySmall'):elements.append(Paragraph(text,styles[style]))
    def fig(name,width=510):
        from PIL import Image as PILImage
        p=FIG/(name+'.png')
        with PILImage.open(p) as im:w,h=im.size
        elements.append(Image(str(p),width=width,height=width*h/w))
    para('Background profiling in the 2021 10% sample','ReportTitle')
    para('Comparison based on v4.9.12 and v4.9.12.5 - 5 September 2026','CaptionSmall')
    para('<b>The framework is literature-grounded; its application still requires calibration.</b> '
         'The released fit uses a Poisson likelihood and correlated Gaussian background constraints. '
         'The latter approximate the count distribution implied by a sideband-trained log-GP. '
         'Kernel hyperparameters are frozen, while correlated background modes are profiled at every tested signal yield. '
         'This is not a fit with one independent unconstrained background parameter per bin.')
    para('Frate et al. [1] connect GP models to constrained Poisson intensities and discuss positive log-intensity models. '
         'They also caution that kernel regularization does not automatically acquire a frequentist auxiliary-measurement interpretation. '
         'Cowan et al. [2] justify nuisance profiling and bounded asymptotic likelihood-ratio mappings under their assumptions; '
         'they do not validate a selected background model. These papers support the ingredients, not unconditional coverage of this frozen HPS configuration.')
    para('<b>The controlled alternative profiles the latent log-background directly:</b> '
         'lambda<sub>i</sub> = exp(g<sub>i</sub> + (R theta)<sub>i</sub>) + A w<sub>i</sub>, '
         'with penalty theta<sup>T</sup>theta / 2. The release instead uses '
         'lambda = b + L theta + A w. The alternative retains the positive, skewed background distribution '
         'whose mean and covariance the release approximates as Gaussian. The fixed reference sets theta = 0 in the released model, holding its GP mean known.')
    para('All three use the same native histogram, 36-300 MeV training support, reviewed kernel coordinates, '
         'moving +/-2.25 sigma mask and fit bins, nominal signal template, and frozen yield-to-coupling conversion. '
         'The alternative uses exp(g), its zero-nuisance median, as its Asimov background. '
         'The largest mean/median shift is only %.6f%%. Every displayed curve includes the same dimuon branching correction.'%(100*s['maximum_log_mean_median_relative']))
    rows=[['Result across all 201 masses','Value'],
          ['Largest log-GP / released limit change',f'{100*s["max_abs_log_current_relative"]:.3f}%'],
          ['Largest change due to model with same solver',f'{100*np.max(np.abs(d.eps2_log_gp/d.eps2_gaussian_control-1)):.3f}%'],
          ['Median fixed / released limit',f'{s["median_fixed_current"]:.3f}'],
          ['Range of fixed / released limit',f'{s["min_fixed_current"]:.3f} to {s["max_fixed_current"]:.3f}']]
    table=Table(rows,colWidths=[360,140],hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef4')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey)]))
    elements.append(table);elements.append(Spacer(1,12))
    para('<b>Interpretation.</b> The count-space Gaussian approximation is not driving the peak-dip pattern. '
         'Within the frozen model, the GP uncertainty is small in fractional terms (maximum marginal standard deviation %.3f%%), '
         'so Gaussian and log-GP profiles are locally very similar. Fixing the background removes meaningful uncertainty and '
         'also changes the fitted signal: it need not lower every observed limit.'%(100*s['maximum_gp_fractional_sd']))
    para('These are conditional asymptotic observed limits, not a replacement calibrated exclusion. '
         'No new toys or unblinding were performed. All 201 native prediction hashes match the release.','CaptionSmall')
    elements.append(PageBreak())
    para('Observed upper limits and numerical separation','ReportTitle')
    fig('observed_limits_2021_comparison')
    para('The orange and blue curves nearly coincide. The fixed-background reference can change the result strongly in either direction. '
         'For example, at 71 MeV the fixed limit is 0.241 times the release; at 78 MeV it is 1.215 times the release. '
         'A tighter fixed-background limit is not evidence of a better background model.','CaptionSmall')
    fig('profile_model_and_numerics')
    para('A stable, centered Poisson deviance and scaled signal coordinate provide a numerical control using exactly the released Gaussian model. '
         'That control differs from the release by up to 1.139% at 86 MeV. Comparing the two background models using the same solver limits '
         'their difference to 0.232%. Independent BFGS fits from two starts at seven masses agree in NLL within 1e-7, and an analytic '
         'one-bin fixed-background check agrees in upper limit within 1e-6 events. All 603 new limits pass positivity, nesting, root and monotonicity checks.','CaptionSmall')
    elements.append(PageBreak())
    para('The low-mass peak and neighboring deficit','ReportTitle')
    fig('fits_65_71')
    para('A single GP-mean baseline is subtracted across each complete panel. The released profiled background and both fitted totals '
         'are drawn only where the likelihood was evaluated. This removes artificial joins caused by replacing the profiled background '
         'with the unprofiled mean immediately outside the window. The grey band is the conditional sideband GP marginal uncertainty; '
         'the data errors show counting fluctuations only. Neither is a calibrated residual significance.','CaptionSmall')
    para('<b>Two effects should be kept separate.</b> The joins in the original residual displays are plotting artifacts. '
         'The neighboring excess-deficit response is a property of the analysis: the v4.9.12.5 positive-injection studies show that '
         'a feature protected by one mass hypothesis can enter training at nearby hypotheses and shift the predicted background. '
         'This log-GP alternative shares that moving mask and retains the same pattern.')
    subset=d.set_index('mass_MeV')
    tab=[['Mass (MeV)','Released r','Log-GP r','Fixed r']]+[[str(m),f'{subset.loc[m,"r_current"]:.3f}',f'{subset.loc[m,"r_log_gp"]:.3f}',f'{subset.loc[m,"r_fixed"]:.3f}'] for m in (65,71,78,182)]
    t=Table(tab,colWidths=[125]*4,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef4')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));elements.append(t)
    para('r is the signed square root of the local profile-likelihood ratio. Negative amplitudes are deficit diagnostics, '
         'not negative physical couplings. These data-selected local values are not independent or global significances.','CaptionSmall')
    elements.append(PageBreak())
    para('Additional fits and the next substantive test','ReportTitle')
    fig('fits_78_182')
    para('<b>A useful next comparison changes the background description.</b> HPS has published simultaneous Poisson '
         'signal-plus-positive-exponential-polynomial fits [3]. A predeclared version, or discrete profiling over a small '
         'set of candidate background functions [4], would test shape assumptions more directly than changing Gaussian '
         'constraints to log-GP constraints. The degree, support and candidate set should be qualified with background and '
         'injection studies, not selected for the most attractive observed limit.')
    para('Before treating a method as a replacement, test held-out spurious signal, injections across neighboring masses, '
         'and full-pipeline signal-plus-background coverage with GP retraining and the stated model-selection procedure. '
         'The existing 4.9.12.5 mechanism tests are conditional on selected generating backgrounds and amplitudes; '
         'the agreement here does not turn them into calibrated coverage or physical evidence.')
    refs=[
      '[1] Frate et al., Modeling Smooth Backgrounds and Generic Localized Signals with Gaussian Processes (2017), Eq. 4 and Sec. III. <link href="https://arxiv.org/abs/1709.05681" color="blue">arXiv:1709.05681</link>.',
      '[2] Cowan et al., Asymptotic formulae for likelihood-based tests of new physics, EPJC 71, 1554 (2011), Sec. 3.7. <link href="https://arxiv.org/abs/1007.1727" color="blue">arXiv:1007.1727</link>.',
      '[3] HPS Collaboration, Searching for Prompt and Long-Lived Dark Photons..., PRD 108, 012015 (2023), Sec. IV.1. <link href="https://arxiv.org/abs/2212.10629" color="blue">arXiv:2212.10629</link>.',
      '[4] Dauncey et al., Handling uncertainties in background shapes: the discrete profiling method, JINST 10, P04015 (2015). <link href="https://arxiv.org/abs/1408.6865" color="blue">arXiv:1408.6865</link>.']
    for ref in refs:para(ref,'CaptionSmall')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
        canvas.drawString(44,23,'HPS-GPR | 2021 10% | conditional background-profile comparison')
        canvas.drawRightString(568,23,str(doc.page))
    doc=SimpleDocTemplate(str(PDF/'background_profile_comparison_2021.pdf'),pagesize=letter,
        leftMargin=44,rightMargin=44,topMargin=38,bottomMargin=38,
        title='2021 background profiling comparison',author='Emrys Peets')
    doc.build(elements,onFirstPage=footer,onLaterPages=footer)


def main():
    FIG.mkdir(exist_ok=True);PDF.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(OUT/'observed_limits.csv');s=json.loads((OUT/'summary.json').read_text())
    new=pd.read_csv(OUT/'fit_plot_data.csv')
    source=REPO/'study_results/v4p9p12_expanded_snapshot_20260905/derived/selected_fit_plot_data.csv'
    old=pd.read_csv(source);old=old[old.dataset.astype(str)=='2021']
    limits(d);decomposition(d);fits(d,new,old,(65,71),'fits_65_71');fits(d,new,old,(78,182),'fits_78_182')
    build_pdf(d,s)
    display=d.copy();display['dimuon_factor']=branching(d.mass_MeV)
    for name in ('current','log_gp','fixed','gaussian_control'):
        display['eps2_'+name+'_dimuon']=d['eps2_'+name]*display.dimuon_factor
    display.to_csv(OUT/'observed_limits_with_display_correction.csv',index=False)
    (OUT/'report_sources.json').write_text(json.dumps({'parent_plot_data':str(source),
        'parent_plot_data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'report_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(PDF/'background_profile_comparison_2021.pdf')


if __name__=='__main__':main()
