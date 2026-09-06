#!/usr/bin/env python3
"""Draw the deficit companion and extend the preserved LaTeX analysis note."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v4p9p16_combined_global_20260906'
OUT=ROOT/'output/pdf/v4p9p16_deficit_extension_20260906'
BLUE,RED,GREEN,GOLD,PURPLE='#18699C','#AE3E30','#1C7044','#C98925','#7B4EA3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def probability(v):
    if v>=.001:return f'{v:.4f}'.rstrip('0').rstrip('.')
    a,b=f'{v:.2e}'.split('e')
    return '$'+a+r'\times10^{'+str(int(b))+'}$'
def boundaries(ax):
    for m in (38.5,49.5,90.5,180.5):ax.axvline(m,color='.65',lw=.7,ls=':',zorder=0)
    ax.set_xlim(18.5,250.5);ax.grid(axis='y',alpha=.17)
def main():
    summary=json.loads((HERE/'analysis/summary.json').read_text());assert summary['passed']
    data=pd.read_csv(HERE/'analysis/deficit_curves.csv')
    d=data[data.method=='profiled'].sort_values('mass_MeV');q=summary['methods']['profiled']
    plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':210})
    fig=plt.figure(figsize=(10.5,7.5))
    grid=fig.add_gridspec(4,1,height_ratios=(.20,1.55,1.35,1.35),
        left=.105,right=.985,top=.85,bottom=.07,hspace=.34)
    strip=fig.add_subplot(grid[0]);axes=[fig.add_subplot(grid[i],sharex=strip) for i in (1,2,3)]
    segments=[(19,38,'2015'),(39,49,'2015\n+2016'),(50,90,'All three'),
        (91,180,'2016 + 2021'),(181,250,'2021')]
    for (lo,hi,label),color in zip(segments,['#DDEAF3','#D8E7DD','#E8E0F0','#F2E5D5','#E5E5E5']):
        strip.add_patch(Rectangle((lo-.5,0),hi-lo+1,1,facecolor=color,edgecolor='white'))
        strip.text((lo+hi)/2,.5,label,ha='center',va='center',fontsize=8,fontweight='semibold')
    strip.set(ylim=(0,1));strip.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
    strip.set_ylabel('Active\ndata',rotation=0,ha='right',va='center',fontsize=8)
    for spine in strip.spines.values():spine.set_visible(False)
    raw,local,glob=axes
    raw.plot(d.mass_MeV,d.observed_r,color=BLUE,lw=1.55,label='Observed signed root')
    raw.plot(d.mass_MeV,d.asimov_r,color=PURPLE,ls='--',lw=1.15,label='Stress-background Asimov offset')
    raw.fill_between(d.mass_MeV,d.observed_r,0,where=d.observed_r<0,color=BLUE,alpha=.12,interpolate=True)
    raw.axhline(0,color='.4',lw=.7)
    raw.set_ylabel(r'Signed root $r$')
    raw.set_title('Negative fitted signals and the background offset',loc='left',fontsize=11,fontweight='semibold')
    raw.legend(loc='upper right',fontsize=8.5,frameon=False)
    for col,color,style,label in [('p_raw_gaussian','.5','--',r'Raw-root reference: $N(0,1)$'),
        ('p_local_deficit',BLUE,'-','Stress-centered Gaussian deficit')]:
        y=d[col].to_numpy()
        local.plot(d.mass_MeV,np.maximum(y,1e-8),color=color,ls=style,lw=1.4,label=label)
        low=y<1e-8
        local.scatter(d.mass_MeV[low],np.full(sum(low),1e-8),color=color,marker='v',s=15,clip_on=False,zorder=5)
    local.set(yscale='log',ylim=(1e-8,1.6),ylabel='Local deficit\nprobability')
    local.set_title('Local deficit tails: only negative raw fits are eligible',loc='left',fontsize=11,fontweight='semibold')
    local.legend(loc='lower right',fontsize=8.4,frameon=False)
    for prefix,color,style,label in [('gp',RED,'-','GP global: centered deficit'),
        ('raw_gp',GOLD,':','GP global: separate raw-depth ordering')]:
        glob.plot(d.mass_MeV,np.where(d[prefix+'_k']>0,d[prefix+'_p'],np.nan),
                  color=color,ls=style,lw=1.5,label=label)
        zeros=d[prefix+'_k']==0
        glob.scatter(d.mass_MeV[zeros],d.loc[zeros,prefix+'_upper95'],color=color,marker='v',s=17,zorder=5)
    selected=d[d.mass_MeV.isin(summary['representative_masses_MeV'])]
    pos=selected[selected.direct_k>0];zero=selected[selected.direct_k==0]
    glob.errorbar(pos.mass_MeV,pos.direct_p,
        yerr=np.array([pos.direct_p-pos.direct_low95,pos.direct_high95-pos.direct_p]),
        fmt='o',ms=3,color=GREEN,elinewidth=1,capsize=2,zorder=6,label='Direct centered deficit: 95% intervals')
    glob.scatter(zero.mass_MeV,zero.direct_upper95,color=GREEN,marker='v',s=25,zorder=6)
    glob.plot([],[],marker='v',color='.3',ls='none',ms=4,label='Zero tails: one-sided 95% upper bounds')
    glob.set(yscale='log',ylim=(1e-6,1.6),ylabel='Global deficit\nprobability',xlabel=r'Mass hypothesis $m_{A^\prime}$ [MeV]')
    glob.set_title('Deficit maxima over the complete 232-point scan',loc='left',fontsize=11,fontweight='semibold')
    glob.legend(loc='lower right',fontsize=8.1,frameon=False)
    for ax in axes:boundaries(ax)
    for ax in (raw,local):ax.tick_params(labelbottom=False)
    fig.suptitle('Combined deficit scan: an illustrative check',fontsize=15,fontweight='semibold',y=.98)
    fig.text(.5,.922,'2015 full + 2016 full + 2021 10%  |  19–250 MeV  |  Profiled likelihood',ha='center',fontsize=10.5)
    fig.text(.5,.890,'Conditional stress background; direction-specific tests after the excess scan.',ha='center',fontsize=9.8,color='.3')
    figures=[]
    for ext in ('pdf','png'):
        p=HERE/'figures'/f'combined_deficit_scan.{ext}';fig.savefig(p,bbox_inches='tight');figures.append(p)
    plt.close(fig)
    note=HERE/'note'
    for p in (PARENT/'note').glob('*.tex'):
        if p.name!='analysis_note.tex':shutil.copyfile(p,note/p.name)
    maintex=(PARENT/'note/analysis_note.tex').read_text()
    maintex=maintex.replace('../figures/','../../v4p9p16_combined_global_20260906/figures/')
    maintex=maintex.replace('Analysis note v4.9.16}',r'Analysis note v4.9.16: deficit extension}')
    maintex=maintex.replace('combined global search and observed limits}', 'combined global search, observed limits and deficit illustration}')
    anchor=r'\clearpage'+'\n'+r'\section{One shared-coupling likelihood}'
    assert maintex.count(anchor)==1
    maintex=maintex.replace(anchor,r'\clearpage'+'\n'+r'\input{deficit_section.tex}'+'\n\n'+anchor)
    independent=json.loads((HERE/'review/independent_final_audit.json').read_text())
    assert independent['passed']
    maintex=maintex.replace('The final manifest binds the report and figures.',
        'The original manifest remains unchanged. The deficit extension and this revised report have a separate manifest under\n'
        +r'\begin{center}\small\path{study_results/v4p9p16_deficit_extension_20260906/}\end{center}'
        +f"\nThe independent deficit audit passed {independent['checked_conditions']} conditions.")
    (note/'analysis_note.tex').write_text(maintex)
    rawq=q['raw_ordering'];g=q['gp_global'];direct=q['direct_global']
    paragraph=(f"The deepest raw profiled deficit is at {rawq['peak_mass_MeV']} MeV, with "
        +f"$r=-{rawq['depth']:.3f}$ and an uncalibrated local reference of "+probability(rawq['raw_gaussian_reference_p'])
        +f". The largest stress-centered deficit instead occurs at {q['peak_mass_MeV']} MeV: "
        +f"$r={q['observed_r']:.3f}$, $a={q['asimov_r']:+.3f}$ and $s={q['response_sd']:.3f}$. "
        +"Its raw reference is "+probability(q['raw_gaussian_reference_p'])+", whereas the conditional local value is "
        +probability(q['local_deficit_p'])+". This contrast again comes mainly from the stress offset.")
    counts=(f"The latter threshold has {g['k']}/200,000 GP and {direct['k']}/1,000 direct exceedances. "
        +"The one-sided 95\\% global upper bounds are "+probability(g['upper95'])+" and "+probability(direct['upper95'])+", respectively. "
        +"Every simulated raw-depth maximum exceeds the observed deficit. These are the same realizations used for the excess scan; no new fits or independent toys were needed.")
    section=r"""\section{Illustrative scan of deficits}
A negative auxiliary signal amplitude describes missing events with the signal-template shape; it is not a physical negative rate or coupling. This figure mirrors the excess scan without changing its fits or limits.

\fig{0.97}{../figures/combined_deficit_scan.pdf}{Profiled deficit scan. Top: observed signed roots and the stress-background offset; shading marks negative fitted contributions. Middle: raw-root Gaussian reference and conditional Gaussian-response tails. Both assign one to nonnegative raw roots. Local triangles mark the $10^{-8}$ display floor. Bottom: two separate global orderings; triangles mark one-sided 95\% zero-count bounds. Direct error bars are central 95\% intervals.}{fig:deficit}

For $z=(r-a)/s$, the conditional local rule is $p^-=\Phi(z)$ when $r<0$, and one otherwise. The principal scan score is $T^-=\max_{m:r_m<0}(-z_m)$, with $-\infty$ for an empty set. The separate raw-depth statistic is $D^-=\max_m\max(0,-r_m)$.

"""+paragraph+'\n\n'+counts+r"""

These illustrative tails follow the excess scan and do not adjust for choosing directions, methods or orderings. The unresolved tails establish neither a particle signal nor physical background validity.
"""
    (note/'deficit_section.tex').write_text(section)
    OUT.mkdir(parents=True,exist_ok=True)
    run=subprocess.run(['tectonic','--keep-logs','--outdir',str(OUT),str(note/'analysis_note.tex')],
        cwd=note,capture_output=True,text=True)
    (note/'build.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(run.stderr)
    pdf=OUT/'HPS_GPR_Analysis_Note_v4p9p16_with_Deficit_Scan.pdf'
    generated=OUT/'analysis_note.pdf';pdf.write_bytes(generated.read_bytes());generated.unlink()
    inputs=[Path(__file__),HERE/'PROTOCOL.md',HERE/'analysis/summary.json',
        HERE/'review/independent_final_audit.json',
        HERE/'analysis/deficit_curves.csv',PARENT/'MANIFEST.csv',
        *note.glob('*.tex'),*PARENT.joinpath('figures').glob('*.pdf')]
    dump(HERE/'provenance/report_build.json',dict(pdf=str(pdf),pdf_sha256=sha(pdf),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        figure_sha256={str(p.relative_to(ROOT)):sha(p) for p in figures}))
    print(run.stdout+run.stderr);print(pdf)
if __name__=='__main__':main()
