#!/usr/bin/env python3
"""Add the presentation extraction section to the preserved v4.9.16 note."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v4p9p16_deficit_extension_20260906'
OUT=ROOT/'output/pdf/v4p9p16_presentation_extractions_20260906'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fig(name,caption,label,width='0.99'):
    return '\\fig{'+width+'}{../figures/'+name+'.pdf}{'+caption+'}{fig:extraction-'+label+'}\n'
def landscape(title,name,caption,label,after=''):
    width='0.90' if name in ('extraction_combined_92','extraction_individual_deficits') else '0.99'
    return '\\clearpage\n\\begin{landscape}\n\\subsection*{'+title+'}\n'+fig(name,caption,label,width)+after+'\n\\end{landscape}\n'
def table(headers,rows,fmt,caption):
    return (r'\begin{table}[H]\centering\small'+'\n'+r'\begin{tabular}{'+fmt+'}'+r'\toprule'+'\n'
        +' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)
        +'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\caption{'+caption+'}'+r'\end{table}'+'\n')

def main():
    note=HERE/'note';note.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    closure=json.loads((HERE/'derived/fit_closure.json').read_text());assert closure['passed']
    fits={r['fit_id']:r for r in closure['checks']}
    summary=pd.read_csv(HERE/'derived/fit_summary.csv',dtype={'dataset':str})
    consistency=pd.read_csv(HERE/'derived/dataset_consistency.csv',dtype={'dataset':str})
    precision=pd.read_csv(HERE/'derived/exposure_precision.csv')
    yields=pd.read_csv(HERE/'derived/exposure_signal_yields.csv')
    retained=pd.read_csv(HERE/'derived/common_display_retention.csv',dtype={'dataset':str})
    selection=json.loads((HERE/'derived/selection.json').read_text())
    sections=[]
    sections.append(r'''\section{Selected signal extractions and staged exposure checks}
\label{sec:extractions}
\textbf{What the displays establish.} The largest observed combined excess is at 66 MeV. All three years prefer a positive fitted signal there, although each estimate has substantial uncertainty. The next peak in the complete search is 21 MeV, where only 2015 contributes. The next leading peak covered by several datasets is 92 MeV, where 2016 and 2021 prefer noticeably different rates. These are useful locations for follow-up; the selected plots alone do not establish a particle or a failure of the background model.

The strongest raw combined deficit is at 72 MeV. Showing it alongside the excesses matters: a background interpolation error can produce both positive and negative residuals, while a positive narrow signal does not by itself explain a nearby missing-event feature. Statistical fluctuations can also produce such patterns. An independent data increment and predictive background checks can distinguish these possibilities more directly than enlarging a selected peak display.

\subsection*{Which locations are shown}
Selection uses the dense \emph{observed profiled} signed root, whose ordering matches its one-sided asymptotic local reference on the positive branch. Select positive local maxima, ordered by decreasing root, with a separation greater than $2.25$ times the larger resolution at the two coordinates. For the combined rule use the largest active resolution. Endpoints remain eligible and are identified. The stress-centered ordering is discussed separately; it does not choose the leading extraction panels.
''')
    rows=[]
    for rec in selection['rankings']:
        key=rec['group'];label={'2015':'2015 full','2016':'2016 full','2021':r'2021 10\%','combined':'Full combined union'}[key]
        entries=[]
        for m in rec['positive_peaks_MeV']:
            r=fits[f'{key}_m{m:03d}']['root'];entries.append(f'{m} ({r:+.2f})')
        m=rec['deepest_deficit_MeV'];r=fits[f'{key}_m{m:03d}']['root']
        rows.append([label,*entries,f'{m} ({r:+.2f})'])
    sections.append(table(['Data','First excess','Second excess','Deepest deficit'],rows,'lrrr',
        'Selected mass in MeV, followed by the signed root in parentheses. These are selected local fit summaries, not global significances. The 19 MeV deficit is the 2015 search endpoint. The additional combined 92 MeV display has $r=+2.42$ and uses 2016 plus 2021.'))
    sections.append(r'''\subsection*{How to read the figures}
The top row shows event counts, the GP mean before profiling, the background fitted together with the signal, and their sum. The lower row subtracts that fitted background and overlays the fitted signal. Black bars show counting uncertainty only. The gray envelope is a zero-centered width guide from the original correlated GP constraint, projected into the display bins. It does not depict that constraint's center, an error band on the fitted background, or the full residual covariance. These residuals are not independent significance measurements.

Display bins are whole original bins grouped to approximately half the mass resolution, with a fixed histogram-edge origin. The likelihood retains every native fit bin. Multi-year sums use a common 1.25 MeV grid restricted to the overlap of the fitted windows. Each year keeps its own resolution, acceptance and count-to-coupling conversion in the likelihood. No fitted curve is extended outside its fit window.
''')
    sections.append(landscape('The leading combined excess: 66 MeV','extraction_combined_66',
        'The 66 MeV common-amplitude fit, shown separately for each year and as a display sum. The shared signed amplitude is $\\widehat\\epsilon^2=4.55\\times10^{-6}$ and $r=2.76$. Every panel uses the same amplitude, with the appropriate year-specific yield and resolution. Summed panels omit boundary pieces; the likelihood root and quoted fitted yields use the full native windows.','combined66'))
    sections.append(landscape('A second location with multiple datasets: 92 MeV','extraction_combined_92',
        'The 92 MeV fit uses 2016 and 2021. The common amplitude is $\\widehat\\epsilon^2=2.97\\times10^{-6}$ and $r=2.42$. Both separate fits are positive, but their preferred rates differ (Fig.~\\ref{fig:extraction-consistency}). A persistence illustration at this common rate is a testable assumption.','combined92'))
    sections.append(landscape('The deepest combined observed deficit: 72 MeV','extraction_combined_72',
        'The deepest raw combined deficit has $r=-3.49$ and a signed auxiliary amplitude $\\widehat\\epsilon^2=-4.88\\times10^{-6}$. All three separate amplitudes are negative, with 2021 providing the strongest individual contribution. Negative templates describe missing events relative to the fitted background, not a physical negative coupling.','combined72'))
    for year,name,context in [
        ('2015','2015 full',r'''The 51 MeV point is the largest observed profiled excess in 2015. The 21 MeV point is also the second leading peak of the full combined union, because 2015 is the only active dataset at this mass. It is only 2 MeV above the search endpoint; the 19 MeV deficit in Fig.~\ref{fig:extraction-individualdeficits} makes boundary and background-shape controls particularly relevant. More 2021 data cannot test this 21 MeV location.'''),
        ('2016','2016 full',r'''The two leading individual 2016 peaks are at 90 and 117 MeV. Their positions differ from the leading combined mass because a common coupling also has to describe the other active dataset contributions. These panels retain the inherited 2016 source-fit waiver, development overlap and numerical qualifications; cleaner presentation does not resolve them.'''),
        ('2021',r'2021 10\%',r'''The released 2021 sample has leading individual peaks at 78 and 65 MeV. These are near, but not interchangeable with, the combined coordinates. Changing the mass or window after seeing a new sample would create another selection. Freeze these individual locations as a declared secondary family if they are followed at the 30\% checkpoint.''')]:
        sections.append('\\clearpage\n\\subsection*{Individual signal extractions: '+name+'}\n'+context+'\n\n'+
            fig('extraction_'+year+'_peaks','The two leading separated observed profiled excesses in '+name+'. Each panel is an independent single-dataset fit. Binning is chosen from the resolution and original histogram origin, not from the observed residual shape.','individual'+year)+
            '\nThe lower panels are background-subtracted displays conditioned on the fitted template. A visible peak here should be assessed together with the background constraint, the fitted amplitude uncertainty, the scan selection and independent-data checks.\n')
    sections.append(landscape('Individual deficits as a background diagnostic','extraction_individual_deficits',
        'Deepest observed negative roots in 2015, 2016 and 2021. The 2015 point lies at the scan endpoint. The 2016 and 2021 minima are at 102 and 71 MeV, respectively. The negative branch provides a diagnostic with the same narrow template; no physical negative signal rate or calibrated deficit discovery is inferred.','individualdeficits'))
    sections.append(landscape('Do separate years support the same rate?','dataset_amplitude_consistency',
        'Independent signed amplitude estimates and local curvature standard errors, compared with the common fit. Define $\\Delta D=2[\\mathrm{NLL}_{\\rm common}-\\sum_d\\mathrm{NLL}_{d,\\rm free}]$. The number of additional amplitude parameters is one fewer than the number of datasets. These masses were selected from the data, so the displayed likelihood losses are conditional descriptive comparisons; no post-selection compatibility probabilities are quoted.','consistency'))
    sections.append(r'''\clearpage
\subsection*{Why the most extreme stress-centered tail need not look like a peak}
The stress-background centering changes the reference distribution of the statistic; it does not change the observed fitted signal. At 76 MeV the combined raw root is only $+0.166$, but the generating stress construction has Asimov root $a=-8.700$ and response width $s=0.979$. Subtracting this large negative offset produces a standardized value of $9.05$. At 83 MeV the observed raw root is $-0.676$, with $a=+7.707$ and $s=0.983$, giving a standardized deficit depth of $8.53$.

The very small conditional tails therefore ask about departure from those particular stress constructions. They do not mean that the data contain a correspondingly large positive or negative fitted signal. At 76 MeV, 2016 prefers a negative amplitude while 2021 prefers a positive one. The common-versus-independent likelihood loss is $\Delta D=9.40$ for two additional amplitudes. This provides another reason to avoid presenting the stress-centered tail as a coherent resonance.
''')
    sections.append(fig('extraction_stress_extrema','Common-window sums at the two principal stress-centered extrema. The small fitted signal templates are consistent with their small raw roots. Large stress offsets cause the extreme centered-tail interpretation. These sums have no independent fit.','stress'))
    sections.append(r'''The asymptotic profiled method is not invalid merely because calibration changes a probability or an observed limit. Its use in a final analysis requires qualification of the background family, the prediction uncertainty, signal response and the relevant sampling distributions. The present displays do not measure expected sensitivity. A weaker calibrated observed limit can reflect a correction to background behavior rather than information being removed from the data.
''')
    sections.append(r'''\clearpage
\subsection*{A 30\% checkpoint before the full 2021 sample}
Use the released 10\% to define the hypotheses. Treat an additional, disjoint 20\% as the next independent test of those fixed locations and rates. Only then inspect the cumulative 30\% combination. This is more informative than a cumulative display alone, because the latter still contains the fluctuation that selected the peak. The same original-event membership and processing must be documented for every increment.

Let $N_{10}$ be the observed first sample, $B_{10}$ its background fitted in the selected common-amplitude model, and $S_{10}$ the signal template at that fitted rate. Figure~\ref{fig:extraction-exposure} compares two illustrative conditional means for a total exposure $f$ times the original 10\%:
\[
\begin{aligned}
\text{background only in added data:}&\quad N_{10}+(f-1)B_{10},\\
\text{selected rate persists:}&\quad N_{10}+(f-1)(B_{10}+S_{10}).
\end{aligned}
\]
The independent additional 20\% has means $2B_{10}$ or $2(B_{10}+S_{10})$. For the conditional cumulative 30\% and 100\% views the background-only counting variances from the future sample are $2B_{10}$ and $9B_{10}$, respectively. The observed first 10\% is held fixed. These figures are mean scenarios, not new observations, toy experiments or discovery projections. They omit uncertainty in the assumed background and selected signal rate. The 92 MeV common-rate assumption is less consistent with the individual estimates than the 66 MeV assumption.
''')
    yr=[]
    for m in [66,92]:
        yy=yields[yields.mass_MeV==m].set_index('exposure_percent')
        yr.append([m,*[f'{yy.loc[p,"template_yield_window"]/1000:.1f}' for p in [10,20,30,100]]])
    sections.append(table(['Mass [MeV]',r'10\%',r'new 20\%',r'30\% total',r'100\% total'],yr,'rrrrr',
        'Illustrative mean signal yields in thousands of events in the complete 2021 fitted window if the selected common-fit rate persists. These are template yields, not the raw total counts or measured future yields. Display sums may omit boundary pieces.'))
    sections.append(r'''A statistics-dominated precision reference can be computed without toys. With the original per-year signal yield vector $u_d$, GP constraint covariance $C_d$ and GP mean $b_d$, define
\[
I_d=u_d^{\mathsf T}[\operatorname{diag}(b_d)+C_d]^{-1}u_d,
\qquad I(f)=I_{2015}+I_{2016}+fI_{2021}.
\]
An absent dataset has $I_d=0$. This local background-Asimov reference follows the usual information-scaling argument~\cite{cowan} and assumes that the \emph{entire} 2021 count covariance scales as $f$, with unchanged acceptance and resolution. Fractional systematic effects or background bias need not improve this way. The factor $\sqrt{I(f)/I(1)}$ below is a precision illustration, not a multiplier for the observed combined root or a forecast global significance.
''')
    pr=[]
    for m in [66,92]:
        pp=precision[precision.mass_MeV==m].set_index('exposure_percent')
        pr.append([m,f'{100*pp.loc[10,"original_2021_information_fraction"]:.1f}'+r'\%',
            f'{pp.loc[30,"combined_precision_gain"]:.2f}',f'{pp.loc[100,"combined_precision_gain"]:.2f}'])
    sections.append(table(['Mass [MeV]','Current 2021 information share',r'30\% gain',r'100\% gain'],pr,'rrrr',
        'Combined precision gain with 2015 and 2016 held fixed, under the stated covariance-scaling assumption. For 2021 alone the corresponding factors would be $\\sqrt{3}$ and $\\sqrt{10}$. There is no 2021 gain at 21 MeV.'))
    sections.append(landscape('Conditional exposure displays: 10\%, new 20\%, 30\%, 100\%','exposure_2021_10_30_100',
        'Background-subtracted 2021 views at fixed 66 and 92 MeV hypotheses. Only the first column contains observed points. Future curves keep the first sample fixed where included and add the mean of either hypothesis. Gray envelopes show only the background counting standard deviation of the added sample. They exclude fitted-background uncertainty and do not imply independent residual bins after a future refit. The vertical scales differ.','exposure'))
    sections.append(r'''\clearpage
\subsection*{What would make the next comparison decisive?}
First, define a disjoint additional 20\% with the same selection, calibrations, mass resolution and normalization convention. Freeze the 66 and 92 MeV primary locations, and state in advance whether the individual 2021 locations at 65 and 78 MeV and the 71/72 MeV deficit checks are a secondary family. Do not reselect the mass from the new spectrum when reporting the fixed-location validation.

Fit the additional 20\% on its own before combining it with the old 10\% or the 2015/2016 samples. Compare its fitted rate, width behavior and background residuals with the stored predictions. A stable signal should reproduce a compatible rate, not merely increase the height of a cumulative display. Rate agreement alone would still leave detector effects or common background structure to investigate.

Use predeclared sideband and predictive checks to test the background interpolation and relevant detector/selection effects, including both positive and negative residuals. The 2016 background-source transition and existing numerical/source qualifications remain part of that work. Model choices must not be selected by whichever one makes the observed peak more impressive.

The 30\% and 100\% cumulative looks share events. Before treating either as a formal discovery or exclusion result, declare the family of mass, direction and exposure tests and the treatment of sequential looks. Rebuild the appropriate signal and background sampling distributions for the new exposure and background family. The existing GP global-tail bank describes the current released sample and cannot calibrate a different exposure.

For an efficient follow-on calculation, begin with ten complete pilot spectra per declared generating hypothesis, retaining each same full spectrum throughout its mass scan. Use new deterministic seed namespaces and nonoverlapping toy IDs. Time the pilot, test exact-fit and approximation closure, then scale in restartable batches only if the validated cost is acceptable. Background-only scans address global discovery tails; calibrated limits additionally require signal-plus-background hypotheses. No new toys were necessary for the present extraction and mean-scenario displays.

\clearpage
\subsection*{Display and reconstruction checks}
Fifteen selected likelihoods were reconstructed with the archived dense solver and exact prediction hashes. There are 26 dataset components. Every background-plus-signal vector closes to the original fitted expectation; all expected Poisson counts stay positive. Every grouping matrix contains only zero or one, with no fractional reassignment or reused bin within a panel. The quoted roots and upper endpoints reproduce the frozen dense scan; display grouping never changes a fit.
''')
    ret=[]
    for m in [66,92,72]:
        rr=retained[retained.mass_MeV==m]
        for r in rr.itertuples():ret.append([m,r.dataset,f'{100*r.observed_fraction:.1f}'+r'\%',f'{100*r.fitted_signal_fraction:.1f}'+r'\%'])
    sections.append(table(['Mass [MeV]','Dataset','Counts retained in sum','Signal template retained'],ret,'rrrr',
        'Common-window summed-display retention relative to the complete native fitted window of each year. The full likelihood, amplitude estimates and total-window yield tables use all fit bins. Individual panels have their own resolution-based grouping.'))
    sections.append(r'''The reusable PDF and PNG figures, native and rebinned arrays, exact grouping maps, exposure tables, build scripts and independent HEP audit are retained in
\begin{center}\small\path{study_results/v4p9p16_presentation_extractions_20260906/}\end{center}
The prior studies were committed, pushed and merged in \href{https://github.com/EmrysPeets/hps-gpr/pull/67}{PR 67}; all 4,778 published file blobs were verified on the merged branch. This presentation extension has a separate artifact manifest. The original study inputs and outputs remain frozen.
''')
    newtext='\n'.join(sections)
    # Ensure the component count follows the machine-readable data.
    newtext=newtext.replace('There are 26 dataset components.',f'There are {closure["n_dataset_fits"]} dataset components.')
    (note/'extraction_section.tex').write_text(newtext)
    for p in (PARENT/'note').glob('*.tex'):
        if p.name=='analysis_note.tex':continue
        text=p.read_text()
        if p.name=='deficit_section.tex':text=text.replace('../figures/','../../v4p9p16_deficit_extension_20260906/figures/')
        (note/p.name).write_text(text)
    maintext=(PARENT/'note/analysis_note.tex').read_text()
    maintext=maintext.replace(r'\usepackage{microtype}',r'\usepackage{microtype,pdflscape}')
    maintext=maintext.replace('Analysis note v4.9.16: deficit extension','Analysis note v4.9.16: signal extractions and staged exposure checks')
    maintext=maintext.replace('combined global search, observed limits and deficit illustration','combined global search, signal extractions and staged exposure checks')
    anchor=r'\clearpage'+'\n'+r'\section{One shared-coupling likelihood}'
    assert maintext.count(anchor)==1
    maintext=maintext.replace(anchor,r'\clearpage'+'\n'+r'\input{extraction_section.tex}'+'\n'+anchor)
    (note/'analysis_note.tex').write_text(maintext)
    run=subprocess.run(['tectonic','--keep-logs','--outdir',str(OUT),str(note/'analysis_note.tex')],cwd=note,capture_output=True,text=True)
    (note/'build.log').write_text(run.stdout+run.stderr)
    print(run.stdout+run.stderr,flush=True)
    if run.returncode:raise RuntimeError('LaTeX build failed')
    pdf=OUT/'HPS_GPR_Analysis_Note_v4p9p16_Signal_Extractions.pdf'
    (OUT/'analysis_note.pdf').replace(pdf)
    input_paths=[Path(__file__),HERE/'PROTOCOL.md',PARENT/'MANIFEST.csv',*(note.glob('*.tex')),
        *(HERE/'figures').glob('*.pdf'),*(HERE/'derived').glob('*.csv'),HERE/'derived/fit_closure.json']
    dump(HERE/'provenance/report_build.json',dict(pdf=str(pdf),pdf_sha256=sha(pdf),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in input_paths}))
    print(pdf)
if __name__=='__main__':main()
