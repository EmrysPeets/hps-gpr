"""Build the standalone, page-planned v6.4 report and its readable Markdown source."""
from pathlib import Path
import html, re, json
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

B=Path(__file__).resolve().parents[1]
D=pd.read_csv(B/'results/centers_and_shapes.csv').set_index('mass_MeV')
S=pd.read_csv(B/'results/shape_comparisons.csv')
R=pd.read_csv(B/'inputs/reference_2021/core_native_diagnostics.csv').set_index('mass_MeV')
LAWS=json.loads((B/'results/location_width_laws.json').read_text())
SUMMARY=json.loads((B/'results/summary.json').read_text())
W=500
for name,file in [('StudySerif','DejaVuSerif.ttf'),('StudySerif-Bold','DejaVuSerif-Bold.ttf'),
                  ('StudySerif-Italic','DejaVuSerif-Italic.ttf'),('StudySerif-BoldItalic','DejaVuSerif-BoldItalic.ttf'),
                  ('StudySans','DejaVuSans.ttf'),('StudyMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(B/'inputs/fonts'/file)))
pdfmetrics.registerFontFamily('StudySerif',normal='StudySerif',bold='StudySerif-Bold',italic='StudySerif-Italic',boldItalic='StudySerif-BoldItalic')
styles={
 'body':ParagraphStyle('body',fontName='StudySerif',fontSize=10.25,leading=14,spaceAfter=8),
 'title':ParagraphStyle('title',fontName='StudySerif-Bold',fontSize=22,leading=26,spaceAfter=10),
 'head':ParagraphStyle('head',fontName='StudySerif-Bold',fontSize=16,leading=20,spaceAfter=10),
 'sub':ParagraphStyle('sub',fontName='StudySerif-Bold',fontSize=11,leading=15,spaceAfter=7),
 'caption':ParagraphStyle('caption',fontName='StudySerif',fontSize=8.5,leading=11,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='StudySerif',fontSize=9,leading=12,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='StudySerif',fontSize=8.2,leading=10.5),
 'code':ParagraphStyle('code',fontName='StudyMono',fontSize=8,leading=11,spaceAfter=8),
}
story=[];md=[]

def p(text,kind='body'):
    story.append(Paragraph(text,styles[kind]))
    plain=html.unescape(re.sub('<[^>]*>','',text))
    md.append(plain+'\n')

def page(title,first=False):
    if not first:story.append(PageBreak())
    p(title,'title' if first else 'head')
    md[-1]='# '+md[-1]

def fig(name,width=W):
    path=B/'figures'/f'{name}.png'
    with PILImage.open(path) as im:w,h=im.size
    story.append(Image(str(path),width=width,height=width*h/w))
    story.append(Spacer(1,5))
    md.append(f'![{name}](../figures/{name}.png)\n')

def table(headers,rows,widths=None,small=False):
    if widths:widths=[w*W/sum(widths) for w in widths]
    data=[[Paragraph('<b>'+str(x)+'</b>',styles['cell']) for x in headers]]
    data.extend([[Paragraph(str(x),styles['cell']) for x in row] for row in rows])
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff3')),
        ('LINEBELOW',(0,0),(-1,0),.65,colors.HexColor('#526b7a')),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#526b7a')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),3 if small else 5),
        ('BOTTOMPADDING',(0,0),(-1,-1),3 if small else 5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
    ]))
    story.append(t);story.append(Spacer(1,9))
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#c2ccd2'));c.setLineWidth(.4);c.line(50,755,562,755)
    c.setFont('StudySans',8);c.setFillColor(colors.HexColor('#4e5b63'))
    c.drawString(50,766,'HPS GPR: 2016 selected signal-MC shapes')
    c.drawRightString(562,766,'Version 6.4 | 25 September 2026')
    c.drawCentredString(306,28,str(doc.page))

def main():
    page('Where does the 2016 signal reconstruct?',first=True)
    p('Core locations, mass resolution and full-shape comparisons','sub')
    p('<b>The result.</b> The supplied smeared and scaled 2016 prompt target-constrained signal histograms have a small downward core displacement over 40-175 MeV. The fitted shifts range from -0.08 to -1.02 MeV. The sign agrees with the earlier 2021 study, but the displacement and non-Gaussian tail contribution are substantially smaller in these 2016 inputs.')
    p('<b>What this suggests for the analysis.</b> Use the native 2016 histograms as the starting signal templates. Between simulated masses, interpolation of nearby 2016 shapes performs better in the checks here than either one common shape or a Gaussian. A shifted Gaussian remains a useful description of the core, provided its fitted yield is not assumed to be the full selected signal yield.')
    fig('centers_and_2021')
    p('<b>Figure 1. How to read the figure.</b> Left: fitted core, median and binned mean need not coincide because the distributions have tails. Right: the same local core definition is applied to 2016 and compared with the saved 2021 results. Bars are MC bin-resampling standard deviations (64 replicas in 2016; 32 in 2021), not detector calibration uncertainties. Reconstruction and selection equivalence between years has not been established.','caption')
    table(['Generated mass','2016 core center','Core shift','MC SD of center'],
          [[f'{m} MeV',f'{D.loc[m,"center_MeV"]:.3f} MeV',f'{D.loc[m,"shift_MeV"]:+.3f} MeV',f'{D.loc[m,"center_mc_sd_MeV"]:.3f} MeV'] for m in [60,100,160]], [118,132,132,130])
    p('<b>Inputs and scope.</b> There are 29 ROOT files, containing 3,770,135 selected entries in total, at 30-175 MeV in nominal 5 MeV steps; 150 MeV is missing. All results use <font name="StudyMono" size="8">h_MinvScSm_GeneralLargeBins_Final_1</font>. Its FEE momentum scaling and smearing are already applied according to the supplied sample description. No extra smearing is applied here. The 30 and 35 MeV samples are shown separately; no observed-data or GP extraction is performed.','small')

    page('1. What do we mean by the central location?')
    p('The central location is not unique for an asymmetric distribution. A Gaussian fit near the peak answers where the core lies. The median divides the selected probability in half. The mean includes the entire tail. This report shows all three, with the fitted core as its primary location so that the comparison follows the v6.3 series.')
    fig('core_fit_examples')
    p('<b>Figure 2.</b> Points are native 0.625 MeV bins with square-root-count bars. Blue curves are local Gaussian-plus-pedestal fits, drawn only over their fitted bins. Green curves show the Gaussian component of that local fit. The vertical red line is the generated mass, and the blue dashed line marks the fitted core. The pedestal represents broad signal shoulders locally; these are signal-MC histograms, so it is not a fitted collision-background component.','caption')
    p('<b>The local fit.</b> The smoothed mode is located within three reference analysis widths of the generated mass. The unsmoothed bin counts within 1.5 reference widths of that mode are then fitted with a bin-integrated Gaussian plus a nonnegative affine pedestal. The Gaussian area, center and width, and the two pedestal endpoints, are free. This is the location convention inherited from v6.1 and used in v6.3.6.')
    p('For each included bin [a<sub>i</sub>, b<sub>i</sub>], the expected count is A[Phi((b<sub>i</sub>-c)/s) - Phi((a<sub>i</sub>-c)/s)] plus the affine pedestal. The fit minimizes Poisson deviance. Integrating across each bin matters because a bin is 0.625 MeV wide; locating a peak only by its tallest bin would quantize the answer.')
    p('<b>What the error bars mean.</b> Sixty-four independent Poisson resamplings of the histogram bins are refitted, including the mode search. The reported center error is the standard deviation of the valid replica centers. All 1,728 replicas in the primary 40-175 MeV domain are valid. This measures finite-MC variation under the independent-bin assumption, conditional on the chosen fit. It does not include calibration, detector or selection uncertainty.')
    p('<b>How much the definition matters.</b> The fit is repeated with half-widths 1.25 and 2.0 reference widths, and without a pedestal at 1.5 and 2.0 widths. The largest absolute change in center is retained as a definition spread, not added in quadrature to the MC error. At 60, 100 and 160 MeV the spreads are 0.044, 0.084 and 0.396 MeV. The 160 MeV center is therefore more definition-dependent than its statistical bar alone suggests.')

    page('2. How wide is the reconstructed signal?')
    p('The supplied histograms already describe smeared, scaled reconstruction. Their fitted core widths increase from 1.56 MeV at 40 MeV to 7.06 MeV at 175 MeV. A width extracted from the core and a width describing the entire distribution answer different questions; they should not be substituted for one another without stating the change.')
    fig('widths_and_tails')
    p('<b>Figure 3.</b> Left: the fitted Gaussian core width, the existing 2016 analysis-resolution curve and half the 16%-84% quantile interval. Right: probability on each side outside two fitted core widths. The Gaussian reference is 2.28% per side. The curves use full selected normalization, and the fitted-core-width bars are the MC resampling standard deviations.','caption')
    p('Over 40-175 MeV the fitted core width is 0.84-0.99 times the existing reference analysis width. The central 68% interval is generally wider than the fitted core width because it includes the shoulders. Neither result is an unsmeared resolution: the unsmeared histogram is only inventoried for presence and is not fitted in this study.')
    table(['Quantity','What it measures','How it is used here'],[
        ['Fitted core width s','Width of the local Gaussian component','Aligns cores and defines the standardized coordinate u'],
        ['Reference width s_ref','Existing 2016 analysis prescription','Defines the local-fit range and reference window comparisons'],
        ['(q84 - q16)/2','Half the central 68% probability interval','Describes the selected distribution without a Gaussian model'],
    ],[114,191,207])
    p('Between 89.2% and 93.7% of the selected probability lies within two fitted core widths; a Gaussian would put 95.45% there. Thus the 2016 samples have relatively compact cores but still have excess tail probability. From 50 MeV upward the lower-mass tail is larger than the upper-mass tail in this definition. At 40 MeV the upper tail is larger, so one universal asymmetric tail would hide a low-mass change.')
    p('The reference resolution is copied from the archived 2016 spectrum configuration. In GeV, with m also in GeV, it is 0.000380 + 0.0410 m - 0.270 m<super>2</super> + 3.490 m<super>3</super> - 11.11 m<super>4</super>. It is a comparison curve in this study; the analysis configuration is not modified.','small')

    page('3. Overlaying the histograms without hiding their tails')
    fig('pole_aligned_overlays')
    p('<b>Figure 4.</b> Each colored curve is a native 2016 histogram, divided by its own full selected count. Left: subtracting only the generated mass leaves the mass-dependent widths visible. Right: dividing that coordinate by the existing analysis width brings the distributions closer together. The dotted Gaussian is centered at the generated mass. The logarithmic vertical axis exposes tail differences.','caption')
    fig('core_aligned_overlays')
    p('<b>Figure 5.</b> The coordinate is u = (reconstructed mass - fitted core center) / fitted core width. Left: each curve is normalized within |u| &lt; 2 to compare the central shapes. Right: each curve retains its original full selected probability. The dashed black curve is an equal-mass average of the 27 aligned empirical distributions, not an analytic fit. The dotted curve is a standard Gaussian.','caption')
    p('<b>Answer.</b> Alignment produces similar central shapes, but it does not make the full distributions identical. The left panel deliberately removes differences in core probability; the right panel restores them. That is why an attractive overlay of normalized peaks is not, by itself, evidence that one common template describes the full selected signal.','small')

    page('4. Is this the same shift and shape as in 2021?')
    p('There is a qualitatively similar downward core shift, but a direct transfer of the 2021 correction would be too large for these 2016 samples. At 60, 100 and 160 MeV the 2021 shifts are -1.208, -2.312 and -3.342 MeV, compared with -0.188, -0.309 and -1.017 MeV here. The distinction is present in both the locations and the full shapes.')
    fig('cross_year_shapes')
    p('<b>Figure 6.</b> Each year is aligned using its own fitted core center and width. Both retain full selected normalization, including the saved 2021 overflow normalization. The percentages in each panel give the probability within two core widths. The 2016 samples are more concentrated around their cores. These are comparisons of supplied samples, not a controlled attribution of the difference to run year, beam energy or calibration.','caption')
    p('<b>A compact description of the 2016 shift.</b> For 40-175 MeV, an equal-weight affine fit gives the following descriptive relation, with all masses in MeV:')
    p('<b>c(m) = m - 0.363673 - 0.563179 (m - 100) / 100.</b>')
    p('Leaving out each mass in turn gives an RMS prediction error of 0.103 MeV and a maximum absolute error of 0.351 MeV. A quadratic reduces the RMS only to 0.102 MeV. The affine relation is therefore a concise trend summary; native centers and interpolation are preferable where local structure matters. No extrapolation below 40 or above 175 MeV is qualified.')
    table(['Shift model','Parameters','Omitted-mass RMS (MeV)','Largest error (MeV)'],
          [[x['model'].capitalize(),len(x['coefficients']),f'{x["loo_rms"]:.3f}',f'{x["loo_max_abs"]:.3f}'] for x in LAWS if x['target']=='shift_MeV'],[133,68,175,136],small=True)
    p('This is a residual displacement in selected MC after the stated smearing and scaling. These histograms do not identify its physical cause or establish a data mass-scale correction. The logarithmic 2021 shift law should not be imported as the 2016 center prescription.','small')

    page('5. Which signal shape is supported by these checks?')
    p('The main comparisons use a Gaussian with the measured core center and width; an equal-mass common empirical shape made without that mass; and an interpolation of the nearest remaining lower and upper samples. The interpolation predicts center, width and standardized shape without using the omitted histogram. A second Gaussian control uses the full histogram mean and RMS, so the Gaussian comparison is not restricted to its local core width.')
    fig('shape_holdout')
    p('<b>Figure 7.</b> The vertical axis is the largest absolute difference between cumulative probabilities, in percentage points. A value of 1 means that at some mass threshold the models differ by one percentage point of total probability. It is a shape discrepancy, not a p-value. Core-only comparisons condition on |u| &lt; 2 and use the measured center and width. The full neighbor-morph check also predicts those quantities.','caption')
    table(['Full-distribution approximation','Median difference','Range','Mass tests'],[
        ['Gaussian at measured core',f'{100*S.gaussian_full_cdf_distance.median():.2f} pp',f'{100*S.gaussian_full_cdf_distance.min():.2f}-{100*S.gaussian_full_cdf_distance.max():.2f} pp','27'],
        ['Gaussian at full mean and RMS',f'{100*S.moment_gaussian_full_cdf_distance.median():.2f} pp',f'{100*S.moment_gaussian_full_cdf_distance.min():.2f}-{100*S.moment_gaussian_full_cdf_distance.max():.2f} pp','27'],
        ['Common shape, omitted mass excluded',f'{100*S.common_loo_full_cdf_distance.median():.2f} pp',f'{100*S.common_loo_full_cdf_distance.min():.2f}-{100*S.common_loo_full_cdf_distance.max():.2f} pp','27'],
        ['Neighbor morph, omitted mass excluded',f'{100*S.morph_full_cdf_distance.median():.2f} pp',f'{100*S.morph_full_cdf_distance.min():.2f}-{100*S.morph_full_cdf_distance.max():.2f} pp','25'],
    ],[223,104,103,82])
    p('<b>Preferred full-shape starting point.</b> Use a native histogram at its generated mass. Between mass points, linearly interpolate the center, interpolate the logarithm of the core width, and mix the neighboring cumulative distributions in the aligned coordinate. The returned probabilities retain the complete selected normalization. The largest held-out discrepancy is 0.76 percentage points, at 45 MeV; endpoints have no two-sided holdout test.')
    p('<b>A simpler core approximation.</b> A Gaussian with a 2016-derived center and width can describe the main peak, and the common empirical shape is a useful intermediate comparison. Neither reproduces every tail. The 150 MeV gap may be interpolated between 145 and 155 MeV, but this is a model prediction, not an available MC sample.')
    p('These deterministic comparisons include finite-MC noise. They do not validate arbitrary intermediate masses, define a template-systematic confidence interval, or establish fitted-yield recovery. A 2016 injection study would be needed to test how each shape interacts with the background fit and yield inference.','small')

    page('6. What changes when the center or window width changes?')
    fig('containment_and_low_mass')
    p('<b>Figure 8.</b> Left: the empirical fraction inside windows extending 2.25 widths on either side. Changing the center at fixed analysis width has a small effect; replacing the analysis width with the fitted core width is a separate change. Right: the sparse low-mass samples are retained visibly instead of being pooled into the common shape. Local plotting ranges do not renormalize any histogram.','caption')
    p('Over 40-175 MeV, the original generated-mass-centered window with the reference width contains 95.28-96.12% of the selected MC. Moving only its center to the fitted core gives 95.25-96.29%. The change is between -0.047 and +0.178 percentage points. It need not be positive at every mass because centering on the core does not optimize an asymmetric full-distribution integral.')
    p('A Gaussian puts 97.56% inside its own 2.25-width window. The remaining difference here is a tail-shape issue, even though the core displacement is small. These are geometric fractions of selected signal histograms, not selection efficiencies, calibrated signal-recovery fractions or an optimization of the GP training mask.')
    p('<b>30 MeV: no reliable fitted core is reported.</b> Only 84 entries survive in the supplied histogram. The primary fit hits its minimum-width bound, and only 34 of 64 resampling fits pass the locator checks. The binned mean is 34.10 MeV and the median is 31.81 MeV, but neither makes the local Gaussian fit reliable. The raw attempted fits remain in the numerical ledger; their apparent narrow width is not used.')
    p('<b>35 MeV: a separate diagnostic.</b> This histogram has 2,603 entries. Its center is 35.276 MeV in the primary convention, but its fitted core width changes from about 0.91 to 1.44 MeV when the pedestal-fit half-width changes from 1.5 to 2.0 reference widths. The mean is 35.695 MeV. This broad-definition sensitivity motivates keeping 35 MeV out of the primary common-shape and smooth-law fits.')
    p('<b>40-175 MeV: the primary comparison domain.</b> All 27 available primary core fits and all their MC resampling fits pass. The 40 MeV point remains in this domain and is the largest common-shape discrepancy, so excluding 30 and 35 MeV does not hide the remaining low-mass shape change. No selection, signal window or background-training guard is changed by this report.')

    page('7. Native central locations and shape summary')
    p('Masses and widths are in MeV. c is the fitted core center; SD is its conditional MC resampling error; s is the fitted core width. F2 is the full selected fraction within c +/- 2s. Spread is the largest center change across the four fit-definition checks. Values are rounded here; the CSV retains more precision.','small')
    headers=['m','Entries','c','c - m','SD(c)','s','Median','Mean','F2 (%)','Spread']
    rows=[]
    for m,r in D.iterrows():
        vals=[str(m),f'{int(r.entries):,}']
        vals += [f'{r.center_MeV:.3f}',f'{r.shift_MeV:+.3f}',f'{r.center_mc_sd_MeV:.3f}',f'{r.sigma_core_MeV:.3f}'] if r.valid else ['--']*4
        vals += [f'{r.median_MeV:.3f}',f'{r.mean_MeV:.3f}',f'{100*r.core_fraction_2:.1f}' if r.valid else '--',f'{r.definition_spread_MeV:.3f}' if r.valid else '--']
        rows.append(vals)
    table(headers,rows,[26,54,54,48,43,44,57,57,43,46],small=True)
    p('The 30 MeV fit is invalid; its location and width are not promoted to measurements. The 35 MeV row is diagnostic and is excluded from the primary 40-175 MeV averages and interpolation prescription. There is no 150 MeV input. Means and quantiles use bin-midpoint or piecewise-uniform within-bin conventions; event-level values cannot be recovered from these histograms.','small')

    for n in range(1,4):
        page(f'Appendix A{n}. Every native mass histogram')
        fig(f'catalogue_{n}',width=488)
        p(f'<b>Figure {8+n}.</b> Native MC is shown with full selected normalization. Red dotted curves are Gaussians at the generated mass with the reference analysis width. Blue dashed curves are unit-area Gaussians at the fitted core with its fitted width; they are comparison shapes, not the local Gaussian component plus pedestal of Figure 2. The logarithmic scale exposes tails. No blue curve is shown for the invalid 30 MeV core fit. Displayed ranges omit distant bins without rescaling the visible probability.','caption')

    page('Appendix B. Reproducibility and interpretation')
    p('<b>Input identity.</b> The 29 supplied ROOT files are copied unchanged from the Downloads/2016_MC_Histograms directory into inputs/root. Their names, byte sizes and SHA-256 hashes are recorded in provenance/source_files.json. Only the smeared/scaled Final_1 mass histogram is analyzed. Every target is a 400-bin TH1D on 0-0.25 GeV, converted once to 0-250 MeV; bin width is 0.625 MeV. All target counts are nonnegative integers, entries equal the bin sum, variances equal counts, and underflow and overflow are zero.')
    p('<b>Provenance boundaries.</b> The prompt-target-constrained sample identity and prior FEE scaling/smearing are supplied by the user. ROOT axis titles are blank and these files have no embedded fits. The files do not independently document the full selection or certify its equivalence to the archived data analysis. Generated masses are read from filenames. No efficiency, rate, exclusion, data mass calibration or discovery claim is derived.')
    p('<b>Saved references.</b> The 2021 centers, errors and histogram extracts are frozen from v6.1, the same inputs used by the introductory pages of v6.3.6. The local-core algorithm and the 2016 reference-resolution coefficients are also pinned. Earlier PDFs are accessed read-only. A separate before/after hash audit records a concurrent v6.3.6 PDF change; no earlier file is overwritten or restored by this study. No remote computation is launched.')
    p('<b>Rebuilding.</b> The package uses Python with NumPy, SciPy, pandas, uproot, Matplotlib, Pillow and ReportLab. The recorded runtime is in provenance/runtime.json. From the extracted study directory, run the following with one numerical thread:','small')
    for cmd in ['export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
                'python3 scripts/analyze.py', 'python3 scripts/make_figures.py',
                'python3 scripts/build_report.py', 'python3 scripts/validate.py']:
        p(cmd,'code')
    table(['Artifact','Purpose'],[
        ['results/centers_and_shapes.csv','Native locations, widths, probabilities and definition spreads'],
        ['results/bootstrap_fits.csv','Every attempted resampling fit, including failures'],
        ['results/fit_definition_sensitivity.csv','Alternative ranges and Gaussian-only local fits'],
        ['results/shape_comparisons.csv','Gaussian, common-shape and neighbor holdout discrepancies'],
        ['histograms/ and figures/','Portable counts, normalized distributions and PDF/PNG figures'],
        ['scripts/template.py','Native or interpolated empirical CDF, with outside-support categories'],
        ['qa/ and MANIFEST.sha256','Numerical, PDF, relocation and artifact-integrity checks'],
    ],[235,277],small=True)
    p('<b>What follows from this study.</b> The supplied 2016 MC supports a modest downward core displacement, relatively Gaussian central shapes and measurable non-Gaussian tails. Its own native and interpolated histograms are the most directly supported starting templates. Establishing their use in the 2016 yield analysis still requires selection equivalence and injection/recovery checks using the intended background procedure.','small')

    pdf=B/'pdf/HPS_GPR_v6p4_2016_MC_Shapes.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=(612,792),rightMargin=50,leftMargin=50,topMargin=49,bottomMargin=43,
                         title='HPS GPR v6.4: 2016 MC central locations and signal shapes',author='HPS analysis study',
                         subject='Smeared, FEE-scaled 2016 prompt target-constrained signal MC',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (B/'source/report.md').write_text('\n'.join(md))
    print(pdf)

if __name__=='__main__':main()
