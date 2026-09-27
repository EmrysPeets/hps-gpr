#!/usr/bin/env python3
"""Create two short explanatory pages and combine them with the vector figure."""
from pathlib import Path
import csv
import json
import shutil
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader, PdfWriter

BASE=Path(__file__).resolve().parents[1]
OUT=BASE.parents[1]/'output'/'pdf'/BASE.name
OUT.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyV57',fontName='Helvetica',fontSize=10.5,
                         leading=14.2,spaceAfter=9,textColor=colors.HexColor('#273442')))
styles.add(ParagraphStyle(name='SmallV57',fontName='Helvetica',fontSize=9.2,
                         leading=12.1,spaceAfter=7,textColor=colors.HexColor('#273442')))
styles.add(ParagraphStyle(name='HeadV57',fontName='Helvetica-Bold',fontSize=13,
                         leading=17,spaceBefore=9,spaceAfter=7,textColor=colors.HexColor('#163c61')))
styles['Title'].fontSize=21;styles['Title'].leading=26;styles['Title'].alignment=TA_LEFT
story=[]
def p(s,style='BodyV57'):story.append(Paragraph(s,styles[style]))
def head(s):p(s,'HeadV57')
def table(rows,widths):
    obj=Table([[Paragraph(str(x),styles['SmallV57']) for x in row] for row in rows],colWidths=widths)
    obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff6')),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#93a9bc')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([obj,Spacer(1,11)])

p('Reading the v5.7 figure','Title')
p('13 September 2026 | Prompt A-prime angular model and existing HPS pair spectra','SmallV57')
p('At fixed decay geometry, raising the beam energy shifts the accepted mass window upward. '
  'The figure makes that kinematic effect visible and places the recorded spectra beneath it. '
  '<b>The calculated curves are conditional angular fractions, not calibrated HPS signal efficiencies.</b>')
head('The 15 mrad scale and the upper edge')
p('For an on-axis parent with energy E = x E(beam), equal-energy massless daughters have '
  'm = E sin(theta). Here theta is one track\'s angle to the beam; the pair opening angle is 2 theta. '
  'At x = 1, the reference values are:')
table([['Run','Beam energy<br/>(GeV)','15 mrad scale<br/>(MeV)','70 mrad model edge<br/>(MeV)'],
       ['2015','1.056','15.84','73.86'],['2016','2.300','34.50','160.87'],
       ['2021','3.740','56.10','261.59']],[58,104,132,206])
p('<b>A 15 mrad vertical gap alone imposes no upper mass cutoff.</b> The 70 mrad polar cap is an '
  'illustrative assumption motivated by a historical design requirement [4]. It is not a verified '
  'outer boundary of the complete HPS apparatus. The first row shows a vertical slice with angles '
  'enlarged for clarity, not a detector engineering drawing.')
p('The lower scales are conditional too: x = 0.8 moves them to 12.67, 27.60 and 44.88 MeV. '
  'A real parent has an energy and direction distribution. The seventh tracking layer and positron '
  'hodoscope used in 2021 also changed the detector response [1]. The common angular window isolates '
  'the energy dependence; it does not reproduce those changes. Neglecting electron mass shifts the '
  'tabulated x = 1 symmetric edges by less than 0.04 MeV.')
head('What is integrated in the middle row')
p('The parent decays promptly along +z. Its transverse-vector rest-frame distribution is '
  '(3/8)(1 + c<super>2</super>), with c = cos(theta*) and uniform azimuth. Both boosted daughters '
  'must be forward and clear the vertical cut |atan(py/pz)| &gt;= 0.015. The colored curves add '
  'atan(pT/pz) &lt;= 0.070. Grey dotted: gap only, x = 1. Solid: both cuts, x = 1. '
  'Dashed: both cuts, x = 0.8. The azimuth is integrated analytically and c numerically.')
p('The denominator is all decays in this specified angular model. No production-spectrum weighting, '
  'polarization mixture, material, field transport, finite sensors, hit selection, trigger, reconstruction '
  'or displaced-decay acceptance is included. A physical efficiency curve requires matched simulated '
  'signal denominators and run-specific selections; a complete three-run set was not established here.')

story.append(PageBreak())
p('Data, sources and validation','Title')
head('Reading the bottom row')
p('The curves are raw selected pairs per MeV in 1 MeV bins. They have no exposure or luminosity '
  'rescaling. They show the mass content of background-dominated samples, not A-prime production '
  'rates. Different selections and exposures prevent an energy-only interpretation of their heights.')
table([['Sample','Histogram','Displayed pairs'],
       ['2015 full','invariant_mass','21,442,838'],
       ['2016 full','h_Minv_General_Final_1','73,218,251'],
       ['2021 10%','preselection/h_invM_8000','141,305,897']],[94,275,131])
p('The inherited display crops are 15, 30 and 36 MeV; the figure ends at 300 MeV. The 2015 input '
  'ends at 150 MeV, so its grey region above 150 denotes unavailable bins. These crops and endpoints '
  'are not detector acceptance boundaries. The native files, exact bin values and source SHA-256 '
  'hashes are saved with the study. Missing/cropped CSV bins are marked explicitly.')
p('Historical HPSTR radiative-MC acceptance products were inspected but not substituted for signal '
  'efficiency: they use SIMP control selections and SLIC-level truth denominators, with unverified '
  'equivalence to these data samples. No matched 2015 counterpart was established.','SmallV57')
head('Primary references')
refs=[
 ('[1] N. Baltzell et al., The Heavy Photon Search Experiment (2022). Table I: beam energies; detector section: upgrades.', 'https://arxiv.org/abs/2203.08324'),
 ('[2] P. H. Adrian et al., original 2015 dark-photon search (2018), p. 2: 1.056 GeV.', 'https://arxiv.org/abs/1807.11530'),
 ('[3] P. H. Adrian et al., 2016 prompt/displaced search (2023), pp. 2, 5: 2.3 GeV and 15 mrad active edge.', 'https://arxiv.org/abs/2212.10629'),
 ('[4] M. Battaglieri et al., HPS Test Detector (2015): approximately 15-70 mrad historical design criterion.', 'https://arxiv.org/abs/1406.6115'),
]
for label,url in refs:p(label+f' <link href="{url}" color="#2166ac">Source</link>.','SmallV57')
p('Additional 2021 trigger/acceptance context: V. Kubarovsky, 15 November 2021, slide 4 '
  '(<link href="https://indico.jlab.org/event/496/contributions/9081/attachments/7389/10202/Kubarovsky_2021_11_15_HPS_trigger.pdf" color="#2166ac">presentation</link>). '
  'Its curves were not digitized or used as the model efficiency.','SmallV57')
head('Checks and reproducibility')
p('Count-preserving rebinning, fraction bounds, mass endpoints and mass/energy scaling passed. '
  'An independent two-dimensional integration of boosted momenta agrees with the analytic angular '
  'calculation within 0.000105 absolute acceptance at the checked points. A separate physics review '
  'confirmed the model formulas. Rendered-page inspection and extracted-text checks accompany the '
  'numerical results. These checks validate the model implementation, not the omitted detector effects.','SmallV57')
p('Reproduce with scripts/make_acceptance.py and scripts/make_methods.py. One numerical worker, '
  'BLAS/OMP threads capped at one; no toys or detector simulation. CPU use was coordinated with '
  'the existing v5.5 and v5.6 tasks. See README.md, references.json, derived/ and qa/.','SmallV57')

def footer(canvas,doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#667580'))
    canvas.drawString(56,28,'HPS v5.7 | Conditional angular acceptance')
    canvas.drawRightString(556,28,f'Explanatory page {doc.page}')

methods=BASE/'figures'/'HPS_v5p7_methods.pdf'
SimpleDocTemplate(str(methods),pagesize=letter,rightMargin=56,leftMargin=56,
                  topMargin=43,bottomMargin=45,title='HPS v5.7 figure methods and sources').build(
                      story,onFirstPage=footer,onLaterPages=footer)
assert len(PdfReader(methods).pages)==2
writer=PdfWriter()
writer.append(BASE/'figures'/'HPS_v5p7_Aprime_acceptance.pdf')
writer.append(methods)
writer.add_metadata({'/Title':'HPS v5.7: angular acceptance and selected mass spectra',
                     '/Subject':'Conditional angular model; raw selected data; methods and provenance'})
combined=OUT/'HPS_v5p7_Aprime_acceptance_study.pdf'
with combined.open('wb') as f:writer.write(f)
for suffix in ('.pdf','.png'):
    shutil.copy2(BASE/'figures'/('HPS_v5p7_Aprime_acceptance'+suffix),OUT/('HPS_v5p7_Aprime_acceptance'+suffix))
print(combined)
