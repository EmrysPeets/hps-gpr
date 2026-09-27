"""Scoped native updates for slides 13, 14 and 29; no API calls."""
from pathlib import Path
import json,re
from PIL import Image
B=Path(__file__).resolve().parent
P=json.loads((B/'presentation-before.json').read_text())
C=json.loads((B/'content/slide29_copy.json').read_text())
E={e['objectId']:e for s in P['slides'] for e in s.get('pageElements',[])}
R=[];IM=[];NOTES=[]
def add(k,v):R.append({k:v})
def rg(a,b):return {'type':'FIXED_RANGE','startIndex':a,'endIndex':b}
def tr(oid,geom):
 x,y,w,h=geom;e=E[oid];u=12700 if e['size']['width']['unit']=='EMU' else 1
 return {'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/(e['size']['width']['magnitude']/u),'scaleY':h/(e['size']['height']['magnitude']/u),'translateX':x,'translateY':y,'unit':'PT'}}
def txt(oid):return ''.join(t.get('textRun',{}).get('content','') for t in E.get(oid,{}).get('shape',{}).get('text',{}).get('textElements',[])).rstrip('\n')
def style(oid,a,b,size=13,bold=False,color=None,baseline='NONE'):
 add('updateTextStyle',{'objectId':oid,'textRange':rg(a,b),'style':{'fontFamily':'EB Garamond','fontSize':{'magnitude':size,'unit':'PT'},'bold':bold,'italic':False,'baselineOffset':baseline,'foregroundColor':{'opaqueColor':{'rgbColor':color or {}}}},'fields':'fontFamily,fontSize,bold,italic,baselineOffset,foregroundColor'})
def text(oid,t,geom=None,size=13,bold=False,spacing=0,color=None):
 if txt(oid):add('deleteText',{'objectId':oid,'textRange':rg(0,len(txt(oid)))})
 add('insertText',{'objectId':oid,'insertionIndex':0,'text':t})
 if geom:add('updatePageElementTransform',tr(oid,geom))
 style(oid,0,len(t),size,bold,color)
 add('deleteParagraphBullets',{'objectId':oid,'textRange':rg(0,len(t))})
 add('updateParagraphStyle',{'objectId':oid,'textRange':rg(0,len(t)),'style':{'lineSpacing':100,'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':spacing,'unit':'PT'},'indentStart':{'magnitude':0,'unit':'PT'},'indentEnd':{'magnitude':0,'unit':'PT'},'indentFirstLine':{'magnitude':0,'unit':'PT'}},'fields':'lineSpacing,spaceAbove,spaceBelow,indentStart,indentEnd,indentFirstLine'})
 add('updateShapeProperties',{'objectId':oid,'shapeProperties':{'contentAlignment':'TOP','autofit':{'autofitType':'NONE'}},'fields':'contentAlignment,autofit.autofitType'})
def box(page,oid,t,geom,size):
 x,y,w,h=geom
 add('createShape',{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}})
 E[oid]={'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}}}
 text(oid,t,size=size)
def image(oid,path,geom):
 path=Path(path).resolve();x,y,w,h=geom;iw,ih=Image.open(path).size;s=min(w/iw,h/ih);ww,hh=iw*s,ih*s
 IM.append({'image_uris':str(path),'requests':[{'updatePageElementTransform':tr(oid,(x+(w-ww)/2,y+(h-hh)/2,ww,hh))},{'replaceImage':{'imageObjectId':oid,'url':str(path),'imageReplaceMethod':'CENTER_INSIDE'}}]})
def notes(n,t):
 s=P['slides'][n-1];oid=s['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
 add('deleteText',{'objectId':oid,'textRange':{'type':'ALL'}});add('insertText',{'objectId':oid,'insertionIndex':0,'text':t});NOTES.append({'slide':n,'notes_id':oid,'text':t})
def baseline(oid,start,end,kind):add('updateTextStyle',{'objectId':oid,'textRange':rg(start,end),'style':{'baselineOffset':kind},'fields':'baselineOffset'})
red={'red':.6,'green':0,'blue':0}

# Existing slide13 objects; all saved numerical values retained.
image('rc22_s13_examples',B/'assets/slide13_explained.png',(25,103,670,229))
image('rc23_s13_residual_eq',B/'assets/standardized_residual.png',(34,347,263,45))
t='ri: standardized data–GP difference in mass bin i.\nDenominator: counting and GP uncertainty combined.\n0 = agreement. Green: ±2. Bins are correlated.'
text('rc22_s13_caption',t,(313,340,379,58),13,spacing=2)
baseline('rc22_s13_caption',1,2,'SUBSCRIPT')
notes(13,'The display label is now 2021 10%. The data are the same selected 10% sample, not a luminosity-rescaled 1% sample.\n\nr_i is the standardized residual for mass bin i: r_i=(n_i-b_i)/sqrt(b_i+C_GP,ii). n_i is its observed count, b_i is its held-out GP prediction, and C_GP,ii is the variance of that GP prediction in count space. The denominator combines the Poisson counting variance (approximated by b_i) and GP prediction variance. Thus r_i=0 means agreement, positive values mean data above prediction, and r_i=+2 means two estimated combined standard deviations above it. The green band is a ±2 reference, not calibrated simultaneous coverage.\n\nEach bin is excluded from its own local GP prediction through the same ±2.25sigma rule. Curves join overlapping predictions; this is not one global GP fit. Archived data-derived kernel states are fixed, with the2015 endpoint kernel frozen above90MeV. Residuals across bins are correlated. All1208 plotted bin values are reused exactly from the previous CSV. Search ranges:2015 full19–100MeV,2016 full39–180MeV,2021 10%50–250MeV.')

# Slide14: many centers plus a pointwise conditional reference.
text('h5d7e580f29331fdc_0_255','2021 10%: exclude ±2.25σ at 42 centers across 50–250 MeV',(36,78,650,26),16)
image('rc22_s14_q_equation',B.parent/'unblind_meeting_RCmeet_20260923/science/assets/slide14_deviance_equation.png',(36,108,360,30))
box(P['slides'][13]['objectId'],'rc23b_s14_bins','Nside counts bins, not fit degrees of freedom.',(420,107,273,32),13.5)
baseline('rc23b_s14_bins',1,5,'SUBSCRIPT')
image('rc22_s14_residualscan',B/'assets/slide14_scan_for_deck.png',(28,144,665,193))
t='Typical: near the toy median (~0.95), within the blue reference band.\nObserved: 0.84–0.96. All 42 values lie inside their pointwise 90% bands.\n1 is not an exact target. Unusually high or low values merit checks.\nConditional fitted-sideband checks. The excluded region needs separate validation.'
text('rc22_s14_qualification',t,(36,340,638,61),12.5,spacing=0)
notes(14,(B/'science/findings.md').read_text()+'\n\nObserved D/N ranges0.839924–0.958769; toy medians range0.943620–0.957209. All42 observed values lie inside their own pointwise5th–95th percentile bands. The42 centers are strongly correlated, so this is not42 independent passes or a global goodness-of-fit probability. Typical values are near the median for each center;1 is only a rough scale, not an exact expectation. Both tails matter: unusually small deviance can motivate flexibility/source checks but does not alone prove overfitting. The GP kernels were learned from observed data and held fixed in these toys; kernel selection and source uncertainty are not repeated.\n\nRegular centers are50,55,...,250MeV, plus78MeV for comparison with the earlier display. Underlying count spectra and production±2.25sigma prescription are unchanged. Exact per-center statistics and finite-toy tail intervals:science/data/sideband_center_summary.csv.')

# Slide29: one boxed BEST density equation, matching the previous user's slide.
for oid in ['rc23_s27_yield_eq','rc23_s27_br_eq','rc23_s27_limit_label']:add('deleteObject',{'objectId':oid})
text('rc23_s27_yield_label','BEST yield relation',(33,84,235,27),18,True,color=red)
text('rc23_s27_plot_key','Observed 90% CLs limits',(294,84,390,27),18,True,color=red)
baseline('rc23_s27_plot_key',15,16,'SUBSCRIPT')
page=P['slides'][28]['objectId'];oid='rc23b_s29_equation_box'
add('createShape',{'objectId':oid,'shapeType':'RECTANGLE','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':240,'unit':'PT'},'height':{'magnitude':78,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':30,'translateY':126,'unit':'PT'}}})
add('updateShapeProperties',{'objectId':oid,'shapeProperties':{'shapeBackgroundFill':{'propertyState':'NOT_RENDERED'},'outline':{'propertyState':'RENDERED','outlineFill':{'solidFill':{'color':{'rgbColor':red},'alpha':1}},'weight':{'magnitude':1.2,'unit':'PT'},'dashStyle':'SOLID'}},'fields':'shapeBackgroundFill,outline'})
image('rc23_s27_eps_eq',B/'assets/BEST_yield_to_coupling.png',(37,136,226,58))
t='Nsigup: 90% signal-yield limit\nfrad: radiative background fraction\ndNbkg/dm: local background density\nNf = 1 / BR(A′ → e⁺e⁻)\nα: fine-structure constant\nmA′: tested mass'
text('rc23_s27_symbols',t,(30,217,240,129),14.5,spacing=4)
baseline('rc23_s27_symbols',1,4,'SUBSCRIPT');baseline('rc23_s27_symbols',4,6,'SUPERSCRIPT')
for pat,offset in [('frad',1),('dNbkg',2),('Nf =',1),('mA′',1)]:
 pos=t.index(pat);end=pos+len(pat) if pat!='Nf =' else pos+2;baseline('rc23_s27_symbols',pos+offset,end,'SUBSCRIPT')
citation='BEST: Bjorken et al., Phys. Rev. D 80, 075018 (2009), Eq. (19), rearranged.'
text('rc23_s27_br_key',citation,(36,368,638,26),11.5)
add('updateTextStyle',{'objectId':'rc23_s27_br_key','textRange':rg(0,len(citation)),'style':{'link':{'url':C['citation_url']},'foregroundColor':{'opaqueColor':{'rgbColor':red}},'underline':False},'fields':'link,foregroundColor,underline'})
notes(29,C['citation_full']+'\n'+C['citation_url']+'\n\n'+'\n\n'.join(C['notes'])+'\n\nLayout reference: '+C['reference']['url']+'\nThe displayed density equation follows BEST Eq.(19) after N_rad=f_rad(dN_bkg/dm)delta_m and cancellation ofdelta_m. It is labeled as rearranged. The plot object, scale and position are unchanged from the preceding revision. The current confidence level remains90%CLs, not the95%historical example. The equation includes the phase-space-aware inverse electron branching factor N_f; the displayed curves already contain this correction.')

(B/'requests.json').write_text(json.dumps({'presentation_id':P['presentationId'],'revision':P['revisionId'],'scope':[13,14,29],'requests':R,'images':IM,'notes':NOTES},indent=2)+'\n')
print(len(R),'native requests;',len(IM),'image replacements')
