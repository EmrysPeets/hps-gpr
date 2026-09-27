"""Targeted native text/layout requests; no live API writes.

Body placeholders are reused. Existing mixed paragraphs on slide 46 are edited
individually, preserving native custom dash bullets. New explanatory copy is
native EB Garamond. Plot/equation images are supplied separately in `images`.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / 'presentation-before.json').read_text())
REQ, IMAGES, CHANGELOG = [], [], []
OBJECTS = {e['objectId']: e for s in P['slides'] for e in s.get('pageElements', [])}
FONT='EB Garamond'

def rng(a,b): return {'type':'FIXED_RANGE','startIndex':a,'endIndex':b}
def add(kind, data): REQ.append({kind:data})
def text_of(oid):
    return ''.join(x.get('textRun',{}).get('content','') for x in OBJECTS[oid]['shape']['text']['textElements'])
def frame(oid,box):
    e=OBJECTS.get(oid)
    x,y,w,h=box
    if e:
        sw=e['size']['width']['magnitude']; sh=e['size']['height']['magnitude']
        if e['size']['width']['unit']=='EMU': sw/=12700;sh/=12700
    else: sw,sh=w,h
    add('updatePageElementTransform',{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/sw,'scaleY':h/sh,'translateX':x,'translateY':y,'unit':'PT'}})
def style(oid,a,b,size=17,bold=False,color=None):
    st={'fontFamily':FONT,'fontSize':{'magnitude':size,'unit':'PT'},'bold':bold,'italic':False,'foregroundColor':{'opaqueColor':{'rgbColor':color or {'red':0,'green':0,'blue':0}}}}
    add('updateTextStyle',{'objectId':oid,'textRange':rng(a,b),'style':st,'fields':'fontFamily,fontSize,bold,italic,foregroundColor'})
def paras(oid,a,b,spacing=5,indent=0):
    add('updateParagraphStyle',{'objectId':oid,'textRange':rng(a,b),'style':{'lineSpacing':100,'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':spacing,'unit':'PT'},'indentStart':{'magnitude':indent,'unit':'PT'},'indentEnd':{'magnitude':0,'unit':'PT'},'indentFirstLine':{'magnitude':indent,'unit':'PT'}},'fields':'lineSpacing,spaceAbove,spaceBelow,indentStart,indentEnd,indentFirstLine'})
def replace(oid,txt,box,size=17,bold=False,spacing=5):
    # All replacements here are instruction placeholders, not mixed source text.
    old=text_of(oid)
    if old.endswith('\n'): old=old[:-1]
    if old: add('deleteText',{'objectId':oid,'textRange':rng(0,len(old))})
    add('insertText',{'objectId':oid,'insertionIndex':0,'text':txt})
    add('deleteParagraphBullets',{'objectId':oid,'textRange':rng(0,len(txt))})
    frame(oid,box)
    style(oid,0,len(txt),size,bold)
    paras(oid,0,len(txt),spacing)
    add('updateShapeProperties',{'objectId':oid,'shapeProperties':{'contentAlignment':'TOP','autofit':{'autofitType':'NONE'}},'fields':'contentAlignment,autofit.autofitType'})
def box(slide,slug,txt,geom,size=17,bold=False,spacing=5):
    oid=f'rcmeet_s{slide:02d}_{slug}'
    x,y,w,h=geom
    add('createShape',{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':P['slides'][slide-1]['objectId'],'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}})
    add('insertText',{'objectId':oid,'insertionIndex':0,'text':txt})
    style(oid,0,len(txt),size,bold); paras(oid,0,len(txt),spacing)
    return oid
def image(slide,slug,filename,geom):
    IMAGES.append({'slide':slide,'pageObjectId':P['slides'][slide-1]['objectId'],'objectId':f'rcmeet_s{slide:02d}_{slug}','local_path':str(ROOT/'science/assets'/filename),'frame':dict(zip(['x','y','width','height'],geom)),'fit':'contain','preserve_aspect_ratio':True})

# 18: preserve meaningful source screenshot; move it clear of the title/body.
# Reconstruct distinct heading/body roles explicitly; existing source text keeps
# its native 17 pt family. Exact specialized math is a separately rendered asset.
replace('h5d7e580f29331fdc_0_360','Validation metrics',(36,83,302,30),17,True)
image(18,'pull_equation','equation18_pull.png',(36,112,302,65))
box(18,'symbol_key','Â: fitted yield; Ainj: injected yield.\nσA: this toy’s post-injection fit error.',(36,179,302,34),13)
box(18,'pull_mean','Pull mean → yield bias\nTarget: mean near 0.',(36,214,302,49),17)
box(18,'pull_width','Pull width → uncertainty scale\nTarget: width near 1.',(36,268,302,49),17)
box(18,'studies','Test zero-signal toys and 1, 3, 5 σ injections.',(36,326,302,41),17)
box(18,'scope','Conditional on the tested source; this is not a coverage result.',(36,370,302,29),13)
frame('h5d7e580f29331fdc_0_284',(365,91,332,286.8))
CHANGELOG.append({'slide':18,'changes':['Explained pull mean and width and their ideal targets; added exact pull equation.','Preserved source-fit/toy diagnostic screenshot, uniformly reduced and moved right to clear the title and text.','Retained zero-signal and 1/3/5-sigma study scope; labeled conditional validation.']})

# 22: injection reference is set before injection; right illustration is labeled.
replace('h5d7e580f29331fdc_0_10','Set the yield from a matched reference fit',(36,83,320,41),17,True)
image(22,'injection_equation','equation22_injection.png',(36,122,320,37))
box(22,'reference','Use the background-only fit error for this source, toy and mass.',(36,167,312,59),17)
box(22,'draw','Add full Gaussian templates to matched background toys.',(36,232,312,57),17)
box(22,'refit','Refit the sidebands, then repeat the signed signal extraction.',(36,296,312,55),17)
box(22,'qualification','The reference error sets the injected yield; it does not guarantee the measured significance.',(36,356,644,32),13)
image(22,'injection_plot','slide22_matched_injection.png',(371,92,325,255))
CHANGELOG.append({'slide':22,'changes':['Defined the matched background-only reference error and 0/1/3/5-sigma injection yields.','Explained matched-toy injection, sideband GP refitting and signed extraction.','Added a clearly labeled injection illustration and distinguished injection target from measured significance.']})

# 25: exactly the requested interpretive statement; all equation/citation media kept.
replace('h5d7e580f29331fdc_0_421','Profile the background in both hypotheses.\nAn excess gives q₀ > 0; a deficit is assigned q₀ = 0.',(36,251,642,84),21,False,7)
CHANGELOG.append({'slide':25,'changes':['Replaced bracket instruction with a concise interpretation of nuisance profiling and the one-sided discovery statistic.','Preserved existing formula, title and Cowan et al. citation.']})

# 26: reuse left placeholder for three steps; retain all four existing equations.
replace('h5d7e580f29331fdc_0_47','Test a signal yield A at a fixed mass.\nProfile the background for each A.\nThe 90% limit is where CLs reaches 0.10.',(36,86,310,127),17,False,8)
add('createParagraphBullets',{'objectId':'h5d7e580f29331fdc_0_47','textRange':rng(0,len('Test a signal yield A at a fixed mass.\nProfile the background for each A.\nThe 90% limit is where CLs reaches 0.10.')),'bulletPreset':'NUMBERED_DIGIT_ALPHA_ROMAN'})
image(26,'cls_plot','slide26_cls_crossing.png',(30,218,319,171))
CHANGELOG.append({'slide':26,'changes':['Added a three-step pointwise CLs explanation.','Added an illustrative CLs crossing at 0.10; retained the existing statistic/probability equations.']})

# 46: edit paragraphs independently to preserve original native dash list.
oid='g409df70e3d8_1_392'
oldp=text_of(oid).splitlines(keepends=True)
newp=['Conditional persistence test: one region at a time\n','Scale the fitted 10% continuum to full exposure (k = 10).\n','Choose a signal yield whose Asimov fit reaches √10 Z₁₀%.\n','Draw 20 Poisson spectra; rerun the moving-window GP and extraction.\n']
pos=0; edits=[]
for op,np in zip(oldp,newp):
    edits.append((pos,pos+len(op.rstrip('\n')),np.rstrip('\n')));pos+=len(op)
for a,b,txt in reversed(edits):
    add('deleteText',{'objectId':oid,'textRange':rng(a,b)})
    add('insertText',{'objectId':oid,'insertionIndex':a,'text':txt})
frame(oid,(36,82,646,180))
pos=0
for i,txt in enumerate(newp):
    style(oid,pos,pos+len(txt)-1,17,i==0)
    # Retain original native dash paragraph markers and indent.
    add('updateParagraphStyle',{'objectId':oid,'textRange':rng(pos,pos+len(txt)-1),'style':{'lineSpacing':100,'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':7,'unit':'PT'}},'fields':'lineSpacing,spaceAbove,spaceBelow'})
    pos+=len(txt)
box(46,'ratio_explanation','R compares injected and continuum limits. R < 1 means tighter neighboring limits; D measures the echo depth.',(36,253,646,59),17)
frame('g409df70e3d8_1_396',(75,318,596,81.4))
CHANGELOG.append({'slide':46,'changes':['Explained the conditional one-region persistence procedure, matched Asimov target, 20 Poisson toys and full refitting.','Explained the retained R and D equation screenshot in audience language.','Preserved custom native dash bullets; moved retained equation proportionally to fit.']})

# 49/50: preserve original titles; dedicated plots make the two GP jobs distinct.
replace('h5d7e580f29331fdc_0_157','Model the signed-root scan response',(36,82,278,38),17,True)
image(49,'field_equation','equation49_response.png',(36,122,271,83))
box(49,'response','D maps bin fluctuations into the signed-root mass scan.',(36,216,272,48),17)
box(49,'retain','Retain the response mean a, scale s and mass correlation R.',(36,266,272,52),17)
box(49,'sample','Draw correlated fields and record their largest excess.',(36,322,272,49),17)
image(49,'response_plot','slide49_response_correlations.png',(320,87,376,274))
box(49,'citation','Ananiev & Read, JINST 18 (2023) P05041; HPS v5.8.5.3.\n2021 native 10%; ±2.25σ; 50–250 MeV; frozen-source conditional calibration.',(36,376,643,24),9.5)
CHANGELOG.append({'slide':49,'changes':['Added correlated signed-root field equation and an explanation of mean, fluctuation scale and mass correlation.','Added archived 2021 response-correlation and saved null-scan displays.','Added Ananiev–Read citation and version/source qualification.']})

replace('h5d7e580f29331fdc_0_164','One mass versus the full search',(36,82,273,38),17,True)
image(50,'significance_equation','equation50_mapping.png',(36,122,271,112))
box(50,'local','Local: the excess at one mass, with the raw asymptotic mapping.',(36,240,272,58),17)
box(50,'global','Global: the largest excess anywhere in 50–250 MeV.',(36,298,272,49),17)
box(50,'conditional','Poisson check: 57/256 exceedances.\n95% interval: 0.173–0.279.',(36,348,272,35),13)
image(50,'local_global_plot','slide50_local_to_global.png',(320,87,376,274))
box(50,'citation','Ananiev & Read, JINST 18 (2023) P05041; HPS v5.8.5.3.\n2021 native 10%; ±2.25σ; 50–250 MeV; frozen-source conditional calibration.',(36,380,643,24),9.5)
CHANGELOG.append({'slide':50,'changes':['Added local/raw and global/maximum equations and a short explanation of the look-elsewhere comparison.','Added the archived raw local curve and source-conditional maximum distribution.','Explicitly separated asymptotic local mapping from conditional global calibration.']})

out={'scope':[18,22,25,26,46,49,50],'requests':REQ,'images':IMAGES,'changelog':CHANGELOG,'notes':['Existing titles, slide numbers, master and layouts unchanged.','All image frames are contain-fit boxes; createImage must use intrinsic aspect ratio.','No live mutation has been performed by this script.']}
(ROOT/'design/requests-specialist.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'requests':len(REQ),'images':len(IMAGES),'slides':out['scope']}))
