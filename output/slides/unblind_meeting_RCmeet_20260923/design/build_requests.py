"""Build scoped native Slides requests. Does not write to Google Slides."""
from pathlib import Path
import json
from PIL import Image
B=Path(__file__).resolve().parents[1]
P=json.loads((B/'presentation-before.json').read_text())
CONTENT=json.loads((B/'content/content.json').read_text())
R=[];IM=[];NOTES=[];FONT='EB Garamond'
OBJ={e['objectId']:e for s in P['slides'] for e in s.get('pageElements',[])}
def add(k,v):R.append({k:v})
def rg(a,b):return {'type':'FIXED_RANGE','startIndex':a,'endIndex':b}
def txt(oid):return ''.join(v.get('textRun',{}).get('content','') for v in OBJ[oid].get('shape',{}).get('text',{}).get('textElements',[]))
def frame(oid,box):
    x,y,w,h=box;e=OBJ[oid];u=12700 if e['size']['width']['unit']=='EMU' else 1
    add('updatePageElementTransform',{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/(e['size']['width']['magnitude']/u),'scaleY':h/(e['size']['height']['magnitude']/u),'translateX':x,'translateY':y,'unit':'PT'}})
def style(oid,a,b,size=17,bold=False,color=None):
    add('updateTextStyle',{'objectId':oid,'textRange':rg(a,b),'style':{'fontFamily':FONT,'fontSize':{'magnitude':size,'unit':'PT'},'bold':bold,'italic':False,'foregroundColor':{'opaqueColor':{'rgbColor':color or {}}}},'fields':'fontFamily,fontSize,bold,italic,foregroundColor'})
def paragraphs(oid,n,spacing=5):
    add('updateParagraphStyle',{'objectId':oid,'textRange':rg(0,n),'style':{'lineSpacing':100,'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':spacing,'unit':'PT'},'indentStart':{'magnitude':0,'unit':'PT'},'indentEnd':{'magnitude':0,'unit':'PT'},'indentFirstLine':{'magnitude':0,'unit':'PT'}},'fields':'lineSpacing,spaceAbove,spaceBelow,indentStart,indentEnd,indentFirstLine'})
def settext(oid,text,box=None,size=None,bold=False,spacing=5):
    old=txt(oid).rstrip('\n') if oid in OBJ else ''
    if old:add('deleteText',{'objectId':oid,'textRange':rg(0,len(old))})
    add('insertText',{'objectId':oid,'insertionIndex':0,'text':text})
    if box:frame(oid,box)
    if size:
        add('deleteParagraphBullets',{'objectId':oid,'textRange':rg(0,len(text))})
        style(oid,0,len(text),size,bold);paragraphs(oid,len(text),spacing)
        add('updateShapeProperties',{'objectId':oid,'shapeProperties':{'contentAlignment':'TOP','autofit':{'autofitType':'NONE'}},'fields':'contentAlignment,autofit.autofitType'})
def box(page,oid,text,geom,size=17,bold=False,color=None):
    x,y,w,h=geom
    add('createShape',{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}})
    OBJ[oid]={'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}}}
    settext(oid,text,size=size,bold=bold)
    if color:style(oid,0,len(text),size,bold,color)
    return oid
def image_existing(oid,path,boxgeom=None):
    path=Path(path).resolve();requests=[{'replaceImage':{'imageObjectId':oid,'url':str(path),'imageReplaceMethod':'CENTER_INSIDE'}}]
    if boxgeom:
        x,y,w,h=boxgeom;iw,ih=Image.open(path).size;f=min(w/iw,h/ih);w2,h2=iw*f,ih*f;x+=(w-w2)/2;y+=(h-h2)/2
        e=OBJ[oid];u=12700 if e['size']['width']['unit']=='EMU' else 1
        requests.insert(0,{'updatePageElementTransform':{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w2/(e['size']['width']['magnitude']/u),'scaleY':h2/(e['size']['height']['magnitude']/u),'translateX':x,'translateY':y,'unit':'PT'}}})
    IM.append({'image_uris':str(path),'requests':requests})
def newimage(page,oid,path,geom):
    path=Path(path).resolve();x,y,w,h=geom;iw,ih=Image.open(path).size;f=min(w/iw,h/ih);ww,hh=iw*f,ih*f;x+=(w-ww)/2;y+=(h-hh)/2
    IM.append({'image_uris':str(path),'requests':[{'createImage':{'objectId':oid,'url':str(path),'elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':ww,'unit':'PT'},'height':{'magnitude':hh,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}}]})
def notes(n,text):NOTES.append({'original_slide':n,'pageId':P['slides'][n-1]['objectId'],'text':text})
def rgb(h):return dict(zip(['red','green','blue'],[int(h[i:i+2],16)/255 for i in [0,2,4]]))

# 9: keep the equation and analytic illustrations; spoken options go in notes.
settext('h5d7e580f29331fdc_0_110','ℓ: smoothness along log mass\nC: variance of the log background\nα ≈ 1/y: noise variance of each bin',(335,78,343,74),17)
notes(9,'Three complete spoken options for this slide:\n\n'+'\n\n'.join(x['label']+'\n'+x['script'] for x in CONTENT['slide09']['whole_slide_script_options'])+'\n\nParameter reminder: α is the bin-noise rule, not a third optimized kernel parameter. C and ℓ are fitted to the sidebands. α_EM on the coupling slide is a different quantity. The zero-count implementation uses α=1.')

# 11: preserve the 78 MeV example and every original profiling equation.
image_existing('rc22_s11_profile78',B/'science/assets/slide11_profile78_clear.png',(453,83,250,264))
legend='Data: n − bGP     GP baseline: 0\nBackground: bprof − bGP\nTotal fit: bprof + Aw − bGP     Signal: Aw\nShading: GP constraint width'
settext('rc22_s11_legend',legend,(444,350,273,49),10.3,spacing=0)
for word,c in [('GP baseline: 0','c99522'),('Background: bprof − bGP','1b77a4'),('Total fit: bprof + Aw − bGP','bb3636'),('Signal: Aw','8054a4')]:
    a=legend.index(word);style('rc22_s11_legend',a,a+len(word),10.3,color=rgb(c))
notes(11,'Lower panel: every quantity is measured relative to the same unprofiled GP mean b_GP. Black is n−b_GP, blue is b_prof−b_GP, red is b_prof+Aw−b_GP, purple is the fitted signal Aw, and gold is zero. Units are 10^3 events/MeV. Shading is the GP constraint width, not a post-fit confidence band. The fixed-state v5.0.5 replay at78MeV reproduces the signed root2.808644975. Sources and full curve data: science/data/slide11_profile78_curves.csv and science/provenance/protocol.json.')

# 13: three campaigns across their search domains, with held-out local predictions.
settext('h5d7e580f29331fdc_0_247','GP predictions across the search regions')
settext('h5d7e580f29331fdc_0_248','Each bin is predicted with its local ±2.25σ neighborhood excluded.',(36,78,650,26),16)
image_existing('rc22_s13_examples',B/'science/assets/slide13_all_datasets_heldout.png',(21,105,678,252))
settext('rc22_s13_caption','Residual = (n − b)/√(b + CGP,ii). Green: ±2 reference band. Neighboring residuals are correlated.',(36,364,644,31),12.5,spacing=0)
notes(13,'Updated display: full2015 search19–100MeV, full2016 search39–180MeV, and2021 native10% search50–250MeV. Each histogram bin is predicted using the nearest integer-mass window, excluding±2.25sigma about that hypothesis. Each shown bin is withheld from its own prediction. The line combines moving local predictions rather than a single global GP fit. Kernel states remain archived;2015 uses the90MeV endpoint kernel above90MeV. Residual=(n−b)/sqrt(b+C_GP,ii). The green±2 interval is a reference band, not a simultaneous confidence or coverage band. Parent input hashes, raw residuals and source choices are saved in science/.')

# 14: score fitted sidebands outside the excluded region.
settext('h5d7e580f29331fdc_0_254','Sideband fit quality')
settext('h5d7e580f29331fdc_0_255','2021 native 10%: score 50–250 MeV outside each ±2.25σ window',(36,78,650,26),16)
image_existing('rc22_s14_q_equation',B/'science/assets/slide14_deviance_equation.png',(48,111,625,38))
image_existing('rc22_s14_residualscan',B/'science/assets/slide14_sideband_deviance.png',(27,155,670,214))
settext('rc22_s14_qualification','Fitted sidebands only. The 78 MeV example is postselected. Toys use a frozen source and fixed kernels.\nPrediction in the excluded region and confidence-limit coverage need separate checks.',(36,371,643,29),11.5,spacing=0)
notes(14,'The previous held-out Q/Nbin curve is retained in the v5.9.5 study, not discarded scientifically. This new diagnostic scores only search-region sideband bins outside the selected±2.25sigma window. The GP trains on all support bins outside the same mask. D_side=2 sum[n log(n/bhat)−n+bhat] and N_side is a bin count, not effective degrees of freedom. D/N_side=.8891,.8522,.9149 for65,78,120MeV. At78MeV237/256 frozen-source refits have at least as large D;95% binomial interval[.8865,.9547]. This is an in-sample conditional check at a postselected anchor, not physical-background adequacy or signal significance. We also computed a binned cumulative shape distance excluding the blind window:186/256 refits exceed the observed value. A naive distribution-free KS probability is inappropriate because the GP was fitted to those bins and the counts are binned. Shape-only cumulative checks also normalize away the count scale; Poisson deviance retains it. A full diagnostic must preserve the source-building and fit procedure in calibration. v5.9.5 separately found a real signed-response offset at78MeV and a20/256 exploratory held-out residual-scan tail.')

# 21: only the requested embedded footer is removed; plotted results are identical.
image_existing('rc22_s21_threshold',B/'assets/slide21_threshold_no_footer.png')

# 22: define reference and matching in ordinary language.
settext('h5d7e580f29331fdc_0_10','Choose the yield before adding signal',(36,82,333,32),17,True)
settext('rcmeet_s22_reference','σA,ref is the yield error from this same background toy and mass, before injection.',(36,168,320,68),17)
settext('rcmeet_s22_draw','Add z × σA,ref events with the full Gaussian template. Reuse the same background toy at each strength.',(36,239,320,69),17)
settext('rcmeet_s22_refit','Refit the sidebands. Allow the fitted signal yield to be positive or negative.',(36,309,320,49),17)
settext('rcmeet_s22_qualification','z sets the injected yield. The measured significance comes from the new fit.',(36,370,648,25),13)
notes(22,'Reference means the pre-injection background-only fit. Matched means the same source, background realization, mass, binning, exclusion window and GP prescription. The reference error defines A_inj=z sigma_A,ref before adding the full Gaussian template, withz=0,1,3,5. We repeat the GP sideband prediction and allow the fitted signal amplitude to float positive or negative for recovery and pull diagnostics. Negative yield represents a fitted deficit, not physical negative production. The pull denominator uses the post-injection error and is distinct from the reference error used to choose injection strength. No new signal-count Poisson draw is implied beyond the archived injection procedure.')

# 26: preserve all four equations and give each an explicit interpretation.
add('deleteObject',{'objectId':'rcmeet_s26_cls_plot'})
walk='Test a yield A at one mass. Refit the background nuisance parameters at each A.\nThe likelihood ratio measures disagreement with A. Set q̃A = 0 when the data prefer a larger yield.\nCLs+b and CLb are tails of the same statistic under signal + background and background alone.\nForm their ratio, CLs. Increase A until CLs(A90) = 0.10 for the 90% upper limit.'
settext('h5d7e580f29331fdc_0_47',walk,(36,85,320,256),17,spacing=13)
add('createParagraphBullets',{'objectId':'h5d7e580f29331fdc_0_47','textRange':rg(0,len(walk)),'bulletPreset':'NUMBERED_DIGIT_ALPHA_ROMAN'})
box(P['slides'][25]['objectId'],'rc23_s26_boundary','If the unrestricted best yield is negative, the denominator uses the physical A = 0 fit.',(42,346,312,48),13)
notes(26,'Equation walkthrough: profiling means optimize correlated background nuisance coordinates at each trialA; it does not mean reoptimize GP kernels for everyA. The bounded statistic uses zero if the unrestricted best yield exceedsA, the unrestricted best fit when0≤Ahat≤A, and the physicalA=0 denominator ifAhat<0. CL_s+b=P(qtilde_A≥qobs|A+background); CL_b=P(qtilde_A≥qobs|background). Their ratio mitigates excessively strong exclusions when sensitivity is poor. The pointwise90%limit is the firstCLs=.10 crossing. Discovery mass-scan trials are a separate calculation. The illustrative zero-best-fit Wald chart was removed to provide readable space for this walkthrough; the four defining equations are retained.')

# 27: keep the exact source plot, fitting its aspect ratio inside the right region.
settext('g409df70e3d8_1_110','Signal yield and coupling limits')
e=OBJ['g409df70e3d8_1_115'];w=e['size']['width']['magnitude'];h=e['size']['height']['magnitude'];scale=min(437/w,285/h);ww,hh=w*scale,h*scale
frame('g409df70e3d8_1_115',(273+(437-ww)/2,84+(285-hh)/2,ww,hh))
page=P['slides'][26]['objectId']
box(page,'rc23_s27_yield_label','Total fitted signal yield',(33,91,231,25),17,True)
newimage(page,'rc23_s27_yield_eq',B/'assets/yield_relation.png',(34,120,231,40))
box(page,'rc23_s27_limit_label','90% upper limit',(33,171,231,25),17,True)
newimage(page,'rc23_s27_eps_eq',B/'assets/epsilon_compact.png',(31,199,235,60))
box(page,'rc23_s27_symbols','ρ: selected prompt counts per mass\nf rad: radiative-trident fraction\nA90: total signal-yield upper limit',(36,270,234,66),13)
newimage(page,'rc23_s27_br_eq',B/'assets/epsilon_physical.png',(32,338,237,32))
box(page,'rc23_s27_br_key','NeffBR = 1 / BR(A′ → e⁺e⁻)',(35,374,235,22),12)
box(page,'rc23_s27_plot_key','Plot: minimal-visible ε², including the branching correction above 2mμ.',(287,366,414,34),12)
notes(27,'Analysis Notev5.0.5 section4 signal-conversion equations: A_d=K_d epsilon_ee², K_d=3π m f_rad,d rho_d/(2 alpha_EM), so epsilon_90,ee²=A90,d/K_d. rho is the selected prompt counts-per-mass density and f_rad the radiative-trident fraction; use consistent mass units. A is the total full-template yield, so no extra blind-window fraction is applied. Physical minimal-visible mixing is epsilon_phys²=N_eff^BR epsilon_ee², N_eff^BR=1/BR(Aprime→ee), equal to1 below2m_mu in the minimal model. Existing plot visually matches the v5.0.5 observed overlay (SHA-identical to v5.0.4). Its generator explicitly multiplies by1+Gamma_mumu/Gamma_ee above2m_mu. This revision preserves the existing plot bytes and changes only its scale/position.')

# New native text slides from the inspected p7 title/body layout.
for key in ['validation_overview','validation_summary']:
    pid='rc23_'+key;title=pid+'_title';body=pid+'_body';num=pid+'_number'
    add('createSlide',{'objectId':pid,'slideLayoutReference':{'layoutId':'p7'},'placeholderIdMappings':[{'layoutPlaceholderObjectId':'p7_i4','objectId':title},{'layoutPlaceholderObjectId':'p7_i16','objectId':body},{'layoutPlaceholderObjectId':'p7_i8','objectId':num}]})
    for oid,geom in [(title,(36,7.6,638,44.5)),(body,(36,73.4,638,299.1))]:
        OBJ[oid]={'size':{'width':{'magnitude':3000000,'unit':'EMU'},'height':{'magnitude':3000000,'unit':'EMU'}}}
    src=CONTENT['new_before_source15' if key=='validation_overview' else 'new_after_source23']
    cfg={'title':src['title'],'items':[{'heading':x['heading'],'text':x['body']} for x in src['rows']],'scope':src['footer'],'notes':'\n\n'.join(src['notes'])+'\nSource: '+src['source']}
    if key=='validation_overview':
        cfg['title']='Background validation strategy'
        cfg['items'][0]['heading']='Generate toys from analytic source fits'
    settext(title,cfg['title'])
    parts=[]
    for item in cfg['items']:parts.append(item['heading']+'\n'+item['text'])
    text='\n'.join(parts)
    settext(body,text,(36,86,646,256),17,spacing=5)
    pos=0
    for item in cfg['items']:
        style(body,pos,pos+len(item['heading']),18,True,rgb('990000'));pos+=len(item['heading'])+1+len(item['text'])+1
    box(pid,pid+'_scope',cfg['scope'],(36,355,646,41),13)
    NOTES.append({'pageId':pid,'text':cfg['notes']})

for n in NOTES:
    if 'original_slide' in n:
        s=P['slides'][n['original_slide']-1];n['notesId']=s['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
(B/'design/native_requests.json').write_text(json.dumps({'presentation_id':P['presentationId'],'revision':P['revisionId'],'requests':R,'images':IM,'notes':NOTES,'original_scope':[9,11,13,14,21,22,26,27]},indent=2)+'\n')
print(json.dumps({'native_requests':len(R),'image_writes':len(IM),'notes':len(NOTES)}))
