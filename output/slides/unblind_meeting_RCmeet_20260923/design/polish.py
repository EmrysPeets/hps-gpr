"""One consolidated render-driven repair: typeset the residual and native subscripts."""
from pathlib import Path
import json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
P=json.loads((B/'raw-output.json').read_text())['structuredContent']
objects={e['objectId']:e for s in P['slides'] for e in s.get('pageElements',[])}
R=[]
def add(k,v): R.append({k:v})
def text(oid): return ''.join(t.get('textRun',{}).get('content','') for t in objects[oid]['shape']['text']['textElements']).rstrip('\n')
def move(oid,x,y):
    t=dict(objects[oid]['transform']);u=12700 if t.get('unit')=='EMU' else 1
    t.update(translateX=x*u,translateY=y*u)
    add('updatePageElementTransform',{'objectId':oid,'applyMode':'ABSOLUTE','transform':t})
def baseline(oid,a,b,kind='SUBSCRIPT'):
    add('updateTextStyle',{'objectId':oid,'textRange':{'type':'FIXED_RANGE','startIndex':a,'endIndex':b},'style':{'baselineOffset':kind},'fields':'baselineOffset'})
def replace(oid,new):
    add('deleteText',{'objectId':oid,'textRange':{'type':'ALL'}})
    add('insertText',{'objectId':oid,'insertionIndex':0,'text':new})
    add('updateTextStyle',{'objectId':oid,'textRange':{'type':'ALL'},'style':{'baselineOffset':'NONE'},'fields':'baselineOffset'})

fig=plt.figure(figsize=(5.9,.65),facecolor='white')
fig.text(.01,.5,r'$r_i=(n_i-\widehat b_i)/\sqrt{\widehat b_i+C_{\mathrm{GP},ii}}$',fontsize=24,va='center',color='black')
for ext in ('png','pdf','svg'):fig.savefig(B/f'assets/residual_definition.{ext}',dpi=320,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
oid='rc22_s13_caption'
replace(oid,'Green: ±2 reference band.\nNeighboring residuals are correlated.')
e=objects[oid];u=12700 if e['size']['width']['unit']=='EMU' else 1
add('updatePageElementTransform',{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':350/(e['size']['width']['magnitude']/u),'scaleY':33/(e['size']['height']['magnitude']/u),'translateX':335,'translateY':360,'unit':'PT'}})
add('updateTextStyle',{'objectId':oid,'textRange':{'type':'ALL'},'style':{'fontSize':{'magnitude':12.5,'unit':'PT'}},'fields':'fontSize'})
move('rc22_s11_legend',444,342)
move('rc22_s14_qualification',36,365)
for oid in ('rcmeet_s22_reference','rcmeet_s22_draw'):
    for m in re.finditer('σA,ref',text(oid)):baseline(oid,m.start()+1,m.end())
oid='rc22_s11_legend'
for m in re.finditer('b(GP|prof)',text(oid)):baseline(oid,m.start()+1,m.end())
oid='h5d7e580f29331fdc_0_47';t=text(oid)
for m in re.finditer('CL(s\\+b|s|b)',t):baseline(oid,m.start()+2,m.end())
for m in re.finditer('q̃A',t):baseline(oid,m.end()-1,m.end())
for m in re.finditer('A90',t):baseline(oid,m.start()+1,m.end())
oid='rc23_s27_symbols';t=text(oid).replace('f rad','frad');replace(oid,t)
for pat in ('frad','A90'):
    m=re.search(pat,t);baseline(oid,m.start()+1,m.end())
baseline('rc23_s27_br_key',1,4)
baseline('rc23_s27_br_key',4,6,'SUPERSCRIPT')
oid='rc23_s27_plot_key';t=text(oid);baseline(oid,t.index('μ'),t.index('μ')+1)

from PIL import Image
asset=(B/'assets/residual_definition.png').resolve();w,h=Image.open(asset).size;scale=min(270/w,29/h);ww,hh=w*scale,h*scale
im={'image_uris':str(asset),'requests':[{'createImage':{'objectId':'rc23_s13_residual_eq','url':str(asset),'elementProperties':{'pageObjectId':P['slides'][12]['objectId'],'size':{'width':{'magnitude':ww,'unit':'PT'},'height':{'magnitude':hh,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':36+(270-ww)/2,'translateY':361+(29-hh)/2,'unit':'PT'}}}}]}
(B/'design/polish_requests.json').write_text(json.dumps({'requests':R,'image':im,'revision':P['revisionId']},indent=2)+'\n')
print(len(R),'native repairs and one equation image')
