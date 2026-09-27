import json
from pathlib import Path
from PIL import Image

W = Path(__file__).resolve().parents[1]
BEFORE=json.loads((W/'presentation-before.json').read_text())
REQUESTS=[]
IMAGES=[]

def slide(n): return BEFORE['slides'][n-1]['objectId']

def delete(oid): REQUESTS.append({'deleteObject':{'objectId':oid}})

def geom(oid,x,y,w,h):
    # Existing shapes use a 3,000,000 EMU square, 236.22047 pt.
    e=next(e for s in BEFORE['slides'] for e in s.get('pageElements',[]) if e['objectId']==oid)
    sz=e['size']; factor=lambda d:d['magnitude']/(12700 if d['unit']=='EMU' else 1)
    REQUESTS.append({'updatePageElementTransform':{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/factor(sz['width']),'scaleY':h/factor(sz['height']),'translateX':x,'translateY':y,'unit':'PT'}}})

def style(oid,size=18,bold=False,color=None,bullets=False):
    st={'fontFamily':'EB Garamond','fontSize':{'magnitude':size,'unit':'PT'},'bold':bold,'foregroundColor':{'opaqueColor':{'rgbColor':color or {'red':0,'green':0,'blue':0}}}}
    REQUESTS.append({'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}})
    REQUESTS.append({'updateParagraphStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':4,'unit':'PT'},'lineSpacing':100,'indentStart':{'magnitude':0,'unit':'PT'},'indentEnd':{'magnitude':0,'unit':'PT'},'indentFirstLine':{'magnitude':0,'unit':'PT'}},'fields':'spaceAbove,spaceBelow,lineSpacing,indentStart,indentEnd,indentFirstLine'}})
    if bullets: REQUESTS.append({'createParagraphBullets':{'objectId':oid,'textRange':{'type':'ALL'},'bulletPreset':'BULLET_DISC_CIRCLE_SQUARE'}})
    else: REQUESTS.append({'deleteParagraphBullets':{'objectId':oid,'textRange':{'type':'ALL'}}})

def replace(oid,text,x=None,y=None,w=None,h=None,size=18,bold=False,bullets=False):
    # Only homogeneous instruction placeholders use this full-range replacement.
    REQUESTS.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':text}}])
    if x is not None: geom(oid,x,y,w,h)
    style(oid,size,bold,bullets=bullets)

def text(n,key,content,x,y,w,h,size=18,bold=False,bullets=False,color=None):
    oid=f'rc22_s{n}_{key}'
    REQUESTS.extend([{'createShape':{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':slide(n),'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':content}}])
    style(oid,size,bold,color,bullets)
    return oid

def image(n,key,path,x,y,w,h):
    path=str(Path(path).resolve()); iw,ih=Image.open(path).size
    scale=min(w/iw,h/ih); dw,dh=iw*scale,ih*scale
    IMAGES.append({'slide':n,'objectId':f'rc22_s{n}_{key}','local_path':path,'x':x+(w-dw)/2,'y':y+(h-dh)/2,'width':dw,'height':dh})

def table(n,key,rows,x,y,w,h,widths=None,font_size=13):
    oid=f'rc22_s{n}_{key}'; nr,nc=len(rows),len(rows[0])
    REQUESTS.append({'createTable':{'objectId':oid,'rows':nr,'columns':nc,'elementProperties':{'pageObjectId':slide(n),'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}})
    if widths:
        for c,cw in enumerate(widths): REQUESTS.append({'updateTableColumnProperties':{'objectId':oid,'columnIndices':[c],'tableColumnProperties':{'columnWidth':{'magnitude':cw,'unit':'PT'}},'fields':'columnWidth'}})
    for r,row in enumerate(rows):
        for c,content in enumerate(row):
            cell={'rowIndex':r,'columnIndex':c}
            REQUESTS.extend([{'insertText':{'objectId':oid,'cellLocation':cell,'text':content,'insertionIndex':0}},{'updateTextStyle':{'objectId':oid,'cellLocation':cell,'textRange':{'type':'ALL'},'style':{'fontFamily':'EB Garamond','fontSize':{'magnitude':font_size,'unit':'PT'},'bold':r==0},'fields':'fontFamily,fontSize,bold'}},{'updateParagraphStyle':{'objectId':oid,'cellLocation':cell,'textRange':{'type':'ALL'},'style':{'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':0,'unit':'PT'}},'fields':'spaceAbove,spaceBelow'}}])
    REQUESTS.append({'updateTableCellProperties':{'objectId':oid,'tableRange':{'location':{'rowIndex':0,'columnIndex':0},'rowSpan':nr,'columnSpan':nc},'tableCellProperties':{'contentAlignment':'MIDDLE'},'fields':'contentAlignment'}})
    return oid

def save():
    (W/'build/requests-root.json').write_text(json.dumps({'requests':REQUESTS,'images':IMAGES},indent=2))
