"""Render every report page and audit text/geometry; manual review is separate."""
from pathlib import Path
import json,hashlib
import fitz
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1]
pdf=B/'pdf/HPS_GPR_v5p9_Signal_Tail_Study.pdf'
doc=fitz.open(pdf);q=B/'qa/rendered';q.mkdir(exist_ok=True)
for old in list(q.glob('page_*.png'))+list(q.glob('contact_*.png')):old.unlink()
texts=[];outside=[]
for i,p in enumerate(doc):
    p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(q/f'page_{i+1:02d}.png')
    texts.append(p.get_text())
    for block in p.get_text('blocks'):
        x0,y0,x1,y1=block[:4]
        if x0<25 or y0<20 or x1>p.rect.width-25 or y1>p.rect.height-20:outside.append(dict(page=i+1,bounds=[x0,y0,x1,y1],text=block[4][:120]))
for first in range(0,len(doc),4):
    sheet=Image.new('RGB',(1224,1644),'#bbbbbb')
    for j in range(first,min(first+4,len(doc))):
        im=Image.open(q/f'page_{j+1:02d}.png');im.thumbnail((602,802));x=(j-first)%2*612;y=(j-first)//2*822
        sheet.paste(im,(x,y+20));ImageDraw.Draw(sheet).text((x+8,y+3),f'Page {j+1}',fill='black')
    sheet.save(q/f'contact_{first//4+1}.png')
text='\n\f\n'.join(texts);(B/'qa/pdf_text.txt').write_text(text)
log=(B/'source/report.log').read_text()
result=dict(pages=len(doc),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
    outside_page_blocks=outside,minimum_page_text_characters=min(map(len,texts)),
    unresolved_references=('??' in text or 'undefined' in log.lower()),overfull_boxes=('Overfull' in log),
    all_main_numbers_present=all(t in text for t in ('5,915','0.529','51','90','78','175','176')),
    manual_visual_review='Required separately; not inferred from this script.')
result['passed']=not outside and not result['unresolved_references'] and not result['overfull_boxes'] and result['all_main_numbers_present']
(B/'qa/artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['passed']
