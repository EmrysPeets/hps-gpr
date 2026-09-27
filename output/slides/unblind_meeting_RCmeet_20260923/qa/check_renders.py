"""Verify preserved slide pixels, excluding the automatic page-number corner."""
from pathlib import Path
import json
from PIL import Image, ImageChops, ImageDraw
B=Path(__file__).resolve().parents[1]
r=json.loads((B/'qa/scope-verification.json').read_text())
rows=[]
for m in r['mapping']:
    if m['original'] in r['original_slides_changed']:continue
    old=Image.open(B/f"renders/before/slide-{m['original']:02d}.png").convert('RGB')
    new=Image.open(B/f"renders/final/slide-{m['final']:02d}.png").convert('RGB')
    d=ImageChops.difference(old,new)
    ImageDraw.Draw(d).rectangle((930,520,1000,563),fill=(0,0,0))
    rows.append({'original':m['original'],'final':m['final'],'identical_outside_page_number':d.getbbox() is None,'difference_bbox':d.getbbox()})
result={'rendered_pages':85,'unchanged_slide_comparisons':len(rows),'identical':sum(x['identical_outside_page_number'] for x in rows),'comparisons':rows}
(B/'qa/pixel-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))
print('Nonidentical:',[x for x in rows if not x['identical_outside_page_number']])
for start in range(1,86,12):
    canvas=Image.new('RGB',(1200,780),'#ededed');draw=ImageDraw.Draw(canvas)
    for j,n in enumerate(range(start,min(start+12,86))):
        im=Image.open(B/f'renders/final/slide-{n:02d}.png').convert('RGB');im.thumbnail((395,222))
        x=(j%3)*400;y=(j//3)*195
        im.thumbnail((395,170));canvas.paste(im,(x,y+21));draw.text((x+5,y+4),str(n),fill='black')
    canvas.save(B/f'qa/final-contact-{start:02d}.png')
