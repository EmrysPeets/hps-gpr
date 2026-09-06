#!/usr/bin/env python3
"""Validate source identity and PDF semantics; render all pages for human QA."""
from pathlib import Path
import csv,hashlib,json,re,subprocess
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];QA=HERE/'qa'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((HERE/'provenance/report_build.json').read_text());pdf=ROOT/build['pdf']
assert sha(pdf)==build['pdf_sha256']
for path,digest in build['input_sha256'].items():assert sha(ROOT/path)==digest,path
parent=HERE.parent/'v4p9p16_probability_echo_review_20260906'
entries=list(csv.DictReader((parent/'MANIFEST.csv').open()))
for row in entries:assert sha(ROOT/row['path'])==row['sha256'],row['path']
for p in (parent/'note').glob('*.tex'):
    if p.name=='analysis_note.tex':continue
    expected=p.read_text().replace('../figures/','../../'+parent.name+'/figures/')
    assert (HERE/'note'/p.name).read_text()==expected,p.name
reader=PdfReader(pdf);text='\n\f\n'.join(p.extract_text() for p in reader.pages)
(QA/'pdf_text.txt').write_text(text)
flat=re.sub(r'\s+',' ',text)
for phrase in ['Candidate removal and traditional signal searches','What was removed and how it was replaced','Traditional local fits on the original data','How to interpret the strongest-looking conventional outcomes','What this implies for presentation and the next data step','Full combined search','Can a real peak make signal echoes','Illustrative scan of deficits','A 30% checkpoint','Exploratory 2015 search','17,430','92.5/50','28.10']:
    assert phrase in flat,phrase
assert '@@' not in text and '??' not in text
log=(pdf.parent/'analysis_note.log').read_text()
assert not any(s in log for s in ['Overfull','undefined references','undefined citations','Missing character']),log
new_start=next(i+1 for i,p in enumerate(reader.pages) if '5 Candidate removal and traditional signal searches' in re.sub(r'\s+',' ',p.extract_text()))
new_end=next(i for i,p in enumerate(reader.pages) if 'Illustrative scan of deficits' in p.extract_text())
for old in QA.glob('page-*.png'):
    if int(old.stem.split('-')[-1])>len(reader.pages):old.unlink()
subprocess.run(['pdftoppm','-scale-to','1100','-png',str(pdf),str(QA/'page')],check=True,capture_output=True)
pages=sorted(QA.glob('page-*.png'));assert len(pages)==len(reader.pages)
for start in range(0,len(pages),8):
    canvas=Image.new('RGB',(4*330,2*460),'#ddd');draw=ImageDraw.Draw(canvas)
    for j,p in enumerate(pages[start:start+8]):
        im=Image.open(p).convert('RGB');im.thumbnail((320,430))
        x=(j%4)*330+(330-im.width)//2;y=(j//4)*460+25
        canvas.paste(im,(x,y));draw.text(((j%4)*330+8,(j//4)*460+7),p.stem,fill='black')
    canvas.save(QA/f'contact-{start//8+1:02d}.png')
out=dict(passed=True,pages=len(reader.pages),new_section=5,new_section_pages=[new_start,new_end],pdf=str(pdf.relative_to(ROOT)),pdf_sha256=sha(pdf),frozen_parent_entries=len(entries),copied_prior_sections_unchanged=True,rendered_pages_sha256={str(p.relative_to(ROOT)):sha(p) for p in pages},visual_review_required=True)
(QA/'product_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rendered_pages_sha256'},indent=2))
