"""Semantic and pixel preservation checks after merging the report.

Run with a Python containing pypdf and Pillow, and pdftoppm on PATH.
Manual visual inspection is a separate required release step.
"""
from pathlib import Path
import hashlib,json,subprocess,datetime,unicodedata
from pypdf import PdfReader
from PIL import Image,ImageChops
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=B/'inputs/parent_2021_report.pdf';appendix=B/'pdf/appendix.pdf'
report=B/'pdf/HPS_GPR_v6p3p2_2021_Study_with_2016_Offset_Appendix.pdf'
readers=[PdfReader(p) for p in (parent,appendix,report)]
assert [len(r.pages) for r in readers]==[9,7,16]
for i in range(9):
    assert readers[0].pages[i].get_contents().get_data()==readers[2].pages[i].get_contents().get_data()
    assert readers[0].pages[i].extract_text()==readers[2].pages[i].extract_text()
for i in range(7):assert readers[1].pages[i].extract_text()==readers[2].pages[i+9].extract_text()
text=' '.join(' '.join(p.extract_text().split()) for p in readers[1].pages)
normalized=''.join(unicodedata.normalize('NFKC',text).split())
for token in ('2016','2,000/2,000','36,000/36,000','16,000/16,000','8,000/8,000','Beta(10,91)','source qualification','0/100','100/100'):
    assert ''.join(token.split()) in normalized,token
folder=B/'qa/rendered';folder.mkdir(exist_ok=True)
for name,path in [('parent',parent),('appendix',appendix),('merged',report)]:
    subprocess.run(['pdftoppm','-scale-to','1500','-png',str(path),str(folder/name)],check=True,capture_output=True)
matches=[]
for i in range(1,17):
    source=folder/(f'parent-{i}.png' if i<=9 else f'appendix-{i-9}.png')
    merged=folder/f'merged-{i:02d}.png'
    a=Image.open(source).convert('RGB');b=Image.open(merged).convert('RGB')
    assert a.size==b.size and ImageChops.difference(a,b).getbbox() is None,f'Pixel mismatch page{i}'
    matches.append(dict(page=i,source=str(source.relative_to(B)),source_sha256=sha(source),merged_sha256=sha(merged),pixel_identical=True))
result=dict(passed=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pages=16,parent_pages_preserved=9,appendix_pages=7,
    parent_pdf_sha256=sha(parent),appendix_pdf_sha256=sha(appendix),merged_pdf_sha256=sha(report),
    parent_page_content_streams_unchanged=True,all_page_text_matches_source=True,all_pages_pixel_identical_to_sources=True,render_max_dimension=1500,
    render_matches=matches,visual_review_required='Inspect all seven appendix pages; unchanged parent pages retain previous release QA.')
(B/'qa/report_automated_qa.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='render_matches'}))
