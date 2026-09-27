#!/usr/bin/env python3
"""Bounded publication checks; reads saved artifacts and runs no scientific fits."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    catalog=json.loads((HERE/'catalog.json').read_text())
    cards=json.loads((HERE/'plot_cards.json').read_text())
    snapshot=json.loads((ROOT/'publication/recent_studies_20260927/snapshot.json').read_text())
    checks=[]
    def check(name,passed,detail):
        if not passed:raise AssertionError((name,detail))
        checks.append(dict(name=name,passed=True,detail=detail))
    expected={x for x in snapshot['roots'] if x.startswith('study_results/') or x.startswith('output/slides/') or x=='apex_initial_studies'}
    check('Catalogue scope',expected=={x['path'] for x in catalog},len(expected))
    check('Unique plot IDs',len(cards)==26 and len({x['id'] for x in cards})==26,len(cards))
    for card in cards:
        check(card['id']+' image provenance',digest(ROOT/card['source_png'])==digest(HERE/card['asset_png'])==card['sha256'],card['source_png'])
        check(card['id']+' evidence exists',(ROOT/card['source_evidence']).is_file(),card['source_evidence'])
        if card.get('asset_pdf'):
            check(card['id']+' vector provenance',digest(ROOT/card['source_pdf'])==digest(HERE/card['asset_pdf']),card['source_pdf'])
    for file in [HERE/'README.md',HERE/'PLOTBOOK.md',HERE/'STUDY_LOG.md',HERE/'PRESENTATION_NOTES.md',ROOT/'docs/STUDY_INDEX.md',HERE/'index.html']:
        text=file.read_text()
        links=re.findall(r'\]\(([^)]+)\)',text) if file.suffix=='.md' else re.findall(r'(?:src|href)="([^"]+)"',text)
        for link in links:
            if link.startswith(('http:','https:','mailto:','#')):continue
            path=unquote(urlsplit(link).path)
            check('Link '+file.name+' -> '+path,(file.parent/path).exists(),path)
    pdf=ROOT/'output/pdf/study_logbook_20260927/HPS_GPR_Recent_Studies_Plot_Logbook.pdf'
    reader=PdfReader(pdf)
    check('32-page PDF',len(reader.pages)==32,len(reader.pages))
    text='\n'.join(p.extract_text() for p in reader.pages)
    (HERE/'qa/pdf_text.txt').write_text(text)
    for needle in ['0.07415','0.05804','0.09095','1/1025','75 / 1,024','fixed local maps','not a physical zero limit']:
        check('PDF statement '+needle,needle in text,needle)
    for card in cards:
        page=reader.pages[2+int(card['id'][1:])]
        content=page.extract_text()
        check(card['id']+' caption and boundary','CLAIM BOUNDARY' in content and card['id'] in content,len(content))
    claims=json.loads((HERE/'qa/source_claim_checks.json').read_text())
    check('Numerical headline checks',claims['passed'] and len(claims['checks'])==17,len(claims['checks']))
    result=dict(passed=True,pdf_sha256=digest(pdf),pages=32,studies=len(catalog),plots=len(cards),checks=len(checks),details=checks,
                scientific_reruns=False,scope='Publication/source validation; archived scientific results retain their original scope.')
    (HERE/'qa/publication_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:',len(checks),'publication checks; 17 numerical headline checks; 32 pages.')


if __name__=='__main__':main()
