"""Semantic and geometric PDF checks plus full-page renders for visual review."""
from pathlib import Path
import json,hashlib,shutil,subprocess
import fitz
B=Path(__file__).resolve().parents[1]

def main():
    pdf=B/'pdf/report.pdf';doc=fitz.open(pdf);out=B/'qa/rendered';out.mkdir(exist_ok=True)
    texts=[p.get_text() for p in doc];alltext='\n'.join(texts)
    assert len(doc)==5,f'Unexpected page count: {len(doc)}'
    assert 'pole' not in alltext.lower() and '\ufffd' not in alltext and '@@' not in alltext
    result=json.loads((B/'results/global_results.json').read_text())
    for row in result['primary']:
        for expected in (str(row['peak_mass_MeV']),f"{row['global_exceedances']}/1024",f"{row['global_p_rank']:.3f}",f"{row['global_Z_excess']:.2f}"):
            assert expected in texts[0],expected
    for term in ('1,024','conditional','76 MeV','Clopper','443,392','175 MeV'):
        assert term in alltext,term
    checks=[]
    for i,p in enumerate(doc):
        for block in p.get_text('blocks'):
            x0,y0,x1,y1,*_=block
            assert x0>=18 and y0>=12 and x1<=p.rect.width-18 and y1<=p.rect.height-12,(i,block)
        checks.append(dict(page=i+1,characters=len(texts[i]),all_text_inside_page=True))
    log=(B/'pdf/report.log').read_text()
    assert 'Overfull' not in log and 'Undefined control sequence' not in log
    poppler=shutil.which('pdftoppm')
    bundled=Path('/Users/emryspeets/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm')
    if poppler is None and bundled.exists():poppler=str(bundled)
    if poppler:
        subprocess.run([poppler,'-r','110','-png',str(pdf),str(out/'page')],check=True)
    else:
        for i,p in enumerate(doc):p.get_pixmap(matrix=fitz.Matrix(110/72,110/72)).save(str(out/f'page-{i+1}.png'))
    (B/'qa/report_text.txt').write_text(alltext)
    qa=dict(passed=True,pages=len(doc),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        semantic_peak_values_verified=True,forbidden_term_absent=True,no_overfull_boxes=True,pages_checked=checks,
        rendered_pages=[str(p.relative_to(B)) for p in sorted(out.glob('page-*.png'))],
        visual_review='Pending manual inspection of the rendered pages')
    (B/'qa/pdf_validation.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
