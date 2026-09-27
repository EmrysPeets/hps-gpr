#!/usr/bin/env python3
"""Semantic/layout checks and 110-dpi renders of all v6.4.4 report pages.

Automated checks do not replace visual inspection. The JSON records that step
as pending even when its automated checks pass. No numerical files are changed.
"""
from pathlib import Path
import hashlib,json,re,sys
import fitz
import pandas as pd

B=Path(__file__).resolve().parents[1]
EXPECTED_PAGES=9
DPI=110
APPENDICES={
    'A':(6,'One shared coupling, then a search over mass'),
    'B':(7,'Compare the two global calibrations'),
    'C':(8,'Two Sidak references with different assumptions'),
    'D':(9,'Finite calibration, retained results and references'),
}
REFERENCES=(
    'https://www.pp.rhul.ac.uk/~cowan/stat/cowan_lee_25jun21.pdf',
    'https://arxiv.org/abs/1005.1891',
    'https://arxiv.org/abs/1602.03765',
)

def compact(text):
    # Strip whitespace only; retain digits, signs and punctuation for table QA.
    return re.sub(r'\s+','',text)

def main():
    pdf=B/'pdf/report.pdf';qa=B/'qa';render=qa/'pdf_render'
    qa.mkdir(exist_ok=True);render.mkdir(exist_ok=True)
    issues=[];checks=[]
    def require(condition,message):
        if not condition:issues.append(message)
        checks.append(dict(check=message,passed=bool(condition)))
    if not pdf.exists():raise FileNotFoundError(pdf)
    doc=fitz.open(pdf);texts=[p.get_text() for p in doc]
    full='\n\n'.join(f'--- PAGE {i+1} ---\n{t}' for i,t in enumerate(texts))
    require(len(doc)==EXPECTED_PAGES,f'Exactly {EXPECTED_PAGES} pages: five baseline plus appendices A-D')
    require(re.search(r'\bpole\b',full,re.I) is None,'Prohibited terminology absent')
    require('\ufffd' not in full,'No replacement-character glyphs')
    require('@@' not in full and re.search(r'\b(?:TODO|TBD)\b',full) is None,'No unresolved authoring placeholders')
    require(all('v6.4.4' in compact(t) for t in texts),'Version 6.4.4 header on every page')
    for letter,(page,title) in APPENDICES.items():
        found=[i+1 for i,t in enumerate(texts) if re.search(r'^Appendix\s+'+letter+r'\.',t,re.M)]
        require(found==[page],f'Appendix {letter} heading occurs only on page {page}')
        if page<=len(texts):require(compact(title) in compact(texts[page-1]),f'Appendix {letter} has its expected title')
    require('If the largest result among these three searches is selected' not in full,'No promoted across-search selection paragraph')
    require('The change from blue to red then includes the search' not in full,'Obsolete local-to-global penalty claim removed')
    for term in ('one shared coupling','leave-one-out','conditional','Clopper','1,024','2,048','0.005','0.05','1/1025'):
        require(compact(term) in compact(full),f'Standalone explanation includes: {term}')

    # The original table must remain the A-only raw-statistic record on page 1.
    baseline=json.loads((B/'results/global_results.json').read_text())
    p1=compact(texts[0])
    for row in baseline['primary']:
        for value in (str(row['peak_mass_MeV']),f"{row['global_exceedances']}/1024",f"{row['global_p_rank']:.3f}",f"{row['global_Z_excess']:.2f}"):
            require(compact(value) in p1,f'Baseline {row["scope"]}: page-1 value {value}')

    summary=pd.read_csv(B/'results/calibrated_summary.csv',dtype={'scope':str})
    fits=json.loads((B/'results/sidak_fit.json').read_text())['fits']
    audit=json.loads((B/'qa/common_coupling_audit.json').read_text())
    def page_contains(page,value,label):
        require(page<=len(texts) and compact(value) in compact(texts[page-1]),label)
    for row in summary.itertuples():
        scope=row.scope
        result_values=(f'{int(row.raw_peak_mass_MeV)}/{int(row.localfirst_peak_mass_MeV)}',
            f'{row.raw_2048_p:.3f}',f'{int(row.minp_B_k)}/1024',f'{row.minp_B_p:.3f}',
            f'[{row.minp_B_p95_low:.3f},{row.minp_B_p95_high:.3f}]',f'{row.minp_B_Z:.2f}')
        for value in result_values:page_contains(7,value,f'Appendix B {scope}: {value}')
        for value in (f'{row.sidak_fitted_p:.3f}',f'{row.sidak_grid_p:.3f}',f'{row.minp_B_p:.3f}'):
            page_contains(8,value,f'Appendix C {scope}: {value}')
        for value in (f'{int(row.raw_1024_k)}/1024',f'{row.raw_1024_p:.3f}',
                      f'{int(row.raw_new1024_k)}/1024',f'{row.raw_new1024_p:.3f}',f'{row.raw_2048_p:.3f}'):
            page_contains(9,value,f'Appendix D {scope}: {value}')
    for row in fits:
        for value in (str(row['grid_points']),f"{row['N_eff']:.2f}"):
            page_contains(8,value,f'Frozen A fit {row["scope"]}: {value}')
    for row in audit['mass_summaries']:
        if row['mass_MeV'] in (67,68,91):
            for value in (str(row['mass_MeV']),f"{row['joint_psi_hat']:.1f}",f"{row['joint_q0']:.3f}",f"{row['sum_individual_q0']:.3f}"):
                page_contains(6,value,f'Coupling audit at {row["mass_MeV"]} MeV: {value}')

    source=(B/'source/report.tex').read_text()
    require(r'\prod_y\mathcal L_y(\psi,\eta_y;m)' in source,'Common-coupling likelihood product retained in source')
    for token in (r'p_{A,m}(q)',r'\min_{m\in\mathcal M}',r'\log(1-G_A)',r'\log(1-\alpha)'):
        require(token in source,f'Calibration formula retained in source: {token}')
    refs=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source)
    require(len(refs)==4,'Four figure inclusions: two retained and two appendix comparisons')
    for path in refs:require((B/'source'/path).exists(),f'Figure file exists: {path}')

    logpath=B/'pdf/report.log'
    require(logpath.exists(),'LaTeX build log present')
    log=logpath.read_text(errors='replace') if logpath.exists() else ''
    bad_log=[]
    patterns=(r'Overfull\s+\\[hv]box',r'Undefined control sequence',r'Missing character:',
              r'LaTeX Error:',r'There were undefined references',r'Warning:.*undefined')
    for pattern in patterns:
        bad_log.extend(m.group(0) for m in re.finditer(pattern,log,re.I))
    require(not bad_log,'No overfull boxes, missing glyphs, undefined references or TeX errors')

    geometry=[];uris=set()
    for i,page in enumerate(doc):
        spans=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if span['text'].strip():spans.append(span)
        escapes=[]
        for span in spans:
            x0,y0,x1,y1=span['bbox']
            if x0<18 or y0<12 or x1>page.rect.width-18 or y1>page.rect.height-12:
                escapes.append(dict(text=span['text'],bbox=list(span['bbox'])))
        require(not escapes,f'Page {i+1}: all text, equations and table glyphs stay within page safety margins')
        require(len(texts[i].strip())>300,f'Page {i+1}: no nearly empty spillover page')
        for link in page.get_links():
            if 'uri' in link:uris.add(link['uri'])
        geometry.append(dict(page=i+1,width_points=page.rect.width,height_points=page.rect.height,
            extracted_characters=len(texts[i]),span_count=len(spans),margin_escapes=escapes))
    for url in REFERENCES:require(url in uris,f'Clickable primary reference present: {url}')

    # Always produce renders, even if a semantic/layout check failed. This lets
    # the parent inspect spillover or clipping without another tool invocation.
    for old in render.glob('page-*.png'):
        if re.fullmatch(r'page-\d+\.png',old.name):old.unlink()
    rendered=[]
    for i,page in enumerate(doc):
        dest=render/f'page-{i+1:02d}.png'
        page.get_pixmap(dpi=DPI,alpha=False).save(str(dest))
        rendered.append(str(dest.relative_to(B)))
    (qa/'calibrated_report_text.txt').write_text(full)
    result=dict(passed=not issues,automated_checks_passed=not issues,pages=len(doc),expected_pages=EXPECTED_PAGES,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        dpi=DPI,renderer='PyMuPDF '+str(fitz.VersionBind),rendered_pages=rendered,
        assertions=len(checks),failed_checks=issues,checks=checks,tex_log_findings=bad_log,pages_checked=geometry,
        hyperlinks=sorted(uris),visual_review_passed=False,
        visual_review='Pending manual inspection of every current rendered page. Automated geometry and TeX checks do not establish absence of overlap or full readability.')
    (qa/'calibrated_pdf_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','pages','expected_pages','assertions','failed_checks','rendered_pages','visual_review']},indent=2))
    return 0 if not issues else 1

if __name__=='__main__':sys.exit(main())
