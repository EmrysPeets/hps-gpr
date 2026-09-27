"""Check retained pages, new numerical statements and rendered PDF layout."""
from pathlib import Path
import hashlib,json,re,sys
import fitz
import pandas as pd
B=Path(__file__).resolve().parents[1]


def compact(t):return re.sub(r'\s+','',t)


def main():
    doc=fitz.open(B/'pdf/report.pdf');parent=fitz.open(B/'provenance/parent_v644_report.pdf')
    texts=[p.get_text() for p in doc];issues=[];checks=0
    def check(ok,reason):
        nonlocal checks
        checks+=1
        if not ok:issues.append(reason)
    check(len(doc)==12,'Expected nine retained pages plus three appendix pages.')
    for i,p in enumerate(parent):
        expected=p.get_text().replace('6.4.4','6.4.5').replace('25 September 2026','26 September 2026')
        check(i<len(doc) and compact(texts[i])==compact(expected),f'Parent page{i+1} changed beyond version/date.')
    full='\n'.join(texts);check(re.search(r'\bpole\b',full,re.I) is None,'Prohibited terminology.')
    check('\ufffd' not in full and '@@' not in full,'Unresolved glyph or placeholder.')
    for i,term in ((9,'E.1.'),(10,'E.2.'),(11,'E.3.')):
        check(i<len(texts) and term in texts[i],f'Missing subsection{term} onpage{i+1}.')
    s=pd.read_csv(B/'results/correlation_global_summary.csv',dtype={'scope':str})
    for r in s.itertuples():
        for v in (f'{r.k}/1024',f'{r.p:.3f}',f'[{r.low:.3f},{r.high:.3f}]',f'{r.Z:.2f}',f'{r.independent_empirical_p:.3f}'):
            check(len(texts)>10 and compact(v) in compact(texts[10]),f'Missing global table value{r.scope}:{v}.')
        for v in (f'{r.local_rank_count}/1025',f'{r.N_toy_equivalent:.1f}',f'[{r.N_toy_equivalent_low:.1f},{r.N_toy_equivalent_high:.1f}]'):
            check(len(texts)>11 and compact(v) in compact(texts[11]),f'Missing equivalent-count value{r.scope}:{v}.')
    w=pd.read_csv(B/'results/resolution_counts.csv',dtype={'scope':str})
    for r in w.itertuples():
        for v in (f'{r.mean_MC_core_width_MeV:.3f}',f'{r.N_MC_resolution:.2f}',f'{r.N_legacy_resolution:.2f}'):
            check(len(texts)>11 and v in texts[11],f'Missing resolution value{r.scope}:{v}.')
    log=(B/'pdf/report.log').read_text()
    check(not any(s in log for s in ('Overfull','Undefined control sequence','Missing character:','LaTeX Error:')),'LaTeX overflow/error.')
    out=B/'qa/rendered';out.mkdir(exist_ok=True)
    for old in out.glob('page-*.png'):old.unlink()
    for i,p in enumerate(doc):
        for block in p.get_text('blocks'):
            x0,y0,x1,y1,*_=block
            check(x0>=18 and y0>=12 and x1<=p.rect.width-18 and y1<=p.rect.height-12,f'Text outside safe page bounds on{i+1}.')
        check(len(texts[i])>300,f'Almost-empty spillover page{i+1}.')
        if i>=9:p.get_pixmap(dpi=115,alpha=False).save(out/f'page-{i+1:02d}.png')
    (B/'qa/report_text.txt').write_text(full)
    result=dict(passed=not issues,checks=checks,pages=len(doc),first_nine_parent_pages_preserved_except_date_and_version=True if not any('Parent page' in x for x in issues) else False,
        pdf_sha256=hashlib.sha256((B/'pdf/report.pdf').read_bytes()).hexdigest(),failed_checks=issues,visual_review_passed=False,
        visual_review='Pending inspection of the three new rendered pages; original nine pages verified against the visually reviewed parent.')
    (B/'qa/pdf_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return int(bool(issues))


if __name__=='__main__':sys.exit(main())
