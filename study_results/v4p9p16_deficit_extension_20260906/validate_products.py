#!/usr/bin/env python3
"""Check preserved source identities and the rendered deficit-note extension."""
from pathlib import Path
import csv,hashlib,json,re,subprocess
from pypdf import PdfReader
import fitz
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v4p9p16_combined_global_20260906'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parent=json.loads((HERE/'provenance/parent.json').read_text())
    summary=json.loads((HERE/'analysis/summary.json').read_text())
    independent=json.loads((HERE/'review/independent_final_audit.json').read_text())
    build=json.loads((HERE/'provenance/report_build.json').read_text())
    checks={}
    checks['parent_manifest_identity']=sha(PARENT/'MANIFEST.csv')==parent['manifest_sha256']
    checks['all_original_products_intact']=all((ROOT/r['path']).stat().st_size==int(r['bytes']) and sha(ROOT/r['path'])==r['sha256']
        for r in csv.DictReader((PARENT/'MANIFEST.csv').open()))
    checks['analysis_passed']=summary['passed'] and all(summary['checks'].values())
    checks['independent_audit_passed']=independent['passed'] and not independent['failures']
    for name,record in [('analysis',summary),('independent',independent),('report',build)]:
        checks[name+'_inputs_intact']=all(sha(ROOT/p)==h for p,h in record['input_sha256'].items())
    checks['figures_intact']=all(sha(ROOT/p)==h for p,h in build['figure_sha256'].items())
    checks['same_realizations_no_new_fits']=summary['gp_realizations_reused']==200000 and summary['direct_joint_scans_reused']==1000 and summary['new_likelihood_fits']==summary['new_independent_toys']==0
    checks['all_parent_tables_preserved']=all(sha(p)==sha(HERE/'note'/p.name) for p in (PARENT/'note').glob('*.tex') if p.name!='analysis_note.tex')
    pdf=Path(build['pdf']);checks['pdf_identity']=sha(pdf)==build['pdf_sha256']
    pages=[p.extract_text() for p in PdfReader(pdf).pages]
    normalized=[re.sub(r'\s+',' ',re.sub(r'-\s*\n\s*','-',p)).replace('−','-') for p in pages]
    whole=' '.join(normalized)
    checks['nine_complete_pages']=len(pages)==9 and all(len(p)>450 for p in pages)
    checks['deficit_section_on_page_three']=normalized[2].startswith('3 Illustrative scan of deficits')
    checks['original_limit_and_excess_retained']='Full combined search' in normalized[1] and '90%' in normalized[1]
    required=('negative rate or coupling','83 MeV','72 MeV','0.2495','-0.676',
        '0/200,000','0/1,000','same realizations','no new fits or independent toys',
        'do not adjust for choosing','physical background validity')
    checks['deficit_scope_and_numbers_visible']=all(x in normalized[2] for x in required)
    checks['references_resolved']=not any(x in whole for x in ('??','TODO','PLACEHOLDER'))
    logs=(HERE/'note/build.log').read_text()+(pdf.parent/'analysis_note.log').read_text()
    checks['no_tex_layout_or_reference_warnings']=not re.search(r'Overfull|undefined|LaTeX Warning',logs,re.I)
    violations=[]
    for i,p in enumerate(fitz.open(pdf),1):
        for b in p.get_text('blocks'):
            if b[4].strip()==str(i) and b[1]>740:continue
            if b[0]<35 or b[1]<30 or b[2]>577 or b[3]>765:
                violations.append(dict(page=i,bbox=b[:4]))
    checks['all_content_in_bounds']=not violations
    renders=sorted((HERE/'qa/pages').glob('page-*.png'))
    checks['all_pages_rendered']=len(renders)==len(pages)
    checks['shared_git_head_preserved']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==parent['git_head']
    checks['shared_index_preserved']=hashlib.sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT)).hexdigest()==parent['index_diff_sha256']
    report=dict(passed=all(checks.values()),check_count=len(checks),checks=checks,
        page_count=len(pages),pdf_sha256=sha(pdf),bounds_violations=violations,
        rendered_pages_sha256={str(p.relative_to(ROOT)):sha(p) for p in renders})
    (HERE/'qa/product_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Deficit product validation failed')
if __name__=='__main__':main()
