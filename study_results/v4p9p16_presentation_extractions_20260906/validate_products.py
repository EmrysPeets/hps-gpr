#!/usr/bin/env python3
"""Verify current artifacts, reviewed input identity and the final rendered note."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
import pandas as pd
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise RuntimeError(name)
    for name in ['extraction','figures','report_build']:
        x=json.loads((HERE/'provenance'/f'{name}.json').read_text())
        for key in ('input_sha256','output_sha256','inputs','figures'):
            if key in x:
                check(name+':'+key,all(sha(ROOT/p)==h for p,h in x[key].items()))
    audit=json.loads((HERE/'review/independent_final_audit.json').read_text())
    check('independent audit passed',audit['passed'])
    check('independent reviewed inputs unchanged',all(sha(ROOT/p)==h for p,h in audit['input_sha256'].items()))
    parents=[]
    for folder in ['v4p9p14_interpretation_global_20260906','v4p9p15_global_2016_2021_20260906',
                   'v4p9p16_combined_global_20260906','v4p9p16_deficit_extension_20260906']:
        mf=HERE.parent/folder/'MANIFEST.csv';rows=list(csv.DictReader(mf.open()))
        check('frozen parent '+folder,all(sha(ROOT/r['path'])==r['sha256'] and
            (ROOT/r['path']).stat().st_size==int(r['bytes']) for r in rows))
        parents.append(dict(manifest=str(mf.relative_to(ROOT)),sha256=sha(mf),files=len(rows)))
    build=json.loads((HERE/'provenance/report_build.json').read_text());pdf=Path(build['pdf'])
    check('PDF build identity',sha(pdf)==build['pdf_sha256'])
    reader=PdfReader(pdf);texts=[p.extract_text() for p in reader.pages];text='\n'.join(texts)
    check('23 pages including inherited note',len(texts)==23)
    check('no blank or orphan pages',all(len(t.strip())>500 for t in texts))
    check('no unresolved references','??' not in text)
    check('new section begins page 4','Selected signal extractions and staged exposure checks' in texts[3])
    check('full combined search retained on page 2','Full combined search' in texts[1])
    check('deficit scan retained on page 3','Illustrative scan of deficits' in texts[2])
    check('exposure figure on page 15',all(s in texts[14] for s in ['Additional 20%','Cumulative 30%','Cumulative 100%']))
    check('fixed-location and sequential qualifications',all(s in text for s in
        ['disjoint','sequential','counting','stress-centered','same original-event membership']))
    log=(HERE/'note/build.log').read_text()
    check('clean LaTeX build',not re.search(r'Overfull|Undefined|undefined references|Citation .*undefined|Error:',log))
    renders=sorted((HERE/'qa/rendered').glob('page-*.png'))
    check('one render per page',len(renders)==len(texts))
    figures=sorted((HERE/'figures').glob('*.pdf'))
    check('11 reusable single-page figures',len(figures)==11 and all(len(PdfReader(p).pages)==1 for p in figures))
    for p in figures:
        t=PdfReader(p).pages[0].extract_text()
        check(p.stem+': PNG partner',p.with_suffix('.png').is_file())
        check(p.stem+': text labels',len(t)>400)
    # Spot-check the explanatory constants against frozen stress-score tables.
    positive=pd.read_csv(HERE.parent/'v4p9p16_combined_global_20260906/global/analysis/pvalue_curves.csv')
    negative=pd.read_csv(HERE.parent/'v4p9p16_deficit_extension_20260906/analysis/deficit_curves.csv')
    a=positive[(positive.method=='profiled')&(positive.mass_MeV==76)].iloc[0]
    b=negative[(negative.method=='profiled')&(negative.mass_MeV==83)].iloc[0]
    check('76 MeV explanation',abs(a.asimov_r+8.700)<.0005 and abs(a.response_sd-.979)<.0005)
    check('83 MeV explanation',abs(b.asimov_r-7.707)<.0005 and abs(b.response_sd-.983)<.0005)
    exposure=json.loads((HERE/'derived/exposure_contract.json').read_text())
    check('no new toys or unblinded data',exposure['new_toys']==exposure['new_unblinded_events']==0)
    backup=json.loads((HERE/'provenance/parent_backup.json').read_text())
    check('parent backup receipt',backup['remote_main_verified_files']==4778 and backup['shared_head_and_index_preserved'])
    result=dict(passed=True,check_count=len(checks),checks=checks,pdf_sha256=sha(pdf),pages=len(texts),
        independent_checked_conditions=audit['checked_conditions'],parent_manifests=parents,
        rendered_pages_sha256={str(p.relative_to(ROOT)):sha(p) for p in renders},
        figure_sha256={str(p.relative_to(ROOT)):sha(p) for p in figures},
        validator_sha256=sha(__file__))
    (HERE/'qa/product_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'qa/extracted_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts)))
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','rendered_pages_sha256','figure_sha256','parent_manifests')},indent=2))
if __name__=='__main__':main()
