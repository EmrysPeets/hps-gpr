#!/usr/bin/env python3
"""Inventory the completed illustrative deficit extension."""
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    product=json.loads((HERE/'qa/product_validation.json').read_text())
    visual=json.loads((HERE/'qa/visual_review.json').read_text())
    independent=json.loads((HERE/'review/independent_final_audit.json').read_text())
    build=json.loads((HERE/'provenance/report_build.json').read_text())
    assert product['passed'] and visual['passed'] and independent['passed']
    pdf=Path(build['pdf'])
    assert sha(pdf)==product['pdf_sha256']==visual['pdf_sha256']==build['pdf_sha256']
    assert all(sha(ROOT/p)==h for p,h in product['rendered_pages_sha256'].items())
    s=json.loads((HERE/'analysis/summary.json').read_text());q=s['methods']['profiled']
    text=f"""# v4.9.16 with an illustrative deficit scan

- [Updated nine-page analysis note](../../{pdf.relative_to(ROOT)})
- [Standalone deficit figure](figures/combined_deficit_scan.pdf)
- [PNG figure](figures/combined_deficit_scan.png)
- [Numerical curves](analysis/deficit_curves.csv)
- [Independent HEP review](review/HEP_DEFICIT_REVIEW.md)

The new deficit figure is page 3, immediately after the complete observed
upper-limit and excess-probability figure. It shows signed fitted roots,
stress-background offsets, local deficit tails and the two distinct
union-global deficit orderings over 19–250 MeV.

The deepest raw profiled deficit is at {q['raw_ordering']['peak_mass_MeV']} MeV
with root {-q['raw_ordering']['depth']:.6f}. The strongest stress-centered
deficit is at {q['peak_mass_MeV']} MeV: the observed root is {q['observed_r']:.6f}
and the stress Asimov offset is {q['asimov_r']:+.6f}. Its extreme conditional
tail is unresolved by both ensembles. Every simulated raw-depth maximum
exceeds the observed raw-depth maximum. These are background diagnostics,
not particle claims or tests of overall goodness of fit.

This extension reuses 1,000 coherent joint Poisson scans and exactly replays
the same 200,000 Gaussian fields per method. Both positive-maxima vectors
were reproduced bitwise before accepting the negative maxima. No new fits,
Poisson spectra or independent Gaussian realizations were added.

The original v4.9.16 directory and PDF remain unchanged. No upper-limit
endpoint or positive-scan result was modified. The deficit direction was
requested after the excess study; these conditional probabilities do not
adjust for selecting a direction, method or ordering. All original 2016
qualifications remain.

## Reproduction

Run from the repository root. Preserve this final manifest before rebuilding.

    python3 -B study_results/v4p9p16_deficit_extension_20260906/analyze_deficits.py
    python3 -B study_results/v4p9p16_deficit_extension_20260906/make_report.py
    python3 -B study_results/v4p9p16_deficit_extension_20260906/review/independent_audit.py

The final independent audit passed {independent['checked_conditions']} conditions.
Product validation passed {product['check_count']} checks; all nine rendered
pages were inspected. Rebuilds require fresh rendering, product validation,
visual review and a new manifest. No fitting runner is needed.

PROTOCOL.md defines the mirrored negative-root gate and both orderings.
The parent manifest is bound in provenance/parent.json and rechecked in full.
MANIFEST.csv covers this derivative and its final PDF; it excludes itself
and its own verification record.
"""
    (HERE/'README.md').write_text(text)
    excluded={HERE/'MANIFEST.csv',HERE/'qa/manifest_verification.json'}
    paths=sorted([p for p in HERE.rglob('*') if p.is_file() and p not in excluded and '__pycache__' not in p.parts]+[pdf])
    with (HERE/'MANIFEST.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);writer.writeheader()
        for p in paths:writer.writerow(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    rows=list(csv.DictReader((HERE/'MANIFEST.csv').open()))
    passed=all(sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==int(r['bytes']) for r in rows)
    result=dict(passed=passed,files=len(rows),manifest_sha256=sha(HERE/'MANIFEST.csv'),pdf_sha256=sha(pdf))
    (HERE/'qa/manifest_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert passed
if __name__=='__main__':main()
