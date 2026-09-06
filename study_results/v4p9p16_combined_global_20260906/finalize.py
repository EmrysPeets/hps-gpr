#!/usr/bin/env python3
"""Seal the independently checked combined numerical and report products."""
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def estimate(t):
    return f"<{t['upper95_one_sided']:.5g} (one-sided 95% bound)" if t['k']==0 else f"{t['p']:.6g}"
def main():
    product=json.loads((HERE/'qa/product_validation.json').read_text())
    visual=json.loads((HERE/'qa/visual_review.json').read_text())
    numeric=json.loads((HERE/'qa/numerical_validation.json').read_text())
    independent=json.loads((HERE/'review/independent_final_audit.json').read_text())
    assert product['passed'] and visual['passed'] and numeric['passed']
    assert independent['complete'] and independent['checked_conditions_passed']
    build=json.loads((HERE/'provenance/report_build.json').read_text())
    pdf=Path(build['pdf'])
    assert sha(pdf)==build['pdf_sha256']==product['pdf_sha256']==visual['pdf_sha256']
    assert all(sha(ROOT/p)==h for p,h in product['rendered_pages_sha256'].items())
    s=json.loads((HERE/'global/analysis/summary.json').read_text())
    rows=[]
    for method in ('profiled','fixed'):
        q=s['methods'][method]
        rows.append(f"| {method} | {q['peak_mass_MeV']} | {q['local_common_truth_p']:.5g} | {estimate(q['global_gp'])} | {q['global_direct']['k']}/1000 |")
    readme=f"""# HPS-GPR analysis note v4.9.16

This is the completed combined search of full 2015, full 2016 and native
2021 10% over all 232 integer masses from 19 through 250 MeV.

- [Analysis note PDF](../../{pdf.relative_to(ROOT)})
- [Full observed limit with local and global p-values below](figures/combined_observed_limit_and_pvalues.pdf)
- [LaTeX source](note/analysis_note.tex)
- [Representative p-value table](global/analysis/representative_pvalues.csv)
- [Full observed upper-limit table](global/observed.csv)
- [Independent HEP review](review/HEP_COMBINED_REVIEW.md)
- [Reproduction and continuation](NEXT_STEPS.md)

The upper curve is the pointwise asymptotic 90% CLs result. The new GP method
estimates global discovery-score tails under one joint stress-background scenario;
it does not make the upper-limit curve toy-calibrated or simultaneous.
Legacy expected bands use a different ensemble and are omitted.

## Full search and probabilities

The active sets are 2015 alone at 19–38 MeV, 2015+2016 at 39–49,
all three at 50–90, 2016+2021 at 91–180, and 2021 alone at 181–250.
These choices follow support and are fixed before the combined results.
Every multi-dataset mass uses the actual shared-coupling likelihood.

| Statistic | Principal peak [MeV] | Local common-background p | GP global p | Direct count |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The separate raw-peak ordering is retained in the note and figure. It is a
different test, not a competing numerical estimate of the same probability.
Zero-count tails are bounds. These conditional results do not establish
discovery evidence, physical background validity, expected sensitivity or
confidence-interval coverage. The inherited 2016 qualifications remain.

## Reuse and numerical work

The study uses ten pilot and 1,000 independent validation joint experiments.
They pair equal row IDs from distinct year-specific full-spectrum RNG streams,
reusing existing spectra without counting copies as new experiments. The
1,626-bin response basis preserves shared-data correlations across membership
boundaries. It contains 1,627 Asimov spectra including the baseline.

There are 142 newly fitted multi-dataset coordinates and 90 reused, validated
single-dataset coordinates. Each new coordinate has an exact pilot and
response stencil. Six complete exact response columns test the boundary
regions. The analyzer samples 200,000 GP fields per method over the whole
union at once; it does not join independent segment maxima.

Final acceptance includes {numeric['check_count']} numerical/product identity
checks, {independent['checked_conditions']} independently implemented checks,
{product['check_count']} report/probability checks, and visual review of all
{product['page_count']} PDF pages. Earlier manifests, the shared Git HEAD and
the index were preserved. No commit, push or merge was made for this new note.

## Artifact map

- global/points: per-mass joint or reused root vectors and numerical audits.
- global/references: paired exact pilot and response reference vectors.
- global/spectra: coherent joint pilot, validation and Asimov spectra.
- global/analysis: p-values, covariance, GP/direct maxima and diagnostics.
- provenance: input, figure and report bindings.
- qa and review: numerical, semantic, independent and rendered checks.

MANIFEST.csv covers this derivative and the final PDF. The manifest and its
self-verification companion are excluded from their own inventory.
"""
    (HERE/'README.md').write_text(readme)
    excluded={HERE/'MANIFEST.csv',HERE/'qa/manifest_verification.json'}
    paths=sorted([p for p in HERE.rglob('*') if p.is_file() and p not in excluded
                  and '__pycache__' not in p.parts]+[pdf])
    with (HERE/'MANIFEST.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256'])
        writer.writeheader()
        for p in paths:writer.writerow(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    rows=list(csv.DictReader((HERE/'MANIFEST.csv').open()))
    passed=all((ROOT/r['path']).stat().st_size==int(r['bytes']) and sha(ROOT/r['path'])==r['sha256'] for r in rows)
    result=dict(passed=passed,files=len(rows),bytes=sum(int(r['bytes']) for r in rows),
        manifest_sha256=sha(HERE/'MANIFEST.csv'),pdf_sha256=sha(pdf))
    (HERE/'qa/manifest_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not passed:raise RuntimeError('Final manifest verification failed')
if __name__=='__main__':main()
