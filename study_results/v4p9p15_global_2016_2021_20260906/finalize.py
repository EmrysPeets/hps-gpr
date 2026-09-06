#!/usr/bin/env python3
"""Write the reader index and seal the checked numerical/report artifacts."""
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def value(t):
    return f"<{t['upper95_one_sided']:.3g} (95% upper bound)" if t['k']==0 else f"{t['p']:.6g}"

def main():
    qa=json.loads((HERE/'qa/product_validation.json').read_text())
    visual=json.loads((HERE/'qa/visual_review.json').read_text())
    assert qa['passed'] and visual['passed']
    rows=[];audits=[]
    for year,label in [('2016','2016 full'),('2021','2021 10%')]:
        s=json.loads((HERE/'global_fast'/year/'analysis/summary.json').read_text())
        audit=json.loads((HERE/'global_fast'/year/'acceleration_validation.json').read_text())
        assert audit['passed']
        for method in ('profiled','fixed'):
            x=s['methods'][method];d=x['global_direct']
            rows.append(f"| {label} | {method} | {x['peak_mass_MeV']} | {x['local_common_truth_p']:.3g} | {value(x['global_gp'])} | {d['k']}/{d['n']} |")
        audits.append(dict(dataset=year,exact_fallback_masses=audit['exact_fallback_masses_MeV'],
                           max_paired_root_error=max(x['max_root_error'] for x in audit['overlaps'].values() if x['comparison_available']),
                           bounded_atom_flips=sum(x['bounded_atom_flips'] for x in audit['overlaps'].values() if x['comparison_available']),
                           max_sentinel_correlation_error=max(x['max_absolute_correlation_error'] for x in audit['full_response'].values())))
    pdf=Path(json.loads((HERE/'provenance/report_build.json').read_text())['pdf'])
    assert qa['pdf_sha256']==visual['pdf_sha256']==sha(pdf)
    assert all(sha(ROOT/name)==digest for name,digest in qa['rendered_pages_sha256'].items())
    readme=f'''# v4.9.15: full 2016 and 2021 10% global-significance study

The [LaTeX reader report](../../{pdf.relative_to(ROOT)}) extends the frozen
2015 study. Its source is [note/reader_report.tex](note/reader_report.tex).
The implementation follows the covariance construction of
[Ananiev and Read](https://arxiv.org/abs/2206.12328v3), explicitly retaining
nonzero offsets and nonunit widths in the likelihood-root field.

Each dataset has ten full-spectrum pilot scans, 1,000 additional independent
Poisson validation scans, and its complete one-bin Asimov-response ensemble
(721 scans for 2016; 423 for 2021). The analyzer samples 200,000 GP fields per
method per dataset. The 2016 grid has 142 points from 39 to 180 MeV; the 2021
grid has 201 points from 50 to 250 MeV. Both have 1 MeV spacing.

## Principal minimum-local-p ordering

| Dataset | Statistic | Peak mass [MeV] | Common-truth local p | GP global p | Direct exceedances |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

These are **conditional stress-background diagnostics**, not final discovery
probabilities or a global calibration of the v4.9.13 two-truth envelope.
The separate raw-root ordering is saved and plotted as a different test.
Zero-count tails are limits, not measured zero probabilities. A small 2016
conditional probability can reject the behavior of its particular archived
background construction without identifying a particle. Its source-fit
waiver, source-development overlap, transition region and inherited numerical
exception remain explicit. A raw global probability near one is not a
goodness-of-fit certificate. No combined-dataset or continuous-mass result
and no expected-sensitivity claim is made.

## Numerical implementation and checks

Exact pilot scans were completed first. The exact 2016 1,000-scan calculation
was paused after 81 complete mass columns because measured scaling was poor.
All those references are retained. The accepted derivative uses the existing
calibration accelerator with per-coordinate checks and an exact fallback for
the entire coordinate. The physical/statistical procedure, spectra, seeds and
mass grid stay the same. Replaying the exact and accelerated backend does not
create additional independent toys.

Every mass has an exact Asimov baseline and a declared bin-response stencil;
six masses per dataset have complete exact Asimov columns. The final audit
checks centered responses, widths and correlations, every available exact
pilot/validation root, their bounded-atom classifications, and all source
hashes. See [the numerical amendment](ACCELERATION_PROTOCOL.md),
[full-response gates](ACCELERATION_RESPONSE_GATES.md), and
[the independent HEP review](review/HEP_EXTENSION_REVIEW.md).

## Files and continuation

- `global_fast/<year>/`: accepted pilot, validation and Asimov products,
  numerical gates, full exact response checks, and execution records.
- `global_fast/<year>/analysis/`: p-value CSVs, covariance matrices, direct/GP
  maxima, marginal diagnostics, tail curves and summary JSON.
- `global/<year>/`: preserved exact pilots and the partial exact 2016
  validation reference. Do not pool these paired replays with accepted toys.
- `figures/`: separate p-value, mean/width, correlation and global-tail plots.
- `provenance/`: source references, timings, backend comparisons and PDF inputs.
- `qa/`: {qa['check_count']} product checks and review of every rendered PDF page.

[NEXT_STEPS.md](NEXT_STEPS.md) gives runnable reproduction and continuation
instructions, independent-seed requirements, finite-tail precision guidance,
finer-grid requirements and the conditions for a joint search. The complete
2015 manifest remains unchanged. `MANIFEST.csv` covers this derivative and its
final PDF; the manifest and its self-check companion are excluded from their
own inventory to avoid recursive hashes.
'''
    (HERE/'README.md').write_text(readme)
    (HERE/'provenance/numerical_release_summary.json').write_text(json.dumps(audits,indent=2)+'\n')
    exclude={HERE/'MANIFEST.csv',HERE/'qa/manifest_verification.json'}
    files=sorted([p for p in HERE.rglob('*') if p.is_file() and p not in exclude and '__pycache__' not in p.parts]+[pdf])
    with (HERE/'MANIFEST.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);writer.writeheader()
        for path in files:writer.writerow(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    rows=list(csv.DictReader((HERE/'MANIFEST.csv').open()))
    passed=all((ROOT/r['path']).stat().st_size==int(r['bytes']) and sha(ROOT/r['path'])==r['sha256'] for r in rows)
    record=dict(passed=passed,files=len(rows),bytes=sum(int(r['bytes']) for r in rows),manifest_sha256=sha(HERE/'MANIFEST.csv'),pdf_sha256=sha(pdf))
    (HERE/'qa/manifest_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not passed:raise RuntimeError('Manifest verification failed')

if __name__=='__main__':main()
