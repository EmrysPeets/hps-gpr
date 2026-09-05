#!/usr/bin/env python3
"""Validate this isolated width scan without running fits or generating toys."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

HERE=Path(__file__).resolve().parent
DERIVED=HERE/'derived'
REPO=HERE.parents[2]


def sha(path):
    # Keep historical provenance unchanged, but verify this checkout's files.
    candidate=Path(path)
    if candidate.is_absolute() and 'study_results' in candidate.parts:
        candidate=REPO.joinpath(*candidate.parts[candidate.parts.index('study_results'):])
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def main(*,write_report=True):
    summary=json.loads((DERIVED/'summary.json').read_text())
    diagnostics=json.loads((DERIVED/'limit_solver_diagnostics.json').read_text())
    with (DERIVED/'width_scan_upper_limits.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    expected={(m,s) for m in range(50,251) for s in (.8,.9,1.,1.1,1.2)}
    keys={(int(row['mass_MeV']),float(row['width_scale'])) for row in rows}
    assert len(rows)==1005 and keys==expected
    nominal={int(row['mass_MeV']):row for row in rows if float(row['width_scale'])==1}
    max_nominal_error=0.
    for row in rows:
        mass=int(row['mass_MeV']);scale=float(row['width_scale'])
        eps2=float(row['eps2_90']);epsilon=float(row['epsilon_90'])
        full=float(row['A90_full_template_events']);window=float(row['A90_fitted_window_events'])
        k=float(row['signal_yield_per_eps2_total']);fraction=float(row['template_fraction_in_fixed_window'])
        assert all(math.isfinite(v) and v>0 for v in (eps2,epsilon,full,window,k,fraction))
        assert 0<fraction<=1 and row['limit_optimizer_ok']=='True'
        assert math.isclose(epsilon**2,eps2,rel_tol=1e-12)
        assert math.isclose(full,k*eps2,rel_tol=1e-12)
        assert math.isclose(window,full*fraction,rel_tol=1e-12)
        assert math.isclose(float(row['signal_yield_per_eps2_fitted_window']),k*fraction,rel_tol=1e-12)
        ratio=eps2/float(nominal[mass]['eps2_90'])
        assert math.isclose(float(row['limit_ratio_to_nominal']),ratio,rel_tol=1e-12)
        assert math.isclose(full/float(nominal[mass]['A90_full_template_events']),ratio,rel_tol=1e-12)
        assert abs(float(row['cls_at_limit'])-.1)<2e-6
        if scale==1:
            max_nominal_error=max(max_nominal_error,abs(eps2/float(row['nominal_saved_eps2_90'])-1))
    assert max_nominal_error<5e-4
    assert len(diagnostics)==1005
    assert {(r['mass_MeV'],r['width_scale']) for r in diagnostics}==expected
    for result in diagnostics:
        assert result['optimizer_ok']
        assert result['bracket_low_cls']>.1>=result['bracket_high_cls']
        assert result['bracket_low_eps2']<=result['eps2_90']<=result['bracket_high_eps2']
        assert result['profile_status']['numerical_monotonicity']['maximum_sampled_cls_increase']<=5e-4
    assert summary['new_toy_spectra']==0 and summary['upper_limits']['number_rejected']==0
    for source in summary['source_hashes'].values():
        assert sha(source['path'])==source['sha256'],source['path']
    for path,digest in summary['protected_sha256'].items():
        assert sha(path)==digest,path
    figures=list((HERE/'figures').glob('*.png'))
    assert len(figures)==4
    for path in figures:
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and path.stat().st_size>20000
    for target in re.findall(r'\]\(([^)]+)\)',(HERE/'README.md').read_text()):
        if target=='derived/validation.json':
            continue
        assert (HERE/target).is_file(),target
    report={'status':'passed','n_mass_width_pairs':len(rows),'new_toy_spectra':0,
            'nominal_max_relative_limit_difference':max_nominal_error,
            'n_protected_unchanged_files':len(summary['protected_sha256']),
            'n_verified_numerical_sources':len(summary['source_hashes']),
            'n_figures':len(figures),'all_limit_conversion_and_solver_checks_passed':True,
            'artifact_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
                               if p.is_file() and p!=DERIVED/'validation.json'}}
    if write_report:
        (DERIVED/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':
    main()
