"""Package the checked observed study with all numerical inputs and checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile
import tempfile

B = Path(__file__).resolve().parents[1]
NAME = 'HPS_GPR_v6p3p8_Observed_Morph_Extraction'
EXCLUDED = {'__pycache__', 'rendered', 'report_rendered', 'figure_rendered', 'revision_rendered'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def release(destination):
    report = B / 'pdf' / (NAME + '.pdf')
    checks = [
        'numerical_validation', 'legacy_baseline', 'statistics_review',
        'fit_replay', 'resume_fix_reproduction', 'figure_visual_qa',
        'report_visual_qa', 'statistical_text_review', 'portable_qa',
    ]
    for name in checks:
        q = json.loads((B / 'qa' / (name + '.json')).read_text())
        assert q['passed'], name
        for key in ('report_sha256', 'pdf_sha256'):
            if key in q:
                assert q[key] == sha(report), name + ': report changed after QA'
    for line in (B / 'provenance/input_manifest.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert sha(B / relative) == digest, relative
    (B / 'status.json').write_text(json.dumps({
        'status': 'complete_observed_morph_extraction_validated',
        'version': '6.3.8', 'mass_range_MeV': [60, 240],
        'mass_spacing_MeV': 1, 'observed_fits': 543,
        'toy_fit_rows': 40800, 'selected_excess_masses_MeV': [67, 79, 185],
        'selected_deficit_mass_MeV': 226,
        'report': str(report.relative_to(B)), 'report_sha256': sha(report),
        'scope': 'Local conditional comparisons; no global or coupling exclusion claim',
    }, indent=2) + '\n')
    manifest = B / 'MANIFEST.sha256'
    files = sorted(p for p in B.rglob('*') if p.is_file() and p != manifest
                   and not EXCLUDED.intersection(p.relative_to(B).parts)
                   and p.name not in ('run.lock', 'STOP')
                   and not p.name.endswith(('.pyc', '.tmp')))
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(B)}\n' for p in files))
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / 'HPS_GPR_v6p3p8_Reproducible_Study.zip'
    work_archive = Path(tempfile.mkdtemp(prefix='hps-v638-package-')) / archive.name
    with zipfile.ZipFile(work_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [manifest]:
            z.write(p, str(Path(B.name) / p.relative_to(B)))
    with zipfile.ZipFile(work_archive) as z:
        assert z.testzip() is None
        for p in files:
            assert hashlib.sha256(z.read(str(Path(B.name) / p.relative_to(B)))).hexdigest() == sha(p)
    staged_archive = destination / (archive.name + '.tmp')
    shutil.copy2(work_archive, staged_archive)
    assert sha(staged_archive) == sha(work_archive)
    staged_archive.replace(archive)
    shutil.rmtree(work_archive.parent)
    final = destination / report.name
    shutil.copy2(report, final)
    shutil.copy2(B / 'README.md', destination / 'README.md')
    info = {'pdf': str(final), 'pdf_sha256': sha(final),
            'archive': str(archive), 'archive_sha256': sha(archive),
            'archive_bytes': archive.stat().st_size, 'files': len(files) + 1,
            'manifest_sha256': sha(manifest), 'all_archive_member_hashes_verified': True}
    (destination / 'release.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    release(parser.parse_args().destination.resolve())
