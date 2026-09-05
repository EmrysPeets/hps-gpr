#!/usr/bin/env python3
"""Portable, read-only validation of the v4.9.12.5 reference snapshot."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import re

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
STUDY=REPO/'study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905'
FINAL=REPO/'study_results/v4p9p12_final_dataset_combinations_20260902'
RUNTIME=REPO/'study_results/v4p9p7_2016_support_combined_100toy_20260902'
MANIFEST=HERE/'release_manifest.json'
EXCLUDED={'.mplcache','__pycache__','.DS_Store'}
AMENDED={str((STUDY/'README.md').relative_to(REPO)),
         str((STUDY/'resolution_width_scan/validate_outputs.py').relative_to(REPO)),
         str((STUDY/'resolution_width_scan/derived/validation.json').relative_to(REPO))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_path(path):
    parts=Path(path).parts
    assert 'study_results' in parts,path
    return REPO.joinpath(*parts[parts.index('study_results'):])


def release_files():
    files=[REPO/'README.md']
    for directory in (HERE,STUDY,FINAL,RUNTIME):
        files.extend(p for p in directory.rglob('*') if p.is_file() and p!=MANIFEST
                     and not any(part in EXCLUDED for part in p.relative_to(directory).parts))
    return sorted(set(files))


def check_study():
    # Verify all historical provenance against this checkout, never the author's paths.
    checked=0
    for relative in ('derived/summary.json','reverse_injection/derived/summary.json',
                     'resolution_width_scan/derived/summary.json'):
        summary=json.loads((STUDY/relative).read_text())
        sources=summary.get('source_hashes',summary.get('sources',{}))
        for source in sources.values():
            target=archived_path(source['path'])
            assert sha(target)==source['sha256'],str(target)
            checked+=1
        for path,digest in summary.get('protected_sha256',{}).items():
            assert sha(archived_path(path))==digest,path
            checked+=1
    validator=STUDY/'resolution_width_scan/validate_outputs.py'
    spec=importlib.util.spec_from_file_location('width_snapshot_validator',validator)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(write_report=False)
    runtime=json.loads((RUNTIME/'runtime_combined/runtime_manifest.json').read_text())
    for relative,digest in runtime['package_files'].items():
        assert sha(RUNTIME/'runtime_combined'/relative)==digest,relative
    for path in release_files():
        if path.suffix=='.py':
            ast.parse(path.read_text(),filename=str(path))
        if path.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if '://' in target or target.startswith('#'):
                    continue
                candidate=(path.parent/target.split('#')[0]).resolve()
                assert candidate.is_relative_to(REPO),target
                assert candidate.is_file(),f'{path}: {target}'
    return checked


def write_manifest(source):
    source=Path(source).resolve()
    records={}
    exact_count=0
    for path in release_files():
        relative=path.relative_to(REPO).as_posix()
        item={'sha256':sha(path),'bytes':path.stat().st_size}
        original=source/relative
        if original.is_file() and relative!='README.md':
            item['source_sha256']=sha(original)
            item['identical_to_source']=item['source_sha256']==item['sha256']
            assert item['identical_to_source'] or relative in AMENDED,relative
            exact_count+=int(item['identical_to_source'])
        records[relative]=item
    payload={'version':'4.9.12.5','date':'2026-09-05',
             'base_commit':'e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6',
             'source_checkout_commit':'cd8f5bf2bae4eff0ce9442be7774bcf74a559c9c',
             'scope':'2021 width scan, peak-dip pilot, reverse injections, and frozen dependencies',
             'new_toys_during_publication':0,'original_cached_toy_spectra':20,
             'byte_identical_source_files':exact_count,'publication_amended_files':sorted(AMENDED),
             'files':records}
    MANIFEST.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',metavar='SOURCE_CHECKOUT',
                        help='Publisher-only: seal files after byte comparison with the original checkout.')
    args=parser.parse_args()
    if args.write_manifest:
        write_manifest(args.write_manifest)
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['version']=='4.9.12.5'
    actual={p.relative_to(REPO).as_posix() for p in release_files()}
    assert actual==set(manifest['files'])
    for relative,record in manifest['files'].items():
        path=REPO/relative
        assert sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],relative
    checked=check_study()
    print(json.dumps({'release':'4.9.12.5','status':'passed','manifest_files':len(actual),
                      'historical_source_checks':checked,
                      'byte_identical_source_files':manifest['byte_identical_source_files'],
                      'new_toys':0},indent=2))


if __name__=='__main__':
    main()
