#!/usr/bin/env python3
"""Verify the published snapshot, restore compressed files, or reconstruct one ZIP."""
import argparse
import contextlib
import gzip
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'publication/recent_studies_20260927'


def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024**2), b''):
        h.update(block)
    return h.hexdigest()


def safe(path):
    p = (ROOT / path).resolve()
    if ROOT not in p.parents:
        raise ValueError('Path outside publication checkout: ' + str(path))
    return p


@contextlib.contextmanager
def payload(record):
    with contextlib.ExitStack() as stack:
        if 'storage_parts' in record:
            stream = stack.enter_context(tempfile.TemporaryFile())
            for part in record['storage_parts']:
                with safe(part['path']).open('rb') as src:
                    if sha(src) != part['sha256']:
                        raise ValueError('Corrupt part: ' + part['path'])
                    src.seek(0)
                    shutil.copyfileobj(src, stream)
            stream.seek(0)
        else:
            stream = stack.enter_context(safe(record['storage_path']).open('rb'))
        if sha(stream) != record['storage_sha256']:
            raise ValueError('Corrupt stored file: ' + record['path'])
        stream.seek(0)
        if record['encoding'] == 'gzip':
            stream = stack.enter_context(gzip.GzipFile(fileobj=stream, mode='rb'))
        yield stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--restore-large', action='store_true')
    parser.add_argument('--archive', help='Original ZIP path, as listed in snapshot.json')
    parser.add_argument('--output', type=Path, help='Destination ZIP; must not already exist')
    args = parser.parse_args()
    manifest = json.loads((BASE / 'snapshot.json').read_text())
    records = manifest['files'] + manifest['archive_objects']
    if args.verify:
        seen = set()
        for record in records:
            key = record['storage_path']
            if key in seen:
                continue
            seen.add(key)
            with payload(record) as stream:
                if sha(stream) != record['sha256']:
                    raise ValueError('Payload mismatch: ' + record['path'])
        total = 0
        for archive in manifest['archives']:
            recipe = json.loads(safe(archive['recipe']).read_text())
            for member in recipe['members']:
                if not member['directory']:
                    assert member['sha256'] == member['storage']['sha256']
                    assert member['storage']['storage_path'] in seen
                total += 1
        print('PASS:',len(seen),'stored payloads;',len(manifest['archives']),'ZIP recipes;',total,'archive members')
    if args.restore_large:
        restored = 0
        for record in manifest['files']:
            if record['encoding'] == 'identity':
                continue
            target = safe(record['path'])
            if target.exists():
                with target.open('rb') as stream:
                    if sha(stream) != record['sha256']:
                        raise ValueError('Refusing to overwrite changed file: ' + str(target))
                continue
            with payload(record) as src, tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as out:
                shutil.copyfileobj(src, out)
                tmp = Path(out.name)
            with tmp.open('rb') as stream:
                assert sha(stream) == record['sha256']
            tmp.replace(target)
            restored += 1
        print('Restored',restored,'large files. Existing identical files were preserved.')
    if args.archive:
        if args.output is None or args.output.exists():
            parser.error('--archive requires a new --output path')
        entry = next(x for x in manifest['archives'] if x['original_path'] == args.archive)
        recipe = json.loads(safe(entry['recipe']).read_text())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,'x',allowZip64=True) as out:
            for member in recipe['members']:
                info = zipfile.ZipInfo(member['name'], tuple(member['date_time']))
                info.external_attr = member['external_attr']
                info.compress_type = member['compression']
                if member['directory']:
                    out.writestr(info,b'')
                else:
                    with payload(member['storage']) as src:
                        data = src.read()
                    assert hashlib.sha256(data).hexdigest() == member['sha256']
                    out.writestr(info,data)
        with zipfile.ZipFile(args.output) as check:
            assert len(check.infolist()) == len(recipe['members'])
            for info, member in zip(check.infolist(),recipe['members']):
                assert info.filename == member['name']
                assert hashlib.sha256(check.read(info)).hexdigest() == member['sha256']
        print('PASS: reconstructed',len(recipe['members']),'exact member payloads;',args.output)
        print('ZIP container metadata/compression may differ from the original container hash.')
    if not (args.verify or args.restore_large or args.archive):
        parser.print_help()


if __name__ == '__main__':
    main()
