#!/usr/bin/env python3
"""Publish a bounded study snapshot, preserving ZIP payloads without duplicate ZIPs."""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('publication/recent_studies_20260927')
LIMIT = 40 * 1024**2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def split_if_large(out, dst, record):
    if out.stat().st_size <= 49 * 1024**2:
        return
    parts = []
    with out.open('rb') as stream:
        for i, block in enumerate(iter(lambda: stream.read(32 * 1024**2), b''), 1):
            part = Path(str(out) + '.part%03d' % i)
            part.write_bytes(block)
            parts.append(dict(path=part.relative_to(dst).as_posix(), sha256=digest(part), bytes=len(block)))
    out.unlink()
    record['storage_parts'] = parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--destination', required=True, type=Path)
    args = ap.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    if src == dst:
        raise SystemExit('Source and publication checkout must differ.')
    roots = []
    for parent in ['study_results', 'output/pdf']:
        for p in sorted((src / parent).iterdir()):
            match = re.search(r'(202609\d{2})', p.name)
            if p.is_dir() and match and '20260913' <= match[1] <= '20260927':
                roots.append(p.relative_to(src).as_posix())
    roots += [p.relative_to(src).as_posix() for p in sorted((src / 'output/slides').iterdir()) if p.is_dir()]
    roots += ['apex_initial_studies', 'docs/2021_10pct_fixed_yield_100toy_handoff.md']
    # Include standalone APEX delivery copies, when present.
    roots += [p.relative_to(src).as_posix() for p in sorted((src / 'output/pdf').glob('*APEX*')) if p.is_file()]
    files, excluded, archives, by_hash = [], [], [], {}
    candidates = []
    for name in roots:
        p = src / name
        candidates.extend([p] if p.is_file() else sorted(p.rglob('*')))
    candidates = sorted(set(p for p in candidates if p.is_file()))

    def store_file(p, logical):
        before = p.stat()
        sha = digest(p)
        out = dst / logical
        out.parent.mkdir(parents=True, exist_ok=True)
        if before.st_size > LIMIT:
            out = Path(str(out) + '.gz')
            with p.open('rb') as source, out.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as target:
                    shutil.copyfileobj(source, target)
            encoding = 'gzip'
        else:
            if out.exists() and digest(out) != sha:
                raise RuntimeError('Refusing to overwrite a different published file: ' + logical)
            shutil.copy2(p, out)
            encoding = 'identity'
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Source changed during snapshot: ' + logical)
        storage_sha = digest(out)
        if encoding == 'identity' and storage_sha != sha:
            raise RuntimeError('Copy mismatch: ' + logical)
        item = dict(path=logical, sha256=sha, bytes=before.st_size,
                    storage_path=out.relative_to(dst).as_posix(), storage_sha256=storage_sha,
                    storage_bytes=out.stat().st_size, encoding=encoding)
        split_if_large(out, dst, item)
        by_hash.setdefault(sha, item)
        return item

    for p in candidates:
        rel = p.relative_to(src).as_posix()
        if p.is_symlink():
            raise RuntimeError('Review symlink before snapshot: ' + rel)
        if p.name == '.DS_Store' or '__pycache__' in p.parts or p.suffix in {'.pyc', '.pyo', '.lock'}:
            excluded.append(dict(path=rel, reason='OS metadata, Python bytecode, or process lock'))
            continue
        if p.suffix.lower() == '.zip':
            archives.append(p)
        else:
            files.append(store_file(p, rel))
    print('Copied', len(files), 'files from', len(roots), 'roots; indexing', len(archives), 'ZIPs.', flush=True)
    recipes, objects = [], []
    for p in archives:
        rel = p.relative_to(src).as_posix()
        zip_sha = digest(p)
        members = []
        with zipfile.ZipFile(p) as archive:
            for info in archive.infolist():
                data = archive.read(info)
                sha = hashlib.sha256(data).hexdigest()
                record = by_hash.get(sha)
                if record is None and not info.is_dir():
                    logical = (BASE / 'archive_objects' / sha[:2] / sha).as_posix()
                    out = dst / logical
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
                    record = dict(path=logical, sha256=sha, bytes=len(data), storage_path=logical,
                                  storage_sha256=sha, storage_bytes=len(data), encoding='identity')
                    if len(data) > LIMIT:
                        gz = Path(str(out) + '.gz')
                        with gz.open('wb') as raw:
                            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as target:
                                target.write(data)
                        out.unlink()
                        record.update(storage_path=gz.relative_to(dst).as_posix(),
                                      storage_sha256=digest(gz), storage_bytes=gz.stat().st_size, encoding='gzip')
                    split_if_large(dst / record['storage_path'], dst, record)
                    by_hash[sha] = record
                    objects.append(record)
                members.append(dict(name=info.filename, directory=info.is_dir(), sha256=sha,
                                    bytes=len(data), storage=record if not info.is_dir() else None,
                                    date_time=list(info.date_time), external_attr=info.external_attr,
                                    compression=info.compress_type))
        recipe_path = BASE / 'archives' / (zip_sha + '.json')
        recipe = dict(original_path=rel, original_sha256=zip_sha, original_bytes=p.stat().st_size,
                      reconstruction='Member bytes and names are exact; ZIP container bytes may differ.', members=members)
        write_json(dst / recipe_path, recipe)
        recipes.append({k: recipe[k] for k in ['original_path','original_sha256','original_bytes']} |
                       dict(recipe=recipe_path.as_posix(), members=len(members)))
        print('Indexed', rel, len(members), 'members', flush=True)
    manifest = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
                    scope='Study folders dated 2026-09-13 through 2026-09-27, saved presentation revisions, APEX comparison, and finalized injection handoff.',
                    original_checkout=str(src), original_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),
                    remote_main_at_start=subprocess.check_output(['git','rev-parse','origin/main'],cwd=src,text=True).strip(),
                    inherited_v505_commit='eb393ba86906a5738cc1cd4574fc5df7e2a43e0c',
                    roots=roots, files=files, archive_objects=objects, archives=recipes, exclusions=excluded)
    write_json(dst / BASE / 'snapshot.json', manifest)
    print(json.dumps(dict(files=len(files), archive_objects=len(objects), archives=len(recipes),
                          excluded=len(excluded), original_GB=sum(x['bytes'] for x in files)/1e9,
                          stored_GB=sum(x['storage_bytes'] for x in files+objects)/1e9), indent=2))


if __name__ == '__main__':
    main()
