#!/usr/bin/env python3
"""Losslessly archive or restore the large 300-toy limit ledger."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    raw = HERE / "derived/toy_limits_300toys.csv"
    archive = raw.with_suffix(".csv.gz")
    metadata = HERE / "qa/ledger_archive_300toys.json"
    manifest = json.loads((HERE / "derived/run_manifest_300toys.json").read_text())
    expected = manifest["artifacts_sha256"]["limits"]
    if args.restore:
        record = json.loads(metadata.read_text())
        if sha256(archive) != record["archive_sha256"]:
            raise SystemExit("Compressed ledger checksum differs from its release record")
        if raw.exists():
            if sha256(raw) != expected:
                raise SystemExit("Existing raw ledger differs; refusing to overwrite it")
            print("Raw ledger already matches the released checksum")
            return
        with tempfile.NamedTemporaryFile(dir=raw.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with gzip.open(archive, "rb") as source:
                shutil.copyfileobj(source, stream)
        if sha256(temporary) != expected:
            raise SystemExit(f"Restored checksum mismatch; diagnostic file retained: {temporary}")
        os.replace(temporary, raw)
        print(f"Restored {raw.name} with exact release checksum")
        return
    if sha256(raw) != expected:
        raise SystemExit("Raw ledger checksum differs from the completed run manifest")
    with tempfile.NamedTemporaryFile(dir=raw.parent, delete=False) as stream:
        temporary = Path(stream.name)
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=6, mtime=0) as packed:
            with raw.open("rb") as source:
                shutil.copyfileobj(source, packed)
    digest = hashlib.sha256()
    with gzip.open(temporary, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Archive round-trip mismatch; diagnostic file retained: {temporary}")
    os.replace(temporary, archive)
    record = {"raw_path": str(raw.relative_to(HERE)), "raw_sha256": expected,
              "raw_bytes": raw.stat().st_size, "archive_path": str(archive.relative_to(HERE)),
              "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
              "round_trip_exact": True, "compression": "gzip, level 6, timestamp 0"}
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
