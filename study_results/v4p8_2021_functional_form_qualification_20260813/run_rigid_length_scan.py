#!/usr/bin/env python3
"""Immutable launcher for the hash-locked v4.8 rigid length scan.

The one-way trust chain is launcher -> external lock -> core and scientific
inputs.  The lock intentionally does not hash this launcher, so there is no
cyclic self-hash.  The launcher reads the core once, verifies those bytes, and
executes exactly the bytes that were verified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "rigid_length_scan_lock.json"
EXPECTED_LOCK_SHA256 = (
    "6e936c649dd2aada712809f54474ee78df5d7bd3ad81c4c7e3a2f97f3509e30f"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    lock_bytes = LOCK_PATH.read_bytes()
    actual_lock_hash = sha256_bytes(lock_bytes)
    if actual_lock_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError(
            "rigid length-scan lock drift: "
            f"expected {EXPECTED_LOCK_SHA256}, found {actual_lock_hash}"
        )
    lock = json.loads(lock_bytes.decode("utf-8"))
    if lock.get("lock_type") != "immutable_launcher_to_core_v1":
        raise RuntimeError("unsupported rigid length-scan lock type")
    driver = lock.get("driver", {})
    core_path = (HERE / str(driver.get("path", ""))).resolve()
    core_bytes = core_path.read_bytes()
    actual_core_hash = sha256_bytes(core_bytes)
    expected_core_hash = str(driver.get("sha256", ""))
    if actual_core_hash != expected_core_hash:
        raise RuntimeError(
            "rigid length-scan core drift: "
            f"expected {expected_core_hash}, found {actual_core_hash}"
        )
    launcher_path = Path(__file__).resolve()
    launcher_hash = sha256_bytes(launcher_path.read_bytes())
    namespace = {
        "__name__": "__main__",
        "__file__": str(core_path),
        "__package__": None,
        "__LENGTH_SCAN_EXECUTED_CORE_SHA256__": actual_core_hash,
        "__LENGTH_SCAN_EXTERNAL_LOCK_SHA256__": actual_lock_hash,
        "__LENGTH_SCAN_LAUNCHER_SHA256__": launcher_hash,
        "__LENGTH_SCAN_LAUNCHER_PATH__": str(launcher_path),
    }
    executable = compile(core_bytes, str(core_path), "exec")
    exec(executable, namespace, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
