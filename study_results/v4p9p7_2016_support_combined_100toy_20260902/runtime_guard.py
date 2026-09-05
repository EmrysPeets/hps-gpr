#!/usr/bin/env python3
"""Activate and attest the complete campaign-local hps_gpr runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable


HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE / "runtime_combined"
PACKAGE_ROOT = RUNTIME_ROOT / "hps_gpr"
MANIFEST = RUNTIME_ROOT / "runtime_manifest.json"
EXPECTED_MANIFEST_SHA256 = (
    "8d20a7f44db25574e20114171d6f014bc116546d0b97c189454d9fceacec767c"
)
EXPECTED_SOURCE_COMMIT = "e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6"


class RuntimeContractError(RuntimeError):
    """Raised if the runtime snapshot or actual module origins differ."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activate_and_verify() -> Dict[str, object]:
    if not MANIFEST.is_file():
        raise RuntimeContractError(f"Missing runtime manifest: {MANIFEST}")
    digest = sha256(MANIFEST)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeContractError(
            f"Runtime manifest SHA-256 {digest} != {EXPECTED_MANIFEST_SHA256}"
        )
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeContractError("Runtime manifest names another source commit.")
    declared = payload.get("package_files", {})
    if int(payload.get("package_file_count", -1)) != len(declared):
        raise RuntimeContractError("Runtime manifest file count is inconsistent.")
    if "hps_gpr/__init__.py" not in declared:
        raise RuntimeContractError("Runtime snapshot is not an importable package.")

    actual = {
        path.relative_to(RUNTIME_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
    }
    if actual != set(declared):
        raise RuntimeContractError(
            "Runtime Python file set differs from its manifest: "
            f"extra={sorted(actual - set(declared))}, "
            f"missing={sorted(set(declared) - actual)}"
        )
    for relative, expected in sorted(declared.items()):
        path = RUNTIME_ROOT / relative
        found = sha256(path)
        if found != expected:
            raise RuntimeContractError(
                f"Runtime module {relative} SHA-256 {found} != {expected}"
            )

    runtime_text = str(RUNTIME_ROOT.resolve())
    sys.path[:] = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()) != runtime_text
    ]
    sys.path.insert(0, runtime_text)
    return {
        "runtime_root": runtime_text,
        "runtime_manifest": str(MANIFEST.resolve()),
        "runtime_manifest_sha256": digest,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "package_file_count": len(declared),
        "package_files": dict(declared),
    }


def assert_import_origins(module_names: Iterable[str]) -> Dict[str, str]:
    """Require every named imported module to resolve inside the snapshot."""

    origins: Dict[str, str] = {}
    package_root = PACKAGE_ROOT.resolve()
    for name in module_names:
        module = importlib.import_module(name)
        raw = getattr(module, "__file__", None)
        if raw is None:
            raise RuntimeContractError(f"Imported module {name} has no file origin.")
        origin = Path(raw).resolve()
        try:
            origin.relative_to(package_root)
        except ValueError as error:
            raise RuntimeContractError(
                f"Imported module {name} came from {origin}, not {package_root}."
            ) from error
        origins[name] = str(origin)
    return origins
