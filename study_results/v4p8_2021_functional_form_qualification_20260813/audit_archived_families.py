#!/usr/bin/env python3
"""Recompute the archived ROOT-family rejection ledger from audit products."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "derived/archived_root_family_edge_audit.csv"
DEFAULT_PATTERN = "/tmp/v4p8_common_family_*_*.root"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebin(values: np.ndarray, factor: int = 5) -> np.ndarray:
    usable = values.size // factor * factor
    return values[:usable].reshape(-1, factor).sum(axis=1)


def metrics(observed: np.ndarray, expected: np.ndarray, n_parameters: int) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if np.any(~np.isfinite(expected)) or np.any(expected <= 0):
        return {
            "pearson_chi2ndf": math.inf,
            "poisson_deviance_ndf": math.inf,
            "max_abs_pearson_residual": math.inf,
            "ndf": max(1, observed.size - n_parameters),
        }
    ndf = max(1, observed.size - n_parameters)
    pearson = float(np.sum((observed - expected) ** 2 / expected) / ndf)
    term = np.where(
        observed > 0,
        observed * np.log(observed / expected) - (observed - expected),
        expected,
    )
    return {
        "pearson_chi2ndf": pearson,
        "poisson_deviance_ndf": float(2.0 * np.sum(term) / ndf),
        "max_abs_pearson_residual": float(
            np.max(np.abs(observed - expected) / np.sqrt(expected))
        ),
        "ndf": int(ndf),
    }


def normalized_bound_distance(parameters: list[dict[str, Any]]) -> float:
    distances: list[float] = []
    for parameter in parameters:
        if bool(parameter.get("fixed")):
            continue
        low = float(parameter["min"])
        high = float(parameter["max"])
        value = float(parameter["value"])
        if not all(math.isfinite(item) for item in (low, high, value)) or high <= low:
            continue
        distances.append(min((value - low) / (high - low), (high - value) / (high - low)))
    return float(min(distances)) if distances else math.nan


def source_and_edge(path: Path) -> tuple[str, int]:
    stem = path.name.removesuffix(".root")
    prefix = "v4p8_common_family_"
    if not stem.startswith(prefix):
        raise RuntimeError(f"unexpected input name: {path}")
    source, edge = stem[len(prefix):].rsplit("_", 1)
    return source, int(edge)


def build_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(glob.glob(pattern)):
        root_path = Path(value)
        metadata_path = Path(str(root_path) + ".metadata.json")
        if not metadata_path.is_file():
            raise RuntimeError(f"missing metadata: {metadata_path}")
        source, edge_mev = source_and_edge(root_path)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        with uproot.open(root_path) as root_file:
            observed, edges = root_file["input_hist"].to_numpy(flow=False)
            observed = np.asarray(observed, dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for fit in payload["fits"]:
                tag = str(fit["tag"])
                expected = np.asarray(
                    root_file[f"{tag}/{tag}_analytic_seed_lumi_scaled"].values(),
                    dtype=float,
                )
                low = float(fit["fit_min_GeV"])
                high = float(fit["fit_max_GeV"])
                mask = (centers >= low) & (centers < high)
                n_parameters = sum(
                    not bool(parameter.get("fixed"))
                    for parameter in fit["parameters"]
                )
                native = metrics(observed[mask], expected[mask], n_parameters)
                factor5 = metrics(
                    rebin(observed[mask]), rebin(expected[mask]), n_parameters
                )
                bound_distance = normalized_bound_distance(fit["parameters"])
                audit_gate = bool(
                    fit["fit_ok"]
                    and native["pearson_chi2ndf"] <= 1.5
                    and native["poisson_deviance_ndf"] <= 1.5
                    and factor5["pearson_chi2ndf"] <= 2.0
                    and factor5["poisson_deviance_ndf"] <= 2.0
                    and factor5["max_abs_pearson_residual"] <= 5.0
                    and bound_distance >= 1e-4
                )
                rows.append(
                    {
                        "source": source,
                        "lower_edge_MeV": edge_mev,
                        "family": tag,
                        "fit_ok": bool(fit["fit_ok"]),
                        "native_pearson_chi2ndf": native["pearson_chi2ndf"],
                        "native_poisson_deviance_ndf": native["poisson_deviance_ndf"],
                        "rebin5_pearson_chi2ndf": factor5["pearson_chi2ndf"],
                        "rebin5_poisson_deviance_ndf": factor5["poisson_deviance_ndf"],
                        "rebin5_max_abs_pearson_residual": factor5["max_abs_pearson_residual"],
                        "minimum_normalized_free_parameter_bound_distance": bound_distance,
                        "aggregate_validation_pass": bool(fit["validation"]["selection_pass"]),
                        "audit_gate_pass": audit_gate,
                        "root_path_at_audit": str(root_path),
                        "root_sha256": sha256_file(root_path),
                        "metadata_path_at_audit": str(metadata_path),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                )
    if not rows:
        raise RuntimeError(f"no ROOT products matched {pattern!r}")
    return rows


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = build_rows(str(args.pattern))
    atomic_csv(args.output, rows)
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "gate_pass_rows": sum(bool(row["audit_gate_pass"]) for row in rows),
        "families": sorted({str(row["family"]) for row in rows}),
        "sources": sorted({str(row["source"]) for row in rows}),
        "edges_MeV": sorted({int(row["lower_edge_MeV"]) for row in rows}),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
