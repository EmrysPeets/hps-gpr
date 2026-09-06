#!/usr/bin/env python3
"""Build hash-locked 100-background continuations for the two 65 MeV lanes.

The analytic means and initial pseudoexperiments are read from their immutable
v4.9 and v4.7.1 products.  Toy seeds use the original namespace and base seed.
The builder regenerates indices 0--99 and fails unless the original 0--24 or
0--19 count arrays are bit-identical to the corresponding archived toys.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
REFERENCE = HERE / "reference"
QA = HERE / "qa"
N_TOTAL = 100


LANES: dict[str, dict[str, Any]] = {
    "native10_fsig": {
        "canonical_root": REFERENCE / "v4p9_fsig_anchor_background_toys_25.root",
        "canonical_root_sha256": "7bbf39aae4891c66e01492f5b8d41c35c04ff6873215d2fda4fdc5b066eb3fd1",
        "canonical_manifest": REFERENCE / "v4p9_fsig_anchor_background_toys_25.manifest.json",
        "canonical_manifest_sha256": "cf43b731e4d48240843eb07ccbfb575ac3989013d45b1684434ceabf0f16cdbf",
        "canonical_fit_summary": REFERENCE / "v4p9_fsig_anchor_fit_summary.json",
        "canonical_fit_summary_sha256": "5b4a1717ff0820f16f6dbb14f6f422591bbc26cc0dbafdb9ee498f784d69cdbf",
        "scenario": "2021_10pct",
        "source_family": "ten_pct",
        "exposure_multiplier": 1,
        "initial_n": 25,
        "base_seed": 20260817,
        "seed_namespace": "fsig_anchor_poisson",
        "container": "toys/fsig_anchor",
        "truth_model": "fSigPowExpQ_anchored_logistic_chebyshev6_C2_stress_truth",
        "analytic_mean_key": "truth/fsig_anchor/2021_10pct_mean",
        "baseline_analytic_mean_key": "truth/baseline_fSigPowExpQ/2021_10pct_mean",
        "output_root": INPUTS / "native10_fsig_background_toys_100.root",
        "output_manifest": INPUTS / "native10_fsig_background_toys_100.manifest.json",
        "expected_mean_total": 141321937,
    },
    "onepctx10_table17": {
        "canonical_root": REFERENCE / "v4p7p1_near_threshold_background_toys_20.root",
        "canonical_root_sha256": "84e43c9c5c9724dc9ae1d37cf1102453f3d4069ad2c11c058b5cbb6f76d776a7",
        "canonical_manifest": REFERENCE / "v4p7p1_near_threshold_background_toys_20.manifest.json",
        "canonical_manifest_sha256": "a492ddbf5fa42541498315cf7b4bc6be44eb5f20b508b396935a03b7d63af292",
        "canonical_fit_summary": REFERENCE / "v4p7p1_near_threshold_fit_summary.json",
        "canonical_fit_summary_sha256": "3a055946c76842371ceb490369cd95836b114a5e86b94e94f391b362177f000e",
        "scenario": "2021_1pct_x10",
        "source_family": "one_pct",
        "exposure_multiplier": 10,
        "initial_n": 20,
        "base_seed": 20260812,
        "seed_namespace": "near_threshold_poisson",
        "container": "toys/near_threshold",
        "truth_model": "near_threshold_logistic_chebyshev5_c2_tail",
        "analytic_mean_key": "truth/near_threshold/2021_1pct_x10_mean",
        "baseline_analytic_mean_key": "truth/baseline_fGenGammaThresh/2021_1pct_x10_mean",
        "output_root": INPUTS / "onepctx10_table17_background_toys_100.root",
        "output_manifest": INPUTS / "onepctx10_table17_background_toys_100.manifest.json",
        "expected_mean_total": 125040440,
    },
}


class ContinuationBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_seed_words(base_seed: int, namespace: str, *parts: object) -> list[int]:
    material = "|".join([str(base_seed), namespace, *map(str, parts)]).encode("utf-8")
    raw = hashlib.sha256(material).digest()[:16]
    return [
        int.from_bytes(raw[index:index + 4], "little")
        for index in range(0, 16, 4)
    ]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_canonical(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("canonical_root", "canonical_manifest", "canonical_fit_summary"):
        path = record[key]
        expected = record[f"{key}_sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise ContinuationBuildError(f"canonical hash mismatch: {path}")
    manifest = json.loads(record["canonical_manifest"].read_text())
    content = dict(manifest)
    recorded = content.pop("manifest_content_sha256", None)
    if not isinstance(recorded, str) or canonical_sha256(content) != recorded:
        raise ContinuationBuildError("canonical manifest content hash mismatch")
    return manifest


def build_lane(name: str, record: dict[str, Any]) -> dict[str, Any]:
    canonical_manifest = validate_canonical(record)
    scenario = record["scenario"]
    initial_n = int(record["initial_n"])
    canonical_toys = {
        (row["scenario"], int(row["toy_index"])): row
        for row in canonical_manifest["toys"]
    }
    output_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    toy_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    with uproot.open(record["canonical_root"]) as source:
        mean, edges = source[record["analytic_mean_key"]].to_numpy()
        baseline, baseline_edges = source[record["baseline_analytic_mean_key"]].to_numpy()
        mean = np.asarray(mean, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        if not np.array_equal(edges, np.asarray(baseline_edges, dtype=np.float64)):
            raise ContinuationBuildError(f"baseline edge mismatch for {name}")
        if not np.isclose(mean.sum(), record["expected_mean_total"], rtol=0.0, atol=1e-6):
            raise ContinuationBuildError(f"analytic-mean normalization mismatch for {name}")
        output_payload[record["analytic_mean_key"]] = (mean, edges)
        output_payload[record["baseline_analytic_mean_key"]] = (baseline, edges)

        for toy_index in range(N_TOTAL):
            seed_words = stable_seed_words(
                record["base_seed"], record["seed_namespace"], scenario, toy_index
            )
            rng = np.random.default_rng(np.random.SeedSequence(seed_words))
            counts = rng.poisson(mean).astype(np.int64)
            key = f"{record['container']}/{scenario}/toy_{toy_index:04d}"
            cohort = "initial_reproduced" if toy_index < initial_n else "independent_continuation"
            if toy_index < initial_n:
                archived, archived_edges = source[key].to_numpy()
                archived = np.rint(np.asarray(archived)).astype(np.int64)
                identical = bool(
                    np.array_equal(counts, archived)
                    and np.array_equal(edges, np.asarray(archived_edges, dtype=np.float64))
                )
                manifest_row = canonical_toys.get((scenario, toy_index))
                manifest_hash_ok = bool(
                    manifest_row
                    and manifest_row.get("counts_sha256") == array_sha256(counts, "<i8")
                )
                if not identical or not manifest_hash_ok:
                    raise ContinuationBuildError(
                        f"initial toy identity failed for {name} index {toy_index}"
                    )
                identity_rows.append({
                    "toy_index": toy_index,
                    "array_identical": identical,
                    "canonical_manifest_hash_identical": manifest_hash_ok,
                    "counts_sha256": array_sha256(counts, "<i8"),
                })
            output_payload[key] = (counts, edges)
            toy_rows.append({
                "source_family": record["source_family"],
                "scenario": scenario,
                "exposure_multiplier": int(record["exposure_multiplier"]),
                "toy_index": toy_index,
                "cohort": cohort,
                "output_histogram": key,
                "seed_namespace": record["seed_namespace"],
                "seed_words": seed_words,
                "counts_sha256": array_sha256(counts, "<i8"),
                "total_count": int(counts.sum()),
                "expected_mean_total": float(mean.sum()),
            })

    record["output_root"].parent.mkdir(parents=True, exist_ok=True)
    temporary = record["output_root"].with_name(
        f".{record['output_root'].name}.{os.getpid()}.tmp"
    )
    with uproot.recreate(temporary) as output:
        for key, histogram in output_payload.items():
            output[key] = histogram
    os.replace(temporary, record["output_root"])

    truth_row = {
        "source_family": record["source_family"],
        "scenario": scenario,
        "exposure_multiplier": int(record["exposure_multiplier"]),
        "analytic_mean_key": record["analytic_mean_key"],
        "baseline_analytic_mean_key": record["baseline_analytic_mean_key"],
        "mean_sha256_float64": array_sha256(mean, "<f8"),
        "mean_total": float(mean.sum()),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study_id": HERE.name,
        "lane": name,
        "generation": "exact original analytic mean and seed namespace extended to toy index 99",
        "ensemble_semantics": (
            f"{N_TOTAL} independent Poisson background spectra: indices 0--{initial_n - 1} "
            f"reproduce the accepted initial ensemble and indices {initial_n}--99 are an "
            "independently seeded continuation"
        ),
        "base_seed": int(record["base_seed"]),
        "n_toys_per_scenario": N_TOTAL,
        "scenarios": [scenario],
        "root": str(record["output_root"].relative_to(HERE)),
        "root_sha256": sha256_file(record["output_root"]),
        "toy_key_template": f"{record['container']}/{{scenario}}/toy_{{toy_index:04d}}",
        "truth_model": record["truth_model"],
        "truths": [truth_row],
        "toys": toy_rows,
        "cohorts": {
            "initial": {"start": 0, "stop_exclusive": initial_n, "n": initial_n},
            "independent_continuation": {
                "start": initial_n,
                "stop_exclusive": N_TOTAL,
                "n": N_TOTAL - initial_n,
            },
        },
        "canonical_source": {
            "root": str(record["canonical_root"].relative_to(HERE)),
            "root_sha256": record["canonical_root_sha256"],
            "manifest": str(record["canonical_manifest"].relative_to(HERE)),
            "manifest_sha256": record["canonical_manifest_sha256"],
            "fit_summary": str(record["canonical_fit_summary"].relative_to(HERE)),
            "fit_summary_sha256": record["canonical_fit_summary_sha256"],
        },
        "initial_identity": {
            "required": initial_n,
            "passed": len(identity_rows),
            "all_arrays_and_manifest_hashes_identical": True,
            "rows": identity_rows,
        },
    }
    content = dict(manifest)
    manifest["manifest_content_sha256"] = canonical_sha256(content)
    atomic_json(record["output_manifest"], manifest)
    return {
        "status": "pass",
        "lane": name,
        "scenario": scenario,
        "n_total": N_TOTAL,
        "initial_identity_n": len(identity_rows),
        "continuation_n": N_TOTAL - initial_n,
        "root": str(record["output_root"].relative_to(HERE)),
        "root_sha256": sha256_file(record["output_root"]),
        "manifest": str(record["output_manifest"].relative_to(HERE)),
        "manifest_sha256": sha256_file(record["output_manifest"]),
    }


def main() -> int:
    results = [build_lane(name, dict(record)) for name, record in LANES.items()]
    report = {
        "schema_version": 1,
        "status": "pass",
        "study_id": HERE.name,
        "n_total_per_lane": N_TOTAL,
        "lanes": results,
    }
    atomic_json(QA / "toy_extension_identity.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
