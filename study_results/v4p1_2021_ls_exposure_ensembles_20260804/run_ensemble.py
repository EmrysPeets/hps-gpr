#!/usr/bin/env python3
"""Resumable paired-exposure length-scale pilot for HPS 2021.

The driver deliberately separates preparation, scan execution, injection
execution, and collection.  Merely invoking the script never starts a fit.
Production work requires the explicit ``--execute`` flag.

All fit imports are routed to the immutable checkout pinned in
``study_spec.json``.  The active development checkout is used only for the
frozen v4 configuration and this study's inputs/outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[1]
SPEC_PATH = STUDY_DIR / "study_spec.json"
TOY_ROOT_PATH = STUDY_DIR / "inputs" / "paired_exposure_toys.root"
TOY_MANIFEST_PATH = STUDY_DIR / "derived" / "paired_exposure_toy_manifest.json"
TASKS_PATH = STUDY_DIR / "derived" / "task_manifest.jsonl"
INJECTION_ANCHOR_DIR = STUDY_DIR / "derived" / "injection_anchors"
INJECTION_ANCHOR_PARTS_DIR = INJECTION_ANCHOR_DIR / "parts"
INJECTION_ANCHOR_LEDGER_PATH = (
    INJECTION_ANCHOR_DIR / "factor15_prefit_asimov_absolute_v1.json"
)
INJECTION_PROTOCOL = "factor15_prefit_asimov_absolute_v1"


class StudyError(RuntimeError):
    """Fail-closed study validation error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_spec() -> Dict[str, Any]:
    spec = _load_json(SPEC_PATH)
    if int(spec.get("schema_version", -1)) != 1:
        raise StudyError("Unsupported or missing study_spec schema_version")
    return spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array_int64(values: Any) -> str:
    import numpy as np

    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_json_sha(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_seed_words(spec: Mapping[str, Any], namespace: str, *parts: object) -> List[int]:
    material = "|".join(
        [str(int(spec["base_seed"])), str(namespace)] + [str(part) for part in parts]
    ).encode("utf-8")
    raw = hashlib.sha256(material).digest()[:16]
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, 16, 4)]


def _stable_seed32(spec: Mapping[str, Any], namespace: str, *parts: object) -> int:
    words = _stable_seed_words(spec, namespace, *parts)
    # sklearn accepts a legacy RandomState seed in [0, 2**32 - 1).
    return int(words[0] % (2**32 - 1))


def _mass_seed(
    spec: Mapping[str, Any], truth_model: str, scenario: str, toy_index: int, mass_gev: float
) -> int:
    mass_kev = int(round(float(mass_gev) * 1_000_000.0))
    # Deliberately independent of the upper-bound factor: paired candidates get
    # the same restart quantiles for the same pseudo-data and mass.
    return _stable_seed32(
        spec, "optimizer", truth_model, scenario, int(toy_index), mass_kev
    )


def _injection_seed(
    spec: Mapping[str, Any], truth_model: str, scenario: str, toy_index: int
) -> int:
    # Deliberately independent of the upper-bound factor so the signal RNG is
    # common across candidate bounds.  The hps_gpr helper adds toy_index.
    base = _stable_seed32(spec, "injection", truth_model, scenario)
    return int((base + int(toy_index)) % (2**32 - 1))


def _injection_mass_optimizer_seed(
    spec: Mapping[str, Any],
    truth_model: str,
    scenario: str,
    toy_index: int,
    mass_gev: float,
) -> int:
    """Pair injection optimizer starts across factors, independently by mass."""

    mass_kev = int(round(float(mass_gev) * 1_000_000.0))
    return _stable_seed32(
        spec,
        "injection_mass_optimizer",
        truth_model,
        scenario,
        int(toy_index),
        mass_kev,
    )


def _injection_protocol_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    closure = dict(spec["injection_closure"])
    protocol = str(closure.get("protocol", ""))
    if protocol != INJECTION_PROTOCOL:
        raise StudyError(
            f"Injection protocol mismatch: expected {INJECTION_PROTOCOL}, got {protocol!r}"
        )
    anchor_factor = int(closure.get("anchor_factor", -1))
    if anchor_factor != 15:
        raise StudyError(f"Injection anchor factor must be 15, got {anchor_factor}")
    if str(closure.get("anchor_sigma_a_source", "")).lower() != "asimov":
        raise StudyError("Injection anchor sigma-A source must be 'asimov'")
    if (
        str(closure.get("anchor_sigma_a_ref_mode", "")).lower()
        != "prefit_asimov"
    ):
        raise StudyError(
            "Injection anchor sigma-A reference mode must be 'prefit_asimov'"
        )
    if int(closure.get("replicas_per_background_toy", -1)) != 1:
        raise StudyError(
            "Signal-draw identity validation currently requires exactly one "
            "injection replica per background toy"
        )
    return closure


def _anchor_mass_key(mass_gev: float) -> str:
    return f"{float(mass_gev):.9f}"


def _anchor_entry_key(
    truth_model: str, scenario: str, toy_index: int, mass_gev: float
) -> Tuple[str, str, int, str]:
    return (
        str(truth_model),
        str(scenario),
        int(toy_index),
        _anchor_mass_key(float(mass_gev)),
    )


def _anchor_identity_id(
    truth_model: str, scenario: str, toy_index: int
) -> str:
    return (
        f"anchor__{str(truth_model)}__{str(scenario)}__"
        f"t{int(toy_index):04d}"
    )


def build_anchor_identities(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    _injection_protocol_spec(spec)
    identities: List[Dict[str, Any]] = []
    for truth_model in sorted(spec["truth_models"]):
        for scenario in sorted(spec["scenarios"]):
            for toy_index in range(int(spec["n_toys"])):
                identities.append(
                    {
                        "anchor_id": _anchor_identity_id(
                            truth_model, scenario, toy_index
                        ),
                        "truth_model": str(truth_model),
                        "function_tag": str(
                            spec["truth_models"][truth_model]["function_tag"]
                        ),
                        "scenario": str(scenario),
                        "toy_index": int(toy_index),
                        "toy_container": f"toys/{truth_model}/{scenario}",
                        "toy_name": f"toy_{toy_index:04d}",
                    }
                )
    return identities


def _anchor_identity_by_id(
    spec: Mapping[str, Any], anchor_id: str
) -> Dict[str, Any]:
    matches = [
        row
        for row in build_anchor_identities(spec)
        if row["anchor_id"] == str(anchor_id)
    ]
    if len(matches) != 1:
        raise StudyError(f"Unknown or duplicate anchor id: {anchor_id}")
    return dict(matches[0])


def _anchor_part_path(identity: Mapping[str, Any]) -> Path:
    return INJECTION_ANCHOR_PARTS_DIR / f"{identity['anchor_id']}.json"


def _resolve_repo_entry(relative: str) -> Path:
    return (REPO_ROOT / relative).resolve()


def _verify_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise StudyError(f"Missing {label}: {path}")
    actual = _sha256_file(path)
    if actual != str(expected):
        raise StudyError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}: {path}"
        )


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def verify_fit_code(spec: Mapping[str, Any]) -> Dict[str, Any]:
    fit = spec["fit_code"]
    fit_repo = Path(fit["repo"]).resolve()
    if not fit_repo.is_dir():
        raise StudyError(f"Missing immutable fit-code checkout: {fit_repo}")

    head = _git_output(fit_repo, "rev-parse", "HEAD")
    if head != fit["commit"]:
        raise StudyError(
            f"Fit-code commit drift: expected {fit['commit']}, got {head}"
        )

    tracked_status = _git_output(fit_repo, "status", "--porcelain", "--", "hps_gpr")
    if tracked_status:
        raise StudyError(
            "Immutable fit-code hps_gpr tree is dirty:\n" + tracked_status
        )

    module_rows: List[Dict[str, str]] = []
    for relative, expected in sorted(fit["module_sha256"].items()):
        path = fit_repo / relative
        _verify_sha(path, expected, f"fit module {relative}")
        module_rows.append(
            {"path": relative, "sha256": expected, "absolute_path": str(path)}
        )
    return {
        "repo": str(fit_repo),
        "commit": head,
        "modules": module_rows,
    }


def _activate_fit_code(spec: Mapping[str, Any]) -> None:
    """Route all hps_gpr imports to the pinned checkout, never this worktree."""

    already = [name for name in sys.modules if name == "hps_gpr" or name.startswith("hps_gpr.")]
    if already:
        raise StudyError(
            "hps_gpr was imported before immutable routing: " + ", ".join(sorted(already))
        )

    fit_repo = Path(spec["fit_code"]["repo"]).resolve()
    active_repo = REPO_ROOT.resolve()

    def resolved_entry(entry: str) -> Optional[Path]:
        try:
            return (Path.cwd() if entry == "" else Path(entry)).resolve()
        except Exception:
            return None

    filtered = [
        entry
        for entry in sys.path
        if resolved_entry(entry) not in {fit_repo, active_repo}
    ]
    sys.path[:] = [str(fit_repo)] + filtered

    import hps_gpr

    imported = Path(hps_gpr.__file__).resolve()
    try:
        imported.relative_to(fit_repo)
    except ValueError as exc:
        raise StudyError(
            f"hps_gpr import escaped immutable checkout: {imported}"
        ) from exc


def _configure_fit_process() -> None:
    mpl = (
        Path(tempfile.gettempdir())
        / f"{load_spec()['study_id']}_mplconfig"
    )
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def verify_base_config(spec: Mapping[str, Any]) -> Path:
    entry = spec["base_config"]
    path = _resolve_repo_entry(entry["path_from_repo"])
    _verify_sha(path, entry["sha256"], "reviewed 2021 k15 observed-only base config")
    return path


def verify_frozen_factor_configs(
    spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    factors = [int(value) for value in spec["length_scale_upper_factors"]]
    expected_keys = {str(value) for value in factors}
    actual_keys = set(
        (spec.get("frozen_generated_config_sha256_by_factor") or {}).keys()
    )
    if actual_keys != expected_keys:
        raise StudyError(
            "Frozen generated-config factor set drift: "
            f"{sorted(actual_keys)} != {sorted(expected_keys)}"
        )
    return [
        {
            "factor": factor,
            "path": str(_verify_frozen_factor_config(spec, factor)),
            "sha256": spec["frozen_generated_config_sha256_by_factor"][
                str(factor)
            ],
        }
        for factor in factors
    ]


def _source_paths(
    spec: Mapping[str, Any], family: str
) -> Tuple[Path, Path, Mapping[str, Any]]:
    entry = spec["source_families"][family]
    root_path = (STUDY_DIR / entry["root_path_from_study"]).resolve()
    metadata_path = (STUDY_DIR / entry["metadata_path_from_study"]).resolve()
    return root_path, metadata_path, entry


def verify_sources(spec: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import uproot

    support = tuple(float(x) for x in spec["support_range_gev"])
    scan = tuple(float(x) for x in spec["scan_range_gev"])
    report: Dict[str, Any] = {}

    for family in sorted(spec["source_families"]):
        root_path, metadata_path, entry = _source_paths(spec, family)
        _verify_sha(root_path, entry["root_sha256"], f"{family} source ROOT")
        _verify_sha(
            metadata_path, entry["metadata_sha256"], f"{family} source metadata"
        )
        metadata = _load_json(metadata_path)
        if tuple(metadata.get("toy_support_range_GeV", [])) != support:
            raise StudyError(f"{family} metadata support range drift")
        if tuple(metadata.get("scan_range_GeV", [])) != scan:
            raise StudyError(f"{family} metadata scan range drift")
        if not bool(metadata.get("primary_validation_pass", False)):
            raise StudyError(f"{family} metadata primary validation did not pass")
        if int(metadata.get("normalization_target_count", -1)) != int(
            entry["normalization_target_count"]
        ):
            raise StudyError(f"{family} normalization target drift")

        fits = {str(row["tag"]): row for row in metadata.get("fits", [])}
        model_report: Dict[str, Any] = {}
        with uproot.open(root_path) as root_file:
            for truth_name, truth in spec["truth_models"].items():
                tag = str(truth["function_tag"])
                if tag not in fits:
                    raise StudyError(f"{family} metadata is missing truth {tag}")
                fit_validation = fits[tag].get("validation", {})
                if not bool(fit_validation.get("selection_pass", False)):
                    raise StudyError(f"{family}/{tag} selection validation failed")

                key = f"{tag}/{tag}_analytic_seed_lumi_scaled"
                if key not in root_file:
                    raise StudyError(f"{family} source ROOT is missing {key}")
                values, edges = root_file[key].to_numpy()
                values = np.asarray(values, dtype=float)
                edges = np.asarray(edges, dtype=float)
                if values.shape != (8000,) or edges.shape != (8001,):
                    raise StudyError(
                        f"{family}/{tag} must preserve native 8000-bin geometry"
                    )
                if not np.isclose(edges[0], 0.0) or not np.isclose(edges[-1], 1.0):
                    raise StudyError(f"{family}/{tag} histogram range is not [0,1] GeV")
                centers = 0.5 * (edges[:-1] + edges[1:])
                outside = (centers < support[0]) | (centers > support[1])
                if np.any(np.abs(values[outside]) > 1e-9):
                    raise StudyError(f"{family}/{tag} has nonzero seed bins outside support")
                if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                    raise StudyError(f"{family}/{tag} seed contains invalid means")
                expected_total = float(entry["normalization_target_count"])
                if not np.isclose(float(values.sum()), expected_total, rtol=0, atol=1e-3):
                    raise StudyError(f"{family}/{tag} seed normalization drift")

                model_report[truth_name] = {
                    "function_tag": tag,
                    "root_key": key,
                    "mean_total": float(values.sum()),
                    "nonzero_bins": int(np.count_nonzero(values)),
                    "fit_ok_root_flag": bool(fits[tag].get("fit_ok", False)),
                    "pearson_chi2ndf": fit_validation.get("full_range", {}).get(
                        "pearson_chi2ndf"
                    )
                    or fits[tag].get("pearson_chi2ndf"),
                    "selection_pass": True,
                }

        report[family] = {
            "root": str(root_path),
            "root_sha256": entry["root_sha256"],
            "metadata": str(metadata_path),
            "metadata_sha256": entry["metadata_sha256"],
            "metadata_primary_function": metadata.get("primary_function"),
            "truth_models": model_report,
        }
    return report


def preflight(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "checked_utc": _utc_now(),
        "study_id": spec["study_id"],
        "base_config": {
            "path": str(verify_base_config(spec)),
            "sha256": spec["base_config"]["sha256"],
        },
        "fit_code": verify_fit_code(spec),
        "frozen_generated_configs": verify_frozen_factor_configs(spec),
        "sources": verify_sources(spec),
        "expected_limit_bands": False,
    }


def _truth_seed(
    spec: Mapping[str, Any], family: str, truth_model: str
) -> Tuple[Any, Any, Dict[str, Any]]:
    import numpy as np
    import uproot

    root_path, metadata_path, entry = _source_paths(spec, family)
    tag = spec["truth_models"][truth_model]["function_tag"]
    key = f"{tag}/{tag}_analytic_seed_lumi_scaled"
    with uproot.open(root_path) as root_file:
        values, edges = root_file[key].to_numpy()
    provenance = {
        "source_family": family,
        "source_root": str(root_path),
        "source_root_sha256": entry["root_sha256"],
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": entry["metadata_sha256"],
        "truth_model": truth_model,
        "function_tag": tag,
        "source_histogram": key,
    }
    return (
        np.asarray(values, dtype=float),
        np.asarray(edges, dtype=float),
        provenance,
    )


def _draw_increment(
    spec: Mapping[str, Any],
    mean: Any,
    truth_model: str,
    family: str,
    toy_index: int,
    stage: str,
    multiplier: int,
) -> Tuple[Any, List[int]]:
    import numpy as np

    words = _stable_seed_words(
        spec, "poisson_increment", truth_model, family, int(toy_index), stage
    )
    rng = np.random.default_rng(np.random.SeedSequence(words))
    draw = rng.poisson(np.asarray(mean, dtype=float) * int(multiplier)).astype(
        np.int64
    )
    return draw, words


def draw_nested_family(
    spec: Mapping[str, Any],
    mean: Any,
    truth_model: str,
    family: str,
    toy_index: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Draw one source family using independent nonnegative increments."""

    if family == "one_pct":
        base_name = "2021_1pct"
        x10_name = "2021_1pct_x10"
        x100_name = "2021_1pct_x100"
        base, seed_base = _draw_increment(
            spec, mean, truth_model, family, toy_index, "base_1x", 1
        )
        plus9, seed_plus9 = _draw_increment(
            spec, mean, truth_model, family, toy_index, "increment_9x", 9
        )
        plus90, seed_plus90 = _draw_increment(
            spec, mean, truth_model, family, toy_index, "increment_90x", 90
        )
        arrays = {
            base_name: base,
            x10_name: base + plus9,
            x100_name: base + plus9 + plus90,
        }
        meta = {
            base_name: {
                "parent": None,
                "increment_multiplier": 1,
                "increment_seed_words": seed_base,
                "increment_sha256": _sha256_array_int64(base),
            },
            x10_name: {
                "parent": base_name,
                "increment_multiplier": 9,
                "increment_seed_words": seed_plus9,
                "increment_sha256": _sha256_array_int64(plus9),
            },
            x100_name: {
                "parent": x10_name,
                "increment_multiplier": 90,
                "increment_seed_words": seed_plus90,
                "increment_sha256": _sha256_array_int64(plus90),
            },
        }
        return arrays, meta

    if family == "ten_pct":
        base_name = "2021_10pct"
        x10_name = "2021_10pct_x10"
        base, seed_base = _draw_increment(
            spec, mean, truth_model, family, toy_index, "base_1x", 1
        )
        plus9, seed_plus9 = _draw_increment(
            spec, mean, truth_model, family, toy_index, "increment_9x", 9
        )
        arrays = {base_name: base, x10_name: base + plus9}
        meta = {
            base_name: {
                "parent": None,
                "increment_multiplier": 1,
                "increment_seed_words": seed_base,
                "increment_sha256": _sha256_array_int64(base),
            },
            x10_name: {
                "parent": base_name,
                "increment_multiplier": 9,
                "increment_seed_words": seed_plus9,
                "increment_sha256": _sha256_array_int64(plus9),
            },
        }
        return arrays, meta
    raise StudyError(f"Unsupported source family: {family}")


def _write_toy_root_and_manifest(
    spec: Mapping[str, Any], force: bool = False
) -> Dict[str, Any]:
    import numpy as np
    import uproot

    if (TOY_ROOT_PATH.exists() or TOY_MANIFEST_PATH.exists()) and not force:
        if TOY_ROOT_PATH.exists() and TOY_MANIFEST_PATH.exists():
            validate_toys(spec)
            return _load_json(TOY_MANIFEST_PATH)
        raise StudyError(
            "Only one paired-toy artifact exists; inspect it and rerun prepare --force"
        )

    TOY_ROOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOY_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = TOY_ROOT_PATH.with_name(
        f".{TOY_ROOT_PATH.stem}.{os.getpid()}.tmp.root"
    )
    if temporary_root.exists():
        raise StudyError(f"Temporary ROOT path already exists: {temporary_root}")

    rows: List[Dict[str, Any]] = []
    truth_rows: List[Dict[str, Any]] = []
    try:
        with uproot.recreate(temporary_root) as output:
            for truth_model in sorted(spec["truth_models"]):
                for family in ("one_pct", "ten_pct"):
                    mean, edges, provenance = _truth_seed(
                        spec, family, truth_model
                    )
                    truth_key = f"truth/{truth_model}/{family}_mean"
                    output[truth_key] = (mean, edges)
                    truth_rows.append(
                        {
                            **provenance,
                            "output_histogram": truth_key,
                            "mean_total": float(mean.sum()),
                            "mean_sha256_float64": hashlib.sha256(
                                np.asarray(mean, dtype="<f8").tobytes(order="C")
                            ).hexdigest(),
                        }
                    )

                    for toy_index in range(int(spec["n_toys"])):
                        arrays, seed_meta = draw_nested_family(
                            spec, mean, truth_model, family, toy_index
                        )
                        for scenario, counts in arrays.items():
                            key = (
                                f"toys/{truth_model}/{scenario}/"
                                f"toy_{int(toy_index):04d}"
                            )
                            output[key] = (counts, edges)
                            scenario_spec = spec["scenarios"][scenario]
                            parent = seed_meta[scenario]["parent"]
                            rows.append(
                                {
                                    "truth_model": truth_model,
                                    "function_tag": provenance["function_tag"],
                                    "source_family": family,
                                    "scenario": scenario,
                                    "toy_index": int(toy_index),
                                    "output_histogram": key,
                                    "parent_scenario": parent,
                                    "parent_output_histogram": (
                                        None
                                        if parent is None
                                        else (
                                            f"toys/{truth_model}/{parent}/"
                                            f"toy_{int(toy_index):04d}"
                                        )
                                    ),
                                    "exposure_multiplier": int(
                                        scenario_spec["exposure_multiplier"]
                                    ),
                                    "increment_multiplier": int(
                                        seed_meta[scenario]["increment_multiplier"]
                                    ),
                                    "increment_seed_words": list(
                                        seed_meta[scenario]["increment_seed_words"]
                                    ),
                                    "increment_sha256": seed_meta[scenario][
                                        "increment_sha256"
                                    ],
                                    "counts_sha256": _sha256_array_int64(counts),
                                    "total_count": int(np.asarray(counts).sum()),
                                    "expected_mean_total": float(
                                        mean.sum()
                                        * int(scenario_spec["exposure_multiplier"])
                                    ),
                                    "source_root": provenance["source_root"],
                                    "source_root_sha256": provenance[
                                        "source_root_sha256"
                                    ],
                                    "source_histogram": provenance[
                                        "source_histogram"
                                    ],
                                }
                            )
        os.replace(temporary_root, TOY_ROOT_PATH)
    except Exception:
        try:
            temporary_root.unlink()
        except FileNotFoundError:
            pass
        raise

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "created_utc": _utc_now(),
        "generation": "nested independent-increment Poisson",
        "base_seed": int(spec["base_seed"]),
        "n_toys": int(spec["n_toys"]),
        "toy_root": str(TOY_ROOT_PATH),
        "toy_root_sha256": _sha256_file(TOY_ROOT_PATH),
        "support_range_gev": spec["support_range_gev"],
        "scan_range_gev": spec["scan_range_gev"],
        "truth_seeds": truth_rows,
        "toys": rows,
    }
    manifest["manifest_content_sha256"] = _canonical_json_sha(manifest)
    _atomic_write_json(TOY_MANIFEST_PATH, manifest)
    validate_toys(spec)
    return manifest


def validate_toys(spec: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import uproot

    if not TOY_ROOT_PATH.is_file() or not TOY_MANIFEST_PATH.is_file():
        raise StudyError("Paired toy ROOT/manifest are not prepared")
    manifest = _load_json(TOY_MANIFEST_PATH)
    expected_content = dict(manifest)
    recorded_content_sha = expected_content.pop("manifest_content_sha256", None)
    if _canonical_json_sha(expected_content) != recorded_content_sha:
        raise StudyError("Toy manifest content SHA mismatch")
    if _sha256_file(TOY_ROOT_PATH) != manifest.get("toy_root_sha256"):
        raise StudyError("Paired toy ROOT SHA mismatch")

    expected_rows = (
        len(spec["truth_models"])
        * len(spec["scenarios"])
        * int(spec["n_toys"])
    )
    if len(manifest.get("toys", [])) != expected_rows:
        raise StudyError(
            f"Toy manifest row count mismatch: {len(manifest.get('toys', []))} "
            f"!= {expected_rows}"
        )

    rows_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    arrays: Dict[Tuple[str, str, int], Any] = {}
    with uproot.open(TOY_ROOT_PATH) as root_file:
        for row in manifest["toys"]:
            key = (
                str(row["truth_model"]),
                str(row["scenario"]),
                int(row["toy_index"]),
            )
            if key in rows_by_key:
                raise StudyError(f"Duplicate toy manifest key: {key}")
            rows_by_key[key] = row
            hist_key = str(row["output_histogram"])
            if hist_key not in root_file:
                raise StudyError(f"Toy ROOT is missing {hist_key}")
            values, edges = root_file[hist_key].to_numpy()
            if not np.allclose(values, np.rint(values), rtol=0, atol=0):
                raise StudyError(f"Toy {hist_key} is not integer-valued")
            counts = np.rint(values).astype(np.int64)
            if counts.shape != (8000,) or np.any(counts < 0):
                raise StudyError(f"Toy {hist_key} has invalid count geometry")
            if _sha256_array_int64(counts) != row["counts_sha256"]:
                raise StudyError(f"Toy {hist_key} count SHA mismatch")
            if int(counts.sum()) != int(row["total_count"]):
                raise StudyError(f"Toy {hist_key} total mismatch")
            arrays[key] = counts

    for key, row in rows_by_key.items():
        parent = row.get("parent_scenario")
        if parent is None:
            continue
        parent_key = (key[0], str(parent), key[2])
        if parent_key not in arrays:
            raise StudyError(f"Missing parent toy for {key}")
        increment = arrays[key] - arrays[parent_key]
        if np.any(increment < 0):
            raise StudyError(f"Nested increment is negative for {key}")
        if _sha256_array_int64(increment) != row["increment_sha256"]:
            raise StudyError(f"Nested increment SHA mismatch for {key}")

    return {
        "validated_utc": _utc_now(),
        "toy_root": str(TOY_ROOT_PATH),
        "toy_root_sha256": manifest["toy_root_sha256"],
        "toy_rows": len(manifest["toys"]),
        "truth_models": sorted(spec["truth_models"]),
        "scenarios": sorted(spec["scenarios"]),
        "n_toys_per_truth_scenario": int(spec["n_toys"]),
    }


def _config_overrides(
    spec: Mapping[str, Any], base: Mapping[str, Any], factor: int
) -> Dict[str, Any]:
    cfg = dict(base)
    upper = dict(cfg.get("kernel_ls_res_upper_factor_by_dataset") or {})
    upper["2021"] = float(factor)
    cfg["kernel_ls_res_upper_factor_by_dataset"] = upper

    # Explicitly keep every expected-band switch off.
    cfg["make_ul_bands"] = False
    cfg["ul_bands_toys"] = 0
    cfg["do_combined_bands"] = False
    cfg["combined_bands_n_toys"] = 0
    cfg["make_eps2_bands"] = False
    cfg["cls_mode"] = "asymptotic"
    cfg["cls_num_toys"] = 0

    # Lean task output; every fit diagnostic remains in results_single.csv.
    cfg["toy_scan_parallel"] = False
    cfg["toy_scan_n_workers"] = 1
    cfg["toy_scan_threads_per_worker"] = 1
    cfg["toy_scan_save_plots"] = False
    cfg["toy_scan_save_fit_json"] = False
    cfg["toy_scan_save_per_mass_folders"] = False

    closure = spec["injection_closure"]
    cfg["inj_dataset_key"] = "2021"
    cfg["inj_masses_gev"] = [float(x) for x in closure["masses_gev"]]
    cfg["inj_strength_mode"] = "sigmaA"
    cfg["inj_sigma_multipliers"] = [
        float(x) for x in closure["sigma_strengths"]
    ]
    cfg["inj_mode"] = closure["mode"]
    cfg["inj_background_mode"] = closure["background_mode"]
    cfg["inj_refit_gp_on_toy"] = bool(closure["refit_gp_on_toy"])
    cfg["inj_refit_gp_optimize"] = bool(closure["refit_gp_optimize"])
    cfg["inj_refit_gp_restarts"] = int(closure["refit_gp_restarts"])
    cfg["inj_refit_fail_on_error"] = False
    cfg["inj_shape_mode"] = closure["shape_mode"]
    cfg["inj_train_exclude_nsigma"] = float(
        closure["train_exclude_nsigma"]
    )
    cfg["inj_write_toy_csv"] = True
    cfg["inj_write_qmu"] = True
    cfg["inj_stream_aggregate"] = True
    cfg["inj_n_workers"] = 1
    cfg["inj_threads_per_worker"] = 1
    cfg["output_dir"] = str(
        STUDY_DIR / "runs" / "config_default" / f"f{int(factor):02d}"
    )
    return cfg


def _write_candidate_configs(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    import yaml

    base_path = verify_base_config(spec)
    with base_path.open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    if not isinstance(base, dict):
        raise StudyError(
            "Reviewed 2021 k15 observed-only base config did not load as a mapping"
        )

    rows: List[Dict[str, Any]] = []
    for factor in spec["length_scale_upper_factors"]:
        factor = int(factor)
        cfg = _config_overrides(spec, base, factor)
        path = STUDY_DIR / "configs" / f"config_2021_lsupper_factor_{factor:02d}.yaml"
        text = yaml.safe_dump(cfg, sort_keys=False)
        _atomic_write_text(path, text)
        provenance = {
            "schema_version": 1,
            "study_id": spec["study_id"],
            "factor": factor,
            "base_config": str(base_path),
            "base_config_sha256": spec["base_config"]["sha256"],
            "generated_config": str(path),
            "generated_config_sha256": _sha256_file(path),
            "fit_code_commit": spec["fit_code"]["commit"],
            "scientific_override": {
                "kernel_ls_res_upper_factor_by_dataset.2021": factor
            },
            "execution_overrides": {
                "expected_limit_bands": False,
                "toy_scan_parallel": False,
                "per_mass_plots": False,
                "per_mass_fit_json": False
            },
            "injection_closure": spec["injection_closure"],
        }
        provenance_path = path.with_suffix(".provenance.json")
        _atomic_write_json(provenance_path, provenance)
        rows.append(provenance)
    return rows


def build_tasks(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for kind in ("scan", "injection"):
        for factor in [int(x) for x in spec["length_scale_upper_factors"]]:
            for truth_model in sorted(spec["truth_models"]):
                for scenario in sorted(spec["scenarios"]):
                    scenario_spec = spec["scenarios"][scenario]
                    for toy_index in range(int(spec["n_toys"])):
                        task_id = (
                            f"{kind}__f{factor:02d}__{truth_model}__"
                            f"{scenario}__t{toy_index:04d}"
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "kind": kind,
                                "factor": factor,
                                "truth_model": truth_model,
                                "function_tag": spec["truth_models"][
                                    truth_model
                                ]["function_tag"],
                                "scenario": scenario,
                                "source_family": scenario_spec["source_family"],
                                "exposure_multiplier": int(
                                    scenario_spec["exposure_multiplier"]
                                ),
                                "toy_index": int(toy_index),
                                "toy_root": str(TOY_ROOT_PATH),
                                "toy_container": f"toys/{truth_model}/{scenario}",
                                "toy_name": f"toy_{toy_index:04d}",
                                "config": str(
                                    STUDY_DIR
                                    / "configs"
                                    / f"config_2021_lsupper_factor_{factor:02d}.yaml"
                                ),
                            }
                        )
    return tasks


def _write_tasks_and_commands(
    spec: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> None:
    lines = [json.dumps(task, sort_keys=True) for task in tasks]
    _atomic_write_text(TASKS_PATH, "\n".join(lines) + "\n")
    command_dir = STUDY_DIR / "commands"
    for kind in ("scan", "injection"):
        commands = [
            f"{sys.executable} {Path(__file__).resolve()} run-task "
            f"{task['task_id']} --execute"
            for task in tasks
            if task["kind"] == kind
        ]
        _atomic_write_text(
            command_dir / f"{kind}_default_grid_commands.txt",
            "\n".join(commands) + "\n",
        )


def prepare(spec: Mapping[str, Any], force_toys: bool = False) -> Dict[str, Any]:
    report = preflight(spec)
    configs = _write_candidate_configs(spec)
    toy_manifest = _write_toy_root_and_manifest(spec, force=force_toys)
    tasks = build_tasks(spec)
    _write_tasks_and_commands(spec, tasks)
    prepared = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "prepared_utc": _utc_now(),
        "preflight": report,
        "configs": configs,
        "paired_toy_root": str(TOY_ROOT_PATH),
        "paired_toy_root_sha256": toy_manifest["toy_root_sha256"],
        "task_manifest": str(TASKS_PATH),
        "scan_tasks": sum(task["kind"] == "scan" for task in tasks),
        "injection_tasks": sum(task["kind"] == "injection" for task in tasks),
        "default_scan_mass_points": mass_grid(spec, None, None, None),
        "expected_limit_bands": False,
    }
    _atomic_write_json(STUDY_DIR / "derived" / "preparation_report.json", prepared)
    return prepared


def load_tasks(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not TASKS_PATH.is_file():
        return build_tasks(spec)
    tasks: List[Dict[str, Any]] = []
    with TASKS_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                tasks.append(json.loads(line))
    expected = build_tasks(spec)
    if tasks != expected:
        raise StudyError("Task manifest drift; rerun prepare after review")
    return tasks


def _factor_config_path(factor: int) -> Path:
    return (
        STUDY_DIR
        / "configs"
        / f"config_2021_lsupper_factor_{int(factor):02d}.yaml"
    )


def _verify_frozen_factor_config(
    spec: Mapping[str, Any], factor: int
) -> Path:
    factor = int(factor)
    expected_by_factor = dict(
        spec.get("frozen_generated_config_sha256_by_factor") or {}
    )
    expected = str(expected_by_factor.get(str(factor), ""))
    if len(expected) != 64:
        raise StudyError(
            f"Missing frozen generated-config SHA-256 for factor {factor}"
        )
    path = _factor_config_path(factor)
    _verify_sha(path, expected, f"frozen factor-{factor} generated config")
    return path


def _validate_anchor_part_payload(
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    closure = _injection_protocol_spec(spec)
    if int(payload.get("schema_version", -1)) != 1:
        raise StudyError(f"Invalid anchor-part schema for {identity['anchor_id']}")
    exact = {
        "study_id": spec["study_id"],
        "injection_protocol": INJECTION_PROTOCOL,
        "anchor_id": identity["anchor_id"],
        "truth_model": identity["truth_model"],
        "function_tag": identity["function_tag"],
        "scenario": identity["scenario"],
        "toy_index": int(identity["toy_index"]),
        "injection_anchor_factor": int(closure["anchor_factor"]),
        "anchor_sigma_a_source": "asimov",
        "anchor_sigma_a_ref_mode": "prefit_asimov",
        "factor15_config_sha256": _sha256_file(
            _verify_frozen_factor_config(
                spec, int(closure["anchor_factor"])
            )
        ),
        "toy_root_sha256": _sha256_file(TOY_ROOT_PATH),
        "fit_code_commit": spec["fit_code"]["commit"],
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise StudyError(
                f"Anchor part {identity['anchor_id']} has {key}="
                f"{payload.get(key)!r}, expected {expected!r}"
            )
    entries = list(payload.get("entries") or [])
    expected_masses = sorted(float(x) for x in closure["masses_gev"])
    found_masses = sorted(float(row["mass_GeV"]) for row in entries)
    if found_masses != expected_masses:
        raise StudyError(
            f"Anchor part {identity['anchor_id']} mass grid mismatch: "
            f"{found_masses} != {expected_masses}"
        )
    expected_tags = [float(x) for x in closure["sigma_strengths"]]
    for entry in entries:
        entry_exact = {
            "truth_model": identity["truth_model"],
            "scenario": identity["scenario"],
            "toy_index": int(identity["toy_index"]),
            "injection_anchor_factor": int(closure["anchor_factor"]),
            "sigmaA_ref_mode": "prefit_asimov",
        }
        for key, expected in entry_exact.items():
            if entry.get(key) != expected:
                raise StudyError(
                    f"Anchor entry {identity['anchor_id']} has {key}="
                    f"{entry.get(key)!r}, expected {expected!r}"
                )
        sigma_ref = float(entry["injection_anchor_sigmaA_ref"])
        if not math.isfinite(sigma_ref) or not (sigma_ref > 0.0):
            raise StudyError(
                f"Nonpositive factor-15 sigmaA_ref in {identity['anchor_id']}"
            )
        points = list(entry.get("strength_points") or [])
        tags = [float(row["injection_anchor_nsigma"]) for row in points]
        if tags != expected_tags:
            raise StudyError(
                f"Anchor strength tags drift in {identity['anchor_id']}: "
                f"{tags} != {expected_tags}"
            )
        for point in points:
            expected_strength = (
                float(point["injection_anchor_nsigma"]) * sigma_ref
            )
            if float(point["injection_anchor_strength"]).hex() != float(
                expected_strength
            ).hex():
                raise StudyError(
                    f"Anchor amplitude is not an exact tag*sigmaA product in "
                    f"{identity['anchor_id']}"
                )
            draw_hash = str(point.get("signal_draw_sha256", ""))
            if len(draw_hash) != 64 or any(
                character not in "0123456789abcdef" for character in draw_hash
            ):
                raise StudyError(
                    f"Invalid signal-draw hash in {identity['anchor_id']}"
                )


def _anchor_part_is_valid(
    spec: Mapping[str, Any], identity: Mapping[str, Any]
) -> bool:
    path = _anchor_part_path(identity)
    if not path.is_file():
        return False
    try:
        _validate_anchor_part_payload(spec, identity, _load_json(path))
        return True
    except Exception:
        return False


def _build_anchor_part_payload(
    spec: Mapping[str, Any], identity: Mapping[str, Any]
) -> Dict[str, Any]:
    import numpy as np

    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import (
        FuncFormToySpec,
        build_funcform_toy_dataset,
        load_funcform_toy_hist,
    )
    from hps_gpr.injection import (
        _build_injection_mass_context,
        _inject_counts_from_template,
        _stable_point_seed,
        _stable_toy_seed,
    )

    closure = _injection_protocol_spec(spec)
    anchor_factor = int(closure["anchor_factor"])
    config_path = _verify_frozen_factor_config(spec, anchor_factor)
    cfg = load_config(str(config_path))
    cfg.inj_strength_mode = "absolute"
    cfg.inj_sigma_a_source = "asimov"
    cfg.inj_sigma_a_ref_mode = "prefit_asimov"
    cfg.inj_n_workers = 1
    datasets = make_datasets(cfg)
    if "2021" not in datasets:
        raise StudyError("Factor-15 anchor config does not enable 2021")

    toy_spec = FuncFormToySpec(
        source_root=str(TOY_ROOT_PATH),
        container=str(identity["toy_container"]),
        function_tag=str(identity["function_tag"]),
        toy_name=str(identity["toy_name"]),
        toy_index=int(identity["toy_index"]),
    )
    toy_hist = load_funcform_toy_hist(
        toy_spec.source_root,
        container=toy_spec.container,
        toy_name=toy_spec.toy_name,
    )
    toy_ds = build_funcform_toy_dataset(datasets["2021"], toy_hist, toy_spec)
    injection_seed = _injection_seed(
        spec,
        str(identity["truth_model"]),
        str(identity["scenario"]),
        int(identity["toy_index"]),
    )

    entries: List[Dict[str, Any]] = []
    for mass in [float(x) for x in closure["masses_gev"]]:
        optimizer_seed = _injection_mass_optimizer_seed(
            spec,
            str(identity["truth_model"]),
            str(identity["scenario"]),
            int(identity["toy_index"]),
            mass,
        )
        np.random.seed(int(optimizer_seed))
        ctx = _build_injection_mass_context(
            toy_ds,
            cfg,
            mass=float(mass),
            seed=int(injection_seed),
            inj_mode=str(closure["mode"]),
            sigma_source="asimov",
            refit_gp_on_toy=bool(closure["refit_gp_on_toy"]),
            refit_restarts=int(closure["refit_gp_restarts"]),
            refit_optimize=bool(closure["refit_gp_optimize"]),
            inj_shape_mode=str(closure["shape_mode"]),
            inj_background_mode=str(closure["background_mode"]),
            train_exclude_nsigma=float(closure["train_exclude_nsigma"]),
            mvn_method=str(getattr(cfg, "mvn_trunc_method", "reject_then_clip")),
            mvn_max_tries=int(getattr(cfg, "mvn_trunc_max_tries", 80)),
        )
        sigma_ref = float(ctx.sigmaA_ref)
        if str(ctx.sigmaA_ref_mode) != "prefit_asimov":
            raise StudyError(
                f"Anchor context resolved sigmaA_ref_mode="
                f"{ctx.sigmaA_ref_mode!r}, expected 'prefit_asimov'"
            )
        if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
            raise StudyError(
                f"Invalid factor-15 sigmaA_ref at mass {mass:.6f}: {sigma_ref}"
            )
        if float(sigma_ref).hex() != float(ctx.sigmaA_ref_prefit).hex():
            raise StudyError(
                f"Factor-15 sigmaA_ref differs from its prefit-Asimov value "
                f"at mass {mass:.6f}"
            )

        strength_points: List[Dict[str, Any]] = []
        for tag in [float(x) for x in closure["sigma_strengths"]]:
            strength = float(tag) * sigma_ref
            point_seed = _stable_point_seed(
                int(injection_seed), str(toy_ds.key), float(mass), strength
            )
            toy_seed = _stable_toy_seed(int(point_seed), 0)
            signal, n_signal_full, _ = _inject_counts_from_template(
                ctx.tmpl_full,
                strength,
                np.random.default_rng(int(toy_seed)),
                str(closure["mode"]),
            )
            signal = np.asarray(signal, dtype=np.int64)
            strength_points.append(
                {
                    "injection_anchor_nsigma": float(tag),
                    "injection_anchor_strength": float(strength),
                    "signal_point_seed": int(point_seed),
                    "signal_toy_seed": int(toy_seed),
                    "signal_Nsig_full": int(n_signal_full),
                    "signal_Nsig_win": int(np.sum(signal[ctx.msk_blind])),
                    "signal_Nsig_train": int(np.sum(signal[ctx.msk_train])),
                    "signal_draw_sha256": _sha256_array_int64(signal),
                }
            )
        entries.append(
            {
                "truth_model": str(identity["truth_model"]),
                "scenario": str(identity["scenario"]),
                "toy_index": int(identity["toy_index"]),
                "mass_GeV": float(mass),
                "injection_anchor_factor": anchor_factor,
                "injection_anchor_sigmaA_ref": sigma_ref,
                "sigmaA_ref_mode": str(ctx.sigmaA_ref_mode),
                "optimizer_seed": int(optimizer_seed),
                "injection_seed": int(injection_seed),
                "signal_template_sha256": hashlib.sha256(
                    np.asarray(ctx.tmpl_full, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
                "strength_points": strength_points,
            }
        )

    payload = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "injection_protocol": INJECTION_PROTOCOL,
        "anchor_id": identity["anchor_id"],
        "truth_model": identity["truth_model"],
        "function_tag": identity["function_tag"],
        "scenario": identity["scenario"],
        "toy_index": int(identity["toy_index"]),
        "injection_anchor_factor": anchor_factor,
        "anchor_sigma_a_source": "asimov",
        "anchor_sigma_a_ref_mode": "prefit_asimov",
        "factor15_config": str(config_path),
        "factor15_config_sha256": _sha256_file(config_path),
        "toy_root": str(TOY_ROOT_PATH),
        "toy_root_sha256": _sha256_file(TOY_ROOT_PATH),
        "fit_code_commit": spec["fit_code"]["commit"],
        "entries": entries,
    }
    _validate_anchor_part_payload(spec, identity, payload)
    return payload


def run_anchor_part(
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    execute: bool,
    force: bool,
    clear_stale_lock: bool,
) -> Dict[str, Any]:
    path = _anchor_part_path(identity)
    if _anchor_part_is_valid(spec, identity) and not force:
        return {
            "anchor_id": identity["anchor_id"],
            "status": "already_complete",
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    if not execute:
        return {
            "anchor_id": identity["anchor_id"],
            "status": "dry_run",
            "path": str(path),
            "command": (
                f"{sys.executable} {Path(__file__).resolve()} run-anchor-part "
                f"{identity['anchor_id']} --execute"
            ),
        }

    lock_root = INJECTION_ANCHOR_PARTS_DIR / "locks" / str(identity["anchor_id"])
    lock = _acquire_lock(lock_root, clear_stale_lock=clear_stale_lock)
    try:
        payload = _build_anchor_part_payload(spec, identity)
        _atomic_write_json(path, payload)
        return {
            "anchor_id": identity["anchor_id"],
            "status": "completed",
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _consolidate_anchor_ledger(spec: Mapping[str, Any]) -> Dict[str, Any]:
    closure = _injection_protocol_spec(spec)
    identities = build_anchor_identities(spec)
    entries: List[Dict[str, Any]] = []
    part_rows: List[Dict[str, str]] = []
    for identity in identities:
        path = _anchor_part_path(identity)
        if not path.is_file():
            raise StudyError(f"Missing anchor part: {path}")
        payload = _load_json(path)
        _validate_anchor_part_payload(spec, identity, payload)
        entries.extend(dict(row) for row in payload["entries"])
        part_rows.append(
            {
                "anchor_id": str(identity["anchor_id"]),
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    entries.sort(
        key=lambda row: _anchor_entry_key(
            str(row["truth_model"]),
            str(row["scenario"]),
            int(row["toy_index"]),
            float(row["mass_GeV"]),
        )
    )
    config_path = _verify_frozen_factor_config(
        spec, int(closure["anchor_factor"])
    )
    manifest = _load_json(TOY_MANIFEST_PATH)
    ledger = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "injection_protocol": INJECTION_PROTOCOL,
        "injection_anchor_factor": int(closure["anchor_factor"]),
        "anchor_sigma_a_source": "asimov",
        "anchor_sigma_a_ref_mode": "prefit_asimov",
        "masses_GeV": [float(x) for x in closure["masses_gev"]],
        "anchor_nsigma_values": [
            float(x) for x in closure["sigma_strengths"]
        ],
        "factor15_config": str(config_path),
        "factor15_config_sha256": _sha256_file(config_path),
        "toy_root": str(TOY_ROOT_PATH),
        "toy_root_sha256": manifest["toy_root_sha256"],
        "fit_code_commit": spec["fit_code"]["commit"],
        "part_count": len(part_rows),
        "entry_count": len(entries),
        "parts": part_rows,
        "entries": entries,
    }
    _atomic_write_json(INJECTION_ANCHOR_LEDGER_PATH, ledger)
    loaded, ledger_sha, _ = load_injection_anchor_ledger(spec)
    return {
        "status": "complete",
        "path": str(INJECTION_ANCHOR_LEDGER_PATH),
        "sha256": ledger_sha,
        "part_count": int(loaded["part_count"]),
        "entry_count": int(loaded["entry_count"]),
    }


def load_injection_anchor_ledger(
    spec: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[Tuple[str, str, int, str], Dict[str, Any]]]:
    closure = _injection_protocol_spec(spec)
    if not INJECTION_ANCHOR_LEDGER_PATH.is_file():
        raise StudyError(
            "Missing factor-15 injection anchor ledger. Run "
            "'prepare-injection-anchors --execute' and review it before "
            "launching injection tasks."
        )
    ledger = _load_json(INJECTION_ANCHOR_LEDGER_PATH)
    exact = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "injection_protocol": INJECTION_PROTOCOL,
        "injection_anchor_factor": int(closure["anchor_factor"]),
        "anchor_sigma_a_source": "asimov",
        "anchor_sigma_a_ref_mode": "prefit_asimov",
        "factor15_config_sha256": _sha256_file(
            _verify_frozen_factor_config(
                spec, int(closure["anchor_factor"])
            )
        ),
        "toy_root_sha256": _sha256_file(TOY_ROOT_PATH),
        "fit_code_commit": spec["fit_code"]["commit"],
    }
    for key, expected in exact.items():
        if ledger.get(key) != expected:
            raise StudyError(
                f"Injection anchor ledger {key}={ledger.get(key)!r}, "
                f"expected {expected!r}"
            )
    entries = list(ledger.get("entries") or [])
    expected_count = (
        len(spec["truth_models"])
        * len(spec["scenarios"])
        * int(spec["n_toys"])
        * len(closure["masses_gev"])
    )
    if len(entries) != expected_count or int(ledger.get("entry_count", -1)) != expected_count:
        raise StudyError(
            f"Injection anchor ledger has {len(entries)} entries; "
            f"expected {expected_count}"
        )
    expected_keys = {
        _anchor_entry_key(truth, scenario, toy_index, mass)
        for truth in spec["truth_models"]
        for scenario in spec["scenarios"]
        for toy_index in range(int(spec["n_toys"]))
        for mass in closure["masses_gev"]
    }
    index: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    for row in entries:
        key = _anchor_entry_key(
            str(row["truth_model"]),
            str(row["scenario"]),
            int(row["toy_index"]),
            float(row["mass_GeV"]),
        )
        if key in index:
            raise StudyError(f"Duplicate injection anchor ledger key: {key}")
        if key not in expected_keys:
            raise StudyError(f"Unexpected injection anchor ledger key: {key}")
        if int(row.get("injection_anchor_factor", -1)) != int(
            closure["anchor_factor"]
        ):
            raise StudyError(f"Injection anchor factor drift for {key}")
        if str(row.get("sigmaA_ref_mode", "")) != "prefit_asimov":
            raise StudyError(f"Injection anchor sigmaA_ref_mode drift for {key}")
        sigma_ref = float(row["injection_anchor_sigmaA_ref"])
        if not math.isfinite(sigma_ref) or not (sigma_ref > 0.0):
            raise StudyError(f"Invalid injection anchor sigmaA_ref for {key}")
        points = list(row.get("strength_points") or [])
        tags = [float(point["injection_anchor_nsigma"]) for point in points]
        if tags != [float(x) for x in closure["sigma_strengths"]]:
            raise StudyError(f"Injection anchor strength-tag drift for {key}")
        for point in points:
            expected_strength = (
                float(point["injection_anchor_nsigma"]) * sigma_ref
            )
            if float(point["injection_anchor_strength"]).hex() != float(
                expected_strength
            ).hex():
                raise StudyError(f"Injection anchor amplitude drift for {key}")
            draw_hash = str(point.get("signal_draw_sha256", ""))
            if len(draw_hash) != 64 or any(
                character not in "0123456789abcdef" for character in draw_hash
            ):
                raise StudyError(f"Injection signal-draw hash drift for {key}")
        index[key] = dict(row)
    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))
        raise StudyError(
            f"Injection anchor ledger is missing {len(missing)} keys"
        )
    return ledger, _sha256_file(INJECTION_ANCHOR_LEDGER_PATH), index


def prepare_injection_anchors(
    spec: Mapping[str, Any],
    *,
    execute: bool,
    max_parts: int,
    workers: int,
    force: bool,
) -> Dict[str, Any]:
    identities = build_anchor_identities(spec)
    pending = [
        identity
        for identity in identities
        if force or not _anchor_part_is_valid(spec, identity)
    ]
    selected = pending[: max(0, int(max_parts))]
    if not execute:
        return {
            "status": "dry_run",
            "injection_protocol": INJECTION_PROTOCOL,
            "injection_anchor_factor": 15,
            "identities_total": len(identities),
            "parts_complete": len(identities) - len(pending),
            "parts_pending": len(pending),
            "parts_selected": len(selected),
            "ledger": str(INJECTION_ANCHOR_LEDGER_PATH),
            "would_write_shared_configs": False,
            "would_write_task_manifest": False,
            "commands": [
                (
                    f"{sys.executable} {Path(__file__).resolve()} "
                    f"run-anchor-part {identity['anchor_id']} --execute"
                )
                for identity in selected
            ],
        }

    commands: List[Tuple[str, List[str]]] = []
    for identity in selected:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-anchor-part",
            str(identity["anchor_id"]),
            "--execute",
        ]
        if force:
            command.append("--force")
        commands.append((str(identity["anchor_id"]), command))

    results: List[Dict[str, Any]] = []

    def launch(item: Tuple[str, List[str]]) -> Dict[str, Any]:
        anchor_id, command = item
        completed = subprocess.run(command, text=True)
        return {
            "anchor_id": anchor_id,
            "returncode": int(completed.returncode),
            "command": command,
        }

    n_workers = max(1, min(int(workers), max(1, len(commands))))
    if n_workers == 1:
        for item in commands:
            result = launch(item)
            results.append(result)
            if result["returncode"] != 0:
                break
    elif commands:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(launch, item): item[0] for item in commands}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "anchor_id": futures[future],
                            "returncode": -1,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        results.sort(key=lambda row: str(row["anchor_id"]))
    failed = [
        row["anchor_id"] for row in results if int(row.get("returncode", -1)) != 0
    ]
    if failed:
        raise StudyError(
            "Injection anchor child failure(s): " + ", ".join(map(str, failed))
        )

    remaining = [
        identity
        for identity in identities
        if not _anchor_part_is_valid(spec, identity)
    ]
    ledger_report: Optional[Dict[str, Any]] = None
    if not remaining:
        ledger_report = _consolidate_anchor_ledger(spec)
    return {
        "status": "complete" if not remaining else "partial",
        "injection_protocol": INJECTION_PROTOCOL,
        "parts_run": len(results),
        "parts_remaining": len(remaining),
        "ledger": ledger_report,
        "results": results,
        "wrote_shared_configs": False,
        "wrote_task_manifest": False,
    }


def mass_grid(
    spec: Mapping[str, Any],
    minimum_mev: Optional[int],
    maximum_mev: Optional[int],
    step_mev: Optional[int],
) -> List[float]:
    default = spec["default_mass_grid_mev"]
    lo = int(default["min"] if minimum_mev is None else minimum_mev)
    hi = int(default["max"] if maximum_mev is None else maximum_mev)
    step = int(default["step"] if step_mev is None else step_mev)
    if step <= 0 or hi < lo:
        raise StudyError("Invalid mass grid")
    allowed = spec["scan_range_gev"]
    if lo < int(round(float(allowed[0]) * 1000)) or hi > int(
        round(float(allowed[1]) * 1000)
    ):
        raise StudyError("Mass-grid override is outside the frozen 2021 scan range")
    values = list(range(lo, hi + 1, step))
    if not values or values[-1] != hi:
        raise StudyError("(max-min) must be exactly divisible by mass step")
    return [value / 1000.0 for value in values]


def _grid_tag(masses: Sequence[float]) -> str:
    mev = [int(round(float(x) * 1000)) for x in masses]
    step = 0 if len(mev) < 2 else mev[1] - mev[0]
    return f"grid_{mev[0]:03d}_{step:03d}_{mev[-1]:03d}"


def _task_by_id(tasks: Sequence[Mapping[str, Any]], task_id: str) -> Dict[str, Any]:
    matches = [dict(task) for task in tasks if task["task_id"] == task_id]
    if len(matches) != 1:
        raise StudyError(f"Unknown or duplicate task id: {task_id}")
    return matches[0]


def _task_root(task: Mapping[str, Any], grid_tag: str) -> Path:
    return (
        STUDY_DIR
        / "runs"
        / str(task["kind"])
        / f"f{int(task['factor']):02d}"
        / str(task["truth_model"])
        / str(task["scenario"])
        / f"toy_{int(task['toy_index']):04d}"
        / grid_tag
    )


def _success_marker_valid(attempt: Path) -> bool:
    marker_path = attempt / "_SUCCESS.json"
    if not marker_path.is_file():
        return False
    try:
        marker = _load_json(marker_path)
        result = Path(marker["result_path"])
        return result.is_file() and _sha256_file(result) == marker["result_sha256"]
    except Exception:
        return False


def _latest_success(task_root: Path) -> Optional[Path]:
    successes = [
        path
        for path in sorted(task_root.glob("attempt_*"))
        if path.is_dir() and _success_marker_valid(path)
    ]
    return successes[-1] if successes else None


def _choose_attempt(task_root: Path, force: bool) -> Path:
    task_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(path for path in task_root.glob("attempt_*") if path.is_dir())
    if attempts and not force and not _success_marker_valid(attempts[-1]):
        return attempts[-1]
    next_index = 1
    if attempts:
        next_index = max(int(path.name.split("_")[-1]) for path in attempts) + 1
    path = task_root / f"attempt_{next_index:03d}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def _acquire_lock(task_root: Path, clear_stale_lock: bool) -> Path:
    task_root.mkdir(parents=True, exist_ok=True)
    lock = task_root / ".run_lock.json"
    if lock.exists() and clear_stale_lock:
        lock.unlink()
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_utc": _utc_now(),
    }
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise StudyError(
            f"Task lock exists: {lock}. Inspect it; use --clear-stale-lock only "
            "after confirming no task process is active."
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return lock


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    task: Mapping[str, Any],
    config_sha: str,
    optimizer_seed: Optional[int] = None,
    injection_seed: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    extra = [
        "study_id",
        "task_id",
        "truth_model",
        "truth_function_tag",
        "study_scenario",
        "source_family",
        "exposure_multiplier",
        "ls_upper_factor_requested",
        "background_toy_index",
        "optimizer_seed",
        "injection_seed",
        "generated_config_sha256",
        "base_config_sha256",
        "fit_code_commit",
        "expected_limit_bands",
    ]
    spec = load_spec()
    enriched: List[Dict[str, Any]] = []
    for input_row in rows:
        row = dict(input_row)
        row.update(
            {
                "study_id": spec["study_id"],
                "task_id": task["task_id"],
                "truth_model": task["truth_model"],
                "truth_function_tag": task["function_tag"],
                "study_scenario": task["scenario"],
                "source_family": task["source_family"],
                "exposure_multiplier": task["exposure_multiplier"],
                "ls_upper_factor_requested": task["factor"],
                "background_toy_index": task["toy_index"],
                "optimizer_seed": "" if optimizer_seed is None else optimizer_seed,
                "injection_seed": "" if injection_seed is None else injection_seed,
                "generated_config_sha256": config_sha,
                "base_config_sha256": spec["base_config"]["sha256"],
                "fit_code_commit": spec["fit_code"]["commit"],
                "expected_limit_bands": False,
            }
        )
        enriched.append(row)
    original = list(rows[0].keys()) if rows else []
    fieldnames = original + [name for name in extra if name not in original]
    return fieldnames, enriched


def _run_one_scan_mass(
    spec: Mapping[str, Any],
    task: Mapping[str, Any],
    attempt: Path,
    mass_gev: float,
    config_sha: str,
) -> Path:
    import numpy as np

    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import (
        FuncFormToySpec,
        build_funcform_toy_dataset,
        load_funcform_toy_hist,
        run_funcform_toy_scans,
    )
    from hps_gpr.io import _build_model

    mass_mev = int(round(float(mass_gev) * 1000))
    part = attempt / "parts" / f"m{mass_mev:03d}MeV"
    result = part / "result_enriched.csv"
    marker_path = part / "_SUCCESS.json"
    if marker_path.is_file():
        marker = _load_json(marker_path)
        if result.is_file() and _sha256_file(result) == marker.get("result_sha256"):
            return result

    seed = _mass_seed(
        spec,
        str(task["truth_model"]),
        str(task["scenario"]),
        int(task["toy_index"]),
        float(mass_gev),
    )
    np.random.seed(seed)
    cfg = load_config(str(task["config"]))
    raw_base = part / "raw"
    cfg.output_dir = str(raw_base)
    cfg.make_ul_bands = False
    cfg.ul_bands_toys = 0
    cfg.do_combined_bands = False
    cfg.combined_bands_n_toys = 0
    cfg.make_eps2_bands = False
    cfg.toy_scan_parallel = False
    cfg.toy_scan_n_workers = 1

    datasets = make_datasets(cfg)
    if "2021" not in datasets:
        raise StudyError("Generated candidate config does not enable 2021")
    toy_spec = FuncFormToySpec(
        source_root=str(TOY_ROOT_PATH),
        container=str(task["toy_container"]),
        function_tag=str(task["function_tag"]),
        toy_name=str(task["toy_name"]),
        toy_index=int(task["toy_index"]),
    )
    run_funcform_toy_scans(
        datasets["2021"],
        cfg,
        [toy_spec],
        base_output_dir=str(raw_base),
        mass_min=float(mass_gev) - 1e-9,
        mass_max=float(mass_gev) + 1e-9,
        save_plots=False,
        save_fit_json=False,
        save_per_mass_folders=False,
        scan_parallel=False,
        scan_n_workers=1,
        scan_parallel_backend="threading",
        scan_threads_per_worker=1,
    )
    raw_result = (
        raw_base
        / "toy_scans"
        / "2021"
        / f"toy_{int(task['toy_index']):04d}"
        / "results_single.csv"
    )
    if not raw_result.is_file():
        raise StudyError(f"Missing one-mass scan output: {raw_result}")
    _, rows = _read_csv_rows(raw_result)
    if len(rows) != 1:
        raise StudyError(f"Expected one row at {mass_gev}, got {len(rows)}")
    returned_mass = float(rows[0]["mass_GeV"])
    if abs(returned_mass - float(mass_gev)) > 5e-7:
        raise StudyError(
            f"Mass-grid mismatch: requested {mass_gev}, got {returned_mass}"
        )
    # The immutable CSV schema predates the explicit geometry columns in the
    # active development tree. Reconstruct and record the exact model geometry
    # from the same immutable _build_model helper instead of assuming it.
    toy_hist = load_funcform_toy_hist(
        str(TOY_ROOT_PATH),
        container=str(task["toy_container"]),
        toy_name=str(task["toy_name"]),
    )
    toy_ds = build_funcform_toy_dataset(datasets["2021"], toy_hist, toy_spec)
    geometry_model = _build_model(
        toy_ds,
        (float(rows[0]["blind_lo"]), float(rows[0]["blind_hi"])),
        rebin=int(cfg.neighborhood_rebin),
        config=cfg,
        mass=float(mass_gev),
    )
    native_edges = np.asarray(toy_hist.axes[0].edges, dtype=float)
    rebinned_edges = np.asarray(
        geometry_model.histogram.axes[0].edges, dtype=float
    )
    centers = np.asarray(
        geometry_model.histogram.axes[0].centers, dtype=float
    )
    native_width = float(np.median(np.diff(native_edges)))
    rebinned_width = float(np.median(np.diff(rebinned_edges)))
    n_full = int(centers.size)
    sigma_val = float(rows[0]["sigma_val"])
    train_nsigma = float(
        getattr(cfg, "gp_train_exclude_nsigma", None) or cfg.blind_nsigma
    )
    train_lo = float(mass_gev) - train_nsigma * sigma_val
    train_hi = float(mass_gev) + train_nsigma * sigma_val
    n_train_low = int(np.count_nonzero(centers < train_lo))
    n_train_high = int(np.count_nonzero(centers > train_hi))
    expected_n_train = n_train_low + n_train_high
    n_blind = int(
        np.count_nonzero(
            (centers >= float(rows[0]["blind_lo"]))
            & (centers <= float(rows[0]["blind_hi"]))
        )
    )
    reported_n_train = int(float(rows[0]["n_train"]))

    if int(cfg.neighborhood_rebin) != 5:
        raise StudyError(f"Production rebin drift: {cfg.neighborhood_rebin}")
    if abs(native_width - 0.000125) > 1e-12:
        raise StudyError(f"Native toy bin-width drift: {native_width}")
    if abs(rebinned_width - 0.000625) > 1e-10 or n_full != 416:
        raise StudyError(
            "Production rebin-5 geometry failed: "
            f"width={rebinned_width}, n_full={n_full}"
        )
    if reported_n_train <= 0 or reported_n_train != expected_n_train:
        raise StudyError(
            "Training-bin accounting failed: "
            f"reported={reported_n_train}, expected={expected_n_train}, "
            f"low={n_train_low}, high={n_train_high}"
        )
    if int(cfg.n_restarts) != 12:
        raise StudyError(f"Production restart count drift: {cfg.n_restarts}")
    if abs(float(rows[0]["ls_hi_over_sigma_x"]) - float(task["factor"])) > 1e-8:
        raise StudyError(
            "Requested length-scale upper factor was not applied: "
            f"{rows[0]['ls_hi_over_sigma_x']} != {task['factor']}"
        )

    rows[0].update(
        {
            "native_input_bin_width_gev": native_width,
            "production_rebin_requested": int(cfg.neighborhood_rebin),
            "rebinned_bin_width_gev": rebinned_width,
            "rebinned_n_full": n_full,
            "rebinned_n_blind": n_blind,
            "rebinned_n_train_expected": expected_n_train,
            "rebinned_n_train_low": n_train_low,
            "rebinned_n_train_high": n_train_high,
            "training_geometry_valid": True,
            "optimizer_restarts_requested": int(cfg.n_restarts),
            "eps2_density_implementation": (
                "immutable_df4d456_rebinned_whole_bin"
            ),
            "eps2_up_promotable": False,
        }
    )
    fields, enriched = _enrich_rows(
        rows, task, config_sha, optimizer_seed=seed
    )
    _write_csv_rows(result, fields, enriched)
    _atomic_write_json(
        marker_path,
        {
            "completed_utc": _utc_now(),
            "mass_gev": float(mass_gev),
            "optimizer_seed": seed,
            "result_path": str(result),
            "result_sha256": _sha256_file(result),
        },
    )
    return result


def _run_scan_task(
    spec: Mapping[str, Any],
    task: Mapping[str, Any],
    attempt: Path,
    masses: Sequence[float],
) -> Path:
    config_path = Path(task["config"])
    frozen_config_path = _verify_frozen_factor_config(
        spec, int(task["factor"])
    )
    if config_path.resolve() != frozen_config_path.resolve():
        raise StudyError(
            f"Task config path {config_path} does not match frozen factor-"
            f"{int(task['factor'])} config {frozen_config_path}"
        )
    config_sha = _sha256_file(config_path)
    part_results = [
        _run_one_scan_mass(spec, task, attempt, mass, config_sha)
        for mass in masses
    ]
    all_rows: List[Dict[str, str]] = []
    fields: List[str] = []
    for path in part_results:
        part_fields, rows = _read_csv_rows(path)
        if not fields:
            fields = part_fields
        elif fields != part_fields:
            raise StudyError(f"Per-mass schema drift in {path}")
        all_rows.extend(rows)
    if len(all_rows) != len(masses):
        raise StudyError("Consolidated scan row count mismatch")
    output = attempt / "results_single_enriched.csv"
    _write_csv_rows(output, fields, all_rows)
    return output


def _run_with_signal_draw_capture(
    injection_module: Any, callback: Any
) -> List[Dict[str, Any]]:
    """Capture the exact signal arrays produced by the pinned injection helper."""

    import numpy as np

    original = injection_module._inject_counts_from_template
    captured: List[Dict[str, Any]] = []

    def capture(
        template: Any, strength: float, rng: Any, mode: str = "multinomial"
    ) -> Tuple[Any, int, float]:
        signal, n_signal, fraction = original(template, strength, rng, mode)
        signal_array = np.asarray(signal, dtype=np.int64)
        captured.append(
            {
                "strength": float(strength),
                "signal_Nsig_full": int(n_signal),
                "signal_draw_sha256": _sha256_array_int64(signal_array),
            }
        )
        return signal, n_signal, fraction

    injection_module._inject_counts_from_template = capture
    try:
        callback()
    finally:
        injection_module._inject_counts_from_template = original
    return captured


def _annotate_fixed_anchor_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    captured: Sequence[Mapping[str, Any]],
    anchor_entry: Mapping[str, Any],
    anchor_ledger_sha256: str,
    candidate_factor: int,
) -> List[Dict[str, Any]]:
    points = list(anchor_entry["strength_points"])
    if not (len(rows) == len(captured) == len(points)):
        raise StudyError(
            "Injection row/signal/anchor cardinality mismatch: "
            f"{len(rows)} rows, {len(captured)} draws, {len(points)} anchors"
        )
    annotated: List[Dict[str, Any]] = []
    sigma_anchor = float(anchor_entry["injection_anchor_sigmaA_ref"])
    for raw, draw, point in zip(rows, captured, points):
        row = dict(raw)
        anchor_strength = float(point["injection_anchor_strength"])
        raw_strength = float(row["strength"])
        captured_strength = float(draw["strength"])
        tolerance = max(1e-12, abs(anchor_strength) * 1e-12)
        if abs(raw_strength - anchor_strength) > tolerance:
            raise StudyError(
                f"Runner strength {raw_strength} differs from anchored absolute "
                f"strength {anchor_strength}"
            )
        if abs(captured_strength - anchor_strength) > tolerance:
            raise StudyError(
                f"Captured strength {captured_strength} differs from anchored "
                f"absolute strength {anchor_strength}"
            )
        actual_hash = str(draw["signal_draw_sha256"])
        reference_hash = str(point["signal_draw_sha256"])
        if actual_hash != reference_hash:
            raise StudyError(
                "Signal realization differs from the factor-15 anchor: "
                f"actual {actual_hash}, reference {reference_hash}"
            )
        actual_full = int(draw["signal_Nsig_full"])
        actual_win = int(float(row["Nsig_win"]))
        actual_train = int(float(row["Nsig_train"]))
        expected_full = int(point["signal_Nsig_full"])
        expected_win = int(point["signal_Nsig_win"])
        expected_train = int(point["signal_Nsig_train"])
        if (actual_full, actual_win, actual_train) != (
            expected_full,
            expected_win,
            expected_train,
        ):
            raise StudyError(
                "Signal Nsig realization differs from the factor-15 anchor: "
                f"actual={(actual_full, actual_win, actual_train)}, "
                f"reference={(expected_full, expected_win, expected_train)}"
            )
        if str(row.get("sigmaA_ref_mode", "")) != "prefit_asimov":
            raise StudyError(
                f"Candidate sigmaA_ref_mode={row.get('sigmaA_ref_mode')!r}; "
                "expected 'prefit_asimov'"
            )
        if int(candidate_factor) == int(anchor_entry["injection_anchor_factor"]):
            candidate_sigma = float(row["sigmaA_ref"])
            sigma_tolerance = max(1e-12, abs(sigma_anchor) * 1e-10)
            if abs(candidate_sigma - sigma_anchor) > sigma_tolerance:
                raise StudyError(
                    "Factor-15 candidate sigmaA_ref does not reproduce its "
                    f"anchor: {candidate_sigma} != {sigma_anchor}"
                )
        row.update(
            {
                "injection_protocol": INJECTION_PROTOCOL,
                "injection_strength_mode": "absolute",
                "injection_anchor_factor": int(
                    anchor_entry["injection_anchor_factor"]
                ),
                "injection_anchor_nsigma": float(
                    point["injection_anchor_nsigma"]
                ),
                "injection_anchor_strength": anchor_strength,
                "injection_anchor_sigmaA_ref": sigma_anchor,
                "injection_anchor_ledger_sha256": str(
                    anchor_ledger_sha256
                ),
                "signal_draw_sha256": actual_hash,
                "signal_draw_reference_sha256": reference_hash,
                "signal_draw_hash_verified": True,
                "signal_Nsig_full": actual_full,
                "signal_Nsig_full_anchor": expected_full,
                "signal_Nsig_win_anchor": expected_win,
                "signal_Nsig_train_anchor": expected_train,
                "signal_Nsig_win_matches_anchor": True,
                "signal_Nsig_train_matches_anchor": True,
            }
        )
        annotated.append(row)
    return annotated


def _run_injection_task(
    spec: Mapping[str, Any],
    task: Mapping[str, Any],
    attempt: Path,
) -> Path:
    import numpy as np

    import hps_gpr.injection as injection_module
    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import (
        FuncFormToySpec,
        run_funcform_injection_extraction,
    )

    config_path = Path(task["config"])
    frozen_config_path = _verify_frozen_factor_config(
        spec, int(task["factor"])
    )
    if config_path.resolve() != frozen_config_path.resolve():
        raise StudyError(
            f"Task config path {config_path} does not match frozen factor-"
            f"{int(task['factor'])} config {frozen_config_path}"
        )
    config_sha = _sha256_file(config_path)
    cfg = load_config(str(config_path))
    cfg.output_dir = str(attempt / "raw")
    cfg.make_ul_bands = False
    cfg.ul_bands_toys = 0
    cfg.do_combined_bands = False
    cfg.combined_bands_n_toys = 0
    cfg.make_eps2_bands = False
    cfg.inj_n_workers = 1
    # The shared candidate cards remain frozen with their original sigmaA-mode
    # declarations. Only injection execution is overridden in memory after a
    # reviewed factor-15 anchor ledger exists.
    cfg.inj_strength_mode = "absolute"
    cfg.inj_sigma_a_source = "asimov"
    cfg.inj_sigma_a_ref_mode = "prefit_asimov"
    datasets = make_datasets(cfg)
    if "2021" not in datasets:
        raise StudyError("Generated candidate config does not enable 2021")

    closure = _injection_protocol_spec(spec)
    _, anchor_ledger_sha, anchor_index = load_injection_anchor_ledger(spec)
    injection_seed = _injection_seed(
        spec,
        str(task["truth_model"]),
        str(task["scenario"]),
        int(task["toy_index"]),
    )
    toy_spec = FuncFormToySpec(
        source_root=str(TOY_ROOT_PATH),
        container=str(task["toy_container"]),
        function_tag=str(task["function_tag"]),
        toy_name=str(task["toy_name"]),
        toy_index=int(task["toy_index"]),
    )

    all_rows: List[Dict[str, Any]] = []
    fields: List[str] = []
    for mass in [float(x) for x in closure["masses_gev"]]:
        anchor_key = _anchor_entry_key(
            str(task["truth_model"]),
            str(task["scenario"]),
            int(task["toy_index"]),
            mass,
        )
        if anchor_key not in anchor_index:
            raise StudyError(f"Missing injection anchor entry: {anchor_key}")
        anchor_entry = anchor_index[anchor_key]
        strengths = [
            float(point["injection_anchor_strength"])
            for point in anchor_entry["strength_points"]
        ]
        optimizer_seed = _injection_mass_optimizer_seed(
            spec,
            str(task["truth_model"]),
            str(task["scenario"]),
            int(task["toy_index"]),
            mass,
        )
        # Reset before every independent mass call. Signal refits at an earlier
        # mass therefore cannot perturb the background-fit starts at a later one.
        np.random.seed(int(optimizer_seed))
        mass_tag = f"m{int(round(mass * 1000.0)):03d}"
        raw_base = attempt / "raw" / mass_tag
        cfg.output_dir = str(raw_base)

        def run_one_mass() -> None:
            run_funcform_injection_extraction(
                datasets["2021"],
                cfg,
                [toy_spec],
                base_output_dir=str(raw_base),
                masses=[float(mass)],
                strengths=strengths,
                n_injection_toys=int(
                    closure["replicas_per_background_toy"]
                ),
                seed=int(injection_seed - int(task["toy_index"])),
                write_toy_csv=True,
            )

        captured = _run_with_signal_draw_capture(
            injection_module, run_one_mass
        )
        raw_result = (
            raw_base
            / "funcform_injection_jobs"
            / "2021"
            / f"toy_{int(task['toy_index']):04d}"
            / "inj_extract_toys_2021.csv"
        )
        if not raw_result.is_file():
            raise StudyError(f"Missing injection toy output: {raw_result}")
        _, rows = _read_csv_rows(raw_result)
        expected = len(strengths) * int(
            closure["replicas_per_background_toy"]
        )
        if len(rows) != expected:
            raise StudyError(
                f"Expected {expected} injection rows at {mass}, got {len(rows)}"
            )
        annotated = _annotate_fixed_anchor_rows(
            rows=rows,
            captured=captured,
            anchor_entry=anchor_entry,
            anchor_ledger_sha256=anchor_ledger_sha,
            candidate_factor=int(task["factor"]),
        )
        part_fields, enriched = _enrich_rows(
            annotated,
            task,
            config_sha,
            optimizer_seed=optimizer_seed,
            injection_seed=injection_seed,
        )
        if not fields:
            fields = part_fields
        elif fields != part_fields:
            raise StudyError(
                f"Per-mass injection schema drift at mass {mass:.6f}"
            )
        all_rows.extend(enriched)

    expected_total = (
        len(closure["masses_gev"])
        * len(closure["sigma_strengths"])
        * int(closure["replicas_per_background_toy"])
    )
    if len(all_rows) != expected_total:
        raise StudyError(
            f"Expected {expected_total} consolidated injection rows, "
            f"got {len(all_rows)}"
        )
    output = attempt / "injection_rows_enriched.csv"
    _write_csv_rows(output, fields, all_rows)
    _atomic_write_json(
        attempt / "injection_protocol_provenance.json",
        {
            "schema_version": 1,
            "study_id": spec["study_id"],
            "task_id": task["task_id"],
            "injection_protocol": INJECTION_PROTOCOL,
            "injection_anchor_factor": int(closure["anchor_factor"]),
            "injection_anchor_ledger": str(INJECTION_ANCHOR_LEDGER_PATH),
            "injection_anchor_ledger_sha256": anchor_ledger_sha,
            "shared_config_unchanged": True,
            "runtime_injection_strength_mode": "absolute",
            "signal_draw_hash_verified": True,
        },
    )
    return output


def run_task(
    spec: Mapping[str, Any],
    task: Mapping[str, Any],
    masses: Sequence[float],
    execute: bool,
    force: bool,
    clear_stale_lock: bool,
) -> Dict[str, Any]:
    grid_tag = "injection_grid" if task["kind"] == "injection" else _grid_tag(masses)
    root = _task_root(task, grid_tag)
    success = _latest_success(root)
    if success is not None and not force:
        return {
            "task_id": task["task_id"],
            "status": "already_complete",
            "attempt": str(success),
            "grid_tag": grid_tag,
        }
    if not execute:
        return {
            "task_id": task["task_id"],
            "status": "dry_run",
            "command": (
                f"{sys.executable} {Path(__file__).resolve()} run-task "
                f"{task['task_id']} --execute"
            ),
            "grid_tag": grid_tag,
            "masses_gev": list(masses) if task["kind"] == "scan" else None,
        }

    _configure_fit_process()
    # preflight and toy validation import NumPy/uproot; configure BLAS before
    # either import occurs in this fresh task process.
    preflight(spec)
    validate_toys(spec)
    _activate_fit_code(spec)

    lock = _acquire_lock(root, clear_stale_lock=clear_stale_lock)
    attempt = _choose_attempt(root, force=force)
    try:
        if task["kind"] == "scan":
            output = _run_scan_task(spec, task, attempt, masses)
        elif task["kind"] == "injection":
            output = _run_injection_task(spec, task, attempt)
        else:
            raise StudyError(f"Unsupported task kind: {task['kind']}")
        marker = {
            "schema_version": 1,
            "study_id": spec["study_id"],
            "task": dict(task),
            "completed_utc": _utc_now(),
            "attempt": str(attempt),
            "grid_tag": grid_tag,
            "masses_gev": list(masses) if task["kind"] == "scan" else None,
            "result_path": str(output),
            "result_sha256": _sha256_file(output),
            "fit_code_commit": spec["fit_code"]["commit"],
            "expected_limit_bands": False,
        }
        _atomic_write_json(attempt / "_SUCCESS.json", marker)
        stale_failure = attempt / "_FAILED.json"
        if stale_failure.exists():
            stale_failure.unlink()
        return {
            "task_id": task["task_id"],
            "status": "completed",
            "attempt": str(attempt),
            "result": str(output),
            "result_sha256": marker["result_sha256"],
        }
    except Exception as exc:
        _atomic_write_json(
            attempt / "_FAILED.json",
            {
                "failed_utc": _utc_now(),
                "task": dict(task),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _task_selected(
    task: Mapping[str, Any],
    kind: Optional[str],
    factors: Sequence[int],
    truths: Sequence[str],
    scenarios: Sequence[str],
) -> bool:
    return (
        (kind is None or task["kind"] == kind)
        and (not factors or int(task["factor"]) in factors)
        and (not truths or task["truth_model"] in truths)
        and (not scenarios or task["scenario"] in scenarios)
    )


def status_rows(
    spec: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    masses: Sequence[float],
    kind: Optional[str] = None,
    factors: Sequence[int] = (),
    truths: Sequence[str] = (),
    scenarios: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        if not _task_selected(task, kind, factors, truths, scenarios):
            continue
        grid_tag = (
            "injection_grid" if task["kind"] == "injection" else _grid_tag(masses)
        )
        root = _task_root(task, grid_tag)
        success = _latest_success(root)
        lock = root / ".run_lock.json"
        rows.append(
            {
                **dict(task),
                "grid_tag": grid_tag,
                "status": (
                    "complete"
                    if success is not None
                    else ("locked" if lock.exists() else "pending")
                ),
                "successful_attempt": "" if success is None else str(success),
                "lock_path": str(lock) if lock.exists() else "",
            }
        )
    return rows


def _print_status_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["kind"]), str(row["status"]))
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"{key[0]:10s} {key[1]:10s} {counts[key]:5d}")


def run_pending(
    spec: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    masses: Sequence[float],
    kind: str,
    max_tasks: int,
    workers: int,
    execute: bool,
    factors: Sequence[int],
    truths: Sequence[str],
    scenarios: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = status_rows(
        spec, tasks, masses, kind, factors, truths, scenarios
    )
    pending = [row for row in rows if row["status"] == "pending"]
    selected = pending[: max(0, int(max_tasks))]
    results: List[Dict[str, Any]] = []
    for row in selected:
        task = _task_by_id(tasks, str(row["task_id"]))
        if not execute:
            results.append(
                run_task(spec, task, masses, False, False, False)
            )
    if not execute or not selected:
        return results

    commands: List[Tuple[str, List[str]]] = []
    for row in selected:
        task = _task_by_id(tasks, str(row["task_id"]))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-task",
            str(task["task_id"]),
            "--execute",
        ]
        if task["kind"] == "scan":
            mev = [int(round(x * 1000)) for x in masses]
            command.extend(
                [
                    "--mass-min-mev",
                    str(mev[0]),
                    "--mass-max-mev",
                    str(mev[-1]),
                ]
            )
            if len(mev) >= 2:
                command.extend(["--mass-step-mev", str(mev[1] - mev[0])])
        if command.count("run-task") != 1 or command[2] != "run-task":
            raise StudyError(f"Invalid run-pending subprocess command: {command}")
        if command[3] != str(task["task_id"]):
            raise StudyError(f"Task id is misplaced in subprocess command: {command}")
        commands.append((str(task["task_id"]), command))

    n_workers = max(1, min(int(workers), len(commands)))

    def launch(item: Tuple[str, List[str]]) -> Dict[str, Any]:
        task_id, command = item
        completed = subprocess.run(command, text=True)
        return {
            "task_id": task_id,
            "returncode": int(completed.returncode),
            "command": command,
        }

    if n_workers == 1:
        for item in commands:
            result = launch(item)
            results.append(result)
            if result["returncode"] != 0:
                break
        failed = [row["task_id"] for row in results if row["returncode"] != 0]
        if failed:
            raise StudyError(
                "run-pending child task failure(s): " + ", ".join(map(str, failed))
            )
        return results

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(launch, item): item[0] for item in commands}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "task_id": futures[future],
                        "returncode": -1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    results.sort(key=lambda row: str(row["task_id"]))
    failed = [row["task_id"] for row in results if row.get("returncode") != 0]
    if failed:
        raise StudyError(
            "run-pending child task failure(s): " + ", ".join(map(str, failed))
        )
    return results


def _load_success_result(attempt: Path) -> Path:
    marker = _load_json(attempt / "_SUCCESS.json")
    result = Path(marker["result_path"])
    if not result.is_file() or _sha256_file(result) != marker["result_sha256"]:
        raise StudyError(f"Invalid successful attempt: {attempt}")
    return result


def validate_collected_injection_pairing(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Fail closed unless each cross-factor group reused one signal draw."""

    required = {
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_GeV",
        "toy",
        "ls_upper_factor_requested",
        "injection_protocol",
        "injection_anchor_factor",
        "injection_anchor_nsigma",
        "injection_anchor_strength",
        "injection_anchor_sigmaA_ref",
        "injection_anchor_ledger_sha256",
        "signal_draw_sha256",
        "signal_draw_hash_verified",
        "Nsig_win",
        "Nsig_train",
        "signal_Nsig_full",
    }
    groups: Dict[Tuple[str, str, int, str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise StudyError(
                "Collected injection row lacks fixed-anchor fields: "
                + ", ".join(missing)
            )
        if str(row["injection_protocol"]) != INJECTION_PROTOCOL:
            raise StudyError("Collected injection protocol drift")
        if str(row["signal_draw_hash_verified"]).lower() not in {
            "true",
            "1",
            "1.0",
        }:
            raise StudyError("Collected injection signal hash is not verified")
        key = (
            str(row["truth_model"]),
            str(row["study_scenario"]),
            int(float(row["background_toy_index"])),
            _anchor_mass_key(float(row["mass_GeV"])),
            int(float(row["toy"])),
            _anchor_mass_key(float(row["injection_anchor_nsigma"])),
        )
        groups.setdefault(key, []).append(row)

    compare_fields = (
        "injection_anchor_factor",
        "injection_anchor_strength",
        "injection_anchor_sigmaA_ref",
        "injection_anchor_ledger_sha256",
        "signal_draw_sha256",
        "Nsig_win",
        "Nsig_train",
        "signal_Nsig_full",
    )
    factor_counts: List[int] = []
    for key, group in groups.items():
        factors = {
            int(float(row["ls_upper_factor_requested"])) for row in group
        }
        if len(factors) != len(group):
            raise StudyError(f"Duplicate candidate factor in injection group {key}")
        factor_counts.append(len(factors))
        reference = group[0]
        for row in group[1:]:
            for field in compare_fields:
                left = str(reference[field])
                right = str(row[field])
                if field in {
                    "injection_anchor_factor",
                    "injection_anchor_strength",
                    "injection_anchor_sigmaA_ref",
                    "Nsig_win",
                    "Nsig_train",
                    "signal_Nsig_full",
                }:
                    same = float(left).hex() == float(right).hex()
                else:
                    same = left == right
                if not same:
                    raise StudyError(
                        f"Cross-factor injection mismatch for {key}, "
                        f"field {field}: {left!r} != {right!r}"
                    )
    return {
        "injection_protocol": INJECTION_PROTOCOL,
        "groups_validated": len(groups),
        "minimum_factors_per_group": min(factor_counts) if factor_counts else 0,
        "maximum_factors_per_group": max(factor_counts) if factor_counts else 0,
        "signal_draw_hash_and_Nsig_identical_within_group": True,
    }


def collect(
    spec: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    masses: Sequence[float],
    kind: str,
    allow_partial: bool,
    factors: Sequence[int] = (),
    truths: Sequence[str] = (),
    scenarios: Sequence[str] = (),
) -> Dict[str, Any]:
    rows = status_rows(
        spec,
        tasks,
        masses,
        kind=kind,
        factors=factors,
        truths=truths,
        scenarios=scenarios,
    )
    if not rows:
        raise StudyError("No tasks match the requested collection selectors")
    incomplete = [row for row in rows if row["status"] != "complete"]
    if incomplete and not allow_partial:
        raise StudyError(
            f"{len(incomplete)} {kind} tasks are incomplete; use --allow-partial "
            "only for explicitly labeled interim diagnostics"
        )

    all_rows: List[Dict[str, str]] = []
    fields: List[str] = []
    for row in rows:
        if row["status"] != "complete":
            continue
        result = _load_success_result(Path(row["successful_attempt"]))
        current_fields, current_rows = _read_csv_rows(result)
        if not fields:
            fields = current_fields
        elif fields != current_fields:
            raise StudyError(f"Collected CSV schema drift: {result}")
        all_rows.extend(current_rows)

    injection_pairing: Optional[Dict[str, Any]] = None
    if kind == "injection" and all_rows:
        injection_pairing = validate_collected_injection_pairing(all_rows)

    tag = "partial" if incomplete else "complete"
    selector_bits: List[str] = []
    if truths:
        selector_bits.append("truth-" + "-".join(sorted(set(map(str, truths)))))
    if factors:
        selector_bits.append(
            "factor-" + "-".join(f"{int(x):02d}" for x in sorted(set(factors)))
        )
    if scenarios:
        selector_bits.append(
            "scenario-" + "-".join(sorted(set(map(str, scenarios))))
        )
    selection_tag = "__".join(selector_bits)
    stem = str(kind) + (f"__{selection_tag}" if selection_tag else "")
    output_dir = STUDY_DIR / "derived"
    output = output_dir / f"{stem}_rows_{tag}.csv"
    if fields:
        _write_csv_rows(output, fields, all_rows)
    status_output = output_dir / f"{stem}_task_status_{tag}.csv"
    status_fields = list(rows[0].keys()) if rows else []
    if status_fields:
        _write_csv_rows(status_output, status_fields, rows)

    report = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "collected_utc": _utc_now(),
        "kind": kind,
        "selectors": {
            "factors": list(factors),
            "truths": list(truths),
            "scenarios": list(scenarios),
        },
        "partial": bool(incomplete),
        "completed_tasks": len(rows) - len(incomplete),
        "incomplete_tasks": len(incomplete),
        "rows": len(all_rows),
        "output": str(output) if fields else None,
        "output_sha256": _sha256_file(output) if fields else None,
        "task_status": str(status_output) if status_fields else None,
        "expected_limit_bands": False,
        "injection_signal_pairing": injection_pairing,
    }
    _atomic_write_json(output_dir / f"{stem}_collection_{tag}.json", report)
    return report


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mass-min-mev", type=int)
    parser.add_argument("--mass-max-mev", type=int)
    parser.add_argument("--mass-step-mev", type=int)


def _grid_from_args(spec: Mapping[str, Any], args: argparse.Namespace) -> List[float]:
    return mass_grid(
        spec,
        getattr(args, "mass_min_mev", None),
        getattr(args, "mass_max_mev", None),
        getattr(args, "mass_step_mev", None),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Read-only provenance and geometry checks")

    prepare_parser = sub.add_parser(
        "prepare", help="Generate paired toys, candidate configs, and task manifests"
    )
    prepare_parser.add_argument(
        "--force-toys",
        action="store_true",
        help="Regenerate paired toys after validating the reviewed source pins",
    )
    sub.add_parser("validate-toys", help="Validate paired counts and nesting")

    anchor_parser = sub.add_parser(
        "prepare-injection-anchors",
        help=(
            "Prepare resumable factor-15 prefit-Asimov absolute-amplitude "
            "anchor parts and consolidate their ledger"
        ),
    )
    anchor_parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement before starting factor-15 anchor fits",
    )
    anchor_parser.add_argument(
        "--max-parts",
        type=int,
        default=100,
        help="Maximum pending truth/scenario/background-toy anchor parts",
    )
    anchor_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel fresh subprocesses; each anchor process is single-threaded",
    )
    anchor_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected valid anchor parts before consolidation",
    )

    anchor_part_parser = sub.add_parser(
        "run-anchor-part",
        help="Run or dry-run one resumable factor-15 anchor part",
    )
    anchor_part_parser.add_argument("anchor_id")
    anchor_part_parser.add_argument("--execute", action="store_true")
    anchor_part_parser.add_argument("--force", action="store_true")
    anchor_part_parser.add_argument("--clear-stale-lock", action="store_true")

    status_parser = sub.add_parser("status", help="Summarize task completion")
    status_parser.add_argument("--kind", choices=("scan", "injection"))
    status_parser.add_argument("--factor", type=int, action="append", default=[])
    status_parser.add_argument("--truth", action="append", default=[])
    status_parser.add_argument("--scenario", action="append", default=[])
    _add_grid_arguments(status_parser)

    run_parser = sub.add_parser("run-task", help="Run or dry-run one manifest task")
    run_parser.add_argument("task_id")
    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement before starting fits",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new attempt even when a valid success exists",
    )
    run_parser.add_argument(
        "--clear-stale-lock",
        action="store_true",
        help="Clear a reviewed stale task lock before running",
    )
    _add_grid_arguments(run_parser)

    pending_parser = sub.add_parser(
        "run-pending", help="Run a bounded number of pending tasks in fresh processes"
    )
    pending_parser.add_argument("--kind", required=True, choices=("scan", "injection"))
    pending_parser.add_argument("--max-tasks", type=int, default=1)
    pending_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel fresh subprocesses; each fit process remains single-threaded",
    )
    pending_parser.add_argument("--factor", type=int, action="append", default=[])
    pending_parser.add_argument("--truth", action="append", default=[])
    pending_parser.add_argument("--scenario", action="append", default=[])
    pending_parser.add_argument("--execute", action="store_true")
    _add_grid_arguments(pending_parser)

    collect_parser = sub.add_parser(
        "collect", help="Fail-closed collection of enriched per-row outputs"
    )
    collect_parser.add_argument("--kind", required=True, choices=("scan", "injection"))
    collect_parser.add_argument("--allow-partial", action="store_true")
    collect_parser.add_argument("--factor", type=int, action="append", default=[])
    collect_parser.add_argument("--truth", action="append", default=[])
    collect_parser.add_argument("--scenario", action="append", default=[])
    _add_grid_arguments(collect_parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_spec()
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(spec), indent=2, sort_keys=True))
            return 0
        if args.command == "prepare":
            print(
                json.dumps(
                    prepare(spec, force_toys=bool(args.force_toys)),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-toys":
            print(json.dumps(validate_toys(spec), indent=2, sort_keys=True))
            return 0
        if args.command == "prepare-injection-anchors":
            print(
                json.dumps(
                    prepare_injection_anchors(
                        spec,
                        execute=bool(args.execute),
                        max_parts=int(args.max_parts),
                        workers=int(args.workers),
                        force=bool(args.force),
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run-anchor-part":
            identity = _anchor_identity_by_id(spec, args.anchor_id)
            if args.execute:
                _configure_fit_process()
                preflight(spec)
                validate_toys(spec)
                _activate_fit_code(spec)
            print(
                json.dumps(
                    run_anchor_part(
                        spec,
                        identity,
                        execute=bool(args.execute),
                        force=bool(args.force),
                        clear_stale_lock=bool(args.clear_stale_lock),
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        tasks = load_tasks(spec)
        masses = _grid_from_args(spec, args)
        if args.command == "status":
            rows = status_rows(
                spec,
                tasks,
                masses,
                kind=args.kind,
                factors=args.factor,
                truths=args.truth,
                scenarios=args.scenario,
            )
            _print_status_summary(rows)
            return 0
        if args.command == "run-task":
            task = _task_by_id(tasks, args.task_id)
            print(
                json.dumps(
                    run_task(
                        spec,
                        task,
                        masses,
                        execute=bool(args.execute),
                        force=bool(args.force),
                        clear_stale_lock=bool(args.clear_stale_lock),
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run-pending":
            print(
                json.dumps(
                    run_pending(
                        spec,
                        tasks,
                        masses,
                        args.kind,
                        args.max_tasks,
                        args.workers,
                        args.execute,
                        args.factor,
                        args.truth,
                        args.scenario,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "collect":
            print(
                json.dumps(
                    collect(
                        spec,
                        tasks,
                        masses,
                        args.kind,
                        bool(args.allow_partial),
                        factors=args.factor,
                        truths=args.truth,
                        scenarios=args.scenario,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        raise StudyError(f"Unknown command: {args.command}")
    except StudyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
