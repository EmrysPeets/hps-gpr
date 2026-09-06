#!/usr/bin/env python3
"""Validate and plot the complete v4.1 paired-exposure ensemble.

This is a postprocessor only.  It never imports the fit implementation and it
has no code path that can start a production fit.  Injection-dependent review
artifacts are written only after both complete collected CSVs pass exact
row-coverage, provenance, fit-success, geometry, paired-seed, and nested-bound
likelihood checks.  A separate scan-only mode validates the final
optimizer-audited scan bundle and never reads injection data.

The all-toy length-scale figures intentionally show every one of the ten raw
toy curves.  They do not contain quantile fills, expected-limit bands, or any
other uncertainty band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
SPEC_PATH = STUDY_DIR / "study_spec.json"
DEFAULT_REVIEWED_SCAN_CSV = (
    STUDY_DIR / "derived" / "scan_reviewed_rows_complete.csv"
)
DEFAULT_SCAN_CSV = DEFAULT_REVIEWED_SCAN_CSV
DEFAULT_INJECTION_CSV = STUDY_DIR / "derived" / "injection_rows_complete.csv"
DERIVED_DIR = STUDY_DIR / "derived"
PLOTS_DIR = STUDY_DIR / "plots"
TOY_MANIFEST_PATH = DERIVED_DIR / "paired_exposure_toy_manifest.json"
TOY_ROOT_PATH = STUDY_DIR / "inputs" / "paired_exposure_toys.root"
SCAN_AUDIT_SUMMARY_PATH = DERIVED_DIR / "scan_optimizer_audit_summary.json"
SOURCE_METADATA_PATHS = {
    "one_pct": (
        STUDY_DIR
        / "inputs"
        / "funcform_seed_2021_1pct_support040_300.root.metadata.json"
    ),
    "ten_pct": (
        STUDY_DIR
        / "inputs"
        / "funcform_seed_2021_10pct_support040_300.root.metadata.json"
    ),
}

SCHEMA_VERSION = 1
LML_ABS_TOLERANCE = 1.0e-4
LML_REL_TOLERANCE = 1.0e-6
FLOAT_RTOL = 1.0e-8
FLOAT_ATOL = 1.0e-10
BOUND_RTOL = 1.0e-6
PROJECTION_CANDIDATE_FACTOR = 20
SCENARIO_DISPLAY_LABELS = {
    "2021_1pct": "2021 1%",
    "2021_1pct_x10": "2021 1% × 10",
    "2021_1pct_x100": "2021 1% × 100",
    "2021_10pct": "2021 10%",
    "2021_10pct_x10": "2021 10% × 10",
}


class ReviewGateError(RuntimeError):
    """Raised when an input is incomplete, failed, or provenance-inconsistent."""


def _scenario_display_label(scenario: str) -> str:
    try:
        return SCENARIO_DISPLAY_LABELS[str(scenario)]
    except KeyError as exc:
        raise ReviewGateError(
            f"No reviewed display label for scenario {scenario!r}"
        ) from exc


def _nested_lml_tolerance(lml_low: float, lml_high: float) -> float:
    """Return the exact absolute-plus-relative tolerance frozen by the audit."""

    return max(
        LML_ABS_TOLERANCE,
        LML_REL_TOLERANCE * max(abs(lml_low), abs(lml_high), 1.0),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relocate_reviewed_path(value: Any) -> Path:
    """Resolve a recorded study path inside the current checkout.

    The immutable ledgers retain the absolute path of the production checkout.
    A GitHub clone or review worktree necessarily has a different prefix.  Only
    the prefix preceding ``study_results/<study-id>`` may change; the complete
    study-relative suffix and the independently checked file hashes remain
    fixed.
    """

    raw = Path(str(value)).expanduser()
    parts = raw.parts
    marker = ("study_results", STUDY_DIR.name)
    for index in range(len(parts) - 1):
        if tuple(parts[index : index + 2]) == marker:
            suffix = Path(*parts[index + 2 :])
            return (STUDY_DIR / suffix).resolve()
    return raw.resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ReviewGateError(f"Missing required JSON: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ReviewGateError(f"Expected a JSON object: {path}")
    return payload


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


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReviewGateError(
            f"{label} is missing {len(missing)} required column(s): "
            + ", ".join(missing)
        )


def _as_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = converted.isna()
        if bool(bad.any()):
            examples = frame.loc[bad, column].astype(str).head(5).tolist()
            raise ReviewGateError(
                f"{label}.{column} contains {int(bad.sum())} non-numeric or "
                f"missing value(s); examples={examples}"
            )
        frame[column] = converted


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    true_values = {"true", "1", "1.0", "yes"}
    false_values = {"false", "0", "0.0", "no"}
    normalized = series.astype(str).str.strip().str.lower()
    valid = normalized.isin(true_values | false_values)
    if not bool(valid.all()):
        examples = series.loc[~valid].astype(str).head(5).tolist()
        raise ReviewGateError(
            f"{label} contains unparseable boolean value(s); examples={examples}"
        )
    return normalized.isin(true_values)


def _require_all_true(frame: pd.DataFrame, column: str, label: str) -> None:
    parsed = _bool_series(frame[column], f"{label}.{column}")
    if not bool(parsed.all()):
        raise ReviewGateError(
            f"{label}.{column} has {int((~parsed).sum())} failed row(s)"
        )


def _require_all_false(frame: pd.DataFrame, column: str, label: str) -> None:
    parsed = _bool_series(frame[column], f"{label}.{column}")
    if bool(parsed.any()):
        raise ReviewGateError(
            f"{label}.{column} has {int(parsed.sum())} unexpectedly true row(s)"
        )


def _require_constant(
    frame: pd.DataFrame, column: str, expected: Any, label: str
) -> None:
    actual = frame[column].astype(str)
    mask = actual != str(expected)
    if bool(mask.any()):
        examples = sorted(actual.loc[mask].unique().tolist())[:5]
        raise ReviewGateError(
            f"{label}.{column} differs from {expected!r} in {int(mask.sum())} "
            f"row(s); examples={examples}"
        )


def _require_close(
    actual: pd.Series,
    expected: Any,
    label: str,
    *,
    rtol: float = FLOAT_RTOL,
    atol: float = FLOAT_ATOL,
) -> None:
    lhs = np.asarray(actual, dtype=float)
    rhs = np.asarray(expected, dtype=float)
    mask = ~np.isclose(lhs, rhs, rtol=rtol, atol=atol)
    if bool(np.any(mask)):
        indices = np.flatnonzero(mask)[:5]
        examples = [
            {
                "row": int(index),
                "actual": float(lhs[index]),
                "expected": float(rhs if rhs.ndim == 0 else rhs[index]),
            }
            for index in indices
        ]
        raise ReviewGateError(
            f"{label} differs in {int(np.count_nonzero(mask))} row(s); "
            f"examples={examples}"
        )


def _require_integral(series: pd.Series, label: str) -> None:
    values = np.asarray(series, dtype=float)
    rounded = np.rint(values)
    bad = (~np.isfinite(values)) | (values != rounded)
    if bool(np.any(bad)):
        examples = values[bad][:5].tolist()
        raise ReviewGateError(
            f"{label} has {int(np.count_nonzero(bad))} non-integral value(s); "
            f"examples={examples}"
        )


def _require_set(actual: Iterable[Any], expected: Iterable[Any], label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set, key=str)
        extra = sorted(actual_set - expected_set, key=str)
        raise ReviewGateError(
            f"{label} set mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )


def _read_complete_csv(path: Path, kind: str) -> pd.DataFrame:
    if "partial" in path.name.lower():
        raise ReviewGateError(f"Refusing explicitly partial {kind} table: {path}")
    if not path.is_file():
        raise ReviewGateError(
            f"Missing complete {kind} table: {path}. Run the collection step "
            "without --allow-partial after every selected task succeeds."
        )
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ReviewGateError(f"Could not read {kind} table {path}: {exc}") from exc
    if frame.empty:
        raise ReviewGateError(f"Complete {kind} table is empty: {path}")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ReviewGateError(
            f"{kind} table has duplicate column names: {duplicates}"
        )
    return frame


def _load_spec() -> Dict[str, Any]:
    spec = _load_json(SPEC_PATH)
    if int(spec.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ReviewGateError("Unsupported or missing study_spec schema_version")
    required = {
        "study_id",
        "n_toys",
        "length_scale_upper_factors",
        "default_mass_grid_mev",
        "truth_models",
        "scenarios",
        "injection_closure",
        "base_config",
        "fit_code",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ReviewGateError(f"study_spec is missing keys: {missing}")
    return spec


def _selected_truths(spec: Mapping[str, Any], requested: Sequence[str]) -> List[str]:
    available = sorted(map(str, spec["truth_models"]))
    if not requested:
        return available
    selected = list(dict.fromkeys(map(str, requested)))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ReviewGateError(
            f"Unknown --truth value(s): {unknown}; expected one of {available}"
        )
    return selected


def _mass_grid_mev(spec: Mapping[str, Any]) -> List[int]:
    grid = spec["default_mass_grid_mev"]
    lo, hi, step = int(grid["min"]), int(grid["max"]), int(grid["step"])
    if step <= 0 or hi < lo or (hi - lo) % step:
        raise ReviewGateError("Invalid default_mass_grid_mev in study_spec")
    return list(range(lo, hi + 1, step))


def _config_hashes(spec: Mapping[str, Any]) -> Dict[int, str]:
    hashes: Dict[int, str] = {}
    for raw_factor in spec["length_scale_upper_factors"]:
        factor = int(raw_factor)
        config = STUDY_DIR / "configs" / f"config_2021_lsupper_factor_{factor:02d}.yaml"
        provenance_path = config.with_suffix(".provenance.json")
        provenance = _load_json(provenance_path)
        if not config.is_file():
            raise ReviewGateError(f"Missing generated factor config: {config}")
        actual_sha = _sha256_file(config)
        expected_sha = str(provenance.get("generated_config_sha256", ""))
        if actual_sha != expected_sha:
            raise ReviewGateError(
                f"Generated factor-{factor} config hash drift: expected "
                f"{expected_sha}, got {actual_sha}"
            )
        if int(provenance.get("factor", -1)) != factor:
            raise ReviewGateError(
                f"Factor provenance mismatch in {provenance_path}"
            )
        if str(provenance.get("study_id")) != str(spec["study_id"]):
            raise ReviewGateError(f"Study-id drift in {provenance_path}")
        if str(provenance.get("base_config_sha256")) != str(
            spec["base_config"]["sha256"]
        ):
            raise ReviewGateError(f"Base-config hash drift in {provenance_path}")
        if str(provenance.get("fit_code_commit")) != str(
            spec["fit_code"]["commit"]
        ):
            raise ReviewGateError(f"Fit-code commit drift in {provenance_path}")
        if bool(provenance.get("execution_overrides", {}).get("expected_limit_bands")):
            raise ReviewGateError(
                f"Expected-limit bands were enabled in {provenance_path}"
            )
        hashes[factor] = actual_sha
    return hashes


def _validate_toy_manifest(spec: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = _load_json(TOY_MANIFEST_PATH)
    if str(manifest.get("study_id")) != str(spec["study_id"]):
        raise ReviewGateError("Paired-toy manifest study-id drift")
    if int(manifest.get("n_toys", -1)) != int(spec["n_toys"]):
        raise ReviewGateError("Paired-toy manifest n_toys drift")
    expected_rows = (
        int(spec["n_toys"])
        * len(spec["truth_models"])
        * len(spec["scenarios"])
    )
    if len(manifest.get("toys", [])) != expected_rows:
        raise ReviewGateError(
            f"Paired-toy manifest coverage mismatch: expected {expected_rows}, "
            f"got {len(manifest.get('toys', []))}"
        )
    manifest_root = _relocate_reviewed_path(manifest.get("toy_root", ""))
    if manifest_root != TOY_ROOT_PATH.resolve():
        raise ReviewGateError(
            f"Paired-toy manifest points to {manifest_root}, not {TOY_ROOT_PATH}"
        )
    if not TOY_ROOT_PATH.is_file():
        raise ReviewGateError(f"Missing paired-toy ROOT file: {TOY_ROOT_PATH}")
    actual_sha = _sha256_file(TOY_ROOT_PATH)
    if actual_sha != str(manifest.get("toy_root_sha256", "")):
        raise ReviewGateError(
            "Paired-toy ROOT hash differs from its reviewed manifest"
        )
    return manifest


def _validate_source_normalizations(
    spec: Mapping[str, Any], toy_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate source support and return source/effective count comparisons."""

    source_targets: Dict[str, int] = {}
    source_roots: Dict[str, Path] = {}
    for family, metadata_path in SOURCE_METADATA_PATHS.items():
        metadata = _load_json(metadata_path)
        if str(metadata.get("dataset", "")) != "2021":
            raise ReviewGateError(
                f"{family} source metadata dataset is not 2021"
            )
        if not np.allclose(
            np.asarray(metadata.get("toy_support_range_GeV", []), dtype=float),
            [0.04, 0.30],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ReviewGateError(
                f"{family} source metadata support is not 40--300 MeV"
            )
        if not np.allclose(
            np.asarray(metadata.get("scan_range_GeV", []), dtype=float),
            [0.05, 0.25],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ReviewGateError(
                f"{family} source metadata scan range is not 50--250 MeV"
            )
        target = int(metadata.get("normalization_target_count", -1))
        if target <= 0:
            raise ReviewGateError(
                f"{family} source normalization target is invalid: {target}"
            )
        source_targets[family] = target
        metadata_suffix = ".metadata.json"
        metadata_text = str(metadata_path.resolve())
        if not metadata_text.endswith(metadata_suffix):
            raise ReviewGateError(
                f"Unexpected source metadata filename: {metadata_path}"
            )
        source_root = Path(
            metadata_text[: -len(metadata_suffix)]
        ).resolve()
        if not source_root.is_file():
            raise ReviewGateError(
                f"{family} source ROOT file is missing: {source_root}"
            )
        source_roots[family] = source_root

    rows = toy_manifest.get("toys", [])
    if not isinstance(rows, list) or not rows:
        raise ReviewGateError("Paired-toy manifest has no toy records")
    toys = pd.DataFrame(rows)
    manifest_columns = [
        "truth_model",
        "scenario",
        "toy_index",
        "source_family",
        "source_root",
        "source_root_sha256",
        "expected_mean_total",
        "exposure_multiplier",
    ]
    _require_columns(toys, manifest_columns, "paired-toy manifest records")
    duplicate = toys.duplicated(
        ["truth_model", "scenario", "toy_index"], keep=False
    )
    if bool(duplicate.any()):
        raise ReviewGateError(
            "Paired-toy manifest has duplicate truth/scenario/toy records"
        )
    expected_keys = pd.MultiIndex.from_product(
        [
            list(map(str, spec["truth_models"])),
            list(map(str, spec["scenarios"])),
            list(range(int(spec["n_toys"]))),
        ],
        names=["truth_model", "scenario", "toy_index"],
    )
    actual_keys = pd.MultiIndex.from_frame(
        toys.loc[:, ["truth_model", "scenario", "toy_index"]]
    )
    if len(expected_keys.difference(actual_keys)) or len(
        actual_keys.difference(expected_keys)
    ):
        raise ReviewGateError(
            "Paired-toy manifest record coverage differs from study_spec"
        )

    for scenario, scenario_spec in spec["scenarios"].items():
        family = str(scenario_spec["source_family"])
        if family not in source_targets:
            raise ReviewGateError(
                f"No reviewed source metadata for source family {family}"
            )
        scenario_rows = toys.loc[toys["scenario"].astype(str) == str(scenario)]
        _require_constant(
            scenario_rows,
            "source_family",
            family,
            f"paired-toy manifest {scenario}",
        )
        expected_multiplier = int(scenario_spec["exposure_multiplier"])
        _require_close(
            pd.to_numeric(
                scenario_rows["exposure_multiplier"], errors="coerce"
            ),
            expected_multiplier,
            f"paired-toy manifest {scenario} exposure multiplier",
            rtol=0.0,
            atol=0.0,
        )
        expected_mean = source_targets[family] * expected_multiplier
        _require_close(
            pd.to_numeric(
                scenario_rows["expected_mean_total"], errors="coerce"
            ),
            expected_mean,
            f"paired-toy manifest {scenario} expected mean",
            rtol=1.0e-12,
            atol=1.0e-6,
        )
        resolved_roots = scenario_rows["source_root"].map(
            _relocate_reviewed_path
        )
        if bool((resolved_roots != source_roots[family]).any()):
            raise ReviewGateError(
                f"paired-toy manifest {scenario} source ROOT drift"
            )
        if scenario_rows["source_root_sha256"].astype(str).nunique() != 1:
            raise ReviewGateError(
                f"paired-toy manifest {scenario} source hash is not constant"
            )

    scaled_spec = spec["scenarios"].get("2021_1pct_x10", {})
    native_spec = spec["scenarios"].get("2021_10pct", {})
    if (
        scaled_spec.get("source_family") != "one_pct"
        or scaled_spec.get("parent") != "2021_1pct"
        or int(scaled_spec.get("exposure_multiplier", -1)) != 10
        or native_spec.get("source_family") != "ten_pct"
        or native_spec.get("parent") is not None
        or int(native_spec.get("exposure_multiplier", -1)) != 1
    ):
        raise ReviewGateError(
            "Native-10% versus 1%-source-x10 scenario lineage drift"
        )
    comparison_rows = toys.loc[
        toys["scenario"].astype(str).isin(
            ["2021_1pct_x10", "2021_10pct"]
        )
    ]
    hashes_by_family = comparison_rows.groupby("source_family")[
        "source_root_sha256"
    ].first()
    if hashes_by_family.nunique() != 2:
        raise ReviewGateError(
            "Native-10% and 1%-source-x10 do not have distinct source hashes"
        )

    one_pct_target = source_targets["one_pct"]
    ten_pct_target = source_targets["ten_pct"]
    return {
        "source_normalization_target_counts": source_targets,
        "source_support_ratio_ten_pct_over_one_pct": (
            ten_pct_target / one_pct_target
        ),
        "effective_target_counts": {
            "2021_1pct_x10": one_pct_target * 10,
            "2021_10pct": ten_pct_target,
        },
        "effective_target_ratio_native10_over_1pct_x10": (
            ten_pct_target / (one_pct_target * 10)
        ),
        "support_range_mev": [40, 300],
        "scan_range_mev": [50, 250],
    }


def _collection_report_path(csv_path: Path, kind: str) -> Path | None:
    suffix = "_rows_complete.csv"
    if not csv_path.name.endswith(suffix):
        raise ReviewGateError(
            f"{kind} input is not a collector-produced complete table: "
            f"{csv_path.name}"
        )
    stem = csv_path.name[: -len(suffix)]
    return csv_path.with_name(f"{stem}_collection_complete.json")


def _validate_collection_report(csv_path: Path, kind: str) -> Dict[str, Any] | None:
    report_path = _collection_report_path(csv_path, kind)
    assert report_path is not None
    report = _load_json(report_path)
    if str(report.get("study_id")) != _load_spec()["study_id"]:
        raise ReviewGateError(f"{kind} collection report study-id drift")
    if bool(report.get("partial", True)):
        raise ReviewGateError(f"{kind} collection report is marked partial")
    if int(report.get("incomplete_tasks", -1)) != 0:
        raise ReviewGateError(f"{kind} collection report has incomplete tasks")
    if str(report.get("kind")) != kind:
        raise ReviewGateError(f"{kind} collection report kind drift")
    report_output = _relocate_reviewed_path(report.get("output", ""))
    if report_output != csv_path.resolve():
        raise ReviewGateError(
            f"{kind} collection report points to {report_output}, not {csv_path}"
        )
    actual_sha = _sha256_file(csv_path)
    if str(report.get("output_sha256")) != actual_sha:
        raise ReviewGateError(
            f"{kind} collection CSV hash differs from its completion report"
        )
    return report


def _validate_reviewed_scan_bundle(
    csv_path: Path, spec: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate the collector handoff and the final optimizer-audit decision."""

    report = _validate_collection_report(csv_path, "scan")
    assert report is not None
    expected_tasks = (
        len(spec["length_scale_upper_factors"])
        * len(spec["truth_models"])
        * len(spec["scenarios"])
        * int(spec["n_toys"])
    )
    expected_rows = expected_tasks * len(_mass_grid_mev(spec))
    report_checks = {
        "schema_version": SCHEMA_VERSION,
        "review_stage": "optimizer_selected_actual_fit_rows",
        "fit_rows_are_actual": True,
        "interpolation_used": False,
        "expected_limit_bands": False,
        "completed_tasks": expected_tasks,
        "rows": expected_rows,
    }
    for key, expected in report_checks.items():
        if report.get(key) != expected:
            raise ReviewGateError(
                f"reviewed scan collection {key} drift: "
                f"expected {expected!r}, got {report.get(key)!r}"
            )
    source_path = _relocate_reviewed_path(report.get("source_output", ""))
    if not source_path.is_file():
        raise ReviewGateError(
            f"reviewed scan source output is missing: {source_path}"
        )
    source_sha = _sha256_file(source_path)
    if source_sha != str(report.get("source_output_sha256", "")):
        raise ReviewGateError(
            "reviewed scan source output hash differs from the collection report"
        )
    if source_sha != _sha256_file(csv_path):
        raise ReviewGateError(
            "reviewed complete scan and optimizer-selected source are not identical"
        )

    audit_path = SCAN_AUDIT_SUMMARY_PATH.resolve()
    reported_audit = _relocate_reviewed_path(
        report.get("optimizer_audit_summary", "")
    )
    if reported_audit != audit_path:
        raise ReviewGateError(
            f"reviewed scan collection points to {reported_audit}, not {audit_path}"
        )
    audit = _load_json(audit_path)
    audit_checks = {
        "schema_version": SCHEMA_VERSION,
        "study_id": spec["study_id"],
        "audit_gate": "optimizer_audit_pass",
        "reviewed_exact_rows": expected_rows,
        "selected_rows_in_ledger": expected_rows,
        "exact_initialization_locks": 0,
        "unresolved_initialization_states": 0,
        "unique_target_rows_requiring_repair": 0,
        "interpolation_used": False,
        "fit_rows_are_actual": True,
        "expected_limit_bands_constructed": False,
    }
    for key, expected in audit_checks.items():
        if audit.get(key) != expected:
            raise ReviewGateError(
                f"optimizer audit {key} drift: "
                f"expected {expected!r}, got {audit.get(key)!r}"
            )
    tolerance = audit.get("lml_tolerance", {})
    if not isinstance(tolerance, Mapping):
        raise ReviewGateError("optimizer audit lml_tolerance is malformed")
    expected_formula = (
        "max(abs_tol, rel_tol*max(abs(low_lml),abs(high_lml),1))"
    )
    if str(tolerance.get("formula", "")) != expected_formula:
        raise ReviewGateError("optimizer audit LML tolerance formula drift")
    if not math.isclose(
        float(tolerance.get("absolute", math.nan)),
        LML_ABS_TOLERANCE,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        float(tolerance.get("relative", math.nan)),
        LML_REL_TOLERANCE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ReviewGateError("optimizer audit LML tolerance constants drift")

    outputs = audit.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ReviewGateError("optimizer audit outputs are malformed")
    expected_output_paths = {
        "reviewed_rows": source_path,
        "reviewed_complete_rows": csv_path.resolve(),
        "reviewed_complete_collection": _collection_report_path(
            csv_path, "scan"
        ).resolve(),
    }
    for key, expected_path in expected_output_paths.items():
        actual_path = _relocate_reviewed_path(outputs.get(key, ""))
        if actual_path != expected_path:
            raise ReviewGateError(
                f"optimizer audit {key} points to {actual_path}, "
                f"not {expected_path}"
            )

    status_counts = audit.get("nested_pair_status_counts", {})
    if not isinstance(status_counts, Mapping):
        raise ReviewGateError(
            "optimizer audit nested_pair_status_counts is malformed"
        )
    unknown_statuses = set(status_counts) - {
        "allowed_domain_gain",
        "consistent_lml_plateau",
    }
    expected_pairs = (
        math.comb(len(spec["length_scale_upper_factors"]), 2)
        * len(spec["truth_models"])
        * len(spec["scenarios"])
        * int(spec["n_toys"])
        * len(_mass_grid_mev(spec))
    )
    if unknown_statuses or sum(map(int, status_counts.values())) != expected_pairs:
        raise ReviewGateError(
            "optimizer audit nested-pair status coverage drift: "
            f"unknown={sorted(unknown_statuses)}, "
            f"rows={sum(map(int, status_counts.values()))}, "
            f"expected={expected_pairs}"
        )
    return report, audit


COMMON_PROVENANCE_COLUMNS = [
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


SCAN_REQUIRED_COLUMNS = COMMON_PROVENANCE_COLUMNS + [
    "toy_index",
    "toy_hist",
    "function_tag",
    "source_model",
    "source_label",
    "source_root",
    "container",
    "dataset",
    "mass_GeV",
    "sigma_val",
    "A_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "extract_success",
    "cls_statistic",
    "cls_calibration",
    "visibility",
    "ls_lo",
    "ls_hi",
    "ls_opt",
    "sigma_x",
    "lml",
    "n_train",
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "ls_opt_over_sigma_x",
    "native_input_bin_width_gev",
    "production_rebin_requested",
    "rebinned_bin_width_gev",
    "rebinned_n_full",
    "rebinned_n_blind",
    "rebinned_n_train_expected",
    "rebinned_n_train_low",
    "rebinned_n_train_high",
    "training_geometry_valid",
    "optimizer_restarts_requested",
    "eps2_density_implementation",
    "eps2_up_promotable",
]

REVIEWED_SCAN_PROVENANCE_COLUMNS = [
    "fit_origin",
    "repair_config_path",
    "repair_config_sha256",
    "repair_fit_is_actual",
    "repair_interpolation_used",
    "actual_fit_selection",
    "initialization_lock",
    "initialization_state_unresolved",
]


INJECTION_REQUIRED_COLUMNS = COMMON_PROVENANCE_COLUMNS + [
    "toy_index",
    "toy_hist",
    "injection_toy",
    "toy",
    "function_tag",
    "source_model",
    "source_label",
    "source_root",
    "container",
    "dataset",
    "mass_GeV",
    "strength",
    "inj_nsigma",
    "injection_protocol",
    "injection_anchor_factor",
    "injection_anchor_nsigma",
    "injection_anchor_strength",
    "injection_anchor_sigmaA_ref",
    "injection_anchor_ledger_sha256",
    "injection_strength_mode",
    "signal_draw_sha256",
    "signal_draw_reference_sha256",
    "signal_draw_hash_verified",
    "signal_Nsig_full",
    "signal_Nsig_full_anchor",
    "signal_Nsig_win_anchor",
    "signal_Nsig_train_anchor",
    "signal_Nsig_win_matches_anchor",
    "signal_Nsig_train_matches_anchor",
    "sigmaA_ref",
    "sigmaA_ref_mode",
    "sigma_val",
    "sigma_x",
    "kernel_ls_res_upper_factor",
    "ls_lo",
    "ls_hi",
    "ls_opt",
    "refit_ls_opt",
    "inj_shape_mode",
    "inj_background_mode",
    "A_hat",
    "sigma_A",
    "Zhat",
    "pull_param",
    "Nsig_win",
    "Nsig_train",
    "success",
    "toy_mode",
    "refit_gp_on_toy",
    "refit_ok",
    "refit_restarts",
    "refit_optimize",
    "refit_fallback_used",
    "refit_error",
    "qmu_ok",
    "qmu_tilde",
    "tmu_tilde",
    "sqrt_qmu_tilde",
    "sqrt_tmu_tilde",
    "qmu_A_test",
    "qmu_branch",
    "qmu_nll_fixed",
    "qmu_nll_unbounded",
    "qmu_nll_null",
]


def _normalize_common_numeric(frame: pd.DataFrame, label: str) -> None:
    columns = [
        "exposure_multiplier",
        "ls_upper_factor_requested",
        "background_toy_index",
        "optimizer_seed",
    ]
    _as_numeric(
        frame,
        columns,
        label,
    )
    for column in columns:
        _require_integral(frame[column], f"{label}.{column}")


def _validate_reviewed_scan_provenance(
    frame: pd.DataFrame,
    config_hashes: Mapping[int, str],
) -> None:
    """Validate optimizer-selected actual rows without treating repairs as nominal.

    A targeted repair may use a salted optimizer seed and a row-specific YAML.
    Those differences are accepted only when the row is an explicitly selected
    actual fit, its repair YAML hash is exact, and no interpolation or unresolved
    initialization state survives.
    """

    _require_columns(
        frame, REVIEWED_SCAN_PROVENANCE_COLUMNS, "reviewed scan"
    )
    origins = frame["fit_origin"].astype(str)
    unknown_origins = set(origins) - {"nominal_scan", "targeted_repair"}
    if unknown_origins:
        raise ReviewGateError(
            "reviewed scan contains unknown fit_origin values: "
            f"{sorted(unknown_origins)}"
        )
    _require_constant(
        frame,
        "actual_fit_selection",
        "highest_finite_lml_actual_row",
        "reviewed scan",
    )
    _require_all_false(frame, "initialization_lock", "reviewed scan")
    _require_all_false(
        frame, "initialization_state_unresolved", "reviewed scan"
    )

    nominal = frame.loc[origins == "nominal_scan"]
    for factor, expected_sha in sorted(config_hashes.items()):
        factor_rows = nominal.loc[
            nominal["ls_upper_factor_requested"].astype(int) == int(factor)
        ]
        if not factor_rows.empty:
            _require_constant(
                factor_rows,
                "generated_config_sha256",
                expected_sha,
                f"reviewed scan nominal factor_{factor}",
            )

    repairs = frame.loc[origins == "targeted_repair"]
    if repairs.empty:
        return
    _require_all_true(repairs, "repair_fit_is_actual", "reviewed scan repair")
    _require_all_false(
        repairs, "repair_interpolation_used", "reviewed scan repair"
    )
    generated_sha = repairs["generated_config_sha256"].astype(str)
    repair_sha = repairs["repair_config_sha256"].astype(str)
    malformed_sha = ~repair_sha.str.fullmatch(r"[0-9a-f]{64}")
    if bool(malformed_sha.any()):
        examples = repair_sha.loc[malformed_sha].head(5).tolist()
        raise ReviewGateError(
            "reviewed scan repair_config_sha256 contains malformed values: "
            f"{examples}"
        )
    mismatch = generated_sha != repair_sha
    if bool(mismatch.any()):
        examples = repairs.loc[
            mismatch,
            [
                "task_id",
                "generated_config_sha256",
                "repair_config_sha256",
            ],
        ].head(5)
        raise ReviewGateError(
            "reviewed scan repair config hash differs from the selected fit "
            f"in {int(mismatch.sum())} row(s): "
            f"{examples.to_dict(orient='records')}"
        )

    checked_paths: Dict[str, str] = {}
    for row in repairs.loc[
        :, ["repair_config_path", "repair_config_sha256"]
    ].drop_duplicates().itertuples(index=False):
        raw_path = str(row.repair_config_path).strip()
        if not raw_path or raw_path.lower() in {"nan", "none"}:
            raise ReviewGateError(
                "reviewed scan repair row is missing repair_config_path"
            )
        config_path = _relocate_reviewed_path(raw_path)
        try:
            config_path.relative_to(STUDY_DIR.resolve())
        except ValueError as exc:
            raise ReviewGateError(
                f"reviewed scan repair config escaped the study: {config_path}"
            ) from exc
        if not config_path.is_file():
            raise ReviewGateError(
                f"reviewed scan repair config is missing: {config_path}"
            )
        expected_sha = str(row.repair_config_sha256)
        actual_sha = checked_paths.setdefault(
            str(config_path), _sha256_file(config_path)
        )
        if actual_sha != expected_sha:
            raise ReviewGateError(
                "reviewed scan repair config hash drift: "
                f"{config_path}; expected {expected_sha}, got {actual_sha}"
            )


def _validate_common_provenance(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    kind: str,
    selected_truths: Sequence[str],
    config_hashes: Mapping[int, str],
    *,
    allow_reviewed_repairs: bool = False,
) -> None:
    label = kind
    _require_constant(frame, "study_id", spec["study_id"], label)
    _require_constant(
        frame, "base_config_sha256", spec["base_config"]["sha256"], label
    )
    _require_constant(frame, "fit_code_commit", spec["fit_code"]["commit"], label)
    _require_all_false(frame, "expected_limit_bands", label)
    _require_set(frame["truth_model"].astype(str), selected_truths, f"{label}.truth_model")
    _require_set(
        frame["study_scenario"].astype(str),
        map(str, spec["scenarios"]),
        f"{label}.study_scenario",
    )
    _require_set(
        frame["ls_upper_factor_requested"].astype(int),
        map(int, spec["length_scale_upper_factors"]),
        f"{label}.ls_upper_factor_requested",
    )

    expected_toys = range(int(spec["n_toys"]))
    _require_set(
        frame["background_toy_index"].astype(int),
        expected_toys,
        f"{label}.background_toy_index",
    )

    for truth in selected_truths:
        truth_spec = spec["truth_models"][truth]
        sub = frame.loc[frame["truth_model"].astype(str) == truth]
        _require_constant(
            sub, "truth_function_tag", truth_spec["function_tag"], f"{label}.{truth}"
        )
        _require_constant(
            sub, "function_tag", truth_spec["function_tag"], f"{label}.{truth}"
        )
        _require_constant(
            sub, "source_label", truth_spec["function_tag"], f"{label}.{truth}"
        )

    for scenario, scenario_spec in spec["scenarios"].items():
        sub = frame.loc[frame["study_scenario"].astype(str) == str(scenario)]
        _require_constant(
            sub,
            "source_family",
            scenario_spec["source_family"],
            f"{label}.{scenario}",
        )
        _require_close(
            sub["exposure_multiplier"],
            float(scenario_spec["exposure_multiplier"]),
            f"{label}.{scenario}.exposure_multiplier",
        )

    if allow_reviewed_repairs:
        if kind != "scan":
            raise ReviewGateError(
                "Reviewed optimizer repairs are supported only for scan rows"
            )
        _validate_reviewed_scan_provenance(frame, config_hashes)
    else:
        for factor, expected_sha in sorted(config_hashes.items()):
            sub = frame.loc[
                frame["ls_upper_factor_requested"].astype(int) == int(factor)
            ]
            _require_constant(
                sub,
                "generated_config_sha256",
                expected_sha,
                f"{label}.factor_{factor}",
            )

    expected_task_id = frame.apply(
        lambda row: (
            f"{kind}__f{int(row['ls_upper_factor_requested']):02d}__"
            f"{row['truth_model']}__{row['study_scenario']}__"
            f"t{int(row['background_toy_index']):04d}"
        ),
        axis=1,
    )
    mismatch = frame["task_id"].astype(str) != expected_task_id.astype(str)
    if bool(mismatch.any()):
        examples = frame.loc[
            mismatch,
            [
                "task_id",
                "ls_upper_factor_requested",
                "truth_model",
                "study_scenario",
                "background_toy_index",
            ],
        ].head(5)
        raise ReviewGateError(
            f"{label}.task_id has {int(mismatch.sum())} inconsistent row(s): "
            f"{examples.to_dict(orient='records')}"
        )

    expected_container = (
        "toys/"
        + frame["truth_model"].astype(str)
        + "/"
        + frame["study_scenario"].astype(str)
    )
    mismatch = frame["container"].astype(str) != expected_container
    if bool(mismatch.any()):
        raise ReviewGateError(
            f"{label}.container has {int(mismatch.sum())} inconsistent row(s)"
        )

    expected_toy_hist = frame["background_toy_index"].map(
        lambda value: f"toy_{int(value):04d}"
    )
    mismatch = frame["toy_hist"].astype(str) != expected_toy_hist
    if bool(mismatch.any()):
        raise ReviewGateError(
            f"{label}.toy_hist has {int(mismatch.sum())} inconsistent row(s)"
        )

    _require_close(
        frame["toy_index"],
        frame["background_toy_index"],
        f"{label}.toy_index/background_toy_index",
        rtol=0,
        atol=0,
    )
    _require_constant(frame, "source_model", "functional_form", label)
    _require_constant(frame, "dataset", "2021", label)
    expected_source_root = TOY_ROOT_PATH.resolve()
    resolved_source_roots = frame["source_root"].map(
        _relocate_reviewed_path
    )
    source_mismatch = resolved_source_roots != expected_source_root
    if bool(source_mismatch.any()):
        examples = (
            frame.loc[source_mismatch, "source_root"].astype(str).head(5).tolist()
        )
        raise ReviewGateError(
            f"{label}.source_root escaped the paired exposure toy file in "
            f"{int(source_mismatch.sum())} row(s); examples={examples}"
        )

    if not allow_reviewed_repairs:
        seed_groups = [
            "truth_model",
            "study_scenario",
            "background_toy_index",
        ]
        if kind in {"scan", "injection"}:
            seed_groups.append("mass_mev")
        seed_counts = frame.groupby(seed_groups, dropna=False)[
            "optimizer_seed"
        ].nunique()
        if bool((seed_counts != 1).any()):
            bad = seed_counts.loc[seed_counts != 1].head(5).to_dict()
            raise ReviewGateError(
                f"{label} optimizer seeds are not paired across factors: {bad}"
            )


def _expected_product(
    factors: Sequence[int],
    truths: Sequence[str],
    scenarios: Sequence[str],
    toys: Sequence[int],
    masses_mev: Sequence[int],
    strengths: Sequence[float] | None = None,
    replicas: Sequence[int] | None = None,
) -> pd.MultiIndex:
    arrays: List[Sequence[Any]] = [
        factors,
        truths,
        scenarios,
        toys,
        masses_mev,
    ]
    names = [
        "factor",
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
    ]
    if strengths is not None:
        arrays.append(strengths)
        names.append("inj_nsigma")
    if replicas is not None:
        arrays.append(replicas)
        names.append("injection_toy")
    return pd.MultiIndex.from_product(arrays, names=names)


def _validate_exact_keys(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    expected: pd.MultiIndex,
    label: str,
) -> None:
    duplicates = frame.duplicated(list(key_columns), keep=False)
    if bool(duplicates.any()):
        examples = (
            frame.loc[duplicates, list(key_columns)].head(10).to_dict(orient="records")
        )
        raise ReviewGateError(
            f"{label} has {int(duplicates.sum())} rows in duplicate key groups; "
            f"examples={examples}"
        )
    actual = pd.MultiIndex.from_frame(frame.loc[:, list(key_columns)])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        raise ReviewGateError(
            f"{label} Cartesian coverage mismatch: expected={len(expected)}, "
            f"actual={len(actual)}, missing={len(missing)} "
            f"(examples={missing.tolist()[:5]}), extra={len(extra)} "
            f"(examples={extra.tolist()[:5]})"
        )
    if len(frame) != len(expected):
        raise ReviewGateError(
            f"{label} row count mismatch: expected {len(expected)}, got {len(frame)}"
        )


def _reject_unknown_categories(
    frame: pd.DataFrame, spec: Mapping[str, Any], label: str
) -> None:
    unknown_truths = set(frame["truth_model"].astype(str)) - set(
        map(str, spec["truth_models"])
    )
    unknown_scenarios = set(frame["study_scenario"].astype(str)) - set(
        map(str, spec["scenarios"])
    )
    factors = pd.to_numeric(
        frame["ls_upper_factor_requested"], errors="coerce"
    )
    if bool(factors.isna().any()):
        raise ReviewGateError(
            f"{label}.ls_upper_factor_requested contains non-numeric values"
        )
    _require_integral(factors, f"{label}.ls_upper_factor_requested")
    unknown_factors = set(factors.astype(int)) - set(
        map(int, spec["length_scale_upper_factors"])
    )
    if unknown_truths or unknown_scenarios or unknown_factors:
        raise ReviewGateError(
            f"{label} contains out-of-spec categories: "
            f"truths={sorted(unknown_truths)}, "
            f"scenarios={sorted(unknown_scenarios)}, "
            f"factors={sorted(unknown_factors)}"
        )


def validate_scan(
    raw: pd.DataFrame,
    spec: Mapping[str, Any],
    selected_truths: Sequence[str],
    config_hashes: Mapping[int, str],
    *,
    allow_reviewed_repairs: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(raw, SCAN_REQUIRED_COLUMNS, "scan")
    _reject_unknown_categories(raw, spec, "scan")
    frame = raw.loc[raw["truth_model"].astype(str).isin(selected_truths)].copy()
    if frame.empty:
        raise ReviewGateError("Scan table has no rows for the selected truth models")
    numeric = [
        "toy_index",
        "mass_GeV",
        "sigma_val",
        "A_up",
        "p0_analytic",
        "Z_analytic",
        "A_hat",
        "sigma_A",
        "ls_lo",
        "ls_hi",
        "ls_opt",
        "sigma_x",
        "lml",
        "n_train",
        "ls_lo_over_sigma_x",
        "ls_hi_over_sigma_x",
        "ls_opt_over_sigma_x",
        "native_input_bin_width_gev",
        "production_rebin_requested",
        "rebinned_bin_width_gev",
        "rebinned_n_full",
        "rebinned_n_blind",
        "rebinned_n_train_expected",
        "rebinned_n_train_low",
        "rebinned_n_train_high",
        "optimizer_restarts_requested",
    ]
    _normalize_common_numeric(frame, "scan")
    _as_numeric(frame, numeric, "scan")
    for column in [
        "toy_index",
        "n_train",
        "production_rebin_requested",
        "rebinned_n_full",
        "rebinned_n_blind",
        "rebinned_n_train_expected",
        "rebinned_n_train_low",
        "rebinned_n_train_high",
        "optimizer_restarts_requested",
    ]:
        _require_integral(frame[column], f"scan.{column}")
    frame["factor"] = frame["ls_upper_factor_requested"].astype(int)
    frame["mass_mev"] = np.rint(frame["mass_GeV"] * 1000.0).astype(int)

    _validate_common_provenance(
        frame,
        spec,
        "scan",
        selected_truths,
        config_hashes,
        allow_reviewed_repairs=allow_reviewed_repairs,
    )
    factors = list(map(int, spec["length_scale_upper_factors"]))
    scenarios = list(map(str, spec["scenarios"]))
    toys = list(range(int(spec["n_toys"])))
    masses_mev = _mass_grid_mev(spec)
    expected = _expected_product(
        factors, selected_truths, scenarios, toys, masses_mev
    )
    scan_keys = [
        "factor",
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
    ]
    _validate_exact_keys(frame, scan_keys, expected, "scan")
    _require_close(
        frame["mass_GeV"],
        frame["mass_mev"] / 1000.0,
        "scan.mass_GeV/mass_mev",
    )

    _require_all_true(frame, "extract_success", "scan")
    _require_all_true(frame, "training_geometry_valid", "scan")
    _require_all_false(frame, "eps2_up_promotable", "scan")
    _require_constant(frame, "cls_statistic", "tilde_q_mu", "scan")
    _require_constant(frame, "cls_calibration", "asymptotic", "scan")
    _require_constant(frame, "visibility", "observed", "scan")
    _require_constant(
        frame,
        "eps2_density_implementation",
        "immutable_df4d456_rebinned_whole_bin",
        "scan",
    )

    finite_positive = ["sigma_val", "sigma_A", "sigma_x", "A_up"]
    for column in finite_positive:
        bad = (~np.isfinite(frame[column])) | (frame[column] <= 0)
        if bool(bad.any()):
            raise ReviewGateError(
                f"scan.{column} has {int(bad.sum())} non-finite/non-positive row(s)"
            )
    for column in ["A_hat", "p0_analytic", "Z_analytic", "lml", "ls_lo", "ls_hi", "ls_opt"]:
        bad = ~np.isfinite(frame[column])
        if bool(bad.any()):
            raise ReviewGateError(
                f"scan.{column} has {int(bad.sum())} non-finite row(s)"
            )
    invalid_p0 = (frame["p0_analytic"] < -FLOAT_ATOL) | (
        frame["p0_analytic"] > 0.5 + FLOAT_ATOL
    )
    if bool(invalid_p0.any()):
        raise ReviewGateError(
            f"scan.p0_analytic has {int(invalid_p0.sum())} out-of-range row(s)"
        )

    _require_close(
        frame["ls_lo_over_sigma_x"],
        frame["ls_lo"] / frame["sigma_x"],
        "scan.ls_lo/sigma_x",
    )
    _require_close(
        frame["ls_hi_over_sigma_x"],
        frame["ls_hi"] / frame["sigma_x"],
        "scan.ls_hi/sigma_x",
    )
    _require_close(
        frame["ls_opt_over_sigma_x"],
        frame["ls_opt"] / frame["sigma_x"],
        "scan.ls_opt/sigma_x",
    )
    _require_close(
        frame["ls_hi_over_sigma_x"],
        frame["factor"],
        "scan.realized upper factor",
    )
    lower_violation = frame["ls_opt"] < frame["ls_lo"] * (1.0 - BOUND_RTOL)
    upper_violation = frame["ls_opt"] > frame["ls_hi"] * (1.0 + BOUND_RTOL)
    if bool((lower_violation | upper_violation).any()):
        raise ReviewGateError(
            "scan optimized length scale escaped its configured bounds in "
            f"{int((lower_violation | upper_violation).sum())} row(s)"
        )

    _require_close(
        frame["native_input_bin_width_gev"],
        0.000125,
        "scan.native input bin width",
        rtol=0,
        atol=1.0e-12,
    )
    _require_close(
        frame["production_rebin_requested"],
        5,
        "scan.production rebin",
        rtol=0,
        atol=0,
    )
    _require_close(
        frame["rebinned_bin_width_gev"],
        0.000625,
        "scan.rebinned bin width",
        rtol=0,
        atol=1.0e-10,
    )
    _require_close(
        frame["rebinned_n_full"], 416, "scan.rebinned_n_full", rtol=0, atol=0
    )
    _require_close(
        frame["n_train"],
        frame["rebinned_n_train_expected"],
        "scan.n_train accounting",
        rtol=0,
        atol=0,
    )
    _require_close(
        frame["rebinned_n_train_expected"],
        frame["rebinned_n_train_low"] + frame["rebinned_n_train_high"],
        "scan.training-side accounting",
        rtol=0,
        atol=0,
    )
    _require_close(
        frame["optimizer_restarts_requested"],
        12,
        "scan.optimizer restarts",
        rtol=0,
        atol=0,
    )

    bound_tolerance = BOUND_RTOL * np.maximum(1.0, frame["factor"].astype(float))
    frame["at_upper_bound"] = (
        frame["ls_opt_over_sigma_x"]
        >= frame["factor"].astype(float) - bound_tolerance
    )

    lml_keys = [
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
    ]
    wide = frame.pivot(index=lml_keys, columns="factor", values="lml")
    if list(wide.columns) != factors:
        wide = wide.reindex(columns=factors)
    audit_rows: List[Dict[str, Any]] = []
    for previous, current in zip(factors[:-1], factors[1:]):
        delta = wide[current] - wide[previous]
        for key, value in delta.items():
            lml_previous = float(wide.loc[key, previous])
            lml_current = float(wide.loc[key, current])
            tolerance = _nested_lml_tolerance(lml_previous, lml_current)
            audit_rows.append(
                {
                    "truth_model": key[0],
                    "study_scenario": key[1],
                    "background_toy_index": int(key[2]),
                    "mass_mev": int(key[3]),
                    "factor_previous": int(previous),
                    "factor_current": int(current),
                    "lml_previous": lml_previous,
                    "lml_current": lml_current,
                    "delta_lml": float(value),
                    "lml_tolerance": float(tolerance),
                    "regression_beyond_tolerance": bool(
                        float(value) < -tolerance
                    ),
                }
            )
    audit = pd.DataFrame(audit_rows)
    regressions = audit["regression_beyond_tolerance"].astype(bool)
    if bool(regressions.any()):
        examples = (
            audit.loc[regressions]
            .sort_values("delta_lml")
            .head(10)
            .to_dict(orient="records")
        )
        raise ReviewGateError(
            "Nested upper-bound likelihood gate failed: a larger feasible "
            "domain lowered the optimized LML beyond "
            f"max({LML_ABS_TOLERANCE:g}, {LML_REL_TOLERANCE:g} * "
            "max(abs(LML_low), abs(LML_high), 1)) "
            f"in {int(regressions.sum())} comparison(s). Rerun those exact "
            f"unchanged-card points; do not interpolate. Examples={examples}"
        )
    return frame, audit


def _assign_fixed_amplitude_levels(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    """Verify explicit factor-15 anchor metadata and attach paired diagnostics.

    The runner must emit the anchor level and ledger identity.  Inferring the
    level by sorting strengths would silently relabel malformed or missing
    injections, so this postprocessor deliberately does not do that.
    """

    closure = spec["injection_closure"]
    expected_protocol = "factor15_prefit_asimov_absolute_v1"
    if str(closure.get("protocol", "")) != expected_protocol:
        raise ReviewGateError(
            "study_spec injection protocol is not "
            f"{expected_protocol}"
        )
    labels = sorted(map(float, closure["sigma_strengths"]))
    if len(labels) != len(set(labels)) or not labels:
        raise ReviewGateError(
            "injection_closure.sigma_strengths must be nonempty and unique"
        )
    if not math.isclose(labels[0], 0.0, rel_tol=0.0, abs_tol=FLOAT_ATOL):
        raise ReviewGateError(
            "Fixed-amplitude protocol requires a zero-signal anchor level"
        )
    anchor_factor = int(
        closure.get(
            "fixed_amplitude_anchor_factor",
            closure.get(
                "amplitude_anchor_factor",
                closure.get("anchor_factor", 15),
            ),
        )
    )
    if anchor_factor != 15:
        raise ReviewGateError(
            f"Protocol factor drift: expected anchor factor 15, got {anchor_factor}"
        )
    factors = set(map(int, spec["length_scale_upper_factors"]))
    if anchor_factor not in factors:
        raise ReviewGateError(
            f"Fixed-amplitude anchor factor {anchor_factor} is not a candidate"
        )
    _require_constant(
        frame,
        "injection_protocol",
        expected_protocol,
        "injection",
    )
    _require_close(
        frame["injection_anchor_factor"],
        anchor_factor,
        "injection.injection_anchor_factor",
        rtol=0,
        atol=0,
    )

    normalized_levels = np.empty(len(frame), dtype=float)
    for index, raw_level in enumerate(frame["injection_anchor_nsigma"]):
        matches = [
            level
            for level in labels
            if math.isclose(
                float(raw_level),
                level,
                rel_tol=FLOAT_RTOL,
                abs_tol=FLOAT_ATOL,
            )
        ]
        if len(matches) != 1:
            raise ReviewGateError(
                "injection.injection_anchor_nsigma contains unexpected value "
                f"{raw_level}"
            )
        normalized_levels[index] = matches[0]
    frame = frame.copy()
    frame["anchor_nsigma"] = normalized_levels

    ledger_values = frame["injection_anchor_ledger_sha256"].astype(str)
    ledger_valid = ledger_values.str.fullmatch(r"[0-9a-f]{64}")
    if not bool(ledger_valid.all()):
        examples = ledger_values.loc[~ledger_valid].head(5).tolist()
        raise ReviewGateError(
            "injection anchor ledger SHA-256 is missing or malformed; "
            f"examples={examples}"
        )
    if ledger_values.nunique() != 1:
        raise ReviewGateError(
            "Injection rows do not share one common factor-15 anchor ledger SHA-256"
        )

    _require_close(
        frame["strength"],
        frame["injection_anchor_strength"],
        "injection absolute strength/anchor strength",
        rtol=1.0e-10,
        atol=1.0e-8,
    )
    _require_close(
        frame["injection_anchor_strength"],
        frame["anchor_nsigma"] * frame["injection_anchor_sigmaA_ref"],
        "injection factor-15 anchor amplitude equation",
        rtol=1.0e-8,
        atol=1.0e-8,
    )

    base = [
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
        "injection_toy",
        "anchor_nsigma",
    ]
    for column in [
        "injection_anchor_strength",
        "injection_anchor_sigmaA_ref",
        "injection_anchor_ledger_sha256",
    ]:
        counts = frame.groupby(base, dropna=False)[column].nunique()
        if bool((counts != 1).any()):
            examples = counts.loc[counts != 1].head(10).to_dict()
            raise ReviewGateError(
                f"injection.{column} is not paired across factors: {examples}"
            )

    anchor_rows = frame.loc[frame["factor"] == anchor_factor]
    _require_close(
        anchor_rows["inj_nsigma"],
        anchor_rows["anchor_nsigma"],
        "injection factor-15 inj_nsigma/anchor_nsigma",
    )
    _require_close(
        anchor_rows["sigmaA_ref"],
        anchor_rows["injection_anchor_sigmaA_ref"],
        "injection factor-15 sigmaA_ref/anchor_sigmaA_ref",
    )

    anchor_columns = base + ["sigma_A"]
    anchors = anchor_rows.loc[:, anchor_columns].rename(
        columns={
            "sigma_A": "anchor_sigma_A",
        }
    )
    anchor_duplicates = anchors.duplicated(base, keep=False)
    if bool(anchor_duplicates.any()):
        raise ReviewGateError(
            "Factor-15 amplitude anchors are not unique for every paired row"
        )
    frame = frame.merge(anchors, on=base, how="left", validate="many_to_one")
    missing_anchor = frame["anchor_sigma_A"].isna()
    if bool(missing_anchor.any()):
        raise ReviewGateError(
            f"Missing factor-{anchor_factor} amplitude companion for "
            f"{int(missing_anchor.sum())} injection row(s)"
        )
    for column in [
        "injection_anchor_sigmaA_ref",
        "anchor_sigma_A",
    ]:
        bad = (~np.isfinite(frame[column])) | (frame[column] <= 0)
        if bool(bad.any()):
            raise ReviewGateError(
                f"injection.{column} has {int(bad.sum())} invalid row(s)"
            )
    frame["sigma_A_over_anchor"] = frame["sigma_A"] / frame["anchor_sigma_A"]
    frame["sigmaA_ref_over_anchor"] = (
        frame["sigmaA_ref"] / frame["injection_anchor_sigmaA_ref"]
    )
    frame["Ahat_minus_Ainj_over_anchor_sigma"] = (
        frame["A_hat"] - frame["strength"]
    ) / frame["anchor_sigma_A"]
    frame["anchor_strength"] = frame["injection_anchor_strength"]
    frame["anchor_sigmaA_ref"] = frame["injection_anchor_sigmaA_ref"]
    frame["fixed_amplitude_anchor_factor"] = anchor_factor
    return frame


def _annotate_qmu_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Record coherent one-sided qmu exceptions without promoting qmu outputs."""

    qmu_ok = _bool_series(frame["qmu_ok"], "injection.qmu_ok")
    failed = ~qmu_ok
    if bool(failed.any()):
        failed_rows = frame.loc[failed]
        coherent = (
            failed_rows["qmu_branch"].astype(str).eq("muhat_gt_test")
            & np.isclose(
                failed_rows["qmu_tilde"].astype(float),
                0.0,
                rtol=0.0,
                atol=FLOAT_ATOL,
            )
            & np.isclose(
                failed_rows["tmu_tilde"].astype(float),
                0.0,
                rtol=0.0,
                atol=FLOAT_ATOL,
            )
            & np.isclose(
                failed_rows["sqrt_qmu_tilde"].astype(float),
                0.0,
                rtol=0.0,
                atol=FLOAT_ATOL,
            )
            & np.isclose(
                failed_rows["sqrt_tmu_tilde"].astype(float),
                0.0,
                rtol=0.0,
                atol=FLOAT_ATOL,
            )
            & (
                failed_rows["A_hat"].astype(float)
                > failed_rows["qmu_A_test"].astype(float)
            )
        )
        if not bool(coherent.all()):
            examples = failed_rows.loc[
                ~coherent,
                [
                    "task_id",
                    "mass_GeV",
                    "strength",
                    "A_hat",
                    "qmu_A_test",
                    "qmu_branch",
                    "qmu_tilde",
                    "tmu_tilde",
                ],
            ].head(10)
            raise ReviewGateError(
                "injection.qmu_ok has incoherent failed row(s): "
                f"{examples.to_dict(orient='records')}"
            )
    frame["qmu_ok_parsed"] = qmu_ok
    frame["qmu_one_sided_zero_branch_diagnostic"] = failed
    frame["qmu_outputs_used_in_postprocess"] = False
    frame["qmu_outputs_promotable"] = False
    return frame


def validate_injection(
    raw: pd.DataFrame,
    spec: Mapping[str, Any],
    selected_truths: Sequence[str],
    config_hashes: Mapping[int, str],
) -> pd.DataFrame:
    _require_columns(raw, INJECTION_REQUIRED_COLUMNS, "injection")
    _reject_unknown_categories(raw, spec, "injection")
    frame = raw.loc[raw["truth_model"].astype(str).isin(selected_truths)].copy()
    if frame.empty:
        raise ReviewGateError(
            "Injection table has no rows for the selected truth models"
        )
    numeric = [
        "toy_index",
        "injection_toy",
        "toy",
        "mass_GeV",
        "strength",
        "inj_nsigma",
        "injection_anchor_factor",
        "injection_anchor_nsigma",
        "injection_anchor_strength",
        "injection_anchor_sigmaA_ref",
        "sigmaA_ref",
        "sigma_val",
        "sigma_x",
        "kernel_ls_res_upper_factor",
        "ls_lo",
        "ls_hi",
        "ls_opt",
        "refit_ls_opt",
        "A_hat",
        "sigma_A",
        "Zhat",
        "pull_param",
        "Nsig_win",
        "Nsig_train",
        "signal_Nsig_full",
        "signal_Nsig_full_anchor",
        "signal_Nsig_win_anchor",
        "signal_Nsig_train_anchor",
        "refit_ok",
        "refit_restarts",
        "qmu_tilde",
        "tmu_tilde",
        "sqrt_qmu_tilde",
        "sqrt_tmu_tilde",
        "qmu_A_test",
        "qmu_nll_fixed",
        "qmu_nll_unbounded",
        "qmu_nll_null",
    ]
    _normalize_common_numeric(frame, "injection")
    _as_numeric(frame, numeric, "injection")
    _as_numeric(frame, ["injection_seed"], "injection")
    for column in [
        "toy_index",
        "injection_toy",
        "toy",
        "injection_anchor_factor",
        "refit_restarts",
        "Nsig_win",
        "Nsig_train",
        "signal_Nsig_full",
        "signal_Nsig_full_anchor",
        "signal_Nsig_win_anchor",
        "signal_Nsig_train_anchor",
        "injection_seed",
    ]:
        _require_integral(frame[column], f"injection.{column}")
    frame["factor"] = frame["ls_upper_factor_requested"].astype(int)
    frame["mass_mev"] = np.rint(frame["mass_GeV"] * 1000.0).astype(int)

    _validate_common_provenance(
        frame, spec, "injection", selected_truths, config_hashes
    )
    closure = spec["injection_closure"]
    factors = list(map(int, spec["length_scale_upper_factors"]))
    scenarios = list(map(str, spec["scenarios"]))
    toys = list(range(int(spec["n_toys"])))
    masses_mev = [int(round(float(value) * 1000.0)) for value in closure["masses_gev"]]
    strengths = list(map(float, closure["sigma_strengths"]))
    replicas = list(range(int(closure["replicas_per_background_toy"])))
    frame = _assign_fixed_amplitude_levels(frame, spec)

    expected = _expected_product(
        factors,
        selected_truths,
        scenarios,
        toys,
        masses_mev,
        strengths,
        replicas,
    )
    injection_keys = [
        "factor",
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
        "anchor_nsigma",
        "injection_toy",
    ]
    expected = expected.set_names(injection_keys)
    _validate_exact_keys(frame, injection_keys, expected, "injection")
    _require_close(
        frame["mass_GeV"],
        frame["mass_mev"] / 1000.0,
        "injection.mass_GeV/mass_mev",
    )

    for column in [
        "success",
        "refit_gp_on_toy",
        "refit_optimize",
        "signal_draw_hash_verified",
        "signal_Nsig_win_matches_anchor",
        "signal_Nsig_train_matches_anchor",
    ]:
        _require_all_true(frame, column, "injection")
    _require_all_false(frame, "refit_fallback_used", "injection")
    _require_close(
        frame["refit_ok"], 1, "injection.refit_ok", rtol=0, atol=0
    )
    _require_close(
        frame["refit_restarts"],
        int(closure["refit_gp_restarts"]),
        "injection.refit_restarts",
        rtol=0,
        atol=0,
    )
    _require_constant(frame, "inj_shape_mode", closure["shape_mode"], "injection")
    _require_constant(
        frame, "inj_background_mode", closure["background_mode"], "injection"
    )
    _require_constant(frame, "toy_mode", "full_refit", "injection")
    _require_constant(
        frame, "injection_strength_mode", "absolute", "injection"
    )
    _require_constant(
        frame, "sigmaA_ref_mode", "prefit_asimov", "injection"
    )
    signal_hash = frame["signal_draw_sha256"].astype(str)
    reference_hash = frame["signal_draw_reference_sha256"].astype(str)
    for values, name in [
        (signal_hash, "signal_draw_sha256"),
        (reference_hash, "signal_draw_reference_sha256"),
    ]:
        valid = values.str.fullmatch(r"[0-9a-f]{64}")
        if not bool(valid.all()):
            examples = values.loc[~valid].head(5).tolist()
            raise ReviewGateError(
                f"injection.{name} contains malformed hash(es): {examples}"
            )
    hash_mismatch = signal_hash != reference_hash
    if bool(hash_mismatch.any()):
        raise ReviewGateError(
            "Injection signal-draw hash differs from the factor-15 anchor in "
            f"{int(hash_mismatch.sum())} row(s)"
        )
    ledger_values = frame["injection_anchor_ledger_sha256"].astype(str)
    ledger_sha = ledger_values.iloc[0]
    ledger_path = (
        DERIVED_DIR
        / "injection_anchors"
        / "factor15_prefit_asimov_absolute_v1.json"
    )
    if not ledger_path.is_file():
        raise ReviewGateError(
            f"Missing reviewed injection anchor ledger: {ledger_path}"
        )
    if _sha256_file(ledger_path) != ledger_sha:
        raise ReviewGateError(
            "Injection anchor ledger hash differs from the collected rows"
        )

    errors = frame["refit_error"].fillna("").astype(str).str.strip()
    nonempty_errors = ~errors.isin({"", "nan", "None"})
    if bool(nonempty_errors.any()):
        examples = errors.loc[nonempty_errors].head(5).tolist()
        raise ReviewGateError(
            f"injection.refit_error is nonempty in {int(nonempty_errors.sum())} "
            f"row(s); examples={examples}"
        )

    finite_columns = [
        "strength",
        "inj_nsigma",
        "sigmaA_ref",
        "sigma_val",
        "sigma_x",
        "ls_lo",
        "ls_hi",
        "ls_opt",
        "refit_ls_opt",
        "A_hat",
        "sigma_A",
        "Zhat",
        "pull_param",
        "qmu_tilde",
        "tmu_tilde",
        "sqrt_qmu_tilde",
        "sqrt_tmu_tilde",
        "qmu_A_test",
        "qmu_nll_fixed",
        "qmu_nll_unbounded",
        "qmu_nll_null",
    ]
    for column in finite_columns:
        bad = ~np.isfinite(frame[column])
        if bool(bad.any()):
            raise ReviewGateError(
                f"injection.{column} has {int(bad.sum())} non-finite row(s)"
            )
    for column in [
        "sigmaA_ref",
        "sigma_val",
        "sigma_x",
        "sigma_A",
    ]:
        bad = frame[column] <= 0
        if bool(bad.any()):
            raise ReviewGateError(
                f"injection.{column} has {int(bad.sum())} non-positive row(s)"
            )
    if bool((frame["strength"] < -FLOAT_ATOL).any()):
        raise ReviewGateError("injection.strength contains negative values")
    signal_count_columns = [
        "Nsig_win",
        "Nsig_train",
        "signal_Nsig_full",
        "signal_Nsig_full_anchor",
        "signal_Nsig_win_anchor",
        "signal_Nsig_train_anchor",
    ]
    if bool((frame[signal_count_columns].to_numpy(float) < 0).any()):
        raise ReviewGateError("injection signal-count diagnostics contain negatives")
    _require_close(
        frame["signal_Nsig_full"],
        frame["signal_Nsig_full_anchor"],
        "injection full signal count/anchor",
        rtol=0,
        atol=0,
    )
    _require_close(
        frame["Nsig_win"],
        frame["signal_Nsig_win_anchor"],
        "injection window signal count/anchor",
        rtol=0,
        atol=0,
    )
    _require_close(
        frame["Nsig_train"],
        frame["signal_Nsig_train_anchor"],
        "injection training signal count/anchor",
        rtol=0,
        atol=0,
    )

    _require_close(
        frame["inj_nsigma"],
        frame["strength"] / frame["sigmaA_ref"],
        "injection.inj_nsigma/absolute-strength relation",
    )
    _require_close(
        frame["Zhat"],
        frame["A_hat"] / frame["sigma_A"],
        "injection.Zhat relation",
    )
    _require_close(
        frame["pull_param"],
        (frame["A_hat"] - frame["strength"]) / frame["sigma_A"],
        "injection.pull relation",
    )
    _require_close(
        frame["sqrt_qmu_tilde"] ** 2,
        frame["qmu_tilde"],
        "injection.sqrt_qmu_tilde relation",
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    _require_close(
        frame["sqrt_tmu_tilde"] ** 2,
        frame["tmu_tilde"],
        "injection.sqrt_tmu_tilde relation",
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    frame = _annotate_qmu_diagnostics(frame)

    _require_close(
        frame["kernel_ls_res_upper_factor"],
        frame["factor"],
        "injection.realized upper factor",
    )
    _require_close(
        frame["ls_hi"] / frame["sigma_x"],
        frame["factor"],
        "injection.ls_hi/sigma_x",
    )
    _require_close(
        frame["refit_ls_opt"],
        frame["ls_opt"],
        "injection.refit_ls_opt/effective ls_opt",
    )
    lower_violation = frame["ls_opt"] < frame["ls_lo"] * (1.0 - BOUND_RTOL)
    upper_violation = frame["ls_opt"] > frame["ls_hi"] * (1.0 + BOUND_RTOL)
    if bool((lower_violation | upper_violation).any()):
        raise ReviewGateError(
            "injection optimized length scale escaped its configured bounds in "
            f"{int((lower_violation | upper_violation).sum())} row(s)"
        )
    refit_lower_violation = (
        frame["refit_ls_opt"] < frame["ls_lo"] * (1.0 - BOUND_RTOL)
    )
    refit_upper_violation = (
        frame["refit_ls_opt"] > frame["ls_hi"] * (1.0 + BOUND_RTOL)
    )
    if bool((refit_lower_violation | refit_upper_violation).any()):
        raise ReviewGateError(
            "injection refit length scale escaped its configured bounds in "
            f"{int((refit_lower_violation | refit_upper_violation).sum())} row(s)"
        )
    frame["refit_ls_opt_over_sigma_x"] = (
        frame["refit_ls_opt"] / frame["sigma_x"]
    )
    bound_tolerance = BOUND_RTOL * np.maximum(1.0, frame["factor"].astype(float))
    frame["refit_at_upper_bound"] = (
        frame["refit_ls_opt_over_sigma_x"]
        >= frame["factor"].astype(float) - bound_tolerance
    )

    expected_toy = (
        frame["background_toy_index"].astype(int) * 1_000_000
        + frame["injection_toy"].astype(int)
    )
    _require_close(
        frame["toy"],
        expected_toy,
        "injection.toy composite identity",
        rtol=0,
        atol=0,
    )

    injection_seed_counts = frame.groupby(
        ["truth_model", "study_scenario", "background_toy_index"],
        dropna=False,
    )["injection_seed"].nunique()
    if bool((injection_seed_counts != 1).any()):
        bad = injection_seed_counts.loc[injection_seed_counts != 1].head(5).to_dict()
        raise ReviewGateError(
            f"injection RNG seeds are not paired across factors: {bad}"
        )
    signal_pairing = [
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
        "injection_toy",
        "anchor_nsigma",
    ]
    for column in ["Nsig_win", "Nsig_train"]:
        counts = frame.groupby(signal_pairing, dropna=False)[column].nunique()
        if bool((counts != 1).any()):
            bad = counts.loc[counts != 1].head(10).to_dict()
            raise ReviewGateError(
                f"injection.{column} is not paired across factors: {bad}"
            )
    counts = frame.groupby(signal_pairing, dropna=False)[
        "signal_draw_sha256"
    ].nunique()
    if bool((counts != 1).any()):
        bad = counts.loc[counts != 1].head(10).to_dict()
        raise ReviewGateError(
            "Poisson signal-bin hashes are not paired across factors: "
            f"{bad}"
        )
    return frame


def build_signal_response_rows(injection: pd.DataFrame) -> pd.DataFrame:
    pairing = [
        "factor",
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
        "injection_toy",
    ]
    zero = (
        injection.loc[
            np.isclose(injection["anchor_nsigma"], 0.0, atol=FLOAT_ATOL),
            pairing + ["A_hat"],
        ]
        .rename(columns={"A_hat": "A_hat_zero"})
        .copy()
    )
    positive = injection.loc[
        injection["anchor_nsigma"] > FLOAT_ATOL,
        pairing
        + [
            "anchor_nsigma",
            "inj_nsigma",
            "strength",
            "anchor_strength",
            "A_hat",
            "sigmaA_ref",
            "injection_anchor_sigmaA_ref",
            "anchor_sigma_A",
            "sigma_A_over_anchor",
            "sigmaA_ref_over_anchor",
            "Ahat_minus_Ainj_over_anchor_sigma",
        ],
    ].copy()
    response = positive.merge(zero, on=pairing, how="left", validate="many_to_one")
    if response["A_hat_zero"].isna().any():
        raise ReviewGateError(
            "Signal-response pairing is missing a zero-signal companion row"
        )
    bad_strength = (~np.isfinite(response["strength"])) | (
        response["strength"] <= 0
    )
    if bool(bad_strength.any()):
        raise ReviewGateError(
            "Signal-response denominator is non-positive or non-finite"
        )
    response["delta_A_hat"] = response["A_hat"] - response["A_hat_zero"]
    response["paired_response"] = response["delta_A_hat"] / response["strength"]
    response["paired_response_candidate_prefit_sigma_units"] = (
        response["delta_A_hat"] / response["sigmaA_ref"]
    )
    response["paired_response_anchor_prefit_sigma_units"] = (
        response["delta_A_hat"] / response["injection_anchor_sigmaA_ref"]
    )
    response["paired_response_anchor_fitted_sigma_units"] = (
        response["delta_A_hat"] / response["anchor_sigma_A"]
    )
    if not bool(np.isfinite(response["paired_response"]).all()):
        raise ReviewGateError("Non-finite paired signal-response diagnostic")
    expected_per_group = (
        injection["mass_mev"].nunique()
        * (injection["anchor_nsigma"].nunique() - 1)
        * injection["background_toy_index"].nunique()
        * injection["injection_toy"].nunique()
    )
    counts = response.groupby(
        ["factor", "truth_model", "study_scenario"]
    ).size()
    if bool((counts != expected_per_group).any()):
        bad = counts.loc[counts != expected_per_group].head(10).to_dict()
        raise ReviewGateError(
            f"Signal-response row count mismatch after zero pairing: {bad}"
        )
    return response.sort_values(
        pairing + ["anchor_nsigma"], kind="mergesort"
    ).reset_index(drop=True)


def build_stratified_signal_summary(
    injection: pd.DataFrame, response: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    """Summarize within homogeneous mass and anchor-level strata.

    Pull widths are never computed after pooling different masses or injected
    amplitudes.  Factor-level displays may take a clearly labeled median of
    these already-stratified diagnostics.
    """

    keys = [
        "truth_model",
        "study_scenario",
        "factor",
        "mass_mev",
        "anchor_nsigma",
    ]
    injection_summary = (
        injection.groupby(keys, as_index=False)
        .agg(
            injection_rows=("pull_param", "size"),
            injection_toys=("background_toy_index", "nunique"),
            sigma_A_over_anchor_median=(
                "sigma_A_over_anchor",
                "median",
            ),
            sigmaA_ref_over_anchor_median=(
                "sigmaA_ref_over_anchor",
                "median",
            ),
            Ahat_minus_Ainj_over_anchor_sigma_median=(
                "Ahat_minus_Ainj_over_anchor_sigma",
                "median",
            ),
            pull_mean=("pull_param", "mean"),
            pull_width=("pull_param", "std"),
            refit_ls_ratio_median=(
                "refit_ls_opt_over_sigma_x",
                "median",
            ),
            refit_ls_ratio_max=("refit_ls_opt_over_sigma_x", "max"),
            refit_bound_rows=("refit_at_upper_bound", "sum"),
            Nsig_win_median=("Nsig_win", "median"),
            Nsig_train_median=("Nsig_train", "median"),
        )
    )
    injection_summary["refit_bound_row_fraction"] = (
        injection_summary["refit_bound_rows"]
        / injection_summary["injection_rows"]
    )
    expected_rows = (
        int(spec["n_toys"])
        * int(spec["injection_closure"]["replicas_per_background_toy"])
    )
    bad_rows = injection_summary["injection_rows"] != expected_rows
    if bool(bad_rows.any()):
        examples = injection_summary.loc[
            bad_rows, keys + ["injection_rows"]
        ].head(10)
        raise ReviewGateError(
            "Homogeneous injection stratum row count mismatch: "
            f"{examples.to_dict(orient='records')}"
        )

    response_summary = (
        response.groupby(keys, as_index=False)
        .agg(
            response_rows=("paired_response", "size"),
            paired_response_mean=("paired_response", "mean"),
            paired_response_median=("paired_response", "median"),
            paired_response_std=("paired_response", "std"),
            paired_response_candidate_prefit_sigma_units_median=(
                "paired_response_candidate_prefit_sigma_units",
                "median",
            ),
            paired_response_anchor_prefit_sigma_units_median=(
                "paired_response_anchor_prefit_sigma_units",
                "median",
            ),
            paired_response_anchor_fitted_sigma_units_median=(
                "paired_response_anchor_fitted_sigma_units",
                "median",
            ),
        )
    )
    if not response_summary.empty:
        bad_response_rows = response_summary["response_rows"] != expected_rows
        if bool(bad_response_rows.any()):
            examples = response_summary.loc[
                bad_response_rows, keys + ["response_rows"]
            ].head(10)
            raise ReviewGateError(
                "Homogeneous signal-response stratum row count mismatch: "
                f"{examples.to_dict(orient='records')}"
            )
    summary = injection_summary.merge(
        response_summary, on=keys, how="left", validate="one_to_one"
    )
    return summary.sort_values(keys, kind="mergesort").reset_index(drop=True)


def build_summary(
    scan: pd.DataFrame,
    signal_strata: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    scan_summary = (
        scan.groupby(["truth_model", "study_scenario", "factor"], as_index=False)
        .agg(
            scan_rows=("ls_opt_over_sigma_x", "size"),
            scan_toys=("background_toy_index", "nunique"),
            scan_masses=("mass_mev", "nunique"),
            ls_ratio_median=("ls_opt_over_sigma_x", "median"),
            ls_ratio_min=("ls_opt_over_sigma_x", "min"),
            ls_ratio_max=("ls_opt_over_sigma_x", "max"),
            bound_rows=("at_upper_bound", "sum"),
        )
    )
    scan_summary["bound_row_fraction"] = (
        scan_summary["bound_rows"] / scan_summary["scan_rows"]
    )
    toy_bound = (
        scan.groupby(
            ["truth_model", "study_scenario", "factor", "background_toy_index"],
            as_index=False,
        )["at_upper_bound"]
        .any()
        .groupby(["truth_model", "study_scenario", "factor"], as_index=False)
        .agg(
            toys_with_any_bound=("at_upper_bound", "sum"),
            total_toys=("at_upper_bound", "size"),
        )
    )
    toy_bound["toy_any_bound_fraction"] = (
        toy_bound["toys_with_any_bound"] / toy_bound["total_toys"]
    )
    injection_summary = (
        signal_strata.groupby(
            ["truth_model", "study_scenario", "factor"], as_index=False
        )
        .agg(
            injection_rows=("injection_rows", "sum"),
            signal_strata=("anchor_nsigma", "size"),
            pull_mean_strata_median=("pull_mean", "median"),
            pull_width_strata_median=("pull_width", "median"),
            sigma_A_over_anchor_median=(
                "sigma_A_over_anchor_median",
                "median",
            ),
            sigmaA_ref_over_anchor_median=(
                "sigmaA_ref_over_anchor_median",
                "median",
            ),
            Ahat_minus_Ainj_over_anchor_sigma_median=(
                "Ahat_minus_Ainj_over_anchor_sigma_median",
                "median",
            ),
            injection_ls_ratio_median=(
                "refit_ls_ratio_median",
                "median",
            ),
            injection_refit_bound_fraction=(
                "refit_bound_row_fraction",
                "mean",
            ),
        )
    )
    response_summary = (
        signal_strata.loc[signal_strata["anchor_nsigma"] > FLOAT_ATOL]
        .groupby(
            ["truth_model", "study_scenario", "factor"], as_index=False
        )
        .agg(
            response_rows=("response_rows", "sum"),
            positive_signal_strata=("anchor_nsigma", "size"),
            paired_response_mean=("paired_response_mean", "median"),
            paired_response_median=("paired_response_median", "median"),
            paired_response_std=("paired_response_std", "median"),
            paired_response_candidate_prefit_sigma_units_median=(
                "paired_response_candidate_prefit_sigma_units_median",
                "median",
            ),
            paired_response_anchor_prefit_sigma_units_median=(
                "paired_response_anchor_prefit_sigma_units_median",
                "median",
            ),
            paired_response_anchor_fitted_sigma_units_median=(
                "paired_response_anchor_fitted_sigma_units_median",
                "median",
            ),
        )
    )
    audit_summary = (
        audit.groupby(
            ["truth_model", "study_scenario", "factor_current"], as_index=False
        )
        .agg(
            adjacent_lml_comparisons=("delta_lml", "size"),
            adjacent_lml_min_delta=("delta_lml", "min"),
            adjacent_lml_median_delta=("delta_lml", "median"),
            adjacent_lml_regressions=(
                "regression_beyond_tolerance",
                "sum",
            ),
        )
        .rename(columns={"factor_current": "factor"})
    )
    factors_min = int(scan["factor"].min())
    first_factor = scan_summary.loc[
        scan_summary["factor"] == factors_min,
        ["truth_model", "study_scenario", "factor"],
    ].copy()
    first_factor["adjacent_lml_comparisons"] = 0
    first_factor["adjacent_lml_min_delta"] = np.nan
    first_factor["adjacent_lml_median_delta"] = np.nan
    first_factor["adjacent_lml_regressions"] = 0
    audit_summary = pd.concat([first_factor, audit_summary], ignore_index=True)

    summary = scan_summary.merge(
        toy_bound,
        on=["truth_model", "study_scenario", "factor"],
        validate="one_to_one",
    )
    summary = summary.merge(
        injection_summary,
        on=["truth_model", "study_scenario", "factor"],
        validate="one_to_one",
    )
    summary = summary.merge(
        response_summary,
        on=["truth_model", "study_scenario", "factor"],
        validate="one_to_one",
    )
    summary = summary.merge(
        audit_summary,
        on=["truth_model", "study_scenario", "factor"],
        validate="one_to_one",
    )
    return summary.sort_values(
        ["truth_model", "study_scenario", "factor"], kind="mergesort"
    ).reset_index(drop=True)


def _configure_matplotlib() -> None:
    mpl_dir = Path(tempfile.gettempdir()) / "v4p1_ensemble_postprocess_mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.25,
        }
    )


def _save_figure_atomic(figure: Any, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(fd)
    try:
        figure.savefig(temporary, **kwargs)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def plot_all_toy_curves(
    scan: pd.DataFrame,
    spec: Mapping[str, Any],
    truth: str,
    *,
    plots_dir: Path = PLOTS_DIR,
) -> List[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    factors = list(map(int, spec["length_scale_upper_factors"]))
    scenarios = list(map(str, spec["scenarios"]))
    toys = list(range(int(spec["n_toys"])))
    toy_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(toys)))
    truth_label = str(spec["truth_models"][truth]["function_tag"])
    outputs: List[Path] = []
    pdf_path = plots_dir / f"fig_v4p1_ensemble_ls_all_toys_{truth}.pdf"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_pdf = tempfile.mkstemp(
        prefix=f".{pdf_path.stem}.", suffix=".pdf", dir=str(plots_dir)
    )
    os.close(fd)
    try:
        with PdfPages(temporary_pdf) as pdf:
            for factor in factors:
                figure, axes = plt.subplots(
                    2, 3, figsize=(11.5, 7.3), sharex=True, sharey=True
                )
                flat = axes.ravel()
                factor_frame = scan.loc[
                    (scan["truth_model"].astype(str) == truth)
                    & (scan["factor"] == factor)
                ]
                for panel, scenario in enumerate(scenarios):
                    axis = flat[panel]
                    scenario_frame = factor_frame.loc[
                        factor_frame["study_scenario"].astype(str) == scenario
                    ]
                    for toy, color in zip(toys, toy_colors):
                        toy_frame = scenario_frame.loc[
                            scenario_frame["background_toy_index"].astype(int)
                            == toy
                        ].sort_values("mass_mev")
                        axis.plot(
                            toy_frame["mass_mev"],
                            toy_frame["ls_opt_over_sigma_x"],
                            color=color,
                            marker="o",
                            markersize=2.3,
                            linewidth=1.0,
                            alpha=0.92,
                        )
                        bound = toy_frame["at_upper_bound"].astype(bool)
                        if bool(bound.any()):
                            axis.scatter(
                                toy_frame.loc[bound, "mass_mev"],
                                toy_frame.loc[bound, "ls_opt_over_sigma_x"],
                                marker="s",
                                s=18,
                                facecolors="none",
                                edgecolors=color,
                                linewidths=0.8,
                                zorder=4,
                            )
                    axis.axhline(
                        factor,
                        color="0.25",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.85,
                    )
                    axis.set_title(_scenario_display_label(scenario))
                    axis.set_xlim(
                        min(_mass_grid_mev(spec)), max(_mass_grid_mev(spec))
                    )
                    axis.set_ylim(0, factor * 1.06)
                    if panel // 3 == 1:
                        axis.set_xlabel("Mass [MeV]")
                    if panel % 3 == 0:
                        axis.set_ylabel(r"Optimized $\ell/\sigma_x$")
                flat[-1].axis("off")
                handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker="o",
                        markersize=3,
                        linewidth=1.2,
                        label=f"toy {toy}",
                    )
                    for toy, color in zip(toys, toy_colors)
                ]
                handles.extend(
                    [
                        Line2D(
                            [0],
                            [0],
                            color="0.25",
                            linestyle="--",
                            label=f"factor-{factor} ceiling",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="0.25",
                            marker="s",
                            markerfacecolor="none",
                            linestyle="none",
                            label="at ceiling",
                        ),
                    ]
                )
                flat[-1].legend(
                    handles=handles,
                    loc="center",
                    ncol=2,
                    frameon=False,
                    title="Raw toy indices",
                )
                factor_role = (
                    " (projection candidate)"
                    if factor == PROJECTION_CANDIDATE_FACTOR
                    else ""
                )
                figure.suptitle(
                    f"2021 paired exposure toys: {truth_label}, "
                    f"upper factor {factor}{factor_role}\n"
                    "Ten raw curves; pairing only within each source family; "
                    "no bands",
                    y=0.985,
                )
                figure.tight_layout(rect=(0, 0, 1, 0.94))
                pdf.savefig(figure, bbox_inches="tight")
                factor_pdf_path = plots_dir / (
                    f"fig_v4p1_ensemble_ls_all_toys_{truth}_"
                    f"f{factor:02d}.pdf"
                )
                png_path = plots_dir / (
                    f"fig_v4p1_ensemble_ls_all_toys_{truth}_f{factor:02d}.png"
                )
                _save_figure_atomic(
                    figure,
                    factor_pdf_path,
                    bbox_inches="tight",
                    facecolor="white",
                )
                _save_figure_atomic(
                    figure,
                    png_path,
                    dpi=240,
                    bbox_inches="tight",
                    facecolor="white",
                )
                outputs.extend([factor_pdf_path, png_path])
                plt.close(figure)
        os.replace(temporary_pdf, pdf_path)
    except Exception:
        try:
            os.unlink(temporary_pdf)
        except FileNotFoundError:
            pass
        raise
    outputs.insert(0, pdf_path)
    return outputs


def build_factor20_toy_median_comparison(
    scan: pd.DataFrame,
    spec: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build unpaired toy-level medians for the native/scaled source comparison."""

    factor = PROJECTION_CANDIDATE_FACTOR
    scenarios = ["2021_1pct_x10", "2021_10pct"]
    truths = ["gengamma", "sigpowexpq"]
    if factor not in set(map(int, spec["length_scale_upper_factors"])):
        raise ReviewGateError(
            f"Projection-candidate factor {factor} is not in study_spec"
        )
    if not set(truths).issubset(set(map(str, spec["truth_models"]))):
        raise ReviewGateError(
            "Factor-20 comparison requires primary gengamma and alternate "
            "sigpowexpq truth lanes"
        )
    if not set(scenarios).issubset(set(map(str, spec["scenarios"]))):
        raise ReviewGateError(
            "Factor-20 comparison scenarios are missing from study_spec"
        )
    required = [
        "factor",
        "truth_model",
        "study_scenario",
        "background_toy_index",
        "mass_mev",
        "ls_opt_over_sigma_x",
        "at_upper_bound",
    ]
    _require_columns(scan, required, "factor-20 comparison scan")
    subset = scan.loc[
        (scan["factor"].astype(int) == factor)
        & scan["truth_model"].astype(str).isin(truths)
        & scan["study_scenario"].astype(str).isin(scenarios)
    ].copy()
    toys = list(range(int(spec["n_toys"])))
    masses = _mass_grid_mev(spec)
    expected = pd.MultiIndex.from_product(
        [truths, scenarios, toys, masses],
        names=[
            "truth_model",
            "study_scenario",
            "background_toy_index",
            "mass_mev",
        ],
    )
    keys = list(expected.names)
    _validate_exact_keys(subset, keys, expected, "factor-20 comparison scan")
    if not bool(np.isfinite(subset["ls_opt_over_sigma_x"]).all()):
        raise ReviewGateError(
            "factor-20 comparison has non-finite optimized length scales"
        )
    subset["near_upper_bound"] = (
        subset["ls_opt_over_sigma_x"].astype(float) >= 0.98 * factor
    )

    toy_rows = (
        subset.groupby(
            [
                "truth_model",
                "study_scenario",
                "background_toy_index",
            ],
            as_index=False,
        )
        .agg(
            n_mass_rows=("mass_mev", "size"),
            n_unique_masses=("mass_mev", "nunique"),
            toy_median_ls_opt_over_sigma_x=(
                "ls_opt_over_sigma_x",
                "median",
            ),
            at_upper_bound_mass_rows=("at_upper_bound", "sum"),
            near_upper_bound_mass_rows=("near_upper_bound", "sum"),
        )
    )
    expected_mass_rows = len(masses)
    bad_mass_rows = (
        (toy_rows["n_mass_rows"] != expected_mass_rows)
        | (toy_rows["n_unique_masses"] != expected_mass_rows)
    )
    if bool(bad_mass_rows.any()):
        examples = toy_rows.loc[bad_mass_rows].head(10).to_dict(
            orient="records"
        )
        raise ReviewGateError(
            "Factor-20 toy medians do not each contain the full mass grid: "
            f"{examples}"
        )
    expected_toy_rows = len(truths) * len(scenarios) * len(toys)
    if len(toy_rows) != expected_toy_rows:
        raise ReviewGateError(
            f"Factor-20 comparison expected {expected_toy_rows} toy medians, "
            f"got {len(toy_rows)}"
        )

    labels = {
        "2021_1pct_x10": "1%-source x10",
        "2021_10pct": "native 10%",
    }
    truth_roles = {"gengamma": "primary", "sigpowexpq": "alternate"}
    source_targets = normalization["source_normalization_target_counts"]
    effective_targets = normalization["effective_target_counts"]
    toy_rows["factor"] = factor
    toy_rows["projection_candidate"] = True
    toy_rows["truth_function_tag"] = toy_rows["truth_model"].map(
        lambda truth: str(spec["truth_models"][truth]["function_tag"])
    )
    toy_rows["truth_role"] = toy_rows["truth_model"].map(truth_roles)
    toy_rows["display_label"] = toy_rows["study_scenario"].map(labels)
    toy_rows["source_family"] = toy_rows["study_scenario"].map(
        lambda scenario: str(spec["scenarios"][scenario]["source_family"])
    )
    toy_rows["exposure_multiplier"] = toy_rows["study_scenario"].map(
        lambda scenario: int(
            spec["scenarios"][scenario]["exposure_multiplier"]
        )
    )
    toy_rows["source_normalization_target_count"] = toy_rows[
        "source_family"
    ].map(lambda family: int(source_targets[family]))
    toy_rows["effective_target_count"] = toy_rows["study_scenario"].map(
        lambda scenario: int(effective_targets[scenario])
    )
    toy_rows["source_families_paired"] = False
    toy_rows["mass_rows_correlated"] = True
    toy_rows["expected_limit_bands"] = False

    summary = (
        toy_rows.groupby(
            [
                "factor",
                "projection_candidate",
                "truth_model",
                "truth_function_tag",
                "truth_role",
                "study_scenario",
                "display_label",
                "source_family",
                "exposure_multiplier",
                "source_normalization_target_count",
                "effective_target_count",
                "source_families_paired",
                "mass_rows_correlated",
                "expected_limit_bands",
            ],
            as_index=False,
        )
        .agg(
            n_independent_toys=("background_toy_index", "nunique"),
            n_mass_points_per_toy=("n_mass_rows", "min"),
            toy_median_min=(
                "toy_median_ls_opt_over_sigma_x",
                "min",
            ),
            toy_median_max=(
                "toy_median_ls_opt_over_sigma_x",
                "max",
            ),
            toy_median_mean=(
                "toy_median_ls_opt_over_sigma_x",
                "mean",
            ),
            toy_median_median=(
                "toy_median_ls_opt_over_sigma_x",
                "median",
            ),
            toy_median_std_ddof1=(
                "toy_median_ls_opt_over_sigma_x",
                "std",
            ),
            at_upper_bound_mass_rows=(
                "at_upper_bound_mass_rows",
                "sum",
            ),
            toys_with_any_upper_bound_mass_row=(
                "at_upper_bound_mass_rows",
                lambda values: int((values > 0).sum()),
            ),
            near_upper_bound_mass_rows=(
                "near_upper_bound_mass_rows",
                "sum",
            ),
            toys_with_any_near_upper_bound_mass_row=(
                "near_upper_bound_mass_rows",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    summary["source_support_ratio_ten_pct_over_one_pct"] = float(
        normalization["source_support_ratio_ten_pct_over_one_pct"]
    )
    summary["effective_target_ratio_native10_over_1pct_x10"] = float(
        normalization[
            "effective_target_ratio_native10_over_1pct_x10"
        ]
    )
    summary = summary.sort_values(
        ["truth_role", "study_scenario"],
        key=lambda values: values.map(
            {
                "primary": 0,
                "alternate": 1,
                "2021_1pct_x10": 0,
                "2021_10pct": 1,
            }
        ),
        kind="mergesort",
    ).reset_index(drop=True)
    toy_rows = toy_rows.sort_values(
        ["truth_role", "study_scenario", "background_toy_index"],
        key=lambda values: values.map(
            {
                "primary": 0,
                "alternate": 1,
                "2021_1pct_x10": 0,
                "2021_10pct": 1,
            }
        ).fillna(values),
        kind="mergesort",
    ).reset_index(drop=True)
    return toy_rows, summary


def plot_factor20_toy_median_comparison(
    toy_rows: pd.DataFrame,
    summary: pd.DataFrame,
    normalization: Mapping[str, Any],
    *,
    plots_dir: Path = PLOTS_DIR,
) -> List[Path]:
    """Plot two independent ten-point strips; never connect source families."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scenarios = ["2021_1pct_x10", "2021_10pct"]
    truths = ["gengamma", "sigpowexpq"]
    colors = {
        "2021_1pct_x10": "#0072B2",
        "2021_10pct": "#D55E00",
    }
    markers = {"2021_1pct_x10": "o", "2021_10pct": "s"}
    labels = {"2021_1pct_x10": "1%-source x10", "2021_10pct": "native 10%"}
    figure, axes = plt.subplots(
        1, 2, figsize=(10.4, 5.25), sharey=True
    )
    offset_orders = {
        "2021_1pct_x10": np.linspace(-0.085, 0.085, 10),
        "2021_10pct": np.asarray(
            [0.075, -0.025, -0.085, 0.035, -0.045, 0.085, -0.065, 0.015, 0.055, -0.005]
        ),
    }
    for axis, truth in zip(axes, truths):
        truth_rows = toy_rows.loc[
            toy_rows["truth_model"].astype(str) == truth
        ]
        for x_value, scenario in enumerate(scenarios):
            rows = truth_rows.loc[
                truth_rows["study_scenario"].astype(str) == scenario
            ].sort_values("background_toy_index")
            values = rows["toy_median_ls_opt_over_sigma_x"].to_numpy(
                dtype=float
            )
            offsets = offset_orders[scenario]
            if len(values) != len(offsets):
                raise ReviewGateError(
                    f"Factor-20 plot expected ten {truth}/{scenario} toys, "
                    f"got {len(values)}"
                )
            axis.scatter(
                x_value + offsets,
                values,
                color=colors[scenario],
                marker=markers[scenario],
                s=42,
                edgecolors="white",
                linewidths=0.65,
                alpha=0.94,
                zorder=3,
            )
            group_median = float(np.median(values))
            axis.plot(
                [x_value - 0.12, x_value + 0.12],
                [group_median, group_median],
                color="0.15",
                linewidth=1.6,
                zorder=4,
            )
            axis.text(
                x_value,
                max(0.45, group_median - 0.65),
                f"median {group_median:.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color="0.2",
            )
        role = str(
            truth_rows["truth_role"].drop_duplicates().iloc[0]
        ).capitalize()
        function_tag = str(
            truth_rows["truth_function_tag"].drop_duplicates().iloc[0]
        )
        axis.set_title(f"{role}: {function_tag}")
        axis.set_xticks([0, 1], [labels[value] for value in scenarios])
        axis.set_xlim(-0.35, 1.35)
        axis.set_ylim(0, PROJECTION_CANDIDATE_FACTOR * 1.04)
        axis.axhline(
            PROJECTION_CANDIDATE_FACTOR,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.set_xlabel("Independent source-family ensemble")
    axes[0].set_ylabel(
        r"Toy median of optimized $\ell/\sigma_x$ across 11 masses"
    )

    alternate_native = summary.loc[
        (summary["truth_model"].astype(str) == "sigpowexpq")
        & (summary["study_scenario"].astype(str) == "2021_10pct")
    ]
    near_rows = int(alternate_native["near_upper_bound_mass_rows"].iloc[0])
    near_toys = int(
        alternate_native[
            "toys_with_any_near_upper_bound_mass_row"
        ].iloc[0]
    )
    support_ratio = float(
        normalization["source_support_ratio_ten_pct_over_one_pct"]
    )
    effective_ratio = float(
        normalization[
            "effective_target_ratio_native10_over_1pct_x10"
        ]
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[scenario],
            color="none",
            markerfacecolor=colors[scenario],
            markeredgecolor="white",
            markersize=7,
            label=labels[scenario],
        )
        for scenario in scenarios
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=1.6,
                label="median of 10 toys",
            ),
            Line2D(
                [0],
                [0],
                color="0.25",
                linestyle="--",
                linewidth=1.0,
                label="factor-20 ceiling",
            ),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.105),
    )
    figure.suptitle(
        "Factor-20 projection candidate: native 2021 10% versus "
        "1%-source x10\n"
        "Independent source families (not paired); source support ratio "
        f"10%/1% = {support_ratio:.3f}",
        y=0.985,
    )
    figure.text(
        0.5,
        0.035,
        "Toy is the independent unit (n=10/category); mass rows are correlated; "
        f"no bands. Effective expected-count ratio = {effective_ratio:.3f}. "
        f"Alternate/native 10%: {near_rows} near-ceiling mass rows in "
        f"{near_toys} toys.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    figure.tight_layout(rect=(0, 0.17, 1, 0.90))
    stem = "fig_v4p1_factor20_native10_vs_1pct_x10_toy_medians"
    pdf_path = plots_dir / f"{stem}.pdf"
    png_path = plots_dir / f"{stem}.png"
    _save_figure_atomic(
        figure, pdf_path, bbox_inches="tight", facecolor="white"
    )
    _save_figure_atomic(
        figure,
        png_path,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return [pdf_path, png_path]


def plot_bound_lml_diagnostics(
    summary: pd.DataFrame,
    spec: Mapping[str, Any],
    truth: str,
    *,
    plots_dir: Path = PLOTS_DIR,
) -> List[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    scenarios = list(map(str, spec["scenarios"]))
    factors = list(map(int, spec["length_scale_upper_factors"]))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(scenarios)))
    truth_label = str(spec["truth_models"][truth]["function_tag"])
    sub = summary.loc[summary["truth_model"].astype(str) == truth]
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.3), sharex=True)
    panels = [
        (
            axes[0, 0],
            "bound_row_fraction",
            "Mass-toy points at upper bound",
            "Fraction of scan rows",
            (0, 1.02),
        ),
        (
            axes[0, 1],
            "toy_any_bound_fraction",
            "Toys with any upper-bound contact",
            "Fraction of toys",
            (0, 1.02),
        ),
        (
            axes[1, 0],
            "adjacent_lml_median_delta",
            "Median adjacent-factor likelihood gain",
            r"Median $\Delta\log\mathcal{L}$",
            None,
        ),
        (
            axes[1, 1],
            "adjacent_lml_min_delta",
            "Worst adjacent-factor likelihood gain",
            r"Minimum $\Delta\log\mathcal{L}$",
            None,
        ),
    ]
    for axis, metric, title, ylabel, ylim in panels:
        for scenario, color in zip(scenarios, colors):
            rows = sub.loc[
                sub["study_scenario"].astype(str) == scenario
            ].sort_values("factor")
            axis.plot(
                rows["factor"],
                rows[metric],
                color=color,
                marker="o",
                markersize=4,
                label=_scenario_display_label(scenario),
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(factors)
        if ylim is not None:
            axis.set_ylim(*ylim)
    axes[1, 0].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[1, 1].axhline(
        0.0,
        color="0.25",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1, 0].set_xlabel(r"Length-scale upper factor [$\sigma_x$]")
    axes[1, 1].set_xlabel(r"Length-scale upper factor [$\sigma_x$]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        f"2021 paired-exposure bound occupancy and LML: {truth_label}\n"
        "Ten-toy screening ensemble; larger factors are ceilings, not targets",
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.94))
    pdf_path = plots_dir / (
        f"fig_v4p1_ensemble_bound_lml_diagnostics_{truth}.pdf"
    )
    png_path = plots_dir / (
        f"fig_v4p1_ensemble_bound_lml_diagnostics_{truth}.png"
    )
    _save_figure_atomic(figure, pdf_path, bbox_inches="tight", facecolor="white")
    _save_figure_atomic(
        figure, png_path, dpi=240, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    return [pdf_path, png_path]


def plot_fixed_amplitude_response(
    summary: pd.DataFrame,
    spec: Mapping[str, Any],
    truth: str,
    *,
    plots_dir: Path = PLOTS_DIR,
) -> List[Path]:
    """Plot signal diagnostics separately from bound occupancy and LML."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scenarios = list(map(str, spec["scenarios"]))
    factors = list(map(int, spec["length_scale_upper_factors"]))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(scenarios)))
    truth_label = str(spec["truth_models"][truth]["function_tag"])
    anchor_factor = int(
        spec["injection_closure"].get(
            "fixed_amplitude_anchor_factor",
            spec["injection_closure"].get(
                "amplitude_anchor_factor",
                spec["injection_closure"].get("anchor_factor", 15),
            ),
        )
    )
    sub = summary.loc[summary["truth_model"].astype(str) == truth]
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.3), sharex=True)
    panels = [
        (
            axes[0, 0],
            "sigma_A_over_anchor_median",
            "Candidate extraction uncertainty",
            rf"Median $\sigma_A/\sigma_A^{{(f={anchor_factor})}}$",
            1.0,
        ),
        (
            axes[0, 1],
            "Ahat_minus_Ainj_over_anchor_sigma_median",
            "Anchor-normalized signed residual",
            rf"Median $(\hat A-A_{{\rm inj}})/\sigma_A^{{(f={anchor_factor})}}$",
            0.0,
        ),
        (
            axes[1, 0],
            "paired_response_median",
            "Fixed-amplitude paired recovery",
            r"Median $(\hat A_z-\hat A_0)/A_{\rm inj}$",
            1.0,
        ),
        (
            axes[1, 1],
            "pull_width_strata_median",
            "Injection pull width by stratum",
            "Median stratum pull width",
            1.0,
        ),
        (
            axes[1, 2],
            "injection_refit_bound_fraction",
            "Injection-refit upper-bound contact",
            "Fraction of injection rows",
            0.0,
        ),
    ]
    for axis, metric, title, ylabel, reference in panels:
        for scenario, color in zip(scenarios, colors):
            rows = sub.loc[
                sub["study_scenario"].astype(str) == scenario
            ].sort_values("factor")
            axis.plot(
                rows["factor"],
                rows[metric],
                color=color,
                marker="o",
                markersize=4,
                label=_scenario_display_label(scenario),
            )
        axis.axhline(
            reference, color="0.25", linestyle="--", linewidth=1.0
        )
        axis.axvline(
            anchor_factor, color="0.55", linestyle=":", linewidth=0.9
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(factors)
    axes[1, 0].set_xlabel(r"Length-scale upper factor [$\sigma_x$]")
    axes[1, 1].set_xlabel(r"Length-scale upper factor [$\sigma_x$]")
    axes[1, 2].set_xlabel(r"Length-scale upper factor [$\sigma_x$]")
    axes[0, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.55",
            linestyle=":",
            linewidth=0.9,
            label=f"amplitude anchor f={anchor_factor}",
        )
    )
    labels.append(f"amplitude anchor f={anchor_factor}")
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        f"2021 fixed-amplitude signal response versus GP bound: {truth_label}\n"
        f"Absolute injected A values anchored once at factor {anchor_factor}; "
        "medians of mass x anchor-level strata; no expected-limit bands",
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0.095, 1, 0.94))
    pdf_path = plots_dir / (
        f"fig_v4p1_ensemble_fixed_amplitude_response_{truth}.pdf"
    )
    png_path = plots_dir / (
        f"fig_v4p1_ensemble_fixed_amplitude_response_{truth}.png"
    )
    _save_figure_atomic(figure, pdf_path, bbox_inches="tight", facecolor="white")
    _save_figure_atomic(
        figure, png_path, dpi=240, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    return [pdf_path, png_path]


def _output_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_scan_note_outputs(
    scan: pd.DataFrame,
    audit: pd.DataFrame,
    spec: Mapping[str, Any],
    truths: Sequence[str],
    scan_csv: Path,
    scan_collection_report: Mapping[str, Any],
    optimizer_audit: Mapping[str, Any],
    toy_manifest: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> List[Path]:
    """Publish scan-only note artifacts without touching injection outputs."""

    manifest_path = DERIVED_DIR / "v4p1_scan_note_artifacts_manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": spec["study_id"],
            "updated_utc": _utc_now(),
            "status": "building",
            "selected_truth_models": list(truths),
            "note": (
                "The scan-only note bundle is being regenerated. The "
                "injection-dependent postprocess manifest is untouched."
            ),
        },
    )
    stage_root = Path(
        tempfile.mkdtemp(prefix=".v4p1_scan_note_stage_", dir=str(STUDY_DIR))
    )
    stage_derived = stage_root / "derived"
    stage_plots = stage_root / "plots"
    staged_outputs: List[Path] = []
    try:
        for truth in truths:
            staged_outputs.extend(
                plot_all_toy_curves(
                    scan, spec, truth, plots_dir=stage_plots
                )
            )
        toy_rows, comparison_summary = (
            build_factor20_toy_median_comparison(
                scan, spec, normalization
            )
        )
        comparison_summary_path = stage_derived / (
            "fig_v4p1_factor20_native10_vs_1pct_x10_"
            "toy_medians_summary.csv"
        )
        _atomic_write_csv(comparison_summary_path, comparison_summary)
        staged_outputs.append(comparison_summary_path)
        staged_outputs.extend(
            plot_factor20_toy_median_comparison(
                toy_rows,
                comparison_summary,
                normalization,
                plots_dir=stage_plots,
            )
        )

        for path in staged_outputs:
            _output_record(path)
        outputs: List[Path] = []
        for staged in staged_outputs:
            relative = staged.relative_to(stage_root)
            final = STUDY_DIR / relative
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            outputs.append(final)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    script_path = Path(__file__).resolve()
    scan_report_path = _collection_report_path(scan_csv, "scan")
    assert scan_report_path is not None
    source_metadata_records = {
        family: _output_record(path)
        for family, path in SOURCE_METADATA_PATHS.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": spec["study_id"],
        "created_utc": _utc_now(),
        "status": "complete",
        "selected_truth_models": list(truths),
        "inputs": {
            "study_spec": _output_record(SPEC_PATH),
            "paired_toy_manifest": _output_record(TOY_MANIFEST_PATH),
            "paired_toy_root": _output_record(TOY_ROOT_PATH),
            "paired_toy_root_manifest_sha256": toy_manifest[
                "toy_root_sha256"
            ],
            "source_metadata": source_metadata_records,
            "reviewed_scan_csv": _output_record(scan_csv),
            "reviewed_scan_collection_report": _output_record(
                scan_report_path
            ),
            "optimizer_audit_summary": _output_record(
                SCAN_AUDIT_SUMMARY_PATH
            ),
            "postprocess_script": _output_record(script_path),
        },
        "validated_rows": {
            "reviewed_scan": int(len(scan)),
            "adjacent_factor_lml_comparisons": int(len(audit)),
            "factor20_comparison_scan_rows": 440,
            "factor20_toy_medians": 40,
        },
        "gates": {
            "reviewed_scan_collection": dict(scan_collection_report),
            "optimizer_audit_gate": optimizer_audit["audit_gate"],
            "actual_fit_rows_only": True,
            "interpolation_used": False,
            "unresolved_initialization_states": 0,
            "exact_cartesian_coverage": True,
            "frozen_or_hashed_repair_provenance": True,
            "adjacent_factor_lml_tolerance": {
                "formula": (
                    "max(abs_tol, rel_tol*max(abs(low_lml),"
                    "abs(high_lml),1))"
                ),
                "absolute": LML_ABS_TOLERANCE,
                "relative": LML_REL_TOLERANCE,
                "recorded_per_pair": True,
            },
            "adjacent_factor_lml_regressions": int(
                audit["regression_beyond_tolerance"].astype(bool).sum()
            ),
        },
        "comparison": {
            "factor": PROJECTION_CANDIDATE_FACTOR,
            "role": "projection_candidate",
            "scenarios": ["2021_1pct_x10", "2021_10pct"],
            "source_families_paired": False,
            "independent_unit": "toy",
            "n_toys_per_category": int(spec["n_toys"]),
            "mass_rows_correlated_within_toy": True,
            "mass_points_per_toy": len(_mass_grid_mev(spec)),
            **dict(normalization),
        },
        "semantics": {
            "all_ten_raw_toy_curves": True,
            "one_page_pdf_per_factor_and_truth": True,
            "factor20_comparison_uses_raw_toy_level_mass_medians": True,
            "source_family_toy_indices_are_not_paired": True,
            "expected_limit_bands": False,
            "limit_bands_created": False,
            "injection_data_read": False,
            "production_fits_executed": False,
            "scope": (
                "optimizer and hyperparameter-support pilot only; not "
                "coverage, an expected-limit band, exclusion, discovery "
                "calibration, or a physics reach statement"
            ),
        },
        "outputs": [_output_record(path) for path in outputs],
    }
    _atomic_write_json(manifest_path, manifest)
    outputs.append(manifest_path)
    return outputs


def _write_outputs(
    scan: pd.DataFrame,
    injection: pd.DataFrame,
    audit: pd.DataFrame,
    response: pd.DataFrame,
    signal_strata: pd.DataFrame,
    summary: pd.DataFrame,
    spec: Mapping[str, Any],
    truths: Sequence[str],
    scan_csv: Path,
    injection_csv: Path,
    collection_reports: Mapping[str, Mapping[str, Any] | None],
    toy_manifest: Mapping[str, Any],
) -> List[Path]:
    manifest_path = DERIVED_DIR / "v4p1_ensemble_postprocess_manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": spec["study_id"],
            "updated_utc": _utc_now(),
            "status": "building",
            "selected_truth_models": list(truths),
            "note": (
                "A prior complete manifest is invalid while the staged bundle "
                "is being regenerated."
            ),
        },
    )
    stage_root = Path(
        tempfile.mkdtemp(prefix=".v4p1_postprocess_stage_", dir=str(STUDY_DIR))
    )
    stage_derived = stage_root / "derived"
    stage_plots = stage_root / "plots"
    staged_outputs: List[Path] = []
    try:
        for truth in truths:
            truth_summary = summary.loc[
                summary["truth_model"].astype(str) == truth
            ].copy()
            truth_audit = audit.loc[
                audit["truth_model"].astype(str) == truth
            ].copy()
            truth_response = response.loc[
                response["truth_model"].astype(str) == truth
            ].copy()
            truth_signal_strata = signal_strata.loc[
                signal_strata["truth_model"].astype(str) == truth
            ].copy()
            summary_path = stage_derived / (
                f"v4p1_ensemble_factor_summary_{truth}.csv"
            )
            audit_path = stage_derived / (
                f"v4p1_ensemble_adjacent_lml_audit_{truth}.csv"
            )
            response_path = stage_derived / (
                f"v4p1_ensemble_signal_response_rows_{truth}.csv"
            )
            strata_path = stage_derived / (
                f"v4p1_ensemble_signal_response_summary_{truth}.csv"
            )
            _atomic_write_csv(summary_path, truth_summary)
            _atomic_write_csv(audit_path, truth_audit)
            _atomic_write_csv(response_path, truth_response)
            _atomic_write_csv(strata_path, truth_signal_strata)
            staged_outputs.extend(
                [summary_path, audit_path, response_path, strata_path]
            )
            staged_outputs.extend(
                plot_all_toy_curves(
                    scan, spec, truth, plots_dir=stage_plots
                )
            )
            staged_outputs.extend(
                plot_bound_lml_diagnostics(
                    summary, spec, truth, plots_dir=stage_plots
                )
            )
            staged_outputs.extend(
                plot_fixed_amplitude_response(
                    summary, spec, truth, plots_dir=stage_plots
                )
            )

        # Hash every staged file before publishing any of them.
        for path in staged_outputs:
            _output_record(path)
        outputs: List[Path] = []
        for staged in staged_outputs:
            relative = staged.relative_to(stage_root)
            final = STUDY_DIR / relative
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            outputs.append(final)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    script_path = Path(__file__).resolve()
    scan_collection = collection_reports.get("scan") or {}
    reviewed_scan = (
        scan_collection.get("review_stage")
        == "optimizer_selected_actual_fit_rows"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": spec["study_id"],
        "created_utc": _utc_now(),
        "status": "complete",
        "selected_truth_models": list(truths),
        "inputs": {
            "study_spec": _output_record(SPEC_PATH),
            "paired_toy_manifest": _output_record(TOY_MANIFEST_PATH),
            "paired_toy_root": _output_record(TOY_ROOT_PATH),
            "paired_toy_root_manifest_sha256": toy_manifest[
                "toy_root_sha256"
            ],
            "scan_csv": _output_record(scan_csv),
            "injection_csv": _output_record(injection_csv),
            "postprocess_script": _output_record(script_path),
            "scan_collection_report": collection_reports.get("scan"),
            "injection_collection_report": collection_reports.get("injection"),
        },
        "validated_rows": {
            "scan": int(len(scan)),
            "injection": int(len(injection)),
            "paired_signal_response": int(len(response)),
            "mass_anchor_signal_strata": int(len(signal_strata)),
            "adjacent_factor_lml_comparisons": int(len(audit)),
            "qmu_ok_false_rows": int(
                (~injection["qmu_ok_parsed"].astype(bool)).sum()
            ),
            "qmu_one_sided_zero_branch_diagnostic_rows": int(
                injection[
                    "qmu_one_sided_zero_branch_diagnostic"
                ].astype(bool).sum()
            ),
        },
        "gates": {
            "exact_cartesian_coverage": True,
            "frozen_nominal_provenance": True,
            "reviewed_hashed_repair_provenance": reviewed_scan,
            "fit_and_refit_success": True,
            "scan_geometry": True,
            "optimizer_seed_policy": (
                "final optimizer audit with independently salted repair seeds"
                if reviewed_scan
                else "paired optimizer seeds across factors"
            ),
            "fixed_amplitude_protocol": (
                "factor15_prefit_asimov_absolute_v1"
            ),
            "paired_signal_counts_across_factors": True,
            "qmu_diagnostic_incoherent_rows": 0,
            "adjacent_factor_lml_tolerance": {
                "formula": (
                    "max(abs_tol, rel_tol*max(abs(low_lml),"
                    "abs(high_lml),1))"
                ),
                "absolute": LML_ABS_TOLERANCE,
                "relative": LML_REL_TOLERANCE,
                "recorded_per_pair": True,
            },
            "adjacent_factor_lml_regressions": int(
                audit["regression_beyond_tolerance"].astype(bool).sum()
            ),
        },
        "semantics": {
            "all_ten_raw_toy_curves": True,
            "expected_limit_bands": False,
            "limit_bands_created": False,
            "eps2_outputs_promotable": False,
            "qmu_outputs_used_in_postprocess": False,
            "qmu_outputs_promotable": False,
            "qmu_ok_values_relabelled": False,
            "signal_response": (
                "paired within each factor/truth/scenario/background-toy/"
                "mass/replica using factor-15-anchored absolute A as "
                "(A_hat_nonzero-A_hat_zero)/A_injected"
            ),
            "fixed_amplitude_gate": (
                "absolute injection strengths must match the explicit "
                "factor-15 anchor ledger row by row across every candidate "
                "factor"
            ),
            "anchor_normalized_sensitivity": (
                "candidate prefit sigmaA_ref divided by the declared "
                "injection_anchor_sigmaA_ref"
            ),
            "anchor_normalized_fitted_uncertainty": (
                "candidate fitted sigma_A divided by the paired factor-15 "
                "fitted sigma_A"
            ),
            "anchor_normalized_residual": (
                "(A_hat-A_injected) divided by the paired factor-15 sigma_A"
            ),
            "pull_width": (
                "sample standard deviation within each homogeneous mass and "
                "anchor-level stratum; the display is the median of stratum "
                "widths"
            ),
            "scope": (
                "screening diagnostics for choosing a length-scale upper "
                "factor; not expected-limit bands, calibrated coverage, or a "
                "physics reach statement"
            ),
        },
        "outputs": [_output_record(path) for path in outputs],
    }
    _atomic_write_json(manifest_path, manifest)
    outputs.append(manifest_path)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan-csv",
        type=Path,
        default=DEFAULT_SCAN_CSV,
        help="Complete collected scan CSV (partial filenames are refused)",
    )
    parser.add_argument(
        "--injection-csv",
        type=Path,
        default=DEFAULT_INJECTION_CSV,
        help="Complete collected injection CSV (partial filenames are refused)",
    )
    parser.add_argument(
        "--scan-note-artifacts",
        action="store_true",
        help=(
            "Validate the final optimizer-reviewed scan and write only raw-toy "
            "note artifacts; do not read injection data"
        ),
    )
    parser.add_argument(
        "--reviewed-scan-csv",
        type=Path,
        default=DEFAULT_REVIEWED_SCAN_CSV,
        help=(
            "Optimizer-reviewed complete scan CSV used by "
            "--scan-note-artifacts"
        ),
    )
    parser.add_argument(
        "--truth",
        action="append",
        default=[],
        help="Truth-model lane to process; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every gate but do not write summaries or plots",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = _load_spec()
        truths = _selected_truths(spec, args.truth)
        config_hashes = _config_hashes(spec)
        toy_manifest = _validate_toy_manifest(spec)
        if args.scan_note_artifacts:
            all_truths = sorted(map(str, spec["truth_models"]))
            if sorted(truths) != all_truths:
                raise ReviewGateError(
                    "--scan-note-artifacts requires both predeclared truth "
                    "lanes so the primary/alternate factor-20 comparison is "
                    "complete"
                )
            reviewed_scan_path = (
                args.reviewed_scan_csv.expanduser().resolve()
            )
            scan_report, optimizer_audit = (
                _validate_reviewed_scan_bundle(reviewed_scan_path, spec)
            )
            scan_raw = _read_complete_csv(
                reviewed_scan_path, "reviewed scan"
            )
            scan, audit = validate_scan(
                scan_raw,
                spec,
                truths,
                config_hashes,
                allow_reviewed_repairs=True,
            )
            normalization = _validate_source_normalizations(
                spec, toy_manifest
            )
            toy_rows, comparison_summary = (
                build_factor20_toy_median_comparison(
                    scan, spec, normalization
                )
            )
            status = {
                "status": "validated",
                "mode": "scan-note-artifacts",
                "truth_models": truths,
                "reviewed_scan_rows": len(scan),
                "adjacent_lml_comparisons": len(audit),
                "factor20_comparison_scan_rows": (
                    len(truths)
                    * 2
                    * int(spec["n_toys"])
                    * len(_mass_grid_mev(spec))
                ),
                "factor20_toy_medians": len(toy_rows),
                "factor20_summary_rows": len(comparison_summary),
                "source_support_ratio_ten_pct_over_one_pct": (
                    normalization[
                        "source_support_ratio_ten_pct_over_one_pct"
                    ]
                ),
                "source_families_paired": False,
                "expected_limit_bands": False,
                "injection_data_read": False,
            }
            if args.validate_only:
                print(json.dumps(status, indent=2, sort_keys=True))
                return 0
            outputs = _write_scan_note_outputs(
                scan,
                audit,
                spec,
                truths,
                reviewed_scan_path,
                scan_report,
                optimizer_audit,
                toy_manifest,
                normalization,
            )
            status["status"] = "complete"
            status["outputs"] = [str(path) for path in outputs]
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0

        scan_path = args.scan_csv.expanduser().resolve()
        injection_path = args.injection_csv.expanduser().resolve()
        scan_raw = _read_complete_csv(scan_path, "scan")
        injection_raw = _read_complete_csv(injection_path, "injection")
        scan_report = _validate_collection_report(scan_path, "scan")
        reviewed_scan = bool(
            scan_report
            and scan_report.get("review_stage")
            == "optimizer_selected_actual_fit_rows"
        )
        if reviewed_scan:
            scan_report, _ = _validate_reviewed_scan_bundle(
                scan_path, spec
            )
        injection_report = _validate_collection_report(
            injection_path, "injection"
        )
        scan, audit = validate_scan(
            scan_raw,
            spec,
            truths,
            config_hashes,
            allow_reviewed_repairs=reviewed_scan,
        )
        injection = validate_injection(
            injection_raw, spec, truths, config_hashes
        )
        response = build_signal_response_rows(injection)
        signal_strata = build_stratified_signal_summary(
            injection, response, spec
        )
        summary = build_summary(scan, signal_strata, audit)
        expected_summary_rows = (
            len(truths)
            * len(spec["scenarios"])
            * len(spec["length_scale_upper_factors"])
        )
        if len(summary) != expected_summary_rows:
            raise ReviewGateError(
                f"Summary coverage mismatch: expected {expected_summary_rows}, "
                f"got {len(summary)}"
            )
        status = {
            "status": "validated",
            "truth_models": truths,
            "scan_rows": len(scan),
            "injection_rows": len(injection),
            "signal_response_rows": len(response),
            "mass_anchor_signal_strata": len(signal_strata),
            "adjacent_lml_comparisons": len(audit),
            "qmu_ok_false_rows": int(
                (~injection["qmu_ok_parsed"].astype(bool)).sum()
            ),
            "qmu_outputs_used_in_postprocess": False,
            "qmu_outputs_promotable": False,
            "expected_limit_bands": False,
        }
        if args.validate_only:
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0
        outputs = _write_outputs(
            scan,
            injection,
            audit,
            response,
            signal_strata,
            summary,
            spec,
            truths,
            scan_path,
            injection_path,
            {
                "scan": scan_report,
                "injection": injection_report,
            },
            toy_manifest,
        )
        status["status"] = "complete"
        status["outputs"] = [str(path) for path in outputs]
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    except ReviewGateError as exc:
        print(f"REVIEW GATE FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
